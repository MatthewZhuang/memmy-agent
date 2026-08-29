import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import Database from "better-sqlite3";
import {
  MemoryDb,
  MemoryService,
  createEmbedder,
  createLlmClient,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type Embedder,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage,
  type MemmyConfig,
  type ModelStatus
} from "../src/index.js";
import { skillMetaFromMemory } from "../src/algorithm/plugin-algorithms.js";
import type { EvolutionJobRecord } from "../src/storage/repositories.js";
import { Repositories } from "../src/storage/repositories.js";
import { PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION } from
  "../src/storage/procedural-trajectory-repository.js";
import { buildTrajectoryWindows } from
  "../src/service/evolution/procedural-window-model.js";
import {
  PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION,
  PROCEDURAL_SKILL_COVERAGE_PROMPT_VERSION
} from
  "../src/service/evolution/procedural-pattern-skill.js";
import {
  evolutionJobDedupeKey,
  type EnqueueJobInput
} from "../src/service/worker/job-handlers.js";
import { newId, stableHash } from "../src/utils/id.js";

interface ReplayArgs {
  help: boolean;
  sourceDbPath?: string;
  outputDbPath?: string;
  outputReportPath?: string;
  configPath?: string;
  thresholdPath?: string;
  maxRounds: number;
  workerLimit: number;
  confirmSkillLlm: boolean;
  skillOnly: boolean;
}

interface DrainResult {
  drained: boolean;
  rounds: number;
  leased: number;
  succeeded: number;
  failed: number;
  elapsedMs: number;
  remainingByStatus: Record<string, number>;
}

interface CleanupResult {
  removedProceduralSkillIds: string[];
  preservedNonProceduralSkillReferences: number;
  removedJobs: number;
  removedEmbeddingRetries: number;
  removedFamilies: number;
  removedClusters: number;
}

interface GuardStats {
  embeddingCalls: number;
  comparisonQueryEmbeddingCalls: number;
  llmCalls: number;
  llmOperations: string[];
}

interface SkillReplaySeed {
  clusterId: string;
  clusterVersionId: string;
  membershipVersion: number;
  userId: string;
  episodeId: string;
  sessionId: string;
  previousDecision: string;
}

type ProceduralWindowConfig = MemmyConfig["algorithm"]["proceduralWindow"];
type ProceduralScale = ProceduralWindowConfig["scales"][number];
type ThresholdScaleOverride = Partial<Pick<ProceduralScale,
  | "coarseSimilarityThreshold"
  | "minStepSimilarity"
  | "minMatchedSteps"
  | "minCoverage"
  | "minAverageMatchSimilarity"
  | "minAlignmentScore"
>>;

interface ThresholdOverrides {
  minSupportEpisodes?: number;
  scales?: Record<string, ThresholdScaleOverride>;
}

const PROCEDURAL_SKILL_ALGORITHM = "procedural.pattern.skill.v1";
const ALLOWED_SKILL_OPERATIONS = new Set([
  `procedural.${PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION}`,
  `procedural.${PROCEDURAL_SKILL_COVERAGE_PROMPT_VERSION}`
]);
const ALLOWED_JOB_TYPES = new Set([
  "trajectory_window_ingest",
  "procedural_skill_induction"
]);

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(helpText());
    return;
  }
  const sourceDbPath = requiredPath(args.sourceDbPath, "--db");
  const outputDbPath = requiredPath(args.outputDbPath, "--output-db");
  const outputReportPath = resolve(args.outputReportPath ?? `${outputDbPath}.report.json`);
  const configPath = args.configPath ? resolve(args.configPath) : undefined;
  const thresholdPath = args.thresholdPath ? resolve(args.thresholdPath) : undefined;
  assertSafePaths(sourceDbPath, outputDbPath, outputReportPath);
  if (!args.confirmSkillLlm) {
    throw new Error(
      "This replay may send admitted procedural evidence to the configured Skill LLM. " +
      "Re-run with --confirm-skill-llm after reviewing the isolated output paths."
    );
  }

  const loaded = loadMemmyConfig(configPath);
  const config = cloneConfig(loaded.config);
  const overrides = thresholdPath ? readThresholdOverrides(thresholdPath) : {};
  applyThresholdOverrides(config.algorithm.proceduralWindow, overrides);
  // This experiment measures Family/Fine/Skill only. Public Skill embeddings
  // are deliberately disabled so the preserved trajectory vectors are the
  // only vectors that can be touched.
  config.algorithm.capture.embedAfterCapture = false;

  mkdirSync(dirname(outputDbPath), { recursive: true });
  mkdirSync(dirname(outputReportPath), { recursive: true });
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();

  const source = new Database(sourceDbPath, { readonly: true, fileMustExist: true });
  let sourceBaseline: ReturnType<typeof databaseSummary>;
  try {
    source.pragma("query_only = ON");
    sourceBaseline = databaseSummary(source);
    await source.backup(outputDbPath);
  } finally {
    source.close();
  }

  const targetDb = new MemoryDb({ path: outputDbPath });
  let service: MemoryService | undefined;
  try {
    const repos = new Repositories(targetDb.db);
    const preservedBefore = preservedEvidenceSummary(targetDb.db);
    const trajectoryBefore = args.skillOnly
      ? preservedTrajectorySummary(targetDb.db)
      : undefined;
    const skillSeeds = args.skillOnly ? collectSkillReplaySeeds(targetDb.db) : [];
    const cleanup = args.skillOnly
      ? cleanupSkillOnly(targetDb.db, repos)
      : cleanupDerivedState(targetDb.db, repos);

    const baseEmbedder = createEmbedder(config.embedding);
    const guardStats: GuardStats = {
      embeddingCalls: 0,
      comparisonQueryEmbeddingCalls: 0,
      llmCalls: 0,
      llmOperations: []
    };
    const embedder = guardedEmbedder(baseEmbedder, guardStats);
    const embeddingSignature = proceduralEmbeddingSignature(embedder);
    const preflight = preflightPreservedEvidence(
      targetDb.db,
      repos,
      config.algorithm.proceduralWindow,
      embeddingSignature
    );
    const seededJobs = args.skillOnly
      ? seedSkillJobs(repos, skillSeeds)
      : seedWindowJobs(repos, preflight.paths);

    const realSkillLlm = createLlmClient(resolveEvolutionConfig(config), {
      modelRole: "memory_evolution"
    });
    const skillLlm = guardedSkillLlm(realSkillLlm, guardStats);
    service = new MemoryService({
      db: targetDb,
      mode: "local",
      config,
      llm: skillLlm,
      skillLlm,
      embedder
    });

    // Family creation is order-sensitive by definition. Window jobs are
    // drained one at a time in stable seeded order; Skill jobs have lower
    // priority and begin only after all Window jobs are complete.
    const windowDrain = args.skillOnly
      ? emptyDrain()
      : await drainWindowJobsSequentially(service, targetDb.db, args.maxRounds);
    const remainingRounds = Math.max(1, args.maxRounds - windowDrain.rounds);
    const finalDrain = await drainWorker(
      service,
      targetDb.db,
      remainingRounds,
      args.workerLimit
    );
    const drain = combineDrain(windowDrain, finalDrain);

    assertReplayGuards(targetDb.db, guardStats);
    const preservedAfter = preservedEvidenceSummary(targetDb.db);
    assertPreservedEvidenceUnchanged(preservedBefore, preservedAfter);
    const trajectoryAfter = args.skillOnly
      ? preservedTrajectorySummary(targetDb.db)
      : undefined;
    if (trajectoryBefore && trajectoryAfter) {
      assertTrajectoryUnchanged(trajectoryBefore, trajectoryAfter);
    }
    assertForeignKeys(targetDb.db);

    const report = {
      schemaVersion: args.skillOnly
        ? "kw-gap-skill-only-replay.v1"
        : "kw-threshold-only-family-replay.v1",
      startedAt,
      finishedAt: new Date().toISOString(),
      elapsedMs: Date.now() - wallStart,
      isolation: {
        sourceDatabase: sourceDbPath,
        outputDatabase: outputDbPath,
        outputReport: outputReportPath,
        sourceOpenedReadOnly: true,
        snapshotMethod: "better-sqlite3.backup",
        overwriteAllowed: false
      },
      config: {
        configPath: configPath ?? null,
        thresholdOverridePath: thresholdPath ?? null,
        overrides,
        mode: args.skillOnly ? "skill-only" : "family-fine-skill",
        proceduralWindow: config.algorithm.proceduralWindow,
        publicSkillEmbeddingDisabled: true
      },
      models: {
        skill: publicModelStatus(skillLlm),
        preservedEmbedding: publicModelStatus(embedder),
        embeddingSignature
      },
      sourceBaseline,
      cleanup,
      preservation: {
        before: preservedBefore,
        after: preservedAfter,
        ...(trajectoryBefore && trajectoryAfter
          ? { trajectoryBefore, trajectoryAfter }
          : {}),
        preflight: {
          activePaths: preflight.paths.length,
          expectedSteps: preflight.expectedSteps,
          expectedWindows: preflight.expectedWindows,
          embeddingSignature
        }
      },
      executionGuards: guardStats,
      jobs: {
        ...(args.skillOnly
          ? { seededSkillJobs: seededJobs, replaySeeds: skillSeeds }
          : { seededWindowJobs: seededJobs }),
        windowDrain,
        finalDrain,
        combined: drain,
        byTypeAndStatus: jobCounts(targetDb.db)
      },
      output: databaseSummary(targetDb.db),
      skills: proceduralSkills(repos),
      skillDecisions: skillDecisionSummary(targetDb.db)
    };
    writeFileSync(outputReportPath, `${JSON.stringify(report, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600
    });
    process.stdout.write(`${JSON.stringify({
      outputDatabase: outputDbPath,
      outputReport: outputReportPath,
      ...(args.skillOnly
        ? { seededSkillJobs: seededJobs }
        : { seededWindowJobs: seededJobs }),
      families: report.output.families,
      clusters: report.output.clusters,
      proceduralSkills: report.skills.length,
      embeddingCalls: guardStats.embeddingCalls,
      skillLlmCalls: guardStats.llmCalls,
      drain
    }, null, 2)}\n`);
    if (!drain.drained) process.exitCode = 2;
  } finally {
    void service;
    targetDb.close();
  }
}

function cleanupDerivedState(db: Database.Database, repos: Repositories): CleanupResult {
  const skillIds = (db.prepare(
    `SELECT id FROM memories
     WHERE memory_layer = 'Skill'
       AND json_extract(properties_json, '$.internal_info.plugin_algorithm') = ?
     ORDER BY id`
  ).all(PROCEDURAL_SKILL_ALGORITHM) as Array<{ id: string }>).map((row) => row.id);
  const skillSet = new Set(skillIds);
  let preservedNonProceduralSkillReferences = 0;
  const episodes = db.prepare(
    `SELECT id, skill_memory_ids_json AS skillIds FROM episodes ORDER BY id`
  ).all() as Array<{ id: string; skillIds: string }>;
  const counts = {
    removedJobs: scalarCount(db, "evolution_jobs"),
    removedEmbeddingRetries: scalarCount(db, "embedding_retry_queue"),
    removedFamilies: scalarCount(db, "trajectory_window_families"),
    removedClusters: scalarCount(db, "trajectory_window_clusters")
  };

  db.transaction(() => {
    db.prepare(`DELETE FROM embedding_retry_queue`).run();
    db.prepare(`DELETE FROM evolution_jobs`).run();
    for (const episode of episodes) {
      const retained = parseStringArray(episode.skillIds).filter((id) => !skillSet.has(id));
      preservedNonProceduralSkillReferences += retained.length;
      db.prepare(
        `UPDATE episodes SET skill_memory_ids_json = ? WHERE id = ?`
      ).run(JSON.stringify(retained), episode.id);
    }
    // Family links cascade through revisions. Cluster versions, canonical
    // keys, members, and trajectory Skill versions cascade through heads.
    db.prepare(`DELETE FROM trajectory_window_families`).run();
    db.prepare(`DELETE FROM trajectory_window_clusters`).run();
    for (const skillId of skillIds) {
      repos.vectors.deleteMemory(skillId);
      db.prepare(`DELETE FROM skill_trials WHERE skill_memory_id = ?`).run(skillId);
      db.prepare(`DELETE FROM memory_change_log WHERE memory_id = ?`).run(skillId);
      db.prepare(`DELETE FROM memories_fts WHERE id = ?`).run(skillId);
      db.prepare(`DELETE FROM memories WHERE id = ?`).run(skillId);
    }
  })();
  assertForeignKeys(db);
  return {
    removedProceduralSkillIds: skillIds,
    preservedNonProceduralSkillReferences,
    ...counts
  };
}

function collectSkillReplaySeeds(db: Database.Database): SkillReplaySeed[] {
  const rows = db.prepare(
    `WITH ranked AS (
       SELECT clusters.id AS clusterId,
              clusters.active_version_id AS clusterVersionId,
              versions.version_no AS membershipVersion,
              clusters.user_id AS userId,
              skills.payload_json AS payloadJson,
              ROW_NUMBER() OVER (
                PARTITION BY clusters.id
                ORDER BY skills.created_at DESC, skills.id DESC
              ) AS replayRank
       FROM trajectory_window_clusters AS clusters
       JOIN trajectory_window_cluster_versions AS versions
         ON versions.id = clusters.active_version_id
       JOIN trajectory_skill_versions AS skills
         ON skills.cluster_id = clusters.id
        AND skills.cluster_version_id = clusters.active_version_id
       WHERE clusters.status = 'active'
         AND clusters.active_version_id IS NOT NULL
     )
     SELECT clusterId, clusterVersionId, membershipVersion, userId, payloadJson
     FROM ranked
     WHERE replayRank = 1
     ORDER BY clusterId`
  ).all() as Array<{
    clusterId: string;
    clusterVersionId: string;
    membershipVersion: number;
    userId: string;
    payloadJson: string;
  }>;
  const lifecycleReasons = [
    "fine-evidence-no-longer-linked",
    "cluster-no-longer-qualified",
    "not-maximal-qualified",
    "insufficient-common-core",
    "governance-disabled"
  ];
  return rows.flatMap((row) => {
    const rawPayload = safeJson(row.payloadJson);
    const payload = isRecord(rawPayload) ? rawPayload : {};
    const reason = typeof payload.reason === "string" ? payload.reason : "admitted";
    if (lifecycleReasons.includes(reason) || reason.startsWith("maximal-suppressed-by:")) {
      return [];
    }
    const member = db.prepare(
      `SELECT members.episode_id AS episodeId, episodes.session_id AS sessionId
       FROM trajectory_window_cluster_members AS members
       JOIN episodes ON episodes.id = members.episode_id
       WHERE members.cluster_version_id = ?
         AND members.evidence_role = 'support'
       ORDER BY members.is_medoid DESC, members.episode_id ASC
       LIMIT 1`
    ).get(row.clusterVersionId) as { episodeId: string; sessionId: string } | undefined;
    if (!member) throw new Error(`Skill replay cluster has no support Episode: ${row.clusterId}`);
    return [{
      clusterId: row.clusterId,
      clusterVersionId: row.clusterVersionId,
      membershipVersion: row.membershipVersion,
      userId: row.userId,
      episodeId: member.episodeId,
      sessionId: member.sessionId,
      previousDecision: reason
    }];
  });
}

function cleanupSkillOnly(db: Database.Database, repos: Repositories): CleanupResult {
  const skillIds = (db.prepare(
    `SELECT id FROM memories
     WHERE memory_layer = 'Skill'
       AND json_extract(properties_json, '$.internal_info.plugin_algorithm') = ?
     ORDER BY id`
  ).all(PROCEDURAL_SKILL_ALGORITHM) as Array<{ id: string }>).map((row) => row.id);
  const skillSet = new Set(skillIds);
  let preservedNonProceduralSkillReferences = 0;
  const episodes = db.prepare(
    `SELECT id, skill_memory_ids_json AS skillIds FROM episodes ORDER BY id`
  ).all() as Array<{ id: string; skillIds: string }>;
  const removedJobs = scalarCount(db, "evolution_jobs");
  const removedEmbeddingRetries = scalarCount(db, "embedding_retry_queue");

  db.transaction(() => {
    db.prepare(`DELETE FROM embedding_retry_queue`).run();
    db.prepare(`DELETE FROM evolution_jobs`).run();
    for (const episode of episodes) {
      const retained = parseStringArray(episode.skillIds).filter((id) => !skillSet.has(id));
      preservedNonProceduralSkillReferences += retained.length;
      db.prepare(`UPDATE episodes SET skill_memory_ids_json = ? WHERE id = ?`)
        .run(JSON.stringify(retained), episode.id);
    }
    db.prepare(
      `UPDATE trajectory_window_clusters
       SET active_skill_version_id = NULL,
           active_skill_memory_id = NULL`
    ).run();
    db.prepare(`DELETE FROM trajectory_skill_versions`).run();
    for (const skillId of skillIds) {
      repos.vectors.deleteMemory(skillId);
      db.prepare(`DELETE FROM skill_trials WHERE skill_memory_id = ?`).run(skillId);
      db.prepare(`DELETE FROM memory_change_log WHERE memory_id = ?`).run(skillId);
      db.prepare(`DELETE FROM memories_fts WHERE id = ?`).run(skillId);
      db.prepare(`DELETE FROM memories WHERE id = ?`).run(skillId);
    }
  })();
  assertForeignKeys(db);
  return {
    removedProceduralSkillIds: skillIds,
    preservedNonProceduralSkillReferences,
    removedJobs,
    removedEmbeddingRetries,
    removedFamilies: 0,
    removedClusters: 0
  };
}

function preflightPreservedEvidence(
  db: Database.Database,
  repos: Repositories,
  procedural: ProceduralWindowConfig,
  embeddingSignature: string
): {
  paths: EvolutionPathSeed[];
  expectedSteps: number;
  expectedWindows: number;
} {
  const storedSignatures = distinct((db.prepare(
    `SELECT embedding_signature AS signature FROM procedural_step_embeddings
     UNION SELECT embedding_signature AS signature FROM trajectory_window_occurrences`
  ).all() as Array<{ signature: string }>).map((row) => row.signature));
  if (storedSignatures.length !== 1 || storedSignatures[0] !== embeddingSignature) {
    throw new Error(
      `Preserved embedding signature mismatch: stored=${storedSignatures.join(",") || "none"}, ` +
      `configured=${embeddingSignature}. Use the same embedding config that created the source DB.`
    );
  }
  const rows = db.prepare(
    `SELECT paths.id, paths.episode_id AS episodeId, paths.user_id AS userId,
            paths.path_hash AS pathHash, paths.source_snapshot_hash AS sourceSnapshotHash,
            episodes.session_id AS sessionId, episodes.r_task AS rTask,
            episodes.reward_detail_json AS rewardDetail
     FROM episode_execution_paths AS paths
     JOIN episodes ON episodes.id = paths.episode_id
     WHERE paths.status = 'active'
     ORDER BY paths.episode_id ASC, paths.id ASC`
  ).all() as EvolutionPathSeed[];
  if (rows.length === 0) throw new Error("The source snapshot contains no active execution Paths");
  let expectedSteps = 0;
  let expectedWindows = 0;
  const specs = procedural.scales.map((scale) => ({
    length: scale.length,
    stride: scale.stride
  }));
  const proceduralScales = new Set(specs.map((spec) => spec.length));
  for (const row of rows) {
    const path = repos.proceduralTrajectory.getPath(row.id);
    if (!path || path.status !== "active") throw new Error(`Active Path not readable: ${row.id}`);
    const stepEmbeddings = repos.proceduralTrajectory.listStepEmbeddings({
      pathId: path.id,
      representationVersion: PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION,
      embeddingSignature
    });
    if (stepEmbeddings.length !== path.path.steps.length ||
        stepEmbeddings.some((item, index) => item.stepId !== path.path.steps[index]?.id)) {
      throw new Error(`Path ${path.id} is missing canonical Step embeddings`);
    }
    expectedSteps += path.path.steps.length;
    const episode = repos.runtime.getEpisode(path.episodeId);
    if (!episode || episode.status !== "closed") {
      throw new Error(`Active Path ${path.id} does not belong to a closed Episode`);
    }
    const expected = buildTrajectoryWindows([{
      ...path.path,
      ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask })
    }], specs);
    // A three-route source DB may also contain V3 Span-15 Windows. They are
    // preserved by the backup but are outside this V2 5/10 replay basis.
    const stored = repos.proceduralTrajectory.listWindowsForPath(path.id)
      .filter((window) => proceduralScales.has(window.scale));
    const storedById = new Map(stored.map((window) => [window.id, window]));
    for (const occurrence of expected) {
      const existing = storedById.get(occurrence.id);
      if (!existing || existing.embeddingSignature !== embeddingSignature ||
          existing.scale !== occurrence.scale || existing.stride !== occurrence.stride ||
          existing.semanticText !== occurrence.semanticText) {
        throw new Error(`Path ${path.id} is missing reusable Window ${occurrence.id}`);
      }
    }
    if (stored.length !== expected.length) {
      throw new Error(
        `Path ${path.id} Window basis differs from current 5/10 projection: ` +
        `stored=${stored.length}, expected=${expected.length}`
      );
    }
    expectedWindows += expected.length;
  }
  return { paths: rows, expectedSteps, expectedWindows };
}

interface EvolutionPathSeed {
  id: string;
  episodeId: string;
  userId: string;
  pathHash: string;
  sourceSnapshotHash: string;
  sessionId: string;
  rTask: number | null;
  rewardDetail: string;
}

function seedWindowJobs(repos: Repositories, paths: readonly EvolutionPathSeed[]): number {
  const base = Date.parse("2026-08-27T12:00:00.000Z");
  paths.forEach((path, index) => {
    enqueueJob(repos, {
      jobType: "trajectory_window_ingest",
      userId: path.userId,
      sessionId: path.sessionId,
      episodeId: path.episodeId,
      payload: {
        reason: "threshold-only.family-replay",
        pathId: path.id,
        pathHash: path.pathHash,
        pathSourceSnapshotHash: path.sourceSnapshotHash,
        sourceSnapshotHash: path.sourceSnapshotHash,
        rewardSnapshotHash: stableHash({
          rTask: path.rTask,
          rewardDetail: safeJson(path.rewardDetail)
        })
      },
      createdAt: new Date(base + index).toISOString()
    });
  });
  return paths.length;
}

function seedSkillJobs(repos: Repositories, seeds: readonly SkillReplaySeed[]): number {
  const base = Date.parse("2026-08-27T14:00:00.000Z");
  seeds.forEach((seed, index) => {
    enqueueJob(repos, {
      jobType: "procedural_skill_induction",
      userId: seed.userId,
      sessionId: seed.sessionId,
      episodeId: seed.episodeId,
      payload: {
        reason: "gap-input.skill-only-replay",
        clusterId: seed.clusterId,
        clusterVersionId: seed.clusterVersionId,
        membershipVersion: seed.membershipVersion,
        inductionVersion: PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION
      },
      createdAt: new Date(base + index).toISOString()
    });
  });
  return seeds.length;
}

function emptyDrain(): DrainResult {
  return {
    drained: true,
    rounds: 0,
    leased: 0,
    succeeded: 0,
    failed: 0,
    elapsedMs: 0,
    remainingByStatus: {}
  };
}

function enqueueJob(repos: Repositories, input: EnqueueJobInput): EvolutionJobRecord {
  const at = input.createdAt ?? new Date().toISOString();
  const dedupeKey = evolutionJobDedupeKey(input);
  return repos.runtime.enqueueJob({
    id: newId("job"),
    jobType: input.jobType,
    status: "queued",
    ...(dedupeKey ? { dedupeKey } : {}),
    userId: input.userId,
    ...(input.sessionId ? { sessionId: input.sessionId } : {}),
    ...(input.episodeId ? { episodeId: input.episodeId } : {}),
    ...(input.targetMemoryId ? { targetMemoryId: input.targetMemoryId } : {}),
    payload: input.payload ?? {},
    attempts: 0,
    maxAttempts: input.maxAttempts ?? 3,
    leasedUntil: null,
    lastError: null,
    createdAt: at,
    updatedAt: at
  });
}

function guardedEmbedder(base: Embedder, stats: GuardStats): Embedder {
  const reject = (operation: string): never => {
    stats.embeddingCalls += 1;
    throw new Error(
      `Threshold-only replay invariant violated: ${operation} attempted a new embedding`
    );
  };
  return {
    config: base.config,
    isRemote: () => base.isRemote(),
    status: () => base.status(),
    embed: async () => reject("embed"),
    embedOne: async (text, mode) => {
      if (mode !== "query") return reject(`embedOne:${mode}`);
      stats.comparisonQueryEmbeddingCalls += 1;
      return base.embedOne(text, mode);
    }
  };
}

function guardedSkillLlm(base: LlmClient, stats: GuardStats): LlmClient {
  const record = (options: LlmCompletionOptions): void => {
    stats.llmCalls += 1;
    stats.llmOperations.push(options.operation);
    if (!ALLOWED_SKILL_OPERATIONS.has(options.operation)) {
      throw new Error(
        `Threshold-only replay blocked non-Skill LLM operation: ${options.operation}`
      );
    }
  };
  return {
    config: base.config,
    isConfigured: () => base.isConfigured(),
    status: () => base.status(),
    complete: async (messages: LlmMessage[], options: LlmCompletionOptions) => {
      record(options);
      return base.complete(messages, options);
    },
    completeJson: async <T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ) => {
      record(options);
      return base.completeJson<T>(messages, options);
    }
  };
}

async function drainWindowJobsSequentially(
  service: MemoryService,
  db: Database.Database,
  maxRounds: number
): Promise<DrainResult> {
  const startedAt = Date.now();
  let leased = 0;
  let succeeded = 0;
  let failed = 0;
  let rounds = 0;
  while (pendingJobCount(db, "trajectory_window_ingest") > 0 && rounds < maxRounds) {
    const result = await service.runWorkerOnce(1);
    rounds += 1;
    leased += result.leased + result.embeddingRetries.leased;
    succeeded += result.succeeded + result.embeddingRetries.succeeded;
    failed += result.failed + result.embeddingRetries.failed;
    if (result.leased === 0 && result.embeddingRetries.leased === 0) break;
  }
  const remainingByStatus = unfinishedJobCounts(db);
  return {
    drained: pendingJobCount(db, "trajectory_window_ingest") === 0,
    rounds,
    leased,
    succeeded,
    failed,
    elapsedMs: Date.now() - startedAt,
    remainingByStatus
  };
}

async function drainWorker(
  service: MemoryService,
  db: Database.Database,
  maxRounds: number,
  workerLimit: number
): Promise<DrainResult> {
  const startedAt = Date.now();
  let leased = 0;
  let succeeded = 0;
  let failed = 0;
  for (let round = 1; round <= maxRounds; round += 1) {
    const result = await service.runWorkerOnce(workerLimit);
    leased += result.leased + result.embeddingRetries.leased;
    succeeded += result.succeeded + result.embeddingRetries.succeeded;
    failed += result.failed + result.embeddingRetries.failed;
    if (result.leased === 0 && result.embeddingRetries.leased === 0) {
      const remainingByStatus = unfinishedJobCounts(db);
      return {
        drained: Object.values(remainingByStatus).every((count) => count === 0),
        rounds: round,
        leased,
        succeeded,
        failed,
        elapsedMs: Date.now() - startedAt,
        remainingByStatus
      };
    }
  }
  return {
    drained: false,
    rounds: maxRounds,
    leased,
    succeeded,
    failed,
    elapsedMs: Date.now() - startedAt,
    remainingByStatus: unfinishedJobCounts(db)
  };
}

function combineDrain(first: DrainResult, second: DrainResult): DrainResult {
  return {
    drained: first.drained && second.drained,
    rounds: first.rounds + second.rounds,
    leased: first.leased + second.leased,
    succeeded: first.succeeded + second.succeeded,
    failed: first.failed + second.failed,
    elapsedMs: first.elapsedMs + second.elapsedMs,
    remainingByStatus: second.remainingByStatus
  };
}

function assertReplayGuards(db: Database.Database, stats: GuardStats): void {
  if (stats.embeddingCalls !== 0) {
    throw new Error(`Replay attempted ${stats.embeddingCalls} forbidden embedding calls`);
  }
  const unexpected = (db.prepare(
    `SELECT DISTINCT job_type AS jobType FROM evolution_jobs ORDER BY job_type`
  ).all() as Array<{ jobType: string }>).map((row) => row.jobType)
    .filter((jobType) => !ALLOWED_JOB_TYPES.has(jobType));
  if (unexpected.length > 0) {
    throw new Error(`Replay created forbidden job types: ${unexpected.join(", ")}`);
  }
  if (stats.llmOperations.some((operation) => !ALLOWED_SKILL_OPERATIONS.has(operation))) {
    throw new Error("Replay invoked a non-Skill LLM operation");
  }
}

function databaseSummary(db: Database.Database) {
  return {
    activePaths: countWhere(db, "episode_execution_paths", "status = 'active'"),
    stepEmbeddings: scalarCount(db, "procedural_step_embeddings"),
    windows: scalarCount(db, "trajectory_window_occurrences"),
    families: groupedCount(db, "trajectory_window_families", "status"),
    familyRevisions: groupedCount(db, "trajectory_window_family_revisions", "status"),
    familyMembers: scalarCount(db, "trajectory_window_family_members"),
    clusters: groupedCount(db, "trajectory_window_clusters", "status"),
    clusterVersions: groupedCount(db, "trajectory_window_cluster_versions", "status"),
    clusterMembers: scalarCount(db, "trajectory_window_cluster_members"),
    canonicalKeys: scalarCount(db, "trajectory_window_cluster_canonical_keys"),
    familyClusterLinks: scalarCount(db, "trajectory_window_family_cluster_links"),
    skillVersions: groupedCount(db, "trajectory_skill_versions", "status"),
    proceduralSkills: countWhere(
      db,
      "memories",
      `memory_layer = 'Skill' AND ` +
      `json_extract(properties_json, '$.internal_info.plugin_algorithm') = ` +
      `'${PROCEDURAL_SKILL_ALGORITHM}'`
    )
  };
}

function preservedEvidenceSummary(db: Database.Database) {
  const pathIds = (db.prepare(
    `SELECT id FROM episode_execution_paths WHERE status = 'active' ORDER BY id`
  ).all() as Array<{ id: string }>).map((row) => row.id);
  const stepIds = (db.prepare(
    `SELECT id FROM procedural_step_embeddings ORDER BY id`
  ).all() as Array<{ id: string }>).map((row) => row.id);
  const windowIds = (db.prepare(
    `SELECT id FROM trajectory_window_occurrences ORDER BY id`
  ).all() as Array<{ id: string }>).map((row) => row.id);
  return {
    activePaths: pathIds.length,
    stepEmbeddings: stepIds.length,
    windows: windowIds.length,
    activePathIdHash: stableHash(pathIds),
    stepEmbeddingIdHash: stableHash(stepIds),
    windowIdHash: stableHash(windowIds)
  };
}

function preservedTrajectorySummary(db: Database.Database) {
  const query = (sql: string): unknown[] => db.prepare(sql).all() as unknown[];
  const basis = {
    families: query(
      `SELECT id, user_id, scale, algorithm_version, config_hash, status,
              active_revision_id
       FROM trajectory_window_families ORDER BY id`
    ),
    familyRevisions: query(
      `SELECT id, family_id, revision_no, membership_hash, evidence_hash,
              medoid_occurrence_id, status
       FROM trajectory_window_family_revisions ORDER BY id`
    ),
    familyMembers: query(
      `SELECT family_revision_id, occurrence_id, episode_id, user_id,
              coarse_similarity, is_medoid
       FROM trajectory_window_family_members
       ORDER BY family_revision_id, occurrence_id`
    ),
    clusters: query(
      `SELECT id, user_id, scale, algorithm_version, config_hash, status,
              active_version_id
       FROM trajectory_window_clusters ORDER BY id`
    ),
    clusterVersions: query(
      `SELECT id, cluster_id, version_no, membership_hash, support_hash,
              medoid_occurrence_id, support_episode_count,
              counterexample_episode_count, unknown_episode_count, metrics_json, status
       FROM trajectory_window_cluster_versions ORDER BY id`
    ),
    clusterMembers: query(
      `SELECT cluster_version_id, occurrence_id, episode_id, evidence_role,
              reward_hash, coarse_similarity, alignment_json, is_medoid
       FROM trajectory_window_cluster_members
       ORDER BY cluster_version_id, occurrence_id`
    ),
    familyClusterLinks: query(
      `SELECT family_revision_id, canonical_key_id, cluster_version_id
       FROM trajectory_window_family_cluster_links
       ORDER BY family_revision_id, canonical_key_id, cluster_version_id`
    )
  };
  return {
    families: basis.families.length,
    familyRevisions: basis.familyRevisions.length,
    familyMembers: basis.familyMembers.length,
    clusters: basis.clusters.length,
    clusterVersions: basis.clusterVersions.length,
    clusterMembers: basis.clusterMembers.length,
    familyClusterLinks: basis.familyClusterLinks.length,
    basisHash: stableHash(basis)
  };
}

function assertTrajectoryUnchanged(
  before: ReturnType<typeof preservedTrajectorySummary>,
  after: ReturnType<typeof preservedTrajectorySummary>
): void {
  for (const key of Object.keys(before) as Array<keyof typeof before>) {
    if (before[key] !== after[key]) {
      throw new Error(`Skill-only replay changed upstream trajectory evidence: ${key}`);
    }
  }
}

function assertPreservedEvidenceUnchanged(
  before: ReturnType<typeof preservedEvidenceSummary>,
  after: ReturnType<typeof preservedEvidenceSummary>
): void {
  for (const key of Object.keys(before) as Array<keyof typeof before>) {
    if (before[key] !== after[key]) {
      throw new Error(`Preserved evidence changed during threshold-only replay: ${key}`);
    }
  }
}

function proceduralSkills(repos: Repositories) {
  return repos.memories.list({
    memoryLayer: "Skill",
    status: ["activated", "resolving", "archived", "deleted"]
  }, 10_000).filter((memory) =>
    memory.properties.internal_info.plugin_algorithm === PROCEDURAL_SKILL_ALGORITHM
  ).map((memory) => {
    const skill = skillMetaFromMemory(memory);
    const internal = memory.properties.internal_info;
    return {
      id: memory.id,
      memoryKey: memory.memoryKey,
      memoryStatus: memory.status,
      name: skill?.name ?? memory.memoryKey ?? memory.id,
      lifecycleStatus: skill?.status ?? "unknown",
      support: skill?.support ?? 0,
      sourceEpisodeIds: stringArrayValue(internal.source_episode_ids),
      sourceSpanOccurrenceIds: stringArrayValue(internal.source_span_occurrence_ids)
    };
  });
}

function skillDecisionSummary(db: Database.Database): Record<string, number> {
  const rows = db.prepare(
    `SELECT CASE
              WHEN json_extract(versions.payload_json, '$.admitted') = 1 THEN 'admitted'
              ELSE COALESCE(json_extract(versions.payload_json, '$.reason'), 'rejected')
            END AS decision,
            COUNT(*) AS count
     FROM trajectory_window_clusters AS clusters
     JOIN trajectory_skill_versions AS versions
       ON versions.id = clusters.active_skill_version_id
     WHERE clusters.status = 'active'
     GROUP BY decision ORDER BY decision`
  ).all() as Array<{ decision: string; count: number }>;
  return Object.fromEntries(rows.map((row) => [row.decision, Number(row.count)]));
}

function readThresholdOverrides(path: string): ThresholdOverrides {
  if (!existsSync(path)) throw new Error(`Threshold override file not found: ${path}`);
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  if (!isRecord(parsed)) throw new Error("Threshold override must be a JSON object");
  const allowedTop = new Set(["minSupportEpisodes", "scales"]);
  rejectUnknownKeys(parsed, allowedTop, "threshold override");
  const result: ThresholdOverrides = {};
  if (parsed.minSupportEpisodes !== undefined) {
    result.minSupportEpisodes = requiredInteger(parsed.minSupportEpisodes, 2, 1_000,
      "minSupportEpisodes");
  }
  if (parsed.scales !== undefined) {
    if (!isRecord(parsed.scales)) throw new Error("scales must be an object keyed by 5 or 10");
    result.scales = {};
    for (const [scale, raw] of Object.entries(parsed.scales)) {
      if (!isRecord(raw)) throw new Error(`scales.${scale} must be an object`);
      const allowed = new Set([
        "coarseSimilarityThreshold",
        "minStepSimilarity",
        "minMatchedSteps",
        "minCoverage",
        "minAverageMatchSimilarity",
        "minAlignmentScore"
      ]);
      rejectUnknownKeys(raw, allowed, `scales.${scale}`);
      const item: ThresholdScaleOverride = {};
      for (const field of allowed) {
        const value = raw[field];
        if (value === undefined) continue;
        if (field === "minMatchedSteps") {
          item.minMatchedSteps = requiredInteger(value, 1, Number(scale), `${scale}.${field}`);
        } else {
          item[field as Exclude<keyof ThresholdScaleOverride, "minMatchedSteps">] =
            requiredUnitInterval(value, `${scale}.${field}`);
        }
      }
      result.scales[scale] = item;
    }
  }
  return result;
}

function applyThresholdOverrides(
  procedural: ProceduralWindowConfig,
  overrides: ThresholdOverrides
): void {
  if (overrides.minSupportEpisodes !== undefined) {
    procedural.minSupportEpisodes = overrides.minSupportEpisodes;
  }
  const scalesByLength = new Map(procedural.scales.map((scale) => [String(scale.length), scale]));
  for (const [length, override] of Object.entries(overrides.scales ?? {})) {
    const scale = scalesByLength.get(length);
    if (!scale) throw new Error(`Threshold override references unavailable scale ${length}`);
    Object.assign(scale, override);
  }
}

function cloneConfig(config: MemmyConfig): MemmyConfig {
  return JSON.parse(JSON.stringify(config)) as MemmyConfig;
}

function proceduralEmbeddingSignature(embedder: Embedder): string {
  const status = embedder.status();
  return `embedding:${stableHash({
    provider: status.provider,
    model: status.model ?? embedder.config.model ?? "unknown",
    mode: embedder.config.mode,
    sourceProvider: embedder.config.sourceProvider ?? null,
    normalize: embedder.config.normalize
  }).slice(0, 24)}`;
}

function publicModelStatus(model: { status(): ModelStatus }): ModelStatus {
  const status = model.status();
  return {
    provider: status.provider,
    ...(status.model ? { model: status.model } : {}),
    configured: status.configured,
    remote: status.remote
  };
}

function parseArgs(argv: string[]): ReplayArgs {
  const args: ReplayArgs = {
    help: false,
    maxRounds: 2_000,
    workerLimit: 4,
    confirmSkillLlm: false,
    skillOnly: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    const next = (): string => {
      const value = argv[index + 1];
      if (!value) throw new Error(`Missing value for ${arg}`);
      index += 1;
      return value;
    };
    switch (arg) {
      case "--help": case "-h": args.help = true; break;
      case "--db": args.sourceDbPath = next(); break;
      case "--output-db": args.outputDbPath = next(); break;
      case "--output": args.outputReportPath = next(); break;
      case "--config": args.configPath = next(); break;
      case "--thresholds": args.thresholdPath = next(); break;
      case "--max-rounds": args.maxRounds = requiredInteger(next(), 1, 100_000, arg); break;
      case "--worker-limit": args.workerLimit = requiredInteger(next(), 1, 100, arg); break;
      case "--confirm-skill-llm": args.confirmSkillLlm = true; break;
      case "--skill-only": args.skillOnly = true; break;
      default: throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function helpText(): string {
  return `Usage:
  node --import tsx scripts/kw-threshold-only-family-replay.ts \\
    --db <source.sqlite> \\
    --output-db <isolated.sqlite> \\
    [--output <report.json>] [--config <memmy.yaml>] \\
    [--thresholds <overrides.json>] [--worker-limit 4] [--max-rounds 2000] \\
    [--skill-only] --confirm-skill-llm

Copies the source with SQLite backup, preserves active Paths, canonical Step
embeddings, and mechanical 5/10 Windows, then rebuilds only v2 Families, Fine
Clusters, and direct procedural Skills. It rejects overwrites, all Step-semantics
LLM calls, every embedding call, and every unrelated job type.

With --skill-only, preserves all existing Path/Step/Window/Family/Fine Cluster
state and reruns only the current stable procedural Skill decisions. This mode
compares Skill input/prompt versions without reclustering.
`;
}

function assertSafePaths(source: string, output: string, report: string): void {
  if (!existsSync(source)) throw new Error(`Source database not found: ${source}`);
  if (source === output) throw new Error("--output-db must differ from --db");
  if (output === report || source === report) throw new Error("Report path must be distinct");
  if (existsSync(output)) throw new Error(`Refusing to overwrite output DB: ${output}`);
  if (existsSync(report)) throw new Error(`Refusing to overwrite output report: ${report}`);
}

function requiredPath(value: string | undefined, flag: string): string {
  if (!value) throw new Error(`Missing required argument ${flag}`);
  return resolve(value);
}

function scalarCount(db: Database.Database, table: string): number {
  assertIdentifier(table);
  return Number((db.prepare(`SELECT COUNT(*) AS count FROM "${table}"`).get() as
    { count: number }).count);
}

function countWhere(db: Database.Database, table: string, where: string): number {
  assertIdentifier(table);
  return Number((db.prepare(
    `SELECT COUNT(*) AS count FROM "${table}" WHERE ${where}`
  ).get() as { count: number }).count);
}

function groupedCount(
  db: Database.Database,
  table: string,
  column: string
): Record<string, number> {
  assertIdentifier(table);
  assertIdentifier(column);
  const rows = db.prepare(
    `SELECT "${column}" AS value, COUNT(*) AS count
     FROM "${table}" GROUP BY "${column}" ORDER BY "${column}"`
  ).all() as Array<{ value: string; count: number }>;
  return Object.fromEntries(rows.map((row) => [row.value, Number(row.count)]));
}

function pendingJobCount(db: Database.Database, jobType: string): number {
  return Number((db.prepare(
    `SELECT COUNT(*) AS count FROM evolution_jobs
     WHERE job_type = ? AND status IN ('queued', 'leased', 'failed')`
  ).get(jobType) as { count: number }).count);
}

function unfinishedJobCounts(db: Database.Database): Record<string, number> {
  const rows = db.prepare(
    `SELECT status, COUNT(*) AS count FROM evolution_jobs
     WHERE status IN ('queued', 'leased', 'failed', 'dead_letter') GROUP BY status`
  ).all() as Array<{ status: string; count: number }>;
  return Object.fromEntries(rows.map((row) => [row.status, Number(row.count)]));
}

function jobCounts(db: Database.Database): Record<string, Record<string, number>> {
  const rows = db.prepare(
    `SELECT job_type AS jobType, status, COUNT(*) AS count
     FROM evolution_jobs GROUP BY job_type, status ORDER BY job_type, status`
  ).all() as Array<{ jobType: string; status: string; count: number }>;
  const result: Record<string, Record<string, number>> = {};
  for (const row of rows) {
    result[row.jobType] ??= {};
    result[row.jobType]![row.status] = Number(row.count);
  }
  return result;
}

function assertForeignKeys(db: Database.Database): void {
  const rows = db.prepare(`PRAGMA foreign_key_check`).all();
  if (rows.length > 0) {
    throw new Error(`Foreign-key check failed after isolated cleanup: ${JSON.stringify(rows)}`);
  }
}

function parseStringArray(value: string): string[] {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function safeJson(value: string): unknown {
  try { return JSON.parse(value); } catch { return value; }
}

function distinct(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function rejectUnknownKeys(
  value: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  label: string
): void {
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) throw new Error(`${label} has unsupported keys: ${unknown.join(", ")}`);
}

function requiredInteger(value: unknown, min: number, max: number, label: string): number {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (!Number.isInteger(parsed) || Number(parsed) < min || Number(parsed) > max) {
    throw new Error(`${label} must be an integer in [${min}, ${max}]`);
  }
  return Number(parsed);
}

function requiredUnitInterval(value: unknown, label: string): number {
  const parsed = typeof value === "string" ? Number(value) : value;
  if (typeof parsed !== "number" || !Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new Error(`${label} must be in (0, 1]`);
  }
  return parsed;
}

function assertIdentifier(value: string): void {
  if (!/^[a-z_][a-z0-9_]*$/i.test(value)) throw new Error(`Unsafe SQL identifier: ${value}`);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
