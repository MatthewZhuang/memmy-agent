import { existsSync, mkdirSync, writeFileSync } from "node:fs";
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
  type MemoryRow
} from "../src/index.js";
import { traceMetaFromMemory } from "../src/algorithm/plugin-algorithms.js";
import {
  buildEpisodeTrajectoryFamily,
  mineLongCommonSpanSequences,
  projectEpisodeToReferenceSpans,
  selectMaximalLongCommonTrajectories,
  trajectoryIntentSequenceText,
  type EpisodeTrajectoryDocumentV1,
  type EpisodeTrajectoryFamilyV1,
  type EpisodeSpanSequenceProjectionV1,
  type LongCommonTrajectoryV1,
  type LongTrajectoryMiningConfigV1
} from "../src/service/evolution/long-trajectory-model.js";
import {
  ProceduralPatternSkillMaterializer,
  type ProceduralPatternSkillInput
} from "../src/service/evolution/procedural-pattern-skill.js";
import { buildLongTrajectorySkillInput } from
  "../src/service/evolution/long-trajectory-skill-input.js";
import {
  buildTrajectoryWindows,
  unitVector,
  type EmbeddedTrajectoryWindowV1,
  type EpisodeExecutionPathLiteV1,
  type ExecutionStepLiteV1
} from "../src/service/evolution/procedural-window-model.js";
import { Repositories } from "../src/storage/repositories.js";
import type { EvolutionJobRecord } from "../src/storage/repositories.js";
import {
  evolutionJobDedupeKey,
  type EnqueueJobInput
} from "../src/service/worker/job-handlers.js";
import { PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION } from
  "../src/storage/procedural-trajectory-repository.js";
import { newId, stableHash } from "../src/utils/id.js";

interface ReplayArgs {
  help: boolean;
  sourceDbPath?: string;
  outputDbPath?: string;
  outputReportPath?: string;
  configPath?: string;
  confirmRemoteModels: boolean;
}

interface ReplayStats {
  embeddingCalls: number;
  embeddedTexts: number;
  skillLlmCalls: number;
}

const WINDOW_SPECS = [
  { length: 5, stride: 2 },
  { length: 10, stride: 5 },
  { length: 15, stride: 7 }
] as const;

const MINING_CONFIG: LongTrajectoryMiningConfigV1 = {
  episodeRecallLimit: 10,
  minEpisodeSimilarity: 0.65,
  minGoalSimilarity: 0.55,
  goalWeight: 0.4,
  trajectoryWeight: 0.6,
  windowTopK: 3,
  coarseThresholds: { 5: 0.76, 10: 0.7, 15: 0.68 },
  minSupportEpisodes: 2,
  minSpanSequenceLength: 2,
  // The V3 channel must combine at least two non-overlapping reference Spans
  // and cover more than V2's single Span-10 capability.
  minTrajectorySpanSteps: 12,
  minEpisodeCoverage: 0.5
};

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(helpText());
    return;
  }
  const sourceDbPath = requiredPath(args.sourceDbPath, "--db");
  const outputDbPath = requiredPath(args.outputDbPath, "--output-db");
  const outputReportPath = resolve(args.outputReportPath ?? `${outputDbPath}.report.json`);
  if (!args.confirmRemoteModels) {
    throw new Error(
      "This replay embeds KW Episode/Window semantics and sends qualified long-trajectory " +
      "evidence to the configured Skill LLM. Re-run with --confirm-remote-models."
    );
  }
  assertSafePaths(sourceDbPath, outputDbPath, outputReportPath);
  mkdirSync(dirname(outputDbPath), { recursive: true });
  mkdirSync(dirname(outputReportPath), { recursive: true });
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();
  const source = new Database(sourceDbPath, { readonly: true, fileMustExist: true });
  try {
    source.pragma("query_only = ON");
    await source.backup(outputDbPath);
  } finally {
    source.close();
  }

  const configPath = args.configPath ? resolve(args.configPath) : undefined;
  const config = loadMemmyConfig(configPath).config;
  const target = new MemoryDb({ path: outputDbPath });
  let service: MemoryService | undefined;
  try {
    const repos = new Repositories(target.db);
    removePriorLongTrajectorySkills(target.db, repos);
    const paths = prepareCandidateFirstReplay(
      target.db,
      repos,
      config.algorithm.skill.outcomeRTaskSuccessThreshold
    );
    service = new MemoryService({
      db: target,
      mode: "local",
      config,
      ...(configPath ? { configPath } : {})
    });
    const episodeRuns: Array<Record<string, unknown>> = [];
    for (const [index, path] of paths.entries()) {
      activateReplayPath(target.db, path.pathId, new Date().toISOString());
      const before = candidateHeadSnapshot(target.db, repos);
      const miningJob = enqueueReplayJob(repos, {
        jobType: "long_trajectory_mining",
        userId: path.userId,
        sessionId: path.sessionId,
        episodeId: path.episodeId,
        targetMemoryId: path.pathId,
        payload: {
          reason: "kw.long-trajectory-v3.candidate-first",
          pathId: path.pathId,
          pathHash: path.pathHash,
          rewardSnapshotHash: stableHash({
            version: "kw-v3-candidate-first-reward.v1",
            episodeId: path.episodeId,
            rTask: path.rTask
          })
        }
      });
      const result = await service.runWorkerOnce(1);
      const storedMiningJob = repos.runtime.getJob(miningJob.id);
      if (storedMiningJob?.status !== "succeeded" || result.failed > 0) {
        throw new Error(
          `Candidate-first mining failed for ${path.episodeId}: ` +
          `${storedMiningJob?.lastError ?? storedMiningJob?.status ?? "missing-job"}`
        );
      }
      const afterMining = candidateHeadSnapshot(target.db, repos);
      const episodeDrain = await drainWorker(service, target.db, 1_000, 1);
      if (!episodeDrain.drained) {
        throw new Error(
          `Episode worker chain did not drain for ${path.episodeId}: ` +
          `${JSON.stringify(episodeDrain)}`
        );
      }
      const after = candidateHeadSnapshot(target.db, repos);
      episodeRuns.push({
        sequenceNo: index + 1,
        episodeId: path.episodeId,
        pathId: path.pathId,
        beforeCandidates: before.length,
        afterMiningCandidates: afterMining.length,
        afterCandidates: after.length,
        changedCandidates: changedCandidateHeads(before, after),
        workerDrain: episodeDrain
      });
      process.stdout.write(
        `[${index + 1}/${paths.length}] ${path.episodeId}: ` +
        `${before.length} -> ${after.length} candidates, ` +
        `${after.filter((item) => item.activeSkillMemoryId).length} linked skills\n`
      );
    }
    const queuedBeforeFinalDrain = unfinishedJobCounts(target.db);
    const drain = await drainWorker(service, target.db, 1_000, 1);
    if (!drain.drained) {
      throw new Error(`Candidate-first replay did not drain: ${JSON.stringify(drain)}`);
    }
    const candidates = candidateReport(target.db, repos);
    const liveSkills = candidates.filter((candidate) => candidate.skill?.live === true);
    const report = {
      schemaVersion: "kw-long-trajectory-v3-candidate-first-replay.v2",
      startedAt,
      finishedAt: new Date().toISOString(),
      elapsedMs: Date.now() - wallStart,
      isolation: {
        sourceDatabase: sourceDbPath,
        outputDatabase: outputDbPath,
        outputReport: outputReportPath,
        sourceOpenedReadOnly: true,
        snapshotMethod: "better-sqlite3.backup"
      },
      models: {
        embedding: publicModelConfig(config.embedding),
        skill: publicModelConfig(resolveEvolutionConfig(config))
      },
      config: {
        mining: config.algorithm.longTrajectory,
        skill: {
          outcomeRTaskSuccessThreshold:
            config.algorithm.skill.outcomeRTaskSuccessThreshold
        },
        processing: {
          episodeOrder: "opened_at ASC, episode_id ASC",
          futurePathsHiddenUntilArrival: true,
          miningJobsRunOneAtATime: true,
          skillInductionAfterEachEpisode: true,
          skillInductionWorkerLimit: 1
        }
      },
      input: {
        episodes: paths.length,
        steps: paths.reduce((sum, item) => sum + item.stepCount, 0),
        arrivalOrder: paths.map((item) => item.episodeId)
      },
      execution: {
        episodeRuns,
        queuedBeforeFinalDrain,
        drain,
        jobs: jobCounts(target.db)
      },
      candidates,
      summary: {
        physicalCandidates: candidates.length,
        candidatesWithMultipleVersions: candidates.filter((candidate) =>
          candidate.versionCount > 1).length,
        candidateVersions: candidates.reduce((sum, candidate) =>
          sum + candidate.versionCount, 0),
        candidateSkillVersions: candidates.reduce((sum, candidate) =>
          sum + candidate.skillVersionHistory.length, 0),
        liveSkills: liveSkills.length,
        rejectedCandidates: candidates.filter((candidate) =>
          candidate.skill?.admitted === false).length,
        liveSkillNames: liveSkills.map((candidate) => candidate.skill?.name)
      }
    };
    writeFileSync(outputReportPath, `${JSON.stringify(report, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600
    });
    process.stdout.write(`${JSON.stringify({
      outputDatabase: outputDbPath,
      outputReport: outputReportPath,
      input: report.input,
      summary: report.summary,
      drain: report.execution.drain
    }, null, 2)}\n`);
  } finally {
    void service;
    target.close();
  }
}

interface CandidateFirstReplayPath {
  pathId: string;
  episodeId: string;
  userId: string;
  sessionId: string;
  pathHash: string;
  stepCount: number;
  rTask: number;
  openedAt: string;
}

interface CandidateHeadSnapshot {
  candidateId: string;
  activeVersionId?: string;
  versionNo?: number;
  supportEpisodeIds: string[];
  activeSkillMemoryId?: string;
}

interface CandidateReportSkill {
  admitted: boolean;
  reason?: string;
  live: boolean;
  memoryId?: string;
  publicStatus?: string;
  semanticStatus?: string;
  name?: string;
  invocationGuide?: string;
  procedure?: unknown;
  verification?: unknown;
  sourceEpisodeIds: string[];
}

interface CandidateReportRow {
  candidateId: string;
  status: string;
  structureKey: string;
  versionCount: number;
  activeVersion?: {
    id: string;
    versionNo: number;
    referenceEpisodeId: string;
    supportEpisodeIds: string[];
    evidenceDelta?: unknown;
    trajectory?: unknown;
  };
  versionHistory: Array<{
    id: string;
    versionNo: number;
    status: string;
    supportEpisodeIds: string[];
  }>;
  skillVersionHistory: Array<{
    id: string;
    versionNo: number;
    candidateVersionId: string;
    status: string;
    admitted: boolean;
    reason?: string;
    skillMemoryId?: string;
  }>;
  skill?: CandidateReportSkill;
}

interface WorkerDrainResult {
  drained: boolean;
  rounds: number;
  leased: number;
  succeeded: number;
  failed: number;
  elapsedMs: number;
  remainingByStatus: Record<string, number>;
}

function prepareCandidateFirstReplay(
  db: Database.Database,
  repos: Repositories,
  successThreshold: number
): CandidateFirstReplayPath[] {
  const paths = db.prepare(
    `SELECT paths.id AS pathId, paths.episode_id AS episodeId,
            paths.user_id AS userId, episodes.session_id AS sessionId,
            paths.path_hash AS pathHash, paths.step_count AS stepCount,
            episodes.r_task AS rTask, episodes.opened_at AS openedAt
     FROM episode_execution_paths AS paths
     JOIN episodes ON episodes.id = paths.episode_id
     WHERE paths.status = 'active' AND episodes.status = 'closed'
       AND episodes.r_task >= ?
     ORDER BY episodes.opened_at ASC, episodes.id ASC`
  ).all(successThreshold) as CandidateFirstReplayPath[];
  if (paths.length < 2) {
    throw new Error("Candidate-first replay requires at least two successful active Paths");
  }
  const at = new Date().toISOString();
  db.transaction(() => {
    db.prepare(`DELETE FROM evolution_jobs`).run();
    db.prepare(`DELETE FROM long_trajectory_skill_versions`).run();
    db.prepare(
      `UPDATE long_trajectory_candidates
       SET active_version_id = NULL, active_skill_version_id = NULL,
           active_skill_memory_id = NULL`
    ).run();
    db.prepare(`DELETE FROM long_trajectory_candidate_versions`).run();
    db.prepare(`DELETE FROM long_trajectory_candidates`).run();
    db.prepare(`DELETE FROM long_trajectory_episode_representations`).run();
    for (const path of paths) {
      db.prepare(
        `UPDATE episode_execution_paths
         SET status = 'superseded', deactivated_at = ? WHERE id = ?`
      ).run(at, path.pathId);
    }
  })();
  if (candidateHeadSnapshot(db, repos).length !== 0) {
    throw new Error("Candidate-first replay cleanup left active V3 Candidates");
  }
  return paths;
}

function activateReplayPath(db: Database.Database, pathId: string, at: string): void {
  const result = db.prepare(
    `UPDATE episode_execution_paths
     SET status = 'active', activated_at = ?, deactivated_at = NULL
     WHERE id = ? AND status = 'superseded'`
  ).run(at, pathId);
  if (result.changes !== 1) throw new Error(`Failed to activate replay Path: ${pathId}`);
}

function enqueueReplayJob(
  repos: Repositories,
  input: EnqueueJobInput
): EvolutionJobRecord {
  const at = new Date().toISOString();
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
    createdAt: input.createdAt ?? at,
    updatedAt: input.createdAt ?? at
  });
}

async function drainWorker(
  service: MemoryService,
  db: Database.Database,
  maxRounds: number,
  workerLimit: number
): Promise<WorkerDrainResult> {
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

function candidateHeadSnapshot(
  db: Database.Database,
  repos: Repositories
): CandidateHeadSnapshot[] {
  const ids = (db.prepare(
    `SELECT id FROM long_trajectory_candidates
     WHERE status = 'active' ORDER BY id ASC`
  ).all() as Array<{ id: string }>).map((item) => item.id);
  return ids.flatMap((candidateId) => {
    const candidate = repos.longTrajectory.getCandidate(candidateId);
    if (!candidate) return [];
    const version = candidate.activeVersionId
      ? repos.longTrajectory.getCandidateVersion(candidate.activeVersionId)
      : undefined;
    return [{
      candidateId,
      ...(candidate.activeVersionId ? { activeVersionId: candidate.activeVersionId } : {}),
      ...(version ? { versionNo: version.versionNo } : {}),
      supportEpisodeIds: version?.supportEpisodeIds ?? [],
      ...(candidate.activeSkillMemoryId
        ? { activeSkillMemoryId: candidate.activeSkillMemoryId }
        : {})
    }];
  });
}

function changedCandidateHeads(
  before: readonly CandidateHeadSnapshot[],
  after: readonly CandidateHeadSnapshot[]
): Array<Record<string, unknown>> {
  const beforeById = new Map(before.map((item) => [item.candidateId, item]));
  return after.flatMap((item) => {
    const previous = beforeById.get(item.candidateId);
    if (previous?.activeVersionId === item.activeVersionId &&
        previous?.activeSkillMemoryId === item.activeSkillMemoryId) return [];
    return [{
      candidateId: item.candidateId,
      change: previous ? "updated" : "created",
      previousVersionNo: previous?.versionNo ?? null,
      activeVersionNo: item.versionNo ?? null,
      previousSupportEpisodeIds: previous?.supportEpisodeIds ?? [],
      supportEpisodeIds: item.supportEpisodeIds
    }];
  });
}

function candidateReport(
  db: Database.Database,
  repos: Repositories
): CandidateReportRow[] {
  const ids = (db.prepare(
    `SELECT id FROM long_trajectory_candidates ORDER BY status, created_at, id`
  ).all() as Array<{ id: string }>).map((item) => item.id);
  return ids.flatMap((candidateId) => {
    const candidate = repos.longTrajectory.getCandidate(candidateId);
    if (!candidate) return [];
    const versions = db.prepare(
      `SELECT id, version_no AS versionNo, status, support_episode_ids_json AS supportJson
       FROM long_trajectory_candidate_versions
       WHERE candidate_id = ? ORDER BY version_no ASC`
    ).all(candidateId) as Array<{
      id: string;
      versionNo: number;
      status: string;
      supportJson: string;
    }>;
    const skillVersions = db.prepare(
      `SELECT id, version_no AS versionNo,
              candidate_version_id AS candidateVersionId, status,
              skill_memory_id AS skillMemoryId, payload_json AS payloadJson
       FROM long_trajectory_skill_versions
       WHERE candidate_id = ? ORDER BY version_no ASC`
    ).all(candidateId) as Array<{
      id: string;
      versionNo: number;
      candidateVersionId: string;
      status: string;
      skillMemoryId: string | null;
      payloadJson: string;
    }>;
    const activeVersion = candidate.activeVersionId
      ? repos.longTrajectory.getCandidateVersion(candidate.activeVersionId)
      : undefined;
    const activeSkillVersion = candidate.activeSkillVersionId
      ? repos.longTrajectory.getSkillVersion(candidate.activeSkillVersionId)
      : undefined;
    const memory = activeSkillVersion?.skillMemoryId
      ? repos.memories.get(activeSkillVersion.skillMemoryId)
      : undefined;
    const internal = memory?.properties.internal_info;
    const skillInfo = internal && isRecord(internal.skill) ? internal.skill : {};
    const sourceEpisodeIds = internal && Array.isArray(internal.source_episode_ids)
      ? internal.source_episode_ids.filter((item): item is string => typeof item === "string")
      : [];
    const admitted = activeSkillVersion?.payload.admitted === true;
    const reason = typeof activeSkillVersion?.payload.reason === "string"
      ? activeSkillVersion.payload.reason
      : undefined;
    const semanticStatus = typeof skillInfo.status === "string" ? skillInfo.status : undefined;
    const live = Boolean(memory && memory.status !== "archived" && memory.status !== "deleted" &&
      (semanticStatus === "candidate" || semanticStatus === "active"));
    const skill: CandidateReportSkill | undefined = activeSkillVersion ? {
      admitted,
      ...(reason ? { reason } : {}),
      live,
      ...(memory ? {
        memoryId: memory.id,
        publicStatus: memory.status
      } : {}),
      ...(semanticStatus ? { semanticStatus } : {}),
      ...(typeof skillInfo.name === "string" ? { name: skillInfo.name } : {}),
      ...(typeof skillInfo.invocation_guide === "string"
        ? { invocationGuide: skillInfo.invocation_guide }
        : {}),
      ...(skillInfo.procedure_json !== undefined
        ? { procedure: skillInfo.procedure_json }
        : {}),
      ...(skillInfo.verification !== undefined
        ? { verification: skillInfo.verification }
        : {}),
      sourceEpisodeIds
    } : undefined;
    return [{
      candidateId,
      status: candidate.status,
      structureKey: candidate.structureKey,
      versionCount: versions.length,
      ...(activeVersion ? {
        activeVersion: {
          id: activeVersion.id,
          versionNo: activeVersion.versionNo,
          referenceEpisodeId: activeVersion.referenceEpisodeId,
          supportEpisodeIds: activeVersion.supportEpisodeIds,
          ...(activeVersion.payload.evidenceDelta !== undefined
            ? { evidenceDelta: activeVersion.payload.evidenceDelta }
            : {}),
          ...(activeVersion.payload.trajectory !== undefined
            ? { trajectory: activeVersion.payload.trajectory }
            : {})
        }
      } : {}),
      versionHistory: versions.map((version) => ({
        id: version.id,
        versionNo: version.versionNo,
        status: version.status,
        supportEpisodeIds: parseStringArray(version.supportJson)
      })),
      skillVersionHistory: skillVersions.map((version) => {
        const payload = parseJsonObject(version.payloadJson);
        return {
          id: version.id,
          versionNo: version.versionNo,
          candidateVersionId: version.candidateVersionId,
          status: version.status,
          admitted: payload.admitted === true,
          ...(typeof payload.reason === "string" ? { reason: payload.reason } : {}),
          ...(version.skillMemoryId ? { skillMemoryId: version.skillMemoryId } : {})
        };
      }),
      ...(skill ? { skill } : {})
    }];
  });
}

function unfinishedJobCounts(db: Database.Database): Record<string, number> {
  const rows = db.prepare(
    `SELECT status, COUNT(*) AS count FROM evolution_jobs
     WHERE status IN ('queued', 'leased', 'failed', 'dead_letter')
     GROUP BY status`
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

function publicModelConfig(config: {
  provider: string;
  model?: string;
  endpoint?: string;
}): Record<string, unknown> {
  return {
    provider: config.provider,
    ...(config.model ? { model: config.model } : {}),
    ...(config.endpoint ? { endpoint: config.endpoint } : {})
  };
}

function parseStringArray(value: string): string[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

function parseJsonObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown;
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

async function loadDocuments(
  db: Database.Database,
  repos: Repositories,
  embedder: Embedder,
  successThreshold: number
): Promise<EpisodeTrajectoryDocumentV1[]> {
  const rows = db.prepare(
    `SELECT paths.id AS pathId, paths.episode_id AS episodeId,
            episodes.title, episodes.summary,
            (SELECT raw.user_text FROM raw_turns AS raw
             WHERE raw.episode_id = episodes.id
             ORDER BY raw.created_at ASC, raw.id ASC LIMIT 1) AS userText,
            (SELECT raw.assistant_text FROM raw_turns AS raw
             WHERE raw.episode_id = episodes.id
               AND raw.assistant_text IS NOT NULL
             ORDER BY raw.created_at DESC, raw.id DESC LIMIT 1) AS terminalResultText
     FROM episode_execution_paths AS paths
     JOIN episodes ON episodes.id = paths.episode_id
     WHERE paths.status = 'active' AND episodes.status = 'closed'
       AND episodes.r_task >= ?
     ORDER BY paths.episode_id ASC`
  ).all(successThreshold) as Array<{
    pathId: string;
    episodeId: string;
    title: string | null;
    summary: string | null;
    userText: string | null;
    terminalResultText: string | null;
  }>;
  const partial: Array<{
    path: EpisodeExecutionPathLiteV1;
    goalText: string;
    terminalResultText: string;
    trajectoryText: string;
    windows: ReturnType<typeof buildTrajectoryWindows>;
    stepVectorById: Map<string, number[]>;
    existingWindowVectorById: Map<string, number[]>;
  }> = [];
  const missingWindowTexts: string[] = [];
  const missingWindowKeys: string[] = [];
  for (const row of rows) {
    const record = repos.proceduralTrajectory.getPath(row.pathId);
    if (!record) continue;
    const episode = repos.runtime.getEpisode(row.episodeId);
    const path: EpisodeExecutionPathLiteV1 = {
      ...record.path,
      ...(episode?.rTask === undefined ? {} : { terminalReward: episode.rTask })
    };
    const stepEmbeddings = repos.proceduralTrajectory.listStepEmbeddings({
      pathId: path.id,
      representationVersion: PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION
    });
    if (stepEmbeddings.length !== path.steps.length) {
      throw new Error(`Path ${path.id} is missing Step embeddings`);
    }
    const stepVectorById = new Map(stepEmbeddings.map((item) => [item.stepId, item.vector]));
    const existingWindowVectorById = new Map(
      repos.proceduralTrajectory.listWindowsForPath(path.id)
        .map((item) => [item.id, item.coarseVector])
    );
    const windows = buildTrajectoryWindows([path], WINDOW_SPECS);
    for (const window of windows) {
      if (existingWindowVectorById.has(window.id)) continue;
      missingWindowKeys.push(window.id);
      missingWindowTexts.push(window.semanticText);
    }
    const goalText = [row.title, row.summary, row.userText]
      .filter((item): item is string => Boolean(item?.trim())).join("\n");
    partial.push({
      path,
      goalText: goalText || `Episode ${row.episodeId}`,
      terminalResultText: row.terminalResultText?.trim() ||
        row.summary?.trim() || "Episode completed successfully.",
      trajectoryText: trajectoryIntentSequenceText(path),
      windows,
      stepVectorById,
      existingWindowVectorById
    });
  }
  const [goalVectors, trajectoryVectors, missingWindowVectors] = await Promise.all([
    embedBatches(embedder, partial.map((item) => item.goalText)),
    embedBatches(embedder, partial.map((item) => item.trajectoryText)),
    embedBatches(embedder, missingWindowTexts)
  ]);
  const missingVectorByWindowId = new Map(missingWindowKeys.map((id, index) => [
    id,
    unitVector(missingWindowVectors[index] ?? [])
  ]));
  return partial.map((item, index) => ({
    path: item.path,
    goalText: item.goalText,
    terminalResultText: item.terminalResultText,
    goalVector: unitVector(goalVectors[index] ?? []),
    trajectoryText: item.trajectoryText,
    trajectoryVector: unitVector(trajectoryVectors[index] ?? []),
    windows: item.windows.map((occurrence): EmbeddedTrajectoryWindowV1 => ({
      occurrence,
      coarseVector: item.existingWindowVectorById.get(occurrence.id) ??
        missingVectorByWindowId.get(occurrence.id) ?? [],
      stepVectors: occurrence.steps.map((step) => {
        const vector = item.stepVectorById.get(step.id);
        if (!vector) throw new Error(`Missing Step vector: ${step.id}`);
        return vector;
      })
    }))
  }));
}

function uniqueFamilies(documents: readonly EpisodeTrajectoryDocumentV1[]): EpisodeTrajectoryFamilyV1[] {
  return documents.flatMap((seed) => {
    const family = buildEpisodeTrajectoryFamily(seed, documents, MINING_CONFIG);
    return family ? [family] : [];
  }).sort((left, right) =>
    left.referenceEpisodeId.localeCompare(right.referenceEpisodeId));
}

function toSkillInput(
  trajectory: LongCommonTrajectoryV1,
  documents: readonly EpisodeTrajectoryDocumentV1[],
  repos: Repositories
): ProceduralPatternSkillInput | undefined {
  return buildLongTrajectorySkillInput({
    trajectory,
    documents,
    userId: documents[0]?.path.userId ?? "",
    sourceTraceIdsForSteps: (episodeId, steps) =>
      sourceTraceIdsForSteps(episodeId, steps, repos)
  });
}

function sourceTraceIdsForSteps(
  episodeId: string,
  steps: readonly ExecutionStepLiteV1[],
  repos: Repositories
): string[] {
  const episode = repos.runtime.getEpisode(episodeId);
  if (!episode) return [];
  const rawTurnIds = new Set(steps.map((step) => step.rawTurnId));
  return repos.memories.getMany(episode.l1MemoryIds).flatMap((memory) => {
    const trace = traceMetaFromMemory(memory);
    return trace?.episodeId === episodeId && trace.rawTurnId && rawTurnIds.has(trace.rawTurnId)
      ? [trace.id]
      : [];
  });
}

function dedupeTrajectories<T extends { trajectory: LongCommonTrajectoryV1; quality: number }>(
  rows: readonly T[]
): T[] {
  const byEvidence = new Map<string, T>();
  for (const row of rows) {
    const key = stableHash({
      supportEpisodeIds: row.trajectory.supportEpisodeIds,
      occurrences: row.trajectory.occurrences.map((item) => ({
        episodeId: item.episodeId,
        episodeSpanIds: item.matches.map((match) => match.episodeSpanId)
      })).sort((left, right) => left.episodeId.localeCompare(right.episodeId))
    });
    const current = byEvidence.get(key);
    if (!current || row.quality > current.quality ||
        (row.quality === current.quality &&
          row.trajectory.id.localeCompare(current.trajectory.id) < 0)) {
      byEvidence.set(key, row);
    }
  }
  return [...byEvidence.values()];
}

function projectionDiagnostics(
  family: EpisodeTrajectoryFamilyV1,
  reference: EpisodeTrajectoryDocumentV1,
  _documents: readonly EpisodeTrajectoryDocumentV1[],
  projections: readonly EpisodeSpanSequenceProjectionV1[],
  trajectories: readonly LongCommonTrajectoryV1[],
  preMaximalLongTrajectories: number
): Record<string, unknown> {
  return {
    familyId: family.id,
    referenceEpisodeId: family.referenceEpisodeId,
    memberEpisodeIds: family.memberEpisodeIds,
    memberCount: family.memberEpisodeIds.length,
    referenceSteps: reference.path.steps.length,
    projections: projections.map((projection) => ({
      episodeId: projection.episodeId,
      matchedSpans: projection.matches.length,
      spanPath: projection.matches.map((match) => match.referenceSpanLabel),
      scales: projection.matches.map((match) => match.scale),
      gaps: projection.gaps.map((gap) => gap.stepIds.length),
      referenceCoverage: projection.referenceCoverage,
      episodeCoverage: projection.episodeCoverage,
      averageCoarseSimilarity: projection.averageCoarseSimilarity
    })),
    projectionsWithMatches: projections.filter((item) => item.matches.length > 0).length,
    projectionsWithAtLeastTwoSpans: projections.filter((item) => item.matches.length >= 2).length,
    preMaximalLongTrajectories,
    maximalLongTrajectories: trajectories.length
  };
}

function candidateSnapshot(row: {
  family: EpisodeTrajectoryFamilyV1;
  trajectory: LongCommonTrajectoryV1;
  input: ProceduralPatternSkillInput;
  quality: number;
}): Record<string, unknown> {
  return {
    familyId: row.family.id,
    referenceEpisodeId: row.trajectory.referenceEpisodeId,
    longTrajectoryId: row.trajectory.id,
    supportEpisodeIds: row.trajectory.supportEpisodeIds,
    referenceRange: [
      row.trajectory.referenceStartStepIndex,
      row.trajectory.referenceEndStepIndex
    ],
    spanSequence: row.trajectory.requiredSpans.map((span) => ({
      label: span.referenceSpanLabel,
      scale: span.scale,
      range: [span.referenceStartStepIndex, span.referenceEndStepIndex],
      semanticText: span.semanticText,
      supportEpisodeIds: span.supportEpisodeIds,
      averageCoarseSimilarity: span.averageCoarseSimilarity
    })),
    occurrences: row.input.evidence.map((occurrence) => ({
      occurrenceId: occurrence.occurrenceId,
      episodeId: occurrence.episodeId,
      alignmentScore: occurrence.alignmentScore,
      alignedSequence: occurrence.alignedSequence.map((step) => ({
        role: step.role,
        stepId: step.stepId,
        stepIndex: step.stepIndex,
        intent: step.intent,
        summary: step.summary,
        outcome: step.outcome,
        ...(step.role === "core" ? {
          anchorId: step.anchorId,
          alignmentGroupId: step.alignmentGroupId,
          matchSimilarity: step.matchSimilarity
        } : step.role === "span_step" ? {
          anchorId: step.anchorId,
          spanSimilarity: step.spanSimilarity
        } : step.role === "gap" ? {
          afterAnchorId: step.afterAnchorId,
          beforeAnchorId: step.beforeAnchorId
        } : {
          spanAnchorId: step.spanAnchorId
        })
      }))
    }))
  };
}

function decisionBase(
  trajectory: LongCommonTrajectoryV1,
  extra: Record<string, unknown>
): Record<string, unknown> {
  return {
    longTrajectoryId: trajectory.id,
    supportEpisodeIds: trajectory.supportEpisodeIds,
    spanTokens: trajectory.requiredSpans.length,
    spanSteps: trajectory.referenceEndStepIndex - trajectory.referenceStartStepIndex + 1,
    coverage: trajectory.averageEpisodeCoverage,
    similarity: trajectory.averageCoarseSimilarity,
    ...extra
  };
}

async function embedBatches(embedder: Embedder, texts: readonly string[]): Promise<number[][]> {
  const result: number[][] = [];
  for (let index = 0; index < texts.length; index += 32) {
    result.push(...await embedder.embed(texts.slice(index, index + 32), "document"));
  }
  return result;
}

function countedEmbedder(base: Embedder, stats: ReplayStats): Embedder {
  return {
    config: base.config,
    isRemote: () => base.isRemote(),
    status: () => base.status(),
    embed: async (texts, role) => {
      if (texts.length === 0) return [];
      stats.embeddingCalls += 1;
      stats.embeddedTexts += texts.length;
      return base.embed(texts, role);
    },
    embedOne: async (text, role) => {
      stats.embeddingCalls += 1;
      stats.embeddedTexts += 1;
      return base.embedOne(text, role);
    }
  };
}

function countedLlm(base: LlmClient, stats: ReplayStats): LlmClient {
  return {
    config: base.config,
    isConfigured: () => base.isConfigured(),
    status: () => base.status(),
    complete: async (messages, options) => {
      stats.skillLlmCalls += 1;
      return base.complete(messages, options);
    },
    completeJson: async <T extends Record<string, unknown>>(messages: Parameters<
      LlmClient["completeJson"]
    >[0], options: Parameters<LlmClient["completeJson"]>[1]): Promise<T> => {
      stats.skillLlmCalls += 1;
      return base.completeJson<T>(messages, options);
    }
  };
}

function removePriorLongTrajectorySkills(db: Database.Database, repos: Repositories): void {
  const ids = (db.prepare(
    `SELECT id FROM memories WHERE memory_layer = 'Skill'
       AND json_extract(properties_json, '$.internal_info.plugin_algorithm') =
           'procedural.long-trajectory.skill.v1'`
  ).all() as Array<{ id: string }>).map((item) => item.id);
  db.transaction(() => {
    for (const id of ids) {
      repos.vectors.deleteMemory(id);
      db.prepare(`DELETE FROM skill_trials WHERE skill_memory_id = ?`).run(id);
      db.prepare(`DELETE FROM memories_fts WHERE id = ?`).run(id);
      db.prepare(`DELETE FROM memories WHERE id = ?`).run(id);
    }
  })();
}

function buildReplayMemory(input: Record<string, unknown>): MemoryRow {
  const at = typeof input.createdAt === "string" ? input.createdAt : new Date().toISOString();
  const layer = String(input.layer) as MemoryRow["memoryLayer"];
  const lifecycle = String(input.lifecycleStatus ?? "active");
  const status = lifecycle === "candidate"
    ? "resolving"
    : lifecycle === "archived" ? "archived" : "activated";
  const tags = Array.isArray(input.tags) ? input.tags.filter((item): item is string =>
    typeof item === "string") : [];
  const info = isRecord(input.info) ? input.info : {};
  const internal = isRecord(input.internal) ? input.internal : {};
  const value = String(input.value ?? "");
  return {
    id: typeof input.id === "string" ? input.id : newId("skill"),
    timeline: at,
    userId: String(input.userId),
    ...(typeof input.conversationId === "string" ? { conversationId: input.conversationId } : {}),
    ...(typeof input.sessionId === "string" ? { sessionId: input.sessionId } : {}),
    ...(typeof input.agentId === "string" ? { agentId: input.agentId } : {}),
    ...(typeof input.appId === "string" ? { appId: input.appId } : {}),
    memoryType: String(input.memoryType),
    status,
    visibility: "private",
    ...(typeof input.key === "string" ? { memoryKey: input.key } : {}),
    memoryValue: value,
    tags,
    info,
    properties: {
      memory_type: String(input.memoryType), status, tags, info,
      internal_info: {
        memory_layer: layer,
        memory_kind: String(input.kind),
        schema_version: 1,
        ...internal
      }
    },
    memoryLayer: layer,
    contentHash: stableHash(value),
    version: 1,
    createdAt: at,
    updatedAt: at,
    deletedAt: null
  };
}

function serializableConfig(config: LongTrajectoryMiningConfigV1): Record<string, unknown> {
  return { ...config };
}

function countBy<T>(items: readonly T[], key: (item: T) => string): Record<string, number> {
  const result: Record<string, number> = {};
  for (const item of items) {
    const itemKey = key(item);
    result[itemKey] = (result[itemKey] ?? 0) + 1;
  }
  return result;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseArgs(argv: string[]): ReplayArgs {
  const result: ReplayArgs = {
    help: false,
    confirmRemoteModels: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    if (arg === "--help" || arg === "-h") result.help = true;
    else if (arg === "--confirm-remote-models") result.confirmRemoteModels = true;
    else if (arg === "--db") result.sourceDbPath = argv[++index];
    else if (arg === "--output-db") result.outputDbPath = argv[++index];
    else if (arg === "--output") result.outputReportPath = argv[++index];
    else if (arg === "--config") result.configPath = argv[++index];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return result;
}

function requiredPath(value: string | undefined, flag: string): string {
  if (!value) throw new Error(`Missing required argument ${flag}`);
  return resolve(value);
}

function assertSafePaths(source: string, outputDb: string, report: string): void {
  if (!existsSync(source)) throw new Error(`Source database does not exist: ${source}`);
  if (source === outputDb || source === report || outputDb === report) {
    throw new Error("Source, output database, and report paths must be distinct");
  }
  if (existsSync(outputDb) || existsSync(report)) {
    throw new Error("Output path already exists; this replay never overwrites files");
  }
}

function helpText(): string {
  return `KW long-trajectory V3 Candidate-first worker replay\n\n` +
    `node --import tsx scripts/kw-long-trajectory-v3-replay.ts \\\n  --db <v2-kw.sqlite> --output-db <fresh.sqlite> --output <report.json> \\\n  [--config <config.yaml>] --confirm-remote-models\n`;
}

void main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
