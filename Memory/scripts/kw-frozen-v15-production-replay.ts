import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import Database from "better-sqlite3";
import {
  MemoryDb,
  MemoryService,
  createEmbedder,
  loadMemmyConfig,
  type Embedder,
  type MemoryRow
} from "../src/index.js";
import { traceMetaFromMemory } from "../src/algorithm/plugin-algorithms.js";
import {
  EXECUTION_STEP_LITE_SCHEMA_VERSION,
  EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
  PROCEDURAL_WINDOW_MINING_VERSION,
  buildTrajectoryWindows,
  extractAlignedCommonCore,
  selectMaximalWindowClusters,
  type EpisodeExecutionPathLiteV1,
  type ExecutionStepLiteOutcome,
  type ExecutionStepLiteV1,
  type TrajectoryWindowClusterV1
} from "../src/service/evolution/procedural-window-model.js";
import {
  bandedMonotonicMatch,
  cosineSimilarity,
  type BandedMonotonicMatchConfig
} from "../src/service/evolution/trajectory-window-alignment.js";
import {
  PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION,
  type EpisodeExecutionPathRecord,
  type TrajectoryWindowOccurrenceRecord
} from "../src/storage/procedural-trajectory-repository.js";
import {
  Repositories,
  type EpisodeRecord,
  type EvolutionJobRecord,
  type RawTurnRecord
} from "../src/storage/repositories.js";
import {
  evolutionJobDedupeKey,
  type EnqueueJobInput
} from "../src/service/worker/job-handlers.js";
import { newId, stableHash } from "../src/utils/id.js";

const FROZEN_PATH_COMPILER_VERSION = "frozen-v15-path-adapter.v1";
const FROZEN_RECONSTRUCTION_VERSION = "episode-procedural-reconstruction.v8";
const PROCEDURAL_SKILL_ALGORITHM = "procedural.pattern.skill.v1";

interface Args {
  help: boolean;
  sourceDbPath?: string;
  v15ResultPath?: string;
  outputDbPath?: string;
  outputReportPath?: string;
  configPath?: string;
  maxRounds: number;
  workerLimit: number;
  confirmRemoteModels: boolean;
  prepareOnly: boolean;
}

interface SqlRow {
  [key: string]: unknown;
}

interface SourceEpisodeRow extends SqlRow {
  id: string;
  session_id: string;
  user_id: string;
  status: string;
  l1_memory_ids_json: string;
  r_task: number | null;
  opened_at: string;
}

interface SourcePathRow {
  id: string;
  episode_id: string;
  user_id: string;
  reconstruction_model: string | null;
  source_snapshot_hash: string;
  terminal_reward: number | null;
  payload_json: string;
}

interface SourceStepRow {
  path_id: string;
  step_id: string;
  episode_id: string;
  step_index: number;
  raw_turn_id: string;
  intent: string;
  summary: string;
  outcome: ExecutionStepLiteOutcome;
  tool_name: string | null;
  step_json: string;
  vector_json: string;
  embedding_dim: number;
}

interface V15CandidateMember {
  episodeId: string;
  scale: number;
  startStepIndex: number;
  endStepIndex: number;
}

interface V15Candidate {
  clusterId: string;
  scale: number;
  members: V15CandidateMember[];
}

interface DrainResult {
  drained: boolean;
  rounds: number;
  leased: number;
  succeeded: number;
  failed: number;
  elapsedMs: number;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(helpText());
    return;
  }
  const sourceDbPath = requiredPath(args.sourceDbPath, "--source-db");
  const v15ResultPath = requiredPath(args.v15ResultPath, "--v15-result");
  const outputDbPath = requiredPath(args.outputDbPath, "--output-db");
  const outputReportPath = resolve(args.outputReportPath ?? `${outputDbPath}.report.json`);
  const configPath = args.configPath ? resolve(args.configPath) : undefined;
  if (!args.prepareOnly && !args.confirmRemoteModels) {
    throw new Error(
      "This experiment sends frozen v15 intent sequences to the configured embedding service " +
      "and qualified clusters to the configured Skill LLM. Re-run with --confirm-remote-models."
    );
  }
  for (const path of [sourceDbPath, v15ResultPath]) {
    if (!existsSync(path)) throw new Error(`Input not found: ${path}`);
  }
  if (sourceDbPath === outputDbPath) throw new Error("--output-db must differ from --source-db");
  if (existsSync(outputDbPath)) throw new Error(`Refusing to overwrite output DB: ${outputDbPath}`);
  if (existsSync(outputReportPath)) {
    throw new Error(`Refusing to overwrite output report: ${outputReportPath}`);
  }

  const loaded = loadMemmyConfig(configPath);
  const embedder = createEmbedder(loaded.config.embedding);
  const embeddingSignature = proceduralEmbeddingSignature(embedder);
  const v15Candidates = readV15Candidates(v15ResultPath);
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();
  mkdirSync(dirname(outputDbPath), { recursive: true });
  mkdirSync(dirname(outputReportPath), { recursive: true });

  const sourceDb = new MemoryDb({ path: sourceDbPath, readonly: true });
  const targetDb = new MemoryDb({ path: outputDbPath });
  let service: MemoryService | undefined;
  try {
    const sourceRepos = new Repositories(sourceDb.db);
    const targetRepos = new Repositories(targetDb.db);
    sourceDb.db.exec("BEGIN");
    const imported = importFrozenCohort(sourceDb.db, sourceRepos, targetDb.db, targetRepos);
    sourceDb.db.exec("ROLLBACK");

    const frozen = seedFrozenPathsAndEmbeddings({
      sourceDb: sourceDb.db,
      targetRepos,
      episodes: imported.episodes,
      embeddingSignature
    });
    seedWindowJobs(targetRepos, frozen.paths, imported.episodes);

    const drain: DrainResult = args.prepareOnly
      ? {
          drained: false,
          rounds: 0,
          leased: 0,
          succeeded: 0,
          failed: 0,
          elapsedMs: 0
        }
      : await (async () => {
          service = new MemoryService({
            db: targetDb,
            mode: "local",
            config: loaded.config,
            ...(configPath ? { configPath } : {}),
            embedder
          });
          return drainWorker(service, args.maxRounds, args.workerLimit);
        })();
    const artifact = buildArtifact({
      startedAt,
      elapsedMs: Date.now() - wallStart,
      sourceDbPath,
      v15ResultPath,
      outputDbPath,
      outputReportPath,
      config: loaded.config.algorithm.proceduralWindow,
      imported,
      frozen,
      embeddingSignature,
      drain,
      targetDb: targetDb.db,
      targetRepos,
      v15Candidates
    });
    writeFileSync(outputReportPath, `${JSON.stringify(artifact, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600
    });
    process.stdout.write(`${JSON.stringify({
      outputDatabase: outputDbPath,
      outputReport: outputReportPath,
      episodes: imported.episodes.length,
      frozenSteps: frozen.stepCount,
      windows: artifact.summary.windows,
      activeClusters: artifact.summary.activeClusters,
      supportedClusters: artifact.summary.supportedClusters,
      maximalClusters: artifact.summary.maximalClusters,
      skills: artifact.summary.skills,
      drain
    }, null, 2)}\n`);
    if (!args.prepareOnly && !drain.drained) process.exitCode = 2;
  } finally {
    if (sourceDb.db.inTransaction) sourceDb.db.exec("ROLLBACK");
    void service;
    targetDb.close();
    sourceDb.close();
  }
}

function importFrozenCohort(
  sourceDb: Database.Database,
  sourceRepos: Repositories,
  targetDb: Database.Database,
  targetRepos: Repositories
): {
  episodes: SourceEpisodeRow[];
  sessionIds: string[];
  rawTurnIds: string[];
  l1Ids: string[];
  omittedL1Ids: string[];
} {
  const episodes = sourceDb.prepare(
    `SELECT * FROM episodes
     WHERE status = 'closed'
       AND id IN (
         SELECT episode_id FROM episode_procedural_paths
         WHERE status = 'active' AND reconstruction_algorithm_version = ?
       )
     ORDER BY opened_at ASC, id ASC`
  ).all(FROZEN_RECONSTRUCTION_VERSION) as SourceEpisodeRow[];
  if (episodes.length === 0) throw new Error("No frozen v15 Episodes found");
  const episodeIds = episodes.map((episode) => episode.id);
  const sessionIds = unique(episodes.map((episode) => episode.session_id));
  const sourceL1Ids = unique(episodes.flatMap((episode) =>
    parseStringArray(episode.l1_memory_ids_json)
  ));
  const memories = sourceRepos.memories.getMany(sourceL1Ids).filter((memory) =>
    memory.memoryLayer === "L1" &&
    memory.status === "activated" &&
    memory.properties.internal_info.plugin_algorithm === "capture.v7"
  );
  const l1Ids = memories.map((memory) => memory.id);
  const l1Set = new Set(l1Ids);
  const omittedL1Ids = sourceL1Ids.filter((id) => !l1Set.has(id));
  const normalizedEpisodes = episodes.map((episode) => ({
    ...episode,
    l1_memory_ids_json: JSON.stringify(
      parseStringArray(episode.l1_memory_ids_json).filter((id) => l1Set.has(id))
    ),
    l2_policy_ids_json: "[]",
    l3_world_model_ids_json: "[]",
    skill_memory_ids_json: "[]",
    feedback_ids_json: "[]",
    decision_repair_ids_json: "[]",
    pipeline_run_id: null,
    pipeline_status: "idle",
    pipeline_error: null
  }));
  const sessions = selectRowsByValues(sourceDb, "sessions", "id", sessionIds);
  const rawTurns = selectRowsByValues(sourceDb, "raw_turns", "episode_id", episodeIds);
  targetDb.transaction(() => {
    insertRows(targetDb, "sessions", sessions);
    insertRows(targetDb, "episodes", normalizedEpisodes);
    insertRows(targetDb, "raw_turns", rawTurns);
    for (const memory of memories) targetRepos.memories.insert(memory);
  })();
  return {
    episodes: normalizedEpisodes,
    sessionIds,
    rawTurnIds: rawTurns.flatMap((row) => typeof row.id === "string" ? [row.id] : []),
    l1Ids,
    omittedL1Ids
  };
}

function seedFrozenPathsAndEmbeddings(input: {
  sourceDb: Database.Database;
  targetRepos: Repositories;
  episodes: SourceEpisodeRow[];
  embeddingSignature: string;
}): {
  paths: EpisodeExecutionPathRecord[];
  stepCount: number;
  sourceEmbeddingProvider: string;
  sourceEmbeddingModel: string;
  sourceEmbeddingDim: number;
} {
  const sourcePaths = input.sourceDb.prepare(
    `SELECT id, episode_id, user_id, reconstruction_model, source_snapshot_hash,
            terminal_reward, payload_json
     FROM episode_procedural_paths
     WHERE status = 'active' AND reconstruction_algorithm_version = ?`
  ).all(FROZEN_RECONSTRUCTION_VERSION) as SourcePathRow[];
  const sourcePathByEpisode = new Map(sourcePaths.map((path) => [path.episode_id, path]));
  const stepStatement = input.sourceDb.prepare(
    `SELECT o.path_id, o.step_id, o.episode_id, o.step_index, o.raw_turn_id,
            o.intent, o.summary, o.outcome, o.tool_name, o.step_json,
            e.vector_json, e.embedding_dim
     FROM procedural_step_occurrences o
     JOIN procedural_step_embeddings e ON e.occurrence_id = o.id
     WHERE o.path_id = ?
     ORDER BY o.step_index ASC`
  );
  const sourceEmbedding = input.sourceDb.prepare(
    `SELECT embedding_provider AS provider, embedding_model AS model,
            MIN(embedding_dim) AS min_dim, MAX(embedding_dim) AS max_dim
     FROM procedural_step_embeddings e
     JOIN procedural_step_occurrences o ON o.id = e.occurrence_id
     JOIN episode_procedural_paths p ON p.id = o.path_id
     WHERE p.status = 'active' AND p.reconstruction_algorithm_version = ?`
  ).get(FROZEN_RECONSTRUCTION_VERSION) as {
    provider: string;
    model: string;
    min_dim: number;
    max_dim: number;
  };
  if (sourceEmbedding.min_dim !== sourceEmbedding.max_dim) {
    throw new Error("Frozen Step embeddings have inconsistent dimensions");
  }

  const paths: EpisodeExecutionPathRecord[] = [];
  let stepCount = 0;
  for (const [episodeIndex, episodeRow] of input.episodes.entries()) {
    const sourcePath = sourcePathByEpisode.get(episodeRow.id);
    if (!sourcePath) throw new Error(`Frozen path missing: ${episodeRow.id}`);
    const sourceSteps = stepStatement.all(sourcePath.id) as SourceStepRow[];
    const episode = input.targetRepos.runtime.getEpisode(episodeRow.id);
    if (!episode) throw new Error(`Imported Episode missing: ${episodeRow.id}`);
    const governed = governedSourceState(input.targetRepos, episode);
    const activeRawTurnIds = new Set(governed.rawTurns.map((turn) => turn.id));
    const steps = sourceSteps.map((row): ExecutionStepLiteV1 => {
      if (!activeRawTurnIds.has(row.raw_turn_id)) {
        throw new Error(`Frozen Step RawTurn is not governed active evidence: ${row.step_id}`);
      }
      const oldStep = parseRecord(row.step_json);
      const action = parseRecord(oldStep.action);
      const oldOutcome = parseRecord(oldStep.outcome);
      const provenance = parseRecord(oldStep.provenance);
      const evidenceRefs = stringArray(oldOutcome.evidenceRefs).length > 0
        ? stringArray(oldOutcome.evidenceRefs)
        : stringArray(action.eventRefs);
      return {
        id: row.step_id,
        schemaVersion: EXECUTION_STEP_LITE_SCHEMA_VERSION,
        episodeId: row.episode_id,
        rawTurnId: row.raw_turn_id,
        turnIndex: integer(oldStep.turnIndex, 0),
        stepIndex: row.step_index,
        kind: action.kind === "response_generation" ? "response_generation" : "tool_action",
        ...(row.tool_name ? { toolName: row.tool_name } : {}),
        ...(Number.isInteger(action.toolCallIndex)
          ? { toolCallIndex: Number(action.toolCallIndex) }
          : {}),
        intent: row.intent,
        summary: row.summary,
        outcome: row.outcome,
        ...(text(oldOutcome.errorCode) ? { errorCode: text(oldOutcome.errorCode) } : {}),
        ...(text(oldStep.retryOfStepId) ? { retryOfStepId: text(oldStep.retryOfStepId) } : {}),
        ...(text(oldStep.recoveryFromStepId)
          ? { recoveryFromStepId: text(oldStep.recoveryFromStepId) }
          : {}),
        evidenceRefs,
        provenance: {
          algorithmVersion: text(provenance.algorithmVersion) ?? FROZEN_RECONSTRUCTION_VERSION,
          ...(sourcePath.reconstruction_model
            ? { model: sourcePath.reconstruction_model }
            : {}),
          sourceSnapshotHash: governed.sourceSnapshotHash
        }
      };
    });
    const pathBasis = {
      schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
      episodeId: episode.id,
      userId: episode.userId,
      sourceSnapshotHash: governed.sourceSnapshotHash,
      compilerVersion: FROZEN_PATH_COMPILER_VERSION,
      modelSignature: `frozen-v15:${sourcePath.reconstruction_model ?? "unknown"}`,
      sourcePathId: sourcePath.id,
      steps
    };
    const pathHash = stableHash(pathBasis);
    const path: EpisodeExecutionPathLiteV1 = {
      id: `episode_execution_path_${pathHash.slice(0, 20)}`,
      schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
      episodeId: episode.id,
      userId: episode.userId,
      sourceRawTurnIds: governed.rawTurns.map((turn) => turn.id),
      steps,
      turnTransitions: [],
      ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask }),
      sourceSnapshotHash: governed.sourceSnapshotHash,
      pathHash,
      compilerVersion: FROZEN_PATH_COMPILER_VERSION,
      modelSignature: `frozen-v15:${sourcePath.reconstruction_model ?? "unknown"}`,
      provenance: {
        algorithmVersion: FROZEN_PATH_COMPILER_VERSION,
        ...(sourcePath.reconstruction_model ? { model: sourcePath.reconstruction_model } : {}),
        inputCandidateCount: steps.length,
        compiledCandidateCount: steps.length,
        truncated: false
      }
    };
    const createdAt = new Date(Date.parse("2026-08-27T10:00:00.000Z") + episodeIndex).toISOString();
    const saved = input.targetRepos.proceduralTrajectory.savePathVersion({
      path,
      createdAt,
      activate: true
    });
    for (const [index, sourceStep] of sourceSteps.entries()) {
      const step = steps[index]!;
      if (sourceStep.embedding_dim !== sourceEmbedding.min_dim) {
        throw new Error(`Frozen Step embedding dimension mismatch: ${step.id}`);
      }
      input.targetRepos.proceduralTrajectory.upsertStepEmbedding({
        pathId: saved.record.id,
        stepId: step.id,
        stepIndex: step.stepIndex,
        representationVersion: PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION,
        embeddingSignature: input.embeddingSignature,
        semanticHash: stableHash(step.intent),
        vector: numberArray(sourceStep.vector_json),
        createdAt
      });
    }
    paths.push(saved.record);
    stepCount += steps.length;
  }
  return {
    paths,
    stepCount,
    sourceEmbeddingProvider: sourceEmbedding.provider,
    sourceEmbeddingModel: sourceEmbedding.model,
    sourceEmbeddingDim: sourceEmbedding.min_dim
  };
}

function governedSourceState(repos: Repositories, episode: EpisodeRecord): {
  rawTurns: RawTurnRecord[];
  sourceTraceIds: string[];
  sourceSnapshotHash: string;
} {
  const rawTurns = repos.runtime.listRawTurnsByEpisode(episode.id, 10_000);
  const sourceTraces = repos.memories.getMany(episode.l1MemoryIds).flatMap((memory) => {
    if (memory.status !== "activated" && memory.status !== "resolving") return [];
    const trace = traceMetaFromMemory(memory);
    return trace?.episodeId === episode.id && trace.rawTurnId
      ? [{ id: trace.id, rawTurnId: trace.rawTurnId, contentHash: memory.contentHash }]
      : [];
  });
  const authorizedRawTurnIds = new Set(sourceTraces.map((trace) => trace.rawTurnId));
  const eligibleRawTurns = rawTurns.filter((rawTurn) =>
    rawTurn.episodeId === episode.id &&
    rawTurn.userId === episode.userId &&
    !rawTurn.redactedAt &&
    !rawTurn.deletedAt &&
    authorizedRawTurnIds.has(rawTurn.id)
  );
  const sourceSnapshotHash = stableHash({
    schema: "episode_trajectory_governed_source.v1",
    episodeId: episode.id,
    userId: episode.userId,
    sourceTraces: sourceTraces.map((trace) => ({
      id: trace.id,
      rawTurnId: trace.rawTurnId,
      contentHash: trace.contentHash ?? null
    })).sort((left, right) => left.id.localeCompare(right.id)),
    rawTurns: rawTurns.map((rawTurn) => ({
      id: rawTurn.id,
      authorized: authorizedRawTurnIds.has(rawTurn.id),
      redactedAt: rawTurn.redactedAt ?? null,
      deletedAt: rawTurn.deletedAt ?? null,
      status: rawTurn.status,
      contentHash: stableHash({
        userText: rawTurn.userText ?? "",
        assistantText: rawTurn.assistantText ?? "",
        reasoningSummary: rawTurn.reasoningSummary ?? "",
        toolCalls: rawTurn.toolCalls,
        toolResults: rawTurn.toolResults,
        messagePayload: rawTurn.messagePayload ?? {}
      })
    }))
  });
  return {
    rawTurns: eligibleRawTurns,
    sourceTraceIds: sourceTraces.map((trace) => trace.id),
    sourceSnapshotHash
  };
}

function seedWindowJobs(
  repos: Repositories,
  paths: EpisodeExecutionPathRecord[],
  episodes: SourceEpisodeRow[]
): void {
  const episodeById = new Map(episodes.map((episode) => [episode.id, episode]));
  for (const [index, path] of paths.entries()) {
    const episode = episodeById.get(path.episodeId)!;
    enqueueJob(repos, {
      jobType: "trajectory_window_ingest",
      userId: path.userId,
      sessionId: episode.session_id,
      episodeId: path.episodeId,
      targetMemoryId: path.id,
      payload: {
        reason: "kw.frozen-v15.production-cluster",
        pathId: path.id,
        pathHash: path.pathHash,
        pathSourceSnapshotHash: path.sourceSnapshotHash,
        sourceSnapshotHash: path.sourceSnapshotHash,
        frozenStepSemantics: true
      },
      createdAt: new Date(Date.parse("2026-08-27T11:00:00.000Z") + index).toISOString()
    });
  }
}

function enqueueJob(repos: Repositories, input: EnqueueJobInput): EvolutionJobRecord {
  const at = input.createdAt ?? new Date().toISOString();
  return repos.runtime.enqueueJob({
    id: newId("job"),
    jobType: input.jobType,
    status: "queued",
    ...(evolutionJobDedupeKey(input) ? { dedupeKey: evolutionJobDedupeKey(input) } : {}),
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

async function drainWorker(
  service: MemoryService,
  maxRounds: number,
  workerLimit: number
): Promise<DrainResult> {
  const started = Date.now();
  let leased = 0;
  let succeeded = 0;
  let failed = 0;
  for (let round = 1; round <= maxRounds; round += 1) {
    const result = await service.runWorkerOnce(workerLimit);
    leased += result.leased + result.embeddingRetries.leased;
    succeeded += result.succeeded + result.embeddingRetries.succeeded;
    failed += result.failed + result.embeddingRetries.failed;
    if (result.leased === 0 && result.embeddingRetries.leased === 0) {
      return { drained: true, rounds: round, leased, succeeded, failed, elapsedMs: Date.now() - started };
    }
  }
  return {
    drained: false,
    rounds: maxRounds,
    leased,
    succeeded,
    failed,
    elapsedMs: Date.now() - started
  };
}

function buildArtifact(input: {
  startedAt: string;
  elapsedMs: number;
  sourceDbPath: string;
  v15ResultPath: string;
  outputDbPath: string;
  outputReportPath: string;
  config: {
    minSupportEpisodes: number;
    medoidSwitchMargin: number;
    scales: Array<{
      length: number;
      stride: number;
      coarseSimilarityThreshold: number;
      bandWidth: number;
      minStepSimilarity: number;
      minMatchedSteps: number;
      minCoverage: number;
      minAverageMatchSimilarity: number;
      maxInternalGap: number;
      gapPenalty: number;
      minAlignmentScore: number;
    }>;
  };
  imported: ReturnType<typeof importFrozenCohort>;
  frozen: ReturnType<typeof seedFrozenPathsAndEmbeddings>;
  embeddingSignature: string;
  drain: DrainResult;
  targetDb: Database.Database;
  targetRepos: Repositories;
  v15Candidates: V15Candidate[];
}) {
  const clusterDomains = loadActiveClusterDomains(input.targetDb, input.targetRepos);
  const scaleConfig = new Map(input.config.scales.map((scale) => [scale.length, scale]));
  const qualified = clusterDomains.filter((cluster) => {
    const config = scaleConfig.get(cluster.scale);
    return config && cluster.supportEpisodeIds.length >= input.config.minSupportEpisodes &&
      Boolean(extractAlignedCommonCore(cluster, {
        minSupportEpisodes: input.config.minSupportEpisodes,
        minCoreSteps: config.minMatchedSteps
      }));
  });
  const maximal = selectMaximalWindowClusters(qualified);
  const unsuppressed = maximal.filter((cluster) => !cluster.suppressedByClusterId);
  const skillMemories = input.targetRepos.memories.list({
    memoryLayer: "Skill",
    status: ["activated", "resolving", "archived", "deleted"]
  }, 10_000).filter((memory) =>
    memory.properties.internal_info.plugin_algorithm === PROCEDURAL_SKILL_ALGORITHM
  );
  const oldCandidateDiagnostics = input.v15Candidates.map((candidate) =>
    diagnoseV15Candidate(candidate, input.targetRepos, clusterDomains, scaleConfig)
  );
  const windowsByScale = groupedScalar(input.targetDb,
    "SELECT scale AS key, COUNT(*) AS count FROM trajectory_window_occurrences GROUP BY scale");
  const clustersByScale = groupedScalar(input.targetDb,
    `SELECT c.scale AS key, COUNT(*) AS count
     FROM trajectory_window_clusters c WHERE c.status = 'active' GROUP BY c.scale`);
  const supportDistribution = groupedScalar(input.targetDb,
    `SELECT v.support_episode_count AS key, COUNT(*) AS count
     FROM trajectory_window_clusters c
     JOIN trajectory_window_cluster_versions v ON v.id = c.active_version_id
     WHERE c.status = 'active'
     GROUP BY v.support_episode_count`);
  return {
    schemaVersion: "kw-frozen-v15-production-replay.v1",
    startedAt: input.startedAt,
    finishedAt: new Date().toISOString(),
    elapsedMs: input.elapsedMs,
    isolation: {
      sourceDatabase: input.sourceDbPath,
      v15Result: input.v15ResultPath,
      outputDatabase: input.outputDbPath,
      outputReport: input.outputReportPath,
      sourceOpenedReadOnly: true,
      targetCreatedFresh: true,
      stepSemanticsLlmCalled: false,
      stepEmbeddingsReusedExactly: true,
      remoteCalls: ["coarse intent-sequence embeddings", "qualified direct Skill induction"]
    },
    frozenInput: {
      reconstructionVersion: FROZEN_RECONSTRUCTION_VERSION,
      reconstructionModel: "deepseek-v4-flash",
      episodes: input.imported.episodes.length,
      rawTurns: input.imported.rawTurnIds.length,
      l1: input.imported.l1Ids.length,
      omittedL1: input.imported.omittedL1Ids.length,
      paths: input.frozen.paths.length,
      steps: input.frozen.stepCount,
      sourceEmbeddingProvider: input.frozen.sourceEmbeddingProvider,
      sourceEmbeddingModel: input.frozen.sourceEmbeddingModel,
      sourceEmbeddingDim: input.frozen.sourceEmbeddingDim,
      productionEmbeddingSignature: input.embeddingSignature
    },
    productionParameters: input.config,
    drain: input.drain,
    jobs: jobCounts(input.targetDb),
    summary: {
      windows: scalar(input.targetDb, "SELECT COUNT(*) AS count FROM trajectory_window_occurrences"),
      windowsByScale,
      activeClusters: clusterDomains.length,
      clustersByScale,
      supportDistribution,
      supportedClusters: clusterDomains.filter((cluster) =>
        cluster.supportEpisodeIds.length >= input.config.minSupportEpisodes).length,
      commonCoreQualifiedClusters: qualified.length,
      maximalClusters: unsuppressed.length,
      suppressedClusters: maximal.length - unsuppressed.length,
      skills: skillMemories.length
    },
    maximalClusters: unsuppressed.map((cluster) => ({
      id: cluster.id,
      scale: cluster.scale,
      supportEpisodeIds: cluster.supportEpisodeIds,
      members: cluster.members.map((member) => ({
        occurrenceId: member.occurrence.id,
        episodeId: member.occurrence.episodeId,
        startStepIndex: member.occurrence.startStepIndex,
        endStepIndex: member.occurrence.endStepIndex,
        alignmentScore: member.alignmentToMedoid.score,
        averageMatchSimilarity: member.alignmentToMedoid.averageMatchSimilarity,
        matchedSteps: member.alignmentToMedoid.matchedSteps
      }))
    })),
    skills: skillMemories.map((memory) => publicSkill(memory)),
    v15CandidateDiagnostics: oldCandidateDiagnostics
  };
}

function loadActiveClusterDomains(
  db: Database.Database,
  repos: Repositories
): TrajectoryWindowClusterV1[] {
  const heads = db.prepare(
    `SELECT id, scale, active_version_id FROM trajectory_window_clusters
     WHERE status = 'active' AND active_version_id IS NOT NULL`
  ).all() as Array<{ id: string; scale: number; active_version_id: string }>;
  return heads.flatMap((head) => {
    const version = repos.proceduralTrajectory.getClusterVersion(head.active_version_id);
    if (!version) return [];
    const rows = repos.proceduralTrajectory.listClusterMembers(version.id);
    const members = rows.flatMap((row) => {
      const window = repos.proceduralTrajectory.getWindow(row.occurrenceId);
      if (!window) return [];
      const alignment = row.alignment as unknown as TrajectoryWindowClusterV1["members"][number]["alignmentToMedoid"];
      return [{
        occurrence: windowOccurrence(window, repos),
        similarityToMedoid: alignment.score,
        alignmentToMedoid: alignment
      }];
    });
    if (members.length === 0) return [];
    const episodeIds = distinct(members.map((member) => member.occurrence.episodeId));
    const supportEpisodeIds = distinct(members.filter((member) =>
      member.occurrence.evidenceRole === "support").map((member) => member.occurrence.episodeId));
    const counterexampleEpisodeIds = distinct(members.filter((member) =>
      member.occurrence.evidenceRole === "counterexample").map((member) => member.occurrence.episodeId));
    const unknownEpisodeIds = distinct(members.filter((member) =>
      member.occurrence.evidenceRole === "unknown").map((member) => member.occurrence.episodeId));
    const similarities = members.map((member) => member.alignmentToMedoid.score);
    return [{
      id: head.id,
      familyId: head.id,
      scale: head.scale,
      medoidOccurrenceId: version.medoidOccurrenceId,
      episodeIds,
      supportEpisodeIds,
      counterexampleEpisodeIds,
      unknownEpisodeIds,
      occurrenceCount: members.length,
      averageSimilarity: average(similarities),
      minimumSimilarity: Math.min(...similarities),
      medoidCentrality: numeric(version.metrics.medoidCentrality, average(similarities)),
      medoidUpdateCount: numeric(version.metrics.medoidUpdateCount, 0),
      members
    }];
  });
}

function diagnoseV15Candidate(
  candidate: V15Candidate,
  repos: Repositories,
  clusterDomains: TrajectoryWindowClusterV1[],
  scaleConfig: Map<number, {
    length: number;
    coarseSimilarityThreshold: number;
    bandWidth: number;
    minStepSimilarity: number;
    minMatchedSteps: number;
    minCoverage: number;
    minAverageMatchSimilarity: number;
    maxInternalGap: number;
    gapPenalty: number;
    minAlignmentScore: number;
  }>
) {
  const windows = candidate.members.map((member) => {
    const path = repos.proceduralTrajectory.getActivePath(member.episodeId);
    return path
      ? repos.proceduralTrajectory.listWindowsForPath(path.id, member.scale).find((window) =>
          window.startStepIndex === member.startStepIndex &&
          window.endStepIndex === member.endStepIndex)
      : undefined;
  });
  const foundWindows = windows.filter((window): window is NonNullable<typeof window> => Boolean(window));
  const memberIds = new Set(foundWindows.map((window) => window.id));
  const reproduced = clusterDomains.find((cluster) =>
    cluster.scale === candidate.scale &&
    foundWindows.length === candidate.members.length &&
    foundWindows.every((window) => cluster.members.some((member) =>
      member.occurrence.id === window.id))
  );
  const config = scaleConfig.get(candidate.scale);
  let directComparison: Record<string, unknown> | undefined;
  if (foundWindows.length === 2 && config) {
    const [left, right] = foundWindows;
    const leftPath = repos.proceduralTrajectory.getPath(left!.pathId)!;
    const rightPath = repos.proceduralTrajectory.getPath(right!.pathId)!;
    const stepVector = (path: EpisodeExecutionPathRecord, stepId: string) => {
      const embedding = repos.proceduralTrajectory.listStepEmbeddings({
        pathId: path.id,
        representationVersion: PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION
      }).find((item) => item.stepId === stepId);
      if (!embedding) throw new Error(`Frozen Step embedding missing during diagnostics: ${stepId}`);
      return embedding.vector;
    };
    const fineConfig: BandedMonotonicMatchConfig = {
      scale: config.length,
      bandWidth: config.bandWidth,
      minStepSimilarity: config.minStepSimilarity,
      minMatchedSteps: config.minMatchedSteps,
      minCoverage: config.minCoverage,
      minAverageMatchSimilarity: config.minAverageMatchSimilarity,
      maxInternalGap: config.maxInternalGap,
      gapPenalty: config.gapPenalty,
      minAlignmentScore: config.minAlignmentScore
    };
    const fine = bandedMonotonicMatch(
      left!.stepIds.map((stepId) => stepVector(leftPath, stepId)),
      right!.stepIds.map((stepId) => stepVector(rightPath, stepId)),
      fineConfig
    );
    const coarse = cosineSimilarity(left!.coarseVector, right!.coarseVector);
    directComparison = {
      coarseSimilarity: coarse,
      coarseThreshold: config.coarseSimilarityThreshold,
      coarseAdmitted: coarse + Number.EPSILON >= config.coarseSimilarityThreshold,
      fine
    };
  }
  return {
    v15ClusterId: candidate.clusterId,
    scale: candidate.scale,
    expectedMembers: candidate.members,
    foundWindowIds: foundWindows.map((window) => window.id),
    allWindowsFound: foundWindows.length === candidate.members.length,
    reproducedAsOneCluster: Boolean(reproduced),
    reproducedClusterId: reproduced?.id,
    currentClusterIdsByWindow: foundWindows.map((window) => ({
      occurrenceId: window.id,
      clusterIds: clusterDomains.filter((cluster) => cluster.members.some((member) =>
        member.occurrence.id === window.id)).map((cluster) => cluster.id)
    })),
    directComparison,
    ignoredMemberIds: [...memberIds].filter((id) =>
      !clusterDomains.some((cluster) => cluster.members.some((member) =>
        member.occurrence.id === id)))
  };
}

function windowOccurrence(
  window: TrajectoryWindowOccurrenceRecord,
  repos: Repositories
): TrajectoryWindowClusterV1["members"][number]["occurrence"] {
  const path = repos.proceduralTrajectory.getPath(window.pathId);
  if (!path) throw new Error(`Window path not found: ${window.pathId}`);
  const stepById = new Map(path.path.steps.map((step) => [step.id, step]));
  const steps = window.stepIds.map((stepId) => stepById.get(stepId));
  if (steps.some((step) => !step)) {
    throw new Error(`Window references missing frozen Step: ${window.id}`);
  }
  return {
    id: window.id,
    schemaVersion: window.schemaVersion,
    episodeId: window.episodeId,
    pathId: window.pathId,
    userId: window.userId,
    ...(window.terminalReward === undefined ? {} : { terminalReward: window.terminalReward }),
    evidenceRole: window.evidenceRole,
    scale: window.scale,
    stride: window.stride,
    startStepIndex: window.startStepIndex,
    endStepIndex: window.endStepIndex,
    semanticText: window.semanticText,
    steps: steps as ExecutionStepLiteV1[]
  };
}

function readV15Candidates(path: string): V15Candidate[] {
  const artifact = JSON.parse(readFileSync(path, "utf8")) as {
    result?: {
      candidateClusterIds?: unknown;
      clusters?: unknown;
    };
  };
  const result = artifact.result;
  if (!result || !Array.isArray(result.candidateClusterIds) || !Array.isArray(result.clusters)) {
    throw new Error("Invalid v15 result artifact");
  }
  const candidateIds = new Set(result.candidateClusterIds.filter((id): id is string =>
    typeof id === "string"));
  return result.clusters.flatMap((value): V15Candidate[] => {
    const cluster = parseRecord(value);
    const clusterId = text(cluster.id);
    if (!clusterId || !candidateIds.has(clusterId) || !Array.isArray(cluster.members)) return [];
    const scale = integer(cluster.scale, 0);
    const members = cluster.members.flatMap((memberValue): V15CandidateMember[] => {
      const member = parseRecord(memberValue);
      const occurrence = parseRecord(member.occurrence);
      const episodeId = text(occurrence.episodeId);
      if (!episodeId) return [];
      return [{
        episodeId,
        scale: integer(occurrence.scale, scale),
        startStepIndex: integer(occurrence.startStepIndex, -1),
        endStepIndex: integer(occurrence.endStepIndex, -1)
      }];
    });
    return [{ clusterId, scale, members }];
  });
}

function publicSkill(memory: MemoryRow): Record<string, unknown> {
  const skill = parseRecord(memory.properties.internal_info.skill);
  return {
    id: memory.id,
    memoryKey: memory.memoryKey,
    memoryStatus: memory.status,
    name: skill.name,
    lifecycleStatus: skill.status,
    support: skill.support,
    sourceEpisodeIds: skill.source_episode_ids,
    sourceSpanOccurrenceIds: skill.source_span_occurrence_ids
  };
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

function jobCounts(db: Database.Database): Record<string, Record<string, number>> {
  const rows = db.prepare(
    `SELECT job_type AS jobType, status, COUNT(*) AS count
     FROM evolution_jobs GROUP BY job_type, status ORDER BY job_type, status`
  ).all() as Array<{ jobType: string; status: string; count: number }>;
  const result: Record<string, Record<string, number>> = {};
  for (const row of rows) {
    result[row.jobType] ??= {};
    result[row.jobType]![row.status] = row.count;
  }
  return result;
}

function scalar(db: Database.Database, sql: string): number {
  const row = db.prepare(sql).get() as { count: number };
  return Number(row.count);
}

function groupedScalar(db: Database.Database, sql: string): Record<string, number> {
  const rows = db.prepare(sql).all() as Array<{ key: string | number; count: number }>;
  return Object.fromEntries(rows.map((row) => [String(row.key), Number(row.count)]));
}

function selectRowsByValues(
  db: Database.Database,
  table: string,
  column: string,
  values: readonly string[]
): SqlRow[] {
  if (values.length === 0) return [];
  assertIdentifier(table);
  assertIdentifier(column);
  return db.prepare(
    `SELECT * FROM "${table}" WHERE "${column}" IN (${values.map(() => "?").join(", ")})`
  ).all(...values) as SqlRow[];
}

function insertRows(db: Database.Database, table: string, rows: readonly SqlRow[]): void {
  if (rows.length === 0) return;
  assertIdentifier(table);
  const targetColumns = new Set((db.prepare(`PRAGMA table_info("${table}")`).all() as Array<{
    name: string;
  }>).map((column) => column.name));
  const columns = Object.keys(rows[0]!).filter((column) => targetColumns.has(column));
  const statement = db.prepare(
    `INSERT INTO "${table}" (${columns.map((column) => `"${column}"`).join(", ")}) ` +
    `VALUES (${columns.map(() => "?").join(", ")})`
  );
  for (const row of rows) statement.run(...columns.map((column) => row[column] ?? null));
}

function parseArgs(argv: readonly string[]): Args {
  const args: Args = {
    help: false,
    maxRounds: 200,
    workerLimit: 100,
    confirmRemoteModels: false,
    prepareOnly: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") args.help = true;
    else if (arg === "--source-db") args.sourceDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--v15-result") args.v15ResultPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output-db") args.outputDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output") args.outputReportPath = requiredValue(argv, ++index, arg);
    else if (arg === "--config") args.configPath = requiredValue(argv, ++index, arg);
    else if (arg === "--max-rounds") args.maxRounds = positiveInt(requiredValue(argv, ++index, arg), arg);
    else if (arg === "--worker-limit") args.workerLimit = positiveInt(requiredValue(argv, ++index, arg), arg);
    else if (arg === "--confirm-remote-models") args.confirmRemoteModels = true;
    else if (arg === "--prepare-only") args.prepareOnly = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function helpText(): string {
  return [
    "Usage:",
    "  node --import tsx scripts/kw-frozen-v15-production-replay.ts",
    "    --source-db <v15.sqlite> --v15-result <v15.json>",
    "    --output-db <fresh.sqlite> [--output <report.json>] [--config <config.yaml>]",
    "    [--max-rounds 200] [--worker-limit 100] --confirm-remote-models",
    "    [--prepare-only]",
    "",
    "The harness copies the 18 reconstruction.v8 Episodes, adapts their frozen 318 Step",
    "semantics into the production lightweight Path schema, imports the exact persisted",
    "Step vectors, and starts at trajectory_window_ingest. It never calls Step Semantics LLM.",
    "Production code generates coarse Window embeddings, incrementally clusters, applies",
    "maximal suppression, and invokes the real direct-Skill LLM for qualified clusters.",
    ""
  ].join("\n");
}

function requiredValue(argv: readonly string[], index: number, option: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${option} requires a value`);
  return value;
}

function requiredPath(value: string | undefined, option: string): string {
  if (!value) throw new Error(`${option} is required`);
  return resolve(value);
}

function positiveInt(value: string, option: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${option} must be positive`);
  return parsed;
}

function parseStringArray(value: string): string[] {
  try {
    return stringArray(JSON.parse(value));
  } catch {
    return [];
  }
}

function numberArray(value: string): number[] {
  const parsed = JSON.parse(value) as unknown;
  if (!Array.isArray(parsed) || parsed.some((item) => typeof item !== "number")) {
    throw new Error("Invalid frozen embedding vector");
  }
  return parsed;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function parseRecord(value: unknown): Record<string, unknown> {
  if (typeof value === "string") {
    try {
      return parseRecord(JSON.parse(value));
    } catch {
      return {};
    }
  }
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function integer(value: unknown, fallback: number): number {
  return Number.isInteger(value) ? Number(value) : fallback;
}

function numeric(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function average(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function distinct(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function assertIdentifier(value: string): void {
  if (!/^[a-z_][a-z0-9_]*$/i.test(value)) throw new Error(`Unsafe SQLite identifier: ${value}`);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
