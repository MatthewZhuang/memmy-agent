import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import Database from "better-sqlite3";
import {
  MemoryDb,
  MemoryService,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type JobType,
  type MemoryRow
} from "../src/index.js";
import {
  skillMetaFromMemory,
  traceMetaFromMemory
} from "../src/algorithm/plugin-algorithms.js";
import {
  Repositories,
  type EvolutionJobRecord
} from "../src/storage/repositories.js";
import { SCHEMA_VERSION } from "../src/storage/schema.js";
import { EPISODE_PATH_COMPILER_VERSION } from "../src/service/evolution/episode-path-compiler.js";
import {
  evolutionJobDedupeKey,
  type EnqueueJobInput
} from "../src/service/worker/job-handlers.js";
import { newId, stableHash } from "../src/utils/id.js";

type ReplayChain = "old" | "new" | "both";

interface ReplayArgs {
  help: boolean;
  sourceDbPath?: string;
  configPath?: string;
  outputDbPath?: string;
  outputReportPath?: string;
  episodeIds: string[];
  cohortSourcePath?: string;
  chain: ReplayChain;
  maxEpisodes?: number;
  maxRounds: number;
  workerLimit: number;
  confirmRemoteModels: boolean;
}

interface SqlRow {
  [key: string]: unknown;
}

interface ImportedEpisodeRow extends SqlRow {
  id: string;
  session_id: string;
  user_id: string;
  status: string;
  l1_memory_ids_json: string;
  raw_turn_ids_json: string;
  r_task: number | null;
  opened_at: string;
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

const PROCEDURAL_SKILL_ALGORITHM = "procedural.pattern.skill.v1";
const UPSTREAM_SKILL_ALGORITHM = "skill.crystallization.v7";
const CAPTURE_L1_ALGORITHM = "capture.v7";

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(helpText());
    return;
  }
  const sourceDbPath = requiredResolvedPath(args.sourceDbPath, "--db");
  const outputDbPath = requiredResolvedPath(args.outputDbPath, "--output-db");
  const outputReportPath = resolve(
    args.outputReportPath ?? `${outputDbPath}.report.json`
  );
  const configPath = args.configPath ? resolve(args.configPath) : undefined;
  const cohortSourcePath = args.cohortSourcePath
    ? resolve(args.cohortSourcePath)
    : undefined;

  if (!args.confirmRemoteModels) {
    throw new Error(
      "This replay calls the configured summary/evolution LLM and embedding model. " +
      "Re-run with --confirm-remote-models after checking the isolated output paths."
    );
  }
  if (!existsSync(sourceDbPath)) throw new Error(`KW source database not found: ${sourceDbPath}`);
  if (sourceDbPath === outputDbPath) throw new Error("--output-db must differ from --db");
  if (existsSync(outputDbPath)) throw new Error(`Refusing to overwrite output DB: ${outputDbPath}`);
  if (existsSync(outputReportPath)) {
    throw new Error(`Refusing to overwrite output report: ${outputReportPath}`);
  }

  const cohort = resolveCohort(args.episodeIds, cohortSourcePath);
  const loaded = loadMemmyConfig(configPath);
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
    const imported = importReplayCohort({
      sourceDb: sourceDb.db,
      sourceRepos,
      targetDb: targetDb.db,
      targetRepos,
      episodeIds: cohort.episodeIds,
      maxEpisodes: args.maxEpisodes
    });
    sourceDb.db.exec("ROLLBACK");

    const seeded = seedReplayJobs({
      repos: targetRepos,
      episodes: imported.episodes,
      importedL1Ids: imported.importedL1Ids,
      chain: args.chain,
      minTraceValue: loaded.config.algorithm.l2Induction.minTraceValue
    });

    service = new MemoryService({
      db: targetDb,
      mode: "local",
      config: loaded.config,
      ...(configPath ? { configPath } : {})
    });
    const drain = await drainWorker(service, targetDb.db, args.maxRounds, args.workerLimit);
    const artifact = buildArtifact({
      startedAt,
      elapsedMs: Date.now() - wallStart,
      sourceDbPath,
      outputDbPath,
      outputReportPath,
      cohortSourcePath,
      requestedEpisodeIds: cohort.episodeIds,
      imported,
      seeded,
      drain,
      chain: args.chain,
      config: loaded.config,
      targetDb: targetDb.db,
      targetRepos
    });
    writeFileSync(outputReportPath, `${JSON.stringify(artifact, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600
    });
    process.stdout.write(`${JSON.stringify({
      outputDatabase: outputDbPath,
      outputReport: outputReportPath,
      cohortSource: cohortSourcePath ?? "explicit/default-all-closed",
      episodes: imported.episodes.length,
      oldSkills: artifact.skills.old.total,
      newSkills: artifact.skills.new.total,
      drain
    }, null, 2)}\n`);
    if (!drain.drained) process.exitCode = 2;
  } finally {
    if (sourceDb.db.inTransaction) sourceDb.db.exec("ROLLBACK");
    // MemoryService does not own/close the injected DB. Close both handles here.
    void service;
    targetDb.close();
    sourceDb.close();
  }
}

function importReplayCohort(input: {
  sourceDb: Database.Database;
  sourceRepos: Repositories;
  targetDb: Database.Database;
  targetRepos: Repositories;
  episodeIds: readonly string[];
  maxEpisodes?: number;
}): {
  episodes: ImportedEpisodeRow[];
  importedSessionIds: string[];
  importedRawTurnIds: string[];
  importedL1Ids: string[];
  omittedL1Ids: string[];
} {
  let episodes = selectEpisodeRows(input.sourceDb, input.episodeIds);
  if (input.maxEpisodes !== undefined) episodes = episodes.slice(0, input.maxEpisodes);
  if (episodes.length === 0) throw new Error("The selected KW cohort contains no closed Episodes");

  const episodeIds = episodes.map((episode) => episode.id);
  const sessionIds = unique(episodes.map((episode) => episode.session_id));
  const sourceL1Ids = unique(episodes.flatMap((episode) =>
    parseStringArray(episode.l1_memory_ids_json)
  ));
  const sourceMemories = input.sourceRepos.memories.getMany(sourceL1Ids)
    .filter((memory) =>
      memory.memoryLayer === "L1" &&
      memory.status === "activated" &&
      memory.properties.internal_info.plugin_algorithm === CAPTURE_L1_ALGORITHM
    );
  const importedL1Ids = sourceMemories.map((memory) => memory.id);
  const importedL1Set = new Set(importedL1Ids);
  const omittedL1Ids = sourceL1Ids.filter((id) => !importedL1Set.has(id));
  const normalizedEpisodes = episodes.map((episode) => ({
    ...episode,
    l1_memory_ids_json: JSON.stringify(
      parseStringArray(episode.l1_memory_ids_json).filter((id) => importedL1Set.has(id))
    ),
    l2_policy_ids_json: "[]",
    l3_world_model_ids_json: "[]",
    skill_memory_ids_json: "[]",
    // Feedback and DecisionRepair rows are deliberately outside this replay's
    // import closure. Their already-materialized terminal reward remains in
    // r_task/reward_detail_json; clear dangling derived references.
    feedback_ids_json: "[]",
    decision_repair_ids_json: "[]",
    pipeline_run_id: null,
    pipeline_status: "idle",
    pipeline_error: null
  }));
  const sessions = selectRowsByValues(input.sourceDb, "sessions", "id", sessionIds);
  const rawTurns = selectRowsByValues(input.sourceDb, "raw_turns", "episode_id", episodeIds);

  input.targetDb.transaction(() => {
    insertRows(input.targetDb, "sessions", sessions);
    insertRows(input.targetDb, "episodes", normalizedEpisodes);
    insertRows(input.targetDb, "raw_turns", rawTurns);
    for (const memory of sourceMemories) input.targetRepos.memories.insert(memory);
  })();

  return {
    episodes: normalizedEpisodes,
    importedSessionIds: sessionIds,
    importedRawTurnIds: rawTurns.flatMap((row) => typeof row.id === "string" ? [row.id] : []),
    importedL1Ids,
    omittedL1Ids
  };
}

function seedReplayJobs(input: {
  repos: Repositories;
  episodes: readonly ImportedEpisodeRow[];
  importedL1Ids: readonly string[];
  chain: ReplayChain;
  minTraceValue: number;
}): {
  oldL2Jobs: number;
  newPathJobs: number;
  oldEligibleTraceIds: string[];
} {
  const episodeIds = new Set(input.episodes.map((episode) => episode.id));
  const traces = input.repos.memories.getMany([...input.importedL1Ids])
    .map((memory) => ({ memory, trace: traceMetaFromMemory(memory) }))
    .filter((item): item is { memory: MemoryRow; trace: NonNullable<ReturnType<typeof traceMetaFromMemory>> } =>
      Boolean(item.trace?.episodeId && episodeIds.has(item.trace.episodeId))
    )
    .sort((left, right) => left.trace.ts - right.trace.ts || left.memory.id.localeCompare(right.memory.id));
  const eligible = traces.filter(({ memory, trace }) =>
    memory.status === "activated" &&
    memory.properties.internal_info.policy_eligible !== false &&
    memory.properties.internal_info.evidence_status !== "provisional" &&
    memory.properties.internal_info.evidence_status !== "disputed" &&
    trace.value >= input.minTraceValue &&
    Boolean(trace.vecSummary ?? trace.vecAction)
  );

  let oldL2Jobs = 0;
  let newPathJobs = 0;
  if (input.chain === "old" || input.chain === "both") {
    for (const { memory, trace } of eligible) {
      enqueueJob(input.repos, {
        jobType: "l2_induction",
        userId: memory.userId,
        sessionId: memory.sessionId,
        episodeId: trace.episodeId,
        targetMemoryId: memory.id,
        payload: {
          reason: "kw.dual-skill-replay.old-chain",
          sourceMemoryId: memory.id,
          episodeTraceIds: [memory.id]
        }
      });
      oldL2Jobs += 1;
    }
  }
  if (input.chain === "new" || input.chain === "both") {
    for (const episode of input.episodes) {
      enqueueJob(input.repos, {
        jobType: "episode_path_compile",
        userId: episode.user_id,
        sessionId: episode.session_id,
        episodeId: episode.id,
        payload: {
          reason: "kw.dual-skill-replay.new-chain",
          episodeId: episode.id,
          semanticsVersion: EPISODE_PATH_COMPILER_VERSION,
          rewardSnapshotHash: stableHash({
            schema: "kw-replay-reward.v1",
            episodeId: episode.id,
            rTask: episode.r_task
          })
        }
      });
      newPathJobs += 1;
    }
  }
  return {
    oldL2Jobs,
    newPathJobs,
    oldEligibleTraceIds: eligible.map(({ memory }) => memory.id)
  };
}

function enqueueJob(repos: Repositories, input: EnqueueJobInput): EvolutionJobRecord {
  const at = new Date().toISOString();
  const record: EvolutionJobRecord = {
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
    createdAt: input.createdAt ?? at,
    updatedAt: input.createdAt ?? at
  };
  return repos.runtime.enqueueJob(record);
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
  const remainingByStatus = unfinishedJobCounts(db);
  return {
    drained: false,
    rounds: maxRounds,
    leased,
    succeeded,
    failed,
    elapsedMs: Date.now() - startedAt,
    remainingByStatus
  };
}

function buildArtifact(input: {
  startedAt: string;
  elapsedMs: number;
  sourceDbPath: string;
  outputDbPath: string;
  outputReportPath: string;
  cohortSourcePath?: string;
  requestedEpisodeIds: readonly string[];
  imported: ReturnType<typeof importReplayCohort>;
  seeded: ReturnType<typeof seedReplayJobs>;
  drain: DrainResult;
  chain: ReplayChain;
  config: ReturnType<typeof loadMemmyConfig>["config"];
  targetDb: Database.Database;
  targetRepos: Repositories;
}) {
  const evolution = resolveEvolutionConfig(input.config);
  const skillMemories = input.targetRepos.memories.list({
    memoryLayer: "Skill",
    status: ["activated", "resolving", "archived", "deleted"]
  }, 10_000);
  const skills = skillMemories.map((memory) => {
    const meta = skillMetaFromMemory(memory);
    const algorithm = typeof memory.properties.internal_info.plugin_algorithm === "string"
      ? memory.properties.internal_info.plugin_algorithm
      : "unknown";
    return {
      id: memory.id,
      memoryKey: memory.memoryKey,
      name: meta?.name ?? memory.memoryKey ?? memory.id,
      memoryStatus: memory.status,
      lifecycleStatus: meta?.status ?? "unknown",
      pluginAlgorithm: algorithm,
      support: meta?.support ?? 0,
      sourcePolicyIds: meta?.sourcePolicyIds ?? []
    };
  });
  const newSkills = skills.filter((skill) => skill.pluginAlgorithm === PROCEDURAL_SKILL_ALGORITHM);
  const oldSkills = skills.filter((skill) => skill.pluginAlgorithm === UPSTREAM_SKILL_ALGORITHM);
  const unknownSkills = skills.filter((skill) =>
    skill.pluginAlgorithm !== PROCEDURAL_SKILL_ALGORITHM &&
    skill.pluginAlgorithm !== UPSTREAM_SKILL_ALGORITHM
  );
  const summarizeSkills = (rows: typeof skills) => ({
    total: rows.length,
    active: rows.filter((row) => row.memoryStatus === "activated").length,
    resolving: rows.filter((row) => row.memoryStatus === "resolving").length,
    archived: rows.filter((row) => row.memoryStatus === "archived").length,
    items: rows
  });

  return {
    schemaVersion: "kw-dual-skill-replay.v2",
    startedAt: input.startedAt,
    finishedAt: new Date().toISOString(),
    elapsedMs: input.elapsedMs,
    chain: input.chain,
    isolation: {
      sourceDatabase: input.sourceDbPath,
      outputDatabase: input.outputDbPath,
      outputReport: input.outputReportPath,
      sourceOpenedReadOnly: true,
      targetCreatedAsFreshSchemaVersion: SCHEMA_VERSION,
      importedLayers: ["sessions", "episodes", "raw_turns", "L1"],
      importedL1Contract: "status=activated AND plugin_algorithm=capture.v7",
      clearedEpisodeReferences: ["feedback_ids", "decision_repair_ids", "l2_policy_ids", "l3_world_model_ids", "skill_memory_ids"],
      preservedRewardFields: ["r_task", "reward_detail_json"],
      excludedHistoricalLayers: ["L2", "L3", "Skill", "span.big_turn.v1 L1", "old Span/Policy/Skill jobs"]
    },
    cohort: {
      source: input.cohortSourcePath ?? "explicit episode IDs or all closed Episodes",
      requestedEpisodeIds: input.requestedEpisodeIds,
      importedEpisodeIds: input.imported.episodes.map((episode) => episode.id),
      importedEpisodes: input.imported.episodes.length,
      importedSessions: input.imported.importedSessionIds.length,
      importedRawTurns: input.imported.importedRawTurnIds.length,
      importedL1: input.imported.importedL1Ids.length,
      omittedDeletedOrMissingL1: input.imported.omittedL1Ids.length,
      rewardDistribution: rewardDistribution(input.imported.episodes)
    },
    models: {
      summary: publicModelConfig(input.config.summary),
      evolution: publicModelConfig(evolution),
      embedding: publicModelConfig(input.config.embedding)
    },
    parameters: {
      oldL2: input.config.algorithm.l2Induction,
      proceduralWindow: input.config.algorithm.proceduralWindow
    },
    seeded: input.seeded,
    drain: input.drain,
    jobs: jobCounts(input.targetDb),
    oldChain: {
      policies: memoryLayerCounts(input.targetDb, "L2"),
      candidatePool: scalarCount(input.targetDb, "l2_candidate_pool"),
      tracePolicyLinks: scalarCount(input.targetDb, "trace_policy_links")
    },
    newChain: {
      executionPaths: scalarCount(input.targetDb, "episode_execution_paths"),
      stepEmbeddings: scalarCount(input.targetDb, "procedural_step_embeddings"),
      windows: scalarCount(input.targetDb, "trajectory_window_occurrences"),
      families: groupedCount(input.targetDb, "trajectory_window_families", "status"),
      familyRevisions: scalarCount(input.targetDb, "trajectory_window_family_revisions"),
      familyMembers: scalarCount(input.targetDb, "trajectory_window_family_members"),
      clusters: groupedCount(input.targetDb, "trajectory_window_clusters", "status"),
      clusterVersions: scalarCount(input.targetDb, "trajectory_window_cluster_versions"),
      clusterMembers: scalarCount(input.targetDb, "trajectory_window_cluster_members"),
      canonicalClusterKeys: scalarCount(
        input.targetDb,
        "trajectory_window_cluster_canonical_keys"
      ),
      familyClusterLinks: scalarCount(
        input.targetDb,
        "trajectory_window_family_cluster_links"
      ),
      skillVersions: scalarCount(input.targetDb, "trajectory_skill_versions")
    },
    skills: {
      old: summarizeSkills(oldSkills),
      new: summarizeSkills(newSkills),
      unknown: summarizeSkills(unknownSkills),
      total: skills.length
    }
  };
}

function selectEpisodeRows(db: Database.Database, episodeIds: readonly string[]): ImportedEpisodeRow[] {
  const where = episodeIds.length > 0
    ? `status = 'closed' AND id IN (${episodeIds.map(() => "?").join(", ")})`
    : "status = 'closed'";
  const rows = db.prepare(
    `SELECT * FROM episodes WHERE ${where} ORDER BY opened_at ASC, id ASC`
  ).all(...episodeIds) as ImportedEpisodeRow[];
  if (episodeIds.length > 0) {
    const found = new Set(rows.map((row) => row.id));
    const missing = episodeIds.filter((id) => !found.has(id));
    if (missing.length > 0) {
      throw new Error(`Closed Episode IDs missing from KW DB: ${missing.join(", ")}`);
    }
  }
  return rows;
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

function resolveCohort(
  explicitEpisodeIds: readonly string[],
  cohortSourcePath: string | undefined
): { episodeIds: string[] } {
  if (!cohortSourcePath) return { episodeIds: unique(explicitEpisodeIds) };
  if (!existsSync(cohortSourcePath)) {
    throw new Error(`Cohort source JSON not found: ${cohortSourcePath}`);
  }
  const parsed = JSON.parse(readFileSync(cohortSourcePath, "utf8")) as {
    episodeIds?: unknown;
  };
  const fromFile = Array.isArray(parsed.episodeIds)
    ? parsed.episodeIds.filter((value): value is string => typeof value === "string" && value.length > 0)
    : [];
  if (fromFile.length === 0) throw new Error("--cohort-source JSON has no episodeIds array");
  if (explicitEpisodeIds.length > 0) {
    throw new Error("Use either --episode-ids or --cohort-source, not both");
  }
  return { episodeIds: unique(fromFile) };
}

function rewardDistribution(episodes: readonly ImportedEpisodeRow[]) {
  return {
    positive: episodes.filter((episode) => Number(episode.r_task) > 0).length,
    zero: episodes.filter((episode) => Number(episode.r_task) === 0).length,
    negative: episodes.filter((episode) => Number(episode.r_task) < 0).length,
    missing: episodes.filter((episode) => episode.r_task === null).length
  };
}

function jobCounts(db: Database.Database): Record<string, Record<string, number>> {
  const rows = db.prepare(
    `SELECT job_type AS jobType, status, COUNT(*) AS count
     FROM evolution_jobs
     GROUP BY job_type, status
     ORDER BY job_type, status`
  ).all() as Array<{ jobType: string; status: string; count: number }>;
  const out: Record<string, Record<string, number>> = {};
  for (const row of rows) {
    out[row.jobType] ??= {};
    out[row.jobType]![row.status] = row.count;
  }
  return out;
}

function unfinishedJobCounts(db: Database.Database): Record<string, number> {
  const rows = db.prepare(
    `SELECT status, COUNT(*) AS count
     FROM evolution_jobs
     WHERE status IN ('queued', 'leased', 'failed', 'dead_letter')
     GROUP BY status`
  ).all() as Array<{ status: string; count: number }>;
  return Object.fromEntries(rows.map((row) => [row.status, Number(row.count)]));
}

function memoryLayerCounts(db: Database.Database, layer: string): Record<string, number> {
  const rows = db.prepare(
    `SELECT status, COUNT(*) AS count FROM memories WHERE memory_layer = ? GROUP BY status`
  ).all(layer) as Array<{ status: string; count: number }>;
  return Object.fromEntries(rows.map((row) => [row.status, row.count]));
}

function groupedCount(
  db: Database.Database,
  table: string,
  column: string
): Record<string, number> {
  assertIdentifier(table);
  assertIdentifier(column);
  const rows = db.prepare(
    `SELECT "${column}" AS value, COUNT(*) AS count FROM "${table}" GROUP BY "${column}"`
  ).all() as Array<{ value: string; count: number }>;
  return Object.fromEntries(rows.map((row) => [row.value, row.count]));
}

function scalarCount(db: Database.Database, table: string): number {
  assertIdentifier(table);
  const row = db.prepare(`SELECT COUNT(*) AS count FROM "${table}"`).get() as {
    count: number;
  };
  return Number(row.count);
}

function publicModelConfig(config: {
  provider: string;
  model?: string;
  endpoint?: string;
}) {
  return {
    provider: config.provider,
    ...(config.model ? { model: config.model } : {}),
    ...(config.endpoint ? { endpoint: config.endpoint } : {})
  };
}

function parseArgs(argv: readonly string[]): ReplayArgs {
  const args: ReplayArgs = {
    help: false,
    episodeIds: [],
    chain: "both",
    maxRounds: 200,
    workerLimit: 100,
    confirmRemoteModels: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") args.help = true;
    else if (arg === "--db") args.sourceDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--config") args.configPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output-db") args.outputDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output") args.outputReportPath = requiredValue(argv, ++index, arg);
    else if (arg === "--episode-ids") {
      args.episodeIds.push(...requiredValue(argv, ++index, arg).split(",")
        .map((value) => value.trim()).filter(Boolean));
    } else if (arg === "--cohort-source") {
      args.cohortSourcePath = requiredValue(argv, ++index, arg);
    } else if (arg === "--chain") {
      const value = requiredValue(argv, ++index, arg);
      if (value !== "old" && value !== "new" && value !== "both") {
        throw new Error("--chain must be old, new, or both");
      }
      args.chain = value;
    } else if (arg === "--max-episodes") {
      args.maxEpisodes = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--max-rounds") {
      args.maxRounds = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--worker-limit") {
      args.workerLimit = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--confirm-remote-models") {
      args.confirmRemoteModels = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  args.episodeIds = unique(args.episodeIds);
  return args;
}

function helpText(): string {
  return [
    "Usage:",
    "  npm run kw-dual-skill:replay -- --db <kw.sqlite> --output-db <fresh.sqlite>",
    "    [--output <report.json>] [--config <config.yaml>]",
    "    [--episode-ids <id1,id2,...> | --cohort-source <prior-result.json>]",
    "    [--chain old|new|both] [--max-episodes N] [--max-rounds 200]",
    "    [--worker-limit 100] --confirm-remote-models",
    "",
    `The source DB is opened read-only. A fresh schema-v${SCHEMA_VERSION} DB receives only sessions,`,
    "closed Episodes, RawTurns, L1 traces (including their vectors), and persisted rewards.",
    "Historical L2/L3/Skill/Span state and jobs are never imported.",
    "",
    "old: seed eligible L1 traces into the upstream Turn→L2→Skill chain",
    "new: seed closed Episodes into the Span-5/10 procedural direct-Skill chain",
    "both: run both chains in the same isolated DB and count Skills by plugin algorithm",
    ""
  ].join("\n");
}

function requiredValue(argv: readonly string[], index: number, option: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${option} requires a value`);
  return value;
}

function positiveInt(value: string, option: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${option} must be positive`);
  return parsed;
}

function requiredResolvedPath(value: string | undefined, option: string): string {
  if (!value) throw new Error(`${option} is required`);
  return resolve(value);
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

function assertIdentifier(value: string): void {
  if (!/^[a-z_][a-z0-9_]*$/i.test(value)) throw new Error(`Unsafe SQLite identifier: ${value}`);
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
