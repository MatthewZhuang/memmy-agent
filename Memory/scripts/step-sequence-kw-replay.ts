import Database from "better-sqlite3";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  MemoryDb,
  MemoryService,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type StepSequenceLearningResult
} from "../src/index.js";
import { Repositories } from "../src/storage/repositories.js";

interface Args {
  dbPath: string;
  configPath?: string;
  outputDbPath: string;
  outputReportPath: string;
  sampleSize: number;
  concurrency: number;
  seed: number;
  reconstructRaw: boolean;
  rewardFilter: "all" | "positive" | "negative" | "zero" | "unknown";
  episodeIds?: string[];
}

interface EpisodeCandidate {
  episodeId: string;
  pathId?: string;
  stepCount: number;
  turnCount: number;
  toolCallCount: number;
  terminalReward?: number;
  stratum: "short" | "medium" | "long";
}

interface EpisodeRun {
  candidate: EpisodeCandidate;
  result?: StepSequenceLearningResult;
  reconstruction?: {
    pathId: string;
    pathHash: string;
    created: boolean;
  };
  elapsedMs: number;
  error?: string;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(dirname(args.outputDbPath), { recursive: true });
  mkdirSync(dirname(args.outputReportPath), { recursive: true });
  const candidates = listCandidates(args.dbPath, args.reconstructRaw, args.rewardFilter);
  const sampled = args.episodeIds
    ? selectRequestedEpisodes(candidates, args.episodeIds)
    : stratifiedSample(candidates, args.sampleSize, args.seed);
  if (sampled.length === 0) throw new Error("No eligible closed KW Episodes were found");
  await cloneDatabase(args.dbPath, args.outputDbPath);
  const loaded = loadMemmyConfig(args.configPath);
  const db = new MemoryDb({ path: args.outputDbPath });
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();
  try {
    const service = new MemoryService({
      db,
      mode: "local",
      config: loaded.config,
      ...(args.configPath ? { configPath: args.configPath } : {})
    });
    const runs = await mapConcurrent(sampled, args.concurrency, async (candidate) => {
      const start = Date.now();
      try {
        const reconstruction = args.reconstructRaw
          ? await service.reconstructProceduralPathForReplay({ episodeId: candidate.episodeId })
          : undefined;
        const result = await service.learnStepSequencesForReplay({
          episodeId: candidate.episodeId
        });
        return {
          candidate,
          ...(reconstruction ? {
            reconstruction: {
              pathId: reconstruction.record.id,
              pathHash: reconstruction.record.pathHash,
              created: reconstruction.created
            }
          } : {}),
          result,
          elapsedMs: Date.now() - start
        } satisfies EpisodeRun;
      } catch (error) {
        return {
          candidate,
          elapsedMs: Date.now() - start,
          error: error instanceof Error ? error.stack ?? error.message : String(error)
        } satisfies EpisodeRun;
      }
    });
    const repos = new Repositories(db.db);
    const namespaceIds = unique(sampled.flatMap((candidate) => {
      const path = repos.proceduralPaths.getActiveForEpisode(candidate.episodeId);
      return path ? [path.namespaceId] : [];
    }));
    const stepPatterns = namespaceIds.flatMap((namespaceId) =>
      repos.stepSequenceLearning.listStepPatterns(namespaceId));
    const policies = namespaceIds.flatMap((namespaceId) =>
      repos.stepSequenceLearning.listActivePolicies(namespaceId));
    const skillPatterns = namespaceIds.flatMap((namespaceId) =>
      repos.stepSequenceLearning.listSkillPatterns(namespaceId));
    const artifact = {
      schemaVersion: "step-sequence-kw-sample.v1",
      startedAt,
      finishedAt: new Date().toISOString(),
      elapsedMs: Date.now() - wallStart,
      sourceDatabase: resolve(args.dbPath),
      outputDatabase: args.outputDbPath,
      sampling: {
        seed: args.seed,
        requested: args.sampleSize,
        selected: sampled.length,
        available: candidates.length,
        concurrency: args.concurrency,
        reconstructRaw: args.reconstructRaw,
        rewardFilter: args.rewardFilter,
        fixedEpisodeOrder: Boolean(args.episodeIds),
        strata: Object.fromEntries(["short", "medium", "long"].map((stratum) => [
          stratum,
          sampled.filter((candidate) => candidate.stratum === stratum).length
        ]))
      },
      models: {
        evolution: publicModel(resolveEvolutionConfig(loaded.config)),
        embedding: publicModel(loaded.config.embedding)
      },
      runs,
      summary: {
        succeeded: runs.filter((run) => !run.error).length,
        failed: runs.filter((run) => run.error).length,
        steps: sum(runs.map((run) => run.result?.stepCount ?? 0)),
        distinctAssignedClusters: countRows(db.db, "procedural_step_clusters"),
        repeatedRawStepPatterns: stepPatterns.filter((pattern) =>
          pattern.distinctEpisodeCount >= 2).length,
        selectedStepPatterns: stepPatterns.filter((pattern) =>
          pattern.lifecycleStatus === "ready" && pattern.selectedEpisodeCount >= 2 &&
          !pattern.supersededByPatternId).length,
        supersededStepPatterns: stepPatterns.filter((pattern) =>
          pattern.supersededByPatternId).length,
        activePolicies: policies.length,
        activeProjections: scalar(db.db,
          `SELECT COUNT(*) FROM episode_step_policy_projections WHERE status = 'active'`),
        selectedSkillPatterns: skillPatterns.filter((pattern) =>
          pattern.lifecycleStatus === "ready" && pattern.selectedEpisodeCount >= 2 &&
          !pattern.supersededByPatternId).length,
        supersededSkillPatterns: skillPatterns.filter((pattern) =>
          pattern.supersededByPatternId).length,
        compiledSkills: skillPatterns.filter((pattern) => pattern.activeSkillMemoryId).length
      }
    };
    writeFileSync(args.outputReportPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
    const markdownPath = args.outputReportPath.replace(/\.json$/i, ".md");
    writeFileSync(markdownPath, renderMarkdown(artifact), "utf8");
    process.stdout.write(`${JSON.stringify({
      outputDatabase: args.outputDbPath,
      outputReport: args.outputReportPath,
      outputMarkdown: markdownPath,
      sampling: artifact.sampling,
      summary: artifact.summary,
      failedEpisodes: runs.filter((run) => run.error).map((run) => ({
        episodeId: run.candidate.episodeId,
        error: run.error?.split("\n")[0]
      }))
    }, null, 2)}\n`);
  } finally {
    db.close();
  }
}

function listCandidates(
  dbPath: string,
  reconstructRaw: boolean,
  rewardFilter: Args["rewardFilter"]
): EpisodeCandidate[] {
  const db = new Database(resolve(dbPath), { readonly: true, fileMustExist: true });
  try {
    const rows = db.prepare(
      `SELECT episodes.id AS episode_id,
              episodes.turn_count,
              episodes.r_task,
              paths.id AS path_id,
              COALESCE(json_array_length(paths.payload_json, '$.steps'), 0) AS step_count,
              COALESCE(SUM(json_array_length(raw_turns.tool_calls_json)), 0) AS tool_call_count
       FROM episodes
       LEFT JOIN episode_procedural_paths AS paths
         ON paths.episode_id = episodes.id AND paths.status = 'active'
       LEFT JOIN raw_turns ON raw_turns.episode_id = episodes.id
       WHERE episodes.status = 'closed'
       GROUP BY episodes.id, paths.id
       HAVING (? = 1 AND tool_call_count > 0)
           OR (? = 0 AND path_id IS NOT NULL AND step_count >= 2)
       ORDER BY episodes.id`
    ).all(reconstructRaw ? 1 : 0, reconstructRaw ? 1 : 0) as Array<{
      episode_id: string;
      turn_count: number;
      r_task: number | null;
      path_id: string | null;
      step_count: number;
      tool_call_count: number;
    }>;
    return rows.filter((row) => rewardMatches(row.r_task, rewardFilter)).map((row) => {
      const size = reconstructRaw ? row.tool_call_count : row.step_count;
      return {
        episodeId: row.episode_id,
        ...(row.path_id ? { pathId: row.path_id } : {}),
        stepCount: row.step_count,
        turnCount: row.turn_count,
        toolCallCount: row.tool_call_count,
        ...(row.r_task === null ? {} : { terminalReward: row.r_task }),
        stratum: size <= 20 ? "short" : size <= 50 ? "medium" : "long"
      };
    });
  } finally {
    db.close();
  }
}

function stratifiedSample(
  candidates: readonly EpisodeCandidate[],
  sampleSize: number,
  seed: number
): EpisodeCandidate[] {
  const rng = mulberry32(seed);
  const strata = (["short", "medium", "long"] as const).map((stratum) => ({
    stratum,
    candidates: shuffle(candidates.filter((candidate) => candidate.stratum === stratum), rng)
  }));
  const selected: EpisodeCandidate[] = [];
  while (selected.length < sampleSize && strata.some((bucket) => bucket.candidates.length > 0)) {
    for (const bucket of strata) {
      const candidate = bucket.candidates.shift();
      if (candidate) selected.push(candidate);
      if (selected.length >= sampleSize) break;
    }
  }
  return selected;
}

async function cloneDatabase(sourcePath: string, outputPath: string): Promise<void> {
  const source = new Database(resolve(sourcePath), { readonly: true, fileMustExist: true });
  try {
    await source.backup(outputPath);
  } finally {
    source.close();
  }
}

async function mapConcurrent<T, R>(
  values: readonly T[],
  concurrency: number,
  worker: (value: T, index: number) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= values.length) return;
      results[index] = await worker(values[index]!, index);
    }
  }));
  return results;
}

function renderMarkdown(artifact: {
  startedAt: string;
  finishedAt: string;
  elapsedMs: number;
  sourceDatabase: string;
  outputDatabase: string;
  sampling: Record<string, unknown>;
  models: Record<string, unknown>;
  runs: EpisodeRun[];
  summary: Record<string, unknown>;
}): string {
  const lines = [
    "# KW Step-Sequence 小样本链路验证",
    "",
    `- Started: ${artifact.startedAt}`,
    `- Finished: ${artifact.finishedAt}`,
    `- Elapsed: ${Math.round(artifact.elapsedMs / 1000)}s`,
    `- Source DB: \`${artifact.sourceDatabase}\``,
    `- Output DB: \`${artifact.outputDatabase}\``,
    "",
    "## Sampling",
    "",
    "```json",
    JSON.stringify(artifact.sampling, null, 2),
    "```",
    "",
    "## Summary",
    "",
    "```json",
    JSON.stringify(artifact.summary, null, 2),
    "```",
    "",
    "## Episodes",
    ""
  ];
  for (const run of artifact.runs) {
    lines.push(
      `### ${run.candidate.episodeId}`,
      "",
      `- Stratum: ${run.candidate.stratum}`,
      `- Source Steps: ${run.candidate.stepCount}`,
      `- Terminal Reward: ${run.candidate.terminalReward ?? "unknown"}`,
      `- Tool calls: ${run.candidate.toolCallCount}`,
      `- Elapsed: ${Math.round(run.elapsedMs / 1000)}s`,
      `- Result: ${run.error ? `failed — ${run.error.split("\n")[0]}` : "succeeded"}`,
      ...(run.result ? [
        `- Learned Steps: ${run.result.stepCount}`,
        `- Assigned Clusters: ${run.result.clusterCount}`,
        `- Repeated Patterns: ${run.result.repeatedPatternCount}`,
        `- Policies: ${run.result.inducedPolicyIds.length}`,
        `- Skills: ${run.result.compiledSkillIds.length}`
      ] : []),
      ""
    );
  }
  return `${lines.join("\n")}\n`;
}

function parseArgs(argv: readonly string[]): Args {
  let dbPath = "";
  let configPath: string | undefined;
  let outputDbPath = "";
  let outputReportPath = "";
  let sampleSize = 12;
  let concurrency = 4;
  let seed = 20260825;
  let reconstructRaw = false;
  let rewardFilter: Args["rewardFilter"] = "all";
  let episodeIds: string[] | undefined;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--db") dbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--config") configPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output-db") outputDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output") outputReportPath = requiredValue(argv, ++index, arg);
    else if (arg === "--sample-size") sampleSize = positiveInt(requiredValue(argv, ++index, arg), arg);
    else if (arg === "--concurrency") concurrency = positiveInt(requiredValue(argv, ++index, arg), arg);
    else if (arg === "--seed") seed = positiveInt(requiredValue(argv, ++index, arg), arg);
    else if (arg === "--reward") {
      const value = requiredValue(argv, ++index, arg);
      if (!["all", "positive", "negative", "zero", "unknown"].includes(value)) {
        throw new Error("--reward must be all, positive, negative, zero, or unknown");
      }
      rewardFilter = value as Args["rewardFilter"];
    }
    else if (arg === "--episode-ids") {
      episodeIds = unique(requiredValue(argv, ++index, arg)
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean));
      if (episodeIds.length === 0) throw new Error("--episode-ids requires at least one Episode ID");
    }
    else if (arg === "--from") {
      const value = requiredValue(argv, ++index, arg);
      if (value !== "path" && value !== "raw") throw new Error("--from must be path or raw");
      reconstructRaw = value === "raw";
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write(helpText());
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!dbPath) throw new Error("--db is required");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  const base = resolve("../../experiments/results/trace2skill-step-sequence-kw");
  return {
    dbPath: resolve(dbPath),
    ...(configPath ? { configPath: resolve(configPath) } : {}),
    outputDbPath: resolve(outputDbPath || `${base}/${stamp}-kw-step-sequence-sample.sqlite`),
    outputReportPath: resolve(outputReportPath || `${base}/${stamp}-kw-step-sequence-sample.json`),
    sampleSize,
    concurrency,
    seed,
    reconstructRaw,
    rewardFilter,
    ...(episodeIds ? { episodeIds } : {})
  };
}

function helpText(): string {
  return `Usage:\n  npm run step-sequence:kw -- --db <kw.sqlite> [--config <config.yaml>] [--from path|raw] [--reward all|positive|negative|zero|unknown] [--episode-ids <id1,id2,...>] [--sample-size 12] [--concurrency 4] [--seed 20260825] [--output-db <copy.sqlite>] [--output <report.json>]\n\nThe command always writes to a cloned database. It samples closed Episodes by reward and short/medium/long trajectory strata, or replays an explicit Episode ID list in the supplied order, and never mutates the source KW database.\n`;
}

function selectRequestedEpisodes(
  candidates: readonly EpisodeCandidate[],
  episodeIds: readonly string[]
): EpisodeCandidate[] {
  const byId = new Map(candidates.map((candidate) => [candidate.episodeId, candidate]));
  const missing = episodeIds.filter((episodeId) => !byId.has(episodeId));
  if (missing.length > 0) {
    throw new Error(`Requested Episodes are unavailable under the current filters: ${missing.join(", ")}`);
  }
  return episodeIds.map((episodeId) => byId.get(episodeId)!);
}

function rewardMatches(
  terminalReward: number | null,
  filter: Args["rewardFilter"]
): boolean {
  if (filter === "all") return true;
  if (filter === "unknown") return terminalReward === null;
  if (terminalReward === null) return false;
  if (filter === "positive") return terminalReward > 0;
  if (filter === "negative") return terminalReward < 0;
  return terminalReward === 0;
}

function requiredValue(argv: readonly string[], index: number, option: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${option} requires a value`);
  return value;
}

function positiveInt(value: string, option: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${option} must be a positive integer`);
  return parsed;
}

function shuffle<T>(values: readonly T[], rng: () => number): T[] {
  const output = [...values];
  for (let index = output.length - 1; index > 0; index -= 1) {
    const selected = Math.floor(rng() * (index + 1));
    [output[index], output[selected]] = [output[selected]!, output[index]!];
  }
  return output;
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let result = value;
    result = Math.imul(result ^ result >>> 15, result | 1);
    result ^= result + Math.imul(result ^ result >>> 7, result | 61);
    return ((result ^ result >>> 14) >>> 0) / 4_294_967_296;
  };
}

function publicModel(config: {
  provider: string;
  model?: string;
  endpoint?: string;
}): Record<string, string> {
  return {
    provider: config.provider,
    ...(config.model ? { model: config.model } : {}),
    ...(config.endpoint ? { endpoint: config.endpoint } : {})
  };
}

function scalar(db: Database.Database, sql: string): number {
  return Number((db.prepare(sql).pluck().get() as number | bigint | undefined) ?? 0);
}

function countRows(db: Database.Database, table: string): number {
  return scalar(db, `SELECT COUNT(*) FROM ${table}`);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
