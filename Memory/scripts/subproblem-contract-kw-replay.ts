import Database from "better-sqlite3";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  MemoryDb,
  MemoryService,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type ExecutionStepV1,
  type SubproblemContractSegmentationResultV1
} from "../src/index.js";
import { Repositories } from "../src/storage/repositories.js";

const DEFAULT_EPISODE_IDS = [
  "episode_a13328d5697374ebb870",
  "episode_561ef31567158709abf7",
  "episode_a211ae9cad49e183f5b7",
  "episode_57363e9ed3d06d661840"
] as const;

const REFERENCES: Record<string, { expectedBoundaries: number[]; rationale: string }> = {
  episode_a13328d5697374ebb870: {
    expectedBoundaries: [69],
    rationale: "药品/价格证据收集 → Excel生成与验收"
  },
  episode_561ef31567158709abf7: {
    expectedBoundaries: [24],
    rationale: "NFT摄影素材研究 → Word文档生成与验收"
  },
  episode_a211ae9cad49e183f5b7: {
    expectedBoundaries: [37, 58],
    rationale: "学术文章检索 → 研究表初稿生成 → verifier反馈修订"
  },
  episode_57363e9ed3d06d661840: {
    expectedBoundaries: [],
    rationale: "PDF deck生成中的失败、修复和验收属于同一交付闭环"
  }
};

interface Args {
  dbPath: string;
  configPath?: string;
  outputDbPath: string;
  outputReportPath: string;
  concurrency: number;
  episodeIds: string[];
  sampleSize?: number;
  seed: number;
  reconstructRaw: boolean;
  excludedEpisodeIds: string[];
}

interface EpisodeCandidate {
  episodeId: string;
  toolCallCount: number;
  stepCount: number;
  stratum: "short" | "medium" | "long";
}

interface EpisodeRun {
  episodeId: string;
  terminalReward?: number;
  expected?: typeof REFERENCES[string];
  steps?: ExecutionStepV1[];
  segmentation?: SubproblemContractSegmentationResultV1;
  elapsedMs: number;
  error?: string;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(dirname(args.outputDbPath), { recursive: true });
  mkdirSync(dirname(args.outputReportPath), { recursive: true });
  const sampled = args.sampleSize === undefined
    ? args.episodeIds.map((episodeId) => ({
        episodeId,
        toolCallCount: 0,
        stepCount: 0,
        stratum: "short" as const
      }))
    : stratifiedSample(
        listCandidates(args.dbPath, args.reconstructRaw).filter((candidate) =>
          !args.excludedEpisodeIds.includes(candidate.episodeId)),
        args.sampleSize,
        args.seed
      );
  if (sampled.length === 0) throw new Error("No eligible closed KW Episodes were found");
  const episodeIds = sampled.map((candidate) => candidate.episodeId);
  process.stdout.write(`${JSON.stringify({
    phase: "selected",
    seed: args.seed,
    reconstructRaw: args.reconstructRaw,
    episodes: sampled
  }, null, 2)}\n`);
  await cloneDatabase(args.dbPath, args.outputDbPath);
  const loaded = loadMemmyConfig(args.configPath);
  const db = new MemoryDb({ path: args.outputDbPath });
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();
  try {
    const repos = new Repositories(db.db);
    const service = new MemoryService({
      db,
      mode: "local",
      config: loaded.config,
      ...(args.configPath ? { configPath: args.configPath } : {})
    });
    const runs = await mapConcurrent(episodeIds, args.concurrency, async (episodeId) => {
      const start = Date.now();
      let episode = repos.runtime.getEpisode(episodeId);
      if (!episode || episode.status !== "closed") {
        return {
          episodeId,
          elapsedMs: Date.now() - start,
          error: `closed Episode not found: ${episodeId}`
        } satisfies EpisodeRun;
      }
      try {
        if (args.reconstructRaw) {
          await service.reconstructProceduralPathForReplay({ episodeId });
          episode = repos.runtime.getEpisode(episodeId);
        }
        const path = repos.proceduralPaths.getActiveForEpisode(episodeId);
        if (!path) throw new Error(`active procedural Path not found: ${episodeId}`);
        const segmentation = await service.segmentEpisodeSubproblemContractsForReplay({ episodeId });
        const run = {
          episodeId,
          ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask }),
          ...(REFERENCES[episodeId] ? { expected: REFERENCES[episodeId] } : {}),
          steps: path.path.steps,
          segmentation,
          elapsedMs: Date.now() - start
        } satisfies EpisodeRun;
        process.stdout.write(`${JSON.stringify({
          phase: "episode-complete",
          episodeId,
          steps: run.steps.length,
          segments: run.segmentation.segments.length,
          boundaries: observedBoundaries(run),
          elapsedMs: run.elapsedMs
        })}\n`);
        return run;
      } catch (error) {
        const path = repos.proceduralPaths.getActiveForEpisode(episodeId);
        return {
          episodeId,
          ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask }),
          ...(REFERENCES[episodeId] ? { expected: REFERENCES[episodeId] } : {}),
          ...(path ? { steps: path.path.steps } : {}),
          elapsedMs: Date.now() - start,
          error: error instanceof Error ? error.stack ?? error.message : String(error)
        } satisfies EpisodeRun;
      }
    });
    const metrics = aggregateMetrics(runs);
    const artifact = {
      schemaVersion: "subproblem-contract-kw-replay.v1",
      startedAt,
      finishedAt: new Date().toISOString(),
      elapsedMs: Date.now() - wallStart,
      sourceDatabase: resolve(args.dbPath),
      outputDatabase: args.outputDbPath,
      concurrency: args.concurrency,
      sampling: {
        mode: args.sampleSize === undefined ? "fixed" : "stratified-random",
        seed: args.seed,
        requested: args.sampleSize ?? args.episodeIds.length,
        selected: sampled.length,
        reconstructRaw: args.reconstructRaw,
        excludedEpisodeIds: args.excludedEpisodeIds,
        episodes: sampled
      },
      models: { evolution: publicModel(resolveEvolutionConfig(loaded.config)) },
      protocol: {
        contract: ["target", "desiredOutcome"],
        windowSize: 15,
        overlap: 5,
        decision: "earliest candidate Step outside the immutable current contract, or null",
        outputTokenBudget: 16_000
      },
      summary: {
        selectedEpisodes: runs.length,
        succeeded: runs.filter((run) => !run.error).length,
        failed: runs.filter((run) => run.error).length,
        totalSteps: sum(runs.map((run) => run.steps?.length ?? 0)),
        totalSegments: sum(runs.map((run) => run.segmentation?.segments.length ?? 0)),
        annotatedEpisodes: runs.filter((run) => run.expected).length,
        exactEpisodeMatches: runs.filter(exactMatch).length,
        ...metrics,
        admissionPassed: runs.every((run) => run.expected) &&
          metrics.precision >= 0.8 && metrics.recall >= 0.8 &&
          runs.filter(exactMatch).length === runs.length
      },
      runs
    };
    writeFileSync(args.outputReportPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
    const markdownPath = args.outputReportPath.replace(/\.json$/i, ".md");
    writeFileSync(markdownPath, renderMarkdown(artifact), "utf8");
    process.stdout.write(`${JSON.stringify({
      outputDatabase: args.outputDbPath,
      outputReport: args.outputReportPath,
      outputMarkdown: markdownPath,
      model: artifact.models.evolution,
      summary: artifact.summary,
      episodes: runs.map((run) => ({
        episodeId: run.episodeId,
        steps: run.steps?.length,
        segments: run.segmentation?.segments.length,
        contracts: run.segmentation?.segments.map((segment) => segment.contract),
        boundaries: observedBoundaries(run),
        expectedBoundaries: run.expected?.expectedBoundaries,
        elapsedMs: run.elapsedMs,
        error: run.error?.split("\n")[0]
      }))
    }, null, 2)}\n`);
  } finally {
    db.close();
  }
}

function renderMarkdown(artifact: {
  startedAt: string;
  finishedAt: string;
  elapsedMs: number;
  sourceDatabase: string;
  outputDatabase: string;
  concurrency: number;
  sampling: Record<string, unknown>;
  models: Record<string, unknown>;
  protocol: Record<string, unknown>;
  summary: Record<string, unknown>;
  runs: EpisodeRun[];
}): string {
  const lines = [
    "# KW 持续 Subproblem Contract 切分实验",
    "",
    `- Started: ${artifact.startedAt}`,
    `- Finished: ${artifact.finishedAt}`,
    `- Elapsed: ${Math.round(artifact.elapsedMs / 1000)}s`,
    `- Source DB: \`${artifact.sourceDatabase}\``,
    `- Output DB: \`${artifact.outputDatabase}\``,
    `- Concurrency: ${artifact.concurrency}`,
    "",
    "## Sampling",
    "",
    "```json",
    JSON.stringify(artifact.sampling, null, 2),
    "```",
    "",
    "## Protocol",
    "",
    "```json",
    JSON.stringify(artifact.protocol, null, 2),
    "```",
    "",
    "## Summary",
    "",
    "```json",
    JSON.stringify(artifact.summary, null, 2),
    "```",
    ""
  ];
  for (const run of artifact.runs) {
    lines.push(
      `## ${run.episodeId}`,
      "",
      `- Reward: ${run.terminalReward ?? "unknown"}`,
      `- Steps: ${run.steps?.length ?? 0}`,
      `- Windows: ${run.segmentation?.windows.length ?? 0}`,
      `- Observed boundaries: ${observedBoundaries(run).join(", ") || "none"}`,
      `- Expected boundaries: ${run.expected?.expectedBoundaries.join(", ") || "none"}`,
      `- Reference: ${run.expected?.rationale ?? "not annotated"}`,
      `- Exact match: ${exactMatch(run)}`,
      `- Elapsed: ${Math.round(run.elapsedMs / 1000)}s`,
      `- Result: ${run.error ? `failed — ${run.error.split("\n")[0]}` : "succeeded"}`,
      ""
    );
    if (!run.segmentation || !run.steps) continue;
    for (const segment of run.segmentation.segments) {
      lines.push(
        `### Segment ${segment.segmentIndex + 1}: Steps ${segment.startStepIndex}–${segment.endStepIndex}`,
        "",
        `- Target: ${segment.contract.target}`,
        `- Desired outcome: ${segment.contract.desiredOutcome}`,
        "",
        "| Step | Turn | Type | Outcome | Intent | Summary |",
        "|---:|---:|---|---|---|---|"
      );
      for (const step of run.steps.slice(segment.startStepIndex, segment.endStepIndex + 1)) {
        lines.push(`| ${step.stepIndex} | ${step.turnIndex} | ${escapeCell(step.action.type)} | ${step.outcome.status} | ${escapeCell(step.action.intent)} | ${escapeCell(step.action.summary)} |`);
      }
      lines.push("");
    }
  }
  return `${lines.join("\n")}\n`;
}

function aggregateMetrics(runs: readonly EpisodeRun[]): {
  truePositive: number;
  falsePositive: number;
  falseNegative: number;
  precision: number;
  recall: number;
  f1: number;
} {
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  for (const run of runs) {
    if (!run.expected || !run.segmentation) continue;
    const expected = new Set(run.expected.expectedBoundaries);
    const observed = new Set(observedBoundaries(run));
    truePositive += [...observed].filter((boundary) => expected.has(boundary)).length;
    falsePositive += [...observed].filter((boundary) => !expected.has(boundary)).length;
    falseNegative += [...expected].filter((boundary) => !observed.has(boundary)).length;
  }
  const precision = truePositive / Math.max(1, truePositive + falsePositive);
  const recall = truePositive / Math.max(1, truePositive + falseNegative);
  const f1 = precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
  return { truePositive, falsePositive, falseNegative, precision, recall, f1 };
}

function exactMatch(run: EpisodeRun): boolean {
  if (!run.expected || !run.segmentation) return false;
  return JSON.stringify(observedBoundaries(run)) === JSON.stringify(run.expected.expectedBoundaries);
}

function observedBoundaries(run: EpisodeRun): number[] {
  return run.segmentation?.segments.slice(1).map((segment) => segment.startStepIndex) ?? [];
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
  worker: (value: T) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= values.length) return;
      results[index] = await worker(values[index]!);
    }
  }));
  return results;
}

function parseArgs(argv: readonly string[]): Args {
  let dbPath = "";
  let configPath: string | undefined;
  let outputDbPath = "";
  let outputReportPath = "";
  let concurrency = 4;
  let episodeIds = [...DEFAULT_EPISODE_IDS];
  let sampleSize: number | undefined;
  let seed = 20260826;
  let reconstructRaw = false;
  let excludedEpisodeIds: string[] = [];
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--db") dbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--config") configPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output-db") outputDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output") outputReportPath = requiredValue(argv, ++index, arg);
    else if (arg === "--concurrency") {
      concurrency = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--sample-size") {
      sampleSize = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--seed") {
      seed = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--from") {
      const value = requiredValue(argv, ++index, arg);
      if (value !== "path" && value !== "raw") throw new Error("--from must be path or raw");
      reconstructRaw = value === "raw";
    } else if (arg === "--exclude-episode-ids") {
      excludedEpisodeIds = requiredValue(argv, ++index, arg).split(",")
        .map((value) => value.trim()).filter(Boolean);
    } else if (arg === "--episode-ids") {
      episodeIds = requiredValue(argv, ++index, arg).split(",")
        .map((value) => value.trim()).filter(Boolean);
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write(helpText());
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!dbPath) throw new Error("--db is required");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  const base = resolve("../../experiments/results/trace2skill-segment-boundary-kw");
  return {
    dbPath: resolve(dbPath),
    ...(configPath ? { configPath: resolve(configPath) } : {}),
    outputDbPath: resolve(outputDbPath || `${base}/${stamp}-kw-subproblem-contract.sqlite`),
    outputReportPath: resolve(outputReportPath || `${base}/${stamp}-kw-subproblem-contract.json`),
    concurrency,
    episodeIds: [...new Set(episodeIds)],
    ...(sampleSize === undefined ? {} : { sampleSize }),
    seed,
    reconstructRaw,
    excludedEpisodeIds: [...new Set(excludedEpisodeIds)]
  };
}

function helpText(): string {
  return "Usage: npm run subproblem-contract:kw -- --db <kw.sqlite> [--config <config.yaml>] [--from path|raw] [--episode-ids <id1,id2,...> | --sample-size 4 --seed 20260826] [--exclude-episode-ids <id1,id2,...>] [--concurrency 4] [--output-db <copy.sqlite>] [--output <report.json>]\n";
}

function listCandidates(dbPath: string, reconstructRaw: boolean): EpisodeCandidate[] {
  const db = new Database(resolve(dbPath), { readonly: true, fileMustExist: true });
  try {
    const rows = db.prepare(
      `SELECT episodes.id AS episode_id,
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
      step_count: number;
      tool_call_count: number;
    }>;
    return rows.map((row) => {
      const size = reconstructRaw ? row.tool_call_count : row.step_count;
      return {
        episodeId: row.episode_id,
        toolCallCount: row.tool_call_count,
        stepCount: row.step_count,
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

function shuffle<T>(values: readonly T[], rng: () => number): T[] {
  const result = [...values];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex]!, result[index]!];
  }
  return result;
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let mixed = value;
    mixed = Math.imul(mixed ^ mixed >>> 15, mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ mixed >>> 7, mixed | 61);
    return ((mixed ^ mixed >>> 14) >>> 0) / 4_294_967_296;
  };
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

function escapeCell(value: unknown): string {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
