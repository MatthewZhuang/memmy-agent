import Database from "better-sqlite3";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  MemoryDb,
  MemoryService,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type EpisodeBoundarySegmentationResultV1,
  type ExecutionStepV1
} from "../src/index.js";

const DEFAULT_EPISODE_IDS = [
  "episode_a13328d5697374ebb870",
  "episode_561ef31567158709abf7",
  "episode_a211ae9cad49e183f5b7",
  "episode_57363e9ed3d06d661840"
] as const;

const REFERENCE_EXPECTATIONS: Record<string, {
  expectedSegmentCount: number;
  rationale: string;
}> = {
  episode_a13328d5697374ebb870: {
    expectedSegmentCount: 2,
    rationale: "药品/价格研究完整生命周期 → Excel生成与验收"
  },
  episode_561ef31567158709abf7: {
    expectedSegmentCount: 2,
    rationale: "NFT摄影素材研究 → Word文档生成与验收"
  },
  episode_a211ae9cad49e183f5b7: {
    expectedSegmentCount: 3,
    rationale: "学术文章检索 → 研究表初稿生成 → verifier反馈修订"
  },
  episode_57363e9ed3d06d661840: {
    expectedSegmentCount: 1,
    rationale: "PDF deck生成中的失败、修复与验证仍属于同一交付子问题"
  }
};

interface Args {
  dbPath: string;
  configPath?: string;
  outputDbPath: string;
  outputReportPath: string;
  concurrency: number;
  episodeIds: string[];
}

interface EpisodeMetadata {
  episodeId: string;
  turnCount: number;
  terminalReward?: number;
  toolCallCount: number;
}

interface EpisodeRun {
  metadata: EpisodeMetadata;
  expected?: typeof REFERENCE_EXPECTATIONS[string];
  pathId?: string;
  pathHash?: string;
  steps?: ExecutionStepV1[];
  segmentation?: EpisodeBoundarySegmentationResultV1;
  elapsedMs: number;
  error?: string;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(dirname(args.outputDbPath), { recursive: true });
  mkdirSync(dirname(args.outputReportPath), { recursive: true });
  const episodes = loadEpisodeMetadata(args.dbPath, args.episodeIds);
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
    const runs = await mapConcurrent(episodes, args.concurrency, async (metadata) => {
      const start = Date.now();
      try {
        const reconstruction = await service.reconstructProceduralPathForReplay({
          episodeId: metadata.episodeId
        });
        const segmentation = await service.segmentEpisodeBoundariesForReplay({
          episodeId: metadata.episodeId
        });
        return {
          metadata,
          ...(REFERENCE_EXPECTATIONS[metadata.episodeId]
            ? { expected: REFERENCE_EXPECTATIONS[metadata.episodeId] }
            : {}),
          pathId: reconstruction.record.id,
          pathHash: reconstruction.record.pathHash,
          steps: reconstruction.record.path.steps,
          segmentation,
          elapsedMs: Date.now() - start
        } satisfies EpisodeRun;
      } catch (error) {
        return {
          metadata,
          ...(REFERENCE_EXPECTATIONS[metadata.episodeId]
            ? { expected: REFERENCE_EXPECTATIONS[metadata.episodeId] }
            : {}),
          elapsedMs: Date.now() - start,
          error: error instanceof Error ? error.stack ?? error.message : String(error)
        } satisfies EpisodeRun;
      }
    });
    const artifact = {
      schemaVersion: "episode-boundary-kw-replay.v1",
      startedAt,
      finishedAt: new Date().toISOString(),
      elapsedMs: Date.now() - wallStart,
      sourceDatabase: resolve(args.dbPath),
      outputDatabase: args.outputDbPath,
      configPath: args.configPath,
      concurrency: args.concurrency,
      models: {
        evolution: publicModel(resolveEvolutionConfig(loaded.config))
      },
      protocol: {
        boundaryWindowSize: 15,
        boundaryOverlap: 5,
        decision: "0=same subproblem, 1=new subproblem"
      },
      summary: {
        selectedEpisodes: episodes.length,
        succeeded: runs.filter((run) => !run.error).length,
        failed: runs.filter((run) => run.error).length,
        totalSteps: sum(runs.map((run) => run.steps?.length ?? 0)),
        totalSegments: sum(runs.map((run) => run.segmentation?.segments.length ?? 0)),
        expectedCountMatches: runs.filter((run) =>
          run.expected && run.segmentation &&
          run.expected.expectedSegmentCount === run.segmentation.segments.length).length
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
        episodeId: run.metadata.episodeId,
        steps: run.steps?.length,
        segments: run.segmentation?.segments.length,
        boundaryBeforeStepIndices: run.segmentation?.decisions
          .filter((decision) => decision.boundary === 1)
          .map((decision) => decision.beforeStepIndex),
        expectedSegments: run.expected?.expectedSegmentCount,
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
  models: Record<string, unknown>;
  protocol: Record<string, unknown>;
  summary: Record<string, unknown>;
  runs: EpisodeRun[];
}): string {
  const lines = [
    "# KW Episode 15/5 二元边界切分实验",
    "",
    `- Started: ${artifact.startedAt}`,
    `- Finished: ${artifact.finishedAt}`,
    `- Elapsed: ${Math.round(artifact.elapsedMs / 1000)}s`,
    `- Source DB: \`${artifact.sourceDatabase}\``,
    `- Output DB: \`${artifact.outputDatabase}\``,
    `- Concurrency: ${artifact.concurrency}`,
    "",
    "## Protocol",
    "",
    "```json",
    JSON.stringify(artifact.protocol, null, 2),
    "```",
    "",
    "## Model",
    "",
    "```json",
    JSON.stringify(artifact.models, null, 2),
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
      `## ${run.metadata.episodeId}`,
      "",
      `- Turns: ${run.metadata.turnCount}`,
      `- Tool calls: ${run.metadata.toolCallCount}`,
      `- Reward: ${run.metadata.terminalReward ?? "unknown"}`,
      `- Reconstructed Steps: ${run.steps?.length ?? 0}`,
      `- Boundary windows: ${run.segmentation?.windows.length ?? 0}`,
      `- Observed Segments: ${run.segmentation?.segments.length ?? 0}`,
      `- Expected Segments: ${run.expected?.expectedSegmentCount ?? "not annotated"}`,
      `- Reference rationale: ${run.expected?.rationale ?? "not annotated"}`,
      `- Elapsed: ${Math.round(run.elapsedMs / 1000)}s`,
      `- Result: ${run.error ? `failed — ${run.error.split("\n")[0]}` : "succeeded"}`,
      ""
    );
    if (!run.segmentation || !run.steps) continue;
    lines.push(
      "### Episode Contract",
      "",
      "```json",
      JSON.stringify(run.segmentation.episodeContract, null, 2),
      "```",
      "",
      `Boundary before Step: ${run.segmentation.decisions
        .filter((decision) => decision.boundary === 1)
        .map((decision) => decision.beforeStepIndex)
        .join(", ") || "none"}`,
      ""
    );
    for (const segment of run.segmentation.segments) {
      lines.push(
        `### Segment ${segment.segmentIndex + 1}: Steps ${segment.startStepIndex}–${segment.endStepIndex}`,
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

function loadEpisodeMetadata(dbPath: string, episodeIds: readonly string[]): EpisodeMetadata[] {
  const db = new Database(resolve(dbPath), { readonly: true, fileMustExist: true });
  try {
    const query = db.prepare(
      `SELECT episodes.id AS episode_id,
              episodes.turn_count,
              episodes.r_task,
              COALESCE(SUM(json_array_length(raw_turns.tool_calls_json)), 0) AS tool_call_count
       FROM episodes
       LEFT JOIN raw_turns ON raw_turns.episode_id = episodes.id
       WHERE episodes.id = ? AND episodes.status = 'closed'
       GROUP BY episodes.id`
    );
    return episodeIds.map((episodeId) => {
      const row = query.get(episodeId) as {
        episode_id: string;
        turn_count: number;
        r_task: number | null;
        tool_call_count: number;
      } | undefined;
      if (!row) throw new Error(`closed KW Episode not found: ${episodeId}`);
      return {
        episodeId: row.episode_id,
        turnCount: row.turn_count,
        toolCallCount: row.tool_call_count,
        ...(row.r_task === null ? {} : { terminalReward: row.r_task })
      };
    });
  } finally {
    db.close();
  }
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
  let concurrency = 2;
  let episodeIds = [...DEFAULT_EPISODE_IDS];
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--db") dbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--config") configPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output-db") outputDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output") outputReportPath = requiredValue(argv, ++index, arg);
    else if (arg === "--concurrency") {
      concurrency = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--episode-ids") {
      episodeIds = requiredValue(argv, ++index, arg).split(",")
        .map((value) => value.trim()).filter(Boolean);
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write(helpText());
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!dbPath) throw new Error("--db is required");
  if (episodeIds.length === 0) throw new Error("--episode-ids requires at least one Episode ID");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  const base = resolve("../../experiments/results/trace2skill-segment-boundary-kw");
  return {
    dbPath: resolve(dbPath),
    ...(configPath ? { configPath: resolve(configPath) } : {}),
    outputDbPath: resolve(outputDbPath || `${base}/${stamp}-kw-boundary.sqlite`),
    outputReportPath: resolve(outputReportPath || `${base}/${stamp}-kw-boundary.json`),
    concurrency,
    episodeIds: [...new Set(episodeIds)]
  };
}

function helpText(): string {
  return "Usage: npm run segment-boundary:kw -- --db <kw.sqlite> [--config <config.yaml>] [--episode-ids <id1,id2,...>] [--concurrency 2] [--output-db <copy.sqlite>] [--output <report.json>]\n";
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
