import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  createLlmClient,
  DEFAULT_MEMMY_CONFIG,
  type LlmClient,
  type LlmConfig
} from "../src/index.js";
import { parseSpanDrafts, type SpanDraft } from "../src/service/evolution/span-model.js";
import { buildSpanTrajectory, type SpanTrajectoryEvent } from "../src/service/evolution/span-trajectory.js";
import { stableStringify } from "../src/utils/id.js";
import { redactSensitiveText } from "../src/utils/sensitive-data.js";
import { clip } from "../src/utils/text.js";
import {
  loadArtifactTraceCases,
  stratifiedTraceSample,
  traceLengthBucket,
  type ArtifactTraceCase,
  type TraceLengthBucket
} from "./span-v2-artifact-traces.js";
import {
  buildFixedToolCallSegments,
  eventsForToolCallRange,
  type Segment
} from "./span-v2-segmentation.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const ARTIFACTS_ROOT = join(REPO_ROOT, "artifacts");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const OUT_JSON = join(OUT_DIR, "span_v2_raw_window_threshold_ablation.json");
const OUT_MD = join(OUT_DIR, "span_v2_raw_window_threshold_ablation.md");

const SAMPLE_LIMIT = positiveIntegerEnv("SPAN_V2_RAW_THRESHOLD_SAMPLE_LIMIT") ?? 16;
const PER_BUCKET_LIMIT = positiveIntegerEnv("SPAN_V2_RAW_THRESHOLD_PER_BUCKET") ?? 4;
const MIN_TOOL_CALLS = positiveIntegerEnv("SPAN_V2_RAW_THRESHOLD_MIN_TOOL_CALLS") ?? 4;
const SAMPLE_SEED = process.env.SPAN_V2_RAW_THRESHOLD_SEED ?? "span-v2-window-ablation-20260817";

interface ThresholdConfig {
  windowSize: number;
  overlap: number;
}

interface MethodTraceResult {
  traceId: string;
  sourceKind: ArtifactTraceCase["sourceKind"];
  lengthBucket: TraceLengthBucket;
  rawToolCalls: number;
  compressedEvents: number;
  promptCalls: number;
  promptChars: number;
  segments: Segment[];
  spans: SpanDraft[];
  failures: string[];
}

interface MethodResult {
  method: string;
  label: string;
  windowSize: number;
  overlap: number;
  traceResults: MethodTraceResult[];
  metrics: ReturnType<typeof summarizeMethod>;
}

const THRESHOLDS: ThresholdConfig[] = [
  { windowSize: 30, overlap: 6 },
  { windowSize: 50, overlap: 10 },
  { windowSize: 80, overlap: 16 }
];

const SPAN_SPLIT_SYSTEM = `You extract reusable local strategy spans from one completed AI-agent trajectory.

You are given a programmatic structured trajectory, not raw full logs. Treat it
as the source of truth. It preserves original tool-call indexes, tool/action
kinds, success/error signals, input/output shapes, artifact signals, and short
redacted evidence snippets.

A span represents one extractable local strategy: one concrete goal pursued
through one coherent observed policy over a contiguous sequence of tool calls.

Boundary rules:
- Every span uses original inclusive tool-call indexes from the input.
- Every span must cover more than three tool calls.
- Spans must not overlap and must remain in execution order.
- Tool calls that are repetitive, transitional, or not reusable may remain
  outside all spans.
- Do not invent actions, results, goals, errors, files, or entities not
  supported by the structured trajectory.
- If a candidate policy needs "and then" to connect independent phases, split
  the span unless those phases are inseparable inside one debug/generation loop.

For goal and policy:
- Describe the reusable subtask objective and observed strategy.
- Prefer task type, artifact type, data shape, tool/library family, error class,
  query pattern, transformation, debugging loop, and verification signal.
- Avoid one-off task names, customer names, file paths, brands, or broad
  policies like "complete the task" or "research and write a report".
- Use the same language as the user's request.

Return strict JSON only:
{
  "reason": "...",
  "spans": [
    {"start": 0, "end": 3, "goal": "...", "policy": "...", "summary": "..."}
  ]
}`;

async function main(): Promise<void> {
  mkdirSync(OUT_DIR, { recursive: true });
  const llm = createEvalLlm();
  const allTraces = loadArtifactTraceCases({
    artifactsRoot: ARTIFACTS_ROOT,
    minToolCalls: MIN_TOOL_CALLS
  });
  const sample = stratifiedTraceSample(allTraces, {
    totalLimit: SAMPLE_LIMIT,
    perBucketLimit: PER_BUCKET_LIMIT,
    seed: SAMPLE_SEED
  });
  const results: MethodResult[] = [];
  for (const config of THRESHOLDS) {
    const traceResults: MethodTraceResult[] = [];
    for (const trace of sample) {
      const trajectory = buildSpanTrajectory(trace.toolCalls);
      traceResults.push(await runThreshold(trace, trajectory, config, llm));
    }
    results.push({
      method: `raw_fixed_${config.windowSize}`,
      label: `Raw fixed ${config.windowSize} + overlap ${config.overlap}`,
      windowSize: config.windowSize,
      overlap: config.overlap,
      traceResults,
      metrics: summarizeMethod(traceResults)
    });
  }
  const output = {
    generatedAt: new Date().toISOString(),
    dataSources: [
      "artifacts/memory-v10-training-20260804/memory.sqlite",
      "artifacts/memory-v10-test-20260804/*/session.jsonl"
    ],
    llm: publicModelConfig(llm.config),
    settings: {
      minToolCalls: MIN_TOOL_CALLS,
      sampleLimit: SAMPLE_LIMIT,
      perBucketLimit: PER_BUCKET_LIMIT,
      sampleSeed: SAMPLE_SEED,
      thresholds: THRESHOLDS
    },
    candidateTraceCount: allTraces.length,
    sampleTraceCount: sample.length,
    sampleDistribution: distribution(sample),
    sampleTraces: sample.map((trace) => ({
      traceId: trace.traceId,
      sourceKind: trace.sourceKind,
      toolCalls: trace.toolCalls.length,
      lengthBucket: traceLengthBucket(trace.toolCalls.length),
      userPreview: clip(redactSensitiveText(trace.userText), 160)
    })),
    results
  };
  writeFileSync(OUT_JSON, JSON.stringify(output, null, 2));
  writeFileSync(OUT_MD, renderReport(output));
}

async function runThreshold(
  trace: ArtifactTraceCase,
  trajectory: SpanTrajectoryEvent[],
  config: ThresholdConfig,
  llm: LlmClient
): Promise<MethodTraceResult> {
  const segments = buildFixedToolCallSegments(
    trace.toolCalls.length,
    config.windowSize,
    config.overlap,
    `raw_fixed_${config.windowSize}`
  ).filter((segment) => segment.end - segment.start + 1 > 3);
  const spans: SpanDraft[] = [];
  const failures: string[] = [];
  let promptCalls = 0;
  let promptChars = 0;
  for (const segment of segments) {
    const content = stableStringify({
      traceId: trace.traceId,
      sourceKind: trace.sourceKind,
      methodSegment: segment,
      userRequest: redactAndClip(trace.userText, 1_500),
      assistantFinalAnswer: redactAndClip(trace.assistantText, 1_000),
      structuredTrajectory: eventsForToolCallRange(trajectory, segment)
    });
    promptCalls += 1;
    promptChars += SPAN_SPLIT_SYSTEM.length + content.length;
    try {
      const result = await llm.completeJson<{ spans?: unknown; reason?: unknown }>([
        { role: "system", content: SPAN_SPLIT_SYSTEM },
        { role: "user", content }
      ], {
        operation: `span.raw_threshold.${config.windowSize}`,
        thinkingMode: "disabled",
        temperature: 0.2,
        maxTokens: 4096,
        timeoutMs: 180_000,
        maxRetries: 2,
        malformedRetries: 2
      });
      spans.push(...parseSpanDrafts(result, trace.toolCalls.length)
        .filter((span) => span.start >= segment.start && span.end <= segment.end));
    } catch (error) {
      failures.push(`${segment.start}-${segment.end}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return {
    traceId: trace.traceId,
    sourceKind: trace.sourceKind,
    lengthBucket: traceLengthBucket(trace.toolCalls.length),
    rawToolCalls: trace.toolCalls.length,
    compressedEvents: trajectory.length,
    promptCalls,
    promptChars,
    segments,
    spans: resolveOverlaps(spans),
    failures
  };
}

function resolveOverlaps(spans: SpanDraft[]): SpanDraft[] {
  const sorted = [...spans]
    .filter((span) => span.end - span.start + 1 > 3)
    .sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));
  const out: SpanDraft[] = [];
  for (const span of sorted) {
    const previous = out.at(-1);
    if (!previous || span.start > previous.end) {
      out.push(span);
      continue;
    }
    const overlap = Math.min(previous.end, span.end) - Math.max(previous.start, span.start) + 1;
    const previousLength = previous.end - previous.start + 1;
    const currentLength = span.end - span.start + 1;
    if (overlap / Math.min(previousLength, currentLength) >= 0.6 && currentLength > previousLength) {
      out[out.length - 1] = span;
      continue;
    }
    if (span.end > previous.end) {
      const trimmed = { ...span, start: previous.end + 1 };
      if (trimmed.end - trimmed.start + 1 > 3) out.push(trimmed);
    }
  }
  return out;
}

function summarizeMethod(traceResults: MethodTraceResult[]) {
  const spans = traceResults.flatMap((result) => result.spans);
  const rawToolCalls = traceResults.reduce((sum, result) => sum + result.rawToolCalls, 0);
  return {
    traceCount: traceResults.length,
    promptCalls: traceResults.reduce((sum, result) => sum + result.promptCalls, 0),
    promptChars: traceResults.reduce((sum, result) => sum + result.promptChars, 0),
    spanCount: spans.length,
    avgSpansPerTrace: round(spans.length / Math.max(1, traceResults.length)),
    avgCallsPerSpan: round(spans.reduce((sum, span) => sum + span.end - span.start + 1, 0) / Math.max(1, spans.length)),
    maxCallsPerSpan: Math.max(0, ...spans.map((span) => span.end - span.start + 1)),
    longSpanCount: spans.filter((span) => span.end - span.start + 1 >= 50).length,
    veryLongSpanCount: spans.filter((span) => span.end - span.start + 1 >= 100).length,
    mixedStageCount: spans.filter((span) => isMixedStage(span.goal, span.policy)).length,
    coverageRate: round(traceResults.reduce((sum, result) => sum + coveredToolCalls(result.spans), 0) / Math.max(1, rawToolCalls)),
    failureCount: traceResults.reduce((sum, result) => sum + result.failures.length, 0),
    byBucket: Object.fromEntries((["small", "medium", "large", "xlarge"] as const).map((bucket) => {
      const items = traceResults.filter((result) => result.lengthBucket === bucket);
      const bucketSpans = items.flatMap((result) => result.spans);
      return [bucket, {
        traceCount: items.length,
        spanCount: bucketSpans.length,
        maxCallsPerSpan: Math.max(0, ...bucketSpans.map((span) => span.end - span.start + 1)),
        longSpanCount: bucketSpans.filter((span) => span.end - span.start + 1 >= 50).length,
        failureCount: items.reduce((sum, result) => sum + result.failures.length, 0)
      }];
    }))
  };
}

function distribution(traces: readonly ArtifactTraceCase[]) {
  return {
    total: traces.length,
    source: Object.fromEntries((["training_sqlite", "test_jsonl"] as const).map((source) => [
      source,
      traces.filter((trace) => trace.sourceKind === source).length
    ])),
    lengthBucket: Object.fromEntries((["small", "medium", "large", "xlarge"] as const).map((bucket) => [
      bucket,
      traces.filter((trace) => traceLengthBucket(trace.toolCalls.length) === bucket).length
    ]))
  };
}

function renderReport(output: {
  generatedAt: string;
  dataSources: string[];
  llm: Record<string, unknown>;
  settings: Record<string, unknown>;
  candidateTraceCount: number;
  sampleTraceCount: number;
  sampleDistribution: ReturnType<typeof distribution>;
  sampleTraces: Array<{
    traceId: string;
    sourceKind: string;
    toolCalls: number;
    lengthBucket: string;
    userPreview: string;
  }>;
  results: MethodResult[];
}): string {
  const lines = [
    "# Raw Fixed 窗口阈值消融实验",
    "",
    `生成时间：${output.generatedAt}`,
    "",
    "## 数据来源与抽样",
    "",
    ...output.dataSources.map((source) => `- ${source}`),
    "",
    `候选 trace：${output.candidateTraceCount} 条；抽样 trace：${output.sampleTraceCount} 条。`,
    `抽样分布：training=${output.sampleDistribution.source.training_sqlite}，test=${output.sampleDistribution.source.test_jsonl}；small=${output.sampleDistribution.lengthBucket.small}，medium=${output.sampleDistribution.lengthBucket.medium}，large=${output.sampleDistribution.lengthBucket.large}，xlarge=${output.sampleDistribution.lengthBucket.xlarge}`,
    "",
    "本实验固定使用压缩 structured trajectory，只改变 raw tool-call 窗口阈值，比较 30/50/80 三组设置。",
    "",
    "## 设置",
    "",
    `- LLM：provider=${output.llm.provider}，model=${output.llm.model}，endpoint=${output.llm.endpoint}`,
    `- sampleLimit=${output.settings.sampleLimit}，perBucket=${output.settings.perBucketLimit}，seed=${output.settings.sampleSeed}`,
    "",
    "## 指标对比",
    "",
    "| 窗口 | spans | avg spans/trace | avg calls/span | max calls/span | >=50 | >=100 | mixed-stage | coverage | prompt calls | prompt chars | failures |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...output.results.map((result) => {
      const m = result.metrics;
      return `| ${result.label} | ${m.spanCount} | ${m.avgSpansPerTrace} | ${m.avgCallsPerSpan} | ${m.maxCallsPerSpan} | ${m.longSpanCount} | ${m.veryLongSpanCount} | ${m.mixedStageCount} | ${pct(m.coverageRate)} | ${m.promptCalls} | ${m.promptChars} | ${m.failureCount} |`;
    }),
    "",
    "## 分层结果",
    "",
    "| 窗口 | small long/fail | medium long/fail | large long/fail | xlarge long/fail |",
    "| --- | ---: | ---: | ---: | ---: |",
    ...output.results.map((result) => {
      const b = result.metrics.byBucket;
      return `| ${result.label} | ${b.small.longSpanCount}/${b.small.failureCount} | ${b.medium.longSpanCount}/${b.medium.failureCount} | ${b.large.longSpanCount}/${b.large.failureCount} | ${b.xlarge.longSpanCount}/${b.xlarge.failureCount} |`;
    }),
    "",
    "## 结论",
    "",
    ...renderConclusion(output.results),
    "",
    "## 样本列表",
    "",
    "| trace | source | calls | bucket | preview |",
    "| --- | --- | ---: | --- | --- |",
    ...output.sampleTraces.map((trace) =>
      `| ${trace.traceId} | ${trace.sourceKind} | ${trace.toolCalls} | ${trace.lengthBucket} | ${trace.userPreview.replace(/\|/gu, "/").replace(/\s+/gu, " ")} |`
    ),
    ""
  ];
  return lines.join("\n");
}

function renderConclusion(results: MethodResult[]): string[] {
  const byWindow = new Map(results.map((result) => [result.windowSize, result]));
  const w30 = byWindow.get(30);
  const w50 = byWindow.get(50);
  const w80 = byWindow.get(80);
  const lines: string[] = [];
  if (w30 && w50 && w80) {
    lines.push(
      `- window=30 调用成本最高（prompt calls=${w30.metrics.promptCalls}），但 max span=${w30.metrics.maxCallsPerSpan}，用于强约束粒度；如果 failure 或 mixed-stage 没有明显下降，则不值得作为默认值。`,
      `- window=50 是中间折中：成本低于 30，高于 80；如果 >=50 和 mixed-stage 接近最优，可以作为默认护栏。`,
      `- window=80 成本最低（prompt calls=${w80.metrics.promptCalls}），但如果 max span 或 >=50 上升，说明窗口过宽，会削弱控制长 span 的目的。`
    );
  }
  const ranked = [...results].sort((a, b) =>
    a.metrics.longSpanCount - b.metrics.longSpanCount ||
    a.metrics.failureCount - b.metrics.failureCount ||
    a.metrics.mixedStageCount - b.metrics.mixedStageCount ||
    a.metrics.promptChars - b.metrics.promptChars
  );
  if (ranked[0]) {
    lines.push(`- 按“优先减少 >=50 span，再看失败数、mixed-stage 和成本”的口径，本轮最优候选是：${ranked[0].label}。`);
  }
  return lines;
}

function coveredToolCalls(spans: readonly SpanDraft[]): number {
  const indexes = new Set<number>();
  for (const span of spans) {
    for (let index = span.start; index <= span.end; index += 1) indexes.add(index);
  }
  return indexes.size;
}

function isMixedStage(goal: string, policy: string): boolean {
  const value = `${goal}\n${policy}`.toLowerCase();
  const signals = [
    /gather|search|fetch|research|collect/u,
    /inspect|schema|load|parse|read/u,
    /generate|create|write|build|populate/u,
    /debug|fix|repair|error|traceback|rerun/u,
    /verify|validate|check|confirm|extract text/u
  ];
  return signals.filter((signal) => signal.test(value)).length >= 3;
}

function createEvalLlm(): LlmClient {
  return createLlmClient({
    ...DEFAULT_MEMMY_CONFIG.evolution,
    provider: "openai_compatible",
    vendor: "deepseek",
    endpoint: process.env.SPAN_V2_LLM_ENDPOINT ?? "https://api-int.memtensor.cn/v1",
    model: process.env.SPAN_V2_LLM_MODEL ?? "deepseek-v4-pro",
    apiKey: requiredSecret("SPAN_V2_LLM_API_KEY", "MEMMY_EVOLUTION_API_KEY", "LLM_API_KEY"),
    enableThinking: false,
    temperature: 0.2,
    maxTokens: 4096,
    timeoutMs: 180_000,
    maxRetries: 2,
    malformedRetries: 2
  });
}

function publicModelConfig(config: Pick<LlmConfig, "provider" | "endpoint" | "model">) {
  return {
    provider: config.provider,
    endpoint: config.endpoint,
    model: config.model
  };
}

function requiredSecret(...names: string[]): string {
  for (const name of names) {
    const value = process.env[name]?.trim();
    if (value) return value;
  }
  throw new Error(`Missing LLM API key. Set one of: ${names.join(", ")}`);
}

function redactAndClip(value: string, maxChars: number): string {
  return clip(redactSensitiveText(value), maxChars);
}

function positiveIntegerEnv(name: string): number | undefined {
  const value = Number(process.env[name]);
  return Number.isInteger(value) && value > 0 ? value : undefined;
}

function round(value: number): number {
  return Number(value.toFixed(2));
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

await main();
