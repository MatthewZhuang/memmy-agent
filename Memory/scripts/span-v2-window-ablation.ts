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
  buildCompressedEventBudgetSegments,
  buildFixedToolCallSegments,
  buildHybridBudgetSegments,
  eventsForToolCallRange,
  type Segment
} from "./span-v2-segmentation.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const ARTIFACTS_ROOT = join(REPO_ROOT, "artifacts");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const OUT_JSON = join(OUT_DIR, "span_v2_window_ablation.json");
const OUT_MD = join(OUT_DIR, "span_v2_window_ablation.md");

const SAMPLE_LIMIT = positiveIntegerEnv("SPAN_V2_WINDOW_ABLATION_SAMPLE_LIMIT") ?? 16;
const PER_BUCKET_LIMIT = positiveIntegerEnv("SPAN_V2_WINDOW_ABLATION_PER_BUCKET") ?? 4;
const MIN_TOOL_CALLS = positiveIntegerEnv("SPAN_V2_WINDOW_ABLATION_MIN_TOOL_CALLS") ?? 4;
const SAMPLE_SEED = process.env.SPAN_V2_WINDOW_ABLATION_SEED ?? "span-v2-window-ablation-20260817";
const RAW_WINDOW_SIZE = positiveIntegerEnv("SPAN_V2_RAW_WINDOW_SIZE") ?? 50;
const RAW_WINDOW_OVERLAP = positiveIntegerEnv("SPAN_V2_RAW_WINDOW_OVERLAP") ?? 10;
const EVENT_WINDOW_SIZE = positiveIntegerEnv("SPAN_V2_EVENT_WINDOW_SIZE") ?? 24;
const EVENT_WINDOW_OVERLAP = positiveIntegerEnv("SPAN_V2_EVENT_WINDOW_OVERLAP") ?? 4;
const HYBRID_RAW_MAX = positiveIntegerEnv("SPAN_V2_HYBRID_RAW_MAX") ?? 60;
const HYBRID_EVENT_MAX = positiveIntegerEnv("SPAN_V2_HYBRID_EVENT_MAX") ?? 24;
const HYBRID_TEXT_MAX = positiveIntegerEnv("SPAN_V2_HYBRID_TEXT_MAX") ?? 30_000;
const HYBRID_EVENT_OVERLAP = positiveIntegerEnv("SPAN_V2_HYBRID_EVENT_OVERLAP") ?? 4;

type MethodId = "direct" | "raw_fixed_50" | "compressed_event_budget" | "hybrid_budget";

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
  method: MethodId;
  label: string;
  traceResults: MethodTraceResult[];
  metrics: ReturnType<typeof summarizeMethod>;
}

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
  const methods: Array<{ id: MethodId; label: string; segments: (trace: ArtifactTraceCase, trajectory: SpanTrajectoryEvent[]) => Segment[] }> = [
    {
      id: "direct",
      label: "Direct LLM：整条压缩 structured trajectory",
      segments: (trace) => [{ start: 0, end: trace.toolCalls.length - 1, reason: "direct" }]
    },
    {
      id: "raw_fixed_50",
      label: `Raw fixed：${RAW_WINDOW_SIZE} tool calls + overlap ${RAW_WINDOW_OVERLAP}`,
      segments: (trace) => buildFixedToolCallSegments(
        trace.toolCalls.length,
        RAW_WINDOW_SIZE,
        RAW_WINDOW_OVERLAP,
        "raw_fixed"
      )
    },
    {
      id: "compressed_event_budget",
      label: `Compressed event budget：${EVENT_WINDOW_SIZE} events + overlap ${EVENT_WINDOW_OVERLAP}`,
      segments: (_trace, trajectory) => buildCompressedEventBudgetSegments(trajectory, {
        maxEvents: EVENT_WINDOW_SIZE,
        overlapEvents: EVENT_WINDOW_OVERLAP
      })
    },
    {
      id: "hybrid_budget",
      label: `Hybrid budget：raw<=${HYBRID_RAW_MAX}, events<=${HYBRID_EVENT_MAX}, text<=${HYBRID_TEXT_MAX}`,
      segments: (_trace, trajectory) => buildHybridBudgetSegments(trajectory, {
        maxRawToolCalls: HYBRID_RAW_MAX,
        maxEvents: HYBRID_EVENT_MAX,
        maxTextChars: HYBRID_TEXT_MAX,
        overlapEvents: HYBRID_EVENT_OVERLAP
      })
    }
  ];

  const results: MethodResult[] = [];
  for (const method of methods) {
    const traceResults: MethodTraceResult[] = [];
    for (const trace of sample) {
      const trajectory = buildSpanTrajectory(trace.toolCalls);
      traceResults.push(await runMethod(trace, trajectory, method, llm));
    }
    results.push({
      method: method.id,
      label: method.label,
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
      rawWindowSize: RAW_WINDOW_SIZE,
      rawWindowOverlap: RAW_WINDOW_OVERLAP,
      eventWindowSize: EVENT_WINDOW_SIZE,
      eventWindowOverlap: EVENT_WINDOW_OVERLAP,
      hybridRawMax: HYBRID_RAW_MAX,
      hybridEventMax: HYBRID_EVENT_MAX,
      hybridTextMax: HYBRID_TEXT_MAX,
      hybridEventOverlap: HYBRID_EVENT_OVERLAP
    },
    candidateTraceCount: allTraces.length,
    candidateDistribution: distribution(allTraces),
    sampleTraceCount: sample.length,
    sampleDistribution: distribution(sample),
    sampleTraces: sample.map((trace) => ({
      traceId: trace.traceId,
      sourceKind: trace.sourceKind,
      sourcePath: trace.sourcePath,
      toolCalls: trace.toolCalls.length,
      lengthBucket: traceLengthBucket(trace.toolCalls.length),
      userPreview: clip(redactSensitiveText(trace.userText), 160)
    })),
    results
  };
  writeFileSync(OUT_JSON, JSON.stringify(output, null, 2));
  writeFileSync(OUT_MD, renderReport(output));
}

async function runMethod(
  trace: ArtifactTraceCase,
  trajectory: SpanTrajectoryEvent[],
  method: {
    id: MethodId;
    segments: (trace: ArtifactTraceCase, trajectory: SpanTrajectoryEvent[]) => Segment[];
  },
  llm: LlmClient
): Promise<MethodTraceResult> {
  const segments = method.segments(trace, trajectory)
    .filter((segment) => segment.end - segment.start + 1 > 3);
  const allSpans: SpanDraft[] = [];
  const failures: string[] = [];
  let promptCalls = 0;
  let promptChars = 0;
  for (const segment of segments) {
    const payload = splitPayload(trace, trajectory, segment);
    const content = stableStringify(payload);
    promptCalls += 1;
    promptChars += SPAN_SPLIT_SYSTEM.length + content.length;
    try {
      const result = await llm.completeJson<{ spans?: unknown; reason?: unknown }>([
        { role: "system", content: SPAN_SPLIT_SYSTEM },
        { role: "user", content }
      ], {
        operation: `span.window_ablation.${method.id}`,
        thinkingMode: "disabled",
        temperature: 0.2,
        maxTokens: 4096,
        timeoutMs: 180_000,
        maxRetries: 2,
        malformedRetries: 2
      });
      allSpans.push(...parseSpanDrafts(result, trace.toolCalls.length)
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
    spans: resolveOverlaps(allSpans),
    failures
  };
}

function splitPayload(
  trace: ArtifactTraceCase,
  trajectory: SpanTrajectoryEvent[],
  segment: Segment
): Record<string, unknown> {
  return {
    traceId: trace.traceId,
    sourceKind: trace.sourceKind,
    methodSegment: segment,
    userRequest: redactAndClip(trace.userText, 1_500),
    assistantFinalAnswer: redactAndClip(trace.assistantText, 1_000),
    structuredTrajectory: eventsForToolCallRange(trajectory, segment)
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
    avgPromptCharsPerTrace: round(traceResults.reduce((sum, result) => sum + result.promptChars, 0) / Math.max(1, traceResults.length)),
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

function renderReport(output: {
  generatedAt: string;
  dataSources: string[];
  llm: Record<string, unknown>;
  settings: Record<string, unknown>;
  candidateTraceCount: number;
  candidateDistribution: ReturnType<typeof distribution>;
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
    "# Span 窗口策略消融实验",
    "",
    `生成时间：${output.generatedAt}`,
    "",
    "## 数据来源",
    "",
    ...output.dataSources.map((source) => `- ${source}`),
    "",
    `候选 trace：${output.candidateTraceCount} 条；抽样 trace：${output.sampleTraceCount} 条。`,
    "",
    "候选分布：",
    "",
    `- 来源：training=${output.candidateDistribution.source.training_sqlite}，test=${output.candidateDistribution.source.test_jsonl}`,
    `- 长度：small=${output.candidateDistribution.lengthBucket.small}，medium=${output.candidateDistribution.lengthBucket.medium}，large=${output.candidateDistribution.lengthBucket.large}，xlarge=${output.candidateDistribution.lengthBucket.xlarge}`,
    "",
    "抽样分布：",
    "",
    `- 来源：training=${output.sampleDistribution.source.training_sqlite}，test=${output.sampleDistribution.source.test_jsonl}`,
    `- 长度：small=${output.sampleDistribution.lengthBucket.small}，medium=${output.sampleDistribution.lengthBucket.medium}，large=${output.sampleDistribution.lengthBucket.large}，xlarge=${output.sampleDistribution.lengthBucket.xlarge}`,
    "",
    "抽样方法：按 tool-call 数分为 small(4-15)、medium(16-40)、large(41-80)、xlarge(>80)，每层最多抽同等数量，并用稳定 seed 做确定性抽样。这样避免只在超长 trace 上得出结论。",
    "",
    "## 设置",
    "",
    `- LLM：provider=${output.llm.provider}，model=${output.llm.model}，endpoint=${output.llm.endpoint}`,
    `- sampleLimit=${output.settings.sampleLimit}，perBucket=${output.settings.perBucketLimit}，seed=${output.settings.sampleSeed}`,
    `- raw fixed：window=${output.settings.rawWindowSize}，overlap=${output.settings.rawWindowOverlap}`,
    `- compressed event：window=${output.settings.eventWindowSize}，overlap=${output.settings.eventWindowOverlap}`,
    `- hybrid：raw<=${output.settings.hybridRawMax}，events<=${output.settings.hybridEventMax}，text<=${output.settings.hybridTextMax}，overlap=${output.settings.hybridEventOverlap}`,
    "",
    "## 指标对比",
    "",
    "| 方法 | spans | avg spans/trace | avg calls/span | max calls/span | >=50 | >=100 | mixed-stage | coverage | prompt calls | prompt chars | failures |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...output.results.map((result) => {
      const m = result.metrics;
      return `| ${result.label} | ${m.spanCount} | ${m.avgSpansPerTrace} | ${m.avgCallsPerSpan} | ${m.maxCallsPerSpan} | ${m.longSpanCount} | ${m.veryLongSpanCount} | ${m.mixedStageCount} | ${pct(m.coverageRate)} | ${m.promptCalls} | ${m.promptChars} | ${m.failureCount} |`;
    }),
    "",
    "## 分层结果",
    "",
    "| 方法 | small long/fail | medium long/fail | large long/fail | xlarge long/fail |",
    "| --- | ---: | ---: | ---: | ---: |",
    ...output.results.map((result) => {
      const b = result.metrics.byBucket;
      return `| ${result.label} | ${b.small.longSpanCount}/${b.small.failureCount} | ${b.medium.longSpanCount}/${b.medium.failureCount} | ${b.large.longSpanCount}/${b.large.failureCount} | ${b.xlarge.longSpanCount}/${b.xlarge.failureCount} |`;
    }),
    "",
    "## 样本列表",
    "",
    "| trace | source | calls | bucket | preview |",
    "| --- | --- | ---: | --- | --- |",
    ...output.sampleTraces.map((trace) =>
      `| ${trace.traceId} | ${trace.sourceKind} | ${trace.toolCalls} | ${trace.lengthBucket} | ${trace.userPreview.replace(/\|/gu, "/").replace(/\s+/gu, " ")} |`
    ),
    "",
    "## 结论",
    "",
    ...renderConclusion(output.results),
    "",
    "## Trace 级样例",
    ""
  ];
  for (const trace of output.sampleTraces.slice(0, 8)) {
    lines.push(`### ${trace.traceId}`, "");
    for (const result of output.results) {
      const item = result.traceResults.find((candidate) => candidate.traceId === trace.traceId);
      if (!item) continue;
      lines.push(`- ${result.label}`);
      lines.push(`  - segments: ${item.segments.map((segment) => `${segment.start}-${segment.end}`).join(", ")}`);
      lines.push(`  - spans: ${item.spans.length === 0 ? "none" : item.spans.map((span) => `${span.start}-${span.end}(${span.end - span.start + 1}) ${span.goal}`).join("；")}`);
      if (item.failures.length > 0) lines.push(`  - failures: ${item.failures.join(" | ")}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

function renderConclusion(results: MethodResult[]): string[] {
  const ranked = [...results].sort((a, b) =>
    a.metrics.longSpanCount - b.metrics.longSpanCount ||
    a.metrics.failureCount - b.metrics.failureCount ||
    a.metrics.mixedStageCount - b.metrics.mixedStageCount ||
    a.metrics.promptChars - b.metrics.promptChars
  );
  const best = ranked[0];
  if (!best) return ["- 本报告用于比较窗口策略，请结合指标和 trace 样例人工审核。"];
  return [
    `- 按“先减少超长 span，再看失败数和 mixed-stage，最后看成本”的口径，当前最优候选是：${best.label}。`,
    "- Direct LLM 是最低成本基线，但如果出现 >=50 或 >=100 的超长 span，就不能作为默认生产路径。",
    "- Raw fixed、compressed event budget、hybrid budget 都只是 LLM 输入护栏，不是最终 span 边界；最终边界仍由 LLM 在窗口内抽取。",
    "- 若 hybrid 的超长 span 与失败数不高于 raw fixed，同时 prompt 成本更低或 mixed-stage 更少，就应优先考虑 hybrid；否则保留 raw fixed 作为稳定 baseline。"
  ];
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
