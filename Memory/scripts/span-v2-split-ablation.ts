import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  createEmbedder,
  createLlmClient,
  DEFAULT_MEMMY_CONFIG,
  MemoryDb,
  type Embedder,
  type LlmClient,
  type LlmConfig,
  type MemoryRow,
  type RawTurnRecord
} from "../src/index.js";
import { Repositories } from "../src/storage/repositories.js";
import { parseSpanDrafts, type SpanDraft } from "../src/service/evolution/span-model.js";
import { buildSpanTrajectory, type SpanTrajectoryEvent } from "../src/service/evolution/span-trajectory.js";
import { stableStringify } from "../src/utils/id.js";
import { isRecord } from "../src/utils/json.js";
import { redactSensitiveText } from "../src/utils/sensitive-data.js";
import { clip } from "../src/utils/text.js";
import {
  buildFixedToolCallSegments,
  eventsForToolCallRange,
  trajectoryText,
  type Segment
} from "./span-v2-segmentation.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const SOURCE_DB = join(OUT_DIR, "span_v2_experiment.sqlite");
const RESULT_JSON = join(OUT_DIR, "span_v2_experiment_results.json");
const OUT_JSON = join(OUT_DIR, "span_v2_split_ablation.json");
const OUT_MD = join(OUT_DIR, "span_v2_split_ablation.md");

const FIXED_WINDOW_SIZE = positiveIntegerEnv("SPAN_V2_FIXED_WINDOW_SIZE") ?? 50;
const FIXED_WINDOW_OVERLAP = positiveIntegerEnv("SPAN_V2_FIXED_WINDOW_OVERLAP") ?? 10;
const EMBEDDING_WINDOW_SIZE = positiveIntegerEnv("SPAN_V2_EMBED_WINDOW_SIZE") ?? 12;
const EMBEDDING_WINDOW_STEP = positiveIntegerEnv("SPAN_V2_EMBED_WINDOW_STEP") ?? 6;
const EMBEDDING_BOUNDARY_THRESHOLD = numberEnv("SPAN_V2_EMBED_BOUNDARY_THRESHOLD") ?? 0.76;
const EMBEDDING_MIN_SEGMENT_CALLS = positiveIntegerEnv("SPAN_V2_EMBED_MIN_SEGMENT_CALLS") ?? 12;

type MethodId = "llm_direct" | "fixed_count_then_llm" | "sliding_embedding_then_llm";

interface StoredSpan {
  id: string;
  sourceTraceId: string;
  range: [number, number];
  toolCallCount: number;
  goal: string;
  policy: string;
  summary: string;
}

interface TraceCase {
  traceId: string;
  sourceTrace: MemoryRow;
  rawTurn: RawTurnRecord;
  rawToolCallCount: number;
  trajectory: SpanTrajectoryEvent[];
}

interface MethodTraceResult {
  traceId: string;
  toolCallCount: number;
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
  const embedder = createEvalEmbedder();
  const cases = loadTraceCases();
  const methods: Array<{ id: MethodId; label: string; run: (traceCase: TraceCase) => Promise<MethodTraceResult> }> = [
    {
      id: "llm_direct",
      label: "LLM 直接拆分整条 structured trajectory",
      run: (traceCase) => runDirect(traceCase, llm)
    },
    {
      id: "fixed_count_then_llm",
      label: `固定 ${FIXED_WINDOW_SIZE} tool-call 窗口 + overlap=${FIXED_WINDOW_OVERLAP} 后 LLM`,
      run: (traceCase) => runSegmented(
        traceCase,
        llm,
        buildFixedToolCallSegments(
          traceCase.rawToolCallCount,
          FIXED_WINDOW_SIZE,
          FIXED_WINDOW_OVERLAP
        )
      )
    },
    {
      id: "sliding_embedding_then_llm",
      label: "滑动 embedding 找候选段后 LLM",
      run: async (traceCase) => runSegmented(
        traceCase,
        llm,
        await embeddingSegments(traceCase.rawToolCallCount, traceCase.trajectory, embedder)
      )
    }
  ];

  const results: MethodResult[] = [];
  for (const method of methods) {
    const traceResults: MethodTraceResult[] = [];
    for (const traceCase of cases) {
      traceResults.push(await method.run(traceCase));
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
    llm: publicModelConfig(llm.config),
    embedding: publicModelConfig(embedder.config),
    traceCount: cases.length,
    traceIds: cases.map((traceCase) => traceCase.traceId),
    settings: {
      fixedWindowSize: FIXED_WINDOW_SIZE,
      fixedWindowOverlap: FIXED_WINDOW_OVERLAP,
      embeddingWindowSize: EMBEDDING_WINDOW_SIZE,
      embeddingWindowStep: EMBEDDING_WINDOW_STEP,
      embeddingBoundaryThreshold: EMBEDDING_BOUNDARY_THRESHOLD,
      embeddingMinSegmentCalls: EMBEDDING_MIN_SEGMENT_CALLS
    },
    results
  };
  writeFileSync(OUT_JSON, JSON.stringify(output, null, 2));
  writeFileSync(OUT_MD, renderReport(output));
}

function loadTraceCases(): TraceCase[] {
  const oldResult = JSON.parse(readFileSync(RESULT_JSON, "utf8")) as {
    spans: Array<StoredSpan>;
  };
  const candidateTraceIds = selectLongSpanTraceIds(oldResult.spans);
  const sourceDb = new MemoryDb({ path: SOURCE_DB });
  const repos = new Repositories(sourceDb.db);
  const cases: TraceCase[] = [];
  for (const traceId of candidateTraceIds) {
    const sourceTrace = repos.memories.get(traceId);
    if (!sourceTrace) continue;
    const rawTurnId = typeof sourceTrace.properties.internal_info.raw_turn_id === "string"
      ? sourceTrace.properties.internal_info.raw_turn_id
      : undefined;
    const rawTurn = rawTurnId ? repos.runtime.getRawTurn(rawTurnId) : undefined;
    if (!rawTurn) continue;
    cases.push({
      traceId,
      sourceTrace,
      rawTurn,
      rawToolCallCount: rawTurn.toolCalls.length,
      trajectory: buildSpanTrajectory(rawTurn.toolCalls)
    });
  }
  sourceDb.close();
  return cases;
}

function selectLongSpanTraceIds(spans: Array<{ sourceTraceId: string; toolCallCount: number }>): string[] {
  const limit = positiveIntegerEnv("SPAN_V2_SPLIT_ABLATION_TRACE_LIMIT") ?? 4;
  return [...new Set(
    spans
      .filter((span) => span.toolCallCount >= 50)
      .sort((a, b) => b.toolCallCount - a.toolCallCount)
      .map((span) => span.sourceTraceId)
  )].slice(0, limit);
}

async function runDirect(traceCase: TraceCase, llm: LlmClient): Promise<MethodTraceResult> {
  const segment = { start: 0, end: traceCase.rawToolCallCount - 1, reason: "full_trace" };
  return runSegmented(traceCase, llm, [segment]);
}

async function runSegmented(
  traceCase: TraceCase,
  llm: LlmClient,
  segments: Segment[]
): Promise<MethodTraceResult> {
  const allSpans: SpanDraft[] = [];
  const failures: string[] = [];
  let promptCalls = 0;
  let promptChars = 0;
  for (const segment of segments) {
    if (segment.end - segment.start + 1 <= 3) continue;
    const payload = splitPayload(traceCase, segment);
    const content = stableStringify(payload);
    promptCalls += 1;
    promptChars += SPAN_SPLIT_SYSTEM.length + content.length;
    try {
      const result = await llm.completeJson<{ spans?: unknown; reason?: unknown }>([
        { role: "system", content: SPAN_SPLIT_SYSTEM },
        { role: "user", content }
      ], {
        operation: `span.split_ablation.${segment.reason}`,
        thinkingMode: "disabled",
        temperature: 0.2,
        maxTokens: 4096,
        timeoutMs: 180_000,
        maxRetries: 2
      });
      const spans = parseSpanDrafts(result, traceCase.rawToolCallCount)
        .filter((span) => span.start >= segment.start && span.end <= segment.end);
      allSpans.push(...spans);
    } catch (error) {
      failures.push(`${segment.start}-${segment.end}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  const spans = resolveOverlaps(allSpans);
  return {
    traceId: traceCase.traceId,
    toolCallCount: traceCase.rawToolCallCount,
    promptCalls,
    promptChars,
    segments,
    spans,
    failures
  };
}

function splitPayload(traceCase: TraceCase, segment: Segment): Record<string, unknown> {
  const internal = traceCase.sourceTrace.properties.internal_info;
  const trace = isRecord(internal.trace) ? internal.trace : {};
  return {
    sourceTraceId: traceCase.traceId,
    methodSegment: segment,
    userRequest: redactAndClip(traceCase.rawTurn.userText ?? "", 1_500),
    assistantFinalAnswer: redactAndClip(traceCase.rawTurn.assistantText ?? "", 1_000),
    traceSummary: redactAndClip(text(trace.summary) ?? text(traceCase.sourceTrace.info.summary) ?? "", 800),
    structuredTrajectory: eventsForToolCallRange(traceCase.trajectory, segment)
  };
}

async function embeddingSegments(
  toolCallCount: number,
  trajectory: SpanTrajectoryEvent[],
  embedder: Embedder
): Promise<Segment[]> {
  if (toolCallCount <= EMBEDDING_WINDOW_SIZE) {
    return [{ start: 0, end: Math.max(0, toolCallCount - 1), reason: "embedding_full_short" }];
  }
  const windows: Array<{ start: number; end: number; text: string }> = [];
  for (let start = 0; start < toolCallCount; start += EMBEDDING_WINDOW_STEP) {
    const end = Math.min(toolCallCount - 1, start + EMBEDDING_WINDOW_SIZE - 1);
    windows.push({
      start,
      end,
      text: trajectoryText(eventsForToolCallRange(trajectory, { start, end }))
    });
    if (end >= toolCallCount - 1) break;
  }
  const vectors = await embedder.embed(windows.map((window) => window.text), "document");
  const boundaries: number[] = [];
  for (let index = 1; index < windows.length; index += 1) {
    const left = vectors[index - 1]!;
    const right = vectors[index]!;
    const similarity = cosine(left, right);
    const boundary = windows[index]!.start;
    const previous = boundaries.at(-1) ?? 0;
    if (
      similarity < EMBEDDING_BOUNDARY_THRESHOLD &&
      boundary - previous >= EMBEDDING_MIN_SEGMENT_CALLS &&
      toolCallCount - boundary >= EMBEDDING_MIN_SEGMENT_CALLS
    ) {
      boundaries.push(boundary);
    }
  }
  return boundariesToSegments(toolCallCount, boundaries, "sliding_embedding");
}

function boundariesToSegments(length: number, boundaries: number[], reason: string): Segment[] {
  const sorted = [...new Set(boundaries)]
    .filter((boundary) => boundary > 0 && boundary < length)
    .sort((a, b) => a - b);
  const starts = [0, ...sorted];
  return starts.map((start, index) => ({
    start,
    end: (sorted[index] ?? length) - 1,
    reason
  })).filter((segment) => segment.end >= segment.start);
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
    const overlapRatio = overlap / Math.min(previousLength, currentLength);
    if (overlapRatio >= 0.6 && currentLength > previousLength) {
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
  const toolCallTotal = traceResults.reduce((sum, result) => sum + result.toolCallCount, 0);
  const covered = traceResults.reduce((sum, result) => sum + coveredToolCalls(result.spans), 0);
  const longThreshold = 50;
  const veryLongThreshold = 100;
  return {
    traceCount: traceResults.length,
    spanCount: spans.length,
    avgSpansPerTrace: round(spans.length / Math.max(1, traceResults.length)),
    avgToolCalls: round(spans.reduce((sum, span) => sum + span.end - span.start + 1, 0) / Math.max(1, spans.length)),
    maxToolCalls: Math.max(0, ...spans.map((span) => span.end - span.start + 1)),
    longSpanCount: spans.filter((span) => span.end - span.start + 1 >= longThreshold).length,
    veryLongSpanCount: spans.filter((span) => span.end - span.start + 1 >= veryLongThreshold).length,
    mixedStageCount: spans.filter((span) => isMixedStage(span.goal, span.policy)).length,
    coverageRate: round(covered / Math.max(1, toolCallTotal)),
    promptCalls: traceResults.reduce((sum, result) => sum + result.promptCalls, 0),
    avgPromptChars: round(traceResults.reduce((sum, result) => sum + result.promptChars, 0) / Math.max(1, traceResults.length)),
    failureCount: traceResults.reduce((sum, result) => sum + result.failures.length, 0)
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
  llm: Record<string, unknown>;
  embedding: Record<string, unknown>;
  traceCount: number;
  traceIds: string[];
  settings: Record<string, unknown>;
  results: MethodResult[];
}): string {
  const lines = [
    "# Span 拆分算法消融实验",
    "",
    `生成时间：${output.generatedAt}`,
    "",
    "## 设置",
    "",
    `- Trace 数：${output.traceCount}`,
    `- Trace IDs：${output.traceIds.join(", ")}`,
    `- LLM：provider=${output.llm.provider}，model=${output.llm.model}，endpoint=${output.llm.endpoint}`,
    `- Embedding：provider=${output.embedding.provider}，model=${output.embedding.model}，endpoint=${output.embedding.endpoint}`,
    `- 固定窗口：size=${output.settings.fixedWindowSize}，overlap=${output.settings.fixedWindowOverlap}`,
    `- 滑动 embedding：window=${output.settings.embeddingWindowSize}，step=${output.settings.embeddingWindowStep}，boundaryThreshold=${output.settings.embeddingBoundaryThreshold}，minSegment=${output.settings.embeddingMinSegmentCalls}`,
    "",
    "## 方法",
    "",
    "- 方法 1：LLM 直接读取整条 structured trajectory 并拆分 span。",
    "- 方法 2：先按固定 tool-call 数量切窗口，再让 LLM 在每个窗口内抽 span，最后确定性去重/裁剪 overlap。",
    "- 方法 3：先对 structured trajectory 的滑动窗口做 embedding，相邻窗口低相似处作为候选边界，再让 LLM 对每个候选段抽 span。",
    "",
    "## 指标对比",
    "",
    "| 方法 | spans | avg spans/trace | avg calls/span | max calls | >=50 | >=100 | mixed-stage | coverage | prompt calls | avg prompt chars | failures |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...output.results.map((result) => {
      const m = result.metrics;
      return `| ${result.label} | ${m.spanCount} | ${m.avgSpansPerTrace} | ${m.avgToolCalls} | ${m.maxToolCalls} | ${m.longSpanCount} | ${m.veryLongSpanCount} | ${m.mixedStageCount} | ${pct(m.coverageRate)} | ${m.promptCalls} | ${m.avgPromptChars} | ${m.failureCount} |`;
    }),
    "",
    "## 实验结论",
    "",
    ...renderConclusion(output.results),
    "",
    "## Trace 级样例",
    ""
  ];
  for (const traceId of output.traceIds) {
    lines.push(`### ${traceId}`, "");
    for (const result of output.results) {
      const trace = result.traceResults.find((item) => item.traceId === traceId);
      if (!trace) continue;
      lines.push(`- ${result.label}`);
      lines.push(`  - segments: ${trace.segments.map((segment) => `${segment.start}-${segment.end}`).join(", ")}`);
      lines.push(`  - spans: ${trace.spans.length === 0 ? "none" : trace.spans.map((span) => `${span.start}-${span.end}(${span.end - span.start + 1}) ${span.goal}`).join("；")}`);
      if (trace.failures.length > 0) lines.push(`  - failures: ${trace.failures.join(" | ")}`);
    }
    lines.push("");
  }
  lines.push(
    "## 初步判断口径",
    "",
    "- 这个实验不评价最终 bucket purity，只评价 LLM 前置拆分策略对 span 边界和粒度的影响。",
    "- 更合理的方案应减少超长 span 和 mixed-stage span，同时不要把大量连续工具调用切成不可复用的碎片。",
    "- coverage 不是越高越好；允许非策略性、重复性、过渡性 tool call 留在 span 外。",
    ""
  );
  return lines.join("\n");
}

function renderConclusion(results: MethodResult[]): string[] {
  const direct = results.find((result) => result.method === "llm_direct");
  const fixed = results.find((result) => result.method === "fixed_count_then_llm");
  const embedding = results.find((result) => result.method === "sliding_embedding_then_llm");
  const lines: string[] = [];
  if (fixed && direct && embedding) {
    lines.push(
      `- 当前小样本下，\`${fixed.label}\` 是三种方案里最均衡的：max calls=${fixed.metrics.maxToolCalls}，>=50 span=${fixed.metrics.longSpanCount}，mixed-stage=${fixed.metrics.mixedStageCount}，span 数=${fixed.metrics.spanCount}。`,
      `- \`${direct.label}\` 成本最低（prompt calls=${direct.metrics.promptCalls}），但仍有 max calls=${direct.metrics.maxToolCalls} 的超长 span，说明 programmatic trajectory 不能单独解决长 trace 下 LLM 不切分的问题。`,
      `- \`${embedding.label}\` 消除了超长 span（max calls=${embedding.metrics.maxToolCalls}），但 span 数=${embedding.metrics.spanCount}、prompt calls=${embedding.metrics.promptCalls}，容易过切并增加后续聚类压力。`,
      "- 因此下一步建议先把固定窗口作为输入预算护栏，再让 LLM 在窗口内抽取 span；滑动 embedding 继续作为实验方向，需要更稳的阈值选择和相邻段合并策略。"
    );
  } else {
    lines.push("- 本轮结果用于比较 LLM 前置拆分策略；请结合指标表和 trace 级样例人工审核。");
  }
  return lines;
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

function createEvalEmbedder(): Embedder {
  return createEmbedder({
    ...DEFAULT_MEMMY_CONFIG.embedding,
    provider: "openai_compatible",
    endpoint: process.env.SPAN_V2_EMBEDDING_ENDPOINT ?? "https://apigw.memtensor.cn/model/embedding/v1",
    model: process.env.SPAN_V2_EMBEDDING_MODEL ?? "bge-m3",
    apiKey: process.env.SPAN_V2_EMBEDDING_API_KEY ?? "EMPTY",
    batchSize: 64,
    timeoutMs: 180_000,
    maxRetries: 2,
    cache: true,
    normalize: true
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

function cosine(left: readonly number[], right: readonly number[]): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  if (leftNorm === 0 || rightNorm === 0) return 0;
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

function redactAndClip(value: string, maxChars: number): string {
  return clip(redactSensitiveText(value), maxChars);
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function positiveIntegerEnv(name: string): number | undefined {
  const value = Number(process.env[name]);
  return Number.isInteger(value) && value > 0 ? value : undefined;
}

function numberEnv(name: string): number | undefined {
  const value = Number(process.env[name]);
  return Number.isFinite(value) ? value : undefined;
}

function round(value: number): number {
  return Number(value.toFixed(2));
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

await main();
