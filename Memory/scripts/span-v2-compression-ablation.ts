import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  createLlmClient,
  DEFAULT_MEMMY_CONFIG,
  MemoryDb,
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

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const SOURCE_DB = join(OUT_DIR, "span_v2_experiment.sqlite");
const RESULT_JSON = join(OUT_DIR, "span_v2_experiment_results.json");
const OUT_JSON = join(OUT_DIR, "span_v2_compression_ablation.json");
const OUT_MD = join(OUT_DIR, "span_v2_compression_ablation.md");

const TRACE_LIMIT = positiveIntegerEnv("SPAN_V2_COMPRESSION_ABLATION_TRACE_LIMIT") ?? 4;

type VariantId = "uncompressed" | "compressed";

interface StoredSpan {
  sourceTraceId: string;
  toolCallCount: number;
}

interface TraceCase {
  traceId: string;
  sourceTrace: MemoryRow;
  rawTurn: RawTurnRecord;
}

interface VariantTraceResult {
  traceId: string;
  rawToolCalls: number;
  eventCount: number;
  repeatedGroups: number;
  callsInsideRepeatedGroups: number;
  trajectoryJsonChars: number;
  evidenceChars: number;
  promptChars: number;
  spans: SpanDraft[];
  failure?: string;
}

interface VariantResult {
  variant: VariantId;
  label: string;
  maxRepeatedCalls: number;
  traceResults: VariantTraceResult[];
  metrics: ReturnType<typeof summarizeVariant>;
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
  const cases = loadTraceCases();
  const variants: Array<{ id: VariantId; label: string; maxRepeatedCalls: number }> = [
    { id: "uncompressed", label: "不压缩 structured trajectory", maxRepeatedCalls: 1 },
    { id: "compressed", label: "压缩连续重复 tool 后 structured trajectory", maxRepeatedCalls: 8 }
  ];
  const results: VariantResult[] = [];
  for (const variant of variants) {
    const traceResults: VariantTraceResult[] = [];
    for (const traceCase of cases) {
      traceResults.push(await runVariant(traceCase, variant, llm));
    }
    results.push({
      variant: variant.id,
      label: variant.label,
      maxRepeatedCalls: variant.maxRepeatedCalls,
      traceResults,
      metrics: summarizeVariant(traceResults)
    });
  }
  const output = {
    generatedAt: new Date().toISOString(),
    llm: publicModelConfig(llm.config),
    traceCount: cases.length,
    traceIds: cases.map((traceCase) => traceCase.traceId),
    variants: results
  };
  writeFileSync(OUT_JSON, JSON.stringify(output, null, 2));
  writeFileSync(OUT_MD, renderReport(output));
}

function loadTraceCases(): TraceCase[] {
  const previous = JSON.parse(readFileSync(RESULT_JSON, "utf8")) as {
    spans?: StoredSpan[];
  };
  const traceIds = [...new Set((previous.spans ?? [])
    .filter((span) => span.toolCallCount >= 50)
    .sort((a, b) => b.toolCallCount - a.toolCallCount)
    .map((span) => span.sourceTraceId)
  )].slice(0, TRACE_LIMIT);
  const db = new MemoryDb({ path: SOURCE_DB });
  const repos = new Repositories(db.db);
  const cases: TraceCase[] = [];
  for (const traceId of traceIds) {
    const sourceTrace = repos.memories.get(traceId);
    const rawTurnId = sourceTrace?.properties.internal_info.raw_turn_id;
    const rawTurn = typeof rawTurnId === "string"
      ? repos.runtime.getRawTurn(rawTurnId)
      : undefined;
    if (!sourceTrace || !rawTurn) continue;
    cases.push({ traceId, sourceTrace, rawTurn });
  }
  db.close();
  return cases;
}

async function runVariant(
  traceCase: TraceCase,
  variant: { id: VariantId; maxRepeatedCalls: number },
  llm: LlmClient
): Promise<VariantTraceResult> {
  const trajectory = buildSpanTrajectory(traceCase.rawTurn.toolCalls, {
    maxRepeatedCalls: variant.maxRepeatedCalls
  });
  const payload = splitPayload(traceCase, trajectory);
  const content = stableStringify(payload);
  const base = {
    traceId: traceCase.traceId,
    rawToolCalls: traceCase.rawTurn.toolCalls.length,
    eventCount: trajectory.length,
    repeatedGroups: trajectory.filter((event) => event.repeated).length,
    callsInsideRepeatedGroups: trajectory
      .filter((event) => event.repeated)
      .reduce((sum, event) => sum + event.callCount, 0),
    trajectoryJsonChars: stableStringify(trajectory).length,
    evidenceChars: trajectory.reduce((sum, event) => sum + (event.evidence?.length ?? 0), 0),
    promptChars: SPAN_SPLIT_SYSTEM.length + content.length
  };
  try {
    const result = await llm.completeJson<{ spans?: unknown; reason?: unknown }>([
      { role: "system", content: SPAN_SPLIT_SYSTEM },
      { role: "user", content }
    ], {
      operation: `span.compression_ablation.${variant.id}`,
      thinkingMode: "disabled",
      temperature: 0.2,
      maxTokens: 4096,
      timeoutMs: 180_000,
      maxRetries: 2,
      malformedRetries: 2
    });
    return {
      ...base,
      spans: parseSpanDrafts(result, traceCase.rawTurn.toolCalls.length)
    };
  } catch (error) {
    return {
      ...base,
      spans: [],
      failure: error instanceof Error ? error.message : String(error)
    };
  }
}

function splitPayload(
  traceCase: TraceCase,
  structuredTrajectory: SpanTrajectoryEvent[]
): Record<string, unknown> {
  const internal = traceCase.sourceTrace.properties.internal_info;
  const trace = isRecord(internal.trace) ? internal.trace : {};
  return {
    sourceTraceId: traceCase.traceId,
    userRequest: redactAndClip(traceCase.rawTurn.userText ?? "", 1_500),
    assistantFinalAnswer: redactAndClip(traceCase.rawTurn.assistantText ?? "", 1_000),
    traceSummary: redactAndClip(text(trace.summary) ?? text(traceCase.sourceTrace.info.summary) ?? "", 800),
    structuredTrajectory
  };
}

function summarizeVariant(traceResults: VariantTraceResult[]) {
  const spans = traceResults.flatMap((result) => result.spans);
  const rawToolCalls = traceResults.reduce((sum, result) => sum + result.rawToolCalls, 0);
  return {
    rawToolCalls,
    eventCount: traceResults.reduce((sum, result) => sum + result.eventCount, 0),
    repeatedGroups: traceResults.reduce((sum, result) => sum + result.repeatedGroups, 0),
    callsInsideRepeatedGroups: traceResults.reduce((sum, result) => sum + result.callsInsideRepeatedGroups, 0),
    trajectoryJsonChars: traceResults.reduce((sum, result) => sum + result.trajectoryJsonChars, 0),
    evidenceChars: traceResults.reduce((sum, result) => sum + result.evidenceChars, 0),
    promptChars: traceResults.reduce((sum, result) => sum + result.promptChars, 0),
    spanCount: spans.length,
    avgSpansPerTrace: round(spans.length / Math.max(1, traceResults.length)),
    avgToolCallsPerSpan: round(spans.reduce((sum, span) => sum + span.end - span.start + 1, 0) / Math.max(1, spans.length)),
    maxToolCallsPerSpan: Math.max(0, ...spans.map((span) => span.end - span.start + 1)),
    longSpanCount: spans.filter((span) => span.end - span.start + 1 >= 50).length,
    veryLongSpanCount: spans.filter((span) => span.end - span.start + 1 >= 100).length,
    mixedStageCount: spans.filter((span) => isMixedStage(span.goal, span.policy)).length,
    coverageRate: round(traceResults.reduce((sum, result) => sum + coveredToolCalls(result.spans), 0) / Math.max(1, rawToolCalls)),
    failureCount: traceResults.filter((result) => result.failure).length
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
  traceCount: number;
  traceIds: string[];
  variants: VariantResult[];
}): string {
  const [uncompressed, compressed] = output.variants;
  const inputComparison = uncompressed && compressed
    ? compareMetrics(uncompressed.metrics, compressed.metrics)
    : undefined;
  const lines = [
    "# Span Structured Trajectory 压缩消融实验",
    "",
    `生成时间：${output.generatedAt}`,
    "",
    "## 背景",
    "",
    "原始 trace 不能直接交给 LLM 做 span 拆分：长轨迹中存在大量连续重复的 search/fetch/read/execute 调用，工具返回也可能很长。当前方案先把 raw tool calls 转为 programmatic structured trajectory，再让 LLM 基于结构化轨迹拆 span。",
    "",
    "本次只比较 structured trajectory 的两种输入形态：",
    "",
    "- 不压缩：每个原始 tool call 对应一个 structured event。",
    "- 压缩后：连续相同 tool/action/success/errorClass 的重复调用合并为一个 event，保留原始 `range`、`callCount`、`repeated` 和截断后的证据。",
    "",
    "压缩不改变 span 边界语义：LLM 输出仍然必须使用原始 tool-call index。",
    "",
    "## 设置",
    "",
    `- Trace 数：${output.traceCount}`,
    `- Trace IDs：${output.traceIds.join(", ")}`,
    `- LLM：provider=${output.llm.provider}，model=${output.llm.model}，endpoint=${output.llm.endpoint}`,
    "- 拆分方式：同一 prompt 直接读取整条 structured trajectory。",
    "",
    "## 输入规模对比",
    "",
    "| 形态 | raw calls | events | repeated groups | repeated calls | trajectory JSON chars | evidence chars | prompt chars |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...output.variants.map((variant) => {
      const m = variant.metrics;
      return `| ${variant.label} | ${m.rawToolCalls} | ${m.eventCount} | ${m.repeatedGroups} | ${m.callsInsideRepeatedGroups} | ${m.trajectoryJsonChars} | ${m.evidenceChars} | ${m.promptChars} |`;
    }),
    "",
    inputComparison
      ? `压缩后 event 数下降 ${pct(inputComparison.eventReduction)}，trajectory JSON 字符下降 ${pct(inputComparison.jsonReduction)}，evidence 字符下降 ${pct(inputComparison.evidenceReduction)}，prompt 总字符下降 ${pct(inputComparison.promptReduction)}。`
      : "",
    "",
    "## LLM 拆分结果对比",
    "",
    "| 形态 | spans | avg spans/trace | avg calls/span | max calls/span | >=50 | >=100 | mixed-stage | coverage | failures |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...output.variants.map((variant) => {
      const m = variant.metrics;
      return `| ${variant.label} | ${m.spanCount} | ${m.avgSpansPerTrace} | ${m.avgToolCallsPerSpan} | ${m.maxToolCallsPerSpan} | ${m.longSpanCount} | ${m.veryLongSpanCount} | ${m.mixedStageCount} | ${pct(m.coverageRate)} | ${m.failureCount} |`;
    }),
    "",
    "## Trace 级对照",
    ""
  ].filter((line) => line !== undefined);
  for (const traceId of output.traceIds) {
    lines.push(`### ${traceId}`, "");
    for (const variant of output.variants) {
      const item = variant.traceResults.find((trace) => trace.traceId === traceId);
      if (!item) continue;
      lines.push(
        `- ${variant.label}`,
        `  - 输入：raw=${item.rawToolCalls}，events=${item.eventCount}，jsonChars=${item.trajectoryJsonChars}，evidenceChars=${item.evidenceChars}`,
        `  - spans：${item.spans.length === 0 ? "none" : item.spans.map((span) => `${span.start}-${span.end}(${span.end - span.start + 1}) ${span.goal}`).join("；")}`
      );
      if (item.failure) lines.push(`  - failure：${item.failure}`);
    }
    lines.push("");
  }
  lines.push(
    "## 结论",
    "",
    ...renderConclusion(output.variants),
    ""
  );
  return lines.join("\n");
}

function compareMetrics(
  uncompressed: VariantResult["metrics"],
  compressed: VariantResult["metrics"]
) {
  return {
    eventReduction: 1 - compressed.eventCount / Math.max(1, uncompressed.eventCount),
    jsonReduction: 1 - compressed.trajectoryJsonChars / Math.max(1, uncompressed.trajectoryJsonChars),
    evidenceReduction: 1 - compressed.evidenceChars / Math.max(1, uncompressed.evidenceChars),
    promptReduction: 1 - compressed.promptChars / Math.max(1, uncompressed.promptChars)
  };
}

function renderConclusion(variants: VariantResult[]): string[] {
  const uncompressed = variants.find((variant) => variant.variant === "uncompressed");
  const compressed = variants.find((variant) => variant.variant === "compressed");
  if (!uncompressed || !compressed) return ["- 本轮结果用于比较压缩前后的输入规模和 LLM direct split 行为。"];
  const c = compareMetrics(uncompressed.metrics, compressed.metrics);
  return [
    `- 压缩主要收益是降低输入噪音：event 数减少 ${pct(c.eventReduction)}，prompt 字符减少 ${pct(c.promptReduction)}。`,
    "- 压缩后的 trajectory 仍保留原始 range，因此不会牺牲 span 边界回溯能力。",
    "- 压缩不是最终分桶质量指标；它解决的是 LLM 前置拆分的输入预算和重复噪音问题。最终仍需要结合固定窗口护栏、span 抽样审核和后续 bucket purity 评估。"
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

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
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
