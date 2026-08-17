import type { MemoryRow } from "../../types.js";
import { stableHash } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { redactSensitiveText } from "../../utils/sensitive-data.js";
import { clip } from "../../utils/text.js";

export const SPAN_SCHEMA_VERSION = "span.v2" as const;
export const SPAN_GOAL_MAX_CHARS = 500;
export const SPAN_POLICY_MAX_CHARS = 1_000;
export const SPAN_SUMMARY_MAX_CHARS = 2_000;

export interface SpanDraft {
  start: number;
  end: number;
  goal: string;
  policy: string;
  summary: string;
}

export interface SpanPayload {
  schema_version: typeof SPAN_SCHEMA_VERSION;
  source_trace_id: string;
  raw_turn_id: string;
  episode_id?: string;
  span_index: number;
  tool_call_start: number;
  tool_call_end: number;
  tool_call_count: number;
  goal: string;
  policy: string;
  summary: string;
  derived: true;
}

export function parseSpanDrafts(result: unknown, toolCallCount: number): SpanDraft[] {
  if (!isRecord(result) || !Array.isArray(result.spans)) {
    throw new Error("span.big_turn returned invalid JSON shape");
  }
  const spans = result.spans.map((value): SpanDraft => {
    if (!isRecord(value)) throw new Error("span.big_turn returned invalid span");
    const start = integer(value.start);
    const end = integer(value.end);
    const goal = boundedText(value.goal, SPAN_GOAL_MAX_CHARS);
    const policy = boundedText(value.policy, SPAN_POLICY_MAX_CHARS);
    const summary = boundedText(value.summary, SPAN_SUMMARY_MAX_CHARS);
    if (
      start === undefined ||
      end === undefined ||
      start < 0 ||
      start > end ||
      end >= toolCallCount ||
      !goal ||
      !policy ||
      !summary
    ) {
      throw new Error("span.big_turn returned invalid Span fields");
    }
    return { start, end, goal, policy, summary };
  });
  validateSpanDrafts(spans, toolCallCount);
  return spans;
}

export function validateSpanDrafts(spans: readonly SpanDraft[], toolCallCount: number): void {
  for (const [index, span] of spans.entries()) {
    if (span.start < 0 || span.start > span.end || span.end >= toolCallCount) {
      throw new Error("span.big_turn returned invalid Span range");
    }
    if (span.end - span.start + 1 <= 3) {
      throw new Error("Span must cover more than three tool calls");
    }
    if (index > 0 && span.start <= spans[index - 1]!.end) {
      throw new Error("span.big_turn returned overlapping or unordered spans");
    }
  }
}

export function spanPayload(memory: MemoryRow): SpanPayload | null {
  const span = memory.properties.internal_info.span;
  if (!isRecord(span) || span.schema_version !== SPAN_SCHEMA_VERSION) return null;
  const start = integer(span.tool_call_start);
  const end = integer(span.tool_call_end);
  const toolCallCount = integer(span.tool_call_count);
  const spanIndex = integer(span.span_index);
  const goal = boundedText(span.goal, SPAN_GOAL_MAX_CHARS);
  const policy = boundedText(span.policy, SPAN_POLICY_MAX_CHARS);
  const summary = boundedText(span.summary, SPAN_SUMMARY_MAX_CHARS);
  if (
    start === undefined ||
    end === undefined ||
    toolCallCount === undefined ||
    toolCallCount !== end - start + 1 ||
    toolCallCount <= 3 ||
    !goal ||
    !policy ||
    !summary ||
    typeof span.source_trace_id !== "string" ||
    typeof span.raw_turn_id !== "string" ||
    spanIndex === undefined ||
    span.derived !== true
  ) {
    return null;
  }
  return {
    schema_version: SPAN_SCHEMA_VERSION,
    source_trace_id: span.source_trace_id,
    raw_turn_id: span.raw_turn_id,
    ...(typeof span.episode_id === "string" ? { episode_id: span.episode_id } : {}),
    span_index: spanIndex,
    tool_call_start: start,
    tool_call_end: end,
    tool_call_count: toolCallCount,
    goal,
    policy,
    summary,
    derived: true
  };
}

export function spanKey(sourceTraceId: string, start: number, end: number): string {
  return `span:${sourceTraceId}:${start}:${end}:v2`;
}

export function spanId(sourceTraceId: string, start: number, end: number): string {
  return `span_${stableHash(`${sourceTraceId}:${start}:${end}:${SPAN_SCHEMA_VERSION}`).slice(0, 20)}`;
}

function boundedText(value: unknown, maxChars: number): string | undefined {
  if (typeof value !== "string") return undefined;
  const cleaned = clip(redactSensitiveText(value.trim()), maxChars).trim();
  return cleaned || undefined;
}

function integer(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) ? value : undefined;
}
