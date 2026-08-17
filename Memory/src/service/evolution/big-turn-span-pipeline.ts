import type { LlmClient } from "../../model/types.js";
import type {
  EvolutionJobRecord,
  RawTurnRecord,
  Repositories
} from "../../storage/repositories.js";
import { kindFromMemory } from "../../storage/repositories.js";
import type { MemoryRow } from "../../types.js";
import { stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { redactSensitiveText } from "../../utils/sensitive-data.js";
import { clip } from "../../utils/text.js";
import { formatZonedTime } from "../../utils/time.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import { spanEmbeddingSourceHash, type SpanEmbeddingVectorField } from "../embedding/embedding-pipeline.js";
import {
  parseSpanDrafts,
  SPAN_SCHEMA_VERSION,
  spanId,
  validateSpanDrafts,
  type SpanDraft,
  type SpanPayload
} from "./span-model.js";
import { buildSpanTrajectory, type SpanTrajectoryEvent } from "./span-trajectory.js";

export const SPAN_BIG_TURN_ENABLED = true;
export const SPAN_BIG_TURN_MIN_TOOL_CALLS = 11;
export const SPAN_BIG_TURN_RAW_WINDOW_SIZE = 30;
export const SPAN_BIG_TURN_RAW_WINDOW_OVERLAP = 6;

const SPAN_BIG_TURN_OPERATION = "span.big_turn.v1";
const SPAN_BIG_TURN_SEGMENT_REASON = "raw_fixed_30_overlap_6";
const SPAN_BIG_TURN_PROMPT = `You segment one completed AI-agent turn into meaningful subtask spans.

You are given a programmatic structured trajectory, not the raw full tool
logs. Treat that structured trajectory as the source of truth. It preserves
original tool-call ranges, tool/action kinds, success/error signals,
input/output shapes, artifact signals, and short redacted evidence snippets.
Consecutive repeated tool calls may be compressed into one trajectory event;
use each event's range field when deciding original span boundaries. Do not
infer details that are not supported by this trajectory.

A span represents one extractable local strategy: one concrete goal pursued
through one coherent observed policy over a contiguous sequence of tool calls.
It is not a single tool call and not merely a change of tool name.

Return zero, one, or many spans as evidence supports. Do not target a fixed
span count. Keep diagnosis and repair of the same problem in one span unless
the repair starts an independently meaningful task with a different observed
policy.

Long traces should usually produce multiple spans when the work moves through
different reusable local strategies. Do not merge evidence gathering, data
inspection, artifact construction, debugging/repair, and final verification
into one span when the tool-call evidence supports separate coherent ranges.
Merge adjacent phases only when they are tightly interleaved and cannot be
understood as independent reusable strategies.

Boundary rules:
- Every span covers a contiguous inclusive tool-call range.
- Every span must cover more than three tool calls. Shorter ranges are not
  extractable small strategies.
- Spans must not overlap and must remain in execution order.
- Tool calls that are repetitive, transitional, or irrelevant to reusable
  experience may remain outside all spans.
- Avoid spans that cover a whole long trace just because the trace pursues one
  user task. Split at durable method shifts: search/fetch evidence collection;
  schema/file inspection; script or artifact generation; iterative error repair;
  conversion/export; and final output verification.
- Do not invent actions, results, goals, or errors.

goal requirements:
- Describe the reusable subtask objective, not the one-off assignment.
- Prefer task type, artifact type, data shape, library/tool family, and error
  class over customer names, file names, paths, brands, or domain nouns.
- Keep task-specific names only when they determine the strategy, such as a
  library/module/API name, file format, schema type, benchmark, or exact error.
- Be independently understandable and suitable for retrieval by a future
  similar task.
- Use the same language as the user's request.

policy requirements:
- Describe the observed reusable strategy used in this span, not an idealized
  future recommendation and not a transcript of exact actions.
- Abstract away one-off task nouns, file names, paths, client names, and final
  deliverable titles. Preserve method-defining details: tool/library family,
  file format, validation method, debugging loop, query pattern, or fallback.
- Focus on the decision pattern or approach evidenced by the tool calls: how
  information was gathered, transformed, generated, debugged, or verified.
- Do not make the policy so broad that unrelated strategies collapse together.
  Distinguish evidence gathering, data inspection, artifact construction,
  conversion, debugging, and verification when the tool-call evidence supports
  different local strategies.
- If a candidate policy needs "and then" to connect several independent phases,
  split the span unless the phases are inseparable within the same debugging or
  generation loop.
- Make the policy embedding-friendly: use stable technical nouns for the method,
  input shape, transformation, debugging loop, and verification signal. Do not
  use a schema label, category enum, or field whose exact text would be needed
  by downstream code.
- Avoid policies that are too broad: generate deliverable; create document;
  research and write report; inspect and generate output; complete task.
- Use the same language as the user's request.

summary requirements:
- State what was done and what result was obtained.
- Preserve important decisions, failures, fixes, and verification results.
- Be concise and evidence-based.

Return JSON only:
{
  "reason": "...",
  "spans": [
    {
      "start": 0,
      "end": 3,
      "goal": "...",
      "policy": "...",
      "summary": "..."
    }
  ]
}`;

interface BigTurnSpanDeps {
  repos: Repositories;
  llm: LlmClient;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
  namespaceIdFromMemory(memory: MemoryRow): string;
  embedAfterCapture(): boolean;
}

export class BigTurnSpanPipeline {
  constructor(private readonly deps: BigTurnSpanDeps) {}

  async splitAndStore(job: EvolutionJobRecord): Promise<void> {
    if (!SPAN_BIG_TURN_ENABLED || !this.deps.llm.isConfigured()) return;
    const source = job.targetMemoryId
      ? this.deps.repos.memories.get(job.targetMemoryId)
      : undefined;
    const rawTurnId = text(job.payload.rawTurnId);
    const rawTurn = rawTurnId
      ? this.deps.repos.runtime.getRawTurn(rawTurnId)
      : undefined;
    if (!source || !rawTurn || rawTurn.toolCalls.length < SPAN_BIG_TURN_MIN_TOOL_CALLS) return;

    const trajectory = buildSpanTrajectory(rawTurn.toolCalls);
    const spans = await this.extractSpansFromSegments(source, rawTurn, job, trajectory);
    if (spans.length === 0) return;
    this.deps.repos.transaction(() => {
      const spanIds: string[] = [];
      for (const [spanIndex, span] of spans.entries()) {
        spanIds.push(this.storeSpan({ source, rawTurn, job, span, spanIndex }));
      }
      this.linkSpansToSourceTrace(source.id, spanIds, job);
    });
  }

  private async extractSpansFromSegments(
    source: MemoryRow,
    rawTurn: RawTurnRecord,
    job: EvolutionJobRecord,
    trajectory: SpanTrajectoryEvent[]
  ): Promise<SpanDraft[]> {
    const segments = fixedRawToolCallSegments(rawTurn.toolCalls.length);
    const drafts: SpanDraft[] = [];
    for (const segment of segments) {
      const result = await this.deps.llm.completeJson<{
        reason?: unknown;
        spans?: unknown;
      }>([
        { role: "system", content: SPAN_BIG_TURN_PROMPT },
        {
          role: "user",
          content: stableStringify(bigTurnPromptPayload(source, rawTurn, job, trajectory, segment))
        }
      ], {
        operation: SPAN_BIG_TURN_OPERATION,
        thinkingMode: "disabled",
        temperature: 0.6,
        maxTokens: 4096
      });
      drafts.push(...parseSpanDrafts(result, rawTurn.toolCalls.length)
        .filter((span) => span.start >= segment.start && span.end <= segment.end));
    }
    const spans = resolveWindowOverlaps(drafts);
    validateSpanDrafts(spans, rawTurn.toolCalls.length);
    return spans;
  }

  private storeSpan(input: {
    source: MemoryRow;
    rawTurn: RawTurnRecord;
    job: EvolutionJobRecord;
    span: SpanDraft;
    spanIndex: number;
  }): string {
    const { source, rawTurn, job, span, spanIndex } = input;
    const id = spanId(source.id, span.start, span.end);
    const payload: SpanPayload = {
      schema_version: SPAN_SCHEMA_VERSION,
      source_trace_id: source.id,
      raw_turn_id: rawTurn.id,
      ...(job.episodeId ? { episode_id: job.episodeId } : {}),
      span_index: spanIndex,
      tool_call_start: span.start,
      tool_call_end: span.end,
      tool_call_count: span.end - span.start + 1,
      goal: span.goal,
      policy: span.policy,
      summary: span.summary,
      derived: true
    };
    const value = [
      `Goal: ${span.goal}`,
      `Policy: ${span.policy}`,
      `Summary: ${span.summary}`
    ].join("\n");
    const memory = this.deps.buildMemory({
      id,
      userId: source.userId,
      conversationId: source.conversationId,
      sessionId: source.sessionId,
      agentId: source.agentId,
      appId: source.appId,
      projectId: text(source.info.project_id),
      profileId: text(source.info.profile_id),
      layer: "L1",
      kind: "span",
      memoryType: "LongTermMemory",
      key: span.goal,
      value,
      tags: ["span", "big-turn", "derived"],
      info: {
        title: span.goal,
        summary: span.summary,
        span_goal: span.goal,
        goal: span.goal,
        policy: span.policy,
        source_trace_id: source.id,
        raw_turn_id: rawTurn.id,
        episode_id: job.episodeId
      },
      internal: {
        source: "worker.span_big_turn.v1",
        plugin_algorithm: "span.big_turn.v1",
        summary: span.summary,
        span: {
          ...payload,
          span_goal: span.goal
        }
      },
      createdAt: job.createdAt
    });
    const previous = this.deps.repos.memories.get(id);
    const saved = previous
      ? this.deps.repos.memories.update({
          ...memory,
          createdAt: previous.createdAt,
          version: previous.version
        })
      : this.deps.repos.memories.insert(memory);
    this.deps.repos.runtime.appendChange({
      memoryId: saved.id,
      namespaceId: this.deps.namespaceIdFromMemory(saved),
      kind: kindFromMemory(saved),
      op: previous ? "updated" : "created",
      entityId: saved.id,
      userId: saved.userId,
      changeType: previous ? "span_updated" : "span_created",
      before: previous,
      after: saved,
      source: "worker.span_big_turn.v1",
      createdAt: job.createdAt
    });
    if (this.deps.embedAfterCapture()) {
      for (const vectorField of ["vec_goal", "vec_policy"] as const) {
        this.enqueueSpanEmbeddingJob(saved, job, vectorField);
      }
    }
    return saved.id;
  }

  private enqueueSpanEmbeddingJob(
    memory: MemoryRow,
    sourceJob: EvolutionJobRecord,
    vectorField: SpanEmbeddingVectorField
  ): void {
    const sourceHash = spanEmbeddingSourceHash(memory, vectorField);
    if (!sourceHash) return;
    this.deps.enqueueJob({
      jobType: "embedding",
      userId: memory.userId,
      sessionId: memory.sessionId,
      episodeId: sourceJob.episodeId,
      targetMemoryId: memory.id,
      dedupeKey: `embedding:${memory.id}:${vectorField}:${sourceHash}`,
      payload: {
        reason: "span.big_turn.v2",
        sourceJobId: sourceJob.id,
        vectorField,
        sourceHash
      },
      maxAttempts: 6,
      createdAt: sourceJob.createdAt
    });
  }

  private linkSpansToSourceTrace(
    sourceTraceId: string,
    spanIds: string[],
    job: EvolutionJobRecord
  ): void {
    const previous = this.deps.repos.memories.get(sourceTraceId);
    if (!previous) {
      throw new Error(`span.big_turn source Trace not found: ${sourceTraceId}`);
    }
    const previousTrace = previous.properties.internal_info.trace;
    if (!isRecord(previousTrace)) {
      throw new Error(`span.big_turn source Trace metadata is invalid: ${sourceTraceId}`);
    }
    const saved = this.deps.repos.memories.update({
      ...previous,
      properties: {
        ...previous.properties,
        internal_info: {
          ...previous.properties.internal_info,
          trace: {
            ...previousTrace,
            span_ids: spanIds
          }
        }
      },
      updatedAt: job.updatedAt
    });
    this.deps.repos.runtime.appendChange({
      memoryId: saved.id,
      namespaceId: this.deps.namespaceIdFromMemory(saved),
      kind: kindFromMemory(saved),
      op: "updated",
      entityId: saved.id,
      userId: saved.userId,
      changeType: "span_links_updated",
      before: previous,
      after: saved,
      source: "worker.span_big_turn.v1",
      createdAt: job.updatedAt
    });
  }
}

function bigTurnPromptPayload(
  source: MemoryRow,
  rawTurn: RawTurnRecord,
  job: EvolutionJobRecord,
  trajectory: SpanTrajectoryEvent[],
  segment: SpanSegment
): Record<string, unknown> {
  const internal = source.properties.internal_info;
  const trace = isRecord(internal.trace) ? internal.trace : {};
  const traceTimestamp = number(trace.ts);
  const traceTimeZone = text(trace.time_zone);
  return {
    sourceTraceId: redactSensitiveText(source.id),
    capturedAt: traceTimestamp === undefined
      ? undefined
      : formatZonedTime(traceTimestamp, traceTimeZone),
    userRequest: redactAndClip(rawTurn.userText ?? "", 2_000),
    assistantFinalAnswer: redactAndClip(rawTurn.assistantText ?? "", 2_000),
    traceSummary: redactAndClip(
      text(trace.summary) ?? text(source.info.summary) ?? "",
      1_000
    ),
    reflection: redactAndClip(
      text(trace.reflection) ?? text(internal.reflection) ?? "",
      1_000
    ),
    segment,
    structuredTrajectory: eventsForToolCallRange(trajectory, segment)
  };
}

interface SpanSegment {
  start: number;
  end: number;
  reason: typeof SPAN_BIG_TURN_SEGMENT_REASON;
}

function fixedRawToolCallSegments(toolCallCount: number): SpanSegment[] {
  if (toolCallCount <= 0) return [];
  const segments: SpanSegment[] = [];
  const step = Math.max(1, SPAN_BIG_TURN_RAW_WINDOW_SIZE - SPAN_BIG_TURN_RAW_WINDOW_OVERLAP);
  for (let start = 0; start < toolCallCount; start += step) {
    const end = Math.min(toolCallCount - 1, start + SPAN_BIG_TURN_RAW_WINDOW_SIZE - 1);
    segments.push({ start, end, reason: SPAN_BIG_TURN_SEGMENT_REASON });
    if (end >= toolCallCount - 1) break;
  }
  return segments;
}

function eventsForToolCallRange(
  trajectory: readonly SpanTrajectoryEvent[],
  segment: Pick<SpanSegment, "start" | "end">
): SpanTrajectoryEvent[] {
  return trajectory
    .filter((event) => event.range[1] >= segment.start && event.range[0] <= segment.end)
    .map((event) => {
      const range: [number, number] = [
        Math.max(event.range[0], segment.start),
        Math.min(event.range[1], segment.end)
      ];
      const callCount = range[1] - range[0] + 1;
      return {
        ...event,
        index: range[0],
        range,
        callCount,
        evidence: event.repeated
          ? replaceEvidenceRepeatRange(event.evidence, callCount, range)
          : event.evidence
      };
    });
}

function replaceEvidenceRepeatRange(
  evidence: string | undefined,
  callCount: number,
  range: [number, number]
): string | undefined {
  if (!evidence) return evidence;
  return evidence.replace(/repeat=\d+ calls range=\d+-\d+/u, `repeat=${callCount} calls range=${range[0]}-${range[1]}`);
}

function resolveWindowOverlaps(spans: readonly SpanDraft[]): SpanDraft[] {
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
    // Keep semantic text and original range aligned. A partially overlapping
    // candidate is discarded unless it clearly supersedes the previous span.
  }
  return out;
}

function redactAndClip(value: string, maxChars: number): string {
  return clip(redactSensitiveText(value), maxChars);
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
