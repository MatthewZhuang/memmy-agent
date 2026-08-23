import type { LlmClient } from "../../model/types.js";
import type { RawTurnRecord } from "../../storage/repositories.js";
import { stableHash, stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { redactSensitiveText } from "../../utils/sensitive-data.js";
import { clip } from "../../utils/text.js";
import {
  applyStateDelta,
  buildEpisodeExecutionPath,
  emptyObservedState,
  SPAN_V3_SCHEMA_VERSION,
  type EpisodeExecutionPathV1,
  type SpanV3,
  type StateDeltaOp,
  type StateDeltaOperation
} from "./span-v3-model.js";
import {
  buildSpanTrajectory,
  type SpanTrajectoryAction
} from "./span-trajectory.js";

export const SPAN_V3_RECONSTRUCTION_VERSION = "episode-span-reconstruction.v1" as const;

const BOUNDARY_OPERATION = "span.v3.boundary.v1";
const STATE_DELTA_OPERATION = "span.v3.state_delta.v1";
const TEXT_PREVIEW_MAX = 1_200;
const EVENT_EVIDENCE_MAX = 600;

const BOUNDARY_PROMPT = `You decide which proposed boundaries separate one completed agent turn into reusable state-transition spans.

The program has already proposed a small set of candidate boundaries using generic structural signals. You may only accept or reject those candidates. Never invent another boundary.

A boundary is valid when the events before it have produced a stable, evidence-supported intermediate state from which a materially different action could be selected. Prefer the coarsest segmentation that preserves meaningful decision points.

Do not split merely because the tool name changes. Keep repeated collection, retries, and tightly coupled operations together when they pursue the same local objective and do not create a durable intermediate state.

Use only the supplied evidence. Return JSON only:
{
  "decisions": [
    {
      "candidate_id": "...",
      "split": true,
      "reason": "...",
      "confidence": 0.0
    }
  ]
}`;

const STATE_DELTA_PROMPT = `You reconstruct evidence-grounded state deltas for an already segmented agent episode.

Do not rewrite a complete state snapshot. Extract only typed changes supported by the supplied user observation, tool result, assistant output, or artifact evidence. Every operation must cite one or more supplied source_refs. Never invent a source reference.

Observation deltas describe exogenous information such as goals, constraints, corrections, or completion signals introduced by the user/environment. Action deltas describe effects actually produced or observed by the agent action. Do not claim a user-provided constraint as an action effect.

Use stable, concise subjects. Preserve task-specific values when they constrain execution. Mark a response-only segment include=false only when it has no substantive task effect, such as a purely social acknowledgement.

Allowed operations:
- goal.set, goal.refine, goal.complete
- constraint.upsert, constraint.remove
- fact.upsert, fact.invalidate
- artifact.upsert, artifact.verify
- issue.upsert, issue.resolve
- verification.set
- status.set, whose value/status must be one of: active, blocked, completed, failed

goal.complete is terminal task completion and automatically implies taskStatus=completed. Use it only when the supplied evidence supports completion.

Return JSON only:
{
  "observations": [
    {
      "source_id": "turn:...:user",
      "operations": [
        { "op": "goal.set", "subject": "task_goal", "value": "...", "status": "in_progress", "source_refs": ["turn:...:user"] }
      ]
    }
  ],
  "actions": [
    {
      "segment_id": "segment_...",
      "include": true,
      "local_goal": "...",
      "summary": "...",
      "outcome_status": "success",
      "verification_refs": [],
      "operations": []
    }
  ]
}`;

type ActionFamily = "observe" | "mutate" | "execute" | "verify" | "communicate" | "other";

export interface EpisodeActionEventV1 {
  id: string;
  rawTurnId: string;
  turnIndex: number;
  eventIndex: number;
  kind: "tool_action" | "response_generation";
  action: SpanTrajectoryAction | "respond";
  family: ActionFamily;
  success: boolean;
  callCount: number;
  errorCount: number;
  sourceRefs: string[];
  evidence?: string;
  toolCallRange?: [number, number];
}

export interface SpanBoundaryCandidateV1 {
  id: string;
  rawTurnId: string;
  afterEventId: string;
  beforeEventId: string;
  signals: Array<"action_family_change" | "outcome_change">;
}

export interface ActionSegmentV1 {
  id: string;
  rawTurnId: string;
  turnIndex: number;
  segmentIndex: number;
  kind: "tool_sequence" | "response_generation";
  events: EpisodeActionEventV1[];
  sourceRefs: string[];
  toolCallRange?: [number, number];
}

interface TurnFrame {
  rawTurn: RawTurnRecord;
  turnIndex: number;
  userSourceId: string;
  assistantSourceId: string;
  events: EpisodeActionEventV1[];
  candidates: SpanBoundaryCandidateV1[];
  segments: ActionSegmentV1[];
}

interface ObservationSemantics {
  sourceId: string;
  operations: StateDeltaOperation[];
}

interface ActionSemantics {
  segmentId: string;
  include: boolean;
  localGoal: string;
  summary: string;
  outcomeStatus: "success" | "failure" | "partial" | "unknown";
  verificationRefs: string[];
  operations: StateDeltaOperation[];
}

interface ReconstructorDeps {
  llm: LlmClient;
}

export class EpisodeSpanV3Reconstructor {
  constructor(private readonly deps: ReconstructorDeps) {}

  async reconstruct(input: {
    episodeId: string;
    rawTurns: readonly RawTurnRecord[];
    terminalReward?: number;
  }): Promise<EpisodeExecutionPathV1> {
    if (!this.deps.llm.isConfigured()) {
      throw new Error("span.v3 reconstruction requires a configured LLM");
    }
    const rawTurns = [...input.rawTurns]
      .filter((turn) => turn.episodeId === input.episodeId && !turn.deletedAt)
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id));
    const sourceSnapshotHash = stableHash(rawTurns.map(sourceTurnSnapshot));
    const frames: TurnFrame[] = [];
    for (const [turnIndex, rawTurn] of rawTurns.entries()) {
      const events = buildEpisodeActionEvents(rawTurn, turnIndex);
      const candidates = proposeSpanBoundaryCandidates(events);
      const acceptedBoundaryIds = await this.acceptedBoundaries({
        episodeId: input.episodeId,
        rawTurns,
        rawTurn,
        events,
        candidates
      });
      frames.push({
        rawTurn,
        turnIndex,
        userSourceId: userSourceId(rawTurn.id),
        assistantSourceId: assistantSourceId(rawTurn.id),
        events,
        candidates,
        segments: segmentActionEvents(rawTurn.id, turnIndex, events, candidates, acceptedBoundaryIds)
      });
    }

    const semantics = await this.extractSemantics(input.episodeId, frames);
    return assembleExecutionPath({
      episodeId: input.episodeId,
      frames,
      semantics,
      sourceSnapshotHash,
      model: this.deps.llm.config.model,
      terminalReward: input.terminalReward
    });
  }

  private async acceptedBoundaries(input: {
    episodeId: string;
    rawTurns: readonly RawTurnRecord[];
    rawTurn: RawTurnRecord;
    events: readonly EpisodeActionEventV1[];
    candidates: readonly SpanBoundaryCandidateV1[];
  }): Promise<Set<string>> {
    if (input.candidates.length === 0) return new Set();
    const result = await this.deps.llm.completeJson<{
      decisions?: unknown;
    }>([
      { role: "system", content: BOUNDARY_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId: input.episodeId,
          episodeContext: input.rawTurns.map((turn) => ({
            rawTurnId: turn.id,
            user: preview(turn.userText),
            assistant: preview(turn.assistantText)
          })),
          targetTurn: {
            rawTurnId: input.rawTurn.id,
            user: preview(input.rawTurn.userText),
            assistant: preview(input.rawTurn.assistantText)
          },
          events: input.events.map(boundaryPromptEvent),
          candidates: input.candidates
        })
      }
    ], {
      operation: BOUNDARY_OPERATION,
      thinkingMode: "disabled",
      temperature: 0,
      maxTokens: 1_500
    });
    return parseAcceptedBoundaryIds(result.decisions, input.candidates);
  }

  private async extractSemantics(
    episodeId: string,
    frames: readonly TurnFrame[]
  ): Promise<{
    observations: ObservationSemantics[];
    actions: ActionSemantics[];
  }> {
    const allowedRefs = allowedSourceRefs(frames);
    const segments = frames.flatMap((frame) => frame.segments);
    if (segments.length === 0) {
      return { observations: [], actions: [] };
    }
    const result = await this.deps.llm.completeJson<{
      observations?: unknown;
      actions?: unknown;
    }>([
      { role: "system", content: STATE_DELTA_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId,
          observations: frames.map((frame) => ({
            sourceId: frame.userSourceId,
            rawTurnId: frame.rawTurn.id,
            userText: preview(frame.rawTurn.userText)
          })),
          actionSegments: segments.map((segment) => {
            const frame = frames[segment.turnIndex]!;
            return {
              segmentId: segment.id,
              rawTurnId: segment.rawTurnId,
              actionKind: segment.kind,
              eventRefs: segment.events.map((event) => event.id),
              sourceRefs: segment.sourceRefs,
              evidence: segment.events.map((event) => ({
                eventId: event.id,
                action: event.action,
                success: event.success,
                evidence: event.evidence
              })),
              assistantFinalAnswer: preview(frame.rawTurn.assistantText),
              assistantSourceId: frame.assistantSourceId
            };
          })
        })
      }
    ], {
      operation: STATE_DELTA_OPERATION,
      thinkingMode: "disabled",
      temperature: 0,
      maxTokens: Math.max(2_500, segments.length * 600)
    });
    return {
      observations: parseObservationSemantics(result.observations, frames, allowedRefs),
      actions: parseActionSemantics(result.actions, segments, allowedRefs)
    };
  }
}

export function buildEpisodeActionEvents(
  rawTurn: RawTurnRecord,
  turnIndex: number
): EpisodeActionEventV1[] {
  const enrichedCalls = enrichToolCalls(rawTurn.toolCalls, rawTurn.toolResults);
  const trajectory = buildSpanTrajectory(enrichedCalls);
  if (trajectory.length > 0) {
    return trajectory.map((event, eventIndex): EpisodeActionEventV1 => {
      const sourceRefs = toolSourceRefs(rawTurn.id, event.range, rawTurn.toolResults);
      return {
        id: actionEventId(rawTurn.id, event.range),
        rawTurnId: rawTurn.id,
        turnIndex,
        eventIndex,
        kind: "tool_action",
        action: event.action,
        family: actionFamily(event.action),
        success: event.success,
        callCount: event.callCount,
        errorCount: event.success ? 0 : event.callCount,
        sourceRefs,
        ...(event.evidence ? { evidence: clip(redactSensitiveText(event.evidence), EVENT_EVIDENCE_MAX) } : {}),
        toolCallRange: event.range
      };
    });
  }
  if (!rawTurn.assistantText?.trim()) return [];
  return [{
    id: `turn:${rawTurn.id}:response`,
    rawTurnId: rawTurn.id,
    turnIndex,
    eventIndex: 0,
    kind: "response_generation",
    action: "respond",
    family: "communicate",
    success: rawTurn.status !== "failed",
    callCount: 0,
    errorCount: rawTurn.status === "failed" ? 1 : 0,
    sourceRefs: [assistantSourceId(rawTurn.id)],
    evidence: preview(rawTurn.assistantText)
  }];
}

export function proposeSpanBoundaryCandidates(
  events: readonly EpisodeActionEventV1[]
): SpanBoundaryCandidateV1[] {
  const candidates: SpanBoundaryCandidateV1[] = [];
  for (let index = 0; index < events.length - 1; index += 1) {
    const left = events[index]!;
    const right = events[index + 1]!;
    const signals: SpanBoundaryCandidateV1["signals"] = [];
    if (left.family !== right.family) signals.push("action_family_change");
    if (left.success !== right.success) signals.push("outcome_change");
    if (signals.length === 0) continue;
    const basis = { rawTurnId: left.rawTurnId, after: left.id, before: right.id, signals };
    candidates.push({
      id: `boundary_${stableHash(basis).slice(0, 16)}`,
      rawTurnId: left.rawTurnId,
      afterEventId: left.id,
      beforeEventId: right.id,
      signals
    });
  }
  return candidates;
}

export function segmentActionEvents(
  rawTurnId: string,
  turnIndex: number,
  events: readonly EpisodeActionEventV1[],
  candidates: readonly SpanBoundaryCandidateV1[],
  acceptedBoundaryIds: ReadonlySet<string>
): ActionSegmentV1[] {
  if (events.length === 0) return [];
  const acceptedAfter = new Set(
    candidates
      .filter((candidate) => acceptedBoundaryIds.has(candidate.id))
      .map((candidate) => candidate.afterEventId)
  );
  const groups: EpisodeActionEventV1[][] = [];
  let current: EpisodeActionEventV1[] = [];
  for (const event of events) {
    current.push(event);
    if (acceptedAfter.has(event.id)) {
      groups.push(current);
      current = [];
    }
  }
  if (current.length > 0) groups.push(current);
  return groups.map((group, segmentIndex) => {
    const sourceRefs = unique(group.flatMap((event) => [event.id, ...event.sourceRefs]));
    const toolRanges = group
      .map((event) => event.toolCallRange)
      .filter((range): range is [number, number] => Boolean(range));
    const basis = { rawTurnId, segmentIndex, eventIds: group.map((event) => event.id) };
    return {
      id: `segment_${stableHash(basis).slice(0, 20)}`,
      rawTurnId,
      turnIndex,
      segmentIndex,
      kind: group.every((event) => event.kind === "response_generation")
        ? "response_generation"
        : "tool_sequence",
      events: group,
      sourceRefs,
      ...(toolRanges.length > 0
        ? { toolCallRange: [toolRanges[0]![0], toolRanges.at(-1)![1]] as [number, number] }
        : {})
    };
  });
}

function assembleExecutionPath(input: {
  episodeId: string;
  frames: readonly TurnFrame[];
  semantics: {
    observations: readonly ObservationSemantics[];
    actions: readonly ActionSemantics[];
  };
  sourceSnapshotHash: string;
  model?: string;
  terminalReward?: number;
}): EpisodeExecutionPathV1 {
  const observationBySource = new Map(
    input.semantics.observations.map((observation) => [observation.sourceId, observation.operations])
  );
  const actionBySegment = new Map(
    input.semantics.actions.map((action) => [action.segmentId, action])
  );
  const items: Array<{
    segment: ActionSegmentV1;
    semantics: ActionSemantics;
    preObservationOps: StateDeltaOperation[];
  }> = [];
  let pendingObservationOps: StateDeltaOperation[] = [];
  for (const frame of input.frames) {
    pendingObservationOps.push(...(observationBySource.get(frame.userSourceId) ?? []));
    let consumedForTurn = false;
    for (const segment of frame.segments) {
      const semantics = actionBySegment.get(segment.id);
      if (!semantics || !semantics.include) continue;
      items.push({
        segment,
        semantics,
        preObservationOps: consumedForTurn ? [] : pendingObservationOps
      });
      pendingObservationOps = [];
      consumedForTurn = true;
    }
  }

  let state = emptyObservedState();
  const states: ReturnType<typeof emptyObservedState>[] = [];
  const spans: SpanV3[] = [];
  if (items.length === 0) {
    state = applyStateDelta(state, pendingObservationOps);
    states.push(state);
    return buildEpisodeExecutionPath({
      episodeId: input.episodeId,
      states,
      spans,
      sourceSnapshotHash: input.sourceSnapshotHash,
      ...(input.terminalReward === undefined ? {} : { terminalReward: input.terminalReward })
    });
  }

  state = applyStateDelta(state, items[0]!.preObservationOps);
  states.push(state);
  for (const [spanIndex, item] of items.entries()) {
    const preState = state;
    const actionState = applyStateDelta(preState, item.semantics.operations);
    const externalObservationDelta = items[spanIndex + 1]?.preObservationOps ?? pendingObservationOps;
    const postState = applyStateDelta(actionState, externalObservationDelta);
    states.push(postState);
    const eventRefs = unique(item.segment.events.map((event) => event.id));
    const errorCount = item.segment.events.reduce((sum, event) => sum + event.errorCount, 0);
    const toolCalls = item.segment.events.reduce((sum, event) => sum + event.callCount, 0);
    const spanBasis = {
      schemaVersion: SPAN_V3_SCHEMA_VERSION,
      episodeId: input.episodeId,
      rawTurnId: item.segment.rawTurnId,
      segmentId: item.segment.id,
      preStateId: preState.id,
      postStateId: postState.id,
      sourceSnapshotHash: input.sourceSnapshotHash
    };
    spans.push({
      id: `span_${stableHash(spanBasis).slice(0, 20)}`,
      schemaVersion: SPAN_V3_SCHEMA_VERSION,
      episodeId: input.episodeId,
      rawTurnId: item.segment.rawTurnId,
      spanIndex,
      preStateId: preState.id,
      action: {
        kind: item.segment.kind,
        localGoal: item.semantics.localGoal,
        summary: item.semantics.summary,
        eventRefs,
        ...(item.segment.toolCallRange ? { toolCallRange: item.segment.toolCallRange } : {})
      },
      actionEffectDelta: item.semantics.operations,
      externalObservationDelta,
      postStateId: postState.id,
      cost: {
        toolCalls,
        retryCount: 0,
        errorCount
      },
      outcome: {
        status: item.semantics.outcomeStatus,
        verificationRefs: item.semantics.verificationRefs,
        ...(spanIndex === items.length - 1 && input.terminalReward !== undefined
          ? { episodeReward: input.terminalReward }
          : {})
      },
      provenance: {
        algorithmVersion: SPAN_V3_RECONSTRUCTION_VERSION,
        ...(input.model ? { model: input.model } : {}),
        sourceSnapshotHash: input.sourceSnapshotHash
      }
    });
    state = postState;
  }

  return buildEpisodeExecutionPath({
    episodeId: input.episodeId,
    states,
    spans,
    sourceSnapshotHash: input.sourceSnapshotHash,
    ...(input.terminalReward === undefined ? {} : { terminalReward: input.terminalReward })
  });
}

function parseAcceptedBoundaryIds(
  value: unknown,
  candidates: readonly SpanBoundaryCandidateV1[]
): Set<string> {
  if (!Array.isArray(value)) throw new Error("span.v3 boundary LLM returned invalid decisions");
  const known = new Set(candidates.map((candidate) => candidate.id));
  const decided = new Set<string>();
  const accepted = new Set<string>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.candidate_id !== "string" || typeof item.split !== "boolean") {
      throw new Error("span.v3 boundary LLM returned invalid decision");
    }
    if (!known.has(item.candidate_id)) {
      throw new Error(`span.v3 boundary LLM invented candidate: ${item.candidate_id}`);
    }
    if (decided.has(item.candidate_id)) {
      throw new Error(`span.v3 boundary LLM duplicated candidate: ${item.candidate_id}`);
    }
    decided.add(item.candidate_id);
    if (item.split) accepted.add(item.candidate_id);
  }
  if (decided.size !== known.size) {
    const omitted = candidates
      .map((candidate) => candidate.id)
      .filter((candidateId) => !decided.has(candidateId));
    throw new Error(`span.v3 boundary LLM omitted candidates: ${omitted.join(", ")}`);
  }
  return accepted;
}

function parseObservationSemantics(
  value: unknown,
  frames: readonly TurnFrame[],
  allowedRefs: ReadonlySet<string>
): ObservationSemantics[] {
  if (!Array.isArray(value)) throw new Error("span.v3 state LLM returned invalid observations");
  const known = new Set(frames.map((frame) => frame.userSourceId));
  const bySource = new Map<string, ObservationSemantics>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.source_id !== "string" || !known.has(item.source_id)) {
      throw new Error("span.v3 state LLM returned unknown observation source");
    }
    bySource.set(item.source_id, {
      sourceId: item.source_id,
      operations: parseStateDeltaOperations(item.operations, allowedRefs)
    });
  }
  return frames.map((frame) => bySource.get(frame.userSourceId) ?? {
    sourceId: frame.userSourceId,
    operations: []
  });
}

function parseActionSemantics(
  value: unknown,
  segments: readonly ActionSegmentV1[],
  allowedRefs: ReadonlySet<string>
): ActionSemantics[] {
  if (!Array.isArray(value)) throw new Error("span.v3 state LLM returned invalid actions");
  const known = new Set(segments.map((segment) => segment.id));
  const parsed = new Map<string, ActionSemantics>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.segment_id !== "string" || !known.has(item.segment_id)) {
      throw new Error("span.v3 state LLM returned unknown action segment");
    }
    if (typeof item.include !== "boolean") {
      throw new Error("span.v3 state LLM returned action without include flag");
    }
    const localGoal = cleanText(item.local_goal, 500);
    const summary = cleanText(item.summary, 1_500);
    if (item.include && (!localGoal || !summary)) {
      throw new Error("span.v3 state LLM returned incomplete action semantics");
    }
    const outcomeStatus = parseOutcomeStatus(item.outcome_status);
    const verificationRefs = stringArray(item.verification_refs);
    assertAllowedRefs(verificationRefs, allowedRefs);
    parsed.set(item.segment_id, {
      segmentId: item.segment_id,
      include: item.include,
      localGoal: localGoal ?? "non-procedural response",
      summary: summary ?? "No substantive task effect.",
      outcomeStatus,
      verificationRefs,
      operations: parseStateDeltaOperations(item.operations, allowedRefs)
    });
  }
  for (const segment of segments) {
    if (!parsed.has(segment.id)) {
      throw new Error(`span.v3 state LLM omitted action segment: ${segment.id}`);
    }
  }
  return segments.map((segment) => parsed.get(segment.id)!);
}

function parseStateDeltaOperations(
  value: unknown,
  allowedRefs: ReadonlySet<string>
): StateDeltaOperation[] {
  if (!Array.isArray(value)) return [];
  return value.map((item): StateDeltaOperation => {
    if (!isRecord(item) || !isStateDeltaOp(item.op)) {
      throw new Error("span.v3 state LLM returned invalid state operation");
    }
    const subject = cleanText(item.subject, 300);
    const sourceRefs = stringArray(item.source_refs);
    if (!subject || sourceRefs.length === 0) {
      throw new Error("span.v3 state operation requires subject and source_refs");
    }
    assertAllowedRefs(sourceRefs, allowedRefs);
    const operation: StateDeltaOperation = {
      op: item.op,
      subject,
      ...(item.value === undefined ? {} : { value: item.value }),
      ...(typeof item.status === "string" && item.status.trim()
        ? { status: clip(redactSensitiveText(item.status.trim()), 120) }
        : {}),
      sourceRefs
    };
    if (operation.op === "status.set") {
      const status = operation.status ?? (
        typeof operation.value === "string" ? operation.value : undefined
      );
      if (!isTaskStatus(status)) {
        throw new Error("span.v3 status.set requires active, blocked, completed, or failed");
      }
    }
    return operation;
  });
}

function allowedSourceRefs(frames: readonly TurnFrame[]): Set<string> {
  return new Set(frames.flatMap((frame) => [
    frame.userSourceId,
    frame.assistantSourceId,
    ...frame.events.flatMap((event) => [event.id, ...event.sourceRefs])
  ]));
}

function assertAllowedRefs(refs: readonly string[], allowed: ReadonlySet<string>): void {
  for (const ref of refs) {
    if (!allowed.has(ref)) throw new Error(`span.v3 state LLM invented source ref: ${ref}`);
  }
}

function enrichToolCalls(toolCalls: readonly unknown[], toolResults: readonly unknown[]): unknown[] {
  return toolCalls.map((call, index) => {
    if (!isRecord(call) || typeof call.name !== "string") return call;
    const result = matchingToolResult(call, index, toolResults);
    if (!result) return call;
    return {
      ...call,
      ...(call.output === undefined && result.output !== undefined ? { output: result.output } : {}),
      ...(call.error === undefined && typeof result.error === "string" ? { error: result.error } : {}),
      ...(call.errorCode === undefined && typeof result.errorCode === "string" ? { errorCode: result.errorCode } : {}),
      ...(call.success === undefined && typeof result.success === "boolean"
        ? { success: result.success }
        : {})
    };
  });
}

function matchingToolResult(
  call: Record<string, unknown>,
  index: number,
  toolResults: readonly unknown[]
): Record<string, unknown> | undefined {
  const callId = text(call.id) ?? text(call.toolCallId);
  const name = text(call.name);
  const exact = toolResults.find((result) => {
    if (!isRecord(result)) return false;
    const resultId = text(result.toolCallId) ?? text(result.id);
    return Boolean(callId && resultId === callId);
  });
  if (isRecord(exact)) return exact;
  const positional = toolResults[index];
  if (isRecord(positional)) {
    const resultName = text(positional.name) ?? text(positional.toolName);
    if (!name || !resultName || name === resultName) return positional;
  }
  return undefined;
}

function toolSourceRefs(
  rawTurnId: string,
  range: [number, number],
  toolResults: readonly unknown[]
): string[] {
  const refs: string[] = [];
  for (let index = range[0]; index <= range[1]; index += 1) {
    refs.push(`turn:${rawTurnId}:tool:${index}`);
    if (toolResults[index] !== undefined) refs.push(`turn:${rawTurnId}:tool_result:${index}`);
  }
  return refs;
}

function actionEventId(rawTurnId: string, range: [number, number]): string {
  return `turn:${rawTurnId}:action:${range[0]}-${range[1]}`;
}

function userSourceId(rawTurnId: string): string {
  return `turn:${rawTurnId}:user`;
}

function assistantSourceId(rawTurnId: string): string {
  return `turn:${rawTurnId}:assistant`;
}

function actionFamily(action: SpanTrajectoryAction): ActionFamily {
  switch (action) {
    case "search":
    case "fetch":
    case "read":
      return "observe";
    case "write":
    case "edit":
      return "mutate";
    case "execute":
      return "execute";
    case "verify":
      return "verify";
    case "communicate":
      return "communicate";
    default:
      return "other";
  }
}

function boundaryPromptEvent(event: EpisodeActionEventV1): Record<string, unknown> {
  return {
    id: event.id,
    action: event.action,
    family: event.family,
    success: event.success,
    callCount: event.callCount,
    evidence: event.evidence
  };
}

function sourceTurnSnapshot(turn: RawTurnRecord): Record<string, unknown> {
  return {
    id: turn.id,
    episodeId: turn.episodeId,
    turnId: turn.turnId,
    userText: turn.userText,
    assistantText: turn.assistantText,
    toolCalls: turn.toolCalls,
    toolResults: turn.toolResults,
    status: turn.status,
    createdAt: turn.createdAt
  };
}

function preview(value: string | undefined): string | undefined {
  if (!value?.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), TEXT_PREVIEW_MAX);
}

function cleanText(value: unknown, maxChars: number): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), maxChars);
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return unique(value.filter((item): item is string => typeof item === "string" && Boolean(item.trim())));
}

function unique<T>(values: readonly T[]): T[] {
  return [...new Set(values)];
}

function parseOutcomeStatus(value: unknown): ActionSemantics["outcomeStatus"] {
  return value === "success" || value === "failure" || value === "partial" || value === "unknown"
    ? value
    : "unknown";
}

function isStateDeltaOp(value: unknown): value is StateDeltaOp {
  return value === "goal.set" ||
    value === "goal.refine" ||
    value === "goal.complete" ||
    value === "constraint.upsert" ||
    value === "constraint.remove" ||
    value === "fact.upsert" ||
    value === "fact.invalidate" ||
    value === "artifact.upsert" ||
    value === "artifact.verify" ||
    value === "issue.upsert" ||
    value === "issue.resolve" ||
    value === "verification.set" ||
    value === "status.set";
}

function isTaskStatus(value: unknown): value is "active" | "blocked" | "completed" | "failed" {
  return value === "active" || value === "blocked" || value === "completed" || value === "failed";
}
