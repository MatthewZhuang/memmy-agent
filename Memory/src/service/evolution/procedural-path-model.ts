import { stableHash } from "../../utils/id.js";
import {
  type ObservedStateV1,
  type StateDeltaOperation
} from "./span-v3-model.js";

export const EXECUTION_STEP_SCHEMA_VERSION = "execution-step.v1" as const;
export const PROCEDURAL_SPAN_SCHEMA_VERSION = "procedural-span.v1" as const;
export const EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION = "episode-execution-path.v2" as const;

export function episodeRewardHash(input: { rTask?: number; rewardDetail: unknown }): string {
  return stableHash({ rTask: input.rTask, rewardDetail: input.rewardDetail });
}

export type ExecutionStepKind = "tool_action" | "response_generation";
export type ExecutionStepOutcome = "success" | "failure" | "partial" | "unknown";
export type ProceduralSpanTermination = "success" | "failure" | "blocked" | "abandoned";

export interface ExecutionStepCostV1 {
  toolCalls: number;
  errorCount: number;
  tokens?: number;
  latencyMs?: number;
}

export interface ExecutionStepV1 {
  id: string;
  schemaVersion: typeof EXECUTION_STEP_SCHEMA_VERSION;
  episodeId: string;
  rawTurnId: string;
  turnIndex: number;
  stepIndex: number;
  preStateId: string;
  action: {
    kind: ExecutionStepKind;
    type: string;
    intent: string;
    summary: string;
    eventRefs: string[];
    toolName?: string;
    toolCallIndex?: number;
  };
  actionEffectDelta: StateDeltaOperation[];
  actionPostStateId: string;
  externalObservationDelta: StateDeltaOperation[];
  postStateId: string;
  outcome: {
    status: ExecutionStepOutcome;
    evidenceRefs: string[];
  };
  retryOfStepId?: string;
  recoveryFromStepId?: string;
  cost: ExecutionStepCostV1;
  provenance: {
    algorithmVersion: string;
    model?: string;
    sourceSnapshotHash: string;
  };
}

export interface SpanSegmentationDecisionV1 {
  spanIndex: number;
  stepIds: string[];
  localGoal: string;
  /** Populated after final Span boundaries are fixed; absent only on legacy paths. */
  capabilityGoal?: string;
  /** Canonical reusable strategy, compiled with capabilityGoal after boundaries are fixed. */
  procedureSemantic?: string;
  entryCondition: string;
  exitCondition: string;
  terminationStatus: ProceduralSpanTermination;
  evidenceRefs: string[];
  reason: string;
  confidence: number;
}

export interface ProceduralSpanV1 {
  id: string;
  schemaVersion: typeof PROCEDURAL_SPAN_SCHEMA_VERSION;
  episodeId: string;
  spanIndex: number;
  localGoal: string;
  /** Optional only so paths persisted before reconstruction v5 remain readable. */
  capabilityGoal?: string;
  /** Optional only so legacy paths can fall back to their ordered Step intents. */
  procedureSemantic?: string;
  entryCondition: string;
  stepIds: string[];
  rawTurnIds: string[];
  preStateId: string;
  postStateId: string;
  termination: {
    status: ProceduralSpanTermination;
    exitCondition: string;
    evidenceRefs: string[];
  };
  cost: {
    steps: number;
    toolCalls: number;
    retryCount: number;
    recoveryCount: number;
    errorCount: number;
    tokens?: number;
    latencyMs?: number;
  };
  segmentation: {
    reason: string;
    confidence: number;
  };
  provenance: {
    algorithmVersion: string;
    model?: string;
    sourceSnapshotHash: string;
  };
}

export interface EpisodeProceduralPathV2 {
  id: string;
  schemaVersion: typeof EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION;
  episodeId: string;
  states: ObservedStateV1[];
  steps: ExecutionStepV1[];
  spans: ProceduralSpanV1[];
  segmentationDecisions: SpanSegmentationDecisionV1[];
  terminalReward?: number;
  sourceSnapshotHash: string;
  pathHash: string;
}

export function buildEpisodeProceduralPath(input: {
  episodeId: string;
  states: readonly ObservedStateV1[];
  steps: readonly ExecutionStepV1[];
  spans: readonly ProceduralSpanV1[];
  segmentationDecisions: readonly SpanSegmentationDecisionV1[];
  sourceSnapshotHash: string;
  terminalReward?: number;
}): EpisodeProceduralPathV2 {
  validateExecutionStepContinuity(input.steps);
  validateProceduralSpanCoverage(input.steps, input.spans);
  const states = uniqueStates(input.states);
  const stateIds = new Set(states.map((state) => state.id));
  for (const step of input.steps) {
    for (const stateId of [step.preStateId, step.actionPostStateId, step.postStateId]) {
      if (!stateIds.has(stateId)) {
        throw new Error(`execution-step references missing state: ${step.id} -> ${stateId}`);
      }
    }
  }
  const hashBasis = {
    schemaVersion: EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION,
    episodeId: input.episodeId,
    states,
    steps: [...input.steps],
    spans: [...input.spans],
    segmentationDecisions: [...input.segmentationDecisions],
    sourceSnapshotHash: input.sourceSnapshotHash,
    ...(input.terminalReward === undefined ? {} : { terminalReward: input.terminalReward })
  };
  const pathHash = stableHash(hashBasis);
  return {
    id: `episode_procedural_path_${pathHash.slice(0, 20)}`,
    ...hashBasis,
    pathHash
  };
}

export function validateExecutionStepContinuity(steps: readonly ExecutionStepV1[]): void {
  for (const [index, step] of steps.entries()) {
    if (step.stepIndex !== index) {
      throw new Error(`execution-step index mismatch: expected ${index}, got ${step.stepIndex}`);
    }
    const previous = steps[index - 1];
    if (previous && previous.postStateId !== step.preStateId) {
      throw new Error(`execution-step discontinuity between ${previous.id} and ${step.id}`);
    }
  }
}

export function validateProceduralSpanCoverage(
  steps: readonly ExecutionStepV1[],
  spans: readonly ProceduralSpanV1[]
): void {
  const expectedStepIds = steps.map((step) => step.id);
  const coveredStepIds = spans.flatMap((span) => span.stepIds);
  if (coveredStepIds.length !== expectedStepIds.length ||
    coveredStepIds.some((stepId, index) => stepId !== expectedStepIds[index])) {
    throw new Error("procedural spans must cover every execution step exactly once and in order");
  }
  const stepById = new Map(steps.map((step) => [step.id, step]));
  for (const [index, span] of spans.entries()) {
    if (span.spanIndex !== index) {
      throw new Error(`procedural-span index mismatch: expected ${index}, got ${span.spanIndex}`);
    }
    if (span.stepIds.length === 0) throw new Error(`procedural-span has no steps: ${span.id}`);
    const first = stepById.get(span.stepIds[0]!);
    const last = stepById.get(span.stepIds.at(-1)!);
    if (!first || !last || span.preStateId !== first.preStateId || span.postStateId !== last.postStateId) {
      throw new Error(`procedural-span state boundary mismatch: ${span.id}`);
    }
    const previous = spans[index - 1];
    if (previous && previous.postStateId !== span.preStateId) {
      throw new Error(`procedural-span discontinuity between ${previous.id} and ${span.id}`);
    }
  }
}

function uniqueStates(states: readonly ObservedStateV1[]): ObservedStateV1[] {
  const byId = new Map<string, ObservedStateV1>();
  for (const state of states) byId.set(state.id, state);
  return [...byId.values()];
}
