import type { LlmClient, LlmMessage } from "../../model/types.js";
import type { RawTurnRecord } from "../../storage/repositories.js";
import { stableHash, stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { redactSensitiveText } from "../../utils/sensitive-data.js";
import { clip } from "../../utils/text.js";
import {
  applyStateDelta,
  emptyObservedState,
  type ObservedStateV1,
  type StateDeltaOp,
  type StateDeltaOperation
} from "./span-v3-model.js";
import {
  buildEpisodeProceduralPath,
  EXECUTION_STEP_SCHEMA_VERSION,
  PROCEDURAL_SPAN_SCHEMA_VERSION,
  type EpisodeProceduralPathV2,
  type ExecutionStepOutcome,
  type ExecutionStepV1,
  type ProceduralSpanTermination,
  type ProceduralSpanV1,
  type SpanSegmentationDecisionV1
} from "./procedural-path-model.js";
import { buildSpanTrajectory, type SpanTrajectoryAction } from "./span-trajectory.js";

export const EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION = "episode-procedural-reconstruction.v3" as const;

const STEP_SEMANTICS_OPERATION = "procedural.step_semantics.v1";
const SPAN_SEGMENTATION_OPERATION = "procedural.span_segmentation.v3.window";
const SPAN_RECONCILIATION_OPERATION = "procedural.span_reconciliation.v2";
const MAX_SEMANTIC_REPAIR_ATTEMPTS = 2;
const STEP_WINDOW_MAX_CANDIDATES = 30;
const STEP_WINDOW_INPUT_CHAR_BUDGET = 30_000;
const STEP_WINDOW_OVERLAP = 5;
const PREVIOUS_TURN_CONTEXT_COUNT = 3;
const TURN_CONTEXT_TEXT_MAX = 700;
const STEP_CONTEXT_TEXT_MAX = 500;
const RECONCILIATION_PROCEDURE_INTENT_MAX = 10;
const TEXT_PREVIEW_MAX = 1_500;
const EVENT_EVIDENCE_MAX = 900;
const STATE_OPERATION_OUTPUT_KEYS = new Set(["op", "subject", "value", "status", "source_refs"]);

export const EXECUTION_STEP_SEMANTICS_PROMPT = `You reconstruct evidence-grounded execution steps from an agent episode.

Each supplied step candidate is one observable action and its immediate result. Annotate what the action attempted, what happened, and only the state changes supported by supplied evidence. Do not group candidates into subtasks in this pass.

Tool-action candidates are procedural and must include=true. A response-generation candidate may include=false only when it is purely social or reports no substantive task effect. Every candidate must appear exactly once.

Observation operations describe exogenous goals, constraints, corrections, feedback, or acceptance introduced by the user/environment. Step operations describe effects actually produced or discovered by that action. Never claim user-provided information as an action effect.

Use retry_of_candidate_id only when the current action directly repeats substantially the same action against the same target after failure or inadequate progress. A changed query, alternate source, or changed strategy is not a retry.

Use recovery_from_candidate_id when the current action changes strategy, source, or method in response to a prior failed/inadequate candidate. Both fields must reference an earlier supplied candidate and must not both be set on one step.

A tool result with deterministic failure evidence must use outcome_status=failure. Use unknown only when the result is genuinely indeterminate, not for an observed error, HTTP failure, or failed command.

Allowed operations:
- goal.set, goal.refine, goal.complete
- constraint.upsert, constraint.remove
- fact.upsert, fact.invalidate
- artifact.upsert, artifact.verify
- issue.upsert, issue.resolve
- verification.set
- status.set, whose value/status must be one of: active, blocked, completed, failed

HARD JSON CONTRACT FOR STATE OPERATIONS (exact and case-sensitive):
- Every item in observations[*].operations and steps[*].operations must use exactly these field names: op, subject, optional value, optional status, source_refs.
- The operation name field MUST be named "op". Never use "operation", "operation_type", "action", or any other alias.
- The evidence field MUST be named "source_refs". Never use "sourceRefs", "sources", or any other alias.
- Do not add fields outside: op, subject, value, status, source_refs.
- If there is no evidence-grounded state change, return an empty operations array.
- Every operation must contain a non-empty subject and a non-empty source_refs array.
- Every operation and outcome evidence reference must cite supplied source_refs. Never invent a source reference.

The only valid operation object shape is:
{
  "op": "fact.upsert",
  "subject": "dependency_status",
  "value": "incompatible versions observed",
  "source_refs": ["turn:...:tool_result:0"]
}

Return JSON only:
{
  "observations": [
    {
      "source_id": "turn:...:user",
      "operations": [
        {
          "op": "goal.set",
          "subject": "task_goal",
          "value": "restore the failing tests",
          "source_refs": ["turn:...:user"]
        }
      ]
    }
  ],
  "steps": [
    {
      "candidate_id": "candidate_...",
      "include": true,
      "intent": "inspect the failing dependency resolution",
      "summary": "Read the lockfile and observed incompatible versions.",
      "outcome_status": "success",
      "evidence_refs": ["turn:...:tool_result:0"],
      "retry_of_candidate_id": null,
      "recovery_from_candidate_id": null,
      "operations": [
        {
          "op": "fact.upsert",
          "subject": "dependency_status",
          "value": "incompatible versions observed",
          "source_refs": ["turn:...:tool_result:0"]
        }
      ]
    }
  ]
}`;

export const PROCEDURAL_SPAN_SEGMENTATION_PROMPT = `You partition one bounded window of new execution steps into provisional procedural spans while preserving continuity with earlier episode context.

A procedural span is the full lifecycle of one local subproblem: it starts when that subproblem becomes active and ends only when it succeeds, fails terminally, becomes blocked, is abandoned, or control returns to the parent task to select a materially different subproblem.

The supplied steps array contains only new steps that must be partitioned. precedingStepContext, previousTurns, and openSpan are read-only context. Never include their IDs in step_ids. Partition the new steps into the minimum number of contiguous provisional spans that preserves real subproblem handoffs.

Important rules:
- A tool-family change is not by itself a span boundary.
- A conversation turn boundary is not by itself a span boundary.
- A computation window boundary is not by itself a span boundary.
- Diagnosis, an attempted fix, failed verification, recovery, a corrected fix, and successful verification normally remain one span when they continue solving the same local subproblem.
- User feedback or clarification may occur inside a span when the same subproblem continues.
- Split only when the prior local subproblem has terminated or control has switched to a materially different local goal.
- Every supplied step must belong to exactly one span, in original order, with no overlap, omission, or reordering.
- continues_previous may be true only for the first returned span and only when it continues the supplied openSpan.
- open_at_window_end must be true only for the final returned span when its local subproblem remains unresolved at the end of this window. Use window.hasFollowingWindow and window.hasFollowingTurn to distinguish a computation frontier from an episode end. It is a streaming marker, not a semantic termination status.
- entry_condition and exit_condition must be observable and evidence-grounded, not hidden chain-of-thought.
- termination_status must be one of success, failure, blocked, abandoned.
- evidence_refs may cite only supplied step IDs or evidence refs.

Local-goal rules:
- local_goal is the canonical core objective of this Span, not a copy of the parent Episode goal.
- Express it as one concise clause containing the concrete target object or capability and the desired observable outcome.
- Ground both the target and outcome in the supplied step intents, summaries, pre/post states, and evidence.
- Do not introduce a problem category, object, cause, or technology that is absent from the supplied evidence.
- Prefer stable domain semantics over transient wording so equivalent executions in different Episodes receive semantically equivalent local_goal values.
- Exclude phase numbers, turn references, user-facing reporting instructions, tool names, implementation sequence, and incidental failed attempts unless they define the subproblem.
- Keep diagnosis, repair, recovery, and verification under the same local_goal when they solve the same subproblem.
- local_goal, entry_condition, exit_condition, reason, and the dominant procedure performed by the steps must describe the same local subproblem.

Return JSON only, conforming to this TypeScript schema. Type names are schema descriptors and must never appear as output values:
type Output = {
  spans: Array<{
    step_ids: string[];
    continues_previous: boolean;
    open_at_window_end: boolean;
    local_goal: string;
    entry_condition: string;
    exit_condition: string;
    termination_status: "success" | "failure" | "blocked" | "abandoned";
    evidence_refs: string[];
    reason: string;
    confidence: number;
  }>;
};`;

export const PROCEDURAL_SPAN_RECONCILIATION_PROMPT = `You reconcile an ordered list of provisional procedural spans from one complete agent episode.

The raw event stream has already been processed in bounded windows. Each provisional span is internally contiguous and cannot be split here. Your job is to merge adjacent provisional spans when they are actually parts of the same local subproblem, especially across computation-window or conversation-turn boundaries.

Rules:
- Use the complete compressed episode view, not turn boundaries, as the semantic frame.
- A computation-window boundary or conversation-turn boundary is never sufficient reason to keep spans separate.
- Merge diagnose, attempted fix, failed verification, recovery, corrected fix, and successful verification when they solve the same local subproblem.
- Keep spans separate when the earlier local subproblem terminated and control moved to a materially different local goal.
- Every provisional_span_id must appear exactly once, in original order, with no overlap, omission, or reordering.
- You may merge adjacent provisional spans but may not split or reorder one.
- entry_condition and exit_condition must be observable and evidence-grounded.
- termination_status must be one of success, failure, blocked, abandoned.
- evidence_refs may cite only supplied step IDs or evidence references.

Final local-goal rules:
- Recompute final local_goal from procedureIntents, entryState, exitState, and evidence. Treat every provisional localGoal as a proposal, not authoritative truth.
- If a provisional localGoal conflicts with the supplied procedureIntents or state transition, correct it.
- Express local_goal as one concise clause containing the concrete target object or capability and the desired observable outcome.
- When merging provisional Spans, describe their shared local subproblem instead of concatenating their labels.
- Do not introduce a problem category, object, cause, or technology absent from the supplied evidence.
- Prefer stable domain semantics over phase numbers, turn references, reporting instructions, tool names, implementation sequence, or transient failed attempts.
- Before returning, verify that local_goal, entry_condition, exit_condition, reason, and procedureIntents all describe the same local subproblem.

Return JSON only, conforming to this TypeScript schema. Type names are schema descriptors and must never appear as output values:
type Output = {
  spans: Array<{
    provisional_span_ids: string[];
    local_goal: string;
    entry_condition: string;
    exit_condition: string;
    termination_status: "success" | "failure" | "blocked" | "abandoned";
    evidence_refs: string[];
    reason: string;
    confidence: number;
  }>;
};`;

export interface ExecutionStepCandidateV1 {
  id: string;
  rawTurnId: string;
  turnIndex: number;
  eventIndex: number;
  kind: "tool_action" | "response_generation";
  action: SpanTrajectoryAction | "respond";
  toolName?: string;
  toolCallIndex?: number;
  heuristicSuccess: boolean;
  sourceRefs: string[];
  evidence?: string;
  toolCalls: number;
  errorCount: number;
}

interface TurnFrame {
  rawTurn: RawTurnRecord;
  turnIndex: number;
  userSourceId: string;
  candidates: ExecutionStepCandidateV1[];
}

interface ObservationSemantics {
  sourceId: string;
  operations: StateDeltaOperation[];
}

interface StepSemantics {
  candidateId: string;
  include: boolean;
  intent: string;
  summary: string;
  outcomeStatus: ExecutionStepOutcome;
  evidenceRefs: string[];
  retryOfCandidateId?: string;
  recoveryFromCandidateId?: string;
  operations: StateDeltaOperation[];
}

interface ProvisionalSpanDecision extends SpanSegmentationDecisionV1 {
  id: string;
  rawTurnId: string;
  windowIndex: number;
  continuesPrevious: boolean;
  openAtWindowEnd: boolean;
}

interface ReconstructorDeps {
  llm: LlmClient;
}

export class EpisodeProceduralReconstructor {
  constructor(private readonly deps: ReconstructorDeps) {}

  async reconstruct(input: {
    episodeId: string;
    rawTurns: readonly RawTurnRecord[];
    terminalReward?: number;
  }): Promise<EpisodeProceduralPathV2> {
    if (!this.deps.llm.isConfigured()) {
      throw new Error("procedural reconstruction requires a configured LLM");
    }
    const rawTurns = [...input.rawTurns]
      .filter((turn) => turn.episodeId === input.episodeId && !turn.deletedAt)
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id));
    const sourceSnapshotHash = stableHash(rawTurns.map(sourceTurnSnapshot));
    const frames = rawTurns.map((rawTurn, turnIndex): TurnFrame => ({
      rawTurn,
      turnIndex,
      userSourceId: userSourceId(rawTurn.id),
      candidates: buildTurnStepCandidates(rawTurn, turnIndex)
    }));
    const semantics = await this.extractStepSemantics(input.episodeId, frames);
    const materialized = materializeExecutionSteps({
      episodeId: input.episodeId,
      frames,
      semantics,
      sourceSnapshotHash,
      model: this.deps.llm.config.model
    });
    const decisions = await this.segmentSpans({
      episodeId: input.episodeId,
      rawTurns,
      steps: materialized.steps,
      states: materialized.states
    });
    const spans = compileProceduralSpans({
      episodeId: input.episodeId,
      steps: materialized.steps,
      decisions,
      sourceSnapshotHash,
      model: this.deps.llm.config.model
    });
    return buildEpisodeProceduralPath({
      episodeId: input.episodeId,
      states: materialized.states,
      steps: materialized.steps,
      spans,
      segmentationDecisions: decisions,
      sourceSnapshotHash,
      ...(input.terminalReward === undefined ? {} : { terminalReward: input.terminalReward })
    });
  }

  private async extractStepSemantics(
    episodeId: string,
    frames: readonly TurnFrame[]
  ): Promise<{ observations: ObservationSemantics[]; steps: StepSemantics[] }> {
    const allCandidates = frames.flatMap((frame) => frame.candidates);
    if (allCandidates.length === 0) return { observations: [], steps: [] };
    const allowedRefs = candidateSourceRefs(frames);
    const observations: ObservationSemantics[] = [];
    const steps: StepSemantics[] = [];
    const processedCandidates: ExecutionStepCandidateV1[] = [];
    const semanticsByCandidate = new Map<string, StepSemantics>();
    for (const frame of frames) {
      const windows = chunkStepCandidates(frame.candidates);
      for (const [windowIndex, candidates] of windows.entries()) {
        const precedingCandidates = processedCandidates.slice(-STEP_WINDOW_OVERLAP);
        const parsed = await this.extractStepSemanticsWindow({
          episodeId,
          frame,
          previousTurns: frames.slice(
            Math.max(0, frame.turnIndex - PREVIOUS_TURN_CONTEXT_COUNT),
            frame.turnIndex
          ),
          windowIndex,
          observationFrames: windowIndex === 0 ? [frame] : [],
          precedingCandidates,
          candidates,
          referenceCandidates: [...processedCandidates, ...candidates],
          semanticsByCandidate,
          allowedRefs
        });
        observations.push(...parsed.observations);
        steps.push(...parsed.steps);
        for (const semantics of parsed.steps) semanticsByCandidate.set(semantics.candidateId, semantics);
        processedCandidates.push(...candidates);
      }
    }
    return { observations, steps };
  }

  private async extractStepSemanticsWindow(input: {
    episodeId: string;
    frame: TurnFrame;
    previousTurns: readonly TurnFrame[];
    windowIndex: number;
    observationFrames: readonly TurnFrame[];
    precedingCandidates: readonly ExecutionStepCandidateV1[];
    candidates: readonly ExecutionStepCandidateV1[];
    referenceCandidates: readonly ExecutionStepCandidateV1[];
    semanticsByCandidate: ReadonlyMap<string, StepSemantics>;
    allowedRefs: ReadonlySet<string>;
  }): Promise<{ observations: ObservationSemantics[]; steps: StepSemantics[] }> {
    type StepSemanticResult = { observations?: unknown; steps?: unknown };
    const messages: LlmMessage[] = [
      { role: "system", content: EXECUTION_STEP_SEMANTICS_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId: input.episodeId,
          window: {
            rawTurnId: input.frame.rawTurn.id,
            turnIndex: input.frame.turnIndex,
            windowIndex: input.windowIndex,
            maxNewCandidates: STEP_WINDOW_MAX_CANDIDATES
          },
          previousTurns: input.previousTurns.map(compactTurnFrame),
          currentTurn: compactTurnFrame(input.frame),
          precedingStepContext: input.precedingCandidates.map((candidate) => {
            const semantics = input.semanticsByCandidate.get(candidate.id);
            return {
              candidateId: candidate.id,
              rawTurnId: candidate.rawTurnId,
              action: candidate.action,
              toolName: candidate.toolName,
              heuristicSuccess: candidate.heuristicSuccess,
              intent: semantics?.intent,
              summary: contextText(semantics?.summary),
              outcome: semantics?.outcomeStatus,
              retryOfCandidateId: semantics?.retryOfCandidateId,
              recoveryFromCandidateId: semantics?.recoveryFromCandidateId
            };
          }),
          observations: input.observationFrames.map((frame) => ({
            sourceId: frame.userSourceId,
            sourceRefs: [frame.userSourceId],
            rawTurnId: frame.rawTurn.id,
            userText: preview(frame.rawTurn.userText)
          })),
          stepCandidates: input.candidates.map((candidate) => ({
            candidateId: candidate.id,
            rawTurnId: candidate.rawTurnId,
            turnIndex: candidate.turnIndex,
            eventIndex: candidate.eventIndex,
            kind: candidate.kind,
            action: candidate.action,
            toolName: candidate.toolName,
            heuristicSuccess: candidate.heuristicSuccess,
            sourceRefs: candidate.sourceRefs,
            evidence: candidate.evidence
          }))
        })
      }
    ];
    const options = {
      operation: `${STEP_SEMANTICS_OPERATION}.window`,
      thinkingMode: "disabled" as const,
      temperature: 0,
      maxTokens: Math.max(3_000, Math.min(12_000, input.candidates.length * 450))
    };
    let result = await this.deps.llm.completeJson<StepSemanticResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseStepSemanticResult(
          result,
          input.observationFrames,
          input.candidates,
          input.referenceCandidates,
          input.allowedRefs
        );
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<StepSemanticResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: stepSemanticRepairInstruction(error, repairNumber) }
        ], {
          ...options,
          operation: `${STEP_SEMANTICS_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }

  private async segmentSpans(input: {
    episodeId: string;
    rawTurns: readonly RawTurnRecord[];
    steps: readonly ExecutionStepV1[];
    states: readonly ObservedStateV1[];
  }): Promise<SpanSegmentationDecisionV1[]> {
    if (input.steps.length === 0) return [];
    const stateById = new Map(input.states.map((state) => [state.id, state]));
    const allowedRefs = new Set(input.steps.flatMap((step) => [
      step.id,
      ...step.action.eventRefs,
      ...step.outcome.evidenceRefs
    ]));
    const proposals: ProvisionalSpanDecision[] = [];
    let openSpan: ProvisionalSpanDecision | undefined;
    let windowIndex = 0;
    for (const [turnIndex, rawTurn] of input.rawTurns.entries()) {
      const turnSteps = input.steps.filter((step) => step.rawTurnId === rawTurn.id);
      for (let start = 0; start < turnSteps.length; start += STEP_WINDOW_MAX_CANDIDATES) {
        const steps = turnSteps.slice(start, start + STEP_WINDOW_MAX_CANDIDATES);
        const firstStepIndex = steps[0]!.stepIndex;
        const precedingSteps = input.steps.slice(
          Math.max(0, firstStepIndex - STEP_WINDOW_OVERLAP),
          firstStepIndex
        );
        const local = await this.segmentSpanWindow({
          episodeId: input.episodeId,
          rawTurn,
          turnIndex,
          windowIndex,
          hasFollowingWindow: start + STEP_WINDOW_MAX_CANDIDATES < turnSteps.length,
          hasFollowingTurn: turnIndex < input.rawTurns.length - 1,
          previousTurns: input.rawTurns.slice(
            Math.max(0, turnIndex - PREVIOUS_TURN_CONTEXT_COUNT),
            turnIndex
          ),
          precedingSteps,
          steps,
          openSpan,
          stateById,
          allowedRefs
        });
        proposals.push(...local);
        openSpan = local.at(-1)?.openAtWindowEnd ? local.at(-1) : undefined;
        windowIndex += 1;
      }
    }
    if (proposals.length === 0) return [];
    if (windowIndex === 1 && proposals.every((proposal) => !proposal.openAtWindowEnd)) {
      return finalDecisionsFromProposals(proposals);
    }
    return this.reconcileSpanProposals({
      episodeId: input.episodeId,
      rawTurns: input.rawTurns,
      proposals,
      steps: input.steps,
      stateById,
      allowedRefs
    });
  }

  private async segmentSpanWindow(input: {
    episodeId: string;
    rawTurn: RawTurnRecord;
    turnIndex: number;
    windowIndex: number;
    hasFollowingWindow: boolean;
    hasFollowingTurn: boolean;
    previousTurns: readonly RawTurnRecord[];
    precedingSteps: readonly ExecutionStepV1[];
    steps: readonly ExecutionStepV1[];
    openSpan?: ProvisionalSpanDecision;
    stateById: ReadonlyMap<string, ObservedStateV1>;
    allowedRefs: ReadonlySet<string>;
  }): Promise<ProvisionalSpanDecision[]> {
    type SegmentationResult = { spans?: unknown };
    const messages: LlmMessage[] = [
      { role: "system", content: PROCEDURAL_SPAN_SEGMENTATION_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId: input.episodeId,
          window: {
            rawTurnId: input.rawTurn.id,
            turnIndex: input.turnIndex,
            windowIndex: input.windowIndex,
            newStepCount: input.steps.length,
            hasFollowingWindow: input.hasFollowingWindow,
            hasFollowingTurn: input.hasFollowingTurn
          },
          previousTurns: input.previousTurns.map(compactRawTurn),
          currentTurn: compactRawTurn(input.rawTurn),
          precedingStepContext: input.precedingSteps.map((step) => compactExecutionStep(step, input.stateById)),
          openSpan: input.openSpan ? compactProvisionalSpan(input.openSpan) : null,
          steps: input.steps.map((step) => compactExecutionStep(step, input.stateById))
        })
      }
    ];
    const options = {
      operation: SPAN_SEGMENTATION_OPERATION,
      thinkingMode: "disabled" as const,
      temperature: 0,
      maxTokens: Math.max(2_500, Math.min(8_000, input.steps.length * 300))
    };
    let result = await this.deps.llm.completeJson<SegmentationResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseProvisionalSpanDecisions(result.spans, input);
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<SegmentationResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: segmentationRepairInstruction(error, "step_ids", repairNumber) }
        ], {
          ...options,
          operation: `${SPAN_SEGMENTATION_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }

  private async reconcileSpanProposals(input: {
    episodeId: string;
    rawTurns: readonly RawTurnRecord[];
    proposals: readonly ProvisionalSpanDecision[];
    steps: readonly ExecutionStepV1[];
    stateById: ReadonlyMap<string, ObservedStateV1>;
    allowedRefs: ReadonlySet<string>;
  }): Promise<SpanSegmentationDecisionV1[]> {
    type ReconciliationResult = { spans?: unknown };
    const stepById = new Map(input.steps.map((step) => [step.id, step]));
    const messages: LlmMessage[] = [
      { role: "system", content: PROCEDURAL_SPAN_RECONCILIATION_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId: input.episodeId,
          episodeGoal: contextText(input.rawTurns.find((turn) => turn.userText?.trim())?.userText),
          turnCount: input.rawTurns.length,
          stepCount: input.steps.length,
          provisionalSpans: input.proposals.map((proposal) => ({
            provisionalSpanId: proposal.id,
            rawTurnId: proposal.rawTurnId,
            windowIndex: proposal.windowIndex,
            stepIds: proposal.stepIds,
            stepRange: [
              input.steps.find((step) => step.id === proposal.stepIds[0])?.stepIndex,
              input.steps.find((step) => step.id === proposal.stepIds.at(-1))?.stepIndex
            ],
            localGoal: proposal.localGoal,
            entryCondition: proposal.entryCondition,
            exitCondition: proposal.exitCondition,
            terminationStatus: proposal.terminationStatus,
            continuesPrevious: proposal.continuesPrevious,
            openAtWindowEnd: proposal.openAtWindowEnd,
            reason: proposal.reason,
            confidence: proposal.confidence,
            procedureIntents: reconciliationProcedureIntents(proposal, stepById),
            entryState: input.stateById.get(
              stepById.get(proposal.stepIds[0] ?? "")?.preStateId ?? ""
            )?.summary,
            exitState: input.stateById.get(
              stepById.get(proposal.stepIds.at(-1) ?? "")?.postStateId ?? ""
            )?.summary
          }))
        })
      }
    ];
    const options = {
      operation: SPAN_RECONCILIATION_OPERATION,
      thinkingMode: "disabled" as const,
      temperature: 0,
      maxTokens: Math.max(2_500, Math.min(12_000, input.proposals.length * 400))
    };
    let result = await this.deps.llm.completeJson<ReconciliationResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseReconciledSpanDecisions(result.spans, input.proposals, input.allowedRefs);
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<ReconciliationResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: segmentationRepairInstruction(error, "provisional_span_ids", repairNumber) }
        ], {
          ...options,
          operation: `${SPAN_RECONCILIATION_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }
}

function parseStepSemanticResult(
  result: { observations?: unknown; steps?: unknown },
  frames: readonly TurnFrame[],
  candidates: readonly ExecutionStepCandidateV1[],
  referenceCandidates: readonly ExecutionStepCandidateV1[],
  allowedRefs: ReadonlySet<string>
): { observations: ObservationSemantics[]; steps: StepSemantics[] } {
  return {
    observations: parseObservationSemantics(result.observations, frames, allowedRefs),
    steps: parseStepSemantics(result.steps, candidates, referenceCandidates, allowedRefs)
  };
}

function stepSemanticRepairInstruction(error: unknown, repairNumber: number): string {
  const message = error instanceof Error ? error.message : String(error);
  const exactContract = `Every observations[*].operations[*] and steps[*].operations[*] object MUST use the exact case-sensitive field names "op", "subject", optional "value", optional "status", and "source_refs". The operation field MUST be "op"; "operation", "operation_type", and "action" are forbidden. The evidence field MUST be "source_refs"; "sourceRefs" and "sources" are forbidden. The only allowed op values are: goal.set, goal.refine, goal.complete, constraint.upsert, constraint.remove, fact.upsert, fact.invalidate, artifact.upsert, artifact.verify, issue.upsert, issue.resolve, verification.set, status.set.`;
  const common = `Deterministic validation error: ${message}\n${exactContract}\nReturn the complete corrected JSON object only. Preserve every supplied observation source and candidate exactly once. Use only supplied source references. Do not add markdown, explanation, comments, or extra top-level fields.`;
  if (repairNumber === 1) {
    return `This is a strict schema-repair task, not a request to reinterpret the episode. Correct the previous JSON.\n${common}\nINVALID: {"operation":"fact.upsert","subject":"dependency_status","source_refs":["turn:...:tool_result:0"]}\nVALID: {"op":"fact.upsert","subject":"dependency_status","source_refs":["turn:...:tool_result:0"]}`;
  }
  return `The previous repair still failed. Discard the previous JSON and regenerate the entire response from the original episode payload and schema. Do not patch or paraphrase the malformed object.\n${common}\nBefore returning, verify that every operation object contains the literal key "op", contains no key named "operation", and that the complete response is one JSON object.`;
}

function segmentationRepairInstruction(error: unknown, coverageField: string, repairNumber: number): string {
  const message = error instanceof Error ? error.message : String(error);
  const base = `Deterministic validation error: ${message}\nReturn one complete corrected JSON object only. Preserve every supplied ${coverageField} exactly once, in original order, with no overlap, omission, or reordering. Do not add markdown, explanation, or comments.`;
  if (repairNumber === 1) return `Correct the previous segmentation JSON without changing evidence-grounded semantics.\n${base}`;
  return `The previous repair still failed. Discard it and regenerate the entire segmentation JSON from the original payload and schema.\n${base}`;
}

function chunkStepCandidates(
  candidates: readonly ExecutionStepCandidateV1[]
): ExecutionStepCandidateV1[][] {
  const windows: ExecutionStepCandidateV1[][] = [];
  let current: ExecutionStepCandidateV1[] = [];
  let currentChars = 0;
  for (const candidate of candidates) {
    const candidateChars = (candidate.evidence?.length ?? 0) + 350;
    if (current.length > 0 && (
      current.length >= STEP_WINDOW_MAX_CANDIDATES ||
      currentChars + candidateChars > STEP_WINDOW_INPUT_CHAR_BUDGET
    )) {
      windows.push(current);
      current = [];
      currentChars = 0;
    }
    current.push(candidate);
    currentChars += candidateChars;
  }
  if (current.length > 0) windows.push(current);
  return windows;
}

function compactTurnFrame(frame: TurnFrame): Record<string, unknown> {
  return compactRawTurn(frame.rawTurn);
}

function compactRawTurn(turn: RawTurnRecord): Record<string, unknown> {
  return {
    rawTurnId: turn.id,
    user: turnContextText(turn.userText),
    assistant: turnContextText(turn.assistantText),
    status: turn.status
  };
}

function compactExecutionStep(
  step: ExecutionStepV1,
  stateById: ReadonlyMap<string, ObservedStateV1>
): Record<string, unknown> {
  return {
    stepId: step.id,
    stepIndex: step.stepIndex,
    rawTurnId: step.rawTurnId,
    kind: step.action.kind,
    actionType: step.action.type,
    toolName: step.action.toolName,
    intent: contextText(step.action.intent),
    summary: contextText(step.action.summary),
    outcome: step.outcome.status,
    retryOfStepId: step.retryOfStepId,
    recoveryFromStepId: step.recoveryFromStepId,
    evidenceRefs: [...new Set([...step.action.eventRefs, ...step.outcome.evidenceRefs])],
    preState: contextText(stateById.get(step.preStateId)?.summary),
    postState: contextText(stateById.get(step.postStateId)?.summary),
    actionDeltaOps: step.actionEffectDelta.map((operation) => operation.op),
    externalObservationOps: step.externalObservationDelta.map((operation) => operation.op)
  };
}

function compactProvisionalSpan(proposal: ProvisionalSpanDecision): Record<string, unknown> {
  return {
    provisionalSpanId: proposal.id,
    rawTurnId: proposal.rawTurnId,
    windowIndex: proposal.windowIndex,
    stepIds: proposal.stepIds,
    localGoal: proposal.localGoal,
    entryCondition: proposal.entryCondition,
    exitCondition: proposal.exitCondition,
    terminationStatus: proposal.terminationStatus,
    continuesPrevious: proposal.continuesPrevious,
    openAtWindowEnd: proposal.openAtWindowEnd,
    reason: proposal.reason
  };
}

function reconciliationProcedureIntents(
  proposal: ProvisionalSpanDecision,
  stepById: ReadonlyMap<string, ExecutionStepV1>
): string[] {
  const intents = proposal.stepIds.map((stepId) => contextText(stepById.get(stepId)?.action.intent))
    .filter((intent): intent is string => Boolean(intent));
  return [...new Set(intents)].slice(0, RECONCILIATION_PROCEDURE_INTENT_MAX);
}

function finalDecisionsFromProposals(
  proposals: readonly ProvisionalSpanDecision[]
): SpanSegmentationDecisionV1[] {
  return proposals.map((proposal, spanIndex) => ({
    spanIndex,
    stepIds: [...proposal.stepIds],
    localGoal: proposal.localGoal,
    entryCondition: proposal.entryCondition,
    exitCondition: proposal.exitCondition,
    terminationStatus: proposal.terminationStatus,
    evidenceRefs: [...proposal.evidenceRefs],
    reason: proposal.reason,
    confidence: proposal.confidence
  }));
}

function turnContextText(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), TURN_CONTEXT_TEXT_MAX);
}

function contextText(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), STEP_CONTEXT_TEXT_MAX);
}

export function buildTurnStepCandidates(
  rawTurn: RawTurnRecord,
  turnIndex: number
): ExecutionStepCandidateV1[] {
  const candidates: ExecutionStepCandidateV1[] = [];
  for (const [toolCallIndex, rawCall] of rawTurn.toolCalls.entries()) {
    const enriched = enrichToolCall(rawCall, toolCallIndex, rawTurn.toolResults);
    const event = buildSpanTrajectory([enriched])[0]!;
    const sourceRefs = toolSourceRefs(rawTurn.id, toolCallIndex, rawTurn.toolResults);
    candidates.push({
      id: stepCandidateId(rawTurn.id, "tool", toolCallIndex),
      rawTurnId: rawTurn.id,
      turnIndex,
      eventIndex: candidates.length,
      kind: "tool_action",
      action: event.action,
      toolName: event.tool,
      toolCallIndex,
      heuristicSuccess: event.success,
      sourceRefs,
      ...(event.evidence
        ? { evidence: clip(redactSensitiveText(event.evidence), EVENT_EVIDENCE_MAX) }
        : {}),
      toolCalls: 1,
      errorCount: event.success ? 0 : 1
    });
  }
  if (rawTurn.assistantText?.trim()) {
    candidates.push({
      id: stepCandidateId(rawTurn.id, "response", candidates.length),
      rawTurnId: rawTurn.id,
      turnIndex,
      eventIndex: candidates.length,
      kind: "response_generation",
      action: "respond",
      heuristicSuccess: rawTurn.status !== "failed",
      sourceRefs: [assistantSourceId(rawTurn.id)],
      evidence: preview(rawTurn.assistantText),
      toolCalls: 0,
      errorCount: rawTurn.status === "failed" ? 1 : 0
    });
  }
  return candidates;
}

function materializeExecutionSteps(input: {
  episodeId: string;
  frames: readonly TurnFrame[];
  semantics: { observations: readonly ObservationSemantics[]; steps: readonly StepSemantics[] };
  sourceSnapshotHash: string;
  model?: string;
}): { states: ObservedStateV1[]; steps: ExecutionStepV1[] } {
  const observationBySource = new Map(
    input.semantics.observations.map((observation) => [observation.sourceId, observation.operations])
  );
  const semanticsByCandidate = new Map(
    input.semantics.steps.map((semantics) => [semantics.candidateId, semantics])
  );
  const includedCandidates = input.frames
    .flatMap((frame) => frame.candidates)
    .filter((candidate) => semanticsByCandidate.get(candidate.id)?.include);
  const stepIdByCandidate = new Map(includedCandidates.map((candidate) => [
    candidate.id,
    `step_${stableHash({
      schemaVersion: EXECUTION_STEP_SCHEMA_VERSION,
      episodeId: input.episodeId,
      candidateId: candidate.id,
      sourceSnapshotHash: input.sourceSnapshotHash
    }).slice(0, 20)}`
  ]));
  const states: ObservedStateV1[] = [];
  const steps: ExecutionStepV1[] = [];
  let state = emptyObservedState();
  states.push(state);
  let pendingObservationOps: StateDeltaOperation[] = [];

  const applyPendingObservation = (): void => {
    if (pendingObservationOps.length === 0) return;
    const observedState = applyStateDelta(state, pendingObservationOps);
    states.push(observedState);
    const previous = steps.at(-1);
    if (previous) {
      previous.externalObservationDelta = [...previous.externalObservationDelta, ...pendingObservationOps];
      previous.postStateId = observedState.id;
    }
    state = observedState;
    pendingObservationOps = [];
  };

  for (const frame of input.frames) {
    pendingObservationOps.push(...(observationBySource.get(frame.userSourceId) ?? []));
    for (const candidate of frame.candidates) {
      const semantics = semanticsByCandidate.get(candidate.id);
      if (!semantics?.include) continue;
      applyPendingObservation();
      const preState = state;
      const actionPostState = applyStateDelta(preState, semantics.operations);
      states.push(actionPostState);
      const stepId = stepIdByCandidate.get(candidate.id)!;
      const retryOfStepId = semantics.retryOfCandidateId
        ? stepIdByCandidate.get(semantics.retryOfCandidateId)
        : undefined;
      const recoveryFromStepId = semantics.recoveryFromCandidateId
        ? stepIdByCandidate.get(semantics.recoveryFromCandidateId)
        : undefined;
      const outcomeStatus = candidate.heuristicSuccess
        ? semantics.outcomeStatus
        : "failure";
      steps.push({
        id: stepId,
        schemaVersion: EXECUTION_STEP_SCHEMA_VERSION,
        episodeId: input.episodeId,
        rawTurnId: candidate.rawTurnId,
        turnIndex: candidate.turnIndex,
        stepIndex: steps.length,
        preStateId: preState.id,
        action: {
          kind: candidate.kind,
          type: candidate.action,
          intent: semantics.intent,
          summary: semantics.summary,
          eventRefs: [...candidate.sourceRefs],
          ...(candidate.toolName ? { toolName: candidate.toolName } : {}),
          ...(candidate.toolCallIndex === undefined ? {} : { toolCallIndex: candidate.toolCallIndex })
        },
        actionEffectDelta: semantics.operations,
        actionPostStateId: actionPostState.id,
        externalObservationDelta: [],
        postStateId: actionPostState.id,
        outcome: {
          status: outcomeStatus,
          evidenceRefs: semantics.evidenceRefs
        },
        ...(retryOfStepId ? { retryOfStepId } : {}),
        ...(recoveryFromStepId ? { recoveryFromStepId } : {}),
        cost: {
          toolCalls: candidate.toolCalls,
          errorCount: outcomeStatus === "failure" ? Math.max(1, candidate.errorCount) : candidate.errorCount
        },
        provenance: {
          algorithmVersion: EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION,
          ...(input.model ? { model: input.model } : {}),
          sourceSnapshotHash: input.sourceSnapshotHash
        }
      });
      state = actionPostState;
    }
  }
  applyPendingObservation();
  return { states, steps };
}

function compileProceduralSpans(input: {
  episodeId: string;
  steps: readonly ExecutionStepV1[];
  decisions: readonly SpanSegmentationDecisionV1[];
  sourceSnapshotHash: string;
  model?: string;
}): ProceduralSpanV1[] {
  const stepById = new Map(input.steps.map((step) => [step.id, step]));
  return input.decisions.map((decision): ProceduralSpanV1 => {
    const steps = decision.stepIds.map((stepId) => stepById.get(stepId)!);
    const first = steps[0]!;
    const last = steps.at(-1)!;
    const tokenValues = steps.map((step) => step.cost.tokens).filter(isNumber);
    const latencyValues = steps.map((step) => step.cost.latencyMs).filter(isNumber);
    const spanBasis = {
      schemaVersion: PROCEDURAL_SPAN_SCHEMA_VERSION,
      episodeId: input.episodeId,
      spanIndex: decision.spanIndex,
      stepIds: decision.stepIds,
      preStateId: first.preStateId,
      postStateId: last.postStateId,
      sourceSnapshotHash: input.sourceSnapshotHash
    };
    return {
      id: `procedural_span_${stableHash(spanBasis).slice(0, 20)}`,
      schemaVersion: PROCEDURAL_SPAN_SCHEMA_VERSION,
      episodeId: input.episodeId,
      spanIndex: decision.spanIndex,
      localGoal: decision.localGoal,
      entryCondition: decision.entryCondition,
      stepIds: [...decision.stepIds],
      rawTurnIds: [...new Set(steps.map((step) => step.rawTurnId))],
      preStateId: first.preStateId,
      postStateId: last.postStateId,
      termination: {
        status: decision.terminationStatus,
        exitCondition: decision.exitCondition,
        evidenceRefs: [...decision.evidenceRefs]
      },
      cost: {
        steps: steps.length,
        toolCalls: sum(steps.map((step) => step.cost.toolCalls)),
        retryCount: steps.filter((step) => Boolean(step.retryOfStepId)).length,
        recoveryCount: steps.filter((step) => Boolean(step.recoveryFromStepId)).length,
        errorCount: sum(steps.map((step) => step.cost.errorCount)),
        ...(tokenValues.length > 0 ? { tokens: sum(tokenValues) } : {}),
        ...(latencyValues.length > 0 ? { latencyMs: sum(latencyValues) } : {})
      },
      segmentation: {
        reason: decision.reason,
        confidence: decision.confidence
      },
      provenance: {
        algorithmVersion: EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION,
        ...(input.model ? { model: input.model } : {}),
        sourceSnapshotHash: input.sourceSnapshotHash
      }
    };
  });
}

function parseObservationSemantics(
  value: unknown,
  frames: readonly TurnFrame[],
  allowedRefs: ReadonlySet<string>
): ObservationSemantics[] {
  if (!Array.isArray(value)) throw new Error("procedural step LLM returned invalid observations");
  const known = new Set(frames.map((frame) => frame.userSourceId));
  const parsed = new Map<string, ObservationSemantics>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.source_id !== "string" || !known.has(item.source_id)) {
      throw new Error("procedural step LLM returned unknown observation source");
    }
    if (parsed.has(item.source_id)) throw new Error("procedural step LLM duplicated observation source");
    parsed.set(item.source_id, {
      sourceId: item.source_id,
      operations: parseStateDeltaOperations(item.operations, allowedRefs)
    });
  }
  return frames.map((frame) => parsed.get(frame.userSourceId) ?? {
    sourceId: frame.userSourceId,
    operations: []
  });
}

function parseStepSemantics(
  value: unknown,
  candidates: readonly ExecutionStepCandidateV1[],
  referenceCandidates: readonly ExecutionStepCandidateV1[],
  allowedRefs: ReadonlySet<string>
): StepSemantics[] {
  if (!Array.isArray(value)) throw new Error("procedural step LLM returned invalid steps");
  const candidateById = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const candidateIndex = new Map(referenceCandidates.map((candidate, index) => [candidate.id, index]));
  const parsed = new Map<string, StepSemantics>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.candidate_id !== "string") {
      throw new Error("procedural step LLM returned invalid candidate ID");
    }
    const candidate = candidateById.get(item.candidate_id);
    if (!candidate) throw new Error(`procedural step LLM invented candidate: ${item.candidate_id}`);
    if (parsed.has(item.candidate_id)) throw new Error(`procedural step LLM duplicated candidate: ${item.candidate_id}`);
    if (typeof item.include !== "boolean") throw new Error("procedural step LLM omitted include flag");
    if (candidate.kind === "tool_action" && item.include !== true) {
      throw new Error(`procedural step LLM excluded tool action: ${candidate.id}`);
    }
    const intent = cleanText(item.intent, 500);
    const summary = cleanText(item.summary, 1_200);
    if (item.include && (!intent || !summary)) {
      throw new Error(`procedural step LLM returned incomplete semantics: ${candidate.id}`);
    }
    const evidenceRefs = stringArray(item.evidence_refs);
    assertAllowedRefs(evidenceRefs, allowedRefs);
    const retryOfCandidateId = typeof item.retry_of_candidate_id === "string" && item.retry_of_candidate_id.trim()
      ? item.retry_of_candidate_id.trim()
      : undefined;
    const recoveryFromCandidateId = typeof item.recovery_from_candidate_id === "string" && item.recovery_from_candidate_id.trim()
      ? item.recovery_from_candidate_id.trim()
      : undefined;
    if (retryOfCandidateId && recoveryFromCandidateId) {
      throw new Error(`procedural step LLM set both retry and recovery references: ${candidate.id}`);
    }
    for (const referencedCandidateId of [retryOfCandidateId, recoveryFromCandidateId]) {
      if (!referencedCandidateId) continue;
      const retryIndex = candidateIndex.get(referencedCandidateId);
      const currentIndex = candidateIndex.get(candidate.id)!;
      if (retryIndex === undefined || retryIndex >= currentIndex) {
        throw new Error(`procedural step LLM returned invalid retry/recovery reference: ${referencedCandidateId}`);
      }
    }
    parsed.set(candidate.id, {
      candidateId: candidate.id,
      include: item.include,
      intent: intent ?? "non-procedural response",
      summary: summary ?? "No substantive task effect.",
      outcomeStatus: parseExecutionStepOutcome(item.outcome_status),
      evidenceRefs,
      ...(retryOfCandidateId ? { retryOfCandidateId } : {}),
      ...(recoveryFromCandidateId ? { recoveryFromCandidateId } : {}),
      operations: parseStateDeltaOperations(item.operations, allowedRefs)
    });
  }
  for (const candidate of candidates) {
    if (!parsed.has(candidate.id)) throw new Error(`procedural step LLM omitted candidate: ${candidate.id}`);
  }
  return candidates.map((candidate) => parsed.get(candidate.id)!);
}

function parseSpanSegmentationDecisions(
  value: unknown,
  steps: readonly ExecutionStepV1[],
  allowedRefs: ReadonlySet<string>
): SpanSegmentationDecisionV1[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("procedural span LLM returned no spans for a non-empty episode");
  }
  const stepIds = new Set(steps.map((step) => step.id));
  const decisions = value.map((item, spanIndex): SpanSegmentationDecisionV1 => {
    if (!isRecord(item)) throw new Error("procedural span LLM returned invalid span");
    const decisionStepIds = stringArray(item.step_ids);
    if (decisionStepIds.length === 0 || decisionStepIds.some((stepId) => !stepIds.has(stepId))) {
      throw new Error("procedural span LLM returned unknown or empty step IDs");
    }
    const localGoal = cleanText(item.local_goal, 600);
    const entryCondition = cleanText(item.entry_condition, 1_000);
    const exitCondition = cleanText(item.exit_condition, 1_000);
    const reason = cleanText(item.reason, 1_500);
    const terminationStatus = parseTerminationStatus(item.termination_status);
    const evidenceRefs = stringArray(item.evidence_refs);
    assertAllowedRefs(evidenceRefs, allowedRefs);
    if (!localGoal || !entryCondition || !exitCondition || !reason || evidenceRefs.length === 0) {
      throw new Error("procedural span LLM returned incomplete span semantics");
    }
    if (typeof item.confidence !== "number" || !Number.isFinite(item.confidence) ||
      item.confidence < 0 || item.confidence > 1) {
      throw new Error("procedural span LLM returned invalid confidence");
    }
    return {
      spanIndex,
      stepIds: decisionStepIds,
      localGoal,
      entryCondition,
      exitCondition,
      terminationStatus,
      evidenceRefs,
      reason,
      confidence: item.confidence
    };
  });
  const flattened = decisions.flatMap((decision) => decision.stepIds);
  const expected = steps.map((step) => step.id);
  if (flattened.length !== expected.length || flattened.some((stepId, index) => stepId !== expected[index])) {
    throw new Error("procedural span LLM must cover every step exactly once and in order");
  }
  return decisions;
}

function parseProvisionalSpanDecisions(
  value: unknown,
  input: {
    rawTurn: RawTurnRecord;
    windowIndex: number;
    steps: readonly ExecutionStepV1[];
    openSpan?: ProvisionalSpanDecision;
    allowedRefs: ReadonlySet<string>;
  }
): ProvisionalSpanDecision[] {
  const decisions = parseSpanSegmentationDecisions(value, input.steps, input.allowedRefs);
  const rawItems = value as Array<Record<string, unknown>>;
  return decisions.map((decision, index): ProvisionalSpanDecision => {
    const rawItem = rawItems[index]!;
    if (typeof rawItem.continues_previous !== "boolean") {
      throw new Error("procedural span LLM omitted continues_previous");
    }
    if (typeof rawItem.open_at_window_end !== "boolean") {
      throw new Error("procedural span LLM omitted open_at_window_end");
    }
    if (rawItem.continues_previous && (index !== 0 || !input.openSpan)) {
      throw new Error("procedural span LLM can continue only the supplied openSpan from the first returned span");
    }
    if (rawItem.open_at_window_end && index !== decisions.length - 1) {
      throw new Error("procedural span LLM can mark only the final returned span open_at_window_end");
    }
    const id = `provisional_span_${stableHash({
      rawTurnId: input.rawTurn.id,
      windowIndex: input.windowIndex,
      stepIds: decision.stepIds
    }).slice(0, 20)}`;
    return {
      ...decision,
      id,
      rawTurnId: input.rawTurn.id,
      windowIndex: input.windowIndex,
      continuesPrevious: rawItem.continues_previous,
      openAtWindowEnd: rawItem.open_at_window_end
    };
  });
}

function parseReconciledSpanDecisions(
  value: unknown,
  proposals: readonly ProvisionalSpanDecision[],
  allowedRefs: ReadonlySet<string>
): SpanSegmentationDecisionV1[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("procedural span reconciliation returned no spans");
  }
  const proposalById = new Map(proposals.map((proposal) => [proposal.id, proposal]));
  const decisions = value.map((item, spanIndex): SpanSegmentationDecisionV1 & {
    provisionalSpanIds: string[];
  } => {
    if (!isRecord(item)) throw new Error("procedural span reconciliation returned invalid span");
    const provisionalSpanIds = stringArray(item.provisional_span_ids);
    if (provisionalSpanIds.length === 0 || provisionalSpanIds.some((id) => !proposalById.has(id))) {
      throw new Error("procedural span reconciliation returned unknown or empty provisional span IDs");
    }
    const localGoal = cleanText(item.local_goal, 600);
    const entryCondition = cleanText(item.entry_condition, 1_000);
    const exitCondition = cleanText(item.exit_condition, 1_000);
    const reason = cleanText(item.reason, 1_500);
    const terminationStatus = parseTerminationStatus(item.termination_status);
    const evidenceRefs = stringArray(item.evidence_refs);
    assertAllowedRefs(evidenceRefs, allowedRefs);
    if (!localGoal || !entryCondition || !exitCondition || !reason || evidenceRefs.length === 0) {
      throw new Error("procedural span reconciliation returned incomplete span semantics");
    }
    if (typeof item.confidence !== "number" || !Number.isFinite(item.confidence) ||
      item.confidence < 0 || item.confidence > 1) {
      throw new Error("procedural span reconciliation returned invalid confidence");
    }
    return {
      spanIndex,
      provisionalSpanIds,
      stepIds: provisionalSpanIds.flatMap((id) => proposalById.get(id)!.stepIds),
      localGoal,
      entryCondition,
      exitCondition,
      terminationStatus,
      evidenceRefs,
      reason,
      confidence: item.confidence
    };
  });
  const flattenedProposalIds = decisions.flatMap((decision) => decision.provisionalSpanIds);
  const expectedProposalIds = proposals.map((proposal) => proposal.id);
  if (flattenedProposalIds.length !== expectedProposalIds.length ||
    flattenedProposalIds.some((id, index) => id !== expectedProposalIds[index])) {
    throw new Error("procedural span reconciliation must cover every provisional span exactly once and in order");
  }
  return decisions.map(({ provisionalSpanIds: _provisionalSpanIds, ...decision }) => decision);
}

function parseStateDeltaOperations(value: unknown, allowedRefs: ReadonlySet<string>): StateDeltaOperation[] {
  if (!Array.isArray(value)) return [];
  return value.map((item, operationIndex): StateDeltaOperation => {
    if (!isRecord(item)) {
      throw new Error(`procedural step LLM returned invalid state operation at index ${operationIndex}: expected an object, received ${typeof item}`);
    }
    if (!("op" in item) && "operation" in item) {
      throw new Error(`procedural step LLM used forbidden field "operation" at state operation index ${operationIndex}; use the exact field name "op"`);
    }
    if (!isStateDeltaOp(item.op)) {
      const operationName = isRecord(item) ? stableStringify(item.op) : typeof item;
      const keys = Object.keys(item).sort().join(",");
      throw new Error(`procedural step LLM returned invalid state operation at index ${operationIndex}: op=${operationName}, keys=${keys}`);
    }
    const unexpectedKeys = Object.keys(item).filter((key) => !STATE_OPERATION_OUTPUT_KEYS.has(key));
    if (unexpectedKeys.length > 0) {
      throw new Error(`procedural step LLM returned unexpected state operation fields at index ${operationIndex}: ${unexpectedKeys.sort().join(",")}`);
    }
    const subject = cleanText(item.subject, 300);
    const sourceRefs = stringArray(item.source_refs);
    if (!subject || sourceRefs.length === 0) {
      throw new Error(`procedural state operation at index ${operationIndex} (${item.op}) requires subject and source_refs`);
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
      const status = operation.status ?? (typeof operation.value === "string" ? operation.value : undefined);
      if (status !== "active" && status !== "blocked" && status !== "completed" && status !== "failed") {
        throw new Error("procedural status.set requires active, blocked, completed, or failed");
      }
    }
    return operation;
  });
}

function candidateSourceRefs(frames: readonly TurnFrame[]): Set<string> {
  return new Set(frames.flatMap((frame) => [
    frame.userSourceId,
    assistantSourceId(frame.rawTurn.id),
    ...frame.candidates.flatMap((candidate) => candidate.sourceRefs)
  ]));
}

function enrichToolCall(
  rawCall: unknown,
  index: number,
  toolResults: readonly unknown[]
): unknown {
  if (!isRecord(rawCall) || typeof rawCall.name !== "string") return rawCall;
  const result = matchingToolResult(rawCall, index, toolResults);
  if (!result) return rawCall;
  return {
    ...rawCall,
    ...(rawCall.output === undefined && result.output !== undefined ? { output: result.output } : {}),
    ...(rawCall.error === undefined && typeof result.error === "string" ? { error: result.error } : {}),
    ...(rawCall.errorCode === undefined && typeof result.errorCode === "string"
      ? { errorCode: result.errorCode }
      : {}),
    ...(rawCall.success === undefined && typeof result.success === "boolean"
      ? { success: result.success }
      : {})
  };
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
  if (!isRecord(positional)) return undefined;
  const resultName = text(positional.name) ?? text(positional.toolName);
  return !name || !resultName || name === resultName ? positional : undefined;
}

function toolSourceRefs(rawTurnId: string, index: number, toolResults: readonly unknown[]): string[] {
  return [
    `turn:${rawTurnId}:tool:${index}`,
    ...(toolResults[index] === undefined ? [] : [`turn:${rawTurnId}:tool_result:${index}`])
  ];
}

function stepCandidateId(rawTurnId: string, kind: "tool" | "response", index: number): string {
  return `candidate_${stableHash({ rawTurnId, kind, index }).slice(0, 20)}`;
}

function userSourceId(rawTurnId: string): string {
  return `turn:${rawTurnId}:user`;
}

function assistantSourceId(rawTurnId: string): string {
  return `turn:${rawTurnId}:assistant`;
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
  return [...new Set(value.filter((item): item is string => typeof item === "string" && Boolean(item.trim())))]
    .map((item) => item.trim());
}

function assertAllowedRefs(refs: readonly string[], allowedRefs: ReadonlySet<string>): void {
  for (const ref of refs) {
    if (!allowedRefs.has(ref)) throw new Error(`procedural reconstruction invented source ref: ${ref}`);
  }
}

function parseExecutionStepOutcome(value: unknown): ExecutionStepOutcome {
  return value === "success" || value === "failure" || value === "partial" || value === "unknown"
    ? value
    : "unknown";
}

function parseTerminationStatus(value: unknown): ProceduralSpanTermination {
  if (value === "success" || value === "failure" || value === "blocked" || value === "abandoned") {
    return value;
  }
  throw new Error("procedural span LLM returned invalid termination status");
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

function isNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
