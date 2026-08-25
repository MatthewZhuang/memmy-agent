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
  type StateDeltaOperation
} from "./span-v3-model.js";
import {
  buildEpisodeProceduralPath,
  EXECUTION_STEP_SCHEMA_VERSION,
  PROCEDURAL_SPAN_SCHEMA_VERSION,
  type EpisodeProceduralPathV2,
  type ExecutionStepOutcome,
  type ExecutionStepV1,
  type ProceduralSpanV1,
  type SpanSegmentationDecisionV1
} from "./procedural-path-model.js";
import { buildSpanTrajectory, type SpanTrajectoryAction } from "./span-trajectory.js";

export const EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION = "episode-procedural-reconstruction.v7" as const;

const TASK_CONTRACT_OPERATION = "procedural.task_contract.v1";
const STEP_SEMANTICS_OPERATION = "procedural.step_semantics.v2";
const MAX_SEMANTIC_REPAIR_ATTEMPTS = 2;
const STEP_WINDOW_MAX_CANDIDATES = 30;
const STEP_WINDOW_INPUT_CHAR_BUDGET = 30_000;
const STEP_WINDOW_OVERLAP = 5;
const PREVIOUS_TURN_CONTEXT_COUNT = 3;
const TURN_CONTEXT_TEXT_MAX = 700;
const STEP_CONTEXT_TEXT_MAX = 500;
const TEXT_PREVIEW_MAX = 1_500;
const EVENT_EVIDENCE_MAX = 900;

export const TASK_CONTRACT_PROMPT = `You extract the user-authored task contract for one complete agent Episode.

Each observation is an immutable user message. Read its complete text and extract only information explicitly introduced by that message:
- goal: the active task objective, or null when the message does not establish or change it;
- constraints: requirements that restrict how the task may be performed;
- acceptance_criteria: observable conditions that determine whether the requested result is acceptable.

Do not summarize tool execution, infer hidden requirements, or emit state operations. Every supplied source_id must appear exactly once. Keep concrete paths, formats, quantities, and named deliverables when they matter.

Return JSON only:
{
  "contracts": [
    {
      "source_id": "turn:...:user",
      "goal": "restore the failing build",
      "constraints": ["keep NodeNext module resolution"],
      "acceptance_criteria": ["the target build and tests pass"]
    }
  ]
}`;

export const EXECUTION_STEP_SEMANTICS_PROMPT = `You reconstruct evidence-grounded execution steps from an agent episode.

Each supplied step candidate is one observable action and its immediate result. Compress it into:
- intent: what the action attempted;
- summary: what the supplied result actually established.

Do not extract task state, observations, state operations, retries, recoveries, outcome labels, or evidence references. Those fields are derived elsewhere from immutable inputs. Do not group candidates into subtasks in this pass.

Tool-action candidates are procedural and must include=true. A response-generation candidate may include=false only when it is purely social or reports no substantive task effect. Only objects inside stepCandidates are output candidates; precedingStepContext is read-only continuity context and must never be returned. Every stepCandidates item must appear exactly once.

Return JSON only:
{
  "steps": [
    {
      "candidate_id": "candidate_...",
      "include": true,
      "intent": "inspect the failing dependency resolution",
      "summary": "Read the lockfile and observed incompatible versions."
    }
  ]
}`;

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

interface TaskContractSemantics {
  sourceId: string;
  goal?: string;
  constraints: string[];
  acceptanceCriteria: string[];
}

interface StepSemantics {
  candidateId: string;
  include: boolean;
  intent: string;
  summary: string;
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
    const taskContracts = await this.extractTaskContracts(input.episodeId, frames);
    const semantics = await this.extractStepSemantics(input.episodeId, frames);
    const skeleton = materializeExecutionSteps({
      episodeId: input.episodeId,
      frames,
      taskContracts,
      semantics,
      sourceSnapshotHash,
      model: this.deps.llm.config.model
    });
    // The Step Sequence learner is the only production consumer. Keep one
    // deterministic envelope solely so episode-execution-path.v2 rows remain
    // readable by installations that already persisted the legacy Span fields.
    // No Span segmentation, reconciliation, capability, state, or credit LLM is
    // called on the production path.
    const decisions = compatibilitySpanDecisions(
      skeleton.steps,
      taskContracts,
      input.terminalReward
    );
    const spans = compileProceduralSpans({
      episodeId: input.episodeId,
      steps: skeleton.steps,
      decisions,
      sourceSnapshotHash,
      model: this.deps.llm.config.model
    });
    return buildEpisodeProceduralPath({
      episodeId: input.episodeId,
      states: skeleton.states,
      steps: skeleton.steps,
      spans,
      segmentationDecisions: decisions,
      sourceSnapshotHash,
      ...(input.terminalReward === undefined ? {} : { terminalReward: input.terminalReward })
    });
  }

  private async extractTaskContracts(
    episodeId: string,
    frames: readonly TurnFrame[]
  ): Promise<TaskContractSemantics[]> {
    if (frames.length === 0) return [];
    type TaskContractResult = { contracts?: unknown };
    const messages: LlmMessage[] = [
      { role: "system", content: TASK_CONTRACT_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId,
          observations: frames.map((frame) => ({
            sourceId: frame.userSourceId,
            rawTurnId: frame.rawTurn.id,
            userText: fullTaskText(frame.rawTurn.userText)
          }))
        })
      }
    ];
    const options = {
      operation: TASK_CONTRACT_OPERATION,
      thinkingMode: "disabled" as const,
      temperature: 0,
      maxTokens: Math.max(1_600, Math.min(6_000, frames.length * 600))
    };
    let result = await this.deps.llm.completeJson<TaskContractResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseTaskContracts(result.contracts, frames);
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<TaskContractResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          {
            role: "user",
            content: contractRepairInstruction(error, repairNumber)
          }
        ], {
          ...options,
          operation: `${TASK_CONTRACT_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }

  private async extractStepSemantics(
    episodeId: string,
    frames: readonly TurnFrame[]
  ): Promise<StepSemantics[]> {
    const allCandidates = frames.flatMap((frame) => frame.candidates);
    if (allCandidates.length === 0) return [];
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
          precedingCandidates,
          candidates,
          semanticsByCandidate
        });
        steps.push(...parsed);
        for (const semantics of parsed) semanticsByCandidate.set(semantics.candidateId, semantics);
        processedCandidates.push(...candidates);
      }
    }
    return steps;
  }

  private async extractStepSemanticsWindow(input: {
    episodeId: string;
    frame: TurnFrame;
    previousTurns: readonly TurnFrame[];
    windowIndex: number;
    precedingCandidates: readonly ExecutionStepCandidateV1[];
    candidates: readonly ExecutionStepCandidateV1[];
    semanticsByCandidate: ReadonlyMap<string, StepSemantics>;
  }): Promise<StepSemantics[]> {
    type StepSemanticResult = { steps?: unknown };
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
              action: candidate.action,
              toolName: candidate.toolName,
              heuristicSuccess: candidate.heuristicSuccess,
              intent: semantics?.intent,
              summary: contextText(semantics?.summary)
            };
          }),
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
      maxTokens: Math.max(1_600, Math.min(5_000, input.candidates.length * 160))
    };
    let result = await this.deps.llm.completeJson<StepSemanticResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseStepSemantics(result.steps, input.candidates);
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<StepSemanticResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          {
            role: "user",
            content: stepSemanticRepairInstruction(error, repairNumber, input.candidates)
          }
        ], {
          ...options,
          operation: `${STEP_SEMANTICS_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }

}

function stepSemanticRepairInstruction(
  error: unknown,
  repairNumber: number,
  candidates: readonly ExecutionStepCandidateV1[]
): string {
  const message = error instanceof Error ? error.message : String(error);
  const allowedCandidateIds = candidates.map((candidate) => candidate.id);
  const common = `Deterministic validation error: ${message}\nThe complete and exclusive allowed candidate_id list is: ${stableStringify(allowedCandidateIds)}. Return each ID from this list exactly once and return no other ID. precedingStepContext is read-only and is not output. Return the complete corrected JSON object only. Each item must contain only candidate_id, include, intent, and summary. Do not add state, outcome, retry, recovery, or evidence fields. Do not add markdown, explanation, comments, or extra top-level fields.`;
  if (repairNumber === 1) {
    return `This is a strict schema-repair task, not a request to reinterpret the episode. Correct the previous JSON.\n${common}`;
  }
  return `The previous repair still failed. Discard the previous JSON and regenerate the entire response from the original episode payload and schema. Do not patch or paraphrase the malformed object.\n${common}\nBefore returning, verify that the complete response is one JSON object and every supplied candidate appears exactly once.`;
}

function contractRepairInstruction(error: unknown, repairNumber: number): string {
  const message = error instanceof Error ? error.message : String(error);
  const prefix = repairNumber === 1
    ? "Correct the previous task-contract JSON."
    : "The previous repair still failed. Discard it and regenerate the complete task-contract JSON.";
  return `${prefix}\nDeterministic validation error: ${message}\nReturn every supplied source_id exactly once with goal, constraints, and acceptance_criteria. Return JSON only.`;
}

/**
 * The Step-sequence learner does not use semantic Span boundaries. A single
 * deterministic compatibility Span keeps the versioned Path readable by
 * older diagnostics without paying for segmentation, reconciliation,
 * capability compilation, or Span-state LLM calls.
 */
function compatibilitySpanDecisions(
  steps: readonly ExecutionStepV1[],
  taskContracts: readonly TaskContractSemantics[],
  terminalReward: number | undefined
): SpanSegmentationDecisionV1[] {
  if (steps.length === 0) return [];
  const first = steps[0]!;
  const last = steps.at(-1)!;
  const explicitGoal = taskContracts.find((contract) => contract.goal)?.goal;
  const localGoal = (contextText(explicitGoal ?? first.action.intent) ?? "Execute the observed procedure")
    .slice(0, 600);
  const procedureSemantic = steps
    .map((step) => contextText(step.action.intent))
    .filter(Boolean)
    .join(" -> ")
    .slice(0, 2_000);
  const terminationStatus = terminalReward !== undefined && terminalReward < 0
    ? "failure" as const
    : last.outcome.status === "failure"
      ? "failure" as const
      : "success" as const;
  return [{
    spanIndex: 0,
    stepIds: steps.map((step) => step.id),
    localGoal,
    capabilityGoal: localGoal,
    procedureSemantic: procedureSemantic ||
      (contextText(last.action.summary) ?? "Complete the observed execution").slice(0, 2_000),
    entryCondition: `Episode execution starts before Step ${first.stepIndex}`,
    exitCondition: (contextText(last.action.summary) ?? "Episode execution completed").slice(0, 1_000),
    terminationStatus,
    evidenceRefs: unique(steps.flatMap((step) => [step.id, ...step.outcome.evidenceRefs])).slice(0, 64),
    reason: "Deterministic compatibility envelope for Step-sequence learning; not a learned Span boundary.",
    confidence: 1
  }];
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
  taskContracts: readonly TaskContractSemantics[];
  semantics: readonly StepSemantics[];
  sourceSnapshotHash: string;
  model?: string;
}): { states: ObservedStateV1[]; steps: ExecutionStepV1[] } {
  const taskContractBySource = new Map(
    input.taskContracts.map((contract) => [contract.sourceId, contract])
  );
  const semanticsByCandidate = new Map(
    input.semantics.map((semantics) => [semantics.candidateId, semantics])
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
  let previousIncludedCandidate: ExecutionStepCandidateV1 | undefined;

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
    const taskContract = taskContractBySource.get(frame.userSourceId);
    if (taskContract) {
      pendingObservationOps.push(...taskContractOperations(taskContract, state));
    }
    for (const candidate of frame.candidates) {
      const semantics = semanticsByCandidate.get(candidate.id);
      if (!semantics?.include) continue;
      applyPendingObservation();
      const preState = state;
      const stepId = stepIdByCandidate.get(candidate.id)!;
      const actionEffectDelta: StateDeltaOperation[] = [];
      const actionPostState = applyStateDelta(preState, actionEffectDelta);
      states.push(actionPostState);
      const previousStep = steps.at(-1);
      const relationship = inferFailureRelationship({
        candidate,
        previousCandidate: previousIncludedCandidate,
        previousStep
      });
      const outcomeStatus: ExecutionStepOutcome = candidate.heuristicSuccess ? "success" : "failure";
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
        actionEffectDelta,
        actionPostStateId: actionPostState.id,
        externalObservationDelta: [],
        postStateId: actionPostState.id,
        outcome: {
          status: outcomeStatus,
          evidenceRefs: [...candidate.sourceRefs]
        },
        ...relationship,
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
      previousIncludedCandidate = candidate;
    }
  }
  applyPendingObservation();
  return { states, steps };
}

function taskContractOperations(
  contract: TaskContractSemantics,
  state: ObservedStateV1
): StateDeltaOperation[] {
  const operations: StateDeltaOperation[] = [];
  if (contract.goal) {
    operations.push({
      op: state.goal ? "goal.refine" : "goal.set",
      subject: "task_goal",
      value: contract.goal,
      status: "in_progress",
      sourceRefs: [contract.sourceId]
    });
  }
  for (const constraint of contract.constraints) {
    operations.push({
      op: "constraint.upsert",
      subject: clip(constraint, 300),
      value: constraint,
      sourceRefs: [contract.sourceId]
    });
  }
  for (const criterion of contract.acceptanceCriteria) {
    operations.push({
      op: "verification.set",
      subject: clip(criterion, 300),
      value: criterion,
      status: "pending",
      sourceRefs: [contract.sourceId]
    });
  }
  return operations;
}

function inferFailureRelationship(input: {
  candidate: ExecutionStepCandidateV1;
  previousCandidate?: ExecutionStepCandidateV1;
  previousStep?: ExecutionStepV1;
}): Pick<ExecutionStepV1, "retryOfStepId" | "recoveryFromStepId"> {
  if (!input.previousCandidate || input.previousStep?.outcome.status !== "failure") return {};
  const repeatsSameAction = input.candidate.action === input.previousCandidate.action &&
    input.candidate.toolName === input.previousCandidate.toolName;
  return repeatsSameAction
    ? { retryOfStepId: input.previousStep.id }
    : { recoveryFromStepId: input.previousStep.id };
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
      capabilityGoal: decision.capabilityGoal ?? decision.localGoal,
      ...(decision.procedureSemantic
        ? { procedureSemantic: decision.procedureSemantic }
        : {}),
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

function parseTaskContracts(
  value: unknown,
  frames: readonly TurnFrame[]
): TaskContractSemantics[] {
  if (!Array.isArray(value)) throw new Error("procedural task contract LLM returned invalid contracts");
  const known = new Set(frames.map((frame) => frame.userSourceId));
  const parsed = new Map<string, TaskContractSemantics>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.source_id !== "string" || !known.has(item.source_id)) {
      throw new Error("procedural task contract LLM returned unknown source");
    }
    if (parsed.has(item.source_id)) throw new Error("procedural task contract LLM duplicated source");
    const goal = cleanText(item.goal, 4_000);
    parsed.set(item.source_id, {
      sourceId: item.source_id,
      ...(goal ? { goal } : {}),
      constraints: cleanStringArray(item.constraints, 2_000),
      acceptanceCriteria: cleanStringArray(item.acceptance_criteria, 2_000)
    });
  }
  return frames.map((frame, index) => {
    const item = parsed.get(frame.userSourceId);
    if (item) return item;
    const fallbackGoal = index === 0 ? cleanText(frame.rawTurn.userText, 4_000) : undefined;
    return {
      sourceId: frame.userSourceId,
      ...(fallbackGoal ? { goal: fallbackGoal } : {}),
      constraints: [],
      acceptanceCriteria: []
    };
  });
}

function parseStepSemantics(
  value: unknown,
  candidates: readonly ExecutionStepCandidateV1[]
): StepSemantics[] {
  if (!Array.isArray(value)) throw new Error("procedural step LLM returned invalid steps");
  const candidateById = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const parsed = new Map<string, StepSemantics>();
  for (const item of value) {
    if (!isRecord(item)) {
      throw new Error("procedural step LLM returned invalid candidate ID");
    }
    const snakeCaseCandidateId = typeof item.candidate_id === "string" ? item.candidate_id : undefined;
    const camelCaseCandidateId = typeof item.candidateId === "string" ? item.candidateId : undefined;
    if (snakeCaseCandidateId && camelCaseCandidateId && snakeCaseCandidateId !== camelCaseCandidateId) {
      throw new Error("procedural step LLM returned conflicting candidate IDs");
    }
    const candidateId = snakeCaseCandidateId ?? camelCaseCandidateId;
    if (!candidateId) throw new Error("procedural step LLM returned invalid candidate ID");
    const candidate = candidateById.get(candidateId);
    if (!candidate) throw new Error(`procedural step LLM invented candidate: ${candidateId}`);
    if (parsed.has(candidateId)) throw new Error(`procedural step LLM duplicated candidate: ${candidateId}`);
    if (typeof item.include !== "boolean") throw new Error("procedural step LLM omitted include flag");
    if (candidate.kind === "tool_action" && item.include !== true) {
      throw new Error(`procedural step LLM excluded tool action: ${candidate.id}`);
    }
    const intent = cleanText(item.intent, 500);
    const summary = cleanText(item.summary, 1_200);
    if (item.include && (!intent || !summary)) {
      throw new Error(`procedural step LLM returned incomplete semantics: ${candidate.id}`);
    }
    parsed.set(candidateId, {
      candidateId: candidate.id,
      include: item.include,
      intent: intent ?? "non-procedural response",
      summary: summary ?? "No substantive task effect."
    });
  }
  for (const candidate of candidates) {
    if (!parsed.has(candidate.id)) throw new Error(`procedural step LLM omitted candidate: ${candidate.id}`);
  }
  return candidates.map((candidate) => parsed.get(candidate.id)!);
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

function fullTaskText(value: string | undefined): string | undefined {
  if (!value?.trim()) return undefined;
  return redactSensitiveText(value.trim());
}

function cleanText(value: unknown, maxChars: number): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), maxChars);
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function cleanStringArray(value: unknown, maxChars: number): string[] {
  if (!Array.isArray(value)) return [];
  return unique(value.flatMap((item) => {
    const cleaned = cleanText(item, maxChars);
    return cleaned ? [cleaned] : [];
  }));
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function isNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
