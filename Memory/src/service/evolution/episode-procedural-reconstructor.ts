import type { LlmClient, LlmMessage } from "../../model/types.js";
import { createMemoryLogger } from "../../logging/logger.js";
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
  type ProceduralSpanTermination,
  type ProceduralSpanV1,
  type SpanSegmentationDecisionV1
} from "./procedural-path-model.js";
import { buildSpanTrajectory, type SpanTrajectoryAction } from "./span-trajectory.js";

export const EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION = "episode-procedural-reconstruction.v6" as const;

const proceduralReconstructorLogger = createMemoryLogger("procedural-reconstructor");

const TASK_CONTRACT_OPERATION = "procedural.task_contract.v1";
const STEP_SEMANTICS_OPERATION = "procedural.step_semantics.v2";
const SPAN_SEGMENTATION_OPERATION = "procedural.span_segmentation.v3.window";
const SPAN_RECONCILIATION_OPERATION = "procedural.span_reconciliation.v2";
const SPAN_CAPABILITY_OPERATION = "procedural.span_capability.v1";
const SPAN_STATE_OPERATION = "procedural.span_state.v1";
const MAX_SEMANTIC_REPAIR_ATTEMPTS = 2;
const STEP_WINDOW_MAX_CANDIDATES = 30;
const STEP_WINDOW_INPUT_CHAR_BUDGET = 30_000;
const STEP_WINDOW_OVERLAP = 5;
const PREVIOUS_TURN_CONTEXT_COUNT = 3;
const TURN_CONTEXT_TEXT_MAX = 700;
const STEP_CONTEXT_TEXT_MAX = 500;
const RECONCILIATION_PROCEDURE_INTENT_MAX = 10;
const MAX_STATE_COMPILATION_STEPS = 20;
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

export const PROCEDURAL_SPAN_STATE_PROMPT = `You compile evidence-grounded state changes for already segmented procedural spans.

The Step pass has already compressed individual actions. Work at Span granularity: one output item describes the durable result of one complete local-subproblem lifecycle. Do not restate every Step and do not emit generic state-operation objects.

Fields:
- effects: durable facts or state changes established by the Span;
- artifacts: files, documents, datasets, configurations, or other deliverables created, updated, or verified. Each artifact status must be exactly created, updated, or verified. Do not emit an artifact merely because it was read or inspected;
- issues_opened: unresolved problems newly established by evidence;
- issues_resolved: problems demonstrably resolved by the Span;
- verification: observable checks and their result.

Use only supplied evidence_refs. Empty arrays are valid. Every span_index must appear exactly once.

Return JSON only:
{
  "spans": [
    {
      "span_index": 0,
      "effects": [{"summary": "the incompatible dependency was corrected", "evidence_refs": ["step_..."]}],
      "artifacts": [{"name": "package.json", "status": "updated", "evidence_refs": ["step_..."]}],
      "issues_opened": [],
      "issues_resolved": [{"issue": "dependency incompatibility", "evidence_refs": ["step_..."]}],
      "verification": [{"criterion": "target tests pass", "status": "passed", "evidence_refs": ["step_..."]}]
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
- Ground both the target and outcome in the supplied task contract, step intents, summaries, outcomes, and evidence.
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
- Recompute final local_goal from the task contract, procedureIntents, entry/exit conditions, and evidence. Treat every provisional localGoal as a proposal, not authoritative truth.
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

export const PROCEDURAL_SPAN_CAPABILITY_PROMPT = `You normalize already-final procedural spans into two independent reusable views.

Span boundaries and concrete local_goal values are final and must not be changed.

Definitions:
- capability_goal answers WHAT reusable outcome is produced. Express one canonical operation over a functional input/output class.
- procedure_semantic answers HOW the outcome is reached. Express an ordered strategy of 2-6 canonical verb phrases separated by " -> ".

Shared abstraction rules:
- Remove concrete subject matter, named entities, dates, quantities, filenames, paths, brands, organizations, locations, and tool or vendor names.
- Normalize equivalent operations to stable vocabulary across unrelated Episodes.
- File containers are not capability identity unless they change the functional artifact class: normalize Word/Markdown/plain text to document, but retain spreadsheet, presentation, dataset, code change, and configuration when materially relevant.
- Do not use vague labels such as "complete the task", "process information", "perform research", or "generate output".
- Distinguish evidence acquisition, artifact synthesis, revision, and verification when they are separate final Spans.

capability_goal rules:
- Keep only the core operation, functional input class, functional output/artifact class, and observable result.
- Exclude method, tool, implementation sequence, retries, debugging, iterative refinement, formatting constraints, quality criteria, and incidental failure/recovery details. Those belong to procedure_semantic or the state contract.
- Prefer a medium-grained canonical clause such as "generate a formatted document from source material", not a task-specific paraphrase.

procedure_semantic rules:
- Preserve the reusable control flow that distinguishes materially different ways to achieve the same goal.
- Normalize concrete calls into canonical phases such as inspect inputs, acquire evidence, synthesize, transform, render, validate, revise, or verify.
- Include recovery/revision only when it is a meaningful part of the demonstrated strategy; omit one-off errors and concrete outcomes.
- Do not repeat the capability_goal, domain nouns, file names, libraries, commands, or vendor/tool names.

Every supplied span_index must appear exactly once. Return JSON only:
{
  "spans": [
    {
      "span_index": 0,
      "capability_goal": "generate a formatted document from source material",
      "procedure_semantic": "inspect requirements and inputs -> synthesize document content -> render the artifact -> validate structure and constraints -> revise until compliant"
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

interface SpanStateEvidenceItem {
  text: string;
  evidenceRefs: string[];
}

interface SpanArtifactStateItem extends SpanStateEvidenceItem {
  status: "created" | "updated" | "verified";
}

interface SpanVerificationStateItem extends SpanStateEvidenceItem {
  status: "passed" | "failed" | "partial" | "unknown";
}

interface SpanStateSemantics {
  spanIndex: number;
  effects: SpanStateEvidenceItem[];
  artifacts: SpanArtifactStateItem[];
  issuesOpened: SpanStateEvidenceItem[];
  issuesResolved: SpanStateEvidenceItem[];
  verification: SpanVerificationStateItem[];
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
  mode?: "span" | "step_sequence";
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
    if (this.deps.mode === "step_sequence") {
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
    const boundaryDecisions = await this.segmentSpans({
      episodeId: input.episodeId,
      rawTurns,
      taskContracts,
      steps: skeleton.steps
    });
    const decisions = await this.compileSpanCapabilities({
      episodeId: input.episodeId,
      steps: skeleton.steps,
      decisions: boundaryDecisions
    });
    const spanStates = await this.compileSpanStates({
      episodeId: input.episodeId,
      taskContracts,
      steps: skeleton.steps,
      decisions
    });
    const materialized = materializeExecutionSteps({
      episodeId: input.episodeId,
      frames,
      taskContracts,
      semantics,
      decisions,
      spanStates,
      sourceSnapshotHash,
      model: this.deps.llm.config.model
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

  private async compileSpanCapabilities(input: {
    episodeId: string;
    steps: readonly ExecutionStepV1[];
    decisions: readonly SpanSegmentationDecisionV1[];
  }): Promise<SpanSegmentationDecisionV1[]> {
    if (input.decisions.length === 0) return [];
    type SpanCapabilityResult = { spans?: unknown };
    const stepById = new Map(input.steps.map((step) => [step.id, step]));
    const messages: LlmMessage[] = [
      { role: "system", content: PROCEDURAL_SPAN_CAPABILITY_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId: input.episodeId,
          spans: input.decisions.map((decision) => ({
            spanIndex: decision.spanIndex,
            localGoal: decision.localGoal,
            entryCondition: decision.entryCondition,
            exitCondition: decision.exitCondition,
            terminationStatus: decision.terminationStatus,
            procedure: selectStateCompilationSteps(
              decision.stepIds.map((stepId) => stepById.get(stepId)!)
            ).map((step) => ({
              intent: contextText(step.action.intent),
              summary: contextText(step.action.summary),
              outcome: step.outcome.status
            }))
          }))
        })
      }
    ];
    const options = {
      operation: SPAN_CAPABILITY_OPERATION,
      thinkingMode: "disabled" as const,
      temperature: 0,
      maxTokens: Math.max(1_600, Math.min(5_000, input.decisions.length * 500))
    };
    let result = await this.deps.llm.completeJson<SpanCapabilityResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        const abstractions = parseSpanCapabilities(result.spans, input.decisions);
        return input.decisions.map((decision) => ({
          ...decision,
          capabilityGoal: abstractions.get(decision.spanIndex)!.capabilityGoal,
          procedureSemantic: abstractions.get(decision.spanIndex)!.procedureSemantic
        }));
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<SpanCapabilityResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: spanCapabilityRepairInstruction(error, repairNumber) }
        ], {
          ...options,
          operation: `${SPAN_CAPABILITY_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }

  private async compileSpanStates(input: {
    episodeId: string;
    taskContracts: readonly TaskContractSemantics[];
    steps: readonly ExecutionStepV1[];
    decisions: readonly SpanSegmentationDecisionV1[];
  }): Promise<SpanStateSemantics[]> {
    if (input.decisions.length === 0) return [];
    type SpanStateResult = { spans?: unknown };
    const stepById = new Map(input.steps.map((step) => [step.id, step]));
    const messages: LlmMessage[] = [
      { role: "system", content: PROCEDURAL_SPAN_STATE_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId: input.episodeId,
          taskContract: compactTaskContracts(input.taskContracts),
          spans: input.decisions.map((decision) => {
            const steps = decision.stepIds.map((stepId) => stepById.get(stepId)!);
            const stateCompilationSteps = selectStateCompilationSteps(steps);
            return {
              spanIndex: decision.spanIndex,
              localGoal: decision.localGoal,
              entryCondition: decision.entryCondition,
              exitCondition: decision.exitCondition,
              terminationStatus: decision.terminationStatus,
              evidenceRefs: unique([
                ...decision.evidenceRefs,
                ...steps.flatMap(stepEvidenceRefs)
              ]),
              stepCount: steps.length,
              steps: stateCompilationSteps.map(compactExecutionStep)
            };
          })
        })
      }
    ];
    const options = {
      operation: SPAN_STATE_OPERATION,
      thinkingMode: "disabled" as const,
      temperature: 0,
      maxTokens: Math.max(2_000, Math.min(8_000, input.decisions.length * 800))
    };
    let result = await this.deps.llm.completeJson<SpanStateResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseSpanStates(result.spans, input.decisions, stepById);
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) {
          const sanitized = parseSpanStates(result.spans, input.decisions, stepById, {
            dropInvalidEvidenceRefs: true
          });
          proceduralReconstructorLogger.warn("span_state.invalid_evidence_refs_dropped", {
            episodeId: input.episodeId,
            repairAttempts: repairAttempt,
            validationError: error instanceof Error ? error.message : String(error)
          });
          return sanitized;
        }
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<SpanStateResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          {
            role: "user",
            content: spanStateRepairInstruction(error, repairNumber)
          }
        ], {
          ...options,
          operation: `${SPAN_STATE_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }

  private async segmentSpans(input: {
    episodeId: string;
    rawTurns: readonly RawTurnRecord[];
    taskContracts: readonly TaskContractSemantics[];
    steps: readonly ExecutionStepV1[];
  }): Promise<SpanSegmentationDecisionV1[]> {
    if (input.steps.length === 0) return [];
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
          taskContracts: input.taskContracts,
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
      taskContracts: input.taskContracts,
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
    taskContracts: readonly TaskContractSemantics[];
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
          taskContract: compactTaskContracts(input.taskContracts),
          previousTurns: input.previousTurns.map(compactRawTurn),
          currentTurn: compactRawTurn(input.rawTurn),
          precedingStepContext: input.precedingSteps.map(compactExecutionStep),
          openSpan: input.openSpan ? compactProvisionalSpan(input.openSpan) : null,
          steps: input.steps.map(compactExecutionStep)
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
    taskContracts: readonly TaskContractSemantics[];
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
          taskContract: compactTaskContracts(input.taskContracts),
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
            procedureIntents: reconciliationProcedureIntents(proposal, stepById)
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

function spanStateRepairInstruction(error: unknown, repairNumber: number): string {
  const message = error instanceof Error ? error.message : String(error);
  const prefix = repairNumber === 1
    ? "Correct the previous Span-state JSON."
    : "The previous repair still failed. Discard it and regenerate the complete Span-state JSON.";
  return `${prefix}\nDeterministic validation error: ${message}\nReturn every supplied span_index exactly once. Use only supplied evidence_refs. Artifact status must be exactly created, updated, or verified; omit merely read or inspected artifacts. Verification status must be exactly passed, failed, partial, or unknown. Return JSON only.`;
}

function spanCapabilityRepairInstruction(error: unknown, repairNumber: number): string {
  const message = error instanceof Error ? error.message : String(error);
  const prefix = repairNumber === 1
    ? "Correct the previous capability-goal JSON without changing any Span boundary."
    : "Discard the previous repair and regenerate the complete capability-goal JSON.";
  return `${prefix}\nDeterministic validation error: ${message}\nReturn every supplied span_index exactly once with only span_index, capability_goal, and procedure_semantic. Keep WHAT and HOW independent. Return JSON only.`;
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

function compactExecutionStep(step: ExecutionStepV1): Record<string, unknown> {
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
    evidenceRefs: stepEvidenceRefs(step)
  };
}

function stepEvidenceRefs(step: ExecutionStepV1): string[] {
  return unique([step.id, ...step.action.eventRefs, ...step.outcome.evidenceRefs]);
}

function compactTaskContracts(
  taskContracts: readonly TaskContractSemantics[]
): Array<Record<string, unknown>> {
  return taskContracts.map((contract) => ({
    sourceId: contract.sourceId,
    goal: contract.goal,
    constraints: contract.constraints,
    acceptanceCriteria: contract.acceptanceCriteria
  }));
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

function selectStateCompilationSteps(steps: readonly ExecutionStepV1[]): ExecutionStepV1[] {
  if (steps.length <= MAX_STATE_COMPILATION_STEPS) return [...steps];
  const selected = new Map<number, ExecutionStepV1>();
  const add = (step: ExecutionStepV1 | undefined): void => {
    if (step) selected.set(step.stepIndex, step);
  };
  add(steps[0]);
  add(steps.at(-1));
  for (const step of steps) {
    if (step.outcome.status === "failure" || step.retryOfStepId || step.recoveryFromStepId) add(step);
  }
  const remaining = Math.max(0, MAX_STATE_COMPILATION_STEPS - selected.size);
  for (let slot = 1; slot <= remaining; slot += 1) {
    const index = Math.round((slot * (steps.length - 1)) / (remaining + 1));
    add(steps[index]);
  }
  return [...selected.values()]
    .sort((left, right) => left.stepIndex - right.stepIndex)
    .slice(0, MAX_STATE_COMPILATION_STEPS);
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
  taskContracts: readonly TaskContractSemantics[];
  semantics: readonly StepSemantics[];
  decisions?: readonly SpanSegmentationDecisionV1[];
  spanStates?: readonly SpanStateSemantics[];
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
  const decisionByFinalStepId = new Map(
    (input.decisions ?? []).map((decision) => [decision.stepIds.at(-1)!, decision])
  );
  const spanStateByIndex = new Map(
    (input.spanStates ?? []).map((spanState) => [spanState.spanIndex, spanState])
  );
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
      const decision = decisionByFinalStepId.get(stepId);
      const actionEffectDelta = decision
        ? spanStateOperations(
            spanStateByIndex.get(decision.spanIndex),
            decision,
            decision.spanIndex === (input.decisions?.length ?? 0) - 1
          )
        : [];
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

function spanStateOperations(
  semantics: SpanStateSemantics | undefined,
  decision: SpanSegmentationDecisionV1,
  finalSpan: boolean
): StateDeltaOperation[] {
  if (!semantics) return [];
  const operations: StateDeltaOperation[] = [
    ...semantics.effects.map((item): StateDeltaOperation => ({
      op: "fact.upsert",
      subject: clip(item.text, 300),
      value: item.text,
      status: "observed",
      sourceRefs: item.evidenceRefs
    })),
    ...semantics.artifacts.map((item): StateDeltaOperation => ({
      op: item.status === "verified" ? "artifact.verify" : "artifact.upsert",
      subject: clip(item.text, 300),
      value: item.text,
      status: item.status,
      sourceRefs: item.evidenceRefs
    })),
    ...semantics.issuesOpened.map((item): StateDeltaOperation => ({
      op: "issue.upsert",
      subject: clip(item.text, 300),
      value: item.text,
      status: "open",
      sourceRefs: item.evidenceRefs
    })),
    ...semantics.issuesResolved.map((item): StateDeltaOperation => ({
      op: "issue.resolve",
      subject: clip(item.text, 300),
      value: item.text,
      status: "resolved",
      sourceRefs: item.evidenceRefs
    })),
    ...semantics.verification.map((item): StateDeltaOperation => ({
      op: "verification.set",
      subject: clip(item.text, 300),
      value: item.text,
      status: item.status,
      sourceRefs: item.evidenceRefs
    }))
  ];
  if (finalSpan) {
    const taskStatus = decision.terminationStatus === "success"
      ? "completed"
      : decision.terminationStatus === "blocked"
        ? "blocked"
        : "failed";
    operations.push({
      op: "status.set",
      subject: "episode_status",
      status: taskStatus,
      sourceRefs: decision.evidenceRefs
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

function parseSpanCapabilities(
  value: unknown,
  decisions: readonly SpanSegmentationDecisionV1[]
): Map<number, { capabilityGoal: string; procedureSemantic: string }> {
  if (!Array.isArray(value)) throw new Error("procedural Span capability LLM returned invalid spans");
  const knownIndexes = new Set(decisions.map((decision) => decision.spanIndex));
  const parsed = new Map<number, { capabilityGoal: string; procedureSemantic: string }>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.span_index !== "number" ||
      !Number.isInteger(item.span_index) || !knownIndexes.has(item.span_index)) {
      throw new Error("procedural Span capability LLM returned unknown span_index");
    }
    if (parsed.has(item.span_index)) {
      throw new Error(`procedural Span capability LLM duplicated span_index: ${item.span_index}`);
    }
    const capabilityGoal = cleanText(item.capability_goal, 600);
    const procedureSemantic = cleanText(item.procedure_semantic, 2_000);
    if (!capabilityGoal || !procedureSemantic) {
      throw new Error(
        `procedural Span capability LLM omitted capability_goal or procedure_semantic: ${item.span_index}`
      );
    }
    parsed.set(item.span_index, { capabilityGoal, procedureSemantic });
  }
  for (const decision of decisions) {
    if (!parsed.has(decision.spanIndex)) {
      throw new Error(`procedural Span capability LLM omitted span_index: ${decision.spanIndex}`);
    }
  }
  return parsed;
}

function parseSpanStates(
  value: unknown,
  decisions: readonly SpanSegmentationDecisionV1[],
  stepById: ReadonlyMap<string, ExecutionStepV1>,
  options: { dropInvalidEvidenceRefs?: boolean } = {}
): SpanStateSemantics[] {
  if (!Array.isArray(value)) throw new Error("procedural Span state LLM returned invalid spans");
  const decisionByIndex = new Map(decisions.map((decision) => [decision.spanIndex, decision]));
  const parsed = new Map<number, SpanStateSemantics>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.span_index !== "number" || !Number.isInteger(item.span_index)) {
      throw new Error("procedural Span state LLM returned invalid span_index");
    }
    const decision = decisionByIndex.get(item.span_index);
    if (!decision) throw new Error(`procedural Span state LLM invented span_index: ${item.span_index}`);
    if (parsed.has(item.span_index)) {
      throw new Error(`procedural Span state LLM duplicated span_index: ${item.span_index}`);
    }
    const allowedRefs = new Set(unique([
      ...decision.evidenceRefs,
      ...decision.stepIds.flatMap((stepId) => stepEvidenceRefs(stepById.get(stepId)!))
    ]));
    parsed.set(item.span_index, {
      spanIndex: item.span_index,
      effects: parseSpanStateItems(item.effects, "summary", allowedRefs, options),
      artifacts: parseArtifactStateItems(item.artifacts, allowedRefs, options),
      issuesOpened: parseSpanStateItems(item.issues_opened, "issue", allowedRefs, options),
      issuesResolved: parseSpanStateItems(item.issues_resolved, "issue", allowedRefs, options),
      verification: parseVerificationStateItems(item.verification, allowedRefs, options)
    });
  }
  for (const decision of decisions) {
    if (!parsed.has(decision.spanIndex)) {
      throw new Error(`procedural Span state LLM omitted span_index: ${decision.spanIndex}`);
    }
  }
  return decisions.map((decision) => parsed.get(decision.spanIndex)!);
}

function parseSpanStateItems(
  value: unknown,
  textField: "summary" | "issue",
  allowedRefs: ReadonlySet<string>,
  options: { dropInvalidEvidenceRefs?: boolean } = {}
): SpanStateEvidenceItem[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error(`procedural Span state ${textField} items must be an array`);
  return value.flatMap((item) => {
    if (!isRecord(item)) throw new Error(`procedural Span state ${textField} item must be an object`);
    const itemText = cleanText(item[textField], 1_500);
    const evidenceRefs = validatedEvidenceRefs(item.evidence_refs, allowedRefs, options);
    if (!itemText) {
      throw new Error(`procedural Span state ${textField} item is incomplete`);
    }
    if (evidenceRefs.length === 0 && options.dropInvalidEvidenceRefs) return [];
    if (evidenceRefs.length === 0) {
      throw new Error(`procedural Span state ${textField} item is incomplete`);
    }
    return [{ text: itemText, evidenceRefs }];
  });
}

function parseArtifactStateItems(
  value: unknown,
  allowedRefs: ReadonlySet<string>,
  options: { dropInvalidEvidenceRefs?: boolean } = {}
): SpanArtifactStateItem[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("procedural Span state artifacts must be an array");
  return value.flatMap((item) => {
    if (!isRecord(item)) throw new Error("procedural Span state artifact item must be an object");
    const name = cleanText(item.name, 1_500);
    const status = item.status;
    const evidenceRefs = validatedEvidenceRefs(item.evidence_refs, allowedRefs, options);
    if (!name ||
      (status !== "created" && status !== "updated" && status !== "verified")) {
      throw new Error(
        `procedural Span state artifact item is invalid (keys=${Object.keys(item).sort().join(",")}, ` +
        `status=${typeof status === "string" ? status : typeof status}, evidenceRefs=${evidenceRefs.length})`
      );
    }
    if (evidenceRefs.length === 0 && options.dropInvalidEvidenceRefs) return [];
    if (evidenceRefs.length === 0) {
      throw new Error("procedural Span state artifact item has no valid evidence refs");
    }
    return [{ text: name, status, evidenceRefs }];
  });
}

function parseVerificationStateItems(
  value: unknown,
  allowedRefs: ReadonlySet<string>,
  options: { dropInvalidEvidenceRefs?: boolean } = {}
): SpanVerificationStateItem[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("procedural Span state verification must be an array");
  return value.flatMap((item) => {
    if (!isRecord(item)) throw new Error("procedural Span state verification item must be an object");
    const criterion = cleanText(item.criterion, 1_500);
    const status = item.status;
    const evidenceRefs = validatedEvidenceRefs(item.evidence_refs, allowedRefs, options);
    if (!criterion ||
      (status !== "passed" && status !== "failed" && status !== "partial" && status !== "unknown")) {
      throw new Error("procedural Span state verification item is invalid");
    }
    if (evidenceRefs.length === 0 && options.dropInvalidEvidenceRefs) return [];
    if (evidenceRefs.length === 0) {
      throw new Error("procedural Span state verification item has no valid evidence refs");
    }
    return [{ text: criterion, status, evidenceRefs }];
  });
}

function validatedEvidenceRefs(
  value: unknown,
  allowedRefs: ReadonlySet<string>,
  options: { dropInvalidEvidenceRefs?: boolean }
): string[] {
  const refs = stringArray(value);
  if (!options.dropInvalidEvidenceRefs) {
    assertAllowedRefs(refs, allowedRefs);
    return refs;
  }
  return refs.filter((ref) => allowedRefs.has(ref));
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

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((item): item is string => typeof item === "string" && Boolean(item.trim())))]
    .map((item) => item.trim());
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

function assertAllowedRefs(refs: readonly string[], allowedRefs: ReadonlySet<string>): void {
  for (const ref of refs) {
    if (!allowedRefs.has(ref)) throw new Error(`procedural reconstruction invented source ref: ${ref}`);
  }
}

function parseTerminationStatus(value: unknown): ProceduralSpanTermination {
  if (value === "success" || value === "failure" || value === "blocked" || value === "abandoned") {
    return value;
  }
  throw new Error("procedural span LLM returned invalid termination status");
}

function isNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
