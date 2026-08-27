import type { LlmClient, LlmMessage } from "../../model/types.js";
import { stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import {
  compileEpisodeTaskContract,
  type EpisodeTaskContractV1
} from "./episode-boundary-segmentation.js";
import type {
  EpisodeProceduralPathV2,
  ExecutionStepV1
} from "./procedural-path-model.js";

export const SUBPROBLEM_CONTRACT_SEGMENTATION_VERSION =
  "subproblem-contract-segmentation.v1" as const;
export const SUBPROBLEM_CONTRACT_OPERATION = "procedural.subproblem_contract.v1";
export const SUBPROBLEM_CONTRACT_WINDOW_SIZE = 15;
export const SUBPROBLEM_CONTRACT_WINDOW_OVERLAP = 5;
export const SUBPROBLEM_CONTRACT_MAX_TOKENS = 16_000;
const MAX_CONTRACT_REPAIR_ATTEMPTS = 2;

export const SUBPROBLEM_CONTRACT_PROMPT = `You segment an agent Episode by maintaining one persistent reusable subproblem contract at a time.

A SubproblemContract has exactly two fields:
- target: the stable object, information set, artifact, or system state being advanced;
- desired_outcome: the locally verifiable result that ends this subproblem.

The contract describes a complete execution lifecycle, not the immediate tool action. Infer it from the whole supplied window and the Episode contract. Use the broadest contract that explains a coherent contiguous prefix without absorbing the next independently reusable objective.

Examples of one lifecycle that must stay under one contract:
- inspect -> incorrect edit -> failed test -> diagnose -> corrected edit -> passed test;
- query several sources -> encounter access failures -> change sources -> obtain enough evidence;
- install a dependency -> write a generator -> execute it -> inspect output -> repair -> verify the artifact.

Different URLs, records, products, files, tools, action types, outcomes, retries, or Turn boundaries do not create a new contract when they still advance or recover the same local outcome. Evidence acquisition and artifact production may be separate contracts only when the collected evidence is itself a completed reusable intermediate result and execution clearly starts producing a different artifact outcome. User feedback starts a new contract only when it introduces a distinct repair target or acceptance outcome after a prior result was delivered.

The request has one of two modes:
- open_segment: infer a contract from the whole candidate window, then return the earliest candidate Step that does not belong to it. The first candidate must belong to the new contract.
- continue_segment: the supplied currentContract is immutable. Return the earliest candidate Step that no longer advances, recovers, or verifies that contract.

contextOnly Steps are read-only continuity context and can never be returned as a boundary. Return null when every candidate belongs to the current contract. Do not revise or echo a contract in continue_segment mode.

Return JSON only:
{
  "contract": {
    "target": "the evidence set needed for the requested analysis",
    "desired_outcome": "sufficient verified evidence has been collected for artifact production"
  },
  "first_non_member_step_index": null
}

In continue_segment mode, contract must be null.`;

export interface SubproblemContractV1 {
  target: string;
  desiredOutcome: string;
}

export interface SubproblemContractWindowV1 {
  windowIndex: number;
  segmentIndex: number;
  mode: "open_segment" | "continue_segment";
  startStepIndex: number;
  endStepIndex: number;
  contextOnlyStepIndices: number[];
  candidateStepIndices: number[];
  firstNonMemberStepIndex?: number;
}

export interface SubproblemContractSegmentV1 {
  segmentIndex: number;
  startStepIndex: number;
  endStepIndex: number;
  contract: SubproblemContractV1;
  stepIds: string[];
  intents: string[];
  summaries: string[];
}

export interface SubproblemContractSegmentationResultV1 {
  schemaVersion: typeof SUBPROBLEM_CONTRACT_SEGMENTATION_VERSION;
  episodeId: string;
  episodeContract: EpisodeTaskContractV1;
  windows: SubproblemContractWindowV1[];
  segments: SubproblemContractSegmentV1[];
  model?: string;
}

interface SubproblemContractSegmenterDeps {
  llm: LlmClient;
  enableThinking: boolean;
}

interface ContractWindowResponse {
  contract?: SubproblemContractV1;
  firstNonMemberStepIndex?: number;
}

export class EpisodeSubproblemContractSegmenter {
  constructor(private readonly deps: SubproblemContractSegmenterDeps) {}

  async segment(
    path: EpisodeProceduralPathV2
  ): Promise<SubproblemContractSegmentationResultV1> {
    if (!this.deps.llm.isConfigured()) {
      throw new Error("subproblem contract segmentation requires a configured LLM");
    }
    const episodeContract = compileEpisodeTaskContract(path.states);
    const windows: SubproblemContractWindowV1[] = [];
    const segments: SubproblemContractSegmentV1[] = [];
    let segmentStartPosition = 0;
    let windowIndex = 0;

    while (segmentStartPosition < path.steps.length) {
      const segmentIndex = segments.length;
      let contract: SubproblemContractV1 | undefined;
      let nextCandidatePosition = segmentStartPosition;
      let segmentClosed = false;

      while (!segmentClosed) {
        const mode = contract ? "continue_segment" as const : "open_segment" as const;
        const contextStartPosition = contract
          ? Math.max(segmentStartPosition, nextCandidatePosition - SUBPROBLEM_CONTRACT_WINDOW_OVERLAP)
          : nextCandidatePosition;
        const contextSteps = path.steps.slice(contextStartPosition, nextCandidatePosition);
        const candidateCapacity = SUBPROBLEM_CONTRACT_WINDOW_SIZE - contextSteps.length;
        const candidateSteps = path.steps.slice(
          nextCandidatePosition,
          nextCandidatePosition + candidateCapacity
        );
        if (candidateSteps.length === 0) {
          if (!contract) throw new Error(`subproblem Segment ${segmentIndex} has no contract`);
          segments.push(compileSegment(
            segmentIndex,
            path.steps.slice(segmentStartPosition),
            contract
          ));
          segmentStartPosition = path.steps.length;
          segmentClosed = true;
          break;
        }
        const response = await this.evaluateWindow({
          episodeId: path.episodeId,
          episodeContract,
          windowIndex,
          segmentIndex,
          mode,
          currentContract: contract,
          contextSteps,
          candidateSteps
        });
        contract = response.contract ?? contract;
        if (!contract) throw new Error(`subproblem Segment ${segmentIndex} contract was not established`);
        const allWindowSteps = [...contextSteps, ...candidateSteps];
        windows.push({
          windowIndex,
          segmentIndex,
          mode,
          startStepIndex: allWindowSteps[0]!.stepIndex,
          endStepIndex: allWindowSteps.at(-1)!.stepIndex,
          contextOnlyStepIndices: contextSteps.map((step) => step.stepIndex),
          candidateStepIndices: candidateSteps.map((step) => step.stepIndex),
          ...(response.firstNonMemberStepIndex === undefined
            ? {}
            : { firstNonMemberStepIndex: response.firstNonMemberStepIndex })
        });
        windowIndex += 1;

        if (response.firstNonMemberStepIndex !== undefined) {
          const boundaryPosition = path.steps.findIndex(
            (step) => step.stepIndex === response.firstNonMemberStepIndex
          );
          if (boundaryPosition <= segmentStartPosition) {
            throw new Error(
              `subproblem boundary does not leave a non-empty Segment: ${response.firstNonMemberStepIndex}`
            );
          }
          segments.push(compileSegment(
            segmentIndex,
            path.steps.slice(segmentStartPosition, boundaryPosition),
            contract
          ));
          segmentStartPosition = boundaryPosition;
          segmentClosed = true;
          continue;
        }

        nextCandidatePosition += candidateSteps.length;
        if (nextCandidatePosition >= path.steps.length) {
          segments.push(compileSegment(
            segmentIndex,
            path.steps.slice(segmentStartPosition),
            contract
          ));
          segmentStartPosition = path.steps.length;
          segmentClosed = true;
        }
      }
    }

    return {
      schemaVersion: SUBPROBLEM_CONTRACT_SEGMENTATION_VERSION,
      episodeId: path.episodeId,
      episodeContract,
      windows,
      segments,
      ...(this.deps.llm.config.model ? { model: this.deps.llm.config.model } : {})
    };
  }

  private async evaluateWindow(input: {
    episodeId: string;
    episodeContract: EpisodeTaskContractV1;
    windowIndex: number;
    segmentIndex: number;
    mode: "open_segment" | "continue_segment";
    currentContract?: SubproblemContractV1;
    contextSteps: readonly ExecutionStepV1[];
    candidateSteps: readonly ExecutionStepV1[];
  }): Promise<ContractWindowResponse> {
    type LlmResult = {
      contract?: unknown;
      first_non_member_step_index?: unknown;
      firstNonMemberStepIndex?: unknown;
    };
    const contextOnly = new Set(input.contextSteps.map((step) => step.stepIndex));
    const allSteps = [...input.contextSteps, ...input.candidateSteps];
    const messages: LlmMessage[] = [
      { role: "system", content: SUBPROBLEM_CONTRACT_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId: input.episodeId,
          episodeContract: input.episodeContract,
          mode: input.mode,
          currentContract: input.currentContract ?? null,
          window: {
            windowIndex: input.windowIndex,
            segmentIndex: input.segmentIndex,
            candidateStepIndices: input.candidateSteps.map((step) => step.stepIndex)
          },
          steps: allSteps.map((step) => ({
            stepIndex: step.stepIndex,
            turnIndex: step.turnIndex,
            contextOnly: contextOnly.has(step.stepIndex),
            actionType: step.action.type,
            toolName: step.action.toolName,
            intent: step.action.intent,
            summary: step.action.summary,
            outcome: step.outcome.status,
            retryOfStepId: step.retryOfStepId,
            recoveryFromStepId: step.recoveryFromStepId
          }))
        })
      }
    ];
    const options = {
      operation: `${SUBPROBLEM_CONTRACT_OPERATION}.${input.mode}`,
      thinkingMode: this.deps.enableThinking ? "enabled" as const : "disabled" as const,
      temperature: 0,
      maxTokens: SUBPROBLEM_CONTRACT_MAX_TOKENS
    };
    let result = await this.deps.llm.completeJson<LlmResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseWindowResponse(result, input);
      } catch (error) {
        if (repairAttempt >= MAX_CONTRACT_REPAIR_ATTEMPTS) throw error;
        const message = error instanceof Error ? error.message : String(error);
        result = await this.deps.llm.completeJson<LlmResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          {
            role: "user",
            content: `Correct only the JSON schema. Validation error: ${message}\nAllowed candidate Step indices are ${stableStringify(input.candidateSteps.map((step) => step.stepIndex))}. first_non_member_step_index must be the earliest non-member candidate or null. In open_segment mode return a non-empty target and desired_outcome. In continue_segment mode return contract:null. Return JSON only.`
          }
        ], {
          ...options,
          operation: `${SUBPROBLEM_CONTRACT_OPERATION}.repair.${repairAttempt + 1}`
        });
      }
    }
  }
}

function parseWindowResponse(
  value: {
    contract?: unknown;
    first_non_member_step_index?: unknown;
    firstNonMemberStepIndex?: unknown;
  },
  input: {
    mode: "open_segment" | "continue_segment";
    candidateSteps: readonly ExecutionStepV1[];
  }
): ContractWindowResponse {
  const contract = parseContract(value.contract);
  if (input.mode === "open_segment" && !contract) {
    throw new Error("open_segment response requires contract");
  }
  if (input.mode === "continue_segment" && contract) {
    throw new Error("continue_segment response must not revise or echo contract");
  }
  const rawBoundary = value.first_non_member_step_index ?? value.firstNonMemberStepIndex;
  let firstNonMemberStepIndex: number | undefined;
  if (rawBoundary !== null && rawBoundary !== undefined) {
    if (!Number.isInteger(rawBoundary)) {
      throw new Error("first_non_member_step_index must be an integer or null");
    }
    firstNonMemberStepIndex = rawBoundary as number;
    const allowed = input.candidateSteps.map((step) => step.stepIndex);
    if (!allowed.includes(firstNonMemberStepIndex)) {
      throw new Error(`unknown or context-only first_non_member_step_index: ${firstNonMemberStepIndex}`);
    }
    if (input.mode === "open_segment" && firstNonMemberStepIndex === allowed[0]) {
      throw new Error("the first open_segment candidate must belong to the new contract");
    }
  }
  return {
    ...(contract ? { contract } : {}),
    ...(firstNonMemberStepIndex === undefined ? {} : { firstNonMemberStepIndex })
  };
}

function parseContract(value: unknown): SubproblemContractV1 | undefined {
  if (value === null || value === undefined) return undefined;
  if (!isRecord(value)) throw new Error("contract must be an object or null");
  const target = requiredText(value.target, "contract.target");
  const desiredOutcome = requiredText(
    value.desired_outcome ?? value.desiredOutcome,
    "contract.desired_outcome"
  );
  return { target, desiredOutcome };
}

function compileSegment(
  segmentIndex: number,
  steps: readonly ExecutionStepV1[],
  contract: SubproblemContractV1
): SubproblemContractSegmentV1 {
  if (steps.length === 0) throw new Error(`subproblem Segment ${segmentIndex} is empty`);
  return {
    segmentIndex,
    startStepIndex: steps[0]!.stepIndex,
    endStepIndex: steps.at(-1)!.stepIndex,
    contract,
    stepIds: steps.map((step) => step.id),
    intents: steps.map((step) => step.action.intent),
    summaries: steps.map((step) => step.action.summary)
  };
}

function requiredText(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${field} must be non-empty`);
  return value.trim();
}
