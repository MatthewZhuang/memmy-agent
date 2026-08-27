import type { LlmClient, LlmMessage } from "../../model/types.js";
import { stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import type { ObservedStateV1 } from "./span-v3-model.js";
import type {
  EpisodeProceduralPathV2,
  ExecutionStepV1
} from "./procedural-path-model.js";

export const EPISODE_BOUNDARY_SEGMENTATION_VERSION =
  "episode-boundary-segmentation.v1" as const;
export const EPISODE_BOUNDARY_OPERATION = "procedural.episode_boundary.v1";
export const EPISODE_BOUNDARY_WINDOW_SIZE = 15;
export const EPISODE_BOUNDARY_WINDOW_OVERLAP = 5;
export const EPISODE_BOUNDARY_MAX_TOKENS = 16_000;
const EPISODE_BOUNDARY_WINDOW_STRIDE =
  EPISODE_BOUNDARY_WINDOW_SIZE - EPISODE_BOUNDARY_WINDOW_OVERLAP;
const MAX_BOUNDARY_REPAIR_ATTEMPTS = 2;

export const EPISODE_BOUNDARY_PROMPT = `You decide whether adjacent execution Steps belong to the same reusable subproblem.

For every required edge before Step i, output exactly one binary decision:
- boundary=0: Step i continues the same subproblem as Step i-1.
- boundary=1: Step i starts a different subproblem.

A subproblem is a continuous attempt to move the same target toward the same locally verifiable outcome. Keep the complete lifecycle together: inspection, evidence collection, an incorrect attempt, failure, diagnosis, a changed strategy, the correct attempt, and verification are one Segment when they still solve the same local problem.

Use boundary=1 only when the local target or locally verifiable outcome changes, or when the previous subproblem has completed, been abandoned, or become blocked and execution begins an independently reusable objective. A tool change, action-type change, retry, failure, recovery, verification Step, response Step, Turn boundary, or new user message is not sufficient by itself. User feedback starts a new Segment only when it introduces a distinct repair target or acceptance contract after the prior result was delivered.

The episodeContract is read-only global context. Steps marked contextOnly are overlap context and must not be returned. Return every requiredBeforeStepIndex exactly once and no other index.

Return JSON only:
{
  "boundaries": [
    { "before_step_index": 15, "boundary": 0 }
  ]
}`;

export interface EpisodeTaskContractV1 {
  activeGoal?: string;
  goalHistory: string[];
  constraints: string[];
  acceptanceCriteria: string[];
}

export interface EpisodeBoundaryDecisionV1 {
  beforeStepIndex: number;
  boundary: 0 | 1;
  windowIndex: number;
}

export interface EpisodeBoundaryWindowV1 {
  windowIndex: number;
  startStepIndex: number;
  endStepIndex: number;
  contextOnlyStepIndices: number[];
  requiredBeforeStepIndices: number[];
}

export interface EpisodeSegmentV1 {
  segmentIndex: number;
  startStepIndex: number;
  endStepIndex: number;
  stepIds: string[];
  intents: string[];
  summaries: string[];
}

export interface EpisodeBoundarySegmentationResultV1 {
  schemaVersion: typeof EPISODE_BOUNDARY_SEGMENTATION_VERSION;
  episodeId: string;
  episodeContract: EpisodeTaskContractV1;
  windows: EpisodeBoundaryWindowV1[];
  decisions: EpisodeBoundaryDecisionV1[];
  segments: EpisodeSegmentV1[];
  model?: string;
}

interface EpisodeBoundarySegmenterDeps {
  llm: LlmClient;
  enableThinking: boolean;
}

interface InternalBoundaryWindow extends EpisodeBoundaryWindowV1 {
  steps: ExecutionStepV1[];
}

export class EpisodeBoundarySegmenter {
  constructor(private readonly deps: EpisodeBoundarySegmenterDeps) {}

  async segment(path: EpisodeProceduralPathV2): Promise<EpisodeBoundarySegmentationResultV1> {
    if (!this.deps.llm.isConfigured()) {
      throw new Error("episode boundary segmentation requires a configured LLM");
    }
    const episodeContract = compileEpisodeTaskContract(path.states);
    const windows = buildEpisodeBoundaryWindows(path.steps);
    const decisions: EpisodeBoundaryDecisionV1[] = [];
    for (const window of windows) {
      decisions.push(...await this.segmentWindow(path.episodeId, episodeContract, window));
    }
    const orderedDecisions = [...decisions].sort(
      (left, right) => left.beforeStepIndex - right.beforeStepIndex
    );
    return {
      schemaVersion: EPISODE_BOUNDARY_SEGMENTATION_VERSION,
      episodeId: path.episodeId,
      episodeContract,
      windows: windows.map(({ steps: _steps, ...window }) => window),
      decisions: orderedDecisions,
      segments: compileEpisodeSegments(path.steps, orderedDecisions),
      ...(this.deps.llm.config.model ? { model: this.deps.llm.config.model } : {})
    };
  }

  private async segmentWindow(
    episodeId: string,
    episodeContract: EpisodeTaskContractV1,
    window: InternalBoundaryWindow
  ): Promise<EpisodeBoundaryDecisionV1[]> {
    type BoundaryResult = { boundaries?: unknown };
    const contextOnly = new Set(window.contextOnlyStepIndices);
    const messages: LlmMessage[] = [
      { role: "system", content: EPISODE_BOUNDARY_PROMPT },
      {
        role: "user",
        content: stableStringify({
          episodeId,
          episodeContract,
          window: {
            windowIndex: window.windowIndex,
            startStepIndex: window.startStepIndex,
            endStepIndex: window.endStepIndex,
            requiredBeforeStepIndices: window.requiredBeforeStepIndices
          },
          steps: window.steps.map((step) => ({
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
      operation: `${EPISODE_BOUNDARY_OPERATION}.window`,
      thinkingMode: this.deps.enableThinking ? "enabled" as const : "disabled" as const,
      temperature: 0,
      maxTokens: EPISODE_BOUNDARY_MAX_TOKENS
    };
    let result = await this.deps.llm.completeJson<BoundaryResult>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return parseBoundaryDecisions(result.boundaries, window);
      } catch (error) {
        if (repairAttempt >= MAX_BOUNDARY_REPAIR_ATTEMPTS) throw error;
        const message = error instanceof Error ? error.message : String(error);
        result = await this.deps.llm.completeJson<BoundaryResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          {
            role: "user",
            content: `Correct only the JSON schema. Validation error: ${message}\nReturn exactly one item for each required before_step_index: ${stableStringify(window.requiredBeforeStepIndices)}. Use only integer boundary 0 or 1. Do not return context-only indices, explanations, markdown, or extra fields.`
          }
        ], {
          ...options,
          operation: `${EPISODE_BOUNDARY_OPERATION}.repair.${repairAttempt + 1}`
        });
      }
    }
  }
}

export function compileEpisodeTaskContract(
  states: readonly ObservedStateV1[]
): EpisodeTaskContractV1 {
  const goals = uniqueText(states.flatMap((state) => textValue(state.goal?.value)));
  const constraints = uniqueText(states.flatMap((state) =>
    state.constraints.flatMap((entry) => textValue(entry.value ?? entry.subject))));
  const acceptanceCriteria = uniqueText(states.flatMap((state) =>
    state.verification.flatMap((entry) => textValue(entry.value ?? entry.subject))));
  const activeGoal = [...states].reverse()
    .flatMap((state) => textValue(state.goal?.value))[0];
  return {
    ...(activeGoal ? { activeGoal } : {}),
    goalHistory: goals,
    constraints,
    acceptanceCriteria
  };
}

export function buildEpisodeBoundaryWindows(
  steps: readonly ExecutionStepV1[]
): InternalBoundaryWindow[] {
  if (steps.length <= 1) return [];
  const windows: InternalBoundaryWindow[] = [];
  for (let start = 0, windowIndex = 0; start < steps.length; start += EPISODE_BOUNDARY_WINDOW_STRIDE) {
    const windowSteps = steps.slice(start, start + EPISODE_BOUNDARY_WINDOW_SIZE);
    if (windowSteps.length === 0) break;
    const newStepOffset = windowIndex === 0 ? 1 : EPISODE_BOUNDARY_WINDOW_OVERLAP;
    const requiredSteps = windowSteps.slice(newStepOffset);
    if (requiredSteps.length === 0) break;
    const contextOnlySteps = windowSteps.slice(0, newStepOffset);
    windows.push({
      windowIndex,
      startStepIndex: windowSteps[0]!.stepIndex,
      endStepIndex: windowSteps.at(-1)!.stepIndex,
      contextOnlyStepIndices: contextOnlySteps.map((step) => step.stepIndex),
      requiredBeforeStepIndices: requiredSteps.map((step) => step.stepIndex),
      steps: [...windowSteps]
    });
    if (start + EPISODE_BOUNDARY_WINDOW_SIZE >= steps.length) break;
    windowIndex += 1;
  }
  return windows;
}

export function compileEpisodeSegments(
  steps: readonly ExecutionStepV1[],
  decisions: readonly EpisodeBoundaryDecisionV1[]
): EpisodeSegmentV1[] {
  if (steps.length === 0) return [];
  const boundaries = new Set(
    decisions.filter((decision) => decision.boundary === 1)
      .map((decision) => decision.beforeStepIndex)
  );
  const groups: ExecutionStepV1[][] = [];
  let current: ExecutionStepV1[] = [];
  for (const step of steps) {
    if (current.length > 0 && boundaries.has(step.stepIndex)) {
      groups.push(current);
      current = [];
    }
    current.push(step);
  }
  if (current.length > 0) groups.push(current);
  return groups.map((group, segmentIndex) => ({
    segmentIndex,
    startStepIndex: group[0]!.stepIndex,
    endStepIndex: group.at(-1)!.stepIndex,
    stepIds: group.map((step) => step.id),
    intents: group.map((step) => step.action.intent),
    summaries: group.map((step) => step.action.summary)
  }));
}

function parseBoundaryDecisions(
  value: unknown,
  window: InternalBoundaryWindow
): EpisodeBoundaryDecisionV1[] {
  if (!Array.isArray(value)) throw new Error("boundary response requires boundaries[]");
  const allowed = new Set(window.requiredBeforeStepIndices);
  const parsed = new Map<number, EpisodeBoundaryDecisionV1>();
  for (const item of value) {
    if (!isRecord(item)) throw new Error("boundary item must be an object");
    const beforeStepIndex = integerField(item, "before_step_index", "beforeStepIndex");
    if (!allowed.has(beforeStepIndex)) {
      throw new Error(`unknown or context-only before_step_index: ${beforeStepIndex}`);
    }
    if (parsed.has(beforeStepIndex)) {
      throw new Error(`duplicate before_step_index: ${beforeStepIndex}`);
    }
    const boundary = integerField(item, "boundary");
    if (boundary !== 0 && boundary !== 1) {
      throw new Error(`boundary must be 0 or 1 for Step ${beforeStepIndex}`);
    }
    parsed.set(beforeStepIndex, {
      beforeStepIndex,
      boundary,
      windowIndex: window.windowIndex
    });
  }
  const missing = window.requiredBeforeStepIndices.filter((stepIndex) => !parsed.has(stepIndex));
  if (missing.length > 0) {
    throw new Error(`missing before_step_index values: ${missing.join(", ")}`);
  }
  return window.requiredBeforeStepIndices.map((stepIndex) => parsed.get(stepIndex)!);
}

function integerField(
  value: Record<string, unknown>,
  key: string,
  alias?: string
): number {
  const raw = value[key] ?? (alias ? value[alias] : undefined);
  if (!Number.isInteger(raw)) throw new Error(`${key} must be an integer`);
  return raw as number;
}

function textValue(value: unknown): string[] {
  if (typeof value !== "string" || !value.trim()) return [];
  return [value.trim()];
}

function uniqueText(values: readonly string[]): string[] {
  return [...new Set(values)];
}
