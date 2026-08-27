import { describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  EPISODE_BOUNDARY_MAX_TOKENS,
  EPISODE_BOUNDARY_OPERATION,
  EPISODE_BOUNDARY_PROMPT,
  EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION,
  EXECUTION_STEP_SCHEMA_VERSION,
  EpisodeBoundarySegmenter,
  buildEpisodeBoundaryWindows,
  emptyObservedState,
  type EpisodeProceduralPathV2,
  type ExecutionStepV1,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage
} from "../../../src/index.js";

describe("EpisodeBoundarySegmenter", () => {
  it("uses 15-Step windows with five read-only overlap Steps and no duplicate decisions", () => {
    const windows = buildEpisodeBoundaryWindows(makeSteps(26));

    expect(windows).toHaveLength(3);
    expect(windows.map((window) => [window.startStepIndex, window.endStepIndex]))
      .toEqual([[0, 14], [10, 24], [20, 25]]);
    expect(windows[0]!.requiredBeforeStepIndices).toEqual(range(1, 14));
    expect(windows[1]!.contextOnlyStepIndices).toEqual(range(10, 14));
    expect(windows[1]!.requiredBeforeStepIndices).toEqual(range(15, 24));
    expect(windows[2]!.contextOnlyStepIndices).toEqual(range(20, 24));
    expect(windows[2]!.requiredBeforeStepIndices).toEqual([25]);
    expect(new Set(windows.flatMap((window) => window.requiredBeforeStepIndices)).size).toBe(25);
  });

  it("keeps repair and verification inside a subproblem and compiles only binary boundaries", async () => {
    const operations: string[] = [];
    const segmenter = new EpisodeBoundarySegmenter({
      llm: boundaryLlm(operations, new Set([8, 20])),
      enableThinking: true
    });
    const result = await segmenter.segment(makePath(26));

    expect(operations).toEqual([
      `${EPISODE_BOUNDARY_OPERATION}.window`,
      `${EPISODE_BOUNDARY_OPERATION}.window`,
      `${EPISODE_BOUNDARY_OPERATION}.window`
    ]);
    expect(result.decisions).toHaveLength(25);
    expect(result.decisions.filter((decision) => decision.boundary === 1)
      .map((decision) => decision.beforeStepIndex)).toEqual([8, 20]);
    expect(result.segments.map((segment) => [segment.startStepIndex, segment.endStepIndex]))
      .toEqual([[0, 7], [8, 19], [20, 25]]);
    expect(EPISODE_BOUNDARY_PROMPT).toContain("incorrect attempt");
    expect(EPISODE_BOUNDARY_PROMPT).toContain("failure");
    expect(EPISODE_BOUNDARY_PROMPT).toContain("verification");
    expect(EPISODE_BOUNDARY_PROMPT).toContain("Turn boundary");
    expect(EPISODE_BOUNDARY_MAX_TOKENS).toBe(16_000);
  });
});

function boundaryLlm(operations: string[], boundaries: ReadonlySet<number>): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/boundary-test",
      model: "boundary-test"
    },
    isConfigured() { return true; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      operations.push(options.operation);
      const payload = JSON.parse(messages.at(-1)!.content) as {
        window: { requiredBeforeStepIndices: number[] };
      };
      return {
        boundaries: payload.window.requiredBeforeStepIndices.map((stepIndex) => ({
          before_step_index: stepIndex,
          boundary: boundaries.has(stepIndex) ? 1 : 0
        }))
      } as unknown as T;
    },
    status() {
      return { provider: "host", model: "boundary-test", configured: true, remote: false };
    }
  };
}

function makePath(count: number): EpisodeProceduralPathV2 {
  const state = emptyObservedState();
  return {
    id: "episode_path_boundary",
    schemaVersion: EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION,
    episodeId: "episode-boundary",
    states: [state],
    steps: makeSteps(count),
    spans: [],
    segmentationDecisions: [],
    sourceSnapshotHash: "snapshot-boundary",
    pathHash: "path-boundary"
  };
}

function makeSteps(count: number): ExecutionStepV1[] {
  const state = emptyObservedState();
  return Array.from({ length: count }, (_, stepIndex) => ({
    id: `step-${stepIndex}`,
    schemaVersion: EXECUTION_STEP_SCHEMA_VERSION,
    episodeId: "episode-boundary",
    rawTurnId: "raw-boundary",
    turnIndex: 0,
    stepIndex,
    preStateId: state.id,
    action: {
      kind: "tool_action",
      type: stepIndex % 3 === 0 ? "inspect" : stepIndex % 3 === 1 ? "execute" : "verify",
      intent: `advance local operation ${stepIndex}`,
      summary: `observable result ${stepIndex}`,
      eventRefs: [`event-${stepIndex}`]
    },
    actionEffectDelta: [],
    actionPostStateId: state.id,
    externalObservationDelta: [],
    postStateId: state.id,
    outcome: {
      status: stepIndex === 6 ? "failure" : "success",
      evidenceRefs: [`event-${stepIndex}`]
    },
    cost: { toolCalls: 1, errorCount: stepIndex === 6 ? 1 : 0 },
    provenance: {
      algorithmVersion: "boundary-test",
      sourceSnapshotHash: "snapshot-boundary"
    }
  }));
}

function range(start: number, end: number): number[] {
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}
