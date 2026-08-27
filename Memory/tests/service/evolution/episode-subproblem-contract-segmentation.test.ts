import { describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION,
  EXECUTION_STEP_SCHEMA_VERSION,
  SUBPROBLEM_CONTRACT_MAX_TOKENS,
  SUBPROBLEM_CONTRACT_OPERATION,
  SUBPROBLEM_CONTRACT_PROMPT,
  EpisodeSubproblemContractSegmenter,
  emptyObservedState,
  type EpisodeProceduralPathV2,
  type ExecutionStepV1,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage
} from "../../../src/index.js";

describe("EpisodeSubproblemContractSegmenter", () => {
  it("opens a persistent contract and returns only the first non-member Step", async () => {
    const calls: Array<{ operation: string; maxTokens?: number }> = [];
    const segmenter = new EpisodeSubproblemContractSegmenter({
      llm: contractLlm(calls, new Map([[0, 8], [8, 20]])),
      enableThinking: true
    });
    const result = await segmenter.segment(makePath(26));

    expect(result.segments.map((segment) => [segment.startStepIndex, segment.endStepIndex]))
      .toEqual([[0, 7], [8, 19], [20, 25]]);
    expect(result.segments.map((segment) => segment.contract.target))
      .toEqual(["target beginning at Step 0", "target beginning at Step 8", "target beginning at Step 20"]);
    expect(result.windows.map((window) => window.mode))
      .toEqual(["open_segment", "open_segment", "open_segment"]);
    expect(calls.every((call) => call.operation === `${SUBPROBLEM_CONTRACT_OPERATION}.open_segment`))
      .toBe(true);
    expect(calls.every((call) => call.maxTokens === 16_000)).toBe(true);
  });

  it("keeps one immutable contract across 15/5 continuation windows", async () => {
    const calls: Array<{ operation: string; maxTokens?: number }> = [];
    const result = await new EpisodeSubproblemContractSegmenter({
      llm: contractLlm(calls, new Map()),
      enableThinking: true
    }).segment(makePath(54));

    expect(result.segments).toHaveLength(1);
    expect(result.segments[0]!.startStepIndex).toBe(0);
    expect(result.segments[0]!.endStepIndex).toBe(53);
    expect(result.windows.map((window) => [
      window.mode,
      window.contextOnlyStepIndices.length,
      window.candidateStepIndices.length
    ])).toEqual([
      ["open_segment", 0, 15],
      ["continue_segment", 5, 10],
      ["continue_segment", 5, 10],
      ["continue_segment", 5, 10],
      ["continue_segment", 5, 9]
    ]);
    expect(result.windows.flatMap((window) => window.candidateStepIndices)).toEqual(range(0, 53));
    expect(calls).toHaveLength(5);
    expect(SUBPROBLEM_CONTRACT_PROMPT).toContain("persistent reusable subproblem contract");
    expect(SUBPROBLEM_CONTRACT_PROMPT).toContain("incorrect edit");
    expect(SUBPROBLEM_CONTRACT_PROMPT).toContain("currentContract is immutable");
    expect(SUBPROBLEM_CONTRACT_MAX_TOKENS).toBe(16_000);
  });
});

function contractLlm(
  calls: Array<{ operation: string; maxTokens?: number }>,
  boundaryBySegmentStart: ReadonlyMap<number, number>
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/subproblem-contract-test",
      model: "subproblem-contract-test"
    },
    isConfigured() { return true; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      calls.push({ operation: options.operation, maxTokens: options.maxTokens });
      const payload = JSON.parse(messages.at(-1)!.content) as {
        mode: "open_segment" | "continue_segment";
        window: { candidateStepIndices: number[] };
      };
      const firstCandidate = payload.window.candidateStepIndices[0]!;
      if (payload.mode === "open_segment") {
        return {
          contract: {
            target: `target beginning at Step ${firstCandidate}`,
            desired_outcome: `outcome beginning at Step ${firstCandidate}`
          },
          first_non_member_step_index: boundaryBySegmentStart.get(firstCandidate) ?? null
        } as unknown as T;
      }
      return {
        contract: null,
        first_non_member_step_index: null
      } as unknown as T;
    },
    status() {
      return {
        provider: "host",
        model: "subproblem-contract-test",
        configured: true,
        remote: false
      };
    }
  };
}

function makePath(count: number): EpisodeProceduralPathV2 {
  const state = emptyObservedState();
  return {
    id: "episode_path_subproblem_contract",
    schemaVersion: EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION,
    episodeId: "episode-subproblem-contract",
    states: [state],
    steps: makeSteps(count),
    spans: [],
    segmentationDecisions: [],
    sourceSnapshotHash: "snapshot-subproblem-contract",
    pathHash: "path-subproblem-contract"
  };
}

function makeSteps(count: number): ExecutionStepV1[] {
  const state = emptyObservedState();
  return Array.from({ length: count }, (_, stepIndex) => ({
    id: `step-${stepIndex}`,
    schemaVersion: EXECUTION_STEP_SCHEMA_VERSION,
    episodeId: "episode-subproblem-contract",
    rawTurnId: "raw-subproblem-contract",
    turnIndex: 0,
    stepIndex,
    preStateId: state.id,
    action: {
      kind: "tool_action",
      type: stepIndex % 4 === 0 ? "inspect" : stepIndex % 4 === 1 ? "edit" :
        stepIndex % 4 === 2 ? "execute" : "verify",
      intent: `perform operation ${stepIndex}`,
      summary: `observe result ${stepIndex}`,
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
      algorithmVersion: "subproblem-contract-test",
      sourceSnapshotHash: "snapshot-subproblem-contract"
    }
  }));
}

function range(start: number, end: number): number[] {
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}
