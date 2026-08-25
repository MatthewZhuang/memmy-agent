import { describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage
} from "../../../src/index.js";
import type { RawTurnRecord } from "../../../src/storage/repositories.js";
import {
  EXECUTION_STEP_SEMANTICS_PROMPT,
  EpisodeProceduralReconstructor,
  TASK_CONTRACT_PROMPT,
  buildTurnStepCandidates
} from "../../../src/service/evolution/episode-procedural-reconstructor.js";

describe("EpisodeProceduralReconstructor / Step-only production path", () => {
  it("keeps the Step LLM contract minimal", () => {
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).toContain("intent");
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).toContain("summary");
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).not.toContain('"observations"');
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).not.toContain('"operations"');
    expect(TASK_CONTRACT_PROMPT).toContain("acceptance_criteria");
  });

  it("never invokes Span segmentation and persists only a compatibility envelope", async () => {
    const operations: string[] = [];
    const payloads: Record<string, unknown>[] = [];
    const llm = stepOnlyLlm(operations, payloads);
    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-step-only",
      rawTurns: [makeRawTurn({
        id: "raw-step-only",
        episodeId: "episode-step-only",
        userText: "Inspect and verify the build",
        toolCalls: [
          tool("read_file", "package.json", "configuration loaded"),
          tool("npm_test", "focused suite", "18 tests passed")
        ]
      })],
      terminalReward: 1
    });

    expect(path.steps).toHaveLength(2);
    expect(path.spans).toHaveLength(1);
    expect(path.spans[0]!.stepIds).toEqual(path.steps.map((step) => step.id));
    expect(path.spans[0]!.segmentation.reason).toContain("compatibility envelope");
    expect(operations).toEqual([
      "procedural.task_contract.v1",
      "procedural.step_semantics.v2.window"
    ]);
    expect(operations.some((operation) => operation.includes("span_"))).toBe(false);
    expect(payloads).toHaveLength(2);
  });

  it("bounds a 65-action turn into three Step-semantic calls without Span calls", async () => {
    const operations: string[] = [];
    const payloads: Record<string, unknown>[] = [];
    const path = await new EpisodeProceduralReconstructor({
      llm: stepOnlyLlm(operations, payloads)
    }).reconstruct({
      episodeId: "episode-long-step-only",
      rawTurns: [makeRawTurn({
        id: "raw-long-step-only",
        episodeId: "episode-long-step-only",
        toolCalls: Array.from({ length: 65 }, (_, index) =>
          tool("exec_command", `command-${index}`, `result-${index}`))
      })]
    });

    expect(path.steps).toHaveLength(65);
    const stepPayloads = payloads.filter((payload) => Array.isArray(payload.stepCandidates));
    expect(stepPayloads.map((payload) => (payload.stepCandidates as unknown[]).length))
      .toEqual([30, 30, 6]);
    expect(operations.filter((operation) => operation.startsWith("procedural.step_semantics")))
      .toHaveLength(3);
    expect(operations.some((operation) => operation.includes("span_"))).toBe(false);
  });

  it("treats protocol-level errors embedded in tool output as failed Step evidence", () => {
    const candidates = buildTurnStepCandidates(makeRawTurn({
      id: "raw-output-error",
      toolCalls: [
        tool("web_search", "query", "Error: search provider unavailable"),
        tool("web_fetch", "https://example.test", JSON.stringify({ status: 403, text: "access denied" }))
      ]
    }), 0);

    const toolCandidates = candidates.filter((candidate) => candidate.kind === "tool_action");
    expect(toolCandidates).toHaveLength(2);
    expect(toolCandidates.every((candidate) => !candidate.heuristicSuccess && candidate.errorCount === 1))
      .toBe(true);
  });
});

function stepOnlyLlm(
  operations: string[],
  payloads: Record<string, unknown>[]
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/step-only-test",
      model: "step-only-test"
    },
    isConfigured() { return true; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      operations.push(options.operation);
      if (options.operation.includes("span_")) {
        throw new Error(`unexpected Span LLM operation: ${options.operation}`);
      }
      const payload = requestPayload(messages);
      payloads.push(payload);
      if (options.operation.startsWith("procedural.task_contract.v1")) {
        const observations = payload.observations as Array<{ sourceId: string; userText?: string }>;
        return {
          contracts: observations.map((observation, index) => ({
            source_id: observation.sourceId,
            goal: index === 0 ? observation.userText ?? "complete the task" : null,
            constraints: [],
            acceptance_criteria: index === 0 ? ["the requested result is verified"] : []
          }))
        } as unknown as T;
      }
      if (options.operation.startsWith("procedural.step_semantics.v2")) {
        const candidates = payload.stepCandidates as Array<{
          candidateId: string;
          kind: "tool_action" | "response_generation";
          action: string;
          toolName?: string;
        }>;
        return {
          steps: candidates.map((candidate, index) => ({
            candidate_id: candidate.candidateId,
            include: candidate.kind === "tool_action",
            intent: `${candidate.action} for the active subproblem`,
            summary: `${candidate.toolName ?? candidate.action} produced observable evidence ${index}.`
          }))
        } as unknown as T;
      }
      throw new Error(`unexpected operation: ${options.operation}`);
    },
    status() {
      return { provider: "host", model: "step-only-test", configured: true, remote: false };
    }
  };
}

function requestPayload(messages: readonly LlmMessage[]): Record<string, unknown> {
  const message = [...messages].reverse().find((item) =>
    item.role === "user" && item.content.trimStart().startsWith("{"));
  return JSON.parse(message?.content ?? "{}") as Record<string, unknown>;
}

function tool(name: string, input: string, output: string): Record<string, unknown> {
  return { name, input: { value: input }, output };
}

function makeRawTurn(overrides: Partial<RawTurnRecord> & { id: string }): RawTurnRecord {
  return {
    sessionId: "session-test",
    episodeId: "episode-test",
    turnId: `turn-${overrides.id}`,
    userId: "user-test",
    userText: "complete the task",
    assistantText: "done",
    toolCalls: [],
    toolResults: [],
    sourceMemoryIds: [],
    usage: {},
    messagePayload: {},
    status: "completed",
    createdAt: "2026-01-01T00:00:00.000Z",
    ...overrides,
    id: overrides.id
  };
}
