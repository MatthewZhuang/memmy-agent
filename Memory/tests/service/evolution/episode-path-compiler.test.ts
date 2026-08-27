import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { DEFAULT_MEMMY_CONFIG } from "../../../src/config/index.js";
import type {
  LlmClient,
  LlmCompletionOptions,
  LlmMessage
} from "../../../src/model/types.js";
import type {
  EpisodeRecord,
  RawTurnRecord
} from "../../../src/storage/repositories.js";
import {
  EpisodePathCompiler,
  STEP_SEMANTICS_PROMPT
} from "../../../src/service/evolution/episode-path-compiler.js";
import {
  buildTrajectoryWindows,
  V15_WINDOW_SPECS
} from "../../../src/service/evolution/procedural-window-model.js";

const AT = "2026-08-27T00:00:00.000Z";

describe("EpisodePathCompiler", () => {
  it("flattens every Turn into one ordered Episode path and permits cross-Turn windows", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const compiler = new EpisodePathCompiler({
      config: DEFAULT_MEMMY_CONFIG,
      llm: semanticsLlm(calls)
    });
    const episode = episodeRecord(0.9);
    const rawTurns = [
      rawTurn("raw-turn-1", "turn-1", 0, 20),
      rawTurn("raw-turn-2", "turn-2", 1, 30),
      rawTurn("raw-turn-3", "turn-3", 2, 20)
    ];

    const path = await compiler.compile({
      episode,
      rawTurns,
      sourceSnapshotHash: "source-snapshot-v1"
    });

    expect(path).not.toBeNull();
    expect(path?.steps).toHaveLength(70);
    expect(path?.steps.map((step) => step.stepIndex)).toEqual(
      Array.from({ length: 70 }, (_, index) => index)
    );
    expect(path?.steps.slice(0, 20).every((step) => step.turnIndex === 0)).toBe(true);
    expect(path?.steps.slice(20, 50).every((step) => step.turnIndex === 1)).toBe(true);
    expect(path?.steps.slice(50).every((step) => step.turnIndex === 2)).toBe(true);
    expect(path?.turnTransitions).toMatchObject([
      { rawTurnId: "raw-turn-1", afterStepIndex: 0 },
      { rawTurnId: "raw-turn-2", beforeStepIndex: 19, afterStepIndex: 20 },
      { rawTurnId: "raw-turn-3", beforeStepIndex: 49, afterStepIndex: 50 }
    ]);

    const windows = buildTrajectoryWindows([path!], V15_WINDOW_SPECS);
    expect(windows.filter((window) => window.scale === 5)).toHaveLength(34);
    expect(windows.filter((window) => window.scale === 10)).toHaveLength(13);
    expect(windows).toContainEqual(expect.objectContaining({
      scale: 5,
      startStepIndex: 18,
      endStepIndex: 22,
      steps: expect.arrayContaining([
        expect.objectContaining({ rawTurnId: "raw-turn-1" }),
        expect.objectContaining({ rawTurnId: "raw-turn-2" })
      ])
    }));
    expect(windows).toContainEqual(expect.objectContaining({
      scale: 10,
      startStepIndex: 45,
      endStepIndex: 54,
      steps: expect.arrayContaining([
        expect.objectContaining({ rawTurnId: "raw-turn-2" }),
        expect.objectContaining({ rawTurnId: "raw-turn-3" })
      ])
    }));

    expect(calls).toHaveLength(4);
    expect(calls.every((call) => call.options.operation === "procedural.step_semantics.v3.window"))
      .toBe(true);
    expect(calls.every((call) => call.options.maxTokens === 16_000)).toBe(true);
    expect(calls.every((call) => call.options.thinkingMode === "disabled")).toBe(true);
    const secondTurnPayload = JSON.parse(calls[1]?.messages[1]?.content ?? "{}") as {
      previousTurns?: unknown[];
      currentTurn?: { rawTurnId?: string };
      precedingStepContext?: unknown[];
    };
    expect(secondTurnPayload.previousTurns).toHaveLength(1);
    expect(secondTurnPayload.currentTurn?.rawTurnId).toBe("raw-turn-2");
    expect(secondTurnPayload.precedingStepContext).toHaveLength(5);
  });

  it("keeps execution identity stable when only the Episode reward changes", async () => {
    const compiler = new EpisodePathCompiler({
      config: DEFAULT_MEMMY_CONFIG,
      llm: semanticsLlm([])
    });
    const rawTurns = [rawTurn("raw-turn-1", "turn-1", 0, 5)];

    const positive = await compiler.compile({
      episode: episodeRecord(0.8, ["raw-turn-1"]),
      rawTurns,
      sourceSnapshotHash: "same-source"
    });
    const negative = await compiler.compile({
      episode: episodeRecord(-0.6, ["raw-turn-1"]),
      rawTurns,
      sourceSnapshotHash: "same-source"
    });

    expect(positive?.id).toBe(negative?.id);
    expect(positive?.pathHash).toBe(negative?.pathHash);
    expect(positive?.terminalReward).toBe(0.8);
    expect(negative?.terminalReward).toBe(-0.6);
    expect(buildTrajectoryWindows([positive!])[0]?.evidenceRole).toBe("support");
    expect(buildTrajectoryWindows([negative!])[0]?.evidenceRole).toBe("counterexample");
  });

  it("uses the exact v15 abstraction contract and compact candidate protocol", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const compiler = new EpisodePathCompiler({
      config: DEFAULT_MEMMY_CONFIG,
      llm: semanticsLlm(calls)
    });
    const turn = rawTurn("raw-turn-contract", "turn-contract", 0, 1);
    turn.toolCalls = [{
      id: "contract-call",
      name: "exec_command",
      input: {
        command: `python /private/project/generate_named_report.py --api-key sk-supersecret123456 ${"x".repeat(240)}`
      },
      thinkingBefore: "The private project is named Bluebird.",
      assistantTextBefore: "I will edit the named report."
    }];
    turn.toolResults = [{
      toolCallId: "contract-call",
      success: true,
      output: { message: `authorization: Bearer private-token-value ${"y".repeat(320)}` }
    }];

    await compiler.compile({
      episode: episodeRecord(0.9, [turn.id]),
      rawTurns: [turn],
      sourceSnapshotHash: "contract-source"
    });

    expect(STEP_SEMANTICS_PROMPT).toContain("reusable atomic operation");
    expect(createHash("sha256").update(STEP_SEMANTICS_PROMPT).digest("hex"))
      .toBe("ecb8a6eefa37b459fc3cd039fb3f3db4c35b988815aaa1a58cc0f6fd097703b5");
    expect(STEP_SEMANTICS_PROMPT).toContain("Abstract project names, filenames, exact paths");
    expect(STEP_SEMANTICS_PROMPT).toContain("success vs failure");
    const systemPrompt = calls[0]?.messages[0]?.content;
    expect(systemPrompt).toBe(STEP_SEMANTICS_PROMPT);
    const payload = JSON.parse(calls[0]?.messages[1]?.content ?? "{}") as {
      stepCandidates?: Array<Record<string, unknown>>;
    };
    const candidate = payload.stepCandidates?.[0] as Record<string, unknown>;
    expect(candidate).toMatchObject({
      kind: "tool_action",
      action: "execute",
      toolName: "exec_command",
      heuristicSuccess: true
    });
    expect(Object.keys(candidate).sort()).toEqual([
      "action",
      "candidateId",
      "eventIndex",
      "evidence",
      "heuristicSuccess",
      "kind",
      "rawTurnId",
      "sourceRefs",
      "toolName",
      "turnIndex"
    ]);
    expect(candidate).not.toHaveProperty("thinking_before");
    expect(candidate).not.toHaveProperty("assistant_text_before");
    expect(candidate).not.toHaveProperty("input");
    expect(candidate).not.toHaveProperty("output");
    expect(String(candidate.evidence).length).toBeLessThanOrEqual(420);
    const serialized = JSON.stringify(payload);
    expect(serialized).toContain("[redacted]");
    expect(serialized).not.toContain("sk-supersecret123456");
    expect(serialized).not.toContain("private-token-value");
    expect(serialized).not.toContain("Bluebird");
  });

  it("restores substantive response-generation Steps without changing the Lite schema", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const turn = rawTurn("raw-turn-response", "turn-response", 0, 1);
    turn.assistantText = "The verified report is ready for delivery.";
    const path = await new EpisodePathCompiler({
      config: DEFAULT_MEMMY_CONFIG,
      llm: semanticsLlm(calls, true)
    }).compile({
      episode: episodeRecord(0.9, [turn.id]),
      rawTurns: [turn],
      sourceSnapshotHash: "response-source"
    });

    expect(path?.steps).toHaveLength(2);
    expect(path?.steps[1]).toMatchObject({
      schemaVersion: "execution-step-lite.v1",
      kind: "response_generation",
      outcome: "success",
      rawTurnId: turn.id
    });
    const payload = JSON.parse(calls[0]?.messages[1]?.content ?? "{}") as {
      stepCandidates?: Array<Record<string, unknown>>;
    };
    expect(payload.stepCandidates?.map((candidate) => candidate.kind)).toEqual([
      "tool_action",
      "response_generation"
    ]);
  });

  it("recovers protocol-level failure outcomes and immediate retry/recovery links", async () => {
    const compiler = new EpisodePathCompiler({
      config: DEFAULT_MEMMY_CONFIG,
      llm: semanticsLlm([])
    });
    const turn = rawTurn("raw-turn-outcomes", "turn-outcomes", 0, 8);
    turn.toolCalls = [
      { id: "call-0", name: "exec_command", input: { command: "python build.py" } },
      { id: "call-1", name: "exec_command", input: { command: "python build.py" } },
      { id: "call-2", name: "test", input: { command: "pytest" } },
      { id: "call-3", name: "edit", input: { path: "build.py" } },
      { id: "call-4", name: "fetch", input: { url: "https://example.test" } },
      { id: "call-5", name: "query", input: { query: "records" } },
      { id: "call-6", name: "read", input: { path: "artifact.txt" } },
      { id: "call-7", name: "exec_command", input: { command: "true" } }
    ];
    turn.toolResults = [
      { toolCallId: "call-0", output: "Error: command exited with 1\nNameError: missing_symbol" },
      { toolCallId: "call-1", success: true, output: "build completed" },
      { toolCallId: "call-2", output: "Traceback (most recent call last):\nAssertionError" },
      { toolCallId: "call-3", success: true, output: "edit applied" },
      { toolCallId: "call-4", status: 403, output: "access denied" },
      { toolCallId: "call-5", output: JSON.stringify({ success: false, error: "provider unavailable" }) },
      { toolCallId: "call-6", output: "artifact contents" },
      { toolCallId: "call-7", output: "Exit code: 0" }
    ];

    const path = await compiler.compile({
      episode: episodeRecord(0.9, [turn.id]),
      rawTurns: [turn],
      sourceSnapshotHash: "outcome-source"
    });

    expect(path?.steps.map((step) => step.outcome)).toEqual([
      "failure",
      "success",
      "failure",
      "success",
      "failure",
      "failure",
      "success",
      "success"
    ]);
    expect(path?.steps[0]?.errorCode).toBe("EXIT_1");
    expect(path?.steps[1]?.retryOfStepId).toBe(path?.steps[0]?.id);
    expect(path?.steps[3]?.recoveryFromStepId).toBe(path?.steps[2]?.id);
    expect(path?.steps[4]?.errorCode).toBe("HTTP_403");
  });

  it("rejects compilation when the canonical semantics LLM is unavailable", async () => {
    const llm = semanticsLlm([]);
    llm.isConfigured = () => false;
    const compiler = new EpisodePathCompiler({
      config: DEFAULT_MEMMY_CONFIG,
      llm
    });
    await expect(compiler.compile({
      episode: episodeRecord(0.9, ["raw-turn-1"]),
      rawTurns: [rawTurn("raw-turn-1", "turn-1", 0, 1)],
      sourceSnapshotHash: "no-llm-source"
    })).rejects.toThrow("canonical Step semantics require a configured LLM");
  });

  it("rejects invalid model output after bounded repairs instead of silently falling back", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const compiler = new EpisodePathCompiler({
      config: DEFAULT_MEMMY_CONFIG,
      llm: invalidSemanticsLlm(calls)
    });

    await expect(compiler.compile({
      episode: episodeRecord(0.9, ["raw-turn-1"]),
      rawTurns: [rawTurn("raw-turn-1", "turn-1", 0, 1)],
      sourceSnapshotHash: "invalid-output-source"
    })).rejects.toThrow("procedural step LLM");
    expect(calls).toHaveLength(3);
    expect(calls.map((call) => call.options.operation)).toEqual([
      "procedural.step_semantics.v3.window",
      "procedural.step_semantics.v3.repair.1",
      "procedural.step_semantics.v3.repair.2"
    ]);
    expect(calls[1]?.messages.at(-1)?.content).toContain("strict schema-repair task");
  });
});

function semanticsLlm(
  calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }>,
  includeResponses = false
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      model: "path-test-model"
    },
    isConfigured: () => true,
    complete: async () => "{}",
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      calls.push({ messages, options });
      const payload = JSON.parse(messages.at(-1)?.content ?? "{}") as {
        stepCandidates?: Array<{
          candidateId: string;
          kind: "tool_action" | "response_generation";
          action: string;
          toolName?: string;
        }>;
      };
      return {
        steps: (payload.stepCandidates ?? []).map((candidate, index) => ({
          candidate_id: candidate.candidateId,
          include: candidate.kind === "tool_action" || includeResponses,
          intent: `${candidate.action} for the active subproblem ${index}`,
          summary: `${candidate.toolName ?? candidate.action} produced observable evidence`
        }))
      } as unknown as T;
    },
    status: () => ({
      provider: "host",
      model: "path-test-model",
      configured: true,
      remote: true
    })
  };
}

function invalidSemanticsLlm(
  calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }>
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      model: "invalid-semantics-test-model"
    },
    isConfigured: () => true,
    complete: async () => "{}",
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      calls.push({ messages, options });
      return { steps: [] } as unknown as T;
    },
    status: () => ({
      provider: "host",
      model: "invalid-semantics-test-model",
      configured: true,
      remote: true
    })
  };
}

function episodeRecord(
  reward: number,
  rawTurnIds = ["raw-turn-1", "raw-turn-2", "raw-turn-3"]
): EpisodeRecord {
  return {
    id: "episode-path-test",
    sessionId: "session-path-test",
    userId: "user-path-test",
    status: "closed",
    l1MemoryIds: [],
    rawTurnIds,
    feedbackIds: [],
    decisionRepairIds: [],
    l2PolicyIds: [],
    l3WorldModelIds: [],
    skillMemoryIds: [],
    turnCount: rawTurnIds.length,
    rTask: reward,
    rewardDetail: { source: "test" },
    pipelineStatus: "idle",
    meta: {},
    openedAt: AT,
    closedAt: AT,
    updatedAt: AT
  };
}

function rawTurn(
  id: string,
  turnId: string,
  turnIndex: number,
  toolCount: number
): RawTurnRecord {
  return {
    id,
    sessionId: "session-path-test",
    episodeId: "episode-path-test",
    turnId,
    userId: "user-path-test",
    userText: turnIndex === 0 ? "Create and validate the artifact" : "Continue the same task",
    assistantText: `Turn ${turnIndex} completed`,
    toolCalls: Array.from({ length: toolCount }, (_, index) => ({
      id: `${id}-tool-${index}`,
      name: index % 3 === 0 ? "inspect" : index % 3 === 1 ? "edit" : "test",
      input: { target: `artifact-${index}` }
    })),
    toolResults: Array.from({ length: toolCount }, (_, index) => ({
      toolCallId: `${id}-tool-${index}`,
      output: { ok: true, index }
    })),
    sourceMemoryIds: [],
    usage: {},
    status: "succeeded",
    createdAt: new Date(Date.parse(AT) + turnIndex * 1_000).toISOString()
  };
}
