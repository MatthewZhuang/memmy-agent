import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage
} from "../../../src/index.js";
import {
  createMemoryServiceFixture,
  runWorkerRounds
} from "../../fixtures/memory-service-fixture.js";

const {
  cleanup,
  createTestService
} = createMemoryServiceFixture();

afterEach(cleanup);

function createRecordingLlm(
  calls: Array<{ options: LlmCompletionOptions }>
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.summary,
      provider: "host",
      endpoint: "http://127.0.0.1/span-big-turn",
      model: "span-big-turn"
    },
    isConfigured() {
      return true;
    },
    async complete() {
      return "{}";
    },
    async completeJson<T extends Record<string, unknown>>(
      _messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      calls.push({ options });
      if (options.operation === "capture.reflection.batch.v13") {
        return { scores: [] } as unknown as T;
      }
      if (options.operation === "capture.summarize") {
        return { l1: { summary: "completed complex tool turn" }, user: null } as unknown as T;
      }
      if (options.operation === "span.big_turn.v1") {
        throw new Error("span.big_turn should not run while SPAN_BIG_TURN_ENABLED is false");
      }
      return {} as T;
    },
    status() {
      return {
        provider: "host",
        model: "span-big-turn",
        configured: true,
        remote: true
      };
    }
  };
}

describe("MemoryService / evolution / span big turn", () => {
  it("does not enqueue or split spans while the pipeline is disabled", async () => {
    const calls: Array<{ options: LlmCompletionOptions }> = [];
    const llm = createRecordingLlm(calls);
    const { db, service } = createTestService({
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          reward: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.reward,
            llmScoring: false
          }
        }
      },
      llm,
      skillLlm: llm
    });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-disabled"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 11 }, (_, index) => ({
      id: `tool-${index}`,
      name: "task_step",
      input: { index }
    }));
    const completed = service.completeTurn("span-big-turn-disabled", {
      namespace,
      sessionId: session.sessionId,
      query: "修复项目构建失败并完成测试验证",
      answer: "已经定位依赖冲突，完成修复并通过构建与测试。",
      toolCalls,
      toolResults: toolCalls.map((call, index) => ({
        toolCallId: call.id,
        name: call.name,
        output: { ok: true, index }
      }))
    });
    service.closeSession(session.sessionId);
    await service.feedback({
      namespace,
      sessionId: session.sessionId,
      episodeId: completed.episodeId,
      l1MemoryId: completed.l1MemoryId,
      channel: "explicit",
      polarity: "positive",
      magnitude: 1,
      rationale: "复杂任务已经正确完成"
    });
    await runWorkerRounds(service, 8);

    expect(calls.some((call) => call.options.operation === "span.big_turn.v1")).toBe(false);
    expect(service.panelJobs({ userId: namespace.userId }).items.some(
      (job) => job.jobType === "span_big_turn"
    )).toBe(false);
    expect(service.panelItems({ namespace, layer: "L1" }).items.some(
      (item) => item.kind === "span"
    )).toBe(false);
    db.close();
  });
});
