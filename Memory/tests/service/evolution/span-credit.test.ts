import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  SpanCreditPipeline,
  type LlmClient,
  type LlmMessage
} from "../../../src/index.js";
import type { LlmCompletionOptions } from "../../../src/model/types.js";
import { Repositories } from "../../../src/storage/repositories.js";
import { persistProceduralPolicyEvidence } from "../../fixtures/procedural-policy-fixture.js";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const { cleanup, createTestService } = createMemoryServiceFixture();

afterEach(cleanup);

describe("Episode SpanCredit scoring", () => {
  it("adds a 3000-token reasoning reserve when thinking is enabled", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-thinking-budget");
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, { quality: 0.5 }),
      enableThinking: true
    });

    await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(calls[0]).toMatchObject({
      operation: "span_credit.score.v1",
      thinkingMode: "enabled",
      maxTokens: 4_400
    });
  });

  it("repairs invalid structured output and derives conserved code-owned credit", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-success");
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, { failFirst: true, quality: -0.25 })
    });

    const result = await pipeline.scoreEpisode({
      episodeId: evidence.episodeId,
      at: "2026-08-20T00:01:00.000Z"
    });

    expect(calls.map((call) => call.operation)).toEqual([
      "span_credit.score.v1",
      "span_credit.score.v1.repair.1"
    ]);
    expect(result.record).toMatchObject({
      status: "active",
      episodeId: evidence.episodeId,
      goalAchievement: 1,
      scorerModel: "span-credit-test"
    });
    expect(result.record.run.statePotentials.map((potential) => potential.progress)).toEqual([0, 1]);
    expect(result.credits).toEqual([
      expect.objectContaining({
        occurrenceId: evidence.occurrence.id,
        goalCredit: 1,
        processQuality: -0.25,
        confidence: 0.98,
        creditScore: 0.98,
        evidenceRole: "support"
      })
    ]);
    expect(result.credits.reduce((sum, credit) => sum + credit.goalCredit, 0)).toBe(1);

    const replay = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });
    expect(replay.created).toBe(false);
    expect(calls).toHaveLength(2);
  });

  it("classifies a grounded blocked Span as counterexample evidence", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-blocked", {
      terminationStatus: "blocked"
    });
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, { quality: -0.8 })
    });

    const result = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(result.record.goalAchievement).toBe(0);
    expect(result.credits).toEqual([
      expect.objectContaining({
        goalCredit: 0,
        processQuality: -0.8,
        creditScore: 0,
        evidenceRole: "counterexample"
      })
    ]);
  });

  it("does not promote a successful but severely wasteful procedure as support evidence", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-wasteful-success");
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, { quality: -0.9 })
    });

    const result = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(result.credits).toEqual([
      expect.objectContaining({
        goalCredit: 1,
        processQuality: -0.9,
        creditScore: 0.98,
        evidenceRole: "counterexample"
      })
    ]);
  });
});

interface CapturedLlmCall {
  messages: LlmMessage[];
  operation: string;
  thinkingMode?: LlmCompletionOptions["thinkingMode"];
  maxTokens?: number;
}

function spanCreditLlm(
  calls: CapturedLlmCall[],
  input: { failFirst?: boolean; quality: number }
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/span-credit-test",
      model: "span-credit-test"
    },
    isConfigured() {
      return true;
    },
    async complete() {
      return "{}";
    },
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      calls.push({
        messages,
        operation: options.operation,
        thinkingMode: options.thinkingMode,
        maxTokens: options.maxTokens
      });
      const payloadMessage = messages.find((message) =>
        message.role === "user" && message.content.includes("terminal_goal_achievement"));
      const payload = JSON.parse(payloadMessage?.content ?? "{}") as {
        spans?: Array<{ occurrence_id: string }>;
      };
      const occurrenceId = payload.spans?.[0]?.occurrence_id ?? "missing-occurrence";
      if (input.failFirst && calls.length === 1) {
        return {
          state_progress: [],
          span_process_quality: [{
            occurrence_id: occurrenceId,
            quality: input.quality,
            evidence_refs: ["invented-evidence-ref"],
            reason: "invalid grounding"
          }]
        } as unknown as T;
      }
      return {
        state_progress: [],
        span_process_quality: [{
          occurrence_id: occurrenceId,
          quality: input.quality,
          evidence_refs: [occurrenceId],
          reason: "The Span reached its recorded exit condition with the observed recovery cost."
        }]
      } as unknown as T;
    },
    status() {
      return {
        provider: "host",
        model: "span-credit-test",
        configured: true,
        remote: true
      };
    }
  };
}
