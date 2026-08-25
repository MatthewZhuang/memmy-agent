import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  SpanCreditPipeline,
  buildEpisodeProceduralPath,
  type LlmClient,
  type LlmMessage,
  type SpanCreditAttributionType
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
      skillLlm: spanCreditLlm(calls),
      enableThinking: true
    });

    await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(calls[0]).toMatchObject({
      operation: "span_credit.score.v2",
      thinkingMode: "enabled",
      maxTokens: 4_400,
      maxLengthRetries: 2
    });
  });

  it("repairs invalid structured output and conserves the Episode reward budget", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-success");
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, { failFirst: true })
    });

    const result = await pipeline.scoreEpisode({
      episodeId: evidence.episodeId,
      at: "2026-08-20T00:01:00.000Z"
    });

    expect(calls.map((call) => call.operation)).toEqual([
      "span_credit.score.v2",
      "span_credit.score.v2.repair.1"
    ]);
    expect(result.record).toMatchObject({
      status: "active",
      episodeId: evidence.episodeId,
      episodeReward: 1,
      scorerModel: "span-credit-test"
    });
    expect(result.record.run.inputCompaction.mode).toBe("full");
    expect(result.credits).toEqual([
      expect.objectContaining({
        occurrenceId: evidence.occurrence.id,
        rewardCredit: 1,
        attributionType: "helpful",
        confidence: 0.9,
        evidenceRole: "support"
      })
    ]);
    expect(result.credits.reduce((sum, credit) => sum + credit.rewardCredit, 0)).toBe(1);

    const replay = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });
    expect(replay.created).toBe(false);
    expect(calls).toHaveLength(2);
  });

  it("keeps an externally blocked Span out of counterexample evidence", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-blocked", {
      terminationStatus: "blocked"
    });
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, { attributionType: "externally_blocked" })
    });

    const result = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(result.record.episodeReward).toBe(0);
    expect(result.credits).toEqual([
      expect.objectContaining({
        rewardCredit: 0,
        attributionType: "externally_blocked",
        evidenceRole: "neutral"
      })
    ]);
  });

  it("assigns signed credits to all Final Spans in one joint call", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-joint-allocation", {
      rTask: 0.6
    });
    const path = splitFixturePath(evidence.path);
    repos.proceduralPaths.save({
      path,
      namespaceId: evidence.occurrence.namespaceId,
      createdAt: "2026-08-20T00:00:30.000Z"
    });
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, {
        allocations: [
          { rewardCredit: -0.2, attributionType: "harmful" },
          { rewardCredit: 0.8, attributionType: "helpful" }
        ]
      })
    });

    const result = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(calls).toHaveLength(1);
    expect(result.credits.map((credit) => ({
      rewardCredit: credit.rewardCredit,
      attributionType: credit.attributionType,
      evidenceRole: credit.evidenceRole
    }))).toEqual([
      { rewardCredit: -0.2, attributionType: "harmful", evidenceRole: "counterexample" },
      { rewardCredit: 0.8, attributionType: "helpful", evidenceRole: "support" }
    ]);
    expect(result.credits.reduce((sum, credit) => sum + credit.rewardCredit, 0)).toBeCloseTo(0.6);
  });

  it("does not turn low-confidence attribution into Policy evidence", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-uncertain");
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls, { attributionType: "uncertain", confidence: 0.4 })
    });

    const result = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(result.credits).toEqual([
      expect.objectContaining({
        rewardCredit: 1,
        attributionType: "uncertain",
        confidence: 0.4,
        evidenceRole: "uncertain"
      })
    ]);
  });

  it("compacts oversized evidence without splitting the Episode into multiple scoring calls", async () => {
    const calls: CapturedLlmCall[] = [];
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "credit-input-compaction");
    const pipeline = new SpanCreditPipeline({
      repos,
      skillLlm: spanCreditLlm(calls),
      maxInputChars: 2_200
    });

    const result = await pipeline.scoreEpisode({ episodeId: evidence.episodeId });

    expect(calls).toHaveLength(1);
    expect(result.record.run.inputCompaction.mode).not.toBe("full");
    expect(result.record.run.inputCompaction.finalChars).toBeLessThanOrEqual(2_200);
    const payload = JSON.parse(calls[0]!.messages[1]!.content) as {
      input_compaction_mode: string;
      spans: Array<{ occurrence_id: string }>;
    };
    expect(payload.input_compaction_mode).toBe(result.record.run.inputCompaction.mode);
    expect(payload.spans.map((span) => span.occurrence_id)).toEqual([evidence.occurrence.id]);
  });
});

interface CapturedLlmCall {
  messages: LlmMessage[];
  operation: string;
  thinkingMode?: LlmCompletionOptions["thinkingMode"];
  maxTokens?: number;
  maxLengthRetries?: number;
}

function spanCreditLlm(
  calls: CapturedLlmCall[],
  input: {
    failFirst?: boolean;
    attributionType?: SpanCreditAttributionType;
    confidence?: number;
    allocations?: Array<{
      rewardCredit: number;
      attributionType: SpanCreditAttributionType;
    }>;
  } = {}
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
        maxTokens: options.maxTokens,
        maxLengthRetries: options.maxLengthRetries
      });
      const payloadMessage = messages.find((message) =>
        message.role === "user" && message.content.includes("episode_reward"));
      const payload = JSON.parse(payloadMessage?.content ?? "{}") as {
        episode_reward?: number;
        spans?: Array<{ occurrence_id: string }>;
      };
      const spans = payload.spans ?? [];
      const occurrenceId = spans[0]?.occurrence_id ?? "missing-occurrence";
      const episodeReward = payload.episode_reward ?? 0;
      if (input.failFirst && calls.length === 1) {
        const invalidCredit = episodeReward === 0 ? 0.2 : episodeReward - 0.2;
        return {
          span_credits: [{
            occurrence_id: occurrenceId,
            reward_credit: invalidCredit,
            attribution_type: invalidCredit > 0 ? "helpful" : "harmful",
            confidence: input.confidence ?? 0.9,
            evidence_refs: [occurrenceId],
            reason: "invalid reward conservation"
          }]
        } as unknown as T;
      }
      const allocations = input.allocations ?? [{
        rewardCredit: episodeReward,
        attributionType: input.attributionType ?? (episodeReward > 0 ? "helpful" : "neutral")
      }];
      return {
        span_credits: spans.map((span, index) => ({
          occurrence_id: span.occurrence_id,
          reward_credit: allocations[index]?.rewardCredit ?? 0,
          attribution_type: allocations[index]?.attributionType ?? "neutral",
          confidence: input.confidence ?? 0.9,
          evidence_refs: [span.occurrence_id],
          reason: "The recorded state transition causally explains its assigned share of the Episode reward."
        }))
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

function splitFixturePath(
  path: ReturnType<typeof persistProceduralPolicyEvidence>["path"]
) {
  const [inspect, repair] = path.steps;
  const [initial, diagnosed, terminal] = path.states;
  const baseSpan = path.spans[0]!;
  const baseDecision = path.segmentationDecisions[0]!;
  if (!inspect || !repair || !initial || !diagnosed || !terminal) {
    throw new Error("joint SpanCredit fixture is incomplete");
  }
  const decisions = [{
    ...baseDecision,
    spanIndex: 0,
    stepIds: [inspect.id],
    localGoal: "Diagnose the dependency conflict",
    capabilityGoal: "Diagnose a reproducible dependency failure",
    entryCondition: "The focused build failure has not been explained",
    exitCondition: "The conflicting dependency constraint is identified",
    terminationStatus: "success" as const,
    evidenceRefs: [...inspect.outcome.evidenceRefs],
    reason: "The diagnostic subproblem reaches a stable root-cause state"
  }, {
    ...baseDecision,
    spanIndex: 1,
    stepIds: [repair.id],
    localGoal: "Repair and verify the dependency constraint",
    capabilityGoal: "Repair and verify a diagnosed dependency failure",
    entryCondition: "The conflicting dependency constraint is identified",
    exitCondition: "The corrected constraint passes the focused build",
    terminationStatus: "success" as const,
    evidenceRefs: [...repair.outcome.evidenceRefs],
    reason: "The repair subproblem exits after focused verification"
  }];
  const spans = [{
    ...baseSpan,
    id: `${baseSpan.id}-diagnose`,
    spanIndex: 0,
    localGoal: decisions[0]!.localGoal,
    capabilityGoal: decisions[0]!.capabilityGoal,
    entryCondition: decisions[0]!.entryCondition,
    stepIds: [inspect.id],
    preStateId: initial.id,
    postStateId: diagnosed.id,
    termination: {
      status: "success" as const,
      exitCondition: decisions[0]!.exitCondition,
      evidenceRefs: decisions[0]!.evidenceRefs
    },
    cost: { steps: 1, toolCalls: 1, retryCount: 0, recoveryCount: 0, errorCount: 1 },
    segmentation: { reason: decisions[0]!.reason, confidence: 0.95 }
  }, {
    ...baseSpan,
    id: `${baseSpan.id}-repair`,
    spanIndex: 1,
    localGoal: decisions[1]!.localGoal,
    capabilityGoal: decisions[1]!.capabilityGoal,
    entryCondition: decisions[1]!.entryCondition,
    stepIds: [repair.id],
    preStateId: diagnosed.id,
    postStateId: terminal.id,
    termination: {
      status: "success" as const,
      exitCondition: decisions[1]!.exitCondition,
      evidenceRefs: decisions[1]!.evidenceRefs
    },
    cost: { steps: 1, toolCalls: 1, retryCount: 0, recoveryCount: 1, errorCount: 0 },
    segmentation: { reason: decisions[1]!.reason, confidence: 0.95 }
  }];
  return buildEpisodeProceduralPath({
    episodeId: path.episodeId,
    states: path.states,
    steps: path.steps,
    spans,
    segmentationDecisions: decisions,
    sourceSnapshotHash: path.sourceSnapshotHash,
    terminalReward: 0.6
  });
}
