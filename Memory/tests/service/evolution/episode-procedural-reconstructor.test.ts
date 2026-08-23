import { describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage
} from "../../../src/index.js";
import type { RawTurnRecord } from "../../../src/storage/repositories.js";
import {
  EpisodeProceduralReconstructor,
  PROCEDURAL_SPAN_RECONCILIATION_PROMPT,
  PROCEDURAL_SPAN_SEGMENTATION_PROMPT,
  buildTurnStepCandidates
} from "../../../src/service/evolution/episode-procedural-reconstructor.js";
import {
  validateExecutionStepContinuity,
  validateProceduralSpanCoverage
} from "../../../src/service/evolution/procedural-path-model.js";

describe("EpisodeProceduralReconstructor", () => {
  it("defines canonical local goals without a domain-specific schema example", () => {
    expect(PROCEDURAL_SPAN_SEGMENTATION_PROMPT).not.toContain("dependency conflict");
    expect(PROCEDURAL_SPAN_RECONCILIATION_PROMPT).not.toContain("dependency conflict");
    expect(PROCEDURAL_SPAN_SEGMENTATION_PROMPT).toContain(
      "canonical core objective of this Span"
    );
    expect(PROCEDURAL_SPAN_RECONCILIATION_PROMPT).toContain(
      "Treat every provisional localGoal as a proposal, not authoritative truth"
    );
  });

  it("treats protocol-level errors embedded in tool output as failed step evidence", () => {
    const rawTurn = makeRawTurn({
      id: "raw-output-error",
      toolCalls: [
        { id: "call-1", name: "web_search", input: { q: "query" } },
        { id: "call-2", name: "web_fetch", input: { url: "https://example.test" } }
      ],
      toolResults: [
        {
          toolCallId: "call-1",
          name: "web_search",
          output: "Error: search provider unavailable"
        },
        {
          toolCallId: "call-2",
          name: "web_fetch",
          output: JSON.stringify({ status: 403, text: "access denied" })
        }
      ]
    });

    const candidates = buildTurnStepCandidates(rawTurn, 0);

    expect(candidates[0]).toMatchObject({
      kind: "tool_action",
      heuristicSuccess: false,
      errorCount: 1
    });
    expect(candidates[0]?.evidence).toContain("Error: search provider unavailable");
    expect(candidates[1]).toMatchObject({
      kind: "tool_action",
      heuristicSuccess: false,
      errorCount: 1
    });
    expect(candidates[1]?.evidence).toContain("HTTP 403");
  });

  it("keeps diagnose-fix-fail-recover-verify as steps inside one single-turn span", async () => {
    const rawTurns = [makeRawTurn({
      id: "raw-recovery",
      episodeId: "episode-recovery",
      userText: "修复依赖冲突导致的测试失败",
      assistantText: "依赖冲突已经修复，测试通过。",
      toolCalls: [
        tool("read_file", "package.json", "dependency-a=1 dependency-b=2"),
        tool("apply_patch", "first fix", "updated package.json"),
        tool("npm_test", "target suite", "Error: incompatible peer dependency"),
        tool("read_file", "lockfile", "dependency-a requires dependency-b=3"),
        tool("apply_patch", "correct fix", "updated dependency-b to 3"),
        tool("npm_test", "target suite", "18 tests passed")
      ]
    })];
    const llm = createProceduralLlm((stepIds) => [stepIds]);

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-recovery",
      rawTurns,
      terminalReward: 1
    });
    const replayed = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-recovery",
      rawTurns,
      terminalReward: 1
    });

    expect(path.schemaVersion).toBe("episode-execution-path.v2");
    expect(path.steps).toHaveLength(6);
    expect(path.spans).toHaveLength(1);
    expect(path.spans[0]).toMatchObject({
      rawTurnIds: ["raw-recovery"],
      cost: { steps: 6, toolCalls: 6, errorCount: 1, retryCount: 0, recoveryCount: 1 },
      termination: { status: "success" }
    });
    expect(path.spans[0]?.stepIds).toEqual(path.steps.map((step) => step.id));
    expect(path.segmentationDecisions[0]).toMatchObject({
      stepIds: path.steps.map((step) => step.id),
      reason: "The grouped steps form one complete local lifecycle.",
      confidence: 0.95
    });
    expect(replayed.pathHash).toBe(path.pathHash);
    expect(path.steps[2]).toMatchObject({ outcome: { status: "failure" }, cost: { errorCount: 1 } });
    expect(path.steps[3]?.recoveryFromStepId).toBe(path.steps[2]?.id);
    expect(() => validateExecutionStepContinuity(path.steps)).not.toThrow();
    expect(() => validateProceduralSpanCoverage(path.steps, path.spans)).not.toThrow();
  });

  it("keeps one subproblem in one span across multiple conversation turns", async () => {
    const rawTurns = [
      makeRawTurn({
        id: "raw-multi-1",
        episodeId: "episode-multi-turn",
        createdAt: "2026-01-01T00:00:00.000Z",
        userText: "修复构建失败",
        assistantText: "我定位到 TypeScript 配置问题。",
        toolCalls: [tool("read_file", "tsconfig.json", "moduleResolution=classic")]
      }),
      makeRawTurn({
        id: "raw-multi-2",
        episodeId: "episode-multi-turn",
        createdAt: "2026-01-01T00:01:00.000Z",
        userText: "保持 NodeNext，不要降级 TypeScript",
        assistantText: "已按约束修复并验证。",
        toolCalls: [
          tool("apply_patch", "tsconfig", "moduleResolution=NodeNext"),
          tool("npm_test", "typecheck", "typecheck passed")
        ]
      })
    ];
    const llm = createProceduralLlm((stepIds) => [stepIds]);

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-multi-turn",
      rawTurns
    });

    expect(path.steps).toHaveLength(3);
    expect(path.spans).toHaveLength(1);
    expect(path.spans[0]?.rawTurnIds).toEqual(["raw-multi-1", "raw-multi-2"]);
    expect(path.steps[0]?.externalObservationDelta).toContainEqual(expect.objectContaining({
      op: "constraint.upsert",
      subject: "user_constraint_raw-multi-2"
    }));
    expect(path.steps[0]?.postStateId).toBe(path.steps[1]?.preStateId);
  });

  it("splits materially different subproblems inside the same conversation turn", async () => {
    const rawTurns = [makeRawTurn({
      id: "raw-two-subproblems",
      episodeId: "episode-two-subproblems",
      userText: "修复测试并构建发布包",
      assistantText: "测试已修复，发布包也已构建验证。",
      toolCalls: [
        tool("read_file", "test.log", "dependency mismatch"),
        tool("apply_patch", "dependency fix", "updated"),
        tool("npm_test", "tests", "tests passed"),
        tool("npm_run", "build", "dist/app.tgz created"),
        tool("verify_artifact", "dist/app.tgz", "checksum verified")
      ]
    })];
    const llm = createProceduralLlm((stepIds) => [stepIds.slice(0, 3), stepIds.slice(3)]);

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-two-subproblems",
      rawTurns
    });

    expect(path.steps).toHaveLength(5);
    expect(path.spans).toHaveLength(2);
    expect(path.spans.map((span) => span.stepIds.length)).toEqual([3, 2]);
    expect(path.spans[0]?.postStateId).toBe(path.spans[1]?.preStateId);
    expect(path.spans.every((span) => span.rawTurnIds.length === 1)).toBe(true);
  });

  it("uses targeted then full-regeneration repair for an operation/op schema mismatch", async () => {
    const baseLlm = createProceduralLlm((stepIds) => [stepIds]);
    const repairPrompts: string[] = [];
    let stepSemanticCalls = 0;
    const llm: LlmClient = {
      ...baseLlm,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        const valid = await baseLlm.completeJson<T>(messages, options);
        if (!options.operation.startsWith("procedural.step_semantics.v1")) return valid;
        stepSemanticCalls += 1;
        if (options.operation.includes(".repair.")) {
          repairPrompts.push(messages.at(-1)?.content ?? "");
        }
        if (stepSemanticCalls <= 2) {
          return JSON.parse(JSON.stringify(valid).replaceAll('"op":', '"operation":')) as T;
        }
        return valid;
      }
    };
    const rawTurns = [makeRawTurn({
      id: "raw-operation-alias-repair",
      episodeId: "episode-operation-alias-repair",
      userText: "修复测试",
      assistantText: "测试已修复。",
      toolCalls: [tool("npm_test", "tests", "tests passed")]
    })];

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-operation-alias-repair",
      rawTurns
    });

    expect(stepSemanticCalls).toBe(3);
    expect(repairPrompts).toHaveLength(2);
    expect(repairPrompts[0]).toContain('"operation"');
    expect(repairPrompts[0]).toContain('exact field name "op"');
    expect(repairPrompts[0]).toContain("INVALID:");
    expect(repairPrompts[1]).toContain("Discard the previous JSON");
    expect(repairPrompts[1]).toContain('literal key "op"');
    expect(path.steps).toHaveLength(1);
    expect(path.steps[0]?.actionEffectDelta.every((operation) => Boolean(operation.op))).toBe(true);
  });

  it("bounds a long turn into 30-step windows with five preceding context steps", async () => {
    const baseLlm = createProceduralLlm((stepIds) => [stepIds]);
    const stepPayloads: Array<Record<string, unknown>> = [];
    const segmentationPayloads: Array<Record<string, unknown>> = [];
    const reconciliationPayloads: Array<Record<string, unknown>> = [];
    const llm: LlmClient = {
      ...baseLlm,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        const payload = llmRequestPayload(messages);
        if (options.operation === "procedural.step_semantics.v1.window") stepPayloads.push(payload);
        if (options.operation === "procedural.span_segmentation.v3.window") segmentationPayloads.push(payload);
        if (options.operation === "procedural.span_reconciliation.v2") reconciliationPayloads.push(payload);
        return baseLlm.completeJson<T>(messages, options);
      }
    };
    const toolCalls = Array.from({ length: 65 }, (_, index) => tool(
      "inspect",
      `target-${index}`,
      index === 0 ? "x".repeat(5_000) : `observed-${index}`
    ));
    const rawTurns = [makeRawTurn({
      id: "raw-long-windowed-turn",
      episodeId: "episode-long-windowed-turn",
      userText: "完成一个包含大量检查步骤的长任务",
      assistantText: "所有检查均已完成。",
      toolCalls
    })];

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-long-windowed-turn",
      rawTurns
    });

    expect(path.steps).toHaveLength(65);
    expect(path.spans).toHaveLength(1);
    expect(stepPayloads.map((payload) => (payload.stepCandidates as unknown[]).length)).toEqual([30, 30, 6]);
    expect(stepPayloads.map((payload) => (payload.precedingStepContext as unknown[]).length)).toEqual([0, 5, 5]);
    const transmittedEvidence = stepPayloads.flatMap((payload) => payload.stepCandidates as Array<{ evidence?: string }>);
    expect(Math.max(...transmittedEvidence.map((candidate) => candidate.evidence?.length ?? 0))).toBeLessThanOrEqual(900);
    expect(segmentationPayloads.map((payload) => (payload.steps as unknown[]).length)).toEqual([30, 30, 5]);
    expect(segmentationPayloads.map((payload) => (payload.precedingStepContext as unknown[]).length)).toEqual([0, 5, 5]);
    expect(segmentationPayloads[0]?.openSpan).toBeNull();
    expect(segmentationPayloads[1]?.openSpan).not.toBeNull();
    expect(reconciliationPayloads).toHaveLength(1);
    expect(reconciliationPayloads[0]?.provisionalSpans).toHaveLength(3);
    expect(reconciliationPayloads[0]?.provisionalSpans).toEqual(expect.arrayContaining([
      expect.objectContaining({
        procedureIntents: expect.arrayContaining([
          expect.stringContaining("active subproblem")
        ])
      })
    ]));
  });

  it("limits each turn window to the previous three turn summaries and carries openSpan", async () => {
    const baseLlm = createProceduralLlm((stepIds) => [stepIds]);
    const segmentationPayloads: Array<Record<string, unknown>> = [];
    const llm: LlmClient = {
      ...baseLlm,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        if (options.operation === "procedural.span_segmentation.v3.window") {
          segmentationPayloads.push(llmRequestPayload(messages));
        }
        return baseLlm.completeJson<T>(messages, options);
      }
    };
    const rawTurns = Array.from({ length: 5 }, (_, index) => makeRawTurn({
      id: `raw-context-${index}`,
      episodeId: "episode-bounded-turn-context",
      createdAt: `2026-01-01T00:0${index}:00.000Z`,
      userText: `继续处理同一问题 ${index}`,
      assistantText: `完成阶段 ${index}`,
      toolCalls: [tool("inspect", `target-${index}`, `observed-${index}`)]
    }));

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-bounded-turn-context",
      rawTurns
    });

    expect(path.steps).toHaveLength(5);
    expect(path.spans).toHaveLength(1);
    expect(segmentationPayloads.map((payload) => (payload.previousTurns as unknown[]).length)).toEqual([0, 1, 2, 3, 3]);
    expect(segmentationPayloads.map((payload) => payload.openSpan === null)).toEqual([true, false, false, false, false]);
  });

  it("rejects a partition that omits or reorders execution steps", async () => {
    const rawTurns = [makeRawTurn({
      id: "raw-invalid-partition",
      episodeId: "episode-invalid-partition",
      toolCalls: [
        tool("read_file", "a", "a"),
        tool("apply_patch", "b", "b")
      ]
    })];
    const llm = createProceduralLlm((stepIds) => [[stepIds[1]!, stepIds[0]!]]);

    await expect(new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-invalid-partition",
      rawTurns
    })).rejects.toThrow("cover every step exactly once and in order");
  });
});

function createProceduralLlm(
  partition: (stepIds: string[]) => string[][]
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/procedural-test",
      model: "procedural-test"
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
      const payloadMessage = [...messages].reverse().find(
        (message) => message.role === "user" && message.content.trimStart().startsWith("{")
      );
      const payload = JSON.parse(payloadMessage?.content ?? "{}") as Record<string, unknown>;
      if (options.operation.startsWith("procedural.step_semantics.v1")) {
        const observations = payload.observations as Array<{
          sourceId: string;
          rawTurnId: string;
          userText?: string;
        }>;
        const candidates = payload.stepCandidates as Array<{
          candidateId: string;
          rawTurnId: string;
          kind: "tool_action" | "response_generation";
          action: string;
          toolName?: string;
          heuristicSuccess: boolean;
          sourceRefs: string[];
        }>;
        return {
          observations: observations.map((observation) => ({
            source_id: observation.sourceId,
            operations: Array.isArray(payload.previousTurns) && payload.previousTurns.length > 0
              ? [{
                  op: "constraint.upsert",
                  subject: `user_constraint_${observation.rawTurnId}`,
                  value: observation.userText ?? "follow-up constraint",
                  source_refs: [observation.sourceId]
                }]
              : [{
                  op: "goal.set",
                  subject: "task_goal",
                  value: observation.userText ?? "complete the task",
                  source_refs: [observation.sourceId]
                }]
          })),
          steps: candidates.map((candidate, candidateIndex) => {
            const include = candidate.kind === "tool_action";
            const evidenceRef = candidate.sourceRefs.at(-1)!;
            const previous = candidates[candidateIndex - 1];
            return {
              candidate_id: candidate.candidateId,
              include,
              intent: include ? `${candidate.action} for the active subproblem` : "report completion",
              summary: include ? `${candidate.toolName ?? candidate.action} produced observable evidence.` : "Social report only.",
              outcome_status: candidate.heuristicSuccess ? "success" : "failure",
              evidence_refs: [evidenceRef],
              retry_of_candidate_id: null,
              recovery_from_candidate_id: include && previous?.heuristicSuccess === false
                ? previous.candidateId
                : null,
              operations: include
                ? [{
                    op: candidate.heuristicSuccess ? "fact.upsert" : "issue.upsert",
                    subject: `step_effect_${candidate.candidateId}`,
                    value: candidate.heuristicSuccess ? "observed" : "failed",
                    source_refs: [evidenceRef]
                  }]
                : []
            };
          })
        } as unknown as T;
      }
      if (options.operation.startsWith("procedural.span_segmentation.")) {
        const steps = payload.steps as Array<{ stepId: string }>;
        const window = payload.window as {
          hasFollowingWindow?: boolean;
          hasFollowingTurn?: boolean;
        };
        const groups = partition(steps.map((step) => step.stepId));
        return {
          spans: groups.map((stepIds, index) => ({
            step_ids: stepIds,
            continues_previous: index === 0 && payload.openSpan !== null,
            open_at_window_end: index === groups.length - 1 &&
              Boolean(window.hasFollowingWindow || window.hasFollowingTurn),
            local_goal: `solve local subproblem ${index + 1}`,
            entry_condition: `subproblem ${index + 1} is unresolved`,
            exit_condition: `subproblem ${index + 1} is resolved and verified`,
            termination_status: "success",
            evidence_refs: [stepIds.at(-1)],
            reason: "The grouped steps form one complete local lifecycle.",
            confidence: 0.95
          }))
        } as unknown as T;
      }
      if (options.operation.startsWith("procedural.span_reconciliation.v2")) {
        const proposals = payload.provisionalSpans as Array<{
          provisionalSpanId: string;
          stepIds: string[];
        }>;
        const desiredGroups = partition(proposals.flatMap((proposal) => proposal.stepIds));
        let proposalIndex = 0;
        return {
          spans: desiredGroups.map((stepIds, index) => {
            const proposalIds: string[] = [];
            let coveredSteps = 0;
            while (proposalIndex < proposals.length && coveredSteps < stepIds.length) {
              const proposal = proposals[proposalIndex++]!;
              proposalIds.push(proposal.provisionalSpanId);
              coveredSteps += proposal.stepIds.length;
            }
            return {
              provisional_span_ids: proposalIds,
              local_goal: `solve reconciled subproblem ${index + 1}`,
              entry_condition: `reconciled subproblem ${index + 1} is unresolved`,
              exit_condition: `reconciled subproblem ${index + 1} is resolved and verified`,
              termination_status: "success",
              evidence_refs: [stepIds.at(-1)],
              reason: "The provisional spans form one complete episode-level lifecycle.",
              confidence: 0.95
            };
          })
        } as unknown as T;
      }
      throw new Error(`unexpected operation: ${options.operation}`);
    },
    status() {
      return {
        provider: "host",
        model: "procedural-test",
        configured: true,
        remote: false
      };
    }
  };
}

function llmRequestPayload(messages: LlmMessage[]): Record<string, unknown> {
  const payloadMessage = [...messages].reverse().find(
    (message) => message.role === "user" && message.content.trimStart().startsWith("{")
  );
  return JSON.parse(payloadMessage?.content ?? "{}") as Record<string, unknown>;
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
