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
  PROCEDURAL_SPAN_CAPABILITY_PROMPT,
  PROCEDURAL_SPAN_STATE_PROMPT,
  PROCEDURAL_SPAN_RECONCILIATION_PROMPT,
  PROCEDURAL_SPAN_SEGMENTATION_PROMPT,
  TASK_CONTRACT_PROMPT,
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
    expect(PROCEDURAL_SPAN_CAPABILITY_PROMPT).toContain(
      "capability_goal answers WHAT reusable outcome is produced"
    );
    expect(PROCEDURAL_SPAN_CAPABILITY_PROMPT).toContain("procedure_semantic answers HOW");
    expect(PROCEDURAL_SPAN_CAPABILITY_PROMPT).toContain("Span boundaries");
  });

  it("keeps the Step LLM contract minimal and moves task/state semantics out of it", () => {
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).toContain("intent");
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).toContain("summary");
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).not.toContain('"observations"');
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).not.toContain('"operations"');
    expect(EXECUTION_STEP_SEMANTICS_PROMPT).not.toContain("retry_of_candidate_id");
    expect(TASK_CONTRACT_PROMPT).toContain("acceptance_criteria");
    expect(PROCEDURAL_SPAN_STATE_PROMPT).toContain("issues_resolved");
    expect(PROCEDURAL_SPAN_STATE_PROMPT).toContain("verification");
    expect(PROCEDURAL_SPAN_STATE_PROMPT).toContain("exactly created, updated, or verified");
  });

  it("skips every Span LLM stage in Step-sequence mode", async () => {
    const operations: string[] = [];
    const base = createProceduralLlm((stepIds) => [stepIds]);
    const llm: LlmClient = {
      ...base,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        operations.push(options.operation);
        if (options.operation.includes("span_")) {
          throw new Error(`unexpected Span LLM call: ${options.operation}`);
        }
        return base.completeJson<T>(messages, options);
      }
    };
    const rawTurns = [makeRawTurn({
      id: "raw-step-sequence-mode",
      episodeId: "episode-step-sequence-mode",
      userText: "Inspect and verify the build",
      assistantText: "The build is verified.",
      toolCalls: [
        tool("read_file", "package.json", "configuration loaded"),
        tool("npm_test", "focused suite", "18 tests passed")
      ]
    })];

    const path = await new EpisodeProceduralReconstructor({
      llm,
      mode: "step_sequence"
    }).reconstruct({
      episodeId: "episode-step-sequence-mode",
      rawTurns,
      terminalReward: 1
    });

    expect(path.steps).toHaveLength(2);
    expect(path.spans).toHaveLength(1);
    expect(path.spans[0]!.stepIds).toEqual(path.steps.map((step) => step.id));
    expect(path.spans[0]!.segmentation.reason).toContain("compatibility envelope");
    expect(operations.some((operation) => operation.startsWith("procedural.task_contract"))).toBe(true);
    expect(operations.some((operation) => operation.startsWith("procedural.step_semantics"))).toBe(true);
    expect(operations.some((operation) => operation.includes("span_"))).toBe(false);
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
      capabilityGoal: "execute reusable subproblem 1 with an observable result",
      procedureSemantic: "inspect inputs -> execute reusable subproblem -> verify the result",
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
    expect(path.steps.slice(0, -1).every((step) => step.actionEffectDelta.length === 0)).toBe(true);
    expect(path.steps.at(-1)?.actionEffectDelta.length).toBeGreaterThan(0);
    const finalState = path.states.find((state) => state.id === path.spans[0]?.postStateId);
    expect(finalState?.summary).toContain("facts=");
    expect(finalState?.taskStatus).toBe("completed");
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
      subject: "保持 NodeNext，不要降级 TypeScript"
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

  it("repairs an incomplete minimal Step response without reintroducing state operations", async () => {
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
        if (!options.operation.startsWith("procedural.step_semantics.v2")) return valid;
        stepSemanticCalls += 1;
        if (options.operation.includes(".repair.")) {
          repairPrompts.push(messages.at(-1)?.content ?? "");
        }
        if (stepSemanticCalls <= 2) {
          return { steps: [] } as unknown as T;
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
    expect(repairPrompts[0]).toContain("candidate");
    expect(repairPrompts[0]).toContain("intent");
    expect(repairPrompts[0]).toContain("summary");
    expect(repairPrompts[1]).toContain("Discard the previous JSON");
    expect(path.steps).toHaveLength(1);
    expect(path.steps[0]?.actionEffectDelta.length).toBeGreaterThan(0);
  });

  it("normalizes an unambiguous camelCase Step candidate ID", async () => {
    const baseLlm = createProceduralLlm((stepIds) => [stepIds]);
    const llm: LlmClient = {
      ...baseLlm,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        const valid = await baseLlm.completeJson<Record<string, unknown>>(messages, options);
        if (!options.operation.startsWith("procedural.step_semantics.v2")) return valid as T;
        const steps = valid.steps as Array<Record<string, unknown>>;
        return {
          steps: steps.map(({ candidate_id: candidateId, ...step }) => ({
            candidateId,
            ...step
          }))
        } as unknown as T;
      }
    };
    const rawTurns = [makeRawTurn({
      id: "raw-camel-case-step-id",
      episodeId: "episode-camel-case-step-id",
      userText: "验证构建",
      assistantText: "构建验证完成。",
      toolCalls: [tool("npm_run", "build", "build passed")]
    })];

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-camel-case-step-id",
      rawTurns
    });

    expect(path.steps).toHaveLength(1);
    expect(path.steps[0]?.action.intent).toContain("active subproblem");
  });

  it("drops unsupported Span-state claims after bounded evidence-ref repairs", async () => {
    const baseLlm = createProceduralLlm((stepIds) => [stepIds]);
    let spanStateCalls = 0;
    const llm: LlmClient = {
      ...baseLlm,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        const valid = await baseLlm.completeJson<Record<string, unknown>>(messages, options);
        if (!options.operation.startsWith("procedural.span_state.v1")) return valid as T;
        spanStateCalls += 1;
        const spans = valid.spans as Array<Record<string, unknown>>;
        return {
          spans: spans.map((span) => ({
            ...span,
            effects: [{
              summary: "unsupported hallucinated effect",
              evidence_refs: ["step_invented"]
            }]
          }))
        } as unknown as T;
      }
    };
    const rawTurns = [makeRawTurn({
      id: "raw-span-state-invalid-ref",
      episodeId: "episode-span-state-invalid-ref",
      userText: "验证构建",
      assistantText: "构建验证完成。",
      toolCalls: [tool("npm_run", "build", "build passed")]
    })];

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-span-state-invalid-ref",
      rawTurns
    });

    expect(spanStateCalls).toBe(3);
    expect(path.states.map((state) => state.summary).join("\n"))
      .not.toContain("unsupported hallucinated effect");
    expect(path.spans).toHaveLength(1);
  });

  it("bounds a long turn into 30-step windows with five preceding context steps", async () => {
    const baseLlm = createProceduralLlm((stepIds) => [stepIds]);
    const stepPayloads: Array<Record<string, unknown>> = [];
    const taskContractPayloads: Array<Record<string, unknown>> = [];
    const segmentationPayloads: Array<Record<string, unknown>> = [];
    const reconciliationPayloads: Array<Record<string, unknown>> = [];
    const capabilityPayloads: Array<Record<string, unknown>> = [];
    const operations: string[] = [];
    const llm: LlmClient = {
      ...baseLlm,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        const payload = llmRequestPayload(messages);
        operations.push(options.operation);
        if (options.operation === "procedural.task_contract.v1") taskContractPayloads.push(payload);
        if (options.operation === "procedural.step_semantics.v2.window") stepPayloads.push(payload);
        if (options.operation === "procedural.span_segmentation.v3.window") segmentationPayloads.push(payload);
        if (options.operation === "procedural.span_reconciliation.v2") reconciliationPayloads.push(payload);
        if (options.operation === "procedural.span_capability.v1") capabilityPayloads.push(payload);
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
      userText: `完成一个包含大量检查步骤的长任务：${"约束与验收条件。".repeat(300)}`,
      assistantText: "所有检查均已完成。",
      toolCalls
    })];

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-long-windowed-turn",
      rawTurns
    });

    expect(path.steps).toHaveLength(65);
    expect(path.spans).toHaveLength(1);
    expect(taskContractPayloads).toHaveLength(1);
    expect(((taskContractPayloads[0]?.observations as Array<{ userText: string }>)[0]?.userText.length ?? 0))
      .toBeGreaterThan(1_500);
    expect(stepPayloads.map((payload) => (payload.stepCandidates as unknown[]).length)).toEqual([30, 30, 6]);
    expect(stepPayloads.map((payload) => (payload.precedingStepContext as unknown[]).length)).toEqual([0, 5, 5]);
    expect(stepPayloads.every((payload) => !("observations" in payload))).toBe(true);
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
    expect(capabilityPayloads).toHaveLength(1);
    expect(capabilityPayloads[0]?.spans).toEqual([
      expect.objectContaining({
        spanIndex: 0,
        localGoal: "solve reconciled subproblem 1"
      })
    ]);
    expect(operations.indexOf("procedural.span_capability.v1"))
      .toBeGreaterThan(operations.indexOf("procedural.span_reconciliation.v2"));
    expect(segmentationPayloads.every((payload) =>
      !(payload.steps as Array<Record<string, unknown>>).some((step) => "capabilityGoal" in step)
    )).toBe(true);
  });

  it("repairs a later window that copies an ID from read-only preceding context", async () => {
    const baseLlm = createProceduralLlm((stepIds) => [stepIds]);
    let precedingCandidateId = "";
    let repairPrompt = "";
    let repairCount = 0;
    const laterWindowPayloads: Array<Record<string, unknown>> = [];
    const llm: LlmClient = {
      ...baseLlm,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        const payload = llmRequestPayload(messages);
        if (options.operation === "procedural.step_semantics.v2.window") {
          const window = payload.window as { windowIndex: number };
          const candidates = payload.stepCandidates as Array<{ candidateId: string }>;
          if (window.windowIndex === 0) {
            precedingCandidateId = candidates[0]!.candidateId;
          } else {
            laterWindowPayloads.push(payload);
            return {
              steps: [{
                candidate_id: precedingCandidateId,
                include: true,
                intent: "copied the preceding context candidate",
                summary: "This item does not belong to the target window."
              }]
            } as unknown as T;
          }
        }
        if (options.operation === "procedural.step_semantics.v2.repair.1") {
          repairCount += 1;
          repairPrompt = messages.at(-1)?.content ?? "";
        }
        return baseLlm.completeJson<T>(messages, options);
      }
    };
    const rawTurns = [makeRawTurn({
      id: "raw-preceding-context-id-repair",
      episodeId: "episode-preceding-context-id-repair",
      userText: "完成一组连续检查",
      assistantText: "连续检查已完成。",
      toolCalls: Array.from({ length: 31 }, (_, index) => tool(
        "inspect",
        `target-${index}`,
        `observed-${index}`
      ))
    })];

    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: "episode-preceding-context-id-repair",
      rawTurns
    });

    expect(path.steps).toHaveLength(31);
    expect(repairCount).toBe(1);
    expect(laterWindowPayloads).toHaveLength(1);
    expect(laterWindowPayloads[0]?.precedingStepContext).toEqual(expect.arrayContaining([
      expect.not.objectContaining({ candidateId: expect.anything() })
    ]));
    const laterCandidateIds = (laterWindowPayloads[0]?.stepCandidates as Array<{ candidateId: string }>)
      .map((candidate) => candidate.candidateId);
    expect(repairPrompt).toContain(JSON.stringify(laterCandidateIds));
    expect(repairPrompt).toContain("exclusive allowed candidate_id list");
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
      if (options.operation.startsWith("procedural.task_contract.v1")) {
        const observations = payload.observations as Array<{
          sourceId: string;
          rawTurnId: string;
          userText?: string;
        }>;
        return {
          contracts: observations.map((observation, index) => ({
            source_id: observation.sourceId,
            goal: index === 0 ? observation.userText ?? "complete the task" : null,
            constraints: index === 0 ? [] : [observation.userText ?? "follow-up constraint"],
            acceptance_criteria: index === 0 ? ["the requested result is verified"] : []
          }))
        } as unknown as T;
      }
      if (options.operation.startsWith("procedural.step_semantics.v2")) {
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
          ...(isRecordForTest(payload.window) && Number(payload.window.windowIndex) > 0
            ? {
                observations: [{
                  source_id: "turn:invented:later-window",
                  operations: [{ operation: "fact.verify" }]
                }]
              }
            : {}),
          steps: candidates.map((candidate, candidateIndex) => {
            const include = candidate.kind === "tool_action";
            return {
              candidate_id: candidate.candidateId,
              include,
              intent: include ? `${candidate.action} for the active subproblem` : "report completion",
              summary: include
                ? `${candidate.toolName ?? candidate.action} produced observable evidence ${candidateIndex}.`
                : "Social report only."
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
      if (options.operation.startsWith("procedural.span_capability.v1")) {
        const spans = payload.spans as Array<{ spanIndex: number }>;
        return {
          spans: spans.map((span) => ({
            span_index: span.spanIndex,
            capability_goal: `execute reusable subproblem ${span.spanIndex + 1} with an observable result`,
            procedure_semantic: "inspect inputs -> execute reusable subproblem -> verify the result"
          }))
        } as unknown as T;
      }
      if (options.operation.startsWith("procedural.span_state.v1")) {
        const spans = payload.spans as Array<{
          spanIndex: number;
          terminationStatus: string;
          evidenceRefs: string[];
          exitCondition: string;
        }>;
        return {
          spans: spans.map((span) => ({
            span_index: span.spanIndex,
            effects: [{
              summary: span.exitCondition,
              evidence_refs: [span.evidenceRefs.at(-1)]
            }],
            artifacts: [],
            issues_opened: span.terminationStatus === "success" ? [] : [{
              issue: span.exitCondition,
              evidence_refs: [span.evidenceRefs.at(-1)]
            }],
            issues_resolved: span.terminationStatus === "success" ? [{
              issue: span.exitCondition,
              evidence_refs: [span.evidenceRefs.at(-1)]
            }] : [],
            verification: [{
              criterion: span.exitCondition,
              status: span.terminationStatus === "success" ? "passed" : "failed",
              evidence_refs: [span.evidenceRefs.at(-1)]
            }]
          }))
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

function isRecordForTest(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
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
