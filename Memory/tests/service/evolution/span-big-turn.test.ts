import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage
} from "../../../src/index.js";
import {
  parseSpanDrafts,
  spanId,
  spanKey,
  validateSpanDrafts
} from "../../../src/service/evolution/span-model.js";
import {
  createCapturingEmbedder,
  createMemoryServiceFixture,
  runWorkerRounds
} from "../../fixtures/memory-service-fixture.js";

const {
  cleanup,
  createTestService
} = createMemoryServiceFixture();

afterEach(cleanup);

function createSpanBigTurnLlm(
  calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }>,
  spanResult: Record<string, unknown> | Array<Record<string, unknown>> = {
    shouldSplit: false,
    reason: "即使 shouldSplit 冲突，也以非空 spans 为准",
    spans: [
      {
        start: 0,
        end: 3,
        goal: "定位构建失败的根本原因",
        policy: "先读取日志和配置定位首个失败依赖边",
        summary: "检查日志和依赖配置，确认失败来自版本冲突"
      },
      {
        start: 4,
        end: 10,
        goal: "修复依赖版本冲突并验证结果",
        policy: "调整依赖版本后立即运行构建与测试验证",
        summary: "调整依赖版本和构建配置，消除不兼容引用"
      }
    ]
  }
): LlmClient {
  let spanCallIndex = 0;
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
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      calls.push({ messages, options });
      if (options.operation === "capture.reflection.batch.v13") {
        const payload = JSON.parse(messages.at(-1)?.content ?? "{}") as {
          steps?: Array<{ idx: number }>;
        };
        return {
          scores: (payload.steps ?? []).map(({ idx }) => ({
            idx,
            relevance: "PIVOTAL",
            reason: "complex task completed"
          }))
        } as unknown as T;
      }
      if (options.operation === "capture.summarize") {
        return {
          summary: "定位构建失败、修改依赖配置并验证修复结果"
        } as unknown as T;
      }
      if (options.operation === "reward.reward.r_human.v7") {
        return {
          goal_achievement: 1,
          process_quality: 0.9,
          user_satisfaction: 1,
          reason: "复杂开发任务已完成"
        } as unknown as T;
      }
      if (options.operation === "span.big_turn.v1") {
        const selected = Array.isArray(spanResult)
          ? spanResult[Math.min(spanCallIndex, spanResult.length - 1)] ?? {}
          : spanResult;
        spanCallIndex += 1;
        return selected as T;
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
  it("accepts Span drafts with goal, observed policy, and summary", () => {
    expect(parseSpanDrafts({
      spans: [{
        start: 0,
        end: 3,
        goal: "diagnose build failure",
        policy: "trace the first failing dependency edge",
        summary: "isolated an incompatible dependency constraint"
      }]
    }, 4)).toEqual([{
      start: 0,
      end: 3,
      goal: "diagnose build failure",
      policy: "trace the first failing dependency edge",
      summary: "isolated an incompatible dependency constraint"
    }]);
  });

  it("accepts any number of qualifying Span drafts", () => {
    const spans = Array.from({ length: 8 }, (_, index) => ({
      start: index * 4,
      end: index * 4 + 3,
      goal: `goal ${index}`,
      policy: `policy ${index}`,
      summary: `summary ${index}`
    }));
    expect(parseSpanDrafts({ spans }, 32)).toHaveLength(8);
  });

  it("rejects missing policy, too-short spans, invalid ranges, and overlapping spans", () => {
    expect(() => parseSpanDrafts({ spans: [
      { start: 0, end: 0, goal: "g", policy: "", summary: "s" }
    ] }, 2)).toThrow("invalid Span fields");
    expect(() => parseSpanDrafts({ spans: [
      { start: 0, end: 2, goal: "g", policy: "p", summary: "s" }
    ] }, 3)).toThrow("Span must cover more than three tool calls");
    expect(() => validateSpanDrafts([
      { start: 0, end: 3, goal: "g1", policy: "p1", summary: "s1" },
      { start: 3, end: 6, goal: "g2", policy: "p2", summary: "s2" }
    ], 7)).toThrow("overlapping or unordered");
  });

  it("derives stable Span identity from source Trace, range, and schema", () => {
    expect(spanKey("trace-1", 4, 9)).toBe("span:trace-1:4:9:v2");
    expect(spanId("trace-1", 4, 9)).toBe(spanId("trace-1", 4, 9));
    expect(spanId("trace-1", 4, 9)).not.toBe(spanId("trace-1", 4, 10));
  });

  it("stores and recalls subtask spans for a positively rewarded complex turn", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const embeddedTexts: string[] = [];
    const embeddedRoles: Array<"query" | "document" | undefined> = [];
    const llm = createSpanBigTurnLlm(calls);
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        retrieval: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.retrieval,
          relativeThresholdFloor: 0,
          smartSeed: false,
          llmFilterEnabled: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: createCapturingEmbedder(embeddedTexts, embeddedRoles)
    });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-user"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 11 }, (_, index) => ({
      id: `tool-${index}`,
      name: index < 4 ? "read_file" : index < 8 ? "apply_patch" : "run_tests",
      input: index === 0
        ? {
            index,
            apiKey: "sk-supersecret123456",
            token: "private-token-value",
            password: "private-password-value",
            note: "password: \"quoted-password-value\""
          }
        : { index }
    }));
    const toolResults = toolCalls.map((call, index) => ({
      toolCallId: call.id,
      name: call.name,
      output: index === 0
        ? {
            ok: true,
            index,
            authorization: "Bearer private-bearer-token",
            secret: "private-secret-value"
          }
        : { ok: true, index }
    }));
    const completed = service.completeTurn("span-big-turn", {
      namespace,
      sessionId: session.sessionId,
      query: "修复项目构建失败并完成测试验证",
      answer: "已经定位依赖冲突，完成修复并通过构建与测试。",
      toolCalls,
      toolResults
    });
    const invalidToolCall = `invalid-${"x".repeat(140)}`;
    const rawToolCalls: unknown[] = [...toolCalls];
    rawToolCalls[5] = invalidToolCall;
    db.db.prepare(
      `UPDATE raw_turns
       SET tool_calls_json = ?
       WHERE id = ?`
    ).run(JSON.stringify(rawToolCalls), completed.rawTurnId);

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

    const spanCall = calls.find((call) => call.options.operation === "span.big_turn.v1");
    expect(JSON.stringify(service.panelJobs({ userId: namespace.userId }).items)).toContain(
      '"jobType":"span_big_turn","status":"succeeded"'
    );
    expect(JSON.stringify(calls.map((call) => call.options.operation))).toContain("span.big_turn.v1");
    expect(spanCall?.options).toMatchObject({
      thinkingMode: "disabled",
      temperature: 0.6,
      maxTokens: 4096
    });
    expect(spanCall?.messages[0]?.content).toContain("Spans must not overlap");
    expect(spanCall?.messages[0]?.content).toContain("may remain outside all spans");
    expect(spanCall?.messages[0]?.content).toContain("more than three tool calls");
    expect(spanCall?.messages[0]?.content).toContain("programmatic structured trajectory");
    expect(spanCall?.messages[0]?.content).toMatch(/Do not merge evidence gathering,\s+data\s+inspection, artifact construction, debugging\/repair, and final verification/);
    expect(spanCall?.messages[0]?.content).toContain("Long traces should usually produce multiple spans");
    expect(spanCall?.messages[0]?.content).not.toMatch(/2 to 6|2\.\.6/);
    const spanPayload = JSON.parse(spanCall?.messages[1]?.content ?? "{}") as {
      userRequest?: string;
      assistantFinalAnswer?: string;
      traceSummary?: string;
      reflection?: string;
      reward?: { rTask?: number; reason?: string };
      structuredTrajectory?: Array<{
        index: number;
        range: [number, number];
        callCount: number;
        repeated: boolean;
        tool?: string;
        action?: string;
        success?: boolean;
        evidence?: string;
      }>;
    };
    expect(spanPayload).toMatchObject({
      userRequest: "修复项目构建失败并完成测试验证",
      assistantFinalAnswer: "已经定位依赖冲突，完成修复并通过构建与测试。",
      traceSummary: "定位构建失败、修改依赖配置并验证修复结果"
    });
    expect(spanPayload).not.toHaveProperty("reward");
    expect(spanPayload.reflection).toBeTruthy();
    expect(spanPayload).not.toHaveProperty("toolCalls");
    expect(spanPayload.structuredTrajectory).toHaveLength(5);
    expect(spanPayload.structuredTrajectory?.map((call) => call.range)).toEqual([
      [0, 3],
      [4, 4],
      [5, 5],
      [6, 7],
      [8, 10]
    ]);
    expect(spanPayload.structuredTrajectory?.reduce((sum, call) => sum + call.callCount, 0)).toBe(11);
    expect(spanPayload.structuredTrajectory?.[0]).toMatchObject({
      index: 0,
      range: [0, 3],
      callCount: 4,
      repeated: true,
      tool: "read_file",
      action: "read",
      success: true
    });
    expect(spanPayload.structuredTrajectory?.[2]).toMatchObject({
      index: 5,
      range: [5, 5],
      tool: "unknown",
      action: "other",
      success: false
    });
    expect((spanPayload.structuredTrajectory?.[2]?.evidence ?? "").length).toBeLessThanOrEqual(420);
    expect(spanPayload.structuredTrajectory?.[2]?.evidence).toMatch(/^"/);
    const serializedSpanPrompt = spanCall?.messages[1]?.content ?? "";
    expect(serializedSpanPrompt).toContain("[redacted]");
    expect(serializedSpanPrompt).not.toContain("sk-supersecret123456");
    expect(serializedSpanPrompt).not.toContain("private-token-value");
    expect(serializedSpanPrompt).not.toContain("private-password-value");
    expect(serializedSpanPrompt).not.toContain("quoted-password-value");
    expect(serializedSpanPrompt).not.toContain("private-bearer-token");
    expect(serializedSpanPrompt).not.toContain("private-secret-value");
    const recall = await service.search({
      namespace,
      query: "定位构建失败的根本原因",
      layers: ["L1"],
      includeInjectedContext: true
    });
    expect(recall.hits).toEqual(expect.arrayContaining([
      expect.objectContaining({
        kind: "span",
        title: "定位构建失败的根本原因",
        snippet: expect.stringContaining("确认失败来自版本冲突")
      })
    ]));
    expect(recall.injectedContext.markdown).toContain("定位构建失败的根本原因");
    expect(recall.injectedContext.markdown).toContain("确认失败来自版本冲突");
    const semanticRecall = await service.search({
      namespace,
      query: "读取日志和配置定位首个失败依赖边",
      layers: ["L1"]
    });
    expect(semanticRecall.hits).toEqual(expect.arrayContaining([
      expect.objectContaining({
        kind: "span",
        title: "定位构建失败的根本原因"
      })
    ]));
    expect(semanticRecall.hits.map((hit) => hit.id)).not.toContain(completed.l1MemoryId);
    expect(semanticRecall.hits.filter((hit) => hit.kind === "span").length).toBeLessThanOrEqual(2);
    expect(embeddedTexts).toContain("Goal: 定位构建失败的根本原因");
    expect(embeddedTexts).toContain("Policy: 先读取日志和配置定位首个失败依赖边");
    expect(embeddedTexts).toContain("Goal: 修复依赖版本冲突并验证结果");
    expect(embeddedTexts).toContain("Policy: 调整依赖版本后立即运行构建与测试验证");
    expect(embeddedTexts.filter((text, index) =>
      embeddedRoles[index] === "document" &&
      /^Goal: |^Policy: /.test(text)
    )).toHaveLength(4);
    const l1Items = service.panelItems({ namespace, layer: "L1" }).items;
    expect(l1Items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ title: "定位构建失败的根本原因" }),
        expect.objectContaining({ title: "修复依赖版本冲突并验证结果" })
      ])
    );
    expect(l1Items.filter((item) => item.kind === "span")).toHaveLength(2);
    const parentRow = db.db.prepare(
      `SELECT properties_json FROM memories WHERE id = ?`
    ).get(completed.l1MemoryId) as { properties_json: string };
    const parentProperties = JSON.parse(parentRow.properties_json) as {
      internal_info: {
        trace: {
          span_ids?: string[];
        };
      };
    };
    const storedSpanIds = l1Items
      .filter((item) => item.kind === "span")
      .map((item) => item.id);
    expect(parentProperties.internal_info.trace.span_ids).toHaveLength(2);
    expect(parentProperties.internal_info.trace.span_ids).toEqual(
      expect.arrayContaining(storedSpanIds)
    );
    const spanEmbeddingRows = db.db.prepare(
      `SELECT memory_id, vector_field
       FROM memory_vector_entries
       WHERE memory_id IN (${storedSpanIds.map(() => "?").join(",")})
       ORDER BY memory_id, vector_field`
    ).all(...storedSpanIds) as Array<{ memory_id: string; vector_field: string }>;
    expect(spanEmbeddingRows).toHaveLength(4);
    for (const spanMemoryId of storedSpanIds) {
      expect(spanEmbeddingRows).toEqual(expect.arrayContaining([
        { memory_id: spanMemoryId, vector_field: "vec_goal" },
        { memory_id: spanMemoryId, vector_field: "vec_policy" }
      ]));
    }
    expect(db.db.prepare(
      `SELECT COUNT(*) AS count
       FROM evolution_jobs
       WHERE target_memory_id IN (${storedSpanIds.map(() => "?").join(",")})
         AND job_type = 'embedding'
         AND json_extract(payload_json, '$.vectorField') IN ('vec_goal', 'vec_policy')`
    ).get(...storedSpanIds)).toEqual({ count: 4 });
    const firstSpanId = l1Items.find((item) => item.title === "定位构建失败的根本原因")?.id;
    const spanDetail = service.getMemory(firstSpanId!, { namespace }) as {
      item: { metadata: { spanDetail?: { toolCallStart?: number; toolCallEnd?: number; toolCalls?: Array<{ id?: string }> } } };
    };
    expect(spanDetail.item.metadata.spanDetail).toMatchObject({
      toolCallStart: 0,
      toolCallEnd: 3,
      toolCalls: toolCalls.slice(0, 4).map(({ id, name, input }) => ({ id, name, input }))
    });
    expect(completed.l1MemoryIds).toHaveLength(1);
    db.close();
  });

  it("keeps the parent trace only when the complex turn has one coherent goal", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const llm = createSpanBigTurnLlm(calls, {
      shouldSplit: false,
      reason: "所有工具调用都服务于同一个迁移目标",
      spans: []
    });
    const { db, service } = createTestService({ llm, skillLlm: llm });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-no-split"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 11 }, (_, index) => ({
      id: `single-goal-tool-${index}`,
      name: "apply_migration_step",
      input: { index }
    }));
    const completed = service.completeTurn("span-big-turn-no-split", {
      namespace,
      sessionId: session.sessionId,
      query: "完成这一个数据库迁移目标",
      answer: "数据库迁移已经完成。",
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
      rationale: "迁移结果正确"
    });
    await runWorkerRounds(service, 8);

    expect(calls.filter((call) => call.options.operation === "span.big_turn.v1")).toHaveLength(1);
    expect(service.panelItems({ namespace, layer: "L1" }).items.filter(
      (item) => item.kind === "span"
    )).toEqual([]);
    db.close();
  });

  it("splits long turns into fixed raw tool-call windows before LLM span extraction", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const llm = createSpanBigTurnLlm(calls, [
      {
        reason: "first fixed window",
        spans: [{
          start: 0,
          end: 11,
          goal: "收集和读取输入证据",
          policy: "先批量读取相关输入和上下文证据，形成后续处理依据",
          summary: "完成输入证据读取"
        }]
      },
      {
        reason: "second fixed window",
        spans: [{
          start: 24,
          end: 41,
          goal: "生成中间制品并修复错误",
          policy: "在生成制品的过程中根据执行错误迭代修复脚本",
          summary: "生成中间制品并完成错误修复"
        }]
      },
      {
        reason: "third fixed window",
        spans: [{
          start: 54,
          end: 64,
          goal: "验证最终输出制品",
          policy: "通过读取输出和运行验证命令确认最终制品可用",
          summary: "最终制品验证通过"
        }]
      }
    ]);
    const { db, service } = createTestService({ llm, skillLlm: llm });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-windowed"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 65 }, (_, index) => ({
      id: `window-tool-${index}`,
      name: index < 24 ? "read_file" : index < 54 ? "execute_script" : "verify_output",
      input: { index }
    }));
    const completed = service.completeTurn("span-big-turn-windowed", {
      namespace,
      sessionId: session.sessionId,
      query: "完成包含读取、生成、验证阶段的长任务",
      answer: "长任务已经完成。",
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
      rationale: "长任务结果正确"
    });
    await runWorkerRounds(service, 8);

    const spanCalls = calls.filter((call) => call.options.operation === "span.big_turn.v1");
    expect(spanCalls).toHaveLength(3);
    const payloads = spanCalls.map((call) => JSON.parse(call.messages[1]?.content ?? "{}") as {
      segment?: { start: number; end: number; reason: string };
      structuredTrajectory?: Array<{ range: [number, number] }>;
    });
    expect(payloads.map((payload) => payload.segment)).toEqual([
      { start: 0, end: 29, reason: "raw_fixed_30_overlap_6" },
      { start: 24, end: 53, reason: "raw_fixed_30_overlap_6" },
      { start: 48, end: 64, reason: "raw_fixed_30_overlap_6" }
    ]);
    for (const payload of payloads) {
      expect(payload.structuredTrajectory?.every((event) =>
        event.range[0] >= payload.segment!.start && event.range[1] <= payload.segment!.end
      )).toBe(true);
    }
    const spans = service.panelItems({ namespace, layer: "L1" }).items.filter(
      (item) => item.kind === "span"
    );
    expect(spans.map((item) => item.title)).toEqual(expect.arrayContaining([
      "收集和读取输入证据",
      "生成中间制品并修复错误",
      "验证最终输出制品"
    ]));
    db.close();
  });

  it("drops partially overlapping window drafts instead of changing their semantic range", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const llm = createSpanBigTurnLlm(calls, [
      {
        spans: [{
          start: 0,
          end: 28,
          goal: "第一阶段策略",
          policy: "完整读取输入证据并形成分析依据",
          summary: "完成第一阶段"
        }]
      },
      {
        spans: [{
          start: 24,
          end: 40,
          goal: "第二阶段策略",
          policy: "在中间制品生成过程中修复执行错误",
          summary: "完成第二阶段"
        }]
      },
      {
        spans: [{
          start: 54,
          end: 64,
          goal: "第三阶段策略",
          policy: "读取输出并执行验证命令确认结果",
          summary: "完成第三阶段"
        }]
      }
    ]);
    const { db, service } = createTestService({ llm, skillLlm: llm });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-overlap-window"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 65 }, (_, index) => ({
      id: `overlap-window-tool-${index}`,
      name: index < 24 ? "read_file" : index < 54 ? "execute_script" : "verify_output",
      input: { index }
    }));
    const completed = service.completeTurn("span-big-turn-overlap-window", {
      namespace,
      sessionId: session.sessionId,
      query: "完成一个包含多个阶段的长任务",
      answer: "长任务已经完成。",
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
      rationale: "长任务结果正确"
    });
    await runWorkerRounds(service, 8);

    const spans = service.panelItems({ namespace, layer: "L1" }).items.filter(
      (item) => item.kind === "span"
    );
    expect(spans.map((item) => item.title)).toEqual(expect.arrayContaining([
      "第一阶段策略",
      "第三阶段策略"
    ]));
    expect(spans.map((item) => item.title)).not.toContain("第二阶段策略");
    db.close();
  });

  it("stores only the successful model result after a failed attempt", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const llm = createSpanBigTurnLlm(calls, [
      {
        shouldSplit: true,
        reason: "invalid overlapping draft",
        spans: [
          {
            start: 0,
            end: 6,
            goal: "无效的第一次分析",
            policy: "错误地让两个策略范围共享边界",
            summary: "范围发生重叠"
          },
          {
            start: 6,
            end: 10,
            goal: "无效的第一次修复",
            policy: "错误地让两个策略范围共享边界",
            summary: "范围发生重叠"
          }
        ]
      },
      {
        shouldSplit: true,
        reason: "valid retry",
        spans: [
          {
            start: 0,
            end: 3,
            goal: "分析问题",
            policy: "先连续检查上下文和失败证据再确定修复点",
            summary: "完成问题分析"
          },
          {
            start: 4,
            end: 10,
            goal: "实施修复并验证结果",
            policy: "应用修复后用后续工具调用验证结果",
            summary: "完成问题修复和结果验证"
          }
        ]
      }
    ]);
    const { db, service } = createTestService({ llm, skillLlm: llm });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-replace"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 11 }, (_, index) => ({
      id: `replace-tool-${index}`,
      name: "task_step",
      input: { index }
    }));
    const completed = service.completeTurn("span-big-turn-replace", {
      namespace,
      sessionId: session.sessionId,
      query: "完成需要多步处理的复杂任务",
      answer: "复杂任务已完成。",
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
      rationale: "任务结果正确"
    });
    await runWorkerRounds(service, 8);

    const spans = service.panelItems({ namespace, layer: "L1" }).items.filter(
      (item) => item.kind === "span"
    );
    expect(calls.filter((call) => call.options.operation === "span.big_turn.v1")).toHaveLength(2);
    expect(spans).toHaveLength(2);
    expect(spans.map((item) => item.title)).toEqual(expect.arrayContaining([
      "分析问题",
      "实施修复并验证结果"
    ]));
    expect(spans.some((item) => item.title.startsWith("无效的第一次"))).toBe(false);
    db.close();
  });

  it("requires both a positive reward and more than ten tool calls", async () => {
    for (const scenario of [
      { userId: "span-big-turn-ten-tools", toolCount: 10, polarity: "positive" as const },
      { userId: "span-big-turn-negative", toolCount: 11, polarity: "negative" as const }
    ]) {
      const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
      const llm = createSpanBigTurnLlm(calls);
      const config = {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          reward: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.reward,
            llmScoring: false
          }
        }
      };
      const { db, service } = createTestService({ config, llm, skillLlm: llm });
      const namespace = {
        source: "codex",
        profileId: "jiang",
        userId: scenario.userId
      };
      const session = service.openSession({ namespace });
      const toolCalls = Array.from({ length: scenario.toolCount }, (_, index) => ({
        id: `${scenario.userId}-tool-${index}`,
        name: "task_step",
        input: { index }
      }));
      const completed = service.completeTurn(`${scenario.userId}-turn`, {
        namespace,
        sessionId: session.sessionId,
        query: "执行复杂任务",
        answer: "任务执行结束。",
        toolCalls,
        toolResults: toolCalls.map((call, index) => ({
          toolCallId: call.id,
          name: call.name,
          output: { ok: scenario.polarity === "positive", index }
        }))
      });
      service.closeSession(session.sessionId);
      await service.feedback({
        namespace,
        sessionId: session.sessionId,
        episodeId: completed.episodeId,
        l1MemoryId: completed.l1MemoryId,
        channel: "explicit",
        polarity: scenario.polarity,
        magnitude: 1,
        rationale: scenario.polarity === "positive" ? "结果正确" : "任务没有正确完成"
      });
      await runWorkerRounds(service, 8);

      expect(calls.some((call) => call.options.operation === "span.big_turn.v1")).toBe(false);
      expect(service.panelItems({ namespace, layer: "L1" }).items.some(
        (item) => item.kind === "span"
      )).toBe(false);
      db.close();
    }
  });

  it("rejects overlapping spans without storing partial results", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const llm = createSpanBigTurnLlm(calls, {
      shouldSplit: true,
      reason: "invalid overlapping draft",
      spans: [
        {
          start: 0,
          end: 6,
          goal: "分析失败原因",
          policy: "连续检查失败证据定位根因",
          summary: "分析构建失败"
        },
        {
          start: 6,
          end: 10,
          goal: "修复失败原因",
          policy: "根据分析结果调整配置并验证",
          summary: "修改配置并验证"
        }
      ]
    });
    const { db, service } = createTestService({ llm, skillLlm: llm });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-overlap"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 11 }, (_, index) => ({
      id: `overlap-tool-${index}`,
      name: "task_step",
      input: { index }
    }));
    const completed = service.completeTurn("span-big-turn-overlap", {
      namespace,
      sessionId: session.sessionId,
      query: "分析并修复构建失败",
      answer: "已经完成分析和修复。",
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
      rationale: "任务结果正确"
    });
    await runWorkerRounds(service, 8);

    expect(calls.filter((call) => call.options.operation === "span.big_turn.v1")).toHaveLength(3);
    expect(service.panelItems({ namespace, layer: "L1" }).items.some(
      (item) => item.kind === "span"
    )).toBe(false);
    expect(service.panelJobs({ userId: namespace.userId }).items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          jobType: "span_big_turn",
          status: "dead_letter",
          lastError: "span.big_turn returned overlapping or unordered spans"
        })
      ])
    );
    db.close();
  });

  it("uses the summary LLM for Span splitting so deployments can choose its model independently", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const summaryLlm = createSpanBigTurnLlm(calls, {
      spans: [{
        start: 0,
        end: 10,
        goal: "拆分长工具轨迹中的局部策略",
        policy: "使用结构化工具轨迹和边界规则抽取连续策略片段",
        summary: "从长工具轨迹中抽取出一个可复用策略片段"
      }]
    });
    const evolutionLlm: LlmClient = {
      ...summaryLlm,
      config: { ...summaryLlm.config, model: "evolution-unconfigured" },
      async completeJson<T extends Record<string, unknown>>(): Promise<T> {
        throw new Error("span splitting should not call evolution LLM");
      }
    };
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        capture: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
          alphaScoring: false
        },
        reward: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.reward,
          llmScoring: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm: summaryLlm,
      skillLlm: evolutionLlm
    });
    const namespace = {
      source: "codex",
      profileId: "jiang",
      userId: "span-big-turn-evolution-llm"
    };
    const session = service.openSession({ namespace });
    const toolCalls = Array.from({ length: 11 }, (_, index) => ({
      id: `evolution-llm-tool-${index}`,
      name: "task_step",
      input: { index }
    }));
    const completed = service.completeTurn("span-big-turn-evolution-llm", {
      namespace,
      sessionId: session.sessionId,
      query: "执行一个需要拆分 span 的长任务",
      answer: "长任务已经完成。",
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
      rationale: "长任务结果正确"
    });
    await runWorkerRounds(service, 8);

    expect(calls.some((call) => call.options.operation === "span.big_turn.v1")).toBe(true);
    expect(service.panelItems({ namespace, layer: "L1" }).items).toEqual(expect.arrayContaining([
      expect.objectContaining({
        kind: "span",
        title: "拆分长工具轨迹中的局部策略"
      })
    ]));
    db.close();
  });
});
