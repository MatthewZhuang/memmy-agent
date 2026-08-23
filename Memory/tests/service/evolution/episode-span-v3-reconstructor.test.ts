import { describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage
} from "../../../src/index.js";
import type { RawTurnRecord } from "../../../src/storage/repositories.js";
import {
  EpisodeSpanV3Reconstructor,
  buildEpisodeActionEvents,
  proposeSpanBoundaryCandidates,
  segmentActionEvents
} from "../../../src/service/evolution/episode-span-v3-reconstructor.js";
import {
  applyStateDelta,
  emptyObservedState,
  validateSpanContinuity
} from "../../../src/service/evolution/span-v3-model.js";

describe("EpisodeSpanV3Reconstructor", () => {
  it("proposes only generic structural candidates and does not use a tool-count threshold", () => {
    const events = buildEpisodeActionEvents(makeRawTurn({
      id: "raw-boundary",
      toolCalls: [
        { name: "web_search", input: { q: "one" }, output: "one" },
        { name: "web_search", input: { q: "two" }, output: "two" },
        { name: "read_file", input: { path: "notes.md" }, output: "notes" },
        { name: "apply_patch", input: { path: "notes.md" }, output: { ok: true } },
        { name: "npm_test", input: { command: "npm test" }, error: "test failed", success: false }
      ]
    }), 0);

    expect(events).toHaveLength(4);
    const candidates = proposeSpanBoundaryCandidates(events);
    expect(candidates).toHaveLength(2);
    expect(candidates.map((candidate) => candidate.signals)).toEqual([
      ["action_family_change"],
      ["action_family_change", "outcome_change"]
    ]);

    const segments = segmentActionEvents(
      "raw-boundary",
      0,
      events,
      candidates,
      new Set([candidates[0]!.id])
    );
    expect(segments.map((segment) => segment.events.length)).toEqual([2, 2]);
    expect(segments.map((segment) => segment.toolCallRange)).toEqual([[0, 2], [3, 4]]);
  });

  it("reduces evidence-grounded deltas deterministically without mutating prior states", () => {
    const initial = emptyObservedState();
    const prepared = applyStateDelta(initial, [
      {
        op: "goal.set",
        subject: "task_goal",
        value: "prepare a release",
        sourceRefs: ["turn:1:user"]
      },
      {
        op: "constraint.upsert",
        subject: "target_branch",
        value: "release",
        sourceRefs: ["turn:1:user"]
      }
    ]);
    const completed = applyStateDelta(prepared, [
      {
        op: "artifact.upsert",
        subject: "release_bundle",
        value: "dist/app.tgz",
        sourceRefs: ["turn:1:action:0-0"]
      },
      {
        op: "artifact.verify",
        subject: "release_bundle",
        sourceRefs: ["turn:1:action:1-1"]
      },
      {
        op: "status.set",
        subject: "task_status",
        value: "completed",
        sourceRefs: ["turn:2:user"]
      }
    ]);

    expect(initial.goal).toBeUndefined();
    expect(prepared.taskStatus).toBe("active");
    expect(applyStateDelta(prepared, [{
      op: "goal.complete",
      subject: "task_goal",
      sourceRefs: ["turn:2:user"]
    }]).taskStatus).toBe("completed");
    expect(completed.taskStatus).toBe("completed");
    expect(completed.artifacts[0]).toMatchObject({
      subject: "release_bundle",
      value: "dist/app.tgz",
      status: "verified",
      sourceRefs: ["turn:1:action:0-0", "turn:1:action:1-1"]
    });
    expect(applyStateDelta(prepared, [
      {
        op: "artifact.upsert",
        subject: "release_bundle",
        value: "dist/app.tgz",
        sourceRefs: ["turn:1:action:0-0"]
      },
      {
        op: "artifact.verify",
        subject: "release_bundle",
        sourceRefs: ["turn:1:action:1-1"]
      },
      {
        op: "status.set",
        subject: "task_status",
        value: "completed",
        sourceRefs: ["turn:2:user"]
      }
    ]).id).toBe(completed.id);
  });

  it("reconstructs one continuous path across a multi-turn episode", async () => {
    const calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }> = [];
    const llm = createEpisodeReconstructionLlm(calls);
    const reconstructor = new EpisodeSpanV3Reconstructor({ llm });
    const rawTurns = travelEpisodeTurns();

    const first = await reconstructor.reconstruct({
      episodeId: "episode-travel",
      rawTurns,
      terminalReward: 1
    });
    const second = await reconstructor.reconstruct({
      episodeId: "episode-travel",
      rawTurns,
      terminalReward: 1
    });

    expect(first.pathHash).toBe(second.pathHash);
    expect(first.spans).toHaveLength(3);
    expect(first.states).toHaveLength(4);
    expect(first.spans.map((span) => span.cost.toolCalls)).toEqual([5, 2, 0]);
    expect(first.spans.map((span) => span.rawTurnId)).toEqual([
      "raw-research",
      "raw-weather",
      "raw-plan"
    ]);
    expect(first.spans[0]?.externalObservationDelta).toContainEqual(expect.objectContaining({
      op: "goal.refine",
      value: "了解九月天气"
    }));
    expect(first.spans[1]?.externalObservationDelta).toContainEqual(expect.objectContaining({
      op: "constraint.upsert",
      subject: "trip_duration",
      value: "两周"
    }));
    expect(first.states.at(-1)).toMatchObject({
      taskStatus: "completed",
      artifacts: [expect.objectContaining({ subject: "travel_itinerary" })]
    });
    expect(first.spans.at(-1)?.outcome.episodeReward).toBe(1);
    expect(() => validateSpanContinuity(first.spans)).not.toThrow();
    for (const [index, span] of first.spans.entries()) {
      expect(span.preStateId).toBe(first.states[index]?.id);
      expect(span.postStateId).toBe(first.states[index + 1]?.id);
    }

    const operationCounts = calls.reduce<Record<string, number>>((counts, call) => {
      counts[call.options.operation] = (counts[call.options.operation] ?? 0) + 1;
      return counts;
    }, {});
    expect(operationCounts).toEqual({
      "span.v3.boundary.v1": 2,
      "span.v3.state_delta.v1": 2
    });
  });

  it("rejects state deltas that cite evidence outside the episode snapshot", async () => {
    const base = createEpisodeReconstructionLlm([]);
    const llm: LlmClient = {
      ...base,
      async completeJson<T extends Record<string, unknown>>(
        messages: LlmMessage[],
        options: LlmCompletionOptions
      ): Promise<T> {
        const result = await base.completeJson<Record<string, unknown>>(messages, options);
        if (options.operation !== "span.v3.state_delta.v1") return result as T;
        const tampered = structuredClone(result) as {
          observations: Array<{
            operations: Array<{ source_refs: string[] }>;
          }>;
        };
        tampered.observations[0]!.operations[0]!.source_refs = ["turn:invented:user"];
        return tampered as unknown as T;
      }
    };

    await expect(new EpisodeSpanV3Reconstructor({ llm }).reconstruct({
      episodeId: "episode-travel",
      rawTurns: travelEpisodeTurns(),
      terminalReward: 1
    })).rejects.toThrow("invented source ref");
  });
});

function createEpisodeReconstructionLlm(
  calls: Array<{ messages: LlmMessage[]; options: LlmCompletionOptions }>
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/span-v3-test",
      model: "span-v3-test"
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
      const payload = JSON.parse(messages.at(-1)?.content ?? "{}") as Record<string, unknown>;
      if (options.operation === "span.v3.boundary.v1") {
        const candidates = payload.candidates as Array<{ id: string }>;
        return {
          decisions: candidates.map((candidate) => ({
            candidate_id: candidate.id,
            split: false,
            reason: "The evidence collection and execution still serve one local objective.",
            confidence: 0.9
          }))
        } as unknown as T;
      }
      if (options.operation === "span.v3.state_delta.v1") {
        return stateDeltaResponse(payload) as unknown as T;
      }
      throw new Error(`unexpected LLM operation: ${options.operation}`);
    },
    status() {
      return {
        provider: "host",
        model: "span-v3-test",
        configured: true,
        remote: false
      };
    }
  };
}

function stateDeltaResponse(payload: Record<string, unknown>): Record<string, unknown> {
  const observations = payload.observations as Array<{
    sourceId: string;
    rawTurnId: string;
  }>;
  const actionSegments = payload.actionSegments as Array<{
    segmentId: string;
    rawTurnId: string;
    eventRefs: string[];
    assistantSourceId: string;
  }>;
  return {
    observations: observations.map((observation) => ({
      source_id: observation.sourceId,
      operations: observationOperations(observation.rawTurnId, observation.sourceId)
    })),
    actions: actionSegments.map((segment) => ({
      segment_id: segment.segmentId,
      ...actionSemantics(segment)
    }))
  };
}

function observationOperations(rawTurnId: string, sourceRef: string): Array<Record<string, unknown>> {
  switch (rawTurnId) {
    case "raw-research":
      return [{
        op: "goal.set",
        subject: "task_goal",
        value: "准备青海旅行",
        status: "in_progress",
        source_refs: [sourceRef]
      }];
    case "raw-weather":
      return [{
        op: "goal.refine",
        subject: "task_goal",
        value: "了解九月天气",
        source_refs: [sourceRef]
      }];
    case "raw-plan":
      return [
        {
          op: "goal.refine",
          subject: "task_goal",
          value: "规划9月初青海行程",
          source_refs: [sourceRef]
        },
        {
          op: "constraint.upsert",
          subject: "trip_origin",
          value: "海南",
          source_refs: [sourceRef]
        },
        {
          op: "constraint.upsert",
          subject: "trip_duration",
          value: "两周",
          source_refs: [sourceRef]
        }
      ];
    case "raw-thanks":
      return [
        {
          op: "goal.complete",
          subject: "task_goal",
          source_refs: [sourceRef]
        },
        {
          op: "status.set",
          subject: "task_status",
          value: "completed",
          source_refs: [sourceRef]
        }
      ];
    default:
      return [];
  }
}

function actionSemantics(segment: {
  rawTurnId: string;
  eventRefs: string[];
  assistantSourceId: string;
}): Record<string, unknown> {
  const eventRef = segment.eventRefs[0] ?? segment.assistantSourceId;
  switch (segment.rawTurnId) {
    case "raw-research":
      return {
        include: true,
        local_goal: "收集旅行准备信息",
        summary: "检索并整理交通、季节与准备事项。",
        outcome_status: "success",
        verification_refs: [],
        operations: [{
          op: "fact.upsert",
          subject: "travel_preparation_guidance",
          value: "已收集",
          source_refs: [eventRef]
        }]
      };
    case "raw-weather":
      return {
        include: true,
        local_goal: "查询九月天气",
        summary: "查询九月初的历史天气和气候特征。",
        outcome_status: "success",
        verification_refs: [],
        operations: [{
          op: "fact.upsert",
          subject: "september_weather",
          value: "已查询",
          source_refs: [eventRef]
        }]
      };
    case "raw-plan":
      return {
        include: true,
        local_goal: "生成两周行程",
        summary: "按出发地、时间和时长生成行程。",
        outcome_status: "success",
        verification_refs: [segment.assistantSourceId],
        operations: [{
          op: "artifact.upsert",
          subject: "travel_itinerary",
          value: "两周行程方案",
          source_refs: [segment.assistantSourceId]
        }]
      };
    default:
      return {
        include: false,
        local_goal: "social acknowledgement",
        summary: "No substantive task effect.",
        outcome_status: "unknown",
        verification_refs: [],
        operations: []
      };
  }
}

function travelEpisodeTurns(): RawTurnRecord[] {
  return [
    makeRawTurn({
      id: "raw-research",
      turnId: "turn-1",
      createdAt: "2026-08-01T00:00:00.000Z",
      userText: "下个月我要去青海旅游，先帮我了解旅行准备。",
      assistantText: "我整理了交通、季节和行前准备信息。",
      toolCalls: [
        { name: "web_search", input: { q: "青海 旅行" }, output: "结果一" },
        { name: "web_search", input: { q: "青海 交通" }, output: "结果二" },
        { name: "web_fetch", input: { url: "https://example.test/a" }, output: "资料" },
        { name: "read_file", input: { path: "weather-skill.md" }, output: "说明" },
        { name: "exec", input: { command: "weather qinghai" }, output: "天气资料" }
      ]
    }),
    makeRawTurn({
      id: "raw-weather",
      turnId: "turn-2",
      createdAt: "2026-08-01T00:01:00.000Z",
      userText: "帮我查下9月的天气。",
      assistantText: "九月初温差较大，需要分层穿衣。",
      toolCalls: [
        { name: "exec", input: { command: "weather history" }, output: "历史数据" },
        { name: "exec", input: { command: "weather climate" }, output: "气候数据" }
      ]
    }),
    makeRawTurn({
      id: "raw-plan",
      turnId: "turn-3",
      createdAt: "2026-08-01T00:02:00.000Z",
      userText: "我准备9月初出行，从海南出发，帮我规划两周行程。",
      assistantText: "这里是一份从海南出发的两周青海行程。"
    }),
    makeRawTurn({
      id: "raw-thanks",
      turnId: "turn-4",
      createdAt: "2026-08-01T00:03:00.000Z",
      userText: "好的，感谢。",
      assistantText: "不客气。"
    })
  ];
}

function makeRawTurn(
  overrides: Partial<RawTurnRecord> & Pick<RawTurnRecord, "id">
): RawTurnRecord {
  return {
    sessionId: "session-travel",
    episodeId: "episode-travel",
    turnId: overrides.turnId ?? overrides.id,
    userId: "user-test",
    userText: overrides.userText ?? "do the task",
    assistantText: overrides.assistantText ?? "done",
    toolCalls: overrides.toolCalls ?? [],
    toolResults: overrides.toolResults ?? [],
    sourceMemoryIds: [],
    usage: {},
    status: overrides.status ?? "completed",
    createdAt: overrides.createdAt ?? "2026-08-01T00:00:00.000Z",
    ...overrides
  };
}
