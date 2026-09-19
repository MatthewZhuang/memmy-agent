import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_MEMMY_CONFIG } from "../../../src/index.js";
import {
  addNegativeFeedbackForTurn,
  addPositiveFeedbackForTurn,
  setL2ClusterFieldsForTest,
  setTraceValueForTest
} from "../../fixtures/evolution-fixture.js";
import { createCapturingL2Llm } from "./evolution-llm-stubs.js";
import { createMemoryServiceFixture, runWorkerRounds } from "../../fixtures/memory-service-fixture.js";

const { cleanup, createTestService } = createMemoryServiceFixture();
afterEach(cleanup);

function unit(x: number, y: number): number[] {
  const norm = Math.hypot(x, y);
  return [x / norm, y / norm];
}

function grayIntent(cosine: number): number[] {
  return unit(cosine, Math.sqrt(1 - cosine * cosine));
}

function clusterService() {
  const calls: Array<{
    messages: Array<{ role: string; content: string }>;
    options: { operation: string };
  }> = [];
  const llm = createCapturingL2Llm(calls);
  return {
    ...createTestService({
    llm,
    skillLlm: llm,
    config: {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          minEpisodesForInduction: 1,
          minGain: -1
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        },
        skill: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.skill,
          useLlm: false
        }
      }
    }
  }),
    calls
  };
}

async function captureEligibleTurns(
  service: ReturnType<typeof clusterService>["service"],
  db: ReturnType<typeof clusterService>["db"],
  userId: string,
  fields: Array<{
    suffix: string;
    intent: string;
    taskSummary: string;
    intentVec: number[];
    taskVec: number[];
    turnRole?: "local_subproblem" | "continuation";
  }>
): Promise<string[]> {
  const session = service.openSession({
    namespace: { source: "codex", profileId: "jiang", userId },
    workspaceId: `workspace-${userId}`
  });
  const ids: string[] = [];
  for (const field of fields) {
    const turn = service.completeTurn(`turn-${userId}-${field.suffix}`, {
      sessionId: session.sessionId,
      episodeId: `episode-${userId}-${field.suffix}`,
      query: field.intent || field.taskSummary,
      answer: "done"
    });
    setL2ClusterFieldsForTest(db, turn.l1MemoryId, field);
    await addPositiveFeedbackForTurn(service, session.sessionId, {
      episodeId: turn.episodeId,
      l1MemoryId: turn.l1MemoryId
    });
    ids.push(turn.l1MemoryId);
  }
  service.closeSession(session.sessionId);
  await runWorkerRounds(service, 8, 50);
  return ids;
}

function listClusters(db: ReturnType<typeof clusterService>["db"], userId: string) {
  return db.db.prepare(
    `SELECT c.id, c.seed_intent, c.member_count, c.l2_memory_id,
            (
              SELECT GROUP_CONCAT(m.l1_memory_id, ',')
              FROM l2_cluster_members m
              WHERE m.cluster_id = c.id
            ) AS member_ids
     FROM l2_clusters c
     WHERE c.user_id = ?
       AND EXISTS (
         SELECT 1 FROM l2_cluster_members m WHERE m.cluster_id = c.id
       )
     ORDER BY c.created_at, c.id`
  ).all(userId) as Array<{
    id: string;
    seed_intent: string;
    member_count: number;
    l2_memory_id: string | null;
    member_ids: string | null;
  }>;
}

describe("positive L2 clustering", () => {
  it("joins the same local intent across different tasks", async () => {
    const { db, service } = clusterService();
    const ids = await captureEligibleTurns(service, db, "cluster-intent-join", [
      {
        suffix: "a",
        intent: "消除查询中的N+1",
        taskSummary: "把这个项目跑通并做完列出的优化",
        intentVec: [1, 0],
        taskVec: [0, 1]
      },
      {
        suffix: "b",
        intent: "去掉列表接口的N+1",
        taskSummary: "写一篇旅行攻略",
        intentVec: grayIntent(0.8),
        taskVec: [1, 0]
      }
    ]);
    const clusters = listClusters(db, "cluster-intent-join");
    expect(clusters).toHaveLength(1);
    expect(clusters[0]?.member_count).toBe(2);
    expect(clusters[0]?.member_ids?.split(",").sort()).toEqual([...ids].sort());
    expect(db.db.prepare(
      `SELECT COUNT(*) AS count FROM memories WHERE user_id = ? AND memory_layer = 'L2'`
    ).get("cluster-intent-join")).toEqual({ count: 1 });
    db.close();
  });

  it("keeps different local intents apart even when the task is the same", async () => {
    const { db, service } = clusterService();
    await captureEligibleTurns(service, db, "cluster-task-only", [
      {
        suffix: "n1",
        intent: "消除查询中的N+1",
        taskSummary: "把这个项目跑通并做完列出的优化",
        intentVec: [1, 0],
        taskVec: [0, 1]
      },
      {
        suffix: "cache",
        intent: "给接口加配置缓存",
        taskSummary: "把这个项目跑通并做完列出的优化",
        intentVec: [0, 1],
        taskVec: [0, 1]
      }
    ]);
    const clusters = listClusters(db, "cluster-task-only");
    expect(clusters).toHaveLength(2);
    expect(clusters.map((cluster) => cluster.member_count)).toEqual([1, 1]);
    expect(db.db.prepare(
      `SELECT COUNT(*) AS count FROM memories WHERE user_id = ? AND memory_layer = 'L2'`
    ).get("cluster-task-only")).toEqual({ count: 2 });
    db.close();
  });

  it("joins the gray zone only when the task is also near", async () => {
    const { db, service } = clusterService();
    const sameTask = await captureEligibleTurns(service, db, "cluster-gray-join", [
      {
        suffix: "seed",
        intent: "消除查询中的N+1",
        taskSummary: "把这个项目跑通并做完列出的优化",
        intentVec: [1, 0],
        taskVec: [0, 1]
      },
      {
        suffix: "gray",
        intent: "减少重复查询",
        taskSummary: "把这个项目跑通并做完列出的优化",
        intentVec: grayIntent(0.55),
        taskVec: [0, 1]
      }
    ]);
    expect(listClusters(db, "cluster-gray-join")).toHaveLength(1);
    expect(listClusters(db, "cluster-gray-join")[0]?.member_ids?.split(",").sort()).toEqual([...sameTask].sort());

    await captureEligibleTurns(service, db, "cluster-gray-split", [
      {
        suffix: "seed",
        intent: "消除查询中的N+1",
        taskSummary: "把这个项目跑通并做完列出的优化",
        intentVec: [1, 0],
        taskVec: [0, 1]
      },
      {
        suffix: "gray",
        intent: "减少重复查询",
        taskSummary: "写一篇旅行攻略",
        intentVec: grayIntent(0.55),
        taskVec: [1, 0]
      }
    ]);
    expect(listClusters(db, "cluster-gray-split")).toHaveLength(2);
    db.close();
  });

  it("does not cluster continuation turns", async () => {
    const { db, service } = clusterService();
    const session = service.openSession({
      namespace: { source: "codex", profileId: "jiang", userId: "cluster-continuation" },
      workspaceId: "workspace-cluster-continuation"
    });
    const turn = service.completeTurn("turn-cluster-continuation-cont", {
      sessionId: session.sessionId,
      episodeId: "episode-cluster-continuation-cont",
      query: "继续帮我完成后续几个优化点",
      answer: "done"
    });
    await runWorkerRounds(service, 3, 50);
    setL2ClusterFieldsForTest(db, turn.l1MemoryId, {
      intent: "",
      taskSummary: "把这个项目跑通并做完列出的优化",
      intentVec: [1, 0],
      taskVec: [0, 1],
      turnRole: "continuation"
    });
    service.closeSession(session.sessionId);
    await runWorkerRounds(service, 6, 50);
    expect(listClusters(db, "cluster-continuation")).toHaveLength(0);
    expect(db.db.prepare(
      `SELECT COUNT(*) AS count FROM memories WHERE user_id = ? AND memory_layer = 'L2'`
    ).get("cluster-continuation")).toEqual({ count: 0 });
    db.close();
  });

  it("writes a cluster failure L2 from only negative local_subproblem turns", async () => {
    const { db, service } = clusterService();
    const session = service.openSession({
      namespace: { source: "codex", profileId: "jiang", userId: "cluster-pure-neg" },
      workspaceId: "workspace-cluster-pure-neg"
    });
    const turn = service.completeTurn("turn-cluster-pure-neg", {
      sessionId: session.sessionId,
      episodeId: "episode-cluster-pure-neg",
      query: "focused pytest still failed after retry",
      answer: "reran the whole suite"
    });
    setL2ClusterFieldsForTest(db, turn.l1MemoryId, {
      intent: "跑聚焦 pytest",
      taskSummary: "修迁移失败",
      intentVec: [1, 0],
      taskVec: [0, 1],
      value: -1
    });
    await addNegativeFeedbackForTurn(service, session.sessionId, {
      episodeId: turn.episodeId,
      l1MemoryId: turn.l1MemoryId
    });
    setTraceValueForTest(db, turn.l1MemoryId, -1);
    service.closeSession(session.sessionId);
    await runWorkerRounds(service, 8, 50);
    const positiveL2 = db.db.prepare(
      `SELECT COUNT(*) AS count
       FROM memories
       WHERE user_id = ?
         AND memory_layer = 'L2'
         AND instr(properties_json, '"experience_type":"success_pattern"') > 0`
    ).get("cluster-pure-neg") as { count: number };
    expect(positiveL2.count).toBe(0);
    const row = db.db.prepare(
      `SELECT id, memory_key, properties_json
       FROM memories
       WHERE user_id = ? AND memory_layer = 'L2'
       LIMIT 1`
    ).get("cluster-pure-neg") as { id: string; memory_key: string; properties_json: string } | undefined;
    expect(row?.memory_key.startsWith("policy:")).toBe(true);
    expect(row?.memory_key.startsWith("policy:avoid:")).toBe(false);
    const properties = JSON.parse(row!.properties_json) as {
      internal_info?: {
        source_repair_ids?: string[];
        policy?: {
          experience_type?: string;
          lesson_kind?: string;
          skill_eligible?: boolean;
          evidence_polarity?: string;
        };
      };
    };
    expect(properties.internal_info?.policy).toMatchObject({
      experience_type: "failure_avoidance",
      lesson_kind: "error_correction",
      skill_eligible: false,
      evidence_polarity: "negative"
    });
    expect(properties.internal_info?.source_repair_ids?.length).toBeGreaterThan(0);
    expect(listClusters(db, "cluster-pure-neg")[0]?.l2_memory_id).toBe(row!.id);
    db.close();
  });

  it("writes lesson_kind both from raw success and failure turns in one cluster", async () => {
    const { db, service, calls } = clusterService();
    const session = service.openSession({
      namespace: { source: "codex", profileId: "jiang", userId: "cluster-lesson-both" },
      workspaceId: "workspace-cluster-lesson-both"
    });
    const success = service.completeTurn("turn-cluster-lesson-ok", {
      sessionId: session.sessionId,
      episodeId: "episode-cluster-lesson-ok",
      query: "run the focused pytest for migration",
      answer: "ran pytest tests/a.py and it passed"
    });
    const failure = service.completeTurn("turn-cluster-lesson-fail", {
      sessionId: session.sessionId,
      episodeId: "episode-cluster-lesson-fail",
      query: "focused pytest still failed after retry",
      answer: "retried the whole suite and it failed"
    });
    setL2ClusterFieldsForTest(db, success.l1MemoryId, {
      intent: "跑聚焦 pytest",
      taskSummary: "修迁移失败",
      intentVec: [1, 0],
      taskVec: [0, 1]
    });
    setL2ClusterFieldsForTest(db, failure.l1MemoryId, {
      intent: "跑聚焦 pytest",
      taskSummary: "修迁移失败",
      intentVec: [1, 0],
      taskVec: [0, 1],
      value: -1
    });
    await addPositiveFeedbackForTurn(service, session.sessionId, {
      episodeId: success.episodeId,
      l1MemoryId: success.l1MemoryId
    });
    await addNegativeFeedbackForTurn(service, session.sessionId, {
      episodeId: failure.episodeId,
      l1MemoryId: failure.l1MemoryId
    });
    setTraceValueForTest(db, failure.l1MemoryId, -1);
    service.closeSession(session.sessionId);
    await runWorkerRounds(service, 8, 50);

    const row = db.db.prepare(
      `SELECT properties_json
       FROM memories
       WHERE user_id = ? AND memory_layer = 'L2'
         AND instr(properties_json, '"experience_type":"success_pattern"') > 0
       LIMIT 1`
    ).get("cluster-lesson-both") as { properties_json: string } | undefined;
    expect(row).toBeTruthy();
    const properties = JSON.parse(row!.properties_json) as {
      internal_info?: { policy?: { lesson_kind?: string; evidence_polarity?: string } };
    };
    expect(properties.internal_info?.policy?.lesson_kind).toBe("both");
    expect(properties.internal_info?.policy?.evidence_polarity).toBe("mixed");
    const induction = calls.find((call) => call.options.operation === "l2.induction.v6");
    expect(induction?.messages.some((message) => message.content.includes("RAW_TURNS"))).toBe(true);
    expect(induction?.messages.some((message) => message.content.includes("run the focused pytest for migration"))).toBe(true);
    expect(induction?.messages.some((message) => message.content.includes("polarity: failure"))).toBe(true);
    db.close();
  });

  it("evolves the same cluster L2 when a new member arrives", async () => {
    const { db, service, calls } = clusterService();
    await captureEligibleTurns(service, db, "cluster-evolve", [{
      suffix: "first",
      intent: "跑聚焦 pytest",
      taskSummary: "修迁移失败",
      intentVec: [1, 0],
      taskVec: [0, 1]
    }]);
    expect(calls.filter((call) => call.options.operation === "l2.induction.v6")).toHaveLength(1);
    const first = db.db.prepare(
      `SELECT id, processed_l1_ids_json
       FROM l2_clusters
       WHERE user_id = ?`
    ).get("cluster-evolve") as { id: string; processed_l1_ids_json: string };
    expect(JSON.parse(first.processed_l1_ids_json)).toHaveLength(1);

    await captureEligibleTurns(service, db, "cluster-evolve", [{
      suffix: "second",
      intent: "跑聚焦 pytest",
      taskSummary: "修迁移失败",
      intentVec: [1, 0],
      taskVec: [0, 1]
    }]);
    const inductionCalls = calls.filter((call) => call.options.operation === "l2.induction.v6");
    expect(inductionCalls).toHaveLength(2);
    expect(inductionCalls[1]?.messages.some((message) => message.content.includes("MODE: evolve"))).toBe(true);
    expect(inductionCalls[1]?.messages.some((message) => message.content.includes("EXISTING_POLICY"))).toBe(true);
    const clusters = listClusters(db, "cluster-evolve");
    expect(clusters).toHaveLength(1);
    expect(db.db.prepare(
      `SELECT COUNT(*) AS count FROM memories WHERE user_id = ? AND memory_layer = 'L2'`
    ).get("cluster-evolve")).toEqual({ count: 1 });
    const processed = db.db.prepare(
      `SELECT processed_l1_ids_json FROM l2_clusters WHERE id = ?`
    ).get(first.id) as { processed_l1_ids_json: string };
    expect(JSON.parse(processed.processed_l1_ids_json)).toHaveLength(2);
    db.close();
  });

  it("upgrades the same cluster L2 when a later positive L1 arrives", async () => {
    const { db, service, calls } = clusterService();
    const session = service.openSession({
      namespace: { source: "codex", profileId: "jiang", userId: "cluster-neg-upgrade" },
      workspaceId: "workspace-cluster-neg-upgrade"
    });
    const failure = service.completeTurn("turn-cluster-neg-upgrade-fail", {
      sessionId: session.sessionId,
      episodeId: "episode-cluster-neg-upgrade-fail",
      query: "focused pytest still failed after retry",
      answer: "retried the whole suite and it failed"
    });
    setL2ClusterFieldsForTest(db, failure.l1MemoryId, {
      intent: "跑聚焦 pytest",
      taskSummary: "修迁移失败",
      intentVec: [1, 0],
      taskVec: [0, 1],
      value: -1
    });
    await addNegativeFeedbackForTurn(service, session.sessionId, {
      episodeId: failure.episodeId,
      l1MemoryId: failure.l1MemoryId
    });
    setTraceValueForTest(db, failure.l1MemoryId, -1);
    service.closeSession(session.sessionId);
    await runWorkerRounds(service, 8, 50);

    const afterNeg = db.db.prepare(
      `SELECT id, memory_key, properties_json
       FROM memories
       WHERE user_id = ? AND memory_layer = 'L2'`
    ).all("cluster-neg-upgrade") as Array<{ id: string; memory_key: string; properties_json: string }>;
    expect(afterNeg).toHaveLength(1);
    expect(afterNeg[0]!.memory_key.startsWith("policy:avoid:")).toBe(false);
    expect(afterNeg[0]!.properties_json).toContain('"evidence_polarity":"negative"');
    const failurePolicyId = afterNeg[0]!.id;

    await captureEligibleTurns(service, db, "cluster-neg-upgrade", [{
      suffix: "ok",
      intent: "跑聚焦 pytest",
      taskSummary: "修迁移失败",
      intentVec: [1, 0],
      taskVec: [0, 1]
    }]);

    const layers = db.db.prepare(
      `SELECT id, status, properties_json
       FROM memories
       WHERE user_id = ? AND memory_layer = 'L2'`
    ).all("cluster-neg-upgrade") as Array<{ id: string; status: string; properties_json: string }>;
    expect(layers).toHaveLength(1);
    expect(layers[0]!.id).toBe(failurePolicyId);
    const properties = JSON.parse(layers[0]!.properties_json) as {
      internal_info?: { policy?: { lesson_kind?: string; evidence_polarity?: string; skill_eligible?: boolean } };
    };
    expect(properties.internal_info?.policy?.lesson_kind).toBe("both");
    expect(properties.internal_info?.policy?.evidence_polarity).toBe("mixed");
    expect(properties.internal_info?.policy?.skill_eligible).toBe(true);
    expect(listClusters(db, "cluster-neg-upgrade")).toHaveLength(1);
    expect(calls.some((call) =>
      call.options.operation === "l2.induction.v6"
      && call.messages.some((message) => message.content.includes("EXISTING_POLICY"))
    )).toBe(true);
    expect(calls.some((call) =>
      call.options.operation === "l2.induction.v6"
      && call.messages.some((message) => message.content.includes("META-POLICY"))
    )).toBe(true);
    db.close();
  });
});
