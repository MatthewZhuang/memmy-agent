import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_MEMMY_CONFIG, type MemoryRow } from "../../../src/index.js";
import { Repositories } from "../../../src/storage/repositories.js";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const {
  cleanup,
  createTestService
} = createMemoryServiceFixture();

afterEach(cleanup);

describe("span cluster policy induction", () => {
  it("induces an L2 Policy from a ready Span bucket without signatures", async () => {
    const { db, service } = createTestService({
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    repos.memories.insert(spanMemory("span-a", "trace-a", "episode-a", "diagnose build failure", "read logs first"));
    repos.memories.insert(spanMemory("span-b", "trace-b", "episode-b", "diagnose build failure", "read logs first"));
    repos.spanClusters.replaceScopePartition({
      scopeId: "user-span-policy:codex:jiang",
      algorithmVersion: "span-cluster.v1",
      clusters: [{
        id: "cluster-policy",
        status: "ready",
        goalCentroid: [1, 0],
        policyCentroid: [1, 0],
        goalThreshold: 0.72,
        policyThreshold: 0.72,
        memberCount: 2,
        distinctSourceCount: 2,
        membershipVersion: "membership-policy",
        anchorSpanId: "span-a",
        members: [
          { spanId: "span-a", sourceTraceId: "trace-a", goalSimilarity: 1, policySimilarity: 1 },
          { spanId: "span-b", sourceTraceId: "trace-b", goalSimilarity: 1, policySimilarity: 1 }
        ]
      }],
      at: "2026-08-14T00:00:00.000Z"
    });
    repos.runtime.enqueueJob({
      id: "job-span-policy",
      jobType: "l2_induction",
      status: "queued",
      dedupeKey: "l2_induction:cluster-policy:membership-policy",
      userId: "user-span-policy",
      payload: {
        clusterId: "cluster-policy",
        membershipVersion: "membership-policy"
      },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-14T00:00:01.000Z",
      updatedAt: "2026-08-14T00:00:01.000Z"
    });

    const run = await service.runWorkerOnce(10);

    expect(run.jobs).toEqual(expect.arrayContaining([
      expect.objectContaining({
        jobId: "job-span-policy",
        jobType: "l2_induction",
        status: "succeeded"
      })
    ]));
    const policy = repos.memories.getByKey("L2", "policy:span-cluster:cluster-policy:span-policy.v1");
    expect(policy).toBeTruthy();
    expect(policy?.properties.internal_info.policy).toMatchObject({
      induction_version: "span-policy.v1",
      span_cluster_id: "cluster-policy",
      cluster_algorithm_version: "span-cluster.v1",
      cluster_membership_version: "membership-policy",
      source_span_ids: ["span-a", "span-b"],
      source_trace_ids: ["trace-a", "trace-b"],
      source_episode_ids: ["episode-a", "episode-b"]
    });
    expect(policy?.properties.internal_info.policy).not.toHaveProperty("signature");
    expect(repos.spanClusters.get("cluster-policy")).toMatchObject({
      status: "promoted",
      promotedPolicyId: policy?.id
    });
    expect(db.db.prepare(
      `SELECT l1_memory_id, l2_memory_id, relation
       FROM trace_policy_links
       ORDER BY l1_memory_id`
    ).all()).toEqual([
      { l1_memory_id: "span-a", l2_memory_id: policy?.id, relation: "supports_span_cluster" },
      { l1_memory_id: "span-b", l2_memory_id: policy?.id, relation: "supports_span_cluster" }
    ]);
    expect(db.db.prepare(`SELECT COUNT(*) AS count FROM l2_candidate_pool`).get()).toEqual({ count: 0 });
    db.close();
  });
});

function spanMemory(
  id: string,
  sourceTraceId: string,
  episodeId: string,
  goal: string,
  policy: string
): MemoryRow {
  const now = "2026-08-14T00:00:00.000Z";
  return {
    id,
    timeline: now,
    userId: "user-span-policy",
    agentId: "codex",
    memoryType: "LongTermMemory",
    status: "activated",
    visibility: "private",
    memoryKey: goal,
    memoryValue: `Goal: ${goal}\nPolicy: ${policy}\nSummary: verified ${id}`,
    tags: ["span", "big-turn", "derived"],
    info: {
      profile_id: "jiang",
      title: goal,
      summary: `verified ${id}`
    },
    properties: {
      internal_info: {
        memory_layer: "L1",
        memory_kind: "span",
        span: {
          schema_version: "span.v2",
          source_trace_id: sourceTraceId,
          raw_turn_id: `raw-${id}`,
          episode_id: episodeId,
          span_index: 0,
          tool_call_start: 0,
          tool_call_end: 3,
          tool_call_count: 4,
          goal,
          policy,
          summary: `verified ${id}`,
          derived: true
        }
      }
    },
    memoryLayer: "L1",
    version: 1,
    createdAt: now,
    updatedAt: now
  };
}
