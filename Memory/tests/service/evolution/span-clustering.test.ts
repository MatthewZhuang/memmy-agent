import { afterEach, describe, expect, it } from "vitest";
import { Repositories } from "../../../src/storage/repositories.js";
import { attachMemoryVector } from "../../../src/storage/memory-vector-state.js";
import { DEFAULT_MEMMY_CONFIG, type MemoryRow } from "../../../src/index.js";
import type { LlmClient } from "../../../src/model/types.js";
import {
  buildSpanPartition,
  findSpanClusterAuditCandidates,
  SPAN_CLUSTER_ALGORITHM_VERSION,
  type ClusterableSpan
} from "../../../src/service/evolution/span-clustering.js";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const {
  cleanup,
  createTestService
} = createMemoryServiceFixture();

afterEach(cleanup);

describe("span clustering", () => {
  it("joins only when both Goal and observed Policy similarities pass thresholds", () => {
    const input = [
      span("anchor", "trace-a", [1, 0], [1, 0], 1),
      span("same-goal-same-policy", "trace-b", [0.9, 0.1], [0.9, 0.1], 2),
      span("same-goal-different-policy", "trace-c", [0.9, 0.1], [0, 1], 3),
      span("different-goal-same-policy", "trace-d", [0, 1], [0.9, 0.1], 4)
    ];

    const clusters = buildSpanPartition({
      scopeId: "scope-a",
      spans: input,
      goalThreshold: 0.72,
      policyThreshold: 0.72
    });

    expect(clusters).toHaveLength(3);
    expect(clusters[0]?.members.map((member) => member.spanId)).toEqual([
      "anchor",
      "same-goal-same-policy"
    ]);
    expect(clusters[1]?.members.map((member) => member.spanId)).toEqual(["same-goal-different-policy"]);
    expect(clusters[2]?.members.map((member) => member.spanId)).toEqual(["different-goal-same-policy"]);
  });

  it("is deterministic under shuffled input and uses stable bucket ids", () => {
    const spans = [
      span("b", "trace-b", [1, 0], [1, 0], 2),
      span("a", "trace-a", [1, 0], [1, 0], 1),
      span("c", "trace-c", [0, 1], [0, 1], 3)
    ];

    const ordered = buildSpanPartition({
      scopeId: "scope-a",
      spans,
      goalThreshold: 0.72,
      policyThreshold: 0.72
    });
    const shuffled = buildSpanPartition({
      scopeId: "scope-a",
      spans: [spans[2]!, spans[0]!, spans[1]!],
      goalThreshold: 0.72,
      policyThreshold: 0.72
    });

    expect(shuffled).toEqual(ordered);
    expect(ordered[0]?.id).toBe(`span_cluster_${SPAN_CLUSTER_ALGORITHM_VERSION}_scope-a_a`);
    expect(ordered[1]?.id).toBe(`span_cluster_${SPAN_CLUSTER_ALGORITHM_VERSION}_scope-a_c`);
  });

  it("breaks compatible bucket ties by min similarity, then goal, policy, and id", () => {
    const spans = [
      span("anchor-a", "trace-a", [1, 0], [1, 0], 1),
      span("anchor-b", "trace-b", [0, 1], [0, 1], 2),
      span("candidate", "trace-c", [0.8, 0.6], [0.8, 0.6], 3)
    ];

    const clusters = buildSpanPartition({
      scopeId: "scope-a",
      spans,
      goalThreshold: 0.7,
      policyThreshold: 0.7
    });

    expect(clusters[0]?.members.map((member) => member.spanId)).toEqual(["anchor-a", "candidate"]);
    expect(clusters[1]?.members.map((member) => member.spanId)).toEqual(["anchor-b"]);
  });

  it("counts repeated source Traces once", () => {
    const clusters = buildSpanPartition({
      scopeId: "scope-a",
      spans: [
        span("a", "trace-a", [1, 0], [1, 0], 1),
        span("b", "trace-a", [1, 0], [1, 0], 2),
        span("c", "trace-b", [1, 0], [1, 0], 3)
      ],
      goalThreshold: 0.72,
      policyThreshold: 0.72
    });

    expect(clusters).toHaveLength(1);
    expect(clusters[0]).toMatchObject({
      memberCount: 3,
      distinctSourceCount: 2
    });
  });

  it("reports cohesion and flags large low-cohesion buckets for audit", () => {
    const tight = buildSpanPartition({
      scopeId: "scope-a",
      spans: [
        span("tight-a", "trace-a", [1, 0], [1, 0], 1),
        span("tight-b", "trace-b", [0.99, 0.01], [0.99, 0.01], 2),
        span("tight-c", "trace-c", [0.98, 0.02], [0.98, 0.02], 3)
      ],
      goalThreshold: 0.7,
      policyThreshold: 0.7
    });
    const broad = buildSpanPartition({
      scopeId: "scope-a",
      spans: [
        span("broad-a", "trace-a", [1, 0], [1, 0], 1),
        span("broad-b", "trace-b", [0.98, 0.02], [0.98, 0.02], 2),
        span("broad-c", "trace-c", [0.55, 0.83], [0.55, 0.83], 3)
      ],
      goalThreshold: 0.5,
      policyThreshold: 0.5
    });

    expect(findSpanClusterAuditCandidates(tight, {
      auditMinMembers: 3,
      auditCohesionThreshold: 0.85,
      auditWeakPairLimit: 1
    })).toEqual([]);
    expect(findSpanClusterAuditCandidates(broad, {
      auditMinMembers: 3,
      auditCohesionThreshold: 0.85,
      auditWeakPairLimit: 1
    })).toEqual([
      expect.objectContaining({
        clusterId: broad[0]!.id,
        memberCount: 3,
        pairCount: 3,
        minPairScore: expect.any(Number),
        p10PairScore: expect.any(Number),
        weakestPairs: [
          expect.objectContaining({
            spanIds: ["broad-a", "broad-c"]
          })
        ]
      })
    ]);
  });

  it("rebuilds one scope from ready Span vectors and enqueues induction for ready buckets", async () => {
    const { db, service } = createTestService({
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            minDistinctSources: 2
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const scopeId = "user-cluster:codex:jiang";
    repos.memories.insert(vectorSpanMemory("span-a", "trace-a", [1, 0], [1, 0]));
    repos.memories.insert(vectorSpanMemory("span-b", "trace-b", [0.95, 0.05], [0.95, 0.05]));
    repos.memories.insert({
      ...vectorSpanMemory("span-other-scope", "trace-other", [1, 0], [1, 0]),
      userId: "other-user"
    });
    repos.runtime.enqueueJob({
      id: "job-span-cluster-scope",
      jobType: "span_cluster",
      status: "queued",
      dedupeKey: `span_cluster:${scopeId}:${SPAN_CLUSTER_ALGORITHM_VERSION}`,
      userId: "user-cluster",
      payload: { scopeId, algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-14T00:00:10.000Z",
      updatedAt: "2026-08-14T00:00:10.000Z"
    });

    const run = await service.runWorkerOnce(10);

    expect(run.jobs).toEqual(expect.arrayContaining([
      expect.objectContaining({
        jobId: "job-span-cluster-scope",
        jobType: "span_cluster",
        status: "succeeded"
      })
    ]));
    const clusters = repos.spanClusters.listByScope(scopeId, SPAN_CLUSTER_ALGORITHM_VERSION);
    expect(clusters).toHaveLength(1);
    expect(clusters[0]).toMatchObject({
      status: "ready",
      memberCount: 2,
      distinctSourceCount: 2
    });
    expect(repos.spanClusters.listByScope("other-user:codex:jiang", SPAN_CLUSTER_ALGORITHM_VERSION))
      .toHaveLength(0);
    expect(db.db.prepare(
      `SELECT job_type, json_extract(payload_json, '$.clusterId') AS cluster_id
       FROM evolution_jobs
       WHERE job_type = 'l2_induction'`
    ).all()).toEqual([
      {
        job_type: "l2_induction",
        cluster_id: clusters[0]!.id
      }
    ]);
    db.close();
  });

  it("associates a new Span to an existing promoted bucket policy without signatures", async () => {
    const { db, service } = createTestService({
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            minDistinctSources: 2
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const scopeId = "user-cluster:codex:jiang";
    repos.memories.insert(policyMemory("policy-existing"));
    repos.memories.insert(vectorSpanMemory("span-a", "trace-a", [1, 0], [1, 0]));
    repos.memories.insert(vectorSpanMemory("span-b", "trace-b", [0.95, 0.05], [0.95, 0.05]));
    repos.spanClusters.replaceScopePartition({
      scopeId,
      algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION,
      clusters: [{
        ...clusterInputForTest("span_cluster_span-cluster.v1_user-cluster:codex:jiang_span-a", ["span-a", "span-b"]),
        status: "promoted",
        promotedPolicyId: "policy-existing"
      }],
      at: "2026-08-14T00:00:00.000Z"
    });
    repos.memories.insert(vectorSpanMemory("span-c", "trace-c", [0.9, 0.1], [0.9, 0.1]));
    repos.runtime.enqueueJob({
      id: "job-span-cluster-promoted",
      jobType: "span_cluster",
      status: "queued",
      userId: "user-cluster",
      payload: { scopeId, algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-14T00:00:10.000Z",
      updatedAt: "2026-08-14T00:00:10.000Z"
    });

    await service.runWorkerOnce(10);

    expect(repos.spanClusters.get("span_cluster_span-cluster.v1_user-cluster:codex:jiang_span-a"))
      .toMatchObject({
        status: "promoted",
        promotedPolicyId: "policy-existing",
        memberCount: 3
      });
    expect(db.db.prepare(
      `SELECT l1_memory_id, l2_memory_id, relation
       FROM trace_policy_links
       WHERE l2_memory_id = 'policy-existing'
       ORDER BY l1_memory_id`
    ).all()).toEqual([
      { l1_memory_id: "span-a", l2_memory_id: "policy-existing", relation: "supports_span_cluster" },
      { l1_memory_id: "span-b", l2_memory_id: "policy-existing", relation: "supports_span_cluster" },
      { l1_memory_id: "span-c", l2_memory_id: "policy-existing", relation: "supports_span_cluster" }
    ]);
    expect(db.db.prepare(`SELECT COUNT(*) AS count FROM evolution_jobs WHERE job_type = 'l2_induction'`).get())
      .toEqual({ count: 0 });
    db.close();
  });

  it("enqueues an audit job before induction for ready buckets with low pairwise cohesion", async () => {
    const { db, service } = createTestService({
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            goalSimilarityThreshold: 0.5,
            policySimilarityThreshold: 0.5,
            minDistinctSources: 2,
            auditMinMembers: 3,
            auditCohesionThreshold: 0.85,
            auditWeakPairLimit: 1
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const scopeId = "user-cluster:codex:jiang";
    repos.memories.insert(vectorSpanMemory("span-a", "trace-a", [1, 0], [1, 0]));
    repos.memories.insert(vectorSpanMemory("span-b", "trace-b", [0.98, 0.02], [0.98, 0.02]));
    repos.memories.insert(vectorSpanMemory("span-c", "trace-c", [0.55, 0.83], [0.55, 0.83]));
    repos.runtime.enqueueJob({
      id: "job-span-cluster-audit",
      jobType: "span_cluster",
      status: "queued",
      userId: "user-cluster",
      payload: { scopeId, algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-14T00:00:10.000Z",
      updatedAt: "2026-08-14T00:00:10.000Z"
    });

    await service.runWorkerOnce(10);

    const jobs = db.db.prepare(
      `SELECT job_type, payload_json FROM evolution_jobs WHERE job_type IN ('span_cluster_audit', 'l2_induction') ORDER BY created_at, id`
    ).all() as Array<{ job_type: string; payload_json: string }>;
    expect(jobs.map((job) => job.job_type)).toEqual(["span_cluster_audit"]);
    expect(JSON.parse(jobs[0]!.payload_json)).toMatchObject({
      reason: "span_cluster.low_cohesion",
      clusterId: expect.stringContaining("span-a"),
      audit: {
        memberCount: 3,
        weakestPairs: [
          expect.objectContaining({
            spanIds: ["span-a", "span-c"]
          })
        ]
      }
    });
    db.close();
  });

  it("uses LLM to judge low-cohesion audit jobs and records split groups", async () => {
    const calls: Array<{ messages: Array<{ role: string; content: string }>; options: { operation: string } }> = [];
    const auditLlm = createAuditLlm(calls, {
      should_split: true,
      reason: "Two policy themes are present.",
      groups: [
        { label: "tight pair", span_ids: ["span-a", "span-b"] },
        { label: "drift member", span_ids: ["span-c"] }
      ]
    });
    const { db, service } = createTestService({
      skillLlm: auditLlm,
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            goalSimilarityThreshold: 0.5,
            policySimilarityThreshold: 0.5,
            minDistinctSources: 2,
            auditMinMembers: 3,
            auditCohesionThreshold: 0.85,
            auditWeakPairLimit: 1
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const scopeId = "user-cluster:codex:jiang";
    repos.memories.insert(vectorSpanMemory("span-a", "trace-a", [1, 0], [1, 0]));
    repos.memories.insert(vectorSpanMemory("span-b", "trace-b", [0.98, 0.02], [0.98, 0.02]));
    repos.memories.insert(vectorSpanMemory("span-c", "trace-c", [0.55, 0.83], [0.55, 0.83]));
    repos.runtime.enqueueJob({
      id: "job-span-cluster-audit-source",
      jobType: "span_cluster",
      status: "queued",
      userId: "user-cluster",
      payload: { scopeId, algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-14T00:00:10.000Z",
      updatedAt: "2026-08-14T00:00:10.000Z"
    });

    await service.runWorkerOnce(10);
    await service.runWorkerOnce(10);

    expect(calls).toEqual([
      expect.objectContaining({
        options: expect.objectContaining({ operation: "span_cluster.audit.v1" })
      })
    ]);
    const auditJob = db.db.prepare(
      `SELECT id, payload_json FROM evolution_jobs WHERE job_type = 'span_cluster_audit'`
    ).get() as { id: string; payload_json: string };
    const auditPayload = JSON.parse(auditJob.payload_json) as { clusterId: string; membershipVersion: string };
    expect(repos.runtime.getKv(`span_cluster_audit:${auditPayload.clusterId}:${auditPayload.membershipVersion}`)?.value)
      .toMatchObject({
        should_split: true,
        groups: [
          { label: "tight pair", span_ids: ["span-a", "span-b"] },
          { label: "drift member", span_ids: ["span-c"] }
        ]
      });
    const rebuiltClusters = repos.spanClusters.listByScope(scopeId, SPAN_CLUSTER_ALGORITHM_VERSION);
    expect(rebuiltClusters.map((cluster) => cluster.memberCount).sort()).toEqual([1, 2]);
    expect(rebuiltClusters.find((cluster) => cluster.memberCount === 2)?.status).toBe("ready");
    db.close();
  });
});

function span(
  id: string,
  sourceTraceId: string,
  vecGoal: number[],
  vecPolicy: number[],
  order: number
): ClusterableSpan {
  return {
    id,
    sourceTraceId,
    createdAt: `2026-08-14T00:00:0${order}.000Z`,
    vecGoal,
    vecPolicy
  };
}

function vectorSpanMemory(
  id: string,
  sourceTraceId: string,
  vecGoal: number[],
  vecPolicy: number[]
): MemoryRow {
  const now = "2026-08-14T00:00:00.000Z";
  const base: MemoryRow = {
    id,
    timeline: now,
    userId: "user-cluster",
    agentId: "codex",
    memoryType: "LongTermMemory",
    status: "activated",
    visibility: "private",
    memoryKey: id,
    memoryValue: `Goal: ${id}\nPolicy: ${id}\nSummary: ${id}`,
    tags: ["span", "big-turn", "derived"],
    info: { profile_id: "jiang" },
    properties: {
      internal_info: {
        memory_layer: "L1",
        memory_kind: "span",
        span: {
          schema_version: "span.v2",
          source_trace_id: sourceTraceId,
          raw_turn_id: `raw-${id}`,
          span_index: 0,
          tool_call_start: 0,
          tool_call_end: 3,
          tool_call_count: 4,
          goal: id,
          policy: id,
          summary: id,
          derived: true
        }
      }
    },
    memoryLayer: "L1",
    version: 1,
    createdAt: now,
    updatedAt: now
  };
  return attachMemoryVector(attachMemoryVector(base, {
    vectorField: "vec_goal",
    vector: vecGoal,
    embeddingModel: "test",
    embeddingProvider: "test"
  }), {
    vectorField: "vec_policy",
    vector: vecPolicy,
    embeddingModel: "test",
    embeddingProvider: "test"
  });
}

function clusterInputForTest(id: string, spanIds: string[]) {
  return {
    id,
    status: "ready" as const,
    goalCentroid: [1, 0],
    policyCentroid: [1, 0],
    goalThreshold: 0.72,
    policyThreshold: 0.72,
    memberCount: spanIds.length,
    distinctSourceCount: spanIds.length,
    membershipVersion: "promoted-membership",
    anchorSpanId: spanIds[0]!,
    members: spanIds.map((spanId) => ({
      spanId,
      sourceTraceId: `trace-${spanId.slice(-1)}`,
      goalSimilarity: 1,
      policySimilarity: 1
    }))
  };
}

function policyMemory(id: string): MemoryRow {
  const now = "2026-08-14T00:00:00.000Z";
  return {
    id,
    timeline: now,
    userId: "user-cluster",
    agentId: "codex",
    memoryType: "LongTermMemory",
    status: "activated",
    visibility: "private",
    memoryKey: "policy-existing",
    memoryValue: "Existing policy",
    tags: ["policy"],
    info: { profile_id: "jiang" },
    properties: {
      internal_info: {
        memory_layer: "L2",
        memory_kind: "policy",
        policy: {
          title: "Existing policy",
          trigger: "Existing policy",
          procedure: "Use existing policy",
          verification: "Verify",
          boundary: "Boundary",
          status: "active"
        }
      }
    },
    memoryLayer: "L2",
    version: 1,
    createdAt: now,
    updatedAt: now
  };
}

function createAuditLlm(
  calls: Array<{ messages: Array<{ role: string; content: string }>; options: { operation: string } }>,
  result: Record<string, unknown>
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/audit",
      model: "audit-test"
    },
    isConfigured() {
      return true;
    },
    async complete() {
      return "unused";
    },
    async completeJson<T extends Record<string, unknown>>(
      messages: Array<{ role: "system" | "user" | "assistant"; content: string }>,
      options: { operation: string }
    ): Promise<T> {
      calls.push({ messages, options });
      return result as T;
    },
    status() {
      return {
        provider: "host",
        model: "audit-test",
        configured: true,
        remote: false
      };
    }
  };
}
