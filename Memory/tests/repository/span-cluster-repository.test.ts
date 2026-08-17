import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { MemoryDb } from "../../src/index.js";
import { Repositories } from "../../src/storage/repositories.js";

describe("repository span cluster persistence", () => {
  it("replaces one scope partition without changing another scope", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-span-clusters-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);

      repos.spanClusters.replaceScopePartition({
        scopeId: "scope-a",
        algorithmVersion: "span-cluster.v1",
        clusters: [{
          id: "cluster-a",
          status: "forming",
          goalCentroid: [1, 0],
          policyCentroid: [0, 1],
          goalThreshold: 0.72,
          policyThreshold: 0.72,
          memberCount: 1,
          distinctSourceCount: 1,
          membershipVersion: "membership-a",
          anchorSpanId: "span-a",
          members: [{
            spanId: "span-a",
            sourceTraceId: "trace-a",
            goalSimilarity: 1,
            policySimilarity: 1
          }]
        }],
        at: "2026-08-14T00:00:00.000Z"
      });
      repos.spanClusters.replaceScopePartition({
        scopeId: "scope-b",
        algorithmVersion: "span-cluster.v1",
        clusters: [{
          id: "cluster-b",
          status: "ready",
          goalCentroid: [0, 1],
          policyCentroid: [1, 0],
          goalThreshold: 0.72,
          policyThreshold: 0.72,
          memberCount: 1,
          distinctSourceCount: 1,
          membershipVersion: "membership-b",
          anchorSpanId: "span-b",
          members: [{
            spanId: "span-b",
            sourceTraceId: "trace-b",
            goalSimilarity: 1,
            policySimilarity: 1
          }]
        }],
        at: "2026-08-14T00:00:01.000Z"
      });

      repos.spanClusters.replaceScopePartition({
        scopeId: "scope-a",
        algorithmVersion: "span-cluster.v1",
        clusters: [{
          id: "cluster-a2",
          status: "ready",
          goalCentroid: [0.9, 0.1],
          policyCentroid: [0.1, 0.9],
          goalThreshold: 0.72,
          policyThreshold: 0.72,
          memberCount: 1,
          distinctSourceCount: 1,
          membershipVersion: "membership-a2",
          anchorSpanId: "span-a2",
          members: [{
            spanId: "span-a2",
            sourceTraceId: "trace-a2",
            goalSimilarity: 1,
            policySimilarity: 1
          }]
        }],
        at: "2026-08-14T00:00:02.000Z"
      });

      expect(repos.spanClusters.listByScope("scope-a", "span-cluster.v1").map((cluster) => cluster.id))
        .toEqual(["cluster-a2"]);
      expect(repos.spanClusters.listByScope("scope-b", "span-cluster.v1").map((cluster) => cluster.id))
        .toEqual(["cluster-b"]);
      expect(repos.spanClusters.listMembers("cluster-a2").map((member) => member.spanId))
        .toEqual(["span-a2"]);
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("prevents one Span from being active in two buckets for the same algorithm", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-span-cluster-unique-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);

      expect(() => repos.spanClusters.replaceScopePartition({
        scopeId: "scope-a",
        algorithmVersion: "span-cluster.v1",
        clusters: [
          clusterInput("cluster-a", "span-shared"),
          clusterInput("cluster-b", "span-shared")
        ],
        at: "2026-08-14T00:00:00.000Z"
      })).toThrow(/active member of more than one Span cluster/);
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("marks clusters promoted and deletes memberships without deleting Span memories", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-span-cluster-members-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);
      repos.memories.insert(spanMemory("span-a"));
      repos.spanClusters.replaceScopePartition({
        scopeId: "scope-a",
        algorithmVersion: "span-cluster.v1",
        clusters: [clusterInput("cluster-a", "span-a")],
        at: "2026-08-14T00:00:00.000Z"
      });

      expect(repos.spanClusters.markPromoted(
        "cluster-a",
        "membership-cluster-a",
        "policy-a",
        "2026-08-14T00:00:01.000Z"
      )).toBe(true);
      expect(repos.spanClusters.get("cluster-a")).toMatchObject({
        status: "promoted",
        promotedPolicyId: "policy-a"
      });
      expect(repos.spanClusters.deleteMembershipsForSpans(["span-a"])).toBe(1);
      expect(repos.spanClusters.listMembers("cluster-a")).toEqual([]);
      expect(repos.memories.get("span-a")?.id).toBe("span-a");
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

function clusterInput(id: string, spanId: string) {
  return {
    id,
    status: "forming" as const,
    goalCentroid: [1, 0],
    policyCentroid: [0, 1],
    goalThreshold: 0.72,
    policyThreshold: 0.72,
    memberCount: 1,
    distinctSourceCount: 1,
    membershipVersion: `membership-${id}`,
    anchorSpanId: spanId,
    members: [{
      spanId,
      sourceTraceId: `trace-${spanId}`,
      goalSimilarity: 1,
      policySimilarity: 1
    }]
  };
}

function spanMemory(id: string) {
  const now = "2026-08-14T00:00:00.000Z";
  return {
    id,
    timeline: now,
    userId: "span-cluster-user",
    memoryType: "LongTermMemory",
    status: "activated" as const,
    visibility: "private",
    memoryKey: id,
    memoryValue: "Goal: cluster\nPolicy: cluster\nSummary: cluster",
    tags: ["span"],
    info: {},
    properties: {
      internal_info: {
        memory_layer: "L1" as const,
        memory_kind: "span" as const,
        span: {
          schema_version: "span.v2",
          source_trace_id: `trace-${id}`,
          raw_turn_id: `raw-${id}`,
          span_index: 0,
          tool_call_start: 0,
          tool_call_end: 3,
          tool_call_count: 4,
          goal: "cluster",
          policy: "cluster",
          summary: "cluster",
          derived: true
        }
      }
    },
    memoryLayer: "L1" as const,
    version: 1,
    createdAt: now,
    updatedAt: now
  };
}
