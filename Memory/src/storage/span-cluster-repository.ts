import type Database from "better-sqlite3";
import { parseJson } from "../utils/json.js";

export const SPAN_CLUSTER_ALGORITHM_VERSION = "span-cluster.v1" as const;

export type SpanClusterAlgorithmVersion = typeof SPAN_CLUSTER_ALGORITHM_VERSION;
export type SpanClusterStatus = "forming" | "ready" | "promoted" | "stale";

export interface SpanClusterRecord {
  id: string;
  scopeId: string;
  algorithmVersion: SpanClusterAlgorithmVersion;
  status: SpanClusterStatus;
  goalCentroid: number[];
  policyCentroid: number[];
  goalThreshold: number;
  policyThreshold: number;
  memberCount: number;
  distinctSourceCount: number;
  membershipVersion: string;
  promotedPolicyId?: string | null;
  anchorSpanId: string;
  createdAt: string;
  updatedAt: string;
}

export interface SpanClusterMemberRecord {
  clusterId: string;
  spanId: string;
  sourceTraceId: string;
  goalSimilarity: number;
  policySimilarity: number;
  createdAt: string;
}

export interface SpanClusterInput {
  id: string;
  status: SpanClusterStatus;
  goalCentroid: number[];
  policyCentroid: number[];
  goalThreshold: number;
  policyThreshold: number;
  memberCount: number;
  distinctSourceCount: number;
  membershipVersion: string;
  promotedPolicyId?: string | null;
  anchorSpanId: string;
  members: Array<{
    spanId: string;
    sourceTraceId: string;
    goalSimilarity: number;
    policySimilarity: number;
  }>;
}

export interface ReplaceSpanClusterPartitionInput {
  scopeId: string;
  algorithmVersion: SpanClusterAlgorithmVersion;
  clusters: SpanClusterInput[];
  at: string;
}

interface SpanClusterSqlRow {
  id: string;
  scope_id: string;
  algorithm_version: SpanClusterAlgorithmVersion;
  status: SpanClusterStatus;
  goal_threshold: number;
  policy_threshold: number;
  goal_centroid_json: string;
  policy_centroid_json: string;
  member_count: number;
  distinct_source_count: number;
  membership_version: string;
  promoted_policy_id: string | null;
  anchor_span_id: string;
  created_at: string;
  updated_at: string;
}

interface SpanClusterMemberSqlRow {
  cluster_id: string;
  span_id: string;
  source_trace_id: string;
  goal_similarity: number;
  policy_similarity: number;
  created_at: string;
}

export class SpanClusterRepository {
  constructor(private readonly db: Database.Database) {}

  replaceScopePartition(input: ReplaceSpanClusterPartitionInput): {
    changedClusterIds: string[];
    staleClusterIds: string[];
  } {
    const seen = new Set<string>();
    for (const cluster of input.clusters) {
      for (const member of cluster.members) {
        const key = `${input.algorithmVersion}:${member.spanId}`;
        if (seen.has(key)) {
          throw new Error(`Span ${member.spanId} would be an active member of more than one Span cluster`);
        }
        seen.add(key);
      }
    }

    const previous = this.listByScope(input.scopeId, input.algorithmVersion);
    const staleClusterIds = previous
      .map((cluster) => cluster.id)
      .filter((id) => !input.clusters.some((cluster) => cluster.id === id));

    this.db.transaction(() => {
      const existingRows = this.db.prepare(
        `SELECT id, created_at
         FROM span_clusters
         WHERE scope_id = ? AND algorithm_version = ?`
      ).all(input.scopeId, input.algorithmVersion) as Array<{ id: string; created_at: string }>;
      const createdAtById = new Map(existingRows.map((row) => [row.id, row.created_at]));
      const ids = input.clusters.map((cluster) => cluster.id);
      if (ids.length > 0) {
        this.db.prepare(
          `DELETE FROM span_cluster_members
           WHERE cluster_id IN (SELECT CAST(value AS TEXT) FROM json_each(?))`
        ).run(JSON.stringify(ids));
      }
      this.db.prepare(
        `DELETE FROM span_clusters
         WHERE scope_id = ? AND algorithm_version = ?`
      ).run(input.scopeId, input.algorithmVersion);

      for (const cluster of input.clusters) {
        this.db.prepare(
          `INSERT INTO span_clusters (
            id, scope_id, algorithm_version, status, goal_threshold, policy_threshold,
            goal_centroid_json, policy_centroid_json, member_count, distinct_source_count,
            membership_version, promoted_policy_id, anchor_span_id, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        ).run(
          cluster.id,
          input.scopeId,
          input.algorithmVersion,
          cluster.status,
          cluster.goalThreshold,
          cluster.policyThreshold,
          JSON.stringify(cluster.goalCentroid),
          JSON.stringify(cluster.policyCentroid),
          cluster.memberCount,
          cluster.distinctSourceCount,
          cluster.membershipVersion,
          cluster.promotedPolicyId ?? null,
          cluster.anchorSpanId,
          createdAtById.get(cluster.id) ?? input.at,
          input.at
        );
        for (const member of cluster.members) {
          this.db.prepare(
            `INSERT INTO span_cluster_members (
              cluster_id, span_id, algorithm_version, source_trace_id,
              goal_similarity, policy_similarity, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)`
          ).run(
            cluster.id,
            member.spanId,
            input.algorithmVersion,
            member.sourceTraceId,
            member.goalSimilarity,
            member.policySimilarity,
            input.at
          );
        }
      }
    })();

    return {
      changedClusterIds: input.clusters.map((cluster) => cluster.id),
      staleClusterIds
    };
  }

  get(id: string): SpanClusterRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM span_clusters WHERE id = ?`
    ).get(id) as SpanClusterSqlRow | undefined;
    return row ? spanClusterFromSql(row) : undefined;
  }

  listByScope(
    scopeId: string,
    algorithmVersion: SpanClusterAlgorithmVersion
  ): SpanClusterRecord[] {
    return (this.db.prepare(
      `SELECT *
       FROM span_clusters
       WHERE scope_id = ? AND algorithm_version = ?
       ORDER BY created_at ASC, id ASC`
    ).all(scopeId, algorithmVersion) as SpanClusterSqlRow[]).map(spanClusterFromSql);
  }

  listMembers(clusterId: string): SpanClusterMemberRecord[] {
    return (this.db.prepare(
      `SELECT cluster_id, span_id, source_trace_id, goal_similarity, policy_similarity, created_at
       FROM span_cluster_members
       WHERE cluster_id = ?
       ORDER BY span_id ASC`
    ).all(clusterId) as SpanClusterMemberSqlRow[]).map(spanClusterMemberFromSql);
  }

  markPromoted(
    clusterId: string,
    membershipVersion: string,
    policyId: string,
    at: string
  ): boolean {
    const result = this.db.prepare(
      `UPDATE span_clusters
       SET status = 'promoted',
           promoted_policy_id = ?,
           updated_at = ?
       WHERE id = ?
         AND membership_version = ?`
    ).run(policyId, at, clusterId, membershipVersion);
    return result.changes > 0;
  }

  deleteMembershipsForSpans(spanIds: readonly string[]): number {
    if (spanIds.length === 0) return 0;
    const result = this.db.prepare(
      `DELETE FROM span_cluster_members
       WHERE span_id IN (SELECT CAST(value AS TEXT) FROM json_each(?))`
    ).run(JSON.stringify(spanIds));
    return result.changes;
  }
}

function spanClusterFromSql(row: SpanClusterSqlRow): SpanClusterRecord {
  return {
    id: row.id,
    scopeId: row.scope_id,
    algorithmVersion: row.algorithm_version,
    status: row.status,
    goalCentroid: parseNumberArray(row.goal_centroid_json),
    policyCentroid: parseNumberArray(row.policy_centroid_json),
    goalThreshold: row.goal_threshold,
    policyThreshold: row.policy_threshold,
    memberCount: row.member_count,
    distinctSourceCount: row.distinct_source_count,
    membershipVersion: row.membership_version,
    promotedPolicyId: row.promoted_policy_id,
    anchorSpanId: row.anchor_span_id,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function spanClusterMemberFromSql(row: SpanClusterMemberSqlRow): SpanClusterMemberRecord {
  return {
    clusterId: row.cluster_id,
    spanId: row.span_id,
    sourceTraceId: row.source_trace_id,
    goalSimilarity: row.goal_similarity,
    policySimilarity: row.policy_similarity,
    createdAt: row.created_at
  };
}

function parseNumberArray(value: string): number[] {
  return parseJson<unknown[]>(value, [])
    .filter((item): item is number => typeof item === "number" && Number.isFinite(item));
}
