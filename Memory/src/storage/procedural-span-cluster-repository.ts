import type Database from "better-sqlite3";
import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";

export const PROCEDURAL_SPAN_CLUSTER_ALGORITHM_VERSION = "procedural-span-cluster.v1" as const;

export type ProceduralSpanEvidenceRole = "support" | "counterexample";
export type ProceduralSpanClusterStatus = "forming" | "ready" | "promoted" | "stale";

export interface ProceduralSpanClusterRecord {
  id: string;
  namespaceId: string;
  algorithmVersion: string;
  status: ProceduralSpanClusterStatus;
  memberCount: number;
  distinctEpisodeCount: number;
  distinctSupportEpisodeCount: number;
  membershipVersion: string;
  anchorOccurrenceId: string;
  activePolicyVersionId?: string | null;
  clusterBasis: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface ProceduralSpanClusterMemberRecord {
  clusterId: string;
  occurrenceId: string;
  algorithmVersion: string;
  episodeId: string;
  evidenceRole: ProceduralSpanEvidenceRole;
  similarity: number;
  createdAt: string;
}

export interface UpsertProceduralSpanClusterInput {
  id: string;
  namespaceId: string;
  algorithmVersion?: string;
  minDistinctSupportEpisodes: number;
  members: Array<{
    occurrenceId: string;
    evidenceRole: ProceduralSpanEvidenceRole;
    similarity: number;
  }>;
  evidenceVersion?: string;
  clusterBasis?: Record<string, unknown>;
  at: string;
}

export interface PrepareProceduralSpanClusterPartitionInput {
  namespaceId: string;
  algorithmVersion: string;
  retainedClusterIds: readonly string[];
  at: string;
}

interface ClusterSqlRow {
  id: string;
  namespace_id: string;
  algorithm_version: string;
  status: ProceduralSpanClusterStatus;
  member_count: number;
  distinct_episode_count: number;
  distinct_support_episode_count: number;
  membership_version: string;
  anchor_occurrence_id: string;
  active_policy_version_id: string | null;
  cluster_basis_json: string;
  created_at: string;
  updated_at: string;
}

interface ClusterMemberSqlRow {
  cluster_id: string;
  occurrence_id: string;
  algorithm_version: string;
  episode_id: string;
  evidence_role: ProceduralSpanEvidenceRole;
  similarity: number;
  created_at: string;
}

interface ActiveOccurrenceRow {
  id: string;
  episode_id: string;
  namespace_id: string;
  path_status: string;
}

export class ProceduralSpanClusterRepository {
  constructor(private readonly db: Database.Database) {}

  upsert(input: UpsertProceduralSpanClusterInput): ProceduralSpanClusterRecord {
    const namespaceId = input.namespaceId.trim();
    if (!namespaceId) throw new Error("procedural Span cluster requires namespaceId");
    if (!Number.isInteger(input.minDistinctSupportEpisodes) || input.minDistinctSupportEpisodes < 2) {
      throw new Error("procedural Span cluster requires at least two distinct support Episodes");
    }
    const algorithmVersion = input.algorithmVersion ?? PROCEDURAL_SPAN_CLUSTER_ALGORITHM_VERSION;
    const members = [...input.members]
      .sort((left, right) => left.occurrenceId.localeCompare(right.occurrenceId));
    if (members.length === 0) throw new Error("procedural Span cluster cannot be empty");
    if (new Set(members.map((member) => member.occurrenceId)).size !== members.length) {
      throw new Error("procedural Span cluster contains duplicate occurrences");
    }
    for (const member of members) {
      if (!Number.isFinite(member.similarity) || member.similarity < 0 || member.similarity > 1) {
        throw new Error(`procedural Span cluster similarity is invalid: ${member.occurrenceId}`);
      }
    }
    const occurrences = members.map((member) => this.activeOccurrence(member.occurrenceId));
    for (const occurrence of occurrences) {
      if (!occurrence || occurrence.path_status !== "active") {
        throw new Error("procedural Span cluster can only use occurrences from active Episode paths");
      }
      if (occurrence.namespace_id !== namespaceId) {
        throw new Error(`procedural Span cluster crosses namespace: ${occurrence.id}`);
      }
    }
    const episodeByOccurrence = new Map(occurrences.map((occurrence) => [occurrence!.id, occurrence!.episode_id]));
    const distinctEpisodeCount = new Set(occurrences.map((occurrence) => occurrence!.episode_id)).size;
    const distinctSupportEpisodeCount = new Set(members
      .filter((member) => member.evidenceRole === "support")
      .map((member) => episodeByOccurrence.get(member.occurrenceId)!)).size;
    const membershipVersion = stableHash({
      algorithmVersion,
      namespaceId,
      ...(input.evidenceVersion ? { evidenceVersion: input.evidenceVersion } : {}),
      members: members.map((member) => ({
        occurrenceId: member.occurrenceId,
        episodeId: episodeByOccurrence.get(member.occurrenceId),
        evidenceRole: member.evidenceRole,
        similarity: roundSimilarity(member.similarity)
      }))
    }).slice(0, 32);
    const previous = this.get(input.id);
    const membershipChanged = Boolean(previous && previous.membershipVersion !== membershipVersion);
    const status: ProceduralSpanClusterStatus = previous?.membershipVersion === membershipVersion &&
      previous.status === "promoted" && previous.activePolicyVersionId
      ? "promoted"
      : distinctSupportEpisodeCount >= input.minDistinctSupportEpisodes ? "ready" : "forming";
    const anchorOccurrenceId = previous && members.some((member) =>
      member.occurrenceId === previous.anchorOccurrenceId && member.evidenceRole === "support"
    )
      ? previous.anchorOccurrenceId
      : members.find((member) => member.evidenceRole === "support")?.occurrenceId ??
        members[0]!.occurrenceId;

    this.db.transaction(() => {
      if (membershipChanged) this.invalidateActivePolicy(input.id, input.at);
      this.db.prepare(
        `INSERT INTO procedural_span_clusters (
          id, namespace_id, algorithm_version, status, member_count,
          distinct_episode_count, distinct_support_episode_count,
          membership_version, anchor_occurrence_id, active_policy_version_id,
          cluster_basis_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          namespace_id = excluded.namespace_id,
          algorithm_version = excluded.algorithm_version,
          status = excluded.status,
          member_count = excluded.member_count,
          distinct_episode_count = excluded.distinct_episode_count,
          distinct_support_episode_count = excluded.distinct_support_episode_count,
          membership_version = excluded.membership_version,
          anchor_occurrence_id = excluded.anchor_occurrence_id,
          active_policy_version_id = CASE
            WHEN procedural_span_clusters.membership_version = excluded.membership_version
              THEN procedural_span_clusters.active_policy_version_id
            ELSE NULL
          END,
          cluster_basis_json = excluded.cluster_basis_json,
          updated_at = excluded.updated_at`
      ).run(
        input.id,
        namespaceId,
        algorithmVersion,
        status,
        members.length,
        distinctEpisodeCount,
        distinctSupportEpisodeCount,
        membershipVersion,
        anchorOccurrenceId,
        previous?.membershipVersion === membershipVersion ? previous.activePolicyVersionId ?? null : null,
        toJson(input.clusterBasis ?? {}),
        previous?.createdAt ?? input.at,
        input.at
      );
      this.db.prepare(
        `DELETE FROM procedural_span_cluster_members WHERE cluster_id = ?`
      ).run(input.id);
      const insertMember = this.db.prepare(
        `INSERT INTO procedural_span_cluster_members (
          cluster_id, occurrence_id, algorithm_version, episode_id,
          evidence_role, similarity, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`
      );
      for (const member of members) {
        insertMember.run(
          input.id,
          member.occurrenceId,
          algorithmVersion,
          episodeByOccurrence.get(member.occurrenceId),
          member.evidenceRole,
          roundSimilarity(member.similarity),
          input.at
        );
      }
    })();
    return this.get(input.id)!;
  }

  get(id: string): ProceduralSpanClusterRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_span_clusters WHERE id = ?`
    ).get(id) as ClusterSqlRow | undefined;
    return row ? clusterFromSql(row) : undefined;
  }

  listByNamespace(
    namespaceId: string,
    algorithmVersion: string = PROCEDURAL_SPAN_CLUSTER_ALGORITHM_VERSION
  ): ProceduralSpanClusterRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_span_clusters
       WHERE namespace_id = ? AND algorithm_version = ?
       ORDER BY created_at ASC, id ASC`
    ).all(namespaceId, algorithmVersion) as ClusterSqlRow[]).map(clusterFromSql);
  }

  listMembers(clusterId: string): ProceduralSpanClusterMemberRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_span_cluster_members
       WHERE cluster_id = ?
       ORDER BY occurrence_id ASC`
    ).all(clusterId) as ClusterMemberSqlRow[]).map(memberFromSql);
  }

  getMemberForOccurrence(
    occurrenceId: string,
    algorithmVersion: string = PROCEDURAL_SPAN_CLUSTER_ALGORITHM_VERSION
  ): ProceduralSpanClusterMemberRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_span_cluster_members
       WHERE occurrence_id = ? AND algorithm_version = ?
       LIMIT 1`
    ).get(occurrenceId, algorithmVersion) as ClusterMemberSqlRow | undefined;
    return row ? memberFromSql(row) : undefined;
  }

  listMembersForEpisode(
    episodeId: string,
    algorithmVersion: string = PROCEDURAL_SPAN_CLUSTER_ALGORITHM_VERSION
  ): ProceduralSpanClusterMemberRecord[] {
    return (this.db.prepare(
      `SELECT members.*
       FROM procedural_span_cluster_members AS members
       WHERE members.episode_id = ? AND members.algorithm_version = ?
       ORDER BY members.cluster_id ASC, members.occurrence_id ASC`
    ).all(episodeId, algorithmVersion) as ClusterMemberSqlRow[]).map(memberFromSql);
  }

  removeMembers(occurrenceIds: readonly string[], algorithmVersion: string): string[] {
    const ids = [...new Set(occurrenceIds)];
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => "?").join(", ");
    const affected = (this.db.prepare(
      `SELECT DISTINCT cluster_id
       FROM procedural_span_cluster_members
       WHERE occurrence_id IN (${placeholders}) AND algorithm_version = ?
       ORDER BY cluster_id ASC`
    ).all(...ids, algorithmVersion) as Array<{ cluster_id: string }>).map((row) => row.cluster_id);
    this.db.prepare(
      `DELETE FROM procedural_span_cluster_members
       WHERE occurrence_id IN (${placeholders}) AND algorithm_version = ?`
    ).run(...ids, algorithmVersion);
    return affected;
  }

  clearAndMarkStale(clusterId: string, at: string): void {
    this.invalidateActivePolicy(clusterId, at);
    this.db.prepare(
      `DELETE FROM procedural_span_cluster_members WHERE cluster_id = ?`
    ).run(clusterId);
    this.db.prepare(
      `UPDATE procedural_span_clusters
       SET status = 'stale', member_count = 0, distinct_episode_count = 0,
           distinct_support_episode_count = 0, active_policy_version_id = NULL,
           updated_at = ?
       WHERE id = ?`
    ).run(at, clusterId);
  }

  prepareNamespacePartition(input: PrepareProceduralSpanClusterPartitionInput): void {
    const retained = new Set(input.retainedClusterIds);
    const existing = this.listByNamespace(input.namespaceId, input.algorithmVersion);
    for (const cluster of existing) {
      if (retained.has(cluster.id)) continue;
      this.invalidateActivePolicy(cluster.id, input.at);
      this.db.prepare(
        `UPDATE procedural_span_clusters
         SET status = 'stale', active_policy_version_id = NULL, updated_at = ?
         WHERE id = ?`
      ).run(input.at, cluster.id);
    }
    this.db.prepare(
      `DELETE FROM procedural_span_cluster_members
       WHERE cluster_id IN (
         SELECT id FROM procedural_span_clusters
         WHERE namespace_id = ? AND algorithm_version = ?
       )`
    ).run(input.namespaceId, input.algorithmVersion);
  }

  markPromoted(
    clusterId: string,
    membershipVersion: string,
    policyVersionId: string,
    at: string
  ): boolean {
    const result = this.db.prepare(
      `UPDATE procedural_span_clusters
       SET status = 'promoted', active_policy_version_id = ?, updated_at = ?
       WHERE id = ? AND membership_version = ?`
    ).run(policyVersionId, at, clusterId, membershipVersion);
    return result.changes === 1;
  }

  private activeOccurrence(id: string): ActiveOccurrenceRow | undefined {
    return this.db.prepare(
      `SELECT occurrences.id, occurrences.episode_id, occurrences.namespace_id,
              paths.status AS path_status
       FROM procedural_span_occurrences AS occurrences
       JOIN episode_procedural_paths AS paths ON paths.id = occurrences.path_id
       WHERE occurrences.id = ?`
    ).get(id) as ActiveOccurrenceRow | undefined;
  }

  private invalidateActivePolicy(clusterId: string, at: string): void {
    this.db.prepare(
      `UPDATE memories
       SET status = 'archived',
           properties_json = json_set(properties_json, '$.status', 'archived'),
           version = version + 1,
           updated_at = ?
       WHERE id IN (
         SELECT l2_memory_id FROM procedural_policy_versions
         WHERE cluster_id = ? AND status = 'active' AND l2_memory_id IS NOT NULL
       ) AND status != 'deleted'`
    ).run(at, clusterId);
    this.db.prepare(
      `UPDATE episode_policy_projections
       SET status = 'inactive', deactivated_at = ?
       WHERE status = 'active'
         AND path_id IN (
           SELECT path_id FROM procedural_policy_occurrences
           WHERE status = 'active'
             AND policy_version_id IN (
               SELECT id FROM procedural_policy_versions
               WHERE cluster_id = ? AND status = 'active'
             )
         )`
    ).run(at, clusterId);
    this.db.prepare(
      `UPDATE procedural_policy_occurrences
       SET status = 'superseded', superseded_at = ?
       WHERE status = 'active'
         AND policy_version_id IN (
           SELECT id FROM procedural_policy_versions
           WHERE cluster_id = ? AND status = 'active'
         )`
    ).run(at, clusterId);
    this.db.prepare(
      `UPDATE procedural_policy_versions
       SET status = 'inactive', deactivated_at = ?
       WHERE cluster_id = ? AND status = 'active'`
    ).run(at, clusterId);
  }
}

function clusterFromSql(row: ClusterSqlRow): ProceduralSpanClusterRecord {
  return {
    id: row.id,
    namespaceId: row.namespace_id,
    algorithmVersion: row.algorithm_version,
    status: row.status,
    memberCount: row.member_count,
    distinctEpisodeCount: row.distinct_episode_count,
    distinctSupportEpisodeCount: row.distinct_support_episode_count,
    membershipVersion: row.membership_version,
    anchorOccurrenceId: row.anchor_occurrence_id,
    activePolicyVersionId: row.active_policy_version_id,
    clusterBasis: parseJson<Record<string, unknown>>(row.cluster_basis_json, {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function memberFromSql(row: ClusterMemberSqlRow): ProceduralSpanClusterMemberRecord {
  return {
    clusterId: row.cluster_id,
    occurrenceId: row.occurrence_id,
    algorithmVersion: row.algorithm_version,
    episodeId: row.episode_id,
    evidenceRole: row.evidence_role,
    similarity: row.similarity,
    createdAt: row.created_at
  };
}

function roundSimilarity(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
