import type Database from "better-sqlite3";
import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";
import type { ProceduralSpanEvidenceRole } from "./procedural-span-cluster-repository.js";
import {
  PROCEDURAL_POLICY_INDUCTION_VERSION,
  PROCEDURAL_POLICY_SCHEMA_VERSION,
  type ProceduralPolicyV1
} from "../service/evolution/procedural-policy-model.js";

export type ProceduralPolicyVersionStatus = "active" | "inactive";
export type ProceduralPolicyOccurrenceStatus = "active" | "superseded";

export interface ProceduralPolicyVersionRecord {
  id: string;
  policyKey: string;
  namespaceId: string;
  clusterId: string;
  clusterMembershipVersion: string;
  schemaVersion: typeof PROCEDURAL_POLICY_SCHEMA_VERSION;
  inductionVersion: typeof PROCEDURAL_POLICY_INDUCTION_VERSION;
  status: ProceduralPolicyVersionStatus;
  title: string;
  confidence: number;
  evidenceHash: string;
  evidenceOccurrenceIds: string[];
  compilerModel?: string;
  l2MemoryId?: string;
  policy: ProceduralPolicyV1;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface ProceduralPolicyOccurrenceRecord {
  id: string;
  policyVersionId: string;
  policyKey: string;
  occurrenceId: string;
  pathId: string;
  spanId: string;
  episodeId: string;
  sessionId: string;
  clusterMembershipVersion: string;
  evidenceRole: ProceduralSpanEvidenceRole;
  matchConfidence: number;
  status: ProceduralPolicyOccurrenceStatus;
  createdAt: string;
  supersededAt?: string | null;
}

export interface SaveProceduralPolicyVersionInput {
  policy: ProceduralPolicyV1;
  l2MemoryId: string;
  occurrences: Array<{
    occurrenceId: string;
    pathId: string;
    spanId: string;
    episodeId: string;
    sessionId: string;
    evidenceRole: ProceduralSpanEvidenceRole;
    matchConfidence: number;
  }>;
  at: string;
}

interface PolicySqlRow {
  id: string;
  policy_key: string;
  namespace_id: string;
  cluster_id: string;
  cluster_membership_version: string;
  schema_version: string;
  induction_version: string;
  status: ProceduralPolicyVersionStatus;
  title: string;
  confidence: number;
  evidence_hash: string;
  evidence_occurrence_ids_json: string;
  compiler_model: string | null;
  l2_memory_id: string | null;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface PolicyOccurrenceSqlRow {
  id: string;
  policy_version_id: string;
  policy_key: string;
  occurrence_id: string;
  path_id: string;
  span_id: string;
  episode_id: string;
  session_id: string;
  cluster_membership_version: string;
  evidence_role: ProceduralSpanEvidenceRole;
  match_confidence: number;
  status: ProceduralPolicyOccurrenceStatus;
  created_at: string;
  superseded_at: string | null;
}

export class ProceduralPolicyRepository {
  constructor(private readonly db: Database.Database) {}

  saveAndActivate(input: SaveProceduralPolicyVersionInput): ProceduralPolicyVersionRecord {
    validatePolicy(input.policy);
    const cluster = this.db.prepare(
      `SELECT namespace_id, membership_version
       FROM procedural_span_clusters WHERE id = ?`
    ).get(input.policy.clusterId) as { namespace_id: string; membership_version: string } | undefined;
    if (!cluster || cluster.namespace_id !== input.policy.namespaceId ||
        cluster.membership_version !== input.policy.clusterMembershipVersion) {
      throw new Error(`procedural Policy cluster membership is stale: ${input.policy.clusterId}`);
    }
    const clusterMembers = this.db.prepare(
      `SELECT members.occurrence_id, members.evidence_role, members.similarity,
              occurrences.path_id, occurrences.span_id, occurrences.episode_id,
              paths.session_id
       FROM procedural_span_cluster_members AS members
       JOIN procedural_span_occurrences AS occurrences
         ON occurrences.id = members.occurrence_id
       JOIN episode_procedural_paths AS paths
         ON paths.id = occurrences.path_id
       WHERE members.cluster_id = ?
       ORDER BY members.occurrence_id`
    ).all(input.policy.clusterId) as Array<{
      occurrence_id: string;
      evidence_role: ProceduralSpanEvidenceRole;
      similarity: number;
      path_id: string;
      span_id: string;
      episode_id: string;
      session_id: string;
    }>;
    const expectedIds = clusterMembers.map((member) => member.occurrence_id);
    const actualIds = [...input.occurrences]
      .map((occurrence) => occurrence.occurrenceId)
      .sort((left, right) => left.localeCompare(right));
    if (JSON.stringify(actualIds) !== JSON.stringify(expectedIds)) {
      throw new Error("procedural Policy occurrence mapping does not match cluster membership");
    }
    const supportIds = clusterMembers
      .filter((member) => member.evidence_role === "support")
      .map((member) => member.occurrence_id);
    if (!sameStrings(input.policy.evidence.occurrenceIds, expectedIds) ||
        !sameStrings(input.policy.evidence.supportOccurrenceIds, supportIds)) {
      throw new Error("procedural Policy evidence does not match cluster membership");
    }
    if (input.policy.evidenceOccurrenceIds.some((id) => !supportIds.includes(id))) {
      throw new Error("procedural Policy selected evidence must be positive support");
    }
    const existing = this.getByClusterMembership(
      input.policy.clusterId,
      input.policy.clusterMembershipVersion,
      input.policy.provenance.inductionVersion
    );
    if (existing) {
      if (existing.id !== input.policy.id || existing.policy.contentHash !== input.policy.contentHash) {
        throw new Error("procedural Policy membership already has a different compiled version");
      }
      this.activateVersion(existing.id, input.at);
      return this.get(existing.id)!;
    }

    const clusterMemberById = new Map(clusterMembers.map((member) => [member.occurrence_id, member]));
    this.db.transaction(() => {
      this.supersedeActive(input.policy.clusterId, input.at);
      this.deactivateActiveProjectionsForPaths(
        input.occurrences.map((occurrence) => occurrence.pathId),
        input.at
      );
      this.db.prepare(
        `INSERT INTO procedural_policy_versions (
          id, policy_key, namespace_id, cluster_id, cluster_membership_version,
          schema_version, induction_version, status, title, confidence,
          evidence_hash, evidence_occurrence_ids_json, compiler_model,
          l2_memory_id, payload_json, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)`
      ).run(
        input.policy.id,
        input.policy.policyKey,
        input.policy.namespaceId,
        input.policy.clusterId,
        input.policy.clusterMembershipVersion,
        input.policy.schemaVersion,
        input.policy.provenance.inductionVersion,
        input.policy.title,
        input.policy.confidence,
        input.policy.provenance.evidenceHash,
        toJson(input.policy.evidenceOccurrenceIds),
        input.policy.provenance.model ?? null,
        input.l2MemoryId,
        toJson(input.policy),
        input.at,
        input.at
      );
      const insertOccurrence = this.db.prepare(
        `INSERT INTO procedural_policy_occurrences (
          id, policy_version_id, policy_key, occurrence_id, path_id, span_id,
          episode_id, session_id, cluster_membership_version, evidence_role,
          match_confidence, status, created_at, superseded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL)`
      );
      for (const occurrence of [...input.occurrences]
        .sort((left, right) => left.occurrenceId.localeCompare(right.occurrenceId))) {
        const clusterMember = clusterMemberById.get(occurrence.occurrenceId)!;
        if (clusterMember.evidence_role !== occurrence.evidenceRole ||
            Math.abs(clusterMember.similarity - occurrence.matchConfidence) > 1e-6 ||
            clusterMember.path_id !== occurrence.pathId ||
            clusterMember.span_id !== occurrence.spanId ||
            clusterMember.episode_id !== occurrence.episodeId ||
            clusterMember.session_id !== occurrence.sessionId) {
          throw new Error(`procedural Policy occurrence evidence mismatch: ${occurrence.occurrenceId}`);
        }
        insertOccurrence.run(
          `policy_occurrence_${stableHash({
            policyVersionId: input.policy.id,
            occurrenceId: occurrence.occurrenceId
          }).slice(0, 20)}`,
          input.policy.id,
          input.policy.policyKey,
          occurrence.occurrenceId,
          occurrence.pathId,
          occurrence.spanId,
          occurrence.episodeId,
          occurrence.sessionId,
          input.policy.clusterMembershipVersion,
          occurrence.evidenceRole,
          occurrence.matchConfidence,
          input.at
        );
      }
      const promoted = this.db.prepare(
        `UPDATE procedural_span_clusters
         SET status = 'promoted', active_policy_version_id = ?, updated_at = ?
         WHERE id = ? AND membership_version = ?`
      ).run(
        input.policy.id,
        input.at,
        input.policy.clusterId,
        input.policy.clusterMembershipVersion
      );
      if (promoted.changes !== 1) {
        throw new Error(`failed to promote procedural Span cluster: ${input.policy.clusterId}`);
      }
    })();
    return this.get(input.policy.id)!;
  }

  get(id: string): ProceduralPolicyVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_policy_versions WHERE id = ?`
    ).get(id) as PolicySqlRow | undefined;
    return row ? policyFromSql(row) : undefined;
  }

  getByClusterMembership(
    clusterId: string,
    membershipVersion: string,
    inductionVersion = PROCEDURAL_POLICY_INDUCTION_VERSION
  ): ProceduralPolicyVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_policy_versions
       WHERE cluster_id = ? AND cluster_membership_version = ? AND induction_version = ?
       LIMIT 1`
    ).get(clusterId, membershipVersion, inductionVersion) as PolicySqlRow | undefined;
    return row ? policyFromSql(row) : undefined;
  }

  getActiveForCluster(clusterId: string): ProceduralPolicyVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_policy_versions
       WHERE cluster_id = ? AND status = 'active'
       LIMIT 1`
    ).get(clusterId) as PolicySqlRow | undefined;
    return row ? policyFromSql(row) : undefined;
  }

  listOccurrences(policyVersionId: string): ProceduralPolicyOccurrenceRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_policy_occurrences
       WHERE policy_version_id = ?
       ORDER BY occurrence_id ASC`
    ).all(policyVersionId) as PolicyOccurrenceSqlRow[]).map(policyOccurrenceFromSql);
  }

  listActiveOccurrencesForPath(pathId: string): ProceduralPolicyOccurrenceRecord[] {
    return (this.db.prepare(
      `SELECT occurrences.*
       FROM procedural_policy_occurrences AS occurrences
       JOIN procedural_policy_versions AS policies
         ON policies.id = occurrences.policy_version_id
       WHERE occurrences.path_id = ?
         AND occurrences.status = 'active'
         AND policies.status = 'active'
       ORDER BY occurrences.occurrence_id ASC, occurrences.policy_version_id ASC`
    ).all(pathId) as PolicyOccurrenceSqlRow[]).map(policyOccurrenceFromSql);
  }

  activateVersion(policyVersionId: string, at: string): ProceduralPolicyVersionRecord {
    const existing = this.get(policyVersionId);
    if (!existing) throw new Error(`procedural Policy version not found: ${policyVersionId}`);
    const cluster = this.db.prepare(
      `SELECT membership_version FROM procedural_span_clusters WHERE id = ?`
    ).get(existing.clusterId) as { membership_version: string } | undefined;
    if (!cluster || cluster.membership_version !== existing.clusterMembershipVersion) {
      throw new Error(`procedural Policy version does not match active cluster membership: ${policyVersionId}`);
    }
    if (existing.status === "active") return existing;
    this.db.transaction(() => {
      this.supersedeActive(existing.clusterId, at);
      this.deactivateActiveProjectionsForPaths(
        this.listOccurrences(policyVersionId).map((occurrence) => occurrence.pathId),
        at
      );
      this.db.prepare(
        `UPDATE procedural_policy_versions
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(at, policyVersionId);
      this.db.prepare(
        `UPDATE procedural_policy_occurrences
         SET status = 'active', superseded_at = NULL
         WHERE policy_version_id = ?`
      ).run(policyVersionId);
      this.db.prepare(
        `UPDATE procedural_span_clusters
         SET status = 'promoted', active_policy_version_id = ?, updated_at = ?
         WHERE id = ? AND membership_version = ?`
      ).run(policyVersionId, at, existing.clusterId, existing.clusterMembershipVersion);
      if (existing.l2MemoryId) {
        this.db.prepare(
          `UPDATE memories
           SET status = 'resolving',
               properties_json = json_set(properties_json, '$.status', 'resolving'),
               version = version + 1,
               updated_at = ?
           WHERE id = ? AND status = 'archived'`
        ).run(at, existing.l2MemoryId);
      }
    })();
    return this.get(policyVersionId)!;
  }

  private supersedeActive(clusterId: string, at: string): void {
    this.archiveActivePolicyMemories(clusterId, at);
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

  private archiveActivePolicyMemories(clusterId: string, at: string): void {
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
  }

  private deactivateActiveProjectionsForPaths(pathIds: readonly string[], at: string): void {
    const update = this.db.prepare(
      `UPDATE episode_policy_projections
       SET status = 'inactive', deactivated_at = ?
       WHERE path_id = ? AND status = 'active'`
    );
    for (const pathId of [...new Set(pathIds)]) update.run(at, pathId);
  }
}

function validatePolicy(policy: ProceduralPolicyV1): void {
  if (policy.schemaVersion !== PROCEDURAL_POLICY_SCHEMA_VERSION ||
      policy.provenance.inductionVersion !== PROCEDURAL_POLICY_INDUCTION_VERSION) {
    throw new Error(`unsupported procedural Policy schema: ${policy.id}`);
  }
  if (policy.id !== `procedural_policy_${stableHash({
    schemaVersion: policy.schemaVersion,
    inductionVersion: policy.provenance.inductionVersion,
    clusterId: policy.clusterId,
    membershipVersion: policy.clusterMembershipVersion,
    evidenceHash: policy.provenance.evidenceHash,
    contentHash: policy.contentHash
  }).slice(0, 20)}`) {
    throw new Error(`procedural Policy integrity check failed: ${policy.id}`);
  }
}

function policyFromSql(row: PolicySqlRow): ProceduralPolicyVersionRecord {
  const policy = parseJson<ProceduralPolicyV1 | null>(row.payload_json, null);
  if (!policy || policy.id !== row.id || policy.policyKey !== row.policy_key ||
      policy.clusterId !== row.cluster_id ||
      policy.clusterMembershipVersion !== row.cluster_membership_version) {
    throw new Error(`stored procedural Policy payload is invalid: ${row.id}`);
  }
  validatePolicy(policy);
  return {
    id: row.id,
    policyKey: row.policy_key,
    namespaceId: row.namespace_id,
    clusterId: row.cluster_id,
    clusterMembershipVersion: row.cluster_membership_version,
    schemaVersion: PROCEDURAL_POLICY_SCHEMA_VERSION,
    inductionVersion: PROCEDURAL_POLICY_INDUCTION_VERSION,
    status: row.status,
    title: row.title,
    confidence: row.confidence,
    evidenceHash: row.evidence_hash,
    evidenceOccurrenceIds: parseStringArray(row.evidence_occurrence_ids_json),
    ...(row.compiler_model ? { compilerModel: row.compiler_model } : {}),
    ...(row.l2_memory_id ? { l2MemoryId: row.l2_memory_id } : {}),
    policy,
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function policyOccurrenceFromSql(row: PolicyOccurrenceSqlRow): ProceduralPolicyOccurrenceRecord {
  return {
    id: row.id,
    policyVersionId: row.policy_version_id,
    policyKey: row.policy_key,
    occurrenceId: row.occurrence_id,
    pathId: row.path_id,
    spanId: row.span_id,
    episodeId: row.episode_id,
    sessionId: row.session_id,
    clusterMembershipVersion: row.cluster_membership_version,
    evidenceRole: row.evidence_role,
    matchConfidence: row.match_confidence,
    status: row.status,
    createdAt: row.created_at,
    supersededAt: row.superseded_at
  };
}

function parseStringArray(value: string): string[] {
  return parseJson<unknown[]>(value, []).filter((item): item is string => typeof item === "string");
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return JSON.stringify([...new Set(left)].sort()) === JSON.stringify([...new Set(right)].sort());
}
