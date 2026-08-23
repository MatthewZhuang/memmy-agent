import type Database from "better-sqlite3";
import {
  EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION,
  EPISODE_POLICY_PROJECTION_SCHEMA_VERSION,
  validateEpisodePolicyProjection,
  type EpisodePolicyProjectionNodeV1,
  type EpisodePolicyProjectionV1
} from "../service/evolution/episode-policy-projection-model.js";
import { parseJson, toJson } from "../utils/json.js";

export type EpisodePolicyProjectionStatus = "active" | "inactive";

export interface EpisodePolicyProjectionRecord {
  id: string;
  episodeId: string;
  pathId: string;
  pathHash: string;
  userId: string;
  sessionId: string;
  namespaceId: string;
  schemaVersion: typeof EPISODE_POLICY_PROJECTION_SCHEMA_VERSION;
  algorithmVersion: typeof EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION;
  assignmentSnapshotHash: string;
  projectionHash: string;
  mappedSpanCount: number;
  unmappedSpanCount: number;
  status: EpisodePolicyProjectionStatus;
  projection: EpisodePolicyProjectionV1;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface SaveEpisodePolicyProjectionInput {
  projection: EpisodePolicyProjectionV1;
  userId: string;
  sessionId: string;
  namespaceId: string;
  activate?: boolean;
  at: string;
}

export interface SaveEpisodePolicyProjectionResult {
  record: EpisodePolicyProjectionRecord;
  created: boolean;
}

interface ProjectionSqlRow {
  id: string;
  episode_id: string;
  path_id: string;
  path_hash: string;
  user_id: string;
  session_id: string;
  namespace_id: string;
  schema_version: string;
  algorithm_version: string;
  assignment_snapshot_hash: string;
  projection_hash: string;
  mapped_span_count: number;
  unmapped_span_count: number;
  status: EpisodePolicyProjectionStatus;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface ProjectionNodeSqlRow {
  node_json: string;
}

export class EpisodePolicyProjectionRepository {
  constructor(private readonly db: Database.Database) {}

  saveAndActivate(input: SaveEpisodePolicyProjectionInput): SaveEpisodePolicyProjectionResult {
    validateEpisodePolicyProjection(input.projection);
    const namespaceId = input.namespaceId.trim();
    if (!namespaceId) throw new Error("Episode Policy Projection requires namespaceId");
    const path = this.db.prepare(
      `SELECT episode_id, user_id, session_id, namespace_id, path_hash, status
       FROM episode_procedural_paths WHERE id = ?`
    ).get(input.projection.pathId) as {
      episode_id: string;
      user_id: string;
      session_id: string;
      namespace_id: string;
      path_hash: string;
      status: string;
    } | undefined;
    if (!path || path.status !== "active" || path.episode_id !== input.projection.episodeId ||
        path.path_hash !== input.projection.pathHash || path.user_id !== input.userId ||
        path.session_id !== input.sessionId || path.namespace_id !== namespaceId) {
      throw new Error(`Episode Policy Projection source path is stale: ${input.projection.pathId}`);
    }
    this.verifyOccurrenceCoverage(input.projection);
    const activate = input.activate ?? true;
    const existing = this.getByHash(input.projection.projectionHash);
    if (existing) {
      this.assertSameSource(existing, input);
      if (activate && existing.status !== "active") this.activateVersion(existing.id, input.at);
      return { record: this.get(existing.id)!, created: false };
    }

    this.db.transaction(() => {
      if (activate) this.deactivateActiveForEpisode(input.projection.episodeId, input.at);
      this.db.prepare(
        `INSERT INTO episode_policy_projections (
          id, episode_id, path_id, path_hash, user_id, session_id, namespace_id,
          schema_version, algorithm_version, assignment_snapshot_hash,
          projection_hash, mapped_span_count, unmapped_span_count, status,
          payload_json, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)`
      ).run(
        input.projection.id,
        input.projection.episodeId,
        input.projection.pathId,
        input.projection.pathHash,
        input.userId,
        input.sessionId,
        namespaceId,
        input.projection.schemaVersion,
        input.projection.algorithmVersion,
        input.projection.assignmentSnapshotHash,
        input.projection.projectionHash,
        input.projection.coverage.mappedSpanCount,
        input.projection.coverage.unmappedSpanCount,
        activate ? "active" : "inactive",
        toJson(input.projection),
        input.at,
        activate ? input.at : null
      );
      const insertNode = this.db.prepare(
        `INSERT INTO episode_policy_projection_nodes (
          projection_id, node_index, node_kind, occurrence_id, span_id, span_index,
          pre_state_id, post_state_id, policy_version_id, policy_key, cluster_id,
          cluster_membership_version, evidence_role, match_confidence,
          unmapped_reason, node_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const node of input.projection.nodes) {
        const policy = node.assignment.kind === "policy" ? node.assignment : undefined;
        const unmapped = node.assignment.kind === "unmapped" ? node.assignment : undefined;
        insertNode.run(
          input.projection.id,
          node.nodeIndex,
          node.assignment.kind,
          node.occurrenceId,
          node.spanId,
          node.spanIndex,
          node.preStateId,
          node.postStateId,
          policy?.policyVersionId ?? null,
          policy?.policyKey ?? null,
          policy?.clusterId ?? unmapped?.clusterId ?? null,
          policy?.clusterMembershipVersion ?? unmapped?.clusterMembershipVersion ?? null,
          policy?.evidenceRole ?? null,
          policy?.matchConfidence ?? null,
          unmapped?.reason ?? null,
          toJson(node)
        );
      }
    })();
    return { record: this.get(input.projection.id)!, created: true };
  }

  get(id: string): EpisodePolicyProjectionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_policy_projections WHERE id = ?`
    ).get(id) as ProjectionSqlRow | undefined;
    return row ? projectionFromSql(row) : undefined;
  }

  getByHash(projectionHash: string): EpisodePolicyProjectionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_policy_projections WHERE projection_hash = ?`
    ).get(projectionHash) as ProjectionSqlRow | undefined;
    return row ? projectionFromSql(row) : undefined;
  }

  getActiveForEpisode(episodeId: string): EpisodePolicyProjectionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_policy_projections
       WHERE episode_id = ? AND status = 'active'
       LIMIT 1`
    ).get(episodeId) as ProjectionSqlRow | undefined;
    return row ? projectionFromSql(row) : undefined;
  }

  listVersionsForEpisode(episodeId: string): EpisodePolicyProjectionRecord[] {
    return (this.db.prepare(
      `SELECT * FROM episode_policy_projections
       WHERE episode_id = ?
       ORDER BY created_at DESC, id DESC`
    ).all(episodeId) as ProjectionSqlRow[]).map(projectionFromSql);
  }

  listNodes(projectionId: string): EpisodePolicyProjectionNodeV1[] {
    return (this.db.prepare(
      `SELECT node_json FROM episode_policy_projection_nodes
       WHERE projection_id = ?
       ORDER BY node_index ASC`
    ).all(projectionId) as ProjectionNodeSqlRow[]).map((row) => {
      const node = parseJson<EpisodePolicyProjectionNodeV1 | null>(row.node_json, null);
      if (!node) throw new Error(`stored Episode Policy Projection node is invalid: ${projectionId}`);
      return node;
    });
  }

  activateVersion(projectionId: string, at: string): EpisodePolicyProjectionRecord {
    const target = this.get(projectionId);
    if (!target) throw new Error(`Episode Policy Projection version not found: ${projectionId}`);
    const path = this.db.prepare(
      `SELECT status, path_hash FROM episode_procedural_paths WHERE id = ?`
    ).get(target.pathId) as { status: string; path_hash: string } | undefined;
    if (!path || path.status !== "active" || path.path_hash !== target.pathHash) {
      throw new Error(`Episode Policy Projection path is not active: ${projectionId}`);
    }
    if (target.status === "active") return target;
    this.db.transaction(() => {
      this.deactivateActiveForEpisode(target.episodeId, at);
      const activated = this.db.prepare(
        `UPDATE episode_policy_projections
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(at, projectionId);
      if (activated.changes !== 1) {
        throw new Error(`failed to activate Episode Policy Projection: ${projectionId}`);
      }
    })();
    return this.get(projectionId)!;
  }

  deactivateActiveForEpisode(episodeId: string, at: string): void {
    this.db.prepare(
      `UPDATE episode_policy_projections
       SET status = 'inactive', deactivated_at = ?
       WHERE episode_id = ? AND status = 'active'`
    ).run(at, episodeId);
  }

  private verifyOccurrenceCoverage(projection: EpisodePolicyProjectionV1): void {
    const rows = this.db.prepare(
      `SELECT id, span_id, span_index, pre_state_id, post_state_id
       FROM procedural_span_occurrences
       WHERE path_id = ?
       ORDER BY span_index ASC`
    ).all(projection.pathId) as Array<{
      id: string;
      span_id: string;
      span_index: number;
      pre_state_id: string;
      post_state_id: string;
    }>;
    if (rows.length !== projection.nodes.length || rows.some((row, index) => {
      const node = projection.nodes[index];
      return !node || row.id !== node.occurrenceId || row.span_id !== node.spanId ||
        row.span_index !== node.spanIndex || row.pre_state_id !== node.preStateId ||
        row.post_state_id !== node.postStateId;
    })) {
      throw new Error(`Episode Policy Projection does not cover its source path: ${projection.id}`);
    }
  }

  private assertSameSource(
    existing: EpisodePolicyProjectionRecord,
    input: SaveEpisodePolicyProjectionInput
  ): void {
    if (existing.episodeId !== input.projection.episodeId ||
        existing.pathId !== input.projection.pathId ||
        existing.pathHash !== input.projection.pathHash ||
        existing.userId !== input.userId || existing.sessionId !== input.sessionId ||
        existing.namespaceId !== input.namespaceId.trim()) {
      throw new Error(`Episode Policy Projection hash has conflicting source: ${existing.projectionHash}`);
    }
  }
}

function projectionFromSql(row: ProjectionSqlRow): EpisodePolicyProjectionRecord {
  const projection = parseJson<EpisodePolicyProjectionV1 | null>(row.payload_json, null);
  if (!projection) throw new Error(`stored Episode Policy Projection payload is invalid: ${row.id}`);
  validateEpisodePolicyProjection(projection);
  if (projection.id !== row.id || projection.episodeId !== row.episode_id ||
      projection.pathId !== row.path_id || projection.pathHash !== row.path_hash ||
      projection.projectionHash !== row.projection_hash ||
      projection.assignmentSnapshotHash !== row.assignment_snapshot_hash ||
      projection.schemaVersion !== row.schema_version ||
      projection.algorithmVersion !== row.algorithm_version ||
      projection.coverage.mappedSpanCount !== row.mapped_span_count ||
      projection.coverage.unmappedSpanCount !== row.unmapped_span_count) {
    throw new Error(`stored Episode Policy Projection metadata mismatch: ${row.id}`);
  }
  return {
    id: row.id,
    episodeId: row.episode_id,
    pathId: row.path_id,
    pathHash: row.path_hash,
    userId: row.user_id,
    sessionId: row.session_id,
    namespaceId: row.namespace_id,
    schemaVersion: EPISODE_POLICY_PROJECTION_SCHEMA_VERSION,
    algorithmVersion: EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION,
    assignmentSnapshotHash: row.assignment_snapshot_hash,
    projectionHash: row.projection_hash,
    mappedSpanCount: row.mapped_span_count,
    unmappedSpanCount: row.unmapped_span_count,
    status: row.status,
    projection,
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}
