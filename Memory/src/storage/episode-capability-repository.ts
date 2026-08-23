import type Database from "better-sqlite3";
import {
  EPISODE_CAPABILITY_AFFINITY_ALGORITHM_VERSION,
  EPISODE_CAPABILITY_SIGNATURE_ALGORITHM_VERSION,
  EPISODE_CAPABILITY_SIGNATURE_SCHEMA_VERSION,
  type EpisodeCapabilityAffinityV1,
  type EpisodeCapabilitySignatureV1,
  type EpisodeCapabilityVectorsV1
} from "../service/evolution/episode-capability-model.js";
import { parseJson, toJson } from "../utils/json.js";

export type EpisodeCapabilitySignatureStatus = "active" | "inactive";

export interface EpisodeCapabilitySignatureRecord {
  signature: EpisodeCapabilitySignatureV1;
  familyId: string;
  embeddingProvider: string;
  embeddingModel: string;
  embeddingDim: number;
  vectors: EpisodeCapabilityVectorsV1;
  status: EpisodeCapabilitySignatureStatus;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface EpisodeCapabilityAffinityRecord {
  affinity: EpisodeCapabilityAffinityV1;
  createdAt: string;
}

interface SignatureSqlRow {
  id: string;
  family_id: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_dim: number;
  goal_vector_json: string;
  state_transition_vector_json: string;
  outcome_vector_json: string;
  context_vector_json: string;
  status: EpisodeCapabilitySignatureStatus;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface AffinitySqlRow {
  payload_json: string;
  created_at: string;
}

export class EpisodeCapabilityRepository {
  constructor(private readonly db: Database.Database) {}

  saveAndActivate(input: {
    signature: EpisodeCapabilitySignatureV1;
    familyId: string;
    vectors: EpisodeCapabilityVectorsV1;
    embeddingProvider: string;
    embeddingModel: string;
    at: string;
  }): { record: EpisodeCapabilitySignatureRecord; created: boolean } {
    validateSignatureInput(input);
    const existing = this.getByHash(input.signature.signatureHash);
    if (existing) {
      if (existing.signature.id !== input.signature.id ||
          existing.signature.projectionId !== input.signature.projectionId ||
          existing.familyId !== input.familyId) {
        throw new Error(`Episode Capability signature hash conflict: ${input.signature.id}`);
      }
      if (existing.status !== "active") this.activate(existing.signature.id, input.at);
      return { record: this.get(existing.signature.id)!, created: false };
    }
    this.verifyProjection(input.signature);
    const dimension = input.vectors.goalVector.length;
    this.db.transaction(() => {
      this.db.prepare(
        `UPDATE episode_capability_signatures
         SET status = 'inactive', deactivated_at = ?
         WHERE episode_id = ? AND status = 'active'`
      ).run(input.at, input.signature.episodeId);
      this.db.prepare(
        `INSERT INTO episode_capability_signatures (
          id, projection_id, episode_id, path_id, namespace_id,
          schema_version, algorithm_version, signature_hash, family_id,
          embedding_provider, embedding_model, embedding_dim,
          goal_vector_json, state_transition_vector_json, outcome_vector_json,
          context_vector_json, status, payload_json, created_at,
          activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  'active', ?, ?, ?, NULL)`
      ).run(
        input.signature.id,
        input.signature.projectionId,
        input.signature.episodeId,
        input.signature.pathId,
        input.signature.namespaceId,
        input.signature.schemaVersion,
        input.signature.algorithmVersion,
        input.signature.signatureHash,
        input.familyId,
        input.embeddingProvider.trim(),
        input.embeddingModel.trim(),
        dimension,
        toJson(input.vectors.goalVector),
        toJson(input.vectors.stateTransitionVector),
        toJson(input.vectors.outcomeVector),
        toJson(input.vectors.contextVector),
        toJson(input.signature),
        input.at,
        input.at
      );
    })();
    return { record: this.get(input.signature.id)!, created: true };
  }

  get(id: string): EpisodeCapabilitySignatureRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_capability_signatures WHERE id = ?`
    ).get(id) as SignatureSqlRow | undefined;
    return row ? signatureFromSql(row) : undefined;
  }

  getByHash(signatureHash: string): EpisodeCapabilitySignatureRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_capability_signatures WHERE signature_hash = ?`
    ).get(signatureHash) as SignatureSqlRow | undefined;
    return row ? signatureFromSql(row) : undefined;
  }

  getActiveForEpisode(episodeId: string): EpisodeCapabilitySignatureRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_capability_signatures
       WHERE episode_id = ? AND status = 'active' LIMIT 1`
    ).get(episodeId) as SignatureSqlRow | undefined;
    return row ? signatureFromSql(row) : undefined;
  }

  listActiveForNamespace(namespaceId: string): EpisodeCapabilitySignatureRecord[] {
    return (this.db.prepare(
      `SELECT * FROM episode_capability_signatures
       WHERE namespace_id = ? AND status = 'active'
       ORDER BY created_at ASC, id ASC`
    ).all(namespaceId) as SignatureSqlRow[]).map(signatureFromSql);
  }

  listActiveCandidatesByPolicyOverlap(input: {
    namespaceId: string;
    policyKeys: readonly string[];
    excludeEpisodeId: string;
    minSharedPolicyKeys?: number;
    limit?: number;
  }): EpisodeCapabilitySignatureRecord[] {
    const policyKeys = [...new Set(input.policyKeys.map((key) => key.trim()).filter(Boolean))];
    const minShared = Math.max(1, Math.trunc(input.minSharedPolicyKeys ?? 2));
    const limit = Math.max(1, Math.min(500, Math.trunc(input.limit ?? 64)));
    if (policyKeys.length < minShared) return [];
    const placeholders = policyKeys.map(() => "?").join(", ");
    return (this.db.prepare(
      `SELECT signatures.*
       FROM episode_capability_signatures AS signatures
       JOIN episode_policy_projection_nodes AS nodes
         ON nodes.projection_id = signatures.projection_id
       WHERE signatures.namespace_id = ?
         AND signatures.status = 'active'
         AND signatures.episode_id != ?
         AND nodes.policy_key IN (${placeholders})
       GROUP BY signatures.id
       HAVING COUNT(DISTINCT nodes.policy_key) >= ?
       ORDER BY COUNT(DISTINCT nodes.policy_key) DESC,
                signatures.created_at DESC, signatures.id ASC
       LIMIT ?`
    ).all(
      input.namespaceId,
      input.excludeEpisodeId,
      ...policyKeys,
      minShared,
      limit
    ) as SignatureSqlRow[]).map(signatureFromSql);
  }

  listActiveForFamily(familyId: string): EpisodeCapabilitySignatureRecord[] {
    return (this.db.prepare(
      `SELECT * FROM episode_capability_signatures
       WHERE family_id = ? AND status = 'active'
       ORDER BY created_at ASC, id ASC`
    ).all(familyId) as SignatureSqlRow[]).map(signatureFromSql);
  }

  saveAffinity(affinity: EpisodeCapabilityAffinityV1, at: string): EpisodeCapabilityAffinityRecord {
    if (affinity.algorithmVersion !== EPISODE_CAPABILITY_AFFINITY_ALGORITHM_VERSION ||
        affinity.leftSignatureId >= affinity.rightSignatureId ||
        affinity.leftEpisodeId === affinity.rightEpisodeId) {
      throw new Error(`invalid Episode Capability affinity: ${affinity.id}`);
    }
    const left = this.get(affinity.leftSignatureId);
    const right = this.get(affinity.rightSignatureId);
    if (!left || !right || left.signature.namespaceId !== affinity.namespaceId ||
        right.signature.namespaceId !== affinity.namespaceId) {
      throw new Error(`Episode Capability affinity source is missing: ${affinity.id}`);
    }
    this.db.prepare(
      `INSERT INTO episode_capability_affinities (
        id, namespace_id, algorithm_version, left_signature_id, right_signature_id,
        left_episode_id, right_episode_id, goal_similarity,
        state_transition_similarity, outcome_similarity, context_similarity,
        path_structure_similarity, combined_similarity, family_eligible,
        payload_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        goal_similarity = excluded.goal_similarity,
        state_transition_similarity = excluded.state_transition_similarity,
        outcome_similarity = excluded.outcome_similarity,
        context_similarity = excluded.context_similarity,
        path_structure_similarity = excluded.path_structure_similarity,
        combined_similarity = excluded.combined_similarity,
        family_eligible = excluded.family_eligible,
        payload_json = excluded.payload_json`
    ).run(
      affinity.id,
      affinity.namespaceId,
      affinity.algorithmVersion,
      affinity.leftSignatureId,
      affinity.rightSignatureId,
      affinity.leftEpisodeId,
      affinity.rightEpisodeId,
      affinity.goalSimilarity,
      affinity.stateTransitionSimilarity,
      affinity.outcomeSimilarity,
      affinity.contextSimilarity,
      affinity.pathStructureSimilarity,
      affinity.combinedSimilarity,
      affinity.familyEligible ? 1 : 0,
      toJson(affinity),
      at
    );
    return this.getAffinity(affinity.id)!;
  }

  getAffinity(id: string): EpisodeCapabilityAffinityRecord | undefined {
    const row = this.db.prepare(
      `SELECT payload_json, created_at FROM episode_capability_affinities WHERE id = ?`
    ).get(id) as AffinitySqlRow | undefined;
    return row ? affinityFromSql(row) : undefined;
  }

  listAffinitiesForSignature(signatureId: string): EpisodeCapabilityAffinityRecord[] {
    return (this.db.prepare(
      `SELECT payload_json, created_at FROM episode_capability_affinities
       WHERE left_signature_id = ? OR right_signature_id = ?
       ORDER BY combined_similarity DESC, id ASC`
    ).all(signatureId, signatureId) as AffinitySqlRow[]).map(affinityFromSql);
  }

  private activate(id: string, at: string): void {
    const record = this.get(id);
    if (!record) throw new Error(`Episode Capability signature not found: ${id}`);
    this.db.transaction(() => {
      this.db.prepare(
        `UPDATE episode_capability_signatures
         SET status = 'inactive', deactivated_at = ?
         WHERE episode_id = ? AND status = 'active'`
      ).run(at, record.signature.episodeId);
      this.db.prepare(
        `UPDATE episode_capability_signatures
         SET status = 'active', activated_at = ?, deactivated_at = NULL WHERE id = ?`
      ).run(at, id);
    })();
  }

  private verifyProjection(signature: EpisodeCapabilitySignatureV1): void {
    const row = this.db.prepare(
      `SELECT episode_id, path_id, path_hash, namespace_id, projection_hash, status
       FROM episode_policy_projections WHERE id = ?`
    ).get(signature.projectionId) as {
      episode_id: string;
      path_id: string;
      path_hash: string;
      namespace_id: string;
      projection_hash: string;
      status: string;
    } | undefined;
    if (!row || row.status !== "active" || row.episode_id !== signature.episodeId ||
        row.path_id !== signature.pathId || row.path_hash !== signature.pathHash ||
        row.namespace_id !== signature.namespaceId ||
        row.projection_hash !== signature.projectionHash) {
      throw new Error(`Episode Capability signature projection is stale: ${signature.id}`);
    }
  }
}

function validateSignatureInput(input: {
  signature: EpisodeCapabilitySignatureV1;
  familyId: string;
  vectors: EpisodeCapabilityVectorsV1;
  embeddingProvider: string;
  embeddingModel: string;
}): void {
  if (input.signature.schemaVersion !== EPISODE_CAPABILITY_SIGNATURE_SCHEMA_VERSION ||
      input.signature.algorithmVersion !== EPISODE_CAPABILITY_SIGNATURE_ALGORITHM_VERSION ||
      !input.familyId.trim() || !input.embeddingProvider.trim() || !input.embeddingModel.trim()) {
    throw new Error(`invalid Episode Capability signature: ${input.signature.id}`);
  }
  const vectors = [
    input.vectors.goalVector,
    input.vectors.stateTransitionVector,
    input.vectors.outcomeVector,
    input.vectors.contextVector
  ];
  const dimension = vectors[0]!.length;
  if (dimension === 0 || vectors.some((vector) =>
    vector.length !== dimension || vector.some((value) => !Number.isFinite(value)))) {
    throw new Error(`invalid Episode Capability vectors: ${input.signature.id}`);
  }
}

function signatureFromSql(row: SignatureSqlRow): EpisodeCapabilitySignatureRecord {
  const signature = parseJson<EpisodeCapabilitySignatureV1 | null>(row.payload_json, null);
  const vectors: EpisodeCapabilityVectorsV1 = {
    goalVector: parseJson<number[]>(row.goal_vector_json, []),
    stateTransitionVector: parseJson<number[]>(row.state_transition_vector_json, []),
    outcomeVector: parseJson<number[]>(row.outcome_vector_json, []),
    contextVector: parseJson<number[]>(row.context_vector_json, [])
  };
  if (!signature || signature.id !== row.id ||
      signature.schemaVersion !== EPISODE_CAPABILITY_SIGNATURE_SCHEMA_VERSION ||
      signature.algorithmVersion !== EPISODE_CAPABILITY_SIGNATURE_ALGORITHM_VERSION ||
      Object.values(vectors).some((vector) => vector.length !== row.embedding_dim)) {
    throw new Error(`stored Episode Capability signature is invalid: ${row.id}`);
  }
  return {
    signature,
    familyId: row.family_id,
    embeddingProvider: row.embedding_provider,
    embeddingModel: row.embedding_model,
    embeddingDim: row.embedding_dim,
    vectors,
    status: row.status,
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function affinityFromSql(row: AffinitySqlRow): EpisodeCapabilityAffinityRecord {
  const affinity = parseJson<EpisodeCapabilityAffinityV1 | null>(row.payload_json, null);
  if (!affinity || affinity.algorithmVersion !== EPISODE_CAPABILITY_AFFINITY_ALGORITHM_VERSION) {
    throw new Error("stored Episode Capability affinity is invalid");
  }
  return { affinity, createdAt: row.created_at };
}
