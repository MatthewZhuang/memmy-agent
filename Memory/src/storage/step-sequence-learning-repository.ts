import type Database from "better-sqlite3";
import {
  EPISODE_STEP_POLICY_PROJECTION_ALGORITHM_VERSION,
  EPISODE_STEP_POLICY_PROJECTION_SCHEMA_VERSION,
  STEP_CLUSTER_ALGORITHM_VERSION,
  STEP_EMBEDDING_VERSION,
  STEP_POLICY_SKILL_MINING_ALGORITHM_VERSION,
  STEP_POLICY_SKILL_PATTERN_SCHEMA_VERSION,
  STEP_POLICY_SKILL_SUPPORT_THRESHOLD,
  STEP_SEQUENCE_MINING_ALGORITHM_VERSION,
  STEP_SEQUENCE_PATTERN_SCHEMA_VERSION,
  STEP_SEQUENCE_POLICY_INDUCTION_VERSION,
  STEP_SEQUENCE_POLICY_SCHEMA_VERSION,
  STEP_SEQUENCE_SUPPORT_THRESHOLD,
  averageVectors,
  cosineSimilarity,
  hasMultipleDistinctValues,
  selectLongestNonOverlapping,
  sequenceOccurrencesFullyCovered,
  type EpisodeStepPolicyProjectionV1,
  type ProceduralStepOccurrenceV1,
  type StepEvidenceRole,
  type StepSequencePolicyV1
} from "../service/evolution/step-sequence-learning-model.js";
import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";

export interface ProceduralStepEmbeddingRecord {
  id: string;
  occurrenceId: string;
  namespaceId: string;
  embeddingVersion: string;
  semanticHash: string;
  embeddingProvider: string;
  embeddingModel: string;
  vector: number[];
  contentHash: string;
  createdAt: string;
  updatedAt: string;
}

export interface ProceduralStepClusterRecord {
  id: string;
  namespaceId: string;
  algorithmVersion: string;
  status: "forming" | "ready" | "stale";
  memberCount: number;
  distinctEpisodeCount: number;
  centerVector: number[];
  centerSemanticText: string;
  membershipVersion: string;
  createdAt: string;
  updatedAt: string;
}

export interface ProceduralStepClusterMemberRecord {
  clusterId: string;
  occurrenceId: string;
  algorithmVersion: string;
  episodeId: string;
  similarity: number;
  createdAt: string;
}

export interface StepSequencePatternOccurrenceRecord {
  id: string;
  patternId: string;
  pathId: string;
  episodeId: string;
  sessionId: string;
  startStepIndex: number;
  endStepIndex: number;
  stepOccurrenceIds: string[];
  clusterIds: string[];
  selected: boolean;
  terminalReward?: number;
  evidenceRole: StepEvidenceRole;
  createdAt: string;
}

export interface StepSequencePatternRecord {
  id: string;
  namespaceId: string;
  sequenceHash: string;
  clusterIds: string[];
  lifecycleStatus: "observed" | "ready" | "stale";
  occurrenceCount: number;
  distinctEpisodeCount: number;
  selectedOccurrenceCount: number;
  selectedEpisodeCount: number;
  isMaximal: boolean;
  membershipVersion: string;
  activePolicyVersionId?: string;
  supersededByPatternId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface StepSequencePolicyVersionRecord {
  id: string;
  policyKey: string;
  namespaceId: string;
  patternId: string;
  patternMembershipVersion: string;
  status: "active" | "inactive";
  title: string;
  confidence: number;
  evidenceHash: string;
  l2MemoryId?: string;
  compilerModel?: string;
  policy: StepSequencePolicyV1;
  createdAt: string;
  activatedAt?: string;
  deactivatedAt?: string;
}

export interface EpisodeStepPolicyProjectionRecord {
  id: string;
  episodeId: string;
  pathId: string;
  pathHash: string;
  userId: string;
  sessionId: string;
  namespaceId: string;
  status: "active" | "inactive";
  projection: EpisodeStepPolicyProjectionV1;
  createdAt: string;
  activatedAt?: string;
  deactivatedAt?: string;
}

export interface StepPolicySkillPatternOccurrenceRecord {
  id: string;
  patternId: string;
  projectionId: string;
  episodeId: string;
  pathId: string;
  sessionId: string;
  startNodeIndex: number;
  endNodeIndex: number;
  policyKeys: string[];
  policyVersionIds: string[];
  stepOccurrenceIds: string[];
  selected: boolean;
  terminalReward?: number;
  evidenceRole: StepEvidenceRole;
  createdAt: string;
}

export interface StepPolicySkillPatternRecord {
  id: string;
  namespaceId: string;
  sequenceHash: string;
  policyKeys: string[];
  lifecycleStatus: "observed" | "ready" | "stale";
  occurrenceCount: number;
  distinctEpisodeCount: number;
  selectedOccurrenceCount: number;
  selectedEpisodeCount: number;
  isMaximal: boolean;
  membershipVersion: string;
  activeSkillMemoryId?: string;
  supersededByPatternId?: string;
  createdAt: string;
  updatedAt: string;
}

export class StepSequenceLearningRepository {
  constructor(private readonly db: Database.Database) {}

  savePathSteps(input: {
    pathId: string;
    occurrences: readonly ProceduralStepOccurrenceV1[];
  }): ProceduralStepOccurrenceV1[] {
    const insert = this.db.prepare(
      `INSERT INTO procedural_step_occurrences (
        id, path_id, path_hash, step_id, episode_id, user_id, session_id,
        namespace_id, step_index, raw_turn_id, intent, summary, semantic_text,
        semantic_hash, outcome, tool_name, pre_state_id, post_state_id,
        step_json, reconstruction_algorithm_version, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(path_id, step_id) DO NOTHING`
    );
    this.db.transaction(() => {
      for (const occurrence of input.occurrences) {
        if (occurrence.pathId !== input.pathId) {
          throw new Error(`Step occurrence belongs to another path: ${occurrence.id}`);
        }
        insert.run(
          occurrence.id,
          occurrence.pathId,
          occurrence.pathHash,
          occurrence.stepId,
          occurrence.episodeId,
          occurrence.userId,
          occurrence.sessionId,
          occurrence.namespaceId,
          occurrence.stepIndex,
          occurrence.rawTurnId,
          occurrence.intent,
          occurrence.summary,
          occurrence.semanticText,
          occurrence.semanticHash,
          occurrence.outcome,
          occurrence.toolName ?? null,
          occurrence.preStateId,
          occurrence.postStateId,
          toJson(occurrence.step),
          occurrence.reconstructionAlgorithmVersion,
          occurrence.createdAt
        );
      }
    })();
    return this.listStepsForPath(input.pathId);
  }

  getStep(id: string): ProceduralStepOccurrenceV1 | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_step_occurrences WHERE id = ?`
    ).get(id) as StepRow | undefined;
    return row ? stepFromSql(row) : undefined;
  }

  listStepsForPath(pathId: string): ProceduralStepOccurrenceV1[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_step_occurrences
       WHERE path_id = ? ORDER BY step_index ASC`
    ).all(pathId) as StepRow[]).map(stepFromSql);
  }

  saveEmbedding(input: {
    occurrence: ProceduralStepOccurrenceV1;
    embeddingProvider: string;
    embeddingModel: string;
    vector: readonly number[];
    at: string;
  }): ProceduralStepEmbeddingRecord {
    validateVector(input.vector, input.occurrence.id);
    const contentHash = stableHash({
      occurrenceId: input.occurrence.id,
      embeddingVersion: STEP_EMBEDDING_VERSION,
      semanticHash: input.occurrence.semanticHash,
      embeddingProvider: input.embeddingProvider,
      embeddingModel: input.embeddingModel,
      vector: input.vector
    });
    const existing = this.getEmbedding(input.occurrence.id, {
      provider: input.embeddingProvider,
      model: input.embeddingModel,
      semanticHash: input.occurrence.semanticHash
    });
    if (existing) return existing;
    const id = `procedural_step_embedding_${contentHash.slice(0, 24)}`;
    this.db.prepare(
      `INSERT INTO procedural_step_embeddings (
        id, occurrence_id, namespace_id, embedding_version, semantic_hash,
        embedding_provider, embedding_model, embedding_dim, vector_json,
        content_hash, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      id,
      input.occurrence.id,
      input.occurrence.namespaceId,
      STEP_EMBEDDING_VERSION,
      input.occurrence.semanticHash,
      input.embeddingProvider,
      input.embeddingModel,
      input.vector.length,
      toJson(input.vector),
      contentHash,
      input.at,
      input.at
    );
    return this.getEmbeddingById(id)!;
  }

  getEmbedding(
    occurrenceId: string,
    input: { provider: string; model: string; semanticHash: string }
  ): ProceduralStepEmbeddingRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_step_embeddings
       WHERE occurrence_id = ? AND embedding_version = ? AND semantic_hash = ?
         AND embedding_provider = ? AND embedding_model = ?
       LIMIT 1`
    ).get(
      occurrenceId,
      STEP_EMBEDDING_VERSION,
      input.semanticHash,
      input.provider,
      input.model
    ) as StepEmbeddingRow | undefined;
    return row ? embeddingFromSql(row) : undefined;
  }

  getEmbeddingById(id: string): ProceduralStepEmbeddingRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_step_embeddings WHERE id = ?`
    ).get(id) as StepEmbeddingRow | undefined;
    return row ? embeddingFromSql(row) : undefined;
  }

  getCurrentEmbedding(occurrenceId: string): ProceduralStepEmbeddingRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_step_embeddings
       WHERE occurrence_id = ? AND embedding_version = ?
       ORDER BY updated_at DESC, id DESC LIMIT 1`
    ).get(occurrenceId, STEP_EMBEDDING_VERSION) as StepEmbeddingRow | undefined;
    return row ? embeddingFromSql(row) : undefined;
  }

  listClusters(namespaceId: string): ProceduralStepClusterRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_step_clusters
       WHERE namespace_id = ? AND algorithm_version = ? AND status != 'stale'
       ORDER BY id ASC`
    ).all(namespaceId, STEP_CLUSTER_ALGORITHM_VERSION) as StepClusterRow[]).map(clusterFromSql);
  }

  getCluster(id: string): ProceduralStepClusterRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_step_clusters WHERE id = ?`
    ).get(id) as StepClusterRow | undefined;
    return row ? clusterFromSql(row) : undefined;
  }

  getClusterMember(occurrenceId: string): ProceduralStepClusterMemberRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_step_cluster_members
       WHERE occurrence_id = ? AND algorithm_version = ? LIMIT 1`
    ).get(occurrenceId, STEP_CLUSTER_ALGORITHM_VERSION) as StepClusterMemberRow | undefined;
    return row ? clusterMemberFromSql(row) : undefined;
  }

  assignStepToCluster(input: {
    occurrence: ProceduralStepOccurrenceV1;
    embedding: ProceduralStepEmbeddingRecord;
    similarityThreshold: number;
    at: string;
  }): ProceduralStepClusterMemberRecord {
    const existing = this.getClusterMember(input.occurrence.id);
    if (existing) return existing;
    const matches = this.listClusters(input.occurrence.namespaceId)
      .map((cluster) => ({
        cluster,
        similarity: cosineSimilarity(input.embedding.vector, cluster.centerVector)
      }))
      .filter((match) => match.similarity >= input.similarityThreshold)
      .sort((left, right) => right.similarity - left.similarity ||
        left.cluster.id.localeCompare(right.cluster.id));
    const selected = matches[0];
    const clusterId = selected?.cluster.id ?? `procedural_step_cluster_${stableHash({
      namespaceId: input.occurrence.namespaceId,
      algorithmVersion: STEP_CLUSTER_ALGORITHM_VERSION,
      anchorOccurrenceId: input.occurrence.id
    }).slice(0, 24)}`;
    this.db.transaction(() => {
      if (!selected) {
        this.db.prepare(
          `INSERT INTO procedural_step_clusters (
            id, namespace_id, algorithm_version, status, member_count,
            distinct_episode_count, center_vector_json, center_semantic_text,
            membership_version, created_at, updated_at
          ) VALUES (?, ?, ?, 'forming', 0, 0, ?, ?, ?, ?, ?)`
        ).run(
          clusterId,
          input.occurrence.namespaceId,
          STEP_CLUSTER_ALGORITHM_VERSION,
          toJson(input.embedding.vector),
          input.occurrence.semanticText,
          stableHash([]),
          input.at,
          input.at
        );
      }
      this.db.prepare(
        `INSERT INTO procedural_step_cluster_members (
          cluster_id, occurrence_id, algorithm_version, episode_id, similarity, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)`
      ).run(
        clusterId,
        input.occurrence.id,
        STEP_CLUSTER_ALGORITHM_VERSION,
        input.occurrence.episodeId,
        selected?.similarity ?? 1,
        input.at
      );
      this.recomputeCluster(clusterId, input.at);
    })();
    return this.getClusterMember(input.occurrence.id)!;
  }

  listClusteredStepsForPath(pathId: string): Array<{
    occurrence: ProceduralStepOccurrenceV1;
    member: ProceduralStepClusterMemberRecord;
  }> {
    const rows = this.db.prepare(
      `SELECT occurrences.*, members.cluster_id, members.algorithm_version AS member_algorithm_version,
              members.similarity, members.created_at AS member_created_at
       FROM procedural_step_occurrences AS occurrences
       JOIN procedural_step_cluster_members AS members
         ON members.occurrence_id = occurrences.id
       WHERE occurrences.path_id = ? AND members.algorithm_version = ?
       ORDER BY occurrences.step_index ASC`
    ).all(pathId, STEP_CLUSTER_ALGORITHM_VERSION) as Array<StepRow & {
      cluster_id: string;
      member_algorithm_version: string;
      similarity: number;
      member_created_at: string;
    }>;
    return rows.map((row) => ({
      occurrence: stepFromSql(row),
      member: {
        clusterId: row.cluster_id,
        occurrenceId: row.id,
        algorithmVersion: row.member_algorithm_version,
        episodeId: row.episode_id,
        similarity: row.similarity,
        createdAt: row.member_created_at
      }
    }));
  }

  replaceStepSequenceOccurrences(input: {
    namespaceId: string;
    episodeId: string;
    pathId: string;
    sessionId: string;
    terminalReward?: number;
    evidenceRole: StepEvidenceRole;
    windows: Array<{
      patternId: string;
      sequenceHash: string;
      clusterIds: string[];
      startStepIndex: number;
      endStepIndex: number;
      stepOccurrenceIds: string[];
    }>;
    at: string;
  }): StepSequencePatternRecord[] {
    this.db.transaction(() => {
      this.db.prepare(
        `DELETE FROM step_sequence_pattern_occurrences WHERE episode_id = ?`
      ).run(input.episodeId);
      const upsertPattern = this.db.prepare(
        `INSERT INTO step_sequence_patterns (
          id, namespace_id, schema_version, algorithm_version, sequence_hash,
          cluster_ids_json, sequence_length, lifecycle_status, occurrence_count,
          distinct_episode_count, is_maximal, membership_version,
          active_policy_version_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'observed', 0, 0, 1, ?, NULL, ?, ?)
        ON CONFLICT(namespace_id, algorithm_version, sequence_hash)
        DO UPDATE SET updated_at = excluded.updated_at`
      );
      const insertOccurrence = this.db.prepare(
        `INSERT INTO step_sequence_pattern_occurrences (
          id, pattern_id, path_id, episode_id, session_id, start_step_index,
          end_step_index, step_occurrence_ids_json, cluster_ids_json,
          terminal_reward, evidence_role, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const window of input.windows) {
        upsertPattern.run(
          window.patternId,
          input.namespaceId,
          STEP_SEQUENCE_PATTERN_SCHEMA_VERSION,
          STEP_SEQUENCE_MINING_ALGORITHM_VERSION,
          window.sequenceHash,
          toJson(window.clusterIds),
          window.clusterIds.length,
          stableHash([]),
          input.at,
          input.at
        );
        const id = `step_sequence_occurrence_${stableHash({
          patternId: window.patternId,
          pathId: input.pathId,
          start: window.startStepIndex,
          end: window.endStepIndex
        }).slice(0, 24)}`;
        insertOccurrence.run(
          id,
          window.patternId,
          input.pathId,
          input.episodeId,
          input.sessionId,
          window.startStepIndex,
          window.endStepIndex,
          toJson(window.stepOccurrenceIds),
          toJson(window.clusterIds),
          input.terminalReward ?? null,
          input.evidenceRole,
          input.at
        );
      }
      this.refreshStepPatterns(input.namespaceId, input.at);
    })();
    return this.listStepPatterns(input.namespaceId);
  }

  listStepPatterns(namespaceId: string): StepSequencePatternRecord[] {
    return (this.db.prepare(
      `SELECT * FROM step_sequence_patterns
       WHERE namespace_id = ? AND algorithm_version = ?
       ORDER BY sequence_length DESC, id ASC`
    ).all(namespaceId, STEP_SEQUENCE_MINING_ALGORITHM_VERSION) as StepPatternRow[])
      .map(stepPatternFromSql);
  }

  getStepPattern(id: string): StepSequencePatternRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM step_sequence_patterns WHERE id = ?`
    ).get(id) as StepPatternRow | undefined;
    return row ? stepPatternFromSql(row) : undefined;
  }

  listStepPatternOccurrences(
    patternId: string,
    options: { selectedOnly?: boolean } = {}
  ): StepSequencePatternOccurrenceRecord[] {
    return (this.db.prepare(
      `SELECT * FROM step_sequence_pattern_occurrences
       WHERE pattern_id = ? ${options.selectedOnly ? "AND is_selected = 1" : ""}
       ORDER BY episode_id, start_step_index`
    ).all(patternId) as StepPatternOccurrenceRow[]).map(stepPatternOccurrenceFromSql);
  }

  saveAndActivatePolicy(input: {
    policy: StepSequencePolicyV1;
    l2MemoryId: string;
    at: string;
  }): StepSequencePolicyVersionRecord {
    const pattern = this.getStepPattern(input.policy.patternId);
    if (!pattern || pattern.membershipVersion !== input.policy.patternMembershipVersion ||
        pattern.lifecycleStatus !== "ready" ||
        pattern.selectedEpisodeCount < STEP_SEQUENCE_SUPPORT_THRESHOLD ||
        pattern.supersededByPatternId) {
      throw new Error(`Step sequence Policy pattern is stale: ${input.policy.patternId}`);
    }
    const existing = this.getPolicyForPatternMembership(
      pattern.id,
      pattern.membershipVersion
    );
    if (existing) return existing;
    this.db.transaction(() => {
      this.db.prepare(
        `UPDATE step_sequence_policy_versions
         SET status = 'inactive', deactivated_at = ?
         WHERE pattern_id = ? AND status = 'active'`
      ).run(input.at, pattern.id);
      this.db.prepare(
        `INSERT INTO step_sequence_policy_versions (
          id, policy_key, namespace_id, pattern_id, pattern_membership_version,
          schema_version, induction_version, status, title, confidence,
          evidence_hash, l2_memory_id, compiler_model, payload_json,
          created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, NULL)`
      ).run(
        input.policy.id,
        input.policy.policyKey,
        input.policy.namespaceId,
        input.policy.patternId,
        input.policy.patternMembershipVersion,
        STEP_SEQUENCE_POLICY_SCHEMA_VERSION,
        input.policy.inductionVersion,
        input.policy.title,
        input.policy.confidence,
        input.policy.provenance.evidenceHash,
        input.l2MemoryId,
        input.policy.provenance.model ?? null,
        toJson(input.policy),
        input.at,
        input.at
      );
      this.db.prepare(
        `UPDATE step_sequence_patterns
         SET active_policy_version_id = ?, updated_at = ? WHERE id = ?`
      ).run(input.policy.id, input.at, pattern.id);
    })();
    return this.getPolicy(input.policy.id)!;
  }

  activatePolicyVersion(id: string, at: string): StepSequencePolicyVersionRecord {
    const policy = this.getPolicy(id);
    if (!policy) throw new Error(`Step sequence Policy not found: ${id}`);
    const pattern = this.getStepPattern(policy.patternId);
    if (!pattern || pattern.membershipVersion !== policy.patternMembershipVersion ||
        pattern.lifecycleStatus !== "ready" ||
        pattern.selectedEpisodeCount < STEP_SEQUENCE_SUPPORT_THRESHOLD ||
        pattern.supersededByPatternId) {
      throw new Error(`Step sequence Policy pattern is not promotable: ${id}`);
    }
    this.db.transaction(() => {
      this.db.prepare(
        `UPDATE step_sequence_policy_versions
         SET status = 'inactive', deactivated_at = ?
         WHERE pattern_id = ? AND status = 'active' AND id != ?`
      ).run(at, pattern.id, id);
      this.db.prepare(
        `UPDATE step_sequence_policy_versions
         SET status = 'active', activated_at = ?, deactivated_at = NULL WHERE id = ?`
      ).run(at, id);
      this.db.prepare(
        `UPDATE step_sequence_patterns
         SET active_policy_version_id = ?, updated_at = ? WHERE id = ?`
      ).run(id, at, pattern.id);
      if (policy.l2MemoryId) {
        this.db.prepare(
          `UPDATE memories
           SET status = 'resolving',
               properties_json = json_set(properties_json, '$.status', 'resolving'),
               version = version + 1,
               updated_at = ?
           WHERE id = ? AND status = 'archived'`
        ).run(at, policy.l2MemoryId);
      }
    })();
    return this.getPolicy(id)!;
  }

  getPolicy(id: string): StepSequencePolicyVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM step_sequence_policy_versions WHERE id = ?`
    ).get(id) as StepPolicyRow | undefined;
    return row ? stepPolicyFromSql(row) : undefined;
  }

  getPolicyByMemoryId(l2MemoryId: string): StepSequencePolicyVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM step_sequence_policy_versions
       WHERE l2_memory_id = ? ORDER BY created_at DESC LIMIT 1`
    ).get(l2MemoryId) as StepPolicyRow | undefined;
    return row ? stepPolicyFromSql(row) : undefined;
  }

  getPolicyRepairForDecision(
    patternId: string,
    repairId: string
  ): StepSequencePolicyVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT versions.*
       FROM step_sequence_policy_versions AS versions
       WHERE versions.pattern_id = ?
         AND EXISTS (
           SELECT 1 FROM json_each(versions.payload_json, '$.revision.repairIds')
           WHERE json_each.value = ?
         )
       ORDER BY versions.created_at DESC LIMIT 1`
    ).get(patternId, repairId) as StepPolicyRow | undefined;
    return row ? stepPolicyFromSql(row) : undefined;
  }

  saveAndActivateRepairedPolicy(input: {
    policy: StepSequencePolicyV1;
    l2MemoryId: string;
    basePolicyVersionId: string;
    at: string;
  }): StepSequencePolicyVersionRecord {
    const existing = this.getPolicy(input.policy.id);
    if (existing) return existing;
    const base = this.getPolicy(input.basePolicyVersionId);
    const pattern = this.getStepPattern(input.policy.patternId);
    if (!base || !pattern || base.status !== "active" || base.patternId !== input.policy.patternId ||
        pattern?.activePolicyVersionId !== base.id ||
        pattern.membershipVersion !== input.policy.patternMembershipVersion ||
        pattern.lifecycleStatus !== "ready" ||
        pattern.selectedEpisodeCount < STEP_SEQUENCE_SUPPORT_THRESHOLD ||
        pattern.supersededByPatternId) {
      throw new Error(`Step sequence Policy repair base is stale: ${input.basePolicyVersionId}`);
    }
    this.db.transaction(() => {
      this.db.prepare(
        `UPDATE step_sequence_policy_versions
         SET status = 'inactive', deactivated_at = ?
         WHERE pattern_id = ? AND status = 'active'`
      ).run(input.at, pattern.id);
      this.db.prepare(
        `INSERT INTO step_sequence_policy_versions (
          id, policy_key, namespace_id, pattern_id, pattern_membership_version,
          schema_version, induction_version, status, title, confidence,
          evidence_hash, l2_memory_id, compiler_model, payload_json,
          created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, NULL)`
      ).run(
        input.policy.id,
        input.policy.policyKey,
        input.policy.namespaceId,
        input.policy.patternId,
        input.policy.patternMembershipVersion,
        STEP_SEQUENCE_POLICY_SCHEMA_VERSION,
        input.policy.inductionVersion,
        input.policy.title,
        input.policy.confidence,
        input.policy.provenance.evidenceHash,
        input.l2MemoryId,
        input.policy.provenance.model ?? null,
        toJson(input.policy),
        input.at,
        input.at
      );
      this.db.prepare(
        `UPDATE step_sequence_patterns
         SET active_policy_version_id = ?, updated_at = ? WHERE id = ?`
      ).run(input.policy.id, input.at, pattern.id);
      if (base.l2MemoryId && base.l2MemoryId !== input.l2MemoryId) {
        this.db.prepare(
          `UPDATE memories
           SET status = 'archived',
               properties_json = json_set(
                 json_set(properties_json, '$.status', 'archived'),
                 '$.internal_info.policy.status', 'archived'
               ),
               version = version + 1,
               updated_at = ?
           WHERE id = ? AND status NOT IN ('archived', 'deleted')`
        ).run(input.at, base.l2MemoryId);
      }
    })();
    return this.getPolicy(input.policy.id)!;
  }

  getPolicyForPatternMembership(
    patternId: string,
    membershipVersion: string
  ): StepSequencePolicyVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM step_sequence_policy_versions
       WHERE pattern_id = ? AND pattern_membership_version = ? AND induction_version = ?
       LIMIT 1`
    ).get(
      patternId,
      membershipVersion,
      STEP_SEQUENCE_POLICY_INDUCTION_VERSION
    ) as StepPolicyRow | undefined;
    return row ? stepPolicyFromSql(row) : undefined;
  }

  listActivePolicies(namespaceId: string): StepSequencePolicyVersionRecord[] {
    return (this.db.prepare(
      `SELECT * FROM step_sequence_policy_versions
       WHERE namespace_id = ? AND status = 'active'
       ORDER BY created_at ASC, id ASC`
    ).all(namespaceId) as StepPolicyRow[]).map(stepPolicyFromSql);
  }

  saveAndActivateProjection(input: {
    projection: EpisodeStepPolicyProjectionV1;
    userId: string;
    sessionId: string;
    namespaceId: string;
    at: string;
  }): EpisodeStepPolicyProjectionRecord {
    const existing = this.getProjectionByHash(input.projection.projectionHash);
    if (existing) return existing;
    this.db.transaction(() => {
      this.db.prepare(
        `UPDATE episode_step_policy_projections
         SET status = 'inactive', deactivated_at = ?
         WHERE episode_id = ? AND status = 'active'`
      ).run(input.at, input.projection.episodeId);
      this.db.prepare(
        `INSERT INTO episode_step_policy_projections (
          id, episode_id, path_id, path_hash, user_id, session_id, namespace_id,
          schema_version, algorithm_version, projection_hash, mapped_step_count,
          unmapped_step_count, status, payload_json, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL)`
      ).run(
        input.projection.id,
        input.projection.episodeId,
        input.projection.pathId,
        input.projection.pathHash,
        input.userId,
        input.sessionId,
        input.namespaceId,
        EPISODE_STEP_POLICY_PROJECTION_SCHEMA_VERSION,
        EPISODE_STEP_POLICY_PROJECTION_ALGORITHM_VERSION,
        input.projection.projectionHash,
        input.projection.mappedStepCount,
        input.projection.unmappedStepCount,
        toJson(input.projection),
        input.at,
        input.at
      );
    })();
    return this.getProjection(input.projection.id)!;
  }

  getProjection(id: string): EpisodeStepPolicyProjectionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_step_policy_projections WHERE id = ?`
    ).get(id) as StepProjectionRow | undefined;
    return row ? stepProjectionFromSql(row) : undefined;
  }

  getProjectionByHash(hash: string): EpisodeStepPolicyProjectionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_step_policy_projections WHERE projection_hash = ?`
    ).get(hash) as StepProjectionRow | undefined;
    return row ? stepProjectionFromSql(row) : undefined;
  }

  getActiveProjectionForEpisode(episodeId: string): EpisodeStepPolicyProjectionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_step_policy_projections
       WHERE episode_id = ? AND status = 'active' LIMIT 1`
    ).get(episodeId) as StepProjectionRow | undefined;
    return row ? stepProjectionFromSql(row) : undefined;
  }

  replaceSkillPatternOccurrences(input: {
    namespaceId: string;
    projectionId: string;
    episodeId: string;
    pathId: string;
    sessionId: string;
    terminalReward?: number;
    evidenceRole: StepEvidenceRole;
    windows: Array<{
      patternId: string;
      sequenceHash: string;
      policyKeys: string[];
      policyVersionIds: string[];
      startNodeIndex: number;
      endNodeIndex: number;
      stepOccurrenceIds: string[];
    }>;
    at: string;
  }): StepPolicySkillPatternRecord[] {
    this.db.transaction(() => {
      this.db.prepare(
        `DELETE FROM step_policy_skill_pattern_occurrences WHERE episode_id = ?`
      ).run(input.episodeId);
      const upsertPattern = this.db.prepare(
        `INSERT INTO step_policy_skill_patterns (
          id, namespace_id, schema_version, algorithm_version, sequence_hash,
          policy_keys_json, lifecycle_status, occurrence_count,
          distinct_episode_count, is_maximal, membership_version,
          active_skill_memory_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'observed', 0, 0, 1, ?, NULL, ?, ?)
        ON CONFLICT(namespace_id, algorithm_version, sequence_hash)
        DO UPDATE SET updated_at = excluded.updated_at`
      );
      const insertOccurrence = this.db.prepare(
        `INSERT INTO step_policy_skill_pattern_occurrences (
          id, pattern_id, projection_id, episode_id, path_id, session_id,
          start_node_index, end_node_index, policy_keys_json,
          policy_version_ids_json, step_occurrence_ids_json, terminal_reward,
          evidence_role, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const window of input.windows) {
        upsertPattern.run(
          window.patternId,
          input.namespaceId,
          STEP_POLICY_SKILL_PATTERN_SCHEMA_VERSION,
          STEP_POLICY_SKILL_MINING_ALGORITHM_VERSION,
          window.sequenceHash,
          toJson(window.policyKeys),
          stableHash([]),
          input.at,
          input.at
        );
        const id = `step_policy_skill_occurrence_${stableHash({
          patternId: window.patternId,
          projectionId: input.projectionId,
          start: window.startNodeIndex,
          end: window.endNodeIndex
        }).slice(0, 24)}`;
        insertOccurrence.run(
          id,
          window.patternId,
          input.projectionId,
          input.episodeId,
          input.pathId,
          input.sessionId,
          window.startNodeIndex,
          window.endNodeIndex,
          toJson(window.policyKeys),
          toJson(window.policyVersionIds),
          toJson(window.stepOccurrenceIds),
          input.terminalReward ?? null,
          input.evidenceRole,
          input.at
        );
      }
      this.refreshSkillPatterns(input.namespaceId, input.at);
    })();
    return this.listSkillPatterns(input.namespaceId);
  }

  listSkillPatterns(namespaceId: string): StepPolicySkillPatternRecord[] {
    return (this.db.prepare(
      `SELECT * FROM step_policy_skill_patterns
       WHERE namespace_id = ? AND algorithm_version = ?
       ORDER BY json_array_length(policy_keys_json) DESC, id ASC`
    ).all(namespaceId, STEP_POLICY_SKILL_MINING_ALGORITHM_VERSION) as SkillPatternRow[])
      .map(skillPatternFromSql);
  }

  getSkillPattern(id: string): StepPolicySkillPatternRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM step_policy_skill_patterns WHERE id = ?`
    ).get(id) as SkillPatternRow | undefined;
    return row ? skillPatternFromSql(row) : undefined;
  }

  listSkillPatternOccurrences(
    patternId: string,
    options: { selectedOnly?: boolean } = {}
  ): StepPolicySkillPatternOccurrenceRecord[] {
    return (this.db.prepare(
      `SELECT * FROM step_policy_skill_pattern_occurrences
       WHERE pattern_id = ? ${options.selectedOnly ? "AND is_selected = 1" : ""}
       ORDER BY episode_id, start_node_index`
    ).all(patternId) as SkillPatternOccurrenceRow[]).map(skillPatternOccurrenceFromSql);
  }

  markSkillCompiled(input: {
    patternId: string;
    membershipVersion: string;
    skillMemoryId: string;
    at: string;
  }): void {
    const pattern = this.getSkillPattern(input.patternId);
    if (!pattern || pattern.membershipVersion !== input.membershipVersion ||
        pattern.lifecycleStatus !== "ready" ||
        pattern.selectedEpisodeCount < STEP_POLICY_SKILL_SUPPORT_THRESHOLD ||
        pattern.supersededByPatternId ||
        !hasMultipleDistinctValues(pattern.policyKeys)) {
      throw new Error(`Step Policy Skill pattern is stale: ${input.patternId}`);
    }
    this.db.prepare(
      `UPDATE step_policy_skill_patterns
       SET active_skill_memory_id = ?, updated_at = ? WHERE id = ?`
    ).run(input.skillMemoryId, input.at, input.patternId);
  }

  activateRepairedSkill(input: {
    patternId: string;
    previousSkillMemoryId: string;
    repairedSkillMemoryId: string;
    at: string;
  }): void {
    const pattern = this.getSkillPattern(input.patternId);
    if (!pattern || pattern.activeSkillMemoryId !== input.previousSkillMemoryId) return;
    this.db.transaction(() => {
      this.db.prepare(
        `UPDATE memories
         SET status = 'archived',
             properties_json = json_set(
               json_set(properties_json, '$.status', 'archived'),
               '$.internal_info.skill.status', 'archived'
             ),
             version = version + 1,
             updated_at = ?
         WHERE id = ? AND status NOT IN ('archived', 'deleted')`
      ).run(input.at, input.previousSkillMemoryId);
      this.db.prepare(
        `UPDATE step_policy_skill_patterns
         SET active_skill_memory_id = ?, updated_at = ? WHERE id = ?`
      ).run(input.repairedSkillMemoryId, input.at, input.patternId);
    })();
  }

  retireStepPatternsCoveredBy(
    supersederPatternId: string,
    at: string
  ): { retiredPatternIds: string[]; affectedEpisodeIds: string[] } {
    const superseder = this.getStepPattern(supersederPatternId);
    if (!superseder?.activePolicyVersionId || superseder.supersededByPatternId) {
      return { retiredPatternIds: [], affectedEpisodeIds: [] };
    }
    const longerOccurrences = this.listStepPatternOccurrences(superseder.id);
    const retiredPatternIds: string[] = [];
    const affectedEpisodeIds = new Set<string>();
    this.db.transaction(() => {
      for (const candidate of this.listStepPatterns(superseder.namespaceId)) {
        if (candidate.id === superseder.id || !candidate.activePolicyVersionId ||
            candidate.clusterIds.length >= superseder.clusterIds.length) continue;
        const shorterOccurrences = this.listStepPatternOccurrences(candidate.id);
        if (!sequenceOccurrencesFullyCovered({
          shorterSequence: candidate.clusterIds,
          longerSequence: superseder.clusterIds,
          shorterOccurrences: shorterOccurrences.map(stepOccurrenceInterval),
          longerOccurrences: longerOccurrences.map(stepOccurrenceInterval)
        })) continue;
        this.db.prepare(
          `UPDATE step_sequence_patterns
           SET lifecycle_status = 'observed', is_maximal = 0,
               superseded_by_pattern_id = ?, updated_at = ?
           WHERE id = ?`
        ).run(superseder.id, at, candidate.id);
        this.retirePolicy(candidate.activePolicyVersionId, candidate.id, at);
        retiredPatternIds.push(candidate.id);
        for (const occurrence of shorterOccurrences) {
          affectedEpisodeIds.add(occurrence.episodeId);
        }
      }
    })();
    return {
      retiredPatternIds,
      affectedEpisodeIds: [...affectedEpisodeIds].sort((left, right) => left.localeCompare(right))
    };
  }

  retireSkillPatternsCoveredBy(
    supersederPatternId: string,
    at: string
  ): string[] {
    const superseder = this.getSkillPattern(supersederPatternId);
    if (!superseder?.activeSkillMemoryId || superseder.supersededByPatternId) return [];
    const longerOccurrences = this.listSkillPatternOccurrences(superseder.id);
    const retiredPatternIds: string[] = [];
    this.db.transaction(() => {
      for (const candidate of this.listSkillPatterns(superseder.namespaceId)) {
        if (candidate.id === superseder.id || !candidate.activeSkillMemoryId ||
            candidate.policyKeys.length >= superseder.policyKeys.length) continue;
        const shorterOccurrences = this.listSkillPatternOccurrences(candidate.id);
        if (!sequenceOccurrencesFullyCovered({
          shorterSequence: candidate.policyKeys,
          longerSequence: superseder.policyKeys,
          shorterOccurrences: shorterOccurrences.map(skillOccurrenceInterval),
          longerOccurrences: longerOccurrences.map(skillOccurrenceInterval)
        })) continue;
        this.db.prepare(
          `UPDATE step_policy_skill_patterns
           SET lifecycle_status = 'observed', is_maximal = 0,
               superseded_by_pattern_id = ?, updated_at = ?
           WHERE id = ?`
        ).run(superseder.id, at, candidate.id);
        this.retireSkill(candidate.activeSkillMemoryId, candidate.id, at);
        retiredPatternIds.push(candidate.id);
      }
    })();
    return retiredPatternIds;
  }

  private recomputeCluster(clusterId: string, at: string): void {
    const rows = this.db.prepare(
      `SELECT members.occurrence_id, members.episode_id, embeddings.vector_json,
              occurrences.semantic_text
       FROM procedural_step_cluster_members AS members
       JOIN procedural_step_embeddings AS embeddings
         ON embeddings.occurrence_id = members.occurrence_id
        AND embeddings.embedding_version = ?
       JOIN procedural_step_occurrences AS occurrences
         ON occurrences.id = members.occurrence_id
       WHERE members.cluster_id = ?
       ORDER BY members.occurrence_id`
    ).all(STEP_EMBEDDING_VERSION, clusterId) as Array<{
      occurrence_id: string;
      episode_id: string;
      vector_json: string;
      semantic_text: string;
    }>;
    const vectors = rows.map((row) => numberArray(row.vector_json));
    const center = averageVectors(vectors);
    const anchor = rows.map((row, index) => ({
      row,
      similarity: cosineSimilarity(vectors[index]!, center)
    })).sort((left, right) => right.similarity - left.similarity ||
      left.row.occurrence_id.localeCompare(right.row.occurrence_id))[0];
    const memberIds = rows.map((row) => row.occurrence_id);
    this.db.prepare(
      `UPDATE procedural_step_clusters
       SET status = ?, member_count = ?, distinct_episode_count = ?,
           center_vector_json = ?, center_semantic_text = ?, membership_version = ?,
           updated_at = ? WHERE id = ?`
    ).run(
      new Set(rows.map((row) => row.episode_id)).size >= 2 ? "ready" : "forming",
      rows.length,
      new Set(rows.map((row) => row.episode_id)).size,
      toJson(center),
      anchor?.row.semantic_text ?? "",
      stableHash(memberIds),
      at,
      clusterId
    );
  }

  private refreshStepPatterns(namespaceId: string, at: string): void {
    const patterns = this.listStepPatterns(namespaceId);
    const patternById = new Map(patterns.map((pattern) => [pattern.id, pattern]));
    const occurrences = new Map(patterns.map((pattern) => [
      pattern.id,
      this.listStepPatternOccurrences(pattern.id)
    ]));
    const episodeIds = new Map(patterns.map((pattern) => [
      pattern.id,
      unique((occurrences.get(pattern.id) ?? []).map((occurrence) => occurrence.episodeId))
    ]));
    const supportOccurrences = new Map(patterns.map((pattern) => [
      pattern.id,
      (occurrences.get(pattern.id) ?? [])
        .filter((occurrence) => occurrence.evidenceRole === "support")
    ]));
    const supportEpisodeIds = new Map(patterns.map((pattern) => [
      pattern.id,
      unique((supportOccurrences.get(pattern.id) ?? []).map((occurrence) => occurrence.episodeId))
    ]));
    const candidatesByPath = new Map<string, Array<{
      id: string;
      startIndex: number;
      endIndex: number;
      sequenceLength: number;
      support: number;
    }>>();
    for (const pattern of patterns) {
      const support = supportEpisodeIds.get(pattern.id)?.length ?? 0;
      if (support < STEP_SEQUENCE_SUPPORT_THRESHOLD) continue;
      for (const occurrence of supportOccurrences.get(pattern.id) ?? []) {
        const key = `${occurrence.episodeId}\u0000${occurrence.pathId}`;
        const candidates = candidatesByPath.get(key) ?? [];
        candidates.push({
          id: occurrence.id,
          startIndex: occurrence.startStepIndex,
          endIndex: occurrence.endStepIndex,
          sequenceLength: pattern.clusterIds.length,
          support
        });
        candidatesByPath.set(key, candidates);
      }
    }
    const selectedIds = new Set([...candidatesByPath.values()].flatMap((candidates) =>
      selectLongestNonOverlapping(candidates).map((candidate) => candidate.id)));
    this.db.prepare(
      `UPDATE step_sequence_pattern_occurrences SET is_selected = 0
       WHERE pattern_id IN (
         SELECT id FROM step_sequence_patterns
         WHERE namespace_id = ? AND algorithm_version = ?
       )`
    ).run(namespaceId, STEP_SEQUENCE_MINING_ALGORITHM_VERSION);
    const selectOccurrence = this.db.prepare(
      `UPDATE step_sequence_pattern_occurrences SET is_selected = 1 WHERE id = ?`
    );
    for (const id of selectedIds) selectOccurrence.run(id);
    const selectedOccurrences = new Map(patterns.map((pattern) => [
      pattern.id,
      (supportOccurrences.get(pattern.id) ?? []).filter((occurrence) => selectedIds.has(occurrence.id))
    ]));
    for (const pattern of patterns) {
      if (!pattern.supersededByPatternId) continue;
      const superseder = patternById.get(pattern.supersededByPatternId);
      const stillCovered = Boolean(superseder?.activePolicyVersionId) &&
        sequenceOccurrencesFullyCovered({
          shorterSequence: pattern.clusterIds,
          longerSequence: superseder!.clusterIds,
          shorterOccurrences: (occurrences.get(pattern.id) ?? []).map(stepOccurrenceInterval),
          longerOccurrences: (occurrences.get(superseder!.id) ?? []).map(stepOccurrenceInterval)
        });
      if (!stillCovered) {
        this.db.prepare(
          `UPDATE step_sequence_patterns
           SET superseded_by_pattern_id = NULL, is_maximal = 1, updated_at = ?
           WHERE id = ?`
        ).run(at, pattern.id);
        pattern.supersededByPatternId = undefined;
        pattern.isMaximal = true;
      }
    }
    const update = this.db.prepare(
      `UPDATE step_sequence_patterns
       SET lifecycle_status = ?, occurrence_count = ?, distinct_episode_count = ?,
           selected_occurrence_count = ?, selected_episode_count = ?,
           is_maximal = ?, membership_version = ?, updated_at = ? WHERE id = ?`
    );
    for (const pattern of patterns) {
      const patternOccurrences = occurrences.get(pattern.id) ?? [];
      const episodes = episodeIds.get(pattern.id) ?? [];
      const selected = selectedOccurrences.get(pattern.id) ?? [];
      const selectedEpisodes = unique(selected.map((occurrence) => occurrence.episodeId));
      const superseded = Boolean(pattern.supersededByPatternId);
      const lifecycle = patternOccurrences.length === 0
        ? "stale"
        : selectedEpisodes.length >= STEP_SEQUENCE_SUPPORT_THRESHOLD && !superseded
          ? "ready"
          : "observed";
      update.run(
        lifecycle,
        patternOccurrences.length,
        episodes.length,
        selected.length,
        selectedEpisodes.length,
        superseded ? 0 : 1,
        stableHash(selected.map((occurrence) => occurrence.id).sort()),
        at,
        pattern.id
      );
      if (lifecycle !== "ready" && pattern.activePolicyVersionId) {
        this.retirePolicy(pattern.activePolicyVersionId, pattern.id, at);
      }
    }
  }

  private refreshSkillPatterns(namespaceId: string, at: string): void {
    const patterns = this.listSkillPatterns(namespaceId);
    const patternById = new Map(patterns.map((pattern) => [pattern.id, pattern]));
    const occurrences = new Map(patterns.map((pattern) => [
      pattern.id,
      this.listSkillPatternOccurrences(pattern.id)
    ]));
    const episodeIds = new Map(patterns.map((pattern) => [
      pattern.id,
      unique((occurrences.get(pattern.id) ?? []).map((occurrence) => occurrence.episodeId))
    ]));
    const supportOccurrences = new Map(patterns.map((pattern) => [
      pattern.id,
      (occurrences.get(pattern.id) ?? [])
        .filter((occurrence) => occurrence.evidenceRole === "support")
    ]));
    const supportEpisodeIds = new Map(patterns.map((pattern) => [
      pattern.id,
      unique((supportOccurrences.get(pattern.id) ?? []).map((occurrence) => occurrence.episodeId))
    ]));
    const candidatesByProjection = new Map<string, Array<{
      id: string;
      startIndex: number;
      endIndex: number;
      sequenceLength: number;
      support: number;
    }>>();
    for (const pattern of patterns) {
      const support = supportEpisodeIds.get(pattern.id)?.length ?? 0;
      if (support < STEP_POLICY_SKILL_SUPPORT_THRESHOLD) continue;
      for (const occurrence of supportOccurrences.get(pattern.id) ?? []) {
        const key = `${occurrence.episodeId}\u0000${occurrence.projectionId}`;
        const candidates = candidatesByProjection.get(key) ?? [];
        candidates.push({
          id: occurrence.id,
          startIndex: occurrence.startNodeIndex,
          endIndex: occurrence.endNodeIndex,
          sequenceLength: pattern.policyKeys.length,
          support
        });
        candidatesByProjection.set(key, candidates);
      }
    }
    const selectedIds = new Set([...candidatesByProjection.values()].flatMap((candidates) =>
      selectLongestNonOverlapping(candidates).map((candidate) => candidate.id)));
    this.db.prepare(
      `UPDATE step_policy_skill_pattern_occurrences SET is_selected = 0
       WHERE pattern_id IN (
         SELECT id FROM step_policy_skill_patterns
         WHERE namespace_id = ? AND algorithm_version = ?
       )`
    ).run(namespaceId, STEP_POLICY_SKILL_MINING_ALGORITHM_VERSION);
    const selectOccurrence = this.db.prepare(
      `UPDATE step_policy_skill_pattern_occurrences SET is_selected = 1 WHERE id = ?`
    );
    for (const id of selectedIds) selectOccurrence.run(id);
    const selectedOccurrences = new Map(patterns.map((pattern) => [
      pattern.id,
      (supportOccurrences.get(pattern.id) ?? []).filter((occurrence) => selectedIds.has(occurrence.id))
    ]));
    for (const pattern of patterns) {
      if (!pattern.supersededByPatternId) continue;
      const superseder = patternById.get(pattern.supersededByPatternId);
      const stillCovered = Boolean(superseder?.activeSkillMemoryId) &&
        sequenceOccurrencesFullyCovered({
          shorterSequence: pattern.policyKeys,
          longerSequence: superseder!.policyKeys,
          shorterOccurrences: (occurrences.get(pattern.id) ?? []).map(skillOccurrenceInterval),
          longerOccurrences: (occurrences.get(superseder!.id) ?? []).map(skillOccurrenceInterval)
        });
      if (!stillCovered) {
        this.db.prepare(
          `UPDATE step_policy_skill_patterns
           SET superseded_by_pattern_id = NULL, is_maximal = 1, updated_at = ?
           WHERE id = ?`
        ).run(at, pattern.id);
        pattern.supersededByPatternId = undefined;
        pattern.isMaximal = true;
      }
    }
    const update = this.db.prepare(
      `UPDATE step_policy_skill_patterns
       SET lifecycle_status = ?, occurrence_count = ?, distinct_episode_count = ?,
           selected_occurrence_count = ?, selected_episode_count = ?,
           is_maximal = ?, membership_version = ?, updated_at = ? WHERE id = ?`
    );
    for (const pattern of patterns) {
      const patternOccurrences = occurrences.get(pattern.id) ?? [];
      const episodes = episodeIds.get(pattern.id) ?? [];
      const selected = selectedOccurrences.get(pattern.id) ?? [];
      const selectedEpisodes = unique(selected.map((occurrence) => occurrence.episodeId));
      const superseded = Boolean(pattern.supersededByPatternId);
      const lifecycle = patternOccurrences.length === 0
        ? "stale"
        : selectedEpisodes.length >= STEP_POLICY_SKILL_SUPPORT_THRESHOLD && !superseded
          ? "ready"
          : "observed";
      update.run(
        lifecycle,
        patternOccurrences.length,
        episodes.length,
        selected.length,
        selectedEpisodes.length,
        superseded ? 0 : 1,
        stableHash(selected.map((occurrence) => occurrence.id).sort()),
        at,
        pattern.id
      );
      if (lifecycle !== "ready" && pattern.activeSkillMemoryId) {
        this.retireSkill(pattern.activeSkillMemoryId, pattern.id, at);
      }
    }
  }

  private retirePolicy(policyVersionId: string, patternId: string, at: string): void {
    const policy = this.getPolicy(policyVersionId);
    this.db.prepare(
      `UPDATE step_sequence_policy_versions
       SET status = 'inactive', deactivated_at = ?
       WHERE id = ? AND status = 'active'`
    ).run(at, policyVersionId);
    this.db.prepare(
      `UPDATE step_sequence_patterns
       SET active_policy_version_id = NULL, updated_at = ?
       WHERE id = ? AND active_policy_version_id = ?`
    ).run(at, patternId, policyVersionId);
    if (policy?.l2MemoryId) {
      this.db.prepare(
        `UPDATE memories
         SET status = 'archived',
             properties_json = json_set(properties_json, '$.status', 'archived'),
             version = version + 1,
             updated_at = ?
         WHERE id = ? AND status NOT IN ('archived', 'deleted')`
      ).run(at, policy.l2MemoryId);
    }
  }

  private retireSkill(skillMemoryId: string, patternId: string, at: string): void {
    this.db.prepare(
      `UPDATE memories
       SET status = 'archived',
           properties_json = json_set(
             json_set(properties_json, '$.status', 'archived'),
             '$.internal_info.step_policy_sequence_skill.executable', json('false')
           ),
           version = version + 1,
           updated_at = ?
       WHERE id = ? AND status NOT IN ('archived', 'deleted')`
    ).run(at, skillMemoryId);
    this.db.prepare(
      `UPDATE step_policy_skill_patterns
       SET active_skill_memory_id = NULL, updated_at = ?
       WHERE id = ? AND active_skill_memory_id = ?`
    ).run(at, patternId, skillMemoryId);
  }
}

interface StepRow {
  id: string;
  path_id: string;
  path_hash: string;
  step_id: string;
  episode_id: string;
  user_id: string;
  session_id: string;
  namespace_id: string;
  step_index: number;
  raw_turn_id: string;
  intent: string;
  summary: string;
  semantic_text: string;
  semantic_hash: string;
  outcome: ProceduralStepOccurrenceV1["outcome"];
  tool_name: string | null;
  pre_state_id: string;
  post_state_id: string;
  step_json: string;
  reconstruction_algorithm_version: string;
  created_at: string;
}

interface StepEmbeddingRow {
  id: string;
  occurrence_id: string;
  namespace_id: string;
  embedding_version: string;
  semantic_hash: string;
  embedding_provider: string;
  embedding_model: string;
  vector_json: string;
  content_hash: string;
  created_at: string;
  updated_at: string;
}

interface StepClusterRow {
  id: string;
  namespace_id: string;
  algorithm_version: string;
  status: ProceduralStepClusterRecord["status"];
  member_count: number;
  distinct_episode_count: number;
  center_vector_json: string;
  center_semantic_text: string;
  membership_version: string;
  created_at: string;
  updated_at: string;
}

interface StepClusterMemberRow {
  cluster_id: string;
  occurrence_id: string;
  algorithm_version: string;
  episode_id: string;
  similarity: number;
  created_at: string;
}

interface StepPatternRow {
  id: string;
  namespace_id: string;
  sequence_hash: string;
  cluster_ids_json: string;
  lifecycle_status: StepSequencePatternRecord["lifecycleStatus"];
  occurrence_count: number;
  distinct_episode_count: number;
  selected_occurrence_count: number;
  selected_episode_count: number;
  is_maximal: number;
  membership_version: string;
  active_policy_version_id: string | null;
  superseded_by_pattern_id: string | null;
  created_at: string;
  updated_at: string;
}

interface StepPatternOccurrenceRow {
  id: string;
  pattern_id: string;
  path_id: string;
  episode_id: string;
  session_id: string;
  start_step_index: number;
  end_step_index: number;
  step_occurrence_ids_json: string;
  cluster_ids_json: string;
  is_selected: number;
  terminal_reward: number | null;
  evidence_role: StepEvidenceRole;
  created_at: string;
}

interface StepPolicyRow {
  id: string;
  policy_key: string;
  namespace_id: string;
  pattern_id: string;
  pattern_membership_version: string;
  status: StepSequencePolicyVersionRecord["status"];
  title: string;
  confidence: number;
  evidence_hash: string;
  l2_memory_id: string | null;
  compiler_model: string | null;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface StepProjectionRow {
  id: string;
  episode_id: string;
  path_id: string;
  path_hash: string;
  user_id: string;
  session_id: string;
  namespace_id: string;
  status: EpisodeStepPolicyProjectionRecord["status"];
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface SkillPatternRow {
  id: string;
  namespace_id: string;
  sequence_hash: string;
  policy_keys_json: string;
  lifecycle_status: StepPolicySkillPatternRecord["lifecycleStatus"];
  occurrence_count: number;
  distinct_episode_count: number;
  selected_occurrence_count: number;
  selected_episode_count: number;
  is_maximal: number;
  membership_version: string;
  active_skill_memory_id: string | null;
  superseded_by_pattern_id: string | null;
  created_at: string;
  updated_at: string;
}

interface SkillPatternOccurrenceRow {
  id: string;
  pattern_id: string;
  projection_id: string;
  episode_id: string;
  path_id: string;
  session_id: string;
  start_node_index: number;
  end_node_index: number;
  policy_keys_json: string;
  policy_version_ids_json: string;
  step_occurrence_ids_json: string;
  is_selected: number;
  terminal_reward: number | null;
  evidence_role: StepEvidenceRole;
  created_at: string;
}

function stepFromSql(row: StepRow): ProceduralStepOccurrenceV1 {
  return {
    id: row.id,
    schemaVersion: "procedural-step-occurrence.v1",
    pathId: row.path_id,
    pathHash: row.path_hash,
    episodeId: row.episode_id,
    userId: row.user_id,
    sessionId: row.session_id,
    namespaceId: row.namespace_id,
    stepId: row.step_id,
    stepIndex: row.step_index,
    rawTurnId: row.raw_turn_id,
    intent: row.intent,
    summary: row.summary,
    semanticText: row.semantic_text,
    semanticHash: row.semantic_hash,
    outcome: row.outcome,
    ...(row.tool_name ? { toolName: row.tool_name } : {}),
    preStateId: row.pre_state_id,
    postStateId: row.post_state_id,
    step: parseJson(row.step_json, null)!,
    reconstructionAlgorithmVersion: row.reconstruction_algorithm_version,
    createdAt: row.created_at
  };
}

function embeddingFromSql(row: StepEmbeddingRow): ProceduralStepEmbeddingRecord {
  return {
    id: row.id,
    occurrenceId: row.occurrence_id,
    namespaceId: row.namespace_id,
    embeddingVersion: row.embedding_version,
    semanticHash: row.semantic_hash,
    embeddingProvider: row.embedding_provider,
    embeddingModel: row.embedding_model,
    vector: numberArray(row.vector_json),
    contentHash: row.content_hash,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function clusterFromSql(row: StepClusterRow): ProceduralStepClusterRecord {
  return {
    id: row.id,
    namespaceId: row.namespace_id,
    algorithmVersion: row.algorithm_version,
    status: row.status,
    memberCount: row.member_count,
    distinctEpisodeCount: row.distinct_episode_count,
    centerVector: numberArray(row.center_vector_json),
    centerSemanticText: row.center_semantic_text,
    membershipVersion: row.membership_version,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function clusterMemberFromSql(row: StepClusterMemberRow): ProceduralStepClusterMemberRecord {
  return {
    clusterId: row.cluster_id,
    occurrenceId: row.occurrence_id,
    algorithmVersion: row.algorithm_version,
    episodeId: row.episode_id,
    similarity: row.similarity,
    createdAt: row.created_at
  };
}

function stepPatternFromSql(row: StepPatternRow): StepSequencePatternRecord {
  return {
    id: row.id,
    namespaceId: row.namespace_id,
    sequenceHash: row.sequence_hash,
    clusterIds: stringArray(row.cluster_ids_json),
    lifecycleStatus: row.lifecycle_status,
    occurrenceCount: row.occurrence_count,
    distinctEpisodeCount: row.distinct_episode_count,
    selectedOccurrenceCount: row.selected_occurrence_count,
    selectedEpisodeCount: row.selected_episode_count,
    isMaximal: row.is_maximal === 1,
    membershipVersion: row.membership_version,
    ...(row.active_policy_version_id ? { activePolicyVersionId: row.active_policy_version_id } : {}),
    ...(row.superseded_by_pattern_id
      ? { supersededByPatternId: row.superseded_by_pattern_id }
      : {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function stepPatternOccurrenceFromSql(row: StepPatternOccurrenceRow): StepSequencePatternOccurrenceRecord {
  return {
    id: row.id,
    patternId: row.pattern_id,
    pathId: row.path_id,
    episodeId: row.episode_id,
    sessionId: row.session_id,
    startStepIndex: row.start_step_index,
    endStepIndex: row.end_step_index,
    stepOccurrenceIds: stringArray(row.step_occurrence_ids_json),
    clusterIds: stringArray(row.cluster_ids_json),
    selected: row.is_selected === 1,
    ...(row.terminal_reward === null ? {} : { terminalReward: row.terminal_reward }),
    evidenceRole: row.evidence_role,
    createdAt: row.created_at
  };
}

function stepPolicyFromSql(row: StepPolicyRow): StepSequencePolicyVersionRecord {
  return {
    id: row.id,
    policyKey: row.policy_key,
    namespaceId: row.namespace_id,
    patternId: row.pattern_id,
    patternMembershipVersion: row.pattern_membership_version,
    status: row.status,
    title: row.title,
    confidence: row.confidence,
    evidenceHash: row.evidence_hash,
    ...(row.l2_memory_id ? { l2MemoryId: row.l2_memory_id } : {}),
    ...(row.compiler_model ? { compilerModel: row.compiler_model } : {}),
    policy: parseJson(row.payload_json, null)!,
    createdAt: row.created_at,
    ...(row.activated_at ? { activatedAt: row.activated_at } : {}),
    ...(row.deactivated_at ? { deactivatedAt: row.deactivated_at } : {})
  };
}

function stepProjectionFromSql(row: StepProjectionRow): EpisodeStepPolicyProjectionRecord {
  return {
    id: row.id,
    episodeId: row.episode_id,
    pathId: row.path_id,
    pathHash: row.path_hash,
    userId: row.user_id,
    sessionId: row.session_id,
    namespaceId: row.namespace_id,
    status: row.status,
    projection: parseJson(row.payload_json, null)!,
    createdAt: row.created_at,
    ...(row.activated_at ? { activatedAt: row.activated_at } : {}),
    ...(row.deactivated_at ? { deactivatedAt: row.deactivated_at } : {})
  };
}

function skillPatternFromSql(row: SkillPatternRow): StepPolicySkillPatternRecord {
  return {
    id: row.id,
    namespaceId: row.namespace_id,
    sequenceHash: row.sequence_hash,
    policyKeys: stringArray(row.policy_keys_json),
    lifecycleStatus: row.lifecycle_status,
    occurrenceCount: row.occurrence_count,
    distinctEpisodeCount: row.distinct_episode_count,
    selectedOccurrenceCount: row.selected_occurrence_count,
    selectedEpisodeCount: row.selected_episode_count,
    isMaximal: row.is_maximal === 1,
    membershipVersion: row.membership_version,
    ...(row.active_skill_memory_id ? { activeSkillMemoryId: row.active_skill_memory_id } : {}),
    ...(row.superseded_by_pattern_id
      ? { supersededByPatternId: row.superseded_by_pattern_id }
      : {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function skillPatternOccurrenceFromSql(
  row: SkillPatternOccurrenceRow
): StepPolicySkillPatternOccurrenceRecord {
  return {
    id: row.id,
    patternId: row.pattern_id,
    projectionId: row.projection_id,
    episodeId: row.episode_id,
    pathId: row.path_id,
    sessionId: row.session_id,
    startNodeIndex: row.start_node_index,
    endNodeIndex: row.end_node_index,
    policyKeys: stringArray(row.policy_keys_json),
    policyVersionIds: stringArray(row.policy_version_ids_json),
    stepOccurrenceIds: stringArray(row.step_occurrence_ids_json),
    selected: row.is_selected === 1,
    ...(row.terminal_reward === null ? {} : { terminalReward: row.terminal_reward }),
    evidenceRole: row.evidence_role,
    createdAt: row.created_at
  };
}

function numberArray(value: string): number[] {
  return parseJson<unknown[]>(value, []).filter((item): item is number =>
    typeof item === "number" && Number.isFinite(item));
}

function stringArray(value: string): string[] {
  return parseJson<unknown[]>(value, []).filter((item): item is string => typeof item === "string");
}

function validateVector(vector: readonly number[], source: string): void {
  if (vector.length === 0 || vector.some((value) => !Number.isFinite(value))) {
    throw new Error(`Step embedding vector is invalid: ${source}`);
  }
}

function stepOccurrenceInterval(occurrence: StepSequencePatternOccurrenceRecord) {
  return {
    episodeId: occurrence.episodeId,
    pathId: occurrence.pathId,
    startIndex: occurrence.startStepIndex,
    endIndex: occurrence.endStepIndex
  };
}

function skillOccurrenceInterval(occurrence: StepPolicySkillPatternOccurrenceRecord) {
  return {
    episodeId: occurrence.episodeId,
    pathId: occurrence.pathId,
    startIndex: occurrence.startNodeIndex,
    endIndex: occurrence.endNodeIndex
  };
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}
