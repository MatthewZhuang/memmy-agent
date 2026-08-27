import type Database from "better-sqlite3";
import {
  EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
  TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION,
  proceduralEvidenceRoleForReward,
  validateEpisodeExecutionPathLite,
  type EpisodeExecutionPathLiteV1,
  type ProceduralEvidenceRole,
  type TrajectoryWindowOccurrenceV1
} from "../service/evolution/procedural-window-model.js";
import { stableHash } from "../utils/id.js";
import { isRecord, parseJson, toJson } from "../utils/json.js";

export const PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION =
  "procedural-step-intent-embedding.v1" as const;
export const TRAJECTORY_WINDOW_FAMILY_REVISION_SCHEMA_VERSION =
  "trajectory-window-family-revision.v1" as const;
export const TRAJECTORY_CLUSTER_VERSION_SCHEMA_VERSION =
  "trajectory-window-cluster-version.v1" as const;
export const TRAJECTORY_SKILL_VERSION_SCHEMA_VERSION =
  "trajectory-skill-version.v1" as const;

export type VersionLifecycleStatus = "active" | "superseded";

export interface EpisodeExecutionPathRecord {
  id: string;
  episodeId: string;
  userId: string;
  schemaVersion: typeof EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION;
  compilationKey: string;
  sourceSnapshotHash: string;
  pathHash: string;
  compilerVersion: string;
  modelSignature?: string;
  sourceRawTurnIds: string[];
  sourceAgentIds: string[];
  stepCount: number;
  status: VersionLifecycleStatus;
  path: EpisodeExecutionPathLiteV1;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface SaveEpisodeExecutionPathInput {
  path: EpisodeExecutionPathLiteV1;
  /** Storage provenance only; clustering scope remains userId, never agentId. */
  sourceAgentIds?: string[];
  /** Optional CAS guard used only when activate is true; null means no active Path was observed. */
  expectedActivePathId?: string | null;
  createdAt: string;
  activate?: boolean;
}

export interface SaveEpisodeExecutionPathResult {
  record: EpisodeExecutionPathRecord;
  created: boolean;
  previousActive?: EpisodeExecutionPathRecord;
}

export interface ProceduralStepEmbeddingRecord {
  id: string;
  pathId: string;
  stepId: string;
  stepIndex: number;
  representationVersion: string;
  embeddingSignature: string;
  semanticHash: string;
  embeddingDim: number;
  vector: number[];
  createdAt: string;
}

export interface UpsertProceduralStepEmbeddingInput {
  pathId: string;
  stepId: string;
  stepIndex: number;
  representationVersion?: string;
  embeddingSignature: string;
  semanticHash: string;
  vector: readonly number[];
  createdAt: string;
}

export interface TrajectoryWindowOccurrenceRecord {
  id: string;
  pathId: string;
  episodeId: string;
  userId: string;
  terminalReward?: number;
  evidenceRole: ProceduralEvidenceRole;
  schemaVersion: typeof TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION;
  windowConfigHash: string;
  scale: number;
  stride: number;
  startStepIndex: number;
  endStepIndex: number;
  stepIds: string[];
  rawTurnIds: string[];
  semanticText: string;
  semanticHash: string;
  coarseRepresentationVersion: string;
  embeddingSignature: string;
  embeddingDim: number;
  coarseVector: number[];
  createdAt: string;
}

export interface InsertTrajectoryWindowInput {
  occurrence: TrajectoryWindowOccurrenceV1;
  windowConfigHash: string;
  coarseRepresentationVersion: string;
  embeddingSignature: string;
  /** Required for a new semantic Window; omitted when reusing an existing occurrence id. */
  coarseVector?: readonly number[];
  createdAt: string;
}

export interface TrajectoryWindowFamilyRecord {
  id: string;
  userId: string;
  scale: number;
  algorithmVersion: string;
  configHash: string;
  embeddingSignature: string;
  status: "active" | "retired";
  activeRevisionId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateTrajectoryWindowFamilyInput {
  id?: string;
  userId: string;
  scale: number;
  algorithmVersion: string;
  configHash: string;
  embeddingSignature: string;
  seedOccurrenceId: string;
  createdAt: string;
}

export interface TrajectoryWindowFamilyMemberInput {
  occurrenceId: string;
  coarseSimilarity: number;
}

export interface TrajectoryWindowFamilyMemberRecord
  extends TrajectoryWindowFamilyMemberInput {
  id: string;
  familyRevisionId: string;
  episodeId: string;
  userId: string;
  isMedoid: boolean;
  createdAt: string;
}

export interface TrajectoryWindowFamilyRevisionRecord {
  id: string;
  familyId: string;
  revisionNo: number;
  membershipHash: string;
  evidenceHash: string;
  medoidOccurrenceId: string;
  metrics: Record<string, unknown>;
  status: VersionLifecycleStatus;
  supersedesRevisionId?: string;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface CommitTrajectoryWindowFamilyRevisionInput {
  familyId: string;
  /** null means that the caller observed an empty Family head. */
  expectedActiveRevisionId: string | null;
  medoidOccurrenceId: string;
  members: TrajectoryWindowFamilyMemberInput[];
  /** Stable hash of the members' reward/evidence snapshot for this revision. */
  evidenceHash: string;
  metrics?: Record<string, unknown>;
  createdAt: string;
}

export interface CommitTrajectoryWindowFamilyRevisionResult {
  record: TrajectoryWindowFamilyRevisionRecord;
  members: TrajectoryWindowFamilyMemberRecord[];
  created: boolean;
  reactivated: boolean;
  previousActive?: TrajectoryWindowFamilyRevisionRecord;
}

export interface ActiveTrajectoryWindowFamilyMedoid {
  family: TrajectoryWindowFamilyRecord;
  revision: TrajectoryWindowFamilyRevisionRecord;
  occurrence: TrajectoryWindowOccurrenceRecord;
}

export interface TrajectoryWindowClusterRecord {
  id: string;
  userId: string;
  scale: number;
  algorithmVersion: string;
  configHash: string;
  status: "active" | "retired";
  activeVersionId?: string;
  activeSkillVersionId?: string;
  activeSkillMemoryId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateTrajectoryWindowClusterInput {
  id?: string;
  userId: string;
  scale: number;
  algorithmVersion: string;
  configHash: string;
  seedOccurrenceId: string;
  createdAt: string;
}

export interface TrajectoryWindowClusterMemberInput {
  occurrenceId: string;
  rewardHash: string;
  coarseSimilarity: number;
  alignment?: Record<string, unknown>;
}

export interface TrajectoryWindowClusterMemberRecord
  extends TrajectoryWindowClusterMemberInput {
  id: string;
  clusterVersionId: string;
  episodeId: string;
  userId: string;
  evidenceRole: ProceduralEvidenceRole;
  terminalReward?: number;
  isMedoid: boolean;
  createdAt: string;
}

export interface TrajectoryWindowClusterVersionRecord {
  id: string;
  clusterId: string;
  versionNo: number;
  membershipHash: string;
  supportHash: string;
  medoidOccurrenceId: string;
  supportEpisodeCount: number;
  counterexampleEpisodeCount: number;
  unknownEpisodeCount: number;
  metrics: Record<string, unknown>;
  status: VersionLifecycleStatus;
  supersedesVersionId?: string;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface CommitTrajectoryWindowClusterVersionInput {
  clusterId: string;
  /** null means that the caller observed an empty cluster head. */
  expectedActiveVersionId: string | null;
  medoidOccurrenceId: string;
  members: TrajectoryWindowClusterMemberInput[];
  metrics?: Record<string, unknown>;
  createdAt: string;
}

export interface CommitTrajectoryWindowClusterVersionResult {
  record: TrajectoryWindowClusterVersionRecord;
  members: TrajectoryWindowClusterMemberRecord[];
  created: boolean;
  reactivated: boolean;
  previousActive?: TrajectoryWindowClusterVersionRecord;
}

export interface ActiveTrajectoryClusterMedoid {
  cluster: TrajectoryWindowClusterRecord;
  version: TrajectoryWindowClusterVersionRecord;
  occurrence: TrajectoryWindowOccurrenceRecord;
}

export interface TrajectoryWindowClusterCanonicalKeyRecord {
  id: string;
  userId: string;
  scale: number;
  algorithmVersion: string;
  configHash: string;
  embeddingSignature: string;
  evidenceSignature: string;
  clusterId: string;
  createdAt: string;
  updatedAt: string;
}

export interface ResolveCanonicalTrajectoryWindowClusterInput {
  userId: string;
  scale: number;
  algorithmVersion: string;
  configHash: string;
  embeddingSignature: string;
  /** Stable signature of the fine-aligned evidence, independent of its coarse Family. */
  evidenceSignature: string;
  seedOccurrenceId: string;
  createdAt: string;
}

export interface ResolveCanonicalTrajectoryWindowClusterResult {
  cluster: TrajectoryWindowClusterRecord;
  canonicalKey: TrajectoryWindowClusterCanonicalKeyRecord;
  created: boolean;
}

export interface TrajectoryWindowFamilyClusterLinkRecord {
  id: string;
  familyRevisionId: string;
  canonicalKeyId: string;
  clusterVersionId: string;
  createdAt: string;
}

export interface TrajectorySkillVersionRecord {
  id: string;
  clusterId: string;
  clusterVersionId: string;
  versionNo: number;
  skillKey: string;
  membershipHash: string;
  supportHash: string;
  contentHash: string;
  skillMemoryId?: string;
  status: VersionLifecycleStatus;
  payload: Record<string, unknown>;
  supersedesVersionId?: string;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface SaveTrajectorySkillVersionInput {
  clusterId: string;
  clusterVersionId: string;
  expectedActiveSkillVersionId: string | null;
  skillKey: string;
  /** A formal upstream Skill memory; omitted while the candidate is not materialized. */
  skillMemoryId?: string;
  payload: Record<string, unknown>;
  contentHash?: string;
  createdAt: string;
}

export interface SaveTrajectorySkillVersionResult {
  record: TrajectorySkillVersionRecord;
  created: boolean;
  reactivated: boolean;
  previousActive?: TrajectorySkillVersionRecord;
}

export class ProceduralTrajectoryCasError extends Error {
  constructor(
    readonly targetId: string,
    readonly expectedVersionId: string | null,
    readonly actualVersionId: string | null
  ) {
    super(
      `procedural trajectory CAS conflict for ${targetId}: expected ` +
      `${expectedVersionId ?? "empty"}, found ${actualVersionId ?? "empty"}`
    );
    this.name = "ProceduralTrajectoryCasError";
  }
}

interface EpisodeSourceRow {
  id: string;
  user_id: string;
  raw_turn_ids_json: string;
  session_source: string;
  r_task: number | null;
}

interface PathSqlRow {
  id: string;
  episode_id: string;
  user_id: string;
  schema_version: string;
  compilation_key: string;
  source_snapshot_hash: string;
  path_hash: string;
  compiler_version: string;
  model_signature: string | null;
  source_raw_turn_ids_json: string;
  source_agent_ids_json: string;
  step_count: number;
  status: VersionLifecycleStatus;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface StepEmbeddingSqlRow {
  id: string;
  path_id: string;
  step_id: string;
  step_index: number;
  representation_version: string;
  embedding_signature: string;
  semantic_hash: string;
  embedding_dim: number;
  vector_json: string;
  created_at: string;
}

interface WindowSqlRow {
  id: string;
  path_id: string;
  episode_id: string;
  user_id: string;
  terminal_reward: number | null;
  evidence_role: ProceduralEvidenceRole;
  schema_version: string;
  window_config_hash: string;
  scale: number;
  stride: number;
  start_step_index: number;
  end_step_index: number;
  step_ids_json: string;
  raw_turn_ids_json: string;
  semantic_text: string;
  semantic_hash: string;
  coarse_representation_version: string;
  embedding_signature: string;
  embedding_dim: number;
  coarse_vector_json: string;
  created_at: string;
}

interface FamilySqlRow {
  id: string;
  user_id: string;
  scale: number;
  algorithm_version: string;
  config_hash: string;
  embedding_signature: string;
  status: "active" | "retired";
  active_revision_id: string | null;
  created_at: string;
  updated_at: string;
}

interface FamilyRevisionSqlRow {
  id: string;
  family_id: string;
  revision_no: number;
  membership_hash: string;
  evidence_hash: string;
  medoid_occurrence_id: string;
  metrics_json: string;
  status: VersionLifecycleStatus;
  supersedes_revision_id: string | null;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface FamilyMemberSqlRow {
  id: string;
  family_revision_id: string;
  occurrence_id: string;
  episode_id: string;
  user_id: string;
  coarse_similarity: number;
  is_medoid: number;
  created_at: string;
}

interface ClusterSqlRow {
  id: string;
  user_id: string;
  scale: number;
  algorithm_version: string;
  config_hash: string;
  status: "active" | "retired";
  active_version_id: string | null;
  active_skill_version_id: string | null;
  active_skill_memory_id: string | null;
  created_at: string;
  updated_at: string;
}

interface ClusterVersionSqlRow {
  id: string;
  cluster_id: string;
  version_no: number;
  membership_hash: string;
  support_hash: string;
  medoid_occurrence_id: string;
  support_episode_count: number;
  counterexample_episode_count: number;
  unknown_episode_count: number;
  metrics_json: string;
  status: VersionLifecycleStatus;
  supersedes_version_id: string | null;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface ClusterMemberSqlRow {
  id: string;
  cluster_version_id: string;
  occurrence_id: string;
  episode_id: string;
  user_id: string;
  evidence_role: ProceduralEvidenceRole;
  reward_hash: string;
  terminal_reward: number | null;
  coarse_similarity: number;
  alignment_json: string;
  is_medoid: number;
  created_at: string;
}

interface ClusterCanonicalKeySqlRow {
  id: string;
  user_id: string;
  scale: number;
  algorithm_version: string;
  config_hash: string;
  embedding_signature: string;
  evidence_signature: string;
  cluster_id: string;
  created_at: string;
  updated_at: string;
}

interface FamilyClusterLinkSqlRow {
  id: string;
  family_revision_id: string;
  canonical_key_id: string;
  cluster_version_id: string;
  created_at: string;
}

interface SkillVersionSqlRow {
  id: string;
  cluster_id: string;
  cluster_version_id: string;
  version_no: number;
  skill_key: string;
  membership_hash: string;
  support_hash: string;
  content_hash: string;
  skill_memory_id: string | null;
  status: VersionLifecycleStatus;
  payload_json: string;
  supersedes_version_id: string | null;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

type ClassifiedTrajectoryWindowClusterMember = TrajectoryWindowClusterMemberInput & {
  evidenceRole: ProceduralEvidenceRole;
  terminalReward?: number;
};

export class ProceduralTrajectoryRepository {
  constructor(private readonly db: Database.Database) {}

  savePathVersion(input: SaveEpisodeExecutionPathInput): SaveEpisodeExecutionPathResult {
    const path = input.path;
    validateEpisodeExecutionPathLite(path);
    if (!path.compilerVersion?.trim()) {
      throw new Error("execution path requires compilerVersion before persistence");
    }
    const episode = this.requireEpisode(path.episodeId);
    if (episode.user_id !== path.userId) {
      throw new Error(`execution path user does not own episode: ${path.episodeId}`);
    }
    const sourceRawTurnIds = uniqueStrings(path.sourceRawTurnIds);
    if (sourceRawTurnIds.length !== path.sourceRawTurnIds.length) {
      throw new Error("execution path sourceRawTurnIds must be unique");
    }
    this.validateRawTurnSources(path.episodeId, path.userId, sourceRawTurnIds);
    const sourceAgentIds = uniqueStrings(input.sourceAgentIds ?? [episode.session_source]);
    const compilationKey = stableHash({
      schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
      episodeId: path.episodeId,
      sourceSnapshotHash: path.sourceSnapshotHash,
      compilerVersion: path.compilerVersion,
      modelSignature: path.modelSignature ?? null
    });
    const activate = input.activate ?? true;
    const hasExpectedActivePath = input.expectedActivePathId !== undefined;

    return this.db.transaction(() => {
      const existing = this.getPathByCompilationKey(path.episodeId, compilationKey);
      if (existing) {
        if (existing.pathHash !== path.pathHash) {
          throw new Error(`execution path compilation key collision: ${compilationKey}`);
        }
        if (activate && existing.status !== "active") {
          const previousActive = this.getActivePath(path.episodeId);
          if (hasExpectedActivePath) {
            assertCas(
              `${path.episodeId}:path`,
              input.expectedActivePathId ?? null,
              previousActive?.id ?? null
            );
          }
          if (previousActive) {
            this.db.prepare(
              `UPDATE episode_execution_paths
               SET status = 'superseded', deactivated_at = ?
               WHERE id = ? AND status = 'active'`
            ).run(input.createdAt, previousActive.id);
          }
          this.db.prepare(
            `UPDATE episode_execution_paths
             SET status = 'active', activated_at = ?, deactivated_at = NULL
             WHERE id = ?`
          ).run(input.createdAt, existing.id);
          return {
            record: this.getPath(existing.id)!,
            created: false,
            ...(previousActive ? { previousActive } : {})
          };
        }
        return { record: existing, created: false };
      }
      const previousActive = activate ? this.getActivePath(path.episodeId) : undefined;
      if (activate && hasExpectedActivePath) {
        assertCas(
          `${path.episodeId}:path`,
          input.expectedActivePathId ?? null,
          previousActive?.id ?? null
        );
      }
      if (activate && previousActive) {
        this.db.prepare(
          `UPDATE episode_execution_paths
           SET status = 'superseded', deactivated_at = ?
           WHERE id = ? AND status = 'active'`
        ).run(input.createdAt, previousActive.id);
      }
      this.db.prepare(
        `INSERT INTO episode_execution_paths (
          id, episode_id, user_id, schema_version, compilation_key,
          source_snapshot_hash, path_hash, compiler_version, model_signature,
          source_raw_turn_ids_json, source_agent_ids_json, step_count, status,
          payload_json, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)`
      ).run(
        path.id,
        path.episodeId,
        path.userId,
        EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
        compilationKey,
        path.sourceSnapshotHash,
        path.pathHash,
        path.compilerVersion,
        path.modelSignature ?? null,
        toJson(sourceRawTurnIds),
        toJson(sourceAgentIds),
        path.steps.length,
        activate ? "active" : "superseded",
        toJson(path),
        input.createdAt,
        activate ? input.createdAt : null
      );
      return {
        record: this.getPath(path.id)!,
        created: true,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  getPath(id: string): EpisodeExecutionPathRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM episode_execution_paths WHERE id = ?`)
      .get(id) as PathSqlRow | undefined;
    return row ? pathFromSql(row) : undefined;
  }

  getActivePath(episodeId: string): EpisodeExecutionPathRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_execution_paths
       WHERE episode_id = ? AND status = 'active' LIMIT 1`
    ).get(episodeId) as PathSqlRow | undefined;
    return row ? pathFromSql(row) : undefined;
  }

  listPathVersions(episodeId: string): EpisodeExecutionPathRecord[] {
    return (this.db.prepare(
      `SELECT * FROM episode_execution_paths
       WHERE episode_id = ? ORDER BY created_at DESC, id DESC`
    ).all(episodeId) as PathSqlRow[]).map(pathFromSql);
  }

  activatePathVersion(pathId: string, at: string): {
    record: EpisodeExecutionPathRecord;
    previousActive?: EpisodeExecutionPathRecord;
  } {
    return this.db.transaction(() => {
      const target = this.getPath(pathId);
      if (!target) throw new Error(`execution path version not found: ${pathId}`);
      if (target.status === "active") return { record: target };
      const previousActive = this.getActivePath(target.episodeId);
      if (previousActive) {
        this.db.prepare(
          `UPDATE episode_execution_paths
           SET status = 'superseded', deactivated_at = ?
           WHERE id = ? AND status = 'active'`
        ).run(at, previousActive.id);
      }
      this.db.prepare(
        `UPDATE episode_execution_paths
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(at, target.id);
      return {
        record: this.getPath(target.id)!,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  /**
   * Governance invalidation deliberately leaves the immutable Path payload and
   * its derived Windows in place for audit/rollback, while removing it from
   * the active evidence graph immediately. A concurrently activated newer Path
   * is never touched because the update is scoped to the observed version id.
   */
  deactivatePathVersion(pathId: string, at: string): EpisodeExecutionPathRecord {
    const target = this.getPath(pathId);
    if (!target) throw new Error(`execution path version not found: ${pathId}`);
    if (target.status !== "active") return target;
    this.db.prepare(
      `UPDATE episode_execution_paths
       SET status = 'superseded', deactivated_at = ?
       WHERE id = ? AND status = 'active'`
    ).run(at, target.id);
    return this.getPath(target.id)!;
  }

  upsertStepEmbedding(input: UpsertProceduralStepEmbeddingInput): {
    record: ProceduralStepEmbeddingRecord;
    created: boolean;
  } {
    const path = this.requirePath(input.pathId);
    const step = path.path.steps[input.stepIndex];
    if (!step || step.id !== input.stepId) {
      throw new Error(`Step embedding does not match path position: ${input.stepId}`);
    }
    const vector = validatedVector(input.vector, "Step embedding");
    const representationVersion = input.representationVersion ??
      PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION;
    const id = `procedural_step_embedding_${stableHash({
      pathId: input.pathId,
      stepId: input.stepId,
      representationVersion,
      embeddingSignature: input.embeddingSignature
    }).slice(0, 24)}`;
    const existing = this.getStepEmbeddingByIdentity(
      input.pathId,
      input.stepId,
      representationVersion,
      input.embeddingSignature
    );
    if (existing) {
      if (existing.semanticHash !== input.semanticHash) {
        throw new Error(`Step embedding semantic hash changed for immutable path Step: ${input.stepId}`);
      }
      return { record: existing, created: false };
    }
    this.db.prepare(
      `INSERT INTO procedural_step_embeddings (
        id, path_id, step_id, step_index, representation_version,
        embedding_signature, semantic_hash, embedding_dim, vector_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      id,
      input.pathId,
      input.stepId,
      input.stepIndex,
      representationVersion,
      input.embeddingSignature,
      input.semanticHash,
      vector.length,
      toJson(vector),
      input.createdAt
    );
    return { record: this.getStepEmbedding(id)!, created: true };
  }

  getStepEmbedding(id: string): ProceduralStepEmbeddingRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM procedural_step_embeddings WHERE id = ?`)
      .get(id) as StepEmbeddingSqlRow | undefined;
    return row ? stepEmbeddingFromSql(row) : undefined;
  }

  listStepEmbeddings(input: {
    pathId: string;
    representationVersion?: string;
    embeddingSignature?: string;
  }): ProceduralStepEmbeddingRecord[] {
    const clauses = ["path_id = ?"];
    const params: string[] = [input.pathId];
    if (input.representationVersion) {
      clauses.push("representation_version = ?");
      params.push(input.representationVersion);
    }
    if (input.embeddingSignature) {
      clauses.push("embedding_signature = ?");
      params.push(input.embeddingSignature);
    }
    return (this.db.prepare(
      `SELECT * FROM procedural_step_embeddings
       WHERE ${clauses.join(" AND ")}
       ORDER BY step_index ASC, id ASC`
    ).all(...params) as StepEmbeddingSqlRow[]).map(stepEmbeddingFromSql);
  }

  insertWindow(input: InsertTrajectoryWindowInput): {
    record: TrajectoryWindowOccurrenceRecord;
    created: boolean;
  } {
    const occurrence = input.occurrence;
    if (occurrence.schemaVersion !== TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION) {
      throw new Error(`unsupported trajectory window occurrence: ${occurrence.id}`);
    }
    const path = this.requirePath(occurrence.pathId);
    if (path.status !== "active") {
      throw new Error(`trajectory window requires the active execution path: ${path.id}`);
    }
    if (occurrence.pathId !== path.id || occurrence.episodeId !== path.episodeId ||
        occurrence.userId !== path.userId) {
      throw new Error("trajectory window occurrence is outside execution path scope");
    }
    const episode = this.requireEpisode(occurrence.episodeId);
    const currentTerminalReward = episode.r_task ?? undefined;
    const currentEvidenceRole = proceduralEvidenceRoleForReward(currentTerminalReward);
    if (occurrence.terminalReward !== currentTerminalReward ||
        occurrence.evidenceRole !== currentEvidenceRole) {
      throw new Error(
        `trajectory window evidence does not match current Episode reward: ${occurrence.id}`
      );
    }
    if (occurrence.scale < 2 || occurrence.stride < 1 ||
        occurrence.endStepIndex - occurrence.startStepIndex + 1 !== occurrence.scale) {
      throw new Error("trajectory window bounds do not match scale");
    }
    const expectedSteps = path.path.steps.slice(
      occurrence.startStepIndex,
      occurrence.endStepIndex + 1
    );
    if (expectedSteps.length !== occurrence.scale ||
        expectedSteps.some((step, index) => step.id !== occurrence.steps[index]?.id)) {
      throw new Error("trajectory window Step ids do not match active path interval");
    }
    const semanticHash = stableHash(occurrence.semanticText);
    const existing = this.getWindow(occurrence.id) ?? this.getWindowByIdentity(input);
    if (existing) {
      if (existing.id !== occurrence.id || existing.pathId !== occurrence.pathId ||
          existing.scale !== occurrence.scale ||
          existing.startStepIndex !== occurrence.startStepIndex ||
          existing.endStepIndex !== occurrence.endStepIndex ||
          existing.semanticHash !== semanticHash ||
          existing.embeddingSignature !== input.embeddingSignature ||
          existing.coarseRepresentationVersion !== input.coarseRepresentationVersion) {
        throw new Error(`trajectory window identity collision: ${existing.id}`);
      }
      if (existing.windowConfigHash !== input.windowConfigHash ||
          existing.stride !== occurrence.stride ||
          existing.terminalReward !== currentTerminalReward ||
          existing.evidenceRole !== currentEvidenceRole) {
        this.db.prepare(
          `UPDATE trajectory_window_occurrences
           SET window_config_hash = ?, stride = ?, terminal_reward = ?, evidence_role = ?
           WHERE id = ?`
        ).run(
          input.windowConfigHash,
          occurrence.stride,
          currentTerminalReward ?? null,
          currentEvidenceRole,
          existing.id
        );
      }
      return { record: this.getWindow(existing.id)!, created: false };
    }
    if (!input.coarseVector) {
      throw new Error("new trajectory window requires a coarse embedding");
    }
    const coarseVector = validatedVector(input.coarseVector, "trajectory window coarse embedding");
    this.db.prepare(
      `INSERT INTO trajectory_window_occurrences (
        id, path_id, episode_id, user_id, terminal_reward, evidence_role,
        schema_version, window_config_hash,
        scale, stride, start_step_index, end_step_index, step_ids_json,
        raw_turn_ids_json, semantic_text, semantic_hash,
        coarse_representation_version, embedding_signature, embedding_dim,
        coarse_vector_json, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      occurrence.id,
      path.id,
      path.episodeId,
      path.userId,
      occurrence.terminalReward ?? null,
      occurrence.evidenceRole,
      TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION,
      input.windowConfigHash,
      occurrence.scale,
      occurrence.stride,
      occurrence.startStepIndex,
      occurrence.endStepIndex,
      toJson(occurrence.steps.map((step) => step.id)),
      toJson(uniqueStrings(occurrence.steps.map((step) => step.rawTurnId))),
      occurrence.semanticText,
      semanticHash,
      input.coarseRepresentationVersion,
      input.embeddingSignature,
      coarseVector.length,
      toJson(coarseVector),
      input.createdAt
    );
    return { record: this.getWindow(occurrence.id)!, created: true };
  }

  getWindow(id: string): TrajectoryWindowOccurrenceRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM trajectory_window_occurrences WHERE id = ?`)
      .get(id) as WindowSqlRow | undefined;
    return row ? windowFromSql(row) : undefined;
  }

  listWindowsForPath(pathId: string, scale?: number): TrajectoryWindowOccurrenceRecord[] {
    const rows = scale === undefined
      ? this.db.prepare(
          `SELECT * FROM trajectory_window_occurrences
           WHERE path_id = ? ORDER BY scale ASC, start_step_index ASC, id ASC`
        ).all(pathId)
      : this.db.prepare(
          `SELECT * FROM trajectory_window_occurrences
           WHERE path_id = ? AND scale = ? ORDER BY start_step_index ASC, id ASC`
        ).all(pathId, scale);
    return (rows as WindowSqlRow[]).map(windowFromSql);
  }

  createFamilyHead(input: CreateTrajectoryWindowFamilyInput): {
    record: TrajectoryWindowFamilyRecord;
    created: boolean;
  } {
    const seed = this.requireActiveWindow(input.seedOccurrenceId);
    if (seed.userId !== input.userId || seed.scale !== input.scale ||
        seed.embeddingSignature !== input.embeddingSignature) {
      throw new Error("trajectory Family seed is outside user, scale, or embedding scope");
    }
    const id = input.id ?? `trajectory_window_family_${stableHash({
      userId: input.userId,
      scale: input.scale,
      algorithmVersion: input.algorithmVersion,
      configHash: input.configHash,
      embeddingSignature: input.embeddingSignature,
      seedOccurrenceId: seed.id
    }).slice(0, 24)}`;
    const existing = this.getFamilyHead(id);
    if (existing) {
      if (existing.userId !== input.userId || existing.scale !== input.scale ||
          existing.algorithmVersion !== input.algorithmVersion ||
          existing.configHash !== input.configHash ||
          existing.embeddingSignature !== input.embeddingSignature) {
        throw new Error(`trajectory Family id collision: ${id}`);
      }
      return { record: existing, created: false };
    }
    this.db.prepare(
      `INSERT INTO trajectory_window_families (
        id, user_id, scale, algorithm_version, config_hash, embedding_signature,
        status, active_revision_id, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?)`
    ).run(
      id,
      input.userId,
      input.scale,
      input.algorithmVersion,
      input.configHash,
      input.embeddingSignature,
      input.createdAt,
      input.createdAt
    );
    return { record: this.getFamilyHead(id)!, created: true };
  }

  getFamilyHead(id: string): TrajectoryWindowFamilyRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM trajectory_window_families WHERE id = ?`)
      .get(id) as FamilySqlRow | undefined;
    return row ? familyFromSql(row) : undefined;
  }

  listActiveFamilyHeads(input: {
    userId: string;
    scale: number;
    algorithmVersion: string;
    configHash: string;
    embeddingSignature: string;
  }): TrajectoryWindowFamilyRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_families
       WHERE user_id = ? AND scale = ? AND algorithm_version = ?
         AND config_hash = ? AND embedding_signature = ? AND status = 'active'
       ORDER BY created_at ASC, id ASC`
    ).all(
      input.userId,
      input.scale,
      input.algorithmVersion,
      input.configHash,
      input.embeddingSignature
    ) as FamilySqlRow[]).map(familyFromSql);
  }

  listActiveFamilyMedoids(input: {
    userId: string;
    scale: number;
    algorithmVersion: string;
    configHash: string;
    embeddingSignature: string;
  }): ActiveTrajectoryWindowFamilyMedoid[] {
    const rows = this.db.prepare(
      `SELECT
         families.id AS family_id,
         revisions.id AS revision_id,
         occurrences.id AS occurrence_id
       FROM trajectory_window_families AS families
       JOIN trajectory_window_family_revisions AS revisions
         ON revisions.id = families.active_revision_id AND revisions.status = 'active'
       JOIN trajectory_window_occurrences AS occurrences
         ON occurrences.id = revisions.medoid_occurrence_id
       JOIN episode_execution_paths AS paths
         ON paths.id = occurrences.path_id AND paths.status = 'active'
       WHERE families.user_id = ? AND families.scale = ?
         AND families.algorithm_version = ? AND families.config_hash = ?
         AND families.embedding_signature = ? AND families.status = 'active'
       ORDER BY families.created_at ASC, families.id ASC`
    ).all(
      input.userId,
      input.scale,
      input.algorithmVersion,
      input.configHash,
      input.embeddingSignature
    ) as Array<{ family_id: string; revision_id: string; occurrence_id: string }>;
    return rows.map((row) => ({
      family: this.getFamilyHead(row.family_id)!,
      revision: this.getFamilyRevision(row.revision_id)!,
      occurrence: this.getWindow(row.occurrence_id)!
    }));
  }

  commitFamilyRevision(
    input: CommitTrajectoryWindowFamilyRevisionInput
  ): CommitTrajectoryWindowFamilyRevisionResult {
    if (input.members.length === 0) {
      throw new Error("trajectory Family revision requires at least one member");
    }
    if (!input.evidenceHash.trim()) {
      throw new Error("trajectory Family revision requires evidenceHash");
    }
    const family = this.requireFamily(input.familyId);
    if (family.status !== "active") {
      throw new Error(`trajectory Family is retired: ${family.id}`);
    }
    const requestedMembers = [...input.members]
      .sort((left, right) => left.occurrenceId.localeCompare(right.occurrenceId));
    if (new Set(requestedMembers.map((member) => member.occurrenceId)).size !== requestedMembers.length) {
      throw new Error("trajectory Family revision contains duplicate occurrence ids");
    }
    const occurrenceById = new Map(requestedMembers.map((member) => {
      validateFamilyMemberInput(member);
      const occurrence = this.requireActiveWindow(member.occurrenceId);
      if (occurrence.userId !== family.userId || occurrence.scale !== family.scale ||
          occurrence.embeddingSignature !== family.embeddingSignature) {
        throw new Error(`trajectory Family member is outside Family scope: ${occurrence.id}`);
      }
      return [member.occurrenceId, occurrence];
    }));
    if (!occurrenceById.has(input.medoidOccurrenceId)) {
      throw new Error("trajectory Family medoid must be a real member occurrence");
    }
    const membershipHash = stableHash({
      schemaVersion: TRAJECTORY_WINDOW_FAMILY_REVISION_SCHEMA_VERSION,
      familyId: family.id,
      medoidOccurrenceId: input.medoidOccurrenceId,
      evidenceHash: input.evidenceHash,
      occurrenceIds: requestedMembers.map((member) => member.occurrenceId)
    });
    const existing = this.getFamilyRevisionByMembershipHash(family.id, membershipHash);

    return this.db.transaction(() => {
      const current = this.requireFamily(family.id);
      if (existing && current.activeRevisionId === existing.id) {
        return {
          record: existing,
          members: this.listFamilyMembers(existing.id),
          created: false,
          reactivated: false
        };
      }
      assertCas(current.id, input.expectedActiveRevisionId, current.activeRevisionId ?? null);
      const previousActive = current.activeRevisionId
        ? this.getFamilyRevision(current.activeRevisionId)
        : undefined;
      if (previousActive) this.supersedeFamilyRevision(previousActive.id, input.createdAt);
      if (existing) {
        this.assertFamilyRevisionSourcesActive(existing.id);
        this.db.prepare(
          `UPDATE trajectory_window_family_revisions
           SET status = 'active', activated_at = ?, deactivated_at = NULL
           WHERE id = ?`
        ).run(input.createdAt, existing.id);
        this.updateFamilyRevisionHead(current.id, existing.id, input.createdAt);
        return {
          record: this.getFamilyRevision(existing.id)!,
          members: this.listFamilyMembers(existing.id),
          created: false,
          reactivated: true,
          ...(previousActive ? { previousActive } : {})
        };
      }

      const revisionNo = this.nextFamilyRevisionNo(current.id);
      const id = `trajectory_window_family_revision_${stableHash({
        familyId: current.id,
        membershipHash
      }).slice(0, 24)}`;
      this.db.prepare(
        `INSERT INTO trajectory_window_family_revisions (
          id, family_id, revision_no, membership_hash, evidence_hash, medoid_occurrence_id,
          metrics_json, status, supersedes_revision_id, created_at,
          activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL)`
      ).run(
        id,
        current.id,
        revisionNo,
        membershipHash,
        input.evidenceHash,
        input.medoidOccurrenceId,
        toJson(input.metrics ?? {}),
        previousActive?.id ?? null,
        input.createdAt,
        input.createdAt
      );
      const insertMember = this.db.prepare(
        `INSERT INTO trajectory_window_family_members (
          id, family_revision_id, occurrence_id, episode_id, user_id,
          coarse_similarity, is_medoid, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const member of requestedMembers) {
        const occurrence = occurrenceById.get(member.occurrenceId)!;
        const memberId = `trajectory_window_family_member_${stableHash({
          familyRevisionId: id,
          occurrenceId: occurrence.id
        }).slice(0, 24)}`;
        insertMember.run(
          memberId,
          id,
          occurrence.id,
          occurrence.episodeId,
          occurrence.userId,
          member.coarseSimilarity,
          occurrence.id === input.medoidOccurrenceId ? 1 : 0,
          input.createdAt
        );
      }
      this.updateFamilyRevisionHead(current.id, id, input.createdAt);
      return {
        record: this.getFamilyRevision(id)!,
        members: this.listFamilyMembers(id),
        created: true,
        reactivated: false,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  getFamilyRevision(id: string): TrajectoryWindowFamilyRevisionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_family_revisions WHERE id = ?`
    ).get(id) as FamilyRevisionSqlRow | undefined;
    return row ? familyRevisionFromSql(row) : undefined;
  }

  listFamilyRevisions(familyId: string): TrajectoryWindowFamilyRevisionRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_family_revisions
       WHERE family_id = ? ORDER BY revision_no DESC, id DESC`
    ).all(familyId) as FamilyRevisionSqlRow[]).map(familyRevisionFromSql);
  }

  listFamilyMembers(familyRevisionId: string): TrajectoryWindowFamilyMemberRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_family_members
       WHERE family_revision_id = ? ORDER BY occurrence_id ASC`
    ).all(familyRevisionId) as FamilyMemberSqlRow[]).map(familyMemberFromSql);
  }

  activateFamilyRevision(input: {
    familyId: string;
    revisionId: string;
    expectedActiveRevisionId: string | null;
    activatedAt: string;
  }): CommitTrajectoryWindowFamilyRevisionResult {
    return this.db.transaction(() => {
      const family = this.requireFamily(input.familyId);
      const target = this.requireFamilyRevision(input.revisionId);
      if (target.familyId !== family.id) {
        throw new Error(`Family revision does not belong to Family: ${target.id}`);
      }
      if (family.activeRevisionId === target.id) {
        return {
          record: target,
          members: this.listFamilyMembers(target.id),
          created: false,
          reactivated: false
        };
      }
      assertCas(family.id, input.expectedActiveRevisionId, family.activeRevisionId ?? null);
      this.assertFamilyRevisionSourcesActive(target.id);
      const previousActive = family.activeRevisionId
        ? this.getFamilyRevision(family.activeRevisionId)
        : undefined;
      if (previousActive) this.supersedeFamilyRevision(previousActive.id, input.activatedAt);
      this.db.prepare(
        `UPDATE trajectory_window_family_revisions
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(input.activatedAt, target.id);
      this.updateFamilyRevisionHead(family.id, target.id, input.activatedAt);
      return {
        record: this.getFamilyRevision(target.id)!,
        members: this.listFamilyMembers(target.id),
        created: false,
        reactivated: true,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  listAffectedFamilyIdsForOccurrences(occurrenceIds: readonly string[]): string[] {
    const ids = uniqueStrings(occurrenceIds);
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => "?").join(", ");
    return (this.db.prepare(
      `SELECT DISTINCT families.id AS id
       FROM trajectory_window_family_members AS members
       JOIN trajectory_window_family_revisions AS revisions
         ON revisions.id = members.family_revision_id AND revisions.status = 'active'
       JOIN trajectory_window_families AS families
         ON families.active_revision_id = revisions.id AND families.status = 'active'
       WHERE members.occurrence_id IN (${placeholders})
       ORDER BY families.id ASC`
    ).all(...ids) as Array<{ id: string }>).map((row) => row.id);
  }

  listAffectedFamilyIdsForPath(pathId: string): string[] {
    return (this.db.prepare(
      `SELECT DISTINCT families.id AS id
       FROM trajectory_window_family_members AS members
       JOIN trajectory_window_family_revisions AS revisions
         ON revisions.id = members.family_revision_id AND revisions.status = 'active'
       JOIN trajectory_window_families AS families
         ON families.active_revision_id = revisions.id AND families.status = 'active'
       JOIN trajectory_window_occurrences AS occurrences
         ON occurrences.id = members.occurrence_id
       WHERE occurrences.path_id = ?
       ORDER BY families.id ASC`
    ).all(pathId) as Array<{ id: string }>).map((row) => row.id);
  }

  retireFamily(
    familyId: string,
    expectedActiveRevisionId: string | null,
    retiredAt: string
  ): TrajectoryWindowFamilyRecord {
    return this.db.transaction(() => {
      const family = this.requireFamily(familyId);
      assertCas(family.id, expectedActiveRevisionId, family.activeRevisionId ?? null);
      if (family.status === "retired") return family;
      this.db.prepare(
        `UPDATE trajectory_window_families
         SET status = 'retired', updated_at = ?
         WHERE id = ?`
      ).run(retiredAt, family.id);
      return this.requireFamily(family.id);
    })();
  }

  createClusterHead(input: CreateTrajectoryWindowClusterInput): {
    record: TrajectoryWindowClusterRecord;
    created: boolean;
  } {
    const seed = this.requireActiveWindow(input.seedOccurrenceId);
    if (seed.userId !== input.userId || seed.scale !== input.scale) {
      throw new Error("trajectory cluster seed is outside user or scale scope");
    }
    const id = input.id ?? `trajectory_window_cluster_${stableHash({
      userId: input.userId,
      scale: input.scale,
      algorithmVersion: input.algorithmVersion,
      configHash: input.configHash,
      seedOccurrenceId: seed.id
    }).slice(0, 24)}`;
    const existing = this.getClusterHead(id);
    if (existing) {
      if (existing.userId !== input.userId || existing.scale !== input.scale ||
          existing.algorithmVersion !== input.algorithmVersion ||
          existing.configHash !== input.configHash) {
        throw new Error(`trajectory cluster id collision: ${id}`);
      }
      return { record: existing, created: false };
    }
    this.db.prepare(
      `INSERT INTO trajectory_window_clusters (
        id, user_id, scale, algorithm_version, config_hash, status,
        active_version_id, active_skill_version_id, active_skill_memory_id,
        created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, 'active', NULL, NULL, NULL, ?, ?)`
    ).run(
      id,
      input.userId,
      input.scale,
      input.algorithmVersion,
      input.configHash,
      input.createdAt,
      input.createdAt
    );
    return { record: this.getClusterHead(id)!, created: true };
  }

  getClusterHead(id: string): TrajectoryWindowClusterRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM trajectory_window_clusters WHERE id = ?`)
      .get(id) as ClusterSqlRow | undefined;
    return row ? clusterFromSql(row) : undefined;
  }

  listActiveClusterHeads(input: {
    userId: string;
    scale: number;
    algorithmVersion: string;
    configHash: string;
  }): TrajectoryWindowClusterRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_clusters
       WHERE user_id = ? AND scale = ? AND algorithm_version = ?
         AND config_hash = ? AND status = 'active'
       ORDER BY created_at ASC, id ASC`
    ).all(
      input.userId,
      input.scale,
      input.algorithmVersion,
      input.configHash
    ) as ClusterSqlRow[]).map(clusterFromSql);
  }

  listActiveClusterMedoids(input: {
    userId: string;
    scale: number;
    algorithmVersion: string;
    configHash: string;
    embeddingSignature: string;
  }): ActiveTrajectoryClusterMedoid[] {
    const rows = this.db.prepare(
      `SELECT
         clusters.id AS cluster_id,
         versions.id AS version_id,
         occurrences.id AS occurrence_id
       FROM trajectory_window_clusters AS clusters
       JOIN trajectory_window_cluster_versions AS versions
         ON versions.id = clusters.active_version_id AND versions.status = 'active'
       JOIN trajectory_window_occurrences AS occurrences
         ON occurrences.id = versions.medoid_occurrence_id
       JOIN episode_execution_paths AS paths
         ON paths.id = occurrences.path_id AND paths.status = 'active'
       WHERE clusters.user_id = ? AND clusters.scale = ?
         AND clusters.algorithm_version = ? AND clusters.config_hash = ?
         AND clusters.status = 'active' AND occurrences.embedding_signature = ?
       ORDER BY clusters.created_at ASC, clusters.id ASC`
    ).all(
      input.userId,
      input.scale,
      input.algorithmVersion,
      input.configHash,
      input.embeddingSignature
    ) as Array<{ cluster_id: string; version_id: string; occurrence_id: string }>;
    return rows.map((row) => ({
      cluster: this.getClusterHead(row.cluster_id)!,
      version: this.getClusterVersion(row.version_id)!,
      occurrence: this.getWindow(row.occurrence_id)!
    }));
  }

  /**
   * Resolves a fine evidence cluster independently of the coarse Family that
   * discovered it. Callers must derive evidenceSignature from the fine-aligned
   * evidence/core, never from a Family id. The unique scoped key serializes
   * concurrent claims and keeps one canonical cluster head reusable by many
   * Family revisions.
   */
  resolveCanonicalClusterHead(
    input: ResolveCanonicalTrajectoryWindowClusterInput
  ): ResolveCanonicalTrajectoryWindowClusterResult {
    if (!input.evidenceSignature.trim()) {
      throw new Error("canonical trajectory cluster requires evidenceSignature");
    }
    const seed = this.requireActiveWindow(input.seedOccurrenceId);
    if (seed.userId !== input.userId || seed.scale !== input.scale ||
        seed.embeddingSignature !== input.embeddingSignature) {
      throw new Error("canonical trajectory cluster seed is outside scoped evidence");
    }
    return this.db.transaction(() => {
      const existing = this.findClusterCanonicalKey(input);
      if (existing) {
        const cluster = this.requireCluster(existing.clusterId);
        if (cluster.status === "retired") {
          this.db.prepare(
            `UPDATE trajectory_window_clusters
             SET status = 'active', updated_at = ?
             WHERE id = ? AND status = 'retired'`
          ).run(input.createdAt, cluster.id);
        }
        return {
          cluster: this.requireCluster(cluster.id),
          canonicalKey: existing,
          created: false
        };
      }
      const canonicalBasis = {
        userId: input.userId,
        scale: input.scale,
        algorithmVersion: input.algorithmVersion,
        configHash: input.configHash,
        embeddingSignature: input.embeddingSignature,
        evidenceSignature: input.evidenceSignature
      };
      const keyId = `trajectory_window_cluster_canonical_${stableHash(canonicalBasis).slice(0, 24)}`;
      const clusterId = `trajectory_window_cluster_${stableHash({
        canonicalBasis,
        kind: "canonical-fine-evidence"
      }).slice(0, 24)}`;
      const cluster = this.createClusterHead({
        id: clusterId,
        userId: input.userId,
        scale: input.scale,
        algorithmVersion: input.algorithmVersion,
        configHash: input.configHash,
        seedOccurrenceId: seed.id,
        createdAt: input.createdAt
      }).record;
      this.db.prepare(
        `INSERT INTO trajectory_window_cluster_canonical_keys (
          id, user_id, scale, algorithm_version, config_hash,
          embedding_signature, evidence_signature, cluster_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        keyId,
        input.userId,
        input.scale,
        input.algorithmVersion,
        input.configHash,
        input.embeddingSignature,
        input.evidenceSignature,
        cluster.id,
        input.createdAt,
        input.createdAt
      );
      return {
        cluster,
        canonicalKey: this.getClusterCanonicalKey(keyId)!,
        created: true
      };
    })();
  }

  getClusterCanonicalKey(id: string): TrajectoryWindowClusterCanonicalKeyRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_cluster_canonical_keys WHERE id = ?`
    ).get(id) as ClusterCanonicalKeySqlRow | undefined;
    return row ? clusterCanonicalKeyFromSql(row) : undefined;
  }

  listClusterCanonicalKeys(clusterId: string): TrajectoryWindowClusterCanonicalKeyRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_cluster_canonical_keys
       WHERE cluster_id = ? ORDER BY created_at ASC, id ASC`
    ).all(clusterId) as ClusterCanonicalKeySqlRow[]).map(clusterCanonicalKeyFromSql);
  }

  linkFamilyRevisionToCluster(input: {
    familyRevisionId: string;
    canonicalKeyId: string;
    clusterVersionId: string;
    createdAt: string;
  }): { record: TrajectoryWindowFamilyClusterLinkRecord; created: boolean } {
    return this.db.transaction(() => {
      const revision = this.requireFamilyRevision(input.familyRevisionId);
      const family = this.requireFamily(revision.familyId);
      assertCas(family.id, revision.id, family.activeRevisionId ?? null);
      const canonicalKey = this.requireClusterCanonicalKey(input.canonicalKeyId);
      const cluster = this.requireCluster(canonicalKey.clusterId);
      const clusterVersion = this.requireClusterVersion(input.clusterVersionId);
      if (clusterVersion.clusterId !== cluster.id) {
        throw new Error("Family link cluster version does not match canonical cluster");
      }
      assertCas(cluster.id, clusterVersion.id, cluster.activeVersionId ?? null);
      if (family.userId !== canonicalKey.userId || family.scale !== canonicalKey.scale ||
          family.algorithmVersion !== canonicalKey.algorithmVersion ||
          family.configHash !== canonicalKey.configHash ||
          family.embeddingSignature !== canonicalKey.embeddingSignature) {
        throw new Error("Family link canonical cluster is outside Family scope");
      }
      const existing = this.db.prepare(
        `SELECT * FROM trajectory_window_family_cluster_links
         WHERE family_revision_id = ? AND canonical_key_id = ? LIMIT 1`
      ).get(revision.id, canonicalKey.id) as FamilyClusterLinkSqlRow | undefined;
      if (existing) {
        if (existing.cluster_version_id !== clusterVersion.id) {
          throw new Error("immutable Family-to-cluster link already targets another version");
        }
        return { record: familyClusterLinkFromSql(existing), created: false };
      }
      const id = `trajectory_window_family_cluster_link_${stableHash({
        familyRevisionId: revision.id,
        canonicalKeyId: canonicalKey.id
      }).slice(0, 24)}`;
      this.db.prepare(
        `INSERT INTO trajectory_window_family_cluster_links (
          id, family_revision_id, canonical_key_id, cluster_version_id, created_at
        ) VALUES (?, ?, ?, ?, ?)`
      ).run(id, revision.id, canonicalKey.id, clusterVersion.id, input.createdAt);
      return { record: this.getFamilyClusterLink(id)!, created: true };
    })();
  }

  getFamilyClusterLink(id: string): TrajectoryWindowFamilyClusterLinkRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_family_cluster_links WHERE id = ?`
    ).get(id) as FamilyClusterLinkSqlRow | undefined;
    return row ? familyClusterLinkFromSql(row) : undefined;
  }

  listFamilyClusterLinks(familyRevisionId: string): TrajectoryWindowFamilyClusterLinkRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_family_cluster_links
       WHERE family_revision_id = ? ORDER BY created_at ASC, id ASC`
    ).all(familyRevisionId) as FamilyClusterLinkSqlRow[]).map(familyClusterLinkFromSql);
  }

  listClusterFamilyLinks(clusterId: string): TrajectoryWindowFamilyClusterLinkRecord[] {
    return (this.db.prepare(
      `SELECT links.*
       FROM trajectory_window_family_cluster_links AS links
       JOIN trajectory_window_cluster_canonical_keys AS canonical
         ON canonical.id = links.canonical_key_id
       WHERE canonical.cluster_id = ?
       ORDER BY links.created_at ASC, links.id ASC`
    ).all(clusterId) as FamilyClusterLinkSqlRow[]).map(familyClusterLinkFromSql);
  }

  commitClusterVersion(
    input: CommitTrajectoryWindowClusterVersionInput
  ): CommitTrajectoryWindowClusterVersionResult {
    if (input.members.length === 0) {
      throw new Error("trajectory cluster version requires at least one member");
    }
    const cluster = this.requireCluster(input.clusterId);
    const requestedMembers = [...input.members]
      .sort((left, right) => left.occurrenceId.localeCompare(right.occurrenceId));
    if (new Set(requestedMembers.map((member) => member.occurrenceId)).size !== requestedMembers.length) {
      throw new Error("trajectory cluster version contains duplicate occurrence ids");
    }
    const occurrenceById = new Map(requestedMembers.map((member) => {
      validateMemberInput(member);
      const occurrence = this.requireActiveWindow(member.occurrenceId);
      if (occurrence.userId !== cluster.userId || occurrence.scale !== cluster.scale) {
        throw new Error(`trajectory cluster member is outside cluster scope: ${occurrence.id}`);
      }
      return [member.occurrenceId, occurrence];
    }));
    const normalizedMembers: ClassifiedTrajectoryWindowClusterMember[] = requestedMembers
      .map((member) => {
        const occurrence = occurrenceById.get(member.occurrenceId)!;
        return {
          ...member,
          evidenceRole: occurrence.evidenceRole,
          ...(occurrence.terminalReward === undefined
            ? {}
            : { terminalReward: occurrence.terminalReward })
        };
      });
    const medoidMember = normalizedMembers.find((member) =>
      member.occurrenceId === input.medoidOccurrenceId);
    if (!medoidMember) throw new Error("trajectory cluster medoid must be a member");
    const supportMembers = normalizedMembers.filter((member) => member.evidenceRole === "support");
    if (supportMembers.length > 0 && medoidMember.evidenceRole !== "support") {
      throw new Error("positive trajectory cluster medoid must come from support evidence");
    }
    const membershipBasis = normalizedMembers.map((member) => ({
      occurrenceId: member.occurrenceId,
      evidenceRole: member.evidenceRole,
      rewardHash: member.rewardHash
    }));
    const membershipHash = stableHash({
      schemaVersion: TRAJECTORY_CLUSTER_VERSION_SCHEMA_VERSION,
      clusterId: cluster.id,
      members: membershipBasis
    });
    const supportHash = stableHash(membershipBasis.filter((member) =>
      member.evidenceRole === "support"));
    const existing = this.getClusterVersionByMembershipHash(cluster.id, membershipHash);

    return this.db.transaction(() => {
      const current = this.requireCluster(cluster.id);
      if (existing && current.activeVersionId === existing.id) {
        return {
          record: existing,
          members: this.listClusterMembers(existing.id),
          created: false,
          reactivated: false
        };
      }
      assertCas(current.id, input.expectedActiveVersionId, current.activeVersionId ?? null);
      const previousActive = current.activeVersionId
        ? this.getClusterVersion(current.activeVersionId)
        : undefined;
      if (previousActive) this.supersedeClusterVersion(previousActive.id, input.createdAt);
      if (existing) {
        this.db.prepare(
          `UPDATE trajectory_window_cluster_versions
           SET status = 'active', activated_at = ?, deactivated_at = NULL
           WHERE id = ?`
        ).run(input.createdAt, existing.id);
        this.updateClusterVersionHead(current.id, existing.id, input.createdAt);
        return {
          record: this.getClusterVersion(existing.id)!,
          members: this.listClusterMembers(existing.id),
          created: false,
          reactivated: true,
          ...(previousActive ? { previousActive } : {})
        };
      }

      const versionNo = this.nextClusterVersionNo(current.id);
      const id = `trajectory_window_cluster_version_${stableHash({
        clusterId: current.id,
        membershipHash
      }).slice(0, 24)}`;
      const counts = evidenceEpisodeCounts(normalizedMembers, occurrenceById);
      this.db.prepare(
        `INSERT INTO trajectory_window_cluster_versions (
          id, cluster_id, version_no, membership_hash, support_hash,
          medoid_occurrence_id, support_episode_count,
          counterexample_episode_count, unknown_episode_count, metrics_json,
          status, supersedes_version_id, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL)`
      ).run(
        id,
        current.id,
        versionNo,
        membershipHash,
        supportHash,
        input.medoidOccurrenceId,
        counts.support,
        counts.counterexample,
        counts.unknown,
        toJson(input.metrics ?? {}),
        previousActive?.id ?? null,
        input.createdAt,
        input.createdAt
      );
      const insertMember = this.db.prepare(
        `INSERT INTO trajectory_window_cluster_members (
          id, cluster_version_id, occurrence_id, episode_id, user_id,
          evidence_role, reward_hash, terminal_reward, coarse_similarity,
          alignment_json, is_medoid, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const member of normalizedMembers) {
        const occurrence = occurrenceById.get(member.occurrenceId)!;
        const memberId = `trajectory_window_cluster_member_${stableHash({
          clusterVersionId: id,
          occurrenceId: occurrence.id
        }).slice(0, 24)}`;
        insertMember.run(
          memberId,
          id,
          occurrence.id,
          occurrence.episodeId,
          occurrence.userId,
          member.evidenceRole,
          member.rewardHash,
          member.terminalReward ?? null,
          member.coarseSimilarity,
          toJson(member.alignment ?? {}),
          occurrence.id === input.medoidOccurrenceId ? 1 : 0,
          input.createdAt
        );
      }
      this.updateClusterVersionHead(current.id, id, input.createdAt);
      return {
        record: this.getClusterVersion(id)!,
        members: this.listClusterMembers(id),
        created: true,
        reactivated: false,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  getClusterVersion(id: string): TrajectoryWindowClusterVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_cluster_versions WHERE id = ?`
    ).get(id) as ClusterVersionSqlRow | undefined;
    return row ? clusterVersionFromSql(row) : undefined;
  }

  listClusterVersions(clusterId: string): TrajectoryWindowClusterVersionRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_cluster_versions
       WHERE cluster_id = ? ORDER BY version_no DESC, id DESC`
    ).all(clusterId) as ClusterVersionSqlRow[]).map(clusterVersionFromSql);
  }

  listClusterMembers(clusterVersionId: string): TrajectoryWindowClusterMemberRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_window_cluster_members
       WHERE cluster_version_id = ? ORDER BY occurrence_id ASC`
    ).all(clusterVersionId) as ClusterMemberSqlRow[]).map(clusterMemberFromSql);
  }

  activateClusterVersion(input: {
    clusterId: string;
    versionId: string;
    expectedActiveVersionId: string | null;
    activatedAt: string;
  }): CommitTrajectoryWindowClusterVersionResult {
    return this.db.transaction(() => {
      const cluster = this.requireCluster(input.clusterId);
      const target = this.requireClusterVersion(input.versionId);
      if (target.clusterId !== cluster.id) {
        throw new Error(`cluster version does not belong to cluster: ${target.id}`);
      }
      if (cluster.activeVersionId === target.id) {
        return {
          record: target,
          members: this.listClusterMembers(target.id),
          created: false,
          reactivated: false
        };
      }
      assertCas(cluster.id, input.expectedActiveVersionId, cluster.activeVersionId ?? null);
      this.assertClusterVersionSourcesActive(target.id);
      const previousActive = cluster.activeVersionId
        ? this.getClusterVersion(cluster.activeVersionId)
        : undefined;
      if (previousActive) this.supersedeClusterVersion(previousActive.id, input.activatedAt);
      this.db.prepare(
        `UPDATE trajectory_window_cluster_versions
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(input.activatedAt, target.id);
      this.updateClusterVersionHead(cluster.id, target.id, input.activatedAt);
      return {
        record: this.getClusterVersion(target.id)!,
        members: this.listClusterMembers(target.id),
        created: false,
        reactivated: true,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  listAffectedClusterIdsForPath(pathId: string): string[] {
    return (this.db.prepare(
      `SELECT DISTINCT versions.cluster_id AS id
       FROM trajectory_window_cluster_members AS members
       JOIN trajectory_window_cluster_versions AS versions
         ON versions.id = members.cluster_version_id
       JOIN trajectory_window_occurrences AS occurrences
         ON occurrences.id = members.occurrence_id
       WHERE occurrences.path_id = ?
       ORDER BY versions.cluster_id ASC`
    ).all(pathId) as Array<{ id: string }>).map((row) => row.id);
  }

  retireCluster(
    clusterId: string,
    expectedActiveVersionId: string | null,
    retiredAt: string
  ): TrajectoryWindowClusterRecord {
    return this.db.transaction(() => {
      const cluster = this.requireCluster(clusterId);
      assertCas(cluster.id, expectedActiveVersionId, cluster.activeVersionId ?? null);
      if (cluster.status === "retired") return cluster;
      this.db.prepare(
        `UPDATE trajectory_window_clusters
         SET status = 'retired', updated_at = ?
         WHERE id = ?`
      ).run(retiredAt, cluster.id);
      return this.requireCluster(cluster.id);
    })();
  }

  saveSkillVersion(input: SaveTrajectorySkillVersionInput): SaveTrajectorySkillVersionResult {
    const cluster = this.requireCluster(input.clusterId);
    if (cluster.status !== "active") {
      throw new Error(`trajectory Skill cluster is retired: ${cluster.id}`);
    }
    const clusterVersion = this.requireClusterVersion(input.clusterVersionId);
    if (clusterVersion.clusterId !== cluster.id) {
      throw new Error("trajectory Skill cluster version does not belong to cluster");
    }
    if (cluster.activeVersionId !== clusterVersion.id) {
      throw new Error("trajectory Skill must be induced from the active cluster version");
    }
    if (input.skillMemoryId) {
      this.assertFormalSkillMemory(input.skillMemoryId, cluster.userId);
    }
    const contentHash = input.contentHash ?? stableHash({
      skillKey: input.skillKey,
      skillMemoryId: input.skillMemoryId ?? null,
      payload: input.payload
    });
    const existing = this.getSkillVersionByContentHash(clusterVersion.id, contentHash);

    return this.db.transaction(() => {
      const current = this.requireCluster(cluster.id);
      if (existing && current.activeSkillVersionId === existing.id) {
        return { record: existing, created: false, reactivated: false };
      }
      assertCas(
        `${cluster.id}:skill`,
        input.expectedActiveSkillVersionId,
        current.activeSkillVersionId ?? null
      );
      const previousActive = current.activeSkillVersionId
        ? this.getSkillVersion(current.activeSkillVersionId)
        : undefined;
      if (previousActive) this.supersedeSkillVersion(previousActive.id, input.createdAt);
      if (existing) {
        this.db.prepare(
          `UPDATE trajectory_skill_versions
           SET status = 'active', activated_at = ?, deactivated_at = NULL
           WHERE id = ?`
        ).run(input.createdAt, existing.id);
        this.updateSkillVersionHead(current.id, existing, input.createdAt);
        return {
          record: this.getSkillVersion(existing.id)!,
          created: false,
          reactivated: true,
          ...(previousActive ? { previousActive } : {})
        };
      }
      const versionNo = this.nextSkillVersionNo(current.id);
      const id = `trajectory_skill_version_${stableHash({
        clusterId: current.id,
        clusterVersionId: clusterVersion.id,
        contentHash
      }).slice(0, 24)}`;
      this.db.prepare(
        `INSERT INTO trajectory_skill_versions (
          id, cluster_id, cluster_version_id, version_no, skill_key,
          membership_hash, support_hash, content_hash, skill_memory_id, status,
          payload_json, supersedes_version_id, created_at, activated_at,
          deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, NULL)`
      ).run(
        id,
        current.id,
        clusterVersion.id,
        versionNo,
        input.skillKey,
        clusterVersion.membershipHash,
        clusterVersion.supportHash,
        contentHash,
        input.skillMemoryId ?? null,
        toJson(input.payload),
        previousActive?.id ?? null,
        input.createdAt,
        input.createdAt
      );
      const record = this.getSkillVersion(id)!;
      this.updateSkillVersionHead(current.id, record, input.createdAt);
      return {
        record,
        created: true,
        reactivated: false,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  getSkillVersion(id: string): TrajectorySkillVersionRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM trajectory_skill_versions WHERE id = ?`)
      .get(id) as SkillVersionSqlRow | undefined;
    return row ? skillVersionFromSql(row) : undefined;
  }

  listSkillVersions(clusterId: string): TrajectorySkillVersionRecord[] {
    return (this.db.prepare(
      `SELECT * FROM trajectory_skill_versions
       WHERE cluster_id = ? ORDER BY version_no DESC, id DESC`
    ).all(clusterId) as SkillVersionSqlRow[]).map(skillVersionFromSql);
  }

  activateSkillVersion(input: {
    clusterId: string;
    versionId: string;
    expectedActiveSkillVersionId: string | null;
    activatedAt: string;
  }): SaveTrajectorySkillVersionResult {
    return this.db.transaction(() => {
      const cluster = this.requireCluster(input.clusterId);
      if (cluster.status !== "active") {
        throw new Error(`trajectory Skill cluster is retired: ${cluster.id}`);
      }
      const target = this.requireSkillVersion(input.versionId);
      if (target.clusterId !== cluster.id) {
        throw new Error(`trajectory Skill version does not belong to cluster: ${target.id}`);
      }
      if (cluster.activeSkillVersionId === target.id) {
        return { record: target, created: false, reactivated: false };
      }
      this.assertClusterVersionSourcesActive(target.clusterVersionId);
      if (target.skillMemoryId) {
        this.assertFormalSkillMemory(target.skillMemoryId, cluster.userId);
      }
      assertCas(
        `${cluster.id}:skill`,
        input.expectedActiveSkillVersionId,
        cluster.activeSkillVersionId ?? null
      );
      const previousActive = cluster.activeSkillVersionId
        ? this.getSkillVersion(cluster.activeSkillVersionId)
        : undefined;
      if (previousActive) this.supersedeSkillVersion(previousActive.id, input.activatedAt);
      this.db.prepare(
        `UPDATE trajectory_skill_versions
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(input.activatedAt, target.id);
      const active = this.getSkillVersion(target.id)!;
      this.updateSkillVersionHead(cluster.id, active, input.activatedAt);
      return {
        record: active,
        created: false,
        reactivated: true,
        ...(previousActive ? { previousActive } : {})
      };
    })();
  }

  private getPathByCompilationKey(
    episodeId: string,
    compilationKey: string
  ): EpisodeExecutionPathRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_execution_paths
       WHERE episode_id = ? AND compilation_key = ? LIMIT 1`
    ).get(episodeId, compilationKey) as PathSqlRow | undefined;
    return row ? pathFromSql(row) : undefined;
  }

  private requireEpisode(id: string): EpisodeSourceRow {
    const row = this.db.prepare(
      `SELECT episodes.id, episodes.user_id, episodes.raw_turn_ids_json,
              episodes.r_task, sessions.source AS session_source
       FROM episodes
       JOIN sessions ON sessions.id = episodes.session_id
       WHERE episodes.id = ?`
    ).get(id) as EpisodeSourceRow | undefined;
    if (!row) throw new Error(`execution path source episode not found: ${id}`);
    return row;
  }

  private validateRawTurnSources(episodeId: string, userId: string, rawTurnIds: string[]): void {
    for (const rawTurnId of rawTurnIds) {
      const row = this.db.prepare(
        `SELECT episode_id, user_id FROM raw_turns WHERE id = ?`
      ).get(rawTurnId) as { episode_id: string; user_id: string } | undefined;
      if (!row || row.episode_id !== episodeId || row.user_id !== userId) {
        throw new Error(`execution path RawTurn is outside episode scope: ${rawTurnId}`);
      }
    }
  }

  private requirePath(id: string): EpisodeExecutionPathRecord {
    const path = this.getPath(id);
    if (!path) throw new Error(`execution path not found: ${id}`);
    return path;
  }

  private getStepEmbeddingByIdentity(
    pathId: string,
    stepId: string,
    representationVersion: string,
    embeddingSignature: string
  ): ProceduralStepEmbeddingRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_step_embeddings
       WHERE path_id = ? AND step_id = ? AND representation_version = ?
         AND embedding_signature = ? LIMIT 1`
    ).get(pathId, stepId, representationVersion, embeddingSignature) as
      StepEmbeddingSqlRow | undefined;
    return row ? stepEmbeddingFromSql(row) : undefined;
  }

  private getWindowByIdentity(input: InsertTrajectoryWindowInput):
    TrajectoryWindowOccurrenceRecord | undefined {
    const occurrence = input.occurrence;
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_occurrences
       WHERE path_id = ? AND window_config_hash = ? AND scale = ?
         AND start_step_index = ? AND end_step_index = ? LIMIT 1`
    ).get(
      occurrence.pathId,
      input.windowConfigHash,
      occurrence.scale,
      occurrence.startStepIndex,
      occurrence.endStepIndex
    ) as WindowSqlRow | undefined;
    return row ? windowFromSql(row) : undefined;
  }

  private requireWindow(id: string): TrajectoryWindowOccurrenceRecord {
    const occurrence = this.getWindow(id);
    if (!occurrence) throw new Error(`trajectory window occurrence not found: ${id}`);
    return occurrence;
  }

  private requireActiveWindow(id: string): TrajectoryWindowOccurrenceRecord {
    const row = this.db.prepare(
      `SELECT occurrences.*
       FROM trajectory_window_occurrences AS occurrences
       JOIN episode_execution_paths AS paths ON paths.id = occurrences.path_id
       WHERE occurrences.id = ? AND paths.status = 'active'`
    ).get(id) as WindowSqlRow | undefined;
    if (!row) throw new Error(`active trajectory window occurrence not found: ${id}`);
    return windowFromSql(row);
  }

  private requireFamily(id: string): TrajectoryWindowFamilyRecord {
    const family = this.getFamilyHead(id);
    if (!family) throw new Error(`trajectory window Family not found: ${id}`);
    return family;
  }

  private getFamilyRevisionByMembershipHash(
    familyId: string,
    membershipHash: string
  ): TrajectoryWindowFamilyRevisionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_family_revisions
       WHERE family_id = ? AND membership_hash = ? LIMIT 1`
    ).get(familyId, membershipHash) as FamilyRevisionSqlRow | undefined;
    return row ? familyRevisionFromSql(row) : undefined;
  }

  private requireFamilyRevision(id: string): TrajectoryWindowFamilyRevisionRecord {
    const revision = this.getFamilyRevision(id);
    if (!revision) throw new Error(`trajectory Family revision not found: ${id}`);
    return revision;
  }

  private nextFamilyRevisionNo(familyId: string): number {
    const row = this.db.prepare(
      `SELECT COALESCE(MAX(revision_no), 0) + 1 AS value
       FROM trajectory_window_family_revisions WHERE family_id = ?`
    ).get(familyId) as { value: number };
    return Number(row.value);
  }

  private supersedeFamilyRevision(id: string, at: string): void {
    this.db.prepare(
      `UPDATE trajectory_window_family_revisions
       SET status = 'superseded', deactivated_at = ?
       WHERE id = ? AND status = 'active'`
    ).run(at, id);
  }

  private updateFamilyRevisionHead(familyId: string, revisionId: string, at: string): void {
    this.db.prepare(
      `UPDATE trajectory_window_families
       SET active_revision_id = ?, status = 'active', updated_at = ?
       WHERE id = ?`
    ).run(revisionId, at, familyId);
  }

  private assertFamilyRevisionSourcesActive(revisionId: string): void {
    const stale = this.db.prepare(
      `SELECT members.occurrence_id
       FROM trajectory_window_family_members AS members
       JOIN trajectory_window_occurrences AS occurrences
         ON occurrences.id = members.occurrence_id
       JOIN episode_execution_paths AS paths ON paths.id = occurrences.path_id
       WHERE members.family_revision_id = ? AND paths.status != 'active'
       LIMIT 1`
    ).get(revisionId) as { occurrence_id: string } | undefined;
    if (stale) {
      throw new Error(
        `cannot activate Family revision with superseded path evidence: ${stale.occurrence_id}`
      );
    }
  }

  private findClusterCanonicalKey(input: {
    userId: string;
    scale: number;
    algorithmVersion: string;
    configHash: string;
    embeddingSignature: string;
    evidenceSignature: string;
  }): TrajectoryWindowClusterCanonicalKeyRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_cluster_canonical_keys
       WHERE user_id = ? AND scale = ? AND algorithm_version = ?
         AND config_hash = ? AND embedding_signature = ? AND evidence_signature = ?
       LIMIT 1`
    ).get(
      input.userId,
      input.scale,
      input.algorithmVersion,
      input.configHash,
      input.embeddingSignature,
      input.evidenceSignature
    ) as ClusterCanonicalKeySqlRow | undefined;
    return row ? clusterCanonicalKeyFromSql(row) : undefined;
  }

  private requireClusterCanonicalKey(id: string): TrajectoryWindowClusterCanonicalKeyRecord {
    const canonicalKey = this.getClusterCanonicalKey(id);
    if (!canonicalKey) throw new Error(`trajectory cluster canonical key not found: ${id}`);
    return canonicalKey;
  }

  private requireCluster(id: string): TrajectoryWindowClusterRecord {
    const cluster = this.getClusterHead(id);
    if (!cluster) throw new Error(`trajectory window cluster not found: ${id}`);
    return cluster;
  }

  private getClusterVersionByMembershipHash(
    clusterId: string,
    membershipHash: string
  ): TrajectoryWindowClusterVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_window_cluster_versions
       WHERE cluster_id = ? AND membership_hash = ? LIMIT 1`
    ).get(clusterId, membershipHash) as ClusterVersionSqlRow | undefined;
    return row ? clusterVersionFromSql(row) : undefined;
  }

  private requireClusterVersion(id: string): TrajectoryWindowClusterVersionRecord {
    const version = this.getClusterVersion(id);
    if (!version) throw new Error(`trajectory cluster version not found: ${id}`);
    return version;
  }

  private nextClusterVersionNo(clusterId: string): number {
    const row = this.db.prepare(
      `SELECT COALESCE(MAX(version_no), 0) + 1 AS value
       FROM trajectory_window_cluster_versions WHERE cluster_id = ?`
    ).get(clusterId) as { value: number };
    return Number(row.value);
  }

  private supersedeClusterVersion(id: string, at: string): void {
    this.db.prepare(
      `UPDATE trajectory_window_cluster_versions
       SET status = 'superseded', deactivated_at = ?
       WHERE id = ? AND status = 'active'`
    ).run(at, id);
  }

  private updateClusterVersionHead(clusterId: string, versionId: string, at: string): void {
    this.db.prepare(
      `UPDATE trajectory_window_clusters
       SET active_version_id = ?, status = 'active', updated_at = ?
       WHERE id = ?`
    ).run(versionId, at, clusterId);
  }

  private assertClusterVersionSourcesActive(versionId: string): void {
    const stale = this.db.prepare(
      `SELECT members.occurrence_id
       FROM trajectory_window_cluster_members AS members
       JOIN trajectory_window_occurrences AS occurrences
         ON occurrences.id = members.occurrence_id
       JOIN episode_execution_paths AS paths ON paths.id = occurrences.path_id
       WHERE members.cluster_version_id = ? AND paths.status != 'active'
       LIMIT 1`
    ).get(versionId) as { occurrence_id: string } | undefined;
    if (stale) {
      throw new Error(
        `cannot activate cluster version with superseded path evidence: ${stale.occurrence_id}`
      );
    }
  }

  private getSkillVersionByContentHash(
    clusterVersionId: string,
    contentHash: string
  ): TrajectorySkillVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM trajectory_skill_versions
       WHERE cluster_version_id = ? AND content_hash = ? LIMIT 1`
    ).get(clusterVersionId, contentHash) as SkillVersionSqlRow | undefined;
    return row ? skillVersionFromSql(row) : undefined;
  }

  private requireSkillVersion(id: string): TrajectorySkillVersionRecord {
    const version = this.getSkillVersion(id);
    if (!version) throw new Error(`trajectory Skill version not found: ${id}`);
    return version;
  }

  private nextSkillVersionNo(clusterId: string): number {
    const row = this.db.prepare(
      `SELECT COALESCE(MAX(version_no), 0) + 1 AS value
       FROM trajectory_skill_versions WHERE cluster_id = ?`
    ).get(clusterId) as { value: number };
    return Number(row.value);
  }

  private supersedeSkillVersion(id: string, at: string): void {
    this.db.prepare(
      `UPDATE trajectory_skill_versions
       SET status = 'superseded', deactivated_at = ?
       WHERE id = ? AND status = 'active'`
    ).run(at, id);
  }

  private updateSkillVersionHead(
    clusterId: string,
    version: TrajectorySkillVersionRecord,
    at: string
  ): void {
    this.db.prepare(
      `UPDATE trajectory_window_clusters
       SET active_skill_version_id = ?, active_skill_memory_id = ?, updated_at = ?
       WHERE id = ?`
    ).run(version.id, version.skillMemoryId ?? null, at, clusterId);
  }

  private assertFormalSkillMemory(id: string, expectedUserId: string): void {
    const row = this.db.prepare(
      `SELECT user_id, memory_layer, memory_type, properties_json
       FROM memories WHERE id = ? AND deleted_at IS NULL`
    ).get(id) as {
      user_id: string;
      memory_layer: string;
      memory_type: string;
      properties_json: string;
    } | undefined;
    if (!row) throw new Error(`trajectory Skill memory not found: ${id}`);
    if (row.user_id !== expectedUserId) {
      throw new Error(`trajectory Skill memory is outside cluster user scope: ${id}`);
    }
    const properties = parseJson<Record<string, unknown>>(row.properties_json, {});
    const internal = isRecord(properties.internal_info) ? properties.internal_info : {};
    if (row.memory_layer !== "Skill" || row.memory_type !== "SkillMemory" ||
        !isRecord(internal.skill)) {
      throw new Error(`trajectory Skill memory is not an upstream Skill schema: ${id}`);
    }
  }
}

function validatedVector(vector: readonly number[], label: string): number[] {
  if (vector.length === 0 || vector.some((value) => !Number.isFinite(value))) {
    throw new Error(`${label} must contain finite values`);
  }
  return [...vector];
}

function validateMemberInput(member: TrajectoryWindowClusterMemberInput): void {
  if (!member.rewardHash) throw new Error("trajectory cluster member requires rewardHash");
  if (!Number.isFinite(member.coarseSimilarity) ||
      member.coarseSimilarity < -1 || member.coarseSimilarity > 1) {
    throw new Error("trajectory cluster member coarseSimilarity must be in [-1, 1]");
  }
}

function validateFamilyMemberInput(member: TrajectoryWindowFamilyMemberInput): void {
  if (!Number.isFinite(member.coarseSimilarity) ||
      member.coarseSimilarity < -1 || member.coarseSimilarity > 1) {
    throw new Error("trajectory Family member coarseSimilarity must be in [-1, 1]");
  }
}

function evidenceEpisodeCounts(
  members: ClassifiedTrajectoryWindowClusterMember[],
  occurrenceById: Map<string, TrajectoryWindowOccurrenceRecord>
): Record<ProceduralEvidenceRole, number> {
  const byRole: Record<ProceduralEvidenceRole, Set<string>> = {
    support: new Set(),
    counterexample: new Set(),
    unknown: new Set()
  };
  for (const member of members) {
    byRole[member.evidenceRole].add(occurrenceById.get(member.occurrenceId)!.episodeId);
  }
  return {
    support: byRole.support.size,
    counterexample: byRole.counterexample.size,
    unknown: byRole.unknown.size
  };
}

function assertCas(targetId: string, expected: string | null, actual: string | null): void {
  if (expected !== actual) throw new ProceduralTrajectoryCasError(targetId, expected, actual);
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.filter((value) => value.length > 0))];
}

function stringArray(value: string): string[] {
  return parseJson<unknown[]>(value, []).filter((item): item is string =>
    typeof item === "string" && item.length > 0);
}

function numberArray(value: string): number[] {
  return parseJson<unknown[]>(value, []).filter((item): item is number =>
    typeof item === "number" && Number.isFinite(item));
}

function pathFromSql(row: PathSqlRow): EpisodeExecutionPathRecord {
  const path = parseJson<EpisodeExecutionPathLiteV1 | null>(row.payload_json, null);
  if (!path || path.schemaVersion !== EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION) {
    throw new Error(`unsupported or corrupt execution path payload: ${row.id}`);
  }
  return {
    id: row.id,
    episodeId: row.episode_id,
    userId: row.user_id,
    schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
    compilationKey: row.compilation_key,
    sourceSnapshotHash: row.source_snapshot_hash,
    pathHash: row.path_hash,
    compilerVersion: row.compiler_version,
    ...(row.model_signature ? { modelSignature: row.model_signature } : {}),
    sourceRawTurnIds: stringArray(row.source_raw_turn_ids_json),
    sourceAgentIds: stringArray(row.source_agent_ids_json),
    stepCount: row.step_count,
    status: row.status,
    path,
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function stepEmbeddingFromSql(row: StepEmbeddingSqlRow): ProceduralStepEmbeddingRecord {
  return {
    id: row.id,
    pathId: row.path_id,
    stepId: row.step_id,
    stepIndex: row.step_index,
    representationVersion: row.representation_version,
    embeddingSignature: row.embedding_signature,
    semanticHash: row.semantic_hash,
    embeddingDim: row.embedding_dim,
    vector: numberArray(row.vector_json),
    createdAt: row.created_at
  };
}

function windowFromSql(row: WindowSqlRow): TrajectoryWindowOccurrenceRecord {
  if (row.schema_version !== TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION) {
    throw new Error(`unsupported trajectory window occurrence: ${row.id}`);
  }
  return {
    id: row.id,
    pathId: row.path_id,
    episodeId: row.episode_id,
    userId: row.user_id,
    ...(row.terminal_reward === null ? {} : { terminalReward: row.terminal_reward }),
    evidenceRole: row.evidence_role,
    schemaVersion: TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION,
    windowConfigHash: row.window_config_hash,
    scale: row.scale,
    stride: row.stride,
    startStepIndex: row.start_step_index,
    endStepIndex: row.end_step_index,
    stepIds: stringArray(row.step_ids_json),
    rawTurnIds: stringArray(row.raw_turn_ids_json),
    semanticText: row.semantic_text,
    semanticHash: row.semantic_hash,
    coarseRepresentationVersion: row.coarse_representation_version,
    embeddingSignature: row.embedding_signature,
    embeddingDim: row.embedding_dim,
    coarseVector: numberArray(row.coarse_vector_json),
    createdAt: row.created_at
  };
}

function familyFromSql(row: FamilySqlRow): TrajectoryWindowFamilyRecord {
  return {
    id: row.id,
    userId: row.user_id,
    scale: row.scale,
    algorithmVersion: row.algorithm_version,
    configHash: row.config_hash,
    embeddingSignature: row.embedding_signature,
    status: row.status,
    ...(row.active_revision_id ? { activeRevisionId: row.active_revision_id } : {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function familyRevisionFromSql(
  row: FamilyRevisionSqlRow
): TrajectoryWindowFamilyRevisionRecord {
  return {
    id: row.id,
    familyId: row.family_id,
    revisionNo: row.revision_no,
    membershipHash: row.membership_hash,
    evidenceHash: row.evidence_hash,
    medoidOccurrenceId: row.medoid_occurrence_id,
    metrics: parseJson(row.metrics_json, {}),
    status: row.status,
    ...(row.supersedes_revision_id ? { supersedesRevisionId: row.supersedes_revision_id } : {}),
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function familyMemberFromSql(row: FamilyMemberSqlRow): TrajectoryWindowFamilyMemberRecord {
  return {
    id: row.id,
    familyRevisionId: row.family_revision_id,
    occurrenceId: row.occurrence_id,
    episodeId: row.episode_id,
    userId: row.user_id,
    coarseSimilarity: row.coarse_similarity,
    isMedoid: row.is_medoid !== 0,
    createdAt: row.created_at
  };
}

function clusterFromSql(row: ClusterSqlRow): TrajectoryWindowClusterRecord {
  return {
    id: row.id,
    userId: row.user_id,
    scale: row.scale,
    algorithmVersion: row.algorithm_version,
    configHash: row.config_hash,
    status: row.status,
    ...(row.active_version_id ? { activeVersionId: row.active_version_id } : {}),
    ...(row.active_skill_version_id
      ? { activeSkillVersionId: row.active_skill_version_id }
      : {}),
    ...(row.active_skill_memory_id ? { activeSkillMemoryId: row.active_skill_memory_id } : {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function clusterVersionFromSql(row: ClusterVersionSqlRow): TrajectoryWindowClusterVersionRecord {
  return {
    id: row.id,
    clusterId: row.cluster_id,
    versionNo: row.version_no,
    membershipHash: row.membership_hash,
    supportHash: row.support_hash,
    medoidOccurrenceId: row.medoid_occurrence_id,
    supportEpisodeCount: row.support_episode_count,
    counterexampleEpisodeCount: row.counterexample_episode_count,
    unknownEpisodeCount: row.unknown_episode_count,
    metrics: parseJson(row.metrics_json, {}),
    status: row.status,
    ...(row.supersedes_version_id ? { supersedesVersionId: row.supersedes_version_id } : {}),
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function clusterMemberFromSql(row: ClusterMemberSqlRow): TrajectoryWindowClusterMemberRecord {
  return {
    id: row.id,
    clusterVersionId: row.cluster_version_id,
    occurrenceId: row.occurrence_id,
    episodeId: row.episode_id,
    userId: row.user_id,
    evidenceRole: row.evidence_role,
    rewardHash: row.reward_hash,
    ...(row.terminal_reward === null ? {} : { terminalReward: row.terminal_reward }),
    coarseSimilarity: row.coarse_similarity,
    alignment: parseJson(row.alignment_json, {}),
    isMedoid: row.is_medoid !== 0,
    createdAt: row.created_at
  };
}

function clusterCanonicalKeyFromSql(
  row: ClusterCanonicalKeySqlRow
): TrajectoryWindowClusterCanonicalKeyRecord {
  return {
    id: row.id,
    userId: row.user_id,
    scale: row.scale,
    algorithmVersion: row.algorithm_version,
    configHash: row.config_hash,
    embeddingSignature: row.embedding_signature,
    evidenceSignature: row.evidence_signature,
    clusterId: row.cluster_id,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function familyClusterLinkFromSql(
  row: FamilyClusterLinkSqlRow
): TrajectoryWindowFamilyClusterLinkRecord {
  return {
    id: row.id,
    familyRevisionId: row.family_revision_id,
    canonicalKeyId: row.canonical_key_id,
    clusterVersionId: row.cluster_version_id,
    createdAt: row.created_at
  };
}

function skillVersionFromSql(row: SkillVersionSqlRow): TrajectorySkillVersionRecord {
  return {
    id: row.id,
    clusterId: row.cluster_id,
    clusterVersionId: row.cluster_version_id,
    versionNo: row.version_no,
    skillKey: row.skill_key,
    membershipHash: row.membership_hash,
    supportHash: row.support_hash,
    contentHash: row.content_hash,
    ...(row.skill_memory_id ? { skillMemoryId: row.skill_memory_id } : {}),
    status: row.status,
    payload: parseJson(row.payload_json, {}),
    ...(row.supersedes_version_id ? { supersedesVersionId: row.supersedes_version_id } : {}),
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}
