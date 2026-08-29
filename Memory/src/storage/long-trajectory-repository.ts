import type Database from "better-sqlite3";

import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";

export const LONG_TRAJECTORY_EPISODE_REPRESENTATION_VERSION =
  "long-trajectory-episode-representation.v1" as const;

export interface LongTrajectoryEpisodeRepresentationRecord {
  id: string;
  pathId: string;
  episodeId: string;
  userId: string;
  representationVersion: string;
  embeddingSignature: string;
  goalText: string;
  terminalResultText: string;
  goalHash: string;
  goalVector: number[];
  trajectoryText: string;
  trajectoryHash: string;
  trajectoryVector: number[];
  createdAt: string;
  updatedAt: string;
}

export interface LongTrajectoryCandidateRecord {
  id: string;
  userId: string;
  algorithmVersion: string;
  configHash: string;
  structureKey: string;
  /** Legacy v9 identity retained for bundle compatibility and audit only. */
  evidenceSignature: string;
  status: "active" | "retired";
  activeVersionId?: string;
  activeSkillVersionId?: string;
  activeSkillMemoryId?: string;
  createdAt: string;
  updatedAt: string;
}

export interface LongTrajectoryCandidateVersionRecord {
  id: string;
  candidateId: string;
  versionNo: number;
  structureHash: string;
  evidenceHash: string;
  supportHash: string;
  referenceEpisodeId: string;
  sourcePathIds: string[];
  supportEpisodeIds: string[];
  payload: Record<string, unknown>;
  status: "active" | "superseded";
  supersedesVersionId?: string;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface LongTrajectorySkillVersionRecord {
  id: string;
  candidateId: string;
  candidateVersionId: string;
  versionNo: number;
  skillKey: string;
  contentHash: string;
  skillMemoryId?: string;
  payload: Record<string, unknown>;
  status: "active" | "superseded";
  supersedesVersionId?: string;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

interface RepresentationSqlRow {
  id: string;
  path_id: string;
  episode_id: string;
  user_id: string;
  representation_version: string;
  embedding_signature: string;
  goal_text: string;
  terminal_result_text: string;
  goal_hash: string;
  goal_vector_json: string;
  trajectory_text: string;
  trajectory_hash: string;
  trajectory_vector_json: string;
  created_at: string;
  updated_at: string;
}

interface CandidateSqlRow {
  id: string;
  user_id: string;
  algorithm_version: string;
  config_hash: string;
  structure_key: string;
  evidence_signature: string;
  status: "active" | "retired";
  active_version_id: string | null;
  active_skill_version_id: string | null;
  active_skill_memory_id: string | null;
  created_at: string;
  updated_at: string;
}

interface CandidateVersionSqlRow {
  id: string;
  candidate_id: string;
  version_no: number;
  structure_hash: string;
  evidence_hash: string;
  support_hash: string;
  reference_episode_id: string;
  source_path_ids_json: string;
  support_episode_ids_json: string;
  payload_json: string;
  status: "active" | "superseded";
  supersedes_version_id: string | null;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface SkillVersionSqlRow {
  id: string;
  candidate_id: string;
  candidate_version_id: string;
  version_no: number;
  skill_key: string;
  content_hash: string;
  skill_memory_id: string | null;
  payload_json: string;
  status: "active" | "superseded";
  supersedes_version_id: string | null;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

export class LongTrajectoryRepository {
  constructor(private readonly db: Database.Database) {}

  getEpisodeRepresentation(input: {
    pathId: string;
    representationVersion: string;
    embeddingSignature: string;
  }): LongTrajectoryEpisodeRepresentationRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM long_trajectory_episode_representations
       WHERE path_id = ? AND representation_version = ? AND embedding_signature = ?
       LIMIT 1`
    ).get(input.pathId, input.representationVersion, input.embeddingSignature) as
      RepresentationSqlRow | undefined;
    return row ? representationFromSql(row) : undefined;
  }

  upsertEpisodeRepresentation(input: Omit<
    LongTrajectoryEpisodeRepresentationRecord,
    "id" | "createdAt" | "updatedAt"
  > & { createdAt: string }): LongTrajectoryEpisodeRepresentationRecord {
    if (input.goalVector.length === 0 || input.trajectoryVector.length === 0 ||
        input.goalVector.length !== input.trajectoryVector.length) {
      throw new Error("long trajectory Episode vectors must be non-empty and dimensionally equal");
    }
    const id = `long_trajectory_episode_representation_${stableHash({
      pathId: input.pathId,
      representationVersion: input.representationVersion,
      embeddingSignature: input.embeddingSignature
    }).slice(0, 24)}`;
    this.db.prepare(
      `INSERT INTO long_trajectory_episode_representations (
        id, path_id, episode_id, user_id, representation_version,
        embedding_signature, goal_text, terminal_result_text, goal_hash,
        goal_vector_json, trajectory_text, trajectory_hash,
        trajectory_vector_json, embedding_dim, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(path_id, representation_version, embedding_signature) DO UPDATE SET
        goal_text = excluded.goal_text,
        terminal_result_text = excluded.terminal_result_text,
        goal_hash = excluded.goal_hash,
        goal_vector_json = excluded.goal_vector_json,
        trajectory_text = excluded.trajectory_text,
        trajectory_hash = excluded.trajectory_hash,
        trajectory_vector_json = excluded.trajectory_vector_json,
        embedding_dim = excluded.embedding_dim,
        updated_at = excluded.updated_at`
    ).run(
      id,
      input.pathId,
      input.episodeId,
      input.userId,
      input.representationVersion,
      input.embeddingSignature,
      input.goalText,
      input.terminalResultText,
      input.goalHash,
      toJson(input.goalVector),
      input.trajectoryText,
      input.trajectoryHash,
      toJson(input.trajectoryVector),
      input.goalVector.length,
      input.createdAt,
      input.createdAt
    );
    return this.getEpisodeRepresentation(input)!;
  }

  listActiveEpisodeRepresentations(input: {
    userId: string;
    representationVersion: string;
    embeddingSignature: string;
  }): LongTrajectoryEpisodeRepresentationRecord[] {
    return (this.db.prepare(
      `SELECT representations.*
       FROM long_trajectory_episode_representations AS representations
       JOIN episode_execution_paths AS paths
         ON paths.id = representations.path_id AND paths.status = 'active'
       JOIN episodes ON episodes.id = representations.episode_id
         AND episodes.status = 'closed'
       WHERE representations.user_id = ? AND representations.representation_version = ?
         AND representations.embedding_signature = ?
       ORDER BY representations.episode_id ASC`
    ).all(input.userId, input.representationVersion, input.embeddingSignature) as
      RepresentationSqlRow[]).map(representationFromSql);
  }

  resolveCandidate(input: {
    userId: string;
    algorithmVersion: string;
    configHash: string;
    structureKey: string;
    createdAt: string;
  }): { record: LongTrajectoryCandidateRecord; created: boolean } {
    if (!input.structureKey.trim()) {
      throw new Error("long trajectory candidate requires structureKey");
    }
    return this.db.transaction(() => {
      const existing = this.findCandidate(input);
      if (existing) {
        if (existing.status === "retired") {
          this.db.prepare(
            `UPDATE long_trajectory_candidates SET status = 'active', updated_at = ? WHERE id = ?`
          ).run(input.createdAt, existing.id);
        }
        return { record: this.getCandidate(existing.id)!, created: false };
      }
      const id = `long_trajectory_candidate_${stableHash({
        userId: input.userId,
        algorithmVersion: input.algorithmVersion,
        configHash: input.configHash,
        structureKey: input.structureKey
      }).slice(0, 24)}`;
      this.db.prepare(
        `INSERT INTO long_trajectory_candidates (
          id, user_id, algorithm_version, config_hash, structure_key, evidence_signature,
          status, active_version_id, active_skill_version_id,
          active_skill_memory_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'active', NULL, NULL, NULL, ?, ?)`
      ).run(
        id,
        input.userId,
        input.algorithmVersion,
        input.configHash,
        input.structureKey,
        input.structureKey,
        input.createdAt,
        input.createdAt
      );
      return { record: this.getCandidate(id)!, created: true };
    })();
  }

  getCandidate(id: string): LongTrajectoryCandidateRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM long_trajectory_candidates WHERE id = ?`)
      .get(id) as CandidateSqlRow | undefined;
    return row ? candidateFromSql(row) : undefined;
  }

  getCandidateVersion(id: string): LongTrajectoryCandidateVersionRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM long_trajectory_candidate_versions WHERE id = ?`)
      .get(id) as CandidateVersionSqlRow | undefined;
    return row ? candidateVersionFromSql(row) : undefined;
  }

  /**
   * Direct Candidate recall for V3. The mining layer performs the semantic
   * Span-sequence projection; this repository method only applies stable user,
   * algorithm, config, lifecycle, and active-version boundaries.
   */
  listActiveCandidates(input: {
    userId: string;
    algorithmVersion: string;
    configHash: string;
  }): LongTrajectoryCandidateRecord[] {
    return (this.db.prepare(
      `SELECT candidates.*
       FROM long_trajectory_candidates AS candidates
       JOIN long_trajectory_candidate_versions AS versions
         ON versions.id = candidates.active_version_id
        AND versions.status = 'active'
       WHERE candidates.user_id = ?
         AND candidates.algorithm_version = ?
         AND candidates.config_hash = ?
         AND candidates.status = 'active'
       ORDER BY (candidates.active_skill_memory_id IS NOT NULL) DESC,
                candidates.updated_at DESC,
                candidates.id ASC`
    ).all(
      input.userId,
      input.algorithmVersion,
      input.configHash
    ) as CandidateSqlRow[]).map(candidateFromSql);
  }

  /**
   * Reverse lookup used by online V3 evolution: recalled Episodes lead to the
   * active Candidates they already support. Candidate semantics are compared
   * by the mining layer afterwards; Episode membership is only a shortlist.
   */
  listActiveCandidatesLinkedToEpisodes(input: {
    userId: string;
    algorithmVersion: string;
    configHash: string;
    episodeIds: readonly string[];
  }): LongTrajectoryCandidateRecord[] {
    const episodeIds = uniqueSorted(input.episodeIds);
    if (episodeIds.length === 0) return [];
    const placeholders = episodeIds.map(() => "?").join(", ");
    return (this.db.prepare(
      `SELECT DISTINCT candidates.*
       FROM long_trajectory_candidates AS candidates
       JOIN long_trajectory_candidate_versions AS versions
         ON versions.id = candidates.active_version_id
        AND versions.status = 'active'
       JOIN json_each(versions.support_episode_ids_json) AS support_episode
       WHERE candidates.user_id = ?
         AND candidates.algorithm_version = ?
         AND candidates.config_hash = ?
         AND candidates.status = 'active'
         AND support_episode.value IN (${placeholders})
       ORDER BY (candidates.active_skill_memory_id IS NOT NULL) DESC,
                candidates.updated_at DESC,
                candidates.id ASC`
    ).all(
      input.userId,
      input.algorithmVersion,
      input.configHash,
      ...episodeIds
    ) as CandidateSqlRow[]).map(candidateFromSql);
  }

  saveCandidateVersion(input: {
    candidateId: string;
    expectedActiveVersionId: string | null;
    structureHash: string;
    evidenceHash: string;
    supportHash: string;
    referenceEpisodeId: string;
    sourcePathIds: string[];
    supportEpisodeIds: string[];
    payload: Record<string, unknown>;
    createdAt: string;
  }): { record: LongTrajectoryCandidateVersionRecord; created: boolean } {
    return this.db.transaction(() => {
      const candidate = this.requireCandidate(input.candidateId);
      assertExpectedVersion(
        candidate.id,
        input.expectedActiveVersionId,
        candidate.activeVersionId ?? null
      );
      const existing = this.findCandidateVersion(
        candidate.id,
        input.structureHash,
        input.evidenceHash,
        input.supportHash
      );
      if (existing) {
        if (candidate.activeVersionId !== existing.id) {
          this.supersedeActiveCandidateVersion(candidate, input.createdAt);
          this.db.prepare(
            `UPDATE long_trajectory_candidate_versions
             SET status = 'active', activated_at = ?, deactivated_at = NULL WHERE id = ?`
          ).run(input.createdAt, existing.id);
          this.updateCandidateVersionHead(candidate.id, existing.id, input.createdAt);
        }
        return { record: this.getCandidateVersion(existing.id)!, created: false };
      }
      this.supersedeActiveCandidateVersion(candidate, input.createdAt);
      const versionNo = Number((this.db.prepare(
        `SELECT COALESCE(MAX(version_no), 0) + 1 AS value
         FROM long_trajectory_candidate_versions WHERE candidate_id = ?`
      ).get(candidate.id) as { value: number }).value);
      const id = `long_trajectory_candidate_version_${stableHash({
        candidateId: candidate.id,
        structureHash: input.structureHash,
        evidenceHash: input.evidenceHash,
        supportHash: input.supportHash
      }).slice(0, 24)}`;
      this.db.prepare(
        `INSERT INTO long_trajectory_candidate_versions (
          id, candidate_id, version_no, structure_hash, evidence_hash, support_hash,
          reference_episode_id, source_path_ids_json, support_episode_ids_json,
          payload_json, status, supersedes_version_id, created_at,
          activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL)`
      ).run(
        id,
        candidate.id,
        versionNo,
        input.structureHash,
        input.evidenceHash,
        input.supportHash,
        input.referenceEpisodeId,
        toJson(uniqueSorted(input.sourcePathIds)),
        toJson(uniqueSorted(input.supportEpisodeIds)),
        toJson(input.payload),
        candidate.activeVersionId ?? null,
        input.createdAt,
        input.createdAt
      );
      this.updateCandidateVersionHead(candidate.id, id, input.createdAt);
      return { record: this.getCandidateVersion(id)!, created: true };
    })();
  }

  listAffectedCandidateIdsForPath(pathId: string): string[] {
    return (this.db.prepare(
      `SELECT DISTINCT versions.candidate_id AS id
       FROM long_trajectory_candidate_versions AS versions
       JOIN long_trajectory_candidates AS candidates
         ON candidates.id = versions.candidate_id
        AND candidates.status = 'active'
        AND candidates.active_version_id = versions.id,
            json_each(versions.source_path_ids_json) AS source_path
       WHERE source_path.value = ?
       ORDER BY versions.candidate_id ASC`
    ).all(pathId) as Array<{ id: string }>).map((row) => row.id);
  }

  retireCandidate(input: {
    candidateId: string;
    expectedActiveVersionId: string | null;
    retiredAt: string;
  }): LongTrajectoryCandidateRecord {
    return this.db.transaction(() => {
      const candidate = this.requireCandidate(input.candidateId);
      assertExpectedVersion(
        candidate.id,
        input.expectedActiveVersionId,
        candidate.activeVersionId ?? null
      );
      this.db.prepare(
        `UPDATE long_trajectory_candidates SET status = 'retired', updated_at = ? WHERE id = ?`
      ).run(input.retiredAt, candidate.id);
      return this.requireCandidate(candidate.id);
    })();
  }

  saveSkillVersion(input: {
    candidateId: string;
    candidateVersionId: string;
    expectedActiveSkillVersionId: string | null;
    skillKey: string;
    skillMemoryId?: string;
    contentHash: string;
    payload: Record<string, unknown>;
    createdAt: string;
  }): { record: LongTrajectorySkillVersionRecord; created: boolean } {
    return this.db.transaction(() => {
      const candidate = this.requireCandidate(input.candidateId);
      if (candidate.status !== "active" || candidate.activeVersionId !== input.candidateVersionId) {
        throw new Error("long trajectory Skill must use the active candidate version");
      }
      assertExpectedVersion(
        `${candidate.id}:skill`,
        input.expectedActiveSkillVersionId,
        candidate.activeSkillVersionId ?? null
      );
      if (input.skillMemoryId) this.assertFormalSkillMemory(input.skillMemoryId, candidate.userId);
      const existing = this.findSkillVersion(input.candidateVersionId, input.contentHash);
      if (existing) {
        if (candidate.activeSkillVersionId !== existing.id) {
          this.supersedeActiveSkillVersion(candidate, input.createdAt);
          this.db.prepare(
            `UPDATE long_trajectory_skill_versions
             SET status = 'active', activated_at = ?, deactivated_at = NULL WHERE id = ?`
          ).run(input.createdAt, existing.id);
          this.updateSkillHead(candidate.id, existing, input.createdAt);
        }
        return { record: this.getSkillVersion(existing.id)!, created: false };
      }
      this.supersedeActiveSkillVersion(candidate, input.createdAt);
      const versionNo = Number((this.db.prepare(
        `SELECT COALESCE(MAX(version_no), 0) + 1 AS value
         FROM long_trajectory_skill_versions WHERE candidate_id = ?`
      ).get(candidate.id) as { value: number }).value);
      const id = `long_trajectory_skill_version_${stableHash({
        candidateId: candidate.id,
        candidateVersionId: input.candidateVersionId,
        contentHash: input.contentHash
      }).slice(0, 24)}`;
      this.db.prepare(
        `INSERT INTO long_trajectory_skill_versions (
          id, candidate_id, candidate_version_id, version_no, skill_key,
          content_hash, skill_memory_id, payload_json, status,
          supersedes_version_id, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL)`
      ).run(
        id,
        candidate.id,
        input.candidateVersionId,
        versionNo,
        input.skillKey,
        input.contentHash,
        input.skillMemoryId ?? null,
        toJson(input.payload),
        candidate.activeSkillVersionId ?? null,
        input.createdAt,
        input.createdAt
      );
      const record = this.getSkillVersion(id)!;
      this.updateSkillHead(candidate.id, record, input.createdAt);
      return { record, created: true };
    })();
  }

  /**
   * Records a failed replacement attempt without changing the last-known-good
   * public Skill or the Candidate's active Skill head.
   */
  recordRejectedSkillVersion(input: {
    candidateId: string;
    candidateVersionId: string;
    expectedActiveSkillVersionId: string | null;
    skillKey: string;
    contentHash: string;
    payload: Record<string, unknown>;
    createdAt: string;
  }): { record: LongTrajectorySkillVersionRecord; created: boolean } {
    return this.db.transaction(() => {
      const candidate = this.requireCandidate(input.candidateId);
      if (candidate.status !== "active" || candidate.activeVersionId !== input.candidateVersionId) {
        throw new Error("long trajectory Skill rejection must use the active candidate version");
      }
      assertExpectedVersion(
        `${candidate.id}:skill`,
        input.expectedActiveSkillVersionId,
        candidate.activeSkillVersionId ?? null
      );
      const existing = this.findSkillVersion(input.candidateVersionId, input.contentHash);
      if (existing) return { record: existing, created: false };
      const versionNo = Number((this.db.prepare(
        `SELECT COALESCE(MAX(version_no), 0) + 1 AS value
         FROM long_trajectory_skill_versions WHERE candidate_id = ?`
      ).get(candidate.id) as { value: number }).value);
      const id = `long_trajectory_skill_version_${stableHash({
        candidateId: candidate.id,
        candidateVersionId: input.candidateVersionId,
        contentHash: input.contentHash
      }).slice(0, 24)}`;
      this.db.prepare(
        `INSERT INTO long_trajectory_skill_versions (
          id, candidate_id, candidate_version_id, version_no, skill_key,
          content_hash, skill_memory_id, payload_json, status,
          supersedes_version_id, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 'superseded', ?, ?, NULL, ?)`
      ).run(
        id,
        candidate.id,
        input.candidateVersionId,
        versionNo,
        input.skillKey,
        input.contentHash,
        toJson(input.payload),
        candidate.activeSkillVersionId ?? null,
        input.createdAt,
        input.createdAt
      );
      return { record: this.getSkillVersion(id)!, created: true };
    })();
  }

  getSkillVersion(id: string): LongTrajectorySkillVersionRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM long_trajectory_skill_versions WHERE id = ?`)
      .get(id) as SkillVersionSqlRow | undefined;
    return row ? skillVersionFromSql(row) : undefined;
  }

  candidateForSkillMemory(skillMemoryId: string): LongTrajectoryCandidateRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM long_trajectory_candidates WHERE active_skill_memory_id = ? LIMIT 1`
    ).get(skillMemoryId) as CandidateSqlRow | undefined;
    return row ? candidateFromSql(row) : undefined;
  }

  private findCandidate(input: {
    userId: string;
    algorithmVersion: string;
    configHash: string;
    structureKey: string;
  }): LongTrajectoryCandidateRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM long_trajectory_candidates
       WHERE user_id = ? AND algorithm_version = ? AND config_hash = ?
         AND structure_key = ? LIMIT 1`
    ).get(
      input.userId,
      input.algorithmVersion,
      input.configHash,
      input.structureKey
    ) as CandidateSqlRow | undefined;
    return row ? candidateFromSql(row) : undefined;
  }

  private findCandidateVersion(
    candidateId: string,
    structureHash: string,
    evidenceHash: string,
    supportHash: string
  ): LongTrajectoryCandidateVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM long_trajectory_candidate_versions
       WHERE candidate_id = ? AND structure_hash = ? AND evidence_hash = ?
         AND support_hash = ? LIMIT 1`
    ).get(candidateId, structureHash, evidenceHash, supportHash) as
      CandidateVersionSqlRow | undefined;
    return row ? candidateVersionFromSql(row) : undefined;
  }

  private findSkillVersion(
    candidateVersionId: string,
    contentHash: string
  ): LongTrajectorySkillVersionRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM long_trajectory_skill_versions
       WHERE candidate_version_id = ? AND content_hash = ? LIMIT 1`
    ).get(candidateVersionId, contentHash) as SkillVersionSqlRow | undefined;
    return row ? skillVersionFromSql(row) : undefined;
  }

  private requireCandidate(id: string): LongTrajectoryCandidateRecord {
    const record = this.getCandidate(id);
    if (!record) throw new Error(`long trajectory candidate not found: ${id}`);
    return record;
  }

  private supersedeActiveCandidateVersion(
    candidate: LongTrajectoryCandidateRecord,
    at: string
  ): void {
    if (!candidate.activeVersionId) return;
    this.db.prepare(
      `UPDATE long_trajectory_candidate_versions
       SET status = 'superseded', deactivated_at = ? WHERE id = ?`
    ).run(at, candidate.activeVersionId);
  }

  private supersedeActiveSkillVersion(candidate: LongTrajectoryCandidateRecord, at: string): void {
    if (!candidate.activeSkillVersionId) return;
    this.db.prepare(
      `UPDATE long_trajectory_skill_versions
       SET status = 'superseded', deactivated_at = ? WHERE id = ?`
    ).run(at, candidate.activeSkillVersionId);
  }

  private updateCandidateVersionHead(candidateId: string, versionId: string, at: string): void {
    this.db.prepare(
      `UPDATE long_trajectory_candidates
       SET status = 'active', active_version_id = ?, updated_at = ? WHERE id = ?`
    ).run(versionId, at, candidateId);
  }

  private updateSkillHead(
    candidateId: string,
    version: LongTrajectorySkillVersionRecord,
    at: string
  ): void {
    this.db.prepare(
      `UPDATE long_trajectory_candidates
       SET active_skill_version_id = ?, active_skill_memory_id = ?, updated_at = ? WHERE id = ?`
    ).run(version.id, version.skillMemoryId ?? null, at, candidateId);
  }

  private assertFormalSkillMemory(memoryId: string, userId: string): void {
    const row = this.db.prepare(
      `SELECT user_id, memory_layer, memory_type, properties_json FROM memories WHERE id = ?`
    ).get(memoryId) as {
      user_id: string;
      memory_layer: string;
      memory_type: string;
      properties_json: string;
    } | undefined;
    const properties = parseJson<Record<string, unknown>>(row?.properties_json ?? "{}", {});
    const internal = properties.internal_info as Record<string, unknown> | undefined;
    if (!row || row.user_id !== userId || row.memory_layer !== "Skill" ||
        row.memory_type !== "SkillMemory" || !internal || !internal.skill) {
      throw new Error("long trajectory Skill version requires an upstream-compatible Skill memory");
    }
  }
}

function representationFromSql(row: RepresentationSqlRow): LongTrajectoryEpisodeRepresentationRecord {
  return {
    id: row.id,
    pathId: row.path_id,
    episodeId: row.episode_id,
    userId: row.user_id,
    representationVersion: row.representation_version,
    embeddingSignature: row.embedding_signature,
    goalText: row.goal_text,
    terminalResultText: row.terminal_result_text,
    goalHash: row.goal_hash,
    goalVector: parseJson<number[]>(row.goal_vector_json, []),
    trajectoryText: row.trajectory_text,
    trajectoryHash: row.trajectory_hash,
    trajectoryVector: parseJson<number[]>(row.trajectory_vector_json, []),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function candidateFromSql(row: CandidateSqlRow): LongTrajectoryCandidateRecord {
  return {
    id: row.id,
    userId: row.user_id,
    algorithmVersion: row.algorithm_version,
    configHash: row.config_hash,
    structureKey: row.structure_key,
    evidenceSignature: row.evidence_signature,
    status: row.status,
    ...(row.active_version_id ? { activeVersionId: row.active_version_id } : {}),
    ...(row.active_skill_version_id ? { activeSkillVersionId: row.active_skill_version_id } : {}),
    ...(row.active_skill_memory_id ? { activeSkillMemoryId: row.active_skill_memory_id } : {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function candidateVersionFromSql(
  row: CandidateVersionSqlRow
): LongTrajectoryCandidateVersionRecord {
  return {
    id: row.id,
    candidateId: row.candidate_id,
    versionNo: row.version_no,
    structureHash: row.structure_hash,
    evidenceHash: row.evidence_hash,
    supportHash: row.support_hash,
    referenceEpisodeId: row.reference_episode_id,
    sourcePathIds: parseJson<string[]>(row.source_path_ids_json, []),
    supportEpisodeIds: parseJson<string[]>(row.support_episode_ids_json, []),
    payload: parseJson<Record<string, unknown>>(row.payload_json, {}),
    status: row.status,
    ...(row.supersedes_version_id ? { supersedesVersionId: row.supersedes_version_id } : {}),
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function skillVersionFromSql(row: SkillVersionSqlRow): LongTrajectorySkillVersionRecord {
  return {
    id: row.id,
    candidateId: row.candidate_id,
    candidateVersionId: row.candidate_version_id,
    versionNo: row.version_no,
    skillKey: row.skill_key,
    contentHash: row.content_hash,
    ...(row.skill_memory_id ? { skillMemoryId: row.skill_memory_id } : {}),
    payload: parseJson<Record<string, unknown>>(row.payload_json, {}),
    status: row.status,
    ...(row.supersedes_version_id ? { supersedesVersionId: row.supersedes_version_id } : {}),
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function assertExpectedVersion(target: string, expected: string | null, actual: string | null): void {
  if (expected !== actual) {
    throw new Error(
      `long trajectory CAS conflict for ${target}: expected ${expected ?? "empty"}, ` +
      `found ${actual ?? "empty"}`
    );
  }
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}
