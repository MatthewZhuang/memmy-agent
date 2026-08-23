import type Database from "better-sqlite3";
import {
  POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
  POLICY_SEQUENCE_MIN_LENGTH,
  POLICY_SEQUENCE_OBSERVED_SUPPORT,
  POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION,
  POLICY_SEQUENCE_READY_SUPPORT,
  PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION,
  PROCEDURAL_SKILL_CANDIDATE_SCHEMA_VERSION,
  buildProceduralSkillCandidate,
  classifyPolicySequencePatternTopology,
  hasMultipleDistinctPolicies,
  mergePolicySequenceOccurrences,
  policySequencePatternIdentity,
  type CapabilityType,
  type PolicySequencePatternLifecycleStatus,
  type PolicySequencePatternOccurrenceV1,
  type ProceduralSkillCandidateLifecycleStatus,
  type ProceduralSkillCandidateV1
} from "../service/evolution/policy-sequence-pattern-model.js";
import type { EpisodePolicyProjectionRecord } from "./episode-policy-projection-repository.js";
import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";

export type PolicySequencePatternOccurrenceStatus = "active" | "superseded";
export type ProceduralSkillCandidateStatus = "active" | "inactive";

export interface PolicySequencePatternRecord {
  id: string;
  namespaceId: string;
  schemaVersion: typeof POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION;
  algorithmVersion: typeof POLICY_SEQUENCE_MINING_ALGORITHM_VERSION;
  sequenceHash: string;
  policyKeys: string[];
  capabilityType: CapabilityType;
  episodeFamilyId?: string;
  lifecycleStatus: PolicySequencePatternLifecycleStatus;
  occurrenceCount: number;
  distinctEpisodeCount: number;
  distinctSupportEpisodeCount: number;
  distinctCounterexampleEpisodeCount: number;
  distinctUncertainEpisodeCount: number;
  isClosed: boolean;
  isMaximal: boolean;
  membershipVersion: string;
  createdAt: string;
  updatedAt: string;
}

export interface PolicySequencePatternOccurrenceRecord {
  occurrence: PolicySequencePatternOccurrenceV1;
  status: PolicySequencePatternOccurrenceStatus;
  createdAt: string;
  supersededAt?: string | null;
}

export interface ProceduralSkillCandidateRecord {
  id: string;
  candidateKey: string;
  namespaceId: string;
  patternId: string;
  patternMembershipVersion: string;
  schemaVersion: typeof PROCEDURAL_SKILL_CANDIDATE_SCHEMA_VERSION;
  inductionVersion: typeof PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION;
  lifecycleStatus: ProceduralSkillCandidateLifecycleStatus;
  status: ProceduralSkillCandidateStatus;
  distinctSupportEpisodeCount: number;
  distinctCounterexampleEpisodeCount: number;
  evidenceHash: string;
  candidate: ProceduralSkillCandidateV1;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface IngestPolicySequenceProjectionResult {
  projectionId: string;
  extractedOccurrenceCount: number;
  affectedPatternIds: string[];
  patterns: PolicySequencePatternRecord[];
  activeCandidates: ProceduralSkillCandidateRecord[];
}

interface PatternSqlRow {
  id: string;
  namespace_id: string;
  schema_version: string;
  algorithm_version: string;
  sequence_hash: string;
  capability_type: CapabilityType;
  episode_family_id: string | null;
  policy_keys_json: string;
  lifecycle_status: PolicySequencePatternLifecycleStatus;
  occurrence_count: number;
  distinct_episode_count: number;
  distinct_support_episode_count: number;
  distinct_counterexample_episode_count: number;
  distinct_uncertain_episode_count: number;
  is_closed: number;
  is_maximal: number;
  membership_version: string;
  created_at: string;
  updated_at: string;
}

interface OccurrenceSqlRow {
  payload_json: string;
  status: PolicySequencePatternOccurrenceStatus;
  created_at: string;
  superseded_at: string | null;
}

interface CandidateSqlRow {
  id: string;
  candidate_key: string;
  namespace_id: string;
  pattern_id: string;
  pattern_membership_version: string;
  schema_version: string;
  induction_version: string;
  lifecycle_status: ProceduralSkillCandidateLifecycleStatus;
  status: ProceduralSkillCandidateStatus;
  distinct_support_episode_count: number;
  distinct_counterexample_episode_count: number;
  evidence_hash: string;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface PatternProjectionSource {
  id: string;
  episodeId: string;
  pathId: string;
  sessionId: string;
  namespaceId: string;
  status: string;
}

export class PolicySequencePatternRepository {
  constructor(private readonly db: Database.Database) {}

  ingestProjection(input: {
    projection: EpisodePolicyProjectionRecord;
    occurrences: readonly PolicySequencePatternOccurrenceV1[];
    at: string;
  }): IngestPolicySequenceProjectionResult {
    this.verifyInput(input.projection, input.occurrences);
    const patterns = this.db.transaction(() => {
      const affectedByNamespace = new Map<string, Set<string>>();
      const previousPatterns = this.db.prepare(
        `SELECT DISTINCT occurrences.pattern_id, patterns.namespace_id
         FROM policy_sequence_pattern_occurrences AS occurrences
         JOIN policy_sequence_patterns AS patterns ON patterns.id = occurrences.pattern_id
         WHERE occurrences.episode_id = ? AND occurrences.status = 'active'`
      ).all(input.projection.episodeId) as Array<{
        pattern_id: string;
        namespace_id: string;
      }>;
      for (const previous of previousPatterns) {
        addAffectedPattern(
          affectedByNamespace,
          previous.namespace_id,
          previous.pattern_id
        );
      }
      this.db.prepare(
        `UPDATE policy_sequence_pattern_occurrences
         SET status = 'superseded', superseded_at = ?
         WHERE episode_id = ? AND status = 'active'`
      ).run(input.at, input.projection.episodeId);

      const upsertPattern = this.db.prepare(
        `INSERT INTO policy_sequence_patterns (
          id, namespace_id, schema_version, algorithm_version, sequence_hash,
          capability_type, episode_family_id, policy_keys_json,
          lifecycle_status, occurrence_count,
          distinct_episode_count, distinct_support_episode_count,
          distinct_counterexample_episode_count, distinct_uncertain_episode_count,
          is_closed, is_maximal, membership_version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'forming', 0, 0, 0, 0, 0, 1, 1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          capability_type = excluded.capability_type,
          episode_family_id = excluded.episode_family_id,
          policy_keys_json = excluded.policy_keys_json,
          updated_at = excluded.updated_at`
      );
      const upsertOccurrence = this.db.prepare(
        `INSERT INTO policy_sequence_pattern_occurrences (
          id, pattern_id, projection_id, episode_id, path_id, session_id,
          start_node_index, end_node_index, evidence_role, terminal_reward,
          payload_json, status, created_at, superseded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, NULL)
        ON CONFLICT(id) DO UPDATE SET
          evidence_role = excluded.evidence_role,
          terminal_reward = excluded.terminal_reward,
          payload_json = excluded.payload_json,
          status = 'active',
          superseded_at = NULL`
      );
      const existingOccurrence = this.db.prepare(
        `SELECT payload_json, status, created_at, superseded_at
         FROM policy_sequence_pattern_occurrences WHERE id = ?`
      );
      for (const inputOccurrence of input.occurrences) {
        const existingRow = existingOccurrence.get(inputOccurrence.id) as OccurrenceSqlRow | undefined;
        const occurrence = existingRow
          ? mergePolicySequenceOccurrences([
              occurrenceFromSql(existingRow).occurrence,
              inputOccurrence
            ])[0]!
          : inputOccurrence;
        const emptyMembershipVersion = stableHash([]);
        upsertPattern.run(
          occurrence.patternId,
          occurrence.namespaceId,
          occurrence.schemaVersion,
          occurrence.algorithmVersion,
          occurrence.sequenceHash,
          occurrence.capabilityType,
          occurrence.capabilityType === "task_skill"
            ? occurrence.episodeFamilyId ?? null
            : null,
          toJson(occurrence.policyKeys),
          emptyMembershipVersion,
          input.at,
          input.at
        );
        upsertOccurrence.run(
          occurrence.id,
          occurrence.patternId,
          occurrence.projectionId,
          occurrence.episodeId,
          occurrence.pathId,
          occurrence.sessionId,
          occurrence.startNodeIndex,
          occurrence.endNodeIndex,
          occurrence.evidenceRole,
          occurrence.terminalReward ?? null,
          toJson(occurrence),
          input.at
        );
        addAffectedPattern(
          affectedByNamespace,
          occurrence.namespaceId,
          occurrence.patternId
        );
      }

      const refreshed: PolicySequencePatternRecord[] = [];
      for (const [namespaceId, directlyAffectedIds] of affectedByNamespace) {
        this.upsertTopologyEdges(namespaceId, directlyAffectedIds);
        refreshed.push(...this.refreshAffectedPatterns(
          namespaceId,
          directlyAffectedIds,
          input.at
        ));
      }
      return uniquePatterns(refreshed);
    })();
    const activeCandidates = patterns
      .map((pattern) => this.getActiveCandidateForPattern(pattern.id))
      .filter((candidate): candidate is ProceduralSkillCandidateRecord => Boolean(candidate));
    return {
      projectionId: input.projection.id,
      extractedOccurrenceCount: input.occurrences.length,
      affectedPatternIds: patterns.map((pattern) => pattern.id),
      patterns,
      activeCandidates
    };
  }

  getPattern(id: string): PolicySequencePatternRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM policy_sequence_patterns WHERE id = ?`
    ).get(id) as PatternSqlRow | undefined;
    return row ? patternFromSql(row) : undefined;
  }

  getPatternBySequence(
    namespaceId: string,
    policyKeys: readonly string[],
    options: {
      capabilityType?: CapabilityType;
      episodeFamilyId?: string;
    } = {}
  ): PolicySequencePatternRecord | undefined {
    const sequenceHash = policySequencePatternIdentity(namespaceId, policyKeys, options).sequenceHash;
    const row = this.db.prepare(
      `SELECT * FROM policy_sequence_patterns
       WHERE namespace_id = ? AND algorithm_version = ? AND sequence_hash = ?`
    ).get(
      namespaceId.trim(),
      POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
      sequenceHash
    ) as PatternSqlRow | undefined;
    return row ? patternFromSql(row) : undefined;
  }

  listPatternsForNamespace(namespaceId: string): PolicySequencePatternRecord[] {
    return (this.db.prepare(
      `SELECT * FROM policy_sequence_patterns
       WHERE namespace_id = ? AND algorithm_version = ?
       ORDER BY json_array_length(policy_keys_json) DESC, sequence_hash ASC`
    ).all(namespaceId, POLICY_SEQUENCE_MINING_ALGORITHM_VERSION) as PatternSqlRow[])
      .map(patternFromSql);
  }

  listActiveOccurrencesForPattern(patternId: string): PolicySequencePatternOccurrenceRecord[] {
    return (this.db.prepare(
      `SELECT payload_json, status, created_at, superseded_at
       FROM policy_sequence_pattern_occurrences
       WHERE pattern_id = ? AND status = 'active'
       ORDER BY episode_id ASC, start_node_index ASC, id ASC`
    ).all(patternId) as OccurrenceSqlRow[]).map(occurrenceFromSql);
  }

  getActiveCandidateForPattern(patternId: string): ProceduralSkillCandidateRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_skill_candidates
       WHERE pattern_id = ? AND status = 'active'
       LIMIT 1`
    ).get(patternId) as CandidateSqlRow | undefined;
    return row ? candidateFromSql(row) : undefined;
  }

  listCandidatesForPattern(patternId: string): ProceduralSkillCandidateRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_skill_candidates
       WHERE pattern_id = ?
       ORDER BY created_at DESC, id DESC`
    ).all(patternId) as CandidateSqlRow[]).map(candidateFromSql);
  }

  listActiveCandidatesForNamespace(namespaceId: string): ProceduralSkillCandidateRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_skill_candidates
       WHERE namespace_id = ? AND status = 'active'
       ORDER BY lifecycle_status DESC, created_at DESC, id DESC`
    ).all(namespaceId) as CandidateSqlRow[]).map(candidateFromSql);
  }

  rebuildNamespace(namespaceId: string, at: string): PolicySequencePatternRecord[] {
    return this.db.transaction(() => {
      const patterns = this.listPatternsForNamespace(namespaceId);
      this.db.prepare(
        `DELETE FROM policy_sequence_pattern_edges
         WHERE parent_pattern_id IN (
           SELECT id FROM policy_sequence_patterns
           WHERE namespace_id = ? AND algorithm_version = ?
         )`
      ).run(namespaceId, POLICY_SEQUENCE_MINING_ALGORITHM_VERSION);
      const patternIds = new Set(patterns.map((pattern) => pattern.id));
      this.upsertTopologyEdges(namespaceId, patternIds);
      return this.refreshAffectedPatterns(namespaceId, patternIds, at);
    })();
  }

  private refreshAffectedPatterns(
    namespaceId: string,
    directlyAffectedIds: ReadonlySet<string>,
    at: string
  ): PolicySequencePatternRecord[] {
    if (directlyAffectedIds.size === 0) return [];
    const directPatterns = this.listPatternsByIds(namespaceId, directlyAffectedIds);
    const targetIds = new Set(directPatterns.map((pattern) => pattern.id));
    for (const parentId of this.listRelatedPatternIds(
      "parent_pattern_id",
      "child_pattern_id",
      targetIds
    )) targetIds.add(parentId);
    const targets = this.listPatternsByIds(namespaceId, targetIds);
    const contextIds = new Set(targets.map((pattern) => pattern.id));
    for (const childId of this.listRelatedPatternIds(
      "child_pattern_id",
      "parent_pattern_id",
      contextIds
    )) contextIds.add(childId);
    const contextPatterns = this.listPatternsByIds(namespaceId, contextIds);
    const occurrenceRows = this.listActiveOccurrenceRows(contextIds);
    const occurrencesByPattern = new Map<string, PolicySequencePatternOccurrenceV1[]>();
    for (const row of occurrenceRows) {
      const values = occurrencesByPattern.get(row.pattern_id) ?? [];
      values.push(occurrenceFromSql(row).occurrence);
      occurrencesByPattern.set(row.pattern_id, values);
    }
    const topology = new Map(classifyPolicySequencePatternTopology(contextPatterns
      .filter((pattern) => (occurrencesByPattern.get(pattern.id)?.length ?? 0) > 0)
      .map((pattern) => ({
        patternId: pattern.id,
        policyKeys: pattern.policyKeys,
        capabilityType: pattern.capabilityType,
        ...(pattern.episodeFamilyId ? { episodeFamilyId: pattern.episodeFamilyId } : {}),
        supportEpisodeIds: distinctEpisodeIds(
          occurrencesByPattern.get(pattern.id) ?? [],
          "support"
        )
      }))).map((item) => [item.patternId, item]));

    const refreshed: PolicySequencePatternRecord[] = [];
    for (const pattern of targets) {
      const occurrences = occurrencesByPattern.get(pattern.id) ?? [];
      const supportEpisodeIds = distinctEpisodeIds(occurrences, "support");
      const counterexampleEpisodeIds = distinctEpisodeIds(occurrences, "counterexample");
      const uncertainEpisodeIds = distinctEpisodeIds(occurrences, "uncertain");
      const allEpisodeIds = unique(occurrences.map((item) => item.episodeId));
      const lifecycleStatus = patternLifecycleStatus(
        occurrences.length,
        supportEpisodeIds.length
      );
      const patternTopology = topology.get(pattern.id) ?? {
        isClosed: true,
        isMaximal: true
      };
      const membershipVersion = stableHash(occurrences
        .map((occurrence) => ({
          id: occurrence.id,
          evidenceRole: occurrence.evidenceRole,
          terminalReward: occurrence.terminalReward
        }))
        .sort((left, right) => left.id.localeCompare(right.id)));
      this.db.prepare(
        `UPDATE policy_sequence_patterns
         SET lifecycle_status = ?, occurrence_count = ?, distinct_episode_count = ?,
             distinct_support_episode_count = ?,
             distinct_counterexample_episode_count = ?,
             distinct_uncertain_episode_count = ?, is_closed = ?, is_maximal = ?,
             membership_version = ?, updated_at = ?
         WHERE id = ?`
      ).run(
        lifecycleStatus,
        occurrences.length,
        allEpisodeIds.length,
        supportEpisodeIds.length,
        counterexampleEpisodeIds.length,
        uncertainEpisodeIds.length,
        patternTopology.isClosed ? 1 : 0,
        patternTopology.isMaximal ? 1 : 0,
        membershipVersion,
        at,
        pattern.id
      );
      const updatedPattern: PolicySequencePatternRecord = {
          ...pattern,
          lifecycleStatus,
          occurrenceCount: occurrences.length,
          distinctEpisodeCount: allEpisodeIds.length,
          distinctSupportEpisodeCount: supportEpisodeIds.length,
          distinctCounterexampleEpisodeCount: counterexampleEpisodeIds.length,
          distinctUncertainEpisodeCount: uncertainEpisodeIds.length,
          isClosed: patternTopology.isClosed,
          isMaximal: patternTopology.isMaximal,
          membershipVersion,
          updatedAt: at
      };
      this.reconcileCandidate({
        pattern: updatedPattern,
        occurrences,
        at
      });
      refreshed.push(updatedPattern);
    }
    return refreshed.sort(comparePattern);
  }

  private upsertTopologyEdges(
    namespaceId: string,
    directlyAffectedIds: ReadonlySet<string>
  ): void {
    const patterns = this.listPatternsByIds(namespaceId, directlyAffectedIds);
    const insert = this.db.prepare(
      `INSERT OR IGNORE INTO policy_sequence_pattern_edges (
         parent_pattern_id, child_pattern_id
       )
       SELECT id, ? FROM policy_sequence_patterns
       WHERE id = ? AND namespace_id = ? AND algorithm_version = ?`
    );
    for (const child of patterns) {
      if (child.policyKeys.length <= POLICY_SEQUENCE_MIN_LENGTH) continue;
      for (const parentKeys of [
        child.policyKeys.slice(1),
        child.policyKeys.slice(0, -1)
      ]) {
        const parentId = policySequencePatternIdentity(
          child.namespaceId,
          parentKeys,
          child.capabilityType === "task_skill"
            ? { capabilityType: "task_skill", episodeFamilyId: child.episodeFamilyId }
            : { capabilityType: "sub_skill" }
        ).patternId;
        insert.run(
          child.id,
          parentId,
          child.namespaceId,
          POLICY_SEQUENCE_MINING_ALGORITHM_VERSION
        );
      }
    }
  }

  private listPatternsByIds(
    namespaceId: string,
    ids: ReadonlySet<string>
  ): PolicySequencePatternRecord[] {
    const rows: PatternSqlRow[] = [];
    for (const batch of chunks([...ids], 400)) {
      if (batch.length === 0) continue;
      const placeholders = batch.map(() => "?").join(", ");
      rows.push(...this.db.prepare(
        `SELECT * FROM policy_sequence_patterns
         WHERE namespace_id = ? AND algorithm_version = ?
           AND id IN (${placeholders})`
      ).all(
        namespaceId,
        POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
        ...batch
      ) as PatternSqlRow[]);
    }
    return rows.map(patternFromSql).sort(comparePattern);
  }

  private listRelatedPatternIds(
    selectColumn: "parent_pattern_id" | "child_pattern_id",
    filterColumn: "parent_pattern_id" | "child_pattern_id",
    ids: ReadonlySet<string>
  ): string[] {
    const result = new Set<string>();
    for (const batch of chunks([...ids], 400)) {
      if (batch.length === 0) continue;
      const placeholders = batch.map(() => "?").join(", ");
      const rows = this.db.prepare(
        `SELECT DISTINCT ${selectColumn} AS id
         FROM policy_sequence_pattern_edges
         WHERE ${filterColumn} IN (${placeholders})`
      ).all(...batch) as Array<{ id: string }>;
      for (const row of rows) result.add(row.id);
    }
    return [...result].sort();
  }

  private listActiveOccurrenceRows(
    patternIds: ReadonlySet<string>
  ): Array<OccurrenceSqlRow & { pattern_id: string }> {
    const rows: Array<OccurrenceSqlRow & { pattern_id: string }> = [];
    for (const batch of chunks([...patternIds], 400)) {
      if (batch.length === 0) continue;
      const placeholders = batch.map(() => "?").join(", ");
      rows.push(...this.db.prepare(
        `SELECT pattern_id, payload_json, status, created_at, superseded_at
         FROM policy_sequence_pattern_occurrences
         WHERE pattern_id IN (${placeholders}) AND status = 'active'
         ORDER BY pattern_id ASC, episode_id ASC, start_node_index ASC, id ASC`
      ).all(...batch) as Array<OccurrenceSqlRow & { pattern_id: string }>);
    }
    return rows;
  }

  private reconcileCandidate(input: {
    pattern: PolicySequencePatternRecord;
    occurrences: readonly PolicySequencePatternOccurrenceV1[];
    at: string;
  }): void {
    if (!input.pattern.isClosed ||
        !hasMultipleDistinctPolicies(input.pattern.policyKeys) ||
        input.pattern.distinctSupportEpisodeCount < POLICY_SEQUENCE_OBSERVED_SUPPORT) {
      this.deactivateActiveCandidate(input.pattern.id, input.at);
      return;
    }
    const candidate = buildProceduralSkillCandidate({
      patternId: input.pattern.id,
      membershipVersion: input.pattern.membershipVersion,
      sequenceHash: input.pattern.sequenceHash,
      namespaceId: input.pattern.namespaceId,
      policyKeys: input.pattern.policyKeys,
      capabilityType: input.pattern.capabilityType,
      ...(input.pattern.episodeFamilyId ? {
        episodeFamilyId: input.pattern.episodeFamilyId
      } : {}),
      isClosed: true,
      isMaximal: input.pattern.isMaximal,
      occurrences: input.occurrences
    });
    const existing = this.db.prepare(
      `SELECT id, status FROM procedural_skill_candidates WHERE id = ?`
    ).get(candidate.id) as { id: string; status: ProceduralSkillCandidateStatus } | undefined;
    const active = this.getActiveCandidateForPattern(input.pattern.id);
    if (active?.id !== candidate.id) this.deactivateActiveCandidate(input.pattern.id, input.at);
    if (!existing) {
      this.db.prepare(
        `INSERT INTO procedural_skill_candidates (
          id, candidate_key, namespace_id, pattern_id, pattern_membership_version,
          schema_version, induction_version, lifecycle_status, status,
          distinct_support_episode_count, distinct_counterexample_episode_count,
          evidence_hash, payload_json, created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, NULL)`
      ).run(
        candidate.id,
        candidate.candidateKey,
        candidate.namespaceId,
        candidate.patternId,
        candidate.patternMembershipVersion,
        candidate.schemaVersion,
        candidate.inductionVersion,
        candidate.lifecycleStatus,
        candidate.supportEpisodeIds.length,
        candidate.counterexampleEpisodeIds.length,
        candidate.evidenceHash,
        toJson(candidate),
        input.at,
        input.at
      );
      return;
    }
    this.db.prepare(
      `UPDATE procedural_skill_candidates
       SET lifecycle_status = ?, status = 'active',
           distinct_support_episode_count = ?,
           distinct_counterexample_episode_count = ?, evidence_hash = ?,
           payload_json = ?, activated_at = ?, deactivated_at = NULL
       WHERE id = ?`
    ).run(
      candidate.lifecycleStatus,
      candidate.supportEpisodeIds.length,
      candidate.counterexampleEpisodeIds.length,
      candidate.evidenceHash,
      toJson(candidate),
      input.at,
      candidate.id
    );
  }

  private deactivateActiveCandidate(patternId: string, at: string): void {
    this.db.prepare(
      `UPDATE procedural_skill_candidates
       SET status = 'inactive', deactivated_at = ?
       WHERE pattern_id = ? AND status = 'active'`
    ).run(at, patternId);
  }

  private verifyInput(
    projection: EpisodePolicyProjectionRecord,
    occurrences: readonly PolicySequencePatternOccurrenceV1[]
  ): void {
    if (projection.status !== "active") {
      throw new Error(`Policy sequence mining requires an active Projection: ${projection.id}`);
    }
    for (const occurrence of occurrences) {
      const occurrenceProjection: PatternProjectionSource | undefined =
        occurrence.projectionId === projection.id
          ? {
              id: projection.id,
              episodeId: projection.episodeId,
              pathId: projection.pathId,
              sessionId: projection.sessionId,
              namespaceId: projection.namespaceId,
              status: projection.status
            }
          : this.db.prepare(
              `SELECT id, episode_id AS episodeId, path_id AS pathId,
                      session_id AS sessionId, namespace_id AS namespaceId, status
               FROM episode_policy_projections WHERE id = ?`
            ).get(occurrence.projectionId) as PatternProjectionSource | undefined;
      if (!occurrenceProjection || occurrenceProjection.status !== "active" ||
          occurrence.episodeId !== occurrenceProjection.episodeId ||
          occurrence.pathId !== occurrenceProjection.pathId ||
          occurrence.sessionId !== occurrenceProjection.sessionId ||
          occurrence.namespaceId !== occurrenceProjection.namespaceId ||
          occurrence.namespaceId !== projection.namespaceId ||
          occurrence.algorithmVersion !== POLICY_SEQUENCE_MINING_ALGORITHM_VERSION ||
          occurrence.schemaVersion !== POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION) {
        throw new Error(`Policy sequence occurrence source mismatch: ${occurrence.id}`);
      }
    }
  }
}

function patternLifecycleStatus(
  occurrenceCount: number,
  distinctSupportEpisodeCount: number
): PolicySequencePatternLifecycleStatus {
  if (occurrenceCount === 0) return "stale";
  if (distinctSupportEpisodeCount >= POLICY_SEQUENCE_READY_SUPPORT) return "ready";
  if (distinctSupportEpisodeCount >= POLICY_SEQUENCE_OBSERVED_SUPPORT) return "observed";
  return "forming";
}

function distinctEpisodeIds(
  occurrences: readonly PolicySequencePatternOccurrenceV1[],
  role: PolicySequencePatternOccurrenceV1["evidenceRole"]
): string[] {
  return unique(occurrences
    .filter((occurrence) => occurrence.evidenceRole === role)
    .map((occurrence) => occurrence.episodeId));
}

function patternFromSql(row: PatternSqlRow): PolicySequencePatternRecord {
  const policyKeys = parseJson<string[]>(row.policy_keys_json, []);
  if (row.schema_version !== POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION ||
      row.algorithm_version !== POLICY_SEQUENCE_MINING_ALGORITHM_VERSION ||
      policyKeys.length < 2) {
    throw new Error(`stored Policy sequence pattern is invalid: ${row.id}`);
  }
  return {
    id: row.id,
    namespaceId: row.namespace_id,
    schemaVersion: POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION,
    algorithmVersion: POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
    sequenceHash: row.sequence_hash,
    policyKeys,
    capabilityType: row.capability_type,
    ...(row.episode_family_id ? { episodeFamilyId: row.episode_family_id } : {}),
    lifecycleStatus: row.lifecycle_status,
    occurrenceCount: row.occurrence_count,
    distinctEpisodeCount: row.distinct_episode_count,
    distinctSupportEpisodeCount: row.distinct_support_episode_count,
    distinctCounterexampleEpisodeCount: row.distinct_counterexample_episode_count,
    distinctUncertainEpisodeCount: row.distinct_uncertain_episode_count,
    isClosed: row.is_closed === 1,
    isMaximal: row.is_maximal === 1,
    membershipVersion: row.membership_version,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function occurrenceFromSql(row: OccurrenceSqlRow): PolicySequencePatternOccurrenceRecord {
  const occurrence = parseJson<PolicySequencePatternOccurrenceV1 | null>(row.payload_json, null);
  if (!occurrence || occurrence.schemaVersion !== POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION ||
      occurrence.algorithmVersion !== POLICY_SEQUENCE_MINING_ALGORITHM_VERSION) {
    throw new Error("stored Policy sequence occurrence is invalid");
  }
  return {
    occurrence,
    status: row.status,
    createdAt: row.created_at,
    supersededAt: row.superseded_at
  };
}

function candidateFromSql(row: CandidateSqlRow): ProceduralSkillCandidateRecord {
  const candidate = parseJson<ProceduralSkillCandidateV1 | null>(row.payload_json, null);
  if (!candidate || candidate.id !== row.id || candidate.patternId !== row.pattern_id ||
      candidate.schemaVersion !== PROCEDURAL_SKILL_CANDIDATE_SCHEMA_VERSION ||
      candidate.inductionVersion !== PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION ||
      candidate.patternMembershipVersion !== row.pattern_membership_version ||
      candidate.evidenceHash !== row.evidence_hash) {
    throw new Error(`stored procedural SkillCandidate is invalid: ${row.id}`);
  }
  return {
    id: row.id,
    candidateKey: row.candidate_key,
    namespaceId: row.namespace_id,
    patternId: row.pattern_id,
    patternMembershipVersion: row.pattern_membership_version,
    schemaVersion: PROCEDURAL_SKILL_CANDIDATE_SCHEMA_VERSION,
    inductionVersion: PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION,
    lifecycleStatus: row.lifecycle_status,
    status: row.status,
    distinctSupportEpisodeCount: row.distinct_support_episode_count,
    distinctCounterexampleEpisodeCount: row.distinct_counterexample_episode_count,
    evidenceHash: row.evidence_hash,
    candidate,
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function addAffectedPattern(
  affectedByNamespace: Map<string, Set<string>>,
  namespaceId: string,
  patternId: string
): void {
  const ids = affectedByNamespace.get(namespaceId) ?? new Set<string>();
  ids.add(patternId);
  affectedByNamespace.set(namespaceId, ids);
}

function uniquePatterns(
  patterns: readonly PolicySequencePatternRecord[]
): PolicySequencePatternRecord[] {
  return [...new Map(patterns.map((pattern) => [pattern.id, pattern])).values()]
    .sort(comparePattern);
}

function comparePattern(
  left: PolicySequencePatternRecord,
  right: PolicySequencePatternRecord
): number {
  return right.policyKeys.length - left.policyKeys.length ||
    left.sequenceHash.localeCompare(right.sequenceHash);
}

function chunks<T>(values: readonly T[], size: number): T[][] {
  const result: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    result.push(values.slice(index, index + size));
  }
  return result;
}
