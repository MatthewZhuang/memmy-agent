import type Database from "better-sqlite3";
import type {
  EpisodeSpanCreditRunV1,
  SpanCreditEvidenceRole,
  SpanCreditV1
} from "../service/evolution/span-credit-model.js";
import { SPAN_CREDIT_ALGORITHM_VERSION } from "../service/evolution/span-credit-model.js";
import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";

export type EpisodeSpanCreditRunStatus = "active" | "inactive";

export interface EpisodeSpanCreditRunRecord {
  id: string;
  episodeId: string;
  pathId: string;
  pathHash: string;
  namespaceId: string;
  schemaVersion: string;
  algorithmVersion: string;
  promptVersion: string;
  rewardHash: string;
  goalAchievement: number;
  status: EpisodeSpanCreditRunStatus;
  scorerModel?: string;
  contentHash: string;
  run: EpisodeSpanCreditRunV1;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface ProceduralSpanCreditRecord extends SpanCreditV1 {
  runId: string;
  episodeId: string;
  pathId: string;
  pathHash: string;
  rewardHash: string;
  createdAt: string;
}

export interface SaveEpisodeSpanCreditRunResult {
  record: EpisodeSpanCreditRunRecord;
  credits: ProceduralSpanCreditRecord[];
  created: boolean;
}

interface CreditRunSqlRow {
  id: string;
  episode_id: string;
  path_id: string;
  path_hash: string;
  namespace_id: string;
  schema_version: string;
  algorithm_version: string;
  prompt_version: string;
  reward_hash: string;
  goal_achievement: number;
  status: EpisodeSpanCreditRunStatus;
  scorer_model: string | null;
  content_hash: string;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface SpanCreditSqlRow {
  id: string;
  run_id: string;
  occurrence_id: string;
  span_id: string;
  span_index: number;
  episode_id: string;
  path_id: string;
  path_hash: string;
  reward_hash: string;
  schema_version: string;
  pre_state_id: string;
  post_state_id: string;
  goal_credit: number;
  process_quality: number;
  confidence: number;
  credit_score: number;
  evidence_role: SpanCreditEvidenceRole;
  evidence_refs_json: string;
  reason: string;
  created_at: string;
}

export class ProceduralSpanCreditRepository {
  constructor(private readonly db: Database.Database) {}

  saveAndActivate(run: EpisodeSpanCreditRunV1, at: string): SaveEpisodeSpanCreditRunResult {
    const existing = this.get(run.id);
    if (existing) {
      if (existing.contentHash !== run.contentHash) {
        throw new Error(`SpanCredit run id collision: ${run.id}`);
      }
      const record = existing.status === "active" ? existing : this.activate(run.id, at);
      return { record, credits: this.listCredits(run.id), created: false };
    }
    this.assertRunSources(run);
    this.db.transaction(() => {
      this.deactivateEpisodeRuns(run.episodeId, at);
      this.db.prepare(
        `INSERT INTO episode_span_credit_runs (
          id, episode_id, path_id, path_hash, namespace_id, schema_version,
          algorithm_version, prompt_version, reward_hash, goal_achievement,
          status, scorer_model, content_hash, payload_json, created_at,
          activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, NULL)`
      ).run(
        run.id,
        run.episodeId,
        run.pathId,
        run.pathHash,
        run.namespaceId,
        run.schemaVersion,
        run.algorithmVersion,
        run.promptVersion,
        run.rewardHash,
        run.goalAchievement,
        run.scorerModel ?? null,
        run.contentHash,
        toJson(run),
        at,
        at
      );
      const insertCredit = this.db.prepare(
        `INSERT INTO procedural_span_credits (
          id, run_id, occurrence_id, span_id, span_index, episode_id, path_id,
          path_hash, reward_hash, schema_version, pre_state_id, post_state_id,
          goal_credit, process_quality, confidence, credit_score, evidence_role,
          evidence_refs_json, reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const credit of run.credits) {
        insertCredit.run(
          credit.id,
          run.id,
          credit.occurrenceId,
          credit.spanId,
          credit.spanIndex,
          run.episodeId,
          run.pathId,
          run.pathHash,
          run.rewardHash,
          credit.schemaVersion,
          credit.preStateId,
          credit.postStateId,
          credit.goalCredit,
          credit.processQuality,
          credit.confidence,
          credit.creditScore,
          credit.evidenceRole,
          toJson(credit.evidenceRefs),
          credit.reason,
          at
        );
      }
    })();
    return { record: this.get(run.id)!, credits: this.listCredits(run.id), created: true };
  }

  get(id: string): EpisodeSpanCreditRunRecord | undefined {
    const row = this.db.prepare(`SELECT * FROM episode_span_credit_runs WHERE id = ?`)
      .get(id) as CreditRunSqlRow | undefined;
    return row ? runFromSql(row) : undefined;
  }

  getActiveForEpisode(episodeId: string): EpisodeSpanCreditRunRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_span_credit_runs
       WHERE episode_id = ? AND status = 'active'
       ORDER BY activated_at DESC, created_at DESC, id DESC
       LIMIT 1`
    ).get(episodeId) as CreditRunSqlRow | undefined;
    return row ? runFromSql(row) : undefined;
  }

  getByBasis(
    pathId: string,
    rewardHash: string,
    algorithmVersion = SPAN_CREDIT_ALGORITHM_VERSION
  ): EpisodeSpanCreditRunRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_span_credit_runs
       WHERE path_id = ? AND reward_hash = ? AND algorithm_version = ?
       LIMIT 1`
    ).get(pathId, rewardHash, algorithmVersion) as CreditRunSqlRow | undefined;
    return row ? runFromSql(row) : undefined;
  }

  getActiveCreditForOccurrence(occurrenceId: string): ProceduralSpanCreditRecord | undefined {
    const row = this.db.prepare(
      `SELECT credits.*
       FROM procedural_span_credits AS credits
       JOIN episode_span_credit_runs AS runs ON runs.id = credits.run_id
       JOIN episode_procedural_paths AS paths ON paths.id = credits.path_id
       WHERE credits.occurrence_id = ?
         AND runs.status = 'active'
         AND paths.status = 'active'
       ORDER BY runs.activated_at DESC, credits.id DESC
       LIMIT 1`
    ).get(occurrenceId) as SpanCreditSqlRow | undefined;
    return row ? creditFromSql(row) : undefined;
  }

  listCredits(runId: string): ProceduralSpanCreditRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_span_credits
       WHERE run_id = ?
       ORDER BY span_index ASC, id ASC`
    ).all(runId) as SpanCreditSqlRow[]).map(creditFromSql);
  }

  listActiveCreditsForEpisode(episodeId: string): ProceduralSpanCreditRecord[] {
    const run = this.getActiveForEpisode(episodeId);
    return run ? this.listCredits(run.id) : [];
  }

  listActiveCreditsForNamespace(namespaceId: string): ProceduralSpanCreditRecord[] {
    return (this.db.prepare(
      `SELECT credits.*
       FROM procedural_span_credits AS credits
       JOIN episode_span_credit_runs AS runs ON runs.id = credits.run_id
       JOIN episode_procedural_paths AS paths ON paths.id = credits.path_id
       WHERE runs.namespace_id = ?
         AND runs.status = 'active'
         AND paths.status = 'active'
       ORDER BY credits.episode_id ASC, credits.span_index ASC, credits.id ASC`
    ).all(namespaceId) as SpanCreditSqlRow[]).map(creditFromSql);
  }

  activeNamespaceRevision(namespaceId: string): string | undefined {
    const rows = this.db.prepare(
      `SELECT runs.id, runs.episode_id, runs.path_id, runs.path_hash,
              runs.reward_hash, runs.content_hash
       FROM episode_span_credit_runs AS runs
       JOIN episode_procedural_paths AS paths ON paths.id = runs.path_id
       WHERE runs.namespace_id = ?
         AND runs.status = 'active'
         AND paths.status = 'active'
       ORDER BY runs.episode_id ASC, runs.id ASC`
    ).all(namespaceId) as Array<{
      id: string;
      episode_id: string;
      path_id: string;
      path_hash: string;
      reward_hash: string;
      content_hash: string;
    }>;
    if (rows.length === 0) return undefined;
    return stableHash({
      namespaceId,
      activeRuns: rows.map((row) => ({
        id: row.id,
        episodeId: row.episode_id,
        pathId: row.path_id,
        pathHash: row.path_hash,
        rewardHash: row.reward_hash,
        contentHash: row.content_hash
      }))
    });
  }

  activate(id: string, at: string): EpisodeSpanCreditRunRecord {
    const target = this.get(id);
    if (!target) throw new Error(`SpanCredit run not found: ${id}`);
    this.assertRunSources(target.run);
    this.db.transaction(() => {
      this.deactivateEpisodeRuns(target.episodeId, at);
      const result = this.db.prepare(
        `UPDATE episode_span_credit_runs
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(at, id);
      if (result.changes !== 1) throw new Error(`failed to activate SpanCredit run: ${id}`);
    })();
    return this.get(id)!;
  }

  private deactivateEpisodeRuns(episodeId: string, at: string): void {
    this.db.prepare(
      `UPDATE episode_span_credit_runs
       SET status = 'inactive', deactivated_at = ?
       WHERE episode_id = ? AND status = 'active'`
    ).run(at, episodeId);
  }

  private assertRunSources(run: EpisodeSpanCreditRunV1): void {
    const path = this.db.prepare(
      `SELECT id, episode_id, path_hash, namespace_id, status
       FROM episode_procedural_paths WHERE id = ?`
    ).get(run.pathId) as {
      id: string;
      episode_id: string;
      path_hash: string;
      namespace_id: string;
      status: string;
    } | undefined;
    if (!path || path.status !== "active") {
      throw new Error(`SpanCredit requires an active Episode path: ${run.pathId}`);
    }
    if (
      path.episode_id !== run.episodeId ||
      path.path_hash !== run.pathHash ||
      path.namespace_id !== run.namespaceId
    ) {
      throw new Error(`SpanCredit path metadata mismatch: ${run.id}`);
    }
    const occurrenceIds = new Set((this.db.prepare(
      `SELECT id FROM procedural_span_occurrences WHERE path_id = ?`
    ).all(run.pathId) as Array<{ id: string }>).map((row) => row.id));
    if (
      run.credits.length !== occurrenceIds.size ||
      run.credits.some((credit) => !occurrenceIds.has(credit.occurrenceId))
    ) {
      throw new Error(`SpanCredit must cover every occurrence in the active path: ${run.id}`);
    }
  }
}

function runFromSql(row: CreditRunSqlRow): EpisodeSpanCreditRunRecord {
  const run = parseJson<EpisodeSpanCreditRunV1 | null>(row.payload_json, null);
  if (!run) throw new Error(`stored SpanCredit run payload is invalid: ${row.id}`);
  return {
    id: row.id,
    episodeId: row.episode_id,
    pathId: row.path_id,
    pathHash: row.path_hash,
    namespaceId: row.namespace_id,
    schemaVersion: row.schema_version,
    algorithmVersion: row.algorithm_version,
    promptVersion: row.prompt_version,
    rewardHash: row.reward_hash,
    goalAchievement: row.goal_achievement,
    status: row.status,
    ...(row.scorer_model ? { scorerModel: row.scorer_model } : {}),
    contentHash: row.content_hash,
    run,
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function creditFromSql(row: SpanCreditSqlRow): ProceduralSpanCreditRecord {
  return {
    id: row.id,
    runId: row.run_id,
    occurrenceId: row.occurrence_id,
    spanId: row.span_id,
    spanIndex: row.span_index,
    episodeId: row.episode_id,
    pathId: row.path_id,
    pathHash: row.path_hash,
    rewardHash: row.reward_hash,
    schemaVersion: row.schema_version as SpanCreditV1["schemaVersion"],
    preStateId: row.pre_state_id,
    postStateId: row.post_state_id,
    goalCredit: row.goal_credit,
    processQuality: row.process_quality,
    confidence: row.confidence,
    creditScore: row.credit_score,
    evidenceRole: row.evidence_role,
    evidenceRefs: parseJson<string[]>(row.evidence_refs_json, []),
    reason: row.reason,
    createdAt: row.created_at
  };
}
