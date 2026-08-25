import type Database from "better-sqlite3";
import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";
import { clip } from "../utils/text.js";
import {
  EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION,
  PROCEDURAL_SPAN_SCHEMA_VERSION,
  buildEpisodeProceduralPath,
  type EpisodeProceduralPathV2,
  type ExecutionStepV1,
  type ProceduralSpanTermination,
  type ProceduralSpanV1
} from "../service/evolution/procedural-path-model.js";

export const PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION = "procedural-span-occurrence.v1" as const;
export const PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSION = "procedural-span-cluster-projection.v5" as const;
const LEGACY_PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSIONS = [
  "procedural-span-cluster-projection.v1",
  "procedural-span-cluster-projection.v2",
  "procedural-span-cluster-projection.v3",
  "procedural-span-cluster-projection.v4"
] as const;
type ProceduralSpanClusterProjectionVersion =
  | typeof PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSION
  | typeof LEGACY_PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSIONS[number];

export type EpisodeProceduralPathStatus = "active" | "inactive";

export interface EpisodeProceduralPathRecord {
  id: string;
  episodeId: string;
  userId: string;
  sessionId: string;
  namespaceId: string;
  schemaVersion: typeof EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION;
  reconstructionAlgorithmVersion: string;
  reconstructionModel?: string;
  sourceSnapshotHash: string;
  pathHash: string;
  terminalReward?: number;
  status: EpisodeProceduralPathStatus;
  path: EpisodeProceduralPathV2;
  createdAt: string;
  activatedAt?: string | null;
  deactivatedAt?: string | null;
}

export interface ProceduralSpanClusterProjectionV1 {
  version: ProceduralSpanClusterProjectionVersion;
  goalText: string;
  conditionText: string;
  procedureText: string;
  effectText: string;
  structureSignature: string;
}

export interface ProceduralSpanOccurrenceRecord {
  id: string;
  pathId: string;
  pathHash: string;
  spanId: string;
  episodeId: string;
  namespaceId: string;
  schemaVersion: typeof PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION;
  spanIndex: number;
  localGoal: string;
  capabilityGoal: string;
  entryCondition: string;
  exitCondition: string;
  terminationStatus: ProceduralSpanTermination;
  rawTurnIds: string[];
  stepIds: string[];
  preStateId: string;
  postStateId: string;
  projection: ProceduralSpanClusterProjectionV1;
  span: ProceduralSpanV1;
  reconstructionAlgorithmVersion: string;
  reconstructionModel?: string;
  createdAt: string;
}

export interface SaveEpisodeProceduralPathInput {
  path: EpisodeProceduralPathV2;
  namespaceId: string;
  reconstructionAlgorithmVersion?: string;
  reconstructionModel?: string;
  activate?: boolean;
  createdAt: string;
}

export interface SaveEpisodeProceduralPathResult {
  record: EpisodeProceduralPathRecord;
  occurrences: ProceduralSpanOccurrenceRecord[];
  created: boolean;
}

interface EpisodeSourceRow {
  id: string;
  user_id: string;
  session_id: string;
}

interface EpisodeProceduralPathSqlRow {
  id: string;
  episode_id: string;
  user_id: string;
  session_id: string;
  namespace_id: string;
  schema_version: string;
  reconstruction_algorithm_version: string;
  reconstruction_model: string | null;
  source_snapshot_hash: string;
  path_hash: string;
  terminal_reward: number | null;
  status: EpisodeProceduralPathStatus;
  payload_json: string;
  created_at: string;
  activated_at: string | null;
  deactivated_at: string | null;
}

interface ProceduralSpanOccurrenceSqlRow {
  id: string;
  path_id: string;
  path_hash: string;
  span_id: string;
  episode_id: string;
  namespace_id: string;
  schema_version: string;
  projection_version: string;
  span_index: number;
  local_goal: string;
  entry_condition: string;
  exit_condition: string;
  termination_status: ProceduralSpanTermination;
  raw_turn_ids_json: string;
  step_ids_json: string;
  pre_state_id: string;
  post_state_id: string;
  goal_text: string;
  condition_text: string;
  procedure_text: string;
  effect_text: string;
  structure_signature: string;
  span_json: string;
  reconstruction_algorithm_version: string;
  reconstruction_model: string | null;
  created_at: string;
}

export class EpisodeProceduralPathRepository {
  constructor(private readonly db: Database.Database) {}

  save(input: SaveEpisodeProceduralPathInput): SaveEpisodeProceduralPathResult {
    verifyEpisodeProceduralPath(input.path);
    const namespaceId = input.namespaceId.trim();
    if (!namespaceId) throw new Error("procedural path persistence requires namespaceId");
    const episode = this.db.prepare(
      `SELECT id, user_id, session_id FROM episodes WHERE id = ?`
    ).get(input.path.episodeId) as EpisodeSourceRow | undefined;
    if (!episode) {
      throw new Error(`procedural path source episode not found: ${input.path.episodeId}`);
    }
    const provenance = resolveReconstructionProvenance(input);
    const activate = input.activate ?? true;
    const existing = this.getByHash(input.path.pathHash);
    if (existing) {
      assertSameStoredSource(existing, {
        episodeId: episode.id,
        userId: episode.user_id,
        sessionId: episode.session_id,
        namespaceId,
        reconstructionAlgorithmVersion: provenance.algorithmVersion,
        reconstructionModel: provenance.model
      });
      if (activate && existing.status !== "active") {
        this.activateVersion(existing.id, input.createdAt);
      }
      return {
        record: this.get(existing.id)!,
        occurrences: this.listOccurrencesForPath(existing.id),
        created: false
      };
    }

    const occurrenceInputs = input.path.spans.map((span) => buildOccurrenceInput({
      path: input.path,
      span,
      namespaceId,
      reconstructionAlgorithmVersion: provenance.algorithmVersion,
      reconstructionModel: provenance.model,
      createdAt: input.createdAt
    }));
    this.db.transaction(() => {
      if (activate) this.deactivateActivePath(input.path.episodeId, input.createdAt);
      this.db.prepare(
        `INSERT INTO episode_procedural_paths (
          id, episode_id, user_id, session_id, namespace_id, schema_version,
          reconstruction_algorithm_version, reconstruction_model,
          source_snapshot_hash, path_hash, terminal_reward, status, payload_json,
          created_at, activated_at, deactivated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        input.path.id,
        episode.id,
        episode.user_id,
        episode.session_id,
        namespaceId,
        input.path.schemaVersion,
        provenance.algorithmVersion,
        provenance.model ?? null,
        input.path.sourceSnapshotHash,
        input.path.pathHash,
        input.path.terminalReward ?? null,
        activate ? "active" : "inactive",
        toJson(input.path),
        input.createdAt,
        activate ? input.createdAt : null,
        null
      );
      const insertOccurrence = this.db.prepare(
        `INSERT INTO procedural_span_occurrences (
          id, path_id, path_hash, span_id, episode_id, namespace_id,
          schema_version, projection_version, span_index, local_goal,
          entry_condition, exit_condition, termination_status,
          raw_turn_ids_json, step_ids_json, pre_state_id, post_state_id,
          goal_text, condition_text, procedure_text, effect_text,
          structure_signature, span_json, reconstruction_algorithm_version,
          reconstruction_model, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      );
      for (const occurrence of occurrenceInputs) {
        insertOccurrence.run(
          occurrence.id,
          occurrence.pathId,
          occurrence.pathHash,
          occurrence.spanId,
          occurrence.episodeId,
          occurrence.namespaceId,
          occurrence.schemaVersion,
          occurrence.projection.version,
          occurrence.spanIndex,
          occurrence.localGoal,
          occurrence.entryCondition,
          occurrence.exitCondition,
          occurrence.terminationStatus,
          toJson(occurrence.rawTurnIds),
          toJson(occurrence.stepIds),
          occurrence.preStateId,
          occurrence.postStateId,
          occurrence.projection.goalText,
          occurrence.projection.conditionText,
          occurrence.projection.procedureText,
          occurrence.projection.effectText,
          occurrence.projection.structureSignature,
          toJson(occurrence.span),
          occurrence.reconstructionAlgorithmVersion,
          occurrence.reconstructionModel ?? null,
          occurrence.createdAt
        );
      }
    })();

    return {
      record: this.get(input.path.id)!,
      occurrences: this.listOccurrencesForPath(input.path.id),
      created: true
    };
  }

  get(id: string): EpisodeProceduralPathRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_procedural_paths WHERE id = ?`
    ).get(id) as EpisodeProceduralPathSqlRow | undefined;
    return row ? pathRecordFromSql(row) : undefined;
  }

  getByHash(pathHash: string): EpisodeProceduralPathRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_procedural_paths WHERE path_hash = ?`
    ).get(pathHash) as EpisodeProceduralPathSqlRow | undefined;
    return row ? pathRecordFromSql(row) : undefined;
  }

  getActiveForEpisode(episodeId: string): EpisodeProceduralPathRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM episode_procedural_paths
       WHERE episode_id = ? AND status = 'active'
       LIMIT 1`
    ).get(episodeId) as EpisodeProceduralPathSqlRow | undefined;
    return row ? pathRecordFromSql(row) : undefined;
  }

  listVersionsForEpisode(episodeId: string): EpisodeProceduralPathRecord[] {
    return (this.db.prepare(
      `SELECT * FROM episode_procedural_paths
       WHERE episode_id = ?
       ORDER BY created_at DESC, id DESC`
    ).all(episodeId) as EpisodeProceduralPathSqlRow[]).map(pathRecordFromSql);
  }

  activateVersion(pathId: string, at: string): EpisodeProceduralPathRecord {
    const target = this.get(pathId);
    if (!target) throw new Error(`procedural path version not found: ${pathId}`);
    if (target.status === "active") return target;
    this.db.transaction(() => {
      this.deactivateActivePath(target.episodeId, at);
      const result = this.db.prepare(
        `UPDATE episode_procedural_paths
         SET status = 'active', activated_at = ?, deactivated_at = NULL
         WHERE id = ?`
      ).run(at, pathId);
      if (result.changes !== 1) throw new Error(`failed to activate procedural path: ${pathId}`);
    })();
    return this.get(pathId)!;
  }

  getOccurrence(id: string): ProceduralSpanOccurrenceRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_span_occurrences WHERE id = ?`
    ).get(id) as ProceduralSpanOccurrenceSqlRow | undefined;
    return row ? occurrenceFromSql(row) : undefined;
  }

  listOccurrencesForPath(pathId: string): ProceduralSpanOccurrenceRecord[] {
    return (this.db.prepare(
      `SELECT * FROM procedural_span_occurrences
       WHERE path_id = ?
       ORDER BY span_index ASC`
    ).all(pathId) as ProceduralSpanOccurrenceSqlRow[]).map(occurrenceFromSql);
  }

  listActiveOccurrencesForNamespace(
    namespaceId: string,
    limit?: number
  ): ProceduralSpanOccurrenceRecord[] {
    const safeLimit = limit === undefined
      ? undefined
      : Math.max(1, Math.min(10_000, Math.trunc(limit)));
    const sql =
      `SELECT occurrences.*
       FROM procedural_span_occurrences AS occurrences
       JOIN episode_procedural_paths AS paths ON paths.id = occurrences.path_id
       WHERE occurrences.namespace_id = ?
         AND paths.status = 'active'
       ORDER BY occurrences.created_at ASC, occurrences.episode_id ASC, occurrences.span_index ASC` +
      (safeLimit === undefined ? "" : " LIMIT ?");
    const rows = safeLimit === undefined
      ? this.db.prepare(sql).all(namespaceId)
      : this.db.prepare(sql).all(namespaceId, safeLimit);
    return (rows as ProceduralSpanOccurrenceSqlRow[]).map(occurrenceFromSql);
  }

  private deactivateActivePath(episodeId: string, at: string): void {
    this.db.prepare(
      `UPDATE episode_policy_projections
       SET status = 'inactive', deactivated_at = ?
       WHERE episode_id = ? AND status = 'active'`
    ).run(at, episodeId);
    const affectedClusters = (this.db.prepare(
      `SELECT DISTINCT members.cluster_id AS id
       FROM procedural_span_cluster_members AS members
       JOIN procedural_span_occurrences AS occurrences
         ON occurrences.id = members.occurrence_id
       JOIN episode_procedural_paths AS paths
         ON paths.id = occurrences.path_id
       WHERE paths.episode_id = ? AND paths.status = 'active'`
    ).all(episodeId) as Array<{ id: string }>).map((row) => row.id);
    for (const clusterId of affectedClusters) {
      this.invalidateClusterForPathChange(clusterId, at);
    }
    this.db.prepare(
      `UPDATE episode_procedural_paths
       SET status = 'inactive', deactivated_at = ?
       WHERE episode_id = ? AND status = 'active'`
    ).run(at, episodeId);
  }

  private invalidateClusterForPathChange(clusterId: string, at: string): void {
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
    this.db.prepare(
      `UPDATE procedural_span_clusters
       SET status = 'stale', active_policy_version_id = NULL, updated_at = ?
       WHERE id = ?`
    ).run(at, clusterId);
  }
}

function verifyEpisodeProceduralPath(path: EpisodeProceduralPathV2): void {
  if (path.schemaVersion !== EPISODE_PROCEDURAL_PATH_SCHEMA_VERSION) {
    throw new Error(`unsupported procedural path schema: ${String(path.schemaVersion)}`);
  }
  const rebuilt = buildEpisodeProceduralPath({
    episodeId: path.episodeId,
    states: path.states,
    steps: path.steps,
    spans: path.spans,
    segmentationDecisions: path.segmentationDecisions,
    sourceSnapshotHash: path.sourceSnapshotHash,
    ...(path.terminalReward === undefined ? {} : { terminalReward: path.terminalReward })
  });
  if (rebuilt.pathHash !== path.pathHash || rebuilt.id !== path.id) {
    throw new Error(`procedural path integrity check failed: ${path.id}`);
  }
  for (const step of path.steps) {
    if (step.episodeId !== path.episodeId || step.provenance.sourceSnapshotHash !== path.sourceSnapshotHash) {
      throw new Error(`procedural path contains step from another source snapshot: ${step.id}`);
    }
  }
  for (const span of path.spans) {
    if (span.schemaVersion !== PROCEDURAL_SPAN_SCHEMA_VERSION ||
        span.episodeId !== path.episodeId ||
        span.provenance.sourceSnapshotHash !== path.sourceSnapshotHash) {
      throw new Error(`procedural path contains invalid Span occurrence source: ${span.id}`);
    }
  }
}

function resolveReconstructionProvenance(input: SaveEpisodeProceduralPathInput): {
  algorithmVersion: string;
  model?: string;
} {
  const provenances = [
    ...input.path.steps.map((step) => step.provenance),
    ...input.path.spans.map((span) => span.provenance)
  ];
  const algorithmVersions = unique(provenances.map((item) => item.algorithmVersion));
  const models = unique(provenances.map((item) => item.model).filter(isString));
  if (algorithmVersions.length > 1) {
    throw new Error("procedural path mixes reconstruction algorithm versions");
  }
  if (models.length > 1) throw new Error("procedural path mixes reconstruction models");
  const algorithmVersion = input.reconstructionAlgorithmVersion ?? algorithmVersions[0];
  if (!algorithmVersion) {
    throw new Error("empty procedural path persistence requires reconstructionAlgorithmVersion");
  }
  if (algorithmVersions[0] && algorithmVersions[0] !== algorithmVersion) {
    throw new Error("procedural path reconstruction algorithm does not match embedded provenance");
  }
  const model = input.reconstructionModel ?? models[0];
  if (input.reconstructionModel && models[0] && input.reconstructionModel !== models[0]) {
    throw new Error("procedural path reconstruction model does not match embedded provenance");
  }
  return { algorithmVersion, ...(model ? { model } : {}) };
}

function buildOccurrenceInput(input: {
  path: EpisodeProceduralPathV2;
  span: ProceduralSpanV1;
  namespaceId: string;
  reconstructionAlgorithmVersion: string;
  reconstructionModel?: string;
  createdAt: string;
}): ProceduralSpanOccurrenceRecord {
  const projection = buildClusterProjection(input.path, input.span);
  return {
    id: `procedural_span_occurrence_${stableHash({
      schemaVersion: PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
      pathHash: input.path.pathHash,
      spanId: input.span.id
    }).slice(0, 20)}`,
    pathId: input.path.id,
    pathHash: input.path.pathHash,
    spanId: input.span.id,
    episodeId: input.path.episodeId,
    namespaceId: input.namespaceId,
    schemaVersion: PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
    spanIndex: input.span.spanIndex,
    localGoal: input.span.localGoal,
    capabilityGoal: input.span.capabilityGoal ?? input.span.localGoal,
    entryCondition: input.span.entryCondition,
    exitCondition: input.span.termination.exitCondition,
    terminationStatus: input.span.termination.status,
    rawTurnIds: [...input.span.rawTurnIds],
    stepIds: [...input.span.stepIds],
    preStateId: input.span.preStateId,
    postStateId: input.span.postStateId,
    projection,
    span: input.span,
    reconstructionAlgorithmVersion: input.reconstructionAlgorithmVersion,
    ...(input.reconstructionModel ? { reconstructionModel: input.reconstructionModel } : {}),
    createdAt: input.createdAt
  };
}

function buildClusterProjection(
  path: EpisodeProceduralPathV2,
  span: ProceduralSpanV1
): ProceduralSpanClusterProjectionV1 {
  const stateById = new Map(path.states.map((state) => [state.id, state]));
  const postState = stateById.get(span.postStateId);
  const stepById = new Map(path.steps.map((step) => [step.id, step]));
  const steps = spanSteps(path, span);
  return {
    version: PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSION,
    goalText: clip(span.capabilityGoal ?? span.localGoal, 4_000),
    conditionText: clip([
      `Entry: ${span.entryCondition}`,
      `Exit: ${span.termination.exitCondition}`
    ].filter(Boolean).join("\n"), 8_000),
    procedureText: proceduralSpanProcedureText(path, span),
    effectText: clip([
      `Termination: ${span.termination.status}`,
      `Exit: ${span.termination.exitCondition}`,
      postState?.summary ? `Post-state: ${postState.summary}` : ""
    ].filter(Boolean).join("\n"), 8_000),
    structureSignature: stableHash({
      version: PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSION,
      steps: steps.map((step) => structuralStep(step, stepById))
    })
  };
}

export function proceduralSpanProcedureText(
  path: EpisodeProceduralPathV2,
  span: ProceduralSpanV1
): string {
  const normalized = span.procedureSemantic?.trim();
  if (normalized) return clip(normalized, 24_000);
  const steps = spanSteps(path, span);
  const toolSteps = steps.filter((step) => step.action.kind === "tool_action");
  const semanticSteps = toolSteps.length > 0 ? toolSteps : steps;
  const intents = semanticSteps.map((step) => step.action.intent.trim())
    .filter(Boolean)
    .filter((intent, index, all) => index === 0 || intent !== all[index - 1]);
  return clip(intents
    .map((intent, index) => `${index + 1}. ${intent}`)
    .join("\n"), 24_000);
}

function spanSteps(path: EpisodeProceduralPathV2, span: ProceduralSpanV1): ExecutionStepV1[] {
  const stepById = new Map(path.steps.map((step) => [step.id, step]));
  return span.stepIds.map((stepId) => {
    const step = stepById.get(stepId);
    if (!step) throw new Error(`procedural Span references missing Step: ${span.id} -> ${stepId}`);
    return step;
  });
}

function structuralStep(
  step: ExecutionStepV1,
  stepById: ReadonlyMap<string, ExecutionStepV1>
): Record<string, unknown> {
  return {
    kind: step.action.kind,
    type: step.action.type,
    toolName: step.action.toolName,
    outcome: step.outcome.status,
    retryDistance: relationshipDistance(step, step.retryOfStepId, stepById),
    recoveryDistance: relationshipDistance(step, step.recoveryFromStepId, stepById),
    actionDeltaOps: step.actionEffectDelta.map((operation) => operation.op),
    observationDeltaOps: step.externalObservationDelta.map((operation) => operation.op)
  };
}

function relationshipDistance(
  step: ExecutionStepV1,
  relatedStepId: string | undefined,
  stepById: ReadonlyMap<string, ExecutionStepV1>
): number | undefined {
  if (!relatedStepId) return undefined;
  const related = stepById.get(relatedStepId);
  return related ? step.stepIndex - related.stepIndex : undefined;
}

function pathRecordFromSql(row: EpisodeProceduralPathSqlRow): EpisodeProceduralPathRecord {
  const path = parseJson<EpisodeProceduralPathV2 | null>(row.payload_json, null);
  if (!path) throw new Error(`stored procedural path payload is invalid: ${row.id}`);
  verifyEpisodeProceduralPath(path);
  if (path.id !== row.id || path.episodeId !== row.episode_id || path.pathHash !== row.path_hash ||
      path.sourceSnapshotHash !== row.source_snapshot_hash || path.schemaVersion !== row.schema_version) {
    throw new Error(`stored procedural path metadata mismatch: ${row.id}`);
  }
  return {
    id: row.id,
    episodeId: row.episode_id,
    userId: row.user_id,
    sessionId: row.session_id,
    namespaceId: row.namespace_id,
    schemaVersion: path.schemaVersion,
    reconstructionAlgorithmVersion: row.reconstruction_algorithm_version,
    ...(row.reconstruction_model ? { reconstructionModel: row.reconstruction_model } : {}),
    sourceSnapshotHash: row.source_snapshot_hash,
    pathHash: row.path_hash,
    ...(row.terminal_reward === null ? {} : { terminalReward: row.terminal_reward }),
    status: row.status,
    path,
    createdAt: row.created_at,
    activatedAt: row.activated_at,
    deactivatedAt: row.deactivated_at
  };
}

function occurrenceFromSql(row: ProceduralSpanOccurrenceSqlRow): ProceduralSpanOccurrenceRecord {
  const span = parseJson<ProceduralSpanV1 | null>(row.span_json, null);
  if (!span || span.id !== row.span_id || span.spanIndex !== row.span_index ||
      span.schemaVersion !== PROCEDURAL_SPAN_SCHEMA_VERSION) {
    throw new Error(`stored procedural Span occurrence payload is invalid: ${row.id}`);
  }
  if (row.schema_version !== PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION ||
      !isSupportedProjectionVersion(row.projection_version)) {
    throw new Error(`unsupported procedural Span occurrence schema: ${row.id}`);
  }
  return {
    id: row.id,
    pathId: row.path_id,
    pathHash: row.path_hash,
    spanId: row.span_id,
    episodeId: row.episode_id,
    namespaceId: row.namespace_id,
    schemaVersion: PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
    spanIndex: row.span_index,
    localGoal: row.local_goal,
    capabilityGoal: row.goal_text,
    entryCondition: row.entry_condition,
    exitCondition: row.exit_condition,
    terminationStatus: row.termination_status,
    rawTurnIds: parseStringArray(row.raw_turn_ids_json),
    stepIds: parseStringArray(row.step_ids_json),
    preStateId: row.pre_state_id,
    postStateId: row.post_state_id,
    projection: {
      version: row.projection_version,
      goalText: row.goal_text,
      conditionText: row.condition_text,
      procedureText: row.procedure_text,
      effectText: row.effect_text,
      structureSignature: row.structure_signature
    },
    span,
    reconstructionAlgorithmVersion: row.reconstruction_algorithm_version,
    ...(row.reconstruction_model ? { reconstructionModel: row.reconstruction_model } : {}),
    createdAt: row.created_at
  };
}

function isSupportedProjectionVersion(value: string): value is ProceduralSpanClusterProjectionVersion {
  return value === PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSION ||
    LEGACY_PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSIONS.includes(
      value as typeof LEGACY_PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSIONS[number]
    );
}

function assertSameStoredSource(
  existing: EpisodeProceduralPathRecord,
  expected: {
    episodeId: string;
    userId: string;
    sessionId: string;
    namespaceId: string;
    reconstructionAlgorithmVersion: string;
    reconstructionModel?: string;
  }
): void {
  if (existing.episodeId !== expected.episodeId ||
      existing.userId !== expected.userId ||
      existing.sessionId !== expected.sessionId ||
      existing.namespaceId !== expected.namespaceId ||
      existing.reconstructionAlgorithmVersion !== expected.reconstructionAlgorithmVersion ||
      existing.reconstructionModel !== expected.reconstructionModel) {
    throw new Error(`procedural path hash already exists with different source metadata: ${existing.pathHash}`);
  }
}

function parseStringArray(value: string): string[] {
  return parseJson<unknown[]>(value, []).filter(isString);
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function isString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}
