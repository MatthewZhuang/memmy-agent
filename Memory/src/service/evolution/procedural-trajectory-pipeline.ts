import { skillMetaFromMemory } from "../../algorithm/plugin-algorithms.js";
import type { MemmyConfig } from "../../config/index.js";
import type { Embedder, LlmClient } from "../../model/types.js";
import type {
  EvolutionJobRecord,
  RawTurnRecord,
  Repositories
} from "../../storage/repositories.js";
import { kindFromMemory } from "../../storage/repositories.js";
import {
  PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION,
  ProceduralTrajectoryCasError,
  type EpisodeExecutionPathRecord,
  type TrajectoryWindowFamilyRecord,
  type TrajectoryWindowFamilyRevisionRecord,
  type TrajectoryWindowClusterRecord,
  type TrajectoryWindowClusterVersionRecord,
  type TrajectoryWindowOccurrenceRecord
} from "../../storage/procedural-trajectory-repository.js";
import type { MemoryRow } from "../../types.js";
import { stableHash } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import {
  EPISODE_PATH_COMPILER_VERSION,
  EpisodePathCompiler
} from "./episode-path-compiler.js";
import {
  PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION,
  ProceduralPatternSkillMaterializer,
  proceduralSkillKey,
  type ProceduralPatternSkillInput,
  type ProceduralSkillAlignedSequenceStep,
  type ProceduralSkillEvidenceOccurrence,
  type ProceduralSkillEvidenceStep,
  type ProceduralSkillExpansionStep
} from "./procedural-pattern-skill.js";
import {
  ANCHORED_COMPLETION_SCHEMA_VERSION,
  PROCEDURAL_WINDOW_MECHANICAL_VERSION,
  PROCEDURAL_WINDOW_MINING_VERSION,
  buildExclusiveFineClusters,
  buildTrajectoryWindows,
  extractAnchoredCompletionOverlay,
  extractAlignedCommonCore,
  fineEvidenceSignature,
  selectConstrainedRealMedoid,
  selectMaximalWindowClusters,
  unitVector,
  type AlignedCommonCoreV1,
  type AnchoredCompletionOccurrenceCandidateV2,
  type EmbeddedTrajectoryWindowV1,
  type EpisodeExecutionPathLiteV1,
  type ExecutionStepLiteV1,
  type MultiScaleWindowSpec,
  type TrajectoryWindowClusterMemberV1,
  type TrajectoryWindowClusterV1,
  type TrajectoryWindowOccurrenceV1
} from "./procedural-window-model.js";
import {
  bandedMonotonicMatch,
  cosineSimilarity,
  selfBandedMonotonicMatch,
  type BandedMonotonicMatchConfig,
  type BandedMonotonicMatchResultV1
} from "./trajectory-window-alignment.js";

const STEP_INTENT_REPRESENTATION_VERSION =
  PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION;
const WINDOW_INTENT_SEQUENCE_REPRESENTATION_VERSION =
  "trajectory-window-intent-sequence.v1";

type TraceMeta = NonNullable<ReturnType<ProceduralTrajectoryPipelineDeps["traceMeta"]>>;

export interface ProceduralTrajectoryPipelineDeps {
  repos: Repositories;
  config: MemmyConfig;
  llm: LlmClient;
  skillLlm: LlmClient;
  embedder: Embedder;
  traceMeta(memory: MemoryRow | null | undefined): {
    id: string;
    userId: string;
    sessionId?: string;
    episodeId?: string;
    rawTurnId?: string;
    value: number;
    priority: number;
    memory: MemoryRow;
  } | null;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  upsertEvolutionMemory(memory: MemoryRow): {
    memory: MemoryRow;
    created: boolean;
    previous?: MemoryRow;
  };
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
  namespaceIdFromMemory(memory: MemoryRow): string | undefined;
}

interface PipelineConfig {
  algorithmVersion: string;
  /**
   * Immutable identity of the mechanical Window projection. Clustering knobs
   * must not change this hash or force duplicate Window rows/embeddings.
   */
  mechanicalWindowHash: string;
  /** Version of medoid admission, alignment, and support policy. */
  clusteringConfigHash: string;
  specs: MultiScaleWindowSpec[];
  coarseThresholdByScale: Readonly<Record<number, number>>;
  fineConfigByScale: Map<number, BandedMonotonicMatchConfig>;
  minSupportEpisodes: number;
  medoidSwitchMargin: number;
  completionExpansionEnabled: boolean;
  maxPrefixExpansionSteps: number;
  maxSuffixExpansionSteps: number;
  minExtensionStepSimilarity: number;
  skillCompletionConfigHash: string;
}

interface ClusterSnapshot {
  head: TrajectoryWindowClusterRecord;
  version: TrajectoryWindowClusterVersionRecord;
  domain: TrajectoryWindowClusterV1;
  core?: AlignedCommonCoreV1;
}

interface MemberCandidate {
  embedded: EmbeddedTrajectoryWindowV1;
  rewardHash: string;
}

interface FamilySnapshot {
  head: TrajectoryWindowFamilyRecord;
  revision: TrajectoryWindowFamilyRevisionRecord;
  members: EmbeddedTrajectoryWindowV1[];
}

/**
 * Durable orchestration for the Episode-level procedural path. Window and
 * cluster state remains internal; only an admitted common core may materialize
 * a public upstream-compatible Skill.
 */
export class ProceduralTrajectoryPipeline {
  private readonly pathCompiler: EpisodePathCompiler;
  private readonly skillMaterializer: ProceduralPatternSkillMaterializer;

  constructor(private readonly deps: ProceduralTrajectoryPipelineDeps) {
    const owner = this;
    this.pathCompiler = new EpisodePathCompiler({
      get config() { return owner.deps.config; },
      // Step semantics are an evolution-time procedural abstraction. Keep the
      // model route aligned with the v15 reconstruction pipeline instead of
      // silently falling back to the memory-summary model.
      get llm() { return owner.deps.skillLlm; }
    });
    this.skillMaterializer = new ProceduralPatternSkillMaterializer({
      repos: deps.repos,
      get config() { return owner.deps.config; },
      get skillLlm() { return owner.deps.skillLlm; },
      traceMeta: deps.traceMeta,
      buildMemory: deps.buildMemory
    });
  }

  /**
   * Remove an Episode's current procedural evidence from every public Skill
   * immediately. RawTurn/L1 governance may then enqueue a new compilation from
   * the still-authorized sources; Episode deletion simply leaves no active Path.
   */
  invalidateEpisodeSources(input: {
    episodeId: string;
    reason: string;
    at: string;
    recompile?: boolean;
  }): void {
    const activePath = this.deps.repos.proceduralTrajectory.getActivePath(input.episodeId);
    const episode = this.deps.repos.runtime.getEpisode(input.episodeId);
    // With no active Path there is no published procedural evidence to
    // invalidate. The normal reward pipeline remains responsible for first
    // compilation; avoiding a synthetic compile here also keeps bulk Episode
    // deletion from enqueueing work after the Episode disappears.
    if (!activePath) return;

    const affectedClusterIds = new Set(
      this.deps.repos.proceduralTrajectory.listAffectedClusterIdsForPath(activePath.id)
    );
    this.deps.repos.proceduralTrajectory.deactivatePathVersion(activePath.id, input.at);
    // Panel governance is synchronous. Make stale procedural knowledge
    // unretrievable immediately, then let the durable compile/ingest jobs
    // rebuild cluster membership from the still-authorized evidence. Do not
    // run embeddings or cluster induction inside a panel mutation.
    for (const clusterId of affectedClusterIds) {
      const head = this.deps.repos.proceduralTrajectory.getClusterHead(clusterId);
      if (head?.activeSkillMemoryId) {
        this.archiveProceduralSkillMemory(
          head.activeSkillMemoryId,
          `source-governance-invalidated:${input.reason}`,
          input.at
        );
      }
    }
    if (input.recompile !== false && episode?.status === "closed") {
      this.enqueueGovernedClusterReconcile(
        activePath,
        input.reason,
        input.at
      );
      this.enqueueGovernedPathCompile(
        episode.id,
        input.reason,
        input.at,
        activePath.id
      );
    }
  }

  /** Persist a panel archive/delete as an explicit no-resurrection tombstone. */
  markProceduralSkillGovernanceDisabled(
    memory: MemoryRow,
    action: "archive" | "delete",
    at: string
  ): MemoryRow {
    const current = this.deps.repos.memories.getIncludingDeleted(memory.id) ?? memory;
    const internal = current.properties.internal_info;
    if (current.memoryLayer !== "Skill" ||
        internal.plugin_algorithm !== "procedural.pattern.skill.v1") {
      return current;
    }
    const skill = isRecord(internal.skill) ? internal.skill : {};
    return this.deps.repos.memories.update({
      ...current,
      properties: {
        ...current.properties,
        internal_info: {
          ...internal,
          procedural_governance: {
            disabled: true,
            action,
            source: "panel",
            at
          },
          skill: {
            ...skill,
            governance_disabled: true,
            governance_action: action,
            governance_disabled_at: at
          }
        }
      },
      updatedAt: at
    });
  }

  async compileEpisodePath(job: EvolutionJobRecord): Promise<void> {
    if (!this.deps.config.algorithm.proceduralWindow.enabled) return;
    const episodeId = job.episodeId ?? payloadString(job, "episodeId");
    if (!episodeId) throw new Error("episode_path_compile requires episodeId");
    const episode = this.deps.repos.runtime.getEpisode(episodeId);
    if (!episode || episode.userId !== job.userId) {
      throw new Error(`episode_path_compile source Episode not found in user scope: ${episodeId}`);
    }
    if (episode.status !== "closed") return;
    const rawTurns = this.deps.repos.runtime.listRawTurnsByEpisode(episode.id, 10_000);
    const sourceState = this.currentSourceState(episode.id, rawTurns);
    const sourceSnapshotHash = sourceState.sourceSnapshotHash;
    const currentModelSignature = this.deps.skillLlm.isConfigured()
      ? `${this.deps.skillLlm.config.provider}:${this.deps.skillLlm.config.model ?? "unknown"}`
      : "deterministic-fallback";
    const active = this.deps.repos.proceduralTrajectory.getActivePath(episode.id);
    const expectedActivePathId = active?.id ?? null;
    if (sourceState.rawTurns.length === 0) {
      if (active) {
        this.invalidateEpisodeSources({
          episodeId: episode.id,
          reason: "no-active-procedural-source",
          at: new Date().toISOString(),
          recompile: false
        });
      }
      return;
    }
    if (active && sourceSnapshotHash && active.sourceSnapshotHash === sourceSnapshotHash &&
        active.compilerVersion === EPISODE_PATH_COMPILER_VERSION &&
        active.modelSignature === currentModelSignature) {
      this.enqueueWindowIngest(job, active, undefined);
      return;
    }
    const path = await this.pathCompiler.compile({
      episode,
      rawTurns: sourceState.rawTurns,
      ...(sourceSnapshotHash ? { sourceSnapshotHash } : {})
    });
    if (!path) return;

    const at = new Date().toISOString();
    let saved;
    try {
      saved = this.deps.repos.proceduralTrajectory.savePathVersion({
        path,
        expectedActivePathId,
        createdAt: at,
        activate: true
      });
    } catch (error) {
      // A newer source snapshot completed while this job was waiting on the
      // Step-semantics LLM. The newer active Path wins; retrying this stale
      // job would only repeat obsolete work and could otherwise roll it back.
      if (error instanceof ProceduralTrajectoryCasError) return;
      throw error;
    }
    const governedPreviousPathId = payloadString(job, "previousPathId");
    const governedPreviousPath = governedPreviousPathId &&
      governedPreviousPathId !== saved.record.id
      ? this.deps.repos.proceduralTrajectory.getPath(governedPreviousPathId)
      : undefined;
    this.enqueueWindowIngest(
      job,
      saved.record,
      saved.previousActive ?? governedPreviousPath
    );
  }

  async ingestTrajectoryWindows(job: EvolutionJobRecord): Promise<void> {
    if (!this.deps.config.algorithm.proceduralWindow.enabled) return;
    const pathId = payloadString(job, "pathId") ?? job.targetMemoryId;
    if (!pathId) throw new Error("trajectory_window_ingest requires pathId");
    const pathRecord = this.deps.repos.proceduralTrajectory.getPath(pathId);
    const reconcileInactivePath = job.payload.reconcileInactivePath === true;
    if (!pathRecord && reconcileInactivePath) return;
    if (!pathRecord || pathRecord.userId !== job.userId) {
      throw new Error(`trajectory_window_ingest Path not found in user scope: ${pathId}`);
    }
    if (reconcileInactivePath) {
      const config = this.pipelineConfig();
      const at = new Date().toISOString();
      const affectedFamilyIds = new Set(
        this.deps.repos.proceduralTrajectory.listAffectedFamilyIdsForPath(pathRecord.id)
      );
      const affectedClusterIds = new Set(
        [...affectedFamilyIds].flatMap((familyId) => this.familyLinkedClusterIds(familyId))
      );
      const orphanedMembers: EmbeddedTrajectoryWindowV1[] = [];
      for (const familyId of [...affectedFamilyIds].sort()) {
        const reconciled = this.reconcileFamilyAfterPathChange(
          familyId,
          config,
          at
        );
        orphanedMembers.push(...reconciled.orphaned);
        for (const clusterId of reconciled.affectedClusterIds) {
          affectedClusterIds.add(clusterId);
        }
      }
      for (const embedded of orderedWindowsForIngestion(orphanedMembers)) {
        for (const familyId of this.ingestOneWindowIntoFamilies(embedded, config, at)) {
          affectedFamilyIds.add(familyId);
        }
      }
      for (const familyId of [...affectedFamilyIds].sort()) {
        for (const clusterId of this.rebuildFamilyFineClusters(familyId, config, at)) {
          affectedClusterIds.add(clusterId);
        }
      }
      this.retireUnlinkedCanonicalClusters(affectedClusterIds, at);
      this.reconcileSkillEligibility(pathRecord.userId, affectedClusterIds, config, at);
      return;
    }
    // A newer reward/source version won before this durable job ran.
    if (pathRecord.status !== "active" ||
        this.deps.repos.proceduralTrajectory.getActivePath(pathRecord.episodeId)?.id !== pathId) {
      return;
    }
    const currentEpisode = this.deps.repos.runtime.getEpisode(pathRecord.episodeId);
    if (!currentEpisode || currentEpisode.userId !== pathRecord.userId) return;
    const config = this.pipelineConfig();
    // windowConfigVersion is read only as a compatibility alias for jobs
    // queued before mechanical/clustering identity was separated.
    const requestedMechanicalHash = payloadString(job, "mechanicalWindowHash") ??
      payloadString(job, "windowConfigVersion");
    const requestedClusteringHash = payloadString(job, "clusteringConfigHash");
    if ((requestedMechanicalHash &&
          requestedMechanicalHash !== config.mechanicalWindowHash) ||
        (requestedClusteringHash &&
          requestedClusteringHash !== config.clusteringConfigHash)) {
      this.deps.enqueueJob({
        jobType: "trajectory_window_ingest",
        userId: pathRecord.userId,
        sessionId: currentEpisode.sessionId,
        episodeId: pathRecord.episodeId,
        payload: {
          reason: "window_config.changed",
          pathId: pathRecord.id,
          pathHash: pathRecord.pathHash,
          pathSourceSnapshotHash: pathRecord.sourceSnapshotHash,
          sourceSnapshotHash: pathRecord.sourceSnapshotHash,
          rewardSnapshotHash: rewardHashForEpisode(currentEpisode),
          mechanicalWindowHash: config.mechanicalWindowHash,
          clusteringConfigHash: config.clusteringConfigHash,
          // Compatibility for already deployed workers reading the old field.
          windowConfigVersion: config.mechanicalWindowHash,
          ...(payloadString(job, "previousPathId")
            ? { previousPathId: payloadString(job, "previousPathId") }
            : {})
        }
      });
      return;
    }

    const at = new Date().toISOString();
    const affectedFamilyIds = new Set<string>();
    const affectedClusterIds = new Set<string>();
    const orphanedMembers: EmbeddedTrajectoryWindowV1[] = [];
    const previousPathId = payloadString(job, "previousPathId");
    if (previousPathId && previousPathId !== pathRecord.id) {
      for (const familyId of this.deps.repos.proceduralTrajectory
        .listAffectedFamilyIdsForPath(previousPathId)) {
        affectedFamilyIds.add(familyId);
        const reconciled = this.reconcileFamilyAfterPathChange(
          familyId,
          config,
          at
        );
        orphanedMembers.push(...reconciled.orphaned);
        for (const clusterId of reconciled.affectedClusterIds) {
          affectedClusterIds.add(clusterId);
        }
      }
    }

    const stepVectorById = await this.ensureStepEmbeddings(pathRecord, at);
    const rewardCurrentPath: EpisodeExecutionPathLiteV1 = {
      ...pathRecord.path,
      ...(currentEpisode.rTask === undefined
        ? { terminalReward: undefined }
        : { terminalReward: currentEpisode.rTask })
    };
    const occurrences = buildTrajectoryWindows([rewardCurrentPath], config.specs);
    const embeddedNew = await this.persistAndEmbedWindows(
      occurrences,
      stepVectorById,
      config,
      at
    );

    // Reward drift updates existing immutable Window identities in place. All
    // active Families containing them must receive a new evidence revision,
    // even when their semantic membership is unchanged.
    for (const familyId of this.deps.repos.proceduralTrajectory
      .listAffectedFamilyIdsForOccurrences(embeddedNew.map((item) => item.occurrence.id))) {
      affectedFamilyIds.add(familyId);
    }

    // Stable local reflow: support evidence seeds Families before read-only
    // counterexamples/unknowns are admitted; within each role, execution order
    // is independent of async embedding completion order.
    for (const embedded of orderedWindowsForIngestion([
      ...orphanedMembers,
      ...embeddedNew
    ])) {
      for (const familyId of this.ingestOneWindowIntoFamilies(embedded, config, at)) {
        affectedFamilyIds.add(familyId);
      }
    }
    for (const familyId of [...affectedFamilyIds].sort()) {
      for (const clusterId of this.rebuildFamilyFineClusters(familyId, config, at)) {
        affectedClusterIds.add(clusterId);
      }
    }

    this.retireUnlinkedCanonicalClusters(affectedClusterIds, at);
    this.reconcileSkillEligibility(pathRecord.userId, affectedClusterIds, config, at);
  }

  async induceProceduralSkill(job: EvolutionJobRecord): Promise<void> {
    if (!this.deps.config.algorithm.proceduralWindow.enabled) return;
    const clusterId = payloadString(job, "clusterId") ?? job.targetMemoryId;
    const clusterVersionId = payloadString(job, "clusterVersionId");
    if (!clusterId || !clusterVersionId) {
      throw new Error("procedural_skill_induction requires clusterId and clusterVersionId");
    }
    const head = this.deps.repos.proceduralTrajectory.getClusterHead(clusterId);
    if (!head || head.userId !== job.userId || head.status !== "active" ||
        head.activeVersionId !== clusterVersionId) {
      return;
    }
    const config = this.pipelineConfig();
    const inductionVersion = this.skillInductionVersion(config);
    if (head.algorithmVersion !== config.algorithmVersion ||
        head.configHash !== config.clusteringConfigHash) {
      return;
    }
    const at = new Date().toISOString();
    if (this.proceduralSkillGovernanceDisabled(head)) {
      const version = head.activeVersionId
        ? this.deps.repos.proceduralTrajectory.getClusterVersion(head.activeVersionId)
        : undefined;
      await this.rejectAndDowngradeSkill(head, version, "governance-disabled", at);
      return;
    }
    const snapshot = this.loadClusterSnapshot(head, config);
    if (!snapshot || !snapshot.core ||
        snapshot.version.supportEpisodeCount < config.minSupportEpisodes) {
      await this.rejectAndDowngradeSkill(head, snapshot?.version, "insufficient-common-core", at);
      return;
    }
    const maximal = this.maximalQualifiedClusters(head.userId, config);
    const selected = maximal.find((item) => item.domain.id === head.id);
    if (!selected || selected.domain.suppressedByClusterId) {
      await this.rejectAndDowngradeSkill(
        head,
        snapshot.version,
        selected?.domain.suppressedByClusterId
          ? `maximal-suppressed-by:${selected.domain.suppressedByClusterId}`
          : "not-maximal-qualified",
        at
      );
      return;
    }

    const input = this.buildSkillInput(snapshot, config);
    if (!input) {
      await this.rejectAndDowngradeSkill(head, snapshot.version, "invalid-positive-evidence", at);
      return;
    }
    const result = await this.skillMaterializer.compile(input);
    if (!result.admitted) {
      await this.rejectAndDowngradeSkill(head, snapshot.version, result.reason, at);
      return;
    }
    // The user may archive/delete the Skill while a slow Skill LLM call is in
    // flight. Recheck the durable tombstone before entering the write txn.
    if (this.proceduralSkillGovernanceDisabled(head)) {
      await this.rejectAndDowngradeSkill(head, snapshot.version, "governance-disabled", at);
      return;
    }

    const committedAt = new Date().toISOString();
    this.deps.repos.transaction(() => {
      const current = this.deps.repos.proceduralTrajectory.getClusterHead(head.id);
      // A slow model response from an older immutable membership version is a
      // successful no-op. Crucially, this check is in the same write
      // transaction as the public Memory upsert and trajectory head advance.
      if (!current || current.status !== "active" ||
          current.activeVersionId !== snapshot.version.id) {
        return;
      }
      const materialized = this.skillMaterializer.materializeDraft(
        result.draft,
        committedAt
      );
      const upsert = this.deps.upsertEvolutionMemory(materialized.memory);
      this.deps.repos.proceduralTrajectory.saveSkillVersion({
        clusterId: head.id,
        clusterVersionId: snapshot.version.id,
        expectedActiveSkillVersionId: current.activeSkillVersionId ?? null,
        skillKey: proceduralSkillKey(head.userId, head.id),
        skillMemoryId: upsert.memory.id,
        payload: {
          admitted: true,
          inductionVersion,
          commonCoreId: snapshot.core!.id,
          patternHash: input.patternHash,
          supportEpisodeIds: input.supportEpisodeIds,
          sourceSpanOccurrenceIds: input.sourceSpanOccurrenceIds,
          skillMemoryId: upsert.memory.id
        },
        contentHash: materialized.contentHash,
        createdAt: committedAt
      });
      for (const episodeId of materialized.sourceEpisodeIds) {
        this.deps.repos.runtime.appendEpisodeDerivedMemory(
          episodeId,
          "Skill",
          upsert.memory.id,
          committedAt
        );
      }
      this.deps.repos.runtime.appendChange({
        memoryId: upsert.memory.id,
        namespaceId: this.deps.namespaceIdFromMemory(upsert.memory),
        kind: kindFromMemory(upsert.memory),
        op: upsert.created ? "created" : "updated",
        entityId: upsert.memory.id,
        userId: input.userId,
        changeType: upsert.created ? "create" : "update",
        before: upsert.previous,
        after: upsert.memory,
        source: "worker.procedural_skill_induction.v1",
        createdAt: committedAt
      });
      if (this.deps.config.algorithm.capture.embedAfterCapture) {
        this.deps.enqueueJob({
          jobType: "embedding",
          userId: input.userId,
          sessionId: materialized.scope.sessionId ?? job.sessionId,
          episodeId: job.episodeId,
          targetMemoryId: upsert.memory.id,
          payload: {
            reason: "procedural.pattern.skill.upserted",
            contentHash: upsert.memory.contentHash
          },
          createdAt: committedAt
        });
      }
    });
  }

  private pipelineConfig(): PipelineConfig {
    const procedural = this.deps.config.algorithm.proceduralWindow;
    const scales = [...procedural.scales]
      .sort((left, right) => left.length - right.length)
      .map((scale) => ({ ...scale }));
    const mechanicalWindowHash = stableHash({
      // Preserve the v1 basis exactly: the v2 Family miner is a clustering
      // upgrade, not a new mechanical Window projection.
      miningVersion: PROCEDURAL_WINDOW_MECHANICAL_VERSION,
      windowSpecs: scales.map(({ length, stride }) => ({ length, stride })),
      stepRepresentationVersion: STEP_INTENT_REPRESENTATION_VERSION,
      coarseRepresentationVersion: WINDOW_INTENT_SEQUENCE_REPRESENTATION_VERSION
    });
    const minSupportEpisodes = Math.max(
      2,
      Math.floor(procedural.minSupportEpisodes)
    );
    const clusteringConfigHash = stableHash({
      algorithmVersion: PROCEDURAL_WINDOW_MINING_VERSION,
      // Cluster membership is meaningful only over one mechanical Window
      // basis. Threshold-only changes still preserve and reuse the Window rows.
      mechanicalWindowHash,
      scales: scales.map((scale) => ({
        length: scale.length,
        coarseSimilarityThreshold: scale.coarseSimilarityThreshold,
        bandWidth: scale.bandWidth,
        minStepSimilarity: scale.minStepSimilarity,
        minMatchedSteps: scale.minMatchedSteps,
        minCoverage: scale.minCoverage,
        minAverageMatchSimilarity: scale.minAverageMatchSimilarity,
        maxInternalGap: scale.maxInternalGap,
        gapPenalty: scale.gapPenalty,
        minAlignmentScore: scale.minAlignmentScore
      })),
      medoidSwitchMargin: procedural.medoidSwitchMargin,
      minSupportEpisodes
    });
    const completionExpansionEnabled = procedural.completionExpansionEnabled;
    const maxPrefixExpansionSteps = Math.max(
      0,
      Math.floor(procedural.maxPrefixExpansionSteps)
    );
    const maxSuffixExpansionSteps = Math.max(
      0,
      Math.floor(procedural.maxSuffixExpansionSteps)
    );
    const minExtensionStepSimilarity = Math.min(
      1,
      Math.max(0.01, procedural.minExtensionStepSimilarity)
    );
    const skillCompletionConfigHash = stableHash({
      version: ANCHORED_COMPLETION_SCHEMA_VERSION,
      completionExpansionEnabled,
      maxPrefixExpansionSteps,
      maxSuffixExpansionSteps,
      minExtensionStepSimilarity,
      minSupportEpisodes
    });
    return {
      algorithmVersion: PROCEDURAL_WINDOW_MINING_VERSION,
      mechanicalWindowHash,
      clusteringConfigHash,
      specs: scales.map((scale) => ({ length: scale.length, stride: scale.stride })),
      coarseThresholdByScale: Object.fromEntries(scales.map((scale) => [
        scale.length,
        scale.coarseSimilarityThreshold
      ])),
      fineConfigByScale: new Map(scales.map((scale) => [scale.length, {
        scale: scale.length,
        bandWidth: scale.bandWidth,
        minStepSimilarity: scale.minStepSimilarity,
        minMatchedSteps: scale.minMatchedSteps,
        minCoverage: scale.minCoverage,
        minAverageMatchSimilarity: scale.minAverageMatchSimilarity,
        maxInternalGap: scale.maxInternalGap,
        gapPenalty: scale.gapPenalty,
        minAlignmentScore: scale.minAlignmentScore
      }])),
      minSupportEpisodes,
      medoidSwitchMargin: procedural.medoidSwitchMargin,
      completionExpansionEnabled,
      maxPrefixExpansionSteps,
      maxSuffixExpansionSteps,
      minExtensionStepSimilarity,
      skillCompletionConfigHash
    };
  }

  private skillInductionVersion(config: PipelineConfig): string {
    return `${PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION}:${config.skillCompletionConfigHash}`;
  }

  private enqueueWindowIngest(
    sourceJob: EvolutionJobRecord,
    path: EpisodeExecutionPathRecord,
    previousPath: EpisodeExecutionPathRecord | undefined
  ): void {
    const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
    if (!episode) return;
    const config = this.pipelineConfig();
    this.deps.enqueueJob({
      jobType: "trajectory_window_ingest",
      userId: path.userId,
      sessionId: episode.sessionId,
      episodeId: path.episodeId,
      payload: {
        reason: previousPath ? "episode_path.compiled" : "episode_path.reused",
        pathId: path.id,
        pathHash: path.pathHash,
        pathSourceSnapshotHash: path.sourceSnapshotHash,
        sourceSnapshotHash: payloadString(sourceJob, "sourceSnapshotHash") ??
          path.sourceSnapshotHash,
        rewardSnapshotHash: payloadString(sourceJob, "rewardSnapshotHash") ??
          rewardHashForEpisode(episode),
        mechanicalWindowHash: config.mechanicalWindowHash,
        clusteringConfigHash: config.clusteringConfigHash,
        // Compatibility for already deployed workers reading the old field.
        windowConfigVersion: config.mechanicalWindowHash,
        ...(previousPath && previousPath.id !== path.id
          ? { previousPathId: previousPath.id }
          : {})
      },
      createdAt: new Date().toISOString()
    });
  }

  private embeddingSignature(): string {
    const status = this.deps.embedder.status();
    return `embedding:${stableHash({
      provider: status.provider,
      model: status.model ?? this.deps.embedder.config.model ?? "unknown",
      mode: this.deps.embedder.config.mode,
      sourceProvider: this.deps.embedder.config.sourceProvider ?? null,
      normalize: this.deps.embedder.config.normalize
    }).slice(0, 24)}`;
  }

  private async ensureStepEmbeddings(
    path: EpisodeExecutionPathRecord,
    at: string
  ): Promise<Map<string, number[]>> {
    const signature = this.embeddingSignature();
    const existing = this.deps.repos.proceduralTrajectory.listStepEmbeddings({
      pathId: path.id,
      representationVersion: STEP_INTENT_REPRESENTATION_VERSION,
      embeddingSignature: signature
    });
    const vectorByStepId = new Map(existing.map((item) => [item.stepId, item.vector]));
    const missing = path.path.steps.filter((step) => !vectorByStepId.has(step.id));
    if (missing.length > 0) {
      const vectors = await this.deps.embedder.embed(
        missing.map((step) => step.intent),
        "document"
      );
      if (vectors.length !== missing.length) {
        throw new Error("Step embedding response count does not match requested Steps");
      }
      for (const [index, step] of missing.entries()) {
        const vector = unitVector(vectors[index] ?? []);
        const saved = this.deps.repos.proceduralTrajectory.upsertStepEmbedding({
          pathId: path.id,
          stepId: step.id,
          stepIndex: step.stepIndex,
          representationVersion: STEP_INTENT_REPRESENTATION_VERSION,
          embeddingSignature: signature,
          semanticHash: stableHash(step.intent),
          vector,
          createdAt: at
        });
        vectorByStepId.set(step.id, saved.record.vector);
      }
    }
    return vectorByStepId;
  }

  private async persistAndEmbedWindows(
    occurrences: TrajectoryWindowOccurrenceV1[],
    stepVectorById: Map<string, number[]>,
    config: PipelineConfig,
    at: string
  ): Promise<EmbeddedTrajectoryWindowV1[]> {
    const signature = this.embeddingSignature();
    const missing = occurrences.filter((occurrence) =>
      !this.deps.repos.proceduralTrajectory.getWindow(occurrence.id));
    const vectors = missing.length > 0
      ? await this.deps.embedder.embed(missing.map((item) => item.semanticText), "document")
      : [];
    if (vectors.length !== missing.length) {
      throw new Error("Window embedding response count does not match requested windows");
    }
    const newVectorByOccurrenceId = new Map(missing.map((occurrence, index) => [
      occurrence.id,
      unitVector(vectors[index] ?? [])
    ]));
    for (const occurrence of occurrences) {
      const existing = this.deps.repos.proceduralTrajectory.getWindow(occurrence.id);
      const coarseVector = existing?.coarseVector ??
        newVectorByOccurrenceId.get(occurrence.id) ?? [];
      this.deps.repos.proceduralTrajectory.insertWindow({
        occurrence,
        windowConfigHash: config.mechanicalWindowHash,
        coarseRepresentationVersion: WINDOW_INTENT_SEQUENCE_REPRESENTATION_VERSION,
        embeddingSignature: signature,
        coarseVector,
        createdAt: at
      });
    }
    return occurrences.map((occurrence) => {
      const record = this.deps.repos.proceduralTrajectory.getWindow(occurrence.id);
      if (!record) throw new Error(`persisted trajectory window not found: ${occurrence.id}`);
      const stepVectors = occurrence.steps.map((step) => {
        const vector = stepVectorById.get(step.id);
        if (!vector) throw new Error(`Step vector missing for window: ${step.id}`);
        return vector;
      });
      return { occurrence, coarseVector: record.coarseVector, stepVectors };
    });
  }

  /**
   * v15 coarse recall: one Window joins every threshold-valid Family medoid.
   * Fine admission is intentionally deferred until all affected Families have
   * been revised, so one job performs one stable Family-local reflow.
   */
  private ingestOneWindowIntoFamilies(
    window: EmbeddedTrajectoryWindowV1,
    config: PipelineConfig,
    at: string
  ): string[] {
    const scale = window.occurrence.scale;
    const threshold = config.coarseThresholdByScale[scale];
    const fineConfig = config.fineConfigByScale.get(scale);
    if (threshold === undefined || !fineConfig) {
      throw new Error(`procedural window config is missing Span-${scale}`);
    }
    const medoids = this.deps.repos.proceduralTrajectory.listActiveFamilyMedoids({
      userId: window.occurrence.userId,
      scale,
      algorithmVersion: config.algorithmVersion,
      configHash: config.clusteringConfigHash,
      embeddingSignature: this.embeddingSignature()
    });
    const matchedFamilyIds: string[] = [];
    for (const medoid of medoids) {
      const similarity = cosineSimilarity(window.coarseVector, medoid.occurrence.coarseVector);
      if (similarity + Number.EPSILON < threshold) continue;
      const members = uniqueEmbeddedWindows([
        ...this.activeFamilyMembers(medoid.revision.id),
        window
      ]);
      const selected = selectConstrainedRealMedoid(members, threshold);
      if (!selected) {
        throw new Error(
          `threshold-valid Family lost a constrained real medoid: ${medoid.family.id}`
        );
      }
      this.commitFamilySelection(
        medoid.family,
        medoid.revision,
        members,
        selected,
        at
      );
      matchedFamilyIds.push(medoid.family.id);
    }
    if (matchedFamilyIds.length > 0) return unique(matchedFamilyIds);

    // Failure and unknown evidence may enrich an existing semantic Family,
    // but can never seed a reusable procedural capability by themselves.
    if (window.occurrence.evidenceRole !== "support") return [];

    const created = this.deps.repos.proceduralTrajectory.createFamilyHead({
      userId: window.occurrence.userId,
      scale,
      algorithmVersion: config.algorithmVersion,
      configHash: config.clusteringConfigHash,
      embeddingSignature: this.embeddingSignature(),
      seedOccurrenceId: window.occurrence.id,
      createdAt: at
    });
    const current = this.deps.repos.proceduralTrajectory.getFamilyHead(created.record.id)!;
    if (!current.activeRevisionId) {
      this.deps.repos.proceduralTrajectory.commitFamilyRevision({
        familyId: current.id,
        expectedActiveRevisionId: null,
        medoidOccurrenceId: window.occurrence.id,
        evidenceHash: this.familyEvidenceHash([window]),
        members: [{
          occurrenceId: window.occurrence.id,
          coarseSimilarity: 1
        }],
        metrics: {
          medoidCentrality: 1,
          minimumSimilarityToMedoid: 1,
          medoidUpdateCount: 0,
          algorithmVersion: config.algorithmVersion,
          configHash: config.clusteringConfigHash
        },
        createdAt: at
      });
    } else {
      const revision = this.deps.repos.proceduralTrajectory.getFamilyRevision(
        current.activeRevisionId
      );
      if (!revision) throw new Error(`active Family revision not found: ${current.id}`);
      const members = uniqueEmbeddedWindows([
        ...this.activeFamilyMembers(revision.id),
        window
      ]);
      const selected = selectConstrainedRealMedoid(members, threshold);
      if (!selected) throw new Error(`singleton Family is incoherent: ${current.id}`);
      this.commitFamilySelection(
        current,
        revision,
        members,
        selected,
        at
      );
    }
    return [current.id];
  }

  private activeFamilyMembers(familyRevisionId: string): EmbeddedTrajectoryWindowV1[] {
    return this.deps.repos.proceduralTrajectory.listFamilyMembers(familyRevisionId)
      .flatMap((member) => {
        const window = this.deps.repos.proceduralTrajectory.getWindow(member.occurrenceId);
        if (!window) return [];
        const activePath = this.deps.repos.proceduralTrajectory.getActivePath(window.episodeId);
        if (!activePath || activePath.id !== window.pathId ||
            !this.pathSourcesRemainActive(activePath)) return [];
        return [this.loadEmbeddedWindow(window)];
      });
  }

  private commitFamilySelection(
    head: TrajectoryWindowFamilyRecord,
    previousRevision: TrajectoryWindowFamilyRevisionRecord,
    members: EmbeddedTrajectoryWindowV1[],
    selected: NonNullable<ReturnType<typeof selectConstrainedRealMedoid>>,
    at: string
  ) {
    const previousUpdates = finiteNumber(previousRevision.metrics.medoidUpdateCount) ?? 0;
    return this.deps.repos.proceduralTrajectory.commitFamilyRevision({
      familyId: head.id,
      expectedActiveRevisionId: previousRevision.id,
      medoidOccurrenceId: selected.medoid.occurrence.id,
      evidenceHash: this.familyEvidenceHash(members),
      members: members.map((member) => ({
        occurrenceId: member.occurrence.id,
        coarseSimilarity: cosineSimilarity(member.coarseVector, selected.medoid.coarseVector)
      })),
      metrics: {
        medoidCentrality: selected.centrality,
        minimumSimilarityToMedoid: selected.minimumSimilarity,
        medoidUpdateCount: previousUpdates +
          (previousRevision.medoidOccurrenceId === selected.medoid.occurrence.id ? 0 : 1)
      },
      createdAt: at
    });
  }

  private familyEvidenceHash(members: readonly EmbeddedTrajectoryWindowV1[]): string {
    return stableHash({
      schema: "trajectory-window-family-evidence.v1",
      evidence: members.map((member) => ({
        occurrenceId: member.occurrence.id,
        evidenceRole: member.occurrence.evidenceRole,
        rewardHash: this.rewardHashForWindow(member.occurrence)
      })).sort((left, right) => left.occurrenceId.localeCompare(right.occurrenceId))
    });
  }

  /** Rebuild only this Family's deterministic, exclusive fine partition. */
  private rebuildFamilyFineClusters(
    familyId: string,
    config: PipelineConfig,
    at: string
  ): Set<string> {
    const affectedClusterIds = new Set(this.familyLinkedClusterIds(familyId));
    const head = this.deps.repos.proceduralTrajectory.getFamilyHead(familyId);
    if (!head || head.status !== "active" || !head.activeRevisionId ||
        head.algorithmVersion !== config.algorithmVersion ||
        head.configHash !== config.clusteringConfigHash ||
        head.embeddingSignature !== this.embeddingSignature()) {
      return affectedClusterIds;
    }
    const revision = this.deps.repos.proceduralTrajectory.getFamilyRevision(
      head.activeRevisionId
    );
    if (!revision) return affectedClusterIds;
    const members = this.activeFamilyMembers(revision.id);
    if (members.length === 0) return affectedClusterIds;
    const medoid = members.find((member) =>
      member.occurrence.id === revision.medoidOccurrenceId);
    if (!medoid) return affectedClusterIds;
    const fineConfig = config.fineConfigByScale.get(head.scale);
    if (!fineConfig) throw new Error(`fine config missing for Span-${head.scale}`);
    const similarities = members.map((member) =>
      cosineSimilarity(member.coarseVector, medoid.coarseVector));
    const family = {
      id: head.id,
      scale: head.scale,
      medoid,
      members,
      medoidCentrality: finiteNumber(revision.metrics.medoidCentrality) ??
        average(similarities),
      minimumSimilarityToMedoid:
        finiteNumber(revision.metrics.minimumSimilarityToMedoid) ??
        Math.min(...similarities),
      medoidUpdateCount: finiteNumber(revision.metrics.medoidUpdateCount) ?? 0
    };
    const fineClusters = buildExclusiveFineClusters(
      family,
      fineConfig,
      config.medoidSwitchMargin
    );
    const embeddedByOccurrenceId = new Map(members.map((member) => [
      member.occurrence.id,
      member
    ]));
    for (const local of fineClusters) {
      const canonical = this.deps.repos.proceduralTrajectory.resolveCanonicalClusterHead({
        userId: head.userId,
        scale: head.scale,
        algorithmVersion: config.algorithmVersion,
        configHash: config.clusteringConfigHash,
        embeddingSignature: head.embeddingSignature,
        evidenceSignature: fineEvidenceSignature(local),
        seedOccurrenceId: local.medoidOccurrenceId,
        createdAt: at
      });
      const currentHead = this.deps.repos.proceduralTrajectory.getClusterHead(
        canonical.cluster.id
      )!;
      const committed = this.deps.repos.proceduralTrajectory.commitClusterVersion({
        clusterId: currentHead.id,
        expectedActiveVersionId: currentHead.activeVersionId ?? null,
        medoidOccurrenceId: local.medoidOccurrenceId,
        members: local.members.map((member) => ({
          occurrenceId: member.occurrence.id,
          rewardHash: this.rewardHashForWindow(member.occurrence),
          coarseSimilarity: cosineSimilarity(
            embeddedByOccurrenceId.get(member.occurrence.id)!.coarseVector,
            embeddedByOccurrenceId.get(local.medoidOccurrenceId)!.coarseVector
          ),
          alignment: alignmentRecord(member.alignmentToMedoid)
        })),
        metrics: {
          averageSimilarity: local.averageSimilarity,
          minimumSimilarity: local.minimumSimilarity,
          medoidCentrality: local.medoidCentrality,
          medoidUpdateCount: local.medoidUpdateCount,
          algorithmVersion: config.algorithmVersion,
          configHash: config.clusteringConfigHash,
          familyRevisionId: revision.id,
          fineEvidenceSignature: canonical.canonicalKey.evidenceSignature
        },
        createdAt: at
      });
      this.deps.repos.proceduralTrajectory.linkFamilyRevisionToCluster({
        familyRevisionId: revision.id,
        canonicalKeyId: canonical.canonicalKey.id,
        clusterVersionId: committed.record.id,
        createdAt: at
      });
      affectedClusterIds.add(currentHead.id);
    }
    return affectedClusterIds;
  }

  private familyLinkedClusterIds(familyId: string): string[] {
    const ids = new Set<string>();
    for (const revision of this.deps.repos.proceduralTrajectory.listFamilyRevisions(familyId)) {
      for (const link of this.deps.repos.proceduralTrajectory.listFamilyClusterLinks(revision.id)) {
        const canonical = this.deps.repos.proceduralTrajectory.getClusterCanonicalKey(
          link.canonicalKeyId
        );
        if (canonical) ids.add(canonical.clusterId);
      }
    }
    return [...ids].sort();
  }

  private clusterHasActiveFamilyLink(clusterId: string): boolean {
    return this.deps.repos.proceduralTrajectory.listClusterFamilyLinks(clusterId)
      .some((link) => {
        const revision = this.deps.repos.proceduralTrajectory.getFamilyRevision(
          link.familyRevisionId
        );
        if (!revision) return false;
        const family = this.deps.repos.proceduralTrajectory.getFamilyHead(revision.familyId);
        return family?.status === "active" && family.activeRevisionId === revision.id;
      });
  }

  private retireUnlinkedCanonicalClusters(clusterIds: Iterable<string>, at: string): void {
    for (const clusterId of [...new Set(clusterIds)].sort()) {
      if (this.clusterHasActiveFamilyLink(clusterId)) continue;
      const head = this.deps.repos.proceduralTrajectory.getClusterHead(clusterId);
      if (!head || head.status !== "active" || !head.activeVersionId) continue;
      const version = this.deps.repos.proceduralTrajectory.getClusterVersion(
        head.activeVersionId
      );
      if (!version) continue;
      this.retireAndDowngradeCluster(
        head,
        version,
        "fine-evidence-no-longer-linked",
        at
      );
    }
  }

  private reconcileFamilyAfterPathChange(
    familyId: string,
    config: PipelineConfig,
    at: string
  ): { orphaned: EmbeddedTrajectoryWindowV1[]; affectedClusterIds: Set<string> } {
    const affectedClusterIds = new Set(this.familyLinkedClusterIds(familyId));
    const head = this.deps.repos.proceduralTrajectory.getFamilyHead(familyId);
    if (!head || head.status !== "active" || !head.activeRevisionId ||
        head.algorithmVersion !== config.algorithmVersion ||
        head.configHash !== config.clusteringConfigHash) {
      return { orphaned: [], affectedClusterIds };
    }
    const revision = this.deps.repos.proceduralTrajectory.getFamilyRevision(
      head.activeRevisionId
    );
    if (!revision) return { orphaned: [], affectedClusterIds };
    const activeMembers = this.activeFamilyMembers(revision.id);
    if (activeMembers.length === 0) {
      this.deps.repos.proceduralTrajectory.retireFamily(head.id, revision.id, at);
      return { orphaned: [], affectedClusterIds };
    }
    const threshold = config.coarseThresholdByScale[head.scale];
    if (threshold === undefined) throw new Error(`coarse config missing for Span-${head.scale}`);
    const selected = selectConstrainedRealMedoid(activeMembers, threshold);
    if (!selected) {
      // Losing a real medoid can make the residual Family incoherent. Retire
      // only this Family and deterministically re-ingest its surviving evidence.
      this.deps.repos.proceduralTrajectory.retireFamily(head.id, revision.id, at);
      return { orphaned: activeMembers, affectedClusterIds };
    }
    this.commitFamilySelection(head, revision, activeMembers, selected, at);
    return { orphaned: [], affectedClusterIds };
  }

  private activeMemberCandidates(clusterVersionId: string): MemberCandidate[] {
    return this.deps.repos.proceduralTrajectory.listClusterMembers(clusterVersionId)
      .flatMap((member) => {
        const window = this.deps.repos.proceduralTrajectory.getWindow(member.occurrenceId);
        if (!window) return [];
        const activePath = this.deps.repos.proceduralTrajectory.getActivePath(window.episodeId);
        if (!activePath || activePath.id !== window.pathId ||
            !this.pathSourcesRemainActive(activePath)) return [];
        return [{
          embedded: this.loadEmbeddedWindow(window),
          rewardHash: this.rewardHashForWindowRecord(window)
        }];
      });
  }

  private loadEmbeddedWindow(record: TrajectoryWindowOccurrenceRecord): EmbeddedTrajectoryWindowV1 {
    const path = this.deps.repos.proceduralTrajectory.getPath(record.pathId);
    if (!path) throw new Error(`trajectory window Path not found: ${record.pathId}`);
    const stepById = new Map(path.path.steps.map((step) => [step.id, step]));
    const steps = record.stepIds.map((stepId) => stepById.get(stepId)).filter(isDefined);
    if (steps.length !== record.scale) {
      throw new Error(`trajectory window Steps are incomplete: ${record.id}`);
    }
    const embeddingByStepId = new Map(this.deps.repos.proceduralTrajectory
      .listStepEmbeddings({
        pathId: record.pathId,
        representationVersion: STEP_INTENT_REPRESENTATION_VERSION,
        embeddingSignature: record.embeddingSignature
      }).map((item) => [item.stepId, item.vector]));
    const stepVectors = steps.map((step) => embeddingByStepId.get(step.id)).filter(isDefined);
    if (stepVectors.length !== record.scale) {
      throw new Error(`trajectory window Step embeddings are incomplete: ${record.id}`);
    }
    return {
      occurrence: occurrenceFromRecord(record, steps),
      coarseVector: record.coarseVector,
      stepVectors
    };
  }

  private loadClusterSnapshot(
    head: TrajectoryWindowClusterRecord,
    config: PipelineConfig
  ): ClusterSnapshot | undefined {
    if (!head.activeVersionId) return undefined;
    const version = this.deps.repos.proceduralTrajectory.getClusterVersion(head.activeVersionId);
    if (!version) return undefined;
    const members = this.activeMemberCandidates(version.id);
    if (members.length === 0) return undefined;
    const medoid = members.find((member) =>
      member.embedded.occurrence.id === version.medoidOccurrenceId)?.embedded;
    if (!medoid) return undefined;
    const fineConfig = config.fineConfigByScale.get(head.scale);
    if (!fineConfig) return undefined;
    const domainMembers: TrajectoryWindowClusterMemberV1[] = members.flatMap((member) => {
      const alignmentToMedoid = member.embedded.occurrence.id === medoid.occurrence.id
        ? selfBandedMonotonicMatch(member.embedded.stepVectors, fineConfig)
        : bandedMonotonicMatch(member.embedded.stepVectors, medoid.stepVectors, fineConfig);
      if (!alignmentToMedoid.admitted) return [];
      return [{
        occurrence: member.embedded.occurrence,
        similarityToMedoid: alignmentToMedoid.score,
        alignmentToMedoid
      }];
    });
    if (domainMembers.length === 0) return undefined;
    const supportEpisodeIds = distinctEpisodeIds(domainMembers.filter((member) =>
      member.occurrence.evidenceRole === "support"));
    const counterexampleEpisodeIds = distinctEpisodeIds(domainMembers.filter((member) =>
      member.occurrence.evidenceRole === "counterexample"));
    const unknownEpisodeIds = distinctEpisodeIds(domainMembers.filter((member) =>
      member.occurrence.evidenceRole === "unknown"));
    const similarities = domainMembers.map((member) => member.similarityToMedoid);
    const domain: TrajectoryWindowClusterV1 = {
      id: head.id,
      familyId: this.activeFamilyIdsForCluster(head.id)[0] ?? head.id,
      scale: head.scale,
      medoidOccurrenceId: medoid.occurrence.id,
      episodeIds: distinctEpisodeIds(domainMembers),
      supportEpisodeIds,
      counterexampleEpisodeIds,
      unknownEpisodeIds,
      occurrenceCount: domainMembers.length,
      averageSimilarity: average(similarities),
      minimumSimilarity: Math.min(...similarities),
      medoidCentrality: finiteNumber(version.metrics.medoidCentrality) ?? average(similarities),
      medoidUpdateCount: finiteNumber(version.metrics.medoidUpdateCount) ?? 0,
      members: domainMembers
    };
    return {
      head,
      version,
      domain,
      core: extractAlignedCommonCore(domain, {
        minSupportEpisodes: config.minSupportEpisodes,
        minCoreSteps: config.fineConfigByScale.get(head.scale)?.minMatchedSteps
      })
    };
  }

  private activeFamilyIdsForCluster(clusterId: string): string[] {
    const ids = new Set<string>();
    for (const link of this.deps.repos.proceduralTrajectory.listClusterFamilyLinks(clusterId)) {
      const revision = this.deps.repos.proceduralTrajectory.getFamilyRevision(
        link.familyRevisionId
      );
      if (!revision) continue;
      const family = this.deps.repos.proceduralTrajectory.getFamilyHead(revision.familyId);
      if (family?.status === "active" && family.activeRevisionId === revision.id) {
        ids.add(family.id);
      }
    }
    return [...ids].sort();
  }

  private maximalQualifiedClusters(
    userId: string,
    config: PipelineConfig
  ): ClusterSnapshot[] {
    const qualified = config.specs.flatMap((spec) =>
      this.deps.repos.proceduralTrajectory.listActiveClusterHeads({
        userId,
        scale: spec.length,
        algorithmVersion: config.algorithmVersion,
        configHash: config.clusteringConfigHash
      }).flatMap((head) => {
        const snapshot = this.loadClusterSnapshot(head, config);
        return snapshot?.core &&
          snapshot.version.supportEpisodeCount >= config.minSupportEpisodes
          ? [snapshot]
          : [];
      }));
    const selectedById = new Map(selectMaximalWindowClusters(
      qualified.map((snapshot) => snapshot.domain)
    ).map((domain) => [domain.id, domain]));
    return qualified.map((snapshot) => ({
      ...snapshot,
      domain: selectedById.get(snapshot.domain.id) ?? snapshot.domain
    }));
  }

  private reconcileSkillEligibility(
    userId: string,
    affectedClusterIds: Set<string>,
    config: PipelineConfig,
    at: string
  ): void {
    const maximal = this.maximalQualifiedClusters(userId, config);
    const maximalIds = new Set(maximal.filter((snapshot) =>
      !snapshot.domain.suppressedByClusterId).map((snapshot) => snapshot.head.id));
    const qualifiedIds = new Set(maximal.map((snapshot) => snapshot.head.id));

    for (const snapshot of maximal) {
      if (this.proceduralSkillGovernanceDisabled(snapshot.head)) {
        this.rejectAndDowngradeSkill(
          snapshot.head,
          snapshot.version,
          "governance-disabled",
          at
        );
        continue;
      }
      if (snapshot.domain.suppressedByClusterId) {
        if (snapshot.head.activeSkillVersionId || this.clusterHadFormalSkill(snapshot.head.id)) {
          this.rejectAndDowngradeSkill(
            snapshot.head,
            snapshot.version,
            `maximal-suppressed-by:${snapshot.domain.suppressedByClusterId}`,
            at
          );
        }
        continue;
      }
      if (this.activeSkillDecisionMatchesClusterVersion(
        snapshot.head,
        snapshot.version.id,
        config
      )) continue;
      this.deps.enqueueJob({
        jobType: "procedural_skill_induction",
        userId,
        episodeId: snapshot.domain.supportEpisodeIds[0],
        targetMemoryId: undefined,
        payload: {
          reason: "trajectory_cluster.version_ready",
          clusterId: snapshot.head.id,
          clusterVersionId: snapshot.version.id,
          membershipVersion: snapshot.version.versionNo,
          inductionVersion: this.skillInductionVersion(config),
          commonCoreId: snapshot.core!.id
        },
        createdAt: at
      });
    }

    for (const clusterId of affectedClusterIds) {
      if (qualifiedIds.has(clusterId) || maximalIds.has(clusterId)) continue;
      const head = this.deps.repos.proceduralTrajectory.getClusterHead(clusterId);
      if (!head || head.status !== "active" || !head.activeVersionId) continue;
      const version = this.deps.repos.proceduralTrajectory.getClusterVersion(head.activeVersionId);
      if (!version || !this.clusterHadFormalSkill(clusterId)) continue;
      this.rejectAndDowngradeSkill(
        head,
        version,
        "cluster-no-longer-qualified",
        at
      );
    }
  }

  private buildSkillInput(
    snapshot: ClusterSnapshot,
    config: PipelineConfig
  ): ProceduralPatternSkillInput | undefined {
    const core = snapshot.core;
    if (!core) return undefined;
    const successfulEpisodeIds = new Set(core.supportEpisodeIds.filter((episodeId) => {
      const episode = this.deps.repos.runtime.getEpisode(episodeId);
      return episode?.userId === snapshot.head.userId &&
        typeof episode.rTask === "number" && episode.rTask > 0;
    }));
    if (successfulEpisodeIds.size < config.minSupportEpisodes) return undefined;
    const positiveSpanOccurrenceIds = new Set(core.spanOccurrences
      .filter((span) => span.evidenceRole === "support" &&
        successfulEpisodeIds.has(span.episodeId))
      .map((span) => span.id));
    const commonCore = core.steps.map((step, index) => ({
      anchorId: `core_${index}`,
      anchorOffset: step.anchorOffset,
      anchorStepId: step.anchorStepId,
      anchorIntent: step.intent,
      anchorSummary: step.summary,
      supportEpisodeIds: step.supportEpisodeIds,
      evidenceStepIds: step.evidenceStepIds
    }));
    const anchorIndexByOffset = new Map(core.steps.map((step, index) => [
      step.anchorOffset,
      index
    ]));
    const memberByWindowId = new Map(snapshot.domain.members.map((member) => [
      member.occurrence.id,
      member
    ]));
    const pathById = new Map<string, EpisodeExecutionPathLiteV1>();
    const vectorByPathId = new Map<string, Map<string, number[]>>();
    const drafts: Array<{
      span: AlignedCommonCoreV1["spanOccurrences"][number];
      path: EpisodeExecutionPathLiteV1;
      member: TrajectoryWindowClusterMemberV1;
      alignedSequence: ProceduralSkillAlignedSequenceStep[];
      prefixSteps: ExecutionStepLiteV1[];
      suffixSteps: ExecutionStepLiteV1[];
      corePreviousStep?: ExecutionStepLiteV1;
      coreNextStep?: ExecutionStepLiteV1;
      expandedPreviousStep?: ExecutionStepLiteV1;
      expandedNextStep?: ExecutionStepLiteV1;
      coreTraceSteps: ExecutionStepLiteV1[];
    }> = [];
    for (const span of core.spanOccurrences) {
      const path = pathById.get(span.pathId) ??
        this.deps.repos.proceduralTrajectory.getPath(span.pathId)?.path;
      if (!path) continue;
      pathById.set(span.pathId, path);
      const stepById = new Map(path.steps.map((step) => [step.id, step]));
      const matchedSteps = span.matchedStepIds
        .map((stepId) => stepById.get(stepId))
        .filter(isDefined);
      const gapSteps = span.gapStepIds
        .map((stepId) => stepById.get(stepId))
        .filter(isDefined);
      if (matchedSteps.length !== span.matchedStepIds.length ||
          gapSteps.length !== span.gapStepIds.length) continue;
      const member = memberByWindowId.get(span.sourceWindowOccurrenceId);
      if (!member) continue;
      const coreMetaByStepId = new Map<string, {
        anchorId: string;
        matchSimilarity: number;
      }>();
      for (const pair of member.alignmentToMedoid.pairs) {
        const anchorIndex = anchorIndexByOffset.get(pair.rightIndex);
        if (anchorIndex === undefined) continue;
        const step = member.occurrence.steps[pair.leftIndex];
        if (!step) continue;
        coreMetaByStepId.set(step.id, {
          anchorId: commonCore[anchorIndex]!.anchorId,
          matchSimilarity: pair.similarity
        });
      }
      if (matchedSteps.some((step) => !coreMetaByStepId.has(step.id))) continue;
      const gapStepIds = new Set(gapSteps.map((step) => step.id));
      const orderedSteps = [...matchedSteps, ...gapSteps]
        .sort((left, right) => left.stepIndex - right.stepIndex ||
          left.id.localeCompare(right.id));
      const alignedSequence: ProceduralSkillAlignedSequenceStep[] = [];
      let invalidSequence = false;
      for (let index = 0; index < orderedSteps.length; index += 1) {
        const step = orderedSteps[index]!;
        const common = coreMetaByStepId.get(step.id);
        if (common) {
          alignedSequence.push({
            role: "core",
            anchorId: common.anchorId,
            matchSimilarity: common.matchSimilarity,
            ...skillEvidenceStep(step)
          });
          continue;
        }
        if (!gapStepIds.has(step.id)) {
          invalidSequence = true;
          break;
        }
        const previous = [...alignedSequence].reverse().find((item) => item.role === "core");
        const nextStep = orderedSteps.slice(index + 1).find((item) =>
          coreMetaByStepId.has(item.id));
        const next = nextStep ? coreMetaByStepId.get(nextStep.id) : undefined;
        if (!previous || previous.role !== "core" || !next) {
          invalidSequence = true;
          break;
        }
        alignedSequence.push({
          role: "gap",
          afterAnchorId: previous.anchorId,
          beforeAnchorId: next.anchorId,
          ...skillEvidenceStep(step)
        });
      }
      if (invalidSequence) continue;
      const pathSteps = [...path.steps]
        .sort((left, right) => left.stepIndex - right.stepIndex ||
          left.id.localeCompare(right.id));
      const windowSteps = [...member.occurrence.steps]
        .sort((left, right) => left.stepIndex - right.stepIndex ||
          left.id.localeCompare(right.id));
      const corePreviousStep = [...windowSteps].reverse().find((step) =>
        step.stepIndex < span.startStepIndex);
      const coreNextStep = windowSteps.find((step) =>
        step.stepIndex > span.endStepIndex);
      const prefixBudget = config.completionExpansionEnabled
        ? config.maxPrefixExpansionSteps
        : 0;
      const suffixBudget = config.completionExpansionEnabled
        ? config.maxSuffixExpansionSteps
        : 0;
      const prefixSteps = prefixBudget === 0
        ? []
        : pathSteps
          .filter((step) => step.stepIndex < span.startStepIndex)
          .slice(-prefixBudget);
      const suffixSteps = pathSteps
        .filter((step) => step.stepIndex > span.endStepIndex)
        .slice(0, suffixBudget);
      const expandedStart = prefixSteps[0]?.stepIndex ?? span.startStepIndex;
      const expandedEnd = suffixSteps.at(-1)?.stepIndex ?? span.endStepIndex;
      const expandedPreviousStep = [...pathSteps].reverse().find((step) =>
        step.stepIndex < expandedStart);
      const expandedNextStep = pathSteps.find((step) => step.stepIndex > expandedEnd);
      const positive = positiveSpanOccurrenceIds.has(span.id);
      const readOnlyCounterexample = span.evidenceRole === "counterexample" ||
        matchedSteps.some((step) => step.outcome === "failure");
      if (!positive && !readOnlyCounterexample) continue;
      drafts.push({
        span,
        path,
        member,
        alignedSequence,
        prefixSteps,
        suffixSteps,
        ...(corePreviousStep ? { corePreviousStep } : {}),
        ...(coreNextStep ? { coreNextStep } : {}),
        ...(expandedPreviousStep ? { expandedPreviousStep } : {}),
        ...(expandedNextStep ? { expandedNextStep } : {}),
        coreTraceSteps: orderedSteps
      });
    }
    const completionCandidates: AnchoredCompletionOccurrenceCandidateV2[] = [];
    for (const draft of drafts) {
      let vectorByStepId = vectorByPathId.get(draft.path.id);
      if (!vectorByStepId) {
        vectorByStepId = new Map(this.deps.repos.proceduralTrajectory
          .listStepEmbeddings({
            pathId: draft.path.id,
            representationVersion: STEP_INTENT_REPRESENTATION_VERSION,
            embeddingSignature: this.embeddingSignature()
          }).map((item) => [item.stepId, item.vector]));
        vectorByPathId.set(draft.path.id, vectorByStepId);
      }
      const prefix = draft.prefixSteps.flatMap((step) => {
        const vector = vectorByStepId!.get(step.id);
        return vector ? [{ stepId: step.id, vector }] : [];
      });
      const suffix = draft.suffixSteps.flatMap((step) => {
        const vector = vectorByStepId!.get(step.id);
        return vector ? [{ stepId: step.id, vector }] : [];
      });
      if (prefix.length !== draft.prefixSteps.length ||
          suffix.length !== draft.suffixSteps.length) continue;
      completionCandidates.push({
        occurrenceId: draft.span.id,
        episodeId: draft.span.episodeId,
        evidenceRole: draft.span.evidenceRole,
        prefix,
        suffix
      });
    }
    if (completionCandidates.length !== drafts.length) return undefined;
    const referenceOccurrenceId = drafts.find((draft) =>
      draft.span.sourceWindowOccurrenceId === core.medoidOccurrenceId &&
      draft.span.evidenceRole === "support")?.span.id;
    const completionOverlay = extractAnchoredCompletionOverlay(
      core.id,
      completionCandidates,
      {
        ...(referenceOccurrenceId ? { referenceOccurrenceId } : {}),
        maxPrefixSteps: config.completionExpansionEnabled
          ? config.maxPrefixExpansionSteps
          : 0,
        maxSuffixSteps: config.completionExpansionEnabled
          ? config.maxSuffixExpansionSteps
          : 0,
        minStepSimilarity: config.minExtensionStepSimilarity,
        minSupportEpisodes: config.minSupportEpisodes
      }
    );
    if (!completionOverlay) return undefined;
    const completionActivated = completionOverlay.sharedPrefix.length > 0 ||
      completionOverlay.sharedSuffix.length > 0;
    const effectiveCompletionOverlay = completionActivated
      ? completionOverlay
      : {
          ...completionOverlay,
          id: `anchored_completion_${stableHash({
            version: ANCHORED_COMPLETION_SCHEMA_VERSION,
            mode: "core_only",
            commonCoreId: core.id,
            referenceOccurrenceId: completionOverlay.referenceOccurrenceId
          }).slice(0, 24)}`,
          maxPrefixSteps: 0,
          maxSuffixSteps: 0,
          sharedPrefix: [],
          sharedSuffix: [],
          projections: completionOverlay.projections.map((item) => ({
            occurrenceId: item.occurrenceId,
            prefix: [],
            suffix: []
          })),
          extensionAgreement: 0
        };
    const projectionByOccurrenceId = new Map(
      effectiveCompletionOverlay.projections.map((item) => [
      item.occurrenceId,
      item
    ]));
    const evidence: ProceduralSkillEvidenceOccurrence[] = drafts.flatMap((draft) => {
      const projection = projectionByOccurrenceId.get(draft.span.id);
      const effectivePrefixSteps = completionActivated ? draft.prefixSteps : [];
      const effectiveSuffixSteps = completionActivated ? draft.suffixSteps : [];
      if (!projection || projection.prefix.length !== effectivePrefixSteps.length ||
          projection.suffix.length !== effectiveSuffixSteps.length) return [];
      const expansionSteps = (
        steps: ExecutionStepLiteV1[],
        projected: typeof projection.prefix,
        side: "prefix" | "suffix"
      ): ProceduralSkillExpansionStep[] => steps.map((step, index) => {
        const item = projected[index]!;
        return {
          role: item.role,
          side,
          ...(item.extensionAnchorId
            ? { extensionAnchorId: item.extensionAnchorId }
            : {}),
          ...(item.matchSimilarity === undefined
            ? {}
            : { matchSimilarity: item.matchSimilarity }),
          ...skillEvidenceStep(step)
        };
      });
      return [{
        occurrenceId: draft.span.id,
        episodeId: draft.span.episodeId,
        pathId: draft.span.pathId,
        scale: draft.span.scale,
        alignmentScore: draft.span.averageMatchSimilarity,
        sourceTraceIds: this.sourceTraceIdsForSteps(
          draft.span.episodeId,
          completionActivated
            ? [...draft.prefixSteps, ...draft.coreTraceSteps, ...draft.suffixSteps]
            : draft.coreTraceSteps
        ),
        prefixExpansion: expansionSteps(
          effectivePrefixSteps,
          projection.prefix,
          "prefix"
        ),
        alignedSequence: draft.alignedSequence,
        suffixExpansion: expansionSteps(
          effectiveSuffixSteps,
          projection.suffix,
          "suffix"
        ),
        boundaryContextReadOnly: {
          ...((completionActivated
            ? draft.expandedPreviousStep
            : draft.corePreviousStep)
            ? { previousStep: skillEvidenceStep((completionActivated
                ? draft.expandedPreviousStep
                : draft.corePreviousStep)!) }
            : {}),
          ...((completionActivated
            ? draft.expandedNextStep
            : draft.coreNextStep)
            ? { nextStep: skillEvidenceStep((completionActivated
                ? draft.expandedNextStep
                : draft.coreNextStep)!) }
            : {})
        }
      }];
    });
    const supportEpisodeIds = unique(evidence
      .filter((item) => positiveSpanOccurrenceIds.has(item.occurrenceId))
      .map((item) => item.episodeId));
    if (supportEpisodeIds.length < config.minSupportEpisodes) return undefined;
    const patternHash = stableHash({
      algorithmVersion: config.algorithmVersion,
      clusterVersionId: snapshot.version.id,
      commonCore: core,
      completion: effectiveCompletionOverlay,
      skillCompletionConfigHash: config.skillCompletionConfigHash
    });
    return {
      patternVersionId: `aligned_core_version_${patternHash.slice(0, 24)}`,
      clusterId: snapshot.head.id,
      clusterVersionId: snapshot.version.id,
      commonCoreId: core.id,
      commonCore,
      completion: {
        id: effectiveCompletionOverlay.id,
        version: effectiveCompletionOverlay.schemaVersion,
        activated: completionActivated,
        referenceOccurrenceId: effectiveCompletionOverlay.referenceOccurrenceId,
        maxPrefixSteps: effectiveCompletionOverlay.maxPrefixSteps,
        maxSuffixSteps: effectiveCompletionOverlay.maxSuffixSteps,
        minStepSimilarity: effectiveCompletionOverlay.minStepSimilarity,
        sharedPrefix: effectiveCompletionOverlay.sharedPrefix,
        sharedSuffix: effectiveCompletionOverlay.sharedSuffix,
        extensionAgreement: effectiveCompletionOverlay.extensionAgreement
      },
      userId: snapshot.head.userId,
      scale: snapshot.head.scale,
      supportEpisodeIds,
      sourceTraceIds: unique(evidence.flatMap((item) => item.sourceTraceIds)),
      sourceSpanOccurrenceIds: core.spanOccurrences.map((item) => item.id),
      counterexampleEpisodeIds: snapshot.domain.counterexampleEpisodeIds,
      evidence,
      confidenceHint: snapshot.domain.averageSimilarity,
      patternHash,
      algorithmVersion: config.algorithmVersion
    };
  }

  private sourceTraceIdsForSteps(episodeId: string, steps: ExecutionStepLiteV1[]): string[] {
    const episode = this.deps.repos.runtime.getEpisode(episodeId);
    if (!episode) return [];
    const rawTurnIds = new Set(steps.map((step) => step.rawTurnId));
    return this.deps.repos.memories.getMany(episode.l1MemoryIds)
      .map((memory) => this.deps.traceMeta(memory))
      .filter((trace): trace is TraceMeta => Boolean(
        trace && trace.userId === episode.userId && trace.episodeId === episodeId &&
        trace.rawTurnId && rawTurnIds.has(trace.rawTurnId)
      ))
      .map((trace) => trace.id);
  }

  private activeSkillMatchesClusterVersion(
    head: TrajectoryWindowClusterRecord,
    clusterVersionId: string
  ): boolean {
    if (!head.activeSkillVersionId || !head.activeSkillMemoryId) return false;
    const version = this.deps.repos.proceduralTrajectory.getSkillVersion(
      head.activeSkillVersionId
    );
    const memory = this.deps.repos.memories.get(head.activeSkillMemoryId);
    return Boolean(memory && version?.clusterVersionId === clusterVersionId &&
      version.skillMemoryId === memory.id && memory.status !== "archived" &&
      skillMetaFromMemory(memory)?.status !== "archived");
  }

  private activeSkillDecisionMatchesClusterVersion(
    head: TrajectoryWindowClusterRecord,
    clusterVersionId: string,
    config: PipelineConfig
  ): boolean {
    if (!head.activeSkillVersionId) return false;
    const version = this.deps.repos.proceduralTrajectory.getSkillVersion(
      head.activeSkillVersionId
    );
    if (!version || version.clusterVersionId !== clusterVersionId) return false;
    if (version.payload.inductionVersion !== this.skillInductionVersion(config)) {
      return false;
    }
    if (version.skillMemoryId) return this.activeSkillMatchesClusterVersion(
      head,
      clusterVersionId
    );
    const reason = typeof version.payload.reason === "string" ? version.payload.reason : "";
    // A previously suppressed shorter Skill must be reconsidered when the
    // covering longer cluster disappears. Other decisions are current for this
    // immutable membership version.
    return !reason.startsWith("maximal-suppressed-by:") && reason !== "not-maximal-qualified";
  }

  private clusterHadFormalSkill(clusterId: string): boolean {
    return this.deps.repos.proceduralTrajectory.listSkillVersions(clusterId)
      .some((version) => Boolean(version.skillMemoryId));
  }

  private retireAndDowngradeCluster(
    head: TrajectoryWindowClusterRecord,
    version: TrajectoryWindowClusterVersionRecord,
    reason: string,
    at: string
  ): void {
    this.rejectAndDowngradeSkill(head, version, reason, at);
    const current = this.deps.repos.proceduralTrajectory.getClusterHead(head.id);
    if (!current || current.status === "retired") return;
    this.deps.repos.proceduralTrajectory.retireCluster(
      current.id,
      current.activeVersionId ?? null,
      at
    );
  }

  private rejectAndDowngradeSkill(
    head: TrajectoryWindowClusterRecord,
    version: TrajectoryWindowClusterVersionRecord | undefined,
    reason: string,
    at: string
  ): void {
    if (!version || head.status !== "active" || head.activeVersionId !== version.id) return;
    const inductionVersion = this.skillInductionVersion(this.pipelineConfig());
    this.deps.repos.transaction(() => {
      const current = this.deps.repos.proceduralTrajectory.getClusterHead(head.id);
      if (!current || current.status !== "active" || current.activeVersionId !== version.id) return;
      const oldMemoryId = current.activeSkillMemoryId;
      // Archive first and advance the rejection head second, but keep both in
      // one SQLite transaction. A CAS failure or process error rolls the
      // archive back, so the head can never forget a still-retrievable Skill.
      if (oldMemoryId) this.archiveProceduralSkillMemory(oldMemoryId, reason, at);
      this.deps.repos.proceduralTrajectory.saveSkillVersion({
        clusterId: current.id,
        clusterVersionId: version.id,
        expectedActiveSkillVersionId: current.activeSkillVersionId ?? null,
        skillKey: proceduralSkillKey(current.userId, current.id),
        payload: {
          admitted: false,
          inductionVersion,
          reason,
          clusterVersionId: version.id,
          supportEpisodeCount: version.supportEpisodeCount
        },
        contentHash: stableHash({
          admitted: false,
          clusterVersionId: version.id,
          inductionVersion,
          reason
        }),
        createdAt: at
      });
    });
  }

  private archiveProceduralSkillMemory(memoryId: string, reason: string, at: string): void {
    const memory = this.deps.repos.memories.get(memoryId);
    if (!memory || memory.status === "archived") return;
    const internal = memory.properties.internal_info;
    if (internal.plugin_algorithm !== "procedural.pattern.skill.v1") return;
    const internalSkill = isRecord(internal.skill) ? internal.skill : {};
    const archived = this.deps.repos.memories.update({
      ...memory,
      status: "archived",
      info: {
        ...memory.info,
        status: "archived",
        procedural_retirement_reason: reason
      },
      properties: {
        ...memory.properties,
        status: "archived",
        internal_info: {
          ...internal,
          status: "archived",
          procedural_retirement_reason: reason,
          skill: {
            ...internalSkill,
            status: "archived",
            procedural_retirement_reason: reason
          }
        }
      },
      updatedAt: at
    });
    this.deps.repos.runtime.appendChange({
      memoryId: archived.id,
      namespaceId: this.deps.namespaceIdFromMemory(archived),
      kind: kindFromMemory(archived),
      op: "archived",
      entityId: archived.id,
      userId: archived.userId,
      changeType: "procedural_skill_cluster_retired",
      before: memory,
      after: archived,
      source: "worker.procedural_skill_induction.lifecycle.v1",
      createdAt: at
    });
  }

  private currentSourceState(
    episodeId: string,
    rawTurns: RawTurnRecord[]
  ): {
    rawTurns: RawTurnRecord[];
    sourceTraceIds: string[];
    sourceSnapshotHash: string;
  } {
    const episode = this.deps.repos.runtime.getEpisode(episodeId);
    if (!episode) {
      return {
        rawTurns: [],
        sourceTraceIds: [],
        sourceSnapshotHash: stableHash({
          schema: "episode_trajectory_governed_source.v1",
          episodeId,
          missingEpisode: true
        })
      };
    }
    const sourceTraces = this.deps.repos.memories.getMany(episode.l1MemoryIds)
      .flatMap((memory) => {
        if (memory.status !== "activated" && memory.status !== "resolving") return [];
        const trace = this.deps.traceMeta(memory);
        return trace?.episodeId === episode.id && trace.rawTurnId
          ? [{ id: trace.id, rawTurnId: trace.rawTurnId, contentHash: memory.contentHash }]
          : [];
      });
    const authorizedRawTurnIds = new Set(sourceTraces.map((trace) => trace.rawTurnId));
    const eligibleRawTurns = rawTurns.filter((rawTurn) =>
      rawTurn.episodeId === episode.id &&
      rawTurn.userId === episode.userId &&
      !rawTurn.redactedAt &&
      !rawTurn.deletedAt &&
      authorizedRawTurnIds.has(rawTurn.id)
    );
    const sourceSnapshotHash = stableHash({
      schema: "episode_trajectory_governed_source.v1",
      episodeId: episode.id,
      userId: episode.userId,
      sourceTraces: sourceTraces
        .map((trace) => ({
          id: trace.id,
          rawTurnId: trace.rawTurnId,
          contentHash: trace.contentHash ?? null
        }))
        .sort((left, right) => left.id.localeCompare(right.id)),
      rawTurns: rawTurns.map((rawTurn) => ({
        id: rawTurn.id,
        authorized: authorizedRawTurnIds.has(rawTurn.id),
        redactedAt: rawTurn.redactedAt ?? null,
        deletedAt: rawTurn.deletedAt ?? null,
        status: rawTurn.status,
        contentHash: stableHash({
          userText: rawTurn.userText ?? "",
          assistantText: rawTurn.assistantText ?? "",
          reasoningSummary: rawTurn.reasoningSummary ?? "",
          toolCalls: rawTurn.toolCalls,
          toolResults: rawTurn.toolResults,
          messagePayload: rawTurn.messagePayload ?? {}
        })
      }))
    });
    return {
      rawTurns: eligibleRawTurns,
      sourceTraceIds: sourceTraces.map((trace) => trace.id),
      sourceSnapshotHash
    };
  }

  private pathSourcesRemainActive(path: EpisodeExecutionPathRecord): boolean {
    const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
    if (!episode || episode.status !== "closed" || episode.userId !== path.userId) return false;
    const rawTurns = this.deps.repos.runtime.listRawTurnsByEpisode(episode.id, 10_000);
    const state = this.currentSourceState(episode.id, rawTurns);
    const activeRawTurnIds = new Set(state.rawTurns.map((rawTurn) => rawTurn.id));
    return path.sourceSnapshotHash === state.sourceSnapshotHash &&
      path.sourceRawTurnIds.length > 0 &&
      path.sourceRawTurnIds.every((rawTurnId) => activeRawTurnIds.has(rawTurnId));
  }

  private enqueueGovernedPathCompile(
    episodeId: string,
    reason: string,
    at: string,
    previousPathId?: string
  ): void {
    const episode = this.deps.repos.runtime.getEpisode(episodeId);
    if (!episode || episode.status !== "closed") return;
    const rawTurns = this.deps.repos.runtime.listRawTurnsByEpisode(episode.id, 10_000);
    const source = this.currentSourceState(episode.id, rawTurns);
    if (source.rawTurns.length === 0) return;
    this.deps.enqueueJob({
      jobType: "episode_path_compile",
      userId: episode.userId,
      sessionId: episode.sessionId,
      episodeId: episode.id,
      payload: {
        reason: `source_governance.${reason}`,
        semanticsVersion: EPISODE_PATH_COMPILER_VERSION,
        sourceSnapshotHash: source.sourceSnapshotHash,
        rewardSnapshotHash: rewardHashForEpisode(episode),
        rawTurnIds: source.rawTurns.map((rawTurn) => rawTurn.id),
        traceIds: source.sourceTraceIds,
        ...(previousPathId ? { previousPathId } : {})
      },
      createdAt: at
    });
  }

  private enqueueGovernedClusterReconcile(
    path: EpisodeExecutionPathRecord,
    reason: string,
    at: string
  ): void {
    const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
    if (!episode || episode.status !== "closed") return;
    const config = this.pipelineConfig();
    const rawTurns = this.deps.repos.runtime.listRawTurnsByEpisode(episode.id, 10_000);
    const source = this.currentSourceState(episode.id, rawTurns);
    this.deps.enqueueJob({
      jobType: "trajectory_window_ingest",
      userId: path.userId,
      sessionId: episode.sessionId,
      episodeId: path.episodeId,
      payload: {
        reason: `source_governance.${reason}`,
        pathId: path.id,
        pathHash: path.pathHash,
        rewardSnapshotHash: rewardHashForEpisode(episode),
        mechanicalWindowHash: config.mechanicalWindowHash,
        clusteringConfigHash: config.clusteringConfigHash,
        windowConfigVersion: config.mechanicalWindowHash,
        reconcileInactivePath: true,
        governanceSnapshotHash: source.sourceSnapshotHash
      },
      createdAt: at
    });
  }

  private proceduralSkillGovernanceDisabled(
    head: TrajectoryWindowClusterRecord
  ): boolean {
    const memory = this.deps.repos.memories.getByKeyIncludingDeleted(
      "Skill",
      proceduralSkillKey(head.userId, head.id)
    );
    if (!memory || memory.userId !== head.userId ||
        memory.properties.internal_info.plugin_algorithm !== "procedural.pattern.skill.v1") {
      return false;
    }
    const governance = memory.properties.internal_info.procedural_governance;
    return isRecord(governance) && governance.disabled === true;
  }

  private rewardHashForWindow(occurrence: TrajectoryWindowOccurrenceV1): string {
    const episode = this.deps.repos.runtime.getEpisode(occurrence.episodeId);
    return episode
      ? rewardHashForEpisode(episode)
      : stableHash({
          episodeId: occurrence.episodeId,
          terminalReward: occurrence.terminalReward ?? null,
          missingEpisode: true
        });
  }

  private rewardHashForWindowRecord(occurrence: TrajectoryWindowOccurrenceRecord): string {
    const episode = this.deps.repos.runtime.getEpisode(occurrence.episodeId);
    return episode
      ? rewardHashForEpisode(episode)
      : stableHash({
          episodeId: occurrence.episodeId,
          terminalReward: occurrence.terminalReward ?? null,
          missingEpisode: true
        });
  }
}

function occurrenceFromRecord(
  record: TrajectoryWindowOccurrenceRecord,
  steps: ExecutionStepLiteV1[]
): TrajectoryWindowOccurrenceV1 {
  return {
    id: record.id,
    schemaVersion: record.schemaVersion,
    episodeId: record.episodeId,
    pathId: record.pathId,
    userId: record.userId,
    ...(record.terminalReward === undefined ? {} : { terminalReward: record.terminalReward }),
    evidenceRole: record.evidenceRole,
    scale: record.scale,
    stride: record.stride,
    startStepIndex: record.startStepIndex,
    endStepIndex: record.endStepIndex,
    semanticText: record.semanticText,
    steps
  };
}

function skillEvidenceStep(step: ExecutionStepLiteV1): ProceduralSkillEvidenceStep {
  return {
    stepId: step.id,
    stepIndex: step.stepIndex,
    ...(step.toolName ? { toolName: step.toolName } : {}),
    intent: step.intent,
    summary: step.summary,
    outcome: step.outcome,
    evidenceRefs: step.evidenceRefs
  };
}

function rewardHashForEpisode(episode: {
  id: string;
  rTask?: number;
  rewardDetail: Record<string, unknown>;
}): string {
  return stableHash({
    schema: "episode-reward-evidence.v1",
    episodeId: episode.id,
    rTask: episode.rTask ?? null,
    rewardDetail: episode.rewardDetail
  });
}

function alignmentRecord(
  alignment: BandedMonotonicMatchResultV1 | undefined
): Record<string, unknown> | undefined {
  return alignment ? { ...alignment } : undefined;
}

function distinctEpisodeIds(members: TrajectoryWindowClusterMemberV1[]): string[] {
  return unique(members.map((member) => member.occurrence.episodeId)).sort();
}

function uniqueEmbeddedWindows(
  windows: readonly EmbeddedTrajectoryWindowV1[]
): EmbeddedTrajectoryWindowV1[] {
  return [...new Map(windows.map((window) => [window.occurrence.id, window])).values()];
}

function orderedWindowsForIngestion(
  windows: readonly EmbeddedTrajectoryWindowV1[]
): EmbeddedTrajectoryWindowV1[] {
  const roleOrder = { support: 0, counterexample: 1, unknown: 2 } as const;
  return uniqueEmbeddedWindows(windows).sort((left, right) =>
    roleOrder[left.occurrence.evidenceRole] - roleOrder[right.occurrence.evidenceRole] ||
    left.occurrence.scale - right.occurrence.scale ||
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex ||
    left.occurrence.id.localeCompare(right.occurrence.id));
}

function payloadString(job: EvolutionJobRecord, key: string): string | undefined {
  const value = job.payload[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function average(values: number[]): number {
  return values.length === 0
    ? 0
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}
