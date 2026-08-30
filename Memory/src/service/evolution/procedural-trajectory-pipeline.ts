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
import {
  LONG_TRAJECTORY_EPISODE_REPRESENTATION_VERSION,
  type LongTrajectoryCandidateRecord,
  type LongTrajectoryEpisodeRepresentationRecord
} from "../../storage/long-trajectory-repository.js";
import type { MemoryRow } from "../../types.js";
import { stableHash } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import {
  EPISODE_PATH_COMPILER_VERSION,
  EpisodePathCompiler
} from "./episode-path-compiler.js";
import {
  PROCEDURAL_LONG_TRAJECTORY_SKILL_PROMPT_VERSION,
  PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION,
  ProceduralPatternSkillMaterializer,
  proceduralSkillKey,
  type ProceduralPatternSkillDraft,
  type ProceduralPatternSkillInput,
  type ProceduralSkillComparisonCandidate,
  type ProceduralSkillCoverageDecision,
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
  orderedEmbeddedWindows,
  selectConstrainedRealMedoid,
  selectMaximalWindowClusters,
  preferredFineClusterPartition,
  resolveFineClusterPartitions,
  tryAbsorbWindowIntoFineCluster,
  type AbsorbFineClusterState,
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
import {
  LONG_TRAJECTORY_MINING_VERSION,
  buildEpisodeTrajectoryFamily,
  extendLongCommonTrajectoryWithProjection,
  mineLongCommonSpanSequences,
  projectEpisodeToReferenceSpans,
  selectMaximalLongCommonTrajectories,
  trajectoryIntentSequenceText,
  type EpisodeTrajectoryDocumentV1,
  type EpisodeTrajectoryFamilyV1,
  type LongCommonTrajectoryV1,
  type LongTrajectoryMiningConfigV1
} from "./long-trajectory-model.js";
import { buildLongTrajectorySkillInput } from "./long-trajectory-skill-input.js";

const STEP_INTENT_REPRESENTATION_VERSION =
  PROCEDURAL_STEP_EMBEDDING_SCHEMA_VERSION;
const WINDOW_INTENT_SEQUENCE_REPRESENTATION_VERSION =
  "trajectory-window-intent-sequence.v1";
const PROCEDURAL_PATTERN_SKILL_ALGORITHM = "procedural.pattern.skill.v1";
const LONG_TRAJECTORY_SKILL_ALGORITHM = "procedural.long-trajectory.skill.v1";
const LEGACY_SKILL_ALGORITHM = "skill.crystallization.v7";
const V2_SKILL_COMPARISON_LIMIT_PER_PHASE = 6;

function isProceduralSkillAlgorithm(value: unknown): boolean {
  return value === PROCEDURAL_PATTERN_SKILL_ALGORITHM ||
    value === LONG_TRAJECTORY_SKILL_ALGORITHM;
}

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

interface LongTrajectoryPipelineConfig {
  mining: LongTrajectoryMiningConfigV1;
  specs: MultiScaleWindowSpec[];
  miningConfigHash: string;
  mechanicalWindowHash: string;
}

interface LongTrajectorySkillCandidateRow {
  trajectory: LongCommonTrajectoryV1;
  skillInput: ProceduralPatternSkillInput;
  /** Present when Episode C semantically matched an already linked Candidate. */
  candidateId?: string;
  candidateStructureKey: string;
  evidenceSignature: string;
  quality: number;
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
    const affectedLongCandidateIds = this.deps.repos.longTrajectory
      .listAffectedCandidateIdsForPath(activePath.id);
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
    for (const candidateId of affectedLongCandidateIds) {
      this.deps.repos.transaction(() => {
        const candidate = this.deps.repos.longTrajectory.getCandidate(candidateId);
        if (!candidate) return;
        if (candidate.activeSkillMemoryId) {
          this.archiveProceduralSkillMemory(
            candidate.activeSkillMemoryId,
            `source-governance-invalidated:${input.reason}`,
            input.at
          );
        }
        if (candidate.status === "retired") return;
        this.deps.repos.longTrajectory.retireCandidate({
          candidateId,
          expectedActiveVersionId: candidate.activeVersionId ?? null,
          retiredAt: input.at
        });
      });
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
        !isProceduralSkillAlgorithm(internal.plugin_algorithm)) {
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
      config.mechanicalWindowHash,
      at,
      true
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
      for (const clusterId of this.absorbWindowsIntoFamilyFineClusters(familyId, config, at)) {
        affectedClusterIds.add(clusterId);
      }
    }

    this.retireUnlinkedCanonicalClusters(affectedClusterIds, at);
    this.reconcileSkillEligibility(pathRecord.userId, affectedClusterIds, config, at);
    this.enqueueLongTrajectoryMining(job, pathRecord, previousPathId);
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
      this.recordProceduralSkillRejection(
        head,
        snapshot?.version,
        "insufficient-common-core",
        at
      );
      return;
    }
    const maximal = this.maximalQualifiedClustersForAffected(
      head.userId,
      new Set([head.id]),
      config
    );
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

    const baseInput = this.buildSkillInput(snapshot, config);
    if (!baseInput) {
      this.recordProceduralSkillRejection(
        head,
        snapshot.version,
        "invalid-positive-evidence",
        at
      );
      return;
    }
    const existingSkillReadOnly = this.existingV2SkillReadOnly(head);
    const previousEpisodeIds = new Set(existingSkillReadOnly?.sourceEpisodeIds ?? []);
    const currentEpisodeIds = new Set(baseInput.supportEpisodeIds);
    const input: ProceduralPatternSkillInput = {
      ...baseInput,
      ...(existingSkillReadOnly ? {
        existingSkillReadOnly,
        evidenceDeltaReadOnly: {
          ...(head.activeSkillVersionId
            ? { previousCandidateVersionId: head.activeSkillVersionId }
            : {}),
          addedEpisodeIds: baseInput.supportEpisodeIds.filter((id) =>
            !previousEpisodeIds.has(id)),
          retainedEpisodeIds: baseInput.supportEpisodeIds.filter((id) =>
            previousEpisodeIds.has(id)),
          removedEpisodeIds: [...previousEpisodeIds].filter((id) =>
            !currentEpisodeIds.has(id)),
          currentEvidenceEpisodeIds: [...baseInput.supportEpisodeIds]
        }
      } : {})
    };
    const result = await this.skillMaterializer.compile(input);
    if (!result.admitted) {
      this.recordProceduralSkillRejection(
        head,
        snapshot.version,
        result.reason,
        at
      );
      return;
    }
    let postDraftCoverageDecision: ProceduralSkillCoverageDecision | undefined;
    if (!existingSkillReadOnly) {
      const comparisonSkills = await this.retrieveV2ComparisonSkills(result.draft);
      const coverage = await this.skillMaterializer.compareDraftCoverage(
        result.draft,
        comparisonSkills
      );
      if (!coverage.ok) {
        this.recordProceduralSkillRejection(
          head,
          snapshot.version,
          coverage.reason,
          at,
          undefined,
          coverage.rawDecision
        );
        return;
      }
      postDraftCoverageDecision = coverage.decision;
      if (coverage.decision.decision === "covered") {
        this.recordProceduralSkillRejection(
          head,
          snapshot.version,
          `covered-by-${coverage.decision.targetRoute.toLowerCase()}:` +
            coverage.decision.targetSkillId,
          at,
          coverage.decision
        );
        return;
      }
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
        skillKey: materialized.memory.memoryKey ?? proceduralSkillKey(head.userId, head.id),
        skillMemoryId: upsert.memory.id,
        payload: {
          admitted: true,
          inductionVersion,
          commonCoreId: snapshot.core!.id,
          patternHash: input.patternHash,
          supportEpisodeIds: input.supportEpisodeIds,
          sourceSpanOccurrenceIds: input.sourceSpanOccurrenceIds,
          skillMemoryId: upsert.memory.id,
          reuseDecision: result.draft.reuseDecision,
          ...(postDraftCoverageDecision ? { postDraftCoverageDecision } : {})
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

  async mineLongTrajectories(job: EvolutionJobRecord): Promise<void> {
    if (!this.deps.config.algorithm.proceduralWindow.enabled ||
        !this.deps.config.algorithm.longTrajectory.enabled) return;
    const pathId = payloadString(job, "pathId") ?? job.targetMemoryId;
    if (!pathId) throw new Error("long_trajectory_mining requires pathId");
    const pathRecord = this.deps.repos.proceduralTrajectory.getPath(pathId);
    if (!pathRecord || pathRecord.userId !== job.userId) {
      throw new Error(`long_trajectory_mining Path not found in user scope: ${pathId}`);
    }
    const previousPathId = payloadString(job, "previousPathId");
    if (previousPathId && previousPathId !== pathRecord.id) {
      this.retireLongTrajectoryCandidatesForPath(
        previousPathId,
        "source-path-superseded",
        new Date().toISOString()
      );
    }
    if (pathRecord.status !== "active" ||
        this.deps.repos.proceduralTrajectory.getActivePath(pathRecord.episodeId)?.id !== pathId) {
      return;
    }
    const seedEpisode = this.deps.repos.runtime.getEpisode(pathRecord.episodeId);
    if (!seedEpisode || seedEpisode.userId !== pathRecord.userId) return;
    const successThreshold = this.deps.config.algorithm.skill.outcomeRTaskSuccessThreshold;
    if (typeof seedEpisode.rTask !== "number" || seedEpisode.rTask < successThreshold) {
      this.retireLongTrajectoryCandidatesForPath(
        pathRecord.id,
        "source-episode-not-successful",
        new Date().toISOString()
      );
      return;
    }
    const config = this.longTrajectoryConfig();
    const requestedConfigHash = payloadString(job, "miningConfigHash");
    if (requestedConfigHash && requestedConfigHash !== config.miningConfigHash) {
      this.enqueueLongTrajectoryMining(job, pathRecord, previousPathId);
      return;
    }

    const activePaths = this.deps.repos.proceduralTrajectory
      .listActivePathsForUser(pathRecord.userId)
      .filter((path) => {
        const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
        return episode?.status === "closed" &&
          typeof episode.rTask === "number" && episode.rTask >= successThreshold;
      });
    if (activePaths.length < config.mining.minSupportEpisodes) return;
    const at = new Date().toISOString();
    const representations = await this.ensureLongTrajectoryRepresentations(activePaths, at);
    const skeletalDocuments = activePaths.flatMap((path) => {
      const representation = representations.get(path.id);
      const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
      if (!representation || !episode) return [];
      return [{
        path: {
          ...path.path,
          ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask })
        },
        goalText: representation.goalText,
        terminalResultText: representation.terminalResultText,
        goalVector: representation.goalVector,
        trajectoryText: representation.trajectoryText,
        trajectoryVector: representation.trajectoryVector,
        windows: []
      } satisfies EpisodeTrajectoryDocumentV1];
    });
    const seed = skeletalDocuments.find((item) => item.path.id === pathRecord.id);
    if (!seed) return;
    const family = buildEpisodeTrajectoryFamily(seed, skeletalDocuments, config.mining);
    if (!family) return;
    const familyPaths = activePaths.filter((path) =>
      family.memberEpisodeIds.includes(path.episodeId));
    const documents: EpisodeTrajectoryDocumentV1[] = [];
    for (const familyPath of familyPaths) {
      const representation = representations.get(familyPath.id);
      if (!representation) continue;
      documents.push(await this.buildLongTrajectoryDocument(
        familyPath,
        representation,
        config,
        at
      ));
    }
    const reference = documents.find((item) =>
      item.path.episodeId === family.referenceEpisodeId);
    if (!reference) return;
    const linkedRows = await this.matchActiveLongTrajectoryCandidates({
      pathRecord,
      seedDocument: reference,
      family,
      documents,
      activePaths,
      representations,
      config,
      at
    });
    const projections = documents
      .filter((item) => item.path.episodeId !== reference.path.episodeId)
      .map((item) => projectEpisodeToReferenceSpans(
        family.id,
        reference,
        item,
        config.mining
      ));
    const maximal = selectMaximalLongCommonTrajectories(
      mineLongCommonSpanSequences(family, documents, projections, config.mining)
    );
    const discoveredRows = maximal.flatMap((trajectory) => {
      const skillInput = buildLongTrajectorySkillInput({
        trajectory,
        documents,
        userId: pathRecord.userId,
        sourceTraceIdsForSteps: (episodeId, steps) =>
          this.sourceTraceIdsForLongTrajectorySteps(episodeId, steps)
      });
      if (!skillInput) return [];
      return [{
        trajectory,
        skillInput,
        candidateStructureKey: trajectory.candidateStructureKey,
        evidenceSignature: longTrajectoryEvidenceSignature(trajectory),
        quality: longTrajectoryQuality(trajectory)
      } satisfies LongTrajectorySkillCandidateRow];
    });
    // Existing Candidate evolution wins when it completely covers a newly
    // discovered path in the seed Episode. Partial overlaps remain visible;
    // disjoint regions can independently become new Candidates.
    const uncoveredDiscoveredRows = discoveredRows.filter((discovered) =>
      !linkedRows.some((linked) => longTrajectoryRowContainsInEpisode(
        linked,
        discovered,
        pathRecord.episodeId
      )));
    const rows = dedupeLongTrajectoryRows([
      ...linkedRows,
      ...uncoveredDiscoveredRows
    ]).sort((left, right) =>
      right.quality - left.quality || left.trajectory.id.localeCompare(right.trajectory.id))
      .slice(0, Math.max(1, Math.floor(
        this.deps.config.algorithm.longTrajectory.maxSkillCandidatesPerEpisode
      )));

    for (const row of rows) {
      const linkedCandidate = row.candidateId
        ? this.deps.repos.longTrajectory.getCandidate(row.candidateId)
        : undefined;
      const resolved = linkedCandidate
        ? { record: linkedCandidate, created: false }
        : this.deps.repos.longTrajectory.resolveCandidate({
            userId: pathRecord.userId,
            algorithmVersion: LONG_TRAJECTORY_MINING_VERSION,
            configHash: config.miningConfigHash,
            structureKey: row.candidateStructureKey,
            createdAt: at
          });
      const previousVersion = resolved.record.activeVersionId
        ? this.deps.repos.longTrajectory.getCandidateVersion(resolved.record.activeVersionId)
        : undefined;
      const previousSupportEpisodeIds = previousVersion?.supportEpisodeIds ?? [];
      const currentEvidenceEpisodeIds = this.validLongTrajectorySupportEpisodeIds(
        row.trajectory.supportEpisodeIds,
        pathRecord.userId,
        successThreshold
      );
      const retainedEpisodeIds = this.validLongTrajectorySupportEpisodeIds(
        previousSupportEpisodeIds,
        pathRecord.userId,
        successThreshold
      );
      const supportEpisodeIds = unique([
        ...retainedEpisodeIds,
        ...currentEvidenceEpisodeIds
      ]).sort();
      const sourcePathIds = supportEpisodeIds.flatMap((episodeId) => {
        const activePath = this.deps.repos.proceduralTrajectory.getActivePath(episodeId);
        return activePath ? [activePath.id] : [];
      });
      const addedEpisodeIds = currentEvidenceEpisodeIds
        .filter((episodeId) => !previousSupportEpisodeIds.includes(episodeId)).sort();
      const removedEpisodeIds = previousSupportEpisodeIds
        .filter((episodeId) => !retainedEpisodeIds.includes(episodeId)).sort();
      const evidenceHash = stableHash({
        version: "long-trajectory-candidate-evidence.v1",
        currentEvidenceSignature: row.evidenceSignature,
        skillEvidencePatternHash: row.skillInput.patternHash,
        supportEpisodeIds,
        sourcePathIds
      });
      const supportHash = stableHash(supportEpisodeIds.map((episodeId) => {
        const episode = this.deps.repos.runtime.getEpisode(episodeId);
        return [
          episodeId,
          episode?.rTask ?? null,
          episode ? rewardHashForEpisode(episode) : "missing-episode"
        ];
      }));
      const evidenceDelta = {
        ...(previousVersion ? { previousCandidateVersionId: previousVersion.id } : {}),
        addedEpisodeIds,
        retainedEpisodeIds,
        removedEpisodeIds,
        currentEvidenceEpisodeIds
      };
      let versionResult;
      try {
        versionResult = this.deps.repos.longTrajectory.saveCandidateVersion({
          candidateId: resolved.record.id,
          expectedActiveVersionId: resolved.record.activeVersionId ?? null,
          structureHash: row.trajectory.structureHash,
          evidenceHash,
          supportHash,
          referenceEpisodeId: row.trajectory.referenceEpisodeId,
          sourcePathIds,
          supportEpisodeIds,
          payload: {
            candidateStructureKey: row.candidateStructureKey,
            evidenceHash,
            family,
            trajectory: row.trajectory,
            skillInput: {
              ...row.skillInput,
              supportEpisodeIds
            },
            evidenceDelta,
            quality: row.quality,
            miningConfigHash: config.miningConfigHash
          },
          createdAt: at
        });
      } catch (error) {
        if (String(error).includes("long trajectory CAS conflict")) continue;
        throw error;
      }
      const version = versionResult.record;
      const activeSkillDecision = resolved.record.activeSkillVersionId
        ? this.deps.repos.longTrajectory.getSkillVersion(resolved.record.activeSkillVersionId)
        : undefined;
      if (!versionResult.created && activeSkillDecision?.candidateVersionId === version.id) {
        continue;
      }
      this.deps.enqueueJob({
        jobType: "long_trajectory_skill_induction",
        userId: pathRecord.userId,
        sessionId: seedEpisode.sessionId,
        episodeId: pathRecord.episodeId,
        targetMemoryId: resolved.record.id,
        payload: {
          candidateId: resolved.record.id,
          candidateVersionId: version.id,
          inductionVersion: PROCEDURAL_LONG_TRAJECTORY_SKILL_PROMPT_VERSION
        },
        createdAt: at
      });
    }
  }

  async induceLongTrajectorySkill(job: EvolutionJobRecord): Promise<void> {
    if (!this.deps.config.algorithm.longTrajectory.enabled) return;
    const candidateId = payloadString(job, "candidateId") ?? job.targetMemoryId;
    const candidateVersionId = payloadString(job, "candidateVersionId");
    if (!candidateId || !candidateVersionId) {
      throw new Error(
        "long_trajectory_skill_induction requires candidateId and candidateVersionId"
      );
    }
    const candidate = this.deps.repos.longTrajectory.getCandidate(candidateId);
    const version = this.deps.repos.longTrajectory.getCandidateVersion(candidateVersionId);
    if (!candidate || !version || candidate.userId !== job.userId ||
        candidate.status !== "active" || candidate.activeVersionId !== version.id ||
        version.candidateId !== candidate.id) return;
    if (this.longTrajectorySkillGovernanceDisabled(candidate)) {
      this.rejectLongTrajectorySkill(candidate, version.id, "governance-disabled", job);
      return;
    }
    const storedInput = version.payload.skillInput;
    if (!isRecord(storedInput)) {
      this.recordLongTrajectorySkillRejection(
        candidate,
        version.id,
        "invalid-stored-skill-input",
        job
      );
      return;
    }
    const evidenceDelta = isRecord(version.payload.evidenceDelta)
      ? version.payload.evidenceDelta
      : undefined;
    const existingSkillReadOnly = this.existingLongTrajectorySkillReadOnly(
      candidate,
      version.supportEpisodeIds,
      stringArray(evidenceDelta?.removedEpisodeIds).length > 0
    );
    const input = {
      ...storedInput,
      patternVersionId: version.id,
      clusterId: candidate.id,
      clusterVersionId: version.id,
      ...(existingSkillReadOnly ? { existingSkillReadOnly } : {}),
      ...(evidenceDelta ? {
        evidenceDeltaReadOnly: {
          ...(typeof evidenceDelta.previousCandidateVersionId === "string"
            ? { previousCandidateVersionId: evidenceDelta.previousCandidateVersionId }
            : {}),
          addedEpisodeIds: stringArray(evidenceDelta.addedEpisodeIds),
          retainedEpisodeIds: stringArray(evidenceDelta.retainedEpisodeIds),
          removedEpisodeIds: stringArray(evidenceDelta.removedEpisodeIds),
          currentEvidenceEpisodeIds: stringArray(evidenceDelta.currentEvidenceEpisodeIds)
        }
      } : {})
    } as unknown as ProceduralPatternSkillInput;
    if (input.origin?.kind !== "long_trajectory" ||
        !Array.isArray(input.evidence) || input.evidence.length < 2) {
      this.recordLongTrajectorySkillRejection(
        candidate,
        version.id,
        "invalid-stored-skill-input",
        job
      );
      return;
    }
    const result = await this.skillMaterializer.compile(input);
    if (!result.admitted) {
      this.recordLongTrajectorySkillRejection(candidate, version.id, result.reason, job);
      return;
    }
    // A panel archive/delete can happen while the Skill LLM is running. Check
    // once before the write transaction for a fast no-op, then again inside
    // the transaction to close the final check-to-upsert race.
    if (this.longTrajectorySkillGovernanceDisabled(candidate)) {
      this.rejectLongTrajectorySkill(candidate, version.id, "governance-disabled", job);
      return;
    }
    const committedAt = new Date().toISOString();
    this.deps.repos.transaction(() => {
      const current = this.deps.repos.longTrajectory.getCandidate(candidate.id);
      if (!current || current.status !== "active" || current.activeVersionId !== version.id) return;
      if (this.longTrajectorySkillGovernanceDisabled(current)) {
        this.rejectLongTrajectorySkillInTransaction(
          current,
          version.id,
          "governance-disabled",
          job,
          committedAt
        );
        return;
      }
      const materialized = this.skillMaterializer.materializeDraft(result.draft, committedAt);
      const upsert = this.deps.upsertEvolutionMemory(materialized.memory);
      this.deps.repos.longTrajectory.saveSkillVersion({
        candidateId: current.id,
        candidateVersionId: version.id,
        expectedActiveSkillVersionId: current.activeSkillVersionId ?? null,
        skillKey: proceduralSkillKey(current.userId, current.id),
        skillMemoryId: upsert.memory.id,
        contentHash: materialized.contentHash,
        payload: {
          admitted: true,
          candidateVersionId: version.id,
          supportEpisodeIds: input.supportEpisodeIds,
          sourceSpanOccurrenceIds: input.sourceSpanOccurrenceIds,
          skillMemoryId: upsert.memory.id
        },
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
        source: "worker.long_trajectory_induction.v1",
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
            reason: "procedural.long-trajectory.skill.upserted",
            contentHash: upsert.memory.contentHash
          },
          createdAt: committedAt
        });
      }
    });
  }

  private existingV2SkillReadOnly(
    head: TrajectoryWindowClusterRecord
  ): ProceduralPatternSkillInput["existingSkillReadOnly"] {
    if (!head.activeSkillMemoryId) return undefined;
    const memory = this.deps.repos.memories.get(head.activeSkillMemoryId);
    if (!memory || memory.userId !== head.userId ||
        memory.status === "archived" || memory.status === "deleted" ||
        memory.properties.internal_info.plugin_algorithm !==
          "procedural.pattern.skill.v1") {
      return undefined;
    }
    const skill = skillMetaFromMemory(memory);
    if (!skill || skill.status === "archived") return undefined;
    const internal = memory.properties.internal_info;
    const internalSkill = isRecord(internal.skill) ? internal.skill : {};
    const procedureJson = isRecord(internal.procedure_json)
      ? internal.procedure_json
      : isRecord(internalSkill.procedure_json)
        ? internalSkill.procedure_json
        : {};
    return {
      memoryId: memory.id,
      memoryVersion: memory.version,
      name: skill.name,
      invocationGuide: skill.invocationGuide,
      procedureJson,
      sourceEpisodeIds: stringArray(internal.source_episode_ids),
      sourceTraceIds: stringArray(internal.source_trace_ids),
      sourceSpanOccurrenceIds: stringArray(internal.source_span_occurrence_ids)
    };
  }

  private existingLongTrajectorySkillReadOnly(
    candidate: LongTrajectoryCandidateRecord,
    supportEpisodeIds: readonly string[],
    discardPriorOccurrenceIds: boolean
  ): ProceduralPatternSkillInput["existingSkillReadOnly"] {
    if (!candidate.activeSkillMemoryId) return undefined;
    const memory = this.deps.repos.memories.get(candidate.activeSkillMemoryId);
    if (!memory || memory.userId !== candidate.userId ||
        memory.status === "archived" || memory.status === "deleted" ||
        memory.properties.internal_info.plugin_algorithm !== LONG_TRAJECTORY_SKILL_ALGORITHM) {
      return undefined;
    }
    const skill = skillMetaFromMemory(memory);
    if (!skill) return undefined;
    const internal = memory.properties.internal_info;
    const internalSkill = isRecord(internal.skill) ? internal.skill : {};
    const procedureJson = isRecord(internal.procedure_json)
      ? internal.procedure_json
      : isRecord(internalSkill.procedure_json)
        ? internalSkill.procedure_json
        : {};
    const supportSet = new Set(supportEpisodeIds);
    const sourceTraceIds = stringArray(internal.source_trace_ids).filter((traceId) => {
      const trace = this.deps.traceMeta(this.deps.repos.memories.get(traceId));
      return Boolean(trace?.episodeId && supportSet.has(trace.episodeId));
    });
    return {
      memoryId: memory.id,
      memoryVersion: memory.version,
      name: skill.name,
      invocationGuide: skill.invocationGuide,
      procedureJson,
      sourceEpisodeIds: stringArray(internal.source_episode_ids)
        .filter((episodeId) => supportSet.has(episodeId)),
      sourceTraceIds,
      sourceSpanOccurrenceIds: discardPriorOccurrenceIds
        ? []
        : stringArray(internal.source_span_occurrence_ids)
    };
  }

  private validLongTrajectorySupportEpisodeIds(
    episodeIds: readonly string[],
    userId: string,
    successThreshold: number
  ): string[] {
    return unique([...episodeIds]).filter((episodeId) => {
      const episode = this.deps.repos.runtime.getEpisode(episodeId);
      const activePath = this.deps.repos.proceduralTrajectory.getActivePath(episodeId);
      return episode?.userId === userId && episode.status === "closed" &&
        typeof episode.rTask === "number" && episode.rTask >= successThreshold &&
        Boolean(activePath);
    }).sort();
  }

  private longTrajectoryConfig(): LongTrajectoryPipelineConfig {
    const configured = this.deps.config.algorithm.longTrajectory;
    const scales = [...configured.scales]
      .sort((left, right) => left.length - right.length)
      .map((item) => ({ ...item }));
    const mining: LongTrajectoryMiningConfigV1 = {
      episodeRecallLimit: Math.max(1, Math.floor(configured.episodeRecallLimit)),
      minEpisodeSimilarity: configured.minEpisodeSimilarity,
      minGoalSimilarity: configured.minGoalSimilarity,
      goalWeight: configured.goalWeight,
      trajectoryWeight: configured.trajectoryWeight,
      windowTopK: Math.max(1, Math.floor(configured.windowTopK)),
      coarseThresholds: Object.fromEntries(scales.map((item) => [
        item.length,
        item.coarseSimilarityThreshold
      ])),
      minSupportEpisodes: Math.max(2, Math.floor(configured.minSupportEpisodes)),
      minSpanSequenceLength: Math.max(2, Math.floor(configured.minSpanSequenceLength)),
      minTrajectorySpanSteps: Math.max(2, Math.floor(configured.minTrajectorySpanSteps)),
      minEpisodeCoverage: configured.minEpisodeCoverage
    };
    const mechanicalWindowHash = stableHash({
      version: "long-trajectory-window-basis.v1",
      specs: scales.map(({ length, stride }) => ({ length, stride })),
      stepRepresentationVersion: STEP_INTENT_REPRESENTATION_VERSION,
      coarseRepresentationVersion: WINDOW_INTENT_SEQUENCE_REPRESENTATION_VERSION
    });
    const miningConfigHash = stableHash({
      algorithmVersion: LONG_TRAJECTORY_MINING_VERSION,
      representationVersion: LONG_TRAJECTORY_EPISODE_REPRESENTATION_VERSION,
      mechanicalWindowHash,
      mining
    });
    return {
      mining,
      specs: scales.map(({ length, stride }) => ({ length, stride })),
      miningConfigHash,
      mechanicalWindowHash
    };
  }

  private enqueueLongTrajectoryMining(
    sourceJob: EvolutionJobRecord,
    path: EpisodeExecutionPathRecord,
    previousPathId?: string
  ): void {
    if (!this.deps.config.algorithm.longTrajectory.enabled) return;
    const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
    if (!episode) return;
    const config = this.longTrajectoryConfig();
    this.deps.enqueueJob({
      jobType: "long_trajectory_mining",
      userId: path.userId,
      sessionId: episode.sessionId,
      episodeId: path.episodeId,
      targetMemoryId: path.id,
      payload: {
        pathId: path.id,
        pathHash: path.pathHash,
        miningConfigHash: config.miningConfigHash,
        rewardSnapshotHash: payloadString(sourceJob, "rewardSnapshotHash") ??
          rewardHashForEpisode(episode),
        ...(previousPathId && previousPathId !== path.id ? { previousPathId } : {})
      },
      createdAt: new Date().toISOString()
    });
  }

  private async ensureLongTrajectoryRepresentations(
    paths: EpisodeExecutionPathRecord[],
    at: string
  ): Promise<Map<string, LongTrajectoryEpisodeRepresentationRecord>> {
    const signature = this.embeddingSignature();
    const prepared = paths.flatMap((path) => {
      const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
      if (!episode) return [];
      const rawTurns = this.deps.repos.runtime.listRawTurnsByEpisode(path.episodeId, 10_000)
        .filter((turn) => !turn.deletedAt && !turn.redactedAt);
      const firstUser = rawTurns.find((turn) => turn.userText?.trim())?.userText;
      const lastAssistant = [...rawTurns].reverse()
        .find((turn) => turn.assistantText?.trim())?.assistantText;
      const goalText = [episode.title, episode.summary, firstUser]
        .filter((item): item is string => Boolean(item?.trim())).join("\n") ||
        `Episode ${episode.id}`;
      const terminalResultText = lastAssistant?.trim() || episode.summary?.trim() ||
        "Episode completed successfully.";
      const trajectoryText = trajectoryIntentSequenceText(path.path);
      return [{ path, episode, goalText, terminalResultText, trajectoryText }];
    });
    const missing = prepared.filter((item) => {
      const existing = this.deps.repos.longTrajectory.getEpisodeRepresentation({
        pathId: item.path.id,
        representationVersion: LONG_TRAJECTORY_EPISODE_REPRESENTATION_VERSION,
        embeddingSignature: signature
      });
      return !existing || existing.goalHash !== stableHash(item.goalText) ||
        existing.trajectoryHash !== stableHash(item.trajectoryText);
    });
    const [goalVectors, trajectoryVectors] = missing.length > 0
      ? await Promise.all([
          this.deps.embedder.embed(missing.map((item) => item.goalText), "document"),
          this.deps.embedder.embed(missing.map((item) => item.trajectoryText), "document")
        ])
      : [[], []];
    if (goalVectors.length !== missing.length || trajectoryVectors.length !== missing.length) {
      throw new Error("long trajectory Episode embedding response count mismatch");
    }
    for (const [index, item] of missing.entries()) {
      this.deps.repos.longTrajectory.upsertEpisodeRepresentation({
        pathId: item.path.id,
        episodeId: item.episode.id,
        userId: item.episode.userId,
        representationVersion: LONG_TRAJECTORY_EPISODE_REPRESENTATION_VERSION,
        embeddingSignature: signature,
        goalText: item.goalText,
        terminalResultText: item.terminalResultText,
        goalHash: stableHash(item.goalText),
        goalVector: unitVector(goalVectors[index] ?? []),
        trajectoryText: item.trajectoryText,
        trajectoryHash: stableHash(item.trajectoryText),
        trajectoryVector: unitVector(trajectoryVectors[index] ?? []),
        createdAt: at
      });
    }
    return new Map(prepared.flatMap((item) => {
      const record = this.deps.repos.longTrajectory.getEpisodeRepresentation({
        pathId: item.path.id,
        representationVersion: LONG_TRAJECTORY_EPISODE_REPRESENTATION_VERSION,
        embeddingSignature: signature
      });
      return record ? [[item.path.id, record] as const] : [];
    }));
  }

  /**
   * Candidate-first V3 routing.
   *
   * Episode C uses both recalled-Episode reverse lookup and direct active
   * Candidate lookup. C is projected onto each Candidate's retained reference
   * Span sequence; a complete monotonic match evolves that Candidate in place.
   * The caller still runs C-centred discovery for regions not fully covered by
   * a matched Candidate.
   */
  private async matchActiveLongTrajectoryCandidates(input: {
    pathRecord: EpisodeExecutionPathRecord;
    seedDocument: EpisodeTrajectoryDocumentV1;
    family: EpisodeTrajectoryFamilyV1;
    documents: readonly EpisodeTrajectoryDocumentV1[];
    activePaths: readonly EpisodeExecutionPathRecord[];
    representations: ReadonlyMap<string, LongTrajectoryEpisodeRepresentationRecord>;
    config: LongTrajectoryPipelineConfig;
    at: string;
  }): Promise<LongTrajectorySkillCandidateRow[]> {
    const recalledEpisodeIds = input.family.memberEpisodeIds.filter((episodeId) =>
      episodeId !== input.pathRecord.episodeId);
    const linkedCandidates = this.deps.repos.longTrajectory.listActiveCandidatesLinkedToEpisodes({
      userId: input.pathRecord.userId,
      algorithmVersion: LONG_TRAJECTORY_MINING_VERSION,
      configHash: input.config.miningConfigHash,
      episodeIds: recalledEpisodeIds
    });
    const directlyRecalledCandidates = this.deps.repos.longTrajectory.listActiveCandidates({
      userId: input.pathRecord.userId,
      algorithmVersion: LONG_TRAJECTORY_MINING_VERSION,
      configHash: input.config.miningConfigHash
    });
    const candidateById = new Map<string, LongTrajectoryCandidateRecord>();
    for (const candidate of [...linkedCandidates, ...directlyRecalledCandidates]) {
      if (!candidateById.has(candidate.id)) candidateById.set(candidate.id, candidate);
    }
    const candidates = [...candidateById.values()];
    const pathByEpisodeId = new Map(input.activePaths.map((path) => [path.episodeId, path]));
    const documentByEpisodeId = new Map(input.documents.map((document) => [
      document.path.episodeId,
      document
    ]));
    const rows: LongTrajectorySkillCandidateRow[] = [];
    for (const candidate of candidates) {
      if (!candidate.activeVersionId) continue;
      const version = this.deps.repos.longTrajectory.getCandidateVersion(
        candidate.activeVersionId
      );
      const trajectory = longCommonTrajectoryFromPayload(version?.payload.trajectory);
      if (!version || !trajectory || trajectory.requiredSpans.length === 0) continue;
      const referencePath = pathByEpisodeId.get(trajectory.referenceEpisodeId);
      if (!referencePath) continue;
      let referenceDocument = documentByEpisodeId.get(trajectory.referenceEpisodeId);
      if (!referenceDocument) {
        const representation = input.representations.get(referencePath.id);
        if (!representation) continue;
        referenceDocument = await this.buildLongTrajectoryDocument(
          referencePath,
          representation,
          input.config,
          input.at
        );
        documentByEpisodeId.set(referenceDocument.path.episodeId, referenceDocument);
      }
      const retainedSpanIds = new Set(trajectory.requiredSpans.map((span) =>
        span.referenceSpanId));
      const retainedReference = {
        ...referenceDocument,
        windows: referenceDocument.windows.filter((window) =>
          retainedSpanIds.has(window.occurrence.id))
      };
      if (retainedReference.windows.length !== retainedSpanIds.size) continue;
      const projection = projectEpisodeToReferenceSpans(
        `candidate_projection_${candidate.id}`,
        retainedReference,
        input.seedDocument,
        input.config.mining
      );
      const matchedReferenceIds = new Set(projection.matches.map((match) =>
        match.referenceSpanId));
      const matchedStepCount = projection.matches.reduce((sum, match) =>
        sum + match.referenceEndStepIndex - match.referenceStartStepIndex + 1, 0);
      const completeCandidateMatch = matchedReferenceIds.size === retainedSpanIds.size &&
        [...retainedSpanIds].every((spanId) => matchedReferenceIds.has(spanId)) &&
        projection.matches.length >= input.config.mining.minSpanSequenceLength &&
        matchedStepCount >= input.config.mining.minTrajectorySpanSteps &&
        projection.referenceCoverage + Number.EPSILON >=
          input.config.mining.minEpisodeCoverage &&
        projection.episodeCoverage + Number.EPSILON >=
          input.config.mining.minEpisodeCoverage;
      if (!completeCandidateMatch) continue;
      const evolved = extendLongCommonTrajectoryWithProjection({
        trajectory,
        episode: input.seedDocument,
        projection
      });
      const skillInputDocuments = uniqueDocuments([
        ...documentByEpisodeId.values(),
        retainedReference,
        input.seedDocument
      ]);
      const skillInput = buildLongTrajectorySkillInput({
        trajectory: evolved,
        documents: skillInputDocuments,
        userId: input.pathRecord.userId,
        sourceTraceIdsForSteps: (episodeId, steps) =>
          this.sourceTraceIdsForLongTrajectorySteps(episodeId, steps)
      });
      if (!skillInput) continue;
      rows.push({
        candidateId: candidate.id,
        trajectory: evolved,
        skillInput,
        candidateStructureKey: candidate.structureKey,
        evidenceSignature: longTrajectoryEvidenceSignature(evolved),
        quality: longTrajectoryQuality(evolved)
      });
    }
    return rows;
  }

  private async buildLongTrajectoryDocument(
    path: EpisodeExecutionPathRecord,
    representation: LongTrajectoryEpisodeRepresentationRecord,
    config: LongTrajectoryPipelineConfig,
    at: string
  ): Promise<EpisodeTrajectoryDocumentV1> {
    const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
    const rewardPath: EpisodeExecutionPathLiteV1 = {
      ...path.path,
      ...(episode?.rTask === undefined ? {} : { terminalReward: episode.rTask })
    };
    const stepVectorById = await this.ensureStepEmbeddings(path, at);
    const occurrences = buildTrajectoryWindows([rewardPath], config.specs);
    const windows = await this.persistAndEmbedWindows(
      occurrences,
      stepVectorById,
      config.mechanicalWindowHash,
      at,
      false
    );
    const v2Scales = new Set(this.pipelineConfig().specs.map((item) => item.length));
    const v3Only = occurrences.filter((item) => !v2Scales.has(item.scale));
    if (v3Only.length > 0) {
      await this.persistAndEmbedWindows(
        v3Only,
        stepVectorById,
        config.mechanicalWindowHash,
        at,
        true
      );
    }
    return {
      path: rewardPath,
      goalText: representation.goalText,
      terminalResultText: representation.terminalResultText,
      goalVector: representation.goalVector,
      trajectoryText: representation.trajectoryText,
      trajectoryVector: representation.trajectoryVector,
      windows
    };
  }

  private sourceTraceIdsForLongTrajectorySteps(
    episodeId: string,
    steps: readonly ExecutionStepLiteV1[]
  ): string[] {
    const episode = this.deps.repos.runtime.getEpisode(episodeId);
    if (!episode) return [];
    const rawTurnIds = new Set(steps.map((step) => step.rawTurnId));
    return this.deps.repos.memories.getMany(episode.l1MemoryIds).flatMap((memory) => {
      const trace = this.deps.traceMeta(memory);
      return trace?.episodeId === episodeId && trace.rawTurnId && rawTurnIds.has(trace.rawTurnId)
        ? [trace.id]
        : [];
    });
  }

  private retireLongTrajectoryCandidatesForPath(pathId: string, reason: string, at: string): void {
    for (const candidateId of this.deps.repos.longTrajectory.listAffectedCandidateIdsForPath(pathId)) {
      this.deps.repos.transaction(() => {
        const candidate = this.deps.repos.longTrajectory.getCandidate(candidateId);
        if (!candidate) return;
        if (candidate.activeSkillMemoryId) {
          this.archiveProceduralSkillMemory(candidate.activeSkillMemoryId, reason, at);
        }
        if (candidate.status === "retired") return;
        this.deps.repos.longTrajectory.retireCandidate({
          candidateId,
          expectedActiveVersionId: candidate.activeVersionId ?? null,
          retiredAt: at
        });
      });
    }
  }

  private rejectLongTrajectorySkill(
    candidate: LongTrajectoryCandidateRecord,
    candidateVersionId: string,
    reason: string,
    job: EvolutionJobRecord
  ): void {
    const at = new Date().toISOString();
    this.deps.repos.transaction(() => {
      const current = this.deps.repos.longTrajectory.getCandidate(candidate.id);
      if (!current || current.status !== "active" ||
          current.activeVersionId !== candidateVersionId) return;
      this.rejectLongTrajectorySkillInTransaction(
        current,
        candidateVersionId,
        reason,
        job,
        at
      );
    });
  }

  private recordLongTrajectorySkillRejection(
    candidate: LongTrajectoryCandidateRecord,
    candidateVersionId: string,
    reason: string,
    job: EvolutionJobRecord
  ): void {
    const at = new Date().toISOString();
    this.deps.repos.transaction(() => {
      const current = this.deps.repos.longTrajectory.getCandidate(candidate.id);
      if (!current || current.status !== "active" ||
          current.activeVersionId !== candidateVersionId) return;
      this.deps.repos.longTrajectory.recordRejectedSkillVersion({
        candidateId: current.id,
        candidateVersionId,
        expectedActiveSkillVersionId: current.activeSkillVersionId ?? null,
        skillKey: proceduralSkillKey(current.userId, current.id),
        contentHash: stableHash({ candidateVersionId, admitted: false, reason }),
        payload: {
          admitted: false,
          candidateVersionId,
          reason,
          jobId: job.id,
          preservedActiveSkillVersionId: current.activeSkillVersionId ?? null,
          preservedActiveSkillMemoryId: current.activeSkillMemoryId ?? null
        },
        createdAt: at
      });
    });
  }

  /** Caller must hold the repository write transaction. */
  private rejectLongTrajectorySkillInTransaction(
    candidate: LongTrajectoryCandidateRecord,
    candidateVersionId: string,
    reason: string,
    job: EvolutionJobRecord,
    at: string
  ): void {
    if (candidate.activeSkillMemoryId) {
      this.archiveProceduralSkillMemory(candidate.activeSkillMemoryId, reason, at);
    }
    this.deps.repos.longTrajectory.saveSkillVersion({
      candidateId: candidate.id,
      candidateVersionId,
      expectedActiveSkillVersionId: candidate.activeSkillVersionId ?? null,
      skillKey: proceduralSkillKey(candidate.userId, candidate.id),
      contentHash: stableHash({ candidateVersionId, admitted: false, reason }),
      payload: { admitted: false, candidateVersionId, reason, jobId: job.id },
      createdAt: at
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
    windowConfigHash: string,
    at: string,
    refreshExistingEvidence: boolean
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
      if (existing && !refreshExistingEvidence) continue;
      const coarseVector = existing?.coarseVector ??
        newVectorByOccurrenceId.get(occurrence.id) ?? [];
      this.deps.repos.proceduralTrajectory.insertWindow({
        occurrence,
        windowConfigHash,
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

  /**
   * Daily ingest: absorb new Family members into existing Fine clusters.
   * Cluster ids stay put when a window only grows a stable medoid. Full remesh
   * remains on the path-reconcile path where members can leave.
   */
  private absorbWindowsIntoFamilyFineClusters(
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
    const familyMembers = this.activeFamilyMembers(revision.id);
    if (familyMembers.length === 0) return affectedClusterIds;
    const familyMedoid = familyMembers.find((member) =>
      member.occurrence.id === revision.medoidOccurrenceId);
    if (!familyMedoid) return affectedClusterIds;
    const fineConfig = config.fineConfigByScale.get(head.scale);
    if (!fineConfig) throw new Error(`fine config missing for Span-${head.scale}`);

    const previousRevision = this.lastLinkedFamilyRevision(familyId, revision.id);
    const live: Array<{
      clusterId?: string;
      previousRevisionId?: string;
      state: AbsorbFineClusterState;
    }> = [];
    const assigned = new Set<string>();
    if (previousRevision) {
      for (const link of this.deps.repos.proceduralTrajectory.listFamilyClusterLinks(
        previousRevision.id
      )) {
        const canonical = this.deps.repos.proceduralTrajectory.getClusterCanonicalKey(
          link.canonicalKeyId
        );
        if (!canonical) continue;
        const cluster = this.deps.repos.proceduralTrajectory.getClusterHead(canonical.clusterId);
        if (!cluster || cluster.status !== "active" || !cluster.activeVersionId) continue;
        const version = this.deps.repos.proceduralTrajectory.getClusterVersion(
          cluster.activeVersionId
        );
        if (!version) continue;
        const members = this.deps.repos.proceduralTrajectory.listClusterMembers(version.id)
          .flatMap((member) => {
            const embedded = familyMembers.find((item) =>
              item.occurrence.id === member.occurrenceId);
            return embedded ? [embedded] : [];
          });
        if (members.length === 0) continue;
        const partitions = resolveFineClusterPartitions(members, fineConfig, {
          preferredMedoidId: version.medoidOccurrenceId,
          medoidSwitchMargin: config.medoidSwitchMargin,
          medoidUpdateCount: finiteNumber(version.metrics.medoidUpdateCount) ?? 0
        });
        const preferred = preferredFineClusterPartition(
          partitions,
          version.medoidOccurrenceId
        );
        for (const partition of partitions) {
          live.push({
            clusterId: partition === preferred ? cluster.id : undefined,
            previousRevisionId: previousRevision.id,
            state: partition
          });
          for (const member of partition.members) assigned.add(member.occurrence.id);
        }
        affectedClusterIds.add(cluster.id);
      }
    }

    for (const window of orderedEmbeddedWindows(familyMembers.filter((member) =>
      !assigned.has(member.occurrence.id)))) {
      const ranked = live
        .map((cluster) => ({
          cluster,
          match: bandedMonotonicMatch(window.stepVectors, cluster.state.medoid.stepVectors, fineConfig)
        }))
        .filter((row) => row.match.admitted)
        .sort((left, right) => right.match.score - left.match.score ||
          (left.cluster.clusterId ?? "").localeCompare(right.cluster.clusterId ?? ""));
      let absorbed = false;
      for (const row of ranked) {
        const decision = tryAbsorbWindowIntoFineCluster(
          row.cluster.state,
          window,
          fineConfig,
          config.medoidSwitchMargin
        );
        if (!decision.accepted) continue;
        row.cluster.state = decision.cluster;
        absorbed = true;
        break;
      }
      if (absorbed) continue;
      if (window.occurrence.evidenceRole !== "support") continue;
      live.push({
        state: {
          medoid: window,
          members: [window],
          medoidUpdateCount: 0
        }
      });
    }

    for (const cluster of this.expandResolvedFineClusters(live, fineConfig, config.medoidSwitchMargin)) {
      const clusterId = this.commitAbsorbedFamilyFineCluster({
        clusterId: cluster.clusterId,
        previousRevisionId: cluster.previousRevisionId,
        state: cluster.state,
        family: head,
        revision,
        familyMedoid,
        config,
        at
      });
      affectedClusterIds.add(clusterId);
    }
    return affectedClusterIds;
  }

  private expandResolvedFineClusters(
    live: Array<{
      clusterId?: string;
      previousRevisionId?: string;
      state: AbsorbFineClusterState;
    }>,
    fineConfig: BandedMonotonicMatchConfig,
    medoidSwitchMargin: number
  ): Array<{
    clusterId?: string;
    previousRevisionId?: string;
    state: AbsorbFineClusterState;
  }> {
    const expanded: Array<{
      clusterId?: string;
      previousRevisionId?: string;
      state: AbsorbFineClusterState;
    }> = [];
    for (const cluster of live) {
      const partitions = resolveFineClusterPartitions(cluster.state.members, fineConfig, {
        preferredMedoidId: cluster.state.medoid.occurrence.id,
        medoidSwitchMargin,
        medoidUpdateCount: cluster.state.medoidUpdateCount
      });
      const preferred = preferredFineClusterPartition(
        partitions,
        cluster.state.medoid.occurrence.id
      );
      for (const partition of partitions) {
        expanded.push({
          clusterId: partition === preferred ? cluster.clusterId : undefined,
          previousRevisionId: cluster.previousRevisionId,
          state: partition
        });
      }
    }
    return expanded;
  }

  private lastLinkedFamilyRevision(
    familyId: string,
    excludeRevisionId?: string
  ): TrajectoryWindowFamilyRevisionRecord | undefined {
    for (const revision of this.deps.repos.proceduralTrajectory.listFamilyRevisions(familyId)) {
      if (excludeRevisionId && revision.id === excludeRevisionId) continue;
      if (this.deps.repos.proceduralTrajectory.listFamilyClusterLinks(revision.id).length > 0) {
        return revision;
      }
    }
    return undefined;
  }

  private commitAbsorbedFamilyFineCluster(input: {
    clusterId?: string;
    previousRevisionId?: string;
    state: AbsorbFineClusterState;
    family: TrajectoryWindowFamilyRecord;
    revision: TrajectoryWindowFamilyRevisionRecord;
    familyMedoid: EmbeddedTrajectoryWindowV1;
    config: PipelineConfig;
    at: string;
  }): string {
    const seed = input.state.members[0] ?? input.state.medoid;
    let clusterId = input.clusterId;
    let canonicalKeyId: string | undefined;
    if (!clusterId) {
      const created = this.deps.repos.proceduralTrajectory.resolveCanonicalClusterHead({
        userId: input.family.userId,
        scale: input.family.scale,
        algorithmVersion: input.config.algorithmVersion,
        configHash: input.config.clusteringConfigHash,
        embeddingSignature: input.family.embeddingSignature,
        evidenceSignature: fineEvidenceSignature({
          scale: input.family.scale,
          members: [{ occurrence: seed.occurrence }]
        } as TrajectoryWindowClusterV1),
        seedOccurrenceId: seed.occurrence.id,
        createdAt: input.at
      });
      clusterId = created.cluster.id;
      canonicalKeyId = created.canonicalKey.id;
    } else {
      canonicalKeyId = this.canonicalKeyIdForCluster(clusterId, input.previousRevisionId);
    }
    const currentHead = this.deps.repos.proceduralTrajectory.getClusterHead(clusterId)!;
    if (!canonicalKeyId) {
      const created = this.deps.repos.proceduralTrajectory.resolveCanonicalClusterHead({
        userId: input.family.userId,
        scale: input.family.scale,
        algorithmVersion: input.config.algorithmVersion,
        configHash: input.config.clusteringConfigHash,
        embeddingSignature: input.family.embeddingSignature,
        evidenceSignature: fineEvidenceSignature({
          scale: input.family.scale,
          members: [{ occurrence: seed.occurrence }]
        } as TrajectoryWindowClusterV1),
        seedOccurrenceId: seed.occurrence.id,
        createdAt: input.at
      });
      if (created.cluster.id !== clusterId) {
        throw new Error(`absorbed Fine cluster lost its canonical key: ${clusterId}`);
      }
      canonicalKeyId = created.canonicalKey.id;
    }
    const committed = this.deps.repos.proceduralTrajectory.commitClusterVersion({
      clusterId,
      expectedActiveVersionId: currentHead.activeVersionId ?? null,
      medoidOccurrenceId: input.state.medoid.occurrence.id,
      members: input.state.members.map((member) => ({
        occurrenceId: member.occurrence.id,
        rewardHash: this.rewardHashForWindow(member.occurrence),
        coarseSimilarity: cosineSimilarity(member.coarseVector, input.familyMedoid.coarseVector),
        alignment: alignmentRecord(
          member.occurrence.id === input.state.medoid.occurrence.id
            ? selfBandedMonotonicMatch(member.stepVectors, input.config.fineConfigByScale.get(input.family.scale)!)
            : bandedMonotonicMatch(
              member.stepVectors,
              input.state.medoid.stepVectors,
              input.config.fineConfigByScale.get(input.family.scale)!
            )
        )
      })),
      metrics: {
        algorithmVersion: input.config.algorithmVersion,
        configHash: input.config.clusteringConfigHash,
        familyRevisionId: input.revision.id,
        medoidUpdateCount: input.state.medoidUpdateCount,
        absorb: true
      },
      createdAt: input.at
    });
    this.deps.repos.proceduralTrajectory.linkFamilyRevisionToCluster({
      familyRevisionId: input.revision.id,
      canonicalKeyId,
      clusterVersionId: committed.record.id,
      createdAt: input.at
    });
    return clusterId;
  }

  private canonicalKeyIdForCluster(
    clusterId: string,
    previousRevisionId?: string
  ): string | undefined {
    if (previousRevisionId) {
      for (const link of this.deps.repos.proceduralTrajectory.listFamilyClusterLinks(
        previousRevisionId
      )) {
        const key = this.deps.repos.proceduralTrajectory.getClusterCanonicalKey(link.canonicalKeyId);
        if (key?.clusterId === clusterId) return key.id;
      }
    }
    return this.deps.repos.proceduralTrajectory.listClusterCanonicalKeys(clusterId)[0]?.id;
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

  private maximalQualifiedClustersForAffected(
    userId: string,
    affectedClusterIds: Set<string>,
    config: PipelineConfig
  ): ClusterSnapshot[] {
    const scope = this.deps.repos.proceduralTrajectory.listMaximalityScope({
      userId,
      algorithmVersion: config.algorithmVersion,
      configHash: config.clusteringConfigHash,
      affectedClusterIds: [...affectedClusterIds]
    });
    const targetIds = new Set(scope.targetClusterIds);
    const qualified = scope.contextClusterIds.flatMap((clusterId) => {
      const head = this.deps.repos.proceduralTrajectory.getClusterHead(clusterId);
      if (!head || head.status !== "active") return [];
      const snapshot = this.loadClusterSnapshot(head, config);
      return snapshot?.core &&
        snapshot.version.supportEpisodeCount >= config.minSupportEpisodes
        ? [snapshot]
        : [];
    });
    const selectedById = new Map(selectMaximalWindowClusters(
      qualified.map((snapshot) => snapshot.domain)
    ).map((domain) => [domain.id, domain]));
    return qualified.filter((snapshot) => targetIds.has(snapshot.head.id))
      .map((snapshot) => ({
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
    const maximal = this.maximalQualifiedClustersForAffected(
      userId,
      affectedClusterIds,
      config
    );
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
    // Keep bounded occurrence-local outward Steps in the Skill input even when
    // no extension is repeated across two Episodes. They remain local_context
    // and may only become provisional conditional guidance; mandatory
    // procedure/verification evidence still requires shared_extension.
    const effectiveCompletionOverlay = completionOverlay;
    const projectionByOccurrenceId = new Map(
      effectiveCompletionOverlay.projections.map((item) => [
      item.occurrenceId,
      item
    ]));
    const evidence: ProceduralSkillEvidenceOccurrence[] = drafts.flatMap((draft) => {
      const projection = projectionByOccurrenceId.get(draft.span.id);
      const effectivePrefixSteps = draft.prefixSteps;
      const effectiveSuffixSteps = draft.suffixSteps;
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
          [...draft.prefixSteps, ...draft.coreTraceSteps, ...draft.suffixSteps]
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
          ...(draft.expandedPreviousStep
            ? { previousStep: skillEvidenceStep(draft.expandedPreviousStep) }
            : {}),
          ...(draft.expandedNextStep
            ? { nextStep: skillEvidenceStep(draft.expandedNextStep) }
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

  private async retrieveV2ComparisonSkills(
    draft: ProceduralPatternSkillDraft
  ): Promise<ProceduralSkillComparisonCandidate[]> {
    const input = draft.input;
    const eligible = this.deps.repos.memories
      .list({
        userId: input.userId,
        memoryLayer: "Skill",
        status: ["activated", "resolving"]
      }, 1_000)
      .flatMap((memory) => {
        const route = proceduralSkillComparisonRoute(memory);
        const skill = skillMetaFromMemory(memory);
        if (!route || !skill || skill.status === "archived") return [];
        return [{ memory, route, skill }];
      });
    if (eligible.length === 0) return [];

    const queryText = [
      draft.parsed.triggerContext,
      draft.parsed.summary,
      ...draft.parsed.procedureSteps.flatMap((step) => [step.title, step.body]),
      ...draft.parsed.verificationSteps.flatMap((step) => [step.check, step.successSignal])
    ].filter(Boolean).join("\n");
    let queryVector: number[] | undefined;
    try {
      if (queryText.trim() && this.deps.embedder.status().configured) {
        queryVector = unitVector(await this.deps.embedder.embedOne(queryText, "query"));
      }
    } catch {
      // Retrieval must degrade to local lexical ranking rather than fail the
      // durable Skill induction job when the embedding endpoint is transiently unavailable.
      queryVector = undefined;
    }
    const ranked = eligible
      .map(({ memory, route, skill }) => {
        const vectorScore = queryVector && skill.vec?.length === queryVector.length
          ? cosineSimilarity(queryVector, skill.vec)
          : 0;
        const lexicalScore = lexicalSkillSimilarity(queryText, skill.invocationGuide);
        return {
          candidate: proceduralSkillComparisonCandidate(memory, route),
          score: Math.max(vectorScore, lexicalScore)
        };
      })
      .sort((left, right) => right.score - left.score ||
        left.candidate.memoryId.localeCompare(right.candidate.memoryId));
    const oldAndV3 = ranked
      .filter((item) => item.candidate.route === "OLD" || item.candidate.route === "V3")
      .slice(0, V2_SKILL_COMPARISON_LIMIT_PER_PHASE);
    const v2 = ranked
      .filter((item) => item.candidate.route === "V2")
      .slice(0, V2_SKILL_COMPARISON_LIMIT_PER_PHASE);
    // Keep the phases physically ordered as well as explicitly separated in
    // the prompt. A dense V2 neighborhood cannot crowd OLD/V3 containment out,
    // and serialized V2 jobs expose earlier same-batch materializations here.
    return [...oldAndV3, ...v2].map((item) => item.candidate);
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
    const inductionVersion = this.skillInductionVersion(config);
    const activeVersion = head.activeSkillVersionId
      ? this.deps.repos.proceduralTrajectory.getSkillVersion(head.activeSkillVersionId)
      : undefined;
    if (activeVersion?.clusterVersionId === clusterVersionId &&
        activeVersion.payload.inductionVersion === inductionVersion) {
      if (activeVersion.skillMemoryId) return this.activeSkillMatchesClusterVersion(
        head,
        clusterVersionId
      );
      const reason = typeof activeVersion.payload.reason === "string"
        ? activeVersion.payload.reason
        : "";
      return !reason.startsWith("maximal-suppressed-by:") &&
        reason !== "not-maximal-qualified";
    }
    const version = this.deps.repos.proceduralTrajectory.listSkillVersions(head.id)
      .find((candidate) => candidate.clusterVersionId === clusterVersionId &&
        !candidate.skillMemoryId && candidate.payload.admitted === false &&
        candidate.payload.inductionVersion === inductionVersion);
    if (!version) return false;
    const reason = typeof version.payload.reason === "string" ? version.payload.reason : "";
    // A previously suppressed shorter Skill must be reconsidered when the
    // covering longer cluster disappears. Other decisions are current for this
    // immutable membership version.
    return !reason.startsWith("maximal-suppressed-by:") && reason !== "not-maximal-qualified";
  }

  private recordProceduralSkillRejection(
    head: TrajectoryWindowClusterRecord,
    version: TrajectoryWindowClusterVersionRecord | undefined,
    reason: string,
    at: string,
    coverageDecision?: {
      decision: "covered";
      relation: "equivalent" | "subset";
      reason: string;
      targetSkillId: string;
      targetRoute: "OLD" | "V2" | "V3";
    },
    rawCoverageDecision?: Record<string, unknown>
  ): void {
    if (!version || head.status !== "active" || head.activeVersionId !== version.id) return;
    const inductionVersion = this.skillInductionVersion(this.pipelineConfig());
    this.deps.repos.transaction(() => {
      const current = this.deps.repos.proceduralTrajectory.getClusterHead(head.id);
      if (!current || current.status !== "active" || current.activeVersionId !== version.id) return;
      this.deps.repos.proceduralTrajectory.recordRejectedSkillVersion({
        clusterId: current.id,
        clusterVersionId: version.id,
        expectedActiveSkillVersionId: current.activeSkillVersionId ?? null,
        skillKey: proceduralSkillKey(current.userId, current.id),
        payload: {
          admitted: false,
          inductionVersion,
          reason,
          clusterVersionId: version.id,
          supportEpisodeCount: version.supportEpisodeCount,
          preservedActiveSkillVersionId: current.activeSkillVersionId ?? null,
          preservedActiveSkillMemoryId: current.activeSkillMemoryId ?? null,
          ...(coverageDecision ? {
            coverageDecision,
            suppressedBySkillId: coverageDecision.targetSkillId,
            suppressedByRoute: coverageDecision.targetRoute
          } : {}),
          ...(rawCoverageDecision ? { rawCoverageDecision } : {})
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
      const otherActiveOwners = oldMemoryId
        ? this.deps.repos.proceduralTrajectory
          .listActiveClusterHeadsBySkillMemoryId(oldMemoryId)
          .filter((candidate) => candidate.id !== current.id)
        : [];
      if (oldMemoryId && otherActiveOwners.length === 0) {
        this.archiveProceduralSkillMemory(oldMemoryId, reason, at);
      }
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
    if (!isProceduralSkillAlgorithm(internal.plugin_algorithm)) return;
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
      source: internal.plugin_algorithm === LONG_TRAJECTORY_SKILL_ALGORITHM
        ? "worker.long_trajectory_induction.lifecycle.v1"
        : "worker.procedural_skill_induction.lifecycle.v1",
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
    const linkedMemory = head.activeSkillMemoryId
      ? this.deps.repos.memories.getIncludingDeleted(head.activeSkillMemoryId)
      : undefined;
    const ownedMemory = this.deps.repos.memories.getByKeyIncludingDeleted(
      "Skill",
      proceduralSkillKey(head.userId, head.id)
    );
    for (const memory of unique([linkedMemory, ownedMemory].filter(isDefined))) {
      if (memory.userId !== head.userId ||
          memory.properties.internal_info.plugin_algorithm !==
            PROCEDURAL_PATTERN_SKILL_ALGORITHM) continue;
      const governance = memory.properties.internal_info.procedural_governance;
      if (isRecord(governance) && governance.disabled === true) return true;
    }
    return false;
  }

  private longTrajectorySkillGovernanceDisabled(
    candidate: LongTrajectoryCandidateRecord
  ): boolean {
    const memory = this.deps.repos.memories.getByKeyIncludingDeleted(
      "Skill",
      proceduralSkillKey(candidate.userId, candidate.id)
    );
    if (!memory || memory.userId !== candidate.userId ||
        memory.properties.internal_info.plugin_algorithm !==
          LONG_TRAJECTORY_SKILL_ALGORITHM) {
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

function longTrajectoryEvidenceSignature(trajectory: LongCommonTrajectoryV1): string {
  return stableHash({
    version: "long-trajectory-evidence-signature.v1",
    supportEpisodeIds: [...trajectory.supportEpisodeIds].sort(),
    occurrences: trajectory.occurrences.map((occurrence) => ({
      episodeId: occurrence.episodeId,
      episodeSpanIds: occurrence.matches.map((match) => match.episodeSpanId)
    })).sort((left, right) => left.episodeId.localeCompare(right.episodeId))
  });
}

function longTrajectoryQuality(trajectory: LongCommonTrajectoryV1): number {
  const coveredSpanSteps = trajectory.requiredSpans.reduce(
    (sum, span) => sum + span.scale,
    0
  );
  return trajectory.supportEpisodeIds.length * coveredSpanSteps *
    trajectory.averageEpisodeCoverage * trajectory.averageCoarseSimilarity;
}

function longCommonTrajectoryFromPayload(value: unknown): LongCommonTrajectoryV1 | undefined {
  if (!isRecord(value) ||
      value.schemaVersion !== "long-common-span-sequence.v1" ||
      typeof value.id !== "string" ||
      typeof value.familyId !== "string" ||
      typeof value.referenceEpisodeId !== "string" ||
      typeof value.candidateStructureKey !== "string" ||
      typeof value.structureHash !== "string" ||
      !Array.isArray(value.requiredSpans) ||
      !Array.isArray(value.supportEpisodeIds) ||
      !Array.isArray(value.occurrences)) return undefined;
  return value as unknown as LongCommonTrajectoryV1;
}

function uniqueDocuments(
  documents: readonly EpisodeTrajectoryDocumentV1[]
): EpisodeTrajectoryDocumentV1[] {
  const byEpisodeId = new Map<string, EpisodeTrajectoryDocumentV1>();
  for (const document of documents) byEpisodeId.set(document.path.episodeId, document);
  return [...byEpisodeId.values()].sort((left, right) =>
    left.path.episodeId.localeCompare(right.path.episodeId));
}

function longTrajectoryRowContainsInEpisode(
  container: LongTrajectorySkillCandidateRow,
  candidate: LongTrajectorySkillCandidateRow,
  episodeId: string
): boolean {
  const containerOccurrence = container.trajectory.occurrences.find((item) =>
    item.episodeId === episodeId);
  const candidateOccurrence = candidate.trajectory.occurrences.find((item) =>
    item.episodeId === episodeId);
  if (!containerOccurrence || !candidateOccurrence) return false;
  const coveredSteps = new Set(containerOccurrence.matches.flatMap((match) =>
    integerRange(match.episodeStartStepIndex, match.episodeEndStepIndex)));
  const candidateSteps = new Set(candidateOccurrence.matches.flatMap((match) =>
    integerRange(match.episodeStartStepIndex, match.episodeEndStepIndex)));
  return candidateSteps.size > 0 && [...candidateSteps].every((step) => coveredSteps.has(step));
}

function dedupeLongTrajectoryRows(
  rows: readonly LongTrajectorySkillCandidateRow[]
): LongTrajectorySkillCandidateRow[] {
  const byStructure = new Map<string, LongTrajectorySkillCandidateRow>();
  for (const row of rows) {
    const current = byStructure.get(row.candidateStructureKey);
    if (!current || row.quality > current.quality ||
        (row.quality === current.quality &&
          row.trajectory.id.localeCompare(current.trajectory.id) < 0)) {
      byStructure.set(row.candidateStructureKey, row);
    }
  }
  return [...byStructure.values()];
}

function integerRange(start: number, end: number): number[] {
  return Array.from({ length: Math.max(0, end - start + 1) }, (_, index) => start + index);
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

function proceduralSkillComparisonRoute(
  memory: MemoryRow
): ProceduralSkillComparisonCandidate["route"] | undefined {
  const algorithm = memory.properties.internal_info.plugin_algorithm;
  if (algorithm === LEGACY_SKILL_ALGORITHM) return "OLD";
  if (algorithm === PROCEDURAL_PATTERN_SKILL_ALGORITHM) return "V2";
  if (algorithm === LONG_TRAJECTORY_SKILL_ALGORITHM) return "V3";
  return undefined;
}

function proceduralSkillComparisonCandidate(
  memory: MemoryRow,
  route: ProceduralSkillComparisonCandidate["route"]
): ProceduralSkillComparisonCandidate {
  const meta = skillMetaFromMemory(memory);
  const internal = memory.properties.internal_info;
  const internalSkill = isRecord(internal.skill) ? internal.skill : {};
  const procedure = isRecord(internal.procedure_json)
    ? internal.procedure_json
    : isRecord(internalSkill.procedure_json)
      ? internalSkill.procedure_json
      : {};
  const steps = Array.isArray(procedure.steps)
    ? procedure.steps.filter(isRecord).map((step) => ({
        title: typeof step.title === "string" ? step.title : "",
        body: typeof step.body === "string" ? step.body : "",
        verification: step.kind === "verification" ||
          (typeof step.title === "string" && /^verify\s*:/i.test(step.title))
      })).filter((step) => step.title || step.body)
    : [];
  return {
    memoryId: memory.id,
    route,
    name: meta?.name ?? memory.memoryKey ?? memory.id,
    invocationGuide: meta?.invocationGuide ?? memory.memoryValue,
    triggerContext: meta?.triggerContext ??
      (typeof procedure.triggerContext === "string" ? procedure.triggerContext : ""),
    summary: typeof procedure.summary === "string" ? procedure.summary : "",
    procedureSteps: steps.filter((step) => !step.verification)
      .map(({ title, body }) => ({ title, body })),
    verificationSteps: steps.filter((step) => step.verification)
      .map(({ title, body }) => ({ title, body }))
  };
}

function lexicalSkillSimilarity(left: string, right: string): number {
  const tokens = (value: string): Set<string> => new Set([
    ...(value.toLowerCase().match(/[a-z0-9_][a-z0-9_./-]{2,}/g) ?? []),
    ...(value.match(/[\p{Script=Han}]{2,}/gu) ?? [])
  ]);
  const leftTokens = tokens(left);
  const rightTokens = tokens(right);
  if (leftTokens.size === 0 || rightTokens.size === 0) return 0;
  let overlap = 0;
  for (const token of leftTokens) if (rightTokens.has(token)) overlap += 1;
  return overlap / Math.sqrt(leftTokens.size * rightTokens.size);
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

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? unique(value.filter((item): item is string => typeof item === "string" && item.length > 0))
    : [];
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}
