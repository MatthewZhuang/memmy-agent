import {
  policyMetaFromMemory,
  skillMetaFromMemory,
  traceMetaFromMemory,
  worldModelMetaFromMemory
} from "../../algorithm/plugin-algorithms.js";
import type { MemmyConfig } from "../../config/index.js";
import type { Embedder, LlmClient } from "../../model/types.js";
import type {
  EpisodeRecord,
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type { SaveEpisodeProceduralPathResult } from
  "../../storage/procedural-path-repository.js";
import type { MemoryRow,ToolCallPayload } from "../../types.js";
import { newId } from "../../utils/id.js";
import { nowIso } from "../../utils/time.js";
import type { ScheduleEmbeddingAfterTextUpdateInput } from "../embedding/embedding-job-processor.js";
import type {
  DecisionRepairTraceSource,
  SynthesizeDecisionRepairDraft
} from "../feedback/feedback-experience.js";
import {
  profileIdFromMemory,
  projectIdFromMemory
} from "../namespace/namespace-scope.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import { NegativeExperiencePipeline } from "./negative-experience-pipeline.js";
import { BigTurnSpanPipeline } from "./big-turn-span-pipeline.js";
import { PolicyInductionEngine } from "./policy-induction.js";
import {
  RewardPipeline,
  type DecisionRepairSummary
} from "./reward-pipeline.js";
import { SkillPipeline } from "./skill-pipeline.js";
import { SpanPipeline } from "./span-pipeline.js";
import type { TurnMemoryCaptureDecision } from "./span-pipeline.js";
import { WorldModelPipeline } from "./world-model-pipeline.js";
import { EpisodeProceduralReconstructor } from "./episode-procedural-reconstructor.js";
import { EpisodeProceduralPathPersistencePipeline } from "./episode-procedural-path-pipeline.js";
import {
  EpisodeBoundarySegmenter,
  type EpisodeBoundarySegmentationResultV1
} from "./episode-boundary-segmentation.js";
import {
  EpisodeSubproblemContractSegmenter,
  type SubproblemContractSegmentationResultV1
} from "./episode-subproblem-contract-segmentation.js";
import { StepSequenceLearningPipeline } from "./step-sequence-learning.js";
import type { StepSequenceLearningResult } from "./step-sequence-learning.js";
import {
  MultiScaleWindowPolicyExperiment,
  type MultiScaleWindowCoarseMembershipMode,
  type MultiScaleWindowPolicyDecisionV1,
  type MultiScaleWindowPolicyExperimentResultV1,
  type MultiScaleWindowSkillCandidateV1,
  type MultiScaleWindowSkillDecisionV1,
  type MultiScaleWindowSpec,
  type PreparedMultiScaleWindowPolicyExperimentV1,
  type TrajectoryWindowClusterV1
} from "./multi-scale-window-policy.js";
import type { BandedMonotonicMatchConfig } from "./trajectory-window-alignment.js";

type TraceMeta = NonNullable<ReturnType<typeof traceMetaFromMemory>>;
type PolicyMeta = NonNullable<ReturnType<typeof policyMetaFromMemory>>;

export interface PolicyEvidencePreflightReport {
  orphanPolicyIds: string[];
  affectedWorldModelIds: string[];
  affectedSkillIds: string[];
  restorablePolicyIds: string[];
}

export interface EvolutionJobProcessorDeps {
  repos: Repositories;
  config: MemmyConfig;
  llm: LlmClient;
  skillLlm: LlmClient;
  embedder: Embedder;
  traceMeta(memory: MemoryRow | undefined | null): TraceMeta | null;
  namespaceIdFromMemory(memory: MemoryRow): string;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
  enqueueEpisodeRewardAfterReflection(
    episode: EpisodeRecord,
    at: string,
    trigger: string
  ): EvolutionJobRecord[];
  finalizeClosedEpisode(
    episode: EpisodeRecord,
    at: string,
    trigger: "episode_rewarded"
  ): EvolutionJobRecord[];
  resolvePendingSkillTrialsForReward(input: {
    userId: string;
    episodeId: string;
    rHuman: number;
    feedbackId?: string;
    at: string;
  }): void;
  decisionRepairTraceSources(memories: MemoryRow[]): DecisionRepairTraceSource[];
  synthesizeDecisionRepairDraft: SynthesizeDecisionRepairDraft;
  scheduleEmbeddingAfterTextUpdate(input: ScheduleEmbeddingAfterTextUpdateInput): void;
  repairEvidenceValueDiff(highValue: MemoryRow[], lowValue: MemoryRow[]): number;
}

export class EvolutionJobProcessor {
  private readonly policy: PolicyInductionEngine;
  private readonly negativeExperience: NegativeExperiencePipeline;
  private readonly reward: RewardPipeline;
  private readonly skill: SkillPipeline;
  private readonly span: SpanPipeline;
  private readonly proceduralPath: EpisodeProceduralPathPersistencePipeline;
  private readonly episodeBoundarySegmenter: EpisodeBoundarySegmenter;
  private readonly episodeSubproblemContractSegmenter: EpisodeSubproblemContractSegmenter;
  private readonly stepSequenceLearning: StepSequenceLearningPipeline;
  private readonly multiScaleWindowPolicy: MultiScaleWindowPolicyExperiment;
  private readonly bigTurnSpan: BigTurnSpanPipeline;
  private readonly worldModel: WorldModelPipeline;

  constructor(private readonly deps: EvolutionJobProcessorDeps) {
    const owner = this;
    this.skill = new SkillPipeline({
      repos: deps.repos,
      get config() { return owner.deps.config; },
      get skillLlm() { return owner.deps.skillLlm; },
      traceMeta: deps.traceMeta,
      buildMemory: deps.buildMemory,
      upsertEvolutionMemory: this.upsertEvolutionMemory.bind(this),
      isArchivedEvolutionMemory: this.isArchivedEvolutionMemory.bind(this),
      enqueueJob: deps.enqueueJob,
      namespaceIdFromMemory: deps.namespaceIdFromMemory
    });
    this.policy = new PolicyInductionEngine({
      get config() { return owner.deps.config; },
      repos: deps.repos,
      nowIso,
      get skillLlm() { return owner.deps.skillLlm; },
      traceMeta: deps.traceMeta,
      projectIdFromMemory,
      profileIdFromMemory,
      buildMemory: deps.buildMemory,
      upsertEvolutionMemory: this.upsertEvolutionMemory.bind(this),
      enqueueJob: deps.enqueueJob,
      enqueueChange: deps.repos.runtime.appendChange.bind(deps.repos.runtime),
      namespaceIdFromMemory: deps.namespaceIdFromMemory,
      onSkillRewardDrift: this.skill.applySkillRewardDriftForPolicy.bind(this.skill)
    });
    this.worldModel = new WorldModelPipeline({
      repos: deps.repos,
      get config() { return owner.deps.config; },
      get skillLlm() { return owner.deps.skillLlm; },
      traceMeta: deps.traceMeta,
      buildMemory: deps.buildMemory,
      upsertEvolutionMemory: this.upsertEvolutionMemory.bind(this),
      isArchivedEvolutionMemory: this.isArchivedEvolutionMemory.bind(this),
      enqueueJob: deps.enqueueJob,
      namespaceIdFromMemory: deps.namespaceIdFromMemory
    });
    this.span = new SpanPipeline({
      repos: deps.repos,
      get config() { return owner.deps.config; },
      get llm() { return owner.deps.llm; },
      get skillLlm() { return owner.deps.skillLlm; },
      traceMeta: deps.traceMeta,
      namespaceIdFromMemory: deps.namespaceIdFromMemory,
      enqueueJob: deps.enqueueJob,
      enqueueEpisodeRewardAfterReflection: deps.enqueueEpisodeRewardAfterReflection,
      scheduleEmbeddingAfterTextUpdate: deps.scheduleEmbeddingAfterTextUpdate
    });
    this.bigTurnSpan = new BigTurnSpanPipeline({
      repos: deps.repos,
      get llm() { return owner.deps.llm; },
      buildMemory: deps.buildMemory,
      enqueueJob: deps.enqueueJob,
      namespaceIdFromMemory: deps.namespaceIdFromMemory,
      embedAfterCapture: () => owner.deps.config.algorithm.capture.embedAfterCapture
    });
    this.proceduralPath = new EpisodeProceduralPathPersistencePipeline({
      repos: deps.repos,
      reconstructor: new EpisodeProceduralReconstructor({
        get llm() { return owner.deps.skillLlm; }
      }),
      enqueueJob: deps.enqueueJob
    });
    this.episodeBoundarySegmenter = new EpisodeBoundarySegmenter({
      get llm() { return owner.deps.skillLlm; },
      get enableThinking() { return owner.deps.config.evolution.enableThinking; }
    });
    this.episodeSubproblemContractSegmenter = new EpisodeSubproblemContractSegmenter({
      get llm() { return owner.deps.skillLlm; },
      get enableThinking() { return owner.deps.config.evolution.enableThinking; }
    });
    this.stepSequenceLearning = new StepSequenceLearningPipeline({
      repos: deps.repos,
      get config() { return owner.deps.config; },
      get embedder() { return owner.deps.embedder; },
      get skillLlm() { return owner.deps.skillLlm; },
      buildMemory: deps.buildMemory,
      upsertEvolutionMemory: this.upsertEvolutionMemory.bind(this),
      enqueueJob: deps.enqueueJob
    });
    this.multiScaleWindowPolicy = new MultiScaleWindowPolicyExperiment({
      get config() { return owner.deps.config; },
      get embedder() { return owner.deps.embedder; },
      get llm() { return owner.deps.skillLlm; }
    });
    this.reward = new RewardPipeline({
      get config() { return owner.deps.config; },
      repos: deps.repos,
      get llm() { return owner.deps.llm; },
      nowIso,
      newId,
      traceMeta: deps.traceMeta,
      namespaceIdFromMemory: deps.namespaceIdFromMemory,
      enqueueJob: deps.enqueueJob,
      finalizeClosedEpisode: deps.finalizeClosedEpisode,
      resolvePendingSkillTrialsForReward: deps.resolvePendingSkillTrialsForReward,
      decisionRepairTraceSources: deps.decisionRepairTraceSources,
      synthesizeDecisionRepairDraft: deps.synthesizeDecisionRepairDraft,
      isTraceEligibleForL2: this.policy.isTraceEligibleForL2.bind(this.policy),
      recordCandidatePoolTrace: this.policy.recordCandidatePoolTrace.bind(this.policy),
      repairEvidenceValueDiff: deps.repairEvidenceValueDiff
    });
    this.negativeExperience = new NegativeExperiencePipeline({
      repos: deps.repos,
      get config() { return owner.deps.config; },
      buildMemory: deps.buildMemory,
      upsertEvolutionMemory: this.upsertEvolutionMemory.bind(this),
      enqueueJob: deps.enqueueJob,
      namespaceIdFromMemory: deps.namespaceIdFromMemory
    });
  }

  async induceL2(job: EvolutionJobRecord): Promise<void> {
    // Drain jobs persisted by the retired Span learning experiments without
    // feeding their incompatible payloads into the upstream L2 inducer.
    if (typeof job.payload.proceduralClusterId === "string" ||
        typeof job.payload.clusterId === "string") return;
    await this.policy.induceL2(job);
  }

  associateL2(job: EvolutionJobRecord): void {
    return this.policy.associateL2(job);
  }

  abstractL3(job: EvolutionJobRecord): Promise<void> {
    return this.worldModel.abstractL3(job);
  }

  crystallizeSkill(job: EvolutionJobRecord): Promise<void> {
    if (typeof job.payload.proceduralSkillCandidateId === "string") return Promise.resolve();
    return this.skill.crystallizeSkill(job);
  }

  reflectTrace(job: EvolutionJobRecord): Promise<void> {
    return this.span.reflectTrace(job);
  }

  applyReward(job: EvolutionJobRecord): Promise<void> {
    return this.reward.applyReward(job);
  }

  splitBigTurn(job: EvolutionJobRecord): Promise<void> {
    return this.bigTurnSpan.splitAndStore(job);
  }

  async reconstructProceduralPath(job: EvolutionJobRecord): Promise<void> {
    await this.proceduralPath.reconstructJob(job);
  }

  async learnStepSequences(job: EvolutionJobRecord): Promise<void> {
    await this.stepSequenceLearning.learnJob(job);
  }

  async repairStepPolicy(job: EvolutionJobRecord): Promise<void> {
    await this.stepSequenceLearning.repairPoliciesJob(job);
  }

  async learnStepSequencesForReplay(input: {
    episodeId: string;
    at?: string;
  }): Promise<StepSequenceLearningResult> {
    const path = this.deps.repos.proceduralPaths.getActiveForEpisode(input.episodeId);
    if (!path) throw new Error(`active procedural Path not found: ${input.episodeId}`);
    return this.stepSequenceLearning.learnPath(path, input.at ?? nowIso());
  }

  async discoverMultiScaleWindowPoliciesForReplay(input: {
    episodeIds: string[];
    specs?: MultiScaleWindowSpec[];
    coarseSimilarityThreshold?: number;
    coarseSimilarityThresholdByScale?: Partial<Record<number, number>>;
    fineMatchConfigs?: BandedMonotonicMatchConfig[];
    medoidSwitchMargin?: number;
    coarseMembershipMode?: MultiScaleWindowCoarseMembershipMode;
    similarityThreshold?: number;
    minSupportEpisodes?: number;
    maxPolicyCandidates?: number;
    policyConcurrency?: number;
    inducePolicies?: boolean;
  }): Promise<MultiScaleWindowPolicyExperimentResultV1> {
    const episodes = input.episodeIds.map((episodeId) => {
      const episode = this.deps.repos.runtime.getEpisode(episodeId);
      const path = this.deps.repos.proceduralPaths.getActiveForEpisode(episodeId);
      if (!episode || episode.status !== "closed" || !path) {
        throw new Error(`closed Episode or active procedural Path not found: ${episodeId}`);
      }
      return {
        episodeId,
        pathId: path.id,
        ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask }),
        steps: path.path.steps
      };
    });
    return this.multiScaleWindowPolicy.run({
      episodes,
      ...(input.specs ? { specs: input.specs } : {}),
      ...(input.coarseSimilarityThreshold === undefined
        ? {}
        : { coarseSimilarityThreshold: input.coarseSimilarityThreshold }),
      ...(input.coarseSimilarityThresholdByScale === undefined
        ? {}
        : { coarseSimilarityThresholdByScale: input.coarseSimilarityThresholdByScale }),
      ...(input.fineMatchConfigs === undefined
        ? {}
        : { fineMatchConfigs: input.fineMatchConfigs }),
      ...(input.medoidSwitchMargin === undefined
        ? {}
        : { medoidSwitchMargin: input.medoidSwitchMargin }),
      ...(input.coarseMembershipMode === undefined
        ? {}
        : { coarseMembershipMode: input.coarseMembershipMode }),
      ...(input.similarityThreshold === undefined
        ? {}
        : { similarityThreshold: input.similarityThreshold }),
      ...(input.minSupportEpisodes === undefined
        ? {}
        : { minSupportEpisodes: input.minSupportEpisodes }),
      ...(input.maxPolicyCandidates === undefined
        ? {}
        : { maxPolicyCandidates: input.maxPolicyCandidates }),
      ...(input.policyConcurrency === undefined
        ? {}
        : { policyConcurrency: input.policyConcurrency }),
      ...(input.inducePolicies === undefined
        ? {}
        : { inducePolicies: input.inducePolicies })
    });
  }

  async prepareMultiScaleWindowPoliciesForReplay(input: {
    episodeIds: string[];
    specs?: MultiScaleWindowSpec[];
  }): Promise<PreparedMultiScaleWindowPolicyExperimentV1> {
    const episodes = input.episodeIds.map((episodeId) => {
      const episode = this.deps.repos.runtime.getEpisode(episodeId);
      const path = this.deps.repos.proceduralPaths.getActiveForEpisode(episodeId);
      if (!episode || episode.status !== "closed" || !path) {
        throw new Error(`closed Episode or active procedural Path not found: ${episodeId}`);
      }
      return {
        episodeId,
        pathId: path.id,
        ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask }),
        steps: path.path.steps
      };
    });
    return this.multiScaleWindowPolicy.prepare({
      episodes,
      ...(input.specs ? { specs: input.specs } : {})
    });
  }

  async induceMultiScaleWindowPoliciesForReplay(input: {
    clusters: TrajectoryWindowClusterV1[];
    concurrency?: number;
  }): Promise<MultiScaleWindowPolicyDecisionV1[]> {
    return this.multiScaleWindowPolicy.inducePolicies(
      input.clusters,
      input.concurrency ?? 4
    );
  }

  async induceMultiScaleWindowSkillsForReplay(input: {
    candidates: MultiScaleWindowSkillCandidateV1[];
    concurrency?: number;
  }): Promise<MultiScaleWindowSkillDecisionV1[]> {
    return this.multiScaleWindowPolicy.induceSkills(
      input.candidates,
      input.concurrency ?? 4
    );
  }

  async reconstructProceduralPathForReplay(input: {
    episodeId: string;
    at?: string;
  }): Promise<SaveEpisodeProceduralPathResult> {
    return this.proceduralPath.reconstructAndPersist({
      episodeId: input.episodeId,
      activate: true,
      ...(input.at ? { createdAt: input.at } : {})
    });
  }

  async segmentEpisodeBoundariesForReplay(input: {
    episodeId: string;
  }): Promise<EpisodeBoundarySegmentationResultV1> {
    const path = this.deps.repos.proceduralPaths.getActiveForEpisode(input.episodeId);
    if (!path) throw new Error(`active procedural Path not found: ${input.episodeId}`);
    return this.episodeBoundarySegmenter.segment(path.path);
  }

  async segmentEpisodeSubproblemContractsForReplay(input: {
    episodeId: string;
  }): Promise<SubproblemContractSegmentationResultV1> {
    const path = this.deps.repos.proceduralPaths.getActiveForEpisode(input.episodeId);
    if (!path) throw new Error(`active procedural Path not found: ${input.episodeId}`);
    return this.episodeSubproblemContractSegmenter.segment(path.path);
  }

  materializeNegativeExperience(job: EvolutionJobRecord): void {
    this.negativeExperience.materialize(job);
  }

  summarizeTraceForCapture(input: {
    trace: TraceMeta;
    userText: string;
    agentText: string;
    toolCalls: ToolCallPayload[];
    reflectionText: string;
  }, options: { strict?: boolean } = {}): Promise<string> {
    return this.span.summarizeTraceForCapture(input, options);
  }

  decideTurnMemoryForCapture(input: {
    trace: TraceMeta;
    userText: string;
    agentText: string;
    toolCalls: ToolCallPayload[];
    reflectionText: string;
  }): Promise<TurnMemoryCaptureDecision> {
    return this.span.decideTurnMemoryForCapture(input);
  }

  findExistingSkillForPolicy(policy: PolicyMeta) {
    return this.skill.findExistingSkillForPolicy(policy);
  }

  previewPolicyEvidenceReconciliation(limit = 10000): PolicyEvidencePreflightReport {
    const orphanPolicyIds: string[] = [];
    const restorablePolicyIds: string[] = [];
    for (const { policy, linkedTraceIds, validTraces } of this.activePolicyEvidenceStates(limit)) {
      if (validTraces.length === 0) {
        orphanPolicyIds.push(policy.id);
      } else if (validTraces.some((trace) => !linkedTraceIds.has(trace.id))) {
        restorablePolicyIds.push(policy.id);
      }
    }

    const orphanIds = new Set(orphanPolicyIds);
    const affectedWorldModelIds = this.deps.repos.memories
      .list({ memoryLayer: "L3", status: ["activated", "resolving"] }, limit)
      .filter((memory) => worldModelMetaFromMemory(memory)?.policyIds.some((id) => orphanIds.has(id)))
      .map((memory) => memory.id);
    const affectedSkillIds = this.deps.repos.memories
      .list({ memoryLayer: "Skill", status: ["activated", "resolving"] }, limit)
      .filter((memory) => skillMetaFromMemory(memory)?.sourcePolicyIds.some((id) => orphanIds.has(id)))
      .map((memory) => memory.id);

    return {
      orphanPolicyIds: orphanPolicyIds.sort(),
      affectedWorldModelIds: affectedWorldModelIds.sort(),
      affectedSkillIds: affectedSkillIds.sort(),
      restorablePolicyIds: restorablePolicyIds.sort()
    };
  }

  reconcileOrphanedPolicies(at: string, limit = 10000): number {
    let reconciled = 0;
    for (const { memory, policy, linkedTraceIds, validTraces } of this.activePolicyEvidenceStates(limit)) {
      if (validTraces.length === 0) {
        const updated = this.policy.recomputePolicyStats(policy.id, at);
        if (updated && policyMetaFromMemory(updated)?.status !== "active") {
          this.invalidatePolicyDependencies(policy.id, at);
          reconciled += 1;
        }
        continue;
      }

      let restored = false;
      for (const trace of validTraces) {
        if (linkedTraceIds.has(trace.id)) continue;
        this.deps.repos.runtime.insertTracePolicyLink({
          userId: memory.userId,
          l1MemoryId: trace.id,
          l2MemoryId: policy.id,
          relation: "supports",
          strength: 1,
          createdAt: at
        });
        restored = true;
      }
      if (!restored) continue;
      this.deps.repos.runtime.appendChange({
        memoryId: memory.id,
        namespaceId: this.deps.namespaceIdFromMemory(memory),
        kind: "policy",
        op: "updated",
        entityId: memory.id,
        userId: memory.userId,
        changeType: "policy_evidence_links_restored",
        before: { linkedTraceIds: Array.from(linkedTraceIds) },
        after: { linkedTraceIds: validTraces.map((trace) => trace.id) },
        source: "startup.policy_evidence_reconciliation",
        createdAt: at
      });
      reconciled += 1;
    }
    return reconciled;
  }

  private activePolicyEvidenceStates(limit: number): Array<{
    memory: MemoryRow;
    policy: PolicyMeta;
    linkedTraceIds: Set<string>;
    validTraces: TraceMeta[];
  }> {
    return this.deps.repos.memories
      .list({ memoryLayer: "L2", status: "activated" }, limit)
      .map((memory) => ({ memory, policy: policyMetaFromMemory(memory) }))
      .filter((item): item is { memory: MemoryRow; policy: PolicyMeta } =>
        item.policy?.status === "active"
      )
      .map(({ memory, policy }) => {
        const linkedTraceIds = new Set(this.deps.repos.runtime.listTracePolicyLinks({
          l2MemoryId: policy.id,
          limit: 1000
        }).map((link) => link.l1MemoryId));
        const candidateTraceIds = Array.from(new Set([
          ...policy.sourceTraceIds,
          ...linkedTraceIds
        ]));
        const validTraces = this.deps.repos.memories
          .getMany(candidateTraceIds)
          .map((candidate) => this.deps.traceMeta(candidate))
          .filter((trace): trace is TraceMeta => Boolean(
            trace && this.policy.isTraceEligibleForL2(trace)
          ));
        return { memory, policy, linkedTraceIds, validTraces };
      });
  }

  invalidateMemoryDependencies(memory: MemoryRow, at: string): void {
    if (memory.memoryLayer === "L1") {
      const policyIds = Array.from(new Set(
        this.deps.repos.runtime
          .listTracePolicyLinks({ l1MemoryId: memory.id, limit: 1000 })
          .map((link) => link.l2MemoryId)
      ));
      for (const policyId of policyIds) {
        const updated = this.policy.recomputePolicyStats(policyId, at);
        const updatedPolicy = updated ? policyMetaFromMemory(updated) : null;
        if (updatedPolicy?.status !== "active") {
          this.invalidatePolicyDependencies(policyId, at);
        }
      }
      return;
    }
    if (memory.memoryLayer === "L2") {
      this.invalidatePolicyDependencies(memory.id, at);
    }
  }

  private invalidatePolicyDependencies(policyId: string, at: string): void {
    this.worldModel.invalidatePolicySource(policyId, at);
    this.skill.invalidatePolicySource(policyId, at);
  }

  upsertEvolutionMemory(memory: MemoryRow): {
    memory: MemoryRow;
    created: boolean;
    previous?: MemoryRow;
  } {
    const previous = memory.memoryKey
      ? this.deps.repos.memories.getByKey(memory.memoryLayer, memory.memoryKey)
      : undefined;
    if (previous && this.isArchivedEvolutionMemory(previous)) {
      return {
        memory: this.deps.repos.memories.insert(memory),
        created: true
      };
    }
    return this.deps.repos.memories.upsertByKey(memory);
  }

  private isArchivedEvolutionMemory(memory: MemoryRow): boolean {
    if (memory.status === "archived") return true;
    if (memory.memoryLayer === "L2") {
      return policyMetaFromMemory(memory)?.status === "archived";
    }
    if (memory.memoryLayer === "Skill") {
      return skillMetaFromMemory(memory)?.status === "archived";
    }
    return false;
  }
}

export type { DecisionRepairSummary };
