import {
  L2_INDUCTION_PROMPT,
  buildPolicyDraft,
  detectDominantLanguage,
  l2CandidateIdFor,
  languageSteeringLine,
  policyMetaFromMemory,
  signatureFromTrace,
  skillMetaFromMemory,
  traceMetaFromMemory,
  tracePolicySimilarity
} from "../../algorithm/plugin-algorithms.js";
import {
  admitL2PolicyDraft,
  asFiniteNumberArray,
  decideL2ClusterJoin,
  decideL2EvolveBranch,
  inferL2LessonKind,
  l2PolicyKeyForCluster,
  mergeL2LessonKind,
  packL2ClusterRawTurns,
  parseL2LessonKind,
  renderL2ClusterEvidence,
  renderL2InductionSystem,
  resolveL2LessonKind,
  SEED_META_POLICY_MD,
  type L2ClusteringConfig,
  type L2LessonKind
} from "../../algorithm/l2-cluster.js";
import { mergeCentroid } from "../../algorithm/trace-direct-skill.js";
import type { MemmyConfig } from "../../config/index.js";
import type { LlmClient } from "../../model/types.js";
import type {
  DecisionRepairRecord,
  EpisodeRecord,
  EvolutionJobRecord,
  FeedbackRecord,
  L2ClusterRecord,
  RawTurnRecord
} from "../../storage/repositories.js";
import { kindFromMemory } from "../../storage/repositories.js";
import type { MemoryRow } from "../../types.js";
import { isRecord } from "../../utils/json.js";
import { newId } from "../../utils/id.js";
import { clip } from "../../utils/text.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import { synthesizeFailureExperienceSink } from "../feedback/feedback-experience.js";
import { isInternalInfoEligibleForPositiveL2 } from "./span-pipeline.js";
import { logEvolutionDecision } from "./evolution-logging.js";

export type PolicyDraft = ReturnType<typeof buildPolicyDraft> & {
  expectedOutcome?: string;
  exclusions?: string[];
  freshnessClass?: "stable" | "dynamic";
  revalidateAfterDays?: number;
  lessonKind?: L2LessonKind;
};
export type PolicyEnhancementResult =
  | { ok: true; draft: PolicyDraft }
  | { ok: false; reason: string };
type TraceMeta = NonNullable<ReturnType<typeof traceMetaFromMemory>>;
type PolicyMeta = NonNullable<ReturnType<typeof policyMetaFromMemory>>;
const L2_INDUCTION_DRAFT_MAX_ATTEMPTS = 3;

type JobChangeKind = "created" | "updated" | "skipped";
type PolicyLifecycleStatus =
  | "candidate"
  | "active"
  | "verification_required"
  | "quarantined"
  | "superseded"
  | "archived";

type CandidatePoolRecord = {
  sourceMemoryId: string;
  candidateKey: string;
  createdAt?: string;
  updatedAt?: string;
  status?: string;
  sourceEpisodeId?: string;
};

type PolicyTraceLink = {
  l1MemoryId: string;
  l2MemoryId: string;
};

type ReposPort = {
  memories: {
    get(id: string): MemoryRow | undefined;
    getByKey(memoryLayer: "L2", memoryKey: string): MemoryRow | undefined;
    list(filter: Record<string, unknown>, limit?: number): MemoryRow[];
    getMany(ids: string[]): MemoryRow[];
    update(memory: MemoryRow): MemoryRow;
  };
  runtime: {
    pruneCandidatePool(now: string): number;
    deletePendingCandidatePoolForSource(sourceMemoryId: string): void;
    insertL2Cluster(cluster: L2ClusterRecord): L2ClusterRecord;
    updateL2Cluster(cluster: L2ClusterRecord): L2ClusterRecord;
    getL2Cluster(id: string): L2ClusterRecord | undefined;
    getL2ClusterForL1(l1MemoryId: string): L2ClusterRecord | undefined;
    listL2ClustersByUser(userId: string, limit?: number): L2ClusterRecord[];
    listL2ClusterMembers(clusterId: string): Array<{ l1MemoryId: string }>;
    getRawTurn(id: string): RawTurnRecord | undefined;
    getEpisode(id: string): EpisodeRecord | undefined;
    listRawTurnsByEpisode(episodeId: string): RawTurnRecord[];
    getDecisionRepair(id: string): DecisionRepairRecord | undefined;
    getFeedback(id: string): FeedbackRecord | undefined;
    upsertL2ClusterMember(member: {
      clusterId: string;
      l1MemoryId: string;
      assignReason: string;
      intentCosine?: number;
      taskCosine?: number;
      assignedAt: string;
    }): unknown;
    listPendingCandidatePool(input: {
      userId?: string;
      now?: string;
      limit?: number;
    }): CandidatePoolRecord[];
    upsertCandidatePoolTrace(input: {
      id: string;
      userId: string;
      sessionId?: string;
      sourceMemoryId: string;
      candidateKey: string;
      candidateValue: string;
      score: number;
      evidence: unknown;
      createdAt: string;
      updatedAt: string;
      expiresAt?: string | null;
    }): void;
    markCandidatePoolPromoted(input: {
      userId?: string;
      candidateKey: string;
      sourceMemoryIds: string[];
      policyId: string;
      at: string;
    }): void;
    listTracePolicyLinks(input: {
      userId?: string;
      l1MemoryId?: string;
      l2MemoryId?: string;
      limit?: number;
    }): PolicyTraceLink[];
    insertTracePolicyLink(input: {
      userId: string;
      l1MemoryId: string;
      l2MemoryId: string;
      relation?: string;
      strength?: number;
      createdAt?: string;
    }): string;
    appendEpisodeDerivedMemory(
      episodeId: string,
      layer: "L1" | "L2" | "L3" | "Skill",
      memoryId: string,
      at: string
    ): void;
  };
};

export interface PolicyInductionDeps {
  config: MemmyConfig;
  repos: ReposPort;
  nowIso: () => string;
  skillLlm: LlmClient;
  traceMeta(memory: MemoryRow | null | undefined): TraceMeta | null;
  projectIdFromMemory(memory: MemoryRow): string | undefined;
  profileIdFromMemory(memory: MemoryRow): string | undefined;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  upsertEvolutionMemory(memory: MemoryRow): {
    memory: MemoryRow;
    created: boolean;
    previous?: MemoryRow;
  };
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
  enqueueChange(input: {
    memoryId: string;
    namespaceId?: string;
    kind: string;
    op: JobChangeKind;
    entityId: string;
    userId: string;
    changeType: string;
    before?: unknown;
    after: unknown;
    source: string;
    createdAt: string;
  }): void;
  namespaceIdFromMemory(memory: MemoryRow): string | undefined;
  onSkillRewardDrift(policy: PolicyMeta, at: string): void;
  queryVector?(text: string): Promise<number[] | undefined>;
}

export class PolicyInductionEngine {
  constructor(private readonly deps: PolicyInductionDeps) {}

  async induceL2(job: EvolutionJobRecord): Promise<void> {
    const source = this.l2InductionSourceForJob(job);
    if (!source) {
      const sourceMemoryId = typeof job.payload?.sourceMemoryId === "string"
        ? job.payload.sourceMemoryId
        : typeof job.payload?.l1MemoryId === "string"
        ? job.payload.l1MemoryId
        : job.targetMemoryId;
      throw new Error(`source memory not found for l2 induction: ${sourceMemoryId ?? job.id}`);
    }
    const sourceTrace = this.deps.traceMeta(source);
    if (!sourceTrace) {
      return;
    }
    const at = this.deps.nowIso();
    const sourceSignature = signatureFromTrace(sourceTrace);
    const sourceNamespaceId = this.deps.namespaceIdFromMemory(source);
    this.deps.repos.runtime.pruneCandidatePool(at);

    const assignMemories = this.deps.repos.memories
      .list({ userId: source.userId, memoryLayer: "L1", status: ["activated", "resolving"] }, 2000)
      .filter((memory) => {
        const trace = this.deps.traceMeta(memory);
        return Boolean(trace && this.isTraceClusterEligible(trace));
      })
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt) || left.id.localeCompare(right.id));
    for (const memory of assignMemories) {
      const trace = this.deps.traceMeta(memory);
      if (!trace || !this.isTraceClusterEligible(trace)) continue;
      const cluster = await this.assignPositiveL2Cluster(memory, at);
      this.deps.repos.runtime.deletePendingCandidatePoolForSource(trace.id);
      if (!cluster) continue;
      if (this.isTraceEligibleForL2(trace) || this.isTraceNegativeForL2(trace)) {
        this.recordCandidatePoolTrace(trace, cluster.id, at);
      }
    }

    const pendingCandidates = this.deps.repos.runtime.listPendingCandidatePool({
      userId: source.userId,
      now: at,
      limit: 2000
    }).filter((candidate) => {
      const memory = this.deps.repos.memories.get(candidate.sourceMemoryId);
      return memory?.userId === source.userId;
    });
    const clusterKeys = uniq(
      pendingCandidates
        .map((candidate) => candidate.candidateKey)
        .filter((clusterKey) => Boolean(this.deps.repos.runtime.getL2Cluster(clusterKey)))
    );

    for (const clusterKey of clusterKeys) {
      const cluster = this.deps.repos.runtime.getL2Cluster(clusterKey);
      if (!cluster) continue;
      const memberTraces = this.clusterMemberTraces(cluster.id, source.userId);
      const positiveTraces = memberTraces.filter((trace) => this.isTraceEligibleForL2(trace));
      const negativeTraces = memberTraces.filter((trace) => this.isTraceNegativeForL2(trace));
      const inductionTraces = positiveTraces.length > 0 ? positiveTraces : negativeTraces;
      if (inductionTraces.length === 0) {
        continue;
      }

      const signature = signatureFromTrace(inductionTraces[0] ?? sourceTrace);
      const distinctEpisodeCount = uniq(
        inductionTraces
          .map((trace) => trace.episodeId)
          .filter((id): id is string => Boolean(id))
      ).length;
      if (distinctEpisodeCount < this.deps.config.algorithm.l2Induction.minEpisodesForInduction) {
        logEvolutionDecision(job, "l2_induction", "gate_not_met", {
          sourceMemoryId: source.id,
          evidenceCount: memberTraces.length,
          distinctEpisodeCount,
          requiredEpisodes: this.deps.config.algorithm.l2Induction.minEpisodesForInduction
        });
        continue;
      }

      const processed = new Set(cluster.processedL1Ids ?? []);
      const newTraces = memberTraces.filter((trace) => !processed.has(trace.id));
      const matchingPolicy = this.findExistingPolicyForL2Bucket(clusterKey);
      const orphanFailurePolicies = this.findFailurePoliciesForCluster(
        cluster,
        memberTraces.map((trace) => trace.id)
      ).filter((policy) => policy.id !== matchingPolicy?.id);
      const branch = decideL2EvolveBranch({
        hasPolicy: Boolean(matchingPolicy),
        positiveCount: positiveTraces.length,
        negativeCount: negativeTraces.length,
        newPositiveCount: newTraces.filter((trace) => this.isTraceEligibleForL2(trace)).length,
        newNegativeCount: newTraces.filter((trace) => this.isTraceNegativeForL2(trace)).length
      });
      const bucketTraceIds = memberTraces.map((trace) => trace.id);
      const inferredLessonKind = inferL2LessonKind(positiveTraces.length, negativeTraces.length);
      if (!inferredLessonKind || branch.action === "skip_no_positive") {
        continue;
      }

      if (positiveTraces.length === 0) {
        await this.induceFailureClusterPolicy({
          job,
          source,
          sourceTrace,
          clusterKey,
          signature,
          memberTraces,
          negativeTraces,
          matchingPolicy,
          branch,
          bucketTraceIds,
          at
        });
        continue;
      }

      if (branch.action === "link_only" && matchingPolicy) {
        this.linkClusterMembers(source.userId, matchingPolicy.id, memberTraces, at);
        this.markCandidatePoolPromoted(source.userId, clusterKey, bucketTraceIds, matchingPolicy.id, at);
        this.recomputePolicyStats(matchingPolicy.id, at, sourceTrace.episodeId);
        this.archiveFailurePolicies(orphanFailurePolicies, matchingPolicy.id, at);
        this.clearClusterNegativeL2(clusterKey, at);
        continue;
      }

      const promptTraces = branch.action === "evolve" ? newTraces : memberTraces;
      const packedTurns = this.packClusterMemberTurns(promptTraces);
      const packedEvidence = renderL2ClusterEvidence({
        clusterId: clusterKey,
        seedIntent: cluster.seedIntent || signature,
        lessonKind: inferredLessonKind,
        mode: branch.action === "evolve" ? "evolve" : "create",
        incrementKind: branch.action === "evolve" ? branch.incrementKind : undefined,
        existingPolicy: matchingPolicy
          ? {
              title: matchingPolicy.title,
              trigger: matchingPolicy.trigger,
              procedure: matchingPolicy.procedure,
              verification: matchingPolicy.verification,
              boundary: matchingPolicy.boundary,
              lessonKind: matchingPolicy.lessonKind
            }
          : undefined,
        existingFailurePolicies: !matchingPolicy && orphanFailurePolicies.length > 0
          ? orphanFailurePolicies.map((policy) => ({
              title: policy.title,
              trigger: policy.trigger,
              procedure: policy.procedure,
              boundary: policy.boundary
            }))
          : undefined,
        turns: packedTurns,
        charCap: this.deps.config.algorithm.l2Induction.traceCharCap
      });

      const policyKey = l2PolicyKeyForCluster(clusterKey);
      const existingPolicyMemory = matchingPolicy?.memory
        ?? this.deps.repos.memories.getByKey("L2", policyKey);
      const existingPolicy = existingPolicyMemory && !this.isArchivedEvolutionMemory(existingPolicyMemory)
        ? policyMetaFromMemory(existingPolicyMemory)
        : null;

      const fallbackDraft = buildPolicyDraft({
        signature,
        evidenceTraces: memberTraces,
        allTraces: this.l2GainReferenceTraces(memberTraces, sourceTrace.episodeId),
        minSupport: this.deps.config.algorithm.l2Induction.minEpisodesForActivation,
        minGain: this.deps.config.algorithm.l2Induction.minGain,
        archiveGain: this.deps.config.algorithm.l2Induction.archiveGain,
        tauSoftmax: this.deps.config.algorithm.l2Induction.tauSoftmax,
        gainEmaAlpha: this.deps.config.algorithm.l2Induction.gainEmaAlpha,
        currentStatus: existingPolicy?.status === "active"
          ? "active"
          : existingPolicy?.status === "candidate"
            ? "candidate"
            : existingPolicy
              ? "archived"
              : undefined,
        currentGain: existingPolicy?.gain,
        currentSupport: existingPolicy?.support
      });

      const enhancement = await this.enhancePolicyDraft({
        packedEvidence,
        languageTexts: packedTurns.flatMap((turn) => [turn.user, turn.assistant]),
        fallback: fallbackDraft,
        inferredLessonKind,
        metaPolicyMd: cluster.metaPolicyMd,
        positiveCount: positiveTraces.length,
        negativeCount: negativeTraces.length
      });
      if (!enhancement.ok) {
        logEvolutionDecision(job, "l2_induction", enhancement.reason, {
          sourceMemoryId: source.id,
          evidenceCount: memberTraces.length,
          distinctEpisodeCount
        });
        this.deps.enqueueChange({
          memoryId: source.id,
          namespaceId: sourceNamespaceId,
          kind: "policy",
          op: "skipped",
          entityId: source.id,
          userId: source.userId,
          changeType: "l2_induction_skipped",
          after: { signature, reason: enhancement.reason, traceIds: bucketTraceIds },
          source: "worker.l2_induction.v7",
          createdAt: at
        });
        continue;
      }

      const lessonKind = existingPolicy?.lessonKind
        ? mergeL2LessonKind(
            existingPolicy.lessonKind,
            resolveL2LessonKind(inferredLessonKind, enhancement.draft.lessonKind)
          )
        : resolveL2LessonKind(inferredLessonKind, enhancement.draft.lessonKind);
      const draft = {
        ...enhancement.draft,
        key: policyKey,
        lessonKind
      };
      const repairs = this.collectClusterRepairs(memberTraces);
      const l2 = this.buildClusterPolicyMemory({
        source,
        draft,
        signature,
        clusterKey,
        memberTraces,
        at,
        existingPolicy: matchingPolicy ?? existingPolicy,
        repairs
      });

      const upsert = this.deps.upsertEvolutionMemory(l2);
      this.linkClusterMembers(source.userId, upsert.memory.id, memberTraces, at);
      this.markCandidatePoolPromoted(source.userId, clusterKey, bucketTraceIds, upsert.memory.id, at);
      this.bindClusterToPolicy(clusterKey, upsert.memory.id, at);
      this.markClusterProcessed(clusterKey, memberTraces.map((trace) => trace.id), at);
      this.archiveFailurePolicies(orphanFailurePolicies, upsert.memory.id, at);
      this.clearClusterNegativeL2(clusterKey, at);

      this.deps.enqueueChange({
        memoryId: upsert.memory.id,
        namespaceId: this.deps.namespaceIdFromMemory(upsert.memory),
        kind: kindFromMemory(upsert.memory),
        op: upsert.created ? "created" : "updated",
        entityId: upsert.memory.id,
        userId: source.userId,
        changeType: upsert.created ? "create" : "update",
        before: upsert.previous,
        after: upsert.memory,
        source: "worker.l2_induction.v7",
        createdAt: at
      });
      if (this.deps.config.algorithm.capture.embedAfterCapture) {
        this.deps.enqueueJob({
          jobType: "embedding",
          userId: source.userId,
          sessionId: source.sessionId,
          episodeId: sourceTrace.episodeId,
          targetMemoryId: upsert.memory.id,
          payload: { reason: "l2.upserted" },
          createdAt: at
        });
      }
      this.deps.enqueueJob({
        jobType: "skill_crystallization",
        userId: source.userId,
        sessionId: source.sessionId,
        episodeId: sourceTrace.episodeId,
        targetMemoryId: upsert.memory.id,
        payload: { signature },
        createdAt: at
      });
    }
  }

  l2InductionSourceForJob(job: EvolutionJobRecord): MemoryRow | undefined {
    const payloadSourceMemoryId = typeof job.payload?.sourceMemoryId === "string"
      ? job.payload.sourceMemoryId
      : typeof job.payload?.l1MemoryId === "string"
      ? job.payload.l1MemoryId
      : undefined;
    const payloadSource = payloadSourceMemoryId ? this.deps.repos.memories.get(payloadSourceMemoryId) : undefined;
    if (payloadSource && this.deps.traceMeta(payloadSource)) {
      return payloadSource;
    }
    const legacyTarget = job.targetMemoryId ? this.deps.repos.memories.get(job.targetMemoryId) : undefined;
    if (legacyTarget && this.deps.traceMeta(legacyTarget)) {
      return legacyTarget;
    }
    return payloadSource ?? legacyTarget;
  }

  associateL2(job: EvolutionJobRecord): void {
    const source = job.targetMemoryId ? this.deps.repos.memories.get(job.targetMemoryId) : undefined;
    if (!source) return;
    const trace = this.deps.traceMeta(source);
    if (!trace || !this.isTraceEligibleForL2(trace)) return;
    const signature = signatureFromTrace(trace);
    const policies = this.deps.repos.memories
      .list({ memoryLayer: "L2", status: ["activated", "resolving"] }, 1000)
      .map((policyMemory) => policyMetaFromMemory(policyMemory))
      .filter((policy): policy is PolicyMeta => Boolean(policy));

    const at = this.deps.nowIso();
    let best: { policy: PolicyMeta; similarity: ReturnType<typeof tracePolicySimilarity> } | null = null;
    for (const policy of policies) {
      const similarity = tracePolicySimilarity(trace, policy);
      if (!best || similarity.score > best.similarity.score) {
        best = { policy, similarity };
      }
    }
    if (!best || best.similarity.score < this.deps.config.algorithm.l2Induction.minSimilarity) {
      return;
    }

    const matchedPolicy = best.policy;
    this.deps.repos.runtime.insertTracePolicyLink({
      userId: source.userId,
      l1MemoryId: source.id,
      l2MemoryId: matchedPolicy.id,
      relation: matchedPolicy.signature === signature ? "matches_signature" : "similar_pattern",
      strength: best.similarity.cosine,
      createdAt: at
    });
    if (trace.episodeId) {
      this.deps.repos.runtime.appendEpisodeDerivedMemory(trace.episodeId, "L2", matchedPolicy.id, at);
    }
    this.recomputePolicyStats(matchedPolicy.id, at, trace.episodeId);
  }

  async enhancePolicyDraft(input: {
    packedEvidence: string;
    languageTexts: string[];
    fallback: PolicyDraft;
    inferredLessonKind: L2LessonKind;
    metaPolicyMd?: string;
    positiveCount: number;
    negativeCount: number;
  }): Promise<PolicyEnhancementResult> {
    if (!this.deps.config.algorithm.l2Induction.useLlm || !this.deps.skillLlm.isConfigured()) {
      return { ok: false, reason: "llm_disabled" };
    }

    try {
      let lastInvalidReason: string | null = null;
      for (let attempt = 0; attempt < L2_INDUCTION_DRAFT_MAX_ATTEMPTS; attempt += 1) {
        const result = await this.deps.skillLlm.completeJson<{
          title?: unknown;
          trigger?: unknown;
          action?: unknown;
          procedure?: unknown;
          verification?: unknown;
          boundary?: unknown;
          rationale?: unknown;
          caveats?: unknown;
          confidence?: unknown;
          support_trace_ids?: unknown;
          tags?: unknown;
          should_generate?: unknown;
          shouldGenerate?: unknown;
          expected_outcome?: unknown;
          expectedOutcome?: unknown;
          exclusions?: unknown;
          freshness_class?: unknown;
          revalidate_after_days?: unknown;
          lesson_kind?: unknown;
        }>([
          {
            role: "system",
            content: `${renderL2InductionSystem(input.metaPolicyMd)}\n\n${L2_INDUCTION_PROMPT.system}`
          },
          {
            role: "system",
            content: languageSteeringLine(detectDominantLanguage(input.languageTexts))
          },
          {
            role: "user",
            content: input.packedEvidence
          }
        ], {
          operation: `${L2_INDUCTION_PROMPT.id}.v${L2_INDUCTION_PROMPT.version}`,
          thinkingMode: "enabled",
          temperature: 0.1,
          maxTokens: 1200
        });

        if (result.should_generate === false || result.shouldGenerate === false) {
          return { ok: false, reason: "generator-declined:no-reusable-policy" };
        }
        const invalidReason = l2InductionInvalidReason(result);
        if (invalidReason) {
          lastInvalidReason = invalidReason;
          continue;
        }

        const exclusions = stringArray(result.exclusions).map(skillMarkdown).filter(Boolean);
        const caveats = stringArray(result.caveats).map(skillMarkdown).filter(Boolean);
        const boundary = typeof result.boundary === "string"
          ? skillMarkdown(result.boundary)
          : exclusions.join("; ") || caveats.join("; ") || input.fallback.boundary;
        const procedure = skillMarkdown(firstString(result.procedure, result.action));
        const verification = typeof result.verification === "string"
          ? skillMarkdown(result.verification)
          : input.fallback.verification;
        const expectedOutcome = skillMarkdown(result.expected_outcome ?? result.expectedOutcome) || verification;
        const freshnessClass = result.freshness_class === "dynamic"
          ? "dynamic"
          : result.freshness_class === "stable"
            ? "stable"
            : inferPolicyFreshness(`${result.trigger ?? ""}\n${procedure}`);
        const revalidateAfterDays = freshnessClass === "dynamic"
          ? clampNumber(numberOr(result.revalidate_after_days, 30), 1, 365)
          : undefined;
        const next = {
          ...input.fallback,
          title: skillText(result.title),
          trigger: skillMarkdown(result.trigger),
          procedure,
          verification,
          boundary,
          expectedOutcome,
          exclusions,
          freshnessClass,
          revalidateAfterDays,
          lessonKind: resolveL2LessonKind(
            input.inferredLessonKind,
            parseL2LessonKind(result.lesson_kind)
          ),
          confidence: clampNumber(numberOr(result.confidence, input.fallback.confidence), 0, 1)
        };
        const admission = admitL2PolicyDraft({
          title: next.title,
          trigger: next.trigger,
          procedure: next.procedure,
          caveats,
          exclusions,
          lessonKind: next.lessonKind,
          positiveCount: input.positiveCount,
          negativeCount: input.negativeCount
        });
        if (!admission.ok) {
          lastInvalidReason = admission.reason;
          continue;
        }

        return {
          ok: true,
          draft: {
            ...next,
            body: renderPolicyBody(next)
          }
        };
      }

      return {
        ok: false,
        reason: lastInvalidReason ?? "llm-failed: l2.induction.invalid: unknown"
      };
    } catch (error) {
      return {
        ok: false,
        reason: `llm-failed: ${errorMessageFromUnknown(error) ?? "unknown"}`
      };
    }
  }

  private async induceFailureClusterPolicy(input: {
    job: EvolutionJobRecord;
    source: MemoryRow;
    sourceTrace: TraceMeta;
    clusterKey: string;
    signature: string;
    memberTraces: TraceMeta[];
    negativeTraces: TraceMeta[];
    matchingPolicy: PolicyMeta | null;
    branch: ReturnType<typeof decideL2EvolveBranch>;
    bucketTraceIds: string[];
    at: string;
  }): Promise<void> {
    const {
      job,
      source,
      sourceTrace,
      clusterKey,
      signature,
      memberTraces,
      negativeTraces,
      matchingPolicy,
      branch,
      bucketTraceIds,
      at
    } = input;
    const repairs = this.collectClusterRepairs(memberTraces);

    if (branch.action === "link_only" && matchingPolicy) {
      this.linkClusterMembers(source.userId, matchingPolicy.id, memberTraces, at);
      this.markCandidatePoolPromoted(source.userId, clusterKey, bucketTraceIds, matchingPolicy.id, at);
      this.bindClusterToPolicy(clusterKey, matchingPolicy.id, at);
      this.markClusterProcessed(clusterKey, memberTraces.map((trace) => trace.id), at);
      this.mergeRepairsOntoPolicy(matchingPolicy, repairs, at);
      return;
    }

    const sinkEpisodeId = [...negativeTraces].reverse().find((trace) => trace.episodeId)?.episodeId;
    const episode = sinkEpisodeId ? this.deps.repos.runtime.getEpisode(sinkEpisodeId) : undefined;
    if (!episode) {
      logEvolutionDecision(job, "l2_induction", "gate_not_met", {
        sourceMemoryId: source.id,
        reason: "no_failure_episode",
        evidenceCount: memberTraces.length
      });
      return;
    }

    const rawTurns = this.deps.repos.runtime.listRawTurnsByEpisode(episode.id);
    const feedbackIds = uniq([
      ...episode.feedbackIds,
      ...memberTraces
        .map((trace) => stringField(trace.memory.properties.internal_info.source_feedback_id))
        .filter(Boolean)
    ]);
    const feedbacks = feedbackIds
      .map((id) => this.deps.repos.runtime.getFeedback(id))
      .filter((item): item is FeedbackRecord => Boolean(item));
    const feedbackText = [
      ...feedbacks.map((item) => item.rationale),
      ...repairs.map((repair) => [repair.issue, repair.preference, repair.antiPattern].filter(Boolean).join("\n")),
      typeof episode.rewardDetail.reason === "string" ? episode.rewardDetail.reason : ""
    ].filter(Boolean).join("\n");
    const episodeContext = rawTurns
      .map((turn, index) => [
        `TURN ${index + 1}`,
        turn.userText ? `User: ${clip(turn.userText, 700)}` : "",
        turn.assistantText ? `Agent: ${clip(turn.assistantText, 900)}` : ""
      ].filter(Boolean).join("\n"))
      .join("\n\n");
    const sink = await synthesizeFailureExperienceSink({
      feedbackText,
      userRequest: rawTurns.find((turn) => Boolean(turn.userText?.trim()))?.userText?.trim() ?? "",
      agentResponse: rawTurns.at(-1)?.assistantText?.trim() ?? "",
      episodeContext,
      allowedTraceIds: memberTraces.map((trace) => trace.id)
    }, { llm: this.deps.skillLlm });
    if (!sink || !isActionableFailureSink(sink)) {
      logEvolutionDecision(job, "l2_induction", "gate_not_met", {
        sourceMemoryId: source.id,
        reason: sink ? "failure_sink_not_actionable" : "failure_sink_missing",
        evidenceCount: memberTraces.length
      });
      return;
    }

    const policyKey = l2PolicyKeyForCluster(clusterKey);
    const sourceEpisodeIds = uniq([
      ...(matchingPolicy?.sourceEpisodeIds ?? []),
      ...memberTraces.map((trace) => trace.episodeId).filter((id): id is string => Boolean(id))
    ]);
    const sourceTraceIds = uniq([
      ...(matchingPolicy?.sourceTraceIds ?? []),
      ...sink.supportTraceIds,
      ...memberTraces.map((trace) => trace.id)
    ]);
    const draft = {
      key: policyKey,
      title: sink.title,
      trigger: sink.trigger,
      procedure: sink.procedure,
      verification: sink.verification,
      boundary: sink.boundary,
      support: Math.max(1, sourceEpisodeIds.length),
      gain: matchingPolicy?.gain ?? 0,
      rawGain: 0,
      confidence: sink.confidence,
      status: matchingPolicy?.status === "active" ? "active" as const : "candidate" as const,
      sourceEpisodeIds,
      sourceTraceIds,
      vec: matchingPolicy?.vec ?? null,
      tags: uniq([...(matchingPolicy?.memory.tags ?? []), "policy", "avoidance", "failure"]),
      body: renderFailureClusterBody({
        title: sink.title,
        trigger: sink.trigger,
        antiPattern: sink.avoid.join("\n"),
        procedure: sink.procedure,
        verification: sink.verification,
        boundary: sink.boundary,
        support: Math.max(1, sourceEpisodeIds.length),
        confidence: sink.confidence
      }),
      lessonKind: matchingPolicy?.lessonKind
        ? mergeL2LessonKind(matchingPolicy.lessonKind, "error_correction")
        : "error_correction"
    } satisfies PolicyDraft;

    const l2 = this.buildClusterPolicyMemory({
      source,
      draft,
      signature,
      clusterKey,
      memberTraces,
      at,
      existingPolicy: matchingPolicy,
      repairs,
      experienceType: sink.experienceType,
      skillEligible: false,
      decisionGuidance: {
        preference: sink.prefer,
        antiPattern: sink.avoid
      },
      sourceRepairIds: repairs.map((repair) => repair.id),
      sourceFeedbackIds: feedbacks.map((item) => item.id),
      evidenceStrength: Math.max(
        0,
        ...feedbacks.map((item) => item.magnitude),
        typeof episode.rTask === "number" ? Math.abs(episode.rTask) : 0
      )
    });
    const upsert = this.deps.upsertEvolutionMemory(l2);
    this.linkClusterMembers(source.userId, upsert.memory.id, memberTraces, at);
    this.markCandidatePoolPromoted(source.userId, clusterKey, bucketTraceIds, upsert.memory.id, at);
    this.bindClusterToPolicy(clusterKey, upsert.memory.id, at);
    this.markClusterProcessed(clusterKey, memberTraces.map((trace) => trace.id), at);
    this.deps.enqueueChange({
      memoryId: upsert.memory.id,
      namespaceId: this.deps.namespaceIdFromMemory(upsert.memory),
      kind: kindFromMemory(upsert.memory),
      op: upsert.created ? "created" : "updated",
      entityId: upsert.memory.id,
      userId: source.userId,
      changeType: upsert.created ? "create" : "update",
      before: upsert.previous,
      after: upsert.memory,
      source: "worker.l2_induction.v7",
      createdAt: at
    });
    if (this.deps.config.algorithm.capture.embedAfterCapture) {
      this.deps.enqueueJob({
        jobType: "embedding",
        userId: source.userId,
        sessionId: source.sessionId,
        episodeId: sourceTrace.episodeId,
        targetMemoryId: upsert.memory.id,
        payload: { reason: "l2.upserted" },
        createdAt: at
      });
    }
  }

  private collectClusterRepairs(traces: TraceMeta[]): DecisionRepairRecord[] {
    const seen = new Set<string>();
    const repairs: DecisionRepairRecord[] = [];
    for (const trace of traces) {
      if (!trace.episodeId) continue;
      const episode = this.deps.repos.runtime.getEpisode(trace.episodeId);
      for (const id of episode?.decisionRepairIds ?? []) {
        if (seen.has(id)) continue;
        const repair = this.deps.repos.runtime.getDecisionRepair(id);
        if (!repair) continue;
        seen.add(id);
        repairs.push(repair);
      }
    }
    return repairs;
  }

  private mergeRepairsOntoPolicy(
    policy: PolicyMeta,
    repairs: DecisionRepairRecord[],
    at: string
  ): void {
    if (repairs.length === 0) return;
    const next = this.buildClusterPolicyMemory({
      source: policy.memory,
      draft: {
        key: policy.memory.memoryKey ?? l2PolicyKeyForCluster(""),
        title: policy.title,
        trigger: policy.trigger,
        procedure: policy.procedure,
        verification: policy.verification,
        boundary: policy.boundary,
        support: policy.support,
        gain: policy.gain,
        rawGain: policy.salience,
        confidence: policy.confidence,
        status: policy.status === "active" ? "active" : "candidate",
        sourceEpisodeIds: policy.sourceEpisodeIds,
        sourceTraceIds: policy.sourceTraceIds,
        vec: policy.vec,
        tags: policy.memory.tags ?? [],
        body: policy.memory.memoryValue,
        lessonKind: policy.lessonKind
      },
      signature: policy.signature,
      clusterKey: stringField(policy.memory.properties.internal_info.l2_cluster_id),
      memberTraces: policy.sourceTraceIds
        .map((id) => this.deps.repos.memories.get(id))
        .map((memory) => this.deps.traceMeta(memory))
        .filter((trace): trace is TraceMeta => Boolean(trace)),
      at,
      existingPolicy: policy,
      repairs,
      experienceType: policy.experienceType,
      skillEligible: policy.skillEligible
    });
    this.deps.upsertEvolutionMemory(next);
  }

  async bindFailureL2(memory: MemoryRow, sourceTraceIds: string[], at: string): Promise<void> {
    const seen = new Set<string>();
    for (const id of sourceTraceIds) {
      const l1 = this.deps.repos.memories.get(id);
      if (!l1) continue;
      const trace = this.deps.traceMeta(l1);
      if (!trace || !this.isTraceClusterEligible(trace)) continue;
      const cluster = await this.assignPositiveL2Cluster(l1, at);
      if (!cluster || seen.has(cluster.id)) continue;
      seen.add(cluster.id);
      if (cluster.negativeL2MemoryId) continue;
      this.deps.repos.runtime.updateL2Cluster({
        ...cluster,
        negativeL2MemoryId: memory.id,
        updatedAt: at
      });
    }
  }

  private findExistingPolicyForL2Bucket(clusterKey: string): PolicyMeta | null {
    const byKey = this.deps.repos.memories.getByKey("L2", l2PolicyKeyForCluster(clusterKey));
    if (byKey && !this.isArchivedEvolutionMemory(byKey)) {
      return policyMetaFromMemory(byKey);
    }
    const cluster = this.deps.repos.runtime.getL2Cluster(clusterKey);
    if (!cluster?.l2MemoryId) return null;
    const linked = this.deps.repos.memories.get(cluster.l2MemoryId);
    if (!linked || this.isArchivedEvolutionMemory(linked)) return null;
    return policyMetaFromMemory(linked);
  }

  private isFailurePolicy(policy: PolicyMeta): boolean {
    return policy.experienceType === "failure_avoidance"
      || policy.experienceType === "repair_instruction"
      || policy.evidencePolarity === "negative";
  }

  private findFailurePoliciesForCluster(cluster: L2ClusterRecord, memberTraceIds: string[]): PolicyMeta[] {
    const found = new Map<string, PolicyMeta>();
    const add = (memory: MemoryRow | undefined) => {
      if (!memory || this.isArchivedEvolutionMemory(memory)) return;
      const policy = policyMetaFromMemory(memory);
      if (!policy || !this.isFailurePolicy(policy)) return;
      found.set(policy.id, policy);
    };
    if (cluster.negativeL2MemoryId) {
      add(this.deps.repos.memories.get(cluster.negativeL2MemoryId));
    }
    const memberSet = new Set(memberTraceIds);
    for (const memory of this.deps.repos.memories.list(
      { userId: cluster.userId, memoryLayer: "L2", status: ["activated", "resolving"] },
      1000
    )) {
      const policy = policyMetaFromMemory(memory);
      if (!policy || !this.isFailurePolicy(policy)) continue;
      if (policy.sourceTraceIds.some((id) => memberSet.has(id))) {
        add(memory);
      }
    }
    return [...found.values()];
  }

  private clearClusterNegativeL2(clusterId: string, at: string): void {
    const cluster = this.deps.repos.runtime.getL2Cluster(clusterId);
    if (!cluster?.negativeL2MemoryId) return;
    this.deps.repos.runtime.updateL2Cluster({
      ...cluster,
      negativeL2MemoryId: undefined,
      updatedAt: at
    });
  }

  private archiveFailurePolicies(policies: PolicyMeta[], replacedBy: string, at: string): void {
    for (const policy of policies) {
      const archived = updatePolicyStats(policy.memory, {
        support: policy.support,
        gain: policy.gain,
        rawGain: policy.salience,
        status: "archived",
        sourceEpisodeIds: policy.sourceEpisodeIds,
        sourceTraceIds: policy.sourceTraceIds,
        updatedAt: at
      });
      const currentPolicy = isRecord(archived.properties.internal_info.policy)
        ? archived.properties.internal_info.policy
        : {};
      const saved = this.deps.repos.memories.update({
        ...archived,
        properties: {
          ...archived.properties,
          internal_info: {
            ...archived.properties.internal_info,
            policy: {
              ...currentPolicy,
              superseded_by: replacedBy,
              status: "archived"
            }
          }
        }
      });
      this.deps.enqueueChange({
        memoryId: saved.id,
        namespaceId: this.deps.namespaceIdFromMemory(saved),
        kind: kindFromMemory(saved),
        op: "updated",
        entityId: saved.id,
        userId: saved.userId,
        changeType: "negative_l2_archived",
        before: policy.memory,
        after: saved,
        source: "worker.l2_induction.v7",
        createdAt: at
      });
    }
  }

  private async assignPositiveL2Cluster(memory: MemoryRow, at: string): Promise<L2ClusterRecord | null> {
    const existingId = stringField(memory.properties.internal_info.l2_cluster_id);
    const existing = existingId ? this.deps.repos.runtime.getL2Cluster(existingId) : undefined;
    if (existing) {
      if (!existingId || existingId !== existing.id) {
        this.persistL1ClusterAssignment(memory, {
          clusterId: existing.id,
          intentVec: existing.intentCentroid ?? asFiniteNumberArray(memory.properties.internal_info.intent_vec),
          taskVec: existing.taskCentroid ?? asFiniteNumberArray(memory.properties.internal_info.task_vec),
          at
        });
      }
      this.deps.repos.runtime.upsertL2ClusterMember({
        clusterId: existing.id,
        l1MemoryId: memory.id,
        assignReason: "already_assigned",
        assignedAt: at
      });
      return existing;
    }

    const intent = stringField(memory.properties.internal_info.intent);
    if (!intent) return null;
    const taskSummary = stringField(memory.properties.internal_info.task_summary);
    const intentVec = asFiniteNumberArray(memory.properties.internal_info.intent_vec)
      ?? await this.embedClusterText(intent);
    if (!intentVec) return null;
    const taskVec = asFiniteNumberArray(memory.properties.internal_info.task_vec)
      ?? (taskSummary ? await this.embedClusterText(taskSummary) : null);

    const candidates = this.deps.repos.runtime.listL2ClustersByUser(memory.userId);
    const decision = decideL2ClusterJoin(
      { intent, intentVec, taskVec },
      candidates.map((cluster) => ({
        id: cluster.id,
        intentCentroid: cluster.intentCentroid,
        taskCentroid: cluster.taskCentroid
      })),
      this.clusteringConfig()
    );
    if (decision.action === "skip") return null;

    let cluster: L2ClusterRecord;
    if (decision.action === "join") {
      const joined = this.deps.repos.runtime.getL2Cluster(decision.clusterId);
      if (!joined) {
        cluster = this.createL2Cluster(memory, intent, taskSummary, intentVec, taskVec, at);
      } else {
        cluster = this.deps.repos.runtime.updateL2Cluster({
          ...joined,
          intentCentroid: mergeCentroid(joined.intentCentroid, intentVec, joined.memberCount),
          taskCentroid: mergeCentroid(joined.taskCentroid, taskVec, joined.memberCount),
          memberCount: joined.memberCount + 1,
          updatedAt: at
        });
      }
    } else {
      cluster = this.createL2Cluster(memory, intent, taskSummary, intentVec, taskVec, at);
    }

    this.persistL1ClusterAssignment(memory, { clusterId: cluster.id, intentVec, taskVec, at });
    this.deps.repos.runtime.upsertL2ClusterMember({
      clusterId: cluster.id,
      l1MemoryId: memory.id,
      assignReason: decision.action === "join" ? decision.reason : decision.reason,
      intentCosine: decision.action === "join" ? decision.intent : decision.best?.intent,
      taskCosine: decision.action === "join"
        ? decision.task ?? undefined
        : decision.best?.task ?? undefined,
      assignedAt: at
    });
    return cluster;
  }

  private createL2Cluster(
    memory: MemoryRow,
    intent: string,
    taskSummary: string,
    intentVec: number[],
    taskVec: number[] | null,
    at: string
  ): L2ClusterRecord {
    return this.deps.repos.runtime.insertL2Cluster({
      id: newId("l2c"),
      userId: memory.userId,
      intentCentroid: intentVec,
      taskCentroid: taskVec,
      seedIntent: intent,
      seedTaskSummary: taskSummary,
      processedL1Ids: [],
      metaPolicyMd: SEED_META_POLICY_MD,
      memberCount: 1,
      createdAt: at,
      updatedAt: at
    });
  }

  private persistL1ClusterAssignment(memory: MemoryRow, input: {
    clusterId: string;
    intentVec: number[] | null;
    taskVec?: number[] | null;
    at: string;
  }): MemoryRow {
    return this.deps.repos.memories.update({
      ...memory,
      properties: {
        ...memory.properties,
        internal_info: {
          ...memory.properties.internal_info,
          l2_cluster_id: input.clusterId,
          ...(input.intentVec ? { intent_vec: input.intentVec } : {}),
          ...(input.taskVec ? { task_vec: input.taskVec } : {})
        }
      },
      updatedAt: input.at
    });
  }

  private bindClusterToPolicy(clusterId: string, policyId: string, at: string): void {
    const cluster = this.deps.repos.runtime.getL2Cluster(clusterId);
    if (!cluster || cluster.l2MemoryId === policyId) return;
    this.deps.repos.runtime.updateL2Cluster({
      ...cluster,
      l2MemoryId: policyId,
      updatedAt: at
    });
  }

  private clusteringConfig(): L2ClusteringConfig {
    const l2 = this.deps.config.algorithm.l2Induction;
    return {
      intentJoin: l2.clusterIntentJoin,
      intentGrayMin: l2.clusterIntentGrayMin,
      taskGray: l2.clusterTaskGray
    };
  }

  private async embedClusterText(text: string): Promise<number[] | null> {
    const trimmed = text.trim();
    if (!trimmed || !this.deps.queryVector) return null;
    return asFiniteNumberArray(await this.deps.queryVector(trimmed));
  }

  recomputePolicyStats(policyId: string, at: string, triggerEpisodeId?: string): MemoryRow | undefined {
    const memory = this.deps.repos.memories.get(policyId);
    if (!memory || memory.memoryLayer !== "L2") return undefined;
    const policy = policyMetaFromMemory(memory);
    if (!policy) return undefined;

    const linkedTraceIds = new Set(
      this.deps.repos.runtime
        .listTracePolicyLinks({ l2MemoryId: policy.id, limit: 1000 })
        .map((link) => link.l1MemoryId)
    );

    const allTraces = this.deps.repos.memories
      .list({ memoryLayer: "L1", status: "activated" }, 1000)
      .map((memory) => this.deps.traceMeta(memory))
      .filter((trace): trace is TraceMeta =>
        Boolean(trace && (this.isTraceEligibleForL2(trace) || this.isTraceNegativeForL2(trace)))
      );

    const linkedTraces = allTraces.filter((trace) => linkedTraceIds.has(trace.id));
    const evidenceTraces = linkedTraces;
    if (evidenceTraces.length === 0) {
      const archived = this.deps.repos.memories.update(updatePolicyStats(memory, {
        support: 0,
        gain: policy.gain,
        rawGain: policy.gain,
        status: "quarantined",
        sourceEpisodeIds: [],
        sourceTraceIds: [],
        updatedAt: at
      }));
      this.deps.enqueueChange({
        memoryId: archived.id,
        namespaceId: this.deps.namespaceIdFromMemory(archived),
        kind: kindFromMemory(archived),
        op: "updated",
        entityId: archived.id,
        userId: archived.userId,
        changeType: "policy_evidence_invalidated",
        before: memory,
        after: archived,
        source: "governance.l1_invalidation",
        createdAt: at
      });
      return archived;
    }

    const gainReferenceTraces = this.l2GainReferenceTraces(evidenceTraces, triggerEpisodeId);
    const stats = buildPolicyDraft({
      signature: policy.signature,
      evidenceTraces,
      allTraces: gainReferenceTraces,
      minSupport: this.deps.config.algorithm.l2Induction.minEpisodesForActivation,
      minGain: this.deps.config.algorithm.l2Induction.minGain,
      archiveGain: this.deps.config.algorithm.l2Induction.archiveGain,
      tauSoftmax: this.deps.config.algorithm.l2Induction.tauSoftmax,
      gainEmaAlpha: this.deps.config.algorithm.l2Induction.gainEmaAlpha,
      currentStatus: policy.status === "active"
        ? "active"
        : policy.status === "archived"
          ? "archived"
          : "candidate",
      currentGain: policy.gain,
      currentSupport: policy.support
    });

    const previous = memory;
    const nextStatus: PolicyLifecycleStatus = policy.status === "active" && stats.status !== "active"
      ? "verification_required"
      : stats.status;
    const next = updatePolicyStats(memory, {
      support: stats.support,
      gain: stats.gain,
      rawGain: stats.rawGain,
      status: nextStatus,
      sourceEpisodeIds: stats.sourceEpisodeIds,
      sourceTraceIds: stats.sourceTraceIds,
      updatedAt: at
    });

    const saved = this.deps.repos.memories.update(next);
    for (const episodeId of stats.sourceEpisodeIds) {
      this.deps.repos.runtime.appendEpisodeDerivedMemory(episodeId, "L2", saved.id, at);
    }

    this.deps.enqueueChange({
      memoryId: saved.id,
      namespaceId: this.deps.namespaceIdFromMemory(saved),
      kind: kindFromMemory(saved),
      op: "updated",
      entityId: saved.id,
      userId: saved.userId,
      changeType: "policy_stats_update",
      before: previous,
      after: saved,
      source: "worker.l2_association.v7",
      createdAt: at
    });

    const savedPolicy = policyMetaFromMemory(saved);
    if (savedPolicy) {
      if (savedPolicy.status === "active") {
        this.deps.enqueueJob({
          jobType: "skill_crystallization",
          userId: saved.userId,
          sessionId: saved.sessionId,
          episodeId: triggerEpisodeId,
          targetMemoryId: saved.id,
          payload: {
            reason: "l2.policy.updated",
            previousStatus: policy.status,
            status: savedPolicy.status
          },
          createdAt: at
        });
      }
      this.deps.onSkillRewardDrift(savedPolicy, at);
    }
    return saved;
  }

  private l2GainReferenceTraces(evidenceTraces: TraceMeta[], triggerEpisodeId?: string): TraceMeta[] {
    const episodeIds = new Set(
      evidenceTraces
        .map((trace) => trace.episodeId)
        .filter((episodeId): episodeId is string => Boolean(episodeId))
    );
    if (triggerEpisodeId) {
      episodeIds.add(triggerEpisodeId);
    }

    const byId = new Map<string, TraceMeta>();
    for (const trace of evidenceTraces) {
      byId.set(trace.id, trace);
    }

    if (episodeIds.size === 0) {
      return [...byId.values()];
    }

    for (const trace of this.deps.repos.memories
      .list({ memoryLayer: "L1", status: "activated" }, 2000)
      .map((memory) => this.deps.traceMeta(memory))
      .filter((trace): trace is TraceMeta =>
        Boolean(trace && trace.episodeId && episodeIds.has(trace.episodeId) && Number.isFinite(trace.value))
      )) {
      byId.set(trace.id, trace);
    }

    return [...byId.values()].sort((a, b) => a.ts - b.ts || a.id.localeCompare(b.id));
  }

  recordCandidatePoolTrace(
    trace: TraceMeta,
    signature: string,
    at: string
  ): void {
    const id = l2CandidateIdFor(signature, trace.id);
    this.deps.repos.runtime.upsertCandidatePoolTrace({
      id,
      userId: trace.userId,
      sessionId: trace.sessionId,
      sourceMemoryId: trace.id,
      candidateKey: signature,
      candidateValue: trace.summary,
      score: trace.priority,
      evidence: {
        traceId: trace.id,
        episodeId: trace.episodeId,
        value: trace.value,
        priority: trace.priority,
        tags: trace.tags
      },
      createdAt: at,
      updatedAt: at,
      expiresAt: this.candidateExpiresAt(at)
    });
  }

  isTraceClusterEligible(trace: TraceMeta): boolean {
    return isInternalInfoEligibleForPositiveL2(trace.memory.properties.internal_info) &&
      trace.memory.properties.internal_info.evidence_status !== "provisional" &&
      trace.memory.properties.internal_info.evidence_status !== "disputed" &&
      Boolean(trace.vecSummary ?? trace.vecAction);
  }

  isTraceEligibleForL2(trace: TraceMeta): boolean {
    return this.isTraceClusterEligible(trace) &&
      trace.value >= this.deps.config.algorithm.l2Induction.minTraceValue;
  }

  isTraceNegativeForL2(trace: TraceMeta): boolean {
    return this.isTraceClusterEligible(trace) && trace.value < 0;
  }

  private clusterMemberTraces(clusterId: string, userId: string): TraceMeta[] {
    return this.deps.repos.runtime.listL2ClusterMembers(clusterId)
      .map((member) => this.deps.repos.memories.get(member.l1MemoryId))
      .map((memory) => this.deps.traceMeta(memory))
      .filter((trace): trace is TraceMeta =>
        Boolean(trace && trace.userId === userId && this.isTraceClusterEligible(trace))
      )
      .sort((left, right) => left.ts - right.ts || left.id.localeCompare(right.id));
  }

  private packClusterMemberTurns(traces: TraceMeta[]) {
    return packL2ClusterRawTurns({
      members: traces.map((trace) => ({
        l1MemoryId: trace.id,
        value: trace.value,
        rawTurn: trace.rawTurnId ? this.deps.repos.runtime.getRawTurn(trace.rawTurnId) : undefined,
        fallback: {
          userText: trace.userText,
          assistantText: trace.agentText,
          toolCalls: trace.toolCalls
        }
      })),
      minPositiveValue: this.deps.config.algorithm.l2Induction.minTraceValue,
      clipChars: this.deps.config.algorithm.l2Induction.traceCharCap
    });
  }

  private linkClusterMembers(
    userId: string,
    policyId: string,
    traces: TraceMeta[],
    at: string
  ): void {
    for (const trace of traces) {
      this.deps.repos.runtime.insertTracePolicyLink({
        userId,
        l1MemoryId: trace.id,
        l2MemoryId: policyId,
        relation: "supports",
        strength: Math.max(0, trace.value),
        createdAt: at
      });
      if (trace.episodeId) {
        this.deps.repos.runtime.appendEpisodeDerivedMemory(trace.episodeId, "L2", policyId, at);
      }
    }
  }

  private markClusterProcessed(clusterId: string, l1Ids: string[], at: string): void {
    const cluster = this.deps.repos.runtime.getL2Cluster(clusterId);
    if (!cluster) return;
    this.deps.repos.runtime.updateL2Cluster({
      ...cluster,
      processedL1Ids: uniq([...(cluster.processedL1Ids ?? []), ...l1Ids]),
      updatedAt: at
    });
  }

  private buildClusterPolicyMemory(input: {
    source: MemoryRow;
    draft: PolicyDraft;
    signature: string;
    clusterKey: string;
    memberTraces: TraceMeta[];
    at: string;
    existingPolicy?: PolicyMeta | null;
    repairs?: DecisionRepairRecord[];
    experienceType?: PolicyMeta["experienceType"];
    skillEligible?: boolean;
    decisionGuidance?: { preference: string[]; antiPattern: string[] };
    sourceRepairIds?: string[];
    sourceFeedbackIds?: string[];
    evidenceStrength?: number;
  }): MemoryRow {
    const freshnessClass = input.draft.freshnessClass
      ?? inferPolicyFreshness(`${input.draft.trigger}\n${input.draft.procedure}`);
    const revalidateAfter = freshnessClass === "dynamic"
      ? new Date(Date.parse(input.at) + (input.draft.revalidateAfterDays ?? 30) * 24 * 60 * 60 * 1000).toISOString()
      : undefined;
    const hasPositive = input.memberTraces.some((trace) => this.isTraceEligibleForL2(trace));
    const hasNegative = input.memberTraces.some((trace) => this.isTraceNegativeForL2(trace));
    const evidencePolarity = hasPositive && hasNegative
      ? "mixed"
      : hasNegative
        ? "negative"
        : "positive";
    const skillEligible = input.skillEligible ?? hasPositive;
    const experienceType = input.experienceType
      ?? (hasPositive ? "success_pattern" : "failure_avoidance");
    const sourceTraceIds = input.draft.sourceTraceIds.length > 0
      ? input.draft.sourceTraceIds
      : input.memberTraces.map((trace) => trace.id);
    const preferences = uniq([
      ...(input.existingPolicy?.decisionGuidance.preference ?? []),
      ...(input.decisionGuidance?.preference ?? []),
      ...(input.repairs ?? []).map((repair) => repair.preference).filter((item): item is string => Boolean(item?.trim()))
    ]);
    const antiPatterns = uniq([
      ...(input.existingPolicy?.decisionGuidance.antiPattern ?? []),
      ...(input.decisionGuidance?.antiPattern ?? []),
      ...(input.draft.exclusions ?? []),
      ...(input.repairs ?? []).map((repair) => repair.antiPattern).filter((item): item is string => Boolean(item?.trim()))
    ]);
    const sourceRepairIds = uniq([
      ...(input.sourceRepairIds ?? []),
      ...stringArray(input.existingPolicy?.memory.properties.internal_info.source_repair_ids),
      ...(input.repairs ?? []).map((repair) => repair.id)
    ]);
    const sourceFeedbackIds = uniq([
      ...(input.sourceFeedbackIds ?? []),
      ...(input.existingPolicy?.sourceFeedbackIds ?? [])
    ]);
    const decisionGuidance = preferences.length > 0 || antiPatterns.length > 0
      ? { preference: preferences, anti_pattern: antiPatterns }
      : undefined;
    return this.deps.buildMemory({
      userId: input.source.userId,
      conversationId: input.source.conversationId,
      sessionId: input.source.sessionId,
      agentId: input.source.agentId,
      appId: input.source.appId,
      projectId: this.deps.projectIdFromMemory(input.source),
      profileId: this.deps.profileIdFromMemory(input.source),
      layer: "L2",
      kind: "policy",
      lifecycleStatus: input.draft.status,
      memoryType: "LongTermMemory",
      key: input.draft.key,
      value: input.draft.body,
      tags: input.draft.tags,
      info: {
        signature: input.signature,
        support: input.draft.support,
        gain: input.draft.gain,
        raw_gain: input.draft.rawGain,
        policy_confidence: input.draft.confidence,
        freshness_class: freshnessClass,
        last_verified_at: input.at,
        ...(revalidateAfter ? { revalidate_after: revalidateAfter } : {}),
        status: input.draft.status,
        source_memory_ids: sourceTraceIds,
        l2_cluster_id: input.clusterKey
      },
      internal: {
        source: "worker.l2_induction.v7",
        plugin_algorithm: "l2.induction.v7",
        l2_cluster_id: input.clusterKey,
        source_memory_ids: sourceTraceIds,
        source_l1_memory_ids: sourceTraceIds,
        source_repair_ids: sourceRepairIds,
        source_feedback_ids: sourceFeedbackIds,
        title: input.draft.title,
        trigger: input.draft.trigger,
        procedure: input.draft.procedure,
        verification: input.draft.verification,
        boundary: input.draft.boundary,
        expected_outcome: input.draft.expectedOutcome,
        exclusions: input.draft.exclusions,
        support: input.draft.support,
        gain: input.draft.gain,
        raw_gain: input.draft.rawGain,
        policy_confidence: input.draft.confidence,
        freshness_class: freshnessClass,
        last_verified_at: input.at,
        ...(revalidateAfter ? { revalidate_after: revalidateAfter } : {}),
        status: input.draft.status,
        source_episode_ids: input.draft.sourceEpisodeIds,
        source_trace_ids: sourceTraceIds,
        ...(decisionGuidance ? { decision_guidance: decisionGuidance } : {}),
        policy: {
          title: input.draft.title,
          trigger: input.draft.trigger,
          procedure: input.draft.procedure,
          verification: input.draft.verification,
          boundary: input.draft.boundary,
          expected_outcome: input.draft.expectedOutcome,
          exclusions: input.draft.exclusions,
          support: input.draft.support,
          gain: input.draft.gain,
          raw_gain: input.draft.rawGain,
          policy_confidence: input.draft.confidence,
          freshness_class: freshnessClass,
          last_verified_at: input.at,
          ...(revalidateAfter ? { revalidate_after: revalidateAfter } : {}),
          status: input.draft.status,
          experience_type: experienceType,
          lesson_kind: input.draft.lessonKind ?? (hasPositive ? "path_compression" : "error_correction"),
          evidence_polarity: evidencePolarity,
          skill_eligible: skillEligible,
          is_caveat: !skillEligible,
          ...(input.evidenceStrength !== undefined ? { evidence_strength: input.evidenceStrength } : {}),
          signature: input.signature,
          source_episode_ids: input.draft.sourceEpisodeIds,
          source_trace_ids: sourceTraceIds,
          source_feedback_ids: sourceFeedbackIds,
          ...(decisionGuidance ? { decision_guidance: decisionGuidance } : {}),
          vec: input.draft.vec
        }
      },
      createdAt: input.at
    });
  }

  private markCandidatePoolPromoted(
    userId: string,
    signature: string,
    sourceMemoryIds: string[],
    policyId: string,
    at: string
  ): void {
    this.deps.repos.runtime.markCandidatePoolPromoted({
      userId,
      candidateKey: signature,
      sourceMemoryIds,
      policyId,
      at
    });
  }

  private candidateExpiresAt(at: string): string {
    const ttlMs = this.deps.config.algorithm.l2Induction.candidateTtlDays * 24 * 60 * 60 * 1000;
    return new Date(Date.parse(at) + ttlMs).toISOString();
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

export function updatePolicyStats(memory: MemoryRow, input: {
  support: number;
  gain: number;
  rawGain: number;
  status: PolicyLifecycleStatus;
  sourceEpisodeIds: string[];
  sourceTraceIds: string[];
  updatedAt: string;
}): MemoryRow {
  const currentPolicy = policyMetaFromMemory(memory);
  const internalPolicy = isRecord(memory.properties.internal_info.policy)
    ? memory.properties.internal_info.policy
    : {};

  const status = memoryStatusForLifecycleStatus(input.status);
  const nextPolicy = {
    ...internalPolicy,
    support: input.support,
    gain: input.gain,
    raw_gain: input.rawGain,
    status: input.status,
    source_episode_ids: input.sourceEpisodeIds,
    source_trace_ids: input.sourceTraceIds
  };

  const body = currentPolicy
    ? renderPolicyBody({
        title: currentPolicy.title,
        trigger: currentPolicy.trigger,
        procedure: currentPolicy.procedure,
        verification: currentPolicy.verification,
        boundary: currentPolicy.boundary,
        support: input.support,
        gain: input.gain,
        rawGain: input.rawGain,
        confidence: currentPolicy.confidence,
        sourceTraceIds: input.sourceTraceIds
      })
    : memory.memoryValue;

  return {
    ...memory,
    status,
    memoryValue: body,
    info: {
      ...memory.info,
      support: input.support,
      gain: input.gain,
      raw_gain: input.rawGain,
      policy_confidence: currentPolicy?.confidence,
      status: input.status,
      source_memory_ids: input.sourceTraceIds
    },
    properties: {
      ...memory.properties,
      status,
      info: {
        ...(memory.properties?.info ?? {}),
        support: input.support,
        gain: input.gain,
        raw_gain: input.rawGain,
        policy_confidence: currentPolicy?.confidence,
        status: input.status,
        source_memory_ids: input.sourceTraceIds
      },
      internal_info: {
        ...(memory.properties?.internal_info ?? {}),
        source_memory_ids: input.sourceTraceIds,
        source_l1_memory_ids: input.sourceTraceIds,
        support: input.support,
        gain: input.gain,
        raw_gain: input.rawGain,
        policy_confidence: currentPolicy?.confidence,
        status: input.status,
        source_episode_ids: input.sourceEpisodeIds,
        source_trace_ids: input.sourceTraceIds,
        policy: nextPolicy
      }
    },
    updatedAt: input.updatedAt
  };
}

function renderFailureClusterBody(input: {
  title: string;
  trigger: string;
  antiPattern: string;
  procedure: string;
  verification: string;
  boundary: string;
  support: number;
  confidence: number;
}): string {
  return [
    input.title,
    `Trigger: ${input.trigger}`,
    `Avoid: ${input.antiPattern}`,
    `Safer behavior: ${input.procedure}`,
    `Verification: ${input.verification}`,
    `Boundary: ${input.boundary}`,
    `Support: ${input.support}`,
    "Gain: 0",
    "Raw gain: 0",
    `Confidence: ${input.confidence}`
  ].join("\n");
}

function isActionableFailureSink(sink: {
  trigger: string;
  prefer: string[];
  avoid: string[];
  confidence: number;
}): boolean {
  if (!sink.trigger.trim() || sink.prefer.length === 0 || sink.avoid.length === 0) return false;
  if (sink.confidence < 0.6) return false;
  const normalize = (value: string) => value.toLowerCase().replace(/\s+/g, " ").trim();
  if (normalize(sink.avoid.join("\n")) === normalize(sink.prefer.join("\n"))) return false;
  return true;
}

function renderPolicyBody(draft: Pick<PolicyDraft, "title" | "trigger" | "procedure" | "verification" | "boundary" | "support" | "gain" | "rawGain" | "confidence" | "sourceTraceIds">): string {
  return [
    draft.title,
    `Trigger: ${draft.trigger}`,
    `Procedure: ${draft.procedure}`,
    `Verification: ${draft.verification}`,
    `Boundary: ${draft.boundary}`,
    `Support: ${draft.support}`,
    `Gain: ${roundNumber(draft.gain)}`,
    `Raw gain: ${roundNumber(draft.rawGain)}`,
    `Confidence: ${roundNumber(draft.confidence)}`,
    `Evidence: ${draft.sourceTraceIds.join(", ")}`
  ].join("\n");
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function l2InductionInvalidReason(result: unknown): string | null {
  if (!isRecord(result)) return "llm-failed: l2.induction.invalid: non-object output";
  if (!firstString(result.title)) return "llm-failed: l2.induction.invalid: missing title";
  if (!firstString(result.trigger)) return "llm-failed: l2.induction.invalid: missing trigger";
  if (!firstString(result.procedure, result.action)) {
    return "llm-failed: l2.induction.invalid: missing procedure";
  }
  if (
    !firstString(result.boundary) &&
    stringArray(result.exclusions).length === 0 &&
    stringArray(result.caveats).length === 0
  ) {
    return "llm-failed: l2.induction.invalid: missing exclusions";
  }
  return null;
}

function inferPolicyFreshness(text: string): "stable" | "dynamic" {
  return /(?:当前|实时|最新|业务数据|数据源|市场|库存|价格|策略变化|产品行为|外部\s*API)|\b(?:current|latest|live|real[- ]time|business data|data source|market|inventory|price|external api|product behavior)\b/i
    .test(text)
    ? "dynamic"
    : "stable";
}

function uniq<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}

function stringField(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && Boolean(item.trim()))
    : [];
}


function skillText(value: unknown): string {
  return stripDangerousMarkdownLinks(stripUnsafeHtml(skillRawString(value)))
    .replace(SKILL_CONTROL_RE, "")
    .trim();
}

function skillMarkdown(value: unknown): string {
  return stripDangerousMarkdownLinks(stripDangerousHtmlBlocks(skillRawString(value)))
    .replace(SKILL_CONTROL_RE, "")
    .trim();
}

function skillRawString(value: unknown): string {
  return value == null ? "" : String(value);
}

const SKILL_HTML_BLOCK_RE = /<\s*(script|style|iframe|object|embed|svg|math|template)\b[^>]*>[\s\S]*?<\s*\/\s*\1\s*>/gi;
const SKILL_DANGEROUS_TAG_RE = /<\/?\s*(script|style|iframe|object|embed|svg|math|template)\b[^>]*>/gi;
const SKILL_HTML_TAG_RE = /<\/?[a-z][a-z0-9:-]*(?:\s+[^<>]*)?>/gi;
const SKILL_CONTROL_RE = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g;
const SKILL_MARKDOWN_LINK_RE = /(!?)\[([^\]\n]*)\]\(((?:\\.|[^()\n]|\([^()\n]*\))+\))/g;

function stripUnsafeHtml(text: string): string {
  return text
    .replace(SKILL_HTML_BLOCK_RE, "")
    .replace(SKILL_HTML_TAG_RE, "");
}

function stripDangerousHtmlBlocks(text: string): string {
  return text.replace(SKILL_HTML_BLOCK_RE, "").replace(SKILL_DANGEROUS_TAG_RE, "");
}

function stripDangerousMarkdownLinks(text: string): string {
  return text.replace(
    SKILL_MARKDOWN_LINK_RE,
    (_match, bang: string, label: string, rawUrl: string) => {
      const url = rawUrl.trim();
      const firstToken = url.split(/\s+/)[0] ?? "";
      if (!isSafeLinkTarget(firstToken)) return `${bang}${label}`;
      return `${bang}[${label}](${url})`;
    }
  );
}

function isSafeLinkTarget(raw: string): boolean {
  const target = raw.trim().replace(/^['"<]+|[>"']+$/g, "");
  if (!target) return false;
  if (target.startsWith("#") || target.startsWith("/") || target.startsWith("./") || target.startsWith("../")) {
    return true;
  }
  try {
    const url = new URL(target);
    return url.protocol === "http:" || url.protocol === "https:" || url.protocol === "mailto:";
  } catch {
    return false;
  }
}

function roundNumber(value: number, digits = 4): number {
  const base = Math.pow(10, digits);
  return Math.round(value * base) / base;
}

function memoryStatusForLifecycleStatus(status: PolicyLifecycleStatus): "activated" | "resolving" | "archived" {
  if (status !== "candidate" && status !== "active") return "archived";
  return status === "candidate" ? "resolving" : "activated";
}

function errorMessageFromUnknown(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (value instanceof Error) return value.message;
  if (typeof value === "string") return value;
  if (isRecord(value)) {
    const message = value.error ?? value.message;
    if (typeof message === "string") return message;
  }
  return undefined;
}
