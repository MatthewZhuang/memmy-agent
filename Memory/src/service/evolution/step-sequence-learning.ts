import type { MemmyConfig } from "../../config/index.js";
import type { Embedder, LlmClient, LlmMessage } from "../../model/types.js";
import type {
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type {
  EpisodeProceduralPathRecord
} from "../../storage/procedural-path-repository.js";
import type {
  EpisodeStepPolicyProjectionRecord,
  StepPolicySkillPatternOccurrenceRecord,
  StepPolicySkillPatternRecord,
  StepSequencePatternOccurrenceRecord,
  StepSequencePatternRecord,
  StepSequencePolicyVersionRecord
} from "../../storage/step-sequence-learning-repository.js";
import type { MemoryRow } from "../../types.js";
import { stableHash, stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { clip } from "../../utils/text.js";
import { skillBetaPosterior, skillSuccessRate } from "../read-model/skill.js";
import {
  EPISODE_STEP_POLICY_PROJECTION_ALGORITHM_VERSION,
  STEP_CLUSTER_SIMILARITY_THRESHOLD,
  STEP_POLICY_SKILL_COMPILER_VERSION,
  STEP_POLICY_SKILL_MAX_LENGTH,
  STEP_POLICY_SKILL_MIN_LENGTH,
  STEP_POLICY_SKILL_SUPPORT_THRESHOLD,
  STEP_SEQUENCE_MAX_LENGTH,
  STEP_SEQUENCE_MIN_LENGTH,
  STEP_SEQUENCE_POLICY_INDUCTION_VERSION,
  STEP_SEQUENCE_SUPPORT_THRESHOLD,
  buildEpisodeStepPolicyProjection,
  buildStepOccurrence,
  buildStepSequencePolicy,
  contiguousWindows,
  hasMultipleDistinctValues,
  policySkillSequenceIdentity,
  selectLongestNonOverlapping,
  stepSequenceIdentity,
  type EpisodeStepPolicyProjectionNodeV1,
  type StepPolicySequenceSkillDraftV1,
  type StepSequencePolicyV1
} from "./step-sequence-learning-model.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";

export const STEP_SEQUENCE_LEARNING_OPERATION = "procedural.step_sequence_learning.v1" as const;
export const STEP_SEQUENCE_POLICY_OPERATION = "procedural.step_sequence_policy.induction.v1" as const;
export const STEP_POLICY_SKILL_OPERATION = "procedural.step_policy_skill.compilation.v1" as const;

const MAX_POLICY_PATTERNS_PER_INGEST = 6;
const MAX_EVIDENCE_OCCURRENCES = 6;
const MAX_POLICY_REPAIR_ATTEMPTS = 1;
const MAX_SKILL_REPAIR_ATTEMPTS = 1;

export const STEP_SEQUENCE_POLICY_PROMPT = `You induce one reusable atomic Policy from repeated contiguous sequences of normalized execution Steps.

Each evidence occurrence contains the same ordered StepCluster backbone and concrete Step intent/result pairs from a different Episode. The StepCluster IDs are internal symbols; do not expose them as the procedure.

Rules:
- Generalize only behavior supported by at least two distinct Episodes.
- Preserve the observed order and method-defining actions.
- A Policy is one reusable local procedure, not an Episode-wide Skill.
- Every procedure and verification item must cite evidence_refs copied exactly from occurrence_id, step_occurrence_id, or step_id values in the payload.
- evidence_occurrence_ids must cite at least two occurrences from distinct Episodes.
- Do not invent tools, checks, outcomes, or applicability conditions.
- Return JSON only with exactly these keys:
{
  "title": "...",
  "goal_pattern": "...",
  "trigger_conditions": ["..."],
  "procedure_steps": [{"instruction": "...", "evidence_refs": ["..."]}],
  "verification_steps": [{"check": "...", "success_signal": "...", "evidence_refs": ["..."]}],
  "do_not_apply_when": ["..."],
  "evidence_occurrence_ids": ["step_sequence_occurrence_..."],
  "confidence": 0.0
}`;

export const STEP_POLICY_SKILL_PROMPT = `You compile one user-invokable Skill from a repeated ordered sequence of distinct Policies.

The payload includes the stable Policy backbone and concrete Step evidence from multiple Episodes. Unmapped work between Policies is included as evidence context but is not automatically mandatory.

Rules:
- Preserve the ordered Policy backbone while expressing an operational end-to-end procedure.
- Generalize only behavior supported by at least two distinct Episodes.
- Do not restate Policy titles without operational details.
- Every Skill step and verification item must cite evidence_refs copied exactly from the payload.
- evidence_occurrence_ids must cite at least two sequence occurrences from distinct Episodes.
- tools must be copied from observed_tools.
- name must be lowercase snake_case, no longer than 48 characters.
- Return JSON only with exactly these keys:
{
  "name": "snake_case_name",
  "display_title": "...",
  "retrieval_blurb": "...",
  "trigger_context": "...",
  "summary": "...",
  "steps": [{"title": "...", "body": "...", "evidence_refs": ["..."]}],
  "verification": [{"check": "...", "success_signal": "...", "evidence_refs": ["..."]}],
  "do_not_use_when": ["..."],
  "tools": ["..."],
  "tags": ["..."],
  "evidence_occurrence_ids": ["step_policy_skill_occurrence_..."],
  "confidence": 0.0
}`;

export interface StepSequenceLearningDeps {
  repos: Repositories;
  config: MemmyConfig;
  embedder: Embedder;
  skillLlm: LlmClient;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  upsertEvolutionMemory(memory: MemoryRow): {
    memory: MemoryRow;
    created: boolean;
    previous?: MemoryRow;
  };
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

export interface StepSequenceLearningResult {
  episodeId: string;
  pathId: string;
  stepCount: number;
  clusterCount: number;
  repeatedPatternCount: number;
  inducedPolicyIds: string[];
  projectionIds: string[];
  readySkillPatternCount: number;
  compiledSkillIds: string[];
}

interface PolicyDraftResult extends Record<string, unknown> {
  title?: unknown;
  goal_pattern?: unknown;
  trigger_conditions?: unknown;
  procedure_steps?: unknown;
  verification_steps?: unknown;
  do_not_apply_when?: unknown;
  evidence_occurrence_ids?: unknown;
  confidence?: unknown;
}

interface SkillDraftResult extends Record<string, unknown> {
  name?: unknown;
  display_title?: unknown;
  retrieval_blurb?: unknown;
  trigger_context?: unknown;
  summary?: unknown;
  steps?: unknown;
  verification?: unknown;
  do_not_use_when?: unknown;
  tools?: unknown;
  tags?: unknown;
  evidence_occurrence_ids?: unknown;
  confidence?: unknown;
}

export class StepSequenceLearningPipeline {
  private readonly policyInFlight = new Map<
    string,
    Promise<StepSequencePolicyVersionRecord | undefined>
  >();
  private readonly skillInFlight = new Map<string, Promise<MemoryRow | undefined>>();

  constructor(private readonly deps: StepSequenceLearningDeps) {}

  async learnJob(job: EvolutionJobRecord): Promise<StepSequenceLearningResult | undefined> {
    if (!job.episodeId) throw new Error(`Step sequence learning job missing episodeId: ${job.id}`);
    const pathId = text(job.payload.pathId);
    const pathHash = text(job.payload.pathHash);
    const path = pathId
      ? this.deps.repos.proceduralPaths.get(pathId)
      : this.deps.repos.proceduralPaths.getActiveForEpisode(job.episodeId);
    if (!path || path.status !== "active" || path.episodeId !== job.episodeId) return undefined;
    if (pathHash && path.pathHash !== pathHash) return undefined;
    return this.learnPath(path, job.updatedAt);
  }

  async learnPath(
    path: EpisodeProceduralPathRecord,
    at: string
  ): Promise<StepSequenceLearningResult> {
    const episode = this.deps.repos.runtime.getEpisode(path.episodeId);
    if (!episode || episode.status !== "closed") {
      throw new Error(`Step sequence learning requires a closed Episode: ${path.episodeId}`);
    }
    const session = this.deps.repos.runtime.getSession(path.sessionId);
    if (!session || session.userId !== path.userId) {
      throw new Error(`Step sequence learning source Session is missing: ${path.sessionId}`);
    }
    const occurrences = path.path.steps.map((step) => buildStepOccurrence({
      pathId: path.id,
      pathHash: path.pathHash,
      episodeId: path.episodeId,
      userId: path.userId,
      sessionId: path.sessionId,
      namespaceId: path.namespaceId,
      reconstructionAlgorithmVersion: path.reconstructionAlgorithmVersion,
      step,
      createdAt: at
    }));
    const savedSteps = this.deps.repos.stepSequenceLearning.savePathSteps({
      pathId: path.id,
      occurrences
    });
    const embeddings = await this.ensureEmbeddings(savedSteps, at);
    for (const occurrence of savedSteps) {
      this.deps.repos.stepSequenceLearning.assignStepToCluster({
        occurrence,
        embedding: embeddings.get(occurrence.id)!,
        similarityThreshold: STEP_CLUSTER_SIMILARITY_THRESHOLD,
        at
      });
    }
    const clustered = this.deps.repos.stepSequenceLearning.listClusteredStepsForPath(path.id);
    if (clustered.length !== savedSteps.length) {
      throw new Error(`Step clustering coverage is incomplete: ${path.id}`);
    }
    const windows = contiguousWindows(
      clustered,
      STEP_SEQUENCE_MIN_LENGTH,
      STEP_SEQUENCE_MAX_LENGTH
    ).map((window) => {
      const clusterIds = window.values.map((item) => item.member.clusterId);
      const identity = stepSequenceIdentity(path.namespaceId, clusterIds);
      return {
        patternId: identity.id,
        sequenceHash: identity.sequenceHash,
        clusterIds,
        startStepIndex: window.values[0]!.occurrence.stepIndex,
        endStepIndex: window.values.at(-1)!.occurrence.stepIndex,
        stepOccurrenceIds: window.values.map((item) => item.occurrence.id)
      };
    });
    const activePolicyByPatternBefore = new Map(
      this.deps.repos.stepSequenceLearning.listStepPatterns(path.namespaceId)
        .map((pattern) => [pattern.id, pattern.activePolicyVersionId] as const)
    );
    const patterns = this.deps.repos.stepSequenceLearning.replaceStepSequenceOccurrences({
      namespaceId: path.namespaceId,
      episodeId: path.episodeId,
      pathId: path.id,
      sessionId: path.sessionId,
      ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask }),
      windows,
      at
    });
    const readyPatterns = patterns
      .filter((pattern) => pattern.lifecycleStatus === "ready" &&
        pattern.selectedEpisodeCount >= STEP_SEQUENCE_SUPPORT_THRESHOLD &&
        !pattern.supersededByPatternId)
      .sort((left, right) => right.selectedEpisodeCount - left.selectedEpisodeCount ||
        right.clusterIds.length - left.clusterIds.length || left.id.localeCompare(right.id))
      .slice(0, MAX_POLICY_PATTERNS_PER_INGEST);
    const inducedPolicies: StepSequencePolicyVersionRecord[] = [];
    const affectedEpisodeIds = new Set<string>([path.episodeId]);
    for (const pattern of patterns) {
      const previousPolicyId = activePolicyByPatternBefore.get(pattern.id);
      if (previousPolicyId && previousPolicyId !== pattern.activePolicyVersionId) {
        for (const occurrence of this.deps.repos.stepSequenceLearning
          .listStepPatternOccurrences(pattern.id)) {
          affectedEpisodeIds.add(occurrence.episodeId);
        }
      }
    }
    for (const pattern of readyPatterns) {
      const policy = await this.ensurePolicy(pattern, at);
      for (const occurrence of this.deps.repos.stepSequenceLearning
        .listStepPatternOccurrences(pattern.id)) {
        affectedEpisodeIds.add(occurrence.episodeId);
      }
      if (policy) {
        inducedPolicies.push(policy);
        const retirement = this.deps.repos.stepSequenceLearning
          .retireStepPatternsCoveredBy(pattern.id, at);
        for (const episodeId of retirement.affectedEpisodeIds) {
          affectedEpisodeIds.add(episodeId);
        }
      }
    }
    const projections: EpisodeStepPolicyProjectionRecord[] = [];
    for (const episodeId of [...affectedEpisodeIds].sort()) {
      const projection = this.projectEpisode(episodeId, at);
      if (projection) projections.push(projection);
    }
    const skillPatterns = this.deps.repos.stepSequenceLearning.listSkillPatterns(path.namespaceId);
    const readySkillPatterns = skillPatterns.filter((pattern) =>
      pattern.lifecycleStatus === "ready" &&
      pattern.selectedEpisodeCount >= STEP_POLICY_SKILL_SUPPORT_THRESHOLD &&
      !pattern.supersededByPatternId &&
      hasMultipleDistinctValues(pattern.policyKeys));
    const compiledSkills: MemoryRow[] = [];
    for (const pattern of readySkillPatterns) {
      const skill = await this.ensureSkill(pattern, at);
      if (skill) {
        compiledSkills.push(skill);
        this.deps.repos.stepSequenceLearning.retireSkillPatternsCoveredBy(pattern.id, at);
      }
    }
    return {
      episodeId: path.episodeId,
      pathId: path.id,
      stepCount: savedSteps.length,
      clusterCount: new Set(clustered.map((item) => item.member.clusterId)).size,
      repeatedPatternCount: patterns.filter((pattern) =>
        pattern.selectedEpisodeCount >= STEP_SEQUENCE_SUPPORT_THRESHOLD).length,
      inducedPolicyIds: unique(inducedPolicies.map((policy) => policy.id)),
      projectionIds: unique(projections.map((projection) => projection.id)),
      readySkillPatternCount: readySkillPatterns.length,
      compiledSkillIds: unique(compiledSkills.map((skill) => skill.id))
    };
  }

  private async ensureEmbeddings(
    occurrences: readonly ReturnType<typeof buildStepOccurrence>[],
    at: string
  ) {
    const provider = this.deps.embedder.config.provider.trim() || "unknown";
    const model = this.deps.embedder.config.model?.trim() || "default";
    const byOccurrence = new Map<string, ReturnType<
      Repositories["stepSequenceLearning"]["saveEmbedding"]
    >>();
    const missing = occurrences.filter((occurrence) => {
      const existing = this.deps.repos.stepSequenceLearning.getEmbedding(occurrence.id, {
        provider,
        model,
        semanticHash: occurrence.semanticHash
      });
      if (existing) byOccurrence.set(occurrence.id, existing);
      return !existing;
    });
    if (missing.length > 0) {
      const vectors = await this.deps.embedder.embed(
        missing.map((occurrence) => occurrence.semanticText),
        "document"
      );
      if (vectors.length !== missing.length) {
        throw new Error(`Step embedding returned ${vectors.length} vectors for ${missing.length} Steps`);
      }
      for (const [index, occurrence] of missing.entries()) {
        const saved = this.deps.repos.stepSequenceLearning.saveEmbedding({
          occurrence,
          embeddingProvider: provider,
          embeddingModel: model,
          vector: vectors[index]!,
          at
        });
        byOccurrence.set(occurrence.id, saved);
      }
    }
    return byOccurrence;
  }

  private async ensurePolicy(
    pattern: StepSequencePatternRecord,
    at: string
  ): Promise<StepSequencePolicyVersionRecord | undefined> {
    const current = this.currentPromotablePolicyPattern(pattern);
    if (!current) return undefined;
    const existing = this.deps.repos.stepSequenceLearning.getPolicyForPatternMembership(
      current.id,
      current.membershipVersion
    );
    if (existing) {
      return existing.status === "active"
        ? existing
        : this.deps.repos.stepSequenceLearning.activatePolicyVersion(existing.id, at);
    }
    const inFlightKey = `${current.id}:${current.membershipVersion}`;
    const inFlight = this.policyInFlight.get(inFlightKey);
    if (inFlight) return inFlight;
    const promise = this.inducePolicy(current, at).finally(() => {
      this.policyInFlight.delete(inFlightKey);
    });
    this.policyInFlight.set(inFlightKey, promise);
    return promise;
  }

  private async inducePolicy(
    pattern: StepSequencePatternRecord,
    at: string
  ): Promise<StepSequencePolicyVersionRecord | undefined> {
    if (!this.deps.skillLlm.isConfigured()) {
      throw new Error("Step sequence Policy induction requires a configured LLM");
    }
    const allOccurrences = this.deps.repos.stepSequenceLearning
      .listStepPatternOccurrences(pattern.id, { selectedOnly: true });
    const selected = selectDistinctEpisodeOccurrences(allOccurrences, MAX_EVIDENCE_OCCURRENCES);
    if (new Set(selected.map((occurrence) => occurrence.episodeId)).size < 2) {
      throw new Error(`Step sequence Policy requires two Episodes: ${pattern.id}`);
    }
    const context = this.policyEvidenceContext(pattern, selected);
    const messages: LlmMessage[] = [
      { role: "system", content: STEP_SEQUENCE_POLICY_PROMPT },
      { role: "user", content: stableStringify(context.payload) }
    ];
    const options = {
      operation: STEP_SEQUENCE_POLICY_OPERATION,
      thinkingMode: this.deps.config.evolution.enableThinking
        ? "enabled" as const
        : "disabled" as const,
      temperature: 0.2,
      maxTokens: 5_000,
      jsonMode: true
    };
    let result = await this.deps.skillLlm.completeJson<PolicyDraftResult>(messages, options);
    let draft: ReturnType<typeof parsePolicyDraft>;
    for (let repair = 0; ; repair += 1) {
      try {
        draft = parsePolicyDraft(result, context);
        break;
      } catch (error) {
        if (repair >= MAX_POLICY_REPAIR_ATTEMPTS) throw error;
        result = await this.deps.skillLlm.completeJson<PolicyDraftResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: repairInstruction(error) }
        ], { ...options, operation: `${STEP_SEQUENCE_POLICY_OPERATION}.repair` });
      }
    }
    const current = this.currentPromotablePolicyPattern(pattern);
    if (!current) return undefined;
    const policy = buildStepSequencePolicy({
      namespaceId: current.namespaceId,
      patternId: current.id,
      patternMembershipVersion: current.membershipVersion,
      clusterIds: current.clusterIds,
      draft,
      model: this.deps.skillLlm.config.model
    });
    const firstOccurrence = selected[0]!;
    const firstStep = this.deps.repos.stepSequenceLearning
      .getStep(firstOccurrence.stepOccurrenceIds[0]!)!;
    const sourceTraceIds = this.traceMemoryIdsForStepOccurrences(
      selected.flatMap((occurrence) => occurrence.stepOccurrenceIds)
    );
    const signature = stepSequencePolicySignature(current.sequenceHash);
    const trigger = policy.triggerConditions.join("\n");
    const procedure = policy.procedureSteps.map((step) => step.instruction).join("\n");
    const verification = policy.verificationSteps.map((step) =>
      `${step.check}: ${step.successSignal}`).join("\n");
    const boundary = policy.doNotApplyWhen.join("\n");
    const expectedOutcome = policy.verificationSteps.map((step) => step.successSignal).join("\n");
    const support = policy.supportEpisodeIds.length;
    const memory = this.deps.buildMemory({
      userId: firstStep.userId,
      layer: "L2",
      kind: "policy",
      lifecycleStatus: "candidate",
      memoryType: "LongTermMemory",
      key: policy.policyKey,
      value: renderPolicy(policy),
      tags: ["policy", "procedural", "step-sequence", "v2", "shadow"],
      info: {
        signature,
        title: policy.title,
        support,
        gain: 0,
        raw_gain: 0,
        policy_confidence: policy.confidence,
        freshness_class: "stable",
        last_verified_at: at,
        status: "candidate",
        source_memory_ids: sourceTraceIds,
        step_sequence_policy_version_id: policy.id,
        source_pattern_id: current.id,
        source_episode_ids: policy.supportEpisodeIds
      },
      internal: {
        source: "worker.step_sequence_policy_induction.v1",
        plugin_algorithm: STEP_SEQUENCE_POLICY_INDUCTION_VERSION,
        source_memory_ids: sourceTraceIds,
        source_l1_memory_ids: sourceTraceIds,
        title: policy.title,
        trigger,
        procedure,
        verification,
        boundary,
        expected_outcome: expectedOutcome,
        exclusions: policy.doNotApplyWhen,
        support,
        gain: 0,
        raw_gain: 0,
        policy_confidence: policy.confidence,
        freshness_class: "stable",
        last_verified_at: at,
        status: "candidate",
        source_episode_ids: policy.supportEpisodeIds,
        source_trace_ids: sourceTraceIds,
        step_sequence_policy: {
          policy_version_id: policy.id,
          pattern_id: current.id,
          pattern_membership_version: current.membershipVersion,
          sequence_hash: current.sequenceHash,
          cluster_ids: current.clusterIds,
          evidence_hash: policy.provenance.evidenceHash
        },
        policy: {
          title: policy.title,
          goal_pattern: policy.goalPattern,
          trigger,
          procedure,
          verification,
          boundary,
          expected_outcome: expectedOutcome,
          exclusions: policy.doNotApplyWhen,
          support,
          gain: 0,
          raw_gain: 0,
          policy_confidence: policy.confidence,
          freshness_class: "stable",
          last_verified_at: at,
          status: "candidate",
          experience_type: "success_pattern",
          evidence_polarity: "positive",
          skill_eligible: false,
          signature,
          induction_version: STEP_SEQUENCE_POLICY_INDUCTION_VERSION,
          source_episode_ids: policy.supportEpisodeIds,
          source_trace_ids: sourceTraceIds,
          source_occurrence_ids: policy.evidenceOccurrenceIds,
          decision_guidance: {
            preference: [],
            anti_pattern: []
          },
          vec: null
        }
      },
      createdAt: at
    });
    let savedMemory: MemoryRow;
    let savedPolicy: StepSequencePolicyVersionRecord;
    this.deps.repos.transaction(() => {
      savedMemory = this.deps.upsertEvolutionMemory(memory).memory;
      savedPolicy = this.deps.repos.stepSequenceLearning.saveAndActivatePolicy({
        policy,
        l2MemoryId: savedMemory.id,
        at
      });
      for (const episodeId of policy.supportEpisodeIds) {
        this.deps.repos.runtime.appendEpisodeDerivedMemory(episodeId, "L2", savedMemory.id, at);
      }
      for (const traceId of sourceTraceIds) {
        this.deps.repos.runtime.insertTracePolicyLink({
          userId: firstStep.userId,
          l1MemoryId: traceId,
          l2MemoryId: savedMemory.id,
          relation: "supports",
          strength: policy.confidence,
          createdAt: at
        });
      }
    });
    this.scheduleEmbedding(savedMemory!, firstStep.sessionId, firstStep.episodeId, at);
    return savedPolicy!;
  }

  private currentPromotablePolicyPattern(
    snapshot: StepSequencePatternRecord
  ): StepSequencePatternRecord | undefined {
    const current = this.deps.repos.stepSequenceLearning.getStepPattern(snapshot.id);
    return current && current.membershipVersion === snapshot.membershipVersion &&
      current.lifecycleStatus === "ready" &&
      current.selectedEpisodeCount >= STEP_SEQUENCE_SUPPORT_THRESHOLD &&
      !current.supersededByPatternId
      ? current
      : undefined;
  }

  private policyEvidenceContext(
    pattern: StepSequencePatternRecord,
    occurrences: readonly StepSequencePatternOccurrenceRecord[]
  ): {
    payload: Record<string, unknown>;
    validRefs: Set<string>;
    occurrenceEpisodes: Map<string, string>;
  } {
    const validRefs = new Set<string>();
    const occurrenceEpisodes = new Map<string, string>();
    const evidence = occurrences.map((occurrence) => {
      validRefs.add(occurrence.id);
      occurrenceEpisodes.set(occurrence.id, occurrence.episodeId);
      const steps = occurrence.stepOccurrenceIds.map((id) => {
        const step = this.deps.repos.stepSequenceLearning.getStep(id);
        if (!step) throw new Error(`Step sequence evidence Step is missing: ${id}`);
        validRefs.add(step.id);
        validRefs.add(step.stepId);
        return {
          step_occurrence_id: step.id,
          step_id: step.stepId,
          step_index: step.stepIndex,
          intent: clip(step.intent, 600),
          summary: clip(step.summary, 900),
          outcome: step.outcome,
          tool_name: step.toolName
        };
      });
      return {
        occurrence_id: occurrence.id,
        episode_id: occurrence.episodeId,
        terminal_reward: occurrence.terminalReward,
        steps
      };
    });
    return {
      payload: {
        pattern: {
          pattern_id: pattern.id,
          sequence_length: pattern.clusterIds.length,
          distinct_episode_count: pattern.selectedEpisodeCount
        },
        evidence
      },
      validRefs,
      occurrenceEpisodes
    };
  }

  private projectEpisode(
    episodeId: string,
    at: string
  ): EpisodeStepPolicyProjectionRecord | undefined {
    const path = this.deps.repos.proceduralPaths.getActiveForEpisode(episodeId);
    if (!path) return undefined;
    const episode = this.deps.repos.runtime.getEpisode(episodeId);
    if (!episode) return undefined;
    const steps = this.deps.repos.stepSequenceLearning.listStepsForPath(path.id);
    if (steps.length === 0) return undefined;
    const candidates: Array<{
      pattern: StepSequencePatternRecord;
      policy: StepSequencePolicyVersionRecord;
      occurrence: StepSequencePatternOccurrenceRecord;
    }> = [];
    for (const pattern of this.deps.repos.stepSequenceLearning.listStepPatterns(path.namespaceId)) {
      if (!pattern.activePolicyVersionId) continue;
      const policy = this.deps.repos.stepSequenceLearning.getPolicy(pattern.activePolicyVersionId);
      if (!policy || policy.status !== "active") continue;
      for (const occurrence of this.deps.repos.stepSequenceLearning
        .listStepPatternOccurrences(pattern.id)) {
        if (occurrence.episodeId === episodeId && occurrence.pathId === path.id) {
          candidates.push({ pattern, policy, occurrence });
        }
      }
    }
    const selected = selectLongestNonOverlapping(candidates.map((candidate) => ({
      ...candidate,
      id: candidate.occurrence.id,
      startIndex: candidate.occurrence.startStepIndex,
      endIndex: candidate.occurrence.endStepIndex,
      sequenceLength: candidate.pattern.clusterIds.length,
      support: candidate.pattern.selectedEpisodeCount
    })));
    const nodes: Array<Omit<EpisodeStepPolicyProjectionNodeV1, "nodeIndex">> = [];
    let cursor = 0;
    for (const item of selected.sort((left, right) =>
      left.occurrence.startStepIndex - right.occurrence.startStepIndex)) {
      if (cursor < item.occurrence.startStepIndex) {
        nodes.push(unmappedNode(steps.slice(cursor, item.occurrence.startStepIndex)));
      }
      const covered = steps.slice(
        item.occurrence.startStepIndex,
        item.occurrence.endStepIndex + 1
      );
      nodes.push({
        kind: "policy",
        startStepIndex: item.occurrence.startStepIndex,
        endStepIndex: item.occurrence.endStepIndex,
        stepOccurrenceIds: covered.map((step) => step.id),
        stepIds: covered.map((step) => step.stepId),
        preStateId: covered[0]!.preStateId,
        postStateId: covered.at(-1)!.postStateId,
        policyVersionId: item.policy.id,
        policyKey: item.policy.policyKey,
        patternId: item.pattern.id,
        patternOccurrenceId: item.occurrence.id,
        supportEpisodeCount: item.pattern.selectedEpisodeCount
      });
      cursor = item.occurrence.endStepIndex + 1;
    }
    if (cursor < steps.length) nodes.push(unmappedNode(steps.slice(cursor)));
    const projection = buildEpisodeStepPolicyProjection({
      episodeId,
      pathId: path.id,
      pathHash: path.pathHash,
      nodes,
      totalStepCount: steps.length
    });
    const saved = this.deps.repos.stepSequenceLearning.saveAndActivateProjection({
      projection,
      userId: path.userId,
      sessionId: path.sessionId,
      namespaceId: path.namespaceId,
      at
    });
    this.mineSkillPatterns(saved, episode.rTask, at);
    return saved;
  }

  private mineSkillPatterns(
    projection: EpisodeStepPolicyProjectionRecord,
    terminalReward: number | undefined,
    at: string
  ): void {
    const policyNodes = projection.projection.nodes.filter((node) => node.kind === "policy");
    const windows = contiguousWindows(
      policyNodes,
      STEP_POLICY_SKILL_MIN_LENGTH,
      STEP_POLICY_SKILL_MAX_LENGTH
    ).flatMap((window) => {
      const policyKeys = window.values.map((node) => node.policyKey!);
      if (!hasMultipleDistinctValues(policyKeys)) return [];
      const identity = policySkillSequenceIdentity(projection.namespaceId, policyKeys);
      return [{
        patternId: identity.id,
        sequenceHash: identity.sequenceHash,
        policyKeys,
        policyVersionIds: window.values.map((node) => node.policyVersionId!),
        startNodeIndex: window.values[0]!.nodeIndex,
        endNodeIndex: window.values.at(-1)!.nodeIndex,
        stepOccurrenceIds: projection.projection.nodes
          .filter((node) => node.nodeIndex >= window.values[0]!.nodeIndex &&
            node.nodeIndex <= window.values.at(-1)!.nodeIndex)
          .flatMap((node) => node.stepOccurrenceIds)
      }];
    });
    this.deps.repos.stepSequenceLearning.replaceSkillPatternOccurrences({
      namespaceId: projection.namespaceId,
      projectionId: projection.id,
      episodeId: projection.episodeId,
      pathId: projection.pathId,
      sessionId: projection.sessionId,
      ...(terminalReward === undefined ? {} : { terminalReward }),
      windows,
      at
    });
  }

  private async ensureSkill(
    pattern: StepPolicySkillPatternRecord,
    at: string
  ): Promise<MemoryRow | undefined> {
    const current = this.currentPromotableSkillPattern(pattern);
    if (!current) return undefined;
    if (current.activeSkillMemoryId) {
      const existing = this.deps.repos.memories.get(current.activeSkillMemoryId);
      if (existing && existing.status !== "archived" && existing.status !== "deleted") return existing;
    }
    const inFlightKey = `${current.id}:${current.membershipVersion}`;
    const inFlight = this.skillInFlight.get(inFlightKey);
    if (inFlight) return inFlight;
    const promise = this.compileSkill(current, at).finally(() => {
      this.skillInFlight.delete(inFlightKey);
    });
    this.skillInFlight.set(inFlightKey, promise);
    return promise;
  }

  private async compileSkill(
    pattern: StepPolicySkillPatternRecord,
    at: string
  ): Promise<MemoryRow | undefined> {
    if (!this.deps.skillLlm.isConfigured()) {
      throw new Error("Step Policy sequence Skill compilation requires a configured LLM");
    }
    const occurrences = selectDistinctEpisodeOccurrences(
      this.deps.repos.stepSequenceLearning.listSkillPatternOccurrences(
        pattern.id,
        { selectedOnly: true }
      ),
      MAX_EVIDENCE_OCCURRENCES
    );
    if (new Set(occurrences.map((occurrence) => occurrence.episodeId)).size < 2) {
      throw new Error(`Step Policy Skill requires two Episodes: ${pattern.id}`);
    }
    const context = this.skillEvidenceContext(pattern, occurrences);
    const messages: LlmMessage[] = [
      { role: "system", content: STEP_POLICY_SKILL_PROMPT },
      { role: "user", content: stableStringify(context.payload) }
    ];
    const options = {
      operation: STEP_POLICY_SKILL_OPERATION,
      thinkingMode: this.deps.config.evolution.enableThinking
        ? "enabled" as const
        : "disabled" as const,
      temperature: 0.2,
      maxTokens: 7_000,
      jsonMode: true
    };
    let result = await this.deps.skillLlm.completeJson<SkillDraftResult>(messages, options);
    let draft: StepPolicySequenceSkillDraftV1;
    for (let repair = 0; ; repair += 1) {
      try {
        draft = parseSkillDraft(result, context);
        break;
      } catch (error) {
        if (repair >= MAX_SKILL_REPAIR_ATTEMPTS) throw error;
        result = await this.deps.skillLlm.completeJson<SkillDraftResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: repairInstruction(error) }
        ], { ...options, operation: `${STEP_POLICY_SKILL_OPERATION}.repair` });
      }
    }
    const current = this.currentPromotableSkillPattern(pattern);
    if (!current) return undefined;
    const firstStep = this.deps.repos.stepSequenceLearning
      .getStep(occurrences[0]!.stepOccurrenceIds[0]!)!;
    const policyVersions = unique(occurrences.flatMap((occurrence) =>
      occurrence.policyVersionIds));
    const policyMemories = policyVersions.flatMap((id) => {
      const policy = this.deps.repos.stepSequenceLearning.getPolicy(id);
      return policy?.l2MemoryId ? [policy.l2MemoryId] : [];
    });
    const sourceStepOccurrenceIds = unique(occurrences.flatMap((occurrence) =>
      occurrence.stepOccurrenceIds));
    const evidenceAnchorIds = this.traceMemoryIdsForStepOccurrences(sourceStepOccurrenceIds);
    const support = current.selectedEpisodeCount;
    const successRate = skillSuccessRate(0, 0);
    const betaPosterior = skillBetaPosterior(0, 0);
    const procedureJson = canonicalStepPolicySkillProcedure({
      draft,
      support,
      successRate,
      betaPosterior,
      patternId: current.id,
      evidenceOccurrenceIds: draft.evidenceOccurrenceIds,
      sourcePolicyIds: policyMemories
    });
    const skillKey = `skill:step-policy-sequence:${current.sequenceHash}`;
    const value = renderSkill(draft);
    const memory = this.deps.buildMemory({
      userId: firstStep.userId,
      layer: "Skill",
      kind: "skill",
      lifecycleStatus: "candidate",
      memoryType: "SkillMemory",
      key: skillKey,
      value,
      tags: ["skill", "procedural", "step-policy-sequence", "v2", "shadow", ...draft.tags],
      info: {
        name: draft.name,
        title: draft.displayTitle,
        eta: draft.confidence,
        status: "candidate",
        support,
        source_memory_ids: policyMemories,
        source_pattern_id: current.id,
        source_episode_ids: unique(occurrences.map((occurrence) => occurrence.episodeId))
      },
      internal: {
        source: "worker.step_policy_sequence_skill_compilation.v1",
        plugin_algorithm: STEP_POLICY_SKILL_COMPILER_VERSION,
        source_memory_ids: policyMemories,
        source_policy_ids: policyMemories,
        source_step_sequence_policy_version_ids: policyVersions,
        source_step_occurrence_ids: sourceStepOccurrenceIds,
        source_step_policy_skill_occurrence_ids: draft.evidenceOccurrenceIds,
        source_path_ids: unique(occurrences.map((occurrence) => occurrence.pathId)),
        source_episode_ids: unique(occurrences.map((occurrence) => occurrence.episodeId)),
        evidence_anchor_ids: evidenceAnchorIds,
        name: draft.name,
        invocation_guide: value,
        procedure_json: procedureJson,
        step_policy_sequence_skill: {
          compiler_version: STEP_POLICY_SKILL_COMPILER_VERSION,
          pattern_id: current.id,
          pattern_membership_version: current.membershipVersion,
          sequence_hash: current.sequenceHash,
          policy_keys: current.policyKeys,
          executable: true
        },
        skill: {
          name: draft.name,
          eta: draft.confidence,
          status: "candidate",
          support,
          gain: 0,
          source_policy_ids: policyMemories,
          source_world_model_ids: [],
          evidence_anchor_ids: evidenceAnchorIds,
          invocation_guide: value,
          procedure_json: procedureJson,
          trials_attempted: 0,
          trials_passed: 0,
          success_rate: successRate,
          beta_posterior: betaPosterior,
          vec: null
        }
      },
      createdAt: at
    });
    let saved: MemoryRow;
    this.deps.repos.transaction(() => {
      saved = this.deps.upsertEvolutionMemory(memory).memory;
      this.deps.repos.stepSequenceLearning.markSkillCompiled({
        patternId: current.id,
        membershipVersion: current.membershipVersion,
        skillMemoryId: saved!.id,
        at
      });
    });
    for (const episodeId of unique(occurrences.map((occurrence) => occurrence.episodeId))) {
      this.deps.repos.runtime.appendEpisodeDerivedMemory(episodeId, "Skill", saved!.id, at);
    }
    this.scheduleEmbedding(saved!, firstStep.sessionId, firstStep.episodeId, at);
    return saved!;
  }

  private currentPromotableSkillPattern(
    snapshot: StepPolicySkillPatternRecord
  ): StepPolicySkillPatternRecord | undefined {
    const current = this.deps.repos.stepSequenceLearning.getSkillPattern(snapshot.id);
    return current && current.membershipVersion === snapshot.membershipVersion &&
      current.lifecycleStatus === "ready" &&
      current.selectedEpisodeCount >= STEP_POLICY_SKILL_SUPPORT_THRESHOLD &&
      !current.supersededByPatternId &&
      hasMultipleDistinctValues(current.policyKeys)
      ? current
      : undefined;
  }

  private skillEvidenceContext(
    pattern: StepPolicySkillPatternRecord,
    occurrences: readonly StepPolicySkillPatternOccurrenceRecord[]
  ): {
    payload: Record<string, unknown>;
    validRefs: Set<string>;
    occurrenceEpisodes: Map<string, string>;
    observedTools: Set<string>;
  } {
    const validRefs = new Set<string>();
    const occurrenceEpisodes = new Map<string, string>();
    const observedTools = new Set<string>();
    const evidence = occurrences.map((occurrence) => {
      validRefs.add(occurrence.id);
      occurrenceEpisodes.set(occurrence.id, occurrence.episodeId);
      const policies = occurrence.policyVersionIds.map((id) => {
        const policy = this.deps.repos.stepSequenceLearning.getPolicy(id);
        if (!policy) throw new Error(`Step Policy Skill evidence Policy is missing: ${id}`);
        validRefs.add(policy.id);
        return {
          policy_version_id: policy.id,
          policy_key: policy.policyKey,
          title: policy.title,
          goal_pattern: policy.policy.goalPattern,
          procedure: policy.policy.procedureSteps.map((step) => step.instruction),
          verification: policy.policy.verificationSteps.map((step) => ({
            check: step.check,
            success_signal: step.successSignal
          }))
        };
      });
      const steps = occurrence.stepOccurrenceIds.map((id) => {
        const step = this.deps.repos.stepSequenceLearning.getStep(id);
        if (!step) throw new Error(`Step Policy Skill evidence Step is missing: ${id}`);
        validRefs.add(step.id);
        validRefs.add(step.stepId);
        if (step.toolName) observedTools.add(step.toolName);
        return {
          step_occurrence_id: step.id,
          step_id: step.stepId,
          step_index: step.stepIndex,
          intent: clip(step.intent, 600),
          summary: clip(step.summary, 900),
          outcome: step.outcome,
          tool_name: step.toolName
        };
      });
      return {
        sequence_occurrence_id: occurrence.id,
        episode_id: occurrence.episodeId,
        terminal_reward: occurrence.terminalReward,
        policies,
        full_step_path: steps
      };
    });
    return {
      payload: {
        pattern: {
          pattern_id: pattern.id,
          policy_keys: pattern.policyKeys,
          distinct_episode_count: pattern.selectedEpisodeCount
        },
        observed_tools: [...observedTools].sort(),
        evidence
      },
      validRefs,
      occurrenceEpisodes,
      observedTools
    };
  }

  private scheduleEmbedding(
    memory: MemoryRow,
    sessionId: string,
    episodeId: string,
    at: string
  ): void {
    if (!this.deps.config.algorithm.capture.embedAfterCapture) return;
    this.deps.enqueueJob({
      jobType: "embedding",
      userId: memory.userId,
      sessionId,
      episodeId,
      targetMemoryId: memory.id,
      payload: {
        reason: "step_sequence_learning.upserted",
        contentHash: memory.contentHash ?? undefined
      },
      createdAt: at
    });
  }

  private traceMemoryIdsForStepOccurrences(stepOccurrenceIds: readonly string[]): string[] {
    const rawTurnIdsByEpisode = new Map<string, Set<string>>();
    for (const occurrenceId of stepOccurrenceIds) {
      const occurrence = this.deps.repos.stepSequenceLearning.getStep(occurrenceId);
      if (!occurrence) continue;
      const rawTurnIds = rawTurnIdsByEpisode.get(occurrence.episodeId) ?? new Set<string>();
      rawTurnIds.add(occurrence.rawTurnId);
      rawTurnIdsByEpisode.set(occurrence.episodeId, rawTurnIds);
    }
    const traceIds: string[] = [];
    for (const [episodeId, rawTurnIds] of rawTurnIdsByEpisode) {
      const episode = this.deps.repos.runtime.getEpisode(episodeId);
      if (!episode) continue;
      for (const memory of this.deps.repos.memories.getMany(episode.l1MemoryIds)) {
        if (memory.memoryLayer !== "L1") continue;
        const rawTurnId = rawTurnIdFromMemory(memory);
        if (rawTurnId && rawTurnIds.has(rawTurnId)) traceIds.push(memory.id);
      }
    }
    return unique(traceIds);
  }
}

function stepSequencePolicySignature(sequenceHash: string): string {
  return `step_sequence|${sequenceHash.slice(0, 24)}|_|_`;
}

function canonicalStepPolicySkillProcedure(input: {
  draft: StepPolicySequenceSkillDraftV1;
  support: number;
  successRate: number;
  betaPosterior: ReturnType<typeof skillBetaPosterior>;
  patternId: string;
  evidenceOccurrenceIds: readonly string[];
  sourcePolicyIds: readonly string[];
}): Record<string, unknown> {
  return {
    retrievalBlurb: input.draft.retrievalBlurb,
    triggerContext: input.draft.triggerContext,
    displayTitle: input.draft.displayTitle,
    summary: input.draft.summary,
    parameters: [],
    preconditions: [],
    steps: input.draft.steps.map((step) => ({
      title: step.title,
      body: step.body,
      evidenceRefs: step.evidenceRefs,
      supportingPolicyIds: [...input.sourcePolicyIds]
    })),
    examples: [],
    verification: input.draft.verification.map((item) => ({
      check: item.check,
      successSignal: item.successSignal,
      evidenceRefs: item.evidenceRefs
    })),
    doNotUseWhen: [...input.draft.doNotUseWhen],
    decisionGuidance: {
      preference: [],
      antiPattern: []
    },
    reliability: {
      supportCount: input.support,
      successRate: input.successRate,
      betaPosterior: input.betaPosterior
    },
    tools: [...input.draft.tools],
    tags: [...input.draft.tags],
    grounding: {
      patternId: input.patternId,
      evidenceOccurrenceIds: [...input.evidenceOccurrenceIds],
      sourcePolicyMemoryIds: [...input.sourcePolicyIds]
    }
  };
}

function rawTurnIdFromMemory(memory: MemoryRow): string | undefined {
  const internal = memory.properties.internal_info;
  const sourceRawTurnId = internal.source_raw_turn_id;
  if (typeof sourceRawTurnId === "string" && sourceRawTurnId) return sourceRawTurnId;
  const rawTurnId = internal.raw_turn_id;
  if (typeof rawTurnId === "string" && rawTurnId) return rawTurnId;
  const trace = internal.trace;
  return isRecord(trace) && typeof trace.raw_turn_id === "string" && trace.raw_turn_id
    ? trace.raw_turn_id
    : undefined;
}

function parsePolicyDraft(
  result: PolicyDraftResult,
  context: {
    validRefs: ReadonlySet<string>;
    occurrenceEpisodes: ReadonlyMap<string, string>;
  }
): Omit<StepSequencePolicyV1,
  "id" | "schemaVersion" | "inductionVersion" | "policyKey" | "namespaceId" |
  "patternId" | "patternMembershipVersion" | "clusterIds" | "provenance" | "contentHash"> {
  assertExactKeys(result, [
    "title", "goal_pattern", "trigger_conditions", "procedure_steps",
    "verification_steps", "do_not_apply_when", "evidence_occurrence_ids", "confidence"
  ], "Step sequence Policy");
  const evidenceOccurrenceIds = stringArrayField(
    result.evidence_occurrence_ids,
    "evidence_occurrence_ids",
    2,
    MAX_EVIDENCE_OCCURRENCES
  );
  for (const id of evidenceOccurrenceIds) {
    if (!context.occurrenceEpisodes.has(id)) throw new Error(`Unknown Policy occurrence ref: ${id}`);
  }
  if (new Set(evidenceOccurrenceIds.map((id) => context.occurrenceEpisodes.get(id))).size < 2) {
    throw new Error("Policy evidence must span two Episodes");
  }
  return {
    title: requiredText(result.title, "title", 200),
    goalPattern: requiredText(result.goal_pattern, "goal_pattern", 1_000),
    triggerConditions: stringArrayField(result.trigger_conditions, "trigger_conditions", 1, 10),
    procedureSteps: groundedSteps(result.procedure_steps, context.validRefs),
    verificationSteps: groundedVerification(result.verification_steps, context.validRefs),
    doNotApplyWhen: stringArrayField(result.do_not_apply_when, "do_not_apply_when", 0, 10),
    evidenceOccurrenceIds: unique(evidenceOccurrenceIds),
    supportEpisodeIds: unique(evidenceOccurrenceIds.map((id) =>
      context.occurrenceEpisodes.get(id)!).filter(Boolean)),
    confidence: confidence(result.confidence)
  };
}

function parseSkillDraft(
  result: SkillDraftResult,
  context: {
    validRefs: ReadonlySet<string>;
    occurrenceEpisodes: ReadonlyMap<string, string>;
    observedTools: ReadonlySet<string>;
  }
): StepPolicySequenceSkillDraftV1 {
  assertExactKeys(result, [
    "name", "display_title", "retrieval_blurb", "trigger_context", "summary",
    "steps", "verification", "do_not_use_when", "tools", "tags",
    "evidence_occurrence_ids", "confidence"
  ], "Step Policy Skill");
  const name = requiredText(result.name, "name", 48);
  if (!/^[a-z][a-z0-9_]*$/.test(name)) throw new Error("Skill name must be snake_case");
  const evidenceOccurrenceIds = stringArrayField(
    result.evidence_occurrence_ids,
    "evidence_occurrence_ids",
    2,
    MAX_EVIDENCE_OCCURRENCES
  );
  for (const id of evidenceOccurrenceIds) {
    if (!context.occurrenceEpisodes.has(id)) throw new Error(`Unknown Skill occurrence ref: ${id}`);
  }
  if (new Set(evidenceOccurrenceIds.map((id) => context.occurrenceEpisodes.get(id))).size < 2) {
    throw new Error("Skill evidence must span two Episodes");
  }
  const tools = stringArrayField(result.tools, "tools", 0, 20);
  for (const tool of tools) {
    if (!context.observedTools.has(tool)) throw new Error(`Skill invented tool: ${tool}`);
  }
  return {
    name,
    displayTitle: requiredText(result.display_title, "display_title", 200),
    retrievalBlurb: requiredText(result.retrieval_blurb, "retrieval_blurb", 1_000),
    triggerContext: requiredText(result.trigger_context, "trigger_context", 1_500),
    summary: requiredText(result.summary, "summary", 2_000),
    steps: skillSteps(result.steps, context.validRefs),
    verification: groundedVerification(result.verification, context.validRefs),
    doNotUseWhen: stringArrayField(result.do_not_use_when, "do_not_use_when", 0, 10),
    tools: unique(tools),
    tags: unique(stringArrayField(result.tags, "tags", 0, 20)),
    evidenceOccurrenceIds: unique(evidenceOccurrenceIds),
    confidence: confidence(result.confidence)
  };
}

function groundedSteps(value: unknown, validRefs: ReadonlySet<string>): StepSequencePolicyV1["procedureSteps"] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 12) {
    throw new Error("procedure_steps must contain 1-12 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`procedure_steps[${index}] must be an object`);
    assertExactKeys(item, ["instruction", "evidence_refs"], `procedure_steps[${index}]`);
    return {
      instruction: requiredText(item.instruction, `procedure_steps[${index}].instruction`, 2_000),
      evidenceRefs: evidenceRefs(item.evidence_refs, validRefs)
    };
  });
}

function groundedVerification(
  value: unknown,
  validRefs: ReadonlySet<string>
): StepSequencePolicyV1["verificationSteps"] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 10) {
    throw new Error("verification must contain 1-10 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`verification[${index}] must be an object`);
    assertExactKeys(item, ["check", "success_signal", "evidence_refs"], `verification[${index}]`);
    return {
      check: requiredText(item.check, `verification[${index}].check`, 1_500),
      successSignal: requiredText(item.success_signal, `verification[${index}].success_signal`, 1_500),
      evidenceRefs: evidenceRefs(item.evidence_refs, validRefs)
    };
  });
}

function skillSteps(value: unknown, validRefs: ReadonlySet<string>): StepPolicySequenceSkillDraftV1["steps"] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 16) {
    throw new Error("Skill steps must contain 1-16 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Skill steps[${index}] must be an object`);
    assertExactKeys(item, ["title", "body", "evidence_refs"], `Skill steps[${index}]`);
    return {
      title: requiredText(item.title, `steps[${index}].title`, 300),
      body: requiredText(item.body, `steps[${index}].body`, 2_500),
      evidenceRefs: evidenceRefs(item.evidence_refs, validRefs)
    };
  });
}

function evidenceRefs(value: unknown, validRefs: ReadonlySet<string>): string[] {
  const refs = stringArrayField(value, "evidence_refs", 1, 24);
  for (const ref of refs) {
    if (!validRefs.has(ref)) throw new Error(`Invented evidence ref: ${ref}`);
  }
  return unique(refs);
}

function unmappedNode(
  steps: readonly ReturnType<typeof buildStepOccurrence>[]
): Omit<EpisodeStepPolicyProjectionNodeV1, "nodeIndex"> {
  if (steps.length === 0) throw new Error("UNMAPPED Step node cannot be empty");
  return {
    kind: "unmapped",
    startStepIndex: steps[0]!.stepIndex,
    endStepIndex: steps.at(-1)!.stepIndex,
    stepOccurrenceIds: steps.map((step) => step.id),
    stepIds: steps.map((step) => step.stepId),
    preStateId: steps[0]!.preStateId,
    postStateId: steps.at(-1)!.postStateId
  };
}

function selectDistinctEpisodeOccurrences<T extends { episodeId: string }>(
  occurrences: readonly T[],
  limit: number
): T[] {
  const selected: T[] = [];
  const episodes = new Set<string>();
  for (const occurrence of occurrences) {
    if (episodes.has(occurrence.episodeId)) continue;
    selected.push(occurrence);
    episodes.add(occurrence.episodeId);
    if (selected.length >= limit) break;
  }
  return selected;
}

function renderPolicy(policy: StepSequencePolicyV1): string {
  return [
    `# ${policy.title}`,
    "",
    "## Goal Pattern",
    policy.goalPattern,
    "",
    "## Trigger",
    ...policy.triggerConditions.map((item) => `- ${item}`),
    "",
    "## Procedure",
    ...policy.procedureSteps.map((step, index) => `${index + 1}. ${step.instruction}`),
    "",
    "## Verification",
    ...policy.verificationSteps.map((step) => `- ${step.check} → ${step.successSignal}`),
    "",
    "## Do Not Apply When",
    ...policy.doNotApplyWhen.map((item) => `- ${item}`)
  ].join("\n");
}

function renderSkill(draft: StepPolicySequenceSkillDraftV1): string {
  return [
    `# ${draft.displayTitle}`,
    "",
    draft.retrievalBlurb,
    "",
    "## When to use",
    draft.triggerContext,
    "",
    "## Procedure",
    ...draft.steps.map((step, index) => `${index + 1}. **${step.title}** — ${step.body}`),
    "",
    "## Verification",
    ...draft.verification.map((step) => `- ${step.check} → ${step.successSignal}`),
    "",
    "## Do not use when",
    ...draft.doNotUseWhen.map((item) => `- ${item}`)
  ].join("\n");
}

function assertExactKeys(record: Record<string, unknown>, allowed: string[], field: string): void {
  const actual = Object.keys(record).sort();
  const expected = [...allowed].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${field} keys must be exactly: ${allowed.join(", ")}`);
  }
}

function requiredText(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${field} must be non-empty text`);
  const normalized = value.trim();
  if (normalized.length > maxLength) throw new Error(`${field} exceeds ${maxLength} characters`);
  return normalized;
}

function stringArrayField(
  value: unknown,
  field: string,
  min: number,
  max: number
): string[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) {
    throw new Error(`${field} must contain ${min}-${max} strings`);
  }
  return value.map((item, index) => requiredText(item, `${field}[${index}]`, 2_000));
}

function confidence(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("confidence must be in [0, 1]");
  }
  return value;
}

function repairInstruction(error: unknown): string {
  return `Correct the complete JSON response. Contract violation: ${
    error instanceof Error ? error.message : String(error)
  }. Reuse only evidence refs from the original payload.`;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}
