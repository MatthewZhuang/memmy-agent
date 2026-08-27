import type { MemmyConfig } from "../../config/index.js";
import type { Embedder, LlmClient, LlmMessage } from "../../model/types.js";
import { stableHash, stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import type { ExecutionStepV1 } from "./procedural-path-model.js";
import { averageVectors, cosineSimilarity } from "./step-sequence-learning-model.js";
import {
  bandedMonotonicMatch,
  selfBandedMonotonicMatch,
  type BandedMonotonicMatchConfig,
  type BandedMonotonicMatchResultV1
} from "./trajectory-window-alignment.js";

export const MULTI_SCALE_WINDOW_POLICY_VERSION =
  "multi-scale-window-policy-experiment.v5" as const;
export const MULTI_SCALE_WINDOW_POLICY_OPERATION =
  "procedural.multi_scale_window_policy.induction.v1" as const;
export const MULTI_SCALE_WINDOW_POLICY_PROMPT_VERSION =
  "multi-scale-window-policy-prompt.v2" as const;
export const MULTI_SCALE_WINDOW_SKILL_OPERATION =
  "procedural.multi_scale_window_skill.induction.v1" as const;
export const MULTI_SCALE_WINDOW_SKILL_PROMPT_VERSION =
  "multi-scale-window-skill-prompt.v1" as const;
export const MULTI_SCALE_WINDOW_COARSE_REPRESENTATION =
  "window-intent-sequence-embedding.v1" as const;
export const MULTI_SCALE_WINDOW_FINE_REPRESENTATION =
  "step-intent-banded-monotonic.v1" as const;
export const MULTI_SCALE_WINDOW_COARSE_CLUSTERING_METHOD =
  "constrained-real-medoid.v1" as const;
export const MULTI_SCALE_WINDOW_MULTI_MEMBERSHIP_CLUSTERING_METHOD =
  "incremental-multi-real-medoid.v1" as const;
export type MultiScaleWindowCoarseMembershipMode = "exclusive" | "multi";
export type MultiScaleWindowFineMembershipMode = "exclusive" | "overlap";
export const DEFAULT_MULTI_SCALE_WINDOW_SPECS = [
  { length: 5, stride: 2 },
  { length: 10, stride: 5 }
] as const;
export const DEFAULT_MULTI_SCALE_WINDOW_COARSE_SIMILARITY_THRESHOLD = 0.80;
/** @deprecated Use DEFAULT_MULTI_SCALE_WINDOW_COARSE_SIMILARITY_THRESHOLD. */
export const DEFAULT_MULTI_SCALE_WINDOW_SIMILARITY_THRESHOLD =
  DEFAULT_MULTI_SCALE_WINDOW_COARSE_SIMILARITY_THRESHOLD;
export const DEFAULT_MULTI_SCALE_WINDOW_MIN_SUPPORT = 2;
export const DEFAULT_MULTI_SCALE_WINDOW_MEDOID_SWITCH_MARGIN = 0.01;
export const DEFAULT_MULTI_SCALE_WINDOW_FINE_MATCH_CONFIGS: readonly BandedMonotonicMatchConfig[] = [
  {
    scale: 5,
    bandWidth: 1,
    minStepSimilarity: 0.70,
    minMatchedSteps: 4,
    minCoverage: 0.80,
    minAverageMatchSimilarity: 0.78,
    maxInternalGap: 1,
    gapPenalty: 0.10,
    minAlignmentScore: 0.62
  },
  {
    scale: 10,
    bandWidth: 2,
    minStepSimilarity: 0.68,
    minMatchedSteps: 7,
    minCoverage: 0.70,
    minAverageMatchSimilarity: 0.76,
    maxInternalGap: 2,
    gapPenalty: 0.10,
    minAlignmentScore: 0.52
  }
] as const;

const MAX_POLICY_EVIDENCE_OCCURRENCES = 6;
const MAX_POLICY_REPAIR_ATTEMPTS = 1;
const MAX_SKILL_REPAIR_ATTEMPTS = 1;

export const MULTI_SCALE_WINDOW_POLICY_PROMPT = `You evaluate one cluster of mechanically generated fixed-length execution windows and, only when justified, induce one reusable atomic Policy.

The windows were first grouped by coarse whole-window semantics, then placed in the same fine cluster only when a banded monotonic Step alignment preserved their shared order with sufficient coverage. Their boundaries are mechanical and may include unrelated prefix or suffix Steps. Extract only a coherent procedure that is genuinely repeated across at least two distinct Episodes.

Rules:
- Reject the cluster when its only commonality is a generic phase such as "use a tool", "write a file", "run a script", "verify output", or "respond to the user" without a shared reusable method.
- Reject when the occurrences pursue materially different procedures even if they share tools or successful outcomes.
- Boundary-specific Steps may be omitted. Preserve the shared order and method-defining actions.
- Outcomes may differ across occurrences; use failures as counterfactual context, never as proof that a failing action succeeds.
- A positive Policy may describe only a subprocedure whose successful completion is explicitly observed in at least two distinct Episodes. If every occurrence fails at the full-window goal, narrow the Policy to an earlier repeated successful subprocedure or reject it.
- Every verification item must cite successful Step evidence from at least two distinct Episodes. Never claim successful completion from failure or unknown evidence.
- Do not invent tools, conditions, checks, or outcomes.
- Every procedure and verification item must cite evidence_refs copied exactly from occurrence_id or step_id values in the payload.
- For an admitted Policy, evidence_occurrence_ids must cite at least two occurrences from distinct Episodes.
- Return JSON only with exactly these keys:
{
  "admit": true,
  "rejection_reason": null,
  "title": "...",
  "goal_pattern": "...",
  "trigger_conditions": ["..."],
  "procedure_steps": [{"instruction": "...", "evidence_refs": ["..."]}],
  "verification_steps": [{"check": "...", "success_signal": "...", "evidence_refs": ["..."]}],
  "do_not_apply_when": ["..."],
  "evidence_occurrence_ids": ["trajectory_window_..."],
  "confidence": 0.0
}

For rejection, set admit=false, provide rejection_reason, and return empty strings/arrays for all Policy fields.`;

export const MULTI_SCALE_WINDOW_SKILL_PROMPT = `You evaluate one candidate Skill built from an ordered sequence of already evidence-grounded Policies. Admit it only when the same ordered sequence is observed in at least two distinct Episodes and forms one coherent, reusable capability.

Rules:
- A Skill must contain at least two distinct source Policies. Reject a repeated single Policy or a cosmetic restatement of one Policy.
- Reject when the Policy sequence is only accidentally adjacent, crosses unrelated goals, or lacks a coherent reusable objective.
- Preserve the source Policy order. Do not invent steps, tools, conditions, outcomes, or verification signals.
- The Skill should compress the repeated Policy sequence into a useful invocation guide; reject generic sequences such as "inspect, act, verify" without a method-specific capability.
- Each procedure and verification item must cite source_policy_ids copied exactly from the payload.
- source_policy_ids must include at least two distinct Policies from the candidate.
- evidence_episode_ids must cite at least two distinct supporting Episodes from the payload.
- Return JSON only with exactly these keys:
{
  "admit": true,
  "rejection_reason": null,
  "name": "...",
  "purpose": "...",
  "trigger_conditions": ["..."],
  "procedure_steps": [{"instruction": "...", "source_policy_ids": ["..."]}],
  "verification_steps": [{"check": "...", "success_signal": "...", "source_policy_ids": ["..."]}],
  "do_not_apply_when": ["..."],
  "source_policy_ids": ["..."],
  "evidence_episode_ids": ["..."],
  "confidence": 0.0
}

For rejection, set admit=false, provide rejection_reason, and return empty strings/arrays for all Skill fields.`;

export interface MultiScaleWindowSpec {
  length: number;
  stride: number;
}

export interface MultiScaleWindowEpisodeInput {
  episodeId: string;
  pathId: string;
  terminalReward?: number;
  steps: ExecutionStepV1[];
}

export interface TrajectoryWindowOccurrenceV1 {
  id: string;
  episodeId: string;
  pathId: string;
  terminalReward?: number;
  scale: number;
  stride: number;
  startStepIndex: number;
  endStepIndex: number;
  semanticText: string;
  steps: ExecutionStepV1[];
}

export interface TrajectoryWindowClusterMemberV1 {
  occurrence: TrajectoryWindowOccurrenceV1;
  /** Fine alignment score to the current medoid. */
  similarityToCenter: number;
  alignmentToMedoid: BandedMonotonicMatchResultV1;
}

export interface TrajectoryWindowFamilyV1 {
  id: string;
  scale: number;
  medoidOccurrenceId: string;
  medoidCentrality: number;
  minimumSimilarityToMedoid: number;
  medoidUpdateCount: number;
  occurrenceCount: number;
  supportEpisodeIds: string[];
  memberOccurrenceIds: string[];
}

export interface TrajectoryWindowClusterV1 {
  id: string;
  familyId: string;
  scale: number;
  medoidOccurrenceId: string;
  supportEpisodeIds: string[];
  occurrenceCount: number;
  /** Fine alignment score to medoid; retained under this name for report compatibility. */
  averageSimilarity: number;
  minimumSimilarity: number;
  medoidCentrality: number;
  medoidUpdateCount: number;
  averagePairwiseSimilarity: number;
  minimumPairwiseSimilarity: number;
  members: TrajectoryWindowClusterMemberV1[];
  suppressedByClusterId?: string;
}

export interface MultiScaleWindowPolicyV1 {
  id: string;
  clusterId: string;
  scale: number;
  title: string;
  goalPattern: string;
  triggerConditions: string[];
  procedureSteps: Array<{ instruction: string; evidenceRefs: string[] }>;
  verificationSteps: Array<{
    check: string;
    successSignal: string;
    evidenceRefs: string[];
  }>;
  doNotApplyWhen: string[];
  evidenceOccurrenceIds: string[];
  supportEpisodeIds: string[];
  confidence: number;
  model?: string;
  promptVersion: typeof MULTI_SCALE_WINDOW_POLICY_PROMPT_VERSION;
}

export interface MultiScaleWindowPolicyDecisionV1 {
  clusterId: string;
  admitted: boolean;
  rejectionReason?: string;
  policy?: MultiScaleWindowPolicyV1;
  error?: string;
}

export interface MultiScaleWindowSkillSequenceSpanV1 {
  policyId: string;
  clusterId: string;
  spanId: string;
  startStepIndex: number;
  endStepIndex: number;
}

export interface MultiScaleWindowSkillCandidateV1 {
  id: string;
  policyIds: string[];
  policies: MultiScaleWindowPolicyV1[];
  supportEpisodeIds: string[];
  evidence: Array<{
    episodeId: string;
    spans: MultiScaleWindowSkillSequenceSpanV1[];
  }>;
}

export interface MultiScaleWindowSkillV1 {
  id: string;
  candidateId: string;
  name: string;
  purpose: string;
  triggerConditions: string[];
  procedureSteps: Array<{ instruction: string; sourcePolicyIds: string[] }>;
  verificationSteps: Array<{
    check: string;
    successSignal: string;
    sourcePolicyIds: string[];
  }>;
  doNotApplyWhen: string[];
  sourcePolicyIds: string[];
  evidenceEpisodeIds: string[];
  confidence: number;
  model?: string;
  promptVersion: typeof MULTI_SCALE_WINDOW_SKILL_PROMPT_VERSION;
}

export interface MultiScaleWindowSkillDecisionV1 {
  candidateId: string;
  admitted: boolean;
  rejectionReason?: string;
  skill?: MultiScaleWindowSkillV1;
  error?: string;
}

export interface MultiScaleWindowPolicyExperimentResultV1 {
  schemaVersion: typeof MULTI_SCALE_WINDOW_POLICY_VERSION;
  specs: MultiScaleWindowSpec[];
  coarseSimilarityThreshold: number;
  coarseSimilarityThresholdByScale: Record<number, number>;
  fineMatchConfigs: BandedMonotonicMatchConfig[];
  /** @deprecated Alias of coarseSimilarityThreshold for older diagnostics. */
  similarityThreshold: number;
  minSupportEpisodes: number;
  episodeCount: number;
  stepCount: number;
  embeddedStepCount: number;
  windowCount: number;
  families: TrajectoryWindowFamilyV1[];
  clusters: TrajectoryWindowClusterV1[];
  candidateClusterIds: string[];
  decisions: MultiScaleWindowPolicyDecisionV1[];
  policies: MultiScaleWindowPolicyV1[];
  representations: {
    coarse: typeof MULTI_SCALE_WINDOW_COARSE_REPRESENTATION;
    fine: typeof MULTI_SCALE_WINDOW_FINE_REPRESENTATION;
  };
  coarseMembershipMode: MultiScaleWindowCoarseMembershipMode;
  coarseClusteringMethod:
    | typeof MULTI_SCALE_WINDOW_COARSE_CLUSTERING_METHOD
    | typeof MULTI_SCALE_WINDOW_MULTI_MEMBERSHIP_CLUSTERING_METHOD;
  models: {
    embedding: Record<string, string>;
    evolution: Record<string, string>;
  };
}

export interface PreparedMultiScaleWindowPolicyExperimentV1 {
  specs: MultiScaleWindowSpec[];
  episodeCount: number;
  stepCount: number;
  embeddedStepCount: number;
  windows: EmbeddedTrajectoryWindowV1[];
  models: {
    embedding: Record<string, string>;
    evolution: Record<string, string>;
  };
}

export interface EmbeddedTrajectoryWindowV1 {
  occurrence: TrajectoryWindowOccurrenceV1;
  coarseVector: number[];
  stepVectors: number[][];
}

export interface TrajectoryWindowClusteringResultV1 {
  families: TrajectoryWindowFamilyV1[];
  clusters: TrajectoryWindowClusterV1[];
}

interface InternalFamily {
  scale: number;
  medoid: EmbeddedTrajectoryWindowV1;
  members: EmbeddedTrajectoryWindowV1[];
  medoidCentrality: number;
  minimumSimilarityToMedoid: number;
  medoidUpdateCount: number;
}

interface CoarseMedoidCandidate {
  medoid: EmbeddedTrajectoryWindowV1;
  centrality: number;
  minimumSimilarity: number;
}

interface InternalFineCluster {
  familyId: string;
  scale: number;
  medoid: EmbeddedTrajectoryWindowV1;
  members: EmbeddedTrajectoryWindowV1[];
  pairwise: Map<string, BandedMonotonicMatchResultV1>;
  medoidUpdateCount: number;
}

interface PolicyDraftResult extends Record<string, unknown> {
  admit?: unknown;
  rejection_reason?: unknown;
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
  admit?: unknown;
  rejection_reason?: unknown;
  name?: unknown;
  purpose?: unknown;
  trigger_conditions?: unknown;
  procedure_steps?: unknown;
  verification_steps?: unknown;
  do_not_apply_when?: unknown;
  source_policy_ids?: unknown;
  evidence_episode_ids?: unknown;
  confidence?: unknown;
}

interface EvidenceAlias {
  sourceId: string;
  episodeId: string;
  outcome?: ExecutionStepV1["outcome"]["status"];
}

export interface MultiScaleWindowPolicyExperimentDeps {
  config: MemmyConfig;
  embedder: Embedder;
  llm: LlmClient;
}

export class MultiScaleWindowPolicyExperiment {
  constructor(private readonly deps: MultiScaleWindowPolicyExperimentDeps) {}

  async prepare(input: {
    episodes: readonly MultiScaleWindowEpisodeInput[];
    specs?: readonly MultiScaleWindowSpec[];
  }): Promise<PreparedMultiScaleWindowPolicyExperimentV1> {
    const specs = validateSpecs(input.specs ?? DEFAULT_MULTI_SCALE_WINDOW_SPECS);
    const windows = buildTrajectoryWindows(input.episodes, specs);
    const uniqueSteps = [...new Map(
      windows.flatMap((window) => window.steps).map((step) => [step.id, step])
    ).values()].sort((left, right) => left.id.localeCompare(right.id));
    const [fineStepVectors, coarseWindowVectors] = await Promise.all([
      embedInBatches(
        this.deps.embedder,
        uniqueSteps.map((step) => step.action.intent.trim()),
        64
      ),
      embedInBatches(
        this.deps.embedder,
        windows.map((window) => window.semanticText),
        64
      )
    ]);
    if (fineStepVectors.length !== uniqueSteps.length) {
      throw new Error(
        `Fine Step embedding returned ${fineStepVectors.length} vectors for ${uniqueSteps.length} Steps`
      );
    }
    if (coarseWindowVectors.length !== windows.length) {
      throw new Error(
        `Coarse window embedding returned ${coarseWindowVectors.length} vectors for ${windows.length} windows`
      );
    }
    const fineStepVectorById = new Map(uniqueSteps.map((step, index) =>
      [step.id, fineStepVectors[index]!]));
    const embeddedWindows = windows.map((occurrence, windowIndex) => {
      const occurrenceFineStepVectors = occurrence.steps.map((step) =>
        unitVector(requiredStepVector(fineStepVectorById, step.id, "Fine")));
      return {
        occurrence,
        stepVectors: occurrenceFineStepVectors,
        coarseVector: unitVector(coarseWindowVectors[windowIndex]!)
      };
    });
    return {
      specs,
      episodeCount: input.episodes.length,
      stepCount: input.episodes.reduce((sum, episode) => sum + episode.steps.length, 0),
      embeddedStepCount: uniqueSteps.length,
      windows: embeddedWindows,
      models: {
        embedding: publicModel(this.deps.embedder.config),
        evolution: publicModel(this.deps.llm.config)
      }
    };
  }

  async run(input: {
    episodes: readonly MultiScaleWindowEpisodeInput[];
    specs?: readonly MultiScaleWindowSpec[];
    coarseSimilarityThreshold?: number;
    coarseSimilarityThresholdByScale?: Readonly<Partial<Record<number, number>>>;
    fineMatchConfigs?: readonly BandedMonotonicMatchConfig[];
    medoidSwitchMargin?: number;
    coarseMembershipMode?: MultiScaleWindowCoarseMembershipMode;
    /** @deprecated Alias of coarseSimilarityThreshold. */
    similarityThreshold?: number;
    minSupportEpisodes?: number;
    maxPolicyCandidates?: number;
    policyConcurrency?: number;
    inducePolicies?: boolean;
  }): Promise<MultiScaleWindowPolicyExperimentResultV1> {
    const prepared = await this.prepare({
      episodes: input.episodes,
      ...(input.specs ? { specs: input.specs } : {})
    });
    const { specs } = prepared;
    const coarseSimilarityThreshold = input.coarseSimilarityThreshold ??
      input.similarityThreshold ?? DEFAULT_MULTI_SCALE_WINDOW_COARSE_SIMILARITY_THRESHOLD;
    const coarseSimilarityThresholdByScale = Object.fromEntries(specs.map((spec) => [
      spec.length,
      input.coarseSimilarityThresholdByScale?.[spec.length] ?? coarseSimilarityThreshold
    ])) as Record<number, number>;
    const fineMatchConfigs = validateFineMatchConfigs(
      input.fineMatchConfigs ?? DEFAULT_MULTI_SCALE_WINDOW_FINE_MATCH_CONFIGS,
      specs
    );
    const medoidSwitchMargin = input.medoidSwitchMargin ??
      DEFAULT_MULTI_SCALE_WINDOW_MEDOID_SWITCH_MARGIN;
    const coarseMembershipMode = input.coarseMembershipMode ?? "exclusive";
    const minSupportEpisodes = input.minSupportEpisodes ??
      DEFAULT_MULTI_SCALE_WINDOW_MIN_SUPPORT;
    if (!(coarseSimilarityThreshold > 0 && coarseSimilarityThreshold <= 1)) {
      throw new Error("multi-scale window coarseSimilarityThreshold must be in (0, 1]");
    }
    for (const [scale, threshold] of Object.entries(coarseSimilarityThresholdByScale)) {
      if (!(threshold > 0 && threshold <= 1)) {
        throw new Error(`multi-scale window coarse threshold for Span-${scale} must be in (0, 1]`);
      }
    }
    if (!(medoidSwitchMargin >= 0 && medoidSwitchMargin <= 1)) {
      throw new Error("multi-scale window medoidSwitchMargin must be in [0, 1]");
    }
    if (!Number.isInteger(minSupportEpisodes) || minSupportEpisodes < 2) {
      throw new Error("multi-scale window minSupportEpisodes must be >= 2");
    }
    const discovery = clusterTrajectoryWindows(prepared.windows, {
      coarseSimilarityThreshold,
      coarseSimilarityThresholdByScale,
      fineMatchConfigs,
      medoidSwitchMargin,
      coarseMembershipMode
    });
    const { families, clusters } = discovery;
    const supported = clusters.filter((cluster) =>
      cluster.supportEpisodeIds.length >= minSupportEpisodes);
    const selected = selectMaximalWindowClusters(supported)
      .filter((cluster) => !cluster.suppressedByClusterId)
      .sort((left, right) => right.scale - left.scale ||
        right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
        right.averageSimilarity - left.averageSimilarity ||
        left.id.localeCompare(right.id))
      .slice(0, input.maxPolicyCandidates ?? 8);
    const decisions = input.inducePolicies === false
      ? []
      : await this.inducePolicies(selected, input.policyConcurrency ?? 4);
    return {
      schemaVersion: MULTI_SCALE_WINDOW_POLICY_VERSION,
      specs,
      coarseSimilarityThreshold,
      coarseSimilarityThresholdByScale,
      fineMatchConfigs,
      similarityThreshold: coarseSimilarityThreshold,
      minSupportEpisodes,
      episodeCount: prepared.episodeCount,
      stepCount: prepared.stepCount,
      embeddedStepCount: prepared.embeddedStepCount,
      windowCount: prepared.windows.length,
      families,
      clusters,
      candidateClusterIds: selected.map((cluster) => cluster.id),
      decisions,
      policies: decisions.flatMap((decision) => decision.policy ? [decision.policy] : []),
      representations: {
        coarse: MULTI_SCALE_WINDOW_COARSE_REPRESENTATION,
        fine: MULTI_SCALE_WINDOW_FINE_REPRESENTATION
      },
      coarseMembershipMode,
      coarseClusteringMethod: coarseMembershipMode === "multi"
        ? MULTI_SCALE_WINDOW_MULTI_MEMBERSHIP_CLUSTERING_METHOD
        : MULTI_SCALE_WINDOW_COARSE_CLUSTERING_METHOD,
      models: prepared.models
    };
  }

  async inducePolicies(
    clusters: readonly TrajectoryWindowClusterV1[],
    concurrency = 4
  ): Promise<MultiScaleWindowPolicyDecisionV1[]> {
    if (!Number.isInteger(concurrency) || concurrency < 1) {
      throw new Error("Policy induction concurrency must be a positive integer");
    }
    return mapConcurrent(clusters, concurrency, async (cluster) => this.inducePolicy(cluster));
  }

  async induceSkills(
    candidates: readonly MultiScaleWindowSkillCandidateV1[],
    concurrency = 4
  ): Promise<MultiScaleWindowSkillDecisionV1[]> {
    if (!Number.isInteger(concurrency) || concurrency < 1) {
      throw new Error("Skill induction concurrency must be a positive integer");
    }
    return mapConcurrent(candidates, concurrency, async (candidate) =>
      this.induceSkill(candidate));
  }

  private async inducePolicy(
    cluster: TrajectoryWindowClusterV1
  ): Promise<MultiScaleWindowPolicyDecisionV1> {
    if (!this.deps.llm.isConfigured()) {
      throw new Error("multi-scale window Policy induction requires a configured LLM");
    }
    const selected = selectDistinctEpisodeMembers(
      cluster.members,
      MAX_POLICY_EVIDENCE_OCCURRENCES
    );
    const refAliases = new Map<string, EvidenceAlias>();
    const occurrenceEpisodes = new Map<string, string>();
    const evidence = selected.map((member, occurrenceIndex) => {
      const occurrence = member.occurrence;
      const occurrenceAlias = `O${occurrenceIndex + 1}`;
      refAliases.set(occurrenceAlias, {
        sourceId: occurrence.id,
        episodeId: occurrence.episodeId
      });
      occurrenceEpisodes.set(occurrenceAlias, occurrence.episodeId);
      return {
        occurrence_id: occurrenceAlias,
        episode_id: occurrence.episodeId,
        step_range: [occurrence.startStepIndex, occurrence.endStepIndex],
        terminal_reward: occurrence.terminalReward ?? null,
        alignment_to_medoid: {
          score: member.alignmentToMedoid.score,
          average_match_similarity: member.alignmentToMedoid.averageMatchSimilarity,
          coverage: member.alignmentToMedoid.coverage,
          matched_steps: member.alignmentToMedoid.matchedSteps,
          internal_gap_steps: member.alignmentToMedoid.internalGapSteps,
          pairs: member.alignmentToMedoid.pairs
        },
        steps: occurrence.steps.map((step, stepOffset) => {
          const stepAlias = `${occurrenceAlias}.S${stepOffset + 1}`;
          refAliases.set(stepAlias, {
            sourceId: step.id,
            episodeId: occurrence.episodeId,
            outcome: step.outcome.status
          });
          return {
            step_id: stepAlias,
            step_index: step.stepIndex,
            action_type: step.action.type,
            tool_name: step.action.toolName ?? null,
            intent: step.action.intent,
            summary: step.action.summary,
            outcome: step.outcome.status
          };
        })
      };
    });
    const payload = {
      cluster: {
        cluster_id: cluster.id,
        family_id: cluster.familyId,
        scale: cluster.scale,
        medoid_occurrence_id: cluster.medoidOccurrenceId,
        support_episode_count: cluster.supportEpisodeIds.length,
        average_alignment_to_medoid: cluster.averageSimilarity,
        minimum_alignment_to_medoid: cluster.minimumSimilarity,
        average_pairwise_similarity: cluster.averagePairwiseSimilarity,
        minimum_pairwise_similarity: cluster.minimumPairwiseSimilarity
      },
      evidence
    };
    const messages: LlmMessage[] = [
      { role: "system", content: MULTI_SCALE_WINDOW_POLICY_PROMPT },
      { role: "user", content: stableStringify(payload) }
    ];
    const options = {
      operation: MULTI_SCALE_WINDOW_POLICY_OPERATION,
      thinkingMode: this.deps.config.evolution.enableThinking
        ? "enabled" as const
        : "disabled" as const,
      temperature: 0.1,
      maxTokens: 8_000,
      jsonMode: true
    };
    try {
      let result = await this.deps.llm.completeJson<PolicyDraftResult>(messages, options);
      for (let repair = 0; ; repair += 1) {
        try {
          return parsePolicyDecision(result, cluster, {
            refAliases,
            occurrenceEpisodes,
            model: this.deps.llm.config.model
          });
        } catch (error) {
          if (repair >= MAX_POLICY_REPAIR_ATTEMPTS) throw error;
          result = await this.deps.llm.completeJson<PolicyDraftResult>([
            ...messages,
            { role: "assistant", content: stableStringify(result) },
            {
              role: "user",
              content: `Correct only the JSON schema and evidence grounding. Error: ${error instanceof Error ? error.message : String(error)}. Return JSON only.`
            }
          ], { ...options, operation: `${MULTI_SCALE_WINDOW_POLICY_OPERATION}.repair` });
        }
      }
    } catch (error) {
      return {
        clusterId: cluster.id,
        admitted: false,
        error: error instanceof Error ? error.stack ?? error.message : String(error)
      };
    }
  }

  private async induceSkill(
    candidate: MultiScaleWindowSkillCandidateV1
  ): Promise<MultiScaleWindowSkillDecisionV1> {
    if (!this.deps.llm.isConfigured()) {
      throw new Error("multi-scale window Skill induction requires a configured LLM");
    }
    const payload = {
      candidate_id: candidate.id,
      support_episode_ids: candidate.supportEpisodeIds,
      policies: candidate.policies.map((policy) => ({
        policy_id: policy.id,
        title: policy.title,
        goal_pattern: policy.goalPattern,
        trigger_conditions: policy.triggerConditions,
        procedure_steps: policy.procedureSteps.map((step) => step.instruction),
        verification_steps: policy.verificationSteps.map((step) => ({
          check: step.check,
          success_signal: step.successSignal
        })),
        do_not_apply_when: policy.doNotApplyWhen
      })),
      evidence: candidate.evidence.map((item) => ({
        episode_id: item.episodeId,
        ordered_spans: item.spans.map((span) => ({
          policy_id: span.policyId,
          cluster_id: span.clusterId,
          span_id: span.spanId,
          step_range: [span.startStepIndex, span.endStepIndex]
        }))
      }))
    };
    const messages: LlmMessage[] = [
      { role: "system", content: MULTI_SCALE_WINDOW_SKILL_PROMPT },
      { role: "user", content: stableStringify(payload) }
    ];
    const options = {
      operation: MULTI_SCALE_WINDOW_SKILL_OPERATION,
      thinkingMode: this.deps.config.evolution.enableThinking
        ? "enabled" as const
        : "disabled" as const,
      temperature: 0.1,
      maxTokens: 8_000,
      jsonMode: true
    };
    try {
      let result = await this.deps.llm.completeJson<SkillDraftResult>(messages, options);
      for (let repair = 0; ; repair += 1) {
        try {
          return parseSkillDecision(result, candidate, this.deps.llm.config.model);
        } catch (error) {
          if (repair >= MAX_SKILL_REPAIR_ATTEMPTS) throw error;
          result = await this.deps.llm.completeJson<SkillDraftResult>([
            ...messages,
            { role: "assistant", content: stableStringify(result) },
            {
              role: "user",
              content: `Correct only the JSON schema and source grounding. Error: ${error instanceof Error ? error.message : String(error)}. Return JSON only.`
            }
          ], { ...options, operation: `${MULTI_SCALE_WINDOW_SKILL_OPERATION}.repair` });
        }
      }
    } catch (error) {
      return {
        candidateId: candidate.id,
        admitted: false,
        error: error instanceof Error ? error.stack ?? error.message : String(error)
      };
    }
  }
}

export function buildTrajectoryWindows(
  episodes: readonly MultiScaleWindowEpisodeInput[],
  specs: readonly MultiScaleWindowSpec[] = DEFAULT_MULTI_SCALE_WINDOW_SPECS
): TrajectoryWindowOccurrenceV1[] {
  const validSpecs = validateSpecs(specs);
  const windows: TrajectoryWindowOccurrenceV1[] = [];
  for (const episode of [...episodes].sort((left, right) =>
    left.episodeId.localeCompare(right.episodeId))) {
    for (const spec of validSpecs) {
      if (episode.steps.length < spec.length) continue;
      const starts: number[] = [];
      for (let start = 0; start + spec.length <= episode.steps.length; start += spec.stride) {
        starts.push(start);
      }
      const tailStart = episode.steps.length - spec.length;
      if (starts.at(-1) !== tailStart) starts.push(tailStart);
      for (const start of starts) {
        const steps = episode.steps.slice(start, start + spec.length);
        const startStepIndex = steps[0]!.stepIndex;
        const endStepIndex = steps.at(-1)!.stepIndex;
        const semanticText = steps.map((step, index) =>
          `${index + 1}. ${step.action.intent.trim()}`).join("\n");
        windows.push({
          id: `trajectory_window_${stableHash({
            version: MULTI_SCALE_WINDOW_POLICY_VERSION,
            episodeId: episode.episodeId,
            pathId: episode.pathId,
            scale: spec.length,
            startStepIndex,
            endStepIndex,
            semanticText
          }).slice(0, 24)}`,
          episodeId: episode.episodeId,
          pathId: episode.pathId,
          ...(episode.terminalReward === undefined
            ? {}
            : { terminalReward: episode.terminalReward }),
          scale: spec.length,
          stride: spec.stride,
          startStepIndex,
          endStepIndex,
          semanticText,
          steps
        });
      }
    }
  }
  return windows;
}

export function clusterTrajectoryWindows(
  windows: readonly EmbeddedTrajectoryWindowV1[],
  options: {
    coarseSimilarityThreshold: number;
    coarseSimilarityThresholdByScale?: Readonly<Partial<Record<number, number>>>;
    fineMatchConfigs: readonly BandedMonotonicMatchConfig[];
    medoidSwitchMargin?: number;
    coarseMembershipMode?: MultiScaleWindowCoarseMembershipMode;
    fineMembershipMode?: MultiScaleWindowFineMembershipMode;
  }
): TrajectoryWindowClusteringResultV1 {
  const medoidSwitchMargin = options.medoidSwitchMargin ??
    DEFAULT_MULTI_SCALE_WINDOW_MEDOID_SWITCH_MARGIN;
  const coarseMembershipMode = options.coarseMembershipMode ?? "exclusive";
  const fineMembershipMode = options.fineMembershipMode ?? "exclusive";
  const fineConfigByScale = new Map(options.fineMatchConfigs.map((config) => [config.scale, config]));
  const thresholdForScale = (scale: number): number =>
    options.coarseSimilarityThresholdByScale?.[scale] ?? options.coarseSimilarityThreshold;
  for (const scale of new Set(windows.map((window) => window.occurrence.scale))) {
    const threshold = thresholdForScale(scale);
    if (!(threshold > 0 && threshold <= 1)) {
      throw new Error(`multi-scale window coarse threshold for Span-${scale} must be in (0, 1]`);
    }
  }
  const orderedWindows = [...windows].sort((left, right) =>
    left.occurrence.scale - right.occurrence.scale ||
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex);
  const internalFamilies = coarseMembershipMode === "multi"
    ? buildMultiMembershipCoarseFamilies(orderedWindows, thresholdForScale)
    : buildExclusiveCoarseFamilies(orderedWindows, thresholdForScale);
  const families: TrajectoryWindowFamilyV1[] = [];
  const rawClusters: TrajectoryWindowClusterV1[] = [];
  for (const family of internalFamilies) {
    const memberOccurrenceIds = family.members.map((member) => member.occurrence.id).sort();
    const familyId = `trajectory_window_family_${stableHash({
      version: MULTI_SCALE_WINDOW_POLICY_VERSION,
      ...(coarseMembershipMode === "exclusive" ? {} : { coarseMembershipMode }),
      scale: family.scale,
      memberOccurrenceIds
    }).slice(0, 24)}`;
    families.push({
      id: familyId,
      scale: family.scale,
      medoidOccurrenceId: family.medoid.occurrence.id,
      medoidCentrality: family.medoidCentrality,
      minimumSimilarityToMedoid: family.minimumSimilarityToMedoid,
      medoidUpdateCount: family.medoidUpdateCount,
      occurrenceCount: family.members.length,
      supportEpisodeIds: [...new Set(
        family.members.map((member) => member.occurrence.episodeId)
      )].sort(),
      memberOccurrenceIds
    });
    const fineConfig = fineConfigByScale.get(family.scale);
    if (!fineConfig) throw new Error(`missing fine match config for scale ${family.scale}`);
    rawClusters.push(...(fineMembershipMode === "overlap"
      ? clusterOverlappingFineFamily(familyId, family.members, fineConfig)
      : clusterFineFamily(familyId, family.members, fineConfig, medoidSwitchMargin)));
  }
  const clusters = coarseMembershipMode === "multi" || fineMembershipMode === "overlap"
    ? dedupeEquivalentFineClusters(rawClusters)
    : rawClusters;
  return {
    families: families.sort((left, right) => left.scale - right.scale ||
      right.occurrenceCount - left.occurrenceCount || left.id.localeCompare(right.id)),
    clusters: clusters.sort((left, right) => left.scale - right.scale ||
      right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
      right.averageSimilarity - left.averageSimilarity || left.id.localeCompare(right.id))
  };
}

function clusterOverlappingFineFamily(
  familyId: string,
  windows: readonly EmbeddedTrajectoryWindowV1[],
  config: BandedMonotonicMatchConfig
): TrajectoryWindowClusterV1[] {
  const ordered = [...windows].sort((left, right) =>
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex);
  const candidates = ordered.flatMap((anchor) => {
    const members = ordered.filter((member) =>
      member.occurrence.id === anchor.occurrence.id ||
      bandedMonotonicMatch(member.stepVectors, anchor.stepVectors, config).admitted);
    const cluster: InternalFineCluster = {
      familyId,
      scale: anchor.occurrence.scale,
      medoid: anchor,
      members,
      pairwise: new Map(),
      medoidUpdateCount: 0
    };
    const finalized = finalizeFineCluster(cluster, config);
    return finalized ? [finalized] : [];
  });
  return dedupeEquivalentFineClusters(candidates)
    .sort((left, right) => right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
      right.averageSimilarity - left.averageSimilarity || left.id.localeCompare(right.id));
}

function buildExclusiveCoarseFamilies(
  windows: readonly EmbeddedTrajectoryWindowV1[],
  thresholdForScale: (scale: number) => number
): InternalFamily[] {
  const internalFamilies: InternalFamily[] = [];
  for (const window of windows) {
    let best: {
      family: InternalFamily;
      candidate: CoarseMedoidCandidate;
      similarityToNewMember: number;
    } | undefined;
    for (const family of internalFamilies) {
      if (family.scale !== window.occurrence.scale) continue;
      const similarityThreshold = thresholdForScale(family.scale);
      const candidate = constrainedCoarseMedoid(
        [...family.members, window],
        similarityThreshold
      );
      if (!candidate) continue;
      const similarityToNewMember = cosineSimilarity(
        window.coarseVector,
        candidate.medoid.coarseVector
      );
      if (!best || similarityToNewMember > best.similarityToNewMember ||
          (similarityToNewMember === best.similarityToNewMember &&
            candidate.centrality > best.candidate.centrality)) {
        best = { family, candidate, similarityToNewMember };
      }
    }
    if (!best) {
      internalFamilies.push({
        scale: window.occurrence.scale,
        medoid: window,
        members: [window],
        medoidCentrality: 1,
        minimumSimilarityToMedoid: 1,
        medoidUpdateCount: 0
      });
      continue;
    }
    const previousMedoidId = best.family.medoid.occurrence.id;
    best.family.members.push(window);
    best.family.medoid = best.candidate.medoid;
    best.family.medoidCentrality = best.candidate.centrality;
    best.family.minimumSimilarityToMedoid = best.candidate.minimumSimilarity;
    if (previousMedoidId !== best.candidate.medoid.occurrence.id) {
      best.family.medoidUpdateCount += 1;
    }
  }
  mergeCoarseFamilies(internalFamilies, thresholdForScale);
  return internalFamilies;
}

function buildMultiMembershipCoarseFamilies(
  windows: readonly EmbeddedTrajectoryWindowV1[],
  thresholdForScale: (scale: number) => number
): InternalFamily[] {
  const families: InternalFamily[] = [];
  for (const window of windows) {
    const similarityThreshold = thresholdForScale(window.occurrence.scale);
    const matched = families.filter((family) =>
      family.scale === window.occurrence.scale &&
      cosineSimilarity(window.coarseVector, family.medoid.coarseVector) + Number.EPSILON >=
        similarityThreshold);
    if (matched.length === 0) {
      families.push({
        scale: window.occurrence.scale,
        medoid: window,
        members: [window],
        medoidCentrality: 1,
        minimumSimilarityToMedoid: 1,
        medoidUpdateCount: 0
      });
      continue;
    }
    for (const family of matched) {
      const previousMedoidId = family.medoid.occurrence.id;
      const candidate = constrainedCoarseMedoid(
        [...family.members, window],
        similarityThreshold
      );
      if (!candidate) {
        throw new Error("multi-membership Family lost its threshold-valid real medoid");
      }
      family.members.push(window);
      family.medoid = candidate.medoid;
      family.medoidCentrality = candidate.centrality;
      family.minimumSimilarityToMedoid = candidate.minimumSimilarity;
      if (previousMedoidId !== candidate.medoid.occurrence.id) {
        family.medoidUpdateCount += 1;
      }
    }
  }
  return dedupeEquivalentCoarseFamilies(families);
}

function dedupeEquivalentCoarseFamilies(
  families: readonly InternalFamily[]
): InternalFamily[] {
  const byMembers = new Map<string, InternalFamily>();
  for (const family of families) {
    const key = `${family.scale}:${family.members.map((member) =>
      member.occurrence.id).sort().join(",")}`;
    const existing = byMembers.get(key);
    if (!existing || family.medoidCentrality > existing.medoidCentrality ||
        (family.medoidCentrality === existing.medoidCentrality &&
          family.minimumSimilarityToMedoid > existing.minimumSimilarityToMedoid) ||
        (family.medoidCentrality === existing.medoidCentrality &&
          family.minimumSimilarityToMedoid === existing.minimumSimilarityToMedoid &&
          family.medoid.occurrence.id.localeCompare(existing.medoid.occurrence.id) < 0)) {
      byMembers.set(key, family);
    }
  }
  return [...byMembers.values()];
}

function dedupeEquivalentFineClusters(
  clusters: readonly TrajectoryWindowClusterV1[]
): TrajectoryWindowClusterV1[] {
  const byEvidence = new Map<string, TrajectoryWindowClusterV1>();
  for (const cluster of clusters) {
    const key = `${cluster.scale}:${cluster.members.map((member) =>
      member.occurrence.id).sort().join(",")}`;
    const existing = byEvidence.get(key);
    if (!existing || cluster.averageSimilarity > existing.averageSimilarity ||
        (cluster.averageSimilarity === existing.averageSimilarity &&
          cluster.averagePairwiseSimilarity > existing.averagePairwiseSimilarity) ||
        (cluster.averageSimilarity === existing.averageSimilarity &&
          cluster.averagePairwiseSimilarity === existing.averagePairwiseSimilarity &&
          cluster.id.localeCompare(existing.id) < 0)) {
      byEvidence.set(key, cluster);
    }
  }
  return [...byEvidence.values()];
}

function clusterFineFamily(
  familyId: string,
  windows: readonly EmbeddedTrajectoryWindowV1[],
  config: BandedMonotonicMatchConfig,
  medoidSwitchMargin: number
): TrajectoryWindowClusterV1[] {
  const internal: InternalFineCluster[] = [];
  for (const window of [...windows].sort((left, right) =>
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex)) {
    let bestCluster: InternalFineCluster | undefined;
    let bestMatch: BandedMonotonicMatchResultV1 | undefined;
    for (const cluster of internal) {
      const match = bandedMonotonicMatch(window.stepVectors, cluster.medoid.stepVectors, config);
      if (!match.admitted) continue;
      if (!bestMatch || match.score > bestMatch.score ||
          (match.score === bestMatch.score &&
            match.averageMatchSimilarity > bestMatch.averageMatchSimilarity)) {
        bestCluster = cluster;
        bestMatch = match;
      }
    }
    if (!bestCluster) {
      internal.push({
        familyId,
        scale: window.occurrence.scale,
        medoid: window,
        members: [window],
        pairwise: new Map(),
        medoidUpdateCount: 0
      });
      continue;
    }
    attachFineMember(bestCluster, window, config, medoidSwitchMargin);
  }
  mergeFineClusters(internal, config);
  return internal.map((cluster) => finalizeFineCluster(cluster, config))
    .filter((cluster): cluster is TrajectoryWindowClusterV1 => Boolean(cluster))
    .sort((left, right) => right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
      right.averageSimilarity - left.averageSimilarity || left.id.localeCompare(right.id));
}

function mergeFineClusters(
  clusters: InternalFineCluster[],
  config: BandedMonotonicMatchConfig
): void {
  const rejectedPairs = new Set<string>();
  while (true) {
    let bestPair: [number, number] | undefined;
    let bestMatch: BandedMonotonicMatchResultV1 | undefined;
    for (let left = 0; left < clusters.length; left += 1) {
      for (let right = left + 1; right < clusters.length; right += 1) {
        const rejectionKey = fineClusterPairKey(clusters[left]!, clusters[right]!);
        if (rejectedPairs.has(rejectionKey)) continue;
        const match = bandedMonotonicMatch(
          clusters[left]!.medoid.stepVectors,
          clusters[right]!.medoid.stepVectors,
          config
        );
        if (!match.admitted) continue;
        if (!bestMatch || match.score > bestMatch.score ||
            (match.score === bestMatch.score &&
              match.averageMatchSimilarity > bestMatch.averageMatchSimilarity)) {
          bestPair = [left, right];
          bestMatch = match;
        }
      }
    }
    if (!bestPair) return;
    const [left, right] = bestPair;
    const merged = mergeFineClusterPair(clusters[left]!, clusters[right]!, config);
    if (!merged) {
      rejectedPairs.add(fineClusterPairKey(clusters[left]!, clusters[right]!));
      continue;
    }
    clusters[left] = merged;
    clusters.splice(right, 1);
    rejectedPairs.clear();
  }
}

function mergeFineClusterPair(
  left: InternalFineCluster,
  right: InternalFineCluster,
  config: BandedMonotonicMatchConfig
): InternalFineCluster | undefined {
  const members = [...new Map(
    [...left.members, ...right.members].map((member) => [member.occurrence.id, member])
  ).values()];
  const merged: InternalFineCluster = {
    familyId: left.familyId,
    scale: left.scale,
    medoid: left.medoid,
    members,
    pairwise: new Map(),
    medoidUpdateCount: left.medoidUpdateCount + right.medoidUpdateCount
  };
  for (let first = 0; first < members.length; first += 1) {
    for (let second = first + 1; second < members.length; second += 1) {
      merged.pairwise.set(
        pairKey(members[first]!.occurrence.id, members[second]!.occurrence.id),
        bandedMonotonicMatch(members[first]!.stepVectors, members[second]!.stepVectors, config)
      );
    }
  }
  const candidate = [...members].sort((first, second) =>
    memberCentrality(merged, second) - memberCentrality(merged, first) ||
    first.occurrence.id.localeCompare(second.occurrence.id))
    .find((possible) => members.every((member) =>
      member.occurrence.id === possible.occurrence.id ||
      pairResult(merged, member, possible, config).admitted));
  if (!candidate) return undefined;
  if (candidate.occurrence.id !== left.medoid.occurrence.id) merged.medoidUpdateCount += 1;
  merged.medoid = candidate;
  return merged;
}

function fineClusterPairKey(left: InternalFineCluster, right: InternalFineCluster): string {
  const leftKey = left.members.map((member) => member.occurrence.id).sort().join(",");
  const rightKey = right.members.map((member) => member.occurrence.id).sort().join(",");
  return leftKey < rightKey ? `${leftKey}|${rightKey}` : `${rightKey}|${leftKey}`;
}

export function selectMaximalWindowClusters(
  clusters: readonly TrajectoryWindowClusterV1[]
): TrajectoryWindowClusterV1[] {
  const selected: TrajectoryWindowClusterV1[] = [];
  for (const cluster of [...clusters].sort((left, right) =>
    right.scale - left.scale ||
    right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
    right.averageSimilarity - left.averageSimilarity || left.id.localeCompare(right.id))) {
    const covering = selected.find((candidate) => clusterFullyCoveredBy(cluster, candidate));
    selected.push(covering ? { ...cluster, suppressedByClusterId: covering.id } : cluster);
  }
  return selected;
}

function mergeCoarseFamilies(
  families: InternalFamily[],
  thresholdForScale: (scale: number) => number
): void {
  while (true) {
    let bestPair: [number, number] | undefined;
    let bestCandidate: CoarseMedoidCandidate | undefined;
    for (let left = 0; left < families.length; left += 1) {
      for (let right = left + 1; right < families.length; right += 1) {
        if (families[left]!.scale !== families[right]!.scale) continue;
        const similarityThreshold = thresholdForScale(families[left]!.scale);
        const candidate = constrainedCoarseMedoid(
          [...families[left]!.members, ...families[right]!.members],
          similarityThreshold
        );
        if (candidate && (!bestCandidate || candidate.centrality > bestCandidate.centrality ||
            (candidate.centrality === bestCandidate.centrality &&
              candidate.minimumSimilarity > bestCandidate.minimumSimilarity))) {
          bestPair = [left, right];
          bestCandidate = candidate;
        }
      }
    }
    if (!bestPair || !bestCandidate) return;
    const [left, right] = bestPair;
    const mergedMembers = [...families[left]!.members, ...families[right]!.members];
    const previousMedoidId = families[left]!.medoid.occurrence.id;
    families[left] = {
      scale: families[left]!.scale,
      medoid: bestCandidate.medoid,
      members: mergedMembers,
      medoidCentrality: bestCandidate.centrality,
      minimumSimilarityToMedoid: bestCandidate.minimumSimilarity,
      medoidUpdateCount: families[left]!.medoidUpdateCount +
        families[right]!.medoidUpdateCount +
        (previousMedoidId === bestCandidate.medoid.occurrence.id ? 0 : 1)
    };
    families.splice(right, 1);
  }
}

function constrainedCoarseMedoid(
  members: readonly EmbeddedTrajectoryWindowV1[],
  similarityThreshold: number
): CoarseMedoidCandidate | undefined {
  let best: CoarseMedoidCandidate | undefined;
  for (const possible of members) {
    const similarities = members.map((member) => cosineSimilarity(
      possible.coarseVector,
      member.coarseVector
    ));
    const minimumSimilarity = Math.min(...similarities);
    if (minimumSimilarity + Number.EPSILON < similarityThreshold) continue;
    const centrality = similarities.reduce((sum, value) => sum + value, 0) /
      similarities.length;
    if (!best || centrality > best.centrality ||
        (centrality === best.centrality && minimumSimilarity > best.minimumSimilarity) ||
        (centrality === best.centrality && minimumSimilarity === best.minimumSimilarity &&
          possible.occurrence.id.localeCompare(best.medoid.occurrence.id) < 0)) {
      best = { medoid: possible, centrality, minimumSimilarity };
    }
  }
  return best;
}

function attachFineMember(
  cluster: InternalFineCluster,
  member: EmbeddedTrajectoryWindowV1,
  config: BandedMonotonicMatchConfig,
  medoidSwitchMargin: number
): void {
  for (const existing of cluster.members) {
    cluster.pairwise.set(
      pairKey(existing.occurrence.id, member.occurrence.id),
      bandedMonotonicMatch(existing.stepVectors, member.stepVectors, config)
    );
  }
  cluster.members.push(member);
  const ranked = [...cluster.members].sort((left, right) =>
    memberCentrality(cluster, right) - memberCentrality(cluster, left) ||
    left.occurrence.id.localeCompare(right.occurrence.id));
  const candidate = ranked[0]!;
  if (candidate.occurrence.id === cluster.medoid.occurrence.id) return;
  const improvement = memberCentrality(cluster, candidate) - memberCentrality(cluster, cluster.medoid);
  if (improvement <= medoidSwitchMargin) return;
  if (!cluster.members.every((existing) => existing.occurrence.id === candidate.occurrence.id ||
    pairResult(cluster, existing, candidate, config).admitted)) return;
  cluster.medoid = candidate;
  cluster.medoidUpdateCount += 1;
}

function finalizeFineCluster(
  cluster: InternalFineCluster,
  config: BandedMonotonicMatchConfig
): TrajectoryWindowClusterV1 | undefined {
  const nonOverlapping = dedupeOverlappingFineMembers(cluster.members, cluster.medoid, config);
  if (nonOverlapping.length === 0) return undefined;
  const activeCluster: InternalFineCluster = {
    ...cluster,
    members: nonOverlapping,
    pairwise: new Map()
  };
  for (let left = 0; left < nonOverlapping.length; left += 1) {
    for (let right = left + 1; right < nonOverlapping.length; right += 1) {
      activeCluster.pairwise.set(
        pairKey(nonOverlapping[left]!.occurrence.id, nonOverlapping[right]!.occurrence.id),
        bandedMonotonicMatch(
          nonOverlapping[left]!.stepVectors,
          nonOverlapping[right]!.stepVectors,
          config
        )
      );
    }
  }
  const ranked = [...nonOverlapping].sort((left, right) =>
    memberCentrality(activeCluster, right) - memberCentrality(activeCluster, left) ||
    left.occurrence.id.localeCompare(right.occurrence.id));
  const eligibleMedoid = ranked.find((candidate) => nonOverlapping.every((member) =>
    member.occurrence.id === candidate.occurrence.id ||
    pairResult(activeCluster, member, candidate, config).admitted));
  const medoid = eligibleMedoid ??
    nonOverlapping.find((member) => member.occurrence.id === cluster.medoid.occurrence.id) ??
    ranked[0]!;
  const medoidChangedDuringFinalization = medoid.occurrence.id !== cluster.medoid.occurrence.id;
  activeCluster.medoid = medoid;
  const members = nonOverlapping.map((member) => {
    const alignmentToMedoid = member.occurrence.id === medoid.occurrence.id
      ? selfBandedMonotonicMatch(member.stepVectors, config)
      : bandedMonotonicMatch(member.stepVectors, medoid.stepVectors, config);
    return {
      occurrence: member.occurrence,
      similarityToCenter: alignmentToMedoid.score,
      alignmentToMedoid
    };
  }).sort((left, right) => left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex);
  const similarities = members.map((member) => member.similarityToCenter);
  const pairwiseSimilarities: number[] = [];
  for (let left = 0; left < nonOverlapping.length; left += 1) {
    for (let right = left + 1; right < nonOverlapping.length; right += 1) {
      pairwiseSimilarities.push(pairResult(
        activeCluster,
        nonOverlapping[left]!,
        nonOverlapping[right]!,
        config
      ).score);
    }
  }
  const supportEpisodeIds = [...new Set(members.map((member) => member.occurrence.episodeId))].sort();
  const signature = members.map((member) => ({
    episodeId: member.occurrence.episodeId,
    start: member.occurrence.startStepIndex,
    end: member.occurrence.endStepIndex
  }));
  return {
    id: `trajectory_window_cluster_${stableHash({
      version: MULTI_SCALE_WINDOW_POLICY_VERSION,
      familyId: cluster.familyId,
      scale: cluster.scale,
      signature
    }).slice(0, 24)}`,
    familyId: cluster.familyId,
    scale: cluster.scale,
    medoidOccurrenceId: medoid.occurrence.id,
    supportEpisodeIds,
    occurrenceCount: members.length,
    averageSimilarity: average(similarities),
    minimumSimilarity: Math.min(...similarities),
    medoidCentrality: memberCentrality(activeCluster, medoid),
    medoidUpdateCount: cluster.medoidUpdateCount + (medoidChangedDuringFinalization ? 1 : 0),
    averagePairwiseSimilarity: pairwiseSimilarities.length > 0
      ? average(pairwiseSimilarities)
      : 1,
    minimumPairwiseSimilarity: pairwiseSimilarities.length > 0
      ? Math.min(...pairwiseSimilarities)
      : 1,
    members
  };
}

function dedupeOverlappingFineMembers(
  members: readonly EmbeddedTrajectoryWindowV1[],
  medoid: EmbeddedTrajectoryWindowV1,
  config: BandedMonotonicMatchConfig
): EmbeddedTrajectoryWindowV1[] {
  const byEpisode = new Map<string, EmbeddedTrajectoryWindowV1[]>();
  for (const member of members) {
    const values = byEpisode.get(member.occurrence.episodeId) ?? [];
    values.push(member);
    byEpisode.set(member.occurrence.episodeId, values);
  }
  const selected: EmbeddedTrajectoryWindowV1[] = [];
  for (const values of byEpisode.values()) {
    const kept: EmbeddedTrajectoryWindowV1[] = [];
    for (const member of [...values].sort((left, right) =>
      fineScoreToMedoid(right, medoid, config) - fineScoreToMedoid(left, medoid, config) ||
      left.occurrence.startStepIndex - right.occurrence.startStepIndex)) {
      if (kept.some((existing) => rangesOverlap(
        existing.occurrence.startStepIndex,
        existing.occurrence.endStepIndex,
        member.occurrence.startStepIndex,
        member.occurrence.endStepIndex
      ))) continue;
      kept.push(member);
    }
    selected.push(...kept);
  }
  return selected;
}

function fineScoreToMedoid(
  member: EmbeddedTrajectoryWindowV1,
  medoid: EmbeddedTrajectoryWindowV1,
  config: BandedMonotonicMatchConfig
): number {
  return member.occurrence.id === medoid.occurrence.id
    ? 1
    : bandedMonotonicMatch(member.stepVectors, medoid.stepVectors, config).score;
}

function memberCentrality(
  cluster: InternalFineCluster,
  member: EmbeddedTrajectoryWindowV1
): number {
  if (cluster.members.length <= 1) return 1;
  let sum = 0;
  for (const other of cluster.members) {
    if (other.occurrence.id === member.occurrence.id) continue;
    sum += cluster.pairwise.get(pairKey(member.occurrence.id, other.occurrence.id))?.score ?? 0;
  }
  return sum / (cluster.members.length - 1);
}

function pairResult(
  cluster: InternalFineCluster,
  left: EmbeddedTrajectoryWindowV1,
  right: EmbeddedTrajectoryWindowV1,
  config: BandedMonotonicMatchConfig
): BandedMonotonicMatchResultV1 {
  const key = pairKey(left.occurrence.id, right.occurrence.id);
  const existing = cluster.pairwise.get(key);
  if (existing) return existing;
  const computed = bandedMonotonicMatch(left.stepVectors, right.stepVectors, config);
  cluster.pairwise.set(key, computed);
  return computed;
}

function pairKey(left: string, right: string): string {
  return left < right ? `${left}|${right}` : `${right}|${left}`;
}

function clusterFullyCoveredBy(
  shorter: TrajectoryWindowClusterV1,
  longer: TrajectoryWindowClusterV1
): boolean {
  if (longer.scale <= shorter.scale) return false;
  const longerEpisodes = new Set(longer.supportEpisodeIds);
  if (!shorter.supportEpisodeIds.every((episodeId) => longerEpisodes.has(episodeId))) return false;
  return shorter.members.every((member) => longer.members.some((candidate) =>
    candidate.occurrence.episodeId === member.occurrence.episodeId &&
    candidate.occurrence.startStepIndex <= member.occurrence.startStepIndex &&
    candidate.occurrence.endStepIndex >= member.occurrence.endStepIndex));
}

function parsePolicyDecision(
  result: PolicyDraftResult,
  cluster: TrajectoryWindowClusterV1,
  context: {
    refAliases: ReadonlyMap<string, EvidenceAlias>;
    occurrenceEpisodes: ReadonlyMap<string, string>;
    model?: string;
  }
): MultiScaleWindowPolicyDecisionV1 {
  const allowed = [
    "admit", "rejection_reason", "title", "goal_pattern", "trigger_conditions",
    "procedure_steps", "verification_steps", "do_not_apply_when",
    "evidence_occurrence_ids", "confidence"
  ];
  const actual = Object.keys(result).sort();
  if (actual.join("|") !== [...allowed].sort().join("|")) {
    throw new Error(`Policy decision keys must be exactly ${allowed.join(", ")}`);
  }
  if (typeof result.admit !== "boolean") throw new Error("admit must be boolean");
  if (!result.admit) {
    return {
      clusterId: cluster.id,
      admitted: false,
      rejectionReason: requiredText(result.rejection_reason, "rejection_reason", 2_000)
    };
  }
  const evidenceOccurrenceAliases = stringArray(
    result.evidence_occurrence_ids,
    "evidence_occurrence_ids",
    2,
    MAX_POLICY_EVIDENCE_OCCURRENCES
  );
  const episodes = new Set<string>();
  const evidenceOccurrenceIds: string[] = [];
  for (const alias of evidenceOccurrenceAliases) {
    const episodeId = context.occurrenceEpisodes.get(alias);
    const occurrenceId = context.refAliases.get(alias)?.sourceId;
    if (!episodeId || !occurrenceId) throw new Error(`unknown evidence occurrence: ${alias}`);
    episodes.add(episodeId);
    evidenceOccurrenceIds.push(occurrenceId);
  }
  if (episodes.size < 2) throw new Error("Policy evidence must span two Episodes");
  const draft = {
    clusterId: cluster.id,
    scale: cluster.scale,
    title: requiredText(result.title, "title", 200),
    goalPattern: requiredText(result.goal_pattern, "goal_pattern", 1_000),
    triggerConditions: stringArray(result.trigger_conditions, "trigger_conditions", 1, 10),
    procedureSteps: groundedProcedure(result.procedure_steps, context.refAliases),
    verificationSteps: groundedVerification(result.verification_steps, context.refAliases),
    doNotApplyWhen: stringArray(result.do_not_apply_when, "do_not_apply_when", 0, 10),
    evidenceOccurrenceIds: [...new Set(evidenceOccurrenceIds)],
    supportEpisodeIds: [...episodes].sort(),
    confidence: confidence(result.confidence),
    ...(context.model ? { model: context.model } : {}),
    promptVersion: MULTI_SCALE_WINDOW_POLICY_PROMPT_VERSION
  };
  const id = `multi_scale_window_policy_${stableHash({
    version: MULTI_SCALE_WINDOW_POLICY_VERSION,
    ...draft
  }).slice(0, 24)}`;
  return { clusterId: cluster.id, admitted: true, policy: { id, ...draft } };
}

function parseSkillDecision(
  result: SkillDraftResult,
  candidate: MultiScaleWindowSkillCandidateV1,
  model?: string
): MultiScaleWindowSkillDecisionV1 {
  const allowed = [
    "admit", "rejection_reason", "name", "purpose", "trigger_conditions",
    "procedure_steps", "verification_steps", "do_not_apply_when",
    "source_policy_ids", "evidence_episode_ids", "confidence"
  ];
  const actual = Object.keys(result).sort();
  if (actual.join("|") !== [...allowed].sort().join("|")) {
    throw new Error(`Skill decision keys must be exactly ${allowed.join(", ")}`);
  }
  if (typeof result.admit !== "boolean") throw new Error("admit must be boolean");
  if (!result.admit) {
    return {
      candidateId: candidate.id,
      admitted: false,
      rejectionReason: requiredText(result.rejection_reason, "rejection_reason", 2_000)
    };
  }
  const allowedPolicyIds = new Set(candidate.policyIds);
  const sourcePolicyIds = groundedIds(
    result.source_policy_ids,
    "source_policy_ids",
    allowedPolicyIds,
    2,
    candidate.policyIds.length
  );
  if (new Set(sourcePolicyIds).size < 2) {
    throw new Error("Skill must cite at least two distinct source Policies");
  }
  const allowedEpisodeIds = new Set(candidate.supportEpisodeIds);
  const evidenceEpisodeIds = groundedIds(
    result.evidence_episode_ids,
    "evidence_episode_ids",
    allowedEpisodeIds,
    2,
    candidate.supportEpisodeIds.length
  );
  if (new Set(evidenceEpisodeIds).size < 2) {
    throw new Error("Skill evidence must span at least two distinct Episodes");
  }
  const draft = {
    candidateId: candidate.id,
    name: requiredText(result.name, "name", 200),
    purpose: requiredText(result.purpose, "purpose", 1_500),
    triggerConditions: stringArray(result.trigger_conditions, "trigger_conditions", 1, 12),
    procedureSteps: groundedSkillProcedure(result.procedure_steps, allowedPolicyIds),
    verificationSteps: groundedSkillVerification(result.verification_steps, allowedPolicyIds),
    doNotApplyWhen: stringArray(result.do_not_apply_when, "do_not_apply_when", 0, 12),
    sourcePolicyIds: [...new Set(sourcePolicyIds)],
    evidenceEpisodeIds: [...new Set(evidenceEpisodeIds)].sort(),
    confidence: confidence(result.confidence),
    ...(model ? { model } : {}),
    promptVersion: MULTI_SCALE_WINDOW_SKILL_PROMPT_VERSION
  };
  const id = `multi_scale_window_skill_${stableHash({
    version: MULTI_SCALE_WINDOW_POLICY_VERSION,
    ...draft
  }).slice(0, 24)}`;
  return { candidateId: candidate.id, admitted: true, skill: { id, ...draft } };
}

function groundedSkillProcedure(
  value: unknown,
  allowedPolicyIds: ReadonlySet<string>
): MultiScaleWindowSkillV1["procedureSteps"] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 16) {
    throw new Error("Skill procedure_steps must contain 2-16 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Skill procedure_steps[${index}] must be an object`);
    return {
      instruction: requiredText(
        item.instruction,
        `Skill procedure_steps[${index}].instruction`,
        2_000
      ),
      sourcePolicyIds: groundedIds(
        item.source_policy_ids,
        `Skill procedure_steps[${index}].source_policy_ids`,
        allowedPolicyIds,
        1,
        allowedPolicyIds.size
      )
    };
  });
}

function groundedSkillVerification(
  value: unknown,
  allowedPolicyIds: ReadonlySet<string>
): MultiScaleWindowSkillV1["verificationSteps"] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 12) {
    throw new Error("Skill verification_steps must contain 1-12 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Skill verification_steps[${index}] must be an object`);
    return {
      check: requiredText(item.check, `Skill verification_steps[${index}].check`, 1_500),
      successSignal: requiredText(
        item.success_signal,
        `Skill verification_steps[${index}].success_signal`,
        1_500
      ),
      sourcePolicyIds: groundedIds(
        item.source_policy_ids,
        `Skill verification_steps[${index}].source_policy_ids`,
        allowedPolicyIds,
        1,
        allowedPolicyIds.size
      )
    };
  });
}

function groundedIds(
  value: unknown,
  field: string,
  allowed: ReadonlySet<string>,
  min: number,
  max: number
): string[] {
  const ids = stringArray(value, field, min, max);
  for (const id of ids) {
    if (!allowed.has(id)) throw new Error(`${field} contains unknown id: ${id}`);
  }
  return ids;
}

function groundedProcedure(
  value: unknown,
  refAliases: ReadonlyMap<string, EvidenceAlias>
): MultiScaleWindowPolicyV1["procedureSteps"] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 12) {
    throw new Error("procedure_steps must contain 1-12 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`procedure_steps[${index}] must be an object`);
    return {
      instruction: requiredText(item.instruction, `procedure_steps[${index}].instruction`, 2_000),
      evidenceRefs: groundedRefs(item.evidence_refs, refAliases)
    };
  });
}

function groundedVerification(
  value: unknown,
  refAliases: ReadonlyMap<string, EvidenceAlias>
): MultiScaleWindowPolicyV1["verificationSteps"] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 10) {
    throw new Error("verification_steps must contain 1-10 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`verification_steps[${index}] must be an object`);
    const refs = evidenceAliases(item.evidence_refs, refAliases);
    const successEpisodes = new Set(refs.filter((ref) => ref.evidence.outcome === "success")
      .map((ref) => ref.evidence.episodeId));
    if (successEpisodes.size < 2) {
      throw new Error(
        `verification_steps[${index}] must cite successful Step evidence from two Episodes`
      );
    }
    return {
      check: requiredText(item.check, `verification_steps[${index}].check`, 1_500),
      successSignal: requiredText(
        item.success_signal,
        `verification_steps[${index}].success_signal`,
        1_500
      ),
      evidenceRefs: [...new Set(refs.map((ref) => ref.evidence.sourceId))]
    };
  });
}

function groundedRefs(value: unknown, refAliases: ReadonlyMap<string, EvidenceAlias>): string[] {
  return [...new Set(evidenceAliases(value, refAliases).map((ref) => ref.evidence.sourceId))];
}

function evidenceAliases(
  value: unknown,
  refAliases: ReadonlyMap<string, EvidenceAlias>
): Array<{ alias: string; evidence: EvidenceAlias }> {
  return stringArray(value, "evidence_refs", 1, 24).map((alias) => {
    const evidence = refAliases.get(alias);
    if (!evidence) throw new Error(`invented evidence ref: ${alias}`);
    return { alias, evidence };
  });
}

function requiredStepVector(
  stepVectorById: ReadonlyMap<string, readonly number[]>,
  stepId: string,
  channel = ""
): readonly number[] {
  const vector = stepVectorById.get(stepId);
  if (!vector) throw new Error(`missing${channel ? ` ${channel}` : ""} Step embedding: ${stepId}`);
  return vector;
}

function unitVector(vector: readonly number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return norm === 0 ? vector.map(() => 0) : vector.map((value) => value / norm);
}

function stringArray(
  value: unknown,
  field: string,
  min: number,
  max: number
): string[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) {
    throw new Error(`${field} must contain ${min}-${max} items`);
  }
  return value.map((item, index) => requiredText(item, `${field}[${index}]`, 2_000));
}

function requiredText(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${field} must be non-empty`);
  const text = value.trim();
  if (text.length > maxLength) throw new Error(`${field} exceeds ${maxLength} characters`);
  return text;
}

function confidence(value: unknown): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) throw new Error("confidence must be numeric");
  return Math.max(0, Math.min(1, numeric));
}

function validateSpecs(specs: readonly MultiScaleWindowSpec[]): MultiScaleWindowSpec[] {
  if (specs.length === 0) throw new Error("at least one window spec is required");
  const normalized = specs.map((spec) => ({ length: spec.length, stride: spec.stride }));
  for (const spec of normalized) {
    if (!Number.isInteger(spec.length) || spec.length < 2) {
      throw new Error("window length must be an integer >= 2");
    }
    if (!Number.isInteger(spec.stride) || spec.stride < 1 || spec.stride > spec.length) {
      throw new Error("window stride must be an integer in [1, length]");
    }
  }
  return [...new Map(normalized.map((spec) => [`${spec.length}:${spec.stride}`, spec])).values()]
    .sort((left, right) => left.length - right.length || left.stride - right.stride);
}

function validateFineMatchConfigs(
  configs: readonly BandedMonotonicMatchConfig[],
  specs: readonly MultiScaleWindowSpec[]
): BandedMonotonicMatchConfig[] {
  const byScale = new Map<number, BandedMonotonicMatchConfig>();
  for (const config of configs) {
    if (byScale.has(config.scale)) {
      throw new Error(`duplicate fine match config for scale ${config.scale}`);
    }
    if (!Number.isInteger(config.scale) || config.scale < 2 ||
        !Number.isInteger(config.bandWidth) || config.bandWidth < 0 ||
        config.bandWidth >= config.scale ||
        !Number.isInteger(config.minMatchedSteps) || config.minMatchedSteps < 1 ||
        config.minMatchedSteps > config.scale ||
        !Number.isInteger(config.maxInternalGap) || config.maxInternalGap < 0) {
      throw new Error(`invalid fine match integer configuration for scale ${config.scale}`);
    }
    for (const [field, value] of Object.entries({
      minStepSimilarity: config.minStepSimilarity,
      minCoverage: config.minCoverage,
      minAverageMatchSimilarity: config.minAverageMatchSimilarity,
      gapPenalty: config.gapPenalty,
      minAlignmentScore: config.minAlignmentScore
    })) {
      if (!(value >= 0 && value <= 1)) {
        throw new Error(`fine match ${field} must be in [0, 1] for scale ${config.scale}`);
      }
    }
    byScale.set(config.scale, { ...config });
  }
  const requiredScales = [...new Set(specs.map((spec) => spec.length))].sort((a, b) => a - b);
  for (const scale of requiredScales) {
    if (!byScale.has(scale)) throw new Error(`missing fine match config for scale ${scale}`);
  }
  return requiredScales.map((scale) => byScale.get(scale)!);
}

async function embedInBatches(
  embedder: Embedder,
  texts: readonly string[],
  batchSize: number
): Promise<number[][]> {
  const vectors: number[][] = [];
  for (let start = 0; start < texts.length; start += batchSize) {
    vectors.push(...await embedder.embed(texts.slice(start, start + batchSize), "document"));
  }
  return vectors;
}

async function mapConcurrent<T, R>(
  values: readonly T[],
  concurrency: number,
  worker: (value: T) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  await Promise.all(Array.from({ length: Math.min(Math.max(1, concurrency), values.length) }, async () => {
    while (true) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= values.length) return;
      results[index] = await worker(values[index]!);
    }
  }));
  return results;
}

function selectDistinctEpisodeMembers(
  members: readonly TrajectoryWindowClusterMemberV1[],
  limit: number
): TrajectoryWindowClusterMemberV1[] {
  const selected: TrajectoryWindowClusterMemberV1[] = [];
  const episodes = new Set<string>();
  for (const member of [...members].sort((left, right) =>
    right.similarityToCenter - left.similarityToCenter)) {
    if (episodes.has(member.occurrence.episodeId)) continue;
    selected.push(member);
    episodes.add(member.occurrence.episodeId);
    if (selected.length >= limit) break;
  }
  return selected;
}

function rangesOverlap(
  leftStart: number,
  leftEnd: number,
  rightStart: number,
  rightEnd: number
): boolean {
  return leftStart <= rightEnd && rightStart <= leftEnd;
}

function average(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function publicModel(config: {
  provider: string;
  model?: string;
  endpoint?: string;
}): Record<string, string> {
  return {
    provider: config.provider,
    ...(config.model ? { model: config.model } : {}),
    ...(config.endpoint ? { endpoint: config.endpoint } : {})
  };
}
