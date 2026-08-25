import type { LlmClient, LlmMessage } from "../../model/types.js";
import type {
  EvolutionJobRecord,
  Repositories,
  SessionRecord
} from "../../storage/repositories.js";
import type { ProceduralSpanOccurrenceRecord } from "../../storage/procedural-path-repository.js";
import type {
  ProceduralSpanClusterMemberRecord,
  ProceduralSpanEvidenceRole
} from "../../storage/procedural-span-cluster-repository.js";
import type { ProceduralPolicyVersionRecord } from "../../storage/procedural-policy-repository.js";
import type { ProceduralSpanCreditRecord } from "../../storage/procedural-span-credit-repository.js";
import type { MemoryRow } from "../../types.js";
import { stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { clip } from "../../utils/text.js";
import {
  PROCEDURAL_POLICY_INDUCTION_VERSION,
  buildProceduralPolicy,
  type ProceduralPolicyDraftV1,
  type ProceduralPolicyEvidenceStepV1,
  type ProceduralPolicyRecoveryRuleV1,
  type ProceduralPolicyVerificationStepV1
} from "./procedural-policy-model.js";
import type { EpisodeProceduralPathV2, ExecutionStepV1 } from "./procedural-path-model.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import { enqueueEpisodePolicyProjection } from "./episode-policy-projection.js";

export const PROCEDURAL_POLICY_INDUCTION_OPERATION = "procedural_policy.induction.v1" as const;
const MAX_POLICY_REPAIR_ATTEMPTS = 2;
const MAX_EVIDENCE_OCCURRENCES = 10;
const MAX_COUNTEREXAMPLES = 2;
const MAX_STEPS_PER_OCCURRENCE = 24;

export const PROCEDURAL_POLICY_INDUCTION_PROMPT = `You induce one atomic reusable Policy from a cluster of concrete procedural Span occurrences.

A Policy is a local strategy for moving from a recognizable entry condition toward a local postcondition. It is not an Episode-wide SOP and not a Skill. Infer it from the original State/Step/State evidence supplied for each occurrence, not from a prewritten Policy label.

Rules:
- Generalize only behavior supported by at least two distinct support Episodes.
- Preserve method-defining details such as action/tool family, debugging loop, retry/recovery pattern, and verification signal.
- Remove one-off file names, customer names, paths, and deliverable titles unless they are necessary applicability constraints.
- Treat counterexample occurrences as boundary or failure evidence, never as positive support.
- Separate the stable procedure from optional recovery rules.
- Every procedure and verification item must cite evidence_refs copied exactly from the input occurrence_id or step_id values.
- Only selected_path_evidence IDs are citable; all_occurrence_summaries_non_citable is coverage metadata.
- evidence_occurrence_ids must contain positive support occurrences only and span at least two distinct Episodes.
- Do not invent tools, state changes, checks, outcomes, or applicability claims.
- Return JSON only, with exactly the following shape and keys:
{
  "title": "...",
  "goal_pattern": "...",
  "trigger_conditions": ["..."],
  "procedure_steps": [
    {"instruction": "...", "evidence_refs": ["occurrence_...", "step_..."]}
  ],
  "recovery_rules": [
    {"condition": "...", "action": "...", "evidence_refs": ["occurrence_...", "step_..."]}
  ],
  "verification_steps": [
    {"check": "...", "success_signal": "...", "evidence_refs": ["occurrence_...", "step_..."]}
  ],
  "apply_when": ["..."],
  "do_not_apply_when": ["..."],
  "invariants": ["..."],
  "expected_effect": "...",
  "evidence_occurrence_ids": ["occurrence_..."],
  "confidence": 0.0
}`;

export interface ProceduralPolicyInductionDeps {
  repos: Repositories;
  skillLlm: LlmClient;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  upsertEvolutionMemory(memory: MemoryRow): {
    memory: MemoryRow;
    created: boolean;
    previous?: MemoryRow;
  };
  enqueueChange(input: {
    memoryId: string;
    namespaceId?: string;
    kind: string;
    op: string;
    entityId: string;
    userId: string;
    changeType: string;
    before?: unknown;
    after: unknown;
    source: string;
    createdAt: string;
  }): void;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
  enableThinking?: boolean;
}

interface PolicyLlmResult extends Record<string, unknown> {
  title?: unknown;
  goal_pattern?: unknown;
  trigger_conditions?: unknown;
  procedure_steps?: unknown;
  recovery_rules?: unknown;
  verification_steps?: unknown;
  apply_when?: unknown;
  do_not_apply_when?: unknown;
  invariants?: unknown;
  expected_effect?: unknown;
  evidence_occurrence_ids?: unknown;
  confidence?: unknown;
}

interface LoadedOccurrence {
  member: ProceduralSpanClusterMemberRecord;
  occurrence: ProceduralSpanOccurrenceRecord;
  path: EpisodeProceduralPathV2;
  session: SessionRecord;
  credit?: ProceduralSpanCreditRecord;
}

export class ProceduralPolicyInductionPipeline {
  constructor(private readonly deps: ProceduralPolicyInductionDeps) {}

  async induce(job: EvolutionJobRecord): Promise<ProceduralPolicyVersionRecord | undefined> {
    const clusterId = text(job.payload.proceduralClusterId);
    const membershipVersion = text(job.payload.membershipVersion);
    if (!clusterId || !membershipVersion) {
      throw new Error(`procedural Policy induction missing cluster payload: ${job.id}`);
    }
    const cluster = this.deps.repos.proceduralSpanClusters.get(clusterId);
    if (!cluster) throw new Error(`procedural Span cluster not found: ${clusterId}`);
    if (cluster.membershipVersion !== membershipVersion) return undefined;
    if (cluster.status === "forming" || cluster.status === "stale") return undefined;
    const existing = this.deps.repos.proceduralPolicies.getByClusterMembership(
      cluster.id,
      cluster.membershipVersion,
      PROCEDURAL_POLICY_INDUCTION_VERSION
    );
    if (existing) {
      const activated = existing.status === "active" && cluster.activePolicyVersionId === existing.id
        ? existing
        : this.deps.repos.proceduralPolicies.activateVersion(existing.id, job.updatedAt);
      this.enqueueProjectionRefresh(activated, job.updatedAt);
      return activated;
    }
    if (!this.deps.skillLlm.isConfigured()) {
      throw new Error("procedural Policy induction requires a configured LLM");
    }
    const loaded = loadClusterOccurrences(
      this.deps.repos,
      cluster.id,
      cluster.clusterBasis.creditGoverned === true
    );
    const supportEpisodes = new Set(loaded
      .filter((item) => item.member.evidenceRole === "support")
      .map((item) => item.occurrence.episodeId));
    if (supportEpisodes.size < 2) {
      throw new Error("procedural Policy induction requires support from two distinct Episodes");
    }
    const selected = selectEvidenceOccurrences(loaded);
    const evidenceContext = buildEvidenceContext(selected);
    const messages: LlmMessage[] = [
      { role: "system", content: PROCEDURAL_POLICY_INDUCTION_PROMPT },
      {
        role: "user",
        content: stableStringify({
          cluster: {
            id: cluster.id,
            namespace_id: cluster.namespaceId,
            membership_version: cluster.membershipVersion,
            member_count: cluster.memberCount,
            distinct_support_episode_count: cluster.distinctSupportEpisodeCount,
            cluster_basis: cluster.clusterBasis
          },
          all_occurrence_summaries_non_citable: loaded.map(compactOccurrenceSummary),
          selected_path_evidence: evidenceContext.payload
        })
      }
    ];
    const options = {
      operation: PROCEDURAL_POLICY_INDUCTION_OPERATION,
      thinkingMode: this.deps.enableThinking ? "enabled" as const : "disabled" as const,
      temperature: 0.2,
      maxTokens: 6_000,
      jsonMode: true
    };
    let result = await this.deps.skillLlm.completeJson<PolicyLlmResult>(messages, options);
    let draft: ProceduralPolicyDraftV1;
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        draft = parsePolicyDraft(result, evidenceContext);
        break;
      } catch (error) {
        if (repairAttempt >= MAX_POLICY_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.skillLlm.completeJson<PolicyLlmResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: policyRepairInstruction(error, repairNumber) }
        ], {
          ...options,
          operation: `${PROCEDURAL_POLICY_INDUCTION_OPERATION}.repair.${repairNumber}`
        });
      }
    }

    const policy = buildProceduralPolicy({
      namespaceId: cluster.namespaceId,
      clusterId: cluster.id,
      clusterMembershipVersion: cluster.membershipVersion,
      draft,
      occurrenceIds: loaded.map((item) => item.occurrence.id),
      supportOccurrenceIds: loaded
        .filter((item) => item.member.evidenceRole === "support")
        .map((item) => item.occurrence.id),
      counterexampleOccurrenceIds: loaded
        .filter((item) => item.member.evidenceRole === "counterexample")
        .map((item) => item.occurrence.id),
      supportEpisodeIds: loaded
        .filter((item) => item.member.evidenceRole === "support")
        .map((item) => item.occurrence.episodeId),
      counterexampleEpisodeIds: loaded
        .filter((item) => item.member.evidenceRole === "counterexample")
        .map((item) => item.occurrence.episodeId),
      pathIds: loaded.map((item) => item.occurrence.pathId),
      spanIds: loaded.map((item) => item.occurrence.spanId),
      sessionIds: loaded.map((item) => item.session.id),
      model: this.deps.skillLlm.config.model
    });
    const scope = commonScope(loaded);
    const memory = this.deps.buildMemory({
      userId: scope.userId,
      layer: "L2",
      kind: "policy",
      lifecycleStatus: "candidate",
      memoryType: "LongTermMemory",
      key: policy.policyKey,
      value: renderPolicyMarkdown(policy),
      tags: ["policy", "procedural", "span-cluster", "shadow"],
      info: {
        title: policy.title,
        support: policy.evidence.supportOccurrenceIds.length,
        policy_confidence: policy.confidence,
        status: "candidate",
        procedural_policy_version_id: policy.id,
        procedural_cluster_id: cluster.id,
        cluster_membership_version: cluster.membershipVersion,
        source_occurrence_ids: policy.evidence.occurrenceIds,
        source_episode_ids: unique(policy.evidence.supportEpisodeIds)
      },
      internal: {
        source: "worker.procedural_policy_induction.v1",
        plugin_algorithm: PROCEDURAL_POLICY_INDUCTION_VERSION,
        procedural_policy: policy,
        source_occurrence_ids: policy.evidence.occurrenceIds,
        source_span_ids: policy.evidence.spanIds,
        source_path_ids: policy.evidence.pathIds,
        source_episode_ids: unique([
          ...policy.evidence.supportEpisodeIds,
          ...policy.evidence.counterexampleEpisodeIds
        ]),
        source_session_ids: policy.evidence.sessionIds,
        policy: {
          title: policy.title,
          trigger: [policy.goalPattern, ...policy.triggerConditions].join("\n"),
          procedure: policy.procedureSteps.map((step) => step.instruction).join("\n"),
          verification: policy.verificationSteps
            .map((step) => `${step.check}: ${step.successSignal}`).join("\n"),
          boundary: [
            ...policy.applyWhen.map((item) => `Apply: ${item}`),
            ...policy.doNotApplyWhen.map((item) => `Do not apply: ${item}`)
          ].join("\n"),
          support: policy.evidence.supportOccurrenceIds.length,
          gain: 0,
          raw_gain: 0,
          policy_confidence: policy.confidence,
          status: "candidate",
          experience_type: "success_pattern",
          evidence_polarity: "positive",
          skill_eligible: false,
          induction_version: PROCEDURAL_POLICY_INDUCTION_VERSION,
          procedural_policy_version_id: policy.id,
          procedural_cluster_id: cluster.id,
          cluster_membership_version: cluster.membershipVersion,
          source_occurrence_ids: policy.evidence.occurrenceIds,
          source_span_ids: policy.evidence.spanIds,
          source_path_ids: policy.evidence.pathIds,
          source_episode_ids: policy.evidence.supportEpisodeIds
        }
      },
      createdAt: job.updatedAt
    });
    let upsert: ReturnType<ProceduralPolicyInductionDeps["upsertEvolutionMemory"]>;
    let saved: ProceduralPolicyVersionRecord;
    this.deps.repos.transaction(() => {
      upsert = this.deps.upsertEvolutionMemory(memory);
      saved = this.deps.repos.proceduralPolicies.saveAndActivate({
        policy,
        l2MemoryId: upsert.memory.id,
        occurrences: loaded.map((item) => ({
          occurrenceId: item.occurrence.id,
          pathId: item.occurrence.pathId,
          spanId: item.occurrence.spanId,
          episodeId: item.occurrence.episodeId,
          sessionId: item.session.id,
          evidenceRole: item.member.evidenceRole,
          matchConfidence: item.member.similarity
        })),
        at: job.updatedAt
      });
    });
    this.deps.enqueueChange({
      memoryId: upsert!.memory.id,
      namespaceId: cluster.namespaceId,
      kind: "policy",
      op: upsert!.created ? "created" : "updated",
      entityId: upsert!.memory.id,
      userId: upsert!.memory.userId,
      changeType: upsert!.created ? "create" : "update",
      before: upsert!.previous,
      after: upsert!.memory,
      source: "worker.procedural_policy_induction.v1",
      createdAt: job.updatedAt
    });
    this.enqueueProjectionRefresh(saved!, job.updatedAt);
    return saved!;
  }

  private enqueueProjectionRefresh(policy: ProceduralPolicyVersionRecord, at: string): void {
    const episodeIds = [...new Set(this.deps.repos.proceduralPolicies
      .listOccurrences(policy.id)
      .filter((occurrence) => occurrence.status === "active")
      .map((occurrence) => occurrence.episodeId))].sort();
    for (const episodeId of episodeIds) {
      enqueueEpisodePolicyProjection({
        repos: this.deps.repos,
        enqueueJob: this.deps.enqueueJob
      }, {
        episodeId,
        at,
        trigger: "procedural_policy_activated",
        policyVersionId: policy.id
      });
    }
  }
}

function loadClusterOccurrences(
  repos: Repositories,
  clusterId: string,
  creditGoverned: boolean
): LoadedOccurrence[] {
  return repos.proceduralSpanClusters.listMembers(clusterId).map((member) => {
    const occurrence = repos.proceduralPaths.getOccurrence(member.occurrenceId);
    if (!occurrence) throw new Error(`procedural Span occurrence not found: ${member.occurrenceId}`);
    const pathRecord = repos.proceduralPaths.get(occurrence.pathId);
    if (!pathRecord || pathRecord.status !== "active") {
      throw new Error(`procedural Span occurrence path is not active: ${occurrence.id}`);
    }
    const session = repos.runtime.getSession(pathRecord.sessionId);
    if (!session || session.userId !== pathRecord.userId) {
      throw new Error(`procedural Policy evidence Session is missing: ${pathRecord.sessionId}`);
    }
    const credit = repos.proceduralSpanCredits.getActiveCreditForOccurrence(occurrence.id);
    if (creditGoverned && (!credit || credit.evidenceRole !== member.evidenceRole)) {
      throw new Error(`procedural Policy evidence role does not match active SpanCredit: ${occurrence.id}`);
    }
    return { member, occurrence, path: pathRecord.path, session, ...(credit ? { credit } : {}) };
  });
}

function selectEvidenceOccurrences(loaded: LoadedOccurrence[]): LoadedOccurrence[] {
  const support = loaded
    .filter((item) => item.member.evidenceRole === "support")
    .sort(compareEvidence);
  const counterexamples = loaded
    .filter((item) => item.member.evidenceRole === "counterexample")
    .sort(compareEvidence)
    .slice(0, MAX_COUNTEREXAMPLES);
  const supportLimit = MAX_EVIDENCE_OCCURRENCES - counterexamples.length;
  const selected: LoadedOccurrence[] = [];
  const selectedIds = new Set<string>();
  const seenEpisodes = new Set<string>();
  const seenStructures = new Set<string>();
  for (const item of support) {
    if (selected.length >= supportLimit) break;
    if (seenEpisodes.has(item.occurrence.episodeId)) continue;
    selected.push(item);
    selectedIds.add(item.occurrence.id);
    seenEpisodes.add(item.occurrence.episodeId);
    seenStructures.add(item.occurrence.projection.structureSignature);
  }
  for (const item of support) {
    if (selected.length >= supportLimit) break;
    if (selectedIds.has(item.occurrence.id) || seenStructures.has(item.occurrence.projection.structureSignature)) continue;
    selected.push(item);
    selectedIds.add(item.occurrence.id);
    seenStructures.add(item.occurrence.projection.structureSignature);
  }
  for (const item of support) {
    if (selected.length >= supportLimit) break;
    if (selectedIds.has(item.occurrence.id)) continue;
    selected.push(item);
    selectedIds.add(item.occurrence.id);
  }
  return [...selected, ...counterexamples]
    .sort((left, right) => left.occurrence.id.localeCompare(right.occurrence.id));
}

function compareEvidence(left: LoadedOccurrence, right: LoadedOccurrence): number {
  return creditEvidenceStrength(right) - creditEvidenceStrength(left) ||
    right.member.similarity - left.member.similarity ||
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.id.localeCompare(right.occurrence.id);
}

function creditEvidenceStrength(item: LoadedOccurrence): number {
  if (!item.credit) return 0;
  return item.member.evidenceRole === "support"
    ? item.credit.rewardCredit
    : -item.credit.rewardCredit;
}

function buildEvidenceContext(selected: LoadedOccurrence[]): {
  payload: unknown[];
  validRefs: Set<string>;
  supportOccurrenceIds: Set<string>;
  occurrenceEpisode: Map<string, string>;
  refOccurrence: Map<string, string>;
} {
  const validRefs = new Set<string>();
  const supportOccurrenceIds = new Set<string>();
  const occurrenceEpisode = new Map<string, string>();
  const refOccurrence = new Map<string, string>();
  const payload = selected.map((item) => {
    validRefs.add(item.occurrence.id);
    refOccurrence.set(item.occurrence.id, item.occurrence.id);
    occurrenceEpisode.set(item.occurrence.id, item.occurrence.episodeId);
    if (item.member.evidenceRole === "support") supportOccurrenceIds.add(item.occurrence.id);
    const stateById = new Map(item.path.states.map((state) => [state.id, state]));
    const stepById = new Map(item.path.steps.map((step) => [step.id, step]));
    const steps = selectSalientSteps(item.occurrence.stepIds.map((stepId) => stepById.get(stepId)!));
    for (const step of steps) {
      validRefs.add(step.id);
      refOccurrence.set(step.id, item.occurrence.id);
    }
    return {
      occurrence_id: item.occurrence.id,
      evidence_role: item.member.evidenceRole,
      match_similarity: item.member.similarity,
      episode_id: item.occurrence.episodeId,
      path_id: item.occurrence.pathId,
      span_id: item.occurrence.spanId,
      local_goal: item.occurrence.localGoal,
      capability_goal: item.occurrence.capabilityGoal,
      entry_condition: item.occurrence.entryCondition,
      exit_condition: item.occurrence.exitCondition,
      termination_status: item.occurrence.terminationStatus,
      span_credit: item.credit ? compactSpanCredit(item.credit) : undefined,
      cost: item.occurrence.span.cost,
      pre_state: clip(stateById.get(item.occurrence.preStateId)?.summary ?? "", 1_000),
      post_state: clip(stateById.get(item.occurrence.postStateId)?.summary ?? "", 1_000),
      projected_procedure: clip(item.occurrence.projection.procedureText, 4_000),
      steps: steps.map((step) => compactStep(step, stateById))
    };
  });
  return { payload, validRefs, supportOccurrenceIds, occurrenceEpisode, refOccurrence };
}

function compactOccurrenceSummary(item: LoadedOccurrence): Record<string, unknown> {
  return {
    occurrence_id: item.occurrence.id,
    episode_id: item.occurrence.episodeId,
    evidence_role: item.member.evidenceRole,
    match_similarity: item.member.similarity,
    local_goal: clip(item.occurrence.localGoal, 500),
    capability_goal: clip(item.occurrence.capabilityGoal, 500),
    entry_condition: clip(item.occurrence.entryCondition, 500),
    exit_condition: clip(item.occurrence.exitCondition, 500),
    termination_status: item.occurrence.terminationStatus,
    span_credit: item.credit ? compactSpanCredit(item.credit) : undefined,
    structure_signature: item.occurrence.projection.structureSignature,
    cost: item.occurrence.span.cost
  };
}

function compactSpanCredit(credit: ProceduralSpanCreditRecord): Record<string, unknown> {
  return {
    reward_credit: credit.rewardCredit,
    attribution_type: credit.attributionType,
    confidence: credit.confidence,
    evidence_role: credit.evidenceRole,
    reason: clip(credit.reason, 500)
  };
}

function compactStep(
  step: ExecutionStepV1,
  stateById: ReadonlyMap<string, { summary: string }>
): Record<string, unknown> {
  return {
    step_id: step.id,
    step_index: step.stepIndex,
    action_kind: step.action.kind,
    action_type: step.action.type,
    tool_name: step.action.toolName,
    intent: clip(step.action.intent, 500),
    summary: clip(step.action.summary, 700),
    outcome: step.outcome.status,
    retry_of_step_id: step.retryOfStepId,
    recovery_from_step_id: step.recoveryFromStepId,
    state_delta_ops: step.actionEffectDelta.map((operation) => operation.op),
    pre_state: clip(stateById.get(step.preStateId)?.summary ?? "", 500),
    post_state: clip(stateById.get(step.postStateId)?.summary ?? "", 500)
  };
}

function selectSalientSteps(steps: ExecutionStepV1[]): ExecutionStepV1[] {
  if (steps.length <= MAX_STEPS_PER_OCCURRENCE) return steps;
  const selected = new Set<number>([0, steps.length - 1]);
  for (const [index, step] of steps.entries()) {
    if (selected.size >= MAX_STEPS_PER_OCCURRENCE) break;
    if (step.outcome.status === "failure" || step.retryOfStepId || step.recoveryFromStepId) {
      selected.add(index);
    }
  }
  for (let slot = 0; selected.size < MAX_STEPS_PER_OCCURRENCE && slot < MAX_STEPS_PER_OCCURRENCE * 3; slot += 1) {
    selected.add(Math.round((slot / (MAX_STEPS_PER_OCCURRENCE * 3 - 1)) * (steps.length - 1)));
  }
  return [...selected]
    .sort((left, right) => left - right)
    .map((index) => steps[index]!);
}

function parsePolicyDraft(
  result: PolicyLlmResult,
  context: ReturnType<typeof buildEvidenceContext>
): ProceduralPolicyDraftV1 {
  assertExactKeys(result, [
    "title", "goal_pattern", "trigger_conditions", "procedure_steps", "recovery_rules",
    "verification_steps", "apply_when", "do_not_apply_when", "invariants",
    "expected_effect", "evidence_occurrence_ids", "confidence"
  ], "Policy");
  const title = requiredText(result.title, "title", 200);
  const goalPattern = requiredText(result.goal_pattern, "goal_pattern", 1_000);
  const triggerConditions = requiredStringArray(result.trigger_conditions, "trigger_conditions", 1, 12, 1_000);
  const procedureSteps = parseProcedureSteps(result.procedure_steps, context.validRefs);
  const recoveryRules = parseRecoveryRules(result.recovery_rules, context.validRefs);
  const verificationSteps = parseVerificationSteps(result.verification_steps, context.validRefs);
  const applyWhen = requiredStringArray(result.apply_when, "apply_when", 1, 12, 1_000);
  const doNotApplyWhen = requiredStringArray(result.do_not_apply_when, "do_not_apply_when", 1, 12, 1_000);
  const invariants = requiredStringArray(result.invariants, "invariants", 0, 12, 1_000);
  const expectedEffect = requiredText(result.expected_effect, "expected_effect", 2_000);
  const evidenceOccurrenceIds = requiredStringArray(
    result.evidence_occurrence_ids,
    "evidence_occurrence_ids",
    2,
    MAX_EVIDENCE_OCCURRENCES,
    200
  );
  for (const occurrenceId of evidenceOccurrenceIds) {
    if (!context.supportOccurrenceIds.has(occurrenceId)) {
      throw new Error(`Policy evidence_occurrence_ids contains non-support evidence: ${occurrenceId}`);
    }
  }
  const distinctEpisodes = new Set(evidenceOccurrenceIds.map((id) => context.occurrenceEpisode.get(id))).size;
  if (distinctEpisodes < 2) {
    throw new Error("Policy evidence_occurrence_ids must span at least two support Episodes");
  }
  const confidence = result.confidence;
  if (typeof confidence !== "number" || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new Error("Policy confidence must be a finite number in [0, 1]");
  }
  const positiveEvidenceEpisodes = new Set([
    ...procedureSteps.flatMap((step) => step.evidenceRefs),
    ...verificationSteps.flatMap((step) => step.evidenceRefs)
  ].map((ref) => context.refOccurrence.get(ref))
    .filter((id): id is string => Boolean(id && context.supportOccurrenceIds.has(id)))
    .map((id) => context.occurrenceEpisode.get(id)));
  if (positiveEvidenceEpisodes.size < 2) {
    throw new Error("Policy procedure and verification must cite support evidence from two Episodes");
  }
  return {
    title,
    goalPattern,
    triggerConditions,
    procedureSteps,
    recoveryRules,
    verificationSteps,
    applyWhen,
    doNotApplyWhen,
    invariants,
    expectedEffect,
    evidenceOccurrenceIds: unique(evidenceOccurrenceIds),
    confidence
  };
}

function parseProcedureSteps(value: unknown, validRefs: ReadonlySet<string>): ProceduralPolicyEvidenceStepV1[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 12) {
    throw new Error("Policy procedure_steps must contain 1-12 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Policy procedure_steps[${index}] must be an object`);
    assertExactKeys(item, ["instruction", "evidence_refs"], `procedure_steps[${index}]`);
    return {
      instruction: requiredText(item.instruction, `procedure_steps[${index}].instruction`, 2_000),
      evidenceRefs: evidenceRefs(item.evidence_refs, validRefs, `procedure_steps[${index}].evidence_refs`)
    };
  });
}

function parseRecoveryRules(value: unknown, validRefs: ReadonlySet<string>): ProceduralPolicyRecoveryRuleV1[] {
  if (!Array.isArray(value) || value.length > 8) {
    throw new Error("Policy recovery_rules must contain 0-8 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Policy recovery_rules[${index}] must be an object`);
    assertExactKeys(item, ["condition", "action", "evidence_refs"], `recovery_rules[${index}]`);
    return {
      condition: requiredText(item.condition, `recovery_rules[${index}].condition`, 1_500),
      action: requiredText(item.action, `recovery_rules[${index}].action`, 2_000),
      evidenceRefs: evidenceRefs(item.evidence_refs, validRefs, `recovery_rules[${index}].evidence_refs`)
    };
  });
}

function parseVerificationSteps(value: unknown, validRefs: ReadonlySet<string>): ProceduralPolicyVerificationStepV1[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 8) {
    throw new Error("Policy verification_steps must contain 1-8 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Policy verification_steps[${index}] must be an object`);
    assertExactKeys(item, ["check", "success_signal", "evidence_refs"], `verification_steps[${index}]`);
    return {
      check: requiredText(item.check, `verification_steps[${index}].check`, 1_500),
      successSignal: requiredText(item.success_signal, `verification_steps[${index}].success_signal`, 1_500),
      evidenceRefs: evidenceRefs(item.evidence_refs, validRefs, `verification_steps[${index}].evidence_refs`)
    };
  });
}

function evidenceRefs(value: unknown, validRefs: ReadonlySet<string>, field: string): string[] {
  const refs = requiredStringArray(value, field, 1, 24, 200);
  for (const ref of refs) {
    if (!validRefs.has(ref)) throw new Error(`Policy ${field} invented evidence ref: ${ref}`);
  }
  return unique(refs);
}

function requiredStringArray(
  value: unknown,
  field: string,
  min: number,
  max: number,
  maxItemLength: number
): string[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) {
    throw new Error(`Policy ${field} must contain ${min}-${max} strings`);
  }
  return value.map((item, index) => requiredText(item, `${field}[${index}]`, maxItemLength));
}

function requiredText(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`Policy ${field} must be non-empty text`);
  const normalized = value.trim();
  if (normalized.length > maxLength) throw new Error(`Policy ${field} exceeds ${maxLength} characters`);
  return normalized;
}

function assertExactKeys(record: Record<string, unknown>, allowed: string[], field: string): void {
  const expected = [...allowed].sort();
  const actual = Object.keys(record).sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`Policy ${field} keys must be exactly: ${allowed.join(", ")}`);
  }
}

function policyRepairInstruction(error: unknown, repairNumber: number): string {
  const reason = error instanceof Error ? error.message : String(error);
  const base = `The previous Policy JSON violated the contract: ${reason}. Use only occurrence_id and step_id evidence_refs copied from the original payload. Preserve the same evidence-grounded semantics.`;
  if (repairNumber === 1) return `Correct the previous JSON in full. ${base}`;
  return `Discard the previous JSON and regenerate the entire response from the original cluster evidence. ${base} Before returning, verify every key and every evidence ref.`;
}

function renderPolicyMarkdown(policy: ReturnType<typeof buildProceduralPolicy>): string {
  const lines = [
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
    "## Recovery",
    ...(policy.recoveryRules.length > 0
      ? policy.recoveryRules.map((rule) => `- If ${rule.condition}, ${rule.action}`)
      : ["- No reusable recovery rule was supported by the evidence."]),
    "",
    "## Verification",
    ...policy.verificationSteps.map((step) => `- ${step.check} → ${step.successSignal}`),
    "",
    "## Boundary",
    ...policy.applyWhen.map((item) => `- Apply when: ${item}`),
    ...policy.doNotApplyWhen.map((item) => `- Do not apply when: ${item}`),
    "",
    "## Expected Effect",
    policy.expectedEffect
  ];
  return lines.join("\n");
}

function commonScope(loaded: LoadedOccurrence[]): {
  userId: string;
} {
  const userIds = unique(loaded.map((item) => item.session.userId));
  if (userIds.length !== 1) throw new Error("procedural Policy cluster crosses users");
  return {
    userId: userIds[0]!
  };
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}
