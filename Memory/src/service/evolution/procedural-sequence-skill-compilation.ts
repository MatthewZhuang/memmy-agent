import { skillMetaFromMemory } from "../../algorithm/plugin-algorithms.js";
import type { MemmyConfig } from "../../config/index.js";
import type { LlmClient, LlmMessage } from "../../model/types.js";
import type {
  EvolutionJobRecord,
  Repositories,
  SessionRecord
} from "../../storage/repositories.js";
import type {
  PolicySequencePatternRecord,
  ProceduralSkillCandidateRecord
} from
  "../../storage/policy-sequence-pattern-repository.js";
import type { ProceduralPolicyVersionRecord } from
  "../../storage/procedural-policy-repository.js";
import type { ProceduralSpanOccurrenceRecord } from
  "../../storage/procedural-path-repository.js";
import type { MemoryRow } from "../../types.js";
import { stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { clip } from "../../utils/text.js";
import { skillBetaPosterior, skillSuccessRate } from "../read-model/skill.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import {
  hasMultipleDistinctPolicies,
  type ProceduralSkillCandidateEvidenceV1
} from "./policy-sequence-pattern-model.js";

export const PROCEDURAL_SEQUENCE_SKILL_COMPILER_VERSION =
  "procedural-sequence-skill-compiler.v2" as const;
export const PROCEDURAL_SEQUENCE_SKILL_PROMPT_VERSION =
  "procedural-sequence-skill-prompt.v2" as const;
export const PROCEDURAL_SEQUENCE_SKILL_OPERATION =
  "procedural_sequence_skill.compilation.v2" as const;

const MAX_REPAIR_ATTEMPTS = 2;
const MAX_SUPPORT_OCCURRENCES = 6;
const MAX_COUNTEREXAMPLES = 2;
const MAX_UNCERTAIN_OCCURRENCES = 1;
const MAX_STEPS_PER_SPAN = 16;

export const PROCEDURAL_SEQUENCE_SKILL_PROMPT = `You compile one reusable Skill from a repeated ordered sequence of atomic Policies.

The input contains a stable Policy backbone plus the complete local State/Span/Step path slices from multiple Episodes. A Skill is a user-invokable procedure, not a restatement of Policy titles.

Rules:
- Preserve the ordered Policy backbone. Inspect full_path and gap_spans before deciding whether intervening work is mandatory, conditional, recovery, retry, or an anti-pattern.
- Generalize only behavior supported by the supplied evidence. Do not invent tools, checks, parameters, recovery behavior, or applicability claims.
- Use the original Span and Step evidence to recover operational details; Policy summaries alone are not sufficient.
- Treat counterexamples as boundary or recovery evidence, never as positive support.
- Every precondition, procedure step, recovery rule, exclusion, and verification item must cite evidence_refs copied exactly from the input.
- Each positive procedure and verification item must be supported by at least two distinct successful Episodes.
- evidence_occurrence_ids must cite successful sequence_occurrence_id values and span at least two distinct Episodes.
- tools must be copied from observed_tools.
- name must be lowercase snake_case and no longer than 48 characters.
- Return JSON only, with exactly these keys and shape:
{
  "name": "snake_case_name",
  "display_title": "...",
  "retrieval_blurb": "...",
  "trigger_context": "...",
  "summary": "...",
  "parameters": [
    {"name": "...", "type": "string", "required": true, "description": "..."}
  ],
  "preconditions": [
    {"condition": "...", "evidence_refs": ["..."]}
  ],
  "steps": [
    {"title": "...", "body": "...", "evidence_refs": ["..."]}
  ],
  "recovery_rules": [
    {"condition": "...", "action": "...", "evidence_refs": ["..."]}
  ],
  "verification": [
    {"check": "...", "success_signal": "...", "evidence_refs": ["..."]}
  ],
  "do_not_use_when": [
    {"condition": "...", "evidence_refs": ["..."]}
  ],
  "decision_guidance": {
    "preference": ["..."],
    "anti_pattern": ["..."]
  },
  "tools": ["..."],
  "tags": ["..."],
  "evidence_occurrence_ids": ["policy_sequence_occurrence_..."],
  "confidence": 0.0
}`;

export interface ProceduralSequenceSkillCompilationDeps {
  repos: Repositories;
  config: MemmyConfig;
  skillLlm: LlmClient;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  upsertEvolutionMemory(memory: MemoryRow): {
    memory: MemoryRow;
    created: boolean;
    previous?: MemoryRow;
  };
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

interface GroundedText {
  text: string;
  evidenceRefs: string[];
}

interface GroundedStep {
  title: string;
  body: string;
  evidenceRefs: string[];
}

interface GroundedRecovery {
  condition: string;
  action: string;
  evidenceRefs: string[];
}

interface GroundedVerification {
  check: string;
  successSignal: string;
  evidenceRefs: string[];
}

interface SequenceSkillDraft {
  name: string;
  displayTitle: string;
  retrievalBlurb: string;
  triggerContext: string;
  summary: string;
  parameters: Array<{
    name: string;
    type: string;
    required: boolean;
    description: string;
  }>;
  preconditions: GroundedText[];
  steps: GroundedStep[];
  recoveryRules: GroundedRecovery[];
  verification: GroundedVerification[];
  doNotUseWhen: GroundedText[];
  decisionGuidance: {
    preference: string[];
    antiPattern: string[];
  };
  tools: string[];
  tags: string[];
  evidenceOccurrenceIds: string[];
  confidence: number;
}

interface LoadedEvidence {
  evidence: ProceduralSkillCandidateEvidenceV1;
  policies: ProceduralPolicyVersionRecord[];
  spans: ProceduralSpanOccurrenceRecord[];
  pathSpans: ProceduralSpanOccurrenceRecord[];
  session: SessionRecord;
  payload: Record<string, unknown>;
}

interface LoadedCandidateContext {
  selected: LoadedEvidence[];
  validRefs: Set<string>;
  supportOccurrenceEpisodes: Map<string, string>;
  refEpisodes: Map<string, Set<string>>;
  observedTools: Set<string>;
  sourcePolicyMemoryIds: string[];
  sessions: SessionRecord[];
  payload: Record<string, unknown>;
}

interface SequenceSkillLlmResult extends Record<string, unknown> {
  name?: unknown;
  display_title?: unknown;
  retrieval_blurb?: unknown;
  trigger_context?: unknown;
  summary?: unknown;
  parameters?: unknown;
  preconditions?: unknown;
  steps?: unknown;
  recovery_rules?: unknown;
  verification?: unknown;
  do_not_use_when?: unknown;
  decision_guidance?: unknown;
  tools?: unknown;
  tags?: unknown;
  evidence_occurrence_ids?: unknown;
  confidence?: unknown;
}

export class ProceduralSequenceSkillCompilationPipeline {
  constructor(private readonly deps: ProceduralSequenceSkillCompilationDeps) {}

  async compileCandidate(
    candidate: ProceduralSkillCandidateRecord,
    input: {
      sessionId?: string;
      episodeId?: string;
      at: string;
    }
  ): Promise<MemoryRow | undefined> {
    return this.compileJob({
      id: `trace2skill_replay_${candidate.id}`,
      jobType: "skill_crystallization",
      status: "leased",
      dedupeKey: `trace2skill_replay:${candidate.id}:${candidate.evidenceHash}`,
      userId: candidate.namespaceId,
      ...(input.sessionId ? { sessionId: input.sessionId } : {}),
      ...(input.episodeId ? { episodeId: input.episodeId } : {}),
      payload: {
        proceduralSkillCandidateId: candidate.id,
        patternId: candidate.patternId,
        evidenceHash: candidate.evidenceHash,
        compilerVersion: PROCEDURAL_SEQUENCE_SKILL_COMPILER_VERSION,
        trigger: "trace2skill_replay"
      },
      attempts: 1,
      maxAttempts: 1,
      leasedUntil: null,
      lastError: null,
      createdAt: input.at,
      updatedAt: input.at
    });
  }

  needsCompilation(candidate: ProceduralSkillCandidateRecord): boolean {
    if (candidate.status !== "active" || candidate.lifecycleStatus !== "ready" ||
        !hasMultipleDistinctPolicies(candidate.candidate.policyKeys)) return false;
    const existing = this.deps.repos.memories.getByKey(
      "Skill",
      proceduralSequenceSkillMemoryKey(candidate)
    );
    return !existing ||
      existing.status === "archived" ||
      existing.status === "deleted" ||
      text(existing.properties.internal_info.procedural_sequence_skill, "evidence_hash") !==
        candidate.evidenceHash;
  }

  retirePattern(pattern: PolicySequencePatternRecord, at: string): MemoryRow | undefined {
    const existing = this.deps.repos.memories.getByKey(
      "Skill",
      proceduralSequenceSkillMemoryKey(pattern)
    );
    if (!existing || existing.status === "archived" || existing.status === "deleted") {
      return existing;
    }
    const internal = existing.properties.internal_info;
    const skill = isRecord(internal.skill) ? internal.skill : {};
    const sequenceSkill = isRecord(internal.procedural_sequence_skill)
      ? internal.procedural_sequence_skill
      : {};
    const retired = this.deps.repos.memories.update({
      ...existing,
      status: "archived",
      properties: {
        ...existing.properties,
        status: "archived",
        internal_info: {
          ...internal,
          skill: {
            ...skill,
            status: "archived"
          },
          procedural_sequence_skill: {
            ...sequenceSkill,
            executable: false,
            retired_at: at,
            retired_reason: "pattern_not_ready"
          }
        }
      },
      updatedAt: at
    });
    this.deps.repos.runtime.appendChange({
      memoryId: retired.id,
      namespaceId: pattern.namespaceId,
      kind: "skill",
      op: "updated",
      entityId: retired.id,
      userId: retired.userId,
      changeType: "procedural_sequence_skill_retired",
      before: existing,
      after: retired,
      source: "worker.procedural_sequence_skill_compilation.v1",
      createdAt: at
    });
    return retired;
  }

  async compileJob(job: EvolutionJobRecord): Promise<MemoryRow | undefined> {
    const candidateId = payloadText(job, "proceduralSkillCandidateId");
    const patternId = payloadText(job, "patternId");
    const evidenceHash = payloadText(job, "evidenceHash");
    if (!candidateId || !patternId || !evidenceHash) {
      throw new Error(`procedural Sequence Skill compilation missing payload: ${job.id}`);
    }
    const candidate = this.deps.repos.policySequencePatterns.getActiveCandidateForPattern(patternId);
    if (!candidate || candidate.id !== candidateId || candidate.evidenceHash !== evidenceHash) {
      return undefined;
    }
    if (candidate.lifecycleStatus !== "ready") return undefined;
    const existing = this.deps.repos.memories.getByKey(
      "Skill",
      proceduralSequenceSkillMemoryKey(candidate)
    );
    if (!this.needsCompilation(candidate)) return existing;
    if (!this.deps.skillLlm.isConfigured()) {
      throw new Error("procedural Sequence Skill compilation requires a configured LLM");
    }

    const context = loadCandidateContext(this.deps.repos, candidate);
    const messages: LlmMessage[] = [
      { role: "system", content: PROCEDURAL_SEQUENCE_SKILL_PROMPT },
      { role: "user", content: stableStringify(context.payload) }
    ];
    const options = {
      operation: PROCEDURAL_SEQUENCE_SKILL_OPERATION,
      thinkingMode: this.deps.config.evolution.enableThinking
        ? "enabled" as const
        : "disabled" as const,
      temperature: 0.2,
      maxTokens: 8_000,
      jsonMode: true
    };
    let result = await this.deps.skillLlm.completeJson<SequenceSkillLlmResult>(messages, options);
    let draft: SequenceSkillDraft;
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        draft = parseSequenceSkillDraft(result, context);
        break;
      } catch (error) {
        if (repairAttempt >= MAX_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.skillLlm.completeJson<SequenceSkillLlmResult>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          { role: "user", content: repairInstruction(error, repairNumber) }
        ], {
          ...options,
          operation: `${PROCEDURAL_SEQUENCE_SKILL_OPERATION}.repair.${repairNumber}`
        });
      }
    }

    const previousMeta = existing ? skillMetaFromMemory(existing) : null;
    const support = candidate.distinctSupportEpisodeCount;
    const eta = previousMeta && previousMeta.trialsAttempted > 0
      ? previousMeta.eta
      : clamp01(Math.max(
          this.deps.config.algorithm.skill.minEtaForRetrieval,
          0.5 * draft.confidence + 0.5 * Math.min(1, support / 2)
        ));
    const trialsAttempted = previousMeta?.trialsAttempted ?? 0;
    const trialsPassed = previousMeta?.trialsPassed ?? 0;
    const successRate = skillSuccessRate(trialsAttempted, trialsPassed);
    const betaPosterior = skillBetaPosterior(trialsAttempted, trialsPassed);
    const scope = commonScope(context.sessions, candidate.namespaceId);
    const procedureJson = buildProcedureJson(candidate, draft, context, {
      support,
      successRate,
      betaPosterior
    });
    const invocationGuide = renderInvocationGuide(draft);
    const evidenceAnchorIds = unique([
      ...candidate.candidate.sourceSpanOccurrenceIds,
      ...context.selected.flatMap((item) => {
        const episode = this.deps.repos.runtime.getEpisode(item.evidence.episodeId);
        return episode?.l1MemoryIds ?? [];
      })
    ]).slice(0, 20);
    const skillMemory = this.deps.buildMemory({
      userId: scope.userId,
      layer: "Skill",
      kind: "skill",
      lifecycleStatus: "candidate",
      memoryType: "SkillMemory",
      key: proceduralSequenceSkillMemoryKey(candidate),
      value: invocationGuide,
      tags: unique([
        "skill",
        "procedural",
        "policy-sequence",
        "sequence-skill",
        candidate.candidate.capabilityType,
        ...candidate.candidate.discoverySources,
        "shadow",
        ...draft.tags
      ]),
      info: {
        name: draft.name,
        title: draft.displayTitle,
        eta,
        status: "candidate",
        support,
        source_memory_ids: context.sourcePolicyMemoryIds,
        source_candidate_id: candidate.id,
        source_pattern_id: candidate.patternId,
        source_episode_ids: candidate.candidate.supportEpisodeIds
      },
      internal: {
        source: "worker.procedural_sequence_skill_compilation.v1",
        plugin_algorithm: PROCEDURAL_SEQUENCE_SKILL_COMPILER_VERSION,
        source_memory_ids: context.sourcePolicyMemoryIds,
        source_policy_ids: context.sourcePolicyMemoryIds,
        source_procedural_policy_version_ids: unique(
          context.selected.flatMap((item) => item.policies.map((policy) => policy.id))
        ),
        source_span_occurrence_ids: candidate.candidate.sourceSpanOccurrenceIds,
        source_span_ids: candidate.candidate.sourceSpanIds,
        source_path_ids: candidate.candidate.sourcePathIds,
        source_episode_ids: unique([
          ...candidate.candidate.supportEpisodeIds,
          ...candidate.candidate.counterexampleEpisodeIds,
          ...candidate.candidate.uncertainEpisodeIds
        ]),
        evidence_anchor_ids: evidenceAnchorIds,
        name: draft.name,
        invocation_guide: invocationGuide,
        procedure_json: procedureJson,
        procedural_sequence_skill: {
          compiler_version: PROCEDURAL_SEQUENCE_SKILL_COMPILER_VERSION,
          prompt_version: PROCEDURAL_SEQUENCE_SKILL_PROMPT_VERSION,
          model: this.deps.skillLlm.config.model,
          candidate_id: candidate.id,
          pattern_id: candidate.patternId,
          pattern_membership_version: candidate.patternMembershipVersion,
          sequence_hash: candidate.candidate.sequenceHash,
          policy_keys: candidate.candidate.policyKeys,
          capability_type: candidate.candidate.capabilityType,
          discovery_sources: candidate.candidate.discoverySources,
          episode_family_id: candidate.candidate.episodeFamilyId ?? null,
          evidence_hash: candidate.evidenceHash,
          executable: true
        },
        skill: {
          name: draft.name,
          eta,
          status: "candidate",
          support,
          gain: 0,
          source_policy_ids: context.sourcePolicyMemoryIds,
          source_world_model_ids: [],
          evidence_anchor_ids: evidenceAnchorIds,
          invocation_guide: invocationGuide,
          procedure_json: procedureJson,
          trials_attempted: trialsAttempted,
          trials_passed: trialsPassed,
          success_rate: successRate,
          beta_posterior: betaPosterior,
          vec: null
        }
      },
      createdAt: job.updatedAt
    });
    const upsert = this.deps.upsertEvolutionMemory(skillMemory);
    for (const episodeId of candidate.candidate.supportEpisodeIds) {
      this.deps.repos.runtime.appendEpisodeDerivedMemory(
        episodeId,
        "Skill",
        upsert.memory.id,
        job.updatedAt
      );
    }
    this.deps.repos.runtime.appendChange({
      memoryId: upsert.memory.id,
      namespaceId: candidate.namespaceId,
      kind: "skill",
      op: upsert.created ? "created" : "updated",
      entityId: upsert.memory.id,
      userId: upsert.memory.userId,
      changeType: upsert.created ? "create" : "update",
      before: upsert.previous,
      after: upsert.memory,
      source: "worker.procedural_sequence_skill_compilation.v1",
      createdAt: job.updatedAt
    });
    if (this.deps.config.algorithm.capture.embedAfterCapture) {
      this.deps.enqueueJob({
        jobType: "embedding",
        userId: upsert.memory.userId,
        sessionId: job.sessionId,
        episodeId: job.episodeId,
        targetMemoryId: upsert.memory.id,
        payload: {
          reason: "procedural_sequence_skill.upserted",
          contentHash: upsert.memory.contentHash ?? undefined
        },
        createdAt: job.updatedAt
      });
    }
    return upsert.memory;
  }
}

export function proceduralSequenceSkillMemoryKey(
  source: Pick<PolicySequencePatternRecord, "sequenceHash"> |
    Pick<ProceduralSkillCandidateRecord, "candidate">
): string {
  const sequenceHash = "sequenceHash" in source
    ? source.sequenceHash
    : source.candidate.sequenceHash;
  return `skill:procedural-policy-sequence:${sequenceHash}`;
}

function loadCandidateContext(
  repos: Repositories,
  candidate: ProceduralSkillCandidateRecord
): LoadedCandidateContext {
  const selectedEvidence = selectEvidence(candidate.candidate.evidence);
  const validRefs = new Set<string>();
  const supportOccurrenceEpisodes = new Map<string, string>();
  const refEpisodes = new Map<string, Set<string>>();
  const observedTools = new Set<string>();
  const sourcePolicyMemoryIds = new Set<string>();
  const sessions = new Map<string, SessionRecord>();
  const selected = selectedEvidence.map((evidence): LoadedEvidence => {
    const pathRecord = repos.proceduralPaths.get(evidence.pathId);
    if (!pathRecord || pathRecord.episodeId !== evidence.episodeId ||
        pathRecord.sessionId !== evidence.sessionId || pathRecord.status !== "active") {
      throw new Error(`procedural Sequence Skill evidence path is stale: ${evidence.pathId}`);
    }
    if (pathRecord.namespaceId !== candidate.namespaceId) {
      throw new Error(`procedural Sequence Skill evidence crosses learning scopes: ${evidence.pathId}`);
    }
    const session = repos.runtime.getSession(evidence.sessionId);
    if (!session || session.userId !== pathRecord.userId) {
      throw new Error(`procedural Sequence Skill evidence Session is missing: ${evidence.sessionId}`);
    }
    sessions.set(session.id, session);
    const policies = evidence.policyVersionIds.map((id, index) => {
      const policy = repos.proceduralPolicies.get(id);
      if (!policy || policy.namespaceId !== candidate.namespaceId ||
          policy.policyKey !== candidate.candidate.policyKeys[index]) {
        throw new Error(`procedural Sequence Skill Policy evidence is missing: ${id}`);
      }
      if (policy.l2MemoryId) sourcePolicyMemoryIds.add(policy.l2MemoryId);
      addRef(validRefs, refEpisodes, policy.id, evidence.episodeId);
      return policy;
    });
    const stateById = new Map(pathRecord.path.states.map((state) => [state.id, state]));
    const stepById = new Map(pathRecord.path.steps.map((step) => [step.id, step]));
    const spans = evidence.spanOccurrenceIds.map((id, index) => {
      const span = repos.proceduralPaths.getOccurrence(id);
      if (!span || span.pathId !== evidence.pathId || span.episodeId !== evidence.episodeId ||
          span.spanId !== evidence.spanIds[index]) {
        throw new Error(`procedural Sequence Skill Span evidence is missing: ${id}`);
      }
      addRef(validRefs, refEpisodes, span.id, evidence.episodeId);
      addRef(validRefs, refEpisodes, span.spanId, evidence.episodeId);
      for (const stepId of span.stepIds) {
        const step = stepById.get(stepId);
        if (!step) throw new Error(`procedural Sequence Skill Step evidence is missing: ${stepId}`);
        addRef(validRefs, refEpisodes, step.id, evidence.episodeId);
        if (step.action.toolName) observedTools.add(step.action.toolName);
      }
      return span;
    });
    const pathSpans = evidence.pathSpanOccurrenceIds.map((id, index) => {
      const span = repos.proceduralPaths.getOccurrence(id);
      if (!span || span.pathId !== evidence.pathId || span.episodeId !== evidence.episodeId ||
          span.spanId !== evidence.pathSpanIds[index]) {
        throw new Error(`procedural Sequence Skill path Span evidence is missing: ${id}`);
      }
      addRef(validRefs, refEpisodes, span.id, evidence.episodeId);
      addRef(validRefs, refEpisodes, span.spanId, evidence.episodeId);
      for (const stepId of span.stepIds) {
        const step = stepById.get(stepId);
        if (!step) throw new Error(`procedural Sequence Skill Step evidence is missing: ${stepId}`);
        addRef(validRefs, refEpisodes, step.id, evidence.episodeId);
        if (step.action.toolName) observedTools.add(step.action.toolName);
      }
      return span;
    });
    addRef(validRefs, refEpisodes, evidence.occurrenceId, evidence.episodeId);
    if (evidence.evidenceRole === "support") {
      supportOccurrenceEpisodes.set(evidence.occurrenceId, evidence.episodeId);
    }
    const rawTurns = repos.runtime.listRawTurnsByEpisode(evidence.episodeId, 8);
    const payload = {
      sequence_occurrence_id: evidence.occurrenceId,
      evidence_role: evidence.evidenceRole,
      episode_id: evidence.episodeId,
      session_id: evidence.sessionId,
      terminal_reward: evidence.terminalReward ?? null,
      episode_context: rawTurns.map((turn) => ({
        raw_turn_id: turn.id,
        user: clip(turn.userText ?? "", 800),
        assistant: clip(turn.assistantText ?? "", 800)
      })),
      cost: evidence.cost,
      discovery_sources: evidence.discoverySources,
      episode_affinity: evidence.episodeAffinity ?? null,
      stages: policies.map((policy, index) => {
        const span = spans[index]!;
        const steps = span.stepIds.slice(0, MAX_STEPS_PER_SPAN).map((stepId) => {
          const step = stepById.get(stepId)!;
          return {
            step_id: step.id,
            pre_state: clip(stateById.get(step.preStateId)?.summary ?? "", 600),
            action_kind: step.action.kind,
            action_type: step.action.type,
            intent: clip(step.action.intent, 500),
            action: clip(step.action.summary, 800),
            tool: step.action.toolName ?? null,
            outcome: step.outcome.status,
            post_state: clip(stateById.get(step.postStateId)?.summary ?? "", 600),
            retry_of_step_id: step.retryOfStepId ?? null,
            recovery_from_step_id: step.recoveryFromStepId ?? null,
            cost: step.cost
          };
        });
        return {
          position: index,
          policy_key: policy.policyKey,
          policy_version_id: policy.id,
          policy: {
            title: policy.policy.title,
            goal_pattern: policy.policy.goalPattern,
            trigger_conditions: policy.policy.triggerConditions,
            procedure_steps: policy.policy.procedureSteps,
            recovery_rules: policy.policy.recoveryRules,
            verification_steps: policy.policy.verificationSteps,
            apply_when: policy.policy.applyWhen,
            do_not_apply_when: policy.policy.doNotApplyWhen,
            invariants: policy.policy.invariants,
            expected_effect: policy.policy.expectedEffect
          },
          span_occurrence_id: span.id,
          span_id: span.spanId,
          local_goal: span.localGoal,
          entry_condition: span.entryCondition,
          pre_state: clip(stateById.get(span.preStateId)?.summary ?? "", 1_000),
          steps,
          exit_condition: span.exitCondition,
          post_state: clip(stateById.get(span.postStateId)?.summary ?? "", 1_000),
          termination_status: span.terminationStatus
        };
      }),
      full_path: pathSpans.map((span, pathIndex) => ({
        path_position: pathIndex,
        node_index: evidence.pathNodeIndexes[pathIndex],
        role: evidence.spanOccurrenceIds.includes(span.id) ? "policy_backbone" : "gap",
        span_occurrence_id: span.id,
        span_id: span.spanId,
        local_goal: span.localGoal,
        entry_condition: span.entryCondition,
        pre_state: clip(stateById.get(span.preStateId)?.summary ?? "", 1_000),
        steps: span.stepIds.slice(0, MAX_STEPS_PER_SPAN).map((stepId) => {
          const step = stepById.get(stepId)!;
          return {
            step_id: step.id,
            intent: clip(step.action.intent, 500),
            action: clip(step.action.summary, 800),
            tool: step.action.toolName ?? null,
            outcome: step.outcome.status,
            retry_of_step_id: step.retryOfStepId ?? null,
            recovery_from_step_id: step.recoveryFromStepId ?? null,
            cost: step.cost
          };
        }),
        exit_condition: span.exitCondition,
        post_state: clip(stateById.get(span.postStateId)?.summary ?? "", 1_000),
        termination_status: span.terminationStatus
      })),
      gap_spans: pathSpans.filter((span) => !evidence.spanOccurrenceIds.includes(span.id))
        .map((span) => ({
          span_occurrence_id: span.id,
          local_goal: span.localGoal,
          entry_condition: span.entryCondition,
          exit_condition: span.exitCondition,
          termination_status: span.terminationStatus,
          cost: span.span.cost
        }))
    };
    return { evidence, policies, spans, pathSpans, session, payload };
  });
  const supportEpisodes = new Set(selected
    .filter((item) => item.evidence.evidenceRole === "support")
    .map((item) => item.evidence.episodeId));
  if (supportEpisodes.size < 2) {
    throw new Error("procedural Sequence Skill compilation requires two selected support Episodes");
  }
  if (sourcePolicyMemoryIds.size === 0) {
    throw new Error("procedural Sequence Skill compilation requires persisted Policy memories");
  }
  return {
    selected,
    validRefs,
    supportOccurrenceEpisodes,
    refEpisodes,
    observedTools,
    sourcePolicyMemoryIds: [...sourcePolicyMemoryIds].sort(),
    sessions: [...sessions.values()],
    payload: {
      candidate: {
        id: candidate.id,
        pattern_id: candidate.patternId,
        namespace_id: candidate.namespaceId,
        policy_keys: candidate.candidate.policyKeys,
        capability_type: candidate.candidate.capabilityType,
        discovery_sources: candidate.candidate.discoverySources,
        episode_family_id: candidate.candidate.episodeFamilyId ?? null,
        distinct_support_episode_count: candidate.distinctSupportEpisodeCount,
        distinct_counterexample_episode_count: candidate.distinctCounterexampleEpisodeCount,
        aggregate_support_cost: candidate.candidate.aggregateSupportCost,
        evidence_hash: candidate.evidenceHash
      },
      observed_tools: [...observedTools].sort(),
      selected_path_evidence: selected.map((item) => item.payload)
    }
  };
}

function selectEvidence(
  evidence: readonly ProceduralSkillCandidateEvidenceV1[]
): ProceduralSkillCandidateEvidenceV1[] {
  const selected: ProceduralSkillCandidateEvidenceV1[] = [];
  const seenSupportEpisodes = new Set<string>();
  for (const item of evidence.filter((value) => value.evidenceRole === "support")) {
    if (selected.length >= MAX_SUPPORT_OCCURRENCES) break;
    if (seenSupportEpisodes.has(item.episodeId)) continue;
    selected.push(item);
    seenSupportEpisodes.add(item.episodeId);
  }
  selected.push(...evidence
    .filter((item) => item.evidenceRole === "counterexample")
    .slice(0, MAX_COUNTEREXAMPLES));
  selected.push(...evidence
    .filter((item) => item.evidenceRole === "uncertain")
    .slice(0, MAX_UNCERTAIN_OCCURRENCES));
  return selected;
}

function parseSequenceSkillDraft(
  value: SequenceSkillLlmResult,
  context: LoadedCandidateContext
): SequenceSkillDraft {
  assertExactKeys(value, [
    "name",
    "display_title",
    "retrieval_blurb",
    "trigger_context",
    "summary",
    "parameters",
    "preconditions",
    "steps",
    "recovery_rules",
    "verification",
    "do_not_use_when",
    "decision_guidance",
    "tools",
    "tags",
    "evidence_occurrence_ids",
    "confidence"
  ], "root");
  const name = requiredText(value.name, "name", 48).toLowerCase();
  if (!/^[a-z0-9]+(?:_[a-z0-9]+)*$/.test(name)) {
    throw new Error("Sequence Skill name must be lowercase snake_case");
  }
  const parameters = parseParameters(value.parameters);
  const preconditions = parseGroundedTextArray(
    value.preconditions,
    "preconditions",
    context,
    true,
    1,
    12
  );
  const steps = parseGroundedSteps(value.steps, context);
  const recoveryRules = parseRecoveryRules(value.recovery_rules, context);
  const verification = parseVerification(value.verification, context);
  const doNotUseWhen = parseGroundedTextArray(
    value.do_not_use_when,
    "do_not_use_when",
    context,
    false,
    0,
    12
  );
  const decisionGuidance = parseDecisionGuidance(value.decision_guidance);
  const tools = optionalStringArray(value.tools, "tools", 0, 32, 100);
  for (const tool of tools) {
    if (!context.observedTools.has(tool)) {
      throw new Error(`Sequence Skill invented tool: ${tool}`);
    }
  }
  const tags = unique(optionalStringArray(value.tags, "tags", 0, 16, 80)
    .map((tag) => tag.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9_-]/g, ""))
    .filter(Boolean));
  const evidenceOccurrenceIds = requiredStringArray(
    value.evidence_occurrence_ids,
    "evidence_occurrence_ids",
    2,
    MAX_SUPPORT_OCCURRENCES,
    200
  );
  const evidenceEpisodes = new Set(evidenceOccurrenceIds.map((id) => {
    const episodeId = context.supportOccurrenceEpisodes.get(id);
    if (!episodeId) throw new Error(`Sequence Skill cited non-support occurrence: ${id}`);
    return episodeId;
  }));
  if (evidenceEpisodes.size < 2) {
    throw new Error("Sequence Skill evidence_occurrence_ids must span two support Episodes");
  }
  const confidence = requiredNumber(value.confidence, "confidence", 0, 1);
  return {
    name,
    displayTitle: requiredText(value.display_title, "display_title", 160),
    retrievalBlurb: requiredText(value.retrieval_blurb, "retrieval_blurb", 500),
    triggerContext: requiredText(value.trigger_context, "trigger_context", 800),
    summary: requiredText(value.summary, "summary", 1_500),
    parameters,
    preconditions,
    steps,
    recoveryRules,
    verification,
    doNotUseWhen,
    decisionGuidance,
    tools: unique(tools),
    tags,
    evidenceOccurrenceIds: unique(evidenceOccurrenceIds),
    confidence
  };
}

function parseParameters(value: unknown): SequenceSkillDraft["parameters"] {
  if (!Array.isArray(value) || value.length > 16) {
    throw new Error("Sequence Skill parameters must contain 0-16 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Sequence Skill parameters[${index}] must be an object`);
    assertExactKeys(item, ["name", "type", "required", "description"], `parameters[${index}]`);
    if (typeof item.required !== "boolean") {
      throw new Error(`Sequence Skill parameters[${index}].required must be boolean`);
    }
    return {
      name: requiredText(item.name, `parameters[${index}].name`, 80),
      type: requiredText(item.type, `parameters[${index}].type`, 80),
      required: item.required,
      description: requiredText(item.description, `parameters[${index}].description`, 400)
    };
  });
}

function parseGroundedTextArray(
  value: unknown,
  field: string,
  context: LoadedCandidateContext,
  requirePositiveSupport: boolean,
  min: number,
  max: number
): GroundedText[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) {
    throw new Error(`Sequence Skill ${field} must contain ${min}-${max} items`);
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Sequence Skill ${field}[${index}] must be an object`);
    assertExactKeys(item, ["condition", "evidence_refs"], `${field}[${index}]`);
    const evidenceRefs = groundedRefs(
      item.evidence_refs,
      `${field}[${index}].evidence_refs`,
      context,
      requirePositiveSupport
    );
    return {
      text: requiredText(item.condition, `${field}[${index}].condition`, 800),
      evidenceRefs
    };
  });
}

function parseGroundedSteps(value: unknown, context: LoadedCandidateContext): GroundedStep[] {
  if (!Array.isArray(value) || value.length < 2 || value.length > 24) {
    throw new Error("Sequence Skill steps must contain 2-24 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Sequence Skill steps[${index}] must be an object`);
    assertExactKeys(item, ["title", "body", "evidence_refs"], `steps[${index}]`);
    return {
      title: requiredText(item.title, `steps[${index}].title`, 160),
      body: requiredText(item.body, `steps[${index}].body`, 1_200),
      evidenceRefs: groundedRefs(
        item.evidence_refs,
        `steps[${index}].evidence_refs`,
        context,
        true
      )
    };
  });
}

function parseRecoveryRules(
  value: unknown,
  context: LoadedCandidateContext
): GroundedRecovery[] {
  if (!Array.isArray(value) || value.length > 12) {
    throw new Error("Sequence Skill recovery_rules must contain 0-12 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Sequence Skill recovery_rules[${index}] must be an object`);
    assertExactKeys(item, ["condition", "action", "evidence_refs"], `recovery_rules[${index}]`);
    return {
      condition: requiredText(item.condition, `recovery_rules[${index}].condition`, 600),
      action: requiredText(item.action, `recovery_rules[${index}].action`, 800),
      evidenceRefs: groundedRefs(
        item.evidence_refs,
        `recovery_rules[${index}].evidence_refs`,
        context,
        false
      )
    };
  });
}

function parseVerification(
  value: unknown,
  context: LoadedCandidateContext
): GroundedVerification[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 12) {
    throw new Error("Sequence Skill verification must contain 1-12 items");
  }
  return value.map((item, index) => {
    if (!isRecord(item)) throw new Error(`Sequence Skill verification[${index}] must be an object`);
    assertExactKeys(item, ["check", "success_signal", "evidence_refs"], `verification[${index}]`);
    return {
      check: requiredText(item.check, `verification[${index}].check`, 600),
      successSignal: requiredText(item.success_signal, `verification[${index}].success_signal`, 600),
      evidenceRefs: groundedRefs(
        item.evidence_refs,
        `verification[${index}].evidence_refs`,
        context,
        true
      )
    };
  });
}

function parseDecisionGuidance(value: unknown): SequenceSkillDraft["decisionGuidance"] {
  if (!isRecord(value)) throw new Error("Sequence Skill decision_guidance must be an object");
  assertExactKeys(value, ["preference", "anti_pattern"], "decision_guidance");
  return {
    preference: optionalStringArray(value.preference, "decision_guidance.preference", 0, 16, 800),
    antiPattern: optionalStringArray(value.anti_pattern, "decision_guidance.anti_pattern", 0, 16, 800)
  };
}

function groundedRefs(
  value: unknown,
  field: string,
  context: LoadedCandidateContext,
  requirePositiveSupport: boolean
): string[] {
  const refs = requiredStringArray(value, field, 1, 24, 200);
  const episodes = new Set<string>();
  for (const ref of refs) {
    if (!context.validRefs.has(ref)) throw new Error(`Sequence Skill ${field} invented ref: ${ref}`);
    for (const episodeId of context.refEpisodes.get(ref) ?? []) episodes.add(episodeId);
  }
  if (requirePositiveSupport) {
    const supportEpisodes = new Set([...episodes].filter((episodeId) =>
      [...context.supportOccurrenceEpisodes.values()].includes(episodeId)
    ));
    if (supportEpisodes.size < 2) {
      throw new Error(`Sequence Skill ${field} requires support from two Episodes`);
    }
  }
  return unique(refs);
}

function buildProcedureJson(
  candidate: ProceduralSkillCandidateRecord,
  draft: SequenceSkillDraft,
  context: LoadedCandidateContext,
  reliability: {
    support: number;
    successRate: number;
    betaPosterior: ReturnType<typeof skillBetaPosterior>;
  }
): Record<string, unknown> {
  return {
    retrievalBlurb: draft.retrievalBlurb,
    triggerContext: draft.triggerContext,
    displayTitle: draft.displayTitle,
    summary: draft.summary,
    parameters: draft.parameters,
    preconditions: draft.preconditions.map((item) => item.text),
    steps: draft.steps.map((step) => ({
      title: step.title,
      body: step.body,
      evidenceRefs: step.evidenceRefs
    })),
    recoveryRules: draft.recoveryRules.map((rule) => ({
      condition: rule.condition,
      action: rule.action,
      evidenceRefs: rule.evidenceRefs
    })),
    verification: draft.verification.map((item) => ({
      check: item.check,
      successSignal: item.successSignal,
      evidenceRefs: item.evidenceRefs
    })),
    doNotUseWhen: draft.doNotUseWhen.map((item) => ({
      condition: item.text,
      evidenceRefs: item.evidenceRefs
    })),
    decisionGuidance: draft.decisionGuidance,
    reliability: {
      supportCount: reliability.support,
      successRate: reliability.successRate,
      betaPosterior: reliability.betaPosterior
    },
    tools: draft.tools,
    tags: draft.tags,
    grounding: {
      candidateId: candidate.id,
      patternId: candidate.patternId,
      capabilityType: candidate.candidate.capabilityType,
      discoverySources: candidate.candidate.discoverySources,
      episodeFamilyId: candidate.candidate.episodeFamilyId ?? null,
      evidenceHash: candidate.evidenceHash,
      evidenceOccurrenceIds: draft.evidenceOccurrenceIds,
      sourcePolicyMemoryIds: context.sourcePolicyMemoryIds
    }
  };
}

function renderInvocationGuide(draft: SequenceSkillDraft): string {
  const lines = [
    `# ${draft.displayTitle}`,
    "",
    draft.retrievalBlurb,
    "",
    draft.summary,
    "",
    "## When to use",
    draft.triggerContext,
    "",
    "## Preconditions",
    ...draft.preconditions.map((item) => `- ${item.text}`),
    "",
    "## Procedure",
    ...draft.steps.map((step, index) => `${index + 1}. **${step.title}** — ${step.body}`)
  ];
  if (draft.recoveryRules.length > 0) {
    lines.push(
      "",
      "## Recovery",
      ...draft.recoveryRules.map((rule) => `- If ${rule.condition}, ${rule.action}`)
    );
  }
  lines.push(
    "",
    "## Verification",
    ...draft.verification.map((item) => `- ${item.check} → ${item.successSignal}`)
  );
  if (draft.doNotUseWhen.length > 0) {
    lines.push(
      "",
      "## Do not use when",
      ...draft.doNotUseWhen.map((item) => `- ${item.text}`)
    );
  }
  return lines.join("\n");
}

function commonScope(
  sessions: readonly SessionRecord[],
  expectedUserId: string
): {
  userId: string;
} {
  const userIds = unique(sessions.map((session) => session.userId));
  if (userIds.length !== 1 || userIds[0] !== expectedUserId) {
    throw new Error("procedural Sequence Skill evidence crosses users");
  }
  return {
    userId: userIds[0]!
  };
}

function addRef(
  validRefs: Set<string>,
  refEpisodes: Map<string, Set<string>>,
  ref: string,
  episodeId: string
): void {
  validRefs.add(ref);
  const episodes = refEpisodes.get(ref) ?? new Set<string>();
  episodes.add(episodeId);
  refEpisodes.set(ref, episodes);
}

function repairInstruction(error: unknown, repairNumber: number): string {
  const reason = error instanceof Error ? error.message : String(error);
  const base = `The previous Skill JSON violated the contract: ${reason}. Cite only exact refs from selected_path_evidence, preserve the ordered Policy sequence, and use only observed_tools.`;
  return repairNumber === 1
    ? `Correct the complete JSON response. ${base}`
    : `Discard the previous JSON and regenerate it from the original evidence. ${base} Verify every key, evidence ref, and tool before returning.`;
}

function payloadText(job: EvolutionJobRecord, key: string): string | undefined {
  const value = job.payload[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function text(value: unknown, key: string): string | undefined {
  if (!isRecord(value)) return undefined;
  const field = value[key];
  return typeof field === "string" && field.trim() ? field.trim() : undefined;
}

function requiredStringArray(
  value: unknown,
  field: string,
  min: number,
  max: number,
  maxItemLength: number
): string[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) {
    throw new Error(`Sequence Skill ${field} must contain ${min}-${max} strings`);
  }
  return value.map((item, index) => requiredText(item, `${field}[${index}]`, maxItemLength));
}

function optionalStringArray(
  value: unknown,
  field: string,
  min: number,
  max: number,
  maxItemLength: number
): string[] {
  return requiredStringArray(value, field, min, max, maxItemLength);
}

function requiredText(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Sequence Skill ${field} must be non-empty text`);
  }
  const normalized = value.trim();
  if (normalized.length > maxLength) {
    throw new Error(`Sequence Skill ${field} exceeds ${maxLength} characters`);
  }
  return normalized;
}

function requiredNumber(
  value: unknown,
  field: string,
  min: number,
  max: number
): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`Sequence Skill ${field} must be between ${min} and ${max}`);
  }
  return value;
}

function assertExactKeys(record: Record<string, unknown>, allowed: string[], field: string): void {
  const expected = [...allowed].sort();
  const actual = Object.keys(record).sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`Sequence Skill ${field} keys must be exactly: ${allowed.join(", ")}`);
  }
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
