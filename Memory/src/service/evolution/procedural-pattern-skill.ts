import { skillMetaFromMemory } from "../../algorithm/plugin-algorithms.js";
import type { MemmyConfig } from "../../config/index.js";
import type { LlmClient } from "../../model/types.js";
import type { Repositories } from "../../storage/repositories.js";
import type { MemoryRow } from "../../types.js";
import { stableHash, stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { clip } from "../../utils/text.js";
import {
  profileIdFromMemory,
  projectIdFromMemory
} from "../namespace/namespace-scope.js";
import {
  coerceSkillExamples,
  coerceSkillName,
  coerceSkillParameters,
  detectSkillModelRefusal,
  skillMarkdown,
  skillText
} from "./skill-pipeline.js";

export const PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION =
  "procedural-pattern-skill.v2" as const;

const PROCEDURAL_PATTERN_SKILL_PROMPT = `You compile one reusable Skill directly from an aligned cross-Episode execution pattern.

The evidence was found by fixed Span-5/Span-10 windows, whole-sequence semantic recall, and banded monotonic Step alignment. Mechanical windows may contain unrelated prefix or suffix actions.

Input semantics:
- common_core.anchors are the cross-Episode aligned positions and the primary proof of a repeated method.
- occurrences[].aligned_sequence projects those anchors back to each concrete Episode in execution order.
- role=core is aligned evidence for one common_core anchor.
- role=gap is an occurrence-local Step between two aligned anchors. A Gap is context for causal completeness, not automatically shared proof.
- boundary_context_read_only exposes at most one adjacent Step on either side for trigger/boundary interpretation. It is never positive procedure or verification evidence.

Admission rules:
- Admit only a coherent, reusable method repeated in at least two distinct successful Episodes.
- Reject generic commonality such as merely calling a tool, writing a file, running a command, or checking output.
- Keep the smallest complete capability that is useful to invoke. Do not combine independent procedures merely because one large window contains both.
- Preserve the shared action order. Failed actions are counterfactual evidence, never proof of success.
- Use Gaps to detect missing causal bridges such as edit -> rerun -> verify. If equivalent Gaps do not occur in at least two distinct successful Episodes, do not turn them into a mandatory procedure Step.
- If the common Core plus its occurrence-local Gaps does not support one shared causally complete method, reject the candidate instead of hiding the disagreement.
- A mandatory procedure or verification item that uses Gap evidence must cite equivalent Gap step_id values from at least two distinct successful Episodes.
- Never cite boundary_context_read_only as procedure or verification evidence.
- counterexamples_read_only may inform boundaries and do_not_apply_when only. Never use it as procedure, verification, support, or evidence_occurrence_ids evidence.
- If counterexamples_read_only is empty, do_not_apply_when may be empty. Any exclusion must be a conservative restatement of a positive precondition, not an invented failure mode.
- Every procedure and verification item must cite exact step_id values from the payload when an occurrence contains a Gap or a failed Step. An occurrence_id may be cited only when its complete aligned_sequence has neither.
- Verification must be grounded by successful evidence from at least two distinct Episodes.
- Do not invent tools, constraints, files, checks, results, or causal explanations.
- evidence_occurrence_ids must cite at least two occurrences from distinct Episodes.

Return JSON only with exactly these keys:
{
  "admit": true,
  "rejection_reason": null,
  "name": "kebab-case-skill-name",
  "display_title": "...",
  "retrieval_blurb": "...",
  "trigger_context": "...",
  "summary": "...",
  "preconditions": ["..."],
  "parameters": [],
  "procedure_steps": [
    {"title": "...", "body": "...", "evidence_refs": ["..."]}
  ],
  "verification_steps": [
    {"check": "...", "success_signal": "...", "evidence_refs": ["..."]}
  ],
  "do_not_apply_when": ["..."],
  "decision_guidance": {"preference": ["..."], "anti_pattern": ["..."]},
  "examples": [],
  "tags": ["..."],
  "tools": ["..."],
  "evidence_occurrence_ids": ["..."],
  "confidence": 0.0
}

For rejection set admit=false, provide rejection_reason, and return empty strings or arrays for every Skill field.`;

export interface ProceduralSkillEvidenceStep {
  stepId: string;
  stepIndex: number;
  toolName?: string;
  intent: string;
  summary: string;
  outcome: "success" | "failure" | "partial" | "unknown";
  evidenceRefs: string[];
}

export interface ProceduralSkillCommonCoreAnchor {
  anchorId: string;
  anchorOffset: number;
  anchorStepId: string;
  anchorIntent: string;
  anchorSummary: string;
  supportEpisodeIds: string[];
  evidenceStepIds: string[];
}

export type ProceduralSkillAlignedSequenceStep =
  | (ProceduralSkillEvidenceStep & {
      role: "core";
      anchorId: string;
      matchSimilarity: number;
    })
  | (ProceduralSkillEvidenceStep & {
      role: "gap";
      afterAnchorId: string;
      beforeAnchorId: string;
    });

export interface ProceduralSkillBoundaryContext {
  previousStep?: ProceduralSkillEvidenceStep;
  nextStep?: ProceduralSkillEvidenceStep;
}

export interface ProceduralSkillEvidenceOccurrence {
  occurrenceId: string;
  episodeId: string;
  pathId: string;
  scale: number;
  alignmentScore: number;
  sourceTraceIds: string[];
  alignedSequence: ProceduralSkillAlignedSequenceStep[];
  boundaryContextReadOnly: ProceduralSkillBoundaryContext;
}

export interface ProceduralPatternSkillInput {
  patternVersionId: string;
  clusterId: string;
  clusterVersionId: string;
  commonCoreId: string;
  commonCore: ProceduralSkillCommonCoreAnchor[];
  userId: string;
  scale: number;
  supportEpisodeIds: string[];
  sourceTraceIds: string[];
  sourceSpanOccurrenceIds: string[];
  counterexampleEpisodeIds: string[];
  evidence: ProceduralSkillEvidenceOccurrence[];
  confidenceHint: number;
  patternHash: string;
  algorithmVersion: string;
}

export type ProceduralPatternSkillCompileResult =
  | { admitted: true; draft: ProceduralPatternSkillDraft }
  | { admitted: false; reason: string };

export interface ProceduralPatternSkillDraft {
  input: ProceduralPatternSkillInput;
  parsed: ParsedSkill;
  evidence: ProceduralSkillEvidenceOccurrence[];
  sourceTraceIds: string[];
  sourceEpisodeIds: string[];
  gain: number;
  confidence: number;
  scope: ProceduralSkillScope;
}

export interface MaterializedProceduralPatternSkill {
  memory: MemoryRow;
  contentHash: string;
  sourceEpisodeIds: string[];
  scope: ProceduralSkillScope;
}

export interface ProceduralSkillScope {
  conversationId?: string;
  sessionId?: string;
  agentId?: string;
  appId?: string;
  projectId?: string;
  profileId?: string;
}

type TraceMeta = NonNullable<ReturnType<ProceduralPatternSkillDeps["traceMeta"]>>;

export interface ProceduralPatternSkillDeps {
  repos: Repositories;
  config: MemmyConfig;
  skillLlm: LlmClient;
  traceMeta(memory: MemoryRow | null | undefined): {
    id: string;
    userId: string;
    sessionId?: string;
    episodeId?: string;
    value: number;
    priority: number;
    memory: MemoryRow;
  } | null;
  buildMemory(input: Record<string, unknown>): MemoryRow;
}

export class ProceduralPatternSkillMaterializer {
  constructor(private readonly deps: ProceduralPatternSkillDeps) {}

  async compile(
    input: ProceduralPatternSkillInput
  ): Promise<ProceduralPatternSkillCompileResult> {
    const successfulEpisodeIds = new Set(input.evidence.flatMap((item) => {
      const episode = this.deps.repos.runtime.getEpisode(item.episodeId);
      return episode?.userId === input.userId &&
        typeof episode.rTask === "number" &&
        episode.rTask >= this.deps.config.algorithm.skill.outcomeRTaskSuccessThreshold
        ? [item.episodeId]
        : [];
    }));
    const declaredCounterexampleIds = new Set(input.counterexampleEpisodeIds);
    const positiveOccurrenceIds = new Set(input.evidence
      .filter((item) => successfulEpisodeIds.has(item.episodeId) && occurrenceHasSuccess(item))
      .map((item) => item.occurrenceId));
    const counterexamples = representativeEvidence(
      input.evidence.filter((item) => {
        const episode = this.deps.repos.runtime.getEpisode(item.episodeId);
        return episode?.userId === input.userId &&
          !positiveOccurrenceIds.has(item.occurrenceId) && (
          occurrenceIsFailure(item) ||
          declaredCounterexampleIds.has(item.episodeId) ||
          (typeof episode.rTask === "number" &&
            episode.rTask <= this.deps.config.algorithm.skill.outcomeRTaskFailureThreshold)
        );
      }),
      this.deps.config.algorithm.proceduralWindow.maxSkillEvidenceEpisodes
    );
    const evidence = representativeEvidence(
      input.evidence.filter((item) => positiveOccurrenceIds.has(item.occurrenceId)),
      this.deps.config.algorithm.proceduralWindow.maxSkillEvidenceEpisodes
    );
    const evidenceEpisodeIds = unique(evidence.map((item) => item.episodeId));
    if (evidenceEpisodeIds.length < this.deps.config.algorithm.proceduralWindow.minSupportEpisodes) {
      return { admitted: false, reason: "insufficient-positive-episode-support" };
    }
    if (!this.deps.config.algorithm.skill.useLlm || !this.deps.skillLlm.isConfigured()) {
      return { admitted: false, reason: "llm-disabled" };
    }
    const alignedSourceTraceIds = new Set(evidence.flatMap((item) => item.sourceTraceIds));
    const sourceTraces = this.deps.repos.memories
      .getMany(unique(input.sourceTraceIds).filter((id) => alignedSourceTraceIds.has(id)))
      .map((memory) => this.deps.traceMeta(memory))
      .filter((trace): trace is TraceMeta => Boolean(
        trace && trace.userId === input.userId && trace.episodeId &&
        evidenceEpisodeIds.includes(trace.episodeId)
      ));
    const sourceTraceIds = unique(sourceTraces.map((trace) => trace.id));
    const sourceEpisodeIds = unique(sourceTraces
      .map((trace) => trace.episodeId)
      .filter(isString));
    if (sourceEpisodeIds.length < this.deps.config.algorithm.proceduralWindow.minSupportEpisodes) {
      return { admitted: false, reason: "parent-traces-do-not-cover-support-episodes" };
    }
    const gain = positiveTraceGain(sourceTraces);
    if (gain < this.deps.config.algorithm.skill.minGain) {
      return { admitted: false, reason: "insufficient-positive-trace-gain" };
    }

    const llmResult = await this.deps.skillLlm.completeJson<Record<string, unknown>>([
      { role: "system", content: PROCEDURAL_PATTERN_SKILL_PROMPT },
      {
        role: "user",
        content: stableStringify({
          pattern: {
            pattern_version_id: input.patternVersionId,
            cluster_id: input.clusterId,
            cluster_version_id: input.clusterVersionId,
            common_core_id: input.commonCoreId,
            scale: input.scale,
            support_episode_count: evidenceEpisodeIds.length,
            confidence_hint: input.confidenceHint
          },
          common_core: {
            id: input.commonCoreId,
            anchors: input.commonCore.map((anchor) => ({
              anchor_id: anchor.anchorId,
              anchor_offset: anchor.anchorOffset,
              anchor_step_id: anchor.anchorStepId,
              anchor_intent: clip(anchor.anchorIntent, 400),
              anchor_summary: clip(anchor.anchorSummary, 600),
              support_episode_ids: anchor.supportEpisodeIds,
              evidence_step_ids: anchor.evidenceStepIds
            }))
          },
          occurrences: evidence.map((occurrence) => ({
            occurrence_id: occurrence.occurrenceId,
            episode_id: occurrence.episodeId,
            alignment_score: occurrence.alignmentScore,
            boundary_context_read_only: serializeBoundaryContext(
              occurrence.boundaryContextReadOnly
            ),
            aligned_sequence: occurrence.alignedSequence.map(serializeAlignedStep)
          })),
          counterexamples_read_only: counterexamples.map((occurrence) => ({
            occurrence_id: occurrence.occurrenceId,
            episode_id: occurrence.episodeId,
            boundary_context_read_only: serializeBoundaryContext(
              occurrence.boundaryContextReadOnly
            ),
            aligned_sequence: occurrence.alignedSequence.map(serializeAlignedStep)
          }))
        })
      }
    ], {
      operation: `procedural.${PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION}`,
      thinkingMode: "enabled",
      temperature: 0,
      maxTokens: 8_192
    });
    if (detectSkillModelRefusal(llmResult)) {
      return { admitted: false, reason: "llm-refusal" };
    }
    const parsed = parseSkillResult(llmResult, evidence);
    if (!parsed.ok) return { admitted: false, reason: parsed.reason };

    const confidence = clamp(
      0.7 * parsed.confidence + 0.3 * clamp(input.confidenceHint, 0, 1),
      0,
      1
    );
    return {
      admitted: true,
      draft: {
        input,
        parsed,
        evidence,
        sourceTraceIds,
        sourceEpisodeIds,
        gain,
        confidence,
        scope: consensusSkillScope(sourceTraces.map((trace) => trace.memory))
      }
    };
  }

  /**
   * Convert an already-validated model draft into the canonical public Skill
   * shape. This method deliberately performs no writes. Callers must invoke it
   * inside the same transaction that validates and advances the source cluster.
   */
  materializeDraft(
    draft: ProceduralPatternSkillDraft,
    createdAt: string
  ): MaterializedProceduralPatternSkill {
    const { input, parsed, evidence, sourceTraceIds, sourceEpisodeIds, gain, confidence, scope } = draft;
    const skillKey = proceduralSkillKey(input.userId, input.clusterId);
    const existingMemory = this.deps.repos.memories.getByKey("Skill", skillKey);
    const existingSkill = existingMemory ? skillMetaFromMemory(existingMemory) : null;
    const trialsAttempted = existingSkill?.trialsAttempted ?? 0;
    const trialsPassed = existingSkill?.trialsPassed ?? 0;
    const successRate = trialsAttempted > 0 ? trialsPassed / trialsAttempted : 0;
    const betaPosterior = {
      alpha: 1 + trialsPassed,
      beta: 1 + Math.max(0, trialsAttempted - trialsPassed),
      mean: (1 + trialsPassed) / (2 + trialsAttempted)
    };
    const eta = existingSkill && existingSkill.trialsAttempted > 0
      ? existingSkill.eta
      : clamp(Math.max(
          this.deps.config.algorithm.skill.minEtaForRetrieval,
          0.5 * clamp(gain, 0, 1) + 0.5 * Math.min(
            1,
            sourceEpisodeIds.length / Math.max(1, this.deps.config.algorithm.skill.minSupport)
          )
        ), 0, 1);
    const status = existingSkill?.status === "active" ? "active" : "candidate";
    const steps = materializedSkillSteps(parsed, evidence);
    const citedEvidenceRefs = unique([
      ...parsed.procedureSteps.flatMap((step) => step.evidenceRefs),
      ...parsed.verificationSteps.flatMap((step) => step.evidenceRefs)
    ]);
    const procedureJson = {
      retrievalBlurb: parsed.retrievalBlurb,
      triggerContext: parsed.triggerContext,
      summary: parsed.summary,
      preconditions: parsed.preconditions,
      parameters: parsed.parameters,
      steps,
      examples: parsed.examples,
      decisionGuidance: parsed.decisionGuidance,
      doNotApplyWhen: parsed.doNotApplyWhen,
      reliability: {
        supportCount: sourceEpisodeIds.length,
        successRate,
        betaPosterior,
        evidenceConfidence: confidence
      },
      tags: parsed.tags,
      tools: parsed.tools
    };
    const invocationGuide = renderInvocationGuide(parsed, sourceEpisodeIds.length, confidence);
    const contentHash = stableHash({
      patternHash: input.patternHash,
      name: parsed.name,
      invocationGuide,
      procedureJson
    });
    const skill = this.deps.buildMemory({
      userId: input.userId,
      conversationId: scope.conversationId,
      sessionId: scope.sessionId,
      agentId: scope.agentId,
      appId: scope.appId,
      projectId: scope.projectId,
      profileId: scope.profileId,
      layer: "Skill",
      kind: "skill",
      lifecycleStatus: status,
      memoryType: "SkillMemory",
      key: skillKey,
      value: invocationGuide,
      tags: unique(["skill", "procedural-pattern", `span-${input.scale}`, ...parsed.tags]),
      info: {
        name: parsed.name,
        eta,
        status,
        source_memory_ids: sourceTraceIds,
        source_policy_ids: [],
        evidence_anchor_ids: sourceTraceIds
      },
      internal: {
        source: "worker.procedural_skill_induction.v1",
        plugin_algorithm: "procedural.pattern.skill.v1",
        read_only: false,
        generated_by_memory_base: true,
        source_memory_ids: sourceTraceIds,
        source_policy_ids: [],
        source_world_model_ids: [],
        evidence_anchor_ids: sourceTraceIds,
        source_trace_ids: sourceTraceIds,
        source_episode_ids: sourceEpisodeIds,
        source_span_occurrence_ids: parsed.evidenceOccurrenceIds,
        source_pattern_version_id: input.patternVersionId,
        source_cluster_id: input.clusterId,
        source_cluster_version_id: input.clusterVersionId,
        window_scale: input.scale,
        alignment_config_version: input.algorithmVersion,
        induction_prompt_version: PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION,
        counterexample_episode_ids: input.counterexampleEpisodeIds,
        pattern_content_hash: input.patternHash,
        name: parsed.name,
        invocation_guide: invocationGuide,
        procedure_json: procedureJson,
        eta,
        support: sourceEpisodeIds.length,
        gain,
        confidence,
        evidence_refs: citedEvidenceRefs,
        status,
        trials_attempted: trialsAttempted,
        trials_passed: trialsPassed,
        success_rate: successRate,
        beta_posterior: betaPosterior,
        skill: {
          name: parsed.name,
          eta,
          status,
          support: sourceEpisodeIds.length,
          gain,
          source_policy_ids: [],
          source_world_model_ids: [],
          evidence_anchor_ids: sourceTraceIds,
          invocation_guide: invocationGuide,
          procedure_json: procedureJson,
          trials_attempted: trialsAttempted,
          trials_passed: trialsPassed,
          success_rate: successRate,
          beta_posterior: betaPosterior,
          verification: {
            ok: true,
            evidence_occurrence_ids: parsed.evidenceOccurrenceIds,
            evidence_refs: citedEvidenceRefs
          },
          vec: null
        }
      },
      createdAt
    });
    return {
      memory: skill,
      contentHash,
      sourceEpisodeIds,
      scope
    };
  }
}

export function proceduralSkillKey(userId: string, clusterId: string): string {
  return `skill:procedural:${stableHash({ userId, clusterId }).slice(0, 20)}`;
}

function consensusSkillScope(memories: MemoryRow[]): ProceduralSkillScope {
  return {
    conversationId: consensusString(memories.map((memory) => memory.conversationId)),
    sessionId: consensusString(memories.map((memory) => memory.sessionId)),
    agentId: consensusString(memories.map((memory) => memory.agentId)),
    appId: consensusString(memories.map((memory) => memory.appId)),
    projectId: consensusString(memories.map(projectIdFromMemory)),
    profileId: consensusString(memories.map(profileIdFromMemory))
  };
}

function consensusString(values: Array<string | undefined>): string | undefined {
  if (values.length === 0) return undefined;
  const [first, ...rest] = values;
  return rest.every((value) => value === first) ? first : undefined;
}

function serializeEvidenceStep(step: ProceduralSkillEvidenceStep): Record<string, unknown> {
  return {
    step_id: step.stepId,
    step_index: step.stepIndex,
    ...(step.toolName ? { tool_name: step.toolName } : {}),
    intent: clip(step.intent, 400),
    summary: clip(step.summary, 600),
    outcome: step.outcome,
    evidence_refs: step.evidenceRefs
  };
}

function serializeAlignedStep(
  step: ProceduralSkillAlignedSequenceStep
): Record<string, unknown> {
  return step.role === "core"
    ? {
        role: "core",
        anchor_id: step.anchorId,
        match_similarity: step.matchSimilarity,
        ...serializeEvidenceStep(step)
      }
    : {
        role: "gap",
        after_anchor_id: step.afterAnchorId,
        before_anchor_id: step.beforeAnchorId,
        ...serializeEvidenceStep(step)
      };
}

function serializeBoundaryContext(
  context: ProceduralSkillBoundaryContext
): Record<string, unknown> {
  return {
    previous_step: context.previousStep
      ? serializeEvidenceStep(context.previousStep)
      : null,
    next_step: context.nextStep ? serializeEvidenceStep(context.nextStep) : null
  };
}

interface ParsedSkill {
  name: string;
  displayTitle: string;
  retrievalBlurb: string;
  triggerContext: string;
  summary: string;
  preconditions: string[];
  parameters: unknown[];
  procedureSteps: Array<{ title: string; body: string; evidenceRefs: string[] }>;
  verificationSteps: Array<{ check: string; successSignal: string; evidenceRefs: string[] }>;
  doNotApplyWhen: string[];
  decisionGuidance: { preference: string[]; antiPattern: string[] };
  examples: unknown[];
  tags: string[];
  tools: string[];
  evidenceOccurrenceIds: string[];
  confidence: number;
}

function materializedSkillSteps(
  parsed: ParsedSkill,
  evidence: ProceduralSkillEvidenceOccurrence[]
): Array<Record<string, unknown>> {
  const occurrencesByRef = new Map<string, string[]>();
  for (const occurrence of evidence) {
    const refs = [
      occurrence.occurrenceId,
      ...occurrence.alignedSequence.flatMap((step) => [step.stepId, ...step.evidenceRefs])
    ];
    for (const ref of refs) {
      occurrencesByRef.set(ref, unique([
        ...(occurrencesByRef.get(ref) ?? []),
        occurrence.occurrenceId
      ]));
    }
  }
  const build = (
    kind: "procedure" | "verification",
    title: string,
    body: string,
    evidenceRefs: string[]
  ): Record<string, unknown> => ({
    id: `step_${stableHash({ kind, title, body }).slice(0, 12)}`,
    title,
    body,
    evidenceRefs,
    supportingPolicyIds: [],
    supportingOccurrenceIds: unique(evidenceRefs.flatMap((ref) =>
      occurrencesByRef.get(ref) ?? []))
  });
  return [
    ...parsed.procedureSteps.map((step) => build(
      "procedure",
      step.title,
      step.body,
      step.evidenceRefs
    )),
    ...parsed.verificationSteps.map((step) => build(
      "verification",
      `Verify: ${step.check}`,
      `Success signal: ${step.successSignal}`,
      step.evidenceRefs
    ))
  ];
}

function parseSkillResult(
  result: Record<string, unknown>,
  evidence: ProceduralSkillEvidenceOccurrence[]
): ({ ok: true } & ParsedSkill) | { ok: false; reason: string } {
  if (result.admit !== true) {
    return { ok: false, reason: skillText(result.rejection_reason) || "llm-rejected" };
  }
  const rawName = skillText(result.name);
  const name = rawName ? coerceSkillName(rawName, "skill") : undefined;
  const displayTitle = skillText(result.display_title) || undefined;
  const retrievalBlurb = skillText(result.retrieval_blurb) || undefined;
  const triggerContext = skillText(result.trigger_context) || undefined;
  const summary = skillText(result.summary) || undefined;
  const confidence = finiteNumber(result.confidence);
  const preconditions = markdownArray(result.preconditions);
  const parameters = coerceSkillParameters(result.parameters);
  const procedureSteps = parseProcedureSteps(result.procedure_steps);
  const verificationSteps = parseVerificationSteps(result.verification_steps);
  const doNotApplyWhen = markdownArray(result.do_not_apply_when);
  const decisionGuidance = parseDecisionGuidance(result.decision_guidance);
  const examples = coerceSkillExamples(result.examples);
  const tags = textArray(result.tags);
  const tools = textArray(result.tools);
  const evidenceOccurrenceIds = stringArray(result.evidence_occurrence_ids);
  if (!name || !displayTitle || !retrievalBlurb || !triggerContext || !summary ||
      confidence === undefined || preconditions.length === 0 || procedureSteps.length === 0 ||
      verificationSteps.length === 0) {
    return { ok: false, reason: "invalid-skill-fields" };
  }
  const byOccurrence = new Map(evidence.map((item) => [item.occurrenceId, item]));
  if (evidenceOccurrenceIds.some((id) => !byOccurrence.has(id))) {
    return { ok: false, reason: "invalid-evidence-occurrence" };
  }
  const citedEpisodes = unique(evidenceOccurrenceIds
    .map((id) => byOccurrence.get(id)?.episodeId)
    .filter(isString));
  if (citedEpisodes.length < 2) return { ok: false, reason: "insufficient-cited-episodes" };
  // A failed action may remain visible to the compiler as counterfactual
  // context, but it can never prove a positive procedure or verification.
  // Occurrence-level citations are therefore allowed only when the complete
  // occurrence contains neither a failed Step nor a Gap. Mixed or gap-bearing
  // windows must cite concrete Steps so the validator can distinguish shared
  // Core proof from occurrence-local causal context.
  const positiveRefsByOccurrence = new Map(evidence.map((item) => [
    item.occurrenceId,
    [
      ...(item.alignedSequence.some((step) =>
        step.outcome === "failure" || step.role === "gap")
        ? []
        : [item.occurrenceId]),
      ...item.alignedSequence.flatMap((step) => step.outcome === "failure"
        ? []
        : [step.stepId, ...step.evidenceRefs])
    ]
  ]));
  const allowedRefs = new Set([...positiveRefsByOccurrence.values()].flat());
  const citedRefs = [
    ...procedureSteps.flatMap((item) => item.evidenceRefs),
    ...verificationSteps.flatMap((item) => item.evidenceRefs)
  ];
  if (citedRefs.length === 0 || citedRefs.some((ref) => !allowedRefs.has(ref))) {
    return { ok: false, reason: "invalid-evidence-citation" };
  }
  const episodesByRef = new Map<string, string[]>();
  for (const occurrence of evidence) {
    for (const ref of positiveRefsByOccurrence.get(occurrence.occurrenceId) ?? []) {
      episodesByRef.set(ref, unique([
        ...(episodesByRef.get(ref) ?? []),
        occurrence.episodeId
      ]));
    }
  }
  const verificationEpisodes = unique(verificationSteps.flatMap((item) =>
    item.evidenceRefs.flatMap((ref) => episodesByRef.get(ref) ?? [])));
  if (verificationEpisodes.length < 2) {
    return { ok: false, reason: "insufficient-verification-episode-support" };
  }
  const gapEpisodesByRef = new Map<string, string[]>();
  for (const occurrence of evidence) {
    for (const step of occurrence.alignedSequence) {
      if (step.role !== "gap" || step.outcome === "failure") continue;
      for (const ref of [step.stepId, ...step.evidenceRefs]) {
        gapEpisodesByRef.set(ref, unique([
          ...(gapEpisodesByRef.get(ref) ?? []),
          occurrence.episodeId
        ]));
      }
    }
  }
  for (const item of [...procedureSteps, ...verificationSteps]) {
    const citedGapRefs = item.evidenceRefs.filter((ref) => gapEpisodesByRef.has(ref));
    if (citedGapRefs.length === 0) continue;
    const supportingEpisodes = unique(citedGapRefs.flatMap((ref) =>
      gapEpisodesByRef.get(ref) ?? []));
    if (supportingEpisodes.length < 2) {
      return { ok: false, reason: "insufficient-gap-episode-support" };
    }
  }
  const evidenceTools = new Set(evidence.flatMap((item) =>
    item.alignedSequence
      .map((step) => step.toolName?.trim().toLowerCase())
      .filter(isString)));
  if (tools.some((tool) => !evidenceTools.has(tool.trim().toLowerCase()))) {
    return { ok: false, reason: "invented-tool" };
  }
  return {
    ok: true,
    name,
    displayTitle,
    retrievalBlurb,
    triggerContext,
    summary,
    preconditions,
    parameters,
    procedureSteps,
    verificationSteps,
    doNotApplyWhen,
    decisionGuidance,
    examples,
    tags,
    tools,
    evidenceOccurrenceIds,
    confidence: clamp(confidence, 0, 1)
  };
}

function parseProcedureSteps(value: unknown): ParsedSkill["procedureSteps"] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const title = skillText(item.title) || undefined;
    const body = skillMarkdown(item.body) || undefined;
    const evidenceRefs = stringArray(item.evidence_refs);
    return title && body && evidenceRefs.length > 0 ? [{ title, body, evidenceRefs }] : [];
  });
}

function parseVerificationSteps(value: unknown): ParsedSkill["verificationSteps"] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const check = skillText(item.check) || undefined;
    const successSignal = skillMarkdown(item.success_signal) || undefined;
    const evidenceRefs = stringArray(item.evidence_refs);
    return check && successSignal && evidenceRefs.length > 0
      ? [{ check, successSignal, evidenceRefs }]
      : [];
  });
}

function parseDecisionGuidance(value: unknown): ParsedSkill["decisionGuidance"] {
  const record = isRecord(value) ? value : {};
  return {
    preference: markdownArray(record.preference),
    antiPattern: markdownArray(record.anti_pattern)
  };
}

function occurrenceHasSuccess(occurrence: ProceduralSkillEvidenceOccurrence): boolean {
  return occurrence.alignedSequence.some((step) => step.outcome === "success");
}

function occurrenceIsFailure(occurrence: ProceduralSkillEvidenceOccurrence): boolean {
  return !occurrenceHasSuccess(occurrence) &&
    occurrence.alignedSequence.some((step) => step.outcome === "failure");
}

function representativeEvidence(
  evidence: ProceduralSkillEvidenceOccurrence[],
  limit: number
): ProceduralSkillEvidenceOccurrence[] {
  const byEpisode = new Map<string, ProceduralSkillEvidenceOccurrence>();
  for (const item of [...evidence].sort((left, right) =>
    right.alignmentScore - left.alignmentScore || left.occurrenceId.localeCompare(right.occurrenceId))) {
    if (!byEpisode.has(item.episodeId)) byEpisode.set(item.episodeId, item);
  }
  return [...byEpisode.values()].slice(0, Math.max(2, limit));
}

function renderInvocationGuide(
  parsed: ParsedSkill,
  support: number,
  confidence: number
): string {
  return [
    `# ${parsed.displayTitle}`,
    "",
    parsed.retrievalBlurb,
    "",
    "## When to use",
    parsed.triggerContext,
    "",
    "## Preconditions",
    ...parsed.preconditions.map((item) => `- ${item}`),
    "",
    "## Procedure",
    ...parsed.procedureSteps.map((item, index) => `${index + 1}. **${item.title}** — ${item.body}`),
    "",
    "## Verification",
    ...parsed.verificationSteps.map((item) => `- ${item.check} Success: ${item.successSignal}`),
    ...(parsed.doNotApplyWhen.length > 0 ? [
      "",
      "## Do not apply when",
      ...parsed.doNotApplyWhen.map((item) => `- ${item}`)
    ] : []),
    "",
    `Evidence support: ${support} Episodes; confidence: ${confidence.toFixed(2)}`
  ].join("\n").trim();
}

function positiveTraceGain(traces: Array<{ value: number; priority: number }>): number {
  const values = traces.map((trace) => trace.value).filter(Number.isFinite);
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function textArray(value: unknown): string[] {
  return Array.isArray(value)
    ? unique(value.map((item) => skillText(item)).filter(Boolean))
    : [];
}

function markdownArray(value: unknown): string[] {
  return Array.isArray(value)
    ? unique(value.map((item) => skillMarkdown(item)).filter(Boolean))
    : [];
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? unique(value.filter(isString).map((item) => item.trim()).filter(Boolean))
    : [];
}

function isString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
