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
  "procedural-pattern-skill.v12" as const;
export const PROCEDURAL_LONG_TRAJECTORY_SKILL_PROMPT_VERSION =
  "procedural-long-trajectory-skill.v3" as const;
export const PROCEDURAL_SKILL_COVERAGE_PROMPT_VERSION =
  "procedural-skill-coverage.v1" as const;

const PROCEDURAL_PATTERN_SKILL_PROMPT = `You compile one reusable Skill directly from an aligned cross-Episode execution pattern.

The input may come from two discovery modes. pattern.origin.kind=aligned_window is the existing aligned-Step pipeline. pattern.origin.kind=long_trajectory is the multi-scale pipeline where one newly completed reference Episode defines a Span-5/Span-10/Span-15 vocabulary and other Episodes are encoded as the longest monotonic non-overlapping sequence of coarsely similar reference Span tokens. Mechanical windows may contain unrelated prefix or suffix actions.

Input semantics:
- For aligned_window input, common_core.anchors are aligned individual Step positions.
- For long_trajectory input, common_core.anchors are repeated reference Span tokens, not individual Step positions.
- occurrences[].aligned_sequence expands every matched Span token back to its concrete Steps in execution order.
- For long_trajectory input, role=span_step is one concrete Step inside a Coarse-matched Span token. It is not pre-aligned to any Step in another Episode. Compare all occurrences yourself and cite only the Steps that support the same abstract operation.
- role=core remains the individual aligned Step representation used by aligned_window input.
- role=local_context is read-only outward context and can never prove a shared procedure or verification claim.
- role=gap is an occurrence-local Step between two aligned anchors. A Gap is context for causal completeness, not automatically shared proof.
- completion.shared_prefix/shared_suffix are bounded outward extensions repeated across successful Episodes. They complete the procedure without changing Cluster identity.
- completion.activated=false means no repeated outward Step was found; the mandatory procedure must be compiled exactly from Core plus Gap evidence.
- prefix_extension/suffix_extension project the outward context into each Episode. role=shared_extension is positive evidence; role=local_context is read-only.
- provisional_extensions contains successful bounded outward Steps observed in only one successful Episode. They are conditional-only evidence: use them only to produce conditional_guidance, never a mandatory procedure or verification item.
- boundary_context_read_only exposes at most one Step beyond the expanded region on either side. It is never positive procedure or verification evidence.
- episode_context_read_only provides the overall user goal and terminal result for domain/coherence checks. It is context only and never positive procedure or verification evidence.
- existing_skill_read_only, when present, is the previously admitted version of this same capability. Revise that Skill in place: preserve still-supported behavior, incorporate the new evidence delta, and remove claims no longer supported. It is prior context, never citation evidence.
- evidence_delta_read_only describes which successful Episodes were added, retained, or removed since the previous CandidateVersion. It controls the scope of the revision but is never positive procedure or verification evidence.
- evidence_anchor_catalog is the only citation interface. Select semantic anchor_id values from this catalog; never copy opaque Step IDs or evidence_refs into the output. The program binds selected Anchors to exact Step IDs after generation.

Admission rules:
- Admit only a coherent, reusable method repeated in at least two distinct successful Episodes.
- Reject generic commonality such as merely calling a tool, writing a file, running a command, or checking output.
- Keep the smallest complete capability that is useful to invoke. Do not combine independent procedures merely because one large window contains both.
- Preserve both the reference Span-token order and the concrete action order inside every matched Span. Failed actions are counterfactual evidence, never proof of success.
- Use Gaps to detect missing causal bridges such as edit -> rerun -> verify. If equivalent Gaps do not occur in at least two distinct successful Episodes, do not turn them into a mandatory procedure Step.
- If the common Core plus its occurrence-local Gaps does not support one shared causally complete method, reject the candidate instead of hiding the disagreement.
- A mandatory procedure or verification item that uses Gap evidence must cite equivalent Gap step_id values from at least two distinct successful Episodes.
- A mandatory procedure or verification item that uses outward extension evidence must cite shared_extension step_id values from at least two distinct successful Episodes.
- A provisional extension may be useful immediately as conditional guidance. State a precise condition and action, cite only its allowed step_id, and keep it explicitly provisional. Never generalize it into an unconditional requirement.
- Never turn local_context into a procedure or verification Step. It may only explain triggers, boundaries, or why the candidate should be rejected. span_step is eligible evidence, but only when the cited Steps from at least two successful Episodes genuinely support the same abstract claim.
- Never cite boundary_context_read_only as procedure or verification evidence.
- Never cite episode_context_read_only as procedure or verification evidence.
- counterexamples_read_only may inform boundaries and do_not_apply_when only. Never select an Anchor from it as procedure, verification, or support evidence.
- If counterexamples_read_only is empty, do_not_apply_when may be empty. Any exclusion must be a conservative restatement of a positive precondition, not an invented failure mode.
- Every procedure and verification item must select source_anchor_ids from anchor_contract.allowed_mandatory_anchor_ids. The program expands those Anchors to exact Step IDs and verifies that the resulting claim is supported by at least two distinct successful Episodes.
- Gap and long-trajectory span_step Anchors may be occurrence-local. When two Episodes contain semantically equivalent operations, select the corresponding Anchor from each Episode for the same claim; the program verifies cross-Episode support after binding.
- Conditional guidance may select only source_anchor_ids from anchor_contract.allowed_conditional_anchor_ids. Those Anchors are single-Episode provisional evidence and can never support a mandatory claim.
- Never output evidence_refs, raw step_id values, occurrence IDs, reference Span IDs, or invented Anchor IDs as citations.
- For a long-trajectory candidate, every reference Span token must contribute at least one mandatory procedure claim whose selected Anchors bind to concrete Steps in at least two distinct Episodes. Reject a candidate that collapses a multi-Span sequence to only its final local tail.
- Verification must be grounded by successful evidence from at least two distinct Episodes.
- If the shared Core is complete without a provisional extension, admit the Core Skill and place supported single-Episode advice in conditional_guidance. If the Core is incomplete without that extension, reject and wait for more evidence.
- Do not invent tools, constraints, files, checks, results, or causal explanations.
- The program derives evidence occurrence IDs from the selected semantic Anchors.

Return JSON only. Use the common keys below. For aligned_window input, also include the required V2 local_subproblem_closure key described after this schema:
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
    {"title": "...", "body": "...", "source_anchor_ids": ["..."]}
  ],
  "verification_steps": [
    {"check": "...", "success_signal": "...", "source_anchor_ids": ["..."]}
  ],
  "do_not_apply_when": ["..."],
  "decision_guidance": {"preference": ["..."], "anti_pattern": ["..."]},
  "conditional_guidance": [
    {"condition": "...", "action": "...", "source_anchor_ids": ["..."]}
  ],
  "examples": [],
  "tags": ["..."],
  "tools": ["..."],
  "confidence": 0.0
}

For a rejection set admit=false, provide rejection_reason, and return empty strings or arrays for every Skill field. Do not decide whether to create, update, or suppress a Skill. Candidate identity and a separate post-Draft coverage stage own those decisions.`;

const PROCEDURAL_SKILL_COVERAGE_PROMPT = `You compare one already compiled V2 Skill Draft against retrieved Skills for the same user.

This call is read-only. Never rewrite the Draft and never decide whether an existing Candidate should be updated. Candidate identity has already been resolved before this call.

Return covered when at least one retrieved Skill is semantically equivalent to the Draft, or when the Draft is fully contained by that Skill:
- the Draft Trigger is equal to or narrower than the target Trigger;
- every required Draft Procedure operation appears in the same order in the target Procedure;
- the Draft Verification contract is already covered by the target Verification.

Return distinct when the Skills merely share a topic, tools, vocabulary, artifact type, or some operations. A missing causal operation, different entry condition, or different resolved-state/success contract makes the Draft distinct.

The target may be OLD, V2, or V3. target_skill_id must be copied exactly from comparison_skills_read_only. Never invent an id.

Return JSON only:
{
  "decision": "covered | distinct",
  "target_skill_id": null,
  "target_route": null,
  "relation": "equivalent | subset | distinct",
  "reason": "..."
}

For covered, target_skill_id and target_route are required and relation must be equivalent or subset. For distinct, both target fields must be null and relation must be distinct.`;

const V2_LOCAL_SUBPROBLEM_CLOSURE_PROMPT = `
V2 local-subproblem closed-loop admission gate (aligned_window input only):
- Perform this gate before drafting any Skill fields.
- Compare the successful occurrences side by side. Identify exactly one local subproblem that the aligned region repeatedly solves; do not use the overall Episode goal as a substitute.
- A closed loop requires all four parts below in the same set of at least two distinct successful Episodes:
  1. entry_condition: the local problem, unmet condition, or actionable starting state;
  2. resolution: the causally relevant action sequence that changes that state;
  3. resolved_state: the state reached after those actions;
  4. success_check: an observed check showing that this local subproblem, not merely the overall Episode, was successfully resolved.
- Do not stitch a problem from one pair of Episodes, a resolution from another pair, and verification from a third pair. local_subproblem_closure.support_episode_ids must name at least two successful Episodes that each support the whole loop.
- Episode-level terminal success, episode_context_read_only, local_context, or a later unrelated success cannot close the loop. A failed action may establish the problem state but can never prove resolution or verification.
- Gaps or provisional extensions seen in only one Episode cannot supply a missing mandatory action, resolved state, or success check.
- If the aligned evidence shows only setup, inspection, repeated tool use, an attempted action without confirmed resolution, a modification without rerun, or execution without a supported success check, reject it. Set admit=false and use a precise rejection_reason beginning with no-cross-episode-local-closed-loop:.
- Only when local_subproblem_closure.closed=true may you draft the V2 Skill. Coverage comparison happens later in a separate read-only call.

For aligned_window input, add this required top-level object to the JSON response:
"local_subproblem_closure": {
  "closed": true,
  "subproblem": "the reusable local problem being solved",
  "entry_condition": "the repeated local starting/problem state",
  "resolution": "the shared causal action sequence",
  "resolved_state": "the repeated state after resolution",
  "success_check": "the repeated observed verification",
  "support_episode_ids": ["episode id from the input", "another episode id from the input"],
  "reason": "why the same Episodes demonstrate a complete local loop"
}

For an aligned_window rejection, still return local_subproblem_closure with closed=false, accurately fill any parts supported by evidence, leave unsupported text fields empty, and explain the missing link in reason.`;

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
      alignmentGroupId?: string;
      matchSimilarity: number;
    })
  | (ProceduralSkillEvidenceStep & {
      role: "gap";
      afterAnchorId: string;
      beforeAnchorId: string;
    })
  | (ProceduralSkillEvidenceStep & {
      role: "span_step";
      anchorId: string;
      spanSimilarity: number;
    })
  | (ProceduralSkillEvidenceStep & {
      role: "local_context";
      spanAnchorId: string;
    });

export interface ProceduralSkillBoundaryContext {
  previousStep?: ProceduralSkillEvidenceStep;
  nextStep?: ProceduralSkillEvidenceStep;
}

export interface ProceduralSkillExtensionAnchor {
  anchorId: string;
  side: "prefix" | "suffix";
  referenceStepId: string;
  supportEpisodeIds: string[];
  evidenceStepIds: string[];
  averageMatchSimilarity: number;
}

export type ProceduralSkillExpansionStep = ProceduralSkillEvidenceStep & {
  role: "shared_extension" | "local_context";
  side: "prefix" | "suffix";
  extensionAnchorId?: string;
  matchSimilarity?: number;
};

export interface ProceduralSkillCompletion {
  id: string;
  version: string;
  activated: boolean;
  referenceOccurrenceId: string;
  maxPrefixSteps: number;
  maxSuffixSteps: number;
  minStepSimilarity: number;
  sharedPrefix: ProceduralSkillExtensionAnchor[];
  sharedSuffix: ProceduralSkillExtensionAnchor[];
  extensionAgreement: number;
}

export interface ProceduralSkillEvidenceOccurrence {
  occurrenceId: string;
  episodeId: string;
  pathId: string;
  scale: number;
  alignmentScore: number;
  sourceTraceIds: string[];
  prefixExpansion: ProceduralSkillExpansionStep[];
  alignedSequence: ProceduralSkillAlignedSequenceStep[];
  suffixExpansion: ProceduralSkillExpansionStep[];
  boundaryContextReadOnly: ProceduralSkillBoundaryContext;
}

export interface ProceduralSkillConditionalGuidance {
  condition: string;
  action: string;
  sourceAnchorIds: string[];
  evidenceRefs: string[];
  evidenceStatus: "provisional";
  supportEpisodeCount: 1;
}

export interface ProceduralPatternSkillInput {
  patternVersionId: string;
  clusterId: string;
  clusterVersionId: string;
  commonCoreId: string;
  commonCore: ProceduralSkillCommonCoreAnchor[];
  completion: ProceduralSkillCompletion;
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
  episodeContextReadOnly?: Array<{
    episodeId: string;
    goal: string;
    terminalResult: string;
  }>;
  existingSkillReadOnly?: {
    memoryId: string;
    memoryVersion: number;
    name: string;
    invocationGuide: string;
    procedureJson: Record<string, unknown>;
    sourceEpisodeIds: string[];
    sourceTraceIds: string[];
    sourceSpanOccurrenceIds: string[];
  };
  evidenceDeltaReadOnly?: {
    previousCandidateVersionId?: string;
    addedEpisodeIds: string[];
    retainedEpisodeIds: string[];
    removedEpisodeIds: string[];
    currentEvidenceEpisodeIds: string[];
  };
  origin?: {
    kind: "long_trajectory";
    episodeFamilyId: string;
    longTrajectoryId: string;
  };
}

export type ProceduralSkillComparisonRoute = "OLD" | "V2" | "V3";

export interface ProceduralSkillComparisonCandidate {
  memoryId: string;
  route: ProceduralSkillComparisonRoute;
  name: string;
  invocationGuide: string;
  triggerContext: string;
  summary: string;
  procedureSteps: Array<{ title: string; body: string }>;
  verificationSteps: Array<{ title: string; body: string }>;
}

export type ProceduralSkillReuseDecision =
  | { action: "create_v2"; relation: "distinct"; reason: string }
  | {
      action: "update_v2";
      relation: "equivalent";
      reason: string;
      targetSkillId: string;
      targetRoute: "V2";
    };

export type ProceduralSkillCoverageDecision =
  | { decision: "distinct"; relation: "distinct"; reason: string }
  | {
      decision: "covered";
      relation: "equivalent" | "subset";
      reason: string;
      targetSkillId: string;
      targetRoute: ProceduralSkillComparisonRoute;
    };

export type ProceduralSkillCoverageResult =
  | { ok: true; decision: ProceduralSkillCoverageDecision }
  | { ok: false; reason: string; rawDecision?: Record<string, unknown> };

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
  reuseDecision: ProceduralSkillReuseDecision;
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
    const extensionEvidence = classifyExtensionEvidence(evidence);
    const evidenceAnchorCatalog = buildEvidenceAnchorCatalog(
      evidence,
      extensionEvidence
    );

    const effectivePromptVersion = input.origin?.kind === "long_trajectory"
      ? PROCEDURAL_LONG_TRAJECTORY_SKILL_PROMPT_VERSION
      : PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION;
    const systemPrompt = input.origin?.kind === "long_trajectory"
      ? PROCEDURAL_PATTERN_SKILL_PROMPT
      : `${PROCEDURAL_PATTERN_SKILL_PROMPT}\n${V2_LOCAL_SUBPROBLEM_CLOSURE_PROMPT}`;
    const llmResult = await this.deps.skillLlm.completeJson<Record<string, unknown>>([
      { role: "system", content: systemPrompt },
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
            confidence_hint: input.confidenceHint,
            origin: input.origin?.kind === "long_trajectory"
              ? {
                  kind: input.origin.kind,
                  episode_family_id: input.origin.episodeFamilyId,
                  long_trajectory_id: input.origin.longTrajectoryId
                }
              : { kind: "aligned_window" }
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
          completion: {
            id: input.completion.id,
            version: input.completion.version,
            activated: input.completion.activated,
            reference_occurrence_id: input.completion.referenceOccurrenceId,
            max_prefix_steps: input.completion.maxPrefixSteps,
            max_suffix_steps: input.completion.maxSuffixSteps,
            min_step_similarity: input.completion.minStepSimilarity,
            extension_agreement: input.completion.extensionAgreement,
            shared_prefix: input.completion.sharedPrefix.map(serializeExtensionAnchor),
            shared_suffix: input.completion.sharedSuffix.map(serializeExtensionAnchor)
          },
          occurrences: evidence.map((occurrence) => ({
            occurrence_id: occurrence.occurrenceId,
            episode_id: occurrence.episodeId,
            alignment_score: occurrence.alignmentScore,
            boundary_context_read_only: serializeBoundaryContext(
              occurrence.boundaryContextReadOnly
            ),
            prefix_extension: occurrence.prefixExpansion.map((step) =>
              serializeExpansionStep(step, evidenceAnchorCatalog, occurrence.occurrenceId)),
            aligned_sequence: occurrence.alignedSequence.map((step) =>
              serializeAlignedStep(step, evidenceAnchorCatalog, occurrence.occurrenceId)),
            suffix_extension: occurrence.suffixExpansion.map((step) =>
              serializeExpansionStep(step, evidenceAnchorCatalog, occurrence.occurrenceId))
          })),
          provisional_extensions: extensionEvidence.provisionalExtensions.map((item) => ({
            anchor_id: evidenceAnchorCatalog.anchorIdByOccurrenceStep.get(
              occurrenceStepKey(item.occurrenceId, item.step.stepId)
            ),
            occurrence_id: item.occurrenceId,
            episode_id: item.episodeId,
            side: item.side,
            allowed_usage: "conditional_only",
            evidence_status: "provisional",
            support_episode_count: 1,
            step: serializeEvidenceStep(item.step)
          })),
          evidence_anchor_catalog: evidenceAnchorCatalog.anchors.map((anchor) => ({
            anchor_id: anchor.anchorId,
            kind: anchor.kind,
            allowed_usage: anchor.allowedUsage,
            ...(anchor.semanticParentAnchorId
              ? { semantic_parent_anchor_id: anchor.semanticParentAnchorId }
              : {}),
            support_episode_ids: anchor.episodeIds,
            support_episode_count: anchor.episodeIds.length,
            occurrence_ids: anchor.occurrenceIds,
            representative_intent: clip(anchor.representativeIntent, 400),
            representative_summary: clip(anchor.representativeSummary, 600),
            tool_names: anchor.toolNames
          })),
          episode_context_read_only: (input.episodeContextReadOnly ?? [])
            .filter((context) => evidenceEpisodeIds.includes(context.episodeId))
            .map((context) => ({
              episode_id: context.episodeId,
              goal: clip(context.goal, 1_200),
              terminal_result: clip(context.terminalResult, 1_200)
            })),
          existing_skill_read_only: input.existingSkillReadOnly
            ? {
                memory_id: input.existingSkillReadOnly.memoryId,
                memory_version: input.existingSkillReadOnly.memoryVersion,
                name: input.existingSkillReadOnly.name,
                invocation_guide: clip(input.existingSkillReadOnly.invocationGuide, 4_000),
                procedure_json: input.existingSkillReadOnly.procedureJson,
                source_episode_ids: input.existingSkillReadOnly.sourceEpisodeIds,
                source_trace_ids: input.existingSkillReadOnly.sourceTraceIds,
                source_span_occurrence_ids:
                  input.existingSkillReadOnly.sourceSpanOccurrenceIds
              }
            : null,
          evidence_delta_read_only: input.evidenceDeltaReadOnly
            ? {
                previous_candidate_version_id:
                  input.evidenceDeltaReadOnly.previousCandidateVersionId ?? null,
                added_episode_ids: input.evidenceDeltaReadOnly.addedEpisodeIds,
                retained_episode_ids: input.evidenceDeltaReadOnly.retainedEpisodeIds,
                removed_episode_ids: input.evidenceDeltaReadOnly.removedEpisodeIds,
                current_evidence_episode_ids:
                  input.evidenceDeltaReadOnly.currentEvidenceEpisodeIds
              }
            : null,
          anchor_contract: {
            allowed_mandatory_anchor_ids: evidenceAnchorCatalog.anchors
              .filter((anchor) => anchor.allowedUsage === "mandatory")
              .map((anchor) => anchor.anchorId),
            allowed_conditional_anchor_ids: evidenceAnchorCatalog.anchors
              .filter((anchor) => anchor.allowedUsage === "conditional_only")
              .map((anchor) => anchor.anchorId),
            forbidden_failed_step_ids: unique(evidence.flatMap((occurrence) => [
              ...occurrence.alignedSequence,
              ...occurrence.prefixExpansion,
              ...occurrence.suffixExpansion
            ].filter((step) => step.outcome === "failure").map((step) => step.stepId))),
            forbidden_local_context_step_ids: unique(evidence.flatMap((occurrence) =>
              occurrence.alignedSequence.filter((step) =>
                step.role === "local_context").map((step) => step.stepId)))
          },
          counterexamples_read_only: counterexamples.map((occurrence) => ({
            occurrence_id: occurrence.occurrenceId,
            episode_id: occurrence.episodeId,
            boundary_context_read_only: serializeBoundaryContext(
              occurrence.boundaryContextReadOnly
            ),
            prefix_extension: occurrence.prefixExpansion.map((step) =>
              serializeExpansionStep(step)),
            aligned_sequence: occurrence.alignedSequence.map((step) =>
              serializeAlignedStep(step)),
            suffix_extension: occurrence.suffixExpansion.map((step) =>
              serializeExpansionStep(step))
          }))
        })
      }
    ], {
      operation: `procedural.${effectivePromptVersion}`,
      thinkingMode: "enabled",
      temperature: 0,
      maxTokens: 8_192
    });
    if (detectSkillModelRefusal(llmResult)) {
      return { admitted: false, reason: "llm-refusal" };
    }
    const localSubproblemClosure = input.origin?.kind === "long_trajectory"
      ? undefined
      : parseV2LocalSubproblemClosure(llmResult, evidence);
    if (localSubproblemClosure && !localSubproblemClosure.ok) {
      return { admitted: false, reason: localSubproblemClosure.reason };
    }
    const parsed = parseSkillResult(
      llmResult,
      evidence,
      input.origin?.kind === "long_trajectory"
        ? input.commonCore.map((anchor) => anchor.anchorId)
        : [],
      extensionEvidence,
      evidenceAnchorCatalog,
      localSubproblemClosure?.closure
    );
    if (!parsed.ok) return { admitted: false, reason: parsed.reason };

    const confidence = clamp(
      0.7 * parsed.confidence + 0.3 * clamp(input.confidenceHint, 0, 1),
      0,
      1
    );
    const reuseDecision: ProceduralSkillReuseDecision =
      input.origin?.kind !== "long_trajectory" && input.existingSkillReadOnly
        ? {
            action: "update_v2",
            relation: "equivalent",
            reason: "existing-candidate-skill",
            targetSkillId: input.existingSkillReadOnly.memoryId,
            targetRoute: "V2"
          }
        : {
            action: "create_v2",
            relation: "distinct",
            reason: input.origin?.kind === "long_trajectory"
              ? "long-trajectory-materialization"
              : "new-candidate"
          };
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
        scope: consensusSkillScope(sourceTraces.map((trace) => trace.memory)),
        reuseDecision
      }
    };
  }

  async compareDraftCoverage(
    draft: ProceduralPatternSkillDraft,
    candidates: readonly ProceduralSkillComparisonCandidate[]
  ): Promise<ProceduralSkillCoverageResult> {
    if (candidates.length === 0) {
      return {
        ok: true,
        decision: {
          decision: "distinct",
          relation: "distinct",
          reason: "no-similar-skill"
        }
      };
    }
    const result = await this.deps.skillLlm.completeJson<Record<string, unknown>>([
      { role: "system", content: PROCEDURAL_SKILL_COVERAGE_PROMPT },
      {
        role: "user",
        content: stableStringify({
          skill_draft_read_only: serializeSkillDraftForCoverage(draft),
          comparison_skills_read_only: candidates.map(serializeComparisonSkill)
        })
      }
    ], {
      operation: `procedural.${PROCEDURAL_SKILL_COVERAGE_PROMPT_VERSION}`,
      thinkingMode: "enabled",
      temperature: 0,
      maxTokens: 2_048
    });
    if (detectSkillModelRefusal(result)) {
      return { ok: false, reason: "coverage-llm-refusal", rawDecision: result };
    }
    return parseSkillCoverageDecision(result, candidates);
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
    const {
      input,
      parsed,
      evidence,
      sourceTraceIds,
      sourceEpisodeIds,
      gain,
      confidence,
      scope,
      reuseDecision
    } = draft;
    const isLongTrajectory = input.origin?.kind === "long_trajectory";
    const reusedMemory = reuseDecision.action === "update_v2"
      ? this.deps.repos.memories.get(reuseDecision.targetSkillId)
      : undefined;
    if (reuseDecision.action === "update_v2" && (
      !reusedMemory ||
      reusedMemory.userId !== input.userId ||
      reusedMemory.status === "archived" ||
      reusedMemory.properties.internal_info.plugin_algorithm !==
        "procedural.pattern.skill.v1"
    )) {
      throw new Error("V2 Skill update target is no longer eligible");
    }
    const skillKey = reusedMemory?.memoryKey ?? proceduralSkillKey(input.userId, input.clusterId);
    const existingMemory = reusedMemory ?? this.deps.repos.memories.getByKey("Skill", skillKey);
    const existingSkill = existingMemory ? skillMetaFromMemory(existingMemory) : null;
    const existingInternal = existingMemory?.properties.internal_info;
    const materializedSourceTraceIds = unique([
      ...stringArray(existingInternal?.source_trace_ids),
      ...(input.existingSkillReadOnly?.sourceTraceIds ?? []),
      ...sourceTraceIds
    ]);
    const materializedSourceEpisodeIds = unique([
      ...stringArray(existingInternal?.source_episode_ids),
      ...(input.existingSkillReadOnly?.sourceEpisodeIds ?? []),
      ...sourceEpisodeIds
    ]);
    const materializedOccurrenceIds = unique([
      ...stringArray(existingInternal?.source_span_occurrence_ids),
      ...(input.existingSkillReadOnly?.sourceSpanOccurrenceIds ?? []),
      ...parsed.evidenceOccurrenceIds
    ]);
    const materializedClusterIds = unique([
      ...stringArray(existingInternal?.source_cluster_ids),
      ...(typeof existingInternal?.source_cluster_id === "string"
        ? [existingInternal.source_cluster_id]
        : []),
      input.clusterId
    ]);
    const materializedClusterVersionIds = unique([
      ...stringArray(existingInternal?.source_cluster_version_ids),
      ...(typeof existingInternal?.source_cluster_version_id === "string"
        ? [existingInternal.source_cluster_version_id]
        : []),
      input.clusterVersionId
    ]);
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
            materializedSourceEpisodeIds.length /
              Math.max(1, this.deps.config.algorithm.skill.minSupport)
          )
        ), 0, 1);
    const status = existingSkill?.status === "active" ? "active" : "candidate";
    const steps = materializedSkillSteps(parsed, evidence);
    const mandatoryEvidenceRefs = unique([
      ...parsed.procedureSteps.flatMap((step) => step.evidenceRefs),
      ...parsed.verificationSteps.flatMap((step) => step.evidenceRefs)
    ]);
    const conditionalEvidenceRefs = unique(parsed.conditionalGuidance.flatMap((item) =>
      item.evidenceRefs));
    const mandatorySemanticAnchorIds = unique([
      ...parsed.procedureSteps.flatMap((step) => step.sourceAnchorIds),
      ...parsed.verificationSteps.flatMap((step) => step.sourceAnchorIds)
    ]);
    const conditionalSemanticAnchorIds = unique(parsed.conditionalGuidance.flatMap((item) =>
      item.sourceAnchorIds));
    const citedEvidenceRefs = unique([
      ...mandatoryEvidenceRefs,
      ...conditionalEvidenceRefs
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
      conditionalGuidance: parsed.conditionalGuidance,
      doNotApplyWhen: parsed.doNotApplyWhen,
      reliability: {
        supportCount: materializedSourceEpisodeIds.length,
        successRate,
        betaPosterior,
        evidenceConfidence: confidence
      },
      tags: parsed.tags,
      tools: parsed.tools
    };
    const invocationGuide = renderInvocationGuide(
      parsed,
      materializedSourceEpisodeIds.length,
      confidence
    );
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
      tags: unique([
        "skill",
        isLongTrajectory ? "long-trajectory" : "procedural-pattern",
        isLongTrajectory ? "episode-family" : `span-${input.scale}`,
        ...parsed.tags
      ]),
      info: {
        name: parsed.name,
        eta,
        status,
        source_memory_ids: materializedSourceTraceIds,
        source_policy_ids: [],
        evidence_anchor_ids: materializedSourceTraceIds
      },
      internal: {
        source: isLongTrajectory
          ? "worker.long_trajectory_induction.v1"
          : "worker.procedural_skill_induction.v1",
        plugin_algorithm: isLongTrajectory
          ? "procedural.long-trajectory.skill.v1"
          : "procedural.pattern.skill.v1",
        read_only: false,
        generated_by_memory_base: true,
        source_memory_ids: materializedSourceTraceIds,
        source_policy_ids: [],
        source_world_model_ids: [],
        evidence_anchor_ids: materializedSourceTraceIds,
        source_trace_ids: materializedSourceTraceIds,
        source_episode_ids: materializedSourceEpisodeIds,
        source_span_occurrence_ids: materializedOccurrenceIds,
        source_pattern_version_id: input.patternVersionId,
        source_cluster_id: input.clusterId,
        source_cluster_version_id: input.clusterVersionId,
        source_cluster_ids: materializedClusterIds,
        source_cluster_version_ids: materializedClusterVersionIds,
        reuse_decision: reuseDecision,
        ...(input.origin ? {
          source_episode_family_id: input.origin.episodeFamilyId,
          source_long_trajectory_id: input.origin.longTrajectoryId
        } : {}),
        source_completion_id: input.completion.id,
        completion_version: input.completion.version,
        completion_activated: input.completion.activated,
        completion_extension_agreement: input.completion.extensionAgreement,
        completion_shared_prefix_anchor_ids: input.completion.sharedPrefix.map((item) =>
          item.anchorId),
        completion_shared_suffix_anchor_ids: input.completion.sharedSuffix.map((item) =>
          item.anchorId),
        evidence_binding_version: "semantic-anchor-binding.v1",
        selected_mandatory_anchor_ids: mandatorySemanticAnchorIds,
        selected_conditional_anchor_ids: conditionalSemanticAnchorIds,
        provisional_extension_evidence_refs: conditionalEvidenceRefs,
        provisional_extension_count: parsed.conditionalGuidance.length,
        core_fallback_applied: parsed.coreFallbackApplied,
        ...(parsed.localSubproblemClosure ? {
          local_subproblem_closure: {
            closed: true,
            subproblem: parsed.localSubproblemClosure.subproblem,
            entry_condition: parsed.localSubproblemClosure.entryCondition,
            resolution: parsed.localSubproblemClosure.resolution,
            resolved_state: parsed.localSubproblemClosure.resolvedState,
            success_check: parsed.localSubproblemClosure.successCheck,
            support_episode_ids: parsed.localSubproblemClosure.supportEpisodeIds,
            reason: parsed.localSubproblemClosure.reason
          }
        } : {}),
        window_scale: input.scale,
        alignment_config_version: input.algorithmVersion,
        induction_prompt_version: input.origin?.kind === "long_trajectory"
          ? PROCEDURAL_LONG_TRAJECTORY_SKILL_PROMPT_VERSION
          : PROCEDURAL_PATTERN_SKILL_PROMPT_VERSION,
        counterexample_episode_ids: input.counterexampleEpisodeIds,
        pattern_content_hash: input.patternHash,
        name: parsed.name,
        invocation_guide: invocationGuide,
        procedure_json: procedureJson,
        eta,
        support: materializedSourceEpisodeIds.length,
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
          support: materializedSourceEpisodeIds.length,
          gain,
          source_policy_ids: [],
          source_world_model_ids: [],
          evidence_anchor_ids: materializedSourceTraceIds,
          invocation_guide: invocationGuide,
          procedure_json: procedureJson,
          trials_attempted: trialsAttempted,
          trials_passed: trialsPassed,
          success_rate: successRate,
          beta_posterior: betaPosterior,
          verification: {
            ok: true,
            evidence_occurrence_ids: materializedOccurrenceIds,
            evidence_refs: mandatoryEvidenceRefs
          },
          vec: null
        }
      },
      createdAt
    });
    return {
      memory: skill,
      contentHash,
      sourceEpisodeIds: materializedSourceEpisodeIds,
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
  step: ProceduralSkillAlignedSequenceStep,
  catalog?: EvidenceAnchorCatalog,
  occurrenceId?: string
): Record<string, unknown> {
  const evidenceAnchorId = catalog && occurrenceId
    ? catalog.anchorIdByOccurrenceStep.get(occurrenceStepKey(occurrenceId, step.stepId))
    : undefined;
  switch (step.role) {
    case "core":
      return {
        role: "core",
        anchor_id: step.anchorId,
        ...(evidenceAnchorId ? { evidence_anchor_id: evidenceAnchorId } : {}),
        ...(step.alignmentGroupId
          ? { alignment_group_id: step.alignmentGroupId }
          : {}),
        match_similarity: step.matchSimilarity,
        ...serializeEvidenceStep(step)
      };
    case "gap":
      return {
        role: "gap",
        ...(evidenceAnchorId ? { evidence_anchor_id: evidenceAnchorId } : {}),
        after_anchor_id: step.afterAnchorId,
        before_anchor_id: step.beforeAnchorId,
        ...serializeEvidenceStep(step)
      };
    case "span_step":
      return {
        role: "span_step",
        anchor_id: step.anchorId,
        ...(evidenceAnchorId ? { evidence_anchor_id: evidenceAnchorId } : {}),
        coarse_span_similarity: step.spanSimilarity,
        ...serializeEvidenceStep(step)
      };
    case "local_context":
      return {
        role: "local_context",
        span_anchor_id: step.spanAnchorId,
        ...serializeEvidenceStep(step)
      };
  }
}

function serializeExtensionAnchor(
  anchor: ProceduralSkillExtensionAnchor
): Record<string, unknown> {
  return {
    anchor_id: anchor.anchorId,
    side: anchor.side,
    reference_step_id: anchor.referenceStepId,
    support_episode_ids: anchor.supportEpisodeIds,
    evidence_step_ids: anchor.evidenceStepIds,
    average_match_similarity: anchor.averageMatchSimilarity
  };
}

function serializeExpansionStep(
  step: ProceduralSkillExpansionStep,
  catalog?: EvidenceAnchorCatalog,
  occurrenceId?: string
): Record<string, unknown> {
  const evidenceAnchorId = catalog && occurrenceId
    ? catalog.anchorIdByOccurrenceStep.get(occurrenceStepKey(occurrenceId, step.stepId))
    : undefined;
  return {
    role: step.role,
    side: step.side,
    ...(evidenceAnchorId ? { evidence_anchor_id: evidenceAnchorId } : {}),
    ...(step.extensionAnchorId
      ? { extension_anchor_id: step.extensionAnchorId }
      : {}),
    ...(step.matchSimilarity === undefined
      ? {}
      : { match_similarity: step.matchSimilarity }),
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
  procedureSteps: Array<{
    title: string;
    body: string;
    sourceAnchorIds: string[];
    evidenceRefs: string[];
  }>;
  verificationSteps: Array<{
    check: string;
    successSignal: string;
    sourceAnchorIds: string[];
    evidenceRefs: string[];
  }>;
  doNotApplyWhen: string[];
  decisionGuidance: { preference: string[]; antiPattern: string[] };
  conditionalGuidance: ProceduralSkillConditionalGuidance[];
  coreFallbackApplied: boolean;
  examples: unknown[];
  tags: string[];
  tools: string[];
  evidenceOccurrenceIds: string[];
  confidence: number;
  localSubproblemClosure?: V2LocalSubproblemClosure;
}

interface V2LocalSubproblemClosure {
  closed: true;
  subproblem: string;
  entryCondition: string;
  resolution: string;
  resolvedState: string;
  successCheck: string;
  supportEpisodeIds: string[];
  reason: string;
}

interface ProvisionalExtensionEvidence {
  occurrenceId: string;
  episodeId: string;
  side: "prefix" | "suffix";
  step: ProceduralSkillExpansionStep;
}

interface ExtensionEvidenceClassification {
  admittedSharedRefs: Set<string>;
  admittedSharedEpisodesByRef: Map<string, string[]>;
  provisionalRefs: Set<string>;
  provisionalEpisodesByRef: Map<string, string[]>;
  provisionalExtensions: ProvisionalExtensionEvidence[];
}

type EvidenceAnchorKind =
  | "core"
  | "gap"
  | "span_step"
  | "shared_extension"
  | "provisional_extension";

interface EvidenceAnchorBinding {
  anchorId: string;
  kind: EvidenceAnchorKind;
  allowedUsage: "mandatory" | "conditional_only";
  semanticParentAnchorId?: string;
  episodeIds: string[];
  occurrenceIds: string[];
  stepIds: string[];
  toolNames: string[];
  representativeIntent: string;
  representativeSummary: string;
}

interface EvidenceAnchorCatalog {
  anchors: EvidenceAnchorBinding[];
  byId: Map<string, EvidenceAnchorBinding>;
  anchorIdByOccurrenceStep: Map<string, string>;
}

interface RawAnchoredClaim {
  sourceAnchorIds: string[];
}

interface BoundAnchoredClaim {
  sourceAnchorIds: string[];
  evidenceRefs: string[];
  episodeIds: string[];
  occurrenceIds: string[];
  bindings: EvidenceAnchorBinding[];
}

function materializedSkillSteps(
  parsed: ParsedSkill,
  evidence: ProceduralSkillEvidenceOccurrence[]
): Array<Record<string, unknown>> {
  const occurrencesByRef = new Map<string, string[]>();
  for (const occurrence of evidence) {
    const refs = [
      occurrence.occurrenceId,
      ...occurrence.prefixExpansion.flatMap((step) => [step.stepId, ...step.evidenceRefs]),
      ...occurrence.alignedSequence.flatMap((step) => [step.stepId, ...step.evidenceRefs]),
      ...occurrence.suffixExpansion.flatMap((step) => [step.stepId, ...step.evidenceRefs])
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
    sourceAnchorIds: string[],
    evidenceRefs: string[]
  ): Record<string, unknown> => ({
    id: `step_${stableHash({ kind, title, body }).slice(0, 12)}`,
    title,
    body,
    sourceAnchorIds,
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
      step.sourceAnchorIds,
      step.evidenceRefs
    )),
    ...parsed.verificationSteps.map((step) => build(
      "verification",
      `Verify: ${step.check}`,
      `Success signal: ${step.successSignal}`,
      step.sourceAnchorIds,
      step.evidenceRefs
    ))
  ];
}

function parseSkillCoverageDecision(
  value: Record<string, unknown>,
  candidates: readonly ProceduralSkillComparisonCandidate[]
): ProceduralSkillCoverageResult {
  const decision = skillText(value.decision);
  const relation = skillText(value.relation);
  const reason = skillText(value.reason) || "model-coverage-decision";
  const targetSkillId = skillText(value.target_skill_id);
  const target = targetSkillId
    ? candidates.find((candidate) => candidate.memoryId === targetSkillId)
    : undefined;
  if (decision === "distinct" && relation === "distinct" && !targetSkillId) {
    return {
      ok: true,
      decision: { decision, relation, reason }
    };
  }
  if (decision === "covered" &&
      (relation === "equivalent" || relation === "subset") && target) {
    return {
      ok: true,
      decision: {
        decision,
        relation,
        reason,
        targetSkillId: target.memoryId,
        targetRoute: target.route
      }
    };
  }
  return { ok: false, reason: "invalid-coverage-decision", rawDecision: value };
}

function serializeSkillDraftForCoverage(draft: ProceduralPatternSkillDraft): {
  name: string;
  trigger_context: string;
  summary: string;
  procedure_steps: Array<{ title: string; body: string }>;
  verification_steps: Array<{ title: string; body: string }>;
} {
  return {
    name: draft.parsed.name,
    trigger_context: draft.parsed.triggerContext,
    summary: draft.parsed.summary,
    procedure_steps: draft.parsed.procedureSteps.map((step) => ({
      title: step.title,
      body: step.body
    })),
    verification_steps: draft.parsed.verificationSteps.map((step) => ({
      title: step.check,
      body: step.successSignal
    }))
  };
}

function serializeComparisonSkill(candidate: ProceduralSkillComparisonCandidate): {
  memory_id: string;
  route: ProceduralSkillComparisonRoute;
  name: string;
  invocation_guide: string;
  trigger_context: string;
  summary: string;
  procedure_steps: Array<{ title: string; body: string }>;
  verification_steps: Array<{ title: string; body: string }>;
} {
  return {
    memory_id: candidate.memoryId,
    route: candidate.route,
    name: candidate.name,
    invocation_guide: clip(candidate.invocationGuide, 4_000),
    trigger_context: clip(candidate.triggerContext, 1_200),
    summary: clip(candidate.summary, 1_200),
    procedure_steps: candidate.procedureSteps.map((step) => ({
      title: clip(step.title, 300),
      body: clip(step.body, 1_000)
    })),
    verification_steps: candidate.verificationSteps.map((step) => ({
      title: clip(step.title, 300),
      body: clip(step.body, 1_000)
    }))
  };
}

function parseSkillResult(
  result: Record<string, unknown>,
  evidence: ProceduralSkillEvidenceOccurrence[],
  requiredProcedureAnchorIds: readonly string[] = [],
  extensionEvidence: ExtensionEvidenceClassification = classifyExtensionEvidence(evidence),
  evidenceAnchorCatalog: EvidenceAnchorCatalog = buildEvidenceAnchorCatalog(
    evidence,
    extensionEvidence
  ),
  localSubproblemClosure?: V2LocalSubproblemClosure
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
  const rawProcedureSteps = parseProcedureSteps(result.procedure_steps);
  const rawVerificationSteps = parseVerificationSteps(result.verification_steps);
  const rawMandatoryItems = [...rawProcedureSteps, ...rawVerificationSteps];
  if (rawMandatoryItems.some((item) => item.sourceAnchorIds.some((anchorId) =>
    !evidenceAnchorCatalog.byId.has(anchorId)))) {
    return { ok: false, reason: "invalid-evidence-anchor" };
  }
  const mandatoryUsesProvisional = (item: RawAnchoredClaim) =>
    item.sourceAnchorIds.some((anchorId) =>
      evidenceAnchorCatalog.byId.get(anchorId)?.allowedUsage === "conditional_only");
  const removedProvisionalMandatoryItems = [
    ...rawProcedureSteps,
    ...rawVerificationSteps
  ].filter(mandatoryUsesProvisional).length;
  const retainedProcedureSteps = rawProcedureSteps.filter((item) =>
    !mandatoryUsesProvisional(item));
  const retainedVerificationSteps = rawVerificationSteps.filter((item) =>
    !mandatoryUsesProvisional(item));
  const boundProcedureClaims = retainedProcedureSteps.map((item) =>
    bindAnchoredClaim(item, evidenceAnchorCatalog));
  const boundVerificationClaims = retainedVerificationSteps.map((item) =>
    bindAnchoredClaim(item, evidenceAnchorCatalog));
  const procedureSteps = retainedProcedureSteps.map((item, index) => ({
    ...item,
    evidenceRefs: boundProcedureClaims[index]!.evidenceRefs
  }));
  const verificationSteps = retainedVerificationSteps.map((item, index) => ({
    ...item,
    evidenceRefs: boundVerificationClaims[index]!.evidenceRefs
  }));
  const doNotApplyWhen = markdownArray(result.do_not_apply_when);
  const decisionGuidance = parseDecisionGuidance(result.decision_guidance);
  const conditionalGuidance = parseConditionalGuidance(
    result.conditional_guidance,
    evidenceAnchorCatalog
  );
  const examples = coerceSkillExamples(result.examples);
  const tags = textArray(result.tags);
  const tools = textArray(result.tools);
  if (!name || !displayTitle || !retrievalBlurb || !triggerContext || !summary ||
      confidence === undefined || preconditions.length === 0 || procedureSteps.length === 0 ||
      verificationSteps.length === 0) {
    return {
      ok: false,
      reason: removedProvisionalMandatoryItems > 0
        ? "core-incomplete-after-provisional-extension-fallback"
        : "invalid-skill-fields"
    };
  }
  const mandatoryClaims = [...boundProcedureClaims, ...boundVerificationClaims];
  if (mandatoryClaims.some((claim) => claim.evidenceRefs.length === 0)) {
    return { ok: false, reason: "invalid-evidence-anchor" };
  }
  for (const claim of mandatoryClaims) {
    const gapEpisodes = unique(claim.bindings
      .filter((binding) => binding.kind === "gap")
      .flatMap((binding) => binding.episodeIds));
    if (gapEpisodes.length === 1) {
      return { ok: false, reason: "insufficient-gap-episode-support" };
    }
    const spanEpisodes = unique(claim.bindings
      .filter((binding) => binding.kind === "span_step")
      .flatMap((binding) => binding.episodeIds));
    if (spanEpisodes.length === 1) {
      return { ok: false, reason: "insufficient-claim-episode-support" };
    }
    if (claim.episodeIds.length < 2) {
      return { ok: false, reason: "insufficient-claim-episode-support" };
    }
  }
  const evidenceOccurrenceIds = unique(mandatoryClaims.flatMap((claim) =>
    claim.occurrenceIds));
  const citedEpisodes = unique(mandatoryClaims.flatMap((claim) => claim.episodeIds));
  if (citedEpisodes.length < 2 || evidenceOccurrenceIds.length < 2) {
    return { ok: false, reason: "insufficient-cited-episodes" };
  }
  if (requiredProcedureAnchorIds.length > 0) {
    for (const anchorId of unique([...requiredProcedureAnchorIds])) {
      const supportingEpisodes = unique(boundProcedureClaims.flatMap((claim) =>
        claim.bindings
          .filter((binding) => binding.semanticParentAnchorId === anchorId)
          .flatMap((binding) => binding.episodeIds)));
      if (supportingEpisodes.length < 2) {
        return { ok: false, reason: "incomplete-reference-span-coverage" };
      }
    }
  }
  const verificationEpisodes = unique(boundVerificationClaims.flatMap((claim) =>
    claim.episodeIds));
  if (verificationEpisodes.length < 2) {
    return { ok: false, reason: "insufficient-verification-episode-support" };
  }
  const selectedAnchorIds = unique([
    ...mandatoryClaims.flatMap((claim) => claim.sourceAnchorIds),
    ...conditionalGuidance.flatMap((item) => item.sourceAnchorIds)
  ]);
  const evidenceTools = new Set(selectedAnchorIds.flatMap((anchorId) =>
    evidenceAnchorCatalog.byId.get(anchorId)?.toolNames ?? [])
    .map((tool) => tool.trim().toLowerCase()));
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
    conditionalGuidance,
    coreFallbackApplied: removedProvisionalMandatoryItems > 0,
    examples,
    tags,
    tools,
    evidenceOccurrenceIds,
    confidence: clamp(confidence, 0, 1),
    ...(localSubproblemClosure ? { localSubproblemClosure } : {})
  };
}

function parseV2LocalSubproblemClosure(
  result: Record<string, unknown>,
  evidence: ProceduralSkillEvidenceOccurrence[]
): { ok: true; closure?: V2LocalSubproblemClosure } | { ok: false; reason: string } {
  if (result.admit !== true) return { ok: true };
  const value = result.local_subproblem_closure;
  if (!isRecord(value)) {
    return { ok: false, reason: "missing-v2-local-subproblem-closure" };
  }
  if (value.closed !== true) {
    return { ok: false, reason: "no-cross-episode-local-closed-loop" };
  }
  const subproblem = skillText(value.subproblem);
  const entryCondition = skillText(value.entry_condition);
  const resolution = skillText(value.resolution);
  const resolvedState = skillText(value.resolved_state);
  const successCheck = skillText(value.success_check);
  const reason = skillText(value.reason);
  if (!subproblem || !entryCondition || !resolution || !resolvedState ||
      !successCheck || !reason) {
    return { ok: false, reason: "incomplete-v2-local-subproblem-closure" };
  }
  const evidenceEpisodeIds = new Set(evidence.map((item) => item.episodeId));
  const rawSupportEpisodeIds = unique(stringArray(value.support_episode_ids));
  if (rawSupportEpisodeIds.length < 2 ||
      rawSupportEpisodeIds.some((episodeId) => !evidenceEpisodeIds.has(episodeId))) {
    return { ok: false, reason: "invalid-v2-local-subproblem-closure-support" };
  }
  return {
    ok: true,
    closure: {
      closed: true,
      subproblem,
      entryCondition,
      resolution,
      resolvedState,
      successCheck,
      supportEpisodeIds: rawSupportEpisodeIds,
      reason
    }
  };
}

function parseProcedureSteps(value: unknown): Array<{
  title: string;
  body: string;
  sourceAnchorIds: string[];
}> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const title = skillText(item.title) || undefined;
    const body = skillMarkdown(item.body) || undefined;
    const sourceAnchorIds = stringArray(item.source_anchor_ids);
    return title && body && sourceAnchorIds.length > 0
      ? [{ title, body, sourceAnchorIds }]
      : [];
  });
}

function parseVerificationSteps(value: unknown): Array<{
  check: string;
  successSignal: string;
  sourceAnchorIds: string[];
}> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const check = skillText(item.check) || undefined;
    const successSignal = skillMarkdown(item.success_signal) || undefined;
    const sourceAnchorIds = stringArray(item.source_anchor_ids);
    return check && successSignal && sourceAnchorIds.length > 0
      ? [{ check, successSignal, sourceAnchorIds }]
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

function parseConditionalGuidance(
  value: unknown,
  evidenceAnchorCatalog: EvidenceAnchorCatalog
): ProceduralSkillConditionalGuidance[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!isRecord(item)) return [];
    const condition = skillMarkdown(item.condition) || undefined;
    const action = skillMarkdown(item.action) || undefined;
    const sourceAnchorIds = stringArray(item.source_anchor_ids);
    const bindings = sourceAnchorIds.map((anchorId) =>
      evidenceAnchorCatalog.byId.get(anchorId));
    if (!condition || !action || sourceAnchorIds.length === 0 ||
        bindings.some((binding) => !binding ||
          binding.allowedUsage !== "conditional_only")) {
      return [];
    }
    const resolvedBindings = bindings.filter((binding): binding is EvidenceAnchorBinding =>
      Boolean(binding));
    const supportEpisodeIds = unique(resolvedBindings.flatMap((binding) =>
      binding.episodeIds));
    if (supportEpisodeIds.length !== 1) return [];
    return [{
      condition,
      action,
      sourceAnchorIds,
      evidenceRefs: unique(resolvedBindings.flatMap((binding) => binding.stepIds)),
      evidenceStatus: "provisional" as const,
      supportEpisodeCount: 1 as const
    }];
  });
}

function classifyExtensionEvidence(
  evidence: ProceduralSkillEvidenceOccurrence[]
): ExtensionEvidenceClassification {
  const sharedByAnchor = new Map<string, Array<{
    occurrenceId: string;
    episodeId: string;
    side: "prefix" | "suffix";
    step: ProceduralSkillExpansionStep;
  }>>();
  const provisionalExtensions: ProvisionalExtensionEvidence[] = [];
  for (const occurrence of evidence) {
    for (const step of [...occurrence.prefixExpansion, ...occurrence.suffixExpansion]) {
      if (step.outcome === "failure") continue;
      if (step.role === "shared_extension" && step.extensionAnchorId) {
        sharedByAnchor.set(step.extensionAnchorId, [
          ...(sharedByAnchor.get(step.extensionAnchorId) ?? []),
          {
            occurrenceId: occurrence.occurrenceId,
            episodeId: occurrence.episodeId,
            side: step.side,
            step
          }
        ]);
      } else if (step.role === "local_context") {
        provisionalExtensions.push({
          occurrenceId: occurrence.occurrenceId,
          episodeId: occurrence.episodeId,
          side: step.side,
          step
        });
      }
    }
  }

  const admittedSharedRefs = new Set<string>();
  const admittedSharedEpisodesByRef = new Map<string, string[]>();
  for (const entries of sharedByAnchor.values()) {
    const supportEpisodeIds = unique(entries.map((item) => item.episodeId));
    if (supportEpisodeIds.length >= 2) {
      for (const item of entries) {
        for (const ref of [item.step.stepId, ...item.step.evidenceRefs]) {
          admittedSharedRefs.add(ref);
          admittedSharedEpisodesByRef.set(ref, unique([
            ...(admittedSharedEpisodesByRef.get(ref) ?? []),
            item.episodeId
          ]));
        }
      }
    } else {
      provisionalExtensions.push(...entries);
    }
  }

  const orderedProvisional = uniqueBy(
    provisionalExtensions.sort((left, right) =>
      left.episodeId.localeCompare(right.episodeId) ||
      left.step.stepIndex - right.step.stepIndex ||
      left.step.stepId.localeCompare(right.step.stepId)),
    (item) => `${item.occurrenceId}:${item.step.stepId}`
  );
  const provisionalRefs = new Set<string>();
  const provisionalEpisodesByRef = new Map<string, string[]>();
  for (const item of orderedProvisional) {
    provisionalRefs.add(item.step.stepId);
    provisionalEpisodesByRef.set(item.step.stepId, [item.episodeId]);
  }
  return {
    admittedSharedRefs,
    admittedSharedEpisodesByRef,
    provisionalRefs,
    provisionalEpisodesByRef,
    provisionalExtensions: orderedProvisional
  };
}

function buildEvidenceAnchorCatalog(
  evidence: ProceduralSkillEvidenceOccurrence[],
  extensionEvidence: ExtensionEvidenceClassification
): EvidenceAnchorCatalog {
  const byId = new Map<string, EvidenceAnchorBinding>();
  const anchorIdByOccurrenceStep = new Map<string, string>();
  const add = (
    anchorId: string,
    kind: EvidenceAnchorKind,
    allowedUsage: EvidenceAnchorBinding["allowedUsage"],
    occurrence: ProceduralSkillEvidenceOccurrence,
    step: ProceduralSkillEvidenceStep,
    semanticParentAnchorId?: string
  ) => {
    if (step.outcome === "failure") return;
    const existing = byId.get(anchorId);
    const toolName = step.toolName?.trim();
    const next: EvidenceAnchorBinding = existing
      ? {
          ...existing,
          episodeIds: unique([...existing.episodeIds, occurrence.episodeId]),
          occurrenceIds: unique([...existing.occurrenceIds, occurrence.occurrenceId]),
          stepIds: unique([...existing.stepIds, step.stepId]),
          toolNames: unique([...existing.toolNames, ...(toolName ? [toolName] : [])])
        }
      : {
          anchorId,
          kind,
          allowedUsage,
          ...(semanticParentAnchorId ? { semanticParentAnchorId } : {}),
          episodeIds: [occurrence.episodeId],
          occurrenceIds: [occurrence.occurrenceId],
          stepIds: [step.stepId],
          toolNames: toolName ? [toolName] : [],
          representativeIntent: step.intent,
          representativeSummary: step.summary
        };
    byId.set(anchorId, next);
    anchorIdByOccurrenceStep.set(
      occurrenceStepKey(occurrence.occurrenceId, step.stepId),
      anchorId
    );
  };

  for (const occurrence of evidence) {
    for (const step of occurrence.alignedSequence) {
      if (step.role === "core") {
        add(step.anchorId, "core", "mandatory", occurrence, step, step.anchorId);
      } else if (step.role === "gap") {
        add(
          `gap_anchor_${stableHash({
            occurrenceId: occurrence.occurrenceId,
            stepId: step.stepId
          }).slice(0, 16)}`,
          "gap",
          "mandatory",
          occurrence,
          step
        );
      } else if (step.role === "span_step") {
        add(
          `span_step_anchor_${stableHash({
            occurrenceId: occurrence.occurrenceId,
            stepId: step.stepId
          }).slice(0, 16)}`,
          "span_step",
          "mandatory",
          occurrence,
          step,
          step.anchorId
        );
      }
    }
    for (const step of [...occurrence.prefixExpansion, ...occurrence.suffixExpansion]) {
      if (step.role !== "shared_extension" || !step.extensionAnchorId ||
          !extensionEvidence.admittedSharedRefs.has(step.stepId)) continue;
      add(
        step.extensionAnchorId,
        "shared_extension",
        "mandatory",
        occurrence,
        step,
        step.extensionAnchorId
      );
    }
  }

  for (const extension of extensionEvidence.provisionalExtensions) {
    const occurrence = evidence.find((item) =>
      item.occurrenceId === extension.occurrenceId);
    if (!occurrence) continue;
    add(
      `provisional_anchor_${stableHash({
        occurrenceId: extension.occurrenceId,
        stepId: extension.step.stepId
      }).slice(0, 16)}`,
      "provisional_extension",
      "conditional_only",
      occurrence,
      extension.step,
      extension.step.extensionAnchorId
    );
  }

  const anchors = [...byId.values()].filter((anchor) =>
    anchor.kind !== "core" || anchor.episodeIds.length >= 2).sort((left, right) =>
    left.allowedUsage.localeCompare(right.allowedUsage) ||
    left.kind.localeCompare(right.kind) ||
    left.anchorId.localeCompare(right.anchorId));
  const allowedAnchorIds = new Set(anchors.map((anchor) => anchor.anchorId));
  return {
    anchors,
    byId: new Map(anchors.map((anchor) => [anchor.anchorId, anchor])),
    anchorIdByOccurrenceStep: new Map([...anchorIdByOccurrenceStep]
      .filter(([, anchorId]) => allowedAnchorIds.has(anchorId)))
  };
}

function bindAnchoredClaim(
  claim: RawAnchoredClaim,
  catalog: EvidenceAnchorCatalog
): BoundAnchoredClaim {
  const sourceAnchorIds = unique(claim.sourceAnchorIds);
  const bindings = sourceAnchorIds
    .map((anchorId) => catalog.byId.get(anchorId))
    .filter((binding): binding is EvidenceAnchorBinding => Boolean(binding));
  return {
    sourceAnchorIds,
    evidenceRefs: unique(bindings.flatMap((binding) => binding.stepIds)),
    episodeIds: unique(bindings.flatMap((binding) => binding.episodeIds)),
    occurrenceIds: unique(bindings.flatMap((binding) => binding.occurrenceIds)),
    bindings
  };
}

function occurrenceStepKey(occurrenceId: string, stepId: string): string {
  return `${occurrenceId}:${stepId}`;
}

function occurrenceHasSuccess(occurrence: ProceduralSkillEvidenceOccurrence): boolean {
  return occurrence.alignedSequence.some((step) =>
    step.role !== "local_context" && step.outcome === "success");
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
    ...(parsed.conditionalGuidance.length > 0 ? [
      "",
      "## Conditional guidance (provisional)",
      ...parsed.conditionalGuidance.map((item) =>
        `- **If ${item.condition}**: ${item.action} ` +
        `(supported by ${item.supportEpisodeCount} Episode; provisional)`)
    ] : []),
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

function uniqueBy<T>(values: T[], key: (value: T) => string): T[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    const identity = key(value);
    if (seen.has(identity)) return false;
    seen.add(identity);
    return true;
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
