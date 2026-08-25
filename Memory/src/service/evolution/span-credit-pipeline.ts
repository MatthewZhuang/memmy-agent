import type { LlmClient, LlmMessage } from "../../model/types.js";
import type {
  EpisodeRecord,
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type {
  EpisodeSpanCreditRunRecord,
  SaveEpisodeSpanCreditRunResult
} from "../../storage/procedural-span-credit-repository.js";
import type { ProceduralSpanOccurrenceRecord } from "../../storage/procedural-path-repository.js";
import { stableHash, stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { clip } from "../../utils/text.js";
import { nowIso } from "../../utils/time.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import type { ObservedStateV1 } from "./span-v3-model.js";
import {
  PROCEDURAL_SPAN_EMBEDDING_VERSION,
  PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
} from "./procedural-span-clustering.js";
import {
  SPAN_CREDIT_ALGORITHM_VERSION,
  buildEpisodeSpanCreditRun,
  type SpanCreditAttributionType,
  type SpanCreditEvidenceRole,
  type SpanCreditInputCompactionV1
} from "./span-credit-model.js";
import type { EpisodeProceduralPathV2, ExecutionStepV1 } from "./procedural-path-model.js";

export const SPAN_CREDIT_SCORING_OPERATION = "span_credit.score.v2" as const;
const MAX_REPAIR_ATTEMPTS = 2;
const POSITIVE_CREDIT_THRESHOLD = 0.1;
const NEGATIVE_CREDIT_THRESHOLD = -0.1;
const MIN_ROLE_CONFIDENCE = 0.6;
const MAX_STEPS_PER_SPAN = 10;
export const SPAN_CREDIT_MAX_INPUT_CHARS = 48_000;
const SPAN_CREDIT_MIN_CONTENT_TOKENS = 1_400;
const SPAN_CREDIT_CONTENT_TOKENS_PER_SPAN = 220;
const SPAN_CREDIT_REASONING_RESERVE_TOKENS = 3_000;

export const SPAN_CREDIT_SCORING_PROMPT = `Assign one Episode's final reward jointly across all ordered procedural Span occurrences.

The input contains the Episode goal, the authoritative episode_reward, ordered boundary states, and every Final Span. A Span is a complete local subproblem lifecycle; recovered failed attempts remain inside that Span. Evaluate the whole path in one global comparison. Do not score Spans independently.

For every Span, estimate its signed causal contribution by asking: if this Span's decision and resulting state transition were removed or replaced by a competent alternative, would the final Episode outcome become worse, better, or materially unchanged?

Reward budget rules:
- reward_credit is in [-1, 1]. Positive helped the final outcome; negative harmed it; near zero was incidental.
- The sum of reward_credit across all returned Spans MUST equal episode_reward exactly (within 0.00001).
- Positive and negative credits may coexist even when episode_reward is zero or negative.
- Do not multiply reward_credit by confidence. confidence reports attribution certainty only.
- Root-cause evidence, resolved uncertainty, produced artifacts, and verification may receive positive credit when they causally enabled the result.
- Avoidable detours, incorrect modifications, unsupported claims, and regressions may receive negative credit even if the Episode later recovered.

attribution_type rules:
- helpful: the agent-controlled Span made a positive causal contribution.
- harmful: an agent-controlled choice made the final result worse.
- externally_blocked: progress was limited by an external dependency or permission, not a reusable agent failure.
- neutral: removing the Span would not materially change the result.
- uncertain: the supplied evidence cannot support a stable attribution.

Grounding rules:
- Return every supplied occurrence_id exactly once and preserve their order.
- evidence_refs may contain only IDs or evidence references supplied in the input.
- Do not invent actions, outcomes, feedback, or causal claims.
- Return JSON only with exactly this shape:
{
  "span_credits": [
    {
      "occurrence_id": "...",
      "reward_credit": 0.25,
      "attribution_type": "helpful",
      "confidence": 0.8,
      "evidence_refs": ["..."],
      "reason": "..."
    }
  ]
}`;

export interface SpanCreditPipelineDeps {
  repos: Repositories;
  skillLlm: LlmClient;
  enableThinking?: boolean;
  maxInputChars?: number;
  enqueueJob?(input: EnqueueJobInput): EvolutionJobRecord;
}

interface ParsedSpanCreditResult {
  occurrenceId: string;
  rewardCredit: number;
  attributionType: SpanCreditAttributionType;
  confidence: number;
  evidenceRefs: string[];
  reason: string;
}

interface ParsedScoringResult {
  credits: ParsedSpanCreditResult[];
}

interface SpanCreditLlmResult extends Record<string, unknown> {
  span_credits?: unknown;
}

interface BoundaryState {
  boundaryIndex: number;
  state: ObservedStateV1;
}

interface ScoringContext {
  episode: EpisodeRecord;
  path: EpisodeProceduralPathV2;
  pathId: string;
  pathHash: string;
  namespaceId: string;
  occurrences: ProceduralSpanOccurrenceRecord[];
  boundaries: BoundaryState[];
  episodeReward: number;
  rewardHash: string;
  validEvidenceRefs: Set<string>;
  payload: Record<string, unknown>;
  inputCompaction: SpanCreditInputCompactionV1;
}

export class SpanCreditPipeline {
  constructor(private readonly deps: SpanCreditPipelineDeps) {}

  async scoreJob(job: EvolutionJobRecord): Promise<SaveEpisodeSpanCreditRunResult | undefined> {
    if (!job.episodeId) throw new Error(`SpanCredit job missing episodeId: ${job.id}`);
    const pathId = text(job.payload.pathId);
    const activePath = this.deps.repos.proceduralPaths.getActiveForEpisode(job.episodeId);
    const episode = this.deps.repos.runtime.getEpisode(job.episodeId);
    if (!activePath || !episode || (pathId && activePath.id !== pathId)) return undefined;
    const queuedPathHash = text(job.payload.pathHash);
    const queuedRewardHash = text(job.payload.rewardHash);
    if (
      (queuedPathHash && queuedPathHash !== activePath.pathHash) ||
      (queuedRewardHash && queuedRewardHash !== spanCreditRewardHash(episode))
    ) return undefined;
    return this.scoreEpisode({ episodeId: job.episodeId, pathId, at: job.updatedAt });
  }

  async scoreEpisode(input: {
    episodeId: string;
    pathId?: string;
    at?: string;
  }): Promise<SaveEpisodeSpanCreditRunResult> {
    const context = this.buildContext(input.episodeId, input.pathId);
    const existing = this.deps.repos.proceduralSpanCredits.getByBasis(
      context.pathId,
      context.rewardHash,
      SPAN_CREDIT_ALGORITHM_VERSION
    );
    if (existing) {
      const at = input.at ?? nowIso();
      const active = existing.status === "active"
        ? existing
        : this.deps.repos.proceduralSpanCredits.activate(existing.id, at);
      const result = {
        record: active,
        credits: this.deps.repos.proceduralSpanCredits.listCredits(active.id),
        created: false
      };
      this.enqueueSemanticClustering(context, active.id, at);
      return result;
    }
    if (!this.deps.skillLlm.isConfigured()) {
      throw new Error("SpanCredit scoring requires a configured evolution LLM");
    }
    const messages: LlmMessage[] = [
      { role: "system", content: SPAN_CREDIT_SCORING_PROMPT },
      { role: "user", content: stableStringify(context.payload) }
    ];
    const options = {
      operation: SPAN_CREDIT_SCORING_OPERATION,
      thinkingMode: this.deps.enableThinking ? "enabled" as const : "disabled" as const,
      temperature: 0,
      maxTokens: spanCreditMaxTokens(
        context.occurrences.length,
        this.deps.enableThinking === true
      ),
      maxLengthRetries: this.deps.enableThinking ? 2 : 1,
      jsonMode: true
    };
    let raw = await this.deps.skillLlm.completeJson<SpanCreditLlmResult>(messages, options);
    let parsed: ParsedScoringResult;
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        parsed = parseScoringResult(raw, context);
        break;
      } catch (error) {
        if (repairAttempt >= MAX_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        raw = await this.deps.skillLlm.completeJson<SpanCreditLlmResult>([
          ...messages,
          { role: "assistant", content: stableStringify(raw) },
          {
            role: "user",
            content: `The previous JSON violated the contract: ${errorMessage(error)}. Return a complete corrected JSON object only.`
          }
        ], {
          ...options,
          operation: `${SPAN_CREDIT_SCORING_OPERATION}.repair.${repairNumber}`
        });
      }
    }
    const parsedByOccurrence = new Map(parsed.credits.map((item) => [item.occurrenceId, item]));
    const credits = context.occurrences.map((occurrence) => {
      const scored = parsedByOccurrence.get(occurrence.id)!;
      const evidenceRole = evidenceRoleFor({
        occurrence,
        rewardCredit: scored.rewardCredit,
        attributionType: scored.attributionType,
        confidence: scored.confidence
      });
      return {
        occurrenceId: occurrence.id,
        spanId: occurrence.spanId,
        spanIndex: occurrence.spanIndex,
        preStateId: occurrence.preStateId,
        postStateId: occurrence.postStateId,
        rewardCredit: scored.rewardCredit,
        attributionType: scored.attributionType,
        confidence: scored.confidence,
        evidenceRole,
        evidenceRefs: unique([
          ...scored.evidenceRefs,
          ...occurrence.span.termination.evidenceRefs
        ]),
        reason: clip(scored.reason, 800)
      };
    });
    const run = buildEpisodeSpanCreditRun({
      episodeId: context.episode.id,
      pathId: context.pathId,
      pathHash: context.pathHash,
      namespaceId: context.namespaceId,
      rewardHash: context.rewardHash,
      episodeReward: context.episodeReward,
      inputCompaction: context.inputCompaction,
      credits,
      scorerModel: this.deps.skillLlm.config.model
    });
    const at = input.at ?? nowIso();
    const result = this.deps.repos.proceduralSpanCredits.saveAndActivate(run, at);
    this.enqueueSemanticClustering(context, result.record.id, at);
    return result;
  }

  private enqueueSemanticClustering(context: ScoringContext, runId: string, at: string): void {
    if (!this.deps.enqueueJob) return;
    this.deps.enqueueJob({
      jobType: "procedural_span_cluster",
      userId: context.episode.userId,
      episodeId: context.episode.id,
      payload: {
        namespaceId: context.namespaceId,
        creditRunId: runId,
        pathId: context.pathId,
        algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
        embeddingVersion: PROCEDURAL_SPAN_EMBEDDING_VERSION
      },
      createdAt: at
    });
  }

  private buildContext(episodeId: string, requestedPathId?: string): ScoringContext {
    const episode = this.deps.repos.runtime.getEpisode(episodeId);
    if (!episode || episode.status !== "closed") {
      throw new Error(`SpanCredit requires a closed Episode: ${episodeId}`);
    }
    if (episode.rTask === undefined) {
      throw new Error(`SpanCredit requires Episode reward: ${episodeId}`);
    }
    const pathRecord = requestedPathId
      ? this.deps.repos.proceduralPaths.get(requestedPathId)
      : this.deps.repos.proceduralPaths.getActiveForEpisode(episodeId);
    if (!pathRecord || pathRecord.status !== "active" || pathRecord.episodeId !== episodeId) {
      throw new Error(`SpanCredit requires the active Episode path: ${requestedPathId ?? episodeId}`);
    }
    const occurrences = this.deps.repos.proceduralPaths.listOccurrencesForPath(pathRecord.id);
    if (occurrences.length === 0 || occurrences.length !== pathRecord.path.spans.length) {
      throw new Error(`SpanCredit path has incomplete occurrences: ${pathRecord.id}`);
    }
    const stateById = new Map(pathRecord.path.states.map((state) => [state.id, state]));
    const boundaries = occurrenceBoundaries(occurrences, stateById);
    const episodeReward = episodeRewardForCredit(episode);
    const rewardHash = spanCreditRewardHash(episode);
    const validEvidenceRefs = collectValidEvidenceRefs(pathRecord.path, occurrences, boundaries);
    const scoringInput = buildScoringPayload({
      episode,
      path: pathRecord.path,
      occurrences,
      boundaries,
      episodeReward
    }, this.deps.maxInputChars ?? SPAN_CREDIT_MAX_INPUT_CHARS);
    return {
      episode,
      path: pathRecord.path,
      pathId: pathRecord.id,
      pathHash: pathRecord.pathHash,
      namespaceId: pathRecord.namespaceId,
      occurrences,
      boundaries,
      episodeReward,
      rewardHash,
      validEvidenceRefs,
      payload: scoringInput.payload,
      inputCompaction: scoringInput.compaction
    };
  }
}

function spanCreditMaxTokens(spanCount: number, thinkingEnabled: boolean): number {
  const contentBudget = Math.max(
    SPAN_CREDIT_MIN_CONTENT_TOKENS,
    spanCount * SPAN_CREDIT_CONTENT_TOKENS_PER_SPAN
  );
  return contentBudget + (thinkingEnabled ? SPAN_CREDIT_REASONING_RESERVE_TOKENS : 0);
}

export function spanCreditRewardHash(episode: Pick<EpisodeRecord, "rTask" | "rewardDetail">): string {
  return stableHash({ rTask: episode.rTask, rewardDetail: episode.rewardDetail });
}

function buildScoringPayload(input: {
  episode: EpisodeRecord;
  path: EpisodeProceduralPathV2;
  occurrences: ProceduralSpanOccurrenceRecord[];
  boundaries: BoundaryState[];
  episodeReward: number;
}, maxInputChars: number): {
  payload: Record<string, unknown>;
  compaction: SpanCreditInputCompactionV1;
} {
  const stepById = new Map(input.path.steps.map((step) => [step.id, step]));
  const selectedSteps = new Map(input.occurrences.map((occurrence) => [
    occurrence.id,
    selectSteps(occurrence.stepIds.map((id) => stepById.get(id)!))
  ]));
  const omittedBySelection = input.path.steps.length - [...selectedSteps.values()]
    .reduce((sum, steps) => sum + steps.length, 0);
  const payloadFor = (
    mode: SpanCreditInputCompactionV1["mode"],
    textLimit: number
  ): Record<string, unknown> => ({
    episode_id: input.episode.id,
    episode_goal: clip(
      episodeGoal(input.episode, input.boundaries[0]!.state),
      mode === "full" ? 1_200 : mode === "compact" ? 600 : Math.max(textLimit, 100)
    ),
    episode_reward: input.episodeReward,
    reward_reason: clip(
      text(input.episode.rewardDetail.reason) ?? "",
      mode === "minimal" ? Math.max(textLimit, 100) : Math.max(textLimit, 400)
    ),
    input_compaction_mode: mode,
    boundary_states: input.boundaries.map(({ boundaryIndex, state }) => ({
      boundary_index: boundaryIndex,
      state_id: state.id,
      summary: clip(state.summary, textLimit),
      task_status: state.taskStatus,
      ...(mode === "minimal" ? {} : {
        goal: state.goal ? compactStateEntry(state.goal, textLimit) : undefined,
        facts: state.facts.slice(0, mode === "full" ? 10 : 4)
          .map((entry) => compactStateEntry(entry, textLimit)),
        issues: state.issues.slice(0, mode === "full" ? 10 : 4)
          .map((entry) => compactStateEntry(entry, textLimit)),
        verification: state.verification.slice(0, mode === "full" ? 10 : 4)
          .map((entry) => compactStateEntry(entry, textLimit)),
        artifacts: state.artifacts.slice(0, mode === "full" ? 10 : 4)
          .map((entry) => compactStateEntry(entry, textLimit))
      })
    })),
    spans: input.occurrences.map((occurrence) => ({
      occurrence_id: occurrence.id,
      span_id: occurrence.spanId,
      span_index: occurrence.spanIndex,
      pre_boundary_index: occurrence.spanIndex,
      post_boundary_index: occurrence.spanIndex + 1,
      local_goal: clip(occurrence.localGoal, textLimit),
      capability_goal: clip(occurrence.capabilityGoal, textLimit),
      entry_condition: clip(occurrence.entryCondition, textLimit),
      exit_condition: clip(occurrence.exitCondition, textLimit),
      termination_status: occurrence.terminationStatus,
      cost: occurrence.span.cost,
      procedure_summary: clip(occurrence.projection.procedureText, textLimit),
      evidence_refs: mode === "minimal"
        ? [occurrence.id]
        : unique([occurrence.id, ...occurrence.span.termination.evidenceRefs]).slice(0, 8),
      ...(mode === "full" ? {
        steps: selectedSteps.get(occurrence.id)!.map(compactStep)
      } : {})
    }))
  });
  const full = payloadFor("full", 1_200);
  const originalChars = stableStringify(full).length;
  if (originalChars <= maxInputChars) {
    return {
      payload: full,
      compaction: {
        mode: "full",
        originalChars,
        finalChars: originalChars,
        omittedStepCount: omittedBySelection
      }
    };
  }
  const compact = payloadFor("compact", 500);
  const compactChars = stableStringify(compact).length;
  if (compactChars <= maxInputChars) {
    return {
      payload: compact,
      compaction: {
        mode: "compact",
        originalChars,
        finalChars: compactChars,
        omittedStepCount: input.path.steps.length
      }
    };
  }
  for (const textLimit of [240, 160, 96, 48, 24]) {
    const minimal = payloadFor("minimal", textLimit);
    const minimalChars = stableStringify(minimal).length;
    if (minimalChars <= maxInputChars) {
      return {
        payload: minimal,
        compaction: {
          mode: "minimal",
          originalChars,
          finalChars: minimalChars,
          omittedStepCount: input.path.steps.length
        }
      };
    }
  }
  throw new Error(
    `SpanCredit irreducible payload exceeds ${maxInputChars} characters; ` +
    `all ${input.occurrences.length} Final Spans were preserved and no multi-call fallback is allowed`
  );
}

function parseScoringResult(raw: SpanCreditLlmResult, context: ScoringContext): ParsedScoringResult {
  assertExactKeys(raw, ["span_credits"], "SpanCredit response");
  const expectedOccurrenceIds = context.occurrences.map((occurrence) => occurrence.id);
  if (!Array.isArray(raw.span_credits) || raw.span_credits.length !== expectedOccurrenceIds.length) {
    throw new Error(`span_credits must contain ${expectedOccurrenceIds.length} entries`);
  }
  const credits = raw.span_credits.map((value) => {
    if (!isRecord(value)) throw new Error("span_credits entries must be objects");
    assertExactKeys(
      value,
      ["occurrence_id", "reward_credit", "attribution_type", "confidence", "evidence_refs", "reason"],
      "span_credits entry"
    );
    const occurrenceId = requiredText(value.occurrence_id, "span_credits.occurrence_id");
    const rewardCredit = score(value.reward_credit, "span_credits.reward_credit");
    const attributionType = attributionTypeValue(value.attribution_type);
    if (attributionType === "helpful" && rewardCredit <= 0) {
      throw new Error(`helpful Span ${occurrenceId} must have positive reward_credit`);
    }
    if (attributionType === "harmful" && rewardCredit >= 0) {
      throw new Error(`harmful Span ${occurrenceId} must have negative reward_credit`);
    }
    if (attributionType === "neutral" && Math.abs(rewardCredit) > 0.05) {
      throw new Error(`neutral Span ${occurrenceId} must have near-zero reward_credit`);
    }
    return {
      occurrenceId,
      rewardCredit,
      attributionType,
      confidence: unitScore(value.confidence, "span_credits.confidence"),
      evidenceRefs: evidenceRefs(value.evidence_refs, context.validEvidenceRefs, "span_credits.evidence_refs"),
      reason: requiredText(value.reason, "span_credits.reason")
    };
  });
  assertExactIds(credits.map((item) => item.occurrenceId), expectedOccurrenceIds, "span_credits occurrence ids");
  if (credits.some((item, index) => item.occurrenceId !== expectedOccurrenceIds[index])) {
    throw new Error("span_credits must preserve the requested occurrence order");
  }
  const allocatedReward = roundUnbounded(
    credits.reduce((sum, credit) => sum + credit.rewardCredit, 0)
  );
  if (Math.abs(allocatedReward - context.episodeReward) > 1e-5) {
    throw new Error(
      `span_credits must conserve episode_reward: credits=${allocatedReward}, episode_reward=${context.episodeReward}`
    );
  }
  return { credits };
}

function occurrenceBoundaries(
  occurrences: ProceduralSpanOccurrenceRecord[],
  stateById: Map<string, ObservedStateV1>
): BoundaryState[] {
  const first = occurrences[0]!;
  const pre = stateById.get(first.preStateId);
  if (!pre) throw new Error(`SpanCredit missing pre-state: ${first.preStateId}`);
  const boundaries: BoundaryState[] = [{ boundaryIndex: 0, state: pre }];
  for (const [index, occurrence] of occurrences.entries()) {
    if (occurrence.spanIndex !== index) throw new Error(`SpanCredit occurrence index mismatch: ${occurrence.id}`);
    const post = stateById.get(occurrence.postStateId);
    if (!post) throw new Error(`SpanCredit missing post-state: ${occurrence.postStateId}`);
    boundaries.push({ boundaryIndex: index + 1, state: post });
  }
  return boundaries;
}

function collectValidEvidenceRefs(
  path: EpisodeProceduralPathV2,
  occurrences: ProceduralSpanOccurrenceRecord[],
  boundaries: BoundaryState[]
): Set<string> {
  const refs = new Set<string>();
  for (const boundary of boundaries) {
    refs.add(boundary.state.id);
    for (const entry of [
      ...(boundary.state.goal ? [boundary.state.goal] : []),
      ...boundary.state.constraints,
      ...boundary.state.facts,
      ...boundary.state.artifacts,
      ...boundary.state.issues,
      ...boundary.state.verification
    ]) {
      for (const ref of entry.sourceRefs) refs.add(ref);
    }
  }
  for (const occurrence of occurrences) {
    refs.add(occurrence.id);
    refs.add(occurrence.spanId);
    for (const ref of occurrence.span.termination.evidenceRefs) refs.add(ref);
  }
  for (const step of path.steps) {
    refs.add(step.id);
    for (const ref of [
      ...step.action.eventRefs,
      ...step.outcome.evidenceRefs,
      ...step.actionEffectDelta.flatMap((delta) => delta.sourceRefs),
      ...step.externalObservationDelta.flatMap((delta) => delta.sourceRefs)
    ]) refs.add(ref);
  }
  return refs;
}

function evidenceRoleFor(input: {
  occurrence: ProceduralSpanOccurrenceRecord;
  rewardCredit: number;
  attributionType: SpanCreditAttributionType;
  confidence: number;
}): SpanCreditEvidenceRole {
  if (input.confidence < MIN_ROLE_CONFIDENCE || input.attributionType === "uncertain") {
    return "uncertain";
  }
  if (
    input.attributionType === "harmful" &&
    input.rewardCredit <= NEGATIVE_CREDIT_THRESHOLD
  ) return "counterexample";
  if (
    input.attributionType === "helpful" &&
    input.rewardCredit >= POSITIVE_CREDIT_THRESHOLD &&
    input.occurrence.terminationStatus === "success"
  ) return "support";
  return "neutral";
}

function episodeRewardForCredit(episode: EpisodeRecord): number {
  if (typeof episode.rTask !== "number" || !Number.isFinite(episode.rTask)) {
    throw new Error(`SpanCredit Episode lacks a finite rTask reward: ${episode.id}`);
  }
  return roundScore(episode.rTask);
}

function episodeGoal(episode: EpisodeRecord, initialState: ObservedStateV1): string {
  const canonicalGoal = text(episode.meta.canonicalGoal);
  if (canonicalGoal) return canonicalGoal;
  const initialUserText = text(episode.meta.initialUserText);
  return canonicalGoal || initialUserText || episode.title || initialState.goal?.subject || initialState.summary;
}

function compactStep(step: ExecutionStepV1): Record<string, unknown> {
  return {
    step_id: step.id,
    action_type: step.action.type,
    intent: clip(step.action.intent, 360),
    summary: clip(step.action.summary, 500),
    outcome: step.outcome.status,
    evidence_refs: unique([...step.action.eventRefs, ...step.outcome.evidenceRefs]),
    retry_of_step_id: step.retryOfStepId,
    recovery_from_step_id: step.recoveryFromStepId,
    cost: step.cost
  };
}

function compactStateEntry(
  entry: ObservedStateV1["facts"][number],
  textLimit: number
): Record<string, unknown> {
  return {
    subject: clip(entry.subject, textLimit),
    ...(entry.status ? { status: clip(entry.status, Math.min(textLimit, 160)) } : {}),
    ...(entry.value === undefined ? {} : {
      value: clip(typeof entry.value === "string" ? entry.value : stableStringify(entry.value), textLimit)
    }),
    source_refs: entry.sourceRefs.slice(0, 4)
  };
}

function selectSteps(steps: ExecutionStepV1[]): ExecutionStepV1[] {
  if (steps.length <= MAX_STEPS_PER_SPAN) return steps;
  const selected = [steps[0]!, steps.at(-1)!];
  for (const step of steps) {
    if (selected.length >= MAX_STEPS_PER_SPAN) break;
    if (
      step.outcome.status === "failure" ||
      step.retryOfStepId ||
      step.recoveryFromStepId ||
      step.actionEffectDelta.some((delta) => delta.op === "issue.resolve" || delta.op === "verification.set")
    ) selected.push(step);
  }
  for (const step of steps) {
    if (selected.length >= MAX_STEPS_PER_SPAN) break;
    selected.push(step);
  }
  return [...new Map(selected.map((step) => [step.id, step])).values()]
    .sort((left, right) => left.stepIndex - right.stepIndex);
}

function evidenceRefs(raw: unknown, valid: Set<string>, field: string): string[] {
  if (!Array.isArray(raw)) throw new Error(`${field} must be an array`);
  const refs = unique(raw.map((value) => requiredText(value, field)));
  const invalid = refs.filter((ref) => !valid.has(ref));
  if (invalid.length > 0) throw new Error(`${field} contains invalid refs: ${invalid.join(", ")}`);
  return refs;
}

function assertExactIds<T extends string | number>(actual: T[], expected: T[], field: string): void {
  if (actual.length !== new Set(actual).size) throw new Error(`${field} contains duplicates`);
  const actualSet = new Set(actual);
  if (expected.some((value) => !actualSet.has(value)) || actual.some((value) => !new Set(expected).has(value))) {
    throw new Error(`${field} must match the requested IDs exactly`);
  }
}

function assertExactKeys(record: Record<string, unknown>, expectedKeys: string[], field: string): void {
  const actual = Object.keys(record).sort();
  const expected = [...expectedKeys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${field} keys must be exactly: ${expectedKeys.join(", ")}`);
  }
}

function requiredText(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${field} must be non-empty text`);
  return value.trim();
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function score(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < -1 || value > 1) {
    throw new Error(`${field} must be a number in [-1, 1]`);
  }
  return roundScore(value);
}

function unitScore(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${field} must be a number in [0, 1]`);
  }
  return Math.round(value * 1_000_000) / 1_000_000;
}

function attributionTypeValue(value: unknown): SpanCreditAttributionType {
  if (
    value === "helpful" || value === "harmful" || value === "externally_blocked" ||
    value === "neutral" || value === "uncertain"
  ) return value;
  throw new Error(
    "span_credits.attribution_type must be helpful, harmful, externally_blocked, neutral, or uncertain"
  );
}

function roundScore(value: number): number {
  return Math.round(Math.max(-1, Math.min(1, value)) * 1_000_000) / 1_000_000;
}

function roundUnbounded(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export type { EpisodeSpanCreditRunRecord };
