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
  SPAN_CREDIT_PROMPT_VERSION,
  buildEpisodeSpanCreditRun,
  type SpanCreditEvidenceRole,
  type SpanStatePotentialV1
} from "./span-credit-model.js";
import type { EpisodeProceduralPathV2, ExecutionStepV1 } from "./procedural-path-model.js";

export const SPAN_CREDIT_SCORING_OPERATION = "span_credit.score.v1" as const;
const MAX_REPAIR_ATTEMPTS = 2;
const POSITIVE_CREDIT_THRESHOLD = 0.1;
const NEGATIVE_CREDIT_THRESHOLD = -0.1;
const MIN_ROLE_CONFIDENCE = 0.6;
const COUNTEREXAMPLE_PROCESS_THRESHOLD = -0.6;
const MIN_SUPPORT_PROCESS_QUALITY = -0.5;
const MAX_STEPS_PER_SPAN = 10;
const SPAN_CREDIT_MIN_CONTENT_TOKENS = 1_400;
const SPAN_CREDIT_CONTENT_TOKENS_PER_SPAN = 260;
const SPAN_CREDIT_REASONING_RESERVE_TOKENS = 3_000;

export const SPAN_CREDIT_SCORING_PROMPT = `You score one complete AI-agent EpisodeExecutionPath for procedural credit assignment.

The input contains an episode goal, an authoritative terminal goal-achievement anchor, ordered boundary states, and ordered procedural Span occurrences. A Span is a complete local subproblem lifecycle; internal failed attempts that recover before the Span exits remain part of that one Span.

Return two judgments only:
1. state_progress: the relative goal progress at every INTERNAL boundary. Do not score boundary 0 or the terminal boundary; the backend anchors them to 0 and terminal_goal_achievement.
2. span_process_quality: execution quality for every Span occurrence.

State progress rules:
- progress is in [-1, 1], relative to boundary 0.
- Positive means evidence-grounded progress toward the active Episode goal.
- Negative means the state is worse than at Episode start.
- Keep every adjacent boundary progress change within [-1, 1], because each Span credit uses that difference directly.
- Root-cause evidence, resolved uncertainty, committed constraints, produced artifacts, and verification can be progress when they enable later execution.
- A plan or claim without observable support is not completed progress.
- Judge the ordered path jointly. Progress may decrease and later recover.

Process quality rules:
- quality is in [-1, 1]. Judge effectiveness, avoidable repetition, recovery, verification, and cost together.
- Do not mark an otherwise successful Span as harmful merely because it contains a failed attempt; reflect recoverable inefficiency in quality.
- A failed/blocked Span may still contain useful evidence, but its process quality must reflect that its local exit condition was not achieved.

Grounding rules:
- Copy boundary_index and occurrence_id exactly.
- evidence_refs may contain only IDs or evidence references supplied in the input.
- Do not invent actions, outcomes, feedback, or causal claims.
- Return JSON only with exactly these top-level keys:
{
  "state_progress": [
    {"boundary_index": 1, "progress": 0.25, "evidence_refs": ["..."], "reason": "..."}
  ],
  "span_process_quality": [
    {"occurrence_id": "...", "quality": 0.5, "evidence_refs": ["..."], "reason": "..."}
  ]
}`;

export interface SpanCreditPipelineDeps {
  repos: Repositories;
  skillLlm: LlmClient;
  enableThinking?: boolean;
  enqueueJob?(input: EnqueueJobInput): EvolutionJobRecord;
}

interface StateProgressResult {
  boundaryIndex: number;
  progress: number;
  evidenceRefs: string[];
  reason: string;
}

interface ProcessQualityResult {
  occurrenceId: string;
  quality: number;
  evidenceRefs: string[];
  reason: string;
}

interface ParsedScoringResult {
  stateProgress: StateProgressResult[];
  processQuality: ProcessQualityResult[];
}

interface SpanCreditLlmResult extends Record<string, unknown> {
  state_progress?: unknown;
  span_process_quality?: unknown;
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
  goalAchievement: number;
  rewardHash: string;
  validEvidenceRefs: Set<string>;
  payload: Record<string, unknown>;
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
    const statePotentials = materializeStatePotentials(context, parsed.stateProgress);
    const potentialByBoundary = new Map(statePotentials.map((item) => [item.boundaryIndex, item]));
    const processByOccurrence = new Map(parsed.processQuality.map((item) => [item.occurrenceId, item]));
    const credits = context.occurrences.map((occurrence, index) => {
      const pre = potentialByBoundary.get(index)!;
      const post = potentialByBoundary.get(index + 1)!;
      const process = processByOccurrence.get(occurrence.id)!;
      const goalCredit = roundScore(post.progress - pre.progress);
      const confidence = creditConfidence(occurrence, pre, post, process);
      const creditScore = roundScore(goalCredit * confidence);
      const evidenceRole = evidenceRoleFor({
        occurrence,
        creditScore,
        processQuality: process.quality,
        confidence
      });
      return {
        occurrenceId: occurrence.id,
        spanId: occurrence.spanId,
        spanIndex: occurrence.spanIndex,
        preStateId: occurrence.preStateId,
        postStateId: occurrence.postStateId,
        goalCredit,
        processQuality: process.quality,
        confidence,
        evidenceRole,
        evidenceRefs: unique([
          ...pre.evidenceRefs,
          ...post.evidenceRefs,
          ...process.evidenceRefs,
          ...occurrence.span.termination.evidenceRefs
        ]),
        reason: clip(`goal: ${goalCredit}; process: ${process.reason}`, 800)
      };
    });
    const run = buildEpisodeSpanCreditRun({
      episodeId: context.episode.id,
      pathId: context.pathId,
      pathHash: context.pathHash,
      namespaceId: context.namespaceId,
      rewardHash: context.rewardHash,
      goalAchievement: context.goalAchievement,
      statePotentials,
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
    const goalAchievement = goalAchievementForEpisode(episode);
    const rewardHash = spanCreditRewardHash(episode);
    const validEvidenceRefs = collectValidEvidenceRefs(pathRecord.path, occurrences, boundaries);
    return {
      episode,
      path: pathRecord.path,
      pathId: pathRecord.id,
      pathHash: pathRecord.pathHash,
      namespaceId: pathRecord.namespaceId,
      occurrences,
      boundaries,
      goalAchievement,
      rewardHash,
      validEvidenceRefs,
      payload: buildScoringPayload({
        episode,
        path: pathRecord.path,
        occurrences,
        boundaries,
        goalAchievement
      })
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
  goalAchievement: number;
}): Record<string, unknown> {
  const stepById = new Map(input.path.steps.map((step) => [step.id, step]));
  return {
    episode_id: input.episode.id,
    episode_goal: episodeGoal(input.episode, input.boundaries[0]!.state),
    terminal_goal_achievement: input.goalAchievement,
    reward_reason: text(input.episode.rewardDetail.reason),
    boundary_states: input.boundaries.map(({ boundaryIndex, state }) => ({
      boundary_index: boundaryIndex,
      state_id: state.id,
      anchored_progress: boundaryIndex === 0
        ? 0
        : boundaryIndex === input.occurrences.length
          ? input.goalAchievement
          : undefined,
      summary: clip(state.summary, 1_200),
      task_status: state.taskStatus,
      goal: state.goal,
      issues: state.issues.slice(0, 12),
      verification: state.verification.slice(0, 12),
      artifacts: state.artifacts.slice(0, 12)
    })),
    spans: input.occurrences.map((occurrence) => ({
      occurrence_id: occurrence.id,
      span_id: occurrence.spanId,
      span_index: occurrence.spanIndex,
      pre_boundary_index: occurrence.spanIndex,
      post_boundary_index: occurrence.spanIndex + 1,
      local_goal: clip(occurrence.localGoal, 600),
      entry_condition: clip(occurrence.entryCondition, 800),
      exit_condition: clip(occurrence.exitCondition, 800),
      termination_status: occurrence.terminationStatus,
      cost: occurrence.span.cost,
      procedure_summary: clip(occurrence.projection.procedureText, 2_000),
      evidence_refs: occurrence.span.termination.evidenceRefs,
      steps: selectSteps(occurrence.stepIds.map((id) => stepById.get(id)!)).map(compactStep)
    }))
  };
}

function parseScoringResult(raw: SpanCreditLlmResult, context: ScoringContext): ParsedScoringResult {
  assertExactKeys(raw, ["state_progress", "span_process_quality"], "SpanCredit response");
  const internalBoundaries = context.boundaries.slice(1, -1).map((item) => item.boundaryIndex);
  const stateProgress = parseStateProgress(raw.state_progress, internalBoundaries, context.validEvidenceRefs);
  const anchoredProgress = [
    0,
    ...stateProgress.map((item) => item.progress),
    context.goalAchievement
  ];
  for (let index = 1; index < anchoredProgress.length; index += 1) {
    const delta = anchoredProgress[index]! - anchoredProgress[index - 1]!;
    if (delta < -1 || delta > 1) {
      throw new Error(`adjacent boundary progress change at Span ${index - 1} must be in [-1, 1]`);
    }
  }
  const processQuality = parseProcessQuality(
    raw.span_process_quality,
    context.occurrences.map((occurrence) => occurrence.id),
    context.validEvidenceRefs
  );
  return { stateProgress, processQuality };
}

function parseStateProgress(
  raw: unknown,
  expectedBoundaryIndices: number[],
  validRefs: Set<string>
): StateProgressResult[] {
  if (!Array.isArray(raw) || raw.length !== expectedBoundaryIndices.length) {
    throw new Error(`state_progress must contain ${expectedBoundaryIndices.length} entries`);
  }
  const parsed = raw.map((value) => {
    if (!isRecord(value)) throw new Error("state_progress entries must be objects");
    assertExactKeys(
      value,
      ["boundary_index", "progress", "evidence_refs", "reason"],
      "state_progress entry"
    );
    const boundaryIndex = integer(value.boundary_index, "state_progress.boundary_index");
    const progress = score(value.progress, "state_progress.progress");
    return {
      boundaryIndex,
      progress,
      evidenceRefs: evidenceRefs(value.evidence_refs, validRefs, "state_progress.evidence_refs"),
      reason: requiredText(value.reason, "state_progress.reason")
    };
  });
  assertExactIds(parsed.map((item) => item.boundaryIndex), expectedBoundaryIndices, "state_progress boundary indices");
  return parsed.sort((left, right) => left.boundaryIndex - right.boundaryIndex);
}

function parseProcessQuality(
  raw: unknown,
  expectedOccurrenceIds: string[],
  validRefs: Set<string>
): ProcessQualityResult[] {
  if (!Array.isArray(raw) || raw.length !== expectedOccurrenceIds.length) {
    throw new Error(`span_process_quality must contain ${expectedOccurrenceIds.length} entries`);
  }
  const parsed = raw.map((value) => {
    if (!isRecord(value)) throw new Error("span_process_quality entries must be objects");
    assertExactKeys(
      value,
      ["occurrence_id", "quality", "evidence_refs", "reason"],
      "span_process_quality entry"
    );
    const occurrenceId = requiredText(value.occurrence_id, "span_process_quality.occurrence_id");
    return {
      occurrenceId,
      quality: score(value.quality, "span_process_quality.quality"),
      evidenceRefs: evidenceRefs(value.evidence_refs, validRefs, "span_process_quality.evidence_refs"),
      reason: requiredText(value.reason, "span_process_quality.reason")
    };
  });
  assertExactIds(parsed.map((item) => item.occurrenceId), expectedOccurrenceIds, "span_process_quality occurrence ids");
  return parsed.sort((left, right) =>
    expectedOccurrenceIds.indexOf(left.occurrenceId) - expectedOccurrenceIds.indexOf(right.occurrenceId));
}

function materializeStatePotentials(
  context: ScoringContext,
  internal: StateProgressResult[]
): SpanStatePotentialV1[] {
  const internalByBoundary = new Map(internal.map((item) => [item.boundaryIndex, item]));
  return context.boundaries.map(({ boundaryIndex, state }) => {
    if (boundaryIndex === 0) {
      return {
        boundaryIndex,
        stateId: state.id,
        progress: 0,
        evidenceRefs: [state.id],
        reason: "Episode start anchor"
      };
    }
    if (boundaryIndex === context.boundaries.length - 1) {
      return {
        boundaryIndex,
        stateId: state.id,
        progress: context.goalAchievement,
        evidenceRefs: unique([state.id, ...state.verification.flatMap((item) => item.sourceRefs)]),
        reason: "Episode goal-achievement anchor"
      };
    }
    const scored = internalByBoundary.get(boundaryIndex)!;
    return {
      boundaryIndex,
      stateId: state.id,
      progress: scored.progress,
      evidenceRefs: scored.evidenceRefs,
      reason: scored.reason
    };
  });
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

function creditConfidence(
  occurrence: ProceduralSpanOccurrenceRecord,
  pre: SpanStatePotentialV1,
  post: SpanStatePotentialV1,
  process: ProcessQualityResult
): number {
  const evidenceCoverage = (
    (pre.evidenceRefs.length > 0 ? 1 : 0.5) +
    (post.evidenceRefs.length > 0 ? 1 : 0.5) +
    (process.evidenceRefs.length > 0 ? 1 : 0.5)
  ) / 3;
  return roundUnit(Math.min(occurrence.span.segmentation.confidence, evidenceCoverage));
}

function evidenceRoleFor(input: {
  occurrence: ProceduralSpanOccurrenceRecord;
  creditScore: number;
  processQuality: number;
  confidence: number;
}): SpanCreditEvidenceRole {
  if (input.confidence < MIN_ROLE_CONFIDENCE) return "uncertain";
  if (
    input.creditScore <= NEGATIVE_CREDIT_THRESHOLD ||
    input.processQuality <= COUNTEREXAMPLE_PROCESS_THRESHOLD ||
    input.occurrence.terminationStatus === "failure" ||
    input.occurrence.terminationStatus === "blocked" ||
    input.occurrence.terminationStatus === "abandoned"
  ) return "counterexample";
  if (
    input.creditScore >= POSITIVE_CREDIT_THRESHOLD &&
    input.processQuality >= MIN_SUPPORT_PROCESS_QUALITY &&
    input.occurrence.terminationStatus === "success"
  ) return "support";
  return "neutral";
}

function goalAchievementForEpisode(episode: EpisodeRecord): number {
  const axes = isRecord(episode.rewardDetail.axes) ? episode.rewardDetail.axes : undefined;
  const candidates = [
    axes?.goalAchievement,
    axes?.goal_achievement,
    episode.rewardDetail.goalAchievement,
    episode.rewardDetail.goal_achievement,
    episode.rTask
  ];
  const value = candidates.find((candidate): candidate is number =>
    typeof candidate === "number" && Number.isFinite(candidate));
  if (value === undefined) throw new Error(`SpanCredit reward lacks goal achievement: ${episode.id}`);
  return roundScore(value);
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

function integer(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isInteger(value)) throw new Error(`${field} must be an integer`);
  return value;
}

function score(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < -1 || value > 1) {
    throw new Error(`${field} must be a number in [-1, 1]`);
  }
  return roundScore(value);
}

function roundScore(value: number): number {
  return Math.round(Math.max(-1, Math.min(1, value)) * 1_000_000) / 1_000_000;
}

function roundUnit(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 1_000_000) / 1_000_000;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export type { EpisodeSpanCreditRunRecord };
