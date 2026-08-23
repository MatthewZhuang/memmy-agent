import { stableHash } from "../../utils/id.js";

export const SPAN_CREDIT_SCHEMA_VERSION = "span-credit.v1" as const;
export const SPAN_CREDIT_ALGORITHM_VERSION = "span-credit-potential-difference.v1" as const;
export const SPAN_CREDIT_PROMPT_VERSION = "span-credit-prompt.v1" as const;

export type SpanCreditEvidenceRole = "support" | "counterexample" | "neutral" | "uncertain";

export interface SpanStatePotentialV1 {
  boundaryIndex: number;
  stateId: string;
  progress: number;
  evidenceRefs: string[];
  reason: string;
}

export interface SpanCreditV1 {
  id: string;
  schemaVersion: typeof SPAN_CREDIT_SCHEMA_VERSION;
  occurrenceId: string;
  spanId: string;
  spanIndex: number;
  preStateId: string;
  postStateId: string;
  goalCredit: number;
  processQuality: number;
  confidence: number;
  creditScore: number;
  evidenceRole: SpanCreditEvidenceRole;
  evidenceRefs: string[];
  reason: string;
}

export interface EpisodeSpanCreditRunV1 {
  id: string;
  schemaVersion: typeof SPAN_CREDIT_SCHEMA_VERSION;
  algorithmVersion: typeof SPAN_CREDIT_ALGORITHM_VERSION;
  promptVersion: typeof SPAN_CREDIT_PROMPT_VERSION;
  episodeId: string;
  pathId: string;
  pathHash: string;
  namespaceId: string;
  rewardHash: string;
  goalAchievement: number;
  statePotentials: SpanStatePotentialV1[];
  credits: SpanCreditV1[];
  scorerModel?: string;
  contentHash: string;
}

export function buildEpisodeSpanCreditRun(input: {
  episodeId: string;
  pathId: string;
  pathHash: string;
  namespaceId: string;
  rewardHash: string;
  goalAchievement: number;
  statePotentials: SpanStatePotentialV1[];
  credits: Array<Omit<SpanCreditV1, "id" | "schemaVersion" | "creditScore">>;
  scorerModel?: string;
}): EpisodeSpanCreditRunV1 {
  const goalAchievement = roundScore(input.goalAchievement);
  const statePotentials = input.statePotentials.map((potential) => ({
    ...potential,
    progress: roundScore(potential.progress),
    evidenceRefs: uniqueSorted(potential.evidenceRefs)
  }));
  const credits = input.credits.map((credit) => {
    const goalCredit = roundScore(credit.goalCredit);
    const processQuality = roundScore(credit.processQuality);
    const confidence = roundUnit(credit.confidence);
    const creditScore = roundScore(goalCredit * confidence);
    const basis = {
      schemaVersion: SPAN_CREDIT_SCHEMA_VERSION,
      episodeId: input.episodeId,
      pathHash: input.pathHash,
      rewardHash: input.rewardHash,
      occurrenceId: credit.occurrenceId,
      goalCredit,
      processQuality,
      confidence,
      evidenceRole: credit.evidenceRole,
      evidenceRefs: uniqueSorted(credit.evidenceRefs),
      reason: credit.reason
    };
    return {
      ...credit,
      id: `span_credit_${stableHash(basis).slice(0, 24)}`,
      schemaVersion: SPAN_CREDIT_SCHEMA_VERSION,
      goalCredit,
      processQuality,
      confidence,
      creditScore,
      evidenceRefs: basis.evidenceRefs
    };
  });
  assertCreditPathShape(statePotentials, credits, goalAchievement);
  const creditedGoal = roundScore(credits.reduce((sum, credit) => sum + credit.goalCredit, 0));
  if (Math.abs(creditedGoal - goalAchievement) > 1e-5) {
    throw new Error(
      `SpanCredit goal conservation failed: credits=${creditedGoal}, goalAchievement=${goalAchievement}`
    );
  }
  const content = {
    schemaVersion: SPAN_CREDIT_SCHEMA_VERSION,
    algorithmVersion: SPAN_CREDIT_ALGORITHM_VERSION,
    promptVersion: SPAN_CREDIT_PROMPT_VERSION,
    episodeId: input.episodeId,
    pathId: input.pathId,
    pathHash: input.pathHash,
    namespaceId: input.namespaceId,
    rewardHash: input.rewardHash,
    goalAchievement,
    statePotentials,
    credits,
    ...(input.scorerModel ? { scorerModel: input.scorerModel } : {})
  };
  const contentHash = stableHash(content);
  return {
    id: `episode_span_credit_run_${contentHash.slice(0, 24)}`,
    ...content,
    contentHash
  };
}

function roundScore(value: number): number {
  if (!Number.isFinite(value)) throw new Error(`SpanCredit score is not finite: ${value}`);
  if (value < -1 || value > 1) throw new Error(`SpanCredit score is outside [-1, 1]: ${value}`);
  return Math.round(value * 1_000_000) / 1_000_000;
}

function roundUnit(value: number): number {
  if (!Number.isFinite(value)) throw new Error(`SpanCredit confidence is not finite: ${value}`);
  if (value < 0 || value > 1) throw new Error(`SpanCredit confidence is outside [0, 1]: ${value}`);
  return Math.round(value * 1_000_000) / 1_000_000;
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function assertCreditPathShape(
  statePotentials: SpanStatePotentialV1[],
  credits: SpanCreditV1[],
  goalAchievement: number
): void {
  if (credits.length === 0 || statePotentials.length !== credits.length + 1) {
    throw new Error("SpanCredit run must contain one more boundary state than Span credits");
  }
  if (Math.abs(statePotentials[0]!.progress) > 1e-6) {
    throw new Error("SpanCredit boundary 0 must be anchored to progress 0");
  }
  if (Math.abs(statePotentials.at(-1)!.progress - goalAchievement) > 1e-6) {
    throw new Error("SpanCredit terminal boundary must equal goalAchievement");
  }
  for (const [index, potential] of statePotentials.entries()) {
    if (potential.boundaryIndex !== index) {
      throw new Error(`SpanCredit boundary index mismatch at ${index}`);
    }
  }
  for (const [index, credit] of credits.entries()) {
    const pre = statePotentials[index]!;
    const post = statePotentials[index + 1]!;
    if (
      credit.spanIndex !== index ||
      credit.preStateId !== pre.stateId ||
      credit.postStateId !== post.stateId
    ) {
      throw new Error(`SpanCredit state transition mismatch at Span ${index}`);
    }
    const expectedCredit = Math.round((post.progress - pre.progress) * 1_000_000) / 1_000_000;
    if (Math.abs(expectedCredit - credit.goalCredit) > 1e-6) {
      throw new Error(`SpanCredit potential difference mismatch at Span ${index}`);
    }
  }
}
