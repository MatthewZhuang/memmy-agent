import { stableHash } from "../../utils/id.js";

export const SPAN_CREDIT_SCHEMA_VERSION = "span-credit.v2" as const;
export const SPAN_CREDIT_ALGORITHM_VERSION = "span-credit-global-reward-budget.v2" as const;
export const SPAN_CREDIT_PROMPT_VERSION = "span-credit-global-reward-budget-prompt.v2" as const;

export type SpanCreditAttributionType =
  | "helpful"
  | "harmful"
  | "externally_blocked"
  | "neutral"
  | "uncertain";

export type SpanCreditEvidenceRole = "support" | "counterexample" | "neutral" | "uncertain";

export interface SpanCreditInputCompactionV1 {
  mode: "full" | "compact" | "minimal";
  originalChars: number;
  finalChars: number;
  omittedStepCount: number;
}

export interface SpanCreditV2 {
  id: string;
  schemaVersion: typeof SPAN_CREDIT_SCHEMA_VERSION;
  occurrenceId: string;
  spanId: string;
  spanIndex: number;
  preStateId: string;
  postStateId: string;
  rewardCredit: number;
  attributionType: SpanCreditAttributionType;
  confidence: number;
  evidenceRole: SpanCreditEvidenceRole;
  evidenceRefs: string[];
  reason: string;
}

export interface EpisodeSpanCreditRunV2 {
  id: string;
  schemaVersion: typeof SPAN_CREDIT_SCHEMA_VERSION;
  algorithmVersion: typeof SPAN_CREDIT_ALGORITHM_VERSION;
  promptVersion: typeof SPAN_CREDIT_PROMPT_VERSION;
  episodeId: string;
  pathId: string;
  pathHash: string;
  namespaceId: string;
  rewardHash: string;
  episodeReward: number;
  inputCompaction: SpanCreditInputCompactionV1;
  credits: SpanCreditV2[];
  scorerModel?: string;
  contentHash: string;
}

export function buildEpisodeSpanCreditRun(input: {
  episodeId: string;
  pathId: string;
  pathHash: string;
  namespaceId: string;
  rewardHash: string;
  episodeReward: number;
  inputCompaction?: SpanCreditInputCompactionV1;
  credits: Array<Omit<SpanCreditV2, "id" | "schemaVersion">>;
  scorerModel?: string;
}): EpisodeSpanCreditRunV2 {
  const episodeReward = roundScore(input.episodeReward);
  const credits = input.credits.map((credit) => {
    const rewardCredit = roundScore(credit.rewardCredit);
    const confidence = roundUnit(credit.confidence);
    const evidenceRefs = uniqueSorted(credit.evidenceRefs);
    const basis = {
      schemaVersion: SPAN_CREDIT_SCHEMA_VERSION,
      episodeId: input.episodeId,
      pathHash: input.pathHash,
      rewardHash: input.rewardHash,
      occurrenceId: credit.occurrenceId,
      rewardCredit,
      attributionType: credit.attributionType,
      confidence,
      evidenceRole: credit.evidenceRole,
      evidenceRefs,
      reason: credit.reason
    };
    return {
      ...credit,
      id: `span_credit_${stableHash(basis).slice(0, 24)}`,
      schemaVersion: SPAN_CREDIT_SCHEMA_VERSION,
      rewardCredit,
      confidence,
      evidenceRefs
    };
  });
  assertCreditPathShape(credits);
  const creditedReward = roundedSum(credits.map((credit) => credit.rewardCredit));
  if (Math.abs(creditedReward - episodeReward) > 1e-5) {
    throw new Error(
      `SpanCredit reward conservation failed: credits=${creditedReward}, episodeReward=${episodeReward}`
    );
  }
  const inputCompaction = input.inputCompaction ?? {
    mode: "full",
    originalChars: 0,
    finalChars: 0,
    omittedStepCount: 0
  };
  const content = {
    schemaVersion: SPAN_CREDIT_SCHEMA_VERSION,
    algorithmVersion: SPAN_CREDIT_ALGORITHM_VERSION,
    promptVersion: SPAN_CREDIT_PROMPT_VERSION,
    episodeId: input.episodeId,
    pathId: input.pathId,
    pathHash: input.pathHash,
    namespaceId: input.namespaceId,
    rewardHash: input.rewardHash,
    episodeReward,
    inputCompaction,
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

function roundedSum(values: readonly number[]): number {
  return Math.round(values.reduce((sum, value) => sum + value, 0) * 1_000_000) / 1_000_000;
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function assertCreditPathShape(credits: SpanCreditV2[]): void {
  if (credits.length === 0) throw new Error("SpanCredit run must contain at least one Span credit");
  const occurrenceIds = new Set<string>();
  for (const [index, credit] of credits.entries()) {
    if (credit.spanIndex !== index) {
      throw new Error(`SpanCredit index mismatch at Span ${index}`);
    }
    if (occurrenceIds.has(credit.occurrenceId)) {
      throw new Error(`SpanCredit occurrence is duplicated: ${credit.occurrenceId}`);
    }
    occurrenceIds.add(credit.occurrenceId);
  }
}
