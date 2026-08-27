import { cosineSimilarity } from "./step-sequence-learning-model.js";

export interface BandedMonotonicMatchConfig {
  scale: number;
  bandWidth: number;
  minStepSimilarity: number;
  minMatchedSteps: number;
  minCoverage: number;
  minAverageMatchSimilarity: number;
  maxInternalGap: number;
  gapPenalty: number;
  minAlignmentScore: number;
}

export interface BandedMonotonicMatchPairV1 {
  leftIndex: number;
  rightIndex: number;
  similarity: number;
}

export interface BandedMonotonicMatchResultV1 {
  admitted: boolean;
  score: number;
  averageMatchSimilarity: number;
  coverage: number;
  matchedSteps: number;
  internalGapSteps: number;
  maxInternalGap: number;
  pairs: BandedMonotonicMatchPairV1[];
  rejectionReasons: string[];
}

interface AlignmentCell {
  objective: number;
  previous?: { i: number; j: number };
  operation: "start" | "match" | "skip_left" | "skip_right";
  similarity?: number;
}

export function bandedMonotonicMatch(
  left: readonly (readonly number[])[],
  right: readonly (readonly number[])[],
  config: BandedMonotonicMatchConfig
): BandedMonotonicMatchResultV1 {
  validateConfig(config);
  if (left.length !== config.scale || right.length !== config.scale) {
    throw new Error(
      `banded match expected scale ${config.scale}, got ${left.length} and ${right.length}`
    );
  }
  const rows = left.length + 1;
  const columns = right.length + 1;
  const cells: Array<Array<AlignmentCell | undefined>> = Array.from(
    { length: rows },
    () => Array.from({ length: columns }, () => undefined)
  );
  cells[0]![0] = { objective: 0, operation: "start" };
  for (let i = 1; i <= Math.min(left.length, config.bandWidth); i += 1) {
    cells[i]![0] = {
      objective: 0,
      previous: { i: i - 1, j: 0 },
      operation: "skip_left"
    };
  }
  for (let j = 1; j <= Math.min(right.length, config.bandWidth); j += 1) {
    cells[0]![j] = {
      objective: 0,
      previous: { i: 0, j: j - 1 },
      operation: "skip_right"
    };
  }
  for (let i = 1; i <= left.length; i += 1) {
    const minJ = Math.max(1, i - config.bandWidth);
    const maxJ = Math.min(right.length, i + config.bandWidth);
    for (let j = minJ; j <= maxJ; j += 1) {
      const candidates: AlignmentCell[] = [];
      const diagonal = cells[i - 1]![j - 1];
      const similarity = cosineSimilarity(left[i - 1]!, right[j - 1]!);
      if (diagonal && similarity >= config.minStepSimilarity) {
        candidates.push({
          objective: diagonal.objective + similarity,
          previous: { i: i - 1, j: j - 1 },
          operation: "match",
          similarity
        });
      }
      const skipLeft = cells[i - 1]![j];
      if (skipLeft) {
        candidates.push({
          objective: skipLeft.objective - config.gapPenalty,
          previous: { i: i - 1, j },
          operation: "skip_left"
        });
      }
      const skipRight = cells[i]![j - 1];
      if (skipRight) {
        candidates.push({
          objective: skipRight.objective - config.gapPenalty,
          previous: { i, j: j - 1 },
          operation: "skip_right"
        });
      }
      cells[i]![j] = candidates.sort((a, b) =>
        b.objective - a.objective || operationPriority(b.operation) - operationPriority(a.operation)
      )[0];
    }
  }
  let endpoint: { i: number; j: number; cell: AlignmentCell } | undefined;
  for (let i = Math.max(0, left.length - config.bandWidth); i <= left.length; i += 1) {
    for (let j = Math.max(0, right.length - config.bandWidth); j <= right.length; j += 1) {
      const cell = cells[i]![j];
      if (!cell) continue;
      if (!endpoint || cell.objective > endpoint.cell.objective) endpoint = { i, j, cell };
    }
  }
  if (!endpoint) return emptyResult(["no alignment path"]);
  const pairs: BandedMonotonicMatchPairV1[] = [];
  let cursor: { i: number; j: number; cell: AlignmentCell } | undefined = endpoint;
  while (cursor) {
    if (cursor.cell.operation === "match") {
      pairs.push({
        leftIndex: cursor.i - 1,
        rightIndex: cursor.j - 1,
        similarity: cursor.cell.similarity!
      });
    }
    const previous: { i: number; j: number } | undefined = cursor.cell.previous;
    if (!previous) break;
    const previousCell: AlignmentCell | undefined = cells[previous.i]![previous.j];
    cursor = previousCell ? { ...previous, cell: previousCell } : undefined;
  }
  pairs.reverse();
  if (pairs.length === 0) return emptyResult(["no Step pair passed minStepSimilarity"]);
  const averageMatchSimilarity = pairs.reduce((sum, pair) => sum + pair.similarity, 0) /
    pairs.length;
  const coverage = pairs.length / Math.max(left.length, right.length);
  let internalGapSteps = 0;
  let maxInternalGap = 0;
  for (let index = 1; index < pairs.length; index += 1) {
    const previous = pairs[index - 1]!;
    const current = pairs[index]!;
    const leftGap = current.leftIndex - previous.leftIndex - 1;
    const rightGap = current.rightIndex - previous.rightIndex - 1;
    internalGapSteps += leftGap + rightGap;
    maxInternalGap = Math.max(maxInternalGap, leftGap, rightGap);
  }
  const score = clamp(
    averageMatchSimilarity * coverage -
      config.gapPenalty * internalGapSteps / Math.max(left.length, right.length),
    0,
    1
  );
  const rejectionReasons: string[] = [];
  if (pairs.length < config.minMatchedSteps) {
    rejectionReasons.push(`matchedSteps ${pairs.length} < ${config.minMatchedSteps}`);
  }
  if (coverage < config.minCoverage) {
    rejectionReasons.push(`coverage ${coverage.toFixed(4)} < ${config.minCoverage}`);
  }
  if (averageMatchSimilarity < config.minAverageMatchSimilarity) {
    rejectionReasons.push(
      `averageMatchSimilarity ${averageMatchSimilarity.toFixed(4)} < ` +
      `${config.minAverageMatchSimilarity}`
    );
  }
  if (maxInternalGap > config.maxInternalGap) {
    rejectionReasons.push(`maxInternalGap ${maxInternalGap} > ${config.maxInternalGap}`);
  }
  if (score < config.minAlignmentScore) {
    rejectionReasons.push(`score ${score.toFixed(4)} < ${config.minAlignmentScore}`);
  }
  return {
    admitted: rejectionReasons.length === 0,
    score,
    averageMatchSimilarity,
    coverage,
    matchedSteps: pairs.length,
    internalGapSteps,
    maxInternalGap,
    pairs,
    rejectionReasons
  };
}

export function selfBandedMonotonicMatch(
  vectors: readonly (readonly number[])[],
  config: BandedMonotonicMatchConfig
): BandedMonotonicMatchResultV1 {
  const pairs = vectors.map((_, index) => ({ leftIndex: index, rightIndex: index, similarity: 1 }));
  return {
    admitted: true,
    score: 1,
    averageMatchSimilarity: 1,
    coverage: 1,
    matchedSteps: vectors.length,
    internalGapSteps: 0,
    maxInternalGap: 0,
    pairs,
    rejectionReasons: []
  };
}

function validateConfig(config: BandedMonotonicMatchConfig): void {
  const unitFields = [
    ["minStepSimilarity", config.minStepSimilarity],
    ["minCoverage", config.minCoverage],
    ["minAverageMatchSimilarity", config.minAverageMatchSimilarity],
    ["minAlignmentScore", config.minAlignmentScore]
  ] as const;
  if (!Number.isInteger(config.scale) || config.scale < 2) {
    throw new Error("banded match scale must be an integer >= 2");
  }
  if (!Number.isInteger(config.bandWidth) || config.bandWidth < 0 ||
      config.bandWidth >= config.scale) {
    throw new Error("banded match bandWidth must be in [0, scale)");
  }
  if (!Number.isInteger(config.minMatchedSteps) || config.minMatchedSteps < 1 ||
      config.minMatchedSteps > config.scale) {
    throw new Error("banded match minMatchedSteps must be in [1, scale]");
  }
  if (!Number.isInteger(config.maxInternalGap) || config.maxInternalGap < 0) {
    throw new Error("banded match maxInternalGap must be an integer >= 0");
  }
  if (!(config.gapPenalty >= 0 && config.gapPenalty <= 1)) {
    throw new Error("banded match gapPenalty must be in [0, 1]");
  }
  for (const [field, value] of unitFields) {
    if (!(value >= 0 && value <= 1)) throw new Error(`${field} must be in [0, 1]`);
  }
}

function operationPriority(operation: AlignmentCell["operation"]): number {
  return operation === "match" ? 3 : operation === "skip_left" ? 2 : 1;
}

function emptyResult(rejectionReasons: string[]): BandedMonotonicMatchResultV1 {
  return {
    admitted: false,
    score: 0,
    averageMatchSimilarity: 0,
    coverage: 0,
    matchedSteps: 0,
    internalGapSteps: 0,
    maxInternalGap: 0,
    pairs: [],
    rejectionReasons
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
