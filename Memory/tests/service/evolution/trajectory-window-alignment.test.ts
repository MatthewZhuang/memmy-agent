import { describe, expect, it } from "vitest";
import {
  bandedMonotonicMatch,
  cosineSimilarity,
  selfBandedMonotonicMatch,
  type BandedMonotonicMatchConfig
} from "../../../src/service/evolution/trajectory-window-alignment.js";

const span5Config: BandedMonotonicMatchConfig = {
  scale: 5,
  bandWidth: 1,
  minStepSimilarity: 0.70,
  minMatchedSteps: 4,
  minCoverage: 0.80,
  minAverageMatchSimilarity: 0.78,
  maxInternalGap: 1,
  gapPenalty: 0.10,
  minAlignmentScore: 0.62
};

describe("trajectory window alignment", () => {
  it("recovers a one-Step boundary shift while preserving monotonic order", () => {
    const a = [1, 0, 0, 0, 0];
    const b = [0, 1, 0, 0, 0];
    const c = [0, 0, 1, 0, 0];
    const d = [0, 0, 0, 1, 0];
    const noise = [0, 0, 0, 0, 1];

    const result = bandedMonotonicMatch(
      [noise, a, b, c, d],
      [a, b, c, d, noise],
      span5Config
    );

    expect(result.admitted).toBe(true);
    expect(result.matchedSteps).toBe(4);
    expect(result.coverage).toBe(0.8);
    expect(result.pairs.map(({ leftIndex, rightIndex }) =>
      [leftIndex, rightIndex])).toEqual([
      [1, 0],
      [2, 1],
      [3, 2],
      [4, 3]
    ]);
  });

  it("rejects the same Steps in reversed procedural order", () => {
    const vectors = [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1]
    ];

    const result = bandedMonotonicMatch(vectors, [...vectors].reverse(), span5Config);

    expect(result.admitted).toBe(false);
    expect(result.matchedSteps).toBeLessThan(4);
  });

  it("provides deterministic self alignment and safe cosine semantics", () => {
    const vectors = Array.from({ length: 5 }, (_, index) =>
      Array.from({ length: 5 }, (__, offset) => Number(index === offset)));

    expect(selfBandedMonotonicMatch(vectors, span5Config)).toMatchObject({
      admitted: true,
      score: 1,
      coverage: 1,
      matchedSteps: 5
    });
    expect(cosineSimilarity([1, 0], [1, 0])).toBe(1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
    expect(cosineSimilarity([], [])).toBe(-1);
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(-1);
  });
});
