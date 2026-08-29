import { describe, expect, it } from "vitest";

import {
  buildEpisodeTrajectoryFamily,
  extendLongCommonTrajectoryWithProjection,
  longTrajectoryCandidateStructureKey,
  mineLongCommonSpanSequences,
  projectEpisodeToReferenceSpans,
  selectMaximalLongCommonTrajectories,
  selectMonotonicSpanPath,
  trajectoryIntentSequenceText,
  type EpisodeTrajectoryDocumentV1,
  type LongCommonTrajectoryV1,
  type LongTrajectoryMiningConfigV1,
  type ReferenceSpanMatchV1
} from "../../../src/service/evolution/long-trajectory-model.js";
import { buildLongTrajectorySkillInput } from
  "../../../src/service/evolution/long-trajectory-skill-input.js";
import {
  buildTrajectoryWindows,
  type EmbeddedTrajectoryWindowV1,
  type EpisodeExecutionPathLiteV1
} from "../../../src/service/evolution/procedural-window-model.js";

describe("reference Span-sequence mining", () => {
  it("keeps the new Episode A as reference instead of replacing it with a medoid", () => {
    const seed = document("episode-a", 15);
    const longer = document("episode-b", 30);
    const family = buildEpisodeTrajectoryFamily(seed, [longer], CONFIG);

    expect(family).toMatchObject({
      seedEpisodeId: "episode-a",
      referenceEpisodeId: "episode-a",
      memberEpisodeIds: ["episode-a", "episode-b"]
    });
  });

  it("uses coarse Span similarity without a Fine Step-alignment gate", () => {
    const reference = document("episode-a", 20);
    const other = document("episode-b", 20);
    other.windows = other.windows.map((window) => ({
      ...window,
      stepVectors: window.stepVectors.map(() => [0, 1])
    }));
    reference.windows = reference.windows.map((window) => ({
      ...window,
      stepVectors: window.stepVectors.map(() => [1, 0])
    }));
    const family = buildEpisodeTrajectoryFamily(reference, [other], CONFIG)!;
    const projection = projectEpisodeToReferenceSpans(
      family.id,
      reference,
      other,
      CONFIG
    );

    expect(projection.matches.length).toBeGreaterThanOrEqual(2);
    expect(projection.matches.every((match) => match.coarseSimilarity === 1)).toBe(true);
  });

  it("encodes B as A Span tokens and preserves an unmatched middle Gap", () => {
    const { reference, other } = gapDocuments();
    const family = buildEpisodeTrajectoryFamily(reference, [other], CONFIG)!;
    const projection = projectEpisodeToReferenceSpans(
      family.id,
      reference,
      other,
      CONFIG
    );

    expect(projection.matches.map((match) => [
      match.scale,
      match.referenceStartStepIndex,
      match.episodeStartStepIndex
    ])).toEqual([
      [5, 0, 0],
      [10, 5, 10],
      [10, 15, 20]
    ]);
    expect(projection.gaps).toHaveLength(1);
    expect(projection.gaps[0]).toMatchObject({ startStepIndex: 5, endStepIndex: 9 });
    expect(projection.gaps[0]!.stepIds).toHaveLength(5);
  });

  it("prefers a longer non-overlapping mixed-scale path", () => {
    const path = selectMonotonicSpanPath([
      match("short-a", 5, 0, 4, 0, 4, 0.9),
      match("short-b", 5, 5, 9, 5, 9, 0.9),
      match("long-a", 10, 0, 9, 0, 9, 0.9),
      match("long-b", 10, 10, 19, 10, 19, 0.9),
      match("reversed", 5, 20, 24, 0, 4, 1)
    ]);

    expect(path.map((item) => item.id)).toEqual(["long-a", "long-b"]);
  });

  it("mines a long Skill candidate from the repeated A-token sequence", () => {
    const { reference, other } = gapDocuments();
    const family = buildEpisodeTrajectoryFamily(reference, [other], CONFIG)!;
    const projection = projectEpisodeToReferenceSpans(
      family.id,
      reference,
      other,
      CONFIG
    );
    const trajectories = mineLongCommonSpanSequences(
      family,
      [reference, other],
      [projection],
      CONFIG
    );

    expect(trajectories).toHaveLength(1);
    expect(trajectories[0]).toMatchObject({
      referenceEpisodeId: "episode-a",
      referenceStartStepIndex: 0,
      referenceEndStepIndex: 24,
      supportEpisodeIds: ["episode-a", "episode-b"]
    });
    expect(trajectories[0]!.requiredSpans.map((span) => span.scale))
      .toEqual([5, 10, 10]);
    expect(trajectories[0]!.occurrences.find((item) => item.episodeId === "episode-b")
      ?.gapStepIds).toHaveLength(5);
  });

  it("evolves one Candidate when Episode C matches its complete retained Span sequence", () => {
    const { reference, other } = gapDocuments();
    const family = buildEpisodeTrajectoryFamily(reference, [other], CONFIG)!;
    const firstProjection = projectEpisodeToReferenceSpans(
      family.id,
      reference,
      other,
      CONFIG
    );
    const original = mineLongCommonSpanSequences(
      family,
      [reference, other],
      [firstProjection],
      CONFIG
    )[0]!;
    const episodeC = {
      ...other,
      path: {
        ...other.path,
        id: "path-episode-c",
        episodeId: "episode-c"
      }
    };
    const nextProjection = {
      ...firstProjection,
      id: `candidate-projection-${original.id}`,
      episodeId: "episode-c"
    };
    const evolved = extendLongCommonTrajectoryWithProjection({
      trajectory: original,
      episode: episodeC,
      projection: nextProjection
    });

    expect(evolved.candidateStructureKey).toBe(original.candidateStructureKey);
    expect(evolved.structureHash).not.toBe(original.structureHash);
    expect(evolved.supportEpisodeIds).toEqual([
      "episode-a",
      "episode-b",
      "episode-c"
    ]);
    expect(evolved.occurrences.map((item) => item.episodeId)).toContain("episode-c");
  });

  it("keeps a Coarse V3 candidate and gives the LLM every matched-Span Step", () => {
    const { reference, other } = gapDocuments();
    const family = buildEpisodeTrajectoryFamily(reference, [other], CONFIG)!;
    const projection = projectEpisodeToReferenceSpans(
      family.id,
      reference,
      other,
      CONFIG
    );
    const mined = mineLongCommonSpanSequences(
      family,
      [reference, other],
      [projection],
      CONFIG
    )[0]!;
    const skillInput = buildLongTrajectorySkillInput({
      trajectory: mined,
      documents: [reference, other],
      userId: reference.path.userId,
      sourceTraceIdsForSteps: (episodeId) => [`trace-${episodeId}`]
    })!;
    const otherEvidence = skillInput.evidence.find((item) =>
      item.episodeId === other.path.episodeId)!;

    expect(skillInput.supportEpisodeIds).toEqual(["episode-a", "episode-b"]);
    expect(skillInput.commonCore).toHaveLength(mined.requiredSpans.length);
    expect(otherEvidence.alignedSequence.some((step) => step.role === "span_step")).toBe(true);
    expect(otherEvidence.alignedSequence.some((step) => step.role === "local_context")).toBe(false);
    expect(otherEvidence.alignedSequence.filter((step) => step.role !== "gap")
      .every((step) => step.role === "span_step" && Boolean(step.anchorId))).toBe(true);
  });

  it("keeps Candidate identity stable when only Episode and Span evidence IDs change", () => {
    const first = trajectory("first-evidence", [[0, 4], [5, 14]]).requiredSpans;
    const second = first.map((span, index) => ({
      ...span,
      anchorId: `other-anchor-${index}`,
      referenceSpanId: `other-span-${index}`,
      referenceSpanLabel: `other-label-${index}`,
      supportEpisodeIds: ["episode-c", "episode-d"],
      evidenceStepIds: [`other-step-${index}`]
    }));
    expect(longTrajectoryCandidateStructureKey(second))
      .toBe(longTrajectoryCandidateStructureKey(first));
    expect(longTrajectoryCandidateStructureKey([{
      ...second[0]!,
      scale: second.reduce((sum, span) => sum + span.scale, 0),
      semanticText: second.map((span) => span.semanticText).join("\n")
    }])).toBe(longTrajectoryCandidateStructureKey(first));
    expect(longTrajectoryCandidateStructureKey([
      ...second.slice(0, 1),
      { ...second[1]!, semanticText: "execute a different capability" }
    ])).not.toBe(longTrajectoryCandidateStructureKey(first));
  });

  it("keeps ABC and a disjoint EF path while suppressing contained AB and BC paths", () => {
    const selected = selectMaximalLongCommonTrajectories([
      trajectory("ab", [[0, 4], [5, 9]]),
      trajectory("bc", [[5, 9], [10, 14]]),
      trajectory("abc", [[0, 4], [5, 9], [10, 14]]),
      trajectory("ef", [[20, 24], [25, 29]])
    ]);

    expect(selected.map((item) => item.id)).toEqual(["abc", "ef"]);
  });

  it("does not suppress a contained path backed by a different Episode set", () => {
    const selected = selectMaximalLongCommonTrajectories([
      trajectory("ab-distinct-support", [[0, 4], [5, 9]], ["episode-a", "episode-b"]),
      trajectory(
        "abc-expanded-support",
        [[0, 4], [5, 9], [10, 14]],
        ["episode-a", "episode-b", "episode-c"]
      )
    ]);

    expect(selected.map((item) => item.id).sort()).toEqual([
      "ab-distinct-support",
      "abc-expanded-support"
    ]);
  });

  it("keeps partially overlapping paths when neither contains the other", () => {
    const selected = selectMaximalLongCommonTrajectories([
      trajectory("ab-partial", [[0, 4], [5, 9]]),
      trajectory("bc-partial", [[5, 9], [10, 14]])
    ]);

    expect(selected.map((item) => item.id).sort()).toEqual([
      "ab-partial",
      "bc-partial"
    ]);
  });

  it("does not form a Family when the seed has no similar Episode", () => {
    const seed = document("episode-a");
    const unrelated = {
      ...document("episode-z"),
      goalVector: [0, 1],
      trajectoryVector: [0, 1]
    };

    expect(buildEpisodeTrajectoryFamily(seed, [unrelated], CONFIG)).toBeUndefined();
  });

  it("does not use a failed Episode as positive support", () => {
    const reference = document("episode-a", 20, 1);
    const failed = document("episode-b", 20, -1);
    const family = buildEpisodeTrajectoryFamily(reference, [failed], CONFIG)!;
    const projection = projectEpisodeToReferenceSpans(
      family.id,
      reference,
      failed,
      CONFIG
    );

    expect(mineLongCommonSpanSequences(
      family,
      [reference, failed],
      [projection],
      CONFIG
    )).toEqual([]);
  });
});

const CONFIG: LongTrajectoryMiningConfigV1 = {
  episodeRecallLimit: 10,
  minEpisodeSimilarity: 0.65,
  minGoalSimilarity: 0.55,
  goalWeight: 0.4,
  trajectoryWeight: 0.6,
  windowTopK: 3,
  coarseThresholds: { 5: 0.76, 10: 0.7, 15: 0.68 },
  minSupportEpisodes: 2,
  minSpanSequenceLength: 2,
  minTrajectorySpanSteps: 12,
  minEpisodeCoverage: 0.5
};

const WINDOW_SPECS = [
  { length: 5, stride: 5 },
  { length: 10, stride: 5 },
  { length: 15, stride: 7 }
] as const;

function document(
  episodeId: string,
  stepCount = 20,
  reward = 1
): EpisodeTrajectoryDocumentV1 {
  const path = pathOf(episodeId, stepCount, reward);
  const occurrences = buildTrajectoryWindows([path], WINDOW_SPECS);
  const windows: EmbeddedTrajectoryWindowV1[] = occurrences.map((occurrence) => ({
    occurrence,
    coarseVector: oneHot(
      occurrence.scale * 100 + occurrence.startStepIndex,
      2000
    ),
    stepVectors: occurrence.steps.map((_, index) => oneHot(index, occurrence.scale))
  }));
  return {
    path,
    goalText: "Generate and verify a structured report",
    terminalResultText: "The structured report was generated and verified.",
    goalVector: [1, 0],
    trajectoryText: trajectoryIntentSequenceText(path),
    trajectoryVector: [1, 0],
    windows
  };
}

function trajectory(
  id: string,
  ranges: ReadonlyArray<readonly [number, number]>,
  supportEpisodeIds: string[] = ["episode-a", "episode-b"]
): LongCommonTrajectoryV1 {
  const requiredSpans = ranges.map(([start, end], index) => ({
    anchorId: `${id}-anchor-${index}`,
    referenceSpanId: `${id}-span-${index}`,
    referenceSpanLabel: `${id}-${index}`,
    scale: end - start + 1,
    referenceStartStepIndex: start,
    referenceEndStepIndex: end,
    semanticText: `${id} span ${index}`,
    summaryText: `${id} span ${index} completed`,
    supportEpisodeIds,
    evidenceStepIds: [`${id}-step-${index}`],
    averageCoarseSimilarity: 0.9
  }));
  return {
    id,
    schemaVersion: "long-common-span-sequence.v1",
    familyId: "family-a",
    referenceEpisodeId: "episode-a",
    referenceStartStepIndex: ranges[0]![0],
    referenceEndStepIndex: ranges.at(-1)![1],
    requiredSpans,
    supportEpisodeIds,
    occurrences: [],
    averageEpisodeCoverage: 1,
    averageCoarseSimilarity: 0.9,
    candidateStructureKey: `structure:${id}`,
    structureHash: id
  };
}

function gapDocuments(): {
  reference: EpisodeTrajectoryDocumentV1;
  other: EpisodeTrajectoryDocumentV1;
} {
  const reference = document("episode-a", 25);
  const other = document("episode-b", 30);
  const dimension = reference.windows.length + other.windows.length;
  reference.windows = reference.windows.map((window, index) => ({
    ...window,
    coarseVector: oneHot(index, dimension)
  }));
  other.windows = other.windows.map((window, index) => ({
    ...window,
    coarseVector: oneHot(reference.windows.length + index, dimension)
  }));
  copyWindowVector(reference, other, 5, 0, 0);
  copyWindowVector(reference, other, 10, 5, 10);
  copyWindowVector(reference, other, 10, 15, 20);
  return { reference, other };
}

function copyWindowVector(
  reference: EpisodeTrajectoryDocumentV1,
  other: EpisodeTrajectoryDocumentV1,
  scale: number,
  referenceStart: number,
  otherStart: number
): void {
  const left = reference.windows.find((window) =>
    window.occurrence.scale === scale &&
    window.occurrence.startStepIndex === referenceStart)!;
  const right = other.windows.find((window) =>
    window.occurrence.scale === scale &&
    window.occurrence.startStepIndex === otherStart)!;
  right.coarseVector = left.coarseVector;
}

function pathOf(episodeId: string, stepCount: number, reward: number): EpisodeExecutionPathLiteV1 {
  const rawTurnId = `raw-${episodeId}`;
  const steps = Array.from({ length: stepCount }, (_, index) => ({
    id: `${episodeId}-step-${index}`,
    episodeId,
    rawTurnId,
    turnIndex: 0,
    stepIndex: index,
    kind: "tool_action" as const,
    toolName: "shell",
    toolCallIndex: index,
    intent: `Perform reusable report operation ${index}`,
    summary: `Report operation ${index} completed`,
    outcome: "success" as const,
    evidenceRefs: [`${rawTurnId}:tool:${index}`],
    provenance: { algorithmVersion: "test", sourceSnapshotHash: "snapshot" }
  }));
  return {
    id: `path-${episodeId}`,
    schemaVersion: "episode-execution-path-lite.v1",
    episodeId,
    userId: "user-long-trajectory",
    sourceRawTurnIds: [rawTurnId],
    steps,
    turnTransitions: [],
    terminalReward: reward,
    sourceSnapshotHash: "snapshot",
    pathHash: `hash-${episodeId}`,
    provenance: { algorithmVersion: "test" }
  };
}

function oneHot(index: number, dimension: number): number[] {
  return Array.from({ length: dimension }, (_, item) => item === index ? 1 : 0);
}

function match(
  id: string,
  scale: number,
  referenceStartStepIndex: number,
  referenceEndStepIndex: number,
  episodeStartStepIndex: number,
  episodeEndStepIndex: number,
  coarseSimilarity: number
): ReferenceSpanMatchV1 {
  return {
    id,
    referenceSpanId: `${id}-reference`,
    referenceSpanLabel: `sp${scale}-${id}`,
    episodeSpanId: `${id}-episode`,
    scale,
    referenceStartStepIndex,
    referenceEndStepIndex,
    episodeStartStepIndex,
    episodeEndStepIndex,
    coarseSimilarity,
    weight: scale * coarseSimilarity
  };
}
