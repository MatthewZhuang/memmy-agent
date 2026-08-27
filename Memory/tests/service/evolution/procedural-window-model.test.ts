import { describe, expect, it } from "vitest";
import {
  ALIGNED_COMMON_CORE_SCHEMA_VERSION,
  EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
  EXECUTION_STEP_LITE_SCHEMA_VERSION,
  PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
  TURN_TRANSITION_SCHEMA_VERSION,
  V15_COARSE_SIMILARITY_THRESHOLDS,
  V15_FINE_MATCH_CONFIGS,
  V15_WINDOW_SPECS,
  buildTrajectoryWindows,
  clusterTrajectoryWindows,
  extractAnchoredCompletionOverlay,
  extractAlignedCommonCore,
  fineEvidenceSignature,
  ingestWindowIntoCoarseFamilies,
  matchingCoarseFamilies,
  proceduralEvidenceRoleForReward,
  selectConstrainedRealMedoid,
  selectMaximalWindowClusters,
  unitVector,
  windowIntentSequenceText,
  type EmbeddedTrajectoryWindowV1,
  type EpisodeExecutionPathLiteV1,
  type ExecutionStepLiteV1,
  type ProceduralWindowFamilyV1,
  type TrajectoryWindowClusterV1,
  type TrajectoryWindowOccurrenceV1
} from "../../../src/service/evolution/procedural-window-model.js";
import {
  bandedMonotonicMatch,
  selfBandedMonotonicMatch
} from "../../../src/service/evolution/trajectory-window-alignment.js";

describe("procedural multi-scale windows", () => {
  it("pins the quality-validated v15 parameters", () => {
    expect(V15_WINDOW_SPECS).toEqual([
      { length: 5, stride: 2 },
      { length: 10, stride: 5 }
    ]);
    expect(V15_COARSE_SIMILARITY_THRESHOLDS).toEqual({ 5: 0.76, 10: 0.70 });
    expect(V15_FINE_MATCH_CONFIGS).toEqual([
      expect.objectContaining({
        scale: 5,
        bandWidth: 1,
        minStepSimilarity: 0.70,
        minMatchedSteps: 4,
        minCoverage: 0.80,
        minAverageMatchSimilarity: 0.78,
        maxInternalGap: 1,
        gapPenalty: 0.10,
        minAlignmentScore: 0.62
      }),
      expect.objectContaining({
        scale: 10,
        bandWidth: 2,
        minStepSimilarity: 0.68,
        minMatchedSteps: 7,
        minCoverage: 0.70,
        minAverageMatchSimilarity: 0.76,
        maxInternalGap: 2,
        gapPenalty: 0.10,
        minAlignmentScore: 0.52
      })
    ]);
  });

  it("treats 20/30/20 Steps as one Episode path and permits cross-Turn windows", () => {
    const path = makePath("episode-70", 70, {
      turnForStep: (index) => index < 20 ? 0 : index < 50 ? 1 : 2
    });

    const windows = buildTrajectoryWindows([path]);

    expect(windows.filter((window) => window.scale === 5)).toHaveLength(34);
    expect(windows.filter((window) => window.scale === 10)).toHaveLength(13);
    expect(windows).toHaveLength(47);
    expect(windows).toContainEqual(expect.objectContaining({
      scale: 5,
      startStepIndex: 18,
      endStepIndex: 22
    }));
    const crossTurn = windows.find((window) =>
      window.scale === 5 && window.startStepIndex === 18);
    expect(new Set(crossTurn?.steps.map((step) => step.turnIndex))).toEqual(new Set([0, 1]));
    expect(windows.filter((window) => window.scale === 5).at(-1)).toMatchObject({
      startStepIndex: 65,
      endStepIndex: 69
    });
  });

  it("never crosses Episode boundaries and embeds only the ordered intent sequence", () => {
    const first = makePath("episode-a", 6);
    const second = makePath("episode-b", 6);
    const windows = buildTrajectoryWindows([first, second], [{ length: 5, stride: 2 }]);

    expect(windows).toHaveLength(4);
    expect(windows.every((window) =>
      window.steps.every((step) => step.episodeId === window.episodeId))).toBe(true);
    expect(windowIntentSequenceText(first.steps.slice(0, 5))).toContain("1. intent-0");
    expect(windowIntentSequenceText(first.steps.slice(0, 5))).not.toContain("summary-");
  });

  it("adds a new window to every existing coarse medoid above the threshold", () => {
    const first = embeddedWindow("episode-a", 0, unitVector([1, 0]), 1);
    const second = embeddedWindow("episode-b", 0, unitVector([0.8, 0.6]), 1);
    const incoming = embeddedWindow("episode-c", 0, unitVector([0.9, 0.3]), 1);
    const families: ProceduralWindowFamilyV1[] = [
      makeFamily("family-a", first),
      makeFamily("family-b", second)
    ];

    expect(matchingCoarseFamilies(incoming, families).map((match) => match.familyId))
      .toEqual(["family-a", "family-b"]);
    const ingested = ingestWindowIntoCoarseFamilies(families, incoming);

    expect(ingested.matchedFamilyIds).toEqual(["family-a", "family-b"]);
    expect(ingested.families.map((family) => family.members.length)).toEqual([2, 2]);
    expect(ingested.families.every((family) =>
      family.members.some((member) => member.occurrence.id === family.medoid.occurrence.id)))
      .toBe(true);
  });

  it("does not let counterexample or unknown evidence seed a coarse Family", () => {
    const failed = embeddedWindow("episode-failed", 0, [1, 0], -1);
    const unknown = embeddedWindow("episode-unknown", 0, [0, 1], undefined);

    expect(ingestWindowIntoCoarseFamilies([], failed)).toMatchObject({
      families: [],
      matchedFamilyIds: []
    });
    expect(ingestWindowIntoCoarseFamilies([], unknown)).toMatchObject({
      families: [],
      matchedFamilyIds: []
    });
  });

  it("deduplicates identical Fine evidence reached through multiple Families", () => {
    const angle = (degrees: number) => degrees * Math.PI / 180;
    const windows = [
      isolatedFineWindow(
        "episode-a-anchor",
        [Math.cos(angle(0)), Math.sin(angle(0))],
        0
      ),
      isolatedFineWindow(
        "episode-b-anchor",
        [Math.cos(angle(80)), Math.sin(angle(80))],
        5
      ),
      isolatedFineWindow(
        "episode-c-bridge",
        [Math.cos(angle(40)), Math.sin(angle(40))],
        10
      ),
      isolatedFineWindow(
        "episode-d-bridge",
        [Math.cos(angle(40)), Math.sin(angle(40))],
        10
      )
    ];

    const forward = clusterTrajectoryWindows(windows);
    const reversed = clusterTrajectoryWindows([...windows].reverse());
    const bridgeOccurrenceIds = windows.slice(2).map((window) => window.occurrence.id).sort();

    expect(forward.families).toHaveLength(2);
    expect(forward.families.filter((family) => bridgeOccurrenceIds.every((occurrenceId) =>
      family.members.some((member) => member.occurrence.id === occurrenceId)))).toHaveLength(2);

    // Both coarse Families yield the same two-member Fine evidence set. It is
    // canonical evidence, so the downstream cluster/Skill candidate appears once.
    const bridgeClusters = forward.clusters.filter((cluster) =>
      cluster.members.map((member) => member.occurrence.id).sort().join("|") ===
        bridgeOccurrenceIds.join("|")
    );
    expect(bridgeClusters).toHaveLength(1);
    expect(forward.clusters).toHaveLength(3);

    const canonicalBridge = bridgeClusters[0]!;
    expect(fineEvidenceSignature({
      ...canonicalBridge,
      members: [...canonicalBridge.members].reverse()
    })).toBe(fineEvidenceSignature(canonicalBridge));

    // Stable evidence signatures must not depend on replay/arrival order.
    expect(clusterEvidenceSignatures(reversed)).toEqual(clusterEvidenceSignatures(forward));
    expect(familyEvidenceSignatures(reversed)).toEqual(familyEvidenceSignatures(forward));
  });

  it("uses a real constrained medoid and refuses similarity chaining", () => {
    const members = [
      embeddedWindow("episode-a", 0, unitVector([1, 0]), 1),
      embeddedWindow("episode-b", 0, unitVector([0.866, 0.5]), 1),
      embeddedWindow("episode-c", 0, unitVector([0.5, 0.866]), 1),
      embeddedWindow("episode-d", 0, unitVector([0, 1]), 1)
    ];

    expect(selectConstrainedRealMedoid(members, 0.76)).toBeUndefined();
    const admitted = selectConstrainedRealMedoid(members.slice(0, 2), 0.76);
    expect(admitted).toBeDefined();
    expect(members.slice(0, 2).map((member) => member.occurrence.id))
      .toContain(admitted?.medoid.occurrence.id);
  });

  it("keeps failed and unknown Episodes out of positive support", () => {
    const windows = [
      embeddedWindow("episode-positive", 0, [1, 0], 1),
      embeddedWindow("episode-negative", 0, [1, 0], -1),
      embeddedWindow("episode-unknown", 0, [1, 0], undefined)
    ];

    const result = clusterTrajectoryWindows(windows);
    const cluster = result.clusters.find((candidate) => candidate.occurrenceCount === 3);

    expect(cluster).toBeDefined();
    expect(cluster?.supportEpisodeIds).toEqual(["episode-positive"]);
    expect(cluster?.counterexampleEpisodeIds).toEqual(["episode-negative"]);
    expect(cluster?.unknownEpisodeIds).toEqual(["episode-unknown"]);
    expect(proceduralEvidenceRoleForReward(1)).toBe("support");
    expect(proceduralEvidenceRoleForReward(-1)).toBe("counterexample");
    expect(proceduralEvidenceRoleForReward(0)).toBe("unknown");
  });

  it("suppresses a fully covered Span-5 cluster but keeps unrelated shorter evidence", () => {
    const short = syntheticCluster("short", 5, [
      syntheticOccurrence("episode-a", 2, 6, 5),
      syntheticOccurrence("episode-b", 4, 8, 5)
    ]);
    const long = syntheticCluster("long", 10, [
      syntheticOccurrence("episode-a", 0, 9, 10),
      syntheticOccurrence("episode-b", 0, 9, 10)
    ]);
    const unrelated = syntheticCluster("unrelated", 5, [
      syntheticOccurrence("episode-a", 20, 24, 5),
      syntheticOccurrence("episode-b", 20, 24, 5)
    ]);

    const selected = selectMaximalWindowClusters([short, long, unrelated]);

    expect(selected.find((cluster) => cluster.id === "short")?.suppressedByClusterId).toBe("long");
    expect(selected.find((cluster) => cluster.id === "unrelated")?.suppressedByClusterId)
      .toBeUndefined();
  });

  it("extracts an aligned common core and projects internal Gap Steps", () => {
    const config = V15_FINE_MATCH_CONFIGS[0]!;
    const anchor = occurrenceWithIntents("episode-anchor", ["a", "b", "c", "d", "e"], 1);
    const shifted = occurrenceWithIntents(
      "episode-shifted",
      ["a", "b", "detour", "c", "d"],
      1
    );
    const negative = occurrenceWithIntents(
      "episode-negative",
      ["a", "b", "detour", "c", "d"],
      -1
    );
    const basis: Record<string, number[]> = {
      a: [1, 0, 0, 0, 0, 0],
      b: [0, 1, 0, 0, 0, 0],
      c: [0, 0, 1, 0, 0, 0],
      d: [0, 0, 0, 1, 0, 0],
      e: [0, 0, 0, 0, 1, 0],
      detour: [0, 0, 0, 0, 0, 1]
    };
    const vectors = (occurrence: TrajectoryWindowOccurrenceV1) =>
      occurrence.steps.map((step) => basis[step.intent]!);
    const anchorVectors = vectors(anchor);
    const shiftedAlignment = bandedMonotonicMatch(vectors(shifted), anchorVectors, config);
    const negativeAlignment = bandedMonotonicMatch(vectors(negative), anchorVectors, config);
    expect(shiftedAlignment.admitted).toBe(true);
    const cluster: TrajectoryWindowClusterV1 = {
      id: "cluster-core",
      familyId: "family-core",
      scale: 5,
      medoidOccurrenceId: anchor.id,
      episodeIds: [anchor.episodeId, shifted.episodeId, negative.episodeId].sort(),
      supportEpisodeIds: [anchor.episodeId, shifted.episodeId].sort(),
      counterexampleEpisodeIds: [negative.episodeId],
      unknownEpisodeIds: [],
      occurrenceCount: 3,
      averageSimilarity: 0.8,
      minimumSimilarity: 0.7,
      medoidCentrality: 0.8,
      medoidUpdateCount: 0,
      members: [
        {
          occurrence: anchor,
          similarityToMedoid: 1,
          alignmentToMedoid: selfBandedMonotonicMatch(anchorVectors, config)
        },
        {
          occurrence: shifted,
          similarityToMedoid: shiftedAlignment.score,
          alignmentToMedoid: shiftedAlignment
        },
        {
          occurrence: negative,
          similarityToMedoid: negativeAlignment.score,
          alignmentToMedoid: negativeAlignment
        }
      ]
    };

    const core = extractAlignedCommonCore(cluster);

    expect(core?.schemaVersion).toBe(ALIGNED_COMMON_CORE_SCHEMA_VERSION);
    expect(core?.anchorOffsets).toEqual([0, 1, 2, 3]);
    expect(core?.supportEpisodeIds).toEqual(["episode-anchor", "episode-shifted"]);
    expect(core?.spanOccurrences).toHaveLength(3);
    const projected = core?.spanOccurrences.find((span) => span.episodeId === "episode-shifted");
    expect(projected).toMatchObject({
      schemaVersion: PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
      startStepIndex: 0,
      endStepIndex: 4,
      evidenceRole: "support"
    });
    expect(projected?.gapStepIds).toEqual([shifted.steps[2]!.id]);
    expect(core?.spanOccurrences.find((span) => span.episodeId === "episode-negative")
      ?.evidenceRole).toBe("counterexample");
  });

  it("projects only repeated outward Steps as shared completion evidence", () => {
    const overlay = extractAnchoredCompletionOverlay("core-1", [
      {
        occurrenceId: "occ-a",
        episodeId: "episode-a",
        evidenceRole: "support",
        prefix: [
          { stepId: "a-prefix-shared", vector: [1, 0, 0] },
          { stepId: "a-prefix-local", vector: [0, 1, 0] }
        ],
        suffix: [{ stepId: "a-suffix-shared", vector: [0, 0, 1] }]
      },
      {
        occurrenceId: "occ-b",
        episodeId: "episode-b",
        evidenceRole: "support",
        prefix: [
          { stepId: "b-prefix-shared", vector: [1, 0, 0] },
          { stepId: "b-prefix-local", vector: [0.5, 0.5, 0] }
        ],
        suffix: [{ stepId: "b-suffix-shared", vector: [0, 0, 1] }]
      }
    ], {
      referenceOccurrenceId: "occ-a",
      maxPrefixSteps: 2,
      maxSuffixSteps: 1,
      minStepSimilarity: 0.8,
      minSupportEpisodes: 2
    });

    expect(overlay?.sharedPrefix).toHaveLength(1);
    expect(overlay?.sharedSuffix).toHaveLength(1);
    expect(overlay?.sharedPrefix[0]).toMatchObject({
      referenceStepId: "a-prefix-shared",
      supportEpisodeIds: ["episode-a", "episode-b"],
      evidenceStepIds: ["a-prefix-shared", "b-prefix-shared"]
    });
    expect(overlay?.projections.find((item) => item.occurrenceId === "occ-a")?.prefix)
      .toEqual([
        expect.objectContaining({ stepId: "a-prefix-shared", role: "shared_extension" }),
        { stepId: "a-prefix-local", role: "local_context" }
      ]);
    expect(overlay?.projections.find((item) => item.occurrenceId === "occ-b")?.prefix)
      .toEqual([
        expect.objectContaining({ stepId: "b-prefix-shared", role: "shared_extension" }),
        { stepId: "b-prefix-local", role: "local_context" }
      ]);
  });

  it("treats zero expansion budget as an empty overlay", () => {
    const overlay = extractAnchoredCompletionOverlay("core-1", [
      {
        occurrenceId: "occ-a",
        episodeId: "episode-a",
        evidenceRole: "support",
        prefix: [{ stepId: "a-prefix", vector: [1, 0] }],
        suffix: [{ stepId: "a-suffix", vector: [0, 1] }]
      },
      {
        occurrenceId: "occ-b",
        episodeId: "episode-b",
        evidenceRole: "support",
        prefix: [{ stepId: "b-prefix", vector: [1, 0] }],
        suffix: [{ stepId: "b-suffix", vector: [0, 1] }]
      }
    ], {
      maxPrefixSteps: 0,
      maxSuffixSteps: 0,
      minStepSimilarity: 0.7,
      minSupportEpisodes: 2
    });

    expect(overlay?.sharedPrefix).toEqual([]);
    expect(overlay?.sharedSuffix).toEqual([]);
    expect(overlay?.projections.every((item) =>
      item.prefix.length === 0 && item.suffix.length === 0)).toBe(true);
  });
});

function makePath(
  episodeId: string,
  stepCount: number,
  options: { turnForStep?: (index: number) => number; reward?: number } = {}
): EpisodeExecutionPathLiteV1 {
  const turnForStep = options.turnForStep ?? (() => 0);
  const steps = Array.from({ length: stepCount }, (_, index) =>
    makeStep(episodeId, index, turnForStep(index)));
  const turnIndexes = [...new Set(steps.map((step) => step.turnIndex))];
  const sourceRawTurnIds = turnIndexes.map((turnIndex) => `${episodeId}-turn-${turnIndex}`);
  return {
    id: `path-${episodeId}`,
    schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
    episodeId,
    userId: "user-1",
    sourceRawTurnIds,
    steps,
    turnTransitions: turnIndexes.map((turnIndex) => {
      const indexes = steps.filter((step) => step.turnIndex === turnIndex)
        .map((step) => step.stepIndex);
      return {
        id: `${episodeId}-transition-${turnIndex}`,
        schemaVersion: TURN_TRANSITION_SCHEMA_VERSION,
        episodeId,
        rawTurnId: `${episodeId}-turn-${turnIndex}`,
        turnIndex,
        ...(indexes[0] === undefined ? {} : { afterStepIndex: indexes[0] }),
        ...(indexes[0] === undefined || indexes[0] === 0
          ? {}
          : { beforeStepIndex: indexes[0] - 1 }),
        userObservation: `turn-${turnIndex}-query`
      };
    }),
    ...(options.reward === undefined ? {} : { terminalReward: options.reward }),
    sourceSnapshotHash: `source-${episodeId}`,
    pathHash: `hash-${episodeId}`,
    provenance: { algorithmVersion: "test.v1" }
  };
}

function makeStep(episodeId: string, stepIndex: number, turnIndex = 0): ExecutionStepLiteV1 {
  return {
    id: `${episodeId}-step-${stepIndex}`,
    schemaVersion: EXECUTION_STEP_LITE_SCHEMA_VERSION,
    episodeId,
    rawTurnId: `${episodeId}-turn-${turnIndex}`,
    turnIndex,
    stepIndex,
    kind: "tool_action",
    toolName: "test_tool",
    toolCallIndex: stepIndex,
    intent: `intent-${stepIndex}`,
    summary: `summary-${stepIndex}`,
    outcome: "success",
    evidenceRefs: [`evidence-${episodeId}-${stepIndex}`],
    provenance: {
      algorithmVersion: "test.v1",
      sourceSnapshotHash: `source-${episodeId}`
    }
  };
}

function embeddedWindow(
  episodeId: string,
  start: number,
  coarseVector: number[],
  reward: number | undefined
): EmbeddedTrajectoryWindowV1 {
  const path = makePath(episodeId, 5, { reward });
  const occurrence = buildTrajectoryWindows([path], [{ length: 5, stride: 1 }])[0]!;
  const stepVectors = Array.from({ length: 5 }, (_, index) =>
    Array.from({ length: 5 }, (__, offset) => Number(index === offset)));
  return {
    occurrence: {
      ...occurrence,
      id: `${occurrence.id}-${start}`,
      startStepIndex: start,
      endStepIndex: start + 4
    },
    coarseVector: unitVector(coarseVector),
    stepVectors
  };
}

function isolatedFineWindow(
  episodeId: string,
  coarseVector: number[],
  fineBasisOffset: number
): EmbeddedTrajectoryWindowV1 {
  const embedded = embeddedWindow(episodeId, 0, coarseVector, 1);
  return {
    ...embedded,
    stepVectors: Array.from({ length: 5 }, (_, stepIndex) =>
      Array.from({ length: 15 }, (__, basisIndex) =>
        Number(basisIndex === fineBasisOffset + stepIndex)))
  };
}

function clusterEvidenceSignatures(result: ReturnType<typeof clusterTrajectoryWindows>): string[] {
  return result.clusters.map((cluster) => `${cluster.scale}:${cluster.members
    .map((member) => member.occurrence.id).sort().join("|")}`).sort();
}

function familyEvidenceSignatures(result: ReturnType<typeof clusterTrajectoryWindows>): string[] {
  return result.families.map((family) => `${family.scale}:${family.members
    .map((member) => member.occurrence.id).sort().join("|")}`).sort();
}

function makeFamily(
  id: string,
  medoid: EmbeddedTrajectoryWindowV1
): ProceduralWindowFamilyV1 {
  return {
    id,
    scale: medoid.occurrence.scale,
    medoid,
    members: [medoid],
    medoidCentrality: 1,
    minimumSimilarityToMedoid: 1,
    medoidUpdateCount: 0
  };
}

function syntheticOccurrence(
  episodeId: string,
  start: number,
  end: number,
  scale: number
): TrajectoryWindowOccurrenceV1 {
  const path = makePath(episodeId, Math.max(end + 1, scale), { reward: 1 });
  const steps = path.steps.slice(start, end + 1);
  return {
    id: `${episodeId}-${start}-${end}`,
    schemaVersion: "trajectory-window-occurrence.v1",
    episodeId,
    pathId: path.id,
    userId: path.userId,
    terminalReward: 1,
    evidenceRole: "support",
    scale,
    stride: scale,
    startStepIndex: start,
    endStepIndex: end,
    semanticText: windowIntentSequenceText(steps),
    steps
  };
}

function syntheticCluster(
  id: string,
  scale: number,
  occurrences: TrajectoryWindowOccurrenceV1[]
): TrajectoryWindowClusterV1 {
  const config = V15_FINE_MATCH_CONFIGS.find((candidate) => candidate.scale === scale)!;
  return {
    id,
    familyId: `family-${id}`,
    scale,
    medoidOccurrenceId: occurrences[0]!.id,
    episodeIds: occurrences.map((occurrence) => occurrence.episodeId).sort(),
    supportEpisodeIds: occurrences.map((occurrence) => occurrence.episodeId).sort(),
    counterexampleEpisodeIds: [],
    unknownEpisodeIds: [],
    occurrenceCount: occurrences.length,
    averageSimilarity: 1,
    minimumSimilarity: 1,
    medoidCentrality: 1,
    medoidUpdateCount: 0,
    members: occurrences.map((occurrence) => ({
      occurrence,
      similarityToMedoid: 1,
      alignmentToMedoid: selfBandedMonotonicMatch(
        Array.from({ length: scale }, () => [1]),
        config
      )
    }))
  };
}

function occurrenceWithIntents(
  episodeId: string,
  intents: string[],
  reward: number
): TrajectoryWindowOccurrenceV1 {
  const path = makePath(episodeId, intents.length, { reward });
  const steps = path.steps.map((step, index) => ({
    ...step,
    intent: intents[index]!,
    summary: `${intents[index]}-result`
  }));
  return {
    id: `occurrence-${episodeId}`,
    schemaVersion: "trajectory-window-occurrence.v1",
    episodeId,
    pathId: path.id,
    userId: path.userId,
    terminalReward: reward,
    evidenceRole: proceduralEvidenceRoleForReward(reward),
    scale: intents.length,
    stride: intents.length,
    startStepIndex: 0,
    endStepIndex: intents.length - 1,
    semanticText: windowIntentSequenceText(steps),
    steps
  };
}
