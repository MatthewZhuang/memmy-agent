import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  EXECUTION_STEP_LITE_SCHEMA_VERSION,
  TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION,
  V15_COARSE_SIMILARITY_THRESHOLDS,
  V15_FINE_MATCH_CONFIGS,
  V15_WINDOW_SPECS,
  clusterTrajectoryWindows,
  extractAlignedCommonCore,
  matchingCoarseFamilies,
  selectMaximalWindowClusters,
  unitVector,
  windowIntentSequenceText,
  type EmbeddedTrajectoryWindowV1,
  type ExecutionStepLiteV1,
  type ProceduralWindowFamilyV1,
  type TrajectoryWindowClusterV1,
  type TrajectoryWindowOccurrenceV1
} from "../../../src/service/evolution/procedural-window-model.js";
import type { BandedMonotonicMatchResultV1 } from
  "../../../src/service/evolution/trajectory-window-alignment.js";

interface GoldenStep {
  rawTurnId: string;
  toolCallIndex: number;
  toolName: string;
  intent: string;
  summary: string;
  outcome: ExecutionStepLiteV1["outcome"];
}

interface GoldenMember {
  stableKey: string;
  episodeId: string;
  terminalReward: number;
  startStepIndex: number;
  endStepIndex: number;
  isMedoid: boolean;
  steps: GoldenStep[];
  alignmentToMedoid: Omit<BandedMonotonicMatchResultV1, "rejectionReasons">;
}

interface GoldenCluster {
  name: string;
  scale: number;
  stableSignature: string;
  expectedCoreAnchorOffsets: number[];
  members: GoldenMember[];
}

interface GoldenFixture {
  schemaVersion: string;
  source: {
    artifact: string;
    episodeCount: number;
    stepCount: number;
    windowCount: number;
    representation: { coarse: string; fine: string };
  };
  config: {
    windowSpecs: Array<{ length: number; stride: number }>;
    coarseSimilarityThresholdByScale: Record<string, number>;
    fineMatchConfigs: typeof V15_FINE_MATCH_CONFIGS;
    minSupportEpisodes: number;
  };
  clusters: GoldenCluster[];
}

const fixture = JSON.parse(readFileSync(
  new URL("../../fixtures/v15-kw-evidence-clusters.json", import.meta.url),
  "utf8"
)) as GoldenFixture;

const EXPECTED_SIGNATURE_DIGESTS: Record<string, string> = {
  "span10-document-generation-recovery":
    "137b50409f0a1a46861783872299054539084707fbf3d32fae5fe5b81e4165da",
  "span5-pdf-generate-validate":
    "984790fcdb630c4a2cd5caf415b5c34e96fdbdb4da95a46e4b386b9a4191ca5f",
  "span5-pdf-debug-retry":
    "f95443ac91092a53fad9f59774c7df24410e1a74c9bb0761a84ec80b82191fc7",
  "span5-generic-debug-retry":
    "9b8286ab6e7838ead08bc6dccfd30031487ec0ee3aac102934b537883100ad03"
};

const EXPECTED_NON_MEDOID_ALIGNMENT = {
  "span10-document-generation-recovery": {
    score: 0.5451595425363727,
    averageMatchSimilarity: 0.7787993464805324,
    coverage: 0.7,
    matchedSteps: 7,
    pairs: [[1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9]]
  },
  "span5-pdf-generate-validate": {
    score: 0.6691557533310478,
    averageMatchSimilarity: 0.8364446916638097,
    coverage: 0.8,
    matchedSteps: 4,
    pairs: [[1, 1], [2, 2], [3, 3], [4, 4]]
  },
  "span5-pdf-debug-retry": {
    score: 0.634407124952643,
    averageMatchSimilarity: 0.7930089061908037,
    coverage: 0.8,
    matchedSteps: 4,
    pairs: [[1, 1], [2, 2], [3, 3], [4, 4]]
  },
  "span5-generic-debug-retry": {
    score: 0.6272037191898067,
    averageMatchSimilarity: 0.7840046489872583,
    coverage: 0.8,
    matchedSteps: 4,
    pairs: [[0, 1], [1, 2], [2, 3], [3, 4]]
  }
} as const;

describe("v15 KW evidence-cluster golden", () => {
  it("pins four source-coordinate signatures and the exact fine alignments", () => {
    expect(fixture.schemaVersion).toBe("v15-kw-evidence-clusters-golden.v1");
    expect(fixture.source).toMatchObject({
      episodeCount: 18,
      stepCount: 318,
      windowCount: 185,
      representation: {
        coarse: "window-intent-sequence-embedding.v1",
        fine: "step-intent-banded-monotonic.v1"
      }
    });
    expect(fixture.config.windowSpecs).toEqual(V15_WINDOW_SPECS);
    expect(fixture.config.coarseSimilarityThresholdByScale).toEqual(
      Object.fromEntries(Object.entries(V15_COARSE_SIMILARITY_THRESHOLDS)
        .map(([scale, threshold]) => [String(scale), threshold]))
    );
    expect(fixture.config.fineMatchConfigs).toEqual(V15_FINE_MATCH_CONFIGS);
    expect(fixture.clusters).toHaveLength(4);

    const actualDigests: Record<string, string> = {};
    for (const cluster of fixture.clusters) {
      expect(cluster.members).toHaveLength(2);
      expect(cluster.members.filter((member) => member.isMedoid)).toHaveLength(1);
      for (const member of cluster.members) {
        expect(member.stableKey).toBe(stableMemberKey(cluster.scale, member));
        expect(member.stableKey).not.toMatch(/trajectory_window_|step_[0-9a-f]{20}/);
      }
      expect(cluster.stableSignature).toBe(clusterSignature(cluster));
      actualDigests[cluster.name] = sha256(cluster.stableSignature);

      const expected = EXPECTED_NON_MEDOID_ALIGNMENT[
        cluster.name as keyof typeof EXPECTED_NON_MEDOID_ALIGNMENT
      ];
      expect(expected).toBeDefined();
      const nonMedoid = cluster.members.find((member) => !member.isMedoid)!;
      expect(nonMedoid.alignmentToMedoid).toMatchObject({
        admitted: true,
        score: expected.score,
        averageMatchSimilarity: expected.averageMatchSimilarity,
        coverage: expected.coverage,
        matchedSteps: expected.matchedSteps,
        internalGapSteps: 0,
        maxInternalGap: 0
      });
      expect(nonMedoid.alignmentToMedoid.pairs.map((pair) =>
        [pair.leftIndex, pair.rightIndex])).toEqual(expected.pairs);
    }

    expect(actualDigests).toEqual(EXPECTED_SIGNATURE_DIGESTS);
  });

  it("reconstructs four common cores and keeps all four evidence clusters maximal", () => {
    const clusters = fixture.clusters.map(toDomainCluster);
    const cores = clusters.map((cluster) => extractAlignedCommonCore(cluster));

    expect(cores.every(Boolean)).toBe(true);
    expect(cores).toHaveLength(4);
    for (const [index, core] of cores.entries()) {
      const golden = fixture.clusters[index]!;
      expect(core?.anchorOffsets).toEqual(golden.expectedCoreAnchorOffsets);
      expect(core?.supportEpisodeIds).toEqual(golden.members
        .map((member) => member.episodeId).sort());
      expect(core?.spanOccurrences).toHaveLength(2);
      expect(core?.spanOccurrences.every((occurrence) =>
        occurrence.matchedStepIds.length === golden.expectedCoreAnchorOffsets.length)).toBe(true);
    }

    const maximal = selectMaximalWindowClusters(clusters);
    expect(maximal).toHaveLength(4);
    expect(maximal.every((cluster) => cluster.suppressedByClusterId === undefined)).toBe(true);
    expect(maximal.map((cluster) => cluster.id).sort()).toEqual(
      fixture.clusters.map((cluster) => `golden-cluster:${cluster.name}`).sort()
    );
  });

  it("keeps multi-family coarse recall while rejecting fine and coarse hard negatives", () => {
    const familyA = syntheticFamily(
      "family-a",
      syntheticWindow("bridge-anchor-a", unitVector([1, 0]), [0, 1, 2, 3, 4])
    );
    const familyB = syntheticFamily(
      "family-b",
      syntheticWindow("bridge-anchor-b", unitVector([0.8, 0.6]), [5, 6, 7, 8, 9])
    );
    const bridge = syntheticWindow(
      "bridge",
      unitVector([0.9, 0.3]),
      [10, 11, 12, 13, 14]
    );

    expect(matchingCoarseFamilies(bridge, [familyA, familyB]).map((match) => match.familyId))
      .toEqual(["family-a", "family-b"]);

    const windows = [
      ...syntheticPair("a", [1, 0, 0, 0, 0], [0, 1, 2, 3, 4]),
      ...syntheticPair("b", [0, 1, 0, 0, 0], [5, 6, 7, 8, 9]),
      ...syntheticPair("c", [0, 0, 1, 0, 0], [10, 11, 12, 13, 14]),
      ...syntheticPair("d", [0, 0, 0, 1, 0], [15, 16, 17, 18, 19]),
      syntheticWindow("hard-fine", [1, 0, 0, 0, 0], [4, 3, 2, 1, 0]),
      syntheticWindow("hard-coarse", [0, 0, 0, 0, 1], [20, 21, 22, 23, 24])
    ];
    const result = clusterTrajectoryWindows(windows);
    const supported = result.clusters.filter((cluster) =>
      cluster.supportEpisodeIds.length >= 2);

    expect(supported).toHaveLength(4);
    expect(supported.map((cluster) => cluster.supportEpisodeIds.join("+")).sort()).toEqual([
      "episode-a-1+episode-a-2",
      "episode-b-1+episode-b-2",
      "episode-c-1+episode-c-2",
      "episode-d-1+episode-d-2"
    ]);
    for (const hardNegative of ["episode-hard-fine", "episode-hard-coarse"]) {
      expect(supported.every((cluster) =>
        !cluster.supportEpisodeIds.includes(hardNegative))).toBe(true);
      expect(result.clusters.some((cluster) =>
        cluster.occurrenceCount === 1 && cluster.supportEpisodeIds.includes(hardNegative))).toBe(true);
    }
  });
});

function stableMemberKey(scale: number, member: GoldenMember): string {
  const refs = member.steps.map((step) => `${step.rawTurnId}#${step.toolCallIndex}`).join(",");
  return `${member.episodeId}|s${scale}|${member.startStepIndex}-${member.endStepIndex}|${refs}`;
}

function clusterSignature(cluster: GoldenCluster): string {
  return cluster.members.map((member) => stableMemberKey(cluster.scale, member))
    .sort().join("||");
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function toDomainCluster(golden: GoldenCluster): TrajectoryWindowClusterV1 {
  const members = golden.members.map((member) => ({
    occurrence: toOccurrence(golden.scale, member),
    similarityToMedoid: member.alignmentToMedoid.score,
    alignmentToMedoid: {
      ...member.alignmentToMedoid,
      rejectionReasons: []
    }
  }));
  const medoid = members.find((_, index) => golden.members[index]!.isMedoid)!;
  const episodeIds = golden.members.map((member) => member.episodeId).sort();
  const similarities = members.map((member) => member.similarityToMedoid);
  return {
    id: `golden-cluster:${golden.name}`,
    familyId: `golden-family:${golden.name}`,
    scale: golden.scale,
    medoidOccurrenceId: medoid.occurrence.id,
    episodeIds,
    supportEpisodeIds: episodeIds,
    counterexampleEpisodeIds: [],
    unknownEpisodeIds: [],
    occurrenceCount: members.length,
    averageSimilarity: similarities.reduce((sum, value) => sum + value, 0) /
      similarities.length,
    minimumSimilarity: Math.min(...similarities),
    medoidCentrality: Math.min(...similarities),
    medoidUpdateCount: 0,
    members
  };
}

function toOccurrence(scale: number, member: GoldenMember): TrajectoryWindowOccurrenceV1 {
  const steps: ExecutionStepLiteV1[] = member.steps.map((step, offset) => ({
    id: `golden-step:${step.rawTurnId}:${step.toolCallIndex}`,
    schemaVersion: EXECUTION_STEP_LITE_SCHEMA_VERSION,
    episodeId: member.episodeId,
    rawTurnId: step.rawTurnId,
    turnIndex: 0,
    stepIndex: member.startStepIndex + offset,
    kind: "tool_action",
    toolName: step.toolName,
    toolCallIndex: step.toolCallIndex,
    intent: step.intent,
    summary: step.summary,
    outcome: step.outcome,
    evidenceRefs: [`raw_turn:${step.rawTurnId}:tool_call:${step.toolCallIndex}`],
    provenance: {
      algorithmVersion: "v15-kw-golden.v1",
      sourceSnapshotHash: `golden-source:${member.episodeId}`
    }
  }));
  const stride = fixture.config.windowSpecs.find((spec) => spec.length === scale)?.stride;
  if (!stride) throw new Error(`missing golden stride for scale ${scale}`);
  return {
    id: `golden-occurrence:${sha256(member.stableKey).slice(0, 24)}`,
    schemaVersion: TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION,
    episodeId: member.episodeId,
    pathId: `golden-path:${member.episodeId}`,
    userId: "golden-user",
    terminalReward: member.terminalReward,
    evidenceRole: "support",
    scale,
    stride,
    startStepIndex: member.startStepIndex,
    endStepIndex: member.endStepIndex,
    semanticText: windowIntentSequenceText(steps),
    steps
  };
}

function syntheticPair(
  name: string,
  coarseVector: number[],
  stepBasisIndexes: number[]
): EmbeddedTrajectoryWindowV1[] {
  return [1, 2].map((suffix) =>
    syntheticWindow(`${name}-${suffix}`, coarseVector, stepBasisIndexes));
}

function syntheticWindow(
  episodeSuffix: string,
  coarseVector: number[],
  stepBasisIndexes: number[]
): EmbeddedTrajectoryWindowV1 {
  const episodeId = `episode-${episodeSuffix}`;
  const steps: ExecutionStepLiteV1[] = stepBasisIndexes.map((basisIndex, stepIndex) => ({
    id: `${episodeId}-step-${stepIndex}`,
    schemaVersion: EXECUTION_STEP_LITE_SCHEMA_VERSION,
    episodeId,
    rawTurnId: `${episodeId}-turn`,
    turnIndex: 0,
    stepIndex,
    kind: "tool_action",
    toolName: "synthetic_tool",
    toolCallIndex: stepIndex,
    intent: `operation-${basisIndex}`,
    summary: `operation-${basisIndex}-result`,
    outcome: "success",
    evidenceRefs: [`${episodeId}:${stepIndex}`],
    provenance: {
      algorithmVersion: "synthetic.v1",
      sourceSnapshotHash: `source-${episodeId}`
    }
  }));
  const occurrence: TrajectoryWindowOccurrenceV1 = {
    id: `occurrence-${episodeId}`,
    schemaVersion: TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION,
    episodeId,
    pathId: `path-${episodeId}`,
    userId: "synthetic-user",
    terminalReward: 1,
    evidenceRole: "support",
    scale: 5,
    stride: 1,
    startStepIndex: 0,
    endStepIndex: 4,
    semanticText: windowIntentSequenceText(steps),
    steps
  };
  return {
    occurrence,
    coarseVector: unitVector(coarseVector),
    stepVectors: stepBasisIndexes.map((basisIndex) => basisVector(25, basisIndex))
  };
}

function syntheticFamily(
  id: string,
  medoid: EmbeddedTrajectoryWindowV1
): ProceduralWindowFamilyV1 {
  return {
    id,
    scale: 5,
    medoid,
    members: [medoid],
    medoidCentrality: 1,
    minimumSimilarityToMedoid: 1,
    medoidUpdateCount: 0
  };
}

function basisVector(dimension: number, index: number): number[] {
  return Array.from({ length: dimension }, (_, offset) => Number(offset === index));
}
