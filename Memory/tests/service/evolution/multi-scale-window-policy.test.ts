import { describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  DEFAULT_MULTI_SCALE_WINDOW_FINE_MATCH_CONFIGS,
  MultiScaleWindowPolicyExperiment,
  bandedMonotonicMatch,
  buildTrajectoryWindows,
  clusterTrajectoryWindows,
  selectMaximalWindowClusters,
  type Embedder,
  type EmbeddedTrajectoryWindowV1,
  type ExecutionStepV1,
  type LlmClient,
  type LlmMessage,
  type MultiScaleWindowEpisodeInput,
  type MultiScaleWindowPolicyV1,
  type MultiScaleWindowSkillCandidateV1,
  type BandedMonotonicMatchConfig
} from "../../../src/index.js";

describe("multi-scale trajectory window Policy experiment", () => {
  it("builds deterministic Span-5 and Span-10 windows and includes the trajectory tail", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 12)
    ], [
      { length: 5, stride: 2 },
      { length: 10, stride: 5 }
    ]);

    expect(windows.map((window) => [
      window.scale,
      window.startStepIndex,
      window.endStepIndex
    ])).toEqual([
      [5, 0, 4],
      [5, 2, 6],
      [5, 4, 8],
      [5, 6, 10],
      [5, 7, 11],
      [10, 0, 9],
      [10, 2, 11]
    ]);
    expect(windows[0]!.semanticText).toBe([
      "1. perform operation 0",
      "2. perform operation 1",
      "3. perform operation 2",
      "4. perform operation 3",
      "5. perform operation 4"
    ].join("\n"));
  });

  it("embeds the ordered intent sequence once per coarse window and excludes Step summaries", async () => {
    const embeddedTexts: string[] = [];
    const experiment = new MultiScaleWindowPolicyExperiment({
      config: DEFAULT_MEMMY_CONFIG,
      embedder: recordingSequenceEmbedder(embeddedTexts),
      llm: unusedLlm()
    });
    const intents = ["inspect", "edit", "execute", "verify", "finish"];
    const result = await experiment.run({
      episodes: [
        episodeWithDomain("episode-pdf-a", intents, "PDF document"),
        episodeWithDomain("episode-package-a", intents, "package dependency")
      ],
      specs: [{ length: 5, stride: 5 }],
      coarseSimilarityThreshold: 0.9,
      minSupportEpisodes: 2,
      inducePolicies: false
    });

    expect(result.representations).toEqual({
      coarse: "window-intent-sequence-embedding.v1",
      fine: "step-intent-banded-monotonic.v1"
    });
    expect(result.families).toHaveLength(1);
    const supported = result.clusters.filter((cluster) => cluster.supportEpisodeIds.length >= 2);
    expect(supported).toHaveLength(1);
    const sequenceText = intents.map((intent, index) => `${index + 1}. ${intent}`).join("\n");
    expect(embeddedTexts.filter((text) => text === sequenceText)).toHaveLength(2);
    expect(embeddedTexts.every((text) => !text.includes("PDF document") &&
      !text.includes("package dependency"))).toBe(true);
  });

  it("uses a constrained real medoid to stop coarse Family chaining", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 5),
      episode("episode-b", 5),
      episode("episode-c", 5),
      episode("episode-d", 5)
    ], [{ length: 5, stride: 5 }]);
    const directions = new Map([
      ["episode-a", direction(-30)],
      ["episode-b", direction(0)],
      ["episode-c", direction(30)],
      ["episode-d", direction(60)]
    ]);
    const embedded: EmbeddedTrajectoryWindowV1[] = windows.map((occurrence) => ({
      occurrence,
      coarseVector: directions.get(occurrence.episodeId)!,
      stepVectors: occurrence.steps.map(() => [1, 0])
    }));
    const result = clusterTrajectoryWindows(embedded, clusteringOptions([5], 0.8));

    expect(result.families).toHaveLength(2);
    expect(result.families.map((family) => family.occurrenceCount)).toEqual([3, 1]);
    const main = result.families[0]!;
    expect(main.medoidOccurrenceId).toBe(
      windows.find((window) => window.episodeId === "episode-b")!.id
    );
    expect(main.minimumSimilarityToMedoid).toBeGreaterThanOrEqual(0.8);
    expect(main.medoidUpdateCount).toBeGreaterThan(0);
    expect(main.memberOccurrenceIds).not.toContain(
      windows.find((window) => window.episodeId === "episode-d")!.id
    );
  });

  it("adds one new window to every existing coarse medoid above the threshold", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 5),
      episode("episode-b", 5),
      episode("episode-x", 5)
    ], [{ length: 5, stride: 5 }]);
    const directions = new Map([
      ["episode-a", direction(0)],
      ["episode-b", direction(80)],
      ["episode-x", direction(40)]
    ]);
    const embedded: EmbeddedTrajectoryWindowV1[] = windows.map((occurrence) => ({
      occurrence,
      coarseVector: directions.get(occurrence.episodeId)!,
      stepVectors: occurrence.steps.map(() => [1, 0])
    }));
    const result = clusterTrajectoryWindows(embedded, {
      ...clusteringOptions([5], 0.76),
      coarseMembershipMode: "multi"
    });
    const xId = windows.find((window) => window.episodeId === "episode-x")!.id;

    expect(result.families).toHaveLength(2);
    expect(result.families.every((family) => family.memberOccurrenceIds.includes(xId)))
      .toBe(true);
    expect(result.families.map((family) => family.occurrenceCount)).toEqual([2, 2]);
    expect(result.clusters.filter((cluster) =>
      cluster.supportEpisodeIds.length >= 2)).toHaveLength(2);
  });

  it("keeps every threshold-valid fine anchor neighborhood in overlap mode", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 5),
      episode("episode-b", 5),
      episode("episode-x", 5)
    ], [{ length: 5, stride: 5 }]);
    const directions = new Map([
      ["episode-a", direction(0)],
      ["episode-b", direction(76)],
      ["episode-x", direction(38)]
    ]);
    const embedded: EmbeddedTrajectoryWindowV1[] = windows.map((occurrence) => ({
      occurrence,
      coarseVector: [1, 0],
      stepVectors: occurrence.steps.map(() => directions.get(occurrence.episodeId)!)
    }));
    const exclusive = clusterTrajectoryWindows(embedded, clusteringOptions([5], 0.9));
    const overlapping = clusterTrajectoryWindows(embedded, {
      ...clusteringOptions([5], 0.9),
      fineMembershipMode: "overlap"
    });
    const xId = windows.find((window) => window.episodeId === "episode-x")!.id;

    expect(exclusive.clusters.filter((cluster) =>
      cluster.supportEpisodeIds.length >= 2)).toHaveLength(1);
    const supportedOverlapping = overlapping.clusters.filter((cluster) =>
      cluster.supportEpisodeIds.length >= 2);
    expect(supportedOverlapping).toHaveLength(3);
    expect(supportedOverlapping.every((cluster) => cluster.members.some((member) =>
      member.occurrence.id === xId))).toBe(true);
  });

  it("uses a lower coarse threshold only for Span-10", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 10),
      episode("episode-b", 10)
    ], [{ length: 10, stride: 10 }]);
    const directions = new Map([
      ["episode-a", direction(0)],
      ["episode-b", direction(44)]
    ]);
    const embedded: EmbeddedTrajectoryWindowV1[] = windows.map((occurrence) => ({
      occurrence,
      coarseVector: directions.get(occurrence.episodeId)!,
      stepVectors: occurrence.steps.map(() => [1, 0])
    }));
    const withoutOverride = clusterTrajectoryWindows(embedded, {
      ...clusteringOptions([10], 0.76),
      coarseMembershipMode: "multi"
    });
    const withOverride = clusterTrajectoryWindows(embedded, {
      ...clusteringOptions([10], 0.76),
      coarseSimilarityThresholdByScale: { 10: 0.70 },
      coarseMembershipMode: "multi"
    });

    expect(withoutOverride.families).toHaveLength(2);
    expect(withOverride.families).toHaveLength(1);
    expect(withOverride.clusters.filter((cluster) =>
      cluster.supportEpisodeIds.length >= 2)).toHaveLength(1);
  });

  it("clusters only within one scale and suppresses a shorter cluster fully covered by a longer one", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 10),
      episode("episode-b", 10)
    ], [
      { length: 5, stride: 2 },
      { length: 10, stride: 5 }
    ]);
    const clusters = clusterTrajectoryWindows(
      identicalEmbeddedWindows(windows),
      clusteringOptions([5, 10], 0.9)
    ).clusters;
    const supported = clusters.filter((cluster) => cluster.supportEpisodeIds.length >= 2);

    expect(supported.map((cluster) => cluster.scale)).toEqual([5, 10]);
    expect(supported.every((cluster) => cluster.supportEpisodeIds.length === 2)).toBe(true);
    expect(supported.find((cluster) => cluster.scale === 5)?.occurrenceCount).toBe(4);
    expect(supported.find((cluster) => cluster.scale === 10)?.occurrenceCount).toBe(2);

    const maximal = selectMaximalWindowClusters(supported);
    const shorter = maximal.find((cluster) => cluster.scale === 5)!;
    const longer = maximal.find((cluster) => cluster.scale === 10)!;
    expect(shorter.suppressedByClusterId).toBe(longer.id);
    expect(longer.suppressedByClusterId).toBeUndefined();
  });

  it("does not suppress a shorter repeated window when the longer cluster lacks its evidence", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 10),
      episode("episode-b", 10),
      episode("episode-c", 5)
    ], [
      { length: 5, stride: 5 },
      { length: 10, stride: 10 }
    ]);
    const clusters = clusterTrajectoryWindows(
      identicalEmbeddedWindows(windows),
      clusteringOptions([5, 10], 0.9)
    ).clusters;
    const supported = clusters.filter((cluster) => cluster.supportEpisodeIds.length >= 2);
    const maximal = selectMaximalWindowClusters(supported);
    const shorter = maximal.find((cluster) => cluster.scale === 5)!;

    expect(shorter.supportEpisodeIds).toEqual(["episode-a", "episode-b", "episode-c"]);
    expect(shorter.suppressedByClusterId).toBeUndefined();
  });

  it("keeps reversed procedures in one coarse Family but separates them during fine matching", async () => {
    const experiment = new MultiScaleWindowPolicyExperiment({
      config: DEFAULT_MEMMY_CONFIG,
      embedder: labeledEmbedder(),
      llm: unusedLlm()
    });
    const forward = ["A", "B", "C", "D", "E"];
    const result = await experiment.run({
      episodes: [
        episodeWithIntents("episode-a", forward),
        episodeWithIntents("episode-b", forward),
        episodeWithIntents("episode-c", [...forward].reverse())
      ],
      specs: [{ length: 5, stride: 5 }],
      coarseSimilarityThreshold: 0.9,
      minSupportEpisodes: 2,
      inducePolicies: false
    });
    const supported = result.clusters.filter((cluster) => cluster.supportEpisodeIds.length >= 2);

    expect(result.embeddedStepCount).toBe(15);
    expect(result.families).toHaveLength(1);
    expect(supported).toHaveLength(1);
    expect(supported[0]!.supportEpisodeIds).toEqual(["episode-a", "episode-b"]);
  });

  it("recovers a one-Step boundary shift while preserving monotonic order", () => {
    const vectors = ["A", "B", "C", "D", "E", "X"].map((_, index, all) =>
      all.map((__, candidate) => candidate === index ? 1 : 0));
    const byLabel = new Map(["A", "B", "C", "D", "E", "X"].map((label, index) =>
      [label, vectors[index]!]));
    const config = fineConfig(5);
    const shifted = bandedMonotonicMatch(
      ["A", "B", "C", "D", "E"].map((label) => byLabel.get(label)!),
      ["X", "A", "B", "C", "D"].map((label) => byLabel.get(label)!),
      config
    );
    const reversed = bandedMonotonicMatch(
      ["A", "B", "C", "D", "E"].map((label) => byLabel.get(label)!),
      ["E", "D", "C", "B", "A"].map((label) => byLabel.get(label)!),
      config
    );

    expect(shifted).toMatchObject({
      admitted: true,
      matchedSteps: 4,
      coverage: 0.8,
      maxInternalGap: 0
    });
    expect(shifted.pairs.map((pair) => [pair.leftIndex, pair.rightIndex]))
      .toEqual([[0, 1], [1, 2], [2, 3], [3, 4]]);
    expect(reversed.admitted).toBe(false);
  });

  it("updates the medoid to a more central real window without creating a centroid sequence", () => {
    const windows = buildTrajectoryWindows([
      episode("episode-a", 5),
      episode("episode-b", 5),
      episode("episode-c", 5),
      episode("episode-d", 5)
    ], [{ length: 5, stride: 5 }]);
    const directions = new Map([
      ["episode-a", direction(0)],
      ["episode-b", direction(70)],
      ["episode-c", direction(40)],
      ["episode-d", direction(35)]
    ]);
    const embedded: EmbeddedTrajectoryWindowV1[] = windows.map((occurrence) => {
      const vector = directions.get(occurrence.episodeId)!;
      return {
        occurrence,
        coarseVector: vector,
        stepVectors: occurrence.steps.map(() => vector)
      };
    });
    const permissive: BandedMonotonicMatchConfig = {
      scale: 5,
      bandWidth: 0,
      minStepSimilarity: 0.3,
      minMatchedSteps: 5,
      minCoverage: 1,
      minAverageMatchSimilarity: 0.3,
      maxInternalGap: 0,
      gapPenalty: 0,
      minAlignmentScore: 0.3
    };
    const result = clusterTrajectoryWindows(embedded, {
      coarseSimilarityThreshold: 0.3,
      fineMatchConfigs: [permissive],
      medoidSwitchMargin: 0
    });

    expect(result.clusters).toHaveLength(1);
    expect(result.clusters[0]!.medoidOccurrenceId)
      .toBe(windows.find((window) => window.episodeId === "episode-d")!.id);
    expect(result.clusters[0]!.medoidUpdateCount).toBeGreaterThan(0);
    expect(result.clusters[0]!.members.every((member) =>
      member.alignmentToMedoid.admitted)).toBe(true);
  });

  it("uses short evidence aliases in the prompt and persists the real source ids", async () => {
    const seenPayloads: Array<Record<string, unknown>> = [];
    const experiment = new MultiScaleWindowPolicyExperiment({
      config: DEFAULT_MEMMY_CONFIG,
      embedder: labeledEmbedder(),
      llm: policyLlm(seenPayloads)
    });
    const intents = ["A", "B", "C", "D", "E"];
    const result = await experiment.run({
      episodes: [
        episodeWithIntents("episode-a", intents),
        episodeWithIntents("episode-b", intents)
      ],
      specs: [{ length: 5, stride: 5 }],
      coarseSimilarityThreshold: 0.9,
      minSupportEpisodes: 2
    });

    expect(result.policies).toHaveLength(1);
    expect(JSON.stringify(seenPayloads[0])).toContain('"occurrence_id":"O1"');
    expect(JSON.stringify(seenPayloads[0])).toContain('"step_id":"O1.S1"');
    expect(result.policies[0]!.evidenceOccurrenceIds.every((id) =>
      id.startsWith("trajectory_window_"))).toBe(true);
    expect(result.policies[0]!.procedureSteps[0]!.evidenceRefs)
      .toEqual(["step-episode-a-0", "step-episode-b-0"]);
  });

  it("admits only an evidence-grounded sequence of at least two distinct Policies as a Skill", async () => {
    const seenPayloads: Array<Record<string, unknown>> = [];
    const experiment = new MultiScaleWindowPolicyExperiment({
      config: DEFAULT_MEMMY_CONFIG,
      embedder: labeledEmbedder(),
      llm: skillLlm(seenPayloads)
    });
    const policies = [testPolicy("policy-a", "Inspect and repair"), testPolicy(
      "policy-b",
      "Verify the repaired artifact"
    )];
    const candidate: MultiScaleWindowSkillCandidateV1 = {
      id: "skill-candidate-a",
      policyIds: policies.map((policy) => policy.id),
      policies,
      supportEpisodeIds: ["episode-a", "episode-b"],
      evidence: ["episode-a", "episode-b"].map((episodeId) => ({
        episodeId,
        spans: policies.map((policy, index) => ({
          policyId: policy.id,
          clusterId: `cluster-${index}`,
          spanId: `span-${episodeId}-${index}`,
          startStepIndex: index * 5,
          endStepIndex: index * 5 + 4
        }))
      }))
    };

    const decisions = await experiment.induceSkills([candidate], 1);

    expect(decisions).toHaveLength(1);
    expect(decisions[0]?.admitted).toBe(true);
    expect(decisions[0]?.skill?.sourcePolicyIds).toEqual(["policy-a", "policy-b"]);
    expect(decisions[0]?.skill?.evidenceEpisodeIds).toEqual(["episode-a", "episode-b"]);
    expect(JSON.stringify(seenPayloads[0])).toContain('"ordered_spans"');
  });
});

function clusteringOptions(scales: readonly number[], coarseSimilarityThreshold: number) {
  return {
    coarseSimilarityThreshold,
    fineMatchConfigs: scales.map((scale) => fineConfig(scale)),
    medoidSwitchMargin: 0.01
  };
}

function fineConfig(scale: number): BandedMonotonicMatchConfig {
  const config = DEFAULT_MULTI_SCALE_WINDOW_FINE_MATCH_CONFIGS.find((item) =>
    item.scale === scale);
  if (!config) throw new Error(`missing test fine config for scale ${scale}`);
  return { ...config };
}

function identicalEmbeddedWindows(
  windows: ReturnType<typeof buildTrajectoryWindows>
): EmbeddedTrajectoryWindowV1[] {
  return windows.map((occurrence) => ({
    occurrence,
    coarseVector: [1, 0],
    stepVectors: occurrence.steps.map(() => [1, 0])
  }));
}

function direction(degrees: number): number[] {
  const radians = degrees * Math.PI / 180;
  return [Math.cos(radians), Math.sin(radians)];
}

function episode(episodeId: string, count: number): MultiScaleWindowEpisodeInput {
  return {
    episodeId,
    pathId: `path-${episodeId}`,
    terminalReward: 1,
    steps: Array.from({ length: count }, (_, index) => step(episodeId, index))
  };
}

function episodeWithIntents(
  episodeId: string,
  intents: readonly string[]
): MultiScaleWindowEpisodeInput {
  const value = episode(episodeId, intents.length);
  return {
    ...value,
    steps: value.steps.map((item, index) => ({
      ...item,
      action: {
        ...item.action,
        intent: intents[index]!,
        summary: `${intents[index]!} completed`
      }
    }))
  };
}

function episodeWithDomain(
  episodeId: string,
  intents: readonly string[],
  domain: string
): MultiScaleWindowEpisodeInput {
  const value = episodeWithIntents(episodeId, intents);
  return {
    ...value,
    steps: value.steps.map((item) => ({
      ...item,
      action: {
        ...item.action,
        summary: `${item.action.intent} completed for ${domain}`
      }
    }))
  };
}

function step(episodeId: string, stepIndex: number): ExecutionStepV1 {
  return {
    id: `step-${episodeId}-${stepIndex}`,
    schemaVersion: "execution-step.v1",
    episodeId,
    rawTurnId: `turn-${episodeId}`,
    turnIndex: 0,
    stepIndex,
    preStateId: `state-${episodeId}`,
    action: {
      kind: "tool_action",
      type: "execute",
      intent: `perform operation ${stepIndex}`,
      summary: `operation ${stepIndex} completed`,
      eventRefs: [`event-${episodeId}-${stepIndex}`],
      toolName: "exec_command"
    },
    actionEffectDelta: [],
    actionPostStateId: `state-${episodeId}`,
    externalObservationDelta: [],
    postStateId: `state-${episodeId}`,
    outcome: {
      status: "success",
      evidenceRefs: [`event-${episodeId}-${stepIndex}`]
    },
    cost: { toolCalls: 1, errorCount: 0 },
    provenance: {
      algorithmVersion: "multi-scale-window-policy-test.v1",
      sourceSnapshotHash: `snapshot-${episodeId}`
    }
  };
}

function labeledEmbedder(): Embedder {
  const labels = ["A", "B", "C", "D", "E"];
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "multi-scale-window-test"
    },
    isRemote() { return false; },
    async embed(texts: string[]) {
      return texts.map((text) => labels.map((label) =>
        label === text || text.split("\n").some((line) => line.endsWith(`. ${label}`)) ? 1 : 0));
    },
    async embedOne(text: string) {
      return (await this.embed([text]))[0]!;
    },
    status() {
      return {
        provider: "local",
        model: "multi-scale-window-test",
        configured: true,
        remote: false
      };
    }
  };
}

function recordingSequenceEmbedder(seenTexts: string[]): Embedder {
  const intentLabels = ["inspect", "edit", "execute", "verify", "finish"];
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "multi-scale-window-sequence-test"
    },
    isRemote() { return false; },
    async embed(texts: string[]) {
      seenTexts.push(...texts);
      return texts.map((text) => {
        if (text.includes("\n")) return [1, 1, 1, 1, 1];
        return intentLabels.map((label) => label === text ? 1 : 0);
      });
    },
    async embedOne(text: string) {
      return (await this.embed([text]))[0]!;
    },
    status() {
      return {
        provider: "local",
        model: "multi-scale-window-sequence-test",
        configured: true,
        remote: false
      };
    }
  };
}

function unusedLlm(): LlmClient {
  return {
    config: DEFAULT_MEMMY_CONFIG.evolution,
    isConfigured() { return false; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>() { return {} as T; },
    status() { return { provider: "", configured: false, remote: false }; }
  };
}

function policyLlm(seenPayloads: Array<Record<string, unknown>>): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      model: "multi-scale-window-test"
    },
    isConfigured() { return true; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>(messages: LlmMessage[]): Promise<T> {
      const payload = JSON.parse(messages.find((message) => message.role === "user")!.content) as
        Record<string, unknown>;
      seenPayloads.push(payload);
      return {
        admit: true,
        rejection_reason: null,
        title: "Execute the aligned workflow",
        goal_pattern: "Complete the repeated aligned workflow",
        trigger_conditions: ["The aligned workflow is requested"],
        procedure_steps: [{
          instruction: "Perform the first aligned operation.",
          evidence_refs: ["O1.S1", "O2.S1"]
        }],
        verification_steps: [{
          check: "Check the final aligned operation.",
          success_signal: "The final operation succeeds.",
          evidence_refs: ["O1.S5", "O2.S5"]
        }],
        do_not_apply_when: [],
        evidence_occurrence_ids: ["O1", "O2"],
        confidence: 0.95
      } as unknown as T;
    },
    status() {
      return {
        provider: "host",
        model: "multi-scale-window-test",
        configured: true,
        remote: false
      };
    }
  };
}

function testPolicy(id: string, title: string): MultiScaleWindowPolicyV1 {
  return {
    id,
    clusterId: `cluster-${id}`,
    scale: 5,
    title,
    goalPattern: title,
    triggerConditions: ["The repeated procedure is required"],
    procedureSteps: [{ instruction: title, evidenceRefs: [`evidence-${id}`] }],
    verificationSteps: [{
      check: `Check ${title}`,
      successSignal: `${title} succeeds`,
      evidenceRefs: [`evidence-${id}`]
    }],
    doNotApplyWhen: [],
    evidenceOccurrenceIds: [`span-${id}-a`, `span-${id}-b`],
    supportEpisodeIds: ["episode-a", "episode-b"],
    confidence: 0.9,
    promptVersion: "multi-scale-window-policy-prompt.v2"
  };
}

function skillLlm(seenPayloads: Array<Record<string, unknown>>): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      model: "multi-scale-window-skill-test"
    },
    isConfigured() { return true; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>(messages: LlmMessage[]): Promise<T> {
      const payload = JSON.parse(messages.find((message) => message.role === "user")!.content) as
        Record<string, unknown>;
      seenPayloads.push(payload);
      return {
        admit: true,
        rejection_reason: null,
        name: "Repair and verify an artifact",
        purpose: "Repair a faulty artifact and verify the corrected result.",
        trigger_conditions: ["An artifact requires repair and verification"],
        procedure_steps: [
          {
            instruction: "Inspect and repair the artifact.",
            source_policy_ids: ["policy-a"]
          },
          {
            instruction: "Verify the repaired artifact.",
            source_policy_ids: ["policy-b"]
          }
        ],
        verification_steps: [{
          check: "Check the repaired artifact",
          success_signal: "The repaired artifact passes verification",
          source_policy_ids: ["policy-b"]
        }],
        do_not_apply_when: [],
        source_policy_ids: ["policy-a", "policy-b"],
        evidence_episode_ids: ["episode-a", "episode-b"],
        confidence: 0.9
      } as unknown as T;
    },
    status() {
      return {
        provider: "host",
        model: "multi-scale-window-skill-test",
        configured: true,
        remote: false
      };
    }
  };
}
