import { describe, expect, it } from "vitest";
import {
  evolutionJobDedupeKey,
  workerJobCanRunInParallel
} from "../../../src/service/worker/job-handlers.js";

describe("procedural trajectory Job wiring", () => {
  it("versions compile dedupe independently by source and reward snapshots", () => {
    const base = {
      jobType: "episode_path_compile" as const,
      episodeId: "episode-1",
      payload: {
        sourceSnapshotHash: "source-a",
        rewardSnapshotHash: "reward-positive",
        semanticsVersion: "semantics-v1"
      }
    };
    const positive = evolutionJobDedupeKey(base);
    const negative = evolutionJobDedupeKey({
      ...base,
      payload: { ...base.payload, rewardSnapshotHash: "reward-negative" }
    });
    const changedSource = evolutionJobDedupeKey({
      ...base,
      payload: { ...base.payload, sourceSnapshotHash: "source-b" }
    });

    expect(positive).not.toBe(negative);
    expect(positive).not.toBe(changedSource);
    expect(positive).toContain("source-a:reward-positive:semantics-v1");
  });

  it("versions window ingest dedupe by reward without changing path identity", () => {
    const positive = evolutionJobDedupeKey({
      jobType: "trajectory_window_ingest",
      episodeId: "episode-1",
      payload: {
        pathId: "path-stable",
        pathHash: "path-hash-stable",
        rewardSnapshotHash: "reward-positive",
        mechanicalWindowHash: "window-v1",
        clusteringConfigHash: "cluster-v1"
      }
    });
    const negative = evolutionJobDedupeKey({
      jobType: "trajectory_window_ingest",
      episodeId: "episode-1",
      payload: {
        pathId: "path-stable",
        pathHash: "path-hash-stable",
        rewardSnapshotHash: "reward-negative",
        mechanicalWindowHash: "window-v1",
        clusteringConfigHash: "cluster-v1"
      }
    });

    expect(positive).not.toBe(negative);
    expect(positive).toContain(
      "path-stable:path-hash-stable:reward-positive:window-v1:cluster-v1"
    );
  });

  it("reruns clustering config changes while preserving mechanical Window identity", () => {
    const base = {
      jobType: "trajectory_window_ingest" as const,
      episodeId: "episode-1",
      payload: {
        pathId: "path-stable",
        pathHash: "path-hash-stable",
        rewardSnapshotHash: "reward-positive",
        mechanicalWindowHash: "window-v1",
        clusteringConfigHash: "cluster-v1"
      }
    };
    const original = evolutionJobDedupeKey(base);
    const thresholdChange = evolutionJobDedupeKey({
      ...base,
      payload: { ...base.payload, clusteringConfigHash: "cluster-v2" }
    });

    expect(original).not.toBe(thresholdChange);
    expect(original).toContain("window-v1:cluster-v1");
    expect(thresholdChange).toContain("window-v1:cluster-v2");
  });

  it("serializes V2 Skill compilations so same-batch reuse sees prior materialization", () => {
    expect(workerJobCanRunInParallel({
      jobType: "procedural_skill_induction"
    } as Parameters<typeof workerJobCanRunInParallel>[0])).toBe(false);
    expect(workerJobCanRunInParallel({
      jobType: "long_trajectory_skill_induction"
    } as Parameters<typeof workerJobCanRunInParallel>[0])).toBe(true);
  });
});
