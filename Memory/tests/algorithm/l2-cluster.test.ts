import { describe, expect, it } from "vitest";
import { cosineSimilarity } from "../../src/algorithm/trace-direct-skill.js";
import {
  admitL2PolicyDraft,
  decideL2ClusterJoin,
  decideL2EvolveBranch,
  inferL2LessonKind,
  mergeL2LessonKind,
  packL2ClusterRawTurns,
  renderL2ClusterEvidence,
  resolveL2LessonKind
} from "../../src/algorithm/l2-cluster.js";

function unit(x: number, y: number): number[] {
  const norm = Math.hypot(x, y);
  return [x / norm, y / norm];
}

function grayIntent(cosine: number): number[] {
  return unit(cosine, Math.sqrt(1 - cosine * cosine));
}

const seed = {
  id: "c-n1",
  intentCentroid: [1, 0],
  taskCentroid: [0, 1]
};

describe("decideL2ClusterJoin", () => {
  it("skips empty intent so continuation never enters a cluster", () => {
    expect(decideL2ClusterJoin({
      intent: "",
      intentVec: [1, 0],
      taskVec: [0, 1]
    }, [seed])).toEqual({ action: "skip", reason: "skip_no_intent" });
  });

  it("joins when intent cosine is at least 0.60 even if the task differs", () => {
    const intentVec = grayIntent(0.8);
    expect(cosineSimilarity(intentVec, seed.intentCentroid)).toBeCloseTo(0.8, 5);
    expect(decideL2ClusterJoin({
      intent: "去掉列表接口的 N+1",
      intentVec,
      taskVec: [1, 0]
    }, [seed])).toMatchObject({
      action: "join",
      clusterId: "c-n1",
      reason: "join_intent"
    });
  });

  it("joins the gray zone only when task cosine is at least 0.70", () => {
    const intentVec = grayIntent(0.55);
    expect(cosineSimilarity(intentVec, seed.intentCentroid)).toBeCloseTo(0.55, 5);
    expect(decideL2ClusterJoin({
      intent: "减少重复查询",
      intentVec,
      taskVec: [0, 1]
    }, [seed])).toMatchObject({
      action: "join",
      clusterId: "c-n1",
      reason: "join_gray_task"
    });
    expect(decideL2ClusterJoin({
      intent: "减少重复查询",
      intentVec,
      taskVec: [1, 0]
    }, [seed])).toMatchObject({
      action: "create",
      reason: "create_gray_task_below"
    });
  });

  it("does not join when only the task is near", () => {
    expect(decideL2ClusterJoin({
      intent: "给接口加配置缓存",
      intentVec: [0, 1],
      taskVec: [0, 1]
    }, [seed])).toMatchObject({
      action: "create",
      reason: "create_intent_below_gray",
      best: { clusterId: "c-n1", intent: 0, task: 1 }
    });
  });

  it("creates a cluster when there is no candidate", () => {
    expect(decideL2ClusterJoin({
      intent: "消除查询中的 N+1",
      intentVec: [1, 0],
      taskVec: [0, 1]
    }, [])).toEqual({ action: "create", reason: "create_no_candidates" });
  });
});

describe("L2 lesson kind and evolve branch", () => {
  it("does not infer a lesson without a positive member", () => {
    expect(inferL2LessonKind(0, 2)).toBeNull();
    expect(inferL2LessonKind(1, 0)).toBe("path_compression");
    expect(inferL2LessonKind(1, 1)).toBe("both");
  });

  it("merges compression and correction into both", () => {
    expect(mergeL2LessonKind("path_compression", "error_correction")).toBe("both");
    expect(resolveL2LessonKind("path_compression", "both")).toBe("both");
    expect(resolveL2LessonKind("both", "path_compression")).toBe("both");
  });

  it("requires a positive member before create or evolve", () => {
    expect(decideL2EvolveBranch({
      hasPolicy: false,
      positiveCount: 0,
      newPositiveCount: 0,
      newNegativeCount: 2
    })).toEqual({ action: "skip_no_positive" });
    expect(decideL2EvolveBranch({
      hasPolicy: false,
      positiveCount: 1,
      newPositiveCount: 1,
      newNegativeCount: 1
    })).toEqual({ action: "create" });
    expect(decideL2EvolveBranch({
      hasPolicy: true,
      positiveCount: 1,
      newPositiveCount: 0,
      newNegativeCount: 0
    })).toEqual({ action: "link_only" });
    expect(decideL2EvolveBranch({
      hasPolicy: true,
      positiveCount: 1,
      newPositiveCount: 0,
      newNegativeCount: 1
    })).toEqual({ action: "evolve", incrementKind: "error_correction" });
  });

  it("packs raw turns with polarity instead of L1 summaries", () => {
    const packed = packL2ClusterRawTurns({
      members: [{
        l1MemoryId: "l1-ok",
        value: 1,
        rawTurn: {
          userText: "run the focused pytest",
          assistantText: "ran pytest tests/a.py",
          toolCalls: [{ name: "pytest", input: "tests/a.py", output: "passed" }]
        }
      }, {
        l1MemoryId: "l1-fail",
        value: -1,
        rawTurn: {
          userText: "same pytest path failed",
          assistantText: "retried the whole suite",
          toolCalls: [{ name: "pytest", input: "tests", output: "failed", success: false }]
        }
      }],
      minPositiveValue: 0.005,
      clipChars: 300
    });
    expect(packed.map((item) => item.polarity)).toEqual(["success", "failure"]);
    const rendered = renderL2ClusterEvidence({
      clusterId: "l2c_1",
      seedIntent: "run focused pytest",
      lessonKind: "both",
      mode: "create",
      turns: packed,
      charCap: 2000
    });
    expect(rendered).toContain("PATTERN_SIGNATURE");
    expect(rendered).toContain("RAW_TURNS");
    expect(rendered).toContain("run the focused pytest");
    expect(rendered).toContain("polarity: failure");
    expect(rendered).toContain("not compressed L1");
    const withFailure = renderL2ClusterEvidence({
      clusterId: "l2c_1",
      seedIntent: "run focused pytest",
      lessonKind: "both",
      mode: "create",
      existingFailurePolicies: [{
        title: "Avoid full-suite retry",
        trigger: "focused pytest failed",
        procedure: "Do not rerun the whole suite after a focused pytest failure.",
        boundary: "pytest only"
      }],
      turns: packed,
      charCap: 2000
    });
    expect(withFailure).toContain("EXISTING_FAILURE_POLICY");
    expect(withFailure).toContain("Avoid full-suite retry");
  });

  it("admits only drafts that can improve future efficiency", () => {
    expect(admitL2PolicyDraft({
      title: "Use focused pytest",
      trigger: "pytest migration failed",
      procedure: "Run the focused pytest, inspect the output, then retry the same test.",
      caveats: ["Do not retry the whole suite first."],
      lessonKind: "both",
      positiveCount: 1,
      negativeCount: 1
    }).ok).toBe(true);
    expect(admitL2PolicyDraft({
      title: "User likes pytest",
      trigger: "conversation mentions pytest",
      procedure: "The user prefers pytest for this project.",
      lessonKind: "path_compression",
      positiveCount: 1,
      negativeCount: 0
    })).toEqual({ ok: false, reason: "admission-declined:no-efficiency-gain" });
    expect(admitL2PolicyDraft({
      title: "Use focused pytest",
      trigger: "pytest migration failed",
      procedure: "Run the focused pytest path that already passed.",
      lessonKind: "path_compression",
      positiveCount: 1,
      negativeCount: 1
    })).toEqual({ ok: false, reason: "admission-declined:missing-error-correction" });
    expect(admitL2PolicyDraft({
      title: "Use focused pytest",
      trigger: "pytest migration failed",
      procedure: "Run the focused pytest path that already passed.",
      lessonKind: "path_compression",
      positiveCount: 0,
      negativeCount: 1
    })).toEqual({ ok: false, reason: "admission-declined:no-positive-anchor" });
  });
});
