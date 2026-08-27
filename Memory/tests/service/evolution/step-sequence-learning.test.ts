import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  STEP_POLICY_SKILL_OPERATION,
  STEP_SEQUENCE_POLICY_OPERATION,
  STEP_SEQUENCE_POLICY_REPAIR_OPERATION,
  buildEpisodeProceduralPath,
  emptyObservedState,
  selectLongestNonOverlapping,
  sequenceOccurrencesFullyCovered,
  type Embedder,
  type ExecutionStepV1,
  type LlmClient,
  type LlmMessage,
  type ProceduralSpanV1,
  type SpanSegmentationDecisionV1
} from "../../../src/index.js";
import {
  policyMetaFromMemory,
  skillMetaFromMemory
} from "../../../src/algorithm/plugin-algorithms.js";
import { Repositories } from "../../../src/storage/repositories.js";
import type { MemoryRow } from "../../../src/types.js";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const { cleanup, createTestService } = createMemoryServiceFixture();

afterEach(cleanup);

describe("StepCluster sequence learning", () => {
  it("selects longest non-overlapping occurrences and requires positional full coverage", () => {
    const selected = selectLongestNonOverlapping([
      { id: "ab", startIndex: 0, endIndex: 1, sequenceLength: 2, support: 4 },
      { id: "abc", startIndex: 0, endIndex: 2, sequenceLength: 3, support: 2 },
      { id: "ef", startIndex: 4, endIndex: 5, sequenceLength: 2, support: 3 }
    ]);
    expect(selected.map((item) => item.id)).toEqual(["abc", "ef"]);

    expect(sequenceOccurrencesFullyCovered({
      shorterSequence: ["A", "B"],
      longerSequence: ["A", "B", "C"],
      shorterOccurrences: [
        { episodeId: "e1", pathId: "p1", startIndex: 0, endIndex: 1 },
        { episodeId: "e2", pathId: "p2", startIndex: 0, endIndex: 1 }
      ],
      longerOccurrences: [
        { episodeId: "e1", pathId: "p1", startIndex: 0, endIndex: 2 },
        { episodeId: "e2", pathId: "p2", startIndex: 0, endIndex: 2 }
      ]
    })).toBe(true);
    expect(sequenceOccurrencesFullyCovered({
      shorterSequence: ["A", "B"],
      longerSequence: ["A", "B", "C"],
      shorterOccurrences: [
        { episodeId: "e1", pathId: "p1", startIndex: 0, endIndex: 1 },
        { episodeId: "e1", pathId: "p1", startIndex: 8, endIndex: 9 }
      ],
      longerOccurrences: [
        { episodeId: "e1", pathId: "p1", startIndex: 0, endIndex: 2 }
      ]
    })).toBe(false);
  });

  it("learns two repeated local Step sequences as Policies and their gapped Policy sequence as a Skill", async () => {
    const operations: string[] = [];
    const embeddedTexts: string[] = [];
    const { db, service } = createTestService({
      skillLlm: stepSequenceLlm(operations),
      embedder: stepSequenceEmbedder(embeddedTexts),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const first = persistStepSequenceEpisode(repos, "a", ["A", "B", "C", "X", "D", "E", "F"]);
    const second = persistStepSequenceEpisode(repos, "b", ["A", "B", "C", "Y", "D", "E", "F"]);

    const firstResult = await service.learnStepSequencesForReplay({
      episodeId: first.episodeId,
      at: "2026-08-25T01:00:00.000Z"
    });
    expect(firstResult.stepCount).toBe(7);
    expect(firstResult.repeatedPatternCount).toBe(0);
    expect(firstResult.inducedPolicyIds).toEqual([]);

    const secondResult = await service.learnStepSequencesForReplay({
      episodeId: second.episodeId,
      at: "2026-08-25T01:01:00.000Z"
    });

    const readyPatterns = repos.stepSequenceLearning.listStepPatterns("user-step-sequence")
      .filter((pattern) => pattern.lifecycleStatus === "ready" && pattern.isMaximal);
    expect(readyPatterns.map((pattern) => pattern.clusterIds.length)).toEqual([3, 3]);
    expect(secondResult.inducedPolicyIds).toHaveLength(2);
    expect(operations.filter((operation) => operation === STEP_SEQUENCE_POLICY_OPERATION))
      .toHaveLength(2);
    expect(operations.filter((operation) => operation === STEP_POLICY_SKILL_OPERATION))
      .toHaveLength(1);

    const projectionA = repos.stepSequenceLearning.getActiveProjectionForEpisode(first.episodeId)!;
    const projectionB = repos.stepSequenceLearning.getActiveProjectionForEpisode(second.episodeId)!;
    for (const projection of [projectionA, projectionB]) {
      expect(projection.projection.nodes.map((node) => node.kind))
        .toEqual(["policy", "unmapped", "policy"]);
      expect(projection.projection.mappedStepCount).toBe(6);
      expect(projection.projection.unmappedStepCount).toBe(1);
    }

    const readySkills = repos.stepSequenceLearning.listSkillPatterns("user-step-sequence")
      .filter((pattern) => pattern.lifecycleStatus === "ready" && pattern.isMaximal);
    expect(readySkills).toHaveLength(1);
    expect(new Set(readySkills[0]!.policyKeys).size).toBe(2);
    expect(readySkills[0]!.activeSkillMemoryId).toBeTruthy();
    const skill = repos.memories.get(readySkills[0]!.activeSkillMemoryId!);
    expect(skill).toMatchObject({
      memoryLayer: "Skill",
      status: "resolving",
      properties: {
        internal_info: {
          step_policy_sequence_skill: {
            executable: true
          }
        }
      }
    });
    const policyMemories = secondResult.inducedPolicyIds.map((id) =>
      repos.stepSequenceLearning.getPolicy(id)?.l2MemoryId)
      .filter((id): id is string => Boolean(id))
      .map((id) => repos.memories.get(id)!);
    expect(policyMemories).toHaveLength(2);
    for (const policyMemory of policyMemories) {
      const policy = policyMetaFromMemory(policyMemory)!;
      expect(policy.title).not.toContain("policy:step-sequence:");
      expect(policy.signature).toMatch(/^step_sequence\|[a-f0-9]{24}\|_\|_$/);
      expect(policy.sourceTraceIds).toHaveLength(2);
      expect(policy.sourceTraceIds.every((id) => repos.memories.get(id)?.memoryLayer === "L1"))
        .toBe(true);
      expect(repos.runtime.listTracePolicyLinks({ l2MemoryId: policy.id }))
        .toHaveLength(2);
    }
    const skillMeta = skillMetaFromMemory(skill!)!;
    expect(skillMeta.retrievalBlurb)
      .toBe("Execute two reusable local procedures with evidence-based verification.");
    expect(skillMeta.triggerContext)
      .toBe("Use when the two-stage workflow must be completed end to end.");
    expect(skillMeta.evidenceAnchorIds).toHaveLength(2);
    expect(skillMeta.evidenceAnchorIds.every((id) => repos.memories.get(id)?.memoryLayer === "L1"))
      .toBe(true);
    const internal = skill!.properties.internal_info;
    const nestedSkill = internal.skill as Record<string, unknown>;
    expect(internal.procedure_json).toEqual(nestedSkill.procedure_json);
    expect(internal.source_step_policy_skill_occurrence_ids)
      .toEqual(expect.arrayContaining([
        expect.stringMatching(/^step_policy_skill_occurrence_/),
        expect.stringMatching(/^step_policy_skill_occurrence_/)
      ]));
    const skillDetail = service.getSkill(skill!.id);
    expect(skillDetail).toMatchObject({
      name: "run_two_stage_workflow",
      evidenceAnchorIds: skillMeta.evidenceAnchorIds,
      reliability: {
        supportCount: 2,
        trialsAttempted: 0,
        trialsPassed: 0
      }
    });
    expect(skillDetail.sourcePolicyIds)
      .toEqual(expect.arrayContaining(policyMemories.map((memory) => memory.id)));
    expect(skillDetail.invocationGuide).toContain("Run and verify the two-stage workflow");
    expect(embeddedTexts).toHaveLength(14);
    expect(embeddedTexts.every((text) => text.startsWith("Intent: "))).toBe(true);
    expect(embeddedTexts.every((text) => !text.includes("\nResult: "))).toBe(true);
  });

  it("keeps a shorter Policy when the longer sequence does not cover its Episode evidence", async () => {
    const operations: string[] = [];
    const { db, service } = createTestService({
      skillLlm: stepSequenceLlm(operations),
      embedder: stepSequenceEmbedder([]),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const episodes = [
      persistStepSequenceEpisode(repos, "a", ["A", "B", "X", "D", "E"]),
      persistStepSequenceEpisode(repos, "b", ["A", "B", "Y", "D", "E"]),
      persistStepSequenceEpisode(repos, "c", ["A", "B", "C", "Z", "D", "E"]),
      persistStepSequenceEpisode(repos, "d", ["A", "B", "C", "W", "D", "E"])
    ];

    for (const [index, episode] of episodes.entries()) {
      await service.learnStepSequencesForReplay({
        episodeId: episode.episodeId,
        at: `2026-08-25T02:0${index}:00.000Z`
      });
    }

    const activePolicies = repos.stepSequenceLearning.listStepPatterns("user-step-sequence")
      .filter((pattern) => pattern.activePolicyVersionId);
    expect(activePolicies.map((pattern) => pattern.clusterIds.length).sort())
      .toEqual([2, 2, 3]);
    expect(activePolicies.every((pattern) => !pattern.supersededByPatternId)).toBe(true);

    for (const episode of episodes) {
      const projection = repos.stepSequenceLearning
        .getActiveProjectionForEpisode(episode.episodeId)!;
      expect(projection.projection.nodes.map((node) => node.kind))
        .toEqual(["policy", "unmapped", "policy"]);
    }

    const readySkills = repos.stepSequenceLearning.listSkillPatterns("user-step-sequence")
      .filter((pattern) => pattern.lifecycleStatus === "ready" && pattern.isMaximal);
    expect(readySkills).toHaveLength(2);
    expect(readySkills.every((pattern) => pattern.activeSkillMemoryId)).toBe(true);
    expect(repos.memories.list({
      memoryLayer: "Skill",
      status: ["resolving", "activated"]
    })).toHaveLength(2);
    expect(operations.filter((operation) => operation === STEP_POLICY_SKILL_OPERATION))
      .toHaveLength(2);
  });

  it("retires a legacy shorter Policy only after a compiled longer Policy covers every occurrence", async () => {
    const { db, service } = createTestService({
      skillLlm: stepSequenceLlm([]),
      embedder: stepSequenceEmbedder([])
    });
    const repos = new Repositories(db.db);
    const episodes = [
      persistStepSequenceEpisode(repos, "a", ["A", "B", "C", "X", "D", "E"]),
      persistStepSequenceEpisode(repos, "b", ["A", "B", "C", "Y", "D", "E"])
    ];
    for (const [index, episode] of episodes.entries()) {
      await service.learnStepSequencesForReplay({
        episodeId: episode.episodeId,
        at: `2026-08-25T03:0${index}:00.000Z`
      });
    }

    const patterns = repos.stepSequenceLearning.listStepPatterns("user-step-sequence");
    const longer = patterns.find((pattern) =>
      pattern.clusterIds.length === 3 && pattern.activePolicyVersionId)!;
    const shorter = patterns.find((pattern) =>
      pattern.clusterIds.length === 2 &&
      repos.stepSequenceLearning.listStepPatternOccurrences(pattern.id)
        .every((occurrence) => occurrence.startStepIndex === 0))!;
    const at = "2026-08-25T03:10:00.000Z";
    repos.memories.insert(legacyArtifactMemory("legacy-short-policy-memory", "L2", at));
    db.db.prepare(
      `INSERT INTO step_sequence_policy_versions (
        id, policy_key, namespace_id, pattern_id, pattern_membership_version,
        schema_version, induction_version, status, title, confidence,
        evidence_hash, l2_memory_id, compiler_model, payload_json,
        created_at, activated_at, deactivated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, NULL)`
    ).run(
      "legacy-short-policy-version",
      "policy:legacy-short",
      shorter.namespaceId,
      shorter.id,
      shorter.membershipVersion,
      "step-sequence-policy.v1",
      "step-sequence-policy-induction.v1",
      "Legacy short Policy",
      0.9,
      "legacy-evidence",
      "legacy-short-policy-memory",
      "fixture-model",
      "{}",
      at,
      at
    );
    db.db.prepare(
      `UPDATE step_sequence_patterns SET active_policy_version_id = ? WHERE id = ?`
    ).run("legacy-short-policy-version", shorter.id);

    const retirement = repos.stepSequenceLearning.retireStepPatternsCoveredBy(longer.id, at);
    expect(retirement.retiredPatternIds).toEqual([shorter.id]);
    expect(retirement.affectedEpisodeIds).toEqual(episodes.map((episode) => episode.episodeId));
    expect(repos.stepSequenceLearning.getStepPattern(shorter.id)).toMatchObject({
      supersededByPatternId: longer.id,
      isMaximal: false
    });
    expect(repos.stepSequenceLearning.getPolicy("legacy-short-policy-version")?.status)
      .toBe("inactive");
    expect(repos.memories.get("legacy-short-policy-memory")?.status).toBe("archived");
  });

  it("retires a fully covered shorter Skill after the longer Skill is compiled", async () => {
    const { db, service } = createTestService({
      skillLlm: stepSequenceLlm([]),
      embedder: stepSequenceEmbedder([])
    });
    const repos = new Repositories(db.db);
    const episodes = [
      persistStepSequenceEpisode(repos, "a", ["A", "B", "X", "C", "D", "Y", "E", "F"]),
      persistStepSequenceEpisode(repos, "b", ["A", "B", "Z", "C", "D", "W", "E", "F"])
    ];
    for (const [index, episode] of episodes.entries()) {
      await service.learnStepSequencesForReplay({
        episodeId: episode.episodeId,
        at: `2026-08-25T04:0${index}:00.000Z`
      });
    }

    const patterns = repos.stepSequenceLearning.listSkillPatterns("user-step-sequence");
    const longer = patterns.find((pattern) =>
      pattern.policyKeys.length === 3 && pattern.activeSkillMemoryId)!;
    const shorter = patterns.find((pattern) =>
      pattern.policyKeys.length === 2 &&
      longer.policyKeys.slice(0, 2).every((key, index) => pattern.policyKeys[index] === key))!;
    const at = "2026-08-25T04:10:00.000Z";
    repos.memories.insert(legacyArtifactMemory("legacy-short-skill-memory", "Skill", at));
    db.db.prepare(
      `UPDATE step_policy_skill_patterns SET active_skill_memory_id = ? WHERE id = ?`
    ).run("legacy-short-skill-memory", shorter.id);

    expect(repos.stepSequenceLearning.retireSkillPatternsCoveredBy(longer.id, at))
      .toEqual([shorter.id]);
    expect(repos.stepSequenceLearning.getSkillPattern(shorter.id)).toMatchObject({
      supersededByPatternId: longer.id,
      isMaximal: false
    });
    expect(repos.memories.get("legacy-short-skill-memory")?.status).toBe("archived");
  });

  it("keeps failed and uncertain Episodes as non-support evidence", async () => {
    const operations: string[] = [];
    const { db, service } = createTestService({
      skillLlm: stepSequenceLlm(operations),
      embedder: stepSequenceEmbedder([]),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const episodes = [
      persistStepSequenceEpisode(repos, "a", ["A", "B", "C", "X", "D", "E", "F"], 1),
      persistStepSequenceEpisode(repos, "b", ["A", "B", "C", "Y", "D", "E", "F"], -1),
      persistStepSequenceEpisode(repos, "c", ["A", "B", "C", "Z", "D", "E", "F"], 0),
      persistStepSequenceEpisode(repos, "d", ["A", "B", "C", "W", "D", "E", "F"], 0.8)
    ];

    for (const [index, episode] of episodes.slice(0, 3).entries()) {
      await service.learnStepSequencesForReplay({
        episodeId: episode.episodeId,
        at: `2026-08-25T05:0${index}:00.000Z`
      });
    }
    expect(operations.filter((operation) => operation === STEP_SEQUENCE_POLICY_OPERATION))
      .toHaveLength(0);
    expect(operations.filter((operation) => operation === STEP_POLICY_SKILL_OPERATION))
      .toHaveLength(0);

    await service.learnStepSequencesForReplay({
      episodeId: episodes[3]!.episodeId,
      at: "2026-08-25T05:03:00.000Z"
    });

    const readyPatterns = repos.stepSequenceLearning.listStepPatterns("user-step-sequence")
      .filter((pattern) => pattern.lifecycleStatus === "ready" && pattern.isMaximal);
    expect(readyPatterns.map((pattern) => pattern.selectedEpisodeCount)).toEqual([2, 2]);
    for (const pattern of readyPatterns) {
      const occurrences = repos.stepSequenceLearning.listStepPatternOccurrences(pattern.id);
      expect(new Set(occurrences.map((occurrence) => occurrence.evidenceRole)))
        .toEqual(new Set(["support", "counterexample", "unknown"]));
      expect(occurrences.filter((occurrence) => occurrence.selected)
        .every((occurrence) => occurrence.evidenceRole === "support")).toBe(true);
      const policy = repos.stepSequenceLearning.getPolicy(pattern.activePolicyVersionId!)!;
      expect(policy.policy.supportEpisodeIds.sort()).toEqual([
        episodes[0]!.episodeId,
        episodes[3]!.episodeId
      ].sort());
    }

    const readySkills = repos.stepSequenceLearning.listSkillPatterns("user-step-sequence")
      .filter((pattern) => pattern.lifecycleStatus === "ready" && pattern.isMaximal);
    expect(readySkills).toHaveLength(1);
    expect(readySkills[0]!.selectedEpisodeCount).toBe(2);
    const skillOccurrences = repos.stepSequenceLearning
      .listSkillPatternOccurrences(readySkills[0]!.id);
    expect(new Set(skillOccurrences.map((occurrence) => occurrence.evidenceRole)))
      .toEqual(new Set(["support", "counterexample", "unknown"]));
    expect(skillOccurrences.filter((occurrence) => occurrence.selected)
      .every((occurrence) => occurrence.evidenceRole === "support")).toBe(true);

    db.db.prepare(
      `UPDATE episodes SET r_task = ?, updated_at = ? WHERE id = ?`
    ).run(-1, "2026-08-25T05:04:00.000Z", episodes[3]!.episodeId);
    await service.learnStepSequencesForReplay({
      episodeId: episodes[3]!.episodeId,
      at: "2026-08-25T05:04:00.000Z"
    });
    expect(readyPatterns.every((pattern) => {
      const refreshed = repos.stepSequenceLearning.getStepPattern(pattern.id)!;
      return refreshed.lifecycleStatus === "observed" && !refreshed.activePolicyVersionId;
    })).toBe(true);
    const retiredSkillPattern = repos.stepSequenceLearning.getSkillPattern(readySkills[0]!.id)!;
    expect(retiredSkillPattern.lifecycleStatus).toBe("stale");
    expect(retiredSkillPattern.activeSkillMemoryId).toBeFalsy();
  });

  it("creates repaired Policy versions and incrementally rebuilds only affected Skill candidates", async () => {
    const operations: string[] = [];
    const { db, service } = createTestService({
      skillLlm: stepSequenceLlm(operations),
      embedder: stepSequenceEmbedder([]),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          },
          feedback: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.feedback,
            useLlm: false,
            cooldownMs: 0
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const episodes = [
      persistStepSequenceEpisode(repos, "a", ["A", "B", "C", "X", "D", "E", "F"], 1),
      persistStepSequenceEpisode(repos, "b", ["A", "B", "C", "Y", "D", "E", "F"], 1)
    ];
    for (const [index, episode] of episodes.entries()) {
      await service.learnStepSequencesForReplay({
        episodeId: episode.episodeId,
        at: `2026-08-25T06:0${index}:00.000Z`
      });
    }

    const policyPatterns = repos.stepSequenceLearning.listStepPatterns("user-step-sequence")
      .filter((pattern) => pattern.activePolicyVersionId);
    const basePolicies = policyPatterns.map((pattern) =>
      repos.stepSequenceLearning.getPolicy(pattern.activePolicyVersionId!)!);
    const basePolicyMemoryIds = basePolicies.map((policy) => policy.l2MemoryId!);
    const skillPattern = repos.stepSequenceLearning.listSkillPatterns("user-step-sequence")
      .find((pattern) => pattern.activeSkillMemoryId)!;
    const baseSkillMemoryId = skillPattern.activeSkillMemoryId!;
    const feedback = await service.feedback({
      sessionId: "session-step-sequence-a",
      episodeId: episodes[0]!.episodeId,
      l1MemoryId: "trace-step-sequence-a",
      channel: "explicit",
      polarity: "negative",
      magnitude: 1,
      rationale: "Wrong: check the precondition before execution and never retry blindly."
    });
    expect(feedback.repair).toMatchObject({
      skipped: false,
      attachedPolicyIds: expect.arrayContaining(basePolicyMemoryIds)
    });
    expect(feedback.repair?.repairJobId).toBeTruthy();
    const repair = repos.runtime.getDecisionRepair(feedback.repair!.repairId!)!;
    const repairJob = repos.runtime.getJob(feedback.repair!.repairJobId!)!;

    const run = await service.runWorkerOnce(10, {
      targetMemoryIds: [repairJob.targetMemoryId!]
    });
    expect(run.failed).toBe(0);
    expect(run.jobs).toEqual(expect.arrayContaining([
      expect.objectContaining({ jobType: "step_policy_repair", status: "succeeded" })
    ]));

    const repairedPolicies = policyPatterns.map((pattern) => {
      const current = repos.stepSequenceLearning.getStepPattern(pattern.id)!;
      expect(current.activePolicyVersionId).not.toBe(pattern.activePolicyVersionId);
      return repos.stepSequenceLearning.getPolicy(current.activePolicyVersionId!)!;
    });
    expect(repairedPolicies.every((policy) =>
      policy.policy.revision?.repairIds.includes(repair.id))).toBe(true);
    expect(repairedPolicies.every((policy) => policy.policy.supportEpisodeIds.length === 2))
      .toBe(true);
    expect(basePolicies.every((policy) =>
      repos.stepSequenceLearning.getPolicy(policy.id)?.status === "inactive")).toBe(true);
    expect(basePolicyMemoryIds.every((id) => repos.memories.get(id)?.status === "archived"))
      .toBe(true);

    const currentSkillPattern = repos.stepSequenceLearning.getSkillPattern(skillPattern.id)!;
    expect(currentSkillPattern.activeSkillMemoryId).toBe(baseSkillMemoryId);
    expect(repos.memories.get(baseSkillMemoryId)?.status).not.toBe("archived");
    const repairCandidates = repos.memories.list({
      memoryLayer: "Skill",
      status: ["resolving", "activated"]
    }, 100).filter((memory) => memory.tags.includes("repair_candidate"));
    expect(repairCandidates).toHaveLength(1);
    expect(repairCandidates[0]).toMatchObject({
      properties: {
        internal_info: {
          repair_origin: true,
          strict_trial: true,
          repair_id: repair.id,
          supersedes_skill_memory_id: baseSkillMemoryId
        }
      }
    });
    expect(repairCandidates[0]!.properties.internal_info
      .source_step_sequence_policy_version_ids).toEqual(expect.arrayContaining(
        repairedPolicies.map((policy) => policy.id)
      ));
    expect(operations.filter((operation) => operation === STEP_SEQUENCE_POLICY_REPAIR_OPERATION))
      .toHaveLength(2);
    expect(operations.filter((operation) => operation === STEP_POLICY_SKILL_OPERATION))
      .toHaveLength(2);
  });
});

function legacyArtifactMemory(
  id: string,
  layer: "L2" | "Skill",
  at: string
): MemoryRow {
  return {
    id,
    timeline: at,
    userId: "user-step-sequence",
    memoryType: layer === "Skill" ? "SkillMemory" : "LongTermMemory",
    status: "resolving",
    visibility: "private",
    memoryKey: `legacy:${id}`,
    memoryValue: id,
    tags: ["procedural", "legacy"],
    info: {},
    properties: {
      status: "resolving",
      internal_info: {
        memory_layer: layer,
        memory_kind: layer === "Skill" ? "skill" : "policy",
        step_policy_sequence_skill: { executable: true }
      }
    },
    memoryLayer: layer,
    contentHash: `${id}-hash`,
    version: 1,
    createdAt: at,
    updatedAt: at,
    deletedAt: null
  };
}

function persistStepSequenceEpisode(
  repos: Repositories,
  suffix: string,
  labels: string[],
  terminalReward = 1
): { episodeId: string; pathId: string } {
  const at = `2026-08-25T00:0${suffix === "a" ? 0 : 1}:00.000Z`;
  const sessionId = `session-step-sequence-${suffix}`;
  const episodeId = `episode-step-sequence-${suffix}`;
  const rawTurnId = `raw-step-sequence-${suffix}`;
  repos.runtime.createSession({
    id: sessionId,
    userId: "user-step-sequence",
    source: "codex",
    profileId: "default",
    status: "closed",
    meta: {},
    openedAt: at,
    lastSeenAt: at,
    closedAt: at,
    updatedAt: at
  });
  repos.runtime.createEpisode({
    id: episodeId,
    sessionId,
    userId: "user-step-sequence",
    status: "closed",
    l1MemoryIds: [],
    rawTurnIds: [rawTurnId],
    feedbackIds: [],
    decisionRepairIds: [],
    l2PolicyIds: [],
    l3WorldModelIds: [],
    skillMemoryIds: [],
    turnCount: 1,
    rTask: terminalReward,
    rewardDetail: {},
    pipelineStatus: "succeeded",
    meta: {},
    openedAt: at,
    closedAt: at,
    updatedAt: at
  });
  repos.runtime.insertRawTurn({
    id: rawTurnId,
    sessionId,
    episodeId,
    turnId: `turn-${suffix}`,
    userId: "user-step-sequence",
    userText: "Run the reusable two-stage workflow",
    assistantText: "The workflow completed and was verified",
    toolCalls: [],
    toolResults: [],
    sourceMemoryIds: [],
    usage: {},
    status: "succeeded",
    createdAt: at
  });
  const traceId = `trace-step-sequence-${suffix}`;
  repos.memories.insert(traceFixtureMemory({
    id: traceId,
    episodeId,
    sessionId,
    rawTurnId,
    at
  }));
  repos.runtime.appendEpisodeTurn(episodeId, rawTurnId, traceId, at);
  const state = emptyObservedState();
  const sourceSnapshotHash = `step-sequence-snapshot-${suffix}`;
  const provenance = {
    algorithmVersion: "episode-procedural-reconstruction.v8",
    model: "fixture-model",
    sourceSnapshotHash
  };
  const steps: ExecutionStepV1[] = labels.map((label, stepIndex) => ({
    id: `step-${suffix}-${label}`,
    schemaVersion: "execution-step.v1",
    episodeId,
    rawTurnId,
    turnIndex: 0,
    stepIndex,
    preStateId: state.id,
    action: {
      kind: "tool_action",
      type: `action_${label.toLowerCase()}`,
      intent: `${label} perform reusable operation`,
      summary: `${label} operation completed with observable evidence`,
      eventRefs: [`event:${suffix}:${label}`],
      toolName: "exec"
    },
    actionEffectDelta: [],
    actionPostStateId: state.id,
    externalObservationDelta: [],
    postStateId: state.id,
    outcome: { status: "success", evidenceRefs: [`event:${suffix}:${label}`] },
    cost: { toolCalls: 1, errorCount: 0 },
    provenance
  }));
  const decision: SpanSegmentationDecisionV1 = {
    spanIndex: 0,
    stepIds: steps.map((step) => step.id),
    localGoal: "Compatibility envelope",
    capabilityGoal: "Execute the fixture workflow",
    procedureSemantic: labels.join(" -> "),
    entryCondition: "Workflow requested",
    exitCondition: "Workflow verified",
    terminationStatus: "success",
    evidenceRefs: steps.map((step) => step.id),
    reason: "Fixture compatibility Span",
    confidence: 1
  };
  const span: ProceduralSpanV1 = {
    id: `span-step-sequence-${suffix}`,
    schemaVersion: "procedural-span.v1",
    episodeId,
    spanIndex: 0,
    localGoal: decision.localGoal,
    capabilityGoal: decision.capabilityGoal,
    procedureSemantic: decision.procedureSemantic,
    entryCondition: decision.entryCondition,
    stepIds: [...decision.stepIds],
    rawTurnIds: [rawTurnId],
    preStateId: state.id,
    postStateId: state.id,
    termination: {
      status: "success",
      exitCondition: decision.exitCondition,
      evidenceRefs: [...decision.evidenceRefs]
    },
    cost: {
      steps: steps.length,
      toolCalls: steps.length,
      retryCount: 0,
      recoveryCount: 0,
      errorCount: 0
    },
    segmentation: { reason: decision.reason, confidence: 1 },
    provenance
  };
  const path = buildEpisodeProceduralPath({
    episodeId,
    states: [state],
    steps,
    spans: [span],
    segmentationDecisions: [decision],
    sourceSnapshotHash,
    terminalReward
  });
  const saved = repos.proceduralPaths.save({
    path,
    namespaceId: "user-step-sequence",
    createdAt: at
  });
  return { episodeId, pathId: saved.record.id };
}

function traceFixtureMemory(input: {
  id: string;
  episodeId: string;
  sessionId: string;
  rawTurnId: string;
  at: string;
}): MemoryRow {
  return {
    id: input.id,
    timeline: input.at,
    userId: "user-step-sequence",
    sessionId: input.sessionId,
    memoryType: "LongTermMemory",
    status: "activated",
    visibility: "private",
    memoryKey: `trace:${input.rawTurnId}`,
    memoryValue: `Trace for ${input.rawTurnId}`,
    tags: ["trace", "fixture"],
    info: {
      episode_id: input.episodeId,
      raw_turn_id: input.rawTurnId
    },
    properties: {
      status: "activated",
      internal_info: {
        memory_layer: "L1",
        memory_kind: "trace",
        source_raw_turn_id: input.rawTurnId,
        raw_turn_id: input.rawTurnId,
        trace: {
          raw_turn_id: input.rawTurnId,
          episode_id: input.episodeId
        }
      }
    },
    memoryLayer: "L1",
    contentHash: `${input.id}-hash`,
    version: 1,
    createdAt: input.at,
    updatedAt: input.at,
    deletedAt: null
  };
}

function stepSequenceEmbedder(seen: string[]): Embedder {
  const labels = ["A", "B", "C", "W", "X", "Y", "Z", "D", "E", "F"];
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "step-sequence-test"
    },
    isRemote() { return false; },
    async embed(texts: string[]) {
      seen.push(...texts);
      return texts.map((text) => {
        const label = labels.find((candidate) => text.includes(`Intent: ${candidate} `));
        if (!label) throw new Error(`Unknown Step semantic: ${text}`);
        return labels.map((candidate) => candidate === label ? 1 : 0);
      });
    },
    async embedOne(text: string) {
      return (await this.embed([text]))[0]!;
    },
    status() {
      return { provider: "local", model: "step-sequence-test", configured: true, remote: false };
    }
  };
}

function stepSequenceLlm(operations: string[]): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/step-sequence-test",
      model: "step-sequence-test"
    },
    isConfigured() { return true; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: { operation: string }
    ): Promise<T> {
      operations.push(options.operation);
      const payload = JSON.parse(messages.find((message) => message.role === "user")!.content) as {
        evidence: Array<Record<string, unknown>>;
      };
      if (options.operation === STEP_SEQUENCE_POLICY_OPERATION ||
          options.operation === STEP_SEQUENCE_POLICY_REPAIR_OPERATION) {
        const evidence = payload.evidence as Array<{
          occurrence_id: string;
          steps: Array<{ step_occurrence_id: string; step_id: string; intent: string }>;
        }>;
        const decisionRepair = (payload as unknown as {
          decision_repair?: { repair_id?: string }
        }).decision_repair;
        const firstLabel = evidence[0]!.steps[0]!.intent.split(" ")[0]!;
        return {
          title: `Execute ${firstLabel} local procedure`,
          goal_pattern: `Complete the reusable ${firstLabel} procedure`,
          trigger_conditions: [`The ${firstLabel} procedure is required`],
          procedure_steps: [{
            instruction: `Run the observed ${firstLabel} procedure in order`,
            evidence_refs: [
              evidence[0]!.occurrence_id,
              evidence[0]!.steps[0]!.step_occurrence_id,
              evidence[1]!.occurrence_id,
              evidence[1]!.steps[0]!.step_id,
              ...(decisionRepair?.repair_id ? [decisionRepair.repair_id] : [])
            ]
          }],
          verification_steps: [{
            check: `Check the ${firstLabel} result`,
            success_signal: `${firstLabel} evidence is present`,
            evidence_refs: [
              evidence[0]!.steps.at(-1)!.step_id,
              evidence[1]!.steps.at(-1)!.step_id
            ]
          }],
          do_not_apply_when: [],
          evidence_occurrence_ids: evidence.map((item) => item.occurrence_id),
          confidence: 0.95
        } as unknown as T;
      }
      const evidence = payload.evidence as Array<{
        sequence_occurrence_id: string;
        full_step_path: Array<{ step_occurrence_id: string; step_id: string }>;
      }>;
      return {
        name: "run_two_stage_workflow",
        display_title: "Run and verify the two-stage workflow",
        retrieval_blurb: "Execute two reusable local procedures with evidence-based verification.",
        trigger_context: "Use when the two-stage workflow must be completed end to end.",
        summary: "Run the first local procedure, retain conditional gap work, then run and verify the second.",
        steps: [{
          title: "Execute both procedures",
          body: "Run the observed procedures in order and preserve only necessary intervening work.",
          evidence_refs: [
            evidence[0]!.sequence_occurrence_id,
            evidence[0]!.full_step_path[0]!.step_occurrence_id,
            evidence[1]!.sequence_occurrence_id,
            evidence[1]!.full_step_path[0]!.step_id
          ]
        }],
        verification: [{
          check: "Verify the final procedure result",
          success_signal: "Both Episodes contain successful final evidence",
          evidence_refs: [
            evidence[0]!.full_step_path.at(-1)!.step_id,
            evidence[1]!.full_step_path.at(-1)!.step_id
          ]
        }],
        do_not_use_when: [],
        tools: ["exec"],
        tags: ["workflow", "verified"],
        evidence_occurrence_ids: evidence.map((item) => item.sequence_occurrence_id),
        confidence: 0.94
      } as unknown as T;
    },
    status() {
      return { provider: "host", model: "step-sequence-test", configured: true, remote: false };
    }
  };
}
