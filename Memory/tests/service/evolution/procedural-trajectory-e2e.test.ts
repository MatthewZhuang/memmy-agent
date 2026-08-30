import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  type Embedder,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage,
  type RuntimeNamespace
} from "../../../src/index.js";
import { skillMetaFromMemory } from "../../../src/algorithm/plugin-algorithms.js";
import { Repositories } from "../../../src/storage/repositories.js";
import { isRecord } from "../../../src/utils/json.js";
import {
  createCapturingEmbedder,
  createMemoryServiceFixture,
  runWorkerRounds
} from "../../fixtures/memory-service-fixture.js";

const USER_ID = "procedural-e2e-user";
const STEP_SEMANTICS_OPERATION_PREFIX = "procedural.step_semantics.";

const {
  cleanup,
  createTestService
} = createMemoryServiceFixture();

afterEach(cleanup);

describe("procedural trajectory direct Skill E2E", () => {
  it("routes Step Semantics through the evolution LLM instead of the summary LLM", async () => {
    const summaryOperations: string[] = [];
    const evolutionOperations: string[] = [];
    const summaryLlm = proceduralE2eLlm(summaryOperations, {
      model: "summary-route-model",
      semanticSource: "summary-route"
    });
    const evolutionLlm = proceduralE2eLlm(evolutionOperations, {
      model: "evolution-route-model",
      semanticSource: "evolution-route"
    });
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm: summaryLlm,
      skillLlm: evolutionLlm,
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);

    const episode = await executeSuccessfulEpisode(service, "codex", "model-route");
    await runWorkerRounds(service, 16);

    const path = repos.proceduralTrajectory.getActivePath(episode.episodeId);
    expect(path).toBeDefined();
    expect(summaryOperations).toContain("capture.summarize");
    expect(summaryOperations.filter(isStepSemanticsOperation)).toEqual([]);
    expect(evolutionOperations.filter(isStepSemanticsOperation).length).toBeGreaterThan(0);
    expect(path?.modelSignature).toBe("host:evolution-route-model");
    expect(path?.path.steps.length).toBeGreaterThan(0);
    expect(path?.path.steps.every((step) =>
      step.intent.startsWith("[evolution-route]"))).toBe(true);
  });

  it("creates one cross-Agent Skill from two successful Episodes and retires it on reward drift", async () => {
    const operations: string[] = [];
    const llm = proceduralE2eLlm(operations);
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);

    const first = await executeSuccessfulEpisode(service, "codex", "first");
    await runWorkerRounds(service, 16);
    const second = await executeSuccessfulEpisode(service, "cursor", "second");
    await runWorkerRounds(service, 20);

    const firstPath = repos.proceduralTrajectory.getActivePath(first.episodeId);
    const secondPath = repos.proceduralTrajectory.getActivePath(second.episodeId);
    expect(firstPath?.path.steps).toHaveLength(5);
    expect(secondPath?.path.steps).toHaveLength(5);
    expect(repos.proceduralTrajectory.listWindowsForPath(firstPath!.id, 5)).toHaveLength(1);
    expect(repos.proceduralTrajectory.listWindowsForPath(secondPath!.id, 5)).toHaveLength(1);

    const clusterRow = db.db.prepare(
      `SELECT id
       FROM trajectory_window_clusters
       WHERE user_id = ? AND scale = 5 AND status = 'active'
       ORDER BY created_at ASC
       LIMIT 1`
    ).get(USER_ID) as { id: string } | undefined;
    expect(clusterRow).toBeDefined();
    const cluster = repos.proceduralTrajectory.getClusterHead(clusterRow!.id)!;
    const positiveVersion = repos.proceduralTrajectory.getClusterVersion(
      cluster.activeVersionId!
    )!;
    expect(positiveVersion).toMatchObject({
      supportEpisodeCount: 2,
      counterexampleEpisodeCount: 0
    });
    const skillMemoryId = cluster.activeSkillMemoryId;
    expect(skillMemoryId).toBeTruthy();
    const skillMemory = repos.memories.get(skillMemoryId!);
    expect(skillMemory).toMatchObject({
      memoryLayer: "Skill",
      memoryType: "SkillMemory",
      userId: USER_ID
    });
    expect(skillMemory?.sessionId).toBeUndefined();
    expect(skillMemory?.agentId).toBeUndefined();
    expect(skillMetaFromMemory(skillMemory!)?.sourcePolicyIds).toEqual([]);
    expect(skillMemory?.properties.internal_info.plugin_algorithm)
      .toBe("procedural.pattern.skill.v1");
    expect(skillMemory?.properties.internal_info.completion_activated).toBe(false);
    expect(skillMemory?.properties.internal_info.completion_shared_prefix_anchor_ids)
      .toEqual([]);
    expect(skillMemory?.properties.internal_info.completion_shared_suffix_anchor_ids)
      .toEqual([]);
    expect(service.listSkills({ userId: USER_ID }).items.map((item) => item.id))
      .toContain(skillMemoryId);
    expect(service.getSkill(skillMemoryId!, { namespace: first.namespace })).toMatchObject({
      id: skillMemoryId,
      sourcePolicyIds: []
    });
    expect(operations.filter(isStepSemanticsOperation)).toHaveLength(2);
    expect(operations).toContain("procedural.procedural-pattern-skill.v12");

    const familyBeforeDrift = db.db.prepare(
      `SELECT id, active_revision_id AS activeRevisionId
       FROM trajectory_window_families
       WHERE user_id = ? AND scale = 5 AND status = 'active'`
    ).get(USER_ID) as { id: string; activeRevisionId: string };
    const familyRevisionBeforeDrift = repos.proceduralTrajectory.getFamilyRevision(
      familyBeforeDrift.activeRevisionId
    )!;

    const secondPathBeforeDrift = {
      id: secondPath!.id,
      pathHash: secondPath!.pathHash,
      stepIds: secondPath!.path.steps.map((step) => step.id)
    };
    const semanticsCallsBeforeDrift = operations.filter(isStepSemanticsOperation).length;
    await service.feedback({
      namespace: second.namespace,
      sessionId: second.sessionId,
      episodeId: second.episodeId,
      l1MemoryId: second.l1MemoryId,
      channel: "explicit",
      polarity: "negative",
      magnitude: 1,
      rationale: "regression invalidated this result"
    });
    await runWorkerRounds(service, 20);

    const secondEpisode = repos.runtime.getEpisode(second.episodeId);
    expect(secondEpisode?.rTask).toBeLessThan(0);
    const secondPathAfterDrift = repos.proceduralTrajectory.getActivePath(second.episodeId)!;
    expect({
      id: secondPathAfterDrift.id,
      pathHash: secondPathAfterDrift.pathHash,
      stepIds: secondPathAfterDrift.path.steps.map((step) => step.id)
    }).toEqual(secondPathBeforeDrift);
    expect(operations.filter(isStepSemanticsOperation)).toHaveLength(
      semanticsCallsBeforeDrift
    );
    const secondWindow = repos.proceduralTrajectory.listWindowsForPath(
      secondPathAfterDrift.id,
      5
    )[0]!;
    expect(secondWindow.evidenceRole).toBe("counterexample");
    expect(secondWindow.terminalReward).toBeLessThan(0);

    const familyAfterDrift = repos.proceduralTrajectory.getFamilyHead(familyBeforeDrift.id)!;
    const familyRevisionAfterDrift = repos.proceduralTrajectory.getFamilyRevision(
      familyAfterDrift.activeRevisionId!
    )!;
    expect(familyRevisionAfterDrift.id).not.toBe(familyRevisionBeforeDrift.id);
    expect(familyRevisionAfterDrift.evidenceHash)
      .not.toBe(familyRevisionBeforeDrift.evidenceHash);
    expect(repos.proceduralTrajectory.listFamilyMembers(familyRevisionAfterDrift.id)
      .map((member) => member.occurrenceId)).toEqual(
        repos.proceduralTrajectory.listFamilyMembers(familyRevisionBeforeDrift.id)
          .map((member) => member.occurrenceId)
      );

    const clusterAfterDrift = repos.proceduralTrajectory.getClusterHead(cluster.id)!;
    expect(clusterAfterDrift.status).toBe("active");
    const driftVersion = repos.proceduralTrajectory.getClusterVersion(
      clusterAfterDrift.activeVersionId!
    )!;
    expect(driftVersion).toMatchObject({
      supportEpisodeCount: 1,
      counterexampleEpisodeCount: 1
    });
    expect(repos.memories.get(skillMemoryId!)?.status).toBe("archived");
    const rejectedDecision = repos.proceduralTrajectory.getSkillVersion(
      clusterAfterDrift.activeSkillVersionId!
    )!;
    expect(rejectedDecision.skillMemoryId).toBeUndefined();
    expect(rejectedDecision.payload).toMatchObject({
      admitted: false,
      reason: "cluster-no-longer-qualified"
    });
  });

  it("absorbs a third similar Episode into the same V2 cluster without archiving the Skill", async () => {
    const operations: string[] = [];
    const llm = proceduralE2eLlm(operations);
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: constantProceduralEmbedder()
    });
    const repos = new Repositories(db.db);

    await executeSuccessfulEpisode(service, "codex", "absorb-first");
    await runWorkerRounds(service, 16);
    await executeSuccessfulEpisode(service, "cursor", "absorb-second");
    await runWorkerRounds(service, 20);

    const clusterAfterTwo = db.db.prepare(
      `SELECT id, active_skill_memory_id AS skillMemoryId
       FROM trajectory_window_clusters
       WHERE user_id = ? AND scale = 5 AND status = 'active'
         AND active_skill_memory_id IS NOT NULL
       ORDER BY created_at ASC
       LIMIT 1`
    ).get(USER_ID) as { id: string; skillMemoryId: string } | undefined;
    expect(clusterAfterTwo).toBeDefined();
    const skillAfterTwo = repos.memories.get(clusterAfterTwo!.skillMemoryId);
    expect(skillAfterTwo?.status).not.toBe("archived");

    await executeSuccessfulEpisode(service, "claude_code", "absorb-third");
    await runWorkerRounds(service, 20);

    const clusterAfterThree = repos.proceduralTrajectory.getClusterHead(clusterAfterTwo!.id);
    expect(clusterAfterThree?.status).toBe("active");
    expect(clusterAfterThree?.activeSkillMemoryId).toBe(clusterAfterTwo!.skillMemoryId);
    const version = repos.proceduralTrajectory.getClusterVersion(
      clusterAfterThree!.activeVersionId!
    )!;
    expect(version.supportEpisodeCount).toBe(3);
    expect(repos.memories.get(clusterAfterTwo!.skillMemoryId)?.status).not.toBe("archived");
  });

  it("shares one Step path between V2 local and V3 long-trajectory Skill mining", async () => {
    const operations: string[] = [];
    const skillPayloads: Array<Record<string, unknown>> = [];
    let semanticSource = "procedural-e2e-original";
    const llm = proceduralE2eLlm(operations, {
      skillPayloads,
      semanticSource: () => semanticSource
    });
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: constantProceduralEmbedder()
    });
    const repos = new Repositories(db.db);

    const first = await executeLongSuccessfulEpisode(service, "codex", "long-first");
    await runWorkerRounds(service, 20);
    const second = await executeLongSuccessfulEpisode(service, "cursor", "long-second");
    await runWorkerRounds(service, 30);

    const firstPath = repos.proceduralTrajectory.getActivePath(first.episodeId)!;
    const secondPath = repos.proceduralTrajectory.getActivePath(second.episodeId)!;
    expect(firstPath.path.steps).toHaveLength(20);
    expect(secondPath.path.steps).toHaveLength(20);
    expect(operations.filter(isStepSemanticsOperation)).toHaveLength(2);
    expect(repos.proceduralTrajectory.listWindowsForPath(firstPath.id, 15).length)
      .toBeGreaterThan(0);
    expect(repos.proceduralTrajectory.listWindowsForPath(secondPath.id, 15).length)
      .toBeGreaterThan(0);

    const candidate = db.db.prepare(
      `SELECT id, active_version_id AS activeVersionId,
              active_skill_memory_id AS activeSkillMemoryId
       FROM long_trajectory_candidates
       WHERE user_id = ? AND status = 'active' AND active_skill_memory_id IS NOT NULL
       ORDER BY created_at ASC LIMIT 1`
    ).get(USER_ID) as {
      id: string;
      activeVersionId: string;
      activeSkillMemoryId: string;
    } | undefined;
    expect(candidate).toBeDefined();
    const candidateVersion = repos.longTrajectory.getCandidateVersion(
      candidate!.activeVersionId
    );
    expect(candidateVersion?.supportEpisodeIds.sort()).toEqual(
      [first.episodeId, second.episodeId].sort()
    );
    const skill = repos.memories.get(candidate!.activeSkillMemoryId);
    expect(skill).toMatchObject({
      memoryLayer: "Skill",
      memoryType: "SkillMemory",
      userId: USER_ID,
      status: "resolving"
    });
    expect(skill?.properties.internal_info).toMatchObject({
      plugin_algorithm: "procedural.long-trajectory.skill.v1",
      source_cluster_id: candidate!.id,
      status: "candidate"
    });
    expect(skillMetaFromMemory(skill!)?.sourcePolicyIds).toEqual([]);
    expect(service.listSkills({ userId: USER_ID }).items.map((item) => item.id))
      .toContain(candidate!.activeSkillMemoryId);

    const initialCandidateVersionId = candidate!.activeVersionId;
    const initialCandidate = repos.longTrajectory.getCandidate(candidate!.id)!;
    const initialSkillVersionId = initialCandidate.activeSkillVersionId;
    const initialSkillMemoryId = initialCandidate.activeSkillMemoryId!;
    const initialMemoryVersion = repos.memories.get(initialSkillMemoryId)!.version;
    const candidateCountBefore = (db.db.prepare(
      `SELECT COUNT(*) AS count FROM long_trajectory_candidates WHERE user_id = ?`
    ).get(USER_ID) as { count: number }).count;
    // Episode C deliberately uses different canonical wording, so its exact
    // structure hash cannot find the old Candidate. Candidate-first projection
    // must still update S1 because the retained Span sequence aligns.
    semanticSource = "procedural-e2e-paraphrased";
    const third = await executeLongSuccessfulEpisode(service, "claude_code", "long-third");
    await runWorkerRounds(service, 30);

    const updatedCandidate = repos.longTrajectory.getCandidate(candidate!.id)!;
    expect((db.db.prepare(
      `SELECT COUNT(*) AS count FROM long_trajectory_candidates WHERE user_id = ?`
    ).get(USER_ID) as { count: number }).count).toBe(candidateCountBefore);
    expect(updatedCandidate.activeVersionId).not.toBe(initialCandidateVersionId);
    expect(updatedCandidate.activeSkillVersionId).not.toBe(initialSkillVersionId);
    expect(updatedCandidate.activeSkillMemoryId).toBe(initialSkillMemoryId);
    const updatedCandidateVersion = repos.longTrajectory.getCandidateVersion(
      updatedCandidate.activeVersionId!
    )!;
    expect(updatedCandidateVersion.supportEpisodeIds.sort()).toEqual(
      [first.episodeId, second.episodeId, third.episodeId].sort()
    );
    expect(repos.longTrajectory.getCandidateVersion(initialCandidateVersionId)?.status)
      .toBe("superseded");
    expect(repos.memories.get(initialSkillMemoryId)!.version).toBeGreaterThan(initialMemoryVersion);
    const incrementalPayload = [...skillPayloads].reverse().find((payload) => {
      const pattern = payload.pattern;
      return isRecord(pattern) && isRecord(pattern.origin) &&
        pattern.origin.kind === "long_trajectory" && isRecord(payload.existing_skill_read_only);
    });
    expect(incrementalPayload).toBeDefined();
    expect(incrementalPayload?.existing_skill_read_only).toMatchObject({
      memory_id: initialSkillMemoryId,
      memory_version: initialMemoryVersion
    });
    expect(incrementalPayload?.evidence_delta_read_only).toMatchObject({
      previous_candidate_version_id: initialCandidateVersionId,
      added_episode_ids: [third.episodeId]
    });
    expect((incrementalPayload?.occurrences as Array<{ episode_id?: string }> | undefined)
      ?.map((occurrence) => occurrence.episode_id)).toContain(third.episodeId);

    service.archiveMemory(first.l1MemoryId, { namespace: first.namespace });
    expect(repos.longTrajectory.getCandidate(candidate!.id)?.status).toBe("retired");
    expect(repos.memories.get(candidate!.activeSkillMemoryId)?.status).toBe("archived");
    expect(service.listSkills({ userId: USER_ID }).items.map((item) => item.id))
      .not.toContain(candidate!.activeSkillMemoryId);
  });

  it("updates a directly recalled Candidate even when Episode reverse lookup cannot find it", async () => {
    const operations: string[] = [];
    let semanticSource = "direct-candidate-original";
    const llm = proceduralE2eLlm(operations, {
      semanticSource: () => semanticSource
    });
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: constantProceduralEmbedder()
    });
    const repos = new Repositories(db.db);

    const first = await executeLongSuccessfulEpisode(service, "codex", "direct-first");
    await runWorkerRounds(service, 20);
    const second = await executeLongSuccessfulEpisode(service, "cursor", "direct-second");
    await runWorkerRounds(service, 30);
    const candidateRow = db.db.prepare(
      `SELECT id FROM long_trajectory_candidates
       WHERE user_id = ? AND status = 'active' AND active_skill_memory_id IS NOT NULL
       ORDER BY created_at ASC LIMIT 1`
    ).get(USER_ID) as { id: string } | undefined;
    expect(candidateRow).toBeDefined();
    const original = repos.longTrajectory.getCandidate(candidateRow!.id)!;
    const candidateCount = (db.db.prepare(
      `SELECT COUNT(*) AS count FROM long_trajectory_candidates WHERE user_id = ?`
    ).get(USER_ID) as { count: number }).count;

    // Simulate an Episode Top-K miss while preserving the Candidate payload.
    // Reverse lookup can no longer find S1, but direct Candidate recall can.
    db.db.prepare(
      `UPDATE long_trajectory_candidate_versions
       SET support_episode_ids_json = ? WHERE id = ?`
    ).run(JSON.stringify(["episode-not-recalled"]), original.activeVersionId);
    expect(repos.longTrajectory.listActiveCandidatesLinkedToEpisodes({
      userId: USER_ID,
      algorithmVersion: "reference-span-sequence-mining.v1",
      configHash: original.configHash,
      episodeIds: [first.episodeId, second.episodeId]
    })).toEqual([]);
    expect(repos.longTrajectory.listActiveCandidates({
      userId: USER_ID,
      algorithmVersion: "reference-span-sequence-mining.v1",
      configHash: original.configHash
    }).map((item) => item.id)).toContain(original.id);

    semanticSource = "direct-candidate-paraphrased";
    const third = await executeLongSuccessfulEpisode(service, "claude_code", "direct-third");
    await runWorkerRounds(service, 30);

    const updated = repos.longTrajectory.getCandidate(original.id)!;
    expect(updated.activeVersionId).not.toBe(original.activeVersionId);
    expect((db.db.prepare(
      `SELECT COUNT(*) AS count FROM long_trajectory_candidates WHERE user_id = ?`
    ).get(USER_ID) as { count: number }).count).toBe(candidateCount);
    expect(repos.longTrajectory.getCandidateVersion(updated.activeVersionId!)
      ?.supportEpisodeIds.sort()).toEqual(
      [first.episodeId, second.episodeId, third.episodeId].sort()
    );
  });

  it("keeps the last-known-good V3 Skill live when its replacement fails validation", async () => {
    const operations: string[] = [];
    let semanticSource = "last-known-good-original";
    let rejectReplacement = false;
    const llm = proceduralE2eLlm(operations, {
      semanticSource: () => semanticSource,
      skillResponse(originKind, _payload, response) {
        if (originKind !== "long_trajectory" || !rejectReplacement) return response;
        const procedureSteps = response.procedure_steps as Array<Record<string, unknown>>;
        return {
          ...response,
          procedure_steps: procedureSteps.map((step, index) => index === 0
            ? { ...step, source_anchor_ids: ["not-an-allowed-evidence-ref"] }
            : step)
        };
      }
    });
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: constantProceduralEmbedder()
    });
    const repos = new Repositories(db.db);

    const first = await executeLongSuccessfulEpisode(service, "codex", "known-good-first");
    await runWorkerRounds(service, 20);
    const second = await executeLongSuccessfulEpisode(service, "cursor", "known-good-second");
    await runWorkerRounds(service, 30);

    const candidateRow = db.db.prepare(
      `SELECT id FROM long_trajectory_candidates
       WHERE user_id = ? AND status = 'active' AND active_skill_memory_id IS NOT NULL
       ORDER BY created_at ASC LIMIT 1`
    ).get(USER_ID) as { id: string } | undefined;
    expect(candidateRow).toBeDefined();
    const originalCandidate = repos.longTrajectory.getCandidate(candidateRow!.id)!;
    const originalCandidateVersionId = originalCandidate.activeVersionId!;
    const originalSkillVersionId = originalCandidate.activeSkillVersionId!;
    const originalSkillMemoryId = originalCandidate.activeSkillMemoryId!;
    const originalSkillMemory = repos.memories.get(originalSkillMemoryId)!;
    expect(originalSkillMemory.status).toBe("resolving");

    // Episode C updates the existing Candidate, but its proposed replacement
    // deliberately violates the citation contract.
    semanticSource = "last-known-good-paraphrased";
    rejectReplacement = true;
    const third = await executeLongSuccessfulEpisode(
      service,
      "claude_code",
      "known-good-third"
    );
    await runWorkerRounds(service, 30);

    const current = repos.longTrajectory.getCandidate(originalCandidate.id)!;
    expect(current.activeVersionId).not.toBe(originalCandidateVersionId);
    expect(current.activeSkillVersionId).toBe(originalSkillVersionId);
    expect(current.activeSkillMemoryId).toBe(originalSkillMemoryId);
    expect(repos.longTrajectory.getSkillVersion(originalSkillVersionId)?.status).toBe("active");
    expect(repos.memories.get(originalSkillMemoryId)).toMatchObject({
      status: "resolving",
      version: originalSkillMemory.version
    });
    expect(service.listSkills({ userId: USER_ID }).items.map((item) => item.id))
      .toContain(originalSkillMemoryId);

    const replacementVersion = repos.longTrajectory.getCandidateVersion(current.activeVersionId!);
    expect(replacementVersion?.supportEpisodeIds.sort()).toEqual(
      [first.episodeId, second.episodeId, third.episodeId].sort()
    );
    const decisions = db.db.prepare(
      `SELECT id, status, skill_memory_id AS skillMemoryId, payload_json AS payloadJson
       FROM long_trajectory_skill_versions
       WHERE candidate_id = ? ORDER BY version_no ASC`
    ).all(current.id) as Array<{
      id: string;
      status: string;
      skillMemoryId: string | null;
      payloadJson: string;
    }>;
    expect(decisions).toHaveLength(2);
    expect(decisions[0]).toMatchObject({
      id: originalSkillVersionId,
      status: "active",
      skillMemoryId: originalSkillMemoryId
    });
    expect(decisions[1]).toMatchObject({ status: "superseded", skillMemoryId: null });
    expect(JSON.parse(decisions[1]!.payloadJson)).toMatchObject({
      admitted: false,
      reason: "invalid-evidence-anchor",
      preservedActiveSkillVersionId: originalSkillVersionId,
      preservedActiveSkillMemoryId: originalSkillMemoryId
    });
  });

  it("does not resurrect a V3 Skill archived while its induction LLM is running", async () => {
    const skillStarted = deferred<void>();
    const releaseSkill = deferred<void>();
    let blockLongTrajectorySkill = false;
    const operations: string[] = [];
    const llm = proceduralE2eLlm(operations, {
      async beforeSkillResponse(originKind) {
        if (!blockLongTrajectorySkill || originKind !== "long_trajectory") return;
        skillStarted.resolve();
        await releaseSkill.promise;
      }
    });
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: constantProceduralEmbedder()
    });
    const repos = new Repositories(db.db);

    const first = await executeLongSuccessfulEpisode(service, "codex", "slow-first");
    await runWorkerRounds(service, 20);
    await executeLongSuccessfulEpisode(service, "cursor", "slow-second");
    await runWorkerRounds(service, 30);

    const candidateRow = db.db.prepare(
      `SELECT id FROM long_trajectory_candidates
       WHERE user_id = ? AND status = 'active' AND active_skill_memory_id IS NOT NULL
       ORDER BY created_at ASC LIMIT 1`
    ).get(USER_ID) as { id: string } | undefined;
    expect(candidateRow).toBeDefined();
    const candidate = repos.longTrajectory.getCandidate(candidateRow!.id)!;
    const archivedSkillId = candidate.activeSkillMemoryId!;
    blockLongTrajectorySkill = true;
    const at = new Date().toISOString();
    repos.runtime.enqueueJob({
      id: `job_slow_v3_${candidate.id}`,
      jobType: "long_trajectory_skill_induction",
      status: "queued",
      userId: USER_ID,
      sessionId: first.sessionId,
      episodeId: first.episodeId,
      targetMemoryId: candidate.id,
      payload: {
        candidateId: candidate.id,
        candidateVersionId: candidate.activeVersionId,
        inductionVersion: "slow-governance-race-test"
      },
      attempts: 0,
      maxAttempts: 1,
      createdAt: at,
      updatedAt: at
    });

    const slowRun = service.runWorkerOnce(100);
    await skillStarted.promise;
    service.archiveMemory(archivedSkillId, { namespace: first.namespace });
    expect(repos.memories.getIncludingDeleted(archivedSkillId)?.status).toBe("archived");
    expect(repos.memories.getIncludingDeleted(archivedSkillId)
      ?.properties.internal_info.procedural_governance).toMatchObject({
      disabled: true,
      action: "archive"
    });

    releaseSkill.resolve();
    await slowRun;

    expect(repos.runtime.getJob(`job_slow_v3_${candidate.id}`)?.status).toBe("succeeded");
    const current = repos.longTrajectory.getCandidate(candidate.id)!;
    expect(current.activeSkillMemoryId).toBeUndefined();
    expect(repos.longTrajectory.getSkillVersion(current.activeSkillVersionId!)?.payload)
      .toMatchObject({ admitted: false, reason: "governance-disabled" });
    expect(repos.memories.getIncludingDeleted(archivedSkillId)?.status).toBe("archived");
    const liveLongTrajectorySkills = db.db.prepare(
      `SELECT COUNT(*) AS count FROM memories
       WHERE user_id = ? AND memory_layer = 'Skill' AND deleted_at IS NULL
         AND status != 'archived'
         AND json_extract(properties_json, '$.internal_info.plugin_algorithm') =
           'procedural.long-trajectory.skill.v1'`
    ).get(USER_ID) as { count: number };
    expect(liveLongTrajectorySkills.count).toBe(0);
  });

  it("removes deactivated Path evidence from its Family and canonical Skill", async () => {
    const operations: string[] = [];
    const llm = proceduralE2eLlm(operations);
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);
    const first = await executeSuccessfulEpisode(service, "codex", "path-active-first");
    await runWorkerRounds(service, 16);
    const invalidated = await executeSuccessfulEpisode(service, "cursor", "path-invalidated");
    await runWorkerRounds(service, 20);

    const invalidatedPath = repos.proceduralTrajectory.getActivePath(invalidated.episodeId)!;
    const pairClusterRow = db.db.prepare(
      `SELECT id FROM trajectory_window_clusters
       WHERE user_id = ? AND scale = 5 AND status = 'active'
         AND active_skill_memory_id IS NOT NULL`
    ).get(USER_ID) as { id: string };
    const pairCluster = repos.proceduralTrajectory.getClusterHead(pairClusterRow.id)!;
    const skillMemoryId = pairCluster.activeSkillMemoryId!;

    service.archiveMemory(invalidated.l1MemoryId, { namespace: invalidated.namespace });
    expect(repos.proceduralTrajectory.getActivePath(invalidated.episodeId)).toBeUndefined();
    expect(repos.proceduralTrajectory.getPath(invalidatedPath.id)?.status).toBe("superseded");
    expect(repos.memories.get(skillMemoryId)?.status).toBe("archived");

    await runWorkerRounds(service, 20);
    const activeFamilies = db.db.prepare(
      `SELECT id, active_revision_id AS activeRevisionId
       FROM trajectory_window_families
       WHERE user_id = ? AND scale = 5 AND status = 'active'`
    ).all(USER_ID) as Array<{ id: string; activeRevisionId: string }>;
    expect(activeFamilies).toHaveLength(1);
    expect(repos.proceduralTrajectory.listFamilyMembers(activeFamilies[0]!.activeRevisionId)
      .map((member) => member.episodeId)).toEqual([first.episodeId]);
    // Seed-stable Fine identity keeps the same cluster when the leftover
    // member is the original medoid; only the deactivated Path is gone.
    const remainingCluster = repos.proceduralTrajectory.getClusterHead(pairCluster.id);
    expect(remainingCluster?.status).toBe("active");
    expect(remainingCluster?.activeSkillMemoryId).toBeUndefined();
    const remainingVersion = repos.proceduralTrajectory.getClusterVersion(
      remainingCluster!.activeVersionId!
    )!;
    expect(remainingVersion.supportEpisodeCount).toBe(1);
    expect(repos.proceduralTrajectory.listClusterMembers(remainingVersion.id)
      .map((member) => member.episodeId)).toEqual([first.episodeId]);
    expect(service.listSkills({ userId: USER_ID }).items.map((item) => item.id))
      .not.toContain(skillMemoryId);
  });

  it("admits one Window to multiple Families and canonicalizes duplicate fine evidence", async () => {
    const operations: string[] = [];
    const llm = proceduralE2eLlm(operations);
    const config = {
      ...DEFAULT_MEMMY_CONFIG,
      algorithm: {
        ...DEFAULT_MEMMY_CONFIG.algorithm,
        negativeExperience: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.negativeExperience,
          enabled: false
        },
        l2Induction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l2Induction,
          useLlm: false
        },
        l3Abstraction: {
          ...DEFAULT_MEMMY_CONFIG.algorithm.l3Abstraction,
          useLlm: false
        }
      }
    };
    const { db, service } = createTestService({
      config,
      llm,
      skillLlm: llm,
      embedder: overlappingFamilyEmbedder()
    });
    const repos = new Repositories(db.db);

    const anchorX = await executeSuccessfulEpisode(service, "codex", "family-x", "x");
    await runWorkerRounds(service, 16);
    const anchorY = await executeSuccessfulEpisode(service, "cursor", "family-y", "y");
    await runWorkerRounds(service, 16);

    const familiesBeforeLocalUpdate = db.db.prepare(
      `SELECT id, active_revision_id AS activeRevisionId
       FROM trajectory_window_families
       WHERE user_id = ? AND scale = 5 AND status = 'active'
       ORDER BY id ASC`
    ).all(USER_ID) as Array<{ id: string; activeRevisionId: string }>;
    expect(familiesBeforeLocalUpdate).toHaveLength(2);
    const familyForEpisode = (episodeId: string) => familiesBeforeLocalUpdate.find((family) =>
      repos.proceduralTrajectory.listFamilyMembers(family.activeRevisionId)
        .some((member) => member.episodeId === episodeId))!;
    const familyXBefore = familyForEpisode(anchorX.episodeId);
    const familyYBefore = familyForEpisode(anchorY.episodeId);
    expect(familyXBefore.id).not.toBe(familyYBefore.id);

    // "u" shares X's coarse vector but has a Fine-orthogonal Step sequence:
    // only Family X should receive a new revision, and no Skill is eligible.
    const localX = await executeSuccessfulEpisode(service, "claude_code", "family-local-x", "u");
    await runWorkerRounds(service, 18);
    const familiesAfterLocalUpdate = db.db.prepare(
      `SELECT id, active_revision_id AS activeRevisionId
       FROM trajectory_window_families
       WHERE user_id = ? AND scale = 5 AND status = 'active'
       ORDER BY id ASC`
    ).all(USER_ID) as Array<{ id: string; activeRevisionId: string }>;
    const familyXAfter = familiesAfterLocalUpdate.find((family) =>
      family.id === familyXBefore.id)!;
    const familyYAfter = familiesAfterLocalUpdate.find((family) =>
      family.id === familyYBefore.id)!;
    expect(familyXAfter.activeRevisionId).not.toBe(familyXBefore.activeRevisionId);
    expect(familyYAfter.activeRevisionId).toBe(familyYBefore.activeRevisionId);
    expect(repos.proceduralTrajectory.listFamilyMembers(familyXAfter.activeRevisionId)
      .map((member) => member.episodeId)).toContain(localX.episodeId);
    expect(operations.filter((operation) =>
      operation === "procedural.procedural-pattern-skill.v12")).toHaveLength(0);

    const sharedA = await executeSuccessfulEpisode(service, "codex", "shared-a", "z");
    await runWorkerRounds(service, 16);
    const sharedB = await executeSuccessfulEpisode(service, "cursor", "shared-b", "z");
    await runWorkerRounds(service, 24);

    const families = db.db.prepare(
      `SELECT id, active_revision_id AS activeRevisionId
       FROM trajectory_window_families
       WHERE user_id = ? AND scale = 5 AND status = 'active'
       ORDER BY id ASC`
    ).all(USER_ID) as Array<{ id: string; activeRevisionId: string }>;
    expect(families).toHaveLength(2);
    const sharedWindowIds = [sharedA, sharedB].map((episode) => {
      const path = repos.proceduralTrajectory.getActivePath(episode.episodeId)!;
      return repos.proceduralTrajectory.listWindowsForPath(path.id, 5)[0]!.id;
    });
    for (const family of families) {
      const memberIds = repos.proceduralTrajectory.listFamilyMembers(family.activeRevisionId)
        .map((member) => member.occurrenceId);
      expect(memberIds).toEqual(expect.arrayContaining(sharedWindowIds));
    }

    const sharedClusters = db.db.prepare(
      `SELECT clusters.id AS id
       FROM trajectory_window_clusters AS clusters
       JOIN trajectory_window_cluster_versions AS versions
         ON versions.id = clusters.active_version_id
       WHERE clusters.user_id = ? AND clusters.scale = 5
         AND clusters.status = 'active' AND versions.support_episode_count = 2`
    ).all(USER_ID) as Array<{ id: string }>;
    expect(sharedClusters).toHaveLength(1);
    const activeFamilyLinks = repos.proceduralTrajectory
      .listClusterFamilyLinks(sharedClusters[0]!.id)
      .filter((link) => families.some((family) =>
        family.activeRevisionId === link.familyRevisionId));
    expect(activeFamilyLinks).toHaveLength(2);

    const sharedCluster = repos.proceduralTrajectory.getClusterHead(sharedClusters[0]!.id)!;
    expect(sharedCluster.activeSkillMemoryId).toBeTruthy();
    expect(operations.filter((operation) =>
      operation === "procedural.procedural-pattern-skill.v12")).toHaveLength(1);
    expect(service.listSkills({ userId: USER_ID }).items.map((item) => item.id))
      .toEqual([sharedCluster.activeSkillMemoryId]);
  });
});

async function executeSuccessfulEpisode(
  service: ReturnType<typeof createTestService>["service"],
  source: string,
  suffix: string,
  toolPrefix = ""
) {
  const namespace: RuntimeNamespace = {
    source,
    profileId: `profile-${suffix}`,
    userId: USER_ID
  };
  const session = service.openSession({ namespace });
  const toolNames = ["inspect_schema", "prepare_inputs", "build_artifact", "run_checks", "verify_output"]
    .map((name) => toolPrefix ? `${toolPrefix}_${name}` : name);
  const completed = service.completeTurn(`turn-${suffix}`, {
    namespace,
    sessionId: session.sessionId,
    query: `Build and validate artifact ${suffix}`,
    answer: `Artifact ${suffix} was built and validation passed.`,
    toolCalls: toolNames.map((name, index) => ({
      id: `${suffix}-tool-${index}`,
      name,
      input: { target: `artifact-${suffix}`, stage: index }
    })),
    toolResults: toolNames.map((name, index) => ({
      toolCallId: `${suffix}-tool-${index}`,
      name,
      output: { ok: true, stage: index }
    }))
  });
  service.closeSession(session.sessionId);
  await service.feedback({
    namespace,
    sessionId: session.sessionId,
    episodeId: completed.episodeId,
    l1MemoryId: completed.l1MemoryId,
    channel: "explicit",
    polarity: "positive",
    magnitude: 1,
    rationale: "the artifact and validation are correct"
  });
  return {
    namespace,
    sessionId: session.sessionId,
    episodeId: completed.episodeId,
    l1MemoryId: completed.l1MemoryId
  };
}

async function executeLongSuccessfulEpisode(
  service: ReturnType<typeof createTestService>["service"],
  source: string,
  suffix: string
) {
  const namespace: RuntimeNamespace = {
    source,
    profileId: `profile-${suffix}`,
    userId: USER_ID
  };
  const session = service.openSession({ namespace });
  const toolNames = Array.from({ length: 20 }, (_, index) =>
    `long_stage_${String(index + 1).padStart(2, "0")}`);
  const completed = service.completeTurn(`turn-${suffix}`, {
    namespace,
    sessionId: session.sessionId,
    query: "Build, repair, and validate a complex document artifact",
    answer: "The complex artifact was built and passed all validation checks.",
    toolCalls: toolNames.map((name, index) => ({
      id: `${suffix}-tool-${index}`,
      name,
      input: { artifactType: "document", stage: index }
    })),
    toolResults: toolNames.map((name, index) => ({
      toolCallId: `${suffix}-tool-${index}`,
      name,
      output: { ok: true, stage: index }
    }))
  });
  service.closeSession(session.sessionId);
  await service.feedback({
    namespace,
    sessionId: session.sessionId,
    episodeId: completed.episodeId,
    l1MemoryId: completed.l1MemoryId,
    channel: "explicit",
    polarity: "positive",
    magnitude: 1,
    rationale: "the long artifact workflow succeeded"
  });
  return {
    namespace,
    sessionId: session.sessionId,
    episodeId: completed.episodeId,
    l1MemoryId: completed.l1MemoryId
  };
}

function constantProceduralEmbedder(): Embedder {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "constant-procedural-test"
    },
    isRemote: () => false,
    embed: async (texts) => texts.map(() => [1, 0, 0, 0]),
    embedOne: async () => [1, 0, 0, 0],
    status: () => ({
      provider: "local",
      model: "constant-procedural-test",
      configured: true,
      remote: false
    })
  };
}

function overlappingFamilyEmbedder(): Embedder {
  const angle = 38 * Math.PI / 180;
  const coarse = {
    x: [Math.cos(angle), Math.sin(angle), 0, 0],
    y: [Math.cos(angle), -Math.sin(angle), 0, 0],
    z: [1, 0, 0, 0],
    u: [Math.cos(angle), Math.sin(angle), 0, 0]
  } as const;
  const fine = {
    x: [1, 0, 0, 0],
    y: [0, 1, 0, 0],
    z: [0, 0, 1, 0],
    u: [0, 0, 0, 1]
  } as const;
  const vectorFor = (text: string): number[] => {
    const kind = text.includes("stage x_") ? "x"
      : text.includes("stage y_") ? "y"
      : text.includes("stage u_") ? "u"
      : "z";
    return [...(/^1\.\s/.test(text) ? coarse[kind] : fine[kind])];
  };
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "overlapping-family-test"
    },
    isRemote: () => false,
    embed: async (texts) => texts.map(vectorFor),
    embedOne: async (text) => vectorFor(text),
    status: () => ({
      provider: "local",
      model: "overlapping-family-test",
      configured: true,
      remote: false
    })
  };
}

function proceduralE2eLlm(
  operations: string[],
  clientOptions: {
    model?: string;
    semanticSource?: string | (() => string);
    skillPayloads?: Array<Record<string, unknown>>;
    beforeSkillResponse?(originKind: string | undefined): Promise<void>;
    skillResponse?(
      originKind: string | undefined,
      payload: Record<string, unknown>,
      response: Record<string, unknown>
    ): Record<string, unknown>;
  } = {}
): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      model: clientOptions.model ?? "procedural-e2e"
    },
    isConfigured: () => true,
    complete: async () => "{}",
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      operations.push(options.operation);
      const content = messages.at(-1)?.content ?? "{}";
      if (options.operation === "capture.reflection.batch.v13") {
        const payload = JSON.parse(content) as { steps?: Array<{ idx: number }> };
        return {
          scores: (payload.steps ?? []).map(({ idx }) => ({
            idx,
            relevance: "PIVOTAL",
            reason: "successful multi-step execution"
          }))
        } as unknown as T;
      }
      if (options.operation === "capture.summarize") {
        const quote = content.match(/\bUSER:\s*(.*?)\s+ASSISTANT:/)?.[1]?.trim() ??
          "Build and validate artifact";
        return {
          l1: {
            summary: "Built the requested artifact and verified the output",
            evidence: [{ quote, role: "user", kind: "task_outcome" }]
          },
          user: null
        } as unknown as T;
      }
      if (options.operation === "reward.reward.r_human.v7") {
        const negative = content.includes("regression invalidated this result");
        const score = negative ? -1 : 1;
        return {
          goal_achievement: score,
          process_quality: score,
          user_satisfaction: score,
          reason: negative ? "latest explicit feedback invalidated the result" : "task succeeded"
        } as unknown as T;
      }
      if (isStepSemanticsOperation(options.operation)) {
        const payload = JSON.parse(content) as {
          stepCandidates?: Array<{
            candidateId: string;
            kind?: "tool_action" | "response_generation";
            toolName?: string;
          }>;
          candidates?: Array<{
            candidate_id: string;
            kind?: "tool_action" | "response_generation";
            tool_name?: string;
          }>;
        };
        const candidates = payload.stepCandidates?.map((candidate) => ({
          candidateId: candidate.candidateId,
          kind: candidate.kind,
          toolName: candidate.toolName
        })) ?? payload.candidates?.map((candidate) => ({
          candidateId: candidate.candidate_id,
          kind: candidate.kind,
          toolName: candidate.tool_name
        })) ?? [];
        return {
          steps: candidates.map((candidate) => ({
            candidate_id: candidate.candidateId,
            include: candidate.kind !== "response_generation",
            intent: `[${typeof clientOptions.semanticSource === "function"
              ? clientOptions.semanticSource()
              : clientOptions.semanticSource ?? "procedural-e2e"}] Complete reusable stage ${candidate.toolName ?? "tool"}`,
            summary: `${candidate.toolName ?? "tool"} completed successfully`
          }))
        } as unknown as T;
      }
      if (options.operation === "procedural.procedural-skill-coverage.v1") {
        return {
          decision: "distinct",
          target_skill_id: null,
          target_route: null,
          relation: "distinct",
          reason: "No existing Skill covers this Draft."
        } as unknown as T;
      }
      if (options.operation === "procedural.procedural-pattern-skill.v12" ||
          options.operation === "procedural.procedural-long-trajectory-skill.v3") {
        const payload = JSON.parse(content) as {
          pattern?: { origin?: { kind?: string } };
          occurrences?: Array<{
            occurrence_id: string;
            episode_id: string;
            aligned_sequence?: Array<{
              anchor_id?: string;
              step_id: string;
              outcome?: string;
            }>;
          }>;
          common_core?: { anchors?: Array<{ anchor_id: string }> };
          evidence_anchor_catalog?: Array<{
            anchor_id: string;
            allowed_usage: "mandatory" | "conditional_only";
            semantic_parent_anchor_id?: string;
            support_episode_ids: string[];
          }>;
        };
        clientOptions.skillPayloads?.push(payload as Record<string, unknown>);
        await clientOptions.beforeSkillResponse?.(payload.pattern?.origin?.kind);
        const mandatoryAnchors = (payload.evidence_anchor_catalog ?? [])
          .filter((anchor) => anchor.allowed_usage === "mandatory");
        const anchorEvidence = (anchorId: string) => mandatoryAnchors
          .filter((anchor) => anchor.anchor_id === anchorId ||
            anchor.semantic_parent_anchor_id === anchorId)
          .map((anchor) => anchor.anchor_id);
        const sharedAnchors = mandatoryAnchors
          .filter((anchor) => anchor.support_episode_ids.length >= 2)
          .map((anchor) => anchor.anchor_id);
        const closureSupportEpisodeIds = [...new Set(mandatoryAnchors.flatMap((anchor) =>
          anchor.support_episode_ids))];
        const longProcedure = payload.pattern?.origin?.kind === "long_trajectory"
          ? (payload.common_core?.anchors ?? []).map((anchor, index) => ({
              title: `Complete repeated Span ${index + 1}`,
              body: "Complete this repeated segment of the long workflow.",
              source_anchor_ids: anchorEvidence(anchor.anchor_id)
            }))
          : [{
              title: "Build through the validated stages",
              body: "Run the shared ordered construction stages.",
              source_anchor_ids: sharedAnchors
            }];
        const response = {
          admit: true,
          rejection_reason: null,
          ...(payload.pattern?.origin?.kind === "long_trajectory" ? {} : {
            local_subproblem_closure: {
              closed: true,
              subproblem: "Build and validate a requested local artifact stage.",
              entry_condition: "The artifact stage has not been completed.",
              resolution: "Run the shared ordered construction actions.",
              resolved_state: "The artifact stage is complete.",
              success_check: "The shared output verification succeeds.",
              support_episode_ids: closureSupportEpisodeIds,
              reason: "The same successful Episodes support construction and verification."
            }
          }),
          name: "build-and-verify-artifact",
          display_title: "Build and verify an artifact",
          retrieval_blurb: "Build an artifact through ordered stages and verify the final output.",
          trigger_context: "Use when an artifact must be built and validated through the same repeatable stages.",
          summary: "Inspect, prepare, build, check, and verify the requested artifact.",
          preconditions: ["The target artifact and expected validation signal are known."],
          parameters: [],
          procedure_steps: longProcedure,
          verification_steps: [{
            check: "Verify the generated artifact",
            success_signal: "The final output check succeeds.",
            source_anchor_ids: payload.pattern?.origin?.kind === "long_trajectory"
              ? anchorEvidence(payload.common_core?.anchors?.at(-1)?.anchor_id ?? "")
              : sharedAnchors.slice(-1)
          }],
          do_not_apply_when: [],
          decision_guidance: { preference: [], anti_pattern: [] },
          examples: [],
          tags: ["artifact", "verification"],
          tools: [],
          confidence: 0.9
        };
        return (clientOptions.skillResponse?.(
          payload.pattern?.origin?.kind,
          payload as Record<string, unknown>,
          response
        ) ?? response) as T;
      }
      return {} as T;
    },
    status: () => ({
      provider: "host",
      model: clientOptions.model ?? "procedural-e2e",
      configured: true,
      remote: true
    })
  };
}

function isStepSemanticsOperation(operation: string): boolean {
  return operation.startsWith(STEP_SEMANTICS_OPERATION_PREFIX);
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
