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
    expect(service.listSkills({ userId: USER_ID }).items.map((item) => item.id))
      .toContain(skillMemoryId);
    expect(service.getSkill(skillMemoryId!, { namespace: first.namespace })).toMatchObject({
      id: skillMemoryId,
      sourcePolicyIds: []
    });
    expect(operations.filter(isStepSemanticsOperation)).toHaveLength(2);
    expect(operations).toContain("procedural.procedural-pattern-skill.v2");

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
    expect(repos.proceduralTrajectory.getClusterHead(pairCluster.id)?.status).toBe("retired");
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
      operation === "procedural.procedural-pattern-skill.v2")).toHaveLength(0);

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
      operation === "procedural.procedural-pattern-skill.v2")).toHaveLength(1);
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
    semanticSource?: string;
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
            intent: `[${clientOptions.semanticSource ?? "procedural-e2e"}] Complete reusable stage ${candidate.toolName ?? "tool"}`,
            summary: `${candidate.toolName ?? "tool"} completed successfully`
          }))
        } as unknown as T;
      }
      if (options.operation === "procedural.procedural-pattern-skill.v2") {
        const payload = JSON.parse(content) as {
          occurrences?: Array<{ occurrence_id: string }>;
        };
        const occurrenceIds = (payload.occurrences ?? []).map((item) => item.occurrence_id);
        return {
          admit: true,
          rejection_reason: null,
          name: "build-and-verify-artifact",
          display_title: "Build and verify an artifact",
          retrieval_blurb: "Build an artifact through ordered stages and verify the final output.",
          trigger_context: "Use when an artifact must be built and validated through the same repeatable stages.",
          summary: "Inspect, prepare, build, check, and verify the requested artifact.",
          preconditions: ["The target artifact and expected validation signal are known."],
          parameters: [],
          procedure_steps: [{
            title: "Build through the validated stages",
            body: "Run the shared ordered construction stages.",
            evidence_refs: occurrenceIds
          }],
          verification_steps: [{
            check: "Verify the generated artifact",
            success_signal: "The final output check succeeds.",
            evidence_refs: occurrenceIds
          }],
          do_not_apply_when: [],
          decision_guidance: { preference: [], anti_pattern: [] },
          examples: [],
          tags: ["artifact", "verification"],
          tools: [],
          evidence_occurrence_ids: occurrenceIds,
          confidence: 0.9
        } as unknown as T;
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
