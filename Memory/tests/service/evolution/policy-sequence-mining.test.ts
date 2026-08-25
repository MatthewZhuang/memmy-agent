import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  EpisodePolicyProjectionPipeline,
  PolicySequenceMiningPipeline,
  PROCEDURAL_SEQUENCE_SKILL_OPERATION,
  PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
  applyStateDelta,
  buildEpisodePolicyProjection,
  buildEpisodeProceduralPath,
  buildProceduralPolicy,
  createEmbedder,
  createLlmClient,
  emptyObservedState,
  extractPolicySequencePatternOccurrences,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type EpisodePolicyProjectionV1,
  type Embedder,
  type ExecutionStepV1,
  type LlmClient,
  type LlmMessage,
  type MemoryRow,
  type ProceduralPolicyV1,
  type ProceduralSpanV1,
  type SpanSegmentationDecisionV1
} from "../../../src/index.js";
import { Repositories } from "../../../src/storage/repositories.js";
import {
  createCapturingEmbedder,
  createMemoryServiceFixture,
  runWorkerRounds
} from "../../fixtures/memory-service-fixture.js";

const NAMESPACE = "user-sequence";
const { cleanup, createTestService } = createMemoryServiceFixture();

afterEach(cleanup);

describe("dual-entry Policy sequence mining v2", () => {
  it("treats UNMAPPED as a hard boundary and retains stable Policy keys plus version snapshots", () => {
    const projection = projectionFixture([
      { key: "policy:a", version: "policy:a:v1" },
      { key: "policy:b", version: "policy:b:v3" },
      null,
      { key: "policy:c", version: "policy:c:v2" },
      { key: "policy:d", version: "policy:d:v7" }
    ]);
    const costs = new Map(projection.nodes.map((node) => [node.occurrenceId, {
      steps: 2,
      toolCalls: 1,
      retryCount: 0,
      recoveryCount: 0,
      errorCount: 0
    }]));

    const occurrences = extractPolicySequencePatternOccurrences({
      projection,
      namespaceId: NAMESPACE,
      sessionId: "session-model",
      terminalReward: 1,
      spanCostsByOccurrenceId: costs
    });

    expect(occurrences.map((item) => item.policyKeys)).toEqual([
      ["policy:a", "policy:b"],
      ["policy:c", "policy:d"]
    ]);
    expect(occurrences.map((item) => item.policyVersionIds)).toEqual([
      ["policy:a:v1", "policy:b:v3"],
      ["policy:c:v2", "policy:d:v7"]
    ]);
    expect(occurrences.every((item) => item.evidenceRole === "support")).toBe(true);
    expect(occurrences[0]?.cost).toMatchObject({ steps: 4, toolCalls: 2 });
  });

  it("filters all-same Policy sequences while retaining windows with a real Policy transition", () => {
    const projection = projectionFixture([
      { key: "policy:a", version: "policy:a:v1" },
      { key: "policy:a", version: "policy:a:v2" },
      { key: "policy:a", version: "policy:a:v3" },
      { key: "policy:b", version: "policy:b:v1" }
    ]);
    const costs = new Map(projection.nodes.map((node) => [node.occurrenceId, {
      steps: 1,
      toolCalls: 1,
      retryCount: 0,
      recoveryCount: 0,
      errorCount: 0
    }]));

    const occurrences = extractPolicySequencePatternOccurrences({
      projection,
      namespaceId: NAMESPACE,
      sessionId: "session-homogeneous-filter",
      terminalReward: 1,
      spanCostsByOccurrenceId: costs
    });

    expect(occurrences.map((item) => item.policyKeys)).toEqual([
      ["policy:a", "policy:a", "policy:a", "policy:b"],
      ["policy:a", "policy:a", "policy:b"],
      ["policy:a", "policy:b"]
    ]);
    expect(occurrences.every((item) => new Set(item.policyKeys).size >= 2)).toBe(true);
  });

  it("does not persist a Pattern or Candidate for repeated all-same Policy sequences", async () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const pipeline = new PolicySequenceMiningPipeline({
      repos,
      embedder: createCapturingEmbedder([])
    });
    for (const [index, suffix] of ["same-policy-a", "same-policy-b"].entries()) {
      const projection = persistSequenceProjection(
        repos,
        suffix,
        ["policy:repeat", "policy:repeat"],
        1
      );
      await pipeline.mineProjection(projection.id, `2026-08-20T00:0${index}:00.000Z`);
    }

    expect(repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      ["policy:repeat", "policy:repeat"]
    )).toBeUndefined();
    expect(repos.policySequencePatterns.listActiveCandidatesForNamespace(NAMESPACE)).toEqual([]);
  });

  it("promotes only closed exact sequences to ready and retains counterexamples", async () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const pipeline = new PolicySequenceMiningPipeline({
      repos,
      embedder: createCapturingEmbedder([])
    });
    const first = persistSequenceProjection(repos, "support-a", ["policy:a", "policy:b", "policy:c"], 1);
    await pipeline.mineProjection(first.id, "2026-08-20T01:00:00.000Z");
    const second = persistSequenceProjection(repos, "support-b", ["policy:a", "policy:b", "policy:c"], 1);
    await pipeline.mineProjection(second.id, "2026-08-20T01:01:00.000Z");

    const pair = repos.policySequencePatterns.getPatternBySequence(NAMESPACE, ["policy:a", "policy:b"]);
    const triple = repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      ["policy:a", "policy:b", "policy:c"]
    );
    expect(pair).toMatchObject({
      lifecycleStatus: "ready",
      distinctSupportEpisodeCount: 2,
      isClosed: false,
      isMaximal: false
    });
    expect(repos.policySequencePatterns.getActiveCandidateForPattern(pair!.id)).toBeUndefined();
    expect(triple).toMatchObject({
      lifecycleStatus: "ready",
      distinctSupportEpisodeCount: 2,
      isClosed: true,
      isMaximal: true
    });
    expect(repos.policySequencePatterns.getActiveCandidateForPattern(triple!.id)?.candidate)
      .toMatchObject({
        lifecycleStatus: "ready",
        capabilityType: "task_skill",
        discoverySources: ["episode_similarity", "policy_sequence_similarity"],
        policyKeys: ["policy:a", "policy:b", "policy:c"],
        supportEpisodeIds: ["episode-support-a", "episode-support-b"],
        provenance: { executable: false }
      });

    const third = persistSequenceProjection(repos, "support-c", ["policy:a", "policy:b", "policy:c"], 1);
    await pipeline.mineProjection(third.id, "2026-08-20T01:02:00.000Z");
    const ready = repos.policySequencePatterns.getActiveCandidateForPattern(triple!.id)!;
    expect(ready.lifecycleStatus).toBe("ready");
    expect(ready.distinctSupportEpisodeCount).toBe(3);
    expect(ready.candidate.aggregateSupportCost).toEqual({
      total: {
        steps: 9,
        toolCalls: 9,
        retryCount: 0,
        recoveryCount: 0,
        errorCount: 0
      },
      meanPerOccurrence: {
        steps: 3,
        toolCalls: 3,
        retryCount: 0,
        recoveryCount: 0,
        errorCount: 0
      }
    });
    expect(new Set(ready.candidate.evidence.flatMap((item) => item.policyVersionIds))).toEqual(
      new Set([
        "policy:a:version-support-a", "policy:b:version-support-a", "policy:c:version-support-a",
        "policy:a:version-support-b", "policy:b:version-support-b", "policy:c:version-support-b",
        "policy:a:version-support-c", "policy:b:version-support-c", "policy:c:version-support-c"
      ])
    );

    const counterexample = persistSequenceProjection(
      repos,
      "counterexample",
      ["policy:a", "policy:b", "policy:c"],
      0
    );
    await pipeline.mineProjection(counterexample.id, "2026-08-20T01:03:00.000Z");
    const withCounterexample = repos.policySequencePatterns.getActiveCandidateForPattern(triple!.id)!;
    expect(withCounterexample.lifecycleStatus).toBe("ready");
    expect(withCounterexample.distinctSupportEpisodeCount).toBe(3);
    expect(withCounterexample.distinctCounterexampleEpisodeCount).toBe(1);
    expect(withCounterexample.candidate.counterexampleEpisodeIds).toEqual([
      "episode-counterexample"
    ]);
    expect(repos.policySequencePatterns.listCandidatesForPattern(triple!.id))
      .toHaveLength(3);
  });

  it("supersedes the previous Episode evidence and versions the candidate when a Projection changes", async () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const pipeline = new PolicySequenceMiningPipeline({
      repos,
      embedder: createCapturingEmbedder([])
    });
    for (const [index, suffix] of ["version-a", "version-b", "version-c"].entries()) {
      const projection = persistSequenceProjection(
        repos,
        suffix,
        ["policy:a", "policy:b", "policy:c"],
        1
      );
      await pipeline.mineProjection(projection.id, `2026-08-20T02:0${index}:00.000Z`);
    }
    const triple = repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      ["policy:a", "policy:b", "policy:c"]
    )!;
    const ready = repos.policySequencePatterns.getActiveCandidateForPattern(triple.id)!;
    const replacement = replaceProjection(
      repos,
      "version-c",
      ["policy:a", "policy:x", "policy:c"],
      "2026-08-20T02:03:00.000Z"
    );

    await pipeline.mineProjection(replacement.id, "2026-08-20T02:04:00.000Z");

    const downgraded = repos.policySequencePatterns.getPattern(triple.id)!;
    const active = repos.policySequencePatterns.getActiveCandidateForPattern(triple.id)!;
    expect(downgraded).toMatchObject({
      lifecycleStatus: "ready",
      occurrenceCount: 2,
      distinctSupportEpisodeCount: 2
    });
    expect(active.id).not.toBe(ready.id);
    expect(active.lifecycleStatus).toBe("ready");
    expect(repos.policySequencePatterns.listCandidatesForPattern(triple.id))
      .toEqual(expect.arrayContaining([
        expect.objectContaining({ id: ready.id, status: "inactive" }),
        expect.objectContaining({ id: active.id, status: "active" })
      ]));
  });

  it("refreshes only the Pattern neighborhood affected by the ingested Episode", async () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const pipeline = new PolicySequenceMiningPipeline({
      repos,
      embedder: createCapturingEmbedder([])
    });
    for (const [index, suffix] of ["unrelated-a", "unrelated-b"].entries()) {
      const projection = persistSequenceProjection(
        repos,
        suffix,
        ["policy:u", "policy:v", "policy:w"],
        1
      );
      await pipeline.mineProjection(projection.id, `2026-08-20T02:1${index}:00.000Z`);
    }
    const firstTarget = persistSequenceProjection(
      repos,
      "incremental-target-a",
      ["policy:a", "policy:b", "policy:c"],
      1
    );
    await pipeline.mineProjection(firstTarget.id, "2026-08-20T02:20:00.000Z");

    const unrelatedPattern = repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      ["policy:u", "policy:v", "policy:w"]
    )!;
    const unrelatedCandidate = repos.policySequencePatterns
      .getActiveCandidateForPattern(unrelatedPattern.id)!;
    const secondTarget = persistSequenceProjection(
      repos,
      "incremental-target-b",
      ["policy:a", "policy:b", "policy:c"],
      1
    );

    const result = await pipeline.mineProjection(
      secondTarget.id,
      "2026-08-20T02:21:00.000Z"
    );

    expect(result.affectedPatternIds).toEqual(result.patterns.map((pattern) => pattern.id));
    expect(result.affectedPatternIds).not.toContain(unrelatedPattern.id);
    expect(result.activeCandidates.map((candidate) => candidate.patternId))
      .not.toContain(unrelatedPattern.id);
    expect(repos.policySequencePatterns.getPattern(unrelatedPattern.id)).toMatchObject({
      updatedAt: unrelatedPattern.updatedAt,
      membershipVersion: unrelatedPattern.membershipVersion,
      distinctSupportEpisodeCount: 2
    });
    expect(repos.policySequencePatterns.getActiveCandidateForPattern(unrelatedPattern.id))
      .toMatchObject({
        id: unrelatedCandidate.id,
        activatedAt: unrelatedCandidate.activatedAt,
        evidenceHash: unrelatedCandidate.evidenceHash
      });
    expect(repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      ["policy:a", "policy:b", "policy:c"]
    )).toMatchObject({
      lifecycleStatus: "ready",
      distinctSupportEpisodeCount: 2,
      isClosed: true
    });
  });

  it("discovers a task Skill from similar Episodes whose shared Policy backbone contains gaps", async () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const pipeline = new PolicySequenceMiningPipeline({
      repos,
      embedder: createCapturingEmbedder([])
    });
    const longPath = persistSequenceProjection(
      repos,
      "episode-first-long",
      ["policy:inspect", "policy:prepare", "policy:detour-a", "policy:detour-b", "policy:verify"],
      1
    );
    const shortPath = persistSequenceProjection(
      repos,
      "episode-first-short",
      ["policy:orient", "policy:prepare", "policy:verify", "policy:report"],
      1
    );

    await pipeline.mineProjection(longPath.id, "2026-08-20T02:10:00.000Z");
    await pipeline.mineProjection(shortPath.id, "2026-08-20T02:11:00.000Z");

    const longSignature = repos.episodeCapabilities.getActiveForEpisode(
      "episode-episode-first-long"
    )!;
    const shortSignature = repos.episodeCapabilities.getActiveForEpisode(
      "episode-episode-first-short"
    )!;
    expect(shortSignature.familyId).toBe(longSignature.familyId);
    expect(repos.episodeCapabilities.listAffinitiesForSignature(shortSignature.signature.id)[0])
      .toMatchObject({ affinity: { familyEligible: true } });

    const taskPattern = repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      ["policy:prepare", "policy:verify"],
      { capabilityType: "task_skill", episodeFamilyId: shortSignature.familyId }
    )!;
    const taskCandidate = repos.policySequencePatterns.getActiveCandidateForPattern(taskPattern.id)!;
    expect(taskCandidate.candidate).toMatchObject({
      lifecycleStatus: "ready",
      capabilityType: "task_skill",
      discoverySources: ["episode_similarity"],
      episodeFamilyId: shortSignature.familyId,
      supportEpisodeIds: ["episode-episode-first-long", "episode-episode-first-short"]
    });
    expect(taskCandidate.candidate.evidence.map((item) => ({
      episodeId: item.episodeId,
      matched: item.spanOccurrenceIds.length,
      fullPath: item.pathSpanOccurrenceIds.length
    }))).toEqual([
      { episodeId: "episode-episode-first-long", matched: 2, fullPath: 4 },
      { episodeId: "episode-episode-first-short", matched: 2, fullPath: 2 }
    ]);
  });

  it("discovers a cross-Family sub-skill from a repeated local Policy sequence", async () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const pipeline = new PolicySequenceMiningPipeline({
      repos,
      embedder: differentFamilyEmbedder()
    });
    const first = persistSequenceProjection(
      repos,
      "policy-first-family-a",
      ["policy:runtime-check", "policy:runtime-fix"],
      1
    );
    const second = persistSequenceProjection(
      repos,
      "policy-first-family-b",
      ["policy:runtime-check", "policy:runtime-fix"],
      1
    );

    await pipeline.mineProjection(first.id, "2026-08-20T02:20:00.000Z");
    await pipeline.mineProjection(second.id, "2026-08-20T02:21:00.000Z");

    const firstSignature = repos.episodeCapabilities.getActiveForEpisode(
      "episode-policy-first-family-a"
    )!;
    const secondSignature = repos.episodeCapabilities.getActiveForEpisode(
      "episode-policy-first-family-b"
    )!;
    expect(firstSignature.familyId).not.toBe(secondSignature.familyId);
    expect(repos.episodeCapabilities.listAffinitiesForSignature(secondSignature.signature.id)[0])
      .toMatchObject({ affinity: { familyEligible: false } });

    const subSkillPattern = repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      ["policy:runtime-check", "policy:runtime-fix"]
    )!;
    expect(repos.policySequencePatterns.getActiveCandidateForPattern(subSkillPattern.id)?.candidate)
      .toMatchObject({
        lifecycleStatus: "ready",
        capabilityType: "sub_skill",
        discoverySources: ["policy_sequence_similarity"],
        supportEpisodeIds: ["episode-policy-first-family-a", "episode-policy-first-family-b"]
      });
  });

  it("compiles an Episode-first task Skill with the complete gap path", async () => {
    const llmCalls: Array<{ messages: LlmMessage[]; operation: string }> = [];
    const { db, service } = createTestService({
      skillLlm: sequenceSkillLlm(llmCalls),
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);
    persistSequenceProjection(
      repos,
      "task-skill-long",
      ["placeholder:inspect", "placeholder:prepare", "placeholder:detour-a",
        "placeholder:detour-b", "placeholder:verify"],
      1
    );
    persistSequenceProjection(
      repos,
      "task-skill-short",
      ["placeholder:orient", "placeholder:prepare", "placeholder:verify", "placeholder:report"],
      1
    );
    const paths = ["task-skill-long", "task-skill-short"].map((suffix) =>
      repos.proceduralPaths.getActiveForEpisode(`episode-${suffix}`)!
    );
    const policies = promoteBackbonePolicies(repos, paths, [[1, 1], [4, 2]]);
    const longProjection = saveProjection(
      repos,
      paths[0]!.id,
      "task-skill-long-final",
      ["policy:inspect-only", policies[0]!.policyKey, "policy:detour-a-only",
        "policy:detour-b-only", policies[1]!.policyKey],
      "2026-08-20T02:30:00.000Z",
      ["policy:inspect-only:v1", policies[0]!.id, "policy:detour-a-only:v1",
        "policy:detour-b-only:v1", policies[1]!.id]
    );
    const shortProjection = saveProjection(
      repos,
      paths[1]!.id,
      "task-skill-short-final",
      ["policy:orient-only", policies[0]!.policyKey, policies[1]!.policyKey, "policy:report-only"],
      "2026-08-20T02:31:00.000Z",
      ["policy:orient-only:v1", policies[0]!.id, policies[1]!.id, "policy:report-only:v1"]
    );
    for (const [index, projection] of [longProjection, shortProjection].entries()) {
      repos.runtime.enqueueJob({
        id: `job-task-skill-gap-${index}`,
        jobType: "policy_sequence_mining",
        status: "queued",
        dedupeKey: `policy_sequence_mining:${projection.id}`,
        userId: NAMESPACE,
        sessionId: paths[index]!.sessionId,
        episodeId: paths[index]!.episodeId,
        payload: {
          projectionId: projection.id,
          projectionHash: projection.projectionHash
        },
        attempts: 0,
        maxAttempts: 3,
        createdAt: `2026-08-20T02:3${index}:30.000Z`,
        updatedAt: `2026-08-20T02:3${index}:30.000Z`
      });
    }

    await runWorkerRounds(service, 4);

    expect(llmCalls.map((call) => call.operation)).toEqual([
      PROCEDURAL_SEQUENCE_SKILL_OPERATION
    ]);
    const payloadMessage = llmCalls[0]!.messages.find((message) =>
      message.role === "user" && message.content.includes("selected_path_evidence")
    );
    const payload = JSON.parse(payloadMessage!.content) as {
      candidate: { capability_type: string; discovery_sources: string[] };
      selected_path_evidence: Array<{
        episode_id: string;
        full_path: Array<{ role: string }>;
        gap_spans: unknown[];
      }>;
    };
    expect(payload.candidate).toMatchObject({
      capability_type: "task_skill",
      discovery_sources: ["episode_similarity"]
    });
    expect(payload.selected_path_evidence.map((item) => ({
      episodeId: item.episode_id,
      fullPath: item.full_path.length,
      gaps: item.gap_spans.length
    }))).toEqual([
      { episodeId: "episode-task-skill-long", fullPath: 4, gaps: 2 },
      { episodeId: "episode-task-skill-short", fullPath: 2, gaps: 0 }
    ]);
    const skill = service.listSkills({ userId: NAMESPACE }).skills[0]!;
    expect(repos.memories.get(skill.id)?.properties.internal_info.procedural_sequence_skill)
      .toMatchObject({
        executable: true,
        capability_type: "task_skill",
        discovery_sources: ["episode_similarity"]
      });
  });

  it("runs as a durable worker job", async () => {
    const { db, service } = createTestService();
    const repos = new Repositories(db.db);
    const projection = persistSequenceProjection(
      repos,
      "worker",
      ["policy:a", "policy:b", "policy:c"],
      1
    );
    repos.runtime.enqueueJob({
      id: "job-policy-sequence-worker",
      jobType: "policy_sequence_mining",
      status: "queued",
      dedupeKey: `policy_sequence_mining:${projection.id}`,
      userId: "user-sequence",
      sessionId: "session-worker",
      episodeId: "episode-worker",
      payload: {
        projectionId: projection.id,
        projectionHash: projection.projectionHash
      },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-20T03:00:00.000Z",
      updatedAt: "2026-08-20T03:00:00.000Z"
    });

    const run = await service.runWorkerOnce(10);

    expect(run.jobs).toContainEqual(expect.objectContaining({
      jobId: "job-policy-sequence-worker",
      jobType: "policy_sequence_mining",
      status: "succeeded"
    }));
    expect(repos.policySequencePatterns.listPatternsForNamespace(NAMESPACE)).toHaveLength(3);
  });

  it("compiles a ready cross-agent Policy sequence into a retrievable and callable Skill", async () => {
    const llmCalls: Array<{ messages: LlmMessage[]; operation: string }> = [];
    const embeddedTexts: string[] = [];
    const { db, service } = createTestService({
      skillLlm: sequenceSkillLlm(llmCalls),
      embedder: createCapturingEmbedder(embeddedTexts),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: true
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const suffixes = ["skill-a", "skill-b"];
    const evidenceScopes = [{
      source: "codex",
      profileId: "codex-profile",
      projectId: "project-a"
    }, {
      source: "cursor",
      profileId: "cursor-profile",
      projectId: "project-b"
    }];
    for (const [index, suffix] of suffixes.entries()) {
      persistSequenceProjection(
        repos,
        suffix,
        ["placeholder:a", "placeholder:b"],
        1,
        evidenceScopes[index]
      );
    }
    const paths = suffixes.map((suffix) =>
      repos.proceduralPaths.getActiveForEpisode(`episode-${suffix}`)!
    );
    const policies = promoteSequencePolicies(repos, paths);
    const projectionPipeline = new EpisodePolicyProjectionPipeline({ repos });
    const projections = paths.map((path, index) => projectionPipeline.projectPath(
      path.id,
      `2026-08-20T04:0${index}:00.000Z`
    ).record);
    expect(projections.every((projection) =>
      projection.projection.nodes.every((node) => node.assignment.kind === "policy")
    )).toBe(true);
    for (const [index, projection] of projections.entries()) {
      repos.runtime.enqueueJob({
        id: `job-policy-sequence-skill-${index}`,
        jobType: "policy_sequence_mining",
        status: "queued",
        dedupeKey: `policy_sequence_mining:${projection.id}`,
        userId: NAMESPACE,
        sessionId: projection.sessionId,
        episodeId: projection.episodeId,
        payload: {
          projectionId: projection.id,
          projectionHash: projection.projectionHash
        },
        attempts: 0,
        maxAttempts: 3,
        createdAt: `2026-08-20T04:1${index}:00.000Z`,
        updatedAt: `2026-08-20T04:1${index}:00.000Z`
      });
    }

    await runWorkerRounds(service, 4);

    const pattern = repos.policySequencePatterns.getPatternBySequence(
      NAMESPACE,
      policies.map((policy) => policy.policyKey)
    )!;
    const candidate = repos.policySequencePatterns.getActiveCandidateForPattern(pattern.id)!;
    expect(candidate).toMatchObject({
      lifecycleStatus: "ready",
      distinctSupportEpisodeCount: 2,
      namespaceId: NAMESPACE,
      candidate: {
        provenance: { executable: false }
      }
    });
    expect(llmCalls.map((call) => call.operation)).toEqual([
      PROCEDURAL_SEQUENCE_SKILL_OPERATION
    ]);
    expect(new Set(paths.map((path) => repos.runtime.getSession(path.sessionId)?.source))).toEqual(
      new Set(["codex", "cursor"])
    );

    const listed = service.listSkills({ userId: NAMESPACE });
    expect(listed.skills).toHaveLength(1);
    const skill = repos.memories.get(listed.skills[0]!.id)!;
    expect(skill).toMatchObject({
      userId: NAMESPACE,
      status: "resolving",
      memoryLayer: "Skill",
      agentId: undefined,
      appId: undefined
    });
    expect(skill.info.project_id).toBeUndefined();
    expect(skill.properties.internal_info).toMatchObject({
      procedural_sequence_skill: {
        candidate_id: candidate.id,
        pattern_id: pattern.id,
        evidence_hash: candidate.evidenceHash,
        capability_type: "task_skill",
        discovery_sources: ["episode_similarity", "policy_sequence_similarity"],
        executable: true
      },
      skill: {
        name: "staged_maintenance_workflow",
        status: "candidate",
        support: 2
      }
    });
    expect(service.getSkill(skill.id).invocationGuide).toContain("Staged maintenance workflow");
    expect(embeddedTexts.some((text) => text.includes("staged_maintenance_workflow"))).toBe(true);
    for (const suffix of suffixes) {
      expect(repos.runtime.getEpisode(`episode-${suffix}`)?.skillMemoryIds).toContain(skill.id);
    }

    const cursorNamespace = {
      source: "cursor",
      profileId: "cursor-profile",
      projectId: "project-b",
      userId: NAMESPACE,
      sessionKey: "cross-agent-skill-use"
    };
    expect(service.getSkill(skill.id, { namespace: cursorNamespace }).id).toBe(skill.id);
    const cursorSession = service.openSession({ namespace: cursorNamespace });
    const trialTurn = service.completeTurn("cross-agent-skill-turn", {
      sessionId: cursorSession.sessionId,
      episodeId: "episode-cross-agent-skill-use",
      query: "Apply the staged maintenance workflow",
      answer: "Applied the shared workflow"
    });
    expect(service.useSkill(skill.id, {
      adapterId: "cursor-test-adapter",
      requestId: "cross-agent-skill-use-1",
      sessionId: cursorSession.sessionId,
      episodeId: trialTurn.episodeId,
      rawTurnId: trialTurn.rawTurnId,
      turnId: trialTurn.turnId,
      namespace: cursorNamespace
    })).toMatchObject({ skillId: skill.id, status: "pending" });

    const downgradedProjection = replaceProjection(
      repos,
      "skill-b",
      [policies[0]!.policyKey, "policy:replacement-stage"],
      "2026-08-20T04:30:00.000Z"
    );
    repos.runtime.enqueueJob({
      id: "job-policy-sequence-skill-downgrade",
      jobType: "policy_sequence_mining",
      status: "queued",
      dedupeKey: `policy_sequence_mining:${downgradedProjection.id}`,
      userId: NAMESPACE,
      sessionId: "session-skill-b",
      episodeId: "episode-skill-b",
      payload: {
        projectionId: downgradedProjection.id,
        projectionHash: downgradedProjection.projectionHash
      },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-20T04:31:00.000Z",
      updatedAt: "2026-08-20T04:31:00.000Z"
    });
    await service.runWorkerOnce(10);
    expect(repos.policySequencePatterns.getPattern(pattern.id)).toMatchObject({
      lifecycleStatus: "forming",
      distinctSupportEpisodeCount: 1
    });
    expect(repos.memories.get(skill.id)).toMatchObject({
      status: "archived",
      properties: {
        internal_info: {
          procedural_sequence_skill: {
            executable: false,
            retired_reason: "pattern_not_ready"
          },
          skill: { status: "archived" }
        }
      }
    });
  });

  it("replays the production Trace2Skill downstream chain and reports both discovery routes", async () => {
    const llmCalls: Array<{ messages: LlmMessage[]; operation: string }> = [];
    const { db, service } = createTestService({
      skillLlm: sequenceSkillLlm(llmCalls),
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);
    for (const suffix of ["replay-a", "replay-b"]) {
      persistSequenceProjection(
        repos,
        suffix,
        ["placeholder:inspect", "placeholder:verify"],
        1
      );
    }
    const paths = ["replay-a", "replay-b"].map((suffix) =>
      repos.proceduralPaths.getActiveForEpisode(`episode-${suffix}`)!
    );
    promoteSequencePolicies(repos, paths);

    const first = await service.replayTrace2SkillEpisode({
      episodeId: "episode-replay-a",
      at: "2026-08-20T05:00:00.000Z"
    });
    expect(first.after).toMatchObject({
      checks: {
        activePath: true,
        pathContinuous: true,
        capabilitySignature: true,
        policySequenceRoute: true,
        readyCandidate: false,
        compiledSkill: false
      }
    });

    const second = await service.replayTrace2SkillEpisode({
      episodeId: "episode-replay-b",
      at: "2026-08-20T05:01:00.000Z"
    });
    expect(second.actions.compiledSkillIds).toHaveLength(1);
    expect(second.after.checks).toMatchObject({
      activePath: true,
      pathContinuous: true,
      capabilitySignature: true,
      episodeFamily: true,
      episodeSimilarityRoute: true,
      policySequenceRoute: true,
      readyCandidate: true,
      compiledSkill: true
    });
    expect(second.after.patterns).toContainEqual(expect.objectContaining({
      capabilityType: "sub_skill",
      occurrence: expect.objectContaining({
        discoverySources: ["episode_similarity", "policy_sequence_similarity"]
      }),
      candidate: expect.objectContaining({
        capabilityType: "task_skill",
        supportEpisodeIds: ["episode-replay-a", "episode-replay-b"]
      }),
      skill: expect.objectContaining({ executable: true })
    }));

    const repeated = await service.replayTrace2SkillEpisode({
      episodeId: "episode-replay-b",
      at: "2026-08-20T05:02:00.000Z"
    });
    expect(repeated.actions.compiledSkillIds).toEqual([]);
    expect(repeated.actions.reusedSkillIds).toEqual(second.actions.compiledSkillIds);
    expect(llmCalls.map((call) => call.operation)).toEqual([
      PROCEDURAL_SEQUENCE_SKILL_OPERATION
    ]);
  });

  it.skipIf(process.env.MEMMY_REAL_MODEL_TEST !== "1")(
    "compiles and embeds a replayed Skill with the configured real models",
    async () => {
      const loaded = loadMemmyConfig(process.env.MEMMY_CONFIG);
      const skillLlm = createLlmClient(resolveEvolutionConfig(loaded.config), {
        modelRole: "memory_evolution"
      });
      const embedder = createEmbedder(loaded.config.embedding);
      if (!skillLlm.isConfigured()) throw new Error("configured evolution LLM is unavailable");
      const { db, service } = createTestService({
        config: loaded.config,
        skillLlm,
        embedder
      });
      const repos = new Repositories(db.db);
      for (const suffix of ["real-replay-a", "real-replay-b"]) {
        persistSequenceProjection(
          repos,
          suffix,
          ["placeholder:inspect", "placeholder:verify"],
          1
        );
      }
      const paths = ["real-replay-a", "real-replay-b"].map((suffix) =>
        repos.proceduralPaths.getActiveForEpisode(`episode-${suffix}`)!
      );
      promoteSequencePolicies(repos, paths);

      await service.replayTrace2SkillEpisode({
        episodeId: "episode-real-replay-a",
        at: "2026-08-23T00:00:00.000Z"
      });
      const replay = await service.replayTrace2SkillEpisode({
        episodeId: "episode-real-replay-b",
        at: "2026-08-23T00:01:00.000Z"
      });
      expect(replay.actions.compiledSkillIds).toHaveLength(1);
      const skillId = replay.actions.compiledSkillIds[0]!;
      for (let round = 0; round < 5; round += 1) {
        const run = await service.runWorkerOnce(10, { targetMemoryIds: [skillId] });
        if (run.leased === 0 && run.embeddingRetries.leased === 0) break;
      }
      const after = service.inspectTrace2SkillEpisode("episode-real-replay-b");
      expect(after.checks).toMatchObject({
        episodeSimilarityRoute: true,
        policySequenceRoute: true,
        readyCandidate: true,
        compiledSkill: true
      });
      expect(service.getSkill(skillId).id).toBe(skillId);
      expect(service.listSkills({ userId: NAMESPACE }).skills.map((skill) => skill.id))
        .toContain(skillId);
      expect(repos.memories.hasVector(skillId, "vec")).toBe(true);
      expect(repos.memories.get(skillId)?.properties.internal_info)
        .toMatchObject({
          procedural_sequence_skill: {
            model: skillLlm.config.model,
            executable: true
          }
        });
    },
    240_000
  );
});

function projectionFixture(
  assignments: readonly ({ key: string; version: string } | null)[]
): EpisodePolicyProjectionV1 {
  return buildEpisodePolicyProjection({
    episodeId: "episode-model",
    pathId: "path-model",
    pathHash: "path-hash-model",
    nodes: assignments.map((assignment, spanIndex) => ({
      occurrenceId: `occurrence-model-${spanIndex}`,
      spanId: `span-model-${spanIndex}`,
      spanIndex,
      localGoal: `Complete stage ${spanIndex}`,
      entryCondition: `Stage ${spanIndex} is pending`,
      exitCondition: `Stage ${spanIndex} is complete`,
      terminationStatus: "success" as const,
      preStateId: `state-model-${spanIndex}`,
      postStateId: `state-model-${spanIndex + 1}`,
      rawTurnIds: ["raw-turn-model"],
      stepIds: [`step-model-${spanIndex}`],
      assignment: assignment ? {
        kind: "policy" as const,
        policyVersionId: assignment.version,
        policyKey: assignment.key,
        clusterId: `cluster-${assignment.key}`,
        clusterMembershipVersion: `membership-${assignment.version}`,
        evidenceRole: "support" as const,
        matchConfidence: 0.95
      } : {
        kind: "unmapped" as const,
        reason: "no_cluster_assignment" as const
      }
    }))
  });
}

function persistSequenceProjection(
  repos: Repositories,
  suffix: string,
  policyKeys: readonly string[],
  terminalReward: number,
  scope: {
    source: string;
    profileId: string;
    projectId: string;
  } = {
    source: "codex",
    profileId: "default",
    projectId: "project-a"
  }
): EpisodePolicyProjectionV1 {
  const at = "2026-08-20T00:00:00.000Z";
  const sessionId = `session-${suffix}`;
  const episodeId = `episode-${suffix}`;
  const rawTurnId = `raw-turn-${suffix}`;
  repos.runtime.createSession({
    id: sessionId,
    userId: "user-sequence",
    source: scope.source,
    profileId: scope.profileId,
    projectId: scope.projectId,
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
    userId: "user-sequence",
    projectId: scope.projectId,
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
    turnId: `host-turn-${suffix}`,
    userId: "user-sequence",
    userText: "Complete the staged maintenance workflow",
    assistantText: terminalReward > 0 ? "All stages passed" : "The workflow failed",
    toolCalls: [],
    toolResults: [],
    sourceMemoryIds: [],
    usage: {},
    status: terminalReward > 0 ? "succeeded" : "failed",
    createdAt: at
  });
  const states = [emptyObservedState()];
  for (let index = 0; index < policyKeys.length; index += 1) {
    states.push(applyStateDelta(states[index]!, [{
      op: "issue.upsert",
      subject: `stage ${index}`,
      status: "completed",
      sourceRefs: [`tool:${suffix}:${index}`]
    }]));
  }
  const provenance = {
    algorithmVersion: "episode-procedural-reconstruction.v2",
    model: "fixture-model",
    sourceSnapshotHash: `snapshot-${suffix}`
  };
  const steps: ExecutionStepV1[] = policyKeys.map((_, index) => ({
    id: `step-${suffix}-${index}`,
    schemaVersion: "execution-step.v1",
    episodeId,
    rawTurnId,
    turnIndex: 0,
    stepIndex: index,
    preStateId: states[index]!.id,
    action: {
      kind: "tool_action",
      type: "complete_stage",
      intent: `Complete workflow stage ${index}`,
      summary: `Stage ${index} completed`,
      eventRefs: [`tool:${suffix}:${index}`],
      toolName: "exec"
    },
    actionEffectDelta: [{
      op: "issue.upsert",
      subject: `stage ${index}`,
      status: "completed",
      sourceRefs: [`tool:${suffix}:${index}`]
    }],
    actionPostStateId: states[index + 1]!.id,
    externalObservationDelta: [],
    postStateId: states[index + 1]!.id,
    outcome: {
      status: "success",
      evidenceRefs: [`tool:${suffix}:${index}`]
    },
    cost: { toolCalls: 1, errorCount: 0 },
    provenance
  }));
  const spans: ProceduralSpanV1[] = steps.map((step, spanIndex) => ({
    id: `span-${suffix}-${spanIndex}`,
    schemaVersion: "procedural-span.v1",
    episodeId,
    spanIndex,
    localGoal: `Complete workflow stage ${spanIndex}`,
    capabilityGoal: `Complete reusable workflow stage ${spanIndex}`,
    entryCondition: `Stage ${spanIndex} is pending`,
    stepIds: [step.id],
    rawTurnIds: [rawTurnId],
    preStateId: step.preStateId,
    postStateId: step.postStateId,
    termination: {
      status: "success",
      exitCondition: `Stage ${spanIndex} is complete`,
      evidenceRefs: [...step.outcome.evidenceRefs]
    },
    cost: {
      steps: 1,
      toolCalls: 1,
      retryCount: 0,
      recoveryCount: 0,
      errorCount: 0
    },
    segmentation: {
      reason: "A bounded stage with an explicit completion state",
      confidence: 0.95
    },
    provenance
  }));
  const decisions: SpanSegmentationDecisionV1[] = spans.map((span) => ({
    spanIndex: span.spanIndex,
    stepIds: [...span.stepIds],
    localGoal: span.localGoal,
    capabilityGoal: span.capabilityGoal!,
    entryCondition: span.entryCondition,
    exitCondition: span.termination.exitCondition,
    terminationStatus: span.termination.status,
    evidenceRefs: [...span.termination.evidenceRefs],
    reason: span.segmentation.reason,
    confidence: span.segmentation.confidence
  }));
  const path = buildEpisodeProceduralPath({
    episodeId,
    states,
    steps,
    spans,
    segmentationDecisions: decisions,
    sourceSnapshotHash: provenance.sourceSnapshotHash,
    terminalReward
  });
  const saved = repos.proceduralPaths.save({
    path,
    namespaceId: NAMESPACE,
    createdAt: at
  });
  return saveProjection(repos, saved.record.id, suffix, policyKeys, at);
}

function replaceProjection(
  repos: Repositories,
  suffix: string,
  policyKeys: readonly string[],
  at: string
): EpisodePolicyProjectionV1 {
  const path = repos.proceduralPaths.getActiveForEpisode(`episode-${suffix}`)!;
  return saveProjection(repos, path.id, `${suffix}-replacement`, policyKeys, at);
}

function saveProjection(
  repos: Repositories,
  pathId: string,
  versionSuffix: string,
  policyKeys: readonly string[],
  at: string,
  policyVersionIds?: readonly string[]
): EpisodePolicyProjectionV1 {
  const path = repos.proceduralPaths.get(pathId)!;
  const occurrences = repos.proceduralPaths.listOccurrencesForPath(pathId);
  const projection = buildEpisodePolicyProjection({
    episodeId: path.episodeId,
    pathId: path.id,
    pathHash: path.pathHash,
    nodes: occurrences.map((occurrence, index) => ({
      occurrenceId: occurrence.id,
      spanId: occurrence.spanId,
      spanIndex: occurrence.spanIndex,
      localGoal: occurrence.localGoal,
      entryCondition: occurrence.entryCondition,
      exitCondition: occurrence.exitCondition,
      terminationStatus: occurrence.terminationStatus,
      preStateId: occurrence.preStateId,
      postStateId: occurrence.postStateId,
      rawTurnIds: [...occurrence.rawTurnIds],
      stepIds: [...occurrence.stepIds],
      assignment: {
        kind: "policy" as const,
        policyVersionId: policyVersionIds?.[index] ??
          `${policyKeys[index]}:version-${versionSuffix}`,
        policyKey: policyKeys[index]!,
        clusterId: `cluster-${policyKeys[index]}`,
        clusterMembershipVersion: `membership-${versionSuffix}-${index}`,
        evidenceRole: path.terminalReward && path.terminalReward > 0
          ? "support" as const
          : "counterexample" as const,
        matchConfidence: 0.95
      }
    }))
  });
  repos.episodePolicyProjections.saveAndActivate({
    projection,
    userId: path.userId,
    sessionId: path.sessionId,
    namespaceId: path.namespaceId,
    at
  });
  return projection;
}

function promoteSequencePolicies(
  repos: Repositories,
  paths: Array<NonNullable<ReturnType<Repositories["proceduralPaths"]["get"]>>>
): ProceduralPolicyV1[] {
  return [0, 1].map((stageIndex) => persistPolicyFixture({
    repos,
    paths,
    occurrenceIndexes: paths.map(() => stageIndex),
    fixtureId: `sequence-stage-${stageIndex}`,
    title: stageIndex === 0 ? "Prepare the maintenance target" : "Verify the maintenance result"
  }));
}

function promoteBackbonePolicies(
  repos: Repositories,
  paths: Array<NonNullable<ReturnType<Repositories["proceduralPaths"]["get"]>>>,
  occurrenceIndexesByStage: readonly (readonly number[])[]
): ProceduralPolicyV1[] {
  return occurrenceIndexesByStage.map((indexes, stageIndex) => persistPolicyFixture({
    repos,
    paths,
    occurrenceIndexes: indexes,
    fixtureId: `episode-backbone-stage-${stageIndex}`,
    title: stageIndex === 0 ? "Prepare the shared target" : "Verify the shared result"
  }));
}

function persistPolicyFixture(input: {
  repos: Repositories;
  paths: Array<NonNullable<ReturnType<Repositories["proceduralPaths"]["get"]>>>;
  occurrenceIndexes: readonly number[];
  fixtureId: string;
  title: string;
}): ProceduralPolicyV1 {
  const at = "2026-08-20T04:00:00.000Z";
  const occurrences = input.paths.map((path, index) =>
    input.repos.proceduralPaths.listOccurrencesForPath(path.id)[input.occurrenceIndexes[index]!]!
  );
  const cluster = input.repos.proceduralSpanClusters.upsert({
    id: `procedural-cluster-${input.fixtureId}`,
    namespaceId: NAMESPACE,
    algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
    minDistinctSupportEpisodes: 2,
    members: occurrences.map((occurrence) => ({
      occurrenceId: occurrence.id,
      evidenceRole: "support" as const,
      similarity: 0.95
    })),
    clusterBasis: { candidateSource: "test.dual-entry-sequence" },
    at
  });
  const policy = buildProceduralPolicy({
    namespaceId: NAMESPACE,
    clusterId: cluster.id,
    clusterMembershipVersion: cluster.membershipVersion,
    draft: {
      title: input.title,
      goalPattern: input.title,
      triggerConditions: [`${input.title} is pending`],
      procedureSteps: [{
        instruction: `${input.title} and record the resulting state.`,
        evidenceRefs: occurrences.map((occurrence) => occurrence.id)
      }],
      recoveryRules: [],
      verificationSteps: [{
        check: `Check that ${input.title.toLowerCase()} completed`,
        successSignal: `${input.title} is complete`,
        evidenceRefs: occurrences.map((occurrence) => occurrence.id)
      }],
      applyWhen: [`${input.title} is pending`],
      doNotApplyWhen: [`${input.title} is already complete`],
      invariants: ["Preserve the observed stage order"],
      expectedEffect: `${input.title} becomes complete`,
      evidenceOccurrenceIds: occurrences.map((occurrence) => occurrence.id),
      confidence: 0.92
    },
    occurrenceIds: occurrences.map((occurrence) => occurrence.id),
    supportOccurrenceIds: occurrences.map((occurrence) => occurrence.id),
    counterexampleOccurrenceIds: [],
    supportEpisodeIds: occurrences.map((occurrence) => occurrence.episodeId),
    counterexampleEpisodeIds: [],
    pathIds: occurrences.map((occurrence) => occurrence.pathId),
    spanIds: occurrences.map((occurrence) => occurrence.spanId),
    sessionIds: input.paths.map((path) => path.sessionId),
    model: "sequence-policy-fixture"
  });
  const l2MemoryId = `policy-memory-${input.fixtureId}`;
  input.repos.memories.insert(policyMemory(l2MemoryId, policy, at));
  input.repos.proceduralPolicies.saveAndActivate({
    policy,
    l2MemoryId,
    occurrences: occurrences.map((occurrence, index) => ({
      occurrenceId: occurrence.id,
      pathId: occurrence.pathId,
      spanId: occurrence.spanId,
      episodeId: occurrence.episodeId,
      sessionId: input.paths[index]!.sessionId,
      evidenceRole: "support" as const,
      matchConfidence: 0.95
    })),
    at
  });
  return policy;
}

function policyMemory(id: string, policy: ProceduralPolicyV1, at: string): MemoryRow {
  const tags = ["policy", "procedural"];
  const info = { title: policy.title, tags };
  return {
    id,
    timeline: at,
    userId: NAMESPACE,
    memoryType: "LongTermMemory",
    status: "resolving",
    visibility: "private",
    memoryKey: policy.policyKey,
    memoryValue: `# ${policy.title}\n\n${policy.expectedEffect}`,
    tags,
    info,
    properties: {
      memory_type: "LongTermMemory",
      status: "resolving",
      tags,
      info,
      internal_info: {
        memory_layer: "L2",
        memory_kind: "policy",
        schema_version: 1,
        policy: {
          title: policy.title,
          eta: policy.confidence,
          status: "candidate",
          support: policy.evidence.supportEpisodeIds.length,
          gain: 0,
          skill_eligible: false,
          source_episode_ids: policy.evidence.supportEpisodeIds
        }
      }
    },
    memoryLayer: "L2",
    contentHash: policy.contentHash,
    version: 1,
    createdAt: at,
    updatedAt: at,
    deletedAt: null
  };
}

function sequenceSkillLlm(calls: Array<{
  messages: LlmMessage[];
  operation: string;
}>): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/procedural-sequence-skill-test",
      model: "procedural-sequence-skill-test"
    },
    isConfigured() {
      return true;
    },
    async complete() {
      return "{}";
    },
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: { operation: string }
    ): Promise<T> {
      calls.push({ messages, operation: options.operation });
      const payloadMessage = messages.find((message) =>
        message.role === "user" && message.content.includes("selected_path_evidence"));
      const payload = JSON.parse(payloadMessage?.content ?? "{}") as {
        selected_path_evidence?: Array<{
          sequence_occurrence_id: string;
          evidence_role: "support" | "counterexample" | "uncertain";
          stages: Array<{ policy_version_id: string }>;
        }>;
      };
      const supports = (payload.selected_path_evidence ?? [])
        .filter((item) => item.evidence_role === "support");
      const stagePolicyIds = supports[0]!.stages.map((stage) => stage.policy_version_id);
      return {
        name: "staged_maintenance_workflow",
        display_title: "Staged maintenance workflow",
        retrieval_blurb: "Use this Skill to complete a repeated two-stage maintenance workflow.",
        trigger_context: "Use when the same target must be prepared before its result is verified.",
        summary: "Prepare the maintenance target, preserve the resulting state, and then verify it.",
        parameters: [],
        preconditions: [{
          condition: "The maintenance target is pending preparation.",
          evidence_refs: [stagePolicyIds[0]]
        }],
        steps: [{
          title: "Prepare the target",
          body: "Prepare the target and record the resulting state for the verification stage.",
          evidence_refs: [stagePolicyIds[0]]
        }, {
          title: "Verify the result",
          body: "Run the observed verification and record its success signal.",
          evidence_refs: [stagePolicyIds[1]]
        }],
        recovery_rules: [],
        verification: [{
          check: "Confirm the second stage completed.",
          success_signal: "The verification stage reports completion.",
          evidence_refs: [stagePolicyIds[1]]
        }],
        do_not_use_when: [],
        decision_guidance: {
          preference: ["Preserve the observed stage order."],
          anti_pattern: []
        },
        tools: ["exec"],
        tags: ["maintenance", "staged-workflow"],
        evidence_occurrence_ids: supports.map((item) => item.sequence_occurrence_id),
        confidence: 0.9
      } as unknown as T;
    },
    status() {
      return {
        provider: "host",
        model: "procedural-sequence-skill-test",
        configured: true,
        remote: true
      };
    }
  };
}

function differentFamilyEmbedder(): Embedder {
  let signatureCall = 0;
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "different-family-test-embedding"
    },
    isRemote() {
      return false;
    },
    async embed(texts: string[]) {
      const vector = signatureCall === 0 ? [1, 0] : [0, 1];
      signatureCall += 1;
      return texts.map(() => [...vector]);
    },
    async embedOne() {
      return [1, 0];
    },
    status() {
      return {
        provider: "local",
        model: "different-family-test-embedding",
        configured: true,
        remote: false
      };
    }
  };
}
