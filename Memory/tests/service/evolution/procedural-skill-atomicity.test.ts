import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_MEMMY_CONFIG,
  type LlmClient,
  type LlmCompletionOptions,
  type LlmMessage,
  type RuntimeNamespace
} from "../../../src/index.js";
import { Repositories } from "../../../src/storage/repositories.js";
import { ProceduralTrajectoryRepository } from "../../../src/storage/procedural-trajectory-repository.js";
import {
  createCapturingEmbedder,
  createMemoryServiceFixture,
  runWorkerRounds
} from "../../fixtures/memory-service-fixture.js";

const USER_ID = "procedural-atomicity-user";
const STEP_SEMANTICS_OPERATION_PREFIX = "procedural.step_semantics.";

const { cleanup, createTestService } = createMemoryServiceFixture();

afterEach(() => {
  vi.restoreAllMocks();
  cleanup();
});

describe("procedural Skill atomic publication", () => {
  it("serializes same-user V2 induction and never publishes a slow obsolete version", async () => {
    const firstSkillStarted = deferred<void>();
    const releaseFirstSkill = deferred<void>();
    let skillCalls = 0;
    const llm = atomicityLlm({
      async onSkillCall(occurrenceIds) {
        skillCalls += 1;
        if (skillCalls === 1) {
          firstSkillStarted.resolve();
          await releaseFirstSkill.promise;
          return skillResponse("obsolete-skill-draft", "obsolete", occurrenceIds);
        }
        return skillResponse("current-skill-draft", "current", occurrenceIds);
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
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);

    await executeSuccessfulEpisode(service, "codex", "first");
    await runWorkerRounds(service, 16);
    await executeSuccessfulEpisode(service, "cursor", "second");
    await runUntilRunnableQueuedSkill(service, repos);

    const obsoleteRun = service.runWorkerOnce(100);
    await firstSkillStarted.promise;

    await executeSuccessfulEpisode(service, "claude", "third");
    await runWorkerRounds(service, 24);
    expect(skillCalls).toBe(1);

    releaseFirstSkill.resolve();
    await obsoleteRun;
    await runWorkerRounds(service, 24);
    expect(skillCalls).toBeGreaterThanOrEqual(2);

    const clusterAfterRelease = activeSpanFiveCluster(repos);
    const currentMemoryId = clusterAfterRelease.activeSkillMemoryId;
    expect(currentMemoryId).toBeTruthy();
    expect(repos.memories.get(currentMemoryId!)?.memoryValue).toContain("current");
    expect(repos.memories.get(currentMemoryId!)?.memoryValue).not.toContain("obsolete");
    expect(repos.proceduralTrajectory.listSkillVersions(clusterAfterRelease.id)
      .filter((version) => version.skillMemoryId)).toHaveLength(1);
  });

  it("keeps the last-known-good V2 Skill live when its replacement fails validation", async () => {
    let rejectReplacement = false;
    let skillCalls = 0;
    const llm = atomicityLlm({
      async onSkillCall(occurrenceIds) {
        skillCalls += 1;
        const response = skillResponse(
          rejectReplacement ? "rejected-replacement" : "known-good-skill",
          rejectReplacement ? "rejected" : "known-good",
          occurrenceIds
        );
        if (!rejectReplacement) return response;
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
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);

    await executeSuccessfulEpisode(service, "codex", "known-good-first");
    await runWorkerRounds(service, 16);
    const second = await executeSuccessfulEpisode(service, "cursor", "known-good-second");
    await runWorkerRounds(service, 20);

    const original = activeSpanFiveCluster(repos);
    const originalClusterVersionId = original.activeVersionId!;
    const originalSkillVersionId = original.activeSkillVersionId!;
    const originalSkillMemoryId = original.activeSkillMemoryId!;
    const originalMemory = repos.memories.get(originalSkillMemoryId)!;
    expect(originalMemory.status).toBe("resolving");

    rejectReplacement = true;
    const previousVersion = repos.proceduralTrajectory.getClusterVersion(
      originalClusterVersionId
    )!;
    const previousMembers = repos.proceduralTrajectory.listClusterMembers(
      originalClusterVersionId
    );
    const committedAt = new Date().toISOString();
    const replacementClusterVersion = repos.proceduralTrajectory.commitClusterVersion({
      clusterId: original.id,
      expectedActiveVersionId: originalClusterVersionId,
      medoidOccurrenceId: previousVersion.medoidOccurrenceId,
      members: [
        ...previousMembers.map((member, index) => ({
          occurrenceId: member.occurrenceId,
          rewardHash: index === 0
            ? `${member.rewardHash}:replacement-revision`
            : member.rewardHash,
          coarseSimilarity: member.coarseSimilarity,
          alignment: member.alignment
        }))
      ],
      metrics: previousVersion.metrics,
      createdAt: committedAt
    }).record;
    const inductionVersion = repos.proceduralTrajectory.getSkillVersion(
      originalSkillVersionId
    )!.payload.inductionVersion as string;
    repos.runtime.enqueueJob({
      id: `job_v2_last_known_good_${replacementClusterVersion.id}`,
      jobType: "procedural_skill_induction",
      status: "queued",
      userId: USER_ID,
      sessionId: second.sessionId,
      episodeId: second.episodeId,
      targetMemoryId: original.id,
      payload: {
        clusterId: original.id,
        clusterVersionId: replacementClusterVersion.id,
        inductionVersion
      },
      attempts: 0,
      maxAttempts: 1,
      createdAt: committedAt,
      updatedAt: committedAt
    });
    await runWorkerRounds(service, 12);

    const current = repos.proceduralTrajectory.getClusterHead(original.id)!;
    expect(current.activeVersionId).not.toBe(originalClusterVersionId);
    expect(current.activeSkillVersionId).toBe(originalSkillVersionId);
    expect(current.activeSkillMemoryId).toBe(originalSkillMemoryId);
    expect(repos.proceduralTrajectory.getSkillVersion(originalSkillVersionId)?.status)
      .toBe("active");
    expect(repos.memories.get(originalSkillMemoryId)).toMatchObject({
      status: "resolving",
      version: originalMemory.version,
      contentHash: originalMemory.contentHash
    });

    const decisions = repos.proceduralTrajectory.listSkillVersions(original.id);
    expect(decisions).toHaveLength(2);
    const rejection = decisions.find((version) => !version.skillMemoryId);
    expect(rejection).toMatchObject({
      clusterVersionId: current.activeVersionId,
      status: "superseded"
    });
    expect(rejection?.payload).toMatchObject({
      admitted: false,
      reason: "invalid-evidence-anchor",
      preservedActiveSkillVersionId: originalSkillVersionId,
      preservedActiveSkillMemoryId: originalSkillMemoryId
    });

    const callsAfterRejectedReplacement = skillCalls;
    await runWorkerRounds(service, 12);
    expect(skillCalls).toBe(callsAfterRejectedReplacement);
  });

  it("rolls public archival back when rejection head publication fails", async () => {
    const llm = atomicityLlm({
      onSkillCall: async (occurrenceIds) =>
        skillResponse("atomic-rejection-skill", "atomic", occurrenceIds)
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
      embedder: createCapturingEmbedder([])
    });
    const repos = new Repositories(db.db);

    await executeSuccessfulEpisode(service, "codex", "rollback-first");
    await runWorkerRounds(service, 16);
    const second = await executeSuccessfulEpisode(service, "cursor", "rollback-second");
    await runWorkerRounds(service, 20);
    const admittedHead = activeSpanFiveCluster(repos);
    const admittedSkillVersionId = admittedHead.activeSkillVersionId;
    const admittedMemoryId = admittedHead.activeSkillMemoryId;
    expect(admittedSkillVersionId).toBeTruthy();
    expect(admittedMemoryId).toBeTruthy();

    const originalSave = ProceduralTrajectoryRepository.prototype.saveSkillVersion;
    const failure = vi.spyOn(
      ProceduralTrajectoryRepository.prototype,
      "saveSkillVersion"
    ).mockImplementation(function (this: ProceduralTrajectoryRepository, input) {
      if (!input.skillMemoryId) throw new Error("injected rejection publication failure");
      return originalSave.call(this, input);
    });

    await service.feedback({
      namespace: second.namespace,
      sessionId: second.sessionId,
      episodeId: second.episodeId,
      l1MemoryId: second.l1MemoryId,
      channel: "explicit",
      polarity: "negative",
      magnitude: 1,
      rationale: "force negative drift for atomic rollback"
    });
    let observedFailure = false;
    for (let round = 0; round < 20 && !observedFailure; round += 1) {
      await service.runWorkerOnce(100);
      observedFailure = Boolean(repos.db.prepare(
        `SELECT 1 AS found FROM evolution_jobs
         WHERE job_type = 'trajectory_window_ingest' AND status = 'failed'
           AND last_error LIKE '%injected rejection publication failure%'
         LIMIT 1`
      ).get());
    }
    expect(observedFailure).toBe(true);

    const headAfterFailedRejection = repos.proceduralTrajectory.getClusterHead(admittedHead.id)!;
    expect(headAfterFailedRejection.activeSkillVersionId).toBe(admittedSkillVersionId);
    expect(headAfterFailedRejection.activeSkillMemoryId).toBe(admittedMemoryId);
    expect(repos.memories.get(admittedMemoryId!)?.status).not.toBe("archived");
    expect(repos.proceduralTrajectory.listSkillVersions(admittedHead.id)
      .filter((version) => !version.skillMemoryId)).toHaveLength(0);

    failure.mockRestore();
    await runWorkerRounds(service, 8);
    const recoveredHead = repos.proceduralTrajectory.getClusterHead(admittedHead.id)!;
    expect(repos.memories.get(admittedMemoryId!)?.status).toBe("archived");
    expect(repos.proceduralTrajectory.getSkillVersion(
      recoveredHead.activeSkillVersionId!
    )?.skillMemoryId).toBeUndefined();
  });
});

async function runUntilRunnableQueuedSkill(
  service: ReturnType<typeof createTestService>["service"],
  repos: Repositories
): Promise<void> {
  for (let round = 0; round < 30; round += 1) {
    const state = repos.db.prepare(
      `SELECT
         SUM(CASE WHEN job_type = 'procedural_skill_induction' AND status = 'queued'
                  THEN 1 ELSE 0 END) AS queuedV2,
         SUM(CASE WHEN job_type IN (
                    'l2_association',
                    'l2_induction',
                    'skill_crystallization',
                    'long_trajectory_mining',
                    'long_trajectory_skill_induction'
                  ) AND status IN ('queued', 'leased', 'failed')
                  THEN 1 ELSE 0 END) AS blockers
       FROM evolution_jobs
       WHERE user_id = ?`
    ).get(USER_ID) as { queuedV2: number; blockers: number };
    if (state.queuedV2 > 0 && state.blockers === 0) return;
    await service.runWorkerOnce(1);
  }
  throw new Error("procedural_skill_induction was not runnable");
}

function activeSpanFiveCluster(repos: Repositories) {
  const row = repos.db.prepare(
    `SELECT id FROM trajectory_window_clusters
     WHERE user_id = ? AND scale = 5 AND status = 'active'
       AND active_skill_memory_id IS NOT NULL
     ORDER BY updated_at DESC, id DESC LIMIT 1`
  ).get(USER_ID) as { id: string } | undefined;
  if (!row) throw new Error("active Span-5 Skill cluster not found");
  return repos.proceduralTrajectory.getClusterHead(row.id)!;
}

async function executeSuccessfulEpisode(
  service: ReturnType<typeof createTestService>["service"],
  source: string,
  suffix: string
): Promise<{
  namespace: RuntimeNamespace;
  sessionId: string;
  episodeId: string;
  l1MemoryId: string;
}> {
  const namespace: RuntimeNamespace = {
    source,
    profileId: `profile-${suffix}`,
    userId: USER_ID
  };
  const session = service.openSession({ namespace });
  const toolNames = ["inspect_schema", "prepare_inputs", "build_artifact", "run_checks", "verify_output"];
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

function atomicityLlm(input: {
  onSkillCall(occurrenceIds: string[]): Promise<Record<string, unknown>>;
}): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      model: "procedural-atomicity"
    },
    isConfigured: () => true,
    complete: async () => "{}",
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
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
        return {
          l1: {
            summary: "Built the requested artifact and verified the output",
            evidence: [{ quote: "Build and validate artifact", role: "user", kind: "task_outcome" }]
          },
          user: null
        } as unknown as T;
      }
      if (options.operation === "reward.reward.r_human.v7") {
        const negative = content.includes("force negative drift for atomic rollback");
        const score = negative ? -1 : 1;
        return {
          goal_achievement: score,
          process_quality: score,
          user_satisfaction: score,
          reason: negative ? "latest feedback invalidated the result" : "task succeeded"
        } as unknown as T;
      }
      if (options.operation.startsWith(STEP_SEMANTICS_OPERATION_PREFIX)) {
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
            intent: `Complete reusable stage ${candidate.toolName ?? "tool"}`,
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
      if (options.operation === "procedural.procedural-pattern-skill.v12") {
        const payload = JSON.parse(content) as {
          evidence_anchor_catalog?: Array<{
            anchor_id: string;
            allowed_usage: "mandatory" | "conditional_only";
            support_episode_ids: string[];
          }>;
        };
        const mandatoryAnchors = (payload.evidence_anchor_catalog ?? [])
          .filter((anchor) => anchor.allowed_usage === "mandatory" &&
            anchor.support_episode_ids.length >= 2);
        const response = await input.onSkillCall(
          mandatoryAnchors.map((anchor) => anchor.anchor_id)
        );
        const supportEpisodeIds = [...new Set(mandatoryAnchors.flatMap((anchor) =>
          anchor.support_episode_ids))];
        return {
          ...response,
          local_subproblem_closure: {
            closed: true,
            subproblem: "Complete a repeated local artifact stage.",
            entry_condition: "The artifact stage is incomplete.",
            resolution: "Run the shared construction actions.",
            resolved_state: "The artifact stage is complete.",
            success_check: "Observe the shared output verification.",
            support_episode_ids: supportEpisodeIds,
            reason: "The same successful Episodes support construction and verification."
          }
        } as unknown as T;
      }
      return {} as T;
    },
    status: () => ({
      provider: "host",
      model: "procedural-atomicity",
      configured: true,
      remote: true
    })
  };
}

function skillResponse(
  name: string,
  marker: string,
  sourceAnchorIds: string[]
): Record<string, unknown> {
  return {
    admit: true,
    rejection_reason: null,
    name,
    display_title: `${marker} Skill`,
    retrieval_blurb: `${marker} reusable artifact procedure`,
    trigger_context: `Use the ${marker} procedure for matching artifact tasks.`,
    summary: `${marker} ordered artifact construction and verification.`,
    preconditions: ["The artifact target is known."],
    parameters: [],
    procedure_steps: [{
      title: `${marker} construction`,
      body: `Run the ${marker} ordered construction stages.`,
      source_anchor_ids: sourceAnchorIds
    }],
    verification_steps: [{
      check: `${marker} verification`,
      success_signal: `The ${marker} output check succeeds.`,
      source_anchor_ids: sourceAnchorIds.slice(-1)
    }],
    do_not_apply_when: [],
    decision_guidance: { preference: [], anti_pattern: [] },
    examples: [],
    tags: [marker],
    tools: [],
    confidence: 0.9
  };
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
