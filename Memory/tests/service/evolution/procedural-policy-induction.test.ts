import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  ProceduralPolicyEvidencePipeline,
  buildEpisodeSpanCreditRun,
  type LlmClient,
  type LlmMessage
} from "../../../src/index.js";
import { Repositories } from "../../../src/storage/repositories.js";
import {
  PROCEDURAL_POLICY_TEST_NAMESPACE,
  buildProceduralPolicyPath,
  persistProceduralPolicyEvidence
} from "../../fixtures/procedural-policy-fixture.js";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const {
  cleanup,
  createTestService
} = createMemoryServiceFixture();

afterEach(cleanup);

describe("procedural Span Policy induction", () => {
  it("induces, repairs, versions, and rolls back a cross-Session candidate Policy", async () => {
    const calls: Array<{
      messages: LlmMessage[];
      operation: string;
    }> = [];
    const { db, service } = createTestService({
      skillLlm: proceduralPolicyLlm(calls),
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
    const supportA = persistProceduralPolicyEvidence(repos, "support-a");
    const supportB = persistProceduralPolicyEvidence(repos, "support-b");
    const initialCluster = repos.proceduralSpanClusters.upsert({
      id: "procedural-cluster-dependency-repair",
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      minDistinctSupportEpisodes: 2,
      members: [
        { occurrenceId: supportA.occurrence.id, evidenceRole: "support", similarity: 0.96 },
        { occurrenceId: supportB.occurrence.id, evidenceRole: "support", similarity: 0.93 }
      ],
      clusterBasis: {
        candidateSource: "test.explicit-cluster",
        comparisonFields: ["goal", "entry_state", "procedure", "effect"]
      },
      at: "2026-08-20T00:01:00.000Z"
    });
    expect(initialCluster).toMatchObject({
      status: "ready",
      distinctSupportEpisodeCount: 2,
      memberCount: 2
    });
    enqueueInduction(repos, "job-procedural-policy-initial", initialCluster.membershipVersion);

    const firstRun = await service.runWorkerOnce(10);

    expect(firstRun.jobs).toEqual(expect.arrayContaining([
      expect.objectContaining({
        jobId: "job-procedural-policy-initial",
        jobType: "l2_induction",
        status: "succeeded"
      })
    ]));
    expect(calls.map((call) => call.operation)).toEqual([
      "procedural_policy.induction.v1",
      "procedural_policy.induction.v1.repair.1"
    ]);
    const promptPayload = JSON.parse(calls[0]!.messages.find((message) =>
      message.role === "user" && message.content.includes("selected_path_evidence"))!.content) as {
        selected_path_evidence: Array<{
          occurrence_id: string;
          episode_id: string;
          path_id: string;
          span_id: string;
          pre_state: string;
          post_state: string;
          steps: Array<{ step_id: string; pre_state: string; post_state: string }>;
        }>;
      };
    expect(promptPayload.selected_path_evidence).toHaveLength(2);
    expect(promptPayload.selected_path_evidence).toEqual(expect.arrayContaining([
      expect.objectContaining({
        occurrence_id: supportA.occurrence.id,
        episode_id: supportA.episodeId,
        path_id: supportA.path.id,
        span_id: supportA.path.spans[0]!.id,
        pre_state: expect.any(String),
        post_state: expect.any(String),
        steps: expect.arrayContaining([
          expect.objectContaining({ step_id: "step-support-a-inspect" }),
          expect.objectContaining({ step_id: "step-support-a-repair" })
        ])
      }),
      expect.objectContaining({ occurrence_id: supportB.occurrence.id })
    ]));

    const initialPolicy = repos.proceduralPolicies.getActiveForCluster(initialCluster.id);
    expect(initialPolicy).toMatchObject({
      status: "active",
      clusterMembershipVersion: initialCluster.membershipVersion,
      evidenceOccurrenceIds: [supportA.occurrence.id, supportB.occurrence.id],
      compilerModel: "procedural-policy-test"
    });
    const initialMappings = repos.proceduralPolicies.listOccurrences(initialPolicy!.id);
    expect(initialMappings).toEqual([
      expect.objectContaining({
        occurrenceId: supportA.occurrence.id,
        episodeId: supportA.episodeId,
        sessionId: supportA.sessionId,
        evidenceRole: "support",
        status: "active"
      }),
      expect.objectContaining({
        occurrenceId: supportB.occurrence.id,
        episodeId: supportB.episodeId,
        sessionId: supportB.sessionId,
        evidenceRole: "support",
        status: "active"
      })
    ]);
    const initialMemory = repos.memories.get(initialPolicy!.l2MemoryId!);
    expect(initialMemory).toMatchObject({
      status: "resolving",
      memoryLayer: "L2",
      memoryKey: "policy:procedural-span-cluster:procedural-cluster-dependency-repair"
    });
    expect(initialMemory?.sessionId).toBeUndefined();
    expect(initialMemory?.conversationId).toBeUndefined();
    expect(initialMemory?.properties.internal_info).toMatchObject({
      source_session_ids: [supportA.sessionId, supportB.sessionId],
      policy: {
        skill_eligible: false,
        status: "candidate",
        source_episode_ids: [supportA.episodeId, supportB.episodeId]
      }
    });

    enqueueInduction(
      repos,
      "job-procedural-policy-idempotent",
      initialCluster.membershipVersion,
      "2026-08-20T00:03:00.000Z"
    );
    await service.runWorkerOnce(10);
    expect(calls).toHaveLength(2);
    expect(db.db.prepare(`SELECT COUNT(*) AS count FROM procedural_policy_versions`).get())
      .toEqual({ count: 1 });

    const counterexample = persistProceduralPolicyEvidence(repos, "counterexample", {
      terminationStatus: "blocked",
      at: "2026-08-20T00:04:00.000Z"
    });
    const changedCluster = repos.proceduralSpanClusters.upsert({
      id: initialCluster.id,
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      minDistinctSupportEpisodes: 2,
      members: [
        { occurrenceId: supportA.occurrence.id, evidenceRole: "support", similarity: 0.96 },
        { occurrenceId: supportB.occurrence.id, evidenceRole: "support", similarity: 0.93 },
        { occurrenceId: counterexample.occurrence.id, evidenceRole: "counterexample", similarity: 0.81 }
      ],
      clusterBasis: { candidateSource: "test.with-counterexample" },
      at: "2026-08-20T00:05:00.000Z"
    });
    expect(changedCluster).toMatchObject({ status: "ready", memberCount: 3 });
    expect(repos.proceduralPolicies.get(initialPolicy!.id)).toMatchObject({ status: "inactive" });
    expect(repos.proceduralPolicies.listOccurrences(initialPolicy!.id)
      .every((mapping) => mapping.status === "superseded")).toBe(true);
    expect(repos.memories.get(initialPolicy!.l2MemoryId!)?.status).toBe("archived");

    enqueueInduction(
      repos,
      "job-procedural-policy-counterexample",
      changedCluster.membershipVersion,
      "2026-08-20T00:06:00.000Z"
    );
    await service.runWorkerOnce(10);
    expect(calls).toHaveLength(3);
    const changedPolicy = repos.proceduralPolicies.getActiveForCluster(initialCluster.id);
    expect(changedPolicy?.id).not.toBe(initialPolicy!.id);
    expect(changedPolicy?.policy.evidence).toMatchObject({
      supportOccurrenceIds: [supportA.occurrence.id, supportB.occurrence.id],
      counterexampleOccurrenceIds: [counterexample.occurrence.id]
    });
    expect(changedPolicy?.evidenceOccurrenceIds).toEqual([
      supportA.occurrence.id,
      supportB.occurrence.id
    ]);
    expect(repos.proceduralPolicies.listOccurrences(changedPolicy!.id)).toContainEqual(
      expect.objectContaining({
        occurrenceId: counterexample.occurrence.id,
        evidenceRole: "counterexample",
        status: "active"
      })
    );

    const rolledBackCluster = repos.proceduralSpanClusters.upsert({
      id: initialCluster.id,
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      minDistinctSupportEpisodes: 2,
      members: [
        { occurrenceId: supportA.occurrence.id, evidenceRole: "support", similarity: 0.96 },
        { occurrenceId: supportB.occurrence.id, evidenceRole: "support", similarity: 0.93 }
      ],
      clusterBasis: { candidateSource: "test.rollback" },
      at: "2026-08-20T00:07:00.000Z"
    });
    expect(rolledBackCluster.membershipVersion).toBe(initialCluster.membershipVersion);
    enqueueInduction(
      repos,
      "job-procedural-policy-rollback",
      rolledBackCluster.membershipVersion,
      "2026-08-20T00:08:00.000Z"
    );
    await service.runWorkerOnce(10);

    expect(calls).toHaveLength(3);
    expect(repos.proceduralPolicies.getActiveForCluster(initialCluster.id)?.id).toBe(initialPolicy!.id);
    expect(repos.proceduralPolicies.listOccurrences(initialPolicy!.id)
      .every((mapping) => mapping.status === "active")).toBe(true);
    expect(repos.memories.get(initialPolicy!.l2MemoryId!)?.status).toBe("resolving");
    expect(repos.proceduralPolicies.get(changedPolicy!.id)).toMatchObject({ status: "inactive" });
    expect(repos.memories.get(changedPolicy!.l2MemoryId!)?.status).toBe("archived");

    const revisedSupportPath = buildProceduralPolicyPath({
      suffix: "support-a-revised",
      episodeId: supportA.episodeId,
      rawTurnId: supportA.rawTurnId,
      terminationStatus: "success"
    });
    repos.proceduralPaths.save({
      path: revisedSupportPath,
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      createdAt: "2026-08-20T00:09:00.000Z"
    });
    expect(repos.proceduralSpanClusters.get(initialCluster.id)).toMatchObject({
      status: "stale",
      activePolicyVersionId: null
    });
    expect(repos.proceduralPolicies.get(initialPolicy!.id)).toMatchObject({ status: "inactive" });
    expect(repos.proceduralPolicies.listOccurrences(initialPolicy!.id)
      .every((mapping) => mapping.status === "superseded")).toBe(true);
    expect(repos.memories.get(initialPolicy!.l2MemoryId!)?.status).toBe("archived");
    db.close();
  });

  it("uses active SpanCredit to assign evidence roles and trigger Policy induction", async () => {
    const calls: Array<{ messages: LlmMessage[]; operation: string }> = [];
    const { db, service } = createTestService({
      skillLlm: proceduralPolicyLlm(calls),
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
    const supportA = persistProceduralPolicyEvidence(repos, "credit-policy-support-a");
    const supportB = persistProceduralPolicyEvidence(repos, "credit-policy-support-b");
    const counterexample = persistProceduralPolicyEvidence(repos, "credit-policy-counterexample", {
      terminationStatus: "blocked"
    });
    const neutral = persistProceduralPolicyEvidence(repos, "credit-policy-neutral");
    saveCredit(repos, supportA, { goalCredit: 1, processQuality: 0.7, evidenceRole: "support" });
    saveCredit(repos, supportB, { goalCredit: 1, processQuality: 0.5, evidenceRole: "support" });
    saveCredit(repos, counterexample, {
      goalCredit: 0,
      processQuality: -0.8,
      evidenceRole: "counterexample"
    });
    saveCredit(repos, neutral, { goalCredit: 0, processQuality: 0.1, evidenceRole: "neutral" });
    let jobIndex = 0;
    const evidencePipeline = new ProceduralPolicyEvidencePipeline({
      repos,
      enqueueJob(input) {
        jobIndex += 1;
        const at = input.createdAt ?? "2026-08-20T00:10:00.000Z";
        return repos.runtime.enqueueJob({
          id: `job-credit-governed-policy-${jobIndex}`,
          jobType: input.jobType,
          status: "queued",
          dedupeKey: input.dedupeKey ?? `credit-governed-policy:${jobIndex}`,
          userId: input.userId,
          sessionId: input.sessionId,
          episodeId: input.episodeId,
          targetMemoryId: input.targetMemoryId,
          payload: input.payload ?? {},
          attempts: 0,
          maxAttempts: input.maxAttempts ?? 3,
          createdAt: at,
          updatedAt: at
        });
      }
    });

    const refreshed = evidencePipeline.refreshCluster({
      clusterId: "procedural-cluster-credit-governed",
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      candidates: [
        { occurrenceId: supportA.occurrence.id, similarity: 0.96 },
        { occurrenceId: supportB.occurrence.id, similarity: 0.94 },
        { occurrenceId: counterexample.occurrence.id, similarity: 0.84 },
        { occurrenceId: neutral.occurrence.id, similarity: 0.91 }
      ],
      clusterBasis: { candidateSource: "test.semantic-neighborhood" },
      at: "2026-08-20T00:10:00.000Z"
    });

    expect(refreshed.excludedOccurrenceIds).toEqual([neutral.occurrence.id]);
    expect(refreshed.cluster).toMatchObject({
      status: "ready",
      memberCount: 3,
      distinctSupportEpisodeCount: 2,
      clusterBasis: {
        creditGoverned: true,
        candidateSource: "test.semantic-neighborhood"
      }
    });
    expect(repos.proceduralSpanClusters.listMembers(refreshed.cluster!.id)).toEqual([
      expect.objectContaining({ occurrenceId: supportA.occurrence.id, evidenceRole: "support" }),
      expect.objectContaining({ occurrenceId: supportB.occurrence.id, evidenceRole: "support" }),
      expect.objectContaining({ occurrenceId: counterexample.occurrence.id, evidenceRole: "counterexample" })
    ]);
    expect(refreshed.inductionJob).toMatchObject({
      jobType: "l2_induction",
      payload: {
        proceduralClusterId: "procedural-cluster-credit-governed",
        membershipVersion: refreshed.cluster!.membershipVersion
      }
    });

    const workerRun = await service.runWorkerOnce(10);

    expect(workerRun.jobs).toContainEqual(expect.objectContaining({
      jobId: refreshed.inductionJob!.id,
      jobType: "l2_induction",
      status: "succeeded"
    }));
    expect(calls.map((call) => call.operation)).toEqual([
      "procedural_policy.induction.v1",
      "procedural_policy.induction.v1.repair.1"
    ]);
    const promptPayload = JSON.parse(calls[0]!.messages.find((message) =>
      message.role === "user" && message.content.includes("selected_path_evidence"))!.content) as {
        selected_path_evidence: Array<{
          occurrence_id: string;
          evidence_role: string;
          span_credit?: { evidence_role: string; credit_score: number; process_quality: number };
        }>;
      };
    expect(promptPayload.selected_path_evidence).toEqual(expect.arrayContaining([
      expect.objectContaining({
        occurrence_id: supportA.occurrence.id,
        evidence_role: "support",
        span_credit: expect.objectContaining({
          evidence_role: "support",
          credit_score: 0.98,
          process_quality: 0.7
        })
      }),
      expect.objectContaining({
        occurrence_id: counterexample.occurrence.id,
        evidence_role: "counterexample",
        span_credit: expect.objectContaining({ evidence_role: "counterexample" })
      })
    ]));
    const policy = repos.proceduralPolicies.getActiveForCluster(refreshed.cluster!.id);
    expect(policy?.policy.evidence).toMatchObject({
      supportOccurrenceIds: [supportA.occurrence.id, supportB.occurrence.id],
      counterexampleOccurrenceIds: [counterexample.occurrence.id]
    });
    db.close();
  });
});

function saveCredit(
  repos: Repositories,
  evidence: ReturnType<typeof persistProceduralPolicyEvidence>,
  input: {
    goalCredit: number;
    processQuality: number;
    evidenceRole: "support" | "counterexample" | "neutral";
  }
): void {
  const preStateId = evidence.occurrence.preStateId;
  const postStateId = evidence.occurrence.postStateId;
  const run = buildEpisodeSpanCreditRun({
    episodeId: evidence.episodeId,
    pathId: evidence.path.id,
    pathHash: evidence.path.pathHash,
    namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
    rewardHash: `reward:${evidence.episodeId}:${input.goalCredit}`,
    goalAchievement: input.goalCredit,
    statePotentials: [{
      boundaryIndex: 0,
      stateId: preStateId,
      progress: 0,
      evidenceRefs: [preStateId],
      reason: "Episode start anchor"
    }, {
      boundaryIndex: 1,
      stateId: postStateId,
      progress: input.goalCredit,
      evidenceRefs: [postStateId],
      reason: "Episode goal-achievement anchor"
    }],
    credits: [{
      occurrenceId: evidence.occurrence.id,
      spanId: evidence.occurrence.spanId,
      spanIndex: evidence.occurrence.spanIndex,
      preStateId,
      postStateId,
      goalCredit: input.goalCredit,
      processQuality: input.processQuality,
      confidence: 0.98,
      evidenceRole: input.evidenceRole,
      evidenceRefs: [evidence.occurrence.id],
      reason: "credit-governed Policy test"
    }],
    scorerModel: "span-credit-test"
  });
  repos.proceduralSpanCredits.saveAndActivate(run, "2026-08-20T00:09:00.000Z");
}

function enqueueInduction(
  repos: Repositories,
  jobId: string,
  membershipVersion: string,
  at = "2026-08-20T00:02:00.000Z"
): void {
  repos.runtime.enqueueJob({
    id: jobId,
    jobType: "l2_induction",
    status: "queued",
    dedupeKey: `${jobId}:${membershipVersion}`,
    userId: "user-procedural-policy",
    payload: {
      proceduralClusterId: "procedural-cluster-dependency-repair",
      membershipVersion
    },
    attempts: 0,
    maxAttempts: 3,
    createdAt: at,
    updatedAt: at
  });
}

function proceduralPolicyLlm(calls: Array<{
  messages: LlmMessage[];
  operation: string;
}>): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/procedural-policy-test",
      model: "procedural-policy-test"
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
          occurrence_id: string;
          evidence_role: "support" | "counterexample";
          steps: Array<{ step_id: string }>;
        }>;
      };
      const supports = (payload.selected_path_evidence ?? [])
        .filter((item) => item.evidence_role === "support");
      const counterexamples = (payload.selected_path_evidence ?? [])
        .filter((item) => item.evidence_role === "counterexample");
      const draft = {
        title: "Diagnose, repair, and verify a dependency conflict",
        goal_pattern: "Turn a reproducible dependency failure into a verified focused build",
        trigger_conditions: [
          "A focused build exposes a concrete dependency constraint conflict",
          "The failure can be reproduced before modifying the dependency declaration"
        ],
        procedure_steps: [{
          instruction: "Inspect and reproduce the focused failure to identify the conflicting constraint.",
          evidence_refs: [supports[0]!.occurrence_id, supports[0]!.steps[0]!.step_id]
        }, {
          instruction: "Correct the constraint and rerun the same focused build.",
          evidence_refs: [supports[1]!.occurrence_id, supports[1]!.steps[1]!.step_id]
        }],
        recovery_rules: counterexamples.length ? [{
          condition: "The corrected constraint exposes a separate platform compatibility blocker.",
          action: "Stop treating the original dependency edit as sufficient and surface the new blocker.",
          evidence_refs: [counterexamples[0]!.occurrence_id, counterexamples[0]!.steps[1]!.step_id]
        }] : [],
        verification_steps: [{
          check: "Rerun the exact focused build used to reproduce the conflict.",
          success_signal: "The build passes and the dependency issue is recorded as resolved.",
          evidence_refs: [supports[0]!.steps[1]!.step_id, supports[1]!.steps[1]!.step_id]
        }],
        apply_when: ["The failure names or isolates a dependency constraint conflict."],
        do_not_apply_when: ["The failure is an unrelated platform compatibility blocker."],
        invariants: ["Use the same focused verification target before and after the edit."],
        expected_effect: "The dependency conflict is resolved with an explicit focused-build success signal.",
        evidence_occurrence_ids: supports.map((item) => item.occurrence_id),
        confidence: 0.88
      };
      if (calls.length === 1) {
        draft.procedure_steps[0]!.evidence_refs = ["invented-step-id"];
      }
      return draft as unknown as T;
    },
    status() {
      return {
        provider: "host",
        model: "procedural-policy-test",
        configured: true,
        remote: true
      };
    }
  };
}
