import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_MEMMY_CONFIG,
  PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
  buildEpisodeSpanCreditRun,
  spanCreditRewardHash,
  type Embedder,
  type LlmClient,
  type LlmMessage
} from "../../../src/index.js";
import { Repositories } from "../../../src/storage/repositories.js";
import { evolutionJobDedupeKey, type EnqueueJobInput } from "../../../src/service/worker/job-handlers.js";
import {
  PROCEDURAL_POLICY_TEST_NAMESPACE,
  buildProceduralPolicyPath,
  persistProceduralPolicyEvidence
} from "../../fixtures/procedural-policy-fixture.js";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const { cleanup, createTestService } = createMemoryServiceFixture();

afterEach(cleanup);

describe("automatic procedural Span semantic clustering", () => {
  it("runs SpanCredit -> embeddings -> cross-Episode cluster -> Policy without supplied candidates", async () => {
    const operations: string[] = [];
    const embeddedTexts: string[] = [];
    const { db, service } = createTestService({
      skillLlm: automaticPipelineLlm(operations),
      embedder: semanticTestEmbedder(embeddedTexts),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          },
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            proceduralSimilarityThreshold: 0.9,
            goalSimilarityThreshold: 0.9,
            policySimilarityThreshold: 0.9,
            minDistinctSources: 2
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const supportA = persistProceduralPolicyEvidence(repos, "auto-support-a");
    const supportB = persistProceduralPolicyEvidence(repos, "auto-support-b");
    const counterexample = persistProceduralPolicyEvidence(repos, "auto-counterexample", {
      terminationStatus: "blocked",
      rTask: -1
    });
    [supportA, supportB, counterexample].forEach((evidence, index) => {
      const episode = repos.runtime.getEpisode(evidence.episodeId)!;
      enqueue(repos, {
        jobType: "span_credit",
        userId: episode.userId,
        episodeId: episode.id,
        payload: {
          pathId: evidence.path.id,
          pathHash: evidence.path.pathHash,
          rewardHash: spanCreditRewardHash(episode)
        },
        createdAt: `2026-08-20T00:0${index + 1}:00.000Z`
      }, index + 1);
    });

    const creditRun = await service.runWorkerOnce(20);
    expect(creditRun.jobs.filter((job) => job.jobType === "span_credit"))
      .toHaveLength(3);
    const queuedClusters = db.db.prepare(
      `SELECT payload_json, dedupe_key FROM evolution_jobs
       WHERE job_type = 'procedural_span_cluster'
       ORDER BY created_at ASC`
    ).all() as Array<{ payload_json: string; dedupe_key: string }>;
    expect(queuedClusters).toHaveLength(3);
    expect(new Set(queuedClusters.map((row) =>
      (JSON.parse(row.payload_json) as { creditRunId: string }).creditRunId
    )).size).toBe(3);
    expect(new Set(queuedClusters.map((row) => row.dedupe_key)).size).toBe(3);

    const firstIncrement = await service.runWorkerOnce(1);
    expect(firstIncrement.jobs).toContainEqual(expect.objectContaining({
      jobType: "procedural_span_cluster",
      status: "succeeded"
    }));
    expect(embeddedTexts).toHaveLength(3);
    expect(repos.proceduralSpanClusters.listByNamespace(
      PROCEDURAL_POLICY_TEST_NAMESPACE,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    )).toEqual([
      expect.objectContaining({ status: "forming", memberCount: 1 })
    ]);

    await service.runWorkerOnce(1);
    expect(embeddedTexts).toHaveLength(6);
    expect(repos.proceduralSpanClusters.listByNamespace(
      PROCEDURAL_POLICY_TEST_NAMESPACE,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    )).toEqual([
      expect.objectContaining({ status: "ready", memberCount: 2 })
    ]);

    await service.runWorkerOnce(1);
    expect(embeddedTexts).toHaveLength(9);
    const clusters = repos.proceduralSpanClusters.listByNamespace(
      PROCEDURAL_POLICY_TEST_NAMESPACE,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    );
    expect(clusters).toHaveLength(1);
    expect(clusters[0]).toMatchObject({
      status: "ready",
      memberCount: 3,
      distinctEpisodeCount: 3,
      distinctSupportEpisodeCount: 2,
      clusterBasis: {
        discoveredAutomatically: true,
        assignmentMode: "incremental-cluster-center",
        membershipSimilarityBasis: "current-cluster-center-capability-goal-normalized-procedure",
        similarityThreshold: 0.9,
        views: ["capability_goal", "state_contract", "procedure_semantic"],
        matchingViews: ["capability_goal", "procedure_semantic"],
        centerEmbedding: {
          version: "procedural-span-cluster-center.v2",
          supportMemberCount: 2
        }
      }
    });
    const members = repos.proceduralSpanClusters.listMembers(clusters[0]!.id);
    expect(members).toHaveLength(3);
    expect(members).toEqual(expect.arrayContaining([
      expect.objectContaining({ occurrenceId: counterexample.occurrence.id, evidenceRole: "counterexample" }),
      expect.objectContaining({ occurrenceId: supportA.occurrence.id, evidenceRole: "support" }),
      expect.objectContaining({ occurrenceId: supportB.occurrence.id, evidenceRole: "support" })
    ]));
    expect(db.db.prepare(
      `SELECT COUNT(*) AS count FROM procedural_span_occurrence_embeddings`
    ).get()).toEqual({ count: 3 });

    const inductionRun = await service.runWorkerOnce(20);
    expect(inductionRun.jobs).toContainEqual(expect.objectContaining({
      jobType: "l2_induction",
      status: "succeeded"
    }));
    const policy = repos.proceduralPolicies.getActiveForCluster(clusters[0]!.id);
    expect(policy?.policy.evidence.supportOccurrenceIds).toEqual(expect.arrayContaining([
      supportA.occurrence.id,
      supportB.occurrence.id
    ]));
    expect(policy?.policy.evidence.supportOccurrenceIds).toHaveLength(2);
    expect(policy?.policy.evidence.counterexampleOccurrenceIds)
      .toEqual([counterexample.occurrence.id]);
    expect(operations.filter((operation) => operation === "span_credit.score.v2"))
      .toHaveLength(3);
    expect(operations).toContain("procedural_policy.induction.v1");

    const revisedCreditRun = buildEpisodeSpanCreditRun({
      episodeId: supportA.episodeId,
      pathId: supportA.path.id,
      pathHash: supportA.path.pathHash,
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      rewardHash: "reward:auto-support-a:revised",
      episodeReward: -1,
      credits: [{
        occurrenceId: supportA.occurrence.id,
        spanId: supportA.occurrence.spanId,
        spanIndex: supportA.occurrence.spanIndex,
        preStateId: supportA.occurrence.preStateId,
        postStateId: supportA.occurrence.postStateId,
        rewardCredit: -1,
        attributionType: "harmful",
        confidence: 0.98,
        evidenceRole: "counterexample",
        evidenceRefs: [supportA.occurrence.id],
        reason: "Reward revision changed the evidence role"
      }],
      scorerModel: "automatic-procedural-test"
    });
    repos.proceduralSpanCredits.saveAndActivate(
      revisedCreditRun,
      "2026-08-20T00:10:00.000Z"
    );
    enqueue(repos, {
      jobType: "procedural_span_cluster",
      userId: "user-procedural-policy",
      episodeId: supportA.episodeId,
      payload: {
        namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
        creditRunId: revisedCreditRun.id,
        pathId: supportA.path.id,
        algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
      },
      createdAt: "2026-08-20T00:11:00.000Z"
    }, 4);
    await service.runWorkerOnce(1);

    expect(embeddedTexts).toHaveLength(9);
    expect(repos.proceduralSpanClusters.get(clusters[0]!.id)).toMatchObject({
      status: "forming",
      memberCount: 3,
      distinctSupportEpisodeCount: 1
    });
    expect(repos.proceduralSpanClusters.listMembers(clusters[0]!.id)).toEqual(expect.arrayContaining([
      expect.objectContaining({ occurrenceId: supportA.occurrence.id, evidenceRole: "counterexample" }),
      expect.objectContaining({ occurrenceId: supportB.occurrence.id, evidenceRole: "support" })
    ]));
    expect(repos.proceduralPolicies.get(policy!.id)).toMatchObject({ status: "inactive" });
    db.close();
  });

  it("clusters by goal and procedure, ignores state-contract mismatch, and still gates both matching views", async () => {
    const embeddedTexts: string[] = [];
    const { db, service } = createTestService({
      skillLlm: automaticPipelineLlm([]),
      embedder: stateContractMismatchTestEmbedder(embeddedTexts),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          },
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            proceduralSimilarityThreshold: 0.8,
            minDistinctSources: 2
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const supports = [
      persistProceduralPolicyEvidence(repos, "three-view-a"),
      persistProceduralPolicyEvidence(repos, "three-view-b"),
      persistProceduralPolicyEvidence(repos, "procedure-mismatch"),
      persistProceduralPolicyEvidence(repos, "goal-mismatch")
    ];
    supports.forEach((evidence, index) => {
      const episode = repos.runtime.getEpisode(evidence.episodeId)!;
      enqueue(repos, {
        jobType: "span_credit",
        userId: episode.userId,
        episodeId: episode.id,
        payload: {
          pathId: evidence.path.id,
          pathHash: evidence.path.pathHash,
          rewardHash: spanCreditRewardHash(episode)
        },
        createdAt: `2026-08-20T04:0${index}:00.000Z`
      }, 30 + index);
    });

    await service.runWorkerOnce(20);
    for (const _support of supports) await service.runWorkerOnce(1);

    const clusters = repos.proceduralSpanClusters.listByNamespace(
      PROCEDURAL_POLICY_TEST_NAMESPACE,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    );
    expect(clusters).toHaveLength(3);
    expect(clusters.map((cluster) => cluster.memberCount).sort()).toEqual([1, 1, 2]);
    const sharedCluster = clusters.find((cluster) => cluster.memberCount === 2)!;
    expect(sharedCluster).toMatchObject({
      status: "ready",
      memberCount: 2,
      distinctSupportEpisodeCount: 2,
      clusterBasis: {
        membershipSimilarityBasis: "current-cluster-center-capability-goal-normalized-procedure",
        views: ["capability_goal", "state_contract", "procedure_semantic"],
        matchingViews: ["capability_goal", "procedure_semantic"]
      }
    });
    expect(repos.proceduralSpanClusters.listMembers(sharedCluster.id))
      .toEqual(expect.arrayContaining([
        expect.objectContaining({ similarity: 1 }),
        expect.objectContaining({ similarity: 1 })
      ]));
    expect(embeddedTexts.filter((text) => text.startsWith("Capability goal:"))).toHaveLength(4);
    expect(embeddedTexts.filter((text) => text.startsWith("State contract:"))).toHaveLength(4);
    expect(embeddedTexts.filter((text) => text.startsWith("Procedure semantics:"))).toHaveLength(4);
    db.close();
  });

  it("assigns against the maintained center instead of requiring similarity to every member", async () => {
    const embeddedTexts: string[] = [];
    const { db, service } = createTestService({
      skillLlm: automaticPipelineLlm([]),
      embedder: evolvingCenterTestEmbedder(embeddedTexts),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          },
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            proceduralSimilarityThreshold: 0.85,
            goalSimilarityThreshold: 0.85,
            policySimilarityThreshold: 0.85,
            minDistinctSources: 2
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const supports = [
      persistProceduralPolicyEvidence(repos, "center-support-a"),
      persistProceduralPolicyEvidence(repos, "center-support-b"),
      persistProceduralPolicyEvidence(repos, "center-support-c")
    ];
    supports.forEach((evidence, index) => {
      const episode = repos.runtime.getEpisode(evidence.episodeId)!;
      enqueue(repos, {
        jobType: "span_credit",
        userId: episode.userId,
        episodeId: episode.id,
        payload: {
          pathId: evidence.path.id,
          pathHash: evidence.path.pathHash,
          rewardHash: spanCreditRewardHash(episode)
        },
        createdAt: `2026-08-20T05:0${index}:00.000Z`
      }, 10 + index);
    });

    await service.runWorkerOnce(20);
    await service.runWorkerOnce(1);
    await service.runWorkerOnce(1);
    await service.runWorkerOnce(1);

    const clusters = repos.proceduralSpanClusters.listByNamespace(
      PROCEDURAL_POLICY_TEST_NAMESPACE,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    );
    expect(clusters).toHaveLength(1);
    expect(clusters[0]).toMatchObject({
      status: "ready",
      memberCount: 3,
      distinctSupportEpisodeCount: 3,
      clusterBasis: {
        assignmentMode: "incremental-cluster-center",
        membershipSimilarityBasis: "current-cluster-center-capability-goal-normalized-procedure",
        centerEmbedding: {
          version: "procedural-span-cluster-center.v2",
          supportMemberCount: 3
        }
      }
    });
    expect(repos.proceduralSpanClusters.listMembers(clusters[0]!.id)).toHaveLength(3);
    expect(embeddedTexts).toHaveLength(9);
    db.close();
  });

  it("shrinks a legacy-scope cluster in its own namespace during user-scope replay", async () => {
    const legacyNamespace = `${PROCEDURAL_POLICY_TEST_NAMESPACE}:project-a:codex:default`;
    const embeddedTexts: string[] = [];
    const { db, service } = createTestService({
      skillLlm: automaticPipelineLlm([]),
      embedder: semanticTestEmbedder(embeddedTexts),
      config: {
        ...DEFAULT_MEMMY_CONFIG,
        algorithm: {
          ...DEFAULT_MEMMY_CONFIG.algorithm,
          capture: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
            embedAfterCapture: false
          },
          spanClustering: {
            ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
            enabled: true,
            proceduralSimilarityThreshold: 0.9,
            goalSimilarityThreshold: 0.9,
            policySimilarityThreshold: 0.9,
            minDistinctSources: 2
          }
        }
      }
    });
    const repos = new Repositories(db.db);
    const legacyA = persistProceduralPolicyEvidence(repos, "legacy-scope-a", {
      namespaceId: legacyNamespace
    });
    const legacyB = persistProceduralPolicyEvidence(repos, "legacy-scope-b", {
      namespaceId: legacyNamespace
    });
    for (const [index, evidence] of [legacyA, legacyB].entries()) {
      const episode = repos.runtime.getEpisode(evidence.episodeId)!;
      enqueue(repos, {
        jobType: "span_credit",
        userId: episode.userId,
        episodeId: episode.id,
        payload: {
          pathId: evidence.path.id,
          pathHash: evidence.path.pathHash,
          rewardHash: spanCreditRewardHash(episode)
        },
        createdAt: `2026-08-20T06:0${index}:00.000Z`
      }, 20 + index);
    }
    await service.runWorkerOnce(10);
    await service.runWorkerOnce(1);
    await service.runWorkerOnce(1);
    const legacyCluster = repos.proceduralSpanClusters.listByNamespace(
      legacyNamespace,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    )[0]!;
    expect(legacyCluster).toMatchObject({ status: "ready", memberCount: 2 });

    const replacementPath = buildProceduralPolicyPath({
      suffix: "user-scope-replay-a",
      episodeId: legacyA.episodeId,
      rawTurnId: legacyA.rawTurnId,
      terminationStatus: "success"
    });
    const replacement = repos.proceduralPaths.save({
      path: replacementPath,
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      createdAt: "2026-08-20T06:10:00.000Z"
    });
    const occurrence = replacement.occurrences[0]!;
    const creditRun = buildEpisodeSpanCreditRun({
      episodeId: legacyA.episodeId,
      pathId: replacement.record.id,
      pathHash: replacement.record.pathHash,
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      rewardHash: "reward:user-scope-replay-a",
      episodeReward: 1,
      credits: [{
        occurrenceId: occurrence.id,
        spanId: occurrence.spanId,
        spanIndex: occurrence.spanIndex,
        preStateId: occurrence.preStateId,
        postStateId: occurrence.postStateId,
        rewardCredit: 1,
        attributionType: "helpful",
        confidence: 0.95,
        evidenceRole: "support",
        evidenceRefs: [occurrence.id],
        reason: "The replayed procedure completed successfully"
      }],
      scorerModel: "scope-migration-test"
    });
    repos.proceduralSpanCredits.saveAndActivate(
      creditRun,
      "2026-08-20T06:11:00.000Z"
    );
    enqueue(repos, {
      jobType: "procedural_span_cluster",
      userId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      episodeId: legacyA.episodeId,
      payload: {
        namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
        creditRunId: creditRun.id,
        pathId: replacement.record.id,
        algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
      },
      createdAt: "2026-08-20T06:12:00.000Z"
    }, 22);

    const replay = await service.runWorkerOnce(1);
    expect(replay.jobs).toContainEqual(expect.objectContaining({
      jobType: "procedural_span_cluster",
      status: "succeeded"
    }));
    expect(repos.proceduralSpanClusters.get(legacyCluster.id)).toMatchObject({
      namespaceId: legacyNamespace,
      status: "forming",
      memberCount: 1
    });
    expect(repos.proceduralSpanClusters.listMembers(legacyCluster.id)
      .map((member) => member.occurrenceId)).toEqual([legacyB.occurrence.id]);
    expect(repos.proceduralSpanClusters.listByNamespace(
      PROCEDURAL_POLICY_TEST_NAMESPACE,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    )).toContainEqual(expect.objectContaining({
      status: "forming",
      memberCount: 1,
      anchorOccurrenceId: occurrence.id
    }));
  });
});

function enqueue(repos: Repositories, input: EnqueueJobInput, index: number): void {
  const at = input.createdAt!;
  repos.runtime.enqueueJob({
    id: `job-auto-procedural-${index}`,
    jobType: input.jobType,
    status: "queued",
    dedupeKey: evolutionJobDedupeKey(input),
    userId: input.userId,
    sessionId: input.sessionId,
    episodeId: input.episodeId,
    targetMemoryId: input.targetMemoryId,
    payload: input.payload ?? {},
    attempts: 0,
    maxAttempts: 3,
    createdAt: at,
    updatedAt: at
  });
}

function semanticTestEmbedder(seen: string[]): Embedder {
  const embed = (text: string): number[] => {
    if (text.startsWith("Capability goal:")) return [1, 0, 0];
    if (text.startsWith("Procedure semantics:")) return [0, 1, 0];
    return [0, 0, 1];
  };
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "procedural-semantic-test"
    },
    isRemote() { return false; },
    async embed(texts: string[]) {
      seen.push(...texts);
      return texts.map(embed);
    },
    async embedOne(text: string) {
      seen.push(text);
      return embed(text);
    },
    status() {
      return {
        provider: "local",
        model: "procedural-semantic-test",
        configured: true,
        remote: false
      };
    }
  };
}

function evolvingCenterTestEmbedder(seen: string[]): Embedder {
  const angles = [-30, 0, 15].map((degrees) => degrees * Math.PI / 180);
  let occurrenceIndex = 0;
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "procedural-center-test"
    },
    isRemote() { return false; },
    async embed(texts: string[]) {
      seen.push(...texts);
      if (texts.length % 3 !== 0) throw new Error("center test expects three views per occurrence");
      const vectors = texts.map((text, index) => {
        if (text.startsWith("State contract:")) return [0, 0, 1];
        const angle = angles[occurrenceIndex + Math.floor(index / 3)];
        if (angle === undefined) throw new Error("center test received too many occurrences");
        return [Math.cos(angle), Math.sin(angle), 0];
      });
      occurrenceIndex += texts.length / 3;
      return vectors;
    },
    async embedOne(text: string) {
      const [vector] = await this.embed([text, "Procedure semantics:", "State contract:"]);
      return vector!;
    },
    status() {
      return {
        provider: "local",
        model: "procedural-center-test",
        configured: true,
        remote: false
      };
    }
  };
}

function stateContractMismatchTestEmbedder(seen: string[]): Embedder {
  let occurrenceIndex = 0;
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: "local",
      model: "procedural-three-view-gate-test"
    },
    isRemote() { return false; },
    async embed(texts: string[]) {
      seen.push(...texts);
      if (texts.length % 3 !== 0) throw new Error("three-view test expects three views per occurrence");
      const vectors = texts.map((text, index) => {
        const itemIndex = occurrenceIndex + Math.floor(index / 3);
        if (text.startsWith("State contract:")) {
          return itemIndex === 0 ? [0, 0, 1] : [0, 0, -1];
        }
        if (text.startsWith("Capability goal:")) {
          return itemIndex === 3 ? [-1, 0, 0] : [1, 0, 0];
        }
        return itemIndex === 2 ? [0, -1, 0] : [0, 1, 0];
      });
      occurrenceIndex += texts.length / 3;
      return vectors;
    },
    async embedOne(text: string) {
      const [vector] = await this.embed([text, "Procedure semantics:", "State contract:"]);
      return vector!;
    },
    status() {
      return {
        provider: "local",
        model: "procedural-three-view-gate-test",
        configured: true,
        remote: false
      };
    }
  };
}

function automaticPipelineLlm(operations: string[]): LlmClient {
  return {
    config: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: "host",
      endpoint: "http://127.0.0.1/automatic-procedural-test",
      model: "automatic-procedural-test"
    },
    isConfigured() { return true; },
    async complete() { return "{}"; },
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: { operation: string }
    ): Promise<T> {
      operations.push(options.operation);
      if (options.operation.startsWith("span_credit.score.v2")) {
        const payload = JSON.parse(messages.find((message) =>
          message.role === "user" && message.content.includes("episode_reward")
        )?.content ?? "{}") as {
          episode_reward: number;
          spans: Array<{ occurrence_id: string }>;
        };
        const occurrenceId = payload.spans[0]!.occurrence_id;
        return {
          span_credits: [{
            occurrence_id: occurrenceId,
            reward_credit: payload.episode_reward,
            attribution_type: payload.episode_reward > 0 ? "helpful" : "harmful",
            confidence: 0.9,
            evidence_refs: [occurrenceId],
            reason: payload.episode_reward > 0
              ? "The repair completed with focused verification."
              : "The repair remained blocked after focused verification."
          }]
        } as unknown as T;
      }
      const payload = JSON.parse(messages.find((message) =>
        message.role === "user" && message.content.includes("selected_path_evidence")
      )?.content ?? "{}") as {
        selected_path_evidence: Array<{
          occurrence_id: string;
          evidence_role: "support" | "counterexample";
          steps: Array<{ step_id: string }>;
        }>;
      };
      const supports = payload.selected_path_evidence.filter((item) => item.evidence_role === "support");
      const counterexamples = payload.selected_path_evidence
        .filter((item) => item.evidence_role === "counterexample");
      return {
        title: "Diagnose, repair, and verify a dependency conflict",
        goal_pattern: "Turn a reproducible dependency failure into a verified focused build",
        trigger_conditions: ["A focused build exposes a dependency constraint conflict."],
        procedure_steps: [{
          instruction: "Inspect and reproduce the focused dependency failure.",
          evidence_refs: [supports[0]!.occurrence_id, supports[0]!.steps[0]!.step_id]
        }, {
          instruction: "Correct the constraint and rerun the focused build.",
          evidence_refs: [supports[1]!.occurrence_id, supports[1]!.steps[1]!.step_id]
        }],
        recovery_rules: [{
          condition: "The edit exposes a separate platform compatibility blocker.",
          action: "Surface the new blocker instead of claiming the dependency repair succeeded.",
          evidence_refs: [counterexamples[0]!.occurrence_id, counterexamples[0]!.steps[1]!.step_id]
        }],
        verification_steps: [{
          check: "Rerun the exact focused build used for reproduction.",
          success_signal: "The focused build passes.",
          evidence_refs: [supports[0]!.steps[1]!.step_id, supports[1]!.steps[1]!.step_id]
        }],
        apply_when: ["The failure isolates a dependency constraint conflict."],
        do_not_apply_when: ["The failure is an unrelated platform compatibility blocker."],
        invariants: ["Use the same focused verification target before and after the edit."],
        expected_effect: "The dependency conflict is resolved with explicit verification.",
        evidence_occurrence_ids: supports.map((item) => item.occurrence_id),
        confidence: 0.9
      } as unknown as T;
    },
    status() {
      return {
        provider: "host",
        model: "automatic-procedural-test",
        configured: true,
        remote: true
      };
    }
  };
}
