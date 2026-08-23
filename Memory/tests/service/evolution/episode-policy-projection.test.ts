import { afterEach, describe, expect, it } from "vitest";
import {
  EpisodePolicyProjectionPipeline,
  PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
  buildEpisodePolicyProjection,
  buildProceduralPolicy,
  type ProceduralPolicyVersionRecord
} from "../../../src/index.js";
import { Repositories } from "../../../src/storage/repositories.js";
import type { ProceduralSpanClusterRecord } from "../../../src/storage/procedural-span-cluster-repository.js";
import {
  PROCEDURAL_POLICY_TEST_NAMESPACE,
  persistProceduralPolicyEvidence,
  type PersistedProceduralPolicyEvidence
} from "../../fixtures/procedural-policy-fixture.js";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const { cleanup, createTestService } = createMemoryServiceFixture();

afterEach(cleanup);

describe("Episode Policy Projection", () => {
  it("preserves one ordered node per Span occurrence without folding repeated Policies", () => {
    const projection = buildEpisodePolicyProjection({
      episodeId: "episode-ordered",
      pathId: "path-ordered",
      pathHash: "path-hash-ordered",
      nodes: [0, 1].map((spanIndex) => ({
        occurrenceId: `occurrence-${spanIndex}`,
        spanId: `span-${spanIndex}`,
        spanIndex,
        localGoal: `goal-${spanIndex}`,
        entryCondition: `entry-${spanIndex}`,
        exitCondition: `exit-${spanIndex}`,
        terminationStatus: "success" as const,
        preStateId: `state-${spanIndex}`,
        postStateId: `state-${spanIndex + 1}`,
        rawTurnIds: [`turn-${spanIndex}`],
        stepIds: [`step-${spanIndex}`],
        assignment: {
          kind: "policy" as const,
          policyVersionId: "policy-version-shared",
          policyKey: "policy:shared",
          clusterId: "cluster-shared",
          clusterMembershipVersion: "membership-shared",
          evidenceRole: "support" as const,
          matchConfidence: 0.95
        }
      }))
    });

    expect(projection.nodes).toHaveLength(2);
    expect(projection.nodes.map((node) => node.nodeIndex)).toEqual([0, 1]);
    expect(projection.nodes.map((node) => node.assignment.kind === "policy"
      ? node.assignment.policyVersionId
      : undefined)).toEqual(["policy-version-shared", "policy-version-shared"]);
    expect(projection.coverage).toEqual({
      totalSpanCount: 2,
      mappedSpanCount: 2,
      unmappedSpanCount: 0,
      mappingRatio: 1
    });
  });

  it("materializes mapped and explicit UNMAPPED projections idempotently", () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const supportA = persistProceduralPolicyEvidence(repos, "projection-support-a");
    const supportB = persistProceduralPolicyEvidence(repos, "projection-support-b");
    const unmapped = persistProceduralPolicyEvidence(repos, "projection-unmapped");
    const cluster = readyCluster(repos, "projection-cluster", [supportA, supportB]);
    const policy = activatePolicy(repos, cluster, [supportA, supportB]);
    const pipeline = new EpisodePolicyProjectionPipeline({ repos });

    const mapped = pipeline.projectPath(supportA.path.id, "2026-08-20T01:00:00.000Z");
    const replay = pipeline.projectPath(supportA.path.id, "2026-08-20T01:00:01.000Z");
    const noPolicy = pipeline.projectPath(unmapped.path.id, "2026-08-20T01:00:02.000Z");

    expect(mapped.created).toBe(true);
    expect(replay.created).toBe(false);
    expect(mapped.record.projection.nodes).toEqual([
      expect.objectContaining({
        nodeIndex: 0,
        occurrenceId: supportA.occurrence.id,
        assignment: {
          kind: "policy",
          policyVersionId: policy.id,
          policyKey: policy.policyKey,
          clusterId: cluster.id,
          clusterMembershipVersion: cluster.membershipVersion,
          evidenceRole: "support",
          matchConfidence: 0.96
        }
      })
    ]);
    expect(mapped.record.projection.coverage.mappingRatio).toBe(1);
    expect(noPolicy.record.projection.nodes[0]?.assignment).toEqual({
      kind: "unmapped",
      reason: "no_cluster_assignment"
    });
    expect(noPolicy.record.projection.coverage).toMatchObject({
      mappedSpanCount: 0,
      unmappedSpanCount: 1,
      mappingRatio: 0
    });
    expect(repos.episodePolicyProjections.listNodes(mapped.record.id))
      .toEqual(mapped.record.projection.nodes);
    expect(repos.episodePolicyProjections.listVersionsForEpisode(supportA.episodeId))
      .toHaveLength(1);
  });

  it("supersedes stale projections when cluster membership and Policy versions change", () => {
    const { db } = createTestService();
    const repos = new Repositories(db.db);
    const supportA = persistProceduralPolicyEvidence(repos, "projection-version-a");
    const supportB = persistProceduralPolicyEvidence(repos, "projection-version-b");
    const counterexample = persistProceduralPolicyEvidence(repos, "projection-version-counter", {
      terminationStatus: "blocked"
    });
    const cluster = readyCluster(repos, "projection-version-cluster", [supportA, supportB]);
    const initialPolicy = activatePolicy(repos, cluster, [supportA, supportB]);
    const pipeline = new EpisodePolicyProjectionPipeline({ repos });
    const initial = pipeline.projectPath(supportA.path.id, "2026-08-20T02:00:00.000Z");

    const changedCluster = repos.proceduralSpanClusters.upsert({
      id: cluster.id,
      namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
      algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
      minDistinctSupportEpisodes: 2,
      members: [
        { occurrenceId: supportA.occurrence.id, evidenceRole: "support", similarity: 0.96 },
        { occurrenceId: supportB.occurrence.id, evidenceRole: "support", similarity: 0.94 },
        { occurrenceId: counterexample.occurrence.id, evidenceRole: "counterexample", similarity: 0.82 }
      ],
      evidenceVersion: "projection-version-evidence-v2",
      at: "2026-08-20T02:00:01.000Z"
    });

    expect(repos.episodePolicyProjections.get(initial.record.id)?.status).toBe("inactive");
    expect(repos.proceduralPolicies.get(initialPolicy.id)?.status).toBe("inactive");
    const pending = pipeline.projectPath(supportA.path.id, "2026-08-20T02:00:02.000Z");
    expect(pending.record.projection.nodes[0]?.assignment).toMatchObject({
      kind: "unmapped",
      reason: "cluster_ready_policy_pending",
      clusterId: cluster.id,
      clusterMembershipVersion: changedCluster.membershipVersion
    });

    const changedPolicy = activatePolicy(repos, changedCluster, [
      supportA,
      supportB,
      counterexample
    ]);
    const remapped = pipeline.projectPath(supportA.path.id, "2026-08-20T02:00:03.000Z");

    expect(remapped.record.id).not.toBe(initial.record.id);
    expect(remapped.record.id).not.toBe(pending.record.id);
    expect(remapped.record.projection.nodes[0]?.assignment).toMatchObject({
      kind: "policy",
      policyVersionId: changedPolicy.id,
      clusterMembershipVersion: changedCluster.membershipVersion
    });
    expect(repos.episodePolicyProjections.listVersionsForEpisode(supportA.episodeId))
      .toHaveLength(3);
    expect(repos.episodePolicyProjections.getActiveForEpisode(supportA.episodeId)?.id)
      .toBe(remapped.record.id);
  });

  it("runs through the durable worker job without invoking an LLM", async () => {
    const { db, service } = createTestService();
    const repos = new Repositories(db.db);
    const evidence = persistProceduralPolicyEvidence(repos, "projection-worker");
    repos.runtime.enqueueJob({
      id: "job-episode-policy-projection-worker",
      jobType: "episode_policy_projection",
      status: "queued",
      dedupeKey: `episode_policy_projection:${evidence.episodeId}:${evidence.path.pathHash}`,
      userId: "user-procedural-policy",
      sessionId: evidence.sessionId,
      episodeId: evidence.episodeId,
      payload: {
        pathId: evidence.path.id,
        pathHash: evidence.path.pathHash,
        trigger: "test"
      },
      attempts: 0,
      maxAttempts: 3,
      createdAt: "2026-08-20T03:00:00.000Z",
      updatedAt: "2026-08-20T03:00:00.000Z"
    });

    const run = await service.runWorkerOnce(10);

    expect(run.jobs).toContainEqual(expect.objectContaining({
      jobId: "job-episode-policy-projection-worker",
      jobType: "episode_policy_projection",
      status: "succeeded"
    }));
    expect(repos.episodePolicyProjections.getActiveForEpisode(evidence.episodeId)?.projection
      .nodes[0]?.assignment).toEqual({
        kind: "unmapped",
        reason: "no_cluster_assignment"
      });
  });
});

function readyCluster(
  repos: Repositories,
  id: string,
  supports: readonly PersistedProceduralPolicyEvidence[]
): ProceduralSpanClusterRecord {
  return repos.proceduralSpanClusters.upsert({
    id,
    namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
    algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
    minDistinctSupportEpisodes: 2,
    members: supports.map((evidence, index) => ({
      occurrenceId: evidence.occurrence.id,
      evidenceRole: "support" as const,
      similarity: index === 0 ? 0.96 : 0.94
    })),
    evidenceVersion: `${id}-evidence-v1`,
    at: "2026-08-20T00:30:00.000Z"
  });
}

function activatePolicy(
  repos: Repositories,
  cluster: ProceduralSpanClusterRecord,
  evidence: readonly PersistedProceduralPolicyEvidence[]
): ProceduralPolicyVersionRecord {
  const support = evidence.filter((item) => item.path.terminalReward === 1);
  const counterexamples = evidence.filter((item) => item.path.terminalReward !== 1);
  const policy = buildProceduralPolicy({
    namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
    clusterId: cluster.id,
    clusterMembershipVersion: cluster.membershipVersion,
    draft: {
      title: "Repair and verify a dependency conflict",
      goalPattern: "Resolve a reproducible dependency conflict",
      triggerConditions: ["A focused build exposes a dependency constraint conflict"],
      procedureSteps: [{
        instruction: "Inspect the conflict, apply the constrained repair, and rerun the focused build",
        evidenceRefs: support.map((item) => item.occurrence.id)
      }],
      recoveryRules: [],
      verificationSteps: [{
        check: "Run the focused build",
        successSignal: "The dependency conflict is resolved",
        evidenceRefs: support.map((item) => item.occurrence.id)
      }],
      applyWhen: ["The dependency failure can be reproduced"],
      doNotApplyWhen: ["The failure has a different root cause"],
      invariants: ["Do not weaken unrelated dependency constraints"],
      expectedEffect: "The focused build passes without the dependency conflict",
      evidenceOccurrenceIds: support.map((item) => item.occurrence.id),
      confidence: 0.9
    },
    occurrenceIds: evidence.map((item) => item.occurrence.id),
    supportOccurrenceIds: support.map((item) => item.occurrence.id),
    counterexampleOccurrenceIds: counterexamples.map((item) => item.occurrence.id),
    supportEpisodeIds: support.map((item) => item.episodeId),
    counterexampleEpisodeIds: counterexamples.map((item) => item.episodeId),
    pathIds: evidence.map((item) => item.path.id),
    spanIds: evidence.map((item) => item.occurrence.spanId),
    sessionIds: evidence.map((item) => item.sessionId),
    model: "projection-test"
  });
  const memberByOccurrence = new Map(repos.proceduralSpanClusters
    .listMembers(cluster.id)
    .map((member) => [member.occurrenceId, member]));
  return repos.proceduralPolicies.saveAndActivate({
    policy,
    l2MemoryId: `projection-policy-memory-${policy.id}`,
    occurrences: evidence.map((item) => {
      const member = memberByOccurrence.get(item.occurrence.id)!;
      return {
        occurrenceId: item.occurrence.id,
        pathId: item.path.id,
        spanId: item.occurrence.spanId,
        episodeId: item.episodeId,
        sessionId: item.sessionId,
        evidenceRole: member.evidenceRole,
        matchConfidence: member.similarity
      };
    }),
    at: "2026-08-20T00:40:00.000Z"
  });
}
