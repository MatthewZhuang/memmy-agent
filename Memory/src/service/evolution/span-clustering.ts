import type { MemmyConfig } from "../../config/index.js";
import type { EvolutionJobRecord, Repositories } from "../../storage/repositories.js";
import type { MemoryRow } from "../../types.js";
import { memoryVector } from "../../storage/memory-vector-state.js";
import { SPAN_CLUSTER_ALGORITHM_VERSION as STORAGE_SPAN_CLUSTER_ALGORITHM_VERSION } from "../../storage/span-cluster-repository.js";
import type { SpanClusterStatus } from "../../storage/span-cluster-repository.js";
import { stableHash } from "../../utils/id.js";
import { spanPayload } from "./span-model.js";
import { namespaceForMemory } from "../namespace/namespace-scope.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";

export const SPAN_CLUSTER_ALGORITHM_VERSION = STORAGE_SPAN_CLUSTER_ALGORITHM_VERSION;

export interface ClusterableSpan {
  id: string;
  sourceTraceId: string;
  createdAt: string;
  vecGoal: number[];
  vecPolicy: number[];
}

export interface SpanPartitionMember {
  spanId: string;
  sourceTraceId: string;
  goalSimilarity: number;
  policySimilarity: number;
  vecGoal?: number[];
  vecPolicy?: number[];
}

export interface SpanPartitionCluster {
  id: string;
  status: SpanClusterStatus;
  promotedPolicyId?: string | null;
  goalCentroid: number[];
  policyCentroid: number[];
  goalThreshold: number;
  policyThreshold: number;
  memberCount: number;
  distinctSourceCount: number;
  membershipVersion: string;
  anchorSpanId: string;
  members: SpanPartitionMember[];
}

export interface SpanClusterAuditCandidate {
  clusterId: string;
  memberCount: number;
  pairCount: number;
  minPairScore: number;
  p10PairScore: number;
  weakestPairs: Array<{
    spanIds: [string, string];
    goalSimilarity: number;
    policySimilarity: number;
    pairScore: number;
  }>;
}

export interface SpanClusteringPipelineDeps {
  repos: Repositories;
  config: MemmyConfig;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

export class SpanClusteringPipeline {
  constructor(private readonly deps: SpanClusteringPipelineDeps) {}

  rebuildScope(job: EvolutionJobRecord): void {
    if (!this.deps.config.algorithm.spanClustering.enabled) return;
    const scopeId = typeof job.payload.scopeId === "string" ? job.payload.scopeId : undefined;
    if (!scopeId) throw new Error(`span_cluster missing scopeId: ${job.id}`);
    const algorithmVersion = typeof job.payload.algorithmVersion === "string"
      ? job.payload.algorithmVersion
      : SPAN_CLUSTER_ALGORITHM_VERSION;
    if (algorithmVersion !== SPAN_CLUSTER_ALGORITHM_VERSION) {
      throw new Error(`unsupported span cluster algorithm: ${algorithmVersion}`);
    }
    const config = this.deps.config.algorithm.spanClustering;
    const spans = this.deps.repos.memories.listReadySpanV2()
      .filter((memory) => namespaceIdFromMemory(memory) === scopeId)
      .map(clusterableSpanFromMemory)
      .filter((span): span is ClusterableSpan => Boolean(span));
    const promotedById = new Map(
      this.deps.repos.spanClusters
        .listByScope(scopeId, SPAN_CLUSTER_ALGORITHM_VERSION)
        .filter((cluster) => cluster.promotedPolicyId)
        .map((cluster) => [cluster.id, cluster])
    );
    const clusters = buildSpanPartition({
      scopeId,
      spans,
      goalThreshold: config.goalSimilarityThreshold,
      policyThreshold: config.policySimilarityThreshold,
      minDistinctSources: config.minDistinctSources
    }).map((cluster) => {
      const previous = promotedById.get(cluster.id);
      return previous?.promotedPolicyId
        ? {
          ...cluster,
          status: "promoted" as const,
          promotedPolicyId: previous.promotedPolicyId
        }
        : cluster;
    });
    const auditCandidates = findSpanClusterAuditCandidates(clusters, {
      auditMinMembers: config.auditMinMembers,
      auditCohesionThreshold: config.auditCohesionThreshold,
      auditWeakPairLimit: config.auditWeakPairLimit
    });
    this.deps.repos.transaction(() => {
      this.deps.repos.spanClusters.replaceScopePartition({
        scopeId,
        algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION,
        clusters,
        at: job.updatedAt
      });
      for (const cluster of clusters) {
        if (cluster.status === "promoted" && cluster.promotedPolicyId) {
          this.associateClusterMembers(cluster.id, cluster.promotedPolicyId, job.updatedAt);
          continue;
        }
        if (cluster.status !== "ready") continue;
        const auditCandidate = auditCandidates.find((candidate) => candidate.clusterId === cluster.id);
        if (auditCandidate) {
          this.deps.enqueueJob({
            jobType: "span_cluster_audit",
            userId: job.userId,
            sessionId: job.sessionId,
            episodeId: job.episodeId,
            dedupeKey: `span_cluster_audit:${cluster.id}:${cluster.membershipVersion}`,
            payload: {
              reason: "span_cluster.low_cohesion",
              clusterId: cluster.id,
              membershipVersion: cluster.membershipVersion,
              algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION,
              audit: auditCandidate
            },
            createdAt: job.updatedAt
          });
          continue;
        }
        this.deps.enqueueJob({
          jobType: "l2_induction",
          userId: job.userId,
          sessionId: job.sessionId,
          episodeId: job.episodeId,
          dedupeKey: `l2_induction:${cluster.id}:${cluster.membershipVersion}`,
          payload: {
            reason: "span_cluster.ready",
            clusterId: cluster.id,
            membershipVersion: cluster.membershipVersion,
            algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION
          },
          createdAt: job.updatedAt
        });
      }
    });
  }

  private associateClusterMembers(clusterId: string, policyId: string, at: string): void {
    const members = this.deps.repos.spanClusters.listMembers(clusterId);
    const spans = this.deps.repos.memories.getMany(members.map((member) => member.spanId));
    for (const span of spans) {
      this.deps.repos.runtime.insertTracePolicyLink({
        userId: span.userId,
        l1MemoryId: span.id,
        l2MemoryId: policyId,
        relation: "supports_span_cluster",
        strength: 1,
        createdAt: at
      });
    }
  }
}

interface WorkingCluster {
  anchorSpanId: string;
  goalCentroid: number[];
  policyCentroid: number[];
  members: Array<SpanPartitionMember & { vecGoal: number[]; vecPolicy: number[] }>;
}

export function buildSpanPartition(input: {
  scopeId: string;
  spans: readonly ClusterableSpan[];
  goalThreshold: number;
  policyThreshold: number;
  minDistinctSources?: number;
}): SpanPartitionCluster[] {
  const clusters: WorkingCluster[] = [];
  const spans = [...input.spans].sort(compareSpans);
  for (const span of spans) {
    const compatible = clusters
      .map((cluster) => {
        const goalSimilarity = cosine(span.vecGoal, cluster.goalCentroid);
        const policySimilarity = cosine(span.vecPolicy, cluster.policyCentroid);
        return { cluster, goalSimilarity, policySimilarity };
      })
      .filter((item) =>
        item.goalSimilarity >= input.goalThreshold &&
        item.policySimilarity >= input.policyThreshold
      )
      .sort((a, b) =>
        Math.min(b.goalSimilarity, b.policySimilarity) - Math.min(a.goalSimilarity, a.policySimilarity) ||
        b.goalSimilarity - a.goalSimilarity ||
        b.policySimilarity - a.policySimilarity ||
        clusterId(input.scopeId, a.cluster.anchorSpanId).localeCompare(clusterId(input.scopeId, b.cluster.anchorSpanId))
      );
    const winner = compatible[0];
    if (winner) {
      winner.cluster.members.push({
        spanId: span.id,
        sourceTraceId: span.sourceTraceId,
        goalSimilarity: roundSimilarity(winner.goalSimilarity),
        policySimilarity: roundSimilarity(winner.policySimilarity),
        vecGoal: span.vecGoal,
        vecPolicy: span.vecPolicy
      });
      winner.cluster.goalCentroid = centroid(winner.cluster.members.map((member) => member.vecGoal));
      winner.cluster.policyCentroid = centroid(winner.cluster.members.map((member) => member.vecPolicy));
      continue;
    }
    clusters.push({
      anchorSpanId: span.id,
      goalCentroid: normalize(span.vecGoal),
      policyCentroid: normalize(span.vecPolicy),
      members: [{
        spanId: span.id,
        sourceTraceId: span.sourceTraceId,
        goalSimilarity: 1,
        policySimilarity: 1,
        vecGoal: span.vecGoal,
        vecPolicy: span.vecPolicy
      }]
    });
  }

  return clusters.map((cluster) => {
    const members = cluster.members
      .sort((a, b) => a.spanId.localeCompare(b.spanId));
    const distinctSourceCount = new Set(members.map((member) => member.sourceTraceId)).size;
    const status: SpanClusterStatus = distinctSourceCount >= (input.minDistinctSources ?? Number.POSITIVE_INFINITY)
      ? "ready"
      : "forming";
    return {
      id: clusterId(input.scopeId, cluster.anchorSpanId),
      status,
      goalCentroid: cluster.goalCentroid,
      policyCentroid: cluster.policyCentroid,
      goalThreshold: input.goalThreshold,
      policyThreshold: input.policyThreshold,
      memberCount: members.length,
      distinctSourceCount,
      membershipVersion: stableHash({
        algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION,
        goalThreshold: input.goalThreshold,
        policyThreshold: input.policyThreshold,
        members: members.map((member) => ({
          spanId: member.spanId,
          goalSimilarity: member.goalSimilarity,
          policySimilarity: member.policySimilarity
        }))
      }).slice(0, 32),
      anchorSpanId: cluster.anchorSpanId,
      members
    };
  });
}

export function findSpanClusterAuditCandidates(
  clusters: readonly SpanPartitionCluster[],
  options: {
    auditMinMembers: number;
    auditCohesionThreshold: number;
    auditWeakPairLimit?: number;
  }
): SpanClusterAuditCandidate[] {
  const candidates: SpanClusterAuditCandidate[] = [];
  for (const cluster of clusters) {
    if (cluster.memberCount < options.auditMinMembers) continue;
    const pairs = pairwiseScores(cluster);
    if (pairs.length === 0) continue;
    const sorted = [...pairs].sort((a, b) => a.pairScore - b.pairScore);
    const p10Index = Math.max(0, Math.ceil(sorted.length * 0.1) - 1);
    const p10PairScore = sorted[p10Index]!.pairScore;
    if (p10PairScore >= options.auditCohesionThreshold) continue;
    candidates.push({
      clusterId: cluster.id,
      memberCount: cluster.memberCount,
      pairCount: sorted.length,
      minPairScore: sorted[0]!.pairScore,
      p10PairScore,
      weakestPairs: sorted.slice(0, options.auditWeakPairLimit ?? 5)
    });
  }
  return candidates;
}

function pairwiseScores(cluster: SpanPartitionCluster): SpanClusterAuditCandidate["weakestPairs"] {
  const pairs: SpanClusterAuditCandidate["weakestPairs"] = [];
  const members = cluster.members;
  for (let i = 0; i < members.length; i += 1) {
    for (let j = i + 1; j < members.length; j += 1) {
      const left = members[i]!;
      const right = members[j]!;
      if (!left.vecGoal || !right.vecGoal || !left.vecPolicy || !right.vecPolicy) continue;
      const goalSimilarity = roundSimilarity(cosine(left.vecGoal, right.vecGoal));
      const policySimilarity = roundSimilarity(cosine(left.vecPolicy, right.vecPolicy));
      pairs.push({
        spanIds: [left.spanId, right.spanId],
        goalSimilarity,
        policySimilarity,
        pairScore: roundSimilarity(Math.min(goalSimilarity, policySimilarity))
      });
    }
  }
  return pairs;
}

function compareSpans(a: ClusterableSpan, b: ClusterableSpan): number {
  return a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id);
}

function clusterableSpanFromMemory(memory: MemoryRow): ClusterableSpan | null {
  const span = spanPayload(memory);
  const vecGoal = memoryVector(memory, "vec_goal");
  const vecPolicy = memoryVector(memory, "vec_policy");
  if (!span || !vecGoal || !vecPolicy) return null;
  return {
    id: memory.id,
    sourceTraceId: span.source_trace_id,
    createdAt: memory.createdAt,
    vecGoal,
    vecPolicy
  };
}

function namespaceIdFromMemory(memory: MemoryRow): string {
  const namespace = namespaceForMemory(memory);
  return [
    namespace.tenantId,
    namespace.userId,
    namespace.projectId ?? namespace.workspaceId,
    namespace.source,
    namespace.profileId
  ].filter(Boolean).join(":");
}

function clusterId(scopeId: string, anchorSpanId: string): string {
  return `span_cluster_${SPAN_CLUSTER_ALGORITHM_VERSION}_${scopeId}_${anchorSpanId}`;
}

function cosine(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let a2 = 0;
  let b2 = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index]! * b[index]!;
    a2 += a[index]! * a[index]!;
    b2 += b[index]! * b[index]!;
  }
  if (a2 === 0 || b2 === 0) return 0;
  return dot / (Math.sqrt(a2) * Math.sqrt(b2));
}

function centroid(vectors: number[][]): number[] {
  const dimension = vectors[0]?.length ?? 0;
  if (dimension === 0) return [];
  const sum = Array.from({ length: dimension }, () => 0);
  for (const vector of vectors) {
    if (vector.length !== dimension) continue;
    for (let index = 0; index < dimension; index += 1) {
      sum[index] = (sum[index] ?? 0) + vector[index]!;
    }
  }
  return normalize(sum.map((value) => value / vectors.length));
}

function normalize(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return norm === 0 ? [...vector] : vector.map((value) => value / norm);
}

function roundSimilarity(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
