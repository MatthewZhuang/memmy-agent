import type { MemmyConfig } from "../../config/index.js";
import type { LlmClient } from "../../model/types.js";
import { memoryVector } from "../../storage/memory-vector-state.js";
import {
  SPAN_CLUSTER_ALGORITHM_VERSION,
  type ClusterableSpan,
  type SpanPartitionCluster,
  buildSpanPartition
} from "./span-clustering.js";
import type {
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type { SpanClusterInput } from "../../storage/span-cluster-repository.js";
import { spanPayload } from "./span-model.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";

export const SPAN_CLUSTER_AUDIT_VERSION = "span-cluster-audit.v1" as const;

export interface SpanClusterAuditDeps {
  config: MemmyConfig;
  repos: Repositories;
  skillLlm: LlmClient;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

export class SpanClusterAuditPipeline {
  constructor(private readonly deps: SpanClusterAuditDeps) {}

  async audit(job: EvolutionJobRecord): Promise<void> {
    const clusterId = typeof job.payload.clusterId === "string" ? job.payload.clusterId : undefined;
    const membershipVersion = typeof job.payload.membershipVersion === "string"
      ? job.payload.membershipVersion
      : undefined;
    if (!clusterId || !membershipVersion) {
      throw new Error(`span cluster audit missing payload: ${job.id}`);
    }
    const cluster = this.deps.repos.spanClusters.get(clusterId);
    if (!cluster) throw new Error(`span cluster not found: ${clusterId}`);
    if (cluster.membershipVersion !== membershipVersion) return;

    const members = this.deps.repos.spanClusters.listMembers(cluster.id);
    const spans = this.deps.repos.memories.getMany(members.map((member) => member.spanId));
    const spanRecords = spans
      .map((memory) => ({ memory, span: spanPayload(memory) }))
      .filter((item): item is { memory: typeof item.memory; span: NonNullable<ReturnType<typeof spanPayload>> } =>
        Boolean(item.span)
      )
      .sort((a, b) => a.memory.id.localeCompare(b.memory.id));
    const result = await this.deps.skillLlm.completeJson<{
      should_split?: boolean;
      reason?: string;
      groups?: Array<{ label?: string; span_ids?: string[] }>;
    }>([
      {
        role: "system",
        content: [
          "You audit Span policy buckets for semantic drift.",
          "Decide whether the bucket should be split into smaller reusable-policy groups.",
          "Use only the provided Span goals, observed policies, summaries, and weak similarity pairs.",
          "Return strict JSON with should_split, reason, and groups. groups must contain span_ids from the input only."
        ].join("\n")
      },
      {
        role: "user",
        content: JSON.stringify({
          cluster_id: cluster.id,
          membership_version: cluster.membershipVersion,
          audit: job.payload.audit,
          spans: spanRecords.map((item) => ({
            span_id: item.memory.id,
            goal: item.span.goal,
            policy: item.span.policy,
            summary: item.span.summary
          }))
        })
      }
    ], {
      operation: "span_cluster.audit.v1",
      thinkingMode: this.deps.config.evolution.enableThinking ? "enabled" : "disabled",
      jsonMode: true
    });

    const sanitized = sanitizeAuditResult(result, spanRecords.map((item) => item.memory.id));
    this.deps.repos.runtime.setKv(
      `span_cluster_audit:${cluster.id}:${cluster.membershipVersion}`,
      serializeAuditResult(sanitized),
      job.updatedAt
    );
    if (sanitized.shouldSplit && sanitized.groups.length > 0) {
      const splitClusters = splitClusterByAuditGroups(cluster, spanRecords, sanitized.groups, this.deps.config);
      if (splitClusters.length > 0) {
        const retainedClusters = this.deps.repos.spanClusters
          .listByScope(cluster.scopeId, SPAN_CLUSTER_ALGORITHM_VERSION)
          .filter((item) => item.id !== cluster.id)
          .map((item) => clusterRecordToInput(item, this.deps.repos));
        this.deps.repos.spanClusters.replaceScopePartition({
          scopeId: cluster.scopeId,
          algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION,
          clusters: [...retainedClusters, ...splitClusters.map(clusterToInput)],
          at: job.updatedAt
        });
        for (const splitCluster of splitClusters) {
          if (splitCluster.status !== "ready") continue;
          enqueueInduction(this.deps.enqueueJob, job, splitCluster);
        }
        return;
      }
    }
    if (!sanitized.shouldSplit) {
      enqueueInduction(this.deps.enqueueJob, job, cluster);
    }
  }
}

interface SanitizedAuditResult {
  shouldSplit: boolean;
  reason: string;
  groups: Array<{ label: string; spanIds: string[] }>;
}

function sanitizeAuditResult(
  result: {
    should_split?: boolean;
    reason?: string;
    groups?: Array<{ label?: string; span_ids?: string[] }>;
  },
  validSpanIds: string[]
): SanitizedAuditResult {
  const valid = new Set(validSpanIds);
  const groups = Array.isArray(result.groups)
    ? result.groups.map((group) => ({
      label: typeof group.label === "string" ? group.label : "",
      spanIds: Array.isArray(group.span_ids)
        ? group.span_ids.filter((id) => typeof id === "string" && valid.has(id))
        : []
    })).filter((group) => group.spanIds.length > 0)
    : [];
  return {
    shouldSplit: result.should_split === true,
    reason: typeof result.reason === "string" ? result.reason : "",
    groups
  };
}

function serializeAuditResult(result: SanitizedAuditResult): Record<string, unknown> {
  return {
    audit_version: SPAN_CLUSTER_AUDIT_VERSION,
    should_split: result.shouldSplit,
    reason: result.reason,
    groups: result.groups.map((group) => ({
      label: group.label,
      span_ids: group.spanIds
    }))
  };
}

function splitClusterByAuditGroups(
  cluster: {
    scopeId: string;
    goalThreshold: number;
    policyThreshold: number;
  },
  spanRecords: Array<{ memory: Parameters<typeof memoryVector>[0]; span: NonNullable<ReturnType<typeof spanPayload>> }>,
  groups: Array<{ spanIds: string[] }>,
  config: MemmyConfig
): SpanPartitionCluster[] {
  const recordsById = new Map(spanRecords.map((record) => [record.memory.id, record]));
  const assigned = new Set<string>();
  const partitions: SpanPartitionCluster[] = [];
  for (const group of groups) {
    const members = group.spanIds
      .filter((spanId) => !assigned.has(spanId))
      .map((spanId) => recordsById.get(spanId))
      .filter((record): record is NonNullable<typeof spanRecords[number]> => Boolean(record));
    if (members.length === 0) continue;
    members.forEach((member) => assigned.add(member.memory.id));
    const clusterable = members
      .map((member): ClusterableSpan | null => {
        const vecGoal = memoryVector(member.memory, "vec_goal");
        const vecPolicy = memoryVector(member.memory, "vec_policy");
        if (!vecGoal || !vecPolicy) return null;
        return {
          id: member.memory.id,
          sourceTraceId: member.span.source_trace_id,
          createdAt: member.memory.createdAt,
          vecGoal,
          vecPolicy
        };
      })
      .filter((item): item is ClusterableSpan => Boolean(item));
    partitions.push(...buildSpanPartition({
      scopeId: cluster.scopeId,
      spans: clusterable,
      goalThreshold: cluster.goalThreshold,
      policyThreshold: cluster.policyThreshold,
      minDistinctSources: config.algorithm.spanClustering.minDistinctSources
    }));
  }
  for (const record of spanRecords) {
    if (assigned.has(record.memory.id)) continue;
    const vecGoal = memoryVector(record.memory, "vec_goal");
    const vecPolicy = memoryVector(record.memory, "vec_policy");
    if (!vecGoal || !vecPolicy) continue;
    partitions.push(...buildSpanPartition({
      scopeId: cluster.scopeId,
      spans: [{
        id: record.memory.id,
        sourceTraceId: record.span.source_trace_id,
        createdAt: record.memory.createdAt,
        vecGoal,
        vecPolicy
      }],
      goalThreshold: cluster.goalThreshold,
      policyThreshold: cluster.policyThreshold,
      minDistinctSources: config.algorithm.spanClustering.minDistinctSources
    }));
  }
  return partitions;
}

function clusterToInput(cluster: SpanPartitionCluster): SpanClusterInput {
  return {
    id: cluster.id,
    status: cluster.status,
    goalCentroid: cluster.goalCentroid,
    policyCentroid: cluster.policyCentroid,
    goalThreshold: cluster.goalThreshold,
    policyThreshold: cluster.policyThreshold,
    memberCount: cluster.memberCount,
    distinctSourceCount: cluster.distinctSourceCount,
    membershipVersion: cluster.membershipVersion,
    promotedPolicyId: cluster.promotedPolicyId,
    anchorSpanId: cluster.anchorSpanId,
    members: cluster.members.map((member) => ({
      spanId: member.spanId,
      sourceTraceId: member.sourceTraceId,
      goalSimilarity: member.goalSimilarity,
      policySimilarity: member.policySimilarity
    }))
  };
}

function clusterRecordToInput(
  cluster: {
    id: string;
    status: "forming" | "ready" | "promoted" | "stale";
    goalCentroid: number[];
    policyCentroid: number[];
    goalThreshold: number;
    policyThreshold: number;
    memberCount: number;
    distinctSourceCount: number;
    membershipVersion: string;
    promotedPolicyId?: string | null;
    anchorSpanId: string;
  },
  repos: Repositories
): SpanClusterInput {
  return {
    id: cluster.id,
    status: cluster.status,
    goalCentroid: cluster.goalCentroid,
    policyCentroid: cluster.policyCentroid,
    goalThreshold: cluster.goalThreshold,
    policyThreshold: cluster.policyThreshold,
    memberCount: cluster.memberCount,
    distinctSourceCount: cluster.distinctSourceCount,
    membershipVersion: cluster.membershipVersion,
    promotedPolicyId: cluster.promotedPolicyId,
    anchorSpanId: cluster.anchorSpanId,
    members: repos.spanClusters.listMembers(cluster.id).map((member) => ({
      spanId: member.spanId,
      sourceTraceId: member.sourceTraceId,
      goalSimilarity: member.goalSimilarity,
      policySimilarity: member.policySimilarity
    }))
  };
}

function enqueueInduction(
  enqueueJob: SpanClusterAuditDeps["enqueueJob"],
  job: EvolutionJobRecord,
  cluster: { id: string; membershipVersion: string }
): void {
  enqueueJob({
    jobType: "l2_induction",
    userId: job.userId,
    sessionId: job.sessionId,
    episodeId: job.episodeId,
    dedupeKey: `l2_induction:${cluster.id}:${cluster.membershipVersion}`,
    payload: {
      reason: "span_cluster.audit_accepted",
      clusterId: cluster.id,
      membershipVersion: cluster.membershipVersion,
      algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION
    },
    createdAt: job.updatedAt
  });
}
