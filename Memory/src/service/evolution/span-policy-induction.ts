import type { MemmyConfig } from "../../config/index.js";
import type { EvolutionJobRecord, Repositories } from "../../storage/repositories.js";
import type { MemoryRow } from "../../types.js";
import { kindFromMemory } from "../../storage/repositories.js";
import { spanPayload } from "./span-model.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";

export const SPAN_POLICY_INDUCTION_VERSION = "span-policy.v1" as const;

export interface SpanPolicyInductionDeps {
  config: MemmyConfig;
  repos: Repositories;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  upsertEvolutionMemory(memory: MemoryRow): {
    memory: MemoryRow;
    created: boolean;
    previous?: MemoryRow;
  };
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
  enqueueChange(input: {
    memoryId: string;
    namespaceId?: string;
    kind: string;
    op: string;
    entityId: string;
    userId: string;
    changeType: string;
    before?: unknown;
    after: unknown;
    source: string;
    createdAt: string;
  }): void;
  namespaceIdFromMemory(memory: MemoryRow): string | undefined;
}

export class SpanPolicyInductionPipeline {
  constructor(private readonly deps: SpanPolicyInductionDeps) {}

  induce(job: EvolutionJobRecord): void {
    const clusterId = typeof job.payload.clusterId === "string" ? job.payload.clusterId : undefined;
    const membershipVersion = typeof job.payload.membershipVersion === "string"
      ? job.payload.membershipVersion
      : undefined;
    if (!clusterId || !membershipVersion) {
      throw new Error(`span policy induction missing cluster payload: ${job.id}`);
    }
    const cluster = this.deps.repos.spanClusters.get(clusterId);
    if (!cluster) throw new Error(`span cluster not found: ${clusterId}`);
    if (cluster.membershipVersion !== membershipVersion) return;
    if (cluster.promotedPolicyId) {
      this.associateMembers(cluster.id, cluster.promotedPolicyId, job.updatedAt);
      return;
    }
    const members = this.deps.repos.spanClusters.listMembers(cluster.id);
    const spans = this.deps.repos.memories.getMany(members.map((member) => member.spanId));
    const evidence = spans
      .map((memory) => ({ memory, span: spanPayload(memory) }))
      .filter((item): item is { memory: MemoryRow; span: NonNullable<ReturnType<typeof spanPayload>> } => Boolean(item.span))
      .sort((a, b) => a.memory.id.localeCompare(b.memory.id));
    if (evidence.length === 0) return;

    const sourceSpanIds = evidence.map((item) => item.memory.id);
    const sourceTraceIds = unique(evidence.map((item) => item.span.source_trace_id));
    const sourceEpisodeIds = unique(evidence
      .map((item) => item.span.episode_id)
      .filter((id): id is string => Boolean(id)));
    const title = mostCommon(evidence.map((item) => item.span.goal)) ?? "Span-derived policy";
    const observedPolicies = unique(evidence.map((item) => item.span.policy));
    const summaries = unique(evidence.map((item) => item.span.summary));
    const body = [
      `# ${title}`,
      "",
      "## Trigger",
      title,
      "",
      "## Procedure",
      observedPolicies.map((policy) => `- ${policy}`).join("\n"),
      "",
      "## Verification",
      summaries.map((summary) => `- ${summary}`).join("\n"),
      "",
      "## Boundary",
      "Apply when both the local goal and observed strategy match this Span cluster."
    ].join("\n");
    const anchor = evidence[0]!.memory;
    const key = `policy:span-cluster:${cluster.id}:${SPAN_POLICY_INDUCTION_VERSION}`;
    const policy = this.deps.buildMemory({
      userId: anchor.userId,
      conversationId: anchor.conversationId,
      sessionId: anchor.sessionId,
      agentId: anchor.agentId,
      appId: anchor.appId,
      projectId: stringField(anchor.info, "project_id"),
      profileId: stringField(anchor.info, "profile_id"),
      layer: "L2",
      kind: "policy",
      lifecycleStatus: "active",
      memoryType: "LongTermMemory",
      key,
      value: body,
      tags: ["policy", "span-cluster"],
      info: {
        title,
        support: sourceSpanIds.length,
        policy_confidence: 0.72,
        status: "active",
        source_span_ids: sourceSpanIds,
        source_trace_ids: sourceTraceIds,
        span_cluster_id: cluster.id
      },
      internal: {
        source: "worker.span_policy_induction.v1",
        plugin_algorithm: "span_policy.induction.v1",
        source_memory_ids: sourceSpanIds,
        source_span_ids: sourceSpanIds,
        source_trace_ids: sourceTraceIds,
        source_episode_ids: sourceEpisodeIds,
        span_cluster_id: cluster.id,
        cluster_algorithm_version: cluster.algorithmVersion,
        cluster_membership_version: cluster.membershipVersion,
        title,
        trigger: title,
        procedure: observedPolicies.join("\n"),
        verification: summaries.join("\n"),
        boundary: "Apply when both the local goal and observed strategy match this Span cluster.",
        support: sourceSpanIds.length,
        gain: 0,
        raw_gain: 0,
        policy_confidence: 0.72,
        status: "active",
        policy: {
          title,
          trigger: title,
          procedure: observedPolicies.join("\n"),
          verification: summaries.join("\n"),
          boundary: "Apply when both the local goal and observed strategy match this Span cluster.",
          support: sourceSpanIds.length,
          gain: 0,
          raw_gain: 0,
          policy_confidence: 0.72,
          status: "active",
          experience_type: "success_pattern",
          evidence_polarity: "positive",
          skill_eligible: true,
          induction_version: SPAN_POLICY_INDUCTION_VERSION,
          span_cluster_id: cluster.id,
          cluster_algorithm_version: cluster.algorithmVersion,
          cluster_membership_version: cluster.membershipVersion,
          source_span_ids: sourceSpanIds,
          source_trace_ids: sourceTraceIds,
          source_episode_ids: sourceEpisodeIds
        }
      },
      createdAt: job.updatedAt
    });
    const upsert = this.deps.upsertEvolutionMemory(policy);
    this.associateMembers(cluster.id, upsert.memory.id, job.updatedAt);
    this.deps.repos.spanClusters.markPromoted(cluster.id, membershipVersion, upsert.memory.id, job.updatedAt);
    this.deps.enqueueChange({
      memoryId: upsert.memory.id,
      namespaceId: this.deps.namespaceIdFromMemory(upsert.memory),
      kind: kindFromMemory(upsert.memory),
      op: upsert.created ? "created" : "updated",
      entityId: upsert.memory.id,
      userId: upsert.memory.userId,
      changeType: upsert.created ? "create" : "update",
      before: upsert.previous,
      after: upsert.memory,
      source: "worker.span_policy_induction.v1",
      createdAt: job.updatedAt
    });
    if (this.deps.config.algorithm.capture.embedAfterCapture) {
      this.deps.enqueueJob({
        jobType: "embedding",
        userId: upsert.memory.userId,
        sessionId: upsert.memory.sessionId,
        episodeId: job.episodeId,
        targetMemoryId: upsert.memory.id,
        payload: { reason: "span_policy.upserted" },
        createdAt: job.updatedAt
      });
    }
  }

  private associateMembers(clusterId: string, policyId: string, at: string): void {
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

function unique(values: string[]): string[] {
  return Array.from(new Set(values));
}

function mostCommon(values: string[]): string | undefined {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0];
}

function stringField(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}
