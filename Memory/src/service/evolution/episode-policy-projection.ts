import type {
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type { ProceduralPolicyOccurrenceRecord } from "../../storage/procedural-policy-repository.js";
import type { ProceduralSpanOccurrenceRecord } from "../../storage/procedural-path-repository.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import {
  EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION,
  buildEpisodePolicyProjection,
  type EpisodePolicyProjectionAssignmentV1,
  type EpisodePolicyProjectionV1
} from "./episode-policy-projection-model.js";
import type {
  EpisodePolicyProjectionRecord,
  SaveEpisodePolicyProjectionResult
} from "../../storage/episode-policy-projection-repository.js";

const ACTIVE_PROCEDURAL_CLUSTER_ALGORITHM_VERSION = "procedural-span-semantic-cluster.v8";

export interface EpisodePolicyProjectionPipelineDeps {
  repos: Repositories;
}

export interface EnqueueEpisodePolicyProjectionDeps {
  repos: Repositories;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

export class EpisodePolicyProjectionPipeline {
  constructor(private readonly deps: EpisodePolicyProjectionPipelineDeps) {}

  projectJob(job: EvolutionJobRecord): SaveEpisodePolicyProjectionResult | undefined {
    if (!job.episodeId) {
      throw new Error(`Episode Policy Projection job missing episodeId: ${job.id}`);
    }
    const path = this.deps.repos.proceduralPaths.getActiveForEpisode(job.episodeId);
    if (!path) return undefined;
    const queuedPathId = payloadText(job, "pathId");
    const queuedPathHash = payloadText(job, "pathHash");
    if ((queuedPathId && queuedPathId !== path.id) ||
        (queuedPathHash && queuedPathHash !== path.pathHash)) {
      return undefined;
    }
    return this.projectPath(path.id, job.updatedAt);
  }

  projectPath(pathId: string, at: string): SaveEpisodePolicyProjectionResult {
    const path = this.deps.repos.proceduralPaths.get(pathId);
    if (!path || path.status !== "active") {
      throw new Error(`Episode Policy Projection requires an active path: ${pathId}`);
    }
    const occurrences = this.deps.repos.proceduralPaths.listOccurrencesForPath(path.id);
    const activeMappings = this.deps.repos.proceduralPolicies.listActiveOccurrencesForPath(path.id);
    const mappingsByOccurrence = groupMappings(activeMappings);
    const projection = buildEpisodePolicyProjection({
      episodeId: path.episodeId,
      pathId: path.id,
      pathHash: path.pathHash,
      nodes: occurrences.map((occurrence) => ({
        ...sourceNode(occurrence),
        assignment: this.assignmentForOccurrence(
          occurrence,
          mappingsByOccurrence.get(occurrence.id) ?? []
        )
      }))
    });
    return this.deps.repos.episodePolicyProjections.saveAndActivate({
      projection,
      userId: path.userId,
      sessionId: path.sessionId,
      namespaceId: path.namespaceId,
      at
    });
  }

  private assignmentForOccurrence(
    occurrence: ProceduralSpanOccurrenceRecord,
    activeMappings: readonly ProceduralPolicyOccurrenceRecord[]
  ): EpisodePolicyProjectionAssignmentV1 {
    const member = this.deps.repos.proceduralSpanClusters.getMemberForOccurrence(
      occurrence.id,
      ACTIVE_PROCEDURAL_CLUSTER_ALGORITHM_VERSION
    );
    if (!member) return { kind: "unmapped", reason: "no_cluster_assignment" };
    const cluster = this.deps.repos.proceduralSpanClusters.get(member.clusterId);
    if (!cluster || cluster.status === "stale") {
      return {
        kind: "unmapped",
        reason: "cluster_stale",
        clusterId: member.clusterId,
        ...(cluster ? {
          clusterMembershipVersion: cluster.membershipVersion,
          clusterStatus: cluster.status
        } : {})
      };
    }
    if (!cluster.activePolicyVersionId || cluster.status !== "promoted") {
      return {
        kind: "unmapped",
        reason: cluster.status === "forming"
          ? "cluster_forming"
          : "cluster_ready_policy_pending",
        clusterId: cluster.id,
        clusterMembershipVersion: cluster.membershipVersion,
        clusterStatus: cluster.status
      };
    }
    const mapping = activeMappings.find((candidate) =>
      candidate.policyVersionId === cluster.activePolicyVersionId &&
      candidate.clusterMembershipVersion === cluster.membershipVersion
    );
    const policy = mapping
      ? this.deps.repos.proceduralPolicies.get(mapping.policyVersionId)
      : undefined;
    if (!mapping || !policy || policy.status !== "active" || policy.clusterId !== cluster.id) {
      return {
        kind: "unmapped",
        reason: "active_policy_occurrence_missing",
        clusterId: cluster.id,
        clusterMembershipVersion: cluster.membershipVersion,
        clusterStatus: cluster.status
      };
    }
    return {
      kind: "policy",
      policyVersionId: policy.id,
      policyKey: policy.policyKey,
      clusterId: policy.clusterId,
      clusterMembershipVersion: policy.clusterMembershipVersion,
      evidenceRole: mapping.evidenceRole,
      matchConfidence: mapping.matchConfidence
    };
  }
}

export function enqueueEpisodePolicyProjection(
  deps: EnqueueEpisodePolicyProjectionDeps,
  input: {
    episodeId: string;
    at: string;
    trigger: string;
    policyVersionId?: string;
  }
): EvolutionJobRecord | undefined {
  const path = deps.repos.proceduralPaths.getActiveForEpisode(input.episodeId);
  const episode = deps.repos.runtime.getEpisode(input.episodeId);
  if (!path || !episode || path.userId !== episode.userId || path.sessionId !== episode.sessionId) {
    return undefined;
  }
  return deps.enqueueJob({
    jobType: "episode_policy_projection",
    userId: episode.userId,
    sessionId: episode.sessionId,
    episodeId: episode.id,
    payload: {
      pathId: path.id,
      pathHash: path.pathHash,
      algorithmVersion: EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION,
      trigger: input.trigger,
      ...(input.policyVersionId ? { policyVersionId: input.policyVersionId } : {})
    },
    createdAt: input.at
  });
}

export function activeEpisodePolicyProjection(
  repos: Repositories,
  episodeId: string
): EpisodePolicyProjectionRecord | undefined {
  return repos.episodePolicyProjections.getActiveForEpisode(episodeId);
}

function sourceNode(occurrence: ProceduralSpanOccurrenceRecord) {
  return {
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
    stepIds: [...occurrence.stepIds]
  };
}

function groupMappings(
  mappings: readonly ProceduralPolicyOccurrenceRecord[]
): Map<string, ProceduralPolicyOccurrenceRecord[]> {
  const grouped = new Map<string, ProceduralPolicyOccurrenceRecord[]>();
  for (const mapping of mappings) {
    const values = grouped.get(mapping.occurrenceId) ?? [];
    values.push(mapping);
    grouped.set(mapping.occurrenceId, values);
  }
  return grouped;
}

function payloadText(job: EvolutionJobRecord, key: string): string | undefined {
  const value = job.payload[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

export type { EpisodePolicyProjectionV1 };
