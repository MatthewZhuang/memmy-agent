import type {
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type { Embedder } from "../../model/types.js";
import type { IngestPolicySequenceProjectionResult } from
  "../../storage/policy-sequence-pattern-repository.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import {
  POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
  extractPolicySequencePatternOccurrences,
  mergePolicySequenceOccurrences,
  type PolicySequenceCostV1
} from "./policy-sequence-pattern-model.js";
import { EpisodeCapabilityDiscoveryPipeline } from "./episode-capability-discovery.js";

export interface PolicySequenceMiningPipelineDeps {
  repos: Repositories;
  embedder: Embedder;
}

export interface EnqueuePolicySequenceMiningDeps {
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

export class PolicySequenceMiningPipeline {
  private readonly episodeCapability: EpisodeCapabilityDiscoveryPipeline;

  constructor(private readonly deps: PolicySequenceMiningPipelineDeps) {
    this.episodeCapability = new EpisodeCapabilityDiscoveryPipeline(deps);
  }

  async mineJob(
    job: EvolutionJobRecord
  ): Promise<IngestPolicySequenceProjectionResult | undefined> {
    const projectionId = payloadText(job, "projectionId");
    if (!projectionId) {
      throw new Error(`Policy sequence mining job missing projectionId: ${job.id}`);
    }
    const projection = this.deps.repos.episodePolicyProjections.get(projectionId);
    if (!projection || projection.status !== "active") return undefined;
    const queuedProjectionHash = payloadText(job, "projectionHash");
    if (queuedProjectionHash && queuedProjectionHash !== projection.projectionHash) {
      return undefined;
    }
    return this.mineProjection(projection.id, job.updatedAt);
  }

  async mineProjection(
    projectionId: string,
    at: string
  ): Promise<IngestPolicySequenceProjectionResult> {
    const projection = this.deps.repos.episodePolicyProjections.get(projectionId);
    if (!projection || projection.status !== "active") {
      throw new Error(`Policy sequence mining requires an active Projection: ${projectionId}`);
    }
    const path = this.deps.repos.proceduralPaths.get(projection.pathId);
    if (!path || path.status !== "active" || path.pathHash !== projection.pathHash) {
      throw new Error(`Policy sequence mining source path is stale: ${projection.pathId}`);
    }
    const spanCosts = new Map<string, PolicySequenceCostV1>();
    for (const occurrence of this.deps.repos.proceduralPaths.listOccurrencesForPath(path.id)) {
      spanCosts.set(occurrence.id, { ...occurrence.span.cost });
    }
    const policyFirstOccurrences = extractPolicySequencePatternOccurrences({
      projection: projection.projection,
      namespaceId: projection.namespaceId,
      sessionId: projection.sessionId,
      ...(path.terminalReward === undefined ? {} : { terminalReward: path.terminalReward }),
      spanCostsByOccurrenceId: spanCosts
    });
    const episodeFirst = await this.episodeCapability.discover({ projection, at });
    const occurrences = mergePolicySequenceOccurrences([
      ...policyFirstOccurrences,
      ...episodeFirst.taskSkillOccurrences
    ]);
    return this.deps.repos.policySequencePatterns.ingestProjection({
      projection,
      occurrences,
      at
    });
  }
}

export function enqueuePolicySequenceMining(
  deps: EnqueuePolicySequenceMiningDeps,
  input: {
    projectionId: string;
    projectionHash: string;
    episodeId: string;
    userId: string;
    sessionId: string;
    at: string;
    trigger: string;
  }
): EvolutionJobRecord {
  return deps.enqueueJob({
    jobType: "policy_sequence_mining",
    userId: input.userId,
    sessionId: input.sessionId,
    episodeId: input.episodeId,
    payload: {
      projectionId: input.projectionId,
      projectionHash: input.projectionHash,
      algorithmVersion: POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
      trigger: input.trigger
    },
    createdAt: input.at
  });
}

function payloadText(job: EvolutionJobRecord, key: string): string | undefined {
  const value = job.payload[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}
