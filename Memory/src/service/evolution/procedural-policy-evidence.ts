import type {
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type { ProceduralSpanClusterRecord } from "../../storage/procedural-span-cluster-repository.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import { stableHash } from "../../utils/id.js";
import {
  SPAN_CREDIT_ALGORITHM_VERSION,
  SPAN_CREDIT_SCHEMA_VERSION
} from "./span-credit-model.js";

export const PROCEDURAL_POLICY_EVIDENCE_VERSION = "procedural-policy-evidence.v1" as const;

export interface ProceduralPolicyEvidenceCandidate {
  occurrenceId: string;
  similarity: number;
}

export interface ProceduralPolicyEvidencePipelineDeps {
  repos: Repositories;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

export interface RefreshProceduralPolicyEvidenceResult {
  cluster?: ProceduralSpanClusterRecord;
  inductionJob?: EvolutionJobRecord;
  excludedOccurrenceIds: string[];
}

export class ProceduralPolicyEvidencePipeline {
  constructor(private readonly deps: ProceduralPolicyEvidencePipelineDeps) {}

  refreshCluster(input: {
    clusterId: string;
    namespaceId: string;
    algorithmVersion?: string;
    candidates: ProceduralPolicyEvidenceCandidate[];
    minDistinctSupportEpisodes?: number;
    clusterBasis?: Record<string, unknown>;
    enqueueInduction?: boolean;
    at: string;
  }): RefreshProceduralPolicyEvidenceResult {
    const members: Array<{
      occurrenceId: string;
      evidenceRole: "support" | "counterexample";
      similarity: number;
    }> = [];
    const creditEvidence: Array<Record<string, unknown>> = [];
    const excludedOccurrenceIds: string[] = [];
    for (const candidate of input.candidates) {
      const credit = this.deps.repos.proceduralSpanCredits.getActiveCreditForOccurrence(candidate.occurrenceId);
      if (!credit || (credit.evidenceRole !== "support" && credit.evidenceRole !== "counterexample")) {
        excludedOccurrenceIds.push(candidate.occurrenceId);
        continue;
      }
      members.push({
        occurrenceId: candidate.occurrenceId,
        evidenceRole: credit.evidenceRole,
        similarity: candidate.similarity
      });
      creditEvidence.push({
        occurrenceId: credit.occurrenceId,
        creditId: credit.id,
        runId: credit.runId,
        evidenceRole: credit.evidenceRole,
        goalCredit: credit.goalCredit,
        processQuality: credit.processQuality,
        confidence: credit.confidence,
        creditScore: credit.creditScore
      });
    }
    if (members.length === 0) return { excludedOccurrenceIds };
    const creditEvidenceVersion = stableHash(creditEvidence.sort((left, right) =>
      String(left.occurrenceId).localeCompare(String(right.occurrenceId))
    ));
    const cluster = this.deps.repos.proceduralSpanClusters.upsert({
      id: input.clusterId,
      namespaceId: input.namespaceId,
      ...(input.algorithmVersion ? { algorithmVersion: input.algorithmVersion } : {}),
      minDistinctSupportEpisodes: input.minDistinctSupportEpisodes ?? 2,
      members,
      evidenceVersion: creditEvidenceVersion,
      clusterBasis: {
        ...(input.clusterBasis ?? {}),
        creditGoverned: true,
        policyEvidenceVersion: PROCEDURAL_POLICY_EVIDENCE_VERSION,
        spanCreditSchemaVersion: SPAN_CREDIT_SCHEMA_VERSION,
        spanCreditAlgorithmVersion: SPAN_CREDIT_ALGORITHM_VERSION,
        creditEvidenceVersion,
        excludedOccurrenceIds: [...excludedOccurrenceIds].sort()
      },
      at: input.at
    });
    if (cluster.status !== "ready" || input.enqueueInduction === false) {
      return { cluster, excludedOccurrenceIds };
    }
    const anchor = this.deps.repos.proceduralPaths.getOccurrence(cluster.anchorOccurrenceId);
    const path = anchor ? this.deps.repos.proceduralPaths.get(anchor.pathId) : undefined;
    if (!anchor || !path) throw new Error(`procedural Policy cluster anchor is missing: ${cluster.id}`);
    const inductionJob = this.deps.enqueueJob({
      jobType: "l2_induction",
      userId: path.userId,
      payload: {
        proceduralClusterId: cluster.id,
        membershipVersion: cluster.membershipVersion,
        evidenceVersion: PROCEDURAL_POLICY_EVIDENCE_VERSION
      },
      createdAt: input.at
    });
    return { cluster, inductionJob, excludedOccurrenceIds };
  }
}
