import type { MemmyConfig } from "../../config/index.js";
import type { Embedder } from "../../model/types.js";
import type {
  EvolutionJobRecord,
  Repositories
} from "../../storage/repositories.js";
import type { ProceduralSpanCreditRecord } from "../../storage/procedural-span-credit-repository.js";
import {
  proceduralSpanProcedureText,
  type EpisodeProceduralPathRecord,
  type ProceduralSpanOccurrenceRecord
} from "../../storage/procedural-path-repository.js";
import type { ProceduralSpanOccurrenceEmbeddingRecord } from "../../storage/procedural-span-embedding-repository.js";
import { stableHash } from "../../utils/id.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import {
  ProceduralPolicyEvidencePipeline,
  type RefreshProceduralPolicyEvidenceResult
} from "./procedural-policy-evidence.js";
import { enqueueEpisodePolicyProjection } from "./episode-policy-projection.js";

export const PROCEDURAL_SPAN_EMBEDDING_VERSION = "procedural-span-embedding.v5" as const;
export const PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION = "procedural-span-semantic-cluster.v8" as const;
const CLUSTER_CENTER_BASIS_VERSION = "procedural-span-cluster-center.v2" as const;
const MAX_COUNTER_BACKFILL_CANDIDATES = 64;
const EMBEDDING_READ_BATCH_SIZE = 500;

export interface ProceduralSpanSemanticCandidate {
  occurrence: ProceduralSpanOccurrenceRecord;
  credit: ProceduralSpanCreditRecord;
  embedding: ProceduralSpanOccurrenceEmbeddingRecord;
}

export interface ProceduralSpanSemanticClusterMember {
  occurrenceId: string;
  evidenceRole: "support" | "counterexample";
  similarity: number;
  goalSimilarity: number;
  stateContractSimilarity: number;
  procedureSimilarity: number;
}

export interface ProceduralSpanSemanticCluster {
  id: string;
  anchorOccurrenceId: string;
  members: ProceduralSpanSemanticClusterMember[];
}

export interface ProceduralSpanSemanticClusteringResult {
  namespaceId: string;
  creditRunId: string;
  clusterCount: number;
  eligibleOccurrenceCount: number;
  embeddedOccurrenceCount: number;
  skippedAsStale: boolean;
  affectedClusterIds: string[];
  evidenceResults: RefreshProceduralPolicyEvidenceResult[];
}

export interface ProceduralSpanSemanticClusteringDeps {
  repos: Repositories;
  config: MemmyConfig;
  embedder: Embedder;
  enqueueJob(input: EnqueueJobInput): EvolutionJobRecord;
}

interface MutableCluster {
  anchor: ProceduralSpanSemanticCandidate;
  support: ProceduralSpanSemanticCandidate[];
  members: ProceduralSpanSemanticClusterMember[];
}

interface PairSimilarity {
  goal: number;
  stateContract: number;
  procedure: number;
  combined: number;
}

interface ClusterCenterEmbedding {
  goalVector: number[];
  stateContractVector: number[];
  procedureVector: number[];
}

export class ProceduralSpanSemanticClusteringPipeline {
  private readonly evidence: ProceduralPolicyEvidencePipeline;

  constructor(private readonly deps: ProceduralSpanSemanticClusteringDeps) {
    this.evidence = new ProceduralPolicyEvidencePipeline({
      repos: deps.repos,
      enqueueJob: deps.enqueueJob
    });
  }

  async ingestCreditRun(job: EvolutionJobRecord): Promise<ProceduralSpanSemanticClusteringResult | undefined> {
    if (!this.deps.config.algorithm.spanClustering.enabled) return undefined;
    const creditRunId = payloadText(job, "creditRunId");
    if (!creditRunId) return undefined;
    const run = this.deps.repos.proceduralSpanCredits.get(creditRunId);
    if (!run || run.status !== "active") {
      return staleResult(payloadText(job, "namespaceId") ?? "unknown", creditRunId);
    }
    const namespaceId = run.namespaceId;
    const queuedNamespace = payloadText(job, "namespaceId");
    if (queuedNamespace && queuedNamespace !== namespaceId) {
      return staleResult(queuedNamespace, creditRunId);
    }
    const path = this.deps.repos.proceduralPaths.get(run.pathId);
    if (!path || path.status !== "active" || path.episodeId !== run.episodeId) {
      return staleResult(namespaceId, creditRunId);
    }
    const occurrenceById = new Map(this.deps.repos.proceduralPaths
      .listOccurrencesForPath(run.pathId)
      .map((occurrence) => [occurrence.id, occurrence]));
    const credits = this.deps.repos.proceduralSpanCredits.listCredits(run.id);
    const eligibleCredits = credits.filter(isEligibleCredit);
    const eligibleOccurrences = eligibleCredits.map((credit) => occurrenceById.get(credit.occurrenceId))
      .filter((occurrence): occurrence is ProceduralSpanOccurrenceRecord => Boolean(occurrence));
    if (eligibleOccurrences.length !== eligibleCredits.length) {
      throw new Error(`procedural Span cluster run has incomplete occurrences: ${run.id}`);
    }
    const embeddings = await this.ensureEmbeddings(eligibleOccurrences, job.updatedAt);
    const latestRun = this.deps.repos.proceduralSpanCredits.get(creditRunId);
    const latestPath = this.deps.repos.proceduralPaths.get(run.pathId);
    if (!latestRun || latestRun.status !== "active" || !latestPath || latestPath.status !== "active") {
      return staleResult(namespaceId, creditRunId);
    }

    const evidenceResults: RefreshProceduralPolicyEvidenceResult[] = [];
    const affected = this.deps.repos.transaction(() => {
      const activeRun = this.deps.repos.proceduralSpanCredits.get(creditRunId);
      const activePath = this.deps.repos.proceduralPaths.get(run.pathId);
      if (!activeRun || activeRun.status !== "active" || !activePath || activePath.status !== "active") {
        return undefined;
      }
      const touched = new Set<string>();
      const impactedEpisodeIds = new Set<string>([run.episodeId]);
      const noteClusterEpisodes = (clusterId: string): void => {
        for (const member of this.deps.repos.proceduralSpanClusters.listMembers(clusterId)) {
          impactedEpisodeIds.add(member.episodeId);
        }
      };
      const activeOccurrenceIds = new Set(occurrenceById.keys());
      const eligibleByOccurrence = new Map(eligibleCredits.map((credit) => [credit.occurrenceId, credit]));
      const episodeMembers = this.deps.repos.proceduralSpanClusters.listMembersForEpisode(
        run.episodeId,
        PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
      );
      const detachIds = episodeMembers.filter((member) => {
        const credit = eligibleByOccurrence.get(member.occurrenceId);
        return !activeOccurrenceIds.has(member.occurrenceId) ||
          !credit || credit.evidenceRole !== member.evidenceRole;
      }).map((member) => member.occurrenceId);
      for (const clusterId of this.deps.repos.proceduralSpanClusters.removeMembers(
        detachIds,
        PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
      )) touched.add(clusterId);
      for (const member of episodeMembers) {
        if (!detachIds.includes(member.occurrenceId) && activeOccurrenceIds.has(member.occurrenceId)) {
          touched.add(member.clusterId);
        }
      }
      for (const clusterId of [...touched]) {
        noteClusterEpisodes(clusterId);
        const clusterNamespace = this.deps.repos.proceduralSpanClusters.get(clusterId)
          ?.namespaceId ?? namespaceId;
        const refreshed = this.refreshCluster(
          clusterId,
          clusterNamespace,
          job.updatedAt,
          false
        );
        if (refreshed) evidenceResults.push(refreshed);
      }

      const ordered = eligibleCredits.map((credit) => ({
        credit,
        occurrence: occurrenceById.get(credit.occurrenceId)!,
        embedding: embeddings.get(credit.occurrenceId)!
      })).sort((left, right) => roleOrder(left.credit.evidenceRole) - roleOrder(right.credit.evidenceRole) ||
        left.occurrence.spanIndex - right.occurrence.spanIndex ||
        left.occurrence.id.localeCompare(right.occurrence.id));
      const supportClusterIds = new Set<string>();
      for (const candidate of ordered) {
        const existing = this.deps.repos.proceduralSpanClusters.getMemberForOccurrence(
          candidate.occurrence.id,
          PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
        );
        if (existing) continue;
        const clusterId = this.assignOccurrence(candidate, namespaceId, job.updatedAt, evidenceResults);
        if (!clusterId) continue;
        touched.add(clusterId);
        if (candidate.credit.evidenceRole === "support") supportClusterIds.add(clusterId);
      }
      if (supportClusterIds.size > 0) {
        for (const clusterId of this.attachPreviouslyUnassignedCounterexamples(
          namespaceId,
          [...supportClusterIds],
          job.updatedAt,
          evidenceResults
        )) touched.add(clusterId);
      }
      for (const clusterId of [...touched].sort()) {
        noteClusterEpisodes(clusterId);
        const clusterNamespace = this.deps.repos.proceduralSpanClusters.get(clusterId)
          ?.namespaceId ?? namespaceId;
        const refreshed = this.refreshCluster(
          clusterId,
          clusterNamespace,
          job.updatedAt,
          true
        );
        if (refreshed) evidenceResults.push(refreshed);
        noteClusterEpisodes(clusterId);
      }
      return {
        clusterIds: [...touched].sort(),
        episodeIds: [...impactedEpisodeIds].sort()
      };
    });
    if (!affected) return staleResult(namespaceId, creditRunId);
    for (const episodeId of affected.episodeIds) {
      enqueueEpisodePolicyProjection({
        repos: this.deps.repos,
        enqueueJob: this.deps.enqueueJob
      }, {
        episodeId,
        at: job.updatedAt,
        trigger: "procedural_span_cluster_updated"
      });
    }
    return {
      namespaceId,
      creditRunId,
      clusterCount: this.deps.repos.proceduralSpanClusters.listByNamespace(
        namespaceId,
        PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
      ).filter((cluster) => cluster.status !== "stale").length,
      eligibleOccurrenceCount: eligibleOccurrences.length,
      embeddedOccurrenceCount: embeddings.size,
      skippedAsStale: false,
      affectedClusterIds: affected.clusterIds,
      evidenceResults
    };
  }

  private assignOccurrence(
    candidate: ProceduralSpanSemanticCandidate,
    namespaceId: string,
    at: string,
    evidenceResults: RefreshProceduralPolicyEvidenceResult[]
  ): string | undefined {
    const match = this.bestClusterMatch(candidate.embedding, namespaceId);
    if (!match && candidate.credit.evidenceRole !== "support") return undefined;
    const clusterId = match?.clusterId ?? `procedural_span_cluster_${stableHash({
      algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
      namespaceId,
      anchorOccurrenceId: candidate.occurrence.id
    }).slice(0, 24)}`;
    const existingMembers = match
      ? this.deps.repos.proceduralSpanClusters.listMembers(clusterId)
      : [];
    const refreshed = this.refreshCandidates({
      clusterId,
      namespaceId,
      candidates: [
        ...existingMembers.map((member) => ({
          occurrenceId: member.occurrenceId,
          similarity: member.similarity
        })),
        {
          occurrenceId: candidate.occurrence.id,
          similarity: match?.similarity.combined ?? 1
        }
      ],
      at,
      enqueueInduction: false
    });
    evidenceResults.push(refreshed);
    return refreshed.cluster?.id;
  }

  private bestClusterMatch(
    candidate: ProceduralSpanOccurrenceEmbeddingRecord,
    namespaceId: string,
    candidateClusterIds?: ReadonlySet<string>
  ): { clusterId: string; similarity: PairSimilarity } | undefined {
    const settings = this.deps.config.algorithm.spanClustering;
    const clusters = this.deps.repos.proceduralSpanClusters.listByNamespace(
      namespaceId,
      PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
    ).filter((cluster) => cluster.status !== "stale" && cluster.memberCount > 0)
      .filter((cluster) => !candidateClusterIds || candidateClusterIds.has(cluster.id));
    const matches = clusters.map((cluster) => {
      const center = clusterCenterFromBasis(
        cluster.clusterBasis,
        embeddingProviderName(this.deps.embedder),
        embeddingModelName(this.deps.embedder)
      );
      return center ? { clusterId: cluster.id, similarity: centerSimilarity(candidate, center) } : undefined;
    }).filter((item): item is NonNullable<typeof item> => Boolean(item))
      .filter(({ similarity }) => passesGates(
        similarity,
        settings.proceduralSimilarityThreshold
      ))
      .sort((left, right) => right.similarity.combined - left.similarity.combined ||
        left.clusterId.localeCompare(right.clusterId));
    return matches[0];
  }

  private attachPreviouslyUnassignedCounterexamples(
    namespaceId: string,
    candidateClusterIds: readonly string[],
    at: string,
    evidenceResults: RefreshProceduralPolicyEvidenceResult[]
  ): string[] {
    const credits = this.deps.repos.proceduralSpanCredits.listActiveCreditsForNamespace(namespaceId)
      .filter((credit) => credit.evidenceRole === "counterexample")
      .filter((credit) => !this.deps.repos.proceduralSpanClusters.getMemberForOccurrence(
        credit.occurrenceId,
        PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION
      ));
    const embeddings = this.currentEmbeddings(credits.map((credit) => credit.occurrenceId));
    const touched = new Set<string>();
    const allowedClusters = new Set(candidateClusterIds);
    const ranked = credits.map((credit) => {
      const embedding = embeddings.get(credit.occurrenceId);
      if (!embedding) return undefined;
      const match = this.bestClusterMatch(embedding, namespaceId, allowedClusters);
      return match ? { credit, embedding, match } : undefined;
    }).filter((item): item is NonNullable<typeof item> => Boolean(item))
      .sort((left, right) => right.match.similarity.combined - left.match.similarity.combined ||
        left.credit.occurrenceId.localeCompare(right.credit.occurrenceId))
      .slice(0, MAX_COUNTER_BACKFILL_CANDIDATES);
    for (const { credit, match } of ranked) {
      const occurrence = this.deps.repos.proceduralPaths.getOccurrence(credit.occurrenceId);
      if (!occurrence) continue;
      const members = this.deps.repos.proceduralSpanClusters.listMembers(match.clusterId);
      const refreshed = this.refreshCandidates({
        clusterId: match.clusterId,
        namespaceId,
        candidates: [
          ...members.map((member) => ({
            occurrenceId: member.occurrenceId,
            similarity: member.similarity
          })),
          { occurrenceId: occurrence.id, similarity: match.similarity.combined }
        ],
        at,
        enqueueInduction: false
      });
      evidenceResults.push(refreshed);
      if (refreshed.cluster) touched.add(refreshed.cluster.id);
    }
    return [...touched];
  }

  private refreshCluster(
    clusterId: string,
    namespaceId: string,
    at: string,
    enqueueInduction: boolean
  ): RefreshProceduralPolicyEvidenceResult | undefined {
    const members = this.deps.repos.proceduralSpanClusters.listMembers(clusterId);
    const governed = members.map((member) => ({
      member,
      credit: this.deps.repos.proceduralSpanCredits.getActiveCreditForOccurrence(member.occurrenceId)
    })).filter(({ credit }) => credit?.evidenceRole === "support" || credit?.evidenceRole === "counterexample");
    if (!governed.some(({ credit }) => credit?.evidenceRole === "support")) {
      this.deps.repos.proceduralSpanClusters.clearAndMarkStale(clusterId, at);
      return undefined;
    }
    return this.refreshCandidates({
      clusterId,
      namespaceId,
      candidates: governed.map(({ member }) => ({
        occurrenceId: member.occurrenceId,
        similarity: member.similarity
      })),
      at,
      enqueueInduction
    });
  }

  private refreshCandidates(input: {
    clusterId: string;
    namespaceId: string;
    candidates: Array<{ occurrenceId: string; similarity: number }>;
    at: string;
    enqueueInduction: boolean;
  }): RefreshProceduralPolicyEvidenceResult {
    const settings = this.deps.config.algorithm.spanClustering;
    const governed = input.candidates.map((candidate) => ({
      candidate,
      credit: this.deps.repos.proceduralSpanCredits.getActiveCreditForOccurrence(candidate.occurrenceId)
    })).filter(({ credit }) => credit?.evidenceRole === "support" || credit?.evidenceRole === "counterexample");
    const embeddings = this.currentEmbeddings(governed.map(({ candidate }) => candidate.occurrenceId));
    if (embeddings.size !== governed.length) {
      throw new Error(`procedural Span cluster center has incomplete embeddings: ${input.clusterId}`);
    }
    const supportEmbeddings = governed
      .filter(({ credit }) => credit?.evidenceRole === "support")
      .map(({ candidate }) => embeddings.get(candidate.occurrenceId)!);
    if (supportEmbeddings.length === 0) {
      throw new Error(`procedural Span cluster center has no support occurrence: ${input.clusterId}`);
    }
    const center = averageEmbeddingCenter(supportEmbeddings);
    const candidates = governed.map(({ candidate }) => ({
      occurrenceId: candidate.occurrenceId,
      similarity: centerSimilarity(embeddings.get(candidate.occurrenceId)!, center).combined
    }));
    return this.evidence.refreshCluster({
      clusterId: input.clusterId,
      namespaceId: input.namespaceId,
      algorithmVersion: PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION,
      candidates,
      minDistinctSupportEpisodes: Math.max(2, settings.minDistinctSources),
      clusterBasis: {
        discoveredAutomatically: true,
        assignmentMode: "incremental-cluster-center",
        membershipSimilarityBasis: "current-cluster-center-capability-goal-normalized-procedure",
        embeddingVersion: PROCEDURAL_SPAN_EMBEDDING_VERSION,
        embeddingProvider: embeddingProviderName(this.deps.embedder),
        embeddingModel: embeddingModelName(this.deps.embedder),
        similarityThreshold: settings.proceduralSimilarityThreshold,
        views: ["capability_goal", "state_contract", "procedure_semantic"],
        matchingViews: ["capability_goal", "procedure_semantic"],
        maxCounterBackfillCandidates: MAX_COUNTER_BACKFILL_CANDIDATES,
        centerEmbedding: {
          version: CLUSTER_CENTER_BASIS_VERSION,
          embeddingProvider: embeddingProviderName(this.deps.embedder),
          embeddingModel: embeddingModelName(this.deps.embedder),
          supportMemberCount: supportEmbeddings.length,
          goalVector: center.goalVector,
          stateContractVector: center.stateContractVector,
          procedureVector: center.procedureVector
        }
      },
      enqueueInduction: input.enqueueInduction,
      at: input.at
    });
  }

  private currentEmbeddings(
    occurrenceIds: readonly string[]
  ): Map<string, ProceduralSpanOccurrenceEmbeddingRecord> {
    const byOccurrence = new Map<string, ProceduralSpanOccurrenceEmbeddingRecord>();
    const ids = [...new Set(occurrenceIds)];
    for (let offset = 0; offset < ids.length; offset += EMBEDDING_READ_BATCH_SIZE) {
      const records = this.deps.repos.proceduralSpanEmbeddings.listByOccurrenceIds(
        ids.slice(offset, offset + EMBEDDING_READ_BATCH_SIZE),
        {
          embeddingVersion: PROCEDURAL_SPAN_EMBEDDING_VERSION,
          embeddingProvider: embeddingProviderName(this.deps.embedder),
          embeddingModel: embeddingModelName(this.deps.embedder)
        }
      );
      for (const record of records) {
        if (!byOccurrence.has(record.occurrenceId)) byOccurrence.set(record.occurrenceId, record);
      }
    }
    return byOccurrence;
  }

  private async ensureEmbeddings(
    occurrences: readonly ProceduralSpanOccurrenceRecord[],
    at: string
  ): Promise<Map<string, ProceduralSpanOccurrenceEmbeddingRecord>> {
    const provider = embeddingProviderName(this.deps.embedder);
    const model = embeddingModelName(this.deps.embedder);
    const records = new Map<string, ProceduralSpanOccurrenceEmbeddingRecord>();
    const pathCache = new Map<string, EpisodeProceduralPathRecord>();
    const missing: Array<{
      occurrence: ProceduralSpanOccurrenceRecord;
      sourceHash: string;
      texts: [string, string, string];
    }> = [];
    for (const occurrence of occurrences) {
      let path = pathCache.get(occurrence.pathId);
      if (!path) {
        path = this.deps.repos.proceduralPaths.get(occurrence.pathId);
        if (!path) throw new Error(`procedural Span embedding path is missing: ${occurrence.pathId}`);
        pathCache.set(occurrence.pathId, path);
      }
      const texts = embeddingTexts(
        occurrence,
        proceduralSpanProcedureText(path.path, occurrence.span)
      );
      const sourceHash = stableHash({
        embeddingVersion: PROCEDURAL_SPAN_EMBEDDING_VERSION,
        projectionVersion: occurrence.projection.version,
        texts
      });
      const existing = this.deps.repos.proceduralSpanEmbeddings.getByBasis({
        occurrenceId: occurrence.id,
        projectionVersion: occurrence.projection.version,
        embeddingVersion: PROCEDURAL_SPAN_EMBEDDING_VERSION,
        sourceHash,
        embeddingProvider: provider,
        embeddingModel: model
      });
      if (existing) records.set(occurrence.id, existing);
      else missing.push({ occurrence, sourceHash, texts });
    }
    if (missing.length === 0) return records;
    const vectors = await this.deps.embedder.embed(missing.flatMap((item) => item.texts), "document");
    if (vectors.length !== missing.length * 3) {
      throw new Error(`procedural Span embedding returned ${vectors.length} vectors for ${missing.length * 3} inputs`);
    }
    for (let index = 0; index < missing.length; index += 1) {
      const item = missing[index]!;
      const goalVector = vectors[index * 3];
      const procedureVector = vectors[index * 3 + 1];
      const stateContractVector = vectors[index * 3 + 2];
      if (!goalVector || !procedureVector || !stateContractVector) {
        throw new Error(`procedural Span embedding output is incomplete: ${item.occurrence.id}`);
      }
      const saved = this.deps.repos.proceduralSpanEmbeddings.save({
        occurrenceId: item.occurrence.id,
        namespaceId: item.occurrence.namespaceId,
        projectionVersion: item.occurrence.projection.version,
        embeddingVersion: PROCEDURAL_SPAN_EMBEDDING_VERSION,
        sourceHash: item.sourceHash,
        embeddingProvider: provider,
        embeddingModel: model,
        goalVector,
        procedureVector,
        // The repository column keeps its historical name for SQLite compatibility.
        effectVector: stateContractVector,
        at
      });
      records.set(item.occurrence.id, saved.record);
    }
    return records;
  }
}

/**
 * Deterministic full-partition oracle for offline rebuilds and tests only.
 * The online worker uses ingestCreditRun and never calls this O(N^2) routine.
 */
export function buildProceduralSpanSemanticPartition(input: {
  namespaceId: string;
  candidates: readonly ProceduralSpanSemanticCandidate[];
  similarityThreshold: number;
  algorithmVersion?: string;
}): ProceduralSpanSemanticCluster[] {
  const algorithmVersion = input.algorithmVersion ?? PROCEDURAL_SPAN_SEMANTIC_CLUSTER_VERSION;
  const supports = input.candidates
    .filter((candidate) => candidate.credit.evidenceRole === "support")
    .sort(compareCandidate);
  const counterexamples = input.candidates
    .filter((candidate) => candidate.credit.evidenceRole === "counterexample")
    .sort(compareCandidate);
  const clusters: MutableCluster[] = [];
  for (const candidate of supports) {
    const matches = clusters.map((cluster) => ({
      cluster,
      similarity: centerSimilarity(candidate.embedding, averageEmbeddingCenter(
        cluster.support.map((support) => support.embedding)
      ))
    })).filter(({ similarity }) => passesGates(
      similarity,
      input.similarityThreshold
    )).sort((left, right) => right.similarity.combined - left.similarity.combined ||
      left.cluster.anchor.occurrence.id.localeCompare(right.cluster.anchor.occurrence.id));
    const selected = matches[0];
    if (!selected) {
      clusters.push({
        anchor: candidate,
        support: [candidate],
        members: [member(candidate, { goal: 1, stateContract: 1, procedure: 1, combined: 1 })]
      });
      continue;
    }
    selected.cluster.support.push(candidate);
    selected.cluster.members.push(member(candidate, selected.similarity));
  }
  for (const candidate of counterexamples) {
    const matches = clusters.map((cluster) => ({
      cluster,
      similarity: centerSimilarity(candidate.embedding, averageEmbeddingCenter(
        cluster.support.map((support) => support.embedding)
      ))
    })).filter(({ similarity }) => passesGates(
      similarity,
      input.similarityThreshold
    )).sort((left, right) => right.similarity.combined - left.similarity.combined ||
      left.cluster.anchor.occurrence.id.localeCompare(right.cluster.anchor.occurrence.id));
    const selected = matches[0];
    if (selected) selected.cluster.members.push(member(candidate, selected.similarity));
  }
  return clusters.map((cluster) => ({
    id: `procedural_span_cluster_${stableHash({
      algorithmVersion,
      namespaceId: input.namespaceId,
      anchorOccurrenceId: cluster.anchor.occurrence.id
    }).slice(0, 24)}`,
    anchorOccurrenceId: cluster.anchor.occurrence.id,
    members: [...cluster.members].sort((left, right) => left.occurrenceId.localeCompare(right.occurrenceId))
  })).sort((left, right) => left.id.localeCompare(right.id));
}

function centerSimilarity(
  candidate: ProceduralSpanOccurrenceEmbeddingRecord,
  center: ClusterCenterEmbedding
): PairSimilarity {
  const goal = cosineSimilarity(candidate.goalVector, center.goalVector);
  const stateContract = cosineSimilarity(candidate.effectVector, center.stateContractVector);
  const procedure = cosineSimilarity(candidate.procedureVector, center.procedureVector);
  return {
    goal,
    stateContract,
    procedure,
    // State contracts remain occurrence evidence for later Policy induction.
    // They do not define Policy identity: one state-conditioned Policy may be
    // observed under materially different entry and exit conditions.
    combined: Math.min(goal, procedure)
  };
}

function averageEmbeddingCenter(
  support: readonly ProceduralSpanOccurrenceEmbeddingRecord[]
): ClusterCenterEmbedding {
  if (support.length === 0) throw new Error("procedural Span cluster has no support member");
  return {
    goalVector: averageVectors(support.map((item) => item.goalVector)),
    stateContractVector: averageVectors(support.map((item) => item.effectVector)),
    procedureVector: averageVectors(support.map((item) => item.procedureVector))
  };
}

function averageVectors(vectors: readonly number[][]): number[] {
  const dimension = vectors[0]?.length ?? 0;
  if (dimension === 0 || vectors.some((vector) => vector.length !== dimension)) {
    throw new Error("procedural Span cluster center has incompatible dimensions");
  }
  const center = Array.from({ length: dimension }, () => 0);
  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) {
      const value = vector[index]!;
      if (!Number.isFinite(value)) throw new Error("procedural Span cluster center contains an invalid value");
      center[index] = center[index]! + value;
    }
  }
  return center.map((value) => roundVectorValue(value / vectors.length));
}

function clusterCenterFromBasis(
  basis: Record<string, unknown>,
  embeddingProvider: string,
  embeddingModel: string
): ClusterCenterEmbedding | undefined {
  const raw = basis.centerEmbedding;
  if (!isRecord(raw) ||
      raw.version !== CLUSTER_CENTER_BASIS_VERSION ||
      raw.embeddingProvider !== embeddingProvider ||
      raw.embeddingModel !== embeddingModel) return undefined;
  const goalVector = numberVector(raw.goalVector);
  const stateContractVector = numberVector(raw.stateContractVector);
  const procedureVector = numberVector(raw.procedureVector);
  if (!goalVector || !stateContractVector || !procedureVector ||
      goalVector.length !== stateContractVector.length ||
      goalVector.length !== procedureVector.length) return undefined;
  return { goalVector, stateContractVector, procedureVector };
}

function numberVector(value: unknown): number[] | undefined {
  if (!Array.isArray(value) || value.length === 0 ||
      value.some((item) => typeof item !== "number" || !Number.isFinite(item))) return undefined;
  return value as number[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  if (left.length === 0 || left.length !== right.length) {
    throw new Error("procedural Span embeddings have incompatible dimensions");
  }
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index]!;
    const rightValue = right[index]!;
    dot += leftValue * rightValue;
    leftNorm += leftValue * leftValue;
    rightNorm += rightValue * rightValue;
  }
  if (leftNorm === 0 || rightNorm === 0) return 0;
  return roundSimilarity(Math.max(0, Math.min(1, dot / Math.sqrt(leftNorm * rightNorm))));
}

function passesGates(similarity: PairSimilarity, threshold: number): boolean {
  return similarity.combined >= threshold;
}

function member(
  candidate: ProceduralSpanSemanticCandidate,
  similarity: PairSimilarity
): ProceduralSpanSemanticClusterMember {
  if (candidate.credit.evidenceRole !== "support" && candidate.credit.evidenceRole !== "counterexample") {
    throw new Error(`ineligible procedural Span evidence role: ${candidate.credit.evidenceRole}`);
  }
  return {
    occurrenceId: candidate.occurrence.id,
    evidenceRole: candidate.credit.evidenceRole,
    similarity: similarity.combined,
    goalSimilarity: similarity.goal,
    stateContractSimilarity: similarity.stateContract,
    procedureSimilarity: similarity.procedure
  };
}

function compareCandidate(left: ProceduralSpanSemanticCandidate, right: ProceduralSpanSemanticCandidate): number {
  return left.occurrence.createdAt.localeCompare(right.occurrence.createdAt) ||
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.spanIndex - right.occurrence.spanIndex ||
    left.occurrence.id.localeCompare(right.occurrence.id);
}

function embeddingTexts(
  occurrence: ProceduralSpanOccurrenceRecord,
  procedureText: string
): [string, string, string] {
  return [
    `Capability goal:\n${occurrence.capabilityGoal}`,
    `Procedure semantics:\n${procedureText}`,
    `State contract:\nEntry: ${occurrence.entryCondition}\nExit: ${occurrence.exitCondition}`
  ];
}

function embeddingProviderName(embedder: Embedder): string {
  return embedder.config.provider.trim() || "unknown";
}

function embeddingModelName(embedder: Embedder): string {
  return embedder.config.model?.trim() || "default";
}

function payloadText(job: EvolutionJobRecord, key: string): string | undefined {
  const value = job.payload[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function isEligibleCredit(credit: ProceduralSpanCreditRecord): boolean {
  return credit.evidenceRole === "support" || credit.evidenceRole === "counterexample";
}

function roleOrder(role: ProceduralSpanCreditRecord["evidenceRole"]): number {
  return role === "support" ? 0 : role === "counterexample" ? 1 : 2;
}

function roundSimilarity(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function roundVectorValue(value: number): number {
  return Math.round(value * 100_000_000) / 100_000_000;
}

function staleResult(namespaceId: string, creditRunId: string): ProceduralSpanSemanticClusteringResult {
  return {
    namespaceId,
    creditRunId,
    clusterCount: 0,
    eligibleOccurrenceCount: 0,
    embeddedOccurrenceCount: 0,
    skippedAsStale: true,
    affectedClusterIds: [],
    evidenceResults: []
  };
}
