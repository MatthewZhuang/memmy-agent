import type { Embedder } from "../../model/types.js";
import type {
  EpisodeCapabilityAffinityRecord,
  EpisodeCapabilitySignatureRecord
} from "../../storage/episode-capability-repository.js";
import type { EpisodePolicyProjectionRecord } from
  "../../storage/episode-policy-projection-repository.js";
import type { Repositories } from "../../storage/repositories.js";
import { stableHash } from "../../utils/id.js";
import {
  alignPolicyBackbone,
  buildEpisodeCapabilitySignature,
  computeEpisodeCapabilityAffinity,
  type EpisodeCapabilityAffinityV1,
  type EpisodeCapabilitySignatureV1,
  type EpisodeCapabilityVectorsV1
} from "./episode-capability-model.js";
import {
  POLICY_SEQUENCE_MIN_LENGTH,
  extractEpisodeSimilarityPatternOccurrences,
  type EpisodeAffinityEvidenceV1,
  type PolicySequenceCostV1,
  type PolicySequencePatternOccurrenceV1
} from "./policy-sequence-pattern-model.js";

export interface EpisodeCapabilityDiscoveryResult {
  signature: EpisodeCapabilitySignatureRecord;
  affinities: EpisodeCapabilityAffinityRecord[];
  taskSkillOccurrences: PolicySequencePatternOccurrenceV1[];
}

export interface EpisodeCapabilityDiscoveryDeps {
  repos: Repositories;
  embedder: Embedder;
}

export class EpisodeCapabilityDiscoveryPipeline {
  constructor(private readonly deps: EpisodeCapabilityDiscoveryDeps) {}

  async discover(input: {
    projection: EpisodePolicyProjectionRecord;
    at: string;
  }): Promise<EpisodeCapabilityDiscoveryResult> {
    const path = this.deps.repos.proceduralPaths.get(input.projection.pathId);
    const episode = this.deps.repos.runtime.getEpisode(input.projection.episodeId);
    if (!path || path.status !== "active" || !episode || episode.status !== "closed" ||
        path.pathHash !== input.projection.pathHash ||
        path.namespaceId !== input.projection.namespaceId) {
      throw new Error(`Episode Capability discovery source is stale: ${input.projection.id}`);
    }
    const signature = buildEpisodeCapabilitySignature({
      projection: input.projection.projection,
      path: path.path,
      namespaceId: input.projection.namespaceId,
      ...(episode.title ? { episodeTitle: episode.title } : {}),
      ...(episode.summary ? { episodeSummary: episode.summary } : {}),
      rawTurns: this.deps.repos.runtime.listRawTurnsByEpisode(episode.id, 100)
    });
    const existing = this.deps.repos.episodeCapabilities.getByHash(signature.signatureHash);
    if (existing) {
      const active = existing.status === "active" ? existing :
        this.deps.repos.episodeCapabilities.saveAndActivate({
          signature: existing.signature,
          familyId: existing.familyId,
          vectors: existing.vectors,
          embeddingProvider: existing.embeddingProvider,
          embeddingModel: existing.embeddingModel,
          at: input.at
        }).record;
      return {
        signature: active,
        affinities: this.deps.repos.episodeCapabilities.listAffinitiesForSignature(
          active.signature.id
        ),
        taskSkillOccurrences: this.taskSkillOccurrences(active)
      };
    }

    const vectors = await this.embedSignature(signature);
    const peers = this.deps.repos.episodeCapabilities.listActiveCandidatesByPolicyOverlap({
      namespaceId: signature.namespaceId,
      policyKeys: signature.policyNodes.map((node) => node.policyKey),
      excludeEpisodeId: signature.episodeId,
      minSharedPolicyKeys: POLICY_SEQUENCE_MIN_LENGTH,
      limit: 64
    });
    const affinityDrafts = peers.map((peer) => computeEpisodeCapabilityAffinity({
      left: signature,
      leftVectors: vectors,
      right: peer.signature,
      rightVectors: peer.vectors
    }));
    const familyId = chooseEpisodeFamily(signature, peers, affinityDrafts);
    const saved = this.deps.repos.episodeCapabilities.saveAndActivate({
      signature,
      familyId,
      vectors,
      embeddingProvider: embeddingProviderName(this.deps.embedder),
      embeddingModel: embeddingModelName(this.deps.embedder),
      at: input.at
    }).record;
    const affinities = affinityDrafts.map((affinity) =>
      this.deps.repos.episodeCapabilities.saveAffinity(affinity, input.at)
    );
    return {
      signature: saved,
      affinities,
      taskSkillOccurrences: this.taskSkillOccurrences(saved)
    };
  }

  private taskSkillOccurrences(
    current: EpisodeCapabilitySignatureRecord
  ): PolicySequencePatternOccurrenceV1[] {
    const currentProjection = this.deps.repos.episodePolicyProjections.get(
      current.signature.projectionId
    );
    if (!currentProjection || currentProjection.status !== "active") return [];
    const affinityByPeer = new Map<string, EpisodeCapabilityAffinityV1>();
    for (const record of this.deps.repos.episodeCapabilities.listAffinitiesForSignature(
      current.signature.id
    )) {
      const affinity = record.affinity;
      const peerId = affinity.leftSignatureId === current.signature.id
        ? affinity.rightSignatureId
        : affinity.leftSignatureId;
      affinityByPeer.set(peerId, affinity);
    }
    const results: PolicySequencePatternOccurrenceV1[] = [];
    const family = this.deps.repos.episodeCapabilities
      .listActiveForFamily(current.familyId)
      .filter((peer) => peer.signature.id !== current.signature.id);
    for (const peer of family) {
      const affinity = affinityByPeer.get(peer.signature.id);
      if (!affinity?.familyEligible) continue;
      const peerProjection = this.deps.repos.episodePolicyProjections.get(
        peer.signature.projectionId
      );
      if (!peerProjection || peerProjection.status !== "active") continue;
      const alignment = alignPolicyBackbone(current.signature, peer.signature);
      if (alignment.policyKeys.length < POLICY_SEQUENCE_MIN_LENGTH) continue;

      const currentMappedCount = current.signature.policyNodes.length;
      const peerMappedCount = peer.signature.policyNodes.length;
      const fullyIdentical = alignment.policyKeys.length === currentMappedCount &&
          alignment.policyKeys.length === peerMappedCount &&
          contiguous(alignment.leftNodeIndexes) && contiguous(alignment.rightNodeIndexes);
      results.push(...this.occurrencesForSide({
        projection: currentProjection,
        signature: current.signature,
        matchedNodeIndexes: alignment.leftNodeIndexes,
        peer,
        affinity,
        patternCapabilityType: fullyIdentical ? "sub_skill" : "task_skill"
      }));
      results.push(...this.occurrencesForSide({
        projection: peerProjection,
        signature: peer.signature,
        matchedNodeIndexes: alignment.rightNodeIndexes,
        peer: current,
        affinity,
        patternCapabilityType: fullyIdentical ? "sub_skill" : "task_skill"
      }));
    }
    return results;
  }

  private occurrencesForSide(input: {
    projection: EpisodePolicyProjectionRecord;
    signature: EpisodeCapabilitySignatureV1;
    matchedNodeIndexes: readonly number[];
    peer: EpisodeCapabilitySignatureRecord;
    affinity: EpisodeCapabilityAffinityV1;
    patternCapabilityType: "task_skill" | "sub_skill";
  }): PolicySequencePatternOccurrenceV1[] {
    const path = this.deps.repos.proceduralPaths.get(input.projection.pathId);
    if (!path || path.status !== "active") return [];
    const costs = new Map<string, PolicySequenceCostV1>();
    for (const occurrence of this.deps.repos.proceduralPaths.listOccurrencesForPath(path.id)) {
      costs.set(occurrence.id, { ...occurrence.span.cost });
    }
    return extractEpisodeSimilarityPatternOccurrences({
      projection: input.projection.projection,
      namespaceId: input.projection.namespaceId,
      sessionId: input.projection.sessionId,
      episodeFamilyId: this.deps.repos.episodeCapabilities
        .get(input.signature.id)!.familyId,
      patternCapabilityType: input.patternCapabilityType,
      matchedNodeIndexes: input.matchedNodeIndexes,
      episodeAffinity: affinityEvidence(input.affinity, input.peer.signature),
      ...(path.terminalReward === undefined ? {} : { terminalReward: path.terminalReward }),
      spanCostsByOccurrenceId: costs
    });
  }

  private async embedSignature(
    signature: EpisodeCapabilitySignatureV1
  ): Promise<EpisodeCapabilityVectorsV1> {
    const vectors = await this.deps.embedder.embed([
      signature.goalText,
      signature.stateTransitionText,
      signature.outcomeText,
      signature.contextText
    ], "document");
    if (vectors.length !== 4) {
      throw new Error(`Episode Capability embedding count mismatch: ${signature.id}`);
    }
    return {
      goalVector: vectors[0]!,
      stateTransitionVector: vectors[1]!,
      outcomeVector: vectors[2]!,
      contextVector: vectors[3]!
    };
  }
}

function chooseEpisodeFamily(
  signature: EpisodeCapabilitySignatureV1,
  peers: readonly EpisodeCapabilitySignatureRecord[],
  affinities: readonly EpisodeCapabilityAffinityV1[]
): string {
  const affinityByPeerId = new Map<string, EpisodeCapabilityAffinityV1>();
  for (const affinity of affinities) {
    const peerId = affinity.leftSignatureId === signature.id
      ? affinity.rightSignatureId
      : affinity.leftSignatureId;
    affinityByPeerId.set(peerId, affinity);
  }
  const families = new Map<string, EpisodeCapabilitySignatureRecord[]>();
  for (const peer of peers) {
    const values = families.get(peer.familyId) ?? [];
    values.push(peer);
    families.set(peer.familyId, values);
  }
  const eligible = [...families.entries()].flatMap(([familyId, members]) => {
    const ordered = [...members].sort((left, right) =>
      left.createdAt.localeCompare(right.createdAt) || left.signature.id.localeCompare(right.signature.id)
    );
    const anchor = ordered[0]!;
    const anchorAffinity = affinityByPeerId.get(anchor.signature.id);
    if (!anchorAffinity?.familyEligible) return [];
    const best = ordered.map((member) => affinityByPeerId.get(member.signature.id))
      .filter((value): value is EpisodeCapabilityAffinityV1 => Boolean(value?.familyEligible))
      .sort((left, right) => right.combinedSimilarity - left.combinedSimilarity ||
        left.id.localeCompare(right.id))[0];
    return best ? [{ familyId, best }] : [];
  }).sort((left, right) => right.best.combinedSimilarity - left.best.combinedSimilarity ||
    left.familyId.localeCompare(right.familyId));
  return eligible[0]?.familyId ?? `episode_family_${stableHash({
    namespaceId: signature.namespaceId,
    seedSignatureId: signature.id
  }).slice(0, 24)}`;
}

function affinityEvidence(
  affinity: EpisodeCapabilityAffinityV1,
  peer: EpisodeCapabilitySignatureV1
): EpisodeAffinityEvidenceV1 {
  return {
    affinityId: affinity.id,
    peerSignatureId: peer.id,
    peerEpisodeId: peer.episodeId,
    goalSimilarity: affinity.goalSimilarity,
    stateTransitionSimilarity: affinity.stateTransitionSimilarity,
    outcomeSimilarity: affinity.outcomeSimilarity,
    contextSimilarity: affinity.contextSimilarity,
    pathStructureSimilarity: affinity.pathStructureSimilarity,
    combinedSimilarity: affinity.combinedSimilarity
  };
}

function contiguous(indexes: readonly number[]): boolean {
  return indexes.every((value, index) => index === 0 || value === indexes[index - 1]! + 1);
}

function embeddingProviderName(embedder: Embedder): string {
  return embedder.config.provider.trim() || "unknown";
}

function embeddingModelName(embedder: Embedder): string {
  return embedder.config.model?.trim() || "default";
}
