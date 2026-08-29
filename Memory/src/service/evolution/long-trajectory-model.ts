import { stableHash } from "../../utils/id.js";
import { cosineSimilarity } from "./trajectory-window-alignment.js";
import type {
  EmbeddedTrajectoryWindowV1,
  EpisodeExecutionPathLiteV1,
  ExecutionStepLiteV1,
  ProceduralEvidenceRole
} from "./procedural-window-model.js";

export const LONG_TRAJECTORY_MINING_VERSION =
  "reference-span-sequence-mining.v1" as const;
export const EPISODE_TRAJECTORY_FAMILY_VERSION =
  "episode-span-reference-family.v1" as const;
export const EPISODE_SPAN_PROJECTION_VERSION =
  "episode-span-sequence-projection.v1" as const;
export const LONG_COMMON_TRAJECTORY_VERSION =
  "long-common-span-sequence.v1" as const;
export const LONG_TRAJECTORY_CANDIDATE_STRUCTURE_VERSION =
  "long-trajectory-candidate-structure.v1" as const;

export interface EpisodeTrajectoryDocumentV1 {
  path: EpisodeExecutionPathLiteV1;
  goalText: string;
  terminalResultText: string;
  goalVector: number[];
  trajectoryText: string;
  trajectoryVector: number[];
  windows: EmbeddedTrajectoryWindowV1[];
}

export interface LongTrajectoryMiningConfigV1 {
  episodeRecallLimit: number;
  minEpisodeSimilarity: number;
  minGoalSimilarity: number;
  goalWeight: number;
  trajectoryWeight: number;
  windowTopK: number;
  coarseThresholds: Readonly<Record<number, number>>;
  minSupportEpisodes: number;
  minSpanSequenceLength: number;
  minTrajectorySpanSteps: number;
  minEpisodeCoverage: number;
}

export interface EpisodeTrajectorySimilarityV1 {
  episodeId: string;
  goalSimilarity: number;
  trajectorySimilarity: number;
  combinedSimilarity: number;
}

/** The newly completed Episode remains the reference coordinate during discovery. */
export interface EpisodeTrajectoryFamilyV1 {
  id: string;
  schemaVersion: typeof EPISODE_TRAJECTORY_FAMILY_VERSION;
  userId: string;
  seedEpisodeId: string;
  referenceEpisodeId: string;
  memberEpisodeIds: string[];
  similarities: EpisodeTrajectorySimilarityV1[];
}

/** One coarse whole-Span match from Episode B into Episode A's Span vocabulary. */
export interface ReferenceSpanMatchV1 {
  id: string;
  referenceSpanId: string;
  referenceSpanLabel: string;
  episodeSpanId: string;
  scale: number;
  referenceStartStepIndex: number;
  referenceEndStepIndex: number;
  episodeStartStepIndex: number;
  episodeEndStepIndex: number;
  coarseSimilarity: number;
  weight: number;
}

export interface SpanSequenceGapV1 {
  id: string;
  afterReferenceSpanId: string;
  beforeReferenceSpanId: string;
  startStepIndex: number;
  endStepIndex: number;
  stepIds: string[];
}

export interface EpisodeSpanSequenceProjectionV1 {
  id: string;
  schemaVersion: typeof EPISODE_SPAN_PROJECTION_VERSION;
  familyId: string;
  referenceEpisodeId: string;
  episodeId: string;
  evidenceRole: ProceduralEvidenceRole;
  matches: ReferenceSpanMatchV1[];
  gaps: SpanSequenceGapV1[];
  referenceCoverage: number;
  episodeCoverage: number;
  averageCoarseSimilarity: number;
}

export interface LongTrajectoryRequiredSpanV1 {
  anchorId: string;
  referenceSpanId: string;
  referenceSpanLabel: string;
  scale: number;
  referenceStartStepIndex: number;
  referenceEndStepIndex: number;
  semanticText: string;
  summaryText: string;
  supportEpisodeIds: string[];
  evidenceStepIds: string[];
  averageCoarseSimilarity: number;
}

export interface LongTrajectoryOccurrenceV1 {
  id: string;
  episodeId: string;
  pathId: string;
  evidenceRole: ProceduralEvidenceRole;
  matches: ReferenceSpanMatchV1[];
  gapStepIds: string[];
  referenceCoverage: number;
  episodeCoverage: number;
  averageCoarseSimilarity: number;
}

export interface LongCommonTrajectoryV1 {
  id: string;
  schemaVersion: typeof LONG_COMMON_TRAJECTORY_VERSION;
  familyId: string;
  referenceEpisodeId: string;
  referenceStartStepIndex: number;
  referenceEndStepIndex: number;
  requiredSpans: LongTrajectoryRequiredSpanV1[];
  supportEpisodeIds: string[];
  occurrences: LongTrajectoryOccurrenceV1[];
  averageEpisodeCoverage: number;
  averageCoarseSimilarity: number;
  /** Stable capability identity; excludes Episode, Path, Span, and occurrence IDs. */
  candidateStructureKey: string;
  structureHash: string;
}

export function episodeTrajectorySimilarity(
  left: EpisodeTrajectoryDocumentV1,
  right: EpisodeTrajectoryDocumentV1,
  config: Pick<LongTrajectoryMiningConfigV1, "goalWeight" | "trajectoryWeight">
): EpisodeTrajectorySimilarityV1 {
  const totalWeight = config.goalWeight + config.trajectoryWeight;
  if (!(config.goalWeight >= 0) || !(config.trajectoryWeight >= 0) || totalWeight <= 0) {
    throw new Error("Episode trajectory similarity weights must contain a positive value");
  }
  const goalSimilarity = cosineSimilarity(left.goalVector, right.goalVector);
  const trajectorySimilarity = cosineSimilarity(left.trajectoryVector, right.trajectoryVector);
  return {
    episodeId: right.path.episodeId,
    goalSimilarity,
    trajectorySimilarity,
    combinedSimilarity: (
      config.goalWeight * goalSimilarity +
      config.trajectoryWeight * trajectorySimilarity
    ) / totalWeight
  };
}

/** Build Episode A's Top-K neighbourhood without replacing A with a medoid. */
export function buildEpisodeTrajectoryFamily(
  seed: EpisodeTrajectoryDocumentV1,
  candidates: readonly EpisodeTrajectoryDocumentV1[],
  config: LongTrajectoryMiningConfigV1
): EpisodeTrajectoryFamilyV1 | undefined {
  validateConfig(config);
  const recalled = candidates.filter((candidate) =>
    candidate.path.userId === seed.path.userId &&
    candidate.path.episodeId !== seed.path.episodeId
  ).map((candidate) => ({
    document: candidate,
    similarity: episodeTrajectorySimilarity(seed, candidate, config)
  })).filter((item) =>
    item.similarity.goalSimilarity + Number.EPSILON >= config.minGoalSimilarity &&
    item.similarity.combinedSimilarity + Number.EPSILON >= config.minEpisodeSimilarity
  ).sort((left, right) =>
    right.similarity.combinedSimilarity - left.similarity.combinedSimilarity ||
    left.document.path.episodeId.localeCompare(right.document.path.episodeId)
  ).slice(0, config.episodeRecallLimit);
  if (recalled.length + 1 < config.minSupportEpisodes) return undefined;

  const similarities = [{
    episodeId: seed.path.episodeId,
    goalSimilarity: 1,
    trajectorySimilarity: 1,
    combinedSimilarity: 1
  }, ...recalled.map((item) => item.similarity)];
  const memberEpisodeIds = [
    seed.path.episodeId,
    ...recalled.map((item) => item.document.path.episodeId)
  ];
  const identity = {
    schemaVersion: EPISODE_TRAJECTORY_FAMILY_VERSION,
    userId: seed.path.userId,
    referenceEpisodeId: seed.path.episodeId,
    memberEpisodeIds
  };
  return {
    id: `episode_span_reference_family_${stableHash(identity).slice(0, 24)}`,
    schemaVersion: EPISODE_TRAJECTORY_FAMILY_VERSION,
    userId: seed.path.userId,
    seedEpisodeId: seed.path.episodeId,
    referenceEpisodeId: seed.path.episodeId,
    memberEpisodeIds,
    similarities
  };
}

/**
 * Encode Episode B in Episode A's multi-scale Span vocabulary. Coarse
 * similarity creates matches; 2-D interval scheduling chooses one longest,
 * monotonic, non-overlapping mixed-scale path. No Fine gate is applied.
 */
export function projectEpisodeToReferenceSpans(
  familyId: string,
  reference: EpisodeTrajectoryDocumentV1,
  episode: EpisodeTrajectoryDocumentV1,
  config: LongTrajectoryMiningConfigV1
): EpisodeSpanSequenceProjectionV1 {
  validateConfig(config);
  if (reference.path.userId !== episode.path.userId) {
    throw new Error("Cannot project Episodes from different users");
  }
  const ordinalByWindowId = spanOrdinals(reference.windows);
  const candidates: ReferenceSpanMatchV1[] = [];
  for (const referenceWindow of sortedWindows(reference.windows)) {
    const scale = referenceWindow.occurrence.scale;
    const threshold = config.coarseThresholds[scale];
    if (threshold === undefined) continue;
    const recalled = episode.windows.filter((window) =>
      window.occurrence.scale === scale
    ).map((window) => ({
      window,
      similarity: cosineSimilarity(referenceWindow.coarseVector, window.coarseVector)
    })).filter((item) => item.similarity + Number.EPSILON >= threshold)
      .sort((left, right) =>
        right.similarity - left.similarity ||
        left.window.occurrence.startStepIndex - right.window.occurrence.startStepIndex ||
        left.window.occurrence.id.localeCompare(right.window.occurrence.id)
      ).slice(0, config.windowTopK);
    for (const item of recalled) {
      const ordinal = ordinalByWindowId.get(referenceWindow.occurrence.id) ?? 0;
      const identity = {
        version: LONG_TRAJECTORY_MINING_VERSION,
        referenceSpanId: referenceWindow.occurrence.id,
        episodeSpanId: item.window.occurrence.id
      };
      candidates.push({
        id: `reference_span_match_${stableHash(identity).slice(0, 24)}`,
        referenceSpanId: referenceWindow.occurrence.id,
        referenceSpanLabel: `sp${scale}-${String(ordinal + 1).padStart(3, "0")}`,
        episodeSpanId: item.window.occurrence.id,
        scale,
        referenceStartStepIndex: referenceWindow.occurrence.startStepIndex,
        referenceEndStepIndex: referenceWindow.occurrence.endStepIndex,
        episodeStartStepIndex: item.window.occurrence.startStepIndex,
        episodeEndStepIndex: item.window.occurrence.endStepIndex,
        coarseSimilarity: item.similarity,
        weight: scale * item.similarity
      });
    }
  }
  const matches = selectMonotonicSpanPath(candidates);
  const gaps = gapsForMatches(episode.path.steps, matches);
  const referenceCoverage = intervalCoverage(matches.map((match) => [
    match.referenceStartStepIndex,
    match.referenceEndStepIndex
  ]));
  const episodeCoverage = intervalCoverage(matches.map((match) => [
    match.episodeStartStepIndex,
    match.episodeEndStepIndex
  ]));
  const identity = {
    version: EPISODE_SPAN_PROJECTION_VERSION,
    familyId,
    referenceEpisodeId: reference.path.episodeId,
    episodeId: episode.path.episodeId,
    matches: matches.map((item) => [item.referenceSpanId, item.episodeSpanId])
  };
  return {
    id: `episode_span_projection_${stableHash(identity).slice(0, 24)}`,
    schemaVersion: EPISODE_SPAN_PROJECTION_VERSION,
    familyId,
    referenceEpisodeId: reference.path.episodeId,
    episodeId: episode.path.episodeId,
    evidenceRole: evidenceRole(episode.path.terminalReward),
    matches,
    gaps,
    referenceCoverage,
    episodeCoverage,
    averageCoarseSimilarity: average(matches.map((item) => item.coarseSimilarity))
  };
}

/** Weighted interval scheduling in both A and B coordinates. */
export function selectMonotonicSpanPath(
  candidates: readonly ReferenceSpanMatchV1[]
): ReferenceSpanMatchV1[] {
  const sorted = [...candidates].sort((left, right) =>
    left.referenceEndStepIndex - right.referenceEndStepIndex ||
    left.episodeEndStepIndex - right.episodeEndStepIndex ||
    right.scale - left.scale ||
    right.coarseSimilarity - left.coarseSimilarity ||
    left.id.localeCompare(right.id));
  if (sorted.length === 0) return [];
  const states = sorted.map((item) => ({
    score: item.weight,
    coveredSteps: item.scale,
    tokenCount: 1,
    previous: -1
  }));
  for (let index = 0; index < sorted.length; index += 1) {
    for (let prior = 0; prior < index; prior += 1) {
      if (!spanMatchesAreMonotonic(sorted[prior]!, sorted[index]!)) continue;
      const candidate = {
        score: states[prior]!.score + sorted[index]!.weight,
        coveredSteps: states[prior]!.coveredSteps + sorted[index]!.scale,
        tokenCount: states[prior]!.tokenCount + 1,
        previous: prior
      };
      if (betterPathState(candidate, states[index]!)) states[index] = candidate;
    }
  }
  let cursor = 0;
  for (let index = 1; index < states.length; index += 1) {
    if (betterPathState(states[index]!, states[cursor]!)) cursor = index;
  }
  const result: ReferenceSpanMatchV1[] = [];
  while (cursor >= 0) {
    result.push(sorted[cursor]!);
    cursor = states[cursor]!.previous;
  }
  return result.reverse();
}

/**
 * Every successful projection contributes its longest A-token path. Identical
 * paths are grouped; A is the identity occurrence, so A+B yields support=2.
 */
export function mineLongCommonSpanSequences(
  family: EpisodeTrajectoryFamilyV1,
  documents: readonly EpisodeTrajectoryDocumentV1[],
  projections: readonly EpisodeSpanSequenceProjectionV1[],
  config: LongTrajectoryMiningConfigV1
): LongCommonTrajectoryV1[] {
  validateConfig(config);
  const documentByEpisodeId = new Map(documents.map((item) => [item.path.episodeId, item]));
  const reference = documentByEpisodeId.get(family.referenceEpisodeId);
  if (!reference) {
    throw new Error(`Episode Family reference is missing: ${family.referenceEpisodeId}`);
  }
  if (evidenceRole(reference.path.terminalReward) !== "support") return [];
  const referenceWindowById = new Map(reference.windows.map((item) => [
    item.occurrence.id,
    item
  ]));
  const grouped = new Map<string, EpisodeSpanSequenceProjectionV1[]>();
  for (const projection of projections) {
    if (projection.evidenceRole !== "support" ||
        !family.memberEpisodeIds.includes(projection.episodeId) ||
        projection.matches.length < config.minSpanSequenceLength ||
        projection.referenceCoverage + Number.EPSILON < config.minEpisodeCoverage ||
        projection.episodeCoverage + Number.EPSILON < config.minEpisodeCoverage ||
        matchedStepCount(projection.matches, "reference") < config.minTrajectorySpanSteps) {
      continue;
    }
    const key = projection.matches.map((item) => item.referenceSpanId).join("|");
    const current = grouped.get(key) ?? [];
    current.push(projection);
    grouped.set(key, current);
  }

  const trajectories: LongCommonTrajectoryV1[] = [];
  for (const group of grouped.values()) {
    const representative = [...group].sort((left, right) =>
      right.averageCoarseSimilarity - left.averageCoarseSimilarity ||
      left.episodeId.localeCompare(right.episodeId))[0]!;
    const supportEpisodeIds = unique([
      reference.path.episodeId,
      ...group.map((item) => item.episodeId)
    ]).sort();
    if (supportEpisodeIds.length < config.minSupportEpisodes) continue;
    const requiredSpans = representative.matches.flatMap((match) => {
      const window = referenceWindowById.get(match.referenceSpanId);
      if (!window) return [];
      const evidenceStepIds = unique(group.flatMap((projection) => {
        const projectionMatch = projection.matches.find((item) =>
          item.referenceSpanId === match.referenceSpanId);
        const occurrence = projectionMatch
          ? documentByEpisodeId.get(projection.episodeId)?.windows.find((item) =>
              item.occurrence.id === projectionMatch.episodeSpanId)
          : undefined;
        return occurrence?.occurrence.steps.map((step) => step.id) ?? [];
      }).concat(window.occurrence.steps.map((step) => step.id)));
      return [{
        anchorId: `reference_span_${stableHash(match.referenceSpanId).slice(0, 16)}`,
        referenceSpanId: match.referenceSpanId,
        referenceSpanLabel: match.referenceSpanLabel,
        scale: match.scale,
        referenceStartStepIndex: match.referenceStartStepIndex,
        referenceEndStepIndex: match.referenceEndStepIndex,
        semanticText: window.occurrence.semanticText,
        summaryText: window.occurrence.steps.map((step) => step.summary).join("\n"),
        supportEpisodeIds,
        evidenceStepIds,
        averageCoarseSimilarity: average(group.map((projection) =>
          projection.matches.find((item) => item.referenceSpanId === match.referenceSpanId)
            ?.coarseSimilarity ?? 0))
      } satisfies LongTrajectoryRequiredSpanV1];
    });
    if (requiredSpans.length < config.minSpanSequenceLength) continue;

    const referenceMatches = representative.matches.map((match) => ({
      ...match,
      id: `reference_span_identity_${stableHash(match.referenceSpanId).slice(0, 20)}`,
      episodeSpanId: match.referenceSpanId,
      episodeStartStepIndex: match.referenceStartStepIndex,
      episodeEndStepIndex: match.referenceEndStepIndex,
      coarseSimilarity: 1,
      weight: match.scale
    }));
    const occurrences: LongTrajectoryOccurrenceV1[] = [
      occurrenceForProjection(reference, family.id, referenceMatches),
      ...group.flatMap((projection) => {
        const document = documentByEpisodeId.get(projection.episodeId);
        return document ? [occurrenceForProjection(document, family.id, projection.matches)] : [];
      })
    ];
    const actualSupportEpisodeIds = unique(occurrences.filter((item) =>
      item.evidenceRole === "support" &&
      item.referenceCoverage + Number.EPSILON >= config.minEpisodeCoverage &&
      item.episodeCoverage + Number.EPSILON >= config.minEpisodeCoverage
    ).map((item) => item.episodeId)).sort();
    if (actualSupportEpisodeIds.length < config.minSupportEpisodes) continue;

    const referenceStartStepIndex = requiredSpans[0]!.referenceStartStepIndex;
    const referenceEndStepIndex = requiredSpans.at(-1)!.referenceEndStepIndex;
    const candidateStructureKey = longTrajectoryCandidateStructureKey(requiredSpans);
    const structure = {
      version: LONG_COMMON_TRAJECTORY_VERSION,
      referenceEpisodeId: reference.path.episodeId,
      referenceSpanIds: requiredSpans.map((item) => item.referenceSpanId),
      supportEpisodeIds: actualSupportEpisodeIds,
      occurrenceEvidence: occurrences.map((item) => ({
        episodeId: item.episodeId,
        episodeSpanIds: item.matches.map((match) => match.episodeSpanId)
      }))
    };
    const structureHash = stableHash(structure);
    trajectories.push({
      id: `long_common_span_sequence_${structureHash.slice(0, 24)}`,
      schemaVersion: LONG_COMMON_TRAJECTORY_VERSION,
      familyId: family.id,
      referenceEpisodeId: reference.path.episodeId,
      referenceStartStepIndex,
      referenceEndStepIndex,
      requiredSpans,
      supportEpisodeIds: actualSupportEpisodeIds,
      occurrences,
      averageEpisodeCoverage: average(occurrences.filter((item) =>
        actualSupportEpisodeIds.includes(item.episodeId)
      ).map((item) => Math.min(item.referenceCoverage, item.episodeCoverage))),
      averageCoarseSimilarity: average(occurrences.filter((item) =>
        actualSupportEpisodeIds.includes(item.episodeId)
      ).map((item) => item.averageCoarseSimilarity)),
      candidateStructureKey,
      structureHash
    });
  }
  return trajectories.sort((left, right) =>
    right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
    right.requiredSpans.length - left.requiredSpans.length ||
    (right.referenceEndStepIndex - right.referenceStartStepIndex) -
      (left.referenceEndStepIndex - left.referenceStartStepIndex) ||
    left.id.localeCompare(right.id));
}

/**
 * Candidate identity describes only the reusable ordered capability skeleton.
 * Evidence membership and reference coordinates belong to CandidateVersion.
 */
export function longTrajectoryCandidateStructureKey(
  spans: readonly LongTrajectoryRequiredSpanV1[]
): string {
  return stableHash({
    version: LONG_TRAJECTORY_CANDIDATE_STRUCTURE_VERSION,
    semanticSequence: spans.flatMap((span) =>
      normalizeCandidateSemanticSequence(span.semanticText))
  });
}

/**
 * Add one newly aligned Episode to an existing Candidate skeleton. The
 * Candidate identity remains stable; only its evidence-bearing trajectory
 * version changes.
 */
export function extendLongCommonTrajectoryWithProjection(input: {
  trajectory: LongCommonTrajectoryV1;
  episode: EpisodeTrajectoryDocumentV1;
  projection: EpisodeSpanSequenceProjectionV1;
}): LongCommonTrajectoryV1 {
  const { trajectory, episode, projection } = input;
  if (projection.episodeId !== episode.path.episodeId ||
      projection.referenceEpisodeId !== trajectory.referenceEpisodeId) {
    throw new Error("Candidate projection does not match its Episode/reference coordinates");
  }
  const requiredSpanIds = new Set(trajectory.requiredSpans.map((span) =>
    span.referenceSpanId));
  if (projection.matches.length !== requiredSpanIds.size ||
      projection.matches.some((match) => !requiredSpanIds.has(match.referenceSpanId))) {
    throw new Error("Candidate evolution requires the complete retained Span sequence");
  }
  const occurrence = occurrenceForProjection(
    episode,
    trajectory.familyId,
    projection.matches
  );
  const occurrences = [
    ...trajectory.occurrences.filter((item) => item.episodeId !== occurrence.episodeId),
    occurrence
  ].sort((left, right) => left.episodeId.localeCompare(right.episodeId));
  const supportEpisodeIds = unique([
    ...trajectory.supportEpisodeIds,
    episode.path.episodeId
  ]).sort();
  const windowById = new Map(episode.windows.map((window) => [
    window.occurrence.id,
    window
  ]));
  const requiredSpans = trajectory.requiredSpans.map((span) => {
    const match = projection.matches.find((item) =>
      item.referenceSpanId === span.referenceSpanId);
    const window = match ? windowById.get(match.episodeSpanId) : undefined;
    return {
      ...span,
      supportEpisodeIds: unique([
        ...span.supportEpisodeIds,
        episode.path.episodeId
      ]).sort(),
      evidenceStepIds: unique([
        ...span.evidenceStepIds,
        ...(window?.occurrence.steps.map((step) => step.id) ?? [])
      ]).sort(),
      averageCoarseSimilarity: average([
        span.averageCoarseSimilarity,
        match?.coarseSimilarity ?? span.averageCoarseSimilarity
      ])
    };
  });
  const structureHash = stableHash({
    version: LONG_COMMON_TRAJECTORY_VERSION,
    candidateStructureKey: trajectory.candidateStructureKey,
    supportEpisodeIds,
    occurrences: occurrences.map((item) => ({
      episodeId: item.episodeId,
      matches: item.matches.map((match) => [match.referenceSpanId, match.episodeSpanId])
    }))
  });
  return {
    ...trajectory,
    id: `long_common_span_sequence_${structureHash.slice(0, 24)}`,
    requiredSpans,
    supportEpisodeIds,
    occurrences,
    averageEpisodeCoverage: average(occurrences.map((item) =>
      Math.min(item.referenceCoverage, item.episodeCoverage))),
    averageCoarseSimilarity: average(occurrences.map((item) =>
      item.averageCoarseSimilarity)),
    structureHash
  };
}

function normalizeCandidateSemanticSequence(value: string): string[] {
  return value.split(/\r?\n/u).map((line) => line
    .toLowerCase()
    .replace(/^\s*\d+[.)]\s*/u, "")
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .replace(/\s+/gu, " ")
    .trim())
    .filter(Boolean);
}

/**
 * Keep only maximal overlapping paths in one reference Episode coordinate.
 * A shorter candidate is suppressed when every reference Step covered by its
 * Span tokens is already covered by a longer admitted candidate. Disjoint
 * paths remain independent candidates.
 */
export function selectMaximalLongCommonTrajectories(
  trajectories: readonly LongCommonTrajectoryV1[]
): LongCommonTrajectoryV1[] {
  const ordered = [...trajectories].sort((left, right) =>
    coveredReferenceStepCount(right) - coveredReferenceStepCount(left) ||
    right.requiredSpans.length - left.requiredSpans.length ||
    right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
    right.averageCoarseSimilarity - left.averageCoarseSimilarity ||
    left.id.localeCompare(right.id));
  const selected: LongCommonTrajectoryV1[] = [];
  for (const candidate of ordered) {
    const contained = selected.some((maximal) =>
      maximal.familyId === candidate.familyId &&
      maximal.referenceEpisodeId === candidate.referenceEpisodeId &&
      sameStringSet(maximal.supportEpisodeIds, candidate.supportEpisodeIds) &&
      referenceSpanCoverageContains(maximal.requiredSpans, candidate.requiredSpans));
    if (!contained) selected.push(candidate);
  }
  return selected.sort((left, right) =>
    left.referenceStartStepIndex - right.referenceStartStepIndex ||
    left.referenceEndStepIndex - right.referenceEndStepIndex ||
    left.id.localeCompare(right.id));
}

export function trajectoryIntentSequenceText(path: EpisodeExecutionPathLiteV1): string {
  return path.steps.map((step, index) => `${index + 1}. ${step.intent.trim()}`).join("\n");
}

function occurrenceForProjection(
  document: EpisodeTrajectoryDocumentV1,
  familyId: string,
  matches: readonly ReferenceSpanMatchV1[]
): LongTrajectoryOccurrenceV1 {
  const gaps = gapsForMatches(document.path.steps, matches);
  const referenceCoverage = intervalCoverage(matches.map((match) => [
    match.referenceStartStepIndex,
    match.referenceEndStepIndex
  ]));
  const episodeCoverage = intervalCoverage(matches.map((match) => [
    match.episodeStartStepIndex,
    match.episodeEndStepIndex
  ]));
  return {
    id: `long_span_sequence_occurrence_${stableHash({
      version: LONG_COMMON_TRAJECTORY_VERSION,
      familyId,
      episodeId: document.path.episodeId,
      matches: matches.map((item) => [item.referenceSpanId, item.episodeSpanId])
    }).slice(0, 24)}`,
    episodeId: document.path.episodeId,
    pathId: document.path.id,
    evidenceRole: evidenceRole(document.path.terminalReward),
    matches: [...matches],
    gapStepIds: gaps.flatMap((gap) => gap.stepIds),
    referenceCoverage,
    episodeCoverage,
    averageCoarseSimilarity: average(matches.map((item) => item.coarseSimilarity))
  };
}

function coveredReferenceStepCount(trajectory: LongCommonTrajectoryV1): number {
  const covered = new Set<number>();
  for (const span of trajectory.requiredSpans) {
    for (let index = span.referenceStartStepIndex;
      index <= span.referenceEndStepIndex; index += 1) {
      covered.add(index);
    }
  }
  return covered.size;
}

function referenceSpanCoverageContains(
  container: readonly LongTrajectoryRequiredSpanV1[],
  candidate: readonly LongTrajectoryRequiredSpanV1[]
): boolean {
  if (container.length === 0 || candidate.length === 0) return false;
  const covered = new Set<number>();
  for (const span of container) {
    for (let index = span.referenceStartStepIndex;
      index <= span.referenceEndStepIndex; index += 1) {
      covered.add(index);
    }
  }
  return candidate.every((span) => {
    for (let index = span.referenceStartStepIndex;
      index <= span.referenceEndStepIndex; index += 1) {
      if (!covered.has(index)) return false;
    }
    return true;
  });
}

function gapsForMatches(
  steps: readonly ExecutionStepLiteV1[],
  matches: readonly ReferenceSpanMatchV1[]
): SpanSequenceGapV1[] {
  const gaps: SpanSequenceGapV1[] = [];
  for (let index = 1; index < matches.length; index += 1) {
    const previous = matches[index - 1]!;
    const current = matches[index]!;
    const gapSteps = steps.filter((step) =>
      step.stepIndex > previous.episodeEndStepIndex &&
      step.stepIndex < current.episodeStartStepIndex);
    if (gapSteps.length === 0) continue;
    const identity = {
      afterReferenceSpanId: previous.referenceSpanId,
      beforeReferenceSpanId: current.referenceSpanId,
      stepIds: gapSteps.map((step) => step.id)
    };
    gaps.push({
      id: `span_sequence_gap_${stableHash(identity).slice(0, 20)}`,
      afterReferenceSpanId: previous.referenceSpanId,
      beforeReferenceSpanId: current.referenceSpanId,
      startStepIndex: gapSteps[0]!.stepIndex,
      endStepIndex: gapSteps.at(-1)!.stepIndex,
      stepIds: gapSteps.map((step) => step.id)
    });
  }
  return gaps;
}

function spanOrdinals(windows: readonly EmbeddedTrajectoryWindowV1[]): Map<string, number> {
  const result = new Map<string, number>();
  const byScale = new Map<number, EmbeddedTrajectoryWindowV1[]>();
  for (const window of windows) {
    const current = byScale.get(window.occurrence.scale) ?? [];
    current.push(window);
    byScale.set(window.occurrence.scale, current);
  }
  for (const scaleWindows of byScale.values()) {
    sortedWindows(scaleWindows).forEach((window, index) =>
      result.set(window.occurrence.id, index));
  }
  return result;
}

function sortedWindows(
  windows: readonly EmbeddedTrajectoryWindowV1[]
): EmbeddedTrajectoryWindowV1[] {
  return [...windows].sort((left, right) =>
    left.occurrence.startStepIndex - right.occurrence.startStepIndex ||
    right.occurrence.scale - left.occurrence.scale ||
    left.occurrence.id.localeCompare(right.occurrence.id));
}

function spanMatchesAreMonotonic(
  left: ReferenceSpanMatchV1,
  right: ReferenceSpanMatchV1
): boolean {
  return left.referenceEndStepIndex < right.referenceStartStepIndex &&
    left.episodeEndStepIndex < right.episodeStartStepIndex;
}

function betterPathState(
  left: { score: number; coveredSteps: number; tokenCount: number; previous: number },
  right: { score: number; coveredSteps: number; tokenCount: number; previous: number }
): boolean {
  return left.score > right.score + Number.EPSILON ||
    (Math.abs(left.score - right.score) <= Number.EPSILON &&
      (left.coveredSteps > right.coveredSteps ||
        (left.coveredSteps === right.coveredSteps && left.tokenCount < right.tokenCount)));
}

function intervalCoverage(intervals: ReadonlyArray<readonly [number, number]>): number {
  if (intervals.length === 0) return 0;
  const sorted = [...intervals].sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  const start = sorted[0]![0];
  const end = sorted.at(-1)![1];
  const covered = sorted.reduce((sum, interval) => sum + interval[1] - interval[0] + 1, 0);
  return covered / Math.max(1, end - start + 1);
}

function matchedStepCount(
  matches: readonly ReferenceSpanMatchV1[],
  coordinate: "reference" | "episode"
): number {
  return matches.reduce((sum, match) => sum + (
    coordinate === "reference"
      ? match.referenceEndStepIndex - match.referenceStartStepIndex + 1
      : match.episodeEndStepIndex - match.episodeStartStepIndex + 1
  ), 0);
}

function evidenceRole(reward: number | undefined): ProceduralEvidenceRole {
  if (typeof reward !== "number" || !Number.isFinite(reward) || reward === 0) return "unknown";
  return reward > 0 ? "support" : "counterexample";
}

function validateConfig(config: LongTrajectoryMiningConfigV1): void {
  if (!Number.isInteger(config.episodeRecallLimit) || config.episodeRecallLimit < 1) {
    throw new Error("episodeRecallLimit must be a positive integer");
  }
  if (!Number.isInteger(config.windowTopK) || config.windowTopK < 1) {
    throw new Error("windowTopK must be a positive integer");
  }
  for (const [name, value] of [
    ["minEpisodeSimilarity", config.minEpisodeSimilarity],
    ["minGoalSimilarity", config.minGoalSimilarity],
    ["minEpisodeCoverage", config.minEpisodeCoverage]
  ] as const) {
    if (!(value >= -1 && value <= 1)) throw new Error(`${name} must be in [-1, 1]`);
  }
  for (const [name, value] of [
    ["minSupportEpisodes", config.minSupportEpisodes],
    ["minSpanSequenceLength", config.minSpanSequenceLength],
    ["minTrajectorySpanSteps", config.minTrajectorySpanSteps]
  ] as const) {
    if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be >= 1`);
  }
}

function average(values: readonly number[]): number {
  return values.length > 0
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) return false;
  const rightSet = new Set(right);
  return left.every((item) => rightSet.has(item));
}
