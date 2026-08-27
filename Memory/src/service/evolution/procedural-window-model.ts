import { stableHash } from "../../utils/id.js";
import {
  bandedMonotonicMatch,
  cosineSimilarity,
  selfBandedMonotonicMatch,
  type BandedMonotonicMatchConfig,
  type BandedMonotonicMatchResultV1
} from "./trajectory-window-alignment.js";

export const EXECUTION_STEP_LITE_SCHEMA_VERSION = "execution-step-lite.v1" as const;
export const TURN_TRANSITION_SCHEMA_VERSION = "turn-transition.v1" as const;
export const EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION =
  "episode-execution-path-lite.v1" as const;
export const TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION =
  "trajectory-window-occurrence.v1" as const;
export const PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION =
  "procedural-span-occurrence.v1" as const;
export const ALIGNED_COMMON_CORE_SCHEMA_VERSION = "aligned-common-core.v1" as const;
export const ANCHORED_COMPLETION_SCHEMA_VERSION = "anchored-completion.v2" as const;
/**
 * Mechanical 5/10 projection identity. Keep this stable so the v2 Family
 * miner can reuse v1 Window rows and embeddings instead of duplicating them.
 */
export const PROCEDURAL_WINDOW_MECHANICAL_VERSION =
  "procedural-window-mining.v1" as const;
/** Multi-Family coarse recall + Family-local exclusive fine clustering. */
export const PROCEDURAL_WINDOW_MINING_VERSION = "procedural-window-mining.v2" as const;
export const FINE_EVIDENCE_SIGNATURE_VERSION =
  "trajectory-window-fine-evidence.v1" as const;

export type ExecutionStepLiteKind = "tool_action" | "response_generation";
export type ExecutionStepLiteOutcome = "success" | "failure" | "partial" | "unknown";
export type TurnGoalChangeType = "continue" | "refine" | "replace" | "unknown";
export type ProceduralEvidenceRole = "support" | "counterexample" | "unknown";

export interface ExecutionStepLiteV1 {
  id: string;
  /** Optional while reading the first lightweight compiler output. */
  schemaVersion?: typeof EXECUTION_STEP_LITE_SCHEMA_VERSION;
  episodeId: string;
  rawTurnId: string;
  turnIndex: number;
  stepIndex: number;
  kind: ExecutionStepLiteKind;
  toolName?: string;
  toolCallIndex?: number;
  intent: string;
  summary: string;
  outcome: ExecutionStepLiteOutcome;
  errorCode?: string;
  retryOfStepId?: string;
  recoveryFromStepId?: string;
  evidenceRefs: string[];
  provenance: {
    algorithmVersion: string;
    model?: string;
    sourceSnapshotHash: string;
  };
}

/**
 * A user message remains visible between adjacent agent Steps but is not
 * counted as an execution Step and therefore does not consume a 5/10 slot.
 */
export interface TurnTransitionV1 {
  id: string;
  schemaVersion: typeof TURN_TRANSITION_SCHEMA_VERSION;
  episodeId: string;
  rawTurnId: string;
  turnIndex: number;
  beforeStepIndex?: number;
  afterStepIndex?: number;
  userObservation?: string;
  sourceRef?: string;
  goalChangeType?: TurnGoalChangeType;
}

export interface EpisodeExecutionPathLiteV1 {
  id: string;
  schemaVersion: typeof EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION;
  episodeId: string;
  userId: string;
  sourceRawTurnIds: string[];
  steps: ExecutionStepLiteV1[];
  turnTransitions: TurnTransitionV1[];
  terminalReward?: number;
  sourceSnapshotHash: string;
  pathHash: string;
  compilerVersion?: string;
  modelSignature?: string;
  provenance: {
    algorithmVersion?: string;
    model?: string;
    inputCandidateCount?: number;
    compiledCandidateCount?: number;
    truncated?: boolean;
  };
}

export interface MultiScaleWindowSpec {
  length: number;
  stride: number;
}

export const V15_WINDOW_SPECS: readonly MultiScaleWindowSpec[] = [
  { length: 5, stride: 2 },
  { length: 10, stride: 5 }
] as const;

export const V15_COARSE_SIMILARITY_THRESHOLDS: Readonly<Record<number, number>> = {
  5: 0.76,
  10: 0.70
};

export const V15_FINE_MATCH_CONFIGS: readonly BandedMonotonicMatchConfig[] = [
  {
    scale: 5,
    bandWidth: 1,
    minStepSimilarity: 0.70,
    minMatchedSteps: 4,
    minCoverage: 0.80,
    minAverageMatchSimilarity: 0.78,
    maxInternalGap: 1,
    gapPenalty: 0.10,
    minAlignmentScore: 0.62
  },
  {
    scale: 10,
    bandWidth: 2,
    minStepSimilarity: 0.68,
    minMatchedSteps: 7,
    minCoverage: 0.70,
    minAverageMatchSimilarity: 0.76,
    maxInternalGap: 2,
    gapPenalty: 0.10,
    minAlignmentScore: 0.52
  }
] as const;

export const V15_FINE_MEDOID_SWITCH_MARGIN = 0.01;
export const V15_MIN_SUPPORT_EPISODES = 2;

export interface TrajectoryWindowOccurrenceV1 {
  id: string;
  schemaVersion: typeof TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION;
  episodeId: string;
  pathId: string;
  userId: string;
  terminalReward?: number;
  evidenceRole: ProceduralEvidenceRole;
  scale: number;
  stride: number;
  startStepIndex: number;
  endStepIndex: number;
  semanticText: string;
  steps: ExecutionStepLiteV1[];
}

export interface EmbeddedTrajectoryWindowV1 {
  occurrence: TrajectoryWindowOccurrenceV1;
  /** Embedding of the complete ordered intent sequence. */
  coarseVector: number[];
  /** One normalized intent embedding per Step, in execution order. */
  stepVectors: number[][];
}

export interface ConstrainedRealMedoidV1 {
  medoid: EmbeddedTrajectoryWindowV1;
  centrality: number;
  minimumSimilarity: number;
}

export interface ProceduralWindowFamilyV1 {
  id: string;
  scale: number;
  medoid: EmbeddedTrajectoryWindowV1;
  members: EmbeddedTrajectoryWindowV1[];
  medoidCentrality: number;
  minimumSimilarityToMedoid: number;
  medoidUpdateCount: number;
}

export interface TrajectoryWindowClusterMemberV1 {
  occurrence: TrajectoryWindowOccurrenceV1;
  similarityToMedoid: number;
  alignmentToMedoid: BandedMonotonicMatchResultV1;
}

export interface TrajectoryWindowClusterV1 {
  id: string;
  familyId: string;
  scale: number;
  medoidOccurrenceId: string;
  episodeIds: string[];
  supportEpisodeIds: string[];
  counterexampleEpisodeIds: string[];
  unknownEpisodeIds: string[];
  occurrenceCount: number;
  averageSimilarity: number;
  minimumSimilarity: number;
  medoidCentrality: number;
  medoidUpdateCount: number;
  members: TrajectoryWindowClusterMemberV1[];
  suppressedByClusterId?: string;
}

export interface TrajectoryWindowClusteringResultV1 {
  families: ProceduralWindowFamilyV1[];
  clusters: TrajectoryWindowClusterV1[];
}

export interface AlignedCommonCoreStepV1 {
  anchorOffset: number;
  anchorStepId: string;
  intent: string;
  summary: string;
  supportEpisodeIds: string[];
  evidenceStepIds: string[];
}

export interface ProceduralSpanOccurrenceV1 {
  id: string;
  schemaVersion: typeof PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION;
  commonCoreId: string;
  clusterId: string;
  sourceWindowOccurrenceId: string;
  episodeId: string;
  pathId: string;
  userId: string;
  evidenceRole: ProceduralEvidenceRole;
  scale: number;
  startStepIndex: number;
  endStepIndex: number;
  matchedStepIds: string[];
  gapStepIds: string[];
  evidenceRefs: string[];
  averageMatchSimilarity: number;
  terminalReward?: number;
}

export interface AlignedCommonCoreV1 {
  id: string;
  schemaVersion: typeof ALIGNED_COMMON_CORE_SCHEMA_VERSION;
  clusterId: string;
  scale: number;
  medoidOccurrenceId: string;
  anchorOffsets: number[];
  steps: AlignedCommonCoreStepV1[];
  supportEpisodeIds: string[];
  spanOccurrences: ProceduralSpanOccurrenceV1[];
}

export interface AnchoredCompletionStepCandidateV2 {
  stepId: string;
  vector: number[];
}

export interface AnchoredCompletionOccurrenceCandidateV2 {
  occurrenceId: string;
  episodeId: string;
  evidenceRole: ProceduralEvidenceRole;
  prefix: AnchoredCompletionStepCandidateV2[];
  suffix: AnchoredCompletionStepCandidateV2[];
}

export interface SharedExtensionAnchorV2 {
  anchorId: string;
  side: "prefix" | "suffix";
  referenceOffset: number;
  referenceStepId: string;
  supportEpisodeIds: string[];
  evidenceStepIds: string[];
  averageMatchSimilarity: number;
}

export interface CompletionStepProjectionV2 {
  stepId: string;
  role: "shared_extension" | "local_context";
  extensionAnchorId?: string;
  matchSimilarity?: number;
}

export interface AnchoredCompletionOccurrenceProjectionV2 {
  occurrenceId: string;
  prefix: CompletionStepProjectionV2[];
  suffix: CompletionStepProjectionV2[];
}

export interface AnchoredCompletionOverlayV2 {
  id: string;
  schemaVersion: typeof ANCHORED_COMPLETION_SCHEMA_VERSION;
  commonCoreId: string;
  referenceOccurrenceId: string;
  maxPrefixSteps: number;
  maxSuffixSteps: number;
  minStepSimilarity: number;
  minSupportEpisodes: number;
  sharedPrefix: SharedExtensionAnchorV2[];
  sharedSuffix: SharedExtensionAnchorV2[];
  projections: AnchoredCompletionOccurrenceProjectionV2[];
  extensionAgreement: number;
}

/**
 * Complete an already-stable Common Core without changing Cluster identity.
 * The reference occurrence provides bounded prefix/suffix candidate anchors;
 * other occurrences align monotonically to those candidates. Only extension
 * positions repeated by enough successful Episodes become shared evidence.
 */
export function extractAnchoredCompletionOverlay(
  commonCoreId: string,
  occurrences: readonly AnchoredCompletionOccurrenceCandidateV2[],
  options: {
    referenceOccurrenceId?: string;
    maxPrefixSteps: number;
    maxSuffixSteps: number;
    minStepSimilarity: number;
    minSupportEpisodes: number;
  }
): AnchoredCompletionOverlayV2 | undefined {
  if (!Number.isInteger(options.maxPrefixSteps) || options.maxPrefixSteps < 0 ||
      !Number.isInteger(options.maxSuffixSteps) || options.maxSuffixSteps < 0) {
    throw new Error("anchored completion expansion budgets must be non-negative integers");
  }
  if (!(options.minStepSimilarity > 0 && options.minStepSimilarity <= 1)) {
    throw new Error("anchored completion minStepSimilarity must be in (0, 1]");
  }
  if (!Number.isInteger(options.minSupportEpisodes) || options.minSupportEpisodes < 2) {
    throw new Error("anchored completion minSupportEpisodes must be >= 2");
  }
  const ordered = occurrences.map((item) => ({
    ...item,
    prefix: options.maxPrefixSteps === 0
      ? []
      : item.prefix.slice(-options.maxPrefixSteps),
    suffix: item.suffix.slice(0, options.maxSuffixSteps)
  })).sort((left, right) =>
    left.episodeId.localeCompare(right.episodeId) ||
    left.occurrenceId.localeCompare(right.occurrenceId));
  const supports = ordered.filter((item) => item.evidenceRole === "support");
  if (supports.length < options.minSupportEpisodes) return undefined;
  const reference = supports.find((item) =>
    item.occurrenceId === options.referenceOccurrenceId) ?? supports[0]!;
  const prefix = projectCompletionSide(
    "prefix",
    reference,
    ordered,
    options.minStepSimilarity,
    options.minSupportEpisodes
  );
  const suffix = projectCompletionSide(
    "suffix",
    reference,
    ordered,
    options.minStepSimilarity,
    options.minSupportEpisodes
  );
  const projections = ordered.map((occurrence) => ({
    occurrenceId: occurrence.occurrenceId,
    prefix: prefix.projections.get(occurrence.occurrenceId) ?? [],
    suffix: suffix.projections.get(occurrence.occurrenceId) ?? []
  }));
  const referenceCandidateCount = reference.prefix.length + reference.suffix.length;
  const sharedCount = prefix.anchors.length + suffix.anchors.length;
  const extensionAgreement = referenceCandidateCount === 0
    ? 1
    : sharedCount / referenceCandidateCount;
  const identity = {
    version: ANCHORED_COMPLETION_SCHEMA_VERSION,
    commonCoreId,
    referenceOccurrenceId: reference.occurrenceId,
    maxPrefixSteps: options.maxPrefixSteps,
    maxSuffixSteps: options.maxSuffixSteps,
    minStepSimilarity: options.minStepSimilarity,
    minSupportEpisodes: options.minSupportEpisodes,
    sharedPrefix: prefix.anchors,
    sharedSuffix: suffix.anchors,
    projections
  };
  return {
    id: `anchored_completion_${stableHash(identity).slice(0, 24)}`,
    schemaVersion: ANCHORED_COMPLETION_SCHEMA_VERSION,
    commonCoreId,
    referenceOccurrenceId: reference.occurrenceId,
    maxPrefixSteps: options.maxPrefixSteps,
    maxSuffixSteps: options.maxSuffixSteps,
    minStepSimilarity: options.minStepSimilarity,
    minSupportEpisodes: options.minSupportEpisodes,
    sharedPrefix: prefix.anchors,
    sharedSuffix: suffix.anchors,
    projections,
    extensionAgreement
  };
}

export function proceduralEvidenceRoleForReward(
  terminalReward: number | undefined
): ProceduralEvidenceRole {
  if (terminalReward === undefined || terminalReward === 0) return "unknown";
  return terminalReward > 0 ? "support" : "counterexample";
}

export function buildTrajectoryWindows(
  paths: readonly EpisodeExecutionPathLiteV1[],
  specs: readonly MultiScaleWindowSpec[] = V15_WINDOW_SPECS
): TrajectoryWindowOccurrenceV1[] {
  const validSpecs = validateWindowSpecs(specs);
  const windows: TrajectoryWindowOccurrenceV1[] = [];
  for (const path of [...paths].sort((left, right) =>
    left.episodeId.localeCompare(right.episodeId) || left.id.localeCompare(right.id))) {
    validateEpisodeExecutionPathLite(path);
    for (const spec of validSpecs) {
      if (path.steps.length < spec.length) continue;
      const starts: number[] = [];
      for (let start = 0; start + spec.length <= path.steps.length; start += spec.stride) {
        starts.push(start);
      }
      const tailStart = path.steps.length - spec.length;
      if (starts.at(-1) !== tailStart) starts.push(tailStart);

      for (const start of starts) {
        const steps = path.steps.slice(start, start + spec.length);
        const startStepIndex = steps[0]!.stepIndex;
        const endStepIndex = steps.at(-1)!.stepIndex;
        const semanticText = windowIntentSequenceText(steps);
        const identity = {
          version: PROCEDURAL_WINDOW_MECHANICAL_VERSION,
          pathId: path.id,
          pathHash: path.pathHash,
          scale: spec.length,
          startStepIndex,
          endStepIndex,
          semanticText
        };
        windows.push({
          id: `trajectory_window_${stableHash(identity).slice(0, 24)}`,
          schemaVersion: TRAJECTORY_WINDOW_OCCURRENCE_SCHEMA_VERSION,
          episodeId: path.episodeId,
          pathId: path.id,
          userId: path.userId,
          ...(path.terminalReward === undefined
            ? {}
            : { terminalReward: path.terminalReward }),
          evidenceRole: proceduralEvidenceRoleForReward(path.terminalReward),
          scale: spec.length,
          stride: spec.stride,
          startStepIndex,
          endStepIndex,
          semanticText,
          steps
        });
      }
    }
  }
  return windows;
}

export function windowIntentSequenceText(steps: readonly ExecutionStepLiteV1[]): string {
  return steps.map((step, index) => `${index + 1}. ${step.intent.trim()}`).join("\n");
}

export function validateEpisodeExecutionPathLite(path: EpisodeExecutionPathLiteV1): void {
  if (path.schemaVersion !== EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION) {
    throw new Error(`unsupported EpisodeExecutionPathLite schema: ${path.schemaVersion}`);
  }
  const rawTurnIds = new Set(path.sourceRawTurnIds);
  for (const [index, step] of path.steps.entries()) {
    if (step.episodeId !== path.episodeId) {
      throw new Error(`execution Step belongs to another Episode: ${step.id}`);
    }
    if (step.stepIndex !== index) {
      throw new Error(`execution Step index mismatch: expected ${index}, got ${step.stepIndex}`);
    }
    if (!rawTurnIds.has(step.rawTurnId)) {
      throw new Error(`execution Step references a RawTurn outside the Path: ${step.id}`);
    }
    if (!step.intent.trim()) throw new Error(`execution Step has empty intent: ${step.id}`);
  }
  for (const transition of path.turnTransitions) {
    if (transition.episodeId !== path.episodeId || !rawTurnIds.has(transition.rawTurnId)) {
      throw new Error(`Turn transition is outside the Path: ${transition.id}`);
    }
    for (const stepIndex of [transition.beforeStepIndex, transition.afterStepIndex]) {
      if (stepIndex !== undefined &&
          (!Number.isInteger(stepIndex) || stepIndex < 0 || stepIndex >= path.steps.length)) {
        throw new Error(`Turn transition references invalid Step index: ${transition.id}`);
      }
    }
  }
}

export function unitVector(vector: readonly number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (norm === 0) return vector.map(() => 0);
  return vector.map((value) => value / norm);
}

export function averageVectors(vectors: readonly (readonly number[])[]): number[] {
  const dimension = vectors[0]?.length ?? 0;
  if (dimension === 0 || vectors.some((vector) => vector.length !== dimension)) return [];
  const center = new Array<number>(dimension).fill(0);
  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) {
      center[index] = center[index]! + vector[index]!;
    }
  }
  return center.map((value) => value / vectors.length);
}

export function selectConstrainedRealMedoid(
  members: readonly EmbeddedTrajectoryWindowV1[],
  similarityThreshold: number
): ConstrainedRealMedoidV1 | undefined {
  if (!(similarityThreshold > 0 && similarityThreshold <= 1)) {
    throw new Error("coarse similarity threshold must be in (0, 1]");
  }
  let best: ConstrainedRealMedoidV1 | undefined;
  for (const possible of members) {
    const similarities = members.map((member) => cosineSimilarity(
      possible.coarseVector,
      member.coarseVector
    ));
    const minimumSimilarity = Math.min(...similarities);
    if (minimumSimilarity + Number.EPSILON < similarityThreshold) continue;
    const centrality = average(similarities);
    if (!best || centrality > best.centrality ||
        (centrality === best.centrality && minimumSimilarity > best.minimumSimilarity) ||
        (centrality === best.centrality && minimumSimilarity === best.minimumSimilarity &&
          possible.occurrence.id.localeCompare(best.medoid.occurrence.id) < 0)) {
      best = { medoid: possible, centrality, minimumSimilarity };
    }
  }
  return best;
}

/** Return every existing Family medoid that passes v15's scale-specific gate. */
export function matchingCoarseFamilies(
  window: EmbeddedTrajectoryWindowV1,
  families: readonly ProceduralWindowFamilyV1[],
  thresholds: Readonly<Record<number, number>> = V15_COARSE_SIMILARITY_THRESHOLDS
): Array<{ familyId: string; similarity: number }> {
  const threshold = requiredScaleValue(thresholds, window.occurrence.scale, "coarse threshold");
  return families.flatMap((family) => {
    if (family.scale !== window.occurrence.scale) return [];
    const similarity = cosineSimilarity(window.coarseVector, family.medoid.coarseVector);
    return similarity + Number.EPSILON >= threshold
      ? [{ familyId: family.id, similarity }]
      : [];
  }).sort((left, right) => right.similarity - left.similarity ||
    left.familyId.localeCompare(right.familyId));
}

/**
 * Increment one window into every threshold-valid coarse Family. Family ids
 * remain anchored to their first occurrence even when the real medoid changes.
 */
export function ingestWindowIntoCoarseFamilies(
  existingFamilies: readonly ProceduralWindowFamilyV1[],
  window: EmbeddedTrajectoryWindowV1,
  thresholds: Readonly<Record<number, number>> = V15_COARSE_SIMILARITY_THRESHOLDS
): { families: ProceduralWindowFamilyV1[]; matchedFamilyIds: string[]; createdFamilyId?: string } {
  validateEmbeddedWindow(window);
  const threshold = requiredScaleValue(thresholds, window.occurrence.scale, "coarse threshold");
  const matches = matchingCoarseFamilies(window, existingFamilies, thresholds);
  const matchedIds = new Set(matches.map((match) => match.familyId));
  if (matches.length === 0) {
    if (window.occurrence.evidenceRole !== "support") {
      return { families: existingFamilies.map(cloneFamily), matchedFamilyIds: [] };
    }
    const created: ProceduralWindowFamilyV1 = {
      id: `trajectory_window_family_${stableHash({
        version: PROCEDURAL_WINDOW_MINING_VERSION,
        scale: window.occurrence.scale,
        anchorOccurrenceId: window.occurrence.id
      }).slice(0, 24)}`,
      scale: window.occurrence.scale,
      medoid: window,
      members: [window],
      medoidCentrality: 1,
      minimumSimilarityToMedoid: 1,
      medoidUpdateCount: 0
    };
    return {
      families: [...existingFamilies.map(cloneFamily), created],
      matchedFamilyIds: [],
      createdFamilyId: created.id
    };
  }

  const families = existingFamilies.map((family): ProceduralWindowFamilyV1 => {
    if (!matchedIds.has(family.id) ||
        family.members.some((member) => member.occurrence.id === window.occurrence.id)) {
      return cloneFamily(family);
    }
    const members = [...family.members, window];
    const candidate = selectConstrainedRealMedoid(members, threshold);
    if (!candidate) {
      throw new Error(`threshold-valid Family lost a constrained real medoid: ${family.id}`);
    }
    return {
      ...family,
      medoid: candidate.medoid,
      members,
      medoidCentrality: candidate.centrality,
      minimumSimilarityToMedoid: candidate.minimumSimilarity,
      medoidUpdateCount: family.medoidUpdateCount +
        (family.medoid.occurrence.id === candidate.medoid.occurrence.id ? 0 : 1)
    };
  });
  return { families, matchedFamilyIds: matches.map((match) => match.familyId) };
}

export function buildIncrementalCoarseFamilies(
  windows: readonly EmbeddedTrajectoryWindowV1[],
  thresholds: Readonly<Record<number, number>> = V15_COARSE_SIMILARITY_THRESHOLDS
): ProceduralWindowFamilyV1[] {
  let families: ProceduralWindowFamilyV1[] = [];
  const roleOrder: Record<ProceduralEvidenceRole, number> = {
    support: 0,
    counterexample: 1,
    unknown: 2
  };
  for (const window of orderedEmbeddedWindows(windows).sort((left, right) =>
    roleOrder[left.occurrence.evidenceRole] - roleOrder[right.occurrence.evidenceRole])) {
    families = ingestWindowIntoCoarseFamilies(families, window, thresholds).families;
  }
  return dedupeEquivalentFamilies(families);
}

export function clusterTrajectoryWindows(
  windows: readonly EmbeddedTrajectoryWindowV1[],
  options: {
    coarseSimilarityThresholds?: Readonly<Record<number, number>>;
    fineMatchConfigs?: readonly BandedMonotonicMatchConfig[];
    medoidSwitchMargin?: number;
  } = {}
): TrajectoryWindowClusteringResultV1 {
  const thresholds = options.coarseSimilarityThresholds ?? V15_COARSE_SIMILARITY_THRESHOLDS;
  const configs = options.fineMatchConfigs ?? V15_FINE_MATCH_CONFIGS;
  const configByScale = new Map(configs.map((config) => [config.scale, config]));
  const families = buildIncrementalCoarseFamilies(windows, thresholds);
  const clusters = families.flatMap((family) => {
    const config = configByScale.get(family.scale);
    if (!config) throw new Error(`missing fine match config for scale ${family.scale}`);
    return buildExclusiveFineClusters(
      family,
      config,
      options.medoidSwitchMargin ?? V15_FINE_MEDOID_SWITCH_MARGIN
    );
  });
  return {
    families: families.sort((left, right) => left.scale - right.scale ||
      right.members.length - left.members.length || left.id.localeCompare(right.id)),
    clusters: dedupeEquivalentClusters(clusters).sort((left, right) =>
      left.scale - right.scale ||
      right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
      right.averageSimilarity - left.averageSimilarity ||
      left.id.localeCompare(right.id))
  };
}

export function selectMaximalWindowClusters(
  clusters: readonly TrajectoryWindowClusterV1[]
): TrajectoryWindowClusterV1[] {
  const selected: TrajectoryWindowClusterV1[] = [];
  for (const cluster of [...clusters].sort((left, right) =>
    right.scale - left.scale ||
    right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
    right.averageSimilarity - left.averageSimilarity || left.id.localeCompare(right.id))) {
    const covering = selected.find((candidate) => clusterFullyCoveredBy(cluster, candidate));
    selected.push(covering ? { ...cluster, suppressedByClusterId: covering.id } : { ...cluster });
  }
  return selected;
}

/**
 * Canonical identity of one fine evidence set. Deliberately excludes the
 * coarse Family id: overlapping coarse recall may discover the same exact
 * fine cluster through several Families, but it must still materialize once.
 */
export function fineEvidenceSignature(cluster: TrajectoryWindowClusterV1): string {
  return `fine_evidence_${stableHash({
    version: FINE_EVIDENCE_SIGNATURE_VERSION,
    scale: cluster.scale,
    occurrenceIds: cluster.members.map((member) => member.occurrence.id).sort()
  }).slice(0, 32)}`;
}

/**
 * Extract medoid positions grounded by the requested number of successful
 * Episodes, then project that shared ordered core back to every aligned window.
 */
export function extractAlignedCommonCore(
  cluster: TrajectoryWindowClusterV1,
  options: {
    minSupportEpisodes?: number;
    minCoreSteps?: number;
  } = {}
): AlignedCommonCoreV1 | undefined {
  const minSupportEpisodes = options.minSupportEpisodes ?? V15_MIN_SUPPORT_EPISODES;
  if (!Number.isInteger(minSupportEpisodes) || minSupportEpisodes < 2) {
    throw new Error("aligned common core minSupportEpisodes must be >= 2");
  }
  const medoidMember = cluster.members.find((member) =>
    member.occurrence.id === cluster.medoidOccurrenceId);
  if (!medoidMember) throw new Error(`cluster medoid occurrence is missing: ${cluster.id}`);
  const medoidSteps = medoidMember.occurrence.steps;
  const evidenceByAnchor = new Map<number, Array<{
    member: TrajectoryWindowClusterMemberV1;
    stepOffset: number;
    similarity: number;
  }>>();

  for (const member of cluster.members) {
    if (member.occurrence.evidenceRole !== "support") continue;
    for (const pair of member.alignmentToMedoid.pairs) {
      const values = evidenceByAnchor.get(pair.rightIndex) ?? [];
      values.push({ member, stepOffset: pair.leftIndex, similarity: pair.similarity });
      evidenceByAnchor.set(pair.rightIndex, values);
    }
  }

  const anchorOffsets = [...evidenceByAnchor.entries()]
    .filter(([, evidence]) => new Set(evidence.map((item) =>
      item.member.occurrence.episodeId)).size >= minSupportEpisodes)
    .map(([offset]) => offset)
    .sort((left, right) => left - right);
  const defaultMinCoreSteps = Math.min(
    V15_FINE_MATCH_CONFIGS.find((config) => config.scale === cluster.scale)?.minMatchedSteps ?? 2,
    cluster.scale
  );
  const minCoreSteps = options.minCoreSteps ?? defaultMinCoreSteps;
  if (anchorOffsets.length < minCoreSteps) return undefined;

  const steps: AlignedCommonCoreStepV1[] = anchorOffsets.map((anchorOffset) => {
    const anchorStep = medoidSteps[anchorOffset];
    if (!anchorStep) throw new Error(`aligned common core anchor is outside medoid: ${anchorOffset}`);
    const evidence = evidenceByAnchor.get(anchorOffset) ?? [];
    return {
      anchorOffset,
      anchorStepId: anchorStep.id,
      intent: anchorStep.intent,
      summary: anchorStep.summary,
      supportEpisodeIds: [...new Set(evidence.map((item) =>
        item.member.occurrence.episodeId))].sort(),
      evidenceStepIds: [...new Set(evidence.map((item) =>
        item.member.occurrence.steps[item.stepOffset]?.id).filter(
          (id): id is string => Boolean(id)
        ))].sort()
    };
  });
  const commonCoreId = `aligned_common_core_${stableHash({
    version: PROCEDURAL_WINDOW_MINING_VERSION,
    clusterId: cluster.id,
    medoidOccurrenceId: cluster.medoidOccurrenceId,
    anchorOffsets,
    anchorStepIds: steps.map((step) => step.anchorStepId)
  }).slice(0, 24)}`;
  const anchorOffsetSet = new Set(anchorOffsets);
  const spanOccurrences = cluster.members.flatMap((member) => {
    const matchedPairs = member.alignmentToMedoid.pairs.filter((pair) =>
      anchorOffsetSet.has(pair.rightIndex));
    // A projected occurrence represents this complete core, not merely another
    // partially overlapping subset. Internal unmatched Steps remain explicit Gaps.
    if (matchedPairs.length < anchorOffsets.length) return [];
    const sortedPairs = [...matchedPairs].sort((left, right) =>
      left.leftIndex - right.leftIndex || left.rightIndex - right.rightIndex);
    const firstOffset = sortedPairs[0]!.leftIndex;
    const lastOffset = sortedPairs.at(-1)!.leftIndex;
    const matchedOffsetSet = new Set(sortedPairs.map((pair) => pair.leftIndex));
    const matchedSteps = sortedPairs.map((pair) => member.occurrence.steps[pair.leftIndex])
      .filter((step): step is ExecutionStepLiteV1 => Boolean(step));
    if (matchedSteps.length !== sortedPairs.length) return [];
    const gapSteps = member.occurrence.steps.slice(firstOffset, lastOffset + 1)
      .filter((_, offset) => !matchedOffsetSet.has(firstOffset + offset));
    const averageMatchSimilarity = average(sortedPairs.map((pair) => pair.similarity));
    const identity = {
      version: PROCEDURAL_WINDOW_MINING_VERSION,
      commonCoreId,
      sourceWindowOccurrenceId: member.occurrence.id,
      matchedStepIds: matchedSteps.map((step) => step.id)
    };
    const occurrence: ProceduralSpanOccurrenceV1 = {
      id: `procedural_span_occurrence_${stableHash(identity).slice(0, 24)}`,
      schemaVersion: PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
      commonCoreId,
      clusterId: cluster.id,
      sourceWindowOccurrenceId: member.occurrence.id,
      episodeId: member.occurrence.episodeId,
      pathId: member.occurrence.pathId,
      userId: member.occurrence.userId,
      evidenceRole: member.occurrence.evidenceRole,
      scale: member.occurrence.scale,
      startStepIndex: matchedSteps[0]!.stepIndex,
      endStepIndex: matchedSteps.at(-1)!.stepIndex,
      matchedStepIds: matchedSteps.map((step) => step.id),
      gapStepIds: gapSteps.map((step) => step.id),
      evidenceRefs: [...new Set(matchedSteps.flatMap((step) => step.evidenceRefs))].sort(),
      averageMatchSimilarity,
      ...(member.occurrence.terminalReward === undefined
        ? {}
        : { terminalReward: member.occurrence.terminalReward })
    };
    return [occurrence];
  });

  const projectedSupportEpisodeIds = [...new Set(spanOccurrences
    .filter((occurrence) => occurrence.evidenceRole === "support")
    .map((occurrence) => occurrence.episodeId))].sort();
  if (projectedSupportEpisodeIds.length < minSupportEpisodes) return undefined;

  return {
    id: commonCoreId,
    schemaVersion: ALIGNED_COMMON_CORE_SCHEMA_VERSION,
    clusterId: cluster.id,
    scale: cluster.scale,
    medoidOccurrenceId: cluster.medoidOccurrenceId,
    anchorOffsets,
    steps,
    supportEpisodeIds: projectedSupportEpisodeIds,
    spanOccurrences
  };
}

function projectCompletionSide(
  side: "prefix" | "suffix",
  reference: AnchoredCompletionOccurrenceCandidateV2,
  occurrences: readonly AnchoredCompletionOccurrenceCandidateV2[],
  minStepSimilarity: number,
  minSupportEpisodes: number
): {
  anchors: SharedExtensionAnchorV2[];
  projections: Map<string, CompletionStepProjectionV2[]>;
} {
  const referenceSteps = reference[side];
  const matches = new Map<string, ReturnType<typeof monotonicExtensionPairs>>();
  for (const occurrence of occurrences) {
    matches.set(occurrence.occurrenceId,
      occurrence.occurrenceId === reference.occurrenceId
        ? referenceSteps.map((_, index) => ({
            leftIndex: index,
            rightIndex: index,
            similarity: 1
          }))
        : monotonicExtensionPairs(
            occurrence[side].map((item) => item.vector),
            referenceSteps.map((item) => item.vector),
            minStepSimilarity
          ));
  }
  const evidenceByReferenceOffset = new Map<number, Map<string, {
    stepId: string;
    similarity: number;
  }>>();
  for (const occurrence of occurrences) {
    if (occurrence.evidenceRole !== "support") continue;
    for (const pair of matches.get(occurrence.occurrenceId) ?? []) {
      const step = occurrence[side][pair.leftIndex];
      if (!step) continue;
      const byEpisode = evidenceByReferenceOffset.get(pair.rightIndex) ?? new Map();
      const previous = byEpisode.get(occurrence.episodeId);
      if (!previous || pair.similarity > previous.similarity) {
        byEpisode.set(occurrence.episodeId, {
          stepId: step.stepId,
          similarity: pair.similarity
        });
      }
      evidenceByReferenceOffset.set(pair.rightIndex, byEpisode);
    }
  }
  const anchors = [...evidenceByReferenceOffset.entries()]
    .filter(([, byEpisode]) => byEpisode.size >= minSupportEpisodes)
    .sort(([left], [right]) => left - right)
    .map(([referenceOffset, byEpisode]) => {
      const evidence = [...byEpisode.entries()].sort(([left], [right]) =>
        left.localeCompare(right));
      return {
        anchorId: `extension_${side}_${referenceOffset}`,
        side,
        referenceOffset,
        referenceStepId: referenceSteps[referenceOffset]!.stepId,
        supportEpisodeIds: evidence.map(([episodeId]) => episodeId),
        evidenceStepIds: evidence.map(([, item]) => item.stepId),
        averageMatchSimilarity: average(evidence.map(([, item]) => item.similarity))
      } satisfies SharedExtensionAnchorV2;
    });
  const anchorByOffset = new Map(anchors.map((anchor) => [
    anchor.referenceOffset,
    anchor
  ]));
  const projections = new Map<string, CompletionStepProjectionV2[]>();
  for (const occurrence of occurrences) {
    const sharedByStepIndex = new Map<number, {
      anchor: SharedExtensionAnchorV2;
      similarity: number;
    }>();
    for (const pair of matches.get(occurrence.occurrenceId) ?? []) {
      const anchor = anchorByOffset.get(pair.rightIndex);
      if (anchor) sharedByStepIndex.set(pair.leftIndex, {
        anchor,
        similarity: pair.similarity
      });
    }
    projections.set(occurrence.occurrenceId, occurrence[side].map((step, index) => {
      const shared = sharedByStepIndex.get(index);
      return shared
        ? {
            stepId: step.stepId,
            role: "shared_extension",
            extensionAnchorId: shared.anchor.anchorId,
            matchSimilarity: shared.similarity
          }
        : { stepId: step.stepId, role: "local_context" };
    }));
  }
  return { anchors, projections };
}

function monotonicExtensionPairs(
  left: readonly (readonly number[])[],
  right: readonly (readonly number[])[],
  minStepSimilarity: number
): Array<{ leftIndex: number; rightIndex: number; similarity: number }> {
  const scores = Array.from({ length: left.length + 1 }, () =>
    Array.from({ length: right.length + 1 }, () => 0));
  const operations = Array.from({ length: left.length + 1 }, () =>
    Array.from({ length: right.length + 1 }, () => "start" as
      "start" | "match" | "skip_left" | "skip_right"));
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      const candidates: Array<{
        score: number;
        operation: "match" | "skip_left" | "skip_right";
        priority: number;
      }> = [
        {
          score: scores[leftIndex - 1]![rightIndex]!,
          operation: "skip_left",
          priority: 1
        },
        {
          score: scores[leftIndex]![rightIndex - 1]!,
          operation: "skip_right",
          priority: 0
        }
      ];
      const similarity = cosineSimilarity(left[leftIndex - 1]!, right[rightIndex - 1]!);
      if (similarity >= minStepSimilarity) {
        candidates.push({
          score: scores[leftIndex - 1]![rightIndex - 1]! + similarity,
          operation: "match",
          priority: 2
        });
      }
      const selected = candidates.sort((a, b) =>
        b.score - a.score || b.priority - a.priority)[0]!;
      scores[leftIndex]![rightIndex] = selected.score;
      operations[leftIndex]![rightIndex] = selected.operation;
    }
  }
  const pairs: Array<{ leftIndex: number; rightIndex: number; similarity: number }> = [];
  let leftIndex = left.length;
  let rightIndex = right.length;
  while (leftIndex > 0 && rightIndex > 0) {
    const operation = operations[leftIndex]![rightIndex]!;
    if (operation === "match") {
      pairs.push({
        leftIndex: leftIndex - 1,
        rightIndex: rightIndex - 1,
        similarity: cosineSimilarity(left[leftIndex - 1]!, right[rightIndex - 1]!)
      });
      leftIndex -= 1;
      rightIndex -= 1;
    } else if (operation === "skip_left") {
      leftIndex -= 1;
    } else {
      rightIndex -= 1;
    }
  }
  return pairs.reverse();
}

interface InternalFineCluster {
  id: string;
  familyId: string;
  scale: number;
  medoid: EmbeddedTrajectoryWindowV1;
  members: EmbeddedTrajectoryWindowV1[];
  medoidUpdateCount: number;
}

export function buildExclusiveFineClusters(
  family: ProceduralWindowFamilyV1,
  config: BandedMonotonicMatchConfig,
  medoidSwitchMargin: number
): TrajectoryWindowClusterV1[] {
  let internal: InternalFineCluster[] = [];
  for (const window of orderedEmbeddedWindows(family.members)) {
    let best: { cluster: InternalFineCluster; match: BandedMonotonicMatchResultV1 } | undefined;
    for (const cluster of internal) {
      const match = bandedMonotonicMatch(window.stepVectors, cluster.medoid.stepVectors, config);
      if (!match.admitted) continue;
      if (!best || match.score > best.match.score ||
          (match.score === best.match.score &&
            match.averageMatchSimilarity > best.match.averageMatchSimilarity)) {
        best = { cluster, match };
      }
    }
    if (!best) {
      internal.push({
        id: `trajectory_window_cluster_${stableHash({
          version: PROCEDURAL_WINDOW_MINING_VERSION,
          familyId: family.id,
          anchorOccurrenceId: window.occurrence.id
        }).slice(0, 24)}`,
        familyId: family.id,
        scale: family.scale,
        medoid: window,
        members: [window],
        medoidUpdateCount: 0
      });
      continue;
    }
    best.cluster.members.push(window);
    const candidate = selectFineRealMedoid(best.cluster.members, config);
    if (!candidate) continue;
    const currentCentrality = fineCentrality(best.cluster.medoid, best.cluster.members, config);
    const nextCentrality = fineCentrality(candidate, best.cluster.members, config);
    if (candidate.occurrence.id !== best.cluster.medoid.occurrence.id &&
        nextCentrality - currentCentrality > medoidSwitchMargin) {
      best.cluster.medoid = candidate;
      best.cluster.medoidUpdateCount += 1;
    }
  }

  internal = mergeCompatibleFineClusters(internal, config);
  return internal.flatMap((cluster) => {
    const finalized = finalizeFineCluster(cluster, config);
    return finalized ? [finalized] : [];
  });
}

function mergeCompatibleFineClusters(
  input: readonly InternalFineCluster[],
  config: BandedMonotonicMatchConfig
): InternalFineCluster[] {
  const clusters = input.map((cluster) => ({ ...cluster, members: [...cluster.members] }));
  while (true) {
    let best: { left: number; right: number; score: number } | undefined;
    for (let left = 0; left < clusters.length; left += 1) {
      for (let right = left + 1; right < clusters.length; right += 1) {
        const match = bandedMonotonicMatch(
          clusters[left]!.medoid.stepVectors,
          clusters[right]!.medoid.stepVectors,
          config
        );
        if (!match.admitted) continue;
        const members = uniqueEmbeddedWindows([
          ...clusters[left]!.members,
          ...clusters[right]!.members
        ]);
        if (!selectFineRealMedoid(members, config)) continue;
        if (!best || match.score > best.score) best = { left, right, score: match.score };
      }
    }
    if (!best) return clusters;
    const left = clusters[best.left]!;
    const right = clusters[best.right]!;
    const members = uniqueEmbeddedWindows([...left.members, ...right.members]);
    const medoid = selectFineRealMedoid(members, config)!;
    clusters[best.left] = {
      ...left,
      medoid,
      members,
      medoidUpdateCount: left.medoidUpdateCount + right.medoidUpdateCount +
        (left.medoid.occurrence.id === medoid.occurrence.id ? 0 : 1)
    };
    clusters.splice(best.right, 1);
  }
}

function finalizeFineCluster(
  cluster: InternalFineCluster,
  config: BandedMonotonicMatchConfig
): TrajectoryWindowClusterV1 | undefined {
  const nonOverlapping = dedupeOverlappingMembers(cluster.members, cluster.medoid, config);
  if (nonOverlapping.length === 0) return undefined;
  const medoid = selectFineRealMedoid(nonOverlapping, config) ??
    nonOverlapping.find((member) => member.occurrence.id === cluster.medoid.occurrence.id) ??
    nonOverlapping[0]!;
  const members = nonOverlapping.map((member): TrajectoryWindowClusterMemberV1 => {
    const alignmentToMedoid = member.occurrence.id === medoid.occurrence.id
      ? selfBandedMonotonicMatch(member.stepVectors, config)
      : bandedMonotonicMatch(member.stepVectors, medoid.stepVectors, config);
    return {
      occurrence: member.occurrence,
      similarityToMedoid: alignmentToMedoid.score,
      alignmentToMedoid
    };
  }).sort((left, right) => left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex);
  const episodeIds = distinctEpisodes(members);
  const supportEpisodeIds = distinctEpisodes(members.filter((member) =>
    member.occurrence.evidenceRole === "support"));
  const counterexampleEpisodeIds = distinctEpisodes(members.filter((member) =>
    member.occurrence.evidenceRole === "counterexample"));
  const unknownEpisodeIds = distinctEpisodes(members.filter((member) =>
    member.occurrence.evidenceRole === "unknown"));
  const similarities = members.map((member) => member.similarityToMedoid);
  return {
    id: cluster.id,
    familyId: cluster.familyId,
    scale: cluster.scale,
    medoidOccurrenceId: medoid.occurrence.id,
    episodeIds,
    supportEpisodeIds,
    counterexampleEpisodeIds,
    unknownEpisodeIds,
    occurrenceCount: members.length,
    averageSimilarity: average(similarities),
    minimumSimilarity: Math.min(...similarities),
    medoidCentrality: fineCentrality(medoid, nonOverlapping, config),
    medoidUpdateCount: cluster.medoidUpdateCount +
      (cluster.medoid.occurrence.id === medoid.occurrence.id ? 0 : 1),
    members
  };
}

function selectFineRealMedoid(
  members: readonly EmbeddedTrajectoryWindowV1[],
  config: BandedMonotonicMatchConfig
): EmbeddedTrajectoryWindowV1 | undefined {
  const hasSupport = members.some((member) =>
    member.occurrence.evidenceRole === "support");
  return [...members].filter((candidate) =>
    (!hasSupport || candidate.occurrence.evidenceRole === "support") && members.every((member) =>
    member.occurrence.id === candidate.occurrence.id ||
    bandedMonotonicMatch(member.stepVectors, candidate.stepVectors, config).admitted))
    .sort((left, right) =>
      fineCentrality(right, members, config) - fineCentrality(left, members, config) ||
      left.occurrence.id.localeCompare(right.occurrence.id))[0];
}

function fineCentrality(
  candidate: EmbeddedTrajectoryWindowV1,
  members: readonly EmbeddedTrajectoryWindowV1[],
  config: BandedMonotonicMatchConfig
): number {
  if (members.length <= 1) return 1;
  return average(members.filter((member) => member.occurrence.id !== candidate.occurrence.id)
    .map((member) => bandedMonotonicMatch(
      member.stepVectors,
      candidate.stepVectors,
      config
    ).score));
}

function dedupeOverlappingMembers(
  members: readonly EmbeddedTrajectoryWindowV1[],
  medoid: EmbeddedTrajectoryWindowV1,
  config: BandedMonotonicMatchConfig
): EmbeddedTrajectoryWindowV1[] {
  const byEpisode = new Map<string, EmbeddedTrajectoryWindowV1[]>();
  for (const member of members) {
    const values = byEpisode.get(member.occurrence.episodeId) ?? [];
    values.push(member);
    byEpisode.set(member.occurrence.episodeId, values);
  }
  const selected: EmbeddedTrajectoryWindowV1[] = [];
  for (const values of byEpisode.values()) {
    const kept: EmbeddedTrajectoryWindowV1[] = [];
    for (const member of [...values].sort((left, right) =>
      fineScoreToMedoid(right, medoid, config) - fineScoreToMedoid(left, medoid, config) ||
      left.occurrence.startStepIndex - right.occurrence.startStepIndex)) {
      if (kept.some((existing) => rangesOverlap(
        existing.occurrence.startStepIndex,
        existing.occurrence.endStepIndex,
        member.occurrence.startStepIndex,
        member.occurrence.endStepIndex
      ))) continue;
      kept.push(member);
    }
    selected.push(...kept);
  }
  return selected;
}

function fineScoreToMedoid(
  member: EmbeddedTrajectoryWindowV1,
  medoid: EmbeddedTrajectoryWindowV1,
  config: BandedMonotonicMatchConfig
): number {
  return member.occurrence.id === medoid.occurrence.id
    ? 1
    : bandedMonotonicMatch(member.stepVectors, medoid.stepVectors, config).score;
}

function clusterFullyCoveredBy(
  shorter: TrajectoryWindowClusterV1,
  longer: TrajectoryWindowClusterV1
): boolean {
  if (longer.scale <= shorter.scale) return false;
  if (shorter.supportEpisodeIds.length === 0) return false;
  const longerEpisodes = new Set(longer.supportEpisodeIds);
  if (!shorter.supportEpisodeIds.every((episodeId) => longerEpisodes.has(episodeId))) return false;
  const shorterSupport = shorter.members.filter((member) =>
    member.occurrence.evidenceRole === "support");
  const longerSupport = longer.members.filter((member) =>
    member.occurrence.evidenceRole === "support");
  return shorterSupport.every((member) => longerSupport.some((candidate) =>
    candidate.occurrence.episodeId === member.occurrence.episodeId &&
    candidate.occurrence.startStepIndex <= member.occurrence.startStepIndex &&
    candidate.occurrence.endStepIndex >= member.occurrence.endStepIndex));
}

function dedupeEquivalentFamilies(
  families: readonly ProceduralWindowFamilyV1[]
): ProceduralWindowFamilyV1[] {
  const byEvidence = new Map<string, ProceduralWindowFamilyV1>();
  for (const family of families) {
    const key = `${family.scale}:${family.members.map((member) =>
      member.occurrence.id).sort().join(",")}`;
    const existing = byEvidence.get(key);
    if (!existing || family.medoidCentrality > existing.medoidCentrality ||
        (family.medoidCentrality === existing.medoidCentrality &&
          family.id.localeCompare(existing.id) < 0)) {
      byEvidence.set(key, family);
    }
  }
  return [...byEvidence.values()];
}

function dedupeEquivalentClusters(
  clusters: readonly TrajectoryWindowClusterV1[]
): TrajectoryWindowClusterV1[] {
  const byEvidence = new Map<string, TrajectoryWindowClusterV1>();
  for (const cluster of clusters) {
    const key = `${cluster.scale}:${cluster.members.map((member) =>
      member.occurrence.id).sort().join(",")}`;
    const existing = byEvidence.get(key);
    if (!existing || cluster.averageSimilarity > existing.averageSimilarity ||
        (cluster.averageSimilarity === existing.averageSimilarity &&
          cluster.id.localeCompare(existing.id) < 0)) {
      byEvidence.set(key, cluster);
    }
  }
  return [...byEvidence.values()];
}

function orderedEmbeddedWindows(
  windows: readonly EmbeddedTrajectoryWindowV1[]
): EmbeddedTrajectoryWindowV1[] {
  const ordered = [...windows].sort((left, right) =>
    left.occurrence.scale - right.occurrence.scale ||
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex ||
    left.occurrence.id.localeCompare(right.occurrence.id));
  for (const window of ordered) validateEmbeddedWindow(window);
  return ordered;
}

function validateEmbeddedWindow(window: EmbeddedTrajectoryWindowV1): void {
  if (window.stepVectors.length !== window.occurrence.scale ||
      window.occurrence.steps.length !== window.occurrence.scale) {
    throw new Error(`embedded window Step count does not match scale: ${window.occurrence.id}`);
  }
  if (window.coarseVector.length === 0 || window.stepVectors.some((vector) => vector.length === 0)) {
    throw new Error(`embedded window contains an empty vector: ${window.occurrence.id}`);
  }
}

function validateWindowSpecs(specs: readonly MultiScaleWindowSpec[]): MultiScaleWindowSpec[] {
  if (specs.length === 0) throw new Error("at least one window spec is required");
  const normalized = specs.map((spec) => ({ length: spec.length, stride: spec.stride }));
  for (const spec of normalized) {
    if (!Number.isInteger(spec.length) || spec.length < 2) {
      throw new Error("window length must be an integer >= 2");
    }
    if (!Number.isInteger(spec.stride) || spec.stride < 1 || spec.stride > spec.length) {
      throw new Error("window stride must be an integer in [1, length]");
    }
  }
  return [...new Map(normalized.map((spec) => [`${spec.length}:${spec.stride}`, spec])).values()]
    .sort((left, right) => left.length - right.length || left.stride - right.stride);
}

function requiredScaleValue(
  values: Readonly<Record<number, number>>,
  scale: number,
  name: string
): number {
  const value = values[scale];
  if (!(value !== undefined && value > 0 && value <= 1)) {
    throw new Error(`missing or invalid ${name} for Span-${scale}`);
  }
  return value;
}

function cloneFamily(family: ProceduralWindowFamilyV1): ProceduralWindowFamilyV1 {
  return { ...family, members: [...family.members] };
}

function uniqueEmbeddedWindows(
  members: readonly EmbeddedTrajectoryWindowV1[]
): EmbeddedTrajectoryWindowV1[] {
  return [...new Map(members.map((member) => [member.occurrence.id, member])).values()];
}

function distinctEpisodes(
  members: readonly TrajectoryWindowClusterMemberV1[]
): string[] {
  return [...new Set(members.map((member) => member.occurrence.episodeId))].sort();
}

function rangesOverlap(
  leftStart: number,
  leftEnd: number,
  rightStart: number,
  rightEnd: number
): boolean {
  return leftStart <= rightEnd && rightStart <= leftEnd;
}

function average(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}
