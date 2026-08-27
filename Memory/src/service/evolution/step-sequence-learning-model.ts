import { stableHash } from "../../utils/id.js";
import type { ExecutionStepOutcome, ExecutionStepV1 } from "./procedural-path-model.js";

export const STEP_OCCURRENCE_SCHEMA_VERSION = "procedural-step-occurrence.v1" as const;
export const STEP_EMBEDDING_VERSION = "procedural-step-intent.v1" as const;
export const STEP_CLUSTER_ALGORITHM_VERSION = "procedural-step-center-cluster.v3" as const;
export const STEP_SEQUENCE_PATTERN_SCHEMA_VERSION = "step-sequence-pattern.v1" as const;
export const STEP_SEQUENCE_MINING_ALGORITHM_VERSION = "step-sequence-exact-contiguous.v1" as const;
export const STEP_SEQUENCE_POLICY_SCHEMA_VERSION = "step-sequence-policy.v1" as const;
export const STEP_SEQUENCE_POLICY_INDUCTION_VERSION = "step-sequence-policy-induction.v1" as const;
export const STEP_SEQUENCE_POLICY_PROMPT_VERSION = "step-sequence-policy-prompt.v1" as const;
export const STEP_SEQUENCE_POLICY_REPAIR_VERSION = "step-sequence-policy-repair.v1" as const;
export const STEP_SEQUENCE_POLICY_REPAIR_PROMPT_VERSION =
  "step-sequence-policy-repair-prompt.v1" as const;
export const EPISODE_STEP_POLICY_PROJECTION_SCHEMA_VERSION =
  "episode-step-policy-projection.v1" as const;
export const EPISODE_STEP_POLICY_PROJECTION_ALGORITHM_VERSION =
  "episode-step-policy-projection-builder.v1" as const;
export const STEP_POLICY_SKILL_PATTERN_SCHEMA_VERSION =
  "step-policy-skill-pattern.v1" as const;
export const STEP_POLICY_SKILL_MINING_ALGORITHM_VERSION =
  "step-policy-sequence-exact.v1" as const;
export const STEP_POLICY_SKILL_COMPILER_VERSION =
  "step-policy-sequence-skill-compiler.v1" as const;

export const STEP_CLUSTER_SIMILARITY_THRESHOLD = 0.8;
export const STEP_SEQUENCE_MIN_LENGTH = 2;
export const STEP_SEQUENCE_MAX_LENGTH = 6;
export const STEP_SEQUENCE_SUPPORT_THRESHOLD = 2;
export const STEP_POLICY_SKILL_MIN_LENGTH = 2;
export const STEP_POLICY_SKILL_MAX_LENGTH = 6;
export const STEP_POLICY_SKILL_SUPPORT_THRESHOLD = 2;

export type StepEvidenceRole = "support" | "counterexample" | "unknown";

export function stepEvidenceRoleFromReward(
  terminalReward: number | undefined,
  input: {
    successThreshold: number;
    failureThreshold: number;
  }
): StepEvidenceRole {
  if (input.successThreshold <= input.failureThreshold) {
    throw new Error("Step evidence success threshold must be greater than failure threshold");
  }
  if (terminalReward === undefined || !Number.isFinite(terminalReward)) return "unknown";
  if (terminalReward >= input.successThreshold) return "support";
  if (terminalReward <= input.failureThreshold) return "counterexample";
  return "unknown";
}

export interface ProceduralStepOccurrenceV1 {
  id: string;
  schemaVersion: typeof STEP_OCCURRENCE_SCHEMA_VERSION;
  pathId: string;
  pathHash: string;
  episodeId: string;
  userId: string;
  sessionId: string;
  namespaceId: string;
  stepId: string;
  stepIndex: number;
  rawTurnId: string;
  intent: string;
  summary: string;
  semanticText: string;
  semanticHash: string;
  outcome: ExecutionStepOutcome;
  toolName?: string;
  preStateId: string;
  postStateId: string;
  step: ExecutionStepV1;
  reconstructionAlgorithmVersion: string;
  createdAt: string;
}

export interface StepSequencePolicyV1 {
  id: string;
  schemaVersion: typeof STEP_SEQUENCE_POLICY_SCHEMA_VERSION;
  inductionVersion: string;
  policyKey: string;
  namespaceId: string;
  patternId: string;
  patternMembershipVersion: string;
  clusterIds: string[];
  title: string;
  goalPattern: string;
  triggerConditions: string[];
  procedureSteps: Array<{
    instruction: string;
    evidenceRefs: string[];
  }>;
  verificationSteps: Array<{
    check: string;
    successSignal: string;
    evidenceRefs: string[];
  }>;
  doNotApplyWhen: string[];
  evidenceOccurrenceIds: string[];
  supportEpisodeIds: string[];
  confidence: number;
  provenance: {
    promptVersion: string;
    evidenceHash: string;
    model?: string;
  };
  revision?: {
    basePolicyVersionId: string;
    repairIds: string[];
  };
  contentHash: string;
}

export interface EpisodeStepPolicyProjectionNodeV1 {
  nodeIndex: number;
  kind: "policy" | "unmapped";
  startStepIndex: number;
  endStepIndex: number;
  stepOccurrenceIds: string[];
  stepIds: string[];
  preStateId: string;
  postStateId: string;
  policyVersionId?: string;
  policyKey?: string;
  patternId?: string;
  patternOccurrenceId?: string;
  supportEpisodeCount?: number;
}

export interface EpisodeStepPolicyProjectionV1 {
  id: string;
  schemaVersion: typeof EPISODE_STEP_POLICY_PROJECTION_SCHEMA_VERSION;
  algorithmVersion: typeof EPISODE_STEP_POLICY_PROJECTION_ALGORITHM_VERSION;
  episodeId: string;
  pathId: string;
  pathHash: string;
  nodes: EpisodeStepPolicyProjectionNodeV1[];
  mappedStepCount: number;
  unmappedStepCount: number;
  projectionHash: string;
}

export interface StepPolicySequenceSkillDraftV1 {
  name: string;
  displayTitle: string;
  retrievalBlurb: string;
  triggerContext: string;
  summary: string;
  steps: Array<{
    title: string;
    body: string;
    evidenceRefs: string[];
  }>;
  verification: Array<{
    check: string;
    successSignal: string;
    evidenceRefs: string[];
  }>;
  doNotUseWhen: string[];
  tools: string[];
  tags: string[];
  evidenceOccurrenceIds: string[];
  confidence: number;
}

export function stepSemanticText(step: ExecutionStepV1): string {
  return `Intent: ${step.action.intent.trim()}`;
}

export function buildStepOccurrence(input: {
  pathId: string;
  pathHash: string;
  episodeId: string;
  userId: string;
  sessionId: string;
  namespaceId: string;
  reconstructionAlgorithmVersion: string;
  step: ExecutionStepV1;
  createdAt: string;
}): ProceduralStepOccurrenceV1 {
  const semanticText = stepSemanticText(input.step);
  const semanticHash = stableHash({
    version: STEP_EMBEDDING_VERSION,
    semanticText
  });
  const id = `procedural_step_occurrence_${stableHash({
    pathId: input.pathId,
    stepId: input.step.id
  }).slice(0, 24)}`;
  return {
    id,
    schemaVersion: STEP_OCCURRENCE_SCHEMA_VERSION,
    pathId: input.pathId,
    pathHash: input.pathHash,
    episodeId: input.episodeId,
    userId: input.userId,
    sessionId: input.sessionId,
    namespaceId: input.namespaceId,
    stepId: input.step.id,
    stepIndex: input.step.stepIndex,
    rawTurnId: input.step.rawTurnId,
    intent: input.step.action.intent.trim(),
    summary: input.step.action.summary.trim(),
    semanticText,
    semanticHash,
    outcome: input.step.outcome.status,
    ...(input.step.action.toolName ? { toolName: input.step.action.toolName } : {}),
    preStateId: input.step.preStateId,
    postStateId: input.step.postStateId,
    step: input.step,
    reconstructionAlgorithmVersion: input.reconstructionAlgorithmVersion,
    createdAt: input.createdAt
  };
}

export function stepSequenceIdentity(namespaceId: string, clusterIds: readonly string[]): {
  id: string;
  sequenceHash: string;
} {
  const sequenceHash = stableHash({
    algorithmVersion: STEP_SEQUENCE_MINING_ALGORITHM_VERSION,
    namespaceId,
    clusterIds
  });
  return {
    id: `step_sequence_pattern_${sequenceHash.slice(0, 24)}`,
    sequenceHash
  };
}

export function policySkillSequenceIdentity(namespaceId: string, policyKeys: readonly string[]): {
  id: string;
  sequenceHash: string;
} {
  const sequenceHash = stableHash({
    algorithmVersion: STEP_POLICY_SKILL_MINING_ALGORITHM_VERSION,
    namespaceId,
    policyKeys
  });
  return {
    id: `step_policy_skill_pattern_${sequenceHash.slice(0, 24)}`,
    sequenceHash
  };
}

export function buildStepSequencePolicy(input: {
  namespaceId: string;
  patternId: string;
  patternMembershipVersion: string;
  clusterIds: readonly string[];
  draft: Omit<StepSequencePolicyV1,
    "id" | "schemaVersion" | "inductionVersion" | "policyKey" | "namespaceId" |
    "patternId" | "patternMembershipVersion" | "clusterIds" | "provenance" | "contentHash">;
  model?: string;
}): StepSequencePolicyV1 {
  const evidenceHash = stableHash({
    patternId: input.patternId,
    membershipVersion: input.patternMembershipVersion,
    evidenceOccurrenceIds: [...input.draft.evidenceOccurrenceIds].sort()
  });
  const contentHash = stableHash({
    ...input.draft,
    clusterIds: input.clusterIds,
    evidenceHash
  });
  const policyKey = `policy:step-sequence:${stableHash({
    namespaceId: input.namespaceId,
    clusterIds: input.clusterIds
  }).slice(0, 24)}`;
  const id = `step_sequence_policy_${stableHash({
    patternId: input.patternId,
    membershipVersion: input.patternMembershipVersion,
    contentHash
  }).slice(0, 24)}`;
  return {
    id,
    schemaVersion: STEP_SEQUENCE_POLICY_SCHEMA_VERSION,
    inductionVersion: STEP_SEQUENCE_POLICY_INDUCTION_VERSION,
    policyKey,
    namespaceId: input.namespaceId,
    patternId: input.patternId,
    patternMembershipVersion: input.patternMembershipVersion,
    clusterIds: [...input.clusterIds],
    ...input.draft,
    provenance: {
      promptVersion: STEP_SEQUENCE_POLICY_PROMPT_VERSION,
      evidenceHash,
      ...(input.model ? { model: input.model } : {})
    },
    contentHash
  };
}

export function buildRepairedStepSequencePolicy(input: {
  base: StepSequencePolicyV1;
  draft: Omit<StepSequencePolicyV1,
    "id" | "schemaVersion" | "inductionVersion" | "policyKey" | "namespaceId" |
    "patternId" | "patternMembershipVersion" | "clusterIds" | "provenance" |
    "revision" | "contentHash">;
  repairId: string;
  model?: string;
}): StepSequencePolicyV1 {
  const repairIds = [...new Set([
    ...(input.base.revision?.repairIds ?? []),
    input.repairId
  ])];
  const evidenceHash = stableHash({
    basePolicyVersionId: input.base.id,
    repairIds,
    evidenceOccurrenceIds: [...input.draft.evidenceOccurrenceIds].sort()
  });
  const contentHash = stableHash({
    ...input.draft,
    clusterIds: input.base.clusterIds,
    evidenceHash,
    repairIds
  });
  const inductionVersion = `${STEP_SEQUENCE_POLICY_REPAIR_VERSION}:${stableHash(repairIds).slice(0, 24)}`;
  const id = `step_sequence_policy_${stableHash({
    patternId: input.base.patternId,
    membershipVersion: input.base.patternMembershipVersion,
    inductionVersion,
    contentHash
  }).slice(0, 24)}`;
  return {
    id,
    schemaVersion: STEP_SEQUENCE_POLICY_SCHEMA_VERSION,
    inductionVersion,
    policyKey: input.base.policyKey,
    namespaceId: input.base.namespaceId,
    patternId: input.base.patternId,
    patternMembershipVersion: input.base.patternMembershipVersion,
    clusterIds: [...input.base.clusterIds],
    ...input.draft,
    provenance: {
      promptVersion: STEP_SEQUENCE_POLICY_REPAIR_PROMPT_VERSION,
      evidenceHash,
      ...(input.model ? { model: input.model } : {})
    },
    revision: {
      basePolicyVersionId: input.base.id,
      repairIds
    },
    contentHash
  };
}

export function buildEpisodeStepPolicyProjection(input: {
  episodeId: string;
  pathId: string;
  pathHash: string;
  nodes: readonly Omit<EpisodeStepPolicyProjectionNodeV1, "nodeIndex">[];
  totalStepCount: number;
}): EpisodeStepPolicyProjectionV1 {
  const nodes = [...input.nodes]
    .sort((left, right) => left.startStepIndex - right.startStepIndex ||
      left.endStepIndex - right.endStepIndex)
    .map((node, nodeIndex) => ({ ...node, nodeIndex }));
  const mappedStepCount = nodes
    .filter((node) => node.kind === "policy")
    .reduce((sum, node) => sum + node.stepOccurrenceIds.length, 0);
  const unmappedStepCount = input.totalStepCount - mappedStepCount;
  const projectionHash = stableHash({
    schemaVersion: EPISODE_STEP_POLICY_PROJECTION_SCHEMA_VERSION,
    algorithmVersion: EPISODE_STEP_POLICY_PROJECTION_ALGORITHM_VERSION,
    episodeId: input.episodeId,
    pathId: input.pathId,
    pathHash: input.pathHash,
    nodes,
    mappedStepCount,
    unmappedStepCount
  });
  return {
    id: `episode_step_policy_projection_${projectionHash.slice(0, 24)}`,
    schemaVersion: EPISODE_STEP_POLICY_PROJECTION_SCHEMA_VERSION,
    algorithmVersion: EPISODE_STEP_POLICY_PROJECTION_ALGORITHM_VERSION,
    episodeId: input.episodeId,
    pathId: input.pathId,
    pathHash: input.pathHash,
    nodes,
    mappedStepCount,
    unmappedStepCount,
    projectionHash
  };
}

export function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  if (left.length === 0 || left.length !== right.length) return -1;
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index]!;
    const b = right[index]!;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  if (leftNorm === 0 || rightNorm === 0) return -1;
  return Math.max(-1, Math.min(1, dot / Math.sqrt(leftNorm * rightNorm)));
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

export function contiguousWindows<T>(
  values: readonly T[],
  minLength: number,
  maxLength: number
): Array<{ start: number; end: number; values: T[] }> {
  const windows: Array<{ start: number; end: number; values: T[] }> = [];
  for (let start = 0; start < values.length; start += 1) {
    for (let length = minLength;
      length <= maxLength && start + length <= values.length;
      length += 1) {
      windows.push({
        start,
        end: start + length - 1,
        values: values.slice(start, start + length)
      });
    }
  }
  return windows;
}

export function hasMultipleDistinctValues(values: readonly string[]): boolean {
  return new Set(values).size >= 2;
}

export interface SequenceIntervalCandidate {
  id: string;
  startIndex: number;
  endIndex: number;
  sequenceLength: number;
  support: number;
}

export interface SequenceOccurrenceInterval {
  episodeId: string;
  pathId: string;
  startIndex: number;
  endIndex: number;
}

/**
 * Selects the longest locally maximal, non-overlapping sequence occurrences.
 * Nested candidates are removed before weighted interval scheduling, so a
 * shorter prefix/suffix cannot compete with a longer occurrence that contains it.
 */
export function selectLongestNonOverlapping<T extends SequenceIntervalCandidate>(
  candidates: readonly T[]
): T[] {
  const locallyMaximal = candidates.filter((candidate) => !candidates.some((other) =>
    other.id !== candidate.id &&
    other.startIndex <= candidate.startIndex &&
    other.endIndex >= candidate.endIndex &&
    other.sequenceLength > candidate.sequenceLength));
  const sorted = [...locallyMaximal].sort((left, right) =>
    left.endIndex - right.endIndex ||
    left.startIndex - right.startIndex ||
    right.sequenceLength - left.sequenceLength ||
    right.support - left.support ||
    left.id.localeCompare(right.id));
  const previous = sorted.map((candidate, index) => {
    let low = 0;
    let high = index - 1;
    let compatible = -1;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      if (sorted[middle]!.endIndex < candidate.startIndex) {
        compatible = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return compatible;
  });
  type Selection = {
    items: T[];
    totalLength: number;
    squaredLength: number;
    totalSupport: number;
  };
  const empty = (): Selection => ({
    items: [],
    totalLength: 0,
    squaredLength: 0,
    totalSupport: 0
  });
  const best: Selection[] = [];
  for (const [index, candidate] of sorted.entries()) {
    const without = index === 0 ? empty() : best[index - 1]!;
    const base = previous[index]! < 0 ? empty() : best[previous[index]!]!;
    const withCandidate: Selection = {
      items: [...base.items, candidate],
      totalLength: base.totalLength + candidate.sequenceLength,
      squaredLength: base.squaredLength + candidate.sequenceLength ** 2,
      totalSupport: base.totalSupport + candidate.support
    };
    best.push(compareSelections(withCandidate, without) >= 0 ? withCandidate : without);
  }
  return (best.at(-1)?.items ?? []).sort((left, right) =>
    left.startIndex - right.startIndex || left.endIndex - right.endIndex ||
    left.id.localeCompare(right.id));
}

export function sequenceOccurrencesFullyCovered(input: {
  shorterSequence: readonly string[];
  longerSequence: readonly string[];
  shorterOccurrences: readonly SequenceOccurrenceInterval[];
  longerOccurrences: readonly SequenceOccurrenceInterval[];
}): boolean {
  if (input.shorterOccurrences.length === 0 ||
      input.longerSequence.length <= input.shorterSequence.length ||
      !containsContiguousSequence(input.longerSequence, input.shorterSequence)) {
    return false;
  }
  return input.shorterOccurrences.every((shorter) => input.longerOccurrences.some((longer) =>
    longer.episodeId === shorter.episodeId &&
    longer.pathId === shorter.pathId &&
    longer.startIndex <= shorter.startIndex &&
    longer.endIndex >= shorter.endIndex));
}

export function containsContiguousSequence(
  haystack: readonly string[],
  needle: readonly string[]
): boolean {
  if (needle.length > haystack.length) return false;
  for (let start = 0; start + needle.length <= haystack.length; start += 1) {
    if (needle.every((value, index) => haystack[start + index] === value)) return true;
  }
  return false;
}

function compareSelections(
  left: {
    items: readonly SequenceIntervalCandidate[];
    totalLength: number;
    squaredLength: number;
    totalSupport: number;
  },
  right: {
    items: readonly SequenceIntervalCandidate[];
    totalLength: number;
    squaredLength: number;
    totalSupport: number;
  }
): number {
  if (left.totalLength !== right.totalLength) return left.totalLength - right.totalLength;
  if (left.squaredLength !== right.squaredLength) return left.squaredLength - right.squaredLength;
  if (left.totalSupport !== right.totalSupport) return left.totalSupport - right.totalSupport;
  if (left.items.length !== right.items.length) return right.items.length - left.items.length;
  const leftKey = left.items.map((item) => item.id).sort().join("\u0000");
  const rightKey = right.items.map((item) => item.id).sort().join("\u0000");
  return rightKey.localeCompare(leftKey);
}
