import type { RawTurnRecord } from "../../storage/repositories.js";
import type { EpisodePolicyProjectionV1 } from "./episode-policy-projection-model.js";
import type { EpisodeProceduralPathV2 } from "./procedural-path-model.js";
import { stableHash } from "../../utils/id.js";
import { clip } from "../../utils/text.js";

export const EPISODE_CAPABILITY_SIGNATURE_SCHEMA_VERSION =
  "episode-capability-signature.v1" as const;
export const EPISODE_CAPABILITY_SIGNATURE_ALGORITHM_VERSION =
  "episode-capability-signature-builder.v1" as const;
export const EPISODE_CAPABILITY_AFFINITY_ALGORITHM_VERSION =
  "episode-capability-affinity.v1" as const;

export const EPISODE_FAMILY_GOAL_THRESHOLD = 0.78;
export const EPISODE_FAMILY_TRANSITION_THRESHOLD = 0.72;
export const EPISODE_FAMILY_COMBINED_THRESHOLD = 0.75;

export interface EpisodeCapabilityPolicyNodeV1 {
  nodeIndex: number;
  policyKey: string;
  policyVersionId: string;
  occurrenceId: string;
  spanId: string;
}

export interface EpisodeCapabilitySignatureV1 {
  id: string;
  schemaVersion: typeof EPISODE_CAPABILITY_SIGNATURE_SCHEMA_VERSION;
  algorithmVersion: typeof EPISODE_CAPABILITY_SIGNATURE_ALGORITHM_VERSION;
  projectionId: string;
  projectionHash: string;
  episodeId: string;
  pathId: string;
  pathHash: string;
  namespaceId: string;
  goalText: string;
  stateTransitionText: string;
  outcomeText: string;
  contextText: string;
  policyNodes: EpisodeCapabilityPolicyNodeV1[];
  signatureHash: string;
}

export interface EpisodeCapabilityVectorsV1 {
  goalVector: number[];
  stateTransitionVector: number[];
  outcomeVector: number[];
  contextVector: number[];
}

export interface EpisodeCapabilityAffinityV1 {
  id: string;
  algorithmVersion: typeof EPISODE_CAPABILITY_AFFINITY_ALGORITHM_VERSION;
  namespaceId: string;
  leftSignatureId: string;
  rightSignatureId: string;
  leftEpisodeId: string;
  rightEpisodeId: string;
  goalSimilarity: number;
  stateTransitionSimilarity: number;
  outcomeSimilarity: number;
  contextSimilarity: number;
  pathStructureSimilarity: number;
  combinedSimilarity: number;
  familyEligible: boolean;
}

export interface PolicyBackboneAlignmentV1 {
  policyKeys: string[];
  leftNodeIndexes: number[];
  rightNodeIndexes: number[];
}

export function buildEpisodeCapabilitySignature(input: {
  projection: EpisodePolicyProjectionV1;
  path: EpisodeProceduralPathV2;
  namespaceId: string;
  episodeTitle?: string;
  episodeSummary?: string;
  rawTurns: readonly RawTurnRecord[];
}): EpisodeCapabilitySignatureV1 {
  if (input.path.id !== input.projection.pathId ||
      input.path.pathHash !== input.projection.pathHash ||
      input.path.episodeId !== input.projection.episodeId) {
    throw new Error("Episode Capability signature source mismatch");
  }
  const namespaceId = input.namespaceId.trim();
  if (!namespaceId) throw new Error("Episode Capability signature requires namespaceId");
  const stateById = new Map(input.path.states.map((state) => [state.id, state]));
  const firstNode = input.projection.nodes[0];
  const lastNode = input.projection.nodes.at(-1);
  if (!firstNode || !lastNode) {
    throw new Error(`Episode Capability signature requires a non-empty path: ${input.path.id}`);
  }
  const initialState = stateById.get(firstNode.preStateId);
  const terminalState = stateById.get(lastNode.postStateId);
  if (!initialState || !terminalState) {
    throw new Error(`Episode Capability signature references missing states: ${input.path.id}`);
  }
  const rawTurns = [...input.rawTurns]
    .filter((turn) => turn.episodeId === input.projection.episodeId && !turn.deletedAt)
    .sort((left, right) => left.createdAt.localeCompare(right.createdAt) ||
      left.id.localeCompare(right.id));
  const firstUser = rawTurns.find((turn) => turn.userText?.trim())?.userText?.trim();
  const lastAssistant = [...rawTurns].reverse()
    .find((turn) => turn.assistantText?.trim())?.assistantText?.trim();
  const policyNodes = input.projection.nodes.flatMap((node): EpisodeCapabilityPolicyNodeV1[] =>
    node.assignment.kind === "policy" ? [{
      nodeIndex: node.nodeIndex,
      policyKey: node.assignment.policyKey,
      policyVersionId: node.assignment.policyVersionId,
      occurrenceId: node.occurrenceId,
      spanId: node.spanId
    }] : []
  );
  const goalText = compactParts([
    input.episodeTitle,
    input.episodeSummary,
    firstUser,
    initialState.goal?.subject,
    displayValue(initialState.goal?.value),
    ...input.projection.nodes.map((node) => node.localGoal)
  ], 8_000);
  const stateTransitionText = compactParts([
    `before: ${initialState.summary}`,
    `after: ${terminalState.summary}`,
    ...input.projection.nodes.map((node) =>
      `${node.entryCondition} -> ${node.exitCondition} [${node.terminationStatus}]`
    )
  ], 10_000);
  const outcomeText = compactParts([
    input.path.terminalReward === undefined
      ? "terminal_reward: unknown"
      : `terminal_reward: ${input.path.terminalReward}`,
    `task_status: ${terminalState.taskStatus}`,
    ...terminalState.verification.map((entry) =>
      `verification: ${entry.subject}=${displayValue(entry.value ?? entry.status)}`
    ),
    lastAssistant
  ], 5_000);
  const tools = unique(input.path.steps
    .map((step) => step.action.toolName)
    .filter((value): value is string => Boolean(value?.trim())));
  const actionTypes = unique(input.path.steps.map((step) => step.action.type));
  const contextText = compactParts([
    ...initialState.constraints.map((entry) =>
      `constraint: ${entry.subject}=${displayValue(entry.value ?? entry.status)}`
    ),
    ...initialState.artifacts.map((entry) =>
      `input_artifact: ${entry.subject}=${displayValue(entry.value ?? entry.status)}`
    ),
    ...terminalState.artifacts.map((entry) =>
      `output_artifact: ${entry.subject}=${displayValue(entry.value ?? entry.status)}`
    ),
    tools.length > 0 ? `tools: ${tools.join(", ")}` : undefined,
    actionTypes.length > 0 ? `action_types: ${actionTypes.join(", ")}` : undefined
  ], 6_000);
  const basis = {
    schemaVersion: EPISODE_CAPABILITY_SIGNATURE_SCHEMA_VERSION,
    algorithmVersion: EPISODE_CAPABILITY_SIGNATURE_ALGORITHM_VERSION,
    projectionId: input.projection.id,
    projectionHash: input.projection.projectionHash,
    episodeId: input.projection.episodeId,
    pathId: input.path.id,
    pathHash: input.path.pathHash,
    namespaceId,
    goalText,
    stateTransitionText,
    outcomeText,
    contextText,
    policyNodes
  };
  const signatureHash = stableHash(basis);
  return {
    id: `episode_capability_signature_${signatureHash.slice(0, 24)}`,
    ...basis,
    signatureHash
  };
}

export function computeEpisodeCapabilityAffinity(input: {
  left: EpisodeCapabilitySignatureV1;
  leftVectors: EpisodeCapabilityVectorsV1;
  right: EpisodeCapabilitySignatureV1;
  rightVectors: EpisodeCapabilityVectorsV1;
}): EpisodeCapabilityAffinityV1 {
  if (input.left.namespaceId !== input.right.namespaceId ||
      input.left.episodeId === input.right.episodeId) {
    throw new Error("Episode Capability affinity requires distinct Episodes in one namespace");
  }
  const [left, leftVectors, right, rightVectors] =
    input.left.id.localeCompare(input.right.id) <= 0
      ? [input.left, input.leftVectors, input.right, input.rightVectors]
      : [input.right, input.rightVectors, input.left, input.leftVectors];
  const goalSimilarity = cosine(leftVectors.goalVector, rightVectors.goalVector);
  const stateTransitionSimilarity = cosine(
    leftVectors.stateTransitionVector,
    rightVectors.stateTransitionVector
  );
  const outcomeSimilarity = cosine(leftVectors.outcomeVector, rightVectors.outcomeVector);
  const contextSimilarity = cosine(leftVectors.contextVector, rightVectors.contextVector);
  const pathStructureSimilarity = policyPathSimilarity(
    left.policyNodes.map((node) => node.policyKey),
    right.policyNodes.map((node) => node.policyKey)
  );
  const combinedSimilarity = round(
    0.30 * goalSimilarity +
    0.30 * stateTransitionSimilarity +
    0.15 * outcomeSimilarity +
    0.10 * contextSimilarity +
    0.15 * pathStructureSimilarity
  );
  const familyEligible = goalSimilarity >= EPISODE_FAMILY_GOAL_THRESHOLD &&
    stateTransitionSimilarity >= EPISODE_FAMILY_TRANSITION_THRESHOLD &&
    combinedSimilarity >= EPISODE_FAMILY_COMBINED_THRESHOLD;
  const basis = {
    algorithmVersion: EPISODE_CAPABILITY_AFFINITY_ALGORITHM_VERSION,
    namespaceId: left.namespaceId,
    leftSignatureId: left.id,
    rightSignatureId: right.id
  };
  return {
    id: `episode_capability_affinity_${stableHash(basis).slice(0, 24)}`,
    ...basis,
    leftEpisodeId: left.episodeId,
    rightEpisodeId: right.episodeId,
    goalSimilarity: round(goalSimilarity),
    stateTransitionSimilarity: round(stateTransitionSimilarity),
    outcomeSimilarity: round(outcomeSimilarity),
    contextSimilarity: round(contextSimilarity),
    pathStructureSimilarity: round(pathStructureSimilarity),
    combinedSimilarity,
    familyEligible
  };
}

export function alignPolicyBackbone(
  left: EpisodeCapabilitySignatureV1,
  right: EpisodeCapabilitySignatureV1
): PolicyBackboneAlignmentV1 {
  const leftKeys = left.policyNodes.map((node) => node.policyKey);
  const rightKeys = right.policyNodes.map((node) => node.policyKey);
  const pairs = longestCommonSubsequencePairs(leftKeys, rightKeys);
  return {
    policyKeys: pairs.map(([leftIndex]) => leftKeys[leftIndex]!),
    leftNodeIndexes: pairs.map(([leftIndex]) => left.policyNodes[leftIndex]!.nodeIndex),
    rightNodeIndexes: pairs.map(([, rightIndex]) => right.policyNodes[rightIndex]!.nodeIndex)
  };
}

export function policyPathSimilarity(
  leftPolicyKeys: readonly string[],
  rightPolicyKeys: readonly string[]
): number {
  if (leftPolicyKeys.length === 0 || rightPolicyKeys.length === 0) return 0;
  const common = longestCommonSubsequencePairs(leftPolicyKeys, rightPolicyKeys).length;
  return (2 * common) / (leftPolicyKeys.length + rightPolicyKeys.length);
}

function longestCommonSubsequencePairs(
  left: readonly string[],
  right: readonly string[]
): Array<[number, number]> {
  const lengths = Array.from({ length: left.length + 1 }, () =>
    Array<number>(right.length + 1).fill(0)
  );
  for (let leftIndex = left.length - 1; leftIndex >= 0; leftIndex -= 1) {
    for (let rightIndex = right.length - 1; rightIndex >= 0; rightIndex -= 1) {
      lengths[leftIndex]![rightIndex] = left[leftIndex] === right[rightIndex]
        ? 1 + lengths[leftIndex + 1]![rightIndex + 1]!
        : Math.max(
            lengths[leftIndex + 1]![rightIndex]!,
            lengths[leftIndex]![rightIndex + 1]!
          );
    }
  }
  const result: Array<[number, number]> = [];
  let leftIndex = 0;
  let rightIndex = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    if (left[leftIndex] === right[rightIndex]) {
      result.push([leftIndex, rightIndex]);
      leftIndex += 1;
      rightIndex += 1;
      continue;
    }
    const skipLeft = lengths[leftIndex + 1]![rightIndex]!;
    const skipRight = lengths[leftIndex]![rightIndex + 1]!;
    if (skipLeft >= skipRight) leftIndex += 1;
    else rightIndex += 1;
  }
  return result;
}

function cosine(left: readonly number[], right: readonly number[]): number {
  if (left.length === 0 || left.length !== right.length) return 0;
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index]!;
    const rightValue = right[index]!;
    if (!Number.isFinite(leftValue) || !Number.isFinite(rightValue)) return 0;
    dot += leftValue * rightValue;
    leftNorm += leftValue * leftValue;
    rightNorm += rightValue * rightValue;
  }
  if (leftNorm === 0 || rightNorm === 0) return 0;
  return Math.max(-1, Math.min(1, dot / Math.sqrt(leftNorm * rightNorm)));
}

function compactParts(parts: readonly (string | undefined)[], maxLength: number): string {
  return clip(unique(parts
    .map((part) => part?.trim())
    .filter((part): part is string => Boolean(part))).join("\n"), maxLength);
}

function displayValue(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
