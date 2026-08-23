import { stableHash } from "../../utils/id.js";
import type { EpisodePolicyProjectionV1 } from "./episode-policy-projection-model.js";

export const POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION = "policy-sequence-pattern.v2" as const;
export const POLICY_SEQUENCE_MINING_ALGORITHM_VERSION =
  "policy-sequence-pattern-miner.v2" as const;
export const PROCEDURAL_SKILL_CANDIDATE_SCHEMA_VERSION =
  "procedural-skill-candidate.v2" as const;
export const PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION =
  "procedural-skill-candidate-dual-discovery.v2" as const;

export const POLICY_SEQUENCE_MIN_LENGTH = 2;
export const POLICY_SEQUENCE_MAX_LENGTH = 6;
export const POLICY_SEQUENCE_OBSERVED_SUPPORT = 2;
export const POLICY_SEQUENCE_READY_SUPPORT = 2;

export type PolicySequenceEvidenceRole = "support" | "counterexample" | "uncertain";
export type CapabilityDiscoverySource = "episode_similarity" | "policy_sequence_similarity";
export type CapabilityType = "task_skill" | "sub_skill";
export type PolicySequencePatternLifecycleStatus = "forming" | "observed" | "ready" | "stale";
export type ProceduralSkillCandidateLifecycleStatus = "observed" | "ready";

export interface PolicySequenceCostV1 {
  steps: number;
  toolCalls: number;
  retryCount: number;
  recoveryCount: number;
  errorCount: number;
  tokens?: number;
  latencyMs?: number;
}

export interface PolicySequencePatternIdentityV1 {
  patternId: string;
  sequenceHash: string;
  namespaceId: string;
  policyKeys: string[];
  capabilityType: CapabilityType;
  episodeFamilyId?: string;
}

export interface EpisodeAffinityEvidenceV1 {
  affinityId: string;
  peerSignatureId: string;
  peerEpisodeId: string;
  goalSimilarity: number;
  stateTransitionSimilarity: number;
  outcomeSimilarity: number;
  contextSimilarity: number;
  pathStructureSimilarity: number;
  combinedSimilarity: number;
}

export interface PolicySequencePatternOccurrenceV1 {
  id: string;
  schemaVersion: typeof POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION;
  algorithmVersion: typeof POLICY_SEQUENCE_MINING_ALGORITHM_VERSION;
  patternId: string;
  sequenceHash: string;
  namespaceId: string;
  projectionId: string;
  episodeId: string;
  pathId: string;
  sessionId: string;
  startNodeIndex: number;
  endNodeIndex: number;
  matchedNodeIndexes: number[];
  pathNodeIndexes: number[];
  policyKeys: string[];
  policyVersionIds: string[];
  spanOccurrenceIds: string[];
  spanIds: string[];
  pathSpanOccurrenceIds: string[];
  pathSpanIds: string[];
  preStateId: string;
  postStateId: string;
  evidenceRole: PolicySequenceEvidenceRole;
  capabilityType: CapabilityType;
  discoverySources: CapabilityDiscoverySource[];
  episodeFamilyId?: string;
  episodeAffinity?: EpisodeAffinityEvidenceV1;
  terminalReward?: number;
  cost: PolicySequenceCostV1;
}

export interface ExtractPolicySequencePatternsInput {
  projection: EpisodePolicyProjectionV1;
  namespaceId: string;
  sessionId: string;
  terminalReward?: number;
  spanCostsByOccurrenceId: ReadonlyMap<string, PolicySequenceCostV1>;
  minLength?: number;
  maxLength?: number;
}

export interface PolicySequencePatternTopologyInput {
  patternId: string;
  policyKeys: string[];
  supportEpisodeIds: string[];
  capabilityType: CapabilityType;
  episodeFamilyId?: string;
}

export interface PolicySequencePatternTopologyV1 {
  patternId: string;
  isClosed: boolean;
  isMaximal: boolean;
}

export interface ProceduralSkillCandidateEvidenceV1 {
  occurrenceId: string;
  projectionId: string;
  episodeId: string;
  pathId: string;
  sessionId: string;
  startNodeIndex: number;
  endNodeIndex: number;
  matchedNodeIndexes: number[];
  pathNodeIndexes: number[];
  policyVersionIds: string[];
  spanOccurrenceIds: string[];
  spanIds: string[];
  pathSpanOccurrenceIds: string[];
  pathSpanIds: string[];
  preStateId: string;
  postStateId: string;
  evidenceRole: PolicySequenceEvidenceRole;
  discoverySources: CapabilityDiscoverySource[];
  episodeAffinity?: EpisodeAffinityEvidenceV1;
  terminalReward?: number;
  cost: PolicySequenceCostV1;
}

export interface ProceduralSkillCandidateV1 {
  id: string;
  candidateKey: string;
  schemaVersion: typeof PROCEDURAL_SKILL_CANDIDATE_SCHEMA_VERSION;
  inductionVersion: typeof PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION;
  patternId: string;
  patternMembershipVersion: string;
  sequenceHash: string;
  namespaceId: string;
  policyKeys: string[];
  capabilityType: CapabilityType;
  discoverySources: CapabilityDiscoverySource[];
  episodeFamilyId?: string;
  lifecycleStatus: ProceduralSkillCandidateLifecycleStatus;
  isClosed: true;
  isMaximal: boolean;
  supportEpisodeIds: string[];
  counterexampleEpisodeIds: string[];
  uncertainEpisodeIds: string[];
  sourceProjectionIds: string[];
  sourcePathIds: string[];
  sourceSpanOccurrenceIds: string[];
  sourceSpanIds: string[];
  aggregateSupportCost: {
    total: PolicySequenceCostV1;
    meanPerOccurrence: PolicySequenceCostV1;
  };
  evidence: ProceduralSkillCandidateEvidenceV1[];
  evidenceHash: string;
  provenance: {
    miningAlgorithmVersion: typeof POLICY_SEQUENCE_MINING_ALGORITHM_VERSION;
    observedSupportThreshold: typeof POLICY_SEQUENCE_OBSERVED_SUPPORT;
    readySupportThreshold: typeof POLICY_SEQUENCE_READY_SUPPORT;
    executable: false;
  };
}

export function policySequencePatternIdentity(
  namespaceId: string,
  policyKeys: readonly string[],
  options: {
    capabilityType?: CapabilityType;
    episodeFamilyId?: string;
  } = {}
): PolicySequencePatternIdentityV1 {
  const normalizedNamespace = namespaceId.trim();
  const normalizedKeys = policyKeys.map((key) => key.trim());
  if (!normalizedNamespace || normalizedKeys.length < POLICY_SEQUENCE_MIN_LENGTH ||
      normalizedKeys.some((key) => !key)) {
    throw new Error("Policy sequence identity requires a namespace and at least two Policy keys");
  }
  const capabilityType = options.capabilityType ?? "sub_skill";
  const episodeFamilyId = options.episodeFamilyId?.trim() || undefined;
  if (capabilityType === "task_skill" && !episodeFamilyId) {
    throw new Error("Task-skill Policy sequence identity requires an Episode Family");
  }
  if (capabilityType === "sub_skill" && episodeFamilyId) {
    throw new Error("Sub-skill Policy sequence identity cannot be scoped to an Episode Family");
  }
  const sequenceHash = stableHash({
    algorithmVersion: POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
    namespaceId: normalizedNamespace,
    policyKeys: normalizedKeys,
    capabilityType,
    ...(episodeFamilyId ? { episodeFamilyId } : {})
  });
  return {
    patternId: `policy_sequence_pattern_${sequenceHash.slice(0, 20)}`,
    sequenceHash,
    namespaceId: normalizedNamespace,
    policyKeys: normalizedKeys,
    capabilityType,
    ...(episodeFamilyId ? { episodeFamilyId } : {})
  };
}

export function hasMultipleDistinctPolicies(policyKeys: readonly string[]): boolean {
  return new Set(policyKeys
    .map((policyKey) => policyKey.trim())
    .filter(Boolean)).size >= 2;
}

export function extractPolicySequencePatternOccurrences(
  input: ExtractPolicySequencePatternsInput
): PolicySequencePatternOccurrenceV1[] {
  const minLength = Math.max(POLICY_SEQUENCE_MIN_LENGTH, Math.trunc(
    input.minLength ?? POLICY_SEQUENCE_MIN_LENGTH
  ));
  const maxLength = Math.min(POLICY_SEQUENCE_MAX_LENGTH, Math.trunc(
    input.maxLength ?? POLICY_SEQUENCE_MAX_LENGTH
  ));
  if (minLength > maxLength) return [];
  const segments: typeof input.projection.nodes[] = [];
  let current: typeof input.projection.nodes = [];
  for (const node of input.projection.nodes) {
    if (node.assignment.kind === "unmapped") {
      if (current.length > 0) segments.push(current);
      current = [];
      continue;
    }
    current.push(node);
  }
  if (current.length > 0) segments.push(current);

  const results: PolicySequencePatternOccurrenceV1[] = [];
  for (const segment of segments) {
    for (let start = 0; start < segment.length; start += 1) {
      const maxWindowLength = Math.min(maxLength, segment.length - start);
      for (let length = minLength; length <= maxWindowLength; length += 1) {
        const window = segment.slice(start, start + length);
        const first = window[0]!;
        const last = window[window.length - 1]!;
        const assignments = window.map((node) => {
          if (node.assignment.kind !== "policy") {
            throw new Error("UNMAPPED node leaked into a Policy sequence window");
          }
          return node.assignment;
        });
        const policyKeys = assignments.map((assignment) => assignment.policyKey);
        if (!hasMultipleDistinctPolicies(policyKeys)) continue;
        const identity = policySequencePatternIdentity(
          input.namespaceId,
          policyKeys
        );
        const cost = window.reduce((total, node) => addCost(
          total,
          input.spanCostsByOccurrenceId.get(node.occurrenceId) ?? emptyCost()
        ), emptyCost());
        const evidenceRole = sequenceEvidenceRole(
          assignments.map((assignment) => assignment.evidenceRole),
          input.terminalReward
        );
        const idHash = stableHash({
          patternId: identity.patternId,
          projectionId: input.projection.id,
          matchedNodeIndexes: window.map((node) => node.nodeIndex)
        });
        results.push({
          id: `policy_sequence_occurrence_${idHash.slice(0, 20)}`,
          schemaVersion: POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION,
          algorithmVersion: POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
          patternId: identity.patternId,
          sequenceHash: identity.sequenceHash,
          namespaceId: identity.namespaceId,
          projectionId: input.projection.id,
          episodeId: input.projection.episodeId,
          pathId: input.projection.pathId,
          sessionId: input.sessionId,
          startNodeIndex: first.nodeIndex,
          endNodeIndex: last.nodeIndex,
          matchedNodeIndexes: window.map((node) => node.nodeIndex),
          pathNodeIndexes: window.map((node) => node.nodeIndex),
          policyKeys: [...identity.policyKeys],
          policyVersionIds: assignments.map((assignment) => assignment.policyVersionId),
          spanOccurrenceIds: window.map((node) => node.occurrenceId),
          spanIds: window.map((node) => node.spanId),
          pathSpanOccurrenceIds: window.map((node) => node.occurrenceId),
          pathSpanIds: window.map((node) => node.spanId),
          preStateId: first.preStateId,
          postStateId: last.postStateId,
          evidenceRole,
          capabilityType: "sub_skill",
          discoverySources: ["policy_sequence_similarity"],
          ...(input.terminalReward === undefined ? {} : {
            terminalReward: input.terminalReward
          }),
          cost
        });
      }
    }
  }
  return results;
}

export function extractEpisodeSimilarityPatternOccurrences(input: {
  projection: EpisodePolicyProjectionV1;
  namespaceId: string;
  sessionId: string;
  episodeFamilyId: string;
  patternCapabilityType?: CapabilityType;
  matchedNodeIndexes: readonly number[];
  episodeAffinity: EpisodeAffinityEvidenceV1;
  terminalReward?: number;
  spanCostsByOccurrenceId: ReadonlyMap<string, PolicySequenceCostV1>;
  minLength?: number;
  maxLength?: number;
}): PolicySequencePatternOccurrenceV1[] {
  const minLength = Math.max(POLICY_SEQUENCE_MIN_LENGTH, Math.trunc(
    input.minLength ?? POLICY_SEQUENCE_MIN_LENGTH
  ));
  const maxLength = Math.min(POLICY_SEQUENCE_MAX_LENGTH, Math.trunc(
    input.maxLength ?? POLICY_SEQUENCE_MAX_LENGTH
  ));
  const indexes = [...input.matchedNodeIndexes];
  if (minLength > maxLength || indexes.length < minLength ||
      indexes.some((value, index) => !Number.isInteger(value) || value < 0 ||
        (index > 0 && value <= indexes[index - 1]!))) {
    return [];
  }
  const nodesByIndex = new Map(input.projection.nodes.map((node) => [node.nodeIndex, node]));
  const matchedNodes = indexes.map((index) => nodesByIndex.get(index));
  if (matchedNodes.some((node) => !node || node.assignment.kind !== "policy")) {
    throw new Error(`Episode-similarity alignment references an unmapped node: ${input.projection.id}`);
  }
  const results: PolicySequencePatternOccurrenceV1[] = [];
  for (let start = 0; start < matchedNodes.length; start += 1) {
    const maxWindowLength = Math.min(maxLength, matchedNodes.length - start);
    for (let length = minLength; length <= maxWindowLength; length += 1) {
      const matched = matchedNodes.slice(start, start + length)
        .filter((node): node is NonNullable<typeof node> => Boolean(node));
      const first = matched[0]!;
      const last = matched.at(-1)!;
      const fullPath = input.projection.nodes.filter((node) =>
        node.nodeIndex >= first.nodeIndex && node.nodeIndex <= last.nodeIndex
      );
      const assignments = matched.map((node) => {
        if (node.assignment.kind !== "policy") {
          throw new Error("UNMAPPED node leaked into an Episode-similarity backbone");
        }
        return node.assignment;
      });
      const policyKeys = assignments.map((assignment) => assignment.policyKey);
      if (!hasMultipleDistinctPolicies(policyKeys)) continue;
      const patternCapabilityType = input.patternCapabilityType ?? "task_skill";
      const identity = policySequencePatternIdentity(
        input.namespaceId,
        policyKeys,
        patternCapabilityType === "task_skill"
          ? { capabilityType: "task_skill", episodeFamilyId: input.episodeFamilyId }
          : { capabilityType: "sub_skill" }
      );
      const cost = fullPath.reduce((total, node) => addCost(
        total,
        input.spanCostsByOccurrenceId.get(node.occurrenceId) ?? emptyCost()
      ), emptyCost());
      const evidenceRole = sequenceEvidenceRole(
        assignments.map((assignment) => assignment.evidenceRole),
        input.terminalReward
      );
      const matchedIndexes = matched.map((node) => node.nodeIndex);
      const idHash = stableHash({
        patternId: identity.patternId,
        projectionId: input.projection.id,
        matchedNodeIndexes: matchedIndexes
      });
      results.push({
        id: `policy_sequence_occurrence_${idHash.slice(0, 20)}`,
        schemaVersion: POLICY_SEQUENCE_PATTERN_SCHEMA_VERSION,
        algorithmVersion: POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
        patternId: identity.patternId,
        sequenceHash: identity.sequenceHash,
        namespaceId: identity.namespaceId,
        projectionId: input.projection.id,
        episodeId: input.projection.episodeId,
        pathId: input.projection.pathId,
        sessionId: input.sessionId,
        startNodeIndex: first.nodeIndex,
        endNodeIndex: last.nodeIndex,
        matchedNodeIndexes: matchedIndexes,
        pathNodeIndexes: fullPath.map((node) => node.nodeIndex),
        policyKeys: [...identity.policyKeys],
        policyVersionIds: assignments.map((assignment) => assignment.policyVersionId),
        spanOccurrenceIds: matched.map((node) => node.occurrenceId),
        spanIds: matched.map((node) => node.spanId),
        pathSpanOccurrenceIds: fullPath.map((node) => node.occurrenceId),
        pathSpanIds: fullPath.map((node) => node.spanId),
        preStateId: first.preStateId,
        postStateId: last.postStateId,
        evidenceRole,
        capabilityType: patternCapabilityType,
        discoverySources: ["episode_similarity"],
        episodeFamilyId: input.episodeFamilyId,
        episodeAffinity: { ...input.episodeAffinity },
        ...(input.terminalReward === undefined ? {} : {
          terminalReward: input.terminalReward
        }),
        cost
      });
    }
  }
  return results;
}

export function mergePolicySequenceOccurrences(
  occurrences: readonly PolicySequencePatternOccurrenceV1[]
): PolicySequencePatternOccurrenceV1[] {
  const byId = new Map<string, PolicySequencePatternOccurrenceV1>();
  for (const occurrence of occurrences) {
    const existing = byId.get(occurrence.id);
    if (!existing) {
      byId.set(occurrence.id, occurrence);
      continue;
    }
    if (existing.patternId !== occurrence.patternId ||
        existing.projectionId !== occurrence.projectionId ||
        existing.evidenceRole !== occurrence.evidenceRole) {
      throw new Error(`conflicting Policy sequence occurrence: ${occurrence.id}`);
    }
    byId.set(occurrence.id, {
      ...existing,
      discoverySources: unique([
        ...existing.discoverySources,
        ...occurrence.discoverySources
      ]) as CapabilityDiscoverySource[],
      episodeFamilyId: existing.episodeFamilyId ?? occurrence.episodeFamilyId,
      episodeAffinity: strongerAffinity(existing.episodeAffinity, occurrence.episodeAffinity)
    });
  }
  return [...byId.values()].sort(compareOccurrence);
}

export function classifyPolicySequencePatternTopology(
  patterns: readonly PolicySequencePatternTopologyInput[],
  observedSupportThreshold = POLICY_SEQUENCE_OBSERVED_SUPPORT
): PolicySequencePatternTopologyV1[] {
  const bySequence = new Map(patterns.map((pattern) => [
    sequenceMapKey(pattern.policyKeys, pattern.capabilityType, pattern.episodeFamilyId),
    pattern
  ]));
  const extensionsByPattern = new Map<string, Map<string, PolicySequencePatternTopologyInput>>();
  for (const extension of patterns) {
    if (extension.policyKeys.length <= POLICY_SEQUENCE_MIN_LENGTH) continue;
    for (const parentKeys of [extension.policyKeys.slice(1), extension.policyKeys.slice(0, -1)]) {
      const parent = bySequence.get(sequenceMapKey(
        parentKeys,
        extension.capabilityType,
        extension.episodeFamilyId
      ));
      if (!parent) continue;
      const values = extensionsByPattern.get(parent.patternId) ?? new Map();
      values.set(extension.patternId, extension);
      extensionsByPattern.set(parent.patternId, values);
    }
  }
  return patterns.map((pattern) => {
    const support = new Set(pattern.supportEpisodeIds);
    const extensions = [...(extensionsByPattern.get(pattern.patternId)?.values() ?? [])];
    return {
      patternId: pattern.patternId,
      isClosed: !extensions.some((extension) =>
        setEquals(support, new Set(extension.supportEpisodeIds))
      ),
      isMaximal: !extensions.some((extension) =>
        new Set(extension.supportEpisodeIds).size >= observedSupportThreshold
      )
    };
  });
}

export function buildProceduralSkillCandidate(input: {
  patternId: string;
  membershipVersion: string;
  sequenceHash: string;
  namespaceId: string;
  policyKeys: readonly string[];
  capabilityType: CapabilityType;
  episodeFamilyId?: string;
  isClosed: true;
  isMaximal: boolean;
  occurrences: readonly PolicySequencePatternOccurrenceV1[];
}): ProceduralSkillCandidateV1 {
  if (!hasMultipleDistinctPolicies(input.policyKeys)) {
    throw new Error("SkillCandidate requires at least two distinct Policy keys");
  }
  const supportEpisodeIds = episodeIds(input.occurrences, "support");
  if (supportEpisodeIds.length < POLICY_SEQUENCE_OBSERVED_SUPPORT) {
    throw new Error(`SkillCandidate requires ${POLICY_SEQUENCE_OBSERVED_SUPPORT} support Episodes`);
  }
  const evidence = [...input.occurrences]
    .sort(compareOccurrence)
    .map((occurrence): ProceduralSkillCandidateEvidenceV1 => ({
      occurrenceId: occurrence.id,
      projectionId: occurrence.projectionId,
      episodeId: occurrence.episodeId,
      pathId: occurrence.pathId,
      sessionId: occurrence.sessionId,
      startNodeIndex: occurrence.startNodeIndex,
      endNodeIndex: occurrence.endNodeIndex,
      matchedNodeIndexes: [...occurrence.matchedNodeIndexes],
      pathNodeIndexes: [...occurrence.pathNodeIndexes],
      policyVersionIds: [...occurrence.policyVersionIds],
      spanOccurrenceIds: [...occurrence.spanOccurrenceIds],
      spanIds: [...occurrence.spanIds],
      pathSpanOccurrenceIds: [...occurrence.pathSpanOccurrenceIds],
      pathSpanIds: [...occurrence.pathSpanIds],
      preStateId: occurrence.preStateId,
      postStateId: occurrence.postStateId,
      evidenceRole: occurrence.evidenceRole,
      discoverySources: [...occurrence.discoverySources],
      ...(occurrence.episodeAffinity ? {
        episodeAffinity: { ...occurrence.episodeAffinity }
      } : {}),
      ...(occurrence.terminalReward === undefined ? {} : {
        terminalReward: occurrence.terminalReward
      }),
      cost: { ...occurrence.cost }
    }));
  const supportOccurrences = input.occurrences.filter((item) => item.evidenceRole === "support");
  const episodeFamilyIds = unique(supportOccurrences
    .filter((item) => item.discoverySources.includes("episode_similarity"))
    .map((item) => item.episodeFamilyId)
    .filter((value): value is string => Boolean(value)));
  const episodeGroundedTask = supportOccurrences.length > 0 &&
    supportOccurrences.every((item) =>
      item.discoverySources.includes("episode_similarity") && Boolean(item.episodeFamilyId)
    ) && episodeFamilyIds.length === 1;
  const capabilityType: CapabilityType = input.capabilityType === "task_skill" || episodeGroundedTask
    ? "task_skill"
    : "sub_skill";
  const episodeFamilyId = input.episodeFamilyId ??
    (episodeGroundedTask ? episodeFamilyIds[0] : undefined);
  const total = supportOccurrences.reduce(
    (sum, occurrence) => addCost(sum, occurrence.cost),
    emptyCost()
  );
  const evidenceHash = stableHash(evidence);
  const lifecycleStatus: ProceduralSkillCandidateLifecycleStatus =
    supportEpisodeIds.length >= POLICY_SEQUENCE_READY_SUPPORT ? "ready" : "observed";
  const idHash = stableHash({
    patternId: input.patternId,
    membershipVersion: input.membershipVersion,
    inductionVersion: PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION,
    evidenceHash
  });
  return {
    id: `procedural_skill_candidate_${idHash.slice(0, 20)}`,
    candidateKey: `policy-sequence:${input.sequenceHash}`,
    schemaVersion: PROCEDURAL_SKILL_CANDIDATE_SCHEMA_VERSION,
    inductionVersion: PROCEDURAL_SKILL_CANDIDATE_INDUCTION_VERSION,
    patternId: input.patternId,
    patternMembershipVersion: input.membershipVersion,
    sequenceHash: input.sequenceHash,
    namespaceId: input.namespaceId,
    policyKeys: [...input.policyKeys],
    capabilityType,
    discoverySources: unique(input.occurrences.flatMap((item) => item.discoverySources)) as
      CapabilityDiscoverySource[],
    ...(episodeFamilyId ? { episodeFamilyId } : {}),
    lifecycleStatus,
    isClosed: true,
    isMaximal: input.isMaximal,
    supportEpisodeIds,
    counterexampleEpisodeIds: episodeIds(input.occurrences, "counterexample"),
    uncertainEpisodeIds: episodeIds(input.occurrences, "uncertain"),
    sourceProjectionIds: unique(input.occurrences.map((item) => item.projectionId)),
    sourcePathIds: unique(input.occurrences.map((item) => item.pathId)),
    sourceSpanOccurrenceIds: unique(input.occurrences.flatMap((item) =>
      item.pathSpanOccurrenceIds
    )),
    sourceSpanIds: unique(input.occurrences.flatMap((item) => item.pathSpanIds)),
    aggregateSupportCost: {
      total,
      meanPerOccurrence: divideCost(total, supportOccurrences.length)
    },
    evidence,
    evidenceHash,
    provenance: {
      miningAlgorithmVersion: POLICY_SEQUENCE_MINING_ALGORITHM_VERSION,
      observedSupportThreshold: POLICY_SEQUENCE_OBSERVED_SUPPORT,
      readySupportThreshold: POLICY_SEQUENCE_READY_SUPPORT,
      executable: false
    }
  };
}

function sequenceEvidenceRole(
  roles: readonly ("support" | "counterexample")[],
  terminalReward: number | undefined
): PolicySequenceEvidenceRole {
  if (roles.includes("counterexample") ||
      (terminalReward !== undefined && terminalReward <= 0)) {
    return "counterexample";
  }
  if (terminalReward !== undefined && terminalReward > 0 &&
      roles.every((role) => role === "support")) {
    return "support";
  }
  return "uncertain";
}

function episodeIds(
  occurrences: readonly PolicySequencePatternOccurrenceV1[],
  role: PolicySequenceEvidenceRole
): string[] {
  return unique(occurrences
    .filter((occurrence) => occurrence.evidenceRole === role)
    .map((occurrence) => occurrence.episodeId));
}

function compareOccurrence(
  left: PolicySequencePatternOccurrenceV1,
  right: PolicySequencePatternOccurrenceV1
): number {
  return left.episodeId.localeCompare(right.episodeId) ||
    left.startNodeIndex - right.startNodeIndex ||
    left.id.localeCompare(right.id);
}

function strongerAffinity(
  left: EpisodeAffinityEvidenceV1 | undefined,
  right: EpisodeAffinityEvidenceV1 | undefined
): EpisodeAffinityEvidenceV1 | undefined {
  if (!left) return right ? { ...right } : undefined;
  if (!right) return { ...left };
  return right.combinedSimilarity > left.combinedSimilarity ? { ...right } : { ...left };
}

function sequenceMapKey(
  policyKeys: readonly string[],
  capabilityType: CapabilityType,
  episodeFamilyId?: string
): string {
  return JSON.stringify([capabilityType, episodeFamilyId ?? null, policyKeys]);
}

function setEquals(left: ReadonlySet<string>, right: ReadonlySet<string>): boolean {
  if (left.size !== right.size) return false;
  for (const value of left) if (!right.has(value)) return false;
  return true;
}

function emptyCost(): PolicySequenceCostV1 {
  return {
    steps: 0,
    toolCalls: 0,
    retryCount: 0,
    recoveryCount: 0,
    errorCount: 0
  };
}

function addCost(left: PolicySequenceCostV1, right: PolicySequenceCostV1): PolicySequenceCostV1 {
  return {
    steps: left.steps + right.steps,
    toolCalls: left.toolCalls + right.toolCalls,
    retryCount: left.retryCount + right.retryCount,
    recoveryCount: left.recoveryCount + right.recoveryCount,
    errorCount: left.errorCount + right.errorCount,
    ...optionalSum("tokens", left, right),
    ...optionalSum("latencyMs", left, right)
  };
}

function divideCost(cost: PolicySequenceCostV1, divisor: number): PolicySequenceCostV1 {
  const safeDivisor = Math.max(1, divisor);
  return {
    steps: round(cost.steps / safeDivisor),
    toolCalls: round(cost.toolCalls / safeDivisor),
    retryCount: round(cost.retryCount / safeDivisor),
    recoveryCount: round(cost.recoveryCount / safeDivisor),
    errorCount: round(cost.errorCount / safeDivisor),
    ...(cost.tokens === undefined ? {} : { tokens: round(cost.tokens / safeDivisor) }),
    ...(cost.latencyMs === undefined ? {} : { latencyMs: round(cost.latencyMs / safeDivisor) })
  };
}

function optionalSum(
  key: "tokens" | "latencyMs",
  left: PolicySequenceCostV1,
  right: PolicySequenceCostV1
): Partial<PolicySequenceCostV1> {
  if (left[key] === undefined && right[key] === undefined) return {};
  return { [key]: (left[key] ?? 0) + (right[key] ?? 0) };
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
