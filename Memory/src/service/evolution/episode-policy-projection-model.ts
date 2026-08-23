import type { ProceduralSpanClusterStatus } from "../../storage/procedural-span-cluster-repository.js";
import type { ProceduralSpanEvidenceRole } from "../../storage/procedural-span-cluster-repository.js";
import type { ProceduralSpanTermination } from "./procedural-path-model.js";
import { stableHash } from "../../utils/id.js";

export const EPISODE_POLICY_PROJECTION_SCHEMA_VERSION = "episode-policy-projection.v1" as const;
export const EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION =
  "episode-policy-projection-builder.v1" as const;

export type EpisodePolicyProjectionNodeKind = "policy" | "unmapped";
export type EpisodePolicyUnmappedReason =
  | "no_cluster_assignment"
  | "cluster_forming"
  | "cluster_ready_policy_pending"
  | "cluster_stale"
  | "active_policy_occurrence_missing";

export interface EpisodePolicyProjectionNodeSourceV1 {
  occurrenceId: string;
  spanId: string;
  spanIndex: number;
  localGoal: string;
  entryCondition: string;
  exitCondition: string;
  terminationStatus: ProceduralSpanTermination;
  preStateId: string;
  postStateId: string;
  rawTurnIds: string[];
  stepIds: string[];
}

export interface EpisodePolicyProjectionPolicyAssignmentV1 {
  kind: "policy";
  policyVersionId: string;
  policyKey: string;
  clusterId: string;
  clusterMembershipVersion: string;
  evidenceRole: ProceduralSpanEvidenceRole;
  matchConfidence: number;
}

export interface EpisodePolicyProjectionUnmappedAssignmentV1 {
  kind: "unmapped";
  reason: EpisodePolicyUnmappedReason;
  clusterId?: string;
  clusterMembershipVersion?: string;
  clusterStatus?: ProceduralSpanClusterStatus;
}

export type EpisodePolicyProjectionAssignmentV1 =
  | EpisodePolicyProjectionPolicyAssignmentV1
  | EpisodePolicyProjectionUnmappedAssignmentV1;

export type EpisodePolicyProjectionNodeV1 = EpisodePolicyProjectionNodeSourceV1 & {
  nodeIndex: number;
  assignment: EpisodePolicyProjectionAssignmentV1;
};

export interface EpisodePolicyProjectionCoverageV1 {
  totalSpanCount: number;
  mappedSpanCount: number;
  unmappedSpanCount: number;
  mappingRatio: number;
}

export interface EpisodePolicyProjectionV1 {
  id: string;
  schemaVersion: typeof EPISODE_POLICY_PROJECTION_SCHEMA_VERSION;
  algorithmVersion: typeof EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION;
  episodeId: string;
  pathId: string;
  pathHash: string;
  assignmentSnapshotHash: string;
  projectionHash: string;
  nodes: EpisodePolicyProjectionNodeV1[];
  coverage: EpisodePolicyProjectionCoverageV1;
}

export interface BuildEpisodePolicyProjectionInput {
  episodeId: string;
  pathId: string;
  pathHash: string;
  nodes: Array<EpisodePolicyProjectionNodeSourceV1 & {
    assignment: EpisodePolicyProjectionAssignmentV1;
  }>;
}

export function buildEpisodePolicyProjection(
  input: BuildEpisodePolicyProjectionInput
): EpisodePolicyProjectionV1 {
  const ordered = [...input.nodes]
    .sort((left, right) => left.spanIndex - right.spanIndex ||
      left.occurrenceId.localeCompare(right.occurrenceId))
    .map((node, nodeIndex): EpisodePolicyProjectionNodeV1 => ({
      ...node,
      rawTurnIds: [...node.rawTurnIds],
      stepIds: [...node.stepIds],
      nodeIndex,
      assignment: { ...node.assignment }
    }));
  validateNodes(ordered);
  const coverage = projectionCoverage(ordered);
  const assignmentSnapshotHash = stableHash(ordered.map(assignmentSnapshotBasis));
  const projectionHash = stableHash({
    schemaVersion: EPISODE_POLICY_PROJECTION_SCHEMA_VERSION,
    algorithmVersion: EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION,
    episodeId: input.episodeId,
    pathId: input.pathId,
    pathHash: input.pathHash,
    assignmentSnapshotHash,
    nodes: ordered,
    coverage
  });
  return {
    id: `episode_policy_projection_${projectionHash.slice(0, 20)}`,
    schemaVersion: EPISODE_POLICY_PROJECTION_SCHEMA_VERSION,
    algorithmVersion: EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION,
    episodeId: input.episodeId,
    pathId: input.pathId,
    pathHash: input.pathHash,
    assignmentSnapshotHash,
    projectionHash,
    nodes: ordered,
    coverage
  };
}

export function validateEpisodePolicyProjection(projection: EpisodePolicyProjectionV1): void {
  if (projection.schemaVersion !== EPISODE_POLICY_PROJECTION_SCHEMA_VERSION ||
      projection.algorithmVersion !== EPISODE_POLICY_PROJECTION_ALGORITHM_VERSION) {
    throw new Error(`unsupported Episode Policy Projection schema: ${projection.id}`);
  }
  validateNodes(projection.nodes);
  const rebuilt = buildEpisodePolicyProjection({
    episodeId: projection.episodeId,
    pathId: projection.pathId,
    pathHash: projection.pathHash,
    nodes: projection.nodes.map(({ nodeIndex: _nodeIndex, ...node }) => node)
  });
  if (rebuilt.id !== projection.id || rebuilt.projectionHash !== projection.projectionHash ||
      rebuilt.assignmentSnapshotHash !== projection.assignmentSnapshotHash ||
      JSON.stringify(rebuilt.coverage) !== JSON.stringify(projection.coverage)) {
    throw new Error(`Episode Policy Projection integrity check failed: ${projection.id}`);
  }
}

function validateNodes(nodes: readonly EpisodePolicyProjectionNodeV1[]): void {
  const occurrenceIds = new Set<string>();
  const spanIndexes = new Set<number>();
  for (const [index, node] of nodes.entries()) {
    if (node.nodeIndex !== index || node.spanIndex !== index) {
      throw new Error(`Episode Policy Projection node order is invalid: ${node.occurrenceId}`);
    }
    if (!node.occurrenceId || !node.spanId || !node.preStateId || !node.postStateId) {
      throw new Error(`Episode Policy Projection node source is incomplete: ${node.occurrenceId}`);
    }
    if (occurrenceIds.has(node.occurrenceId) || spanIndexes.has(node.spanIndex)) {
      throw new Error(`Episode Policy Projection contains duplicate Span occurrence: ${node.occurrenceId}`);
    }
    occurrenceIds.add(node.occurrenceId);
    spanIndexes.add(node.spanIndex);
    if (node.assignment.kind === "policy" &&
        (!Number.isFinite(node.assignment.matchConfidence) ||
         node.assignment.matchConfidence < 0 || node.assignment.matchConfidence > 1)) {
      throw new Error(`Episode Policy Projection match confidence is invalid: ${node.occurrenceId}`);
    }
  }
}

function projectionCoverage(
  nodes: readonly EpisodePolicyProjectionNodeV1[]
): EpisodePolicyProjectionCoverageV1 {
  const mappedSpanCount = nodes.filter((node) => node.assignment.kind === "policy").length;
  const totalSpanCount = nodes.length;
  return {
    totalSpanCount,
    mappedSpanCount,
    unmappedSpanCount: totalSpanCount - mappedSpanCount,
    mappingRatio: totalSpanCount === 0 ? 0 : round(mappedSpanCount / totalSpanCount)
  };
}

function assignmentSnapshotBasis(node: EpisodePolicyProjectionNodeV1): Record<string, unknown> {
  return {
    occurrenceId: node.occurrenceId,
    spanIndex: node.spanIndex,
    assignment: node.assignment
  };
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}
