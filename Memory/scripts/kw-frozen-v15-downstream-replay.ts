import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import Database from "better-sqlite3";
import {
  EXECUTION_STEP_LITE_SCHEMA_VERSION,
  PROCEDURAL_WINDOW_MINING_VERSION,
  V15_FINE_MATCH_CONFIGS,
  V15_FINE_MEDOID_SWITCH_MARGIN,
  extractAlignedCommonCore,
  selectMaximalWindowClusters,
  type EmbeddedTrajectoryWindowV1,
  type ExecutionStepLiteOutcome,
  type ExecutionStepLiteV1,
  type ProceduralEvidenceRole,
  type TrajectoryWindowClusterMemberV1,
  type TrajectoryWindowClusterV1,
  type TrajectoryWindowOccurrenceV1
} from "../src/service/evolution/procedural-window-model.js";
import {
  bandedMonotonicMatch,
  selfBandedMonotonicMatch,
  type BandedMonotonicMatchConfig,
  type BandedMonotonicMatchResultV1
} from "../src/service/evolution/trajectory-window-alignment.js";
import { stableHash } from "../src/utils/id.js";

const V15_EXPERIMENT_VERSION = "multi-scale-window-policy-experiment.v5";
const V15_PATH_VERSION = "episode-procedural-reconstruction.v8";
const WINDOW_SPECS = [
  { length: 5, stride: 2 },
  { length: 10, stride: 5 }
] as const;

interface Args {
  sourceDbPath: string;
  v15ResultPath: string;
  outputReportPath: string;
}

interface PathRow {
  id: string;
  episode_id: string;
  user_id: string;
  terminal_reward: number | null;
  payload_json: string;
  opened_at: string;
}

interface StepEmbeddingRow {
  step_id: string;
  vector_json: string;
}

interface V15Family {
  id: string;
  scale: number;
  medoidOccurrenceId: string;
  memberOccurrenceIds: string[];
}

interface V15ClusterMember {
  occurrence: {
    id: string;
    episodeId: string;
    scale: number;
    startStepIndex: number;
    endStepIndex: number;
  };
  similarityToCenter: number;
  alignmentToMedoid: BandedMonotonicMatchResultV1;
}

interface V15Cluster {
  id: string;
  scale: number;
  medoidOccurrenceId: string;
  supportEpisodeIds: string[];
  members: V15ClusterMember[];
}

interface V15Artifact {
  result: {
    families: V15Family[];
    clusters: V15Cluster[];
    candidateClusterIds: string[];
  };
}

interface InternalCluster {
  id: string;
  familyId: string;
  scale: number;
  medoid: EmbeddedTrajectoryWindowV1;
  members: EmbeddedTrajectoryWindowV1[];
  medoidUpdateCount: number;
}

interface MedoidCandidate {
  medoid: EmbeddedTrajectoryWindowV1;
  centrality: number;
  alignments: Map<string, BandedMonotonicMatchResultV1>;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  for (const input of [args.sourceDbPath, args.v15ResultPath]) {
    if (!existsSync(input)) throw new Error(`Input not found: ${input}`);
  }
  if (existsSync(args.outputReportPath)) {
    throw new Error(`Refusing to overwrite output report: ${args.outputReportPath}`);
  }

  const artifact = JSON.parse(readFileSync(args.v15ResultPath, "utf8")) as V15Artifact;
  const db = new Database(args.sourceDbPath, { readonly: true, fileMustExist: true });
  try {
    const loaded = loadFrozenWindows(db);
    const windowById = new Map(loaded.windows.map((window) => [window.occurrence.id, window]));
    const familyMemberIds = unique(artifact.result.families.flatMap((family) =>
      family.memberOccurrenceIds));
    const missing = familyMemberIds.filter((id) => !windowById.has(id));
    const extra = loaded.windows.map((window) => window.occurrence.id)
      .filter((id) => !new Set(familyMemberIds).has(id));
    if (missing.length > 0 || extra.length > 0) {
      throw new Error(
        `Frozen Window identity mismatch: missing=${missing.join(",")}; extra=${extra.join(",")}`
      );
    }

    const familyRuns = artifact.result.families.map((family) => {
      const windows = family.memberOccurrenceIds.map((id) => windowById.get(id)!);
      return {
        family,
        clusters: clusterFamilyWithProductionFine(windows, family, configForScale(family.scale))
      };
    });
    const rawClusters = familyRuns.flatMap((run) => run.clusters);
    const clusters = dedupeEquivalentClusters(rawClusters);
    const supported = clusters.filter((cluster) => cluster.supportEpisodeIds.length >= 2);
    const qualified = supported.flatMap((cluster) => {
      const core = extractAlignedCommonCore(cluster, {
        minSupportEpisodes: 2,
        minCoreSteps: configForScale(cluster.scale).minMatchedSteps
      });
      return core ? [{ cluster, core }] : [];
    });
    const maximalById = new Map(selectMaximalWindowClusters(
      qualified.map((item) => item.cluster)
    ).map((cluster) => [cluster.id, cluster]));
    const qualifiedWithSuppression = qualified.map((item) => ({
      ...item,
      cluster: maximalById.get(item.cluster.id) ?? item.cluster
    }));
    const maximal = qualifiedWithSuppression.filter((item) =>
      !item.cluster.suppressedByClusterId);

    const oldCandidateIdSet = new Set(artifact.result.candidateClusterIds);
    const oldCandidates = artifact.result.clusters.filter((cluster) =>
      oldCandidateIdSet.has(cluster.id));
    const exactOldDomains = oldCandidates.map((cluster) =>
      currentDomainFromExactV15Cluster(cluster, windowById));
    const exactOldQualified = exactOldDomains.flatMap((cluster) => {
      const core = extractAlignedCommonCore(cluster, {
        minSupportEpisodes: 2,
        minCoreSteps: configForScale(cluster.scale).minMatchedSteps
      });
      return core ? [{ cluster, core }] : [];
    });
    const exactOldMaximal = selectMaximalWindowClusters(
      exactOldQualified.map((item) => item.cluster)
    ).filter((cluster) => !cluster.suppressedByClusterId);
    const oldBySignature = new Map(oldCandidates.map((cluster) => [
      evidenceSignatureFromV15(cluster),
      cluster
    ]));
    const supportedBySignature = new Map(supported.map((cluster) => [
      evidenceSignature(cluster),
      cluster
    ]));
    const qualifiedBySignature = new Map(qualifiedWithSuppression.map((item) => [
      evidenceSignature(item.cluster),
      item
    ]));
    const maximalBySignature = new Map(maximal.map((item) => [
      evidenceSignature(item.cluster),
      item
    ]));

    const oldCandidateAudit = [...oldBySignature.entries()].map(([signature, old]) => ({
      ...auditOldCandidate(old, artifact.result.families, windowById, rawClusters, clusters),
      oldClusterId: old.id,
      scale: old.scale,
      signature,
      evidence: evidenceCoordinatesFromV15(old),
      reproducedSupported: supportedBySignature.has(signature),
      reproducedQualified: qualifiedBySignature.has(signature),
      reproducedMaximal: maximalBySignature.has(signature),
      currentClusterId: supportedBySignature.get(signature)?.id ?? null,
      suppressedByClusterId:
        qualifiedBySignature.get(signature)?.cluster.suppressedByClusterId ?? null,
      coreStepCount: qualifiedBySignature.get(signature)?.core.steps.length ?? 0
    }));

    const report = {
      schemaVersion: "kw-frozen-v15-coarse-current-downstream.v1",
      createdAt: new Date().toISOString(),
      sourceDatabase: args.sourceDbPath,
      v15Result: args.v15ResultPath,
      isolation: {
        remoteCalls: 0,
        stepSemantics: "frozen from v15 SQLite",
        stepEmbeddings: "frozen 1024-D vectors from v15 SQLite",
        coarseMemberships: "frozen from v15 result.families[].memberOccurrenceIds",
        windowEmbeddings: "not used",
        downstream: [
          "current production fine admission",
          "current production coherent real-medoid selection",
          "current production per-Episode overlap suppression",
          "current production aligned-common-core qualification",
          "current production maximal suppression"
        ]
      },
      parameters: {
        windowSpecs: WINDOW_SPECS,
        fineMatchConfigs: V15_FINE_MATCH_CONFIGS,
        medoidSwitchMargin: V15_FINE_MEDOID_SWITCH_MARGIN,
        minSupportEpisodes: 2
      },
      summary: {
        episodes: loaded.episodeCount,
        steps: loaded.stepCount,
        windows: loaded.windows.length,
        span5Windows: loaded.windows.filter((window) => window.occurrence.scale === 5).length,
        span10Windows: loaded.windows.filter((window) => window.occurrence.scale === 10).length,
        frozenCoarseFamilies: artifact.result.families.length,
        rawFamilyFineClusters: rawClusters.length,
        deduplicatedFineClusters: clusters.length,
        supportedClusters: supported.length,
        supportedSpan5Clusters: supported.filter((cluster) => cluster.scale === 5).length,
        supportedSpan10Clusters: supported.filter((cluster) => cluster.scale === 10).length,
        commonCoreQualifiedClusters: qualified.length,
        maximalQualifiedClusters: maximal.length,
        oldCandidateCount: oldCandidates.length,
        exactV15CandidatesCurrentCoreQualified: exactOldQualified.length,
        exactV15CandidatesCurrentMaximal: exactOldMaximal.length,
        oldCandidatesReproducedSupported: oldCandidateAudit.filter((item) =>
          item.reproducedSupported).length,
        oldCandidatesReproducedQualified: oldCandidateAudit.filter((item) =>
          item.reproducedQualified).length,
        oldCandidatesReproducedMaximal: oldCandidateAudit.filter((item) =>
          item.reproducedMaximal).length
      },
      oldCandidateAudit,
      exactV15FineOutputCurrentDownstream: exactOldQualified.map((item) => ({
        ...clusterSummary(item.cluster),
        commonCoreStepCount: item.core.steps.length,
        commonCoreAnchorOffsets: item.core.anchorOffsets,
        projectedSupportEpisodeIds: item.core.supportEpisodeIds,
        maximal: exactOldMaximal.some((cluster) => cluster.id === item.cluster.id)
      })),
      supportedClusters: supported.map(clusterSummary),
      qualifiedClusters: qualifiedWithSuppression.map((item) => ({
        ...clusterSummary(item.cluster),
        commonCoreStepCount: item.core.steps.length,
        commonCoreAnchorOffsets: item.core.anchorOffsets,
        projectedSupportEpisodeIds: item.core.supportEpisodeIds
      }))
    };
    mkdirSync(dirname(args.outputReportPath), { recursive: true });
    writeFileSync(args.outputReportPath, `${JSON.stringify(report, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600
    });
    process.stdout.write(`${JSON.stringify({
      outputReport: args.outputReportPath,
      ...report.summary,
      oldCandidateAudit
    }, null, 2)}\n`);
  } finally {
    db.close();
  }
}

function currentDomainFromExactV15Cluster(
  old: V15Cluster,
  windowById: Map<string, EmbeddedTrajectoryWindowV1>
): TrajectoryWindowClusterV1 {
  const members: TrajectoryWindowClusterMemberV1[] = old.members.map((member) => {
    const frozen = windowById.get(member.occurrence.id);
    if (!frozen) throw new Error(`Old candidate Window missing: ${member.occurrence.id}`);
    return {
      occurrence: frozen.occurrence,
      similarityToMedoid: member.alignmentToMedoid.score,
      alignmentToMedoid: member.alignmentToMedoid
    };
  });
  const similarities = members.map((member) => member.similarityToMedoid);
  return {
    id: old.id,
    familyId: old.id,
    scale: old.scale,
    medoidOccurrenceId: old.medoidOccurrenceId,
    episodeIds: distinctEpisodes(members),
    supportEpisodeIds: distinctEpisodes(members.filter((member) =>
      member.occurrence.evidenceRole === "support")),
    counterexampleEpisodeIds: distinctEpisodes(members.filter((member) =>
      member.occurrence.evidenceRole === "counterexample")),
    unknownEpisodeIds: distinctEpisodes(members.filter((member) =>
      member.occurrence.evidenceRole === "unknown")),
    occurrenceCount: members.length,
    averageSimilarity: average(similarities),
    minimumSimilarity: Math.min(...similarities),
    medoidCentrality: average(similarities),
    medoidUpdateCount: 0,
    members
  };
}

function loadFrozenWindows(db: Database.Database): {
  windows: EmbeddedTrajectoryWindowV1[];
  episodeCount: number;
  stepCount: number;
} {
  const paths = db.prepare(
    `SELECT p.id, p.episode_id, p.user_id, p.terminal_reward, p.payload_json, e.opened_at
     FROM episode_procedural_paths p
     JOIN episodes e ON e.id = p.episode_id
     WHERE p.status = 'active' AND p.reconstruction_algorithm_version = ?
     ORDER BY p.episode_id ASC, p.id ASC`
  ).all(V15_PATH_VERSION) as PathRow[];
  const vectorRows = db.prepare(
    `SELECT o.step_id, e.vector_json
     FROM procedural_step_occurrences o
     JOIN procedural_step_embeddings e ON e.occurrence_id = o.id
     JOIN episode_procedural_paths p ON p.id = o.path_id
     WHERE p.status = 'active' AND p.reconstruction_algorithm_version = ?`
  ).all(V15_PATH_VERSION) as StepEmbeddingRow[];
  const vectorByStepId = new Map(vectorRows.map((row) => [
    row.step_id,
    unitVector(numberArray(row.vector_json))
  ]));
  const windows: EmbeddedTrajectoryWindowV1[] = [];
  let stepCount = 0;
  for (const pathRow of paths) {
    const payload = record(JSON.parse(pathRow.payload_json));
    const oldSteps = array(payload.steps).map(record);
    const steps = oldSteps.map((oldStep, index) => adaptStep(oldStep, pathRow, index));
    stepCount += steps.length;
    for (const spec of WINDOW_SPECS) {
      if (steps.length < spec.length) continue;
      const starts: number[] = [];
      for (let start = 0; start + spec.length <= steps.length; start += spec.stride) {
        starts.push(start);
      }
      const tailStart = steps.length - spec.length;
      if (starts.at(-1) !== tailStart) starts.push(tailStart);
      for (const start of starts) {
        const windowSteps = steps.slice(start, start + spec.length);
        const semanticText = windowSteps.map((step, index) =>
          `${index + 1}. ${step.intent.trim()}`).join("\n");
        const startStepIndex = windowSteps[0]!.stepIndex;
        const endStepIndex = windowSteps.at(-1)!.stepIndex;
        const oldId = `trajectory_window_${stableHash({
          version: V15_EXPERIMENT_VERSION,
          episodeId: pathRow.episode_id,
          pathId: pathRow.id,
          scale: spec.length,
          startStepIndex,
          endStepIndex,
          semanticText
        }).slice(0, 24)}`;
        const occurrence: TrajectoryWindowOccurrenceV1 = {
          id: oldId,
          schemaVersion: "trajectory-window-occurrence.v1",
          episodeId: pathRow.episode_id,
          pathId: pathRow.id,
          userId: pathRow.user_id,
          ...(pathRow.terminal_reward === null
            ? {}
            : { terminalReward: pathRow.terminal_reward }),
          evidenceRole: roleForReward(pathRow.terminal_reward),
          scale: spec.length,
          stride: spec.stride,
          startStepIndex,
          endStepIndex,
          semanticText,
          steps: windowSteps
        };
        windows.push({
          occurrence,
          coarseVector: [],
          stepVectors: windowSteps.map((step) => {
            const vector = vectorByStepId.get(step.id);
            if (!vector) throw new Error(`Frozen Step vector missing: ${step.id}`);
            return vector;
          })
        });
      }
    }
  }
  return { windows, episodeCount: paths.length, stepCount };
}

function adaptStep(oldStep: Record<string, unknown>, path: PathRow, index: number): ExecutionStepLiteV1 {
  const action = record(oldStep.action);
  const outcome = record(oldStep.outcome);
  const provenance = record(oldStep.provenance);
  const kind = action.kind === "response_generation" ? "response_generation" : "tool_action";
  const status = outcomeStatus(outcome.status);
  const evidenceRefs = stringArray(outcome.evidenceRefs).length > 0
    ? stringArray(outcome.evidenceRefs)
    : stringArray(action.eventRefs);
  return {
    id: string(oldStep.id) ?? `missing_step_${path.episode_id}_${index}`,
    schemaVersion: EXECUTION_STEP_LITE_SCHEMA_VERSION,
    episodeId: path.episode_id,
    rawTurnId: string(oldStep.rawTurnId) ?? `missing_raw_turn_${path.episode_id}`,
    turnIndex: integer(oldStep.turnIndex, 0),
    stepIndex: integer(oldStep.stepIndex, index),
    kind,
    ...(string(action.toolName) ? { toolName: string(action.toolName)! } : {}),
    ...(Number.isInteger(action.toolCallIndex)
      ? { toolCallIndex: Number(action.toolCallIndex) }
      : {}),
    intent: string(action.intent) ?? "unknown intent",
    summary: string(action.summary) ?? "unknown summary",
    outcome: status,
    ...(string(outcome.errorCode) ? { errorCode: string(outcome.errorCode)! } : {}),
    ...(string(oldStep.retryOfStepId) ? { retryOfStepId: string(oldStep.retryOfStepId)! } : {}),
    ...(string(oldStep.recoveryFromStepId)
      ? { recoveryFromStepId: string(oldStep.recoveryFromStepId)! }
      : {}),
    evidenceRefs,
    provenance: {
      algorithmVersion: string(provenance.algorithmVersion) ?? V15_PATH_VERSION,
      ...(string(provenance.model) ? { model: string(provenance.model)! } : {}),
      sourceSnapshotHash: string(provenance.sourceSnapshotHash) ?? "frozen-v15"
    }
  };
}

function clusterFamilyWithProductionFine(
  windows: EmbeddedTrajectoryWindowV1[],
  family: V15Family,
  config: BandedMonotonicMatchConfig
): TrajectoryWindowClusterV1[] {
  const ordered = [...windows].sort((left, right) =>
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex ||
    left.occurrence.id.localeCompare(right.occurrence.id));
  const clusters: InternalCluster[] = [];
  for (const window of ordered) {
    let joined = false;
    for (const cluster of clusters) {
      const alignment = window.occurrence.id === cluster.medoid.occurrence.id
        ? selfBandedMonotonicMatch(window.stepVectors, config)
        : bandedMonotonicMatch(window.stepVectors, cluster.medoid.stepVectors, config);
      if (!alignment.admitted) continue;
      const selected = selectProductionClusterMembers(
        uniqueWindows([...cluster.members, window]),
        cluster.medoid.occurrence.id,
        config
      );
      if (!selected) continue;
      joined = true;
      const oldMedoidId = cluster.medoid.occurrence.id;
      cluster.medoid = selected.medoid;
      cluster.members = selected.members;
      if (oldMedoidId !== selected.medoid.occurrence.id) cluster.medoidUpdateCount += 1;
    }
    if (!joined) {
      clusters.push({
        id: `offline_prod_cluster_${stableHash({
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
    }
  }
  return clusters.map((cluster) => finalizeCluster(cluster, config));
}

function selectProductionClusterMembers(
  members: EmbeddedTrajectoryWindowV1[],
  previousMedoidOccurrenceId: string | undefined,
  config: BandedMonotonicMatchConfig
): { medoid: EmbeddedTrajectoryWindowV1; members: EmbeddedTrajectoryWindowV1[] } | undefined {
  const candidates = coherentMedoidCandidates(members, config);
  if (candidates.length === 0) return undefined;
  const previous = candidates.find((candidate) =>
    candidate.medoid.occurrence.id === previousMedoidOccurrenceId);
  const best = candidates[0]!;
  const chosen = previous && best.medoid.occurrence.id !== previous.medoid.occurrence.id &&
    best.centrality - previous.centrality <= V15_FINE_MEDOID_SWITCH_MARGIN
    ? previous
    : best;
  const nonOverlapping = selectNonOverlappingMembers(members, chosen.medoid, config);
  const finalCandidates = coherentMedoidCandidates(nonOverlapping, config);
  if (finalCandidates.length === 0) return undefined;
  const finalPrevious = finalCandidates.find((candidate) =>
    candidate.medoid.occurrence.id === chosen.medoid.occurrence.id);
  const final = finalPrevious ?? finalCandidates[0]!;
  return { medoid: final.medoid, members: nonOverlapping };
}

function coherentMedoidCandidates(
  members: EmbeddedTrajectoryWindowV1[],
  config: BandedMonotonicMatchConfig
): MedoidCandidate[] {
  const hasSupport = members.some((member) => member.occurrence.evidenceRole === "support");
  return members.flatMap((candidate): MedoidCandidate[] => {
    if (hasSupport && candidate.occurrence.evidenceRole !== "support") return [];
    const alignments = new Map<string, BandedMonotonicMatchResultV1>();
    for (const member of members) {
      const alignment = member.occurrence.id === candidate.occurrence.id
        ? selfBandedMonotonicMatch(member.stepVectors, config)
        : bandedMonotonicMatch(member.stepVectors, candidate.stepVectors, config);
      if (!alignment.admitted) return [];
      alignments.set(member.occurrence.id, alignment);
    }
    return [{
      medoid: candidate,
      centrality: average([...alignments.values()].map((alignment) => alignment.score)),
      alignments
    }];
  }).sort((left, right) => right.centrality - left.centrality ||
    left.medoid.occurrence.id.localeCompare(right.medoid.occurrence.id));
}

function selectNonOverlappingMembers(
  members: EmbeddedTrajectoryWindowV1[],
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
      scoreToMedoid(right, medoid, config) - scoreToMedoid(left, medoid, config) ||
      left.occurrence.startStepIndex - right.occurrence.startStepIndex ||
      left.occurrence.id.localeCompare(right.occurrence.id))) {
      if (kept.some((existing) => rangesOverlap(existing, member))) continue;
      kept.push(member);
    }
    selected.push(...kept);
  }
  return selected.sort((left, right) =>
    left.occurrence.episodeId.localeCompare(right.occurrence.episodeId) ||
    left.occurrence.startStepIndex - right.occurrence.startStepIndex ||
    left.occurrence.id.localeCompare(right.occurrence.id));
}

function finalizeCluster(
  cluster: InternalCluster,
  config: BandedMonotonicMatchConfig
): TrajectoryWindowClusterV1 {
  const selected = selectProductionClusterMembers(
    cluster.members,
    cluster.medoid.occurrence.id,
    config
  );
  if (!selected) throw new Error(`Fine cluster lost a coherent medoid: ${cluster.id}`);
  const members: TrajectoryWindowClusterMemberV1[] = selected.members.map((member) => {
    const alignmentToMedoid = member.occurrence.id === selected.medoid.occurrence.id
      ? selfBandedMonotonicMatch(member.stepVectors, config)
      : bandedMonotonicMatch(member.stepVectors, selected.medoid.stepVectors, config);
    return {
      occurrence: member.occurrence,
      similarityToMedoid: alignmentToMedoid.score,
      alignmentToMedoid
    };
  });
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
    medoidOccurrenceId: selected.medoid.occurrence.id,
    episodeIds,
    supportEpisodeIds,
    counterexampleEpisodeIds,
    unknownEpisodeIds,
    occurrenceCount: members.length,
    averageSimilarity: average(similarities),
    minimumSimilarity: Math.min(...similarities),
    medoidCentrality: average(similarities),
    medoidUpdateCount: cluster.medoidUpdateCount +
      (cluster.medoid.occurrence.id === selected.medoid.occurrence.id ? 0 : 1),
    members
  };
}

function dedupeEquivalentClusters(clusters: TrajectoryWindowClusterV1[]): TrajectoryWindowClusterV1[] {
  const byEvidence = new Map<string, TrajectoryWindowClusterV1>();
  for (const cluster of clusters) {
    const key = evidenceSignature(cluster);
    const existing = byEvidence.get(key);
    if (!existing || cluster.averageSimilarity > existing.averageSimilarity ||
        (cluster.averageSimilarity === existing.averageSimilarity &&
          cluster.id.localeCompare(existing.id) < 0)) {
      byEvidence.set(key, cluster);
    }
  }
  return [...byEvidence.values()].sort((left, right) =>
    left.scale - right.scale ||
    right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
    right.averageSimilarity - left.averageSimilarity ||
    left.id.localeCompare(right.id));
}

function clusterSummary(cluster: TrajectoryWindowClusterV1) {
  return {
    id: cluster.id,
    scale: cluster.scale,
    signature: evidenceSignature(cluster),
    evidence: cluster.members.map((member) => ({
      occurrenceId: member.occurrence.id,
      episodeId: member.occurrence.episodeId,
      startStepIndex: member.occurrence.startStepIndex,
      endStepIndex: member.occurrence.endStepIndex,
      evidenceRole: member.occurrence.evidenceRole,
      alignmentScore: member.alignmentToMedoid.score,
      averageMatchSimilarity: member.alignmentToMedoid.averageMatchSimilarity,
      coverage: member.alignmentToMedoid.coverage,
      matchedSteps: member.alignmentToMedoid.matchedSteps
    })),
    supportEpisodeIds: cluster.supportEpisodeIds,
    averageSimilarity: cluster.averageSimilarity,
    minimumSimilarity: cluster.minimumSimilarity,
    medoidOccurrenceId: cluster.medoidOccurrenceId,
    suppressedByClusterId: cluster.suppressedByClusterId ?? null
  };
}

function evidenceSignature(cluster: TrajectoryWindowClusterV1): string {
  return `${cluster.scale}|${cluster.members.map((member) => member.occurrence.id).sort().join(",")}`;
}

function evidenceSignatureFromV15(cluster: V15Cluster): string {
  return `${cluster.scale}|${cluster.members.map((member) => member.occurrence.id).sort().join(",")}`;
}

function evidenceCoordinatesFromV15(cluster: V15Cluster) {
  return cluster.members.map((member) => ({
    occurrenceId: member.occurrence.id,
    episodeId: member.occurrence.episodeId,
    startStepIndex: member.occurrence.startStepIndex,
    endStepIndex: member.occurrence.endStepIndex
  }));
}

function auditOldCandidate(
  old: V15Cluster,
  families: V15Family[],
  windowById: Map<string, EmbeddedTrajectoryWindowV1>,
  rawClusters: TrajectoryWindowClusterV1[],
  clusters: TrajectoryWindowClusterV1[]
) {
  const memberIds = old.members.map((member) => member.occurrence.id);
  const memberIdSet = new Set(memberIds);
  const oldMedoid = windowById.get(old.medoidOccurrenceId);
  const other = old.members.map((member) => windowById.get(member.occurrence.id))
    .find((window) => window && window.occurrence.id !== old.medoidOccurrenceId);
  const directFine = oldMedoid && other
    ? bandedMonotonicMatch(other.stepVectors, oldMedoid.stepVectors, configForScale(old.scale))
    : undefined;
  const sharedCoarseFamilyIds = families.filter((family) =>
    memberIds.every((id) => family.memberOccurrenceIds.includes(id))).map((family) => family.id);
  const rawContaining = rawClusters.filter((cluster) => cluster.members.some((member) =>
    memberIdSet.has(member.occurrence.id)));
  const deduplicatedContaining = clusters.filter((cluster) => cluster.members.some((member) =>
    memberIdSet.has(member.occurrence.id)));
  return {
    sharedCoarseFamilyIds,
    directFineOnFrozenSqliteStepVectors: directFine ? {
      admitted: directFine.admitted,
      score: directFine.score,
      averageMatchSimilarity: directFine.averageMatchSimilarity,
      coverage: directFine.coverage,
      matchedSteps: directFine.matchedSteps,
      rejectionReasons: directFine.rejectionReasons
    } : null,
    rawFineClustersContainingEitherMember: rawContaining.map((cluster) => ({
      id: cluster.id,
      familyId: cluster.familyId,
      signature: evidenceSignature(cluster),
      containsOldMemberIds: cluster.members.map((member) => member.occurrence.id)
        .filter((id) => memberIdSet.has(id)),
      supportEpisodeCount: cluster.supportEpisodeIds.length
    })),
    deduplicatedClustersContainingEitherMember: deduplicatedContaining.map((cluster) => ({
      id: cluster.id,
      familyId: cluster.familyId,
      signature: evidenceSignature(cluster),
      containsOldMemberIds: cluster.members.map((member) => member.occurrence.id)
        .filter((id) => memberIdSet.has(id)),
      supportEpisodeCount: cluster.supportEpisodeIds.length
    }))
  };
}

function configForScale(scale: number): BandedMonotonicMatchConfig {
  const config = V15_FINE_MATCH_CONFIGS.find((candidate) => candidate.scale === scale);
  if (!config) throw new Error(`Missing fine config for Span-${scale}`);
  return config;
}

function roleForReward(reward: number | null): ProceduralEvidenceRole {
  if (reward === null || reward === 0) return "unknown";
  return reward > 0 ? "support" : "counterexample";
}

function scoreToMedoid(
  member: EmbeddedTrajectoryWindowV1,
  medoid: EmbeddedTrajectoryWindowV1,
  config: BandedMonotonicMatchConfig
): number {
  return member.occurrence.id === medoid.occurrence.id
    ? 1
    : bandedMonotonicMatch(member.stepVectors, medoid.stepVectors, config).score;
}

function rangesOverlap(
  left: EmbeddedTrajectoryWindowV1,
  right: EmbeddedTrajectoryWindowV1
): boolean {
  return left.occurrence.startStepIndex <= right.occurrence.endStepIndex &&
    right.occurrence.startStepIndex <= left.occurrence.endStepIndex;
}

function uniqueWindows(windows: EmbeddedTrajectoryWindowV1[]): EmbeddedTrajectoryWindowV1[] {
  return [...new Map(windows.map((window) => [window.occurrence.id, window])).values()];
}

function distinctEpisodes(members: TrajectoryWindowClusterMemberV1[]): string[] {
  return unique(members.map((member) => member.occurrence.episodeId)).sort();
}

function unitVector(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return norm === 0 ? vector.map(() => 0) : vector.map((value) => value / norm);
}

function average(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function numberArray(json: string): number[] {
  const parsed = JSON.parse(json) as unknown;
  if (!Array.isArray(parsed) || parsed.some((value) => typeof value !== "number")) {
    throw new Error("Invalid frozen embedding vector");
  }
  return parsed as number[];
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function string(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function integer(value: unknown, fallback: number): number {
  return Number.isInteger(value) ? Number(value) : fallback;
}

function outcomeStatus(value: unknown): ExecutionStepLiteOutcome {
  return value === "success" || value === "failure" || value === "partial"
    ? value
    : "unknown";
}

function parseArgs(argv: string[]): Args {
  const defaults = resolve(
    "../../experiments/results/trace2skill-multi-scale-window-kw"
  );
  const args: Args = {
    sourceDbPath: resolve(defaults, "20260826-kw-multi-medoid-span5-076-span10-070-cluster-v15.sqlite"),
    v15ResultPath: resolve(defaults, "20260826-kw-multi-medoid-span5-076-span10-070-cluster-v15.json"),
    outputReportPath: resolve(defaults, "20260827-kw-frozen-v15-coarse-current-downstream-v1.json")
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]!;
    const next = argv[index + 1];
    if (token === "--source-db" && next) {
      args.sourceDbPath = resolve(next);
      index += 1;
    } else if (token === "--v15-result" && next) {
      args.v15ResultPath = resolve(next);
      index += 1;
    } else if (token === "--output-report" && next) {
      args.outputReportPath = resolve(next);
      index += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${token}`);
    }
  }
  return args;
}

main();
