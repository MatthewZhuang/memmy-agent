import Database from "better-sqlite3";
import { fork } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { availableParallelism, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  MemoryDb,
  MemoryService,
  clusterTrajectoryWindows,
  loadMemmyConfig,
  selectMaximalWindowClusters,
  type BandedMonotonicMatchConfig,
  type EmbeddedTrajectoryWindowV1,
  type MultiScaleWindowFineMembershipMode,
  type MultiScaleWindowSpec,
  type MultiScaleWindowPolicyDecisionV1,
  type MultiScaleWindowPolicyV1,
  type MultiScaleWindowSkillCandidateV1,
  type MultiScaleWindowSkillDecisionV1,
  type PreparedMultiScaleWindowPolicyExperimentV1,
  type TrajectoryWindowClusterV1
} from "../src/index.js";

const SWEEP_SCHEMA_VERSION = "multi-scale-window-fine-sweep.v1" as const;
const DEFAULT_SPECS: MultiScaleWindowSpec[] = [
  { length: 5, stride: 2 },
  { length: 10, stride: 5 }
];

interface FineLevel {
  name: string;
  minStepSimilarity: number;
  minAverageMatchSimilarity: number;
  minAlignmentScore: number;
}

const SPAN5_LEVELS: FineLevel[] = [
  {
    name: "baseline",
    minStepSimilarity: 0.70,
    minAverageMatchSimilarity: 0.78,
    minAlignmentScore: 0.62
  },
  {
    name: "relax-1",
    minStepSimilarity: 0.69,
    minAverageMatchSimilarity: 0.77,
    minAlignmentScore: 0.60
  },
  {
    name: "relax-2",
    minStepSimilarity: 0.68,
    minAverageMatchSimilarity: 0.76,
    minAlignmentScore: 0.58
  },
  {
    name: "relax-3",
    minStepSimilarity: 0.67,
    minAverageMatchSimilarity: 0.75,
    minAlignmentScore: 0.56
  },
  {
    name: "relax-4",
    minStepSimilarity: 0.66,
    minAverageMatchSimilarity: 0.74,
    minAlignmentScore: 0.54
  }
];

const SPAN10_LEVELS: FineLevel[] = [
  {
    name: "baseline",
    minStepSimilarity: 0.68,
    minAverageMatchSimilarity: 0.76,
    minAlignmentScore: 0.52
  },
  {
    name: "relax-1",
    minStepSimilarity: 0.67,
    minAverageMatchSimilarity: 0.75,
    minAlignmentScore: 0.50
  },
  {
    name: "relax-2",
    minStepSimilarity: 0.66,
    minAverageMatchSimilarity: 0.74,
    minAlignmentScore: 0.48
  },
  {
    name: "relax-3",
    minStepSimilarity: 0.65,
    minAverageMatchSimilarity: 0.73,
    minAlignmentScore: 0.46
  },
  {
    name: "relax-4",
    minStepSimilarity: 0.64,
    minAverageMatchSimilarity: 0.72,
    minAlignmentScore: 0.44
  }
];

interface SweepSetting {
  id: string;
  suite: "coupled-grid" | "single-axis" | "match-coverage-ablation" |
    "policy-coverage-ablation";
  span5Level: string;
  span10Level: string;
  fineMatchConfigs: BandedMonotonicMatchConfig[];
  fineMembershipMode?: MultiScaleWindowFineMembershipMode;
}

interface Args {
  baselineArtifactPath?: string;
  dbPath: string;
  configPath?: string;
  outputDbPath: string;
  outputPath: string;
  episodeIds: string[];
  specs: MultiScaleWindowSpec[];
  span5CoarseSimilarityThreshold: number;
  span10CoarseSimilarityThreshold: number;
  coarseMembershipMode: "exclusive" | "multi";
  minSupportEpisodes: number;
  concurrency: number;
  matchCoverageAblation: boolean;
  policyCoverageAblation: boolean;
}

interface WorkerInput {
  setting: SweepSetting;
  windows: EmbeddedTrajectoryWindowV1[];
  episodeCount: number;
  embeddedStepCount: number;
  span5CoarseSimilarityThreshold: number;
  span10CoarseSimilarityThreshold: number;
  coarseMembershipMode: "exclusive" | "multi";
  minSupportEpisodes: number;
}

interface WorkerLaunchInput extends Omit<WorkerInput, "windows"> {
  cachePath: string;
}

interface ProjectedSpan {
  clusterId: string;
  spanId: string;
  episodeId: string;
  startStepIndex: number;
  endStepIndex: number;
  scale: number;
  supportEpisodeCount: number;
  averagePairwiseSimilarity: number;
}

interface SkillSequenceCandidate {
  skillCandidateId?: string;
  policyClusterIds: string[];
  policyIds?: string[];
  supportEpisodeIds: string[];
  evidence: Array<{
    episodeId: string;
    spans: ProjectedSpan[];
  }>;
  semanticQualityStatus: "pending_policy_and_skill_review";
}

interface SweepMetrics {
  totalEmbeddedSteps: number;
  clusters: number;
  singletonClusters: number;
  maxClusterSpans: number;
  clustersLargerThanSix: number;
  supportedClusters: number;
  supportedSpan5Clusters: number;
  supportedSpan10Clusters: number;
  maximalPolicyCandidates: number;
  uniqueCandidateSpans: number;
  candidateSpanCoverage: number;
  projectedEpisodes: number;
  episodesWithAtLeastTwoPolicies: number;
  projectedSteps: number;
  projectedStepCoverage: number;
  repeatablePolicyBigrams: number;
  repeatablePolicyTrigrams: number;
  admittedPolicies: number;
  rejectedPolicies: number;
  failedPolicyInductions: number;
  skillCandidates: number;
  admittedSkills: number;
  rejectedSkills: number;
  failedSkillInductions: number;
}

interface SweepRun {
  setting: SweepSetting;
  metrics: SweepMetrics;
  candidateClusters: TrajectoryWindowClusterV1[];
  episodePolicyPaths: Array<{
    episodeId: string;
    spans: ProjectedSpan[];
  }>;
  skillSequenceCandidates: SkillSequenceCandidate[];
  policyDecisions: MultiScaleWindowPolicyDecisionV1[];
  skillDecisions: MultiScaleWindowSkillDecisionV1[];
  finalQualityStatus: "not_approved_pending_policy_and_skill_review";
}

interface WorkerResponse {
  ok: boolean;
  run?: SweepRun;
  error?: string;
}

interface BaselineArtifact {
  sourceDatabase: string;
  episodeIds: string[];
  result?: {
    specs?: MultiScaleWindowSpec[];
    coarseSimilarityThresholdByScale?: Record<string, number>;
    coarseMembershipMode?: "exclusive" | "multi";
    minSupportEpisodes?: number;
  };
}

if (!process.argv.includes("--sweep-worker")) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
} else {
  process.once("message", (input: WorkerLaunchInput) => {
    const windows = JSON.parse(readFileSync(input.cachePath, "utf8")) as
      EmbeddedTrajectoryWindowV1[];
    runWorker({ ...input, windows });
  });
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const settings = args.matchCoverageAblation
    ? buildMatchCoverageAblationSettings()
    : args.policyCoverageAblation
      ? buildPolicyCoverageAblationSettings()
      : buildSettings();
  mkdirSync(dirname(args.outputDbPath), { recursive: true });
  mkdirSync(dirname(args.outputPath), { recursive: true });
  await cloneDatabase(args.dbPath, args.outputDbPath);
  const loaded = loadMemmyConfig(args.configPath);
  const db = new MemoryDb({ path: args.outputDbPath });
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();
  let prepared: PreparedMultiScaleWindowPolicyExperimentV1;
  try {
    const service = new MemoryService({
      db,
      mode: "local",
      config: loaded.config,
      ...(args.configPath ? { configPath: args.configPath } : {})
    });
    process.stdout.write(`${JSON.stringify({
      phase: "embedding",
      episodes: args.episodeIds.length,
      settings: settings.length,
      concurrency: args.concurrency
    })}\n`);
    prepared = await service.prepareMultiScaleWindowPoliciesForReplay({
      episodeIds: args.episodeIds,
      specs: args.specs
    });
  } finally {
    db.close();
  }

  process.stdout.write(`${JSON.stringify({
    phase: "parallel_fine_sweep",
    windows: prepared.windows.length,
    embeddedSteps: prepared.embeddedStepCount,
    settings: settings.length,
    concurrency: args.concurrency
  })}\n`);
  const runs = await runSettingsInParallel(settings, prepared, args);
  const evaluatedRuns = await evaluateRunsWithPolicyAndSkillLlm(runs, args, loaded.config);
  const rankedRuns = [...evaluatedRuns].sort(compareRuns);
  const artifact = {
    schemaVersion: SWEEP_SCHEMA_VERSION,
    startedAt,
    finishedAt: new Date().toISOString(),
    elapsedMs: Date.now() - wallStart,
    sourceDatabase: resolve(args.dbPath),
    outputDatabase: args.outputDbPath,
    baselineArtifact: args.baselineArtifactPath,
    episodeIds: args.episodeIds,
    specs: args.specs,
    coarseSimilarityThresholdByScale: {
      5: args.span5CoarseSimilarityThreshold,
      10: args.span10CoarseSimilarityThreshold
    },
    coarseMembershipMode: args.coarseMembershipMode,
    minSupportEpisodes: args.minSupportEpisodes,
    models: prepared.models,
    preparation: {
      episodes: prepared.episodeCount,
      steps: prepared.stepCount,
      embeddedSteps: prepared.embeddedStepCount,
      spans: prepared.windows.length,
      embeddingPasses: 1,
      parallelSettings: settings.length,
      concurrency: args.concurrency
    },
    evaluationPolicy: {
      automaticRankingOnly: true,
      winnerSelectionAllowed: false,
      policyQualityGate: [
        "same reusable method across Episodes",
        "atomic and non-generic procedure",
        "evidence-grounded ordered Steps",
        "no unsupported success claims",
        "mechanical boundary noise excluded"
      ],
      skillQualityGate: [
        "at least two distinct Policies",
        "same ordered Policy sequence in at least two Episodes",
        "coherent reusable goal or sub-goal",
        "non-trivial compression benefit",
        "all source Policies pass Policy quality review"
      ]
    },
    provisionalShortlist: rankedRuns.slice(0, 3).map((run, index) => ({
      rank: index + 1,
      settingId: run.setting.id,
      metrics: run.metrics,
      finalQualityStatus: run.finalQualityStatus
    })),
    runs: evaluatedRuns
  };
  writeFileSync(args.outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  const markdownPath = args.outputPath.replace(/\.json$/i, ".md");
  const csvPath = args.outputPath.replace(/\.json$/i, ".csv");
  writeFileSync(markdownPath, renderMarkdown(artifact), "utf8");
  writeFileSync(csvPath, renderCsv(rankedRuns), "utf8");
  process.stdout.write(`${JSON.stringify({
    phase: "complete",
    output: args.outputPath,
    markdown: markdownPath,
    csv: csvPath,
    provisionalShortlist: artifact.provisionalShortlist
  }, null, 2)}\n`);
}

async function evaluateRunsWithPolicyAndSkillLlm(
  runs: readonly SweepRun[],
  args: Args,
  config: ReturnType<typeof loadMemmyConfig>["config"]
): Promise<SweepRun[]> {
  const db = new MemoryDb({ path: args.outputDbPath });
  try {
    const service = new MemoryService({
      db,
      mode: "local",
      config,
      ...(args.configPath ? { configPath: args.configPath } : {})
    });
    const policyClusterByEvaluationId = new Map<string, TrajectoryWindowClusterV1>();
    const evaluationIdByRunCluster = new Map<string, string>();
    for (const run of runs) {
      for (const cluster of run.candidateClusters) {
        const evaluationId = policyClusterEvaluationId(cluster);
        evaluationIdByRunCluster.set(`${run.setting.id}|${cluster.id}`, evaluationId);
        if (!policyClusterByEvaluationId.has(evaluationId)) {
          policyClusterByEvaluationId.set(evaluationId, { ...cluster, id: evaluationId });
        }
      }
    }
    const uniquePolicyClusters = [...policyClusterByEvaluationId.values()];
    process.stdout.write(`${JSON.stringify({
      phase: "policy_llm",
      uniqueEvidenceClusters: uniquePolicyClusters.length,
      concurrency: Math.min(4, args.concurrency)
    })}\n`);
    const policyDecisions = await service.induceMultiScaleWindowPoliciesForReplay({
      clusters: uniquePolicyClusters,
      concurrency: Math.min(4, args.concurrency)
    });
    const policyDecisionByEvaluationId = new Map(policyDecisions.map((decision) =>
      [decision.clusterId, decision]));

    const policyEvaluated = runs.map((run) => evaluateRunPolicies(
      run,
      evaluationIdByRunCluster,
      policyDecisionByEvaluationId,
      !args.policyCoverageAblation
    ));
    if (args.policyCoverageAblation) {
      process.stdout.write(`${JSON.stringify({
        phase: "skill_llm",
        skipped: true,
        reason: "policy coverage ablation"
      })}\n`);
      return policyEvaluated;
    }
    const uniqueSkillCandidates = new Map<string, MultiScaleWindowSkillCandidateV1>();
    for (const run of policyEvaluated) {
      const policiesByClusterId = new Map(run.policyDecisions.flatMap((decision) =>
        decision.policy ? [[decision.clusterId, decision.policy] as const] : []));
      run.skillSequenceCandidates = run.skillSequenceCandidates.map((candidate) => {
        const skillCandidate = buildSkillCandidate(candidate, policiesByClusterId);
        uniqueSkillCandidates.set(skillCandidate.id, skillCandidate);
        return {
          ...candidate,
          skillCandidateId: skillCandidate.id,
          policyIds: skillCandidate.policyIds
        };
      });
    }
    process.stdout.write(`${JSON.stringify({
      phase: "skill_llm",
      uniqueSkillCandidates: uniqueSkillCandidates.size,
      concurrency: Math.min(4, args.concurrency)
    })}\n`);
    const skillDecisions = uniqueSkillCandidates.size === 0
      ? []
      : await service.induceMultiScaleWindowSkillsForReplay({
          candidates: [...uniqueSkillCandidates.values()],
          concurrency: Math.min(4, args.concurrency)
        });
    const skillDecisionByCandidateId = new Map(skillDecisions.map((decision) =>
      [decision.candidateId, decision]));
    return policyEvaluated.map((run) => {
      const decisions = run.skillSequenceCandidates.flatMap((candidate) => {
        const decision = candidate.skillCandidateId
          ? skillDecisionByCandidateId.get(candidate.skillCandidateId)
          : undefined;
        return decision ? [decision] : [];
      });
      return {
        ...run,
        metrics: {
          ...run.metrics,
          skillCandidates: run.skillSequenceCandidates.length,
          admittedSkills: decisions.filter((decision) => decision.admitted).length,
          rejectedSkills: decisions.filter((decision) =>
            !decision.admitted && !decision.error).length,
          failedSkillInductions: decisions.filter((decision) => decision.error).length
        },
        skillDecisions: decisions
      };
    });
  } finally {
    db.close();
  }
}

function evaluateRunPolicies(
  run: SweepRun,
  evaluationIdByRunCluster: ReadonlyMap<string, string>,
  policyDecisionByEvaluationId: ReadonlyMap<string, MultiScaleWindowPolicyDecisionV1>,
  mineSkills: boolean
): SweepRun {
  const decisions: MultiScaleWindowPolicyDecisionV1[] = [];
  const policyByClusterId = new Map<string, MultiScaleWindowPolicyV1>();
  for (const cluster of run.candidateClusters) {
    const evaluationId = evaluationIdByRunCluster.get(`${run.setting.id}|${cluster.id}`);
    const decision = evaluationId
      ? policyDecisionByEvaluationId.get(evaluationId)
      : undefined;
    if (!decision) continue;
    const mapped: MultiScaleWindowPolicyDecisionV1 = {
      ...decision,
      clusterId: cluster.id,
      ...(decision.policy
        ? { policy: { ...decision.policy, clusterId: cluster.id } }
        : {})
    };
    decisions.push(mapped);
    if (mapped.policy) policyByClusterId.set(cluster.id, mapped.policy);
  }
  const admittedClusters = run.candidateClusters.filter((cluster) =>
    policyByClusterId.has(cluster.id));
  const episodePolicyPaths = projectEpisodePolicyPaths(admittedClusters);
  const skillSequenceCandidates = mineSkills
    ? mineSkillSequenceCandidates(episodePolicyPaths, 2)
    : [];
  const projectedStepIds = new Set(episodePolicyPaths.flatMap((path) =>
    path.spans.flatMap((span) => {
      const cluster = admittedClusters.find((candidate) => candidate.id === span.clusterId);
      const member = cluster?.members.find((candidate) =>
        candidate.occurrence.id === span.spanId);
      return member?.occurrence.steps.map((step) => step.id) ?? [];
    })));
  return {
    ...run,
    metrics: {
      ...run.metrics,
      projectedEpisodes: episodePolicyPaths.filter((path) => path.spans.length > 0).length,
      episodesWithAtLeastTwoPolicies: episodePolicyPaths.filter((path) =>
        path.spans.length >= 2).length,
      projectedSteps: projectedStepIds.size,
      projectedStepCoverage: ratio(projectedStepIds.size, run.metrics.totalEmbeddedSteps),
      repeatablePolicyBigrams: skillSequenceCandidates.filter((candidate) =>
        candidate.policyClusterIds.length === 2).length,
      repeatablePolicyTrigrams: skillSequenceCandidates.filter((candidate) =>
        candidate.policyClusterIds.length === 3).length,
      admittedPolicies: decisions.filter((decision) => decision.admitted).length,
      rejectedPolicies: decisions.filter((decision) =>
        !decision.admitted && !decision.error).length,
      failedPolicyInductions: decisions.filter((decision) => decision.error).length
    },
    episodePolicyPaths,
    skillSequenceCandidates,
    policyDecisions: decisions
  };
}

function buildSkillCandidate(
  sequence: SkillSequenceCandidate,
  policiesByClusterId: ReadonlyMap<string, MultiScaleWindowPolicyV1>
): MultiScaleWindowSkillCandidateV1 {
  const policies = sequence.policyClusterIds.map((clusterId) => {
    const policy = policiesByClusterId.get(clusterId);
    if (!policy) throw new Error(`missing admitted Policy for Cluster ${clusterId}`);
    return policy;
  });
  const policyIds = policies.map((policy) => policy.id);
  const evidence = sequence.evidence.map((item) => ({
    episodeId: item.episodeId,
    spans: item.spans.map((span) => {
      const policy = policiesByClusterId.get(span.clusterId);
      if (!policy) throw new Error(`missing admitted Policy for Span Cluster ${span.clusterId}`);
      return {
        policyId: policy.id,
        clusterId: span.clusterId,
        spanId: span.spanId,
        startStepIndex: span.startStepIndex,
        endStepIndex: span.endStepIndex
      };
    })
  }));
  const supportEpisodeIds = sequence.supportEpisodeIds;
  const id = `multi_scale_window_skill_candidate_${hashJson({
    policyIds,
    supportEpisodeIds,
    evidence
  }).slice(0, 24)}`;
  return { id, policyIds, policies, supportEpisodeIds, evidence };
}

function policyClusterEvaluationId(cluster: TrajectoryWindowClusterV1): string {
  return `sweep_policy_cluster_${hashJson(cluster).slice(0, 24)}`;
}

function hashJson(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function runWorker(input: WorkerInput): void {
  try {
    const discovery = clusterTrajectoryWindows(input.windows, {
      coarseSimilarityThreshold: input.span5CoarseSimilarityThreshold,
      coarseSimilarityThresholdByScale: {
        5: input.span5CoarseSimilarityThreshold,
        10: input.span10CoarseSimilarityThreshold
      },
      fineMatchConfigs: input.setting.fineMatchConfigs,
      coarseMembershipMode: input.coarseMembershipMode,
      fineMembershipMode: input.setting.fineMembershipMode ?? "exclusive"
    });
    const supported = discovery.clusters.filter((cluster) =>
      cluster.supportEpisodeIds.length >= input.minSupportEpisodes);
    const candidates = selectMaximalWindowClusters(supported)
      .filter((cluster) => !cluster.suppressedByClusterId)
      .sort((left, right) => right.scale - left.scale ||
        right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
        right.averageSimilarity - left.averageSimilarity ||
        left.id.localeCompare(right.id));
    const episodePolicyPaths = projectEpisodePolicyPaths(candidates);
    const skillSequenceCandidates = mineSkillSequenceCandidates(episodePolicyPaths, 2);
    const candidateSpanIds = new Set(candidates.flatMap((cluster) =>
      cluster.members.map((member) => member.occurrence.id)));
    const projectedStepIds = new Set(episodePolicyPaths.flatMap((path) =>
      path.spans.flatMap((span) => {
        const cluster = candidates.find((candidate) => candidate.id === span.clusterId);
        const member = cluster?.members.find((candidate) =>
          candidate.occurrence.id === span.spanId);
        return member?.occurrence.steps.map((step) => step.id) ?? [];
      })));
    const metrics: SweepMetrics = {
      totalEmbeddedSteps: input.embeddedStepCount,
      clusters: discovery.clusters.length,
      singletonClusters: discovery.clusters.filter((cluster) =>
        cluster.occurrenceCount === 1).length,
      maxClusterSpans: Math.max(0, ...discovery.clusters.map((cluster) =>
        cluster.occurrenceCount)),
      clustersLargerThanSix: discovery.clusters.filter((cluster) =>
        cluster.occurrenceCount > 6).length,
      supportedClusters: supported.length,
      supportedSpan5Clusters: supported.filter((cluster) => cluster.scale === 5).length,
      supportedSpan10Clusters: supported.filter((cluster) => cluster.scale === 10).length,
      maximalPolicyCandidates: candidates.length,
      uniqueCandidateSpans: candidateSpanIds.size,
      candidateSpanCoverage: ratio(candidateSpanIds.size, input.windows.length),
      projectedEpisodes: episodePolicyPaths.filter((path) => path.spans.length > 0).length,
      episodesWithAtLeastTwoPolicies: episodePolicyPaths.filter((path) =>
        path.spans.length >= 2).length,
      projectedSteps: projectedStepIds.size,
      projectedStepCoverage: ratio(projectedStepIds.size, input.embeddedStepCount),
      repeatablePolicyBigrams: skillSequenceCandidates.filter((candidate) =>
        candidate.policyClusterIds.length === 2).length,
      repeatablePolicyTrigrams: skillSequenceCandidates.filter((candidate) =>
        candidate.policyClusterIds.length === 3).length,
      admittedPolicies: 0,
      rejectedPolicies: 0,
      failedPolicyInductions: 0,
      skillCandidates: 0,
      admittedSkills: 0,
      rejectedSkills: 0,
      failedSkillInductions: 0
    };
    const response: WorkerResponse = {
      ok: true,
      run: {
        setting: input.setting,
        metrics,
        candidateClusters: candidates,
        episodePolicyPaths,
        skillSequenceCandidates,
        policyDecisions: [],
        skillDecisions: [],
        finalQualityStatus: "not_approved_pending_policy_and_skill_review"
      }
    };
    sendWorkerResponse(response);
  } catch (error) {
    const response: WorkerResponse = {
      ok: false,
      error: error instanceof Error ? error.stack ?? error.message : String(error)
    };
    sendWorkerResponse(response);
  }
}

function sendWorkerResponse(response: WorkerResponse): void {
  if (!process.send) {
    process.stderr.write("fine sweep worker has no IPC channel\n");
    process.exitCode = 1;
    return;
  }
  process.send(response, (error) => {
    if (error) {
      process.stderr.write(`${error.stack ?? error.message}\n`);
      process.exitCode = 1;
    }
    process.disconnect();
  });
}

async function runSettingsInParallel(
  settings: readonly SweepSetting[],
  prepared: PreparedMultiScaleWindowPolicyExperimentV1,
  args: Args
): Promise<SweepRun[]> {
  const cacheDirectory = mkdtempSync(join(tmpdir(), "memmy-fine-sweep-"));
  const cachePath = join(cacheDirectory, "embedded-windows.json");
  writeFileSync(cachePath, JSON.stringify(prepared.windows), "utf8");
  try {
    const results = new Array<SweepRun>(settings.length);
    let nextIndex = 0;
    let completed = 0;
    const workerCount = Math.min(args.concurrency, settings.length);
    await Promise.all(Array.from({ length: workerCount }, async () => {
      while (true) {
        const index = nextIndex;
        nextIndex += 1;
        if (index >= settings.length) return;
        const setting = settings[index]!;
        const input: WorkerLaunchInput = {
          setting,
          cachePath,
          episodeCount: prepared.episodeCount,
          embeddedStepCount: prepared.embeddedStepCount,
          span5CoarseSimilarityThreshold: args.span5CoarseSimilarityThreshold,
          span10CoarseSimilarityThreshold: args.span10CoarseSimilarityThreshold,
          coarseMembershipMode: args.coarseMembershipMode,
          minSupportEpisodes: args.minSupportEpisodes
        };
        results[index] = await runOneWorker(input);
        completed += 1;
        process.stdout.write(`${JSON.stringify({
          phase: "parallel_fine_sweep",
          completed,
          total: settings.length,
          setting: setting.id,
          metrics: results[index]!.metrics
        })}\n`);
      }
    }));
    return results;
  } finally {
    rmSync(cacheDirectory, { recursive: true, force: true });
  }
}

function runOneWorker(input: WorkerLaunchInput): Promise<SweepRun> {
  return new Promise((resolvePromise, rejectPromise) => {
    const worker = fork(fileURLToPath(import.meta.url), ["--sweep-worker"], {
      execArgv: ["--import", "tsx"],
      stdio: ["ignore", "ignore", "inherit", "ipc"]
    });
    let settled = false;
    worker.once("message", (response: WorkerResponse) => {
      settled = true;
      if (response.ok && response.run) resolvePromise(response.run);
      else rejectPromise(new Error(response.error ?? "fine sweep worker failed"));
    });
    worker.once("error", (error) => {
      settled = true;
      rejectPromise(error);
    });
    worker.once("exit", (code) => {
      if (!settled) {
        rejectPromise(new Error(`fine sweep worker exited before returning a result: ${code}`));
      }
    });
    worker.send(input, (error) => {
      if (error && !settled) {
        settled = true;
        rejectPromise(error);
      }
    });
  });
}

function buildSettings(): SweepSetting[] {
  const settings: SweepSetting[] = [];
  for (const span5 of SPAN5_LEVELS) {
    for (const span10 of SPAN10_LEVELS) {
      settings.push({
        id: `grid__span5-${span5.name}__span10-${span10.name}`,
        suite: "coupled-grid",
        span5Level: span5.name,
        span10Level: span10.name,
        fineMatchConfigs: fineConfigs(span5, span10)
      });
    }
  }
  const baseline5 = SPAN5_LEVELS[0]!;
  const baseline10 = SPAN10_LEVELS[0]!;
  const axes: Array<keyof Pick<FineLevel,
    "minStepSimilarity" | "minAverageMatchSimilarity" | "minAlignmentScore">> = [
      "minStepSimilarity",
      "minAverageMatchSimilarity",
      "minAlignmentScore"
    ];
  for (const relaxed of SPAN5_LEVELS.slice(1)) {
    for (const axis of axes) {
      const span5 = { ...baseline5, [axis]: relaxed[axis] };
      settings.push({
        id: `axis__span5-${axis}-${relaxed.name}__span10-baseline`,
        suite: "single-axis",
        span5Level: `${axis}:${relaxed.name}`,
        span10Level: "baseline",
        fineMatchConfigs: fineConfigs(span5, baseline10)
      });
    }
  }
  for (const relaxed of SPAN10_LEVELS.slice(1)) {
    for (const axis of axes) {
      const span10 = { ...baseline10, [axis]: relaxed[axis] };
      settings.push({
        id: `axis__span5-baseline__span10-${axis}-${relaxed.name}`,
        suite: "single-axis",
        span5Level: "baseline",
        span10Level: `${axis}:${relaxed.name}`,
        fineMatchConfigs: fineConfigs(baseline5, span10)
      });
    }
  }
  return settings;
}

function buildMatchCoverageAblationSettings(): SweepSetting[] {
  const span5 = SPAN5_LEVELS.at(-1)!;
  const span10 = SPAN10_LEVELS.at(-1)!;
  const control = fineConfigs(span5, span10);
  const relaxed = control.map((config) => config.scale === 5
    ? { ...config, minMatchedSteps: 3, minCoverage: 0.60 }
    : { ...config, minMatchedSteps: 6, minCoverage: 0.60 });
  return [
    {
      id: "match-coverage__control-4of5-7of10",
      suite: "match-coverage-ablation",
      span5Level: "relax-4:matches-4:coverage-0.8",
      span10Level: "relax-4:matches-7:coverage-0.7",
      fineMatchConfigs: control
    },
    {
      id: "match-coverage__experiment-3of5-6of10",
      suite: "match-coverage-ablation",
      span5Level: "relax-4:matches-3:coverage-0.6",
      span10Level: "relax-4:matches-6:coverage-0.6",
      fineMatchConfigs: relaxed
    }
  ];
}

function buildPolicyCoverageAblationSettings(): SweepSetting[] {
  const span5 = SPAN5_LEVELS.at(-1)!;
  const span10 = SPAN10_LEVELS.at(-1)!;
  const control = fineConfigs(span5, span10);
  const relaxed = control.map((config) => config.scale === 5
    ? { ...config, minMatchedSteps: 3, minCoverage: 0.60 }
    : { ...config, minMatchedSteps: 6, minCoverage: 0.60 });
  return [
    {
      id: "policy-coverage__exclusive-control-4of5-7of10",
      suite: "policy-coverage-ablation",
      span5Level: "relax-4:matches-4:coverage-0.8",
      span10Level: "relax-4:matches-7:coverage-0.7",
      fineMatchConfigs: control,
      fineMembershipMode: "exclusive"
    },
    {
      id: "policy-coverage__overlap-control-4of5-7of10",
      suite: "policy-coverage-ablation",
      span5Level: "relax-4:matches-4:coverage-0.8",
      span10Level: "relax-4:matches-7:coverage-0.7",
      fineMatchConfigs: control,
      fineMembershipMode: "overlap"
    },
    {
      id: "policy-coverage__overlap-experiment-3of5-6of10",
      suite: "policy-coverage-ablation",
      span5Level: "relax-4:matches-3:coverage-0.6",
      span10Level: "relax-4:matches-6:coverage-0.6",
      fineMatchConfigs: relaxed,
      fineMembershipMode: "overlap"
    }
  ];
}

function fineConfigs(
  span5: FineLevel,
  span10: FineLevel
): BandedMonotonicMatchConfig[] {
  return [
    {
      scale: 5,
      bandWidth: 1,
      minStepSimilarity: span5.minStepSimilarity,
      minMatchedSteps: 4,
      minCoverage: 0.80,
      minAverageMatchSimilarity: span5.minAverageMatchSimilarity,
      maxInternalGap: 1,
      gapPenalty: 0.10,
      minAlignmentScore: span5.minAlignmentScore
    },
    {
      scale: 10,
      bandWidth: 2,
      minStepSimilarity: span10.minStepSimilarity,
      minMatchedSteps: 7,
      minCoverage: 0.70,
      minAverageMatchSimilarity: span10.minAverageMatchSimilarity,
      maxInternalGap: 2,
      gapPenalty: 0.10,
      minAlignmentScore: span10.minAlignmentScore
    }
  ];
}

function projectEpisodePolicyPaths(
  clusters: readonly TrajectoryWindowClusterV1[]
): Array<{ episodeId: string; spans: ProjectedSpan[] }> {
  const byEpisode = new Map<string, ProjectedSpan[]>();
  for (const cluster of clusters) {
    for (const member of cluster.members) {
      const occurrence = member.occurrence;
      const span: ProjectedSpan = {
        clusterId: cluster.id,
        spanId: occurrence.id,
        episodeId: occurrence.episodeId,
        startStepIndex: occurrence.startStepIndex,
        endStepIndex: occurrence.endStepIndex,
        scale: occurrence.scale,
        supportEpisodeCount: cluster.supportEpisodeIds.length,
        averagePairwiseSimilarity: cluster.averagePairwiseSimilarity
      };
      const current = byEpisode.get(occurrence.episodeId) ?? [];
      current.push(span);
      byEpisode.set(occurrence.episodeId, current);
    }
  }
  return [...byEpisode.entries()].map(([episodeId, spans]) => ({
    episodeId,
    spans: selectLongestNonOverlappingSpans(spans)
  })).sort((left, right) => left.episodeId.localeCompare(right.episodeId));
}

function selectLongestNonOverlappingSpans(spans: readonly ProjectedSpan[]): ProjectedSpan[] {
  const uniqueBySpan = new Map<string, ProjectedSpan>();
  for (const span of spans) {
    const existing = uniqueBySpan.get(span.spanId);
    if (!existing || compareProjectedSpanPriority(span, existing) < 0) {
      uniqueBySpan.set(span.spanId, span);
    }
  }
  const selected: ProjectedSpan[] = [];
  for (const span of [...uniqueBySpan.values()].sort(compareProjectedSpanPriority)) {
    if (selected.some((existing) => rangesOverlap(
      span.startStepIndex,
      span.endStepIndex,
      existing.startStepIndex,
      existing.endStepIndex
    ))) continue;
    selected.push(span);
  }
  return selected.sort((left, right) => left.startStepIndex - right.startStepIndex ||
    right.scale - left.scale || left.clusterId.localeCompare(right.clusterId));
}

function compareProjectedSpanPriority(left: ProjectedSpan, right: ProjectedSpan): number {
  return right.scale - left.scale ||
    right.supportEpisodeCount - left.supportEpisodeCount ||
    right.averagePairwiseSimilarity - left.averagePairwiseSimilarity ||
    left.startStepIndex - right.startStepIndex ||
    left.clusterId.localeCompare(right.clusterId);
}

function mineSkillSequenceCandidates(
  paths: readonly Array<{ episodeId: string; spans: ProjectedSpan[] }>,
  minSupportEpisodes: number
): SkillSequenceCandidate[] {
  const candidates = new Map<string, {
    policyClusterIds: string[];
    evidence: Map<string, ProjectedSpan[]>;
  }>();
  for (const length of [2, 3]) {
    for (const path of paths) {
      for (let start = 0; start + length <= path.spans.length; start += 1) {
        const spans = path.spans.slice(start, start + length);
        const policyClusterIds = spans.map((span) => span.clusterId);
        if (new Set(policyClusterIds).size < 2) continue;
        const key = policyClusterIds.join("→");
        const candidate = candidates.get(key) ?? {
          policyClusterIds,
          evidence: new Map<string, ProjectedSpan[]>()
        };
        candidate.evidence.set(path.episodeId, spans);
        candidates.set(key, candidate);
      }
    }
  }
  return [...candidates.values()]
    .filter((candidate) => candidate.evidence.size >= minSupportEpisodes)
    .map((candidate) => ({
      policyClusterIds: candidate.policyClusterIds,
      supportEpisodeIds: [...candidate.evidence.keys()].sort(),
      evidence: [...candidate.evidence.entries()].map(([episodeId, spans]) => ({
        episodeId,
        spans
      })).sort((left, right) => left.episodeId.localeCompare(right.episodeId)),
      semanticQualityStatus: "pending_policy_and_skill_review" as const
    }))
    .sort((left, right) => right.policyClusterIds.length - left.policyClusterIds.length ||
      right.supportEpisodeIds.length - left.supportEpisodeIds.length ||
      left.policyClusterIds.join("→").localeCompare(right.policyClusterIds.join("→")));
}

function compareRuns(left: SweepRun, right: SweepRun): number {
  return right.metrics.admittedSkills - left.metrics.admittedSkills ||
    left.metrics.failedSkillInductions - right.metrics.failedSkillInductions ||
    right.metrics.repeatablePolicyTrigrams - left.metrics.repeatablePolicyTrigrams ||
    right.metrics.repeatablePolicyBigrams - left.metrics.repeatablePolicyBigrams ||
    right.metrics.episodesWithAtLeastTwoPolicies - left.metrics.episodesWithAtLeastTwoPolicies ||
    right.metrics.admittedPolicies - left.metrics.admittedPolicies ||
    left.metrics.failedPolicyInductions - right.metrics.failedPolicyInductions ||
    right.metrics.projectedStepCoverage - left.metrics.projectedStepCoverage ||
    right.metrics.candidateSpanCoverage - left.metrics.candidateSpanCoverage ||
    left.metrics.clustersLargerThanSix - right.metrics.clustersLargerThanSix ||
    left.setting.id.localeCompare(right.setting.id);
}

function parseArgs(argv: readonly string[]): Args {
  let baselineArtifactPath: string | undefined;
  let dbPath = "";
  let configPath: string | undefined;
  let outputDbPath = "";
  let outputPath = "";
  let episodeIds: string[] = [];
  let specs = [...DEFAULT_SPECS];
  let span5CoarseSimilarityThreshold = 0.76;
  let span10CoarseSimilarityThreshold = 0.70;
  let coarseMembershipMode: Args["coarseMembershipMode"] = "multi";
  let minSupportEpisodes = 2;
  let concurrency = Math.max(1, Math.min(9, availableParallelism()));
  let matchCoverageAblation = false;
  let policyCoverageAblation = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--baseline") baselineArtifactPath = resolve(requiredValue(argv, ++index, arg));
    else if (arg === "--db") dbPath = resolve(requiredValue(argv, ++index, arg));
    else if (arg === "--config") configPath = resolve(requiredValue(argv, ++index, arg));
    else if (arg === "--output-db") outputDbPath = resolve(requiredValue(argv, ++index, arg));
    else if (arg === "--output") outputPath = resolve(requiredValue(argv, ++index, arg));
    else if (arg === "--episode-ids") {
      episodeIds = requiredValue(argv, ++index, arg).split(",")
        .map((value) => value.trim()).filter(Boolean);
    } else if (arg === "--span5-coarse-similarity") {
      span5CoarseSimilarityThreshold = unitInterval(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--span10-coarse-similarity") {
      span10CoarseSimilarityThreshold = unitInterval(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--coarse-membership") {
      const value = requiredValue(argv, ++index, arg);
      if (value !== "exclusive" && value !== "multi") {
        throw new Error("--coarse-membership must be exclusive or multi");
      }
      coarseMembershipMode = value;
    } else if (arg === "--match-coverage-ablation") {
      matchCoverageAblation = true;
    } else if (arg === "--policy-coverage-ablation") {
      policyCoverageAblation = true;
    } else if (arg === "--min-support") {
      minSupportEpisodes = positiveInt(requiredValue(argv, ++index, arg), arg);
      if (minSupportEpisodes < 2) throw new Error("--min-support must be >= 2");
    } else if (arg === "--concurrency") {
      concurrency = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write(helpText());
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }

  if (baselineArtifactPath) {
    const baseline = parseBaselineArtifact(baselineArtifactPath);
    if (!dbPath) dbPath = resolve(baseline.sourceDatabase);
    if (episodeIds.length === 0) episodeIds = baseline.episodeIds;
    if (baseline.result?.specs) specs = baseline.result.specs;
    span5CoarseSimilarityThreshold =
      baseline.result?.coarseSimilarityThresholdByScale?.["5"] ??
      span5CoarseSimilarityThreshold;
    span10CoarseSimilarityThreshold =
      baseline.result?.coarseSimilarityThresholdByScale?.["10"] ??
      span10CoarseSimilarityThreshold;
    coarseMembershipMode = baseline.result?.coarseMembershipMode ?? coarseMembershipMode;
    minSupportEpisodes = baseline.result?.minSupportEpisodes ?? minSupportEpisodes;
  }
  if (!dbPath) throw new Error("--db or --baseline is required");
  if (matchCoverageAblation && policyCoverageAblation) {
    throw new Error("choose only one ablation mode");
  }
  if (episodeIds.length < 2) throw new Error("at least two --episode-ids are required");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  const base = resolve("../../experiments/results/trace2skill-multi-scale-window-kw");
  return {
    ...(baselineArtifactPath ? { baselineArtifactPath } : {}),
    dbPath,
    ...(configPath ? { configPath } : {}),
    outputDbPath: outputDbPath || `${base}/${stamp}-kw-fine-sweep.sqlite`,
    outputPath: outputPath || `${base}/${stamp}-kw-fine-sweep.json`,
    episodeIds: [...new Set(episodeIds)],
    specs,
    span5CoarseSimilarityThreshold,
    span10CoarseSimilarityThreshold,
    coarseMembershipMode,
    minSupportEpisodes,
    concurrency,
    matchCoverageAblation,
    policyCoverageAblation
  };
}

function parseBaselineArtifact(path: string): BaselineArtifact {
  const value = JSON.parse(readFileSync(path, "utf8")) as Partial<BaselineArtifact>;
  if (typeof value.sourceDatabase !== "string" || !Array.isArray(value.episodeIds) ||
      !value.episodeIds.every((item) => typeof item === "string")) {
    throw new Error(`invalid baseline artifact: ${path}`);
  }
  return value as BaselineArtifact;
}

function renderMarkdown(artifact: {
  startedAt: string;
  finishedAt: string;
  elapsedMs: number;
  sourceDatabase: string;
  outputDatabase: string;
  preparation: Record<string, unknown>;
  provisionalShortlist: Array<{
    rank: number;
    settingId: string;
    metrics: SweepMetrics;
    finalQualityStatus: string;
  }>;
  runs: SweepRun[];
}): string {
  const rankedRuns = [...artifact.runs].sort(compareRuns);
  const lines = [
    "# KW Multi-Scale Span Fine-Parameter Sweep",
    "",
    `- Started: ${artifact.startedAt}`,
    `- Finished: ${artifact.finishedAt}`,
    `- Elapsed: ${Math.round(artifact.elapsedMs / 1000)}s`,
    `- Source DB: \`${artifact.sourceDatabase}\``,
    `- Output DB: \`${artifact.outputDatabase}\``,
    `- Preparation: \`${JSON.stringify(artifact.preparation)}\``,
    "- Final decision: **not approved; pending Policy and Skill semantic review**",
    "",
    "## Parallel Sweep Ranking",
    "",
    "| Rank | Setting | Candidates | Admitted Policies | Span coverage | Episodes ≥2 Policies | Repeated bigrams | Skill candidates | Admitted Skills | Max cluster |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ...rankedRuns.map((run, index) =>
      `| ${index + 1} | ${run.setting.id} | ${run.metrics.maximalPolicyCandidates} | ${run.metrics.admittedPolicies} | ${percent(run.metrics.candidateSpanCoverage)} | ${run.metrics.episodesWithAtLeastTwoPolicies} | ${run.metrics.repeatablePolicyBigrams} | ${run.metrics.skillCandidates} | ${run.metrics.admittedSkills} | ${run.metrics.maxClusterSpans} |`),
    "",
    "## Quality Gate",
    "",
    "The ranking above measures recall and downstream sequence feasibility only. It must not be used to change defaults until the shortlisted candidate Clusters are induced into Policies and the resulting Skill candidates pass semantic and evidence review.",
    "",
    "## Provisional Shortlist",
    "",
    ...artifact.provisionalShortlist.flatMap((item) => [
      `### ${item.rank}. ${item.settingId}`,
      "",
      "```json",
      JSON.stringify(item.metrics, null, 2),
      "```",
      "",
      `Status: ${item.finalQualityStatus}`,
      ""
    ])
  ];
  return `${lines.join("\n")}\n`;
}

function renderCsv(runs: readonly SweepRun[]): string {
  const header = [
    "rank",
    "setting",
    "supported_clusters",
    "policy_candidates",
    "admitted_policies",
    "rejected_policies",
    "failed_policy_inductions",
    "candidate_span_coverage",
    "episodes_with_2_policies",
    "projected_step_coverage",
    "repeatable_bigrams",
    "repeatable_trigrams",
    "skill_candidates",
    "admitted_skills",
    "rejected_skills",
    "failed_skill_inductions",
    "max_cluster_spans",
    "clusters_larger_than_6",
    "quality_status"
  ];
  const rows = [...runs].sort(compareRuns).map((run, index) => [
    index + 1,
    run.setting.id,
    run.metrics.supportedClusters,
    run.metrics.maximalPolicyCandidates,
    run.metrics.admittedPolicies,
    run.metrics.rejectedPolicies,
    run.metrics.failedPolicyInductions,
    run.metrics.candidateSpanCoverage,
    run.metrics.episodesWithAtLeastTwoPolicies,
    run.metrics.projectedStepCoverage,
    run.metrics.repeatablePolicyBigrams,
    run.metrics.repeatablePolicyTrigrams,
    run.metrics.skillCandidates,
    run.metrics.admittedSkills,
    run.metrics.rejectedSkills,
    run.metrics.failedSkillInductions,
    run.metrics.maxClusterSpans,
    run.metrics.clustersLargerThanSix,
    run.finalQualityStatus
  ].join(","));
  return `${header.join(",")}\n${rows.join("\n")}\n`;
}

function helpText(): string {
  return "Usage: npm run multi-scale-window-fine-sweep:kw -- --baseline <prior-result.json> [--config <config.yaml>] [--concurrency 9] [--match-coverage-ablation | --policy-coverage-ablation] [--output-db <copy.sqlite>] [--output <result.json>]\n";
}

async function cloneDatabase(sourcePath: string, outputPath: string): Promise<void> {
  const source = new Database(resolve(sourcePath), { readonly: true, fileMustExist: true });
  try {
    await source.backup(outputPath);
  } finally {
    source.close();
  }
}

function requiredValue(argv: readonly string[], index: number, option: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${option} requires a value`);
  return value;
}

function positiveInt(value: string, option: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`${option} must be positive`);
  return parsed;
}

function unitInterval(value: string, option: string): number {
  const parsed = Number(value);
  if (!(parsed > 0 && parsed <= 1)) throw new Error(`${option} must be in (0, 1]`);
  return parsed;
}

function rangesOverlap(
  leftStart: number,
  leftEnd: number,
  rightStart: number,
  rightEnd: number
): boolean {
  return leftStart <= rightEnd && rightStart <= leftEnd;
}

function ratio(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
