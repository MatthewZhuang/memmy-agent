import Database from "better-sqlite3";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import {
  MemoryDb,
  MemoryService,
  loadMemmyConfig,
  type MultiScaleWindowPolicyExperimentResultV1
} from "../src/index.js";

interface Args {
  dbPath: string;
  configPath?: string;
  outputDbPath: string;
  outputReportPath: string;
  episodeIds: string[];
  coarseSimilarityThreshold: number;
  span10CoarseSimilarityThreshold?: number;
  coarseMembershipMode: "exclusive" | "multi";
  minSupportEpisodes: number;
  maxPolicyCandidates: number;
  policyConcurrency: number;
  inducePolicies: boolean;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(dirname(args.outputDbPath), { recursive: true });
  mkdirSync(dirname(args.outputReportPath), { recursive: true });
  await cloneDatabase(args.dbPath, args.outputDbPath);
  const loaded = loadMemmyConfig(args.configPath);
  const db = new MemoryDb({ path: args.outputDbPath });
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();
  try {
    const service = new MemoryService({
      db,
      mode: "local",
      config: loaded.config,
      ...(args.configPath ? { configPath: args.configPath } : {})
    });
    const result = await service.discoverMultiScaleWindowPoliciesForReplay({
      episodeIds: args.episodeIds,
      specs: [
        { length: 5, stride: 2 },
        { length: 10, stride: 5 }
      ],
      coarseSimilarityThreshold: args.coarseSimilarityThreshold,
      ...(args.span10CoarseSimilarityThreshold === undefined
        ? {}
        : { coarseSimilarityThresholdByScale: { 10: args.span10CoarseSimilarityThreshold } }),
      coarseMembershipMode: args.coarseMembershipMode,
      minSupportEpisodes: args.minSupportEpisodes,
      maxPolicyCandidates: args.maxPolicyCandidates,
      policyConcurrency: args.policyConcurrency,
      inducePolicies: args.inducePolicies
    });
    const artifact = {
      schemaVersion: "multi-scale-window-policy-kw-replay.v1",
      startedAt,
      finishedAt: new Date().toISOString(),
      elapsedMs: Date.now() - wallStart,
      sourceDatabase: resolve(args.dbPath),
      outputDatabase: args.outputDbPath,
      episodeIds: args.episodeIds,
      summary: summarize(result),
      result
    };
    writeFileSync(args.outputReportPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
    const markdownPath = args.outputReportPath.replace(/\.json$/i, ".md");
    writeFileSync(markdownPath, renderMarkdown(artifact), "utf8");
    process.stdout.write(`${JSON.stringify({
      outputDatabase: args.outputDbPath,
      outputReport: args.outputReportPath,
      outputMarkdown: markdownPath,
      models: result.models,
      summary: artifact.summary,
      candidates: result.candidateClusterIds.map((clusterId) => {
        const cluster = result.clusters.find((item) => item.id === clusterId)!;
        const decision = result.decisions.find((item) => item.clusterId === clusterId);
        return {
          clusterId,
          familyId: cluster.familyId,
          scale: cluster.scale,
          medoidOccurrenceId: cluster.medoidOccurrenceId,
          medoidCentrality: cluster.medoidCentrality,
          medoidUpdateCount: cluster.medoidUpdateCount,
          supportEpisodes: cluster.supportEpisodeIds.length,
          occurrences: cluster.occurrenceCount,
          averageSimilarity: cluster.averageSimilarity,
          minimumSimilarity: cluster.minimumSimilarity,
          averagePairwiseSimilarity: cluster.averagePairwiseSimilarity,
          minimumPairwiseSimilarity: cluster.minimumPairwiseSimilarity,
          admitted: decision?.admitted,
          policyTitle: decision?.policy?.title,
          rejectionReason: decision?.rejectionReason,
          error: decision?.error?.split("\n")[0]
        };
      })
    }, null, 2)}\n`);
  } finally {
    db.close();
  }
}

function summarize(result: MultiScaleWindowPolicyExperimentResultV1) {
  const supported = result.clusters.filter((cluster) =>
    cluster.supportEpisodeIds.length >= result.minSupportEpisodes);
  const membershipCounts = new Map<string, number>();
  for (const family of result.families) {
    for (const occurrenceId of family.memberOccurrenceIds) {
      membershipCounts.set(occurrenceId, (membershipCounts.get(occurrenceId) ?? 0) + 1);
    }
  }
  return {
    episodes: result.episodeCount,
    steps: result.stepCount,
    embeddedSteps: result.embeddedStepCount,
    windows: result.windowCount,
    coarseMemberships: [...membershipCounts.values()].reduce((sum, count) => sum + count, 0),
    multiplyAssignedWindows: [...membershipCounts.values()].filter((count) => count > 1).length,
    maxMembershipsPerWindow: Math.max(0, ...membershipCounts.values()),
    coarseFamilies: result.families.length,
    coarseSpan5Families: result.families.filter((family) => family.scale === 5).length,
    coarseSpan10Families: result.families.filter((family) => family.scale === 10).length,
    clusters: result.clusters.length,
    supportedClusters: supported.length,
    supportedSpan5Clusters: supported.filter((cluster) => cluster.scale === 5).length,
    supportedSpan10Clusters: supported.filter((cluster) => cluster.scale === 10).length,
    maximalPolicyCandidates: result.candidateClusterIds.length,
    admittedPolicies: result.policies.length,
    rejectedCandidates: result.decisions.filter((decision) =>
      !decision.admitted && !decision.error).length,
    failedInductions: result.decisions.filter((decision) => decision.error).length
  };
}

function renderMarkdown(artifact: {
  startedAt: string;
  finishedAt: string;
  elapsedMs: number;
  sourceDatabase: string;
  outputDatabase: string;
  episodeIds: string[];
  summary: ReturnType<typeof summarize>;
  result: MultiScaleWindowPolicyExperimentResultV1;
}): string {
  const lines = [
    "# KW Multi-Scale Window → Policy 实验",
    "",
    `- Started: ${artifact.startedAt}`,
    `- Finished: ${artifact.finishedAt}`,
    `- Elapsed: ${Math.round(artifact.elapsedMs / 1000)}s`,
    `- Source DB: \`${artifact.sourceDatabase}\``,
    `- Output DB: \`${artifact.outputDatabase}\``,
    `- Episode count: ${artifact.episodeIds.length}`,
    `- Coarse similarity threshold: ${artifact.result.coarseSimilarityThreshold}`,
    `- Coarse thresholds by scale: ${Object.entries(artifact.result.coarseSimilarityThresholdByScale)
      .map(([scale, threshold]) => `Span-${scale}=${threshold}`).join(", ")}`,
    `- Coarse membership mode: ${artifact.result.coarseMembershipMode}`,
    `- Coarse representation: ${artifact.result.representations.coarse}`,
    `- Coarse clustering: ${artifact.result.coarseClusteringMethod}`,
    `- Fine representation: ${artifact.result.representations.fine}`,
    `- Specs: ${artifact.result.specs.map((spec) => `${spec.length}/${spec.stride}`).join(", ")}`,
    `- Fine configs: ${artifact.result.fineMatchConfigs.map((config) =>
      `Span-${config.scale}(band=${config.bandWidth},matches>=${config.minMatchedSteps},coverage>=${config.minCoverage},avg>=${config.minAverageMatchSimilarity})`
    ).join(", ")}`,
    "",
    "## Summary",
    "",
    "```json",
    JSON.stringify(artifact.summary, null, 2),
    "```",
    "",
    "## Candidate Clusters and Policy Decisions",
    ""
  ];
  for (const clusterId of artifact.result.candidateClusterIds) {
    const cluster = artifact.result.clusters.find((item) => item.id === clusterId)!;
    const family = artifact.result.families.find((item) => item.id === cluster.familyId)!;
    const decision = artifact.result.decisions.find((item) => item.clusterId === clusterId);
    lines.push(
      `### ${cluster.id}`,
      "",
      `- Scale: Span-${cluster.scale}`,
      `- Family: ${cluster.familyId}`,
      `- Family medoid: ${family.medoidOccurrenceId}`,
      `- Family medoid centrality / minimum / updates: ${family.medoidCentrality.toFixed(4)} / ${family.minimumSimilarityToMedoid.toFixed(4)} / ${family.medoidUpdateCount}`,
      `- Medoid: ${cluster.medoidOccurrenceId}`,
      `- Medoid centrality / updates: ${cluster.medoidCentrality.toFixed(4)} / ${cluster.medoidUpdateCount}`,
      `- Support Episodes: ${cluster.supportEpisodeIds.length}`,
      `- Occurrences: ${cluster.occurrenceCount}`,
      `- Similarity avg/min: ${cluster.averageSimilarity.toFixed(4)} / ${cluster.minimumSimilarity.toFixed(4)}`,
      `- Pairwise similarity avg/min: ${cluster.averagePairwiseSimilarity.toFixed(4)} / ${cluster.minimumPairwiseSimilarity.toFixed(4)}`,
      `- Decision: ${decision?.admitted ? "admitted" : decision?.error ? "failed" : decision ? "rejected" : "cluster-only"}`,
      ...(decision?.rejectionReason ? [`- Rejection: ${decision.rejectionReason}`] : []),
      ...(decision?.error ? [`- Error: ${decision.error.split("\n")[0]}`] : []),
      ""
    );
    if (decision?.policy) {
      const policy = decision.policy;
      lines.push(
        `#### Policy: ${policy.title}`,
        "",
        `- Goal: ${policy.goalPattern}`,
        `- Confidence: ${policy.confidence}`,
        `- Support: ${policy.supportEpisodeIds.join(", ")}`,
        "",
        "Procedure:",
        "",
        ...policy.procedureSteps.map((step, index) =>
          `${index + 1}. ${step.instruction}  \n   Evidence: ${step.evidenceRefs.join(", ")}`),
        "",
        "Verification:",
        "",
        ...policy.verificationSteps.map((step) =>
          `- ${step.check} → ${step.successSignal}  \n  Evidence: ${step.evidenceRefs.join(", ")}`),
        ""
      );
    }
    lines.push("#### Evidence Windows", "");
    for (const member of cluster.members) {
      const occurrence = member.occurrence;
      lines.push(
        `##### ${occurrence.episodeId} Steps ${occurrence.startStepIndex}–${occurrence.endStepIndex}`,
        "",
        `Alignment to medoid: score=${member.alignmentToMedoid.score.toFixed(4)}, avg=${member.alignmentToMedoid.averageMatchSimilarity.toFixed(4)}, coverage=${member.alignmentToMedoid.coverage.toFixed(4)}, matched=${member.alignmentToMedoid.matchedSteps}, internalGap=${member.alignmentToMedoid.internalGapSteps}`,
        "",
        `Matched positions: ${member.alignmentToMedoid.pairs.map((pair) =>
          `${pair.leftIndex + 1}↔${pair.rightIndex + 1}(${pair.similarity.toFixed(3)})`
        ).join(", ")}`,
        "",
        "| Step | Outcome | Intent | Summary |",
        "|---:|---|---|---|",
        ...occurrence.steps.map((step) =>
          `| ${step.stepIndex} | ${step.outcome.status} | ${escapeCell(step.action.intent)} | ${escapeCell(step.action.summary)} |`),
        ""
      );
    }
  }
  return `${lines.join("\n")}\n`;
}

async function cloneDatabase(sourcePath: string, outputPath: string): Promise<void> {
  const source = new Database(resolve(sourcePath), { readonly: true, fileMustExist: true });
  try {
    await source.backup(outputPath);
  } finally {
    source.close();
  }
}

function parseArgs(argv: readonly string[]): Args {
  let dbPath = "";
  let configPath: string | undefined;
  let outputDbPath = "";
  let outputReportPath = "";
  let episodeIds: string[] = [];
  let coarseSimilarityThreshold = 0.80;
  let span10CoarseSimilarityThreshold: number | undefined;
  let coarseMembershipMode: Args["coarseMembershipMode"] = "exclusive";
  let minSupportEpisodes = 2;
  let maxPolicyCandidates = 8;
  let policyConcurrency = 4;
  let inducePolicies = true;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--db") dbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--config") configPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output-db") outputDbPath = requiredValue(argv, ++index, arg);
    else if (arg === "--output") outputReportPath = requiredValue(argv, ++index, arg);
    else if (arg === "--episode-ids") {
      episodeIds = requiredValue(argv, ++index, arg).split(",")
        .map((value) => value.trim()).filter(Boolean);
    } else if (arg === "--coarse-similarity" || arg === "--similarity") {
      coarseSimilarityThreshold = Number(requiredValue(argv, ++index, arg));
    } else if (arg === "--span10-coarse-similarity") {
      span10CoarseSimilarityThreshold = Number(requiredValue(argv, ++index, arg));
    } else if (arg === "--coarse-membership") {
      const value = requiredValue(argv, ++index, arg);
      if (value !== "exclusive" && value !== "multi") {
        throw new Error("--coarse-membership must be exclusive or multi");
      }
      coarseMembershipMode = value;
    } else if (arg === "--min-support") {
      minSupportEpisodes = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--max-policies") {
      maxPolicyCandidates = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--policy-concurrency") {
      policyConcurrency = positiveInt(requiredValue(argv, ++index, arg), arg);
    } else if (arg === "--cluster-only") {
      inducePolicies = false;
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write(helpText());
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!dbPath) throw new Error("--db is required");
  if (episodeIds.length < 2) throw new Error("--episode-ids requires at least two IDs");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  const base = resolve("../../experiments/results/trace2skill-multi-scale-window-kw");
  return {
    dbPath: resolve(dbPath),
    ...(configPath ? { configPath: resolve(configPath) } : {}),
    outputDbPath: resolve(outputDbPath || `${base}/${stamp}-kw-multi-scale-window.sqlite`),
    outputReportPath: resolve(outputReportPath || `${base}/${stamp}-kw-multi-scale-window.json`),
    episodeIds: [...new Set(episodeIds)],
    coarseSimilarityThreshold,
    ...(span10CoarseSimilarityThreshold === undefined
      ? {}
      : { span10CoarseSimilarityThreshold }),
    coarseMembershipMode,
    minSupportEpisodes,
    maxPolicyCandidates,
    policyConcurrency,
    inducePolicies
  };
}

function helpText(): string {
  return "Usage: npm run multi-scale-window-policy:kw -- --db <kw.sqlite> --episode-ids <id1,id2,...> [--config <config.yaml>] [--coarse-similarity 0.80] [--span10-coarse-similarity 0.70] [--coarse-membership exclusive|multi] [--min-support 2] [--max-policies 8] [--policy-concurrency 4] [--cluster-only] [--output-db <copy.sqlite>] [--output <report.json>]\n";
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

function escapeCell(value: unknown): string {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
