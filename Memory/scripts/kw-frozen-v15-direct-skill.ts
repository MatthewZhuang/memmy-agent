import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import Database from "better-sqlite3";
import {
  MemoryDb,
  MemoryService,
  loadMemmyConfig,
  type LlmClient,
  type MemoryRow
} from "../src/index.js";
import {
  skillMetaFromMemory,
  traceMetaFromMemory
} from "../src/algorithm/plugin-algorithms.js";
import {
  ProceduralPatternSkillMaterializer,
  type ProceduralPatternSkillInput,
  type ProceduralSkillEvidenceOccurrence
} from "../src/service/evolution/procedural-pattern-skill.js";
import { Repositories } from "../src/storage/repositories.js";
import { stableHash } from "../src/utils/id.js";

interface Args {
  help: boolean;
  sourceDbPath?: string;
  v15JsonPath?: string;
  configPath?: string;
  outputDbPath?: string;
  outputReportPath?: string;
  confirmRemoteModels: boolean;
}

interface V15Step {
  id: string;
  rawTurnId: string;
  stepIndex: number;
  action: {
    toolName?: string;
    intent: string;
    summary: string;
    eventRefs?: string[];
  };
  outcome: {
    status: "success" | "failure" | "partial" | "unknown";
    evidenceRefs?: string[];
  };
}

interface V15Occurrence {
  id: string;
  episodeId: string;
  pathId: string;
  scale: number;
  startStepIndex: number;
  endStepIndex: number;
  steps: V15Step[];
}

interface V15AlignmentPair {
  leftIndex: number;
  rightIndex: number;
  similarity: number;
}

interface V15ClusterMember {
  occurrence: V15Occurrence;
  alignmentToMedoid: {
    admitted: boolean;
    score: number;
    averageMatchSimilarity: number;
    pairs: V15AlignmentPair[];
  };
}

interface V15Cluster {
  id: string;
  scale: number;
  medoidOccurrenceId: string;
  supportEpisodeIds: string[];
  averageSimilarity: number;
  members: V15ClusterMember[];
}

interface V15Policy {
  id: string;
  clusterId: string;
  title: string;
  goalPattern: string;
}

interface V15Artifact {
  schemaVersion: string;
  result: {
    policies: V15Policy[];
    clusters: V15Cluster[];
  };
}

interface ServiceInternals {
  skillLlm: LlmClient;
  buildMemory(input: Record<string, unknown>): MemoryRow;
  evolutionJobs: {
    upsertEvolutionMemory(memory: MemoryRow): {
      memory: MemoryRow;
      created: boolean;
      previous?: MemoryRow;
    };
  };
}

const EXPECTED_PLUGIN_ALGORITHM = "procedural.pattern.skill.v1";

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(helpText());
    return;
  }
  if (!args.confirmRemoteModels) {
    throw new Error(
      "This experiment sends frozen, clipped v15 Step semantics to the configured Skill LLM. " +
      "Re-run with --confirm-remote-models after checking the isolated output paths."
    );
  }

  const sourceDbPath = requiredPath(args.sourceDbPath, "--source-db");
  const v15JsonPath = requiredPath(args.v15JsonPath, "--v15-json");
  const outputDbPath = requiredPath(args.outputDbPath, "--output-db", false);
  const outputReportPath = requiredPath(args.outputReportPath, "--output-report", false);
  const configPath = args.configPath ? resolve(args.configPath) : undefined;
  if (sourceDbPath === outputDbPath) throw new Error("--source-db and --output-db must differ");
  if (existsSync(outputDbPath)) throw new Error(`Refusing to overwrite output DB: ${outputDbPath}`);
  if (existsSync(outputReportPath)) {
    throw new Error(`Refusing to overwrite output report: ${outputReportPath}`);
  }

  const artifact = JSON.parse(readFileSync(v15JsonPath, "utf8")) as V15Artifact;
  const policies = artifact.result.policies;
  const policyClusterIds = new Set(policies.map((policy) => policy.clusterId));
  const clusters = artifact.result.clusters.filter((cluster) => policyClusterIds.has(cluster.id));
  if (policies.length !== 4 || clusters.length !== 4) {
    throw new Error(`Expected exactly 4 admitted v15 Policies/clusters, got ${policies.length}/${clusters.length}`);
  }

  mkdirSync(dirname(outputDbPath), { recursive: true });
  mkdirSync(dirname(outputReportPath), { recursive: true });
  const sourceDb = new MemoryDb({ path: sourceDbPath, readonly: true });
  const targetDb = new MemoryDb({ path: outputDbPath });
  let service: MemoryService | undefined;
  const startedAt = new Date().toISOString();
  const wallStart = Date.now();
  try {
    const sourceRepos = new Repositories(sourceDb.db);
    const targetRepos = new Repositories(targetDb.db);
    const episodeIds = unique(clusters.flatMap((cluster) => cluster.supportEpisodeIds));
    const imported = importMinimalCohort({
      sourceDb: sourceDb.db,
      sourceRepos,
      targetDb: targetDb.db,
      targetRepos,
      episodeIds
    });

    const loaded = loadMemmyConfig(configPath);
    service = new MemoryService({
      db: targetDb,
      mode: "local",
      config: loaded.config,
      ...(configPath ? { configPath } : {})
    });
    const internals = service as unknown as ServiceInternals;
    const materializer = new ProceduralPatternSkillMaterializer({
      repos: targetRepos,
      config: loaded.config,
      skillLlm: internals.skillLlm,
      traceMeta(memory) {
        return memory ? traceMetaFromMemory(memory) : null;
      },
      buildMemory(input) {
        return internals.buildMemory.call(service, input);
      }
    });

    const policyByClusterId = new Map(policies.map((policy) => [policy.clusterId, policy]));
    const traceIdsByEpisode = new Map(imported.traces.map((trace) => [trace.episodeId, trace.id]));
    const results: Array<Record<string, unknown>> = [];
    for (const cluster of clusters) {
      const frozen = freezeInput(cluster, traceIdsByEpisode, imported.userId);
      const policy = policyByClusterId.get(cluster.id);
      let compileResult;
      try {
        compileResult = await materializer.compile(frozen.input);
      } catch (error) {
        results.push({
          clusterId: cluster.id,
          policyId: policy?.id,
          policyTitle: policy?.title,
          admitted: false,
          failureStage: "compile-exception",
          reason: errorMessage(error),
          frozenEvidence: frozen.audit
        });
        continue;
      }
      if (!compileResult.admitted) {
        results.push({
          clusterId: cluster.id,
          policyId: policy?.id,
          policyTitle: policy?.title,
          admitted: false,
          failureStage: "compile-rejection",
          reason: compileResult.reason,
          frozenEvidence: frozen.audit
        });
        continue;
      }

      const materialized = materializer.materializeDraft(
        compileResult.draft,
        new Date().toISOString()
      );
      const upsert = internals.evolutionJobs.upsertEvolutionMemory(materialized.memory);
      const memory = upsert.memory;
      const meta = skillMetaFromMemory(memory);
      const internal = memory.properties.internal_info;
      const procedureJson = record(internal.procedure_json);
      const skill = record(internal.skill);
      const schemaChecks = {
        memoryLayerSkill: memory.memoryLayer === "Skill",
        pluginAlgorithm: internal.plugin_algorithm === EXPECTED_PLUGIN_ALGORITHM,
        canonicalSkillMeta: Boolean(meta),
        sourcePolicyIdsEmpty: array(internal.source_policy_ids).length === 0 &&
          array(skill.source_policy_ids).length === 0,
        sourceClusterMatches: internal.source_cluster_id === cluster.id,
        sourceEpisodesMatch: sameSet(
          stringArray(internal.source_episode_ids),
          cluster.supportEpisodeIds
        ),
        evidenceOccurrencesCoverTwoEpisodes: stringArray(
          record(skill.verification).evidence_occurrence_ids
        ).length >= 2,
        hasProcedureSteps: array(procedureJson.steps).length > 0
      };
      results.push({
        clusterId: cluster.id,
        policyId: policy?.id,
        policyTitle: policy?.title,
        admitted: true,
        failureStage: null,
        reason: null,
        frozenEvidence: frozen.audit,
        materialized: {
          memoryId: memory.id,
          memoryKey: memory.memoryKey,
          memoryStatus: memory.status,
          lifecycleStatus: meta?.status,
          name: meta?.name,
          support: meta?.support,
          gain: meta?.gain,
          confidence: internal.confidence,
          sourceMemoryIds: stringArray(internal.source_memory_ids),
          sourceEpisodeIds: stringArray(internal.source_episode_ids),
          sourcePolicyIds: stringArray(internal.source_policy_ids),
          sourceOccurrenceIds: stringArray(internal.source_span_occurrence_ids),
          procedureStepCount: array(procedureJson.steps).length,
          verificationEvidenceOccurrenceIds: stringArray(
            record(skill.verification).evidence_occurrence_ids
          ),
          pluginAlgorithm: internal.plugin_algorithm,
          schemaChecks,
          allSchemaChecksPassed: Object.values(schemaChecks).every(Boolean),
          created: upsert.created
        }
      });
    }

    const storedSkills = targetRepos.memories.list({
      userId: imported.userId,
      layers: ["Skill"],
      includeDeleted: true,
      limit: 100
    }).filter((memory) =>
      memory.properties.internal_info.plugin_algorithm === EXPECTED_PLUGIN_ALGORITHM
    );
    const report = {
      schemaVersion: "kw-frozen-v15-direct-skill-experiment.v1",
      startedAt,
      finishedAt: new Date().toISOString(),
      elapsedMs: Date.now() - wallStart,
      isolation: {
        sourceDbPath,
        v15JsonPath,
        outputDbPath,
        outputReportPath,
        sourceDatabaseReadOnly: true,
        productionClusteringBypassed: true,
        frozenInputs: true
      },
      models: service.health().models,
      imported: {
        userId: imported.userId,
        episodeIds,
        traceIds: imported.traces.map((trace) => trace.id),
        episodeCount: episodeIds.length,
        traceCount: imported.traces.length
      },
      input: {
        v15SchemaVersion: artifact.schemaVersion,
        admittedPolicyCount: policies.length,
        frozenClusterCount: clusters.length
      },
      summary: {
        clusters: results.length,
        admitted: results.filter((item) => item.admitted === true).length,
        rejected: results.filter((item) => item.failureStage === "compile-rejection").length,
        exceptions: results.filter((item) => item.failureStage === "compile-exception").length,
        storedProceduralSkills: storedSkills.length
      },
      results
    };
    writeFileSync(outputReportPath, `${JSON.stringify(report, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o600
    });
    process.stdout.write(`${JSON.stringify({
      outputDatabase: outputDbPath,
      outputReport: outputReportPath,
      summary: report.summary,
      skills: results.map((item) => ({
        clusterId: item.clusterId,
        admitted: item.admitted,
        reason: item.reason,
        memoryId: record(item.materialized).memoryId,
        name: record(item.materialized).name
      }))
    }, null, 2)}\n`);
  } finally {
    void service;
    targetDb.close();
    sourceDb.close();
  }
}

function freezeInput(
  cluster: V15Cluster,
  traceIdsByEpisode: Map<string, string>,
  userId: string
): {
  input: ProceduralPatternSkillInput;
  audit: Record<string, unknown>;
} {
  const medoid = cluster.members.find((member) =>
    member.occurrence.id === cluster.medoidOccurrenceId
  );
  if (!medoid) throw new Error(`v15 cluster medoid missing: ${cluster.id}`);
  const supportMembers = cluster.members.filter((member) =>
    cluster.supportEpisodeIds.includes(member.occurrence.episodeId)
  );
  const episodesByAnchor = new Map<number, Set<string>>();
  for (const member of supportMembers) {
    for (const pair of member.alignmentToMedoid.pairs) {
      const episodes = episodesByAnchor.get(pair.rightIndex) ?? new Set<string>();
      episodes.add(member.occurrence.episodeId);
      episodesByAnchor.set(pair.rightIndex, episodes);
    }
  }
  const anchorOffsets = [...episodesByAnchor.entries()]
    .filter(([, episodes]) => episodes.size >= 2)
    .map(([offset]) => offset)
    .sort((left, right) => left - right);
  if (anchorOffsets.length === 0) throw new Error(`v15 cluster has no aligned core: ${cluster.id}`);
  const anchorSet = new Set(anchorOffsets);

  const evidence: ProceduralSkillEvidenceOccurrence[] = supportMembers.map((member) => {
    const selectedPairs = member.alignmentToMedoid.pairs
      .filter((pair) => anchorSet.has(pair.rightIndex))
      .sort((left, right) => left.leftIndex - right.leftIndex);
    if (selectedPairs.length !== anchorOffsets.length) {
      throw new Error(`v15 member does not cover the complete aligned core: ${member.occurrence.id}`);
    }
    const sourceTraceId = traceIdsByEpisode.get(member.occurrence.episodeId);
    if (!sourceTraceId) {
      throw new Error(`source L1 trace missing for Episode: ${member.occurrence.episodeId}`);
    }
    const steps = selectedPairs.map((pair) => {
      const step = member.occurrence.steps[pair.leftIndex];
      if (!step) throw new Error(`aligned Step is outside occurrence: ${member.occurrence.id}`);
      return {
        stepId: step.id,
        stepIndex: step.stepIndex,
        ...(step.action.toolName ? { toolName: step.action.toolName } : {}),
        intent: step.action.intent,
        summary: step.action.summary,
        outcome: step.outcome.status,
        evidenceRefs: unique([
          ...(step.outcome.evidenceRefs ?? []),
          ...(step.action.eventRefs ?? [])
        ])
      };
    });
    return {
      occurrenceId: member.occurrence.id,
      episodeId: member.occurrence.episodeId,
      pathId: member.occurrence.pathId,
      scale: member.occurrence.scale,
      alignmentScore: average(selectedPairs.map((pair) => pair.similarity)),
      sourceTraceIds: [sourceTraceId],
      steps
    };
  });
  const sourceTraceIds = unique(evidence.flatMap((item) => item.sourceTraceIds));
  const patternHash = stableHash({
    experiment: "frozen-v15-direct-skill.v1",
    clusterId: cluster.id,
    medoidOccurrenceId: cluster.medoidOccurrenceId,
    anchorOffsets,
    evidence
  });
  const input: ProceduralPatternSkillInput = {
    patternVersionId: `frozen_v15_pattern_${patternHash.slice(0, 24)}`,
    clusterId: cluster.id,
    clusterVersionId: `frozen_v15_cluster_version_${stableHash({
      clusterId: cluster.id,
      evidence: evidence.map((item) => item.occurrenceId)
    }).slice(0, 24)}`,
    userId,
    scale: cluster.scale,
    supportEpisodeIds: cluster.supportEpisodeIds,
    sourceTraceIds,
    sourceSpanOccurrenceIds: evidence.map((item) => item.occurrenceId),
    counterexampleEpisodeIds: [],
    evidence,
    confidenceHint: clamp(cluster.averageSimilarity, 0, 1),
    patternHash,
    algorithmVersion: "frozen-v15-direct-materializer-experiment.v1"
  };
  return {
    input,
    audit: {
      scale: cluster.scale,
      medoidOccurrenceId: cluster.medoidOccurrenceId,
      anchorOffsets,
      supportEpisodeIds: cluster.supportEpisodeIds,
      occurrences: evidence.map((item) => ({
        occurrenceId: item.occurrenceId,
        episodeId: item.episodeId,
        alignmentScore: item.alignmentScore,
        sourceTraceIds: item.sourceTraceIds,
        steps: item.steps.map((step) => ({
          stepId: step.stepId,
          stepIndex: step.stepIndex,
          toolName: step.toolName,
          intent: step.intent,
          summary: step.summary,
          outcome: step.outcome,
          evidenceRefs: step.evidenceRefs
        }))
      }))
    }
  };
}

function importMinimalCohort(input: {
  sourceDb: Database.Database;
  sourceRepos: Repositories;
  targetDb: Database.Database;
  targetRepos: Repositories;
  episodeIds: string[];
}): {
  userId: string;
  traces: Array<{ id: string; episodeId: string }>;
} {
  const placeholders = input.episodeIds.map(() => "?").join(",");
  const episodes = input.sourceDb.prepare(
    `SELECT * FROM episodes WHERE id IN (${placeholders})`
  ).all(...input.episodeIds) as Array<Record<string, unknown>>;
  if (episodes.length !== input.episodeIds.length) {
    throw new Error(`source DB has ${episodes.length}/${input.episodeIds.length} required Episodes`);
  }
  const userIds = unique(episodes.map((row) => text(row.user_id)).filter(isString));
  if (userIds.length !== 1) throw new Error("frozen clusters must share one user");
  const sessionIds = unique(episodes.map((row) => text(row.session_id)).filter(isString));
  const sessions = selectRows(input.sourceDb, "sessions", "id", sessionIds);
  const rawTurns = selectRows(input.sourceDb, "raw_turns", "episode_id", input.episodeIds);
  const traceIds = unique(episodes.flatMap((episode) => parseStringArray(episode.l1_memory_ids_json)));
  const traceMemories = input.sourceRepos.memories.getMany(traceIds).filter((memory) =>
    memory.memoryLayer === "L1" && memory.status === "activated"
  );
  const traceEpisodePairs = traceMemories.flatMap((memory) => {
    const trace = traceMetaFromMemory(memory);
    return trace?.episodeId && input.episodeIds.includes(trace.episodeId)
      ? [{ id: memory.id, episodeId: trace.episodeId }]
      : [];
  });
  for (const episodeId of input.episodeIds) {
    if (!traceEpisodePairs.some((item) => item.episodeId === episodeId)) {
      throw new Error(`source DB has no active L1 trace for Episode: ${episodeId}`);
    }
  }
  input.targetDb.transaction(() => {
    insertRows(input.targetDb, "sessions", sessions);
    insertRows(input.targetDb, "episodes", episodes);
    insertRows(input.targetDb, "raw_turns", rawTurns);
    for (const memory of traceMemories) input.targetRepos.memories.insert(memory);
  })();
  return { userId: userIds[0]!, traces: traceEpisodePairs };
}

function selectRows(
  db: Database.Database,
  table: "sessions" | "raw_turns",
  column: string,
  values: string[]
): Array<Record<string, unknown>> {
  if (values.length === 0) return [];
  const placeholders = values.map(() => "?").join(",");
  return db.prepare(`SELECT * FROM ${table} WHERE ${column} IN (${placeholders})`)
    .all(...values) as Array<Record<string, unknown>>;
}

function insertRows(
  db: Database.Database,
  table: "sessions" | "episodes" | "raw_turns",
  rows: Array<Record<string, unknown>>
): void {
  if (rows.length === 0) return;
  const columns = Object.keys(rows[0]!);
  const statement = db.prepare(
    `INSERT INTO ${table} (${columns.join(",")}) VALUES (${columns.map(() => "?").join(",")})`
  );
  for (const row of rows) statement.run(...columns.map((column) => row[column]));
}

function parseArgs(values: string[]): Args {
  const args: Args = { help: false, confirmRemoteModels: false };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (value === "--help" || value === "-h") args.help = true;
    else if (value === "--confirm-remote-models") args.confirmRemoteModels = true;
    else if (value === "--source-db") args.sourceDbPath = values[++index];
    else if (value === "--v15-json") args.v15JsonPath = values[++index];
    else if (value === "--config") args.configPath = values[++index];
    else if (value === "--output-db") args.outputDbPath = values[++index];
    else if (value === "--output-report") args.outputReportPath = values[++index];
    else throw new Error(`Unknown argument: ${value}`);
  }
  return args;
}

function requiredPath(value: string | undefined, flag: string, mustExist = true): string {
  if (!value) throw new Error(`${flag} is required`);
  const path = resolve(value);
  if (mustExist && !existsSync(path)) throw new Error(`${flag} not found: ${path}`);
  return path;
}

function helpText(): string {
  return `Usage: node --import tsx scripts/kw-frozen-v15-direct-skill.ts \\
  --source-db <current isolated KW DB> \\
  --v15-json <v15 Policy JSON> \\
  --output-db <new isolated SQLite path> \\
  --output-report <new JSON report path> \\
  [--config <Memmy config>] \\
  --confirm-remote-models\n`;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter(isString) : [];
}

function parseStringArray(value: unknown): string[] {
  if (typeof value !== "string") return [];
  try {
    return stringArray(JSON.parse(value));
  } catch {
    return [];
  }
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function isString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function sameSet(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((item) => right.includes(item));
}

function average(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

void main().catch((error) => {
  process.stderr.write(`${errorMessage(error)}\n`);
  process.exitCode = 1;
});
