import Database from "better-sqlite3";
import {
  mkdirSync,
  mkdtempSync,
  renameSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, resolve } from "node:path";
import {
  MemoryDb,
  MemoryService,
  inspectTrace2SkillEpisode,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type Trace2SkillDiagnosticReportV1,
  type Trace2SkillReplayResultV1
} from "../src/index.js";
import { Repositories } from "../src/storage/repositories.js";
import { redactSensitiveText } from "../src/utils/sensitive-data.js";
import { clip } from "../src/utils/text.js";

type ReplayMode = "inspect" | "dry-run" | "commit";
type ReplayStart = "raw_episode" | "active_procedural_path";

interface ReplayArgs {
  episodeIds: string[];
  mode: ReplayMode;
  dbPath?: string;
  configPath?: string;
  outputPath?: string;
  confirmWrite: boolean;
  replayStart: ReplayStart;
}

interface SkillValidation {
  skillId: string;
  readable: boolean;
  listed: boolean;
  vectorReady: boolean;
}

interface EpisodeReplayArtifact {
  episodeId: string;
  diagnostic?: Trace2SkillDiagnosticReportV1;
  replay?: Trace2SkillReplayResultV1;
  rawReconstruction?: {
    pathId: string;
    pathHash: string;
    created: boolean;
  };
  skillValidation: SkillValidation[];
  pipelineHealthy: boolean;
  skillEligible: boolean;
  skillCompiled: boolean;
  decision: "compiled" | "reused" | "correctly_abstained" | "blocked";
  effective: boolean;
  effectiveReasons: string[];
  decisionReasons: string[];
}

interface ReplayArtifact {
  schemaVersion: "trace2skill-replay-artifact.v1";
  generatedAt: string;
  mode: ReplayMode;
  replayStart: ReplayStart;
  database: {
    sourcePath: string;
    writesAppliedToSource: boolean;
    sourceReadViaConsistentClone: boolean;
  };
  privacy: {
    rawTurnsPersisted: false;
    internalReasoningPersisted: false;
    credentialsPersisted: false;
    artifactText: "redacted-and-clipped";
  };
  models: {
    evolution: { provider: string; model?: string; endpoint?: string };
    embedding: { provider: string; model?: string; endpoint?: string };
  };
  episodes: EpisodeReplayArtifact[];
  summary: {
    requested: number;
    effective: number;
    ineffective: number;
    pipelineHealthy: number;
    skillEligible: number;
    skillCompiled: number;
    correctlyAbstained: number;
    blocked: number;
    compiledSkillIds: string[];
  };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const loaded = loadMemmyConfig(args.configPath);
  const configuredDbPath = args.dbPath ?? loaded.config.storage.sqlitePath;
  if (!configuredDbPath) {
    throw new Error("No SQLite database path. Pass --db or configure storage.sqlitePath.");
  }
  const sourceDbPath = resolve(configuredDbPath);
  if (args.mode === "commit" && !args.confirmWrite) {
    throw new Error("commit mode requires --confirm-write");
  }
  if (args.mode === "commit" && args.replayStart === "raw_episode") {
    throw new Error(
      "raw Episode reconstruction is intentionally dry-run only; use --mode dry-run --from raw"
    );
  }

  const tempDir = args.mode === "dry-run"
    ? mkdtempSync(resolve(tmpdir(), "memmy-trace2skill-replay-"))
    : undefined;
  const workingDbPath = tempDir ? resolve(tempDir, "memory.sqlite") : sourceDbPath;
  try {
    if (tempDir) {
      await cloneSqliteDatabase(sourceDbPath, workingDbPath, args.episodeIds);
    }
    const episodes = args.mode === "inspect"
      ? inspectEpisodes(workingDbPath, args.episodeIds)
      : await replayEpisodes({
          dbPath: workingDbPath,
          configPath: args.configPath,
          episodeIds: args.episodeIds,
          config: loaded.config,
          replayStart: args.replayStart
        });
    const compiledSkillIds = unique(episodes.flatMap((episode) =>
      episode.replay?.actions.compiledSkillIds ?? []
    ));
    const evolution = resolveEvolutionConfig(loaded.config);
    const artifact: ReplayArtifact = {
      schemaVersion: "trace2skill-replay-artifact.v1",
      generatedAt: new Date().toISOString(),
      mode: args.mode,
      replayStart: args.replayStart,
      database: {
        sourcePath: sourceDbPath,
        writesAppliedToSource: args.mode === "commit",
        sourceReadViaConsistentClone: Boolean(tempDir)
      },
      privacy: {
        rawTurnsPersisted: false,
        internalReasoningPersisted: false,
        credentialsPersisted: false,
        artifactText: "redacted-and-clipped"
      },
      models: {
        evolution: publicModelConfig(evolution),
        embedding: publicModelConfig(loaded.config.embedding)
      },
      episodes,
      summary: {
        requested: episodes.length,
        effective: episodes.filter((episode) => episode.effective).length,
        ineffective: episodes.filter((episode) => !episode.effective).length,
        pipelineHealthy: episodes.filter((episode) => episode.pipelineHealthy).length,
        skillEligible: episodes.filter((episode) => episode.skillEligible).length,
        skillCompiled: episodes.filter((episode) => episode.skillCompiled).length,
        correctlyAbstained: episodes.filter((episode) =>
          episode.decision === "correctly_abstained"
        ).length,
        blocked: episodes.filter((episode) => episode.decision === "blocked").length,
        compiledSkillIds
      }
    };
    const safeArtifact = sanitizeArtifact(artifact) as ReplayArtifact;
    const persisted = args.outputPath
      ? persistArtifact(args.outputPath, safeArtifact)
      : undefined;
    process.stdout.write(`${JSON.stringify({
      mode: artifact.mode,
      replayStart: artifact.replayStart,
      summary: artifact.summary,
      episodes: artifact.episodes.map((episode) => ({
        episodeId: episode.episodeId,
        pipelineHealthy: episode.pipelineHealthy,
        skillEligible: episode.skillEligible,
        skillCompiled: episode.skillCompiled,
        decision: episode.decision,
        effective: episode.effective,
        reasons: episode.effectiveReasons,
        decisionReasons: episode.decisionReasons,
        blockers: (episode.replay?.after ?? episode.diagnostic)?.blockers ?? [],
        compiledSkillIds: episode.replay?.actions.compiledSkillIds ?? []
      })),
      ...(persisted ? { persistedFiles: persisted } : {})
    }, null, 2)}\n`);
  } finally {
    if (tempDir) rmSync(tempDir, { recursive: true, force: true });
  }
}

function inspectEpisodes(dbPath: string, episodeIds: readonly string[]): EpisodeReplayArtifact[] {
  const db = new MemoryDb({ path: dbPath, readonly: true });
  try {
    const repos = new Repositories(db.db);
    return episodeIds.map((episodeId) => {
      const diagnostic = inspectTrace2SkillEpisode(repos, episodeId);
      const effectiveReasons = pipelineHealthReasons(diagnostic, []);
      const decision = replayDecision(diagnostic, [], [], effectiveReasons);
      return {
        episodeId,
        diagnostic,
        skillValidation: [],
        pipelineHealthy: effectiveReasons.length === 0,
        skillEligible: diagnostic.checks.readyCandidate,
        skillCompiled: diagnostic.checks.compiledSkill,
        decision,
        effective: effectiveReasons.length === 0,
        effectiveReasons,
        decisionReasons: decisionReasons(diagnostic)
      };
    });
  } finally {
    db.close();
  }
}

async function replayEpisodes(input: {
  dbPath: string;
  configPath?: string;
  episodeIds: readonly string[];
  config: ReturnType<typeof loadMemmyConfig>["config"];
  replayStart: ReplayStart;
}): Promise<EpisodeReplayArtifact[]> {
  const db = new MemoryDb({ path: input.dbPath });
  try {
    const service = new MemoryService({
      db,
      mode: "local",
      config: input.config,
      ...(input.configPath ? { configPath: input.configPath } : {})
    });
    const repos = new Repositories(db.db);
    const results: EpisodeReplayArtifact[] = [];
    const rawReconstruction = new Map<string, EpisodeReplayArtifact["rawReconstruction"]>();
    if (input.replayStart === "raw_episode") {
      for (const episodeId of input.episodeIds) {
        const reconstructed = await service.reconstructProceduralPathForReplay({ episodeId });
        rawReconstruction.set(episodeId, {
          pathId: reconstructed.record.id,
          pathHash: reconstructed.record.pathHash,
          created: reconstructed.created
        });
      }
      await drainEvolutionJobs(service);
    }
    for (const episodeId of input.episodeIds) {
      const replay = await service.replayTrace2SkillEpisode({ episodeId });
      const skillIds = unique([
        ...replay.actions.compiledSkillIds,
        ...replay.actions.reusedSkillIds
      ]);
      await drainTargetedEmbeddingJobs(service, skillIds);
      replay.after = service.inspectTrace2SkillEpisode(episodeId);
      const skillValidation = skillIds.map((skillId) => validateSkill(
        service,
        repos,
        replay.after.episode.userId,
        skillId
      ));
      const effectiveReasons = pipelineHealthReasons(replay.after, skillValidation);
      const decision = replayDecision(
        replay.after,
        replay.actions.compiledSkillIds,
        replay.actions.reusedSkillIds,
        effectiveReasons
      );
      results.push({
        episodeId,
        replay,
        ...(rawReconstruction.get(episodeId)
          ? { rawReconstruction: rawReconstruction.get(episodeId) }
          : {}),
        skillValidation,
        pipelineHealthy: effectiveReasons.length === 0,
        skillEligible: replay.after.checks.readyCandidate,
        skillCompiled: replay.after.checks.compiledSkill,
        decision,
        effective: effectiveReasons.length === 0,
        effectiveReasons,
        decisionReasons: decisionReasons(replay.after)
      });
    }
    for (const item of results) {
      if (!item.replay) continue;
      item.replay.after = service.inspectTrace2SkillEpisode(item.episodeId);
      const finalSkillIds = unique(item.replay.after.patterns.flatMap((pattern) =>
        pattern.skill?.executable ? [pattern.skill.id] : []
      ));
      item.skillValidation = finalSkillIds.map((skillId) => validateSkill(
        service,
        repos,
        item.replay!.after.episode.userId,
        skillId
      ));
      item.effectiveReasons = pipelineHealthReasons(item.replay.after, item.skillValidation);
      item.pipelineHealthy = item.effectiveReasons.length === 0;
      item.skillEligible = item.replay.after.checks.readyCandidate;
      item.skillCompiled = item.replay.after.checks.compiledSkill;
      item.decision = replayDecision(
        item.replay.after,
        item.replay.actions.compiledSkillIds,
        item.replay.actions.reusedSkillIds,
        item.effectiveReasons
      );
      item.decisionReasons = decisionReasons(item.replay.after);
      item.effective = item.effectiveReasons.length === 0;
    }
    return results;
  } finally {
    db.close();
  }
}

async function drainEvolutionJobs(service: MemoryService): Promise<void> {
  for (let round = 0; round < 50; round += 1) {
    const result = await service.runWorkerOnce(100);
    if (result.leased === 0 && result.embeddingRetries.leased === 0) return;
  }
  throw new Error("Trace2Skill raw replay did not drain its worker queue after 50 rounds");
}

async function drainTargetedEmbeddingJobs(
  service: MemoryService,
  skillIds: readonly string[]
): Promise<void> {
  if (skillIds.length === 0) return;
  for (let round = 0; round < 5; round += 1) {
    const result = await service.runWorkerOnce(Math.max(10, skillIds.length), {
      targetMemoryIds: [...skillIds]
    });
    if (result.leased === 0 && result.embeddingRetries.leased === 0) return;
  }
}

function validateSkill(
  service: MemoryService,
  repos: Repositories,
  userId: string,
  skillId: string
): SkillValidation {
  let readable = false;
  try {
    readable = service.getSkill(skillId).id === skillId;
  } catch {
    readable = false;
  }
  return {
    skillId,
    readable,
    listed: service.listSkills({ userId }).skills.some((skill) => skill.id === skillId),
    vectorReady: repos.memories.hasVector(skillId, "vec")
  };
}

function pipelineHealthReasons(
  report: Trace2SkillDiagnosticReportV1,
  skills: readonly SkillValidation[]
): string[] {
  const reasons: string[] = [];
  if (!report.checks.episodeClosed) reasons.push("Episode is not closed");
  if (!report.checks.activePath) reasons.push("active procedural path is missing");
  if (!report.checks.pathContinuous) reasons.push("procedural path is discontinuous");
  if (report.checks.spanCreditCoverage < 1) reasons.push("SpanCredit coverage is incomplete");
  if (!report.checks.capabilitySignature) {
    reasons.push("Episode Capability Signature was not materialized");
  }
  for (const skill of skills) {
    if (!skill.readable) reasons.push(`Skill ${skill.skillId} is not readable`);
    if (!skill.listed) reasons.push(`Skill ${skill.skillId} is not listed`);
    if (!skill.vectorReady) reasons.push(`Skill ${skill.skillId} has no retrieval vector`);
  }
  return unique(reasons);
}

function replayDecision(
  report: Trace2SkillDiagnosticReportV1,
  compiledSkillIds: readonly string[],
  reusedSkillIds: readonly string[],
  healthReasons: readonly string[]
): EpisodeReplayArtifact["decision"] {
  if (healthReasons.length > 0) return "blocked";
  if (compiledSkillIds.length > 0) return "compiled";
  if (reusedSkillIds.length > 0 || report.checks.compiledSkill) return "reused";
  if (report.checks.readyCandidate) return "blocked";
  return "correctly_abstained";
}

function decisionReasons(report: Trace2SkillDiagnosticReportV1): string[] {
  const reasons = [...report.blockers];
  if (!report.checks.episodeSimilarityRoute && !report.checks.policySequenceRoute) {
    reasons.push("neither discovery route produced an Episode occurrence");
  }
  return unique(reasons);
}

async function cloneSqliteDatabase(
  sourcePath: string,
  targetPath: string,
  episodeIds: readonly string[]
): Promise<void> {
  mkdirSync(dirname(targetPath), { recursive: true });
  const fresh = new MemoryDb({ path: targetPath });
  fresh.close();
  const source = new Database(sourcePath, { readonly: true, fileMustExist: true });
  const target = new Database(targetPath);
  try {
    source.pragma("busy_timeout = 30000");
    source.exec("BEGIN");
    source.prepare("SELECT COUNT(*) AS count FROM episodes").get();
    target.pragma("foreign_keys = OFF");
    target.transaction(() => {
      const seedPathRows = selectRowsByValues(
        source,
        "episode_procedural_paths",
        "episode_id",
        episodeIds
      );
      const namespaceIds = stringColumn(seedPathRows, "namespace_id");
      const pathRows = namespaceIds.length > 0
        ? selectRowsByValues(
            source,
            "episode_procedural_paths",
            "namespace_id",
            namespaceIds
          )
        : seedPathRows;
      const scopeEpisodeIds = unique([
        ...episodeIds,
        ...stringColumn(pathRows, "episode_id")
      ]);
      const episodeRows = selectRowsByValues(source, "episodes", "id", scopeEpisodeIds);
      copyRows(target, "episodes", episodeRows);
      const sessionIds = unique(episodeRows.flatMap((row) =>
        typeof row.session_id === "string" ? [row.session_id] : []
      ));
      copyRows(target, "sessions", selectRowsByValues(source, "sessions", "id", sessionIds));
      copyRows(target, "raw_turns", selectRowsByValues(
        source,
        "raw_turns",
        "episode_id",
        scopeEpisodeIds
      ));
      copyRows(target, "episode_procedural_paths", pathRows);
      const pathIds = stringColumn(pathRows, "id");
      const spanRows = selectRowsByValues(
        source,
        "procedural_span_occurrences",
        "path_id",
        pathIds
      );
      copyRows(target, "procedural_span_occurrences", spanRows);
      const occurrenceIds = stringColumn(spanRows, "id");
      const creditRunRows = selectRowsByValues(
        source,
        "episode_span_credit_runs",
        "episode_id",
        scopeEpisodeIds
      );
      copyRows(target, "episode_span_credit_runs", creditRunRows);
      copyRows(target, "procedural_span_credits", selectRowsByValues(
        source,
        "procedural_span_credits",
        "run_id",
        stringColumn(creditRunRows, "id")
      ));
      const clusterMemberRows = selectRowsByValues(
        source,
        "procedural_span_cluster_members",
        "occurrence_id",
        occurrenceIds
      );
      copyRows(target, "procedural_span_cluster_members", clusterMemberRows);
      copyRows(target, "procedural_span_clusters", selectRowsByValues(
        source,
        "procedural_span_clusters",
        "id",
        stringColumn(clusterMemberRows, "cluster_id")
      ));
      const policyOccurrenceRows = selectRowsByValues(
        source,
        "procedural_policy_occurrences",
        "path_id",
        pathIds
      );
      copyRows(target, "procedural_policy_occurrences", policyOccurrenceRows);
      const policyRows = selectRowsByValues(
        source,
        "procedural_policy_versions",
        "id",
        stringColumn(policyOccurrenceRows, "policy_version_id")
      );
      copyRows(target, "procedural_policy_versions", policyRows);
      const memoryIds = stringColumn(policyRows, "l2_memory_id");
      copyRows(target, "memories", selectRowsByValues(source, "memories", "id", memoryIds));
      if (sourceTableExists(source, "memory_processing_state")) {
        copyRows(target, "memory_processing_state", selectRowsByValues(
          source,
          "memory_processing_state",
          "memory_id",
          memoryIds
        ));
      }
      const projectionRows = selectRowsByValues(
        source,
        "episode_policy_projections",
        "episode_id",
        scopeEpisodeIds
      );
      copyRows(target, "episode_policy_projections", projectionRows);
      copyRows(target, "episode_policy_projection_nodes", selectRowsByValues(
        source,
        "episode_policy_projection_nodes",
        "projection_id",
        stringColumn(projectionRows, "id")
      ));
    })();
    source.exec("ROLLBACK");
  } finally {
    if (source.inTransaction) source.exec("ROLLBACK");
    source.close();
    target.close();
  }
}

function selectRowsByValues(
  db: Database.Database,
  table: string,
  column: string,
  values: readonly string[]
): Array<Record<string, unknown>> {
  if (values.length === 0 || !sourceTableExists(db, table)) return [];
  const placeholders = values.map(() => "?").join(", ");
  return db.prepare(
    `SELECT * FROM ${quoteIdentifier(table)} WHERE ${quoteIdentifier(column)} IN (${placeholders})`
  ).all(...values) as Array<Record<string, unknown>>;
}

function copyRows(
  target: Database.Database,
  table: string,
  rows: readonly Record<string, unknown>[]
): void {
  if (rows.length === 0) return;
  const targetColumns = new Set((target.prepare(
    `PRAGMA table_info(${quoteIdentifier(table)})`
  ).all() as Array<{ name: string }>).map((column) => column.name));
  const columns = Object.keys(rows[0]!).filter((column) => targetColumns.has(column));
  if (columns.length === 0) return;
  const insert = target.prepare(
    `INSERT INTO ${quoteIdentifier(table)} (${columns.map(quoteIdentifier).join(", ")}) ` +
    `VALUES (${columns.map(() => "?").join(", ")})`
  );
  for (const row of rows) insert.run(...columns.map((column) => row[column] ?? null));
}

function sourceTableExists(db: Database.Database, table: string): boolean {
  return Boolean(db.prepare(
    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?"
  ).get(table));
}

function stringColumn(rows: readonly Record<string, unknown>[], column: string): string[] {
  return unique(rows.flatMap((row) => typeof row[column] === "string" ? [row[column]] : []));
}

function quoteIdentifier(value: string): string {
  if (!/^[a-z_][a-z0-9_]*$/i.test(value)) throw new Error(`Unsafe SQLite identifier: ${value}`);
  return `"${value}"`;
}

function publicModelConfig(config: {
  provider: string;
  model?: string;
  endpoint?: string;
}): { provider: string; model?: string; endpoint?: string } {
  return {
    provider: config.provider,
    ...(config.model ? { model: config.model } : {}),
    ...(config.endpoint ? { endpoint: config.endpoint } : {})
  };
}

function sanitizeArtifact(value: unknown, depth = 0): unknown {
  if (depth > 16) return "[truncated]";
  if (typeof value === "string") return clip(redactSensitiveText(value), 20_000);
  if (typeof value === "number" || typeof value === "boolean" || value === null) return value;
  if (Array.isArray(value)) {
    return value.slice(0, 1_000).map((item) => sanitizeArtifact(item, depth + 1));
  }
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([key]) => !/thinking|reasoning|chain.?of.?thought/i.test(key))
    .map(([key, nested]) => [
      key,
      /api.?key|token|secret|password|authorization|credential|cookie/i.test(key)
        ? "[redacted]"
        : sanitizeArtifact(nested, depth + 1)
    ]));
}

function persistArtifact(
  requestedPath: string,
  artifact: ReplayArtifact
): { json: string; markdown: string } {
  const jsonPath = resolve(requestedPath);
  if (!jsonPath.endsWith(".json")) throw new Error("--output must point to a .json file");
  const markdownPath = jsonPath.slice(0, -5) + ".md";
  atomicWrite(jsonPath, `${JSON.stringify(artifact, null, 2)}\n`);
  atomicWrite(markdownPath, renderMarkdown(artifact));
  return { json: jsonPath, markdown: markdownPath };
}

function atomicWrite(path: string, content: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = resolve(dirname(path), `.${basename(path)}.${process.pid}.tmp`);
  writeFileSync(temporary, content, { encoding: "utf8", mode: 0o600 });
  renameSync(temporary, path);
}

function renderMarkdown(artifact: ReplayArtifact): string {
  const lines = [
    "# Trace2Skill Replay Report",
    "",
    `- Generated: ${artifact.generatedAt}`,
    `- Mode: ${artifact.mode}`,
    `- Replay start: ${artifact.replayStart}`,
    `- Evolution model: ${artifact.models.evolution.provider}/${artifact.models.evolution.model ?? "default"}`,
    `- Embedding model: ${artifact.models.embedding.provider}/${artifact.models.embedding.model ?? "default"}`,
    `- Pipeline healthy: ${artifact.summary.pipelineHealthy}/${artifact.summary.requested}`,
    `- Skill eligible / compiled: ${artifact.summary.skillEligible}/${artifact.summary.skillCompiled}`,
    `- Correctly abstained / blocked: ${artifact.summary.correctlyAbstained}/${artifact.summary.blocked}`,
    ""
  ];
  for (const item of artifact.episodes) {
    const report = item.replay?.after ?? item.diagnostic;
    lines.push(`## ${item.episodeId}`, "");
    lines.push(`- Pipeline healthy: ${item.pipelineHealthy ? "yes" : "no"}`);
    lines.push(`- Decision: ${item.decision}`);
    lines.push(`- Path / Span / Step: ${report?.path?.id ?? "missing"} / ${report?.path?.spanCount ?? 0} / ${report?.path?.stepCount ?? 0}`);
    lines.push(`- Policy coverage: ${formatRatio(report?.checks.policyProjectionCoverage ?? 0)}`);
    lines.push(`- Episode family: ${report?.capability?.familyId ?? "missing"}`);
    lines.push(`- Episode-first route: ${report?.checks.episodeSimilarityRoute ? "hit" : "no hit"}`);
    lines.push(`- Policy-first route: ${report?.checks.policySequenceRoute ? "hit" : "no hit"}`);
    lines.push(`- Ready candidate: ${report?.checks.readyCandidate ? "yes" : "no"}`);
    lines.push(`- Compiled Skill: ${report?.checks.compiledSkill ? "yes" : "no"}`);
    if (item.replay) {
      if (item.rawReconstruction) {
        lines.push(`- Raw reconstruction: ${item.rawReconstruction.pathId} (${item.rawReconstruction.created ? "new" : "reused"})`);
      }
      lines.push(`- Affected patterns: ${item.replay.actions.affectedPatternIds.join(", ") || "none"}`);
      lines.push(`- Compiled skills: ${item.replay.actions.compiledSkillIds.join(", ") || "none"}`);
      lines.push(`- Reused skills: ${item.replay.actions.reusedSkillIds.join(", ") || "none"}`);
    }
    if (report?.capability?.affinities.length) {
      lines.push("", "### Episode affinities", "");
      lines.push("| Peer Episode | Goal | Transition | Outcome | Context | Path | Combined | Family |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
      for (const affinity of report.capability.affinities) {
        lines.push(`| ${affinity.peerEpisodeId} | ${affinity.goalSimilarity.toFixed(3)} | ${affinity.stateTransitionSimilarity.toFixed(3)} | ${affinity.outcomeSimilarity.toFixed(3)} | ${affinity.contextSimilarity.toFixed(3)} | ${affinity.pathStructureSimilarity.toFixed(3)} | ${affinity.combinedSimilarity.toFixed(3)} | ${affinity.familyEligible ? "yes" : "no"} |`);
      }
    }
    if (report?.patterns.length) {
      lines.push("", "### Patterns", "");
      lines.push("| Type | Policies | Routes | Gap nodes | Support | Candidate | Skill |", "| --- | --- | --- | --- | ---: | --- | --- |");
      for (const pattern of report.patterns) {
        lines.push(`| ${pattern.capabilityType} | ${pattern.policyKeys.join(" → ")} | ${pattern.occurrence.discoverySources.join(" + ")} | ${pattern.occurrence.gapNodeIndexes.join(", ") || "—"} | ${pattern.distinctSupportEpisodeCount} | ${pattern.candidate?.lifecycleStatus ?? "—"} | ${pattern.skill?.id ?? "—"} |`);
      }
    }
    if (
      item.effectiveReasons.length > 0 ||
      item.decisionReasons.length > 0 ||
      (report?.blockers.length ?? 0) > 0
    ) {
      lines.push("", "### Diagnostics", "");
      for (const reason of unique([
        ...item.effectiveReasons,
        ...item.decisionReasons,
        ...(report?.blockers ?? [])
      ])) {
        lines.push(`- ${reason}`);
      }
    }
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
}

function formatRatio(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

function parseArgs(argv: readonly string[]): ReplayArgs {
  const episodeIds: string[] = [];
  let mode: ReplayMode = "inspect";
  let dbPath: string | undefined;
  let configPath: string | undefined;
  let outputPath: string | undefined;
  let confirmWrite = false;
  let replayStart: ReplayStart = "active_procedural_path";
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--episode" || arg === "--episode-id") {
      episodeIds.push(...requiredValue(argv, ++index, arg).split(",").map((item) => item.trim()));
    } else if (arg === "--mode") {
      const value = requiredValue(argv, ++index, "--mode");
      if (value !== "inspect" && value !== "dry-run" && value !== "commit") {
        throw new Error(`Invalid --mode: ${value}`);
      }
      mode = value;
    } else if (arg === "--from") {
      const value = requiredValue(argv, ++index, "--from");
      if (value !== "raw" && value !== "path") throw new Error(`Invalid --from: ${value}`);
      replayStart = value === "raw" ? "raw_episode" : "active_procedural_path";
    } else if (arg === "--db") dbPath = requiredValue(argv, ++index, "--db");
    else if (arg === "--config") configPath = requiredValue(argv, ++index, "--config");
    else if (arg === "--output") outputPath = requiredValue(argv, ++index, "--output");
    else if (arg === "--confirm-write") confirmWrite = true;
    else if (arg === "--help" || arg === "-h") {
      process.stdout.write(helpText());
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  const normalizedEpisodeIds = unique(episodeIds.filter(Boolean));
  if (normalizedEpisodeIds.length === 0) throw new Error("Pass at least one --episode");
  return {
    episodeIds: normalizedEpisodeIds,
    mode,
    ...(dbPath ? { dbPath } : {}),
    ...(configPath ? { configPath } : {}),
    ...(outputPath ? { outputPath } : {}),
    confirmWrite,
    replayStart
  };
}

function requiredValue(argv: readonly string[], index: number, option: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`${option} requires a value`);
  return value;
}

function helpText(): string {
  return `Usage:\n  npm run trace2skill:replay -- --episode <id> [--episode <id>] --mode <inspect|dry-run|commit> [--from <path|raw>] [--db <path>] [--config <path>] [--output <report.json>] [--confirm-write]\n\nModes:\n  inspect  Read persisted Trace2Skill state; no model calls or writes.\n  dry-run  Clone the scoped Trace2Skill dependency closure, run production code on the clone, and discard it.\n  commit   Replay the active path against the source database; requires --confirm-write.\n\nReplay start:\n  path  Reuse the active procedural path and replay Projection through Skill (default).\n  raw   Dry-run only. Reconstruct Span/State from stored raw turns, then drain SpanCredit, clustering, Policy, Projection, mining, and Skill jobs.\n`;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
});
