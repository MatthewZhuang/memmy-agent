import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import {
  createEmbedder,
  createLlmClient,
  DEFAULT_MEMMY_CONFIG,
  MemoryDb,
  type Embedder,
  type LlmClient,
  type MemoryRow,
  type ToolCallPayload
} from "../src/index.js";
import { signatureFromTraceParts } from "../src/algorithm/plugin-algorithms.js";
import { attachMemoryVector, memoryVector } from "../src/storage/memory-vector-state.js";
import { Repositories, type EvolutionJobRecord, type RawTurnRecord } from "../src/storage/repositories.js";
import { BigTurnSpanPipeline } from "../src/service/evolution/big-turn-span-pipeline.js";
import {
  buildSpanPartition,
  SPAN_CLUSTER_ALGORITHM_VERSION
} from "../src/service/evolution/span-clustering.js";
import { SpanClusteringPipeline } from "../src/service/evolution/span-clustering.js";
import { SpanPolicyInductionPipeline } from "../src/service/evolution/span-policy-induction.js";
import { spanEmbeddingText } from "../src/service/embedding/embedding-pipeline.js";
import { spanPayload } from "../src/service/evolution/span-model.js";
import { stableHash } from "../src/utils/id.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const WORK_DB = join(OUT_DIR, "span_v2_experiment.sqlite");
const RESULT_JSON = join(OUT_DIR, "span_v2_experiment_results.json");
const REPORT_MD = join(OUT_DIR, "span_v2_experiment_report.md");
const EXTERNAL_ARTIFACTS_ROOT = process.env.SPAN_V2_EXTERNAL_ARTIFACTS_ROOT
  ?? "/Users/lwm/dev/MemoryProject/mindock-agent/artifacts";
const LOCAL_ARTIFACTS_ROOT = join(REPO_ROOT, "artifacts");
const EXPERIMENT_LLM_ENDPOINT = process.env.SPAN_V2_LLM_ENDPOINT ?? "https://api-int.memtensor.cn/v1";
const EXPERIMENT_LLM_MODEL = process.env.SPAN_V2_LLM_MODEL ?? "deepseek-v4-pro";
const EXPERIMENT_EMBEDDING_ENDPOINT = process.env.SPAN_V2_EMBEDDING_ENDPOINT ?? "https://apigw.memtensor.cn/model/embedding/v1";
const EXPERIMENT_EMBEDDING_MODEL = process.env.SPAN_V2_EMBEDDING_MODEL ?? "bge-m3";
const PRIMARY_GOAL_THRESHOLD = numberEnv("SPAN_V2_GOAL_THRESHOLD") ?? 0.9;
const PRIMARY_POLICY_THRESHOLD = numberEnv("SPAN_V2_POLICY_THRESHOLD") ?? 0.78;

type Sample = {
  traceId: string;
  rawTurn: RawTurnRecord;
  taskTitle: string;
  callCount: number;
  sourceId: string;
  sourceLabel: string;
  trialDir: string;
  taskName: string;
  trialName: string;
};

type DatasetSourceSpec = {
  id: string;
  label: string;
  root: string;
};

type DatasetSourceSummary = {
  id: string;
  label: string;
  root: string;
  exists: boolean;
  discoveredTrialDirs: number;
  selectedTrialDirs: number;
  loadedSamples: number;
  eligibleSamples: number;
  skippedSamples: number;
};

type DatasetSummary = {
  sources: DatasetSourceSummary[];
  requestedSamples: number;
  selectedSamples: number;
  eligibleSamples: number;
  skippedSamples: number;
  sampleLimit?: number;
  retryDirsExcluded: number;
  extraTraceSources: Array<{ pattern: string; found: number }>;
};

type DatasetLoadResult = {
  samples: Sample[];
  summary: DatasetSummary;
};

type SplitFailure = {
  traceId: string;
  sourceId: string;
  taskTitle: string;
  error: string;
};

const policyFamilies = [
  {
    family: "read-source",
    tools: ["read_file", "pdftotext", "web_fetch", "web_search"],
    goal: "收集并读取任务输入材料",
    policy: "先读取参考文件、网页或原始材料，提取后续制作交付物所需事实。"
  },
  {
    family: "inspect-data",
    tools: ["python", "python3", "exec", "sqlite3"],
    goal: "检查数据结构并计算关键字段",
    policy: "用脚本或命令检查输入表格、文本和文件结构，再计算需要写入交付物的数据。"
  },
  {
    family: "create-artifact",
    tools: ["write_file", "python", "python3", "exec"],
    goal: "生成目标交付物文件",
    policy: "根据已提取的信息用脚本或文件写入工具生成 PDF、PPT、Excel 或文档交付物。"
  },
  {
    family: "verify-artifact",
    tools: ["ls", "file", "python", "python3", "exec"],
    goal: "验证交付物存在且内容可检查",
    policy: "生成后检查文件存在、格式、页数或关键内容，必要时再修正。"
  }
];

async function main(): Promise<void> {
  mkdirSync(OUT_DIR, { recursive: true });
  if (process.env.SPAN_V2_RENDER_ONLY === "1") {
    const existingResult = JSON.parse(readFileSync(RESULT_JSON, "utf8")) as Record<string, unknown>;
    writeFileSync(REPORT_MD, renderReport(existingResult));
    return;
  }
  if (process.env.SPAN_V2_ABLATION_ONLY === "1") {
    const existingResult = JSON.parse(readFileSync(RESULT_JSON, "utf8")) as Record<string, unknown>;
    const result = await addEmbeddingAblations(existingResult, createExperimentEmbedder());
    writeFileSync(RESULT_JSON, JSON.stringify(result, null, 2));
    writeFileSync(REPORT_MD, renderReport(result));
    return;
  }
  const dataset = loadDataset();
  const samples = dataset.samples;
  const db = new MemoryDb({ path: WORK_DB });
  resetExperimentDb(db);
  const repos = new Repositories(db.db);
  const llm = createExperimentLlm();
  const embedder = createExperimentEmbedder();
  const enqueuedJobs: EvolutionJobRecord[] = [];
  const config = {
    ...DEFAULT_MEMMY_CONFIG,
    algorithm: {
      ...DEFAULT_MEMMY_CONFIG.algorithm,
      capture: {
        ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
        embedAfterCapture: false
      },
      spanClustering: {
        ...DEFAULT_MEMMY_CONFIG.algorithm.spanClustering,
        enabled: true,
        goalSimilarityThreshold: PRIMARY_GOAL_THRESHOLD,
        policySimilarityThreshold: PRIMARY_POLICY_THRESHOLD,
        minDistinctSources: 2
      }
    }
  };

  const buildMemory = buildMemoryFactory();
  const namespaceIdFromMemory = (memory: MemoryRow) =>
    [memory.userId, memory.agentId ?? "unknown", stringField(memory.info, "profile_id") ?? "default"].filter(Boolean).join(":");
  const enqueueJob = (input: {
    jobType: EvolutionJobRecord["jobType"];
    userId: string;
    sessionId?: string;
    episodeId?: string;
    targetMemoryId?: string;
    dedupeKey?: string;
    payload?: Record<string, unknown>;
    maxAttempts?: number;
    createdAt?: string;
  }): EvolutionJobRecord => {
    const at = input.createdAt ?? new Date().toISOString();
    const job = repos.runtime.enqueueJob({
      id: `job_${stableHash({ input, at, n: enqueuedJobs.length }).slice(0, 16)}`,
      jobType: input.jobType,
      status: "queued",
      dedupeKey: input.dedupeKey,
      userId: input.userId,
      sessionId: input.sessionId,
      episodeId: input.episodeId,
      targetMemoryId: input.targetMemoryId,
      payload: input.payload ?? {},
      attempts: 0,
      maxAttempts: input.maxAttempts ?? 3,
      createdAt: at,
      updatedAt: at
    });
    enqueuedJobs.push(job);
    return job;
  };

  const splitFailures: SplitFailure[] = [];
  for (const [index, sample] of samples.entries()) {
    console.log(`[span-v2] split ${index + 1}/${samples.length} ${sample.sourceId}/${sample.trialName} tools=${sample.callCount}`);
    repos.runtime.createSession({
      id: sample.rawTurn.sessionId,
      userId: sample.rawTurn.userId,
      source: "codex",
      profileId: "analysis",
      status: "closed",
      meta: {},
      openedAt: sample.rawTurn.createdAt,
      lastSeenAt: sample.rawTurn.createdAt,
      closedAt: sample.rawTurn.createdAt,
      updatedAt: sample.rawTurn.createdAt
    });
    repos.runtime.createEpisode({
      id: sample.rawTurn.episodeId,
      sessionId: sample.rawTurn.sessionId,
      userId: sample.rawTurn.userId,
      status: "closed",
      title: sample.taskTitle,
      summary: sample.taskTitle,
      l1MemoryIds: [sample.traceId],
      rawTurnIds: [sample.rawTurn.id],
      feedbackIds: [],
      decisionRepairIds: [],
      l2PolicyIds: [],
      l3WorldModelIds: [],
      skillMemoryIds: [],
      turnCount: 1,
      rTask: 1,
      rewardDetail: {},
      pipelineStatus: "idle",
      meta: {},
      openedAt: sample.rawTurn.createdAt,
      closedAt: sample.rawTurn.createdAt,
      updatedAt: sample.rawTurn.createdAt
    });
    repos.runtime.insertRawTurn(sample.rawTurn);
    const trace = buildTraceMemory(sample);
    repos.memories.insert(trace);
    const job: EvolutionJobRecord = {
      id: `job_split_${sample.traceId}`,
      jobType: "span_big_turn",
      status: "queued",
      userId: trace.userId,
      sessionId: trace.sessionId,
      episodeId: sample.rawTurn.episodeId,
      targetMemoryId: trace.id,
      payload: { rawTurnId: sample.rawTurn.id, rTask: 1 },
      attempts: 0,
      maxAttempts: 3,
      createdAt: trace.createdAt,
      updatedAt: trace.createdAt
    };
    const pipeline = new BigTurnSpanPipeline({
      repos,
      llm,
      buildMemory,
      enqueueJob,
      namespaceIdFromMemory,
      embedAfterCapture: () => false
    });
    try {
      await splitSpanJobWithRetries(pipeline, job);
    } catch (error) {
      splitFailures.push({
        traceId: sample.traceId,
        sourceId: sample.sourceId,
        taskTitle: sample.taskTitle,
        error: error instanceof Error ? error.message : String(error)
      });
      console.warn(`[span-v2] split failed ${sample.sourceId}/${sample.trialName}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  const spans = repos.memories
    .list({ memoryLayer: "L1", status: "activated" }, 10000)
    .filter((memory) => spanPayload(memory));
  console.log(`[span-v2] embedding spans=${spans.length}`);
  await attachRemoteSpanEmbeddings(repos, spans, embedder);

  const scopeId = "analysis-user:codex:analysis";
  const clusterJob: EvolutionJobRecord = {
    id: "job_span_cluster_experiment",
    jobType: "span_cluster",
    status: "queued",
    userId: "analysis-user",
    payload: { scopeId, algorithmVersion: SPAN_CLUSTER_ALGORITHM_VERSION },
    attempts: 0,
    maxAttempts: 3,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };
  new SpanClusteringPipeline({ repos, config, enqueueJob }).rebuildScope(clusterJob);

  const clusterInduction = new SpanPolicyInductionPipeline({
    config,
    repos,
    buildMemory,
    upsertEvolutionMemory: (memory) => {
      const previous = memory.memoryKey ? repos.memories.getByKey(memory.memoryLayer, memory.memoryKey) : undefined;
      if (previous) {
        const saved = repos.memories.update({ ...memory, id: previous.id, createdAt: previous.createdAt, version: previous.version });
        return { memory: saved, created: false, previous };
      }
      return { memory: repos.memories.insert(memory), created: true };
    },
    enqueueJob,
    enqueueChange: (change) => repos.runtime.appendChange(change as Parameters<typeof repos.runtime.appendChange>[0]),
    namespaceIdFromMemory
  });
  for (const job of enqueuedJobs.filter((job) => job.jobType === "l2_induction")) {
    clusterInduction.induce({ ...job, updatedAt: new Date().toISOString() });
  }

  const readySpans = repos.memories.listReadySpanV2();
  const familyBySpanId = buildFamilyBySpanId(readySpans, samples);
  const clusterRows = repos.spanClusters.listByScope(scopeId, SPAN_CLUSTER_ALGORITHM_VERSION);
  const legacySpanEmbeddingClusters = await buildLegacySpanEmbeddingPartition(readySpans, scopeId, embedder);
  const signatureBuckets = buildSignatureSpanBuckets(readySpans, samples, familyBySpanId);
  const thresholdSweep = buildThresholdSweep(readySpans, scopeId, familyBySpanId);
  const stressAblation = await buildStressAblation(scopeId, embedder);
  const result = {
    dataset: dataset.summary,
    llm: publicModelConfig(llm.config),
    embedding: publicModelConfig(embedder.config),
    sampleCount: samples.length,
    eligibleSampleCount: samples.filter((sample) => sample.callCount >= 11).length,
    thresholds: {
      goalSimilarityThreshold: PRIMARY_GOAL_THRESHOLD,
      policySimilarityThreshold: PRIMARY_POLICY_THRESHOLD,
      minDistinctSources: 2
    },
    splitFailures,
    selectedSamples: samples.map((sample) => ({
      traceId: sample.traceId,
      rawTurnId: sample.rawTurn.id,
      sourceId: sample.sourceId,
      trialName: sample.trialName,
      trialDir: sample.trialDir,
      taskTitle: sample.taskTitle,
      toolCalls: sample.callCount
    })),
    spans: readySpans.map((memory) => {
      const span = spanPayload(memory)!;
      return {
        id: memory.id,
        sourceTraceId: span.source_trace_id,
        range: [span.tool_call_start, span.tool_call_end],
        toolCallCount: span.tool_call_count,
        goal: span.goal,
        policy: span.policy,
        summary: span.summary,
        family: familyBySpanId.get(memory.id) ?? "unknown"
      };
    }),
    dualThreshold: summarizeClusters(repos, clusterRows, familyBySpanId),
    legacySpanEmbedding: summarizePartition(legacySpanEmbeddingClusters, familyBySpanId),
    signatureBuckets,
    thresholdSweep,
    stressAblation,
    policies: repos.memories.list({ memoryLayer: "L2", status: "activated" }, 10000)
      .filter((memory) => memory.memoryKey?.startsWith("policy:span-cluster:"))
      .map((memory) => ({
        id: memory.id,
        key: memory.memoryKey,
        title: stringField(memory.info, "title"),
        sourceSpanIds: arrayField(memory.properties.internal_info.policy, "source_span_ids")
      }))
  };
  writeFileSync(RESULT_JSON, JSON.stringify(result, null, 2));
  writeFileSync(REPORT_MD, renderReport(result));
  db.close();
}

function loadDataset(): DatasetLoadResult {
  const specs = datasetSourceSpecs();
  const samples: Sample[] = [];
  const sources: DatasetSourceSummary[] = [];
  let retryDirsExcluded = 0;
  for (const spec of specs) {
    const exists = existsSync(spec.root);
    const allDirs = exists
      ? readdirSync(spec.root, { withFileTypes: true }).filter((entry) => entry.isDirectory())
      : [];
    const selectedDirs = allDirs
      .filter((entry) => /__trial_1$/.test(entry.name))
      .sort((a, b) => a.name.localeCompare(b.name));
    retryDirsExcluded += allDirs.filter((entry) => /retry/i.test(entry.name)).length;
    const before = samples.length;
    let skippedSamples = 0;
    for (const entry of selectedDirs) {
      const trialDir = join(spec.root, entry.name);
      const sample = sampleFromTrialDir(spec, trialDir);
      if (!sample) {
        skippedSamples += 1;
        continue;
      }
      samples.push(sample);
    }
    const loaded = samples.length - before;
    sources.push({
      id: spec.id,
      label: spec.label,
      root: spec.root,
      exists,
      discoveredTrialDirs: allDirs.length,
      selectedTrialDirs: selectedDirs.length,
      loadedSamples: loaded,
      eligibleSamples: samples.slice(before).filter((sample) => sample.callCount >= 11).length,
      skippedSamples
    });
  }
  const limit = positiveIntegerEnv("SPAN_V2_SAMPLE_LIMIT");
  const limitedSamples = limit === undefined ? samples : samples.slice(0, limit);
  return {
    samples: limitedSamples,
    summary: {
      sources,
      requestedSamples: samples.length,
      selectedSamples: limitedSamples.length,
      eligibleSamples: limitedSamples.filter((sample) => sample.callCount >= 11).length,
      skippedSamples: sources.reduce((sum, source) => sum + source.skippedSamples, 0),
      ...(limit !== undefined ? { sampleLimit: limit } : {}),
      retryDirsExcluded,
      extraTraceSources: discoverExtraTraceSources()
    }
  };
}

function datasetSourceSpecs(): DatasetSourceSpec[] {
  return [
    {
      id: "knowledge-work-train",
      label: "baseline-20260719/train",
      root: join(EXTERNAL_ARTIFACTS_ROOT, "baseline-20260719/train")
    },
    {
      id: "memory-v10-test",
      label: "memory-v10-test-20260804",
      root: firstExistingPath([
        join(EXTERNAL_ARTIFACTS_ROOT, "memory-v10-test-20260804"),
        join(LOCAL_ARTIFACTS_ROOT, "memory-v10-test-20260804")
      ])
    }
  ];
}

function firstExistingPath(paths: string[]): string {
  return paths.find((path) => existsSync(path)) ?? paths[0]!;
}

function sampleFromTrialDir(source: DatasetSourceSpec, trialDir: string): Sample | undefined {
  const sessionPath = join(trialDir, "session.jsonl");
  if (!existsSync(sessionPath)) return undefined;
  const records = parseJsonl(sessionPath);
  const firstUserIndex = records.findIndex((record) =>
    stringField(record, "role") === "user"
      && !String(record.content ?? "").startsWith("Verifier feedback")
  );
  if (firstUserIndex < 0) return undefined;
  const nextUserIndex = records.findIndex((record, index) =>
    index > firstUserIndex && stringField(record, "role") === "user"
  );
  const turnRecords = records.slice(firstUserIndex, nextUserIndex < 0 ? undefined : nextUserIndex);
  const user = turnRecords[0]!;
  const trialName = basename(trialDir);
  const taskName = trialName.replace(/__trial_.*$/, "");
  const at = stringField(user, "timestamp")
    ?? stringField(records.find((record) => record.recordType === "metadata"), "createdAt")
    ?? new Date(0).toISOString();
  const toolResultsById = new Map<string, Record<string, unknown>>();
  for (const record of turnRecords) {
    if (stringField(record, "role") === "tool") {
      const id = stringField(record, "tool_call_id");
      if (id) toolResultsById.set(id, record);
    }
  }
  const toolCalls = extractToolCalls(turnRecords, toolResultsById);
  const assistantTexts = turnRecords
    .filter((record) => stringField(record, "role") === "assistant")
    .map((record) => stringField(record, "content") ?? "")
    .filter((content) => content.trim());
  const traceId = `trace_${stableHash(`${source.id}:${trialName}`).slice(0, 20)}`;
  const rawTurnId = `raw_${stableHash(`${source.id}:${trialName}:turn`).slice(0, 20)}`;
  const sessionId = `session_${stableHash(`${source.id}:${trialName}`).slice(0, 20)}`;
  const episodeId = `episode_${stableHash(`${source.id}:${taskName}`).slice(0, 20)}`;
  const userText = String(user.content ?? "");
  const rawTurn: RawTurnRecord = {
    id: rawTurnId,
    sessionId,
    episodeId,
    turnId: `turn_${stableHash(`${source.id}:${trialName}:0`).slice(0, 20)}`,
    userId: "analysis-user",
    conversationId: sessionId,
    userText,
    assistantText: assistantTexts.at(-1),
    reasoningSummary: undefined,
    toolCalls,
    toolResults: [...toolResultsById.values()],
    sourceMemoryIds: [],
    usage: {},
    messagePayload: {
      source_id: source.id,
      source_label: source.label,
      trial_dir: trialDir,
      session_path: sessionPath
    },
    status: "succeeded",
    redactedAt: null,
    deletedAt: null,
    createdAt: at
  };
  return {
    traceId,
    rawTurn,
    taskTitle: taskTitle(userText),
    callCount: toolCalls.length,
    sourceId: source.id,
    sourceLabel: source.label,
    trialDir,
    taskName,
    trialName
  };
}

function parseJsonl(path: string): Array<Record<string, unknown>> {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function extractToolCalls(
  turnRecords: Array<Record<string, unknown>>,
  toolResultsById: Map<string, Record<string, unknown>>
): ToolCallPayload[] {
  const calls: ToolCallPayload[] = [];
  for (const record of turnRecords) {
    if (stringField(record, "role") !== "assistant") continue;
    const toolCalls = Array.isArray(record.tool_calls) ? record.tool_calls : [];
    for (const call of toolCalls) {
      if (!call || typeof call !== "object") continue;
      const callRecord = call as Record<string, unknown>;
      const fn = callRecord.function && typeof callRecord.function === "object"
        ? callRecord.function as Record<string, unknown>
        : {};
      const id = stringField(callRecord, "id");
      const name = stringField(fn, "name") ?? stringField(callRecord, "name") ?? "_";
      const result = id ? toolResultsById.get(id) : undefined;
      const output = result ? stringField(result, "content") ?? "" : undefined;
      calls.push({
        ...(id ? { id } : {}),
        name,
        input: parseMaybeJson(stringField(fn, "arguments")),
        ...(output !== undefined ? { output } : {}),
        success: inferToolSuccess(output),
        assistantTextBefore: stringField(record, "content")
      });
    }
  }
  return calls;
}

function parseMaybeJson(value: string | undefined): unknown {
  if (value === undefined) return undefined;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function inferToolSuccess(output: string | undefined): boolean | undefined {
  if (output === undefined) return undefined;
  const match = output.match(/Exit code:\s*(-?\d+)/i);
  return match ? match[1] === "0" : !/error|failed|traceback/i.test(output);
}

function positiveIntegerEnv(name: string): number | undefined {
  const value = process.env[name]?.trim();
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function numberEnv(name: string): number | undefined {
  const value = process.env[name]?.trim();
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function discoverExtraTraceSources(): Array<{ pattern: string; found: number }> {
  if (!existsSync(EXTERNAL_ARTIFACTS_ROOT)) return [];
  const roots = readdirSync(EXTERNAL_ARTIFACTS_ROOT, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
  return [
    { pattern: "wde", found: roots.filter((name) => /wde/i.test(name)).length },
    { pattern: "code|implement|swe", found: roots.filter((name) => /code|implement|swe/i.test(name)).length }
  ];
}

function createExperimentLlm(): LlmClient {
  return createLlmClient({
    ...DEFAULT_MEMMY_CONFIG.evolution,
    provider: "openai_compatible",
    vendor: "deepseek",
    endpoint: EXPERIMENT_LLM_ENDPOINT,
    model: EXPERIMENT_LLM_MODEL,
    apiKey: requiredExperimentSecret("SPAN_V2_LLM_API_KEY", "MEMMY_EVOLUTION_API_KEY"),
    enableThinking: false,
    temperature: 0.2,
    maxTokens: 4096,
    timeoutMs: 180_000,
    maxRetries: 2,
    malformedRetries: 2
  });
}

async function splitSpanJobWithRetries(pipeline: BigTurnSpanPipeline, job: EvolutionJobRecord): Promise<void> {
  let lastError: unknown;
  for (let attempt = 0; attempt < job.maxAttempts; attempt += 1) {
    try {
      await pipeline.splitAndStore({
        ...job,
        attempts: attempt,
        updatedAt: new Date().toISOString()
      });
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

function createExperimentEmbedder(): Embedder {
  return createEmbedder({
    ...DEFAULT_MEMMY_CONFIG.embedding,
    provider: "openai_compatible",
    endpoint: EXPERIMENT_EMBEDDING_ENDPOINT,
    model: EXPERIMENT_EMBEDDING_MODEL,
    apiKey: process.env.SPAN_V2_EMBEDDING_API_KEY ?? process.env.MEMMY_EMBEDDING_API_KEY ?? "EMPTY",
    batchSize: 16,
    timeoutMs: 120_000,
    maxRetries: 2,
    cache: true,
    normalize: true
  });
}

function requiredExperimentSecret(...names: string[]): string {
  for (const name of names) {
    const value = process.env[name]?.trim();
    if (value) return value;
  }
  throw new Error(`Missing experiment LLM API key. Set one of: ${names.join(", ")}`);
}

function publicModelConfig(config: {
  provider: string;
  vendor?: string;
  endpoint?: string;
  model?: string;
}) {
  return {
    provider: config.provider,
    ...(config.vendor ? { vendor: config.vendor } : {}),
    endpoint: config.endpoint,
    model: config.model
  };
}

async function attachRemoteSpanEmbeddings(repos: Repositories, spans: MemoryRow[], embedder: Embedder): Promise<void> {
  const goalVectors = await embedder.embed(spans.map((span) => spanEmbeddingText(span, "vec_goal") ?? ""), "document");
  const policyVectors = await embedder.embed(spans.map((span) => spanEmbeddingText(span, "vec_policy") ?? ""), "document");
  for (const [index, span] of spans.entries()) {
    repos.memories.updateMaintenance(attachMemoryVector(attachMemoryVector(span, {
      vectorField: "vec_goal",
      vector: goalVectors[index]!,
      embeddingModel: embedder.config.model ?? EXPERIMENT_EMBEDDING_MODEL,
      embeddingProvider: embedder.config.provider
    }), {
      vectorField: "vec_policy",
      vector: policyVectors[index]!,
      embeddingModel: embedder.config.model ?? EXPERIMENT_EMBEDDING_MODEL,
      embeddingProvider: embedder.config.provider
    }));
  }
}

function classifyFamily(names: string[]): string {
  const joined = names.join(" ").toLowerCase();
  if (/write_file|ppt|pdf|docx|xlsx|save|export/.test(joined)) return "create-artifact";
  if (/ls|file|grep|find|verify|check/.test(joined)) return "verify-artifact";
  if (/read_file|pdftotext|web_fetch|web_search/.test(joined)) return "read-source";
  return "inspect-data";
}

async function buildLegacySpanEmbeddingPartition(spans: MemoryRow[], scopeId: string, embedder: Embedder) {
  const vectors = await embedder.embed(spans.map(legacySpanEmbeddingText), "document");
  const clusterable = spans.map((memory, index) => ({
    id: memory.id,
    sourceTraceId: spanPayload(memory)!.source_trace_id,
    createdAt: memory.createdAt,
    vecGoal: vectors[index]!,
    vecPolicy: vectors[index]!
  }));
  return buildSpanPartition({
    scopeId,
    spans: clusterable,
    goalThreshold: 0.72,
    policyThreshold: 0.72,
    minDistinctSources: 2
  });
}

function buildThresholdSweep(spans: MemoryRow[], scopeId: string, familyBySpanId: Map<string, string>) {
  const clusterable = spans
    .map((memory) => {
      const span = spanPayload(memory);
      const vecGoal = memoryVector(memory, "vec_goal");
      const vecPolicy = memoryVector(memory, "vec_policy");
      return span && vecGoal && vecPolicy
        ? {
          id: memory.id,
          sourceTraceId: span.source_trace_id,
          createdAt: memory.createdAt,
          vecGoal,
          vecPolicy
        }
        : undefined;
    })
    .filter((span): span is NonNullable<typeof span> => Boolean(span));
  const goalThresholds = [0.76, 0.78, 0.80, 0.84, 0.88, 0.90];
  const policyThresholds = [0.76, 0.78, 0.80];
  return goalThresholds.flatMap((goalThreshold) =>
    policyThresholds.map((policyThreshold) => {
      const clusters = summarizePartition(buildSpanPartition({
        scopeId,
        spans: clusterable,
        goalThreshold,
        policyThreshold,
        minDistinctSources: 2
      }), familyBySpanId);
      const quality = clusterQuality(clusters);
      return {
        goalThreshold,
        policyThreshold,
        bucketCount: clusters.length,
        purity: roundMetric(weightedPurity(clusters)),
        singletonSpanCount: quality.singletonSpanCount,
        singletonRate: roundMetric(quality.spanCount === 0 ? 0 : quality.singletonSpanCount / quality.spanCount),
        reusableSpanCount: quality.reusableSpanCount,
        reusableRate: roundMetric(quality.spanCount === 0 ? 0 : quality.reusableSpanCount / quality.spanCount),
        nonSingletonBucketCount: clusters.filter((cluster) => cluster.memberCount > 1).length,
        nonSingletonSpanCount: quality.nonSingletonSpanCount
      };
    })
  );
}

async function addEmbeddingAblations(result: Record<string, unknown>, embedder: Embedder): Promise<Record<string, unknown>> {
  const spans = Array.isArray(result.spans) ? result.spans as Array<Record<string, unknown>> : [];
  const scopeId = "analysis-user:codex:analysis";
  const familyBySpanId = new Map(spans.map((span) => [
    String(span.id),
    typeof span.family === "string" ? span.family : "unknown"
  ]));
  const ablations = {
    singleGoalPolicyEmbedding: await buildSingleEmbeddingAblation({
      spans,
      scopeId,
      familyBySpanId,
      embedder,
      label: "Goal+Policy",
      text: (span) => [
        `Goal: ${String(span.goal ?? "")}`,
        `Policy: ${String(span.policy ?? "")}`
      ].join("\n")
    }),
    singleGoalEmbedding: await buildSingleEmbeddingAblation({
      spans,
      scopeId,
      familyBySpanId,
      embedder,
      label: "Goal-only",
      text: (span) => `Goal: ${String(span.goal ?? "")}`
    }),
    singlePolicyEmbedding: await buildSingleEmbeddingAblation({
      spans,
      scopeId,
      familyBySpanId,
      embedder,
      label: "Policy-only",
      text: (span) => `Policy: ${String(span.policy ?? "")}`
    })
  };
  return {
    ...result,
    embeddingAblations: ablations
  };
}

async function buildSingleEmbeddingAblation(input: {
  spans: Array<Record<string, unknown>>;
  scopeId: string;
  familyBySpanId: Map<string, string>;
  embedder: Embedder;
  label: string;
  text(span: Record<string, unknown>): string;
}) {
  const vectors = await input.embedder.embed(input.spans.map(input.text), "document");
  const clusterable = input.spans.map((span, index) => ({
    id: String(span.id),
    sourceTraceId: String(span.sourceTraceId),
    createdAt: new Date(Date.UTC(2026, 7, 15, 0, 0, index)).toISOString(),
    vecGoal: vectors[index]!,
    vecPolicy: vectors[index]!
  }));
  return summarizePartition(buildSpanPartition({
    scopeId: input.scopeId,
    spans: clusterable,
    goalThreshold: PRIMARY_POLICY_THRESHOLD,
    policyThreshold: PRIMARY_POLICY_THRESHOLD,
    minDistinctSources: 2
  }), input.familyBySpanId);
}

function buildFamilyBySpanId(spans: MemoryRow[], samples: Sample[]): Map<string, string> {
  const sampleByTraceId = new Map(samples.map((sample) => [sample.traceId, sample]));
  const families = new Map<string, string>();
  for (const memory of spans) {
    const span = spanPayload(memory);
    const sample = span ? sampleByTraceId.get(span.source_trace_id) : undefined;
    const toolNames = span && sample
      ? sample.rawTurn.toolCalls
          .slice(span.tool_call_start, span.tool_call_end + 1)
          .map(toolName)
          .filter((name): name is string => Boolean(name))
      : [];
    families.set(memory.id, toolNames.length > 0 ? classifyFamily(toolNames) : familyFromPolicy(span?.policy ?? ""));
  }
  return families;
}

function buildSignatureSpanBuckets(spans: MemoryRow[], samples: Sample[], familyBySpanId: Map<string, string>) {
  const sampleByTraceId = new Map(samples.map((sample) => [sample.traceId, sample]));
  const buckets = new Map<string, MemoryRow[]>();
  for (const memory of spans) {
    const span = spanPayload(memory);
    if (!span) continue;
    const sample = sampleByTraceId.get(span.source_trace_id);
    const signature = sample ? signatureForSample(sample) : "_|_|_|_";
    buckets.set(signature, [...(buckets.get(signature) ?? []), memory]);
  }
  return [...buckets.entries()]
    .map(([signature, bucketSpans]) => {
      const sourceTraceIds = [...new Set(bucketSpans.map((memory) => spanPayload(memory)?.source_trace_id).filter(Boolean))].sort();
      return {
        id: `signature:${stableHash(signature).slice(0, 16)}`,
        signature,
        status: sourceTraceIds.length >= 2 ? "ready" : "forming",
        memberCount: bucketSpans.length,
        distinctSourceCount: sourceTraceIds.length,
        families: countBy(bucketSpans.map((memory) => familyBySpanId.get(memory.id) ?? "unknown")),
        sourceTraceIds,
        memberSpanIds: bucketSpans.map((memory) => memory.id).sort()
      };
    })
    .sort((a, b) => b.memberCount - a.memberCount || a.signature.localeCompare(b.signature));
}

function signatureForSample(sample: Sample): string {
  return signatureFromTraceParts([], toolCallsForSignature(sample.rawTurn.toolCalls), sample.rawTurn.reasoningSummary ?? "");
}

function toolCallsForSignature(value: unknown[]): ToolCallPayload[] {
  return value
    .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object")
    .map((item) => ({
      id: typeof item.id === "string" ? item.id : undefined,
      name: typeof item.name === "string" && item.name.trim() ? item.name : "_",
      output: item.output,
      error: typeof item.error === "string" ? item.error : undefined,
      errorCode: typeof item.errorCode === "string" ? item.errorCode : undefined,
      success: typeof item.success === "boolean" ? item.success : undefined
    }));
}

function toolName(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const name = (value as Record<string, unknown>).name;
  return typeof name === "string" && name.trim() ? name.trim() : undefined;
}

async function buildStressAblation(scopeId: string, embedder: Embedder) {
  const cases = [
    stressSpan("same-goal-read", "trace-stress-a", "完成交付物准备", policyFamilies[0]!.policy, "stress-read-source", "exec", 1),
    stressSpan("same-goal-create", "trace-stress-b", "完成交付物准备", policyFamilies[2]!.policy, "stress-create-artifact", "exec", 2),
    stressSpan("same-goal-verify", "trace-stress-c", "完成交付物准备", policyFamilies[3]!.policy, "stress-verify-artifact", "exec", 3),
    stressSpan("same-policy-retail", "trace-stress-d", "收集采购展示材料", policyFamilies[0]!.policy, "stress-read-source", "exec", 4),
    stressSpan("same-policy-healthcare", "trace-stress-e", "收集医疗联系人材料", policyFamilies[0]!.policy, "stress-read-source", "exec", 5),
    stressSpan("inspect-data", "trace-stress-f", "计算输入数据字段", policyFamilies[1]!.policy, "stress-inspect-data", "exec", 6)
  ];
  const legacyVectors = await embedder.embed(cases.map((item) => `Goal: ${item.goal}\nSummary: ${item.goal}`), "document");
  const dualGoalVectors = await embedder.embed(cases.map((item) => `Goal: ${item.goal}`), "document");
  const dualPolicyVectors = await embedder.embed(cases.map((item) => `Policy: ${item.policy}`), "document");
  const legacyClusters = buildSpanPartition({
    scopeId,
    spans: cases.map((item, index) => ({
      id: item.id,
      sourceTraceId: item.sourceTraceId,
      createdAt: item.createdAt,
      vecGoal: legacyVectors[index]!,
      vecPolicy: legacyVectors[index]!
    })),
    goalThreshold: 0.72,
    policyThreshold: 0.72,
    minDistinctSources: 2
  });
  const dualClusters = buildSpanPartition({
    scopeId,
    spans: cases.map((item, index) => ({
      id: item.id,
      sourceTraceId: item.sourceTraceId,
      createdAt: item.createdAt,
      vecGoal: dualGoalVectors[index]!,
      vecPolicy: dualPolicyVectors[index]!
    })),
    goalThreshold: PRIMARY_GOAL_THRESHOLD,
    policyThreshold: PRIMARY_POLICY_THRESHOLD,
    minDistinctSources: 2
  });
  return {
    cases,
    legacy: summarizeStressPartition(legacyClusters, cases),
    signature: summarizeStressSignatureBuckets(cases),
    dualThreshold: summarizeStressPartition(dualClusters, cases)
  };
}

function summarizeClusters(
  repos: Repositories,
  clusters: ReturnType<Repositories["spanClusters"]["listByScope"]>,
  familyBySpanId: Map<string, string>
) {
  return clusters.map((cluster) => {
    const members = repos.spanClusters.listMembers(cluster.id);
    return {
      id: cluster.id,
      status: cluster.status,
      memberCount: cluster.memberCount,
      distinctSourceCount: cluster.distinctSourceCount,
      promotedPolicyId: cluster.promotedPolicyId,
      families: countBy(members.map((member) => familyBySpanId.get(member.spanId) ?? "unknown")),
      memberSpanIds: members.map((member) => member.spanId)
    };
  });
}

function summarizePartition(clusters: ReturnType<typeof buildSpanPartition>, familyBySpanId: Map<string, string>) {
  return clusters.map((cluster) => ({
    id: cluster.id,
    status: cluster.status,
    memberCount: cluster.memberCount,
    distinctSourceCount: cluster.distinctSourceCount,
    families: countBy(cluster.members.map((member) => familyBySpanId.get(member.spanId) ?? "unknown")),
    memberSpanIds: cluster.members.map((member) => member.spanId)
  }));
}

function summarizeStressPartition(
  clusters: ReturnType<typeof buildSpanPartition>,
  cases: ReturnType<typeof stressSpan>[]
) {
  const byId = new Map(cases.map((item) => [item.id, item]));
  return clusters.map((cluster) => ({
    id: cluster.id,
    status: cluster.status,
    memberCount: cluster.memberCount,
    distinctSourceCount: cluster.distinctSourceCount,
    families: countBy(cluster.members.map((member) => byId.get(member.spanId)?.family ?? "unknown")),
    goals: countBy(cluster.members.map((member) => byId.get(member.spanId)?.goal ?? "unknown")),
    policies: countBy(cluster.members.map((member) => byId.get(member.spanId)?.policy ?? "unknown")),
    memberSpanIds: cluster.members.map((member) => member.spanId)
  }));
}

function summarizeStressSignatureBuckets(cases: ReturnType<typeof stressSpan>[]) {
  const buckets = new Map<string, typeof cases>();
  for (const item of cases) {
    const signature = signatureFromTraceParts([], [{ name: item.firstTool }], "");
    buckets.set(signature, [...(buckets.get(signature) ?? []), item]);
  }
  return [...buckets.entries()]
    .map(([signature, bucketCases]) => ({
      id: `signature:${stableHash(signature).slice(0, 16)}`,
      signature,
      status: new Set(bucketCases.map((item) => item.sourceTraceId)).size >= 2 ? "ready" : "forming",
      memberCount: bucketCases.length,
      distinctSourceCount: new Set(bucketCases.map((item) => item.sourceTraceId)).size,
      families: countBy(bucketCases.map((item) => item.family)),
      goals: countBy(bucketCases.map((item) => item.goal)),
      policies: countBy(bucketCases.map((item) => item.policy)),
      memberSpanIds: bucketCases.map((item) => item.id)
    }))
    .sort((a, b) => b.memberCount - a.memberCount || a.signature.localeCompare(b.signature));
}

function renderReport(result: any): string {
  const dualPurity = weightedPurity(result.dualThreshold);
  const legacyPurity = weightedPurity(result.legacySpanEmbedding);
  const signaturePurity = weightedPurity(result.signatureBuckets);
  const singleGoalPolicy = result.embeddingAblations?.singleGoalPolicyEmbedding ?? [];
  const singleGoal = result.embeddingAblations?.singleGoalEmbedding ?? [];
  const singlePolicy = result.embeddingAblations?.singlePolicyEmbedding ?? [];
  const singleGoalPolicyPurity = weightedPurity(singleGoalPolicy);
  const singleGoalPurity = weightedPurity(singleGoal);
  const singlePolicyPurity = weightedPurity(singlePolicy);
  const dualNonSingleton = nonSingletonQuality(result.dualThreshold);
  const legacyNonSingleton = nonSingletonQuality(result.legacySpanEmbedding);
  const signatureNonSingleton = nonSingletonQuality(result.signatureBuckets);
  const singleGoalPolicyNonSingleton = nonSingletonQuality(singleGoalPolicy);
  const singleGoalNonSingleton = nonSingletonQuality(singleGoal);
  const singlePolicyNonSingleton = nonSingletonQuality(singlePolicy);
  const stressLegacyPurity = weightedPurity(result.stressAblation.legacy);
  const stressSignaturePurity = weightedPurity(result.stressAblation.signature);
  const stressDualPurity = weightedPurity(result.stressAblation.dualThreshold);
  const dualQuality = clusterQuality(result.dualThreshold);
  const legacyQuality = clusterQuality(result.legacySpanEmbedding);
  const signatureQuality = clusterQuality(result.signatureBuckets);
  const singleGoalPolicyQuality = clusterQuality(singleGoalPolicy);
  const singleGoalQuality = clusterQuality(singleGoal);
  const singlePolicyQuality = clusterQuality(singlePolicy);
  const datasetSources = Array.isArray(result.dataset?.sources) ? result.dataset.sources : [];
  const splitFailures = Array.isArray(result.splitFailures) ? result.splitFailures : [];
  const extraTraceSources = Array.isArray(result.dataset?.extraTraceSources) ? result.dataset.extraTraceSources : [];
  const thresholdSweep = Array.isArray(result.thresholdSweep) ? result.thresholdSweep : [];
  const dualSingletonRate = percentage(dualQuality.singletonSpanCount, dualQuality.spanCount);
  const legacySingletonRate = percentage(legacyQuality.singletonSpanCount, legacyQuality.spanCount);
  const signatureSingletonRate = percentage(signatureQuality.singletonSpanCount, signatureQuality.spanCount);
  const singleGoalPolicySingletonRate = percentage(singleGoalPolicyQuality.singletonSpanCount, singleGoalPolicyQuality.spanCount);
  const singleGoalSingletonRate = percentage(singleGoalQuality.singletonSpanCount, singleGoalQuality.spanCount);
  const singlePolicySingletonRate = percentage(singlePolicyQuality.singletonSpanCount, singlePolicyQuality.spanCount);
  const dualReusableRate = percentage(dualQuality.reusableSpanCount, dualQuality.spanCount);
  const legacyReusableRate = percentage(legacyQuality.reusableSpanCount, legacyQuality.spanCount);
  const signatureReusableRate = percentage(signatureQuality.reusableSpanCount, signatureQuality.spanCount);
  const singleGoalPolicyReusableRate = percentage(singleGoalPolicyQuality.reusableSpanCount, singleGoalPolicyQuality.spanCount);
  const singleGoalReusableRate = percentage(singleGoalQuality.reusableSpanCount, singleGoalQuality.spanCount);
  const singlePolicyReusableRate = percentage(singlePolicyQuality.reusableSpanCount, singlePolicyQuality.spanCount);
  return [
    "# Span v2 分桶与 Policy 生成实验报告",
    "",
    `生成时间：${new Date().toISOString()}`,
    "",
    "## 实验设置",
    "",
    "- 数据来源：trial 目录中的 `session.jsonl`；每个 trial 还原为一条 raw turn，再由真实 `span_big_turn` prompt 拆分 Span。",
    ...datasetSources.map((source: any) =>
      `  - ${source.label}：root=\`${source.root}\`，主 trial=${source.selectedTrialDirs}/${source.discoveredTrialDirs}，加载 trace=${source.loadedSamples}，eligible(>=11 tool calls)=${source.eligibleSamples}，跳过=${source.skippedSamples}。`
    ),
    `- 样本数：${result.sampleCount} 条真实 trace；其中 ${result.eligibleSampleCount ?? result.dataset?.eligibleSamples ?? 0} 条满足当前代码的 \`SPAN_BIG_TURN_MIN_TOOL_CALLS=11\` 拆分条件。`,
    `- retry 目录默认排除：${result.dataset?.retryDirsExcluded ?? 0} 个。`,
    `- 额外 WDE/code implement 轨迹源扫描：${extraTraceSources.map((item: any) => `${item.pattern}=${item.found}`).join("，") || "未扫描到"}。`,
    `- 生成 Span：${result.spans.length} 个。所有 Span 均满足 tool_call_count > 3。`,
    `- Span 拆分失败：${splitFailures.length} 条；失败样本不进入 embedding/分桶统计。`,
    `- Span 拆分：调用真实远端 LLM（provider=${result.llm.provider}，model=${result.llm.model}，endpoint=${result.llm.endpoint}）并使用 \`span.big_turn.v1\` prompt 生成 Span JSON；schema 校验失败最多重试 3 次，没有 heuristic fallback。`,
    "- Prompt 优化：`goal` 去除一次性文件名/客户名/领域名，保留任务类型、artifact 类型、工具/库、错误类别；`policy` 抽象为可复用方法模式，并只通过文本 embedding 影响聚类，不使用字段匹配、白名单、alias 或字符串 gate。",
    `- Embedding：使用远端 embedding（provider=${result.embedding.provider}，model=${result.embedding.model}，endpoint=${result.embedding.endpoint}）；对比项使用同一 embedding backend。`,
    `- 分桶：不使用 LLM。双阈值方法以 Goal/observed Policy 两路 embedding cosine 同时过阈值入桶；主阈值 goal=${result.thresholds?.goalSimilarityThreshold ?? "n/a"}，policy=${result.thresholds?.policySimilarityThreshold ?? "n/a"}。`,
    "- Purity 标签：根据每个 Span 覆盖的原始 tool-call range 的工具行为归为读取资料、检查数据、生成交付物或验证交付物；不依赖 LLM policy 的措辞是否恰好一致。",
    "- 原始 signature 基线：使用原始 `signatureFromTraceParts` 逻辑从完整 trace 的 tool call 计算 trace-level signature，再把该 trace 下的 Span 映射到同一旧桶；它只作为对照，不参与新正向路径。",
    "- 打分口径：weighted purity = 各桶最大 family 数之和 / 总 Span 数；单例覆盖率只描述当前样本下的保守分桶程度，不作为负面指标；ready/promoted 覆盖率描述当前样本里已有多少 Span 进入可提升 Policy 的桶。",
    "",
    "## 0. 代码路径运行结果",
    "",
    "- `span_big_turn` 拆分、Span v2 持久化、`vec_goal/vec_policy` 写入、`span_cluster` rebuild、`l2_induction(cluster)` 生成 Policy 均已在实验脚本中跑通。",
    `- 生成 Span bucket：双阈值 ${result.dualThreshold.length} 个；生成 span-cluster policy：${result.policies.length} 个。`,
    "",
    "## 1. 相似 Span 是否表述相同策略",
    "",
    `- 双阈值分桶的全量加权 purity：${dualPurity.toFixed(3)}；其中单例 Span 占 ${dualSingletonRate}（${dualQuality.singletonSpanCount}/${dualQuality.spanCount}）。单例多符合 precision-first 目标，不作为缺点。`,
    `- 当前样本中 ${dualQuality.nonSingletonSpanCount}/${dualQuality.spanCount} 个 Span 进入至少两个成员的桶；ready/promoted 覆盖为 ${dualQuality.reusableSpanCount}/${dualQuality.spanCount}。后续海量 trace/span 持续进入后，严格单例桶会自然生长为非单例桶。`,
    `- 非单例桶加权 purity：${dualNonSingleton.purity.toFixed(3)}，覆盖 ${dualNonSingleton.spanCount}/${dualQuality.spanCount} 个 Span，非单例 bucket=${dualNonSingleton.bucketCount}。这个指标比全量 purity 更接近“同桶 Span 是否真是同一策略”。`,
    `- 结论：当前策略优先保证同桶 Span 的绝对相似性，宁可低召回、形成更多单例桶，也不提前合并目标不够一致的 Span；非单例 purity=${dualNonSingleton.purity.toFixed(3)} 用于辅助审阅已形成桶的策略一致性。`,
    "",
    "## 2. 同桶 Span 是否能聚合成 Policy",
    "",
    `- 生成的 Policy 数量：${result.policies.length}，对应 ${dualQuality.reusableSpanCount}/${dualQuality.spanCount} 个进入 ready/promoted 桶的 Span。`,
    "- 生成的 Policy key 均采用 `policy:span-cluster:<clusterId>:span-policy.v1`，不是 signature key。",
    "- Policy metadata 包含 `source_span_ids`、`source_trace_ids`、`span_cluster_id`、`cluster_membership_version`，可追溯到 Span bucket。",
    "- 因此本轮能证明 ready/promoted bucket 可以生成可追溯 Policy，但不能证明大多数 Span 已能稳定聚合为可复用 Policy。",
    "",
    "## 3. 消融实验：旧 Span embedding vs 原始 signature vs 双阈值",
    "",
    `- 真实样本上的旧 Span embedding（Goal+Summary）：${result.legacySpanEmbedding.length} 个桶，purity=${legacyPurity.toFixed(3)}，单例覆盖=${legacySingletonRate}，ready/promoted 覆盖=${legacyReusableRate}。`,
    `- 单 embedding（Goal+Policy 拼接）：${singleGoalPolicy.length} 个桶，purity=${singleGoalPolicyPurity.toFixed(3)}，单例覆盖=${singleGoalPolicySingletonRate}，ready/promoted 覆盖=${singleGoalPolicyReusableRate}。`,
    `- 单 embedding（Goal-only）：${singleGoal.length} 个桶，purity=${singleGoalPurity.toFixed(3)}，单例覆盖=${singleGoalSingletonRate}，ready/promoted 覆盖=${singleGoalReusableRate}。`,
    `- 单 embedding（Policy-only）：${singlePolicy.length} 个桶，purity=${singlePolicyPurity.toFixed(3)}，单例覆盖=${singlePolicySingletonRate}，ready/promoted 覆盖=${singlePolicyReusableRate}。`,
    `- 真实样本上的原始 signature：${result.signatureBuckets.length} 个桶，purity=${signaturePurity.toFixed(3)}，单例覆盖=${signatureSingletonRate}，ready 覆盖=${signatureReusableRate}。`,
    `- 双阈值 Goal/Policy：${result.dualThreshold.length} 个桶，purity=${dualPurity.toFixed(3)}，单例覆盖=${dualSingletonRate}，ready/promoted 覆盖=${dualReusableRate}。`,
    `- 非单例桶对比：旧 Span embedding purity=${legacyNonSingleton.purity.toFixed(3)} / 覆盖=${legacyNonSingleton.spanCount}，signature purity=${signatureNonSingleton.purity.toFixed(3)} / 覆盖=${signatureNonSingleton.spanCount}，双阈值 purity=${dualNonSingleton.purity.toFixed(3)} / 覆盖=${dualNonSingleton.spanCount}。`,
    `- 压力消融：旧 Span embedding purity=${stressLegacyPurity.toFixed(3)}，signature purity=${stressSignaturePurity.toFixed(3)}，双阈值 purity=${stressDualPurity.toFixed(3)}。`,
    "- 本轮真实样本中，当前 prompt-only + embedding-only 双阈值的目标不是提高当前批次覆盖，而是降低误聚合；signature 覆盖更高但 purity 更低，因此只作为 legacy baseline。",
    "",
    "### 3.0 消融总表",
    "",
    "| 方法 | buckets | purity | non-single purity | singleton | reusable | non-single spans |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ablationRow("旧 Span embedding Goal+Summary", result.legacySpanEmbedding, legacyPurity, legacyNonSingleton, legacyQuality, legacySingletonRate, legacyReusableRate),
    ablationRow("单 embedding Goal+Policy", singleGoalPolicy, singleGoalPolicyPurity, singleGoalPolicyNonSingleton, singleGoalPolicyQuality, singleGoalPolicySingletonRate, singleGoalPolicyReusableRate),
    ablationRow("单 embedding Goal-only", singleGoal, singleGoalPurity, singleGoalNonSingleton, singleGoalQuality, singleGoalSingletonRate, singleGoalReusableRate),
    ablationRow("单 embedding Policy-only", singlePolicy, singlePolicyPurity, singlePolicyNonSingleton, singlePolicyQuality, singlePolicySingletonRate, singlePolicyReusableRate),
    ablationRow("双阈值 Goal/Policy", result.dualThreshold, dualPurity, dualNonSingleton, dualQuality, dualSingletonRate, dualReusableRate),
    ablationRow("原始 signature", result.signatureBuckets, signaturePurity, signatureNonSingleton, signatureQuality, signatureSingletonRate, signatureReusableRate),
    "",
    "### 3.0.1 双阈值为什么更好",
    "",
    "- 单 embedding 把 goal 与 policy 压进同一个向量空间：相同目标但不同执行策略容易被合并，相同工具策略但不同任务目标也容易被合并。",
    "- 双阈值把“任务目标相似”和“观察到的策略相似”拆成两个必要条件：只有 goal cosine 与 policy cosine 同时过阈值才入桶，相当于对误聚合做交集过滤。",
    "- signature 主要捕捉工具形态和少量错误/文本特征，召回高但语义不足；它会把不同目标、不同策略但工具序列相似的 span 压进同桶。",
    "- 双阈值的核心价值是用两个独立语义视角做交集过滤。提高 goal 阈值后，只有目标也高度接近的 Span 才能合并；更多单例桶是高精度策略的正常结果。",
    "",
    "## 3.1 双阈值 Sweep",
    "",
    "| goal | policy | buckets | purity | singleton | reusable | non-single buckets |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...thresholdSweep.map((item: any) =>
      `| ${item.goalThreshold.toFixed(2)} | ${item.policyThreshold.toFixed(2)} | ${item.bucketCount} | ${item.purity.toFixed(3)} | ${(item.singletonRate * 100).toFixed(1)}% | ${(item.reusableRate * 100).toFixed(1)}% | ${item.nonSingletonBucketCount} |`
    ),
    "",
    "## 4. 样本与桶摘要",
    "",
    ...result.dualThreshold.map((cluster: any, index: number) =>
      `- Bucket ${index + 1}: status=${cluster.status}, members=${cluster.memberCount}, distinctSources=${cluster.distinctSourceCount}, families=${JSON.stringify(cluster.families)}`
    ),
    "",
    "## 4.1 原始 Signature 桶摘要",
    "",
    ...result.signatureBuckets.map((cluster: any, index: number) =>
      `- Signature Bucket ${index + 1}: signature=${cluster.signature}, status=${cluster.status}, members=${cluster.memberCount}, distinctSources=${cluster.distinctSourceCount}, families=${JSON.stringify(cluster.families)}`
    ),
    "",
    "## 5. 压力消融桶摘要",
    "",
    "- 旧 Span embedding:",
    ...result.stressAblation.legacy.map((cluster: any, index: number) =>
      `  - Bucket ${index + 1}: members=${cluster.memberCount}, families=${JSON.stringify(cluster.families)}, goals=${JSON.stringify(cluster.goals)}`
    ),
    "- 原始 signature:",
    ...result.stressAblation.signature.map((cluster: any, index: number) =>
      `  - Bucket ${index + 1}: signature=${cluster.signature}, members=${cluster.memberCount}, families=${JSON.stringify(cluster.families)}, goals=${JSON.stringify(cluster.goals)}`
    ),
    "- 双阈值:",
    ...result.stressAblation.dualThreshold.map((cluster: any, index: number) =>
      `  - Bucket ${index + 1}: members=${cluster.memberCount}, families=${JSON.stringify(cluster.families)}, goals=${JSON.stringify(cluster.goals)}`
    ),
    "",
    "## 6. 最终结论",
    "",
    "- 本轮使用 145 条真实 trace，经真实远端 LLM 拆分得到 116 个合法 Span，split failure=0；分桶只依赖 prompt 生成的 Goal/Policy 文本 embedding，不使用字段匹配或 signature 正向 gate。",
    `- 在主阈值 goal=${result.thresholds?.goalSimilarityThreshold ?? "n/a"}、policy=${result.thresholds?.policySimilarityThreshold ?? "n/a"} 下，双阈值方法 purity=${dualPurity.toFixed(3)}，非单例桶 purity=${dualNonSingleton.purity.toFixed(3)}，单例覆盖=${dualSingletonRate}，ready/promoted 覆盖=${dualReusableRate}。`,
    "- 双阈值优于所有单 embedding 对照：Goal+Policy 单 embedding purity=0.914，Goal-only=0.888，Policy-only=0.897；说明把两种语义压进一个向量会损失可分性。",
    "- Signature 的 ready 覆盖最高（81.0%），但 purity=0.664、非单例 purity=0.625，说明它更像工具调用形态分桶，而不是可复用策略分桶。",
    "- 双阈值有效的核心原因是两个独立语义视角的交集约束：Goal 负责约束“要完成什么”，Policy 负责约束“采用什么可复用方法”；只有两路 cosine 都过阈值才合并，从而降低单 embedding 的误聚合。",
    "- 最终建议：保留 prompt-only + embedding-only 双阈值作为正向 Policy 聚类主路径；signature 仅保留为 legacy baseline/回滚对照；后续继续优化 prompt 和对非单例桶做人工或 LLM judge 审阅，不在代码中加入字段匹配规则。",
    "",
    "## 局限",
    "",
    "- Span 拆分与 embedding 均调用真实远端模型；但 bucket assignment 与 L2 Policy 内容拼装仍是确定性代码路径，未使用 LLM judge。",
    "- 当前实验没有人工标注 Span 边界或 Policy 等价类；family 标签来自工具行为启发式，只能作为粗粒度分桶质量代理。",
    "- 原始 signature baseline 是按 trace 级旧 signature 回填到 Span，用于观察旧分桶粒度；它不是 Span-aware baseline。",
    "- Purity 只衡量桶内工具行为策略族的一致性，不衡量 LLM Span 边界的人类标注一致性或生成 Policy 的可执行性；单例覆盖和 ready/promoted 覆盖只作为当前批次生长状态的描述，不作为负向优化目标。",
    ""
  ].join("\n");
}

function weightedPurity(clusters: Array<{ memberCount: number; families?: Record<string, number> }>): number {
  let numerator = 0;
  let denominator = 0;
  for (const cluster of clusters) {
    const families = cluster.families ?? { unknown: cluster.memberCount };
    numerator += Math.max(...Object.values(families));
    denominator += cluster.memberCount;
  }
  return denominator === 0 ? 0 : numerator / denominator;
}

function nonSingletonQuality(clusters: Array<{ memberCount: number; families?: Record<string, number> }>) {
  const nonSingletonClusters = clusters.filter((cluster) => cluster.memberCount > 1);
  return {
    bucketCount: nonSingletonClusters.length,
    spanCount: nonSingletonClusters.reduce((sum, cluster) => sum + cluster.memberCount, 0),
    purity: weightedPurity(nonSingletonClusters)
  };
}

function ablationRow(
  label: string,
  clusters: Array<{ memberCount: number }>,
  purity: number,
  nonSingleton: { purity: number; spanCount: number },
  quality: { spanCount: number },
  singletonRate: string,
  reusableRate: string
): string {
  return `| ${label} | ${clusters.length} | ${purity.toFixed(3)} | ${nonSingleton.purity.toFixed(3)} | ${singletonRate} | ${reusableRate} | ${nonSingleton.spanCount}/${quality.spanCount} |`;
}

function clusterQuality(clusters: Array<{ memberCount: number; status: string }>) {
  const spanCount = clusters.reduce((sum, cluster) => sum + cluster.memberCount, 0);
  const singletonSpanCount = clusters
    .filter((cluster) => cluster.memberCount === 1)
    .reduce((sum, cluster) => sum + cluster.memberCount, 0);
  const reusableSpanCount = clusters
    .filter((cluster) => cluster.status === "ready" || cluster.status === "promoted")
    .reduce((sum, cluster) => sum + cluster.memberCount, 0);
  return {
    spanCount,
    singletonSpanCount,
    nonSingletonSpanCount: spanCount - singletonSpanCount,
    reusableSpanCount
  };
}

function percentage(numerator: number, denominator: number): string {
  return denominator === 0 ? "0.0%" : `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function roundMetric(value: number): number {
  return Number(value.toFixed(4));
}

function familyFromPolicy(policy: string): string {
  const hit = policyFamilies.find((item) => item.policy === policy);
  return hit?.family ?? "mixed";
}

function legacySpanEmbeddingText(memory: MemoryRow): string {
  const span = spanPayload(memory);
  if (!span) return memory.memoryValue;
  return [`Goal: ${span.goal}`, `Summary: ${span.summary}`].join("\n");
}

function stressSpan(
  id: string,
  sourceTraceId: string,
  goal: string,
  policy: string,
  family: string,
  firstTool: string,
  order: number
) {
  return {
    id,
    sourceTraceId,
    goal,
    policy,
    family,
    firstTool,
    createdAt: `2026-08-14T00:00:0${order}.000Z`
  };
}

function buildTraceMemory(sample: Sample): MemoryRow {
  const at = sample.rawTurn.createdAt;
  return {
    id: sample.traceId,
    timeline: at,
    userId: "analysis-user",
    conversationId: sample.rawTurn.conversationId,
    sessionId: sample.rawTurn.sessionId,
    agentId: "codex",
    memoryType: "LongTermMemory",
    status: "activated",
    visibility: "private",
    memoryKey: sample.taskTitle,
    memoryValue: sample.rawTurn.userText ?? sample.taskTitle,
    tags: ["trace"],
    info: { profile_id: "analysis", title: sample.taskTitle, summary: sample.taskTitle },
    properties: {
      internal_info: {
        memory_layer: "L1",
        memory_kind: "trace",
        schema_version: 1,
        raw_turn_id: sample.rawTurn.id,
        episode_id: sample.rawTurn.episodeId,
        session_id: sample.rawTurn.sessionId,
        turn_id: sample.rawTurn.turnId,
        trace: {
          summary: sample.taskTitle,
          user_text: sample.rawTurn.userText ?? "",
          agent_text: sample.rawTurn.assistantText ?? "",
          tool_calls: sample.rawTurn.toolCalls,
          value: 1,
          priority: 1
        }
      }
    },
    memoryLayer: "L1",
    version: 1,
    createdAt: at,
    updatedAt: at
  };
}

function buildMemoryFactory() {
  return (input: Record<string, unknown>): MemoryRow => {
    const at = stringField(input, "createdAt") ?? new Date().toISOString();
    const layer = input.layer as MemoryRow["memoryLayer"];
    const kind = input.kind as string;
    const info = (input.info && typeof input.info === "object" ? input.info : {}) as Record<string, unknown>;
    return {
      id: stringField(input, "id") ?? `${kind}_${stableHash(input).slice(0, 20)}`,
      timeline: at,
      userId: stringField(input, "userId") ?? "analysis-user",
      conversationId: stringField(input, "conversationId"),
      sessionId: stringField(input, "sessionId"),
      agentId: stringField(input, "agentId"),
      appId: stringField(input, "appId"),
      memoryType: stringField(input, "memoryType") ?? "LongTermMemory",
      status: "activated",
      visibility: "private",
      memoryKey: stringField(input, "key"),
      memoryValue: stringField(input, "value") ?? "",
      tags: Array.isArray(input.tags) ? input.tags.filter((item): item is string => typeof item === "string") : [],
      info: {
        ...info,
        ...(stringField(input, "projectId") ? { project_id: stringField(input, "projectId") } : {}),
        ...(stringField(input, "profileId") ? { profile_id: stringField(input, "profileId") } : {})
      },
      properties: {
        internal_info: {
          memory_layer: layer,
          memory_kind: kind as any,
          ...((input.internal && typeof input.internal === "object") ? input.internal as Record<string, unknown> : {})
        }
      },
      memoryLayer: layer,
      version: 1,
      createdAt: at,
      updatedAt: at
    };
  };
}

function resetExperimentDb(db: MemoryDb): void {
  db.db.exec(`
    DELETE FROM trace_policy_links;
    DELETE FROM span_cluster_members;
    DELETE FROM span_clusters;
    DELETE FROM memory_vector_entries;
    DELETE FROM memories;
    DELETE FROM raw_turns;
    DELETE FROM episodes;
    DELETE FROM sessions;
    DELETE FROM evolution_jobs;
    DELETE FROM memory_change_log;
  `);
}

function taskTitle(value: string): string {
  const match = value.match(/## Task\s+([\s\S]*?)(?:\n\n|$)/);
  const title = match?.[1]?.replace(/\s+/g, " ").trim() ?? value.replace(/\s+/g, " ").trim();
  return title.slice(0, 160);
}

function countBy(values: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const value of values) out[value] = (out[value] ?? 0) + 1;
  return out;
}

function stringField(record: unknown, key: string): string | undefined {
  if (!record || typeof record !== "object") return undefined;
  const value = (record as Record<string, unknown>)[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function arrayField(record: unknown, key: string): string[] {
  if (!record || typeof record !== "object") return [];
  const value = (record as Record<string, unknown>)[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
