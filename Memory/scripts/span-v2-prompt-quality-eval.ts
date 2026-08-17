import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  createLlmClient,
  DEFAULT_MEMMY_CONFIG,
  MemoryDb,
  type LlmClient,
  type MemoryRow,
  type RawTurnRecord
} from "../src/index.js";
import { Repositories, type EvolutionJobRecord } from "../src/storage/repositories.js";
import { BigTurnSpanPipeline } from "../src/service/evolution/big-turn-span-pipeline.js";
import { spanPayload } from "../src/service/evolution/span-model.js";
import { stableHash } from "../src/utils/id.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const SOURCE_DB = join(OUT_DIR, "span_v2_experiment.sqlite");
const RESULT_JSON = join(OUT_DIR, "span_v2_experiment_results.json");
const WORK_DB = join(OUT_DIR, "span_v2_prompt_quality_eval.sqlite");
const OUT_JSON = join(OUT_DIR, "span_v2_prompt_quality_eval.json");
const OUT_MD = join(OUT_DIR, "span_v2_prompt_quality_eval.md");

async function main(): Promise<void> {
  mkdirSync(OUT_DIR, { recursive: true });
  const oldResult = JSON.parse(readFileSync(RESULT_JSON, "utf8")) as {
    spans: Array<{
      id: string;
      sourceTraceId: string;
      range: [number, number];
      toolCallCount: number;
      goal: string;
      policy: string;
      summary: string;
      family: string;
    }>;
  };
  const candidateTraceIds = selectLongSpanTraceIds(oldResult.spans);
  const sourceDb = new MemoryDb({ path: SOURCE_DB });
  const sourceRepos = new Repositories(sourceDb.db);
  const workDb = new MemoryDb({ path: WORK_DB });
  resetWorkDb(workDb);
  const repos = new Repositories(workDb.db);
  const buildMemory = buildMemoryFactory();
  const llm = createEvalLlm();
  const splitFailures: Array<{ traceId: string; error: string }> = [];

  for (const [index, traceId] of candidateTraceIds.entries()) {
    const sourceTrace = sourceRepos.memories.get(traceId);
    if (!sourceTrace) continue;
    const rawTurnId = typeof sourceTrace.properties.internal_info.raw_turn_id === "string"
      ? sourceTrace.properties.internal_info.raw_turn_id
      : undefined;
    const rawTurn = rawTurnId ? sourceRepos.runtime.getRawTurn(rawTurnId) : undefined;
    if (!rawTurn) continue;
    repos.runtime.createSession({
      id: rawTurn.sessionId,
      userId: rawTurn.userId,
      source: "codex",
      profileId: "analysis",
      status: "closed",
      meta: {},
      openedAt: rawTurn.createdAt,
      lastSeenAt: rawTurn.createdAt,
      closedAt: rawTurn.createdAt,
      updatedAt: rawTurn.createdAt
    });
    repos.runtime.createEpisode({
      id: rawTurn.episodeId,
      sessionId: rawTurn.sessionId,
      userId: rawTurn.userId,
      status: "closed",
      title: sourceTrace.memoryKey ?? traceId,
      summary: sourceTrace.memoryKey ?? traceId,
      l1MemoryIds: [traceId],
      rawTurnIds: [rawTurn.id],
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
      openedAt: rawTurn.createdAt,
      closedAt: rawTurn.createdAt,
      updatedAt: rawTurn.createdAt
    });
    repos.runtime.insertRawTurn(rawTurn);
    repos.memories.insert(copyTrace(sourceTrace));
    const pipeline = new BigTurnSpanPipeline({
      repos,
      llm,
      buildMemory,
      enqueueJob: (input) => repos.runtime.enqueueJob({
        id: `job_${stableHash({ input, index }).slice(0, 16)}`,
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
        createdAt: input.createdAt ?? rawTurn.createdAt,
        updatedAt: input.createdAt ?? rawTurn.createdAt
      }),
      namespaceIdFromMemory: () => "analysis-user:codex:analysis",
      embedAfterCapture: () => false
    });
    try {
      await pipeline.splitAndStore({
        id: `job_split_eval_${traceId}`,
        jobType: "span_big_turn",
        status: "queued",
        userId: sourceTrace.userId,
        sessionId: sourceTrace.sessionId,
        episodeId: rawTurn.episodeId,
        targetMemoryId: traceId,
        payload: { rawTurnId: rawTurn.id, rTask: 1 },
        attempts: 0,
        maxAttempts: 3,
        createdAt: rawTurn.createdAt,
        updatedAt: rawTurn.createdAt
      });
    } catch (error) {
      splitFailures.push({ traceId, error: error instanceof Error ? error.message : String(error) });
    }
  }

  const newSpans = repos.memories.list({ memoryLayer: "L1", status: "activated" }, 10000)
    .filter((memory) => spanPayload(memory))
    .map((memory) => {
      const span = spanPayload(memory)!;
      return {
        id: memory.id,
        sourceTraceId: span.source_trace_id,
        range: [span.tool_call_start, span.tool_call_end] as [number, number],
        toolCallCount: span.tool_call_count,
        goal: span.goal,
        policy: span.policy,
        summary: span.summary,
        mixedStage: isMixedStage(span.goal, span.policy)
      };
    });
  const oldSelected = oldResult.spans
    .filter((span) => candidateTraceIds.includes(span.sourceTraceId))
    .map((span) => ({
      ...span,
      mixedStage: isMixedStage(span.goal, span.policy)
    }));
  const result = {
    generatedAt: new Date().toISOString(),
    llm: publicModelConfig(llm.config),
    traceCount: candidateTraceIds.length,
    traceIds: candidateTraceIds,
    splitFailures,
    old: summarize(oldSelected),
    nextPrompt: summarize(newSpans),
    oldSpans: oldSelected,
    newSpans
  };
  writeFileSync(OUT_JSON, JSON.stringify(result, null, 2));
  writeFileSync(OUT_MD, renderReport(result));
  sourceDb.close();
  workDb.close();
}

function selectLongSpanTraceIds(spans: Array<{ sourceTraceId: string; toolCallCount: number }>): string[] {
  const traceIds = [...new Set(
    spans
      .filter((span) => span.toolCallCount >= 50)
      .sort((a, b) => b.toolCallCount - a.toolCallCount)
      .map((span) => span.sourceTraceId)
  )];
  return traceIds.slice(0, positiveIntegerEnv("SPAN_V2_PROMPT_EVAL_TRACE_LIMIT") ?? 8);
}

function summarize(spans: Array<{ toolCallCount: number; mixedStage: boolean }>) {
  const longThreshold = 50;
  return {
    spanCount: spans.length,
    avgToolCalls: round(spans.reduce((sum, span) => sum + span.toolCallCount, 0) / Math.max(1, spans.length)),
    maxToolCalls: Math.max(0, ...spans.map((span) => span.toolCallCount)),
    longSpanCount: spans.filter((span) => span.toolCallCount >= longThreshold).length,
    longSpanRate: round(spans.filter((span) => span.toolCallCount >= longThreshold).length / Math.max(1, spans.length)),
    mixedStageCount: spans.filter((span) => span.mixedStage).length,
    mixedStageRate: round(spans.filter((span) => span.mixedStage).length / Math.max(1, spans.length))
  };
}

function isMixedStage(goal: string, policy: string): boolean {
  const text = `${goal}\n${policy}`.toLowerCase();
  const signals = [
    /gather|search|fetch|research|collect/u,
    /inspect|schema|load|parse|read/u,
    /generate|create|write|build|populate/u,
    /debug|fix|repair|error|traceback|rerun/u,
    /verify|validate|check|confirm|extract text/u
  ];
  return signals.filter((signal) => signal.test(text)).length >= 3;
}

function renderReport(result: any): string {
  return [
    "# Span Prompt 质量小样本重跑评估",
    "",
    `生成时间：${result.generatedAt}`,
    "",
    "## 设置",
    "",
    `- Trace 数：${result.traceCount}`,
    `- LLM：provider=${result.llm.provider}，model=${result.llm.model}，endpoint=${result.llm.endpoint}`,
    "- 只重跑旧结果中包含 >=50 tool calls 长 Span 的 trace；未重新计算 embedding，也未重新聚类。",
    "- mixedStage 是启发式指标：goal/policy 中同时出现资料搜集、数据检查、生成、调试、验证等 3 类以上阶段信号时记为混合阶段 Span。",
    "",
    "## 指标对比",
    "",
    "| 版本 | spans | avg tool calls | max tool calls | long spans >=50 | mixed-stage spans |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    `| 旧 prompt 结果 | ${result.old.spanCount} | ${result.old.avgToolCalls} | ${result.old.maxToolCalls} | ${result.old.longSpanCount} (${pct(result.old.longSpanRate)}) | ${result.old.mixedStageCount} (${pct(result.old.mixedStageRate)}) |`,
    `| 新 prompt 重跑 | ${result.nextPrompt.spanCount} | ${result.nextPrompt.avgToolCalls} | ${result.nextPrompt.maxToolCalls} | ${result.nextPrompt.longSpanCount} (${pct(result.nextPrompt.longSpanRate)}) | ${result.nextPrompt.mixedStageCount} (${pct(result.nextPrompt.mixedStageRate)}) |`,
    "",
    "## 新 Prompt Span 样例",
    "",
    ...result.newSpans.slice(0, 20).map((span: any) =>
      `- ${span.id}: trace=${span.sourceTraceId}, range=${span.range.join("-")}, calls=${span.toolCallCount}, mixed=${span.mixedStage}, goal=${span.goal}`
    ),
    "",
    "## 失败",
    "",
    result.splitFailures.length === 0
      ? "- split failure=0"
      : result.splitFailures.map((failure: any) => `- ${failure.traceId}: ${failure.error}`).join("\n"),
    ""
  ].join("\n");
}

function copyTrace(memory: MemoryRow): MemoryRow {
  return {
    ...memory,
    tags: [...memory.tags],
    info: { ...memory.info },
    properties: JSON.parse(JSON.stringify(memory.properties)) as MemoryRow["properties"]
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

function resetWorkDb(db: MemoryDb): void {
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
    DELETE FROM runtime_kv;
  `);
}

function createEvalLlm(): LlmClient {
  return createLlmClient({
    ...DEFAULT_MEMMY_CONFIG.evolution,
    provider: "openai_compatible",
    vendor: "deepseek",
    endpoint: process.env.SPAN_V2_LLM_ENDPOINT ?? "https://api-int.memtensor.cn/v1",
    model: process.env.SPAN_V2_LLM_MODEL ?? "deepseek-v4-pro",
    apiKey: requiredSecret("SPAN_V2_LLM_API_KEY", "MEMMY_EVOLUTION_API_KEY"),
    enableThinking: false,
    temperature: 0.2,
    maxTokens: 4096,
    timeoutMs: 180_000,
    maxRetries: 2,
    malformedRetries: 2
  });
}

function requiredSecret(...names: string[]): string {
  for (const name of names) {
    const value = process.env[name]?.trim();
    if (value) return value;
  }
  throw new Error(`Missing LLM API key. Set one of: ${names.join(", ")}`);
}

function publicModelConfig(config: { provider: string; endpoint?: string; model?: string }) {
  return {
    provider: config.provider,
    endpoint: config.endpoint,
    model: config.model
  };
}

function positiveIntegerEnv(name: string): number | undefined {
  const value = Number(process.env[name]);
  return Number.isInteger(value) && value > 0 ? value : undefined;
}

function stringField(input: Record<string, unknown>, key: string): string | undefined {
  const value = input[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function round(value: number): number {
  return Number(value.toFixed(2));
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

await main();
