import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { buildSpanTrajectory } from "../src/service/evolution/span-trajectory.js";
import { stableStringify } from "../src/utils/id.js";
import {
  loadArtifactTraceCases,
  traceLengthBucket,
  type ArtifactTraceCase,
  type TraceLengthBucket
} from "./span-v2-artifact-traces.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const ARTIFACTS_ROOT = join(REPO_ROOT, "artifacts");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const OUT_JSON = join(OUT_DIR, "span_v2_cleaning_stats.json");
const OUT_MD = join(OUT_DIR, "span_v2_cleaning_stats.md");
const MIN_TOOL_CALLS = positiveIntegerEnv("SPAN_V2_CLEANING_STATS_MIN_TOOL_CALLS") ?? 4;

function main(): void {
  mkdirSync(OUT_DIR, { recursive: true });
  const traces = loadArtifactTraceCases({
    artifactsRoot: ARTIFACTS_ROOT,
    minToolCalls: MIN_TOOL_CALLS
  });
  const rows = traces.map(traceStats);
  const output = {
    generatedAt: new Date().toISOString(),
    dataSources: [
      "artifacts/memory-v10-training-20260804/memory.sqlite",
      "artifacts/memory-v10-test-20260804/*/session.jsonl"
    ],
    minToolCalls: MIN_TOOL_CALLS,
    traceCount: rows.length,
    distribution: {
      source: Object.fromEntries((["training_sqlite", "test_jsonl"] as const).map((source) => [
        source,
        rows.filter((row) => row.sourceKind === source).length
      ])),
      lengthBucket: Object.fromEntries((["small", "medium", "large", "xlarge"] as const).map((bucket) => [
        bucket,
        rows.filter((row) => row.lengthBucket === bucket).length
      ]))
    },
    overall: summarize(rows),
    byBucket: Object.fromEntries((["small", "medium", "large", "xlarge"] as const).map((bucket) => [
      bucket,
      summarize(rows.filter((row) => row.lengthBucket === bucket))
    ])),
    topCompression: [...rows]
      .sort((a, b) => b.eventReduction - a.eventReduction)
      .slice(0, 12)
  };
  writeFileSync(OUT_JSON, JSON.stringify(output, null, 2));
  writeFileSync(OUT_MD, renderReport(output));
}

function traceStats(trace: ArtifactTraceCase) {
  const uncompressed = buildSpanTrajectory(trace.toolCalls, { maxRepeatedCalls: 1 });
  const compressed = buildSpanTrajectory(trace.toolCalls);
  const uncompressedJsonChars = stableStringify(uncompressed).length;
  const compressedJsonChars = stableStringify(compressed).length;
  const uncompressedEvidenceChars = evidenceChars(uncompressed);
  const compressedEvidenceChars = evidenceChars(compressed);
  const uncompressedPromptChars = promptChars(trace, uncompressed);
  const compressedPromptChars = promptChars(trace, compressed);
  return {
    traceId: trace.traceId,
    sourceKind: trace.sourceKind,
    lengthBucket: traceLengthBucket(trace.toolCalls.length),
    rawToolCalls: trace.toolCalls.length,
    uncompressedEvents: uncompressed.length,
    compressedEvents: compressed.length,
    repeatedGroups: compressed.filter((event) => event.repeated).length,
    callsInsideRepeatedGroups: compressed
      .filter((event) => event.repeated)
      .reduce((sum, event) => sum + event.callCount, 0),
    uncompressedJsonChars,
    compressedJsonChars,
    uncompressedEvidenceChars,
    compressedEvidenceChars,
    uncompressedPromptChars,
    compressedPromptChars,
    eventReduction: reduction(uncompressed.length, compressed.length),
    jsonReduction: reduction(uncompressedJsonChars, compressedJsonChars),
    evidenceReduction: reduction(uncompressedEvidenceChars, compressedEvidenceChars),
    promptReduction: reduction(uncompressedPromptChars, compressedPromptChars)
  };
}

function summarize(rows: Array<ReturnType<typeof traceStats>>) {
  const rawToolCalls = sum(rows, "rawToolCalls");
  const uncompressedEvents = sum(rows, "uncompressedEvents");
  const compressedEvents = sum(rows, "compressedEvents");
  const uncompressedJsonChars = sum(rows, "uncompressedJsonChars");
  const compressedJsonChars = sum(rows, "compressedJsonChars");
  const uncompressedEvidenceChars = sum(rows, "uncompressedEvidenceChars");
  const compressedEvidenceChars = sum(rows, "compressedEvidenceChars");
  const uncompressedPromptChars = sum(rows, "uncompressedPromptChars");
  const compressedPromptChars = sum(rows, "compressedPromptChars");
  return {
    traceCount: rows.length,
    rawToolCalls,
    uncompressedEvents,
    compressedEvents,
    eventReduction: round(reduction(uncompressedEvents, compressedEvents)),
    repeatedGroups: sum(rows, "repeatedGroups"),
    callsInsideRepeatedGroups: sum(rows, "callsInsideRepeatedGroups"),
    uncompressedJsonChars,
    compressedJsonChars,
    jsonReduction: round(reduction(uncompressedJsonChars, compressedJsonChars)),
    uncompressedEvidenceChars,
    compressedEvidenceChars,
    evidenceReduction: round(reduction(uncompressedEvidenceChars, compressedEvidenceChars)),
    uncompressedPromptChars,
    compressedPromptChars,
    promptReduction: round(reduction(uncompressedPromptChars, compressedPromptChars)),
    avgEventReduction: round(rows.reduce((total, row) => total + row.eventReduction, 0) / Math.max(1, rows.length)),
    avgPromptReduction: round(rows.reduce((total, row) => total + row.promptReduction, 0) / Math.max(1, rows.length))
  };
}

function renderReport(output: {
  generatedAt: string;
  dataSources: string[];
  minToolCalls: number;
  traceCount: number;
  distribution: {
    source: Record<string, number>;
    lengthBucket: Record<string, number>;
  };
  overall: ReturnType<typeof summarize>;
  byBucket: Record<string, ReturnType<typeof summarize>>;
  topCompression: Array<ReturnType<typeof traceStats>>;
}): string {
  const lines = [
    "# Span Trace 清洗压缩全量统计",
    "",
    `生成时间：${output.generatedAt}`,
    "",
    "## 数据来源",
    "",
    ...output.dataSources.map((source) => `- ${source}`),
    "",
    `统计范围：tool calls >= ${output.minToolCalls} 的 artifacts 候选 trace，共 ${output.traceCount} 条。`,
    "",
    `来源分布：training=${output.distribution.source.training_sqlite}，test=${output.distribution.source.test_jsonl}`,
    `长度分布：small=${output.distribution.lengthBucket.small}，medium=${output.distribution.lengthBucket.medium}，large=${output.distribution.lengthBucket.large}，xlarge=${output.distribution.lengthBucket.xlarge}`,
    "",
    "## 指标含义",
    "",
    "- `uncompressedEvents`：不合并重复 tool call 时的 structured event 数，基本等于原始 tool call 数。",
    "- `compressedEvents`：连续相同 tool/action/success/errorClass 合并后的 structured event 数。",
    "- `jsonChars`：structured trajectory 序列化后的字符数，反映轨迹主体输入规模。",
    "- `evidenceChars`：event 中证据片段总字符数，反映 tool result preview 规模。",
    "- `promptChars`：用户任务摘要字段 + structured trajectory 的估算 prompt 字符数，不等于 tokenizer token，但可用于比较压缩前后相对成本。",
    "- 降幅是按总量计算：`(压缩前 - 压缩后) / 压缩前`。",
    "",
    "## 总体结果",
    "",
    "| trace | raw calls | events before | events after | event reduction | prompt chars before | prompt chars after | prompt reduction |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    summaryRow("overall", output.overall),
    "",
    "## 分层结果",
    "",
    "| bucket | traces | raw calls | events before | events after | event reduction | prompt reduction |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...(["small", "medium", "large", "xlarge"] as const).map((bucket) => bucketRow(bucket, output.byBucket[bucket]!)),
    "",
    "## 压缩收益最高样例",
    "",
    "| trace | source | bucket | raw calls | events after | event reduction | prompt reduction |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ...output.topCompression.map((row) =>
      `| ${row.traceId} | ${row.sourceKind} | ${row.lengthBucket} | ${row.rawToolCalls} | ${row.compressedEvents} | ${pct(row.eventReduction)} | ${pct(row.promptReduction)} |`
    ),
    "",
    "## 结论",
    "",
    `- 在全部 ${output.traceCount} 条 artifacts 候选 trace 上，压缩后 event 总量下降 ${pct(output.overall.eventReduction)}，估算 prompt 字符下降 ${pct(output.overall.promptReduction)}。`,
    "- 该数据说明压缩主要降低 LLM 输入规模和重复噪音；它不是 span 质量指标，也不能证明压缩后 span 一定更准。",
    "- 压缩仍保留原始 range/callCount，因此不改变 LLM 输出 span 边界的原始 tool-call index 语义。",
    ""
  ];
  return lines.join("\n");
}

function summaryRow(label: string, value: ReturnType<typeof summarize>): string {
  return `| ${label} | ${value.rawToolCalls} | ${value.uncompressedEvents} | ${value.compressedEvents} | ${pct(value.eventReduction)} | ${value.uncompressedPromptChars} | ${value.compressedPromptChars} | ${pct(value.promptReduction)} |`;
}

function bucketRow(label: string, value: ReturnType<typeof summarize>): string {
  return `| ${label} | ${value.traceCount} | ${value.rawToolCalls} | ${value.uncompressedEvents} | ${value.compressedEvents} | ${pct(value.eventReduction)} | ${pct(value.promptReduction)} |`;
}

function promptChars(trace: ArtifactTraceCase, trajectory: unknown[]): number {
  return stableStringify({
    traceId: trace.traceId,
    sourceKind: trace.sourceKind,
    userRequest: trace.userText.slice(0, 1_500),
    assistantFinalAnswer: trace.assistantText.slice(0, 1_000),
    structuredTrajectory: trajectory
  }).length;
}

function evidenceChars(events: Array<{ evidence?: string }>): number {
  return events.reduce((total, event) => total + (event.evidence?.length ?? 0), 0);
}

function reduction(before: number, after: number): number {
  if (before <= 0) return 0;
  return (before - after) / before;
}

function sum<T extends Record<string, unknown>>(rows: T[], key: keyof T): number {
  return rows.reduce((total, row) => total + Number(row[key] ?? 0), 0);
}

function positiveIntegerEnv(name: string): number | undefined {
  const value = Number(process.env[name]);
  return Number.isInteger(value) && value > 0 ? value : undefined;
}

function round(value: number): number {
  return Number(value.toFixed(4));
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

main();
