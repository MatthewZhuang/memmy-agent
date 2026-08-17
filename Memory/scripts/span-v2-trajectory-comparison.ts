import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
  MemoryDb,
  type MemoryRow,
  type RawTurnRecord
} from "../src/index.js";
import { Repositories } from "../src/storage/repositories.js";
import { buildSpanTrajectory, type SpanTrajectoryEvent } from "../src/service/evolution/span-trajectory.js";
import { eventsForToolCallRange } from "./span-v2-segmentation.js";
import { stableStringify } from "../src/utils/id.js";
import { redactSensitiveText } from "../src/utils/sensitive-data.js";
import { clip } from "../src/utils/text.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const SOURCE_DB = join(OUT_DIR, "span_v2_experiment.sqlite");
const RESULT_JSON = join(OUT_DIR, "span_v2_split_ablation.json");
const OUT_MD = join(OUT_DIR, "span_v2_trajectory_comparison.md");

interface SampleRange {
  label: string;
  start: number;
  end: number;
}

const SAMPLE_RANGES: Record<string, SampleRange[]> = {
  trace_34b400b688a7a786e2cf: [
    { label: "历史费率检索起始段", start: 0, end: 7 },
    { label: "PDF/费率提取中段", start: 72, end: 83 },
    { label: "后续重复检索段", start: 186, end: 199 }
  ],
  trace_eddec799ab47b1b126e6: [
    { label: "药物清单识别段", start: 0, end: 11 },
    { label: "价格批量检索中段", start: 48, end: 59 },
    { label: "Excel 填充验证段", start: 102, end: 112 }
  ],
  trace_b9773dcb11d9706ca47a: [
    { label: "学校资料检索起始段", start: 0, end: 11 },
    { label: "受阻检索与替代来源段", start: 24, end: 41 },
    { label: "PDF 生成调试段", start: 69, end: 84 }
  ]
};

function main(): void {
  mkdirSync(OUT_DIR, { recursive: true });
  const traceIds = selectTraceIds();
  const db = new MemoryDb({ path: SOURCE_DB });
  const repos = new Repositories(db.db);
  const lines = [
    "# 原始 Trace 与 Structured Trajectory 对比",
    "",
    "本报告抽样展示原始 tool call 与 programmatic structured trajectory 的对应关系。结构化结果只用于压缩和保留事实，不负责生成 goal/policy，也不改变 tool-call index。",
    "",
    "## 表征字段",
    "",
    "| 字段 | 含义 |",
    "| --- | --- |",
    "| `index` | 原始 tool call 的位置，保证可以回溯 |",
    "| `tool` | 工具名 |",
    "| `action` | 通用动作类型，不包含业务领域分类 |",
    "| `success/errorClass` | 成功状态和通用错误类别 |",
    "| `inputShape/outputShape` | 输入输出结构形状，不保留大段原文 |",
    "| `artifactSignal` | 检测到的文件/产物后缀信号 |",
    "| `evidence` | 脱敏后的短证据片段 |",
    "",
    "## 抽样对照",
    ""
  ];

  for (const traceId of traceIds) {
    const source = repos.memories.get(traceId);
    if (!source) continue;
    const rawTurnId = source.properties.internal_info.raw_turn_id;
    const rawTurn = typeof rawTurnId === "string" ? repos.runtime.getRawTurn(rawTurnId) : undefined;
    if (!rawTurn) continue;
    const trajectory = buildSpanTrajectory(rawTurn.toolCalls);
    lines.push(...renderTrace(traceId, source, rawTurn, trajectory));
  }
  db.close();
  writeFileSync(OUT_MD, lines.join("\n"));
}

function selectTraceIds(): string[] {
  const result = JSON.parse(readFileSync(RESULT_JSON, "utf8")) as { traceIds?: unknown };
  const ids = Array.isArray(result.traceIds)
    ? result.traceIds.filter((value): value is string => typeof value === "string")
    : [];
  const preferred = Object.keys(SAMPLE_RANGES);
  return preferred.filter((id) => ids.includes(id)).slice(0, 3);
}

function renderTrace(
  traceId: string,
  source: MemoryRow,
  rawTurn: RawTurnRecord,
  trajectory: SpanTrajectoryEvent[]
): string[] {
  const lines = [
    `### ${traceId}`,
    "",
    `- 原始 tool call 数：${rawTurn.toolCalls.length}`,
    `- structured event 数：${trajectory.length}`,
    `- 用户请求：${clip(redactSensitiveText(rawTurn.userText ?? ""), 500) || "未记录"}`,
    `- 结构化目标：保留 index 和通用行为事实，去掉大段输入输出和敏感值；未做语义 span 判断。`,
    ""
  ];

  for (const range of SAMPLE_RANGES[traceId] ?? []) {
    const start = Math.max(0, range.start);
    const end = Math.min(rawTurn.toolCalls.length - 1, range.end);
    if (start > end) continue;
    lines.push(`#### ${range.label}（${start}-${end}）`, "");
    lines.push("**原始 tool calls**", "", "```text");
    for (let index = start; index <= end; index += 1) {
      lines.push(formatRawCall(index, rawTurn.toolCalls[index]));
    }
    lines.push("```", "", "**Structured trajectory**", "", "```json");
    for (const event of eventsForToolCallRange(trajectory, { start, end })) {
      lines.push(JSON.stringify(event));
    }
    lines.push("```", "");
  }

  lines.push(
    "**观察**",
    "",
    "- 原始 trace 保留了大量任务实体、URL、文件内容、重复查询参数和完整输出，适合审计但不适合直接交给 LLM 做长轨迹拆分。",
    "- structured trajectory 保留了动作顺序、成功/失败、错误类别、输入输出形状和少量证据，因此可以支持候选边界判断，同时显著减少上下文噪音。",
    "- 结构化层仍然保留原始 index，所以后续 LLM 输出的 span range 可以回到原始 trace 验证。",
    "- 需要注意：`action` 是通用工具行为归类，不等于 policy；最终 goal/policy 仍应由后续 LLM 基于 structured trajectory 生成。",
    ""
  );
  return lines;
}

function formatRawCall(index: number, value: unknown): string {
  if (!isRecordLike(value)) {
    return `[${index}] invalid=${clip(redactSensitiveText(stableStringify(value)), 280)}`;
  }
  const name = typeof value.name === "string" ? value.name : "unknown";
  const input = clip(redactSensitiveText(stableStringify(value.input ?? null)), 180);
  const output = clip(redactSensitiveText(stableStringify(value.output ?? null)), 180);
  const error = typeof value.error === "string" ? clip(redactSensitiveText(value.error), 140) : "";
  return `[${index}] tool=${name} success=${value.success ?? !error} input=${input} output=${output}${error ? ` error=${error}` : ""}`;
}

function isRecordLike(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

main();
