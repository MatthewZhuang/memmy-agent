import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { MemoryDb } from "../src/index.js";
import { Repositories } from "../src/storage/repositories.js";
import { buildSpanTrajectory, type SpanTrajectoryEvent } from "../src/service/evolution/span-trajectory.js";
import { redactSensitiveText } from "../src/utils/sensitive-data.js";
import { clip } from "../src/utils/text.js";

const REPO_ROOT = join(dirname(new URL(import.meta.url).pathname), "../..");
const OUT_DIR = join(REPO_ROOT, "artifacts/analysis");
const SOURCE_DB = join(OUT_DIR, "span_v2_experiment.sqlite");
const OUT_JSON = join(OUT_DIR, "span_v2_long_span_audit.json");
const OUT_MD = join(OUT_DIR, "span_v2_long_span_audit.md");

const CASES = [
  {
    traceId: "trace_34b400b688a7a786e2cf",
    start: 47,
    end: 157,
    label: "Shipping flat-rate 价格提取"
  },
  {
    traceId: "trace_eddec799ab47b1b126e6",
    start: 12,
    end: 101,
    label: "Medication 价格批量检索"
  }
];

interface AuditCase {
  traceId: string;
  label: string;
  start: number;
  end: number;
  toolCallCount: number;
  actionRuns: Array<{ start: number; end: number; action: string; tools: string[] }>;
  transitions: Array<{ index: number; from: string; to: string }>;
  samples: Array<{ index: number; raw: string; structured: SpanTrajectoryEvent }>;
}

function main(): void {
  mkdirSync(OUT_DIR, { recursive: true });
  const db = new MemoryDb({ path: SOURCE_DB });
  const repos = new Repositories(db.db);
  const audits: AuditCase[] = [];

  for (const item of CASES) {
    const source = repos.memories.get(item.traceId);
    if (!source) continue;
    const rawTurnId = source.properties.internal_info.raw_turn_id;
    if (typeof rawTurnId !== "string") continue;
    const rawTurn = repos.runtime.getRawTurn(rawTurnId);
    if (!rawTurn) continue;
    const trajectory = buildSpanTrajectory(rawTurn.toolCalls);
    const events = trajectory
      .filter((event) => event.range[1] >= item.start && event.range[0] <= item.end)
      .map((event) => clipEventToRange(event, item.start, item.end));
    audits.push({
      traceId: item.traceId,
      label: item.label,
      start: item.start,
      end: item.end,
      toolCallCount: events.reduce((sum, event) => sum + event.callCount, 0),
      actionRuns: buildActionRuns(events),
      transitions: buildTransitions(events),
      samples: sampleEvents(events, rawTurn.toolCalls)
    });
  }
  db.close();
  writeFileSync(OUT_JSON, JSON.stringify(audits, null, 2));
  writeFileSync(OUT_MD, renderMarkdown(audits));
}

function buildActionRuns(events: SpanTrajectoryEvent[]): AuditCase["actionRuns"] {
  const runs: AuditCase["actionRuns"] = [];
  for (const event of events) {
    const last = runs.at(-1);
    if (last && last.action === event.action && event.range[0] === last.end + 1) {
      last.end = event.range[1];
      if (!last.tools.includes(event.tool)) last.tools.push(event.tool);
      continue;
    }
    runs.push({
      start: event.range[0],
      end: event.range[1],
      action: event.action,
      tools: [event.tool]
    });
  }
  return runs;
}

function clipEventToRange(
  event: SpanTrajectoryEvent,
  start: number,
  end: number
): SpanTrajectoryEvent {
  const clippedStart = Math.max(event.range[0], start);
  const clippedEnd = Math.min(event.range[1], end);
  const callCount = Math.max(1, clippedEnd - clippedStart + 1);
  return {
    ...event,
    index: clippedStart,
    range: [clippedStart, clippedEnd],
    callCount,
    repeated: callCount > 1
  };
}

function buildTransitions(events: SpanTrajectoryEvent[]): AuditCase["transitions"] {
  const transitions: AuditCase["transitions"] = [];
  for (let index = 1; index < events.length; index += 1) {
    const previous = events[index - 1]!;
    const current = events[index]!;
    if (previous.action !== current.action) {
      transitions.push({
        index: current.index,
        from: previous.action,
        to: current.action
      });
    }
  }
  return transitions;
}

function sampleEvents(events: SpanTrajectoryEvent[], rawCalls: unknown[]): AuditCase["samples"] {
  const selected = new Set<number>();
  const first = events.slice(0, 3);
  const middle = events.slice(Math.max(0, Math.floor(events.length / 2) - 1), Math.floor(events.length / 2) + 2);
  const last = events.slice(-3);
  for (const event of [...first, ...middle, ...last]) selected.add(event.range[0]);
  return [...selected].sort((a, b) => a - b).map((index) => ({
    index,
    raw: clip(redactSensitiveText(JSON.stringify(rawCalls[index] ?? null)), 500),
    structured: events.find((event) => event.range[0] <= index && event.range[1] >= index)!
  }));
}

function renderMarkdown(audits: AuditCase[]): string {
  const lines = [
    "# 超长 Span 原因审计",
    "",
    "本审计只检查超长 Span 覆盖的真实 tool-call 结构，不直接接受 LLM 生成的 goal/policy 作为合理性证明。",
    "",
    "## 判断标准",
    "",
    "- 同一个 Span 内是否只有一个稳定的主目标和方法。",
    "- 是否出现明显的方法阶段切换，例如检索 -> PDF 下载/解析 -> 脚本处理 -> 验证。",
    "- 是否只是同一策略对多个实体/年份/产品重复执行。重复执行可以属于一个策略，但需要区分“批量任务”与“可复用小策略”。",
    "- 是否出现多个独立产物、多个错误恢复路径或多个阶段性目标。",
    ""
  ];

  for (const audit of audits) {
    lines.push(
      `## ${audit.label}`,
      "",
      `- Trace：${audit.traceId}`,
      `- Range：${audit.start}-${audit.end}`,
      `- Tool calls：${audit.toolCallCount}`,
      `- Action runs：${audit.actionRuns.map((run) => `${run.start}-${run.end} ${run.action}[${run.tools.join(",")}]`).join("；")}`,
      `- Action transitions：${audit.transitions.map((transition) => `${transition.index} ${transition.from}->${transition.to}`).join("；") || "none"}`,
      "",
      "### 原始与结构化样本",
      ""
    );
    for (const sample of audit.samples) {
      lines.push(
        `#### index=${sample.index}`,
        "",
        "**原始**",
        "",
        "```json",
        sample.raw,
        "```",
        "",
        "**结构化**",
        "",
        "```json",
        JSON.stringify(sample.structured),
        "```",
        ""
      );
    }
    lines.push("### 人工判断", "", judgement(audit), "");
  }

  lines.push(
    "## 总结",
    "",
    "- 如果 Span 内只是同一种检索/解析策略对多个实体或多个价格项重复执行，它在“观察到的批处理方法”层面可以成立，但不一定适合作为最小可复用 policy span。",
    "- 如果 Span 内包含检索、下载、解析、失败恢复和多种工具链切换，则应认为存在阶段混合，超长不是合理的最小策略边界。",
    "- 本次 shipping 超长 Span 明显包含多承运商、多 PDF、命令行解析和失败回退，属于过宽合并，不应直接作为一个 policy span。",
    "- medication 价格 Span 的主体是逐药物价格检索，策略相对一致，但 90-call 长度来自大量实体重复，属于“方法一致但粒度过粗”；更适合按稳定检索子目标或固定窗口拆分后再聚合。",
    ""
  );
  return lines.join("\n");
}

function judgement(audit: AuditCase): string {
  if (audit.traceId === "trace_34b400b688a7a786e2cf") {
    return "不合理。虽然 goal 都指向 flat-rate shipping 价格，但 range 内同时包含多个承运商、不同 PDF 来源、下载、pdftotext/pypdf 解析、失败回退和重复检索。它更像一个任务级批处理过程，而不是单一可复用小策略。";
  }
  return "部分合理但粒度过粗。主体始终是逐个药物查询价格，方法一致；但 90 个 call 覆盖了大量不同药物，且包含 blocked lookup 后换查询方式的恢复过程。可描述为同一批处理策略，但不宜直接作为最小 policy span。";
}

main();
