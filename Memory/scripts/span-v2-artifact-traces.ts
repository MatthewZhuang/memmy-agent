import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import Database from "better-sqlite3";
import type { ToolCallPayload } from "../src/types.js";
import { stableHash } from "../src/utils/id.js";
import { isRecord } from "../src/utils/json.js";

export type ArtifactTraceSourceKind = "training_sqlite" | "test_jsonl";
export type TraceLengthBucket = "small" | "medium" | "large" | "xlarge";

export interface ArtifactTraceCase {
  traceId: string;
  sourceKind: ArtifactTraceSourceKind;
  sourcePath: string;
  userText: string;
  assistantText: string;
  toolCalls: ToolCallPayload[];
}

export interface LoadArtifactTraceOptions {
  artifactsRoot: string;
  minToolCalls?: number;
}

export interface StratifiedTraceSampleOptions {
  totalLimit: number;
  perBucketLimit: number;
  seed?: string;
}

export function loadArtifactTraceCases(options: LoadArtifactTraceOptions): ArtifactTraceCase[] {
  const minToolCalls = Math.max(1, Math.floor(options.minToolCalls ?? 4));
  const traces = [
    ...loadTrainingSqliteTraces(options.artifactsRoot),
    ...loadTestJsonlTraces(options.artifactsRoot)
  ];
  const seen = new Set<string>();
  const out: ArtifactTraceCase[] = [];
  for (const trace of traces) {
    if (trace.toolCalls.length < minToolCalls) continue;
    const key = stableHash(`${trace.userText}\n${trace.toolCalls.length}\n${trace.toolCalls.map((call) => call.name).join(",")}`);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(trace);
  }
  return out.sort((a, b) => a.traceId.localeCompare(b.traceId));
}

export function stratifiedTraceSample(
  traces: readonly ArtifactTraceCase[],
  options: StratifiedTraceSampleOptions
): ArtifactTraceCase[] {
  const selected: ArtifactTraceCase[] = [];
  for (const bucket of ["small", "medium", "large", "xlarge"] as const) {
    const bucketItems = traces
      .filter((trace) => traceLengthBucket(trace.toolCalls.length) === bucket)
      .sort((a, b) => seededRank(a, options.seed).localeCompare(seededRank(b, options.seed)));
    selected.push(...bucketItems.slice(0, options.perBucketLimit));
  }
  if (selected.length >= options.totalLimit) return selected.slice(0, options.totalLimit);
  const selectedIds = new Set(selected.map((trace) => trace.traceId));
  const rest = traces
    .filter((trace) => !selectedIds.has(trace.traceId))
    .sort((a, b) => seededRank(a, options.seed).localeCompare(seededRank(b, options.seed)));
  return [...selected, ...rest.slice(0, Math.max(0, options.totalLimit - selected.length))];
}

export function traceLengthBucket(toolCallCount: number): TraceLengthBucket {
  if (toolCallCount <= 15) return "small";
  if (toolCallCount <= 40) return "medium";
  if (toolCallCount <= 80) return "large";
  return "xlarge";
}

function loadTrainingSqliteTraces(artifactsRoot: string): ArtifactTraceCase[] {
  const dbPath = join(artifactsRoot, "memory-v10-training-20260804", "memory.sqlite");
  if (!existsSync(dbPath)) return [];
  const db = new Database(dbPath, { readonly: true });
  try {
    const rows = db.prepare(`
      SELECT id, user_text, assistant_text, tool_calls_json, tool_results_json
      FROM raw_turns
      WHERE json_array_length(tool_calls_json) >= 1
    `).all() as Array<{
      id: string;
      user_text: string | null;
      assistant_text: string | null;
      tool_calls_json: string;
      tool_results_json: string;
    }>;
    return rows.map((row) => ({
      traceId: `training:${row.id}`,
      sourceKind: "training_sqlite" as const,
      sourcePath: dbPath,
      userText: row.user_text ?? "",
      assistantText: row.assistant_text ?? "",
      toolCalls: normalizeStoredToolCalls(row.tool_calls_json, row.tool_results_json)
    }));
  } finally {
    db.close();
  }
}

function loadTestJsonlTraces(artifactsRoot: string): ArtifactTraceCase[] {
  const testRoot = join(artifactsRoot, "memory-v10-test-20260804");
  if (!existsSync(testRoot)) return [];
  const traces: ArtifactTraceCase[] = [];
  for (const entry of readdirSync(testRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.includes("__trial_")) continue;
    const sessionPath = join(testRoot, entry.name, "session.jsonl");
    if (!existsSync(sessionPath)) continue;
    const trace = parseSessionJsonl(sessionPath, `test:${entry.name}`);
    if (trace) traces.push(trace);
  }
  return traces;
}

function normalizeStoredToolCalls(
  toolCallsJson: string,
  toolResultsJson: string
): ToolCallPayload[] {
  const calls = parseJsonArray(toolCallsJson);
  const results = parseJsonArray(toolResultsJson);
  const resultsById = new Map<string, Record<string, unknown>>();
  for (const result of results) {
    if (!isRecord(result)) continue;
    const id = text(result.toolCallId) ?? text(result.id);
    if (id) resultsById.set(id, result);
  }
  return calls.flatMap((value, index): ToolCallPayload[] => {
    if (!isRecord(value)) return [];
    const id = text(value.id) ?? `tool_${index}`;
    const result = resultsById.get(id);
    const name = text(value.name) ?? text(result?.name) ?? "unknown";
    return [{
      id,
      name,
      input: value.input,
      output: result?.output ?? value.output,
      error: text(result?.error) ?? text(value.error),
      errorCode: text(result?.errorCode) ?? text(value.errorCode),
      success: boolean(result?.success) ?? boolean(value.success) ?? !(result?.error ?? value.error)
    }];
  });
}

function parseSessionJsonl(path: string, traceId: string): ArtifactTraceCase | null {
  const messages = readFileSync(path, "utf8")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as unknown];
      } catch {
        return [];
      }
    })
    .filter(isRecord);
  const userText = messages.find((message) => message.role === "user" && typeof message.content === "string")?.content;
  const assistantTexts = messages
    .filter((message) => message.role === "assistant" && typeof message.content === "string" && message.content.trim())
    .map((message) => String(message.content));
  const toolOutputs = new Map<string, { output?: unknown; error?: string; name?: string }>();
  for (const message of messages) {
    if (message.role !== "tool") continue;
    const id = text(message.tool_call_id);
    if (!id) continue;
    toolOutputs.set(id, {
      name: text(message.name),
      output: message.content,
      error: text(message.error)
    });
  }
  const toolCalls: ToolCallPayload[] = [];
  for (const message of messages) {
    if (message.role !== "assistant" || !Array.isArray(message.tool_calls)) continue;
    for (const [index, call] of message.tool_calls.entries()) {
      if (!isRecord(call)) continue;
      const id = text(call.id) ?? `tool_${toolCalls.length}_${index}`;
      const fn = isRecord(call.function) ? call.function : {};
      const name = text(fn.name) ?? text(call.name) ?? "unknown";
      const output = toolOutputs.get(id);
      toolCalls.push({
        id,
        name,
        input: parseFunctionArguments(fn.arguments),
        output: output?.output,
        error: output?.error,
        success: !output?.error
      });
    }
  }
  if (!userText || toolCalls.length === 0) return null;
  return {
    traceId,
    sourceKind: "test_jsonl",
    sourcePath: path,
    userText: String(userText),
    assistantText: assistantTexts.at(-1) ?? "",
    toolCalls
  };
}

function parseFunctionArguments(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function parseJsonArray(value: string): unknown[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function seededRank(trace: ArtifactTraceCase, seed: string | undefined): string {
  return stableHash(`${seed ?? ""}:${trace.traceId}:${trace.userText.slice(0, 200)}`);
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function boolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}
