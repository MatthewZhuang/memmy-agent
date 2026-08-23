import type { ToolCallPayload } from "../../types.js";
import { stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { redactSensitiveText } from "../../utils/sensitive-data.js";
import { clip } from "../../utils/text.js";

export type SpanTrajectoryAction =
  | "search"
  | "fetch"
  | "read"
  | "write"
  | "execute"
  | "edit"
  | "verify"
  | "communicate"
  | "other";

export interface SpanTrajectoryEvent {
  index: number;
  range: [number, number];
  callCount: number;
  repeated: boolean;
  tool: string;
  action: SpanTrajectoryAction;
  success: boolean;
  errorClass?: string;
  inputShape?: string;
  outputShape?: string;
  artifactSignal?: string;
  inputPreview?: string;
  outputPreview?: string;
  errorPreview?: string;
  evidence?: string;
}

const TOOL_NAME_MAX = 120;
const SHAPE_MAX = 160;
const INPUT_PREVIEW_MAX = 160;
const OUTPUT_PREVIEW_MAX = 220;
const ERROR_PREVIEW_MAX = 160;
const EVIDENCE_MAX = 420;
const DEFAULT_MAX_REPEATED_CALLS = 8;

export function buildSpanTrajectory(
  toolCalls: readonly unknown[],
  options: { maxRepeatedCalls?: number } = {}
): SpanTrajectoryEvent[] {
  const maxRepeatedCalls = Math.max(
    1,
    Math.floor(options.maxRepeatedCalls ?? DEFAULT_MAX_REPEATED_CALLS)
  );
  const events: SpanTrajectoryEvent[] = [];
  for (const [index, call] of toolCalls.entries()) {
    const event = buildEvent(call, index);
    const previous = events.at(-1);
    if (previous && canMerge(previous, event, maxRepeatedCalls)) {
      mergeEvent(previous, event);
    } else {
      events.push(event);
    }
  }
  return events;
}

function buildEvent(call: unknown, index: number): SpanTrajectoryEvent {
  if (!isToolCall(call)) {
    const preview = redactAndClip(stableStringify(call), OUTPUT_PREVIEW_MAX);
    return {
      index,
      range: [index, index],
      callCount: 1,
      repeated: false,
      tool: "unknown",
      action: "other",
      success: false,
      inputShape: "raw",
      outputPreview: preview,
      evidence: preview
    };
  }
  const tool = redactAndClip(call.name, TOOL_NAME_MAX);
  const inferredError = inferredToolError(call);
  const success = call.success === false || inferredError
    ? false
    : call.success ?? true;
  const errorClass = classifyError(call, inferredError);
  const inputPreview = previewValue(call.input, INPUT_PREVIEW_MAX);
  const outputPreview = previewValue(call.output, OUTPUT_PREVIEW_MAX);
  const errorPreview = inferredError
    ? redactAndClip(inferredError, ERROR_PREVIEW_MAX)
    : undefined;
  return {
    index,
    range: [index, index],
    callCount: 1,
    repeated: false,
    tool,
    action: classifyAction(call),
    success,
    ...(errorClass ? { errorClass } : {}),
    inputShape: describeShape(call.input),
    outputShape: describeShape(call.output),
    ...artifactSignal(call),
    ...(inputPreview ? { inputPreview } : {}),
    ...(outputPreview ? { outputPreview } : {}),
    ...(errorPreview ? { errorPreview } : {}),
    evidence: evidenceSnippet({ inputPreview, outputPreview, errorPreview })
  };
}

function canMerge(
  previous: SpanTrajectoryEvent,
  current: SpanTrajectoryEvent,
  maxRepeatedCalls: number
): boolean {
  return (
    previous.range[1] + 1 === current.range[0] &&
    previous.callCount < maxRepeatedCalls &&
    previous.tool === current.tool &&
    previous.action === current.action &&
    previous.success === current.success &&
    previous.errorClass === current.errorClass
  );
}

function mergeEvent(previous: SpanTrajectoryEvent, current: SpanTrajectoryEvent): void {
  previous.range[1] = current.range[1];
  previous.callCount += current.callCount;
  previous.repeated = true;
  previous.artifactSignal = mergeText(previous.artifactSignal, current.artifactSignal, 4, SHAPE_MAX);
  previous.inputPreview = mergeText(previous.inputPreview, current.inputPreview, 2, INPUT_PREVIEW_MAX);
  previous.outputPreview = mergeText(previous.outputPreview, current.outputPreview, 2, OUTPUT_PREVIEW_MAX);
  previous.errorPreview = mergeText(previous.errorPreview, current.errorPreview, 2, ERROR_PREVIEW_MAX);
  previous.evidence = evidenceSnippet({
    inputPreview: previous.inputPreview,
    outputPreview: previous.outputPreview,
    errorPreview: previous.errorPreview,
    repeated: previous.callCount,
    range: previous.range
  });
}

function mergeText(
  previous: string | undefined,
  current: string | undefined,
  maxItems: number,
  maxChars: number
): string | undefined {
  const values = [...new Set(
    [previous, current]
      .flatMap((value) => value?.split("\n") ?? [])
      .map((value) => value.trim())
      .filter(Boolean)
  )].slice(0, maxItems);
  return values.length > 0 ? redactAndClip(values.join("\n"), maxChars) : undefined;
}

function classifyAction(call: ToolCallPayload): SpanTrajectoryAction {
  const name = call.name.toLowerCase();
  if (hasToolToken(name, ["search", "grep", "rg", "find", "query", "lookup"])) return "search";
  if (hasToolToken(name, ["fetch", "open", "download", "crawl", "request", "http", "web"])) return "fetch";
  if (hasToolToken(name, ["read", "cat", "sed", "head", "tail", "list", "ls", "inspect", "show"])) return "read";
  if (hasToolToken(name, ["write", "save", "create", "insert", "export"])) return "write";
  if (hasToolToken(name, ["exec", "run", "shell", "bash", "test", "npm", "pytest", "tsc", "python", "node"])) return "execute";
  if (hasToolToken(name, ["edit", "patch", "apply", "replace", "update"])) return "edit";
  if (hasToolToken(name, ["verify", "validate", "check", "lint", "typecheck"])) return "verify";
  if (hasToolToken(name, ["comment", "message", "reply", "ask"])) return "communicate";
  return "other";
}

function hasToolToken(name: string, tokens: readonly string[]): boolean {
  const normalized = name.split(/[^a-z0-9]+/u).filter(Boolean);
  return tokens.some((token) => normalized.includes(token) || name.includes(token));
}

function classifyError(call: ToolCallPayload, inferredError?: string): string | undefined {
  if (!inferredError && !call.errorCode && call.success !== false) return undefined;
  const raw = `${call.errorCode ?? ""}\n${inferredError ?? ""}`.toLowerCase();
  if (/timeout|timed out|deadline/u.test(raw)) return "timeout";
  if (/403|forbidden|permission|unauthorized|401/u.test(raw)) return "access";
  if (/404|not found|enoent/u.test(raw)) return "not_found";
  if (/syntax|parse|json/u.test(raw)) return "syntax";
  if (/typeerror|referenceerror|nameerror|undefined|null/u.test(raw)) return "runtime";
  if (/assert|expect|test failed|failure/u.test(raw)) return "test_failure";
  if (/rate.?limit|429/u.test(raw)) return "rate_limit";
  return "error";
}

function inferredToolError(call: ToolCallPayload): string | undefined {
  if (typeof call.error === "string" && call.error.trim()) return call.error.trim();
  if (call.success === false) return "Tool reported success=false";
  const output = call.output;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        const parsed = JSON.parse(trimmed) as unknown;
        const structuredError = inferredStructuredOutputError(parsed);
        if (structuredError) return structuredError;
      } catch {
        // Non-JSON text is evaluated by the generic protocol markers below.
      }
    }
    if (/^(?:error|failed|failure|exception|fatal)\b\s*[:\-]?/iu.test(trimmed)) {
      return trimmed;
    }
    if (/\bexit\s+code\s*[:=]?\s*[1-9]\d*\b/iu.test(trimmed)) {
      return trimmed;
    }
    return undefined;
  }
  return inferredStructuredOutputError(output);
}

function inferredStructuredOutputError(output: unknown): string | undefined {
  if (!isRecord(output)) return undefined;
  if (output.success === false || output.ok === false) {
    return previewValue(output.error ?? output.message ?? output, ERROR_PREVIEW_MAX);
  }
  if (typeof output.error === "string" && output.error.trim()) return output.error.trim();
  const status = typeof output.status === "number"
    ? output.status
    : typeof output.status === "string" && /^\d{3}$/u.test(output.status.trim())
      ? Number(output.status)
      : undefined;
  if (status !== undefined && status >= 400) {
    const detail = previewValue(output.message ?? output.statusText ?? output.text, 100);
    return `HTTP ${status}${detail ? `: ${detail}` : ""}`;
  }
  return undefined;
}

function describeShape(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") return clip(`string:${value.length}`, SHAPE_MAX);
  if (typeof value === "number" || typeof value === "boolean") return typeof value;
  if (Array.isArray(value)) return clip(`array:${value.length}`, SHAPE_MAX);
  if (!isRecord(value)) return typeof value;
  const keys = Object.keys(value)
    .filter((key) => !sensitiveKey(key))
    .slice(0, 8);
  return clip(`object:{${keys.join(",")}}`, SHAPE_MAX);
}

function artifactSignal(call: ToolCallPayload): Pick<SpanTrajectoryEvent, "artifactSignal"> {
  const text = stableStringify({ input: call.input, output: call.output });
  const redacted = redactSensitiveText(text);
  const matches = [...redacted.matchAll(/\b[\w./-]+\.(?:ts|tsx|js|jsx|py|json|md|txt|csv|xlsx|docx|pptx|pdf|sqlite|db|html|css)\b/giu)]
    .map((match) => match[0])
    .slice(0, 4);
  if (matches.length === 0) return {};
  return { artifactSignal: matches.join(", ") };
}

function evidenceSnippet(input: {
  inputPreview?: string;
  outputPreview?: string;
  errorPreview?: string;
  repeated?: number;
  range?: [number, number];
}): string | undefined {
  const parts = [
    input.repeated && input.range
      ? `repeat=${input.repeated} calls range=${input.range[0]}-${input.range[1]}`
      : "",
    input.errorPreview ? `error=${input.errorPreview}` : "",
    input.inputPreview ? `input=${input.inputPreview}` : "",
    input.outputPreview ? `output=${input.outputPreview}` : ""
  ].filter(Boolean);
  if (parts.length === 0) return undefined;
  return redactAndClip(parts.join(" | "), EVIDENCE_MAX);
}

function previewValue(value: unknown, maxChars: number): string | undefined {
  if (value === undefined || value === null) return undefined;
  return redactAndClip(stableStringify(value), maxChars);
}

function sensitiveKey(key: string): boolean {
  return /api.?key|token|secret|password|authorization|credential/i.test(key);
}

function redactAndClip(value: string, maxChars: number): string {
  return clip(redactSensitiveText(value), maxChars);
}

function isToolCall(value: unknown): value is ToolCallPayload {
  return isRecord(value) && typeof value.name === "string" && value.name.length > 0;
}
