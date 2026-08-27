import type { ToolCallPayload } from "../../types.js";
import { isRecord } from "../../utils/json.js";
import { clip } from "../../utils/text.js";

export type NormalizedToolOutcome = "success" | "failure" | "unknown";

export interface ToolOutcomeInference {
  status: NormalizedToolOutcome;
  reason?: string;
  errorCode?: string;
}

export function inferToolOutcome(input: {
  call?: ToolCallPayload;
  result?: unknown;
  completionObserved?: boolean;
}): ToolOutcomeInference {
  const call = input.call;
  if (call?.success === false) {
    return failure(
      call.error ?? "The tool explicitly reported success=false.",
      normalizedExplicitErrorCode(call.errorCode)
    );
  }
  if (call?.error?.trim()) {
    return failure(call.error, normalizedExplicitErrorCode(call.errorCode));
  }
  const explicitErrorCode = normalizedExplicitErrorCode(call?.errorCode);
  if (explicitErrorCode) {
    return failure(`The tool reported error code ${explicitErrorCode}.`, explicitErrorCode);
  }

  const seen = new Set<object>();
  const failureSignal = findFailureSignal(input.result, seen) ??
    findFailureSignal(call?.output, seen);
  if (failureSignal) return failureSignal;

  if (call?.success === true) return { status: "success" };
  if (hasPositiveCompletionSignal(input.result, new Set()) ||
      hasPositiveCompletionSignal(call?.output, new Set())) {
    return { status: "success" };
  }
  if (input.completionObserved && input.result !== undefined) {
    return { status: "success" };
  }
  return { status: "unknown" };
}

function findFailureSignal(
  value: unknown,
  seen: Set<object>,
  depth = 0
): ToolOutcomeInference | undefined {
  if (value === undefined || value === null || depth > 5) return undefined;
  if (typeof value === "string") return failureFromText(value, seen, depth);
  if (value instanceof Error) return failure(value.message, value.name || "ERROR");
  if (typeof value !== "object") return undefined;
  if (seen.has(value)) return undefined;
  seen.add(value);

  if (Array.isArray(value)) {
    for (const item of value.slice(0, 32)) {
      const signal = findFailureSignal(item, seen, depth + 1);
      if (signal) return signal;
    }
    return undefined;
  }
  if (!isRecord(value)) return undefined;

  const explicitCode = normalizedExplicitErrorCode(
    stringValue(value.errorCode) ?? stringValue(value.error_code)
  );
  if (value.success === false || value.ok === false || value.isError === true || value.is_error === true) {
    return failure(
      failureMessage(value) ?? "The tool returned a structured failure result.",
      explicitCode
    );
  }

  const exitCode = numericValue(value.exitCode ?? value.exit_code);
  if (exitCode !== undefined && exitCode !== 0) {
    return failure(
      failureMessage(value) ?? `The tool exited with non-zero code ${exitCode}.`,
      `EXIT_${exitCode}`
    );
  }

  const httpStatus = numericValue(
    value.httpStatus ?? value.http_status ?? value.statusCode ?? value.status_code
  );
  if (httpStatus !== undefined && httpStatus >= 400) {
    return failure(
      failureMessage(value) ?? `The tool returned HTTP ${httpStatus}.`,
      `HTTP_${httpStatus}`
    );
  }

  const status = value.status;
  const numericStatus = numericValue(status);
  if (numericStatus !== undefined && numericStatus >= 400) {
    return failure(
      failureMessage(value) ?? `The tool returned HTTP ${numericStatus}.`,
      `HTTP_${numericStatus}`
    );
  }
  if (typeof status === "string" && FAILURE_STATUSES.has(status.trim().toLowerCase())) {
    return failure(
      failureMessage(value) ?? `The tool returned status ${status.trim()}.`,
      explicitCode ?? status.trim().toUpperCase()
    );
  }

  if (hasSubstantiveError(value.error)) {
    const nested = findFailureSignal(value.error, seen, depth + 1);
    return nested ?? failure(
      textPreview(value.error) ?? "The tool returned a structured error.",
      explicitCode
    );
  }
  if (explicitCode) {
    return failure(
      failureMessage(value) ?? `The tool reported error code ${explicitCode}.`,
      explicitCode
    );
  }

  for (const key of ["message", "output", "result", "content", "text", "stderr"] as const) {
    const signal = findFailureSignal(value[key], seen, depth + 1);
    if (signal) return signal;
  }
  return undefined;
}

function failureFromText(
  value: string,
  seen: Set<object>,
  depth: number
): ToolOutcomeInference | undefined {
  const text = value.trim();
  if (!text) return undefined;
  if ((text.startsWith("{") || text.startsWith("[")) && text.length <= 100_000) {
    try {
      const structured = JSON.parse(text) as unknown;
      const signal = findFailureSignal(structured, seen, depth + 1);
      if (signal) return signal;
    } catch {
      // Non-JSON output is handled by the protocol markers below.
    }
  }

  const exitCode = capturedNumber(text, [
    /\bexit\s+(?:code|status)\s*[:=]?\s*(-?\d+)\b/iu,
    /\b(?:process|command|script|tool)?\s*(?:exited?|terminated)(?:\s+with)?(?:\s+(?:exit\s+)?(?:code|status))?\s*[:=]?\s*(-?\d+)\b/iu
  ]);
  if (exitCode !== undefined && exitCode !== 0) {
    return failure(text, `EXIT_${exitCode}`);
  }
  const httpStatus = capturedNumber(text, [
    /\bHTTP(?:\/\d(?:\.\d)?)?\s+([45]\d{2})\b/iu,
    /\bstatus(?:\s+code)?\s*[:=]?\s*([45]\d{2})\b/iu
  ]);
  if (httpStatus !== undefined) return failure(text, `HTTP_${httpStatus}`);
  if (/traceback\s*\(most recent call last\)/iu.test(text)) {
    return failure(text, "TRACEBACK");
  }
  if (/^\s*(?:error|failed|failure|exception|fatal)\b\s*[:\-]?/iu.test(text) ||
      /^\s*npm\s+err!/iu.test(text) ||
      /^\s*(?:command|script|execution|tests?)\s+failed\b/iu.test(text)) {
    return failure(text, "ERROR_OUTPUT");
  }
  return undefined;
}

function hasPositiveCompletionSignal(value: unknown, seen: Set<object>, depth = 0): boolean {
  if (value === undefined || value === null || depth > 5) return false;
  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return false;
    if ((text.startsWith("{") || text.startsWith("[")) && text.length <= 100_000) {
      try {
        return hasPositiveCompletionSignal(JSON.parse(text) as unknown, seen, depth + 1);
      } catch {
        // Fall through to explicit textual completion markers.
      }
    }
    return /\bexit\s+(?:code|status)\s*[:=]?\s*0\b/iu.test(text) ||
      /\b(?:process|command|script|tool)\s+(?:exited?|terminated)(?:\s+with)?(?:\s+(?:exit\s+)?(?:code|status))?\s*[:=]?\s*0\b/iu.test(text);
  }
  if (typeof value !== "object") return false;
  if (seen.has(value)) return false;
  seen.add(value);
  if (Array.isArray(value)) {
    return value.slice(0, 32).some((item) => hasPositiveCompletionSignal(item, seen, depth + 1));
  }
  if (!isRecord(value)) return false;
  if (value.success === true || value.ok === true) return true;
  const exitCode = numericValue(value.exitCode ?? value.exit_code);
  if (exitCode === 0) return true;
  const httpStatus = numericValue(
    value.httpStatus ?? value.http_status ?? value.statusCode ?? value.status_code
  );
  if (httpStatus !== undefined && httpStatus >= 200 && httpStatus < 400) return true;
  if (typeof value.status === "string" && SUCCESS_STATUSES.has(value.status.trim().toLowerCase())) {
    return true;
  }
  return ["output", "result", "content"]
    .some((key) => hasPositiveCompletionSignal(value[key], seen, depth + 1));
}

function failureMessage(value: Record<string, unknown>): string | undefined {
  for (const candidate of [value.error, value.message, value.stderr, value.text]) {
    const preview = textPreview(candidate);
    if (preview) return preview;
  }
  return undefined;
}

function failure(reason: string, errorCode?: string): ToolOutcomeInference {
  const normalizedReason = clip(reason.trim(), 400);
  return {
    status: "failure",
    reason: normalizedReason || "The tool returned failure evidence.",
    ...(errorCode ? { errorCode } : {})
  };
}

function normalizedExplicitErrorCode(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  if (!normalized || /^(?:0|ok|success|succeeded|passed|2\d\d)$/iu.test(normalized)) return undefined;
  return clip(normalized, 120);
}

function hasSubstantiveError(value: unknown): boolean {
  if (value === undefined || value === null || value === false) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (isRecord(value)) return Object.keys(value).length > 0;
  return true;
}

function textPreview(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim()) return clip(value.trim(), 400);
  if (value instanceof Error) return clip(value.message, 400);
  if (isRecord(value)) {
    for (const key of ["message", "error", "detail"] as const) {
      const nested = textPreview(value[key]);
      if (nested) return nested;
    }
  }
  return undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function numericValue(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string" || !/^-?\d+$/u.test(value.trim())) return undefined;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : undefined;
}

function capturedNumber(value: string, patterns: readonly RegExp[]): number | undefined {
  for (const pattern of patterns) {
    const matched = value.match(pattern)?.[1];
    if (matched === undefined) continue;
    const parsed = Number(matched);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

const FAILURE_STATUSES = new Set(["failed", "failure", "error", "errored", "cancelled"]);
const SUCCESS_STATUSES = new Set(["succeeded", "success", "ok", "passed", "completed", "complete"]);
