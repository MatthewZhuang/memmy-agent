import type { MemoryRow } from "../../types.js";
import { attachMemoryVector, memoryVector } from "../../storage/memory-vector-state.js";
import type {
  EmbeddingRetryRecord,
  EmbeddingRetryTargetKind,
  EmbeddingRetryVectorField
} from "../../storage/repositories.js";
import {
  policyMetaFromMemory,
  RETRIEVAL_DOCUMENT_VERSION,
  retrievalDocumentForMemory,
  skillMetaFromMemory,
  traceMetaFromMemory,
  worldModelMetaFromMemory
} from "../../algorithm/plugin-algorithms.js";
import { isRecord } from "../../utils/json.js";
import { clip } from "../../utils/text.js";
import { stableHash } from "../../utils/id.js";
import { spanPayload } from "../evolution/span-model.js";

const EMBEDDING_RETRY_BASE_BACKOFF_MS = 60_000;
const EMBEDDING_RETRY_MAX_BACKOFF_MS = 60 * 60_000;
const NEGATIVE_POLICY_EMBEDDING_TOKEN_LIMIT = 2_048;

export type SpanEmbeddingVectorField = "vec_goal" | "vec_policy";

export interface EmbeddingRetryRunItem {
  id: string;
  status: EmbeddingRetryRecord["status"];
  targetKind: EmbeddingRetryTargetKind;
  targetMemoryId: string;
  vectorField: EmbeddingRetryVectorField;
  attempts: number;
  lastError?: string | null;
}

export function embeddingTextForMemory(memory: MemoryRow): string {
  const trace = traceMetaFromMemory(memory);
  if (trace) {
    return [trace.summary, trace.reflection ?? ""]
      .filter(Boolean)
      .join("\n");
  }
  const policy = policyMetaFromMemory(memory);
  if (policy) {
    if (policy.experienceType === "failure_avoidance" || policy.evidencePolarity === "negative") {
      const text = [policy.title, policy.trigger].filter(Boolean).join("\n");
      return exceedsMixedLanguageTokenLimit(text, NEGATIVE_POLICY_EMBEDDING_TOKEN_LIMIT)
        ? policy.title
        : text;
    }
    return [policy.title, policy.trigger, policy.procedure, policy.verification, policy.boundary]
      .filter(Boolean)
      .join("\n");
  }
  const skill = skillMetaFromMemory(memory);
  if (skill) {
    return retrievalDocumentForMemory(memory);
  }
  const world = worldModelMetaFromMemory(memory);
  if (world) {
    return retrievalDocumentForMemory(memory);
  }
  return memory.memoryValue;
}

function exceedsMixedLanguageTokenLimit(value: string, limit: number): boolean {
  let count = 0;
  for (const _match of value.matchAll(/\p{Script=Han}|[A-Za-z]+(?:['’-][A-Za-z]+)*/gu)) {
    count += 1;
    if (count > limit) return true;
  }
  return false;
}

export function traceSummaryEmbeddingText(memory: MemoryRow): string | undefined {
  const span = isRecord(memory.properties.internal_info.span)
    ? memory.properties.internal_info.span
    : undefined;
  const spanGoal = span ? stringFromRecord(span, "span_goal") : undefined;
  const spanSummary = span ? stringFromRecord(span, "summary") : undefined;
  if (spanGoal && spanSummary) {
    return [
      `Goal: ${spanGoal}`,
      `Summary: ${spanSummary}`
    ].join("\n");
  }
  const trace = traceMetaFromMemory(memory);
  const summary = firstRealSummary(
    trace?.summary,
    stringFromRecord(memory.info, "summary"),
    stringFromRecord(memory.properties.internal_info, "summary")
  );
  if (!summary) return undefined;
  const originalExchange = trace
    ? clip([trace.userText, trace.agentText].filter(Boolean).join("\n"), 3_000)
    : "";
  return [
    `Summary: ${summary}`,
    ...(originalExchange ? [`Original exchange:\n${originalExchange}`] : [])
  ].join("\n\n");
}

export function spanEmbeddingText(
  memory: MemoryRow,
  field: SpanEmbeddingVectorField
): string | undefined {
  const span = spanPayload(memory);
  if (!span) return undefined;
  return field === "vec_goal"
    ? `Goal: ${span.goal}`
    : `Policy: ${span.policy}`;
}

export function spanEmbeddingSourceHash(
  memory: MemoryRow,
  field: SpanEmbeddingVectorField
): string | undefined {
  const text = spanEmbeddingText(memory, field);
  return text ? stableHash({ field, text }) : undefined;
}

export function embeddingRetrySourceText(
  memory: MemoryRow,
  vectorField: EmbeddingRetryVectorField
): string | undefined {
  if (spanPayload(memory)) {
    return vectorField === "vec_goal" || vectorField === "vec_policy"
      ? spanEmbeddingText(memory, vectorField)
      : undefined;
  }
  return memory.memoryLayer === "Skill" || memory.memoryLayer === "L3"
    ? embeddingTextForMemory(memory)
    : undefined;
}

export function spanHasBothEmbeddings(memory: MemoryRow): boolean {
  return Boolean(spanPayload(memory) && memoryVector(memory, "vec_goal") && memoryVector(memory, "vec_policy"));
}

export function embeddingRetryTargetKindForMemory(memory: MemoryRow): EmbeddingRetryTargetKind {
  if (spanPayload(memory)) return "span";
  if (memory.memoryLayer === "L1") return "trace";
  if (memory.memoryLayer === "L2") return "policy";
  if (memory.memoryLayer === "L3") return "world_model";
  return "skill";
}

export function embeddingRetryVectorFieldForMemory(memory: MemoryRow): EmbeddingRetryVectorField {
  if (spanPayload(memory)) return "vec_goal";
  return memory.memoryLayer === "L1" ? "vec_summary" : "vec";
}

export function embeddingRetryBackoffMs(attemptNo: number): number {
  return Math.min(
    EMBEDDING_RETRY_MAX_BACKOFF_MS,
    EMBEDDING_RETRY_BASE_BACKOFF_MS * 2 ** Math.max(0, attemptNo - 1)
  );
}

export function embeddingRetryToRunItem(retry: EmbeddingRetryRecord): EmbeddingRetryRunItem {
  return {
    id: retry.id,
    status: retry.status,
    targetKind: retry.targetKind,
    targetMemoryId: retry.targetId,
    vectorField: retry.vectorField,
    attempts: retry.attempts,
    lastError: retry.lastError
  };
}

export function updateMemoryVectorField(
  memory: MemoryRow,
  vectorField: EmbeddingRetryVectorField,
  vector: number[],
  input: { provider: string; model: string; updatedAt: string; sourceHash?: string }
): MemoryRow {
  const internal = memory.properties.internal_info;
  const nextInternal: Record<string, unknown> = { ...internal };
  if (memory.memoryLayer === "L1" && isRecord(internal.trace)) {
    nextInternal.trace = { ...internal.trace };
  } else if (memory.memoryLayer === "L2" && isRecord(internal.policy)) {
    nextInternal.policy = { ...internal.policy };
  } else if (memory.memoryLayer === "L3" && isRecord(internal.world_model)) {
    nextInternal.world_model = { ...internal.world_model };
  } else if (memory.memoryLayer === "Skill" && isRecord(internal.skill)) {
    nextInternal.skill = { ...internal.skill };
  }
  if ((memory.memoryLayer === "L3" || memory.memoryLayer === "Skill") && input.sourceHash) {
    nextInternal.retrieval_index = {
      version: RETRIEVAL_DOCUMENT_VERSION,
      source_hash: input.sourceHash,
      indexed_at: input.updatedAt
    };
  }

  const updated = {
    ...memory,
    properties: {
      ...memory.properties,
      internal_info: { ...memory.properties.internal_info, ...nextInternal }
    },
    updatedAt: input.updatedAt
  };
  return attachMemoryVector(updated, {
    vectorField,
    vector,
    embeddingProvider: input.provider,
    embeddingModel: input.model
  });
}

function firstRealSummary(...values: Array<string | undefined>): string | undefined {
  return values
    .map((value) => value?.trim())
    .find((value): value is string => Boolean(value && !isImportSummaryPlaceholder(value)));
}

function isImportSummaryPlaceholder(value: string | undefined): boolean {
  const first = value
    ?.split(/\r?\n/)
    .map((line) => line.replace(/^\s*#{1,6}\s+/, "").trim())
    .find(Boolean);
  return Boolean(first && /^(user|assistant|system|tool|developer|摘要排队中|摘要整理中)$/i.test(first));
}

function stringFromRecord(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" ? value : undefined;
}
