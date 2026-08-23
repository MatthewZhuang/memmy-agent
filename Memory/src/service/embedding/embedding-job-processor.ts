import { isRecord } from "../../utils/json.js";
import { clip,firstLine } from "../../utils/text.js";
/**
 * Embedding and trace-summary worker domain, extracted from MemoryService.
 *
 * This module intentionally has no MemoryService import.  The host owns the
 * generic job-enqueue policy; this processor owns the job-specific state
 * transitions, model calls, and change records.
 */
import { retrievalDocumentSourceHash,traceMetaFromMemory } from "../../algorithm/plugin-algorithms.js";
import type { Embedder,LlmClient } from "../../model/types.js";
import type { EmbeddingRetryRecord,EmbeddingRetryVectorField,EvolutionJobRecord,Repositories } from "../../storage/repositories.js";
import { kindFromMemory } from "../../storage/repositories.js";
import type { JobType,MemoryProcessingState,MemoryRow,ToolCallPayload,UserMemoryType } from "../../types.js";
import { stableHash } from "../../utils/id.js";
import { isMemmyRecallToolName } from "../../utils/memmy-context-tags.js";
import {
  firstRealSummary,
  importStatusTags,
  isImportSummaryPlaceholder,
  memoryHasImportPipeline,
  memoryNeedsImportSummary,
  updateImportPipelineStatus,
  updateTraceImportSummary
} from "../import/import-job-processor.js";
import {
  IMPORT_DEFAULT_ALPHA,
  IMPORT_DEFAULT_PRIORITY,
  IMPORT_DEFAULT_VALUE
} from "../import/memory-import-pipeline.js";
import { namespaceForMemory } from "../namespace/namespace-scope.js";
import { processingJobMatchesMemory } from "../worker/job-handlers.js";
import {
  buildUserMemory,
  classifyUserMemory,
  isDynamicCurrentFactQuery,
  isTaskLinkedUserFeedback,
  isUserMemoryQuestion
} from "../user-memory/user-memory.js";
import {
  embeddingTextForMemory,
  spanEmbeddingSourceHash,
  spanEmbeddingText,
  spanHasBothEmbeddings,
  type SpanEmbeddingVectorField,
  traceSummaryEmbeddingText,
  updateMemoryVectorField
} from "./embedding-pipeline.js";
import { spanPayload } from "../evolution/span-model.js";

type TraceMeta = NonNullable<ReturnType<typeof traceMetaFromMemory>>;
type TurnCaptureDecision = {
  createL1: boolean;
  l1Summary: string;
  createUserMemory: boolean;
  userMemoryTypes: UserMemoryType[];
  userMemoryEvidence: Array<{ quote: string; type: UserMemoryType }>;
  l1Evidence: Array<{ quote: string; sourceRole: "user" | "assistant" | "tool"; kind: string }>;
  reason: string;
};

export interface PreparedEmbeddingJob {
  job: EvolutionJobRecord;
  memory: MemoryRow;
  text: string;
  role: "document" | "query";
  vectorField: EmbeddingRetryVectorField;
  sourceHash?: string;
}

export interface EnqueueWorkerJobInput {
  jobType: JobType;
  userId: string;
  sessionId?: string;
  episodeId?: string;
  targetMemoryId?: string;
  dedupeKey?: string;
  payload?: Record<string, unknown>;
  maxAttempts?: number;
  createdAt?: string;
}

export interface PersistEmbeddingVectorInput {
  memoryId: string;
  vectorField: EmbeddingRetryVectorField;
  vector: number[];
  attemptCount: number;
  source: string;
  sourceHash?: string;
  allowedProcessingStates?: MemoryProcessingState[];
  finalize?: (saved: MemoryRow, hadProcessing: boolean, at: string) => void;
}

export interface ScheduleEmbeddingAfterTextUpdateInput {
  memory: MemoryRow;
  sourceJob: EvolutionJobRecord;
  reason: string;
  vectorField: EmbeddingRetryVectorField;
  clearExistingVector: boolean;
  allowedProcessingStates?: MemoryProcessingState[];
  textOnlyAttemptCount: number;
  at?: string;
}

/** The host supplies cross-domain policy rather than this module reaching into MemoryService. */
export interface EmbeddingJobProcessorDeps {
  repos: Pick<Repositories, "transaction" | "memories" | "userMemories" | "processing" | "runtime">;
  embedder: Embedder;
  llm: LlmClient;
  capture: { embedAfterCapture: boolean };
  nowIso: () => string;
  enqueueJob: (input: EnqueueWorkerJobInput) => EvolutionJobRecord;
  enqueueImportSummaryIfMissing(memory: MemoryRow, at: string): void;
  enqueueEmbeddingRetry(
    memory: MemoryRow,
    sourceText: string,
    at: string,
    vectorField?: EmbeddingRetryVectorField
  ): EmbeddingRetryRecord;
  appendEmbeddingRetryChange(
    retry: EmbeddingRetryRecord,
    op: "queued" | "retry" | "succeeded" | "failed",
    before?: EmbeddingRetryRecord,
    error?: { code: string; message: string }
  ): void;
  summarizeTraceForCapture(
    input: {
      trace: TraceMeta;
      userText: string;
      agentText: string;
      toolCalls: ToolCallPayload[];
      reflectionText: string;
    },
    options?: { strict?: boolean }
  ): Promise<string>;
  decideTurnMemoryForCapture(input: {
    trace: TraceMeta;
    userText: string;
    agentText: string;
    toolCalls: ToolCallPayload[];
    reflectionText: string;
  }): Promise<TurnCaptureDecision>;
}

export class EmbeddingJobProcessor {
  constructor(private readonly deps: EmbeddingJobProcessorDeps) {}

  async embedMemory(job: EvolutionJobRecord): Promise<void> {
    const item = this.prepareEmbeddingJob(job);
    if (!item) return;
    try {
      const [vector] = await this.deps.embedder.embed([item.text], item.role);
      this.applyEmbeddingVector(item, vector ?? []);
    } catch (error) {
      if (!this.enqueueEmbeddingRetryAfterFailure(item, error)) throw error;
    }
  }

  async embedUserMemory(job: EvolutionJobRecord): Promise<void> {
    const memory = job.targetMemoryId
      ? this.deps.repos.userMemories.get(job.targetMemoryId)
      : undefined;
    if (!memory) throw new Error(`user memory embedding target not found: ${job.targetMemoryId ?? "unknown"}`);
    const contentHash = typeof job.payload.contentHash === "string" ? job.payload.contentHash : undefined;
    if (contentHash && contentHash !== stableHash(memory.content)) return;
    const vector = await this.deps.embedder.embedOne(memory.content, "document");
    const current = this.deps.repos.userMemories.get(memory.id);
    if (!current || (contentHash && contentHash !== stableHash(current.content))) return;
    const at = this.deps.nowIso();
    const saved = this.deps.repos.userMemories.updateEmbedding(current.id, vector, {
      model: this.deps.embedder.config.model,
      provider: this.deps.embedder.config.provider,
      updatedAt: at
    });
    if (!saved) return;
    this.deps.repos.runtime.appendChange({
      memoryId: saved.id,
      kind: "user_memory",
      op: "updated",
      entityId: saved.id,
      userId: saved.userId,
      changeType: "user_memory_embedding_updated",
      before: current,
      after: saved,
      source: "worker.user_memory_embedding",
      createdAt: at
    });
  }

  prepareEmbeddingJob(job: EvolutionJobRecord): PreparedEmbeddingJob | null {
    const memory = job.targetMemoryId ? this.deps.repos.memories.get(job.targetMemoryId) : undefined;
    if (!memory) throw new Error(`embedding target not found: ${job.targetMemoryId ?? "unknown"}`);
    if (!processingJobMatchesMemory(job, memory)) return null;

    const span = spanPayload(memory);
    if (span) {
      const vectorField = spanVectorFieldFromPayload(job.payload.vectorField);
      if (!vectorField) {
        throw new Error(`span embedding job missing vector field: ${job.id}`);
      }
      const text = spanEmbeddingText(memory, vectorField);
      const sourceHash = spanEmbeddingSourceHash(memory, vectorField);
      if (!text || !sourceHash) return null;
      if (typeof job.payload.sourceHash === "string" && job.payload.sourceHash !== sourceHash) return null;
      return { job, memory, text, role: "document", vectorField, sourceHash };
    }

    if (memory.memoryLayer === "L1") {
      if (memoryNeedsImportSummary(memory)) {
        this.deps.enqueueImportSummaryIfMissing(memory, this.deps.nowIso());
        return null;
      }
      const text = traceSummaryEmbeddingText(memory);
      if (!text) {
        this.deps.enqueueImportSummaryIfMissing(memory, this.deps.nowIso());
        return null;
      }
      return { job, memory, text, role: "document", vectorField: "vec_summary" };
    }

    const text = embeddingTextForMemory(memory);
    return {
      job,
      memory,
      text,
      role: "query",
      vectorField: "vec",
      sourceHash: memory.memoryLayer === "Skill" || memory.memoryLayer === "L3"
        ? retrievalDocumentSourceHash(memory)
        : undefined
    };
  }

  applyEmbeddingVector(item: PreparedEmbeddingJob, vector: number[]): void {
    const current = this.deps.repos.memories.get(item.memory.id);
    if (!current) throw new Error(`embedding target not found: ${item.memory.id}`);
    if (!processingJobMatchesMemory(item.job, current)) return;
    const span = spanPayload(current);
    const spanField = spanVectorFieldFromPayload(item.vectorField);
    if (span && spanField && item.sourceHash && spanEmbeddingSourceHash(current, spanField) !== item.sourceHash) return;
    if (!span && item.sourceHash && retrievalDocumentSourceHash(current) !== item.sourceHash) return;
    this.persistEmbeddingVector({
      memoryId: current.id,
      vectorField: item.vectorField,
      vector,
      attemptCount: item.job.attempts,
      source: "worker.embedding",
      sourceHash: item.sourceHash,
      allowedProcessingStates: ["embedding_pending", "embedding"],
      finalize: (saved, hadProcessing, at) => {
        if (hadProcessing) this.deps.repos.runtime.completeJob(item.job.id, at);
        this.enqueueSpanClusterIfReady(saved, item.job, at);
      }
    });
  }

  persistEmbeddingVector(input: PersistEmbeddingVectorInput): MemoryRow {
    const current = this.deps.repos.memories.get(input.memoryId);
    if (!current) throw new Error(`embedding target not found: ${input.memoryId}`);
    const at = this.deps.nowIso();
    let saved = current;
    this.deps.repos.transaction(() => {
      const vectorized = updateMemoryVectorField(current, input.vectorField, input.vector, {
        model: this.deps.embedder.config.model ?? this.deps.embedder.config.provider,
        provider: this.deps.embedder.config.provider,
        updatedAt: at,
        sourceHash: input.sourceHash
      });
      saved = this.deps.repos.memories.updateMaintenance(
        current.memoryLayer === "L1" && !spanPayload(current)
          ? updateImportPipelineStatus(vectorized, "indexed", at)
          : vectorized
      );
      const hadProcessing = Boolean(this.deps.repos.processing.get(saved.id));
      if (hadProcessing) {
        this.deps.repos.processing.update(saved.id, {
          state: "ready", stage: null, activeJobId: null, attemptCount: input.attemptCount,
          retryAction: "retry", errorCode: null, errorMessage: null, failedAt: null, updatedAt: at
        }, input.allowedProcessingStates ?? ["embedding_pending", "embedding"]);
      }
      this.appendMemoryChange(saved, current, input.source, at);
      input.finalize?.(saved, hadProcessing, at);
    });
    return saved;
  }

  enqueueEmbeddingRetryAfterFailure(item: PreparedEmbeddingJob, error: unknown): boolean {
    if (this.deps.repos.processing.get(item.memory.id)) return false;
    const retry = this.deps.enqueueEmbeddingRetry(
      item.memory,
      item.text,
      this.deps.nowIso(),
      item.vectorField
    );
    this.deps.appendEmbeddingRetryChange(retry, "queued", undefined, {
      code: "embedding_error",
      message: error instanceof Error ? error.message : String(error)
    });
    return true;
  }

  async summarizeImportedTrace(job: EvolutionJobRecord): Promise<void> {
    const memory = job.targetMemoryId ? this.deps.repos.memories.get(job.targetMemoryId) : undefined;
    if (!memory || memory.memoryLayer !== "L1") {
      throw new Error(`import summary target not found: ${job.targetMemoryId ?? "unknown"}`);
    }
    if (!processingJobMatchesMemory(job, memory)) return;
    const trace = traceMetaFromMemory(memory);
    if (!trace) throw new Error(`import trace payload is missing: ${memory.id}`);

    const generated = this.deps.llm.isConfigured()
      ? await this.deps.summarizeTraceForCapture({ trace, userText: trace.userText, agentText: trace.agentText, toolCalls: trace.toolCalls, reflectionText: "" }, { strict: true })
      : fallbackImportSummary(trace, memory);
    const summary = firstRealSummary(generated) ?? fallbackImportSummary(trace, memory);
    const at = this.deps.nowIso();
    const current = this.deps.repos.memories.get(memory.id);
    if (!current || !processingJobMatchesMemory(job, current)) return;

    this.deps.repos.transaction(() => {
      const previous = current;
      const next = updateImportPipelineStatus(updateTraceImportSummary(current, {
        summary, alpha: IMPORT_DEFAULT_ALPHA, value: IMPORT_DEFAULT_VALUE, priority: IMPORT_DEFAULT_PRIORITY,
        tags: importStatusTags(memory.tags, "indexing"), updatedAt: at
      }), "indexing", at);
      const saved = this.deps.repos.memories.update(next);
      this.appendMemoryChange(saved, previous, "worker.import_summary", at);
      this.scheduleEmbeddingAfterTextUpdate({
        memory: saved,
        sourceJob: job,
        reason: "import.summary.updated",
        vectorField: "vec_summary",
        clearExistingVector: false,
        allowedProcessingStates: ["summary_pending", "summarizing"],
        textOnlyAttemptCount: 0,
        at
      });
    });
  }

  async summarizeCapturedTrace(job: EvolutionJobRecord): Promise<void> {
    const memory = job.targetMemoryId ? this.deps.repos.memories.get(job.targetMemoryId) : undefined;
    if (!memory || memory.memoryLayer !== "L1" || memoryHasImportPipeline(memory)) {
      throw new Error(`trace summary target is invalid: ${job.targetMemoryId ?? "unknown"}`);
    }
    if (!processingJobMatchesMemory(job, memory)) return;
    const trace = traceMetaFromMemory(memory);
    if (!trace) throw new Error(`trace payload is missing: ${memory.id}`);

    const decideCapture = job.payload.decideCapture === true;
    const proposedDecision = decideCapture
      ? await this.deps.decideTurnMemoryForCapture({
          trace,
          userText: trace.userText,
          agentText: trace.agentText,
          toolCalls: trace.toolCalls,
          reflectionText: ""
        })
      : undefined;
    const proposedSummary = proposedDecision
      ? proposedDecision.l1Summary
      : this.deps.llm.isConfigured()
        ? await this.deps.summarizeTraceForCapture({ trace, userText: trace.userText, agentText: trace.agentText, toolCalls: trace.toolCalls, reflectionText: "" }, { strict: true })
        : trace.summary || fallbackTraceSummary(trace);
    const at = this.deps.nowIso();
    const current = this.deps.repos.memories.get(memory.id);
    if (!current || !processingJobMatchesMemory(job, current)) return;
    const currentTrace = traceMetaFromMemory(current);
    if (!currentTrace) throw new Error(`trace payload is missing: ${current.id}`);
    const decision = proposedDecision
      ? constrainTurnMemoryDecision(proposedDecision, current, currentTrace)
      : undefined;
    const summary = decision?.l1Summary ?? proposedSummary;

    this.deps.repos.transaction(() => {
      if (decision?.createUserMemory && job.payload.captureUserMemory === true) {
        this.captureUserMemoryFromDecision(current, currentTrace, decision.userMemoryTypes, job, at);
      }
      if (decision && !decision.createL1) {
        const rejected = this.deps.repos.memories.update(
          recordTurnMemoryDecision(current, decision, "rejected", at)
        );
        const deleted = this.deps.repos.memories.softDelete(rejected.id, at);
        if (deleted) this.appendMemoryChange(deleted, current, "worker.turn_memory_decision.rejected", at);
        this.deps.repos.processing.delete(current.id);
        return;
      }
      const previous = current;
      const summarized = summary.trim() && summary.trim() !== currentTrace.summary.trim()
        ? this.deps.repos.memories.update(updateTraceSummary(current, { summary: summary.trim(), updatedAt: at }))
        : previous;
      const saved = decision
        ? this.deps.repos.memories.update(acceptTurnMemoryDecision(summarized, decision, at))
        : summarized;
      if (saved !== previous) this.appendMemoryChange(saved, previous, "worker.trace_summary", at);
      this.scheduleEmbeddingAfterTextUpdate({
        memory: saved,
        sourceJob: job,
        reason: "trace.summary.updated",
        vectorField: "vec_summary",
        clearExistingVector: false,
        allowedProcessingStates: ["summary_pending", "summarizing"],
        textOnlyAttemptCount: job.attempts,
        at
      });
    });
  }

  private captureUserMemoryFromDecision(
    sourceMemory: MemoryRow,
    trace: TraceMeta,
    memoryTypes: UserMemoryType[],
    job: EvolutionJobRecord,
    at: string
  ): void {
    const content = trace.userText.trim();
    const sourceTurnId = trace.rawTurnId;
    if (!content || !sourceTurnId || memoryTypes.length === 0) return;
    const sourceAt = Number.isFinite(trace.ts) ? new Date(trace.ts).toISOString() : at;
    const candidate = buildUserMemory({
      id: `user_memory_${stableHash(`${sourceTurnId}:${content}`).slice(0, 20)}`,
      sourceTurnId,
      userId: sourceMemory.userId,
      memoryTypes,
      content,
      createdAt: sourceAt
    });
    const upsert = this.deps.repos.userMemories.upsertExact(candidate);
    this.deps.repos.runtime.appendChange({
      memoryId: upsert.memory.id,
      kind: "user_memory",
      op: upsert.created ? "created" : "updated",
      entityId: upsert.memory.id,
      userId: upsert.memory.userId,
      changeType: upsert.created ? "user_memory_created" : "user_memory_updated",
      before: upsert.previous,
      after: upsert.memory,
      source: "worker.turn_memory_decision",
      createdAt: at
    });
    if (!upsert.created || !this.deps.capture.embedAfterCapture) return;
    this.deps.enqueueJob({
      jobType: "user_memory_embedding",
      userId: upsert.memory.userId,
      sessionId: sourceMemory.sessionId,
      episodeId: job.episodeId,
      targetMemoryId: upsert.memory.id,
      payload: { contentHash: stableHash(upsert.memory.content) },
      maxAttempts: 6,
      createdAt: at
    });
  }

  scheduleEmbeddingAfterTextUpdate(input: ScheduleEmbeddingAfterTextUpdateInput): void {
    const { memory, sourceJob, allowedProcessingStates } = input;
    const at = input.at ?? this.deps.nowIso();
    if (input.clearExistingVector && this.deps.repos.processing.get(memory.id)) {
      this.deps.repos.memories.deleteVector(memory.id, input.vectorField);
    }
    if (!this.deps.capture.embedAfterCapture) {
      this.markReadyTextOnly(memory, input.textOnlyAttemptCount, at, allowedProcessingStates);
      return;
    }
    const embeddingJob = this.enqueueEmbeddingJob(memory, sourceJob, at, input.reason);
    this.markEmbeddingPending(memory, embeddingJob.id, at, allowedProcessingStates);
  }

  private enqueueEmbeddingJob(memory: MemoryRow, source: EvolutionJobRecord, at: string, reason: string): EvolutionJobRecord {
    return this.deps.enqueueJob({
      jobType: "embedding", userId: memory.userId, sessionId: memory.sessionId, episodeId: source.episodeId,
      targetMemoryId: memory.id,
      payload: { reason, sourceJobId: source.id, contentHash: memory.contentHash },
      maxAttempts: 6, createdAt: at
    });
  }

  private enqueueSpanClusterIfReady(memory: MemoryRow, sourceJob: EvolutionJobRecord, at: string): void {
    if (!spanHasBothEmbeddings(memory)) return;
    const scopeId = namespaceIdFromMemory(memory);
    this.deps.enqueueJob({
      jobType: "span_cluster",
      userId: memory.userId,
      sessionId: memory.sessionId,
      episodeId: sourceJob.episodeId,
      payload: {
        reason: "span.embeddings.ready",
        sourceJobId: sourceJob.id,
        scopeId,
        algorithmVersion: "span-cluster.v1"
      },
      createdAt: at
    });
  }

  private markReadyTextOnly(memory: MemoryRow, attemptCount: number, at: string, allowedStates?: MemoryProcessingState[]): void {
    this.deps.repos.processing.update(memory.id, {
      state: "ready_text_only", stage: null, activeJobId: null, attemptCount, retryAction: "retry",
      errorCode: null, errorMessage: null, failedAt: null, updatedAt: at
    }, allowedStates);
  }

  private markEmbeddingPending(memory: MemoryRow, activeJobId: string, at: string, allowedStates?: MemoryProcessingState[]): void {
    this.deps.repos.processing.update(memory.id, {
      state: "embedding_pending", stage: "embedding", activeJobId, attemptCount: 0, retryAction: "retry",
      updatedAt: at
    }, allowedStates);
  }

  private appendMemoryChange(after: MemoryRow, before: MemoryRow, source: string, createdAt: string): void {
    this.deps.repos.runtime.appendChange({
      memoryId: after.id, namespaceId: namespaceIdFromMemory(after), kind: kindFromMemory(after), op: "updated",
      entityId: after.id, userId: after.userId, changeType: "update", before, after, source, createdAt
    });
  }
}

function spanVectorFieldFromPayload(value: unknown): SpanEmbeddingVectorField | undefined {
  return value === "vec_goal" || value === "vec_policy" ? value : undefined;
}

export function updateTraceSummary(memory: MemoryRow, input: { summary: string; updatedAt: string }): MemoryRow {
  const trace = traceMetaFromMemory(memory);
  if (!trace) return memory;
  const internalTrace = isRecord(memory.properties.internal_info.trace) ? memory.properties.internal_info.trace : {};
  const nextTrace = { ...internalTrace, summary: input.summary, summary_at: input.updatedAt };
  return { ...memory, memoryValue: renderTraceMemoryValue({
    summary: input.summary, rawTurnId: stringFromRecord(internalTrace, "raw_turn_id"), stepIndex: numberFromRecord(internalTrace, "step_index"),
    userText: trace.userText, agentText: trace.agentText, toolCalls: trace.toolCalls,
    reflection: { text: trace.reflection, alpha: trace.alpha }, value: trace.value, priority: trace.priority
  }), info: { ...memory.info, summary: input.summary }, properties: {
    ...memory.properties, info: { ...(memory.properties.info ?? {}), summary: input.summary },
    internal_info: { ...memory.properties.internal_info, summary: input.summary, trace: nextTrace }
  }, updatedAt: input.updatedAt };
}

function acceptTurnMemoryDecision(
  memory: MemoryRow,
  decision: TurnCaptureDecision,
  updatedAt: string
): MemoryRow {
  const internal = memory.properties.internal_info;
  const pending = isRecord(internal.capture_decision) ? internal.capture_decision : {};
  const originalEvidenceStatus = typeof pending.original_evidence_status === "string"
    ? pending.original_evidence_status
    : undefined;
  const { evidence_status: _infoEvidenceStatus, ...info } = memory.info;
  const { evidence_status: _propertyInfoEvidenceStatus, ...propertyInfo } = memory.properties.info ?? {};
  const { evidence_status: _internalEvidenceStatus, ...internalWithoutEvidence } = internal;
  return {
    ...memory,
    status: "activated",
    info: {
      ...info,
      ...(originalEvidenceStatus ? { evidence_status: originalEvidenceStatus } : {})
    },
    properties: {
      ...memory.properties,
      status: "activated",
      info: {
        ...propertyInfo,
        ...(originalEvidenceStatus ? { evidence_status: originalEvidenceStatus } : {})
      },
      internal_info: {
        ...internalWithoutEvidence,
        ...(originalEvidenceStatus ? { evidence_status: originalEvidenceStatus } : {}),
        capture_decision: recordTurnMemoryDecisionFields(pending, decision, "accepted", updatedAt)
      }
    },
    updatedAt
  };
}

function recordTurnMemoryDecision(
  memory: MemoryRow,
  decision: TurnCaptureDecision,
  status: "accepted" | "rejected",
  updatedAt: string
): MemoryRow {
  const internal = memory.properties.internal_info;
  const pending = isRecord(internal.capture_decision) ? internal.capture_decision : {};
  return {
    ...memory,
    properties: {
      ...memory.properties,
      internal_info: {
        ...internal,
        capture_decision: recordTurnMemoryDecisionFields(pending, decision, status, updatedAt)
      }
    },
    updatedAt
  };
}

function recordTurnMemoryDecisionFields(
  pending: Record<string, unknown>,
  decision: TurnCaptureDecision,
  status: "accepted" | "rejected",
  updatedAt: string
): Record<string, unknown> {
  return {
    ...pending,
    status,
    create_l1: decision.createL1,
    create_user_memory: decision.createUserMemory,
    user_memory_types: decision.userMemoryTypes,
    user_memory_evidence: decision.userMemoryEvidence,
    l1_evidence: decision.l1Evidence.map((item) => ({
      quote: item.quote,
      source_role: item.sourceRole,
      kind: item.kind
    })),
    reason: decision.reason,
    decided_at: updatedAt
  };
}

function constrainTurnMemoryDecision(
  decision: TurnCaptureDecision,
  memory: MemoryRow,
  trace: TraceMeta
): typeof decision {
  const text = trace.userText.trim();
  const inferredTypes = classifyUserMemory(text);
  const evidenceTypes = decision.userMemoryEvidence.map((item) => item.type);
  const groundedUserMemoryTypes = uniq([...inferredTypes, ...evidenceTypes]);
  const userMemoryTypes = groundedUserMemoryTypes.length > 0
    ? groundedUserMemoryTypes
    : decision.userMemoryTypes;
  const dynamicCurrent = isDynamicCurrentFactQuery(text);
  const userMemoryQuestion = isUserMemoryQuestion(text);
  const taskLinkedFeedback = isTaskLinkedUserFeedback(text);
  const verifiedToolObservation = hasVerifiedDurableToolObservation(memory, trace, dynamicCurrent);
  const taskOutcome = taskLinkedFeedback && hasTaskOutcomeEvidence(trace);

  const createUserMemory = !dynamicCurrent && !userMemoryQuestion && userMemoryTypes.length > 0 &&
    decision.createUserMemory && decision.userMemoryEvidence.length > 0;
  let createL1 = decision.createL1 && decision.l1Evidence.length > 0;
  const guards: string[] = [];
  if (decision.createUserMemory && decision.userMemoryEvidence.length === 0) {
    guards.push("user-memory-evidence-missing");
  }
  if (decision.createL1 && decision.l1Evidence.length === 0) {
    guards.push("l1-evidence-missing");
  }
  if (dynamicCurrent) {
    createL1 = false;
    guards.push("dynamic-current");
  } else if (verifiedToolObservation) {
    createL1 = true;
    guards.push("verified-tool-evidence");
  } else if (userMemoryQuestion) {
    createL1 = false;
    guards.push("user-memory-question");
  } else if (taskLinkedFeedback) {
    createL1 = taskOutcome;
    guards.push(taskOutcome ? "task-outcome" : "feedback-without-outcome");
  }

  return {
    ...decision,
    createL1,
    l1Summary: createL1 ? decision.l1Summary.trim() || fallbackTraceSummary(trace) : "",
    createUserMemory,
    userMemoryTypes: createUserMemory ? userMemoryTypes : [],
    reason: clip([decision.reason, guards.length > 0 ? `guards=${guards.join(",")}` : ""].filter(Boolean).join("; "), 300)
  };
}

function hasVerifiedDurableToolObservation(
  memory: MemoryRow,
  trace: TraceMeta,
  dynamicCurrent: boolean
): boolean {
  if (dynamicCurrent || trace.toolCalls.length === 0) return false;
  const captureDecision = isRecord(memory.properties.internal_info.capture_decision)
    ? memory.properties.internal_info.capture_decision
    : {};
  if (captureDecision.original_evidence_status !== "verified") return false;
  return trace.toolCalls.some((call) => !isMemmyRecallToolName(call.name));
}

function hasTaskOutcomeEvidence(trace: TraceMeta): boolean {
  if (trace.toolCalls.some((call) => !call.error && (call.output !== undefined || call.success === true))) return true;
  return /(?:已|已经|完成|修改|改为|精简|修复|通过(?:了)?测试|验证成功)|\b(?:completed|updated|changed|simplified|fixed|tests? passed|verified successfully)\b/i
    .test(trace.agentText);
}

function uniq<T>(values: readonly T[]): T[] {
  return [...new Set(values)];
}

function fallbackImportSummary(trace: TraceMeta, memory: MemoryRow): string {
  const title = stringFromRecord(memory.info, "title");
  const summary = [trace.userText, trace.agentText, title].map((value) => firstLine(value ?? "")).find((value) => value && !isImportSummaryPlaceholder(value));
  return clip(summary || "导入记忆", 200);
}

function fallbackTraceSummary(trace: TraceMeta): string {
  return clip(firstLine([trace.summary, trace.userText, trace.agentText].filter(Boolean).join("\n")) || "trace memory", 200);
}

function renderTraceMemoryValue(step: { summary: string; rawTurnId?: string; stepIndex?: number; userText?: string; agentText?: string; toolCalls: Array<{ name: string; input?: unknown; output?: unknown; error?: string }>; reflection: { text: string | null; alpha: number }; value: number; priority: number }): string {
  return [
    `Summary: ${step.summary}`, step.rawTurnId ? `RawTurn: ${step.rawTurnId}` : undefined,
    typeof step.stepIndex === "number" ? `TraceStep: ${step.stepIndex}` : undefined,
    step.userText ? `User:\n${step.userText}` : undefined,
    step.toolCalls.length ? ["Tool calls:", ...step.toolCalls.map((call) => `- ${call.name}${call.error ? ` error=${clip(call.error, 160)}` : ""}`)].join("\n") : undefined,
    step.agentText ? `Agent:\n${step.agentText}` : undefined,
    step.reflection.text ? `Reflection: ${clip(step.reflection.text, 800)}` : undefined,
    `Alpha: ${step.reflection.alpha}`, `Value: ${step.value}`, `Priority: ${step.priority}`
  ].filter(Boolean).join("\n");
}

function namespaceIdFromMemory(memory: MemoryRow): string {
  const namespace = namespaceForMemory(memory);
  return [namespace.tenantId, namespace.userId, namespace.projectId ?? namespace.workspaceId, namespace.source, namespace.profileId].filter(Boolean).join(":");
}
function stringFromRecord(record: Record<string, unknown>, key: string): string | undefined { const value = record[key]; return typeof value === "string" ? value : undefined; }
function numberFromRecord(record: Record<string, unknown>, key: string): number | undefined { const value = record[key]; return typeof value === "number" && Number.isFinite(value) ? value : undefined; }
