import type { MemmyConfig } from "../../config/index.js";
import type { LlmClient } from "../../model/types.js";
import type { EpisodeRecord, RawTurnRecord } from "../../storage/repositories.js";
import type { ToolCallPayload } from "../../types.js";
import { stableHash, stableStringify } from "../../utils/id.js";
import { isRecord } from "../../utils/json.js";
import { redactSensitiveText } from "../../utils/sensitive-data.js";
import { clip } from "../../utils/text.js";
import { inferToolOutcome } from "../turn/tool-outcome.js";
import type {
  EpisodeExecutionPathLiteV1,
  ExecutionStepLiteV1,
  TurnTransitionV1
} from "./procedural-window-model.js";
import {
  EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
  EXECUTION_STEP_LITE_SCHEMA_VERSION,
  TURN_TRANSITION_SCHEMA_VERSION
} from "./procedural-window-model.js";

export const EPISODE_PATH_COMPILER_VERSION = "episode-path-compiler-lite.v3" as const;
export const STEP_SEMANTICS_VERSION = "execution-step-semantics-lite.v3" as const;

const STEP_SEMANTICS_OPERATION = "procedural.step_semantics.v3";
const MAX_SEMANTIC_REPAIR_ATTEMPTS = 2;
const STEP_WINDOW_MAX_CANDIDATES = 30;
const STEP_WINDOW_INPUT_CHAR_BUDGET = 30_000;
const STEP_WINDOW_OVERLAP = 5;
const PREVIOUS_TURN_CONTEXT_COUNT = 3;
const TURN_CONTEXT_TEXT_MAX = 700;
const STEP_CONTEXT_TEXT_MAX = 500;
const TEXT_PREVIEW_MAX = 1_500;
const INPUT_PREVIEW_MAX = 160;
const OUTPUT_PREVIEW_MAX = 220;
const ERROR_PREVIEW_MAX = 160;
const EVIDENCE_MAX = 420;

/**
 * Kept byte-for-byte compatible with the v15 reconstruction.v8 experiment.
 * The Lite compiler changes persistence, not the Step-semantics contract.
 */
export const STEP_SEMANTICS_PROMPT = `You reconstruct evidence-grounded, reusable execution-step semantics from an agent episode.

Each supplied step candidate is one observable action and its immediate result. Compress it into:
- intent: the reusable atomic operation the action attempted;
- summary: the normalized immediate result or state effect established by the supplied evidence.

Normalize incidental instance details so semantically equivalent Steps can be recognized across Episodes:
- Abstract project names, filenames, exact paths, URLs, domains, record IDs, literal values, and raw error strings when they do not change the reusable operation.
- Preserve distinctions that change the procedure: inspect vs create vs edit vs execute vs verify; searching vs fetching; checking existence vs validating content; and success vs failure.
- Preserve the result category or discovered constraint when it changes the next decision, such as access denied, missing dependency, invalid structure, execution failure, or verified artifact.
- Use the narrowest reusable wording supported by the evidence. Do not collapse the intent into vague phrases such as "handle the task", "process data", or "work on the file".

Examples of the intended abstraction level:
- "fetch historical data from MarketWatch" with "HTTP 401 Unauthorized" becomes intent "retrieve historical market data from an external source" and summary "data retrieval failed because access authorization was unavailable";
- "write generate_report.py" with "file created" becomes intent "create an artifact-generation script" and summary "the generation script source was created";
- writing a script, executing it, checking that its output exists, and validating its content remain four different operations.

Do not extract task state, observations, state operations, retries, recoveries, outcome labels, or evidence references. Those fields are derived elsewhere from immutable inputs. Do not group candidates into subtasks in this pass.

Tool-action candidates are procedural and must include=true. A response-generation candidate may include=false only when it is purely social or reports no substantive task effect. Only objects inside stepCandidates are output candidates; precedingStepContext is read-only continuity context and must never be returned. Every stepCandidates item must appear exactly once.

Return JSON only:
{
  "steps": [
    {
      "candidate_id": "candidate_...",
      "include": true,
      "intent": "inspect the failing dependency resolution",
      "summary": "Read the lockfile and observed incompatible versions."
    }
  ]
}`;

interface StepCandidate {
  id: string;
  episodeId: string;
  rawTurnId: string;
  turnIndex: number;
  eventIndex: number;
  kind: ExecutionStepLiteV1["kind"];
  action: NormalizedStepOperator | "respond";
  toolName?: string;
  toolCallIndex?: number;
  toolCall?: ToolCallPayload;
  heuristicSuccess: boolean;
  outcome: ExecutionStepLiteV1["outcome"];
  outcomeReason?: string;
  errorCode?: string;
  evidenceRefs: string[];
  promptSourceRefs: string[];
  evidence?: string;
}

interface TurnFrame {
  rawTurn: RawTurnRecord;
  turnIndex: number;
  candidates: StepCandidate[];
}

type NormalizedStepOperator =
  | "search"
  | "fetch"
  | "read"
  | "write"
  | "execute"
  | "edit"
  | "verify"
  | "communicate"
  | "other";

interface StepSemantic {
  candidateId: string;
  include: boolean;
  intent: string;
  summary: string;
}

export interface EpisodePathCompilerDeps {
  config: MemmyConfig;
  llm: LlmClient;
}

export class EpisodePathCompiler {
  constructor(private readonly deps: EpisodePathCompilerDeps) {}

  async compile(input: {
    episode: EpisodeRecord;
    rawTurns: RawTurnRecord[];
    sourceSnapshotHash?: string;
  }): Promise<EpisodeExecutionPathLiteV1 | null> {
    const rawTurns = orderRawTurns(input.episode, input.rawTurns)
      .filter((turn) => !turn.deletedAt && !turn.redactedAt);
    const allFrames = rawTurns.map((rawTurn, turnIndex): TurnFrame => ({
      rawTurn,
      turnIndex,
      candidates: buildTurnCandidates(input.episode.id, rawTurn, turnIndex)
    }));
    const allCandidates = allFrames.flatMap((frame) => frame.candidates);
    if (allCandidates.length === 0) return null;
    const config = this.deps.config.algorithm.proceduralWindow;
    const candidates = allCandidates.slice(0, Math.max(1, config.maxStepsPerEpisode));
    const includedCandidateIds = new Set(candidates.map((candidate) => candidate.id));
    const frames = allFrames.map((frame) => ({
      ...frame,
      candidates: frame.candidates.filter((candidate) => includedCandidateIds.has(candidate.id))
    }));
    const sourceSnapshotHash = input.sourceSnapshotHash ?? stableHash({
      compilerVersion: EPISODE_PATH_COMPILER_VERSION,
      episodeId: input.episode.id,
      rawTurns: rawTurns.map((turn) => ({
        id: turn.id,
        turnId: turn.turnId,
        createdAt: turn.createdAt,
        userText: turn.userText,
        assistantText: turn.assistantText,
        toolCalls: turn.toolCalls,
        toolResults: turn.toolResults,
        status: turn.status
      }))
    });

    const semantics: StepSemantic[] = [];
    const processedCandidates: StepCandidate[] = [];
    const semanticsByCandidate = new Map<string, StepSemantic>();
    for (const frame of frames) {
      const windows = chunkStepCandidates(frame.candidates);
      for (const [windowIndex, windowCandidates] of windows.entries()) {
        const precedingCandidates = processedCandidates.slice(-STEP_WINDOW_OVERLAP);
        const parsed = await this.summarizeWindow({
          episodeId: input.episode.id,
          frame,
          previousTurns: frames.slice(
            Math.max(0, frame.turnIndex - PREVIOUS_TURN_CONTEXT_COUNT),
            frame.turnIndex
          ),
          windowIndex,
          precedingCandidates,
          candidates: windowCandidates,
          semanticsByCandidate
        });
        semantics.push(...parsed);
        for (const semantic of parsed) semanticsByCandidate.set(semantic.candidateId, semantic);
        processedCandidates.push(...windowCandidates);
      }
    }
    const semanticByCandidate = new Map(semantics.map((item) => [item.candidateId, item]));
    const steps: ExecutionStepLiteV1[] = [];
    let previousIncludedCandidate: StepCandidate | undefined;
    for (const candidate of candidates) {
      const semantic = semanticByCandidate.get(candidate.id);
      if (!semantic) {
        throw new Error(`canonical Step semantics missing candidate ${candidate.id}`);
      }
      if (!semantic.include) continue;
      const previous = steps.at(-1);
      const previousFailed = previous?.outcome === "failure";
      const repeatsSameAction = previousFailed && previousIncludedCandidate &&
        previousIncludedCandidate.action === candidate.action &&
        previousIncludedCandidate.toolName === candidate.toolName;
      const retryOfStepId = repeatsSameAction ? previous.id : undefined;
      const recoveryFromStepId = previousFailed && !retryOfStepId ? previous.id : undefined;
      const stepIndex = steps.length;
      steps.push({
        id: `execution_step_${stableHash({
          sourceSnapshotHash,
          candidateId: candidate.id,
          semantic: {
            intent: semantic.intent,
            summary: semantic.summary
          }
        }).slice(0, 20)}`,
        schemaVersion: EXECUTION_STEP_LITE_SCHEMA_VERSION,
        episodeId: candidate.episodeId,
        rawTurnId: candidate.rawTurnId,
        turnIndex: candidate.turnIndex,
        stepIndex,
        kind: candidate.kind,
        ...(candidate.toolName ? { toolName: candidate.toolName } : {}),
        ...(candidate.toolCallIndex === undefined
          ? {}
          : { toolCallIndex: candidate.toolCallIndex }),
        intent: semantic.intent,
        summary: semantic.summary,
        outcome: candidate.outcome,
        ...(candidate.errorCode ? { errorCode: candidate.errorCode } : {}),
        ...(retryOfStepId ? { retryOfStepId } : {}),
        ...(recoveryFromStepId ? { recoveryFromStepId } : {}),
        evidenceRefs: candidate.evidenceRefs,
        provenance: {
          algorithmVersion: STEP_SEMANTICS_VERSION,
          ...(this.deps.llm.isConfigured() && this.deps.llm.config.model
            ? { model: this.deps.llm.config.model }
            : {}),
          sourceSnapshotHash
        }
      });
      previousIncludedCandidate = candidate;
    }
    if (steps.length === 0) return null;

    const turnTransitions = buildTurnTransitions(rawTurns, steps);
    const modelSignature = this.deps.llm.isConfigured()
      ? `${this.deps.llm.config.provider}:${this.deps.llm.config.model ?? "unknown"}`
      : "deterministic-fallback";
    const pathHash = stableHash({
      schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
      episodeId: input.episode.id,
      userId: input.episode.userId,
      sourceSnapshotHash,
      compilerVersion: EPISODE_PATH_COMPILER_VERSION,
      modelSignature,
      steps,
      turnTransitions
    });
    return {
      id: `episode_execution_path_${pathHash.slice(0, 20)}`,
      schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
      episodeId: input.episode.id,
      userId: input.episode.userId,
      sourceRawTurnIds: rawTurns.map((turn) => turn.id),
      steps,
      turnTransitions,
      terminalReward: input.episode.rTask,
      sourceSnapshotHash,
      pathHash,
      compilerVersion: EPISODE_PATH_COMPILER_VERSION,
      modelSignature,
      provenance: {
        algorithmVersion: EPISODE_PATH_COMPILER_VERSION,
        ...(this.deps.llm.isConfigured() && this.deps.llm.config.model
          ? { model: this.deps.llm.config.model }
          : {}),
        inputCandidateCount: allCandidates.length,
        compiledCandidateCount: candidates.length,
        truncated: candidates.length < allCandidates.length
      }
    };
  }

  private async summarizeWindow(input: {
    episodeId: string;
    frame: TurnFrame;
    previousTurns: TurnFrame[];
    windowIndex: number;
    precedingCandidates: StepCandidate[];
    candidates: StepCandidate[];
    semanticsByCandidate: ReadonlyMap<string, StepSemantic>;
  }): Promise<StepSemantic[]> {
    if (!this.deps.llm.isConfigured()) {
      throw new Error("canonical Step semantics require a configured LLM");
    }
    const messages = [
      { role: "system" as const, content: STEP_SEMANTICS_PROMPT },
      {
        role: "user" as const,
        content: stableStringify({
          episodeId: input.episodeId,
          window: {
            rawTurnId: input.frame.rawTurn.id,
            turnIndex: input.frame.turnIndex,
            windowIndex: input.windowIndex,
            maxNewCandidates: STEP_WINDOW_MAX_CANDIDATES
          },
          previousTurns: input.previousTurns.map(compactTurnFrame),
          currentTurn: compactTurnFrame(input.frame),
          precedingStepContext: input.precedingCandidates.map((candidate) => {
            const semantic = input.semanticsByCandidate.get(candidate.id);
            return {
              action: candidate.action,
              toolName: candidate.toolName,
              heuristicSuccess: candidate.heuristicSuccess,
              intent: semantic?.intent,
              summary: contextText(semantic?.summary)
            };
          }),
          stepCandidates: input.candidates.map(candidatePromptValue)
        })
      }
    ];
    const options = {
      operation: `${STEP_SEMANTICS_OPERATION}.window`,
      thinkingMode: "disabled" as const,
      temperature: 0,
      maxTokens: 16_000
    };
    let result = await this.deps.llm.completeJson<{ steps?: unknown }>(messages, options);
    for (let repairAttempt = 0; ; repairAttempt += 1) {
      try {
        return validateSemantics(result.steps, input.candidates);
      } catch (error) {
        if (repairAttempt >= MAX_SEMANTIC_REPAIR_ATTEMPTS) throw error;
        const repairNumber = repairAttempt + 1;
        result = await this.deps.llm.completeJson<{ steps?: unknown }>([
          ...messages,
          { role: "assistant", content: stableStringify(result) },
          {
            role: "user",
            content: stepSemanticRepairInstruction(error, repairNumber, input.candidates)
          }
        ], {
          ...options,
          operation: `${STEP_SEMANTICS_OPERATION}.repair.${repairNumber}`
        });
      }
    }
  }
}

function buildTurnCandidates(
  episodeId: string,
  rawTurn: RawTurnRecord,
  turnIndex: number
): StepCandidate[] {
  const candidates: StepCandidate[] = [];
  for (const [toolCallIndex, value] of rawTurn.toolCalls.entries()) {
    if (!isToolCall(value)) continue;
    const matchedResult = matchingToolResult(value, toolCallIndex, rawTurn.toolResults);
    const toolCall = enrichToolCall(value, matchedResult);
    const inferredOutcome = inferToolOutcome({
      call: toolCall,
      result: matchedResult,
      completionObserved: matchedResult !== undefined
    });
    const v15Error = v15InferredToolError(toolCall);
    const errorCode = text(toolCall.errorCode) ?? inferredOutcome.errorCode;
    const inputPreview = previewValue(toolCall.input, INPUT_PREVIEW_MAX);
    const outputPreview = previewValue(toolCall.output, OUTPUT_PREVIEW_MAX);
    const errorPreview = v15Error ? redactClip(v15Error, ERROR_PREVIEW_MAX) : undefined;
    const evidence = evidenceSnippet({ inputPreview, outputPreview, errorPreview });
    candidates.push({
      id: stepCandidateId(rawTurn.id, "tool", toolCallIndex),
      episodeId,
      rawTurnId: rawTurn.id,
      turnIndex,
      eventIndex: candidates.length,
      kind: "tool_action",
      action: normalizedStepOperator(toolCall.name),
      toolName: redactClip(toolCall.name, 120) ?? toolCall.name,
      toolCallIndex,
      toolCall,
      heuristicSuccess: toolCall.success === false || v15Error
        ? false
        : toolCall.success ?? true,
      outcome: inferredOutcome.status,
      ...(inferredOutcome.reason ? { outcomeReason: inferredOutcome.reason } : {}),
      ...(errorCode ? { errorCode } : {}),
      evidenceRefs: [
        `raw_turn:${rawTurn.id}:tool_call:${toolCallIndex}`,
        ...(matchedResult === undefined
          ? []
          : [`raw_turn:${rawTurn.id}:tool_result:${toolCallIndex}`])
      ],
      promptSourceRefs: toolSourceRefs(rawTurn.id, toolCallIndex, rawTurn.toolResults),
      ...(evidence ? { evidence } : {})
    });
  }
  if (rawTurn.assistantText?.trim()) {
    const heuristicSuccess = rawTurn.status !== "failed";
    candidates.push({
      id: stepCandidateId(rawTurn.id, "response", candidates.length),
      episodeId,
      rawTurnId: rawTurn.id,
      turnIndex,
      eventIndex: candidates.length,
      kind: "response_generation",
      action: "respond",
      heuristicSuccess,
      outcome: heuristicSuccess ? "success" : "failure",
      evidenceRefs: [`raw_turn:${rawTurn.id}:assistant`],
      promptSourceRefs: [`turn:${rawTurn.id}:assistant`],
      evidence: preview(rawTurn.assistantText)
    });
  }
  return candidates;
}

function candidatePromptValue(candidate: StepCandidate): Record<string, unknown> {
  return {
    candidateId: candidate.id,
    rawTurnId: candidate.rawTurnId,
    turnIndex: candidate.turnIndex,
    eventIndex: candidate.eventIndex,
    kind: candidate.kind,
    action: candidate.action,
    toolName: candidate.toolName,
    heuristicSuccess: candidate.heuristicSuccess,
    sourceRefs: candidate.promptSourceRefs,
    evidence: candidate.evidence
  };
}

function chunkStepCandidates(candidates: StepCandidate[]): StepCandidate[][] {
  const windows: StepCandidate[][] = [];
  let current: StepCandidate[] = [];
  let currentChars = 0;
  for (const candidate of candidates) {
    const candidateChars = (candidate.evidence?.length ?? 0) + 350;
    if (current.length > 0 && (
      current.length >= STEP_WINDOW_MAX_CANDIDATES ||
      currentChars + candidateChars > STEP_WINDOW_INPUT_CHAR_BUDGET
    )) {
      windows.push(current);
      current = [];
      currentChars = 0;
    }
    current.push(candidate);
    currentChars += candidateChars;
  }
  if (current.length > 0) windows.push(current);
  return windows;
}

function compactTurnFrame(frame: TurnFrame): Record<string, unknown> {
  return {
    rawTurnId: frame.rawTurn.id,
    user: turnContextText(frame.rawTurn.userText),
    assistant: turnContextText(frame.rawTurn.assistantText),
    status: frame.rawTurn.status
  };
}

function turnContextText(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), TURN_CONTEXT_TEXT_MAX);
}

function contextText(value: unknown): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), STEP_CONTEXT_TEXT_MAX);
}

function stepSemanticRepairInstruction(
  error: unknown,
  repairNumber: number,
  candidates: StepCandidate[]
): string {
  const message = error instanceof Error ? error.message : String(error);
  const allowedCandidateIds = candidates.map((candidate) => candidate.id);
  const common = `Deterministic validation error: ${message}\nThe complete and exclusive allowed candidate_id list is: ${stableStringify(allowedCandidateIds)}. Return each ID from this list exactly once and return no other ID. precedingStepContext is read-only and is not output. Return the complete corrected JSON object only. Each item must contain only candidate_id, include, intent, and summary. Do not add state, outcome, retry, recovery, or evidence fields. Do not add markdown, explanation, comments, or extra top-level fields.`;
  if (repairNumber === 1) {
    return `This is a strict schema-repair task, not a request to reinterpret the episode. Correct the previous JSON.\n${common}`;
  }
  return `The previous repair still failed. Discard the previous JSON and regenerate the entire response from the original episode payload and schema. Do not patch or paraphrase the malformed object.\n${common}\nBefore returning, verify that the complete response is one JSON object and every supplied candidate appears exactly once.`;
}

function validateSemantics(value: unknown, candidates: StepCandidate[]): StepSemantic[] {
  if (!Array.isArray(value)) throw new Error("procedural step LLM returned invalid steps");
  const candidateById = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const parsed = new Map<string, StepSemantic>();
  for (const item of value) {
    if (!isRecord(item)) throw new Error("procedural step LLM returned invalid candidate ID");
    const snakeCaseCandidateId = text(item.candidate_id);
    const camelCaseCandidateId = text(item.candidateId);
    if (snakeCaseCandidateId && camelCaseCandidateId && snakeCaseCandidateId !== camelCaseCandidateId) {
      throw new Error("procedural step LLM returned conflicting candidate IDs");
    }
    const candidateId = snakeCaseCandidateId ?? camelCaseCandidateId;
    if (!candidateId) throw new Error("procedural step LLM returned invalid candidate ID");
    const candidate = candidateById.get(candidateId);
    if (!candidate) throw new Error(`procedural step LLM invented candidate: ${candidateId}`);
    if (parsed.has(candidateId)) throw new Error(`procedural step LLM duplicated candidate: ${candidateId}`);
    if (typeof item.include !== "boolean") throw new Error("procedural step LLM omitted include flag");
    if (candidate.kind === "tool_action" && item.include !== true) {
      throw new Error(`procedural step LLM excluded tool action: ${candidate.id}`);
    }
    const intent = cleanText(item.intent, 500);
    const summary = cleanText(item.summary, 1_200);
    if (item.include && (!intent || !summary)) {
      throw new Error(`procedural step LLM returned incomplete semantics: ${candidate.id}`);
    }
    parsed.set(candidateId, {
      candidateId: candidate.id,
      include: item.include,
      intent: intent ?? "non-procedural response",
      summary: summary ?? "No substantive task effect."
    });
  }
  for (const candidate of candidates) {
    if (!parsed.has(candidate.id)) throw new Error(`procedural step LLM omitted candidate: ${candidate.id}`);
  }
  return candidates.map((candidate) => parsed.get(candidate.id)!);
}

function buildTurnTransitions(
  rawTurns: RawTurnRecord[],
  steps: ExecutionStepLiteV1[]
): TurnTransitionV1[] {
  return rawTurns.map((turn, turnIndex) => {
    const turnSteps = steps.filter((step) => step.rawTurnId === turn.id);
    const first = turnSteps[0];
    const previous = steps.filter((step) => step.turnIndex < turnIndex).at(-1);
    return {
      id: `turn_transition_${stableHash({
        episodeId: steps[0]?.episodeId,
        rawTurnId: turn.id,
        turnIndex,
        userText: turn.userText ?? ""
      }).slice(0, 20)}`,
      schemaVersion: TURN_TRANSITION_SCHEMA_VERSION,
      episodeId: steps[0]!.episodeId,
      rawTurnId: turn.id,
      turnIndex,
      ...(previous ? { beforeStepIndex: previous.stepIndex } : {}),
      ...(first ? { afterStepIndex: first.stepIndex } : {}),
      userObservation: redactClip(turn.userText, 1_200),
      sourceRef: `raw_turn:${turn.id}:user`,
      goalChangeType: "unknown"
    };
  });
}

function orderRawTurns(episode: EpisodeRecord, rawTurns: RawTurnRecord[]): RawTurnRecord[] {
  const byId = new Map(rawTurns.map((turn) => [turn.id, turn]));
  const ordered = episode.rawTurnIds.map((id) => byId.get(id)).filter(isDefined);
  const seen = new Set(ordered.map((turn) => turn.id));
  return [
    ...ordered,
    ...rawTurns.filter((turn) => !seen.has(turn.id)).sort((left, right) =>
      left.createdAt.localeCompare(right.createdAt) || left.id.localeCompare(right.id))
  ];
}

function matchingToolResult(
  call: ToolCallPayload,
  index: number,
  results: unknown[]
): Record<string, unknown> | undefined {
  const callRecord = call as unknown as Record<string, unknown>;
  const callId = text(call.id) ?? text(callRecord.toolCallId);
  const name = text(call.name);
  const exact = results.find((result) => {
    if (!isRecord(result)) return false;
    const resultId = text(result.toolCallId) ?? text(result.id);
    return Boolean(callId && resultId === callId);
  });
  if (isRecord(exact)) return exact;
  const positional = results[index];
  if (!isRecord(positional)) return undefined;
  const resultName = text(positional.name) ?? text(positional.toolName);
  return !name || !resultName || name === resultName ? positional : undefined;
}

function enrichToolCall(call: ToolCallPayload, result: unknown): ToolCallPayload {
  if (!isRecord(result)) return call;
  return {
    ...call,
    ...(call.output === undefined && result.output !== undefined ? { output: result.output } : {}),
    error: call.error ?? text(result.error),
    errorCode: call.errorCode ?? text(result.errorCode) ?? text(result.error_code),
    success: call.success ?? (typeof result.success === "boolean" ? result.success : undefined)
  };
}

function toolSourceRefs(rawTurnId: string, index: number, toolResults: unknown[]): string[] {
  return [
    `turn:${rawTurnId}:tool:${index}`,
    ...(toolResults[index] === undefined ? [] : [`turn:${rawTurnId}:tool_result:${index}`])
  ];
}

function stepCandidateId(rawTurnId: string, kind: "tool" | "response", index: number): string {
  return `candidate_${stableHash({ rawTurnId, kind, index }).slice(0, 20)}`;
}

function normalizedStepOperator(toolName: string): NormalizedStepOperator {
  const name = toolName.toLowerCase();
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

function v15InferredToolError(call: ToolCallPayload): string | undefined {
  if (typeof call.error === "string" && call.error.trim()) return call.error.trim();
  if (call.success === false) return "Tool reported success=false";
  const output = call.output;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        const structuredError = v15StructuredOutputError(JSON.parse(trimmed) as unknown);
        if (structuredError) return structuredError;
      } catch {
        // Non-JSON text is evaluated by the v15 protocol markers below.
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
  return v15StructuredOutputError(output);
}

function v15StructuredOutputError(output: unknown): string | undefined {
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

function evidenceSnippet(input: {
  inputPreview?: string;
  outputPreview?: string;
  errorPreview?: string;
}): string | undefined {
  const parts = [
    input.errorPreview ? `error=${input.errorPreview}` : "",
    input.inputPreview ? `input=${input.inputPreview}` : "",
    input.outputPreview ? `output=${input.outputPreview}` : ""
  ].filter(Boolean);
  if (parts.length === 0) return undefined;
  return redactClip(parts.join(" | "), EVIDENCE_MAX);
}

function previewValue(value: unknown, maxChars: number): string | undefined {
  if (value === undefined || value === null) return undefined;
  return redactClip(stableStringify(value), maxChars);
}

function preview(value: string | undefined): string | undefined {
  if (!value?.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), TEXT_PREVIEW_MAX);
}

function isToolCall(value: unknown): value is ToolCallPayload {
  return isRecord(value) && typeof value.name === "string" && value.name.trim().length > 0;
}

function redactClip(value: unknown, maxChars: number): string | undefined {
  const normalized = typeof value === "string" ? value : value == null ? "" : stableStringify(value);
  const redacted = redactSensitiveText(normalized).trim();
  return redacted ? clip(redacted, maxChars) : undefined;
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function cleanText(value: unknown, maxChars: number): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  return clip(redactSensitiveText(value.trim()), maxChars);
}

function isDefined<T>(value: T | null | undefined): value is T {
  return value !== undefined && value !== null;
}
