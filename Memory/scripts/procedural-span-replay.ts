import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import Database from "better-sqlite3";
import {
  createLlmClient,
  EpisodeProceduralReconstructor,
  loadMemmyConfig,
  resolveEvolutionConfig,
  buildTurnStepCandidates,
  type EpisodeProceduralPathV2
} from "../src/index.js";
import type { LlmClient, LlmCompletionOptions, LlmMessage } from "../src/model/types.js";
import { Repositories, type RawTurnRecord } from "../src/storage/repositories.js";
import { redactSensitiveText } from "../src/utils/sensitive-data.js";
import { clip } from "../src/utils/text.js";

const REMOTE_STRING_MAX = 4_000;
const REMOTE_COLLECTION_MAX = 120;
const REMOTE_DEPTH_MAX = 8;
const SENSITIVE_FIELD = /api.?key|token|secret|password|authorization|credential|cookie/i;
const INTERNAL_REASONING_FIELD = /thinking|reasoning|assistant.?text.?before|chain.?of.?thought/i;

interface ReplayArgs {
  episodeId: string;
  dbPath?: string;
  configPath?: string;
  outputPath?: string;
  full: boolean;
  structuralOnly: boolean;
  traceBoundaries: boolean;
}

interface BoundaryTraceCall {
  operation: string;
  request: unknown;
  response: unknown;
}

interface ReplayArtifact {
  schemaVersion: "procedural-span-replay-artifact.v1";
  generatedAt: string;
  privacy: {
    databaseReadOnly: true;
    remoteInput: "recursively-redacted-and-clipped";
    rawCredentialsPersisted: false;
    internalReasoningPersisted: false;
  };
  replay: {
    provider: string;
    model?: string;
    endpoint?: string;
  };
  episode: {
    id: string;
    status: string;
    reward?: number;
    openedAt: string;
    closedAt?: string | null;
    turns: Array<{
      rawTurnId: string;
      createdAt: string;
      status: string;
      userText?: string;
      assistantText?: string;
      toolCallCount: number;
    }>;
  };
  executionPath: EpisodeProceduralPathV2;
  boundaryTrace?: BoundaryTraceCall[];
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const loaded = loadMemmyConfig(args.configPath);
  const dbPath = args.dbPath ?? loaded.config.storage.sqlitePath;
  if (!dbPath) throw new Error("No SQLite database path. Pass --db or configure storage.sqlitePath.");
  const db = new Database(dbPath, { readonly: true, fileMustExist: true });
  try {
    const repos = new Repositories(db);
    const episode = repos.runtime.getEpisode(args.episodeId);
    if (!episode) throw new Error(`Episode not found: ${args.episodeId}`);
    const rawTurns = repos.runtime.listRawTurnsByEpisode(args.episodeId, 1_000)
      .filter((turn) => !turn.deletedAt);
    if (rawTurns.length === 0) throw new Error(`Episode has no readable raw turns: ${args.episodeId}`);
    if (args.structuralOnly) {
      const turns = rawTurns.map((turn, turnIndex) => {
        const candidates = buildTurnStepCandidates(turn, turnIndex);
        return {
          rawTurnId: turn.id,
          rawToolCallCount: turn.toolCalls.length,
          candidateCount: candidates.length,
          toolStepCount: candidates.filter((candidate) => candidate.kind === "tool_action").length,
          responseStepCount: candidates.filter((candidate) => candidate.kind === "response_generation").length,
          heuristicFailureCount: candidates.filter((candidate) => !candidate.heuristicSuccess).length,
          candidates: candidates.map((candidate) => ({
            candidateId: candidate.id,
            kind: candidate.kind,
            action: candidate.action,
            toolName: candidate.toolName,
            heuristicSuccess: candidate.heuristicSuccess,
            toolCallIndex: candidate.toolCallIndex
          }))
        };
      });
      const audit = {
        mode: "readonly-local-procedural-structure-audit",
        episodeId: episode.id,
        turnCount: rawTurns.length,
        rawToolCallCount: rawTurns.reduce((count, turn) => count + turn.toolCalls.length, 0),
        candidateStepCount: turns.reduce((count, turn) => count + turn.candidateCount, 0),
        toolStepCount: turns.reduce((count, turn) => count + turn.toolStepCount, 0),
        responseStepCount: turns.reduce((count, turn) => count + turn.responseStepCount, 0),
        heuristicFailureCount: turns.reduce((count, turn) => count + turn.heuristicFailureCount, 0),
        turns
      };
      const persistedFile = args.outputPath ? persistJsonArtifact(args.outputPath, audit) : undefined;
      process.stdout.write(`${JSON.stringify({
        ...audit,
        ...(persistedFile ? { persistedFile } : {})
      }, null, 2)}\n`);
      return;
    }
    const configuredLlm = createLlmClient(resolveEvolutionConfig(loaded.config), {
      modelRole: "memory_evolution"
    });
    const boundaryTrace: BoundaryTraceCall[] = [];
    const llm = args.traceBoundaries
      ? traceBoundaryCalls(configuredLlm, boundaryTrace)
      : configuredLlm;
    if (!llm.isConfigured()) throw new Error("The configured evolution/summary LLM is unavailable.");
    const remoteTurns = rawTurns.map(sanitizeRawTurn);
    const path = await new EpisodeProceduralReconstructor({ llm }).reconstruct({
      episodeId: episode.id,
      rawTurns: remoteTurns,
      ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask })
    });
    const artifact: ReplayArtifact = {
      schemaVersion: "procedural-span-replay-artifact.v1",
      generatedAt: new Date().toISOString(),
      privacy: {
        databaseReadOnly: true,
        remoteInput: "recursively-redacted-and-clipped",
        rawCredentialsPersisted: false,
        internalReasoningPersisted: false
      },
      replay: {
        provider: llm.config.provider,
        ...(llm.config.model ? { model: llm.config.model } : {}),
        ...(llm.config.endpoint ? { endpoint: llm.config.endpoint } : {})
      },
      episode: {
        id: episode.id,
        status: episode.status,
        ...(episode.rTask === undefined ? {} : { reward: episode.rTask }),
        openedAt: episode.openedAt,
        closedAt: episode.closedAt,
        turns: remoteTurns.map((turn) => ({
          rawTurnId: turn.id,
          createdAt: turn.createdAt,
          status: turn.status,
          ...(turn.userText ? { userText: turn.userText } : {}),
          ...(turn.assistantText ? { assistantText: turn.assistantText } : {}),
          toolCallCount: turn.toolCalls.length
        }))
      },
      executionPath: path,
      ...(args.traceBoundaries ? { boundaryTrace } : {})
    };
    const persistedFile = args.outputPath ? persistJsonArtifact(args.outputPath, artifact) : undefined;
    const output = args.full
      ? artifact
      : {
          mode: "readonly-redacted-procedural-replay",
          schemaVersion: path.schemaVersion,
          episodeId: path.episodeId,
          model: llm.config.model,
          turnCount: rawTurns.length,
          stepCount: path.steps.length,
          spanCount: path.spans.length,
          pathHash: path.pathHash,
          stepContinuity: path.steps.every((step, index) => {
            const previous = path.steps[index - 1];
            return !previous || previous.postStateId === step.preStateId;
          }),
          spanContinuity: path.spans.every((span, index) => {
            const previous = path.spans[index - 1];
            return !previous || previous.postStateId === span.preStateId;
          }),
          spans: path.spans.map((span) => ({
            spanIndex: span.spanIndex,
            localGoal: span.localGoal,
            capabilityGoal: span.capabilityGoal,
            rawTurnIds: span.rawTurnIds,
            stepCount: span.cost.steps,
            toolCalls: span.cost.toolCalls,
            retryCount: span.cost.retryCount,
            recoveryCount: span.cost.recoveryCount,
            errorCount: span.cost.errorCount,
            termination: span.termination.status,
            exitCondition: span.termination.exitCondition,
            confidence: span.segmentation.confidence
          })),
          ...(args.traceBoundaries
            ? {
                boundaryTraceCalls: boundaryTrace.length,
                boundaryTraceOperations: boundaryTrace.map((call) => call.operation)
              }
            : {}),
          ...(persistedFile ? { persistedFile } : {})
        };
    process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
  } finally {
    db.close();
  }
}

function sanitizeRawTurn(turn: RawTurnRecord): RawTurnRecord {
  return {
    ...turn,
    ...(turn.userText === undefined ? {} : { userText: sanitizeText(turn.userText) }),
    ...(turn.assistantText === undefined ? {} : { assistantText: sanitizeText(turn.assistantText) }),
    reasoningSummary: undefined,
    toolCalls: turn.toolCalls.map((value) => sanitizeValue(value)),
    toolResults: turn.toolResults.map((value) => sanitizeValue(value)),
    sourceMemoryIds: [],
    usage: {},
    messagePayload: {}
  };
}

function sanitizeValue(value: unknown, depth = 0): unknown {
  if (depth > REMOTE_DEPTH_MAX) return "[TRUNCATED:depth]";
  if (typeof value === "string") return sanitizeText(value);
  if (typeof value === "number" || typeof value === "boolean" || value === null) return value;
  if (Array.isArray(value)) {
    return value.slice(0, REMOTE_COLLECTION_MAX).map((item) => sanitizeValue(item, depth + 1));
  }
  if (!value || typeof value !== "object") return value;
  const sanitized: Record<string, unknown> = {};
  for (const [key, nested] of Object.entries(value).slice(0, REMOTE_COLLECTION_MAX)) {
    if (SENSITIVE_FIELD.test(key)) {
      sanitized[key] = "[REDACTED:sensitive_field]";
    } else if (INTERNAL_REASONING_FIELD.test(key)) {
      sanitized[key] = "[REDACTED:internal_reasoning]";
    } else {
      sanitized[key] = sanitizeValue(nested, depth + 1);
    }
  }
  return sanitized;
}

function sanitizeText(value: string): string {
  return clip(redactSensitiveText(value), REMOTE_STRING_MAX);
}

function traceBoundaryCalls(
  delegate: LlmClient,
  trace: BoundaryTraceCall[]
): LlmClient {
  return {
    config: delegate.config,
    isConfigured: () => delegate.isConfigured(),
    complete: (messages, options) => delegate.complete(messages, options),
    async completeJson<T extends Record<string, unknown>>(
      messages: LlmMessage[],
      options: LlmCompletionOptions
    ): Promise<T> {
      const result = await delegate.completeJson<T>(messages, options);
      if (
        options.operation.startsWith("procedural.span_segmentation.") ||
        options.operation.startsWith("procedural.span_reconciliation.")
      ) {
        trace.push({
          operation: options.operation,
          request: sanitizeValue(llmRequestPayload(messages)),
          response: sanitizeValue(result)
        });
      }
      return result;
    },
    status: () => delegate.status()
  };
}

function llmRequestPayload(messages: readonly LlmMessage[]): unknown {
  const payload = [...messages].reverse().find(
    (message) => message.role === "user" && message.content.trimStart().startsWith("{")
  );
  if (!payload) return {};
  try {
    return JSON.parse(payload.content) as unknown;
  } catch {
    return { parseError: true };
  }
}

function persistJsonArtifact(outputPath: string, artifact: unknown): string {
  const resolved = resolve(outputPath);
  mkdirSync(dirname(resolved), { recursive: true });
  const temporary = resolve(dirname(resolved), `.${basename(resolved)}.${process.pid}.tmp`);
  writeFileSync(temporary, `${JSON.stringify(artifact, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
  renameSync(temporary, resolved);
  return resolved;
}

function parseArgs(argv: readonly string[]): ReplayArgs {
  let episodeId: string | undefined;
  let dbPath: string | undefined;
  let configPath: string | undefined;
  let outputPath: string | undefined;
  let full = false;
  let structuralOnly = false;
  let traceBoundaries = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--episode") episodeId = requiredValue(argv, ++index, "--episode");
    else if (arg === "--db") dbPath = requiredValue(argv, ++index, "--db");
    else if (arg === "--config") configPath = requiredValue(argv, ++index, "--config");
    else if (arg === "--output") outputPath = requiredValue(argv, ++index, "--output");
    else if (arg === "--full") full = true;
    else if (arg === "--structural-only") structuralOnly = true;
    else if (arg === "--trace-boundaries") traceBoundaries = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!episodeId) throw new Error("Missing required --episode <episode-id>");
  return {
    episodeId,
    full,
    structuralOnly,
    traceBoundaries,
    ...(dbPath ? { dbPath } : {}),
    ...(configPath ? { configPath } : {}),
    ...(outputPath ? { outputPath } : {})
  };
}

function requiredValue(argv: readonly string[], index: number, flag: string): string {
  const value = argv[index];
  if (!value || value.startsWith("--")) throw new Error(`Missing value for ${flag}`);
  return value;
}

await main();
