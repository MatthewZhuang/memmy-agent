import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import Database from "better-sqlite3";
import {
  createLlmClient,
  loadMemmyConfig,
  resolveEvolutionConfig,
  type EpisodeExecutionPathV1
} from "../src/index.js";
import {
  EpisodeSpanV3Reconstructor,
  buildEpisodeActionEvents,
  proposeSpanBoundaryCandidates
} from "../src/service/evolution/episode-span-v3-reconstructor.js";
import { Repositories, type RawTurnRecord } from "../src/storage/repositories.js";
import { redactSensitiveText } from "../src/utils/sensitive-data.js";
import { clip } from "../src/utils/text.js";

const REMOTE_REPLAY_STRING_MAX = 2_000;
const REMOTE_REPLAY_COLLECTION_MAX = 100;
const REMOTE_REPLAY_DEPTH_MAX = 8;
const ARTIFACT_STRING_MAX = 20_000;
const ARTIFACT_COLLECTION_MAX = 500;
const ARTIFACT_DEPTH_MAX = 12;
const SENSITIVE_FIELD = /api.?key|token|secret|password|authorization|credential|cookie/i;
const INTERNAL_REASONING_FIELD = /thinking|reasoning|assistant.?text.?before|chain.?of.?thought/i;

interface ReplayArgs {
  episodeId: string;
  dbPath?: string;
  configPath?: string;
  outputPath?: string;
  full: boolean;
  structuralOnly: boolean;
}

interface ReplayArtifactTurn {
  rawTurnId: string;
  turnId: string;
  createdAt: string;
  status: string;
  userText?: string;
  assistantText?: string;
  toolCalls: unknown[];
  toolResults: unknown[];
}

interface ReplayArtifact {
  schemaVersion: "span-v3-replay-artifact.v1";
  generatedAt: string;
  privacy: {
    databaseReadOnly: true;
    remoteInput: "recursively-redacted-and-clipped";
    persistedInput: "recursively-redacted";
    rawCredentialsPersisted: false;
    internalReasoningPersisted: false;
  };
  replay: {
    model?: string;
    provider: string;
    endpoint?: string;
    candidateBoundaryCount: number;
  };
  episode: {
    id: string;
    status: string;
    reward?: number;
    openedAt: string;
    closedAt?: string | null;
    turns: ReplayArtifactTurn[];
  };
  structuralAudit: unknown[];
  executionPath: EpisodeExecutionPathV1;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const loaded = loadMemmyConfig(args.configPath);
  const dbPath = args.dbPath ?? loaded.config.storage.sqlitePath;
  if (!dbPath) {
    throw new Error("No SQLite database path. Pass --db or configure storage.sqlitePath.");
  }
  const db = new Database(dbPath, { readonly: true, fileMustExist: true });
  try {
    const repos = new Repositories(db);
    const episode = repos.runtime.getEpisode(args.episodeId);
    if (!episode) throw new Error(`Episode not found: ${args.episodeId}`);
    const rawTurns = repos.runtime.listRawTurnsByEpisode(args.episodeId, 1_000)
      .filter((turn) => !turn.deletedAt);
    if (rawTurns.length === 0) {
      throw new Error(`Episode has no readable raw turns: ${args.episodeId}`);
    }
    const structuralAudit = rawTurns.map((turn, turnIndex) => {
      const events = buildEpisodeActionEvents(turn, turnIndex);
      const candidates = proposeSpanBoundaryCandidates(events);
      return {
        rawTurnId: turn.id,
        rawToolCallCount: turn.toolCalls.length,
        normalizedEventCount: events.length,
        actionFamilies: events.map((event) => event.family),
        outcomeSequence: events.map((event) => event.success ? "success" : "failure"),
        candidates: candidates.map((candidate) => ({
          afterEventId: candidate.afterEventId,
          beforeEventId: candidate.beforeEventId,
          signals: candidate.signals
        }))
      };
    });
    const candidateCount = structuralAudit.reduce(
      (count, turn) => count + turn.candidates.length,
      0
    );
    if (args.structuralOnly) {
      process.stdout.write(`${JSON.stringify({
        mode: "readonly-structural-audit",
        episodeId: episode.id,
        rawTurnCount: rawTurns.length,
        rawToolCallCount: rawTurns.reduce((count, turn) => count + turn.toolCalls.length, 0),
        normalizedEventCount: structuralAudit.reduce(
          (count, turn) => count + turn.normalizedEventCount,
          0
        ),
        candidateBoundaryCount: candidateCount,
        turns: structuralAudit
      }, null, 2)}\n`);
      return;
    }
    const llm = createLlmClient(resolveEvolutionConfig(loaded.config), {
      modelRole: "memory_evolution"
    });
    if (!llm.isConfigured()) {
      throw new Error("The configured evolution/summary LLM is unavailable.");
    }
    const remoteReplayTurns = rawTurns.map(sanitizeRawTurnForRemoteReplay);
    const path = await new EpisodeSpanV3Reconstructor({ llm }).reconstruct({
      episodeId: episode.id,
      rawTurns: remoteReplayTurns,
      ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask })
    });
    const artifact: ReplayArtifact = {
      schemaVersion: "span-v3-replay-artifact.v1",
      generatedAt: new Date().toISOString(),
      privacy: {
        databaseReadOnly: true,
        remoteInput: "recursively-redacted-and-clipped",
        persistedInput: "recursively-redacted",
        rawCredentialsPersisted: false,
        internalReasoningPersisted: false
      },
      replay: {
        model: llm.config.model,
        provider: llm.config.provider,
        endpoint: llm.config.endpoint,
        candidateBoundaryCount: candidateCount
      },
      episode: {
        id: episode.id,
        status: episode.status,
        ...(episode.rTask === undefined ? {} : { reward: episode.rTask }),
        openedAt: episode.openedAt,
        closedAt: episode.closedAt,
        turns: rawTurns.map(sanitizeRawTurnForArtifact)
      },
      structuralAudit,
      executionPath: path
    };
    const persistedPaths = args.outputPath
      ? persistReplayArtifact(args.outputPath, artifact)
      : undefined;
    const output = args.full
      ? artifact
      : {
          ...replaySummary(path, rawTurns.length, candidateCount, llm.config.model),
          ...(persistedPaths ? { persistedFiles: persistedPaths } : {})
        };
    process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
  } finally {
    db.close();
  }
}

function replaySummary(
  path: EpisodeExecutionPathV1,
  rawTurnCount: number,
  candidateBoundaryCount: number,
  model: string | undefined
): Record<string, unknown> {
  const sameTurnAdjacencies = path.spans.reduce((count, span, index) => {
    const previous = path.spans[index - 1];
    return count + (previous?.rawTurnId === span.rawTurnId ? 1 : 0);
  }, 0);
  return {
    mode: "readonly-redacted-shadow-replay",
    redactedRemoteInput: true,
    schemaVersion: path.schemaVersion,
    episodeId: path.episodeId,
    model,
    rawTurnCount,
    candidateBoundaryCount,
    materializedSameTurnBoundaries: sameTurnAdjacencies,
    stateCount: path.states.length,
    spanCount: path.spans.length,
    pathHash: path.pathHash,
    continuity: path.spans.every((span, index) => {
      const previous = path.spans[index - 1];
      return !previous || previous.postStateId === span.preStateId;
    }),
    spans: path.spans.map((span) => ({
      spanIndex: span.spanIndex,
      rawTurnId: span.rawTurnId,
      kind: span.action.kind,
      localGoal: span.action.localGoal,
      summary: span.action.summary,
      toolCallRange: span.action.toolCallRange,
      toolCalls: span.cost.toolCalls,
      actionDeltaOps: span.actionEffectDelta.map((operation) => operation.op),
      externalObservationOps: span.externalObservationDelta.map((operation) => operation.op),
      outcome: span.outcome.status,
      preStateId: span.preStateId,
      postStateId: span.postStateId
    })),
    terminalState: path.states.at(-1)
      ? {
          id: path.states.at(-1)!.id,
          taskStatus: path.states.at(-1)!.taskStatus,
          constraintCount: path.states.at(-1)!.constraints.length,
          factCount: path.states.at(-1)!.facts.length,
          artifactCount: path.states.at(-1)!.artifacts.length,
          openIssueCount: path.states.at(-1)!.issues.filter((issue) => issue.status !== "resolved").length,
          verificationCount: path.states.at(-1)!.verification.length
        }
      : undefined
  };
}

function sanitizeRawTurnForRemoteReplay(turn: RawTurnRecord): RawTurnRecord {
  return {
    ...turn,
    ...(turn.userText === undefined ? {} : { userText: sanitizeText(turn.userText) }),
    ...(turn.assistantText === undefined ? {} : { assistantText: sanitizeText(turn.assistantText) }),
    reasoningSummary: undefined,
    toolCalls: turn.toolCalls.map((value) => sanitizeRemoteValue(value)),
    toolResults: turn.toolResults.map((value) => sanitizeRemoteValue(value)),
    sourceMemoryIds: [],
    usage: {},
    messagePayload: {}
  };
}

function sanitizeRawTurnForArtifact(turn: RawTurnRecord): ReplayArtifactTurn {
  return {
    rawTurnId: turn.id,
    turnId: turn.turnId,
    createdAt: turn.createdAt,
    status: turn.status,
    ...(turn.userText === undefined
      ? {}
      : { userText: sanitizeText(turn.userText, ARTIFACT_STRING_MAX) }),
    ...(turn.assistantText === undefined
      ? {}
      : { assistantText: sanitizeText(turn.assistantText, ARTIFACT_STRING_MAX) }),
    toolCalls: turn.toolCalls.map((value) => sanitizeArtifactValue(value)),
    toolResults: turn.toolResults.map((value) => sanitizeArtifactValue(value))
  };
}

function sanitizeRemoteValue(value: unknown, depth = 0): unknown {
  return sanitizeValue(value, {
    stringMax: REMOTE_REPLAY_STRING_MAX,
    collectionMax: REMOTE_REPLAY_COLLECTION_MAX,
    depthMax: REMOTE_REPLAY_DEPTH_MAX
  }, depth);
}

function sanitizeArtifactValue(value: unknown, depth = 0): unknown {
  return sanitizeValue(value, {
    stringMax: ARTIFACT_STRING_MAX,
    collectionMax: ARTIFACT_COLLECTION_MAX,
    depthMax: ARTIFACT_DEPTH_MAX
  }, depth);
}

function sanitizeValue(
  value: unknown,
  limits: { stringMax: number; collectionMax: number; depthMax: number },
  depth: number
): unknown {
  if (depth >= limits.depthMax) return "[truncated]";
  if (typeof value === "string") return sanitizeText(value, limits.stringMax);
  if (typeof value === "number" || typeof value === "boolean" || value === null) return value;
  if (Array.isArray(value)) {
    return value
      .slice(0, limits.collectionMax)
      .map((item) => sanitizeValue(item, limits, depth + 1));
  }
  if (typeof value !== "object" || value === undefined) return undefined;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !INTERNAL_REASONING_FIELD.test(key))
      .slice(0, limits.collectionMax)
      .map(([key, item]) => [
        key,
        SENSITIVE_FIELD.test(key) ? "[redacted]" : sanitizeValue(item, limits, depth + 1)
      ])
  );
}

function sanitizeText(value: string, maxChars = REMOTE_REPLAY_STRING_MAX): string {
  return clip(redactSensitiveText(value), maxChars);
}

function persistReplayArtifact(
  requestedJsonPath: string,
  artifact: ReplayArtifact
): { json: string; markdown: string } {
  const jsonPath = resolve(requestedJsonPath);
  if (!jsonPath.endsWith(".json")) {
    throw new Error("--output must point to a .json file");
  }
  const markdownPath = jsonPath.slice(0, -5) + ".md";
  mkdirSync(dirname(jsonPath), { recursive: true });
  atomicWrite(jsonPath, `${JSON.stringify(artifact, null, 2)}\n`);
  atomicWrite(markdownPath, renderReplayMarkdown(artifact));
  return { json: jsonPath, markdown: markdownPath };
}

function atomicWrite(path: string, content: string): void {
  const temporaryPath = `${path}.tmp`;
  writeFileSync(temporaryPath, content, "utf8");
  renameSync(temporaryPath, path);
}

function renderReplayMarkdown(artifact: ReplayArtifact): string {
  const stateById = new Map(
    artifact.executionPath.states.map((state) => [state.id, state])
  );
  const lines = [
    `# Span v3 Replay — ${artifact.episode.id}`,
    "",
    "> Local-only experiment artifact. Episode content is recursively redacted; credentials and internal reasoning are not persisted.",
    "",
    `- Generated: ${artifact.generatedAt}`,
    `- Model: ${artifact.replay.model ?? "unknown"}`,
    `- Provider: ${artifact.replay.provider}`,
    `- Reward: ${artifact.episode.reward ?? "unknown"}`,
    `- Turns: ${artifact.episode.turns.length}`,
    `- Candidate boundaries: ${artifact.replay.candidateBoundaryCount}`,
    `- Materialized spans: ${artifact.executionPath.spans.length}`,
    `- Path hash: \`${artifact.executionPath.pathHash}\``,
    "",
    "## Complete Episode",
    ""
  ];
  for (const [turnIndex, turn] of artifact.episode.turns.entries()) {
    lines.push(
      `### Turn ${turnIndex + 1} — \`${turn.rawTurnId}\``,
      "",
      "**User**",
      "",
      fenced(turn.userText ?? "(empty)"),
      "",
      `**Tools (${turn.toolCalls.length})**`,
      "",
      turn.toolCalls.length > 0
        ? turn.toolCalls.map((call, index) => `${index + 1}. \`${toolName(call)}\``).join("\n")
        : "None",
      "",
      "<details><summary><strong>Assistant</strong></summary>",
      "",
      fenced(turn.assistantText ?? "(empty)"),
      "",
      "</details>",
      ""
    );
  }
  lines.push("## Episode Execution Path", "");
  for (const span of artifact.executionPath.spans) {
    const preState = stateById.get(span.preStateId);
    const postState = stateById.get(span.postStateId);
    lines.push(
      `### Span ${span.spanIndex} — ${span.action.localGoal}`,
      "",
      `- ID: \`${span.id}\``,
      `- Raw turn: \`${span.rawTurnId}\``,
      `- Action kind: \`${span.action.kind}\``,
      `- Tool calls: ${span.cost.toolCalls}`,
      `- Outcome: \`${span.outcome.status}\``,
      `- Pre-state: \`${span.preStateId}\` — ${preState?.summary ?? "missing"}`,
      `- Post-state: \`${span.postStateId}\` — ${postState?.summary ?? "missing"}`,
      "",
      "**Action summary**",
      "",
      span.action.summary,
      "",
      "**Action effect delta**",
      "",
      fenced(JSON.stringify(span.actionEffectDelta, null, 2), "json"),
      "",
      "**External observation delta**",
      "",
      fenced(JSON.stringify(span.externalObservationDelta, null, 2), "json"),
      ""
    );
  }
  lines.push(
    "## Final State",
    "",
    fenced(JSON.stringify(artifact.executionPath.states.at(-1), null, 2), "json"),
    ""
  );
  return `${lines.join("\n")}\n`;
}

function toolName(value: unknown): string {
  if (typeof value !== "object" || value === null || !("name" in value)) return "unknown";
  const name = (value as { name?: unknown }).name;
  return typeof name === "string" && name.trim() ? name.trim() : "unknown";
}

function fenced(value: string, language = "text"): string {
  return `\`\`\`\`\`${language}\n${value}\n\`\`\`\`\``;
}

function parseArgs(argv: readonly string[]): ReplayArgs {
  let episodeId: string | undefined;
  let dbPath: string | undefined;
  let configPath: string | undefined;
  let outputPath: string | undefined;
  let full = false;
  let structuralOnly = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    if (arg === "--full") {
      full = true;
      continue;
    }
    if (arg === "--structural-only") {
      structuralOnly = true;
      continue;
    }
    const [name, inlineValue] = arg.split("=", 2);
    if (name !== "--episode" && name !== "--db" && name !== "--config" && name !== "--output") {
      throw new Error(`Unknown argument: ${arg}`);
    }
    const value = inlineValue ?? argv[++index];
    if (!value?.trim()) throw new Error(`Missing value for ${name}`);
    if (name === "--episode") episodeId = value;
    if (name === "--db") dbPath = value;
    if (name === "--config") configPath = value;
    if (name === "--output") outputPath = value;
  }
  if (!episodeId) {
    throw new Error(
      "Usage: npm run span:v3:replay -w @memmy/memory -- --episode <id> [--db <path>] [--config <path>] [--output <result.json>] [--structural-only] [--full]"
    );
  }
  return {
    episodeId,
    ...(dbPath ? { dbPath } : {}),
    ...(configPath ? { configPath } : {}),
    ...(outputPath ? { outputPath } : {}),
    full,
    structuralOnly
  };
}

await main();
