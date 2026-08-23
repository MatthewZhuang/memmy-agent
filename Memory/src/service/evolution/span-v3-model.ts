import { stableHash, stableStringify } from "../../utils/id.js";

export const SPAN_V3_SCHEMA_VERSION = "span.v3" as const;
export const OBSERVED_STATE_SCHEMA_VERSION = "observed-state.v1" as const;
export const EPISODE_EXECUTION_PATH_SCHEMA_VERSION = "episode-execution-path.v1" as const;

export type StateDeltaOp =
  | "goal.set"
  | "goal.refine"
  | "goal.complete"
  | "constraint.upsert"
  | "constraint.remove"
  | "fact.upsert"
  | "fact.invalidate"
  | "artifact.upsert"
  | "artifact.verify"
  | "issue.upsert"
  | "issue.resolve"
  | "verification.set"
  | "status.set";

export type TaskStatus = "active" | "blocked" | "completed" | "failed";

export interface StateDeltaOperation {
  op: StateDeltaOp;
  subject: string;
  value?: unknown;
  status?: string;
  sourceRefs: string[];
}

export interface ObservedStateEntry {
  subject: string;
  value?: unknown;
  status?: string;
  sourceRefs: string[];
}

export interface ObservedStateV1 {
  id: string;
  schemaVersion: typeof OBSERVED_STATE_SCHEMA_VERSION;
  goal?: ObservedStateEntry;
  constraints: ObservedStateEntry[];
  facts: ObservedStateEntry[];
  artifacts: ObservedStateEntry[];
  issues: ObservedStateEntry[];
  verification: ObservedStateEntry[];
  taskStatus: TaskStatus;
  summary: string;
}

export interface SpanV3Action {
  kind: "tool_sequence" | "response_generation";
  localGoal: string;
  summary: string;
  eventRefs: string[];
  toolCallRange?: [number, number];
}

export interface SpanV3Cost {
  toolCalls: number;
  retryCount: number;
  errorCount: number;
  tokens?: number;
  latencyMs?: number;
}

export interface SpanV3Outcome {
  status: "success" | "failure" | "partial" | "unknown";
  verificationRefs: string[];
  episodeReward?: number;
}

export interface SpanV3 {
  id: string;
  schemaVersion: typeof SPAN_V3_SCHEMA_VERSION;
  episodeId: string;
  rawTurnId: string;
  spanIndex: number;
  preStateId: string;
  action: SpanV3Action;
  actionEffectDelta: StateDeltaOperation[];
  externalObservationDelta: StateDeltaOperation[];
  postStateId: string;
  cost: SpanV3Cost;
  outcome: SpanV3Outcome;
  provenance: {
    algorithmVersion: string;
    model?: string;
    sourceSnapshotHash: string;
  };
}

export interface EpisodeExecutionPathV1 {
  id: string;
  schemaVersion: typeof EPISODE_EXECUTION_PATH_SCHEMA_VERSION;
  episodeId: string;
  states: ObservedStateV1[];
  spans: SpanV3[];
  terminalReward?: number;
  sourceSnapshotHash: string;
  pathHash: string;
}

interface StateData {
  schemaVersion: typeof OBSERVED_STATE_SCHEMA_VERSION;
  goal?: ObservedStateEntry;
  constraints: ObservedStateEntry[];
  facts: ObservedStateEntry[];
  artifacts: ObservedStateEntry[];
  issues: ObservedStateEntry[];
  verification: ObservedStateEntry[];
  taskStatus: TaskStatus;
}

export function emptyObservedState(): ObservedStateV1 {
  return materializeState({
    schemaVersion: OBSERVED_STATE_SCHEMA_VERSION,
    constraints: [],
    facts: [],
    artifacts: [],
    issues: [],
    verification: [],
    taskStatus: "active"
  });
}

export function applyStateDelta(
  current: ObservedStateV1,
  operations: readonly StateDeltaOperation[]
): ObservedStateV1 {
  const goal = current.goal ? cloneEntry(current.goal) : undefined;
  const constraints = entryMap(current.constraints);
  const facts = entryMap(current.facts);
  const artifacts = entryMap(current.artifacts);
  const issues = entryMap(current.issues);
  const verification = entryMap(current.verification);
  let nextGoal = goal;
  let taskStatus = current.taskStatus;

  for (const operation of operations) {
    switch (operation.op) {
      case "goal.set":
      case "goal.refine":
        nextGoal = entryFromOperation(operation, operation.status ?? "in_progress");
        break;
      case "goal.complete":
        nextGoal = mergeEntry(nextGoal, operation, "satisfied");
        taskStatus = "completed";
        break;
      case "constraint.upsert":
        upsertEntry(constraints, operation);
        break;
      case "constraint.remove":
        constraints.delete(operation.subject);
        break;
      case "fact.upsert":
        upsertEntry(facts, operation, operation.status ?? "observed");
        break;
      case "fact.invalidate":
        upsertEntry(facts, operation, "invalidated");
        break;
      case "artifact.upsert":
        upsertEntry(artifacts, operation, operation.status ?? "observed");
        break;
      case "artifact.verify":
        upsertEntry(artifacts, operation, operation.status ?? "verified");
        break;
      case "issue.upsert":
        upsertEntry(issues, operation, operation.status ?? "open");
        break;
      case "issue.resolve":
        upsertEntry(issues, operation, "resolved");
        break;
      case "verification.set":
        upsertEntry(verification, operation, operation.status ?? stringValue(operation.value) ?? "unknown");
        break;
      case "status.set": {
        const candidate = operation.status ?? stringValue(operation.value);
        if (isTaskStatus(candidate)) taskStatus = candidate;
        break;
      }
    }
  }

  return materializeState({
    schemaVersion: OBSERVED_STATE_SCHEMA_VERSION,
    ...(nextGoal ? { goal: nextGoal } : {}),
    constraints: sortedEntries(constraints),
    facts: sortedEntries(facts),
    artifacts: sortedEntries(artifacts),
    issues: sortedEntries(issues),
    verification: sortedEntries(verification),
    taskStatus
  });
}

export function buildEpisodeExecutionPath(input: {
  episodeId: string;
  states: readonly ObservedStateV1[];
  spans: readonly SpanV3[];
  sourceSnapshotHash: string;
  terminalReward?: number;
}): EpisodeExecutionPathV1 {
  validateSpanContinuity(input.spans);
  const states = uniqueStates(input.states);
  const spans = [...input.spans];
  const stateIds = new Set(states.map((state) => state.id));
  for (const span of spans) {
    if (!stateIds.has(span.preStateId) || !stateIds.has(span.postStateId)) {
      throw new Error(`span.v3 references missing state: ${span.id}`);
    }
  }
  const hashBasis = {
    schemaVersion: EPISODE_EXECUTION_PATH_SCHEMA_VERSION,
    episodeId: input.episodeId,
    states,
    spans,
    sourceSnapshotHash: input.sourceSnapshotHash,
    ...(input.terminalReward === undefined ? {} : { terminalReward: input.terminalReward })
  };
  const pathHash = stableHash(hashBasis);
  return {
    id: `episode_path_${pathHash.slice(0, 20)}`,
    ...hashBasis,
    pathHash
  };
}

export function validateSpanContinuity(spans: readonly SpanV3[]): void {
  for (const [index, span] of spans.entries()) {
    if (span.spanIndex !== index) {
      throw new Error(`span.v3 index mismatch: expected ${index}, got ${span.spanIndex}`);
    }
    const previous = spans[index - 1];
    if (previous && previous.postStateId !== span.preStateId) {
      throw new Error(`span.v3 discontinuity between ${previous.id} and ${span.id}`);
    }
  }
}

export function observedStateSummary(state: Omit<ObservedStateV1, "id" | "summary">): string {
  const parts: string[] = [];
  if (state.goal) {
    parts.push(`goal=${displayValue(state.goal.value ?? state.goal.subject)}`);
  }
  if (state.constraints.length > 0) {
    parts.push(`constraints=${state.constraints.map(entryLabel).join(", ")}`);
  }
  const openIssues = state.issues.filter((entry) => entry.status !== "resolved");
  if (openIssues.length > 0) {
    parts.push(`open_issues=${openIssues.map((entry) => entry.subject).join(", ")}`);
  }
  if (state.artifacts.length > 0) {
    parts.push(`artifacts=${state.artifacts.map(entryLabel).join(", ")}`);
  }
  if (state.verification.length > 0) {
    parts.push(`verification=${state.verification.map(entryLabel).join(", ")}`);
  }
  parts.push(`status=${state.taskStatus}`);
  return parts.join("; ");
}

function materializeState(data: StateData): ObservedStateV1 {
  const canonical: Omit<ObservedStateV1, "id" | "summary"> = {
    schemaVersion: data.schemaVersion,
    ...(data.goal ? { goal: cloneEntry(data.goal) } : {}),
    constraints: data.constraints.map(cloneEntry),
    facts: data.facts.map(cloneEntry),
    artifacts: data.artifacts.map(cloneEntry),
    issues: data.issues.map(cloneEntry),
    verification: data.verification.map(cloneEntry),
    taskStatus: data.taskStatus
  };
  const id = `state_${stableHash(canonical).slice(0, 20)}`;
  return {
    id,
    ...canonical,
    summary: observedStateSummary(canonical)
  };
}

function entryMap(entries: readonly ObservedStateEntry[]): Map<string, ObservedStateEntry> {
  return new Map(entries.map((entry) => [entry.subject, cloneEntry(entry)]));
}

function upsertEntry(
  map: Map<string, ObservedStateEntry>,
  operation: StateDeltaOperation,
  fallbackStatus?: string
): void {
  map.set(
    operation.subject,
    mergeEntry(map.get(operation.subject), operation, fallbackStatus)
  );
}

function mergeEntry(
  previous: ObservedStateEntry | undefined,
  operation: StateDeltaOperation,
  fallbackStatus?: string
): ObservedStateEntry {
  const status = operation.status ?? fallbackStatus ?? previous?.status;
  return {
    subject: operation.subject,
    ...(operation.value === undefined
      ? previous?.value === undefined ? {} : { value: previous.value }
      : { value: operation.value }),
    ...(status ? { status } : {}),
    sourceRefs: uniqueSorted([...(previous?.sourceRefs ?? []), ...operation.sourceRefs])
  };
}

function entryFromOperation(operation: StateDeltaOperation, fallbackStatus?: string): ObservedStateEntry {
  return mergeEntry(undefined, operation, fallbackStatus);
}

function cloneEntry(entry: ObservedStateEntry): ObservedStateEntry {
  return {
    subject: entry.subject,
    ...(entry.value === undefined ? {} : { value: entry.value }),
    ...(entry.status === undefined ? {} : { status: entry.status }),
    sourceRefs: [...entry.sourceRefs]
  };
}

function sortedEntries(map: Map<string, ObservedStateEntry>): ObservedStateEntry[] {
  return [...map.values()]
    .map(cloneEntry)
    .sort((a, b) => a.subject.localeCompare(b.subject));
}

function uniqueStates(states: readonly ObservedStateV1[]): ObservedStateV1[] {
  const byId = new Map<string, ObservedStateV1>();
  for (const state of states) byId.set(state.id, state);
  return [...byId.values()];
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function entryLabel(entry: ObservedStateEntry): string {
  const value = entry.value === undefined ? entry.subject : `${entry.subject}:${displayValue(entry.value)}`;
  return entry.status ? `${value}(${entry.status})` : value;
}

function displayValue(value: unknown): string {
  if (typeof value === "string") return value;
  return stableStringify(value);
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function isTaskStatus(value: unknown): value is TaskStatus {
  return value === "active" || value === "blocked" || value === "completed" || value === "failed";
}
