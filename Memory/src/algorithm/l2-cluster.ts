import { cosineSimilarity, packRawTurnEvidence, type RawTurnLike } from "./trace-direct-skill.js";

export const DEFAULT_L2_CLUSTERING = {
  intentJoin: 0.6,
  intentGrayMin: 0.5,
  taskGray: 0.7
} as const;

export type L2ClusteringConfig = {
  intentJoin: number;
  intentGrayMin: number;
  taskGray: number;
};

export type L2ClusterItem = {
  intent: string;
  intentVec: number[] | null;
  taskVec?: number[] | null;
};

export type L2ClusterCandidate = {
  id: string;
  intentCentroid: number[] | null;
  taskCentroid?: number[] | null;
};

export type L2ClusterDecision =
  | { action: "skip"; reason: "skip_no_intent" }
  | {
      action: "create";
      reason: "create_no_intent_vec" | "create_no_candidates" | "create_intent_below_gray" | "create_gray_task_below";
      best?: { clusterId: string; intent: number; task: number | null };
    }
  | {
      action: "join";
      clusterId: string;
      reason: "join_intent" | "join_gray_task";
      intent: number;
      task: number | null;
    };

export type L2LessonKind = "path_compression" | "error_correction" | "both";
export type L2EvidencePolarity = "success" | "failure" | "unknown";

export type L2EvolveDecision =
  | { action: "skip_no_positive" }
  | { action: "create" }
  | { action: "link_only" }
  | { action: "evolve"; incrementKind: L2LessonKind };

export const SEED_META_POLICY_MD = `You extract one reusable local how-to from RAW_TURNS.

A useful L2 policy must improve future agent efficiency:
- path_compression: the next time this local intent appears, take the shorter successful path. This is NOT rewriting or compressing stored L1 rows.
- error_correction: name the failed move and the correction that avoids it.
- both: the policy shortens the success path and also blocks the failed move.

should_generate=false when the evidence cannot do either: chitchat, a preference, a world fact, one-off acceptance text, or an unresolved contradiction.

L2 is a local procedure for one intent cluster. Skill is a callable SOP with tools and steps. Do not write Skill procedure_json, retrieval blurb, or a full playbook.
L3 is the environment world model. Do not write standalone topology or declarative env facts.

Evidence is original turns. Never invent a policy to fill the schema.`;

const ACTIONABLE_RE = /(?:\b(?:run|rerun|retry|inspect|check|fix|add|use|call|open|write)\b|先|再|然后|运行|检查|重试|修复|避免|不要)/i;

export function admitL2PolicyDraft(input: {
  title: string;
  trigger: string;
  procedure: string;
  caveats?: string[];
  exclusions?: string[];
  lessonKind: L2LessonKind;
  positiveCount: number;
  negativeCount: number;
}): { ok: true } | { ok: false; reason: string } {
  if (input.positiveCount <= 0) {
    return { ok: false, reason: "admission-declined:no-positive-anchor" };
  }
  if (!input.trigger.trim() || !input.procedure.trim()) {
    return { ok: false, reason: "admission-declined:missing-procedure" };
  }
  if (input.procedure.trim().length < 20 || !ACTIONABLE_RE.test(input.procedure)) {
    return { ok: false, reason: "admission-declined:no-efficiency-gain" };
  }
  const corrections = [...(input.caveats ?? []), ...(input.exclusions ?? [])].filter((item) => item.trim());
  if (input.negativeCount > 0 && corrections.length === 0 && input.lessonKind === "path_compression") {
    return { ok: false, reason: "admission-declined:missing-error-correction" };
  }
  return { ok: true };
}

export function renderL2InductionSystem(metaPolicyMd?: string): string {
  const meta = (metaPolicyMd ?? "").trim() || SEED_META_POLICY_MD;
  return ["META-POLICY (optimizer discipline, do not copy into the policy):", meta].join("\n");
}

export type L2PackedMember = {
  l1MemoryId: string;
  polarity: L2EvidencePolarity;
  user: string;
  assistant: string;
  tools: Array<{ name: string; input: unknown; output: unknown; success: boolean }>;
};

export function l2PolicyKeyForCluster(clusterId: string): string {
  return `policy:${clusterId}`;
}

export function asFiniteNumberArray(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const numbers = value.filter((item): item is number => typeof item === "number" && Number.isFinite(item));
  return numbers.length === value.length ? numbers : null;
}

export function polarityForL2Value(value: number, minPositive: number): L2EvidencePolarity {
  if (value >= minPositive) return "success";
  if (value < 0) return "failure";
  return "unknown";
}

export function inferL2LessonKind(positiveCount: number, negativeCount: number): L2LessonKind | null {
  if (positiveCount <= 0) return null;
  return negativeCount > 0 ? "both" : "path_compression";
}

export function mergeL2LessonKind(
  existing: L2LessonKind | undefined,
  next: L2LessonKind
): L2LessonKind {
  if (!existing || existing === next) return next;
  return "both";
}

export function parseL2LessonKind(value: unknown): L2LessonKind | undefined {
  return value === "path_compression" || value === "error_correction" || value === "both"
    ? value
    : undefined;
}

export function resolveL2LessonKind(inferred: L2LessonKind, fromLlm?: L2LessonKind): L2LessonKind {
  if (inferred === "both" || fromLlm === "both") return "both";
  return inferred;
}

export function decideL2EvolveBranch(input: {
  hasPolicy: boolean;
  positiveCount: number;
  newPositiveCount: number;
  newNegativeCount: number;
}): L2EvolveDecision {
  if (input.positiveCount <= 0) return { action: "skip_no_positive" };
  if (!input.hasPolicy) return { action: "create" };
  if (input.newPositiveCount <= 0 && input.newNegativeCount <= 0) return { action: "link_only" };
  const incrementKind: L2LessonKind = input.newPositiveCount > 0 && input.newNegativeCount > 0
    ? "both"
    : input.newNegativeCount > 0
      ? "error_correction"
      : "path_compression";
  return { action: "evolve", incrementKind };
}

export function packL2ClusterRawTurns(input: {
  members: Array<{
    l1MemoryId: string;
    value: number;
    rawTurn?: RawTurnLike | null;
    fallback?: RawTurnLike | null;
  }>;
  minPositiveValue: number;
  clipChars: number;
}): L2PackedMember[] {
  return input.members.map((member) => {
    const source = member.rawTurn ?? member.fallback ?? {};
    const [packed] = packRawTurnEvidence([source], input.clipChars);
    return {
      l1MemoryId: member.l1MemoryId,
      polarity: polarityForL2Value(member.value, input.minPositiveValue),
      user: clipText(packed?.user ?? "", Math.min(200, input.clipChars)),
      assistant: clipText(packed?.assistant ?? "", Math.min(300, input.clipChars)),
      tools: packed?.tools ?? []
    };
  });
}

export function renderL2ClusterEvidence(input: {
  clusterId: string;
  seedIntent: string;
  lessonKind: L2LessonKind;
  mode: "create" | "evolve";
  incrementKind?: L2LessonKind;
  existingPolicy?: {
    title: string;
    trigger: string;
    procedure: string;
    verification: string;
    boundary: string;
    lessonKind?: string;
  };
  existingFailurePolicies?: Array<{
    title: string;
    trigger: string;
    procedure: string;
    boundary: string;
  }>;
  turns: L2PackedMember[];
  charCap: number;
}): string {
  const header = [
    `PATTERN_SIGNATURE: ${input.seedIntent || input.clusterId}`,
    `CLUSTER_ID: ${input.clusterId}`,
    `LESSON_KIND: ${input.lessonKind}`,
    `MODE: ${input.mode}`,
    ...(input.incrementKind ? [`INCREMENT_KIND: ${input.incrementKind}`] : [])
  ];
  if (input.existingFailurePolicies?.length) {
    header.push("EXISTING_FAILURE_POLICY (merge into the new positive policy, then this failure L2 will be archived):");
    for (const failure of input.existingFailurePolicies) {
      header.push(
        `title: ${failure.title}`,
        `trigger: ${failure.trigger}`,
        `procedure: ${clipText(failure.procedure, 400)}`,
        `boundary: ${failure.boundary}`
      );
    }
  }
  if (input.existingPolicy) {
    header.push(
      "EXISTING_POLICY:",
      `title: ${input.existingPolicy.title}`,
      `trigger: ${input.existingPolicy.trigger}`,
      `procedure: ${clipText(input.existingPolicy.procedure, 400)}`,
      `verification: ${input.existingPolicy.verification}`,
      `boundary: ${input.existingPolicy.boundary}`,
      `lesson_kind: ${input.existingPolicy.lessonKind ?? "-"}`
    );
  }
  header.push("RAW_TURNS (one per block, not compressed L1):");
  const prefix = header.join("\n");
  const blocks: string[] = [];
  let budget = Math.max(400, input.charCap - prefix.length - 100);
  for (const turn of input.turns) {
    const tools = turn.tools.length === 0
      ? "-"
      : turn.tools.slice(0, 3).map((tool) => {
        const output = clipText(stringifyUnknown(tool.output), 80);
        const args = clipText(stringifyUnknown(tool.input), 40);
        return `${tool.name}(${args}) -> ${output}`;
      }).join("; ");
    const block = [
      "---",
      `l1_id: ${turn.l1MemoryId}`,
      `polarity: ${turn.polarity}`,
      `user: ${turn.user}`,
      `assistant: ${turn.assistant}`,
      `tools: ${tools}`
    ].join("\n");
    if (block.length > budget) {
      blocks.push(block.slice(0, budget));
      break;
    }
    blocks.push(block);
    budget -= block.length;
  }
  return `${prefix}\n${blocks.join("\n")}`;
}

export function decideL2ClusterJoin(
  item: L2ClusterItem,
  clusters: L2ClusterCandidate[],
  config: L2ClusteringConfig = DEFAULT_L2_CLUSTERING
): L2ClusterDecision {
  if (!item.intent.trim()) {
    return { action: "skip", reason: "skip_no_intent" };
  }
  if (!item.intentVec?.length) {
    return { action: "create", reason: "create_no_intent_vec" };
  }

  let bestJoin: Extract<L2ClusterDecision, { action: "join" }> | undefined;
  let bestNear: { clusterId: string; intent: number; task: number | null } | undefined;

  for (const cluster of clusters) {
    const intent = cosineSimilarity(item.intentVec, cluster.intentCentroid);
    if (intent === null) continue;
    const task = cosineSimilarity(item.taskVec, cluster.taskCentroid);
    if (!bestNear || intent > bestNear.intent) {
      bestNear = { clusterId: cluster.id, intent, task };
    }

    let reason: "join_intent" | "join_gray_task" | undefined;
    if (intent >= config.intentJoin) {
      reason = "join_intent";
    } else if (intent >= config.intentGrayMin && task !== null && task >= config.taskGray) {
      reason = "join_gray_task";
    }
    if (!reason) continue;
    if (!bestJoin || intent > bestJoin.intent) {
      bestJoin = {
        action: "join",
        clusterId: cluster.id,
        reason,
        intent,
        task
      };
    }
  }

  if (bestJoin) return bestJoin;
  if (clusters.length === 0 || !bestNear) {
    return { action: "create", reason: "create_no_candidates" };
  }
  if (bestNear.intent < config.intentGrayMin) {
    return { action: "create", reason: "create_intent_below_gray", best: bestNear };
  }
  return { action: "create", reason: "create_gray_task_below", best: bestNear };
}

function clipText(value: string, max: number): string {
  if (!value) return "";
  if (value.length <= max) return value;
  return `${value.slice(0, Math.max(0, max - 3)).trimEnd()}...`;
}

function stringifyUnknown(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
