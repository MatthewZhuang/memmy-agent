#!/usr/bin/env npx tsx
/**
 * Replay SpreadsheetBench train episodes through MemoryService.
 *
 * Experience = existing train traces (SpreadsheetBench rollouts).
 * Skill assign/evolve = memmy skill_cluster_assign + skill_batch_evolve.
 */
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { MemoryDb, MemoryService } from "../src/index.js";
import { Repositories } from "../src/storage/repositories.js";
import { loadEvalConfig } from "./eval-spreadsheet-config.js";

const DATASET = process.env.SPREADSHEET_DATASET
  ?? "/root/gyh/Trace2Skill/data/spreadsheetbench_verified/spreadsheetbench_verified_400/dataset.json";
const OFFICIAL = process.env.SPREADSHEET_OFFICIAL
  ?? "/root/gyh/Trace2Skill/outputs/eval_qwen36flash/runs/creation_train_parametric/outputs/eval_official_results.json";
const LOG_DIR = process.env.SPREADSHEET_LOG_DIR
  ?? "/root/gyh/Trace2Skill/outputs/eval_qwen36flash/runs/creation_train_parametric/logs";
const OUT_DIR = process.env.SPREADSHEET_MEMMY_OUT
  ?? "/root/gyh/memmy_eval/EvoAgentBench_MemOS/outputs/spreadsheet_memmy_direct";
const USER_ID = process.env.SPREADSHEET_MEMMY_USER ?? "spreadsheet-memmy-eval";
const CONFIG_PATH = process.env.MEMMY_CONFIG ?? `${process.env.HOME}/.memmy/config.yaml`;
const START = Number(process.env.SPREADSHEET_START ?? 0);
const LIMIT = Number(process.env.SPREADSHEET_LIMIT ?? 200);

const SECTION_RE = /^## \[(\d+)\]\s+(SYSTEM|USER|ASSISTANT)\s*$/gm;
const ACTION_JSON_RE = /Action:\s*(\{[\s\S]*?\n\})/;

type OfficialResult = {
  id: string;
  success?: boolean;
  soft_score?: number;
  hard_score?: number;
  instruction_type?: string;
  error?: string;
  test_cases?: Array<{
    gt_file?: string;
    output_file?: string;
    passed?: boolean;
    message?: string;
  }>;
};

type DatasetItem = {
  id: string;
  instruction: string;
  instruction_type?: string;
  answer_position?: string;
  answer_sheet?: string;
  spreadsheet_path?: string;
};

function clip(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, limit)}…`;
}

function rTaskOf(result: OfficialResult | undefined): number {
  if (!result) return -1;
  if (result.success === true || Number(result.soft_score ?? 0) >= 0.5) return 1;
  return -1;
}

const OFFICIAL_RUBRIC = `Official SpreadsheetBench grader rubric (hard score):
1. Only cells in answer_position are graded (Sheet!A1 or Sheet!A1:B10, comma-separated).
2. Cell-Level: those exact cells. Sheet-Level: answer_position is the max editable range; every cell in that range is compared.
3. Golden and prediction are opened data_only=True; unevaluated formulas read as empty and fail.
4. Values are transformed before equality: numbers/numeric strings round to 2 decimals; datetime -> Excel serial rounded to 0; time -> HH:MM; "" and None are equal; types must match after transform.
5. Walk stops at the first mismatch. Official message is that first cell only.
6. Instance hard-passes only when every golden/output pair passes.`;

function officialFeedbackQuery(item: DatasetItem, official: OfficialResult | undefined): string {
  const success = official?.success === true || Number(official?.soft_score ?? 0) >= 0.5;
  const failures = (official?.test_cases ?? []).filter((item) => item.passed === false);
  const why = failures.length > 0
    ? failures.map((item) => `- ${item.output_file ?? item.gt_file ?? "output"}: ${item.message || "failed with no message"}`).join("\n")
    : official?.error
      ? String(official.error)
      : success
        ? "All graded cells matched the golden workbook."
        : "Official grader marked this attempt failed without a cell message.";
  return [
    "Verifier feedback for the previous attempt.",
    "The official SpreadsheetBench grader checked your previous workbook and returned the result below.",
    `Related benchmark case: ${item.id}`,
    `Instruction type: ${item.instruction_type ?? official?.instruction_type ?? ""}`,
    `Answer sheet: ${item.answer_sheet ?? ""}`,
    `Answer position: ${item.answer_position ?? ""}`,
    `Official success: ${success}`,
    `Official soft_score: ${official?.soft_score ?? 0}`,
    `Official hard_score: ${official?.hard_score ?? (success ? 1 : 0)}`,
    "",
    "Official rubric / judgment logic:",
    OFFICIAL_RUBRIC,
    "",
    success ? "Verifier feedback:" : "Why this attempt failed (official messages):",
    why,
    "",
    "This feedback does not change the official score of the previous attempt.",
    "Please briefly reflect on what you would keep and what you would improve next time."
  ].join("\n");
}

function parseLog(path: string): { instruction: string; answer: string; toolCalls: Array<{
  name: string;
  input: unknown;
  output: unknown;
  success: boolean;
}> } {
  const text = readFileSync(path, "utf8");
  const starts = [...text.matchAll(SECTION_RE)];
  const sections: Array<{ role: string; body: string }> = [];
  for (let i = 0; i < starts.length; i += 1) {
    const match = starts[i]!;
    const end = starts[i + 1]?.index ?? text.length;
    sections.push({
      role: match[2] ?? "USER",
      body: text.slice((match.index ?? 0) + match[0].length, end).trim()
    });
  }
  let instruction = "";
  for (const section of sections) {
    if (section.role === "USER" && section.body.includes("### instruction")) {
      const block = section.body.match(/### instruction\n([\s\S]*?)(?:\n### |\n---|\Z)/);
      instruction = block?.[1]?.trim() ?? "";
      break;
    }
  }
  const toolCalls: Array<{ name: string; input: unknown; output: unknown; success: boolean }> = [];
  let answer = "";
  for (let i = 0; i < sections.length; i += 1) {
    const section = sections[i]!;
    if (section.role !== "ASSISTANT") continue;
    answer = clip(section.body, 2500);
    const action = ACTION_JSON_RE.exec(section.body);
    if (!action) continue;
    let parsed: { name?: string; arguments?: unknown } = {};
    try {
      parsed = JSON.parse(action[1] ?? "{}") as { name?: string; arguments?: unknown };
    } catch {
      parsed = { name: "bash", arguments: { command: clip(action[1] ?? "", 2000) } };
    }
    let observation = "";
    const next = sections[i + 1];
    if (next?.role === "USER" && next.body.startsWith("Observation:")) {
      observation = next.body;
    }
    toolCalls.push({
      name: String(parsed.name || "bash"),
      input: parsed.arguments ?? parsed,
      output: clip(observation, 3000),
      success: !/^Observation:[\s\S]{0,200}error/i.test(observation)
    });
  }
  return { instruction, answer, toolCalls };
}

function findLog(id: string): string | undefined {
  const succeed = join(LOG_DIR, `cli_skill_preloaded_agent_${id}_SUCCEED.md`);
  const failed = join(LOG_DIR, `cli_skill_preloaded_agent_${id}_FAILED.md`);
  if (existsSync(succeed)) return succeed;
  if (existsSync(failed)) return failed;
  return undefined;
}

function countRows(db: MemoryDb, sql: string): number {
  return (db.db.prepare(sql).get() as { count: number }).count;
}

async function drainSkillJobs(service: MemoryService, db: MemoryDb): Promise<void> {
  for (let round = 0; round < 24; round += 1) {
    const queued = countRows(
      db,
      `SELECT COUNT(*) AS count FROM evolution_jobs
       WHERE status = 'queued' AND job_type IN ('skill_cluster_assign', 'skill_batch_evolve')`
    );
    if (queued === 0) return;
    await service.runWorkerOnce(20);
  }
}

async function main(): Promise<void> {
  const dataset = JSON.parse(readFileSync(DATASET, "utf8")) as DatasetItem[];
  const official = JSON.parse(readFileSync(OFFICIAL, "utf8")) as { results: OfficialResult[] };
  const officialById = new Map(official.results.map((item) => [String(item.id), item]));
  const train = dataset.slice(START, START + LIMIT);
  const sqlitePath = join(OUT_DIR, "memory.sqlite");
  mkdirSync(OUT_DIR, { recursive: true });
  mkdirSync(join(OUT_DIR, "skills"), { recursive: true });

  const config = loadEvalConfig(CONFIG_PATH, sqlitePath);
  const db = new MemoryDb({ path: sqlitePath });
  const service = new MemoryService({ db, mode: "local", config });
  const repos = new Repositories(db.db);

  const events: Array<Record<string, unknown>> = [];
  let created = 0;
  let joined = 0;

  for (const [index, item] of train.entries()) {
    const id = String(item.id);
    const officialRow = officialById.get(id);
    const logPath = findLog(id);
    const parsed = logPath
      ? parseLog(logPath)
      : { instruction: item.instruction, answer: "", toolCalls: [] };
    const query = [
      parsed.instruction || item.instruction,
      `instruction_type: ${item.instruction_type ?? ""}`,
      `answer_position: ${item.answer_position ?? ""}`,
      `answer_sheet: ${item.answer_sheet ?? ""}`,
      "Save the workbook to output.xlsx."
    ].filter(Boolean).join("\n");
    const rTask = rTaskOf(officialRow);
    const clustersBefore = countRows(db, "SELECT COUNT(*) AS count FROM skill_clusters");
    const skillsBefore = countRows(db, "SELECT COUNT(*) AS count FROM memories WHERE memory_layer = 'Skill'");
    const session = service.openSession({
      namespace: { source: "codex", profileId: "spreadsheetbench", userId: USER_ID }
    });
    const complete = service.completeTurn(`ssb-${id}-${index}`, {
      sessionId: session.sessionId,
      query,
      answer: parsed.answer || "Completed the spreadsheet edit.",
      toolCalls: [
        ...(parsed.toolCalls.length > 0
          ? parsed.toolCalls
          : [{
            name: "bash",
            input: { command: "python -c \"import openpyxl; openpyxl.load_workbook('input.xlsx')\"" },
            output: { ok: true },
            success: rTask > 0
          }]),
        {
          name: "spreadsheetbench_official_grader",
          input: {
            instruction_type: item.instruction_type ?? officialRow?.instruction_type ?? "",
            answer_position: item.answer_position ?? "",
            rubric: OFFICIAL_RUBRIC
          },
          output: {
            success: rTask > 0,
            soft_score: officialRow?.soft_score ?? 0,
            hard_score: officialRow?.hard_score ?? (rTask > 0 ? 1 : 0),
            test_cases: officialRow?.test_cases ?? [],
            error: officialRow?.error ?? null
          },
          success: rTask > 0
        }
      ]
    });
    const feedback = service.completeTurn(`ssb-${id}-${index}-fb`, {
      sessionId: session.sessionId,
      query: officialFeedbackQuery(item, officialRow),
      answer: rTask > 0
        ? "Acknowledged. The official grader passed the graded cells."
        : "Acknowledged. I will treat the official cell mismatches and value-transform rules as failure constraints next time.",
      toolCalls: []
    });
    const at = new Date().toISOString();
    db.db.prepare(`DELETE FROM evolution_jobs WHERE status = 'queued' AND job_type NOT IN ('skill_cluster_assign', 'skill_batch_evolve')`)
      .run();
    const episodeId = complete.episodeId;
    db.db.prepare(`UPDATE episodes SET status = 'closed', closed_at = ?, updated_at = ? WHERE id = ?`)
      .run(at, at, episodeId);
    if (feedback.episodeId && feedback.episodeId !== episodeId) {
      db.db.prepare(`UPDATE episodes SET status = 'closed', closed_at = ?, updated_at = ? WHERE id = ?`)
        .run(at, at, feedback.episodeId);
    }
    repos.runtime.updateEpisodeReward(complete.episodeId, {
      rTask,
      rewardDetail: { source: "spreadsheetbench.official", instanceId: id }
    }, at);
    repos.runtime.enqueueJob({
      id: `job_assign_${id}_${index}`,
      jobType: "skill_cluster_assign",
      status: "queued",
      userId: USER_ID,
      sessionId: session.sessionId,
      episodeId: complete.episodeId,
      payload: { reason: "spreadsheetbench.replay", instanceId: id },
      attempts: 0,
      maxAttempts: 3,
      createdAt: at,
      updatedAt: at
    });
    try {
      await drainSkillJobs(service, db);
    } catch (error) {
      events.push({
        id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
    const clustersAfter = countRows(db, "SELECT COUNT(*) AS count FROM skill_clusters");
    const skillsAfter = countRows(db, "SELECT COUNT(*) AS count FROM memories WHERE memory_layer = 'Skill'");
    if (clustersAfter > clustersBefore) created += 1;
    else joined += 1;
    const cluster = db.db.prepare(
      `SELECT c.id, c.skill_memory_id, c.member_count, c.tools_json, c.artifacts_json
       FROM skill_cluster_members m
       JOIN skill_clusters c ON c.id = m.cluster_id
       WHERE m.episode_id = ?
       ORDER BY m.assigned_at DESC
       LIMIT 1`
    ).get(complete.episodeId) as {
      id?: string;
      skill_memory_id?: string | null;
      member_count?: number;
      tools_json?: string;
      artifacts_json?: string;
    } | undefined;
    events.push({
      index,
      id,
      rTask,
      episodeId: complete.episodeId,
      assign: clustersAfter > clustersBefore ? "create" : "join",
      clusterId: cluster?.id,
      memberCount: cluster?.member_count,
      skillMemoryId: cluster?.skill_memory_id ?? null,
      skillDelta: skillsAfter - skillsBefore,
      tools: cluster?.tools_json,
      artifacts: cluster?.artifacts_json
    });
    console.log(JSON.stringify(events.at(-1)));
  }

  const clusters = db.db.prepare(
    `SELECT id, member_count, skill_memory_id, tools_json, artifacts_json, meta_skill_md
     FROM skill_clusters`
  ).all() as Array<{
    id: string;
    member_count: number;
    skill_memory_id: string | null;
    tools_json: string;
    artifacts_json: string;
    meta_skill_md: string;
  }>;
  const skills = db.db.prepare(
    `SELECT id, memory_value, properties_json, updated_at
     FROM memories
     WHERE memory_layer = 'Skill'`
  ).all() as Array<{ id: string; memory_value: string; properties_json: string; updated_at: string }>;
  for (const [index, skill] of skills.entries()) {
    writeFileSync(join(OUT_DIR, "skills", `skill-${index + 1}-${skill.id}.md`), skill.memory_value, "utf8");
  }
  const rejected = events.filter((item) => item.rTask === 1 && !item.skillMemoryId && item.assign === "create").length
    + events.filter((item) => item.skillDelta === 0 && item.assign === "join" && item.rTask === 1).length;
  const summary = {
    protocol: "SpreadsheetBench 0:200 replayed into MemoryService skill_cluster_assign/evolve",
    userId: USER_ID,
    sqlitePath,
    train: train.length,
    clusters: clusters.map((cluster) => ({
      id: cluster.id,
      members: cluster.member_count,
      skillMemoryId: cluster.skill_memory_id,
      tools: JSON.parse(cluster.tools_json || "[]"),
      artifacts: JSON.parse(cluster.artifacts_json || "[]")
    })),
    skills: skills.length,
    created,
    joined,
    rejectedApprox: rejected,
    events
  };
  writeFileSync(join(OUT_DIR, "train_summary.json"), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify({
    clusters: clusters.length,
    skills: skills.length,
    created,
    joined,
    sqlitePath
  }, null, 2));
  db.close();
}

void main().catch((error) => {
  console.error(error);
  process.exit(1);
});
