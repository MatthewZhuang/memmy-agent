import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";
import {
  loadArtifactTraceCases,
  stratifiedTraceSample,
  traceLengthBucket
} from "../../scripts/span-v2-artifact-traces.js";

describe("span v2 artifact trace loader", () => {
  it("loads training sqlite raw turns and test session jsonl traces from artifacts", () => {
    const root = mkdtempSync(join(tmpdir(), "span-v2-artifacts-"));
    try {
      const trainDir = join(root, "memory-v10-training-20260804");
      mkdirSync(trainDir, { recursive: true });
      const dbPath = join(trainDir, "memory.sqlite");
      const db = new Database(dbPath);
      db.exec(`
        CREATE TABLE raw_turns (
          id TEXT PRIMARY KEY,
          user_text TEXT,
          assistant_text TEXT,
          tool_calls_json TEXT NOT NULL,
          tool_results_json TEXT NOT NULL
        );
      `);
      db.prepare(`
        INSERT INTO raw_turns
        (id, user_text, assistant_text, tool_calls_json, tool_results_json)
        VALUES (?, ?, ?, ?, ?)
      `).run(
        "raw_train_1",
        "create spreadsheet",
        "done",
        JSON.stringify([
          { id: "a", name: "read_file", input: { path: "a.xlsx" } },
          { id: "b", name: "run_python" },
          { id: "c", name: "run_python" },
          { id: "d", name: "write_file" }
        ]),
        "[]"
      );
      db.close();

      const testDir = join(root, "memory-v10-test-20260804", "task_a__trial_1");
      mkdirSync(testDir, { recursive: true });
      writeFileSync(join(testDir, "session.jsonl"), [
        JSON.stringify({ role: "user", content: "make presentation" }),
        JSON.stringify({
          role: "assistant",
          content: "I will inspect files",
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "exec", arguments: "{\"cmd\":\"ls\"}" } },
            { id: "call_2", type: "function", function: { name: "exec", arguments: "{\"cmd\":\"python build.py\"}" } }
          ]
        }),
        JSON.stringify({ role: "tool", tool_call_id: "call_1", name: "exec", content: "files" }),
        JSON.stringify({ role: "tool", tool_call_id: "call_2", name: "exec", content: "ok" }),
        JSON.stringify({
          role: "assistant",
          content: "continue",
          tool_calls: [
            { id: "call_3", type: "function", function: { name: "read_file", arguments: "{\"path\":\"out.pdf\"}" } },
            { id: "call_4", type: "function", function: { name: "exec", arguments: "{\"cmd\":\"file out.pdf\"}" } }
          ]
        }),
        JSON.stringify({ role: "tool", tool_call_id: "call_3", name: "read_file", content: "pdf" }),
        JSON.stringify({ role: "tool", tool_call_id: "call_4", name: "exec", content: "PDF document" }),
        JSON.stringify({ role: "assistant", content: "done" })
      ].join("\n"));

      const traces = loadArtifactTraceCases({ artifactsRoot: root, minToolCalls: 4 });

      expect(traces).toHaveLength(2);
      expect(traces.map((trace) => trace.sourceKind).sort()).toEqual(["test_jsonl", "training_sqlite"]);
      expect(traces.find((trace) => trace.sourceKind === "training_sqlite")).toMatchObject({
        traceId: "training:raw_train_1",
        userText: "create spreadsheet",
        assistantText: "done",
        toolCalls: expect.arrayContaining([
          expect.objectContaining({ name: "read_file" })
        ])
      });
      expect(traces.find((trace) => trace.sourceKind === "test_jsonl")).toMatchObject({
        traceId: "test:task_a__trial_1",
        userText: "make presentation",
        assistantText: "done",
        toolCalls: expect.arrayContaining([
          expect.objectContaining({
            id: "call_1",
            name: "exec",
            input: { cmd: "ls" },
            output: "files",
            success: true
          })
        ])
      });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("samples across length buckets deterministically", () => {
    const traces = [
      { traceId: "s1", toolCalls: Array(4).fill({ name: "a" }) },
      { traceId: "s2", toolCalls: Array(10).fill({ name: "a" }) },
      { traceId: "m1", toolCalls: Array(20).fill({ name: "a" }) },
      { traceId: "l1", toolCalls: Array(60).fill({ name: "a" }) },
      { traceId: "x1", toolCalls: Array(100).fill({ name: "a" }) }
    ].map((trace) => ({
      ...trace,
      sourceKind: "test_jsonl" as const,
      sourcePath: trace.traceId,
      userText: trace.traceId,
      assistantText: ""
    }));

    expect(traceLengthBucket(10)).toBe("small");
    expect(traceLengthBucket(20)).toBe("medium");
    expect(traceLengthBucket(60)).toBe("large");
    expect(traceLengthBucket(100)).toBe("xlarge");
    expect(stratifiedTraceSample(traces, {
      totalLimit: 4,
      perBucketLimit: 1,
      seed: "stable"
    }).map((trace) => traceLengthBucket(trace.toolCalls.length)).sort()).toEqual([
      "large",
      "medium",
      "small",
      "xlarge"
    ]);
  });
});
