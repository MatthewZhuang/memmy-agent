import { describe, expect, it } from "vitest";
import { buildSpanTrajectory } from "../../../src/service/evolution/span-trajectory.js";

describe("span trajectory", () => {
  it("keeps one structured event per tool call without dropping indexes", () => {
    const trajectory = buildSpanTrajectory([
      {
        id: "a",
        name: "web_search",
        input: { q: "price list", apiKey: "sk-hidden" },
        output: { results: ["one", "two"], authorization: "Bearer hidden" },
        success: true
      },
      {
        id: "b",
        name: "apply_patch",
        input: { path: "src/app.ts", patch: "secret-token-value" },
        output: { path: "src/app.ts", ok: true }
      },
      {
        id: "c",
        name: "npm_test",
        input: { command: "npm test" },
        error: "AssertionError: expected 1 to equal 2",
        success: false
      },
      `invalid-${"x".repeat(80)}`
    ]);

    expect(trajectory).toHaveLength(4);
    expect(trajectory.map((event) => event.index)).toEqual([0, 1, 2, 3]);
    expect(trajectory[0]).toMatchObject({
      range: [0, 0],
      callCount: 1,
      repeated: false,
      tool: "web_search",
      action: "search",
      success: true,
      inputShape: "object:{q}",
      outputShape: "object:{results}",
      outputPreview: expect.stringContaining("results")
    });
    expect(trajectory[1]).toMatchObject({
      tool: "apply_patch",
      action: "edit",
      success: true,
      artifactSignal: "src/app.ts, src/app.ts"
    });
    expect(trajectory[2]).toMatchObject({
      tool: "npm_test",
      action: "execute",
      success: false,
      errorClass: "test_failure",
      errorPreview: expect.stringContaining("AssertionError")
    });
    expect(trajectory[3]).toMatchObject({
      index: 3,
      tool: "unknown",
      action: "other",
      success: false,
      inputShape: "raw"
    });
  });

  it("redacts sensitive values from evidence snippets", () => {
    const trajectory = buildSpanTrajectory([{
      name: "read_file",
      input: {
        path: "config.json",
        token: "private-token-value",
        password: "private-password-value"
      },
      output: {
        ok: true,
        secret: "private-secret-value",
        content: "visible content"
      }
    }]);

    const serialized = JSON.stringify(trajectory);
    expect(serialized).toContain("[redacted]");
    expect(serialized).toContain("visible content");
    expect(serialized).not.toContain("private-token-value");
    expect(serialized).not.toContain("private-password-value");
    expect(serialized).not.toContain("private-secret-value");
  });

  it("compresses consecutive repeated calls while preserving original ranges", () => {
    const trajectory = buildSpanTrajectory([
      { name: "web_search", input: { query: "one" }, output: "one" },
      { name: "web_search", input: { query: "two" }, output: "two" },
      { name: "web_search", input: { query: "three" }, output: "three" },
      { name: "read_file", input: { path: "result.txt" }, output: "done" }
    ], { maxRepeatedCalls: 8 });

    expect(trajectory).toHaveLength(2);
    expect(trajectory[0]).toMatchObject({
      index: 0,
      range: [0, 2],
      callCount: 3,
      repeated: true,
      tool: "web_search",
      action: "search"
    });
    expect(trajectory[0]?.evidence).toContain("repeat=3 calls range=0-2");
    expect(trajectory[1]).toMatchObject({
      index: 3,
      range: [3, 3],
      callCount: 1,
      repeated: false
    });
  });

  it("keeps merged repeated previews within the configured trajectory budget", () => {
    const trajectory = buildSpanTrajectory([
      { name: "web_search", output: "a".repeat(220) },
      { name: "web_search", output: "b".repeat(220) },
      { name: "web_search", output: "c".repeat(220) }
    ]);

    expect(trajectory).toHaveLength(1);
    expect(trajectory[0]?.outputPreview?.length).toBeLessThanOrEqual(220);
    expect(trajectory[0]?.evidence?.length).toBeLessThanOrEqual(420);
  });

  it("starts a new group after an error or after the generic repeat cap", () => {
    const trajectory = buildSpanTrajectory([
      { name: "web_search", input: { query: "one" }, output: "one" },
      { name: "web_search", input: { query: "two" }, output: "two" },
      { name: "web_search", input: { query: "blocked" }, error: "429 rate limit", success: false },
      { name: "web_search", input: { query: "retry" }, output: "retry" },
      { name: "web_search", input: { query: "again" }, output: "again" }
    ], { maxRepeatedCalls: 2 });

    expect(trajectory.map((event) => event.range)).toEqual([
      [0, 1],
      [2, 2],
      [3, 4]
    ]);
    expect(trajectory[1]).toMatchObject({
      success: false,
      errorClass: "rate_limit"
    });
  });
});
