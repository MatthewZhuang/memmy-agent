import { describe, expect, it } from "vitest";
import {
  buildCompressedEventBudgetSegments,
  buildFixedToolCallSegments,
  buildHybridBudgetSegments,
  eventsForToolCallRange,
  trajectoryText
} from "../../scripts/span-v2-segmentation.js";
import type { SpanTrajectoryEvent } from "../../src/service/evolution/span-trajectory.js";

function event(input: {
  index: number;
  range: [number, number];
  callCount: number;
  tool: string;
}): SpanTrajectoryEvent {
  return {
    index: input.index,
    range: input.range,
    callCount: input.callCount,
    repeated: input.callCount > 1,
    tool: input.tool,
    action: "search",
    success: true
  };
}

describe("span v2 split ablation segmentation helpers", () => {
  it("builds fixed windows over original tool-call indexes", () => {
    expect(buildFixedToolCallSegments(113, 50, 10)).toEqual([
      { start: 0, end: 49, reason: "fixed_count" },
      { start: 40, end: 89, reason: "fixed_count" },
      { start: 80, end: 112, reason: "fixed_count" }
    ]);
  });

  it("selects compressed trajectory events by overlap with raw tool-call range", () => {
    const trajectory = [
      event({ index: 0, range: [0, 7], callCount: 8, tool: "web_search" }),
      event({ index: 8, range: [8, 8], callCount: 1, tool: "read_file" }),
      event({ index: 9, range: [9, 16], callCount: 8, tool: "web_search" })
    ];

    expect(eventsForToolCallRange(trajectory, { start: 6, end: 10 }))
      .toEqual([trajectory[0], trajectory[1], trajectory[2]]);
    expect(trajectoryText(eventsForToolCallRange(
      trajectory,
      { start: 6, end: 10 }
    ))).toContain("range=9-16");
  });

  it("builds budget windows over compressed events while returning raw tool-call ranges", () => {
    const trajectory = [
      event({ index: 0, range: [0, 7], callCount: 8, tool: "web_search" }),
      event({ index: 8, range: [8, 10], callCount: 3, tool: "web_fetch" }),
      event({ index: 11, range: [11, 18], callCount: 8, tool: "web_search" }),
      event({ index: 19, range: [19, 22], callCount: 4, tool: "read_file" }),
      event({ index: 23, range: [23, 30], callCount: 8, tool: "web_search" })
    ];

    expect(buildCompressedEventBudgetSegments(trajectory, {
      maxEvents: 2,
      overlapEvents: 1
    })).toEqual([
      { start: 0, end: 10, reason: "compressed_event_budget" },
      { start: 8, end: 18, reason: "compressed_event_budget" },
      { start: 11, end: 22, reason: "compressed_event_budget" },
      { start: 19, end: 30, reason: "compressed_event_budget" }
    ]);
  });

  it("builds hybrid windows constrained by raw calls, event count, and text budget", () => {
    const trajectory = [
      event({ index: 0, range: [0, 7], callCount: 8, tool: "web_search" }),
      event({ index: 8, range: [8, 10], callCount: 3, tool: "web_fetch" }),
      event({ index: 11, range: [11, 18], callCount: 8, tool: "web_search" }),
      event({ index: 19, range: [19, 22], callCount: 4, tool: "read_file" }),
      event({ index: 23, range: [23, 30], callCount: 8, tool: "web_search" })
    ];

    expect(buildHybridBudgetSegments(trajectory, {
      maxRawToolCalls: 12,
      maxEvents: 3,
      maxTextChars: 1_000,
      overlapEvents: 1,
      eventText: (item) => item.tool
    })).toEqual([
      { start: 0, end: 10, reason: "hybrid_budget" },
      { start: 8, end: 18, reason: "hybrid_budget" },
      { start: 11, end: 22, reason: "hybrid_budget" },
      { start: 19, end: 30, reason: "hybrid_budget" }
    ]);

    expect(buildHybridBudgetSegments(trajectory, {
      maxRawToolCalls: 100,
      maxEvents: 100,
      maxTextChars: 20,
      overlapEvents: 0,
      eventText: (item) => item.tool
    })).toEqual([
      { start: 0, end: 10, reason: "hybrid_budget" },
      { start: 11, end: 22, reason: "hybrid_budget" },
      { start: 23, end: 30, reason: "hybrid_budget" }
    ]);
  });

  it("builds hybrid windows with the default single-event budget text", () => {
    const trajectory = [
      event({ index: 0, range: [0, 7], callCount: 8, tool: "web_search" }),
      event({ index: 8, range: [8, 10], callCount: 3, tool: "web_fetch" }),
      event({ index: 11, range: [11, 18], callCount: 8, tool: "web_search" })
    ];

    expect(buildHybridBudgetSegments(trajectory, {
      maxRawToolCalls: 20,
      maxEvents: 2,
      maxTextChars: 10_000,
      overlapEvents: 0
    })).toEqual([
      { start: 0, end: 10, reason: "hybrid_budget" },
      { start: 11, end: 18, reason: "hybrid_budget" }
    ]);
  });
});
