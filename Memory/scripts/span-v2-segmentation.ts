import type { SpanTrajectoryEvent } from "../src/service/evolution/span-trajectory.js";

export interface Segment {
  start: number;
  end: number;
  reason: string;
}

export interface CompressedEventBudgetOptions {
  maxEvents: number;
  overlapEvents: number;
  reason?: string;
}

export interface HybridBudgetOptions {
  maxRawToolCalls: number;
  maxEvents: number;
  maxTextChars: number;
  overlapEvents: number;
  eventText?: (event: SpanTrajectoryEvent) => string;
  reason?: string;
}

export function buildFixedToolCallSegments(
  toolCallCount: number,
  windowSize: number,
  overlap: number,
  reason = "fixed_count"
): Segment[] {
  if (toolCallCount <= 0) return [];
  const segments: Segment[] = [];
  const step = Math.max(1, windowSize - overlap);
  for (let start = 0; start < toolCallCount; start += step) {
    const end = Math.min(toolCallCount - 1, start + windowSize - 1);
    segments.push({ start, end, reason });
    if (end >= toolCallCount - 1) break;
  }
  return segments;
}

export function eventsForToolCallRange(
  trajectory: readonly SpanTrajectoryEvent[],
  segment: Pick<Segment, "start" | "end">
): SpanTrajectoryEvent[] {
  return trajectory.filter((event) =>
    event.range[1] >= segment.start && event.range[0] <= segment.end
  );
}

export function buildCompressedEventBudgetSegments(
  trajectory: readonly SpanTrajectoryEvent[],
  options: CompressedEventBudgetOptions
): Segment[] {
  return buildEventWindowSegments(trajectory, {
    maxEvents: options.maxEvents,
    overlapEvents: options.overlapEvents,
    reason: options.reason ?? "compressed_event_budget"
  });
}

export function buildHybridBudgetSegments(
  trajectory: readonly SpanTrajectoryEvent[],
  options: HybridBudgetOptions
): Segment[] {
  return buildEventWindowSegments(trajectory, {
    maxRawToolCalls: options.maxRawToolCalls,
    maxEvents: options.maxEvents,
    maxTextChars: options.maxTextChars,
    overlapEvents: options.overlapEvents,
    eventText: options.eventText ?? trajectoryEventBudgetText,
    reason: options.reason ?? "hybrid_budget"
  });
}

function buildEventWindowSegments(
  trajectory: readonly SpanTrajectoryEvent[],
  options: {
    maxRawToolCalls?: number;
    maxEvents: number;
    maxTextChars?: number;
    overlapEvents: number;
    eventText?: (event: SpanTrajectoryEvent) => string;
    reason: string;
  }
): Segment[] {
  if (trajectory.length === 0) return [];
  const maxEvents = Math.max(1, Math.floor(options.maxEvents));
  const overlapEvents = Math.max(0, Math.floor(options.overlapEvents));
  const segments: Segment[] = [];
  let startEventIndex = 0;
  while (startEventIndex < trajectory.length) {
    let endEventIndex = startEventIndex;
    let textChars = 0;
    for (let index = startEventIndex; index < trajectory.length; index += 1) {
      const first = trajectory[startEventIndex]!;
      const current = trajectory[index]!;
      const eventCount = index - startEventIndex + 1;
      const rawCallCount = current.range[1] - first.range[0] + 1;
      const nextTextChars = textChars + (
        options.eventText ? options.eventText(current).length : 0
      );
      const exceedsBudget =
        eventCount > maxEvents ||
        (options.maxRawToolCalls !== undefined && rawCallCount > options.maxRawToolCalls) ||
        (options.maxTextChars !== undefined && nextTextChars > options.maxTextChars);
      if (exceedsBudget && index > startEventIndex) break;
      endEventIndex = index;
      textChars = nextTextChars;
      if (exceedsBudget) break;
    }
    const first = trajectory[startEventIndex]!;
    const last = trajectory[endEventIndex]!;
    segments.push({
      start: first.range[0],
      end: last.range[1],
      reason: options.reason
    });
    if (endEventIndex >= trajectory.length - 1) break;
    const nextStart = Math.max(startEventIndex + 1, endEventIndex - overlapEvents + 1);
    startEventIndex = nextStart;
  }
  return segments;
}

export function trajectoryText(events: readonly SpanTrajectoryEvent[]): string {
  return events.map((event) => [
    `${event.index}:${event.tool}`,
    `range=${event.range[0]}-${event.range[1]}`,
    `calls=${event.callCount}`,
    `action=${event.action}`,
    `success=${event.success}`,
    event.errorClass ? `error=${event.errorClass}` : "",
    event.inputShape ? `in=${event.inputShape}` : "",
    event.outputShape ? `out=${event.outputShape}` : "",
    event.artifactSignal ? `artifact=${event.artifactSignal}` : "",
    event.evidence ? `evidence=${event.evidence}` : ""
  ].filter(Boolean).join(" ")).join("\n");
}

function trajectoryEventBudgetText(event: SpanTrajectoryEvent): string {
  return [
    `${event.index}:${event.tool}`,
    `${event.range[0]}-${event.range[1]}`,
    `${event.callCount}`,
    event.action,
    `${event.success}`,
    event.errorClass ?? "",
    event.inputShape ?? "",
    event.outputShape ?? "",
    event.artifactSignal ?? "",
    event.evidence ?? ""
  ].filter(Boolean).join(" ");
}
