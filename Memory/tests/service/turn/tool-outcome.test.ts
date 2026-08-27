import { describe, expect, it } from "vitest";
import { toolObservationEvent } from "../../../src/service/session/session-turn-service.js";
import { inferToolOutcome } from "../../../src/service/turn/tool-outcome.js";

describe("canonical tool outcome inference", () => {
  it.each([
    ["error prefix", "Error: command unavailable", "ERROR_OUTPUT"],
    ["non-zero exit", "command exited with 7", "EXIT_7"],
    ["traceback", "Traceback (most recent call last):\nNameError: missing", "TRACEBACK"],
    ["HTTP failure", "request failed with HTTP 503", "HTTP_503"],
    ["structured failure", { success: false, error: "provider unavailable" }, undefined],
    ["structured HTTP failure", { status: 429, message: "rate limited" }, "HTTP_429"]
  ])("recognizes %s evidence embedded in tool output", (_label, output, errorCode) => {
    const inferred = inferToolOutcome({
      call: { name: "tool", output, success: true }
    });

    expect(inferred.status).toBe("failure");
    if (errorCode) expect(inferred.errorCode).toBe(errorCode);
  });

  it("does not treat an arbitrary non-empty output as success", () => {
    expect(inferToolOutcome({
      call: { name: "read", output: "artifact contents" }
    })).toEqual({ status: "unknown" });
    expect(inferToolOutcome({
      call: {
        name: "extract_text",
        output: "Training guide\nError prevention and tests failed examples are discussed here."
      }
    })).toEqual({ status: "unknown" });
  });

  it("accepts explicit structured and exit-zero success evidence", () => {
    expect(inferToolOutcome({
      call: { name: "write", output: { ok: true } }
    }).status).toBe("success");
    expect(inferToolOutcome({
      call: { name: "exec", output: "Exit code: 0" }
    }).status).toBe("success");
  });

  it("shares protocol-error inference with live tool observation ingestion", () => {
    const observed = toolObservationEvent({
      sessionId: "session-outcome-test",
      toolName: "exec_command",
      toolCallId: "call-outcome-test",
      result: "Error: command exited with 1\nNameError: missing_symbol"
    });

    expect(observed.phase).toBe("complete");
    expect(observed.event.success).toBe(false);
    expect(observed.event.errorCode).toBe("EXIT_1");
    expect(observed.event.error).toContain("NameError");
  });

  it("keeps an explicit observation error failed even when it has no string message", () => {
    const observed = toolObservationEvent({
      sessionId: "session-outcome-test",
      toolName: "custom_tool",
      error: { code: "CUSTOM_FAILURE" }
    });

    expect(observed.phase).toBe("error");
    expect(observed.event.success).toBe(false);
    expect(observed.event.error).toBe("The tool observation reported an error.");
  });

  it("uses the successful completion channel only after excluding failure markers", () => {
    expect(inferToolOutcome({
      call: { name: "fetch", output: "retrieved records" },
      result: "retrieved records",
      completionObserved: true
    }).status).toBe("success");
    expect(inferToolOutcome({
      call: { name: "fetch", output: "HTTP 403: access denied" },
      result: "HTTP 403: access denied",
      completionObserved: true
    }).status).toBe("failure");
  });
});
