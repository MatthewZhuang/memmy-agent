import { describe, expect, it } from "vitest";
import {
  isInternalInfoEligibleForPositiveL2,
  resolveCaptureTurnRole
} from "../../../src/service/evolution/span-pipeline.js";
import { stripL1CaptureChrome } from "../../../src/service/user-memory/user-memory.js";

describe("turn memory capture schema", () => {
  it("treats local_subproblem with a distinct intent as L2-eligible", () => {
    expect(resolveCaptureTurnRole({
      turnRole: "local_subproblem",
      taskSummary: "把这个项目跑通并做完列出的优化",
      intent: "消除查询中的N+1"
    })).toEqual({
      turnRole: "local_subproblem",
      intent: "消除查询中的N+1",
      policyEligible: true
    });
  });

  it("downgrades a restated intent to continuation", () => {
    expect(resolveCaptureTurnRole({
      turnRole: "local_subproblem",
      taskSummary: "把这个项目跑通",
      intent: "把这个项目跑通"
    })).toEqual({
      turnRole: "continuation",
      intent: "",
      policyEligible: false
    });
  });

  it("defaults missing turn_role to continuation", () => {
    expect(resolveCaptureTurnRole({
      turnRole: undefined,
      taskSummary: "确认是否喜欢看财经类新闻",
      intent: "should be dropped"
    })).toEqual({
      turnRole: "continuation",
      intent: "",
      policyEligible: false
    });
  });

  it("uses turn_role for new L2 eligibility and policy_eligible only as a legacy fallback", () => {
    expect(isInternalInfoEligibleForPositiveL2({
      turn_role: "local_subproblem",
      intent: "消除查询中的N+1"
    })).toBe(true);
    expect(isInternalInfoEligibleForPositiveL2({
      turn_role: "continuation",
      intent: "",
      policy_eligible: true
    })).toBe(false);
    expect(isInternalInfoEligibleForPositiveL2({
      policy_eligible: true
    })).toBe(true);
    expect(isInternalInfoEligibleForPositiveL2({
      policy_eligible: false
    })).toBe(false);
  });

  it("strips chrome before previous-query context", () => {
    expect(stripL1CaptureChrome("<timestamp>Thursday</timestamp>\n<user_query>\n德国物价怎么样\n</user_query>"))
      .toBe("德国物价怎么样");
  });
});
