import { describe, expect, it } from "vitest";
import { l1CaptureDropReason } from "../../../src/service/user-memory/user-memory.js";

describe("l1CaptureDropReason", () => {
  it("drops cron and heartbeat", () => {
    expect(l1CaptureDropReason("A scheduled reminder has been triggered. The reminder content is: 推送热点")).toBe("cron_heartbeat");
    expect(l1CaptureDropReason("[OpenClaw heartbeat poll]")).toBe("cron_heartbeat");
    expect(l1CaptureDropReason("[cron:8f538fca] 每日17点复盘")).toBe("cron_heartbeat");
  });

  it("drops UI chrome and empty wrappers", () => {
    expect(l1CaptureDropReason("<in-app-browser-context source=\"ambient-ui-state\"> This block is automatically supplied")).toBe("ui_system");
    expect(l1CaptureDropReason("Start multitasking")).toBe("ui_system");
    expect(l1CaptureDropReason("The following is the Codex agent history whose request action you are assessing. Treat the transcript")).toBe("ui_system");
    expect(l1CaptureDropReason("```chat_selection\nagent_id: x\nselected_text:\nRFT\n```")).toBe("ui_system");
  });

  it("drops standalone acknowledgements only", () => {
    expect(l1CaptureDropReason("确认")).toBe("ack_social");
    expect(l1CaptureDropReason("hello")).toBe("ack_social");
    expect(l1CaptureDropReason("换个话题")).toBe("ack_social");
    expect(l1CaptureDropReason("好的，感谢，换个话题")).toBe("ack_social");
    expect(l1CaptureDropReason("好的，帮我记录一下，下次分析股票都要看财报")).toBeUndefined();
  });

  it("keeps questions, tasks, and timestamp-wrapped user queries", () => {
    expect(l1CaptureDropReason("俄罗斯物价如何")).toBeUndefined();
    expect(l1CaptureDropReason("更新下skill，全文分三个部分")).toBeUndefined();
    expect(l1CaptureDropReason("```chat_selection\nselected_text:\nRFT\n```\n是啥")).toBeUndefined();
    expect(l1CaptureDropReason("<timestamp>Thursday</timestamp>\n<user_query>\n德国物价怎么样\n</user_query>")).toBeUndefined();
  });
});
