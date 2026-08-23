// @vitest-environment happy-dom

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { WorkspaceEnvironmentSnapshot } from "../../api/memmy-agent-client.js";
import { I18nProvider } from "../../i18n/i18n-provider.js";
import { AgentWorkspaceContext } from "../agent-workspace-context.js";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function environment(branch: string | null = "zy_git_v1.0.7"): WorkspaceEnvironmentSnapshot {
  return {
    scope_kind: "project",
    scope_key: "project-1",
    cwd: "/workspace/memmy-agent",
    status: "ready",
    revision: `revision-${branch ?? "detached"}`,
    captured_at: "2026-08-11T08:00:00.000Z",
    repository: {
      display_name: "memmy-agent",
      root: "/workspace/memmy-agent",
      head_sha: "84d10f8f00",
      branch,
      detached: branch == null,
      upstream: null,
      ahead: 0,
      behind: 0,
      worktree: "clean",
    },
    changes: { file_count: 0, additions: 0, deletions: 0, conflicts: 0, staged: 0, unstaged: 0, untracked: 0 },
    goal: null,
  };
}

describe("AgentWorkspaceContext", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function renderContext(
    snapshot = environment(),
    onSwitchBranch = vi.fn(async () => true),
    onCreateOrCheckoutBranch = vi.fn(async () => true),
  ) {
    const branches = ["main", "zy_git_v1.0.7", "zy_v1.0.6", "release/1", "release/2", "release/3", "release/4", "release/5"];
    act(() => {
      root.render(
        <I18nProvider language="zh-CN">
          <AgentWorkspaceContext
            snapshot={snapshot}
            branches={branches}
            loading={false}
            error={null}
            onSwitchBranch={onSwitchBranch}
            onCreateOrCheckoutBranch={onCreateOrCheckoutBranch}
          />
        </I18nProvider>
      );
    });
    return { branches, onSwitchBranch, onCreateOrCheckoutBranch };
  }

  it("shows the local mode and current branch from the shared snapshot", () => {
    renderContext();

    expect(container.textContent).toContain("本地");
    expect(container.textContent).toContain("zy_git_v1.0.7");
  });

  it("renders nothing when the shared snapshot is not a Git repository", () => {
    act(() => {
      root.render(
        <I18nProvider language="zh-CN">
          <AgentWorkspaceContext
            snapshot={{ ...environment(), status: "not_git", repository: null, changes: null }}
            branches={[]}
            loading={false}
            error={null}
            onSwitchBranch={vi.fn(async () => true)}
            onCreateOrCheckoutBranch={vi.fn(async () => true)}
          />
        </I18nProvider>
      );
    });

    expect(container.innerHTML).toBe("");
  });

  it("opens the work mode menu without presenting unsupported modes as active actions", () => {
    renderContext();
    const modeButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "本地");

    act(() => modeButton!.click());

    expect(container.textContent).toContain("工作模式");
    expect([...container.querySelectorAll("button:disabled")].map((button) => button.textContent)).toEqual([
      "新工作树",
    ]);
  });

  it("searches local branches and switches the selected branch", async () => {
    const { onSwitchBranch } = renderContext();
    const branchButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "zy_git_v1.0.7");
    act(() => branchButton!.click());

    expect((container.querySelector('[role="listbox"]') as HTMLElement).style.getPropertyValue("--visible-branch-count")).toBe("5");
    const search = container.querySelector('input[aria-label="搜索分支"]') as HTMLInputElement;
    act(() => {
      search.value = "main";
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const mainOption = [...container.querySelectorAll('[role="option"]')].find((option) => option.textContent?.includes("main")) as HTMLButtonElement;
    await act(async () => mainOption.click());

    expect(onSwitchBranch).toHaveBeenCalledWith("main");
  });

  it("keeps the create action outside the five-row scroller and creates a new branch", async () => {
    const { onCreateOrCheckoutBranch } = renderContext();
    const branchButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "zy_git_v1.0.7");
    act(() => branchButton!.click());

    const listbox = container.querySelector('[role="listbox"]') as HTMLElement;
    const createButton = [...container.querySelectorAll("button")]
      .find((button) => button.textContent === "创建或检出新分支") as HTMLButtonElement;
    expect(listbox.contains(createButton)).toBe(false);

    act(() => createButton.click());
    const input = container.querySelector('input[aria-label="新分支名称"]') as HTMLInputElement;
    act(() => {
      const setInputValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      setInputValue?.call(input, "feature/new-branch");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const confirmButton = [...container.querySelectorAll("button")]
      .find((button) => button.textContent === "创建并检出") as HTMLButtonElement;
    await act(async () => confirmButton.click());

    expect(onCreateOrCheckoutBranch).toHaveBeenCalledWith("feature/new-branch");
  });

  it("requires confirmation before switching a dirty workspace", async () => {
    const confirm = vi.fn(() => false);
    vi.stubGlobal("confirm", confirm);
    const dirty = environment();
    dirty.repository = dirty.repository ? { ...dirty.repository, worktree: "dirty" } : null;
    const { onSwitchBranch } = renderContext(dirty);
    const branchButton = [...container.querySelectorAll("button")].find((button) => button.textContent === "zy_git_v1.0.7");
    act(() => branchButton!.click());
    const mainOption = [...container.querySelectorAll('[role="option"]')].find((option) => option.textContent?.includes("main")) as HTMLButtonElement;
    await act(async () => mainOption.click());

    expect(confirm).toHaveBeenCalledTimes(1);
    expect(onSwitchBranch).not.toHaveBeenCalled();
  });
});
