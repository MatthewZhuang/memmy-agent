// @vitest-environment happy-dom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../../../i18n/i18n-provider.js";
import { UserMemoriesSubPage } from "../user-memories-sub-page.js";
import { createMemoryRuntimeClientStub, panelItemsOutput } from "./fixtures.js";

describe("UserMemoriesSubPage interaction", () => {
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
  });

  it("loads the UserMemory layer and deletes the selected record", async () => {
    const item = {
      id: "user_memory_1",
      kind: "user_memory" as const,
      memoryLayer: "UserMemory" as const,
      status: "activated" as const,
      title: "我最喜欢的水果是苹果",
      summary: "我最喜欢的水果是苹果",
      tags: ["User Preference"],
      metadata: {
        memoryTypes: ["User Preference"],
        sourceTurnId: "turn-1",
        sourceTurnRefs: ["turn-1"]
      },
      createdAt: "2026-08-17T00:00:00.000Z",
      updatedAt: "2026-08-17T00:00:00.000Z",
      version: 1
    };
    const listPanelItems = vi.fn(async () => panelItemsOutput([item]));
    const deleteMemory = vi.fn(async () => ({
      ok: true as const,
      id: item.id,
      kind: item.kind,
      status: "deleted" as const,
      changeSeq: 1,
      syncCursor: "cursor-1",
      serverTime: "2026-08-17T00:00:00.000Z"
    }));
    const client = createMemoryRuntimeClientStub({ listPanelItems, deleteMemory });

    await act(async () => {
      root.render(
        <I18nProvider language="zh-CN">
          <UserMemoriesSubPage client={client} />
        </I18nProvider>
      );
    });

    expect(listPanelItems).toHaveBeenCalledWith({ layer: "UserMemory", page: 1 });
    act(() => container.querySelector<HTMLButtonElement>(".memory-card")?.click());
    const deleteButton = container.querySelector<HTMLButtonElement>(".memory-delete-button");
    act(() => deleteButton?.click());
    await act(async () => deleteButton?.click());

    expect(deleteMemory).toHaveBeenCalledWith(item.id);
  });
});
