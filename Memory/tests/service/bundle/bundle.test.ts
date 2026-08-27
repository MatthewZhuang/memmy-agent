import { afterEach, describe, expect, it } from "vitest";
import { createMemoryServiceFixture } from "../../fixtures/memory-service-fixture.js";

const {
  cleanup,
  createTestService
} = createMemoryServiceFixture();

afterEach(cleanup);

describe("MemoryService / bundle", () => {
  it("redacts procedural Path and Window text when raw text export is disabled", () => {
    const { db, service } = createTestService();
    const namespace = {
      source: "codex",
      profileId: "default",
      userId: "bundle-procedural-user"
    };
    const session = service.openSession({ namespace });
    const completed = service.completeTurn("bundle-procedural-turn", {
      sessionId: session.sessionId,
      query: "raw user request",
      answer: "raw assistant answer"
    });
    const at = new Date().toISOString();
    const pathId = "path-bundle-redaction";
    const stepId = "step-bundle-redaction";
    db.db.prepare(
      `INSERT INTO episode_execution_paths (
         id, episode_id, user_id, schema_version, compilation_key,
         source_snapshot_hash, path_hash, compiler_version, model_signature,
         source_raw_turn_ids_json, source_agent_ids_json, step_count, status,
         payload_json, created_at, activated_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)`
    ).run(
      pathId,
      completed.episodeId,
      namespace.userId,
      "episode-execution-path-lite.v1",
      "compile-bundle-redaction",
      "source-bundle-redaction",
      "hash-bundle-redaction",
      "compiler-bundle-redaction",
      "model-bundle-redaction",
      JSON.stringify([completed.rawTurnId]),
      JSON.stringify([namespace.source]),
      1,
      JSON.stringify({
        id: pathId,
        steps: [{
          id: stepId,
          intent: "DERIVED_PRIVATE_INTENT",
          summary: "DERIVED_PRIVATE_SUMMARY"
        }],
        turnTransitions: [{ userObservation: "RAW_PRIVATE_OBSERVATION" }]
      }),
      at,
      at
    );
    db.db.prepare(
      `INSERT INTO trajectory_window_occurrences (
         id, path_id, episode_id, user_id, terminal_reward, evidence_role,
         schema_version, window_config_hash, scale, stride,
         start_step_index, end_step_index, step_ids_json, raw_turn_ids_json,
         semantic_text, semantic_hash, coarse_representation_version,
         embedding_signature, embedding_dim, coarse_vector_json, created_at
       ) VALUES (?, ?, ?, ?, NULL, 'unknown', ?, ?, 2, 1, 0, 1, ?, ?, ?, ?, ?, ?, 2, ?, ?)`
    ).run(
      "window-bundle-redaction",
      pathId,
      completed.episodeId,
      namespace.userId,
      "trajectory-window-occurrence.v1",
      "window-config-bundle-redaction",
      JSON.stringify([stepId, `${stepId}-2`]),
      JSON.stringify([completed.rawTurnId]),
      "DERIVED_PRIVATE_WINDOW_TEXT",
      "semantic-private-hash",
      "intent-sequence-v1",
      "embedding-bundle-redaction",
      JSON.stringify([1, 0]),
      at
    );

    const full = service.exportBundle({ includeRawText: true });
    expect(JSON.stringify(full.tables.episode_execution_paths))
      .toContain("DERIVED_PRIVATE_INTENT");
    expect(JSON.stringify(full.tables.trajectory_window_occurrences))
      .toContain("DERIVED_PRIVATE_WINDOW_TEXT");

    const redacted = service.exportBundle();
    const serialized = JSON.stringify({
      paths: redacted.tables.episode_execution_paths,
      windows: redacted.tables.trajectory_window_occurrences
    });
    expect(serialized).not.toContain("DERIVED_PRIVATE_INTENT");
    expect(serialized).not.toContain("DERIVED_PRIVATE_SUMMARY");
    expect(serialized).not.toContain("RAW_PRIVATE_OBSERVATION");
    expect(serialized).not.toContain("DERIVED_PRIVATE_WINDOW_TEXT");
    expect(serialized).toContain("[REDACTED]");
  });

  it("redacts in-flight L3 evidence while full bundles preserve recoverable runtime state", () => {
    const first = createTestService();
    const namespace = {
      source: "codex",
      profileId: "default",
      sessionKey: "bundle-l3-session",
      userId: "bundle-l3-user",
    };
    const opened = first.service.openSession({
      l3WorldModelProtocolVersion: 2,
      l3WorldModelTransition: "resume_only",
      workspaceUri: "file:///tmp/bundle-l3-project",
      workspaceHostId: "a".repeat(64),
      namespace,
    });
    const firstTurn = first.service.completeTurn("bundle-l3-turn-1", {
      sessionId: opened.sessionId,
      query: "Keep project edits inside the configured module boundary.",
      answer: "The edit stayed inside the configured module.",
    });
    first.service.l3WorldModelBoundary(opened.sessionId, {
      requestId: "53126537-2c75-48be-91f5-d32a6d93f6f7",
      adapterId: "codex-memory",
      source: "codex",
      namespace: { ...namespace, projectId: opened.projectId! },
      trigger: "token_compaction",
      throughL1MemoryId: firstTurn.l1MemoryId,
    });
    first.service.completeTurn("bundle-l3-turn-2", {
      sessionId: opened.sessionId,
      query: "A later turn has not reached a boundary yet.",
      answer: "It remains an unfrozen trace.",
    });

    const full = first.service.exportBundle({ includeRawText: true });
    expect(full.tables.l3_world_model_evidence_batches).toHaveLength(1);
    expect(full.tables.l3_world_model_batch_targets).toHaveLength(2);
    expect((full.tables.evolution_jobs as Array<Record<string, unknown>>)
      .filter((row) => row.job_type === "l3_world_model_update")).toHaveLength(2);
    expect(full.tables.l3_world_model_project_environment_state).toHaveLength(1);
    expect(full.tables.l3_world_model_project_environment_operations).toBeUndefined();

    const redacted = first.service.exportBundle();
    expect(redacted.tables.l3_world_model_evidence_batches).toEqual([]);
    expect(redacted.tables.l3_world_model_batch_targets).toEqual([]);
    expect((redacted.tables.evolution_jobs as Array<Record<string, unknown>>)
      .filter((row) => row.job_type === "l3_world_model_update" || row.job_type === "project_environment_profile"))
      .toEqual([]);
    expect(redacted.tables.l3_world_model_project_environment_operations).toBeUndefined();
    expect(redacted.tables.l3_world_model_project_environment_state).toEqual([
      expect.objectContaining({
        status: "uninitialized",
        current_scan_id: null,
        last_error: null,
      }),
    ]);
    expect(redacted.tables.l3_world_model_session_cursors).toEqual([
      expect.objectContaining({ session_id: opened.sessionId, last_scheduled_seq: 2 }),
    ]);
    expect(first.db.db.prepare(
      `SELECT last_scheduled_seq FROM l3_world_model_session_cursors WHERE session_id = ?`
    ).get(opened.sessionId)).toEqual({ last_scheduled_seq: 1 });

    const second = createTestService();
    expect(second.service.importBundle({ bundle: redacted }).ok).toBe(true);
    expect(second.db.db.prepare(
      `SELECT last_scheduled_seq FROM l3_world_model_session_cursors WHERE session_id = ?`
    ).get(opened.sessionId)).toEqual({ last_scheduled_seq: 2 });
  });

  it("exports bundles across namespaces", async () => {
    const { db, service } = createTestService();
    const namespaceA = {
      source: "codex",
      profileId: "default",
      userId: "shared-export-user",
      workspaceId: "workspace-export-a"
    };
    const namespaceB = {
      source: "codex",
      profileId: "default",
      userId: "shared-export-user",
      workspaceId: "workspace-export-b"
    };
    const sessionA = service.openSession({ namespace: namespaceA });
    const sessionB = service.openSession({ namespace: namespaceB });
    const completeA = service.completeTurn("turn-export-a", {
      sessionId: sessionA.sessionId,
      query: "scoped export memory alpha",
      answer: "stored only in export namespace alpha",
      artifacts: [{
        kind: "file",
        uri: "file:///tmp/export-alpha.txt"
      }]
    });
    const completeB = service.completeTurn("turn-export-b", {
      sessionId: sessionB.sessionId,
      query: "scoped export memory beta",
      answer: "stored only in export namespace beta"
    });
    await service.runWorkerOnce(20);
    const recallA = await service.search({
      namespace: namespaceA,
      query: "scoped export memory alpha",
      layers: ["L1"],
      limit: 5
    });
    const recallB = await service.search({
      namespace: namespaceB,
      query: "scoped export memory beta",
      layers: ["L1"],
      limit: 5
    });

    const bundleA = service.exportBundle({ namespace: namespaceA });
    const memoryIds = (bundleA.tables.memories as Array<Record<string, unknown>>).map((row) => row.id);
    expect(memoryIds).toContain(completeA.l1MemoryId);
    expect(memoryIds).toContain(completeB.l1MemoryId);
    const sessionIds = (bundleA.tables.sessions as Array<Record<string, unknown>>).map((row) => row.id);
    expect(sessionIds.sort()).toEqual([sessionA.sessionId, sessionB.sessionId].sort());
    const rawTurnIds = (bundleA.tables.raw_turns as Array<Record<string, unknown>>).map((row) => row.id);
    expect(rawTurnIds).toContain(completeA.rawTurnId);
    expect(rawTurnIds).toContain(completeB.rawTurnId);
    const recallIds = (bundleA.tables.recall_events as Array<Record<string, unknown>>).map((row) => row.id);
    expect(recallIds).toContain(recallA.searchEventId);
    expect(recallIds).toContain(recallB.searchEventId);
    const artifactRawTurnIds = (bundleA.tables.artifacts as Array<Record<string, unknown>>)
      .map((row) => row.raw_turn_id);
    expect(artifactRawTurnIds).toEqual([completeA.rawTurnId]);
    const jobSessionIds = new Set((bundleA.tables.evolution_jobs as Array<Record<string, unknown>>)
      .map((row) => row.session_id));
    expect(jobSessionIds).toEqual(new Set([sessionA.sessionId, sessionB.sessionId]));
    const changeNamespaces = new Set((bundleA.tables.memory_change_log as Array<Record<string, unknown>>)
      .map((row) => row.namespace_id));
    expect([...changeNamespaces].some((namespace) => String(namespace).includes("workspace-export-a"))).toBe(true);
    expect([...changeNamespaces].some((namespace) => String(namespace).includes("workspace-export-b"))).toBe(true);

    db.close();
  });
});
