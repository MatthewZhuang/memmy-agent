import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";
import { MemoryDb, SCHEMA_MIGRATION_ID, SCHEMA_VERSION } from "../../src/index.js";
import { Repositories } from "../../src/storage/repositories.js";
import type { MemoryRow } from "../../src/types.js";

describe("repository sqlite schema contract", () => {
  it("adds global memory indexes when reopening a database with only user-prefixed indexes", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-global-index-upgrade-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const initial = new MemoryDb({ path: dbPath });
      initial.db.exec(`
        DROP INDEX idx_memories_layer_status_updated;
        DROP INDEX idx_memories_conversation_updated;
        DROP INDEX idx_memories_content_hash_layer;
        DROP INDEX idx_memories_key_layer;
        CREATE INDEX idx_memories_user_layer_status_updated
          ON memories (user_id, memory_layer, status, updated_at DESC);
        CREATE INDEX idx_memories_user_conversation
          ON memories (user_id, conversation_id, updated_at DESC);
        CREATE INDEX idx_memories_hash
          ON memories (user_id, content_hash, memory_layer);
        CREATE INDEX idx_memories_key
          ON memories (user_id, memory_key, memory_layer);
      `);
      initial.close();

      const reopened = new MemoryDb({ path: dbPath });
      const indexes = reopened.db
        .prepare(`PRAGMA index_list(memories)`)
        .all() as Array<{ name: string }>;
      expect(indexes.map((index) => index.name)).toEqual(expect.arrayContaining([
        "idx_memories_layer_status_updated",
        "idx_memories_layer_status_created",
        "idx_memories_conversation_updated",
        "idx_memories_content_hash_layer",
        "idx_memories_key_layer"
      ]));
      const queryPlan = reopened.db.prepare(
        `EXPLAIN QUERY PLAN
         SELECT *
         FROM memories
         WHERE memory_layer = ?
           AND memory_key = ?
           AND deleted_at IS NULL`
      ).all("L1", "shared-key") as Array<{ detail: string }>;
      expect(queryPlan.some((step) => step.detail.includes("idx_memories_key_layer"))).toBe(true);
      reopened.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("creates the runtime tables on a fresh sqlite database", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-schema-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const schema = db.schemaVersion();
      expect(schema.version).toBe(SCHEMA_VERSION);
      expect(schema.lastMigrationId).toBe(SCHEMA_MIGRATION_ID);

      const tables = db.db
        .prepare(`SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')`)
        .all() as Array<{ name: string }>;
      expect(tables.map((table) => table.name)).toEqual(expect.arrayContaining([
        "memories",
        "sessions",
        "episodes",
        "raw_turns",
        "feedback",
        "decision_repairs",
        "l2_candidate_pool",
        "trace_policy_links",
        "episode_procedural_paths",
        "procedural_span_occurrences",
        "procedural_step_occurrences",
        "procedural_step_embeddings",
        "procedural_step_clusters",
        "procedural_step_cluster_members",
        "step_sequence_patterns",
        "step_sequence_pattern_occurrences",
        "step_sequence_policy_versions",
        "episode_step_policy_projections",
        "step_policy_skill_patterns",
        "step_policy_skill_pattern_occurrences",
        "procedural_span_occurrence_embeddings",
        "episode_span_credit_runs",
        "procedural_span_credits",
        "procedural_span_clusters",
        "procedural_span_cluster_members",
        "procedural_policy_versions",
        "procedural_policy_occurrences",
        "episode_policy_projections",
        "episode_policy_projection_nodes",
        "episode_capability_signatures",
        "episode_capability_affinities",
        "policy_sequence_patterns",
        "policy_sequence_pattern_edges",
        "policy_sequence_pattern_occurrences",
        "procedural_skill_candidates",
        "span_clusters",
        "span_cluster_members",
        "skill_trials",
        "recall_events",
        "memory_change_log",
        "idempotency_keys",
        "evolution_jobs",
        "embedding_retry_queue",
        "memory_processing_state",
        "artifacts",
        "audit_logs",
        "memory_vector_entries"
      ]));
      expect(tables.map((table) => table.name)).not.toEqual(expect.arrayContaining([
        "memory_embeddings",
        "memory_vectors"
      ]));
      const spanCreditColumns = db.db.prepare(
        `PRAGMA table_info(procedural_span_credits)`
      ).all() as Array<{ name: string }>;
      expect(spanCreditColumns.map((column) => column.name)).toContain("attribution_type");
      const stepPatternColumns = db.db.prepare(
        `PRAGMA table_info(step_sequence_patterns)`
      ).all() as Array<{ name: string }>;
      expect(stepPatternColumns.map((column) => column.name)).toEqual(expect.arrayContaining([
        "selected_occurrence_count",
        "selected_episode_count",
        "superseded_by_pattern_id"
      ]));
      const stepOccurrenceColumns = db.db.prepare(
        `PRAGMA table_info(step_sequence_pattern_occurrences)`
      ).all() as Array<{ name: string }>;
      expect(stepOccurrenceColumns.map((column) => column.name)).toContain("is_selected");
      const skillPatternColumns = db.db.prepare(
        `PRAGMA table_info(step_policy_skill_patterns)`
      ).all() as Array<{ name: string }>;
      expect(skillPatternColumns.map((column) => column.name)).toEqual(expect.arrayContaining([
        "selected_occurrence_count",
        "selected_episode_count",
        "superseded_by_pattern_id"
      ]));
      const skillOccurrenceColumns = db.db.prepare(
        `PRAGMA table_info(step_policy_skill_pattern_occurrences)`
      ).all() as Array<{ name: string }>;
      expect(skillOccurrenceColumns.map((column) => column.name)).toContain("is_selected");
      expect(db.db.prepare(`SELECT vec_version() AS version`).get()).toEqual({ version: "v0.1.9" });
      const memoryColumns = db.db.prepare(`PRAGMA table_info(memories)`).all() as Array<{ name: string }>;
      expect(memoryColumns.map((column) => column.name)).not.toEqual(expect.arrayContaining([
        "embedding",
        "embedding_model",
        "embedding_dim"
      ]));
      const vectorEntryColumns = db.db
        .prepare(`PRAGMA table_info(memory_vector_entries)`)
        .all() as Array<{ name: string }>;
      expect(vectorEntryColumns.map((column) => column.name)).not.toContain("embedding");
      db.db.prepare(
        `INSERT INTO memories (
          id, timeline, user_id, memory_type, memory_value, tags_json, info_json,
          properties_json, memory_layer, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        "schema-vector-field-span",
        "2026-08-14T00:00:00.000Z",
        "schema-user",
        "LongTermMemory",
        "schema vector field span",
        "[]",
        "{}",
        JSON.stringify({ internal_info: { memory_layer: "L1", memory_kind: "span" } }),
        "L1",
        "2026-08-14T00:00:00.000Z",
        "2026-08-14T00:00:00.000Z"
      );
      db.db.prepare(
        `INSERT INTO memory_vector_entries (
          memory_id, vector_field, embedding_model, embedding_provider, embedding_dim, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)`
      ).run("schema-vector-field-span", "vec_goal", "test", "test", 3, "2026-08-14T00:00:00.000Z");
      db.db.prepare(
        `INSERT INTO memory_vector_entries (
          memory_id, vector_field, embedding_model, embedding_provider, embedding_dim, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)`
      ).run("schema-vector-field-span", "vec_policy", "test", "test", 3, "2026-08-14T00:00:00.000Z");
      const tableNames = new Set(tables.map((table) => table.name));
      expect([...tableNames].some((name) => name.startsWith("cloud_"))).toBe(false);

      const skillTrialColumns = db.db
        .prepare(`PRAGMA table_info(skill_trials)`)
        .all() as Array<{ name: string; notnull: number }>;
      const sessionColumns = db.db
        .prepare(`PRAGMA table_info(sessions)`)
        .all() as Array<{ name: string }>;
      expect(sessionColumns.map((column) => column.name)).toContain("source");
      expect(sessionColumns.map((column) => column.name)).not.toContain("agent_kind");
      expect(sessionColumns.map((column) => column.name)).toContain("last_seen_at");
      const sessionIndexes = db.db
        .prepare(`PRAGMA index_list(sessions)`)
        .all() as Array<{ name: string }>;
      expect(sessionIndexes.map((index) => index.name)).toContain("idx_sessions_host_scope");
      const episodeColumns = db.db
        .prepare(`PRAGMA table_info(episodes)`)
        .all() as Array<{ name: string }>;
      expect(episodeColumns.map((column) => column.name)).toEqual(expect.arrayContaining([
        "turn_count",
        "feedback_ids_json",
        "decision_repair_ids_json",
        "l2_policy_ids_json",
        "l3_world_model_ids_json",
        "skill_memory_ids_json",
        "r_task",
        "reward_detail_json",
        "pipeline_status"
      ]));
      const episodeIndexes = db.db
        .prepare(`PRAGMA index_list(episodes)`)
        .all() as Array<{ name: string }>;
      expect(episodeIndexes.map((index) => index.name)).toEqual(expect.arrayContaining([
        "idx_episodes_project_updated",
        "idx_episodes_pipeline"
      ]));
      expect(skillTrialColumns.find((column) => column.name === "episode_id")?.notnull).toBe(1);
      expect(skillTrialColumns.map((column) => column.name)).toEqual(expect.arrayContaining([
        "status",
        "outcome",
        "feedback_id"
      ]));
      const skillTrialIndexes = db.db
        .prepare(`PRAGMA index_list(skill_trials)`)
        .all() as Array<{ name: string }>;
      expect(skillTrialIndexes.map((index) => index.name)).toEqual(expect.arrayContaining([
        "idx_skill_trials_skill_created",
        "idx_skill_trials_user_status",
        "idx_skill_trials_episode_status",
        "idx_skill_trials_l1_status",
        "idx_skill_trials_raw_status"
      ]));
      const feedbackIndexes = db.db
        .prepare(`PRAGMA index_list(feedback)`)
        .all() as Array<{ name: string }>;
      expect(feedbackIndexes.map((index) => index.name)).toEqual(expect.arrayContaining([
        "idx_feedback_user_created",
        "idx_feedback_episode_created",
        "idx_feedback_raw_turn_created",
        "idx_feedback_context"
      ]));
      const tracePolicyLinkIndexes = db.db
        .prepare(`PRAGMA index_list(trace_policy_links)`)
        .all() as Array<{ name: string }>;
      expect(tracePolicyLinkIndexes.map((index) => index.name)).toEqual(expect.arrayContaining([
        "idx_trace_policy_links_l1",
        "idx_trace_policy_links_l2"
      ]));
      const candidatePoolColumns = db.db
        .prepare(`PRAGMA table_info(l2_candidate_pool)`)
        .all() as Array<{ name: string }>;
      expect(candidatePoolColumns.map((column) => column.name)).toContain("expires_at");
      const repairColumns = db.db
        .prepare(`PRAGMA table_info(decision_repairs)`)
        .all() as Array<{ name: string }>;
      expect(repairColumns.map((column) => column.name)).toEqual(expect.arrayContaining([
        "project_id",
        "context_hash",
        "preference",
        "anti_pattern",
        "high_value_memory_ids_json",
        "low_value_memory_ids_json",
        "attached_policy_memory_ids_json",
        "meta_json"
      ]));
      const repairIndexes = db.db
        .prepare(`PRAGMA index_list(decision_repairs)`)
        .all() as Array<{ name: string }>;
      expect(repairIndexes.map((index) => index.name)).toEqual(expect.arrayContaining([
        "idx_decision_repairs_context",
        "idx_decision_repairs_episode"
      ]));
      const retryColumns = db.db
        .prepare(`PRAGMA table_info(embedding_retry_queue)`)
        .all() as Array<{ name: string }>;
      expect(retryColumns.map((column) => column.name)).toEqual(expect.arrayContaining([
        "target_kind",
        "target_id",
        "vector_field",
        "source_text",
        "embed_role",
        "claimed_by",
        "lease_until"
      ]));
      const retryIndexes = db.db
        .prepare(`PRAGMA index_list(embedding_retry_queue)`)
        .all() as Array<{ name: string }>;
      expect(retryIndexes.map((index) => index.name)).toEqual(expect.arrayContaining([
        "idx_embedding_retry_due",
        "idx_embedding_retry_target"
      ]));
      db.db.prepare(
        `INSERT INTO embedding_retry_queue (
          id, target_kind, target_id, vector_field, source_text, embed_role,
          status, attempts, max_attempts, next_attempt_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      ).run(
        "retry-span-goal",
        "span",
        "schema-vector-field-span",
        "vec_goal",
        "Goal: schema accepts span goal embeddings",
        "document",
        "pending",
        0,
        5,
        0,
        0,
        "2026-08-14T00:00:00.000Z"
      );
      const apiLogColumns = db.db.prepare(`PRAGMA table_info(api_logs)`).all() as Array<{ name: string }>;
      expect(apiLogColumns.map((column) => column.name)).toContain("source_agent");
      const apiLogIndexes = db.db.prepare(`PRAGMA index_list(api_logs)`).all() as Array<{ name: string }>;
      expect(apiLogIndexes.map((index) => index.name)).toContain("idx_api_logs_tool_source_time");
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("migrates schema v2 to the latest version without deleting user data", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-v2-source-agent-migration-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      const repos = new Repositories(seeded.db);
      repos.memories.insert(schemaVectorMemory());
      repos.runtime.createSession({
        id: "session-preserved",
        userId: "old-user",
        source: "codex",
        profileId: "default",
        status: "open",
        meta: {},
        openedAt: "2026-01-01T00:00:00.000Z",
        updatedAt: "2026-01-01T00:00:00.000Z"
      });
      repos.runtime.insertApiLog({
        toolName: "memory_search",
        sourceAgent: "codex",
        inputJson: JSON.stringify({ query: "preserve this log", sessionId: "session-preserved" }),
        outputJson: JSON.stringify({ candidates: [] }),
        durationMs: 3,
        success: true,
        calledAt: "2026-01-01T00:00:01.000Z"
      });
      seeded.close();

      const v2 = new Database(dbPath);
      v2.exec(`
        DROP INDEX idx_api_logs_tool_source_time;
        ALTER TABLE api_logs DROP COLUMN source_agent;
        DELETE FROM schema_migrations;
        INSERT INTO schema_migrations (id, version, applied_at, checksum)
        VALUES ('002_sqlite_vec_storage', 2, '2026-01-01T00:00:00.000Z', 'v2');
      `);
      v2.close();

      const migrated = new MemoryDb({ path: dbPath });
      expect(migrated.schemaVersion()).toEqual({
        version: SCHEMA_VERSION,
        lastMigrationId: SCHEMA_MIGRATION_ID
      });
      expect(migrated.db.prepare(`SELECT COUNT(*) AS count FROM memories`).get()).toEqual({ count: 1 });
      expect(migrated.db.prepare(`SELECT COUNT(*) AS count FROM sessions`).get()).toEqual({ count: 1 });
      expect(migrated.db.prepare(`SELECT COUNT(*) AS count FROM api_logs`).get()).toEqual({ count: 1 });
      expect(migrated.db.prepare(`SELECT input_json, source_agent FROM api_logs`).get()).toEqual({
        input_json: JSON.stringify({ query: "preserve this log", sessionId: "session-preserved" }),
        source_agent: null
      });
      expect((migrated.db.prepare(`PRAGMA index_list(api_logs)`).all() as Array<{ name: string }>)
        .map((index) => index.name)).toContain("idx_api_logs_tool_source_time");
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
      migrated.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("migrates schema v5 by adding procedural Path, cluster, and Policy tables", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-v5-procedural-path-migration-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      const repos = new Repositories(seeded.db);
      repos.memories.insert(schemaVectorMemory());
      seeded.db.exec(`
        DROP TABLE episode_capability_affinities;
        DROP TABLE episode_capability_signatures;
        DROP TABLE procedural_skill_candidates;
        DROP TABLE policy_sequence_pattern_occurrences;
        DROP TABLE policy_sequence_patterns;
        DROP TABLE procedural_policy_occurrences;
        DROP TABLE procedural_policy_versions;
        DROP TABLE episode_policy_projection_nodes;
        DROP TABLE episode_policy_projections;
        DROP TABLE procedural_span_cluster_members;
        DROP TABLE procedural_span_clusters;
        DROP TABLE procedural_span_credits;
        DROP TABLE episode_span_credit_runs;
        DROP TABLE procedural_span_occurrences;
        DROP TABLE episode_procedural_paths;
        DELETE FROM schema_migrations;
        INSERT INTO schema_migrations (id, version, applied_at, checksum)
        VALUES ('005_span_embedding_vectors', 5, '2026-08-14T00:00:00.000Z', 'v5');
      `);
      seeded.close();

      const migrated = new MemoryDb({ path: dbPath });
      const tables = (migrated.db.prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table'`
      ).all() as Array<{ name: string }>).map((table) => table.name);
      expect(migrated.schemaVersion()).toEqual({
        version: SCHEMA_VERSION,
        lastMigrationId: SCHEMA_MIGRATION_ID
      });
      expect(tables).toEqual(expect.arrayContaining([
        "episode_procedural_paths",
        "procedural_span_occurrences",
        "episode_span_credit_runs",
        "procedural_span_credits",
        "procedural_span_clusters",
        "procedural_span_cluster_members",
        "procedural_policy_versions",
        "procedural_policy_occurrences",
        "episode_policy_projections",
        "episode_policy_projection_nodes",
        "episode_capability_signatures",
        "episode_capability_affinities",
        "policy_sequence_patterns",
        "policy_sequence_pattern_occurrences",
        "procedural_skill_candidates"
      ]));
      expect(migrated.db.prepare(`SELECT COUNT(*) AS count FROM memories`).get()).toEqual({ count: 1 });
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
      migrated.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("migrates schema v9 by adding Episode Policy Projection tables", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-v9-policy-projection-migration-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      seeded.db.exec(`
        DROP TABLE episode_capability_affinities;
        DROP TABLE episode_capability_signatures;
        DROP TABLE procedural_skill_candidates;
        DROP TABLE policy_sequence_pattern_occurrences;
        DROP TABLE policy_sequence_patterns;
        DROP TABLE episode_policy_projection_nodes;
        DROP TABLE episode_policy_projections;
        DELETE FROM schema_migrations;
        INSERT INTO schema_migrations (id, version, applied_at, checksum)
        VALUES ('009_procedural_span_embeddings', 9, '2026-08-20T00:00:00.000Z', 'v9');
      `);
      seeded.close();

      const migrated = new MemoryDb({ path: dbPath });
      const tables = (migrated.db.prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table'`
      ).all() as Array<{ name: string }>).map((table) => table.name);
      expect(migrated.schemaVersion()).toEqual({
        version: SCHEMA_VERSION,
        lastMigrationId: SCHEMA_MIGRATION_ID
      });
      expect(tables).toEqual(expect.arrayContaining([
        "episode_policy_projections",
        "episode_policy_projection_nodes",
        "episode_capability_signatures",
        "episode_capability_affinities",
        "policy_sequence_patterns",
        "policy_sequence_pattern_occurrences",
        "procedural_skill_candidates"
      ]));
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
      migrated.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("migrates schema v10 by adding Policy sequence pattern and SkillCandidate tables", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-v10-policy-sequence-migration-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      seeded.db.exec(`
        DROP TABLE procedural_skill_candidates;
        DROP TABLE policy_sequence_pattern_occurrences;
        DROP TABLE policy_sequence_patterns;
        DELETE FROM schema_migrations;
        INSERT INTO schema_migrations (id, version, applied_at, checksum)
        VALUES ('010_episode_policy_projections', 10, '2026-08-20T00:00:00.000Z', 'v10');
      `);
      seeded.close();

      const migrated = new MemoryDb({ path: dbPath });
      const tables = (migrated.db.prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table'`
      ).all() as Array<{ name: string }>).map((table) => table.name);
      expect(migrated.schemaVersion()).toEqual({
        version: SCHEMA_VERSION,
        lastMigrationId: SCHEMA_MIGRATION_ID
      });
      expect(tables).toEqual(expect.arrayContaining([
        "policy_sequence_patterns",
        "policy_sequence_pattern_occurrences",
        "procedural_skill_candidates"
      ]));
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
      migrated.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("migrates schema v11 by adding Episode Capability discovery and pattern routing fields", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-v11-episode-capability-migration-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      const repos = new Repositories(seeded.db);
      repos.memories.insert(schemaTraceMemory("v11-trace", "already migrated trace", false));
      seeded.db.exec(`
        DROP TABLE episode_capability_affinities;
        DROP TABLE episode_capability_signatures;
        DROP TABLE procedural_skill_candidates;
        DROP TABLE policy_sequence_pattern_occurrences;
        DROP TABLE policy_sequence_patterns;
        CREATE TABLE policy_sequence_patterns (
          id TEXT PRIMARY KEY,
          namespace_id TEXT NOT NULL,
          schema_version TEXT NOT NULL,
          algorithm_version TEXT NOT NULL,
          sequence_hash TEXT NOT NULL,
          policy_keys_json TEXT NOT NULL CHECK (json_valid(policy_keys_json)),
          lifecycle_status TEXT NOT NULL,
          occurrence_count INTEGER NOT NULL DEFAULT 0,
          distinct_episode_count INTEGER NOT NULL DEFAULT 0,
          distinct_support_episode_count INTEGER NOT NULL DEFAULT 0,
          distinct_counterexample_episode_count INTEGER NOT NULL DEFAULT 0,
          distinct_uncertain_episode_count INTEGER NOT NULL DEFAULT 0,
          is_closed INTEGER NOT NULL DEFAULT 1,
          is_maximal INTEGER NOT NULL DEFAULT 1,
          membership_version TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          UNIQUE (namespace_id, algorithm_version, sequence_hash)
        );
        DELETE FROM schema_migrations;
        INSERT INTO schema_migrations (id, version, applied_at, checksum)
        VALUES ('011_policy_sequence_patterns', 11, '2026-08-20T00:00:00.000Z', 'v11');
        CREATE TRIGGER reject_v11_memory_rewrite
        BEFORE UPDATE ON memories
        BEGIN
          SELECT RAISE(ABORT, 'v11 memory rows must not be rewritten');
        END;
      `);
      seeded.close();

      const migrated = new MemoryDb({ path: dbPath });
      const tables = (migrated.db.prepare(
        `SELECT name FROM sqlite_master WHERE type = 'table'`
      ).all() as Array<{ name: string }>).map((table) => table.name);
      const patternColumns = (migrated.db.prepare(
        `PRAGMA table_info(policy_sequence_patterns)`
      ).all() as Array<{ name: string }>).map((column) => column.name);
      expect(migrated.schemaVersion()).toEqual({
        version: SCHEMA_VERSION,
        lastMigrationId: SCHEMA_MIGRATION_ID
      });
      expect(tables).toEqual(expect.arrayContaining([
        "episode_capability_signatures",
        "episode_capability_affinities",
        "policy_sequence_pattern_occurrences",
        "procedural_skill_candidates"
      ]));
      expect(patternColumns).toEqual(expect.arrayContaining([
        "capability_type",
        "episode_family_id"
      ]));
      expect(migrated.db.prepare(`SELECT tags_json FROM memories WHERE id = 'v11-trace'`).get())
        .toEqual({ tags_json: JSON.stringify(schemaTraceMemory("v11-trace", "already migrated trace", false).tags) });
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
      migrated.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("migrates schema v12 by adding the incremental Policy sequence topology index", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-v12-policy-sequence-edges-migration-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      seeded.db.exec(`
        DROP TABLE policy_sequence_pattern_edges;
        DELETE FROM schema_migrations;
        INSERT INTO schema_migrations (id, version, applied_at, checksum)
        VALUES ('012_episode_capability_discovery', 12, '2026-08-20T00:00:00.000Z', 'v12');
      `);
      seeded.close();

      const migrated = new MemoryDb({ path: dbPath });
      expect(migrated.schemaVersion()).toEqual({
        version: SCHEMA_VERSION,
        lastMigrationId: SCHEMA_MIGRATION_ID
      });
      expect(migrated.db.prepare(
        `SELECT name FROM sqlite_master
         WHERE type = 'table' AND name = 'policy_sequence_pattern_edges'`
      ).get()).toEqual({ name: "policy_sequence_pattern_edges" });
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
      migrated.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("migrates v3 trace memories into explicit processing states without losing search data or vectors", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-v3-processing-migration-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      const repos = new Repositories(seeded.db);
      repos.memories.insert(schemaTraceMemory("legacy-ready", "legacy ready summary", true));
      repos.memories.insert(schemaTraceMemory("legacy-embedding", "legacy searchable summary", false));
      repos.memories.insert(schemaTraceMemory("legacy-summary", "摘要排队中", false));
      repos.memories.insert(schemaTraceMemory("legacy-failed", "legacy failed summary", false));
      repos.runtime.enqueueJob({
        id: "legacy-failed-job",
        jobType: "import_summary",
        status: "queued",
        dedupeKey: "import_summary:legacy-failed",
        userId: "old-user",
        targetMemoryId: "legacy-failed",
        payload: {},
        attempts: 3,
        maxAttempts: 3,
        createdAt: "2026-01-01T00:00:00.000Z",
        updatedAt: "2026-01-01T00:00:00.000Z"
      });
      seeded.db.prepare(`
        UPDATE evolution_jobs
        SET status = 'dead_letter', last_error = 'legacy provider failed'
        WHERE id = 'legacy-failed-job'
      `).run();
      seeded.db.exec(`
        DROP TABLE memory_processing_state;
        DELETE FROM schema_migrations;
        INSERT INTO schema_migrations (id, version, applied_at, checksum)
        VALUES ('003_runtime_schema', 3, '2026-01-01T00:00:00.000Z', 'v3');
      `);
      seeded.close();

      const migrated = new MemoryDb({ path: dbPath });
      const processingRows = migrated.db.prepare(`
        SELECT memory_id, state, stage, error_message
        FROM memory_processing_state
        ORDER BY memory_id
      `).all() as Array<{ memory_id: string; state: string; stage: string | null; error_message: string | null }>;
      expect(processingRows).toEqual([
        {
          memory_id: "legacy-embedding",
          state: "embedding_pending",
          stage: "embedding",
          error_message: null
        },
        {
          memory_id: "legacy-failed",
          state: "failed",
          stage: "summary",
          error_message: "legacy provider failed"
        },
        {
          memory_id: "legacy-ready",
          state: "ready",
          stage: null,
          error_message: null
        },
        {
          memory_id: "legacy-summary",
          state: "summary_pending",
          stage: "summary",
          error_message: null
        }
      ]);
      expect(migrated.db.prepare(`SELECT COUNT(*) AS count FROM memories`).get()).toEqual({ count: 4 });
      expect(migrated.db.prepare(`
        SELECT embedding_dim
        FROM memory_vector_entries
        WHERE memory_id = 'legacy-ready' AND vector_field = 'vec_summary'
      `).get()).toEqual({ embedding_dim: 3 });
      expect(migrated.db.prepare(`SELECT COUNT(*) AS count FROM memories_fts`).get()).toEqual({ count: 4 });
      const tags = migrated.db.prepare(`SELECT tags_json FROM memories WHERE id = 'legacy-embedding'`).get() as {
        tags_json: string;
      };
      expect(JSON.parse(tags.tags_json)).toEqual(["agent-source", "hermes", "legacy-user-tag"]);
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
      migrated.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects an unknown schema without changing user data", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-incompatible-schema-"));
    try {
      const dbPath = join(root, "memory.sqlite");
      const incompatible = new Database(dbPath);
      incompatible.exec(`
        CREATE TABLE schema_migrations (
          id TEXT PRIMARY KEY,
          version INTEGER NOT NULL,
          applied_at TEXT NOT NULL,
          checksum TEXT NOT NULL
        );
        INSERT INTO schema_migrations VALUES ('001_runtime_schema', 1, '2026-01-01', 'old');
        CREATE TABLE memories (id TEXT PRIMARY KEY);
        INSERT INTO memories VALUES ('old-memory');
      `);
      incompatible.close();

      expect(() => new MemoryDb({ path: dbPath })).toThrow(/left unchanged/);
      const verified = new Database(dbPath);
      expect(verified.prepare(`SELECT id FROM memories`).get()).toEqual({ id: "old-memory" });
      expect(verified.prepare(`SELECT version FROM schema_migrations`).get()).toEqual({ version: 1 });
      verified.close();
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("keeps vectors intact when an unknown schema is rejected", () => {
    const root = mkdtempSync(join(tmpdir(), "mindock-repo-incompatible-vec-schema-"));
    const dbPath = join(root, "memory.sqlite");
    try {
      const seeded = new MemoryDb({ path: dbPath });
      const repos = new Repositories(seeded.db);
      repos.memories.insert(schemaVectorMemory());
      expect(sqliteNames(seeded, "memory_vec_2%")).toEqual(expect.arrayContaining([
        "memory_vec_2",
        "memory_vec_2_chunks",
        "memory_vec_2_rowids"
      ]));
      seeded.db.prepare(
        `UPDATE schema_migrations SET version = 1, checksum = 'incompatible'`
      ).run();
      seeded.close();

      expect(() => new MemoryDb({ path: dbPath })).toThrow(/left unchanged/);
      const verified = new Database(dbPath);
      expect(verified.prepare(`SELECT COUNT(*) AS count FROM memories`).get()).toEqual({ count: 1 });
      expect(verified.prepare(`SELECT COUNT(*) AS count FROM memory_vector_entries`).get())
        .toEqual({ count: 1 });
      const names = (verified.prepare(
        `SELECT name FROM sqlite_master WHERE name LIKE 'memory_vec_2%' ORDER BY name`
      ).all() as Array<{ name: string }>).map((row) => row.name);
      expect(names).toEqual(expect.arrayContaining(["memory_vec_2", "memory_vec_2_chunks", "memory_vec_2_rowids"]));
      verified.close();
      expect(existsSync(`${dbPath}.pre-v${SCHEMA_VERSION}.bak`)).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

});

function schemaVectorMemory(): MemoryRow {
  const at = "2026-01-01T00:00:00.000Z";
  return {
    id: "old-vector-memory",
    timeline: at,
    userId: "old-user",
    memoryType: "LongTermMemory",
    status: "activated",
    visibility: "private",
    memoryKey: "policy:old-vector-memory",
    memoryValue: "old vector memory",
    tags: [],
    info: {},
    properties: {
      internal_info: {
        memory_layer: "L2",
        memory_kind: "policy",
        policy: {
          status: "active",
          vec: [1, 0],
          embedding_model: "old-model"
        }
      }
    },
    memoryLayer: "L2",
    contentHash: "old-vector-memory-hash",
    version: 1,
    createdAt: at,
    updatedAt: at,
    deletedAt: null
  };
}

function schemaTraceMemory(id: string, summary: string, withVector: boolean): MemoryRow {
  const at = "2026-01-01T00:00:00.000Z";
  return {
    id,
    timeline: at,
    userId: "old-user",
    agentId: "hermes",
    memoryType: "LongTermMemory",
    status: "activated",
    visibility: "private",
    memoryKey: `memory.add:agent-source:hermes:${id}`,
    memoryValue: `Summary: ${summary}\n\nUser:\nlegacy searchable content`,
    tags: ["agent-source", "hermes", "legacy-user-tag", "摘要总结中", "索引建立中"],
    info: {
      summary,
      source: "hermes",
      import_pipeline: { status: "indexing" }
    },
    properties: {
      internal_info: {
        memory_layer: "L1",
        memory_kind: "trace",
        plugin_algorithm: "memory.add.import_async.v2",
        import_pipeline: { status: "indexing" },
        trace: {
          summary,
          user_text: "legacy searchable content",
          agent_text: "legacy answer",
          tool_calls: [],
          ...(withVector
            ? {
              vec_summary: [1, 0, 0],
              embedding_model: "legacy-embedding-model"
            }
            : {})
        }
      }
    },
    memoryLayer: "L1",
    contentHash: `${id}-hash`,
    version: 1,
    createdAt: at,
    updatedAt: at,
    deletedAt: null
  };
}

function sqliteNames(db: MemoryDb, pattern: string): string[] {
  return (db.db.prepare(
    `SELECT name FROM sqlite_master WHERE name LIKE ? ORDER BY name`
  ).all(pattern) as Array<{ name: string }>).map((row) => row.name);
}
