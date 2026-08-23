import type Database from "better-sqlite3";

export const SCHEMA_VERSION = 13;
export const SCHEMA_MIGRATION_ID = "013_incremental_policy_sequence_refresh";
const API_LOG_SOURCE_AGENT_MIGRATION_FROM_VERSION = 2;
const MEMORY_PROCESSING_STATE_MIGRATION_VERSION = 4;
const PROCESSING_TAGS = new Set([
  "摘要排队中",
  "摘要整理中",
  "摘要总结中",
  "建立索引中",
  "索引建立中",
  "索引已建立",
  "处理失败"
]);

const statements = [
  `CREATE TABLE IF NOT EXISTS schema_migrations (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL,
    checksum TEXT NOT NULL
  )`,

  `CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    timeline TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT,
    session_id TEXT,
    agent_id TEXT,
    app_id TEXT,
    memory_type TEXT NOT NULL DEFAULT 'LongTermMemory',
    status TEXT NOT NULL DEFAULT 'activated'
      CHECK (status IN ('activated', 'resolving', 'archived', 'deleted')),
    visibility TEXT NOT NULL DEFAULT 'private',
    memory_key TEXT,
    memory_value TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tags_json)),
    info_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(info_json)),
    properties_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(properties_json)),
    memory_layer TEXT NOT NULL CHECK (memory_layer IN ('L1', 'L2', 'L3', 'Skill')),
    content_hash TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  )`,

  `CREATE INDEX IF NOT EXISTS idx_memories_layer_status_updated
    ON memories (memory_layer, status, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_layer_status_created
    ON memories (memory_layer, status, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_conversation_updated
    ON memories (conversation_id, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_session_layer
    ON memories (session_id, memory_layer, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_agent_app
    ON memories (agent_id, app_id, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_content_hash_layer
    ON memories (content_hash, memory_layer)`,
  `CREATE INDEX IF NOT EXISTS idx_memories_key_layer
    ON memories (memory_key, memory_layer)`,

  `CREATE TABLE IF NOT EXISTS user_memories (
    id TEXT PRIMARY KEY,
    source_turn_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    memory_types_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(memory_types_json)),
    content TEXT NOT NULL,
    normalized_user_text_hash TEXT NOT NULL,
    source_turn_refs_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(source_turn_refs_json)),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    replaces_memory_id TEXT,
    replaced_by_memory_id TEXT,
    archived_at TEXT,
    archive_reason TEXT,
    embedding_json TEXT CHECK (embedding_json IS NULL OR json_valid(embedding_json)),
    embedding_model TEXT,
    embedding_provider TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
  )`,
  `CREATE INDEX IF NOT EXISTS idx_user_memories_user_status_updated
    ON user_memories (user_id, status, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_user_memories_source_turn
    ON user_memories (source_turn_id)`,
  `CREATE INDEX IF NOT EXISTS idx_user_memories_exact_text
    ON user_memories (user_id, normalized_user_text_hash, status, updated_at DESC)`,
  `CREATE VIRTUAL TABLE IF NOT EXISTS user_memories_fts USING fts5 (
    id UNINDEXED,
    content,
    memory_types,
    tokenize='unicode61'
  )`,

  `CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5 (
    id UNINDEXED,
    identifier,
    memory_value,
    tags,
    tokenize='unicode61'
  )`,

  `CREATE TABLE IF NOT EXISTS memory_vector_entries (
    id INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    vector_field TEXT NOT NULL CHECK (vector_field IN ('vec_summary', 'vec_action', 'vec', 'vec_goal', 'vec_policy')),
    embedding_model TEXT,
    embedding_provider TEXT,
    embedding_dim INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (memory_id, vector_field)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_memory_vector_entries_field_updated
    ON memory_vector_entries (vector_field, updated_at DESC, id DESC)`,

  `CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    source TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    profile_label TEXT,
    workspace_id TEXT,
    workspace_path TEXT,
    host_session_key TEXT,
    conversation_id TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'processing', 'closed')),
    meta_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(meta_json)),
    opened_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    closed_at TEXT,
    updated_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
    ON sessions (user_id, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_sessions_host_key
    ON sessions (host_session_key)`,
  `CREATE INDEX IF NOT EXISTS idx_sessions_host_scope
    ON sessions (user_id, source, profile_id, host_session_key, status)`,

  `CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    project_id TEXT,
    conversation_id TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'processing', 'closed')),
    title TEXT,
    summary TEXT,
    l1_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(l1_memory_ids_json)),
    raw_turn_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(raw_turn_ids_json)),
    feedback_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(feedback_ids_json)),
    decision_repair_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(decision_repair_ids_json)),
    l2_policy_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(l2_policy_ids_json)),
    l3_world_model_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(l3_world_model_ids_json)),
    skill_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(skill_memory_ids_json)),
    turn_count INTEGER NOT NULL DEFAULT 0,
    r_task REAL,
    reward_detail_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(reward_detail_json)),
    pipeline_run_id TEXT,
    pipeline_status TEXT NOT NULL DEFAULT 'idle' CHECK (pipeline_status IN ('idle', 'running', 'succeeded', 'failed')),
    pipeline_error TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(meta_json)),
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    updated_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_episodes_session_updated
    ON episodes (session_id, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_episodes_project_updated
    ON episodes (project_id, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_episodes_pipeline
    ON episodes (pipeline_status, updated_at DESC)`,

  `CREATE TABLE IF NOT EXISTS raw_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT,
    user_text TEXT,
    assistant_text TEXT,
    reasoning_summary TEXT,
    tool_calls_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tool_calls_json)),
    tool_results_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tool_results_json)),
    source_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(source_memory_ids_json)),
    usage_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(usage_json)),
    message_payload_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(message_payload_json)),
    status TEXT NOT NULL DEFAULT 'succeeded',
    redacted_at TEXT,
    deleted_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, turn_id)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_raw_turns_episode_created
    ON raw_turns (episode_id, created_at ASC)`,

  `CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    conversation_id TEXT,
    session_id TEXT,
    episode_id TEXT,
    l1_memory_id TEXT,
    raw_turn_id TEXT,
    channel TEXT NOT NULL CHECK (channel IN ('explicit', 'implicit')),
    polarity TEXT NOT NULL CHECK (polarity IN ('positive', 'negative', 'neutral')),
    magnitude REAL NOT NULL DEFAULT 1,
    rationale TEXT,
    raw_payload_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(raw_payload_json)),
    context_hash TEXT,
    created_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_feedback_session_created
    ON feedback (session_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_feedback_memory
    ON feedback (l1_memory_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_feedback_user_created
    ON feedback (user_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_feedback_episode_created
    ON feedback (episode_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_feedback_raw_turn_created
    ON feedback (raw_turn_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_feedback_context
    ON feedback (user_id, project_id, context_hash, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS decision_repairs (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    episode_id TEXT,
    raw_turn_id TEXT,
    user_id TEXT NOT NULL,
    project_id TEXT,
    context_hash TEXT,
    issue TEXT NOT NULL,
    suggestion TEXT NOT NULL,
    preference TEXT,
    anti_pattern TEXT,
    high_value_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(high_value_memory_ids_json)),
    low_value_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(low_value_memory_ids_json)),
    attached_policy_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(attached_policy_memory_ids_json)),
    feedback_id TEXT,
    validated INTEGER NOT NULL DEFAULT 0 CHECK (validated IN (0, 1)),
    source_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(source_json)),
    meta_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(meta_json)),
    created_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_decision_repairs_context
    ON decision_repairs (user_id, project_id, context_hash)`,
  `CREATE INDEX IF NOT EXISTS idx_decision_repairs_episode
    ON decision_repairs (episode_id, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS l2_candidate_pool (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    source_memory_id TEXT NOT NULL,
    candidate_key TEXT NOT NULL,
    candidate_value TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'promoted', 'rejected')),
    evidence_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(evidence_json)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
  )`,
  `CREATE INDEX IF NOT EXISTS idx_l2_candidate_status
    ON l2_candidate_pool (user_id, status, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_l2_candidate_pending_expiry
    ON l2_candidate_pool (user_id, candidate_key, status, expires_at)`,

  `CREATE TABLE IF NOT EXISTS trace_policy_links (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    l1_memory_id TEXT NOT NULL,
    l2_memory_id TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'supports',
    strength REAL NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    UNIQUE (l1_memory_id, l2_memory_id, relation)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_trace_policy_links_l1
    ON trace_policy_links (user_id, l1_memory_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_trace_policy_links_l2
    ON trace_policy_links (user_id, l2_memory_id, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS episode_procedural_paths (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    reconstruction_algorithm_version TEXT NOT NULL,
    reconstruction_model TEXT,
    source_snapshot_hash TEXT NOT NULL,
    path_hash TEXT NOT NULL UNIQUE,
    terminal_reward REAL,
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    deactivated_at TEXT
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_episode_procedural_paths_active
    ON episode_procedural_paths (episode_id) WHERE status = 'active'`,
  `CREATE INDEX IF NOT EXISTS idx_episode_procedural_paths_episode_created
    ON episode_procedural_paths (episode_id, created_at DESC, id DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_episode_procedural_paths_namespace_status
    ON episode_procedural_paths (namespace_id, status, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS procedural_span_occurrences (
    id TEXT PRIMARY KEY,
    path_id TEXT NOT NULL REFERENCES episode_procedural_paths(id) ON DELETE CASCADE,
    path_hash TEXT NOT NULL,
    span_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    projection_version TEXT NOT NULL,
    span_index INTEGER NOT NULL,
    local_goal TEXT NOT NULL,
    entry_condition TEXT NOT NULL,
    exit_condition TEXT NOT NULL,
    termination_status TEXT NOT NULL
      CHECK (termination_status IN ('success', 'failure', 'blocked', 'abandoned')),
    raw_turn_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(raw_turn_ids_json)),
    step_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(step_ids_json)),
    pre_state_id TEXT NOT NULL,
    post_state_id TEXT NOT NULL,
    goal_text TEXT NOT NULL,
    condition_text TEXT NOT NULL,
    procedure_text TEXT NOT NULL,
    effect_text TEXT NOT NULL,
    structure_signature TEXT NOT NULL,
    span_json TEXT NOT NULL CHECK (json_valid(span_json)),
    reconstruction_algorithm_version TEXT NOT NULL,
    reconstruction_model TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (path_id, span_id),
    UNIQUE (path_id, span_index)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_occurrences_path
    ON procedural_span_occurrences (path_id, span_index ASC)`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_occurrences_namespace_episode
    ON procedural_span_occurrences (namespace_id, episode_id, span_index ASC)`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_occurrences_span
    ON procedural_span_occurrences (span_id, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS procedural_span_occurrence_embeddings (
    id TEXT PRIMARY KEY,
    occurrence_id TEXT NOT NULL REFERENCES procedural_span_occurrences(id) ON DELETE CASCADE,
    namespace_id TEXT NOT NULL,
    projection_version TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
    goal_vector_json TEXT NOT NULL CHECK (json_valid(goal_vector_json)),
    procedure_vector_json TEXT NOT NULL CHECK (json_valid(procedure_vector_json)),
    effect_vector_json TEXT NOT NULL CHECK (json_valid(effect_vector_json)),
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (
      occurrence_id, projection_version, embedding_version,
      source_hash, embedding_provider, embedding_model
    )
  )`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_embeddings_occurrence
    ON procedural_span_occurrence_embeddings (occurrence_id, updated_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_embeddings_namespace_model
    ON procedural_span_occurrence_embeddings (
      namespace_id, embedding_version, embedding_provider, embedding_model, updated_at DESC
    )`,

  `CREATE TABLE IF NOT EXISTS episode_span_credit_runs (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    path_id TEXT NOT NULL REFERENCES episode_procedural_paths(id) ON DELETE CASCADE,
    path_hash TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    reward_hash TEXT NOT NULL,
    goal_achievement REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    scorer_model TEXT,
    content_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    deactivated_at TEXT,
    UNIQUE (path_id, reward_hash, algorithm_version)
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_episode_span_credit_runs_active
    ON episode_span_credit_runs (episode_id) WHERE status = 'active'`,
  `CREATE INDEX IF NOT EXISTS idx_episode_span_credit_runs_path_created
    ON episode_span_credit_runs (path_id, created_at DESC, id DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_episode_span_credit_runs_namespace_status
    ON episode_span_credit_runs (namespace_id, status, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS procedural_span_credits (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES episode_span_credit_runs(id) ON DELETE CASCADE,
    occurrence_id TEXT NOT NULL REFERENCES procedural_span_occurrences(id) ON DELETE CASCADE,
    span_id TEXT NOT NULL,
    span_index INTEGER NOT NULL,
    episode_id TEXT NOT NULL,
    path_id TEXT NOT NULL,
    path_hash TEXT NOT NULL,
    reward_hash TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    pre_state_id TEXT NOT NULL,
    post_state_id TEXT NOT NULL,
    goal_credit REAL NOT NULL CHECK (goal_credit >= -1 AND goal_credit <= 1),
    process_quality REAL NOT NULL CHECK (process_quality >= -1 AND process_quality <= 1),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    credit_score REAL NOT NULL CHECK (credit_score >= -1 AND credit_score <= 1),
    evidence_role TEXT NOT NULL
      CHECK (evidence_role IN ('support', 'counterexample', 'neutral', 'uncertain')),
    evidence_refs_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(evidence_refs_json)),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, occurrence_id),
    UNIQUE (run_id, span_index)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_credits_occurrence
    ON procedural_span_credits (occurrence_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_credits_episode_role
    ON procedural_span_credits (episode_id, evidence_role, span_index ASC)`,

  `CREATE TABLE IF NOT EXISTS procedural_span_clusters (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('forming', 'ready', 'promoted', 'stale')),
    member_count INTEGER NOT NULL,
    distinct_episode_count INTEGER NOT NULL,
    distinct_support_episode_count INTEGER NOT NULL,
    membership_version TEXT NOT NULL,
    anchor_occurrence_id TEXT NOT NULL,
    active_policy_version_id TEXT,
    cluster_basis_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(cluster_basis_json)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_clusters_scope_status
    ON procedural_span_clusters (namespace_id, algorithm_version, status, updated_at DESC)`,

  `CREATE TABLE IF NOT EXISTS procedural_span_cluster_members (
    cluster_id TEXT NOT NULL REFERENCES procedural_span_clusters(id) ON DELETE CASCADE,
    occurrence_id TEXT NOT NULL REFERENCES procedural_span_occurrences(id) ON DELETE CASCADE,
    algorithm_version TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    evidence_role TEXT NOT NULL CHECK (evidence_role IN ('support', 'counterexample')),
    similarity REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (cluster_id, occurrence_id)
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_procedural_span_cluster_member_algorithm
    ON procedural_span_cluster_members (occurrence_id, algorithm_version)`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_cluster_members_cluster
    ON procedural_span_cluster_members (cluster_id, evidence_role, episode_id)`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_span_cluster_members_episode_algorithm
    ON procedural_span_cluster_members (episode_id, algorithm_version, occurrence_id)`,

  `CREATE TABLE IF NOT EXISTS procedural_policy_versions (
    id TEXT PRIMARY KEY,
    policy_key TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL REFERENCES procedural_span_clusters(id) ON DELETE CASCADE,
    cluster_membership_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    induction_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    title TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_hash TEXT NOT NULL,
    evidence_occurrence_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(evidence_occurrence_ids_json)),
    compiler_model TEXT,
    l2_memory_id TEXT,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    deactivated_at TEXT,
    UNIQUE (cluster_id, cluster_membership_version, induction_version)
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_procedural_policy_versions_active
    ON procedural_policy_versions (cluster_id) WHERE status = 'active'`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_policy_versions_scope_status
    ON procedural_policy_versions (namespace_id, status, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS procedural_policy_occurrences (
    id TEXT PRIMARY KEY,
    policy_version_id TEXT NOT NULL REFERENCES procedural_policy_versions(id) ON DELETE CASCADE,
    policy_key TEXT NOT NULL,
    occurrence_id TEXT NOT NULL REFERENCES procedural_span_occurrences(id) ON DELETE CASCADE,
    path_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    cluster_membership_version TEXT NOT NULL,
    evidence_role TEXT NOT NULL CHECK (evidence_role IN ('support', 'counterexample')),
    match_confidence REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
    created_at TEXT NOT NULL,
    superseded_at TEXT,
    UNIQUE (policy_version_id, occurrence_id)
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_procedural_policy_occurrence_active
    ON procedural_policy_occurrences (policy_key, occurrence_id) WHERE status = 'active'`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_policy_occurrences_episode
    ON procedural_policy_occurrences (episode_id, status, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_policy_occurrences_policy
    ON procedural_policy_occurrences (policy_version_id, status, occurrence_id)`,

  `CREATE TABLE IF NOT EXISTS episode_policy_projections (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    path_id TEXT NOT NULL REFERENCES episode_procedural_paths(id) ON DELETE CASCADE,
    path_hash TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    assignment_snapshot_hash TEXT NOT NULL,
    projection_hash TEXT NOT NULL UNIQUE,
    mapped_span_count INTEGER NOT NULL CHECK (mapped_span_count >= 0),
    unmapped_span_count INTEGER NOT NULL CHECK (unmapped_span_count >= 0),
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    deactivated_at TEXT
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_episode_policy_projections_active
    ON episode_policy_projections (episode_id) WHERE status = 'active'`,
  `CREATE INDEX IF NOT EXISTS idx_episode_policy_projections_path_created
    ON episode_policy_projections (path_id, created_at DESC, id DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_episode_policy_projections_namespace_status
    ON episode_policy_projections (namespace_id, status, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS episode_policy_projection_nodes (
    projection_id TEXT NOT NULL REFERENCES episode_policy_projections(id) ON DELETE CASCADE,
    node_index INTEGER NOT NULL CHECK (node_index >= 0),
    node_kind TEXT NOT NULL CHECK (node_kind IN ('policy', 'unmapped')),
    occurrence_id TEXT NOT NULL REFERENCES procedural_span_occurrences(id) ON DELETE CASCADE,
    span_id TEXT NOT NULL,
    span_index INTEGER NOT NULL CHECK (span_index >= 0),
    pre_state_id TEXT NOT NULL,
    post_state_id TEXT NOT NULL,
    policy_version_id TEXT,
    policy_key TEXT,
    cluster_id TEXT,
    cluster_membership_version TEXT,
    evidence_role TEXT CHECK (evidence_role IS NULL OR evidence_role IN ('support', 'counterexample')),
    match_confidence REAL CHECK (match_confidence IS NULL OR (match_confidence >= 0 AND match_confidence <= 1)),
    unmapped_reason TEXT CHECK (unmapped_reason IS NULL OR unmapped_reason IN (
      'no_cluster_assignment', 'cluster_forming', 'cluster_ready_policy_pending',
      'cluster_stale', 'active_policy_occurrence_missing'
    )),
    node_json TEXT NOT NULL CHECK (json_valid(node_json)),
    PRIMARY KEY (projection_id, node_index),
    UNIQUE (projection_id, occurrence_id)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_episode_policy_projection_nodes_policy
    ON episode_policy_projection_nodes (policy_version_id, projection_id, node_index)`,
  `CREATE INDEX IF NOT EXISTS idx_episode_policy_projection_nodes_policy_key
    ON episode_policy_projection_nodes (policy_key, projection_id, node_index)`,
  `CREATE INDEX IF NOT EXISTS idx_episode_policy_projection_nodes_occurrence
    ON episode_policy_projection_nodes (occurrence_id, projection_id)`,

  `CREATE TABLE IF NOT EXISTS episode_capability_signatures (
    id TEXT PRIMARY KEY,
    projection_id TEXT NOT NULL REFERENCES episode_policy_projections(id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    path_id TEXT NOT NULL REFERENCES episode_procedural_paths(id) ON DELETE CASCADE,
    namespace_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    signature_hash TEXT NOT NULL UNIQUE,
    family_id TEXT NOT NULL,
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
    goal_vector_json TEXT NOT NULL CHECK (json_valid(goal_vector_json)),
    state_transition_vector_json TEXT NOT NULL CHECK (json_valid(state_transition_vector_json)),
    outcome_vector_json TEXT NOT NULL CHECK (json_valid(outcome_vector_json)),
    context_vector_json TEXT NOT NULL CHECK (json_valid(context_vector_json)),
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    deactivated_at TEXT,
    UNIQUE (projection_id, algorithm_version)
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_episode_capability_signatures_active
    ON episode_capability_signatures (episode_id) WHERE status = 'active'`,
  `CREATE INDEX IF NOT EXISTS idx_episode_capability_signatures_scope_family
    ON episode_capability_signatures (namespace_id, family_id, status, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS episode_capability_affinities (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    left_signature_id TEXT NOT NULL
      REFERENCES episode_capability_signatures(id) ON DELETE CASCADE,
    right_signature_id TEXT NOT NULL
      REFERENCES episode_capability_signatures(id) ON DELETE CASCADE,
    left_episode_id TEXT NOT NULL,
    right_episode_id TEXT NOT NULL,
    goal_similarity REAL NOT NULL,
    state_transition_similarity REAL NOT NULL,
    outcome_similarity REAL NOT NULL,
    context_similarity REAL NOT NULL,
    path_structure_similarity REAL NOT NULL,
    combined_similarity REAL NOT NULL,
    family_eligible INTEGER NOT NULL CHECK (family_eligible IN (0, 1)),
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL,
    UNIQUE (left_signature_id, right_signature_id, algorithm_version),
    CHECK (left_signature_id < right_signature_id)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_episode_capability_affinities_scope_eligible
    ON episode_capability_affinities
      (namespace_id, family_eligible, combined_similarity DESC, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS policy_sequence_patterns (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    sequence_hash TEXT NOT NULL,
    capability_type TEXT NOT NULL DEFAULT 'sub_skill'
      CHECK (capability_type IN ('task_skill', 'sub_skill')),
    episode_family_id TEXT,
    policy_keys_json TEXT NOT NULL CHECK (json_valid(policy_keys_json)),
    lifecycle_status TEXT NOT NULL
      CHECK (lifecycle_status IN ('forming', 'observed', 'ready', 'stale')),
    occurrence_count INTEGER NOT NULL DEFAULT 0 CHECK (occurrence_count >= 0),
    distinct_episode_count INTEGER NOT NULL DEFAULT 0 CHECK (distinct_episode_count >= 0),
    distinct_support_episode_count INTEGER NOT NULL DEFAULT 0
      CHECK (distinct_support_episode_count >= 0),
    distinct_counterexample_episode_count INTEGER NOT NULL DEFAULT 0
      CHECK (distinct_counterexample_episode_count >= 0),
    distinct_uncertain_episode_count INTEGER NOT NULL DEFAULT 0
      CHECK (distinct_uncertain_episode_count >= 0),
    is_closed INTEGER NOT NULL DEFAULT 1 CHECK (is_closed IN (0, 1)),
    is_maximal INTEGER NOT NULL DEFAULT 1 CHECK (is_maximal IN (0, 1)),
    membership_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (namespace_id, algorithm_version, sequence_hash)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_policy_sequence_patterns_scope_status
    ON policy_sequence_patterns (namespace_id, algorithm_version, lifecycle_status, updated_at DESC)`,

  `CREATE TABLE IF NOT EXISTS policy_sequence_pattern_edges (
    parent_pattern_id TEXT NOT NULL
      REFERENCES policy_sequence_patterns(id) ON DELETE CASCADE,
    child_pattern_id TEXT NOT NULL
      REFERENCES policy_sequence_patterns(id) ON DELETE CASCADE,
    PRIMARY KEY (parent_pattern_id, child_pattern_id),
    CHECK (parent_pattern_id != child_pattern_id)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_policy_sequence_pattern_edges_child
    ON policy_sequence_pattern_edges (child_pattern_id, parent_pattern_id)`,

  `CREATE TABLE IF NOT EXISTS policy_sequence_pattern_occurrences (
    id TEXT PRIMARY KEY,
    pattern_id TEXT NOT NULL REFERENCES policy_sequence_patterns(id) ON DELETE CASCADE,
    projection_id TEXT NOT NULL REFERENCES episode_policy_projections(id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL,
    path_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    start_node_index INTEGER NOT NULL CHECK (start_node_index >= 0),
    end_node_index INTEGER NOT NULL CHECK (end_node_index >= start_node_index),
    evidence_role TEXT NOT NULL
      CHECK (evidence_role IN ('support', 'counterexample', 'uncertain')),
    terminal_reward REAL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
    created_at TEXT NOT NULL,
    superseded_at TEXT,
    UNIQUE (pattern_id, projection_id, start_node_index, end_node_index)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_policy_sequence_occurrences_pattern_status
    ON policy_sequence_pattern_occurrences (pattern_id, status, episode_id, start_node_index)`,
  `CREATE INDEX IF NOT EXISTS idx_policy_sequence_occurrences_episode_status
    ON policy_sequence_pattern_occurrences (episode_id, status, projection_id)`,

  `CREATE TABLE IF NOT EXISTS procedural_skill_candidates (
    id TEXT PRIMARY KEY,
    candidate_key TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    pattern_id TEXT NOT NULL REFERENCES policy_sequence_patterns(id) ON DELETE CASCADE,
    pattern_membership_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    induction_version TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL CHECK (lifecycle_status IN ('observed', 'ready')),
    status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
    distinct_support_episode_count INTEGER NOT NULL
      CHECK (distinct_support_episode_count >= 0),
    distinct_counterexample_episode_count INTEGER NOT NULL
      CHECK (distinct_counterexample_episode_count >= 0),
    evidence_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    deactivated_at TEXT,
    UNIQUE (pattern_id, pattern_membership_version, induction_version)
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_procedural_skill_candidates_active
    ON procedural_skill_candidates (pattern_id) WHERE status = 'active'`,
  `CREATE INDEX IF NOT EXISTS idx_procedural_skill_candidates_scope_status
    ON procedural_skill_candidates (namespace_id, lifecycle_status, status, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS span_clusters (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('forming', 'ready', 'promoted', 'stale')),
    goal_threshold REAL NOT NULL,
    policy_threshold REAL NOT NULL,
    goal_centroid_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(goal_centroid_json)),
    policy_centroid_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(policy_centroid_json)),
    member_count INTEGER NOT NULL DEFAULT 0,
    distinct_source_count INTEGER NOT NULL DEFAULT 0,
    membership_version TEXT NOT NULL,
    promoted_policy_id TEXT,
    anchor_span_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_span_clusters_scope_status
    ON span_clusters (scope_id, algorithm_version, status, updated_at DESC)`,

  `CREATE TABLE IF NOT EXISTS span_cluster_members (
    cluster_id TEXT NOT NULL REFERENCES span_clusters(id) ON DELETE CASCADE,
    span_id TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    source_trace_id TEXT NOT NULL,
    goal_similarity REAL NOT NULL,
    policy_similarity REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (cluster_id, span_id)
  )`,
  `CREATE UNIQUE INDEX IF NOT EXISTS uq_span_cluster_member_active_algorithm
    ON span_cluster_members (span_id, algorithm_version)`,
  `CREATE INDEX IF NOT EXISTS idx_span_cluster_members_cluster
    ON span_cluster_members (cluster_id, span_id)`,

  `CREATE TABLE IF NOT EXISTS skill_trials (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    skill_memory_id TEXT NOT NULL,
    session_id TEXT,
    episode_id TEXT NOT NULL,
    l1_memory_id TEXT,
    raw_turn_id TEXT,
    turn_id TEXT,
    tool_call_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
      CHECK (status IN ('pending', 'pass', 'fail', 'unknown')),
    outcome TEXT NOT NULL DEFAULT 'unknown'
      CHECK (outcome IN ('unknown', 'success', 'failure', 'cancelled')),
    feedback_id TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
  )`,
  `CREATE INDEX IF NOT EXISTS idx_skill_trials_skill_created
    ON skill_trials (skill_memory_id, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_skill_trials_user_status
    ON skill_trials (user_id, status, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_skill_trials_episode_status
    ON skill_trials (episode_id, status, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_skill_trials_l1_status
    ON skill_trials (l1_memory_id, status, created_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_skill_trials_raw_status
    ON skill_trials (raw_turn_id, status, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS recall_events (
    id TEXT PRIMARY KEY,
    namespace_id TEXT,
    session_id TEXT,
    episode_id TEXT,
    turn_id TEXT,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    query_hash TEXT,
    layers_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(layers_json)),
    candidate_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(candidate_memory_ids_json)),
    injected_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(injected_memory_ids_json)),
    hit_memory_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(hit_memory_ids_json)),
    dropped_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(dropped_json)),
    outcome TEXT NOT NULL DEFAULT 'pending' CHECK (outcome IN ('pending', 'positive', 'negative', 'ignored')),
    request_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(request_json)),
    query_id TEXT,
    user_memory_candidate_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(user_memory_candidate_ids_json)),
    l1_candidate_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(l1_candidate_ids_json)),
    merged_source_turn_ids_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(merged_source_turn_ids_json)),
    member_memory_ids_by_source_turn_id_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(member_memory_ids_by_source_turn_id_json)),
    created_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_recall_events_session_created
    ON recall_events (session_id, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS api_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL CHECK (tool_name IN ('memory_add', 'memory_search', 'skill_generate', 'skill_evolve')),
    source_agent TEXT,
    input_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(input_json)),
    output_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(output_json)),
    duration_ms INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1 CHECK (success IN (0, 1)),
    called_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_api_logs_tool_time
    ON api_logs (tool_name, called_at DESC)`,
  `CREATE INDEX IF NOT EXISTS idx_api_logs_tool_source_time
    ON api_logs (tool_name, source_agent, called_at DESC)`,

  `CREATE TABLE IF NOT EXISTS memory_change_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    namespace_id TEXT,
    kind TEXT,
    op TEXT,
    entity_id TEXT,
    user_id TEXT NOT NULL,
    change_type TEXT NOT NULL,
    version INTEGER,
    before_json TEXT,
    after_json TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_memory_change_log_namespace_seq
    ON memory_change_log (namespace_id, seq)`,
  `CREATE INDEX IF NOT EXISTS idx_memory_change_log_created
    ON memory_change_log (created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS idempotency_keys (
    key TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL CHECK (json_valid(response_json)),
    created_at TEXT NOT NULL,
    expires_at TEXT
  )`,

  `CREATE TABLE IF NOT EXISTS evolution_jobs (
    id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
      CHECK (status IN ('queued', 'leased', 'succeeded', 'failed', 'dead_letter')),
    dedupe_key TEXT,
    user_id TEXT NOT NULL,
    session_id TEXT,
    episode_id TEXT,
    target_memory_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(payload_json)),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    leased_until TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_evolution_jobs_status_created
    ON evolution_jobs (status, created_at ASC)`,
  `CREATE INDEX IF NOT EXISTS idx_evolution_jobs_target
    ON evolution_jobs (target_memory_id, job_type)`,

  `CREATE TABLE IF NOT EXISTS embedding_retry_queue (
    id TEXT PRIMARY KEY,
    target_kind TEXT NOT NULL CHECK (target_kind IN ('trace', 'span', 'policy', 'world_model', 'skill')),
    target_id TEXT NOT NULL,
    vector_field TEXT NOT NULL CHECK (vector_field IN ('vec_summary', 'vec_action', 'vec', 'vec_goal', 'vec_policy')),
    source_text TEXT NOT NULL,
    embed_role TEXT NOT NULL DEFAULT 'document' CHECK (embed_role IN ('document', 'query')),
    status TEXT NOT NULL DEFAULT 'pending'
      CHECK (status IN ('pending', 'in_progress', 'failed', 'succeeded')),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 6,
    next_attempt_at INTEGER NOT NULL,
    claimed_by TEXT,
    lease_until INTEGER,
    last_error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (target_kind, target_id, vector_field)
  )`,
  `CREATE INDEX IF NOT EXISTS idx_embedding_retry_due
    ON embedding_retry_queue (status, next_attempt_at)`,
  `CREATE INDEX IF NOT EXISTS idx_embedding_retry_target
    ON embedding_retry_queue (target_kind, target_id)`,

  `CREATE TABLE IF NOT EXISTS memory_processing_state (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN (
      'summary_pending', 'summarizing', 'embedding_pending', 'embedding',
      'ready', 'ready_text_only', 'failed'
    )),
    stage TEXT CHECK (stage IN ('summary', 'embedding')),
    active_job_id TEXT REFERENCES evolution_jobs(id) ON DELETE SET NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    manual_retry_count INTEGER NOT NULL DEFAULT 0,
    retry_action TEXT NOT NULL DEFAULT 'retry'
      CHECK (retry_action IN ('retry', 'open_settings', 'none')),
    error_code TEXT,
    error_message TEXT,
    failed_at TEXT,
    updated_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_memory_processing_state_state_updated
    ON memory_processing_state (state, updated_at ASC)`,
  `CREATE INDEX IF NOT EXISTS idx_memory_processing_state_active_job
    ON memory_processing_state (active_job_id)`,

  `CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    episode_id TEXT,
    raw_turn_id TEXT,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    uri TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_artifacts_session_created
    ON artifacts (session_id, created_at DESC)`,

  `CREATE TABLE IF NOT EXISTS runtime_kv (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(value_json)),
    updated_at TEXT NOT NULL
  )`,

  `CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    actor_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(actor_json)),
    action TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(meta_json)),
    created_at TEXT NOT NULL
  )`,
  `CREATE INDEX IF NOT EXISTS idx_audit_logs_user_created
    ON audit_logs (user_id, created_at DESC)`
];

export function migrate(db: Database.Database): void {
  const now = new Date().toISOString();
  const checksum = String(statements.join("\n").length);
  const foreignKeys = Number(db.pragma("foreign_keys", { simple: true }) ?? 0);
  const hasMemories = tableExists(db, "memories");
  const version = currentSchemaVersion(db);

  if (hasMemories && version !== SCHEMA_VERSION && version !== 2 && version !== 3 && version !== 4 && version !== 5 && version !== 6 && version !== 7 && version !== 8 && version !== 9 && version !== 10 && version !== 11 && version !== 12) {
    throw new Error(
      `Unsupported memory database schema version ${version}; the database was left unchanged`
    );
  }
  if (version > SCHEMA_VERSION) {
    throw new Error(
      `Memory database schema version ${version} is newer than supported version ${SCHEMA_VERSION}`
    );
  }

  db.pragma("foreign_keys = OFF");
  try {
    db.transaction(() => {
      if (version === API_LOG_SOURCE_AGENT_MIGRATION_FROM_VERSION &&
          !columnExists(db, "api_logs", "source_agent")) {
        db.prepare(`ALTER TABLE api_logs ADD COLUMN source_agent TEXT`).run();
      }
      for (const statement of statements) {
        db.prepare(statement).run();
      }
      ensureEpisodeCapabilityDiscoverySchema(db);
      if (version < 13) backfillPolicySequencePatternEdges(db);
      ensureSpanEmbeddingVectorSchema(db);
      addColumnIfMissing(db, "recall_events", "query_id", "TEXT");
      addColumnIfMissing(db, "recall_events", "user_memory_candidate_ids_json", "TEXT NOT NULL DEFAULT '[]'");
      addColumnIfMissing(db, "recall_events", "l1_candidate_ids_json", "TEXT NOT NULL DEFAULT '[]'");
      addColumnIfMissing(db, "recall_events", "merged_source_turn_ids_json", "TEXT NOT NULL DEFAULT '[]'");
      addColumnIfMissing(db, "recall_events", "member_memory_ids_by_source_turn_id_json", "TEXT NOT NULL DEFAULT '{}'");
      db.prepare(
        `CREATE UNIQUE INDEX IF NOT EXISTS uq_evolution_jobs_active_dedupe
         ON evolution_jobs (dedupe_key)
         WHERE dedupe_key IS NOT NULL AND status IN ('queued', 'leased', 'failed')`
      ).run();

      if (hasMemories && version < MEMORY_PROCESSING_STATE_MIGRATION_VERSION) {
        backfillMemoryProcessingState(db, now);
        removeLegacyProcessingMetadata(db);
      }

      db.prepare(
        `INSERT INTO schema_migrations (id, version, applied_at, checksum)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           version = excluded.version,
           applied_at = excluded.applied_at,
           checksum = excluded.checksum`
      ).run(SCHEMA_MIGRATION_ID, SCHEMA_VERSION, now, checksum);
    })();
  } finally {
    db.pragma(`foreign_keys = ${foreignKeys ? "ON" : "OFF"}`);
  }
}

function addColumnIfMissing(
  db: Database.Database,
  table: string,
  column: string,
  definition: string
): void {
  if (!columnExists(db, table, column)) {
    db.prepare(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`).run();
  }
}

function ensureEpisodeCapabilityDiscoverySchema(db: Database.Database): void {
  if (!columnExists(db, "policy_sequence_patterns", "capability_type")) {
    db.prepare(
      `ALTER TABLE policy_sequence_patterns
       ADD COLUMN capability_type TEXT NOT NULL DEFAULT 'sub_skill'
       CHECK (capability_type IN ('task_skill', 'sub_skill'))`
    ).run();
  }
  if (!columnExists(db, "policy_sequence_patterns", "episode_family_id")) {
    db.prepare(
      `ALTER TABLE policy_sequence_patterns ADD COLUMN episode_family_id TEXT`
    ).run();
  }
}

function backfillPolicySequencePatternEdges(db: Database.Database): void {
  if (!tableExists(db, "policy_sequence_patterns") ||
      !tableExists(db, "policy_sequence_pattern_edges")) return;
  const rows = db.prepare(
    `SELECT id, namespace_id, algorithm_version, capability_type,
            episode_family_id, policy_keys_json
     FROM policy_sequence_patterns`
  ).all() as Array<{
    id: string;
    namespace_id: string;
    algorithm_version: string;
    capability_type: string;
    episode_family_id: string | null;
    policy_keys_json: string;
  }>;
  const patterns = rows.map((row) => ({
    ...row,
    policyKeys: parseStringArray(row.policy_keys_json)
  }));
  const idBySequence = new Map(patterns.map((pattern) => [
    policySequenceTopologyKey(pattern, pattern.policyKeys),
    pattern.id
  ]));
  const insert = db.prepare(
    `INSERT OR IGNORE INTO policy_sequence_pattern_edges (
       parent_pattern_id, child_pattern_id
     ) VALUES (?, ?)`
  );
  for (const child of patterns) {
    if (child.policyKeys.length <= 2) continue;
    for (const parentKeys of [
      child.policyKeys.slice(1),
      child.policyKeys.slice(0, -1)
    ]) {
      const parentId = idBySequence.get(policySequenceTopologyKey(child, parentKeys));
      if (parentId) insert.run(parentId, child.id);
    }
  }
}

function policySequenceTopologyKey(
  pattern: {
    namespace_id: string;
    algorithm_version: string;
    capability_type: string;
    episode_family_id: string | null;
  },
  policyKeys: readonly string[]
): string {
  return JSON.stringify([
    pattern.namespace_id,
    pattern.algorithm_version,
    pattern.capability_type,
    pattern.episode_family_id,
    policyKeys
  ]);
}

function parseStringArray(value: string): string[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) && parsed.every((item) => typeof item === "string")
      ? parsed
      : [];
  } catch {
    return [];
  }
}

function currentSchemaVersion(db: Database.Database): number {
  if (!tableExists(db, "schema_migrations")) return 0;
  return Number((db.prepare(
    `SELECT MAX(version) AS version FROM schema_migrations`
  ).get() as { version?: number } | undefined)?.version ?? 0);
}

function columnExists(db: Database.Database, table: string, column: string): boolean {
  if (!tableExists(db, table)) return false;
  return (db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>)
    .some((item) => item.name === column);
}

function ensureSpanEmbeddingVectorSchema(db: Database.Database): void {
  if (!tableSqlContains(db, "memory_vector_entries", "vec_policy")) {
    db.exec(`
      ALTER TABLE memory_vector_entries RENAME TO memory_vector_entries_legacy_span_vectors;
      CREATE TABLE memory_vector_entries (
        id INTEGER PRIMARY KEY,
        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        vector_field TEXT NOT NULL CHECK (vector_field IN ('vec_summary', 'vec_action', 'vec', 'vec_goal', 'vec_policy')),
        embedding_model TEXT,
        embedding_provider TEXT,
        embedding_dim INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (memory_id, vector_field)
      );
      INSERT INTO memory_vector_entries (
        id, memory_id, vector_field, embedding_model, embedding_provider, embedding_dim, updated_at
      )
      SELECT id, memory_id, vector_field, embedding_model, embedding_provider, embedding_dim, updated_at
      FROM memory_vector_entries_legacy_span_vectors;
      DROP TABLE memory_vector_entries_legacy_span_vectors;
    `);
  }
  db.prepare(
    `CREATE INDEX IF NOT EXISTS idx_memory_vector_entries_field_updated
     ON memory_vector_entries (vector_field, updated_at DESC, id DESC)`
  ).run();

  if (!tableSqlContains(db, "embedding_retry_queue", "'span'")) {
    db.exec(`
      ALTER TABLE embedding_retry_queue RENAME TO embedding_retry_queue_legacy_span_vectors;
      CREATE TABLE embedding_retry_queue (
        id TEXT PRIMARY KEY,
        target_kind TEXT NOT NULL CHECK (target_kind IN ('trace', 'span', 'policy', 'world_model', 'skill')),
        target_id TEXT NOT NULL,
        vector_field TEXT NOT NULL CHECK (vector_field IN ('vec_summary', 'vec_action', 'vec', 'vec_goal', 'vec_policy')),
        source_text TEXT NOT NULL,
        embed_role TEXT NOT NULL DEFAULT 'document' CHECK (embed_role IN ('document', 'query')),
        status TEXT NOT NULL DEFAULT 'pending'
          CHECK (status IN ('pending', 'in_progress', 'failed', 'succeeded')),
        attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 6,
        next_attempt_at INTEGER NOT NULL,
        claimed_by TEXT,
        lease_until INTEGER,
        last_error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        UNIQUE (target_kind, target_id, vector_field)
      );
      INSERT INTO embedding_retry_queue (
        id, target_kind, target_id, vector_field, source_text, embed_role,
        status, attempts, max_attempts, next_attempt_at, claimed_by,
        lease_until, last_error, created_at, updated_at
      )
      SELECT
        id, target_kind, target_id, vector_field, source_text, embed_role,
        status, attempts, max_attempts, next_attempt_at, claimed_by,
        lease_until, last_error, created_at, updated_at
      FROM embedding_retry_queue_legacy_span_vectors;
      DROP TABLE embedding_retry_queue_legacy_span_vectors;
    `);
  }
  db.prepare(
    `CREATE INDEX IF NOT EXISTS idx_embedding_retry_due
     ON embedding_retry_queue (status, next_attempt_at)`
  ).run();
  db.prepare(
    `CREATE INDEX IF NOT EXISTS idx_embedding_retry_target
     ON embedding_retry_queue (target_kind, target_id)`
  ).run();
}

function tableSqlContains(db: Database.Database, table: string, fragment: string): boolean {
  const row = db.prepare(
    `SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?`
  ).get(table) as { sql?: string } | undefined;
  return Boolean(row?.sql?.includes(fragment));
}

function backfillMemoryProcessingState(db: Database.Database, now: string): void {
  db.prepare(
    `INSERT INTO memory_processing_state (
       memory_id, state, stage, active_job_id, attempt_count, manual_retry_count,
       retry_action, error_code, error_message, failed_at, updated_at
     )
     SELECT
       memories.id,
       CASE
         WHEN EXISTS (
           SELECT 1 FROM memory_vector_entries
           WHERE memory_id = memories.id AND vector_field = 'vec_summary'
         ) THEN 'ready'
         WHEN COALESCE(latest_job.status, '') = 'dead_letter'
           OR COALESCE(latest_retry.status, '') = 'failed' THEN 'failed'
         WHEN COALESCE(latest_job.job_type, '') IN ('trace_summary', 'import_summary')
           THEN 'summary_pending'
         WHEN TRIM(COALESCE(
           json_extract(properties_json, '$.internal_info.trace.summary'),
           json_extract(info_json, '$.summary'),
           json_extract(properties_json, '$.internal_info.summary'),
           ''
         )) = '' THEN 'summary_pending'
         WHEN LOWER(TRIM(COALESCE(
           json_extract(properties_json, '$.internal_info.trace.summary'),
           json_extract(info_json, '$.summary'),
           json_extract(properties_json, '$.internal_info.summary'),
           ''
         ))) IN ('user', 'assistant', 'system', 'tool', 'developer', '摘要排队中', '摘要整理中', '摘要总结中')
           THEN 'summary_pending'
         ELSE 'embedding_pending'
       END,
       CASE
         WHEN EXISTS (
           SELECT 1 FROM memory_vector_entries
           WHERE memory_id = memories.id AND vector_field = 'vec_summary'
         ) THEN NULL
         WHEN COALESCE(latest_job.job_type, '') IN ('trace_summary', 'import_summary') THEN 'summary'
         WHEN COALESCE(latest_job.job_type, '') = 'embedding'
           OR COALESCE(latest_retry.status, '') = 'failed' THEN 'embedding'
         WHEN TRIM(COALESCE(
           json_extract(properties_json, '$.internal_info.trace.summary'),
           json_extract(info_json, '$.summary'),
           json_extract(properties_json, '$.internal_info.summary'),
           ''
         )) = '' THEN 'summary'
         WHEN LOWER(TRIM(COALESCE(
           json_extract(properties_json, '$.internal_info.trace.summary'),
           json_extract(info_json, '$.summary'),
           json_extract(properties_json, '$.internal_info.summary'),
           ''
         ))) IN ('user', 'assistant', 'system', 'tool', 'developer', '摘要排队中', '摘要整理中', '摘要总结中')
           THEN 'summary'
         ELSE 'embedding'
       END,
       NULL,
       MAX(COALESCE(latest_job.attempts, 0), COALESCE(latest_retry.attempts, 0)),
       0,
       'retry',
       CASE
         WHEN COALESCE(latest_job.status, '') = 'dead_letter' THEN 'processing_failed'
         WHEN COALESCE(latest_retry.status, '') = 'failed' THEN 'embedding_failed'
         ELSE NULL
       END,
       COALESCE(latest_job.last_error, latest_retry.last_error),
       CASE
         WHEN COALESCE(latest_job.status, '') = 'dead_letter' THEN latest_job.updated_at
         WHEN COALESCE(latest_retry.status, '') = 'failed'
           THEN datetime(latest_retry.updated_at / 1000, 'unixepoch')
         ELSE NULL
       END,
       ?
     FROM memories
     LEFT JOIN evolution_jobs AS latest_job ON latest_job.id = (
       SELECT id FROM evolution_jobs
       WHERE target_memory_id = memories.id
         AND job_type IN ('trace_summary', 'import_summary', 'embedding')
       ORDER BY updated_at DESC, rowid DESC LIMIT 1
     )
     LEFT JOIN embedding_retry_queue AS latest_retry ON latest_retry.id = (
       SELECT id FROM embedding_retry_queue
       WHERE target_kind = 'trace' AND target_id = memories.id
         AND vector_field = 'vec_summary'
       ORDER BY updated_at DESC, rowid DESC LIMIT 1
     )
     WHERE memories.deleted_at IS NULL
       AND memories.status != 'deleted'
       AND memories.memory_layer = 'L1'
       AND json_extract(properties_json, '$.internal_info.memory_kind') = 'trace'
     ON CONFLICT(memory_id) DO NOTHING`
  ).run(now);

  db.prepare(
    `DELETE FROM evolution_jobs
     WHERE job_type IN ('trace_summary', 'import_summary', 'embedding')
       AND target_memory_id IN (SELECT memory_id FROM memory_processing_state)`
  ).run();
  db.prepare(
    `DELETE FROM embedding_retry_queue
     WHERE target_kind = 'trace'
       AND target_id IN (SELECT memory_id FROM memory_processing_state)`
  ).run();
}

function removeLegacyProcessingMetadata(db: Database.Database): void {
  const rows = db.prepare(
    `SELECT id, tags_json, info_json, properties_json, memory_value, status, deleted_at
     FROM memories
     WHERE memory_layer = 'L1'
       AND json_extract(properties_json, '$.internal_info.memory_kind') = 'trace'`
  ).all() as Array<{
    id: string;
    tags_json: string;
    info_json: string;
    properties_json: string;
    memory_value: string;
    status: string;
    deleted_at: string | null;
  }>;

  for (const row of rows) {
    const tags = stripProcessingTags(parseJsonArray(row.tags_json));
    const info = parseJsonObject(row.info_json);
    const properties = parseJsonObject(row.properties_json);
    delete info.import_pipeline;
    if (Array.isArray(info.tags)) info.tags = stripProcessingTags(info.tags);
    if (Array.isArray(properties.tags)) properties.tags = stripProcessingTags(properties.tags);
    const publicInfo = isRecord(properties.info) ? properties.info : undefined;
    if (publicInfo) {
      delete publicInfo.import_pipeline;
      if (Array.isArray(publicInfo.tags)) publicInfo.tags = stripProcessingTags(publicInfo.tags);
    }
    const internalInfo = isRecord(properties.internal_info) ? properties.internal_info : undefined;
    if (internalInfo) delete internalInfo.import_pipeline;

    db.prepare(
      `UPDATE memories SET tags_json = ?, info_json = ?, properties_json = ? WHERE id = ?`
    ).run(JSON.stringify(tags), JSON.stringify(info), JSON.stringify(properties), row.id);
    db.prepare(`DELETE FROM memories_fts WHERE id = ?`).run(row.id);
    if (!row.deleted_at && row.status !== "deleted") {
      db.prepare(
        `INSERT INTO memories_fts (id, identifier, memory_value, tags) VALUES (?, ?, ?, ?)`
      ).run(row.id, row.id, row.memory_value, tags.join(" "));
    }
  }
}

function parseJsonArray(value: string): unknown[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function parseJsonObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown;
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function stripProcessingTags(values: unknown[]): string[] {
  return [...new Set(values
    .filter((value): value is string => typeof value === "string")
    .filter((value) => !PROCESSING_TAGS.has(value)))];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function tableExists(db: Database.Database, table: string): boolean {
  return Boolean(db.prepare(`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?`).get(table));
}

export function getSchemaVersion(db: Database.Database): {
  version: number;
  lastMigrationId?: string;
} {
  const row = db
    .prepare(
      `SELECT id, version
       FROM schema_migrations
       ORDER BY version DESC
       LIMIT 1`
    )
    .get() as { id: string; version: number } | undefined;

  return {
    version: row?.version ?? 0,
    lastMigrationId: row?.id
  };
}
