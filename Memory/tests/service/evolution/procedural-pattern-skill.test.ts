import { describe, expect, it } from "vitest";

import { skillMetaFromMemory } from "../../../src/algorithm/plugin-algorithms.js";
import { DEFAULT_MEMMY_CONFIG, type MemmyConfig } from "../../../src/config/index.js";
import type { LlmClient, LlmMessage } from "../../../src/model/types.js";
import {
  ProceduralPatternSkillMaterializer,
  type ProceduralPatternSkillInput
} from "../../../src/service/evolution/procedural-pattern-skill.js";
import { SkillPipeline } from "../../../src/service/evolution/skill-pipeline.js";
import type {
  EpisodeRecord,
  EvolutionJobRecord,
  Repositories
} from "../../../src/storage/repositories.js";
import type { MemoryKind, MemoryRow } from "../../../src/types.js";
import { stableHash } from "../../../src/utils/id.js";

const AT = "2026-08-27T00:00:00.000Z";
const USER_ID = "user-procedural-skill";

describe("procedural pattern Skill compatibility", () => {
  it("materializes the canonical Skill shape with trace provenance and a current embedding hash", async () => {
    const harness = createHarness();
    const result = await harness.induce(skillInput());

    expect(result.admitted).toBe(true);
    expect(harness.savedMemory).toBeDefined();
    const saved = harness.savedMemory!;
    const meta = skillMetaFromMemory(saved);
    expect(meta).not.toBeNull();
    expect(meta?.sourcePolicyIds).toEqual([]);
    expect(meta?.evidenceAnchorIds).toEqual(["trace-a", "trace-b"]);
    expect(saved.properties.internal_info.source_memory_ids).toEqual(["trace-a", "trace-b"]);
    expect(saved.properties.internal_info.source_policy_ids).toEqual([]);

    const procedure = saved.properties.internal_info.procedure_json as {
      steps: Array<{
        id?: string;
        body?: string;
        evidenceRefs?: string[];
        supportingPolicyIds?: string[];
        supportingOccurrenceIds?: string[];
      }>;
    };
    expect(procedure.steps).toHaveLength(2);
    expect(procedure.steps[0]).toMatchObject({
      id: expect.stringMatching(/^step_[a-f0-9]{12}$/),
      evidenceRefs: ["step-a", "step-b"],
      supportingPolicyIds: [],
      supportingOccurrenceIds: ["occ-a", "occ-b"]
    });
    expect(procedure.steps[0]?.body).not.toContain("<script>");

    const embedding = harness.enqueued.find((item) => item.jobType === "embedding");
    expect(embedding?.targetMemoryId).toBe(saved.id);
    expect(embedding?.payload.contentHash).toBe(saved.contentHash);
  });

  it.each([
    ["failed", -0.3],
    ["unknown", undefined]
  ] as const)("does not count %s Episodes as positive support", async (_label, reward) => {
    const harness = createHarness({
      episodeRewards: { "episode-a": 0.9, "episode-b": reward }
    });
    const result = await harness.induce(skillInput());

    expect(result).toEqual({
      admitted: false,
      reason: "insufficient-positive-episode-support"
    });
    expect(harness.llmCalls).toBe(0);
    expect(harness.savedMemory).toBeUndefined();
  });

  it("does not floor non-positive trace gain into an admitted Skill", async () => {
    const harness = createHarness({ traceValues: [-0.4, -0.2] });
    const result = await harness.induce(skillInput());

    expect(result).toEqual({
      admitted: false,
      reason: "insufficient-positive-trace-gain"
    });
    expect(harness.savedMemory).toBeUndefined();
  });

  it("does not discard a negative trace when calculating mixed evidence gain", async () => {
    const harness = createHarness({ traceValues: [0.8, -0.9] });
    const result = await harness.induce(skillInput());

    expect(result).toEqual({
      admitted: false,
      reason: "insufficient-positive-trace-gain"
    });
    expect(harness.llmCalls).toBe(0);
    expect(harness.savedMemory).toBeUndefined();
  });

  it("keeps direct procedural Skills out of legacy Policy-to-Skill merge selection", async () => {
    const harness = createHarness();
    await harness.induce(skillInput());
    const directSkill = harness.savedMemory!;
    const policyMemory = memory({
      id: "policy-legacy",
      layer: "L2",
      kind: "policy",
      key: "policy:legacy",
      value: "Generate, validate, and inspect a document artifact.",
      internal: {}
    });
    const policy = {
      id: policyMemory.id,
      memory: policyMemory,
      title: "Generate and validate document artifacts",
      trigger: "A document artifact must be generated and checked.",
      procedure: "Generate the artifact and validate its output.",
      verification: "Confirm the artifact exists and passes validation.",
      boundary: "Use only for document artifact tasks.",
      support: 2,
      gain: 0.8,
      confidence: 0.8,
      status: "active",
      experienceType: "success_pattern",
      evidencePolarity: "positive",
      skillEligible: true,
      signature: "document|generate|shell|none",
      sourceEpisodeIds: ["episode-a", "episode-b"],
      sourceTraceIds: ["trace-a", "trace-b"],
      sourceFeedbackIds: [],
      decisionGuidance: { preference: [], antiPattern: [] },
      freshnessClass: "stable",
      salience: 0.8,
      vec: null,
      updatedAtMs: Date.parse(AT)
    } as const;
    const pipeline = new SkillPipeline({
      repos: {
        memories: { list: () => [directSkill] }
      } as unknown as Repositories,
      config: config(),
      skillLlm: llm(validSkillResponse(), () => undefined),
      traceMeta: () => null,
      buildMemory: () => directSkill,
      upsertEvolutionMemory: (item) => ({ memory: item, created: false }),
      isArchivedEvolutionMemory: () => false,
      enqueueJob: () => job(),
      namespaceIdFromMemory: () => "ns"
    });

    expect(pipeline.findExistingSkillForPolicy(policy as never)).toBeNull();
  });

  it("uses a stable user-level scope when supporting traces span sessions and agents", async () => {
    const harness = createHarness();
    const firstInput = skillInput();
    await harness.induce(firstInput);
    const first = harness.savedMemory!;
    const firstScope = memoryScope(first);

    await harness.induce({
      ...firstInput,
      sourceTraceIds: [...firstInput.sourceTraceIds].reverse(),
      evidence: [...firstInput.evidence].reverse()
    });
    const second = harness.savedMemory!;

    expect(firstScope).toEqual({
      conversationId: undefined,
      sessionId: undefined,
      agentId: undefined,
      appId: undefined,
      projectId: undefined,
      profileId: undefined
    });
    expect(memoryScope(second)).toEqual(firstScope);
    expect(harness.enqueued.filter((item) => item.jobType === "embedding")
      .map((item) => item.sessionId)).toEqual(["job-session", "job-session"]);
  });

  it("exposes failed occurrences only as read-only counterexamples", async () => {
    const response = validSkillResponse();
    response.do_not_apply_when = [];
    const harness = createHarness({
      episodeRewards: { "episode-a": 0.9, "episode-b": 0.8, "episode-c": 0.9 },
      skillResponse: response
    });
    const input = skillInput();
    input.evidence.push(occurrence("occ-c", "episode-c", "step-c", "failure"));

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
    const payload = harness.llmPayloads[0]!;
    expect(payload.occurrences).toEqual(expect.arrayContaining([
      expect.objectContaining({ occurrence_id: "occ-a" }),
      expect.objectContaining({ occurrence_id: "occ-b" })
    ]));
    expect(payload.occurrences).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ occurrence_id: "occ-c" })
    ]));
    expect(payload.counterexamples_read_only).toEqual([
      expect.objectContaining({ occurrence_id: "occ-c", episode_id: "episode-c" })
    ]);
  });

  it("rejects failed Steps as positive procedure or verification evidence", async () => {
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Use the failed attempt",
      body: "Treat the failed action as the reusable procedure.",
      evidence_refs: ["failed-step-a", "failed-step-b"]
    }];
    response.verification_steps = [{
      check: "Use the failed result",
      success_signal: "The failed action is treated as success.",
      evidence_refs: ["failed-step-a", "failed-step-b"]
    }];
    const input = skillInput();
    input.evidence[0]!.alignedSequence.splice(1, 0, {
      role: "core",
      anchorId: "core_failed",
      matchSimilarity: 0.9,
      stepId: "failed-step-a",
      stepIndex: 1,
      toolName: "shell",
      intent: "Attempt an invalid command",
      summary: "The command failed",
      outcome: "failure",
      evidenceRefs: ["failed-evidence-a"]
    });
    input.evidence[1]!.alignedSequence.push({
      role: "core",
      anchorId: "core_failed",
      matchSimilarity: 0.9,
      stepId: "failed-step-b",
      stepIndex: 1,
      toolName: "shell",
      intent: "Attempt an invalid command",
      summary: "The command failed",
      outcome: "failure",
      evidenceRefs: ["failed-evidence-b"]
    });
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result).toEqual({ admitted: false, reason: "invalid-evidence-citation" });
    expect(harness.savedMemory).toBeUndefined();
  });

  it("serializes the common Core, occurrence-local Gaps, and read-only boundaries", async () => {
    const input = skillInput();
    input.evidence[0]!.alignedSequence.splice(1, 0, {
      role: "gap",
      afterAnchorId: "core_0",
      beforeAnchorId: "core_1",
      stepId: "gap-a",
      stepIndex: 1,
      toolName: "shell",
      intent: "Rerun the corrected generator",
      summary: "The corrected generator completed",
      outcome: "success",
      evidenceRefs: ["gap-evidence-a"]
    });
    input.evidence[0]!.boundaryContextReadOnly.previousStep = {
      stepId: "boundary-a",
      stepIndex: -1,
      toolName: "shell",
      intent: "Inspect unrelated setup",
      summary: "Setup was inspected",
      outcome: "success",
      evidenceRefs: ["boundary-evidence-a"]
    };
    const harness = createHarness();

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
    expect(harness.llmPayloads[0]).toMatchObject({
      common_core: {
        id: "common-core-1",
        anchors: expect.arrayContaining([expect.objectContaining({
          anchor_id: "core_0",
          anchor_intent: "Generate the requested artifact"
        })])
      },
      occurrences: expect.arrayContaining([expect.objectContaining({
        occurrence_id: "occ-a",
        aligned_sequence: expect.arrayContaining([
          expect.objectContaining({ role: "core", step_id: "step-a" }),
          expect.objectContaining({
            role: "gap",
            step_id: "gap-a",
            after_anchor_id: "core_0",
            before_anchor_id: "core_1"
          })
        ]),
        boundary_context_read_only: expect.objectContaining({
          previous_step: expect.objectContaining({ step_id: "boundary-a" })
        })
      })])
    });
  });

  it("rejects a mandatory Gap supported by only one successful Episode", async () => {
    const input = skillInput();
    input.evidence[0]!.alignedSequence.splice(1, 0, {
      role: "gap",
      afterAnchorId: "core_0",
      beforeAnchorId: "core_1",
      stepId: "gap-a",
      stepIndex: 1,
      toolName: "shell",
      intent: "Rerun the corrected generator",
      summary: "The corrected generator completed",
      outcome: "success",
      evidenceRefs: ["gap-evidence-a"]
    });
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Rerun generator",
      body: "Rerun the corrected generator.",
      evidence_refs: ["gap-a"]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result).toEqual({
      admitted: false,
      reason: "insufficient-gap-episode-support"
    });
  });

  it("allows an equivalent Gap cited from two successful Episodes", async () => {
    const input = skillInput();
    for (const [index, occurrenceItem] of input.evidence.entries()) {
      occurrenceItem.alignedSequence.splice(1, 0, {
        role: "gap",
        afterAnchorId: "core_0",
        beforeAnchorId: "core_1",
        stepId: `gap-${index}`,
        stepIndex: 1,
        toolName: "shell",
        intent: "Rerun the corrected generator",
        summary: "The corrected generator completed",
        outcome: "success",
        evidenceRefs: [`gap-evidence-${index}`]
      });
    }
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Rerun generator",
      body: "Rerun the corrected generator.",
      evidence_refs: ["gap-0", "gap-1"]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
  });

  it("never accepts boundary context as positive evidence", async () => {
    const input = skillInput();
    input.evidence[0]!.boundaryContextReadOnly.previousStep = {
      stepId: "boundary-a",
      stepIndex: -1,
      toolName: "shell",
      intent: "Inspect unrelated setup",
      summary: "Setup was inspected",
      outcome: "success",
      evidenceRefs: ["boundary-evidence-a"]
    };
    input.evidence[1]!.boundaryContextReadOnly.previousStep = {
      stepId: "boundary-b",
      stepIndex: -1,
      toolName: "shell",
      intent: "Inspect unrelated setup",
      summary: "Setup was inspected",
      outcome: "success",
      evidenceRefs: ["boundary-evidence-b"]
    };
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Use boundary",
      body: "Treat setup context as the method.",
      evidence_refs: ["boundary-a", "boundary-b"]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result).toEqual({ admitted: false, reason: "invalid-evidence-citation" });
  });

  it("allows an empty do-not-apply section when no counterexample exists", async () => {
    const response = validSkillResponse();
    response.do_not_apply_when = [];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(skillInput());

    expect(result.admitted).toBe(true);
    expect(harness.savedMemory?.memoryValue).not.toContain("## Do not apply when");
    expect(harness.llmPayloads[0]?.counterexamples_read_only).toEqual([]);
  });
});

function createHarness(options: {
  episodeRewards?: Record<string, number | undefined>;
  traceValues?: [number, number];
  skillResponse?: Record<string, unknown>;
} = {}) {
  const episodeRewards = options.episodeRewards ?? {
    "episode-a": 0.9,
    "episode-b": 0.8
  };
  const traces = new Map<string, MemoryRow>([
    ["trace-a", traceMemory("trace-a", "episode-a")],
    ["trace-b", traceMemory("trace-b", "episode-b")]
  ]);
  const traceValues = options.traceValues ?? [0.8, 0.7];
  const enqueued: Array<Record<string, unknown> & { jobType: string; payload: Record<string, unknown> }> = [];
  let savedMemory: MemoryRow | undefined;
  let llmCalls = 0;
  const llmPayloads: Array<Record<string, unknown>> = [];
  const episodes = new Map(Object.entries(episodeRewards).map(([id, rTask]) => [
    id,
    episode(id, rTask)
  ]));
  const repositories = {
    memories: {
      getMany(ids: string[]) {
        return ids.flatMap((id) => traces.get(id) ? [traces.get(id)!] : []);
      },
      getByKey() {
        return savedMemory;
      }
    },
    runtime: {
      getEpisode(id: string) {
        return episodes.get(id);
      },
      appendEpisodeDerivedMemory() {},
      appendChange() {}
    }
  } as unknown as Repositories;
  const materializer = new ProceduralPatternSkillMaterializer({
    repos: repositories,
    config: config(),
    skillLlm: llm(options.skillResponse ?? validSkillResponse(), (messages) => {
      llmCalls += 1;
      const user = messages.find((message) => message.role === "user")?.content ?? "{}";
      llmPayloads.push(JSON.parse(user) as Record<string, unknown>);
    }),
    traceMeta(item) {
      if (!item) return null;
      const index = item.id === "trace-a" ? 0 : 1;
      return {
        id: item.id,
        userId: item.userId,
        sessionId: item.sessionId,
        episodeId: item.properties.internal_info.episode_id as string,
        value: traceValues[index]!,
        priority: traceValues[index]!,
        memory: item
      };
    },
    buildMemory(input) {
      const lifecycle = input.lifecycleStatus as "candidate" | "active";
      const info = {
        ...(input.info as Record<string, unknown>),
        ...(typeof input.projectId === "string" ? { project_id: input.projectId } : {}),
        ...(typeof input.profileId === "string" ? { profile_id: input.profileId } : {})
      };
      const built = memory({
        id: "skill-procedural",
        layer: "Skill",
        kind: "skill",
        key: input.key as string,
        value: input.value as string,
        status: lifecycle === "candidate" ? "resolving" : "activated",
        tags: input.tags as string[],
        info,
        internal: input.internal as Record<string, unknown>
      });
      assignOptionalScope(built, input);
      return built;
    }
  });
  async function induce(input: ProceduralPatternSkillInput) {
    const result = await materializer.compile(input);
    if (!result.admitted) return result;
    const materialized = materializer.materializeDraft(result.draft, AT);
    savedMemory = materialized.memory;
    if (config().algorithm.capture.embedAfterCapture) {
      enqueued.push({
        jobType: "embedding",
        userId: input.userId,
        sessionId: materialized.scope.sessionId ?? "job-session",
        targetMemoryId: savedMemory.id,
        payload: {
          reason: "procedural.pattern.skill.upserted",
          contentHash: savedMemory.contentHash
        }
      });
    }
    return result;
  }
  return {
    materializer,
    induce,
    enqueued,
    llmPayloads,
    get savedMemory() { return savedMemory; },
    get llmCalls() { return llmCalls; }
  };
}

function skillInput(): ProceduralPatternSkillInput {
  return {
    patternVersionId: "pattern-version-1",
    clusterId: "cluster-1",
    clusterVersionId: "cluster-version-1",
    commonCoreId: "common-core-1",
    commonCore: [
      {
        anchorId: "core_0",
        anchorOffset: 0,
        anchorStepId: "step-a",
        anchorIntent: "Generate the requested artifact",
        anchorSummary: "The artifact was generated",
        supportEpisodeIds: ["episode-a", "episode-b"],
        evidenceStepIds: ["step-a", "step-b"]
      },
      {
        anchorId: "core_1",
        anchorOffset: 1,
        anchorStepId: "step-a-verify",
        anchorIntent: "Validate the generated artifact",
        anchorSummary: "Artifact validation succeeded",
        supportEpisodeIds: ["episode-a", "episode-b"],
        evidenceStepIds: ["step-a-verify", "step-b-verify"]
      }
    ],
    userId: USER_ID,
    scale: 5,
    supportEpisodeIds: ["episode-a", "episode-b"],
    sourceTraceIds: ["trace-a", "trace-b"],
    sourceSpanOccurrenceIds: ["occ-a", "occ-b"],
    counterexampleEpisodeIds: [],
    evidence: [
      occurrence("occ-a", "episode-a", "step-a"),
      occurrence("occ-b", "episode-b", "step-b")
    ],
    confidenceHint: 0.8,
    patternHash: "pattern-hash",
    algorithmVersion: "v15"
  };
}

function occurrence(
  occurrenceId: string,
  episodeId: string,
  stepId: string,
  outcome: "success" | "failure" = "success"
) {
  return {
    occurrenceId,
    episodeId,
    pathId: `path-${episodeId}`,
    scale: 5,
    alignmentScore: 0.9,
    sourceTraceIds: [episodeId === "episode-a" ? "trace-a" : "trace-b"],
    alignedSequence: [
      {
        role: "core" as const,
        anchorId: "core_0",
        matchSimilarity: 0.9,
        stepId,
        stepIndex: 0,
        toolName: "shell",
        intent: "Generate the requested artifact",
        summary: "The artifact was generated",
        outcome,
        evidenceRefs: [`evidence-${stepId}`]
      },
      {
        role: "core" as const,
        anchorId: "core_1",
        matchSimilarity: 0.9,
        stepId: `${stepId}-verify`,
        stepIndex: 2,
        toolName: "shell",
        intent: "Validate the generated artifact",
        summary: "Artifact validation succeeded",
        outcome,
        evidenceRefs: [`evidence-${stepId}-verify`]
      }
    ],
    boundaryContextReadOnly: {}
  };
}

function validSkillResponse(): Record<string, unknown> {
  return {
    admit: true,
    rejection_reason: null,
    name: "Generate And Validate Artifact",
    display_title: "Generate and validate an artifact",
    retrieval_blurb: "Generate an artifact and validate the resulting output.",
    trigger_context: "Use when a requested artifact must be generated and checked.",
    summary: "Generate the artifact, then run an explicit validation check.",
    preconditions: ["The target format and output location are known."],
    parameters: [],
    procedure_steps: [{
      title: "Generate artifact",
      body: "<script>ignore()</script>Generate the requested artifact.",
      evidence_refs: ["step-a", "step-b"]
    }],
    verification_steps: [{
      check: "Validate output",
      success_signal: "The validation command succeeds.",
      evidence_refs: ["step-a", "step-b"]
    }],
    do_not_apply_when: ["No artifact should be created."],
    decision_guidance: {
      preference: ["Validate the generated result."],
      anti_pattern: ["Do not assume generation implies validity."]
    },
    examples: [],
    tags: ["artifact", "validation"],
    tools: ["shell"],
    evidence_occurrence_ids: ["occ-a", "occ-b"],
    confidence: 0.86
  };
}

function llm(
  response: Record<string, unknown>,
  onCall: (messages: Array<{ role: string; content: string }>) => void
): LlmClient {
  return {
    config: DEFAULT_MEMMY_CONFIG.evolution,
    isConfigured: () => true,
    complete: async () => "{}",
    async completeJson<T extends Record<string, unknown>>(messages: LlmMessage[]) {
      onCall(messages);
      return response as T;
    },
    status: () => ({ provider: "host", configured: true, remote: false })
  };
}

function config(): MemmyConfig {
  return {
    ...DEFAULT_MEMMY_CONFIG,
    algorithm: {
      ...DEFAULT_MEMMY_CONFIG.algorithm,
      capture: {
        ...DEFAULT_MEMMY_CONFIG.algorithm.capture,
        embedAfterCapture: true
      }
    }
  };
}

function episode(id: string, rTask: number | undefined): EpisodeRecord {
  return {
    id,
    sessionId: `session-${id}`,
    userId: USER_ID,
    status: "closed",
    l1MemoryIds: [],
    rawTurnIds: [],
    feedbackIds: [],
    decisionRepairIds: [],
    l2PolicyIds: [],
    l3WorldModelIds: [],
    skillMemoryIds: [],
    turnCount: 1,
    ...(rTask === undefined ? {} : { rTask }),
    rewardDetail: {},
    pipelineStatus: "succeeded",
    meta: {},
    openedAt: AT,
    closedAt: AT,
    updatedAt: AT
  };
}

function traceMemory(id: string, episodeId: string): MemoryRow {
  const item = memory({
    id,
    layer: "L1",
    kind: "trace",
    key: `trace:${id}`,
    value: `Trace ${id}`,
    internal: { episode_id: episodeId }
  });
  const suffix = episodeId.endsWith("a") ? "a" : "b";
  item.conversationId = `conversation-${suffix}`;
  item.sessionId = `session-${suffix}`;
  item.agentId = `agent-${suffix}`;
  item.appId = `workspace-${suffix}`;
  item.info = {
    ...item.info,
    project_id: `project-${suffix}`,
    profile_id: `profile-${suffix}`
  };
  item.properties.info = item.info;
  return item;
}

function memory(input: {
  id: string;
  layer: MemoryRow["memoryLayer"];
  kind: MemoryKind;
  key: string;
  value: string;
  status?: MemoryRow["status"];
  tags?: string[];
  info?: Record<string, unknown>;
  internal: Record<string, unknown>;
}): MemoryRow {
  const info = input.info ?? {};
  const tags = input.tags ?? [];
  const status = input.status ?? "activated";
  return {
    id: input.id,
    timeline: AT,
    userId: USER_ID,
    conversationId: "conversation-1",
    sessionId: "session-1",
    agentId: "codex",
    appId: "workspace-1",
    memoryType: input.layer === "Skill" ? "SkillMemory" : "LongTermMemory",
    status,
    visibility: "private",
    memoryKey: input.key,
    memoryValue: input.value,
    tags,
    info,
    properties: {
      memory_type: input.layer === "Skill" ? "SkillMemory" : "LongTermMemory",
      status,
      tags,
      info,
      internal_info: {
        memory_layer: input.layer,
        memory_kind: input.kind,
        schema_version: 1,
        ...input.internal
      }
    },
    memoryLayer: input.layer,
    contentHash: stableHash(input.value),
    version: 1,
    createdAt: AT,
    updatedAt: AT,
    deletedAt: null
  };
}

function assignOptionalScope(memory: MemoryRow, input: Record<string, unknown>): void {
  for (const field of ["conversationId", "sessionId", "agentId", "appId"] as const) {
    const value = input[field];
    if (typeof value === "string") memory[field] = value;
    else delete memory[field];
  }
}

function memoryScope(memory: MemoryRow): Record<string, string | undefined> {
  return {
    conversationId: memory.conversationId,
    sessionId: memory.sessionId,
    agentId: memory.agentId,
    appId: memory.appId,
    projectId: typeof memory.info.project_id === "string" ? memory.info.project_id : undefined,
    profileId: typeof memory.info.profile_id === "string" ? memory.info.profile_id : undefined
  };
}

function job(
  jobType: EvolutionJobRecord["jobType"] = "procedural_skill_induction",
  targetMemoryId?: string,
  payload: Record<string, unknown> = {}
): EvolutionJobRecord {
  return {
    id: `job-${jobType}`,
    jobType,
    status: "queued",
    userId: USER_ID,
    sessionId: "job-session",
    episodeId: "episode-b",
    ...(targetMemoryId ? { targetMemoryId } : {}),
    payload,
    attempts: 0,
    maxAttempts: 3,
    createdAt: AT,
    updatedAt: AT
  };
}
