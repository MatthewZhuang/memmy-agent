import { describe, expect, it } from "vitest";

import { skillMetaFromMemory } from "../../../src/algorithm/plugin-algorithms.js";
import { DEFAULT_MEMMY_CONFIG, type MemmyConfig } from "../../../src/config/index.js";
import type { LlmClient, LlmMessage } from "../../../src/model/types.js";
import {
  ProceduralPatternSkillMaterializer,
  type ProceduralPatternSkillInput,
  type ProceduralSkillComparisonCandidate
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
    expect(saved.properties.internal_info.completion_activated).toBe(false);
    expect(harness.llmPayloads[0]).toMatchObject({
      evidence_anchor_catalog: expect.arrayContaining([
        expect.objectContaining({
          anchor_id: "core_0",
          allowed_usage: "mandatory",
          support_episode_ids: ["episode-a", "episode-b"]
        })
      ]),
      anchor_contract: {
        allowed_mandatory_anchor_ids: expect.arrayContaining(["core_0", "core_1"]),
        allowed_conditional_anchor_ids: []
      },
      completion: {
        activated: false,
        shared_prefix: [],
        shared_suffix: []
      },
      occurrences: expect.arrayContaining([expect.objectContaining({
        prefix_extension: [],
        suffix_extension: []
      })])
    });

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
      sourceAnchorIds: ["core_0"],
      evidenceRefs: ["step-a", "step-b"],
      supportingPolicyIds: [],
      supportingOccurrenceIds: ["occ-a", "occ-b"]
    });
    expect(procedure.steps[0]?.body).not.toContain("<script>");
    expect(saved.properties.internal_info).toMatchObject({
      evidence_binding_version: "semantic-anchor-binding.v1",
      selected_mandatory_anchor_ids: ["core_0", "core_1"],
      local_subproblem_closure: {
        closed: true,
        support_episode_ids: ["episode-a", "episode-b"]
      }
    });
    expect(harness.llmSystemPrompts[0]).toContain(
      "V2 local-subproblem closed-loop admission gate"
    );

    const embedding = harness.enqueued.find((item) => item.jobType === "embedding");
    expect(embedding?.targetMemoryId).toBe(saved.id);
    expect(embedding?.payload.contentHash).toBe(saved.contentHash);
  });

  it("requires the V2 model to establish a cross-Episode local closed loop", async () => {
    const response = validSkillResponse();
    delete response.local_subproblem_closure;
    const missing = createHarness({ skillResponse: response });

    expect(await missing.induce(skillInput())).toEqual({
      admitted: false,
      reason: "missing-v2-local-subproblem-closure"
    });
    expect(missing.savedMemory).toBeUndefined();

    const openLoopResponse = validSkillResponse();
    openLoopResponse.local_subproblem_closure = {
      closed: false,
      subproblem: "A generator failed.",
      entry_condition: "The first execution failed.",
      resolution: "The script was edited.",
      resolved_state: "",
      success_check: "",
      support_episode_ids: ["episode-a", "episode-b"],
      reason: "Neither Episode reran and verified the edited script."
    };
    const openLoop = createHarness({ skillResponse: openLoopResponse });

    expect(await openLoop.induce(skillInput())).toEqual({
      admitted: false,
      reason: "no-cross-episode-local-closed-loop"
    });
    expect(openLoop.savedMemory).toBeUndefined();
  });

  it("accepts a model rejection when the local subproblem has no shared success check", async () => {
    const response = validSkillResponse();
    response.admit = false;
    response.rejection_reason =
      "no-cross-episode-local-closed-loop:missing-shared-success-check";
    response.local_subproblem_closure = {
      closed: false,
      subproblem: "Repair a failed generator.",
      entry_condition: "The generator failed.",
      resolution: "The generator was edited.",
      resolved_state: "",
      success_check: "",
      support_episode_ids: ["episode-a", "episode-b"],
      reason: "The aligned region contains no repeated rerun and verification."
    };
    const harness = createHarness({ skillResponse: response });

    expect(await harness.induce(skillInput())).toEqual({
      admitted: false,
      reason: "no-cross-episode-local-closed-loop:missing-shared-success-check"
    });
    expect(harness.savedMemory).toBeUndefined();
  });

  it("keeps the public Skill schema while recording long-trajectory origin", async () => {
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Generate and validate artifact",
      body: "Generate the requested artifact and validate its output.",
      source_anchor_ids: ["core_0", "core_1"]
    }];
    const harness = createHarness({ skillResponse: response });
    const input = skillInput();
    input.origin = {
      kind: "long_trajectory",
      episodeFamilyId: "episode-family-1",
      longTrajectoryId: "long-trajectory-1"
    };
    input.episodeContextReadOnly = [
      {
        episodeId: "episode-a",
        goal: "Generate a validated artifact for the user.",
        terminalResult: "The artifact was generated and validated."
      },
      {
        episodeId: "episode-b",
        goal: "Produce and verify the requested output.",
        terminalResult: "The output passed verification."
      }
    ];

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
    const saved = harness.savedMemory!;
    expect(skillMetaFromMemory(saved)).not.toBeNull();
    expect(saved.tags).toEqual(expect.arrayContaining([
      "skill",
      "long-trajectory",
      "episode-family"
    ]));
    expect(saved.properties.internal_info).toMatchObject({
      source: "worker.long_trajectory_induction.v1",
      plugin_algorithm: "procedural.long-trajectory.skill.v1",
      source_episode_family_id: "episode-family-1",
      source_long_trajectory_id: "long-trajectory-1"
    });
    expect(harness.llmPayloads[0]).toMatchObject({
      pattern: {
        origin: {
          kind: "long_trajectory",
          episode_family_id: "episode-family-1",
          long_trajectory_id: "long-trajectory-1"
        }
      },
      episode_context_read_only: [
        {
          episode_id: "episode-a",
          goal: "Generate a validated artifact for the user.",
          terminal_result: "The artifact was generated and validated."
        },
        {
          episode_id: "episode-b",
          goal: "Produce and verify the requested output.",
          terminal_result: "The output passed verification."
        }
      ]
    });
    expect(harness.llmSystemPrompts[0]).not.toContain(
      "V2 local-subproblem closed-loop admission gate"
    );
  });

  it("rejects a long-trajectory Skill that drops a required reference Span", async () => {
    const harness = createHarness();
    const input = skillInput();
    input.origin = {
      kind: "long_trajectory",
      episodeFamilyId: "episode-family-1",
      longTrajectoryId: "long-trajectory-1"
    };

    const result = await harness.induce(input);

    expect(result).toEqual({
      admitted: false,
      reason: "incomplete-reference-span-coverage"
    });
    expect(harness.savedMemory).toBeUndefined();
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
      skillLlm: llm(() => validSkillResponse()),
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
      source_anchor_ids: ["core_failed"]
    }];
    response.verification_steps = [{
      check: "Use the failed result",
      success_signal: "The failed action is treated as success.",
      source_anchor_ids: ["core_failed"]
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

    expect(result).toEqual({
      admitted: false,
      reason: "invalid-evidence-anchor"
    });
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
      source_anchor_ids: [gapAnchorId("occ-a", "gap-a")]
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
      source_anchor_ids: [
        gapAnchorId("occ-a", "gap-0"),
        gapAnchorId("occ-b", "gap-1")
      ]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
  });

  it("allows a shared outward extension cited by two successful Episodes", async () => {
    const input = skillInput();
    input.completion.activated = true;
    input.completion.sharedSuffix = [{
      anchorId: "extension_suffix_0",
      side: "suffix",
      referenceStepId: "extension-a",
      supportEpisodeIds: ["episode-a", "episode-b"],
      evidenceStepIds: ["extension-a", "extension-b"],
      averageMatchSimilarity: 0.91
    }];
    for (const [index, item] of input.evidence.entries()) {
      item.suffixExpansion.push({
        role: "shared_extension",
        side: "suffix",
        extensionAnchorId: "extension_suffix_0",
        matchSimilarity: 0.91,
        stepId: `extension-${index === 0 ? "a" : "b"}`,
        stepIndex: 3,
        toolName: "shell",
        intent: "Inspect the generated artifact content",
        summary: "The generated artifact content was valid",
        outcome: "success",
        evidenceRefs: [`extension-evidence-${index}`]
      });
    }
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Inspect generated content",
      body: "Inspect the generated artifact content.",
      source_anchor_ids: ["extension_suffix_0"]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
    expect(harness.llmPayloads[0]).toMatchObject({
      completion: {
        activated: true,
        shared_suffix: [expect.objectContaining({
          anchor_id: "extension_suffix_0"
        })]
      },
      occurrences: expect.arrayContaining([expect.objectContaining({
        suffix_extension: [expect.objectContaining({
          role: "shared_extension",
          extension_anchor_id: "extension_suffix_0"
        })]
      })])
    });
  });

  it("never accepts an occurrence-local outward Step as positive evidence", async () => {
    const input = skillInput();
    for (const [index, item] of input.evidence.entries()) {
      item.prefixExpansion.push({
        role: "local_context",
        side: "prefix",
        stepId: `local-${index}`,
        stepIndex: -1,
        toolName: "shell",
        intent: "Perform occurrence-specific setup",
        summary: "Local setup completed",
        outcome: "success",
        evidenceRefs: [`local-evidence-${index}`]
      });
    }
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Use local setup",
      body: "Make occurrence-specific setup mandatory.",
      source_anchor_ids: [
        provisionalAnchorId("occ-a", "local-0"),
        provisionalAnchorId("occ-b", "local-1")
      ]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result).toEqual({
      admitted: false,
      reason: "core-incomplete-after-provisional-extension-fallback"
    });
  });

  it("keeps explicitly read-only V3 local context forbidden as positive evidence", async () => {
    const input = skillInput();
    input.origin = {
      kind: "long_trajectory",
      episodeFamilyId: "family-soft-fine",
      longTrajectoryId: "trajectory-soft-fine"
    };
    for (const occurrenceItem of input.evidence) {
      const original = occurrenceItem.alignedSequence[0]!;
      occurrenceItem.alignedSequence[0] = {
        role: "local_context",
        spanAnchorId: "core_0",
        stepId: original.stepId,
        stepIndex: original.stepIndex,
        ...(original.toolName ? { toolName: original.toolName } : {}),
        intent: original.intent,
        summary: original.summary,
        outcome: original.outcome,
        evidenceRefs: original.evidenceRefs
      };
    }
    const harness = createHarness();

    const result = await harness.induce(input);

    expect(result).toEqual({ admitted: false, reason: "invalid-evidence-anchor" });
    expect(harness.llmPayloads[0]?.anchor_contract).toMatchObject({
      forbidden_local_context_step_ids: ["step-a", "step-b"]
    });
  });

  it("lets V3 ground claims across Episodes without a Fine alignment group", async () => {
    const input = skillInput();
    input.origin = {
      kind: "long_trajectory",
      episodeFamilyId: "family-soft-fine",
      longTrajectoryId: "trajectory-soft-fine"
    };
    for (const occurrenceItem of input.evidence) {
      occurrenceItem.alignedSequence = occurrenceItem.alignedSequence.map((step) =>
        step.role === "core"
          ? {
              role: "span_step",
              anchorId: step.anchorId,
              spanSimilarity: 0.8,
              stepId: step.stepId,
              stepIndex: step.stepIndex,
              ...(step.toolName ? { toolName: step.toolName } : {}),
              intent: step.intent,
              summary: step.summary,
              outcome: step.outcome,
              evidenceRefs: step.evidenceRefs
            }
          : step);
    }
    const response = validSkillResponse();
    response.procedure_steps = [
      {
        title: "Generate artifact",
        body: "Generate the requested artifact.",
        source_anchor_ids: [
          spanStepAnchorId("occ-a", "step-a"),
          spanStepAnchorId("occ-b", "step-b")
        ]
      },
      {
        title: "Validate artifact",
        body: "Validate the generated artifact.",
        source_anchor_ids: [
          spanStepAnchorId("occ-a", "step-a-verify"),
          spanStepAnchorId("occ-b", "step-b-verify")
        ]
      }
    ];
    response.verification_steps = [{
      check: "Validate output",
      success_signal: "The validation command succeeds.",
      source_anchor_ids: [
        spanStepAnchorId("occ-a", "step-a-verify"),
        spanStepAnchorId("occ-b", "step-b-verify")
      ]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
    expect(harness.llmPayloads[0]?.occurrences).toEqual(expect.arrayContaining([
      expect.objectContaining({
        aligned_sequence: expect.arrayContaining([
          expect.objectContaining({ role: "span_step", anchor_id: "core_0" })
        ])
      })
    ]));
  });

  it("still requires every V3 Skill claim to cite two successful Episodes", async () => {
    const input = skillInput();
    input.origin = {
      kind: "long_trajectory",
      episodeFamilyId: "family-coarse-span",
      longTrajectoryId: "trajectory-coarse-span"
    };
    for (const occurrenceItem of input.evidence) {
      occurrenceItem.alignedSequence = occurrenceItem.alignedSequence.map((step) =>
        step.role === "core"
          ? {
              role: "span_step",
              anchorId: step.anchorId,
              spanSimilarity: 0.8,
              stepId: step.stepId,
              stepIndex: step.stepIndex,
              ...(step.toolName ? { toolName: step.toolName } : {}),
              intent: step.intent,
              summary: step.summary,
              outcome: step.outcome,
              evidenceRefs: step.evidenceRefs
            }
          : step);
    }
    const response = validSkillResponse();
    response.procedure_steps = [
      {
        title: "Generate artifact",
        body: "Generate the requested artifact.",
        source_anchor_ids: [
          spanStepAnchorId("occ-a", "step-a"),
          spanStepAnchorId("occ-b", "step-b")
        ]
      },
      {
        title: "Validate artifact",
        body: "Validate the generated artifact.",
        source_anchor_ids: [
          spanStepAnchorId("occ-a", "step-a-verify"),
          spanStepAnchorId("occ-b", "step-b-verify")
        ]
      },
      {
        title: "Use an unsupported local refinement",
        body: "Apply a refinement observed in only one Episode.",
        source_anchor_ids: [spanStepAnchorId("occ-a", "step-a")]
      }
    ];
    response.verification_steps = [{
      check: "Validate output",
      success_signal: "The validation command succeeds.",
      source_anchor_ids: [
        spanStepAnchorId("occ-a", "step-a-verify"),
        spanStepAnchorId("occ-b", "step-b-verify")
      ]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result).toEqual({
      admitted: false,
      reason: "insufficient-claim-episode-support"
    });
  });

  it("keeps a single-Episode extension as conditional guidance and falls back to Core", async () => {
    const input = skillInput();
    input.evidence[0]!.suffixExpansion.push({
      role: "shared_extension",
      side: "suffix",
      extensionAnchorId: "extension_suffix_0",
      matchSimilarity: 1,
      stepId: "extension-a",
      stepIndex: 3,
      toolName: "shell",
      intent: "Inspect generated content",
      summary: "Generated content was inspected",
      outcome: "success",
      evidenceRefs: ["extension-evidence-a"]
    });
    const response = validSkillResponse();
    response.procedure_steps = [
      ...(response.procedure_steps as Array<Record<string, unknown>>),
      {
        title: "Inspect generated content",
        body: "Inspect the generated content.",
        source_anchor_ids: [provisionalAnchorId("occ-a", "extension-a")]
      }
    ];
    response.conditional_guidance = [{
      condition: "The generated content requires an additional format-specific check.",
      action: "Inspect the generated content before delivery.",
      source_anchor_ids: [provisionalAnchorId("occ-a", "extension-a")]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
    expect(harness.llmPayloads[0]).toMatchObject({
      provisional_extensions: [expect.objectContaining({
        episode_id: "episode-a",
        allowed_usage: "conditional_only",
        evidence_status: "provisional",
        step: expect.objectContaining({ step_id: "extension-a" })
      })],
      anchor_contract: {
        allowed_conditional_anchor_ids: [provisionalAnchorId("occ-a", "extension-a")]
      }
    });
    const procedure = harness.savedMemory?.properties.internal_info.procedure_json as {
      steps: Array<{ title: string }>;
      conditionalGuidance: Array<{
        condition: string;
        action: string;
        evidenceStatus: string;
        supportEpisodeCount: number;
      }>;
    };
    expect(procedure.steps.map((item) => item.title)).not.toContain(
      "Inspect generated content"
    );
    expect(procedure.conditionalGuidance).toEqual([expect.objectContaining({
      evidenceStatus: "provisional",
      supportEpisodeCount: 1
    })]);
    expect(harness.savedMemory?.properties.internal_info).toMatchObject({
      core_fallback_applied: true,
      provisional_extension_evidence_refs: ["extension-a"],
      provisional_extension_count: 1
    });
    expect(harness.savedMemory?.memoryValue).toContain(
      "## Conditional guidance (provisional)"
    );
  });

  it("waits for more evidence when removing a provisional extension leaves no Core procedure", async () => {
    const input = skillInput();
    input.evidence[0]!.suffixExpansion.push({
      role: "local_context",
      side: "suffix",
      stepId: "extension-a",
      stepIndex: 3,
      toolName: "shell",
      intent: "Inspect generated content",
      summary: "Generated content was inspected",
      outcome: "success",
      evidenceRefs: ["extension-evidence-a"]
    });
    const response = validSkillResponse();
    response.procedure_steps = [{
      title: "Inspect generated content",
      body: "Inspect the generated content.",
      source_anchor_ids: [provisionalAnchorId("occ-a", "extension-a")]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result).toEqual({
      admitted: false,
      reason: "core-incomplete-after-provisional-extension-fallback"
    });
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
      source_anchor_ids: ["boundary-a", "boundary-b"]
    }];
    const harness = createHarness({ skillResponse: response });

    const result = await harness.induce(input);

    expect(result).toEqual({ admitted: false, reason: "invalid-evidence-anchor" });
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

  it("generates a V2 draft first, then suppresses it in a separate coverage call", async () => {
    const coverageResponse = {
      decision: "covered",
      target_skill_id: "skill-v3-existing",
      target_route: "V3",
      relation: "subset",
      reason: "The V2 procedure is an ordered subset of the V3 procedure."
    };
    const harness = createHarness({
      coverageResponse,
      comparisonSkills: [comparisonSkill("skill-v3-existing", "V3")]
    });

    const result = await harness.induce(skillInput());

    expect(result).toEqual({
      admitted: false,
      reason: "covered-by-v3:skill-v3-existing",
      coverageDecision: {
        decision: "covered",
        targetSkillId: "skill-v3-existing",
        targetRoute: "V3",
        relation: "subset",
        reason: "The V2 procedure is an ordered subset of the V3 procedure."
      }
    });
    expect(harness.savedMemory).toBeUndefined();
    expect(harness.llmPayloads).toHaveLength(2);
    expect(harness.llmPayloads[0]).not.toHaveProperty("comparison_skills_read_only");
    expect(harness.llmPayloads[1]).toMatchObject({
      skill_draft_read_only: expect.objectContaining({
        name: "generate_and_validate_artifact"
      }),
      comparison_skills_read_only: [
        expect.objectContaining({ memory_id: "skill-v3-existing", route: "V3" })
      ]
    });
  });

  it("filters a new Candidate equivalent to an existing V2 Skill instead of updating it", async () => {
    const existing = existingV2Skill("skill-v2-existing", "skill:procedural:canonical");
    const harness = createHarness({
      existingSkills: [existing],
      comparisonSkills: [comparisonSkill(existing.id, "V2")],
      coverageResponse: {
        decision: "covered",
        target_skill_id: existing.id,
        target_route: "V2",
        relation: "equivalent",
        reason: "The completed Draft expresses the same capability."
      }
    });

    const result = await harness.induce(skillInput());

    expect(result).toEqual({
      admitted: false,
      reason: `covered-by-v2:${existing.id}`,
      coverageDecision: {
        decision: "covered",
        targetSkillId: existing.id,
        targetRoute: "V2",
        relation: "equivalent",
        reason: "The completed Draft expresses the same capability."
      }
    });
    expect(harness.savedMemory).toBeUndefined();
    expect(harness.llmCalls).toBe(2);
  });

  it("creates a new V2 Skill only after the completed Draft is classified distinct", async () => {
    const harness = createHarness({
      comparisonSkills: [comparisonSkill("skill-old-related", "OLD")],
      coverageResponse: {
        decision: "distinct",
        target_skill_id: null,
        target_route: null,
        relation: "distinct",
        reason: "The resolved-state and verification contracts differ."
      }
    });

    const result = await harness.induce(skillInput());

    expect(result.admitted).toBe(true);
    expect(harness.savedMemory).toBeDefined();
    expect(harness.savedMemory?.properties.internal_info.reuse_decision).toMatchObject({
      action: "create_v2",
      relation: "distinct"
    });
    expect(harness.llmCalls).toBe(2);
  });

  it("returns the raw post-Draft coverage output when its protocol is invalid", async () => {
    const harness = createHarness({
      comparisonSkills: [comparisonSkill("skill-old-related", "OLD")],
      coverageResponse: {
        decision: "covered",
        target_skill_id: null,
        target_route: "OLD",
        relation: "equivalent",
        reason: "Missing the required target id."
      }
    });

    const compiled = await harness.materializer.compile(skillInput());
    expect(compiled.admitted).toBe(true);
    if (!compiled.admitted) return;
    const coverage = await harness.materializer.compareDraftCoverage(
      compiled.draft,
      [comparisonSkill("skill-old-related", "OLD")]
    );

    expect(coverage).toEqual({
      ok: false,
      reason: "invalid-coverage-decision",
      rawDecision: {
        decision: "covered",
        target_skill_id: null,
        target_route: "OLD",
        relation: "equivalent",
        reason: "Missing the required target id."
      }
    });
  });

  it("updates the Skill already owned by the same Candidate without a coverage call", async () => {
    const existing = existingV2Skill("skill-v2-existing", "skill:procedural:canonical");
    const input = skillInput();
    input.existingSkillReadOnly = existingSkillReadOnly(existing);
    const harness = createHarness({ existingSkills: [existing] });

    const result = await harness.induce(input);

    expect(result.admitted).toBe(true);
    expect(harness.savedMemory?.memoryKey).toBe("skill:procedural:canonical");
    expect(harness.savedMemory?.properties.internal_info).toMatchObject({
      source_episode_ids: ["episode-old", "episode-a", "episode-b"],
      source_cluster_ids: ["cluster-old", "cluster-1"],
      reuse_decision: {
        action: "update_v2",
        targetSkillId: existing.id,
        targetRoute: "V2"
      }
    });
    expect(harness.llmCalls).toBe(1);
    expect(harness.llmPayloads[0]).not.toHaveProperty("comparison_skills_read_only");
    expect(harness.llmPayloads[0]?.existing_skill_read_only).toMatchObject({
      memory_id: existing.id
    });
  });
});

function createHarness(options: {
  episodeRewards?: Record<string, number | undefined>;
  traceValues?: [number, number];
  skillResponse?: Record<string, unknown>;
  coverageResponse?: Record<string, unknown>;
  comparisonSkills?: ProceduralSkillComparisonCandidate[];
  existingSkills?: MemoryRow[];
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
  const llmSystemPrompts: string[] = [];
  const episodes = new Map(Object.entries(episodeRewards).map(([id, rTask]) => [
    id,
    episode(id, rTask)
  ]));
  const repositories = {
    memories: {
      get(id: string) {
        return options.existingSkills?.find((memory) => memory.id === id) ??
          (savedMemory?.id === id ? savedMemory : undefined);
      },
      getMany(ids: string[]) {
        return ids.flatMap((id) => traces.get(id) ? [traces.get(id)!] : []);
      },
      getByKey(_layer: string, key: string) {
        return options.existingSkills?.find((memory) => memory.memoryKey === key) ?? savedMemory;
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
    skillLlm: llm((messages) => {
      llmCalls += 1;
      const system = messages.find((message) => message.role === "system")?.content ?? "";
      llmSystemPrompts.push(system);
      const user = messages.find((message) => message.role === "user")?.content ?? "{}";
      llmPayloads.push(JSON.parse(user) as Record<string, unknown>);
      return system.includes("already compiled V2 Skill Draft")
        ? options.coverageResponse ?? {
            decision: "distinct",
            target_skill_id: null,
            target_route: null,
            relation: "distinct",
            reason: "No retrieved Skill covers the Draft."
          }
        : options.skillResponse ?? validSkillResponse();
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
    if (!input.existingSkillReadOnly && (options.comparisonSkills?.length ?? 0) > 0) {
      const coverage = await materializer.compareDraftCoverage(
        result.draft,
        options.comparisonSkills ?? []
      );
      if (!coverage.ok) return { admitted: false as const, reason: coverage.reason };
      if (coverage.decision.decision === "covered") {
        return {
          admitted: false as const,
          reason: `covered-by-${coverage.decision.targetRoute.toLowerCase()}:` +
            coverage.decision.targetSkillId,
          coverageDecision: coverage.decision
        };
      }
    }
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
    llmSystemPrompts,
    get savedMemory() { return savedMemory; },
    get llmCalls() { return llmCalls; }
  };
}

function comparisonSkill(
  memoryId: string,
  route: "OLD" | "V2" | "V3"
): ProceduralSkillComparisonCandidate {
  return {
    memoryId,
    route,
    name: "artifact-generation-recovery",
    invocationGuide: "Repair a failed generator, rerun it, and verify the output.",
    triggerContext: "Use after an artifact generator fails.",
    summary: "Repair, rerun, and verify the generated artifact.",
    procedureSteps: [
      { title: "Repair generator", body: "Repair the identified generator error." },
      { title: "Rerun generator", body: "Rerun the corrected generator." }
    ],
    verificationSteps: [{ title: "Verify output", body: "Confirm the artifact exists." }]
  };
}

function existingSkillReadOnly(
  existing: MemoryRow
): NonNullable<ProceduralPatternSkillInput["existingSkillReadOnly"]> {
  const skill = skillMetaFromMemory(existing)!;
  const internal = existing.properties.internal_info;
  return {
    memoryId: existing.id,
    memoryVersion: existing.version,
    name: skill.name,
    invocationGuide: skill.invocationGuide,
    procedureJson: internal.procedure_json as Record<string, unknown> ?? {},
    sourceEpisodeIds: internal.source_episode_ids as string[] ?? [],
    sourceTraceIds: internal.source_trace_ids as string[] ?? [],
    sourceSpanOccurrenceIds: internal.source_span_occurrence_ids as string[] ?? []
  };
}

function existingV2Skill(id: string, key: string): MemoryRow {
  return memory({
    id,
    layer: "Skill",
    kind: "skill",
    key,
    value: "# Existing V2 Skill\n\nRepair, rerun, and verify.",
    status: "resolving",
    internal: {
      plugin_algorithm: "procedural.pattern.skill.v1",
      source_trace_ids: ["trace-old"],
      source_episode_ids: ["episode-old"],
      source_span_occurrence_ids: ["occ-old"],
      source_cluster_id: "cluster-old",
      source_cluster_version_id: "cluster-version-old",
      skill: {
        name: "artifact-generation-recovery",
        eta: 0.7,
        status: "candidate",
        support: 1,
        source_policy_ids: [],
        source_world_model_ids: [],
        evidence_anchor_ids: ["trace-old"],
        invocation_guide: "Repair, rerun, and verify.",
        procedure_json: {
          triggerContext: "Use after a generator fails.",
          summary: "Repair and verify the generated artifact.",
          steps: []
        },
        trials_attempted: 0,
        trials_passed: 0,
        success_rate: 0,
        beta_posterior: { alpha: 1, beta: 1, mean: 0.5 }
      }
    }
  });
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
    completion: {
      id: "anchored-completion-1",
      version: "anchored-completion.v2",
      activated: false,
      referenceOccurrenceId: "occ-a",
      maxPrefixSteps: 3,
      maxSuffixSteps: 5,
      minStepSimilarity: 0.7,
      sharedPrefix: [],
      sharedSuffix: [],
      extensionAgreement: 1
    },
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
    prefixExpansion: [],
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
    suffixExpansion: [],
    boundaryContextReadOnly: {}
  };
}

function validSkillResponse(): Record<string, unknown> {
  return {
    admit: true,
    rejection_reason: null,
    local_subproblem_closure: {
      closed: true,
      subproblem: "Produce a requested artifact whose result must be checked.",
      entry_condition: "The requested artifact does not yet exist.",
      resolution: "Generate the artifact using the repeated construction action.",
      resolved_state: "The artifact exists after generation.",
      success_check: "Run the repeated validation action and observe success.",
      support_episode_ids: ["episode-a", "episode-b"],
      reason: "Both successful Episodes contain generation followed by explicit validation."
    },
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
      source_anchor_ids: ["core_0"]
    }],
    verification_steps: [{
      check: "Validate output",
      success_signal: "The validation command succeeds.",
      source_anchor_ids: ["core_1"]
    }],
    do_not_apply_when: ["No artifact should be created."],
    decision_guidance: {
      preference: ["Validate the generated result."],
      anti_pattern: ["Do not assume generation implies validity."]
    },
    conditional_guidance: [],
    examples: [],
    tags: ["artifact", "validation"],
    tools: ["shell"],
    confidence: 0.86
  };
}

function gapAnchorId(occurrenceId: string, stepId: string): string {
  return `gap_anchor_${stableHash({ occurrenceId, stepId }).slice(0, 16)}`;
}

function spanStepAnchorId(occurrenceId: string, stepId: string): string {
  return `span_step_anchor_${stableHash({ occurrenceId, stepId }).slice(0, 16)}`;
}

function provisionalAnchorId(occurrenceId: string, stepId: string): string {
  return `provisional_anchor_${stableHash({ occurrenceId, stepId }).slice(0, 16)}`;
}

function llm(
  respond: (messages: Array<{ role: string; content: string }>) => Record<string, unknown>
): LlmClient {
  return {
    config: DEFAULT_MEMMY_CONFIG.evolution,
    isConfigured: () => true,
    complete: async () => "{}",
    async completeJson<T extends Record<string, unknown>>(messages: LlmMessage[]) {
      return respond(messages) as T;
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
