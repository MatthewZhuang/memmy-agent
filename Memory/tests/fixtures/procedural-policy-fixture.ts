import {
  applyStateDelta,
  buildEpisodeProceduralPath,
  emptyObservedState,
  type EpisodeProceduralPathV2,
  type ExecutionStepV1,
  type ProceduralSpanOccurrenceRecord,
  type ProceduralSpanTermination,
  type ProceduralSpanV1,
  type SpanSegmentationDecisionV1
} from "../../src/index.js";
import type { Repositories } from "../../src/storage/repositories.js";

export const PROCEDURAL_POLICY_TEST_NAMESPACE =
  "user-procedural-policy";

export interface PersistedProceduralPolicyEvidence {
  sessionId: string;
  episodeId: string;
  rawTurnId: string;
  path: EpisodeProceduralPathV2;
  occurrence: ProceduralSpanOccurrenceRecord;
}

export function persistProceduralPolicyEvidence(
  repos: Repositories,
  suffix: string,
  input: {
    terminationStatus?: ProceduralSpanTermination;
    rTask?: number;
    at?: string;
    namespaceId?: string;
  } = {}
): PersistedProceduralPolicyEvidence {
  const at = input.at ?? "2026-08-20T00:00:00.000Z";
  const sessionId = `session-${suffix}`;
  const episodeId = `episode-${suffix}`;
  const rawTurnId = `raw-turn-${suffix}`;
  const terminationStatus = input.terminationStatus ?? "success";
  repos.runtime.createSession({
    id: sessionId,
    userId: "user-procedural-policy",
    source: "codex",
    profileId: "default",
    projectId: "project-a",
    status: "closed",
    meta: {},
    openedAt: at,
    lastSeenAt: at,
    closedAt: at,
    updatedAt: at
  });
  repos.runtime.createEpisode({
    id: episodeId,
    sessionId,
    userId: "user-procedural-policy",
    projectId: "project-a",
    status: "closed",
    l1MemoryIds: [],
    rawTurnIds: [rawTurnId],
    feedbackIds: [],
    decisionRepairIds: [],
    l2PolicyIds: [],
    l3WorldModelIds: [],
    skillMemoryIds: [],
    turnCount: 1,
    rTask: input.rTask ?? (terminationStatus === "success" ? 1 : 0),
    rewardDetail: {},
    pipelineStatus: "succeeded",
    meta: {},
    openedAt: at,
    closedAt: at,
    updatedAt: at
  });
  repos.runtime.insertRawTurn({
    id: rawTurnId,
    sessionId,
    episodeId,
    turnId: `host-turn-${suffix}`,
    userId: "user-procedural-policy",
    userText: "Repair the dependency conflict and verify the build",
    assistantText: terminationStatus === "success"
      ? "The corrected constraint passed the focused build."
      : "The retry still failed with a platform compatibility error.",
    toolCalls: [],
    toolResults: [],
    sourceMemoryIds: [],
    usage: {},
    status: terminationStatus === "success" ? "succeeded" : "failed",
    createdAt: at
  });
  const path = buildProceduralPolicyPath({ suffix, episodeId, rawTurnId, terminationStatus });
  const saved = repos.proceduralPaths.save({
    path,
    namespaceId: input.namespaceId ?? PROCEDURAL_POLICY_TEST_NAMESPACE,
    createdAt: at
  });
  return {
    sessionId,
    episodeId,
    rawTurnId,
    path,
    occurrence: saved.occurrences[0]!
  };
}

export function buildProceduralPolicyPath(input: {
  suffix: string;
  episodeId: string;
  rawTurnId: string;
  terminationStatus: ProceduralSpanTermination;
}): EpisodeProceduralPathV2 {
  const sourceSnapshotHash = `procedural-policy-snapshot-${input.suffix}`;
  const provenance = {
    algorithmVersion: "episode-procedural-reconstruction.v2",
    model: "fixture-model",
    sourceSnapshotHash
  };
  const initial = emptyObservedState();
  const diagnosed = applyStateDelta(initial, [{
    op: "issue.upsert",
    subject: "dependency conflict",
    status: "diagnosed",
    sourceRefs: [`tool:${input.suffix}:inspect`]
  }]);
  const terminal = input.terminationStatus === "success"
    ? applyStateDelta(diagnosed, [{
        op: "issue.resolve",
        subject: "dependency conflict",
        status: "resolved",
        sourceRefs: [`tool:${input.suffix}:repair`]
      }, {
        op: "verification.set",
        subject: "focused build",
        status: "passed",
        sourceRefs: [`tool:${input.suffix}:repair`]
      }])
    : applyStateDelta(diagnosed, [{
        op: "issue.upsert",
        subject: "platform compatibility error",
        status: "blocked",
        sourceRefs: [`tool:${input.suffix}:repair`]
      }, {
        op: "verification.set",
        subject: "focused build",
        status: "failed",
        sourceRefs: [`tool:${input.suffix}:repair`]
      }]);
  const steps: ExecutionStepV1[] = [{
    id: `step-${input.suffix}-inspect`,
    schemaVersion: "execution-step.v1",
    episodeId: input.episodeId,
    rawTurnId: input.rawTurnId,
    turnIndex: 0,
    stepIndex: 0,
    preStateId: initial.id,
    action: {
      kind: "tool_action",
      type: "inspect_and_test",
      intent: "Inspect the dependency error and reproduce the focused failure",
      summary: "The focused build exposed the conflicting dependency constraint",
      eventRefs: [`tool:${input.suffix}:inspect`],
      toolName: "exec"
    },
    actionEffectDelta: [{
      op: "issue.upsert",
      subject: "dependency conflict",
      status: "diagnosed",
      sourceRefs: [`tool:${input.suffix}:inspect`]
    }],
    actionPostStateId: diagnosed.id,
    externalObservationDelta: [],
    postStateId: diagnosed.id,
    outcome: { status: "failure", evidenceRefs: [`tool:${input.suffix}:inspect`] },
    cost: { toolCalls: 1, errorCount: 1 },
    provenance
  }, {
    id: `step-${input.suffix}-repair`,
    schemaVersion: "execution-step.v1",
    episodeId: input.episodeId,
    rawTurnId: input.rawTurnId,
    turnIndex: 0,
    stepIndex: 1,
    preStateId: diagnosed.id,
    action: {
      kind: "tool_action",
      type: "repair_and_test",
      intent: "Correct the dependency constraint and rerun the focused build",
      summary: input.terminationStatus === "success"
        ? "The corrected constraint passed the focused build"
        : "The correction exposed a platform compatibility blocker",
      eventRefs: [`tool:${input.suffix}:repair`],
      toolName: "exec"
    },
    actionEffectDelta: input.terminationStatus === "success" ? [{
      op: "issue.resolve",
      subject: "dependency conflict",
      status: "resolved",
      sourceRefs: [`tool:${input.suffix}:repair`]
    }, {
      op: "verification.set",
      subject: "focused build",
      status: "passed",
      sourceRefs: [`tool:${input.suffix}:repair`]
    }] : [{
      op: "issue.upsert",
      subject: "platform compatibility error",
      status: "blocked",
      sourceRefs: [`tool:${input.suffix}:repair`]
    }, {
      op: "verification.set",
      subject: "focused build",
      status: "failed",
      sourceRefs: [`tool:${input.suffix}:repair`]
    }],
    actionPostStateId: terminal.id,
    externalObservationDelta: [],
    postStateId: terminal.id,
    outcome: {
      status: input.terminationStatus === "success" ? "success" : "failure",
      evidenceRefs: [`tool:${input.suffix}:repair`]
    },
    recoveryFromStepId: `step-${input.suffix}-inspect`,
    cost: {
      toolCalls: 1,
      errorCount: input.terminationStatus === "success" ? 0 : 1
    },
    provenance
  }];
  const exitCondition = input.terminationStatus === "success"
    ? "The corrected dependency passes the focused build"
    : "The attempted correction remains blocked by platform compatibility";
  const decision: SpanSegmentationDecisionV1 = {
    spanIndex: 0,
    stepIds: steps.map((step) => step.id),
    localGoal: "Resolve a dependency failure",
    capabilityGoal: "Diagnose, repair, and verify a dependency constraint failure",
    entryCondition: "A focused build fails because dependency constraints conflict",
    exitCondition,
    terminationStatus: input.terminationStatus,
    evidenceRefs: steps.flatMap((step) => step.outcome.evidenceRefs),
    reason: "One diagnostic-to-repair loop with an explicit verification result",
    confidence: 0.98
  };
  const span: ProceduralSpanV1 = {
    id: `procedural-span-${input.suffix}`,
    schemaVersion: "procedural-span.v1",
    episodeId: input.episodeId,
    spanIndex: 0,
    localGoal: decision.localGoal,
    capabilityGoal: decision.capabilityGoal,
    entryCondition: decision.entryCondition,
    stepIds: [...decision.stepIds],
    rawTurnIds: [input.rawTurnId],
    preStateId: initial.id,
    postStateId: terminal.id,
    termination: {
      status: input.terminationStatus,
      exitCondition,
      evidenceRefs: [...decision.evidenceRefs]
    },
    cost: {
      steps: 2,
      toolCalls: 2,
      retryCount: 0,
      recoveryCount: 1,
      errorCount: input.terminationStatus === "success" ? 1 : 2
    },
    segmentation: { reason: decision.reason, confidence: decision.confidence },
    provenance
  };
  return buildEpisodeProceduralPath({
    episodeId: input.episodeId,
    states: [initial, diagnosed, terminal],
    steps,
    spans: [span],
    segmentationDecisions: [decision],
    sourceSnapshotHash,
    terminalReward: input.terminationStatus === "success" ? 1 : 0
  });
}
