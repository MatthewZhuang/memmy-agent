import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  MemoryDb,
  EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION,
  EpisodeProceduralPathPersistencePipeline,
  PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSION,
  PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
  applyStateDelta,
  buildEpisodeProceduralPath,
  emptyObservedState,
  type EpisodeProceduralPathV2,
  type ExecutionStepV1,
  type ProceduralSpanV1,
  type SpanSegmentationDecisionV1
} from "../../src/index.js";
import { Repositories, type EvolutionJobRecord } from "../../src/storage/repositories.js";
import type { EnqueueJobInput } from "../../src/service/worker/job-handlers.js";

const ALGORITHM_VERSION = EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION;
const MODEL = "gpt-5.6-terra";
const NAMESPACE_ID = "tenant:user:project:codex:default";
const SESSION_NAMESPACE_ID = "user-a";

describe("repository procedural path persistence", () => {
  it("persists an active Episode path and cluster-ready Span occurrence idempotently", () => {
    const root = mkdtempSync(join(tmpdir(), "memmy-procedural-path-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);
      seedEpisode(repos, "episode-a");
      const path = proceduralPath("episode-a", "a");

      const first = repos.proceduralPaths.save({
        path,
        namespaceId: NAMESPACE_ID,
        createdAt: "2026-08-20T00:00:00.000Z"
      });
      const replay = repos.proceduralPaths.save({
        path,
        namespaceId: NAMESPACE_ID,
        createdAt: "2026-08-20T00:00:01.000Z"
      });

      expect(first.created).toBe(true);
      expect(replay.created).toBe(false);
      expect(first.record).toMatchObject({
        id: path.id,
        episodeId: "episode-a",
        userId: "user-a",
        sessionId: "session-a",
        namespaceId: NAMESPACE_ID,
        status: "active",
        reconstructionAlgorithmVersion: ALGORITHM_VERSION,
        reconstructionModel: MODEL,
        pathHash: path.pathHash
      });
      expect(repos.proceduralPaths.listVersionsForEpisode("episode-a")).toHaveLength(1);
      expect(first.occurrences).toHaveLength(1);
      expect(first.occurrences[0]).toMatchObject({
        pathId: path.id,
        pathHash: path.pathHash,
        spanId: path.spans[0]!.id,
        schemaVersion: PROCEDURAL_SPAN_OCCURRENCE_SCHEMA_VERSION,
        stepIds: path.spans[0]!.stepIds,
        preStateId: path.spans[0]!.preStateId,
        postStateId: path.spans[0]!.postStateId,
        projection: {
          version: PROCEDURAL_SPAN_CLUSTER_PROJECTION_VERSION,
          goalText: "Resolve a dependency failure",
          structureSignature: expect.stringMatching(/^[a-f0-9]{64}$/)
        }
      });
      expect(first.occurrences[0]!.projection.procedureText)
        .toContain("Correct the dependency constraint and rerun tests");
      expect(first.occurrences[0]!.projection.procedureText).not.toContain("tool=");
      expect(first.occurrences[0]!.projection.procedureText).not.toContain("recovery_from=");
      expect(first.occurrences[0]!.projection.conditionText).toBe(
        "Entry: Tests fail because dependency constraints conflict " +
        "Exit: The corrected dependency passes the test suite"
      );
      expect(first.occurrences[0]!.projection.conditionText).not.toContain("Pre-state:");
      expect(first.occurrences[0]!.projection.effectText).toContain("Termination: success");
      expect(repos.proceduralPaths.listActiveOccurrencesForNamespace(NAMESPACE_ID)
        .map((occurrence) => occurrence.id)).toEqual([first.occurrences[0]!.id]);
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("keeps immutable versions and rolls the active Episode path backward without deleting evidence", () => {
    const root = mkdtempSync(join(tmpdir(), "memmy-procedural-path-versions-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);
      seedEpisode(repos, "episode-a");
      const firstPath = proceduralPath("episode-a", "first");
      const secondPath = proceduralPath("episode-a", "second");
      const first = repos.proceduralPaths.save({
        path: firstPath,
        namespaceId: NAMESPACE_ID,
        createdAt: "2026-08-20T00:00:00.000Z"
      });
      const second = repos.proceduralPaths.save({
        path: secondPath,
        namespaceId: NAMESPACE_ID,
        createdAt: "2026-08-20T00:00:01.000Z"
      });

      expect(repos.proceduralPaths.get(firstPath.id)).toMatchObject({
        status: "inactive",
        deactivatedAt: "2026-08-20T00:00:01.000Z"
      });
      expect(repos.proceduralPaths.getActiveForEpisode("episode-a")?.id).toBe(secondPath.id);
      expect(repos.proceduralPaths.listActiveOccurrencesForNamespace(NAMESPACE_ID)
        .map((occurrence) => occurrence.pathId)).toEqual([secondPath.id]);

      repos.proceduralPaths.activateVersion(firstPath.id, "2026-08-20T00:00:02.000Z");

      expect(repos.proceduralPaths.getActiveForEpisode("episode-a")?.id).toBe(firstPath.id);
      expect(repos.proceduralPaths.get(secondPath.id)).toMatchObject({
        status: "inactive",
        deactivatedAt: "2026-08-20T00:00:02.000Z"
      });
      expect(repos.proceduralPaths.listVersionsForEpisode("episode-a")).toHaveLength(2);
      expect(repos.proceduralPaths.listOccurrencesForPath(firstPath.id)).toHaveLength(1);
      expect(repos.proceduralPaths.listOccurrencesForPath(secondPath.id)).toHaveLength(1);
      expect(first.occurrences[0]!.projection.structureSignature)
        .toBe(second.occurrences[0]!.projection.structureSignature);
      expect(repos.proceduralPaths.listActiveOccurrencesForNamespace(NAMESPACE_ID)
        .map((occurrence) => occurrence.pathId)).toEqual([firstPath.id]);
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects tampered paths and paths without a persisted source Episode", () => {
    const root = mkdtempSync(join(tmpdir(), "memmy-procedural-path-invalid-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);
      const path = proceduralPath("episode-a", "invalid");
      expect(() => repos.proceduralPaths.save({
        path: { ...path, pathHash: "tampered" },
        namespaceId: NAMESPACE_ID,
        createdAt: "2026-08-20T00:00:00.000Z"
      })).toThrow(/integrity check failed/);
      expect(() => repos.proceduralPaths.save({
        path,
        namespaceId: NAMESPACE_ID,
        createdAt: "2026-08-20T00:00:00.000Z"
      })).toThrow(/source episode not found/);
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("reconstructs a closed Episode from persisted RawTurns and activates the result", async () => {
    const root = mkdtempSync(join(tmpdir(), "memmy-procedural-path-pipeline-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);
      seedEpisode(repos, "episode-a");
      const path = proceduralPath("episode-a", "pipeline");
      const calls: Array<{ episodeId: string; rawTurnIds: string[]; terminalReward?: number }> = [];
      const queued: EnqueueJobInput[] = [];
      const pipeline = new EpisodeProceduralPathPersistencePipeline({
        repos,
        reconstructor: {
          async reconstruct(input) {
            calls.push({
              episodeId: input.episodeId,
              rawTurnIds: input.rawTurns.map((turn) => turn.id),
              ...(input.terminalReward === undefined ? {} : { terminalReward: input.terminalReward })
            });
            return path;
          }
        },
        enqueueJob(input) {
          queued.push(input);
          const at = input.createdAt ?? "2026-08-20T00:00:01.000Z";
          return {
            id: "job-span-credit-after-path",
            jobType: input.jobType,
            status: "queued",
            dedupeKey: "job-span-credit-after-path",
            userId: input.userId,
            sessionId: input.sessionId,
            episodeId: input.episodeId,
            targetMemoryId: input.targetMemoryId,
            payload: input.payload ?? {},
            attempts: 0,
            maxAttempts: input.maxAttempts ?? 3,
            createdAt: at,
            updatedAt: at
          };
        }
      });

      const result = await pipeline.reconstructAndPersist({
        episodeId: "episode-a",
        createdAt: "2026-08-20T00:00:01.000Z"
      });

      expect(calls).toEqual([{
        episodeId: "episode-a",
        rawTurnIds: ["turn-a"],
        terminalReward: 1
      }]);
      expect(result.record).toMatchObject({
        id: path.id,
        namespaceId: SESSION_NAMESPACE_ID,
        status: "active"
      });
      expect(repos.proceduralPaths.listActiveOccurrencesForNamespace(SESSION_NAMESPACE_ID))
        .toHaveLength(1);
      expect(queued).toHaveLength(2);
      expect(queued).toEqual(expect.arrayContaining([
        expect.objectContaining({
          jobType: "span_credit",
          userId: "user-a",
          sessionId: "session-a",
          episodeId: "episode-a",
          payload: {
            pathId: path.id,
            pathHash: path.pathHash,
            rewardHash: expect.stringMatching(/^[a-f0-9]{64}$/)
          }
        }),
        expect.objectContaining({
          jobType: "episode_policy_projection",
          userId: "user-a",
          sessionId: "session-a",
          episodeId: "episode-a",
          payload: expect.objectContaining({
            pathId: path.id,
            pathHash: path.pathHash,
            trigger: "procedural_path_activated"
          })
        })
      ]));
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("reconstructs instead of reusing an active path from an older algorithm version", async () => {
    const root = mkdtempSync(join(tmpdir(), "memmy-procedural-path-version-refresh-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);
      seedEpisode(repos, "episode-a");
      const legacyPath = proceduralPath(
        "episode-a",
        "legacy",
        "episode-procedural-reconstruction.v2"
      );
      repos.proceduralPaths.save({
        path: legacyPath,
        namespaceId: SESSION_NAMESPACE_ID,
        createdAt: "2026-08-20T00:00:00.000Z"
      });
      const refreshedPath = proceduralPath("episode-a", "refreshed");
      let reconstructCalls = 0;
      const pipeline = new EpisodeProceduralPathPersistencePipeline({
        repos,
        reconstructor: {
          async reconstruct() {
            reconstructCalls += 1;
            return refreshedPath;
          }
        }
      });
      const job: EvolutionJobRecord = {
        id: "job-refresh-procedural-path",
        jobType: "procedural_path",
        status: "leased",
        userId: "user-a",
        sessionId: "session-a",
        episodeId: "episode-a",
        payload: {},
        attempts: 1,
        maxAttempts: 3,
        createdAt: "2026-08-20T00:00:01.000Z",
        updatedAt: "2026-08-20T00:00:01.000Z"
      };

      const result = await pipeline.reconstructJob(job);

      expect(reconstructCalls).toBe(1);
      expect(result?.record).toMatchObject({
        id: refreshedPath.id,
        status: "active",
        reconstructionAlgorithmVersion: EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION
      });
      expect(repos.proceduralPaths.get(legacyPath.id)).toMatchObject({ status: "inactive" });
      expect(repos.proceduralPaths.listVersionsForEpisode("episode-a")).toHaveLength(2);
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

function seedEpisode(repos: Repositories, episodeId: string): void {
  const now = "2026-08-20T00:00:00.000Z";
  repos.runtime.createSession({
    id: "session-a",
    userId: "user-a",
    source: "codex",
    profileId: "default",
    projectId: "project-a",
    status: "closed",
    meta: {},
    openedAt: now,
    lastSeenAt: now,
    closedAt: now,
    updatedAt: now
  });
  repos.runtime.createEpisode({
    id: episodeId,
    sessionId: "session-a",
    userId: "user-a",
    projectId: "project-a",
    status: "closed",
    l1MemoryIds: [],
    rawTurnIds: ["turn-a"],
    feedbackIds: [],
    decisionRepairIds: [],
    l2PolicyIds: [],
    l3WorldModelIds: [],
    skillMemoryIds: [],
    turnCount: 1,
    rTask: 1,
    rewardDetail: {},
    pipelineStatus: "succeeded",
    meta: {},
    openedAt: now,
    closedAt: now,
    updatedAt: now
  });
  repos.runtime.insertRawTurn({
    id: "turn-a",
    sessionId: "session-a",
    episodeId,
    turnId: "host-turn-a",
    userId: "user-a",
    userText: "Resolve the dependency failure",
    assistantText: "The dependency was corrected and tests passed.",
    toolCalls: [],
    toolResults: [],
    sourceMemoryIds: [],
    usage: {},
    status: "succeeded",
    createdAt: now
  });
}

function proceduralPath(
  episodeId: string,
  suffix: string,
  algorithmVersion: string = ALGORITHM_VERSION
): EpisodeProceduralPathV2 {
  const sourceSnapshotHash = `snapshot-${suffix}`;
  const provenance = {
    algorithmVersion,
    model: MODEL,
    sourceSnapshotHash
  };
  const initial = emptyObservedState();
  const diagnosed = applyStateDelta(initial, [{
    op: "issue.upsert",
    subject: "dependency conflict",
    status: "diagnosed",
    sourceRefs: [`tool:${suffix}:0`]
  }]);
  const resolved = applyStateDelta(diagnosed, [{
    op: "issue.resolve",
    subject: "dependency conflict",
    status: "resolved",
    sourceRefs: [`tool:${suffix}:1`]
  }, {
    op: "verification.set",
    subject: "test suite",
    status: "passed",
    sourceRefs: [`tool:${suffix}:1`]
  }]);
  const steps: ExecutionStepV1[] = [{
    id: `step-${suffix}-0`,
    schemaVersion: "execution-step.v1",
    episodeId,
    rawTurnId: "turn-a",
    turnIndex: 0,
    stepIndex: 0,
    preStateId: initial.id,
    action: {
      kind: "tool_action",
      type: "test",
      intent: "Verify the candidate dependency fix",
      summary: "The first candidate still failed",
      eventRefs: [`tool:${suffix}:0`],
      toolName: "exec"
    },
    actionEffectDelta: [{
      op: "issue.upsert",
      subject: "dependency conflict",
      status: "diagnosed",
      sourceRefs: [`tool:${suffix}:0`]
    }],
    actionPostStateId: diagnosed.id,
    externalObservationDelta: [],
    postStateId: diagnosed.id,
    outcome: { status: "failure", evidenceRefs: [`tool:${suffix}:0`] },
    cost: { toolCalls: 1, errorCount: 1 },
    provenance
  }, {
    id: `step-${suffix}-1`,
    schemaVersion: "execution-step.v1",
    episodeId,
    rawTurnId: "turn-a",
    turnIndex: 0,
    stepIndex: 1,
    preStateId: diagnosed.id,
    action: {
      kind: "tool_action",
      type: "repair_and_test",
      intent: "Correct the dependency constraint and rerun tests",
      summary: "The corrected constraint passed verification",
      eventRefs: [`tool:${suffix}:1`],
      toolName: "exec"
    },
    actionEffectDelta: [{
      op: "issue.resolve",
      subject: "dependency conflict",
      status: "resolved",
      sourceRefs: [`tool:${suffix}:1`]
    }, {
      op: "verification.set",
      subject: "test suite",
      status: "passed",
      sourceRefs: [`tool:${suffix}:1`]
    }],
    actionPostStateId: resolved.id,
    externalObservationDelta: [],
    postStateId: resolved.id,
    outcome: { status: "success", evidenceRefs: [`tool:${suffix}:1`] },
    recoveryFromStepId: `step-${suffix}-0`,
    cost: { toolCalls: 1, errorCount: 0 },
    provenance
  }];
  const decision: SpanSegmentationDecisionV1 = {
    spanIndex: 0,
    stepIds: steps.map((step) => step.id),
    localGoal: "Resolve a dependency failure",
    entryCondition: "Tests fail because dependency constraints conflict",
    exitCondition: "The corrected dependency passes the test suite",
    terminationStatus: "success",
    evidenceRefs: steps.flatMap((step) => step.outcome.evidenceRefs),
    reason: "One diagnosis, failed attempt, recovery, and verified fix",
    confidence: 0.99
  };
  const span: ProceduralSpanV1 = {
    id: `procedural-span-${suffix}`,
    schemaVersion: "procedural-span.v1",
    episodeId,
    spanIndex: 0,
    localGoal: decision.localGoal,
    entryCondition: decision.entryCondition,
    stepIds: [...decision.stepIds],
    rawTurnIds: ["turn-a"],
    preStateId: initial.id,
    postStateId: resolved.id,
    termination: {
      status: "success",
      exitCondition: decision.exitCondition,
      evidenceRefs: [...decision.evidenceRefs]
    },
    cost: {
      steps: 2,
      toolCalls: 2,
      retryCount: 0,
      recoveryCount: 1,
      errorCount: 1
    },
    segmentation: { reason: decision.reason, confidence: decision.confidence },
    provenance
  };
  return buildEpisodeProceduralPath({
    episodeId,
    states: [initial, diagnosed, resolved],
    steps,
    spans: [span],
    segmentationDecisions: [decision],
    sourceSnapshotHash,
    terminalReward: 1
  });
}
