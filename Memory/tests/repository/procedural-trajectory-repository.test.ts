import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { MemoryDb } from "../../src/storage/db.js";
import {
  EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
  EXECUTION_STEP_LITE_SCHEMA_VERSION,
  TURN_TRANSITION_SCHEMA_VERSION,
  buildTrajectoryWindows,
  type EpisodeExecutionPathLiteV1,
  type ExecutionStepLiteV1,
  type TurnTransitionV1
} from "../../src/service/evolution/procedural-window-model.js";
import {
  ProceduralTrajectoryCasError,
  type TrajectoryWindowOccurrenceRecord
} from "../../src/storage/procedural-trajectory-repository.js";
import {
  Repositories,
  type EpisodeRecord,
  type RawTurnRecord,
  type SessionRecord
} from "../../src/storage/repositories.js";
import type { MemoryRow } from "../../src/types.js";
import { stableHash } from "../../src/utils/id.js";

const AT = "2026-08-27T08:00:00.000Z";

describe("procedural trajectory repository", () => {
  it("persists one lightweight cross-Turn path and deterministic 5-Step windows", () => {
    withRepositories(({ repos }) => {
      seedEpisode(repos, {
        episodeId: "episode_path_a",
        sessionId: "session_path_a",
        userId: "user-a",
        source: "codex",
        rawTurnIds: ["raw_path_a_1", "raw_path_a_2"]
      });
      const steps = executionSteps(
        "episode_path_a",
        "raw_path_a_1",
        "raw_path_a_2",
        "snapshot-a-v1"
      );
      const path = executionPath({
        episodeId: "episode_path_a",
        userId: "user-a",
        sourceRawTurnIds: ["raw_path_a_1", "raw_path_a_2"],
        sourceSnapshotHash: "snapshot-a-v1",
        steps,
        turnTransitions: [{
          id: "turn_transition_path_a_2",
          schemaVersion: TURN_TRANSITION_SCHEMA_VERSION,
          episodeId: "episode_path_a",
          turnIndex: 1,
          rawTurnId: "raw_path_a_2",
          beforeStepIndex: 2,
          afterStepIndex: 3,
          userObservation: "继续修复并验证",
          sourceRef: "raw_turn:raw_path_a_2:user"
        }]
      });
      const input = {
        path,
        sourceAgentIds: ["codex"],
        createdAt: AT
      };

      const first = repos.proceduralTrajectory.savePathVersion(input);
      const duplicate = repos.proceduralTrajectory.savePathVersion(input);
      expect(first.created).toBe(true);
      expect(duplicate).toMatchObject({ created: false, record: { id: first.record.id } });
      expect(first.record.path.steps).toHaveLength(6);
      expect(first.record.path.turnTransitions).toEqual([expect.objectContaining({
        afterStepIndex: 3,
        rawTurnId: "raw_path_a_2"
      })]);

      const embedding = repos.proceduralTrajectory.upsertStepEmbedding({
        pathId: first.record.id,
        stepId: steps[0]!.id,
        stepIndex: 0,
        embeddingSignature: "embed/model-a:3",
        semanticHash: "step-semantic-0",
        vector: [1, 0, 0],
        createdAt: AT
      });
      const duplicateEmbedding = repos.proceduralTrajectory.upsertStepEmbedding({
        pathId: first.record.id,
        stepId: steps[0]!.id,
        stepIndex: 0,
        embeddingSignature: "embed/model-a:3",
        semanticHash: "step-semantic-0",
        vector: [0.999, 0.001, 0],
        createdAt: AT
      });
      expect(embedding.created).toBe(true);
      expect(duplicateEmbedding).toMatchObject({
        created: false,
        record: { id: embedding.record.id, vector: [1, 0, 0] }
      });

      const windowInput = {
        occurrence: buildTrajectoryWindows(
          [first.record.path],
          [{ length: 5, stride: 2 }]
        ).find((item) => item.startStepIndex === 1)!,
        windowConfigHash: "window-5-2-config",
        coarseRepresentationVersion: "window-intent-sequence.v1",
        embeddingSignature: "embed/model-a:3",
        coarseVector: [0.8, 0.1, 0.1],
        createdAt: AT
      };
      const window = repos.proceduralTrajectory.insertWindow(windowInput);
      const duplicateWindow = repos.proceduralTrajectory.insertWindow(windowInput);
      expect(window.created).toBe(true);
      expect(duplicateWindow).toMatchObject({ created: false, record: { id: window.record.id } });
      expect(window.record).toMatchObject({
        startStepIndex: 1,
        endStepIndex: 5,
        rawTurnIds: ["raw_path_a_1", "raw_path_a_2"]
      });

      const replacement = repos.proceduralTrajectory.savePathVersion({
        ...input,
        path: executionPath({
          episodeId: "episode_path_a",
          userId: "user-a",
          sourceRawTurnIds: ["raw_path_a_1", "raw_path_a_2"],
          sourceSnapshotHash: "snapshot-a-v2",
          steps: executionSteps(
            "episode_path_a",
            "raw_path_a_1",
            "raw_path_a_2",
            "snapshot-a-v2"
          ),
          turnTransitions: path.turnTransitions
        }),
        createdAt: "2026-08-27T08:01:00.000Z"
      });
      expect(replacement.created).toBe(true);
      expect(replacement.previousActive?.id).toBe(first.record.id);
      expect(repos.proceduralTrajectory.getPath(first.record.id)?.status).toBe("superseded");

      const rollback = repos.proceduralTrajectory.activatePathVersion(
        first.record.id,
        "2026-08-27T08:02:00.000Z"
      );
      expect(rollback.record.status).toBe("active");
      expect(rollback.previousActive?.id).toBe(replacement.record.id);
      expect(repos.proceduralTrajectory.getActivePath("episode_path_a")?.id)
        .toBe(first.record.id);
    });
  });

  it("rejects a stale compile result that returns after a newer Path is active", () => {
    withRepositories(({ repos }) => {
      const episodeId = "episode_path_race_stale";
      const rawTurnIds = ["raw_path_race_stale_1", "raw_path_race_stale_2"];
      seedEpisode(repos, {
        episodeId,
        sessionId: "session_path_race_stale",
        userId: "path-race-user",
        source: "codex",
        rawTurnIds
      });
      const basePath = testExecutionPath(
        episodeId,
        "path-race-user",
        rawTurnIds,
        "snapshot-race-base"
      );
      const base = repos.proceduralTrajectory.savePathVersion({
        path: basePath,
        expectedActivePathId: null,
        createdAt: AT
      }).record;
      const newerPath = testExecutionPath(
        episodeId,
        "path-race-user",
        rawTurnIds,
        "snapshot-race-newer"
      );
      const stalePath = testExecutionPath(
        episodeId,
        "path-race-user",
        rawTurnIds,
        "snapshot-race-stale"
      );

      const newer = repos.proceduralTrajectory.savePathVersion({
        path: newerPath,
        expectedActivePathId: base.id,
        createdAt: "2026-08-27T08:20:00.000Z"
      }).record;
      expect(() => repos.proceduralTrajectory.savePathVersion({
        path: stalePath,
        expectedActivePathId: base.id,
        createdAt: "2026-08-27T08:21:00.000Z"
      })).toThrow(ProceduralTrajectoryCasError);

      expect(repos.proceduralTrajectory.getActivePath(episodeId)?.id).toBe(newer.id);
      expect(repos.proceduralTrajectory.getPath(stalePath.id)).toBeUndefined();
      expect(repos.proceduralTrajectory.getPath(base.id)?.status).toBe("superseded");
    });
  });

  it("lets a newer compile normally replace the currently observed active Path", () => {
    withRepositories(({ repos }) => {
      const episodeId = "episode_path_race_ordered";
      const rawTurnIds = ["raw_path_race_ordered_1", "raw_path_race_ordered_2"];
      seedEpisode(repos, {
        episodeId,
        sessionId: "session_path_race_ordered",
        userId: "path-race-user",
        source: "codex",
        rawTurnIds
      });
      const base = repos.proceduralTrajectory.savePathVersion({
        path: testExecutionPath(
          episodeId,
          "path-race-user",
          rawTurnIds,
          "snapshot-ordered-base"
        ),
        expectedActivePathId: null,
        createdAt: AT
      }).record;
      const olderCompile = repos.proceduralTrajectory.savePathVersion({
        path: testExecutionPath(
          episodeId,
          "path-race-user",
          rawTurnIds,
          "snapshot-ordered-old-compile"
        ),
        expectedActivePathId: base.id,
        createdAt: "2026-08-27T08:22:00.000Z"
      }).record;
      const newerCompile = repos.proceduralTrajectory.savePathVersion({
        path: testExecutionPath(
          episodeId,
          "path-race-user",
          rawTurnIds,
          "snapshot-ordered-new-compile"
        ),
        expectedActivePathId: olderCompile.id,
        createdAt: "2026-08-27T08:23:00.000Z"
      }).record;

      expect(repos.proceduralTrajectory.getActivePath(episodeId)?.id).toBe(newerCompile.id);
      expect(repos.proceduralTrajectory.getPath(olderCompile.id)?.status).toBe("superseded");
      expect(repos.proceduralTrajectory.listPathVersions(episodeId)).toHaveLength(3);
    });
  });

  it("versions overlapping Families and canonicalizes identical Fine evidence once", () => {
    withRepositories(({ repos, db }) => {
      const userId = "family-canonical-user";
      const anchorA = seedWindow(repos, {
        userId, source: "codex", suffix: "family-anchor-a", vector: [1, 0, 0]
      });
      const anchorB = seedWindow(repos, {
        userId, source: "cursor", suffix: "family-anchor-b", vector: [0, 1, 0]
      });
      const bridgeC = seedWindow(repos, {
        userId, source: "claude_code", suffix: "family-bridge-c", vector: [0.7, 0.7, 0]
      });
      const bridgeD = seedWindow(repos, {
        userId, source: "gemini", suffix: "family-bridge-d", vector: [0.7, 0.7, 0]
      });
      const later = seedWindow(repos, {
        userId, source: "codex", suffix: "family-later", vector: [0.8, 0.6, 0]
      });
      const familyInput = {
        userId,
        scale: 5,
        algorithmVersion: "multi-scale-window.v15",
        configHash: "v15-family-config",
        embeddingSignature: "embed/model-a:3",
        createdAt: AT
      };
      const familyA = repos.proceduralTrajectory.createFamilyHead({
        ...familyInput,
        seedOccurrenceId: anchorA.id
      }).record;
      const familyB = repos.proceduralTrajectory.createFamilyHead({
        ...familyInput,
        seedOccurrenceId: anchorB.id
      }).record;
      const membersA = [anchorA, bridgeC, bridgeD].map((occurrence) => ({
        occurrenceId: occurrence.id,
        coarseSimilarity: occurrence.id === anchorA.id ? 0.77 : 1
      }));
      const membersB = [anchorB, bridgeC, bridgeD].map((occurrence) => ({
        occurrenceId: occurrence.id,
        coarseSimilarity: occurrence.id === anchorB.id ? 0.77 : 1
      }));
      const revisionA1 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyA.id,
        expectedActiveRevisionId: null,
        medoidOccurrenceId: bridgeC.id,
        members: membersA,
        evidenceHash: "evidence-family-a-v1",
        createdAt: AT
      });
      const revisionB1 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyB.id,
        expectedActiveRevisionId: null,
        medoidOccurrenceId: bridgeD.id,
        members: membersB,
        evidenceHash: "evidence-family-b-v1",
        createdAt: AT
      });

      // The same Window occurrence may be recalled into every threshold-valid Family.
      expect(repos.proceduralTrajectory.listAffectedFamilyIdsForOccurrences([bridgeC.id]))
        .toEqual([familyA.id, familyB.id].sort());
      // Family identity is set-like: replay order cannot change its revision signature.
      expect(repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyA.id,
        expectedActiveRevisionId: revisionA1.record.id,
        medoidOccurrenceId: bridgeC.id,
        members: [...membersA].reverse(),
        evidenceHash: "evidence-family-a-v1",
        createdAt: "2026-08-27T08:01:00.000Z"
      })).toMatchObject({
        created: false,
        reactivated: false,
        record: { id: revisionA1.record.id, membershipHash: revisionA1.record.membershipHash }
      });

      const evidenceSignature = stableHash({
        scale: 5,
        occurrenceIds: [bridgeC.id, bridgeD.id].sort()
      });
      const canonicalA = repos.proceduralTrajectory.resolveCanonicalClusterHead({
        ...familyInput,
        evidenceSignature,
        seedOccurrenceId: bridgeC.id
      });
      const canonicalB = repos.proceduralTrajectory.resolveCanonicalClusterHead({
        ...familyInput,
        evidenceSignature,
        seedOccurrenceId: bridgeD.id,
        createdAt: "2026-08-27T08:02:00.000Z"
      });
      expect(canonicalA.created).toBe(true);
      expect(canonicalB).toMatchObject({
        created: false,
        cluster: { id: canonicalA.cluster.id },
        canonicalKey: { id: canonicalA.canonicalKey.id }
      });

      const clusterVersion = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: canonicalA.cluster.id,
        expectedActiveVersionId: null,
        medoidOccurrenceId: bridgeC.id,
        members: [
          supportMember(bridgeC.id, "reward-family-c"),
          supportMember(bridgeD.id, "reward-family-d")
        ],
        createdAt: "2026-08-27T08:03:00.000Z"
      }).record;
      for (const revision of [revisionA1.record, revisionB1.record]) {
        repos.proceduralTrajectory.linkFamilyRevisionToCluster({
          familyRevisionId: revision.id,
          canonicalKeyId: canonicalA.canonicalKey.id,
          clusterVersionId: clusterVersion.id,
          createdAt: "2026-08-27T08:04:00.000Z"
        });
      }
      expect(repos.proceduralTrajectory.listClusterFamilyLinks(canonicalA.cluster.id))
        .toHaveLength(2);
      expect((db.db.prepare(
        `SELECT COUNT(*) AS count FROM trajectory_window_cluster_canonical_keys
         WHERE evidence_signature = ?`
      ).get(evidenceSignature) as { count: number }).count).toBe(1);

      // A later Episode only revises the Family it actually matched.
      const revisionA2 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyA.id,
        expectedActiveRevisionId: revisionA1.record.id,
        medoidOccurrenceId: bridgeC.id,
        members: [...membersA, { occurrenceId: later.id, coarseSimilarity: 0.98 }],
        evidenceHash: "evidence-family-a-v2",
        createdAt: "2026-08-27T08:05:00.000Z"
      }).record;
      expect(repos.proceduralTrajectory.getFamilyHead(familyB.id)?.activeRevisionId)
        .toBe(revisionB1.record.id);
      expect(repos.proceduralTrajectory.listAffectedFamilyIdsForOccurrences([later.id]))
        .toEqual([familyA.id]);

      // Reward drift keeps structural membership but advances the immutable
      // evidence snapshot, so a new cluster version can link without rewriting
      // the historical Family revision.
      const bridgeDPath = repos.proceduralTrajectory.getPath(bridgeD.pathId)!;
      repos.runtime.updateEpisodeReward(bridgeD.episodeId, {
        rTask: -0.9,
        rewardDetail: { phase: "revised", reason: "post-hoc failure" }
      }, "2026-08-27T08:06:00.000Z");
      const refreshedD = buildTrajectoryWindows(
        [bridgeDPath.path],
        [{ length: 5, stride: 2 }]
      ).find((occurrence) => occurrence.id === bridgeD.id)!;
      repos.proceduralTrajectory.insertWindow({
        occurrence: { ...refreshedD, terminalReward: -0.9, evidenceRole: "counterexample" },
        windowConfigHash: bridgeD.windowConfigHash,
        coarseRepresentationVersion: bridgeD.coarseRepresentationVersion,
        embeddingSignature: bridgeD.embeddingSignature,
        createdAt: "2026-08-27T08:06:00.000Z"
      });
      const revisionB2 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyB.id,
        expectedActiveRevisionId: revisionB1.record.id,
        medoidOccurrenceId: bridgeD.id,
        members: membersB,
        evidenceHash: "evidence-family-b-v2-negative",
        createdAt: "2026-08-27T08:06:01.000Z"
      });
      expect(revisionB2.record).toMatchObject({
        revisionNo: 2,
        evidenceHash: "evidence-family-b-v2-negative"
      });
      expect(revisionB2.record.membershipHash).not.toBe(revisionB1.record.membershipHash);
      const clusterVersion2 = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: canonicalA.cluster.id,
        expectedActiveVersionId: clusterVersion.id,
        medoidOccurrenceId: bridgeC.id,
        members: [
          supportMember(bridgeC.id, "reward-family-c"),
          {
            occurrenceId: bridgeD.id,
            rewardHash: "reward-family-d-negative",
            coarseSimilarity: 0.9
          }
        ],
        createdAt: "2026-08-27T08:06:02.000Z"
      }).record;
      expect(repos.proceduralTrajectory.linkFamilyRevisionToCluster({
        familyRevisionId: revisionB2.record.id,
        canonicalKeyId: canonicalA.canonicalKey.id,
        clusterVersionId: clusterVersion2.id,
        createdAt: "2026-08-27T08:06:03.000Z"
      }).created).toBe(true);
      expect(repos.proceduralTrajectory.getFamilyHead(familyA.id)?.activeRevisionId)
        .toBe(revisionA2.id);
      expect(repos.proceduralTrajectory.getFamilyHead(familyB.id)?.activeRevisionId)
        .toBe(revisionB2.record.id);

      // Once a source Path is superseded, a historical Family revision that
      // includes its Window cannot be reactivated.
      const bridgeCPath = repos.proceduralTrajectory.getPath(bridgeC.pathId)!;
      repos.proceduralTrajectory.savePathVersion({
        path: testExecutionPath(
          bridgeC.episodeId,
          userId,
          bridgeCPath.sourceRawTurnIds,
          "snapshot-family-bridge-c-v2"
        ),
        expectedActivePathId: bridgeCPath.id,
        createdAt: "2026-08-27T08:07:00.000Z"
      });
      expect(repos.proceduralTrajectory.listAffectedFamilyIdsForPath(bridgeCPath.id))
        .toEqual([familyA.id, familyB.id].sort());
      const revisionA3 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyA.id,
        expectedActiveRevisionId: revisionA2.id,
        medoidOccurrenceId: bridgeD.id,
        members: [anchorA, bridgeD, later].map((occurrence) => ({
          occurrenceId: occurrence.id,
          coarseSimilarity: 0.9
        })),
        evidenceHash: "evidence-family-a-v3",
        createdAt: "2026-08-27T08:08:00.000Z"
      }).record;
      expect(() => repos.proceduralTrajectory.activateFamilyRevision({
        familyId: familyA.id,
        revisionId: revisionA1.record.id,
        expectedActiveRevisionId: revisionA3.id,
        activatedAt: "2026-08-27T08:09:00.000Z"
      })).toThrow(/superseded path evidence/);
    });
  });

  it("increments medoid revisions with CAS while isolating evidence roles and users", () => {
    withRepositories(({ repos }) => {
      const windowA = seedWindow(repos, {
        userId: "shared-user",
        source: "codex",
        suffix: "a",
        vector: [1, 0, 0]
      });
      const windowB = seedWindow(repos, {
        userId: "shared-user",
        source: "claude_code",
        suffix: "b",
        vector: [0.98, 0.02, 0]
      });
      const windowFailure = seedWindow(repos, {
        userId: "shared-user",
        source: "cursor",
        suffix: "failure",
        vector: [0.96, 0.04, 0],
        terminalReward: -0.8
      });
      const otherUserWindow = seedWindow(repos, {
        userId: "other-user",
        source: "codex",
        suffix: "other",
        vector: [1, 0, 0]
      });

      const head = repos.proceduralTrajectory.createClusterHead({
        userId: "shared-user",
        scale: 5,
        algorithmVersion: "multi-scale-window.v1",
        configHash: "v15-config",
        seedOccurrenceId: windowA.id,
        createdAt: AT
      }).record;
      const initialMembers = [
        supportMember(windowA.id, "reward-a"),
        supportMember(windowB.id, "reward-b")
      ];
      const v1 = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: null,
        medoidOccurrenceId: windowA.id,
        members: initialMembers,
        metrics: { averageSimilarity: 0.91 },
        createdAt: AT
      });
      expect(v1.record).toMatchObject({
        versionNo: 1,
        supportEpisodeCount: 2,
        counterexampleEpisodeCount: 0,
        status: "active"
      });
      expect(repos.proceduralTrajectory.listActiveClusterMedoids({
        userId: "shared-user",
        scale: 5,
        algorithmVersion: "multi-scale-window.v1",
        configHash: "v15-config",
        embeddingSignature: "embed/model-a:3"
      })).toEqual([expect.objectContaining({ occurrence: expect.objectContaining({ id: windowA.id }) })]);

      const retry = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: null,
        medoidOccurrenceId: windowA.id,
        members: initialMembers,
        metrics: { averageSimilarity: 0.91 },
        createdAt: AT
      });
      expect(retry).toMatchObject({ created: false, reactivated: false, record: { id: v1.record.id } });

      const v2 = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: v1.record.id,
        medoidOccurrenceId: windowB.id,
        members: [
          ...initialMembers,
          {
            occurrenceId: windowFailure.id,
            rewardHash: "reward-failure",
            coarseSimilarity: 0.88,
            alignment: { admitted: true, matchedSteps: 4 }
          }
        ],
        createdAt: "2026-08-27T08:03:00.000Z"
      });
      expect(v2.record).toMatchObject({
        versionNo: 2,
        supportEpisodeCount: 2,
        counterexampleEpisodeCount: 1
      });
      expect(v2.record.supportHash).toBe(v1.record.supportHash);
      expect(v2.record.membershipHash).not.toBe(v1.record.membershipHash);
      expect(repos.proceduralTrajectory.listClusterMembers(v2.record.id)
        .filter((member) => member.evidenceRole === "support")).toHaveLength(2);

      expect(() => repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: v1.record.id,
        medoidOccurrenceId: windowA.id,
        members: initialMembers.slice(0, 1),
        createdAt: "2026-08-27T08:04:00.000Z"
      })).toThrow(ProceduralTrajectoryCasError);

      expect(() => repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: v2.record.id,
        medoidOccurrenceId: windowA.id,
        members: [...initialMembers, supportMember(otherUserWindow.id, "reward-other")],
        createdAt: "2026-08-27T08:04:00.000Z"
      })).toThrow(/outside cluster scope/);

      const rollback = repos.proceduralTrajectory.activateClusterVersion({
        clusterId: head.id,
        versionId: v1.record.id,
        expectedActiveVersionId: v2.record.id,
        activatedAt: "2026-08-27T08:05:00.000Z"
      });
      expect(rollback).toMatchObject({
        reactivated: true,
        record: { id: v1.record.id, status: "active" },
        previousActive: { id: v2.record.id }
      });
      expect(repos.proceduralTrajectory.getClusterVersion(v2.record.id)?.status)
        .toBe("superseded");

      expect(() => repos.proceduralTrajectory.retireCluster(
        head.id,
        v2.record.id,
        "2026-08-27T08:06:00.000Z"
      )).toThrow(ProceduralTrajectoryCasError);
      expect(repos.proceduralTrajectory.retireCluster(
        head.id,
        v1.record.id,
        "2026-08-27T08:06:00.000Z"
      )).toMatchObject({ status: "retired", activeVersionId: v1.record.id });
    });
  });

  it("versions overlapping coarse Families and canonicalizes one fine cluster across them", () => {
    withRepositories(({ repos }) => {
      const windowA = seedWindow(repos, {
        userId: "family-user",
        source: "codex",
        suffix: "family-a",
        vector: [1, 0, 0]
      });
      const windowB = seedWindow(repos, {
        userId: "family-user",
        source: "claude_code",
        suffix: "family-b",
        vector: [0.97, 0.03, 0]
      });
      const windowC = seedWindow(repos, {
        userId: "family-user",
        source: "cursor",
        suffix: "family-c",
        vector: [0.94, 0.06, 0]
      });
      const familyScope = {
        userId: "family-user",
        scale: 5,
        algorithmVersion: "overlapping-medoid-family.v1",
        configHash: "family-config-v1",
        embeddingSignature: "embed/model-a:3"
      };
      const familyA = repos.proceduralTrajectory.createFamilyHead({
        ...familyScope,
        seedOccurrenceId: windowA.id,
        createdAt: AT
      }).record;
      const familyB = repos.proceduralTrajectory.createFamilyHead({
        ...familyScope,
        seedOccurrenceId: windowB.id,
        createdAt: AT
      }).record;
      const familyARevision1 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyA.id,
        expectedActiveRevisionId: null,
        medoidOccurrenceId: windowA.id,
        members: [
          { occurrenceId: windowA.id, coarseSimilarity: 1 },
          { occurrenceId: windowB.id, coarseSimilarity: 0.91 }
        ],
        evidenceHash: "family-a-evidence-v1",
        metrics: { threshold: 0.76 },
        createdAt: AT
      });
      const familyBRevision1 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyB.id,
        expectedActiveRevisionId: null,
        medoidOccurrenceId: windowB.id,
        members: [
          { occurrenceId: windowB.id, coarseSimilarity: 1 },
          { occurrenceId: windowC.id, coarseSimilarity: 0.89 }
        ],
        evidenceHash: "family-b-evidence-v1",
        createdAt: AT
      });

      expect(repos.proceduralTrajectory.listActiveFamilyMedoids(familyScope))
        .toEqual(expect.arrayContaining([
          expect.objectContaining({ family: expect.objectContaining({ id: familyA.id }) }),
          expect.objectContaining({ family: expect.objectContaining({ id: familyB.id }) })
        ]));
      expect(repos.proceduralTrajectory.listAffectedFamilyIdsForOccurrences([windowB.id]))
        .toEqual([familyA.id, familyB.id].sort());
      expect(repos.proceduralTrajectory.listAffectedFamilyIdsForPath(windowA.pathId))
        .toEqual([familyA.id]);

      const canonical = repos.proceduralTrajectory.resolveCanonicalClusterHead({
        ...familyScope,
        evidenceSignature: "fine-alignment:inspect-repair-verify:v1",
        seedOccurrenceId: windowA.id,
        createdAt: AT
      });
      const sameCanonical = repos.proceduralTrajectory.resolveCanonicalClusterHead({
        ...familyScope,
        evidenceSignature: "fine-alignment:inspect-repair-verify:v1",
        seedOccurrenceId: windowB.id,
        createdAt: "2026-08-27T08:01:00.000Z"
      });
      expect(canonical.created).toBe(true);
      expect(sameCanonical).toMatchObject({
        created: false,
        cluster: { id: canonical.cluster.id },
        canonicalKey: { id: canonical.canonicalKey.id }
      });
      const clusterVersion = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: canonical.cluster.id,
        expectedActiveVersionId: null,
        medoidOccurrenceId: windowA.id,
        members: [
          supportMember(windowA.id, "family-reward-a"),
          supportMember(windowB.id, "family-reward-b")
        ],
        createdAt: AT
      }).record;
      repos.proceduralTrajectory.linkFamilyRevisionToCluster({
        familyRevisionId: familyARevision1.record.id,
        canonicalKeyId: canonical.canonicalKey.id,
        clusterVersionId: clusterVersion.id,
        createdAt: AT
      });
      repos.proceduralTrajectory.linkFamilyRevisionToCluster({
        familyRevisionId: familyBRevision1.record.id,
        canonicalKeyId: canonical.canonicalKey.id,
        clusterVersionId: clusterVersion.id,
        createdAt: AT
      });
      expect(repos.proceduralTrajectory.listClusterFamilyLinks(canonical.cluster.id))
        .toHaveLength(2);

      expect(repos.proceduralTrajectory.retireCluster(
        canonical.cluster.id,
        clusterVersion.id,
        "2026-08-27T08:01:30.000Z"
      ).status).toBe("retired");
      expect(repos.proceduralTrajectory.resolveCanonicalClusterHead({
        ...familyScope,
        evidenceSignature: "fine-alignment:inspect-repair-verify:v1",
        seedOccurrenceId: windowB.id,
        createdAt: "2026-08-27T08:01:31.000Z"
      })).toMatchObject({
        created: false,
        cluster: { id: canonical.cluster.id, status: "active" }
      });

      const familyARevision2 = repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyA.id,
        expectedActiveRevisionId: familyARevision1.record.id,
        medoidOccurrenceId: windowB.id,
        members: [
          { occurrenceId: windowA.id, coarseSimilarity: 0.94 },
          { occurrenceId: windowB.id, coarseSimilarity: 1 },
          { occurrenceId: windowC.id, coarseSimilarity: 0.88 }
        ],
        evidenceHash: "family-a-evidence-v2",
        createdAt: "2026-08-27T08:02:00.000Z"
      });
      expect(familyARevision2.record).toMatchObject({ revisionNo: 2, status: "active" });
      expect(repos.proceduralTrajectory.listFamilyMembers(familyARevision1.record.id))
        .toHaveLength(2);
      expect(() => repos.proceduralTrajectory.commitFamilyRevision({
        familyId: familyA.id,
        expectedActiveRevisionId: familyARevision1.record.id,
        medoidOccurrenceId: windowA.id,
        members: [{ occurrenceId: windowA.id, coarseSimilarity: 1 }],
        evidenceHash: "family-a-evidence-stale",
        createdAt: "2026-08-27T08:03:00.000Z"
      })).toThrow(ProceduralTrajectoryCasError);

      const bundle = repos.runtime.exportBundleTables(true);
      withRepositories(({ repos: imported }) => {
        const result = imported.runtime.importBundleTables(bundle);
        expect(result.inserted.trajectory_window_families).toBe(2);
        expect(result.inserted.trajectory_window_family_revisions).toBe(3);
        expect(result.inserted.trajectory_window_family_members).toBe(7);
        expect(result.inserted.trajectory_window_cluster_canonical_keys).toBe(1);
        expect(result.inserted.trajectory_window_family_cluster_links).toBe(2);
        expect(imported.proceduralTrajectory.getFamilyHead(familyA.id)).toMatchObject({
          activeRevisionId: familyARevision2.record.id
        });
        expect(imported.proceduralTrajectory.listClusterFamilyLinks(canonical.cluster.id))
          .toHaveLength(2);
      });
    });
  });

  it("refreshes current Window evidence without recompiling or re-embedding", () => {
    withRepositories(({ repos }) => {
      const windowA = seedWindow(repos, {
        userId: "reward-drift-user",
        source: "codex",
        suffix: "reward-drift-a",
        vector: [1, 0, 0]
      });
      const windowB = seedWindow(repos, {
        userId: "reward-drift-user",
        source: "claude_code",
        suffix: "reward-drift-b",
        vector: [0.98, 0.02, 0]
      });
      const pathA = repos.proceduralTrajectory.getPath(windowA.pathId)!;
      const embeddingsBefore = repos.proceduralTrajectory.listStepEmbeddings({
        pathId: pathA.id
      });
      const head = repos.proceduralTrajectory.createClusterHead({
        userId: "reward-drift-user",
        scale: 5,
        algorithmVersion: "multi-scale-window.v1",
        configHash: "v15-config",
        seedOccurrenceId: windowA.id,
        createdAt: AT
      }).record;
      const v1 = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: null,
        medoidOccurrenceId: windowA.id,
        members: [
          supportMember(windowA.id, "reward-a-positive"),
          supportMember(windowB.id, "reward-b-positive")
        ],
        createdAt: AT
      });
      expect(v1.record).toMatchObject({
        supportEpisodeCount: 2,
        counterexampleEpisodeCount: 0
      });

      repos.runtime.updateEpisodeReward(windowA.episodeId, {
        rTask: -0.7,
        rewardDetail: { phase: "revised", reason: "post-hoc failure" }
      }, "2026-08-27T08:10:00.000Z");
      const baseOccurrence = buildTrajectoryWindows(
        [pathA.path],
        [{ length: 5, stride: 2 }]
      ).find((item) => item.id === windowA.id)!;
      const refreshed = repos.proceduralTrajectory.insertWindow({
        occurrence: {
          ...baseOccurrence,
          terminalReward: -0.7,
          evidenceRole: "counterexample"
        },
        windowConfigHash: windowA.windowConfigHash,
        coarseRepresentationVersion: windowA.coarseRepresentationVersion,
        embeddingSignature: windowA.embeddingSignature,
        coarseVector: windowA.coarseVector,
        createdAt: "2026-08-27T08:10:00.000Z"
      });
      expect(refreshed).toMatchObject({
        created: false,
        record: {
          id: windowA.id,
          evidenceRole: "counterexample",
          terminalReward: -0.7,
          coarseVector: windowA.coarseVector
        }
      });
      expect(repos.proceduralTrajectory.getActivePath(windowA.episodeId)?.id).toBe(pathA.id);
      expect(repos.proceduralTrajectory.listStepEmbeddings({ pathId: pathA.id }))
        .toEqual(embeddingsBefore);

      const v2 = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: v1.record.id,
        medoidOccurrenceId: windowB.id,
        members: [
          {
            occurrenceId: windowA.id,
            rewardHash: "reward-a-negative",
            coarseSimilarity: 0.9
          },
          supportMember(windowB.id, "reward-b-positive")
        ],
        createdAt: "2026-08-27T08:11:00.000Z"
      });
      expect(v2.record).toMatchObject({
        versionNo: 2,
        supportEpisodeCount: 1,
        counterexampleEpisodeCount: 1
      });
      expect(repos.proceduralTrajectory.listClusterMembers(v1.record.id)
        .find((member) => member.occurrenceId === windowA.id)?.evidenceRole).toBe("support");
      expect(repos.proceduralTrajectory.listClusterMembers(v2.record.id)
        .find((member) => member.occurrenceId === windowA.id)).toMatchObject({
          evidenceRole: "counterexample",
          terminalReward: -0.7,
          rewardHash: "reward-a-negative"
        });
    });
  });

  it("reuses one semantic Window across mechanical stride and config drift", () => {
    withRepositories(({ repos }) => {
      const original = seedWindow(repos, {
        userId: "mechanical-drift-user",
        source: "codex",
        suffix: "mechanical-drift",
        vector: [0.7, 0.2, 0.1]
      });
      const head = repos.proceduralTrajectory.createClusterHead({
        userId: original.userId,
        scale: original.scale,
        algorithmVersion: "multi-scale-window.v1",
        configHash: "v15-config",
        seedOccurrenceId: original.id,
        createdAt: AT
      }).record;
      const historicalVersion = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: null,
        medoidOccurrenceId: original.id,
        members: [supportMember(original.id, "reward-mechanical-v1")],
        createdAt: AT
      });
      const path = repos.proceduralTrajectory.getPath(original.pathId)!;
      const occurrenceWithNewStride = buildTrajectoryWindows(
        [path.path],
        [{ length: 5, stride: 1 }]
      ).find((item) => item.startStepIndex === original.startStepIndex)!;
      expect(occurrenceWithNewStride).toMatchObject({ id: original.id, stride: 1 });

      const refreshed = repos.proceduralTrajectory.insertWindow({
        occurrence: occurrenceWithNewStride,
        windowConfigHash: "window-5-1-config",
        coarseRepresentationVersion: original.coarseRepresentationVersion,
        embeddingSignature: original.embeddingSignature,
        createdAt: "2026-08-27T08:12:00.000Z"
      });
      expect(refreshed).toMatchObject({
        created: false,
        record: {
          id: original.id,
          stride: 1,
          windowConfigHash: "window-5-1-config",
          coarseVector: original.coarseVector
        }
      });
      expect(repos.proceduralTrajectory.listWindowsForPath(path.id)).toHaveLength(1);
      expect(repos.proceduralTrajectory.getClusterVersion(historicalVersion.record.id))
        .toMatchObject({
          id: historicalVersion.record.id,
          membershipHash: historicalVersion.record.membershipHash,
          status: "active"
        });
      expect(repos.proceduralTrajectory.listClusterMembers(historicalVersion.record.id))
        .toEqual([expect.objectContaining({ occurrenceId: original.id })]);
    });
  });

  it("versions and rolls back formal upstream Skill snapshots and bundle data", () => {
    withRepositories(({ repos }) => {
      const windowA = seedWindow(repos, {
        userId: "skill-user",
        source: "codex",
        suffix: "skill-a",
        vector: [1, 0, 0]
      });
      const windowB = seedWindow(repos, {
        userId: "skill-user",
        source: "claude_code",
        suffix: "skill-b",
        vector: [0.95, 0.05, 0]
      });
      const head = repos.proceduralTrajectory.createClusterHead({
        userId: "skill-user",
        scale: 5,
        algorithmVersion: "multi-scale-window.v1",
        configHash: "v15-config",
        seedOccurrenceId: windowA.id,
        createdAt: AT
      }).record;
      const clusterVersion = repos.proceduralTrajectory.commitClusterVersion({
        clusterId: head.id,
        expectedActiveVersionId: null,
        medoidOccurrenceId: windowA.id,
        members: [supportMember(windowA.id, "skill-reward-a"), supportMember(windowB.id, "skill-reward-b")],
        createdAt: AT
      }).record;
      repos.memories.insert(skillMemory("skill_trajectory_user", "skill-user"));

      const v1 = repos.proceduralTrajectory.saveSkillVersion({
        clusterId: head.id,
        clusterVersionId: clusterVersion.id,
        expectedActiveSkillVersionId: null,
        skillKey: "skill:trajectory:skill-user:cluster-a",
        skillMemoryId: "skill_trajectory_user",
        payload: upstreamSkillSnapshot("Trajectory Skill v1"),
        createdAt: AT
      });
      const retry = repos.proceduralTrajectory.saveSkillVersion({
        clusterId: head.id,
        clusterVersionId: clusterVersion.id,
        expectedActiveSkillVersionId: null,
        skillKey: "skill:trajectory:skill-user:cluster-a",
        skillMemoryId: "skill_trajectory_user",
        payload: upstreamSkillSnapshot("Trajectory Skill v1"),
        createdAt: AT
      });
      expect(retry).toMatchObject({ created: false, reactivated: false, record: { id: v1.record.id } });

      const v2 = repos.proceduralTrajectory.saveSkillVersion({
        clusterId: head.id,
        clusterVersionId: clusterVersion.id,
        expectedActiveSkillVersionId: v1.record.id,
        skillKey: "skill:trajectory:skill-user:cluster-a",
        skillMemoryId: "skill_trajectory_user",
        payload: upstreamSkillSnapshot("Trajectory Skill v2"),
        createdAt: "2026-08-27T08:06:00.000Z"
      });
      expect(v2.record).toMatchObject({ versionNo: 2, status: "active" });
      expect(repos.proceduralTrajectory.getSkillVersion(v1.record.id)?.status)
        .toBe("superseded");

      const rollback = repos.proceduralTrajectory.activateSkillVersion({
        clusterId: head.id,
        versionId: v1.record.id,
        expectedActiveSkillVersionId: v2.record.id,
        activatedAt: "2026-08-27T08:07:00.000Z"
      });
      expect(rollback.record).toMatchObject({ id: v1.record.id, status: "active" });
      expect(repos.proceduralTrajectory.getClusterHead(head.id)).toMatchObject({
        activeSkillVersionId: v1.record.id,
        activeSkillMemoryId: "skill_trajectory_user"
      });

      const bundle = repos.runtime.exportBundleTables(true);
      withRepositories(({ repos: imported }) => {
        const result = imported.runtime.importBundleTables(bundle);
        expect(result.inserted.episode_execution_paths).toBe(2);
        expect(result.inserted.trajectory_window_occurrences).toBe(2);
        expect(imported.proceduralTrajectory.getClusterHead(head.id)).toMatchObject({
          activeVersionId: clusterVersion.id,
          activeSkillVersionId: v1.record.id
        });
        expect(imported.proceduralTrajectory.listSkillVersions(head.id)).toHaveLength(2);
      });
    });
  });
});

function withRepositories(
  run: (context: { db: MemoryDb; repos: Repositories }) => void
): void {
  const root = mkdtempSync(join(tmpdir(), "memmy-procedural-trajectory-"));
  try {
    const db = new MemoryDb({ path: join(root, "memory.sqlite") });
    const repos = new Repositories(db.db);
    run({ db, repos });
    db.close();
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function seedEpisode(
  repos: Repositories,
  input: {
    episodeId: string;
    sessionId: string;
    userId: string;
    source: string;
    rawTurnIds: string[];
    terminalReward?: number;
  }
): void {
  const session: SessionRecord = {
    id: input.sessionId,
    userId: input.userId,
    source: input.source,
    profileId: "default",
    status: "closed",
    meta: {},
    openedAt: AT,
    lastSeenAt: AT,
    closedAt: AT,
    updatedAt: AT
  };
  repos.runtime.createSession(session);
  const episode: EpisodeRecord = {
    id: input.episodeId,
    sessionId: input.sessionId,
    userId: input.userId,
    status: "closed",
    l1MemoryIds: [],
    rawTurnIds: input.rawTurnIds,
    feedbackIds: [],
    decisionRepairIds: [],
    l2PolicyIds: [],
    l3WorldModelIds: [],
    skillMemoryIds: [],
    turnCount: input.rawTurnIds.length,
    rTask: input.terminalReward ?? 0.8,
    rewardDetail: { phase: "final" },
    pipelineStatus: "idle",
    meta: {},
    openedAt: AT,
    closedAt: AT,
    updatedAt: AT
  };
  repos.runtime.createEpisode(episode);
  input.rawTurnIds.forEach((id, index) => {
    const rawTurn: RawTurnRecord = {
      id,
      sessionId: input.sessionId,
      episodeId: input.episodeId,
      turnId: `turn-${index}`,
      userId: input.userId,
      userText: index === 0 ? "开始任务" : "继续并修正",
      assistantText: "完成",
      toolCalls: [],
      toolResults: [],
      sourceMemoryIds: [],
      usage: {},
      status: "succeeded",
      createdAt: new Date(Date.parse(AT) + index * 1000).toISOString()
    };
    repos.runtime.insertRawTurn(rawTurn);
  });
}

function executionSteps(
  episodeId: string,
  firstRawTurnId: string,
  secondRawTurnId: string,
  sourceSnapshotHash: string
): ExecutionStepLiteV1[] {
  return Array.from({ length: 6 }, (_, index) => ({
    id: `step_${firstRawTurnId}_${index}`,
    schemaVersion: EXECUTION_STEP_LITE_SCHEMA_VERSION,
    episodeId,
    stepIndex: index,
    turnIndex: index < 3 ? 0 : 1,
    rawTurnId: index < 3 ? firstRawTurnId : secondRawTurnId,
    kind: "tool_action" as const,
    toolCallIndex: index < 3 ? index : index - 3,
    toolName: index % 2 === 0 ? "exec_command" : "apply_patch",
    intent: `执行子步骤 ${index}`,
    summary: `子步骤 ${index} 已完成`,
    outcome: "success" as const,
    evidenceRefs: [`raw_turn:${index < 3 ? firstRawTurnId : secondRawTurnId}:tool:${index}`],
    provenance: {
      algorithmVersion: "execution-step-semantics-lite.v1",
      model: "provider/model-a",
      sourceSnapshotHash
    }
  }));
}

function executionPath(input: {
  episodeId: string;
  userId: string;
  sourceRawTurnIds: string[];
  sourceSnapshotHash: string;
  steps: ExecutionStepLiteV1[];
  turnTransitions?: TurnTransitionV1[];
  terminalReward?: number;
}): EpisodeExecutionPathLiteV1 {
  const basis = {
    schemaVersion: EPISODE_EXECUTION_PATH_LITE_SCHEMA_VERSION,
    episodeId: input.episodeId,
    userId: input.userId,
    sourceRawTurnIds: input.sourceRawTurnIds,
    steps: input.steps,
    turnTransitions: input.turnTransitions ?? [],
    terminalReward: input.terminalReward ?? 0.8,
    sourceSnapshotHash: input.sourceSnapshotHash,
    compilerVersion: "path-compiler.v1",
    modelSignature: "provider/model-a",
    provenance: {
      algorithmVersion: "path-compiler.v1",
      inputCandidateCount: input.steps.length,
      compiledCandidateCount: input.steps.length,
      truncated: false
    }
  };
  const pathHash = stableHash(basis);
  return {
    id: `episode_execution_path_${pathHash.slice(0, 20)}`,
    pathHash,
    ...basis
  };
}

function testExecutionPath(
  episodeId: string,
  userId: string,
  rawTurnIds: string[],
  sourceSnapshotHash: string
): EpisodeExecutionPathLiteV1 {
  const steps = executionSteps(
    episodeId,
    rawTurnIds[0]!,
    rawTurnIds[1]!,
    sourceSnapshotHash
  );
  return executionPath({
    episodeId,
    userId,
    sourceRawTurnIds: rawTurnIds,
    sourceSnapshotHash,
    steps
  });
}

function seedWindow(
  repos: Repositories,
  input: {
    userId: string;
    source: string;
    suffix: string;
    vector: number[];
    terminalReward?: number;
  }
): TrajectoryWindowOccurrenceRecord {
  const sessionId = `session_${input.suffix}`;
  const episodeId = `episode_${input.suffix}`;
  const rawTurnIds = [`raw_${input.suffix}_1`, `raw_${input.suffix}_2`];
  seedEpisode(repos, { ...input, sessionId, episodeId, rawTurnIds });
  const sourceSnapshotHash = `snapshot-${input.suffix}`;
  const steps = executionSteps(
    episodeId,
    rawTurnIds[0]!,
    rawTurnIds[1]!,
    sourceSnapshotHash
  );
  const path = repos.proceduralTrajectory.savePathVersion({
    path: executionPath({
      episodeId,
      userId: input.userId,
      sourceRawTurnIds: rawTurnIds,
      sourceSnapshotHash,
      steps,
      terminalReward: input.terminalReward
    }),
    sourceAgentIds: [input.source],
    createdAt: AT
  }).record;
  for (const step of steps.slice(0, 5)) {
    repos.proceduralTrajectory.upsertStepEmbedding({
      pathId: path.id,
      stepId: step.id,
      stepIndex: step.stepIndex,
      embeddingSignature: "embed/model-a:3",
      semanticHash: `semantic-${step.id}`,
      vector: input.vector,
      createdAt: AT
    });
  }
  return repos.proceduralTrajectory.insertWindow({
    occurrence: buildTrajectoryWindows(
      [path.path],
      [{ length: 5, stride: 2 }]
    ).find((item) => item.startStepIndex === 0)!,
    windowConfigHash: "window-5-2-config",
    coarseRepresentationVersion: "window-intent-sequence.v1",
    embeddingSignature: "embed/model-a:3",
    coarseVector: input.vector,
    createdAt: AT
  }).record;
}

function supportMember(occurrenceId: string, rewardHash: string) {
  return {
    occurrenceId,
    rewardHash,
    coarseSimilarity: 0.9,
    alignment: { admitted: true, matchedSteps: 4 }
  };
}

function upstreamSkillSnapshot(name: string): Record<string, unknown> {
  return {
    name,
    eta: 0.9,
    status: "active",
    support: 2,
    gain: 0.2,
    source_policy_ids: [],
    source_world_model_ids: [],
    evidence_anchor_ids: ["episode_skill-a", "episode_skill-b"],
    invocation_guide: "Use this when a matching execution problem appears.",
    procedure_json: {
      retrievalBlurb: "Inspect, repair, and verify a compatible artifact.",
      triggerContext: "A matching procedural execution problem is observed.",
      steps: [
        { title: "Inspect", body: "Inspect the concrete artifact and failure." },
        { title: "Repair", body: "Apply the grounded repair." },
        { title: "Verify", body: "Run the relevant validation." }
      ]
    },
    trials_attempted: 0,
    trials_passed: 0,
    success_rate: 0,
    beta_posterior: { alpha: 1, beta: 1, mean: 0.5 }
  };
}

function skillMemory(id: string, userId: string): MemoryRow {
  const skill = upstreamSkillSnapshot("Trajectory Skill v1");
  return {
    id,
    timeline: AT,
    userId,
    memoryType: "SkillMemory",
    status: "activated",
    visibility: "private",
    memoryKey: `skill:trajectory:${userId}:cluster-a`,
    memoryValue: "Use this when a matching execution problem appears.",
    tags: ["skill", "procedural"],
    info: {},
    properties: {
      memory_type: "SkillMemory",
      status: "activated",
      tags: ["skill", "procedural"],
      info: {},
      internal_info: {
        memory_layer: "Skill",
        memory_kind: "skill",
        schema_version: 1,
        skill
      }
    },
    memoryLayer: "Skill",
    contentHash: "skill-content",
    version: 1,
    createdAt: AT,
    updatedAt: AT,
    deletedAt: null
  };
}
