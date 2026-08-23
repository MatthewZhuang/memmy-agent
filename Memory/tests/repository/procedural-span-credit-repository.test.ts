import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  MemoryDb,
  buildEpisodeSpanCreditRun
} from "../../src/index.js";
import { Repositories } from "../../src/storage/repositories.js";
import {
  PROCEDURAL_POLICY_TEST_NAMESPACE,
  persistProceduralPolicyEvidence
} from "../fixtures/procedural-policy-fixture.js";

describe("repository procedural SpanCredit", () => {
  it("persists immutable reward/path versions and rolls the active run backward", () => {
    const root = mkdtempSync(join(tmpdir(), "memmy-procedural-span-credit-"));
    try {
      const db = new MemoryDb({ path: join(root, "memory.sqlite") });
      const repos = new Repositories(db.db);
      const evidence = persistProceduralPolicyEvidence(repos, "credit-repository");
      const firstRun = creditRun(evidence, {
        rewardHash: "reward-hash-v1",
        goalAchievement: 1,
        processQuality: 0.4,
        evidenceRole: "support"
      });
      const first = repos.proceduralSpanCredits.saveAndActivate(
        firstRun,
        "2026-08-20T00:01:00.000Z"
      );
      const replay = repos.proceduralSpanCredits.saveAndActivate(
        firstRun,
        "2026-08-20T00:01:01.000Z"
      );

      expect(first.created).toBe(true);
      expect(replay.created).toBe(false);
      expect(first.record).toMatchObject({
        episodeId: evidence.episodeId,
        pathId: evidence.path.id,
        pathHash: evidence.path.pathHash,
        namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
        rewardHash: "reward-hash-v1",
        status: "active",
        goalAchievement: 1
      });
      expect(first.credits).toEqual([
        expect.objectContaining({
          occurrenceId: evidence.occurrence.id,
          goalCredit: 1,
          processQuality: 0.4,
          confidence: 0.98,
          creditScore: 0.98,
          evidenceRole: "support"
        })
      ]);
      expect(repos.proceduralSpanCredits.getActiveCreditForOccurrence(evidence.occurrence.id))
        .toMatchObject({ runId: firstRun.id, evidenceRole: "support" });
      const firstRevision = repos.proceduralSpanCredits.activeNamespaceRevision(
        PROCEDURAL_POLICY_TEST_NAMESPACE
      );
      expect(firstRevision).toMatch(/^[a-f0-9]{64}$/);
      expect(repos.proceduralSpanCredits.listActiveCreditsForNamespace(
        PROCEDURAL_POLICY_TEST_NAMESPACE
      )).toHaveLength(1);

      const secondRun = creditRun(evidence, {
        rewardHash: "reward-hash-v2",
        goalAchievement: 0,
        processQuality: -0.6,
        evidenceRole: "counterexample"
      });
      repos.proceduralSpanCredits.saveAndActivate(secondRun, "2026-08-20T00:02:00.000Z");
      expect(repos.proceduralSpanCredits.get(firstRun.id)).toMatchObject({ status: "inactive" });
      expect(repos.proceduralSpanCredits.getActiveCreditForOccurrence(evidence.occurrence.id))
        .toMatchObject({ runId: secondRun.id, evidenceRole: "counterexample" });
      expect(repos.proceduralSpanCredits.activeNamespaceRevision(
        PROCEDURAL_POLICY_TEST_NAMESPACE
      )).not.toBe(firstRevision);

      repos.proceduralSpanCredits.activate(firstRun.id, "2026-08-20T00:03:00.000Z");
      expect(repos.proceduralSpanCredits.getActiveForEpisode(evidence.episodeId)?.id).toBe(firstRun.id);
      expect(repos.proceduralSpanCredits.get(secondRun.id)).toMatchObject({ status: "inactive" });
      expect(repos.proceduralSpanCredits.listCredits(secondRun.id)).toHaveLength(1);
      db.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects a run without a verifiable boundary-to-credit path", () => {
    expect(() => buildEpisodeSpanCreditRun({
      episodeId: "episode",
      pathId: "path",
      pathHash: "path-hash",
      namespaceId: "namespace",
      rewardHash: "reward-hash",
      goalAchievement: 1,
      statePotentials: [],
      credits: []
    })).toThrow(/one more boundary state/);
  });
});

function creditRun(
  evidence: ReturnType<typeof persistProceduralPolicyEvidence>,
  input: {
    rewardHash: string;
    goalAchievement: number;
    processQuality: number;
    evidenceRole: "support" | "counterexample";
  }
) {
  const preStateId = evidence.occurrence.preStateId;
  const postStateId = evidence.occurrence.postStateId;
  return buildEpisodeSpanCreditRun({
    episodeId: evidence.episodeId,
    pathId: evidence.path.id,
    pathHash: evidence.path.pathHash,
    namespaceId: PROCEDURAL_POLICY_TEST_NAMESPACE,
    rewardHash: input.rewardHash,
    goalAchievement: input.goalAchievement,
    statePotentials: [{
      boundaryIndex: 0,
      stateId: preStateId,
      progress: 0,
      evidenceRefs: [preStateId],
      reason: "Episode start anchor"
    }, {
      boundaryIndex: 1,
      stateId: postStateId,
      progress: input.goalAchievement,
      evidenceRefs: [postStateId],
      reason: "Episode goal-achievement anchor"
    }],
    credits: [{
      occurrenceId: evidence.occurrence.id,
      spanId: evidence.occurrence.spanId,
      spanIndex: 0,
      preStateId,
      postStateId,
      goalCredit: input.goalAchievement,
      processQuality: input.processQuality,
      confidence: 0.98,
      evidenceRole: input.evidenceRole,
      evidenceRefs: [evidence.occurrence.id],
      reason: "repository test"
    }],
    scorerModel: "repository-test"
  });
}
