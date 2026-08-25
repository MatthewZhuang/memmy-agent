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
        episodeReward: 1,
        attributionType: "helpful",
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
        episodeReward: 1
      });
      expect(first.credits).toEqual([
        expect.objectContaining({
          occurrenceId: evidence.occurrence.id,
          rewardCredit: 1,
          attributionType: "helpful",
          confidence: 0.98,
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
        episodeReward: -1,
        attributionType: "harmful",
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
      episodeReward: 1,
      credits: []
    })).toThrow(/at least one Span credit/);
  });
});

function creditRun(
  evidence: ReturnType<typeof persistProceduralPolicyEvidence>,
  input: {
    rewardHash: string;
    episodeReward: number;
    attributionType: "helpful" | "harmful";
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
    episodeReward: input.episodeReward,
    credits: [{
      occurrenceId: evidence.occurrence.id,
      spanId: evidence.occurrence.spanId,
      spanIndex: 0,
      preStateId,
      postStateId,
      rewardCredit: input.episodeReward,
      attributionType: input.attributionType,
      confidence: 0.98,
      evidenceRole: input.evidenceRole,
      evidenceRefs: [evidence.occurrence.id],
      reason: "repository test"
    }],
    scorerModel: "repository-test"
  });
}
