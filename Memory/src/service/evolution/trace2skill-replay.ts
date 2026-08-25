import type { Repositories } from "../../storage/repositories.js";
import type {
  PolicySequencePatternRecord,
  ProceduralSkillCandidateRecord
} from "../../storage/policy-sequence-pattern-repository.js";
import type { MemoryRow } from "../../types.js";
import { isRecord } from "../../utils/json.js";
import { nowIso } from "../../utils/time.js";
import { EpisodePolicyProjectionPipeline } from "./episode-policy-projection.js";
import {
  EPISODE_FAMILY_COMBINED_THRESHOLD,
  EPISODE_FAMILY_GOAL_THRESHOLD,
  EPISODE_FAMILY_TRANSITION_THRESHOLD
} from "./episode-capability-model.js";
import { PolicySequenceMiningPipeline } from "./policy-sequence-mining.js";
import {
  ProceduralSequenceSkillCompilationPipeline,
  proceduralSequenceSkillMemoryKey
} from "./procedural-sequence-skill-compilation.js";

const ACTIVE_PROCEDURAL_CLUSTER_ALGORITHM_VERSION =
  "procedural-span-semantic-cluster.v8";

export const TRACE2SKILL_DIAGNOSTIC_SCHEMA_VERSION =
  "trace2skill-diagnostic.v1" as const;
export const TRACE2SKILL_REPLAY_SCHEMA_VERSION =
  "trace2skill-replay.v1" as const;

export interface Trace2SkillDiagnosticReportV1 {
  schemaVersion: typeof TRACE2SKILL_DIAGNOSTIC_SCHEMA_VERSION;
  generatedAt: string;
  episode: {
    id: string;
    userId: string;
    sessionId: string;
    status: string;
    terminalReward?: number;
    turnCount: number;
  };
  path?: {
    id: string;
    schemaVersion: string;
    algorithmVersion: string;
    pathHash: string;
    stateCount: number;
    stepCount: number;
    spanCount: number;
    continuous: boolean;
    spans: Array<{
      occurrenceId: string;
      spanId: string;
      spanIndex: number;
      localGoal: string;
      capabilityGoal: string;
      procedureSemantic: string;
      entryCondition: string;
      exitCondition: string;
      terminationStatus: string;
      rawTurnIds: string[];
      stepIds: string[];
      cost: Record<string, number | undefined>;
      credit?: {
        rewardCredit: number;
        attributionType: string;
        confidence: number;
        evidenceRole: string;
        reason: string;
      };
      cluster?: {
        id: string;
        status?: string;
        membershipVersion?: string;
        similarity: number;
        evidenceRole: string;
      };
      policy?: {
        id: string;
        policyKey: string;
        title: string;
        confidence: number;
        evidenceRole: string;
        matchConfidence: number;
      };
    }>;
  };
  spanCredit?: {
    runId: string;
    rewardHash: string;
    algorithmVersion: string;
    creditCount: number;
  };
  projection?: {
    id: string;
    projectionHash: string;
    namespaceId: string;
    mappedSpanCount: number;
    unmappedSpanCount: number;
    nodes: Array<{
      nodeIndex: number;
      occurrenceId: string;
      localGoal: string;
      assignment: unknown;
    }>;
  };
  capability?: {
    signatureId: string;
    familyId: string;
    signatureHash: string;
    embeddingProvider: string;
    embeddingModel: string;
    goalText: string;
    stateTransitionText: string;
    outcomeText: string;
    contextText: string;
    policyKeys: string[];
    thresholds: {
      goal: number;
      stateTransition: number;
      combined: number;
    };
    affinities: Array<{
      id: string;
      peerEpisodeId: string;
      goalSimilarity: number;
      stateTransitionSimilarity: number;
      outcomeSimilarity: number;
      contextSimilarity: number;
      pathStructureSimilarity: number;
      combinedSimilarity: number;
      familyEligible: boolean;
    }>;
  };
  patterns: Array<{
    id: string;
    policyKeys: string[];
    capabilityType: string;
    episodeFamilyId?: string;
    lifecycleStatus: string;
    distinctSupportEpisodeCount: number;
    occurrence: {
      id: string;
      evidenceRole: string;
      discoverySources: string[];
      matchedNodeIndexes: number[];
      pathNodeIndexes: number[];
      gapNodeIndexes: number[];
      peerEpisodeId?: string;
      combinedSimilarity?: number;
    };
    candidate?: {
      id: string;
      lifecycleStatus: string;
      evidenceHash: string;
      capabilityType: string;
      episodeFamilyId?: string;
      supportEpisodeIds: string[];
      counterexampleEpisodeIds: string[];
      discoverySources: string[];
      executable: boolean;
    };
    skill?: {
      id: string;
      status: string;
      memoryKey?: string;
      name?: string;
      executable: boolean;
      evidenceHash?: string;
    };
  }>;
  checks: {
    episodeClosed: boolean;
    activePath: boolean;
    pathContinuous: boolean;
    spanCreditCoverage: number;
    policyProjectionCoverage: number;
    capabilitySignature: boolean;
    episodeFamily: boolean;
    episodeSimilarityRoute: boolean;
    policySequenceRoute: boolean;
    readyCandidate: boolean;
    compiledSkill: boolean;
  };
  blockers: string[];
}

export interface Trace2SkillReplayResultV1 {
  schemaVersion: typeof TRACE2SKILL_REPLAY_SCHEMA_VERSION;
  mode: "replay";
  generatedAt: string;
  episodeId: string;
  actions: {
    projectionId: string;
    projectionCreated: boolean;
    extractedOccurrenceCount: number;
    affectedPatternIds: string[];
    compiledSkillIds: string[];
    reusedSkillIds: string[];
    retiredSkillIds: string[];
  };
  before: Trace2SkillDiagnosticReportV1;
  after: Trace2SkillDiagnosticReportV1;
}

export interface Trace2SkillReplayServiceDeps {
  repos: Repositories;
  projection: EpisodePolicyProjectionPipeline;
  mining: PolicySequenceMiningPipeline;
  skillCompilation: ProceduralSequenceSkillCompilationPipeline;
}

/**
 * Replays only the deterministic downstream Trace2Skill projection on the
 * supplied database. A caller implements dry-run semantics by giving this
 * service a consistent database clone; commit mode supplies the live database.
 */
export class Trace2SkillReplayService {
  constructor(private readonly deps: Trace2SkillReplayServiceDeps) {}

  inspect(episodeId: string, generatedAt = nowIso()): Trace2SkillDiagnosticReportV1 {
    return inspectTrace2SkillEpisode(this.deps.repos, episodeId, generatedAt);
  }

  async replay(input: {
    episodeId: string;
    at?: string;
  }): Promise<Trace2SkillReplayResultV1> {
    const at = input.at ?? nowIso();
    const episode = this.deps.repos.runtime.getEpisode(input.episodeId);
    if (!episode) throw new Error(`Trace2Skill replay Episode not found: ${input.episodeId}`);
    if (episode.status !== "closed") {
      throw new Error(`Trace2Skill replay requires a closed Episode: ${input.episodeId}`);
    }
    const path = this.deps.repos.proceduralPaths.getActiveForEpisode(input.episodeId);
    if (!path) {
      throw new Error(`Trace2Skill replay requires an active procedural path: ${input.episodeId}`);
    }

    const before = this.inspect(input.episodeId, at);
    const projection = this.deps.projection.projectPath(path.id, at);
    const mining = await this.deps.mining.mineProjection(projection.record.id, at);
    const affectedPatterns = mining.patterns;
    const compiledSkillIds: string[] = [];
    const reusedSkillIds: string[] = [];
    const retiredSkillIds: string[] = [];

    for (const pattern of affectedPatterns) {
      const candidate = this.deps.repos.policySequencePatterns
        .getActiveCandidateForPattern(pattern.id);
      if (!candidate || candidate.lifecycleStatus !== "ready") {
        const retired = this.deps.skillCompilation.retirePattern(pattern, at);
        if (retired) retiredSkillIds.push(retired.id);
        continue;
      }
      const needsCompilation = this.deps.skillCompilation.needsCompilation(candidate);
      const skill = await this.deps.skillCompilation.compileCandidate(candidate, {
        sessionId: episode.sessionId,
        episodeId: episode.id,
        at
      });
      if (!skill) continue;
      if (needsCompilation) compiledSkillIds.push(skill.id);
      else reusedSkillIds.push(skill.id);
    }

    return {
      schemaVersion: TRACE2SKILL_REPLAY_SCHEMA_VERSION,
      mode: "replay",
      generatedAt: at,
      episodeId: input.episodeId,
      actions: {
        projectionId: projection.record.id,
        projectionCreated: projection.created,
        extractedOccurrenceCount: mining.extractedOccurrenceCount,
        affectedPatternIds: affectedPatterns.map((pattern) => pattern.id),
        compiledSkillIds: unique(compiledSkillIds),
        reusedSkillIds: unique(reusedSkillIds),
        retiredSkillIds: unique(retiredSkillIds)
      },
      before,
      after: this.inspect(input.episodeId, nowIso())
    };
  }
}

export function inspectTrace2SkillEpisode(
  repos: Repositories,
  episodeId: string,
  generatedAt = nowIso()
): Trace2SkillDiagnosticReportV1 {
  const episode = repos.runtime.getEpisode(episodeId);
  if (!episode) throw new Error(`Trace2Skill diagnostic Episode not found: ${episodeId}`);
  const path = repos.proceduralPaths.getActiveForEpisode(episodeId);
  const occurrences = path ? repos.proceduralPaths.listOccurrencesForPath(path.id) : [];
  const creditRun = repos.proceduralSpanCredits.getActiveForEpisode(episodeId);
  const credits = creditRun ? repos.proceduralSpanCredits.listCredits(creditRun.id) : [];
  const creditByOccurrence = new Map(credits.map((credit) => [credit.occurrenceId, credit]));
  const clusterMembers = repos.proceduralSpanClusters.listMembersForEpisode(
    episodeId,
    ACTIVE_PROCEDURAL_CLUSTER_ALGORITHM_VERSION
  );
  const clusterByOccurrence = new Map(clusterMembers.map((member) => [member.occurrenceId, member]));
  const policyMappings = path ? repos.proceduralPolicies.listActiveOccurrencesForPath(path.id) : [];
  const policyByOccurrence = new Map(policyMappings.map((mapping) => [mapping.occurrenceId, mapping]));
  const projection = repos.episodePolicyProjections.getActiveForEpisode(episodeId);
  const signature = safeRead(
    () => repos.episodeCapabilities.getActiveForEpisode(episodeId),
    undefined
  );
  const patterns = projection
    ? safeRead(() => episodePatternReports(repos, projection.namespaceId, episodeId), [])
    : [];
  const continuous = Boolean(path) && path!.path.spans.every((span, index) => {
    const previous = path!.path.spans[index - 1];
    return !previous || previous.postStateId === span.preStateId;
  });
  const mappedSpanCount = projection?.mappedSpanCount ?? 0;
  const spanCount = occurrences.length;
  const checks = {
    episodeClosed: episode.status === "closed",
    activePath: Boolean(path),
    pathContinuous: continuous,
    spanCreditCoverage: spanCount === 0 ? 0 : credits.length / spanCount,
    policyProjectionCoverage: spanCount === 0 ? 0 : mappedSpanCount / spanCount,
    capabilitySignature: Boolean(signature),
    episodeFamily: Boolean(signature?.familyId),
    episodeSimilarityRoute: patterns.some((pattern) =>
      pattern.occurrence.discoverySources.includes("episode_similarity")
    ),
    policySequenceRoute: patterns.some((pattern) =>
      pattern.occurrence.discoverySources.includes("policy_sequence_similarity")
    ),
    readyCandidate: patterns.some((pattern) => pattern.candidate?.lifecycleStatus === "ready"),
    compiledSkill: patterns.some((pattern) => pattern.skill?.executable)
  };
  const blockers = diagnosticBlockers(checks, {
    spanCount,
    mappedSpanCount,
    terminalReward: episode.rTask
  });

  return {
    schemaVersion: TRACE2SKILL_DIAGNOSTIC_SCHEMA_VERSION,
    generatedAt,
    episode: {
      id: episode.id,
      userId: episode.userId,
      sessionId: episode.sessionId,
      status: episode.status,
      ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask }),
      turnCount: episode.turnCount
    },
    ...(path ? {
      path: {
        id: path.id,
        schemaVersion: path.path.schemaVersion,
        algorithmVersion: path.reconstructionAlgorithmVersion,
        pathHash: path.pathHash,
        stateCount: path.path.states.length,
        stepCount: path.path.steps.length,
        spanCount: path.path.spans.length,
        continuous,
        spans: occurrences.map((occurrence) => {
          const span = occurrence.span;
          const credit = creditByOccurrence.get(occurrence.id);
          const member = clusterByOccurrence.get(occurrence.id);
          const cluster = member ? repos.proceduralSpanClusters.get(member.clusterId) : undefined;
          const mapping = policyByOccurrence.get(occurrence.id);
          const policy = mapping ? repos.proceduralPolicies.get(mapping.policyVersionId) : undefined;
          return {
            occurrenceId: occurrence.id,
            spanId: span.id,
            spanIndex: span.spanIndex,
            localGoal: span.localGoal,
            capabilityGoal: occurrence.capabilityGoal,
            procedureSemantic: occurrence.projection.procedureText,
            entryCondition: span.entryCondition,
            exitCondition: span.termination.exitCondition,
            terminationStatus: span.termination.status,
            rawTurnIds: [...span.rawTurnIds],
            stepIds: [...span.stepIds],
            cost: { ...span.cost },
            ...(credit ? {
              credit: {
                rewardCredit: credit.rewardCredit,
                attributionType: credit.attributionType,
                confidence: credit.confidence,
                evidenceRole: credit.evidenceRole,
                reason: credit.reason
              }
            } : {}),
            ...(member ? {
              cluster: {
                id: member.clusterId,
                ...(cluster ? {
                  status: cluster.status,
                  membershipVersion: cluster.membershipVersion
                } : {}),
                similarity: member.similarity,
                evidenceRole: member.evidenceRole
              }
            } : {}),
            ...(mapping && policy ? {
              policy: {
                id: policy.id,
                policyKey: policy.policyKey,
                title: policy.title,
                confidence: policy.confidence,
                evidenceRole: mapping.evidenceRole,
                matchConfidence: mapping.matchConfidence
              }
            } : {})
          };
        })
      }
    } : {}),
    ...(creditRun ? {
      spanCredit: {
        runId: creditRun.id,
        rewardHash: creditRun.rewardHash,
        algorithmVersion: creditRun.algorithmVersion,
        creditCount: credits.length
      }
    } : {}),
    ...(projection ? {
      projection: {
        id: projection.id,
        projectionHash: projection.projectionHash,
        namespaceId: projection.namespaceId,
        mappedSpanCount: projection.mappedSpanCount,
        unmappedSpanCount: projection.unmappedSpanCount,
        nodes: projection.projection.nodes.map((node) => ({
          nodeIndex: node.nodeIndex,
          occurrenceId: node.occurrenceId,
          localGoal: node.localGoal,
          assignment: node.assignment
        }))
      }
    } : {}),
    ...(signature ? {
      capability: {
        signatureId: signature.signature.id,
        familyId: signature.familyId,
        signatureHash: signature.signature.signatureHash,
        embeddingProvider: signature.embeddingProvider,
        embeddingModel: signature.embeddingModel,
        goalText: signature.signature.goalText,
        stateTransitionText: signature.signature.stateTransitionText,
        outcomeText: signature.signature.outcomeText,
        contextText: signature.signature.contextText,
        policyKeys: signature.signature.policyNodes.map((node) => node.policyKey),
        thresholds: {
          goal: EPISODE_FAMILY_GOAL_THRESHOLD,
          stateTransition: EPISODE_FAMILY_TRANSITION_THRESHOLD,
          combined: EPISODE_FAMILY_COMBINED_THRESHOLD
        },
        affinities: repos.episodeCapabilities.listAffinitiesForSignature(signature.signature.id)
          .map((record) => {
            const affinity = record.affinity;
            return {
              id: affinity.id,
              peerEpisodeId: affinity.leftEpisodeId === episodeId
                ? affinity.rightEpisodeId
                : affinity.leftEpisodeId,
              goalSimilarity: affinity.goalSimilarity,
              stateTransitionSimilarity: affinity.stateTransitionSimilarity,
              outcomeSimilarity: affinity.outcomeSimilarity,
              contextSimilarity: affinity.contextSimilarity,
              pathStructureSimilarity: affinity.pathStructureSimilarity,
              combinedSimilarity: affinity.combinedSimilarity,
              familyEligible: affinity.familyEligible
            };
          })
      }
    } : {}),
    patterns,
    checks,
    blockers
  };
}

function episodePatternReports(
  repos: Repositories,
  namespaceId: string,
  episodeId: string
): Trace2SkillDiagnosticReportV1["patterns"] {
  const results: Trace2SkillDiagnosticReportV1["patterns"] = [];
  for (const pattern of repos.policySequencePatterns.listPatternsForNamespace(namespaceId)) {
    for (const record of repos.policySequencePatterns.listActiveOccurrencesForPattern(pattern.id)) {
      const occurrence = record.occurrence;
      if (occurrence.episodeId !== episodeId) continue;
      const candidate = repos.policySequencePatterns.getActiveCandidateForPattern(pattern.id);
      const skill = repos.memories.getByKey(
        "Skill",
        proceduralSequenceSkillMemoryKey(candidate ?? pattern)
      );
      results.push({
        id: pattern.id,
        policyKeys: [...pattern.policyKeys],
        capabilityType: pattern.capabilityType,
        ...(pattern.episodeFamilyId ? { episodeFamilyId: pattern.episodeFamilyId } : {}),
        lifecycleStatus: pattern.lifecycleStatus,
        distinctSupportEpisodeCount: pattern.distinctSupportEpisodeCount,
        occurrence: {
          id: occurrence.id,
          evidenceRole: occurrence.evidenceRole,
          discoverySources: [...occurrence.discoverySources],
          matchedNodeIndexes: [...occurrence.matchedNodeIndexes],
          pathNodeIndexes: [...occurrence.pathNodeIndexes],
          gapNodeIndexes: occurrence.pathNodeIndexes.filter((index) =>
            !occurrence.matchedNodeIndexes.includes(index)
          ),
          ...(occurrence.episodeAffinity ? {
            peerEpisodeId: occurrence.episodeAffinity.peerEpisodeId,
            combinedSimilarity: occurrence.episodeAffinity.combinedSimilarity
          } : {})
        },
        ...(candidate ? { candidate: candidateReport(candidate) } : {}),
        ...(skill ? { skill: skillReport(skill) } : {})
      });
    }
  }
  return results.sort((left, right) =>
    right.policyKeys.length - left.policyKeys.length || left.id.localeCompare(right.id)
  );
}

function candidateReport(
  candidate: ProceduralSkillCandidateRecord
): NonNullable<Trace2SkillDiagnosticReportV1["patterns"][number]["candidate"]> {
  return {
    id: candidate.id,
    lifecycleStatus: candidate.lifecycleStatus,
    evidenceHash: candidate.evidenceHash,
    capabilityType: candidate.candidate.capabilityType,
    ...(candidate.candidate.episodeFamilyId
      ? { episodeFamilyId: candidate.candidate.episodeFamilyId }
      : {}),
    supportEpisodeIds: [...candidate.candidate.supportEpisodeIds],
    counterexampleEpisodeIds: [...candidate.candidate.counterexampleEpisodeIds],
    discoverySources: [...candidate.candidate.discoverySources],
    executable: candidate.candidate.provenance.executable
  };
}

function skillReport(
  skill: MemoryRow
): NonNullable<Trace2SkillDiagnosticReportV1["patterns"][number]["skill"]> {
  const internal = skill.properties.internal_info;
  const procedural = isRecord(internal.procedural_sequence_skill)
    ? internal.procedural_sequence_skill
    : {};
  const skillMeta = isRecord(internal.skill) ? internal.skill : {};
  return {
    id: skill.id,
    status: skill.status,
    ...(skill.memoryKey ? { memoryKey: skill.memoryKey } : {}),
    ...(typeof skillMeta.name === "string" ? { name: skillMeta.name } : {}),
    executable: procedural.executable === true &&
      skill.status !== "archived" && skill.status !== "deleted",
    ...(typeof procedural.evidence_hash === "string"
      ? { evidenceHash: procedural.evidence_hash }
      : {})
  };
}

function diagnosticBlockers(
  checks: Trace2SkillDiagnosticReportV1["checks"],
  input: { spanCount: number; mappedSpanCount: number; terminalReward?: number }
): string[] {
  const blockers: string[] = [];
  if (!checks.episodeClosed) blockers.push("episode_not_closed");
  if (!checks.activePath) blockers.push("active_procedural_path_missing");
  if (checks.activePath && !checks.pathContinuous) blockers.push("procedural_path_discontinuous");
  if (input.terminalReward === undefined) blockers.push("terminal_reward_missing");
  else if (checks.spanCreditCoverage < 1) blockers.push("span_credit_incomplete");
  if (input.spanCount > 0 && input.mappedSpanCount === 0) blockers.push("no_span_mapped_to_policy");
  else if (input.mappedSpanCount < 2) blockers.push("fewer_than_two_mapped_policies");
  if (input.mappedSpanCount >= 2 && !checks.capabilitySignature) {
    blockers.push("episode_capability_signature_missing");
  }
  if (checks.capabilitySignature && !checks.readyCandidate) {
    blockers.push("insufficient_repeated_episode_evidence_for_ready_candidate");
  }
  if (checks.readyCandidate && !checks.compiledSkill) blockers.push("ready_candidate_not_compiled");
  return blockers;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function safeRead<T>(read: () => T, fallback: T): T {
  try {
    return read();
  } catch (error) {
    if (error instanceof Error && /no such (?:table|column)/i.test(error.message)) {
      return fallback;
    }
    throw error;
  }
}
