import type {
  ProceduralPatternSkillInput,
  ProceduralSkillAlignedSequenceStep,
  ProceduralSkillEvidenceStep
} from "./procedural-pattern-skill.js";
import type { ExecutionStepLiteV1 } from "./procedural-window-model.js";
import { stableHash } from "../../utils/id.js";
import type {
  EpisodeTrajectoryDocumentV1,
  LongCommonTrajectoryV1
} from "./long-trajectory-model.js";

export function buildLongTrajectorySkillInput(input: {
  trajectory: LongCommonTrajectoryV1;
  documents: readonly EpisodeTrajectoryDocumentV1[];
  userId: string;
  sourceTraceIdsForSteps(episodeId: string, steps: readonly ExecutionStepLiteV1[]): string[];
}): ProceduralPatternSkillInput | undefined {
  const { trajectory } = input;
  const documentByEpisodeId = new Map(input.documents.map((item) => [
    item.path.episodeId,
    item
  ]));
  const reference = documentByEpisodeId.get(trajectory.referenceEpisodeId);
  if (!reference) return undefined;
  const referenceWindowById = new Map(reference.windows.map((item) => [
    item.occurrence.id,
    item
  ]));
  const requiredSpanById = new Map(trajectory.requiredSpans.map((item) => [
    item.referenceSpanId,
    item
  ]));
  const prepared = trajectory.occurrences.flatMap((occurrence) => {
    const document = documentByEpisodeId.get(occurrence.episodeId);
    if (!document) return [];
    const orderedMatches = [...occurrence.matches].sort((left, right) =>
      left.episodeStartStepIndex - right.episodeStartStepIndex ||
      left.episodeEndStepIndex - right.episodeEndStepIndex);
    const spans = orderedMatches.flatMap((match) => {
      const requiredSpan = requiredSpanById.get(match.referenceSpanId);
      const referenceWindow = referenceWindowById.get(match.referenceSpanId);
      const episodeWindow = document.windows.find((item) =>
        item.occurrence.id === match.episodeSpanId);
      if (!requiredSpan || !referenceWindow || !episodeWindow) return [];
      return [{
        match,
        requiredSpan,
        referenceWindow,
        episodeWindow
      }];
    });
    if (spans.length !== orderedMatches.length) return [];
    return [{ occurrence, document, orderedMatches, spans }];
  });
  if (prepared.length < 2) return undefined;

  const commonCore = trajectory.requiredSpans.map((span, index) => {
    const referenceWindow = referenceWindowById.get(span.referenceSpanId);
    return {
      anchorId: span.anchorId,
      anchorOffset: index,
      anchorStepId: referenceWindow?.occurrence.steps[0]?.id ?? span.referenceSpanId,
      anchorIntent: `${span.referenceSpanLabel}: ${span.semanticText}`,
      anchorSummary: span.summaryText,
      supportEpisodeIds: span.supportEpisodeIds,
      evidenceStepIds: span.evidenceStepIds
    };
  });

  const evidence = prepared.flatMap((item) => {
    const alignedSequence: ProceduralSkillAlignedSequenceStep[] = [];
    for (const [spanIndex, span] of item.spans.entries()) {
      for (const step of span.episodeWindow.occurrence.steps) {
        alignedSequence.push({
          role: "span_step",
          anchorId: span.requiredSpan.anchorId,
          spanSimilarity: span.match.coarseSimilarity,
          ...skillStep(step)
        });
      }
      const next = item.spans[spanIndex + 1];
      if (!next) continue;
      for (const gap of item.document.path.steps.filter((step) =>
        step.stepIndex > span.match.episodeEndStepIndex &&
        step.stepIndex < next.match.episodeStartStepIndex)) {
        alignedSequence.push({
          role: "gap",
          afterAnchorId: span.requiredSpan.anchorId,
          beforeAnchorId: next.requiredSpan.anchorId,
          ...skillStep(gap)
        });
      }
    }
    if (alignedSequence.length === 0) return [];
    const stepById = new Map(item.document.path.steps.map((step) => [step.id, step]));
    const relevantSteps = alignedSequence.map((item) => stepById.get(item.stepId))
      .filter((item): item is ExecutionStepLiteV1 => Boolean(item));
    const sourceTraceIds = input.sourceTraceIdsForSteps(
      item.occurrence.episodeId,
      relevantSteps
    );
    const firstIndex = item.orderedMatches[0]!.episodeStartStepIndex;
    const lastIndex = item.orderedMatches.at(-1)!.episodeEndStepIndex;
    return [{
      occurrenceId: item.occurrence.id,
      episodeId: item.occurrence.episodeId,
      pathId: item.occurrence.pathId,
      scale: trajectory.referenceEndStepIndex - trajectory.referenceStartStepIndex + 1,
      alignmentScore: item.occurrence.averageCoarseSimilarity *
        Math.min(item.occurrence.referenceCoverage, item.occurrence.episodeCoverage),
      sourceTraceIds,
      prefixExpansion: [],
      alignedSequence,
      suffixExpansion: [],
      boundaryContextReadOnly: {
        ...(item.document.path.steps[firstIndex - 1]
          ? { previousStep: skillStep(item.document.path.steps[firstIndex - 1]!) }
          : {}),
        ...(item.document.path.steps[lastIndex + 1]
          ? { nextStep: skillStep(item.document.path.steps[lastIndex + 1]!) }
          : {})
      }
    }];
  });
  const sourceTraceIds = unique(evidence.flatMap((item) => item.sourceTraceIds));
  if (sourceTraceIds.length === 0 || evidence.length < 2) return undefined;
  return {
    patternVersionId: `long_trajectory_version_${trajectory.structureHash.slice(0, 24)}`,
    clusterId: trajectory.id,
    clusterVersionId: `long_trajectory_cluster_version_${trajectory.structureHash.slice(0, 24)}`,
    commonCoreId: trajectory.id,
    commonCore,
    completion: {
      id: `long_trajectory_completion_${trajectory.structureHash.slice(0, 24)}`,
      version: "long-trajectory-completion.v1",
      activated: false,
      referenceOccurrenceId: evidence[0]!.occurrenceId,
      maxPrefixSteps: 0,
      maxSuffixSteps: 0,
      minStepSimilarity: 0.7,
      sharedPrefix: [],
      sharedSuffix: [],
      extensionAgreement: 0
    },
    userId: input.userId,
    scale: trajectory.referenceEndStepIndex - trajectory.referenceStartStepIndex + 1,
    supportEpisodeIds: trajectory.supportEpisodeIds,
    sourceTraceIds,
    sourceSpanOccurrenceIds: trajectory.occurrences.map((item) => item.id),
    counterexampleEpisodeIds: [],
    evidence,
    confidenceHint: trajectory.averageEpisodeCoverage * trajectory.averageCoarseSimilarity,
    patternHash: stableHash({
      version: "long-trajectory-coarse-span-evidence.v1",
      trajectoryStructureHash: trajectory.structureHash,
      commonCore,
      occurrences: evidence.map((occurrence) => ({
        episodeId: occurrence.episodeId,
        alignedSequence: occurrence.alignedSequence.map((step) => ({
          stepId: step.stepId,
          role: step.role,
          ...(step.role === "span_step"
            ? { anchorId: step.anchorId, spanSimilarity: step.spanSimilarity }
            : step.role === "gap"
              ? {
                  afterAnchorId: step.afterAnchorId,
                  beforeAnchorId: step.beforeAnchorId
                }
              : {})
        }))
      }))
    }),
    algorithmVersion: "reference-span-sequence-mining.v2",
    episodeContextReadOnly: trajectory.supportEpisodeIds.flatMap((episodeId) => {
      const document = documentByEpisodeId.get(episodeId);
      return document ? [{
        episodeId,
        goal: document.goalText,
        terminalResult: document.terminalResultText
      }] : [];
    }),
    origin: {
      kind: "long_trajectory",
      episodeFamilyId: trajectory.familyId,
      longTrajectoryId: trajectory.id
    }
  };
}

function skillStep(step: ExecutionStepLiteV1): ProceduralSkillEvidenceStep {
  return {
    stepId: step.id,
    stepIndex: step.stepIndex,
    ...(step.toolName ? { toolName: step.toolName } : {}),
    intent: step.intent,
    summary: step.summary,
    outcome: step.outcome,
    evidenceRefs: step.evidenceRefs
  };
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}
