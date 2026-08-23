import { stableHash } from "../../utils/id.js";

export const PROCEDURAL_POLICY_SCHEMA_VERSION = "procedural-policy.v1" as const;
export const PROCEDURAL_POLICY_INDUCTION_VERSION = "procedural-policy-induction.v1" as const;
export const PROCEDURAL_POLICY_PROMPT_VERSION = "procedural-policy-prompt.v1" as const;

export interface ProceduralPolicyEvidenceStepV1 {
  instruction: string;
  evidenceRefs: string[];
}

export interface ProceduralPolicyRecoveryRuleV1 {
  condition: string;
  action: string;
  evidenceRefs: string[];
}

export interface ProceduralPolicyVerificationStepV1 {
  check: string;
  successSignal: string;
  evidenceRefs: string[];
}

export interface ProceduralPolicyDraftV1 {
  title: string;
  goalPattern: string;
  triggerConditions: string[];
  procedureSteps: ProceduralPolicyEvidenceStepV1[];
  recoveryRules: ProceduralPolicyRecoveryRuleV1[];
  verificationSteps: ProceduralPolicyVerificationStepV1[];
  applyWhen: string[];
  doNotApplyWhen: string[];
  invariants: string[];
  expectedEffect: string;
  evidenceOccurrenceIds: string[];
  confidence: number;
}

export interface ProceduralPolicyEvidenceV1 {
  occurrenceIds: string[];
  supportOccurrenceIds: string[];
  counterexampleOccurrenceIds: string[];
  supportEpisodeIds: string[];
  counterexampleEpisodeIds: string[];
  pathIds: string[];
  spanIds: string[];
  sessionIds: string[];
}

export interface ProceduralPolicyProvenanceV1 {
  inductionVersion: typeof PROCEDURAL_POLICY_INDUCTION_VERSION;
  promptVersion: typeof PROCEDURAL_POLICY_PROMPT_VERSION;
  model?: string;
  evidenceHash: string;
}

export interface ProceduralPolicyV1 extends ProceduralPolicyDraftV1 {
  id: string;
  schemaVersion: typeof PROCEDURAL_POLICY_SCHEMA_VERSION;
  policyKey: string;
  namespaceId: string;
  clusterId: string;
  clusterMembershipVersion: string;
  evidence: ProceduralPolicyEvidenceV1;
  provenance: ProceduralPolicyProvenanceV1;
  contentHash: string;
}

export function buildProceduralPolicy(input: {
  namespaceId: string;
  clusterId: string;
  clusterMembershipVersion: string;
  draft: ProceduralPolicyDraftV1;
  occurrenceIds: string[];
  supportOccurrenceIds: string[];
  counterexampleOccurrenceIds: string[];
  supportEpisodeIds: string[];
  counterexampleEpisodeIds: string[];
  pathIds: string[];
  spanIds: string[];
  sessionIds: string[];
  model?: string;
}): ProceduralPolicyV1 {
  const policyKey = `policy:procedural-span-cluster:${input.clusterId}`;
  const evidence = {
    occurrenceIds: uniqueSorted(input.occurrenceIds),
    supportOccurrenceIds: uniqueSorted(input.supportOccurrenceIds),
    counterexampleOccurrenceIds: uniqueSorted(input.counterexampleOccurrenceIds),
    supportEpisodeIds: uniqueSorted(input.supportEpisodeIds),
    counterexampleEpisodeIds: uniqueSorted(input.counterexampleEpisodeIds),
    pathIds: uniqueSorted(input.pathIds),
    spanIds: uniqueSorted(input.spanIds),
    sessionIds: uniqueSorted(input.sessionIds)
  };
  const evidenceHash = stableHash({
    clusterId: input.clusterId,
    membershipVersion: input.clusterMembershipVersion,
    evidence
  });
  const contentHash = stableHash(input.draft);
  const id = `procedural_policy_${stableHash({
    schemaVersion: PROCEDURAL_POLICY_SCHEMA_VERSION,
    inductionVersion: PROCEDURAL_POLICY_INDUCTION_VERSION,
    clusterId: input.clusterId,
    membershipVersion: input.clusterMembershipVersion,
    evidenceHash,
    contentHash
  }).slice(0, 20)}`;
  return {
    id,
    schemaVersion: PROCEDURAL_POLICY_SCHEMA_VERSION,
    policyKey,
    namespaceId: input.namespaceId,
    clusterId: input.clusterId,
    clusterMembershipVersion: input.clusterMembershipVersion,
    ...input.draft,
    evidence,
    provenance: {
      inductionVersion: PROCEDURAL_POLICY_INDUCTION_VERSION,
      promptVersion: PROCEDURAL_POLICY_PROMPT_VERSION,
      ...(input.model ? { model: input.model } : {}),
      evidenceHash
    },
    contentHash
  };
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}
