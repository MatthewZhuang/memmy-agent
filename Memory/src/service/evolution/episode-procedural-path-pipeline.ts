import type { EvolutionJobRecord, RawTurnRecord, Repositories } from "../../storage/repositories.js";
import type { SaveEpisodeProceduralPathResult } from "../../storage/procedural-path-repository.js";
import { nowIso } from "../../utils/time.js";
import { proceduralLearningScopeIdForSession } from "../namespace/namespace-scope.js";
import { EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION } from "./episode-procedural-reconstructor.js";
import type { EpisodeProceduralPathV2 } from "./procedural-path-model.js";
import { spanCreditRewardHash } from "./span-credit-pipeline.js";
import type { EnqueueJobInput } from "../worker/job-handlers.js";
import { enqueueEpisodePolicyProjection } from "./episode-policy-projection.js";

export interface EpisodeProceduralPathReconstructor {
  reconstruct(input: {
    episodeId: string;
    rawTurns: readonly RawTurnRecord[];
    terminalReward?: number;
  }): Promise<EpisodeProceduralPathV2>;
}

export interface EpisodeProceduralPathPersistencePipelineDeps {
  repos: Repositories;
  reconstructor: EpisodeProceduralPathReconstructor;
  enqueueJob?(input: EnqueueJobInput): EvolutionJobRecord;
}

export class EpisodeProceduralPathPersistencePipeline {
  constructor(private readonly deps: EpisodeProceduralPathPersistencePipelineDeps) {}

  async reconstructJob(job: EvolutionJobRecord): Promise<SaveEpisodeProceduralPathResult | undefined> {
    if (!job.episodeId) throw new Error(`procedural path job missing episodeId: ${job.id}`);
    const episode = this.deps.repos.runtime.getEpisode(job.episodeId);
    if (!episode) throw new Error(`procedural path source episode not found: ${job.episodeId}`);
    const queuedRewardHash = typeof job.payload.rewardHash === "string"
      ? job.payload.rewardHash
      : undefined;
    if (queuedRewardHash && queuedRewardHash !== spanCreditRewardHash(episode)) return undefined;
    const activePath = this.deps.repos.proceduralPaths.getActiveForEpisode(episode.id);
    if (
      activePath &&
      activePath.reconstructionAlgorithmVersion === EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION
    ) {
      this.enqueueSpanCredit(episode, activePath, job.updatedAt);
      this.enqueueProjection(episode.id, job.updatedAt, "procedural_path_reused");
      return undefined;
    }
    return this.reconstructAndPersist({
      episodeId: job.episodeId,
      createdAt: job.updatedAt
    });
  }

  async reconstructAndPersist(input: {
    episodeId: string;
    activate?: boolean;
    createdAt?: string;
  }): Promise<SaveEpisodeProceduralPathResult> {
    const episode = this.deps.repos.runtime.getEpisode(input.episodeId);
    if (!episode) throw new Error(`procedural path source episode not found: ${input.episodeId}`);
    if (episode.status !== "closed") {
      throw new Error(`procedural path reconstruction requires a closed Episode: ${input.episodeId}`);
    }
    const session = this.deps.repos.runtime.getSession(episode.sessionId);
    if (!session || session.userId !== episode.userId) {
      throw new Error(`procedural path source Session is missing or inconsistent: ${episode.sessionId}`);
    }
    const limit = Math.max(100, episode.turnCount, episode.rawTurnIds.length);
    const rawTurns = this.deps.repos.runtime.listRawTurnsByEpisode(episode.id, limit);
    const loadedIds = new Set(rawTurns.map((turn) => turn.id));
    const missingRawTurnIds = episode.rawTurnIds.filter((rawTurnId) => !loadedIds.has(rawTurnId));
    if (missingRawTurnIds.length > 0) {
      throw new Error(`procedural path source Episode is missing RawTurns: ${missingRawTurnIds.join(", ")}`);
    }
    const path = await this.deps.reconstructor.reconstruct({
      episodeId: episode.id,
      rawTurns,
      ...(episode.rTask === undefined ? {} : { terminalReward: episode.rTask })
    });
    if (path.episodeId !== episode.id) {
      throw new Error(`procedural path reconstructor returned another Episode: ${path.episodeId}`);
    }
    const createdAt = input.createdAt ?? nowIso();
    const saved = this.deps.repos.proceduralPaths.save({
      path,
      namespaceId: proceduralLearningScopeIdForSession(session),
      reconstructionAlgorithmVersion: EPISODE_PROCEDURAL_RECONSTRUCTION_VERSION,
      activate: input.activate,
      createdAt
    });
    if (saved.record.status === "active") {
      this.enqueueSpanCredit(episode, saved.record, createdAt);
      this.enqueueProjection(episode.id, createdAt, "procedural_path_activated");
    }
    return saved;
  }

  private enqueueSpanCredit(
    episode: NonNullable<ReturnType<Repositories["runtime"]["getEpisode"]>>,
    path: SaveEpisodeProceduralPathResult["record"],
    createdAt: string
  ): void {
    if (episode.rTask === undefined || !this.deps.enqueueJob) return;
    const occurrences = this.deps.repos.proceduralPaths.listOccurrencesForPath(path.id);
    if (occurrences.length === 0) return;
    this.deps.enqueueJob({
      jobType: "span_credit",
      userId: episode.userId,
      sessionId: episode.sessionId,
      episodeId: episode.id,
      payload: {
        pathId: path.id,
        pathHash: path.pathHash,
        rewardHash: spanCreditRewardHash(episode)
      },
      createdAt
    });
  }

  private enqueueProjection(episodeId: string, at: string, trigger: string): void {
    if (!this.deps.enqueueJob) return;
    enqueueEpisodePolicyProjection({
      repos: this.deps.repos,
      enqueueJob: this.deps.enqueueJob
    }, { episodeId, at, trigger });
  }
}
