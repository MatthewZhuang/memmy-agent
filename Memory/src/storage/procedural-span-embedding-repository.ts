import type Database from "better-sqlite3";
import { stableHash } from "../utils/id.js";
import { parseJson, toJson } from "../utils/json.js";

export interface ProceduralSpanOccurrenceEmbeddingRecord {
  id: string;
  occurrenceId: string;
  namespaceId: string;
  projectionVersion: string;
  embeddingVersion: string;
  sourceHash: string;
  embeddingProvider: string;
  embeddingModel: string;
  embeddingDim: number;
  goalVector: number[];
  procedureVector: number[];
  effectVector: number[];
  contentHash: string;
  createdAt: string;
  updatedAt: string;
}

export interface SaveProceduralSpanOccurrenceEmbeddingInput {
  occurrenceId: string;
  namespaceId: string;
  projectionVersion: string;
  embeddingVersion: string;
  sourceHash: string;
  embeddingProvider: string;
  embeddingModel: string;
  goalVector: number[];
  procedureVector: number[];
  effectVector: number[];
  at: string;
}

interface EmbeddingSqlRow {
  id: string;
  occurrence_id: string;
  namespace_id: string;
  projection_version: string;
  embedding_version: string;
  source_hash: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_dim: number;
  goal_vector_json: string;
  procedure_vector_json: string;
  effect_vector_json: string;
  content_hash: string;
  created_at: string;
  updated_at: string;
}

export class ProceduralSpanEmbeddingRepository {
  constructor(private readonly db: Database.Database) {}

  save(input: SaveProceduralSpanOccurrenceEmbeddingInput): {
    record: ProceduralSpanOccurrenceEmbeddingRecord;
    created: boolean;
  } {
    const occurrence = this.db.prepare(
      `SELECT namespace_id, projection_version
       FROM procedural_span_occurrences
       WHERE id = ?`
    ).get(input.occurrenceId) as {
      namespace_id: string;
      projection_version: string;
    } | undefined;
    if (!occurrence) throw new Error(`procedural Span occurrence not found: ${input.occurrenceId}`);
    if (
      occurrence.namespace_id !== input.namespaceId ||
      occurrence.projection_version !== input.projectionVersion
    ) {
      throw new Error(`procedural Span embedding source mismatch: ${input.occurrenceId}`);
    }
    const provider = requiredText(input.embeddingProvider, "embeddingProvider");
    const model = requiredText(input.embeddingModel, "embeddingModel");
    const vectors = [input.goalVector, input.procedureVector, input.effectVector];
    const dimension = input.goalVector.length;
    if (
      dimension === 0 ||
      vectors.some((vector) => vector.length !== dimension || vector.some((value) => !Number.isFinite(value)))
    ) {
      throw new Error(`procedural Span embedding vectors are invalid: ${input.occurrenceId}`);
    }
    const basis = {
      occurrenceId: input.occurrenceId,
      namespaceId: input.namespaceId,
      projectionVersion: input.projectionVersion,
      embeddingVersion: input.embeddingVersion,
      sourceHash: input.sourceHash,
      embeddingProvider: provider,
      embeddingModel: model,
      embeddingDim: dimension,
      goalVector: input.goalVector,
      procedureVector: input.procedureVector,
      effectVector: input.effectVector
    };
    const contentHash = stableHash(basis);
    const existing = this.getByBasis({
      occurrenceId: input.occurrenceId,
      projectionVersion: input.projectionVersion,
      embeddingVersion: input.embeddingVersion,
      sourceHash: input.sourceHash,
      embeddingProvider: provider,
      embeddingModel: model
    });
    if (existing) {
      if (existing.contentHash !== contentHash) {
        throw new Error(`procedural Span embedding basis produced different vectors: ${input.occurrenceId}`);
      }
      return { record: existing, created: false };
    }
    const id = `procedural_span_embedding_${contentHash.slice(0, 24)}`;
    this.db.prepare(
      `INSERT INTO procedural_span_occurrence_embeddings (
        id, occurrence_id, namespace_id, projection_version, embedding_version,
        source_hash, embedding_provider, embedding_model, embedding_dim,
        goal_vector_json, procedure_vector_json, effect_vector_json,
        content_hash, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      id,
      input.occurrenceId,
      input.namespaceId,
      input.projectionVersion,
      input.embeddingVersion,
      input.sourceHash,
      provider,
      model,
      dimension,
      toJson(input.goalVector),
      toJson(input.procedureVector),
      toJson(input.effectVector),
      contentHash,
      input.at,
      input.at
    );
    return { record: this.get(id)!, created: true };
  }

  get(id: string): ProceduralSpanOccurrenceEmbeddingRecord | undefined {
    const row = this.db.prepare(
      `SELECT * FROM procedural_span_occurrence_embeddings WHERE id = ?`
    ).get(id) as EmbeddingSqlRow | undefined;
    return row ? fromSql(row) : undefined;
  }

  getByBasis(input: {
    occurrenceId: string;
    projectionVersion: string;
    embeddingVersion: string;
    sourceHash: string;
    embeddingProvider: string;
    embeddingModel: string;
  }): ProceduralSpanOccurrenceEmbeddingRecord | undefined {
    const row = this.db.prepare(
      `SELECT *
       FROM procedural_span_occurrence_embeddings
       WHERE occurrence_id = ?
         AND projection_version = ?
         AND embedding_version = ?
         AND source_hash = ?
         AND embedding_provider = ?
         AND embedding_model = ?
       LIMIT 1`
    ).get(
      input.occurrenceId,
      input.projectionVersion,
      input.embeddingVersion,
      input.sourceHash,
      input.embeddingProvider,
      input.embeddingModel
    ) as EmbeddingSqlRow | undefined;
    return row ? fromSql(row) : undefined;
  }

  listByOccurrenceIds(
    occurrenceIds: readonly string[],
    input: {
      embeddingVersion: string;
      embeddingProvider: string;
      embeddingModel: string;
    }
  ): ProceduralSpanOccurrenceEmbeddingRecord[] {
    const ids = [...new Set(occurrenceIds)];
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => "?").join(", ");
    return (this.db.prepare(
      `SELECT *
       FROM procedural_span_occurrence_embeddings
       WHERE occurrence_id IN (${placeholders})
         AND embedding_version = ?
         AND embedding_provider = ?
         AND embedding_model = ?
       ORDER BY occurrence_id ASC, updated_at DESC, id DESC`
    ).all(
      ...ids,
      input.embeddingVersion,
      input.embeddingProvider,
      input.embeddingModel
    ) as EmbeddingSqlRow[]).map(fromSql);
  }
}

function fromSql(row: EmbeddingSqlRow): ProceduralSpanOccurrenceEmbeddingRecord {
  return {
    id: row.id,
    occurrenceId: row.occurrence_id,
    namespaceId: row.namespace_id,
    projectionVersion: row.projection_version,
    embeddingVersion: row.embedding_version,
    sourceHash: row.source_hash,
    embeddingProvider: row.embedding_provider,
    embeddingModel: row.embedding_model,
    embeddingDim: row.embedding_dim,
    goalVector: parseJson<number[]>(row.goal_vector_json, []),
    procedureVector: parseJson<number[]>(row.procedure_vector_json, []),
    effectVector: parseJson<number[]>(row.effect_vector_json, []),
    contentHash: row.content_hash,
    createdAt: row.created_at,
    updatedAt: row.updated_at
  };
}

function requiredText(value: string, field: string): string {
  const normalized = value.trim();
  if (!normalized) throw new Error(`procedural Span embedding ${field} is required`);
  return normalized;
}
