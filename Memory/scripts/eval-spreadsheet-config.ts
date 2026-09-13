import { existsSync, readFileSync } from "node:fs";
import { parse as parseYaml } from "yaml";
import { DEFAULT_MEMMY_CONFIG, type MemmyConfig } from "../src/index.js";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

export function loadEvalConfig(agentConfigPath: string, sqlitePath: string): MemmyConfig {
  const root = existsSync(agentConfigPath)
    ? asRecord(parseYaml(readFileSync(agentConfigPath, "utf8")))
    : {};
  const memory = asRecord(root.memmyMemory);
  const byok = asRecord(asRecord(memory.profiles).byok);
  const embedding = asRecord(byok.embedding);
  const evolution = asRecord(byok.evolution);
  const summary = asRecord(byok.summary);
  return {
    ...DEFAULT_MEMMY_CONFIG,
    storage: {
      ...DEFAULT_MEMMY_CONFIG.storage,
      mode: "local",
      backend: "sqlite",
      sqlitePath
    },
    embedding: {
      ...DEFAULT_MEMMY_CONFIG.embedding,
      provider: (embedding.provider as MemmyConfig["embedding"]["provider"]) || "openai_compatible",
      endpoint: String(embedding.endpoint ?? DEFAULT_MEMMY_CONFIG.embedding.endpoint ?? ""),
      model: String(embedding.model ?? "bge-m3"),
      apiKey: String(embedding.apiKey ?? process.env.MEMMY_EMBEDDING_API_KEY ?? "EMPTY")
    },
    evolution: {
      ...DEFAULT_MEMMY_CONFIG.evolution,
      provider: (evolution.provider as MemmyConfig["evolution"]["provider"]) || "openai_compatible",
      vendor: typeof evolution.vendor === "string" ? evolution.vendor as MemmyConfig["evolution"]["vendor"] : DEFAULT_MEMMY_CONFIG.evolution.vendor,
      endpoint: String(evolution.endpoint ?? process.env.JUDGE_API_BASE ?? DEFAULT_MEMMY_CONFIG.evolution.endpoint ?? ""),
      model: String(evolution.model ?? "qwen3.6-flash"),
      apiKey: String(evolution.apiKey ?? process.env.JUDGE_API_KEY ?? process.env.MEMMY_EVOLUTION_API_KEY ?? ""),
      enableThinking: false
    },
    summary: {
      ...DEFAULT_MEMMY_CONFIG.summary,
      provider: (summary.provider as MemmyConfig["summary"]["provider"]) || DEFAULT_MEMMY_CONFIG.summary.provider,
      endpoint: String(summary.endpoint ?? DEFAULT_MEMMY_CONFIG.summary.endpoint ?? ""),
      model: String(summary.model ?? DEFAULT_MEMMY_CONFIG.summary.model ?? ""),
      apiKey: String(summary.apiKey ?? process.env.JUDGE_API_KEY ?? ""),
      enableThinking: false
    },
    algorithm: {
      ...DEFAULT_MEMMY_CONFIG.algorithm,
      enableMemoryAdd: true,
      enableMemorySearch: true,
      retrieval: {
        ...DEFAULT_MEMMY_CONFIG.algorithm.retrieval,
        skillInjectionMode: "full",
        skillFullMaxChars: 16_384
      },
      skill: {
        ...DEFAULT_MEMMY_CONFIG.algorithm.skill,
        directFromTrace: true,
        useLlm: true,
        clusterJoinThreshold: 0.5,
        clusterJoinThresholdEmpty: 0.7
      }
    }
  };
}
