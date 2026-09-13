#!/usr/bin/env npx tsx
import { MemoryDb, MemoryService, listenMemoryHttpServer } from "../src/index.js";
import { loadEvalConfig } from "./eval-spreadsheet-config.js";

const sqlitePath = process.env.MEMMY_SQLITE_PATH
  ?? (process.env.SPREADSHEET_MEMMY_OUT
    ? `${process.env.SPREADSHEET_MEMMY_OUT}/memory.sqlite`
    : "/root/gyh/memmy_eval/EvoAgentBench_MemOS/outputs/spreadsheet_memmy_direct/memory.sqlite");
const configPath = process.env.MEMMY_AGENT_CONFIG ?? process.env.MEMMY_CONFIG ?? `${process.env.HOME}/.memmy/config.yaml`;
const host = process.env.MEMMY_MEMORY_HOST ?? "127.0.0.1";
const port = Number(process.env.MEMMY_MEMORY_PORT ?? 18961);
const truthy = (value: string | undefined, fallback: boolean) => {
  if (value == null) return fallback;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
};

const config = loadEvalConfig(configPath, sqlitePath);
config.algorithm.enableMemoryAdd = truthy(process.env.MEMMY_ENABLE_ADD, false);
config.algorithm.enableMemorySearch = truthy(process.env.MEMMY_ENABLE_SEARCH, true);
const skillMode = (process.env.MEMMY_SKILL_INJECTION_MODE ?? "full").trim().toLowerCase();
config.algorithm.retrieval.skillInjectionMode = skillMode === "summary" ? "summary" : "full";
const fullMax = Number(process.env.MEMMY_SKILL_FULL_MAX_CHARS ?? 16_384);
if (Number.isFinite(fullMax) && fullMax >= 1000) {
  config.algorithm.retrieval.skillFullMaxChars = Math.floor(fullMax);
}
const db = new MemoryDb({ path: sqlitePath });
const service = new MemoryService({ db, mode: "local", config });
const listening = await listenMemoryHttpServer({ service, host, port });
console.log(`eval memory listening on ${listening.url} db=${sqlitePath}`);
