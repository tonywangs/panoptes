/**
 * Build-time JSON loaders. Every page calls these in a Server Component;
 * Next.js's static generation runs them once at build time and caches.
 */

import fs from "node:fs";
import path from "node:path";
import {
  CalibrationData,
  EvalRow,
  ItemPrompt,
  JudgePair,
  ParetoPoint,
  RunSummary,
  Summary,
  UQResult,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

function readJSON<T>(filename: string): T {
  const p = path.join(DATA_DIR, filename);
  return JSON.parse(fs.readFileSync(p, "utf-8")) as T;
}

function readJSONOrNull<T>(filename: string): T | null {
  const p = path.join(DATA_DIR, filename);
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf-8")) as T;
}

export function loadSummary(): Summary {
  return readJSON<Summary>("summary.json");
}

export function loadRuns(): RunSummary[] {
  return readJSON<RunSummary[]>("runs.json");
}

/** Substantive runs only — filters out smoke/test data. */
export function loadHeadlineRuns(): RunSummary[] {
  return loadRuns().filter((r) => r.n_items >= 10);
}

export function loadRun(runId: string): RunSummary | null {
  return loadRuns().find((r) => r.run_id === runId) ?? null;
}

export function loadRows(runId: string): EvalRow[] {
  return readJSONOrNull<EvalRow[]>(`rows-${runId}.json`) ?? [];
}

export function loadUQ(runId: string): UQResult[] {
  return readJSONOrNull<UQResult[]>(`uq-${runId}.json`) ?? [];
}

export function loadJudgePairs(runId: string): JudgePair[] {
  return readJSONOrNull<JudgePair[]>(`judge-pairs-${runId}.json`) ?? [];
}

export function loadPareto(runId: string): ParetoPoint[] {
  return readJSONOrNull<ParetoPoint[]>(`pareto-${runId}.json`) ?? [];
}

export function loadItemPrompts(): Record<string, ItemPrompt> {
  return readJSONOrNull<Record<string, ItemPrompt>>("items.json") ?? {};
}

export function loadCalibration(): CalibrationData | null {
  return readJSONOrNull<CalibrationData>("calibration.json");
}

/** Find the run that scored a given item (first match wins). */
export function findItemSource(itemId: string): { run: RunSummary; rows: EvalRow[] } | null {
  for (const run of loadHeadlineRuns()) {
    const rows = loadRows(run.run_id).filter((r) => r.item_id === itemId);
    if (rows.length > 0) {
      return { run, rows };
    }
  }
  // Fallback to non-headline runs
  for (const run of loadRuns()) {
    const rows = loadRows(run.run_id).filter((r) => r.item_id === itemId);
    if (rows.length > 0) {
      return { run, rows };
    }
  }
  return null;
}

/** Distinct item ids that appear in any headline run. */
export function loadAllItemIds(): string[] {
  const ids = new Set<string>();
  for (const run of loadHeadlineRuns()) {
    for (const row of loadRows(run.run_id)) {
      ids.add(row.item_id);
    }
  }
  return Array.from(ids).sort();
}
