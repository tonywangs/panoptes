/**
 * TypeScript types mirroring the JSON shapes written by scripts/export_for_web.py.
 * Keep this in sync with that script — the build will TS-error if shapes drift.
 */

export type Summary = {
  n_runs: number;
  n_items_total: number;
  n_calls_total: number;
  cost_total_usd: number;
  judges_seen: string[];
  calibration_headline?: CalibrationHeadline;
};

export type CalibrationHeadline = {
  judge: string;
  alpha: number;
  nominal: number;
  empirical: number;
  gap_pp: number;
  summary: string;
};

export type RunConfig = {
  alpha?: number;
  uq_methods?: string[];
  judges?: string[];
  n_items?: number;
  n_samples?: number;
  temperature_sampling?: number;
  strategy?: "all" | "single" | "escalation" | "bandit";
  model_under_test?: string;
};

export type RunSummary = {
  run_id: string;
  created_at_utc: string | null;
  panoptes_version: string | null;
  config: RunConfig;
  strategy: string;
  n_items: number;
  n_calls: number;
  n_judges: number;
  judges: string[];
  cost_usd: number;
  cost_by_judge: Record<string, number>;
  tokens: { input: number; output: number; cache_read: number; cache_creation: number };
  n_uq_results: number;
  source_file: string;
};

export type EvalRow = {
  item_id: string;
  benchmark: string;
  task_family: string;
  judge_id: string;
  prompt_version_hash: string;
  model_under_test: string;
  model_response: string;
  score_value: number;
  score_scale: string;
  likert: number | null;
  rationale: string;
  flags: string[];
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  cost_usd: number;
  latency_ms: number;
  sample_index: number;
  temperature: number;
  timestamp_utc: string | null;
};

export type UQResult = {
  item_id: string;
  judge_id: string;
  method: string;
  value: Record<string, unknown> | null;
};

/** Stats can be null when an input column is constant (no rank info). */
export type JudgePair = {
  judge_a: string;
  judge_b: string;
  n_items: number;
  spearman: { point: number | null; ci_low: number | null; ci_high: number | null };
  kendall: { point: number | null; ci_low: number | null; ci_high: number | null };
  permutation: { observed: number | null; p_value: number | null };
  pairs: { item_id: string; a: number; b: number }[];
};

export type ParetoPoint = {
  alpha: number;
  target_coverage: number;
  mean_width: number;
  empirical_coverage: number | null;
};

export type ItemPrompt = {
  prompt: string;
  canonical_solution: string;
  entry_point: string;
  source: string;
  original_entry_point?: string;
};

export type CalibrationRow = {
  judge: string;
  alpha: number;
  nominal: number;
  empirical: number;
  gap_pp: number;
  q: number;
  n_cal: number;
  n_test: number;
};

export type CalibrationData = {
  generator: { model: string; temperature: number };
  candidates: { n: number; n_pass: number; n_fail: number; pass_rate: number };
  judges: { judge_id: string; valid_scores: number; total_calls: number; note?: string }[];
  conformal_coverage: CalibrationRow[];
  headline: CalibrationHeadline;
  methodology_notes: string[];
  total_wall_time_s: number;
  source: string;
};
