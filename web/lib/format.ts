/** Display helpers shared across pages. */

export function formatUSD(value: number): string {
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  if (value < 1) return `$${value.toFixed(3)}`;
  return `$${value.toFixed(2)}`;
}

export function formatTokens(value: number): string {
  if (value < 1000) return value.toString();
  if (value < 1_000_000) return `${(value / 1000).toFixed(1)}k`;
  return `${(value / 1_000_000).toFixed(2)}M`;
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value: number): string {
  return value.toFixed(3);
}

/** Compact judge id: 'anthropic:claude-sonnet-4-6:rubric_code_v1' -> 'sonnet'. */
export function shortJudge(judgeId: string): string {
  const parts = judgeId.split(":");
  if (parts.length < 2) return judgeId;
  const model = parts[1];
  if (model.includes("claude-haiku")) return "claude-haiku";
  if (model.includes("claude-sonnet")) return "claude-sonnet";
  if (model.includes("claude-opus")) return "claude-opus";
  if (model.includes("gpt-4o-mini")) return "gpt-4o-mini";
  if (model.includes("gpt-4o")) return "gpt-4o";
  if (model.includes("gemini")) return "gemini";
  return model;
}

export function judgeColor(judgeId: string): string {
  const s = shortJudge(judgeId);
  if (s === "claude-haiku") return "text-amber-400";
  if (s === "claude-sonnet") return "text-violet-400";
  if (s === "claude-opus") return "text-fuchsia-400";
  if (s === "gpt-4o-mini") return "text-emerald-400";
  if (s === "gpt-4o") return "text-emerald-500";
  if (s === "gemini") return "text-sky-400";
  return "text-zinc-400";
}

export function judgeChartColor(judgeId: string): string {
  const s = shortJudge(judgeId);
  if (s === "claude-haiku") return "#fbbf24";
  if (s === "claude-sonnet") return "#a78bfa";
  if (s === "claude-opus") return "#e879f9";
  if (s === "gpt-4o-mini") return "#34d399";
  if (s === "gpt-4o") return "#10b981";
  if (s === "gemini") return "#38bdf8";
  return "#a1a1aa";
}

export function gapColor(gapPp: number): "good" | "ok" | "warn" {
  if (gapPp <= 5) return "good";
  if (gapPp <= 10) return "ok";
  return "warn";
}

export function shortItemId(itemId: string): string {
  return itemId.replace(/^calib::/, "calib·").replace(/^HumanEval\//, "HE/");
}
