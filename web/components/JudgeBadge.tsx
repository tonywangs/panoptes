import { judgeColor, shortJudge } from "@/lib/format";

export function JudgeBadge({ judgeId }: { judgeId: string }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium font-mono"
      style={{ background: "var(--surface-2)" }}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${judgeColor(judgeId).replace("text-", "bg-")}`} />
      <span className={judgeColor(judgeId)}>{shortJudge(judgeId)}</span>
    </span>
  );
}

export function ScoreBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div
        className="h-1.5 w-24 rounded-full overflow-hidden"
        style={{ background: "var(--surface-2)" }}
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background:
              value >= 0.7
                ? "#10b981"
                : value >= 0.4
                  ? "#f59e0b"
                  : "#f43f5e",
          }}
        />
      </div>
      <span className="text-sm font-mono tabular-nums">{value.toFixed(3)}</span>
    </div>
  );
}
