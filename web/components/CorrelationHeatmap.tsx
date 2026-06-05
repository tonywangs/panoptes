import { JudgeBadge } from "@/components/JudgeBadge";
import { shortJudge } from "@/lib/format";
import { JudgePair } from "@/lib/types";

/**
 * Visual matrix of pairwise Spearman ρ. Stronger color = higher correlation.
 * Diagonal is always 1.0 by definition (judge vs itself).
 */
export function CorrelationHeatmap({
  judges,
  pairs,
}: {
  judges: string[];
  pairs: JudgePair[];
}) {
  const lookup = new Map<string, number | null>();
  for (const p of pairs) {
    lookup.set(`${p.judge_a}|${p.judge_b}`, p.spearman.point);
    lookup.set(`${p.judge_b}|${p.judge_a}`, p.spearman.point);
  }

  function bgFor(rho: number | null): string {
    if (rho === null || Number.isNaN(rho)) return "var(--surface-2)";
    // map [-1, 1] to color intensity. Emerald for positive, rose for negative.
    if (rho >= 0) {
      const a = Math.min(1, Math.abs(rho));
      return `color-mix(in srgb, #10b981 ${(a * 100).toFixed(0)}%, var(--surface))`;
    }
    const a = Math.min(1, Math.abs(rho));
    return `color-mix(in srgb, #f43f5e ${(a * 100).toFixed(0)}%, var(--surface))`;
  }

  function textColor(rho: number | null): string {
    if (rho === null) return "var(--foreground-muted)";
    return Math.abs(rho) > 0.55 ? "white" : "var(--foreground)";
  }

  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 text-sm">
        <thead>
          <tr>
            <th></th>
            {judges.map((j) => (
              <th key={j} className="px-2 py-1 text-xs font-normal align-bottom">
                <div className="-rotate-45 origin-bottom-left translate-y-2 whitespace-nowrap">
                  <span className="font-mono">{shortJudge(j)}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {judges.map((row) => (
            <tr key={row}>
              <th className="px-3 py-2 text-xs font-normal text-right">
                <JudgeBadge judgeId={row} />
              </th>
              {judges.map((col) => {
                const rho = row === col ? 1 : (lookup.get(`${row}|${col}`) ?? null);
                return (
                  <td
                    key={col}
                    className="h-12 w-16 rounded-md text-center text-sm font-mono tabular-nums"
                    style={{
                      background: bgFor(rho),
                      color: textColor(rho),
                      border: row === col ? "2px dashed var(--border)" : undefined,
                    }}
                    title={`${shortJudge(row)} vs ${shortJudge(col)}: ρ = ${rho ?? "n/a"}`}
                  >
                    {rho === null ? "—" : rho.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
