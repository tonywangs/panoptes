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

  // Sizes chosen so the rotated column labels never overflow.
  const cellW = 72;
  const cellH = 48;
  const headerH = 140;
  const rowLabelW = 200;

  return (
    <div className="overflow-x-auto">
      <table
        className="border-separate"
        style={{ borderSpacing: 4, marginTop: 8 }}
      >
        <thead>
          <tr style={{ height: headerH }}>
            <th style={{ width: rowLabelW }} />
            {judges.map((j) => (
              <th
                key={j}
                style={{ width: cellW, height: headerH, verticalAlign: "bottom", padding: 0 }}
              >
                <div
                  style={{
                    width: cellW,
                    height: headerH,
                    position: "relative",
                  }}
                >
                  <div
                    className="font-mono text-xs"
                    style={{
                      position: "absolute",
                      left: cellW / 2,
                      bottom: 6,
                      transform: "rotate(-50deg) translateY(-50%)",
                      transformOrigin: "left bottom",
                      whiteSpace: "nowrap",
                      color: "var(--foreground-muted)",
                    }}
                    title={shortJudge(j)}
                  >
                    {shortJudge(j)}
                  </div>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {judges.map((row) => (
            <tr key={row}>
              <th
                style={{
                  width: rowLabelW,
                  height: cellH,
                  textAlign: "right",
                  paddingRight: 12,
                  fontWeight: 400,
                }}
              >
                <JudgeBadge judgeId={row} />
              </th>
              {judges.map((col) => {
                const rho = row === col ? 1 : (lookup.get(`${row}|${col}`) ?? null);
                return (
                  <td
                    key={col}
                    className="rounded-md text-center text-sm font-mono tabular-nums"
                    style={{
                      width: cellW,
                      height: cellH,
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
      <div className="flex items-center gap-3 mt-3 text-xs muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-3 h-3 rounded" style={{ background: "#10b981" }} />
          high agreement
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-3 h-3 rounded" style={{ background: "color-mix(in srgb, #10b981 40%, var(--surface))" }} />
          partial
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-3 h-3 rounded" style={{ background: "var(--surface-2)" }} />
          low / no signal
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-3 h-3 rounded" style={{ background: "#f43f5e" }} />
          disagreement
        </span>
      </div>
    </div>
  );
}
