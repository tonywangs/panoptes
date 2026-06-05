"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { judgeChartColor, shortJudge } from "@/lib/format";
import { TOOLTIP_CONTENT_STYLE, TOOLTIP_ITEM_STYLE, TOOLTIP_LABEL_STYLE } from "@/lib/chart";

type Row = {
  judge: string;
  alpha: number;
  nominal: number;
  empirical: number;
};

/**
 * Empirical coverage vs nominal coverage, by judge. The diagonal is "perfect"
 * calibration; the shaded region above is "overcovers" (safe, conservative),
 * and below is "undercovers" (the bad direction). Points on or near the diagonal
 * are the ideal — and that's the conformal-prediction guarantee.
 */
export function ReliabilityDiagram({ rows }: { rows: Row[] }) {
  const judges = Array.from(new Set(rows.map((r) => r.judge))).sort();
  // Build a denser "diagonal + shaded" trace; one diagonal line covering [0, 1].
  const diagonal = Array.from({ length: 41 }, (_, i) => {
    const x = i / 40;
    return { x, perfect: x, safe: 1 };
  });

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <ComposedChart margin={{ top: 12, right: 20, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0.6, 1]}
            ticks={[0.6, 0.7, 0.8, 0.9, 0.95, 1]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{
              value: "nominal coverage (1 − α)",
              position: "insideBottom",
              offset: -10,
              fill: "var(--foreground-muted)",
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            domain={[0.6, 1.0]}
            ticks={[0.6, 0.7, 0.8, 0.9, 1]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{
              value: "empirical coverage",
              angle: -90,
              position: "insideLeft",
              offset: 18,
              fill: "var(--foreground-muted)",
              fontSize: 11,
            }}
          />
          <Tooltip
            contentStyle={TOOLTIP_CONTENT_STYLE}
            itemStyle={TOOLTIP_ITEM_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(v: number, name: string) =>
              typeof v === "number" ? [v.toFixed(3), name] : [v, name]
            }
            labelFormatter={() => ""}
          />
          {/* Safe-zone: empirical >= nominal (conformal lower bound holds) */}
          <Area
            data={diagonal}
            dataKey="safe"
            type="linear"
            stroke="none"
            fill="#10b98122"
            isAnimationActive={false}
            name="safe (≥ nominal)"
          />
          <Line
            data={diagonal}
            dataKey="perfect"
            type="linear"
            stroke="#a1a1aa"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            name="perfect (1−α line)"
          />
          {judges.map((j) => {
            const pts = rows
              .filter((r) => r.judge === j)
              .map((r) => ({ x: r.nominal, y: r.empirical }));
            return (
              <Scatter
                key={j}
                data={pts}
                fill={judgeChartColor(j)}
                stroke={judgeChartColor(j)}
                shape="circle"
                name={shortJudge(j)}
              />
            );
          })}
          <ReferenceLine x={0.9} stroke="var(--border)" strokeDasharray="2 2" />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-3 justify-center mt-2 text-xs">
        {judges.map((j) => (
          <span key={j} className="inline-flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ background: judgeChartColor(j) }}
            />
            <span className="muted">{shortJudge(j)}</span>
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <span className="w-3 h-2 rounded" style={{ background: "#10b98122" }} />
          <span className="muted">overcoverage (safe)</span>
        </span>
      </div>
    </div>
  );
}
