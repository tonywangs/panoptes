"use client";

import {
  CartesianGrid,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { judgeChartColor, shortJudge } from "@/lib/format";
import { TOOLTIP_CONTENT_STYLE, TOOLTIP_ITEM_STYLE, TOOLTIP_LABEL_STYLE } from "@/lib/chart";

type Group = {
  judge: string;
  values: number[];
};

/**
 * One row per judge showing all sampling-pass scores as dots on the [0, 1]
 * axis, plus a vertical bar at the mean and a shaded ±1σ band. Communicates
 * "did the judge agree with itself across draws?" much better than a single
 * mean number.
 */
export function SamplingDistribution({ groups }: { groups: Group[] }) {
  if (groups.length === 0) return null;
  // y axis is judge index; jitter dots a bit so overlapping samples are visible
  const data = groups.flatMap((g, i) =>
    g.values.map((v, k) => ({
      x: v,
      y: i + 0.05 * Math.sin((k + 1) * 7),
      judge: g.judge,
      idx: k,
    })),
  );
  const stats = groups.map((g) => {
    const n = g.values.length;
    const mean = g.values.reduce((s, v) => s + v, 0) / n;
    const variance =
      n > 1 ? g.values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
    const sd = Math.sqrt(variance);
    return { judge: g.judge, mean, sd };
  });

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 16, right: 30, bottom: 28, left: 90 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{
              value: "score",
              position: "insideBottom",
              offset: -14,
              fill: "var(--foreground-muted)",
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[-0.6, groups.length - 0.4]}
            ticks={groups.map((_, i) => i)}
            tickFormatter={(v: number) => shortJudge(groups[v]?.judge ?? "")}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            width={80}
          />
          <ZAxis range={[40, 40]} />
          <Tooltip
            cursor={false}
            contentStyle={TOOLTIP_CONTENT_STYLE}
            itemStyle={TOOLTIP_ITEM_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(v: number, name: string) =>
              name === "x" ? [(v as number).toFixed(3), "score"] : [v, name]
            }
            labelFormatter={() => ""}
          />
          {stats.map((s, i) => (
            <ReferenceArea
              key={`band-${s.judge}`}
              x1={Math.max(0, s.mean - s.sd)}
              x2={Math.min(1, s.mean + s.sd)}
              y1={i - 0.25}
              y2={i + 0.25}
              fill={judgeChartColor(s.judge)}
              fillOpacity={0.12}
              stroke="none"
              ifOverflow="hidden"
            />
          ))}
          {stats.map((s, i) => (
            <ReferenceLine
              key={`mean-${s.judge}`}
              segment={[
                { x: s.mean, y: i - 0.25 },
                { x: s.mean, y: i + 0.25 },
              ]}
              stroke={judgeChartColor(s.judge)}
              strokeWidth={2}
            />
          ))}
          {groups.map((g) => (
            <Scatter
              key={g.judge}
              data={data.filter((d) => d.judge === g.judge)}
              fill={judgeChartColor(g.judge)}
              fillOpacity={0.85}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
