"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TOOLTIP_CONTENT_STYLE, TOOLTIP_ITEM_STYLE, TOOLTIP_LABEL_STYLE } from "@/lib/chart";

type Point = { alpha: number; target_coverage: number; mean_width: number; empirical_coverage: number | null };

export function ParetoChart({ data }: { data: Point[] }) {
  const sorted = [...data].sort((a, b) => a.alpha - b.alpha);
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <LineChart data={sorted} margin={{ top: 10, right: 20, bottom: 5, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="alpha"
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{ value: "α", position: "insideBottom", fill: "var(--foreground-muted)", fontSize: 11, offset: -2 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{ value: "coverage", angle: -90, position: "insideLeft", fill: "var(--foreground-muted)", fontSize: 11, offset: 18 }}
          />
          <Tooltip
            contentStyle={TOOLTIP_CONTENT_STYLE}
            itemStyle={TOOLTIP_ITEM_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(v: number) => (typeof v === "number" ? v.toFixed(3) : v)}
          />
          <ReferenceLine y={1} stroke="var(--border)" strokeDasharray="2 2" />
          <Line
            dataKey="target_coverage"
            stroke="#a1a1aa"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            name="nominal (1−α)"
          />
          <Line
            dataKey="empirical_coverage"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ r: 3 }}
            name="empirical"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
