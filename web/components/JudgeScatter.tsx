"use client";

import {
  CartesianGrid,
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

export function JudgeScatter({
  judgeA,
  judgeB,
  pairs,
}: {
  judgeA: string;
  judgeB: string;
  pairs: { item_id: string; a: number; b: number }[];
}) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 5, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            type="number"
            dataKey="a"
            domain={[0, 1]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{ value: shortJudge(judgeA), position: "insideBottom", offset: -2, fill: "var(--foreground-muted)", fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="b"
            domain={[0, 1]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{ value: shortJudge(judgeB), angle: -90, position: "insideLeft", fill: "var(--foreground-muted)", fontSize: 11, offset: 18 }}
          />
          <ZAxis range={[50, 50]} />
          <Tooltip
            cursor={{ stroke: "var(--border)" }}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <ReferenceLine
            stroke="var(--foreground-muted)"
            strokeDasharray="3 3"
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
          />
          <Scatter data={pairs} fill={judgeChartColor(judgeA)} fillOpacity={0.7} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
