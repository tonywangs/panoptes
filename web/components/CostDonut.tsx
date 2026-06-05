"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { formatUSD, judgeChartColor, shortJudge } from "@/lib/format";

export function CostDonut({ data }: { data: Record<string, number> }) {
  const rows = Object.entries(data).map(([judge, cost]) => ({
    judge,
    name: shortJudge(judge),
    value: cost,
    color: judgeChartColor(judge),
  }));
  const total = rows.reduce((s, r) => s + r.value, 0);
  return (
    <div className="relative h-56 w-full">
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={rows}
            dataKey="value"
            nameKey="name"
            innerRadius="58%"
            outerRadius="86%"
            paddingAngle={2}
            stroke="var(--surface)"
            strokeWidth={2}
          >
            {rows.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number) => formatUSD(value)}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <div className="text-xs muted">total</div>
        <div className="text-xl font-semibold tabular-nums">{formatUSD(total)}</div>
      </div>
    </div>
  );
}
