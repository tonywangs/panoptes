"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { judgeChartColor, shortJudge } from "@/lib/format";

type RawRow = { judge_id: string; score_value: number; sample_index: number };

const BIN_EDGES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01];

function binLabel(i: number): string {
  return `${BIN_EDGES[i].toFixed(1)}–${BIN_EDGES[i + 1].toFixed(1)}`;
}

export function ScoreHistogram({ rows }: { rows: RawRow[] }) {
  const judges = Array.from(new Set(rows.filter((r) => r.sample_index === 0).map((r) => r.judge_id))).sort();
  const bins = BIN_EDGES.slice(0, -1).map((_, i) => {
    const row: Record<string, string | number> = { bin: binLabel(i) };
    for (const j of judges) row[shortJudge(j)] = 0;
    return row;
  });
  for (const r of rows) {
    if (r.sample_index !== 0) continue;
    const idx = BIN_EDGES.findIndex((edge, i) => i < BIN_EDGES.length - 1 && r.score_value < BIN_EDGES[i + 1]);
    if (idx === -1) continue;
    bins[idx][shortJudge(r.judge_id)] = (bins[idx][shortJudge(r.judge_id)] as number) + 1;
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={bins} margin={{ top: 5, right: 0, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis dataKey="bin" tick={{ fill: "var(--foreground-muted)", fontSize: 11 }} />
          <YAxis tick={{ fill: "var(--foreground-muted)", fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {judges.map((j) => (
            <Bar key={j} dataKey={shortJudge(j)} fill={judgeChartColor(j)} radius={[4, 4, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
