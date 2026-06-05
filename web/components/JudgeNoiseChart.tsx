"use client";

import {
  CartesianGrid,
  Cell,
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

type Point = { judge: string; score: number; sample: number; isPoint: boolean };

/**
 * A single chart that shows three judges scoring the same item: point-pass
 * scores as solid dots, sampling-pass draws as hollow dots, mean per judge
 * as a vertical bar. Drives home "the judges literally disagree, and even
 * disagree with themselves."
 */
export function JudgeNoiseChart({
  pointRows,
  samplesByJudge,
}: {
  pointRows: { judge_id: string; score_value: number }[];
  samplesByJudge: Record<string, number[]>;
}) {
  const judges = Array.from(
    new Set([...pointRows.map((r) => r.judge_id), ...Object.keys(samplesByJudge)]),
  );
  // y is judge index
  const points: Point[] = [];
  judges.forEach((j, i) => {
    const point = pointRows.find((r) => r.judge_id === j);
    if (point) points.push({ judge: j, score: point.score_value, sample: i, isPoint: true });
    (samplesByJudge[j] ?? []).forEach((v) =>
      points.push({ judge: j, score: v, sample: i, isPoint: false }),
    );
  });
  const means = judges.map((j) => {
    const all = [
      ...(pointRows.find((r) => r.judge_id === j) ? [pointRows.find((r) => r.judge_id === j)!.score_value] : []),
      ...(samplesByJudge[j] ?? []),
    ];
    const mean = all.length ? all.reduce((s, v) => s + v, 0) / all.length : 0;
    return { judge: j, mean };
  });

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 12, right: 24, bottom: 28, left: 110 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            type="number"
            dataKey="score"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            label={{
              value: "score",
              position: "insideBottom",
              offset: -12,
              fill: "var(--foreground-muted)",
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="sample"
            domain={[-0.6, judges.length - 0.4]}
            ticks={judges.map((_, i) => i)}
            tickFormatter={(v: number) => shortJudge(judges[v] ?? "")}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            width={100}
          />
          <ZAxis range={[50, 50]} />
          <Tooltip
            cursor={false}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v: number, name: string) =>
              name === "score" ? [v.toFixed(3), "score"] : [v, name]
            }
            labelFormatter={() => ""}
          />
          {means.map((m, i) => (
            <ReferenceLine
              key={`mean-${m.judge}`}
              segment={[
                { x: m.mean, y: i - 0.3 },
                { x: m.mean, y: i + 0.3 },
              ]}
              stroke={judgeChartColor(m.judge)}
              strokeWidth={2.5}
            />
          ))}
          <Scatter data={points}>
            {points.map((p, k) => (
              <Cell
                key={k}
                fill={p.isPoint ? judgeChartColor(p.judge) : "transparent"}
                stroke={judgeChartColor(p.judge)}
                strokeWidth={p.isPoint ? 0 : 1.5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-4 justify-center text-xs muted mt-1">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-zinc-400" /> point pass (temp 0)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full border-2 border-zinc-400" /> sampling draws (temp 1)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-[2.5px] bg-zinc-400" /> mean
        </span>
      </div>
    </div>
  );
}
