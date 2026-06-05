"use client";

import {
  CartesianGrid,
  Label,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { TOOLTIP_CONTENT_STYLE, TOOLTIP_ITEM_STYLE, TOOLTIP_LABEL_STYLE } from "@/lib/chart";

type Row = {
  run_id: string;
  strategy: string;
  cost_per_item: number;
  n_judges: number;
  n_items: number;
};

const COLOR: Record<string, string> = {
  all: "#a78bfa",
  bandit: "#10b981",
  escalation: "#f59e0b",
  single: "#38bdf8",
};

/**
 * One dot per run. X = judges available, Y = cost per item. The bandit aims
 * to live in the lower-right quadrant: more judges available but smarter
 * routing keeps the per-item cost down vs. always-call-all.
 */
export function StrategyTradeoff({ rows }: { rows: Row[] }) {
  const strategies = Array.from(new Set(rows.map((r) => r.strategy)));
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            type="number"
            dataKey="n_judges"
            domain={[0, "dataMax + 1"]}
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
          >
            <Label
              value="judges in the pool"
              position="insideBottom"
              offset={-10}
              fill="var(--foreground-muted)"
              fontSize={11}
            />
          </XAxis>
          <YAxis
            type="number"
            dataKey="cost_per_item"
            tick={{ fill: "var(--foreground-muted)", fontSize: 11 }}
            tickFormatter={(v) => `$${v.toFixed(3)}`}
          >
            <Label
              value="cost per item"
              angle={-90}
              position="insideLeft"
              offset={2}
              fill="var(--foreground-muted)"
              fontSize={11}
            />
          </YAxis>
          <ZAxis dataKey="n_items" range={[80, 280]} />
          <Tooltip
            contentStyle={TOOLTIP_CONTENT_STYLE}
            itemStyle={TOOLTIP_ITEM_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(v: number | string, name: string) => {
              if (name === "cost_per_item" && typeof v === "number") return [`$${v.toFixed(4)}`, "cost/item"];
              if (name === "n_judges") return [v, "judges"];
              if (name === "n_items") return [v, "items"];
              return [v, name];
            }}
            labelFormatter={() => ""}
          />
          {strategies.map((s) => (
            <Scatter
              key={s}
              name={s}
              data={rows.filter((r) => r.strategy === s)}
              fill={COLOR[s] ?? "#a1a1aa"}
              fillOpacity={0.85}
              stroke={COLOR[s] ?? "#a1a1aa"}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-3 justify-center mt-2 text-xs">
        {strategies.map((s) => (
          <span key={s} className="inline-flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ background: COLOR[s] ?? "#a1a1aa" }}
            />
            <span className="muted">{s}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
