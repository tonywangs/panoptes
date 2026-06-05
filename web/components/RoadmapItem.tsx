import { ReactNode } from "react";

type Status = "shipped" | "near" | "mid" | "long";

const STATUS_STYLE: Record<Status, { color: string; label: string }> = {
  shipped: { color: "#10b981", label: "shipped" },
  near: { color: "#fbbf24", label: "next" },
  mid: { color: "#a78bfa", label: "soon" },
  long: { color: "#71717a", label: "later" },
};

export function RoadmapItem({
  status,
  title,
  body,
  effort,
  children,
}: {
  status: Status;
  title: string;
  body: ReactNode;
  effort?: string;
  children?: ReactNode;
}) {
  const s = STATUS_STYLE[status];
  return (
    <div
      className="rounded-2xl px-5 py-5 flex gap-4"
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderLeft: `4px solid ${s.color}`,
      }}
    >
      <div className="flex flex-col items-center pt-1 shrink-0">
        <span
          className="w-3 h-3 rounded-full"
          style={{
            background: status === "shipped" ? s.color : "transparent",
            border: `2px solid ${s.color}`,
            boxShadow: status === "shipped" ? `0 0 0 4px ${s.color}22` : undefined,
          }}
        />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div className="font-medium">{title}</div>
          <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-wider">
            <span style={{ color: s.color }}>{s.label}</span>
            {effort && (
              <span
                className="px-2 py-0.5 rounded"
                style={{ background: "var(--surface-2)", color: "var(--foreground-muted)" }}
              >
                {effort}
              </span>
            )}
          </div>
        </div>
        <p className="mt-2 text-sm muted leading-relaxed">{body}</p>
        {children && <div className="mt-3 text-xs muted">{children}</div>}
      </div>
    </div>
  );
}
