import { CheckCircle2, Layers, RefreshCw, XCircle } from "lucide-react";

/**
 * 2×2 grid of the four uncertainty regimes. The whole reason the framework
 * decomposes total variance into aleatoric + epistemic is that these four
 * cells call for *different actions* — accepting, sampling more, calling
 * a stronger judge, or flagging the item as inherently hard.
 */
export function UncertaintyQuadrant() {
  const cells = [
    {
      title: "trust the score",
      ale: "low",
      epi: "low",
      icon: CheckCircle2,
      color: "#10b981",
      detail: "All judges agree, each is self-consistent. Score is reliable. Move on.",
      action: "accept",
    },
    {
      title: "call more judges",
      ale: "low",
      epi: "high",
      icon: Layers,
      color: "#a78bfa",
      detail: "Judges disagree, but each is self-consistent. Add a third judge or escalate to a stronger one.",
      action: "route",
    },
    {
      title: "sample again",
      ale: "high",
      epi: "low",
      icon: RefreshCw,
      color: "#38bdf8",
      detail: "Judges agree on average but each is internally noisy. More temperature samples will tighten the CI.",
      action: "resample",
    },
    {
      title: "flag the item",
      ale: "high",
      epi: "high",
      icon: XCircle,
      color: "#f43f5e",
      detail: "Judges disagree AND each is noisy with itself. The task itself may be ambiguous. Surface, don't average.",
      action: "skip / surface",
    },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 max-w-3xl">
      {cells.map((c) => {
        const Icon = c.icon;
        return (
          <div
            key={c.title}
            className="rounded-xl px-4 py-4 relative"
            style={{
              background: "var(--surface)",
              border: `1px solid color-mix(in srgb, ${c.color} 30%, transparent)`,
            }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className="w-7 h-7 rounded-md flex items-center justify-center"
                  style={{ background: `${c.color}22`, color: c.color }}
                >
                  <Icon size={15} />
                </span>
                <div className="text-sm font-medium">{c.title}</div>
              </div>
              <span
                className="text-[10px] font-mono px-2 py-0.5 rounded uppercase tracking-wider"
                style={{ color: c.color, background: `${c.color}11` }}
              >
                {c.action}
              </span>
            </div>
            <div className="mt-2 flex items-center gap-2 text-[11px] muted">
              <span>aleatoric: <span style={{ color: c.color }}>{c.ale}</span></span>
              <span>·</span>
              <span>epistemic: <span style={{ color: c.color }}>{c.epi}</span></span>
            </div>
            <p className="mt-2 text-xs muted leading-relaxed">{c.detail}</p>
          </div>
        );
      })}
    </div>
  );
}
