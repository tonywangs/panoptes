import { ArrowRight } from "lucide-react";

const STEPS = [
  {
    title: "Task + candidate",
    body: "An LLM-generated answer to a task. The thing we want to evaluate.",
    color: "#38bdf8",
  },
  {
    title: "Heterogeneous jury",
    body: "Multiple judges (Anthropic, OpenAI, Google) score it on the [0, 1] scale via tool-use structured output.",
    color: "#a78bfa",
  },
  {
    title: "Sampling pass",
    body: "Each judge is sampled k times at temperature 1, giving us a distribution rather than a point.",
    color: "#fbbf24",
  },
  {
    title: "Decompose + calibrate",
    body: "Aleatoric vs epistemic split. Conformal-prediction interval with finite-sample coverage at 1 − α.",
    color: "#10b981",
  },
  {
    title: "Smart routing",
    body: "Thompson-sampling bandit learns which judges give the most information per dollar, and stops there.",
    color: "#e879f9",
  },
] as const;

export function PipelineDiagram() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
      {STEPS.map((s, i) => (
        <div key={s.title} className="relative">
          <div
            className="rounded-2xl px-4 py-4 h-full flex flex-col"
            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
          >
            <div className="flex items-center gap-2 mb-2">
              <span
                className="w-6 h-6 rounded-md flex items-center justify-center text-[11px] font-semibold"
                style={{ background: `${s.color}22`, color: s.color }}
              >
                {i + 1}
              </span>
              <div className="font-medium text-sm">{s.title}</div>
            </div>
            <div className="text-xs muted leading-relaxed">{s.body}</div>
          </div>
          {i < STEPS.length - 1 && (
            <ArrowRight
              size={14}
              className="hidden md:block absolute top-1/2 -right-2.5 -translate-y-1/2 muted"
            />
          )}
        </div>
      ))}
    </div>
  );
}
