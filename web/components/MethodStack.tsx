/**
 * Visual hierarchy of the statistical methods PANOPTES stacks on top of
 * each other. Pure presentation — same content as the methods page but
 * styled as a layered diagram instead of a long list.
 */
const LAYERS = [
  {
    title: "Routing",
    body: "Thompson-sampling bandit decides which judges are worth calling per item.",
    method: "Russo & Van Roy 2018",
    color: "#e879f9",
  },
  {
    title: "Decomposition",
    body: "Total variance split into aleatoric (irreducible) and epistemic (reducible).",
    method: "Kendall & Gal 2017",
    color: "#a78bfa",
  },
  {
    title: "Conformal prediction",
    body: "Split / adaptive (CQR) / Mondrian — finite-sample 1−α coverage guarantees.",
    method: "Vovk/Gammerman/Shafer 2005 · Romano/Patterson/Candès 2019",
    color: "#10b981",
  },
  {
    title: "Aggregation",
    body: "Hierarchical-Gaussian EM combines noisy judges into a posterior over latent quality.",
    method: "Dawid & Skene 1979",
    color: "#38bdf8",
  },
  {
    title: "Sampling-UQ",
    body: "Semantic entropy + Bayesian-bootstrap self-consistency at temperature 1.",
    method: "Farquhar et al. Nature 2024 · Rubin 1981",
    color: "#fbbf24",
  },
  {
    title: "Heterogeneous jury",
    body: "Anthropic, OpenAI, Google judges, all behind one provider-agnostic Protocol.",
    method: "PANOPTES infra",
    color: "#f43f5e",
  },
];

export function MethodStack() {
  return (
    <div className="flex flex-col gap-2">
      {LAYERS.map((l, i) => (
        <div
          key={l.title}
          className="rounded-xl px-4 py-3"
          style={{
            background: "var(--surface)",
            borderLeft: `4px solid ${l.color}`,
            border: "1px solid var(--border)",
            marginLeft: `${i * 6}px`,
            marginRight: `${i * 6}px`,
          }}
        >
          <div className="flex items-baseline justify-between gap-3">
            <div>
              <div className="font-medium text-sm" style={{ color: l.color }}>
                {l.title}
              </div>
              <p className="mt-0.5 text-sm muted leading-snug">{l.body}</p>
            </div>
            <div className="text-[11px] muted font-mono shrink-0 hidden md:block">{l.method}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
