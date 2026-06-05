import { Card, CardTitle } from "@/components/Card";

type Method = {
  name: string;
  status: "shipped" | "planned";
  module: string;
  summary: string;
  refs: string[];
};

const METHODS: Method[] = [
  {
    name: "Split conformal prediction",
    status: "shipped",
    module: "panoptes.uq.conformal_split",
    summary:
      "Finite-sample marginal coverage ≥ 1 − α using the ceil((n+1)(1−α))/n quantile correction. Bounded clip to [0, 1] on the rubric scale.",
    refs: [
      "Papadopoulos, Proedrou, Vovk, Gammerman (2002). Inductive Confidence Machines for Regression. ECML.",
      "Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World. Springer.",
      "Angelopoulos, Bates (2023). A Gentle Introduction to Conformal Prediction. arXiv:2107.07511.",
    ],
  },
  {
    name: "Conformalized Quantile Regression (CQR)",
    status: "shipped",
    module: "panoptes.uq.conformal_adaptive",
    summary:
      "Input-adaptive intervals via sklearn GradientBoostingRegressor(loss='quantile') on judge-output features. Width shrinks where the quantile regressors are confident.",
    refs: ["Romano, Patterson, Candès (2019). Conformalized Quantile Regression. NeurIPS."],
  },
  {
    name: "Mondrian / group-conditional conformal",
    status: "shipped",
    module: "panoptes.uq.conformal_mondrian",
    summary:
      "Per-task-family quantiles. Conditional coverage P(Y ∈ C(X) | g(X) = g) ≥ 1 − α within each group. Falls back to pooled marginal when n_group < 50.",
    refs: ["Vovk, Lindsay, Nouretdinov, Gammerman (2003). Mondrian Confidence Machine."],
  },
  {
    name: "Semantic entropy",
    status: "shipped",
    module: "panoptes.uq.semantic_entropy",
    summary:
      "Bidirectional NLI clustering of temperature samples, Shannon entropy over cluster sizes bounded in [0, log N]. Two backends: local DeBERTa-v3-large-mnli (HF) and LLM-as-NLI.",
    refs: [
      "Farquhar, Kossen, Kuhn, Gal (2024). Detecting hallucinations in large language models using semantic entropy. Nature.",
    ],
  },
  {
    name: "Self-consistency variance",
    status: "shipped",
    module: "panoptes.uq.self_consistency",
    summary:
      "MC variance + IQR + Bayesian bootstrap CI (Dirichlet(1,...,1) weights) over n temperature samples per (judge, item) pair.",
    refs: [
      "Wang, Wei, Schuurmans, Le, Chi, et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR.",
      "Rubin (1981). The Bayesian Bootstrap. Annals of Statistics 9(1).",
    ],
  },
  {
    name: "Hierarchical-Gaussian jury aggregation",
    status: "shipped",
    module: "panoptes.uq.disagreement",
    summary:
      "Closed-form EM for score_ij = θ_i + bias_j + ε_ij with ε_ij ~ N(0, σ_j²). Recovers per-item posterior over latent quality θ plus per-judge bias and precision. Identifiability via Σ_j bias_j = 0.",
    refs: [
      "Dawid, Skene (1979). Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. JRSS-C.",
      "Hovy, Berg-Kirkpatrick, Vaswani, Hovy (2013). Learning Whom to Trust with MACE. NAACL.",
    ],
  },
  {
    name: "Aleatoric / epistemic decomposition",
    status: "shipped",
    module: "panoptes.uq.decomposition",
    summary:
      "Var_total = E_j[Var(score | judge=j)] + Var_j[E(score | judge=j)]. Nested resampling — outer over judges (epistemic), inner over temperature samples (aleatoric).",
    refs: [
      "Kendall, Gal (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NeurIPS.",
      "Depeweg, Hernández-Lobato, Doshi-Velez, Udluft (2018). Decomposition of Uncertainty in Bayesian Deep Learning. ICML.",
    ],
  },
  {
    name: "Thompson-sampling jury routing",
    status: "shipped",
    module: "panoptes.routing.bandit",
    summary:
      "Beta(α, β) per (judge, task_family) arm. Reward = epistemic-variance reduction / dollars. Online updates after each item. State serializable for warm-start across runs.",
    refs: [
      "Russo, Van Roy, Kazerouni, Osband, Wen (2018). A Tutorial on Thompson Sampling. arXiv:1707.02038.",
      "Chapelle, Li (2011). An Empirical Evaluation of Thompson Sampling. NeurIPS.",
    ],
  },
  {
    name: "Coverage / calibration diagnostics",
    status: "shipped",
    module: "panoptes.stats",
    summary:
      "Marginal coverage with Clopper-Pearson CIs; conditional coverage per task family; reliability diagram with bootstrap bands; ECE / MCE / Brier; paired-bootstrap Spearman/Kendall + permutation test for judge disagreement.",
    refs: [
      "Naeini, Cooper, Hauskrecht (2015). ECE / MCE.",
      "Gneiting, Raftery (2007). Sharpness vs calibration framing.",
      "Bröcker, Smith (2007). Reliability bootstrap bands.",
    ],
  },
];

export default function MethodsPage() {
  return (
    <div className="flex flex-col gap-8">
      <header>
        <CardTitle>methods</CardTitle>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Math &amp; citations</h1>
        <p className="mt-2 max-w-3xl muted">
          One section per implemented method, with the paper(s) it cites and the Python module the
          implementation lives in. The framework is built around the principle that every
          statistical claim is paper-grounded and the residuals are auditable.
        </p>
      </header>

      <div className="flex flex-col gap-4">
        {METHODS.map((m) => (
          <Card key={m.name}>
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-lg font-medium">{m.name}</div>
                <div className="text-xs font-mono muted mt-0.5">{m.module}</div>
              </div>
              <span
                className={`text-xs px-2 py-0.5 rounded-md ${
                  m.status === "shipped"
                    ? "bg-emerald-500/10 text-emerald-500"
                    : "bg-amber-500/10 text-amber-500"
                }`}
              >
                {m.status}
              </span>
            </div>
            <p className="mt-3 text-sm leading-relaxed">{m.summary}</p>
            <div className="mt-3">
              <div className="text-xs uppercase tracking-wider muted">references</div>
              <ul className="mt-1 text-sm muted list-disc pl-5 space-y-1">
                {m.refs.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
