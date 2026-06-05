import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";
import { Card, CardTitle, Metric } from "@/components/Card";
import { JudgeBadge } from "@/components/JudgeBadge";
import {
  loadCalibration,
  loadHeadlineRuns,
  loadSummary,
} from "@/lib/data";
import { formatPercent, formatUSD, shortJudge } from "@/lib/format";

export default function Overview() {
  const summary = loadSummary();
  const runs = loadHeadlineRuns();
  const calib = loadCalibration();

  return (
    <div className="flex flex-col gap-10">
      <header>
        <div className="text-sm uppercase tracking-[0.18em] muted">PANOPTES</div>
        <h1 className="mt-2 text-4xl md:text-5xl font-semibold tracking-tight">
          Uncertainty-aware <span className="text-emerald-500">LLM evaluation</span>
        </h1>
        <p className="mt-4 max-w-2xl text-lg muted leading-relaxed">
          Every <span className="text-foreground font-medium">(task, response, judge)</span> tuple
          produces a calibrated probability distribution over the true quality, decomposed into
          aleatoric (irreducible) and epistemic (reducible) components, with finite-sample
          statistical guarantees from conformal prediction.
        </p>
      </header>

      {calib && (
        <Card className="border-emerald-500/30">
          <div className="flex items-start gap-4">
            <div className="hidden sm:flex shrink-0 w-10 h-10 rounded-xl items-center justify-center bg-emerald-500/10 ring-1 ring-emerald-500/30">
              <Sparkles size={18} className="text-emerald-500" />
            </div>
            <div className="flex-1">
              <CardTitle>Headline · split conformal on the held-out calibration probe</CardTitle>
              <div className="mt-2 text-2xl md:text-3xl font-semibold tracking-tight">
                {formatPercent(calib.headline.empirical, 0)} empirical coverage at the nominal{" "}
                {formatPercent(calib.headline.nominal, 0)} target
                <span className="ml-3 inline-flex items-center gap-1 text-base font-medium text-emerald-500 align-middle">
                  · {calib.headline.gap_pp.toFixed(1)}pp gap
                </span>
              </div>
              <p className="mt-3 text-sm muted leading-relaxed max-w-3xl">
                {calib.headline.summary} Verified on obfuscated HumanEval candidates graded by a
                sandbox executor (real ground truth, not inter-judge proxy).
              </p>
              <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
                <JudgeBadge judgeId={calib.headline.judge} />
                <span className="muted">·</span>
                <span className="muted">α = {calib.headline.alpha}</span>
                <span className="muted">·</span>
                <span className="muted">n_test = 25</span>
                <Link
                  href="/calibration"
                  className="ml-auto inline-flex items-center gap-1 text-sm text-emerald-500 hover:underline"
                >
                  Full coverage table <ArrowRight size={14} />
                </Link>
              </div>
            </div>
          </div>
        </Card>
      )}

      <section>
        <CardTitle>at a glance</CardTitle>
        <div className="mt-3 grid grid-cols-2 lg:grid-cols-4 gap-4">
          <Metric
            label="evaluation runs"
            value={summary.n_runs}
            hint={`${summary.n_items_total} items judged total`}
          />
          <Metric label="LLM judge calls" value={summary.n_calls_total.toLocaleString()} />
          <Metric label="total spend" value={formatUSD(summary.cost_total_usd)} />
          <Metric
            label="distinct judges"
            value={summary.judges_seen.length}
            hint={summary.judges_seen.map(shortJudge).join(", ")}
          />
        </div>
      </section>

      <section>
        <div className="flex items-baseline justify-between">
          <CardTitle>runs</CardTitle>
          <Link href="/runs" className="text-xs muted hover:text-foreground">
            see all →
          </Link>
        </div>
        <div className="mt-3 grid md:grid-cols-2 gap-4">
          {runs.map((r) => (
            <Link key={r.run_id} href={`/runs/${r.run_id}`} className="block group">
              <Card className="transition-colors group-hover:border-emerald-500/40">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="font-mono text-xs muted">{r.run_id}</div>
                    <div className="mt-1 text-lg font-medium">
                      strategy:{" "}
                      <span
                        className={
                          r.strategy === "bandit" ? "text-emerald-500" : "text-foreground"
                        }
                      >
                        {r.strategy}
                      </span>
                    </div>
                  </div>
                  <ArrowRight
                    size={18}
                    className="muted shrink-0 group-hover:text-emerald-500 transition-colors"
                  />
                </div>
                <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
                  <div>
                    <div className="text-xs muted">items</div>
                    <div className="font-mono tabular-nums">{r.n_items}</div>
                  </div>
                  <div>
                    <div className="text-xs muted">calls</div>
                    <div className="font-mono tabular-nums">{r.n_calls}</div>
                  </div>
                  <div>
                    <div className="text-xs muted">cost</div>
                    <div className="font-mono tabular-nums">{formatUSD(r.cost_usd)}</div>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {r.judges.map((j) => (
                    <JudgeBadge key={j} judgeId={j} />
                  ))}
                </div>
              </Card>
            </Link>
          ))}
        </div>
      </section>

      <section>
        <CardTitle>what's measured</CardTitle>
        <div className="mt-3 grid md:grid-cols-3 gap-4">
          <Card>
            <div className="font-medium">Conformal prediction</div>
            <p className="mt-2 text-sm muted leading-relaxed">
              Split, adaptive (CQR), and Mondrian variants. Distribution-free prediction intervals
              with finite-sample marginal coverage at 1 − α.
            </p>
          </Card>
          <Card>
            <div className="font-medium">Semantic entropy</div>
            <p className="mt-2 text-sm muted leading-relaxed">
              Bidirectional NLI clustering of temperature-sampled responses (Farquhar et al.{" "}
              <span className="italic">Nature</span> 2024). Two backends: local DeBERTa-v3-mnli or
              LLM-as-NLI.
            </p>
          </Card>
          <Card>
            <div className="font-medium">Smart routing</div>
            <p className="mt-2 text-sm muted leading-relaxed">
              Thompson-sampling bandit over (judge, task_family) arms. Reward = epistemic-variance
              reduction per dollar. Plus an escalation policy and the all-judges baseline.
            </p>
          </Card>
        </div>
      </section>
    </div>
  );
}
