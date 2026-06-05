import Link from "next/link";
import { ArrowRight, Sparkles, AlertTriangle, Layers, ScatterChart as ScatterIcon, BarChart3 } from "lucide-react";
import { Card, CardTitle, Metric } from "@/components/Card";
import { JudgeBadge } from "@/components/JudgeBadge";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import { StrategyTradeoff } from "@/components/StrategyTradeoff";
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

  const tradeoffRows = runs.map((r) => ({
    run_id: r.run_id,
    strategy: r.strategy,
    cost_per_item: r.n_items > 0 ? r.cost_usd / r.n_items : 0,
    n_judges: r.n_judges,
    n_items: r.n_items,
  }));

  return (
    <div className="flex flex-col gap-14">
      {/* HERO */}
      <header>
        <div className="text-sm uppercase tracking-[0.18em] muted">PANOPTES</div>
        <h1 className="mt-2 text-4xl md:text-5xl font-semibold tracking-tight">
          Uncertainty-aware <span className="text-emerald-500">LLM evaluation</span>
        </h1>
        <p className="mt-4 max-w-3xl text-lg muted leading-relaxed">
          Every <span className="text-foreground font-medium">(task, response, judge)</span> tuple
          produces a calibrated probability distribution over the true quality, decomposed into
          aleatoric (irreducible) and epistemic (reducible) components, with finite-sample
          statistical guarantees from conformal prediction.
        </p>
        <div className="mt-6 flex flex-wrap gap-3 text-sm">
          <Link
            href="/calibration"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 text-emerald-500 ring-1 ring-emerald-500/30 hover:bg-emerald-500/20 transition-colors"
          >
            <BarChart3 size={15} /> see the calibration result
          </Link>
          <Link
            href="/runs"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-[var(--surface-2)] transition-colors"
            style={{ border: "1px solid var(--border)" }}
          >
            <Layers size={15} /> browse runs
          </Link>
          <Link
            href="/methods"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-[var(--surface-2)] transition-colors"
            style={{ border: "1px solid var(--border)" }}
          >
            methods + citations
          </Link>
        </div>
      </header>

      {/* PROBLEM */}
      <section className="grid lg:grid-cols-[1.2fr_1fr] gap-6 items-start">
        <Card className="border-amber-500/20">
          <div className="flex items-start gap-4">
            <div className="hidden sm:flex shrink-0 w-10 h-10 rounded-xl items-center justify-center bg-amber-500/10 ring-1 ring-amber-500/30">
              <AlertTriangle size={18} className="text-amber-500" />
            </div>
            <div>
              <CardTitle>the problem</CardTitle>
              <h2 className="mt-2 text-xl font-medium leading-snug">
                The standard "ask one LLM to grade another" loop hides a lot of noise.
              </h2>
              <p className="mt-3 text-sm muted leading-relaxed">
                LLM judges disagree with each other; they disagree with themselves under
                resampling; and they're miscalibrated — a "0.85" from one judge is not the same
                quality bar as a "0.85" from another. Existing eval frameworks (Promptfoo, OpenAI
                Evals, Inspect) report a point score and stop, treating that single number as if it
                were ground truth.
              </p>
              <p className="mt-2 text-sm muted leading-relaxed">
                PANOPTES treats judge noise as a <span className="text-foreground">statistical
                inference problem</span>: produce a <span className="text-foreground">posterior
                distribution</span> over the true quality, not a point. Then act on that
                distribution — call more judges when the answer is genuinely uncertain, accept the
                score when it's not.
              </p>
            </div>
          </div>
        </Card>
        <Card>
          <CardTitle>three things only PANOPTES does</CardTitle>
          <ul className="mt-3 space-y-3 text-sm">
            <li className="flex gap-3">
              <span className="font-mono text-emerald-500 shrink-0">01</span>
              <span>
                <span className="font-medium">Finite-sample coverage.</span> Conformal prediction
                gives prediction intervals with <em>provable</em> 1 − α coverage — no Gaussian
                assumptions, just exchangeability of calibration data.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="font-mono text-emerald-500 shrink-0">02</span>
              <span>
                <span className="font-medium">Aleatoric vs epistemic.</span> Decompose total
                variance into the part that <em>more samples won't fix</em> (genuine task
                ambiguity) and the part that <em>more judges would fix</em> (model disagreement).
              </span>
            </li>
            <li className="flex gap-3">
              <span className="font-mono text-emerald-500 shrink-0">03</span>
              <span>
                <span className="font-medium">Information-aware routing.</span> A Thompson-sampling
                bandit learns which judges give the most epistemic-variance reduction per dollar
                and stops calling the others.
              </span>
            </li>
          </ul>
        </Card>
      </section>

      {/* PIPELINE */}
      <section>
        <CardTitle>how it works</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">The pipeline, one item at a time</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Five stages. Stage 1–3 produce the raw signal; 4 turns it into a calibrated posterior;
          5 closes the loop by deciding what to do on the next item.
        </p>
        <div className="mt-5">
          <PipelineDiagram />
        </div>
      </section>

      {/* HEADLINE RESULT */}
      {calib && (
        <section>
          <CardTitle>headline result</CardTitle>
          <Card className="mt-3 border-emerald-500/30">
            <div className="flex items-start gap-4">
              <div className="hidden sm:flex shrink-0 w-10 h-10 rounded-xl items-center justify-center bg-emerald-500/10 ring-1 ring-emerald-500/30">
                <Sparkles size={18} className="text-emerald-500" />
              </div>
              <div className="flex-1">
                <div className="text-xs uppercase tracking-wider muted">
                  split conformal · held-out calibration probe · obfuscated HumanEval
                </div>
                <div className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">
                  {formatPercent(calib.headline.empirical, 0)} empirical coverage at{" "}
                  {formatPercent(calib.headline.nominal, 0)} target
                  <span className="ml-3 inline-flex items-center gap-1 text-lg font-medium text-emerald-500 align-middle">
                    · {calib.headline.gap_pp.toFixed(1)}pp gap
                  </span>
                </div>
                <p className="mt-3 text-sm muted leading-relaxed max-w-3xl">
                  The framework's central claim is that conformal-prediction intervals on judge
                  scores carry meaningful frequentist coverage. To verify, we obfuscated HumanEval
                  problems (so judges can't pattern-match on memorized solutions), generated
                  candidates with gpt-4o-mini, graded them with a sandboxed Python executor for
                  ground truth, and measured how often a 90% interval actually contains the truth.
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
                    full coverage table + reliability diagram <ArrowRight size={14} />
                  </Link>
                </div>
              </div>
            </div>
          </Card>
        </section>
      )}

      {/* AT A GLANCE */}
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

      {/* STRATEGY TRADEOFF */}
      <section className="grid lg:grid-cols-[1fr_1.2fr] gap-6 items-start">
        <div>
          <CardTitle>routing tradeoff</CardTitle>
          <h2 className="mt-2 text-2xl font-medium tracking-tight">Bandit vs. all-judges</h2>
          <p className="mt-3 text-sm muted leading-relaxed">
            Each dot is one evaluation run. The y-axis is cost per item; the x-axis is the size
            of the judge pool the run had access to. The naïve "call every judge on every item"
            strategy lives in the upper-right. The Thompson-sampling bandit aims for the
            lower-right: <span className="text-foreground">more judges available, but smarter
            decisions about which to call</span>, so cost per item stays flat or drops.
          </p>
          <p className="mt-2 text-sm muted leading-relaxed">
            Dot size encodes the number of items in the run. Hover to see exact numbers.
          </p>
        </div>
        <Card>
          <StrategyTradeoff rows={tradeoffRows} />
        </Card>
      </section>

      {/* RUNS */}
      <section>
        <div className="flex items-baseline justify-between">
          <CardTitle>runs you can drill into</CardTitle>
          <Link href="/runs" className="text-xs muted hover:text-foreground">
            see all →
          </Link>
        </div>
        <p className="mt-2 max-w-3xl text-sm muted leading-relaxed">
          A run is one invocation of <code className="text-foreground font-mono">panoptes
          eval</code>. Each has a routing strategy (all / bandit / escalation / single), a set of
          judges, and a benchmark. Click in to see the cost breakdown, score distribution, and
          item-by-item dashboard.
        </p>
        <div className="mt-4 grid md:grid-cols-2 gap-4">
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

      {/* WHAT'S MEASURED */}
      <section>
        <CardTitle>what's measured</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Every claim is paper-grounded</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Eight statistical methods, each cited in <Link href="/methods" className="text-emerald-500 hover:underline">METHODS.md</Link>. The whole framework is built on the
          principle that if we report a number, we can point to the paper it comes from and the
          residuals are auditable.
        </p>
        <div className="mt-4 grid md:grid-cols-3 gap-4">
          <Card>
            <div className="font-medium">Conformal prediction</div>
            <p className="mt-2 text-sm muted leading-relaxed">
              Split, adaptive (CQR), and Mondrian variants. Distribution-free prediction intervals
              with finite-sample marginal coverage at 1 − α.
            </p>
            <div className="mt-3 text-xs muted">
              Vovk/Gammerman/Shafer 2005 · Romano/Patterson/Candès 2019
            </div>
          </Card>
          <Card>
            <div className="font-medium">Semantic entropy</div>
            <p className="mt-2 text-sm muted leading-relaxed">
              Bidirectional NLI clustering of temperature-sampled responses. Detects "the judge
              keeps saying the same number for different reasons" hallucination patterns.
            </p>
            <div className="mt-3 text-xs muted">Farquhar et al. <em>Nature</em> 2024</div>
          </Card>
          <Card>
            <div className="font-medium">Smart routing</div>
            <p className="mt-2 text-sm muted leading-relaxed">
              Thompson-sampling bandit over (judge, task_family) arms. Reward = epistemic-variance
              reduction per dollar. Plus an escalation policy and the all-judges baseline.
            </p>
            <div className="mt-3 text-xs muted">Russo &amp; Van Roy 2018 · Chapelle &amp; Li 2011</div>
          </Card>
        </div>
      </section>
    </div>
  );
}
