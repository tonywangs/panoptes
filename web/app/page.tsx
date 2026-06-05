import Link from "next/link";
import { ArrowRight, BookOpen, Sparkles, Layers, BarChart3 } from "lucide-react";
import { Card, CardTitle, Metric } from "@/components/Card";
import { JudgeBadge } from "@/components/JudgeBadge";
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
            href="/background"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 text-emerald-500 ring-1 ring-emerald-500/30 hover:bg-emerald-500/20 transition-colors"
          >
            <BookOpen size={15} /> read the background first
          </Link>
          <Link
            href="/calibration"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-[var(--surface-2)] transition-colors"
            style={{ border: "1px solid var(--border)" }}
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
        </div>
      </header>

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
                <div className="mt-2 text-3xl md:text-5xl font-semibold tracking-tight">
                  <span className="text-emerald-500">{formatPercent(calib.headline.empirical, 0)}</span>{" "}
                  empirical coverage at {formatPercent(calib.headline.nominal, 0)} nominal
                </div>
                <div className="mt-2 text-base muted">
                  · <span className="text-emerald-500 font-medium">{calib.headline.gap_pp.toFixed(1)}pp gap</span>{" "}
                  · finite-sample guaranteed under exchangeability
                </div>
                <p className="mt-4 text-sm muted leading-relaxed max-w-3xl">
                  The framework's central claim is that conformal-prediction intervals on judge
                  scores carry real frequentist coverage. To verify, we obfuscated HumanEval so
                  judges can't pattern-match memorized solutions, generated candidates with
                  gpt-4o-mini, graded them with a sandboxed Python executor for ground truth, and
                  measured how often a 90% interval contains the truth on a held-out set.
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
            Dot size encodes the number of items in the run.
          </p>
          <Link
            href="/background"
            className="mt-4 inline-flex items-center gap-1 text-sm text-emerald-500 hover:underline"
          >
            why this tradeoff matters <ArrowRight size={14} />
          </Link>
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
          Each card is one <code className="text-foreground font-mono">panoptes eval</code>{" "}
          invocation. Click in to see the cost breakdown, score distribution, and item-by-item
          dashboard.
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

      {/* WHERE NEXT */}
      <section>
        <CardTitle>what to look at next</CardTitle>
        <div className="mt-3 grid md:grid-cols-3 gap-3">
          <NextLink
            href="/background"
            label="The problem this solves"
            body="Why LLM-as-judge eval is broken and how PANOPTES addresses it. The 'set the table' page."
          />
          <NextLink
            href="/calibration"
            label="Does it actually work?"
            body="The reliability diagram and full coverage table on a held-out calibration probe."
          />
          <NextLink
            href="/judges"
            label="How much do judges disagree?"
            body="Pairwise correlation heatmap with paired-bootstrap CIs and permutation tests."
          />
        </div>
      </section>
    </div>
  );
}

function NextLink({ href, label, body }: { href: string; label: string; body: string }) {
  return (
    <Link href={href} className="block group">
      <Card className="h-full transition-colors group-hover:border-emerald-500/40">
        <div className="flex items-baseline justify-between gap-2">
          <div className="font-medium">{label}</div>
          <ArrowRight size={14} className="muted group-hover:text-emerald-500 transition-colors" />
        </div>
        <p className="mt-2 text-sm muted leading-relaxed">{body}</p>
      </Card>
    </Link>
  );
}
