import { Card, CardTitle } from "@/components/Card";
import { JudgeBadge } from "@/components/JudgeBadge";
import { ParetoChart } from "@/components/ParetoChart";
import { ReliabilityDiagram } from "@/components/ReliabilityDiagram";
import {
  loadCalibration,
  loadHeadlineRuns,
  loadPareto,
} from "@/lib/data";
import { formatPercent, gapColor } from "@/lib/format";

export default function CalibrationPage() {
  const calib = loadCalibration();
  const runs = loadHeadlineRuns();
  const featured = [...runs].sort((a, b) => b.n_calls - a.n_calls)[0];
  const pareto = featured ? loadPareto(featured.run_id) : [];

  if (!calib) {
    return (
      <Card>
        <div className="muted text-sm">
          No calibration data exported yet. Run scripts/calibration_probe.py and rerun the export.
        </div>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-12">
      <header>
        <CardTitle>calibration probe</CardTitle>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight">
          Does the <span className="text-emerald-500">90% interval</span> actually contain the
          truth <span className="text-emerald-500">90%</span> of the time?
        </h1>
        <p className="mt-4 max-w-3xl text-base muted leading-relaxed">
          Conformal prediction guarantees, on paper, that the prediction interval contains the true
          value at least <code className="text-foreground font-mono">1 − α</code> of the time. This
          page measures whether that guarantee actually holds on a real held-out test set, with
          real ground-truth labels.
        </p>
      </header>

      {/* HEADLINE */}
      <section>
        <Card className="border-emerald-500/30">
          <div className="text-xs uppercase tracking-wider muted">
            headline · split conformal · {calib.headline.judge.includes("claude") ? "Claude Sonnet" : "GPT-4o"}
          </div>
          <div className="mt-2 text-3xl md:text-4xl font-semibold tracking-tight">
            <span className="text-emerald-500">{formatPercent(calib.headline.empirical, 0)}</span>{" "}
            empirical coverage at the nominal{" "}
            {formatPercent(calib.headline.nominal, 0)} target ·{" "}
            <span className="text-emerald-500">{calib.headline.gap_pp.toFixed(1)}pp gap</span>
          </div>
          <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">{calib.headline.summary}</p>
        </Card>
      </section>

      {/* IN PLAIN ENGLISH */}
      <section className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardTitle>conformal prediction in 90 seconds</CardTitle>
          <ol className="mt-3 space-y-3 text-sm leading-relaxed">
            <li className="flex gap-3">
              <span className="font-mono text-emerald-500 shrink-0">1</span>
              <span>
                Take a held-out calibration set with known labels. For each (item, judge), compute
                a <em>conformity score</em> — here, the absolute difference between the judge's
                score and the ground-truth label.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="font-mono text-emerald-500 shrink-0">2</span>
              <span>
                Sort those conformity scores. Take the{" "}
                <code className="font-mono text-foreground">⌈(n + 1)(1 − α)⌉ / n</code>
                -th empirical quantile. Call it <code className="font-mono text-foreground">q</code>.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="font-mono text-emerald-500 shrink-0">3</span>
              <span>
                On a new (test) item, the prediction interval is{" "}
                <code className="font-mono text-foreground">[ŷ − q, ŷ + q]</code>. That's it.
              </span>
            </li>
          </ol>
          <p className="mt-4 text-sm muted leading-relaxed">
            <span className="text-foreground">The guarantee</span>: under exchangeability of
            calibration and test data, the true label falls inside that interval with probability
            ≥ 1 − α. <em>No</em> Gaussian assumption, no parametric model. Finite-sample valid.
          </p>
        </Card>
        <Card>
          <CardTitle>why this benchmark is the right test</CardTitle>
          <ul className="mt-3 space-y-3 text-sm leading-relaxed">
            <li className="flex gap-3">
              <span className="text-emerald-500 shrink-0">•</span>
              <span>
                <span className="font-medium">Obfuscated HumanEval.</span> Every problem's
                entry-point function is renamed to an opaque hash, so judges can't pattern-match
                memorized solutions. (Memorized solutions would inflate scores artificially.)
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-emerald-500 shrink-0">•</span>
              <span>
                <span className="font-medium">Real ground truth.</span> Each candidate solution is
                executed in a sandboxed Python subprocess against the rewritten test block. Pass /
                fail is mechanical — not another LLM's opinion.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-emerald-500 shrink-0">•</span>
              <span>
                <span className="font-medium">50 / 50 split.</span> Half the items fit the
                conformal quantile; the other half measures whether the quantile actually holds.
                The deterministic seed makes the split reproducible.
              </span>
            </li>
            <li className="flex gap-3">
              <span className="text-emerald-500 shrink-0">•</span>
              <span>
                <span className="font-medium">Honest noise.</span> With n_test ≈ 25, the standard
                error on an empirical coverage estimate is ≈ ±6pp. The 2pp gap is{" "}
                <em>consistent with valid coverage</em>, not a claim of perfect calibration.
              </span>
            </li>
          </ul>
        </Card>
      </section>

      {/* RELIABILITY DIAGRAM */}
      <section>
        <CardTitle>reliability diagram</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Empirical vs nominal, every α</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Each dot is one (judge, α) measurement. The dashed diagonal is "perfect" calibration —
          empirical equals nominal. The shaded green region above the diagonal is the safe-side
          direction (over-covers, conservative). Conformal's theorem says points should fall in
          the green region or on the line; the failure mode is points falling below the diagonal.
        </p>
        <Card className="mt-4">
          <ReliabilityDiagram rows={calib.conformal_coverage} />
        </Card>
        <p className="mt-3 text-xs muted leading-relaxed max-w-3xl">
          Read this as: at the bottom-left (α = 0.4), the target is 60% coverage. At the top-right
          (α = 0.05), the target is 95% coverage. The dots cluster on or above the line — exactly
          where the theorem says they should be.
        </p>
      </section>

      {/* COVERAGE TABLE */}
      <section>
        <CardTitle>full coverage table</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Per-judge × per-α</h2>
        <p className="mt-2 max-w-3xl text-sm muted leading-relaxed">
          Green = empirical is within 5pp of nominal. Amber = within 10pp. Red = more than 10pp
          off. Over-covering counts as "fine" — the theorem is a lower bound, not equality. The
          one row that matters most for the v1.0 spec target is Claude at α = 0.10.
        </p>
        <div className="mt-4 surface rounded-2xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr
                className="text-left text-xs uppercase tracking-wider muted"
                style={{ background: "var(--surface-2)" }}
              >
                <th className="px-4 py-3">judge</th>
                <th className="px-4 py-3 text-right">α</th>
                <th className="px-4 py-3 text-right">nominal (1−α)</th>
                <th className="px-4 py-3 text-right">empirical</th>
                <th className="px-4 py-3 text-right">|emp − nom|</th>
                <th className="px-4 py-3 text-right">q</th>
                <th className="px-4 py-3 text-right">n_cal</th>
                <th className="px-4 py-3 text-right">n_test</th>
              </tr>
            </thead>
            <tbody>
              {calib.conformal_coverage.map((row, i) => {
                const gap = gapColor(row.gap_pp);
                const gapClass =
                  gap === "good"
                    ? "text-emerald-500"
                    : gap === "ok"
                      ? "text-amber-400"
                      : "text-rose-400";
                return (
                  <tr key={i} className="border-t" style={{ borderColor: "var(--border)" }}>
                    <td className="px-4 py-3">
                      <JudgeBadge judgeId={row.judge} />
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums">{row.alpha.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums">{row.nominal.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums">{row.empirical.toFixed(2)}</td>
                    <td className={`px-4 py-3 text-right font-mono tabular-nums ${gapClass}`}>
                      {row.gap_pp.toFixed(1)}pp
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums muted">
                      {row.q.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums muted">{row.n_cal}</td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums muted">{row.n_test}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* PARETO */}
      <section>
        <CardTitle>coverage–width Pareto (inter-judge stand-in)</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Sweeping α</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          A second view: instead of fixing α at 0.10, what happens as we sweep α from 0.5 down to
          0.01? The dashed gray line is the nominal coverage target (1 − α); the green line is
          the empirical coverage on the same data. These curves come from the inter-judge spread
          inside one production run — not from the held-out calibration probe — but they tell the
          same story: empirical tracks or exceeds nominal across the range.
        </p>
        {featured && (
          <div className="text-xs muted mt-2">
            from run <span className="font-mono">{featured.run_id}</span> · strategy{" "}
            <span className="text-foreground">{featured.strategy}</span>
          </div>
        )}
        <Card className="mt-4">
          <ParetoChart data={pareto} />
        </Card>
      </section>

      {/* HOW THE DATA WAS GENERATED */}
      <section className="grid lg:grid-cols-[1fr_1fr] gap-6">
        <Card>
          <CardTitle>candidate generation</CardTitle>
          <h2 className="mt-2 text-lg font-medium tracking-tight">
            We make the judges judge real noisy code.
          </h2>
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-xs muted">model</div>
              <div className="font-mono">{calib.generator.model}</div>
            </div>
            <div>
              <div className="text-xs muted">temperature</div>
              <div className="font-mono">{calib.generator.temperature}</div>
            </div>
            <div>
              <div className="text-xs muted">pass rate</div>
              <div className="font-mono text-emerald-500">
                {formatPercent(calib.candidates.pass_rate, 0)}
              </div>
            </div>
            <div>
              <div className="text-xs muted">passes / fails</div>
              <div className="font-mono">
                {calib.candidates.n_pass} / {calib.candidates.n_fail}
              </div>
            </div>
          </div>
          <p className="mt-4 text-xs muted leading-relaxed">
            Mid-temperature so candidates aren't all correct — we need a meaningful mix of
            pass/fail to actually measure calibration. (94% pass rate is on the high side; with a
            weaker generator the high-α coverage rows would land closer to nominal instead of
            over-covering.)
          </p>
        </Card>
        <Card>
          <CardTitle>judges</CardTitle>
          <div className="mt-3 space-y-3 text-sm">
            {calib.judges.map((j) => (
              <div key={j.judge_id}>
                <JudgeBadge judgeId={j.judge_id} />
                <div className="mt-1 text-xs muted">
                  {j.valid_scores}/{j.total_calls} valid scores
                </div>
                {j.note && <div className="mt-1 text-xs muted leading-relaxed">{j.note}</div>}
              </div>
            ))}
          </div>
        </Card>
      </section>

      {/* METHODOLOGY */}
      <section>
        <CardTitle>methodology fine print</CardTitle>
        <Card className="mt-3">
          <ul className="text-sm leading-relaxed list-disc pl-6 space-y-2 muted">
            {calib.methodology_notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </Card>
      </section>
    </div>
  );
}
