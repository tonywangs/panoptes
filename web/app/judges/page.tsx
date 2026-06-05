import { Card, CardTitle } from "@/components/Card";
import { CorrelationHeatmap } from "@/components/CorrelationHeatmap";
import { JudgeBadge } from "@/components/JudgeBadge";
import { JudgeScatter } from "@/components/JudgeScatter";
import { loadHeadlineRuns, loadJudgePairs } from "@/lib/data";
import { shortJudge } from "@/lib/format";

export default function JudgesPage() {
  const runs = loadHeadlineRuns();
  const featured = [...runs].sort((a, b) => b.n_judges - a.n_judges)[0];
  const pairs = featured ? loadJudgePairs(featured.run_id) : [];
  const judgesInFeatured = featured ? featured.judges : [];

  return (
    <div className="flex flex-col gap-12">
      <header>
        <CardTitle>judge agreement</CardTitle>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight">
          Do the judges agree with each other?
        </h1>
        <p className="mt-4 max-w-3xl text-base muted leading-relaxed">
          PANOPTES routinely runs three different LLM judges (Claude, GPT-4o, Gemini) on the same
          task. If they're all rating the same latent quality on the same scale, their scores
          should be tightly correlated. If they're not, the framework's job is to give us a
          principled way to combine their disagreement into a posterior. But it's worth knowing
          how much disagreement there is to begin with.
        </p>
      </header>

      {/* IN PLAIN ENGLISH */}
      <section className="grid md:grid-cols-3 gap-4">
        <Card>
          <div className="font-medium">Spearman ρ</div>
          <p className="mt-2 text-sm muted leading-relaxed">
            Rank-correlation: do the judges agree on{" "}
            <span className="text-foreground">which items are better than which</span>? Insensitive
            to systematic bias (one judge rating 0.7 where another rates 0.5 is fine, as long as
            they preserve the ordering).
          </p>
        </Card>
        <Card>
          <div className="font-medium">Kendall τ</div>
          <p className="mt-2 text-sm muted leading-relaxed">
            Same idea as ρ but counts concordant vs. discordant <em>pairs</em> of items. More
            conservative; less sensitive to outliers. A different lens on the same "do they rank
            things the same way" question.
          </p>
        </Card>
        <Card>
          <div className="font-medium">Permutation p</div>
          <p className="mt-2 text-sm muted leading-relaxed">
            Probability the observed mean disagreement <code className="font-mono text-foreground">|a − b|</code>{" "}
            would arise if we randomly shuffled which score belongs to which judge. Small p →
            disagreement is structural, not noise.
          </p>
        </Card>
      </section>

      {/* CORRELATION HEATMAP */}
      <section>
        <CardTitle>at a glance</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Pairwise Spearman ρ heatmap</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Each cell is the rank-correlation between two judges. Greener = more agreement on
          ordering. Diagonal is by definition 1.0 (judge vs itself). Empty cells = the two judges
          didn't both score any common items in this run.
        </p>
        {featured && (
          <div className="text-xs muted mt-2">
            from run <span className="font-mono">{featured.run_id}</span> · strategy{" "}
            <span className="text-foreground">{featured.strategy}</span>
          </div>
        )}
        <Card className="mt-4">
          <CorrelationHeatmap judges={judgesInFeatured} pairs={pairs} />
        </Card>
      </section>

      {/* PER-PAIR SCATTERS */}
      <section>
        <CardTitle>per pair</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Scatter + bootstrap CIs</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          One scatter per judge pair. Each dot is one item; the dashed line is "perfect agreement."
          The Spearman ρ and Kendall τ next to it come with 90% paired-bootstrap CIs. The
          framework <em>never</em> reports rank correlation as a point estimate.
        </p>
        {pairs.length === 0 ? (
          <Card>
            <div className="muted text-sm mt-4">No judge pairs available. Runs must include ≥ 2 judges.</div>
          </Card>
        ) : (
          <div className="grid lg:grid-cols-2 gap-6 mt-4">
            {pairs.map((p) => (
              <Card key={`${p.judge_a}|${p.judge_b}`}>
                <div className="flex items-center gap-2 mb-1">
                  <JudgeBadge judgeId={p.judge_a} />
                  <span className="muted">vs</span>
                  <JudgeBadge judgeId={p.judge_b} />
                  <span className="ml-auto text-xs muted">n={p.n_items}</span>
                </div>
                <JudgeScatter judgeA={p.judge_a} judgeB={p.judge_b} pairs={p.pairs} />
                <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                  <Stat
                    label="Spearman ρ"
                    point={p.spearman.point}
                    ci={[p.spearman.ci_low, p.spearman.ci_high]}
                  />
                  <Stat
                    label="Kendall τ"
                    point={p.kendall.point}
                    ci={[p.kendall.ci_low, p.kendall.ci_high]}
                  />
                  <div>
                    <div className="text-xs muted">permutation p</div>
                    <div className="font-mono tabular-nums">{fmt(p.permutation.p_value)}</div>
                    <div className="text-xs muted">obs {fmt(p.permutation.observed)}</div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </section>

      {/* PER RUN COMPARISON */}
      <section>
        <CardTitle>by run</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">How agreement changes by strategy</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          The same judges may agree more or less depending on which items they were asked to rate.
          Runs using <span className="text-foreground">bandit</span> routing tend to select harder
          items more often, which can suppress correlation; the all-judges runs see every item.
        </p>
        <div className="mt-4 grid md:grid-cols-2 gap-4">
          {runs
            .filter((r) => r.n_judges >= 2)
            .map((r) => {
              const ps = loadJudgePairs(r.run_id);
              return (
                <Card key={r.run_id}>
                  <div className="font-mono text-xs muted">{r.run_id}</div>
                  <div className="text-lg font-medium">{r.strategy}</div>
                  <div className="mt-3 flex flex-col gap-1 text-sm">
                    {ps.map((p) => (
                      <div
                        key={`${p.judge_a}|${p.judge_b}`}
                        className="flex items-center gap-2 text-xs"
                      >
                        <span className="font-mono muted w-20">{shortJudge(p.judge_a)}</span>
                        <span className="muted">↔</span>
                        <span className="font-mono muted w-20">{shortJudge(p.judge_b)}</span>
                        <span className="ml-auto tabular-nums">ρ = {fmt(p.spearman.point)}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              );
            })}
        </div>
      </section>
    </div>
  );
}

function fmt(v: number | null, digits = 3): string {
  if (v === null || Number.isNaN(v)) return "n/a";
  return v.toFixed(digits);
}

function Stat({
  label,
  point,
  ci,
}: {
  label: string;
  point: number | null;
  ci: [number | null, number | null];
}) {
  return (
    <div>
      <div className="text-xs muted">{label}</div>
      <div className="font-mono tabular-nums">{fmt(point)}</div>
      <div className="text-xs muted tabular-nums">
        [{fmt(ci[0])}, {fmt(ci[1])}]
      </div>
    </div>
  );
}
