import { Card, CardTitle } from "@/components/Card";
import { JudgeBadge } from "@/components/JudgeBadge";
import { JudgeScatter } from "@/components/JudgeScatter";
import { loadHeadlineRuns, loadJudgePairs } from "@/lib/data";
import { shortJudge } from "@/lib/format";

export default function JudgesPage() {
  const runs = loadHeadlineRuns();
  // Pick the run with the most judges for the headline scatter.
  const featured = [...runs].sort((a, b) => b.n_judges - a.n_judges)[0];
  const pairs = featured ? loadJudgePairs(featured.run_id) : [];

  return (
    <div className="flex flex-col gap-8">
      <header>
        <CardTitle>judge agreement</CardTitle>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">
          Pairwise judge correlation
        </h1>
        <p className="mt-2 max-w-2xl muted">
          Paired bootstrap CIs on Spearman ρ and Kendall τ across items both judges scored, plus
          a permutation test for "judges disagree more than chance." Significant rank correlation
          says the judges are tracking the same latent quality; a small permutation p-value says
          their disagreements are real, not random.
        </p>
        {featured && (
          <div className="mt-3 muted text-sm">
            from run <span className="font-mono">{featured.run_id}</span> · strategy{" "}
            <span className="text-foreground font-medium">{featured.strategy}</span>
          </div>
        )}
      </header>

      {pairs.length === 0 ? (
        <Card>
          <div className="muted text-sm">No judge pairs available — runs must include ≥ 2 judges.</div>
        </Card>
      ) : (
        <div className="grid lg:grid-cols-2 gap-6">
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

      <section>
        <CardTitle>by run</CardTitle>
        <div className="mt-3 grid md:grid-cols-2 gap-4">
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
                        <span className="font-mono muted w-16">{shortJudge(p.judge_a)}</span>
                        <span className="muted">↔</span>
                        <span className="font-mono muted w-16">{shortJudge(p.judge_b)}</span>
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
