import Link from "next/link";
import { notFound } from "next/navigation";
import { Card, CardTitle } from "@/components/Card";
import { CodeBlock } from "@/components/CodeBlock";
import { JudgeBadge, ScoreBar } from "@/components/JudgeBadge";
import { SamplingDistribution } from "@/components/SamplingDistribution";
import { UQPanel } from "@/components/UQPanel";
import {
  findItemSource,
  loadAllItemIds,
  loadItemPrompts,
  loadUQ,
} from "@/lib/data";
import {
  formatScore,
  formatUSD,
  judgeColor,
  shortItemId,
  shortJudge,
} from "@/lib/format";

export function generateStaticParams() {
  return loadAllItemIds().map((id) => ({ itemId: encodeURIComponent(id) }));
}

export default async function ItemPage(props: PageProps<"/items/[itemId]">) {
  const { itemId: rawId } = await props.params;
  const itemId = decodeURIComponent(rawId);
  const source = findItemSource(itemId);
  if (!source) notFound();
  const { run, rows } = source;
  const point = rows.filter((r) => r.sample_index === 0);
  const samples = rows.filter((r) => r.sample_index > 0);
  const prompts = loadItemPrompts();
  const prompt = prompts[itemId];
  const uq = loadUQ(run.run_id).filter((u) => u.item_id === itemId);

  // group samples by judge
  const samplesByJudge: Record<string, number[]> = {};
  for (const s of samples) {
    (samplesByJudge[s.judge_id] ??= []).push(s.score_value);
  }
  const sampleGroups = Object.entries(samplesByJudge).map(([judge, values]) => ({
    judge,
    values,
  }));

  // simple aggregate stats for the headline
  const meanScore =
    point.length === 0 ? null : point.reduce((s, r) => s + r.score_value, 0) / point.length;
  const minScore = point.length === 0 ? null : Math.min(...point.map((r) => r.score_value));
  const maxScore = point.length === 0 ? null : Math.max(...point.map((r) => r.score_value));
  const spread = minScore !== null && maxScore !== null ? maxScore - minScore : null;

  return (
    <div className="flex flex-col gap-12">
      <header>
        <Link href={`/runs/${run.run_id}`} className="text-xs muted hover:text-foreground">
          ← {run.run_id}
        </Link>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight font-mono">
          {shortItemId(itemId)}
        </h1>
        <div className="mt-1 muted text-sm">
          {point[0]?.benchmark} · {point[0]?.task_family}
        </div>
        <p className="mt-4 max-w-3xl muted leading-relaxed">
          This is <span className="text-foreground">one (task, candidate response) pair</span>{" "}
          flowing through the full PANOPTES pipeline. Each section below is a stage of the
          analysis: the task itself, the candidate solution being evaluated, every judge's score
          + rationale, the sampling-pass dispersion that captures within-judge noise, and the
          uncertainty-quantification metrics computed on top.
        </p>
        {meanScore !== null && (
          <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Mini label="mean score" value={formatScore(meanScore)} />
            <Mini label="inter-judge spread" value={spread !== null ? spread.toFixed(3) : "—"} />
            <Mini label="judges polled" value={point.length} />
            <Mini label="sampling draws" value={samples.length} />
          </div>
        )}
      </header>

      {prompt && (
        <section>
          <CardTitle>1. the task</CardTitle>
          <p className="mt-2 max-w-3xl text-sm muted leading-relaxed">
            The function signature + docstring presented to both the model under test and to every
            judge.
            {prompt.original_entry_point && (
              <>
                {" "}
                The entry-point name was obfuscated from{" "}
                <span className="font-mono text-foreground">{prompt.original_entry_point}</span>{" "}
                to <span className="font-mono text-foreground">{prompt.entry_point}</span> so the
                judges can't pattern-match a memorized HumanEval solution.
              </>
            )}
          </p>
          <div className="mt-4">
            <CodeBlock code={prompt.prompt} />
          </div>
        </section>
      )}

      <section>
        <CardTitle>2. the candidate response judged</CardTitle>
        <p className="mt-2 max-w-3xl text-sm muted leading-relaxed">
          The full solution every judge is grading. For these runs the candidate is the reference
          solution prepended with the task signature so it parses as a complete program.
        </p>
        <div className="mt-4">
          <CodeBlock code={point[0]?.model_response ?? ""} />
        </div>
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardTitle>3a. point-pass scores (temperature 0)</CardTitle>
          <p className="mt-2 text-xs muted leading-relaxed">
            One call per judge at <code className="font-mono">temperature=0</code>. This is the
            "best single guess" each judge has. Disagreement here is structural. The judges
            literally see this candidate differently.
          </p>
          <div className="mt-4 flex flex-col gap-3">
            {point.map((r) => (
              <div key={r.judge_id} className="flex items-center gap-3">
                <span className={`w-32 font-mono text-sm ${judgeColor(r.judge_id)}`}>
                  {shortJudge(r.judge_id)}
                </span>
                <ScoreBar value={r.score_value} />
                <span className="ml-auto text-xs muted font-mono">{formatUSD(r.cost_usd)}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card>
          <CardTitle>3b. sampling-pass dispersion (temperature 1)</CardTitle>
          <p className="mt-2 text-xs muted leading-relaxed">
            n draws per judge at <code className="font-mono">temperature=1</code>. Dots are
            individual draws; vertical bar is the mean; shaded band is ±1σ. Wide band = the judge
            is uncertain even with itself; narrow band = it's consistent.
          </p>
          <div className="mt-2">
            {sampleGroups.length === 0 ? (
              <div className="text-sm muted mt-4">no sampling pass for this run</div>
            ) : (
              <SamplingDistribution groups={sampleGroups} />
            )}
          </div>
        </Card>
      </section>

      {uq.length > 0 && (
        <section>
          <CardTitle>4. uncertainty quantification</CardTitle>
          <p className="mt-2 max-w-3xl text-sm muted leading-relaxed">
            Three statistical methods computed on top of the raw scores above. Each one tells you
            something different about how much to trust the headline number.
          </p>
          <div className="mt-4">
            <UQPanel results={uq} />
          </div>
          <p className="mt-3 text-xs muted leading-relaxed max-w-3xl">
            <span className="text-foreground">How to read these:</span>{" "}
            <em>self-consistency</em> = does the judge agree with itself when resampled? Narrower
            CI means more consistent. <em>Semantic entropy</em> = do the judge's
            rationales cluster into one meaning, or several? Higher entropy means the judge is
            internally conflicted about <em>why</em>. <em>Decomposition</em> = how much of the
            total variance comes from within-judge sampling noise (aleatoric, hard to fix) vs
            between-judge disagreement (epistemic, fixable by calling more judges).
          </p>
        </section>
      )}

      <section>
        <CardTitle>5. judge rationales</CardTitle>
        <p className="mt-2 max-w-3xl text-sm muted leading-relaxed">
          The natural-language explanation each judge gave alongside its score. These are produced
          via tool-use structured output, so the score and the rationale are guaranteed to come
          from the same forward pass.
        </p>
        <div className="mt-4 flex flex-col gap-4">
          {point.map((r) => (
            <Card key={r.judge_id}>
              <div className="flex items-baseline justify-between">
                <JudgeBadge judgeId={r.judge_id} />
                <div className="text-xs muted font-mono">
                  score {formatScore(r.score_value)} ·{" "}
                  {r.cache_read_tokens > 0 ? "cache hit · " : ""}
                  {r.input_tokens + r.output_tokens} tokens · {Math.round(r.latency_ms)}ms
                </div>
              </div>
              <p className="mt-3 text-sm leading-relaxed whitespace-pre-wrap">{r.rationale}</p>
              {r.flags.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {r.flags.map((f) => (
                    <span
                      key={f}
                      className="text-xs px-2 py-0.5 rounded-md"
                      style={{ background: "var(--surface-2)" }}
                    >
                      {f}
                    </span>
                  ))}
                </div>
              )}
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl px-4 py-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
      <div className="text-xs muted">{label}</div>
      <div className="text-lg font-mono tabular-nums">{value}</div>
    </div>
  );
}
