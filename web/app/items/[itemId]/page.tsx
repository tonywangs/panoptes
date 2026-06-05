import Link from "next/link";
import { notFound } from "next/navigation";
import { Card, CardTitle } from "@/components/Card";
import { CodeBlock } from "@/components/CodeBlock";
import { JudgeBadge, ScoreBar } from "@/components/JudgeBadge";
import {
  findItemSource,
  loadAllItemIds,
  loadItemPrompts,
  loadUQ,
} from "@/lib/data";
import {
  formatPercent,
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

  // group samples by judge for the sampling-pass distribution
  const samplesByJudge: Record<string, number[]> = {};
  for (const s of samples) {
    (samplesByJudge[s.judge_id] ??= []).push(s.score_value);
  }

  return (
    <div className="flex flex-col gap-8">
      <header>
        <Link href={`/runs/${run.run_id}`} className="text-xs muted hover:text-foreground">
          ← {run.run_id}
        </Link>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight font-mono">{shortItemId(itemId)}</h1>
        <div className="mt-1 muted text-sm">
          {point[0]?.benchmark} · {point[0]?.task_family}
        </div>
      </header>

      {prompt && (
        <section>
          <CardTitle>task</CardTitle>
          <div className="mt-3">
            <CodeBlock code={prompt.prompt} />
          </div>
          <div className="mt-1 text-xs muted">
            source: {prompt.source}
            {prompt.original_entry_point && (
              <>
                {" "}· obfuscated from <span className="font-mono">{prompt.original_entry_point}</span> →{" "}
                <span className="font-mono">{prompt.entry_point}</span>
              </>
            )}
          </div>
        </section>
      )}

      <section>
        <CardTitle>candidate response judged</CardTitle>
        <div className="mt-3">
          <CodeBlock code={point[0]?.model_response ?? ""} />
        </div>
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardTitle>point-pass scores (temperature 0)</CardTitle>
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
          <CardTitle>sampling-pass dispersion (temperature 1)</CardTitle>
          <div className="mt-4 flex flex-col gap-3">
            {Object.keys(samplesByJudge).length === 0 ? (
              <div className="text-sm muted">no sampling pass for this run</div>
            ) : (
              Object.entries(samplesByJudge).map(([judge, vals]) => {
                const n = vals.length;
                const mean = vals.reduce((s, v) => s + v, 0) / n;
                const variance =
                  n > 1 ? vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
                const min = Math.min(...vals);
                const max = Math.max(...vals);
                return (
                  <div key={judge} className="text-sm">
                    <div className="flex items-baseline gap-2">
                      <span className={`font-mono ${judgeColor(judge)}`}>
                        {shortJudge(judge)}
                      </span>
                      <span className="text-xs muted">n={n}</span>
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-xs muted font-mono">
                      <span>mean {formatScore(mean)}</span>
                      <span>·</span>
                      <span>var {variance.toFixed(4)}</span>
                      <span>·</span>
                      <span>
                        range [{formatScore(min)}, {formatScore(max)}]
                      </span>
                    </div>
                    <div className="mt-2 relative h-2 rounded-full overflow-hidden" style={{ background: "var(--surface-2)" }}>
                      {vals.map((v, i) => (
                        <span
                          key={i}
                          className="absolute top-0 bottom-0 w-1.5 -ml-0.75 rounded-full"
                          style={{ left: `${v * 100}%`, background: "var(--accent)" }}
                        />
                      ))}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </Card>
      </section>

      {uq.length > 0 && (
        <section>
          <CardTitle>uncertainty quantification</CardTitle>
          <div className="mt-3 grid md:grid-cols-2 gap-4">
            {uq.map((u, i) => (
              <Card key={i}>
                <div className="flex items-baseline justify-between">
                  <div className="font-medium">{u.method}</div>
                  <JudgeBadge judgeId={u.judge_id} />
                </div>
                <pre className="mt-3 text-xs muted font-mono whitespace-pre-wrap break-words">
                  {JSON.stringify(u.value, null, 2)}
                </pre>
              </Card>
            ))}
          </div>
        </section>
      )}

      <section>
        <CardTitle>judge rationales</CardTitle>
        <div className="mt-3 flex flex-col gap-4">
          {point.map((r) => (
            <Card key={r.judge_id}>
              <div className="flex items-baseline justify-between">
                <JudgeBadge judgeId={r.judge_id} />
                <div className="text-xs muted font-mono">
                  score {formatScore(r.score_value)} · {r.cache_read_tokens > 0 ? "cache hit · " : ""}
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
