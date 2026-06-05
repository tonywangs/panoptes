import Link from "next/link";
import { AlertTriangle, ArrowRight, BookOpen, Brain, Layers, Lightbulb, ScatterChart } from "lucide-react";
import { Card, CardTitle } from "@/components/Card";
import { BeforeAfter } from "@/components/BeforeAfter";
import { JudgeBadge } from "@/components/JudgeBadge";
import { JudgeNoiseChart } from "@/components/JudgeNoiseChart";
import { MethodStack } from "@/components/MethodStack";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import { UncertaintyQuadrant } from "@/components/UncertaintyQuadrant";
import { findItemSource, loadAllItemIds } from "@/lib/data";
import { formatScore, shortJudge } from "@/lib/format";

/**
 * The "tell me what this whole field is" page. Lives between Overview
 * (which is more "here are the headline numbers") and the per-run /
 * per-item drill-downs. Designed to be read aloud during the demo.
 */
export default function BackgroundPage() {
  // Pick a real example to use in the "judges disagree" demo. We want one
  // where there's actually visible spread between judges and at least one
  // sampling pass attached.
  const noiseExample = pickNoisyItem();

  return (
    <div className="flex flex-col gap-14">
      <header>
        <CardTitle>background</CardTitle>
        <h1 className="mt-2 text-4xl md:text-5xl font-semibold tracking-tight">
          Why <span className="text-emerald-500">evaluating LLMs is broken</span>, and what to do
          about it.
        </h1>
        <p className="mt-4 max-w-3xl text-lg muted leading-relaxed">
          The state of LLM evaluation today is "ask another LLM to grade the output and report a
          number." That works as a quick signal, but it hides three real problems that compound at
          scale: judges disagree with each other, judges disagree with themselves under
          resampling, and a single number gives no idea how much to trust the result. PANOPTES is
          built to make those problems first-class.
        </p>
      </header>

      {/* WHAT THE STATUS QUO LOOKS LIKE */}
      <section>
        <CardTitle>what the field looks like today</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">A single number, taken at face value</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Promptfoo, OpenAI Evals, LangSmith, Inspect — the major eval frameworks all share the
          same shape. You define a rubric, you point one LLM (the "judge") at the responses to
          score, you get back a scalar. Sometimes a 0–1 score, sometimes a Likert. There's
          usually no second judge for comparison, no resampling-variance budget, no confidence
          interval, no theoretical guarantee.
        </p>
        <div className="mt-6">
          <BeforeAfter />
        </div>
      </section>

      {/* JUDGES ARE NOISY */}
      <section>
        <CardTitle>problem 1 — judges are noisy</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">
          The same task, the same response, three judges, three different numbers.
        </h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Real PANOPTES data: one HumanEval problem judged by three frontier LLMs at temperature 0
          (solid dots), plus several samples each at temperature 1 (hollow dots). Means are bars.
          If "the judges all measure the same thing on the same scale" were true, those dots would
          stack on top of each other. They don't.
        </p>
        {noiseExample ? (
          <Card className="mt-4">
            <div className="flex items-baseline justify-between mb-2">
              <div className="text-xs muted font-mono">item {noiseExample.itemId}</div>
              <div className="text-xs muted">range: {noiseExample.range.toFixed(3)}</div>
            </div>
            <JudgeNoiseChart
              pointRows={noiseExample.point.map((r) => ({ judge_id: r.judge_id, score_value: r.score_value }))}
              samplesByJudge={noiseExample.samplesByJudge}
            />
          </Card>
        ) : (
          <Card className="mt-4">
            <div className="text-sm muted">No item with both point + sampling data found.</div>
          </Card>
        )}
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Notice two failures of the "single number" story: (1) the judges' point estimates are
          structurally different even at temperature 0 — they literally see the candidate
          differently. (2) Each judge's own sampling-pass dots spread out at temperature 1 — even
          a single judge isn't sure what its own number should be.
        </p>
      </section>

      {/* TWO KINDS OF UNCERTAINTY */}
      <section>
        <CardTitle>problem 2 — two kinds of uncertainty</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">
          Some noise is fixable. Some isn't.
        </h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Lumping all uncertainty into a single "± something" hides the fact that two very
          different things drive it: <span className="text-foreground">aleatoric</span> uncertainty
          comes from genuine task ambiguity — no amount of additional sampling will resolve it —
          and <span className="text-foreground">epistemic</span> uncertainty comes from
          disagreement between judges, which <em>does</em> shrink as you call more / stronger
          judges. The right action depends on which one dominates.
        </p>
        <div className="mt-6">
          <UncertaintyQuadrant />
        </div>
        <p className="mt-4 max-w-3xl text-sm muted leading-relaxed">
          PANOPTES estimates this split via <span className="text-foreground">nested
          resampling</span> — outer bootstrap over judges (epistemic), inner over temperature
          samples within judge (aleatoric). The numbers feed straight into the routing layer: if
          epistemic dominates, the bandit calls another judge; if aleatoric dominates, calling
          more judges won't help, so it stops.
        </p>
      </section>

      {/* NO GUARANTEES */}
      <section>
        <CardTitle>problem 3 — no guarantees</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">
          What does a "90% confidence interval" on an LLM score even mean?
        </h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Most CI machinery assumes Gaussian residuals or large-sample asymptotics. Neither holds
          for an LLM judge's score, which is bounded in [0, 1], heavily multimodal, and trained
          on data we don't get to see. PANOPTES sidesteps this with{" "}
          <span className="text-foreground">conformal prediction</span>: a calibration recipe that
          guarantees the prediction interval contains the true value at least 1 − α of the time,
          under nothing more than exchangeability of the calibration set. No parametric model, no
          Gaussian assumption, finite-sample valid.
        </p>
        <div className="mt-4 grid md:grid-cols-2 gap-4">
          <Card>
            <div className="flex items-start gap-3">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                style={{ background: "#a78bfa22", color: "#a78bfa" }}
              >
                <AlertTriangle size={18} />
              </div>
              <div>
                <div className="text-sm font-medium">a normal "± 2σ" CI</div>
                <p className="mt-1 text-xs muted leading-relaxed">
                  Assumes the underlying distribution is Gaussian. Mostly meaningless for an
                  LLM-judge score that's bounded, multimodal, and shaped by alignment training.
                  Coverage is a wish, not a guarantee.
                </p>
              </div>
            </div>
          </Card>
          <Card className="border-emerald-500/30">
            <div className="flex items-start gap-3">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                style={{ background: "#10b98122", color: "#10b981" }}
              >
                <BookOpen size={18} />
              </div>
              <div>
                <div className="text-sm font-medium">conformal interval</div>
                <p className="mt-1 text-xs muted leading-relaxed">
                  Calibrated on a held-out set, the <code className="font-mono text-foreground">⌈(n+1)(1−α)⌉/n</code>
                  -th quantile of conformity scores gives a finite-sample-valid interval. No
                  distributional assumption. We verify the guarantee empirically on{" "}
                  <Link href="/calibration" className="text-emerald-500 hover:underline">
                    /calibration
                  </Link>.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </section>

      {/* THE STACK */}
      <section>
        <CardTitle>how PANOPTES addresses all three</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">A stack of statistical methods, each paper-grounded</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          The framework composes six layers. The bottom layer talks to LLM providers; everything
          above it is statistics. Every layer cites the paper its math comes from.
        </p>
        <div className="mt-6">
          <MethodStack />
        </div>
      </section>

      {/* THE PIPELINE */}
      <section>
        <CardTitle>at runtime, one item at a time</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Five stages, every evaluation</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          When you run <code className="font-mono text-foreground">panoptes eval humaneval</code>,
          every item flows through these five stages. Stages 1–3 produce the raw signal;
          stage 4 turns that signal into a calibrated posterior; stage 5 closes the loop by
          deciding what to do on the next item.
        </p>
        <div className="mt-5">
          <PipelineDiagram />
        </div>
      </section>

      {/* WHO BENEFITS */}
      <section>
        <CardTitle>who actually needs this</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Use cases &amp; potential impact</h2>
        <div className="mt-4 grid md:grid-cols-2 gap-4">
          <UseCase
            icon={Brain}
            color="#10b981"
            title="Model providers running safety / capability evals at scale"
            body="Anthropic, OpenAI, Google all run thousands of LLM-judge evals per release. PANOPTES tells them which judgments to trust and which to escalate, instead of averaging through the noise."
          />
          <UseCase
            icon={ScatterChart}
            color="#a78bfa"
            title="Benchmark authors who need to defend their numbers"
            body="If your paper says 'model X beats model Y by 4.2 points,' PANOPTES gives you an honest CI on that gap plus a calibrated p-value, instead of a brittle point estimate."
          />
          <UseCase
            icon={Layers}
            color="#38bdf8"
            title="Eng teams shipping LLM-graded user-facing pipelines"
            body="Quality control, content moderation, claim-verification — anywhere an LLM judges another LLM's output and the result matters. Knowing when the judge isn't sure is the whole game."
          />
          <UseCase
            icon={Lightbulb}
            color="#fbbf24"
            title="Researchers studying judge bias or alignment"
            body="The hierarchical-Gaussian aggregator exposes per-judge bias and precision as first-class outputs. PANOPTES lets you audit judge behavior, not just consume it."
          />
        </div>
      </section>

      {/* WHAT'S NEXT */}
      <section className="grid md:grid-cols-[1fr_auto] gap-6 items-end">
        <div>
          <CardTitle>where to go from here</CardTitle>
          <h2 className="mt-2 text-2xl font-medium tracking-tight">
            The rest of the site is the empirical receipts.
          </h2>
          <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
            Every claim on this background page is backed by data on the deeper pages. The
            calibration page measures whether the conformal guarantee actually holds. The judges
            page shows real inter-judge agreement. The runs and items pages show the framework in
            action on real benchmark data. The methods page lists every paper.
          </p>
        </div>
        <div className="flex flex-col gap-2 shrink-0">
          <Link
            href="/calibration"
            className="inline-flex items-center justify-between gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 text-emerald-500 ring-1 ring-emerald-500/30 hover:bg-emerald-500/20 transition-colors text-sm"
          >
            see the calibration result <ArrowRight size={14} />
          </Link>
          <Link
            href="/runs"
            className="inline-flex items-center justify-between gap-2 px-4 py-2 rounded-lg hover:bg-[var(--surface-2)] transition-colors text-sm"
            style={{ border: "1px solid var(--border)" }}
          >
            browse runs <ArrowRight size={14} />
          </Link>
          <Link
            href="/methods"
            className="inline-flex items-center justify-between gap-2 px-4 py-2 rounded-lg hover:bg-[var(--surface-2)] transition-colors text-sm"
            style={{ border: "1px solid var(--border)" }}
          >
            paper citations <ArrowRight size={14} />
          </Link>
        </div>
      </section>
    </div>
  );
}

function UseCase({
  icon: Icon,
  color,
  title,
  body,
}: {
  icon: typeof Brain;
  color: string;
  title: string;
  body: string;
}) {
  return (
    <Card>
      <div className="flex items-start gap-3">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: `${color}22`, color }}
        >
          <Icon size={18} />
        </div>
        <div>
          <div className="text-sm font-medium">{title}</div>
          <p className="mt-1.5 text-xs muted leading-relaxed">{body}</p>
        </div>
      </div>
    </Card>
  );
}

/**
 * Pick a real (item, run) where there's at least 2 judges in the point pass
 * AND at least 2 sampling-pass rows. Prefer items with high inter-judge
 * spread so the noise illustration is visually obvious.
 */
function pickNoisyItem(): {
  itemId: string;
  point: { judge_id: string; score_value: number }[];
  samplesByJudge: Record<string, number[]>;
  range: number;
} | null {
  const ids = loadAllItemIds();
  let best: ReturnType<typeof pickNoisyItem> = null;
  for (const itemId of ids) {
    const src = findItemSource(itemId);
    if (!src) continue;
    const point = src.rows.filter((r) => r.sample_index === 0);
    if (point.length < 2) continue;
    const scores = point.map((r) => r.score_value);
    const range = Math.max(...scores) - Math.min(...scores);
    if (range < 0.05) continue;
    const samplesByJudge: Record<string, number[]> = {};
    for (const r of src.rows.filter((r) => r.sample_index > 0)) {
      (samplesByJudge[r.judge_id] ??= []).push(r.score_value);
    }
    if (Object.keys(samplesByJudge).length < 2) continue;
    if (!best || range > best.range) {
      best = {
        itemId,
        point: point.map((r) => ({ judge_id: r.judge_id, score_value: r.score_value })),
        samplesByJudge,
        range,
      };
    }
  }
  return best;
}
