import Link from "next/link";
import { AlertTriangle, ArrowRight, BookOpen, Brain, Layers, Lightbulb, ScatterChart } from "lucide-react";
import { Card, CardTitle } from "@/components/Card";
import { BeforeAfter } from "@/components/BeforeAfter";
import { JudgeNoiseChart } from "@/components/JudgeNoiseChart";
import { MethodStack } from "@/components/MethodStack";
import { FadeIn, Stagger, StaggerChild } from "@/components/Motion";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import { UncertaintyQuadrant } from "@/components/UncertaintyQuadrant";
import { findItemSource, loadAllItemIds } from "@/lib/data";

export default function BackgroundPage() {
  const noiseExample = pickNoisyItem();

  return (
    <div className="flex flex-col gap-14 grid-bg">
      <FadeIn as="header" y={24}>
        <CardTitle>background</CardTitle>
        <h1 className="mt-2 text-4xl md:text-5xl font-semibold tracking-tight">
          Why <span className="text-emerald-500">evaluating LLMs is broken</span>, and what to do
          about it.
        </h1>
        <p className="mt-4 max-w-3xl text-lg muted leading-relaxed">
          The state of LLM evaluation today is "ask another LLM to grade the output and report a
          number." That works as a quick signal. It also hides three real problems that compound
          at scale: judges disagree with each other, judges disagree with themselves, and a single
          number tells you nothing about how much to trust the result. PANOPTES is built to make
          all three first-class.
        </p>
      </FadeIn>

      {/* WHAT THE STATUS QUO LOOKS LIKE */}
      <FadeIn>
        <CardTitle>what the field looks like today</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">A single number, taken at face value</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Promptfoo, OpenAI Evals, LangSmith, Inspect. The major eval frameworks all share the
          same shape. You write a rubric, you point one LLM at the responses to score them, you
          get back a scalar. Usually no second judge for comparison. No resampling budget. No
          confidence interval. No theoretical guarantee.
        </p>
        <div className="mt-6">
          <BeforeAfter />
        </div>
      </FadeIn>

      {/* JUDGES ARE NOISY */}
      <FadeIn>
        <CardTitle>problem 1 · judges are noisy</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">
          Same task, same response, three judges, three different numbers.
        </h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Real PANOPTES data below. One HumanEval problem judged by three frontier LLMs at
          temperature 0 (solid dots), plus several samples each at temperature 1 (hollow dots).
          The vertical bars are per-judge means. If "the judges all measure the same thing on the
          same scale" were true, those dots would stack on top of each other. They don't.
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
            <div className="text-sm muted">No item with both point and sampling data found.</div>
          </Card>
        )}
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Notice the two ways the "single number" story breaks. First, the judges' point estimates
          are structurally different even at temperature 0. They literally see this candidate
          differently. Second, each judge's own sampling-pass dots spread out at temperature 1.
          Even a single judge isn't sure what its own number should be.
        </p>
      </FadeIn>

      {/* TWO KINDS OF UNCERTAINTY */}
      <FadeIn>
        <CardTitle>problem 2 · two kinds of uncertainty</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">
          Some noise is fixable. Some isn't.
        </h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Lumping all uncertainty into one ± something hides the fact that two very different
          things drive it. <span className="text-foreground">Aleatoric</span> uncertainty comes
          from genuine task ambiguity. No amount of extra sampling resolves it.{" "}
          <span className="text-foreground">Epistemic</span> uncertainty comes from disagreement
          between judges, which <em>does</em> shrink as you call more or stronger judges. The
          right action depends on which one dominates.
        </p>
        <div className="mt-6">
          <UncertaintyQuadrant />
        </div>
        <p className="mt-4 max-w-3xl text-sm muted leading-relaxed">
          PANOPTES estimates this split through nested resampling. The outer bootstrap is over
          judges (that captures epistemic), the inner bootstrap is over temperature samples within
          judge (that captures aleatoric). Those numbers feed straight into the routing layer. If
          epistemic dominates, the bandit calls another judge. If aleatoric dominates, calling
          more judges won't help, so it stops.
        </p>
      </FadeIn>

      {/* NO GUARANTEES */}
      <FadeIn>
        <CardTitle>problem 3 · no guarantees</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">
          What does a "90% confidence interval" on an LLM score even mean?
        </h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          Most CI machinery assumes Gaussian residuals or large-sample asymptotics. Neither holds
          for an LLM judge score, which is bounded in [0, 1], heavily multimodal, and shaped by
          training data we don't get to see. PANOPTES sidesteps the whole problem with{" "}
          <span className="text-foreground">conformal prediction</span>: a calibration recipe that
          guarantees the prediction interval contains the true value at least 1 − α of the time,
          under nothing more than exchangeability of the calibration set. No parametric model. No
          Gaussian assumption. Finite-sample valid.
        </p>
        <Stagger className="mt-4 grid md:grid-cols-2 gap-4">
          <StaggerChild>
            <Card>
              <div className="flex items-start gap-3">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                  style={{ background: "#a78bfa22", color: "#a78bfa" }}
                >
                  <AlertTriangle size={18} />
                </div>
                <div>
                  <div className="text-sm font-medium">a typical "± 2σ" CI</div>
                  <p className="mt-1 text-xs muted leading-relaxed">
                    Assumes the underlying distribution is Gaussian. Mostly meaningless for an
                    LLM-judge score that's bounded, multimodal, and shaped by alignment training.
                    Coverage is a wish, not a guarantee.
                  </p>
                </div>
              </div>
            </Card>
          </StaggerChild>
          <StaggerChild>
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
          </StaggerChild>
        </Stagger>
      </FadeIn>

      {/* THE STACK */}
      <FadeIn>
        <CardTitle>how PANOPTES addresses all three</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">A stack of statistical methods, each paper-grounded</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          The framework composes six layers. The bottom layer talks to LLM providers. Everything
          above it is statistics. Every layer cites the paper its math comes from.
        </p>
        <div className="mt-6">
          <MethodStack />
        </div>
      </FadeIn>

      {/* THE PIPELINE */}
      <FadeIn>
        <CardTitle>at runtime, one item at a time</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Five stages, every evaluation</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          When you run <code className="font-mono text-foreground">panoptes eval humaneval</code>,
          every item flows through these five stages. Stages 1 through 3 produce the raw signal.
          Stage 4 turns that signal into a calibrated posterior. Stage 5 closes the loop by
          deciding what to do on the next item.
        </p>
        <div className="mt-5">
          <PipelineDiagram />
        </div>
      </FadeIn>

      {/* WHO BENEFITS */}
      <FadeIn>
        <CardTitle>who actually needs this</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Use cases and potential impact</h2>
        <Stagger className="mt-4 grid md:grid-cols-2 gap-4">
          <StaggerChild>
            <UseCase
              icon={Brain}
              color="#10b981"
              title="Model providers running safety and capability evals at scale"
              body="Anthropic, OpenAI, Google all run thousands of LLM-judge evals per release. PANOPTES tells them which judgments to trust and which to escalate, instead of averaging through the noise."
            />
          </StaggerChild>
          <StaggerChild>
            <UseCase
              icon={ScatterChart}
              color="#a78bfa"
              title="Benchmark authors who need to defend their numbers"
              body="If your paper says 'model X beats model Y by 4.2 points,' PANOPTES gives you an honest CI on that gap plus a calibrated p-value, instead of a brittle point estimate."
            />
          </StaggerChild>
          <StaggerChild>
            <UseCase
              icon={Layers}
              color="#38bdf8"
              title="Eng teams shipping LLM-graded user-facing pipelines"
              body="Quality control, content moderation, claim-verification. Anywhere an LLM judges another LLM's output and the result matters. Knowing when the judge isn't sure is the whole game."
            />
          </StaggerChild>
          <StaggerChild>
            <UseCase
              icon={Lightbulb}
              color="#fbbf24"
              title="Researchers studying judge bias or alignment"
              body="The hierarchical-Gaussian aggregator exposes per-judge bias and precision as first-class outputs. PANOPTES lets you audit judge behavior, not just consume it."
            />
          </StaggerChild>
        </Stagger>
      </FadeIn>

      {/* WHAT'S NEXT */}
      <FadeIn>
        <div className="grid md:grid-cols-[1fr_auto] gap-6 items-end">
          <div>
            <CardTitle>where to go from here</CardTitle>
            <h2 className="mt-2 text-2xl font-medium tracking-tight">
              The rest of the site is the empirical receipts.
            </h2>
            <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
              Every claim on this page is backed by data on the deeper pages. The calibration page
              measures whether the conformal guarantee actually holds. The judges page shows real
              inter-judge agreement. The runs and items pages show the framework in action on
              real benchmark data. The methods page lists every paper.
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
        </div>
      </FadeIn>
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
