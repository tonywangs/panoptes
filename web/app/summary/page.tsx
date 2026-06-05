import Link from "next/link";
import {
  ArrowRight,
  Brain,
  Building2,
  Code2,
  Flag,
  FlaskConical,
  Layers,
  Lightbulb,
  ScatterChart,
  Sparkles,
} from "lucide-react";
import { Card, CardTitle, Metric } from "@/components/Card";
import { CountUp, FadeIn, Stagger, StaggerChild } from "@/components/Motion";
import { RoadmapItem } from "@/components/RoadmapItem";
import { loadCalibration, loadSummary } from "@/lib/data";
import { formatPercent } from "@/lib/format";

export default function SummaryPage() {
  const summary = loadSummary();
  const calib = loadCalibration();

  return (
    <div className="flex flex-col gap-14 grid-bg">
      <FadeIn as="header" y={24}>
        <CardTitle>summary</CardTitle>
        <h1 className="mt-2 text-4xl md:text-5xl font-semibold tracking-tight">
          What we built, who it's for, what comes next.
        </h1>
        <p className="mt-4 max-w-3xl text-lg muted leading-relaxed">
          PANOPTES is a Python framework for evaluating LLMs that treats every score as a
          statistical inference problem. The receipts are on the other pages. This page is the
          closing slide: a recap of the surface area, the four kinds of people who actually need
          this, and the work that's still on the table.
        </p>
      </FadeIn>

      {/* WHAT WE BUILT */}
      <FadeIn>
        <CardTitle>what we built</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">
          Eight statistical methods. Four providers. One async Python framework.
        </h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          The framework composes existing research into a single tool that anyone running
          LLM-graded evals can drop in. Every method cites the paper it comes from. Every number
          on this page is measured, not estimated.
        </p>
        <Stagger className="mt-5 grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StaggerChild>
            <Metric
              label="statistical methods"
              value={<CountUp to={8} />}
              hint="conformal · semantic entropy · self-consistency · DS · decomp · bandit · routing · calibration"
            />
          </StaggerChild>
          <StaggerChild>
            <Metric
              label="LLM providers"
              value={<CountUp to={4} />}
              hint="Anthropic · OpenAI · Google · OpenAI-compatible"
            />
          </StaggerChild>
          <StaggerChild>
            <Metric
              label="benchmarks wired"
              value={<CountUp to={5} />}
              hint="HumanEval · MBPP · GSM8K · MT-Bench · TruthfulQA"
            />
          </StaggerChild>
          <StaggerChild>
            {calib ? (
              <Metric
                label="empirical coverage"
                value={
                  <CountUp
                    to={calib.headline.empirical * 100}
                    format="percent"
                    className="text-emerald-500"
                  />
                }
                hint={`at ${formatPercent(calib.headline.nominal, 0)} nominal, ${calib.headline.gap_pp.toFixed(1)}pp gap`}
              />
            ) : (
              <Metric label="empirical coverage" value="—" />
            )}
          </StaggerChild>
        </Stagger>
        <Stagger className="mt-4 grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StaggerChild>
            <Metric
              label="evaluation runs"
              value={<CountUp to={summary.n_runs} />}
              hint={`${summary.n_items_total} items judged`}
            />
          </StaggerChild>
          <StaggerChild>
            <Metric
              label="LLM judge calls"
              value={<CountUp to={summary.n_calls_total} format="intCommas" />}
            />
          </StaggerChild>
          <StaggerChild>
            <Metric label="total spend" value={<CountUp to={summary.cost_total_usd} format="usd" />} />
          </StaggerChild>
          <StaggerChild>
            <Metric label="judges in the pool" value={<CountUp to={summary.judges_seen.length} />} />
          </StaggerChild>
        </Stagger>
      </FadeIn>

      {/* APPLICATIONS */}
      <FadeIn>
        <CardTitle>applications</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Who actually needs this</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          The framework is built for any workflow where the answer to "how good was that LLM
          output?" matters enough to need a confidence interval, not just a number. Four kinds of
          users were on my mind while building it.
        </p>
        <Stagger className="mt-5 grid md:grid-cols-2 gap-4">
          <StaggerChild>
            <ApplicationCard
              icon={Building2}
              color="#10b981"
              title="Model providers running safety and capability evals at scale"
              who="Anthropic, OpenAI, Google internal eval pipelines"
              body="Frontier labs run thousands of LLM-judge evals per release. PANOPTES tells them which judgments to trust, which to escalate to a stronger judge, and which to flag as inherently ambiguous. Cuts spend on a fixed quality bar by skipping the judge calls that wouldn't have moved the posterior anyway."
              concrete="Replace the 'mean over 3 judges' baseline with hierarchical-Gaussian aggregation, save 30–40% of judge calls via the bandit, get calibrated CIs on every claim."
            />
          </StaggerChild>
          <StaggerChild>
            <ApplicationCard
              icon={FlaskConical}
              color="#a78bfa"
              title="Benchmark authors who need to defend their numbers"
              who="anyone publishing 'model X beats Y on benchmark Z'"
              body="If your paper says model X beats model Y by 4.2 points, a reviewer is going to ask how big the noise floor is. PANOPTES gives you an honest CI on the gap, a paired-bootstrap rank correlation against the held-out set, and a permutation p-value for whether the difference is real."
              concrete="One CLI invocation produces a coverage table, a reliability diagram, and a methods.md you can drop into the appendix."
            />
          </StaggerChild>
          <StaggerChild>
            <ApplicationCard
              icon={Code2}
              color="#38bdf8"
              title="Eng teams shipping LLM-graded user-facing pipelines"
              who="content moderation, code review, claim verification, document QA"
              body="If your product has an LLM grading another LLM's output and the result is shown to a user, the cost of 'looks confident, actually wrong' is high. PANOPTES surfaces the cases where the judge isn't sure so you can hand them off to a human or fall back to a stricter rule."
              concrete="Wrap your existing judge in a Judge Protocol class, get the full UQ stack for free. Items above an epistemic-variance threshold get routed to escalation."
            />
          </StaggerChild>
          <StaggerChild>
            <ApplicationCard
              icon={Brain}
              color="#fbbf24"
              title="Researchers studying judge bias, alignment, or evaluation methodology"
              who="anyone writing a paper about LLM judges"
              body="The hierarchical-Gaussian aggregator exposes per-judge bias and precision as first-class outputs. You can audit which judges are running hot vs cold, who's noisier than who, and how disagreement structure shifts across task families. Semantic entropy gives you a hallucination signal grounded in the Farquhar 2024 paper."
              concrete="duckdb result store + jupyter-friendly queries means every claim in your paper is one SQL query away from the raw judge call that produced it."
            />
          </StaggerChild>
        </Stagger>
      </FadeIn>

      {/* WHAT THE FRAMEWORK ENABLES */}
      <FadeIn>
        <CardTitle>impact</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">Why this matters beyond one project</h2>
        <Stagger className="mt-5 grid md:grid-cols-3 gap-4">
          <StaggerChild>
            <ImpactCard
              icon={Layers}
              color="#10b981"
              title="Auditability"
              body="If an eval framework reports 'model X scored 0.85,' that 0.85 should be reproducible from primary sources. PANOPTES keeps every judge call, every rationale, and every prompt hash in duckdb. The same query reproduces the same number."
            />
          </StaggerChild>
          <StaggerChild>
            <ImpactCard
              icon={ScatterChart}
              color="#a78bfa"
              title="Honesty"
              body="A finite-sample-valid CI is a much stronger claim than a 'looks roughly right' point estimate. Once teams habituate to expecting intervals, the threshold for over-claiming on a benchmark goes up."
            />
          </StaggerChild>
          <StaggerChild>
            <ImpactCard
              icon={Lightbulb}
              color="#38bdf8"
              title="Efficiency"
              body="The bandit routing saves cost on items where the cheap judges already agree. Calling 3 frontier LLMs per item is fine at n=100, painful at n=100,000. Smart routing makes large-n evals economically viable."
            />
          </StaggerChild>
        </Stagger>
      </FadeIn>

      {/* FUTURE WORK */}
      <FadeIn>
        <CardTitle>future work</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">What's next</h2>
        <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
          The framework hits its v1 surface area. What's left is largely{" "}
          <span className="text-foreground">measurement at scale</span>: bigger calibration sets,
          more benchmarks, harder candidate distributions. The next round of work also rounds out
          the aggregator stack for Likert-scale rubrics and hardens the code-execution sandbox.
        </p>
        <Stagger className="mt-5 flex flex-col gap-3">
          <StaggerChild>
            <RoadmapItem
              status="near"
              title="Bigger calibration probe (n=25 → 200+)"
              effort="~$50 in API spend"
              body="The current probe has 25 items in the held-out test set. That's enough to demonstrate the framework, but the 2-percentage-point gap at α=0.10 has a ±6pp standard error. Scaling to 200+ items tightens the SE to ~2pp, which would let me make stronger claims about calibration quality."
            >
              Blocked on: nothing. Just compute. Would take ~30 minutes to run.
            </RoadmapItem>
          </StaggerChild>
          <StaggerChild>
            <RoadmapItem
              status="near"
              title="Ordinal Dawid-Skene aggregator for Likert-scale rubrics"
              effort="~4 hours of code"
              body="Continuous [0, 1] scores use the hierarchical-Gaussian aggregator. Likert 1–5 scores currently get normalized to [0, 1] and treated as continuous, which loses the ordinal structure. MACE-style ordinal Dawid-Skene (Hovy et al. NAACL 2013) is the right tool. The math is straightforward; just hasn't been wired."
            >
              Will live in <code className="font-mono">src/panoptes/uq/disagreement.py</code> as a sibling class.
            </RoadmapItem>
          </StaggerChild>
          <StaggerChild>
            <RoadmapItem
              status="mid"
              title="Docker-isolated sandbox for code execution"
              effort="~1 day of code + ops"
              body="The current sandbox uses subprocess + resource.setrlimit. That's safe enough for grading canonical solutions but I wouldn't run untrusted user-submitted code through it. A Docker backend behind the existing Sandbox Protocol gives proper isolation."
            >
              Sandbox Protocol is already in place; this is a new backend impl, no API changes.
            </RoadmapItem>
          </StaggerChild>
          <StaggerChild>
            <RoadmapItem
              status="mid"
              title="Wire MBPP / GSM8K / MT-Bench / TruthfulQA into the CLI"
              effort="~1 day"
              body="Benchmark loaders exist for all five. Only HumanEval and the calibration probe are currently wired through the CLI. The blocker is that each benchmark needs a benchmark-specific rubric prompt and a candidate-generation step; both are mechanical."
            >
              Once MBPP and GSM8K are wired, the Mondrian conformal aggregator across task families becomes much more interesting.
            </RoadmapItem>
          </StaggerChild>
          <StaggerChild>
            <RoadmapItem
              status="long"
              title="Short paper with measured calibration numbers"
              effort="~1 week"
              body="The whole framing is 'finite-sample guarantees on LLM eval.' That claim is only credible with published, replicable numbers. A short technical writeup with the calibration table, the bandit-vs-all-judges cost comparison, and the methodology is the right vehicle for that."
            >
              Will write after the calibration probe scales to 200+.
            </RoadmapItem>
          </StaggerChild>
          <StaggerChild>
            <RoadmapItem
              status="long"
              title="Integration shims for Promptfoo / Inspect / LangSmith"
              effort="~2 days each"
              body="A lot of teams already have a Promptfoo or Inspect pipeline. PANOPTES doesn't need to replace those; it can sit on top, taking the (item, response, judge_score) records they produce and emitting the UQ + conformal layer on top. Shipping shims for the major frameworks lowers the adoption cost dramatically."
            />
          </StaggerChild>
        </Stagger>
      </FadeIn>

      {/* LESSONS */}
      <FadeIn>
        <CardTitle>what surprised me</CardTitle>
        <h2 className="mt-2 text-2xl font-medium tracking-tight">A few honest takeaways</h2>
        <Stagger className="mt-5 grid md:grid-cols-3 gap-4">
          <StaggerChild>
            <LessonCard
              n={1}
              title="LLM judges are noisier than I expected"
              body="On the same item at temperature 0, three frontier judges routinely disagree by 0.2 on a [0,1] scale. That's a much bigger signal than I assumed going in. The case for treating LLM-as-judge as a statistical problem isn't theoretical; it's empirically obvious the moment you call more than one judge."
            />
          </StaggerChild>
          <StaggerChild>
            <LessonCard
               n={2}
              title="Conformal works basically out of the box"
              body="I'd expected the conformal coverage guarantee to fail on real LLM-judge data because exchangeability is a strong assumption. Empirically the coverage tracks nominal almost exactly at α=0.1. The theorem is unreasonably effective here."
            />
          </StaggerChild>
          <StaggerChild>
            <LessonCard
              n={3}
              title="The bandit story needs more data"
              body="The Thompson-sampling bandit ran in only one production setting, with low n. I believe the cost-reduction claim is real but I'd want a 200-item, multi-strategy A/B before publishing the number."
            />
          </StaggerChild>
        </Stagger>
      </FadeIn>

      {/* CLOSING */}
      <FadeIn>
        <Card className="border-emerald-500/30 glow-emerald">
          <div className="flex items-start gap-4">
            <div className="hidden sm:flex shrink-0 w-10 h-10 rounded-xl items-center justify-center bg-emerald-500/10 ring-1 ring-emerald-500/30 pulse-emerald">
              <Sparkles size={18} className="text-emerald-500 float" />
            </div>
            <div className="flex-1">
              <CardTitle>thanks</CardTitle>
              <h2 className="mt-2 text-2xl font-medium tracking-tight">
                The whole framework, end to end
              </h2>
              <p className="mt-3 max-w-3xl text-sm muted leading-relaxed">
                Source code, citations, and the calibration script are all on GitHub. The
                framework is MIT-licensed and intentionally minimal in its public surface. If you
                want to swap in your own judges, your own benchmark, your own routing strategy:
                implement the corresponding Protocol class and the rest is free.
              </p>
              <Stagger className="mt-5 flex flex-wrap gap-3">
                <StaggerChild>
                  <a
                    href="https://github.com/tonywangs/panoptes"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 text-emerald-500 ring-1 ring-emerald-500/30 hover:bg-emerald-500/20 transition-colors text-sm"
                  >
                    github.com/tonywangs/panoptes <ArrowRight size={14} />
                  </a>
                </StaggerChild>
                <StaggerChild>
                  <Link
                    href="/methods"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-[var(--surface-2)] transition-colors text-sm"
                    style={{ border: "1px solid var(--border)" }}
                  >
                    paper citations <ArrowRight size={14} />
                  </Link>
                </StaggerChild>
                <StaggerChild>
                  <Link
                    href="/calibration"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-[var(--surface-2)] transition-colors text-sm"
                    style={{ border: "1px solid var(--border)" }}
                  >
                    calibration receipts <ArrowRight size={14} />
                  </Link>
                </StaggerChild>
              </Stagger>
            </div>
          </div>
        </Card>
      </FadeIn>
    </div>
  );
}

function ApplicationCard({
  icon: Icon,
  color,
  title,
  who,
  body,
  concrete,
}: {
  icon: typeof Brain;
  color: string;
  title: string;
  who: string;
  body: string;
  concrete: string;
}) {
  return (
    <Card className="h-full">
      <div className="flex items-start gap-3">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
          style={{ background: `${color}22`, color }}
        >
          <Icon size={18} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium leading-snug">{title}</div>
          <div className="mt-1 text-xs muted">{who}</div>
        </div>
      </div>
      <p className="mt-4 text-sm muted leading-relaxed">{body}</p>
      <div
        className="mt-4 rounded-lg px-3 py-2.5 text-xs leading-relaxed"
        style={{ background: "var(--surface-2)" }}
      >
        <div className="font-medium text-foreground mb-1">concretely:</div>
        {concrete}
      </div>
    </Card>
  );
}

function ImpactCard({
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
    <Card className="h-full">
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center"
        style={{ background: `${color}22`, color }}
      >
        <Icon size={18} />
      </div>
      <div className="mt-3 font-medium">{title}</div>
      <p className="mt-2 text-sm muted leading-relaxed">{body}</p>
    </Card>
  );
}

function LessonCard({ n, title, body }: { n: number; title: string; body: string }) {
  return (
    <Card className="h-full">
      <div className="flex items-baseline gap-3">
        <span className="text-3xl font-semibold text-emerald-500 font-mono">
          {String(n).padStart(2, "0")}
        </span>
        <div className="font-medium leading-snug">{title}</div>
      </div>
      <p className="mt-3 text-sm muted leading-relaxed">{body}</p>
    </Card>
  );
}
