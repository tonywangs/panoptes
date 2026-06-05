import { ArrowRight } from "lucide-react";

/**
 * Side-by-side card: the typical LLM-as-judge output (one number) vs what
 * PANOPTES surfaces (a posterior, a decomposition, a conformal interval).
 * Lives on the background page to motivate the whole project.
 */
export function BeforeAfter() {
  return (
    <div className="grid md:grid-cols-[1fr_auto_1fr] gap-4 md:gap-6 items-stretch">
      <div
        className="rounded-2xl px-5 py-5 relative"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
      >
        <div className="text-xs uppercase tracking-wider muted">most eval frameworks</div>
        <div className="mt-2 text-lg font-medium">"GPT-4 graded it 0.85."</div>
        <div className="mt-3 text-6xl font-semibold tabular-nums">0.85</div>
        <ul className="mt-4 text-sm muted space-y-1.5 leading-relaxed">
          <li>· no confidence interval</li>
          <li>· no second judge for comparison</li>
          <li>· no record of resampling variance</li>
          <li>· "0.85" from one judge ≠ "0.85" from another</li>
        </ul>
      </div>
      <div className="hidden md:flex items-center justify-center">
        <ArrowRight className="text-emerald-500" size={32} />
      </div>
      <div className="md:hidden flex items-center justify-center py-2">
        <ArrowRight className="text-emerald-500 rotate-90" size={24} />
      </div>
      <div
        className="rounded-2xl px-5 py-5 relative"
        style={{
          background: "var(--surface)",
          border: "1px solid color-mix(in srgb, #10b981 35%, transparent)",
        }}
      >
        <div className="text-xs uppercase tracking-wider text-emerald-500">PANOPTES</div>
        <div className="mt-2 text-lg font-medium">"True quality is in [0.71, 0.93] with 90% coverage."</div>
        <div className="mt-3 text-6xl font-semibold tabular-nums">
          0.82
          <span className="ml-2 text-2xl font-normal text-emerald-500 align-middle">
            ± 0.11
          </span>
        </div>
        <ul className="mt-4 text-sm space-y-1.5 leading-relaxed">
          <li className="text-foreground">
            · 3 judges, 5 temperature samples each
          </li>
          <li className="text-foreground">
            · 65% epistemic variance. Call more judges.
          </li>
          <li className="text-foreground">
            · 35% aleatoric. Task itself is ambiguous.
          </li>
          <li className="text-foreground">
            · conformal interval at α = 0.10, finite-sample valid
          </li>
        </ul>
      </div>
    </div>
  );
}
