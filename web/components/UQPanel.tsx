import { JudgeBadge } from "@/components/JudgeBadge";
import { UQResult } from "@/lib/types";
import { judgeChartColor, shortJudge } from "@/lib/format";

/**
 * Pretty-rendered UQ blobs. Replaces the previous raw-JSON viewer with
 * dedicated cards per method:
 *   - self-consistency : mean ± Bayesian-bootstrap CI, IQR, sample count
 *   - semantic-entropy : entropy value with bar against log(N) max, cluster split
 *   - decomposition    : aleatoric vs epistemic horizontal bars
 *
 * Anything we don't recognize falls back to a labeled key-value table.
 */
export function UQPanel({ results }: { results: UQResult[] }) {
  if (results.length === 0) return null;
  return (
    <div className="grid md:grid-cols-2 gap-4">
      {results.map((u, i) => (
        <UQCard key={`${u.method}-${u.judge_id}-${i}`} result={u} />
      ))}
    </div>
  );
}

function UQCard({ result }: { result: UQResult }) {
  const value = (result.value ?? {}) as Record<string, unknown>;
  const judgeLabel = result.judge_id === "__aggregate__" ? "aggregate (all judges)" : null;
  return (
    <div
      className="rounded-2xl px-5 py-4"
      style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider muted">{result.method}</div>
        {judgeLabel ? (
          <span className="text-xs muted">{judgeLabel}</span>
        ) : (
          <JudgeBadge judgeId={result.judge_id} />
        )}
      </div>
      <div className="mt-3">
        {result.method === "self-consistency" ? (
          <SelfConsistency v={value} />
        ) : result.method === "semantic-entropy" ? (
          <SemanticEntropy v={value} />
        ) : result.method === "decomposition" ? (
          <Decomposition v={value} />
        ) : (
          <RawValue v={value} />
        )}
      </div>
    </div>
  );
}

function SelfConsistency({ v }: { v: Record<string, unknown> }) {
  const mean = numberOr(v.mean, 0);
  const ciLow = numberOr(v.ci_low, mean);
  const ciHigh = numberOr(v.ci_high, mean);
  const variance = numberOr(v.variance, 0);
  const n = numberOr(v.n_samples, 0);
  return (
    <div>
      <div className="text-2xl font-semibold tabular-nums">
        {mean.toFixed(3)}
        <span className="text-sm muted ml-2 font-normal">
          [{ciLow.toFixed(3)}, {ciHigh.toFixed(3)}]
        </span>
      </div>
      <div className="text-xs muted">posterior mean · 90% Bayesian-bootstrap CI</div>
      <div className="mt-3 relative h-2 rounded-full overflow-hidden" style={{ background: "var(--surface-2)" }}>
        <span
          className="absolute top-0 bottom-0 rounded-full"
          style={{
            left: `${ciLow * 100}%`,
            right: `${(1 - ciHigh) * 100}%`,
            background: "var(--accent)",
            opacity: 0.6,
          }}
        />
        <span
          className="absolute top-0 bottom-0 w-0.5"
          style={{ left: `${mean * 100}%`, background: "var(--accent)" }}
        />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs muted">
        <div>variance: <span className="text-foreground font-mono">{variance.toExponential(2)}</span></div>
        <div>n samples: <span className="text-foreground font-mono">{n}</span></div>
      </div>
    </div>
  );
}

function SemanticEntropy({ v }: { v: Record<string, unknown> }) {
  const entropy = numberOr(v.entropy, 0);
  const n = numberOr(v.n_samples, 0);
  const nClusters = numberOr(v.n_clusters, 0);
  const sizes = Array.isArray(v.cluster_sizes) ? (v.cluster_sizes as number[]) : [];
  const maxEntropy = n > 1 ? Math.log(n) : 1;
  const ratio = Math.min(1, entropy / maxEntropy);
  const palette = ["#10b981", "#a78bfa", "#38bdf8", "#fbbf24", "#f43f5e", "#34d399"];
  return (
    <div>
      <div className="text-2xl font-semibold tabular-nums">
        H = {entropy.toFixed(3)}
        <span className="text-sm muted ml-2 font-normal">/ log {n} ≈ {maxEntropy.toFixed(3)}</span>
      </div>
      <div className="text-xs muted">
        {nClusters} semantic cluster{nClusters === 1 ? "" : "s"} via bidirectional NLI
      </div>
      <div className="mt-3">
        <div className="text-xs muted mb-1">entropy as fraction of max</div>
        <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--surface-2)" }}>
          <div
            className="h-full rounded-full"
            style={{
              width: `${ratio * 100}%`,
              background:
                ratio < 0.2 ? "#10b981" : ratio < 0.6 ? "#fbbf24" : "#f43f5e",
            }}
          />
        </div>
      </div>
      {sizes.length > 0 && (
        <div className="mt-3">
          <div className="text-xs muted mb-1">cluster sizes</div>
          <div className="flex w-full h-3 rounded-full overflow-hidden" style={{ background: "var(--surface-2)" }}>
            {sizes.map((s, i) => (
              <span
                key={i}
                style={{
                  flex: s,
                  background: palette[i % palette.length],
                  borderRight: i < sizes.length - 1 ? "2px solid var(--surface)" : undefined,
                }}
                title={`cluster ${i + 1}: ${s} samples`}
              />
            ))}
          </div>
          <div className="mt-1 text-xs muted font-mono">
            {sizes.map((s, i) => `c${i + 1}=${s}`).join(" · ")}
          </div>
        </div>
      )}
    </div>
  );
}

function Decomposition({ v }: { v: Record<string, unknown> }) {
  const total = numberOr(v.total, 0);
  const aleatoric = numberOr(v.aleatoric, 0);
  const epistemic = numberOr(v.epistemic, 0);
  const aPct = total > 0 ? (aleatoric / total) * 100 : 0;
  const ePct = total > 0 ? (epistemic / total) * 100 : 0;
  const nJudges = numberOr(v.n_judges, 0);
  return (
    <div>
      <div className="text-2xl font-semibold tabular-nums">
        Var = {total.toExponential(2)}
      </div>
      <div className="text-xs muted">total predictive variance, {nJudges} judges</div>
      <div className="mt-4 flex h-6 w-full rounded-md overflow-hidden" style={{ background: "var(--surface-2)" }}>
        <span
          className="flex items-center justify-center text-[10px] font-medium text-white"
          style={{ width: `${aPct}%`, background: "#38bdf8" }}
        >
          {aPct >= 12 && `${aPct.toFixed(0)}% aleatoric`}
        </span>
        <span
          className="flex items-center justify-center text-[10px] font-medium text-white"
          style={{ width: `${ePct}%`, background: "#a78bfa" }}
        >
          {ePct >= 12 && `${ePct.toFixed(0)}% epistemic`}
        </span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <div>
          <div className="muted">aleatoric (irreducible)</div>
          <div className="text-foreground font-mono">{aleatoric.toExponential(2)}</div>
        </div>
        <div>
          <div className="muted">epistemic (reducible)</div>
          <div className="text-foreground font-mono">{epistemic.toExponential(2)}</div>
        </div>
      </div>
    </div>
  );
}

function RawValue({ v }: { v: Record<string, unknown> }) {
  return (
    <pre className="text-xs muted font-mono whitespace-pre-wrap break-words max-h-48 overflow-auto">
      {JSON.stringify(v, null, 2)}
    </pre>
  );
}

function numberOr(v: unknown, fallback: number): number {
  return typeof v === "number" && !Number.isNaN(v) ? v : fallback;
}
