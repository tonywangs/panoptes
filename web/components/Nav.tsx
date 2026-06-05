import Link from "next/link";
import {
  Activity,
  BarChart3,
  BookOpen,
  FileText,
  Flag,
  Layers,
  ScatterChart,
  Sigma,
} from "lucide-react";

const ITEMS = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/background", label: "Background", icon: BookOpen },
  { href: "/runs", label: "Runs", icon: Layers },
  { href: "/judges", label: "Judges", icon: ScatterChart },
  { href: "/calibration", label: "Calibration", icon: BarChart3 },
  { href: "/methods", label: "Methods", icon: FileText },
  { href: "/summary", label: "Summary", icon: Flag },
];

export function Sidebar() {
  return (
    <aside
      className="hidden md:flex flex-col w-64 shrink-0 border-r min-h-screen p-6"
      style={{ borderColor: "var(--border)", background: "var(--background-elevated)" }}
    >
      <Link href="/" className="flex items-center gap-3 mb-12 group">
        <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-emerald-500/10 ring-1 ring-emerald-500/30">
          <Sigma size={18} className="text-emerald-500" />
        </div>
        <div>
          <div className="text-lg font-semibold tracking-tight">PANOPTES</div>
          <div className="text-[11px] muted leading-none mt-0.5">uncertainty-aware eval</div>
        </div>
      </Link>
      <nav className="flex flex-col gap-1">
        {ITEMS.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm hover:bg-[var(--surface-2)] transition-colors"
          >
            <Icon size={16} className="muted" />
            <span>{label}</span>
          </Link>
        ))}
      </nav>
      <div className="mt-auto pt-6 text-xs muted leading-relaxed">
        <p>
          Calibrated posteriors over LLM-judge scores. Decomposed into aleatoric and epistemic
          components.
        </p>
        <a
          href="https://github.com/tonywangs/panoptes"
          className="inline-block mt-3 underline decoration-dotted underline-offset-2 hover:text-foreground"
        >
          github.com/tonywangs/panoptes
        </a>
      </div>
    </aside>
  );
}

export function MobileNav() {
  return (
    <header
      className="md:hidden sticky top-0 z-10 px-4 py-3 flex items-center gap-2 border-b backdrop-blur"
      style={{ borderColor: "var(--border)", background: "color-mix(in srgb, var(--background) 80%, transparent)" }}
    >
      <Link href="/" className="flex items-center gap-2">
        <div className="w-7 h-7 rounded-lg flex items-center justify-center bg-emerald-500/10 ring-1 ring-emerald-500/30">
          <Sigma size={14} className="text-emerald-500" />
        </div>
        <span className="font-semibold">PANOPTES</span>
      </Link>
      <nav className="ml-auto flex items-center gap-3 text-xs">
        {ITEMS.map(({ href, label }) => (
          <Link key={href} href={href} className="muted hover:text-foreground">
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
