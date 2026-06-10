import type { Probabilities } from "@/lib/artifacts";
import { pct } from "@/lib/format";

const outcomes = ["home", "draw", "away"] as const;

export function ProbabilityBars({
  label,
  probabilities,
}: {
  label: string;
  probabilities?: Probabilities | null;
}) {
  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--paper)]/70 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">{label}</h3>
        <span className="text-xs text-[var(--muted)]">{probabilities ? "available" : "missing"}</span>
      </div>
      <div className="space-y-3">
        {outcomes.map((outcome) => {
          const value = Number(probabilities?.[outcome] ?? 0);
          return (
            <div key={outcome}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-medium uppercase tracking-[0.12em]">{outcome}</span>
                <span className="mono">{pct(value, 1)}</span>
              </div>
              <div className="prob-track">
                <div className="prob-fill" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
