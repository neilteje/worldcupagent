import type { Decision } from "@/lib/artifacts";
import { compactDate, pct, signed } from "@/lib/format";

export function DecisionList({
  decisions,
  selectedFile,
}: {
  decisions: Decision[];
  selectedFile?: string;
}) {
  return (
    <aside className="panel sticky top-4 max-h-[calc(100vh-2rem)] overflow-auto rounded-[1.35rem] p-3">
      <div className="mb-3 px-2">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Runs</p>
        <h2 className="mt-1 text-lg font-semibold">Decision artifacts</h2>
      </div>
      <div className="space-y-2">
        {decisions.map((decision) => {
          const selected = decision._file === selectedFile;
          return (
            <a
              key={decision._file}
              href={`/?decision=${encodeURIComponent(decision._file ?? "")}`}
              className={[
                "block rounded-2xl border p-3 transition-colors",
                selected
                  ? "border-[var(--accent)] bg-[var(--accent-soft)]/45"
                  : "border-transparent hover:border-[var(--line)] hover:bg-[var(--panel)]",
              ].join(" ")}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold">{decision.fixture_code}</p>
                  <p className="mt-0.5 text-xs text-[var(--muted)]">{decision.window} · {compactDate(decision._mtime)}</p>
                </div>
                <span className={decision.action === "BET" ? "rounded-full bg-[var(--accent)] px-2 py-1 text-xs font-bold text-white" : "rounded-full bg-[var(--panel-strong)] px-2 py-1 text-xs font-bold"}>
                  {decision.action ?? "n/a"}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                <span>Edge {signed(decision.best_edge)}</span>
                <span>{decision.edge_tier ?? "none"}</span>
                <span>{pct(decision.confidence)}</span>
              </div>
            </a>
          );
        })}
      </div>
    </aside>
  );
}
