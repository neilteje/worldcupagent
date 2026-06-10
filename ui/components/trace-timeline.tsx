import type { LedgerRecord } from "@/lib/artifacts";
import { compactDate, labelize } from "@/lib/format";

const behaviorTone: Record<string, string> = {
  Observing: "bg-sky-100 text-sky-900",
  Planning: "bg-stone-200 text-stone-900",
  ToolCalling: "bg-amber-100 text-amber-950",
  Thinking: "bg-emerald-100 text-emerald-950",
  Acting: "bg-blue-100 text-blue-950",
  Reflecting: "bg-zinc-200 text-zinc-950",
};

export function TraceTimeline({ records }: { records?: LedgerRecord[] }) {
  if (!records?.length) {
    return <EmptyTrace />;
  }

  return (
    <section className="panel rounded-[1.35rem] p-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Reasoning Ledger</p>
          <h2 className="mt-1 text-xl font-semibold">Trace timeline</h2>
        </div>
        <p className="text-sm text-[var(--muted)]">{records.length} records</p>
      </div>
      <div className="mt-5 space-y-3">
        {records.map((record, index) => {
          const trace = record.reasoning_trace ?? {};
          const model = record.model_invocation ?? {};
          return (
            <details key={record.record_id ?? index} className="group rounded-2xl border border-[var(--line)] bg-[var(--paper)]/72 p-4">
              <summary className="flex cursor-pointer list-none flex-wrap items-center gap-3">
                <span className="mono text-xs text-[var(--faint)]">{String(index + 1).padStart(2, "0")}</span>
                <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${behaviorTone[record.behavior ?? ""] ?? "bg-stone-200"}`}>
                  {record.behavior}
                </span>
                <span className="font-semibold">{record.label}</span>
                <span className="ml-auto text-xs text-[var(--muted)]">{compactDate(record.timestamp)}</span>
              </summary>
              <div className="mt-4 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
                <div className="space-y-3 text-sm">
                  <Field label="Objective" value={trace.objective} />
                  <Field label="Method" value={trace.method} />
                  <Field label="Decision rule" value={trace.decision_rule} />
                  <Field label="Output" value={trace.output_summary} />
                </div>
                <div className="space-y-3 text-sm">
                  <Field label="Model" value={model.model_name ? `${model.provider ?? "unknown"} / ${model.model_name}` : undefined} />
                  <Field label="Risk controls" value={Array.isArray(trace.risk_controls) ? trace.risk_controls.map(labelize).join(", ") : undefined} />
                  <Field label="Evidence refs" value={Array.isArray(trace.evidence_refs) ? trace.evidence_refs.length : undefined} />
                  <Field label="Parents" value={record.parent_ids?.length ?? 0} />
                </div>
              </div>
            </details>
          );
        })}
      </div>
    </section>
  );
}

function Field({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">{label}</p>
      <p className="mt-1 leading-6">{value === undefined || value === "" ? "n/a" : String(value)}</p>
    </div>
  );
}

function EmptyTrace() {
  return (
    <section className="panel rounded-[1.35rem] p-6">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Reasoning Ledger</p>
      <h2 className="mt-1 text-xl font-semibold">No trace found</h2>
      <p className="mt-2 text-sm text-[var(--muted)]">The selected decision has no matching `storage/runs` artifact.</p>
    </section>
  );
}
