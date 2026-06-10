import { DecisionList } from "@/components/decision-list";
import { HarnessPanel } from "@/components/harness-panel";
import { JsonBlock } from "@/components/json-block";
import { ProbabilityBars } from "@/components/probability-bars";
import { TraceTimeline } from "@/components/trace-timeline";
import { getDecision, getRun, listDecisions, listHarnessSessions, summarize } from "@/lib/artifacts";
import { labelize, money, pct, signed } from "@/lib/format";

export const dynamic = "force-dynamic";

type PageProps = {
  searchParams?: Promise<{ decision?: string }>;
};

export default async function Home({ searchParams }: PageProps) {
  const params = await searchParams;
  const decisions = await listDecisions();
  const harnessSessions = await listHarnessSessions();
  const selected = await getDecision(params?.decision);
  const run = await getRun(selected?.session_id);
  const summary = summarize(decisions);

  if (!selected) {
    return (
      <main className="mx-auto max-w-5xl p-8">
        <section className="panel rounded-[1.5rem] p-8">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">World Cup Agent Console</p>
          <h1 className="mt-2 text-3xl font-semibold">No decision artifacts found</h1>
          <p className="mt-3 text-[var(--muted)]">Run the agent once, then refresh this page. Expected path: `storage/decisions/*.json`.</p>
        </section>
      </main>
    );
  }

  const orderPayload = selected.order?.payload as Record<string, unknown> | undefined;
  const llmParsed = selected.llm_central?.parsed as Record<string, unknown> | undefined;

  return (
    <main className="p-4 lg:p-6">
      <div className="mx-auto grid max-w-[92rem] gap-5 lg:grid-cols-[22rem_1fr]">
        <DecisionList decisions={decisions} selectedFile={selected._file} />

        <div className="space-y-5">
          <header className="panel overflow-hidden rounded-[1.6rem]">
            <div className="grid gap-0 lg:grid-cols-[1fr_22rem]">
              <div className="p-6 lg:p-8">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">World Cup Agent Console</p>
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <h1 className="text-3xl font-semibold tracking-[-0.03em]">{selected.fixture_code}</h1>
                  <span className="rounded-full bg-[var(--panel-strong)] px-3 py-1 text-sm font-semibold">{selected.window}</span>
                  <span className={selected.action === "BET" ? "rounded-full bg-[var(--accent)] px-3 py-1 text-sm font-bold text-white" : "rounded-full bg-stone-800 px-3 py-1 text-sm font-bold text-stone-50"}>
                    {selected.action}
                  </span>
                </div>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--muted)]">
                  {selected.edge_reason ?? "No edge reason recorded."}
                </p>
              </div>
              <div className="border-t border-[var(--line)] bg-[var(--panel)] p-6 lg:border-l lg:border-t-0">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <Metric label="Mode" value={labelize(selected.decision_mode)} />
                  <Metric label="Trace" value={pct(selected.trace_quality?.score)} />
                  <Metric label="Confidence" value={pct(selected.confidence)} />
                  <Metric label="Uncertainty" value={pct(selected.uncertainty)} />
                </div>
              </div>
            </div>
          </header>

          <section className="metric-grid">
            <MetricCard label="Decisions" value={summary.total} detail={`${summary.bets} bet · ${summary.skips} skip`} />
            <MetricCard label="Orders submitted" value={summary.submitted} detail={selected.dry_run ? "current run is dry-run" : "live routing enabled"} />
            <MetricCard label="Average confidence" value={pct(summary.avgConfidence)} detail="across local artifacts" />
            <MetricCard label="Average trace score" value={pct(summary.avgTrace)} detail="ledger quality" />
          </section>

          <HarnessPanel sessions={harnessSessions} />

          <section className="grid gap-4 xl:grid-cols-4">
            <ProbabilityBars label="Final model" probabilities={selected.final_probs} />
            <ProbabilityBars label="Market" probabilities={selected.market_probs} />
            <ProbabilityBars label="Bookmaker" probabilities={selected.bookmaker_probs} />
            <ProbabilityBars label="Sportmonks" probabilities={selected.sportmonks_probs} />
          </section>

          <section className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
            <Panel title="Trade Gate" eyebrow="Action">
              <div className="grid gap-3 sm:grid-cols-2">
                <Metric label="Best outcome" value={labelize(selected.best_outcome)} />
                <Metric label="Best edge" value={signed(selected.best_edge)} />
                <Metric label="Edge tier" value={labelize(selected.edge_tier)} />
                <Metric label="Edge type" value={labelize(selected.edge_type)} />
                <Metric label="Prediction submitted" value={String(Boolean(selected.prediction_submitted))} />
                <Metric label="Order submitted" value={String(Boolean(selected.order_submitted))} />
              </div>
              <div className="mt-5 rounded-2xl bg-[var(--panel)] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Order payload</p>
                <div className="mt-3 grid gap-2 text-sm sm:grid-cols-2">
                  <span>Team: <b>{String(orderPayload?.team_code ?? "n/a")}</b></span>
                  <span>Size: <b>{money(orderPayload?.usd_size)}</b></span>
                  <span>Limit: <b>{String(orderPayload?.limit_price ?? "n/a")}</b></span>
                  <span>TIF: <b>{String(orderPayload?.time_in_force_seconds ?? "n/a")}s</b></span>
                </div>
              </div>
            </Panel>

            <Panel title="Risk Stack" eyebrow="Blocking logic">
              <FlagList title="Blocking" flags={selected.blocking_risk_flags} tone="danger" />
              <FlagList title="All flags" flags={selected.risk_flags} />
            </Panel>
          </section>

          <section className="grid gap-5 xl:grid-cols-2">
            <Panel title="Data Completeness" eyebrow="Inputs">
              <div className="mb-4 flex items-center justify-between rounded-2xl bg-[var(--panel)] p-4">
                <span className="font-semibold">Completeness score</span>
                <span className="mono text-lg">{pct(selected.data_completeness?.score)}</span>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {Object.entries(selected.data_completeness?.checks ?? {}).map(([key, value]) => (
                  <span key={key} className="rounded-xl border border-[var(--line)] px-3 py-2 text-sm">
                    {labelize(key)}: <b>{value ? "yes" : "missing"}</b>
                  </span>
                ))}
              </div>
            </Panel>

            <Panel title="Model Composition" eyebrow="Deterministic component">
              <div className="grid gap-2 sm:grid-cols-2">
                {Object.entries(selected.deterministic_weights ?? {}).map(([key, value]) => (
                  <Metric key={key} label={labelize(key)} value={pct(value, 1)} />
                ))}
              </div>
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Probability steps</p>
                <div className="mt-2 space-y-2">
                  {(selected.probability_steps ?? []).map((step, index) => (
                    <div key={index} className="rounded-xl bg-[var(--panel)] px-3 py-2 text-sm">
                      <b>{labelize(step.name)}</b>
                      <span className="ml-2 text-[var(--muted)]">
                        {step.probabilities ? JSON.stringify(step.probabilities) : "no probabilities"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>
          </section>

          <section className="grid gap-5 xl:grid-cols-2">
            <Panel title="LLM Central / Council" eyebrow="Reasoning">
              <div className="grid gap-3 sm:grid-cols-2">
                <Metric label="Called" value={String(Boolean(selected.llm_central?.called))} />
                <Metric label="Provider" value={String(selected.llm_central?.provider ?? "n/a")} />
                <Metric label="Model" value={String(selected.llm_central?.model ?? "n/a")} />
                <Metric label="Recommendation" value={String(llmParsed?.recommendation ?? "n/a")} />
                <Metric label="Risk posture" value={String(llmParsed?.risk_posture ?? "n/a")} />
                <Metric label="Order authority" value={String(selected.llm_central?.order_authorization_allowed ?? "n/a")} />
              </div>
              <p className="mt-4 text-sm leading-6 text-[var(--muted)]">{String(llmParsed?.rationale ?? "No LLM central rationale recorded.")}</p>
            </Panel>

            <Panel title="Top Signals" eyebrow="Evidence">
              <div className="space-y-3">
                {(selected.top_signals ?? []).slice(0, 8).map((signal, index) => (
                  <div key={index} className="rounded-2xl bg-[var(--panel)] p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <b>{labelize(signal.name)}</b>
                      {signal.direction ? <span className="rounded-full bg-[var(--panel-strong)] px-2 py-0.5 text-xs">{String(signal.direction)}</span> : null}
                    </div>
                    <p className="mt-1 text-sm text-[var(--muted)]">{String(signal.reason ?? "No reason recorded.")}</p>
                  </div>
                ))}
              </div>
            </Panel>
          </section>

          <TraceTimeline records={run?.records} />

          <section className="grid gap-5 xl:grid-cols-2">
            <Panel title="Source Reconciliation" eyebrow="Disagreement">
              <JsonBlock value={selected.source_reconciliation} />
            </Panel>
            <Panel title="Raw Decision JSON" eyebrow="Debug">
              <JsonBlock value={selected} />
            </Panel>
          </section>
        </div>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-2xl bg-[var(--paper)]/65 p-3">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">{label}</p>
      <p className="mt-1 break-words text-sm font-semibold">{String(value ?? "n/a")}</p>
    </div>
  );
}

function MetricCard({ label, value, detail }: { label: string; value: unknown; detail: string }) {
  return (
    <section className="panel rounded-[1.25rem] p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">{label}</p>
      <p className="mt-2 text-2xl font-semibold tracking-[-0.03em]">{String(value)}</p>
      <p className="mt-1 text-sm text-[var(--muted)]">{detail}</p>
    </section>
  );
}

function Panel({ title, eyebrow, children }: { title: string; eyebrow: string; children: React.ReactNode }) {
  return (
    <section className="panel rounded-[1.35rem] p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">{eyebrow}</p>
      <h2 className="mt-1 text-xl font-semibold">{title}</h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function FlagList({ title, flags, tone = "neutral" }: { title: string; flags?: string[]; tone?: "neutral" | "danger" }) {
  return (
    <div className="mb-4">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">{title}</p>
      <div className="flex flex-wrap gap-2">
        {(flags?.length ? flags : ["none"]).map((flag) => (
          <span
            key={flag}
            className={tone === "danger" ? "rounded-full bg-red-100 px-3 py-1 text-sm font-semibold text-red-950" : "rounded-full bg-[var(--panel-strong)] px-3 py-1 text-sm"}
          >
            {labelize(flag)}
          </span>
        ))}
      </div>
    </div>
  );
}
