import type { HarnessSession, HarnessTrade } from "@/lib/artifacts";
import { compactDate, money, pct, signed } from "@/lib/format";

export function HarnessPanel({ sessions }: { sessions: HarnessSession[] }) {
  if (!sessions.length) {
    return (
      <section className="panel rounded-[1.35rem] p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Paper Trading</p>
        <h2 className="mt-1 text-xl font-semibold">No harness sessions</h2>
        <p className="mt-2 text-sm text-[var(--muted)]">Run `python -m harness init`, then `python -m harness now` or `python -m harness run`.</p>
      </section>
    );
  }

  const session = sessions[0];
  const trades = Object.values(session.agents).flatMap((book) => book.trades ?? []);
  const open = trades.filter((trade) => trade.status === "open").length;
  const settled = trades.filter((trade) => trade.status === "won" || trade.status === "lost").length;
  const backtest = session.summary?.backtest as Record<string, unknown> | undefined;
  const council = backtest?.council as Record<string, unknown> | undefined;
  const componentReport = session.summary?.component_report as Record<string, unknown> | undefined;

  return (
    <section className="panel rounded-[1.35rem] p-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">Paper Trading</p>
          <h2 className="mt-1 text-xl font-semibold">Harness session: {session.name}</h2>
        </div>
        <p className="text-sm text-[var(--muted)]">{compactDate(session.created_at)} · {session.fixturesCount} fixtures · {session.profilesCount} profiles</p>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-4">
        <HarnessMetric label="Trades" value={trades.length} detail={`${open} open · ${settled} settled`} />
        <HarnessMetric label="Windows run" value={session.windows.length} detail="from windows.jsonl" />
        <HarnessMetric label="Results known" value={session.resultsKnown} detail="from results.json" />
        <HarnessMetric label="Report" value={session.summary ? "ready" : "pending"} detail="run harness report" />
      </div>

      {backtest ? (
        <div className="mt-4 rounded-2xl border border-[var(--line)] bg-[var(--paper)]/72 p-4">
          <div className="grid gap-3 sm:grid-cols-5">
            <HarnessMetric label="Dataset" value={String(backtest.dataset ?? "n/a")} detail={String(backtest.historical_source ?? "historical replay")} />
            <HarnessMetric label="Engine" value={String(backtest.engine ?? "n/a")} detail="prediction mode" />
            <HarnessMetric label="Matches" value={String(backtest.matches ?? session.matches.length)} detail={`${String(backtest.tradable_markets ?? "n/a")} tradable markets`} />
            <HarnessMetric label="Lineups" value={String(backtest.lineup_rows ?? "n/a")} detail="real lineup rows" />
            <HarnessMetric label="Files" value="matches" detail="matches.json + matches.csv" />
          </div>
          {council ? (
            <div className="mt-3 grid gap-3 sm:grid-cols-4">
              <HarnessMetric label="Council OK" value={String(council.calls_ok ?? 0)} detail="successful council forecasts" />
              <HarnessMetric label="Fallbacks" value={String(council.fallbacks ?? 0)} detail="used deterministic fallback" />
              <HarnessMetric label="Roles" value={Array.isArray(council.roles) ? council.roles.length : 0} detail="pulse scout analyst devil judge" />
              <HarnessMetric label="Leakage policy" value="locked" detail={String(council.historical_leakage_policy ?? "historical inputs only")} />
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="mt-5 grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
        <div className="rounded-2xl border border-[var(--line)] bg-[var(--paper)]/72 p-4">
          <h3 className="font-semibold">Agent bankrolls</h3>
          <div className="mt-3 space-y-3">
            {Object.entries(session.agents).map(([name, book]) => {
              const start = Number(book.start_bankroll ?? 0);
              const bankroll = Number(book.bankroll ?? 0);
              const delta = start ? (bankroll - start) / start : 0;
              return (
                <div key={name} className="rounded-xl bg-[var(--panel)] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="font-semibold">{name}</p>
                      <p className="text-xs text-[var(--muted)]">{book.label ?? "profile"}</p>
                    </div>
                    <div className="text-right">
                      <p className="mono font-semibold">{money(bankroll)}</p>
                      <p className={delta >= 0 ? "text-xs text-emerald-800" : "text-xs text-red-800"}>{signed(delta)}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--paper)]/72">
          <div className="border-b border-[var(--line)] p-4">
            <h3 className="font-semibold">Paper trades</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">Open trades settle after `results.json` is filled and `python -m harness settle` runs.</p>
          </div>
          <div className="overflow-auto">
            <table className="w-full min-w-[48rem] text-left text-sm">
              <thead className="bg-[var(--panel)] text-xs uppercase tracking-[0.12em] text-[var(--muted)]">
                <tr>
                  <th className="px-4 py-3">Agent</th>
                  <th className="px-4 py-3">Fixture</th>
                  <th className="px-4 py-3">Pick</th>
                  <th className="px-4 py-3">Stake</th>
                  <th className="px-4 py-3">Entry</th>
                  <th className="px-4 py-3">Edge</th>
                  <th className="px-4 py-3">EV</th>
                  <th className="px-4 py-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {trades.length ? trades.map((trade) => <TradeRow key={trade.trade_id} trade={trade} />) : (
                  <tr>
                    <td className="px-4 py-6 text-[var(--muted)]" colSpan={8}>No paper trades yet. The policy bars may not have cleared.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {componentReport ? <ComponentReport report={componentReport} /> : null}

      {session.matches.length ? (
        <div className="mt-5 overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--paper)]/72">
          <div className="border-b border-[var(--line)] p-4">
            <h3 className="font-semibold">Backtest match replay</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">Each row shows the historical inputs, predicted distribution, true result, and trade count.</p>
          </div>
          <div className="overflow-auto">
            <table className="w-full min-w-[64rem] text-left text-sm">
              <thead className="bg-[var(--panel)] text-xs uppercase tracking-[0.12em] text-[var(--muted)]">
                <tr>
                  <th className="px-4 py-3">Match</th>
                  <th className="px-4 py-3">Stage</th>
                  <th className="px-4 py-3">Prediction</th>
                  <th className="px-4 py-3">Market</th>
                  <th className="px-4 py-3">Result</th>
                  <th className="px-4 py-3">Trades</th>
                  <th className="px-4 py-3">Inputs</th>
                </tr>
              </thead>
              <tbody>
                {session.matches.slice(0, 30).map((match) => <MatchRow key={String(match.fixture_code)} match={match} />)}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function HarnessMetric({ label, value, detail }: { label: string; value: unknown; detail: string }) {
  return (
    <div className="rounded-2xl bg-[var(--paper)]/72 p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">{label}</p>
      <p className="mt-1 text-xl font-semibold">{String(value)}</p>
      <p className="mt-1 text-sm text-[var(--muted)]">{detail}</p>
    </div>
  );
}

function TradeRow({ trade }: { trade: HarnessTrade }) {
  return (
    <tr className="border-t border-[var(--line)]">
      <td className="px-4 py-3 font-semibold">{trade.agent}</td>
      <td className="px-4 py-3">{trade.fixture_code} <span className="text-[var(--muted)]">{trade.window}</span></td>
      <td className="px-4 py-3">{trade.outcome} <span className="text-[var(--muted)]">({trade.slot})</span></td>
      <td className="px-4 py-3 mono">{money(trade.stake)}</td>
      <td className="px-4 py-3 mono">{pct(trade.entry_price, 1)}</td>
      <td className="px-4 py-3 mono">{signed(trade.edge_vs_fair)}</td>
      <td className="px-4 py-3 mono">{pct(trade.ev_per_dollar, 1)}</td>
      <td className="px-4 py-3">
        <span className="rounded-full bg-[var(--panel-strong)] px-2.5 py-1 text-xs font-semibold">{trade.status}</span>
      </td>
    </tr>
  );
}

function MatchRow({ match }: { match: Record<string, unknown> }) {
  const prediction = match.prediction as { probabilities?: Record<string, number> } | undefined;
  const inputs = match.inputs as { market?: Record<string, number>; odds_quality?: { flags?: string[] } } | undefined;
  const homeCode = String(match.home_code ?? "home");
  const awayCode = String(match.away_code ?? "away");
  const probs = prediction?.probabilities ?? {};
  const market = inputs?.market ?? {};
  const trades = Array.isArray(match.trades) ? match.trades.length : 0;
  const flags = inputs?.odds_quality?.flags ?? [];
  const modelDetail = match.model_detail as { council?: Record<string, unknown> } | undefined;
  const council = modelDetail?.council;
  const profileReports = (modelDetail as { profile_reports?: Record<string, Record<string, unknown>> } | undefined)?.profile_reports ?? {};
  const modelSignals = (match.model_signals as Array<Record<string, unknown>> | undefined) ?? [];
  const skipSummary = Object.entries(profileReports)
    .map(([profile, report]) => `${profile}: ${Array.isArray(report.picked_codes) && report.picked_codes.length ? `picked ${report.picked_codes.join("/")}` : (report.skip_reasons as string[] | undefined)?.join(", ") || "no action"}`)
    .join(" · ");

  return (
    <tr className="border-t border-[var(--line)] align-top">
      <td className="px-4 py-3">
        <p className="font-semibold">{String(match.teams ?? match.fixture_code)}</p>
        <p className="text-xs text-[var(--muted)]">{String(match.fixture_code)} · {compactDate(String(match.kickoff_utc ?? ""))}</p>
      </td>
      <td className="px-4 py-3">{String(match.stage ?? "n/a")}</td>
      <td className="px-4 py-3 mono">
        {homeCode} {pct(probs[homeCode], 0)} · D {pct(probs.draw, 0)} · {awayCode} {pct(probs[awayCode], 0)}
      </td>
      <td className="px-4 py-3 mono">
        H {pct(market.home, 0)} · D {pct(market.draw, 0)} · A {pct(market.away, 0)}
      </td>
      <td className="px-4 py-3">
        <span className="rounded-full bg-[var(--panel-strong)] px-2.5 py-1 text-xs font-semibold">{String(match.result)}</span>
      </td>
      <td className="px-4 py-3">{trades}</td>
      <td className="px-4 py-3 text-xs text-[var(--muted)]">
        <p>{flags.length ? flags.join(", ") : String(match.source ?? "historical")}</p>
        {council ? (
          <p className={council.ok ? "mt-1 text-emerald-800" : "mt-1 text-red-800"}>
            council {council.ok ? "ok" : "fallback"}{council.market_alignment ? ` · ${String(council.market_alignment)}` : ""}
          </p>
        ) : null}
        {skipSummary ? <p className="mt-1">{skipSummary}</p> : null}
        {modelSignals.length ? (
          <div className="mt-2 flex max-w-[24rem] flex-wrap gap-1">
            {modelSignals.slice(0, 7).map((signal) => (
              <span
                key={String(signal.model)}
                className={signal.hit ? "rounded-full bg-emerald-100 px-2 py-0.5 text-[0.68rem] font-semibold text-emerald-950" : "rounded-full bg-red-100 px-2 py-0.5 text-[0.68rem] font-semibold text-red-950"}
                title={`${String(signal.model)} picked ${String(signal.pick_slot)} at ${pct(Number(signal.pick_probability), 1)}`}
              >
                {String(signal.model).replace("step:", "")}: {String(signal.pick_slot)}
              </span>
            ))}
          </div>
        ) : null}
      </td>
    </tr>
  );
}

function ComponentReport({ report }: { report: Record<string, unknown> }) {
  const accuracyByArchetype = report.accuracy_by_archetype as Record<string, { n?: number; accuracy?: number }> | undefined;
  const accuracyByPickSlot = report.accuracy_by_pick_slot as Record<string, { n?: number; accuracy?: number }> | undefined;
  const avgSourceWeights = report.avg_source_weights as Record<string, number> | undefined;
  const stepMovement = report.probability_step_movement as Record<string, number> | undefined;
  const drawModel = report.draw_model as { avg_abs_delta?: number; reasons?: Record<string, number> } | undefined;
  const consensusCases = report.consensus_cases as Record<string, number> | undefined;
  const profileGateSummary = report.profile_gate_summary as Record<string, Record<string, number>> | undefined;
  const modelWinnerAccuracy = report.model_winner_accuracy as Record<string, { n?: number; wins?: number; accuracy?: number }> | undefined;

  return (
    <div className="mt-5 grid gap-4 xl:grid-cols-2">
      <div className="rounded-2xl border border-[var(--line)] bg-[var(--paper)]/72 p-4">
        <h3 className="font-semibold">Deterministic component report</h3>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <MiniTable title="Accuracy by archetype" rows={Object.entries(accuracyByArchetype ?? {}).map(([k, v]) => [k, `${pct(v.accuracy, 1)} (${v.n})`])} />
          <MiniTable title="Accuracy by pick" rows={Object.entries(accuracyByPickSlot ?? {}).map(([k, v]) => [k, `${pct(v.accuracy, 1)} (${v.n})`])} />
          <MiniTable title="Model winner %" rows={Object.entries(modelWinnerAccuracy ?? {}).map(([k, v]) => [k, `${pct(v.accuracy, 1)} (${v.wins}/${v.n})`])} />
          <MiniTable title="Avg source weights" rows={Object.entries(avgSourceWeights ?? {}).map(([k, v]) => [k, pct(v, 1)])} />
          <MiniTable title="Step movement" rows={Object.entries(stepMovement ?? {}).map(([k, v]) => [k, signed(v)])} />
        </div>
      </div>
      <div className="rounded-2xl border border-[var(--line)] bg-[var(--paper)]/72 p-4">
        <h3 className="font-semibold">Trading gates</h3>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <MiniTable title="Consensus cases" rows={Object.entries(consensusCases ?? {}).map(([k, v]) => [k, String(v)])} />
          <MiniTable
            title="Draw model"
            rows={[
              ["avg abs delta", signed(drawModel?.avg_abs_delta)] as [string, string],
              ...Object.entries(drawModel?.reasons ?? {}).map(([k, v]) => [k, String(v)] as [string, string]),
            ]}
          />
          {Object.entries(profileGateSummary ?? {}).map(([profile, values]) => (
            <MiniTable key={profile} title={`${profile} gates`} rows={Object.entries(values).map(([k, v]) => [k.replaceAll("_", " "), String(v)])} />
          ))}
        </div>
      </div>
    </div>
  );
}

function MiniTable({ title, rows }: { title: string; rows: Array<[string, string]> }) {
  return (
    <div className="rounded-xl bg-[var(--panel)] p-3">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">{title}</p>
      <div className="mt-2 space-y-1 text-sm">
        {rows.length ? rows.slice(0, 8).map(([label, value]) => (
          <div key={`${title}-${label}`} className="flex justify-between gap-3">
            <span className="truncate text-[var(--muted)]">{label}</span>
            <span className="mono font-semibold">{value}</span>
          </div>
        )) : <span className="text-[var(--muted)]">No data</span>}
      </div>
    </div>
  );
}
