import { promises as fs } from "fs";
import path from "path";

export type Probabilities = Partial<Record<"home" | "draw" | "away", number>>;

export type Decision = {
  session_id?: string;
  fixture_code?: string;
  window?: string;
  teams?: string;
  final_probs?: Probabilities;
  market_probs?: Probabilities | null;
  bookmaker_probs?: Probabilities | null;
  sportmonks_probs?: Probabilities | null;
  halftime?: Record<string, unknown> | null;
  lineup?: Record<string, unknown> | null;
  data_completeness?: { score?: number; checks?: Record<string, boolean>; missing?: string[] };
  archetype?: Record<string, unknown> | null;
  source_reliability?: Record<string, unknown> | null;
  market_stale?: Record<string, unknown> | null;
  source_reconciliation?: Record<string, unknown> | null;
  source_contribution?: Record<string, Probabilities> | null;
  deterministic_weights?: Record<string, number> | null;
  probability_steps?: Array<Record<string, unknown>>;
  arbiter?: Record<string, unknown> | null;
  council_reconciliation?: Record<string, unknown> | null;
  counterfactuals?: Array<Record<string, unknown>>;
  llm_claims?: Record<string, unknown> | null;
  consensus_case?: string;
  best_outcome?: string;
  best_edge?: number;
  edge_tier?: string;
  edge_type?: string;
  edge_reason?: string;
  confidence?: number;
  uncertainty?: number;
  top_signals?: Array<Record<string, unknown>>;
  llm_analysis?: Record<string, unknown> | null;
  action?: "BET" | "SKIP" | string;
  risk_flags?: string[];
  blocking_risk_flags?: string[];
  prediction_submitted?: boolean;
  order_submitted?: boolean;
  dry_run?: boolean;
  ledger_submitted?: boolean;
  ledger_records?: number;
  ledger_dag_valid?: boolean;
  trace_quality?: Record<string, unknown>;
  order?: Record<string, unknown>;
  decision_mode?: string;
  llm_central?: Record<string, unknown> | null;
  _file?: string;
  _mtime?: string;
};

export type LedgerRecord = {
  record_id?: string;
  session_id?: string;
  behavior?: string;
  label?: string;
  timestamp?: string;
  parent_ids?: string[];
  reasoning_trace?: Record<string, unknown>;
  model_invocation?: Record<string, unknown>;
  output_payload?: unknown;
  parameters?: unknown;
  success?: boolean;
  [key: string]: unknown;
};

export type RunArtifact = {
  session_id?: string;
  records?: LedgerRecord[];
  _file?: string;
};

export type HarnessTrade = {
  trade_id?: string;
  agent?: string;
  fixture_code?: string;
  window?: string;
  slot?: string;
  outcome?: string;
  stake?: number;
  entry_price?: number;
  our_prob?: number;
  fair_prob?: number;
  edge_vs_fair?: number;
  ev_per_dollar?: number;
  market_source?: string;
  status?: string;
  pnl?: number;
  ts?: string;
};

export type HarnessAgentBook = {
  label?: string;
  start_bankroll?: number;
  bankroll?: number;
  trades?: HarnessTrade[];
};

export type HarnessSession = {
  name: string;
  created_at?: string;
  agents: Record<string, HarnessAgentBook>;
  windows: Array<Record<string, unknown>>;
  matches: Array<Record<string, unknown>>;
  summary?: Record<string, unknown> | null;
  resultsKnown: number;
  fixturesCount: number;
  profilesCount: number;
};

const repoRoot = path.resolve(process.cwd(), "..");
const storageRoot = path.join(repoRoot, "storage");

async function readJson<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

export async function listDecisions(): Promise<Decision[]> {
  const dir = path.join(storageRoot, "decisions");
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
  const files = entries.filter((entry) => entry.isFile() && entry.name.endsWith(".json"));
  const decisions: Array<Decision | null> = await Promise.all(
    files.map(async (entry) => {
      const filePath = path.join(dir, entry.name);
      const [raw, stat] = await Promise.all([readJson<Decision>(filePath), fs.stat(filePath)]);
      if (!raw) return null;
      return { ...raw, _file: entry.name, _mtime: stat.mtime.toISOString() };
    }),
  );

  const present = decisions.filter((decision): decision is Decision => decision !== null);
  return present
    .sort((a, b) => Date.parse(b._mtime ?? "") - Date.parse(a._mtime ?? ""));
}

export async function getDecision(fileName?: string): Promise<Decision | null> {
  const decisions = await listDecisions();
  if (!decisions.length) return null;
  if (!fileName) return decisions[0];
  return decisions.find((decision) => decision._file === fileName || decision.session_id === fileName) ?? decisions[0];
}

export async function getRun(sessionId?: string): Promise<RunArtifact | null> {
  if (!sessionId) return null;
  const dir = path.join(storageRoot, "runs");
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
  const candidates = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json") && entry.name.startsWith(sessionId))
    .map((entry) => entry.name);
  const exact = candidates.find((name) => name === `${sessionId}.json`) ?? candidates[0];
  if (!exact) return null;
  const run = await readJson<RunArtifact>(path.join(dir, exact));
  return run ? { ...run, _file: exact } : null;
}

export async function listHarnessSessions(): Promise<HarnessSession[]> {
  const root = path.join(storageRoot, "harness");
  const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  const sessions = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => readHarnessSession(entry.name)),
  );
  return sessions
    .filter((session): session is HarnessSession => Boolean(session))
    .sort((a, b) => String(b.created_at ?? b.name).localeCompare(String(a.created_at ?? a.name)));
}

async function readHarnessSession(name: string): Promise<HarnessSession | null> {
  const dir = path.join(storageRoot, "harness", name);
  const ledger = await readJson<{ created_at?: string; agents?: Record<string, HarnessAgentBook> }>(path.join(dir, "ledger.json"));
  if (!ledger?.agents) return null;
  const [windows, matches, summary, results, fixtures, profiles] = await Promise.all([
    readJsonl(path.join(dir, "windows.jsonl")),
    readJson<Array<Record<string, unknown>>>(path.join(dir, "matches.json")),
    readJson<Record<string, unknown>>(path.join(dir, "summary.json")),
    readJson<Record<string, { result_slot?: string } | string>>(path.join(dir, "results.json")),
    readJson<unknown[]>(path.join(dir, "fixtures.json")),
    readJson<Record<string, unknown>>(path.join(dir, "profiles.json")),
  ]);
  const resultsKnown = Object.values(results ?? {}).filter((value) => {
    const slot = typeof value === "string" ? value : value?.result_slot;
    return slot === "home" || slot === "draw" || slot === "away";
  }).length;
  return {
    name,
    created_at: ledger.created_at,
    agents: ledger.agents,
    windows,
    matches: Array.isArray(matches) ? matches : [],
    summary,
    resultsKnown,
    fixturesCount: Array.isArray(fixtures) ? fixtures.length : 0,
    profilesCount: profiles ? Object.keys(profiles).length : Object.keys(ledger.agents).length,
  };
}

async function readJsonl(filePath: string): Promise<Array<Record<string, unknown>>> {
  try {
    const text = await fs.readFile(filePath, "utf8");
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Record<string, unknown>);
  } catch {
    return [];
  }
}

export function summarize(decisions: Decision[]) {
  const bets = decisions.filter((decision) => decision.action === "BET").length;
  const submitted = decisions.filter((decision) => decision.order_submitted).length;
  const avgConfidence = average(decisions.map((decision) => decision.confidence));
  const avgTrace = average(decisions.map((decision) => Number(decision.trace_quality?.score)));
  const blockers = new Map<string, number>();
  for (const decision of decisions) {
    for (const flag of decision.blocking_risk_flags ?? []) {
      blockers.set(flag, (blockers.get(flag) ?? 0) + 1);
    }
  }
  return {
    total: decisions.length,
    bets,
    skips: decisions.length - bets,
    submitted,
    avgConfidence,
    avgTrace,
    topBlockers: [...blockers.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6),
  };
}

function average(values: Array<number | undefined>): number {
  const nums = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!nums.length) return 0;
  return nums.reduce((sum, value) => sum + value, 0) / nums.length;
}
