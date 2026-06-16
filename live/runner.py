"""
The end-to-end live loop: run once, leave it running until the final.

    python -m live run

Behaviour:
  - Discovers the WC2026 schedule (refreshed every few hours).
  - For every fixture, schedules a PRE_MATCH cycle (default: 45 min before
    kickoff, after lineups usually drop) and an HT cycle (kickoff + 46').
  - Before running a cycle it confirms with GET /v1/arena/matches/{id} that
    the window is actually open (server time, not local clock). If
    `current_window` is missing, it infers the window from the API's explicit
    open/lock timestamps.
  - Every completed (fixture, window) is recorded in storage/live/state.json —
    kill the process anytime; on restart it continues where it left off.
  - A settlement watcher resolves finished fixtures and logs realized results
    for the retrospective report.

Nothing in the loop is allowed to crash it: every cycle is wrapped, failures
are retried up to live.state.MAX_ATTEMPTS while the window is still open.
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone

from data import sportmonks
from live.arena_client import ArenaClient
from live.cycle import run_window_cycle
from live.roster import LiveAgent, load_roster
from live.state import LiveState
from live import metrics

# ── Timing knobs ────────────────────────────────────────────────────────────
PRE_LEAD_MIN = 45          # run PRE_MATCH at kickoff − 45'
PRE_GRACE_MIN = 5          # … but never start within 5' of kickoff blind
HT_OFFSET_MIN = 46         # run HT at kickoff + 46'
HT_DEADLINE_MIN = 70       # HT window has certainly closed by kickoff + 70'
SETTLE_AFTER_MIN = 130     # check settlement from kickoff + 130'
SCHEDULE_REFRESH_S = 6 * 3600
LOOP_SLEEP_MAX_S = 300


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_kickoff(s: str) -> datetime | None:
    """Sportmonks 'starting_at' is naive UTC ('2026-06-11 18:00:00')."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _millis(value) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def flatten_schedule(schedule: list[dict]) -> list[dict]:
    """Schedule entries nest stage → rounds → fixtures; some are direct."""
    fixtures: list[dict] = []
    for entry in schedule:
        for rnd in (entry.get("rounds") or []):
            fixtures.extend(rnd.get("fixtures") or [])
        if entry.get("id") and entry.get("participants"):
            fixtures.append(entry)
    seen, out = set(), []
    for fx in fixtures:
        fid = fx.get("id")
        if fid and fid not in seen and fx.get("starting_at"):
            seen.add(fid)
            out.append(fx)
    out.sort(key=lambda f: str(f.get("starting_at")))
    return out


class LiveRunner:
    def __init__(self, agents: list[LiveAgent] | None = None,
                 state: LiveState | None = None, dry_run: bool = False):
        self.agents = agents or load_roster()
        self.state = state or LiveState()
        self.dry_run = dry_run
        # Any key can read shared endpoints; use the first agent's.
        self.reader = ArenaClient(self.agents[0].api_key, "reader")
        self._fixtures: list[dict] = []
        self._schedule_ts: float = 0.0

    # ── schedule ──────────────────────────────────────────────────────────

    def refresh_schedule(self, force: bool = False) -> None:
        if not force and self._fixtures and time.time() - self._schedule_ts < SCHEDULE_REFRESH_S:
            return
        try:
            self._fixtures = flatten_schedule(sportmonks.get_season_schedule())
            self._schedule_ts = time.time()
            print(f"[runner] schedule: {len(self._fixtures)} fixtures")
        except Exception as exc:
            print(f"[runner] schedule refresh FAILED: {exc!r} "
                  f"(keeping {len(self._fixtures)} cached)")

    # ── window verification (server-side truth) ───────────────────────────

    def window_open(self, fixture_id: int, window: str) -> bool | None:
        """
        True/False from the matches endpoint; None when the endpoint is
        unavailable (caller falls back to the local-clock heuristic).

        The arena may return `current_window: null` even though explicit
        pre-match/HT open and lock timestamps are present. In that case, infer
        from `server_ts_utc` and the relevant timestamp pair.
        """
        try:
            m = self.reader.match(fixture_id)
        except Exception:
            return None
        if not m:
            return None
        cur = (m.get("current_window") or "").upper().replace("-", "_")
        if window == "PRE_MATCH":
            if cur:
                return cur in ("PRE_MATCH", "PREMATCH")
            server_ts = _millis(m.get("server_ts_utc"))
            lock_ts = _millis(m.get("pre_match_lock_utc") or m.get("prematch_lock_utc"))
            if server_ts is not None and lock_ts is not None:
                return server_ts < lock_ts
            return False

        if cur:
            return cur in ("HT", "HALF_TIME", "HALFTIME")
        server_ts = _millis(m.get("server_ts_utc"))
        open_ts = _millis(m.get("ht_open_utc") or m.get("half_time_open_utc"))
        lock_ts = _millis(m.get("ht_lock_utc") or m.get("half_time_lock_utc"))
        if server_ts is not None and open_ts is not None and lock_ts is not None:
            return open_ts <= server_ts < lock_ts
        return False

    # ── one pass over due work ────────────────────────────────────────────

    def pending_windows(self) -> list[dict]:
        """All not-yet-done windows with their trigger/expiry times."""
        now = _utcnow()
        items = []
        for fx in self._fixtures:
            fid = fx["id"]
            ko = _parse_kickoff(fx.get("starting_at"))
            if ko is None:
                continue
            items.append({
                "fixture_id": fid, "window": "PRE_MATCH", "kickoff": ko,
                "name": fx.get("name", "?"),
                "trigger": ko - timedelta(minutes=PRE_LEAD_MIN),
                "expiry": ko - timedelta(minutes=0),
            })
            items.append({
                "fixture_id": fid, "window": "HT", "kickoff": ko,
                "name": fx.get("name", "?"),
                "trigger": ko + timedelta(minutes=HT_OFFSET_MIN),
                "expiry": ko + timedelta(minutes=HT_DEADLINE_MIN),
            })
        out = []
        for it in items:
            if self.state.window_done(it["fixture_id"], it["window"]):
                continue
            if self.state.window_exhausted(it["fixture_id"], it["window"]):
                continue
            if it["expiry"] < now:
                # Window gone before we ever ran it (e.g. downtime) → missed.
                self.state.mark_window(it["fixture_id"], it["window"], "missed",
                                       fixture_name=it["name"])
                metrics.log_event("missed_window", fixture_id=it["fixture_id"],
                                  window=it["window"], fixture_name=it["name"])
                print(f"[runner] MISSED {it['name']} {it['window']} (expired)")
                continue
            out.append(it)
        out.sort(key=lambda i: i["trigger"])
        return out

    def run_due(self, pending: list[dict]) -> int:
        now = _utcnow()
        ran = 0
        for it in pending:
            if it["trigger"] > now:
                break  # sorted; nothing else is due
            fid, window, name = it["fixture_id"], it["window"], it["name"]

            # Confirm against the arena's clock/window state.
            is_open = self.window_open(fid, window)
            if is_open is False:
                if window == "HT":
                    # HT not open yet or already closed per server timestamps.
                    if _utcnow() > it["expiry"] - timedelta(minutes=2):
                        self.state.mark_window(fid, window, "skipped", fixture_name=name)
                        metrics.log_event("skipped_window", fixture_id=fid,
                                          window=window, reason="ht_window_not_open")
                        print(f"[runner] {name} HT window never opened — skipped")
                    continue
                # PRE_MATCH closed per server → too late.
                self.state.mark_window(fid, window, "missed", fixture_name=name)
                print(f"[runner] {name} PRE_MATCH already closed per server — missed")
                continue
            if is_open is None and window == "PRE_MATCH":
                # Endpoint unavailable: local-clock guard — don't start blind
                # inside the last PRE_GRACE_MIN minutes.
                if now > it["kickoff"] - timedelta(minutes=PRE_GRACE_MIN):
                    self.state.mark_window(fid, window, "missed", fixture_name=name)
                    print(f"[runner] {name} too close to kickoff — missed")
                    continue
            if is_open is None and window == "HT":
                # Can't verify HT is open; HT is disabled arena-side for now,
                # so don't fire blind — wait, and skip at expiry.
                if _utcnow() > it["expiry"] - timedelta(minutes=2):
                    self.state.mark_window(fid, window, "skipped", fixture_name=name)
                    metrics.log_event("skipped_window", fixture_id=fid,
                                      window=window, reason="ht_unverifiable")
                    print(f"[runner] {name} HT unverifiable — skipped")
                continue

            print(f"\n[runner] ═══ {name}  {window}  "
                  f"(fixture {fid}, attempt {self.state.window_attempts(fid, window) + 1}) ═══")
            prematch_note = None
            if window == "HT":
                pre = self.state.window(fid, "PRE_MATCH") or {}
                for a in (pre.get("agents") or {}).values():
                    if a.get("prediction"):
                        prematch_note = a["prediction"]
                        break
            try:
                result = run_window_cycle(fid, window, self.agents,
                                          prematch_note=prematch_note,
                                          dry_run=self.dry_run)
                agent_results = result.get("agents") or {}
                # If EVERY agent tail errored, the window didn't really run —
                # mark it failed so it's retried while the window is still open
                # (rather than silently marked done forever).
                all_failed = bool(agent_results) and all(
                    isinstance(r, dict) and r.get("error") for r in agent_results.values())
                if all_failed:
                    print(f"[runner] all {len(agent_results)} agents errored for "
                          f"{name} {window} — marking failed (will retry)")
                    metrics.log_event("error", fixture_id=fid, window=window,
                                      scope="all_agents_failed")
                    self.state.mark_window(fid, window, "failed", fixture_name=name,
                                           agents=agent_results)
                else:
                    self.state.mark_window(fid, window, "dry_run" if self.dry_run else "done",
                                           fixture_name=result.get("fixture_name", name),
                                           agents=agent_results)
                    ran += 1
            except Exception as exc:
                print(f"[runner] cycle FAILED for {name} {window}: {exc!r}")
                metrics.log_event("error", fixture_id=fid, window=window,
                                  error=repr(exc))
                self.state.mark_window(fid, window, "failed", fixture_name=name)
        return ran

    # ── settlement watcher ────────────────────────────────────────────────

    @staticmethod
    def _winner_from_settlement(s: dict) -> tuple[str | None, str | None]:
        """
        Tolerant parse: find the outcome whose resolved price is ~1.
        Returns (slot, team_code); (None, None) if unresolved.
        """
        def entries(obj):
            if isinstance(obj, dict):
                if "outcomes" in obj and isinstance(obj["outcomes"], dict):
                    for slot, o in obj["outcomes"].items():
                        yield slot, o
                for key in ("markets", "outcomes", "prices", "data"):
                    v = obj.get(key)
                    if isinstance(v, list):
                        for o in v:
                            yield None, o
            elif isinstance(obj, list):
                for o in obj:
                    yield None, o

        for slot, o in entries(s):
            if not isinstance(o, dict):
                continue
            price = (o.get("resolved_price") or o.get("settlement_price")
                     or o.get("price") or o.get("final_price"))
            try:
                price = float(price)
            except (TypeError, ValueError):
                continue
            if price >= 0.99:
                code = (o.get("team_code") or o.get("outcome") or o.get("code")
                        or o.get("market_slug") or slot)
                slot_guess = slot if slot in ("home", "draw", "away") else (
                    "draw" if str(code).lower() == "draw" else None)
                return slot_guess, str(code) if code else None
        return None, None

    def settle_pass(self) -> None:
        now = _utcnow()
        for fx in self._fixtures:
            fid = fx["id"]
            ko = _parse_kickoff(fx.get("starting_at"))
            if ko is None or now < ko + timedelta(minutes=SETTLE_AFTER_MIN):
                continue
            if self.state.settled(fid):
                continue
            # Only chase settlement for fixtures we actually engaged with.
            if not (self.state.window(fid, "PRE_MATCH") or self.state.window(fid, "HT")):
                continue
            s_rec = self.state.settlement(fid) or {}
            if s_rec.get("attempts", 0) >= 12:    # ~give up after a day of passes
                continue
            try:
                s = self.reader.settlement(fid)
            except Exception:
                s = None
            if not s:
                self.state.mark_settlement(fid, resolved=False)
                continue
            slot, code = self._winner_from_settlement(s)
            if code is None:
                self.state.mark_settlement(fid, resolved=False)
                continue
            self.state.mark_settlement(fid, resolved=True,
                                       winner_slot=slot, winner_code=code)
            metrics.log_event("settlement", fixture_id=fid,
                              fixture_name=fx.get("name", "?"),
                              winner_slot=slot, winner_code=code, raw=s)
            print(f"[runner] settled {fx.get('name', fid)}: winner={code}")
            # Snapshot each agent's wallet + per-fixture orders for the report.
            for agent in self.agents:
                try:
                    client = ArenaClient(agent.api_key, agent.name)
                    wallet = client.wallet()
                    orders = [o for o in client.orders()
                              if str(o.get("fixture_id") or o.get("fixture_code")) == str(fid)]
                    metrics.log_event("agent_settlement", fixture_id=fid,
                                      agent=agent.name, wallet=wallet, orders=orders)
                except Exception as exc:
                    print(f"  [{agent.name}] settlement snapshot failed: {exc!r}")

    # ── the loop ──────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        names = ", ".join(a.name for a in self.agents)
        print(f"[runner] starting live loop — agents: {names}"
              f"{'  (DRY RUN)' if self.dry_run else ''}")
        metrics.log_event("runner_start", agents=[a.name for a in self.agents],
                          dry_run=self.dry_run)
        self.refresh_schedule(force=True)
        while True:
            try:
                self.refresh_schedule()
                pending = self.pending_windows()
                self.run_due(pending)
                self.settle_pass()
                self.state.heartbeat()

                pending = self.pending_windows()
                now = _utcnow()
                if pending:
                    nxt = pending[0]
                    wait = max(5.0, min((nxt["trigger"] - now).total_seconds(),
                                        LOOP_SLEEP_MAX_S))
                    if (nxt["trigger"] - now).total_seconds() > 60:
                        print(f"[runner] next: {nxt['name']} {nxt['window']} at "
                              f"{nxt['trigger']:%Y-%m-%d %H:%M}Z  "
                              f"(sleeping {wait/60:.1f} min)")
                else:
                    wait = LOOP_SLEEP_MAX_S
                    print("[runner] no pending windows — idling")
                time.sleep(wait)
            except KeyboardInterrupt:
                print("\n[runner] interrupted — state saved; restart resumes here.")
                metrics.log_event("runner_stop", reason="keyboard_interrupt")
                return
            except Exception as exc:
                print(f"[runner] LOOP ERROR (continuing in 60s): {exc!r}")
                metrics.log_event("error", scope="loop", error=repr(exc))
                time.sleep(60)
