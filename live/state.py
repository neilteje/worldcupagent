"""
Persistent, resumable run state — the reason you can kill the loop anytime.

One JSON document at storage/live/state.json, written atomically
(tmp + os.replace) after every mutation. On startup the runner loads it and
skips everything already done, so a restart never reprocesses game 1.

Shape:
{
  "windows": {
    "19609127:PRE_MATCH": {
      "status": "done" | "dry_run" | "failed" | "missed" | "skipped",
      "attempts": 2,
      "ts": "...",                      # last touch
      "fixture_name": "...",
      "agents": { "monk": {...summary...}, ... }
    }
  },
  "settlements": {
    "19609127": {"resolved": true, "winner_slot": "home", "winner_code": "MEX",
                  "ts": "...", "attempts": 3}
  },
  "meta": {"started_at": "...", "last_heartbeat": "..."}
}
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "storage" / "live" / "state.json"

# A failed window is retried while it's still open, up to this many attempts.
MAX_ATTEMPTS = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LiveState:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self._doc: dict = {"windows": {}, "settlements": {}, "meta": {}}
        if self.path.exists():
            try:
                self._doc = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                # Never destroy a corrupt state file — move it aside and start clean.
                backup = self.path.with_suffix(".corrupt.json")
                self.path.rename(backup)
                print(f"[state] WARNING: state file unreadable ({exc!r}); "
                      f"moved to {backup.name} and starting fresh")
        self._doc.setdefault("windows", {})
        self._doc.setdefault("settlements", {})
        self._doc.setdefault("meta", {})
        self._doc["meta"].setdefault("started_at", _now())

    # ── persistence ────────────────────────────────────────────────────────

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._doc, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def heartbeat(self) -> None:
        self._doc["meta"]["last_heartbeat"] = _now()
        self.save()

    # ── windows ────────────────────────────────────────────────────────────

    @staticmethod
    def wkey(fixture_id: int | str, window: str) -> str:
        return f"{fixture_id}:{window}"

    def window(self, fixture_id: int | str, window: str) -> dict | None:
        return self._doc["windows"].get(self.wkey(fixture_id, window))

    def window_done(self, fixture_id: int | str, window: str) -> bool:
        w = self.window(fixture_id, window)
        if w and w.get("status") == "done":
            agents = w.get("agents") or {}
            if agents and all((a.get("ledger") or {}).get("dry_run") for a in agents.values()):
                return False
        return bool(w and w.get("status") in ("done", "missed", "skipped"))

    def window_attempts(self, fixture_id: int | str, window: str) -> int:
        w = self.window(fixture_id, window)
        return int(w.get("attempts", 0)) if w else 0

    def window_exhausted(self, fixture_id: int | str, window: str) -> bool:
        w = self.window(fixture_id, window)
        return bool(w and w.get("status") == "failed"
                    and int(w.get("attempts", 0)) >= MAX_ATTEMPTS)

    def mark_window(self, fixture_id: int | str, window: str, status: str,
                    fixture_name: str = "", agents: dict | None = None) -> None:
        key = self.wkey(fixture_id, window)
        cur = self._doc["windows"].get(key) or {}
        cur.update(
            status=status,
            attempts=int(cur.get("attempts", 0)) + (1 if status in ("done", "failed") else 0),
            ts=_now(),
            fixture_name=fixture_name or cur.get("fixture_name", ""),
        )
        if agents:
            cur.setdefault("agents", {}).update(agents)
        self._doc["windows"][key] = cur
        self.save()

    # ── settlements ────────────────────────────────────────────────────────

    def settlement(self, fixture_id: int | str) -> dict | None:
        return self._doc["settlements"].get(str(fixture_id))

    def settled(self, fixture_id: int | str) -> bool:
        s = self.settlement(fixture_id)
        return bool(s and s.get("resolved"))

    def mark_settlement(self, fixture_id: int | str, resolved: bool,
                        winner_slot: str | None = None,
                        winner_code: str | None = None) -> None:
        cur = self._doc["settlements"].get(str(fixture_id)) or {}
        cur.update(resolved=resolved, ts=_now(),
                   attempts=int(cur.get("attempts", 0)) + 1)
        if winner_slot:
            cur["winner_slot"] = winner_slot
        if winner_code:
            cur["winner_code"] = winner_code
        self._doc["settlements"][str(fixture_id)] = cur
        self.save()

    # ── introspection ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        windows = self._doc["windows"]
        by_status: dict[str, int] = {}
        for w in windows.values():
            by_status[w.get("status", "?")] = by_status.get(w.get("status", "?"), 0) + 1
        return {
            "windows_total": len(windows),
            "by_status": by_status,
            "settled": sum(1 for s in self._doc["settlements"].values() if s.get("resolved")),
            "meta": self._doc["meta"],
        }

    def doc(self) -> dict:
        return self._doc
