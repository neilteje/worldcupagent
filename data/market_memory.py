from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json


def _path(storage_dir: Path, fixture_code: str) -> Path:
    safe = str(fixture_code).replace('/', '_')
    return storage_dir / "price_history" / f"{safe}.jsonl"


def append_price_history(storage_dir: Path, fixture_code: str, window: str, raw_midpoints: dict, normalized_probs: dict | None) -> None:
    path = _path(storage_dir, fixture_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "fixture_code": fixture_code, "window": window, "raw_midpoints": raw_midpoints, "normalized_probs": normalized_probs}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_price_history(storage_dir: Path, fixture_code: str, limit: int = 20) -> list[dict]:
    path = _path(storage_dir, fixture_code)
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def previous_normalized_probs(storage_dir: Path, fixture_code: str, current_window: str | None = None) -> dict | None:
    rows = read_price_history(storage_dir, fixture_code, limit=10)
    if len(rows) < 2:
        return None
    # latest row is usually the current snapshot appended by this run, so inspect the previous one.
    for row in reversed(rows[:-1]):
        if current_window is None or row.get("window") == current_window:
            probs = row.get("normalized_probs")
            return probs if isinstance(probs, dict) else None
    return None
