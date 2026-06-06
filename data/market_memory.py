from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json


def append_price_history(storage_dir: Path | str, fixture_code: str, window: str, raw_midpoints: dict, normalized_probs: dict | None) -> Path:
    path = Path(storage_dir) / "price_history" / f"{fixture_code}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "fixture_code": fixture_code, "window": window, "raw_midpoints": raw_midpoints, "normalized_probs": normalized_probs}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return path
