from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid
import time

@dataclass
class LedgerRecord:
    session_id: str
    behavior: str
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "0.3"
    client_ts_utc: int = field(default_factory=lambda: int(time.time() * 1000))
    payload: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "session_id": self.session_id,
            "behavior": self.behavior,
            "parent_ids": self.parent_ids,
            "upstream_record_id": self.parent_ids,
            "timestamp": self.timestamp,
            "client_ts_utc": self.client_ts_utc,
        }
        d.update(self.payload)
        return d
