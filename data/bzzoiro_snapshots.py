import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from agents.contracts import ProviderSnapshot

def generate_payload_hash(payload: dict | list | None) -> str:
    """Generate a stable SHA-256 hash of a JSON-serializable payload."""
    if payload is None:
        return hashlib.sha256(b"null").hexdigest()
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def create_bzzoiro_snapshot(
    resource_type: str,
    provider_id: str | None,
    payload: dict | list | None,
    success: bool,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ProviderSnapshot:
    """Create a ProviderSnapshot for BZZOIRO data with stable hashing."""
    now = datetime.now(timezone.utc)
    
    # Try to extract provider_updated_at from common BZZOIRO fields
    updated_at = None
    if isinstance(payload, dict):
        # Different endpoints use different timestamp fields
        updated_str = payload.get("updated_at") or payload.get("last_update")
        if updated_str:
            try:
                updated_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
            except ValueError:
                pass

    return ProviderSnapshot(
        provider="bzzoiro",
        resource_type=resource_type,
        provider_id=provider_id,
        retrieved_at=now,
        provider_updated_at=updated_at,
        payload_hash=generate_payload_hash(payload),
        payload=payload,
        success=success,
        stale=False, # Would be set by cache logic
        error_code=error_code,
        error_message=error_message,
        warnings=tuple() if success else (error_message or "Unknown error",)
    )
