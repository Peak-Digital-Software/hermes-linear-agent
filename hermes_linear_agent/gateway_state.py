"""Read the Hermes gateway lifecycle state through the public status API."""

from __future__ import annotations

READY_STATES = frozenset({"running", "degraded"})


def gateway_state() -> str | None:
    """`gateway_state` from the runtime status record.

    None when the import fails, there is no record, or the record is stale.
    Never raises.
    """
    try:
        from gateway.status import read_runtime_status, runtime_status_is_stale

        record = read_runtime_status()
        if not record or runtime_status_is_stale(record):
            return None
        state = record.get("gateway_state")
        return state if isinstance(state, str) and state else None
    except Exception:
        return None


def gateway_draining() -> bool:
    return gateway_state() == "draining"


def gateway_ready() -> bool:
    return gateway_state() in READY_STATES
