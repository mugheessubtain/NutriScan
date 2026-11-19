"""Records every cleaning rule's effect (rows before -> after) in reports/tables/cleaning_log.json.

Each training step calls `record(...)` as it applies a rule; re-running a step overwrites its own entries, so the
file always describes the latest run. The web app's Data cleaning page reads this file.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config

LOG_PATH = config.TABLES / "cleaning_log.json"

ORDER = ["prepare", "step1", "step2", "step3", "step3b"]


def _load() -> dict:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"entries": {}}


def record(step: str, rule_id: str, *, dataset: str, issue: str, fix: str, why: str,
           before: int, after: int, unit: str = "rows", kind: str = "removed") -> None:
    """kind: 'removed' (rows dropped), 'fixed' (values repaired / set to missing), 'kept' (counted, not dropped)."""
    log = _load()
    log["entries"][f"{step}:{rule_id}"] = {
        "step": step, "id": rule_id, "dataset": dataset, "issue": issue, "fix": fix, "why": why,
        "before": int(before), "after": int(after), "unit": unit, "kind": kind,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    log["synthetic"] = not config.OFF_META.exists()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")


def entries() -> list[dict]:
    """All entries in pipeline order (prepare -> step3b), then in the order they were recorded."""
    log = _load()
    items = list(log.get("entries", {}).values())
    return sorted(items, key=lambda e: ORDER.index(e["step"]) if e["step"] in ORDER else 99)


def is_synthetic() -> bool | None:
    return _load().get("synthetic")
