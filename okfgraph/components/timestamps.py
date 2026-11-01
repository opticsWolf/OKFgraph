"""Concept times (0.11.0): when a concept entered the graph, when its
content last changed, and the one time search uses.

- ``created_at``: first import into this graph; kept across re-imports.
- ``updated_at``: the last import that changed the content (body or
  frontmatter, compared by ``content_sig``). Not the file mtime: a
  checkout or copy touches every file without changing any.
- ``effective_at``: the authored date when the frontmatter has one
  (``timestamp``, else ``updated`` / ``date`` / ``created``, the
  Obsidian conventions), otherwise ``updated_at``. ``since`` / ``until``
  and the recency boost read this one.

All values are naive UTC, like every Ladybug TIMESTAMP column here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

#: Frontmatter keys that date a concept, most specific first.
AUTHORED_KEYS = ("timestamp", "updated", "date", "created")


def utc_naive(value: Any) -> Optional[datetime]:
    """``value`` (datetime, date or ISO string) as naive UTC, else None."""
    if isinstance(value, str):
        text = value.strip().strip('"')
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return None


def authored_time(data: Dict[str, Any]) -> Optional[datetime]:
    """The first parseable authored date in a concept's fields."""
    for key in AUTHORED_KEYS:
        ts = utc_naive(data.get(key))
        if ts is not None:
            return ts
    return None


def content_sig(data: Dict[str, Any], body: str) -> str:
    """Hash of body + frontmatter; equal sigs mean nothing changed."""
    meta = {k: v for k, v in data.items() if k not in ("body", "embedding")}
    blob = json.dumps(meta, sort_keys=True, default=str) + "\0" + (body or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extra_value(value: Any) -> str:
    """A frontmatter value as stored in the ``extra`` MAP.

    Strings stay raw; dates (YAML ``date: 2024-01-02``) become ISO
    strings; everything else is JSON.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return json.dumps(value, default=str)


def stamp(data: Dict[str, Any], body: str, prev: Optional[Dict[str, Any]],
          now: datetime) -> Dict[str, Any]:
    """``created_at`` / ``updated_at`` / ``effective_at`` / ``content_sig``
    for a concept about to be written; ``prev`` is its stored row (or
    None when new)."""
    sig = content_sig(data, body)
    prev = prev or {}
    created = utc_naive(prev.get("created_at")) or now
    updated = utc_naive(prev.get("updated_at"))
    if updated is None or prev.get("content_sig") != sig:
        updated = now
    return {
        "created_at": created,
        "updated_at": updated,
        "effective_at": authored_time(data) or updated,
        "content_sig": sig,
    }


def recency_decay(effective: Any, now: datetime, half_life_days: float) -> float:
    """``0.5 ** (age / half_life)``: 1.0 now, 0.5 one half-life ago, 0
    without a time. A future date counts as now."""
    ts = utc_naive(effective)
    if ts is None:
        return 0.0
    age = max((now - ts).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age / half_life_days)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
