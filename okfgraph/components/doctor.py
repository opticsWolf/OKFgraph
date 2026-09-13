"""Doctor: scored graph health + safe `--fix` repairs.

Generalizes ``broken-links``/``repair-links`` (kept as-is) to a full scan:
``broken_link`` (error), ``orphan`` / ``stale`` / ``duplicate_title`` /
``missing_description`` (warnings), and info-level ``hub_concentration``
which never affects the score. ``--fix`` applies only unambiguous repairs —
ISO timestamp normalization and broken-link re-pointing (exact id or unique
wikilink name) — and never touches ``reviewed: true`` concepts, while still
reporting findings on them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Deduction per finding: (points, cap). Score = max(0, 100 - deductions).
DEDUCTIONS = {
    "broken_link": (5, 40),
    "orphan_hash": (5, 30),
    "orphan": (2, 20),
    "duplicate_title": (3, 15),
    "stale": (1, 10),
    "missing_description": (1, 15),
}

_TIMESTAMP_FORMATS = (
    "%d.%m.%Y", "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S",
    "%Y/%m/%d", "%Y/%m/%d %H:%M:%S",
)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse ISO plus a few common human variants; None when unparseable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def is_reviewed(extra: Any) -> bool:
    """True when the concept's extra MAP carries ``reviewed: true``."""
    if not isinstance(extra, dict):
        return False
    return str(extra.get("reviewed", "")).strip().lower() in ("true", "1", "yes")


class DoctorManager:
    """Health scan + safe repairs over the live graph."""

    def __init__(self, conn, import_mgr):
        self.conn = conn
        self.import_mgr = import_mgr

    # -- data ---------------------------------------------------------

    def _concepts(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "MATCH (c:Concept) "
            "RETURN c.id, c.title, c.description, c.timestamp, c.extra"
        ).rows_as_dict().get_all()
        return [
            {
                "id": r["c.id"],
                "title": r.get("c.title"),
                "description": r.get("c.description"),
                "timestamp": r.get("c.timestamp"),
                "extra": r.get("c.extra") or {},
                "reviewed": is_reviewed(r.get("c.extra") or {}),
            }
            for r in rows
        ]

    def _degrees(self) -> Dict[str, Tuple[int, int]]:
        """Concept id -> (out_degree, in_degree) over LINKS_TO."""
        deg: Dict[str, List[int]] = {}
        for r in self.conn.execute(
            "MATCH (a:Concept)-[e:LINKS_TO]->(b:Concept) "
            "RETURN a.id AS src, b.id AS dst"
        ).rows_as_dict().get_all():
            deg.setdefault(r["src"], [0, 0])[0] += 1
            deg.setdefault(r["dst"], [0, 0])[1] += 1
        return {k: (v[0], v[1]) for k, v in deg.items()}

    # -- diagnose ------------------------------------------------------

    def diagnose(self, stale_days: int = 365) -> Dict[str, Any]:
        """Run the full health scan. Deterministic finding order (rule, path)."""
        concepts = self._concepts()
        degrees = self._degrees()
        now = datetime.now(timezone.utc)

        findings: List[Dict[str, Any]] = []

        def _add(path: str, severity: str, rule: str, message: str) -> None:
            findings.append({
                "path": path, "severity": severity,
                "rule": rule, "message": message,
            })

        for bl in self.conn.execute(
            "MATCH (bl:BrokenLink) "
            "RETURN bl.source_id AS source, bl.target_id AS target"
        ).rows_as_dict().get_all():
            _add(bl["source"], "error", "broken_link",
                 f"links to missing concept '{bl['target']}'")

        # Orphan hashes (0.2.15): FileHash rows whose concept_id matches no
        # Concept and no DeletedConcept — signature of a crashed import that
        # committed hashes but not concepts. Repaired by --fix.
        live_ids = {c["id"] for c in concepts}
        try:
            tombstoned_ids = {
                r["oid"]
                for r in self.conn.execute(
                    "MATCH (d:DeletedConcept) RETURN d.original_id AS oid"
                ).rows_as_dict().get_all()
            }
        except Exception:
            tombstoned_ids = set()
        for fh in self.conn.execute(
            "MATCH (f:FileHash) "
            "RETURN f.path AS path, f.concept_id AS cid"
        ).rows_as_dict().get_all():
            if fh["cid"] not in live_ids and fh["cid"] not in tombstoned_ids:
                _add(fh["path"], "error", "orphan_hash",
                     f"hash tracked for missing concept '{fh['cid']}' "
                     f"(crashed import?) — cleared by --fix")

        titles: Dict[str, List[str]] = {}
        for c in concepts:
            if c["title"]:
                titles.setdefault(c["title"], []).append(c["id"])
        for title, ids in titles.items():
            if len(ids) > 1:
                for cid in sorted(ids):
                    others = sorted(i for i in ids if i != cid)
                    _add(cid, "warn", "duplicate_title",
                         f"title '{title}' also used by {', '.join(others)}")

        for c in concepts:
            cid = c["id"]
            if cid.endswith("index") or cid.endswith("log"):
                continue
            out_d, in_d = degrees.get(cid, (0, 0))
            if out_d == 0 and in_d == 0:
                _add(cid, "warn", "orphan", "no incoming or outgoing links")
            if not (c["description"] or "").strip():
                _add(cid, "warn", "missing_description", "empty description")
            ts = parse_timestamp(c["timestamp"])
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = (now - ts).total_seconds() / 86400
                if age_days > stale_days:
                    _add(cid, "warn", "stale",
                         f"timestamp {c['timestamp']} is {int(age_days)} days old")

        indeg = sorted(
            ((cid, d[1]) for cid, d in degrees.items() if d[1] > 0),
            key=lambda kv: (-kv[1], kv[0]),
        )[:3]
        info = [
            {"rule": "hub_concentration",
             "message": f"'{cid}' has {n} incoming link(s)"}
            for cid, n in indeg
        ]

        findings.sort(key=lambda f: (f["rule"], f["path"]))
        totals: Dict[str, int] = {}
        for f in findings:
            totals[f["rule"]] = totals.get(f["rule"], 0) + 1
        score = 100
        for rule, count in totals.items():
            points, cap = DEDUCTIONS.get(rule, (0, 0))
            score -= min(points * count, cap)
        score = max(0, score)

        return {
            "score": score,
            "concepts": len(concepts),
            "findings": findings,
            "summary": totals,
            "info": info,
        }

    # -- fix ------------------------------------------------------------

    def fix(self) -> Dict[str, Any]:
        """Apply safe repairs; ``reviewed: true`` concepts are never modified."""
        concepts = self._concepts()
        reviewed = {c["id"] for c in concepts if c["reviewed"]}
        skipped: List[str] = sorted(reviewed)

        normalized = 0
        for c in concepts:
            if c["id"] in reviewed:
                continue
            ts = parse_timestamp(c["timestamp"])
            if ts is None or c["timestamp"] is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # The column is TIMESTAMP-typed: compare instants, write datetimes.
            current = c["timestamp"]
            current_dt = current if isinstance(current, datetime) else ts
            if current_dt.tzinfo is None:
                current_dt = current_dt.replace(tzinfo=timezone.utc)
            if current_dt.astimezone(timezone.utc) != ts.astimezone(timezone.utc) \
                    or not isinstance(current, datetime):
                self.conn.execute(
                    "MATCH (c:Concept {id: $id}) SET c.timestamp = $ts",
                    {"id": c["id"], "ts": ts},
                )
                normalized += 1

        repaired = self.import_mgr.repair_links(skip_sources=reviewed)

        # Orphan-hash repair (0.2.15): drop wedge rows so the next import
        # re-walks those directories. No concepts are touched, so the
        # reviewed-concepts rule doesn't apply.
        cleared = self.import_mgr.delta_mgr._clear_orphan_hashes()

        return {
            "repaired_links": repaired,
            "normalized_timestamps": normalized,
            "skipped_reviewed": skipped,
            "cleared_orphan_hashes": cleared["cleared_orphan_hashes"],
            "cleared_dir_hashes": cleared["cleared_dir_hashes"],
        }
