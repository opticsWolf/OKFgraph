#!/usr/bin/env python3
"""Compare Python and Rust v10 DB dumps semantically.

Internal Ladybug IDs and wall-clock graph times are intentionally stripped:
they are provenance, not graph behavior. Floats are rounded to the scenario
tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

VOLATILE = {"created_at", "updated_at", "effective_at"}
INTERNAL = {"_ID", "_LABEL", "_SRC", "_DST"}


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {
            key: clean(item)
            for key, item in value.items()
            if key not in VOLATILE and key not in INTERNAL
        }
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def load(path: Path) -> dict[str, list[Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[Any]] = {}
    for table, rows in raw.items():
        cleaned = []
        for row in rows:
            if isinstance(row, dict) and len(row) == 1:
                key = next(iter(row))
                if key in ("n", "r") and isinstance(row[key], dict):
                    row = row[key]
            if table == "BrokenLink" and isinstance(row, dict):
                row = {k: v for k, v in row.items() if k != "timestamp"}
            if table == "Meta" and isinstance(row, dict):
                if row.get("key") == "detached_at":
                    row = {**row, "value": 0}
            if table == "SourceRoot" and isinstance(row, dict):
                row = {k: v for k, v in row.items() if k not in {"path", "last_seen"}}
            if table == "DeletedConcept" and isinstance(row, dict):
                row = {
                    k: v
                    for k, v in row.items()
                    if k not in {"id", "deleted_at", "snapshot"}
                }
            cleaned.append(clean(row))
        out[table] = sorted(
            cleaned, key=lambda value: json.dumps(value, sort_keys=True, ensure_ascii=True)
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--rust", required=True)
    ap.add_argument("--ignore-key", action="append", default=[])
    args = ap.parse_args()

    left = load(Path(args.python))
    right = load(Path(args.rust))
    for key in args.ignore_key:
        for tables in (left, right):
            for rows in tables.values():
                for row in rows:
                    if isinstance(row, dict):
                        row.pop(key, None)
    diffs: list[str] = []
    for table in sorted(set(left) | set(right)):
        a, b = left.get(table, []), right.get(table, [])
        if len(a) != len(b):
            diffs.append(f"{table}: length {len(a)} != {len(b)}")
        for index, (x, y) in enumerate(zip(a, b)):
            if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
                diffs.append(f"{table}[{index}]: {x!r} != {y!r}")
    for line in diffs[:200]:
        print(line)
    print(f"comparison: {'OK' if not diffs else 'FAIL'} ({len(diffs)} difference(s))")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
