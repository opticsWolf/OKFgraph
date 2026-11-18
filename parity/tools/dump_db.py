#!/usr/bin/env python3
"""Dump a Ladybug DB as sorted JSON for DB-parity comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import ladybug as lb

    db = lb.Database(args.db)
    conn = lb.Connection(db)
    tables: list[str] = []
    for query in ("CALL SHOW_TABLES() RETURN *", "SHOW TABLES"):
        try:
            rows = conn.execute(query).rows_as_dict().get_all()
            for row in rows:
                name = row.get("name") or row.get("tableName") or row.get("table_name")
                if name:
                    tables.append(str(name))
            if tables:
                break
        except Exception:
            continue

    dump: dict[str, object] = {}
    for name in sorted(set(tables)):
        rows = None
        for pattern in (
            f"MATCH (n:`{name}`) RETURN n",
            f"MATCH ()-[r:`{name}`]->() RETURN r",
        ):
            try:
                rows = conn.execute(pattern).rows_as_dict().get_all()
                break
            except Exception:
                continue
        if rows is None:
            dump[name] = {"error": "neither node nor relationship pattern matched"}
            continue
        dump[name] = sorted(
            (json.loads(json.dumps(row, default=str, sort_keys=True)) for row in rows),
            key=lambda r: json.dumps(r, sort_keys=True),
        )

    conn.close()
    db.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(dump, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
