#!/usr/bin/env python3
"""Verify that Python 0.15 can reopen a DB written by the Rust lbug spike."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    import ladybug as lb

    db = lb.Database(args.db)
    conn = lb.Connection(db)
    try:
        for ext in ("vector", "fts"):
            conn.execute(f"LOAD {ext};")
        rows = conn.execute(
            "MATCH (s:Spike) RETURN s.id AS id, s.text AS text"
        ).rows_as_dict().get_all()
    finally:
        conn.close()
        db.close()

    ids = {str(row["id"]) for row in rows}
    print(f"Spike rows: {sorted(ids)}")
    if "rust-write" not in ids:
        raise SystemExit("FAIL: Rust-written row is not visible to Python 0.15")
    print("OK: Python 0.15 reopened the Rust-written DB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
