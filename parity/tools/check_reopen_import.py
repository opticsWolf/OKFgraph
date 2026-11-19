#!/usr/bin/env python3
"""Verify Python 0.15 reopens a DB imported by the Rust Stage B pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--concept", default="alpha")
    ap.add_argument("--query", default="beta")
    args = ap.parse_args()

    from okfgraph import OKFRouter

    router = OKFRouter(
        db_path=args.db,
        model_id="google/embeddinggemma-2",
        embedding_dim=512,
        device="cuda",
        precision="auto",
    )
    try:
        concept = router.get_by_id(args.concept)
        hits = router.search(args.query, limit=5)
        assert concept is not None and concept.id == args.concept
        assert hits
        print(f"OK: Python 0.15 reopened Rust import ({len(hits)} search hits)")
    finally:
        router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
