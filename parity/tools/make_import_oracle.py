#!/usr/bin/env python3
"""Create a fresh Python 0.15 v10 DB by importing an on-disk corpus."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from okfgraph import OKFRouter

    corpus = Path(args.corpus)
    files = sorted(corpus.rglob("*.md"))
    if not files:
        raise SystemExit(f"no markdown files under {corpus}")
    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)

    router = OKFRouter(
        db_path=str(db),
        bundle_root=str(corpus),
        model_id="google/embeddinggemma-2",
        embedding_dim=512,
        device="cuda",
        precision="auto",
        enable_chunking=True,
        chunk_size=512,
        chunk_overlap=40,
    )
    try:
        report = router.import_files([str(p) for p in files], mode="text")
        out = {
            "concept_ids": report["concept_ids"],
            "images": report["images"],
            "used_cuda": bool(router.encoder.used_cuda),
        }
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out))
    finally:
        router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
