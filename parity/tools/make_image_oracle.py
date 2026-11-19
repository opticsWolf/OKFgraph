#!/usr/bin/env python3
"""Create a GPU-embedded Python 0.15 v10 fixture with real ImageAsset rows.

The oracle ingests small PNGs with ``mode=text`` so the fixture needs no
vision model download: captions are embedded with the graph's real text
model into the same space the paired vision slot uses. The Rust image-search
op queries the stored image index with the same text-model query vector.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING)

IMAGES = {
    "red-fox.png": (190, 70, 45),
    "blue-jay.png": (40, 90, 190),
    "city-lights.png": (25, 30, 70),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="google/embeddinggemma-2")
    ap.add_argument("--embedding-dim", type=int, default=512)
    ap.add_argument("--image-mode", default="text", choices=["text", "optional", "omni"])
    args = ap.parse_args()

    from PIL import Image
    from okfgraph import OKFRouter

    corpus = Path(args.corpus)
    corpus.mkdir(parents=True, exist_ok=True)
    for name, colour in IMAGES.items():
        Image.new("RGB", (32, 24), colour).save(corpus / name)

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)

    router = OKFRouter(
        db_path=str(db),
        bundle_root=str(corpus),
        model_id=args.model_id,
        embedding_dim=args.embedding_dim,
        device="cuda",
        precision="auto",
        enable_chunking=True,
        chunk_size=128,
        chunk_overlap=8,
    )
    try:
        files = sorted(corpus.glob("*.md"))
        report = router.import_files([str(p) for p in files], mode=args.image_mode)
        queries = ["red fox", "blue jay", "city lights", "animal"]
        searches = {q: router.search(q, target="images", limit=5) for q in queries}
        result = {
            "concept_ids": report["concept_ids"],
            "images": report["images"],
            "used_cuda": bool(router.encoder.used_cuda),
            "searches": searches,
        }
        Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({
            "concepts": len(report["concept_ids"]),
            "images": report["images"],
            "used_cuda": result["used_cuda"],
            "image_mode": args.image_mode,
        }))
    finally:
        router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
