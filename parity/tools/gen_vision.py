#!/usr/bin/env python3
"""Generate the Python oracle golden for the native vision lane.

Uses embroider's own ``prepare_image`` (Pillow bicubic) and ``JinaV5Vision``
on the cached jina omni-nano slot, paired with jina text-nano. No downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--model-id", default="jina-v5-omni-nano-retrieval-vision")
    ap.add_argument("--dim", type=int, default=768)
    args = ap.parse_args()

    import embroider

    model_id = args.model_id
    cls = (
        embroider.GemmaVision
        if model_id.startswith("google/")
        else embroider.JinaV5Vision
    )
    vision = cls.open(
        model_id, truncate_dim=args.dim, device="cuda", precision="auto")
    cases = {}
    for name in args.images:
        data = Path(name).read_bytes()
        rgb, height, width = embroider.prepare_image(data, model_id)
        cases[Path(name).name] = {
            "height": height,
            "width": width,
            "pixels_sha256": hashlib.sha256(rgb).hexdigest(),
            "vector": vision.encode_image(rgb, height, width),
        }
    Path(args.out).write_text(json.dumps({
        "model_id": model_id,
        "dim": vision.dim,
        "used_cuda": vision.used_cuda,
        "precision": vision.precision,
        "cases": cases,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "used_cuda": vision.used_cuda}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
