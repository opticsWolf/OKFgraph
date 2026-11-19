#!/usr/bin/env python3
"""Compare the native vision-lane spike with the Python embroider golden."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--rust", required=True)
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    rust = json.loads(Path(args.rust).read_text(encoding="utf-8"))
    diffs = []
    for field in ("model_id", "dim", "precision", "used_cuda"):
        if golden.get(field) != rust.get(field):
            diffs.append(f"{field}: {golden.get(field)!r} != {rust.get(field)!r}")
    for name, case in golden["cases"].items():
        got = rust.get("cases", {}).get(name)
        if got is None:
            diffs.append(f"cases.{name}: missing")
            continue
        for field in ("height", "width", "pixels_sha256"):
            if case[field] != got.get(field):
                diffs.append(f"cases.{name}.{field}: {case[field]!r} != {got.get(field)!r}")
        left, right = case["vector"], got.get("vector", [])
        if len(left) != len(right):
            diffs.append(f"cases.{name}.vector: length {len(left)} != {len(right)}")
            continue
        max_abs = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
        if max_abs > 1e-5:
            diffs.append(f"cases.{name}.vector: max_abs {max_abs}")
        dot = sum(a * b for a, b in zip(left, right))
        norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
        cosine = dot / norm if norm else 1.0
        if cosine < 0.999:
            diffs.append(f"cases.{name}.cosine: {cosine}")
    for diff in diffs:
        print(diff)
    print(f"comparison: {'OK' if not diffs else 'FAIL'} ({len(diffs)} difference(s))")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
