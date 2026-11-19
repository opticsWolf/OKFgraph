#!/usr/bin/env python3
"""Compare normalized Python and Rust scenario output."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def diff(a: Any, b: Any, path: str, tol: float, out: list[str]) -> None:
    if isinstance(a, float) and isinstance(b, float):
        if not (math.isclose(a, b, rel_tol=tol, abs_tol=tol) or (math.isnan(a) and math.isnan(b))):
            out.append(f"{path}: {a!r} != {b!r}")
        return
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, dict):
        keys = set(a) | set(b)
        if path == "$":
            keys.discard("runner")  # provenance only, not behavior
        for key in sorted(keys):
            if key not in a or key not in b:
                out.append(f"{path}.{key}: missing on one side")
            else:
                diff(a[key], b[key], f"{path}.{key}", tol, out)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", tol, out)
        return
    if a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--rust", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    a = json.loads(Path(args.python).read_text(encoding="utf-8"))
    b = json.loads(Path(args.rust).read_text(encoding="utf-8"))
    diffs: list[str] = []
    diff(a, b, "$", args.tolerance, diffs)
    for line in diffs[:200]:
        print(line)
    print(f"comparison: {'FAIL' if diffs else 'OK'} ({len(diffs)} difference(s))")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
