#!/usr/bin/env python3
"""Compare normalized Python and Rust scenario output."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def normalize(value: Any, key: str = "") -> Any:
    """Python builds tag lists from ``set``; order is not a contract."""
    if isinstance(value, dict):
        return {k: normalize(v, k) for k, v in value.items()}
    if isinstance(value, list):
        items = [normalize(item, key) for item in value]
        if key == "tags" and all(isinstance(item, str) for item in items):
            return sorted(items)
        return items
    return value


def drop_info_rules(value: Any, rules: set[str]) -> Any:
    """Hide only environment-specific doctor ``info`` entries.

    The whole ``info`` array used to be ignored, which also hid portable
    probes (index inventory, image routes, lanes, stale locks). Drop the
    explicitly listed rules instead so the remaining entries are compared.
    """
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "info" and isinstance(item, list):
                value[key] = [
                    entry for entry in item
                    if not (isinstance(entry, dict) and entry.get("rule") in rules)
                ]
            else:
                drop_info_rules(item, rules)
    elif isinstance(value, list):
        for item in value:
            drop_info_rules(item, rules)
    return value


def diff(a: Any, b: Any, path: str, tol: float, out: list[str], ignore: set[str]) -> None:
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
            if key in ignore:
                continue
            if key not in a or key not in b:
                out.append(f"{path}.{key}: missing on one side")
            else:
                diff(a[key], b[key], f"{path}.{key}", tol, out, ignore)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", tol, out, ignore)
        return
    if a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--rust", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument(
        "--ignore-key", action="append", default=[],
        help="skip this object key at any depth (repeatable)")
    ap.add_argument(
        "--ignore-info-rule", action="append", default=[],
        help="skip only this doctor info rule (repeatable)")
    args = ap.parse_args()

    ignored_rules = set(args.ignore_info_rule)
    a = drop_info_rules(
        normalize(json.loads(Path(args.python).read_text(encoding="utf-8"))),
        ignored_rules,
    )
    b = drop_info_rules(
        normalize(json.loads(Path(args.rust).read_text(encoding="utf-8"))),
        ignored_rules,
    )
    diffs: list[str] = []
    diff(a, b, "$", args.tolerance, diffs, set(args.ignore_key))
    for line in diffs[:200]:
        print(line)
    print(f"comparison: {'FAIL' if diffs else 'OK'} ({len(diffs)} difference(s))")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
