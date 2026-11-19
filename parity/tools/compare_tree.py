#!/usr/bin/env python3
"""Compare two exported directory trees byte-for-byte."""
from __future__ import annotations

import argparse
import filecmp
from pathlib import Path


def snapshot(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--rust", required=True)
    args = ap.parse_args()
    left, right = snapshot(Path(args.python)), snapshot(Path(args.rust))
    diffs: list[str] = []
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            diffs.append(f"{name}: missing on one side")
        elif not filecmp.cmp(left[name], right[name], shallow=False):
            diffs.append(f"{name}: content differs")
    for line in diffs[:200]:
        print(line)
    print(f"tree: {'OK' if not diffs else 'FAIL'} ({len(diffs)} difference(s))")
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
