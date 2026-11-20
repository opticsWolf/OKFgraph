#!/usr/bin/env python3
"""Assert every public op appears in at least one parity scenario."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    source = (ROOT / "src/ops/mod.rs").read_text(encoding="utf-8")
    block = re.search(r"pub const OP_NAMES: &\[&str\] = &\[(.*?)\];", source, re.S)
    if not block:
        raise SystemExit("cannot find OP_NAMES")
    ops = re.findall(r'"([a-z_]+)"', block.group(1))
    covered: set[str] = set()
    for path in sorted((ROOT / "parity/scenarios").glob("*.json")):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        for call in scenario.get("ops") or []:
            covered.add(call.get("op", ""))
    missing = [op for op in ops if op not in covered]
    print(f"ops: {len(ops)}, covered: {len(ops) - len(missing)}")
    if missing:
        print("missing:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
