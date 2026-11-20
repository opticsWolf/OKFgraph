#!/usr/bin/env python3
"""Assert every public op has success coverage and typed-error coverage.

The original checker only looked for an op name in a scenario file. BUG-044
requires both a successful call and a typed `OKFError` envelope (or an explicit
waiver for ops whose public 0.15 signature has no failure branch). Success is
read from the scenario definitions; error coverage is read from the normalized
`parity/out/<scenario>.python.json` outputs produced by `run_matrix.py`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Public ops that cannot raise a typed OKFError in Python 0.15. A missing
# required Python argument raises TypeError, which is not part of the OKF
# surface; returning `None` for an unknown id is a success.
ERROR_WAIVERS = {
    "get_by_id": "unknown ids return None; missing arguments are Python TypeErrors",
    "lanes": "read-only settings view with no validation branch",
    "list_broken_links": "unconditional graph read",
    "list_deleted": "unconditional graph read",
    "list_directory": "unconditional graph read",
    "pin_info": "unconditional pin inventory",
    "purge_deleted": "unconditional purge of expired rows",
    "reindex": "unconditional index rebuild",
    "repair_links": "unconditional link repair",
}


def main() -> int:
    source = (ROOT / "src/ops/mod.rs").read_text(encoding="utf-8")
    block = re.search(r"pub const OP_NAMES: &\[&str\] = &\[(.*?)\];", source, re.S)
    if not block:
        raise SystemExit("cannot find OP_NAMES")
    ops = re.findall(r'"([a-z_]+)"', block.group(1))

    covered: set[str] = set()
    for path in sorted((ROOT / "parity/scenarios").glob("*.json")):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        covered.update(call.get("op", "") for call in scenario.get("ops") or [])

    errors: dict[str, list[str]] = {}
    outputs = list((ROOT / "parity/out").glob("*.python.json"))
    for path in outputs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for call in data.get("ops") or []:
            if not call.get("ok"):
                errors.setdefault(call.get("op", ""), []).append(path.stem)

    missing_success = [op for op in ops if op not in covered]
    missing_error = [
        op for op in ops
        if op not in errors and op not in ERROR_WAIVERS
    ]
    print(
        f"ops: {len(ops)}, success: {len(ops) - len(missing_success)}, "
        f"typed error: {len(ops) - len(missing_error) - len(set(ops) & set(ERROR_WAIVERS))}, "
        f"documented waivers: {len(set(ops) & set(ERROR_WAIVERS))}"
    )
    print(f"outputs scanned: {len(outputs)}")
    failed = False
    if missing_success:
        print("missing success:", ", ".join(missing_success))
        failed = True
    if missing_error:
        print("missing typed error:", ", ".join(missing_error))
        failed = True
    if failed:
        print("run `python parity/tools/run_matrix.py` first if outputs are stale")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
