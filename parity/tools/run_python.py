#!/usr/bin/env python3
"""Run a parity scenario against the frozen Python oracle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_error_payload(exc: BaseException) -> dict[str, Any]:
    envelope = getattr(exc, "envelope", None)
    if callable(envelope):
        try:
            return envelope()
        except Exception:
            pass
    return {
        "ok": False,
        "op": None,
        "data": None,
        "warnings": [],
        "error": {"code": getattr(exc, "code", "INTERNAL"), "message": str(exc)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    from okfgraph import OKFRouter  # oracle import, on purpose

    settings = dict(scenario.get("settings") or {})
    router = OKFRouter(**settings)
    results: list[dict[str, Any]] = []
    try:
        for call in scenario.get("ops") or []:
            op = call["op"]
            params = call.get("params") or {}
            try:
                data = getattr(router, op)(**params)
                results.append({"ok": True, "op": op, "data": data})
            except Exception as exc:  # oracle errors are part of the golden output
                payload = load_error_payload(exc)
                payload.setdefault("op", op)
                results.append(payload)
    finally:
        router.close()

    out = {"runner": "python", "scenario": scenario.get("name"), "ops": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
