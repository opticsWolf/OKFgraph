#!/usr/bin/env python3
"""Generate pycompat golden values from the frozen Python 0.15 oracle."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "oracle_timestamps", ROOT / "okfgraph" / "components" / "timestamps.py"
)
_timestamps = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_timestamps)
content_sig = _timestamps.content_sig
extra_value = _timestamps.extra_value

OUT = ROOT / "parity" / "golden" / "pycompat.json"


def encode(value):
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__type__": "date", "value": value.isoformat()}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def main() -> int:
    dt = datetime(2024, 1, 2, 3, 4, 5)
    day = date(2024, 1, 2)

    json_cases = []
    for name, value, sort_keys in [
        ("unicode", {"k": "héllo ☃ \u2028 \u2029\t\n\"\\"}, False),
        ("nested", {"a": [1, 2.0, None, True], "b": {"x": "y"}}, False),
        ("sort_keys", {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}, True),
        ("floats", [0.0, 1.0, -0.0, 1e-7, 1e20, 1.2345678901234567], False),
        ("default_str", {"d": dt, "day": day}, False),
    ]:
        json_cases.append(
            {
                "name": name,
                "value": encode(value),
                "sort_keys": sort_keys,
                "expected": json.dumps(value, sort_keys=sort_keys, default=str),
            }
        )

    extra_cases = []
    for name, value in [
        ("string", "héllo"),
        ("date", day),
        ("datetime", dt),
        ("int", 42),
        ("bool", True),
        ("none", None),
        ("list", [1, "x", dt]),
        ("dict", {"b": 2, "a": 1}),
        ("float", 1e-7),
    ]:
        extra_cases.append(
            {"name": name, "value": encode(value), "expected": extra_value(value)}
        )

    sig_cases = []
    for name, data, body in [
        ("simple", {"title": "Alpha", "tags": ["x", "y"]}, "hello"),
        ("datetime", {"timestamp": dt, "extra": {"k": "v"}}, "héllo\n☃"),
        ("empty_body", {"id": "alpha"}, ""),
    ]:
        sig_cases.append(
            {"name": name, "data": encode(data), "body": body, "expected": content_sig(data, body)}
        )

    out = {
        "json_dumps": json_cases,
        "extra_value": extra_cases,
        "content_sig": sig_cases,
        "datetime": {
            "str": str(dt),
            "isoformat": dt.isoformat(),
            "date_str": str(day),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
