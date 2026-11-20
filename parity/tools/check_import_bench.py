#!/usr/bin/env python3
"""Timing smoke for the real-model filters import (BUG-040).

This is deliberately a smoke test rather than a cross-machine SLA: it proves
the measured path still imports the corpus with a real embedder and that an
unchanged re-import stays a delta no-op. Record observed numbers in
`docs/plan-bug-fixes.md` when hardware changes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="parity/corpus/filters")
    ap.add_argument("--max-first-seconds", type=float, default=60.0)
    ap.add_argument("--max-second-seconds", type=float, default=5.0)
    ap.add_argument("--min-concepts", type=int, default=6)
    args = ap.parse_args()

    proc = subprocess.run(
        ["cargo", "run", "-q", "-p", "okfgraph", "--example", "import_bench", "--", args.corpus],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        print(proc.stderr[-2000:])
        raise SystemExit(f"import_bench failed ({proc.returncode})")
    report = json.loads(proc.stdout)
    print(
        "import benchmark: "
        f"{report['concepts']} concepts, first {report['first_seconds']:.3f}s, "
        f"unchanged re-import {report['second_seconds']:.3f}s, "
        f"queued {report['second_queued']}"
    )
    failures = []
    if report["concepts"] < args.min_concepts:
        failures.append(f"only {report['concepts']} concepts imported")
    if report["first_seconds"] > args.max_first_seconds:
        failures.append(f"cold import too slow: {report['first_seconds']:.3f}s")
    if report["second_seconds"] > args.max_second_seconds:
        failures.append(f"delta no-op too slow: {report['second_seconds']:.3f}s")
    if report["second_queued"] != 0:
        failures.append(f"unchanged re-import queued {report['second_queued']} files")
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        return 1
    print("import benchmark: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
