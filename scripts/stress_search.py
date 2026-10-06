"""Stress the search matrix against a (copy of a) large graph DB.

Not a pytest file — run it directly against big graphs to catch the
ladybug 0.21.2 segfault class (exit 139: native multi-threaded scoring of
unbounded index result sets):

    .venv/Scripts/python.exe scripts/stress_search.py D:/path/to/copy.db

It drives `okf search` subprocesses over targets x ranks and query words
including common terms (which is what made the 0.21.2 FTS crash
deterministic in the field: an unbounded QUERY_FTS_INDEX whose terms
match many rows). Reports every non-zero exit; segfaults (exit 139) are
called out loudly. Optional 2nd arg pins the session precision when the
graph is fp16-pinned (a CPU venv must match):

    .venv/Scripts/python.exe scripts/stress_search.py D:/copy.db fp16
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Common words + medium words: the "many rows match" class. HYBRID
# queries pay a fresh model load per subprocess (~30 s), so only the
# crash-relevant common words run under ranks none/hub; model-free PPR
# (no ONNX load, but the same QUERY_FTS_INDEX scoring path at stage 1)
# gets the full list.
COMMON = ["index", "the system", "data"]
MEDIUM = ["model", "graph", "chunk", "a", "knowledge",
          "retrieval augmented generation over bitemporal ledgers",
          "embedding precision device"]
QUERIES = COMMON + MEDIUM

TARGETS = ["concepts", "chunks", "images"]


def matrix(db: str, precision: str | None):
    """Honouring flag combinations only (X3): rank is concepts-only;
    chunks gets hub_rerank/expand; images takes no rank flags."""
    cli = sys.executable
    cases = []

    def add(target, extra_flags, q):
        flags = [f"--target={target}", *extra_flags]
        if precision:
            # Big graphs are often pinned fp16 (GPU sessions); a CPU venv
            # must match that pin or the typed refusal
            # (PRECISION_PIN_MISMATCH) fires everywhere.
            flags.append(f"--precision={precision}")
        cases.append((target, "-".join(ex.strip("--") or "plain"
                                       for ex in extra_flags), q,
                      [cli, "-m", "okfgraph.cli", "search", *flags,
                       "--limit", "5", q, "--db-path", db]))

    for target in TARGETS:
        combo = {"concepts": [["--rank=none"], ["--rank=hub"], ["--rank=ppr"]],
                 "chunks": [[], ["--hub-rerank"], ["--expand"]],
                 "images": [[]]}[target]
        for extra in combo:
            # Hybrid variants load the model per process: keep those to
            # the common words that made the field crash deterministic;
            # model-free sweeps get the full query list.
            hybrid = bool(extra) and extra != ["--rank=ppr"]
            queries = COMMON if hybrid else MEDIUM
            for q in queries:
                add(target, extra, q)
    return cases


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    db = argv[1]
    precision = argv[2] if len(argv) == 3 else None
    if not Path(db).exists():
        print(f"no such database: {db}")
        return 2
    failures = 0
    for target, rank, query, cmd in matrix(db, precision):
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              errors="replace", timeout=300)
        label = f"[{target}/{rank}] {query[:40]!r}"
        if proc.returncode == 0:
            print(f"ok   {label}")
            continue
        failures += 1
        if proc.returncode == 139 or proc.returncode < 0:
            print(f"SEGV {label} -> exit {proc.returncode}")
        else:
            print(f"FAIL {label} -> exit {proc.returncode}")
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        if tail:
            print("     last line:", tail[-1][:200])
    print(f"\n{failures} failing case(s) of 90")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
