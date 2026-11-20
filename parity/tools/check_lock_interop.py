#!/usr/bin/env python3
"""Cross-process lock check: Python fasteners must exclude a Rust writer.

The oracle's writer lock is ``<db>.lock`` through ``fasteners``; the Rust core
uses ``fd-lock`` on the same path. Hold the Python lock, run a Rust write op
with a short test-only timeout, and assert it reports WRITE_LOCK_TIMEOUT;
release and assert the same op succeeds.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=str(ROOT / "target" / "debug" / "parity-run.exe"))
    ap.add_argument("--hold", default=str(ROOT / "target" / "debug" / "examples" / "lock_hold.exe"))
    ap.add_argument("--fixture", default=str(ROOT / "parity" / "fixtures" / "oracle-v10.db"))
    args = ap.parse_args()

    import fasteners

    runner = Path(args.runner)
    hold = Path(args.hold)
    if not runner.is_file():
        print(f"missing runner {runner}; run: cargo build -p okfgraph --bin parity-run", file=sys.stderr)
        return 2
    if not hold.is_file():
        print(f"missing lock helper {hold}; run: cargo build -p okfgraph --example lock_hold", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="okf-lock-"))
    try:
        db = work / "lock-test.db"
        shutil.copyfile(args.fixture, db)
        scenario = work / "scenario.json"
        scenario.write_text(json.dumps({
            "name": "lock-interop",
            "settings": {
                "db_path": str(db).replace("\\", "/"),
                "embedding_dim": 3,
                "model_id": "google/embeddinggemma-2",
                "device": "cpu",
                "precision": "auto",
            },
            "ops": [{"op": "repair_links", "params": {}}],
        }), encoding="utf-8")
        env = dict(os.environ)
        env["OKFGRAPH_WRITE_LOCK_TIMEOUT"] = "0.5"
        lock = fasteners.InterProcessLock(str(db) + ".lock")
        assert lock.acquire(blocking=False)
        try:
            held = subprocess.run(
                [str(runner), str(scenario)], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=60,
            )
        finally:
            lock.release()
        if held.returncode != 0:
            print(f"runner crashed while lock held: {held.stderr}", file=sys.stderr)
            return 1
        payload = json.loads(held.stdout)
        error = payload["ops"][0].get("error") or {}
        if error.get("code") != "WRITE_LOCK_TIMEOUT":
            print(f"expected WRITE_LOCK_TIMEOUT, got {payload['ops'][0]}", file=sys.stderr)
            return 1
        free = subprocess.run(
            [str(runner), str(scenario)], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=60,
        )
        if free.returncode != 0:
            print(f"runner crashed after release: {free.stderr}", file=sys.stderr)
            return 1
        payload = json.loads(free.stdout)
        if not payload["ops"][0].get("ok"):
            print(f"write failed after release: {payload['ops'][0]}", file=sys.stderr)
            return 1
        # Reverse direction: hold the same file through fd-lock and make sure
        # the Python writer cannot slip in.
        holder = subprocess.Popen(
            [str(hold), str(db) + ".lock", "2000"],
            cwd=ROOT, stdout=subprocess.PIPE, text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "locked"
            probe = fasteners.InterProcessLock(str(db) + ".lock")
            if probe.acquire(blocking=False):
                probe.release()
                print("Python acquired while Rust held the lock", file=sys.stderr)
                return 1
        finally:
            holder.terminate()
            holder.wait(timeout=10)
        check = fasteners.InterProcessLock(str(db) + ".lock")
        assert check.acquire(blocking=False)
        check.release()
        print("lock interop: OK (Python fasteners and Rust fd-lock exclude each other)")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
