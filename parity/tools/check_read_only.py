#!/usr/bin/env python3
"""Read-only regression: a Stage-A handle must refuse write ops with DB_LOCKED."""
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


def run(runner: Path, scenario: Path, env: dict[str, str]) -> dict:
    result = subprocess.run(
        [str(runner), str(scenario)], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default=str(ROOT / "target" / "debug" / "parity-run.exe"))
    ap.add_argument("--fixture", default=str(ROOT / "parity" / "fixtures" / "oracle-v10.db"))
    args = ap.parse_args()
    runner = Path(args.runner)
    if not runner.is_file():
        print(f"missing runner {runner}; run: cargo build -p okfgraph --bin parity-run", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="okf-readonly-"))
    try:
        db = work / "readonly.db"
        shutil.copyfile(args.fixture, db)
        scenario = work / "scenario.json"
        scenario.write_text(json.dumps({
            "name": "read-only-write-refusal",
            "settings": {
                "db_path": str(db).replace("\\", "/"),
                "embedding_dim": 3,
                "model_id": "google/embeddinggemma-2",
                "device": "cpu",
                "precision": "auto",
            },
            "ops": [{"op": "reindex", "params": {}}],
        }), encoding="utf-8")
        env = dict(os.environ)
        env["OKFGRAPH_PARITY_READ_ONLY"] = "1"
        payload = run(runner, scenario, env)
        error = payload["ops"][0].get("error") or {}
        if error.get("code") != "DB_LOCKED":
            print(f"expected DB_LOCKED, got {payload['ops'][0]}", file=sys.stderr)
            return 1
        env.pop("OKFGRAPH_PARITY_READ_ONLY", None)
        payload = run(runner, scenario, env)
        if not payload["ops"][0].get("ok"):
            print(f"writable run failed: {payload['ops'][0]}", file=sys.stderr)
            return 1
        print("read-only: OK (write refused with DB_LOCKED; writable handle succeeds)")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
