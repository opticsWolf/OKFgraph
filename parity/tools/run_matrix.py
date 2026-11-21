#!/usr/bin/env python3
"""Run the full differential matrix and fail on any unexpected divergence.

The older workflow was a hand-written shell loop: it compared ops output with
a long list of volatile keys and quietly skipped DB dumps for several write
scenarios. This runner centralizes the normalization rules and makes the
matrix exit non-zero when a scenario diverges. The one intentional divergence
(`ingest-pdf-no-plugin`) is skipped unless --include-divergence is given.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import oracle_root  # noqa: E402

OPS_IGNORE_KEYS = [
    "created_at", "updated_at", "effective_at", "timestamp", "path",
    "md_path", "image_dir", "output_dir", "root", "dir", "new_path",
    "old_path", "seconds", "message", "recoverable_until", "trashed_files",
    "deleted_at", "age_seconds", "bundle", "detached_at", "since",
    "last_seen",
]

DB_IGNORE_KEYS = [
    "created_at", "updated_at", "effective_at", "timestamp", "path",
    "detached_at", "since", "last_seen",
]

INFO_IGNORE_RULES = [
    "ort_runtime", "ort_gpu_unused", "converter_cache", "cache_hygiene",
    "pdf_converter", "db_created",
]

# These scenarios only read a fixture copy; their DB dumps intentionally
# differ in pin metadata from a freshly opened Rust handle. Write scenarios
# with a runner-specific DB path are compared below.
DB_COMPARE_SKIP = {
    "doctor-empty", "doctor-params", "ingest-params",
    "ingest-thoughts", "ingest-thoughts-bundle-root",
    "ingest-thoughts-errors", "ingest-thoughts-folder",
}


def expand(value: Any, runner: str) -> Any:
    if isinstance(value, str):
        return value.replace("@RUNNER@", runner)
    if isinstance(value, list):
        return [expand(item, runner) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, runner) for key, item in value.items()}
    return value


def run(cmd: list[str], env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env or os.environ.copy(),
        check=False,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", default="parity/scenarios")
    ap.add_argument("--out-dir", default="parity/out")
    ap.add_argument("--oracle-python", default="parity/.venv-oracle/Scripts/python.exe")
    ap.add_argument("--oracle-root", default=None)
    ap.add_argument("--rust-bin", default="")
    ap.add_argument("--pdf-plugin", default="")
    ap.add_argument("--include-divergence", action="store_true")
    ap.add_argument("--only", action="append", default=[], help="run only this scenario name")
    ap.add_argument("--skip", action="append", default=[], help="skip this scenario name")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    oracle_python = str(ROOT / args.oracle_python)
    scenarios = sorted((ROOT / args.scenario_dir).glob("*.json"))
    if args.only:
        scenarios = [p for p in scenarios if p.stem in set(args.only)]
    fails: list[str] = []
    checked = 0

    for scenario in scenarios:
        name = scenario.stem
        if name in args.skip:
            continue
        if name == "ingest-pdf-no-plugin" and not args.include_divergence:
            print(f"[skip] {name}: intentional Python INTERNAL / Rust CONVERSION_FAILED divergence")
            continue
        data = json.loads(scenario.read_text(encoding="utf-8"))
        py_out = out / f"{name}.python.json"
        rust_out = out / f"{name}.rust.json"
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONIOENCODING"] = "utf-8"
        env["OKFGRAPH_ORACLE_ROOT"] = str(oracle_root(args.oracle_root))
        if args.pdf_plugin:
            env["OKFGRAPH_PDF_PLUGIN"] = args.pdf_plugin

        py = run([
            oracle_python, "parity/tools/run_python.py",
            "--scenario", str(scenario.relative_to(ROOT)),
            "--out", str(py_out.relative_to(ROOT)),
        ], env, capture=True)
        if py.returncode:
            fails.append(f"{name}: python runner: {py.stderr.strip()[-400:]}")
            print(f"[FAIL] {name}: python runner")
            continue

        db_path = (data.get("settings") or {}).get("db_path")
        py_db = ROOT / expand(db_path, "python") if db_path else None
        py_dump = out / f"{name}.python-dump.json"
        if py_db and "@RUNNER@" in db_path and py_db.exists():
            dump = run([
                oracle_python, "parity/tools/dump_db.py",
                "--db", str(py_db), "--out", str(py_dump),
            ], env, capture=True)
            if dump.returncode:
                fails.append(f"{name}: python dump: {dump.stderr.strip()[-400:]}")

        if args.rust_bin:
            rust_cmd = [args.rust_bin, str(scenario)]
        else:
            rust_cmd = ["cargo", "run", "-q", "-p", "okfgraph", "--bin", "parity-run", "--", str(scenario)]
        rust = run(rust_cmd, env, capture=True)
        if rust.returncode:
            fails.append(f"{name}: rust runner: {rust.stderr.strip()[-400:]}")
            print(f"[FAIL] {name}: rust runner")
            continue
        rust_out.write_text(rust.stdout, encoding="utf-8")

        compare = [sys.executable, "parity/tools/compare.py",
                   "--python", str(py_out), "--rust", str(rust_out)]
        for key in OPS_IGNORE_KEYS:
            compare += ["--ignore-key", key]
        for rule in INFO_IGNORE_RULES:
            compare += ["--ignore-info-rule", rule]
        cmp = run(compare, env, capture=True)
        if cmp.returncode:
            fails.append(f"{name}: ops: {cmp.stdout.strip()[:800]}")
            print(f"[FAIL] {name}: ops")
            continue

        if py_db and "@RUNNER@" in db_path and name not in DB_COMPARE_SKIP:
            rust_db = ROOT / expand(db_path, "rust")
            rust_dump = out / f"{name}.rust-dump.json"
            if rust_db.exists() and py_dump.exists():
                dump = run(["cargo", "run", "-q", "-p", "okfgraph", "--example", "dump_db",
                            "--", str(rust_db)], env, capture=True)
                if dump.returncode:
                    fails.append(f"{name}: rust dump: {dump.stderr.strip()[-400:]}")
                else:
                    rust_dump.write_text(dump.stdout, encoding="utf-8")
                    db_cmp = [sys.executable, "parity/tools/compare_db.py",
                              "--python", str(py_dump), "--rust", str(rust_dump)]
                    for key in DB_IGNORE_KEYS:
                        db_cmp += ["--ignore-key", key]
                    result = run(db_cmp, env, capture=True)
                    if result.returncode:
                        fails.append(f"{name}: db: {result.stdout.strip()[:800]}")
                        print(f"[FAIL] {name}: db")
                        continue

        checked += 1
        print(f"[ OK ] {name}")

    print(f"TOTAL:{checked} FAILS:{len(fails)}")
    for failure in fails:
        print(failure)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
