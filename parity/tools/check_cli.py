#!/usr/bin/env python3
"""Differential smoke for the Rust CLI command tree and --json envelopes."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import diff, drop_info_rules, normalize  # noqa: E402

HOST_INFO_RULES = {
    "ort_runtime",
    "ort_gpu_unused",
    "converter_cache",
    "cache_hygiene",
    "pdf_converter",
    "db_created",
}

# (args after the global flags, expected exit code)
CASES = [
    (["read", "alpha"], 0),
    (["read", "alpha", "--include", "chunks"], 0),
    (["read", "missing"], 1),
    (["traverse", ""], 0),
    (["traverse", "alpha", "--depth", "1"], 0),
    (["images", "alpha"], 0),
    (["deleted-list"], 0),
    (["broken-links"], 0),
    (["retitle-thoughts"], 0),
    (["doctor", "--stale-days", "100000"], 0),
    (["model-info"], 0),
    (["lint", "parity/corpus/minimal"], 0),
    (["diff", "parity/corpus/minimal", "parity/corpus/minimal"], 0),
]


def run_cli(exe: list[str], db: Path, args: list[str]) -> tuple[int, dict]:
    cmd = exe + [
        "--json",
        "--db-path", str(db),
        "--bundle-root", "parity/corpus/minimal",
        *args,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    try:
        payload = json.loads(stdout)
    except ValueError:
        raise SystemExit(f"non-JSON CLI output for {args}: {proc.stdout[:400]!r}\n{proc.stderr}")
    return proc.returncode, payload


def normalize_envelope(value: dict) -> dict:
    value = drop_info_rules(normalize(value), HOST_INFO_RULES)
    # Model cache paths/precision depend on the host's ORT installation; keep
    # the envelope and model id, like the doctor info-rule exclusions.
    if value.get("op") == "model_info" and isinstance(value.get("data"), dict):
        value["data"] = {"model_id": value["data"].get("model_id")}
    data = value.get("data")
    if isinstance(data, dict):
        report = data.get("report")
        if isinstance(report, dict) and isinstance(report.get("roots"), list):
            for root in report["roots"]:
                if isinstance(root, dict) and "path" in root:
                    root["path"] = "<root>"
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="parity/.venv-oracle/Scripts/python.exe")
    ap.add_argument("--rust", default="target/debug/okf.exe")
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10.db")
    args = ap.parse_args()

    python = [str(ROOT / args.python), "-m", "okfgraph.cli"]
    rust = [str(ROOT / args.rust)]
    py_db = ROOT / "parity/out/cli-diff-python.db"
    rs_db = ROOT / "parity/out/cli-diff-rust.db"
    py_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / args.fixture, py_db)
    shutil.copyfile(ROOT / args.fixture, rs_db)

    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    os.environ.update(env)

    # Command registration: the Rust help must list exactly the oracle verbs.
    import importlib.util
    spec = importlib.util.spec_from_file_location("oracle_cli", ROOT / "okfgraph/cli.py")
    assert spec and spec.loader
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    expected = set(oracle._COMMANDS)
    help_text = subprocess.run(rust + ["--help"], cwd=ROOT, capture_output=True, text=True).stdout
    block = help_text.split("Commands:", 1)[1].split("Options:", 1)[0]
    actual = {line.strip().split()[0] for line in block.splitlines() if line.strip()}
    actual.discard("help")
    if actual != expected:
        raise SystemExit(f"CLI verb mismatch: missing {expected - actual}, extra {actual - expected}")

    # Exit codes outside the JSON case matrix: argparse usage errors are 2,
    # and a bare invocation prints help and exits 0 on both CLIs.
    for label, extra, expected_code in [
        ("usage: read without id", ["--json", "read"], 2),
        ("no subcommand", [], 0),
    ]:
        py_rc = subprocess.run(python + extra, cwd=ROOT, capture_output=True).returncode
        rs_rc = subprocess.run(rust + extra, cwd=ROOT, capture_output=True).returncode
        if py_rc != expected_code or rs_rc != expected_code:
            raise SystemExit(
                f"{label}: exit python={py_rc} rust={rs_rc} expected={expected_code}"
            )
        print(f"[ OK ] {label}")

    failures = 0
    for argv, expected_code in CASES:
        py_rc, py_env = run_cli(python, py_db, argv)
        rs_rc, rs_env = run_cli(rust, rs_db, argv)
        diffs: list[str] = []
        if py_rc != expected_code or rs_rc != expected_code:
            diffs.append(f"exit: python={py_rc} rust={rs_rc} expected={expected_code}")
        diff(
            normalize_envelope(py_env),
            normalize_envelope(rs_env),
            "$",
            1e-6,
            diffs,
            {"crash_report"},
        )
        if diffs:
            failures += 1
            print(f"[FAIL] {' '.join(argv)}")
            for line in diffs[:20]:
                print(f"  {line}")
        else:
            print(f"[ OK ] {' '.join(argv)}")
    print(f"CLI: {len(CASES) - failures}/{len(CASES)} OK")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
