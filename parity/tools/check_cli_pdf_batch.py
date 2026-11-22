#!/usr/bin/env python3
"""Differential CLI smoke for multi-path PDF ingestion (BUG-087).

The API takes a single PDF. Folders, globs and multiple paths are expanded in
the CLI, so this gate compares full CLI envelopes rather than router ops.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import diff, normalize  # noqa: E402
from oracle import oracle_root  # noqa: E402


def run(exe: list[str], db: Path, args: list[str], env: dict[str, str]) -> tuple[int, dict]:
    cmd = exe + [
        "--json",
        "--db-path", str(db),
        "--bundle-root", "parity/corpus/minimal",
        *args,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    try:
        payload = json.loads(stdout)
    except ValueError as exc:
        raise SystemExit(
            f"non-JSON CLI output for {args}: {stdout[:400]!r}\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        ) from exc
    return proc.returncode, payload


def normalize_envelope(value: dict, tmp: Path) -> dict:
    value = normalize(value)
    value.pop("warnings", None)
    text = json.dumps(value)
    text = text.replace(str(tmp).replace("\\", "/"), "<tmp>")
    text = text.replace(str(tmp), "<tmp>")
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="parity/.venv-oracle/Scripts/python.exe")
    ap.add_argument("--rust", default="target/debug/okf.exe")
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10-import-minimal.db")
    ap.add_argument("--pdf-plugin", default=None)
    ap.add_argument("--oracle-root", default=None)
    args = ap.parse_args()

    oracle_dir = oracle_root(args.oracle_root)
    python = [str(ROOT / args.python), "-m", "okfgraph.cli"]
    rust = [str(ROOT / args.rust)]
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
        OKFGRAPH_NO_SERVER="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(oracle_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if args.pdf_plugin:
        env["OKFGRAPH_PDF_PLUGIN"] = str(ROOT / args.pdf_plugin)

    tmp = ROOT / "parity/tmp/cli-pdf-batch"
    if tmp.exists():
        shutil.rmtree(tmp)
    docs = tmp / "docs"
    docs.mkdir(parents=True)
    source = ROOT / "parity/fixtures/sample.pdf"
    for name in ["a.pdf", "b.pdf", "docs/c.pdf"]:
        shutil.copyfile(source, tmp / name)
    (docs / "notes.txt").write_text("not convertible\n", encoding="utf-8")
    missing = tmp / "missing-*.pdf"

    cases = [
        ("two paths", ["ingest", "--kind", "pdf", "--pdf-path", str(tmp / "a.pdf"), str(tmp / "b.pdf")], 0),
        ("folder", ["ingest", "--kind", "pdf", "--pdf-path", str(docs)], 0),
        ("glob", ["ingest", "--kind", "pdf", "--pdf-path", str(docs / "*.pdf")], 0),
        ("unmatched glob", ["ingest", "--kind", "pdf", "--pdf-path", str(missing)], 2),
        (
            "mixed valid and missing",
            ["ingest", "--kind", "pdf", "--pdf-path", str(tmp / "a.pdf"), str(tmp / "missing.pdf")],
            2,
        ),
    ]

    failures = 0
    for label, argv, expected_code in cases:
        py_db = tmp / f"{label.replace(' ', '-')}-python.db"
        rs_db = tmp / f"{label.replace(' ', '-')}-rust.db"
        shutil.copyfile(ROOT / args.fixture, py_db)
        shutil.copyfile(ROOT / args.fixture, rs_db)
        py_rc, py_payload = run(python, py_db, argv, env)
        rs_rc, rs_payload = run(rust, rs_db, argv, env)
        diffs: list[str] = []
        if py_rc != expected_code or rs_rc != expected_code:
            diffs.append(f"exit: python={py_rc} rust={rs_rc} expected={expected_code}")
        diff(
            normalize_envelope(py_payload, tmp),
            normalize_envelope(rs_payload, tmp),
            "$",
            1e-6,
            diffs,
            {"crash_report", "md_path", "asset_dir", "image_dir"},
        )
        if diffs:
            failures += 1
            print(f"[FAIL] {label}")
            for line in diffs[:20]:
                print(f"  {line}")
        else:
            print(f"[ OK ] {label}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
