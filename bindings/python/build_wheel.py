#!/usr/bin/env python3
"""Build the okfgraph wheel with the facade staged next to the binding.

Maturin resolves ``python-source`` relative to this crate directory and does not
package a parent path such as ``../../python``. The canonical facade remains at
the repository root under ``python/``; this script copies it into a temporary
``bindings/python/python`` staging directory, runs maturin, and then removes the
staging copy.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "python"
STAGE = HERE / "python"
WHEELS = ROOT / "target" / "wheels"


def main() -> int:
    shutil.rmtree(STAGE, ignore_errors=True)
    try:
        for name in ("okfgraph", "okfgraph_pdf"):
            shutil.copytree(
                SOURCE / name,
                STAGE / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        WHEELS.mkdir(parents=True, exist_ok=True)
        # Maturin can skip DLL staging when it sees an already-patched artifact
        # in this scratch dir, leaving a .pyd that imports missing hashed DLLs.
        shutil.rmtree(ROOT / "target" / "maturin", ignore_errors=True)
        extra_args = list(sys.argv[1:])
        python = os.environ.get("OKFGRAPH_WHEEL_PYTHON") or sys.executable
        if "--python" in extra_args:
            index = extra_args.index("--python")
            python = extra_args[index + 1]
            del extra_args[index:index + 2]
        # Repair by default: a Windows wheel without its OpenSSL DLLs fails
        # on any clean machine. Callers may still override this explicitly.
        if not any(arg.startswith("--auditwheel") for arg in extra_args):
            extra_args.insert(0, "--auditwheel=repair")
        if not any(arg in ("--release", "--profile") for arg in extra_args):
            extra_args += ["--profile", "dev"]
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--interpreter",
                python,
                "--out",
                str(WHEELS),
                *extra_args,
            ],
            cwd=HERE,
        )
    finally:
        shutil.rmtree(STAGE, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
