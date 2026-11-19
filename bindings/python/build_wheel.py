#!/usr/bin/env python3
"""Build the okfgraph wheel with the facade staged next to the binding.

Maturin resolves ``python-source`` relative to this crate directory and does not
package a parent path such as ``../../python``. The canonical facade remains at
the repository root under ``python/``; this script copies it into a temporary
``bindings/python/python`` staging directory, runs maturin, and then removes the
staging copy.
"""
from __future__ import annotations

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
            shutil.copytree(SOURCE / name, STAGE / name)
        WHEELS.mkdir(parents=True, exist_ok=True)
        extra_args = sys.argv[1:] or ["--auditwheel=repair", "--profile", "dev"]
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--interpreter",
                sys.executable,
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
