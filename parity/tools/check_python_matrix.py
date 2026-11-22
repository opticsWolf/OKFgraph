#!/usr/bin/env python3
"""Install the built wheels into clean venvs for each supported Python."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "parity/out"
TOOLS = ROOT / "parity/tools"


def resolve_wheel(pattern: str) -> Path:
    path = ROOT / "target/wheels" / pattern
    if "*" not in pattern:
        if not path.is_file():
            raise SystemExit(f"missing wheel: {path}")
        return path
    matches = sorted(
        path.parent.glob(path.name),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if len(matches) != 1:
        raise SystemExit(f"expected one wheel for {pattern}, found {matches}")
    return matches[0]


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("OKFGRAPH_PDF_PLUGIN", None)
    parts = [
        part for part in env.get("PATH", "").split(os.pathsep)
        if "vcpkg" not in part.lower()
    ]
    env["PATH"] = os.pathsep.join(parts)
    return env


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=OUT, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["3.12", "3.13", "3.14"])
    ap.add_argument("--main", default="okfgraph-*.whl")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pdf-smoke", action="store_true")
    args = ap.parse_args()

    wheels = [resolve_wheel(args.main)]
    OUT.mkdir(parents=True, exist_ok=True)

    for version in args.versions:
        venv = OUT / f"wheel-matrix-{version}"
        shutil.rmtree(venv, ignore_errors=True)
        run(["uv", "venv", "--python", version, str(venv)])
        python = venv_python(venv)
        command = ["uv", "pip", "install", "--python", str(python)]
        if args.offline:
            command.append("--offline")
        command += [str(path) for path in wheels]
        run(command)

        db = OUT / f"wheel-matrix-{version}.db"
        shutil.copyfile(ROOT / "parity/fixtures/oracle-v10.db", db)
        env = clean_env()
        run([str(python), str(TOOLS / "check_wheel.py")], env)
        run([str(python), str(TOOLS / "check_facade.py"), "--db", str(db)], env)
        if args.pdf_smoke:
            run([str(python), str(TOOLS / "check_pdf_wheel.py")])
        print(f"OK: Python {version} wheel matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
