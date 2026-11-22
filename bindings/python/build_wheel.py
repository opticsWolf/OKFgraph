#!/usr/bin/env python3
"""Build the okfgraph wheel with the facade staged next to the binding.

Maturin resolves ``python-source`` relative to this crate directory and does not
package a parent path such as ``../../python``. The canonical facade remains at
the repository root under ``python/``; this script copies it into a temporary
``bindings/python/python`` staging directory, runs maturin, and then removes the
staging copy.

The native PDF plugin is built first and staged inside the ``okfgraph`` package
as ``okfgraph/_plugin/<library>``, so a single wheel (and a single PyPI
project) carries both the core and the optional converter.
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
PLUGIN_LIBRARIES = (
    "okfgraph_pdf.dll",
    "libokfgraph_pdf.so",
    "libokfgraph_pdf.dylib",
)


def profile_dir(args: list[str]) -> tuple[str, list[str]]:
    """Return the cargo profile directory and the args that select it."""
    if "--release" in args:
        return "release", ["--release"]
    if "--profile" in args:
        index = args.index("--profile")
        name = args[index + 1]
        # Cargo maps the built-in `dev`/`test` profiles to target/debug.
        directory = "debug" if name in ("dev", "test") else name
        return directory, ["--profile", name]
    return "debug", ["--profile", "dev"]


def build_plugin(profile_args: list[str], directory: str) -> None:
    subprocess.check_call(
        ["cargo", "build", "-p", "okfgraph-pdf", *profile_args],
        cwd=ROOT,
    )
    built = ROOT / "target" / directory
    source = next(
        (built / name for name in PLUGIN_LIBRARIES if (built / name).is_file()),
        None,
    )
    if source is None:
        raise SystemExit(f"no PDF plugin library in {built}")
    destination = STAGE / "okfgraph" / "_plugin"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination / source.name)


def main() -> int:
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
    directory, profile_args = profile_dir(extra_args)

    shutil.rmtree(STAGE, ignore_errors=True)
    try:
        shutil.copytree(
            SOURCE / "okfgraph",
            STAGE / "okfgraph",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_plugin"),
        )
        build_plugin(profile_args, directory)
        WHEELS.mkdir(parents=True, exist_ok=True)
        # Maturin can skip DLL staging when it sees an already-patched artifact
        # in this scratch dir, leaving a .pyd that imports missing hashed DLLs.
        shutil.rmtree(ROOT / "target" / "maturin", ignore_errors=True)
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
