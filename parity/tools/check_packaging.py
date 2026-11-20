#!/usr/bin/env python3
"""Static wheel/packaging gate for the slim wheel + native PDF add-on."""
from __future__ import annotations

import argparse
import glob
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def wheel_path(pattern: str) -> Path:
    matches = sorted((ROOT / "target/wheels").glob(pattern))
    if not matches:
        raise SystemExit(f"no wheel matching {pattern}")
    return matches[-1]


def metadata(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
    assert len(names) == 1, names
    return zf.read(names[0]).decode("utf-8")


def check_main(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        meta = metadata(zf)
        assert re.search(r"Requires-Python: >=3\.12,\s*<3\.15", meta), meta
        assert "Requires-Dist: pydantic>=2.0" in meta, meta
        assert not re.search(r"^Requires-Dist: (bobine|ladybug|mcp|numpy)", meta, re.M), meta
        for extra in ("cpu", "gpu", "pdf"):
            assert f"Provides-Extra: {extra}" in meta, meta
        assert "Requires-Dist: okfgraph-pdf ; extra == 'pdf'" in meta, meta
        assert not any(n.startswith("okfgraph_pdf/") for n in names), names
        assert not any("__pycache__" in n or n.endswith(".pyc") for n in names), names
        assert any(n.startswith("okfgraph/_native.") for n in names), names
        if "-win_amd64" in path.name:
            libraries = [n for n in names if n.startswith("okfgraph.libs/")]
            assert any("libssl" in n for n in libraries), libraries
            assert any("libcrypto" in n for n in libraries), libraries
        assert any(n.endswith("entry_points.txt") for n in names), names
    print(f"OK: {path.name} is slim and abi3 (auditwheel repair enabled)")


def check_pdf(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        meta = metadata(zf)
        assert "Name: okfgraph-pdf" in meta, meta
        assert "Requires-Python: >=3.12,<3.15" in meta, meta
        assert "okfgraph_pdf/__init__.py" in names, names
        libraries = [
            n for n in names
            if n.endswith((".dll", ".so", ".dylib"))
        ]
        assert len(libraries) == 1, libraries
        assert not any(n.startswith("okfgraph/") for n in names), names
    print(f"OK: {path.name} ships only the locator and native plugin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="okfgraph-*.whl")
    ap.add_argument("--pdf", default="okfgraph_pdf-*.whl")
    args = ap.parse_args()
    check_main(wheel_path(args.main))
    check_pdf(wheel_path(args.pdf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
