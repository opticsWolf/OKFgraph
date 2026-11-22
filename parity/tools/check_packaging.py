#!/usr/bin/env python3
"""Static wheel/packaging gate for the single bundled okfgraph wheel."""
from __future__ import annotations

import argparse
import glob
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def wheel_path(pattern: str) -> Path:
    matches = sorted(
        (ROOT / "target/wheels").glob(pattern),
        key=lambda path: path.stat().st_mtime,
    )
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
        assert "Requires-Dist: okfgraph-pdf" not in meta, meta
        assert not any(n.startswith("okfgraph_pdf/") for n in names), names
        plugins = [n for n in names if n.startswith("okfgraph/_plugin/")]
        assert len(plugins) == 1, plugins
        assert plugins[0].endswith((".dll", ".so", ".dylib")), plugins
        assert not any("__pycache__" in n or n.endswith(".pyc") for n in names), names
        assert any(n.startswith("okfgraph/_native.") for n in names), names
        if "-win_amd64" in path.name:
            libraries = [n for n in names if n.startswith("okfgraph.libs/")]
            assert any("libssl" in n for n in libraries), libraries
            assert any("libcrypto" in n for n in libraries), libraries
        assert any(n.endswith("entry_points.txt") for n in names), names
        assert "Description-Content-Type: text/markdown" in meta, meta
        assert "Project-URL: Homepage," in meta, meta
    print(f"OK: {path.name} bundles the facade, native core and PDF plugin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="okfgraph-*.whl")
    args = ap.parse_args()
    check_main(wheel_path(args.main))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
