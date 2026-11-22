#!/usr/bin/env python3
"""End-to-end PDF smoke: the bundled plugin is discovered and converts."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "parity/out"
os.chdir(OUT)
os.environ.pop("OKFGRAPH_PDF_PLUGIN", None)

from okfgraph import OKFRouter  # noqa: E402
from okfgraph._pdf import bundled_plugin  # noqa: E402


def main() -> int:
    plugin = bundled_plugin()
    assert plugin is not None, "no okfgraph/_plugin library in the wheel"
    assert plugin.is_file(), plugin

    db = OUT / "pdf-wheel-smoke.db"
    shutil.copyfile(ROOT / "parity/fixtures/oracle-v10-import-minimal.db", db)
    output = OUT / "pdf-wheel-smoke-out"
    shutil.rmtree(output, ignore_errors=True)

    router = OKFRouter(db_path=str(db), bundle_root="../corpus/minimal")
    assert os.environ.get("OKFGRAPH_PDF_PLUGIN") == str(plugin), os.environ.get(
        "OKFGRAPH_PDF_PLUGIN"
    )
    report = router.ingest(
        kind="pdf",
        pdf_path=str(ROOT / "parity/fixtures/sample.pdf"),
        output_dir=str(output),
        auto_import=False,
    )
    assert report["page_count"] == 1, report
    assert (output / "sample.md").is_file(), sorted(os.listdir(output))
    router.close()
    print("OK: the bundled PDF plugin is auto-discovered and converts a real PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
