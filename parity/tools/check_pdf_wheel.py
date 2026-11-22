#!/usr/bin/env python3
"""End-to-end PDF add-on wheel smoke: locator, auto-discovery and conversion."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "parity/out"
os.chdir(OUT)
os.environ.pop("OKFGRAPH_PDF_PLUGIN", None)

import okfgraph_pdf  # noqa: E402
from okfgraph import OKFRouter  # noqa: E402


def main() -> int:
    plugin = okfgraph_pdf.plugin_path()
    assert plugin and Path(plugin).is_file(), plugin
    assert okfgraph_pdf.version() == "0.16.0", okfgraph_pdf.version()

    db = OUT / "pdf-wheel-smoke.db"
    shutil.copyfile(ROOT / "parity/fixtures/oracle-v10-import-minimal.db", db)
    output = OUT / "pdf-wheel-smoke-out"
    shutil.rmtree(output, ignore_errors=True)

    router = OKFRouter(db_path=str(db), bundle_root="../corpus/minimal")
    assert os.environ.get("OKFGRAPH_PDF_PLUGIN") == plugin, os.environ.get(
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
    print("OK: okfgraph-pdf wheel is auto-discovered and converts a real PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
