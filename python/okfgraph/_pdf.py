"""Packaging-only glue for the bundled native PDF converter.

The PDF plugin ships inside the ``okfgraph`` package as
``okfgraph/_plugin/<library>``. The Rust core resolves plugins through
``OKFGRAPH_PDF_PLUGIN``; this helper points that variable at the bundled
library. It is not part of the public Python API and contains no conversion
logic.
"""
from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

PLUGIN_ENV = "OKFGRAPH_PDF_PLUGIN"
LIBRARY_NAMES = (
    "okfgraph_pdf.dll",
    "libokfgraph_pdf.so",
    "libokfgraph_pdf.dylib",
)


def bundled_plugin() -> Path | None:
    """Return the plugin library shipped in this wheel, if any."""
    directory = Path(__file__).resolve().parent / "_plugin"
    for name in LIBRARY_NAMES:
        path = directory / name
        if path.is_file():
            return path
    return None


def resolve_plugin() -> None:
    """Point the native core at the bundled plugin."""
    if os.environ.get(PLUGIN_ENV):
        return
    path = bundled_plugin()
    if path is not None:
        os.environ[PLUGIN_ENV] = str(path)
        return
    # A separately installed add-on (0.16 pre-bundle builds) still works.
    if find_spec("okfgraph_pdf") is None:
        return
    from okfgraph_pdf import plugin_path  # type: ignore[import-not-found]

    legacy = plugin_path()
    if legacy:
        os.environ[PLUGIN_ENV] = legacy
