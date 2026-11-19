"""Packaging-only glue for the optional native PDF add-on.

The Rust core resolves plugins through ``OKFGRAPH_PDF_PLUGIN``; this helper
sets it from the separate ``okfgraph-pdf`` distribution when present. It is
not part of the public Python API and contains no conversion logic.
"""
from __future__ import annotations

import os
from importlib.util import find_spec

PLUGIN_ENV = "OKFGRAPH_PDF_PLUGIN"


def resolve_plugin() -> None:
    """Point the native core at the optional plugin, if installed."""
    if os.environ.get(PLUGIN_ENV):
        return
    if find_spec("okfgraph_pdf") is None:
        return
    from okfgraph_pdf import plugin_path  # type: ignore[import-not-found]

    path = plugin_path()
    if path:
        os.environ[PLUGIN_ENV] = path
