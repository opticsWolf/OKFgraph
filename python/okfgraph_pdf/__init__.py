"""Locator for the optional native PDF add-on (packaging glue only)."""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["plugin_path", "version", "PLUGIN_ENV"]


PLUGIN_ENV = "OKFGRAPH_PDF_PLUGIN"
_VERSION = "0.16.0"


def version() -> str:
    return _VERSION


def plugin_path() -> str | None:
    """Path to the loaded plugin, from the explicit env var or the wheel."""
    explicit = os.environ.get(PLUGIN_ENV)
    if explicit:
        return explicit
    for suffix in ("okfgraph_pdf.dll", "libokfgraph_pdf.so", "libokfgraph_pdf.dylib"):
        candidate = Path(__file__).with_name(suffix)
        if candidate.is_file():
            return str(candidate)
    return None
