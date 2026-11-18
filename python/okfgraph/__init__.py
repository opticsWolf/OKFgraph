"""Thin Python facade for the Rust OKFgraph core (M0 scaffold).

The full public API lands in M5. This module deliberately avoids importing
`_native` at module import time so start-up failures remain reportable.
"""
from __future__ import annotations

_EXPORTS = {
    "OKFRouter": ("okfgraph.router", "OKFRouter"),
    "cli_main": ("okfgraph.cli", "main"),
}

__all__ = ["version", "schema_version", "op_names", *_EXPORTS]


def __getattr__(name: str):
    if name in ("version", "schema_version", "op_names"):
        from . import _native

        value = getattr(_native, name)
        globals()[name] = value
        return value
    try:
        module, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'okfgraph' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value
    return value
