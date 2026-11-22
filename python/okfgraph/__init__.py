"""Thin public facade for the Rust OKFgraph 0.16.1 core.

Only documented API objects are exported here. The native extension is loaded
lazily so import failures remain reportable; there is no Python fallback.
See ``docs/python-binding.md`` and ``docs/quickref.md``.
"""
from __future__ import annotations

_EXPORTS = {
    "OKFRouter": ("okfgraph.router", "OKFRouter"),
    "OKFError": ("okfgraph.errors", "OKFError"),
    "UsageError": ("okfgraph.errors", "UsageError"),
    "StateError": ("okfgraph.errors", "StateError"),
    "OutcomeError": ("okfgraph.errors", "OutcomeError"),
    "ConceptModel": ("okfgraph.models", "ConceptModel"),
    "ChunkModel": ("okfgraph.models", "ChunkModel"),
    "ImageAssetModel": ("okfgraph.models", "ImageAssetModel"),
    "OKFClient": ("okfgraph.client", "OKFClient"),
    "discover": ("okfgraph.client", "discover"),
    "serve_file_for": ("okfgraph.client", "serve_file_for"),
    "SETTINGS": ("okfgraph.settings", "SETTINGS"),
    "DEFAULT_MODEL_ID": ("okfgraph.settings", "DEFAULT_MODEL_ID"),
    "DEFAULT_DB_PATH": ("okfgraph.settings", "DEFAULT_DB_PATH"),
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
