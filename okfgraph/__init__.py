"""OKF Knowledge Graph — Ladybug-backed knowledge system with ONNX + Jina v5 embeddings."""

# Lazy exports (PEP 562): importing a light submodule (okfgraph.crash, the
# okf start-up guard in okfgraph._entry) must not drag in the router and
# every native dependency, or an import-time failure there could never be
# caught and reported.
_EXPORTS = {
    "ConceptModel": ("okfgraph.models", "ConceptModel"),
    "OKFRouter": ("okfgraph.router", "OKFRouter"),
    "cli_main": ("okfgraph.cli", "main"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'okfgraph' has no attribute {name!r}") from None
    import importlib
    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value
    return value
