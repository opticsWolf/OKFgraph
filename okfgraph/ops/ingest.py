"""Canonical ingest op: ``ingest(kind, …)`` — markdown, PDF, thoughts.

The one implementation lives on
:class:`~okfgraph.components.ingest.IngestManager.ingest`; the op exposes
it on the router facade (one dispatch instead of five method spellings,
§3.5). Required-param gaps raise ``MISSING_PARAM``; absent paths raise
``FILE_NOT_FOUND``.
"""
from __future__ import annotations

__all__ = ["IngestOps"]


class IngestOps:
    """Mixin for :class:`~okfgraph.OKFRouter`."""

    def ingest(self, kind: str, **kwargs):
        """Add content to the graph. See ``IngestManager.ingest`` for the
        full contract: kind={md|pdf|thoughts}, per-kind required params,
        ``auto_import`` (default True), ``prune_missing``, ``force``."""
        return self.ingest_mgr.ingest(kind, **kwargs)
