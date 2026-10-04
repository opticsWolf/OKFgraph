"""Canonical operations, assembled as mixins for :class:`~okfgraph.OKFRouter`.

Each operation has exactly one implementation, here — the CLI, MCP and
Python surfaces adapt to it (see ``docs/surface-unification-plan.md``).
"""
from __future__ import annotations

from .ingest import IngestOps
from .query import QueryOps

__all__ = ["IngestOps", "QueryOps"]
