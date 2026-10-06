"""Canonical export ops: ``export_bundle`` and ``export_concept``.

One implementation per op (§3.6): both take ``output_dir`` — the v1
"`--output` maps to output_dir/output_path by branch" exception is gone.
The component keeps an ``output_path``-based helper internally.
"""
from __future__ import annotations

from pathlib import Path

from okfgraph.errors import OKFError, UsageError

__all__ = ["ExportOps"]

_FLAVORS = ("okf", "obsidian")


class ExportOps:
    """Mixin for :class:`~okfgraph.OKFRouter`."""

    def export_bundle(
        self,
        output_dir,
        *,
        directory_id=None,
        concept_type=None,
        tags=None,
        flavor: str = "okf",
    ) -> dict:
        """Export concepts back to an OKF bundle directory.

        Returns ``{"output_dir", "concept_ids", "flavor"}`` (a bare list
        in 0.9). Filters: ``directory_id`` subtree, ``concept_type``,
        ``tags`` (all must match).
        """
        if flavor not in _FLAVORS:
            raise UsageError(
                "BAD_VALUE",
                f"flavor must be 'okf' or 'obsidian', got '{flavor}'",
                op="export_bundle",
                fields={"flavor": flavor},
            )
        ids = self.export_mgr.export_bundle(
            output_dir=Path(output_dir),
            directory_id=directory_id,
            concept_type=concept_type,
            tags=tags,
            flavor=flavor,
        )
        return {"output_dir": str(Path(output_dir)), "concept_ids": ids,
                "flavor": flavor}

    def export_concept(self, concept_id: str, *, output_dir, flavor: str = "okf") -> dict:
        """Export a single concept to ``<output_dir>/<concept_id>.md``.

        Returns ``{"concept_id", "path"}``. Unknown concept raises
        ``UNKNOWN_CONCEPT`` (was FileNotFoundError).
        """
        if not output_dir:
            raise UsageError(
                "MISSING_PARAM",
                "export_concept requires output_dir",
                op="export_concept",
                fields={"concept_id": concept_id},
            )
        if flavor not in _FLAVORS:
            raise UsageError(
                "BAD_VALUE",
                f"flavor must be 'okf' or 'obsidian', got '{flavor}'",
                op="export_concept",
                fields={"flavor": flavor},
            )
        if self.get_by_id(concept_id) is None:
            raise OKFError(
                "UNKNOWN_CONCEPT",
                f"concept '{concept_id}' does not exist",
                op="export_concept",
                fields={"concept_id": concept_id},
            )
        out_dir = Path(output_dir)
        file_path = out_dir / f"{concept_id}.md"
        self.export_mgr.export_to_okf(concept_id, file_path, flavor=flavor)
        return {"concept_id": concept_id, "path": str(file_path)}
