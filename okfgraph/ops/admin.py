"""Canonical admin ops (CLI + Python; MCP excluded by policy, §6).

One implementation per op here — import, health, hygiene. Return shapes:

* ``import_bundle`` → ``{"concept_ids", "images": {id: count}}``
* ``import_file`` → ``{"concept_id", "images": count}``
* ``doctor`` → ``{"report", "fixed"?}`` (``strict`` raises
  ``DOCTOR_FINDINGS`` with the report still available in ``fields``)
* ``diff`` → the report dict; different-graph/dirs raises ``DIFF_DIFFERENT``
* ``lint`` → the lint report; errors raise ``LINT_ERRORS``
* ``produce`` → ``{"producer", "files", "root", "prefix", "lint"}``
* ``reindex`` → ``{"rebuilt": bool}``
* ``list_deleted`` → rows; ``recover_deleted`` → row; ``purge_deleted`` →
  ``{"purged": n}``
* ``detach`` → the detach report (verbatim)
* ``repair_links`` → ``{"repaired": n}``
"""
from __future__ import annotations

from pathlib import Path

from okfgraph.errors import OKFError, UsageError

__all__ = ["AdminOps"]


class AdminOps:
    """Mixin for :class:`~okfgraph.OKFRouter`."""

    # ------------------------------------------------------------------
    # init / model_info
    # ------------------------------------------------------------------

    def init(self) -> dict:
        """Construct (and close) the router; returns the adopted pins."""
        self.close()
        return {
            "db_path": str(self.db_path),
            "embedding_dim": self.embedding_dim,
            "model_id": self.model_id,
        }

    @staticmethod
    def model_info(model_id=None, cache_dir=None) -> dict:
        """Model cache status without loading the model (static)."""
        from okfgraph.components.embedding import EmbeddingEngine
        return EmbeddingEngine.model_info(model_id=model_id, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # import
    # ------------------------------------------------------------------

    def import_bundle(self, bundle_path=None, *, batch_size: int = 32,
                      mode: str = "text", prune_missing: bool = False,
                      force: bool = False, alias=None) -> dict:
        """Import an entire bundle directory (default: the mirrored root).

        ``alias`` stays Python-only (PDF work-dir namespaces). Returns
        ``{"concept_ids", "images": {id: count}}``.
        """
        ids = self.import_mgr.import_bundle(
            bundle_path,
            batch_size=batch_size,
            mode=mode,
            prune_missing=prune_missing,
            force=force,
            alias=alias,
        )
        images = {cid: len(self.image_mgr.list_images(cid)) for cid in ids}
        return {"concept_ids": ids, "images": images}

    def import_file(self, file_path, *, mode: str = "text", force: bool = False) -> dict:
        """Import a single OKF file. Missing files raise ``FILE_NOT_FOUND``
        (today: a logged warning and exit 0)."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise OKFError(
                "FILE_NOT_FOUND",
                f"file not found: {file_path}",
                op="import_file",
                fields={"file_path": str(file_path)},
            )
        cid = self.import_mgr.import_from_okf(file_path, mode=mode, force=force)
        return {"concept_id": cid, "images": len(self.image_mgr.list_images(cid))}

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    def doctor(self, *, stale_days: int = 365, fix: bool = False,
               strict: bool = False) -> dict:
        """Scored health scan; ``fix`` applies safe repairs first.

        Returns ``{"report", "fixed"?}``. With ``strict``, findings raise
        ``DOCTOR_FINDINGS`` (exit 1) with the report in ``fields``.
        """
        result: dict = {}
        if fix:
            result["fixed"] = self.doctor_mgr.fix()
        report = self.doctor_mgr.diagnose(stale_days=stale_days)
        result["report"] = report
        if strict and report["findings"]:
            raise OKFError(
                "DOCTOR_FINDINGS",
                f"doctor found {len(report['findings'])} finding(s)",
                op="doctor",
                fields={"report": report, "fixed": result.get("fixed")},
            )
        return result

    def diff(self, old=None, new=None) -> dict:
        """Structural diff; snapshot (two dirs, router-free) or drift."""
        from okfgraph.components.diff import DiffManager

        if old and new and Path(old).is_dir() and Path(new).is_dir():
            result = DiffManager(None).diff_dirs(Path(old), Path(new))
        elif old and new:
            raise UsageError(
                "BAD_VALUE",
                "diff needs two bundle directories (or one side + --db/--bundle-root)",
                op="diff",
                fields={"old": str(old), "new": str(new)},
            )
        else:
            side = Path(old or new) if (old or new) else None
            if side is not None and not side.is_dir():
                raise UsageError(
                    "BAD_VALUE",
                    f"not a bundle directory: {side}",
                    op="diff",
                    fields={"side": str(side)},
                )
            try:
                result = self.diff_mgr.diff_db_dir(side)
            except ValueError as exc:
                # File-free router with no explicit side: same refusal,
                # typed (§4).
                raise UsageError("BAD_VALUE", str(exc), op="diff") from exc
        if not result["identical"]:
            raise OKFError(
                "DIFF_DIFFERENT",
                "structures differ (see fields.report)",
                op="diff",
                fields={"report": result},
            )
        return result

    # ------------------------------------------------------------------
    # hygiene
    # ------------------------------------------------------------------

    @staticmethod
    def lint(bundle_dir=".") -> dict:
        """Pre-import bundle gate; router-free. Errors → ``LINT_ERRORS``
        with the full report in ``fields`` (doctor-style outcome)."""
        from okfgraph.components.lint import lint_bundle
        report = lint_bundle(Path(bundle_dir))
        if report["errors"]:
            raise OKFError(
                "LINT_ERRORS",
                f"{len(report['errors'])} lint error(s)",
                op="lint",
                fields={"errors": report["errors"], "report": report},
            )
        return report

    @staticmethod
    def produce(source_type: str, source_path=None, output_dir=".",
                *, prefix=None, overwrite: bool = False) -> dict:
        """Generate a bundle from a data source, then lint it.

        Returns ``{"producer", "files", "root", "prefix", "lint"}``.
        Unknown producers raise ``BAD_VALUE``; unreadable sources raise
        ``FILE_NOT_FOUND``.
        """
        from okfgraph.components.lint import lint_bundle
        from okfgraph.components.producers import PRODUCERS, producer_for
        if source_type not in PRODUCERS:
            raise UsageError(
                "BAD_VALUE",
                f"unknown producer {source_type!r}",
                op="produce",
                fields={"source_type": source_type,
                        "available": sorted(PRODUCERS)},
            )
        out = Path(output_dir or ".")
        try:
            bundle = producer_for(source_type).produce(
                source_path, out, prefix=prefix, overwrite=overwrite,
            )
        except FileNotFoundError as exc:
            raise OKFError(
                "FILE_NOT_FOUND",
                str(exc),
                op="produce",
                fields={"source_path": str(source_path)},
            ) from exc
        report = lint_bundle(out)
        return {
            "producer": bundle.producer,
            "files": bundle.files,
            "root": str(bundle.root / bundle.prefix),
            "prefix": bundle.prefix,
            "lint": report,
        }

    def reindex(self, *, if_dirty: bool = False) -> dict:
        """Rebuild search indexes; ``{"rebuilt": bool}``."""
        rebuilt = self.schema_mgr.reindex(force=not if_dirty)
        return {"rebuilt": bool(rebuilt)}

    def list_deleted(self) -> list:
        """Soft-deleted concepts with recovery status."""
        return self.purge_mgr.list_deleted_concepts()

    def recover_deleted(self, concept_id: str) -> dict:
        """Recover a soft-deleted concept; failure → ``NOT_RECOVERABLE``"""
        if not self.purge_mgr.recover_deleted(concept_id):
            raise OKFError(
                "NOT_RECOVERABLE",
                f"concept '{concept_id}' not found or past recovery window",
                op="recover_deleted",
                fields={"concept_id": concept_id},
            )
        return {"concept_id": concept_id, "recovered": True}

    def purge_deleted(self, *, older_than: int = None) -> dict:
        """Permanently delete expired soft-deleted concepts."""
        count = self.purge_mgr.purge_deleted_concepts(older_than=older_than)
        return {"purged": count}

    def detach(self, *, bundle_path=None, verify: bool = True,
               force: bool = False) -> dict:
        """End the mirror relationship; the DB becomes the artifact."""
        return self.import_mgr.detach(
            bundle_path=Path(bundle_path) if bundle_path else None,
            verify=verify,
            force=force,
        )

    # ------------------------------------------------------------------
    # links
    # ------------------------------------------------------------------

    def list_broken_links(self) -> list:
        return self.import_mgr.list_broken_links()

    def repair_links(self) -> dict:
        return {"repaired": self.import_mgr.repair_links()}

    # ------------------------------------------------------------------
    # images (Q5 → B: full ops on all three surfaces)
    # ------------------------------------------------------------------

    def list_images(self, concept_id: str) -> list:
        """List image assets attached to a concept."""
        if self.get_by_id(concept_id) is None:
            raise OKFError(
                "UNKNOWN_CONCEPT",
                f"concept '{concept_id}' does not exist",
                op="list_images",
                fields={"concept_id": concept_id},
            )
        return self.image_mgr.list_images(concept_id)

    def get_image(self, asset_id: str) -> dict:
        """Fetch one image: metadata rows + base64 ``data`` (MCP shape)."""
        import base64
        row = self.image_mgr.get_image_data(asset_id)
        if row is None:
            raise OKFError(
                "UNKNOWN_ASSET",
                f"image asset '{asset_id}' does not exist",
                op="get_image",
                fields={"asset_id": asset_id},
            )
        data = row.pop("data", None)
        if data is not None:
            row["data"] = base64.b64encode(bytes(data)).decode("ascii")
        return row
