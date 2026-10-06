"""Canonical admin ops (CLI + Python; MCP excluded by policy, §6).

One implementation per op here — import, health, hygiene. Return shapes:

* ``import_bundle`` → ``{"concept_ids", "images": {id: count}}``
* ``import_file`` → ``{"concept_id", "images": count}``
* ``doctor`` → ``{"report", "fixed"}`` (``strict`` raises
  ``DOCTOR_FINDINGS`` with that result on ``err.data``)
* ``diff`` / ``diff_dirs`` → the report dict; differences raise
  ``DIFF_DIFFERENT`` with the report on ``err.data``
* ``lint`` → the lint report; errors raise ``LINT_ERRORS`` (report on
  ``err.data``)
* ``produce`` → ``{"producer", "files", "root", "prefix", "lint"}``
* ``reindex`` → ``{"rebuilt": bool}``
* ``list_deleted`` → rows; ``recover_deleted`` → ``{"concept_id",
  "recovered"}``; ``purge_deleted`` → ``{"purged": n}``
* ``detach`` → the detach report (verbatim)
* ``list_broken_links`` → rows; ``repair_links`` → ``{"repaired": n}``
* ``list_images`` → rows; ``get_image`` → row with base64 ``data``
"""
from __future__ import annotations

from pathlib import Path

from okfgraph.errors import OKFError

__all__ = ["AdminOps"]


def _require_dir(path, *, op: str, name: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise OKFError(
            "FILE_NOT_FOUND",
            f"not a bundle directory: {path}",
            op=op,
            fields={name: str(path)},
        )
    return path


def _diff_outcome(result: dict) -> dict:
    if not result["identical"]:
        raise OKFError(
            "DIFF_DIFFERENT",
            "structures differ (report in data)",
            op="diff",
            data=result,
        )
    return result


class AdminOps:
    """Mixin for :class:`~okfgraph.OKFRouter`."""

    # ------------------------------------------------------------------
    # init / model_info
    # ------------------------------------------------------------------

    def init(self) -> dict:
        """Report the adopted pins, then close the router.

        CLI ``okf init`` semantics: constructing the router created the
        schema; this checkpoints and releases the file lock. The router is
        unusable afterwards — Python callers rarely need this op.
        """
        pins = {
            "db_path": str(self.db_path),
            "embedding_dim": self.embedding_dim,
            "model_id": self.model_id,
        }
        self.close()
        return pins

    @staticmethod
    def model_info(model_id=None, cache_dir=None, *, device="auto",
                   precision=None) -> dict:
        """Model cache status without loading the model (static).

        Wraps ``embroider.cache_info`` (B1-a): the returned ``repo`` is
        the precision-specific mirror the engine would really open, and
        ``precision="auto"`` follows the device.
        """
        from okfgraph.components.embedding import EmbeddingEngine
        return EmbeddingEngine.model_info(model_id=model_id, cache_dir=cache_dir,
                                          device=device, precision=precision)

    # ------------------------------------------------------------------
    # import
    # ------------------------------------------------------------------

    def import_bundle(self, bundle_path=None, *, batch_size: int = 32,
                      mode: str = "text", prune_missing: bool = False,
                      force: bool = False, alias=None) -> dict:
        """Import an entire bundle directory (default: every configured root).

        ``alias`` stays Python-only (PDF work-dir namespaces). Returns
        ``{"concept_ids", "images": {id: count}}``.
        """
        if bundle_path is not None:
            bundle_path = _require_dir(bundle_path, op="import_bundle",
                                       name="bundle_path")
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
        """Import a single OKF file. Missing files raise ``FILE_NOT_FOUND``."""
        file_path = Path(file_path)
        if not file_path.is_file():
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

        Returns ``{"report", "fixed"}`` (``fixed`` is None without
        ``fix``). With ``strict``, findings raise ``DOCTOR_FINDINGS``
        (exit 1) with that same result on ``err.data``.
        """
        fixed = self.doctor_mgr.fix() if fix else None
        report = self.doctor_mgr.diagnose(stale_days=stale_days)
        result = {"report": report, "fixed": fixed}
        if strict and report["findings"]:
            raise OKFError(
                "DOCTOR_FINDINGS",
                f"doctor found {len(report['findings'])} finding(s)",
                op="doctor",
                fields={"findings": len(report["findings"]),
                        "score": report["score"]},
                data=result,
            )
        return result

    @staticmethod
    def diff_dirs(old, new) -> dict:
        """Snapshot diff of two bundle directories (router-free, static)."""
        from okfgraph.components.diff import DiffManager
        old = _require_dir(old, op="diff", name="old")
        new = _require_dir(new, op="diff", name="new")
        return _diff_outcome(DiffManager(None).diff_dirs(old, new))

    def diff(self, old=None, new=None) -> dict:
        """Structural diff: snapshot (two dirs) or drift (graph vs one dir,
        or vs the configured roots when no side is given)."""
        if old and new:
            return self.diff_dirs(old, new)
        side = old or new
        if side:
            side = _require_dir(side, op="diff", name="old" if old else "new")
        try:
            result = self.diff_mgr.diff_db_dir(side)
        except ValueError as exc:
            # File-free router with no explicit side: same refusal, typed.
            raise OKFError("BAD_VALUE", str(exc), op="diff") from exc
        return _diff_outcome(result)

    # ------------------------------------------------------------------
    # hygiene
    # ------------------------------------------------------------------

    @staticmethod
    def lint(bundle_dir=".") -> dict:
        """Pre-import bundle gate; router-free. Errors → ``LINT_ERRORS``
        with the full report on ``err.data`` (doctor-style outcome)."""
        from okfgraph.components.lint import lint_bundle
        bundle_dir = _require_dir(bundle_dir, op="lint", name="bundle_dir")
        report = lint_bundle(bundle_dir)
        if report["errors"]:
            raise OKFError(
                "LINT_ERRORS",
                f"{len(report['errors'])} lint error(s)",
                op="lint",
                fields={"errors": len(report["errors"])},
                data=report,
            )
        return report

    @staticmethod
    def produce(source_type: str, source_path=None, output_dir=".",
                *, prefix=None, overwrite: bool = False) -> dict:
        """Generate a bundle from a data source, then lint it.

        Returns ``{"producer", "files", "root", "prefix", "lint"}``; lint
        errors in the produced bundle are reported, not raised (inspect
        ``lint["clean"]``). Unknown producers raise ``BAD_VALUE``;
        unreadable sources raise ``FILE_NOT_FOUND``.
        """
        from okfgraph.components.lint import lint_bundle
        from okfgraph.components.producers import PRODUCERS, producer_for
        if source_type not in PRODUCERS:
            raise OKFError(
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
        except OKFError:
            raise
        except FileNotFoundError as exc:
            raise OKFError(
                "FILE_NOT_FOUND",
                str(exc),
                op="produce",
                fields={"source_path": str(source_path)},
            ) from exc
        except ValueError as exc:
            # Unreadable/empty source or a bad prefix: the producer's
            # refusals are caller-facing, not bugs.
            raise OKFError(
                "BAD_VALUE",
                str(exc),
                op="produce",
                fields={"source_path": str(source_path), "prefix": prefix},
            ) from exc
        return {
            "producer": bundle.producer,
            "files": bundle.files,
            "root": str(bundle.root / bundle.prefix),
            "prefix": bundle.prefix,
            "lint": lint_bundle(out),
        }

    def reindex(self, *, if_dirty: bool = False) -> dict:
        """Rebuild search indexes; ``{"rebuilt": bool}``."""
        rebuilt = self.schema_mgr.reindex(force=not if_dirty)
        return {"rebuilt": bool(rebuilt)}

    # ------------------------------------------------------------------
    # delete / soft-delete trio (D2-a)
    # ------------------------------------------------------------------

    def delete(self, concept_id: str) -> dict:
        """Soft-delete one concept (recoverable for 24h, then purge —
        ``deleted-recover`` / ``deleted-purge`` manage the lifecycle).

        Refuses with ``BAD_VALUE`` while a source file that backs the
        concept still exists — the next ``import --all`` would resurrect
        it, so graph and bundle must agree first: delete or move the
        file, then run ``import --all --prune-missing``. Rootless
        concepts (thoughts, file-less graph contents) delete freely.
        NOT on MCP (destructive; agents get ``ingest`` only).
        """
        from okfgraph.errors import UsageError, StateError
        concept = self.search_engine.get_by_id(concept_id)
        if concept is None:
            raise UsageError(
                "BAD_VALUE",
                f"unknown concept '{concept_id}'",
                op="delete",
                fields={"concept_id": concept_id},
                remedy="list ids with a search first",
            )
        # FileHash maps bundle-relative keys (with @alias/ prefixes on
        # named roots) to concept ids; any still-existing source means
        # delete is the wrong tool here.
        rows = self.conn.execute(
            "MATCH (f:FileHash {concept_id: $cid}) RETURN f.path AS p",
            {"cid": concept_id},
        ).rows_as_dict().get_all()
        alive = []
        for row in rows:
            rel = str(row.get("p") or "")
            if not rel:
                continue
            for existing in self._resolve_file_paths(rel):
                if existing.exists():
                    alive.append(str(existing))
        if alive:
            raise UsageError(
                "BAD_VALUE",
                f"'{concept_id}' is backed by existing source file(s): "
                f"{', '.join(alive)}",
                op="delete",
                fields={"concept_id": concept_id, "sources": alive},
                remedy="delete or move the file(s) from the bundle, then "
                       "run 'okf import --all --prune-missing' to drop "
                       "the concept",
            )
        if not self.purge_mgr._soft_delete_concept(concept_id):
            raise StateError(
                "NOT_RECOVERABLE",
                f"concept '{concept_id}' vanished mid-operation",
                op="delete",
            )
        from datetime import datetime, timedelta
        until = datetime.now() + timedelta(
            seconds=self.purge_mgr.SOFT_DELETE_WINDOW)
        return {
            "concept_id": concept_id,
            "deleted": True,
            "recoverable_until": until.isoformat(),
        }

    def _resolve_file_paths(self, rel: str) -> list:
        """Map a FileHash key (bundle-relative, optionally ``@alias/``
        prefixed) to its absolute candidate paths across roots."""
        out = []
        key = rel.replace("\\", "/")
        if key.startswith("@") and "/" in key:
            alias, native = key[1:].split("/", 1)
            root = self.import_mgr.roots.get(alias)
            if root is not None:
                out.append(Path(root) / native)
            # Unknown alias: also try the primary root with the key as-is
            primary = self.import_mgr.bundle_root
            if primary is not None:
                out.append(Path(primary) / key)
        else:
            primary = self.import_mgr.bundle_root
            if primary is not None:
                out.append(Path(primary) / key)
        return out

    def list_deleted(self) -> list:
        """Soft-deleted concepts with recovery status."""
        return self.purge_mgr.list_deleted_concepts()

    def recover_deleted(self, concept_id: str) -> dict:
        """Recover a soft-deleted concept; failure → ``NOT_RECOVERABLE``."""
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
        """Links whose target concept does not exist: ``[{source, target}]``."""
        return self.import_mgr.list_broken_links()

    def repair_links(self) -> dict:
        """Re-resolve broken links against current concepts."""
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
                remedy="search first to find IDs",
            )
        return self.image_mgr.list_images(concept_id)

    def get_image(self, asset_id: str) -> dict:
        """Fetch one image: metadata plus base64 ``data`` (None when the
        asset has no stored bytes)."""
        import base64
        row = self.image_mgr.get_image_data(asset_id)
        if row is None:
            raise OKFError(
                "UNKNOWN_ASSET",
                f"image asset '{asset_id}' does not exist",
                op="get_image",
                fields={"asset_id": asset_id},
                remedy="list_images(concept_id) names the asset ids",
            )
        data = row.pop("data", None)
        row["data"] = (base64.b64encode(bytes(data)).decode("ascii")
                       if data is not None else None)
        return row
