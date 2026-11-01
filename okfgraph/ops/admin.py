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
                      force: bool = False, alias=None,
                      rebuild: str = "auto") -> dict:
        """Import an entire bundle directory (default: every configured root).

        ``alias`` stays Python-only (PDF work-dir namespaces). ``rebuild``
        (0.11.0, O8) is the index step: ``auto`` | ``defer`` | ``force``.
        Returns ``{"concept_ids", "images": {id: count}, "duplicates",
        "moved", "delta": {changed_dirs, queued, actually_changed}}``.
        ``duplicates`` / ``moved`` report frontmatter-identity refusals and
        moves (O3); ``delta`` describes the last delta pass (empty for
        isolated or work-dir imports).
        """
        if bundle_path is not None:
            bundle_path = _require_dir(bundle_path, op="import_bundle",
                                       name="bundle_path")
        with self.import_mgr.rebuild(rebuild):
            ids = self.import_mgr.import_bundle(
                bundle_path,
                batch_size=batch_size,
                mode=mode,
                prune_missing=prune_missing,
                force=force,
                alias=alias,
            )
        images = {cid: len(self.image_mgr.list_images(cid)) for cid in ids}
        report = self.import_mgr.last_import_report or {}
        delta = dict(getattr(self.import_mgr, "last_delta_stats", {}) or {})
        return {"concept_ids": ids, "images": images,
                "duplicates": list(report.get("duplicates", [])),
                "moved": list(report.get("moved", [])), "delta": delta}

    def import_files(self, paths, *, mode: str = "text", force: bool = False,
                     identity_root=None, rebuild: str = "auto") -> dict:
        """Import exactly these files, one index step at the end (0.11.0, O8).

        Returns ``{"concept_ids", "images": {id: count}}``. Missing files
        raise ``FILE_NOT_FOUND`` before anything is written.
        """
        with self.import_mgr.rebuild(rebuild):
            ids = self.import_mgr.import_files(
                [Path(p) for p in paths], mode=mode, force=force,
                identity_root=identity_root)
        images = {cid: len(self.image_mgr.list_images(cid)) for cid in ids}
        return {"concept_ids": ids, "images": images}

    def import_file(self, file_path, *, mode: str = "text", force: bool = False,
                      identity_root=None, rebuild: str = "auto") -> dict:
        """Import a single OKF file. Missing files raise ``FILE_NOT_FOUND``.

        ``identity_root`` (0.10.1 A1) is an explicit hierarchy base for
        positional imports: the file must live under it, and the concept ID
        is minted relative to it. ``None`` keeps the legacy configured-root
        lookup with the bare-stem fallback (plus a hierarchy-dropped
        warning). Return shape is unchanged.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise OKFError(
                "FILE_NOT_FOUND",
                f"file not found: {file_path}",
                op="import_file",
                fields={"file_path": str(file_path)},
            )
        with self.import_mgr.rebuild(rebuild):
            cid = self.import_mgr.import_from_okf(
                file_path, mode=mode, force=force, identity_root=identity_root)
        return {"concept_id": cid, "images": len(self.image_mgr.list_images(cid))}

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    def doctor(self, *, stale_days: int = 365, fix: bool = False,
               strict: bool = False, cache_dir: str | None = None) -> dict:
        """Scored health scan; ``fix`` applies safe repairs first.

        ``cache_dir`` pins the hub-cache root the informational cache
        sections inspect (default: the effective HF hub cache).

        Returns ``{"report", "fixed"}`` (``fixed`` is None without
        ``fix``). With ``strict``, findings raise ``DOCTOR_FINDINGS``
        (exit 1) with that same result on ``err.data``.
        """
        fixed = self.doctor_mgr.fix() if fix else None
        report = self.doctor_mgr.diagnose(stale_days=stale_days,
                                          cache_dir=cache_dir)
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
    def lint(bundle_dir=".", identity: str = "path") -> dict:
        """Pre-import bundle gate; router-free. Errors → ``LINT_ERRORS``
        with the full report on ``err.data`` (doctor-style outcome).
        ``identity`` (0.11.0) must match the graph's ``[import] identity``."""
        from okfgraph.components.lint import lint_bundle
        bundle_dir = _require_dir(bundle_dir, op="lint", name="bundle_dir")
        report = lint_bundle(bundle_dir, identity=identity)
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
        from okfgraph.components.producers import (
            available_producers, producer_for, run_producer)
        registry = available_producers()
        if source_type not in registry:
            raise OKFError(
                "BAD_VALUE",
                f"unknown producer {source_type!r}",
                op="produce",
                fields={"source_type": source_type,
                        "available": sorted(registry)},
            )
        out = Path(output_dir or ".")
        try:
            bundle = run_producer(
                producer_for(source_type), source_path, out,
                prefix=prefix, overwrite=overwrite,
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

    def sync(self, producer: str, source=None, *, output_dir=None,
             prefix=None, rebuild: str = "auto") -> dict:
        """Run a producer incrementally and import what it wrote (0.11.0, O10).

        The producer gets the cursor it returned last time (stored per
        producer name, plus ``:prefix`` when one is given), writes only what
        changed into ``output_dir`` (default: the bundle root), and exactly
        ``ProducedBundle.written`` is imported via ``import_files``. The new
        cursor is stored only after that import succeeds, so a failed run
        repeats instead of skipping. Returns ``{"producer", "cursor",
        "written", "concept_ids"}``.

        A producer that predates cursors (no ``written``) has its whole
        ``<prefix>/`` tree imported each run.
        """
        import time as _time
        from okfgraph.components.import_ import is_concept_file
        from okfgraph.components.producers import (
            available_producers, producer_for, run_producer)
        if producer not in available_producers():
            raise OKFError("BAD_VALUE", f"unknown producer {producer!r}", op="sync",
                           fields={"producer": producer,
                                   "available": sorted(available_producers())})
        out = Path(output_dir) if output_dir else self.bundle_root
        if out is None:
            raise OKFError("MISSING_PARAM",
                           "sync needs output_dir (or a router bundle_root)",
                           op="sync", fields={"output_dir": None})
        key = producer + (f":{prefix}" if prefix else "")
        rows = self.conn.execute(
            "MATCH (p:ProducerCursor {name: $n}) RETURN p.cursor AS c",
            {"n": key}).rows_as_dict().get_all()
        cursor = rows[0]["c"] if rows else None
        try:
            bundle = run_producer(producer_for(producer), source, out,
                                  prefix=prefix, overwrite=True, cursor=cursor)
        except FileNotFoundError as exc:
            raise OKFError("FILE_NOT_FOUND", str(exc), op="sync",
                           fields={"source": str(source)}) from exc
        except ValueError as exc:
            raise OKFError("BAD_VALUE", str(exc), op="sync",
                           fields={"source": str(source)}) from exc
        written = list(bundle.written)
        if not written and bundle.cursor is None:
            written = sorted((Path(bundle.root) / bundle.prefix).rglob("*.md"))
        files = [Path(p) for p in written if is_concept_file(Path(p))]
        ids: list = []
        if files:
            ids = self.import_files(files, rebuild=rebuild)["concept_ids"]
        if bundle.cursor is not None:
            self.conn.execute(
                "MERGE (p:ProducerCursor {name: $n}) "
                "SET p.cursor = $c, p.updated_at = $t",
                {"n": key, "c": bundle.cursor, "t": int(_time.time())})
        return {"producer": producer, "cursor": bundle.cursor,
                "written": [str(p) for p in files], "concept_ids": ids}

    @staticmethod
    def setup_extensions(target=None, source=None, *, download: bool = True) -> dict:
        """Fill an offline extension directory (router-free, 0.11.0, O5).

        Returns ``{"target", "copied", "verified", "ok", "toml"}``; point
        ``[database] extension_dir`` at ``target``. Nothing to copy raises
        ``FILE_NOT_FOUND`` naming the missing extension(s).
        """
        from okfgraph.components.extensions import setup_extensions
        try:
            return setup_extensions(
                Path(target) if target else None,
                Path(source) if source else None,
                download=download)
        except FileNotFoundError as exc:
            raise OKFError("FILE_NOT_FOUND", str(exc), op="setup_extensions",
                           fields={"source": str(source) if source else None},
                           remedy="pass --from DIR holding libvector/libfts "
                                  ".lbug_extension from an online machine") from exc

    def reindex(self, *, if_dirty: bool = False) -> dict:
        """Rebuild search indexes; ``{"rebuilt": bool}``."""
        rebuilt = self.schema_mgr.reindex(force=not if_dirty)
        return {"rebuilt": bool(rebuilt)}

    # ------------------------------------------------------------------
    # delete / soft-delete trio (D2-a)
    # ------------------------------------------------------------------

    def delete(self, concept_id: str, *, remove_file: bool = False) -> dict:
        """Soft-delete one concept (recoverable for 24h, then purge —
        ``deleted-recover`` / ``deleted-purge`` manage the lifecycle).

        Without ``remove_file``, refuses with ``BAD_VALUE`` while a source
        file that backs the concept still exists — the next ``import --all``
        would resurrect it, so graph and bundle must agree first.

        ``remove_file=True`` (0.11.0, O11 — what the MCP ``delete`` tool
        uses) keeps graph and bundle in agreement itself: the backing
        file(s) move to ``<root>/.okf-trash/<ts>/`` (a dot-dir, never
        walked), the delta bookkeeping for them retires, and the concept
        is soft-deleted. ``recover_deleted`` moves the files back and
        re-imports them. Rootless concepts delete freely either way.
        """
        from okfgraph.errors import UsageError, StateError
        concept = self.search_engine.get_by_id(concept_id)
        if concept is None:
            # Same code as read/traverse/export_concept/recover (§4 table).
            raise StateError(
                "UNKNOWN_CONCEPT",
                f"concept '{concept_id}' does not exist",
                op="delete",
                fields={"concept_id": concept_id},
                remedy="search first to find IDs",
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
        trashed: list = []
        if alive and remove_file:
            trashed = self._trash_files(concept_id, rows)
            alive = []
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
        if trashed:
            self._remember_trash(concept_id, trashed)
        from datetime import datetime, timedelta
        until = datetime.now() + timedelta(
            seconds=self.purge_mgr.SOFT_DELETE_WINDOW)
        return {
            "concept_id": concept_id,
            "deleted": True,
            "recoverable_until": until.isoformat(),
            "trashed_files": [t["trash"] for t in trashed],
        }

    def _trash_files(self, concept_id: str, rows: list) -> list:
        """Move a concept's backing files into ``.okf-trash`` (O11)."""
        import time as _time
        stamp = _time.strftime("%Y%m%dT%H%M%S")
        moved = []
        for row in rows:
            key = str(row.get("p") or "")
            for src in self._resolve_file_paths(key):
                if not src.exists():
                    continue
                root = None
                for cand in [self.import_mgr.bundle_root, *self.import_mgr.roots.values()]:
                    if cand is not None and src.resolve().is_relative_to(Path(cand).resolve()):
                        root = Path(cand)
                        break
                if root is None:
                    continue
                rel = src.resolve().relative_to(root.resolve())
                dst = root / ".okf-trash" / stamp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                alias = (key[1:].split("/", 1)[0] if key.startswith("@") else "")
                self.import_mgr._detector_for(alias).forget_file(key)
                moved.append({"source": str(src), "trash": str(dst)})
        return moved

    def _remember_trash(self, concept_id: str, trashed: list) -> None:
        """Note trashed files on the DeletedConcept snapshot for recovery."""
        import json as _json
        rows = self.conn.execute(
            "MATCH (d:DeletedConcept {original_id: $id}) "
            "RETURN d.id AS id, d.snapshot AS s ORDER BY d.deleted_at DESC",
            {"id": concept_id},
        ).rows_as_dict().get_all()
        if not rows:
            return
        snap = _json.loads(rows[0]["s"] or "{}")
        snap["trashed_files"] = trashed
        self.conn.execute(
            "MATCH (d:DeletedConcept {id: $id}) SET d.snapshot = $s",
            {"id": rows[0]["id"], "s": _json.dumps(snap, default=str)},
        )

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
        """Recover a soft-deleted concept; failure → ``NOT_RECOVERABLE``.

        0.11.0: files a ``delete(remove_file=True)`` trashed move back and
        are re-imported (hashes and links current again); a graph-only
        concept gets its outgoing links re-resolved, and links that other
        concepts held to it (kept as BrokenLink records, O1) are repaired.
        """
        import json as _json
        rows = self.conn.execute(
            "MATCH (d:DeletedConcept {original_id: $id}) RETURN d.snapshot AS s "
            "ORDER BY d.deleted_at DESC",
            {"id": concept_id},
        ).rows_as_dict().get_all()
        trashed = []
        if rows:
            try:
                trashed = _json.loads(rows[0]["s"] or "{}").get("trashed_files") or []
            except (ValueError, TypeError):
                trashed = []
        if not self.purge_mgr.recover_deleted(concept_id):
            raise OKFError(
                "NOT_RECOVERABLE",
                f"concept '{concept_id}' not found or past recovery window",
                op="recover_deleted",
                fields={"concept_id": concept_id},
            )
        restored = []
        for t in trashed:
            src, dst = Path(t["trash"]), Path(t["source"])
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                restored.append(str(dst))
        with self._write_lock_ctx():
            if restored:
                for path in restored:
                    self.import_mgr.import_from_okf(Path(path), force=True)
            else:
                concept = self.get_by_id(concept_id)
                if concept is not None:
                    self.import_mgr._extract_links_for_concept(
                        concept_id, concept.body or "",
                        self.import_mgr._relations_of(concept))
                self.import_mgr.repair_links()
        return {"concept_id": concept_id, "recovered": True,
                "restored_files": restored}

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
