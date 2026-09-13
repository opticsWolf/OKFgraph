"""Change-detection (directory/file hashing) extracted during the OKFRouter Phase 1 refactor.

Bodies are verbatim from okfgraph/router.py; the facade (OKFRouter) owns
the shared resources (conn, embedder, tokenizer, ...) and injects them
here. Public callers reach these via router.<method> (component bridge).
"""
import hashlib
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from okfgraph.components.import_ import is_concept_file

logger = logging.getLogger(__name__)

class DeltaDetector:
    """Detects which source files/directories changed since last ingest."""

    SUPPORTED_SOURCE_EXTS = (".md", ".markdown", ".txt")

    def __init__(self, conn, bundle_root):
        self.conn = conn
        self.bundle_root = bundle_root
        self._suspended = False

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Bypass the delta baseline (Phase 0, 0.2.15).

        Work-dir imports (PDF auto-import from a TemporaryDirectory) must
        not read or write the shared DirHash/FileHash state: their
        root-relative keys would collide with the real bundle's (notably
        the top-level ``"."`` key). While suspended, detection reports
        every file as changed and all persists are no-ops.
        """
        prev, self._suspended = self._suspended, True
        try:
            yield
        finally:
            self._suspended = prev

    def _file_hash(self, file_path: Path) -> str:
        """SHA-256 hex digest of a file's raw bytes."""
        return hashlib.sha256(file_path.read_bytes()).hexdigest()


    def _compute_directory_hash(self, dir_path: Path) -> str:
        """Compute a combined hash for a directory's contents.

        Hashes all .md/.txt files in the directory (recursively) sorted by relative path,
        then hashes the concatenation of all file hashes. Returns a single
        SHA-256 hex digest that changes if any file in the subtree changes.
        """
        file_hashes = []
        for fp in sorted(dir_path.rglob("*")):
            if is_concept_file(fp):
                rel = str(fp.relative_to(dir_path))
                fh = self._file_hash(fp)
                file_hashes.append((rel, fh))
        combined = "|".join(f"{rel}:{fh}" for rel, fh in file_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


    def _compute_directory_hash_with_files(self, dir_path: Path) -> tuple[str, List[str]]:
        """Compute a combined hash for a directory's contents and return file paths.

        Returns (hash, [relative_file_paths]) where file_paths can be used to
        identify deleted files when the directory is removed.
        """
        file_hashes = []
        file_paths = []
        for fp in sorted(dir_path.rglob("*")):
            if is_concept_file(fp):
                rel = str(fp.relative_to(dir_path))
                fh = self._file_hash(fp)
                file_hashes.append((rel, fh))
                file_paths.append(rel)
        combined = "|".join(f"{rel}:{fh}" for rel, fh in file_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest(), file_paths


    def _load_directory_hashes(self) -> Dict[str, Dict]:
        """Load the persisted directory→{hash, files} mapping, or return empty dict."""
        try:
            rows = self.conn.execute(
                "MATCH (d:DirHash) RETURN d.path AS p, d.hash AS h, d.files AS f"
            ).rows_as_dict().get_all()
            if rows:
                result = {}
                for r in rows:
                    files_str = r.get("f") or ""
                    try:
                        files = json.loads(files_str) if files_str else []
                    except (json.JSONDecodeError, TypeError):
                        files = []
                    result[r["p"]] = {"hash": r["h"], "files": files}
                return result
        except Exception as exc:
            logger.debug("could not load directory hashes: %s", exc)
        return {}


    def _store_directory_hashes(self, hashes: Dict[str, Dict]) -> None:
        """Persist a path→{hash, files} mapping in the DirHash table."""
        if self._suspended:
            return
        try:
            for path, data in hashes.items():
                files_str = json.dumps(data.get("files", []))
                self.conn.execute(
                    """
                    MERGE (d:DirHash {path: $p})
                    SET d.hash = $h, d.files = $f
                    """,
                    {"p": path, "h": data["hash"], "f": files_str},
                )
        except Exception as exc:
            logger.debug("could not store directory hashes: %s", exc)


    def _changed_directories(
        self, source_files: List[Path]
    ) -> tuple[List[Path], List[str], Dict[str, Dict]]:
        """Return (changed_files, deleted_paths, dir_updates) — no side effects.

        Groups files by parent directory, computes combined directory hashes,
        and skips entire subtrees when the directory hash hasn't changed.
        Only files in changed directories are returned as changed.
        Deleted paths are relative file paths: from the stored map of
        deleted/emptied directories, plus files deleted inside surviving
        directories (stored ``files`` vs current files, Phase 0).

        ``dir_updates`` is the would-be DirHash state. The caller persists it
        only after concepts commit (crash-consistency: hashes must never
        describe state newer than the graph). Nothing is written here.
        """
        if self._suspended:
            return list(source_files), [], {}
        stored_dir_hashes = self._load_directory_hashes()

        # Group files by parent directory
        dir_files: Dict[str, List[Path]] = {}
        for fp in source_files:
            parent = str(fp.parent.relative_to(self.bundle_root))
            dir_files.setdefault(parent, []).append(fp)

        # Compute current directory hashes with file paths
        current_dir_hashes: Dict[str, Dict] = {}
        changed: List[Path] = []
        changed_dirs: Set[str] = set()
        for dir_rel, files in dir_files.items():
            dir_path = self.bundle_root / dir_rel
            if dir_path.exists():
                dir_hash, file_paths = self._compute_directory_hash_with_files(dir_path)
            else:
                dir_hash, file_paths = "", []
            current_dir_hashes[dir_rel] = {"hash": dir_hash, "files": file_paths}

            # Check if directory hash changed
            stored_data = stored_dir_hashes.get(dir_rel, {})
            stored_hash = stored_data.get("hash") if stored_data else None
            if dir_hash != stored_hash:
                # Directory changed — add all files in it
                changed.extend(files)
                changed_dirs.add(dir_rel)

        # Detect deleted directories: dirs in stored map but not on disk
        # (a dir left with zero files also vanishes from the walk, which is
        # exactly the "all files deleted" case).
        stored_dirs = set(stored_dir_hashes.keys())
        current_dirs = set(current_dir_hashes.keys())
        deleted_dirs = sorted(stored_dirs - current_dirs)

        # Collect deleted file paths from deleted directories
        deleted_paths: List[str] = []
        for del_dir in deleted_dirs:
            stored_data = stored_dir_hashes.get(del_dir, {})
            stored_files = stored_data.get("files", []) if stored_data else []
            for rel_file in stored_files:
                # Use native path separator to match FileHash entries
                deleted_paths.append(str(Path(del_dir) / rel_file))

        # Per-file deletions inside surviving but changed directories:
        # a file deleted while siblings remain never empties the dir, so
        # the directory-granular check above cannot see it.
        for dir_rel in sorted(changed_dirs):
            dir_path = self.bundle_root / dir_rel
            if not dir_path.exists():
                continue
            stored_data = stored_dir_hashes.get(dir_rel, {})
            stored_files = set(stored_data.get("files", []) if stored_data else [])
            if not stored_files:
                continue
            current_names = {str(fp.relative_to(dir_path)) for fp in dir_files[dir_rel]}
            for rel_file in sorted(stored_files - current_names):
                deleted_paths.append(str(Path(dir_rel) / rel_file))

        if changed:
            logger.info(
                "directory-delta: %d changed dir(s), %d changed files out of %d total",
                len(changed_dirs),
                len(changed),
                len(source_files),
            )
        else:
            logger.info("directory-delta: no changes detected")

        if deleted_paths:
            logger.info(
                "directory-delta: %d deleted file(s): %s",
                len(deleted_paths),
                deleted_paths[:5],  # Limit output
            )

        return changed, deleted_paths, current_dir_hashes


    def _store_file_hashes(self, hashes: Dict[str, str]) -> None:
        """Persist a path→hash mapping in the FileHash table.

        Upserts each row so the table always reflects the current state.
        Also stores the concept_id for each file to enable safe purge.
        Callers commit only hashes of successfully parsed files, after the
        concept transaction commits (Phase 0 crash-consistency).
        """
        if self._suspended:
            return
        try:
            for path, h in hashes.items():
                # Derive concept_id from path: strip extension, normalise separators.
                concept_id = str(Path(path).with_suffix("")).replace("\\", "/")  # remove .md / .txt suffix, use forward slashes
                self.conn.execute(
                    """
                    MERGE (f:FileHash {path: $p})
                    SET f.hash = $h, f.concept_id = $c
                    """,
                    {"p": path, "h": h, "c": concept_id},
                )
        except Exception as exc:
            logger.debug("could not store file hashes: %s", exc)


    # -- mirror-deletion tombstones (Phase 0, 0.2.15) --------------------

    def _record_deletions(self, paths: List[str]) -> int:
        """Persist deletion tombstones (DeletedPath rows), idempotently.

        A tombstone records that `path` was missing from the bundle at
        detection time. It survives no-purge runs so a later
        ``import --purge-deleted`` still sees the deletion (DirHash.files
        diffs alone are one-shot for surviving directories).
        """
        if self._suspended or not paths:
            return 0
        now = int(time.time())
        recorded = 0
        try:
            for p in paths:
                self.conn.execute(
                    "MERGE (d:DeletedPath {path: $p}) SET d.detected_at = $t",
                    {"p": p, "t": now},
                )
                recorded += 1
        except Exception as exc:
            logger.debug("could not record deletions: %s", exc)
        return recorded

    def _load_pending_deletions(self) -> List[str]:
        """All recorded-but-unresolved deletion paths, in stable order."""
        try:
            rows = self.conn.execute(
                "MATCH (d:DeletedPath) RETURN d.path AS p ORDER BY d.path"
            ).rows_as_dict().get_all()
            return [r["p"] for r in rows] if rows else []
        except Exception as exc:
            logger.debug("could not load pending deletions: %s", exc)
        return []

    def _clear_deletion(self, path: str) -> None:
        """Drop one tombstone (purged, or stale with no matching concept)."""
        try:
            self.conn.execute(
                "MATCH (d:DeletedPath {path: $p}) DELETE d", {"p": path}
            )
        except Exception as exc:
            logger.debug("could not clear deletion %s: %s", path, exc)

    # -- wedge repair (Phase 0, 0.2.15) ----------------------------------

    @staticmethod
    def _parent_dir_key(native_rel_path: str) -> str:
        """DirHash key for the directory containing a FileHash-style path.

        ``str(Path("w.md").parent)`` is already ``"."``, matching the
        top-level key `_changed_directories` writes — no special case needed.
        """
        return str(Path(native_rel_path).parent)

    def _clear_orphan_hashes(self) -> Dict[str, int]:
        """Delete FileHash rows with no Concept and no DeletedConcept.

        Signature of a crashed import (hashes committed, concepts not).
        Parent DirHash rows are dropped too so the next import re-walks
        those directories instead of trusting the wedge. Used by
        ``okf doctor --fix``; returns what was cleared.
        """
        cleared_files = 0
        cleared_dirs = 0
        try:
            rows = self.conn.execute(
                "MATCH (f:FileHash) RETURN f.path AS p, f.concept_id AS c"
            ).rows_as_dict().get_all() or []
            live = {
                r["cid"]
                for r in (
                    self.conn.execute("MATCH (c:Concept) RETURN c.id AS cid")
                    .rows_as_dict().get_all()
                    or []
                )
            }
            try:
                tombstoned = {
                    r["oid"]
                    for r in (
                        self.conn.execute(
                            "MATCH (d:DeletedConcept) RETURN d.original_id AS oid"
                        )
                        .rows_as_dict().get_all()
                        or []
                    )
                }
            except Exception:
                tombstoned = set()
            orphans = [
                r for r in rows
                if r["c"] not in live and r["c"] not in tombstoned
            ]
            dir_keys = set()
            for r in orphans:
                self.conn.execute(
                    "MATCH (f:FileHash {path: $p}) DELETE f", {"p": r["p"]}
                )
                cleared_files += 1
                dir_keys.add(self._parent_dir_key(r["p"]))
            for d in dir_keys:
                self.conn.execute(
                    "MATCH (dh:DirHash {path: $d}) DELETE dh", {"d": d}
                )
                cleared_dirs += 1
        except Exception as exc:
            logger.debug("could not clear orphan hashes: %s", exc)
        return {
            "cleared_orphan_hashes": cleared_files,
            "cleared_dir_hashes": cleared_dirs,
        }

    # -- detach state (Phase 1, 0.2.16) ----------------------------------

    def is_detached(self) -> bool:
        """True once `okf detach` has ended the mirror relationship."""
        try:
            rows = self.conn.execute(
                "MATCH (m:Meta {key: 'detached'}) RETURN m.value AS v"
            ).rows_as_dict().get_all()
            return bool(rows and rows[0]["v"])
        except Exception:
            return False

    def get_detached_state(self) -> Optional[Dict[str, Any]]:
        """Provenance recorded at detach, or None when attached."""
        if not self.is_detached():
            return None
        try:
            rows = self.conn.execute(
                "MATCH (s:SourceRoot) RETURN s.alias AS alias, s.path AS path, "
                "s.file_count AS file_count, s.last_seen AS last_seen"
            ).rows_as_dict().get_all()
        except Exception:
            rows = []
        try:
            at = self.conn.execute(
                "MATCH (m:Meta {key: 'detached_at'}) RETURN m.value AS v"
            ).rows_as_dict().get_all()
            detached_at = at[0]["v"] if at else 0
        except Exception:
            detached_at = 0
        return {
            "detached_at": detached_at,
            "roots": [
                {"alias": r["alias"], "path": r["path"],
                 "file_count": r["file_count"], "last_seen": r["last_seen"]}
                for r in (rows or [])
            ],
        }

    def set_detached(self, path: str, file_count: int, alias: str = "") -> Dict[str, Any]:
        """Record detachment of one source tree; returns the stored row."""
        now = int(time.time())
        self.conn.execute(
            "MERGE (s:SourceRoot {alias: $a}) "
            "SET s.path = $p, s.file_count = $n, s.last_seen = $t",
            {"a": alias, "p": path, "n": int(file_count), "t": now},
        )
        self.conn.execute(
            "MERGE (m:Meta {key: 'detached'}) SET m.value = 1"
        )
        self.conn.execute(
            "MERGE (m:Meta {key: 'detached_at'}) SET m.value = $t", {"t": now}
        )
        return {"alias": alias, "path": path,
                "file_count": int(file_count), "last_seen": now,
                "detached_at": now}

    def clear_detached(self) -> None:
        """Clear detachment after a successful --force re-attach."""
        self.conn.execute("MATCH (m:Meta {key: 'detached'}) DELETE m")
        self.conn.execute("MATCH (m:Meta {key: 'detached_at'}) DELETE m")
        self.conn.execute("MATCH (s:SourceRoot) DELETE s")


    def _load_file_hash_concept_ids(self) -> Dict[str, str]:
        """Load the persisted path→concept_id mapping, or return empty dict."""
        try:
            rows = self.conn.execute(
                "MATCH (f:FileHash) RETURN f.path AS p, f.concept_id AS c"
            ).rows_as_dict().get_all()
            if rows:
                return {r["p"]: r["c"] for r in rows}
        except Exception as exc:
            logger.debug("could not load file hash concept ids: %s", exc)
        return {}


    # NOTE: the old per-file `_changed_files` detector was removed in 0.2.15
    # (Phase 0). `_changed_directories` is the single delta implementation;
    # it now also covers per-file deletions inside surviving directories.


