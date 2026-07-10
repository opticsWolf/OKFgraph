"""Change-detection (directory/file hashing) extracted during the OKFRouter Phase 1 refactor.

Bodies are verbatim from okfgraph/router.py; the facade (OKFRouter) owns
the shared resources (conn, embedder, tokenizer, ...) and injects them
here. Public callers reach these via router.<method> (component bridge).
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List
logger = logging.getLogger(__name__)

class DeltaDetector:
    """Detects which source files/directories changed since last ingest."""

    SUPPORTED_SOURCE_EXTS = (".md", ".markdown", ".txt")

    def __init__(self, conn, bundle_root):
        self.conn = conn
        self.bundle_root = bundle_root

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
            if fp.is_file() and fp.suffix.lower() in self.SUPPORTED_SOURCE_EXTS:
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
            if fp.is_file() and fp.suffix.lower() in self.SUPPORTED_SOURCE_EXTS:
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


    def _changed_directories(self, source_files: List[Path]) -> tuple[List[Path], List[str]]:
        """Return (changed_files, deleted_paths) using directory-level hash aggregation.

        Groups files by parent directory, computes combined directory hashes,
        and skips entire subtrees when the directory hash hasn't changed.
        Only files in changed directories are returned as changed.
        Deleted paths are relative file paths from the stored map of deleted directories.
        """
        stored_dir_hashes = self._load_directory_hashes()

        # Group files by parent directory
        dir_files: Dict[str, List[Path]] = {}
        for fp in source_files:
            parent = str(fp.parent.relative_to(self.bundle_root))
            dir_files.setdefault(parent, []).append(fp)

        # Compute current directory hashes with file paths
        current_dir_hashes: Dict[str, Dict] = {}
        changed: List[Path] = []
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

        # Detect deleted directories: dirs in stored map but not on disk
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

        # Persist the new directory hashes for the next run
        self._store_directory_hashes(current_dir_hashes)

        if changed:
            logger.info(
                "directory-delta: %d changed dir(s), %d changed files out of %d total",
                len([d for d in dir_files if current_dir_hashes.get(d, {}).get("hash") != stored_dir_hashes.get(d, {}).get("hash")]),
                len(changed),
                len(source_files),
            )
        else:
            logger.info("directory-delta: no changes detected")

        if deleted_paths:
            logger.info(
                "directory-delta: %d deleted file(s) in %d deleted directory(s): %s",
                len(deleted_paths),
                len(deleted_dirs),
                deleted_paths[:5],  # Limit output
            )

        return changed, deleted_paths


    def _store_file_hashes(self, hashes: Dict[str, str]) -> None:
        """Persist a path→hash mapping in the FileHash table.

        Upserts each row so the table always reflects the current state.
        Also stores the concept_id for each file to enable safe purge.
        """
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


    def _load_file_hashes(self) -> Dict[str, str]:
        """Load the persisted path→hash mapping, or return empty dict."""
        try:
            rows = self.conn.execute(
                "MATCH (f:FileHash) RETURN f.path AS p, f.hash AS h"
            ).rows_as_dict().get_all()
            if rows:
                return {r["p"]: r["h"] for r in rows}
        except Exception as exc:
            logger.debug("could not load file hashes: %s", exc)
        return {}


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


    def _changed_files(
        self, source_files: List[Path]
    ) -> tuple[List[Path], List[str]]:
        """Return (changed_files, deleted_paths) since last import.

        Compares SHA-256 of each file against the hashes persisted from the
        previous ``import_bundle()`` call.  New files (not in the stored map)
        are treated as changed.  Files in the stored map but absent from disk
        are returned as deleted_paths.

        The stored map is updated after the check.
        """
        stored = self._load_file_hashes()

        current: Dict[str, str] = {}
        changed: List[Path] = []
        for fp in source_files:
            rel = str(fp.relative_to(self.bundle_root))
            h = self._file_hash(fp)
            current[rel] = h
            if h != stored.get(rel):
                changed.append(fp)

        # Detect deleted files: paths in stored map but not on disk.
        stored_paths = set(stored.keys())
        current_paths = set(current.keys())
        deleted = sorted(stored_paths - current_paths)

        # Persist the new mapping for the next run.
        self._store_file_hashes(current)

        if changed:
            logger.info(
                "delta: %d changed / %d total files",
                len(changed),
                len(source_files),
            )
        else:
            logger.info("delta: no changes detected")

        if deleted:
            logger.info(
                "delta: %d deleted files detected: %s",
                len(deleted),
                deleted,
            )

        return changed, deleted


