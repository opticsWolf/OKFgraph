"""Schema / metadata / index orchestration extracted during the OKFRouter Phase 1 refactor.

Bodies are verbatim from okfgraph/router.py; the facade (OKFRouter) owns
the shared resources (conn, embedder, tokenizer, ...) and injects them
here. Public callers reach these via router.<method> (component bridge).
"""
import logging
import re
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)

class SchemaManager:
    """Owns schema migrations, meta KV store, and search-index rebuild."""

    SCHEMA_VERSION = 5  # bumped when the on-disk schema changes

    def __init__(self, conn, embedding_dim, write_lock_ctx):
        self.conn = conn
        self.embedding_dim = embedding_dim
        self._write_lock_ctx = write_lock_ctx
        self._search_available = False

    def _migrate_v1_to_v2(self) -> None:
        """v1 → v2: Add Chunk table, PART_OF rel, Chunk vector/FTS indexes.

        This corresponds to the v4.0 → v5.0 transition (graph-aware chunking).
        """
        dim = self.embedding_dim
        self.conn.execute(f"""
            CREATE NODE TABLE IF NOT EXISTS Chunk (
                id STRING PRIMARY KEY,
                parent_doc_id STRING,
                chunk_index INT64,
                chunk_text STRING,
                block_type STRING,
                start_offset INT64,
                end_offset INT64,
                embedding FLOAT[{dim}]
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS PART_OF (
                FROM Concept TO Chunk
            )
        """)
        # Create Chunk indexes (idempotent — Ladybug skips if present)
        for table, name, create_sql in self._index_specs():
            if table == "Chunk":
                try:
                    self.conn.execute(create_sql)
                except Exception as e:
                    logger.debug("chunk index %s skipped during migration: %s", name, e)
        logger.info("Schema migrated: v1 → v2 (Chunk + PART_OF)")


    def _migrate_v2_to_v3(self) -> None:
        """v2 → v3: Add FileHash table with concept_id column.

        This corresponds to the v5.1 transition (delta detection & purge).
        """
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS FileHash (
                path STRING PRIMARY KEY,
                hash STRING,
                concept_id STRING
            )
        """)
        logger.info("Schema migrated: v2 → v3 (FileHash + concept_id)")


    def _migrate_v3_to_v4(self) -> None:
        """v3 → v4: Add DirHash table for directory-level hash aggregation.

        This corresponds to the v5.5 transition (directory-level delta detection).
        """
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS DirHash (
                path STRING PRIMARY KEY,
                hash STRING,
                files STRING
            )
        """)
        logger.info("Schema migrated: v3 → v4 (DirHash)")


    def _migrate_v4_to_v5(self) -> None:
        """v4 → v5: Add DeletedConcept table for soft-delete recovery.

        This corresponds to the v5.8 transition (soft-delete with recovery window).
        """
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS DeletedConcept (
                id STRING PRIMARY KEY,
                original_id STRING,
                title STRING,
                body STRING,
                deleted_at STRING,
                type STRING,
                tags STRING
            )
        """)
        logger.info("Schema migrated: v4 → v5 (DeletedConcept)")


    _MIGRATIONS = {}
    _MIGRATIONS[1] = _migrate_v1_to_v2
    _MIGRATIONS[2] = _migrate_v2_to_v3
    _MIGRATIONS[3] = _migrate_v3_to_v4
    _MIGRATIONS[4] = _migrate_v4_to_v5

    def _ensure_schema(self) -> None:
        """Create schema, extensions, and indexes if they don't exist."""
        # Search extensions are optional at construction time. Text-only
        # ingestion, graph traversal, list_directory and get_by_id all work
        # without them; only hybrid/image *search* needs vector + fts. If the
        # extensions can't be installed/loaded (e.g. the extension repository
        # is unreachable), degrade gracefully rather than making the whole
        # router unusable — a search call will then raise a clear error.
        self._search_available = True
        for ext in ("vector", "fts"):
            try:
                self.conn.execute(f"INSTALL {ext};")
                self.conn.execute(f"LOAD {ext};")
            except Exception as e:
                self._search_available = False
                logger.warning(
                    "Could not load the '%s' extension (%s). Vector/FTS search "
                    "will be unavailable; ingestion and graph queries still work.",
                    ext, e,
                )

        self.conn.execute(f"""
            CREATE NODE TABLE IF NOT EXISTS Concept (
                id STRING PRIMARY KEY,
                type STRING,
                title STRING,
                description STRING,
                resource STRING,
                tags STRING[],
                timestamp TIMESTAMP,
                body STRING,
                embedding FLOAT[{self.embedding_dim}],
                extra MAP(STRING, STRING)
            )
        """)
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Directory (id STRING PRIMARY KEY)
        """)

        # ImageAsset: unified embedding column (text-model alt-text vectors and
        # omni-model image vectors share this single space / index).
        self.conn.execute(f"""
            CREATE NODE TABLE IF NOT EXISTS ImageAsset (
                id STRING PRIMARY KEY,
                file_name STRING,
                mime_type STRING,
                alt_text STRING,
                caption STRING,
                embed_route STRING,
                content_hash STRING,
                data BLOB,
                embedding FLOAT[{self.embedding_dim}]
            )
        """)

        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS CONTAINS (
                FROM Directory TO Directory,
                FROM Directory TO Concept
            )
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS LINKS_TO (FROM Concept TO Concept)
        """)
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS INCLUDES_ASSET (FROM Concept TO ImageAsset)
        """)

        self.conn.execute(f"""
            CREATE NODE TABLE IF NOT EXISTS Chunk (
                id STRING PRIMARY KEY,
                parent_doc_id STRING,
                chunk_index INT64,
                chunk_text STRING,
                block_type STRING,
                start_offset INT64,
                end_offset INT64,
                embedding FLOAT[{self.embedding_dim}]
            )
        """)

        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS PART_OF (
                FROM Concept TO Chunk
            )
        """)

        # BrokenLink table — tracks links to concepts not yet imported
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS BrokenLink (
                id STRING PRIMARY KEY,
                source_id STRING,
                target_id STRING,
                timestamp TIMESTAMP
            )
        """)

        # Meta — small key/value store. Used for index dirty-tracking:
        # 'write_epoch' bumps on every index-affecting write (concept/image
        # upsert or delete); 'indexed_epoch' records the write_epoch at which the
        # search indexes were last (re)built. Indexes are dirty when
        # write_epoch > indexed_epoch, which drives change-driven rebuilds.
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Meta (key STRING PRIMARY KEY, value INT64)
        """)

        # FileHash — per-file SHA-256 mapping for delta detection.
        # Stores which files changed since the last import_bundle() call.
        # concept_id maps the file path to its Concept node for safe purge.
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS FileHash (
                path STRING PRIMARY KEY,
                hash STRING,
                concept_id STRING
            )
        """)

        # DirHash — per-directory combined hash for subtree-level delta detection.
        # Stores a SHA-256 hash of all files in a directory subtree. When the
        # hash hasn't changed, the entire subtree is skipped on re-import.
        # The 'files' column stores a JSON string of relative file paths for purge tracking.
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS DirHash (
                path STRING PRIMARY KEY,
                hash STRING,
                files STRING
            )
        """)

        # Run schema migrations if the on-disk version is behind.
        self._run_schema_migrations()

        # When opening a pre-existing DB, the stored embedding column dimension
        # is authoritative (CREATE TABLE IF NOT EXISTS won't have changed it).
        # Adopt it so query vectors match — otherwise a caller that opened the
        # DB with the wrong --dim would silently produce dimension-mismatch
        # errors on every vector search (a common search-browser footgun).
        self._adopt_existing_embedding_dim()

        # Create the search indexes if they don't yet exist. On construction we
        # do NOT drop/rebuild (that would be costly every time a large DB is
        # merely opened for reading); imports rebuild them explicitly so newly
        # written rows become searchable — see _build_search_indexes().
        self._build_search_indexes(rebuild=False)


    def _adopt_existing_embedding_dim(self) -> None:
        """If the Concept.embedding column already exists, honour its dimension.

        Parses ``FLOAT[N]`` from the stored schema and, if it differs from the
        requested ``embedding_dim``, warns and adopts the stored value so query
        encoding and vector indexes stay consistent with what's on disk.
        """
        try:
            rows = self.conn.execute(
                "CALL TABLE_INFO('Concept') RETURN *"
            ).rows_as_dict().get_all()
        except Exception:
            return
        for r in rows:
            if r.get("name") == "embedding":
                m = re.search(r"\[(\d+)\]", str(r.get("type") or ""))
                if m:
                    stored = int(m.group(1))
                    if stored != self.embedding_dim:
                        logger.warning(
                            "Opened a DB whose embedding dimension is %d, but "
                            "embedding_dim=%d was requested. Using the stored "
                            "dimension (%d) to stay consistent with the data.",
                            stored, self.embedding_dim, stored,
                        )
                        self.embedding_dim = stored
                break


    def _run_schema_migrations(self) -> None:
        """Run any pending schema migrations.

        Reads the stored schema version from the Meta table. If it is behind
        ``SCHEMA_VERSION``, runs each migration function in order and bumps
        the version. New databases start at version 0 (no migrations needed
        because ``_ensure_schema`` creates the full schema).
        """
        current = self._get_meta("schema_version", 0)
        target = self.SCHEMA_VERSION

        if current >= target:
            return  # already up to date

        if current == 0:
            # Fresh database — schema was created by _ensure_schema above.
            # Just stamp the version.
            self._set_meta("schema_version", target)
            logger.debug("New database — schema version set to %d", target)
            return

        logger.info(
            "Schema migration: current=%d, target=%d — running %d migration(s)",
            current, target, target - current,
        )

        for v in range(current, target):
            fn = self._MIGRATIONS.get(v)
            if fn is None:
                logger.error(
                    "No migration function for version %d → %d. "
                    "Database may be in an inconsistent state.",
                    v, v + 1,
                )
                break
            try:
                fn(self)
            except Exception as e:
                logger.error(
                    "Migration v%d → v%d failed: %s. Database may be unusable.",
                    v, v + 1, e,
                )
                raise
            self._set_meta("schema_version", v + 1)

        logger.info("Schema migrations complete — version %d", target)


    def _index_specs(self):
        vec = (
            "CALL CREATE_VECTOR_INDEX('{table}', '{name}', 'embedding', "
            "mu := 30, ml := 60, metric := 'cosine', efc := 200)"
        )
        return [
            ("Concept", "concept_embedding", vec.format(table="Concept", name="concept_embedding")),
            ("Concept", "concept_fts",
             "CALL CREATE_FTS_INDEX('Concept', 'concept_fts', ['title', 'description', 'body'])"),
            ("ImageAsset", "image_omni_idx", vec.format(table="ImageAsset", name="image_omni_idx")),
            ("Chunk", "chunk_embedding", vec.format(table="Chunk", name="chunk_embedding")),
            ("Chunk", "chunk_fts",
             "CALL CREATE_FTS_INDEX('Chunk', 'chunk_fts', ['chunk_text'])"),
        ]


    def _build_search_indexes(self, rebuild: bool, force: bool = False) -> bool:
        """Create (or rebuild) the vector + FTS indexes.

        Ladybug's vector/FTS indexes are built over a table's *current*
        contents; rows inserted after an index is created are not returned by
        search until the index is rebuilt. Import paths therefore call this with
        ``rebuild=True`` after data is written.

        Rebuilds are *change-driven*: when ``rebuild`` is requested we skip the
        work unless the indexes are actually dirty (``write_epoch >
        indexed_epoch``), so a no-op import or a redundant call costs nothing.
        Pass ``force=True`` (the manual ``reindex`` path) to rebuild regardless —
        e.g. to repair a DB written by an older build whose markers don't exist.

        ``rebuild=False`` (construction) only creates missing indexes; it never
        drops, never checks dirty, and never stamps — merely opening a DB should
        not trigger an O(N) rebuild.

        Returns True if indexes were (re)built, False if skipped.
        """
        if not getattr(self, "_search_available", False):
            return False
        if rebuild and not force and not self._indexes_dirty():
            logger.debug("Search indexes already up to date; skipping rebuild.")
            return False

        # Capture the epoch we're about to satisfy *before* building, so the
        # stamp reflects the data the index was built over.
        target_epoch = self._get_meta("write_epoch")
        built = False
        for table, name, create_sql in self._index_specs():
            # Ladybug: DROP INDEX leaves stale internal state that prevents
            # recreation with the same name. Instead, rely on CREATE ... IF
            # NOT EXISTS semantics (Ladybug silently skips if present).
            try:
                self.conn.execute(create_sql)
                built = True
            except Exception as e:
                # Already exists (rebuild=False) or table empty — both benign.
                logger.debug("create index %s.%s skipped: %s", table, name, e)
        if rebuild:
            self._set_meta("indexed_epoch", target_epoch)
        return built


    def reindex(self, force: bool = True) -> bool:
        """Rebuild the search indexes on demand (recovery / deferred workflows).

        Use for a DB built by an older version with stale indexes, after a crash
        between the data commit and the index build, or when single-file imports
        deferred rebuilding. Returns True if a rebuild ran.
        """
        # Acquire write lock (Gap #7b)
        with self._write_lock_ctx():
            return self._build_search_indexes(rebuild=True, force=force)


    def _get_meta(self, key: str, default: int = 0) -> int:
        try:
            rows = self.conn.execute(
                "MATCH (m:Meta {key: $k}) RETURN m.value AS v", {"k": key}
            ).rows_as_dict().get_all()
        except Exception:
            return default
        return rows[0]["v"] if rows else default


    def _set_meta(self, key: str, value: int) -> None:
        try:
            self.conn.execute(
                "MERGE (m:Meta {key: $k}) SET m.value = $v", {"k": key, "v": int(value)}
            )
        except Exception as e:
            logger.debug("could not set meta %s=%s: %s", key, value, e)


    def _bump_write_epoch(self) -> None:
        """Mark the search indexes dirty after an index-affecting write."""
        try:
            self.conn.execute(
                """
                MERGE (m:Meta {key: 'write_epoch'})
                ON CREATE SET m.value = 1
                ON MATCH SET m.value = m.value + 1
                """
            )
        except Exception as e:
            logger.debug("could not bump write_epoch: %s", e)


    def _indexes_dirty(self) -> bool:
        return self._get_meta("write_epoch") > self._get_meta("indexed_epoch")


