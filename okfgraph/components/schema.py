"""Schema / metadata / index orchestration extracted during the OKFRouter Phase 1 refactor.

Bodies are verbatim from okfgraph/router.py; the facade (OKFRouter) owns
the shared resources (conn, embedder, tokenizer, ...) and injects them
here. Public callers reach these via router.<method> (component bridge).
"""
import logging
import re
from pathlib import Path
logger = logging.getLogger(__name__)

class SchemaManager:
    """Owns schema migrations, meta KV store, and search-index rebuild."""

    SCHEMA_VERSION = 10  # bumped when the on-disk schema changes

    #: Ladybug extensions hybrid search needs (O5).
    SEARCH_EXTENSIONS = ("vector", "fts")

    def __init__(self, conn, embedding_dim, write_lock_ctx, extension_dir=None):
        self.conn = conn
        self.embedding_dim = embedding_dim
        self._write_lock_ctx = write_lock_ctx
        self._search_available = False
        # Offline extensions (0.11.0, O5): a directory holding
        # lib<ext>.lbug_extension files. When set, extensions load from
        # there by path and are never downloaded.
        self.extension_dir = Path(extension_dir).expanduser() if extension_dir else None
        #: Why search is unavailable, per extension (doctor reads this).
        self.extension_errors: dict = {}

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

    def _migrate_v5_to_v6(self) -> None:
        """v5 → v6: DeletedConcept gains the full-fidelity snapshot column.

        Three on-disk states exist: v5 tables from v1–v4 upgrades (ALTER
        them), 0.2.x fresh DBs that never got the table (CREATE it whole),
        and DBs whose fresh-create already made the v6 shape (nothing to
        do — the ensure step runs before migrations). Probe the actual
        columns via TABLE_INFO; never assume.
        """
        try:
            info = self.conn.execute(
                "CALL TABLE_INFO('DeletedConcept') RETURN *"
            ).rows_as_dict().get_all()
        except Exception:
            info = None
        if not info:
            self.conn.execute("""
                CREATE NODE TABLE DeletedConcept (
                    id STRING PRIMARY KEY,
                    original_id STRING,
                    title STRING,
                    body STRING,
                    deleted_at STRING,
                    type STRING,
                    tags STRING,
                    snapshot STRING
                )
            """)
        elif not any(r.get("name") == "snapshot" for r in info):
            self.conn.execute("ALTER TABLE DeletedConcept ADD snapshot STRING")
        else:
            logger.debug("DeletedConcept already has snapshot — nothing to do")
            return
        logger.info("Schema migrated: v5 → v6 (DeletedConcept.snapshot)")


    def _migrate_v6_to_v7(self) -> None:
        """v6 to v7: SourceRoot provenance table for `okf detach`.

        Detached graphs record which source tree they were cut from so a
        later `--force` import can verify it is re-attaching the same tree.
        Plain CREATE IF NOT EXISTS - idempotent whether or not the ensure
        step already made the table.
        """
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS SourceRoot (
                alias STRING PRIMARY KEY,
                path STRING,
                file_count INT64,
                last_seen INT64
            )
        """)
        logger.info("Schema migrated: v6 to v7 (SourceRoot)")

    def _migrate_v7_to_v8(self) -> None:
        """v7 → v8: parent-DirHash key (``dir``) on FileHash/DeletedPath.

        Multi-root (0.4.0, Phase 2 §2.2) namespaces delta keys per root
        (``@alias/rel``), and string math on a prefixed path cannot tell
        ``@bb/.`` apart from a legacy ``@x`` row — so writers store the
        exact parent DirHash key and purge/orphan-scrub join on it.
        Backfill uses legacy parent math, which is exact for every
        pre-0.4.0 row (all single-root bare paths). Probe-never-assume:
        TABLE_INFO decides whether each ALTER is needed.
        """
        for table in ("FileHash", "DeletedPath"):
            try:
                info = self.conn.execute(
                    f"CALL TABLE_INFO('{table}') RETURN *"
                ).rows_as_dict().get_all()
            except Exception:
                info = None
            if info and not any(r.get("name") == "dir" for r in info):
                self.conn.execute(f"ALTER TABLE {table} ADD dir STRING")
        # Backfill rows that predate the column (legacy math is exact:
        # no namespaced keys exist before 0.4.0).
        try:
            rows = self.conn.execute(
                "MATCH (f:FileHash) WHERE f.dir IS NULL "
                "RETURN f.path AS p"
            ).rows_as_dict().get_all() or []
            for r in rows:
                self.conn.execute(
                    "MATCH (f:FileHash {path: $p}) SET f.dir = $d",
                    {"p": r["p"],
                     "d": str(Path(r["p"]).parent)},
                )
            rows = self.conn.execute(
                "MATCH (d:DeletedPath) WHERE d.dir IS NULL "
                "RETURN d.path AS p"
            ).rows_as_dict().get_all() or []
            for r in rows:
                self.conn.execute(
                    "MATCH (d:DeletedPath {path: $p}) SET d.dir = $d",
                    {"p": r["p"],
                     "d": str(Path(r["p"]).parent)},
                )
        except Exception as exc:
            logger.debug("v7→v8 dir backfill skipped: %s", exc)
        logger.info("Schema migrated: v7 → v8 (mirror dir keys)")


    def _migrate_v8_to_v9(self) -> None:
        """v8 → v9 (0.11.0): typed links, link diagnostics, producer cursors.

        - ``LINKS_TO.kind`` (O6), existing edges become ``'mention'`` — every
          edge before 0.11.0 came from a body link.
        - ``BrokenLink.kind`` / ``reason`` / ``candidates`` / ``raw`` (O6,
          O12): which relation missed, why (``missing`` | ``ambiguous``),
          and the link as written, so repair resolves it as import did.
        - ``ProducerCursor`` (O10): one row per producer name.

        Probe-never-assume: TABLE_INFO decides each ALTER, so a partially
        migrated file (crash mid-rung) finishes cleanly.
        """
        def _cols(table):
            try:
                rows = self.conn.execute(
                    f"CALL TABLE_INFO('{table}') RETURN *"
                ).rows_as_dict().get_all()
            except Exception:
                return None
            return {r.get("name") for r in rows}

        cols = _cols("LINKS_TO")
        if cols is not None and "kind" not in cols:
            self.conn.execute(
                "ALTER TABLE LINKS_TO ADD kind STRING DEFAULT 'mention'")
        cols = _cols("BrokenLink")
        if cols is not None:
            for col in ("kind", "reason", "candidates", "raw"):
                if col not in cols:
                    self.conn.execute(f"ALTER TABLE BrokenLink ADD {col} STRING")
        self._create_v9_tables()
        logger.info("Schema migrated: v8 → v9 (typed links, producer cursors)")

    def _migrate_v9_to_v10(self) -> None:
        """v9 → v10 (0.11.0): graph-kept concept times.

        ``created_at`` / ``updated_at`` / ``effective_at`` / ``content_sig``
        on Concept (:mod:`okfgraph.components.timestamps`). Existing
        concepts get the migration time as created/updated (nothing older
        is known; file mtimes are not trusted), and ``effective_at`` from
        their authored date where they have one.
        """
        from okfgraph.components.timestamps import authored_time, utc_now

        try:
            rows = self.conn.execute(
                "CALL TABLE_INFO('Concept') RETURN *").rows_as_dict().get_all()
        except Exception:
            return  # no Concept table yet: _ensure_schema creates it whole
        cols = {r.get("name") for r in rows}
        for col, typ in (("created_at", "TIMESTAMP"), ("updated_at", "TIMESTAMP"),
                         ("effective_at", "TIMESTAMP"), ("content_sig", "STRING")):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE Concept ADD {col} {typ}")
        now = utc_now()
        self.conn.execute(
            "MATCH (c:Concept) WHERE c.created_at IS NULL "
            "SET c.created_at = $now, c.updated_at = $now", {"now": now})
        authored = 0
        for r in self.conn.execute(
                "MATCH (c:Concept) WHERE c.effective_at IS NULL "
                "RETURN c.id AS id, c.timestamp AS timestamp, c.extra AS extra"
        ).rows_as_dict().get_all():
            ts = authored_time({**(r.get("extra") or {}), "timestamp": r.get("timestamp")})
            if ts is not None:
                self.conn.execute(
                    "MATCH (c:Concept {id: $id}) SET c.effective_at = $ts",
                    {"id": r["id"], "ts": ts})
                authored += 1
        self.conn.execute(
            "MATCH (c:Concept) WHERE c.effective_at IS NULL "
            "SET c.effective_at = c.updated_at")
        logger.info("Schema migrated: v9 → v10 (concept times; %d authored)", authored)

    def _create_v9_tables(self) -> None:
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS ProducerCursor (
                name STRING PRIMARY KEY,
                cursor STRING,
                updated_at INT64
            )
        """)

    _MIGRATIONS = {}
    _MIGRATIONS[1] = _migrate_v1_to_v2
    _MIGRATIONS[2] = _migrate_v2_to_v3
    _MIGRATIONS[3] = _migrate_v3_to_v4
    _MIGRATIONS[4] = _migrate_v4_to_v5
    _MIGRATIONS[5] = _migrate_v5_to_v6
    _MIGRATIONS[6] = _migrate_v6_to_v7
    _MIGRATIONS[7] = _migrate_v7_to_v8
    _MIGRATIONS[8] = _migrate_v8_to_v9
    _MIGRATIONS[9] = _migrate_v9_to_v10

    def _ensure_schema(self) -> None:
        """Create schema, extensions, and indexes if they don't exist."""
        # Search extensions are optional at construction time. Text-only
        # ingestion, graph traversal, list_directory and get_by_id all work
        # without them; only hybrid/image *search* needs vector + fts. If the
        # extensions can't be installed/loaded (e.g. the extension repository
        # is unreachable), degrade gracefully rather than making the whole
        # router unusable — a search call will then raise a clear error.
        self._search_available = True
        self.extension_errors = {}
        for ext in self.SEARCH_EXTENSIONS:
            try:
                self.load_extension(self.conn, ext)
            except Exception as e:
                self._search_available = False
                self.extension_errors[ext] = str(e)
                logger.warning(
                    "Could not load the '%s' extension (%s). Vector/FTS search "
                    "will be unavailable; ingestion and graph queries still work.%s",
                    ext, e,
                    "" if self.extension_dir else
                    " Run 'okf setup extensions' and set [database] "
                    "extension_dir for offline machines.",
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
                extra MAP(STRING, STRING),
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                effective_at TIMESTAMP,
                content_sig STRING
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
        # kind (0.11.0, O6): 'mention' for body links, the declared kind
        # for frontmatter relations (supersedes, supports, ...).
        self.conn.execute("""
            CREATE REL TABLE IF NOT EXISTS LINKS_TO (
                FROM Concept TO Concept, kind STRING DEFAULT 'mention'
            )
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
        # kind/reason/candidates/raw (0.11.0, O6/O12): which relation missed,
        # 'missing' or 'ambiguous', the claiming ids (JSON) when ambiguous,
        # and the link as written (repair re-resolves it with import's rules).
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS BrokenLink (
                id STRING PRIMARY KEY,
                source_id STRING,
                target_id STRING,
                timestamp TIMESTAMP,
                kind STRING,
                reason STRING,
                candidates STRING,
                raw STRING
            )
        """)

        # DeletedConcept — soft-delete staging with a full-fidelity
        # snapshot (v6: every Concept column as JSON, so recover restores
        # embeddings/metadata instead of a lobotomized row).
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS DeletedConcept (
                id STRING PRIMARY KEY,
                original_id STRING,
                title STRING,
                body STRING,
                deleted_at STRING,
                type STRING,
                tags STRING,
                snapshot STRING
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
                concept_id STRING,
                dir STRING
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

        # DeletedPath — durable mirror-deletion tombstones (Phase 0, 0.2.15).
        # Written at detection time, consumed by import --purge-deleted. A
        # tombstone survives no-purge runs (unlike DirHash.files diffs, which
        # are one-shot for surviving dirs). IF NOT EXISTS: pre-0.2.15 DBs
        # gain it on open — no migration, no version bump.
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS DeletedPath (
                path STRING PRIMARY KEY,
                detected_at INT64,
                dir STRING
            )
        """)

        # SourceRoot — detach provenance (Phase 1, 0.2.16). One row per
        # source tree the graph was cut from via `okf detach` (single row,
        # alias "", until multi-root lands). A later --force import
        # re-attaches only when its root matches this provenance.
        # IF NOT EXISTS covers fresh DBs; v6 DBs gain it via _migrate_v6_to_v7.
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS SourceRoot (
                alias STRING PRIMARY KEY,
                path STRING,
                file_count INT64,
                last_seen INT64
            )
        """)

        # Run schema migrations if the on-disk version is behind.
        self._run_schema_migrations()
        # v9 side tables: created after migrations so an old file gains
        # them in its own rung first; fresh files get them here.
        self._create_v9_tables()

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


    def extension_file(self, ext: str):
        """The ``lib<ext>.lbug_extension`` under ``extension_dir``, or None.

        Two layouts are accepted: flat (``DIR/libvector.lbug_extension``,
        what ``okf setup extensions`` writes) and Ladybug's own cache layout
        (``DIR/vector/libvector.lbug_extension``), so a copied
        ``~/.lbdb/extension/<ver>/<platform>`` directory works as-is.
        """
        if self.extension_dir is None:
            return None
        name = f"lib{ext}.lbug_extension"
        for cand in (self.extension_dir / name, self.extension_dir / ext / name):
            if cand.is_file():
                return cand
        return None

    def load_extension(self, conn, ext: str, install: bool = True) -> None:
        """Load one search extension on ``conn`` (0.11.0, O5).

        With ``extension_dir`` set the extension loads by path and a missing
        file is an error — never a silent download attempt. Without it,
        Ladybug's own ``INSTALL`` (network, cached under ``~/.lbdb``) runs
        first, as before. ``LOAD`` is per connection, so short-lived index
        connections call this too.
        """
        if self.extension_dir is not None:
            path = self.extension_file(ext)
            if path is None:
                raise FileNotFoundError(
                    f"lib{ext}.lbug_extension not found in extension_dir "
                    f"{self.extension_dir} (run 'okf setup extensions')")
            conn.execute(f"LOAD EXTENSION '{path.as_posix()}'")
            return
        if install:
            conn.execute(f"INSTALL {ext};")
        conn.execute(f"LOAD {ext};")

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


    #: Rebuild policies a write may ask for (0.11.0, O8).
    REBUILD_POLICIES = ("auto", "defer", "force")

    def _build_search_indexes(self, rebuild: bool, force: bool = False) -> bool:
        """Create missing vector + FTS indexes; with ``force``, recreate them.

        Measured on Ladybug 0.21.2 (0.11.0, O8): both index kinds are
        maintained **in place** — rows inserted, replaced (``DETACH DELETE``
        + ``CREATE`` in one transaction) or written after a reopen are
        returned by ``QUERY_VECTOR_INDEX`` / ``QUERY_FTS_INDEX`` with no
        rebuild, and creating an index on an empty table works. The pre-0.11
        "rebuild after every import" was therefore a ``CREATE`` that failed
        with *already exists* and was swallowed: harmless, and never needed.

        So the non-forced path only creates indexes that are missing (a
        table first created by a migration, a DB opened while the
        extensions were unavailable) and stamps ``indexed_epoch``. ``force``
        (``okf reindex``) drops and recreates every index — the recovery
        path for a file whose index state is damaged; before 0.11 it hit the
        same swallowed *already exists* and rebuilt nothing.

        Returns True if any index was created.
        """
        if not getattr(self, "_search_available", False):
            return False
        if rebuild and not force and not self._indexes_dirty():
            logger.debug("Search indexes already up to date; skipping.")
            return False

        target_epoch = self._get_meta("write_epoch")
        built = False
        drops = {
            "vector": "CALL DROP_VECTOR_INDEX('{table}', '{name}')",
            "fts": "CALL DROP_FTS_INDEX('{table}', '{name}')",
        }
        for table, name, create_sql in self._index_specs():
            if force:
                kind = "fts" if "FTS" in create_sql else "vector"
                try:
                    self.conn.execute(drops[kind].format(table=table, name=name))
                except Exception as e:
                    logger.debug("drop index %s.%s skipped: %s", table, name, e)
            try:
                self.conn.execute(create_sql)
                built = True
            except Exception as e:
                # Already exists — benign, the index is maintained in place.
                logger.debug("create index %s.%s skipped: %s", table, name, e)
        if rebuild:
            self._set_meta("indexed_epoch", target_epoch)
        return built

    def after_write(self, policy: str = "auto") -> bool:
        """Index step every write path ends with (0.11.0, O8).

        ``auto`` creates any missing index and marks the epoch indexed;
        ``defer`` skips even that (bulk writers that will call
        ``reindex`` or ``after_write`` once at the end); ``force``
        recreates every index. Because the indexes are maintained in
        place, ``defer`` never hides a written concept from search.
        """
        if policy not in self.REBUILD_POLICIES:
            raise ValueError(
                f"rebuild must be one of {self.REBUILD_POLICIES}, got {policy!r}")
        if policy == "defer":
            return False
        return self._build_search_indexes(rebuild=True, force=(policy == "force"))

    def reindex(self, force: bool = True) -> bool:
        """Rebuild the search indexes on demand (recovery).

        ``force=True`` drops and recreates every index; ``force=False``
        only creates missing ones when the epoch says writes happened since
        the last index step. Returns True if anything was created.
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


