"""OKFRouter — Ladybug-backed knowledge graph with Jina v5 embeddings.

Design choices:
    - **Rust embedding core**: The jina-embeddings-v5 model runs through
      the embroider wheel (ONNX Runtime, no torch / optimum / transformers
      anywhere in the process). Task prefix, tokenization, last-token
      pooling, L2 normalisation and Matryoshka truncation all live in Rust;
      ``tests/test_parity.py`` pins the numerics.
    - **CUDA is opportunistic**: the loaded ORT build decides (CUDA EP
      present or not); a missing GPU warns once and falls back to CPU,
      never fatal.
    - **Last-token pooling**: Required by jina-embeddings-v5. Mean pooling
      produces vectors in a different embedding space that will NOT align
      with the omni model's image embeddings in the unified ImageAsset
      index.
"""

import hashlib
import json
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import frontmatter
import ladybug as lb
from fasteners import InterProcessLock
import numpy as np
import yaml

from okfgraph.images import (
    EmbedRoute,
    IngestMode,
    build_extracted_images,
    plan_embedding,
)
import mordant

from okfgraph.models import ChunkModel, ConceptModel
from okfgraph.components import (
    DeltaDetector,
    DiffManager,
    DoctorManager,
    EmbeddingEngine,
    ExportManager,
    ImageAssetManager,
    ImportManager,
    IngestManager,
    PurgeManager,
    SchemaManager,
    SearchEngine,
)

logger = logging.getLogger(__name__)


class OKFRouter:
    """Routes OKF concepts through a Ladybug graph + vector + FTS database."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    # Valid Matryoshka truncation levels shared by jina-embeddings-v5
    # (text-small-retrieval and omni-small-retrieval both support these).
    ALLOWED_DIMS = (32, 64, 128, 256, 512, 768, 1024)

    # ── Component-backed classmethods (Phase 1 refactor) ──────────
    # These are owned by EmbeddingEngine; aliases keep the public API
    # (e.g. OKFRouter.model_info(...)) stable.
    model_info = EmbeddingEngine.model_info
    default_cache_dir = EmbeddingEngine.default_cache_dir

    # Schema constants/registry live on SchemaManager (Phase 1 refactor).
    # Class-level aliases keep the public API + schema tests stable.
    SCHEMA_VERSION = SchemaManager.SCHEMA_VERSION
    _MIGRATIONS = SchemaManager._MIGRATIONS

    # ── Schema versioning ─────────────────────────────────────────
    # Incremented every time the on-disk schema changes in a way that
    # requires a migration step (new table, new column, new index).

    # Migration registry: version → migration function.
    # Each function receives `self` (the router) and must be idempotent.

    # Soft-delete recovery window (seconds). Concepts deleted within this
    # window can be recovered. Default: 24 hours.
    SOFT_DELETE_WINDOW = 24 * 60 * 60  # 24 hours

    # ------------------------------------------------------------------
    # Schema migrations
    # ------------------------------------------------------------------

    # Register migrations

    def __init__(
        self,
        db_path: str,
        bundle_root: str,
        model_id: str = "jinaai/jina-embeddings-v5-text-small-retrieval",
        omni_model_id: str = "jinaai/jina-embeddings-v5-omni-small-retrieval",
        embedding_dim: int = 512,
        cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
        allow_remote_images: bool = False,
        allowed_image_domains: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 40,
        enable_chunking: bool = True,
        wal_mode: bool = False,
        converter=None,
    ):
        """Open (or create) the graph database and wire up all managers.

        Args:
            db_path: Path to the Ladybug database file.
            bundle_root: Root directory of the OKF markdown bundle.
            model_id: HuggingFace ID of the Jina v5 text embedding model.
            omni_model_id: HuggingFace ID of the Jina v5 omni model (images).
            embedding_dim: Truncated Matryoshka dimension (<= 1024).
            cache_dir: Model cache directory (defaults per-platform).
            model_path: Explicit local ONNX file (with `tokenizer_path`);
                skips every download — air-gapped / reproducible installs.
                An external-data sidecar must sit next to this file.
            tokenizer_path: Explicit local `tokenizer.json` (with `model_path`).
            device: "cpu" (CUDA is opportunistic inside the Rust loader).
            allow_remote_images: Whether http(s) image URLs may be fetched.
            allowed_image_domains: Domain allowlist for remote images.
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between consecutive chunks in tokens.
            enable_chunking: If False, store whole documents (no chunks).
            wal_mode: Enable Ladybug WAL mode for concurrent readers.
            converter: DocumentConverter used for PDF ingestion (see
                ``okfgraph.components.converters``). Defaults to
                ``BobineConverter()`` built lazily — any object with a
                ``convert(pdf_path, output_dir, *, on_page=None)`` method
                returning a ``ConvertedDocument`` works. Per-call override
                via ``ingest_mgr.ingest_pdf(..., converter=...)``.
        """
        if embedding_dim > 1024:
            raise ValueError(f"embedding_dim must be <= 1024 (model output), got {embedding_dim}")
        if embedding_dim < 32:
            raise ValueError(f"embedding_dim must be >= 32, got {embedding_dim}")
        if embedding_dim not in self.ALLOWED_DIMS:
            logger.warning(
                "embedding_dim=%d is not an official Matryoshka dimension %s; "
                "retrieval quality may be suboptimal. Consider 256 or 512.",
                embedding_dim, self.ALLOWED_DIMS,
            )

        self.db = lb.Database(db_path)
        self.conn = lb.Connection(self.db)

        # WAL mode (Gap #7a) — enables concurrent reads during writes.
        if wal_mode:
            self.conn.execute("PRAGMA journal_mode=WAL")
            logger.debug("WAL mode enabled for %s", db_path)

        # Inter-process write lock (Gap #7b) — prevents concurrent writers
        # from corrupting the database. Uses a lock file alongside the DB.
        db_path_obj = Path(db_path)
        lock_path = str(db_path_obj.with_suffix(db_path_obj.suffix + ".lock"))
        self._write_lock = InterProcessLock(lock_path)
        self._write_lock_timeout = 300  # 5 min timeout for acquire()
        logger.debug("write lock file: %s", lock_path)

        self.bundle_root = Path(bundle_root).resolve()
        self.embedding_dim = embedding_dim
        self.model_id = model_id
        self.omni_model_id = omni_model_id
        self.cache_dir = cache_dir
        self.device = device
        self.allow_remote_images = allow_remote_images
        self.allowed_image_domains = allowed_image_domains or []

        # Chunking configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_chunking = enable_chunking

        # Omni (multimodal) model is loaded lazily — text-only ingestion never
        # pays the cost of pulling in the ~1.5B-param vision tower.
        self._omni = None

        # Text embeddings: Rust embroider wheel (Jina v5 via ORT). No Python
        # fallback — a mid-run stack switch would silently mix vector spaces
        # in one index.
        from okfgraph.components.embedding import LazyRustEncoder, resolve_ort_dylib
        # Resolve first: the native module loads ORT dynamically, so the
        # shared runtime choice must be fixed before importing it.
        self.ort_dylib = resolve_ort_dylib()
        try:
            import embroider
        except ImportError:
            raise RuntimeError(
                "the embroider wheel is required for text embeddings: "
                "pip install 'embroider>=0.1,<0.2'"
            ) from None
        # The Rust crate knows auto/cpu/cuda; map torch-style aliases.
        # Validate eagerly so a bad device still fails at construction —
        # the session itself opens lazily on first encode (see below).
        rust_device = {"mps": "auto"}.get(device, device)
        if rust_device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"device must be 'auto', 'cpu' or 'cuda', got '{device}'"
            )
        if (model_path is None) != (tokenizer_path is None):
            raise ValueError(
                "model_path and tokenizer_path must be given together "
                "(explicit local files skip all downloads)"
            )
        explicit_files = model_path is not None
        if explicit_files:
            for label, path in (
                ("model_path", model_path),
                ("tokenizer_path", tokenizer_path),
            ):
                if not Path(path).is_file():
                    raise FileNotFoundError(
                        f"{label} not found: {path} "
                        "(explicit paths skip all downloads)"
                    )
            logger.debug("using explicit model files: %s", model_path)

        def _report_encoder_open(encoder) -> None:
            logger.info(
                "text embeddings: %s dim=%d cuda=%s",
                model_id, self.embedding_dim, encoder.used_cuda,
            )
            if device == "cuda" and not encoder.used_cuda:
                logger.warning(
                    "CUDA requested but the loaded ONNX Runtime has no CUDA execution "
                    "provider — running on CPU. Install onnxruntime-gpu for acceleration."
                )

        # Text embeddings: Rust embroider wheel (Jina v5 via ORT). No Python
        # fallback — a mid-run stack switch would silently mix vector spaces
        # in one index. The wheel import above stays fail-fast; the session
        # open is lazy so model-free commands (PPR search, budgeted reads,
        # diff, doctor) never pay model-download/session-build costs.
        if explicit_files:
            session_factory = lambda: embroider.JinaV5.open_files(
                str(model_path),
                str(tokenizer_path),
                # Read off self: the stored dimension wins over the
                # requested one (§4.1 adoption) and factories run lazily,
                # after construction — so this is the adopted value.
                truncate_dim=self.embedding_dim,
                device=rust_device,
            )
            tokenizer_factory = lambda: embroider.JinaTokenizer.open_files(
                str(tokenizer_path),
            )
        else:
            session_factory = lambda: embroider.JinaV5.open(
                model_id,
                # See above: adopted dimension, resolved at first encode.
                truncate_dim=self.embedding_dim,
                device=rust_device,
                cache_dir=cache_dir,
            )
            tokenizer_factory = lambda: embroider.JinaTokenizer.open(
                model_id,
                cache_dir=cache_dir,
            )
        self.encoder = LazyRustEncoder(
            model_id=model_id,
            truncate_dim=embedding_dim,
            device=rust_device,
            session_factory=session_factory,
            tokenizer_factory=tokenizer_factory,
            on_open=_report_encoder_open,
        )

        # ── Component wiring (Phase 1-3 refactor) ───────────────────
        # The facade owns the resources (conn, encoder, lock) and injects
        # them into focused component objects.
        self.schema_mgr = SchemaManager(
            self.conn, self.embedding_dim, self._write_lock_ctx
        )
        self.delta_mgr = DeltaDetector(self.conn, self.bundle_root)
        self.purge_mgr = PurgeManager(self.conn, self._write_lock_ctx)

        self.schema_mgr._ensure_schema()

        # SchemaManager may have adopted a stored embedding dimension
        # (e.g. opening a 512-dim DB with dim=12). Sync it back so the
        # facade and the embedding engine agree.
        self.embedding_dim = self.schema_mgr.embedding_dim
        # The encoder was built pre-adoption: its factory now reads the
        # adopted dim lazily (above), but its stored dim/reporting must
        # agree too — same pattern as search_engine._search_available below.
        self.encoder._truncate_dim = self.embedding_dim
        self.embed_engine = EmbeddingEngine(
            self.encoder, self.embedding_dim,
            self.device, self.cache_dir, self.model_id, self.omni_model_id,
            self._omni, self.chunk_size, self.chunk_overlap, self.enable_chunking,
            self.conn, self.ort_dylib,
        )
        self.image_mgr = ImageAssetManager(
            self.conn, self.embed_engine, self.schema_mgr,
            self.allow_remote_images, self.allowed_image_domains, self.bundle_root,
            self.db,
        )
        self.search_engine = SearchEngine(
            self.conn, self.embedding_dim, self.embed_engine, self.db,
        )
        # `_search_available` is owned by SchemaManager (set in `_ensure_schema`);
        # SearchEngine needs it to decide whether to raise on unavailable search.
        self.search_engine._search_available = self.schema_mgr._search_available
        self.import_mgr = ImportManager(
            self.conn, self.bundle_root, self._write_lock_ctx,
            self.enable_chunking, self.schema_mgr, self.delta_mgr, self.embed_engine,
            self.image_mgr, self.purge_mgr,
            self.encoder.count_tokens, embroider.MAX_LENGTH,
        )
        self.ingest_mgr = IngestManager(
            self._write_lock_ctx, self.bundle_root, self.device,
            self.import_mgr, self.delta_mgr, converter,
        )
        self.export_mgr = ExportManager(self.conn, self.search_engine)
        self.doctor_mgr = DoctorManager(self.conn, self.import_mgr)
        self.diff_mgr = DiffManager(self.conn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Checkpoint and close the database connection.

        Ladybug allows only one live handle per database file, and an open
        writer that never checkpoints can leave a WAL that a later reader
        refuses to open ("WAL file is corrupted"). Ingestion tools should call
        this (or use the router as a context manager) so the store is flushed
        before another process — e.g. the search browser — opens it.
        """
        if getattr(self, "conn", None) is None:
            return
        try:
            self.conn.execute("CHECKPOINT")
        except Exception as e:
            logger.debug("checkpoint on close failed: %s", e)
        for handle in (getattr(self, "conn", None), getattr(self, "db", None)):
            try:
                if handle is not None and hasattr(handle, "close") and not getattr(handle, "is_closed", False):
                    handle.close()
            except Exception as e:
                logger.debug("close failed: %s", e)
        self.conn = None
        self.db = None

        # Release write lock if held (Gap #7b)
        if hasattr(self, "_write_lock") and self._write_lock is not None:
            try:
                self._write_lock.release()
            except Exception:
                pass
            self._write_lock = None

    @contextmanager
    def _write_lock_ctx(self):
        """Context manager for inter-process write locking (Gap #7b).

        Acquires the file-based lock before any write operation and releases
        it on exit. If the lock cannot be acquired within the timeout (5 min),
        raises a RuntimeError.
        """
        if self._write_lock is None:
            # No lock configured — proceed without locking
            yield
            return

        timeout = getattr(self, "_write_lock_timeout", 300)
        try:
            acquired = self._write_lock.acquire(timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Failed to acquire write lock: {e}") from e

        if not acquired:
            raise RuntimeError(
                "Write lock acquisition timed out (another process is writing). "
                "Wait for the other writer to finish or increase the timeout."
            )

        try:
            yield
        finally:
            try:
                self._write_lock.release()
            except Exception:
                pass

    def __enter__(self) -> "OKFRouter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Component-backed public API (Phase 2 refactor)
    # Explicit 1-line proxies so the public surface stays on the facade.
    # These make the API explicit and survive class-level ``hasattr``
    # checks in tests; all remaining methods live on the components and
    # are reached directly (the Phase 1 __getattr__ bridge was removed in Phase 4).
    # ------------------------------------------------------------------

    def get_by_id(self, concept_id: str):
        return self.search_engine.get_by_id(concept_id)

    def list_directory(self, directory_id: str):
        return self.search_engine.list_directory(directory_id)

    def search_hybrid(self, *args, **kwargs):
        return self.search_engine.search_hybrid(*args, **kwargs)

    def traverse(self, *args, **kwargs):
        return self.search_engine.traverse(*args, **kwargs)

    def import_from_okf(self, *args, **kwargs):
        return self.import_mgr.import_from_okf(*args, **kwargs)

    def export_to_okf(self, *args, **kwargs):
        return self.export_mgr.export_to_okf(*args, **kwargs)

    def list_broken_links(self, *args, **kwargs):
        return self.import_mgr.list_broken_links(*args, **kwargs)

    def repair_links(self, *args, **kwargs):
        return self.import_mgr.repair_links(*args, **kwargs)

    def diagnose(self, *args, **kwargs):
        return self.doctor_mgr.diagnose(*args, **kwargs)

    def doctor_fix(self, *args, **kwargs):
        return self.doctor_mgr.fix(*args, **kwargs)

    def diff_dirs(self, *args, **kwargs):
        return self.diff_mgr.diff_dirs(*args, **kwargs)

    def diff_db_dir(self, *args, **kwargs):
        return self.diff_mgr.diff_db_dir(*args, **kwargs)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Search index (re)build
    # ------------------------------------------------------------------

    # (table, index_name, create-statement) for every vector/FTS index.
    # ------------------------------------------------------------------
    # Index dirty-tracking (Meta key/value markers)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Delta detection (file-level hash skip)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Purge (safe deletion of a concept and all its dependents)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Soft-Delete with Recovery (Gap #1d)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Omni (multimodal) embedding — lazy-loaded
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    # Source files the ingestion pipeline understands. Frontmatter is honoured
    # when present (Markdown); plain .txt is treated as body-only.
    SUPPORTED_SOURCE_EXTS = (".md", ".markdown", ".txt")

    # ------------------------------------------------------------------
    # Image ingestion (unified text / omni embedding space)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Broken Links
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Graph-Aware Retrieval
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Chunk Query
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Markdown Linting (Gap #5c)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Single-File Import Helpers (Gap #5c)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PDF Ingestion (Gap #5b)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Markdown Ingestion (Gap #5c)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Thought Ingestion (Gap #5c)
    # ------------------------------------------------------------------

