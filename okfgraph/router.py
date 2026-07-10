"""OKFRouter — Ladybug-backed knowledge graph with ONNX + Jina v5 embeddings.

Design choices:
    - **No torch dependency for inference**: The ONNX model (via optimum)
      returns numpy arrays. All post-processing (last-token pooling, L2
      normalisation, Matryoshka truncation) is done with numpy, avoiding
      the need for a CUDA-compiled torch. This lets the router run on any
      Python environment that has numpy and onnxruntime-gpu, without
      requiring torch at all.
    - **ONNX via optimum**: The jina-embeddings-v5 model is loaded as an
      ORTModelForFeatureExtraction (export=False) so it runs the raw ONNX
      graph through ONNX Runtime. Providers can be set to
      CUDAExecutionProvider for GPU acceleration or CPUExecutionProvider
      for CPU-only environments.
    - **Numpy tensors**: Tokenizer output uses ``return_tensors="np"`` so
      the entire pipeline stays on numpy. The ONNX model accepts numpy
      inputs and returns numpy outputs, keeping the code free of torch
      tensor operations.
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
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

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
        device: str = "cpu",
        allow_remote_images: bool = False,
        allowed_image_domains: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 40,
        enable_chunking: bool = True,
        wal_mode: bool = False,
    ):
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

        # ONNX tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,  # Required for Jina's custom tokenizer
            cache_dir=cache_dir,
        )

        # Provider selection with fallback
        self._cuda_fallback = False

        try:
            if device == "cuda":
                # NOTE: optimum 2.1.0 has a bug where passing a list as the
                # "provider" arg causes double-wrapping (['CUDA','CPU'] ->
                # [['CUDA','CPU']]) and validation fails.  We work around it
                # by passing the list via the "providers" kwarg instead.
                #
                # Also: IO binding with CUDA provider allocates torch tensors
                # via torch.empty(..., device=self.device), which fails when
                # torch was built without CUDA support.  Disable IO binding so
                # optimum uses numpy arrays internally and lets ONNX Runtime
                # handle GPU memory directly.
                self.embedder = ORTModelForFeatureExtraction.from_pretrained(
                    model_id,
                    export=False,
                    subfolder="onnx",
                    cache_dir=cache_dir,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    use_io_binding=False,
                )
            else:
                self.embedder = ORTModelForFeatureExtraction.from_pretrained(
                    model_id,
                    export=False,
                    subfolder="onnx",
                    cache_dir=cache_dir,
                )
        except ValueError as e:
            # CUDA not available — fall back to CPU
            if device == "cuda":
                self._cuda_fallback = True
                logger.warning(
                    f"CUDA unavailable ({e}) — falling back to CPUExecutionProvider. "
                    f"Install onnxruntime-gpu for GPU acceleration: pip install onnxruntime-gpu"
                )
                self.embedder = ORTModelForFeatureExtraction.from_pretrained(
                    model_id,
                    export=False,
                    subfolder="onnx",
                    cache_dir=cache_dir,
                )
            else:
                raise

        # Detect actual provider used (skip if already warned via fallback)
        if not self._cuda_fallback:
            try:
                actual_provider = self.embedder.session.get_providers()[0]
            except Exception:
                actual_provider = "CPUExecutionProvider"
            if device == "cuda" and actual_provider != "CUDAExecutionProvider":
                logger.warning(
                    f"CUDA requested but {actual_provider} is active. "
                    f"Install onnxruntime-gpu for GPU acceleration: pip install onnxruntime-gpu"
                )

        # ── Component wiring (Phase 1-3 refactor) ───────────────────
        # The facade owns the resources (conn, embedder, tokenizer, lock)
        # and injects them into focused component objects. ImportManager,
        # IngestManager and ExportManager are wired in Phase 3.
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
        self.embed_engine = EmbeddingEngine(
            self.embedder, self.tokenizer, self.embedding_dim,
            self.device, self.cache_dir, self.model_id, self.omni_model_id,
            self._omni, self.chunk_size, self.chunk_overlap, self.enable_chunking,
            self.conn,
        )
        self.image_mgr = ImageAssetManager(
            self.conn, self.embed_engine, self.schema_mgr,
            self.allow_remote_images, self.allowed_image_domains, self.bundle_root,
        )
        self.search_engine = SearchEngine(
            self.conn, self.tokenizer, self.embedding_dim, self.embed_engine,
        )
        # `_search_available` is owned by SchemaManager (set in `_ensure_schema`);
        # SearchEngine needs it to decide whether to raise on unavailable search.
        self.search_engine._search_available = self.schema_mgr._search_available
        self.import_mgr = ImportManager(
            self.conn, self.bundle_root, self._write_lock_ctx, self.tokenizer,
            self.enable_chunking, self.schema_mgr, self.delta_mgr, self.embed_engine,
            self.image_mgr, self.purge_mgr,
        )
        self.ingest_mgr = IngestManager(
            self._write_lock_ctx, self.bundle_root, self.device,
            self.import_mgr, self.delta_mgr,
        )
        self.export_mgr = ExportManager(self.conn, self.search_engine)

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

