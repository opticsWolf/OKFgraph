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
      produces vectors in a different embedding space that will NOT match
      stored vectors.
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

from okfgraph.ops import AdminOps, ExportOps, IngestOps, QueryOps

logger = logging.getLogger(__name__)


class OKFRouter(AdminOps, ExportOps, IngestOps, QueryOps):
    """Routes OKF concepts through a Ladybug graph + vector + FTS database.

    Canonical operations (``okfgraph.ops`` mixins) are part of the facade:
    ``search``, ``read``, ``traverse`` (QueryOps) and the ingest/export/admin
    ops as they land. Component-backed helpers stay reachable but should
    not be used by surface adapters.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    # Valid Matryoshka truncation levels for jina-embeddings-v5-text.
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
        bundle_root: Optional[str] = None,
        model_id: str = "jinaai/jina-embeddings-v5-text-small-retrieval",
        embedding_dim: int = 512,
        max_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "auto",
        precision: str = "auto",
        cpu_arena: bool = False,
        image_model_id: Optional[str] = None,
        image_precision: str = "auto",
        allow_remote_images: bool = False,
        allowed_image_domains: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 40,
        enable_chunking: bool = True,
        wal_mode: bool = False,
        converter=None,
        roots: Optional[Dict[str, str]] = None,
    ):
        """Open (or create) the graph database and wire up all managers.

        Args:
            db_path: Path to the Ladybug database file.
            bundle_root: Root directory of the OKF markdown bundle.
                Optional (file-free mode): thoughts ingest, search, read,
                traverse, doctor, and export all work without one. Any
                file-side call (bundle import/diff/detach, image staging
                outside an explicit file's dir) fails fast naming the
                missing root instead of crashing on a None path.
            model_id: HuggingFace ID of the Jina v5 text embedding model.
            embedding_dim: Truncated Matryoshka dimension (<= 1024).
            cache_dir: Model cache directory (defaults per-platform).
            model_path: Explicit local ONNX file (with `tokenizer_path`);
                skips every download — air-gapped / reproducible installs.
                An external-data sidecar must sit next to this file.
            tokenizer_path: Explicit local `tokenizer.json` (with `model_path`).
            device: "auto" (default) resolves to CUDA when the loaded ORT
                has a CUDA provider, else CPU. "cpu" / "cuda" pin it;
                CUDA-requested-but-missing warns and degrades to CPU.
            precision: "auto" (default) follows the resolved device
                (CUDA → FP16, CPU → FP32). "fp32" / "fp16" pin it;
                explicit fp16 on CPU warns (slow, not corrupt). FP16
                weights download from the published mirror repo; explicit
                `model_path` files bypass selection (they report fp32).
                The first session open pins the graph's precision in Meta
                and later opens refuse on mismatch — never mix precisions
                in one graph (reimport fresh to switch).
            cpu_arena: Enable the CPU arena allocator (default False: ~8x
                lower peak RSS for ~1.4x encode time, measured).
            image_model_id: Vision model id for image-content search
                (default: embroider's vision contract; registry only, no
                explicit-files path). Image vectors share the vision
                model's text-partner space — a non-partner text model is
                refused at first vision encode (use mode=text captions).
            image_precision: "auto" (default, follows device: CUDA →
                FP16, CPU → FP32), "fp32" or "fp16". Explicit fp16 on
                CPU fails fast (the vision graph stalls on CPU). Pinned
                per graph like the text precision — never mix.
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
        from okfgraph.components.embedding import resolve_ort_dylib
        # Resolve first: the native module loads ORT dynamically, so the
        # shared runtime choice must be fixed before importing it. Imported
        # here (not lazily below) so validation above can read its constants.
        self.ort_dylib = resolve_ort_dylib()
        try:
            import embroider
        except ImportError:
            raise RuntimeError(
                "the embroider wheel is required for text embeddings: "
                "pip install 'embroider>=0.2,<0.3'"
            ) from None
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
        # Token truncation ceiling (0.3.0, embroider>=0.1.4): None selects
        # the compat default 8192; 1..=32768 (Qwen3 position ceiling).
        # Long-doc vectors change when the limit is raised — reimport fully
        # after changing it, don't mix limits in one graph.
        if max_length is not None and not 1 <= max_length <= embroider.MODEL_MAX_TOKENS:
            raise ValueError(
                f"max_length must be within 1..={embroider.MODEL_MAX_TOKENS}, got {max_length}"
            )
        self.max_length = max_length if max_length is not None else embroider.MAX_LENGTH

        self.db = lb.Database(db_path)
        self.conn = lb.Connection(self.db)
        self.db_path = db_path

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

        self.bundle_root = Path(bundle_root).resolve() if bundle_root is not None else None
        # Multi-root (0.4.0, Phase 2 §2.1): additional {alias: path} trees.
        # The constructor bundle_root stays the primary tree (bare IDs);
        # named roots mint `@alias/rel` IDs. Validated fail-fast: unknown
        # shapes, bad aliases, and overlapping trees are all rejected here.
        # A None primary (file-free mode) simply skips the overlap check.
        from okfgraph.components.roots import validate_roots as _vr
        self.roots = _vr(
            roots,
            primary=str(self.bundle_root) if self.bundle_root is not None else None,
        )
        self.embedding_dim = embedding_dim
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self.allow_remote_images = allow_remote_images
        self.allowed_image_domains = allowed_image_domains or []

        # Chunking configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_chunking = enable_chunking

        # Text embeddings: Rust embroider wheel (Jina v5 via ORT). No Python
        # fallback — a mid-run stack switch would silently mix vector spaces
        # in one index.
        from okfgraph.components.embedding import LazyRustEncoder
        # The Rust crate knows auto/cpu/cuda; map torch-style aliases.
        # Validate eagerly so a bad device still fails at construction —
        # the session itself opens lazily on first encode (see below).
        rust_device = {"mps": "auto"}.get(device, device)
        if rust_device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"device must be 'auto', 'cpu' or 'cuda', got '{device}'"
            )
        if precision not in ("auto", "fp32", "fp16", "int8"):
            raise ValueError(
                f"precision must be 'auto', 'fp32', 'fp16' or 'int8', got '{precision}'"
            )
        if image_precision not in ("auto", "fp32", "fp16"):
            raise ValueError(
                f"image_precision must be 'auto', 'fp32' or 'fp16', got "
                f"'{image_precision}' (vision ships no int8 artifact)"
            )
        # Default vision id comes off the installed wheel when it knows
        # the contract (>=0.3); the literal keeps older wheels failing
        # with "unknown image model" instead of AttributeError.
        if image_model_id is None:
            image_model_id = getattr(
                embroider, "VISION_NANO_MODEL",
                "jina-v5-omni-nano-retrieval-vision",
            )
        self.image_model_id = image_model_id
        self.image_precision = image_precision
        if not model_id or "/" not in model_id or any(
            c in model_id for c in ("'", '"', "\\", ";")
        ):
            raise ValueError(
                f"model_id must be 'owner/name' without quotes/semicolons, got '{model_id}'"
            )
        if precision == "fp16" and rust_device == "cpu":
            logger.warning(
                "precision='fp16' with device='cpu': FP16 on CPU runs >40x "
                "slower than FP32-CPU (emulated kernels). Use precision='auto' "
                "or 'fp32' for CPU sessions."
            )
        self.precision = precision
        self.cpu_arena = cpu_arena
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
            # Precision pin (fail-closed): the first open records the
            # landed precision; later opens refuse on mismatch so FP16
            # and FP32 vectors never share one graph.
            from okfgraph.components.embedding import (
                enforce_model_pin,
                enforce_precision_pin,
            )
            pinned = enforce_precision_pin(self.conn, encoder.precision)
            # Model pin (fail-closed, 0.6.0): different weights live in
            # different spaces — a model switch forces a fresh reimport.
            # Explicit local files bypass the registry (model_id is the
            # file path, unique per file set, so pinning still separates).
            enforce_model_pin(self.conn, model_id)
            logger.info(
                "text embeddings: %s dim=%d cuda=%s precision=%s",
                model_id, self.embedding_dim, encoder.used_cuda, pinned,
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
        def _ort_missing() -> bool:
            # Core installs carry no ORT wheel (it lives in okfgraph[cpu|gpu]).
            if self.ort_dylib is not None:
                return False
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                return True
            return False

        _ORT_HINT = ("no ONNX Runtime installed: pip install 'okfgraph[cpu]' "
                     "or 'okfgraph[gpu]' (exactly one), or set ORT_DYLIB_PATH "
                     "to an onnxruntime 1.29 library")

        if explicit_files:
            def _open_session():
                return embroider.JinaV5.open_files(
                    str(model_path),
                    str(tokenizer_path),
                    # Read off self: the stored dimension wins over the
                    # requested one (§4.1 adoption) and factories run lazily,
                    # after construction — so this is the adopted value.
                    truncate_dim=self.embedding_dim,
                    device=rust_device,
                    max_length=self.max_length,
                    cpu_arena=self.cpu_arena,
                )
            tokenizer_factory = lambda: embroider.JinaTokenizer.open_files(
                str(tokenizer_path),
            )
        else:
            def _open_session():
                return embroider.JinaV5.open(
                    model_id,
                    # See above: adopted dimension, resolved at first encode.
                    truncate_dim=self.embedding_dim,
                    device=rust_device,
                    cache_dir=cache_dir,
                    max_length=self.max_length,
                    precision=self.precision,
                    cpu_arena=self.cpu_arena,
                )
            tokenizer_factory = lambda: embroider.JinaTokenizer.open(
                model_id,
                cache_dir=cache_dir,
            )

        def session_factory():
            if _ort_missing():
                # Fail before JinaV5.open: a failed ORT init poisons ort's
                # global lock and aborts the process at teardown, so the
                # backend must never be touched when no runtime exists.
                raise RuntimeError(_ORT_HINT)
            # No post-hoc wrap: _ort_missing() already returned False and
            # nothing changes in between, so a second check after a failed
            # open could never fire. Backend errors (including pyo3
            # PanicException, a BaseException) propagate raw and are cached
            # by LazyRustEncoder.
            return _open_session()
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
        # (The vision encoder's stored dim is synced where it is built.)
        self.encoder._truncate_dim = self.embedding_dim
        # Vision session (0.8.0): same laziness as text — compat and pins
        # run inside the factory, so text-only graphs never pay for them.
        # No explicit-files path: vision opens by registry id only.
        from okfgraph.components.embedding import (
            enforce_image_model_pin,
            enforce_image_precision_pin,
            enforce_vision_compat,
            LazyVisionEncoder,
        )

        def _report_vision_open(encoder) -> None:
            pinned = enforce_image_precision_pin(self.conn, encoder.precision)
            enforce_image_model_pin(self.conn, image_model_id)
            logger.info(
                "image embeddings: %s dim=%d cuda=%s precision=%s",
                image_model_id, self.embedding_dim, encoder.used_cuda, pinned,
            )

        def _vision_session_factory():
            if _ort_missing():
                raise RuntimeError(_ORT_HINT)
            # Fail-closed compat BEFORE any download: wrong text model,
            # oversize dim, or unknown image id all refuse here.
            enforce_vision_compat(
                self.conn,
                text_model_id=model_id,
                embedding_dim=self.embedding_dim,
                image_model_id=image_model_id,
            )
            return embroider.JinaV5Vision.open(
                image_model_id,
                truncate_dim=self.embedding_dim,
                device=rust_device,
                cache_dir=cache_dir,
                precision=image_precision,
                gpu_mem_limit=None,
            )

        self.vision_encoder = LazyVisionEncoder(
            image_model_id=image_model_id,
            truncate_dim=embedding_dim,
            device=rust_device,
            session_factory=_vision_session_factory,
            on_open=_report_vision_open,
            precision=image_precision,
        )
        # Same pre-adoption sync as the text encoder above: the factory
        # reads the adopted dim lazily, the stored dim follows it here.
        self.vision_encoder._truncate_dim = self.embedding_dim
        self.embed_engine = EmbeddingEngine(
            self.encoder, self.embedding_dim,
            self.device, self.cache_dir, self.model_id,
            self.chunk_size, self.chunk_overlap, self.enable_chunking,
            self.conn, self.ort_dylib,
            vision_encoder=self.vision_encoder,
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
            self.encoder.count_tokens, self.max_length,
            db_path=db_path, roots=self.roots,
        )
        self.ingest_mgr = IngestManager(
            self._write_lock_ctx, self.bundle_root, self.device,
            self.import_mgr, self.delta_mgr, converter,
        )
        self.export_mgr = ExportManager(self.conn, self.search_engine)
        self.doctor_mgr = DoctorManager(self.conn, self.import_mgr)
        self.diff_mgr = DiffManager(
            self.conn, roots=self.roots, primary=self.bundle_root
        )

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
            raise OKFError("WRITE_LOCK_TIMEOUT",
                           f"Failed to acquire write lock: {e}") from e

        if not acquired:
            raise OKFError(
                "WRITE_LOCK_TIMEOUT",
                "Write lock acquisition timed out (another process is writing). "
                "Wait for the other writer to finish or increase the timeout.",
                remedy="close concurrent writers or raise the lock timeout",
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
    # Image ingestion (caption-based text embeddings since 0.7.0)
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

