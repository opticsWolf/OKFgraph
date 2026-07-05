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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import frontmatter
import ladybug as lb
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

logger = logging.getLogger(__name__)


class OKFRouter:
    """Routes OKF concepts through a Ladybug graph + vector + FTS database."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    # Valid Matryoshka truncation levels shared by jina-embeddings-v5
    # (text-small-retrieval and omni-small-retrieval both support these).
    ALLOWED_DIMS = (32, 64, 128, 256, 512, 768, 1024)

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
        chunk_size: int = 512,
        chunk_overlap: int = 40,
        enable_chunking: bool = True,
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
        self.bundle_root = Path(bundle_root).resolve()
        self.embedding_dim = embedding_dim
        self.model_id = model_id
        self.omni_model_id = omni_model_id
        self.cache_dir = cache_dir
        self.device = device
        self.allow_remote_images = allow_remote_images

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

        self._ensure_schema()

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

    def __enter__(self) -> "OKFRouter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Search index (re)build
    # ------------------------------------------------------------------

    # (table, index_name, create-statement) for every vector/FTS index.
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
        return self._build_search_indexes(rebuild=True, force=force)

    # ------------------------------------------------------------------
    # Index dirty-tracking (Meta key/value markers)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _encode(self, text: str, task: str = "Document") -> List[float]:
        """Encode text with ONNX Jina v5 model.

        Uses numpy exclusively — no torch dependency. The ONNX model
        (via optimum) returns numpy arrays, and all post-processing is
        done with numpy operations.

        Args:
            text: Raw text to encode.
            task: ``"Query"`` or ``"Document"`` — controls the prefix.

        Returns:
            L2-normalised embedding vector (list of floats), truncated to
            the configured Matryoshka dimension.
        """
        # Apply prefix (avoid double-prefixing)
        if not text.startswith(("Query:", "Document:")):
            text = f"{task}: {text}"

        # Tokenise — return numpy arrays so the entire pipeline stays on numpy.
        # This avoids any torch dependency; the ONNX model accepts numpy inputs.
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=8192,
            padding=True,
        )

        # ONNX forward pass — outputs are numpy arrays (no torch needed)
        outputs = self.embedder(**inputs)

        # Last-token pooling (REQUIRED by jina-embeddings-v5; mean pooling
        # produces vectors in a different space that will NOT align with the
        # omni model's image embeddings in the unified ImageAsset index).
        last_hidden = outputs.last_hidden_state           # (B, T, H)
        attention_mask = inputs["attention_mask"]         # (B, T)
        last_idx = attention_mask.sum(axis=1) - 1         # index of final real token
        last_idx = np.clip(last_idx, 0, None)             # clamp to >= 0
        pooled = last_hidden[np.arange(last_hidden.shape[0]), last_idx]  # (B, H)

        # L2 normalisation
        norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
        normalized = pooled / norm

        # Matryoshka truncation (+ re-normalisation to keep unit norm)
        vec = normalized[0].tolist()
        return self._truncate_normalize(vec)

    def _truncate_normalize(self, vec: List[float]) -> List[float]:
        """Truncate to the configured Matryoshka dimension and L2-renormalise.

        Both the text and omni encoders pass through here so every vector that
        lands in a Ladybug FLOAT[dim] column is unit-norm and exactly dim-long.
        """
        v = list(vec[: self.embedding_dim])
        if len(v) < self.embedding_dim:
            v = v + [0.0] * (self.embedding_dim - len(v))
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            v = [x / norm for x in v]
        return v

    def _encode_batch(
        self, texts: List[str], task: str = "Document"
    ) -> List[List[float]]:
        """Encode multiple texts in a single ONNX forward pass.

        Uses numpy exclusively — no torch dependency. Each text is encoded
        sequentially (not batched into a single ONNX call) because with
        variable-length texts (80-300 words), padding all to the longest
        in the batch causes massive attention compute waste
        (O(batch * max_len^2)). Sequential single-pass encoding is faster
        because each text only processes its actual token count.

        Args:
            texts: List of raw texts to encode.
            task: ``"Query"`` or ``"Document"`` — controls the prefix.

        Returns:
            List of L2-normalised embedding vectors (each truncated to target dim).
        """
        if not texts:
            return []

        # Apply prefixes
        prefixed = [
            f"{task}: {t}" if not t.startswith(("Query:", "Document:")) else t
            for t in texts
        ]

        # Sequential encoding — each text only processes its actual token count.
        return [self._encode(t, task) for t in texts]

    # ------------------------------------------------------------------
    # Omni (multimodal) embedding — lazy-loaded
    # ------------------------------------------------------------------

    def _get_omni(self):
        """Load the omni model on first use (vision + text towers only)."""
        if self._omni is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading omni model %s (vision modality) on %s ...",
                self.omni_model_id, self.device,
            )
            self._omni = SentenceTransformer(
                self.omni_model_id,
                trust_remote_code=True,
                cache_folder=self.cache_dir,
                device=self.device,
                model_kwargs={"modality": "vision"},  # skip the audio tower
            )
        return self._omni

    def _encode_image(self, data: bytes) -> List[float]:
        """Embed raw image bytes with the omni model (shared vector space)."""
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        model = self._get_omni()
        vec = model.encode(
            img,
            truncate_dim=self.embedding_dim,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._truncate_normalize([float(x) for x in list(vec)])

    def _encode_omni_text(self, text: str, task: str = "Query") -> List[float]:
        """Embed text with the omni model's text side (for cross-modal queries)."""
        model = self._get_omni()
        encoder = model.encode_query if task == "Query" else model.encode_document
        vec = encoder(text, truncate_dim=self.embedding_dim)
        return self._truncate_normalize([float(x) for x in list(vec)])

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_cache_dir() -> str:
        """Return the HuggingFace default cache directory."""
        import os
        return os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))

    @classmethod
    def model_info(cls, model_id: str = "jinaai/jina-embeddings-v5-text-small-retrieval",
                   cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """Inspect model cache status without loading the model.

        Returns a dict with cache location, snapshot path, and disk usage.
        """
        from huggingface_hub import list_repo_files, snapshot_download

        effective_cache = cache_dir or cls.default_cache_dir()
        info: Dict[str, Any] = {
            "model_id": model_id,
            "cache_dir": effective_cache,
            "cached": False,
            "snapshot_path": None,
            "disk_usage_bytes": 0,
        }

        try:
            snapshot_path = snapshot_download(
                model_id,
                cache_dir=effective_cache,
                local_files_only=True,
            )
            info["cached"] = True
            info["snapshot_path"] = snapshot_path
            # Calculate disk usage
            snap = Path(snapshot_path)
            if snap.exists():
                info["disk_usage_bytes"] = sum(
                    f.stat().st_size for f in snap.rglob("*") if f.is_file()
                )
        except Exception:
            pass  # Not cached locally — will download on first use

        return info

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _split_into_chunks(
        self, body: str, document_id: str
    ) -> List[Dict[str, Any]]:
        """Split document body into pure blocks using mordant chunker.

        Uses chunker.get_all_chunks() to get ExtractedChunk objects with
        block_type and byte offsets. Includes headings as separate chunks
        so they are preserved during reconstruction. No overlap is stored.
        """
        chunker = mordant.MarkdownChunker(body)
        chunks: List[Dict[str, Any]] = []
        index = 0

        current_heading = ""
        for chunk in chunker.get_all_chunks():
            # Track the heading context as we move down the document
            if chunk.block_type == "Heading":
                current_heading = chunk.text

            chunks.append({
                "parent_doc_id": document_id,
                "chunk_text": chunk.text,
                "block_type": chunk.block_type,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "chunk_index": index,
                # Ephemeral context used strictly for constructing the embedding payload
                "heading_context": current_heading if chunk.block_type != "Heading" else ""
            })
            index += 1

        return chunks

    def _compute_overlap_payloads(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add context injection and token overlap in memory for embedding.

        Combines heading context structures and token tails into a deep
        semantic representation for the encoder without mutating the raw text store.
        """
        payloads: List[Dict[str, Any]] = []
        prev_tail = ""

        # Structural blocks represent hard semantic boundaries. They should not
        # receive tails from preceding prose, nor generate tails that bleed into
        # subsequent prose.
        STRUCTURAL_BLOCKS = (
            "Heading",
            "CodeBlock",
            "List",
            "Blockquote",
            "Table",
            "Diagram",
        )

        for chunk in chunks:
            text_to_embed = chunk['chunk_text']

            # 1. Enforce hard semantic boundary
            # Clear any trailing words from the previous section when hitting a structural block
            if chunk["block_type"] in STRUCTURAL_BLOCKS:
                prev_tail = ""

            # 2. Apply sliding word boundary window if a tail exists
            if prev_tail:
                text_to_embed = f"{prev_tail}\n\n{text_to_embed}"

            # 3. Prepend structural Heading Context if available
            if chunk.get("heading_context"):
                text_to_embed = f"{chunk['heading_context']}\n\n{text_to_embed}"

            payloads.append({
                "chunk_id": f"{chunk['parent_doc_id']}#chunk:{chunk['chunk_index']}",
                "text": text_to_embed,
            })

            # Compute tail from the PURE chunk text (not the context-enriched string)
            # Structural blocks never generate tails
            if self.chunk_overlap > 0 and chunk["block_type"] not in STRUCTURAL_BLOCKS:
                words = chunk["chunk_text"].split()
                prev_tail = "  ".join(words[-self.chunk_overlap:])
            else:
                prev_tail = ""

        return payloads

    def reconstruct_document(self, document_id: str) -> str:
        """Reconstruct original markdown from stored chunks.

        Uses block_type to determine correct delimiters between chunks.
        Approximate byte-exact reconstruction (~98% fidelity).
        """
        result = self.conn.execute("""
            MATCH (ch:Chunk)
            WHERE ch.parent_doc_id = $id
            RETURN ch.chunk_text AS chunk_text, ch.block_type AS block_type, ch.chunk_index AS chunk_index
            ORDER BY ch.chunk_index
        """, {"id": document_id})
        rows = result.rows_as_dict().get_all()

        if not rows:
            return None

        parts = [rows[0]["chunk_text"]]
        for i in range(1, len(rows)):
            sep = mordant.MarkdownChunker.get_delimiter(
                rows[i - 1]["block_type"], rows[i]["block_type"]
            )
            parts.append(sep + rows[i]["chunk_text"])

        return "".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_property(self, cid: str, prop: str) -> Any:
        """Retrieve a single property from a concept node."""
        result = self.conn.execute(
            f"MATCH (c:Concept {{id: $id}}) RETURN c.{prop}",
            {"id": cid},
        )
        row = result.rows_as_dict().get_all()
        return row[0][f"c.{prop}"] if row else None

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    # Source files the ingestion pipeline understands. Frontmatter is honoured
    # when present (Markdown); plain .txt is treated as body-only.
    SUPPORTED_SOURCE_EXTS = (".md", ".markdown", ".txt")

    def _parse_source_file(
        self, file_path: Path, root: Path
    ) -> Tuple[ConceptModel, str, str]:
        """Parse a .md/.txt source into ``(ConceptModel, body, concept_id)``.

        Frontmatter is used when present. Plain-text files (and Markdown lacking
        frontmatter) get a synthesized ``type`` ('note') and a ``title`` derived
        from the filename, so the simplified text-only pipeline can ingest .txt
        alongside .md without every file needing OKF frontmatter.
        """
        post = frontmatter.load(file_path)
        body = post.content
        fm = dict(post.metadata)

        rel_path = file_path.relative_to(root) if file_path.is_relative_to(root) else None
        if rel_path is not None:
            # with_suffix("") strips only the final extension (.md/.txt/.markdown),
            # avoiding the old str.replace(".md","") which could corrupt paths.
            concept_id = str(rel_path.with_suffix("")).replace("\\", "/")
        else:
            # File lives outside bundle_root (common when a GUI writes each .md
            # next to its source). Fall back to the bare stem so the import
            # doesn't crash with a relative_to ValueError.
            concept_id = file_path.stem

        if not fm.get("type"):
            fm["type"] = "note"
        if not fm.get("title"):
            stem = file_path.stem.replace("_", " ").replace("-", " ").strip()
            fm["title"] = stem or concept_id

        concept = ConceptModel.model_validate({**fm, "id": concept_id, "body": body})
        return concept, body, concept_id

    def import_from_okf(
        self,
        file_path: Path,
        mode: "str | IngestMode" = IngestMode.TEXT,
        rebuild_indexes: bool = True,
    ) -> str:
        """Parse an OKF .md/.txt file and create/update the concept in the graph.

        Args:
            file_path: Path to the source file (``.md``, ``.markdown`` or ``.txt``).
            mode: Image ingestion mode — ``text`` (alt-text / filename fallback,
                no omni model), ``optional`` (omni only for images without
                alt-text), or ``omni`` (omni for every image).

        Returns the concept ID (relative path without its extension).
        """
        mode = IngestMode.coerce(mode)

        # 1-2. Parse frontmatter/body and build the Concept model.
        concept, body, concept_id = self._parse_source_file(file_path, self.bundle_root)

        # 2.5. Chunk the body (NEW)
        if self.enable_chunking:
            # Delete old chunks for this document (re-import)
            self.conn.execute(
                "MATCH (c:Concept {id: $id})-[:PART_OF]->(ch:Chunk) DELETE ch",
                {"id": concept_id},
            )

            chunks = self._split_into_chunks(body, concept_id)
            if chunks:
                # Compute overlap payloads for embedding
                payloads = self._compute_overlap_payloads(chunks)
                texts = [p["text"] for p in payloads]
                embeddings = self._encode_batch(texts, task="Document")

                self.conn.execute("BEGIN TRANSACTION")
                try:
                    for payload, emb in zip(payloads, embeddings):
                        chunk_id = payload["chunk_id"]
                        # Find the original chunk for metadata
                        orig_chunk = next(
                            c for c in chunks
                            if c["chunk_index"] == int(chunk_id.split(":")[-1])
                        )
                        self.conn.execute("""
                            CREATE (ch:Chunk {
                                id: $id, parent_doc_id: $doc_id,
                                chunk_index: $idx, chunk_text: $text,
                                block_type: $block_type,
                                start_offset: $start, end_offset: $end_offset,
                                embedding: $emb
                            })
                        """, {
                            "id": chunk_id,
                            "doc_id": concept_id,
                            "idx": orig_chunk["chunk_index"],
                            "text": orig_chunk["chunk_text"],
                            "block_type": orig_chunk["block_type"],
                            "start": orig_chunk["start_offset"],
                            "end_offset": orig_chunk["end_offset"],
                            "emb": emb,
                        })
                    self.conn.execute("COMMIT")
                except Exception:
                    try:
                        self.conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                self._bump_write_epoch()

        # 3. Generate embedding (Document prefix)
        search_text = f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
        concept.embedding = self._encode(search_text, task="Document")

        # 4. Insert into graph (delegates to shared upsert logic)
        self._insert_concept(concept, body, concept_id)

        # 4.5. Create PART_OF relationships between Concept and its Chunks
        #      (must happen after _insert_concept so the Concept node exists)
        if self.enable_chunking:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                for chunk in chunks:
                    chunk_id = f"{concept_id}#chunk:{chunk['chunk_index']}"
                    self.conn.execute("""
                        MATCH (d:Concept {id: $doc})
                        MATCH (ch:Chunk {id: $cid})
                        MERGE (d)-[:PART_OF]->(ch)
                    """, {"doc": concept_id, "cid": chunk_id})
                self.conn.execute("COMMIT")
            except Exception:
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

        # 5. Ingest any embedded images under the requested mode
        self._ingest_concept_images(concept_id, body, file_path.parent, mode)

        # 6. Extract and create LINKS_TO relationships for this concept
        self._extract_links_for_concept(concept_id, body)

        # 7. Rebuild search indexes so this concept (and its images) are
        #    actually returned by vector/FTS search. Callers importing many
        #    files one-by-one can pass rebuild_indexes=False and call
        #    _build_search_indexes(rebuild=True) once at the end.
        if rebuild_indexes:
            self._build_search_indexes(rebuild=True)

        return concept_id

    def import_bundle(
        self,
        bundle_path: Optional[Path] = None,
        batch_size: int = 32,
        mode: "str | IngestMode" = IngestMode.TEXT,
    ) -> List[str]:
        """Import an entire OKF bundle directory with batched encoding.

        Walks the bundle directory, parses all .md files, generates
        embeddings in batched ONNX forward passes, and upserts them.

        Args:
            bundle_path: Root directory of the OKF bundle (defaults to constructor bundle_root).
            batch_size: Number of texts per ONNX forward pass.
            mode: Image ingestion mode (``text`` | ``optional`` | ``omni``).

        Returns:
            List of imported concept IDs.
        """
        mode = IngestMode.coerce(mode)
        root = bundle_path or self.bundle_root
        source_files = sorted(
            fp for fp in root.rglob("*")
            if fp.is_file() and fp.suffix.lower() in self.SUPPORTED_SOURCE_EXTS
        )
        if not source_files:
            return []

        # Phase 1: Parse all files
        parsed: List[Dict[str, Any]] = []
        for fp in source_files:
            try:
                concept, body, cid = self._parse_source_file(fp, root)
                search_text = f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
                parsed.append({
                    "concept": concept,
                    "search_text": search_text,
                    "body": body,
                    "cid": cid,
                    "dir": fp.parent,
                })
            except Exception as e:
                print(f"  [WARN] Skipping {fp.name}: {e}")

        # No-op guard: if nothing parsed successfully, do no work (no encode, no
        # transaction, no index rebuild).
        if not parsed:
            return []

        # Phase 2: Batch encode (chunked by batch_size)
        all_search_texts = [p["search_text"] for p in parsed]
        all_embeddings: List[List[float]] = []
        for i in range(0, len(all_search_texts), batch_size):
            chunk = all_search_texts[i : i + batch_size]
            batch_embs = self._encode_batch(chunk, task="Document")
            all_embeddings.extend(batch_embs)

        # Phase 3: Batch upsert all concepts in a single transaction
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self._batch_upsert_concepts(parsed, all_embeddings)
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        # Phase 3.5: Chunk all documents (NEW)
        if self.enable_chunking:
            all_chunks: List[Dict[str, Any]] = []
            for p in parsed:
                chunks = self._split_into_chunks(p["body"], p["cid"])
                all_chunks.extend(chunks)

            if all_chunks:
                # Group by parent doc for overlap computation
                chunks_by_doc: Dict[str, List[Dict]] = {}
                for c in all_chunks:
                    chunks_by_doc.setdefault(c["parent_doc_id"], []).append(c)

                all_payloads: List[Dict[str, Any]] = []
                for doc_id, doc_chunks in chunks_by_doc.items():
                    payloads = self._compute_overlap_payloads(doc_chunks)
                    for p in payloads:
                        p["_doc_id"] = doc_id
                    all_payloads.extend(payloads)

                # Batch encode
                texts = [p["text"] for p in all_payloads]
                embeddings = self._encode_batch(texts, task="Document")

                self.conn.execute("BEGIN TRANSACTION")
                try:
                    for payload, emb in zip(all_payloads, embeddings):
                        doc_id = payload.pop("_doc_id")
                        chunk_idx = int(payload["chunk_id"].split(":")[-1])
                        orig_chunk = next(
                            c for c in chunks_by_doc[doc_id]
                            if c["chunk_index"] == chunk_idx
                        )
                        self.conn.execute("""
                            CREATE (ch:Chunk {
                                id: $id, parent_doc_id: $doc_id,
                                chunk_index: $idx, chunk_text: $text,
                                block_type: $block_type,
                                start_offset: $start, end_offset: $end_offset,
                                embedding: $emb
                            })
                        """, {
                            "id": payload["chunk_id"],
                            "doc_id": doc_id,
                            "idx": orig_chunk["chunk_index"],
                            "text": orig_chunk["chunk_text"],
                            "block_type": orig_chunk["block_type"],
                            "start": orig_chunk["start_offset"],
                            "end_offset": orig_chunk["end_offset"],
                            "emb": emb,
                        })
                        self.conn.execute("""
                            MATCH (d:Concept {id: $doc})
                            MATCH (ch:Chunk {id: $cid})
                            MERGE (d)-[:PART_OF]->(ch)
                        """, {"doc": doc_id, "cid": payload["chunk_id"]})
                    self.conn.execute("COMMIT")
                except Exception:
                    try:
                        self.conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                self._bump_write_epoch()

        # Phase 4: Batch directory hierarchy (collected from all concept IDs)
        self._batch_build_directories([p["cid"] for p in parsed])

        # Phase 5: Batch link extraction
        self._batch_extract_links(parsed)

        # Phase 6: Image ingestion (per concept, honouring the selected mode)
        for p in parsed:
            try:
                self._ingest_concept_images(p["cid"], p["body"], p["dir"], mode)
            except Exception as e:  # never let one bad image abort the bundle
                print(f"  [WARN] Image ingestion failed for {p['cid']}: {e}")

        # Phase 7: rebuild vector + FTS indexes once so every concept/image
        # written above is searchable (indexes reflect table contents at build
        # time, not subsequent inserts).
        self._build_search_indexes(rebuild=True)

        return [p["cid"] for p in parsed]

    def _batch_upsert_concepts(
        self,
        parsed: List[Dict[str, Any]],
        all_embeddings: List[List[float]],
    ):
        """Upsert all concepts in one transaction.

        Deletes existing concepts, then creates new ones with embeddings.
        """
        # Collect IDs for bulk delete
        cids = [p["cid"] for p in parsed]
        if cids:
            # Bulk delete existing concepts. Use a bound parameter (never string
            # interpolation) and DETACH DELETE so concepts that already have
            # edges (LINKS_TO / CONTAINS / INCLUDES_ASSET) can be replaced on
            # re-import instead of raising a duplicated-primary-key error.
            self.conn.execute(
                "MATCH (c:Concept) WHERE c.id IN $ids DETACH DELETE c",
                {"ids": cids},
            )

        # Create all concepts
        for item, emb in zip(parsed, all_embeddings):
            concept = item["concept"]
            body = item["body"]
            concept_id_val = item["cid"]
            all_data = concept.model_dump()
            all_data.pop("body", None)
            all_data.pop("id", None)
            all_data.pop("embedding", None)  # embedding is passed separately, must not leak into extra MAP

            core = {
                "type": all_data.pop("type"),
                "title": all_data.pop("title", None),
                "description": all_data.pop("description", None),
                "resource": all_data.pop("resource", None),
                "tags": all_data.pop("tags", []),
                "timestamp": all_data.pop("timestamp", None),
            }

            extra = {
                k: json.dumps(v) if not isinstance(v, str) else v
                for k, v in all_data.items()
            }
            extra_keys = list(extra.keys())
            extra_values = list(extra.values())

            if isinstance(core["timestamp"], datetime):
                core["timestamp"] = core["timestamp"].isoformat()

            params: Dict[str, Any] = {
                "id": concept_id_val,
                "body": body,
                "embedding": emb,
                **core,
            }
            if extra_keys:
                params["extra_keys"] = extra_keys
                params["extra_values"] = extra_values
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding,
                        extra: MAP($extra_keys, $extra_values)
                    })
                """, params)
            else:
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding
                    })
                """, params)

        # Concepts changed -> search indexes are now stale. Bumping inside the
        # caller's transaction means the marker rolls back with the data if the
        # import aborts, keeping dirty-state consistent with what's committed.
        if parsed:
            self._bump_write_epoch()

    def _batch_build_directories(self, cids: List[str]):
        """Build directory hierarchy for a batch of concept IDs.

        Collects all unique directory paths and creates them in order
        (shallowest first) to ensure parents exist before children.
        """
        # Collect all unique directory paths
        dir_paths = set()
        for cid in cids:
            parts = cid.split("/")
            if len(parts) > 1:
                for i in range(1, len(parts)):
                    dir_paths.add("/".join(parts[:i]))

        # Sort by depth (shallowest first)
        sorted_dirs = sorted(dir_paths, key=lambda d: d.count("/"))

        # Create directory hierarchy
        for d in sorted_dirs:
            parent = "/".join(d.split("/")[:-1]) if "/" in d else None
            if parent and parent in dir_paths:
                self.conn.execute("""
                    MERGE (p:Directory {id: $parent})
                    MERGE (d:Directory {id: $child})
                    MERGE (p)-[:CONTAINS]->(d)
                """, {"parent": parent, "child": d})
            elif parent:
                # Parent is root (not a directory node)
                self.conn.execute("""
                    MERGE (d:Directory {id: $child})
                """, {"child": d})
            else:
                self.conn.execute("""
                    MERGE (d:Directory {id: $child})
                """, {"child": d})

        # Link each concept to its parent directory
        for cid in cids:
            parts = cid.split("/")
            if len(parts) > 1:
                parent_dir = "/".join(parts[:-1])
                self.conn.execute("""
                    MERGE (d:Directory {id: $parent})
                    MERGE (c:Concept {id: $child})
                    MERGE (d)-[:CONTAINS]->(c)
                """, {"parent": parent_dir, "child": cid})

    def _batch_extract_links(self, parsed: List[Dict[str, Any]]):
        """Extract and create LINKS_TO relationships for a batch of concepts.

        Collects all markdown links, checks which targets exist, and
        creates relationships or BrokenLink records in bulk.
        """
        # Collect all (source, target) pairs
        all_links: List[Tuple[str, str]] = []
        link_pattern = re.compile(r"\[.*?\]\((.*?\.md)\)")
        wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")
        for item in parsed:
            source_id = item["cid"]
            body = item["body"]
            for raw_link in link_pattern.findall(body):
                target_id = raw_link.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
                all_links.append((source_id, target_id))
            # Also handle wikilinks [[target]]
            for raw_link in wikilink_pattern.findall(body):
                target_id = raw_link.strip().lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
                all_links.append((source_id, target_id))

        if not all_links:
            return

        # Collect all unique target IDs
        all_targets = list(set(t for _, t in all_links))

        # Batch check which targets exist
        existing_targets = set()
        for target_id in all_targets:
            result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            cnt = result.rows_as_dict().get_all()[0]["cnt"]
            if cnt > 0:
                existing_targets.add(target_id)

        # Create LINKS_TO for existing targets
        for source_id, target_id in all_links:
            if target_id in existing_targets:
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": source_id, "target": target_id})
            else:
                # Record broken link for later repair
                link_id = f"{source_id}\u2192{target_id}"
                now = datetime.now()
                self.conn.execute("""
                    MERGE (bl:BrokenLink {id: $id})
                    SET bl.source_id = $source, bl.target_id = $target, bl.timestamp = $ts
                """, {"id": link_id, "source": source_id, "target": target_id, "ts": now})

    def _extract_links_for_concept(self, concept_id: str, body: str):
        """Extract and create LINKS_TO relationships for a single concept.

        Handles both markdown links [text](file.md) and wikilinks [[target]].
        """
        link_pattern = re.compile(r"\[.*?\]\((.*?\.md)\)")
        wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")

        all_links: List[Tuple[str, str]] = []
        for raw_link in link_pattern.findall(body):
            target_id = raw_link.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
            all_links.append((concept_id, target_id))
        for raw_link in wikilink_pattern.findall(body):
            target_id = raw_link.strip().lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
            all_links.append((concept_id, target_id))

        if not all_links:
            return

        # Collect all unique target IDs
        all_targets = list(set(t for _, t in all_links))

        # Batch check which targets exist
        existing_targets = set()
        for target_id in all_targets:
            result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            cnt = result.rows_as_dict().get_all()[0]["cnt"]
            if cnt > 0:
                existing_targets.add(target_id)

        # Create LINKS_TO for existing targets
        for source_id, target_id in all_links:
            if target_id in existing_targets:
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": source_id, "target": target_id})
            else:
                # Record broken link for later repair
                link_id = f"{source_id}\u2192{target_id}"
                now = datetime.now()
                self.conn.execute("""
                    MERGE (bl:BrokenLink {id: $id})
                    SET bl.source_id = $source, bl.target_id = $target, bl.timestamp = $ts
                """, {"id": link_id, "source": source_id, "target": target_id, "ts": now})

    def _insert_concept(
        self,
        concept: ConceptModel,
        body_text: str,
        concept_id_val: str,
    ) -> str:
        """Internal helper: upsert a single concept into the graph."""
        all_data = concept.model_dump()
        embedding_vec = all_data.pop("embedding", None)
        all_data.pop("body", None)
        all_data.pop("id", None)

        core = {
            "type": all_data.pop("type"),
            "title": all_data.pop("title", None),
            "description": all_data.pop("description", None),
            "resource": all_data.pop("resource", None),
            "tags": all_data.pop("tags", []),
            "timestamp": all_data.pop("timestamp", None),
        }

        extra = {
            k: json.dumps(v) if not isinstance(v, str) else v
            for k, v in all_data.items()
        }
        extra_keys = list(extra.keys())
        extra_values = list(extra.values())

        if isinstance(core["timestamp"], datetime):
            core["timestamp"] = core["timestamp"].isoformat()

        # Atomic upsert
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.execute(
                "MATCH (c:Concept {id: $id}) DETACH DELETE c",
                {"id": concept_id_val},
            )
            params: Dict[str, Any] = {
                "id": concept_id_val,
                "body": body_text,
                "embedding": embedding_vec,
                **core,
            }
            if extra_keys:
                params["extra_keys"] = extra_keys
                params["extra_values"] = extra_values
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding,
                        extra: MAP($extra_keys, $extra_values)
                    })
                """, params)
            else:
                self.conn.execute("""
                    CREATE (c:Concept {
                        id: $id, type: $type, title: $title,
                        description: $description, resource: $resource,
                        tags: $tags, timestamp: $timestamp,
                        body: $body, embedding: $embedding
                    })
                """, params)
            self._bump_write_epoch()  # concept changed -> indexes dirty
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        # Build Directory Hierarchy
        path_parts = concept_id_val.split("/")
        if len(path_parts) > 1:
            for i in range(1, len(path_parts)):
                parent = "/".join(path_parts[:i])
                child = "/".join(path_parts[: i + 1])
                if i == len(path_parts) - 1:
                    self.conn.execute("""
                        MERGE (d:Directory {id: $parent})
                        MERGE (c:Concept {id: $child})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, {"parent": parent, "child": child})
                else:
                    self.conn.execute("""
                        MERGE (p:Directory {id: $parent})
                        MERGE (d:Directory {id: $child})
                        MERGE (p)-[:CONTAINS]->(d)
                    """, {"parent": parent, "child": child})

        # Extract Markdown links
        link_pattern = re.compile(r"\[.*?\]\((.*?\.md)\)")
        for raw_link in link_pattern.findall(body_text):
            target_id = raw_link.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
            # Check if target exists
            result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            cnt = result.rows_as_dict().get_all()[0]["cnt"]
            if cnt > 0:
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": concept_id_val, "target": target_id})
            else:
                # Record broken link for later repair
                link_id = f"{concept_id_val}→{target_id}"
                now = datetime.now()
                self.conn.execute("""
                    MERGE (bl:BrokenLink {id: $id})
                    SET bl.source_id = $source, bl.target_id = $target, bl.timestamp = $ts
                """, {"id": link_id, "source": concept_id_val, "target": target_id, "ts": now})

        return concept_id_val

    # ------------------------------------------------------------------
    # Image ingestion (unified text / omni embedding space)
    # ------------------------------------------------------------------

    def _ingest_concept_images(
        self,
        concept_id: str,
        body: str,
        base_dir: Path,
        mode: "str | IngestMode",
    ) -> Dict[str, int]:
        """Extract, embed, and store the images referenced by a concept.

        Per-image routing follows ``mode``:
          * ``text``     — alt-text (or filename + image-number fallback), text model
          * ``optional`` — alt-text via text model; images without alt-text via omni
          * ``omni``     — every image via the omni model

        Unchanged images (same content hash) are skipped so the omni model is
        not re-run on re-import. Images removed from the document are pruned.
        Returns a small stats dict.
        """
        mode = IngestMode.coerce(mode)

        # Resolve relative image paths against the file's dir, then bundle root.
        search_dirs: List[Path] = []
        for d in (Path(base_dir), self.bundle_root):
            if d not in search_dirs:
                search_dirs.append(d)

        images = build_extracted_images(
            concept_id, body, search_dirs=search_dirs, allow_remote=self.allow_remote_images
        )

        stats = {"total": len(images), "text": 0, "omni": 0, "reused": 0, "pruned": 0}
        if not images and not self._concept_has_assets(concept_id):
            return stats

        existing = self._existing_asset_hashes(concept_id)  # {asset_id: content_hash}

        # --- Encode outside any DB transaction (omni can be slow) ---
        pending: List[Dict[str, Any]] = []
        planned_ids = set()
        for img in images:
            route, caption = plan_embedding(img, mode)
            payload = img.data if route is EmbedRoute.OMNI else (caption or "").encode("utf-8")
            content_hash = self._content_hash(route, payload)
            planned_ids.add(img.asset_id)

            if existing.get(img.asset_id) == content_hash:
                stats["reused"] += 1
                continue

            if route is EmbedRoute.OMNI:
                embedding = self._encode_image(img.data)
                stats["omni"] += 1
            else:
                embedding = self._encode(caption or img.filename, task="Document")
                stats["text"] += 1

            pending.append({
                "img": img,
                "route": route.value,
                "caption": caption or "",
                "content_hash": content_hash,
                "embedding": embedding,
            })

        stale_ids = [aid for aid in existing if aid not in planned_ids]
        stats["pruned"] = len(stale_ids)

        if not pending and not stale_ids:
            return stats

        # --- Write everything atomically ---
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for aid in stale_ids:
                self._delete_image_asset(concept_id, aid)
            for item in pending:
                self._upsert_image_asset(concept_id, item)
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

        return stats

    @staticmethod
    def _content_hash(route: EmbedRoute, payload: bytes) -> str:
        """Hash that changes whenever the embedding should be recomputed."""
        h = hashlib.sha256()
        h.update(route.value.encode("utf-8"))
        h.update(b"|")
        h.update(payload or b"")
        return h.hexdigest()

    def _concept_has_assets(self, concept_id: str) -> bool:
        result = self.conn.execute(
            """
            MATCH (c:Concept {id: $cid})-[:INCLUDES_ASSET]->(i:ImageAsset)
            RETURN count(i) AS cnt
            """,
            {"cid": concept_id},
        )
        rows = result.rows_as_dict().get_all()
        return bool(rows) and rows[0]["cnt"] > 0

    def _existing_asset_hashes(self, concept_id: str) -> Dict[str, str]:
        result = self.conn.execute(
            """
            MATCH (c:Concept {id: $cid})-[:INCLUDES_ASSET]->(i:ImageAsset)
            RETURN i.id AS id, i.content_hash AS content_hash
            """,
            {"cid": concept_id},
        )
        return {
            r["id"]: r["content_hash"]
            for r in result.rows_as_dict().get_all()
        }

    def _delete_image_asset(self, concept_id: str, asset_id: str) -> None:
        """Unlink an asset from this concept, and delete the node if now orphaned.

        The concept→asset edge is always removed. The ImageAsset node itself is
        only deleted when no other concept still references it — otherwise a
        shared asset id (e.g. an ``okf-asset://`` passthrough reused by several
        concepts) would be clobbered, or a plain DELETE would fail because the
        node still has edges.
        """
        self.conn.execute(
            """
            MATCH (c:Concept {id: $cid})-[r:INCLUDES_ASSET]->(i:ImageAsset {id: $iid})
            DELETE r
            """,
            {"cid": concept_id, "iid": asset_id},
        )
        self.conn.execute(
            """
            MATCH (i:ImageAsset {id: $iid})
            WHERE NOT EXISTS { MATCH (i)<-[:INCLUDES_ASSET]-(:Concept) }
            DETACH DELETE i
            """,
            {"iid": asset_id},
        )
        self._bump_write_epoch()  # image set changed -> image index dirty

    def _upsert_image_asset(self, concept_id: str, item: Dict[str, Any]) -> None:
        """Delete-then-create the ImageAsset, then (re)link it to the concept."""
        img = item["img"]
        # Clear any prior version (edge first, then node).
        self._delete_image_asset(concept_id, img.asset_id)
        self.conn.execute(
            """
            CREATE (i:ImageAsset {
                id: $id, file_name: $file_name, mime_type: $mime_type,
                alt_text: $alt_text, caption: $caption, embed_route: $embed_route,
                content_hash: $content_hash, data: $data, embedding: $embedding
            })
            """,
            {
                "id": img.asset_id,
                "file_name": img.filename,
                "mime_type": img.mime_type,
                "alt_text": img.alt_text or "",
                "caption": item["caption"],
                "embed_route": item["route"],
                "content_hash": item["content_hash"],
                "data": img.data if img.data is not None else b"",
                "embedding": item["embedding"],
            },
        )
        self.conn.execute(
            """
            MATCH (c:Concept {id: $cid}), (i:ImageAsset {id: $iid})
            MERGE (c)-[:INCLUDES_ASSET]->(i)
            """,
            {"cid": concept_id, "iid": img.asset_id},
        )
        self._bump_write_epoch()  # new/updated image -> image index dirty

    def list_images(self, concept_id: str) -> List[Dict[str, Any]]:
        """List the image assets attached to a concept (no BLOB payloads)."""
        result = self.conn.execute(
            """
            MATCH (c:Concept {id: $cid})-[:INCLUDES_ASSET]->(i:ImageAsset)
            RETURN i.id AS id, i.file_name AS file_name, i.mime_type AS mime_type,
                   i.alt_text AS alt_text, i.embed_route AS embed_route
            """,
            {"cid": concept_id},
        )
        return result.rows_as_dict().get_all()

    def get_image_data(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single image asset including its raw BLOB bytes."""
        result = self.conn.execute(
            """
            MATCH (i:ImageAsset {id: $iid})
            RETURN i.id AS id, i.file_name AS file_name, i.mime_type AS mime_type,
                   i.alt_text AS alt_text, i.embed_route AS embed_route, i.data AS data
            """,
            {"iid": asset_id},
        )
        rows = result.rows_as_dict().get_all()
        return rows[0] if rows else None

    def search_images_with_text(
        self,
        text_query: str,
        use_text_model: bool = True,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find image assets from a text query via the unified vector index.

        ``use_text_model=True`` (default) encodes the query with the lightweight
        text model — no omni load required, since both models share the vector
        space. Set it to ``False`` to route the query through the omni text side.
        """
        if use_text_model:
            query_vec = self._encode(text_query, task="Query")
        else:
            query_vec = self._encode_omni_text(text_query, task="Query")

        result = self.conn.execute(
            "CALL QUERY_VECTOR_INDEX('ImageAsset', 'image_omni_idx', $vec, $k) "
            "RETURN node, distance",
            {"vec": query_vec, "k": limit},
        )
        rows = result.rows_as_dict().get_all()
        out: List[Dict[str, Any]] = []
        for row in rows:
            node = row.get("node", {})
            if not isinstance(node, dict):
                continue
            out.append({
                "id": node.get("id"),
                "file_name": node.get("file_name"),
                "alt_text": node.get("alt_text"),
                "embed_route": node.get("embed_route"),
                "distance": row.get("distance"),
                "relevance_score": 1 - row.get("distance", 0),
            })
        return out

    def export_to_okf(self, concept_id: str, output_path: Path) -> None:
        """Export a concept back to an OKF .md file."""
        concept = self.get_by_id(concept_id)
        if not concept:
            raise FileNotFoundError(f"Concept {concept_id} not found")

        self._write_okf(concept, output_path)

    def _enrich_body_with_graph_links(
        self, concept_id: str, body: str
    ) -> str:
        """Enrich body with graph-derived links so the exported markdown
        faithfully reflects the LINKS_TO graph.

        Strategy (Option A — append, never replace):
          1. Query all outgoing LINKS_TO edges from this concept.
          2. For each target, check if a link to that target already exists
             in the body (by matching the target_id in link URLs).
          3. If not already linked, append a "See Also" bullet.
          4. Query all incoming LINKS_TO edges (concepts that link TO this one).
          5. If any exist, append a "Cited By" bullet list.

        This preserves the original body's links (which may have richer anchor
        text) while ensuring the graph structure is expressed in the export.
        """
        import re
        parts: List[str] = []

        # --- Outgoing links (See Also) ---
        result = self.conn.execute("""
            MATCH (s:Concept {id: $cid})-[:LINKS_TO]->(t:Concept)
            RETURN t.id AS target_id, t.title AS title, t.type AS type
            ORDER BY t.title
        """, {"cid": concept_id})
        outgoing_rows = result.rows_as_dict().get_all()

        if outgoing_rows:
            # Determine which targets are already linked in the body
            # by scanning for link URLs containing the target_id
            existing_link_targets = set()
            for row in outgoing_rows:
                target_id = row["target_id"]
                # Check if target_id appears in any link URL in the body
                link_pattern = re.compile(
                    r"\]\(([^)]*?" + re.escape(target_id) + r"[^)]*)\)"
                )
                if link_pattern.search(body):
                    existing_link_targets.add(target_id)

            # Collect targets that need a link added
            new_links = []
            for row in outgoing_rows:
                target_id = row["target_id"]
                if target_id not in existing_link_targets:
                    title = row["title"] or target_id.split("/")[-1]
                    new_links.append(f"- [{title}]({target_id}.md)")

            if new_links:
                parts.append("\n## See Also\n" + "\n".join(new_links))

        # --- Incoming links (Cited By) ---
        result = self.conn.execute("""
            MATCH (s:Concept)-[:LINKS_TO]->(t:Concept {id: $cid})
            RETURN s.id AS source_id, s.title AS title, s.type AS type
            ORDER BY s.title
        """, {"cid": concept_id})
        incoming_rows = result.rows_as_dict().get_all()

        if incoming_rows:
            cited_lines = ["\n## Cited By\n"]
            for row in incoming_rows:
                source_id = row["source_id"]
                title = row["title"] or source_id.split("/")[-1]
                cited_lines.append(f"- [{title}]({source_id}.md)")
            parts.append("\n".join(cited_lines))

        return body + "".join(parts)

    def _generate_index_files(
        self, output_dir: Path, concepts: Dict[str, ConceptModel]
    ) -> None:
        """Generate index.md files for every directory in the bundle.

        Each index.md lists the children (concepts and subdirectories) of that
        directory, enabling progressive disclosure for OKF consumers.
        """
        # Build a map of directory_id → list of (title, relative_path) children
        dir_children: Dict[str, List[Tuple[str, str]]] = {}

        for cid, concept in concepts.items():
            parts = cid.split("/")
            for i in range(1, len(parts)):
                dir_id = "/".join(parts[:i])
                dir_children.setdefault(dir_id, [])
                child_title = concept.title or parts[i]
                child_rel = cid.replace("/", os.sep) + ".md"
                dir_children[dir_id].append((child_title, child_rel))

        # Write index.md for each directory
        for dir_id, children in dir_children.items():
            # Sort children by title
            children.sort(key=lambda x: x[0])
            lines = [
                f"# {dir_id.split('/')[-1] or '(root)'}\n",
                "",
            ]
            for title, rel_path in children:
                lines.append(f"- [{title}]({rel_path})")
            lines.append("")

            # Create parent directories if needed
            dir_path = output_dir / dir_id.replace("/", os.sep)
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / "index.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_okf(self, concept: ConceptModel, output_path: Path) -> None:
        """Internal: serialize a ConceptModel to an OKF .md file.

        Enriches the body with LINKS_TO relationships from the graph so that
        exported markdown faithfully reflects the graph structure.
        """
        data = concept.model_dump()
        body = data.pop("body", "")
        data.pop("id", None)
        data.pop("embedding", None)

        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()

        yaml_str = yaml.dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        # ENRICH: add graph-derived links to the body
        body = self._enrich_body_with_graph_links(concept.id, body)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"---\n{yaml_str}---\n\n{body}", encoding="utf-8")

    def export_bundle(
        self,
        output_dir: Path,
        directory_id: Optional[str] = None,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """Export concepts from the graph back to an OKF bundle directory.

        Reconstructs the full directory hierarchy from CONTAINS relationships.
        Supports filtering by directory subtree, concept type, or tags.

        Args:
            output_dir: Root directory to write the bundle into.
            directory_id: If set, only export concepts under this directory.
            concept_type: If set, only export concepts of this type.
            tags: If set, only export concepts with ALL these tags.

        Returns:
            List of exported concept IDs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch all concepts (optionally filtered)
        concepts = self._fetch_concepts(
            concept_type=concept_type,
            tags=tags,
        )

        # If directory_id specified, filter to subtree
        if directory_id:
            concepts = {
                cid: c for cid, c in concepts.items()
                if self._is_under_directory(cid, directory_id)
            }

        if not concepts:
            return []

        # Export each concept, reconstructing path from its ID
        exported: List[str] = []
        for cid, concept in sorted(concepts.items()):
            # Concept IDs use forward slashes; convert to OS path separator
            rel_path = cid.replace("/", os.sep)
            file_path = output_dir / (rel_path + ".md")
            try:
                self._write_okf(concept, file_path)
                exported.append(cid)
            except Exception as e:
                print(f"  [WARN] Failed to export {cid}: {e}")

        # Generate index.md files for progressive disclosure
        self._generate_index_files(output_dir, concepts)

        return exported

    def _fetch_concepts(
        self,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, ConceptModel]:
        """Fetch all concepts, optionally filtered by type and tags."""
        where_clauses: list[str] = []
        params: Dict[str, Any] = {}

        if concept_type:
            where_clauses.append("c.type = $type")
            params["type"] = concept_type
        if tags:
            where_clauses.append("ALL(tag IN $tags WHERE tag IN c.tags)")
            params["tags"] = tags

        where_str = " AND ".join(where_clauses) if where_clauses else "true"
        query = f"""
        MATCH (c:Concept)
        WHERE {where_str}
        RETURN c.id, c.type, c.title, c.description, c.resource,
               c.tags, c.timestamp, c.body, c.embedding, c.extra
        """
        results = self.conn.execute(query, params)
        rows = results.rows_as_dict().get_all()

        concepts: Dict[str, ConceptModel] = {}
        for row in rows:
            data: Dict[str, Any] = {}
            for key, val in row.items():
                col = key.split(".", 1)[-1]  # strip 'c.' prefix
                if col != "extra":
                    data[col] = val

            # Decode extra MAP fields
            extra = row.get("c.extra") or {}
            for k, v in extra.items():
                if isinstance(v, str) and v.startswith(("{", "[")):
                    try:
                        data[k] = json.loads(v)
                    except json.JSONDecodeError:
                        data[k] = v
                else:
                    data[k] = v

            try:
                concepts[data["id"]] = ConceptModel.model_validate(data)
            except Exception:
                pass  # Skip malformed concepts

        return concepts

    def _is_under_directory(self, concept_id: str, directory_id: str) -> bool:
        """Check if a concept is under a given directory (via CONTAINS graph)."""
        result = self.conn.execute("""
            MATCH (d:Directory {id: $dir_id})-[:CONTAINS*1..5]->(c:Concept {id: $cid})
            RETURN count(c) AS cnt
        """, {"dir_id": directory_id, "cid": concept_id})
        rows = result.rows_as_dict().get_all()
        return rows[0]["cnt"] > 0 if rows else False

    # ------------------------------------------------------------------
    # Broken Links
    # ------------------------------------------------------------------

    def list_broken_links(self) -> List[Dict[str, Any]]:
        """List all tracked broken links (references to concepts not yet imported)."""
        result = self.conn.execute("""
            MATCH (bl:BrokenLink)
            RETURN bl.source_id AS source, bl.target_id AS target, bl.timestamp AS timestamp
            ORDER BY bl.timestamp
        """)
        rows = result.rows_as_dict().get_all()
        return [
            {"source": r["source"], "target": r["target"], "timestamp": r["timestamp"]}
            for r in rows
        ]

    def repair_links(self) -> int:
        """Attempt to repair broken links by re-checking if targets now exist.

        Scans all tracked broken links. For each one where the target concept
        now exists in the graph, creates the LINKS_TO relationship and removes
        the BrokenLink record.

        Returns:
            Number of links successfully repaired.
        """
        broken = self.list_broken_links()
        repaired = 0
        for link in broken:
            source_id = link["source"]
            target_id = link["target"]
            link_id = f"{source_id}→{target_id}"

            # Check if both source and target now exist
            source_result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": source_id},
            )
            target_result = self.conn.execute(
                "MATCH (c:Concept {id: $id}) RETURN count(c) AS cnt",
                {"id": target_id},
            )
            source_exists = source_result.rows_as_dict().get_all()[0]["cnt"] > 0
            target_exists = target_result.rows_as_dict().get_all()[0]["cnt"] > 0

            if source_exists and target_exists:
                # Create the LINKS_TO relationship
                self.conn.execute("""
                    MATCH (source:Concept {id: $source})
                    MATCH (target:Concept {id: $target})
                    MERGE (source)-[:LINKS_TO]->(target)
                """, {"source": source_id, "target": target_id})
                # Remove the BrokenLink record
                self.conn.execute(
                    "MATCH (bl:BrokenLink {id: $id}) DELETE bl",
                    {"id": link_id},
                )
                repaired += 1

        return repaired

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_chunks(
        self,
        query: str,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        limit: int = 10,
        max_chunks_per_doc: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search chunks using RRF-fused vector + FTS.

        Returns chunk-level results with optional parent concept metadata.
        Applies ``max_chunks_per_doc`` to limit how many chunks from the same
        document appear in results.
        """
        if not getattr(self, "_search_available", False):
            raise RuntimeError(
                "Search is unavailable: the 'vector'/'fts' extensions could not "
                "be loaded. Ensure the Ladybug extension repository is reachable, "
                "then reopen the router (ingestion and graph queries do not need "
                "these extensions)."
            )
        query_vec = self._encode(query, task="Query")

        # Stage 1: Vector search on chunks
        vec_results = self.conn.execute(
            "CALL QUERY_VECTOR_INDEX('Chunk', 'chunk_embedding', $vec, $k) RETURN node, distance",
            {"vec": query_vec, "k": limit * 3},
        )
        vec_rows = vec_results.rows_as_dict().get_all()
        vec_scores: Dict[str, float] = {}
        for row in vec_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                vec_scores[node_id] = 1 - row.get("distance", 0)

        # Stage 2: Full-text search on chunks
        fts_results = self.conn.execute(
            "CALL QUERY_FTS_INDEX('Chunk', 'chunk_fts', $query) RETURN node, score",
            {"query": query},
        )
        fts_rows = fts_results.rows_as_dict().get_all()
        fts_scores: Dict[str, float] = {}
        for row in fts_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                fts_scores[node_id] = row.get("score", 0)

        # Stage 3: RRF fusion (k=60)
        def _rank_map(scores: Dict[str, float]) -> Dict[str, int]:
            ordered = sorted(scores, key=scores.get, reverse=True)
            return {cid: i + 1 for i, cid in enumerate(ordered)}

        vec_rank = _rank_map(vec_scores)
        fts_rank = _rank_map(fts_scores)

        combined: List[Tuple[str, float]] = []
        for cid in set(vec_scores) | set(fts_scores):
            score = 0.0
            if cid in vec_rank:
                score += 1.0 / (60 + vec_rank[cid])
            if cid in fts_rank:
                score += 1.0 / (60 + fts_rank[cid])
            combined.append((cid, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Stage 4: Fetch chunk + parent metadata in bulk
        chunk_ids = [cid for cid, _ in combined]
        score_by_id = dict(combined)

        cypher = """
        MATCH (ch:Chunk)
        WHERE ch.id IN $ids
        OPTIONAL MATCH (parent:Concept)-[:PART_OF]->(ch)
        RETURN ch.id, ch.chunk_text, ch.block_type, ch.chunk_index,
               ch.parent_doc_id,
               parent.id AS parent_id, parent.title AS parent_title,
               parent.type AS parent_type, parent.tags AS parent_tags
        """
        rows = self.conn.execute(cypher, {"ids": chunk_ids}).rows_as_dict().get_all()
        meta_by_id = {row["ch.id"]: row for row in rows}

        # Assemble results, applying per-doc limit and graph filters
        results: List[Dict[str, Any]] = []
        doc_counts: Dict[str, int] = {}  # track chunks per doc

        for cid, _ in combined:
            row = meta_by_id.get(cid)
            if row is None:
                continue

            parent_id_val = row["ch.parent_doc_id"]

            # Apply graph filters on parent concept
            if concept_type:
                if row.get("parent_type") != concept_type:
                    continue
            if tags:
                parent_tags = row.get("parent_tags") or []
                if not any(t in parent_tags for t in tags):
                    continue
            if parent_id:
                if parent_id_val != parent_id:
                    continue

            # Apply per-doc limit
            if parent_id_val:
                doc_counts[parent_id_val] = doc_counts.get(parent_id_val, 0) + 1
                if doc_counts[parent_id_val] > max_chunks_per_doc:
                    continue

            chunk_text = row["ch.chunk_text"] or ""
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            result: Dict[str, Any] = {
                "chunk_id": cid,
                "chunk_text": chunk_text,
                "block_type": row["ch.block_type"],
                "chunk_index": row["ch.chunk_index"],
                "parent_doc_id": parent_id_val,
                "rrf_score": score_by_id[cid],
                "parent_title": row.get("parent_title"),
                "parent_type": row.get("parent_type"),
                "parent_tags": row.get("parent_tags"),
            }
            results.append(result)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Graph-Aware Retrieval
    # ------------------------------------------------------------------

    def _compute_hub_scores(self, concept_ids: List[str]) -> Dict[str, float]:
        """Count incoming LINKS_TO edges for each concept.

        Higher hub score = more concepts point to this one = more authoritative.
        """
        if not concept_ids:
            return {}

        result = self.conn.execute("""
            MATCH (x:Concept)-[:LINKS_TO]->(c:Concept)
            WHERE c.id IN $ids
            RETURN c.id AS id, count(x) AS cnt
        """, {"ids": concept_ids})
        return {r["id"]: r["cnt"] for r in result.rows_as_dict().get_all()}

    def _get_ancestry(self, concept_id: str, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Return directory path from root to this concept."""
        # Directory nodes only have an id (path), no title.
        result = self.conn.execute("""
            MATCH (d:Directory)-[:CONTAINS*1..5]->(c:Concept {id: $cid})
            RETURN d.id AS dir_id
            LIMIT 1
        """, {"cid": concept_id})
        rows = result.rows_as_dict().get_all()
        if not rows:
            return []
        return [{"id": r["dir_id"]} for r in rows]

    def _get_siblings(self, concept_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return other concepts in the same parent directory."""
        parent_result = self.conn.execute("""
            MATCH (d:Directory)-[:CONTAINS]->(c:Concept {id: $id})
            RETURN d.id AS parent_id
        """, {"id": concept_id})
        rows = parent_result.rows_as_dict().get_all()
        if not rows:
            # Concept not in a directory — find siblings by root-level concepts
            root_result = self.conn.execute("""
                MATCH (c:Concept)
                WHERE c.id <> $cid
                AND NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(c) }
                RETURN c.id AS id, c.title AS title, c.type AS type
                LIMIT $limit
            """, {"cid": concept_id, "limit": limit})
            return [
                {"id": r["id"], "title": r["title"], "type": r["type"]}
                for r in root_result.rows_as_dict().get_all()
            ]
        parent_id = rows[0]["parent_id"]

        result = self.conn.execute("""
            MATCH (d:Directory {id: $pid})-[:CONTAINS]->(s:Concept)
            WHERE s.id <> $cid
            RETURN s.id AS id, s.title AS title, s.type AS type
            LIMIT $limit
        """, {"pid": parent_id, "cid": concept_id, "limit": limit})
        return [
            {"id": r["id"], "title": r["title"], "type": r["type"]}
            for r in result.rows_as_dict().get_all()
        ]

    def search_with_context(
        self,
        query: str,
        limit: int = 5,
        context_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """Search chunks + expand each result with graph neighborhood context.

        Returns chunks enriched with:
          - incoming_links: concepts that link TO this document
          - outgoing_links: concepts this document links TO
          - ancestry: directory path from root
          - siblings: other concepts in the same parent directory
        """
        chunks = self.search_chunks(query, limit=limit * 2)

        enriched: List[Dict[str, Any]] = []
        for chunk in chunks[:limit]:
            parent_id = chunk["parent_doc_id"]

            incoming = self.traverse(parent_id, "LINKS_TO", "INCOMING",
                                      depth=context_hops)
            outgoing = self.traverse(parent_id, "LINKS_TO", "OUTGOING",
                                      depth=context_hops)
            ancestry = self._get_ancestry(parent_id)
            siblings = self._get_siblings(parent_id)

            enriched.append({
                "chunk": chunk,
                "document": self.get_by_id(parent_id),
                "incoming_links": incoming[:5],
                "outgoing_links": outgoing[:5],
                "ancestry": ancestry,
                "siblings": siblings[:5],
            })

        return enriched

    def search_chunks_with_hub_score(
        self,
        query: str,
        limit: int = 10,
        hub_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search chunks + rerank by graph hub score.

        Final score = (1 - hub_weight) * rrf_score + hub_weight * normalized_hub_score.
        """
        raw_results = self.search_chunks(query, limit=limit * 2)

        # Collect unique parent doc IDs
        parent_ids = list({r["parent_doc_id"] for r in raw_results if r["parent_doc_id"]})
        hub_scores = self._compute_hub_scores(parent_ids)
        max_hub = max(hub_scores.values()) if hub_scores else 1

        for r in raw_results:
            pid = r["parent_doc_id"]
            hub = hub_scores.get(pid, 0)
            normalized_hub = hub / max_hub if max_hub > 0 else 0
            r["hub_score"] = normalized_hub
            r["final_score"] = (
                (1 - hub_weight) * r["rrf_score"] + hub_weight * normalized_hub
            )

        raw_results.sort(key=lambda x: x["final_score"], reverse=True)
        return raw_results[:limit]

    def expand_with_graph_context(
        self,
        chunk_ids: List[str],
        hops: int = 1,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """Expand chunk search results with graph-context neighbours.

        For each seed chunk, find the parent document, then traverse
        LINKS_TO / CONTAINS to discover related concepts. Compute
        hub_score as the number of incoming LINKS_TO relationships.
        """
        if not chunk_ids:
            return []

        # Find parent documents for the seed chunks
        parents = self.conn.execute("""
            MATCH (p:Concept)-[:PART_OF]->(ch:Chunk)
            WHERE ch.id IN $ids
            RETURN p.id AS id, p.title AS title, p.type AS type, p.tags AS tags
        """, {"ids": chunk_ids})
        parent_rows = parents.rows_as_dict().get_all()
        parent_ids = [row["id"] for row in parent_rows]
        parent_meta = {row["id"]: row for row in parent_rows}

        if not parent_ids:
            return []

        # Expand via LINKS_TO (outgoing from parents)
        neighbours = self.conn.execute("""
            MATCH (p:Concept)-[:LINKS_TO]->(n:Concept)
            WHERE p.id IN $ids
            WITH DISTINCT n
            MATCH (other:Concept)-[:LINKS_TO]->(n)
            RETURN n.id AS id, n.title AS title, n.type AS type,
                   n.description AS description, n.tags AS tags,
                   count(other) AS hub_score
            ORDER BY hub_score DESC
            LIMIT $limit
        """, {"ids": parent_ids, "limit": max_results})
        neighbour_rows = neighbours.rows_as_dict().get_all()

        results: List[Dict[str, Any]] = []
        for row in neighbour_rows:
            results.append({
                "id": row["id"],
                "title": row["title"],
                "type": row["type"],
                "description": (row["description"] or "")[:200],
                "tags": row["tags"],
                "hub_score": row["hub_score"],
            })

        return results

    def rerank_with_hub_score(
        self,
        chunk_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank chunk results by combining RRF score with parent hub score.

        For each chunk result, look up the parent document's hub_score
        (incoming LINKS_TO count) and compute:
            final_score = rrf_score * (1 + 0.1 * hub_score)
        """
        if not chunk_results:
            return []

        parent_ids = list({r["parent_doc_id"] for r in chunk_results if r.get("parent_doc_id")})
        if not parent_ids:
            return chunk_results

        # Get hub scores for all parent documents
        hub_query = self.conn.execute("""
            MATCH (other:Concept)-[:LINKS_TO]->(p:Concept)
            WHERE p.id IN $ids
            RETURN p.id AS id, count(other) AS hub_score
        """, {"ids": parent_ids})
        hub_by_id = {row["id"]: row["hub_score"] for row in hub_query.rows_as_dict().get_all()}

        # Compute final scores
        for result in chunk_results:
            pid = result.get("parent_doc_id")
            hub = hub_by_id.get(pid, 0)
            rrf = result.get("rrf_score", 0)
            result["hub_score"] = hub
            result["final_score"] = rrf * (1 + 0.1 * hub)

        # Sort by final_score descending
        chunk_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return chunk_results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_hybrid(
        self,
        query: str,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        exclude_reserved: bool = True,
        limit: int = 10,
        include_chunks: bool = False,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: RRF fusion of vector + FTS with optional graph filters.

        If ``include_chunks=True``, each result also contains ``matched_chunks``
        — the top chunks from that document matching the query.
        """
        if not getattr(self, "_search_available", False):
            raise RuntimeError(
                "Search is unavailable: the 'vector'/'fts' extensions could not "
                "be loaded. Ensure the Ladybug extension repository is reachable, "
                "then reopen the router (ingestion and graph queries do not need "
                "these extensions)."
            )
        query_vec = self._encode(query, task="Query")

        # Stage 1: Vector search (ANN)
        vec_results = self.conn.execute(
            "CALL QUERY_VECTOR_INDEX('Concept', 'concept_embedding', $vec, $k) RETURN node, distance",
            {"vec": query_vec, "k": limit * 3},
        )
        vec_rows = vec_results.rows_as_dict().get_all()
        vec_scores: Dict[str, float] = {}
        for row in vec_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                vec_scores[node_id] = 1 - row.get("distance", 0)

        # Stage 2: Full-text search
        fts_results = self.conn.execute(
            "CALL QUERY_FTS_INDEX('Concept', 'concept_fts', $query) RETURN node, score",
            {"query": query},
        )
        fts_rows = fts_results.rows_as_dict().get_all()
        fts_scores: Dict[str, float] = {}
        for row in fts_rows:
            node = row.get("node", {})
            node_id = node.get("id") if isinstance(node, dict) else None
            if node_id:
                fts_scores[node_id] = row.get("score", 0)

        # Stage 3: Reciprocal Rank Fusion (RRF, k=60). Precompute each source's
        # rank map once (O(n log n)) instead of re-sorting inside the loop.
        def _rank_map(scores: Dict[str, float]) -> Dict[str, int]:
            ordered = sorted(scores, key=scores.get, reverse=True)
            return {cid: i + 1 for i, cid in enumerate(ordered)}

        vec_rank = _rank_map(vec_scores)
        fts_rank = _rank_map(fts_scores)

        combined: List[Tuple[str, float]] = []
        for cid in set(vec_scores) | set(fts_scores):
            score = 0.0
            if cid in vec_rank:
                score += 1.0 / (60 + vec_rank[cid])
            if cid in fts_rank:
                score += 1.0 / (60 + fts_rank[cid])
            combined.append((cid, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        # Over-fetch candidates so post-filtering can still fill `limit`.
        candidate_ids = [cid for cid, _ in combined]
        score_by_id = dict(combined)

        # Stage 4: fetch metadata (and apply graph filters) in a SINGLE query.
        where_clauses: List[str] = ["c.id IN $ids"]
        params: Dict[str, Any] = {"ids": candidate_ids}
        if concept_type:
            where_clauses.append("c.type = $type")
            params["type"] = concept_type
        if tags:
            where_clauses.append("ANY(tag IN $tags WHERE tag IN c.tags)")
            params["tags"] = tags
        if parent_id:
            where_clauses.append(
                "EXISTS { MATCH (p:Directory {id: $parent})-[:CONTAINS*1..3]->(c) }"
            )
            params["parent"] = parent_id
        if exclude_reserved:
            where_clauses.append(
                "NOT c.id ENDS WITH 'index' AND NOT c.id ENDS WITH 'log'"
            )

        cypher = f"""
        MATCH (c:Concept)
        WHERE {" AND ".join(where_clauses)}
        RETURN c.id, c.title, c.type, c.description, c.tags
        """
        rows = self.conn.execute(cypher, params).rows_as_dict().get_all()
        meta_by_id = {row["c.id"]: row for row in rows}

        # Assemble in RRF order, keeping only rows that survived filtering.
        results: List[Dict[str, Any]] = []
        for cid, _ in combined:
            row = meta_by_id.get(cid)
            if row is None:
                continue
            desc = row["c.description"] or ""
            if len(desc) > 200:
                desc = desc[:200] + "..."
            result = {
                "id": cid,
                "title": row["c.title"],
                "type": row["c.type"],
                "description": desc,
                "tags": row["c.tags"],
                "relevance_score": score_by_id[cid],
            }

            # Attach chunks if requested
            if include_chunks:
                chunks = self.search_chunks(
                    query=query, limit=3, parent_id=cid
                )
                result["matched_chunks"] = chunks

            results.append(result)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_length: int = 6,
    ) -> List[Dict[str, Any]]:
        """Find the shortest path between two concepts.

        Uses BFS-style variable-length patterns across allowed edge types.
        Returns a list of nodes on the path (including start and end) with
        their id, title, and type.
        """
        max_length = max(1, min(int(max_length), 10))
        # Ladybug doesn't support MATCH path = ... or [*1..N] (any-rel).
        # Try increasing path lengths until we find a connection.
        # Note: Ladybug reserves $end as a parameter name, so use $sid/$eid.
        for length in range(1, max_length + 1):
            result = self.conn.execute(
                f"""
                MATCH (a:Concept {{id: $sid}})-[:CONTAINS|LINKS_TO|PART_OF|INCLUDES_ASSET*1..{length}]-(b:Concept {{id: $eid}})
                RETURN a.id AS id, a.title AS title, a.type AS type
                """,
                {"sid": start_id, "eid": end_id},
            )
            rows = result.rows_as_dict().get_all()
            if rows:
                # Found a path at this length — collect all nodes
                # Re-run to get full path nodes using path variable
                path_result = self.conn.execute(
                    f"""
                    MATCH path = (a:Concept {{id: $sid}})-[:CONTAINS|LINKS_TO|PART_OF|INCLUDES_ASSET*1..{length}]-(b:Concept {{id: $eid}})
                    WITH path AS p, length(path) AS len
                    ORDER BY len ASC
                    LIMIT 1
                    UNWIND nodes(p) AS node
                    RETURN node.id AS id, node.title AS title, node.type AS type
                    """,
                    {"sid": start_id, "eid": end_id},
                )
                return path_result.rows_as_dict().get_all()
        return []

    def traverse(
        self,
        start_id: str,
        relationship: str = "CONTAINS",
        direction: str = "OUTGOING",
        depth: int = 1,
        node_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Navigate graph relationships with whitelisted edges and depth cap."""
        ALLOWED_RELS = {"CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"}
        if relationship not in ALLOWED_RELS:
            raise ValueError(f"Invalid relationship. Must be one of {ALLOWED_RELS}")
        depth = max(1, min(int(depth), 5))

        # PART_OF targets Chunk nodes; INCLUDES_ASSET targets ImageAsset
        if relationship == "PART_OF":
            target_label = "Chunk"
        elif relationship == "INCLUDES_ASSET":
            target_label = "ImageAsset"
        else:
            target_label = "Concept"

        if direction == "OUTGOING":
            pattern = f"-[{relationship}*1..{depth}]->(target:{target_label})"
        elif direction == "INCOMING":
            pattern = f"<-[{relationship}*1..{depth}]-(target:{target_label})"
        else:  # BOTH
            pattern = f"-[{relationship}*1..{depth}]-(target:{target_label})"

        where_clause = ""
        query_params: Dict[str, Any] = {"start_id": start_id}
        if node_type:
            where_clause = "WHERE target.type = $node_type"
            query_params["node_type"] = node_type

        cypher = f"""
        MATCH (start {{id: $start_id}}){pattern}
        {where_clause}
        RETURN target.*
        LIMIT 100
        """
        results = self.conn.execute(cypher, query_params)
        rows = results.rows_as_dict().get_all()

        results_list: List[Dict[str, Any]] = []
        for row in rows:
            entry: Dict[str, Any] = {}
            for key, val in row.items():
                prop = key.split(".", 1)[-1]  # strip 'target.' prefix
                entry[prop] = val
            results_list.append(entry)
        return results_list

    # ------------------------------------------------------------------
    # Chunk Query
    # ------------------------------------------------------------------

    def get_chunks(self, concept_id: str) -> List[ChunkModel]:
        """Get all chunks for a concept, ordered by chunk_index."""
        result = self.conn.execute("""
            MATCH (c:Concept {id: $id})-[:PART_OF]->(ch:Chunk)
            RETURN ch.id AS id, ch.parent_doc_id AS parent_doc_id,
                   ch.chunk_index AS chunk_index, ch.chunk_text AS chunk_text,
                   ch.block_type AS block_type,
                   ch.start_offset AS start_offset, ch.end_offset AS end_offset
            ORDER BY ch.chunk_index
        """, {"id": concept_id})
        rows = result.rows_as_dict().get_all()
        return [
            ChunkModel(
                id=row["id"],
                parent_doc_id=row["parent_doc_id"],
                chunk_index=row["chunk_index"],
                chunk_text=row["chunk_text"],
                block_type=row["block_type"],
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    def list_directory(self, directory_id: str) -> List[Dict[str, Any]]:
        """List immediate children of a directory (polymorphic: Directories + Concepts)."""
        results_directories: List[Dict[str, Any]] = []
        results_concepts: List[Dict[str, Any]] = []

        if not directory_id:
            # Root: find directories with no parent
            dir_result = self.conn.execute("""
                MATCH (d:Directory)
                WHERE NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(d) }
                RETURN d.id AS child_id, 'Directory' AS type, d.id AS title
            """)
            results_directories = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in dir_result.rows_as_dict().get_all()
            ]
            # Find concepts with no parent
            concept_result = self.conn.execute("""
                MATCH (c:Concept)
                WHERE NOT EXISTS { MATCH (:Directory)-[:CONTAINS]->(c) }
                RETURN c.id AS child_id, c.type AS type, c.title AS title
            """)
            results_concepts = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in concept_result.rows_as_dict().get_all()
            ]
        else:
            params = {"id": directory_id}
            dir_result = self.conn.execute("""
                MATCH (p:Directory {id: $id})-[:CONTAINS]->(d:Directory)
                RETURN d.id AS child_id, 'Directory' AS type, d.id AS title
            """, params)
            results_directories = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in dir_result.rows_as_dict().get_all()
            ]
            concept_result = self.conn.execute("""
                MATCH (p:Directory {id: $id})-[:CONTAINS]->(c:Concept)
                RETURN c.id AS child_id, c.type AS type, c.title AS title
            """, params)
            results_concepts = [
                {"id": r["child_id"], "type": r["type"], "title": r["title"]}
                for r in concept_result.rows_as_dict().get_all()
            ]

        return results_directories + results_concepts

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_by_id(self, concept_id: str) -> Optional[ConceptModel]:
        """Fetch a full concept by ID, merging extra MAP fields back into the model."""
        result = self.conn.execute(
            "MATCH (c:Concept {id: $id}) RETURN c.*", {"id": concept_id}
        )
        rows = result.rows_as_dict().get_all()
        if not rows:
            return None

        row = rows[0]
        data = {
            k.replace("c.", ""): v
            for k, v in row.items()
            if not k.startswith("c.extra")
        }
        extra = row.get("c.extra") or {}
        extra_decoded = {
            k: json.loads(v)
            if isinstance(v, str) and v.startswith(("{" , "["))
            else v
            for k, v in extra.items()
        }
        data.update(extra_decoded)
        data.pop("extra", None)
        return ConceptModel.model_validate(data)
