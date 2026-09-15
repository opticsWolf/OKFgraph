"""Vector encoding, chunking, and document reconstruction extracted during the OKFRouter Phase 1 refactor.

Bodies are verbatim from okfgraph/router.py; the facade (OKFRouter) owns
the shared resources (conn, embedder, tokenizer, ...) and injects them
here. Public callers reach these via router.<method> (component bridge).
"""
import logging
import math
import re
import threading
from pathlib import Path

import mordant
from typing import Any, Dict, List, Optional

#: Suffix marking post-split continuation pieces (0.5.1 chunk cap).
#: A mordant block larger than ``chunk_size`` words is subdivided into
#: exact-tiling pieces; the first keeps the block type, continuations
#: append "+" so reconstruction joins them with "" instead of the
#: inter-block delimiter. Stored as-is (plain STRING column, no schema
#: change); consumers compare against the base type via _base_block_type.
CONTINUATION_SUFFIX = "+"


def _base_block_type(block_type: str) -> str:
    """Strip the continuation suffix, if present."""
    if block_type.endswith(CONTINUATION_SUFFIX):
        return block_type[: -len(CONTINUATION_SUFFIX)]
    return block_type


def _cap_chunk_budget(
    chunk: Dict[str, Any],
    max_units: int,
    counter,  # Callable[[str], int]: tokens in production, words as fallback
) -> List[Dict[str, Any]]:
    """Subdivide an oversized chunk into budget-capped pieces (pure).

    mordant splits purely by block structure with no size limit, so a
    single giant block (code fence, table, wall of prose) becomes one
    giant chunk. The budget is measured with the real tokenizer
    (production) or word counts (``counter=None`` fallback, cold/tests).
    Cut points fall on word starts — found by binary search over word
    spans, so only oversized chunks pay, logarithmically — and pieces
    tile the parent text *exactly*: ``"".join(piece texts) == parent
    text`` and the byte offsets tile the parent span with zero gaps,
    which is what lets :meth:`reconstruct_document` rejoin
    continuations losslessly. The first piece keeps the block type (and
    start offset); each continuation appends :data:`CONTINUATION_SUFFIX`.
    Chunks at or under budget pass through untouched. A single word over
    budget (e.g. a base64 blob) is unsplittable at word granularity and
    is emitted alone — the context-window warning remains the backstop.
    """
    text = chunk["chunk_text"]
    limit = max(1, max_units)
    if counter(text) <= limit:
        return [chunk]
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    if not spans:
        return [chunk]
    # Word indices where each piece begins; piece j covers words
    # [starts[j], starts[j+1]). Binary-search the widest fitting end.
    starts = [0]
    start = 0
    n = len(spans)
    while start < n:
        if counter(text[spans[start][0]:spans[start][1]]) > limit:
            end = start + 1  # unsplittable single word, emitted alone
        else:
            lo, hi = start + 1, n
            base = 0 if start == 0 else spans[start][0]
            while lo < hi:
                mid = (lo + hi + 1) // 2
                # Validate the EXACT emitted span, trailing whitespace
                # included (a "\n\n" tail can cost a token of its own).
                cand_end = spans[mid][0] if mid < n else len(text)
                if counter(text[base:cand_end]) <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            end = lo
        starts.append(end)
        start = end
    # Char boundaries at word starts; leading/inter-word whitespace
    # accrues to the preceding piece. Exact tiling by construction.
    bounds = [0] + [spans[i][0] for i in starts[1:-1]] + [len(text)]
    base_start = chunk["start_offset"]
    pieces = []
    for i in range(len(bounds) - 1):
        piece_text = text[bounds[i]:bounds[i + 1]]
        pieces.append({
            **chunk,
            "chunk_text": piece_text,
            "block_type": chunk["block_type"]
            if i == 0 else chunk["block_type"] + CONTINUATION_SUFFIX,
            "start_offset": base_start + len(text[:bounds[i]].encode("utf-8")),
            "end_offset": base_start + len(text[:bounds[i + 1]].encode("utf-8")),
        })
    return pieces


#: Meta key pinning the weight precision a graph was imported with.
#: Stored as INT64 (16/32) — the Meta value column is integer-typed.
PRECISION_META_KEY = "embedding_precision"
_PRECISION_CODE = {"fp16": 16, "fp32": 32}
#: Meta key pinning the text model a graph was imported with. Model ids
#: are strings, so this lives in its own MetaText table (Meta.value is
#: INT64). Same fail-closed adoption semantics as the precision pin:
#: first open records, later opens refuse on mismatch, empty graphs re-pin.
MODEL_META_KEY = "embedding_model"


def enforce_precision_pin(conn, precision: str) -> str:
    """Fail-closed precision pin for a graph (0.5.0).

    FP16 and FP32 vectors share the dimension but live in different
    spaces — delta hashes would never catch a precision switch, so the
    first session open records the precision in Meta and every later
    open refuses on mismatch. Empty graphs re-pin silently (adoption,
    no migration); explicit local files always report ``fp32``.

    Returns the pinned precision. Raises RuntimeError on mismatch.
    """
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS Meta (key STRING PRIMARY KEY, value INT64)"
    )
    try:
        rows = conn.execute(
            f"MATCH (m:Meta {{key: '{PRECISION_META_KEY}'}}) RETURN m.value AS v"
        ).rows_as_dict().get_all()
    except Exception:
        rows = []
    if rows:
        pinned = {16: "fp16", 32: "fp32"}.get(rows[0]["v"], None)
        if pinned is None or pinned == precision:
            return precision if pinned is None else pinned
        # Mismatch — but an empty graph carries no vectors, so re-pin.
        try:
            n = conn.execute(
                "MATCH (c:Concept) RETURN c.id AS cid LIMIT 1"
            ).rows_as_dict().get_all()
        except Exception:
            n = []
        if not n:
            conn.execute(
                f"MERGE (m:Meta {{key: '{PRECISION_META_KEY}'}}) "
                f"SET m.value = {_PRECISION_CODE[precision]}"
            )
            logger.info(
                "empty graph re-pinned to precision=%s", precision)
            return precision
        raise RuntimeError(
            f"graph is pinned to precision={pinned} but the session opened "
            f"with precision={precision}: FP16 and FP32 vectors share the "
            f"dimension but live in different spaces and must never mix. "
            f"Reimport into a fresh database with precision={precision}, or "
            f"reopen with precision={pinned}."
        )
    conn.execute(
        f"MERGE (m:Meta {{key: '{PRECISION_META_KEY}'}}) "
        f"SET m.value = {_PRECISION_CODE[precision]}"
    )
    logger.debug("graph pinned to precision=%s", precision)
    return precision


def enforce_model_pin(conn, model_id: str) -> str:
    """Fail-closed model pin for a graph (0.6.0).

    Different weights = different vector space: a model switch is
    undetectable to delta hashes and silently corrupts retrieval, so the
    first session open records the model id in MetaText and every later
    open refuses on mismatch. Empty graphs re-pin silently (adoption,
    no migration) — the same contract as `enforce_precision_pin`.

    Returns the pinned model id. Raises RuntimeError on mismatch.
    """
    conn.execute(
        "CREATE NODE TABLE IF NOT EXISTS MetaText (key STRING PRIMARY KEY, value STRING)"
    )
    try:
        rows = conn.execute(
            f"MATCH (m:MetaText {{key: '{MODEL_META_KEY}'}}) RETURN m.value AS v"
        ).rows_as_dict().get_all()
    except Exception:
        rows = []
    if rows:
        pinned = rows[0]["v"]
        if pinned == model_id:
            return pinned
        try:
            n = conn.execute(
                "MATCH (c:Concept) RETURN c.id AS cid LIMIT 1"
            ).rows_as_dict().get_all()
        except Exception:
            n = []
        if not n:
            conn.execute(
                f"MERGE (m:MetaText {{key: '{MODEL_META_KEY}'}}) "
                f"SET m.value = '{model_id}'"
            )
            logger.info("empty graph re-pinned to model=%s", model_id)
            return model_id
        raise RuntimeError(
            f"graph is pinned to model={pinned} but the session opened "
            f"with model={model_id}: different weights live in different "
            f"vector spaces and must never mix. Reimport into a fresh "
            f"database with model={model_id}, or reopen with model={pinned}."
        )
    conn.execute(
        f"MERGE (m:MetaText {{key: '{MODEL_META_KEY}'}}) "
        f"SET m.value = '{model_id}'"
    )
    logger.debug("graph pinned to model=%s", model_id)
    return model_id
logger = logging.getLogger(__name__)

_ORT_MODULE_NAMES = ("onnxruntime", "onnxruntime-gpu")


def _candidate_ort_library_names(os_name=None, sys_platform=None):
    """Return the ORT library filenames for an OS/platform pair.

    Takes explicit arguments so unit tests can cover Windows/macOS/Linux
    from any host. Linux uses a versioned ``libonnxruntime.so.*`` glob
    (handled by :func:`_find_runtime_in_package`); this helper returns the
    unversioned fallback name for that platform.
    """
    import os
    import sys
    if os_name is None:
        os_name = os.name
    if sys_platform is None:
        sys_platform = sys.platform
    if os_name == "nt":
        return ("onnxruntime.dll",)
    if sys_platform == "darwin":
        return ("libonnxruntime.dylib",)
    return ("libonnxruntime.so",)


def _find_runtime_in_package(package_dir, os_name=None, sys_platform=None):
    """Find an ORT shared library inside a pip-installed package directory.

    Returns a :class:`Path` or ``None``. Prefers versioned Linux libraries
    (``libonnxruntime.so.*``) over the unversioned ``libonnxruntime.so``.
    """
    import os
    import sys
    if os_name is None:
        os_name = os.name
    if sys_platform is None:
        sys_platform = sys.platform
    capi_dir = Path(package_dir) / "capi"
    if not capi_dir.is_dir():
        return None
    if os_name == "nt" or sys_platform == "darwin":
        candidate = capi_dir / _candidate_ort_library_names(os_name, sys_platform)[0]
        return candidate if candidate.is_file() else None
    versioned = sorted(capi_dir.glob("libonnxruntime.so.*"))
    if versioned:
        return versioned[-1]
    candidate = capi_dir / "libonnxruntime.so"
    return candidate if candidate.is_file() else None


def _configure_windows_ort_dll_directory(runtime_path) -> bool:
    """Add the ORT ``capi`` directory to Windows DLL resolution.

    Best-effort only: returns ``False`` (never raises) when unavailable.
    """
    import os
    if os.name != "nt":
        return False
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if not callable(add_dll_directory):
        return False
    try:
        add_dll_directory(str(Path(runtime_path).parent))
        return True
    except OSError as exc:
        logger.debug("ORT DLL directory configuration failed: %s", exc)
        return False


def _warm_ort_gpu_dlls(module) -> bool:
    """Preload NVIDIA DLLs when the installed ORT package exposes CUDA.

    This mirrors bobine's GPU bootstrap: only attempt the preload when the
    package reports a CUDA execution provider, and never let it fail
    runtime discovery.
    """
    preload = getattr(module, "preload_dlls", None)
    if not callable(preload):
        return False
    try:
        providers = list(module.get_available_providers())
    except Exception as exc:
        logger.debug("ORT provider query failed: %s", exc)
        return False
    if "CUDAExecutionProvider" not in providers:
        return False
    try:
        preload()
        return True
    except Exception as exc:
        logger.debug("ORT GPU DLL preload failed: %s", exc)
        return False


def resolve_ort_dylib(*, warm_gpu: bool = True, os_name=None, sys_platform=None) -> Optional[str]:
    """Point ``ORT_DYLIB_PATH`` at a pip-installed ORT build when unset.

    Both bobine and embroider load ONNX Runtime dynamically; sharing one
    binary avoids version/CUDA drift between the two runtimes. Explicit
    user configuration always wins — this only fills the gap.

    Resolution order:

    1. Existing ``ORT_DYLIB_PATH``.
    2. Pip-installed ``onnxruntime`` or ``onnxruntime-gpu`` package.
    3. OS loader path (represented by returning ``None``).

    Missing runtimes never raise here; session creation or encoding is the
    fail-fast boundary. ``os_name``/``sys_platform`` are explicit so unit
    tests can cover every platform from any host (patching ``os.name``
    would break ``pathlib`` dispatch on Windows).
    """
    import os
    explicit = os.environ.get("ORT_DYLIB_PATH")
    if explicit:
        return explicit
    for module_name in _ORT_MODULE_NAMES:
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        try:
            package_dir = Path(str(module.__file__)).parent
        except Exception:
            continue
        found = _find_runtime_in_package(package_dir, os_name, sys_platform)
        if found is None:
            continue
        os.environ["ORT_DYLIB_PATH"] = str(found)
        _configure_windows_ort_dll_directory(found)
        if warm_gpu:
            _warm_ort_gpu_dlls(module)
        logger.debug("resolved ORT runtime: %s", found)
        return str(found)
    return None


class LazyRustEncoder:
    """Defers the ONNX session open until the first real encode.

    Router construction stays cheap: the ``embroider`` wheel import is still
    validated eagerly (fail fast on a missing install), but ``JinaV5.open``
    — model download + session build — waits for the first ``encode`` /
    ``encode_batch`` / ``used_cuda`` access. Token counting uses the
    separate lightweight ``JinaTokenizer`` handle, so budgeted reads and the
    context-window guard stay cold too.

    A failed session open is cached and re-raised: configuration errors stay
    fail-fast (once, at first encode) instead of retrying network/model
    acquisition on every call. Thread-safe: concurrent first encodes open
    exactly one session.
    """

    def __init__(self, *, model_id, truncate_dim, device,
                 session_factory, tokenizer_factory, on_open=None):
        self._model_id = model_id
        self._truncate_dim = truncate_dim
        self._device = device
        self._session_factory = session_factory
        self._tokenizer_factory = tokenizer_factory
        self._on_open = on_open
        self._lock = threading.Lock()
        self._encoder = None
        self._encoder_error = None
        self._open_reported = False
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """True once the ONNX session has been opened."""
        return self._encoder is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._truncate_dim

    @property
    def used_cuda(self) -> bool:
        """Effective device — opens the session on first access."""
        return bool(self._get_encoder().used_cuda)

    def encode(self, text: str, task: str = "Document"):
        return self._get_encoder().encode(text, task=task)

    def encode_batch(self, texts, task: str = "Document"):
        return self._get_encoder().encode_batch(texts, task=task)

    def count_tokens(self, text: str) -> int:
        """Exact count via the tokenizer-only handle (never opens the session)."""
        return int(self._get_tokenizer().count_tokens(text))

    def _get_encoder(self):
        encoder = self._encoder
        if encoder is not None:
            return encoder
        with self._lock:
            if self._encoder is not None:
                return self._encoder
            if self._encoder_error is not None:
                raise self._encoder_error
            try:
                encoder = self._session_factory()
            except Exception as exc:
                self._encoder_error = exc
                raise
            self._encoder = encoder
            if self._on_open is not None and not self._open_reported:
                self._open_reported = True
                self._on_open(encoder)
            return encoder

    def _get_tokenizer(self):
        tokenizer = self._tokenizer
        if tokenizer is not None:
            return tokenizer
        with self._lock:
            if self._tokenizer is not None:
                return self._tokenizer
            try:
                tokenizer = self._tokenizer_factory()
            except Exception as exc:
                logger.debug("tokenizer-only open failed: %s", exc)
                raise
            self._tokenizer = tokenizer
            return tokenizer

    def __repr__(self) -> str:
        state = "loaded" if self._encoder is not None else "cold"
        return (
            f"LazyRustEncoder({self._model_id}, "
            f"dim={self._truncate_dim}, {state})"
        )


class EmbeddingEngine:
    """Owns the embedding model and chunking logic.

    Text embeddings come from the Rust embroider wheel (Jina v5 via ORT):
    prefixed, last-token pooled, truncated. There is no Python fallback —
    a mid-run stack switch would silently mix vector spaces in one index.
    """

    def __init__(self, rust_encoder, embedding_dim, device,
                 cache_dir, model_id, omni_model_id, omni,
                 chunk_size, chunk_overlap, enable_chunking, conn,
                 ort_dylib=None):
        self.encoder = rust_encoder
        self.embedding_dim = embedding_dim
        self.device = device
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.omni_model_id = omni_model_id
        self._omni = omni
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_chunking = enable_chunking
        self.conn = conn
        self.ort_dylib = ort_dylib

    def _encode(self, text: str, task: str = "Document") -> List[float]:
        """Encode text with the Rust Jina v5 encoder.

        Returns an L2-normalised vector, truncated to the Matryoshka dim.
        Last-token pooling is REQUIRED (mean pooling lands in a different
        space that will NOT align with the omni image embeddings).
        """
        return self.encoder.encode(text, task=task)

    def count_tokens(self, text: str) -> int:
        """Count tokens with the Rust encoder, falling back to chars/4.

        Used for token-budgeted reads and the context-window guard. Never
        raises: without a tokenizer a rough estimate beats no answer.
        """
        try:
            return int(self.encoder.count_tokens(text))
        except Exception:
            return max(1, len(text) // 4)


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
        """Encode multiple texts via one Rust call (sequential inside,
        avoiding padded-batch attention waste on variable-length docs).

        Args:
            texts: List of raw texts to encode.
            task: ``"Query"`` or ``"Document"`` — controls the prefix.

        Returns:
            List of L2-normalised embedding vectors (each truncated to target dim).
        """
        if not texts:
            return []
        # Prefix guard lives in Rust; one boundary crossing for the batch.
        return self.encoder.encode_batch(texts, task=task)


    def _get_omni(self):
        """Load the omni model on first use (vision + text towers only)."""
        if self._omni is None:
            from sentence_transformers import SentenceTransformer

            # The torch path takes torch device strings — "auto" (the
            # 0.5.0 router default) is not one. Precision selection does
            # not apply here: omni always runs FP32 on this path.
            device = self.device
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            logger.info(
                "Loading omni model %s (vision modality) on %s ...",
                self.omni_model_id, device,
            )
            self._omni = SentenceTransformer(
                self.omni_model_id,
                trust_remote_code=True,
                cache_folder=self.cache_dir,
                device=device,
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
            # Continuation pieces ("Type+") inherit the parent's
            # structural behaviour: code stays tail-free, prose chains.
            base_type = _base_block_type(chunk["block_type"])

            # 1. Enforce hard semantic boundary
            # Clear any trailing words from the previous section when hitting a structural block
            if base_type in STRUCTURAL_BLOCKS:
                prev_tail = ""

            # 2. Apply sliding word boundary window if a tail exists.
            # Bound (0.5.1): never prepend more context than the receiving
            # chunk's own content. A fixed 40-word tail drowns small chunks
            # (tail/pure up to 3950% wild); A/B measured small-chunk
            # self-hit@1 0.843 -> 0.887 and neighbor-steals 30 -> 10.
            if prev_tail:
                keep = max(1, len(chunk["chunk_text"].split()))
                prev_tail = "  ".join(prev_tail.split()[-keep:])
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
            if self.chunk_overlap > 0 and base_type not in STRUCTURAL_BLOCKS:
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
            prev_bt = rows[i - 1]["block_type"]
            cur_bt = rows[i]["block_type"]
            # Continuation pieces tile the parent span exactly (zero-gap
            # offsets); rejoin with "" instead of the inter-block
            # delimiter. Base-normalised fallback keeps pre-cap graphs
            # (no "+" types) on the identical path as before.
            if (cur_bt.endswith(CONTINUATION_SUFFIX)
                    and _base_block_type(cur_bt) == _base_block_type(prev_bt)):
                sep = ""
            else:
                sep = mordant.MarkdownChunker.get_delimiter(
                    _base_block_type(prev_bt), _base_block_type(cur_bt)
                )
            parts.append(sep + rows[i]["chunk_text"])

        return "".join(parts)


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
        self, body: str, document_id: str, count_tokens=None,
    ) -> List[Dict[str, Any]]:
        """Split document body into pure blocks using mordant chunker.

        Uses chunker.get_all_chunks() to get ExtractedChunk objects with
        block_type and byte offsets. Includes headings as separate chunks
        so they are preserved during reconstruction. No overlap is stored.

        Post-split cap (0.5.1): mordant has no size limit, so any block
        over ``chunk_size`` *tokens* (measured with ``count_tokens``,
        the production tokenizer counter) is subdivided by
        :func:`_cap_chunk_budget` into exact-tiling continuation pieces
        ("Type+"). ``count_tokens=None`` falls back to word counts
        (cold paths / tests — no tokenizer I/O). ``chunk_index`` is
        renumbered over the final list so chunk ids stay dense.
        """
        chunker = mordant.MarkdownChunker(body)
        chunks: List[Dict[str, Any]] = []

        current_heading = ""
        for chunk in chunker.get_all_chunks():
            # Track the heading context as we move down the document
            if chunk.block_type == "Heading":
                current_heading = chunk.text

            base = {
                "parent_doc_id": document_id,
                "chunk_text": chunk.text,
                "block_type": chunk.block_type,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "chunk_index": -1,  # renumbered below, post-split
                # Ephemeral context used strictly for constructing the embedding payload
                "heading_context": current_heading if chunk.block_type != "Heading" else ""
            }
            # Token-measured cap in production; word-count fallback keeps
            # cold paths (tests, tooling without a session) tokenizer-free.
            counter = count_tokens or (lambda t: len(t.split()))
            chunks.extend(_cap_chunk_budget(base, self.chunk_size, counter))

        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index
        return chunks


