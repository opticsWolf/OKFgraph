"""Vector encoding, chunking, and document reconstruction extracted during the OKFRouter Phase 1 refactor.

Bodies are verbatim from okfgraph/router.py; the facade (OKFRouter) owns
the shared resources (conn, embedder, tokenizer, ...) and injects them
here. Public callers reach these via router.<method> (component bridge).
"""
import logging
import math
from pathlib import Path

import mordant
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)

def resolve_ort_dylib() -> Optional[str]:
    """Point ``ORT_DYLIB_PATH`` at the pip-installed ORT build when unset.

    Both bobine and okf-embed load ONNX Runtime dynamically; sharing one
    binary avoids version/CUDA drift between the two runtimes. Explicit
    user configuration always wins — this only fills the gap.
    """
    import os
    if os.environ.get("ORT_DYLIB_PATH"):
        return os.environ["ORT_DYLIB_PATH"]
    try:
        import onnxruntime
        dll = Path(str(onnxruntime.__file__)).parent / "capi" / "onnxruntime.dll"
        if dll.exists():
            os.environ["ORT_DYLIB_PATH"] = str(dll)
            return str(dll)
    except ImportError:
        pass
    return None


class EmbeddingEngine:
    """Owns the embedding model and chunking logic.

    Text embeddings come from the Rust okf_embed wheel (Jina v5 via ORT):
    prefixed, last-token pooled, truncated. There is no Python fallback —
    a mid-run stack switch would silently mix vector spaces in one index.
    """

    def __init__(self, rust_encoder, embedding_dim, device,
                 cache_dir, model_id, omni_model_id, omni,
                 chunk_size, chunk_overlap, enable_chunking, conn):
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


