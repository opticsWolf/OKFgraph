"""EmbeddingEngine — ONNX inference, Matryoshka truncation, chunk splitting,
and document reconstruction.

Extracted from ``okfgraph.router.OKFRouter`` sections:
  - embedding (router.py 1276–1425, 1426–1471 classmethods)
  - chunking (router.py 1421–1558)
  - reconstruction (router.py 1558–1588)

The facade owns the ONNX ``embedder`` / ``tokenizer`` and passes them in.
The omni (multimodal) model is loaded lazily inside this engine.
"""

from typing import Any, List, Optional

import logging

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(
        self,
        embedder,
        tokenizer,
        embedding_dim: int,
        device: str,
        cache_dir: Optional[str],
        model_id: str,
        omni_model_id: str,
        chunk_size: int,
        chunk_overlap: int,
        enable_chunking: bool,
        conn,
    ):
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.device = device
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.omni_model_id = omni_model_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_chunking = enable_chunking
        self.conn = conn
        self._omni = None

    # --- embedding ---
    def _encode(self, text: str, task: str = "Document") -> List[float]: ...
    def _truncate_normalize(self, vec: List[float]) -> List[float]: ...
    def _encode_batch(self, texts: List[str], task: str = "Document") -> List[List[float]]: ...
    def _get_omni(self): ...
    def _encode_image(self, data: bytes) -> List[float]: ...
    def _encode_omni_text(self, text: str, task: str = "Query") -> List[float]: ...

    # --- classmethods (no instance state) ---
    @staticmethod
    def default_cache_dir() -> str: ...
    @classmethod
    def model_info(cls, model_id: str = "jinaai/jina-embeddings-v5-text-small-retrieval") -> dict: ...

    # --- chunking ---
    def _split_into_chunks(self, text: str): ...
    def _compute_overlap_payloads(self, chunks): ...

    # --- reconstruction ---
    def reconstruct_document(self, document_id: str) -> str: ...
