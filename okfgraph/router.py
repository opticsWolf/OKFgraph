"""OKFRouter — Ladybug-backed knowledge graph with ONNX + Jina v5 embeddings."""

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
import torch
import yaml
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from okfgraph.images import (
    EmbedRoute,
    IngestMode,
    build_extracted_images,
    plan_embedding,
)
from okfgraph.models import ConceptModel

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

        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = None  # Let onnxruntime use its defaults

        try:
            if providers:
                self.embedder = ORTModelForFeatureExtraction.from_pretrained(
                    model_id,
                    export=False,
                    subfolder="onnx",
                    cache_dir=cache_dir,
                    provider=providers,
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
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create schema, extensions, and indexes if they don't exist."""
        for ext in ("vector", "fts"):
            self.conn.execute(f"INSTALL {ext};")
            self.conn.execute(f"LOAD {ext};")

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

        # BrokenLink table — tracks links to concepts not yet imported
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS BrokenLink (
                id STRING PRIMARY KEY,
                source_id STRING,
                target_id STRING,
                timestamp TIMESTAMP
            )
        """)

        # Vector index (ignore if already exists)
        try:
            self.conn.execute("""
                CALL CREATE_VECTOR_INDEX(
                    'Concept', 'concept_embedding', 'embedding',
                    mu := 30, ml := 60, metric := 'cosine', efc := 200
                )
            """)
        except Exception:
            pass  # index already exists

        # FTS index (ignore if already exists)
        try:
            self.conn.execute("""
                CALL CREATE_FTS_INDEX('Concept', 'concept_fts', ['title', 'description', 'body'])
            """)
        except Exception:
            pass  # index already exists

        # Unified image vector index (ignore if already exists)
        try:
            self.conn.execute("""
                CALL CREATE_VECTOR_INDEX(
                    'ImageAsset', 'image_omni_idx', 'embedding',
                    mu := 30, ml := 60, metric := 'cosine', efc := 200
                )
            """)
        except Exception:
            pass  # index already exists

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _encode(self, text: str, task: str = "Document") -> List[float]:
        """Encode text with ONNX Jina v5 model.

        Args:
            text: Raw text to encode.
            task: ``"Query"`` or ``"Document"`` — controls the prefix.

        Returns:
            L2-normalised embedding vector (list of floats).
        """
        # Apply prefix (avoid double-prefixing)
        if not text.startswith(("Query:", "Document:")):
            text = f"{task}: {text}"

        # Tokenise
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            padding=True,
        )

        # ONNX forward pass
        with torch.no_grad():
            outputs = self.embedder(**inputs)

        # Last-token pooling (REQUIRED by jina-embeddings-v5; mean pooling
        # produces vectors in a different space that will NOT align with the
        # omni model's image embeddings in the unified ImageAsset index).
        last_hidden = outputs.last_hidden_state           # (B, T, H)
        attention_mask = inputs["attention_mask"]         # (B, T)
        last_idx = attention_mask.sum(dim=1) - 1          # index of final real token
        last_idx = last_idx.clamp(min=0).to(torch.long)
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_idx]  # (B, H)

        # L2 normalisation
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

        # Matryoshka truncation (+ re-normalisation to keep unit norm)
        vec = normalized.squeeze(0).tolist()
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

        # Batch encode — call _encode sequentially to avoid padding overhead.
        # With variable-length texts (80-300 words), padding all to the longest
        # in the batch causes massive attention compute waste (O(batch*max_len^2)).
        # Sequential single-pass encoding is faster because each text only
        # processes its actual token count.
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

    def import_from_okf(
        self,
        file_path: Path,
        mode: "str | IngestMode" = IngestMode.TEXT,
    ) -> str:
        """Parse an OKF .md file and create/update the concept in the graph.

        Args:
            file_path: Path to the ``.md`` file.
            mode: Image ingestion mode — ``text`` (alt-text / filename fallback,
                no omni model), ``optional`` (omni only for images without
                alt-text), or ``omni`` (omni for every image).

        Returns the concept ID (relative path without .md extension).
        """
        mode = IngestMode.coerce(mode)

        # 1. Parse frontmatter + body
        post = frontmatter.load(file_path)
        body = post.content
        frontmatter_data = dict(post.metadata)

        rel_path = file_path.relative_to(self.bundle_root)
        concept_id = str(rel_path).replace("\\", "/").replace(".md", "")

        # 2. Build Concept model
        data = {**frontmatter_data, "id": concept_id, "body": body}
        concept = ConceptModel.model_validate(data)

        # 3. Generate embedding (Document prefix)
        search_text = f"{concept.title or ''} {concept.description or ''} {concept.body or ''}"
        concept.embedding = self._encode(search_text, task="Document")

        # 4. Insert into graph (delegates to shared upsert logic)
        self._insert_concept(concept, body, concept_id)

        # 5. Ingest any embedded images under the requested mode
        self._ingest_concept_images(concept_id, body, file_path.parent, mode)

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
        md_files = sorted(root.rglob("*.md"))
        if not md_files:
            return []

        # Phase 1: Parse all files
        parsed: List[Dict[str, Any]] = []
        for fp in md_files:
            try:
                post = frontmatter.load(fp)
                body = post.content
                fm = dict(post.metadata)
                rel_path = fp.relative_to(root)
                cid = str(rel_path).replace("\\", "/").replace(".md", "")
                data = {**fm, "id": cid, "body": body}
                concept = ConceptModel.model_validate(data)
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
            # Bulk delete existing concepts
            placeholders = ", ".join([f"'$cid'" for cid in cids])
            self.conn.execute(f"""
                MATCH (c:Concept) WHERE c.id IN [{placeholders}] DELETE c
            """)

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
        for item in parsed:
            source_id = item["cid"]
            body = item["body"]
            for raw_link in link_pattern.findall(body):
                target_id = raw_link.lstrip("./").replace("\\", "/").replace(".md", "").lstrip("/")
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
                "MATCH (c:Concept {id: $id}) DELETE c",
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
        """Remove an asset's edge then the node (vector-indexed nodes block SET)."""
        self.conn.execute(
            """
            MATCH (c:Concept {id: $cid})-[r:INCLUDES_ASSET]->(i:ImageAsset {id: $iid})
            DELETE r
            """,
            {"cid": concept_id, "iid": asset_id},
        )
        self.conn.execute(
            "MATCH (i:ImageAsset {id: $iid}) DELETE i",
            {"iid": asset_id},
        )

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

    def _write_okf(self, concept: ConceptModel, output_path: Path) -> None:
        """Internal: serialize a ConceptModel to an OKF .md file."""
        data = concept.model_dump()
        body = data.pop("body", "")
        data.pop("id", None)
        data.pop("embedding", None)

        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()

        yaml_str = yaml.dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
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

    def search_hybrid(
        self,
        query: str,
        concept_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        exclude_reserved: bool = True,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: RRF fusion of vector + FTS with optional graph filters."""
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

        # Stage 3: Reciprocal Rank Fusion (RRF, k=60)
        all_ids = set(vec_scores) | set(fts_scores)
        combined: List[tuple[str, float]] = []
        for cid in all_ids:
            v_rank = (
                sorted(vec_scores, key=vec_scores.get, reverse=True).index(cid) + 1
                if cid in vec_scores
                else 0
            )
            f_rank = (
                sorted(fts_scores, key=fts_scores.get, reverse=True).index(cid) + 1
                if cid in fts_scores
                else 0
            )
            score = (1 / (60 + v_rank) if v_rank else 0) + (
                1 / (60 + f_rank) if f_rank else 0
            )
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in combined[:limit]]

        # Stage 4: Apply graph filters
        if concept_type or tags or parent_id or exclude_reserved:
            where_clauses: list[str] = []
            params: Dict[str, Any] = {"ids": top_ids}

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

            where_str = " AND ".join(where_clauses) if where_clauses else "true"

            cypher = f"""
            MATCH (c:Concept)
            WHERE c.id IN $ids AND {where_str}
            RETURN c.id, c.title, c.type, c.description, c.tags
            """
            filtered = self.conn.execute(cypher, params)
            filtered_rows = filtered.rows_as_dict().get_all()

            # Preserve RRF ordering
            ordered = [row for row in filtered_rows if row["c.id"] in top_ids]
            top_concepts: List[Dict[str, Any]] = []
            for row in ordered:
                row_id = row["c.id"]
                score = combined[[cid for cid, _ in combined].index(row_id)][1]
                desc = row["c.description"]
                if desc and len(desc) > 200:
                    desc = desc[:200] + "..."
                top_concepts.append({
                    "id": row_id,
                    "title": row["c.title"],
                    "type": row["c.type"],
                    "description": desc,
                    "tags": row["c.tags"],
                    "relevance_score": score,
                })
            return top_concepts

        # No filters — return top IDs directly
        return [
            {
                "id": cid,
                "title": self._get_property(cid, "title"),
                "type": self._get_property(cid, "type"),
                "description": (
                    (self._get_property(cid, "description") or "")[:200] + "..."
                ),
                "tags": self._get_property(cid, "tags"),
                "relevance_score": score,
            }
            for cid, score in combined[:limit]
        ]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        start_id: str,
        relationship: str = "CONTAINS",
        direction: str = "OUTGOING",
        depth: int = 1,
        node_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Navigate graph relationships with whitelisted edges and depth cap."""
        ALLOWED_RELS = {"CONTAINS", "LINKS_TO"}
        if relationship not in ALLOWED_RELS:
            raise ValueError(f"Invalid relationship. Must be one of {ALLOWED_RELS}")
        depth = max(1, min(int(depth), 5))

        if direction == "OUTGOING":
            pattern = f"-[{relationship}*1..{depth}]->(target:Concept)"
        elif direction == "INCOMING":
            pattern = f"<-[{relationship}*1..{depth}]-(target:Concept)"
        else:  # BOTH
            pattern = f"-[{relationship}*1..{depth}]-(target:Concept)"

        where_clause = ""
        if node_type:
            where_clause = f"WHERE target.type = '{node_type}'"

        cypher = f"""
        MATCH (start {{id: $start_id}}){pattern}
        {where_clause}
        RETURN target.id, target.title, target.type, target.description, target.tags
        LIMIT 100
        """
        results = self.conn.execute(cypher, {"start_id": start_id})
        rows = results.rows_as_dict().get_all()
        return [
            {
                "id": row["target.id"],
                "title": row["target.title"],
                "type": row["target.type"],
                "description": row["target.description"],
                "tags": row["target.tags"],
            }
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
