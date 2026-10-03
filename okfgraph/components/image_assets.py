"""ImageAssetManager — okf-asset:// URI storage, content-hash deduplication,
and text-based image search.

Encoding delegates to the injected EmbeddingEngine (text captions) and its
lazy vision session (image bytes, 0.8.0+); write-epoch bumps route to the
injected SchemaManager.
"""

from __future__ import annotations
import base64
import hashlib
import logging
import math
import mimetypes
import re
import urllib.parse
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from okfgraph.images import (
    EmbedRoute,
    IngestMode,
    _is_remote_src,
    build_extracted_images,
    fallback_caption,
    plan_embedding,
    prepare_vision_rgb,
)
from okfgraph.models import ChunkModel, ConceptModel

logger = logging.getLogger(__name__)

class ImageAssetManager:
    def __init__(
        self,
        conn,
        embed_engine,
        schema_mgr,
        allow_remote_images: bool,
        allowed_image_domains: List[str],
        bundle_root,
        db=None,
    ):
        self.conn = conn
        # Ladybug Database handle for short-lived index connections
        # (same QUERY_*_INDEX segfault workaround as SearchEngine).
        self._db = db
        self.embed_engine = embed_engine
        self.schema_mgr = schema_mgr
        self.allow_remote_images = allow_remote_images
        self.allowed_image_domains = allowed_image_domains or []
        self.bundle_root = bundle_root

    def _ingest_concept_images(
        self,
        concept_id: str,
        body: str,
        base_dir: Path,
        mode: "str | IngestMode",
    ) -> Dict[str, int]:
        """Extract, embed, and store the images referenced by a concept.

        Caption images embed their caption with the text model; vision
        images embed their bytes with the ONNX vision model (0.8.0+).
        The vision content hash covers (model, precision, contract), so a
        contract change re-embeds only image vectors.

        Unchanged images (same content hash) are skipped on re-import. Images
        removed from the document are pruned. Pre-0.7.0 rows stored with
        ``route='omni'`` hash a different payload, so they miss the reuse check
        and are re-embedded by caption automatically.
        Returns a small stats dict.
        """
        mode = IngestMode.coerce(mode)

        # Resolve relative image paths against the file's dir, then bundle root.
        search_dirs: List[Path] = []
        for d in (Path(base_dir), self.bundle_root):
            if d not in search_dirs:
                search_dirs.append(d)

        images = build_extracted_images(
            concept_id, body, search_dirs=search_dirs,
            allow_remote=self.allow_remote_images,
            allowed_domains=self.allowed_image_domains,
            bundle_root=self.bundle_root,
        )

        stats = {"total": len(images), "text": 0, "vision": 0, "omni": 0, "reused": 0, "pruned": 0, "skipped": 0}
        if not images and not self._concept_has_assets(concept_id):
            return stats

        # Node-direct lookup (0.2.18): concept replacement (DETACH DELETE
        # on reimport) drops INCLUDES_ASSET edges, so an edge-joined lookup
        # goes blind exactly when it is needed (DB-only reuse). Match nodes
        # by id among this run's planned assets instead.
        planned_aids = [img.asset_id for img in images]
        existing = self._existing_assets(planned_aids)  # {aid: (hash, has_data)}

        # --- Encode outside any DB transaction ---
        pending: List[Dict[str, Any]] = []
        planned_ids = set()
        for img in images:
            # Unresolvable refs (prose placeholders like `![alt](rel)` or
            # `![x](<id>)`, deleted files) carry no bytes. Embed nothing for
            # them on first sight — otherwise every doc that merely mentions
            # image syntax grows phantom asset rows. An asset that already
            # exists keeps the DB-only reuse path below (sources may be
            # legitimately gone on a detached graph). Remote srcs are exempt:
            # a failed fetch retries next import instead of giving up.
            known = existing.get(img.asset_id)
            if (
                img.data is None
                and not _is_remote_src(img.src)
                and (known is None or not known[1])
            ):
                stats["skipped"] += 1
                continue
            route, caption = plan_embedding(img, mode)
            if route is EmbedRoute.VISION:
                try:
                    rgb, rh, rw = self._prepare_vision(img)
                except (ValueError, RuntimeError) as exc:
                    # Undecodable bytes (or no vision wheel): graceful
                    # fallback to the caption path, like missing bytes.
                    logger.warning(
                        "image %s: %s — embedding caption instead",
                        img.asset_id, exc,
                    )
                    route = EmbedRoute.TEXT
                    caption = img.alt_text if img.has_alt_text else fallback_caption(
                        img.filename, img.index, img.concept_id
                    )
            if route is EmbedRoute.VISION:
                from okfgraph.components.embedding import vision_content_hash
                ve = self.embed_engine.vision_encoder
                content_hash = vision_content_hash(
                    ve.model_id, self._vision_precision(), img.data
                )
            else:
                content_hash = self._content_hash(route, (caption or "").encode("utf-8"))
            planned_ids.add(img.asset_id)

            if known is not None and known[0] == content_hash:
                stats["reused"] += 1
                # Re-link: concept replacement (DETACH DELETE on reimport)
                # drops INCLUDES_ASSET edges, and reuse otherwise writes
                # nothing — without this MERGE every doc edit silently
                # unlinks its images.
                self.conn.execute(
                    """
                    MATCH (c:Concept {id: $cid}), (i:ImageAsset {id: $iid})
                    MERGE (c)-[:INCLUDES_ASSET]->(i)
                    """,
                    {"cid": concept_id, "iid": img.asset_id},
                )
                continue

            # EmbedRoute.OMNI is never minted since 0.7.0; stale pre-0.7.0 rows
            # arrive here as TEXT (their stored omni hash missed above) and are
            # re-embedded by caption. `stats["omni"]` stays 0 by construction.
            if route is EmbedRoute.VISION:
                embedding = self._encode_vision_image(rgb, rh, rw)
                stats["vision"] += 1
                caption = ""  # bytes embedded, not text
            else:
                embedding = self.embed_engine._encode(caption or img.filename, task="Document")
                stats["text"] += 1

            pending.append({
                "img": img,
                "route": route.value,
                "caption": caption or "",
                "content_hash": content_hash,
                "embedding": embedding,
            })
        if stats["vision"] and self.embed_engine.device == "cpu":
            logger.info(
                "vision batch on CPU: %d image(s) at ~0.6-6.7 s each; "
                "use device='cuda' or mode='text' (captions) to skip the wait",
                stats["vision"],
            )

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
        """Hash that changes whenever the embedding should be recomputed.

        Frozen for the TEXT route (caption churn on upgrade would re-embed
        every caption row for identical vectors); the VISION route hashes
        via `vision_content_hash` (model + precision + contract).
        """
        h = hashlib.sha256()
        h.update(route.value.encode("utf-8"))
        h.update(b"|")
        h.update(payload or b"")
        return h.hexdigest()

    def _vision_precision(self) -> str:
        """Resolved vision precision for the content hash (0.8.0).

        Mirrors the session's `auto`-follows-device resolution (embroiders
        `Precision::resolve` against the landed device): an opened session
        reports itself, otherwise the device request plus the CUDA probe
        predicts it. The image-precision pin still fails closed on a real
        mismatch at open; the hash only separates reuse buckets.
        """
        ve = self.embed_engine.vision_encoder
        cfg = (ve._precision_cfg if ve is not None else "auto") or "auto"
        if cfg != "auto":
            return cfg
        if ve is not None and ve.is_loaded:
            return "fp16" if ve.used_cuda else "fp32"
        device = ve._device if ve is not None else "auto"
        if device == "cpu":
            return "fp32"
        try:
            from embroider import cuda_available
            cuda = bool(cuda_available())
        except Exception:
            cuda = False
        return "fp16" if cuda else "fp32"

    def _prepare_vision(self, img) -> Tuple[bytes, int, int]:
        """Decode + resize image bytes for the vision encoder (0.8.0)."""
        try:
            import embroider
        except ImportError:
            raise RuntimeError(
                "image-content search needs embroider>=0.3 "
                "(JinaV5Vision); use mode=text (captions)"
            ) from None
        target_size_fn = getattr(embroider, "vision_target_size", None)
        if target_size_fn is None:
            raise RuntimeError(
                "image-content search needs embroider>=0.3 "
                "(vision_target_size); use mode=text (captions)"
            )
        return prepare_vision_rgb(img.data, target_size_fn=target_size_fn)

    def _encode_vision_image(self, rgb: bytes, h: int, w: int):
        """Embed one resized RGB buffer via the lazy vision session."""
        ve = self.embed_engine.vision_encoder
        if ve is None:
            raise RuntimeError(
                "vision encoder not wired (router without image support); "
                "use mode=text (captions)"
            )
        return ve.encode_image(rgb, h, w)


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


    def _existing_assets(self, asset_ids: List[str]) -> Dict[str, Any]:
        """Known assets among the planned ids: {aid: (content_hash, has_data)}.

        Node-direct (no edge join): concept replacement drops edges, and
        ``has_data`` tells never-resolved phantoms (stored ``b""``) apart
        from DB-only assets whose source files are legitimately gone.
        """
        if not asset_ids:
            return {}
        result = self.conn.execute(
            """
            MATCH (i:ImageAsset)
            WHERE i.id IN $aids
            RETURN i.id AS id, i.content_hash AS content_hash, i.data AS data
            """,
            {"aids": asset_ids},
        )
        out = {}
        for r in result.rows_as_dict().get_all():
            data = r["data"] or b""
            out[r["id"]] = (r["content_hash"], len(data) > 0)
        return out


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
        # Orphan cleanup WITHOUT a WHERE NOT EXISTS subquery (0.2.18): that
        # formulation segfaults ladybug 0.20.3's native runtime when it runs
        # after an earlier image-ingest transaction in the same session
        # (deterministic repro: ingest a doc with image refs, then ingest a
        # second doc whose upsert deletes — access violation inside the
        # delete's HNSW index maintenance; isolated statements survive, so
        # only the full sequence triggers it). Count-then-delete is proven
        # safe against the same recipe. Revisit on a ladybug upgrade.
        rows = self.conn.execute(
            """
            MATCH (i:ImageAsset {id: $iid})<-[:INCLUDES_ASSET]-(:Concept)
            RETURN count(*) AS n
            """,
            {"iid": asset_id},
        ).rows_as_dict().get_all()
        if rows and rows[0]["n"] == 0:
            self.conn.execute(
                "MATCH (i:ImageAsset {id: $iid}) DETACH DELETE i",
                {"iid": asset_id},
            )
        self.schema_mgr._bump_write_epoch()  # image set changed -> image index dirty


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
        self.schema_mgr._bump_write_epoch()  # new/updated image -> image index dirty


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
        """Find image assets from a text query via the vector index.

        The query is encoded with the text model. ``use_text_model=False``
        (omni text side) was removed in 0.7.0 with the torch path.
        """
        if not use_text_model:
            raise ValueError(
                "use_text_model=False was removed in 0.7.0 (torch path); "
                "image search is caption-based"
            )
        query_vec = self.embed_engine._encode(text_query, task="Query")

        if self._db is None:
            result = self.conn.execute(
                "CALL QUERY_VECTOR_INDEX('ImageAsset', 'image_omni_idx', $vec, $k) "
                "RETURN node, distance",
                {"vec": query_vec, "k": limit},
            )
            rows = result.rows_as_dict().get_all()
        else:
            import ladybug as lb

            c = lb.Connection(self._db)
            try:
                c.execute("LOAD EXTENSION VECTOR")
                result = c.execute(
                    "CALL QUERY_VECTOR_INDEX('ImageAsset', 'image_omni_idx', $vec, $k) "
                    "RETURN node, distance",
                    {"vec": query_vec, "k": limit},
                )
                rows = result.rows_as_dict().get_all()
            finally:
                c.close()
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

