"""ImageAssetManager — okf-asset:// URI storage, content-hash deduplication,
and text-based image search.

Encoding delegates to the injected EmbeddingEngine; write-epoch bumps route
to the injected SchemaManager.
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
    build_extracted_images,
    plan_embedding,
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
    ):
        self.conn = conn
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
            concept_id, body, search_dirs=search_dirs,
            allow_remote=self.allow_remote_images,
            allowed_domains=self.allowed_image_domains,
            bundle_root=self.bundle_root,
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
                embedding = self.embed_engine._encode_image(img.data)
                stats["omni"] += 1
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
        """Find image assets from a text query via the unified vector index.

        ``use_text_model=True`` (default) encodes the query with the lightweight
        text model — no omni load required, since both models share the vector
        space. Set it to ``False`` to route the query through the omni text side.
        """
        if use_text_model:
            query_vec = self.embed_engine._encode(text_query, task="Query")
        else:
            query_vec = self.embed_engine._encode_omni_text(text_query, task="Query")

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

