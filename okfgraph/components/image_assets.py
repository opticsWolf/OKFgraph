"""ImageAssetManager — okf-asset:// URI storage, content-hash deduplication,
and text-based image search.

Encoding delegates to the injected EmbeddingEngine (text captions) and its
lazy vision session (image bytes, 0.8.0+); write-epoch bumps route to the
injected SchemaManager.
"""

from __future__ import annotations
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
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

logger = logging.getLogger(__name__)


def _is_lane_outage(exc: BaseException) -> bool:
    """True for ``EXTERNAL_UNAVAILABLE`` — an outage, not a bad image.

    ``OKFError`` builds ``StateError``, which subclasses ``RuntimeError``,
    so the caption fallback's ``except`` would otherwise swallow a downed
    vision host and turn every image of the import into a caption.
    """
    return getattr(exc, "code", "") == "EXTERNAL_UNAVAILABLE"


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
        #: caption fallbacks this concept took (one summary warn)
        self._vision_failures = 0
        self._vision_failure_reason = ""
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
            if d is None or d in search_dirs:
                continue
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
            prep = None
            if route is EmbedRoute.VISION:
                try:
                    prep = self._vision_encoder_call(img)
                except (ValueError, RuntimeError) as exc:
                    if _is_lane_outage(exc):
                        raise
                    # Undecodable bytes (or no vision wheel): graceful
                    # fallback to the caption path, like missing bytes.
                    self._note_vision_failure(exc)
                    route, caption = self._caption_route(img)
            if route is EmbedRoute.VISION:
                from okfgraph.components.embedding import vision_content_hash
                ve = self.embed_engine.vision_encoder
                # Precision first: on the external lane asking for it is
                # the identity probe (no model loads), and the hash needs
                # the landed value, not a prediction.
                precision = self._vision_precision()
                content_hash = vision_content_hash(
                    ve.model_id, precision, img.data
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
                try:
                    embedding = prep()
                except (ValueError, RuntimeError) as exc:
                    # 0.14.0 (step H): the open/encode half falls back too.
                    # A declared lane (no vision partner), a missing ONNX
                    # runtime, an image pin refusal or a host refusing one
                    # image all land here — the caption is a working row,
                    # the exception would have been a dropped concept.
                    # An *outage* is the exception: OKFError(StateError) is
                    # also a RuntimeError, so it has to be named explicitly
                    # — a downed host must stop the import, never caption
                    # every image.
                    if _is_lane_outage(exc):
                        raise
                    self._note_vision_failure(exc)
                    route, caption = self._caption_route(img)
                    content_hash = self._content_hash(
                        route, (caption or "").encode("utf-8"))
            if route is EmbedRoute.VISION:
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

        if self._vision_failures:
            # One line per concept, not one per image: a declared lane (or
            # a host refusing every image) must not write a hundred
            # identical warnings.
            logger.warning(
                "image ingestion: %d image(s) embedded as captions: %s",
                self._vision_failures, self._vision_failure_reason,
            )
            self._vision_failures = 0
            self._vision_failure_reason = ""
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

    def _note_vision_failure(self, exc: BaseException) -> None:
        """Count a caption fallback; keep the first reason for the summary."""
        self._vision_failures += 1
        if not self._vision_failure_reason:
            self._vision_failure_reason = f"{type(exc).__name__}: {str(exc)[:160]}"

    @staticmethod
    def _caption_route(img):
        """The caption route for an image the vision lane could not embed."""
        caption = (img.alt_text if img.has_alt_text
                   else fallback_caption(img.filename, img.index, img.concept_id))
        return EmbedRoute.TEXT, caption

    def _vision_encoder_call(self, img):
        """Prepare *img* and return a zero-arg call that embeds it.

        Two lanes, one seam: the crate session wants a resized RGB buffer
        (so the work happens here, once), the external host wants the
        original bytes and resizes on its side. Raising from this call is
        the caption fallback; raising from the returned callable is too
        (0.14.0, step H).
        """
        ve = self.embed_engine.vision_encoder
        if ve is None:
            raise RuntimeError(
                "vision encoder not wired (router without image support); "
                "use mode=text (captions)"
            )
        if getattr(ve, "backend", "embroider") == "external":
            self._probe_decodable(img.data)
            return lambda: ve.encode_image_bytes(img.data)
        rgb, rh, rw = self._prepare_vision(img)
        return lambda: ve.encode_image(rgb, rh, rw)

    @staticmethod
    def _probe_decodable(data: bytes) -> None:
        """Cheap decode check for the external lane (the host resizes).

        Pillow is not a hard dependency of okfgraph: without it the check
        is skipped and a corrupt image is caught by the host's refusal
        instead (which also falls back to the caption).
        """
        if not data:
            raise ValueError("image has no bytes")
        try:
            from PIL import Image
        except ImportError:      # pragma: no cover - Pillow ships with embroider
            return
        import io
        try:
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()
        except Exception as exc:  # noqa: BLE001 - any decode failure
            raise ValueError(f"image bytes do not decode: {exc}") from exc

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
        # The external lane asked for `auto` must not consult the crate at
        # all (0.14.0): opening *that* lane is an identity probe, not a
        # model load, so ask it — it names the precision it landed on.
        if ve is not None and (ve.is_loaded
                               or getattr(ve, "backend", "embroider") == "external"):
            landed = getattr(ve, "precision", None)
            if isinstance(landed, str) and landed in ("fp16", "fp32"):
                return landed
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
                "image-content search needs embroider>=0.14 "
                "(the vision slot: JinaV5Vision or GemmaVision); "
                "use mode=text (captions)"
            ) from None
        # The host-side resize contract differs per slot: the jina omni
        # grid vs the gemma vision processor (resize to multiples of 48,
        # patch 16, [-1,-1] pad). Read the encoder's model id — the lazy
        # store hands it over pre-open, like `_encode_vision_image` does.
        ve = self.embed_engine.vision_encoder
        if (getattr(ve, "model_id", "") or "").startswith("google/"):
            target_size_fn = getattr(embroider, "gemma_vision_target_size",
                                     None)
        else:
            target_size_fn = getattr(embroider, "vision_target_size", None)
        if target_size_fn is None:
            need = ("gemma_vision_target_size" if target_size_fn is None and
                    (getattr(ve, "model_id", "") or "").startswith("google/")
                    else "vision_target_size")
            raise RuntimeError(
                f"image-content search needs embroider>=0.14 "
                f"({need}); use mode=text (captions)"
            )
        return prepare_vision_rgb(img.data, target_size_fn=target_size_fn)

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
        nodes = [row.get("node", {}) for row in rows]
        owners = self._first_owner_by_asset(
            [n["id"] for n in nodes if isinstance(n, dict) and n.get("id")])
        out: List[Dict[str, Any]] = []
        for row in rows:
            node = row.get("node", {})
            if not isinstance(node, dict):
                continue
            # ``id`` is the owning concept (stable key on every result row);
            # the asset keeps its own id under ``asset_id``. An orphan asset
            # has no owner, so id and concept_id are None.
            owner = owners.get(node.get("id"))
            out.append({
                "id": owner,
                "concept_id": owner,
                "asset_id": node.get("id"),
                "file_name": node.get("file_name"),
                "alt_text": node.get("alt_text"),
                "embed_route": node.get("embed_route"),
                "distance": row.get("distance"),
                "relevance_score": 1 - row.get("distance", 0),
            })
        return out

    def _first_owner_by_asset(self, asset_ids: List[str]) -> Dict[str, str]:
        """Owning concept per asset: {asset_id: concept_id}.

        An asset shared by several concepts maps to the smallest concept id,
        so the choice is stable across runs.
        """
        if not asset_ids:
            return {}
        rows = self.conn.execute(
            """
            MATCH (c:Concept)-[:INCLUDES_ASSET]->(i:ImageAsset)
            WHERE i.id IN $aids
            RETURN i.id AS aid, c.id AS cid
            """,
            {"aids": asset_ids},
        ).rows_as_dict().get_all()
        owners: Dict[str, str] = {}
        for r in rows:
            current = owners.get(r["aid"])
            if current is None or r["cid"] < current:
                owners[r["aid"]] = r["cid"]
        return owners

