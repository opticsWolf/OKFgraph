"""ImageAssetManager — okf-asset:// URI storage, content-hash deduplication,
and text-based image search.

Extracted from ``okfgraph.router.OKFRouter`` section:
  - image assets (router.py 2403–2652)

The facade owns ``allow_remote_images`` / ``allowed_image_domains`` /
``bundle_root`` and passes them in. Image search delegates encoding to the
``EmbeddingEngine`` passed via ``embed_engine``.
"""

from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class ImageAssetManager:
    def __init__(
        self,
        conn,
        embed_engine,
        allow_remote_images: bool,
        allowed_image_domains: List[str],
        bundle_root,
    ):
        self.conn = conn
        self.embed_engine = embed_engine
        self.allow_remote_images = allow_remote_images
        self.allowed_image_domains = allowed_image_domains or []
        self.bundle_root = bundle_root

    def _ingest_concept_images(self, *args, **kwargs) -> None: ...
    def _content_hash(self, route, payload: bytes) -> str: ...
    def _concept_has_assets(self, concept_id: str) -> bool: ...
    def _existing_asset_hashes(self, concept_id: str) -> Dict[str, str]: ...
    def _delete_image_asset(self, concept_id: str, asset_id: str) -> None: ...
    def _upsert_image_asset(self, concept_id: str, item: Dict[str, Any]) -> None: ...
    def list_images(self, concept_id: str) -> List[Dict[str, Any]]: ...
    def get_image_data(self, asset_id: str) -> Optional[Dict[str, Any]]: ...
    def search_images_with_text(self, text_query: str, *args, **kwargs) -> List[Dict[str, Any]]: ...
