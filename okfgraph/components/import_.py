"""ImportManager — the full import pipeline: parse, encode, upsert, chunk,
link, image-stage, and index.

Extracted from ``okfgraph.router.OKFRouter`` sections:
  - parsing / import entry points (router.py 1598–1753)
  - import pipeline (router.py 1754–2406)
  - single-concept import helper (router.py 3799–3919)

Depends on: SchemaManager (index rebuild, meta), DeltaDetector (changed
files), EmbeddingEngine (encode), ImageAssetManager (images), and the
facade's write-lock context.
"""

from typing import Any, Callable, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class ImportManager:
    def __init__(
        self,
        conn,
        schema_mgr,
        delta_mgr,
        embed_engine,
        image_mgr,
        write_lock_ctx: Callable,
        bundle_root,
    ):
        self.conn = conn
        self.schema_mgr = schema_mgr
        self.delta_mgr = delta_mgr
        self.embed_engine = embed_engine
        self.image_mgr = image_mgr
        self._write_lock_ctx = write_lock_ctx
        self.bundle_root = bundle_root

    def _parse_source_file(self, file_path, root): ...
    def import_from_okf(self, *args, **kwargs) -> List[str]: ...
    def import_bundle(self, *args, **kwargs) -> List[str]: ...
    def _import_bundle_inner(self, *args, **kwargs) -> List[str]: ...
    def _import_single_concept(self, concept, body: str, mode: str) -> Dict[str, Any]: ...
    def _import_chunks_for_concept(self, parsed_item: Dict[str, Any]) -> None: ...
    def _batch_upsert_concepts(self, cids: List[str]): ...
    def _batch_build_directories(self, cids: List[str]): ...
    def _batch_extract_links(self, parsed: List[Dict[str, Any]]): ...
    def _extract_links_for_concept(self, concept_id: str, body: str): ...
    def _insert_concept(self, *args, **kwargs): ...
