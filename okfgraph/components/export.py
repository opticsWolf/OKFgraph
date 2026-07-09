"""ExportManager — graph-enriched export and index-file generation.

Extracted from ``okfgraph.router.OKFRouter`` sections:
  - export (router.py 2652–2899)
  - export helpers (router.py 2900–2912)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class ExportManager:
    def __init__(self, conn):
        self.conn = conn

    def export_to_okf(self, concept_id: str, output_path: Path) -> None: ...
    def _enrich_body_with_graph_links(self, *args, **kwargs) -> str: ...
    def _generate_index_files(self, *args, **kwargs) -> None: ...
    def _write_okf(self, concept, output_path: Path) -> None: ...
    def export_bundle(self, *args, **kwargs) -> Dict[str, Any]: ...
    def _fetch_concepts(self, *args, **kwargs) -> List[Dict[str, Any]]: ...
    def _is_under_directory(self, concept_id: str, directory_id: str) -> bool: ...
