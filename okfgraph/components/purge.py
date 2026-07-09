"""PurgeManager — hard purge, soft-delete, and recovery-window management.

Extracted from ``okfgraph.router.OKFRouter`` sections:
  - purge (router.py 954–1064)
  - soft-delete with recovery (router.py 1065–1279)
"""

from typing import Any, Callable, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

# Soft-delete recovery window (seconds). Concepts deleted within this
# window can be recovered. Default: 24 hours.
SOFT_DELETE_WINDOW = 24 * 60 * 60


class PurgeManager:
    def __init__(self, conn, write_lock_ctx: Callable):
        self.conn = conn
        self._write_lock_ctx = write_lock_ctx

    def _purge_concept(self, concept_id: str) -> bool: ...
    def _soft_delete_concept(self, concept_id: str) -> bool: ...
    def _soft_delete_concept_inner(self, concept_id: str) -> bool: ...
    def _recover_concept(self, concept_id: str) -> bool: ...
    def _recover_concept_inner(self, concept_id: str) -> bool: ...
    def list_deleted_concepts(self) -> List[Dict[str, Any]]: ...
    def purge_deleted_concepts(self, older_than: Optional[int] = None) -> int: ...
    def _purge_deleted_concepts_inner(self, older_than: Optional[int]) -> int: ...
