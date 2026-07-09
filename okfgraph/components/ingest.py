"""IngestManager — high-level ingest entry points: PDF, Markdown, and
LLM "thoughts".

Extracted from ``okfgraph.router.OKFRouter`` sections:
  - ingest_pdf (router.py 3920–4107)
  - ingest_md (router.py 4108–4209)
  - ingest_thoughts (router.py 4210–end)

Depends on: ImportManager (bundle/single-concept import), SchemaManager
(index rebuild), the lint module, and the facade's write-lock context.
"""

from typing import Any, Callable, Dict, List, Optional

import logging

from okfgraph.components.lint import lint_converted_md, lint_converted_md_str

logger = logging.getLogger(__name__)


class IngestManager:
    def __init__(
        self,
        conn,
        import_mgr,
        schema_mgr,
        write_lock_ctx: Callable,
        bundle_root,
    ):
        self.conn = conn
        self.import_mgr = import_mgr
        self.schema_mgr = schema_mgr
        self._write_lock_ctx = write_lock_ctx
        self.bundle_root = bundle_root

    def ingest_pdf(self, *args, **kwargs) -> Dict[str, Any]: ...
    def ingest_md(self, *args, **kwargs) -> Dict[str, Any]: ...
    def _ingest_md_inner(self, *args, **kwargs) -> Dict[str, Any]: ...
    def ingest_thoughts(self, *args, **kwargs) -> Dict[str, Any]: ...
    def _ingest_thoughts_inner(self, *args, **kwargs) -> Dict[str, Any]: ...

    # Re-export lint helpers so callers can use them through the manager.
    lint_converted_md = staticmethod(lint_converted_md)
    lint_converted_md_str = staticmethod(lint_converted_md_str)
