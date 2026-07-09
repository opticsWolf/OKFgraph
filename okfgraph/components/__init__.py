"""OKFRouter component objects.

These are extracted from the monolithic ``okfgraph.router.OKFRouter`` as part
of the Phase 0–4 refactor (see ``docs/plan-router-refactor-components.md``).
Each component receives its dependencies explicitly via ``__init__`` (dependency
injection) rather than reaching into a shared ``self``.

Phase 0 status: classes and method signatures are scaffolded; method bodies are
stubs (``...``) until their respective phase moves the implementation over.
``lint`` is fully implemented (self-contained, no router state required).
"""

from okfgraph.components.schema import SchemaManager
from okfgraph.components.delta import DeltaDetector
from okfgraph.components.purge import PurgeManager
from okfgraph.components.embedding import EmbeddingEngine
from okfgraph.components.image_assets import ImageAssetManager
from okfgraph.components.search import SearchEngine
from okfgraph.components.import_ import ImportManager
from okfgraph.components.export import ExportManager
from okfgraph.components.ingest import IngestManager

__all__ = [
    "SchemaManager",
    "DeltaDetector",
    "PurgeManager",
    "EmbeddingEngine",
    "ImageAssetManager",
    "SearchEngine",
    "ImportManager",
    "ExportManager",
    "IngestManager",
]
