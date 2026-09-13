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
from okfgraph.components.embedding import EmbeddingEngine, LazyRustEncoder
from okfgraph.components.image_assets import ImageAssetManager
from okfgraph.components.search import SearchEngine
from okfgraph.components.import_ import ImportManager, parse_source_file
from okfgraph.components.export import ExportManager
from okfgraph.components.ingest import IngestManager
from okfgraph.components.converters import BobineConverter, ConvertedDocument
from okfgraph.components.ranking import ppr, seed_ranked_ppr, seeds
from okfgraph.components.links import (
    build_name_index,
    extract_md_links,
    extract_wikilinks,
    normalize_path_link,
    resolve_wiki,
)
from okfgraph.components.diff import DiffManager, DiffState, state_of_dir
from okfgraph.components.doctor import DoctorManager
from okfgraph.components.lint import lint_bundle

__all__ = [
    "SchemaManager",
    "DeltaDetector",
    "PurgeManager",
    "EmbeddingEngine",
    "LazyRustEncoder",
    "ImageAssetManager",
    "SearchEngine",
    "ImportManager",
    "ExportManager",
    "IngestManager",
    "BobineConverter",
    "ConvertedDocument",
    "parse_source_file",
    "seeds",
    "ppr",
    "seed_ranked_ppr",
    "build_name_index",
    "extract_md_links",
    "extract_wikilinks",
    "normalize_path_link",
    "resolve_wiki",
    "DiffManager",
    "DiffState",
    "state_of_dir",
    "DoctorManager",
    "lint_bundle",
]
