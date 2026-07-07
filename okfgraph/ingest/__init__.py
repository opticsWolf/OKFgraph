"""okfgraph.ingest — PDF/Office → Markdown ingestion engine.

ONNX/Rapid-based replacement for the PaddleOCR/PaddlePaddle stack.
Runs on a single ``onnxruntime`` wheel with no CUDA-version coupling.

Public API
----------
- ``ConverterConfig`` / ``RoutingMode`` — configuration and routing modes.
- ``HybridConverter`` — core conversion pipeline (Qt-independent).
- ``OnnxRapidEngine`` — lazy ONNX model manager.
- ``html_tables_to_gfm`` — HTML table → GFM pipe-table converter.
- ``stage_images_as_okf_assets`` — okf-asset:// staging for extracted images.
- ``check_rapid_versions`` — runtime version check (Gap #15).
"""

from okfgraph.ingest.config import ConverterConfig, RoutingMode
from okfgraph.ingest.engine import OnnxRapidEngine
from okfgraph.ingest.converter import HybridConverter
from okfgraph.ingest.tables import html_tables_to_gfm
from okfgraph.ingest.assets import (
    stage_images_as_okf_assets,
    asset_id,
    ASSET_STORE_DIRNAME,
)
from okfgraph.ingest.versions import check_rapid_versions

__all__ = [
    "ConverterConfig",
    "RoutingMode",
    "HybridConverter",
    "OnnxRapidEngine",
    "html_tables_to_gfm",
    "stage_images_as_okf_assets",
    "asset_id",
    "ASSET_STORE_DIRNAME",
    "check_rapid_versions",
]
