"""Public settings constants and read-only table for the facade."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

DEFAULT_MODEL_ID = 'google/embeddinggemma-2'
DEFAULT_RERANK_MODEL_ID = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
GRAPH_HOME = '.okfgraph'
DEFAULT_DB_PATH = '.okfgraph/okfgraph.db'
LEGACY_DB_PATH = 'okfgraph.db'
DEVICES = ('auto', 'cpu', 'cuda')
PRECISIONS = ('auto', 'fp32', 'fp16', 'int8')


@dataclass(frozen=True)
class Setting:
    name: str
    kind: str
    default: Any
    section: str
    help: str
    choices: Tuple[str, ...] = ()
    cli: bool = True
    mcp: bool = True
    flag: Optional[str] = None
    dest: Optional[str] = None
    invert: bool = False
    required_mcp: bool = False


SETTINGS = (
    Setting('db_path', 'str', '.okfgraph/okfgraph.db', 'database', 'Path to the Ladybug database file, relative to the current directory; init/import/ingest create it and its folders (okf default: .okfgraph/okfgraph.db; okf-mcp needs it set)', (), True, True, None, None, False, True),
    Setting('bundle_root', 'str', None, '', "Primary bundle root directory (default: bundle/ beside a db inside a .okfgraph folder, else okf uses the current directory and okf-mcp / okf serve the db's folder)", (), True, True, None, None, False, False),
    Setting('wal_mode', 'bool', False, 'database', 'Enable WAL mode for concurrent reads', (), True, True, None, None, False, False),
    Setting('extension_dir', 'str', None, 'database', "Directory holding libvector/libfts .lbug_extension files: load from there and never download (offline machines; fill it with 'okf setup extensions')", (), True, True, None, None, False, False),
    Setting('embedding_dim', 'int', 512, 'embedding', 'Truncated Matryoshka dimension (ladder 32/64/128/256/512/768/1024; switching forces a fresh reimport)', (), True, True, None, None, False, False),
    Setting('model_id', 'str', 'google/embeddinggemma-2', 'embedding', 'Text embedding model id (registry; switching forces a fresh reimport)', (), True, True, None, None, False, False),
    Setting('device', 'str', 'auto', 'embedding', 'Inference device: auto (CUDA when present), cpu or cuda', ('auto', 'cpu', 'cuda'), True, True, None, None, False, False),
    Setting('backend', 'str', 'embroider', 'embedding', "Embedding engine: 'embroider' (the crate, ONNX in this process) or 'external' (an OpenAI-shaped /embeddings HTTP endpoint that owns its own model)", ('embroider', 'external'), True, True, None, None, False, False),
    Setting('endpoint_url', 'str', '', 'embedding', "External engine base URL, e.g. http://127.0.0.1:8931 (required for backend='external')", (), True, True, None, None, False, False),
    Setting('api_key_env', 'str', '', 'embedding', "Name of the env var carrying the external endpoint's bearer key (optional)", (), True, True, None, None, False, False),
    Setting('embed_identity', 'str', 'auto', 'embedding', "How an external engine is identified: 'auto' asks the endpoint (GET /v1/models; an embroider host names its family, width, precision and space, anything else keeps the declared contract), 'declared' trusts the embed_* settings alone", ('auto', 'declared'), True, True, None, None, False, False),
    Setting('embed_family', 'str', '', 'embedding', "Declared family id for an external engine, e.g. 'nomic-v1.5' (required for backend='external' unless an embroider host answers the identity probe; must agree with it if given)", (), True, True, None, None, False, False),
    Setting('embed_pooling', 'str', 'mean', 'embedding', 'Declared pooling identity of the external engine', ('mean', 'last', 'cls'), True, True, None, None, False, False),
    Setting('embed_prefixes', 'str', 'none', 'embedding', "External per-task prefixes: 'none', a built-in family name (nomic-v1.5), or 'Query=search_query:|Document=search_document:' pairs", (), True, True, None, None, False, False),
    Setting('embed_max_tokens', 'int', 32768, 'embedding', "The external engine's true input token ceiling", (), True, True, None, None, False, False),
    Setting('embed_counter', 'str', 'heuristic', 'embedding', "Where external token-count budgets come from: 'server-tokenize' (the host's POST /tokenize, exact; fails over loudly), 'gguf-meta' (a chars/token prior read from the model's GGUF header), 'declared' (embed_chars_per_token), 'heuristic' (len/4, approximate, logs once)", ('server-tokenize', 'gguf-meta', 'declared', 'heuristic'), True, True, None, None, False, False),
    Setting('embed_counter_gguf', 'str', '', 'embedding', "Path to the model's GGUF file, required for embed_counter='gguf-meta'; the header's vocabulary yields a measured chars/token prior (pure Python, no tokenizer wheel)", (), True, True, None, None, False, False),
    Setting('embed_chars_per_token', 'float', 0.0, 'embedding', "Declared chars-per-token constant, required (0.5..64) for embed_counter='declared'", (), True, True, None, None, False, False),
    Setting('precision', 'str', 'auto', 'embedding', 'Weight precision: auto follows device (CUDA->FP16, CPU->FP32); int8 explicit only, needs a measured artifact', ('auto', 'fp32', 'fp16', 'int8'), True, True, None, None, False, False),
    Setting('image_model_id', 'str', None, 'embedding', "Vision model id for image-content search (None follows the text model's registry vision pairing — the gemma-2 vision slot for gemma-2 graphs, the jina omni slot for text-nano graphs; needs embroider>=0.15.1)", (), True, True, None, None, False, False),
    Setting('image_precision', 'str', 'auto', 'embedding', 'Vision weight precision: auto follows device; explicit fp16 on CPU fails fast; int8 is explicit-only', ('auto', 'fp32', 'fp16', 'int8'), True, True, None, None, False, False),
    Setting('image_backend', 'str', 'embroider', 'embedding', "Vision engine: 'embroider' (the crate, ONNX in this process) or 'external' (an embroider.serve vision host over HTTP, so no ONNX loads here)", ('embroider', 'external'), True, True, None, None, False, False),
    Setting('image_endpoint_url', 'str', '', 'embedding', "Base URL of an embroider vision host, e.g. http://127.0.0.1:8932 (required for image_backend='external')", (), True, True, None, None, False, False),
    Setting('cpu_arena', 'bool', False, 'embedding', 'Enable the CPU arena allocator (default off: ~8x lower peak RSS for ~1.4x encode time)', (), True, True, None, None, False, False),
    Setting('cache_dir', 'str', None, 'embedding', 'Model cache directory (default: ~/.cache/huggingface/hub)', (), True, True, None, None, False, False),
    Setting('max_length', 'int', None, 'embedding', 'Token truncation ceiling 1..=32768; raising it changes long-doc vectors — reimport fully after changing', (), True, True, None, None, False, False),
    Setting('enable_chunking', 'bool', True, 'import', 'Chunk documents during ingestion', (), True, True, '--no-chunking', 'no_chunking', True, False),
    Setting('chunk_size', 'int', 512, 'import', 'Chunk token budget: larger blocks are split to fit', (), True, True, None, None, False, False),
    Setting('chunk_overlap', 'int', 40, 'import', 'Overlap in words between chunks (must be < chunk_size)', (), True, True, None, None, False, False),
    Setting('allow_remote_images', 'bool', False, 'import', 'Allow fetching remote images (SSRF risk — use with caution)', (), True, True, None, None, False, False),
    Setting('allowed_image_domains', 'csv', [], 'import', 'Allowed domains for remote images (comma-separated)', (), True, True, None, None, False, False),
    Setting('mode', 'str', 'text', 'import', 'Image ingestion route: text (captions), optional/omni (ONNX vision content; needs a text-nano graph)', ('text', 'optional', 'omni'), False, False, None, None, False, False),
    Setting('batch_size', 'int', 32, 'import', 'Encode batch for import ops', (), False, False, None, None, False, False),
    Setting('identity', 'str', 'path', 'import', 'Concept identity: path (ids from file paths) or frontmatter (a frontmatter id: is the concept id; moves keep links)', ('path', 'frontmatter'), True, True, None, None, False, False),
    Setting('pdf_device', 'str', 'auto', 'import', 'PDF/Office conversion device: auto (CUDA when present), cpu or cuda (bobine, the pdf extra)', ('auto', 'cpu', 'cuda'), True, True, None, None, False, False),
    Setting('pdf_precision', 'str', 'auto', 'import', 'PDF/Office converter weight precision: auto follows device (CUDA->FP16, CPU->FP32), fp32, fp16 or int8 (bobine)', ('auto', 'fp32', 'fp16', 'int8'), True, True, None, None, False, False),
    Setting('pdf_idle_seconds', 'int', 300, 'import', "Seconds a long-running host (okf serve, okf-mcp) keeps the warm PDF converter's ONNX sessions after the last conversion; 0 releases them after every conversion", (), False, False, None, None, False, False),
    Setting('exclude_tags', 'csv', [], 'search', 'Tags hidden from search unless a call passes exclude_tags (e.g. superseded)', (), False, False, None, None, False, False),
    Setting('rerank_model_id', 'str', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'search', 'Cross-encoder for search rerank (registry id; cross-encoder/ms-marco-MiniLM-L-6-v2 is smaller and English-only)', (), True, True, None, None, False, False),
    Setting('rerank_device', 'str', 'auto', 'search', 'Rerank inference device: auto (CUDA when present), cpu or cuda', ('auto', 'cpu', 'cuda'), True, True, None, None, False, False),
    Setting('rerank_precision', 'str', 'auto', 'search', 'Rerank weight precision: auto (= fp32), fp32 or q8', ('auto', 'fp32', 'q8'), True, True, None, None, False, False),
    Setting('rerank_top_n', 'int', 30, 'search', 'Candidates the cross-encoder rescores per search (1..200)', (), True, True, None, None, False, False),
    Setting('rerank_window', 'int', None, 'search', "Token window for long texts (pair cap; default: the model's)", (), False, False, None, None, False, False),
    Setting('rerank_stride', 'int', None, 'search', 'Token step between long-text windows (default: half the window)', (), False, False, None, None, False, False),
    Setting('rerank_backend', 'str', 'embroider', 'search', "Rerank lane: 'embroider' (the crate cross-encoder) or 'external' (a POST /v1/rerank endpoint), 0.12.0", ('embroider', 'external'), True, True, None, None, False, False),
    Setting('rerank_endpoint_url', 'str', '', 'search', "External rerank base URL, e.g. http://127.0.0.1:8932 (optional; required for rerank_backend='external')", (), True, True, None, None, False, False),
    Setting('rerank_api_key_env', 'str', '', 'search', "Name of the env var carrying the external rerank endpoint's bearer key (optional)", (), True, True, None, None, False, False),
    Setting('rerank_identity', 'str', 'auto', 'search', "How an external reranker is identified: 'auto' asks the host (GET /v1/models; an embroider host names its family and ceiling and counts exactly), 'declared' trusts the rerank_* settings alone", ('auto', 'declared'), True, True, None, None, False, False),
    Setting('rerank_family', 'str', '', 'search', "Declared rerank family label for provenance, e.g. 'bge-v2-m3' (required for rerank_backend='external' unless an embroider host answers the identity probe; must agree with it if given)", (), True, True, None, None, False, False),
    Setting('rerank_max_tokens', 'int', 8192, 'search', "The external engine's declared per-pair token ceiling (the short/long split point)", (), False, False, None, None, False, False),
    Setting('rerank_count', 'str', 'heuristic', 'search', "How the external reranker counts tokens for the short/long split: heuristic (len/4, one warning) or server-tokenize (exact — the rerank host's POST /tokenize, gated: texts far from the pair cap never pay a call, only the ambiguous middle band does; host failure degrades to heuristic with one warning)", ('heuristic', 'server-tokenize'), True, True, None, None, False, False),
    Setting('efs', 'int', None, 'search', "Vector index search breadth, 16..1000 (higher: slower, closer to exact; default: ladybug's)", (), False, False, None, None, False, False),
)


def env_name(name: str) -> str:
    return f"OKFGRAPH_{name.upper()}"


def flag_name(name: str) -> str:
    return "--" + name.replace("_", "-")

MCP_REQUIRED = tuple(row.name for row in SETTINGS if row.required_mcp)
