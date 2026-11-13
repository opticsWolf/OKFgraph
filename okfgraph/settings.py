"""One settings table for OKFgraph.

Canonical names are the ``OKFRouter`` constructor kwargs. Everything else
is derived from the table below — never hand-maintained elsewhere:

- CLI flag ``--<kebab(name)>`` (``enable_chunking`` inverts to ``--no-chunking``)
- MCP boot flag (same spelling; ``db_path`` is required there)
- TOML key ``[<section>] <name>``, top-level ``bundle_root`` and ``[[roots]]``
- env ``OKFGRAPH_<NAME_UPPER>``

Precedence stays CLI > env > TOML > defaults, evaluated **per key over the
layers that set it**: a layer's value counts whenever that layer set the key
at all, so a higher layer can reset a lower layer's value back to the field
default (``--chunk-overlap 0`` is honoured; ``OKFGRAPH_EMBEDDING_DIM=512``
overrides a TOML ``1024``).

TOML discovery: ``okfgraph.toml`` in the CWD, then in the bundle root, then
``~/.config/okfgraph/config.toml`` (first match wins). Invalid TOML or failed
validation raise ``ValueError`` — invalid configuration is refused, not
silently skipped and not merely logged.
"""

import argparse
import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Default text embedding model. Adding registry entries never moves this:
#: different weights = different vector space, so the default is frozen
#: (a model switch forces a full reimport, enforced by the model pin).
#: The default embedding engine (0.12.0): EmbeddingGemma 2 — native 768,
#: Matryoshka ladder 128/256/512/768 (the 512 default stays on-ladder), 8K
#: context, mean pooling + Gemma prompts applied inside the crate, and
#: Apache-2.0 where jina-v5-small is CC BY-NC. Existing graphs keep their
#: pinned model: one model per graph, never moved by a registry change.
DEFAULT_MODEL_ID = "google/embeddinggemma-2"
#: The 0.11-era default was `jinaai/jina-embeddings-v5-text-small-retrieval`
#: (task prompts + last-token pooling); it stays registered — a graph can
#: still be created on it with an explicit `model_id`, and every graph
#: created on 0.11 keeps its pin.

#: Cross-encoder for opt-in search rerank (0.11.0). It writes no vectors,
#: so switching it needs no reimport. Multilingual by default (the
#: Phase 3 eval: English-only ``cross-encoder/ms-marco-MiniLM-L-6-v2``
#: lowered German results); that model stays an option for English graphs.
DEFAULT_RERANK_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

#: Full Matryoshka ladder, mirroring OKFRouter.ALLOWED_DIMS — the router
#: warns (not errors) off-ladder, so the settings layer warns the same way.
_DIM_LADDER = (32, 64, 128, 256, 512, 768, 1024)

#: Device and weight-precision words the ONNX lanes accept. Bobine exports the
#: same tuples (``bobine.DEVICES`` / ``bobine.PRECISIONS``); a test pins them,
#: so the table can validate on machines without the pdf extra and never drift
#: from the converter's own vocabulary.
DEVICES = ("auto", "cpu", "cuda")
PRECISIONS = ("auto", "fp32", "fp16", "int8")

#: A graph's home (0.12.0): the database and its bundle tree live in one
#: hidden directory, ``.okfgraph/okfgraph.db`` + ``.okfgraph/bundle``.
GRAPH_HOME = ".okfgraph"
DEFAULT_DB_PATH = f"{GRAPH_HOME}/okfgraph.db"
#: Pre-0.12.0 default. A CWD holding it (and no ``.okfgraph/``) keeps it.
LEGACY_DB_PATH = "okfgraph.db"

_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")

#: Retired spellings (pre-0.10.0), used only to raise a clear migration
#: error instead of an "unknown key" shrug. Compat itself is irrelevant:
#: nothing is accepted, the message just says where the name went.
_LEGACY_TOML = {
    ("database", "path"): "[database] path -> [database] db_path",
    ("database", "dim"): "[database] dim -> [embedding] embedding_dim",
    (None, "bundle"): "top-level 'bundle' -> 'bundle_root'",
    ("import", "no_chunking"): "[import] no_chunking -> [import] enable_chunking",
    ("embedding", "omni_model_id"):
        "embedding.omni_model_id was removed (torch path, gone since 0.7.0) — delete the key",
}
_LEGACY_ENV = {
    "OKFGRAPH_DB": "OKFGRAPH_DB_PATH",
    "OKFGRAPH_DIM": "OKFGRAPH_EMBEDDING_DIM",
    "OKFGRAPH_MODEL": "OKFGRAPH_MODEL_ID",
    "OKFGRAPH_BUNDLE": "OKFGRAPH_BUNDLE_ROOT",
    "OKFGRAPH_IMAGE_MODEL": "OKFGRAPH_IMAGE_MODEL_ID",
    "OKFGRAPH_NO_CHUNKING": "OKFGRAPH_ENABLE_CHUNKING (inverted sense: 0 disables)",
    "OKFGRAPH_OMNI_MODEL_ID":
        "OKFGRAPH_OMNI_MODEL_ID was removed (torch path, gone since 0.7.0) — unset it",
}


@dataclass(frozen=True)
class Setting:
    """One canonical field.

    kind: "str" | "int" | "bool" | "csv" (list[str]; CSV on env/CLI).
    section: TOML section, "" for top level.
    cli/mcp: include as a CLI global flag / MCP boot flag.
    flag/dest: override the derived flag spelling / argparse dest.
    invert: store-true flag whose *presence* flips a True default.
    required_mcp: the MCP server refuses to boot unless some layer (flag,
        env or TOML) sets it — the default is never good enough there.
    """

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


#: The table. Row order = validation report order (database, embedding, import, search).
SETTINGS = [
    # ── database ────────────────────────────────────────────────────────────
    Setting("db_path", "str", DEFAULT_DB_PATH, "database",
            "Path to the Ladybug database file, relative to the current "
            "directory; init/import/ingest create it and its folders (okf default: "
            f"{DEFAULT_DB_PATH}; okf-mcp needs it set)", required_mcp=True),
    # Top-level TOML key (not inside [database]); no default here — see
    # Settings.default_bundle_root (``<home>/bundle`` for a db inside a
    # .okfgraph home, else the CLI's "." / serve's and MCP's db parent).
    Setting("bundle_root", "str", None, "",
            "Primary bundle root directory (default: bundle/ beside a db "
            "inside a .okfgraph folder, else okf uses the current "
            "directory and okf-mcp / okf serve the db's folder)"),
    Setting("wal_mode", "bool", False, "database",
            "Enable WAL mode for concurrent reads"),
    Setting("extension_dir", "str", None, "database",
            "Directory holding libvector/libfts .lbug_extension files: load "
            "from there and never download (offline machines; fill it with "
            "'okf setup extensions')"),
    # ── embedding ───────────────────────────────────────────────────────────
    Setting("embedding_dim", "int", 512, "embedding",
            "Truncated Matryoshka dimension (ladder 32/64/128/256/512/768/1024; "
            "switching forces a fresh reimport)"),
    Setting("model_id", "str", DEFAULT_MODEL_ID, "embedding",
            "Text embedding model id (registry; switching forces a fresh reimport)"),
    Setting("device", "str", "auto", "embedding",
            "Inference device: auto (CUDA when present), cpu or cuda",
            choices=DEVICES),
    Setting("backend", "str", "embroider", "embedding",
            "Embedding engine: 'embroider' (the crate, ONNX in this "
            "process) or 'external' (an OpenAI-shaped /embeddings HTTP "
            "endpoint that owns its own model)",
            choices=("embroider", "external")),
    Setting("endpoint_url", "str", "", "embedding",
            "External engine base URL, e.g. http://127.0.0.1:8931 "
            "(required for backend='external')"),
    Setting("api_key_env", "str", "", "embedding",
            "Name of the env var carrying the external endpoint's bearer "
            "key (optional)"),
    Setting("embed_identity", "str", "auto", "embedding",
            "How an external engine is identified: 'auto' asks the endpoint "
            "(GET /v1/models; an embroider host names its family, width, "
            "precision and space, anything else keeps the declared "
            "contract), 'declared' trusts the embed_* settings alone",
            choices=("auto", "declared")),
    Setting("embed_family", "str", "", "embedding",
            "Declared family id for an external engine, e.g. 'nomic-v1.5' "
            "(required for backend='external' unless an embroider host "
            "answers the identity probe; must agree with it if given)"),
    Setting("embed_pooling", "str", "mean", "embedding",
            "Declared pooling identity of the external engine",
            choices=("mean", "last", "cls")),
    Setting("embed_prefixes", "str", "none", "embedding",
            "External per-task prefixes: 'none', a built-in family name "
            "(nomic-v1.5), or 'Query=search_query:|Document=search_document:' "
            "pairs"),
    Setting("embed_max_tokens", "int", 32768, "embedding",
            "The external engine's true input token ceiling"),
    Setting("embed_counter", "str", "heuristic", "embedding",
            "Where external token-count budgets come from: 'server-tokenize' "
            "(the host's POST /tokenize, exact; fails over loudly), "
            "'gguf-meta' (a chars/token prior read from the model's GGUF "
            "header), 'declared' (embed_chars_per_token), 'heuristic' "
            "(len/4, approximate, logs once)",
            choices=("server-tokenize", "gguf-meta", "declared", "heuristic")),
    Setting("embed_counter_gguf", "str", "", "embedding",
            "Path to the model's GGUF file, required for "
            "embed_counter='gguf-meta'; the header's vocabulary yields a "
            "measured chars/token prior (pure Python, no tokenizer wheel)"),
    Setting("embed_chars_per_token", "float", 0.0, "embedding",
            "Declared chars-per-token constant, required (0.5..64) for "
            "embed_counter='declared'"),
    Setting("precision", "str", "auto", "embedding",
            "Weight precision: auto follows device (CUDA->FP16, CPU->FP32); "
            "int8 explicit only, needs a measured artifact",
            choices=PRECISIONS),
    Setting("image_model_id", "str", None, "embedding",
            "Vision model id for image-content search (None follows the text "
            "model's registry vision pairing — the gemma-2 vision slot for "
            "gemma-2 graphs, the jina omni slot for text-nano graphs; needs "
            "embroider>=0.14)"),
    Setting("image_precision", "str", "auto", "embedding",
            "Vision weight precision: auto follows device; explicit fp16 on "
            "CPU fails fast",
            choices=("auto", "fp32", "fp16")),
    Setting("image_backend", "str", "embroider", "embedding",
            "Vision engine: 'embroider' (the crate, ONNX in this process) or "
            "'external' (an embroider.serve vision host over HTTP, so no "
            "ONNX loads here)",
            choices=("embroider", "external")),
    Setting("image_endpoint_url", "str", "", "embedding",
            "Base URL of an embroider vision host, e.g. "
            "http://127.0.0.1:8932 (required for image_backend='external')"),
    Setting("cpu_arena", "bool", False, "embedding",
            "Enable the CPU arena allocator (default off: ~8x lower peak "
            "RSS for ~1.4x encode time)"),
    Setting("cache_dir", "str", None, "embedding",
            "Model cache directory (default: ~/.cache/huggingface/hub)"),
    Setting("max_length", "int", None, "embedding",
            "Token truncation ceiling 1..=32768; raising it changes long-doc "
            "vectors — reimport fully after changing"),
    # ── import ──────────────────────────────────────────────────────────────
    Setting("enable_chunking", "bool", True, "import",
            "Chunk documents during ingestion",
            flag="--no-chunking", dest="no_chunking", invert=True),
    Setting("chunk_size", "int", 512, "import",
            "Chunk token budget: larger blocks are split to fit"),
    Setting("chunk_overlap", "int", 40, "import",
            "Overlap in words between chunks (must be < chunk_size)"),
    Setting("allow_remote_images", "bool", False, "import",
            "Allow fetching remote images (SSRF risk — use with caution)"),
    Setting("allowed_image_domains", "csv", [], "import",
            "Allowed domains for remote images (comma-separated)"),
    # TOML/env-only defaults for the import/ingest ops — no CLI/MCP flag.
    Setting("mode", "str", "text", "import",
            "Image ingestion route: text (captions), optional/omni "
            "(ONNX vision content; needs a text-nano graph)",
            choices=("text", "optional", "omni"), cli=False, mcp=False),
    Setting("batch_size", "int", 32, "import",
            "Encode batch for import ops", cli=False, mcp=False),
    Setting("identity", "str", "path", "import",
            "Concept identity: path (ids from file paths) or frontmatter "
            "(a frontmatter id: is the concept id; moves keep links)",
            choices=("path", "frontmatter")),
    # PDF / Office conversion (bobine, the pdf extra). The device and
    # precision words are the converter's own (0.14.1); the fast "never"
    # routing loads no ONNX at all, so they only matter for ONNX routes.
    Setting("pdf_device", "str", "auto", "import",
            "PDF/Office conversion device: auto (CUDA when present), cpu or "
            "cuda (bobine, the pdf extra)",
            choices=DEVICES),
    Setting("pdf_precision", "str", "auto", "import",
            "PDF/Office converter weight precision: auto follows device "
            "(CUDA->FP16, CPU->FP32), fp32, fp16 or int8 (bobine)",
            choices=PRECISIONS),
    Setting("pdf_idle_seconds", "int", 300, "import",
            "Seconds a long-running host (okf serve, okf-mcp) keeps the warm "
            "PDF converter's ONNX sessions after the last conversion; 0 "
            "releases them after every conversion", cli=False, mcp=False),
    # ── search ──────────────────────────────────────────────────────────────
    # TOML/env only: `search --exclude-tags` is the per-call spelling, and
    # a global flag of the same name would collide with it.
    Setting("exclude_tags", "csv", [], "search",
            "Tags hidden from search unless a call passes exclude_tags "
            "(e.g. superseded)", cli=False, mcp=False),
    # Cross-encoder rerank (0.11.0): used only when a search asks for it
    # (rerank=True / rank='cross'); the session opens on first use.
    Setting("rerank_model_id", "str", DEFAULT_RERANK_MODEL_ID, "search",
            "Cross-encoder for search rerank (registry id; "
            "cross-encoder/ms-marco-MiniLM-L-6-v2 is smaller and English-only)"),
    Setting("rerank_device", "str", "auto", "search",
            "Rerank inference device: auto (CUDA when present), cpu or cuda",
            choices=("auto", "cpu", "cuda")),
    Setting("rerank_precision", "str", "auto", "search",
            "Rerank weight precision: auto (= fp32), fp32 or q8",
            choices=("auto", "fp32", "q8")),
    Setting("rerank_top_n", "int", 30, "search",
            "Candidates the cross-encoder rescores per search (1..200)"),
    Setting("rerank_window", "int", None, "search",
            "Token window for long texts (pair cap; default: the model's)",
            cli=False, mcp=False),
    Setting("rerank_stride", "int", None, "search",
            "Token step between long-text windows (default: half the window)",
            cli=False, mcp=False),
    Setting("rerank_backend", "str", "embroider", "search",
            "Rerank lane: 'embroider' (the crate cross-encoder) or 'external' "
            "(a POST /v1/rerank endpoint), 0.12.0",
            choices=("embroider", "external")),
    Setting("rerank_endpoint_url", "str", "", "search",
            "External rerank base URL, e.g. http://127.0.0.1:8932 (optional; "
            "required for rerank_backend='external')"),
    Setting("rerank_api_key_env", "str", "", "search",
            "Name of the env var carrying the external rerank endpoint's "
            "bearer key (optional)"),
    Setting("rerank_identity", "str", "auto", "search",
            "How an external reranker is identified: 'auto' asks the host "
            "(GET /v1/models; an embroider host names its family and "
            "ceiling and counts exactly), 'declared' trusts the rerank_* "
            "settings alone",
            choices=("auto", "declared")),
    Setting("rerank_family", "str", "", "search",
            "Declared rerank family label for provenance, e.g. 'bge-v2-m3' "
            "(required for rerank_backend='external' unless an embroider "
            "host answers the identity probe; must agree with it if given)"),
    Setting("rerank_max_tokens", "int", 8192, "search",
            "The external engine's declared per-pair token ceiling (the "
            "short/long split point)", cli=False, mcp=False),
    Setting("rerank_count", "str", "heuristic", "search",
            "How the external reranker counts tokens for the short/long "
            "split: heuristic (len/4, one warning) or server-tokenize "
            "(exact — the rerank host's POST /tokenize, gated: texts far "
            "from the pair cap never pay a call, only the ambiguous middle "
            "band does; host failure degrades to heuristic with one "
            "warning)", choices=("heuristic", "server-tokenize")),
    # TOML/env only: `search --efs` is the per-call spelling.
    Setting("efs", "int", None, "search",
            "Vector index search breadth, 16..1000 (higher: slower, closer to "
            "exact; default: ladybug's)", cli=False, mcp=False),
]


def env_name(name: str) -> str:
    """The env spelling for a canonical field name."""
    return f"OKFGRAPH_{name.upper()}"


def flag_name(name: str) -> str:
    """The derived CLI/MCP flag spelling for a canonical field name."""
    return "--" + name.replace("_", "-")


def _parse(kind: str, raw: Any, origin: str) -> Any:
    """Parse one value of the given kind; raise ValueError on garbage."""
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        word = str(raw).strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
        raise ValueError(
            f"{origin}: expected one of {', '.join(_TRUE_WORDS + _FALSE_WORDS)}"
            f" (got {raw!r})")
    if kind == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"{origin}: expected an integer (got {raw!r})") from None
    if kind == "float":
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"{origin}: expected a number (got {raw!r})") from None
    if kind == "csv":
        if isinstance(raw, (list, tuple)):
            items = [str(x).strip() for x in raw]
        else:
            items = [part.strip() for part in str(raw).split(",")]
        return [item for item in items if item]
    return str(raw)


def _parse_roots(raw: Any) -> Dict[str, str]:
    """Parse ``ALIAS=PATH`` items (CLI ``--root``); dicts pass through."""
    if isinstance(raw, dict):
        return dict(raw)
    parsed: Dict[str, str] = {}
    for item in raw or ():
        alias, sep, path = str(item).partition("=")
        if not sep or not alias or not path:
            raise ValueError(f"--root must be ALIAS=PATH, got {item!r}")
        parsed[alias] = path
    return parsed


def _handle_bundle_root(value: Any) -> str:
    """Collision guard for the primary root.

    ``--bundle-root ALIAS=PATH`` was retired (named roots ride ``--root``).
    A value containing '=' that is not an existing path is almost certainly
    that old spelling — refuse it with a pointer at ``--root``.
    """
    text = str(value)
    if "=" in text and not Path(text).exists():
        raise ValueError(
            f"--bundle-root ALIAS=PATH was retired: the primary root is "
            f"--bundle-root PATH, named roots move to --root ALIAS=PATH "
            f"(got {text!r})")
    return text


@dataclass
class Settings:
    """Flat canonical settings; attributes are the OKFRouter kwargs."""

    db_path: str = DEFAULT_DB_PATH
    bundle_root: Optional[str] = None
    wal_mode: bool = False
    extension_dir: Optional[str] = None
    embedding_dim: int = 512
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    precision: str = "auto"
    backend: str = "embroider"
    endpoint_url: str = ""
    api_key_env: str = ""
    embed_identity: str = "auto"
    embed_family: str = ""
    embed_pooling: str = "mean"
    embed_prefixes: str = "none"
    embed_max_tokens: int = 32768
    embed_counter: str = "heuristic"
    embed_counter_gguf: str = ""
    embed_chars_per_token: float = 0.0
    image_model_id: Optional[str] = None
    image_precision: str = "auto"
    image_backend: str = "embroider"
    image_endpoint_url: str = ""
    cpu_arena: bool = False
    cache_dir: Optional[str] = None
    max_length: Optional[int] = None
    enable_chunking: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 40
    allow_remote_images: bool = False
    allowed_image_domains: List[str] = field(default_factory=list)
    mode: str = "text"
    batch_size: int = 32
    identity: str = "path"
    pdf_device: str = "auto"
    pdf_precision: str = "auto"
    pdf_idle_seconds: int = 300
    exclude_tags: List[str] = field(default_factory=list)
    rerank_model_id: str = DEFAULT_RERANK_MODEL_ID
    rerank_device: str = "auto"
    rerank_precision: str = "auto"
    rerank_top_n: int = 30
    rerank_backend: str = "embroider"
    rerank_endpoint_url: str = ""
    rerank_api_key_env: str = ""
    rerank_identity: str = "auto"
    rerank_family: str = ""
    rerank_max_tokens: int = 8192
    rerank_count: str = "heuristic"
    rerank_window: Optional[int] = None
    rerank_stride: Optional[int] = None
    efs: Optional[int] = None
    roots:Dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------------- load --
    @classmethod
    def load(
        cls,
        bundle_root: Optional[str] = None,
        cli_args: Optional[Dict[str, object]] = None,
        *,
        environ: Optional[Dict[str, str]] = None,
        require: Tuple[str, ...] = (),
    ) -> "Settings":
        """Merge TOML, environment and CLI onto defaults.

        Args:
            bundle_root: extra TOML lookup directory (after the CWD).
            cli_args: dict keyed by canonical field names (``roots`` may be a
                ``ALIAS=PATH`` list or a dict; ``enable_chunking`` may ride as
                the inverted ``no_chunking`` bool — see ``cli_signal``).
            environ: env mapping (defaults to ``os.environ``; tests inject).
            require: field names some layer must set explicitly (the MCP
                server requires ``db_path``: its CWD is the client's).

        Raises:
            ValueError: retired spellings, garbage values, invalid TOML,
                a missing required field or failed validation. Invalid
                configuration is refused.
        """
        env = os.environ if environ is None else environ
        for legacy, message in _LEGACY_ENV.items():
            if legacy in env:
                raise ValueError(f"{legacy} was renamed: use {message}")

        taken: Dict[str, Any] = {}
        from_cli: Dict[str, Any] = {}
        data, toml_path = _read_toml(bundle_root)
        if data:
            _merge_layer(taken, _from_toml(data, toml_path))
        _merge_layer(taken, _from_env(env))
        if cli_args:
            for key, value in cli_args.items():
                row = _row_by_name.get(key)
                if key == "roots":
                    from_cli[key] = value
                elif row is None:
                    raise ValueError(f"unknown setting: {key}")
                elif row.name == "bundle_root":
                    from_cli[key] = _handle_bundle_root(value)
                else:
                    from_cli[key] = _parse(row.kind, value, flag_name(row.name))
            _merge_layer(taken, from_cli)

        for name in require:
            if name not in taken:
                row = _row_by_name[name]
                where = f"[{row.section}] {name}" if row.section else name
                raise ValueError(
                    f"{name} must be set: {flag_name(name)}, "
                    f"{env_name(name)} or okfgraph.toml {where}")

        settings = cls()
        for setting_row in SETTINGS:
            if setting_row.name in taken:
                setattr(settings, setting_row.name, taken[setting_row.name])
        if "roots" in taken:
            settings.roots = _parse_roots(taken["roots"])
        if "db_path" not in taken and _legacy_home():
            logger.warning(
                "using the legacy database ./%s: the default is now ./%s "
                "(move it there, or pin db_path in okfgraph.toml)",
                LEGACY_DB_PATH, DEFAULT_DB_PATH)
            settings.db_path = LEGACY_DB_PATH

        errors, warnings = settings.checks()
        for warning in warnings:
            logger.warning("%s", warning)
        if errors:
            raise ValueError(
                "Configuration invalid:\n"
                + "\n".join(f"  - {e}" for e in errors))
        return settings

    # ----------------------------------------------------------- validate --
    def checks(self) -> Tuple[List[str], List[str]]:
        """Validate the merged settings.

        Returns (errors, warnings). Errors refuse the configuration;
        warnings are logged (off-ladder dimensions degrade gracefully — the
        router owns the quality warning).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not self.db_path or not self.db_path.strip():
            errors.append("database.db_path must be a non-empty string")

        if not 32 <= self.embedding_dim <= 1024:
            errors.append(
                "embedding.embedding_dim must be between 32 and 1024")
        elif self.embedding_dim not in _DIM_LADDER:
            warnings.append(
                f"embedding.embedding_dim={self.embedding_dim} is not a "
                f"recommended Matryoshka dimension; consider 256 or 512")

        if not self.model_id or "/" not in self.model_id:
            errors.append(
                f"embedding.model_id must be 'owner/name', got '{self.model_id}'")
        elif any(c in self.model_id for c in ("'", '"', "\\", ";")):
            # The model pin stores the id in a Cypher string literal.
            errors.append(
                f"embedding.model_id must not contain quotes/semicolons, "
                f"got '{self.model_id}'")

        for row in SETTINGS:
            value = getattr(self, row.name)
            if row.choices and value not in row.choices:
                errors.append(
                    f"{row.section}.{row.name} must be one of "
                    f"{row.choices}, got '{value}'")
        if self.max_length is not None and not 1 <= self.max_length <= 32768:
            errors.append(
                f"embedding.max_length must be within 1..=32768, got "
                f"{self.max_length}")
        if self.cache_dir and not Path(self.cache_dir).is_absolute():
            errors.append("embedding.cache_dir must be an absolute path")
        if self.pdf_idle_seconds < 0:
            errors.append(
                f"import.pdf_idle_seconds must be >= 0, got {self.pdf_idle_seconds}")

        if self.backend == "external":
            if not self.endpoint_url.lower().startswith(("http://", "https://")):
                errors.append(
                    "embedding.endpoint_url must be an http(s) URL for "
                    "backend='external'")
            # With embed_identity='auto' an embroider host may supply the
            # family at boot; the engine refuses then if nobody does.
            if "/" in self.embed_family or (
                    not self.embed_family
                    and self.embed_identity == "declared"):
                errors.append(
                    "embedding.embed_family must be a short declared name "
                    "for backend='external' with embed_identity='declared'")
            if not 64 <= self.embed_max_tokens <= 1048576:
                errors.append(
                    "embedding.embed_max_tokens must be within 64..1048576")
            if self.embed_counter == "gguf-meta" and not (
                    self.embed_counter_gguf
                    and Path(self.embed_counter_gguf).is_file()):
                errors.append(
                    "embedding.embed_counter_gguf must be the path to an "
                    "existing GGUF file when embed_counter='gguf-meta'")
            if self.embed_counter == "declared" and not (
                    0.5 <= self.embed_chars_per_token <= 64.0):
                errors.append(
                    "embedding.embed_chars_per_token must be within 0.5..64 "
                    "when embed_counter='declared'")
        elif (self.endpoint_url or self.embed_family
              or self.embed_counter != "heuristic"
              or self.embed_counter_gguf or self.embed_chars_per_token):
            errors.append(
                "embedding.endpoint_url/embed_family/embed_counter* are "
                "external-backend settings — set "
                "embedding.backend='external' or drop them")

        if self.image_backend == "external":
            if not self.image_endpoint_url.lower().startswith(
                    ("http://", "https://")):
                errors.append(
                    "embedding.image_endpoint_url must be an http(s) URL "
                    "for image_backend='external'")
        elif self.image_endpoint_url:
            errors.append(
                "embedding.image_endpoint_url is an external-backend "
                "setting — set embedding.image_backend='external' or drop it")

        if self.rerank_backend == "external":
            if not self.rerank_endpoint_url.lower().startswith(("http://", "https://")):
                errors.append(
                    "search.rerank_endpoint_url must be an http(s) URL for "
                    "rerank_backend='external'")
            if "/" in self.rerank_family or (
                    not self.rerank_family
                    and self.rerank_identity == "declared"):
                errors.append(
                    "search.rerank_family must be a short declared label "
                    "for rerank_backend='external'")
            if not 64 <= self.rerank_max_tokens <= 1048576:
                errors.append(
                    "search.rerank_max_tokens must be within 64..1048576")
        elif self.rerank_endpoint_url or self.rerank_family or (
                self.rerank_count != "heuristic"):
            errors.append(
                "search.rerank_endpoint_url/rerank_family/rerank_count are "
                "external-lane settings — set "
                "search.rerank_backend='external' or drop them")

        if not 1 <= self.batch_size <= 256:
            errors.append("import.batch_size must be between 1 and 256")
        if not 64 <= self.chunk_size <= 8192:
            errors.append("import.chunk_size must be between 64 and 8192")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            errors.append("import.chunk_overlap must be >= 0 and < chunk_size")
        if any(not domain.strip() for domain in self.allowed_image_domains):
            errors.append("allowed_image_domains contains empty entries")
        if self.extension_dir and not Path(self.extension_dir).expanduser().is_dir():
            errors.append(
                f"database.extension_dir is not a directory: {self.extension_dir} "
                "(create it with 'okf setup extensions')")

        if not self.rerank_model_id or "/" not in self.rerank_model_id:
            errors.append(
                f"search.rerank_model_id must be 'owner/name', got "
                f"'{self.rerank_model_id}'")
        if not 1 <= self.rerank_top_n <= 200:
            errors.append(
                f"search.rerank_top_n must be within 1..200, got {self.rerank_top_n}")
        if self.rerank_window is not None and not 16 <= self.rerank_window <= 8192:
            errors.append(
                f"search.rerank_window must be within 16..8192, got "
                f"{self.rerank_window}")
        if self.rerank_stride is not None and not (
                1 <= self.rerank_stride <= (self.rerank_window or 8192)):
            errors.append(
                f"search.rerank_stride must be within 1..rerank_window, got "
                f"{self.rerank_stride}")
        if self.efs is not None and not 16 <= self.efs <= 1000:
            errors.append(f"search.efs must be within 16..1000, got {self.efs}")

        return errors, warnings

    # ------------------------------------------------------------ outputs --
    def default_bundle_root(self, fallback: str) -> str:
        """The primary bundle root a surface opens the router with.

        An explicit ``bundle_root`` always wins. Otherwise the layout of
        the db decides, not how it was named: a db inside a ``.okfgraph``
        home (the default ``.okfgraph/okfgraph.db``, or Silk's explicit
        ``<root>/.okfgraph/okfgraph.db``) gets ``<home>/bundle``, created
        on first use. Any other db path keeps the surface's ``fallback``
        (CLI ``.``, serve/MCP the db's parent), as before 0.12.0.

        A home whose db already exists without ``bundle/`` was filled
        against the fallback tree; it keeps that tree (one warning) so a
        bundle import never reads the old files as deleted.
        """
        if self.bundle_root:
            return self.bundle_root
        db = Path(self.db_path)
        if db.parent.name != GRAPH_HOME:
            return fallback
        bundle = db.parent / "bundle"
        if not bundle.is_dir():
            if db.exists():
                logger.warning(
                    "%s has no %s/: keeping bundle root %s (pass "
                    "--bundle-root to choose another)", db, bundle, fallback)
                return fallback
            bundle.mkdir(parents=True, exist_ok=True)
        return str(bundle)

    def router_kwargs(self) -> Dict[str, Any]:
        """Keyword args for ``OKFRouter``. ``db_path`` and ``bundle_root``
        are excluded: surfaces pick their bundle with
        :meth:`default_bundle_root` and pass ``db_path`` explicitly."""
        return dict(
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            embed_backend=self.backend,
            endpoint_url=self.endpoint_url,
            api_key_env=self.api_key_env,
            embed_family=self.embed_family,
            embed_identity=self.embed_identity,
            embed_pooling=self.embed_pooling,
            embed_prefixes=self.embed_prefixes,
            embed_max_tokens=self.embed_max_tokens,
            embed_counter=self.embed_counter,
            embed_counter_gguf=self.embed_counter_gguf,
            embed_chars_per_token=self.embed_chars_per_token,
            cache_dir=self.cache_dir,
            device=self.device,
            precision=self.precision,
            cpu_arena=self.cpu_arena,
            image_model_id=self.image_model_id,
            image_precision=self.image_precision,
            image_backend=self.image_backend,
            image_endpoint_url=self.image_endpoint_url,
            allow_remote_images=self.allow_remote_images,
            allowed_image_domains=list(self.allowed_image_domains),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            enable_chunking=self.enable_chunking,
            wal_mode=self.wal_mode,
            roots=dict(self.roots) or None,
            extension_dir=self.extension_dir,
            identity=self.identity,
            pdf_device=self.pdf_device,
            pdf_precision=self.pdf_precision,
            pdf_idle_seconds=self.pdf_idle_seconds,
            exclude_tags=list(self.exclude_tags),
            rerank_model_id=self.rerank_model_id,
            rerank_device=self.rerank_device,
            rerank_precision=self.rerank_precision,
            rerank_top_n=self.rerank_top_n,
            rerank_window=self.rerank_window,
            rerank_stride=self.rerank_stride,
            rerank_backend=self.rerank_backend,
            rerank_endpoint_url=self.rerank_endpoint_url,
            rerank_api_key_env=self.rerank_api_key_env,
            rerank_family=self.rerank_family,
            rerank_identity=self.rerank_identity,
            rerank_max_tokens=self.rerank_max_tokens,
            efs=self.efs,
        )


_row_by_name = {row.name: row for row in SETTINGS}

#: Fields the MCP server needs set explicitly (``Settings.load(require=…)``).
MCP_REQUIRED = tuple(row.name for row in SETTINGS if row.required_mcp)


def _legacy_home() -> bool:
    """The CWD predates the ``.okfgraph/`` home: ``./okfgraph.db`` and no
    ``./.okfgraph/``. Nothing is ever moved; the old file stays in use."""
    return Path(LEGACY_DB_PATH).exists() and not Path(GRAPH_HOME).exists()


def _merge_layer(target: Dict[str, Any], layer: Dict[str, Any]) -> None:
    """Later layers overwrite earlier ones, key by key."""
    for key, value in layer.items():
        target[key] = value


def _from_env(env: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in SETTINGS:
        key = env_name(row.name)
        if key not in env or env[key] == "":
            continue
        out[row.name] = _parse(row.kind, env[key], key)
    return out


def _read_toml(bundle_root: Optional[str]) -> Tuple[Optional[Dict], Optional[Path]]:
    """First match wins: CWD, bundle root, ~/.config/okfgraph/config.toml."""
    candidates = [Path.cwd() / "okfgraph.toml"]
    if bundle_root:
        candidates.append(Path(bundle_root) / "okfgraph.toml")
    candidates.append(Path.home() / ".config" / "okfgraph" / "config.toml")
    for path in candidates:
        if path.is_file():
            try:
                with open(path, "rb") as fh:
                    return tomllib.load(fh), path
            except tomllib.TOMLDecodeError as e:
                raise ValueError(f"{path}: invalid TOML: {e}") from None
    return None, None


def _from_toml(data: Dict, toml_path: Path) -> Dict[str, Any]:
    """Flatten sectioned TOML onto canonical field names."""
    out: Dict[str, Any] = {}
    sections: Dict[str, Dict] = {}
    for key, value in data.items():
        if key == "roots":
            out["roots"] = _roots_from_toml(value, toml_path.parent)
        elif isinstance(value, dict):
            sections[key] = value
        elif (None, key) in _LEGACY_TOML:
            raise ValueError(f"okfgraph.toml: {_LEGACY_TOML[(None, key)]}")
        else:
            row = _row_by_name.get(key)
            if row is not None and row.section == "":
                out[row.name] = _parse(
                    row.kind, value, f"okfgraph.toml: {key}")
            else:
                logger.warning(
                    "okfgraph.toml: unknown top-level key %r ignored", key)
    for section, fields in sections.items():
        for field_name, value in fields.items():
            if (section, field_name) in _LEGACY_TOML:
                raise ValueError(
                    f"okfgraph.toml: {_LEGACY_TOML[(section, field_name)]}")
            row = _row_by_name.get(field_name)
            if row is None or row.section != section:
                logger.warning(
                    "okfgraph.toml: [%s] %r is not a setting — ignored",
                    section, field_name)
                continue
            out[row.name] = _parse(
                row.kind, value, f"okfgraph.toml: [{section}] {field_name}")
    return out


def _roots_from_toml(entries: Any, base: Path) -> Dict[str, str]:
    """Parse ``[[roots]]`` list of {alias, path}; paths may be relative to
    the TOML file (like an Obsidian vault config)."""
    if not isinstance(entries, list):
        raise ValueError("[[roots]] must be a list of {alias, path} tables")
    parsed: Dict[str, str] = {}
    for entry in entries:
        if (not isinstance(entry, dict)
                or not isinstance(entry.get("alias"), str)
                or not isinstance(entry.get("path"), str)):
            raise ValueError(
                "[[roots]] entries need string 'alias' and 'path'")
        if entry["alias"] in parsed:
            raise ValueError(f"duplicate [[roots]] alias {entry['alias']!r}")
        relative = Path(entry["path"])
        parsed[entry["alias"]] = (
            entry["path"] if relative.is_absolute()
            else str(base / relative))
    return parsed


def cli_signal(args: Any) -> Dict[str, Any]:
    """Collect the settings an argparse namespace actually set.

    Table rows generate flags with ``default=SUPPRESS``, so an unset flag is
    simply absent from the namespace — presence, not truthiness, is what
    carries the value. Store-true inversion rows translate here
    (``no_chunking=True`` -> ``enable_chunking=False``). ``--root`` items are
    returned raw; ``Settings.load`` parses them.
    """
    out: Dict[str, Any] = {}
    for row in SETTINGS:
        if not row.cli:
            continue
        dest = row.dest or row.name
        value = getattr(args, dest, None)
        if value is None:
            continue
        out[row.name] = (not value) if row.invert else value
    raw_roots = getattr(args, "roots", None)
    if raw_roots:
        out["roots"] = raw_roots
    return out


def set_cli_flags(parser: Any, *, hidden: bool = False, mcp: bool = False) -> None:
    """Add the generated global flags to a parser.

    ``hidden=True`` marks the flags with ``_okf_global`` so the CLI's slim
    per-command help formatter hides them (they show once in top-level help,
    ``hidden=False`` there). ``mcp=True`` builds the MCP boot parser
    (rows with ``mcp=False`` skipped; ``required_mcp`` is enforced by
    ``Settings.load(require=…)`` so env/TOML can satisfy it). All
    defaults are ``argparse.SUPPRESS`` — see ``cli_signal``.
    """

    def _add(*a: Any, **k: Any) -> Any:
        act = parser.add_argument(*a, **k)
        if hidden:
            act._okf_global = True  # noqa: SLF001 — the formatter's marker
        return act

    for row in SETTINGS:
        if mcp and not row.mcp:
            continue
        if not row.cli:
            continue
        flag = row.flag or flag_name(row.name)
        help_text = row.help + (
            f" (choices: {', '.join(row.choices)})"
            if row.choices and row.kind != "bool" else "")
        if row.kind == "bool":
            _add(flag, action="store_true", default=argparse.SUPPRESS,
                 help=help_text)
        elif row.kind == "csv":
            _add(flag, default=argparse.SUPPRESS, metavar="A,B", help=help_text)
        else:
            opts: Dict[str, Any] = dict(default=argparse.SUPPRESS, help=help_text)
            if row.kind == "int":
                opts["type"] = int
            if row.kind == "float":
                opts["type"] = float
            if row.choices:
                opts["choices"] = row.choices
            _add(flag, **opts)
    _add("--root", action="append", default=argparse.SUPPRESS, dest="roots",
         metavar="ALIAS=PATH",
         help="Additional named bundle root (repeatable; combines with "
              "--bundle-root). Named roots mint @alias/rel IDs.")
