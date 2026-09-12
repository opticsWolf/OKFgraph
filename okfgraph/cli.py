"""OKF CLI — Command-line interface for the OKF knowledge graph."""

import argparse
import cProfile
import io
import json
import logging
import sys
from pathlib import Path

from okfgraph.router import OKFRouter
from okfgraph.config import OKFConfig
from okfgraph.components.converters import BobineConverter

# ── Logging setup ──────────────────────────────────────────────────────────
# Structured logging with stdlib (Gap #10).
# Loguru was rejected: third-party dependency for a CLI tool where stdlib
# logging is sufficient. The goal is consistent, structured, debuggable
# logging without adding extra dependencies.

_LOG_HANDLER = None


def _setup_logging(verbose: bool = False, quiet: bool = False, log_file: str = "") -> None:
    """Configure logging for the CLI.

    Precedence: quiet > verbose > default.
    - quiet: ERROR and above only
    - default: INFO
    - verbose: DEBUG
    """
    global _LOG_HANDLER

    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicates across invocations
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Console handler with structured format
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    console.setLevel(level)
    root.addHandler(console)
    _LOG_HANDLER = console

    # Optional file handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)
        root.addHandler(file_handler)


def _teardown_logging() -> None:
    """Remove handlers to avoid leaks across CLI invocations."""
    global _LOG_HANDLER
    root = logging.getLogger()
    if _LOG_HANDLER and _LOG_HANDLER in root.handlers:
        root.removeHandler(_LOG_HANDLER)
    _LOG_HANDLER = None


# ── helpers ────────────────────────────────────────────────────────────────

class _SlimHelpFormatter(argparse.HelpFormatter):
    """Subcommand help without the repeated global flags.

    Global options (connection, models, logging) are identical on every
    command, so printing them 15 times costs agents ~4k tokens to learn
    nothing. They are documented once in top-level ``okf --help`` and
    remain fully functional on every subcommand.
    """

    def add_arguments(self, actions):
        super().add_arguments(
            [a for a in actions if not getattr(a, "_okf_global", False)]
        )

    def add_usage(self, usage, actions, groups=(), prefix=None):
        super().add_usage(
            usage,
            [a for a in actions if not getattr(a, "_okf_global", False)],
            groups,
            prefix,
        )


class _SubParser(argparse.ArgumentParser):
    """Subcommand parser with slim help + pointer to global options."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", _SlimHelpFormatter)
        kwargs.setdefault(
            "epilog",
            "Global options hidden; see 'okf --help' (or okfgraph.toml).",
        )
        super().__init__(*args, **kwargs)


def _add_global(parser, mark=True):
    """Add --db / --bundle / --dim / --cache-dir / --device to any subparser.

    With mark=True (subcommands) the flags are tagged so _SlimHelpFormatter
    hides them from per-command help; they stay functional and are shown
    once in top-level help (mark=False there).
    """
    def _add(*a, **k):
        act = parser.add_argument(*a, **k)
        if mark:
            act._okf_global = True  # noqa: SLF001 — our own marker
        return act

    _add("--db", default=None, help="Database path (default: okfgraph.db, or from okfgraph.toml)")
    _add("--bundle", default=None, help="Bundle root directory (default: ., or from okfgraph.toml)")
    _add("--dim", type=int, default=None, help="Embedding dimension (Matryoshka; default: 512, or from okfgraph.toml)")
    _add("--cache-dir", default=None, help="HuggingFace model cache directory (default: ~/.cache/huggingface, or from okfgraph.toml)")
    _add("--device", default=None, choices=["cpu", "cuda"], help="Inference device: cpu or cuda (default: cpu, or from okfgraph.toml)")
    _add("--omni-model-id", default=None, help="Multimodal model ID for image embeddings (default from okfgraph.toml)")
    _add("--chunk-size", type=int, default=None, help="Chunk size in words for overlap (default: 512, or from okfgraph.toml)")
    _add("--chunk-overlap", type=int, default=None, help="Overlap in words between chunks (default: 40, or from okfgraph.toml)")
    _add("--no-chunking", action="store_true", help="Disable chunking during ingestion")
    _add("--wal-mode", action="store_true", help="Enable SQLite WAL mode for concurrent reads (Gap #7a)")
    _add("--allow-remote-images", action="store_true", help="Allow fetching remote images (SSRF risk — use with caution)")
    _add("--allowed-image-domains", default=None, help="Comma-separated list of allowed domains for remote images (Gap #9a)")


def _add_logging_flags(parser, mark=True):
    """Add --verbose / --quiet / --log-file / --profile to a subparser."""
    for args, kwargs in [
        (("--verbose", "-v"), {"action": "store_true", "help": "Enable debug logging"}),
        (("--quiet", "-q"), {"action": "store_true", "help": "Suppress all logging except errors"}),
        (("--log-file",), {"default": "", "help": "Write logs to file (with 5MB rotation)"}),
        (("--profile",), {"action": "store_true", "help": "Enable cProfile for the current invocation (outputs to stdout)"}),
    ]:
        act = parser.add_argument(*args, **kwargs)
        if mark:
            act._okf_global = True  # noqa: SLF001 — our own marker


# Routers opened during a CLI invocation, closed (checkpointed) on exit so a
# writer never leaves an un-checkpointed WAL that a later open would reject.
_OPEN_ROUTERS = []


def _router(args):
    """Build an OKFRouter from parsed args (registered for cleanup on exit).

    Uses the config module to merge CLI args with TOML file and env vars.
    Precedence: CLI > env > file > defaults.
    """
    # Build CLI args dict (only non-None values override config)
    cli_dict = {}
    for attr in ("db", "bundle", "dim", "cache_dir", "device",
                 "omni_model_id", "chunk_size", "chunk_overlap",
                 "no_chunking", "mode", "batch_size",
                 "allow_remote_images", "wal_mode", "allowed_image_domains"):
        val = getattr(args, attr, None)
        if val is not None:
            cli_dict[attr] = val

    # Resolve bundle root for TOML lookup
    bundle_root = cli_dict.get("bundle") or "."

    # Load merged config
    config = OKFConfig.load(bundle_root=bundle_root, cli_args=cli_dict)

    # Build allowed_image_domains list
    allowed_domains = config.import_config.allowed_image_domains
    if getattr(args, "allowed_image_domains", None):
        allowed_domains = [d.strip() for d in args.allowed_image_domains.split(",") if d.strip()]

    router = OKFRouter(
        db_path=config.database.path,
        bundle_root=str(config.bundle),
        embedding_dim=config.database.dim,
        omni_model_id=config.embedding.omni_model_id,
        cache_dir=config.embedding.cache_dir,
        device=config.embedding.device,
        allow_remote_images=config.import_config.allow_remote_images,
        allowed_image_domains=allowed_domains,
        chunk_size=config.import_config.chunk_size,
        chunk_overlap=config.import_config.chunk_overlap,
        enable_chunking=not config.import_config.no_chunking,
        wal_mode=config.database.wal_mode,
    )
    _OPEN_ROUTERS.append(router)
    return router


def _close_routers():
    """Checkpoint + close every router opened this invocation."""
    while _OPEN_ROUTERS:
        router = _OPEN_ROUTERS.pop()
        try:
            router.close()
        except Exception:
            pass


# ── command handlers ───────────────────────────────────────────────────────

def _init(args):
    db_path = str(args.db)
    logger = logging.getLogger("cli")
    logger.info("initializing database at %s (dim=%d)", db_path, args.dim)
    _router(args)
    logger.info("database initialized (embedding_dim=%d)", args.dim)


def _model_info(args):
    """Show model cache status without loading the model."""
    logger = logging.getLogger("cli")
    info = OKFRouter.model_info(
        model_id=getattr(args, "model_id", "jinaai/jina-embeddings-v5-text-small-retrieval"),
        cache_dir=getattr(args, "cache_dir", None),
    )
    logger.info("model: %s", info['model_id'])
    logger.info("cache: %s", info['cache_dir'])
    if info["cached"]:
        logger.info("status: cached")
        logger.info("path: %s", info['snapshot_path'])
        size_gb = info["disk_usage_bytes"] / (1024 ** 3)
        logger.info("size: %.2f GB", size_gb)
    else:
        default_cache = OKFRouter.default_cache_dir()
        logger.info("status: not cached (will download on first use)")
        logger.info("will use: %s", default_cache)


def _import(args):
    logger = logging.getLogger("cli")
    router = _router(args)
    mode = getattr(args, "mode", "text")
    purge = getattr(args, "purge", False)
    if getattr(args, "import_all", False):
        bundle_path = Path(args.bundle) if args.bundle else None
        ids = router.import_mgr.import_bundle(
            bundle_path,
            batch_size=getattr(args, "batch_size", 32) or 32,
            mode=mode,
            purge_deleted=purge,
        )
        logger.info("imported %d concept(s) (mode: %s)", len(ids), mode)
        for cid in ids:
            n = len(router.image_mgr.list_images(cid))
            suffix = f"  [{n} image(s)]" if n else ""
            logger.info("  %s%s", cid, suffix)
    else:
        for fp in args.files:
            path = Path(fp)
            if not path.exists():
                logger.warning("skipping %s: file not found", fp)
                continue
            cid = router.import_from_okf(path, mode=mode)
            imgs = router.image_mgr.list_images(cid)
            suffix = f" ({len(imgs)} image(s), mode: {mode})" if imgs else ""
            logger.info("imported: %s%s", cid, suffix)


def _search(args):
    """Unified search: concepts (default), chunks, or images."""
    router = _router(args)
    target = getattr(args, "target", "concepts") or "concepts"
    limit = getattr(args, "limit", 10) or 10
    if target == "images":
        results = router.image_mgr.search_images_with_text(
            text_query=args.query,
            use_text_model=not getattr(args, "use_omni", False),
            limit=limit,
        )
        if not results:
            print("No image results found.")
            return
        print(f"Found {len(results)} image(s):\n")
        for i, r in enumerate(results, 1):
            label = r.get("alt_text") or r.get("file_name") or r.get("id")
            print(f"  {i}. [{r['relevance_score']:.4f}] {label} ({r.get('embed_route')})")
            print(f"     file: {r.get('file_name')}")
            print(f"     id: {r['id']}")
            print()
        return
    tags = args.tags.split(",") if getattr(args, "tags", None) else None
    filt = {
        "concept_type": getattr(args, "type", None),
        "tags": tags,
        "parent_id": getattr(args, "parent", None),
    }
    filt = {k: v for k, v in filt.items() if v is not None}
    if target == "chunks":
        if getattr(args, "hub_rerank", False):
            results = router.search_engine.search_chunks_with_hub_score(
                query=args.query,
                limit=limit,
                hub_weight=getattr(args, "hub_weight", 0.3),
            )
            if not results:
                print("No results found.")
                return
            print(f"Found {len(results)} result(s):\n")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['final_score']:.4f}] {r['parent_title']} §{r['chunk_index']}")
                print(f"     hub={r['hub_score']:.2f} rrf={r['rrf_score']:.4f}")
                print(f"     {r['chunk_text'][:150]}")
                print()
            return
        if getattr(args, "expand", False):
            results = router.search_engine.search_with_context(
                query=args.query,
                limit=min(limit, 20),
                context_hops=getattr(args, "context_hops", 1),
            )
            if not results:
                print("No results found.")
                return
            print(f"Found {len(results)} result(s):\n")
            for i, r in enumerate(results, 1):
                chunk = r["chunk"]
                print(f"  {i}. [{chunk['rrf_score']:.4f}] {chunk['parent_title']} §{chunk['chunk_index']}")
                print(f"     {chunk['chunk_text'][:150]}")
                if r["incoming_links"]:
                    titles = [l.get("title", l.get("id", "?")) for l in r["incoming_links"][:3]]
                    print(f"     ← linked by: {', '.join(titles)}")
                if r["outgoing_links"]:
                    titles = [l.get("title", l.get("id", "?")) for l in r["outgoing_links"][:3]]
                    print(f"     → links to: {', '.join(titles)}")
                if r["ancestry"]:
                    print(f"     path: {' → '.join(a['title'] for a in r['ancestry'])}")
                if r["siblings"]:
                    print(f"     siblings: {', '.join(s['title'] for s in r['siblings'][:3])}")
                print()
            return
        results = router.search_engine.search_chunks(
            query=args.query,
            limit=limit,
            **filt,
        )
        if not results:
            print("No chunk results found.")
            return
        print(f"Found {len(results)} chunk(s):\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['rrf_score']:.4f}] {r['block_type']} #{r['chunk_index']}")
            text = r.get("chunk_text", "")
            print(f"     {text[:150]}")
            if r.get("parent_title"):
                print(f"     parent: {r['parent_title']}")
            print(f"     id: {r['chunk_id']}")
            print()
        return
    results = router.search_hybrid(
        query=args.query,
        limit=limit,
        include_chunks=getattr(args, "chunks", False),
        **filt,
    )
    if not results:
        print("No results found.")
        return
    print(f"Found {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['relevance_score']:.4f}] {r['title']} ({r['type']})")
        desc = r.get("description") or ""
        if desc:
            print(f"     {desc[:120]}")
        if r.get("tags"):
            print(f"     tags: {', '.join(r['tags'])}")
        print(f"     id: {r['id']}")
        # Print matched chunks if requested
        if r.get("matched_chunks"):
            print(f"     matched chunks:")
            for mc in r["matched_chunks"][:3]:
                print(f"       [{mc['rrf_score']:.4f}] {mc['block_type']} #{mc['chunk_index']}")
                print(f"         {mc.get('chunk_text', '')[:100]}")
        print()


def _traverse(args):
    """Unified traversal: relationships, directory listing, or shortest path."""
    router = _router(args)
    target = getattr(args, "target", None)
    start = getattr(args, "start_id", "") or ""
    if not start:
        items = router.list_directory("")
        if not items:
            print("Directory is empty.")
            return
        print("Contents of '(root)':\n")
        for item in items:
            icon = "[D]" if item["type"] == "Directory" else "[F]"
            print(f"  {icon} {item['title']} ({item['type']})")
            print(f"     id: {item['id']}")
        return
    if target:
        nodes = router.search_engine.find_path(
            start, target, max_length=getattr(args, "max_path_length", 6)
        )
        if not nodes:
            print(f"No path found between '{start}' and '{target}'.")
            return
        print(f"Path ({len(nodes)} nodes):")
        for i, n in enumerate(nodes, 1):
            print(f"  {i}. {n.get('title', '?')} ({n.get('type', '?')})")
            print(f"     id: {n['id']}")
        return
    results = router.traverse(
        start_id=start,
        relationship=args.relationship,
        direction=args.direction,
        depth=args.depth,
        node_type=getattr(args, "type", None),
    )
    if not results:
        print("No results found.")
        return
    print(f"Found {len(results)} node(s):\n")
    for r in results:
        print(f"  {r['id']} ({r['type']})")
        if r.get("title"):
            print(f"    title: {r['title']}")


def _read(args):
    """Unified read: body (default), chunks, document, or context."""
    router = _router(args)
    include = getattr(args, "include", "body") or "body"
    cid = args.concept_id
    if include == "chunks":
        chunks = router.search_engine.get_chunks(cid)
        if not chunks:
            print("No chunks found for this concept.")
            return
        print(f"Chunks for '{cid}' ({len(chunks)} total):\n")
        for c in chunks:
            text = c.chunk_text[:120]
            print(f"  #{c.chunk_index} [{c.block_type}] {text}")
        return
    if include == "document":
        text = router.embed_engine.reconstruct_document(cid)
        if not text:
            print("No chunks found for this concept.")
            return
        if getattr(args, "output", None):
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"[OK] Reconstructed document written to {args.output}")
        else:
            print(text)
        return
    if include == "context":
        incoming = router.traverse(cid, "LINKS_TO", "INCOMING", 1)[:10]
        outgoing = router.traverse(cid, "LINKS_TO", "OUTGOING", 1)[:10]
        ancestry = router.search_engine._get_ancestry(cid)
        siblings = router.search_engine._get_siblings(cid)[:10]
        if incoming:
            print("Linked by:")
            for l in incoming:
                print(f"  {l.get('title', l.get('id', '?'))} (id: {l['id']})")
        if outgoing:
            print("Links to:")
            for l in outgoing:
                print(f"  {l.get('title', l.get('id', '?'))} (id: {l['id']})")
        if ancestry:
            print(f"Path: {' → '.join(a['title'] for a in ancestry)}")
        if siblings:
            print("Siblings:")
            for s in siblings:
                print(f"  {s['title']} ({s['type']})")
                print(f"     id: {s['id']}")
        if not (incoming or outgoing or ancestry or siblings):
            print(f"No context found for '{cid}'.")
        return
    concept = router.get_by_id(cid)
    if not concept:
        print(f"Concept '{cid}' not found.")
        return
    data = concept.model_dump()
    body = data.pop("body", "")
    data.pop("embedding", None)
    print(json.dumps(data, indent=2, default=str))
    if body:
        print(f"\n--- BODY ---\n{body}")


def _export(args):
    router = _router(args)
    if getattr(args, "export_all", False):
        tags = args.tags.split(",") if args.tags else None
        ids = router.export_mgr.export_bundle(
            output_dir=Path(args.output),
            directory_id=args.parent,
            concept_type=args.type,
            tags=tags,
        )
        print(f"[OK] Exported {len(ids)} concepts to {args.output}")
    else:
        cid = args.concept_id
        output_path = Path(args.output) / f"{cid}.md"
        router.export_to_okf(cid, output_path)
        print(f"[OK] Exported {cid} → {output_path}")


def _broken_links(args):
    logger = logging.getLogger("cli")
    router = _router(args)
    broken = router.list_broken_links()
    if not broken:
        logger.info("no broken links found")
        return
    logger.info("found %d broken link(s)", len(broken))
    for link in broken:
        logger.info("  %s → %s", link['source'], link['target'])


def _repair_links(args):
    logger = logging.getLogger("cli")
    router = _router(args)
    count = router.repair_links()
    logger.info("repaired %d link(s)", count)


def _reindex(args):
    logger = logging.getLogger("cli")
    router = _router(args)
    ran = router.schema_mgr.reindex(force=not getattr(args, "if_dirty", False))
    if ran:
        logger.info("search indexes rebuilt")
    else:
        logger.info("search indexes already up to date; nothing to do.")


def _ingest(args):
    """Unified ingest: markdown file, PDF (default), or raw thoughts.

    Delegates to IngestManager.ingest_md / ingest_pdf / ingest_thoughts.
    PDF conversion uses the configured DocumentConverter (bobine default).
    """
    logger = logging.getLogger("cli")
    router = _router(args)
    kind = getattr(args, "kind", "pdf") or "pdf"
    tags = args.tags.split(",") if getattr(args, "tags", None) else None
    if kind == "md":
        md_file = getattr(args, "md_file", None)
        if not md_file:
            print("[ERROR] --md-file is required for --kind md")
            return
        result = router.ingest_mgr.ingest_md(
            md_path=md_file,
            concept_id=getattr(args, "concept_id", None),
            title=getattr(args, "title", None),
            description=getattr(args, "description", None),
            tags=tags,
            mode=getattr(args, "mode", "text") or "text",
        )
        print(f"[OK] Imported {result['concept_id']} ({result['chunk_count']} chunks)")
        return
    if kind == "thoughts":
        if not getattr(args, "thoughts", None) or not getattr(args, "topic", None):
            print("[ERROR] --thoughts and --topic are required for --kind thoughts")
            return
        result = router.ingest_mgr.ingest_thoughts(
            args.thoughts,
            topic=args.topic,
            concept_id=getattr(args, "concept_id", None),
            tags=tags,
        )
        print(f"[OK] Stored thought {result['concept_id']}")
        return
    pdf_path = Path(getattr(args, "pdf_file", None) or "")
    if not pdf_path.name or not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        return

    auto_import = getattr(args, "auto_import", False)
    output_dir = getattr(args, "output", None)
    if not auto_import and not output_dir:
        output_dir = Path(".")

    def on_page(idx, total):
        print(f"  page {idx + 1}/{total}", end="\r")

    try:
        from okfgraph.components.converters import BobineConverter
        converter = BobineConverter(
            routing_mode=getattr(args, "routing_mode", "auto") or "auto",
            extract_images=not getattr(args, "no_extract_images", False),
        )
        result = router.ingest_mgr.ingest_pdf(
            pdf_path,
            auto_import=auto_import,
            output_dir=output_dir,
            mode=getattr(args, "mode", "text") or "text",
            batch_size=getattr(args, "batch_size", 32) or 32,
            purge_deleted=getattr(args, "purge", False),
            on_page=on_page,
            converter=converter,
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return
    print()  # newline after progress

    logger.info("written %s", result["md_path"])
    logger.info("assets in %s", result["image_dir"])
    if auto_import:
        for cid in result["concept_ids"]:
            n = len(router.image_mgr.list_images(cid))
            suffix = f"  [{n} image(s)]" if n else ""
            logger.info("  %s%s", cid, suffix)
    else:
        logger.info("run 'okf import --all --bundle %s' to import.", output_dir)


def _deleted_list(args):
    """List soft-deleted concepts with recovery status."""
    router = _router(args)
    deleted = router.purge_mgr.list_deleted_concepts()
    if not deleted:
        print("No soft-deleted concepts found.")
        return
    print(f"Soft-deleted concepts ({len(deleted)} total):\n")
    for d in deleted:
        status = "recoverable" if d["recoverable"] else "expired"
        print(f"  [{status}] {d['concept_id']}")
        print(f"    title: {d['title']}")
        print(f"    type: {d['type']}")
        print(f"    deleted: {d['deleted_at']} ({d['age_seconds']:.0f}s ago)")
        print()


def _deleted_recover(args):
    """Recover a soft-deleted concept."""
    router = _router(args)
    success = router.purge_mgr._recover_concept(args.concept_id)
    if success:
        print(f"[OK] Recovered concept '{args.concept_id}'.")
    else:
        print(f"[ERROR] Concept '{args.concept_id}' not found or past recovery window.")


def _deleted_purge(args):
    """Permanently delete expired soft-deleted concepts."""
    router = _router(args)
    older_than = getattr(args, "older_than", None)
    count = router.purge_mgr.purge_deleted_concepts(older_than=older_than)
    print(f"[OK] Permanently deleted {count} expired concept(s).")


def _shell(args):
    router = _router(args)
    banner = """OKF Interactive Shell
========================================
Commands:
  import <file> [mode]       — import single OKF file (mode: text|optional|omni)
  import-bundle [path] [mode]— import entire bundle (mode: text|optional|omni)
  search [target:]<query>    — search concepts (default), chunks:, images:
  search <query> expand      — chunk hits + graph neighborhood
  search <query> hub         — chunk hits reranked by hub score
  read <id> [chunks|document|context] — read a concept (default: body)
  traverse [id] [rel] [dir] [depth] — traverse (no id = root listing)
  traverse <id1> <id2>       — shortest path between two concepts
  images <concept_id>        — list images attached to a concept
  export-bundle <output_dir> — export all concepts
  export <id> <output_dir>   — export single concept
  ingest <file> [--auto-import] — ingest .md or .pdf (auto-import PDFs)
  model-info                 — show model cache status
  help                       — show this help
  quit / exit                — exit shell
========================================"""

    print(banner)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Bye.")
            break

        elif cmd == "help":
            print(banner)

        elif cmd == "import" and rest:
            tokens = rest.strip().split()
            mode = "text"
            if tokens and tokens[-1].lower() in ("text", "optional", "omni"):
                mode = tokens[-1].lower()
                tokens = tokens[:-1]
            fp = Path(" ".join(tokens))
            if not fp.exists():
                print(f"Error: {fp} not found")
                continue
            cid = router.import_from_okf(fp, mode=mode)
            imgs = router.image_mgr.list_images(cid)
            suffix = f" ({len(imgs)} image(s), mode: {mode})" if imgs else ""
            print(f"[OK] Imported: {cid}{suffix}")

        elif cmd == "import-bundle":
            tokens = rest.strip().split()
            mode = "text"
            if tokens and tokens[-1].lower() in ("text", "optional", "omni"):
                mode = tokens[-1].lower()
                tokens = tokens[:-1]
            bundle_path = Path(" ".join(tokens)) if tokens else None
            ids = router.import_mgr.import_bundle(bundle_path, mode=mode)
            print(f"[OK] Imported {len(ids)} concepts (image mode: {mode})")

        elif cmd == "search" and rest:
            tokens = rest.strip().split()
            first = tokens[0]
            target = "concepts"
            if ":" in first and first.split(":")[0] in ("concepts", "chunks", "images"):
                target, first = first.split(":", 1)
                tokens[0] = first
            query = tokens[0]
            expand = "expand" in tokens[1:]
            hub = "hub" in tokens[1:]
            type_filter = tags_filter = parent_filter = None
            for t in tokens[1:]:
                if t.startswith("type:"):
                    type_filter = t[5:]
                elif t.startswith("tags:"):
                    tags_filter = t[5:].split(",")
                elif t.startswith("parent:"):
                    parent_filter = t[7:]
            if target == "images":
                results = router.image_mgr.search_images_with_text(query)
                for i, r in enumerate(results, 1):
                    label = r.get("alt_text") or r.get("file_name") or r.get("id")
                    print(f"  {i}. [{r['relevance_score']:.4f}] {label} ({r.get('embed_route')})")
                    print(f"     id: {r['id']}")
            elif target == "chunks":
                if hub:
                    results = router.search_engine.search_chunks_with_hub_score(query)
                elif expand:
                    results = router.search_engine.search_with_context(query)
                else:
                    results = router.search_engine.search_chunks(query)
                for i, r in enumerate(results, 1):
                    chunk = r.get("chunk", r)
                    score = r.get("final_score", chunk.get("rrf_score", 0))
                    print(f"  {i}. [{score:.4f}] {chunk.get('parent_title', '?')} §{chunk.get('chunk_index', '?')}")
                    print(f"     {chunk.get('chunk_text', '')[:150]}")
            else:
                results = router.search_hybrid(
                    query=query, concept_type=type_filter,
                    tags=tags_filter, parent_id=parent_filter,
                )
                for i, r in enumerate(results, 1):
                    print(f"  {i}. [{r['relevance_score']:.4f}] {r['title']} ({r['type']})")
                    desc = r.get("description") or ""
                    if desc:
                        print(f"     {desc[:120]}")

        elif cmd == "read" and rest:
            tokens = rest.strip().split()
            cid = tokens[0]
            include = tokens[1] if len(tokens) > 1 else "body"
            if include == "chunks":
                chunks = router.search_engine.get_chunks(cid)
                if not chunks:
                    print("No chunks found.")
                for c in chunks:
                    print(f"  #{c.chunk_index} [{c.block_type}] {c.chunk_text[:120]}")
            elif include == "document":
                text = router.embed_engine.reconstruct_document(cid)
                print(text if text else "No chunks found for this concept.")
            elif include == "context":
                for l in router.traverse(cid, "LINKS_TO", "INCOMING", 1)[:10]:
                    print(f"  ← {l.get('title', l.get('id', '?'))}")
                for l in router.traverse(cid, "LINKS_TO", "OUTGOING", 1)[:10]:
                    print(f"  → {l.get('title', l.get('id', '?'))}")
                for a in router.search_engine._get_ancestry(cid):
                    print(f"  ↑ {a['title']}")
            else:
                concept = router.get_by_id(cid)
                if concept:
                    data = concept.model_dump()
                    body = data.pop("body", "")
                    data.pop("embedding", None)
                    print(json.dumps(data, indent=2, default=str))
                    if body:
                        print(f"\n--- BODY ---\n{body}")
                else:
                    print(f"Concept '{cid}' not found")

        elif cmd == "traverse":
            tokens = rest.strip().split()
            if not tokens:
                for item in router.list_directory(""):
                    icon = "[D]" if item["type"] == "Directory" else "[F]"
                    print(f"  {icon} {item['title']} ({item['type']})")
            elif len(tokens) == 2 and tokens[1] not in ("CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"):
                nodes = router.search_engine.find_path(tokens[0], tokens[1])
                if not nodes:
                    print(f"No path found between '{tokens[0]}' and '{tokens[1]}'.")
                else:
                    print(f"Path ({len(nodes)} nodes):")
                    for i, n in enumerate(nodes, 1):
                        print(f"  {i}. {n.get('title', '?')} ({n.get('type', '?')})")
                        print(f"     id: {n['id']}")
            else:
                start_id = tokens[0]
                rel = tokens[1] if len(tokens) > 1 else "CONTAINS"
                direction = tokens[2] if len(tokens) > 2 else "OUTGOING"
                depth = int(tokens[3]) if len(tokens) > 3 else 1
                results = router.traverse(start_id, rel, direction, depth)
                for r in results:
                    print(f"  {r['id']} ({r['type']}) — {r.get('title', '')}")

        elif cmd == "images" and rest:
            imgs = router.image_mgr.list_images(rest.strip())
            if not imgs:
                print("No images attached.")
            for im in imgs:
                alt = im.get("alt_text") or "(no alt-text)"
                print(f"  [{im.get('embed_route')}] {im.get('file_name')} — {alt}")
                print(f"     id: {im.get('id')}")

        elif cmd == "export-bundle" and rest:
            ids = router.export_mgr.export_bundle(Path(rest.strip()))
            print(f"[OK] Exported {len(ids)} concepts to {rest.strip()}")

        elif cmd == "export" and rest:
            tokens = rest.strip().split(None, 1)
            if len(tokens) == 2:
                cid, out_dir = tokens
                output_path = Path(out_dir) / f"{cid}.md"
                router.export_to_okf(cid, output_path)
                print(f"[OK] Exported {cid} → {output_path}")
            else:
                print("Usage: export <concept_id> <output_dir>")

        elif cmd == "model-info":
            info = OKFRouter.model_info(cache_dir=router.cache_dir)
            print(f"Model:  {info['model_id']}")
            print(f"Cache:  {info['cache_dir']}")
            if info["cached"]:
                size_gb = info["disk_usage_bytes"] / (1024 ** 3)
                print(f"Status: cached ({size_gb:.2f} GB)")
                print(f"Path:   {info['snapshot_path']}")
            else:
                print("Status: not cached (will download on first use)")

        elif cmd == "broken-links":
            broken = router.list_broken_links()
            if not broken:
                print("No broken links found.")
            else:
                print(f"Found {len(broken)} broken link(s):")
                for link in broken:
                    print(f"  {link['source']} → {link['target']}")

        elif cmd == "repair-links":
            count = router.repair_links()
            print(f"[OK] Repaired {count} link(s)")

        elif cmd == "ingest" and rest:
            # Minimal shell dispatch for ingest — delegates to the CLI handler.
            from okfgraph.cli import _ingest
            from types import SimpleNamespace
            src_path = rest.strip()
            is_md = src_path.lower().endswith(".md")
            shell_args = SimpleNamespace(
                kind="md" if is_md else "pdf",
                md_file=src_path if is_md else None,
                pdf_file=None if is_md else src_path,
                thoughts=None,
                topic=None,
                concept_id=None,
                title=None,
                description=None,
                tags=None,
                auto_import=False,
                output=None,
                routing_mode="auto",
                mode="text",
                batch_size=32,
                purge=False,
                no_extract_images=False,
                db=args.db,
                bundle=args.bundle,
                dim=args.dim,
                cache_dir=getattr(args, "cache_dir", None),
                device=getattr(args, "device", "cpu"),
                omni_model_id=getattr(args, "omni_model_id", None),
                chunk_size=getattr(args, "chunk_size", 512),
                chunk_overlap=getattr(args, "chunk_overlap", 40),
                no_chunking=False,
                allow_remote_images=False,
            )
            _ingest(shell_args)

        else:
            print(f"Unknown command: {cmd}. Type 'help' for usage.")


# ── argument parser ────────────────────────────────────────────────────────

def _global_options_epilog() -> str:
    """Render the global flags once for top-level ``okf --help``.

    Single source of truth: builds a throwaway parser with the same helpers
    (unmarked, so nothing is hidden) and reuses its options section.
    """
    probe = argparse.ArgumentParser(prog="okf")
    _add_global(probe, mark=False)
    _add_logging_flags(probe, mark=False)
    text = probe.format_help()
    try:
        body = text.split("options:", 1)[1]
    except IndexError:  # pragma: no cover - Python <3.11 wording
        body = text.split("optional arguments:", 1)[1]
    return "Global options (every command; may also come from okfgraph.toml):" + body


def build_parser():
    parser = argparse.ArgumentParser(
        prog="okf",
        description="OKF Knowledge Graph CLI — LadybugDB + Jina v5 embeddings",
        epilog=_global_options_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run", parser_class=_SubParser)

    # init
    p = sub.add_parser("init", help="Initialize database and schema")
    _add_global(p)
    _add_logging_flags(p)

    # model-info
    p = sub.add_parser("model-info", help="Show model cache status")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--model-id", default="jinaai/jina-embeddings-v5-text-small-retrieval", help="Model ID to inspect")

    # import
    p = sub.add_parser("import", help="Import OKF files")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("files", nargs="*", help="Files to import")
    p.add_argument("--all", action="store_true", dest="import_all", help="Import entire bundle")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding (default: 32)")
    p.add_argument(
        "--mode", default="text", choices=["text", "optional", "omni"],
        help="Image ingestion mode: text (alt-text/filename, no omni), "
             "optional (omni only for images lacking alt-text), "
             "omni (omni for every image). Default: text",
    )
    p.add_argument(
        "--purge", action="store_true", default=False,
        help="Also purge concepts whose source files were deleted from disk "
             "(removes concept, chunks, links, and orphaned image assets)",
    )

    # search (unified: concepts, chunks, images)
    p = sub.add_parser("search", help="Search concepts, chunks, or images")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--target", default="concepts", choices=["concepts", "chunks", "images"],
                   help="What to search (default: concepts)")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument("--type", help="Concept type filter (concepts/chunks)")
    p.add_argument("--tags", help="Comma-separated tag filters (concepts/chunks)")
    p.add_argument("--parent", help="Parent directory ID (concepts/chunks)")
    p.add_argument("--chunks", action="store_true", help="Include matched chunks per concept result")
    p.add_argument("--expand", action="store_true", help="Chunks: attach graph neighborhood to each hit")
    p.add_argument("--context-hops", type=int, default=1, help="Expansion hops with --expand (default: 1)")
    p.add_argument("--hub-rerank", action="store_true", help="Chunks: rerank by graph hub score")
    p.add_argument("--hub-weight", type=float, default=0.3, help="Hub weight with --hub-rerank (default: 0.3)")
    p.add_argument("--use-omni", action="store_true", help="Images: encode query with omni text side")

    # read (unified: body, chunks, document, context)
    p = sub.add_parser("read", help="Read a concept: body, chunks, document, or context")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")
    p.add_argument("--include", default="body", choices=["body", "chunks", "document", "context"],
                   help="What to return (default: body)")
    p.add_argument("--output", help="Output file for --include document (default: stdout)")

    # traverse (unified: relationships, directory listing, shortest path)
    p = sub.add_parser("traverse", help="Traverse relationships, list directories, find paths")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("start_id", nargs="?", default="", help="Starting concept or directory ID (empty = root listing)")
    p.add_argument("--relationship", default="CONTAINS", choices=["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"])
    p.add_argument("--direction", default="OUTGOING", choices=["OUTGOING", "INCOMING", "BOTH"])
    p.add_argument("--depth", type=int, default=1, help="Max depth (1-5)")
    p.add_argument("--type", help="Target node type filter")
    p.add_argument("--target", default=None, help="Find shortest path from start_id to this ID instead of traversing")
    p.add_argument("--max-path-length", type=int, default=6, help="Max path length with --target (default: 6)")

    # ingest (unified: markdown, PDF, thoughts)
    p = sub.add_parser("ingest", help="Add content: markdown file, PDF, or thoughts")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--kind", default="pdf", choices=["md", "pdf", "thoughts"],
                   help="What to ingest (default: pdf)")
    p.add_argument("--md-file", default=None, help="Markdown file (--kind md)")
    p.add_argument("--pdf-file", default=None, help="PDF file (--kind pdf)")
    p.add_argument("--thoughts", default=None, help="Raw reasoning text (--kind thoughts)")
    p.add_argument("--topic", default=None, help="Topic (--kind thoughts)")
    p.add_argument("--concept-id", default=None, help="Explicit concept ID (--kind md/thoughts)")
    p.add_argument("--title", default=None, help="Title override (--kind md)")
    p.add_argument("--description", default=None, help="Description override (--kind md)")
    p.add_argument("--tags", default=None, help="Comma-separated tags (--kind md/thoughts)")
    p.add_argument(
        "--auto-import", action="store_true",
        help="Auto-import converted markdown into the graph (--kind pdf)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory for converted markdown (default: current dir)",
    )
    p.add_argument(
        "--routing-mode", default="auto",
        choices=["auto", "surgical", "always", "never"],
        help="ONNX routing mode (--kind pdf, default: auto)",
    )
    p.add_argument(
        "--mode", default="text", choices=["text", "optional", "omni"],
        help="Image ingestion mode (default: text)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for encoding during auto-import (default: 32)",
    )
    p.add_argument(
        "--purge", action="store_true", default=False,
        help="Purge deleted concepts during auto-import",
    )
    p.add_argument(
        "--no-extract-images", action="store_true",
        help="Do not extract embedded images from the PDF",
    )

    # export
    p = sub.add_parser("export", help="Export concepts")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--all", action="store_true", dest="export_all", help="Export entire bundle")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--concept-id", help="Concept ID (for single export)")
    p.add_argument("--type", help="Concept type filter")
    p.add_argument("--tags", help="Comma-separated tag filters")
    p.add_argument("--parent", help="Parent directory ID")

    # shell
    p = sub.add_parser("shell", help="Interactive REPL")
    _add_global(p)
    _add_logging_flags(p)

    # broken-links
    p = sub.add_parser("broken-links", help="List broken (orphan) links")
    _add_global(p)
    _add_logging_flags(p)

    # repair-links
    p = sub.add_parser("repair-links", help="Repair broken links by re-checking targets")
    _add_global(p)
    _add_logging_flags(p)

    # reindex
    p = sub.add_parser("reindex", help="Rebuild vector + FTS search indexes")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--if-dirty", action="store_true",
                   help="Only rebuild if data changed since the last index build")

    # Soft-delete commands (Gap #1d)
    p = sub.add_parser("deleted-list", help="List soft-deleted concepts")
    _add_global(p)
    _add_logging_flags(p)

    p = sub.add_parser("deleted-recover", help="Recover a soft-deleted concept")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID to recover")

    p = sub.add_parser("deleted-purge", help="Permanently delete expired soft-deleted concepts")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--older-than", type=int, default=None, help="Override recovery window (seconds)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Setup logging (Gap #10)
    _setup_logging(
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
        log_file=getattr(args, "log_file", ""),
    )

    logger = logging.getLogger("cli")

    # Profile flag (Gap #10C)
    if getattr(args, "profile", False):
        logger.info("profiling enabled")
        profiler = cProfile.Profile()
        profiler.enable()

    commands = {
        "init": _init,
        "model-info": _model_info,
        "import": _import,
        "ingest": _ingest,
        "search": _search,
        "read": _read,
        "traverse": _traverse,
        "export": _export,
        "shell": _shell,
        "broken-links": _broken_links,
        "repair-links": _repair_links,
        "reindex": _reindex,
        "deleted-list": _deleted_list,
        "deleted-recover": _deleted_recover,
        "deleted-purge": _deleted_purge,
    }

    try:
        commands[args.command](args)
    finally:
        _close_routers()
        _teardown_logging()

    # Profile output (Gap #10C)
    if getattr(args, "profile", False):
        profiler.disable()
        import pstats
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)
        print(stream.getvalue(), file=sys.stderr)


if __name__ == "__main__":
    main()
