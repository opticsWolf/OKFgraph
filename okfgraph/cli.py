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

def _add_global(parser):
    """Add --db / --bundle / --dim / --cache-dir / --device to any subparser."""
    parser.add_argument("--db", default=None, help="Database path (default: okfgraph.db, or from okfgraph.toml)")
    parser.add_argument("--bundle", default=None, help="Bundle root directory (default: ., or from okfgraph.toml)")
    parser.add_argument("--dim", type=int, default=None, help="Embedding dimension (Matryoshka; default: 512, or from okfgraph.toml)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model cache directory (default: ~/.cache/huggingface, or from okfgraph.toml)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Inference device: cpu or cuda (default: cpu, or from okfgraph.toml)")
    parser.add_argument("--omni-model-id", default=None, help="Multimodal model ID for image embeddings (default from okfgraph.toml)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in words for overlap (default: 512, or from okfgraph.toml)")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Overlap in words between chunks (default: 40, or from okfgraph.toml)")
    parser.add_argument("--no-chunking", action="store_true", help="Disable chunking during ingestion")
    parser.add_argument("--wal-mode", action="store_true", help="Enable SQLite WAL mode for concurrent reads (Gap #7a)")
    parser.add_argument("--allow-remote-images", action="store_true", help="Allow fetching remote images (SSRF risk — use with caution)")
    parser.add_argument("--allowed-image-domains", default=None, help="Comma-separated list of allowed domains for remote images (Gap #9a)")


def _add_logging_flags(parser):
    """Add --verbose / --quiet / --log-file / --profile to a subparser."""
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all logging except errors")
    parser.add_argument("--log-file", default="", help="Write logs to file (with 5MB rotation)")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile for the current invocation (outputs to stdout)")


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


def _search_images(args):
    router = _router(args)
    results = router.image_mgr.search_images_with_text(
        text_query=args.query,
        use_text_model=not getattr(args, "use_omni", False),
        limit=args.limit,
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


def _search(args):
    router = _router(args)
    tags = args.tags.split(",") if args.tags else None
    results = router.search_hybrid(
        query=args.query,
        concept_type=args.type,
        tags=tags,
        parent_id=args.parent,
        limit=args.limit,
        include_chunks=getattr(args, "chunks", False),
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
    router = _router(args)
    results = router.traverse(
        start_id=args.start_id,
        relationship=args.relationship,
        direction=args.direction,
        depth=args.depth,
        node_type=args.type,
    )
    if not results:
        print("No results found.")
        return
    print(f"Found {len(results)} node(s):\n")
    for r in results:
        print(f"  {r['id']} ({r['type']})")
        if r.get("title"):
            print(f"    title: {r['title']}")


def _list(args):
    router = _router(args)
    items = router.list_directory(args.directory)
    if not items:
        print("Directory is empty.")
        return
    print(f"Contents of '{args.directory or '(root)'}':\n")
    for item in items:
        icon = "[D]" if item["type"] == "Directory" else "[F]"
        print(f"  {icon} {item['title']} ({item['type']})")
        print(f"     id: {item['id']}")


def _get(args):
    router = _router(args)
    concept = router.get_by_id(args.concept_id)
    if not concept:
        print(f"Concept '{args.concept_id}' not found.")
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


def _ingest_pdf(args):
    """Convert a PDF to markdown and optionally import into the graph.

    Uses the HybridConverter pipeline (pdf_oxide fast path + ONNX/Rapid
    heavy passes). When ``--auto-import`` is set the produced markdown is
    imported into the graph via ``import_bundle()`` and the temporary
    directory is cleaned up automatically.
    """
    from tempfile import TemporaryDirectory

    from okfgraph.ingest import ConverterConfig, HybridConverter, RoutingMode
    from okfgraph.ingest.assets import stage_images_as_okf_assets

    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        return

    # Build converter config from CLI args
    mode_str = getattr(args, "routing_mode", "auto").lower()
    routing = RoutingMode(mode_str)
    device = getattr(args, "device", "cuda")

    config = ConverterConfig(
        routing_mode=routing,
        device=device,
        extract_images=getattr(args, "extract_images", True),
    )

    converter = HybridConverter(config)

    try:
        # Ensure ONNX models are loaded (if routing mode requires them)
        converter.ensure_models()

        if getattr(args, "auto_import", False):
            # Auto-import: convert to temp dir, import into graph, clean up.
            router = _router(args)

            with TemporaryDirectory(prefix="okf_ingest_") as tmp:
                work_dir = Path(tmp)
                logger = logging.getLogger("cli")
                logger.info("converting %s → %s", pdf_path, work_dir)

                md = converter.convert_pdf(
                    path=pdf_path,
                    work_dir=work_dir,
                    should_continue=lambda: True,
                    on_page=lambda idx, total: print(
                        f"  page {idx + 1}/{total}", end="\r"
                    ),
                )
                print()  # newline after progress

                # Write markdown to temp dir
                stem = pdf_path.stem
                md_path = work_dir / f"{stem}.md"
                md_path.write_text(md, encoding="utf-8")

                # Stage any extracted images as okf-asset:// URIs
                img_dir = work_dir / "_assets"
                img_dir.mkdir(exist_ok=True)
                for img in work_dir.glob("*"):
                    if img.is_file() and img.suffix.lower() in (
                        ".png", ".jpg", ".jpeg", ".gif", ".webp",
                    ):
                        img.rename(img_dir / img.name)

                md_text = md_path.read_text(encoding="utf-8")
                md_text, img_count = stage_images_as_okf_assets(
                    md_text, img_dir, pdf_path, work_dir, stem
                )
                md_path.write_text(md_text, encoding="utf-8")

                # Lint converted markdown (Gap #5c)
                lint_result = router.ingest_mgr._lint_converted_md(md_path, auto_fix=True)
                if lint_result["fixed"]:
                    md_path.write_text(lint_result["content"], encoding="utf-8")
                    logger.info(
                        "linted %s: fixed %d issues",
                        md_path.name,
                        lint_result["fixed_count"],
                    )
                if lint_result["errors"]:
                    logger.warning(
                        "PDF output has %d structural errors — proceeding anyway",
                        len(lint_result["errors"]),
                    )

                # Import into the graph
                mode = getattr(args, "mode", "text")
                purge = getattr(args, "purge", False)
                ids = router.import_mgr.import_bundle(
                    work_dir,
                    batch_size=getattr(args, "batch_size", 32) or 32,
                    mode=mode,
                    purge_deleted=purge,
                )
                logger.info("imported %d concept(s) from %s", len(ids), pdf_path)
                for cid in ids:
                    n = len(router.image_mgr.list_images(cid))
                    suffix = f"  [{n} image(s)]" if n else ""
                    logger.info("  %s%s", cid, suffix)
        else:
            # Output-only: convert to the specified output directory.
            logger = logging.getLogger("cli")
            output_dir = Path(args.output) if args.output else Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("converting %s → %s", pdf_path, output_dir)

            md = converter.convert_pdf(
                path=pdf_path,
                work_dir=output_dir,
                should_continue=lambda: True,
                on_page=lambda idx, total: print(
                    f"  page {idx + 1}/{total}", end="\r"
                ),
            )
            print()

            # Write markdown
            stem = pdf_path.stem
            md_path = output_dir / f"{stem}.md"
            md_path.write_text(md, encoding="utf-8")

            # Stage images
            img_dir = output_dir / "_assets"
            img_dir.mkdir(exist_ok=True)
            for img in output_dir.glob("*"):
                if img.is_file() and img.suffix.lower() in (
                    ".png", ".jpg", ".jpeg", ".gif", ".webp",
                ):
                    img.rename(img_dir / img.name)

            md_text = md_path.read_text(encoding="utf-8")
            md_text, img_count = stage_images_as_okf_assets(
                md_text, img_dir, pdf_path, output_dir, stem
            )
            md_path.write_text(md_text, encoding="utf-8")

            logger.info("written %s", md_path)
            logger.info("assets in %s", img_dir)
            logger.info("run 'okf import --all --bundle %s' to import.", output_dir)

    finally:
        converter.close()


def _search_chunks(args):
    router = _router(args)
    results = router.search_engine.search_chunks(
        query=args.query,
        limit=args.limit,
        include_parent=not getattr(args, "no-parent", False),
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


def _chunks(args):
    router = _router(args)
    chunks = router.search_engine.get_chunks(args.concept_id)
    if not chunks:
        print("No chunks found for this concept.")
        return
    print(f"Chunks for '{args.concept_id}' ({len(chunks)} total):\n")
    for c in chunks:
        text = c.chunk_text[:120]
        print(f"  #{c.chunk_index} [{c.block_type}] {text}")


def _context(args):
    router = _router(args)
    results = router.search_engine.search_with_context(
        query=args.query,
        limit=args.limit,
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


def _hub_search(args):
    router = _router(args)
    results = router.search_engine.search_chunks_with_hub_score(
        query=args.query,
        limit=args.limit,
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


def _siblings(args):
    router = _router(args)
    sibs = router.search_engine._get_siblings(args.concept_id)
    if not sibs:
        print("No siblings found.")
        return
    print(f"Siblings of '{args.concept_id}':\n")
    for s in sibs:
        print(f"  {s['title']} ({s['type']})")
        print(f"     id: {s['id']}")


def _ancestry(args):
    router = _router(args)
    path = router.search_engine._get_ancestry(args.concept_id)
    if not path:
        print(f"No directory ancestry for '{args.concept_id}'.")
        return
    print(f"Path for '{args.concept_id}':\n")
    for a in path:
        print(f"  {a['title']} (id: {a['id']})")


def _reconstruct(args):
    router = _router(args)
    text = router.embed_engine.reconstruct_document(args.concept_id)
    if not text:
        print("No chunks found for this concept.")
        return
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[OK] Reconstructed document written to {args.output}")
    else:
        print(text)


def _path(args):
    router = _router(args)
    nodes = router.search_engine.find_path(args.id1, args.id2, max_length=getattr(args, "max_length", 6))
    if not nodes:
        print(f"No path found between '{args.id1}' and '{args.id2}'.")
        return
    print(f"Path ({len(nodes)} nodes):")
    for i, n in enumerate(nodes, 1):
        print(f"  {i}. {n.get('title', '?')} ({n.get('type', '?')})")
        print(f"     id: {n['id']}")


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
  search <query>             — hybrid search (document-level)
  search <query> type:<type> — search with type filter
  search <query> tags:a,b    — search with tag filters
  search <query> parent:<id> — search under directory
  search-chunks <query>      — search document chunks (RRF-fused)
  context <query>            — search with graph neighborhood context
  hub-search <query>         — chunk search reranked by hub score
  path <id1> <id2>           — find shortest path between two concepts
  siblings <concept_id>      — list sibling concepts in same directory
  ancestry <concept_id>      — show directory path from root
  reconstruct <concept_id>   — reconstruct original document from chunks
  search-images <query>      — find images via the unified vector index
  images <concept_id>        — list images attached to a concept
  traverse <id> [rel] [dir] [depth] — graph traversal
  list [directory_id]        — list directory contents
  get <concept_id>           — fetch full concept
  export-bundle <output_dir> — export all concepts
  export <id> <output_dir>   — export single concept
  ingest <pdf> [--auto-import] — convert PDF to markdown, optionally import
  model-info                 — show model cache status
  help                       — show this help
  quit / exit                — exit shell

Image modes:
  text      alt-text, or filename+image-number when absent (no omni model)
  optional  omni only for images that have no alt-text
  omni      every image embedded by the omni multimodal model
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
            query = tokens[0]
            type_filter = tags_filter = parent_filter = None
            for t in tokens[1:]:
                if t.startswith("type:"):
                    type_filter = t[5:]
                elif t.startswith("tags:"):
                    tags_filter = t[5:].split(",")
                elif t.startswith("parent:"):
                    parent_filter = t[7:]
            results = router.search_hybrid(
                query=query, concept_type=type_filter,
                tags=tags_filter, parent_id=parent_filter,
            )
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['relevance_score']:.4f}] {r['title']} ({r['type']})")
                desc = r.get("description") or ""
                if desc:
                    print(f"     {desc[:120]}")

        elif cmd == "search-images" and rest:
            results = router.image_mgr.search_images_with_text(rest.strip())
            if not results:
                print("No image results found.")
            for i, r in enumerate(results, 1):
                label = r.get("alt_text") or r.get("file_name") or r.get("id")
                print(f"  {i}. [{r['relevance_score']:.4f}] {label} ({r.get('embed_route')})")
                print(f"     id: {r['id']}")

        elif cmd == "search-chunks" and rest:
            results = router.search_engine.search_chunks(rest.strip())
            if not results:
                print("No chunk results found.")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['rrf_score']:.4f}] {r['block_type']} #{r['chunk_index']}")
                print(f"     {r.get('chunk_text', '')[:150]}")
                if r.get("parent_title"):
                    print(f"     parent: {r['parent_title']}")

        elif cmd == "context" and rest:
            results = router.search_engine.search_with_context(rest.strip())
            if not results:
                print("No results found.")
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

        elif cmd == "hub-search" and rest:
            results = router.search_engine.search_chunks_with_hub_score(rest.strip())
            if not results:
                print("No results found.")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['final_score']:.4f}] {r['parent_title']} §{r['chunk_index']}")
                print(f"     hub={r['hub_score']:.2f} rrf={r['rrf_score']:.2f}")
                print(f"     {r['chunk_text'][:150]}")

        elif cmd == "path" and rest:
            tokens = rest.strip().split()
            if len(tokens) < 2:
                print("Usage: path <id1> <id2>")
            else:
                nodes = router.search_engine.find_path(tokens[0], tokens[1])
                if not nodes:
                    print(f"No path found between '{tokens[0]}' and '{tokens[1]}'.")
                else:
                    print(f"Path ({len(nodes)} nodes):")
                    for i, n in enumerate(nodes, 1):
                        print(f"  {i}. {n.get('title', '?')} ({n.get('type', '?')})")
                        print(f"     id: {n['id']}")

        elif cmd == "siblings" and rest:
            sibs = router.search_engine._get_siblings(rest.strip())
            if not sibs:
                print("No siblings found.")
            for s in sibs:
                print(f"  {s['title']} ({s['type']})")

        elif cmd == "ancestry" and rest:
            path = router.search_engine._get_ancestry(rest.strip())
            if not path:
                print("No directory ancestry.")
            for a in path:
                print(f"  {a['title']}")

        elif cmd == "chunks" and rest:
            chunks = router.search_engine.get_chunks(rest.strip())
            if not chunks:
                print("No chunks found.")
            for c in chunks:
                text = c.chunk_text[:120]
                print(f"  #{c.chunk_index} [{c.block_type}] {text}")

        elif cmd == "reconstruct" and rest:
            text = router.embed_engine.reconstruct_document(rest.strip())
            if not text:
                print("No chunks found for this concept.")
            else:
                print(text)

        elif cmd == "images" and rest:
            imgs = router.image_mgr.list_images(rest.strip())
            if not imgs:
                print("No images attached.")
            for im in imgs:
                alt = im.get("alt_text") or "(no alt-text)"
                print(f"  [{im.get('embed_route')}] {im.get('file_name')} — {alt}")
                print(f"     id: {im.get('id')}")

        elif cmd == "traverse" and rest:
            tokens = rest.strip().split()
            start_id = tokens[0]
            rel = tokens[1] if len(tokens) > 1 else "CONTAINS"
            direction = tokens[2] if len(tokens) > 2 else "OUTGOING"
            depth = int(tokens[3]) if len(tokens) > 3 else 1
            results = router.traverse(start_id, rel, direction, depth)
            for r in results:
                print(f"  {r['id']} ({r['type']}) — {r.get('title', '')}")

        elif cmd == "list":
            dir_id = rest.strip() if rest.strip() else ""
            items = router.list_directory(dir_id)
            for item in items:
                icon = "[D]" if item["type"] == "Directory" else "[F]"
                print(f"  {icon} {item['title']} ({item['type']})")

        elif cmd == "get" and rest:
            concept = router.get_by_id(rest.strip())
            if concept:
                data = concept.model_dump()
                body = data.pop("body", "")
                data.pop("embedding", None)
                print(json.dumps(data, indent=2, default=str))
                if body:
                    print(f"\n--- BODY ---\n{body}")
            else:
                print(f"Concept '{rest.strip()}' not found")

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
            from okfgraph.cli import _ingest_pdf
            from types import SimpleNamespace
            pdf_path = rest.strip()
            shell_args = SimpleNamespace(
                pdf_file=pdf_path,
                auto_import=False,
                output=None,
                routing_mode="auto",
                mode="text",
                batch_size=32,
                purge=False,
                extract_images=True,
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
            _ingest_pdf(shell_args)

        else:
            print(f"Unknown command: {cmd}. Type 'help' for usage.")


# ── argument parser ────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="okf",
        description="OKF Knowledge Graph CLI — LadybugDB + Jina v5 embeddings",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

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

    # search-images
    p = sub.add_parser("search-images", help="Find images from a text query (unified index)")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("query", help="Text query")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument(
        "--use-omni", action="store_true",
        help="Encode the query with the omni text side instead of the text model",
    )

    # search
    p = sub.add_parser("search", help="Hybrid search")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--type", help="Concept type filter")
    p.add_argument("--tags", help="Comma-separated tag filters")
    p.add_argument("--parent", help="Parent directory ID")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument("--chunks", action="store_true", help="Include matched chunks per result")

    # traverse
    p = sub.add_parser("traverse", help="Graph traversal")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("start_id", help="Starting concept or directory ID")
    p.add_argument("--relationship", default="CONTAINS", choices=["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"])
    p.add_argument("--direction", default="OUTGOING", choices=["OUTGOING", "INCOMING", "BOTH"])
    p.add_argument("--depth", type=int, default=1, help="Max depth (1-5)")
    p.add_argument("--type", help="Target node type filter")

    # list
    p = sub.add_parser("list", help="List directory contents")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("directory", nargs="?", default="", help="Directory ID (empty for root)")

    # get
    p = sub.add_parser("get", help="Get concept by ID")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")

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

    # ingest (PDF → markdown → optional import)
    p = sub.add_parser("ingest", help="Convert PDF to markdown and optionally import")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("pdf_file", help="Path to the PDF file")
    p.add_argument(
        "--auto-import", action="store_true",
        help="Auto-import the converted markdown into the graph",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory for converted markdown (default: current dir)",
    )
    p.add_argument(
        "--routing-mode", default="auto",
        choices=["auto", "surgical", "always", "never"],
        help="ONNX routing mode (default: auto)",
    )
    p.add_argument(
        "--mode", default="text", choices=["text", "optional", "omni"],
        help="Image ingestion mode for auto-import (default: text)",
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

    # search-chunks
    p = sub.add_parser("search-chunks", help="Search document chunks (RRF-fused vector + FTS)")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument("--no-parent", action="store_true", help="Exclude parent document metadata")

    # chunks
    p = sub.add_parser("chunks", help="List chunks for a concept")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")

    # context
    p = sub.add_parser("context", help="Search with graph neighborhood context")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--limit", type=int, default=5, help="Max results (default: 5)")
    p.add_argument("--context-hops", type=int, default=1, help="Graph expansion hops (default: 1)")

    # hub-search
    p = sub.add_parser("hub-search", help="Chunk search reranked by hub score")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument("--hub-weight", type=float, default=0.3, help="Hub score weight (default: 0.3)")

    # siblings
    p = sub.add_parser("siblings", help="List sibling concepts in same directory")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")

    # ancestry
    p = sub.add_parser("ancestry", help="Show directory path from root")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")

    # reconstruct
    p = sub.add_parser("reconstruct", help="Reconstruct original document from chunks")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")
    p.add_argument("--output", help="Output file path (default: stdout)")

    # path
    p = sub.add_parser("path", help="Find shortest path between two concepts")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("id1", help="Starting concept ID")
    p.add_argument("id2", help="Ending concept ID")
    p.add_argument("--max-length", type=int, default=6, help="Max path length (default: 6)")

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
        "ingest": _ingest_pdf,
        "search": _search,
        "search-images": _search_images,
        "search-chunks": _search_chunks,
        "context": _context,
        "hub-search": _hub_search,
        "traverse": _traverse,
        "list": _list,
        "get": _get,
        "chunks": _chunks,
        "export": _export,
        "reconstruct": _reconstruct,
        "path": _path,
        "shell": _shell,
        "broken-links": _broken_links,
        "repair-links": _repair_links,
        "reindex": _reindex,
        "siblings": _siblings,
        "ancestry": _ancestry,
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
