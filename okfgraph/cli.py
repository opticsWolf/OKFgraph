"""OKF CLI — Command-line interface for the OKF knowledge graph."""

import argparse
import json
import sys
from pathlib import Path

from okfgraph.router import OKFRouter


# ── helpers ────────────────────────────────────────────────────────────────

def _add_global(parser):
    """Add --db / --bundle / --dim / --cache-dir / --device to any subparser."""
    parser.add_argument("--db", default="okfgraph.db", help="Database path (default: okfgraph.db)")
    parser.add_argument("--bundle", default=".", help="Bundle root directory (default: .)")
    parser.add_argument("--dim", type=int, default=512, help="Embedding dimension (Matryoshka; default: 512)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace model cache directory (default: ~/.cache/huggingface)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device: cpu or cuda (default: cpu)")
    parser.add_argument("--omni-model-id", default="jinaai/jina-embeddings-v5-omni-small-retrieval", help="Multimodal model ID for image embeddings")


def _router(args):
    """Build an OKFRouter from parsed args."""
    return OKFRouter(
        db_path=str(args.db),
        bundle_root=str(args.bundle),
        embedding_dim=args.dim,
        omni_model_id=getattr(args, "omni_model_id", "jinaai/jina-embeddings-v5-omni-small-retrieval"),
        cache_dir=getattr(args, "cache_dir", None),
        device=getattr(args, "device", "cpu"),
        allow_remote_images=getattr(args, "allow_remote_images", False),
    )


# ── command handlers ───────────────────────────────────────────────────────

def _init(args):
    db_path = str(args.db)
    print(f"Initializing database at {db_path} (dim={args.dim})...")
    _router(args)
    print(f"[OK] Database initialized (embedding_dim={args.dim})")


def _model_info(args):
    """Show model cache status without loading the model."""
    info = OKFRouter.model_info(
        model_id=getattr(args, "model_id", "jinaai/jina-embeddings-v5-text-small-retrieval"),
        cache_dir=getattr(args, "cache_dir", None),
    )
    print(f"Model:  {info['model_id']}")
    print(f"Cache:  {info['cache_dir']}")
    if info["cached"]:
        print(f"Status: cached")
        print(f"Path:   {info['snapshot_path']}")
        size_gb = info["disk_usage_bytes"] / (1024 ** 3)
        print(f"Size:   {size_gb:.2f} GB")
    else:
        default_cache = OKFRouter.default_cache_dir()
        print(f"Status: not cached (will download on first use)")
        print(f"Will use: {default_cache}")


def _import(args):
    router = _router(args)
    mode = getattr(args, "mode", "text")
    if getattr(args, "import_all", False):
        bundle_path = Path(args.bundle) if args.bundle else None
        ids = router.import_bundle(
            bundle_path, batch_size=getattr(args, "batch_size", 32) or 32, mode=mode
        )
        print(f"[OK] Imported {len(ids)} concepts (image mode: {mode})")
        for cid in ids:
            n = len(router.list_images(cid))
            suffix = f"  [{n} image(s)]" if n else ""
            print(f"  {cid}{suffix}")
    else:
        for fp in args.files:
            path = Path(fp)
            if not path.exists():
                print(f"[WARN] Skipping {fp}: file not found")
                continue
            cid = router.import_from_okf(path, mode=mode)
            imgs = router.list_images(cid)
            suffix = f" ({len(imgs)} image(s), mode: {mode})" if imgs else ""
            print(f"[OK] Imported: {cid}{suffix}")


def _search_images(args):
    router = _router(args)
    results = router.search_images_with_text(
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
        ids = router.export_bundle(
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
    router = _router(args)
    broken = router.list_broken_links()
    if not broken:
        print("No broken links found.")
        return
    print(f"Found {len(broken)} broken link(s):\n")
    for link in broken:
        print(f"  {link['source']} → {link['target']}")


def _repair_links(args):
    router = _router(args)
    count = router.repair_links()
    print(f"[OK] Repaired {count} link(s)")


def _shell(args):
    router = _router(args)
    banner = """OKF Interactive Shell
========================================
Commands:
  import <file> [mode]       — import single OKF file (mode: text|optional|omni)
  import-bundle [path] [mode]— import entire bundle (mode: text|optional|omni)
  search <query>             — hybrid search
  search <query> type:<type> — search with type filter
  search <query> tags:a,b    — search with tag filters
  search <query> parent:<id> — search under directory
  search-images <query>      — find images via the unified vector index
  images <concept_id>        — list images attached to a concept
  traverse <id> [rel] [dir] [depth] — graph traversal
  list [directory_id]        — list directory contents
  get <concept_id>           — fetch full concept
  export-bundle <output_dir> — export all concepts
  export <id> <output_dir>   — export single concept
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
            imgs = router.list_images(cid)
            suffix = f" ({len(imgs)} image(s), mode: {mode})" if imgs else ""
            print(f"[OK] Imported: {cid}{suffix}")

        elif cmd == "import-bundle":
            tokens = rest.strip().split()
            mode = "text"
            if tokens and tokens[-1].lower() in ("text", "optional", "omni"):
                mode = tokens[-1].lower()
                tokens = tokens[:-1]
            bundle_path = Path(" ".join(tokens)) if tokens else None
            ids = router.import_bundle(bundle_path, mode=mode)
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
            results = router.search_images_with_text(rest.strip())
            if not results:
                print("No image results found.")
            for i, r in enumerate(results, 1):
                label = r.get("alt_text") or r.get("file_name") or r.get("id")
                print(f"  {i}. [{r['relevance_score']:.4f}] {label} ({r.get('embed_route')})")
                print(f"     id: {r['id']}")

        elif cmd == "images" and rest:
            imgs = router.list_images(rest.strip())
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
            ids = router.export_bundle(Path(rest.strip()))
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

    # model-info
    p = sub.add_parser("model-info", help="Show model cache status")
    _add_global(p)
    p.add_argument("--model-id", default="jinaai/jina-embeddings-v5-text-small-retrieval", help="Model ID to inspect")

    # import
    p = sub.add_parser("import", help="Import OKF files")
    _add_global(p)
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
        "--allow-remote-images", action="store_true",
        help="Fetch http(s) image URLs during ingestion (off by default)",
    )

    # search-images
    p = sub.add_parser("search-images", help="Find images from a text query (unified index)")
    _add_global(p)
    p.add_argument("query", help="Text query")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument(
        "--use-omni", action="store_true",
        help="Encode the query with the omni text side instead of the text model",
    )

    # search
    p = sub.add_parser("search", help="Hybrid search")
    _add_global(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--type", help="Concept type filter")
    p.add_argument("--tags", help="Comma-separated tag filters")
    p.add_argument("--parent", help="Parent directory ID")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")

    # traverse
    p = sub.add_parser("traverse", help="Graph traversal")
    _add_global(p)
    p.add_argument("start_id", help="Starting concept or directory ID")
    p.add_argument("--relationship", default="CONTAINS", choices=["CONTAINS", "LINKS_TO"])
    p.add_argument("--direction", default="OUTGOING", choices=["OUTGOING", "INCOMING", "BOTH"])
    p.add_argument("--depth", type=int, default=1, help="Max depth (1-5)")
    p.add_argument("--type", help="Target node type filter")

    # list
    p = sub.add_parser("list", help="List directory contents")
    _add_global(p)
    p.add_argument("directory", nargs="?", default="", help="Directory ID (empty for root)")

    # get
    p = sub.add_parser("get", help="Get concept by ID")
    _add_global(p)
    p.add_argument("concept_id", help="Concept ID")

    # export
    p = sub.add_parser("export", help="Export concepts")
    _add_global(p)
    p.add_argument("--all", action="store_true", dest="export_all", help="Export entire bundle")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--concept-id", help="Concept ID (for single export)")
    p.add_argument("--type", help="Concept type filter")
    p.add_argument("--tags", help="Comma-separated tag filters")
    p.add_argument("--parent", help="Parent directory ID")

    # shell
    p = sub.add_parser("shell", help="Interactive REPL")
    _add_global(p)

    # broken-links
    p = sub.add_parser("broken-links", help="List broken (orphan) links")
    _add_global(p)

    # repair-links
    p = sub.add_parser("repair-links", help="Repair broken links by re-checking targets")
    _add_global(p)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "init": _init,
        "model-info": _model_info,
        "import": _import,
        "search": _search,
        "search-images": _search_images,
        "traverse": _traverse,
        "list": _list,
        "get": _get,
        "export": _export,
        "shell": _shell,
        "broken-links": _broken_links,
        "repair-links": _repair_links,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
