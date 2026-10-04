"""OKF CLI — Command-line interface for the OKF knowledge graph."""

import argparse
import cProfile
import io
import json
import logging
import sys
from pathlib import Path

from okfgraph.errors import (OKFError, envelope, internal_error, UsageError,
                             OUTCOME_CODES)
from okfgraph.router import OKFRouter
from okfgraph.settings import Settings, cli_signal, set_cli_flags
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
    """Add the generated global settings flags to any subparser.

    With mark=True (subcommands) the flags are tagged so _SlimHelpFormatter
    hides them from per-command help; they stay functional and are shown
    once in top-level help (mark=False there). The flags, their defaults
    (all argparse.SUPPRESS) and their help text derive from the settings
    table in okfgraph.settings — the CLI hand-maintains none of them.
    """
    set_cli_flags(parser, hidden=mark)


def _add_logging_flags(parser, mark=True):
    """Add --verbose / --quiet / --log-file / --profile to a subparser."""
    for args, kwargs in [
        (("--verbose", "-v"), {"action": "store_true", "help": "Enable debug logging"}),
        (("--quiet", "-q"), {"action": "store_true", "help": "Suppress all logging except errors"}),
        (("--log-file",), {"default": "", "help": "Write logs to file (with 5MB rotation)"}),
        (("--profile",), {"action": "store_true", "help": "Enable cProfile for the current invocation (outputs to stdout)"}),
        (("--json",), {"action": "store_true", "help": "Print the result envelope on stdout instead of the human renderer (D5)"}),
    ]:
        act = parser.add_argument(*args, **kwargs)
        if mark:
            act._okf_global = True  # noqa: SLF001 — our own marker


def _emit(args, op, payload):
    """``--json`` path: print the success envelope on stdout (returns True)."""
    if getattr(args, "json", False):
        print(json.dumps(envelope(op, data=payload), default=str, indent=2))
        return True
    return False


def _emit_cli_error(args, err):
    """Error adapter (§4/D6): envelope on stdout under --json, human line
    on stderr anyway. Returns the error's exit code."""
    if getattr(args, "json", False):
        env = envelope(args.command, error=err)
        if err.code in OUTCOME_CODES and err.fields and "report" in err.fields:
            # Outcome reports ride in data, never inside the error (§4).
            env["data"] = err.fields.pop("report")
        print(json.dumps(env, default=str, indent=2))
    print(f"[ERROR] {err.titles()}", file=sys.stderr)
    return err.exit_code


def _main_catchall(args):
    """Dispatch one command with the §4 catch-all: OKFError → its exit code,
    KeyboardInterrupt → 130, everything else → INTERNAL (never a bare
    traceback). Returns the process exit code."""
    command = _COMMANDS[args.command]
    try:
        return command(args)
    except OKFError as err:
        # Handlers catch their own op failures; this backstop makes the
        # contract hold even where a render path let one through.
        return _emit_cli_error(args, err)
    except KeyboardInterrupt:
        print("[ERROR] interrupted (130)", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — the §4 catch-all *is* the contract
        err = internal_error(exc, op=args.command,
                             remedy="this is a bug — report it with the message above")
        logging.getLogger("cli").exception("INTERNAL in %s", args.command)
        return _emit_cli_error(args, err)



# Routers opened during a CLI invocation, closed (checkpointed) on exit so a
# writer never leaves an un-checkpointed WAL that a later open would reject.
_OPEN_ROUTERS = []


# Ambient router injection for in-process adapters (the interactive shell)
# that must reuse ONE router across many op calls: each CLI handler calls
# _router(args); while an injection is active it returns the injected router
# instead of opening another (a second router on the same db would hit the
# ladybug file lock).
_INJECTED_ROUTER: list = []  # see _router; the shell injects its session


def _router(args):
    """Build an OKFRouter from parsed args (registered for cleanup on exit).

    Signals only what argparse actually set (generated flags default to
    SUPPRESS) into the one settings table; precedence stays
    CLI > env > TOML > defaults, per key. Bad values are a clean usage
    error, not a traceback.
    """
    if _INJECTED_ROUTER:
        return _INJECTED_ROUTER[-1]
    try:
        signal = cli_signal(args)
        settings = Settings.load(
            bundle_root=signal.get("bundle_root") or ".", cli_args=signal)
    except ValueError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(2)
    router = OKFRouter(
        db_path=settings.db_path,
        bundle_root=settings.bundle_root or ".",
        **settings.router_kwargs(),
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
    try:
        signal = cli_signal(args)
        settings = Settings.load(
            bundle_root=signal.get("bundle_root") or ".", cli_args=signal)
    except ValueError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(2)
    logger = logging.getLogger("cli")
    logger.info("initializing database at %s (dim=%d)",
                settings.db_path, settings.embedding_dim)
    pins = _router(args).init()
    logger.info("database initialized (embedding_dim=%d)",
                pins["embedding_dim"])


def _model_info(args):
    """Show model cache status without loading the model."""
    logger = logging.getLogger("cli")
    try:
        signal = cli_signal(args)
        settings = Settings.load(
            bundle_root=signal.get("bundle_root") or ".", cli_args=signal)
    except ValueError as e:
        print(f"[ERROR] {e}")
        raise SystemExit(2)
    info = OKFRouter.model_info(
        model_id=settings.model_id,
        cache_dir=settings.cache_dir,
    )
    if _emit(args, "model_info", info):
        return
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
    purge = getattr(args, "prune_missing", False)
    try:
        return _import_inner(args, router, mode, purge)
    except OKFError as err:
        print(f"[ERROR] {err.code}: {err.message}", file=sys.stderr)
        return err.exit_code
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1


def _import_inner(args, router, mode, purge):
    logger = logging.getLogger("cli")
    if getattr(args, "import_all", False):
        # --bundle-path pins the single-tree import for this call only;
        # --bundle-root never pins. No scope clash exists any more.
        pinned = getattr(args, "bundle_path", None)
        bundle_path = Path(pinned) if pinned else None
        result = router.import_bundle(
            bundle_path,
            batch_size=getattr(args, "batch_size", 32) or 32,
            mode=mode,
            prune_missing=purge,
            force=getattr(args, "force", False),
        )
        ids = result["concept_ids"]
        images = result["images"]
        logger.info("imported %d concept(s) (mode: %s)", len(ids), mode)
        for cid in ids:
            n = images.get(cid, 0)
            suffix = f"  [{n} image(s)]" if n else ""
            logger.info("  %s%s", cid, suffix)
    else:
        for fp in args.files:
            path = Path(fp)
            if not path.exists():
                logger.warning("skipping %s: file not found", fp)
                continue
            try:
                result = router.import_file(
                    path, mode=mode, force=getattr(args, "force", False)
                )
            except OKFError as err:
                logger.error("%s: %s", err.code, err.message)
                return err.exit_code
            cid = result["concept_id"]
            n = result["images"]
            suffix = f" ({n} image(s), mode: {mode})" if n else ""
            logger.info("imported: %s%s", cid, suffix)


def _search(args):
    """Unified search: concepts (default), chunks, or images.

    The canonical op (:mod:`okfgraph.ops`) owns routing, filters and
    refusal rules; the CLI renders and reports refusal errors.
    """
    router = _router(args)
    target = getattr(args, "target", "concepts") or "concepts"
    limit = getattr(args, "limit", 10) or 10
    try:
        results = router.search(
            query=args.query,
            target=target,
            limit=limit,
            concept_type=getattr(args, "concept_type", None),
            tags=args.tags.split(",") if getattr(args, "tags", None) else None,
            parent_id=getattr(args, "parent_id", None),
            include_chunks=bool(getattr(args, "include_chunks", False)),
            expand=bool(getattr(args, "expand", False)),
            context_hops=getattr(args, "context_hops", 1),
            hub_rerank=bool(getattr(args, "hub_rerank", False)),
            hub_weight=getattr(args, "hub_weight", 0.3),
            rank=getattr(args, "rank", "none") or "none",
        )
    except OKFError as err:
        return _emit_cli_error(args, err)
    if _emit(args, "search", results):
        return
    if target == "images":
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
    if target == "chunks":
        if getattr(args, "hub_rerank", False):
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
    try:
        results = router.traverse(
            start,
            relationship=args.relationship,
            direction=args.direction,
            depth=args.depth,
            node_type=getattr(args, "node_type", None),
            target=target,
            max_path_length=getattr(args, "max_path_length", 6),
        )
    except OKFError as err:
        return _emit_cli_error(args, err)
    if _emit(args, "traverse", results):
        return
    if not start:
        items = results
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
        nodes = results
        if not nodes:
            print(f"No path found between '{start}' and '{target}'.")
            return
        print(f"Path ({len(nodes)} nodes):")
        for i, n in enumerate(nodes, 1):
            print(f"  {i}. {n.get('title', '?')} ({n.get('type', '?')})")
            print(f"     id: {n['id']}")
        return
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
    max_tokens = getattr(args, "max_tokens", None)
    try:
        if max_tokens:
            reading = router.read(
                cid, include=include, max_tokens=max_tokens,
            )
            if _emit(args, "read", reading):
                return
            flag = " (truncated)" if reading["truncated"] else ""
            print(f"[{reading['used']}/{reading['budget']} tokens{flag}] {cid}\n")
            for sec in reading["sections"]:
                print(f"## {sec['title'] or sec['id']} [{sec['kind']} | {sec['id']}]")
                print(sec["text"])
                print()
            return
        if include == "chunks":
            chunks = router.read(cid, include="chunks")
            if _emit(args, "read", chunks):
                return
            if not chunks:
                print("No chunks found for this concept.")
                return
            print(f"Chunks for '{cid}' ({len(chunks)} total):\n")
            for c in chunks:
                text = c.chunk_text[:120]
                print(f"  #{c.chunk_index} [{c.block_type}] {text}")
            return
        if include == "document":
            payload = router.read(cid, include="document")
            if _emit(args, "read", payload):
                return
            markdown = payload["markdown"]
            if not markdown:
                print("No chunks found for this concept.")
                return
            if getattr(args, "output_path", None):
                Path(args.output_path).write_text(markdown, encoding="utf-8")
                print(f"[OK] Reconstructed document written to {args.output_path}")
            else:
                print(markdown)
            return
        if include == "context":
            ctx = router.read(cid, include="context")
            if _emit(args, "read", ctx):
                return
            incoming = ctx["incoming_links"]
            outgoing = ctx["outgoing_links"]
            ancestry = ctx["ancestry"]
            siblings = ctx["siblings"]
            if incoming:
                print("Linked by:")
                for l in incoming:
                    print(f"  {l.get('title', l.get('id', '?'))} (id: {l['id']})")
            if outgoing:
                print("Links to:")
                for l in outgoing:
                    print(f"  {l.get('title', l.get('id', '?'))} (id: {l['id']})")
            if ancestry:
                print(f"Path: {' → '.join((a.get('title') or a['id']) for a in ancestry)}")
            if siblings:
                print("Siblings:")
                for s in siblings:
                    print(f"  {s['title']} ({s['type']})")
                    print(f"     id: {s['id']}")
            if not (incoming or outgoing or ancestry or siblings):
                print(f"No context found for '{cid}'.")
            return
        data = router.read(cid, include="body")
    except OKFError as err:
        return _emit_cli_error(args, err)
    if _emit(args, "read", data):
        return
    body = data.pop("body", "")
    print(json.dumps(data, indent=2, default=str))
    if body:
        print(f"\n--- BODY ---\n{body}")


def _export(args):
    """Export the whole bundle (--all) or a single concept."""
    router = _router(args)
    flavor = getattr(args, "flavor", "okf") or "okf"
    try:
        if getattr(args, "export_all", False):
            tags = args.tags.split(",") if args.tags else None
            result = router.export_bundle(
                output_dir=Path(args.output_dir),
                directory_id=getattr(args, "directory_id", None),
                concept_type=getattr(args, "concept_type", None),
                tags=tags,
                flavor=flavor,
            )
            if _emit(args, "export_bundle", result):
                return
            print(f"[OK] Exported {len(result['concept_ids'])} concepts to "
                  f"{result['output_dir']} (flavor: {result['flavor']})")
        else:
            result = router.export_concept(
                args.concept_id,
                output_dir=Path(args.output_dir),
                flavor=flavor,
            )
            if _emit(args, "export_concept", result):
                return
            print(f"[OK] Exported {result['concept_id']} → {result['path']} (flavor: {flavor})")
    except OKFError as err:
        return _emit_cli_error(args, err)


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
    result = router.repair_links()
    logger.info("repaired %d link(s)", result["repaired"])


def _lint(args):
    """Pre-import bundle gate: frontmatter + link validation.

    Deliberately router-free (no DB, no ~30s model cold-boot): lint answers
    "is this bundle well-formed?" before an import cycle is spent.
    Exit 0 = clean (warnings ok), 1 = errors, 2 = usage (bad dir).
    """
    given = getattr(args, "dir", None) or "."
    target = Path(given)
    if not target.is_dir():
        return _emit_cli_error(args, UsageError(
            "FILE_NOT_FOUND", f"not a bundle directory: {target}",
            fields={"dir": str(target)},
            remedy="pass a bundle directory produced by export/produce"))
    try:
        report = OKFRouter.lint(target)
    except OKFError as err:
        report = err.fields["report"]
        if getattr(args, "json", False):
            # Outcome envelope: the report rides in data (§4).
            env = envelope("lint", data=report)
            env["ok"] = False
            env["error"] = {"code": err.code, "message": err.message}
            print(json.dumps(env, default=str, indent=2))
        else:
            print(f"{report['files']} file(s): "
                  f"{len(report['errors'])} error(s), "
                  f"{len(report['warnings'])} warning(s)")
            for e in report["errors"]:
                print(f"  [ERROR] {e['file']} {e['rule']}: {e['message']}")
            for w in report["warnings"]:
                print(f"  [warn] {w['file']} {w['rule']}: {w['message']}")
        return 1
    if getattr(args, "json", False):
        env = envelope("lint", data=report)
        if not report["clean"]:
            # Outcome envelope: errors ride in data (§4); exit 1.
            env["ok"] = False
            env["error"] = {"code": "LINT_ERRORS",
                            "message": f"{len(report['errors'])} lint error(s)"}
        print(json.dumps(env, default=str, indent=2))
        return 0 if report["clean"] else 1
    print(f"{report['files']} file(s): "
          f"{len(report['errors'])} error(s), "
          f"{len(report['warnings'])} warning(s)")
    for e in report["errors"]:
        print(f"  [ERROR] {e['file']} {e['rule']}: {e['message']}")
    for w in report["warnings"]:
        print(f"  [warn] {w['file']} {w['rule']}: {w['message']}")
    if report["clean"]:
        print("Bundle is lint-clean (safe to import).")
    return 0 if report["clean"] else 1


def _produce(args):
    """Generate a bundle from a data source (bundle-hardening §3).

    Router-free like lint (no DB, no model cold-boot): produce answers
    "what would this source look like as concepts?" The emitted bundle is
    pre-flighted with lint_bundle against the output root, so the report
    below covers both generation and well-formedness.
    Exit 0 = produced + lint-clean, 1 = produced but lint errors (or the
    source is unreadable), 2 = usage (unknown producer, missing file).
    """
    which = getattr(args, "from_", None)
    source = getattr(args, "source", None)
    out = Path(str(getattr(args, "output", None) or "."))
    try:
        result = OKFRouter.produce(
            which, source, out,
            prefix=getattr(args, "prefix", None) or None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    except OKFError as err:
        print(f"[ERROR] {err.code}: {err.message}", file=sys.stderr)
        return err.exit_code
    report = result["lint"]
    print(f"[OK] {result['producer']}: {result['files']} file(s) "
          f"from {result['root']}/")
    n_err, n_warn = len(report["errors"]), len(report["warnings"])
    print(f"lint: {report['files']} file(s): {n_err} error(s), {n_warn} warning(s)")
    for e in report["errors"]:
        print(f"  [ERROR] {e['file']} {e['rule']}: {e['message']}")
    for w in report["warnings"]:
        print(f"  [warn] {w['file']} {w['rule']}: {w['message']}")
    if report["clean"]:
        print("Bundle is lint-clean (safe to import).")
        return 0
    return 1


def _no_db_diff(old, new):
    """Snapshot diff without a router (the op class lives on the router;
    the static path is expressed via a throwaway instance-free call)."""
    from okfgraph.components.diff import DiffManager
    if old and new and Path(old).is_dir() and Path(new).is_dir():
        return DiffManager(None).diff_dirs(Path(old), Path(new))
    raise OKFError("BAD_VALUE", "diff needs two bundle directories (or one side + --db/--bundle-root)",
                   fields={"old": str(old), "new": str(new)})


def _diff(args):
    """Structural diff: snapshot (dir vs dir) or drift (graph vs dir).

    Returns an exit code (0 = identical, 1 = different) for CI gating;
    main() propagates int returns to sys.exit.
    """
    old = getattr(args, "old", None)
    new = getattr(args, "new", None)
    as_json = getattr(args, "json", False)
    try:
        # The op refuses BAD_VALUE for malformed sides and raises
        # DIFF_DIFFERENT (report inside error.fields) when they differ.
        # A router is only needed for drift; snapshot mode is router-free.
        needs_db = not (old and new)
        router = _router(args) if needs_db else None
        result = router.diff(old, new) if router else _no_db_diff(old, new)
    except OKFError as err:
        if err.code == "DIFF_DIFFERENT":
            report = err.fields["report"]
            if as_json:
                # Outcome envelope: the report rides in data (§4).
                env = envelope("diff", data=report)
                env["error"] = {"code": err.code, "message": err.message}
                print(json.dumps(env, default=str, indent=2))
            else:
                _print_diff(report)
            return 1
        return _emit_cli_error(args, err)
    if as_json:
        env = envelope("diff", data=result)
        env["ok"] = bool(result["identical"])
        if not result["identical"]:
            env["error"] = {"code": "DIFF_DIFFERENT",
                            "message": "structures differ"}
        print(json.dumps(env, default=str, indent=2))
    else:
        _print_diff(result)
    return 0 if result["identical"] else 1


def _print_diff(result) -> None:
    """Human-readable rendering of a structural diff report."""
    if result["identical"]:
        print("No structural differences.")
        return
    for cid in result["added"]:
        print(f"  + concept {cid}")
    for cid in result["removed"]:
        print(f"  - concept {cid}")
    for cid in result["changed"]:
        print(f"  ~ body {cid}")
    for r in result["retitled"]:
        print(f"  ~ title {r['id']}: {r['old']!r} -> {r['new']!r}")
    for r in result["retyped"]:
        print(f"  ~ type {r['id']}: {r['old']!r} -> {r['new']!r}")
    for s, t in result["edges_added"]:
        print(f"  + edge {s} -> {t}")
    for s, t in result["edges_removed"]:
        print(f"  - edge {s} -> {t}")
    for s, t in result["broken_new"]:
        print(f"  + broken {s} -> {t}")
    for s, t in result["broken_fixed"]:
        print(f"  - broken {s} -> {t}")


def _detach(args):
    """End the mirror relationship: the DB becomes the artifact."""
    logger = logging.getLogger("cli")
    router = _router(args)
    bundle = getattr(args, "bundle_path", None)
    try:
        report = router.detach(
            bundle_path=Path(bundle) if bundle else None,
            verify=not getattr(args, "no_verify", False),
            force=getattr(args, "force", False),
        )
    except OKFError as err:
        print(f"[ERROR] {err.code}: {err.message}", file=sys.stderr)
        return err.exit_code
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1
    if report["verified"]:
        logger.info("verified %d source file(s) against the graph", report["file_count"])
    else:
        logger.info("detached without verification")
    if report["mismatched"]:
        logger.warning("acknowledged %d mismatched file(s)", len(report["mismatched"]))
    for u in report["untracked"][:10]:
        logger.warning("  untracked (WILL-NOT-SURVIVE): %s", u["path"])
    for n in report["non_source_files"][:10]:
        logger.warning("  source-only (WILL-NOT-SURVIVE): %s", n)
    if report["already_sourceless"]:
        logger.info("%d tracked source(s) already live only in the graph",
                      report["already_sourceless"])
    prov = report["provenance"]
    rows = prov if isinstance(prov, list) else [prov]
    for row in rows:
        label = f"@{row['alias']}" if row.get("alias") else "<primary>"
        print(f"[OK] detached {label} from {row['path']} "
              f"({row['file_count']} file(s) at detach).")
    print("Reads/search/export/recover keep working; imports refuse without --force.")
    return 0


def _doctor(args):
    """Scored health scan, optionally with safe --fix repairs.

    Returns an exit code with --strict (1 when any finding exists).
    """
    router = _router(args)
    try:
        result = router.doctor(
            stale_days=getattr(args, "stale_days", 365) or 365,
            fix=getattr(args, "fix", False),
            strict=getattr(args, "strict", False),
        )
    except OKFError as err:
        # DOCTOR_FINDINGS keeps the report in fields; render it, then exit 1.
        result = {"report": err.fields["report"]}
        if "fixed" in err.fields:
            result["fixed"] = err.fields["fixed"]
        strict_cause = True
    else:
        strict_cause = False
    report = result["report"]
    if getattr(args, "json", False):
        env = envelope("doctor", data={"report": report, "fixed": result.get("fixed")})
        if strict_cause:
            env["ok"] = False
            env["error"] = {"code": "DOCTOR_FINDINGS",
                            "message": f"doctor found "
                                       f"{len(report['findings'])} finding(s)"}
        print(json.dumps(env, default=str, indent=2))
        return 1 if strict_cause else 0
    if "fixed" in result:
        fixed = result["fixed"]
        print(f"[OK] repaired {fixed['repaired_links']} link(s), "
              f"normalized {fixed['normalized_timestamps']} timestamp(s), "
              f"cleared {fixed.get('cleared_orphan_hashes', 0)} orphan hash row(s)")
        if fixed["skipped_reviewed"]:
            print(f"  skipped reviewed: {', '.join(fixed['skipped_reviewed'])}")
    print(f"Health score: {report['score']}/100 ({report['concepts']} concepts)")
    for f in report["findings"]:
        print(f"  [{f['severity']}] {f['rule']} {f['path']}: {f['message']}")
    for i in report["info"]:
        print(f"  (info) {i['rule']}: {i['message']}")
    if report.get("detached"):
        d = report["detached"]
        roots = ", ".join(r["path"] for r in d.get("roots", [])) or "<none>"
        print(f"  [detached] mirror ended (epoch {d.get('detached_at')}); "
              f"sources: {roots}")
    for rt in report.get("roots", []):
        state = "present" if rt["present"] else "ABSENT"
        print(f"  [root {rt['alias']}] {state} {rt['path']} "
              f"({rt['tracked_files']} tracked file(s), "
              f"{rt['concepts']} concept(s))")
    if strict_cause:
        return 1
    return 0


def _reindex(args):
    logger = logging.getLogger("cli")
    router = _router(args)
    ran = router.reindex(if_dirty=bool(getattr(args, "if_dirty", False)))["rebuilt"]
    if ran:
        logger.info("search indexes rebuilt")
    else:
        logger.info("search indexes already up to date; nothing to do.")


def _ingest(args):
    """Unified ingest: markdown file, PDF (default), or raw thoughts.

    The canonical op owns dispatch, file/param validation and converter
    construction; the CLI renders and keeps its page progress printer.
    """
    logger = logging.getLogger("cli")
    router = _router(args)
    kind = getattr(args, "kind", None)

    tags = args.tags.split(",") if getattr(args, "tags", None) else None
    auto_import = getattr(args, "auto_import", True)
    output_dir = getattr(args, "output_dir", None)
    on_page = None
    if kind == "pdf":
        def on_page(idx, total):
            print(f"  page {idx + 1}/{total}", end="\r")

    try:
        result = router.ingest(
            kind,
            md_path=getattr(args, "md_path", None),
            pdf_path=getattr(args, "pdf_path", None),
            thoughts=getattr(args, "thoughts", None),
            topic=getattr(args, "topic", None),
            concept_id=getattr(args, "concept_id", None),
            title=getattr(args, "title", None),
            description=getattr(args, "description", None),
            tags=tags,
            mode=getattr(args, "mode", "text") or "text",
            routing_mode=getattr(args, "routing_mode", "auto") or "auto",
            extract_images=not getattr(args, "no_extract_images", False),
            auto_import=auto_import,
            output_dir=output_dir,
            batch_size=getattr(args, "batch_size", 32) or 32,
            prune_missing=getattr(args, "prune_missing", False),
            force=getattr(args, "force", False),
            on_page=on_page,
        )
    except OKFError as err:
        print(f"[ERROR] {err.code}: {err.message}", file=sys.stderr)
        return err.exit_code
    if kind == "pdf":
        print()  # newline after PDF progress (harmless for md/thoughts)

    if kind == "md":
        print(f"[OK] Imported {result['concept_id']} ({result['chunk_count']} chunks)")
        return
    if kind == "thoughts":
        print(f"[OK] Stored thought {result['concept_id']}")
        return

    logger.info("written %s", result["md_path"])
    logger.info("assets in %s", result["image_dir"])
    if auto_import:
        for cid in result["concept_ids"]:
            n = len(router.image_mgr.list_images(cid))
            suffix = f"  [{n} image(s)]" if n else ""
            logger.info("  %s%s", cid, suffix)
    else:
        logger.info("run 'okf import --all --bundle-root %s' to import.", output_dir)


def _images(args):
    """List image assets attached to a concept (UNKNOWN_CONCEPT -> exit 1)."""
    router = _router(args)
    try:
        imgs = router.list_images(args.concept_id)
    except OKFError as err:
        return _emit_cli_error(args, err)
    if _emit(args, "list_images", imgs):
        return
    if not imgs:
        print("No images attached.")
        return
    print(f"Images for '{args.concept_id}' ({len(imgs)} total)")
    for im in imgs:
        alt = im.get("alt_text") or "(no alt-text)"
        print(f"  [{im.get('embed_route')}] {im.get('file_name')} — {alt}")
        print(f"     id: {im.get('id')}")


def _image(args):
    """Fetch one image asset: metadata + optional bytes (--output-path)."""
    import base64
    router = _router(args)
    try:
        row = router.get_image(args.asset_id)
    except OKFError as err:
        return _emit_cli_error(args, err)
    if _emit(args, "get_image", row):
        return
    if getattr(args, "output_path", None):
        Path(args.output_path).write_bytes(
            base64.b64decode(row["data"]))
        print(f"[OK] Image bytes written to {args.output_path}")
    alt = row.get("alt_text") or "(no alt-text)"
    print(f"[{row.get('embed_route')}] {row.get('file_name')} — {alt}")
    print(f"   id: {row.get('id')}")
    print(f"   mime: {row.get('mime_type')}")


def _deleted_list(args):
    """List soft-deleted concepts with recovery status."""
    router = _router(args)
    deleted = router.list_deleted()
    if _emit(args, "list_deleted", deleted):
        return
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
    """Recover a soft-deleted concept (NOT_RECOVERABLE -> exit 1)."""
    router = _router(args)
    try:
        router.recover_deleted(args.concept_id)
    except OKFError as err:
        print(f"[ERROR] {err.code}: {err.message}", file=sys.stderr)
        return err.exit_code
    print(f"[OK] Recovered concept '{args.concept_id}'.")


def _deleted_purge(args):
    """Permanently delete expired soft-deleted concepts."""
    router = _router(args)
    older_than = getattr(args, "older_than", None)
    result = router.purge_deleted(older_than=older_than)
    print(f"[OK] Permanently deleted {result['purged']} expired concept(s).")


def _shell_argv(cmd: str, rest: str):
    """Translate one shell line into argv for :func:`build_parser`.

    The shell keeps its ergonomic shorthand (``chunks:`` prefixes,
    bare ``expand``/``hub`` modifiers, the two-forms traverse) but the
    resulting argv always resolves through the SAME subparsers the CLI
    uses — flags have exactly one definition and one renderer.
    Returns ``None`` for words that are not shell verbs.
    """
    tokens = rest.strip().split()
    if cmd == "import":
        return ["import"] + tokens if tokens else None
    if cmd == "import-bundle":
        argv = ["import", "--all"]
        if tokens:
            argv += ["--bundle-path", " ".join(tokens)]
        return argv
    if cmd == "search":
        if not tokens:
            return None
        query = tokens[0]
        modifiers = {t for t in tokens[1:]}
        extra = []
        target = None
        if ":" in query and query.split(":")[0] in ("concepts", "chunks", "images"):
            target, query = query.split(":", 1)
        for t in sorted(modifiers):
            if t == "expand":
                extra.append("--expand")
            elif t == "hub":
                extra.append("--hub-rerank")
            elif t.startswith("type:"):
                extra += ["--concept-type", t[5:]]
            elif t.startswith("tags:"):
                extra += ["--tags", t[5:]]
            elif t.startswith("parent:"):
                extra += ["--parent-id", t[7:]]
            else:
                query_parts = [query, t]
                query = " ".join(query_parts)
        if "expand" in modifiers or "hub" in modifiers:
            target = target or "chunks"
        argv = ["search", query]
        if target and target != "concepts":
            argv += ["--target", target]
        argv += extra
        return argv
    if cmd == "read":
        if not tokens:
            return None
        argv = ["read", tokens[0]]
        if len(tokens) > 1:
            argv += ["--include", tokens[1]]
        return argv
    if cmd == "traverse":
        rels = ("CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET")
        if not tokens:
            return ["traverse"]
        if len(tokens) == 1:
            return ["traverse", tokens[0]]
        if len(tokens) == 2 and tokens[1].upper() not in rels:
            return ["traverse", tokens[0], "--target", tokens[1]]
        argv = ["traverse", tokens[0]]
        if len(tokens) > 1:
            argv += ["--relationship", tokens[1].upper()]
        if len(tokens) > 2:
            argv += ["--direction", tokens[2]]
        if len(tokens) > 3:
            argv += ["--depth", tokens[3]]
        return argv
    if cmd == "images":
        return ["images", rest.strip()]
    if cmd == "export-bundle":
        return ["export", "--all", "--output-dir", rest.strip()]
    if cmd == "export":
        if len(tokens) != 2:
            return None
        return ["export", "--concept-id", tokens[0], "--output-dir", tokens[1]]
    if cmd == "ingest":
        if not tokens:
            return None
        path = tokens[0]
        auto = "--auto-import" in tokens[1:]
        kind = "md" if path.lower().endswith(".md") else "pdf"
        argv = ["ingest", "--kind", kind]
        if kind == "md":
            argv += ["--md-path", path]
        else:
            argv += ["--pdf-path", path]
        if not auto:
            argv += ["--no-auto-import"]
        return argv
    if cmd in ("model-info", "broken-links", "repair-links"):
        return [cmd]
    return None


def _shell(args, router=None):
    """Interactive shell: a REPL over the same subcommands as the CLI.

    Every line is translated by :func:`_shell_argv` and handed to
    ``build_parser()`` + the same handler as ``okf <verb>`` (§5 CLI row):
    flags exist once and each op has one renderer. The session's own
    router is reused for every line (a fresh ``_router`` per line would
    clash on the ladybug file lock).
    """
    router = router or _router(args)
    base = {k: v for k, v in vars(args).items() if k != "command"}
    parser = build_parser()
    banner = """OKF Interactive Shell
========================================
Commands:
  import <file>              — import single OKF file (same as okf import)
  import-bundle [path]       — import entire bundle (same as okf import --all)
  search [target:]<query> [hub|expand|type:X|tags:a,b|parent:Y] — search
  read <id> [chunks|document|context] — read a concept (default: body)
  traverse [id] [REL] [DIR] [depth] — traverse; 2 ids = shortest path
  images <concept_id>        — list images attached to a concept
  export-bundle <output_dir> — export all concepts
  export <id> <output_dir>   — export single concept
  ingest <file> [--auto-import] — ingest .md or .pdf (converts PDFs by default)
  model-info                 — show model cache status
  broken-links / repair-links — link hygiene
  help                       — show this help
  quit / exit                — exit shell
Any CLI flag works after the command, e.g. search --rank ppr --limit 5.
========================================"""

    print(banner)

    _INJECTED_ROUTER.append(router)
    try:
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

            if cmd == "help":
                print(banner)
                continue

            argv = _shell_argv(cmd, rest)
            if argv is None:
                print(f"Unknown command: {cmd}. Type 'help' for usage.")
                continue

            # Parse with the session's settings pre-seeded: generated global
            # flags are SUPPRESS-defaulted, so only args typed on the line
            # overwrite them (D5/D6 hold; the shell is human-only by default).
            ns = argparse.Namespace(**base)
            try:
                sub = parser.parse_args(argv, namespace=ns)
            except SystemExit:
                print(f"Error: bad usage for '{cmd}'.")
                continue
            try:
                _COMMANDS[sub.command](sub)
            except OKFError as err:
                # Handlers catch their own op errors; this keeps the REPL
                # alive for anything a renderer path let through (§4).
                print(f"[ERROR] {err.titles()}", file=sys.stderr)
    finally:
        _INJECTED_ROUTER.clear()


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

    # import
    p = sub.add_parser("import", help="Import OKF files")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("files", nargs="*", help="Files to import")
    p.add_argument("--all", action="store_true", dest="import_all",
                     help="Import entire bundle (omit --bundle-path to import every "
                     "configured root; with --bundle-path, only that tree)")
    p.add_argument("--bundle-path", default=None,
                   help="Pin a single tree for this import (with --all); "
                        "the primary root still comes from --bundle-root/TOML")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding (default: 32)")
    p.add_argument(
        "--mode", default="text", choices=["text", "optional", "omni"],
        help="Image ingestion mode: text (captions), optional (vision for "
             "images lacking alt-text), omni (vision for every image).",
    )
    p.add_argument(
        "--prune-missing", dest="prune_missing", action="store_true", default=False,
        help="Also drop concepts whose source files were deleted from disk "
             "(removes concept, chunks, links, and orphaned image assets)",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Bypass the detached-graph refusal: re-attach when the bundle "
             "matches the recorded source tree, or acknowledge an explicitly "
             "addressed file import.",
    )

    # search (unified: concepts, chunks, images)
    p = sub.add_parser("search", help="Search concepts, chunks, or images")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("query", help="Search query")
    p.add_argument("--target", default="concepts", choices=["concepts", "chunks", "images"],
                   help="What to search (default: concepts)")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument("--concept-type", dest="concept_type", help="Concept type filter (concepts/chunks)")
    p.add_argument("--tags", help="Comma-separated tag filters (concepts/chunks)")
    p.add_argument("--parent-id", dest="parent_id", help="Parent directory ID (concepts/chunks)")
    p.add_argument("--include-chunks", dest="include_chunks", action="store_true", help="Include matched chunks per concept result")
    p.add_argument("--expand", action="store_true", help="Chunks: attach graph neighborhood to each hit")
    p.add_argument("--context-hops", type=int, default=1, help="Expansion hops with --expand (default: 1)")
    p.add_argument("--hub-rerank", action="store_true", help="Chunks: rerank by graph hub score")
    p.add_argument("--hub-weight", type=float, default=0.3, help="Hub weight with --hub-rerank (default: 0.3)")
    p.add_argument("--rank", default="none", choices=["none", "hub", "ppr"],
                   help="Concepts: ranking — none (RRF order), hub (blend incoming-link "
                   "authority), ppr (model-free lexical-seed PPR, no ONNX load). "
                   "Default: none")

    # read (unified: body, chunks, document, context)
    p = sub.add_parser("read", help="Read a concept: body, chunks, document, or context")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")
    p.add_argument("--include", default="body", choices=["body", "chunks", "document", "context"],
                   help="What to return (default: body)")
    p.add_argument("--output-path", dest="output_path", help="Output file for --include document (default: stdout)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Token budget: assemble self + PPR-ranked neighbours "
                   "(index-first for context), truncating to fit")

    # traverse (unified: relationships, directory listing, shortest path)
    p = sub.add_parser("traverse", help="Traverse relationships, list directories, find paths")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("start_id", nargs="?", default="", help="Starting concept or directory ID (empty = root listing)")
    p.add_argument("--relationship", default="CONTAINS", choices=["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"])
    p.add_argument("--direction", default="OUTGOING", choices=["OUTGOING", "INCOMING", "BOTH"])
    p.add_argument("--depth", type=int, default=1, help="Max depth (1-5)")
    p.add_argument("--node-type", dest="node_type", help="Target node type filter")
    p.add_argument("--target", default=None, help="Find shortest path from start_id to this ID instead of traversing")
    p.add_argument("--max-path-length", type=int, default=6, help="Max path length with --target (default: 6)")

    # ingest (unified: markdown, PDF, thoughts)
    p = sub.add_parser("ingest", help="Add content: markdown file, PDF, or thoughts")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--kind", required=True, choices=["md", "pdf", "thoughts"],
                   help="What to ingest")
    p.add_argument("--md-path", dest="md_path", default=None, help="Markdown file (--kind md)")
    p.add_argument("--pdf-path", dest="pdf_path", default=None, help="File to convert (--kind pdf): PDF or Office (docx/xlsx/pptx, legacy doc/xls/ppt)")
    p.add_argument("--thoughts", default=None, help="Raw reasoning text (--kind thoughts)")
    p.add_argument("--topic", default=None, help="Topic (--kind thoughts)")
    p.add_argument("--concept-id", default=None, help="Explicit concept ID (--kind md/thoughts; slashes are namespaces; thoughts default to thoughts/<topic>/<ts>_<id>)")
    p.add_argument("--title", default=None, help="Title override (--kind md)")
    p.add_argument("--description", default=None, help="Description override (--kind md)")
    p.add_argument("--tags", default=None, help="Comma-separated tags (--kind md/thoughts)")
    p.add_argument(
        "--no-auto-import", action="store_false", dest="auto_import",
        help="Convert only (kind=pdf): write markdown, skip the import",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Output directory for converted markdown (kind=pdf, convert-only; default: current dir)",
    )
    p.add_argument(
        "--routing-mode", default="auto",
        choices=["auto", "surgical", "always", "never"],
        help="ONNX routing mode (--kind pdf, default: auto)",
    )
    p.add_argument(
        "--mode", default="text", choices=["text", "optional", "omni"],
        help="Image ingestion mode: text (captions), optional/omni (ONNX vision).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for encoding during auto-import (default: 32)",
    )
    p.add_argument(
        "--prune-missing", dest="prune_missing", action="store_true", default=False,
        help="Drop concepts whose source files vanished during auto-import",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Bypass the detached-graph refusal for single-file / thought "
             "ingest (never re-attaches; PDF auto-import cannot re-attach "
             "from a temp dir).",
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
    p.add_argument("--output-dir", dest="output_dir", required=True, help="Output directory")
    p.add_argument("--concept-id", help="Concept ID (for single export)")
    p.add_argument("--concept-type", dest="concept_type", help="Concept type filter")
    p.add_argument("--tags", help="Comma-separated tag filters")
    p.add_argument("--directory-id", dest="directory_id", help="Parent directory ID")
    p.add_argument("--flavor", default="okf", choices=["okf", "obsidian"],
                   help="Link flavor: okf ([t](id.md) + index files) or obsidian "
                   "([[Title]] wikilinks, no index files). Default: okf")

    # diff (structural: snapshot dir-vs-dir, or drift graph-vs-dir)
    p = sub.add_parser("diff", help="Structural diff: concepts/edges/broken-link deltas")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("old", nargs="?", default=None,
                   help="Old side: bundle dir (with NEW: snapshot; alone: drift vs graph)")
    p.add_argument("new", nargs="?", default=None, help="New side: bundle dir (snapshot mode)")

    # doctor (scored health + safe repairs)
    p = sub.add_parser("doctor", help="Health scan: score, findings, safe --fix")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 when any finding exists (CI gate)")
    p.add_argument("--fix", action="store_true",
                   help="Apply safe repairs (link re-points, timestamp normalization; "
                   "never touches reviewed:true concepts)")
    p.add_argument("--stale-days", type=int, default=365,
                   help="Age threshold for 'stale' findings (default: 365)")

    # lint (pre-import bundle gate: no DB, no model load)
    p = sub.add_parser("lint", help="Validate bundle frontmatter + links before import")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("dir", nargs="?", default=None,
                   help="Bundle directory (default: --bundle, okfgraph.toml, or .)")

    # produce
    p = sub.add_parser("produce", help="Generate a bundle from a data source")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--from", dest="from_", required=True,
                   help="Producer name (today: sqlite)")
    p.add_argument("--source", required=True,
                   help="Source to read (sqlite: path to the .db file)")
    p.add_argument("--output", default=None,
                   help="Bundle root to write under (default: .); files go "
                   "to <output>/<prefix>/ so lint + import agree on ids")
    p.add_argument("--prefix", default=None,
                   help="Namespace directory (default: the producer's own)")
    p.add_argument("--overwrite", action="store_true",
                   help="Rewrite existing producer output files")

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

    p = sub.add_parser("detach", help="End the mirror relationship: the DB becomes the artifact")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("--bundle-path", default=None,
                   help="Bundle tree to detach (default: the primary root)")
    p.add_argument("--no-verify", action="store_true", default=False,
                   help="Skip the sources-vs-graph fidelity check (for already-removed trees)")
    p.add_argument("--force", action="store_true", default=False,
                   help="Acknowledge mismatches / untracked files / source-only "
                   "artifacts (WILL-NOT-SURVIVE) and detach anyway")

    # images (Q5 -> B: image ops on all surfaces)
    p = sub.add_parser("images", help="List image assets attached to a concept")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("concept_id", help="Concept ID")

    # image
    p = sub.add_parser("image", help="Fetch one image asset (metadata; --output-path writes bytes)")
    _add_global(p)
    _add_logging_flags(p)
    p.add_argument("asset_id", help="Image asset ID")
    p.add_argument("--output-path", dest="output_path", default=None,
                   help="Write the image bytes here (default: metadata only)")

    return parser


_COMMANDS = {
    "init": _init,
    "model-info": _model_info,
    "import": _import,
    "ingest": _ingest,
    "search": _search,
    "read": _read,
    "traverse": _traverse,
    "export": _export,
    "images": _images,
    "image": _image,
    "shell": _shell,
    "broken-links": _broken_links,
    "repair-links": _repair_links,
    "diff": _diff,
    "doctor": _doctor,
    "lint": _lint,
    "produce": _produce,
    "reindex": _reindex,
    "deleted-list": _deleted_list,
    "deleted-recover": _deleted_recover,
    "deleted-purge": _deleted_purge,
    "detach": _detach,
}


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

    commands = _COMMANDS

    try:
        ret = _main_catchall(args)
    finally:
        _close_routers()
        _teardown_logging()
    # Handlers return an int exit code when CI-gateable (diff/doctor);
    # everything else returns None (= success).
    if isinstance(ret, int) and ret != 0:
        sys.exit(ret)

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
