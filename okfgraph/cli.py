"""OKF CLI — Command-line interface for the OKF knowledge graph.

Every handler is parse → op → render: the canonical op on
:class:`~okfgraph.router.OKFRouter` owns validation and refusals, the
handler only maps flags to op params and hands the result to
:func:`_out`, which prints the envelope under ``--json`` or calls the
command's human renderer. Errors are never handled per command:
:func:`_main_catchall` turns any ``OKFError`` into ``[ERROR] CODE: …`` on
stderr (+ the error envelope under ``--json``) and its exit code; outcome
errors (diff differs, doctor findings, lint errors) render their report
first. Results go to stdout, logs to stderr (D5/D6).
"""

import argparse
import base64
import cProfile
import io
import json
import logging
import sys
from pathlib import Path

from okfgraph.errors import (OKFError, dumps_envelope, envelope,
                             internal_error)
from okfgraph.router import OKFRouter
from okfgraph.settings import Settings, cli_signal, set_cli_flags

logger = logging.getLogger("cli")

# ── Logging setup ──────────────────────────────────────────────────────────

_LOG_HANDLER = None


def _setup_logging(verbose: bool = False, quiet: bool = False, log_file: str = "",
                   console_level: int = None) -> None:
    """Configure logging for the CLI.

    Precedence: quiet > verbose > console_level > default.
    - quiet: ERROR and above only
    - verbose: DEBUG
    - default: INFO (console_level=None); ``--json`` passes WARNING so
      harnesses that merge stderr into stdout still parse the envelope
      (D3): results go to stdout, warnings/errors to stderr.

    ``console_level`` only quiets the console; ``--log-file`` keeps the
    base level (INFO by default) so a ``--json`` run still logs fully.
    """
    global _LOG_HANDLER

    if quiet:
        base = logging.ERROR
    elif verbose:
        base = logging.DEBUG
    else:
        base = logging.INFO
    level = base
    if console_level is not None and not (quiet or verbose):
        level = console_level

    root = logging.getLogger()
    root.setLevel(min(level, base) if log_file else level)
    # Remove any existing handlers to avoid duplicates across invocations
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    console.setLevel(level)
    root.addHandler(console)
    _LOG_HANDLER = console

    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(base)
        root.addHandler(file_handler)


def _teardown_logging() -> None:
    """Remove handlers to avoid leaks across CLI invocations."""
    global _LOG_HANDLER
    root = logging.getLogger()
    if _LOG_HANDLER and _LOG_HANDLER in root.handlers:
        root.removeHandler(_LOG_HANDLER)
    _LOG_HANDLER = None


# ── help formatting ────────────────────────────────────────────────────────

class _SlimHelpFormatter(argparse.HelpFormatter):
    """Subcommand help without the repeated global flags.

    Global options (connection, models, logging) are identical on every
    command, so printing them per command costs agents ~4k tokens to learn
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
        # One spelling alive: no prefix-abbreviation aliases (e.g. a bare
        # `--output` must not silently resolve to `--output-dir`).
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def _add_global(parser, mark=True):
    """Add the generated settings flags plus the presentation flags.

    With mark=True (subcommands) the flags are tagged so _SlimHelpFormatter
    hides them from per-command help; they stay functional and are shown
    once in top-level help (mark=False there). Settings flags derive from
    the table in okfgraph.settings — the CLI hand-maintains none of them.
    """
    set_cli_flags(parser, hidden=mark)
    for args, kwargs in [
        (("--verbose", "-v"), {"action": "store_true", "help": "Enable debug logging"}),
        (("--quiet", "-q"), {"action": "store_true", "help": "Suppress all logging except errors"}),
        (("--log-file",), {"default": "", "help": "Write logs to file (with 5MB rotation)"}),
        (("--profile",), {"action": "store_true", "help": "Profile the invocation with cProfile (stats on stderr)"}),
        (("--json",), {"action": "store_true", "help": "Print the result envelope {ok, op, data, warnings, error} on stdout instead of the human rendering"}),
    ]:
        act = parser.add_argument(*args, **kwargs)
        if mark:
            act._okf_global = True  # noqa: SLF001 — our own marker


# ── settings, routers, output ──────────────────────────────────────────────

# Routers opened during a CLI invocation, closed (checkpointed) on exit so a
# writer never leaves an un-checkpointed WAL that a later open would reject.
_OPEN_ROUTERS = []

# Ambient router injection for in-process adapters (the interactive shell)
# that must reuse ONE router across many op calls: while an injection is
# active _router returns it instead of opening another (a second router on
# the same db would hit the ladybug file lock).
_INJECTED_ROUTER: list = []


def _settings(args) -> Settings:
    """Merged settings for this invocation (cached on the namespace).

    Only flags argparse actually set reach the table (generated flags
    default to SUPPRESS); precedence is CLI > env > TOML > defaults, per
    key. Invalid configuration is ``CONFIG_INVALID`` (exit 2).
    """
    cached = getattr(args, "_okf_settings", None)
    if cached is not None:
        return cached
    try:
        signal = cli_signal(args)
        settings = Settings.load(
            bundle_root=signal.get("bundle_root") or ".", cli_args=signal)
    except ValueError as e:
        raise OKFError("CONFIG_INVALID", str(e)) from None
    args._okf_settings = settings
    return settings


def _router(args):
    """Build an OKFRouter from parsed args (registered for cleanup on exit)."""
    if _INJECTED_ROUTER:
        return _INJECTED_ROUTER[-1]
    settings = _settings(args)
    try:
        router = OKFRouter(
            db_path=settings.db_path,
            bundle_root=settings.bundle_root or ".",
            **settings.router_kwargs(),
        )
    except OKFError:
        raise
    except (ValueError, FileNotFoundError) as e:
        # Constructor validation (dims, device, roots, explicit model
        # files) is configuration, not a bug.
        raise OKFError("CONFIG_INVALID", str(e)) from None
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


def _out(args, op, data, render=None):
    """Emit one op result: the envelope under ``--json``, else ``render``.

    Returns the renderer's exit code (None = 0).
    """
    if getattr(args, "json", False):
        print(dumps_envelope(envelope(op, data=data)))
        return None
    if render is not None:
        return render(data)
    return None


def _csv(value):
    return [t.strip() for t in value.split(",") if t.strip()] if value else None


# ── human renderers ────────────────────────────────────────────────────────

def _print_findings(report, prefix=""):
    for e in report["errors"]:
        print(f"{prefix}  [ERROR] {e['file']} {e['rule']}: {e['message']}")
    for w in report["warnings"]:
        print(f"{prefix}  [warn] {w['file']} {w['rule']}: {w['message']}")


def _render_lint(report, label=""):
    print(f"{label}{report['files']} file(s): "
          f"{len(report['errors'])} error(s), "
          f"{len(report['warnings'])} warning(s)")
    _print_findings(report)
    if report["clean"]:
        print("Bundle is lint-clean (safe to import).")


def _render_diff(result) -> None:
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


def _render_doctor(result) -> None:
    report, fixed = result["report"], result.get("fixed")
    if fixed:
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


#: Outcome errors render their report (``err.data``) before the [ERROR]
#: line — the report *is* the result; the exit code signals the outcome.
_OUTCOME_RENDERERS = {
    "lint": _render_lint,
    "diff": _render_diff,
    "doctor": _render_doctor,
}


def _render_concepts(results):
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
        if r.get("matched_chunks"):
            print("     matched chunks:")
            for mc in r["matched_chunks"][:3]:
                print(f"       [{mc['rrf_score']:.4f}] {mc['block_type']} #{mc['chunk_index']}")
                print(f"         {mc.get('chunk_text', '')[:100]}")
        print()


def _render_chunks(results):
    if not results:
        print("No chunk results found.")
        return
    print(f"Found {len(results)} chunk(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['rrf_score']:.4f}] {r['block_type']} #{r['chunk_index']}")
        print(f"     {r.get('chunk_text', '')[:150]}")
        if r.get("parent_title"):
            print(f"     parent: {r['parent_title']}")
        print(f"     id: {r['chunk_id']}")
        print()


def _render_hub_chunks(results):
    if not results:
        print("No results found.")
        return
    print(f"Found {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['final_score']:.4f}] {r['parent_title']} §{r['chunk_index']}")
        print(f"     hub={r['hub_score']:.2f} rrf={r['rrf_score']:.4f}")
        print(f"     {r['chunk_text'][:150]}")
        print()


def _titles(rows, n=3):
    return ", ".join(r.get("title") or r.get("id", "?") for r in rows[:n])


def _render_expanded(results):
    if not results:
        print("No results found.")
        return
    print(f"Found {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        chunk = r["chunk"]
        print(f"  {i}. [{chunk['rrf_score']:.4f}] {chunk['parent_title']} §{chunk['chunk_index']}")
        print(f"     {chunk['chunk_text'][:150]}")
        if r["incoming_links"]:
            print(f"     ← linked by: {_titles(r['incoming_links'])}")
        if r["outgoing_links"]:
            print(f"     → links to: {_titles(r['outgoing_links'])}")
        if r["ancestry"]:
            print(f"     path: {' → '.join(a['title'] for a in r['ancestry'])}")
        if r["siblings"]:
            print(f"     siblings: {_titles(r['siblings'])}")
        print()


def _render_images(results):
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


# ── command handlers ───────────────────────────────────────────────────────

def _init(args):
    settings = _settings(args)
    logger.info("initializing database at %s (dim=%d)",
                settings.db_path, settings.embedding_dim)

    def render(pins):
        print(f"[OK] database ready: {pins['db_path']} "
              f"(embedding_dim={pins['embedding_dim']}, model={pins['model_id']})")

    return _out(args, "init", _router(args).init(), render)


def _model_info(args):
    """Show model cache status without loading the model."""
    settings = _settings(args)
    info = OKFRouter.model_info(model_id=settings.model_id,
                                cache_dir=settings.cache_dir,
                                device=settings.device,
                                precision=settings.precision)
    if args.converters:
        from okfgraph.components.converters import converter_status
        info = {**info, "converters": converter_status(
            cache_dir=settings.cache_dir)}

    def render(info):
        print(f"model: {info['model_id']}")
        print(f"repo: {info['repo']} ({info['precision']})")
        print(f"cache: {info['cache_dir']}")
        if info["cached"]:
            print("status: cached")
            print(f"path: {info['snapshot_path']}")
            print(f"size: {info['disk_usage_bytes'] / (1024 ** 3):.2f} GB")
        else:
            print("status: not cached (will download on first use)")
        conv = info.get("converters")
        if conv is not None:
            if conv["available"]:
                reps = conv["reports"]
                n = sum(1 for r in reps if r.get("cached"))
                print(f"converters: {n}/{len(reps)} families cached")
                for r in reps:
                    if r.get("cached"):
                        gb = r["disk_usage_bytes"] / (1024 ** 3)
                        print(f"  [cached] {r['repo']} ({gb:.2f} GB)")
                    else:
                        print(f"  [fetch-on-use] {r['repo']}")
            else:
                print(conv["note"])

    return _out(args, "model_info", info, render)


def _import(args):
    """Delta-aware import: every configured root (--all) or explicit files."""
    settings = _settings(args)
    mode = args.mode or settings.mode
    force = args.force
    if args.import_all:
        if args.files:
            raise OKFError("BAD_VALUE", "import --all takes no FILES (use --bundle-path DIR to pin one tree)",
                           op="import_bundle", fields={"files": args.files})
        router = _router(args)
        result = router.import_bundle(
            args.bundle_path,
            batch_size=args.batch_size or settings.batch_size,
            mode=mode,
            prune_missing=args.prune_missing,
            force=force,
        )

        def render(result):
            print(f"[OK] imported {len(result['concept_ids'])} concept(s) (mode: {mode})")
            for cid in result["concept_ids"]:
                n = result["images"].get(cid, 0)
                print(f"  {cid}" + (f"  [{n} image(s)]" if n else ""))

        return _out(args, "import_bundle", result, render)

    if not args.files:
        raise OKFError("MISSING_PARAM", "import needs FILES or --all",
                       op="import_file")
    for flag, value in (("--bundle-path", args.bundle_path),
                        ("--prune-missing", args.prune_missing)):
        if value:
            raise OKFError("BAD_VALUE", f"{flag} only applies with --all",
                           op="import_file", fields={"flag": flag})
    missing = [f for f in args.files if not Path(f).is_file()]
    if missing:
        # Refuse up front: never import half a file list.
        raise OKFError("FILE_NOT_FOUND", f"file(s) not found: {', '.join(missing)}",
                       op="import_file", fields={"files": missing})
    router = _router(args)
    results = [router.import_file(f, mode=mode, force=force) for f in args.files]

    def render(results):
        for r in results:
            n = r["images"]
            print(f"[OK] imported: {r['concept_id']}"
                  + (f" ({n} image(s), mode: {mode})" if n else ""))

    return _out(args, "import_file", results, render)


def _search(args):
    """Unified search: concepts (default), chunks, or images."""
    results = _router(args).search(
        query=args.query,
        target=args.target,
        limit=args.limit,
        concept_type=args.concept_type,
        tags=_csv(args.tags),
        parent_id=args.parent_id,
        include_chunks=args.include_chunks,
        max_chunks_per_doc=args.max_chunks_per_doc,
        expand=args.expand,
        context_hops=args.context_hops,
        hub_rerank=args.hub_rerank,
        hub_weight=args.hub_weight,
        rank=args.rank,
    )
    if args.target == "images":
        render = _render_images
    elif args.target == "chunks":
        render = (_render_hub_chunks if args.hub_rerank
                  else _render_expanded if args.expand else _render_chunks)
    else:
        render = _render_concepts
    return _out(args, "search", results, render)


def _traverse(args):
    """Unified traversal: relationships, directory listing, or shortest path."""
    start, target = args.start_id or "", args.target
    results = _router(args).traverse(
        start,
        relationship=args.relationship,
        direction=args.direction,
        depth=args.depth,
        node_type=args.node_type,
        target=target,
        max_path_length=args.max_path_length,
    )

    def render(results):
        if not start:
            if not results:
                print("Directory is empty.")
                return
            print("Contents of '(root)':\n")
            for item in results:
                icon = "[D]" if item["type"] == "Directory" else "[F]"
                print(f"  {icon} {item['title']} ({item['type']})")
                print(f"     id: {item['id']}")
        elif target:
            if not results:
                print(f"No path found between '{start}' and '{target}'.")
                return
            print(f"Path ({len(results)} nodes):")
            for i, n in enumerate(results, 1):
                print(f"  {i}. {n.get('title', '?')} ({n.get('type', '?')})")
                print(f"     id: {n['id']}")
        elif not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} node(s):\n")
            for r in results:
                print(f"  {r['id']} ({r['type']})")
                if r.get("title"):
                    print(f"    title: {r['title']}")

    return _out(args, "traverse", results, render)


def _read(args):
    """Unified read: body (default), chunks, document, or context."""
    cid, include = args.concept_id, args.include
    if args.output_path and (include != "document" or args.max_tokens):
        raise OKFError("BAD_VALUE", "--output-path only applies to --include document",
                       op="read", fields={"include": include})
    data = _router(args).read(cid, include=include, max_tokens=args.max_tokens)

    def render_budget(reading):
        flag = " (truncated)" if reading["truncated"] else ""
        print(f"[{reading['used']}/{reading['budget']} tokens{flag}] {cid}\n")
        for sec in reading["sections"]:
            print(f"## {sec['title'] or sec['id']} [{sec['kind']} | {sec['id']}]")
            print(sec["text"])
            print()

    def render_chunks(chunks):
        if not chunks:
            print("No chunks found for this concept.")
            return
        print(f"Chunks for '{cid}' ({len(chunks)} total):\n")
        for c in chunks:
            print(f"  #{c['chunk_index']} [{c['block_type']}] {c['chunk_text'][:120]}")

    def render_document(payload):
        if args.output_path:
            Path(args.output_path).write_text(payload["markdown"], encoding="utf-8")
            print(f"[OK] Reconstructed document written to {args.output_path}")
        else:
            print(payload["markdown"])

    def render_context(ctx):
        if ctx["incoming_links"]:
            print("Linked by:")
            for link in ctx["incoming_links"]:
                print(f"  {link.get('title', link.get('id', '?'))} (id: {link['id']})")
        if ctx["outgoing_links"]:
            print("Links to:")
            for link in ctx["outgoing_links"]:
                print(f"  {link.get('title', link.get('id', '?'))} (id: {link['id']})")
        if ctx["ancestry"]:
            print(f"Path: {' → '.join((a.get('title') or a['id']) for a in ctx['ancestry'])}")
        if ctx["siblings"]:
            print("Siblings:")
            for s in ctx["siblings"]:
                print(f"  {s['title']} ({s['type']})")
                print(f"     id: {s['id']}")
        if not any(ctx.values()):
            print(f"No context found for '{cid}'.")

    def render_body(concept):
        concept = dict(concept)
        body = concept.pop("body", "")
        print(json.dumps(concept, indent=2, default=str))
        if body:
            print(f"\n--- BODY ---\n{body}")

    if args.max_tokens:
        render = render_budget
    else:
        render = {"chunks": render_chunks, "document": render_document,
                  "context": render_context}.get(include, render_body)
    return _out(args, "read", data, render)


def _export(args):
    """Export the whole bundle (--all) or a single concept (--concept-id)."""
    if args.export_all:
        if args.concept_id:
            raise OKFError("BAD_VALUE", "export takes --all or --concept-id, not both",
                           op="export_bundle")
        result = _router(args).export_bundle(
            args.output_dir,
            directory_id=args.directory_id,
            concept_type=args.concept_type,
            tags=_csv(args.tags),
            flavor=args.flavor,
        )
        return _out(args, "export_bundle", result, lambda r: print(
            f"[OK] Exported {len(r['concept_ids'])} concepts to "
            f"{r['output_dir']} (flavor: {r['flavor']})"))
    if not args.concept_id:
        raise OKFError("MISSING_PARAM", "export needs --all or --concept-id ID",
                       op="export_concept")
    ignored = [f for f, v in (("--directory-id", args.directory_id),
                              ("--concept-type", args.concept_type),
                              ("--tags", args.tags)) if v]
    if ignored:
        raise OKFError("BAD_VALUE", f"single-concept export ignores: {', '.join(ignored)}",
                       op="export_concept", fields={"ignored": ignored},
                       remedy="filters apply to --all only")
    result = _router(args).export_concept(args.concept_id, output_dir=args.output_dir,
                                          flavor=args.flavor)
    return _out(args, "export_concept", result, lambda r: print(
        f"[OK] Exported {r['concept_id']} → {r['path']} (flavor: {args.flavor})"))


def _broken_links(args):
    def render(broken):
        if not broken:
            print("No broken links found.")
            return
        print(f"Found {len(broken)} broken link(s):")
        for link in broken:
            print(f"  {link['source']} → {link['target']}")

    return _out(args, "list_broken_links", _router(args).list_broken_links(), render)


def _repair_links(args):
    return _out(args, "repair_links", _router(args).repair_links(),
                lambda r: print(f"[OK] repaired {r['repaired']} link(s)"))


def _lint(args):
    """Pre-import bundle gate: frontmatter + link validation.

    Router-free (no DB, no model cold-boot). Exit 0 = clean (warnings ok),
    1 = errors (LINT_ERRORS, report still rendered), 2 = not a directory.
    """
    return _out(args, "lint", OKFRouter.lint(args.dir or "."), _render_lint)


def _produce(args):
    """Generate a bundle from a data source, then lint it (router-free).

    Exit 0 = produced + lint-clean, 1 = produced but lint errors, 2 =
    usage (unknown producer, unreadable source).
    """
    result = OKFRouter.produce(
        args.from_, args.source, args.output_dir or ".",
        prefix=args.prefix, overwrite=args.overwrite,
    )

    def render(result):
        print(f"[OK] {result['producer']}: {result['files']} file(s) "
              f"from {result['root']}/")
        _render_lint(result["lint"], label="lint: ")
        return 0 if result["lint"]["clean"] else 1

    rc = _out(args, "produce", result, render)
    return rc if rc is not None else (0 if result["lint"]["clean"] else 1)


def _diff(args):
    """Structural diff: snapshot (dir vs dir) or drift (graph vs dir/roots).

    Exit 0 = identical, 1 = different (DIFF_DIFFERENT, report rendered).
    """
    old, new = args.old, args.new
    if old and new:
        result = OKFRouter.diff_dirs(old, new)  # snapshot: router-free
    else:
        result = _router(args).diff(old, new)
    return _out(args, "diff", result, _render_diff)


def _detach(args):
    """End the mirror relationship: the DB becomes the artifact."""
    report = _router(args).detach(
        bundle_path=args.bundle_path,
        verify=not args.no_verify,
        force=args.force,
    )

    def render(report):
        if report["verified"]:
            print(f"verified {report['file_count']} source file(s) against the graph")
        else:
            print("detached without verification")
        if report["mismatched"]:
            print(f"acknowledged {len(report['mismatched'])} mismatched file(s)")
        for u in report["untracked"][:10]:
            print(f"  untracked (WILL-NOT-SURVIVE): {u['path']}")
        for n in report["non_source_files"][:10]:
            print(f"  source-only (WILL-NOT-SURVIVE): {n}")
        if report["already_sourceless"]:
            print(f"{report['already_sourceless']} tracked source(s) already "
                  "live only in the graph")
        prov = report["provenance"]
        for row in prov if isinstance(prov, list) else [prov]:
            label = f"@{row['alias']}" if row.get("alias") else "<primary>"
            print(f"[OK] detached {label} from {row['path']} "
                  f"({row['file_count']} file(s) at detach).")
        print("Reads/search/export/recover keep working; imports refuse without --force.")

    return _out(args, "detach", report, render)


def _doctor(args):
    """Scored health scan, optionally with safe --fix repairs.

    Exit 1 with --strict when any finding exists (DOCTOR_FINDINGS).
    """
    result = _router(args).doctor(
        stale_days=args.stale_days, fix=args.fix, strict=args.strict,
        cache_dir=_settings(args).cache_dir,
    )
    return _out(args, "doctor", result, _render_doctor)


def _reindex(args):
    def render(r):
        print("[OK] search indexes rebuilt" if r["rebuilt"]
              else "Search indexes already up to date; nothing to do.")

    return _out(args, "reindex", _router(args).reindex(if_dirty=args.if_dirty), render)


def _ingest(args):
    """Unified ingest: markdown file, PDF/Office, or raw thoughts."""
    settings = _settings(args)
    router = _router(args)
    kind = args.kind
    on_page = None
    if kind == "pdf" and not args.json:
        def on_page(idx, total):
            print(f"  page {idx + 1}/{total}", end="\r", file=sys.stderr)

    result = router.ingest(
        kind,
        md_path=args.md_path,
        pdf_path=args.pdf_path,
        thoughts=args.thoughts,
        topic=args.topic,
        concept_id=args.concept_id,
        title=args.title,
        description=args.description,
        tags=_csv(args.tags),
        mode=args.mode or settings.mode,
        routing_mode=args.routing_mode,
        extract_images=not args.no_extract_images,
        auto_import=args.auto_import,
        output_dir=args.output_dir,
        batch_size=args.batch_size or settings.batch_size,
        prune_missing=args.prune_missing,
        force=args.force,
        on_page=on_page,
    )
    if on_page is not None:
        print(file=sys.stderr)  # end the \r progress line

    def render(result):
        if kind == "md":
            print(f"[OK] Imported {result['concept_id']} ({result['chunk_count']} chunks)")
        elif kind == "thoughts":
            print(f"[OK] Stored thought {result['concept_id']}")
        elif args.auto_import:
            print(f"[OK] Imported {len(result['concept_ids'])} concept(s) "
                  f"from {result['page_count']} page(s)")
            for cid in result["concept_ids"]:
                n = len(router.image_mgr.list_images(cid))
                print(f"  {cid}" + (f"  [{n} image(s)]" if n else ""))
        else:
            out_dir = Path(result["md_path"]).parent
            print(f"[OK] Converted to {result['md_path']} (assets: {result['image_dir']})")
            print(f"Import with: okf import --all --bundle-path {out_dir}")

    return _out(args, "ingest", result, render)


def _images(args):
    """List image assets attached to a concept."""
    cid = args.concept_id

    def render(imgs):
        if not imgs:
            print("No images attached.")
            return
        print(f"Images for '{cid}' ({len(imgs)} total)")
        for im in imgs:
            alt = im.get("alt_text") or "(no alt-text)"
            print(f"  [{im.get('embed_route')}] {im.get('file_name')} — {alt}")
            print(f"     id: {im.get('id')}")

    return _out(args, "list_images", _router(args).list_images(cid), render)


def _image(args):
    """Fetch one image asset: metadata + optional bytes (--output-path)."""
    row = _router(args).get_image(args.asset_id)
    if args.output_path:
        if row["data"] is None:
            raise OKFError("UNKNOWN_ASSET", f"image asset '{args.asset_id}' has no stored bytes",
                           op="get_image", fields={"asset_id": args.asset_id})
        Path(args.output_path).write_bytes(base64.b64decode(row["data"]))

    def render(row):
        if args.output_path:
            print(f"[OK] Image bytes written to {args.output_path}")
        alt = row.get("alt_text") or "(no alt-text)"
        print(f"[{row.get('embed_route')}] {row.get('file_name')} — {alt}")
        print(f"   id: {row.get('id')}")
        print(f"   mime: {row.get('mime_type')}")

    return _out(args, "get_image", row, render)


def _deleted_list(args):
    """List soft-deleted concepts with recovery status."""
    def render(deleted):
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

    return _out(args, "list_deleted", _router(args).list_deleted(), render)


def _deleted_recover(args):
    """Recover a soft-deleted concept (NOT_RECOVERABLE -> exit 1)."""
    return _out(args, "recover_deleted", _router(args).recover_deleted(args.concept_id),
                lambda r: print(f"[OK] Recovered concept '{r['concept_id']}'."))


def _deleted_delete(args):
    """Soft-delete a concept (op delete; file-backed + existing source
    is a BAD_VALUE refusal naming the file and the pruning remedy)."""
    def render(r):
        print(f"[OK] Deleted concept '{r['concept_id']}' "
              f"(soft-deleted; recoverable until {r['recoverable_until']}).")
        print("     'okf deleted-recover' undoes it; file-backed concepts "
              "are refused while their source exists.")

    return _out(args, "delete", _router(args).delete(args.concept_id), render)


def _deleted_purge(args):
    """Permanently delete expired soft-deleted concepts."""
    return _out(args, "purge_deleted",
                _router(args).purge_deleted(older_than=args.older_than),
                lambda r: print(f"[OK] Permanently deleted {r['purged']} expired concept(s)."))


# ── interactive shell ──────────────────────────────────────────────────────

_SHELL_BANNER = """OKF Interactive Shell
========================================
Commands:
  import <file>              — import single OKF file (same as okf import)
  import-bundle [path]       — import entire bundle (same as okf import --all)
  search [target:]<query> [hub|expand|type:X|tags:a,b|parent:Y] — search
  read <id> [chunks|document|context] — read a concept (default: body)
  traverse [id] [REL] [DIR] [depth] — traverse; 2 ids = shortest path
  images <concept_id>        — list images attached to a concept
  delete <id>                — soft-delete a concept (file-backed + existing
                               source is refused; recoverable for 24h)
  export-bundle <output_dir> — export all concepts
  export <id> <output_dir>   — export single concept
  ingest <file> [--no-auto-import] — ingest .md, or convert + import a PDF/Office file
  model-info                 — show model cache status
  broken-links / repair-links — link hygiene
  help                       — show this help
  quit / exit                — exit shell
Any CLI flag works after the command, e.g. search --rank ppr --limit 5.
========================================"""


def _shell_argv(cmd: str, rest: str):
    """Translate one shell line into argv for :func:`build_parser`.

    The shell keeps its ergonomic shorthand (``chunks:`` prefixes, bare
    ``expand``/``hub`` modifiers, the two-forms traverse) but the
    resulting argv always resolves through the SAME subparsers the CLI
    uses — flags have exactly one definition and one renderer. Tokens
    starting with ``--`` pass through unchanged. Returns ``None`` for
    words that are not shell verbs.
    """
    tokens = rest.strip().split()
    flags = [t for t in tokens if t.startswith("-")]
    words = [t for t in tokens if not t.startswith("-")]
    if cmd == "import":
        return ["import"] + tokens if tokens else None
    if cmd == "import-bundle":
        argv = ["import", "--all"]
        if words:
            argv += ["--bundle-path", " ".join(words)]
        return argv + flags
    if cmd == "search":
        if not words:
            return None
        target = None
        query = words[0]
        if ":" in query and query.split(":")[0] in ("concepts", "chunks", "images"):
            target, query = query.split(":", 1)
        extra, terms = [], [query]
        for t in words[1:]:
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
                terms.append(t)
        if "--expand" in extra or "--hub-rerank" in extra:
            target = target or "chunks"
        argv = ["search", " ".join(terms)]
        if target and target != "concepts":
            argv += ["--target", target]
        return argv + extra + flags
    if cmd == "read":
        if not words:
            return None
        argv = ["read", words[0]]
        if len(words) > 1:
            argv += ["--include", words[1]]
        return argv + flags
    if cmd == "traverse":
        rels = ("CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET")
        if not words:
            return ["traverse"] + flags
        argv = ["traverse", words[0]]
        if len(words) == 2 and words[1].upper() not in rels:
            return argv + ["--target", words[1]] + flags
        if len(words) > 1:
            argv += ["--relationship", words[1].upper()]
        if len(words) > 2:
            argv += ["--direction", words[2].upper()]
        if len(words) > 3:
            argv += ["--depth", words[3]]
        return argv + flags
    if cmd == "images":
        return ["images"] + tokens if words else None
    if cmd == "export-bundle":
        return ["export", "--all", "--output-dir", words[0]] + flags if words else None
    if cmd == "export":
        if len(words) != 2:
            return None
        return ["export", "--concept-id", words[0], "--output-dir", words[1]] + flags
    if cmd == "ingest":
        if not words:
            return None
        path = words[0]
        kind = "md" if path.lower().endswith((".md", ".markdown")) else "pdf"
        return ["ingest", "--kind", kind, f"--{kind}-path", path] + flags
    if cmd in ("model-info", "broken-links", "repair-links",
               "deleted-list", "deleted-recover", "deleted-purge"):
        return [cmd] + flags
    if cmd == "delete" and flags:
        return ["delete", *flags]
    return None


def _shell(args, router=None):
    """Interactive shell: a REPL over the same subcommands as the CLI.

    Every line is translated by :func:`_shell_argv` and handed to
    ``build_parser()`` + the same handler as ``okf <verb>``: flags exist
    once and each op has one renderer. The session's own router is reused
    for every line (a fresh ``_router`` per line would clash on the
    ladybug file lock).
    """
    router = router or _router(args)
    base = {k: v for k, v in vars(args).items() if k != "command"}
    parser = build_parser()
    print(_SHELL_BANNER)

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
                print(_SHELL_BANNER)
                continue
            argv = _shell_argv(cmd, rest)
            if argv is None:
                print(f"Unknown command: {cmd}. Type 'help' for usage.")
                continue
            # Parse with the session's settings pre-seeded: generated global
            # flags are SUPPRESS-defaulted, so only args typed on the line
            # overwrite them.
            try:
                sub = parser.parse_args(argv, namespace=argparse.Namespace(**base))
            except SystemExit:
                print(f"Error: bad usage for '{cmd}'.")
                continue
            # Same error contract as one-shot commands; the REPL survives.
            _main_catchall(sub)
    finally:
        _INJECTED_ROUTER.clear()


# ── argument parser ────────────────────────────────────────────────────────

def _global_options_epilog() -> str:
    """Render the global flags once for top-level ``okf --help``."""
    probe = argparse.ArgumentParser(prog="okf")
    _add_global(probe, mark=False)
    text = probe.format_help()
    try:
        body = text.split("options:", 1)[1]
    except IndexError:  # pragma: no cover - Python <3.11 wording
        body = text.split("optional arguments:", 1)[1]
    return "Global options (every command; may also come from okfgraph.toml):" + body


_MODES = ["text", "optional", "omni"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="okf",
        description="OKF Knowledge Graph CLI — LadybugDB + Jina v5 embeddings",
        epilog=_global_options_epilog(),
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run", parser_class=_SubParser)

    def command(name, help):
        p = sub.add_parser(name, help=help)
        _add_global(p)
        return p

    command("init", "Initialize database and schema")
    p = command("model-info", "Show model cache status")
    p.add_argument("--converters", action="store_true",
                   help="Include converter-model cache status "
                   "(bobine >=0.6.0, offline)")

    p = command("import", "Import OKF files (delta-aware)")
    p.add_argument("files", nargs="*", help="Files to import")
    p.add_argument("--all", action="store_true", dest="import_all",
                   help="Import entire bundle (omit --bundle-path to import every "
                   "configured root; with --bundle-path, only that tree)")
    p.add_argument("--bundle-path", default=None,
                   help="Pin a single tree for this import (with --all); "
                        "the primary root still comes from --bundle-root/TOML")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Encode batch size (default: [import] batch_size, 32)")
    p.add_argument("--mode", default=None, choices=_MODES,
                   help="Image ingestion mode: text (captions), optional (vision for "
                   "images lacking alt-text), omni (vision for every image). "
                   "Default: [import] mode, text")
    p.add_argument("--prune-missing", action="store_true",
                   help="With --all: also drop concepts whose source files were deleted "
                   "(removes concept, chunks, links, and orphaned image assets)")
    p.add_argument("--force", action="store_true",
                   help="Bypass the detached-graph refusal: re-attach when the bundle "
                   "matches the recorded source tree, or acknowledge an explicitly "
                   "addressed file import.")

    p = command("search", "Search concepts, chunks, or images")
    p.add_argument("query", help="Search query")
    p.add_argument("--target", default="concepts", choices=["concepts", "chunks", "images"],
                   help="What to search (default: concepts)")
    p.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    p.add_argument("--concept-type", help="Concept type filter (concepts; plain chunks)")
    p.add_argument("--tags", help="Comma-separated tag filters, all must match (concepts; plain chunks)")
    p.add_argument("--parent-id", help="Parent directory ID (concepts; plain chunks)")
    p.add_argument("--include-chunks", action="store_true",
                   help="Concepts: include matched chunks per hit")
    p.add_argument("--max-chunks-per-doc", type=int, default=3,
                   help="Plain chunks: cap hits per source document (default: 3)")
    p.add_argument("--expand", action="store_true", help="Chunks: attach graph neighborhood to each hit")
    p.add_argument("--context-hops", type=int, default=1, help="Neighborhood depth with --expand (default: 1)")
    p.add_argument("--hub-rerank", action="store_true", help="Chunks: rerank by graph hub score (wins over --expand)")
    p.add_argument("--hub-weight", type=float, default=0.3,
                   help="Hub weight for --hub-rerank or --rank hub (default: 0.3)")
    p.add_argument("--rank", default="none", choices=["none", "hub", "ppr"],
                   help="Concepts: ranking — none (RRF order), hub (blend incoming-link "
                   "authority), ppr (model-free lexical-seed PPR, no ONNX load). "
                   "Default: none")

    p = command("read", "Read a concept: body, chunks, document, or context")
    p.add_argument("concept_id", help="Concept ID")
    p.add_argument("--include", default="body", choices=["body", "chunks", "document", "context"],
                   help="What to return (default: body)")
    p.add_argument("--output-path", help="Write --include document to this file (default: stdout)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Token budget: assemble self + PPR-ranked neighbours "
                   "(index-first for context), truncating to fit")

    p = command("traverse", "Traverse relationships, list directories, find paths")
    p.add_argument("start_id", nargs="?", default="", help="Starting concept or directory ID (empty = root listing)")
    p.add_argument("--relationship", default="CONTAINS", choices=["CONTAINS", "LINKS_TO", "PART_OF", "INCLUDES_ASSET"])
    p.add_argument("--direction", default="OUTGOING", choices=["OUTGOING", "INCOMING", "BOTH"])
    p.add_argument("--depth", type=int, default=1, help="Max depth (1-5)")
    p.add_argument("--node-type", help="Target node type filter")
    p.add_argument("--target", default=None, help="Find shortest path from start_id to this ID instead of traversing")
    p.add_argument("--max-path-length", type=int, default=6, help="Max path length with --target (default: 6)")

    p = command("ingest", "Add content: markdown file, PDF/Office file, or thoughts")
    p.add_argument("--kind", required=True, choices=["md", "pdf", "thoughts"],
                   help="What to ingest")
    p.add_argument("--md-path", default=None, help="Markdown file (--kind md)")
    p.add_argument("--pdf-path", default=None, help="File to convert (--kind pdf): PDF or Office (docx/xlsx/pptx, legacy doc/xls/ppt)")
    p.add_argument("--thoughts", default=None, help="Raw reasoning text (--kind thoughts)")
    p.add_argument("--topic", default=None, help="Topic (--kind thoughts)")
    p.add_argument("--concept-id", default=None, help="Explicit concept ID (--kind md/thoughts; slashes are namespaces; thoughts default to thoughts/<topic>/<ts>_<id>)")
    p.add_argument("--title", default=None, help="Title override (--kind md)")
    p.add_argument("--description", default=None, help="Description override (--kind md)")
    p.add_argument("--tags", default=None, help="Comma-separated tags (--kind md/thoughts)")
    p.add_argument("--no-auto-import", action="store_false", dest="auto_import",
                   help="Convert only (--kind pdf): write markdown, skip the import")
    p.add_argument("--output-dir", default=None,
                   help="Where convert-only output goes (--kind pdf --no-auto-import; "
                   "default: next to the source file)")
    p.add_argument("--routing-mode", default="auto",
                   choices=["auto", "surgical", "always", "never"],
                   help="Converter ONNX routing (--kind pdf, default: auto; never = fast path)")
    p.add_argument("--mode", default=None, choices=_MODES,
                   help="Image ingestion mode: text (captions), optional/omni (ONNX "
                   "vision). Default: [import] mode, text")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Encode batch size during auto-import (default: [import] batch_size, 32)")
    p.add_argument("--prune-missing", action="store_true",
                   help="Drop concepts whose source files vanished during auto-import")
    p.add_argument("--force", action="store_true",
                   help="Bypass the detached-graph refusal for single-file / thought "
                   "ingest (never re-attaches; PDF auto-import cannot re-attach "
                   "from a temp dir).")
    p.add_argument("--no-extract-images", action="store_true",
                   help="Do not extract embedded images (--kind pdf)")

    p = command("export", "Export the bundle (--all) or one concept (--concept-id)")
    p.add_argument("--all", action="store_true", dest="export_all", help="Export entire bundle")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--concept-id", help="Concept ID (single export)")
    p.add_argument("--concept-type", help="Concept type filter (--all)")
    p.add_argument("--tags", help="Comma-separated tag filters, all must match (--all)")
    p.add_argument("--directory-id", help="Only concepts under this directory (--all)")
    p.add_argument("--flavor", default="okf", choices=["okf", "obsidian"],
                   help="Link flavor: okf ([t](id.md) + index files) or obsidian "
                   "([[Title]] wikilinks, no index files). Default: okf")

    p = command("diff", "Structural diff: concepts/edges/broken-link deltas")
    p.add_argument("old", nargs="?", default=None,
                   help="Old side: bundle dir (with NEW: snapshot; alone: drift vs graph)")
    p.add_argument("new", nargs="?", default=None, help="New side: bundle dir (snapshot mode)")

    p = command("doctor", "Health scan: score, findings, safe --fix")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 when any finding exists (CI gate)")
    p.add_argument("--fix", action="store_true",
                   help="Apply safe repairs (link re-points, timestamp normalization; "
                   "never touches reviewed:true concepts)")
    p.add_argument("--stale-days", type=int, default=365,
                   help="Age threshold for 'stale' findings (default: 365)")

    p = command("lint", "Validate bundle frontmatter + links before import")
    p.add_argument("dir", nargs="?", default=None,
                   help="Bundle directory (default: current directory)")

    p = command("produce", "Generate a bundle from a data source")
    p.add_argument("--from", dest="from_", required=True,
                   help="Producer name (today: sqlite)")
    p.add_argument("--source", required=True,
                   help="Source to read (sqlite: path to the .db file)")
    p.add_argument("--output-dir", default=None,
                   help="Bundle root to write under (default: .); files go "
                   "to <output-dir>/<prefix>/ so lint + import agree on ids")
    p.add_argument("--prefix", default=None,
                   help="Namespace directory (default: the producer's own)")
    p.add_argument("--overwrite", action="store_true",
                   help="Rewrite existing producer output files")

    command("shell", "Interactive REPL")
    command("broken-links", "List broken (orphan) links")
    command("repair-links", "Repair broken links by re-checking targets")

    p = command("reindex", "Rebuild vector + FTS search indexes")
    p.add_argument("--if-dirty", action="store_true",
                   help="Only rebuild if data changed since the last index build")

    command("deleted-list", "List soft-deleted concepts")
    p = command("deleted-recover", "Recover a soft-deleted concept")
    p.add_argument("concept_id", help="Concept ID to recover")
    p = command("delete", "Soft-delete a concept (recoverable for 24h; refuses "
                           "file-backed concepts whose source still exists)")
    p.add_argument("concept_id", help="Concept ID to delete")
    p = command("deleted-purge", "Permanently delete expired soft-deleted concepts")
    p.add_argument("--older-than", type=int, default=None, help="Override recovery window (seconds)")

    p = command("detach", "End the mirror relationship: the DB becomes the artifact")
    p.add_argument("--bundle-path", default=None,
                   help="Bundle tree to detach (default: the primary root)")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the sources-vs-graph fidelity check (for already-removed trees)")
    p.add_argument("--force", action="store_true",
                   help="Acknowledge mismatches / untracked files / source-only "
                   "artifacts (WILL-NOT-SURVIVE) and detach anyway")

    p = command("images", "List image assets attached to a concept")
    p.add_argument("concept_id", help="Concept ID")

    p = command("image", "Fetch one image asset (metadata; --output-path writes bytes)")
    p.add_argument("asset_id", help="Image asset ID")
    p.add_argument("--output-path", default=None,
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
    "delete": _deleted_delete,
    "deleted-purge": _deleted_purge,
    "detach": _detach,
}


# ── dispatch ───────────────────────────────────────────────────────────────

def _emit_cli_error(args, err):
    """Error adapter (§4/D6): the error envelope on stdout under --json
    (outcome reports in ``data``), else the outcome report rendered; the
    ``[ERROR]`` line always on stderr. Returns the error's exit code."""
    op = err.op or args.command
    if getattr(args, "json", False):
        print(dumps_envelope(envelope(op, error=err)))
    elif err.data is not None and args.command in _OUTCOME_RENDERERS:
        _OUTCOME_RENDERERS[args.command](err.data)
    print(f"[ERROR] {err.titles()}", file=sys.stderr)
    return err.exit_code


def _main_catchall(args):
    """Dispatch one command with the §4 catch-all: OKFError → its exit code,
    KeyboardInterrupt → 130, everything else → INTERNAL (never a bare
    traceback). Returns the process exit code."""
    try:
        return _COMMANDS[args.command](args) or 0
    except OKFError as err:
        return _emit_cli_error(args, err)
    except KeyboardInterrupt:
        print("[ERROR] interrupted (130)", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — the §4 catch-all *is* the contract
        logger.exception("INTERNAL in %s", args.command)
        # The console traceback is easily lost (`2>&1 | tail -n 3`); the
        # crash file keeps every frame and the [ERROR] line names it.
        from okfgraph.crash import write_crash_report
        path = write_crash_report(exc, command=args.command)
        err = internal_error(
            exc, op=args.command,
            remedy=(f"this is a bug — full traceback: {path}; report it with that file"
                    if path else
                    "this is a bug — report it with the traceback above"))
        if path:
            err.fields["crash_report"] = path
        return _emit_cli_error(args, err)


def _force_utf8_streams():
    """Windows consoles default to cp1252 — graph content (emojis,
    `\u2190`, `\u2264`, German umlauts in names) crashes every print()
    with UnicodeEncodeError typed as INTERNAL. Ship UTF-8 like every
    modern CLI does; `errors="replace"` keeps unencodable bytes visible
    instead of aborting the render. Guarded: pytest's capsys replaces
    the streams with StringIO, which has no reconfigure."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass  # already detached / redirected; render as-is


def main():
    _force_utf8_streams()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    _setup_logging(
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
        log_file=getattr(args, "log_file", ""),
        console_level=(logging.WARNING
                       if getattr(args, "json", False)
                       and not getattr(args, "verbose", False)
                       and not getattr(args, "quiet", False)
                       else None),
    )

    profiler = cProfile.Profile() if getattr(args, "profile", False) else None
    if profiler is not None:
        profiler.enable()
    try:
        ret = _main_catchall(args)
    finally:
        _close_routers()
        if profiler is not None:
            profiler.disable()
            import pstats
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
            print(stream.getvalue(), file=sys.stderr)
        _teardown_logging()
    if ret:
        sys.exit(ret)


if __name__ == "__main__":
    main()
