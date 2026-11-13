# -*- coding: utf-8 -*-
"""New-empty-graph notice and --pdf-path expansion (0.14.2, plan G1/G2).

G1: a graph the process created on this boot warns while it is still empty,
so a mistyped ``OKFGRAPH_DB_PATH`` does not read as a healthy empty graph.
The notice rides on every MCP tool envelope's ``warnings`` (agents read
those), appears in ``/health``, in ``okf doctor``, and in the server log.

G2: ``--pdf-path`` accepts folders and ``*``/``?`` patterns. PowerShell does
not expand patterns for programs, so okfgraph expands them itself, keeping
only the extensions bobine converts, sorted by name.
"""
from __future__ import annotations

import json
import types

import pytest

from okfgraph.cli import PDF_EXTENSIONS, _expand_pdf_paths
from okfgraph.errors import OKFError

from tests._v011 import stub_encoder  # noqa: F401  (fixture import)


# ---- G1: the boot notice ---------------------------------------------------

def _router(tmp_path, name="kb.db", **kwargs):
    from okfgraph.router import OKFRouter

    return OKFRouter(db_path=str(tmp_path / name),
                     bundle_root=str(tmp_path), **kwargs)


def test_a_created_graph_is_noticed_until_something_is_written(tmp_path):
    router = _router(tmp_path)
    try:
        assert router.db_created is True
        notice = router.creation_notice()
        assert notice and "created at" in notice and "still empty" in notice
        assert str((tmp_path / "kb.db").resolve()) in notice
        assert "OKFGRAPH_DB_PATH" in notice

        router.ingest("thoughts", thoughts="a durable thought",
                      topic="notices")
        assert router.concept_count() == 1
        assert router.creation_notice() is None
    finally:
        router.close()


def test_an_existing_graph_has_no_notice(tmp_path):
    first = _router(tmp_path)
    first.ingest("thoughts", thoughts="kept", topic="notices")
    first.close()

    second = _router(tmp_path)
    try:
        assert second.db_created is False
        assert second.creation_notice() is None
    finally:
        second.close()


def test_doctor_reports_created_with_the_absolute_path(tmp_path):
    router = _router(tmp_path)
    try:
        report = router.doctor()["report"]
        row = next(r for r in report["info"] if r["rule"] == "db_created")
        assert "created at" in row["message"]
        assert str((tmp_path / "kb.db").resolve()) in row["message"]
        assert "still empty" in row["message"]

        router.ingest("thoughts", thoughts="now it has content", topic="t")
        report = router.doctor()["report"]
        row = next(r for r in report["info"] if r["rule"] == "db_created")
        assert "1 concepts" in row["message"]
    finally:
        router.close()


def test_mcp_tool_envelopes_carry_the_warning(tmp_path):
    """Every successful tool result warns while the graph is new + empty."""
    from okfgraph import mcp_server

    router = _router(tmp_path)
    try:
        mcp_server._record_creation_notice(router)
        payload = json.loads(mcp_server._tool_result(
            "read", lambda: router.concept_count()))
        assert payload["ok"] is True
        assert any("created at" in w for w in payload["warnings"])

        router.ingest("thoughts", thoughts="content ends the notice",
                      topic="t")
        payload = json.loads(mcp_server._tool_result(
            "read", lambda: router.concept_count()))
        assert payload["warnings"] == []
    finally:
        mcp_server._CREATION_NOTICE = None
        router.close()


def test_mcp_does_not_warn_for_an_existing_graph(tmp_path):
    from okfgraph import mcp_server

    first = _router(tmp_path)
    first.ingest("thoughts", thoughts="old knowledge", topic="t")
    first.close()

    router = _router(tmp_path)
    try:
        mcp_server._record_creation_notice(router)
        payload = json.loads(mcp_server._tool_result(
            "read", lambda: router.concept_count()))
        assert payload["warnings"] == []
    finally:
        mcp_server._CREATION_NOTICE = None
        router.close()


def test_health_reports_created(tmp_path, stub_encoder):
    """A running ``okf serve`` on a fresh path says ``created: true``."""
    import json
    import threading

    from okfgraph.client import OKFClient, serve_file_for
    from okfgraph.serve import serve
    from okfgraph.settings import Settings

    db = tmp_path / "kb.db"
    settings = Settings(db_path=str(db), bundle_root=str(tmp_path))
    ready, stop = threading.Event(), threading.Event()
    # unix:// runs on TCP loopback where Unix sockets are unavailable
    # (Windows); the discovery file carries the real URL either way.
    thread = threading.Thread(
        target=serve, args=(settings, f"unix://{tmp_path / 'okf.sock'}"),
        kwargs={"ready": ready, "stop": stop}, daemon=True)
    thread.start()
    assert ready.wait(60), "okf serve did not start"
    url = json.loads(serve_file_for(db).read_text(encoding="utf-8"))["url"]
    try:
        health = OKFClient(url).health()
        assert health["created"] is True
        assert health["db_path"] == str(db.resolve())
    finally:
        stop.set()
        thread.join(60)


# ---- G2: folders and patterns ---------------------------------------------

def _touch(folder, name):
    path = folder / name
    path.write_bytes(b"%PDF-1.4 stub")
    return path


def test_pdf_extensions_pin_the_converter_set():
    # bobine's SUPPORTED_EXTENSIONS (src/pipeline.rs). B6 will export it and
    # this test will compare instead of pinning; until then a change here is
    # a deliberate act.
    assert PDF_EXTENSIONS == ("pdf", "docx", "xlsx", "pptx", "doc", "xls", "ppt")


def test_a_folder_expands_to_convertible_files_sorted(tmp_path):
    _touch(tmp_path, "b.PDF")
    _touch(tmp_path, "a.docx")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"png")
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub, "nested.pdf")

    expanded = _expand_pdf_paths([str(tmp_path)])
    names = [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in expanded]
    assert names == ["a.docx", "b.PDF"]      # sorted, no recursion, filtered
    assert all("nested" not in p for p in expanded)


def test_an_empty_folder_names_itself(tmp_path):
    with pytest.raises(OKFError) as excinfo:
        _expand_pdf_paths([str(tmp_path)])
    assert excinfo.value.code == "FILE_NOT_FOUND"
    assert str(tmp_path) in excinfo.value.message


def test_patterns_expand_in_process(tmp_path):
    _touch(tmp_path, "one.pdf")
    _touch(tmp_path, "two.pdf")
    _touch(tmp_path, "three.docx")
    pattern = str(tmp_path / "*.pdf")
    expanded = _expand_pdf_paths([pattern])
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in expanded] == [
        "one.pdf", "two.pdf"]


def test_a_pattern_matching_nothing_names_the_pattern(tmp_path):
    pattern = str(tmp_path / "*.pdf")
    with pytest.raises(OKFError) as excinfo:
        _expand_pdf_paths([pattern])
    assert excinfo.value.code == "FILE_NOT_FOUND"
    assert pattern in excinfo.value.message


def test_a_pattern_keeps_only_convertible_extensions(tmp_path):
    _touch(tmp_path, "keep.pdf")
    (tmp_path / "skip.md").write_text("x", encoding="utf-8")
    expanded = _expand_pdf_paths([str(tmp_path / "*")])
    assert len(expanded) == 1 and expanded[0].endswith("keep.pdf")


def test_a_literal_path_is_passed_through(tmp_path):
    # bobine's own refusal is the better error for a file the user named.
    odd = tmp_path / "photo.png"
    odd.write_bytes(b"png")
    assert _expand_pdf_paths([str(odd)]) == [str(odd)]


@pytest.mark.pdf
def test_cli_folder_ingest_converts_every_file(tmp_path, capsys, monkeypatch):
    """A folder on --pdf-path converts all its documents in one command."""
    pytest.importorskip("bobine")
    import sys as _sys

    from okfgraph.cli import main

    sys_path_pdf = tmp_path / "docs"
    sys_path_pdf.mkdir()
    for name in ("one.pdf", "two.pdf"):
        (sys_path_pdf / name).write_bytes(
            __import__("tests.test_pdf_converter", fromlist=["_min_pdf"])
            ._min_pdf(f"Folder page {name}"))

    monkeypatch.setattr(_sys, "argv", [
        "okf", "ingest", "--kind", "pdf",
        "--pdf-path", str(sys_path_pdf),
        "--no-auto-import", "--routing-mode", "never",
        "--db-path", str(tmp_path / "kb.db"),
        "--bundle-root", str(tmp_path),
        "--json",
    ])
    code = main()
    out = json.loads(capsys.readouterr().out)
    assert code is None                       # None = success
    assert out["ok"] is True
    assert [f["ok"] for f in out["data"]["files"]] == [True, True]
    assert all("md_path" in f["result"] for f in out["data"]["files"])


def test_the_notice_latches_off_and_a_failed_count_says_nothing(tmp_path):
    router = _router(tmp_path)
    try:
        router.ingest("thoughts", thoughts="first concept", topic="t")
        assert router.creation_notice() is None
        # Latched: an empty count later (all concepts deleted) cannot revive it.
        router._count_concepts = lambda: 0
        assert router.creation_notice() is None
    finally:
        router.close()

    fresh = _router(tmp_path, name="fresh.db")
    fresh.close()
    # A closed connection is not evidence of an empty graph.
    assert fresh.creation_notice() is None


def test_okf_serve_mcp_mount_carries_the_warning(tmp_path):
    """okf serve hands its locked router to the MCP lifespan: still warned."""
    import asyncio

    from okfgraph import mcp_server
    from okfgraph.serve import LockedRouter

    router = _router(tmp_path)
    try:
        lifespan = mcp_server.make_lifespan(None, LockedRouter(router))

        async def one_session():
            async with lifespan(None):
                return json.loads(mcp_server._tool_result(
                    "read", lambda: router.concept_count()))

        payload = asyncio.run(one_session())
        assert any("created at" in w for w in payload["warnings"])
    finally:
        mcp_server._CREATION_NOTICE = None
        router.close()


def test_mcp_shutdown_forgets_the_notice(tmp_path, stub_encoder):
    import asyncio

    from okfgraph import mcp_server
    from okfgraph.settings import Settings

    settings = Settings(db_path=str(tmp_path / "kb.db"),
                        bundle_root=str(tmp_path))
    lifespan = mcp_server.make_lifespan(settings)

    async def one_session():
        async with lifespan(None):
            assert mcp_server._CREATION_NOTICE is not None

    asyncio.run(one_session())
    assert mcp_server._CREATION_NOTICE is None


def test_office_owner_and_hidden_files_are_skipped(tmp_path):
    _touch(tmp_path, "report.docx")
    _touch(tmp_path, "~$report.docx")     # Word's owner file while open
    _touch(tmp_path, ".hidden.pdf")
    names = [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
             for p in _expand_pdf_paths([str(tmp_path)])]
    assert names == ["report.docx"]
    names = [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
             for p in _expand_pdf_paths([str(tmp_path / "*.docx")])]
    assert names == ["report.docx"]


def test_a_file_reached_twice_is_converted_once(tmp_path):
    one = _touch(tmp_path, "one.pdf")
    _touch(tmp_path, "two.pdf")
    expanded = _expand_pdf_paths(
        [str(one), str(tmp_path), str(tmp_path / "*.pdf")])
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in expanded] == [
        "one.pdf", "two.pdf"]


def test_brackets_in_a_pattern_are_literal(tmp_path):
    folder = tmp_path / "papers [2024]"
    folder.mkdir()
    _touch(folder, "a.pdf")
    expanded = _expand_pdf_paths([str(folder / "*.pdf")])
    assert len(expanded) == 1 and expanded[0].endswith("a.pdf")


def test_a_failed_tool_carries_the_notice_too(tmp_path):
    """Failures are results: the agent still learns the path is new."""
    from okfgraph import mcp_server
    from okfgraph.errors import OKFError
    from okfgraph.mcp_server import _ToolFailure

    router = _router(tmp_path)
    try:
        mcp_server._record_creation_notice(router)

        def boom():
            raise OKFError("UNKNOWN_CONCEPT", "nope")

        with pytest.raises(_ToolFailure) as excinfo:
            mcp_server._tool_result("read", boom)
        assert any("created at" in w for w in excinfo.value.payload["warnings"])
    finally:
        mcp_server._CREATION_NOTICE = None
        router.close()
