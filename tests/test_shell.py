"""§5 CLI row — the interactive shell routes lines through ``build_parser()``.

Before: the shell carried its own hand-rolled ``elif`` dispatcher — 13
verbs with duplicated flag handling and renderers that drifted from the
CLI. After: a thin shorthand layer translates each line to argv
(``search chunks:<q> hub`` → ``search --target chunks <q> --hub-rerank``)
and argparse drives the SAME subparsers + handlers as ``okf <verb>``.
One source of truth for flags, one renderer per op.

Model-heavy: the module fixture pays a single encoder boot (one bundle
import); every test reuses that router in-process with piped lines.
"""
import argparse
import builtins
import logging

import pytest

from okfgraph.cli import _OPEN_ROUTERS, _router, _setup_logging, _shell

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def shell_env(tmp_path_factory):
    """One DB + one import (one encoder boot) shared by every shell test."""
    tmp = tmp_path_factory.mktemp("shell")
    bundle = tmp / "bundle"
    bundle.mkdir()
    (bundle / "hub.md").write_text(
        "---\ntitle: Central Hub\ntype: note\nid: hub\n---\n\n"
        "Everything points here.\n\n[[spoke]]\n",
        encoding="utf-8",
    )
    (bundle / "spoke.md").write_text(
        "---\ntitle: Spoke One\ntype: note\nid: spoke\n---\n\n"
        "Points at the hub.\n",
        encoding="utf-8",
    )
    ns = argparse.Namespace(
        db_path=str(tmp / "shell.db"),
        bundle_root=str(bundle),
        enable_chunking=True,
    )
    # Real shells configure logging in main(); in-process we re-apply it
    # per drive so the StreamHandler binds to this test's capsys buffers.
    router = _router(ns)
    result = router.import_bundle(None)
    assert set(result["concept_ids"]) == {"hub", "spoke"}, result
    yield {"ns": ns, "tmp": tmp, "bundle": bundle, "router": router}
    router.close()
    while router in _OPEN_ROUTERS:
        _OPEN_ROUTERS.remove(router)


def _drive(monkeypatch, capsys, caplog, env, lines):
    """Feed lines to the shell REPL; the session ends with 'quit'.

    Logging notes: the CLI renderers log through the ``cli`` logger;
    pytest replaces the root handlers with the caplog handler, so the
    stable place to read them in-process is ``caplog.text`` (merged into
    ``combined``). Direct prints stay in ``out``/``err`` as usual.
    """
    ns, router = env["ns"], env["router"]
    _setup_logging(verbose=False, quiet=False, log_file="")
    logging.getLogger("cli").setLevel(logging.INFO)
    feed = iter(list(lines) + ["quit"])
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(feed))
    # One router per module fixture: injecting it keeps every session on
    # the same file lock (and the same warm encoder) — a fresh _router
    # per drive would clash on the ladybug lock.
    _shell(ns, router=router)
    out, err = capsys.readouterr()
    combined = out + err + caplog.text
    return combined, out, err


class TestParsing:
    """The REPL keeps its ergonomics through the parser layer."""

    def test_unknown_command_survives(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                ["definitely-not-a-verb"])
        assert "definitely-not-a-verb" in combined
        assert "Bye." in combined  # the REPL did not die

    def test_help_shows_commands(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env, ["help"])
        assert "Commands:" in out
        assert "quit" in combined

    def test_quit_and_exit_aliases(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(monkeypatch, capsys, caplog, shell_env, ["exit"])
        assert "Bye." in combined

    def test_blank_line_is_ignored(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(monkeypatch, capsys, caplog, shell_env, ["", "   "])
        assert "Bye." in combined


class TestTraverse:
    def test_root_listing(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env, ["traverse"])
        assert "[F]" in out
        assert "Central Hub" in out

    def test_path_shortcut(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["traverse hub spoke"])
        assert "Path (" in out
        assert "Spoke One" in out

    def test_walk_with_relationship(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["traverse hub LINKS_TO OUTGOING 1"])
        assert "spoke" in out

    def test_walk_default_relationship(self, monkeypatch, capsys, caplog, shell_env):
        # spoke CONTAINS nothing → the CLI renderer's empty answer.
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["traverse spoke"])
        assert "No results found." in out


class TestRead:
    def test_body(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env, ["read hub"])
        assert "--- BODY ---" in out
        assert "Everything points here." in out

    def test_chunks(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["read hub chunks"])
        assert "Chunks for 'hub'" in out
        assert "#0" in out

    def test_document(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["read hub document"])
        assert "points here" in out

    def test_context_links(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["read hub context"])
        assert "Links to:" in out
        assert "Spoke One" in out

    def test_unknown_concept_is_typed(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, err = _drive(monkeypatch, capsys, caplog, shell_env,
                                    ["read no-such-concept"])
        assert "UNKNOWN_CONCEPT" in err
        assert "Bye." in out  # session survived


class TestSearch:
    def test_concepts(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["search Everything points here"])
        assert "Found" in out
        assert "Central Hub" in out

    def test_chunks_prefix(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["search chunks:everything points here"])
        assert "chunk(s)" in out or "chunk result" in out

    def test_chunks_hub_modifier(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["search chunks:everything points here hub"])
        assert "hub=" in out  # the hub-rerank renderer's signature output

    def test_images_prefix(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                  ["search images:anything"])
        assert "No image results found." in out


class TestImportAndExport:
    def test_single_file(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(
            monkeypatch, capsys, caplog, shell_env,
            [f"import {shell_env['bundle'] / 'hub.md'}"])
        assert "imported" in combined
        assert "hub" in combined

    def test_import_bundle(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                ["import-bundle"])
        # The fixture already imported everything, so the delta re-run
        # is clean — the point is the router's import path: it reports
        # the concept count (0 here, delta-clean).
        assert "concept(s)" in combined

    def test_export_bundle(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(
            monkeypatch, capsys, caplog, shell_env,
            [f"export-bundle {shell_env['tmp'] / 'exported'}"])
        assert "[OK] Exported 2 concepts" in out

    def test_export_concept(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, _ = _drive(
            monkeypatch, capsys, caplog, shell_env,
            [f"export spoke {shell_env['tmp'] / 'exported_one'}"])
        assert "[OK] Exported spoke" in out


class TestHousekeeping:
    def test_images_unknown_concept(self, monkeypatch, capsys, caplog, shell_env):
        combined, out, err = _drive(monkeypatch, capsys, caplog, shell_env,
                                    ["images no-such-concept"])
        assert "UNKNOWN_CONCEPT" in err
        assert "Bye." in out

    def test_model_info(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                ["model-info"])
        # The CLI renderer logs (stderr): "status: cached|not cached".
        assert "status: " in combined

    def test_broken_links(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                ["broken-links"])
        assert "broken link" in combined

    def test_repair_links(self, monkeypatch, capsys, caplog, shell_env):
        combined, _, _ = _drive(monkeypatch, capsys, caplog, shell_env,
                                ["repair-links"])
        assert "repaired" in combined
