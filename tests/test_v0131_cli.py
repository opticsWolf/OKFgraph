"""0.13.1: CLI paths and setup fixes.

Global flags before the command, missing db folders created only by the
commands that may create a database (init/import/ingest), the default
paths named in help, the skills shipped in the package, and several PDFs
in one ``okf ingest`` process.
"""
from __future__ import annotations

import json

import pytest

from okfgraph import OKFRouter
from okfgraph.cli import (_hoist_globals, _ingest_many_pdfs,
                          _require_db_unless_creating, build_parser, main)
from okfgraph.components.skills import bundled_skills
from okfgraph.errors import OKFError

from tests._v011 import stub_encoder  # noqa: F401  (fixture import)


@pytest.mark.parametrize("argv, expected", [
    (["--db-path", "x.db", "init"], ["init", "--db-path", "x.db"]),
    (["--db-path=x.db", "-q", "search", "a"], ["search", "--db-path=x.db", "-q", "a"]),
    (["--json", "--root", "w=D:/w", "import", "--all"],
     ["import", "--json", "--root", "w=D:/w", "--all"]),
    (["init", "--db-path", "x.db"], ["init", "--db-path", "x.db"]),
    (["--db-path", "init"], ["--db-path", "init"]),     # value, not a command
    (["--nope", "init"], ["--nope", "init"]),           # left to argparse
    (["-h"], ["-h"]),
])
def test_global_flags_move_after_the_command(argv, expected):
    assert _hoist_globals(argv) == expected


def test_hoisted_globals_parse():
    args = build_parser().parse_args(_hoist_globals(["--db-path", "x.db", "--json", "init"]))
    assert args.command == "init" and args.db_path == "x.db" and args.json


def test_help_names_the_default_paths():
    text = build_parser().format_help()
    for needle in (".okfgraph/okfgraph.db", ".okfgraph/bundle", "OKFGRAPH_DB_PATH",
                   "~/.config/okfgraph/config.toml", "before or after the command"):
        assert needle in text
    assert "Jina" not in text


def test_a_missing_db_folder_is_created(tmp_path, stub_encoder, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kb").mkdir()
    OKFRouter(db_path="new/deep/kb.db", bundle_root="kb").close()
    assert (tmp_path / "new" / "deep" / "kb.db").exists()


@pytest.mark.parametrize("command", ["search", "read", "doctor", "export", "shell"])
def test_read_commands_refuse_a_missing_database(tmp_path, command):
    db = tmp_path / "typo" / "kb.db"
    with pytest.raises(OKFError) as e:
        _require_db_unless_creating(command, str(db))
    assert e.value.code == "FILE_NOT_FOUND" and "okf init" in e.value.remedy
    assert not (tmp_path / "typo").exists()


@pytest.mark.parametrize("command", ["init", "import", "ingest"])
def test_creating_commands_may_make_a_new_database(tmp_path, command):
    _require_db_unless_creating(command, str(tmp_path / "new" / "kb.db"))


def test_cli_search_on_a_mistyped_path_creates_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["okf", "search", "x", "--db-path", "typo/kb.db"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2
    assert "FILE_NOT_FOUND" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_skills_ship_in_the_package():
    assert set(bundled_skills()) == {"okfgraph-mcp", "okfgraph-cli", "okfgraph-ingest"}


def test_setup_skills_installs_updates_and_refuses_unknown(tmp_path):
    first = OKFRouter.setup_skills(tmp_path)
    assert sorted(first["installed"]) == ["okfgraph-cli", "okfgraph-ingest", "okfgraph-mcp"]
    (tmp_path / "okfgraph-mcp" / "SKILL.md").write_text("old", encoding="utf-8")
    again = OKFRouter.setup_skills(tmp_path, ["okfgraph-mcp", "okfgraph-cli"])
    assert again["updated"] == ["okfgraph-mcp"] and again["unchanged"] == ["okfgraph-cli"]
    with pytest.raises(OKFError) as e:
        OKFRouter.setup_skills(tmp_path, ["nope"])
    assert e.value.code == "BAD_VALUE"


class _FakeRouter:
    def __init__(self):
        self.calls = []

    def ingest(self, kind, *, pdf_path, **_):
        self.calls.append(pdf_path)
        if "missing" in pdf_path:
            raise OKFError("FILE_NOT_FOUND", f"PDF not found: {pdf_path}")
        if "broken" in pdf_path:
            raise RuntimeError("bobine: corrupt xref")
        return {"concept_ids": [pdf_path], "page_count": 1,
                "md_path": "x.md", "image_dir": "x"}


def _args(*paths, json_out=False):
    argv = ["ingest", "--kind", "pdf", "--pdf-path", *paths] + (["--json"] if json_out else [])
    return build_parser().parse_args(argv)


def test_many_pdfs_run_in_one_process_and_keep_going(capsys):
    router = _FakeRouter()
    args = _args("a.pdf", "missing.pdf", "broken.pdf", "b.pdf", json_out=True)
    settings = type("S", (), {"mode": "text", "batch_size": 32})()
    code = _ingest_many_pdfs(args, router, settings, args.pdf_path, None)
    assert router.calls == ["a.pdf", "missing.pdf", "broken.pdf", "b.pdf"]
    assert code == 2  # FILE_NOT_FOUND, the first failure
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["error"]["code"] == "FILE_NOT_FOUND"
    assert [(f["ok"], (f.get("error") or {}).get("code")) for f in payload["data"]["files"]] == [
        (True, None), (False, "FILE_NOT_FOUND"), (False, "INTERNAL"), (True, None)]


def test_one_pdf_keeps_the_single_file_shape():
    assert _args("a.pdf").pdf_path == ["a.pdf"]
