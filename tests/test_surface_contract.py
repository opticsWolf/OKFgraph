"""Surface contract: refusals, result shapes and config errors shared by
the CLI, MCP and Python adapters (docs/surface-unification-plan.md §3–§5).

Op-level tests run against the router double from test_mcp_server (the
canonical ops over stub components), CLI tests only exercise paths that
refuse before a router is opened, so none of this needs a database.
"""
import json
import warnings

import pytest

from okfgraph.cli import _main_catchall, build_parser
from okfgraph.errors import (OKFError, StateError, UsageError, envelope)
from okfgraph.settings import MCP_REQUIRED, SETTINGS, Settings
from tests.test_mcp_server import _StubRouter


def _router():
    return _StubRouter([])


def _refused(fn, *args, **kwargs) -> OKFError:
    with pytest.raises(OKFError) as exc:
        fn(*args, **kwargs)
    return exc.value


# ---- search: passed params the chosen path ignores are refused -----------
class TestSearchRefusals:
    @pytest.mark.parametrize("target, kwargs, ignored", [
        ("chunks", {"context_hops": 2}, ["context_hops"]),          # plain path
        ("chunks", {"include_chunks": True}, ["include_chunks"]),
        ("concepts", {"max_chunks_per_doc": 5}, ["max_chunks_per_doc"]),
        ("concepts", {"hub_weight": 0.5}, ["hub_weight"]),          # rank none
        ("concepts", {"expand": True}, ["expand"]),
        ("images", {"tags": ["a"]}, ["tags"]),
    ])
    def test_ignored_params_named(self, target, kwargs, ignored):
        err = _refused(_router().search, "q", target=target, **kwargs)
        assert err.code == "BAD_VALUE"
        assert err.fields["ignored"] == ignored

    def test_ppr_refuses_include_chunks(self):
        err = _refused(_router().search, "q", rank="ppr", include_chunks=True)
        assert err.fields["ignored"] == ["include_chunks"]

    def test_honoured_combinations_pass(self):
        r = _router()
        r.search("q", rank="hub", hub_weight=0.5)
        r.search("q", target="chunks", max_chunks_per_doc=1, tags=["a"])
        r.search("q", target="chunks", expand=True, context_hops=2)
        # hub_rerank supersedes expand rather than refusing it.
        r.search("q", target="chunks", hub_rerank=True, expand=True)

    @pytest.mark.parametrize("kwargs", [
        {"limit": 0}, {"target": "docs"}, {"rank": "pagerank"},
    ])
    def test_bad_values(self, kwargs):
        assert _refused(_router().search, "q", **kwargs).code == "BAD_VALUE"


# ---- traverse ------------------------------------------------------------
class TestTraverseRefusals:
    def test_walk_refuses_max_path_length(self):
        err = _refused(_router().traverse, "a", max_path_length=3)
        assert err.fields == {"mode": "walk", "ignored": ["max_path_length"]}

    def test_root_listing_refuses_walk_params(self):
        err = _refused(_router().traverse, "", depth=2)
        assert err.fields["ignored"] == ["depth"]

    def test_path_refuses_walk_params(self):
        err = _refused(_router().traverse, "a", target="b",
                       relationship="LINKS_TO")
        assert err.fields["ignored"] == ["relationship"]

    def test_unknown_start_is_state_error(self):
        r = _router()
        r.search_engine.node_exists = lambda node_id: False
        err = _refused(r.traverse, "ghost")
        assert err.code == "UNKNOWN_CONCEPT" and err.exit_code == 1


# ---- read ----------------------------------------------------------------
class _Model:
    def __init__(self, **fields):
        self.fields = fields

    def model_dump(self, exclude=()):
        return {k: v for k, v in self.fields.items() if k not in exclude}


class TestReadShapes:
    def test_chunks_are_plain_dicts_without_embeddings(self):
        r = _router()
        r.search_engine.get_chunks = lambda cid: [
            _Model(chunk_index=0, chunk_text="a", embedding=[0.1]),
            {"chunk_index": 1, "chunk_text": "b"},
        ]
        chunks = r.read("c1", include="chunks")
        assert chunks == [{"chunk_index": 0, "chunk_text": "a"},
                          {"chunk_index": 1, "chunk_text": "b"}]
        json.dumps(chunks)  # JSON-able (D2)

    def test_bad_budget_refused(self):
        assert _refused(_router().read, "c1", max_tokens=0).code == "BAD_VALUE"


# ---- ingest ------------------------------------------------------------------
class TestIngestRefusals:
    @staticmethod
    def _mgr():
        from okfgraph.components.ingest import IngestManager
        return IngestManager.__new__(IngestManager)  # refusals need no state

    @pytest.mark.parametrize("kind, kwargs, ignored", [
        ("md", {"md_path": "x.md", "topic": "t"}, ["topic"]),
        ("thoughts", {"thoughts": "t", "topic": "t", "title": "T"}, ["title"]),
        ("pdf", {"pdf_path": "x.pdf", "tags": ["a"]}, ["tags"]),
        ("pdf", {"pdf_path": "x.pdf", "output_dir": "o"}, ["output_dir"]),
    ])
    def test_other_kinds_params_refused(self, kind, kwargs, ignored):
        err = _refused(self._mgr().ingest, kind, **kwargs)
        assert err.code == "BAD_VALUE" and err.fields["ignored"] == ignored


# ---- errors / envelope -----------------------------------------------------
class TestErrorRouting:
    def test_class_follows_code_not_constructor(self):
        err = UsageError("UNKNOWN_CONCEPT", "x")
        assert isinstance(err, StateError) and err.exit_code == 1

    def test_pickle_round_trip(self):
        import pickle
        err = OKFError("LINT_ERRORS", "2 errors", op="lint",
                       fields={"errors": 2}, data={"files": 3})
        back = pickle.loads(pickle.dumps(err))
        assert type(back) is type(err)
        assert (back.code, back.op, back.fields, back.data) == (
            "LINT_ERRORS", "lint", {"errors": 2}, {"files": 3})

    def test_outcome_data_rides_into_the_envelope(self):
        report = {"identical": False, "added": ["gamma"]}
        err = OKFError("DIFF_DIFFERENT", "differ", op="diff", data=report)
        env = envelope("diff", error=err)
        assert env["ok"] is False
        assert env["data"] == report
        assert env["error"]["code"] == "DIFF_DIFFERENT"
        assert "data" not in env["error"]

    def test_warning_messages_render_as_text(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("careful", UserWarning)
        env = envelope("search", data=[], warnings=caught)
        assert env["warnings"] == ["careful"]


# ---- settings ----------------------------------------------------------------
class TestSettings:
    def test_dataclass_defaults_match_the_table(self):
        defaults = Settings()
        for row in SETTINGS:
            assert getattr(defaults, row.name) == row.default, row.name

    def test_mcp_requires_db_path(self):
        assert MCP_REQUIRED == ("db_path",)

    def test_require_names_every_layer(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError) as exc:
            Settings.load(environ={}, require=MCP_REQUIRED)
        msg = str(exc.value)
        assert "--db-path" in msg and "OKFGRAPH_DB_PATH" in msg
        assert "[database] db_path" in msg

    @pytest.mark.parametrize("layer", ["env", "toml"])
    def test_require_satisfied_by_env_or_toml(self, tmp_path, monkeypatch, layer):
        monkeypatch.chdir(tmp_path)
        environ = {}
        if layer == "env":
            environ["OKFGRAPH_DB_PATH"] = "g.db"
        else:
            (tmp_path / "okfgraph.toml").write_text(
                '[database]\ndb_path = "g.db"\n', encoding="utf-8")
        s = Settings.load(environ=environ, require=MCP_REQUIRED)
        assert s.db_path == "g.db"


# ---- CLI: refusals before any router opens ------------------------------------
def _cli(argv, capsys):
    rc = _main_catchall(build_parser().parse_args(argv))
    out, err = capsys.readouterr()
    return rc, out, err


class TestCliRefusals:
    def test_import_missing_file_refused_up_front(self, tmp_path, capsys):
        good = tmp_path / "a.md"
        good.write_text("# a\n", encoding="utf-8")
        rc, out, err = _cli(["import", str(good), str(tmp_path / "nope.md")], capsys)
        assert rc == 2
        assert "FILE_NOT_FOUND" in err and "nope.md" in err

    def test_import_without_files_or_all(self, capsys):
        rc, _, err = _cli(["import"], capsys)
        assert rc == 2 and "MISSING_PARAM" in err

    def test_export_needs_a_scope(self, tmp_path, capsys):
        rc, _, err = _cli(["export", "--output-dir", str(tmp_path)], capsys)
        assert rc == 2 and "MISSING_PARAM" in err

    def test_export_both_scopes_refused(self, tmp_path, capsys):
        rc, _, err = _cli(["export", "--all", "--concept-id", "x",
                           "--output-dir", str(tmp_path)], capsys)
        assert rc == 2 and "BAD_VALUE" in err

    def test_json_error_envelope_on_stdout(self, capsys):
        rc, out, err = _cli(["import", "--json"], capsys)
        env = json.loads(out)
        assert rc == 2
        assert env["ok"] is False and env["op"] == "import_file"
        assert env["error"]["code"] == "MISSING_PARAM"
        assert "[ERROR] MISSING_PARAM" in err

    def test_bad_config_is_config_invalid(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc, _, err = _cli(["init", "--embedding-dim", "7"], capsys)
        assert rc == 2 and "CONFIG_INVALID" in err

    def test_diff_snapshot_is_router_free(self, tmp_path, capsys):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        rc, out, _ = _cli(["diff", str(tmp_path / "a"), str(tmp_path / "b"),
                           "--json"], capsys)
        assert rc == 0 and json.loads(out)["data"]["identical"] is True
