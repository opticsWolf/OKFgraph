"""D2-a: the `delete(concept_id)` op — soft-delete with graph/bundle guard.

Covers the plan's D2-a contract:
* success → {"concept_id", "deleted": True, "recoverable_until", ...},
  recoverable through `recover_deleted` and visible in `list_deleted`;
* refusal `BAD_VALUE` + remedy while the concept's source file exists;
* after removing the source, delete succeeds;
* unknown id → typed BAD_VALUE;
* NOT exposed on the MCP tool registry (destructive by policy).
"""
from pathlib import Path

import pytest


BUNDLE = "---\ntype: note\ntitle: D\n---\nDelete-me body.\n"


@pytest.fixture()
def env(tmp_path):
    from okfgraph.router import OKFRouter
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "d.md").write_text(BUNDLE, encoding="utf-8")
    r = OKFRouter(db_path=str(tmp_path / "kb.db"),
                  bundle_root=str(tmp_path / "kb"))
    r.import_bundle(None)  # registers FileHash rows (the delete guard reads those)
    yield {"tmp": tmp_path, "router": r, "ns_id": "d"}
    r.close()


class TestDeleteOp:
    def test_result_shape_and_recovery_cycle(self, env):
        tmp, r, cid = env["tmp"], env["router"], env["ns_id"]
        (tmp / "kb" / "d.md").unlink()  # source gone, delete is allowed
        result = r.delete(cid)
        assert result["concept_id"] == cid
        assert result["deleted"] is True
        assert result["recoverable_until"]
        deleted = r.list_deleted()
        assert any(d["concept_id"] == cid and d["recoverable"] for d in deleted)
        assert r.recover_deleted(cid)["recovered"] is True
        assert r.get_by_id(cid) is not None

    def test_refuses_while_source_exists(self, env):
        r = env["router"]
        from okfgraph.errors import OKFError
        with pytest.raises(OKFError) as exc:
            r.delete(env["ns_id"])
        assert exc.value.code == "BAD_VALUE"
        assert exc.value.remedy and "--prune-missing" in exc.value.remedy
        assert any("d.md" in s for s in exc.value.fields["sources"])

    def test_delete_after_source_removed(self, env):
        tmp, r, cid = env["tmp"], env["router"], env["ns_id"]
        (tmp / "kb" / "d.md").unlink()
        result = r.delete(cid)
        assert result["deleted"] is True
        # The next import no longer resurrects it (file gone).
        r.import_bundle(None)
        assert r.get_by_id(cid) is None

    def test_unknown_concept_refused(self, env):
        from okfgraph.errors import OKFError
        with pytest.raises(OKFError) as exc:
            env["router"].delete("no/such-id")
        assert exc.value.code == "BAD_VALUE"

    def test_thoughts_concept_deletes_freely(self, env):
        tmp, r = env["tmp"], env["router"]
        (tmp / "kb" / "d.md").unlink()  # clear the FileHash guard first
        r.ingest("thoughts", thoughts="delete me now", topic="t")
        hit = r.search("delete me now")[0]["id"]
        result = r.delete(hit)
        assert result["deleted"] is True


class TestDeleteSurface:
    def test_not_on_mcp(self):
        import asyncio
        from okfgraph.settings import Settings
        from okfgraph.mcp_server import create_mcp_server
        srv = create_mcp_server(Settings())
        tools = [t.name for t in asyncio.run(srv.list_tools())]
        assert "delete" not in tools

    def test_cli_verb_parses(self):
        from okfgraph.cli import build_parser
        args = build_parser().parse_args(
            ["delete", "d", "--db-path", "x.db"])
        assert args.command == "delete"
        assert args.concept_id == "d"
