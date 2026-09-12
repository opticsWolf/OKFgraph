"""Tests for the OKFgraph MCP server."""

import json
import tempfile
from pathlib import Path

import pytest

from okfgraph.mcp_server import create_mcp_server


class TestMCPServer:
    """Tests for the MCP server tool registry and basic functionality."""

    def test_server_creates(self):
        """Server creates without error."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            assert mcp is not None

    def test_all_tools_registered(self):
        """The five consolidated tools are registered."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()
            names = [t.name for t in tools]

            assert names == ["search", "read", "traverse", "ingest", "export_bundle"]

    def test_read_tools_have_read_only_hint(self):
        """Read-only tools have read_only_hint=True."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            read_tools = ["search", "read", "traverse"]

            for tool in tools:
                if tool.name in read_tools:
                    assert tool.annotations.read_only_hint is True, (
                        f"{tool.name} should have read_only_hint=True"
                    )

    def test_write_tools_have_read_only_hint_false(self):
        """Write tools have read_only_hint=False."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            write_tools = ["export_bundle", "ingest"]

            for tool in tools:
                if tool.name in write_tools:
                    assert tool.annotations.read_only_hint is False, (
                        f"{tool.name} should have read_only_hint=False"
                    )

    def test_tool_has_description(self):
        """All tools have a non-empty description."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            for tool in tools:
                assert (
                    tool.description and len(tool.description) > 10
                ), f"Tool {tool.name} has empty or short description"

    def test_tool_has_parameters(self):
        """All tools have parameter definitions."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            for tool in tools:
                assert tool.parameters is not None, (
                    f"Tool {tool.name} has no parameters"
                )
                assert "properties" in tool.parameters, (
                    f"Tool {tool.name} has no properties in schema"
                )

    def test_search_schema(self):
        """search exposes target/expand/hub_rerank routing."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            search = next(t for t in tools if t.name == "search")
            props = search.parameters["properties"]
            assert "query" in props
            assert "query" in search.parameters["required"]
            assert props["target"]["default"] == "concepts"
            for flag in ("expand", "hub_rerank", "limit", "type_filter", "tags", "parent_id"):
                assert flag in props, f"search missing {flag}"

    def test_read_schema(self):
        """read exposes the include selector."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            read = next(t for t in tools if t.name == "read")
            props = read.parameters["properties"]
            assert "concept_id" in props
            assert "concept_id" in read.parameters["required"]
            assert props["include"]["default"] == "body"

    def test_traverse_schema(self):
        """traverse exposes path mode via target."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            traverse = next(t for t in tools if t.name == "traverse")
            props = traverse.parameters["properties"]
            assert "start_id" in props
            assert "target" in props
            assert "max_path_length" in props

    def test_ingest_schema(self):
        """ingest exposes the kind discriminator."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            ingest = next(t for t in tools if t.name == "ingest")
            props = ingest.parameters["properties"]
            assert "kind" in props
            assert "kind" in ingest.parameters["required"]
            for p in ("md_path", "pdf_path", "thoughts", "topic", "routing_mode", "mode", "extract_images"):
                assert p in props, f"ingest missing {p}"

    def test_server_name_and_instructions(self):
        """Server has correct name and instructions."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            assert mcp.name == "OKFgraph MCP Server"
            assert "ONNX" in mcp.instructions
            assert "Jina v5" in mcp.instructions

    @pytest.mark.anyio
    async def test_lifespan_yields_router_and_closes(self, tmp_path, monkeypatch):
        """Lifespan builds the router, yields GraphContext, and closes it.

        Regression test: the lifespan closure used to rebind bundle_root
        (UnboundLocalError), so the MCP server could never start.
        OKFRouter is stubbed — this guards the wiring, not the engine.
        """
        import okfgraph.mcp_server as ms

        built = {}
        closed = []

        class StubRouter:
            def __init__(self, **kwargs):
                built.update(kwargs)

            def close(self):
                closed.append(True)

        monkeypatch.setattr(ms, "OKFRouter", StubRouter)
        mcp = create_mcp_server(
            db_path=str(tmp_path / "t.db"), bundle_root=str(tmp_path),
        )
        async with mcp.settings.lifespan(mcp) as gctx:
            assert isinstance(gctx, ms.GraphContext)
            assert isinstance(gctx.router, StubRouter)
            assert built["db_path"] == str(tmp_path / "t.db")
            assert built["bundle_root"] == str(tmp_path)
            assert closed == []
        assert closed == [True]

    def test_get_router_resolves_from_lifespan_context(self):
        """_get_router extracts the router from a v2 lifespan context."""
        from types import SimpleNamespace
        from okfgraph.mcp_server import GraphContext, _get_router

        sentinel = object()
        ctx = SimpleNamespace(
            request_context=SimpleNamespace(
                lifespan_context=GraphContext(router=sentinel),
            ),
        )
        assert _get_router(ctx) is sentinel


class _StubSearchEngine:
    def __init__(self, calls):
        self.calls = calls

    def search_chunks(self, query, limit=10, **filt):
        self.calls.append(("search_chunks", query, limit, filt))
        return [{"chunk_id": "c1"}]

    def search_with_context(self, query, limit=5, context_hops=1):
        self.calls.append(("search_with_context", query, limit, context_hops))
        return [{"chunk": {"chunk_id": "c1"}}]

    def search_chunks_with_hub_score(self, query, limit=10, hub_weight=0.3):
        self.calls.append(("hub", query, limit, hub_weight))
        return [{"chunk_id": "c1"}]

    def get_chunks(self, concept_id):
        self.calls.append(("get_chunks", concept_id))
        return []

    def find_path(self, start_id, end_id, max_length=6):
        self.calls.append(("find_path", start_id, end_id, max_length))
        return [{"id": end_id}]

    def _get_ancestry(self, concept_id):
        return []

    def _get_siblings(self, concept_id):
        return []


class _StubEmbedEngine:
    def __init__(self, calls):
        self.calls = calls

    def reconstruct_document(self, concept_id):
        self.calls.append(("reconstruct", concept_id))
        return "# doc"


class _StubIngestMgr:
    def __init__(self, calls):
        self.calls = calls

    def ingest_md(self, **kwargs):
        self.calls.append(("ingest_md", kwargs))
        return {"concept_id": "c-md"}

    def ingest_pdf(self, **kwargs):
        self.calls.append(("ingest_pdf", kwargs))
        return {"concept_ids": ["c-pdf"]}

    def ingest_thoughts(self, thoughts, topic=None, concept_id=None, tags=None):
        self.calls.append(("ingest_thoughts", thoughts, topic))
        return {"concept_id": "c-t"}


class _StubRouter:
    def __init__(self, calls):
        self.calls = calls
        self.search_engine = _StubSearchEngine(calls)
        self.embed_engine = _StubEmbedEngine(calls)
        self.ingest_mgr = _StubIngestMgr(calls)

    def search_hybrid(self, query, limit=10, **filt):
        self.calls.append(("search_hybrid", query, limit, filt))
        return [{"id": "c1"}]

    def search_images(self, query, limit=10):
        self.calls.append(("search_images", query, limit))
        return []

    def traverse(self, start_id, rel="CONTAINS", direction="OUTGOING", depth=1):
        self.calls.append(("traverse", start_id, rel, direction, depth))
        return []

    def get_by_id(self, concept_id):
        self.calls.append(("get_by_id", concept_id))
        return None if concept_id == "missing" else {"id": concept_id}

    def list_directory(self, directory_id):
        self.calls.append(("list_directory", directory_id))
        return []


class TestToolDispatch:
    """The consolidated tools route to the right router methods."""

    @staticmethod
    def _fn(name):
        mcp = create_mcp_server(db_path=":memory:")
        return mcp._tool_manager.get_tool(name).fn

    @staticmethod
    def _ctx(stub):
        from types import SimpleNamespace
        from okfgraph.mcp_server import GraphContext
        return SimpleNamespace(
            request_context=SimpleNamespace(
                lifespan_context=GraphContext(router=stub),
            ),
        )

    def test_search_concepts_default(self):
        calls = []
        out = self._fn("search")("q", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "search_hybrid"
        assert '"c1"' in out

    def test_search_chunks_plain_and_filtered(self):
        calls = []
        self._fn("search")("q", target="chunks", tags=["a"], ctx=self._ctx(_StubRouter(calls)))
        name, _, _, filt = calls[0]
        assert name == "search_chunks" and filt == {"tags": ["a"]}

    def test_search_chunks_expand(self):
        calls = []
        self._fn("search")("q", target="chunks", expand=True, ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "search_with_context"

    def test_search_chunks_hub_wins_over_expand(self):
        calls = []
        self._fn("search")(
            "q", target="chunks", expand=True, hub_rerank=True,
            ctx=self._ctx(_StubRouter(calls)),
        )
        assert calls[0][0] == "hub"

    def test_search_images(self):
        calls = []
        self._fn("search")("cat", target="images", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "search_images"

    def test_read_body_and_missing(self):
        calls = []
        out = self._fn("read")("c1", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "get_by_id"
        out = self._fn("read")("missing", ctx=self._ctx(_StubRouter(calls)))
        assert "not found" in out

    def test_read_variants(self):
        for include, expect in [("chunks", "get_chunks"), ("document", "reconstruct"), ("context", "traverse")]:
            calls = []
            self._fn("read")("c1", include=include, ctx=self._ctx(_StubRouter(calls)))
            assert any(c[0] == expect for c in calls), include

    def test_traverse_modes(self):
        calls = []
        self._fn("traverse")("a", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "traverse"
        calls = []
        self._fn("traverse")("a", target="b", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "find_path"
        calls = []
        self._fn("traverse")("", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "list_directory"

    def test_ingest_kinds(self):
        calls = []
        self._fn("ingest")(kind="md", md_path="x.md", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "ingest_md"
        calls = []
        self._fn("ingest")(kind="pdf", pdf_path="x.pdf", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "ingest_pdf"
        calls = []
        self._fn("ingest")(kind="thoughts", thoughts="t", topic="top", ctx=self._ctx(_StubRouter(calls)))
        assert calls[0][0] == "ingest_thoughts"

    def test_ingest_validates_required(self):
        calls = []
        out = self._fn("ingest")(kind="md", ctx=self._ctx(_StubRouter(calls)))
        assert out.startswith("error:") and calls == []
        out = self._fn("ingest")(kind="thoughts", thoughts="t", ctx=self._ctx(_StubRouter(calls)))
        assert out.startswith("error:") and calls == []
