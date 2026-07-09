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
        """All 16 tools are registered."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()
            names = [t.name for t in tools]

            expected = [
                "search_hybrid",
                "traverse",
                "get_by_id",
                "list_directory",
                "search_images",
                "search_chunks",
                "search_with_context",
                "search_chunks_with_hub_score",
                "expand_with_graph_context",
                "get_chunks",
                "reconstruct_document",
                "find_path",
                "export_bundle",
                "ingest_md",
                "ingest_thoughts",
                "ingest_pdf",
            ]
            assert len(tools) == 16
            for name in expected:
                assert name in names, f"Tool {name} not registered"

    def test_read_tools_have_read_only_hint(self):
        """Read-only tools have read_only_hint=True."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            read_tools = [
                "search_hybrid",
                "traverse",
                "get_by_id",
                "list_directory",
                "search_images",
                "search_chunks",
                "search_with_context",
                "search_chunks_with_hub_score",
                "expand_with_graph_context",
                "get_chunks",
                "reconstruct_document",
                "find_path",
            ]

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

            write_tools = [
                "export_bundle",
                "ingest_md",
                "ingest_thoughts",
                "ingest_pdf",
            ]

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

    def test_search_hybrid_schema(self):
        """search_hybrid has correct schema."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            search = next(t for t in tools if t.name == "search_hybrid")
            props = search.parameters["properties"]
            assert "query" in props
            assert "query" in search.parameters["required"]

    def test_ingest_md_schema(self):
        """ingest_md has correct schema."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            ingest = next(t for t in tools if t.name == "ingest_md")
            props = ingest.parameters["properties"]
            assert "md_path" in props
            assert "md_path" in ingest.parameters["required"]

    def test_ingest_thoughts_schema(self):
        """ingest_thoughts has correct schema."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            thoughts = next(t for t in tools if t.name == "ingest_thoughts")
            props = thoughts.parameters["properties"]
            assert "thoughts" in props
            assert "topic" in props
            assert set(thoughts.parameters["required"]) == {"thoughts", "topic"}

    def test_ingest_pdf_schema(self):
        """ingest_pdf has correct schema."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            tools = mcp._tool_manager.list_tools()

            pdf = next(t for t in tools if t.name == "ingest_pdf")
            props = pdf.parameters["properties"]
            assert "pdf_path" in props
            assert "pdf_path" in pdf.parameters["required"]
            assert "routing_mode" in props
            assert "mode" in props
            assert "extract_images" in props

    def test_server_name_and_instructions(self):
        """Server has correct name and instructions."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = f"{tmp}/test.db"
            mcp = create_mcp_server(db_path=db_path)
            assert mcp.name == "OKFgraph MCP Server"
            assert "ONNX" in mcp.instructions
            assert "Jina v5" in mcp.instructions
