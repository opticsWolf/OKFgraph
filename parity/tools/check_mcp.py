#!/usr/bin/env python3
"""Public rmcp contract smoke against the Rust `okf-mcp` stdio server."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TOOLS = {
    "search",
    "read",
    "traverse",
    "ingest",
    "export_bundle",
    "export_concept",
    "list_images",
    "get_image",
    "update",
    "supersede",
    "delete",
}


async def run(exe: Path, db: Path) -> None:
    params = StdioServerParameters(
        command=str(exe),
        args=["--db-path", str(db)],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == EXPECTED_TOOLS, (names ^ EXPECTED_TOOLS)
            by_name = {tool.name: tool for tool in listed.tools}
            assert by_name["read"].annotations.read_only_hint is True
            assert by_name["delete"].annotations.destructive_hint is True
            assert "concept_id" in by_name["read"].input_schema["properties"]
            result = await session.call_tool("read", {"concept_id": "alpha"})
            text = "".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
            env = json.loads(text)
            assert env["ok"] and env["op"] == "read", env
            assert env["data"]["id"] == "alpha", env["data"]
            missing = await session.call_tool("read", {"concept_id": "nope"})
            text = "".join(
                block.text for block in missing.content if getattr(block, "text", None)
            )
            env = json.loads(text)
            assert not env["ok"] and env["error"]["code"] == "UNKNOWN_CONCEPT", env
    print("OK: Rust MCP server exposes 11 annotated tools with envelope results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="target/debug/okf-mcp.exe")
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10.db")
    args = ap.parse_args()
    exe = Path(args.exe)
    if not exe.is_absolute():
        exe = ROOT / exe
    db = ROOT / "parity/out/mcp-smoke.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / args.fixture, db)
    import asyncio

    asyncio.run(run(exe, db))


if __name__ == "__main__":
    main()
