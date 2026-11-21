#!/usr/bin/env python3
"""MCP Streamable HTTP mount on a running Rust `okf serve` owner."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TOOLS = {
    "search", "read", "traverse", "ingest", "export_bundle", "export_concept",
    "list_images", "get_image", "update", "supersede", "delete",
}


async def probe(url: str) -> None:
    async with streamable_http_client(f"{url}/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert names == EXPECTED_TOOLS, names ^ EXPECTED_TOOLS
            by_name = {tool.name: tool for tool in listed.tools}
            assert by_name["read"].annotations.read_only_hint is True
            assert by_name["delete"].annotations.destructive_hint is True
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
    print("OK: okf serve /mcp exposes 11 annotated tools over Streamable HTTP")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-exe", default="target/debug/okf.exe")
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10.db")
    ap.add_argument("--listen", default="127.0.0.1:8787")
    args = ap.parse_args()
    exe = Path(args.serve_exe)
    if not exe.is_absolute():
        exe = ROOT / exe
    db = ROOT / "parity/out/mcp-http.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / args.fixture, db)
    disc = Path(str(db) + ".serve.json")
    if disc.exists():
        disc.unlink()
    owner = subprocess.Popen(
        [str(exe), "--db-path", str(db), "serve", "--listen", args.listen],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = None
        for _ in range(100):
            if disc.exists():
                url = json.loads(disc.read_text(encoding="utf-8"))["url"]
                break
            if owner.poll() is not None:
                raise SystemExit(f"serve exited: {owner.stderr.read()}")
            time.sleep(0.1)
        assert url, "serve did not write its discovery file"
        asyncio.run(probe(url))
    finally:
        owner.terminate()
        try:
            owner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owner.kill()


if __name__ == "__main__":
    main()
