#!/usr/bin/env python3
"""MCP stdio proxy against a running Rust `okf serve` owner."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]


async def probe(mcp_exe: Path, db: Path, url: str) -> None:
    params = StdioServerParameters(
        command=str(mcp_exe),
        args=["--db-path", str(db), "--connect", url],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert len(tools.tools) == 11
            result = await session.call_tool("read", {"concept_id": "alpha"})
            text = "".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
            env = json.loads(text)
            assert env["ok"] and env["data"]["id"] == "alpha", env
    print("OK: okf-mcp --connect proxies tools through the serve owner")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-exe", default="target/debug/okf.exe")
    ap.add_argument("--mcp-exe", default="target/debug/okf-mcp.exe")
    ap.add_argument("--fixture", default="parity/fixtures/oracle-v10.db")
    ap.add_argument("--listen", default="127.0.0.1:8787")
    args = ap.parse_args()
    serve_exe = Path(args.serve_exe)
    mcp_exe = Path(args.mcp_exe)
    if not serve_exe.is_absolute():
        serve_exe = ROOT / serve_exe
    if not mcp_exe.is_absolute():
        mcp_exe = ROOT / mcp_exe
    db = ROOT / "parity/out/mcp-connect.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / args.fixture, db)
    disc = Path(str(db) + ".serve.json")
    if disc.exists():
        disc.unlink()
    owner = subprocess.Popen(
        [str(serve_exe), "--db-path", str(db), "serve", "--listen", args.listen],
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
        asyncio.run(probe(mcp_exe, db, url))
    finally:
        owner.terminate()
        try:
            owner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            owner.kill()


if __name__ == "__main__":
    main()
