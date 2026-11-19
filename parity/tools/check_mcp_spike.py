#!/usr/bin/env python3
"""Connect the Python MCP SDK to the Rust stdio spike server."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def run(exe: Path) -> None:
    params = StdioServerParameters(command=str(exe), args=[])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            print("tools:", names)
            if "echo" not in names:
                raise SystemExit("FAIL: echo tool missing")
            result = await session.call_tool("echo", {"text": "ping"})
            text = "".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
            print("echo result:", text)
            if text != "echo:ping":
                raise SystemExit(f"FAIL: unexpected result {text!r}")
    print("OK: Python MCP client connected to Rust stdio server")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    args = ap.parse_args()
    asyncio.run(run(Path(args.exe)))


if __name__ == "__main__":
    main()
