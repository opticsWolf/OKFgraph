"""MCP console-script shim. M0 scaffold."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print("okf-mcp: Rust MCP server lands in M4", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
