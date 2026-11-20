"""MCP console-script shim delegating to the Rust rmcp server."""
from __future__ import annotations

import sys
from typing import Optional

from . import _native
from ._pdf import resolve_plugin

__all__ = ["main"]


def main(argv: Optional[list[str]] = None) -> int:
    resolve_plugin()
    return int(_native.run_mcp(list(sys.argv if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
