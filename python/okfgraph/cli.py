"""Console-script shim delegating to the Rust CLI."""
from __future__ import annotations

import sys
from typing import Optional

from . import _native


def main(argv: Optional[list[str]] = None) -> int:
    return int(_native.run_cli(list(sys.argv if argv is None else argv)))


if __name__ == "__main__":
    raise SystemExit(main())
