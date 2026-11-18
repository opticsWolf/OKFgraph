"""Console-script shim. M0 scaffold."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print("okf: Rust CLI lands in M4", file=sys.stderr)
    return 2
