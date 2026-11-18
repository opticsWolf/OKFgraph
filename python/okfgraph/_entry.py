"""Installed console-script entry point."""
from __future__ import annotations

from .cli import main as okf


if __name__ == "__main__":
    raise SystemExit(okf())
