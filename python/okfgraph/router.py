"""Public OKFRouter facade. M0 scaffold; full signatures land in M5."""
from __future__ import annotations

from typing import Any


class OKFRouter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "OKFRouter lands with the Rust core; see docs/plan-rust-core.md"
        )
