"""Public HTTP/UDS client surface (transport lands in Stage C)."""
from __future__ import annotations

from typing import Any, Optional

from . import _native
from .errors import OKFError

OPS = tuple(_native.op_names())


class OKFClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise OKFError("BAD_VALUE", "OKFClient transport lands in Stage C")


def discover(*args: Any, **kwargs: Any) -> Optional[dict]:
    return None


def serve_file_for(*args: Any, **kwargs: Any) -> Optional[str]:
    return None
