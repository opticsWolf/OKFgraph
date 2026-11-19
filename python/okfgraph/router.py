"""Public OKFRouter facade over the Rust core. No business logic."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import _native
from ._pdf import resolve_plugin
from .errors import OKFError
from .models import ConceptModel

_READ_METHODS = {
    "get_by_id": "get_by_id",
    "list_directory": "list_directory",
    "read": "read",
    "traverse": "traverse",
    "list_images": "list_images",
    "get_image": "get_image",
    "list_deleted": "list_deleted",
    "list_broken_links": "list_broken_links",
    "pin_info": "pin_info",
    "lanes": "lanes",
    "doctor": "doctor",
}


class OKFRouter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        resolve_plugin()
        if args:
            raise OKFError("BAD_VALUE", "OKFRouter takes keyword settings only")
        params = {key: value for key, value in kwargs.items() if value is not None}
        if "embed_backend" in params:
            params["backend"] = params.pop("embed_backend")
        self._native = _native.NativeRouter.open(json.dumps(params, default=str))

    def close(self) -> None:
        self._native.close()

    def __enter__(self) -> "OKFRouter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def call(self, op: str, params: Optional[Dict[str, Any]] = None) -> Any:
        try:
            raw = self._native.call(op, json.dumps(params or {}, default=str))
        except RuntimeError as exc:
            raise OKFError.from_native(str(exc)) from None
        return json.loads(raw)

    def get_by_id(self, concept_id: str) -> Optional[ConceptModel]:
        data = self._call("get_by_id", concept_id=concept_id)
        return ConceptModel(**data) if data is not None else None

    def list_directory(self, directory_id: str = "") -> List[Dict[str, Any]]:
        return self._call("list_directory", directory_id=directory_id)

    def read(self, concept_id: str, *, include: str = "body") -> Any:
        data = self._call("read", concept_id=concept_id, include=include)
        if include == "body" and isinstance(data, dict):
            return ConceptModel(**data)
        return data

    def traverse(self, start_id: str = "", **kwargs: Any) -> List[Dict[str, Any]]:
        return self._call("traverse", start_id=start_id, **kwargs)

    def list_images(self, concept_id: str) -> List[Dict[str, Any]]:
        return self._call("list_images", concept_id=concept_id)

    def get_image(self, asset_id: str) -> Dict[str, Any]:
        return self._call("get_image", asset_id=asset_id)

    def list_deleted(self) -> List[Dict[str, Any]]:
        return self._call("list_deleted")

    def list_broken_links(self) -> List[Dict[str, Any]]:
        return self._call("list_broken_links")

    def pin_info(self) -> Dict[str, Any]:
        return self._call("pin_info")

    def lanes(self) -> Dict[str, Any]:
        return self._call("lanes")

    def doctor(self) -> Dict[str, Any]:
        return self._call("doctor")

    def _call(self, op: str, **params: Any) -> Any:
        return self.call(_READ_METHODS[op], params)
