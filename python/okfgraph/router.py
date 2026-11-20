"""Public OKFRouter facade over the Rust core. No business logic."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from . import _native
from ._pdf import resolve_plugin
from .errors import OKFError
from .models import ConceptModel


class OKFRouter:
    def __init__(
        self,
        db_path: Optional[str] = None,
        bundle_root: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        resolve_plugin()
        if db_path is None:
            raise TypeError("OKFRouter() missing required argument: 'db_path'")
        params: Dict[str, Any] = {"db_path": db_path}
        if bundle_root is not None:
            params["bundle_root"] = bundle_root
        params.update({key: value for key, value in kwargs.items() if value is not None})
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

    # ------------------------------------------------------------------
    # query / read
    # ------------------------------------------------------------------

    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        return self.call("search", {"query": query, **kwargs})

    def read(
        self,
        concept_id: str,
        *,
        include: str = "body",
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        params: Dict[str, Any] = {"concept_id": concept_id, "include": include, **kwargs}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        data = self.call("read", params)
        if include == "body" and isinstance(data, dict):
            return ConceptModel(**data)
        return data

    def traverse(self, start_id: str = "", **kwargs: Any) -> List[Dict[str, Any]]:
        return self.call("traverse", {"start_id": start_id, **kwargs})

    def embed(self, *, texts: Iterable[str], task: str = "Document", **kwargs: Any) -> Any:
        return self.call("embed", {"texts": list(texts), "task": task, **kwargs})

    # ------------------------------------------------------------------
    # ingest / import / export
    # ------------------------------------------------------------------

    def ingest(self, kind: str, **kwargs: Any) -> Dict[str, Any]:
        return self.call("ingest", {"kind": kind, **kwargs})

    def import_bundle(self, bundle_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(kwargs)
        if bundle_path is not None:
            params["bundle_path"] = bundle_path
        return self.call("import_bundle", params)

    def import_file(self, file_path: str, **kwargs: Any) -> Dict[str, Any]:
        return self.call("import_file", {"file_path": file_path, **kwargs})

    def import_files(self, paths: Iterable[str], **kwargs: Any) -> Dict[str, Any]:
        return self.call("import_files", {"paths": list(paths), **kwargs})

    def sync(self, producer: str, source: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {"producer": producer, **kwargs}
        if source is not None:
            params["source"] = source
        return self.call("sync", params)

    def produce(
        self,
        source_type: str,
        source_path: Optional[str] = None,
        output_dir: str = ".",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "source_type": source_type,
            "output_dir": output_dir,
            **kwargs,
        }
        if source_path is not None:
            params["source_path"] = source_path
        return self.call("produce", params)

    def export_bundle(self, output_dir: str, **kwargs: Any) -> Dict[str, Any]:
        return self.call("export_bundle", {"output_dir": output_dir, **kwargs})

    def export_concept(self, concept_id: str, *, output_dir: str, **kwargs: Any) -> Dict[str, Any]:
        return self.call(
            "export_concept",
            {"concept_id": concept_id, "output_dir": output_dir, **kwargs},
        )

    # ------------------------------------------------------------------
    # writes / lifecycle
    # ------------------------------------------------------------------

    def update(self, concept_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self.call("update", {"concept_id": concept_id, **kwargs})

    def supersede(self, old_id: str, new_id: str) -> Dict[str, Any]:
        return self.call("supersede", {"old_id": old_id, "new_id": new_id})

    def delete(self, concept_id: str, *, remove_file: bool = False) -> Dict[str, Any]:
        return self.call(
            "delete", {"concept_id": concept_id, "remove_file": remove_file}
        )

    def recover_deleted(self, concept_id: str) -> Dict[str, Any]:
        return self.call("recover_deleted", {"concept_id": concept_id})

    def purge_deleted(self, *, older_than: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if older_than is not None:
            params["older_than"] = older_than
        return self.call("purge_deleted", params)

    def detach(self, **kwargs: Any) -> Dict[str, Any]:
        return self.call("detach", kwargs)

    def repair_links(self) -> Dict[str, Any]:
        return self.call("repair_links", {})

    # ------------------------------------------------------------------
    # health / diagnostics
    # ------------------------------------------------------------------

    def diff(self, old: Optional[str] = None, new: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if old is not None:
            params["old"] = old
        if new is not None:
            params["new"] = new
        return self.call("diff", params)

    def lint(self, bundle_dir: str = ".", identity: str = "path") -> Dict[str, Any]:
        return self.call("lint", {"bundle_dir": bundle_dir, "identity": identity})

    def reindex(self, *, if_dirty: bool = False) -> Dict[str, Any]:
        return self.call("reindex", {"if_dirty": if_dirty})

    def reembed(self, **kwargs: Any) -> Dict[str, Any]:
        return self.call("reembed", kwargs)

    def doctor(
        self,
        *,
        stale_days: int = 365,
        fix: bool = False,
        strict: bool = False,
        cache_dir: str | None = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "stale_days": stale_days,
            "fix": fix,
            "strict": strict,
        }
        if cache_dir is not None:
            params["cache_dir"] = cache_dir
        return self.call("doctor", params)

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------

    def get_by_id(self, concept_id: str) -> Optional[ConceptModel]:
        data = self.call("get_by_id", {"concept_id": concept_id})
        return ConceptModel(**data) if data is not None else None

    def list_directory(self, directory_id: str = "") -> List[Dict[str, Any]]:
        return self.call("list_directory", {"directory_id": directory_id})

    def list_images(self, concept_id: str) -> List[Dict[str, Any]]:
        return self.call("list_images", {"concept_id": concept_id})

    def get_image(self, asset_id: str) -> Dict[str, Any]:
        return self.call("get_image", {"asset_id": asset_id})

    def list_deleted(self) -> List[Dict[str, Any]]:
        return self.call("list_deleted", {})

    def list_broken_links(self) -> List[Dict[str, Any]]:
        return self.call("list_broken_links", {})

    def pin_info(self) -> Dict[str, Any]:
        return self.call("pin_info", {})

    def lanes(self) -> Dict[str, Any]:
        return self.call("lanes", {})
