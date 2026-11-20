"""Public HTTP client surface for an ``okf serve`` owner.

The transport is stdlib-only HTTP/1.1: ``POST /ops/<op>`` takes the same
keyword arguments as :class:`~okfgraph.OKFRouter` and returns the op's
``data``; failures raise the same :class:`~okfgraph.errors.OKFError`.
"""
from __future__ import annotations

import http.client
import inspect
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .errors import OKFError

__all__ = ["OKFClient", "OPS", "serve_file_for", "discover"]

OPS = (
    "search", "read", "traverse", "embed",
    "ingest", "export_bundle", "export_concept",
    "import_bundle", "import_file", "import_files", "sync",
    "update", "supersede", "delete",
    "doctor", "diff", "reindex", "reembed", "pin_info", "lanes",
    "list_broken_links", "repair_links",
    "list_deleted", "recover_deleted", "purge_deleted", "detach",
    "list_images", "get_image",
    "get_by_id", "list_directory",
)


def serve_file_for(db_path: str | os.PathLike) -> Path:
    """The discovery file a running ``okf serve`` writes next to the db."""
    p = Path(db_path).expanduser().resolve()
    return p.with_name(p.name + ".serve.json")


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def discover(db_path: str | os.PathLike, timeout: float = 2.0) -> Optional[str]:
    """URL of a live server owning ``db_path``, or None for a stale file."""
    try:
        info = json.loads(serve_file_for(db_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url, pid = info.get("url"), info.get("pid")
    if not url or not isinstance(pid, int) or not _pid_alive(pid):
        return None
    try:
        health = OKFClient(url, timeout=timeout).health()
    except Exception:
        return None
    expected = Path(db_path).expanduser().resolve()
    return url if Path(health.get("db_path", "")).resolve() == expected else None


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


class OKFClient:
    """Router-shaped client for an ``okf serve`` owner."""

    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url
        self.timeout = timeout
        parsed = urlparse(url)
        if parsed.scheme == "unix":
            self._unix = parsed.path or parsed.netloc
            self._host = self._port = None
        elif parsed.scheme == "http":
            self._unix = None
            self._host = parsed.hostname or "127.0.0.1"
            self._port = parsed.port or 80
        else:
            raise ValueError(
                f"OKFClient url must be http://host:port or unix:///path, got {url!r}"
            )

    def _conn(self) -> http.client.HTTPConnection:
        if self._unix:
            return _UnixHTTPConnection(self._unix, self.timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        conn = self._conn()
        try:
            body = json.dumps(payload or {}, default=str).encode("utf-8")
            conn.request(
                method,
                path,
                body=body if method == "POST" else None,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
        finally:
            conn.close()
        try:
            return json.loads(raw)
        except ValueError:
            raise OKFError("INTERNAL", f"server answered non-JSON ({raw[:200]!r})") from None

    @staticmethod
    def _unwrap(env: dict) -> Any:
        if env.get("ok"):
            return env.get("data")
        err = env.get("error") or {}
        raise OKFError(
            err.get("code") or "INTERNAL",
            err.get("message") or "server error",
            op=env.get("op"),
            fields=err.get("fields"),
            remedy=err.get("remedy"),
            data=env.get("data"),
        )

    def health(self) -> Dict[str, Any]:
        return self._unwrap(self._request("GET", "/health"))

    def call(self, op: str, **kwargs: Any) -> Any:
        if op not in OPS:
            raise OKFError("BAD_VALUE", f"unknown op {op!r}", fields={"op": op})
        return self._unwrap(self._request("POST", f"/ops/{op}", kwargs))

    def jobs(self) -> list:
        return self._unwrap(self._request("GET", "/jobs"))

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._unwrap(self._request("GET", f"/jobs/{job_id}"))

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self._unwrap(self._request("POST", f"/jobs/{job_id}/cancel"))

    def reembed_job(self, **kwargs: Any) -> str:
        out = self.call("reembed", mode="job", **kwargs)
        return str(out.get("job"))

    def close(self) -> None:
        """Nothing to release: one connection per call."""

    def __enter__(self) -> "OKFClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        if name not in OPS:
            raise AttributeError(name)
        from .router import OKFRouter

        target = getattr(OKFRouter, name)
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            sig = None

        def op(*args: Any, **kwargs: Any) -> Any:
            if args and sig is not None:
                bound = sig.bind_partial(None, *args, **kwargs)
                kwargs = {k: v for k, v in bound.arguments.items() if k != "self"}
            elif args:
                raise TypeError(f"{name}() takes keyword arguments only here")
            return self.call(name, **kwargs)

        op.__name__ = name
        op.__doc__ = getattr(target, "__doc__", None)
        return op

    def __repr__(self) -> str:
        return f"OKFClient({self.url!r})"
