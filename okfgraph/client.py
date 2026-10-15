"""``OKFClient``: the router's ops over HTTP, against one ``okf serve`` owner.

Ladybug allows one process per database file. Before 0.11 a second process
(another MCP client, the CLI next to a running harness) failed with
``DB_LOCKED``. ``okf serve`` (0.11.0, O4) owns the file; everything else
talks to it:

* ``OKFClient(url)`` — same op names and keyword arguments as
  :class:`~okfgraph.OKFRouter`, results are the op's ``data``, failures
  raise the same :class:`~okfgraph.errors.OKFError` (code, fields, remedy,
  data) the router would have raised.
* the CLI — uses the server automatically when one is registered for the
  database (``<db_path>.serve.json``) and answers ``/health``;
* ``okf-mcp --connect URL`` — a stdio MCP proxy, so existing ``.mcp.json``
  setups keep working against one owner.

Transport: plain HTTP/1.1, JSON in and the result envelope out
(``POST /ops/<op>``), over TCP (``http://127.0.0.1:PORT``) or a Unix socket
(``unix:///path/okf.sock``). Stdlib only — no extra dependency.
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

from okfgraph.errors import OKFError

__all__ = ["OKFClient", "OPS", "serve_file_for", "discover"]

#: Ops the server exposes (the router's canonical vocabulary, 0.11.0).
#: ``init`` is absent on purpose: it closes the router — a client must not
#: close the owner's database.
OPS = (
    # query
    "search", "read", "traverse",
    # ingest / export
    "ingest", "export_bundle", "export_concept",
    # import
    "import_bundle", "import_file", "import_files", "sync",
    # write (O11)
    "update", "supersede", "delete",
    # health / hygiene
    "doctor", "diff", "reindex", "list_broken_links", "repair_links",
    "list_deleted", "recover_deleted", "purge_deleted", "detach",
    # images
    "list_images", "get_image",
    # info
    "get_by_id", "list_directory",
)


def serve_file_for(db_path: str | os.PathLike) -> Path:
    """The discovery file a running ``okf serve`` writes next to the db."""
    p = Path(db_path).expanduser().resolve()
    return p.with_name(p.name + ".serve.json")


def _pid_alive(pid: int) -> bool:
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
    """URL of a live server owning ``db_path``, or None.

    Live = the discovery file names a running pid **and** ``/health``
    answers for the same database. A stale file (crashed server) is
    ignored, never trusted.
    """
    sf = serve_file_for(db_path)
    try:
        info = json.loads(sf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url, pid = info.get("url"), info.get("pid")
    if not url or not isinstance(pid, int) or not _pid_alive(pid):
        return None
    try:
        health = OKFClient(url, timeout=timeout).health()
    except Exception:
        return None
    same = Path(health.get("db_path", "")).resolve() == Path(db_path).expanduser().resolve()
    return url if same else None


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):  # noqa: D401 - http.client hook
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


class OKFClient:
    """Router-shaped client for an ``okf serve`` owner (0.11.0, O4).

    ``OKFClient("http://127.0.0.1:8765")`` or
    ``OKFClient("unix:///run/okf.sock")``. Positional arguments bind by the
    router's own signatures, so ``client.search("q", limit=3)`` and
    ``router.search("q", limit=3)`` send the same call.
    """

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
            raise ValueError(f"OKFClient url must be http://host:port or unix:///path, got {url!r}")

    # -- transport ----------------------------------------------------------

    def _conn(self) -> http.client.HTTPConnection:
        if self._unix:
            return _UnixHTTPConnection(self._unix, self.timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        conn = self._conn()
        try:
            body = json.dumps(payload or {}, default=str).encode("utf-8")
            conn.request(method, path, body=body if method == "POST" else None,
                         headers={"Content-Type": "application/json"})
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
            # Outcome errors (doctor/diff/lint) carry their report in the
            # envelope's data slot.
            data=env.get("data"),
        )

    # -- API ----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self._unwrap(self._request("GET", "/health"))

    def call(self, op: str, **kwargs) -> Any:
        """Run one op on the server and return its ``data``."""
        if op not in OPS:
            raise OKFError("BAD_VALUE", f"unknown op {op!r}", fields={"op": op})
        return self._unwrap(self._request("POST", f"/ops/{op}", kwargs))

    def close(self) -> None:
        """Nothing to release: one connection per call."""

    def __enter__(self) -> "OKFClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getattr__(self, name: str):
        if name not in OPS:
            raise AttributeError(name)
        from okfgraph.router import OKFRouter
        target = getattr(OKFRouter, name)
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            sig = None

        def op(*args, **kwargs):
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
