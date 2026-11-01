"""``okf serve``: one process owns the database, every other client talks to it.

0.11.0, O4. Ladybug allows one process per database file, and MCP over stdio
starts one server per client — so two harnesses, or Silk plus one MCP
client, or the CLI next to either, collided with ``DB_LOCKED``.

The owner opens one :class:`~okfgraph.OKFRouter` and serves it two ways on
one listener:

* ``/mcp`` — MCP over streamable HTTP (the same tools as ``okf-mcp``);
* ``POST /ops/<op>`` — the router's ops as JSON in, the result envelope
  ``{ok, op, data, warnings, error}`` out (what :class:`OKFClient` and the
  CLI use); ``GET /health`` answers who owns which database.

Every call runs on a worker thread under one lock, so writes queue in the
server (on top of the existing inter-process write lock) and the Ladybug
connection is never used by two threads at once. A discovery file
``<db_path>.serve.json`` lets the CLI find the owner; it is removed on a
clean shutdown and ignored when stale.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from okfgraph.client import OPS, serve_file_for
from okfgraph.errors import OKFError, envelope, internal_error

logger = logging.getLogger(__name__)

__all__ = ["LockedRouter", "build_app", "parse_listen", "serve"]


class LockedRouter:
    """Serialise every op of one router behind one lock.

    The MCP tools and the JSON endpoint share this object, so a search from
    an MCP client and an import from the CLI never touch the Ladybug
    connection at the same time.
    """

    def __init__(self, router):
        self._router = router
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def __getattr__(self, name: str):
        attr = getattr(self._router, name)
        if not callable(attr):
            return attr

        def locked(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)

        locked.__name__ = name
        return locked


def parse_listen(listen: str) -> Tuple[Optional[str], Optional[int], Optional[str], str]:
    """``unix:///path`` or ``host:port`` → ``(host, port, uds, url)``."""
    if listen.startswith("unix://"):
        path = urlparse(listen).path or listen[len("unix://"):]
        return None, None, path, f"unix://{path}"
    host, _, port = listen.rpartition(":")
    if not port.isdigit():
        raise OKFError("BAD_VALUE", f"--listen must be HOST:PORT or unix:///path, got {listen!r}",
                       fields={"listen": listen})
    host = host or "127.0.0.1"
    return host, int(port), None, f"http://{host}:{int(port)}"


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def build_app(router, settings=None):
    """Starlette app serving ``router`` over MCP (``/mcp``) and JSON ops.

    ``router`` is used as-is (and not closed here): the caller owns it.
    Returns ``(app, locked_router)``.
    """
    import anyio
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from okfgraph.mcp_server import create_mcp_server

    locked = LockedRouter(router)
    mcp = create_mcp_server(settings, router=locked)

    async def run_op(op: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        def call():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    data = getattr(locked, op)(**kwargs)
                except OKFError as exc:
                    return envelope(op, error=exc, warnings=caught)
                except TypeError as exc:
                    return envelope(op, error=OKFError(
                        "BAD_VALUE", f"{op}: {exc}", op=op,
                        fields={"params": sorted(kwargs)}), warnings=caught)
                except Exception as exc:  # noqa: BLE001 - adapter boundary
                    logger.exception("op %s failed", op)
                    return envelope(op, error=internal_error(exc, op=op),
                                    warnings=caught)
                return envelope(op, data=_json_safe(data), warnings=caught)
        return await anyio.to_thread.run_sync(call)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse(envelope("health", data={
            "db_path": str(Path(router.db_path).resolve()),
            "pid": os.getpid(),
            "ops": list(OPS),
        }))

    @mcp.custom_route("/ops/{op}", methods=["POST"])
    async def ops(request: Request) -> JSONResponse:
        op = request.path_params["op"]
        if op not in OPS:
            return JSONResponse(envelope(op, error=OKFError(
                "BAD_VALUE", f"unknown op {op!r}", op=op,
                fields={"op": op, "ops": list(OPS)})), status_code=404)
        try:
            raw = await request.body()
            kwargs = json.loads(raw or b"{}")
            if not isinstance(kwargs, dict):
                raise ValueError("body must be a JSON object of keyword arguments")
        except ValueError as exc:
            return JSONResponse(envelope(op, error=OKFError(
                "BAD_VALUE", f"bad request body: {exc}", op=op)), status_code=400)
        env = await run_op(op, kwargs)
        return JSONResponse(env)

    app = mcp.streamable_http_app()
    return app, locked


def serve(settings, listen: str, *, ready: Optional[threading.Event] = None,
          stop: Optional[threading.Event] = None) -> None:
    """Open the owner router and serve until interrupted (``okf serve``).

    ``ready``/``stop`` let tests run the server on a thread.
    """
    import uvicorn

    from okfgraph.router import OKFRouter

    host, port, uds, url = parse_listen(listen)
    if uds and os.name == "nt":
        # asyncio has no Unix-socket server on Windows: serve TCP loopback on
        # a free port; the discovery file carries the real URL to clients.
        logger.warning("okf serve: Unix sockets are not supported on Windows; "
                       "listening on 127.0.0.1 (free port) instead of %s", url)
        host, port, uds = "127.0.0.1", 0, None
    if uds and Path(uds).exists():
        Path(uds).unlink()
    router = OKFRouter(
        db_path=settings.db_path,
        bundle_root=settings.bundle_root or str(Path(settings.db_path).parent),
        **settings.router_kwargs(),
    )
    sf = serve_file_for(settings.db_path)
    try:
        app, _locked = build_app(router, settings)
        config = uvicorn.Config(app, host=host or "127.0.0.1", port=port or 0,
                                uds=uds, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        if stop is not None:
            def _watch():
                stop.wait()
                server.should_exit = True
            threading.Thread(target=_watch, daemon=True).start()

        def _started():
            nonlocal url
            if not uds:
                # Port 0 binds a free port: publish the one actually bound.
                bound = server.servers[0].sockets[0].getsockname()
                url = f"http://{host or '127.0.0.1'}:{bound[1]}"
            sf.write_text(json.dumps({
                "url": url, "pid": os.getpid(), "db_path": str(Path(settings.db_path).resolve()),
                "started_at": int(time.time()),
            }), encoding="utf-8")
            logger.info("okf serve: %s owns %s", url, settings.db_path)
            if ready is not None:
                ready.set()

        orig_startup = server.startup

        async def startup(sockets=None):
            await orig_startup(sockets=sockets)
            _started()

        server.startup = startup
        server.run()
    finally:
        try:
            info = json.loads(sf.read_text(encoding="utf-8"))
            if info.get("pid") == os.getpid():
                sf.unlink()
        except (OSError, ValueError):
            pass
        router.close()
        if uds and Path(uds).exists():
            try:
                Path(uds).unlink()
            except OSError:
                pass
