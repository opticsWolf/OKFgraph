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
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from okfgraph.client import OPS, serve_file_for
from okfgraph.errors import OKFError, envelope, internal_error

logger = logging.getLogger(__name__)

__all__ = ["LockedRouter", "JobRegistry", "build_app", "parse_listen", "serve"]

#: Ops that may run as a job (``POST /ops/<op> {"mode": "job"}``). A job
#: returns a job id inline and keeps the worker on a thread the caller can
#: watch with ``GET /jobs/<id>`` and stop with ``POST /jobs/<id>/cancel``.
JOB_OPS = ("reembed",)


class JobRegistry:
    """In-memory jobs for ``okf serve`` (0.12.0).

    A long op — the re-embed over a big graph takes hours — used to ride
    one blocking HTTP request: Silk could only abandon it by dropping the
    connection, and the op kept the graph lock while every other call
    queued. The job lane instead returns a job id inline; the op thread
    runs under the same lock as every other call, checks a cancel flag
    between batches, and reports phase progress back into this registry.

    Job records live in memory only (a restarted owner has no jobs);
    terminal records retire one hour after they finish, so a client that
    saw the id can still read the result.

    ``cancel`` needs no graph lock: it only sets a flag the op checks at
    batch boundaries, so answering it never queues behind the very thing
    being cancelled.
    """

    KEEP_TERMINAL = 64
    RETIRE_AFTER = 3600.0      # seconds a terminal record stays visible

    def __init__(self, locked: "LockedRouter"):
        self._locked = locked
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._mu = threading.Lock()

    # ── record helpers (op thread + HTTP threads) ────────────────────
    def _new(self, op: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        rec = {
            "id": uuid.uuid4().hex[:12],
            "op": op,
            "status": "queued",
            "kwargs": {k: v for k, v in kwargs.items() if k != "mode"},
            "phase": None, "done": 0, "total": 0,
            "submitted": int(time.time()),
            "started": None, "finished": None,
            "result": None, "error": None,
            "cancel_event": threading.Event(),
            "_mu": threading.Lock(),
        }
        return rec

    @staticmethod
    def _set_progress(rec, phase: str, done: int, total: int) -> None:
        with rec["_mu"]:
            rec["phase"] = str(phase)
            rec["done"] = int(done)
            rec["total"] = int(total)

    @staticmethod
    def _set_status(rec, status: str, *, result=None, error=None) -> None:
        with rec["_mu"]:
            if status == "running" and rec["started"] is None:
                rec["started"] = time.time()
            rec["status"] = status
            if result is not None or status == "done" or status == "cancelled":
                rec["result"] = result
            if error is not None:
                rec["error"] = error
            rec["finished"] = time.time() if status in (
                "done", "cancelled", "failed") else None

    # ── lifecycle ────────────────────────────────────────────────────
    def running_job(self, op: str) -> Optional[Dict[str, Any]]:
        with self._mu:
            for rec in self._jobs.values():
                if rec["op"] == op and rec["status"] in (
                        "queued", "running", "cancelling"):
                    return rec
        return None

    def submit(self, op: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Start *op* as a background job; the record's public view."""
        if op not in JOB_OPS:
            raise OKFError("BAD_VALUE", f"{op!r} cannot run as a job",
                           op="jobs", fields={"job_ops": list(JOB_OPS)})
        busy = self.running_job(op)
        if busy is not None or getattr(
                self._locked._router, "_reembed_active", False):
            raise OKFError(
                "REEMBED_IN_PROGRESS",
                "another reembed is already running (or queued) on this graph",
                op="reembed",
                remedy="watch GET /jobs/<id>, or cancel it with "
                       "POST /jobs/<id>/cancel")
        rec = self._new(op, kwargs)
        with self._mu:
            self._retire_locked()
            self._jobs[rec["id"]] = rec

        cancelled = rec["cancel_event"].is_set
        progress = (lambda phase, done, total:
                    self._set_progress(rec, phase, done, total))

        def run():
            with self._locked.lock:
                self._set_status(rec, "running")
                kwargs2 = dict(rec["kwargs"])
                kwargs2["progress"] = progress
                kwargs2["cancelled"] = rec["cancel_event"].is_set
                try:
                    data = getattr(self._locked, op)(**kwargs2)
                except OKFError as exc:
                    self._set_status(rec, "failed", error=exc.envelope())
                except Exception as exc:  # noqa: BLE001
                    logger.exception("job %s failed", rec["id"])
                    # the record must stay JSONable: the envelope dict, not
                    # the exception object (dumps_envelope semantics)
                    self._set_status(
                        rec, "failed",
                        error=internal_error(exc, op=op).envelope())
                else:
                    was_cancelled = bool((data or {}).get("cancelled"))
                    self._set_status(
                        rec, "cancelled" if was_cancelled else "done",
                        result=_json_safe(data))

        threading.Thread(target=run, daemon=True,
                         name=f"okf-job-{op}-{rec['id']}").start()
        return rec

    def cancel(self, job_id: str) -> Dict[str, Any]:
        rec = self._jobs.get(job_id)
        if rec is None:
            raise OKFError("BAD_VALUE", f"unknown job {job_id!r}",
                           op="jobs", fields={"job": job_id})
        rec["cancel_event"].set()
        if rec["status"] in ("queued", "running"):
            with rec["_mu"]:
                if rec["status"] in ("queued", "running"):
                    rec["status"] = "cancelling"
        return self.view(rec)

    def view(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """The public JSON shape of one job record."""
        with rec["_mu"]:
            out = {k: v for k, v in rec.items()
                   if k not in ("_mu", "cancel_event", "kwargs")}
            out["kwargs"] = {k: v for k, v in rec["kwargs"].items()}
            out["elapsed"] = (round((rec["finished"] or time.time())
                                    - (rec["started"] or rec["submitted"]), 1)
                              if rec["started"] else 0.0)
        return out

    def get(self, job_id: str) -> Dict[str, Any]:
        rec = self._jobs.get(job_id)
        if rec is None:
            raise OKFError("BAD_VALUE", f"unknown job {job_id!r}",
                           op="jobs", fields={"job": job_id})
        return self.view(rec)

    def list(self) -> list:
        with self._mu:
            self._retire_locked()
            rows = [self.view(rec) for rec in self._jobs.values()]
        rows.sort(key=lambda r: r["submitted"], reverse=True)
        for r in rows:
            r.pop("result", None)
            r.pop("error", None)
        return rows[:self.KEEP_TERMINAL]

    def _retire_locked(self) -> None:
        """Drop terminal records past their visibility window."""
        now = time.time()
        for jid in [j for j, r in self._jobs.items()
                    if r.get("finished")
                    and now - r["finished"] > self.RETIRE_AFTER]:
            self._jobs.pop(jid, None)


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
    """Starlette app serving ``router`` over MCP (``/mcp``), JSON ops
    and (0.12.0) the job lane.

    ``router`` is used as-is (and not closed here): the caller owns it.
    Returns ``(app, locked_router)``.
    """
    import anyio
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from okfgraph.mcp_server import create_mcp_server

    locked = LockedRouter(router)
    jobs = JobRegistry(locked)
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
            "lanes": router.lanes(),   # no DB read, no session opened
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
        # A second reembed must never queue silently behind the batch lock:
        # refuse at submit with the dedicated code (the honest diagnostic of
        # the job lane; the op-level check stays as the race backstop).
        if op in JOB_OPS and getattr(router, "_reembed_active", False):
            return JSONResponse(envelope(op, error=OKFError(
                "REEMBED_IN_PROGRESS",
                "another reembed is already running on this graph",
                op=op, remedy="poll GET /jobs/<id> or cancel it")),
                status_code=409)
        # The job lane (0.12.0): a long op submitted with mode="job" answers
        # a job id inline and keeps running on a worker thread the client
        # polls (GET /jobs/<id>) and can stop (POST /jobs/<id>/cancel)
        # between op batches. The default stays the blocking call so
        # existing clients keep their contract.
        mode = None
        if op in JOB_OPS and "mode" in kwargs:
            mode = kwargs.pop("mode")
            if mode not in ("sync", "job"):
                return JSONResponse(envelope(op, error=OKFError(
                    "BAD_VALUE",
                    f"mode must be 'sync' or 'job', got {mode!r}", op=op)),
                    status_code=400)
        if mode == "job":
            try:
                rec = jobs.submit(op, kwargs)
            except OKFError as exc:
                code = getattr(exc, "code", "INTERNAL")
                return JSONResponse(
                    envelope(op, error=exc), status_code=409 if code ==
                    "REEMBED_IN_PROGRESS" else 400)
            return JSONResponse(envelope(
                op, data={"job": rec["id"], "status": rec["status"],
                          "op": rec["op"]}))
        env = await run_op(op, kwargs)
        return JSONResponse(env)

    @mcp.custom_route("/jobs", methods=["GET"])
    async def jobs_list(request: Request) -> JSONResponse:
        return JSONResponse(envelope("jobs", data=jobs.list()))

    @mcp.custom_route("/jobs/{job_id}", methods=["GET"])
    async def jobs_get(request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        try:
            data = jobs.get(job_id)
        except OKFError as exc:
            return JSONResponse(envelope("job", error=exc), status_code=404)
        return JSONResponse(envelope("job", data=data))

    @mcp.custom_route("/jobs/{job_id}/cancel", methods=["POST"])
    async def jobs_cancel(request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        return JSONResponse(envelope("jobs.cancel", data=jobs.cancel(job_id)))

    app = mcp.streamable_http_app()
    app.state.jobs = jobs
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
        bundle_root=settings.default_bundle_root(str(Path(settings.db_path).parent)),
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
