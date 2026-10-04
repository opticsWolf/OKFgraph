"""Typed operation failures and the result envelope.

okfgraph 0.10 unification: operations are dict/list-returning and raise
:class:`OKFError` on failure. Adapters translate:

* CLI: ``[ERROR] <code>: <message>`` on stderr. Exit code 2 for usage
  problems (:class:`UsageError`), 1 for state problems
  (:class:`StateError`, :class:`OutcomeError`).
* MCP: handlers catch and return ``"error: <code>: <message>"``.
* Python: catch the exception, or render :func:`envelope`.

Codes are a closed set. ``INTERNAL`` is the fallback for bugs and
unclassified exceptions; adapters use it when they catch a non-OKFError
exception.
"""
from __future__ import annotations

import json
import warnings as _warnings_module
import builtins as _builtins

from typing import Any, Dict, Optional

# `UserWarning` is a builtin; this env's `warnings` module does not re-export
# it reliably, so take it from builtins (present in every CPython).
_UserWarning: type = __import__("builtins").UserWarning

__all__ = [
    "OKFError",
    "UsageError",
    "StateError",
    "OutcomeError",
    "OKFWarning",
    "envelope",
    "internal_error",
]

USAGE_CODES = {
    "BAD_VALUE",
    "MISSING_PARAM",
    "FILE_NOT_FOUND",
    "CONFIG_INVALID",
}

STATE_CODES = {
    "UNKNOWN_CONCEPT",
    "UNKNOWN_ASSET",
    "NOT_RECOVERABLE",
    "DETACHED",
    "PURGE_REFUSED_ABSENT_ROOT",
    "SEARCH_UNAVAILABLE",
    "WRITE_LOCK_TIMEOUT",
    "DIM_MISMATCH",
    "MODEL_PIN_MISMATCH",
    "PRECISION_PIN_MISMATCH",
    "IMAGE_PIN_MISMATCH",
    "VISION_INCOMPATIBLE",
    "NO_ORT_RUNTIME",
}

OUTCOME_CODES = {
    "DOCTOR_FINDINGS",
    "LINT_ERRORS",
    "DIFF_DIFFERENT",
}

_VALID_CODES = USAGE_CODES | STATE_CODES | OUTCOME_CODES | {"INTERNAL"}


class OKFError(Exception):
    """A failed operation with a stable, documented error code.

    The subclass is picked by code (rule §4): usage codes yield a
    ValueError-compatible exception, state/outcome codes a
    RuntimeError-compatible one — so legacy ``except ValueError``/
    ``except RuntimeError`` sites keep working while conversion proceeds.
    """

    kind = "state"

    def __new__(cls, code, message, **kw):
        if code not in _VALID_CODES:
            raise ValueError(f"unknown error code: {code}")
        if cls is OKFError:
            if code in USAGE_CODES:
                cls = UsageError
            elif code in OUTCOME_CODES:
                cls = OutcomeError
            else:
                cls = StateError
        return super().__new__(cls)

    def __init__(
        self,
        code: str,
        message: str,
        *,
        op: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        remedy: Optional[str] = None,
    ):
        if code not in _VALID_CODES:
            raise ValueError(f"unknown error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.op = op
        self.fields = fields
        self.remedy = remedy

    @property
    def exit_code(self) -> int:
        # §4: usage -> 2, state/outcome -> 1 (diff is a domain outcome).
        return 2 if self.code in USAGE_CODES else 1

    def titles(self) -> str:
        """Adapter-facing one-liner: ``CODE: message``."""
        title = f"{self.code}: {self.message}"
        return f"{title} {self.remedy}" if self.remedy else title

    def envelope(self) -> Dict[str, Any]:
        """Machine-facing body for the ``error`` field of an envelope."""
        return {
            "code": self.code,
            "message": self.message,
            "fields": self.fields,
            "remedy": self.remedy,
        }


class UsageError(OKFError, ValueError):
    """Caller passed something there is no sensible interpretation of."""

    kind = "usage"


class StateError(OKFError, RuntimeError):
    """The request was reasonable; the current state cannot serve it."""

    kind = "state"


class OutcomeError(OKFError, RuntimeError):
    """A command ran fine and the outcome itself is the notable result.

    The envelope still carries the findings as ``data`` (the adapter's
    human renderer shows them), with exit code 1 signalling the outcome.
    """

    kind = "outcome"


def envelope(op: str, data: Any = None, *, warnings=(), error=None) -> Dict[str, Any]:
    """Build the uniform result transport: ``{ok, op, data, warnings, error?}``."""
    err = error.envelope() if error is not None else None
    return {
        "ok": err is None,
        "op": op,
        "data": data,
        "warnings": [str(w) for w in warnings],
        "error": err,
    }

class OKFWarning(_UserWarning):
    """A non-fatal anomaly; adapters surface it on ``envelope()["warnings"]``."""

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message


def internal_error(exc: BaseException, *, op: Optional[str] = None,
                   remedy: Optional[str] = None) -> "OKFError":
    """Wrap an unclassified exception as ``INTERNAL`` (code table §4).

    Adapters use this as the catch-all so no bare traceback ever crosses
    the CLI/MCP boundary; the exception type rides in ``fields["type"]``.
    """
    return OKFError(
        "INTERNAL",
        f"{type(exc).__name__}: {exc}".strip(),
        op=op,
        fields={"type": type(exc).__name__},
        remedy=remedy,
    )


def envelope_of_error(err: BaseException, *, op: str, warnings=()) -> Dict[str, Any]:
    """Build the error envelope for *any* exception (typed or not)."""
    typed = err if isinstance(err, OKFError) else internal_error(err, op=op)
    return envelope(op, error=typed, warnings=warnings)


def dumps_envelope(payload: Dict[str, Any]) -> str:
    """Canonical JSON frame for the MCP/CLI ``--json`` transport."""
    return json.dumps(payload, default=str, indent=2)
