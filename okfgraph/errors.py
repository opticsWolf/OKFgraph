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

from typing import Any, Dict, Optional

__all__ = [
    "OKFError",
    "UsageError",
    "StateError",
    "OutcomeError",
    "envelope",
]

USAGE_CODES = {
    "BAD_VALUE",
    "MISSING_PARAM",
    "FILE_NOT_FOUND",
    "WRITE_LOCK_TIMEOUT",
    "DIM_MISMATCH",
    "MODEL_PIN_MISMATCH",
    "PRECISION_PIN_MISMATCH",
    "IMAGE_PIN_MISMATCH",
    "VISION_INCOMPATIBLE",
    "NO_ORT_RUNTIME",
    "CONFIG_INVALID",
}

STATE_CODES = {
    "UNKNOWN_CONCEPT",
    "UNKNOWN_ASSET",
    "NOT_RECOVERABLE",
    "DETACHED",
    "PURGE_REFUSED_ABSENT_ROOT",
    "SEARCH_UNAVAILABLE",
}

OUTCOME_CODES = {
    "DOCTOR_FINDINGS",
    "LINT_ERRORS",
    "DIFF_DIFFERENT",
}

#: Outcome codes that signal a usage-mapping exit code 2 (diff):
#: everything else in OUTCOME_CODES exits 1 while still carrying its body.
_UNUSUAL_OUTCOME_EXITS = {"DIFF_DIFFERENT"}

_VALID_CODES = USAGE_CODES | STATE_CODES | OUTCOME_CODES | {"INTERNAL"}


class OKFError(Exception):
    """A failed operation with a stable, documented error code."""

    kind = "state"

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
        if self.kind == "usage":
            return 2
        if self.kind == "outcome":
            return 2 if self.code in _UNUSUAL_OUTCOME_EXITS else 1
        return 1

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


class UsageError(OKFError):
    """Caller passed something there is no sensible interpretation of."""

    kind = "usage"


class StateError(OKFError):
    """The request was reasonable; the current state cannot serve it."""

    kind = "state"


class OutcomeError(OKFError):
    """A command ran fine and the outcome itself is the notable result.

    The envelope still carries the findings as ``data`` (or the exit code
    maps them), so this reads as 'finished, with an outcome'.
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
