"""Typed operation failures and the result envelope.

okfgraph 0.10 unification: operations are dict/list-returning and raise
:class:`OKFError` on failure. Adapters translate:

* CLI: ``[ERROR] <code>: <message>`` on stderr (plus the envelope on
  stdout under ``--json``). Exit code 2 for usage problems
  (:class:`UsageError`), 1 for state problems and outcomes
  (:class:`StateError`, :class:`OutcomeError`).
* MCP: the tool raises, so the result carries ``isError: true`` with the
  error envelope as text.
* Python: catch the exception, or render :func:`envelope`.

Codes are a closed set and the code alone picks the class —
``UsageError("UNKNOWN_CONCEPT", …)`` still yields a :class:`StateError`.
``INTERNAL`` is the fallback for bugs and unclassified exceptions;
adapters use it when they catch a non-OKFError exception.

Outcome errors (diff differs, doctor findings, lint errors) are domain
results, not failures of the call: the would-be result rides on
``err.data`` and lands in the envelope's ``data``, never inside ``error``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

__all__ = [
    "OKFError",
    "UsageError",
    "StateError",
    "OutcomeError",
    "OKFWarning",
    "USAGE_CODES",
    "STATE_CODES",
    "OUTCOME_CODES",
    "envelope",
    "internal_error",
    "dumps_envelope",
]

USAGE_CODES = frozenset({
    "BAD_VALUE",
    "MISSING_PARAM",
    "FILE_NOT_FOUND",
    "CONFIG_INVALID",
})

STATE_CODES = frozenset({
    "UNKNOWN_CONCEPT",
    "UNKNOWN_ASSET",
    "NOT_RECOVERABLE",
    "DETACHED",
    "DETACH_REFUSED",
    "PURGE_REFUSED_ABSENT_ROOT",
    "SEARCH_UNAVAILABLE",
    "WRITE_LOCK_TIMEOUT",
    "DIM_MISMATCH",
    "MODEL_PIN_MISMATCH",
    "PRECISION_PIN_MISMATCH",
    "IMAGE_PIN_MISMATCH",
    "VISION_INCOMPATIBLE",
    "NO_ORT_RUNTIME",
    "INTERNAL",
})

OUTCOME_CODES = frozenset({
    "DOCTOR_FINDINGS",
    "LINT_ERRORS",
    "DIFF_DIFFERENT",
})

_VALID_CODES = USAGE_CODES | STATE_CODES | OUTCOME_CODES


class OKFError(Exception):
    """A failed operation with a stable, documented error code.

    The class is picked by code, whichever constructor is called: usage
    codes yield a ValueError-compatible :class:`UsageError`, state codes
    a RuntimeError-compatible :class:`StateError`, outcome codes an
    :class:`OutcomeError` — so ``except ValueError``/``except
    RuntimeError`` sites keep working.
    """

    kind = "state"

    def __new__(cls, code, *args, **kwargs):
        if code not in _VALID_CODES:
            raise ValueError(f"unknown error code: {code}")
        if code in USAGE_CODES:
            target = UsageError
        elif code in OUTCOME_CODES:
            target = OutcomeError
        else:
            target = StateError
        # Not super(): under UsageError's MRO that is ValueError.__new__,
        # which rejects a StateError target.
        inst = Exception.__new__(target)
        if not issubclass(target, cls):
            # Python only runs __init__ when __new__ returns an instance of
            # the called class (e.g. UsageError("UNKNOWN_CONCEPT") yields a
            # StateError), so initialise the sibling here.
            inst.__init__(code, *args, **kwargs)
        return inst

    def __init__(
        self,
        code: str,
        message: str,
        *,
        op: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        remedy: Optional[str] = None,
        data: Any = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.op = op
        self.fields = fields
        self.remedy = remedy
        # Outcome errors only: the result the op would have returned.
        self.data = data

    def __reduce__(self):
        # Default exception pickling replays args=(message,) into __new__,
        # which would read the message as a code.
        return (type(self), (self.code, self.message), self.__dict__)

    @property
    def exit_code(self) -> int:
        # §4: usage -> 2, state/outcome -> 1 (diff is a domain outcome).
        return 2 if self.code in USAGE_CODES else 1

    def titles(self) -> str:
        """Adapter-facing one-liner: ``CODE: message [remedy]``."""
        title = f"{self.code}: {self.message}"
        return f"{title} ({self.remedy})" if self.remedy else title

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

    ``data`` carries the result (the envelope's ``data``; the CLI's human
    renderer shows it), with exit code 1 signalling the outcome.
    """

    kind = "outcome"


class OKFWarning(UserWarning):
    """A non-fatal anomaly; adapters surface it on ``envelope()["warnings"]``."""

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message


def _warning_text(w: Any) -> str:
    # warnings.catch_warnings(record=True) yields WarningMessage records,
    # whose str() is a debug repr; the message is what callers want.
    return str(getattr(w, "message", w))


def envelope(op: str, data: Any = None, *, warnings=(), error=None) -> Dict[str, Any]:
    """Build the uniform result transport: ``{ok, op, data, warnings, error}``.

    With an outcome ``error`` and no explicit ``data``, the error's
    ``data`` (the report) fills the envelope's ``data`` slot.
    """
    if error is not None and data is None:
        data = error.data
    return {
        "ok": error is None,
        "op": op,
        "data": data,
        "warnings": [_warning_text(w) for w in warnings],
        "error": error.envelope() if error is not None else None,
    }


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


def dumps_envelope(payload: Dict[str, Any]) -> str:
    """Canonical JSON frame for the MCP/CLI ``--json`` transport."""
    return json.dumps(payload, default=str, indent=2)
