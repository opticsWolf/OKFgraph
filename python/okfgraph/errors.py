"""Public error classes and result envelope for the Rust-backed facade."""
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

USAGE_CODES = frozenset({"BAD_VALUE", "MISSING_PARAM", "FILE_NOT_FOUND", "CONFIG_INVALID"})
STATE_CODES = frozenset({
    "UNKNOWN_CONCEPT", "UNKNOWN_ASSET", "NOT_RECOVERABLE", "DETACHED",
    "DETACH_REFUSED", "PURGE_REFUSED_ABSENT_ROOT", "SEARCH_UNAVAILABLE",
    "WRITE_LOCK_TIMEOUT", "DB_LOCKED", "DIM_MISMATCH", "MODEL_PIN_MISMATCH",
    "PRECISION_PIN_MISMATCH", "SPACE_PIN_MISMATCH", "IMAGE_PIN_MISMATCH",
    "VISION_INCOMPATIBLE", "EXTERNAL_UNAVAILABLE", "CONVERSION_FAILED",
    "NO_ORT_RUNTIME", "EMBROIDER_TOO_OLD", "REEMBED_IN_PROGRESS", "INTERNAL",
})
OUTCOME_CODES = frozenset({"DOCTOR_FINDINGS", "LINT_ERRORS", "DIFF_DIFFERENT"})
_VALID_CODES = USAGE_CODES | STATE_CODES | OUTCOME_CODES


class OKFError(Exception):
    kind = "state"

    def __new__(cls, code, *args, **kwargs):
        if code not in _VALID_CODES:
            raise ValueError(f"unknown error code: {code}")
        target = UsageError if code in USAGE_CODES else OutcomeError if code in OUTCOME_CODES else StateError
        inst = Exception.__new__(target)
        if not issubclass(target, cls):
            inst.__init__(code, *args, **kwargs)
        return inst

    def __init__(self, code, message, *, op=None, fields=None, remedy=None, data=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.op = op
        self.fields = fields
        self.remedy = remedy
        self.data = data

    def __reduce__(self):
        return (type(self), (self.code, self.message), self.__dict__)

    @classmethod
    def from_native(cls, payload: str) -> "OKFError":
        data = json.loads(payload)
        return cls(
            data.get("code", "INTERNAL"),
            data.get("message", ""),
            op=data.get("op"),
            fields=data.get("fields"),
            remedy=data.get("remedy"),
            data=data.get("data"),
        )

    @property
    def exit_code(self) -> int:
        return 2 if self.code in USAGE_CODES else 1

    def titles(self) -> str:
        title = f"{self.code}: {self.message}"
        return f"{title} ({self.remedy})" if self.remedy else title

    def envelope(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fields": self.fields,
            "remedy": self.remedy,
        }


class UsageError(OKFError, ValueError):
    kind = "usage"


class StateError(OKFError, RuntimeError):
    kind = "state"


class OutcomeError(OKFError, RuntimeError):
    kind = "outcome"


class OKFWarning(UserWarning):
    def __init__(self, message, *, code=None):
        super().__init__(message)
        self.code = code
        self.message = message


def _warning_text(w: Any) -> str:
    return str(getattr(w, "message", w))


def envelope(op: str, data: Any = None, *, warnings=(), error=None) -> Dict[str, Any]:
    if error is not None and data is None:
        data = error.data
    return {
        "ok": error is None,
        "op": op,
        "data": data,
        "warnings": [_warning_text(w) for w in warnings],
        "error": error.envelope() if error is not None else None,
    }


def internal_error(exc: BaseException, *, op: Optional[str] = None, remedy: Optional[str] = None) -> OKFError:
    return OKFError(
        "INTERNAL",
        f"{type(exc).__name__}: {exc}".strip(),
        op=op,
        fields={"type": type(exc).__name__},
        remedy=remedy,
    )


def dumps_envelope(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str, indent=2)
