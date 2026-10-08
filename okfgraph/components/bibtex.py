"""Minimal BibTeX parser for ``ingest --kind bib`` (stdlib only).

Scope: one concept per ``@entry``. Handles braced (nestable), ``"quoted"``,
bare and ``#``-concatenated values, ``%`` comments, ``@string`` macro
definition + expansion; ``@preamble``/``@comment`` are skipped. Anything
else malformed is collected per-entry (``skipped``) instead of aborting
the run — a 5,000-entry bibliography with 3 broken entries imports 4,997
concepts plus 3 warnings, not a traceback.

Deliberately not handled: LaTeX-to-Unicode conversion (kept verbatim),
``@crossref`` resolution (fields stay as written).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BibEntry:
    entrytype: str
    key: str
    fields: Dict[str, str] = field(default_factory=dict)
    line: int = 0


@dataclass
class BibFile:
    entries: List[BibEntry] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)


def _strip_comments(text: str) -> str:
    """Drop ``%``-to-end-of-line comments (outside values, best effort).

    A ``%`` inside a braced/quoted value is data (e.g. ``50\\%``), so this
    runs as a small state machine rather than a line regex.
    """
    out: List[str] = []
    i, n = 0, len(text)
    depth = 0
    in_quote = False
    while i < n:
        ch = text[i]
        if in_quote:
            out.append(ch)
            if ch == '"' and text[i - 1:i] != "\\":
                in_quote = False
            i += 1
        elif ch == '"':
            in_quote = True
            out.append(ch)
            i += 1
        elif ch == "{":
            depth += 1
            out.append(ch)
            i += 1
        elif ch == "}":
            depth = max(0, depth - 1)
            out.append(ch)
            i += 1
        elif ch == "%" and depth == 0:
            while i < n and text[i] != "\n":
                i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _read_braced(text: str, i: int) -> Tuple[str, int]:
    """Read ``{...}`` starting at ``text[i] == '{'``; return (inner, next)."""
    assert text[i] == "{"
    depth = 0
    j = i
    n = len(text)
    while j < n:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces")


def _read_quoted(text: str, i: int) -> Tuple[str, int]:
    """Read ``\"...\"`` starting at ``text[i] == '\"'``; braces may nest."""
    assert text[i] == '"'
    j = i + 1
    n = len(text)
    buf: List[str] = []
    depth = 0
    while j < n:
        ch = text[j]
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == '"' and depth == 0:
            return "".join(buf), j + 1
        else:
            buf.append(ch)
        j += 1
    raise ValueError("unterminated quote")


def _split_fields(body: str) -> List[str]:
    """Split an entry body on top-level commas (depth/quote aware)."""
    parts: List[str] = []
    depth = 0
    in_quote = False
    cur: List[str] = []
    for ch in body:
        if in_quote:
            cur.append(ch)
            if ch == '"':
                in_quote = False
        elif ch == '"':
            in_quote = True
            cur.append(ch)
        elif ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _eval_value(raw: str, strings: Dict[str, str]) -> str:
    """Evaluate one field value: concat parts, expand ``@string`` macros."""
    parts = [p.strip() for p in raw.split("#")]
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) >= 2 and part[0] == "{" and part[-1] == "}":
            out.append(part[1:-1])
        elif len(part) >= 2 and part[0] == '"' and part[-1] == '"':
            inner, _ = _read_quoted(part, 0)
            out.append(inner)
        elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]*", part):
            key = part.lower()
            if key in strings:
                out.append(strings[key])
            elif part.isdigit():
                out.append(part)
            else:
                out.append(part)  # unknown macro: keep verbatim, don't crash
        else:
            out.append(part)  # bare number or literal
    return "".join(out)


def _find_entries(text: str) -> List[Tuple[str, str, str, int]]:
    """Yield (entrytype, key, body, line) for each ``@type{key, ...}`` / ``(...)``."""
    found = []
    i, n = 0, len(text)
    while i < n:
        at = text.find("@", i)
        if at == -1:
            break
        m = re.match(r"@([A-Za-z]+)\s*([{\(])", text[at:])
        if not m:
            i = at + 1
            continue
        etype = m.group(1).lower()
        opener = m.group(2)
        closer = "}" if opener == "{" else ")"
        line = text.count("\n", 0, at) + 1
        j = at + m.end()
        if etype in ("comment", "preamble"):
            # Opaque bodies: skip a balanced span, but real-world files
            # contain unbalanced junk here — then fall back to end of
            # line instead of swallowing the rest of the file. Preamble
            # bodies are spec-quoted, so quotes are honoured only there
            # (a stray quote in a comment must not break counting).
            line_end = text.find("\n", j)
            line_end = n if line_end == -1 else line_end
            depth = 1
            k = j
            in_q = False
            while k < n and depth:
                ch = text[k]
                if etype == "preamble" and ch == '"':
                    in_q = not in_q
                elif not in_q:
                    if ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                k += 1
            i = k if depth == 0 else line_end + 1
            continue
        if etype == "string":
            # No citation key: `@string{name = value}` — the whole span
            # is the body.
            depth = 1
            bstart = j
            in_quote = False
            while j < n and depth:
                ch = text[j]
                if in_quote:
                    if ch == '"':
                        in_quote = False
                elif ch == '"':
                    in_quote = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                j += 1
            if depth != 0:
                i = at + 1
                found.append((etype, "", "", line))
                continue
            found.append((etype, "", text[bstart:j - 1], line))
            i = j
            continue
        # Key runs to the first top-level comma.
        kstart = j
        while j < n and text[j] != ",":
            if text[j] == closer:
                break
            j += 1
        key = text[kstart:j].strip()
        if j >= n or text[j] != ",":
            i = j
            continue
        j += 1
        depth = 1
        bstart = j
        in_quote = False
        while j < n and depth:
            ch = text[j]
            if in_quote:
                if ch == '"':
                    in_quote = False
            elif ch == '"':
                in_quote = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
            j += 1
        if depth != 0:
            # Unbalanced: report the entry as skipped, resume after "@".
            found.append((etype, key, "", line))
            i = at + 1
            continue
        found.append((etype, key, text[bstart:j - 1], line))
        i = j
    return found


def parse_bibtex(text: str) -> BibFile:
    """Parse BibTeX source into entries + per-entry skip reports."""
    result = BibFile()
    strings: Dict[str, str] = {}
    text = _strip_comments(text)
    for etype, key, body, line in _find_entries(text):
        if etype == "string":
            if not body.strip():
                result.skipped.append(
                    {"line": str(line), "reason": "bad @string: unbalanced delimiters"})
                continue
            try:
                for part in _split_fields(body):
                    name, _, value = part.partition("=")
                    name = name.strip().lower()
                    if name and value.strip():
                        strings[name] = _eval_value(value.strip(), strings)
            except ValueError as exc:
                result.skipped.append(
                    {"line": str(line), "reason": f"bad @string: {exc}"})
            continue
        if not key:
            result.skipped.append(
                {"line": str(line), "reason": f"@{etype} without a key"})
            continue
        if body == "" and etype not in ("comment", "preamble"):
            result.skipped.append(
                {"line": str(line),
                 "reason": f"@{etype}{{{key}}} has unbalanced delimiters"})
            continue
        try:
            fields: Dict[str, str] = {}
            for part in _split_fields(body):
                name, sep, value = part.partition("=")
                if not sep:
                    continue
                fields[name.strip().lower()] = _eval_value(
                    value.strip(), strings)
        except ValueError as exc:
            result.skipped.append(
                {"line": str(line),
                 "reason": f"@{etype}{{{key}}}: {exc}"})
            continue
        result.entries.append(
            BibEntry(entrytype=etype, key=key, fields=fields, line=line))
    return result


def unwrap_braces(value: str) -> str:
    """Drop brace groups but keep their content (``{DNA}`` → ``DNA``)."""
    prev = None
    cur = value
    while prev != cur:
        prev = cur
        cur = re.sub(r"{([^{}]*)}", r"\1", cur)
    return cur


def slugify_key(key: str) -> str:
    """Map a citation key to one safe ID level (case preserved).

    Keeps ``[A-Za-z0-9_.-]`` and collapses everything else to ``_`` — so no
    ``/`` (nesting) and no ``..`` (export traversal) can come from a key.
    Mirrors the ``_slugify_topic`` character class, minus lowercasing:
    citation keys are case-sensitive identifiers.
    """
    slug = re.sub(r"[^A-Za-z0-9_.\-]+", "_", key).strip("._")
    return slug[:64] or "untitled"
