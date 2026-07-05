"""Minimal, dependency-free HTML table → GFM pipe-table converter.

Parses a single ``<table>`` into rows of cell text. Bails (``complex=True``)
on row/colspans so that complex tables stay as raw HTML (which Markdown
renderers accept).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional


class _SimpleTableParser(HTMLParser):
    """Parses ONE ``<table>`` into rows of cell-text."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.header_flags: list[bool] = []
        self._cur: list[str] = []
        self._buf: list[str] = []
        self._in_cell = False
        self._row_has_th = False
        self.complex = False

    # -- HTMLParser hooks ---------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:
        ad = dict(attrs)
        if tag == "tr":
            self._cur = []
            self._row_has_th = False
        elif tag in ("td", "th"):
            if ad.get("colspan") not in (None, "1") or ad.get("rowspan") not in (None, "1"):
                self.complex = True
            self._in_cell = True
            self._buf = []
            if tag == "th":
                self._row_has_th = True
        elif tag == "br":
            if self._in_cell:
                self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip().replace("|", r"\|")
            self._cur.append(text)
            self._in_cell = False
        elif tag == "tr":
            if self._cur:
                self.rows.append(self._cur)
                self.header_flags.append(self._row_has_th)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._buf.append(data)


def _one_table_to_gfm(html: str) -> Optional[str]:
    """Convert a single ``<table>`` block to a GFM pipe table, or None if complex."""
    p = _SimpleTableParser()
    try:
        p.feed(html)
    except Exception:
        return None
    if p.complex or not p.rows:
        return None

    ncols = max(len(r) for r in p.rows)
    rows = [r + [""] * (ncols - len(r)) for r in p.rows]
    header_idx = next((i for i, f in enumerate(p.header_flags) if f), 0)
    header = rows[header_idx]
    body = [r for i, r in enumerate(rows) if i != header_idx]

    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def html_tables_to_gfm(md: str) -> str:
    """Replace simple ``<table>`` blocks with GFM pipe tables; leave complex ones as-is."""

    def _repl(m: "re.Match") -> str:
        gfm = _one_table_to_gfm(m.group(0))
        return f"\n\n{gfm}\n\n" if gfm else m.group(0)

    return re.sub(r"<table\b.*?</table>", _repl, md, flags=re.IGNORECASE | re.DOTALL)
