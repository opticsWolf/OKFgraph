"""Source-producer seam: databases (and future formats) to bundles.

The knowledge graph never talks to a data source directly — it talks to a
:class:`SourceProducer` that emits a plain markdown bundle under an
``output_prefix`` namespace. Normal ``import --all`` then picks the files
up, so emitted links become ``LINKS_TO`` edges through existing import and
the bundle stays source of truth.

Mirrors the :mod:`converters` philosophy: the provider owns its options, a
missing optional dependency is a clear error, never a silent fallback. All
producers are deterministic (sorted enumeration, fixed templates, no
timestamps) and stdlib-only unless the provider declares an extra.

Planned providers: SQLite (here, stdlib), DOCX (needs an extra — the seam
is ready, the provider is not).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Protocol, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProducedBundle:
    """What a producer run wrote.

    Attributes:
        root: The bundle root the files were written under (== ``output_dir``).
        prefix: The namespace directory inside ``root`` (``<root>/<prefix>/``).
        files: Number of markdown files written (tables + overview).
        producer: Registry name of the provider (``"sqlite"``).
        source: Display name of the source (e.g. the database filename).
    """

    root: Path
    prefix: str
    files: int
    producer: str
    source: str


class SourceProducer(Protocol):
    """One-method protocol every bundle producer must satisfy."""

    name: ClassVar[str]
    output_prefix: ClassVar[str]
    description: ClassVar[str]

    def produce(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        prefix: str | None = None,
        overwrite: bool = False,
    ) -> ProducedBundle:
        """Emit a bundle for ``source`` under ``output_dir/<prefix>/``.

        Must create missing directories, refuse to clobber existing files
        unless ``overwrite`` is true, and return what was written. A source
        the provider cannot read is a clear error (``FileNotFoundError`` /
        ``ValueError``) — never an empty bundle.
        """
        ...  # pragma: no cover - protocol stub


def _safe_filename(name: str) -> str:
    """Map an arbitrary source name to a portable ``.md`` stem (no ext)."""
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name).strip("._")
    return safe or "table"


@dataclass(frozen=True)
class _Column:
    name: str
    type: str
    nullable: bool
    default: str
    pk: bool


@dataclass(frozen=True)
class _ForeignKey:
    from_col: str
    ref_table: str
    ref_col: str


@dataclass(frozen=True)
class _Table:
    name: str
    stem: str  # sanitized filename stem (collision-free within the run)
    columns: Tuple[_Column, ...]
    fks: Tuple[_ForeignKey, ...]
    row_count: int


class SQLiteProducer:
    """Emit one concept per SQLite table (stdlib ``sqlite3`` — no new deps).

    Mapping (google-okf's ``MySQLProducer`` with zero new dependencies):

    - ``PRAGMA table_info`` → columns (schema table in the body, counts and
      primary key in frontmatter extras).
    - ``PRAGMA foreign_key_list`` → outgoing ``[t](<prefix>/tables/t.md)``
      links (holder → referenced) plus a ``Relationships`` section naming
      both sides, so FTS/PPR can answer "which tables touch X?".
    - ``SELECT COUNT(*)`` → factual ``row_count`` extra (transcription, not
      observation — see deferred §5 of the bundle-hardening plan).
    - ``<prefix>/overview.md`` links every table (deterministic order).

    Links are bundle-root-relative (``<prefix>/tables/<stem>.md``): import
    resolves path links against root-relative concept ids, so the produced
    directory must be imported with the bundle root above ``<prefix>/`` —
    which is exactly what ``okf produce --out <bundle-root>`` sets up, and
    what the lint pre-flight checks.

    Views are skipped (schema-only mapping); attached databases are out of
    scope (open the file you mean).
    """

    name: ClassVar[str] = "sqlite"
    output_prefix: ClassVar[str] = "database"
    description: ClassVar[str] = "SQLite database file → one concept per table"

    def produce(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        prefix: str | None = None,
        overwrite: bool = False,
    ) -> ProducedBundle:
        src = Path(source)
        if not src.is_file():
            raise FileNotFoundError(f"SQLite source not found: {src}")
        prefix = prefix or self.output_prefix
        if "/" in prefix or "\\" in prefix or prefix in (".", ".."):
            raise ValueError(f"prefix must be a single directory name, got {prefix!r}")

        tables = self._read_schema(src)
        if not tables:
            raise ValueError(f"no user tables found in SQLite database: {src}")

        dest = Path(output_dir) / prefix / "tables"
        dest.mkdir(parents=True, exist_ok=True)
        paths = [dest / f"{t.stem}.md" for t in tables]
        paths.append(Path(output_dir) / prefix / "overview.md")
        if not overwrite:
            existing = [str(p) for p in paths if p.exists()]
            if existing:
                raise FileExistsError(
                    "refusing to clobber existing producer output "
                    f"(pass overwrite=True): {', '.join(existing)}"
                )

        referenced_by: Dict[str, List[_ForeignKey]] = {t.name: [] for t in tables}
        for t in tables:
            for fk in t.fks:
                if fk.ref_table in referenced_by:
                    referenced_by[fk.ref_table].append(
                        _ForeignKey(t.name, fk.from_col, fk.ref_col)
                    )

        by_name = {t.name: t for t in tables}
        for t in tables:
            incoming = sorted(
                referenced_by[t.name], key=lambda k: (k.from_col, k.ref_table)
            )
            (dest / f"{t.stem}.md").write_text(
                self._table_doc(src.name, prefix, t, incoming, by_name),
                encoding="utf-8",
            )
        (Path(output_dir) / prefix / "overview.md").write_text(
            self._overview_doc(src.name, prefix, tables), encoding="utf-8"
        )
        logger.info("produced %d concepts from %s under %s/", len(paths), src, prefix)
        return ProducedBundle(
            root=Path(output_dir), prefix=prefix, files=len(paths),
            producer=self.name, source=src.name,
        )

    # -- introspection ----------------------------------------------------

    def _read_schema(self, src: Path) -> List[_Table]:
        try:
            con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise ValueError(f"not a SQLite database: {src} ({exc})") from exc
        try:
            try:
                names = [
                    r[0]
                    for r in con.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                        "ORDER BY name"
                    ).fetchall()
                ]
            except sqlite3.DatabaseError as exc:
                raise ValueError(f"not a SQLite database: {src} ({exc})") from exc
            stems: Dict[str, str] = {}
            tables: List[_Table] = []
            for name in names:
                stem = _safe_filename(name)
                n = 2
                while stem in stems.values():
                    stem = f"{_safe_filename(name)}_{n}"
                    n += 1
                stems[name] = stem
                cols = tuple(
                    _Column(
                        name=str(r[1]),
                        type=str(r[2] or ""),
                        nullable=not bool(r[3]),
                        default="" if r[4] is None else str(r[4]),
                        pk=bool(r[5]),
                    )
                    for r in con.execute(f'PRAGMA table_info("{name}")').fetchall()
                )
                fks = tuple(
                    sorted(
                        (
                            _ForeignKey(
                                from_col=str(r[3]), ref_table=str(r[2]),
                                ref_col=str(r[4]),
                            )
                            for r in con.execute(
                                f'PRAGMA foreign_key_list("{name}")'
                            ).fetchall()
                        ),
                        key=lambda k: (k.from_col, k.ref_table, k.ref_col),
                    )
                )
                try:
                    row_count = int(
                        con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                    )
                except sqlite3.Error as exc:
                    raise ValueError(
                        f"cannot read table {name!r} in {src} ({exc})"
                    ) from exc
                tables.append(_Table(name=name, stem=stems[name], columns=cols,
                                     fks=fks, row_count=row_count))
            return tables
        finally:
            con.close()

    # -- rendering --------------------------------------------------------

    def _table_doc(
        self,
        db_name: str,
        prefix: str,
        table: _Table,
        incoming: List[_ForeignKey],
        by_name: Dict[str, _Table],
    ) -> str:
        pk_cols = [c.name for c in table.columns if c.pk]
        fm = {
            "title": f"Table {table.name}",
            "type": "table",
            "database": db_name,
            "table": table.name,
            "column_count": len(table.columns),
            "primary_key": pk_cols[0] if len(pk_cols) == 1 else "",
            "composite_primary_key": pk_cols if len(pk_cols) > 1 else [],
            "row_count": table.row_count,
        }
        lines = ["---", yaml.safe_dump(fm, sort_keys=False).rstrip(), "---", "",
                 f"# Table {table.name}", "", "## Schema", "",
                 "| column | type | nullable | default | key |",
                 "| --- | --- | --- | --- | --- |"]
        for c in table.columns:
            default = f"`{c.default}`" if c.default else "—"
            lines.append(
                f"| `{c.name}` | {c.type or '—'} "
                f"| {'yes' if c.nullable else 'no'} | {default} "
                f"| {'PK' if c.pk else ''} |"
            )
        lines += ["", "## Relationships", ""]
        if not table.fks and not incoming:
            lines.append("None.")
        else:
            for fk in table.fks:
                target = by_name.get(fk.ref_table)
                if target is None:
                    lines.append(
                        f"- `{table.name}.{fk.from_col}` → "
                        f"`{fk.ref_table}.{fk.ref_col}` (external table, no concept)"
                    )
                else:
                    lines.append(
                        f"- [{fk.ref_table}]({prefix}/tables/{target.stem}.md) — "
                        f"`{table.name}.{fk.from_col}` → "
                        f"`{fk.ref_table}.{fk.ref_col}`"
                    )
            for inc in incoming:
                if inc.ref_table == table.name:
                    continue  # self-FK: already linked under References above
                lines.append(
                    f"- Referenced by `{inc.ref_table}.{inc.from_col}`"
                )
        return "\n".join(lines) + "\n"

    def _overview_doc(
        self, db_name: str, prefix: str, tables: List[_Table]
    ) -> str:
        fm = {
            "title": f"Database {db_name}",
            "type": "database",
            "database": db_name,
            "table_count": len(tables),
        }
        lines = ["---", yaml.safe_dump(fm, sort_keys=False).rstrip(), "---", "",
                 f"# Database {db_name}", "", "## Tables", ""]
        for t in tables:
            lines.append(
                f"- [{t.name}]({prefix}/tables/{t.stem}.md) — "
                f"{len(t.columns)} columns, {t.row_count} rows"
            )
        return "\n".join(lines) + "\n"


PRODUCERS: Dict[str, type] = {
    SQLiteProducer.name: SQLiteProducer,
}


def producer_for(name: str) -> SourceProducer:
    """Instantiate the registered producer ``name`` (``ValueError`` if unknown)."""
    try:
        return PRODUCERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown producer {name!r} (available: {sorted(PRODUCERS)})"
        ) from None


__all__ = [
    "ProducedBundle",
    "SourceProducer",
    "SQLiteProducer",
    "PRODUCERS",
    "producer_for",
]
