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

Observation notes (bundle-hardening §5) live here too: producers record
computed, deterministic judgments (null rates, orphan FKs, type variance)
in a ``## Observations`` section — transcription plus signal, so FTS/PPR
can answer "which tables look unhealthy?". Every produce run also
appends one line to ``<prefix>/log.md`` (the bundle changelog; a reserved
name, never imported).
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

    Observations (fixed check-suite, all deterministic, no timestamps):

    - empty table (0 rows),
    - per-column NULL rate ≥ ``NULL_RATE_WARN`` (one pass per table),
    - per-column duplicate rate ≥ ``DUP_RATE_WARN`` (over present values),
    - orphan foreign keys (holder values with no match; NULL holders are
      not orphans),
    - storage-type variance (``typeof()`` yielding ≥2 real types — SQLite
      dynamic typing; ``'null'`` alone never counts).

    Tables above ``observation_row_cap`` rows get a single skip line
    instead (offline producer, bounded cost). Tables with no findings
    have no ``## Observations`` section at all — no "all good!" noise.
    """

    NULL_RATE_WARN: ClassVar[float] = 0.5
    DUP_RATE_WARN: ClassVar[float] = 0.1

    def __init__(self, *, observation_row_cap: int = 100_000):
        if observation_row_cap < 0:
            raise ValueError(
                "observation_row_cap must be >= 0, "
                f"got {observation_row_cap}"
            )
        self.observation_row_cap = observation_row_cap

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

        # Observations are computed before rendering: the overview marks
        # tables that have any, and the changelog totals them.
        observations: Dict[str, List[str]] = {}
        con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            for t in tables:
                observations[t.name] = self._observe(con, t)
        finally:
            con.close()

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
                self._table_doc(
                    src.name, prefix, t, incoming, by_name,
                    observations[t.name],
                ),
                encoding="utf-8",
            )
        (Path(output_dir) / prefix / "overview.md").write_text(
            self._overview_doc(src.name, prefix, tables, observations),
            encoding="utf-8",
        )
        self._append_log(
            Path(output_dir) / prefix, src.name, tables, observations
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

    # -- observations ------------------------------------------------------

    @staticmethod
    def _qi(name: str) -> str:
        """Quote a SQLite identifier (table/column names come from the DB)."""
        return '"' + name.replace('"', '""') + '"'

    def _observe(self, con: "sqlite3.Connection", table: _Table) -> List[str]:
        """Run the fixed check-suite over one table (deterministic order)."""
        if table.row_count == 0:
            return [f"Table `{table.name}` is empty (0 rows)."]
        if table.row_count > self.observation_row_cap:
            return [
                f"Observations skipped for `{table.name}`: "
                f"{table.row_count} rows > {self.observation_row_cap} cap."
            ]
        t = self._qi(table.name)
        findings: List[str] = []
        # One pass for every column: present count + distinct count.
        cols = ", ".join(
            f"COUNT({self._qi(c.name)}), "
            f"COUNT(DISTINCT {self._qi(c.name)})" for c in table.columns
        )
        counts = con.execute(f"SELECT {cols} FROM {t}").fetchone()
        for c, (present, distinct) in zip(
            table.columns, zip(counts[0::2], counts[1::2])
        ):
            null_rate = 1.0 - present / table.row_count
            if null_rate >= self.NULL_RATE_WARN:
                findings.append(
                    f"`{table.name}.{c.name}` is NULL in "
                    f"{null_rate:.0%} of rows "
                    f"({table.row_count - present}/{table.row_count}) — "
                    "column may be deprecated or unpopulated."
                )
            # Holder-side FK columns duplicate by design (many children per
            # parent) — flagging them for "missing uniqueness" would be
            # noise. Integrity there is the orphan check's job below.
            holder_cols = {fk.from_col for fk in table.fks}
            if present > 0 and c.name not in holder_cols:
                dup_rate = 1.0 - distinct / present
                if dup_rate >= self.DUP_RATE_WARN:
                    findings.append(
                        f"`{table.name}.{c.name}` has "
                        f"{dup_rate:.0%} duplicate values "
                        f"({distinct} distinct / {present} present) — "
                        "uniqueness constraint missing?"
                    )
            types = sorted(
                r[0] for r in con.execute(
                    f"SELECT DISTINCT typeof({self._qi(c.name)}) FROM {t}"
                ).fetchall()
            )
            real = [x for x in types if x != "null"]
            if len(real) >= 2:
                findings.append(
                    f"`{table.name}.{c.name}` stores multiple types: "
                    f"{', '.join(real)} — schema variance, check producers."
                )
        for fk in table.fks:
            orphans = con.execute(
                f"SELECT COUNT(*) FROM {t} "
                f"WHERE {self._qi(fk.from_col)} IS NOT NULL "
                f"AND NOT EXISTS (SELECT 1 FROM {self._qi(fk.ref_table)} "
                f"WHERE {self._qi(fk.ref_table)}."
                f"{self._qi(fk.ref_col)} = {t}.{self._qi(fk.from_col)})"
            ).fetchone()[0]
            if orphans:
                findings.append(
                    f"`{table.name}.{fk.from_col}` has {orphans} value(s) "
                    f"with no match in `{fk.ref_table}.{fk.ref_col}` — "
                    "orphaned references."
                )
        return findings

    @staticmethod
    def _append_log(
        prefix_dir: Path, db_name: str,
        tables: List[_Table], observations: Dict[str, List[str]],
    ) -> None:
        """Append one content-addressed line to ``<prefix>/log.md``.

        No timestamps (determinism invariant) — the entry names the
        producer, source, file set, and finding count. A consecutive
        duplicate (idempotent rerun) is skipped so the log doesn't grow.
        """
        total_obs = sum(len(v) for v in observations.values())
        names = ", ".join(sorted(t.name for t in tables))
        obs_word = "observation" + ("s" if total_obs != 1 else "")
        entry = (
            f"produce sqlite {db_name} → {len(tables) + 1} files "
            f"({len(tables)} tables [{names}] + overview, "
            f"{total_obs} {obs_word})\n"
        )
        log = prefix_dir / "log.md"
        if log.is_file():
            tail = log.read_text(encoding="utf-8").splitlines(keepends=True)
            if tail and tail[-1] == entry:
                return
            with log.open("a", encoding="utf-8") as fh:
                fh.write(entry)
        else:
            log.write_text("# Producer log\n\n" + entry, encoding="utf-8")

    # -- rendering --------------------------------------------------------

    def _table_doc(
        self,
        db_name: str,
        prefix: str,
        table: _Table,
        incoming: List[_ForeignKey],
        by_name: Dict[str, _Table],
        observations: List[str],
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
        if observations:
            lines += ["", "## Observations", ""]
            lines += [f"- {finding}" for finding in observations]
        return "\n".join(lines) + "\n"

    def _overview_doc(
        self, db_name: str, prefix: str, tables: List[_Table],
        observations: Dict[str, List[str]],
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
            n_obs = len(observations.get(t.name, []))
            marker = f", {n_obs} observation" + ("s" if n_obs != 1 else "") if n_obs else ""
            lines.append(
                f"- [{t.name}]({prefix}/tables/{t.stem}.md) — "
                f"{len(t.columns)} columns, {t.row_count} rows{marker}"
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
