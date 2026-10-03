"""Bundle-hardening §5: observation notes + producer changelog.

A scratch clinic database (built with stdlib sqlite3, no binary fixture)
pins the fixed check-suite: empty tables, NULL rates, duplicate rates
(with the FK-holder exemption), orphan FKs (NULL holders are not orphans),
storage-type variance, the over-cap skip, and the log.md lifecycle
(append, dedupe, reserved-name exclusion from import/lint).
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from okfgraph.components.import_ import RESERVED_FILENAMES, is_concept_file
from okfgraph.components.lint import lint_bundle
from okfgraph.components.producers import SQLiteProducer
from okfgraph.router import OKFRouter


def _make_clinic_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE patients (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT
        );
        INSERT INTO patients VALUES
            (1, 'ann', '111'), (2, 'bob', '222'), (3, 'cid', NULL),
            (4, 'dan', NULL), (5, 'eve', NULL);
        CREATE TABLE visits (
            id INTEGER PRIMARY KEY,
            patient_id INTEGER REFERENCES patients(id),
            code TEXT
        );
        INSERT INTO visits VALUES
            (1, 1, 'A'), (2, 2, 'A'), (3, 999, 'A'), (4, NULL, 'B'),
            (5, 1, 'A'), (6, 2, 'A'), (7, 1, 'A'), (8, 2, 'A'),
            (9, 1, 'A'), (10, 2, 'A');
        CREATE TABLE empty_log (id INTEGER PRIMARY KEY, msg TEXT);
        CREATE TABLE mixed (id INTEGER PRIMARY KEY, val NUMERIC);
        INSERT INTO mixed VALUES (1, 'abc'), (2, 123), (3, 4.5), (4, NULL);
        CREATE TABLE settings (k TEXT PRIMARY KEY, v TEXT NOT NULL);
        INSERT INTO settings VALUES ('a', '1'), ('b', '2');
        """
    )
    con.close()


@pytest.fixture(scope="module")
def clinic():
    d = Path(tempfile.mkdtemp())
    db = d / "clinic.db"
    _make_clinic_db(db)
    out = d / "bundle"
    bundle = SQLiteProducer().produce(db, out)
    yield {"dir": d, "db": db, "out": out, "bundle": bundle}
    shutil.rmtree(d, ignore_errors=True)


def _body(clinic, stem):
    return (clinic["out"] / "database" / "tables" / f"{stem}.md").read_text(
        encoding="utf-8"
    )


class TestFindings:
    def test_empty_table(self, clinic):
        body = _body(clinic, "empty_log")
        assert "## Observations" in body
        assert "`empty_log` is empty (0 rows)" in body

    def test_null_rate(self, clinic):
        body = _body(clinic, "patients")
        assert "`patients.phone` is NULL in 60% of rows (3/5)" in body

    def test_below_threshold_silent(self, clinic):
        # patients.name: 0% null, all distinct → no finding lines at all.
        body = _body(clinic, "patients")
        assert "patients.name" not in body.split("## Observations")[1]

    def test_dup_rate(self, clinic):
        body = _body(clinic, "visits")
        assert "`visits.code` has 80% duplicate values (2 distinct / 10 present)" in body

    def test_fk_holder_dup_exempt(self, clinic):
        # visits.patient_id duplicates 9:3 by design (many visits per
        # patient) — the orphan check covers its integrity, not dup-rate.
        body = _body(clinic, "visits")
        dup_lines = [l for l in body.splitlines() if "duplicate values" in l]
        assert not any("visits.patient_id" in l for l in dup_lines)

    def test_orphan_fk_counts_non_null_misses(self, clinic):
        body = _body(clinic, "visits")
        # patient_id 999 is orphaned; the NULL holder is not.
        assert "`visits.patient_id` has 1 value(s) with no match" in body

    def test_type_variance(self, clinic):
        body = _body(clinic, "mixed")
        assert "`mixed.val` stores multiple types: integer, real, text" in body

    def test_clean_table_has_no_section(self, clinic):
        assert "## Observations" not in _body(clinic, "settings")


class TestOverviewMarkers:
    def test_markers(self, clinic):
        overview = (clinic["out"] / "database" / "overview.md").read_text(
            encoding="utf-8"
        )
        assert "visits.md) — 3 columns, 10 rows, 2 observations" in overview
        assert "settings.md) — 2 columns, 2 rows\n" in overview


class TestRowCap:
    def test_over_cap_skips_with_note(self, clinic, tmp_path):
        out = tmp_path / "capped"
        SQLiteProducer(observation_row_cap=4).produce(clinic["db"], out)
        body = (out / "database" / "tables" / "visits.md").read_text(
            encoding="utf-8"
        )
        assert "## Observations" in body
        assert "Observations skipped for `visits`: 10 rows > 4 cap." in body

    def test_empty_table_reported_regardless_of_cap(self, tmp_path, clinic):
        out = tmp_path / "capped"
        SQLiteProducer(observation_row_cap=0).produce(clinic["db"], out)
        body = (out / "database" / "tables" / "empty_log.md").read_text(
            encoding="utf-8"
        )
        assert "`empty_log` is empty (0 rows)" in body

    def test_bad_cap_rejected(self):
        with pytest.raises(ValueError, match="observation_row_cap"):
            SQLiteProducer(observation_row_cap=-1)


class TestLog:
    def test_log_written(self, clinic):
        log = clinic["out"] / "database" / "log.md"
        assert log.is_file()
        text = log.read_text(encoding="utf-8")
        assert text.startswith("# Producer log\n\n")
        assert (
            "produce sqlite clinic.db → 6 files "
            "(5 tables [empty_log, mixed, patients, settings, visits] "
            "+ overview, 5 observations)" in text
        )

    def test_rerun_does_not_grow_log(self, clinic, tmp_path):
        # Deterministic rerun: identical table bytes AND no log growth
        # (consecutive-duplicate suppression).
        first = {
            p.relative_to(clinic["out"]): p.read_bytes()
            for p in sorted(clinic["out"].rglob("*.md"))
        }
        SQLiteProducer().produce(clinic["db"], clinic["out"], overwrite=True)
        second = {
            p.relative_to(clinic["out"]): p.read_bytes()
            for p in sorted(clinic["out"].rglob("*.md"))
        }
        assert first == second

    def test_second_source_appends(self, clinic, tmp_path):
        # Self-contained output (never mutates the shared fixture): two
        # runs append two lines.
        out = tmp_path / "o"
        db2 = tmp_path / "other.db"
        con = sqlite3.connect(str(db2))
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        con.close()
        SQLiteProducer().produce(clinic["db"], out)
        SQLiteProducer().produce(db2, out, overwrite=True)
        lines = (out / "database" / "log.md").read_text(
            encoding="utf-8"
        ).splitlines()
        assert any(l.startswith("produce sqlite clinic.db") for l in lines)
        assert any(l.startswith("produce sqlite other.db") for l in lines)

    def test_log_is_reserved(self):
        assert "log.md" in RESERVED_FILENAMES
        assert not is_concept_file(Path("x") / "log.md")
        assert not is_concept_file(Path("x") / "LOG.MD")

    def test_log_skipped_on_import_and_lint(self, clinic):
        assert lint_bundle(clinic["out"])["errors"] == []
        assert lint_bundle(clinic["out"])["warnings"] == []
        r = OKFRouter(
            db_path=str(clinic["dir"] / "obs.db"),
            bundle_root=str(clinic["out"]),
            embedding_dim=512,
            device="cpu",
        )
        try:
            ids = r.import_mgr.import_bundle(clinic["out"])
            assert len(ids) == 6  # 5 tables + overview, no log concept
            assert not [i for i in ids if i.endswith("/log")]
            assert r.list_broken_links() == []
        finally:
            r.close()
