#!/usr/bin/env python3
"""Validate OOXML parity fixtures before the differential matrix.

A malformed package can masquerade as a converter refusal: the old
``sample.pptx`` carried the XLSX root relationship and therefore failed with
``missing required part: xl/workbook.xml`` in both runners even though Bobine
0.9.1 converts a well-formed PPTX. This guard keeps each OOXML fixture's root
``officeDocument`` relationship consistent with its extension and checks that
the declared main part is present.

Run standalone with ``python parity/tools/check_fixtures.py``; ``run_matrix.py``
calls the same validator before any scenario executes.
"""
from __future__ import annotations

import posixpath
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]

REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
OFFICE_DOCUMENT = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
    "officeDocument"
)
MAIN_PARTS = {
    "docx": "word/document.xml",
    "xlsx": "xl/workbook.xml",
    "pptx": "ppt/presentation.xml",
}


def normalize_target(source: str, target: str) -> str:
    """Resolve an OPC relationship target to a package-relative zip path."""
    if target.startswith("/"):
        return target.lstrip("/")
    # Package-level relationships in `/_rels/.rels` are relative to the
    # package root, not to the `_rels/` directory holding the part.
    base = "" if source == "_rels/.rels" else posixpath.dirname(source)
    return posixpath.normpath(posixpath.join(base, target)).lstrip("/")


def check_file(path: Path) -> list[str]:
    """Return validation errors for one OOXML fixture (empty when valid)."""
    ext = path.suffix.lower().lstrip(".")
    expected = MAIN_PARTS.get(ext)
    if expected is None:
        return []
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            try:
                rels = ElementTree.fromstring(archive.read("_rels/.rels"))
            except KeyError:
                return [f"{path}: missing _rels/.rels"]
            except ElementTree.ParseError as err:
                return [f"{path}: invalid _rels/.rels: {err}"]
            targets = [
                rel.get("Target")
                for rel in rels.findall(f"{REL_NS}Relationship")
                if rel.get("Type") == OFFICE_DOCUMENT and rel.get("Target")
            ]
            if len(targets) != 1:
                return [
                    f"{path}: expected one officeDocument root relationship, "
                    f"found {len(targets)}"
                ]
            main = normalize_target("_rels/.rels", targets[0])
            if main != expected:
                return [
                    f"{path}: .{ext} main part is '{main}', expected '{expected}'"
                ]
            if expected not in names:
                return [f"{path}: declared main part '{expected}' is absent"]
    except (OSError, zipfile.BadZipFile) as err:
        return [f"{path}: cannot read OOXML package: {err}"]
    return []


def validate_fixtures(fixture_dir: Path | None = None) -> list[str]:
    """Validate every OOXML fixture in ``fixture_dir`` (default: parity/fixtures)."""
    directory = fixture_dir or ROOT / "parity" / "fixtures"
    errors: list[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower().lstrip(".") in MAIN_PARTS:
            errors.extend(check_file(path))
    return errors


def main() -> int:
    errors = validate_fixtures()
    for error in errors:
        print(f"[FAIL] {error}")
    if errors:
        print(f"FIXTURE CHECK: {len(errors)} error(s)")
        return 1
    print("OK: OOXML fixture root relationships match their extensions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
