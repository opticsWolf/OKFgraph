# Intentionally invalid Office packages

`broken.pptx` is a minimal PPTX whose package-level `officeDocument`
relationship points at `xl/workbook.xml` instead of `ppt/presentation.xml`.
Both Bobine 0.9.1 runners report `CONVERSION_FAILED` with
`fields: {path, converter: "bobine"}` for it.

`parity/tools/check_fixtures.py` validates `parity/fixtures/` only, so this
deliberately malformed package lives here instead. It is exercised by the
last op of `parity/scenarios/ingest-office-formats.json`; well-formed PPTX
conversion is covered by the preceding ops and by `parity/fixtures/sample.pptx`.
