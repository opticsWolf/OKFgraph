# OKFgraph 0.16.1 release notes

## Summary

0.16.1 is a patch release on top of the 0.16.0 Rust-core cutover. There is **no
public API, router op, CLI verb, MCP tool, setting or schema change**. The
release refreshes the two native Markdown/document dependencies and keeps their
previous behavior where the new versions offer opt-in extras.

The 0.16.0 cutover notes remain in
[`release-notes-0.16.0.md`](release-notes-0.16.0.md).

## Changed

- **bobine 0.9.1 → 0.10.0**: tagged-PDF math and provenance, explicit 256 MiB
  input/body limits, binary text fallback detection, and corrected PPTX slide
  image ordering. These affect conversion output only; the converter API and
  the `CONVERSION_FAILED` contract are unchanged.
- **mordant 0.10.1 → 0.11.0**: new runtime engine switches, structured render
  options, definition-list/fenced-div/link-attribute extensions and KaTeX
  0.18.5. 0.16.1 keeps the old engine set by building with
  `default-features = false` plus the explicit `std`, `html-entities`,
  `chunker`, `linter`, `meta`, `math`, `footnotes` and `emoji` features, so
  Markdown output is unchanged.
- The parity oracle venv now installs `bobine 0.10.0` and `mordant 0.11.0`
  alongside the frozen Python 0.15 source at tag `python-oracle-0.15.0`.

## Compatibility

- v10 databases and bundles remain compatible; no migration is needed from
  0.16.0.
- Public Python imports, `OKFRouter`/`OKFClient` methods, models, errors,
  settings constants, console scripts and module entry shims are unchanged.
- One `okfgraph` wheel still carries the facade, the Rust core and the bundled
  PDF/Office plugin. `cpu`/`gpu` still select ONNX Runtime; `pdf` remains a
  no-op alias.
- Schema, envelope order, error codes, jobs, HTTP and MCP surfaces are
  unchanged.

## Verification

- Targeted differential scenarios: `minimal`, `lint`, `import-minimal`,
  `ingest-md`, `ingest-office-formats`, `ingest-office-real` and
  `ingest-pdf-real` — **7/7 green**.
- Rust tests: **111** all-feature, **86** no-default, **9** fuzz properties.
- `cargo fmt --all -- --check`, all-feature and no-default clippy, and
  `cargo deny check` are clean.
- Rebuilt wheel passes `check_packaging.py`, `check_wheel.py`,
  `check_facade.py` and `check_pdf_wheel.py`.

## Upgrade

```bash
pip install --upgrade "okfgraph[cpu,pdf]"   # or [gpu,pdf]
```

No reindex or re-embed is required when upgrading from 0.16.0.
