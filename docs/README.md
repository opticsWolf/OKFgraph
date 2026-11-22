# Documentation index

OKFgraph 0.16.1 is Rust-first. This index separates current documentation
from historical design and release records. When a historical document
conflicts with a current one, the current one wins.

## Current — start here

| Document | What it answers |
| --- | --- |
| [`../README.md`](../README.md) | Install, quick start, surfaces, architecture overview |
| [`quickref.md`](quickref.md) | One-page CLI/Python/ops/settings/env/error reference |
| [`rust-architecture.md`](rust-architecture.md) | Runtime model, store, locks, search, ingest, packaging |
| [`python-binding.md`](python-binding.md) | Supported Python public API and failure behavior |
| [`migration-guide.md`](migration-guide.md) | Upgrading from Python 0.15 |
| [`release-notes-0.16.1.md`](release-notes-0.16.1.md) | Patch notes for the dependency refresh |
| [`release-notes-0.16.0.md`](release-notes-0.16.0.md) | What changed in the stable cutover |
| [`rust-compat.md`](rust-compat.md) | Executable compatibility checklist |
| [`release-gate.md`](release-gate.md) | Manual GPU release sign-off procedure |
| [`license-review.md`](license-review.md) | Rust dependency license inventory |
| [`performance-baseline.md`](performance-baseline.md) | Generated paired benchmark numbers |
| [`performance-findings.md`](performance-findings.md) | Performance root causes and gate |
| [`../parity/README.md`](../parity/README.md) | Differential harness, oracle tag, scenario catalog |

## Current — project plans and bug register

| Document | Status |
| --- | --- |
| [`plan-rust-core.md`](plan-rust-core.md) | Milestones M0–M6 and Rust-core contract |
| [`plan-bug-fixes.md`](plan-bug-fixes.md) | BUG-001…089 evidence, resolutions and gate obligations |

## Historical — Python 0.15 implementation records

These documents explain why behaviors exist. They describe the old Python
core, not the shipped implementation:

- [`architecture.md`](architecture.md) — Python 0.15 architecture (v6.8)
- [`compat.md`](compat.md) — Python 0.15 model/precision compatibility
- [`converters.md`](converters.md) — old converter seam
- [`chunking-and-graph-retrieval.md`](chunking-and-graph-retrieval.md)
- [`search-options-analysis.md`](search-options-analysis.md)
- [`surface-unification-plan.md`](surface-unification-plan.md)
- [`harness-integration.md`](harness-integration.md)
- [`reranker-integration-plan.md`](reranker-integration-plan.md)
- [`plan-declared-image-identity.md`](plan-declared-image-identity.md)
- [`plan-embedder-loader.md`](plan-embedder-loader.md)
- [`eg2-int8-gates.md`](eg2-int8-gates.md)
- [`bench-eg2-precision-width.md`](bench-eg2-precision-width.md)
- [`ladybug-index-crash-log.md`](ladybug-index-crash-log.md)
- [`implementation-plan-0.14.md`](implementation-plan-0.14.md)
- [`improvement-planning-0.10.1.md`](improvement-planning-0.10.1.md)
- [`issue-solving-implementation-plan.md`](issue-solving-implementation-plan.md)
- [`release-0.10.0-plan.md`](release-0.10.0-plan.md)
- [`archive/`](archive/README.md) — earlier plans and status notes

The frozen Python source itself is available at tag `python-oracle-0.15.0`.
`parity/tools/oracle.py` checks it out into `parity/tmp/oracle-0.15.0` for
differential testing.

## Website

[`../pages/index.html`](../pages/index.html) is the public landing page. Keep
its release badge and terminology aligned with this index.
