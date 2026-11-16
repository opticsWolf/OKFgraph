# Compatibility notes

## Image fusion identity and okfgraph 0.15.0

The W3 fusion pin needs `GemmaVision.text_space` from embroider
**0.15.1**. The dependency floor is therefore
`embroider>=0.15.1,<0.16`; 0.15.0 predates the fusion-space
identity and cannot verify that text and image precision match.

## int8-pinned graphs and okfgraph 0.14.2

Checked **2026-11-16**; full record in
[`eg2-int8-gates.md`](eg2-int8-gates.md).

okfgraph 0.14.2 predates the `int8: 8` precision code. Opening a new
int8-pinned graph with it and requesting `fp32` does **not** silently
adopt the graph: the encoder lands fp32, and the space pin refuses the
mismatch with

```
StateError: graph vectors live in another space than this session's
(pinned ... precision=int8, landed fp32).
Vectors from different spaces must never mix.
```

No version/schema floor is needed for this path. The protection is the
space pin, not the old codec; new graphs always write one, and current
builds fail closed on an unknown stored code.

Do not serve int8 graphs to older clients that predate the space pin.
