"""0.2.17 — length-bucketed batch encoding.

No encoder needed: the helper takes an encode function, so a fake
proves order restoration, bucket caps, and monotonic bucket maxima.
"""

from okfgraph.components.import_ import _length_bucketed_encode


def _recording_fake():
    calls = []

    def encode(batch):
        calls.append((len(batch), max(len(t) for t in batch)))
        return [f"emb:{t}" for t in batch]

    encode.calls = calls
    return encode


def test_order_restored_and_batches_bounded():
    fake = _recording_fake()
    texts = ["x" * 9000, "a", "mm", "y" * 5000, "b", "z" * 100]
    out = _length_bucketed_encode(texts, 2, fake)
    assert out == [f"emb:{t}" for t in texts]
    # Shortest-first: bucket maxima non-decreasing, sizes capped.
    maxima = [m for _, m in fake.calls]
    assert maxima == sorted(maxima)
    assert all(n <= 2 for n, _ in fake.calls)
    assert sum(n for n, _ in fake.calls) == len(texts)


def test_progress_reports_buckets():
    seen = []
    texts = ["a", "bb", "ccc"]
    out = _length_bucketed_encode(
        texts, 2, lambda b: list(b),
        progress=lambda d, t, l: seen.append((d, t, l)),
    )
    assert out == texts
    assert seen == [(1, 2, 2), (2, 2, 3)]


def test_empty_and_single():
    assert _length_bucketed_encode([], 8, lambda b: []) == []
    assert _length_bucketed_encode(["solo"], 8, lambda b: ["E"]) == ["E"]
