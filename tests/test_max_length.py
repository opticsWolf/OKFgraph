"""0.2.21 — configurable token limit (max_length).

Pure-config parts (defaults, validation, TOML/env/CLI merge, bucket
budgeting) run anywhere. Router/session parts need embroider>=0.1.4
(`MODEL_MAX_TOKENS`, the `max_length` open kwarg).
"""

import pytest

from okfgraph.cli import build_parser
from okfgraph.components.import_ import _length_bucketed_encode
from okfgraph.config import OKFConfig


def _recording_fake(counts):
    calls = []

    def encode(batch):
        calls.append([len(t) for t in batch])
        return [f"emb:{t}" for t in batch]

    def count(text):
        return counts[text]

    encode.calls = calls
    return encode, count


class TestConfig:
    def test_default_is_none(self):
        assert OKFConfig().embedding.max_length is None

    def test_validate_range(self):
        c = OKFConfig()
        assert c.validate() == []
        c.embedding.max_length = 0
        with pytest.raises(ValueError, match="max_length"):
            c.validate()
        c.embedding.max_length = 32769
        with pytest.raises(ValueError, match="max_length"):
            c.validate()
        c.embedding.max_length = 32768
        assert c.validate() == []

    def test_toml_and_env_and_cli(self, tmp_path, monkeypatch):
        (tmp_path / "okfgraph.toml").write_text(
            '[embedding]\nmax_length = 16384\n', encoding="utf-8")
        assert OKFConfig.load(bundle_root=tmp_path).embedding.max_length == 16384
        monkeypatch.setenv("OKFGRAPH_MAX_LENGTH", "4096")
        assert OKFConfig.load(bundle_root=tmp_path).embedding.max_length == 4096
        cfg = OKFConfig.load(bundle_root=tmp_path, cli_args={"max_length": 2048})
        assert cfg.embedding.max_length == 2048


class TestCLI:
    def test_max_length_flag(self):
        args = build_parser().parse_args(["import", "--all", "--max-length", "32768"])
        assert args.max_length == 32768

    def test_max_length_default_none(self):
        args = build_parser().parse_args(["import", "--all"])
        assert args.max_length is None


class TestTokenBudget:
    TEXTS = ["a", "bb", "cccc", "dddddddd"]

    def test_order_restored_under_budget(self):
        counts = {"a": 1, "bb": 2, "cccc": 4, "dddddddd": 8}
        fake, count = _recording_fake(counts)
        out = _length_bucketed_encode(
            self.TEXTS, 8, fake, token_budget=100, count_fn=count)
        assert out == [f"emb:{t}" for t in self.TEXTS]

    def test_bins_respect_budget(self):
        counts = {"a": 1, "bb": 2, "cccc": 4, "dddddddd": 8}
        fake, count = _recording_fake(counts)
        _length_bucketed_encode(
            self.TEXTS, 8, fake, token_budget=9, count_fn=count)
        # Sorted asc (1,2,4,8), greedy ≤9: [1+2+4]=7, [8] solo.
        assert [sum(b) for b in fake.calls] == [7, 8]

    def test_count_cap_still_applies(self):
        counts = {t: 1 for t in self.TEXTS}
        fake, count = _recording_fake(counts)
        _length_bucketed_encode(
            self.TEXTS, 2, fake, token_budget=100, count_fn=count)
        assert [len(b) for b in fake.calls] == [2, 2]

    def test_no_budget_no_count_uses_legacy_path(self):
        seen = []

        def encode(batch):
            seen.append(len(batch))
            return list(batch)

        out = _length_bucketed_encode(["x" * 10, "y"], 8, encode)
        assert out == ["x" * 10, "y"]
        assert seen == [2]


class TestRouter:
    def test_default_limit_is_8192(self, tmp_path):
        from okfgraph.router import OKFRouter

        r = OKFRouter(db_path=str(tmp_path / "m.db"), bundle_root=str(tmp_path),
                      embedding_dim=512, enable_chunking=False, device="cpu")
        try:
            import embroider

            assert r.max_length == embroider.MAX_LENGTH == 8192
            assert r.import_mgr.context_window == 8192
        finally:
            r.close()

    def test_explicit_limit_validated_and_stored(self, tmp_path):
        import embroider

        from okfgraph.router import OKFRouter

        for bad in (0, -5, embroider.MODEL_MAX_TOKENS + 1):
            with pytest.raises(ValueError, match="max_length"):
                OKFRouter(db_path=str(tmp_path / "m.db"),
                          bundle_root=str(tmp_path), max_length=bad,
                          embedding_dim=512, enable_chunking=False, device="cpu")
        r = OKFRouter(db_path=str(tmp_path / "m.db"), bundle_root=str(tmp_path),
                      max_length=1024, embedding_dim=512,
                      enable_chunking=False, device="cpu")
        try:
            assert r.max_length == 1024
            assert r.import_mgr.context_window == 1024
        finally:
            r.close()

    def test_short_text_identical_across_limits(self):
        import embroider
        from okfgraph.components.embedding import resolve_ort_dylib

        resolve_ort_dylib()
        kwargs = {"truncate_dim": 512, "device": "cpu"}
        a = embroider.JinaV5.open(
            "jinaai/jina-embeddings-v5-text-small-retrieval", **kwargs,
            max_length=8192)
        b = embroider.JinaV5.open(
            "jinaai/jina-embeddings-v5-text-small-retrieval", **kwargs,
            max_length=32768)
        assert a.max_length == 8192
        assert b.max_length == 32768
        va = a.encode("tiny doc", task="Document")
        vb = b.encode("tiny doc", task="Document")
        assert va == vb
