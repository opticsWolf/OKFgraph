"""okfgraph.settings — the one settings table.

Covers: defaults, TOML discovery order, presence-based layer precedence
(CLI > env > TOML > defaults, per key), validation (errors refuse,
off-ladder dims warn), the retired-spelling hints, [[roots]] handling and
the --bundle-root collision guard.
"""

import argparse
import tomllib

import pytest

from okfgraph.settings import (
    DEFAULT_MODEL_ID,
    Settings,
    cli_signal,
    set_cli_flags,
)

NANO = "jinaai/jina-embeddings-v5-text-nano-retrieval"


# ---- defaults --------------------------------------------------------------
def test_table_defaults():
    s = Settings()
    assert s.db_path == "okfgraph.db"
    assert s.bundle_root is None
    assert s.embedding_dim == 512
    assert s.model_id == DEFAULT_MODEL_ID
    assert s.device == "auto"
    assert s.precision == "auto"
    assert s.image_model_id is None
    assert s.image_precision == "auto"
    assert s.cpu_arena is False
    assert s.cache_dir is None
    assert s.max_length is None
    assert s.enable_chunking is True
    assert s.chunk_size == 512
    assert s.chunk_overlap == 40
    assert s.wal_mode is False
    assert s.allow_remote_images is False
    assert s.allowed_image_domains == []
    assert s.mode == "text"
    assert s.batch_size == 32
    assert s.roots == {}


def test_default_model_id_frozen():
    assert DEFAULT_MODEL_ID == "jinaai/jina-embeddings-v5-text-small-retrieval"


# ---- TOML discovery --------------------------------------------------------
def test_toml_from_bundle_root(tmp_path):
    (tmp_path / "okfgraph.toml").write_text(
        '[embedding]\nmax_length = 16384\n', encoding="utf-8")
    s = Settings.load(bundle_root=str(tmp_path))
    assert s.max_length == 16384


def test_toml_cwd_wins_over_bundle_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "okfgraph.toml").write_text(
        '[embedding]\nembedding_dim = 512\n', encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "okfgraph.toml").write_text(
        '[embedding]\nembedding_dim = 768\n', encoding="utf-8")
    assert Settings.load(bundle_root=str(other)).embedding_dim == 512


def test_invalid_toml_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "okfgraph.toml").write_text("broken =", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid TOML"):
        Settings.load()


# ---- layer precedence (presence-based, X5) ---------------------------------
def test_env_beats_toml_even_to_the_default(tmp_path):
    (tmp_path / "okfgraph.toml").write_text(
        '[embedding]\nembedding_dim = 1024\n', encoding="utf-8")
    s = Settings.load(
        bundle_root=str(tmp_path),
        environ={"OKFGRAPH_EMBEDDING_DIM": "512"})
    assert s.embedding_dim == 512


def test_cli_beats_env(tmp_path):
    s = Settings.load(
        environ={"OKFGRAPH_EMBEDDING_DIM": "768"},
        cli_args={"embedding_dim": 256})
    assert s.embedding_dim == 256


def test_zero_and_false_are_present_values(tmp_path):
    # --chunk-overlap 0 / an env false must not be swallowed as "missing".
    (tmp_path / "okfgraph.toml").write_text(
        '[import]\nchunk_overlap = 40\nenable_chunking = true\n',
        encoding="utf-8")
    s = Settings.load(bundle_root=str(tmp_path), cli_args={"chunk_overlap": 0})
    assert s.chunk_overlap == 0
    s = Settings.load(
        bundle_root=str(tmp_path),
        environ={"OKFGRAPH_ENABLE_CHUNKING": "0"})
    assert s.enable_chunking is False


def test_csv_and_bool_env_parsing(monkeypatch):
    s = Settings.load(environ={
        "OKFGRAPH_ALLOWED_IMAGE_DOMAINS": " a.com , b.org , ",
        "OKFGRAPH_PRECISION": "fp16",
    })
    assert s.allowed_image_domains == ["a.com", "b.org"]
    assert s.precision == "fp16"


def test_garbage_env_value_is_refused_with_the_key_named(monkeypatch):
    with pytest.raises(ValueError, match="OKFGRAPH_EMBEDDING_DIM"):
        Settings.load(environ={"OKFGRAPH_EMBEDDING_DIM": "big"})


# ---- validation ------------------------------------------------------------
def test_valid_settings_pass():
    assert Settings(device="cpu").checks() == ([], [])


def test_dim_range_is_an_error():
    errors, _ = Settings(embedding_dim=16).checks()
    assert any("between 32 and 1024" in e for e in errors)
    errors, _ = Settings(embedding_dim=2048).checks()
    assert any("between 32 and 1024" in e for e in errors)


def test_off_ladder_dim_warns_not_errors():
    errors, warnings = Settings(embedding_dim=400).checks()
    assert errors == []
    assert any("Matryoshka" in w for w in warnings)
    errors, warnings = Settings(embedding_dim=768).checks()
    assert not any("Matryoshka" in w for w in warnings)


def test_dim_error_refuses_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "okfgraph.toml").write_text(
        '[embedding]\nembedding_dim = 16\n', encoding="utf-8")
    with pytest.raises(ValueError, match="between 32 and 1024"):
        Settings.load()


def test_empty_db_path_rejected():
    errors, _ = Settings(db_path="").checks()
    assert any("non-empty" in e for e in errors)


def test_model_id_checks():
    for bad in ["", "no-slash", "a\"/b", "a'/b", "a;/b", "a\\/b"]:
        errors, _ = Settings(model_id=bad).checks()
        assert any("embedding.model_id" in e for e in errors), bad
    assert Settings(model_id=NANO).checks() == ([], [])
    assert Settings(model_id="someone/custom-embed").checks() == ([], [])


def test_device_choices():
    errors, _ = Settings(device="gpu").checks()
    assert any("embedding.device must be one of" in e for e in errors)


def test_precision_checks():
    errors, _ = Settings(precision="int4").checks()
    assert any("embedding.precision" in e for e in errors)
    assert Settings(precision="int8").checks() == ([], [])


def test_image_precision_checks():
    errors, _ = Settings(image_precision="int8").checks()
    assert any("embedding.image_precision" in e for e in errors)


def test_max_length_range():
    errors, _ = Settings(max_length=0).checks()
    assert any("max_length" in e for e in errors)
    errors, _ = Settings(max_length=32769).checks()
    assert any("max_length" in e for e in errors)
    assert Settings(max_length=32768).checks() == ([], [])


def test_cache_dir_must_be_absolute(tmp_path):
    errors, _ = Settings(cache_dir="./models").checks()
    assert any("absolute path" in e for e in errors)
    assert Settings(cache_dir=str(tmp_path)).checks() == ([], [])


def test_import_field_checks():
    errors, _ = Settings(mode="fast").checks()
    assert any("import.mode must be one of" in e for e in errors)
    errors, _ = Settings(batch_size=0).checks()
    assert any("batch_size" in e for e in errors)
    errors, _ = Settings(batch_size=300).checks()
    assert any("batch_size" in e for e in errors)
    errors, _ = Settings(chunk_size=32).checks()
    assert any("chunk_size" in e for e in errors)
    errors, _ = Settings(chunk_size=99999).checks()
    assert any("chunk_size" in e for e in errors)
    errors, _ = Settings(chunk_overlap=-1).checks()
    assert any("chunk_overlap" in e for e in errors)
    errors, _ = Settings(chunk_overlap=600).checks()
    assert any("chunk_overlap must be >= 0 and < chunk_size" in e for e in errors)
    errors, _ = Settings(allowed_image_domains=[" "]).checks()
    assert any("empty entries" in e for e in errors)


def test_multiple_errors_listed_together():
    with pytest.raises(ValueError) as exc:
        Settings.load(cli_args={"embedding_dim": 16, "device": "gpu"})
    text = str(exc.value)
    assert "32 and 1024" in text
    assert "must be one of" in text


# ---- retired spellings: clear errors, not silent acceptance ----------------
def test_legacy_env_names_are_refused(monkeypatch):
    for env_key in ("OKFGRAPH_DB", "OKFGRAPH_DIM", "OKFGRAPH_MODEL",
                    "OKFGRAPH_BUNDLE", "OKFGRAPH_IMAGE_MODEL",
                    "OKFGRAPH_NO_CHUNKING"):
        with pytest.raises(ValueError, match=env_key):
            Settings.load(environ={env_key: "anything"})


def test_legacy_toml_keys_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for text, needle in [
        ('bundle = "kb"', "bundle_root"),
        ('[database]\npath = "kb.db"', "db_path"),
        ('[database]\ndim = 512', "embedding_dim"),
        ('[import]\nno_chunking = true', "enable_chunking"),
        ('[embedding]\nomni_model_id = "x"', "omni_model_id"),
    ]:
        (tmp_path / "okfgraph.toml").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=needle):
            Settings.load()


# ---- the --bundle-root collision guard (D9) --------------------------------
def test_bundle_root_with_equals_is_refused():
    # Old scripts passed ALIAS=PATH here; named roots now ride --root.
    with pytest.raises(ValueError, match="--root"):
        Settings.load(cli_args={"bundle_root": "aa=/definitely-not-here"})


def test_bundle_root_with_equals_but_real_path_is_allowed(tmp_path):
    odd = tmp_path / "weird=name"
    odd.mkdir()
    s = Settings.load(cli_args={"bundle_root": str(odd)})
    assert s.bundle_root == str(odd)


# ---- roots -----------------------------------------------------------------
def test_roots_cli_list_parses():
    s = Settings.load(cli_args={"roots": ["bb=/x", "cc=/y"]})
    assert s.roots == {"bb": "/x", "cc": "/y"}


def test_roots_cli_list_replaces_toml(tmp_path):
    (tmp_path / "okfgraph.toml").write_text(
        '[[roots]]\nalias = "aa"\npath = "docs"\n', encoding="utf-8")
    assert Settings.load(bundle_root=str(tmp_path)).roots == {
        "aa": str(tmp_path / "docs")}
    s = Settings.load(
        bundle_root=str(tmp_path), cli_args={"roots": ["bb=/x"]})
    assert s.roots == {"bb": "/x"}


def test_roots_bad_item_is_a_usage_error():
    with pytest.raises(ValueError, match="ALIAS=PATH"):
        Settings.load(cli_args={"roots": ["broken"]})


# ---- flag generation + cli_signal ------------------------------------------
def _parse(argv):
    parser = argparse.ArgumentParser()
    set_cli_flags(parser)
    return parser.parse_args(argv)


def test_cli_signal_carries_only_set_flags():
    sig = cli_signal(_parse([]))
    assert sig == {}
    sig = cli_signal(_parse(["--no-chunking", "--embedding-dim", "256",
                             "--root", "aa=/tmp/x", "--wal-mode"]))
    assert sig == {
        "enable_chunking": False,   # inverted: --no-chunking present
        "wal_mode": True,
        "embedding_dim": 256,
        "roots": ["aa=/tmp/x"],
    }


def test_flag_defaults_are_suppressed():
    # An unset global flag is absent from the namespace (rule: presence,
    # not truthiness).
    args = _parse([])
    assert getattr(args, "max_length", None) is None
    with pytest.raises(AttributeError):
        args.embedding_dim  # noqa: B018 — absent by design


def test_full_toml_round_trip(tmp_path):
    """Round-trip: every TOML key parses and lands on its canonical field."""
    text = "\n".join([
        'bundle_root = "kb"',
        "[database]",
        'db_path = "kb.db"',
        "wal_mode = true",
        "[embedding]",
        "embedding_dim = 512",
        f'model_id = "{NANO}"',
        'device = "cpu"',
        'precision = "fp32"',
        "max_length = 8192",
        "[import]",
        'mode = "text"',
        "batch_size = 64",
        "chunk_size = 512",
        "chunk_overlap = 40",
        "enable_chunking = true",
    ])
    (tmp_path / "okfgraph.toml").write_text(text, encoding="utf-8")
    s = Settings.load(bundle_root=str(tmp_path))
    assert s.bundle_root == "kb"
    assert s.db_path == "kb.db"
    assert s.wal_mode is True
    assert s.embedding_dim == 512
    assert s.model_id == NANO
    assert s.device == "cpu"
    assert s.precision == "fp32"
    assert s.max_length == 8192
    assert s.mode == "text"
    assert s.batch_size == 64
    assert s.enable_chunking is True
