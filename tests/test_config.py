"""Tests for okfgraph.config — schema validation."""

from okfgraph.config import DatabaseConfig, EmbeddingConfig, ImportConfig, OKFConfig


class TestDatabaseConfigValidation:
    """Test database config validation."""

    def test_valid_config(self):
        config = DatabaseConfig(path="test.db", dim=512, wal_mode=True)
        errors = config.validate()
        assert errors == []

    def test_empty_path(self):
        config = DatabaseConfig(path="", dim=512)
        errors = config.validate()
        assert any("non-empty" in e for e in errors)

    def test_dim_too_small(self):
        config = DatabaseConfig(dim=16)
        errors = config.validate()
        assert any("between 32 and 1024" in e for e in errors)

    def test_dim_too_large(self):
        config = DatabaseConfig(dim=2048)
        errors = config.validate()
        assert any("between 32 and 1024" in e for e in errors)

    def test_dim_not_recommended(self):
        config = DatabaseConfig(dim=400)
        errors = config.validate()
        assert any("Matryoshka" in e for e in errors)

    def test_dim_recommended_values(self):
        for dim in (128, 256, 512, 768, 1024):
            config = DatabaseConfig(dim=dim)
            errors = config.validate()
            assert not any("Matryoshka" in e for e in errors)


class TestEmbeddingConfigValidation:
    """Test embedding config validation."""

    def test_valid_config(self):
        config = EmbeddingConfig(device="cpu")
        errors = config.validate()
        assert errors == []

    def test_valid_cuda_device(self):
        config = EmbeddingConfig(device="cuda")
        errors = config.validate()
        assert errors == []

    def test_invalid_device(self):
        config = EmbeddingConfig(device="gpu")
        errors = config.validate()
        assert any("must be one of" in e for e in errors)

    def test_relative_cache_dir(self):
        config = EmbeddingConfig(cache_dir="./models")
        errors = config.validate()
        assert any("absolute path" in e for e in errors)

    def test_absolute_cache_dir(self, tmp_path):
        config = EmbeddingConfig(cache_dir=str(tmp_path))
        errors = config.validate()
        assert not any("absolute path" in e for e in errors)

    def test_empty_omni_model_id(self):
        config = EmbeddingConfig(omni_model_id="")
        errors = config.validate()
        assert any("non-empty" in e for e in errors)


class TestImportConfigValidation:
    """Test import config validation."""

    def test_valid_config(self):
        config = ImportConfig(mode="text", batch_size=32)
        errors = config.validate()
        assert errors == []

    def test_valid_modes(self):
        for mode in ("text", "optional", "omni"):
            config = ImportConfig(mode=mode)
            errors = config.validate()
            assert not any("must be one of" in e for e in errors)

    def test_invalid_mode(self):
        config = ImportConfig(mode="fast")
        errors = config.validate()
        assert any("must be one of" in e for e in errors)

    def test_batch_size_too_small(self):
        config = ImportConfig(batch_size=0)
        errors = config.validate()
        assert any("between 1 and 256" in e for e in errors)

    def test_batch_size_too_large(self):
        config = ImportConfig(batch_size=512)
        errors = config.validate()
        assert any("between 1 and 256" in e for e in errors)

    def test_chunk_size_too_small(self):
        config = ImportConfig(chunk_size=32)
        errors = config.validate()
        assert any("between 64 and 8192" in e for e in errors)

    def test_chunk_size_too_large(self):
        config = ImportConfig(chunk_size=16384)
        errors = config.validate()
        assert any("between 64 and 8192" in e for e in errors)

    def test_chunk_overlap_negative(self):
        config = ImportConfig(chunk_overlap=-10)
        errors = config.validate()
        assert any(">= 0" in e for e in errors)

    def test_chunk_overlap_exceeds_chunk_size(self):
        config = ImportConfig(chunk_size=512, chunk_overlap=600)
        errors = config.validate()
        assert any("< chunk_size" in e for e in errors)

    def test_empty_allowed_domains(self):
        config = ImportConfig(allowed_image_domains=["", "example.com"])
        errors = config.validate()
        assert any("empty entries" in e for e in errors)


class TestOKFConfigValidation:
    """Test full config validation."""

    def test_valid_full_config(self):
        config = OKFConfig()
        errors = config.validate()
        assert errors == []

    def test_multiple_errors(self):
        config = OKFConfig(
            database=DatabaseConfig(dim=16),
            embedding=EmbeddingConfig(device="gpu"),
            import_config=ImportConfig(mode="fast"),
        )
        try:
            config.validate()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "32 and 1024" in str(e)
            assert "must be one of" in str(e)

    def test_load_with_valid_config(self):
        config = OKFConfig.load()
        # Should not raise
        assert config is not None
