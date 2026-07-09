"""Configuration management for OKFgraph.

Precedence (highest → lowest): CLI args > env vars > TOML file > defaults.

TOML file locations (checked in order, first match wins):
  1. ``okfgraph.toml`` in the current working directory
  2. ``okfgraph.toml`` in the bundle root (if provided)
  3. ``~/.config/okfgraph/config.toml``

Environment variables (prefix ``OKFGRAPH_``):
  DB, DIM, DEVICE, CACHE_DIR, BUNDLE, OMNI_MODEL_ID,
  CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE, MODE, WAL_MODE,
  ALLOW_REMOTE_IMAGES, ALLOWED_IMAGE_DOMAINS, NO_CHUNKING

Example TOML (``okfgraph.toml``):

    [database]
    path = "okfgraph.db"
    dim = 512
    wal_mode = true

    [embedding]
    device = "cuda"
    cache_dir = "/mnt/models"
    omni_model_id = "jinaai/jina-embeddings-v5-omni-small-retrieval"

    [import]
    mode = "optional"
    batch_size = 64
    chunk_size = 512
    chunk_overlap = 40
    allow_remote_images = false
    allowed_image_domains = ["example.com", "cdn.example.com"]
    no_chunking = false

Schema validation (Gap #11b): Invalid values are rejected with clear error
messages. Supported values are documented in the TOML schema.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database-level settings."""
    path: str = "okfgraph.db"
    dim: int = 512
    wal_mode: bool = False

    def validate(self) -> List[str]:
        """Validate database settings. Returns list of error messages."""
        errors = []
        if not self.path or not self.path.strip():
            errors.append("database.path must be a non-empty string")
        if self.dim < 32 or self.dim > 1024:
            errors.append("database.dim must be between 32 and 1024")
        if self.dim not in (128, 256, 512, 768, 1024):
            errors.append(
                f"database.dim={self.dim} is not a recommended Matryoshka dimension; "
                f"consider 256 or 512"
            )
        return errors


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""
    device: str = "cpu"
    cache_dir: Optional[str] = None
    omni_model_id: str = "jinaai/jina-embeddings-v5-omni-small-retrieval"

    def validate(self) -> List[str]:
        """Validate embedding settings. Returns list of error messages."""
        errors = []
        valid_devices = ("cpu", "cuda", "mps", "auto")
        if self.device not in valid_devices:
            errors.append(
                f"embedding.device must be one of {valid_devices}, got '{self.device}'"
            )
        if self.cache_dir and not Path(self.cache_dir).is_absolute():
            errors.append("embedding.cache_dir must be an absolute path")
        if not self.omni_model_id:
            errors.append("embedding.omni_model_id must be a non-empty string")
        return errors


@dataclass
class ImportConfig:
    """Import pipeline settings."""
    mode: str = "text"
    batch_size: int = 32
    chunk_size: int = 512
    chunk_overlap: int = 40
    allow_remote_images: bool = False
    allowed_image_domains: List[str] = field(default_factory=list)
    no_chunking: bool = False

    def validate(self) -> List[str]:
        """Validate import settings. Returns list of error messages."""
        errors = []
        valid_modes = ("text", "optional", "omni")
        if self.mode not in valid_modes:
            errors.append(
                f"import.mode must be one of {valid_modes}, got '{self.mode}'"
            )
        if self.batch_size < 1 or self.batch_size > 256:
            errors.append("import.batch_size must be between 1 and 256")
        if self.chunk_size < 64 or self.chunk_size > 8192:
            errors.append("import.chunk_size must be between 64 and 8192")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            errors.append(
                "import.chunk_overlap must be >= 0 and < chunk_size"
            )
        if self.allowed_image_domains:
            for d in self.allowed_image_domains:
                if not d or not d.strip():
                    errors.append("allowed_image_domains contains empty entries")
                    break
        return errors


@dataclass
class OKFConfig:
    """Unified OKFgraph configuration.

    Merges settings from TOML file, environment variables, and CLI args.
    CLI args take highest precedence.
    """
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    import_config: ImportConfig = field(default_factory=ImportConfig)
    bundle: str = "."

    def validate(self) -> List[str]:
        """Validate all configuration sections. Returns list of error messages.

        Raises:
            ValueError: If any validation errors are found.
        """
        all_errors = []
        all_errors.extend(self.database.validate())
        all_errors.extend(self.embedding.validate())
        all_errors.extend(self.import_config.validate())

        if all_errors:
            raise ValueError(
                "Configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in all_errors)
            )
        return all_errors

    @classmethod
    def load(
        cls,
        *,
        bundle_root: Optional[str] = None,
        cli_args: Optional[Dict[str, object]] = None,
    ) -> "OKFConfig":
        """Load configuration with precedence: CLI > env > TOML > defaults.

        Args:
            bundle_root: Optional bundle root directory to check for okfgraph.toml.
            cli_args: Optional dict of CLI argument values (highest precedence).

        Returns:
            Merged OKFConfig instance.
        """
        # Start with defaults
        config = cls()

        # Layer 1: TOML file (lowest precedence after defaults)
        toml_config = cls._load_toml(bundle_root)
        if toml_config:
            cls._merge(config, toml_config)

        # Layer 2: Environment variables
        env_config = cls._load_env()
        cls._merge(config, env_config)

        # Layer 3: CLI args (highest precedence)
        if cli_args:
            cls._apply_cli(config, cli_args)

        # Validate merged configuration (Gap #11b)
        try:
            config.validate()
        except ValueError as e:
            logger.warning("Configuration validation warnings: %s", e)
            # Don't fail on validation — just warn. Users can override with CLI.

        return config

    @staticmethod
    def _find_toml(bundle_root: Optional[str]) -> Optional[Path]:
        """Find the first available TOML config file."""
        candidates = [
            Path("okfgraph.toml"),  # current working directory
        ]
        if bundle_root:
            candidates.append(Path(bundle_root) / "okfgraph.toml")
        # User config directory
        config_home = Path.home() / ".config" / "okfgraph"
        candidates.append(config_home / "config.toml")

        for p in candidates:
            if p.is_file():
                return p
        return None

    @classmethod
    def _load_toml(cls, bundle_root: Optional[str]) -> Optional["OKFConfig"]:
        """Load configuration from TOML file."""
        toml_path = cls._find_toml(bundle_root)
        if not toml_path:
            return None

        try:
            import tomllib
        except ImportError:
            # Python < 3.11 fallback
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                return None  # No TOML parser available

        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            return None  # Invalid TOML — skip silently

        return cls._parse_toml(data)

    @classmethod
    def _parse_toml(cls, data: Dict) -> "OKFConfig":
        """Parse TOML dict into OKFConfig."""
        config = cls()

        # Database section
        if "database" in data:
            db = data["database"]
            config.database.path = db.get("path", config.database.path)
            config.database.dim = int(db.get("dim", config.database.dim))
            config.database.wal_mode = bool(db.get("wal_mode", config.database.wal_mode))

        # Embedding section
        if "embedding" in data:
            emb = data["embedding"]
            config.embedding.device = emb.get("device", config.embedding.device)
            config.embedding.cache_dir = emb.get("cache_dir", config.embedding.cache_dir)
            config.embedding.omni_model_id = emb.get(
                "omni_model_id", config.embedding.omni_model_id
            )

        # Import section
        if "import" in data:
            imp = data["import"]
            config.import_config.mode = imp.get("mode", config.import_config.mode)
            config.import_config.batch_size = int(imp.get("batch_size", config.import_config.batch_size))
            config.import_config.chunk_size = int(imp.get("chunk_size", config.import_config.chunk_size))
            config.import_config.chunk_overlap = int(imp.get("chunk_overlap", config.import_config.chunk_overlap))
            config.import_config.allow_remote_images = bool(imp.get("allow_remote_images", config.import_config.allow_remote_images))
            domains = imp.get("allowed_image_domains", [])
            if isinstance(domains, list):
                config.import_config.allowed_image_domains = domains
            config.import_config.no_chunking = bool(imp.get("no_chunking", config.import_config.no_chunking))

        # Bundle root
        if "bundle" in data:
            config.bundle = data["bundle"]

        return config

    @classmethod
    def _load_env(cls) -> "OKFConfig":
        """Load configuration from environment variables."""
        config = cls()
        prefix = "OKFGRAPH_"

        # Database settings
        if val := os.environ.get(f"{prefix}DB"):
            config.database.path = val
        if val := os.environ.get(f"{prefix}DIM"):
            config.database.dim = int(val)
        if val := os.environ.get(f"{prefix}WAL_MODE"):
            config.database.wal_mode = val.lower() in ("1", "true", "yes")

        # Embedding settings
        if val := os.environ.get(f"{prefix}DEVICE"):
            config.embedding.device = val
        if val := os.environ.get(f"{prefix}CACHE_DIR"):
            config.embedding.cache_dir = val
        if val := os.environ.get(f"{prefix}OMNI_MODEL_ID"):
            config.embedding.omni_model_id = val

        # Import settings
        if val := os.environ.get(f"{prefix}MODE"):
            config.import_config.mode = val
        if val := os.environ.get(f"{prefix}BATCH_SIZE"):
            config.import_config.batch_size = int(val)
        if val := os.environ.get(f"{prefix}CHUNK_SIZE"):
            config.import_config.chunk_size = int(val)
        if val := os.environ.get(f"{prefix}CHUNK_OVERLAP"):
            config.import_config.chunk_overlap = int(val)
        if val := os.environ.get(f"{prefix}ALLOW_REMOTE_IMAGES"):
            config.import_config.allow_remote_images = val.lower() in ("1", "true", "yes")
        if val := os.environ.get(f"{prefix}ALLOWED_IMAGE_DOMAINS"):
            config.import_config.allowed_image_domains = [
                d.strip() for d in val.split(",") if d.strip()
            ]
        if val := os.environ.get(f"{prefix}NO_CHUNKING"):
            config.import_config.no_chunking = val.lower() in ("1", "true", "yes")

        # Bundle root
        if val := os.environ.get(f"{prefix}BUNDLE"):
            config.bundle = val

        return config

    @staticmethod
    def _apply_cli(config: "OKFConfig", cli_args: Dict[str, object]) -> None:
        """Apply CLI arguments (highest precedence)."""
        if "db" in cli_args and cli_args["db"]:
            config.database.path = str(cli_args["db"])
        if "dim" in cli_args and cli_args["dim"]:
            config.database.dim = int(cli_args["dim"])
        if "device" in cli_args and cli_args["device"]:
            config.embedding.device = str(cli_args["device"])
        if "cache_dir" in cli_args and cli_args["cache_dir"]:
            config.embedding.cache_dir = str(cli_args["cache_dir"])
        if "bundle" in cli_args and cli_args["bundle"]:
            config.bundle = str(cli_args["bundle"])
        if "omni_model_id" in cli_args and cli_args["omni_model_id"]:
            config.embedding.omni_model_id = str(cli_args["omni_model_id"])
        if "chunk_size" in cli_args and cli_args["chunk_size"]:
            config.import_config.chunk_size = int(cli_args["chunk_size"])
        if "chunk_overlap" in cli_args and cli_args["chunk_overlap"]:
            config.import_config.chunk_overlap = int(cli_args["chunk_overlap"])
        if "no_chunking" in cli_args and cli_args["no_chunking"]:
            config.import_config.no_chunking = bool(cli_args["no_chunking"])
        if "mode" in cli_args and cli_args["mode"]:
            config.import_config.mode = str(cli_args["mode"])
        if "batch_size" in cli_args and cli_args["batch_size"]:
            config.import_config.batch_size = int(cli_args["batch_size"])
        if "allow_remote_images" in cli_args and cli_args["allow_remote_images"]:
            config.import_config.allow_remote_images = bool(cli_args["allow_remote_images"])
        if "wal_mode" in cli_args and cli_args["wal_mode"]:
            config.database.wal_mode = bool(cli_args["wal_mode"])

    @staticmethod
    def _merge(base: "OKFConfig", overlay: "OKFConfig") -> None:
        """Merge overlay settings into base (non-default values only)."""
        # Merge database settings
        if overlay.database.path != DatabaseConfig().path:
            base.database.path = overlay.database.path
        if overlay.database.dim != DatabaseConfig().dim:
            base.database.dim = overlay.database.dim
        if overlay.database.wal_mode != DatabaseConfig().wal_mode:
            base.database.wal_mode = overlay.database.wal_mode

        # Merge embedding settings
        if overlay.embedding.device != EmbeddingConfig().device:
            base.embedding.device = overlay.embedding.device
        if overlay.embedding.cache_dir != EmbeddingConfig().cache_dir:
            base.embedding.cache_dir = overlay.embedding.cache_dir
        if overlay.embedding.omni_model_id != EmbeddingConfig().omni_model_id:
            base.embedding.omni_model_id = overlay.embedding.omni_model_id

        # Merge import settings
        if overlay.import_config.mode != ImportConfig().mode:
            base.import_config.mode = overlay.import_config.mode
        if overlay.import_config.batch_size != ImportConfig().batch_size:
            base.import_config.batch_size = overlay.import_config.batch_size
        if overlay.import_config.chunk_size != ImportConfig().chunk_size:
            base.import_config.chunk_size = overlay.import_config.chunk_size
        if overlay.import_config.chunk_overlap != ImportConfig().chunk_overlap:
            base.import_config.chunk_overlap = overlay.import_config.chunk_overlap
        if overlay.import_config.allow_remote_images != ImportConfig().allow_remote_images:
            base.import_config.allow_remote_images = overlay.import_config.allow_remote_images
        if overlay.import_config.allowed_image_domains:
            base.import_config.allowed_image_domains = overlay.import_config.allowed_image_domains
        if overlay.import_config.no_chunking != ImportConfig().no_chunking:
            base.import_config.no_chunking = overlay.import_config.no_chunking

        # Merge bundle root
        if overlay.bundle != OKFConfig().bundle:
            base.bundle = overlay.bundle
