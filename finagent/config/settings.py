"""
Configuration Settings
Central configuration for the FinAgent system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Application settings with defaults and environment variable support."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))

    # API Keys (load from environment)
    anthropic_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    finnhub_key: Optional[str] = None
    llama_cloud_api_key: Optional[str] = None

    # Broker API keys (optional)
    zerodha_api_key: Optional[str] = None
    zerodha_api_secret: Optional[str] = None
    upstox_api_key: Optional[str] = None

    # Database settings
    market_data_db: str = "data/cache/market_data.db"
    circulars_db: str = "data/cache/circulars.db"
    compliance_db: str = "logs/compliance.db"
    risk_profiles_db: str = "data/cache/risk_profiles.db"

    # Vector store settings
    vector_store_path: str = "data/vectors"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Cache settings
    cache_ttl_hours: int = 24
    price_cache_ttl_minutes: int = 5

    # Rate limiting
    yfinance_rate_limit: float = 2.0  # requests per second
    nse_rate_limit: float = 1.0
    bse_rate_limit: float = 1.0

    # Analysis thresholds
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    high_beta_threshold: float = 1.5
    low_beta_threshold: float = 0.8

    # Risk profile settings
    profile_expiry_months: int = 12

    # Compliance
    compliance_retention_years: int = 5
    enable_audit_logging: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Load settings from environment variables."""
        # API Keys
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.newsapi_key = os.getenv("NEWSAPI_KEY", self.newsapi_key)
        self.finnhub_key = os.getenv("FINNHUB_KEY", self.finnhub_key)
        self.llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY", self.llama_cloud_api_key)

        # Broker keys
        self.zerodha_api_key = os.getenv("ZERODHA_API_KEY", self.zerodha_api_key)
        self.zerodha_api_secret = os.getenv("ZERODHA_API_SECRET", self.zerodha_api_secret)
        self.upstox_api_key = os.getenv("UPSTOX_API_KEY", self.upstox_api_key)

        # Log level
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

        # Create directories
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        dirs = [
            self.data_dir,
            self.logs_dir,
            self.cache_dir,
            Path(self.vector_store_path),
            Path("data/circulars")
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=self.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.logs_dir / "finagent.log")
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding sensitive data)."""
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "logs_dir": str(self.logs_dir),
            "vector_store_path": self.vector_store_path,
            "embedding_model": self.embedding_model,
            "cache_ttl_hours": self.cache_ttl_hours,
            "log_level": self.log_level,
            "has_newsapi_key": self.newsapi_key is not None,
            "has_finnhub_key": self.finnhub_key is not None,
            "has_broker_keys": self.zerodha_api_key is not None or self.upstox_api_key is not None
        }

    @classmethod
    def from_file(cls, config_path: str) -> "Settings":
        """Load settings from a JSON config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(**config)

    def save_to_file(self, config_path: str):
        """Save non-sensitive settings to a config file."""
        config = self.to_dict()

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def configure_settings(**kwargs) -> Settings:
    """Configure settings with custom values."""
    global _settings
    _settings = Settings(**kwargs)
    return _settings
