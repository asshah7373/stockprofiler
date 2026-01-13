"""
Cache Manager
SQLite-based caching for API responses and computed data.
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class CacheManager:
    """
    SQLite-based cache for storing API responses and computed results.

    Features:
    - TTL (time-to-live) support
    - Automatic expiration cleanup
    - Key namespacing
    - Serialization of complex objects
    """

    def __init__(
        self,
        db_path: str = "data/cache/app_cache.db",
        default_ttl_hours: int = 24
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.logger = logging.getLogger(__name__)
        self._init_db()

    def _init_db(self):
        """Initialize cache database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                value BLOB NOT NULL,
                value_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                hits INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_namespace
            ON cache(namespace)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires
            ON cache(expires_at)
        """)

        conn.commit()
        conn.close()

    def _generate_key(self, key: str, namespace: str = "default") -> str:
        """Generate cache key with namespace."""
        return f"{namespace}:{key}"

    def _serialize(self, value: Any) -> tuple:
        """Serialize value for storage."""
        if isinstance(value, (dict, list)):
            return json.dumps(value).encode(), "json"
        elif isinstance(value, str):
            return value.encode(), "str"
        elif isinstance(value, (int, float)):
            return str(value).encode(), "number"
        else:
            return pickle.dumps(value), "pickle"

    def _deserialize(self, data: bytes, value_type: str) -> Any:
        """Deserialize stored value."""
        if value_type == "json":
            return json.loads(data.decode())
        elif value_type == "str":
            return data.decode()
        elif value_type == "number":
            val = data.decode()
            return float(val) if '.' in val else int(val)
        else:
            return pickle.loads(data)

    def get(
        self,
        key: str,
        namespace: str = "default"
    ) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            Cached value or None if not found/expired
        """
        full_key = self._generate_key(key, namespace)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT value, value_type, expires_at, hits
            FROM cache
            WHERE key = ?
        """, (full_key,))

        row = cursor.fetchone()

        if row:
            value, value_type, expires_at, hits = row

            # Check expiration
            if datetime.fromisoformat(expires_at) < datetime.now():
                # Expired, delete and return None
                cursor.execute("DELETE FROM cache WHERE key = ?", (full_key,))
                conn.commit()
                conn.close()
                return None

            # Update hit count
            cursor.execute(
                "UPDATE cache SET hits = ? WHERE key = ?",
                (hits + 1, full_key)
            )
            conn.commit()
            conn.close()

            return self._deserialize(value, value_type)

        conn.close()
        return None

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl_hours: Optional[int] = None
    ):
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl_hours: Time-to-live in hours (None = default)
        """
        full_key = self._generate_key(key, namespace)
        data, value_type = self._serialize(value)

        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
        expires_at = (datetime.now() + ttl).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO cache
            (key, namespace, value, value_type, created_at, expires_at, hits)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (
            full_key,
            namespace,
            data,
            value_type,
            datetime.now().isoformat(),
            expires_at
        ))

        conn.commit()
        conn.close()

    def delete(self, key: str, namespace: str = "default"):
        """Delete a cached value."""
        full_key = self._generate_key(key, namespace)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?", (full_key,))
        conn.commit()
        conn.close()

    def clear_namespace(self, namespace: str):
        """Clear all entries in a namespace."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE namespace = ?", (namespace,))
        conn.commit()
        conn.close()
        self.logger.info(f"Cleared cache namespace: {namespace}")

    def clear_expired(self) -> int:
        """Remove all expired entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM cache WHERE expires_at < ?",
            (datetime.now().isoformat(),)
        )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        self.logger.info(f"Cleared {deleted} expired cache entries")
        return deleted

    def clear_all(self):
        """Clear entire cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        self.logger.warning("Cleared all cache entries")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total entries
        cursor.execute("SELECT COUNT(*) FROM cache")
        total = cursor.fetchone()[0]

        # Entries by namespace
        cursor.execute("""
            SELECT namespace, COUNT(*) as count
            FROM cache
            GROUP BY namespace
        """)
        namespaces = {row[0]: row[1] for row in cursor.fetchall()}

        # Total hits
        cursor.execute("SELECT SUM(hits) FROM cache")
        total_hits = cursor.fetchone()[0] or 0

        # Expired entries
        cursor.execute(
            "SELECT COUNT(*) FROM cache WHERE expires_at < ?",
            (datetime.now().isoformat(),)
        )
        expired = cursor.fetchone()[0]

        # DB size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        conn.close()

        return {
            "total_entries": total,
            "expired_entries": expired,
            "total_hits": total_hits,
            "namespaces": namespaces,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / 1024 / 1024, 2)
        }

    def cache_decorator(
        self,
        namespace: str = "default",
        ttl_hours: int = 24
    ):
        """
        Decorator for caching function results.

        Usage:
            @cache.cache_decorator(namespace="api", ttl_hours=1)
            def fetch_data(ticker):
                ...
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [func.__name__] + [str(a) for a in args]
                key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

                # Check cache
                cached = self.get(cache_key, namespace)
                if cached is not None:
                    return cached

                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, namespace, ttl_hours)

                return result

            return wrapper
        return decorator
