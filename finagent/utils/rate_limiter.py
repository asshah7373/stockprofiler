"""
Rate Limiter
API rate limiting to respect source limits.
"""

import time
import threading
from collections import defaultdict
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit."""
    requests_per_second: float
    requests_per_minute: int
    requests_per_hour: int
    max_burst: int = 5


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Supports multiple named limiters for different APIs.
    Thread-safe implementation.
    """

    # Default rate limits for known APIs
    DEFAULT_LIMITS = {
        "yfinance": RateLimitConfig(
            requests_per_second=2,
            requests_per_minute=100,
            requests_per_hour=2000,
            max_burst=5
        ),
        "nse": RateLimitConfig(
            requests_per_second=1,
            requests_per_minute=30,
            requests_per_hour=500,
            max_burst=3
        ),
        "bse": RateLimitConfig(
            requests_per_second=1,
            requests_per_minute=30,
            requests_per_hour=500,
            max_burst=3
        ),
        "newsapi": RateLimitConfig(
            requests_per_second=1,
            requests_per_minute=50,
            requests_per_hour=500,
            max_burst=3
        ),
        "default": RateLimitConfig(
            requests_per_second=2,
            requests_per_minute=60,
            requests_per_hour=1000,
            max_burst=5
        )
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.locks = defaultdict(threading.Lock)
        self.buckets = {}  # name -> {tokens, last_refill}
        self.request_counts = defaultdict(list)  # name -> [timestamps]
        self.configs = dict(self.DEFAULT_LIMITS)

    def configure(self, name: str, config: RateLimitConfig):
        """Configure rate limit for a named API."""
        self.configs[name] = config
        self.logger.info(f"Configured rate limit for {name}: {config}")

    def _get_config(self, name: str) -> RateLimitConfig:
        """Get rate limit config for a name."""
        return self.configs.get(name, self.configs["default"])

    def _refill_bucket(self, name: str) -> float:
        """Refill token bucket and return available tokens."""
        config = self._get_config(name)

        if name not in self.buckets:
            self.buckets[name] = {
                "tokens": config.max_burst,
                "last_refill": time.time()
            }
            return config.max_burst

        bucket = self.buckets[name]
        now = time.time()
        time_passed = now - bucket["last_refill"]

        # Add tokens based on time passed
        new_tokens = time_passed * config.requests_per_second
        bucket["tokens"] = min(config.max_burst, bucket["tokens"] + new_tokens)
        bucket["last_refill"] = now

        return bucket["tokens"]

    def _clean_old_requests(self, name: str, window_seconds: int = 3600):
        """Remove request timestamps older than window."""
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        self.request_counts[name] = [
            ts for ts in self.request_counts[name]
            if ts > cutoff
        ]

    def _check_rate_limits(self, name: str) -> tuple:
        """Check if rate limits are exceeded. Returns (allowed, wait_time)."""
        config = self._get_config(name)
        self._clean_old_requests(name)

        now = datetime.now()
        requests = self.request_counts[name]

        # Check per-minute limit
        minute_ago = now - timedelta(minutes=1)
        minute_requests = sum(1 for ts in requests if ts > minute_ago)
        if minute_requests >= config.requests_per_minute:
            # Find when oldest request in window will expire
            oldest_in_minute = [ts for ts in requests if ts > minute_ago]
            if oldest_in_minute:
                wait = (min(oldest_in_minute) + timedelta(minutes=1) - now).total_seconds()
                return False, max(0, wait)

        # Check per-hour limit
        hour_ago = now - timedelta(hours=1)
        hour_requests = sum(1 for ts in requests if ts > hour_ago)
        if hour_requests >= config.requests_per_hour:
            oldest_in_hour = [ts for ts in requests if ts > hour_ago]
            if oldest_in_hour:
                wait = (min(oldest_in_hour) + timedelta(hours=1) - now).total_seconds()
                return False, max(0, wait)

        return True, 0

    def acquire(self, name: str = "default", block: bool = True) -> bool:
        """
        Acquire permission to make a request.

        Args:
            name: API name
            block: If True, wait until allowed. If False, return immediately.

        Returns:
            True if allowed, False if blocked and block=False
        """
        with self.locks[name]:
            config = self._get_config(name)

            while True:
                # Refill bucket
                tokens = self._refill_bucket(name)

                # Check token availability
                if tokens >= 1:
                    # Check rate limits
                    allowed, wait_time = self._check_rate_limits(name)

                    if allowed:
                        # Consume token and record request
                        self.buckets[name]["tokens"] -= 1
                        self.request_counts[name].append(datetime.now())
                        return True

                    if not block:
                        return False

                    # Wait and retry
                    self.logger.debug(f"Rate limited for {name}, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                else:
                    if not block:
                        return False

                    # Wait for token refill
                    wait = 1.0 / config.requests_per_second
                    time.sleep(wait)

    def wait(self, name: str = "default"):
        """Wait until a request is allowed. Alias for acquire(block=True)."""
        self.acquire(name, block=True)

    def try_acquire(self, name: str = "default") -> bool:
        """Try to acquire without blocking."""
        return self.acquire(name, block=False)

    def get_status(self, name: str = "default") -> Dict:
        """Get current rate limit status."""
        config = self._get_config(name)
        self._clean_old_requests(name)

        now = datetime.now()
        requests = self.request_counts[name]

        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        return {
            "name": name,
            "tokens_available": self.buckets.get(name, {}).get("tokens", config.max_burst),
            "requests_last_minute": sum(1 for ts in requests if ts > minute_ago),
            "requests_last_hour": sum(1 for ts in requests if ts > hour_ago),
            "limits": {
                "per_second": config.requests_per_second,
                "per_minute": config.requests_per_minute,
                "per_hour": config.requests_per_hour,
                "max_burst": config.max_burst
            }
        }

    def reset(self, name: Optional[str] = None):
        """Reset rate limiter state."""
        if name:
            if name in self.buckets:
                del self.buckets[name]
            if name in self.request_counts:
                del self.request_counts[name]
        else:
            self.buckets.clear()
            self.request_counts.clear()


class RateLimitedSession:
    """
    Wrapper for requests with automatic rate limiting.
    """

    def __init__(self, limiter: RateLimiter, name: str = "default"):
        self.limiter = limiter
        self.name = name
        self.logger = logging.getLogger(__name__)

    def request(self, method: str, url: str, **kwargs):
        """Make a rate-limited request."""
        import requests

        self.limiter.wait(self.name)
        self.logger.debug(f"Making {method} request to {url}")

        response = requests.request(method, url, **kwargs)
        return response

    def get(self, url: str, **kwargs):
        """Rate-limited GET request."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        """Rate-limited POST request."""
        return self.request("POST", url, **kwargs)
