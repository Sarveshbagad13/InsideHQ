"""In-memory Rate Limiter & Token Budget Service (Feature 19).

Provides per-user sliding-window rate limiting on /chat endpoints (stricter
for expensive operations like streaming) and daily token-cost budget enforcement
returning HTTP 429 when limits are exceeded.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory sliding window rate limiter and daily token budget manager."""

    def __init__(
        self,
        default_window_seconds: float = 60.0,
        default_request_limit: int = 10,
        expensive_request_limit: int = 3,
        daily_token_budget: int = 10000,
    ):
        self.window_seconds = default_window_seconds
        self.default_request_limit = default_request_limit
        self.expensive_request_limit = expensive_request_limit
        self.daily_token_budget = daily_token_budget

        # user_id -> list of float timestamps
        self._user_timestamps: Dict[uuid.UUID, List[float]] = {}
        # user_id -> {"date": "YYYY-MM-DD", "tokens": int}
        self._user_daily_tokens: Dict[uuid.UUID, Dict[str, Any]] = {}

    def _get_today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def check_rate_limit(
        self,
        user_id: uuid.UUID,
        is_expensive: bool = False,
        now: Optional[float] = None,
    ) -> None:
        """Enforce sliding window rate limit for user.

        Raises HTTP 429 if the request count exceeds limit within the sliding window.
        """
        current_time = now if now is not None else time.time()
        limit = self.expensive_request_limit if is_expensive else self.default_request_limit
        cutoff = current_time - self.window_seconds

        timestamps = self._user_timestamps.get(user_id, [])
        # Retain only timestamps within the current sliding window
        valid_timestamps = [t for t in timestamps if t > cutoff]
        self._user_timestamps[user_id] = valid_timestamps

        if len(valid_timestamps) >= limit:
            retry_after = int(self.window_seconds - (current_time - valid_timestamps[0]))
            retry_after = max(retry_after, 1)
            logger.warning(
                "Rate limit exceeded for user %s (is_expensive=%s, count=%d, limit=%d)",
                user_id,
                is_expensive,
                len(valid_timestamps),
                limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: Maximum {limit} requests per {int(self.window_seconds)}s window.",
                headers={"Retry-After": str(retry_after)},
            )

        valid_timestamps.append(current_time)

    def check_daily_token_budget(self, user_id: uuid.UUID, estimated_tokens: int = 0) -> None:
        """Verify that user has not exceeded their daily token budget.

        Raises HTTP 429 if cumulative daily tokens exceed budget.
        """
        today = self._get_today_str()
        record = self._user_daily_tokens.get(user_id, {"date": today, "tokens": 0})

        if record["date"] != today:
            record = {"date": today, "tokens": 0}

        if record["tokens"] + estimated_tokens >= self.daily_token_budget:
            logger.warning(
                "Daily token budget exceeded for user %s (used=%d, requested=%d, budget=%d)",
                user_id,
                record["tokens"],
                estimated_tokens,
                self.daily_token_budget,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily token budget exceeded ({self.daily_token_budget} tokens/day).",
                headers={"Retry-After": "86400"},
            )

    def record_token_usage(self, user_id: uuid.UUID, token_count: int) -> None:
        """Record consumed tokens for user's daily budget."""
        today = self._get_today_str()
        record = self._user_daily_tokens.get(user_id, {"date": today, "tokens": 0})

        if record["date"] != today:
            record = {"date": today, "tokens": 0}

        record["tokens"] += token_count
        self._user_daily_tokens[user_id] = record

    def get_user_usage(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Return current rate limit and token usage metrics for a user."""
        cutoff = time.time() - self.window_seconds
        valid_ts = [t for t in self._user_timestamps.get(user_id, []) if t > cutoff]
        today = self._get_today_str()
        token_record = self._user_daily_tokens.get(user_id, {"date": today, "tokens": 0})
        tokens_today = token_record["tokens"] if token_record["date"] == today else 0

        return {
            "requests_in_window": len(valid_ts),
            "window_seconds": self.window_seconds,
            "daily_tokens_used": tokens_today,
            "daily_token_budget": self.daily_token_budget,
        }

    def reset_user(self, user_id: uuid.UUID) -> None:
        """Reset rate limit timestamps and daily token usage for a user."""
        self._user_timestamps.pop(user_id, None)
        self._user_daily_tokens.pop(user_id, None)

    def reset_all(self) -> None:
        """Reset all in-memory rate limiting and token tracking data across all users."""
        self._user_timestamps.clear()
        self._user_daily_tokens.clear()


# Global rate limiter singleton instance
rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Dependency getter for RateLimiter instance."""
    return rate_limiter
