from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional


class TokenUsageTracker:

    _lock = Lock()
    _totals: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._totals = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }

    @classmethod
    def record(cls, usage: Optional[Any]) -> None:
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
        with cls._lock:
            cls._totals["prompt_tokens"] += prompt
            cls._totals["completion_tokens"] += completion
            cls._totals["total_tokens"] += total
            cls._totals["calls"] += 1

    @classmethod
    def snapshot(cls) -> Dict[str, int]:
        with cls._lock:
            return dict(cls._totals)
