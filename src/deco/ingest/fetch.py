"""Rate-limited HTTP fetching with exponential backoff, plus an on-disk image cache."""

import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

_RETRY_STATUS_FLOOR = 500


class RateLimiter:
    """Blocks in `.wait()` to enforce at most `rps` calls per second.

    `sleep`/`clock` are injectable so tests can drive it with a fake clock
    instead of burning wall-clock time.
    """

    def __init__(
        self,
        rps: float,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval = 1.0 / rps
        self._sleep = sleep
        self._clock = clock
        self._next_allowed: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._next_allowed is None:
            self._next_allowed = now
        if now < self._next_allowed:
            self._sleep(self._next_allowed - now)
        self._next_allowed = max(now, self._next_allowed) + self._min_interval


def fetch_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    limiter: RateLimiter,
    max_backoff_s: float,
    retries: int = 6,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET `url`, retrying 429/5xx responses with exponential backoff.

    Backoff doubles starting at 1s, capped at `max_backoff_s`. After the
    initial attempt plus `retries` retries all fail, raises
    `httpx.HTTPStatusError` (via `response.raise_for_status()`). Any other
    error status (e.g. 404) raises immediately without retrying.
    """
    backoff = 1.0
    for attempt in range(retries + 1):
        limiter.wait()
        response = client.get(url)
        if response.status_code == 429 or response.status_code >= _RETRY_STATUS_FLOOR:
            if attempt == retries:
                response.raise_for_status()
            sleep(min(backoff, max_backoff_s))
            backoff = min(backoff * 2, max_backoff_s)
            continue
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")  # pragma: no cover


class ImageCache:
    """On-disk cache of fetched image bytes, keyed by record_id.

    Layout: `root/<source>/<source_object_id>.jpg`.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, record_id: str) -> Path:
        source, _, source_object_id = record_id.partition(":")
        return self.root / source / f"{source_object_id}.jpg"

    def has(self, record_id: str) -> bool:
        return self.path_for(record_id).exists()

    def put(self, record_id: str, data: bytes) -> Path:
        path = self.path_for(record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
        return path
