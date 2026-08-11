"""Tests for deco.ingest.fetch: RateLimiter, fetch_with_backoff, ImageCache."""

from pathlib import Path

import httpx
import pytest

from deco.ingest.fetch import ImageCache, RateLimiter, fetch_with_backoff


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


class NullLimiter:
    def wait(self) -> None:
        pass


def test_rate_limiter_enforces_rps():
    clock = FakeClock()
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.now += seconds

    limiter = RateLimiter(rps=2, sleep=fake_sleep, clock=clock.time)
    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert sum(sleeps) >= 1.0


def test_fetch_with_backoff_retries_then_succeeds():
    statuses = iter([429, 500, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, content=b"image-bytes")
        return httpx.Response(status)

    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))

    response = fetch_with_backoff(
        client,
        "https://images.example.com/a.jpg",
        limiter=NullLimiter(),
        max_backoff_s=60.0,
        sleep=sleeps.append,
    )

    assert response.status_code == 200
    assert response.content == b"image-bytes"
    assert sleeps == [1, 2]


def test_fetch_with_backoff_retries_403_then_succeeds():
    statuses = iter([403, 429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, content=b"image-bytes")
        return httpx.Response(status)

    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))

    response = fetch_with_backoff(
        client,
        "https://images.example.com/a.jpg",
        limiter=NullLimiter(),
        max_backoff_s=60.0,
        sleep=sleeps.append,
    )

    assert response.status_code == 200
    assert response.content == b"image-bytes"
    assert sleeps == [1, 2]


def test_fetch_with_backoff_raises_after_exhausting_retries():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_with_backoff(
            client,
            "https://images.example.com/a.jpg",
            limiter=NullLimiter(),
            max_backoff_s=60.0,
            sleep=lambda _: None,
        )

    assert calls == 7  # initial attempt + 6 retries


def test_image_cache_put_then_has_and_path_layout(tmp_path: Path):
    cache = ImageCache(tmp_path)
    record_id = "met:436535"

    assert cache.has(record_id) is False

    path = cache.put(record_id, b"jpeg-bytes")

    assert path == tmp_path / "met" / "436535.jpg"
    assert cache.path_for(record_id) == path
    assert cache.has(record_id) is True
    assert path.read_bytes() == b"jpeg-bytes"
    # atomic write: no leftover temp file in the cache directory
    assert {p.name for p in path.parent.iterdir()} == {"436535.jpg"}
