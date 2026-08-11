"""Tests for deco.ingest.run: resumable, idempotent ingest orchestration."""

from pathlib import Path

import httpx

from deco.config import IngestConfig
from deco.ingest.fetch import ImageCache
from deco.ingest.met import MetAdapter
from deco.ingest.run import run_ingest
from deco.stores.memory import InMemoryRecordStore

FIXTURE = Path(__file__).parent / "fixtures" / "met_sample.csv"

IMAGE_BYTES = b"fake-jpeg-bytes"


def make_handler(image_calls: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/objects/436535"):
            return httpx.Response(
                200, json={"primaryImage": "https://images.example.com/436535.jpg"}
            )
        if path.endswith("/objects/436532"):
            # this record's primary image never resolves -> should be skipped
            return httpx.Response(200, json={"primaryImage": ""})
        if path == "/436535.jpg":
            image_calls.append(path)
            return httpx.Response(200, content=IMAGE_BYTES)
        raise AssertionError(f"unexpected request: {request.url}")

    return handler


def test_run_ingest_is_resumable_and_skips_no_image_records(tmp_path: Path):
    image_calls: list[str] = []
    client = httpx.Client(transport=httpx.MockTransport(make_handler(image_calls)))
    adapter = MetAdapter(FIXTURE, client=client)
    store = InMemoryRecordStore()
    cache = ImageCache(tmp_path)
    cfg = IngestConfig(rate_limit_rps=1000.0, max_backoff_s=1.0)

    first = run_ingest(adapter, store, cache, client, cfg)

    assert first.no_image == 1
    assert first.images_fetched == 1
    assert first.images_cached == 0
    assert store.count() == 1
    assert store.get("met:436532") is None
    resolved = store.get("met:436535")
    assert resolved is not None
    assert resolved.image_url == "https://images.example.com/436535.jpg"
    assert len(image_calls) == 1

    second = run_ingest(adapter, store, cache, client, cfg)

    assert second.images_fetched == 0
    assert second.images_cached == second.records
    assert store.count() == 1
    assert len(image_calls) == 1  # never re-downloaded the cached image
