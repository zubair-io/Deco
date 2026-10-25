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
    csv_path = tmp_path / "met.csv"
    csv_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    adapter = MetAdapter(csv_path, client=client)
    store = InMemoryRecordStore()
    cache = ImageCache(tmp_path / "images")
    cfg = IngestConfig(rate_limit_rps=1000.0, max_backoff_s=1.0)

    first = run_ingest(adapter, store, cache, client, cfg)

    assert first.no_image == 1
    assert first.images_fetched == 1
    assert first.images_cached == 0
    assert first.failed == 0
    assert store.count() == 1
    assert store.get("met:436532") is None
    resolved = store.get("met:436535")
    assert resolved is not None
    assert resolved.image_url == "https://images.example.com/436535.jpg"
    assert resolved.title == "Wheat Field with Cypresses"
    assert len(image_calls) == 1

    # Simulate a nightly Met CSV refresh that updates a title for an
    # already-ingested record; this must not re-resolve or re-fetch its image.
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace(
            "Wheat Field with Cypresses", "Wheat Field with Cypresses (Retitled)"
        ),
        encoding="utf-8",
    )

    second = run_ingest(adapter, store, cache, client, cfg)

    assert second.images_fetched == 0
    assert second.images_cached == second.records
    assert second.failed == 0
    assert store.count() == 1
    assert len(image_calls) == 1  # never re-downloaded the cached image
    updated = store.get("met:436535")
    assert updated is not None
    assert updated.title == "Wheat Field with Cypresses (Retitled)"
    assert updated.image_url == "https://images.example.com/436535.jpg"


def test_run_ingest_continues_past_dead_image_url_and_counts_failed(tmp_path: Path):
    csv_text = (
        "Object Number,Is Highlight,Is Timeline Work,Is Public Domain,Object ID,"
        "Department,Title,Artist Display Name,Object Date,Medium,Wikidata URL\n"
        "1,False,False,True,111,European Paintings,Good Painting,Artist A,1900,Oil,\n"
        "2,False,False,True,222,European Paintings,Dead Painting,Artist B,1900,Oil,\n"
    )
    csv_path = tmp_path / "met.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/objects/111"):
            return httpx.Response(200, json={"primaryImage": "https://images.example.com/111.jpg"})
        if path.endswith("/objects/222"):
            return httpx.Response(200, json={"primaryImage": "https://images.example.com/222.jpg"})
        if path == "/111.jpg":
            return httpx.Response(200, content=IMAGE_BYTES)
        if path == "/222.jpg":
            return httpx.Response(404)
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = MetAdapter(csv_path, client=client)
    store = InMemoryRecordStore()
    cache = ImageCache(tmp_path / "images")
    cfg = IngestConfig(rate_limit_rps=1000.0, max_backoff_s=1.0)

    stats = run_ingest(adapter, store, cache, client, cfg)

    assert stats.failed == 1
    assert stats.images_fetched == 1
    assert stats.no_image == 0
    # Metadata for both records is upserted even though one image 404s.
    assert store.count() == 2
    assert store.get("met:111").image_url == "https://images.example.com/111.jpg"
    assert store.get("met:222").image_url == "https://images.example.com/222.jpg"
    assert cache.has("met:111") is True
    assert cache.has("met:222") is False
