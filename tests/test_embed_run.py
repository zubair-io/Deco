"""Tests for deco.embed.run: resumable embed pass."""

from pathlib import Path

import numpy as np
from PIL import Image

from deco.embed.embedder import FakeEmbedder
from deco.embed.run import run_embed_pass
from deco.ingest.fetch import ImageCache
from deco.records import Record
from deco.stores.memory import InMemoryRecordStore


class InMemoryVectorSink:
    """Dict-backed VectorSink used by tests instead of a real vector index."""

    def __init__(self) -> None:
        self._vectors: dict[tuple[str, str], np.ndarray] = {}

    def add(self, record_id: str, vector: np.ndarray, model_version: str) -> None:
        self._vectors[(record_id, model_version)] = vector

    def has(self, record_id: str, model_version: str) -> bool:
        return (record_id, model_version) in self._vectors


def _record(source_object_id: str) -> Record:
    return Record(
        source="met",
        source_object_id=source_object_id,
        title="t",
        artist="a",
        date="d",
        medium="m",
        accession_number="acc",
        image_url="https://example.com/x.jpg",
        license="CC0",
    )


def _write_image(cache: ImageCache, record_id: str, color: tuple[int, int, int] = (10, 20, 30)):
    path = cache.path_for(record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="JPEG")


def test_run_embed_pass_embeds_cached_images_and_is_resumable(tmp_path: Path):
    store = InMemoryRecordStore()
    store.upsert(_record("1"))
    store.upsert(_record("2"))
    cache = ImageCache(tmp_path)
    _write_image(cache, "met:1", (10, 20, 30))
    _write_image(cache, "met:2", (200, 100, 50))
    embedder = FakeEmbedder()
    sink = InMemoryVectorSink()

    first = run_embed_pass(store, cache, embedder, sink)

    assert first.embedded == 2
    assert first.skipped_existing == 0
    assert first.no_image == 0
    assert first.failed == 0
    assert sink.has("met:1", "fake-v1")
    assert sink.has("met:2", "fake-v1")

    second = run_embed_pass(store, cache, embedder, sink)

    assert second.embedded == 0
    assert second.skipped_existing == 2


def test_run_embed_pass_re_embeds_all_under_new_model_version(tmp_path: Path):
    store = InMemoryRecordStore()
    store.upsert(_record("1"))
    store.upsert(_record("2"))
    cache = ImageCache(tmp_path)
    _write_image(cache, "met:1")
    _write_image(cache, "met:2")
    sink = InMemoryVectorSink()
    run_embed_pass(store, cache, FakeEmbedder(model_version="fake-v1"), sink)

    stats = run_embed_pass(store, cache, FakeEmbedder(model_version="fake-v2"), sink)

    assert stats.embedded == 2
    assert stats.skipped_existing == 0
    assert sink.has("met:1", "fake-v1")
    assert sink.has("met:1", "fake-v2")


def test_run_embed_pass_counts_records_without_cached_image(tmp_path: Path):
    store = InMemoryRecordStore()
    store.upsert(_record("1"))
    cache = ImageCache(tmp_path)
    sink = InMemoryVectorSink()

    stats = run_embed_pass(store, cache, FakeEmbedder(), sink)

    assert stats.no_image == 1
    assert stats.embedded == 0


def test_run_embed_pass_counts_corrupt_image_as_failed_and_continues(tmp_path: Path):
    store = InMemoryRecordStore()
    store.upsert(_record("1"))
    store.upsert(_record("2"))
    cache = ImageCache(tmp_path)
    path = cache.path_for("met:1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-image")
    _write_image(cache, "met:2")
    sink = InMemoryVectorSink()

    stats = run_embed_pass(store, cache, FakeEmbedder(), sink)

    assert stats.failed == 1
    assert stats.embedded == 1
    assert sink.has("met:2", "fake-v1")
    assert not sink.has("met:1", "fake-v1")
