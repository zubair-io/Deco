"""Tests for deco.api.pipeline: end-to-end identify orchestration."""

from PIL import Image

from deco.api.pipeline import IdentifyPipeline
from deco.config import (
    DecoConfig,
    EmbeddingConfig,
    GateConfig,
    IngestConfig,
    PathsConfig,
    RetrievalConfig,
)
from deco.embed.embedder import FakeEmbedder
from deco.index.memory import InMemoryVectorIndex
from deco.records import Record
from deco.rerank.base import FakeReranker
from deco.stores.memory import InMemoryRecordStore


def _record(source_object_id: str) -> Record:
    return Record(
        source="met",
        source_object_id=source_object_id,
        title=f"Title {source_object_id}",
        artist="Artist",
        date="1900",
        medium="Oil on canvas",
        accession_number=f"acc-{source_object_id}",
        image_url=f"https://example.com/{source_object_id}.jpg",
        license="CC0",
    )


def _cfg(min_cosine: float = 0.0, min_inliers: int = 30) -> DecoConfig:
    return DecoConfig(
        gate=GateConfig(min_cosine=min_cosine, min_inliers=min_inliers),
        retrieval=RetrievalConfig(top_k=3, rerank_top_n=3),
        ingest=IngestConfig(rate_limit_rps=1.0, max_backoff_s=1.0),
        embedding=EmbeddingConfig(model_version="fake-v1"),
        paths=PathsConfig(image_cache_dir="/tmp/deco-cache"),
    )


def _make_corpus() -> tuple[InMemoryRecordStore, InMemoryVectorIndex, dict[str, Image.Image]]:
    store = InMemoryRecordStore()
    index = InMemoryVectorIndex()
    embedder = FakeEmbedder(dim=8, model_version="fake-v1")
    images: dict[str, Image.Image] = {}

    for source_object_id, color in [("A", (255, 0, 0)), ("B", (0, 255, 0)), ("C", (0, 0, 255))]:
        record = _record(source_object_id)
        store.upsert(record)
        img = Image.new("RGB", (4, 4), color)
        images[record.record_id] = img
        vector = embedder.embed(img)
        index.add(record.record_id, vector, embedder.model_version)

    return store, index, images


def _image_loader(images: dict[str, Image.Image]):
    def loader(record_id: str) -> Image.Image | None:
        return images.get(record_id)

    return loader


def test_identify_confident_returns_geometric_winner():
    store, index, images = _make_corpus()
    embedder = FakeEmbedder(dim=8, model_version="fake-v1")
    # Reranker scripts B as the clear geometric winner regardless of cosine order.
    reranker = FakeReranker(scores={"met:A": 5, "met:B": 120, "met:C": 0})
    cfg = _cfg(min_cosine=0.0, min_inliers=30)

    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        image_loader=_image_loader(images),
        reranker=reranker,
        cfg=cfg,
    )

    query_img = Image.new("RGB", (4, 4), (0, 255, 0))  # same content as B's fixture
    result = pipeline.identify(query_img)

    assert result.confident is True
    assert result.matches[0].record.record_id == "met:B"
    assert result.matches[0].inliers == 120


def test_identify_not_confident_when_all_inliers_zero():
    store, index, images = _make_corpus()
    embedder = FakeEmbedder(dim=8, model_version="fake-v1")
    reranker = FakeReranker(scores={"met:A": 0, "met:B": 0, "met:C": 0})
    cfg = _cfg(min_cosine=0.0, min_inliers=30)

    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        image_loader=_image_loader(images),
        reranker=reranker,
        cfg=cfg,
    )

    query_img = Image.new("RGB", (4, 4), (10, 20, 30))
    result = pipeline.identify(query_img)

    assert result.confident is False


def test_identify_drops_candidates_missing_from_store(caplog):
    store, index, images = _make_corpus()
    # Remove B's record from the store but leave it in the index/images.
    store._records.pop("met:B")

    embedder = FakeEmbedder(dim=8, model_version="fake-v1")
    reranker = FakeReranker(scores={"met:A": 10, "met:B": 120, "met:C": 5})
    cfg = _cfg(min_cosine=0.0, min_inliers=0)

    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        image_loader=_image_loader(images),
        reranker=reranker,
        cfg=cfg,
    )

    query_img = Image.new("RGB", (4, 4), (0, 255, 0))
    with caplog.at_level("WARNING"):
        result = pipeline.identify(query_img)

    record_ids = [m.record.record_id for m in result.matches]
    assert "met:B" not in record_ids
    assert any("met:B" in message for message in caplog.messages)
