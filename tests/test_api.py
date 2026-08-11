"""Tests for deco.api.app: POST /identify and GET /healthz."""

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from deco.api.app import create_app
from deco.api.pipeline import IdentifyPipeline, IdentifyResult, Match
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


def _record() -> Record:
    return Record(
        source="met",
        source_object_id="B",
        title="Title B",
        artist="Artist",
        date="1900",
        medium="Oil on canvas",
        accession_number="acc-B",
        image_url="https://example.com/B.jpg",
        license="CC0",
    )


def _cfg(min_cosine: float = 0.0, min_inliers: int = 0) -> DecoConfig:
    return DecoConfig(
        gate=GateConfig(min_cosine=min_cosine, min_inliers=min_inliers),
        retrieval=RetrievalConfig(top_k=3, rerank_top_n=3),
        ingest=IngestConfig(rate_limit_rps=1.0, max_backoff_s=1.0),
        embedding=EmbeddingConfig(model_version="fake-v1"),
        paths=PathsConfig(image_cache_dir="/tmp/deco-cache"),
    )


def _real_pipeline(min_cosine: float, min_inliers: int, scores: dict[str, int]) -> IdentifyPipeline:
    store = InMemoryRecordStore()
    record = _record()
    store.upsert(record)

    index = InMemoryVectorIndex()
    embedder = FakeEmbedder(dim=8, model_version="fake-v1")
    fixture_img = Image.new("RGB", (4, 4), (0, 255, 0))
    index.add(record.record_id, embedder.embed(fixture_img), embedder.model_version)

    def image_loader(record_id: str) -> Image.Image | None:
        return fixture_img if record_id == record.record_id else None

    reranker = FakeReranker(scores=scores)
    cfg = _cfg(min_cosine=min_cosine, min_inliers=min_inliers)
    return IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        image_loader=image_loader,
        reranker=reranker,
        cfg=cfg,
    )


class _StubPipeline:
    """Fake pipeline returning a fixed IdentifyResult regardless of input image."""

    def __init__(self, result: IdentifyResult) -> None:
        self._result = result

    def identify(self, image: Image.Image) -> IdentifyResult:
        return self._result


def _png_bytes(color: tuple[int, int, int] = (0, 255, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


def test_healthz():
    app = create_app(_real_pipeline(0.0, 0, {"met:B": 50}))
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_identify_multipart_confident_returns_match():
    pipeline = _real_pipeline(min_cosine=0.0, min_inliers=30, scores={"met:B": 50})
    app = create_app(pipeline)
    client = TestClient(app)

    response = client.post("/identify", files={"image": ("query.png", _png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["confident"] is True
    assert body["no_confident_match"] is False
    assert len(body["matches"]) == 1
    match = body["matches"][0]
    assert match["record"]["source_object_id"] == "B"
    assert match["inliers"] == 50
    assert isinstance(match["score"], float)


def test_identify_not_confident_returns_empty_matches():
    record = Record(
        source="met",
        source_object_id="B",
        title="Title B",
        artist="Artist",
        date="1900",
        medium="Oil on canvas",
        accession_number="acc-B",
        image_url="https://example.com/B.jpg",
        license="CC0",
    )
    not_confident = IdentifyResult(
        confident=False, matches=[Match(record=record, cosine=0.9, inliers=0)]
    )
    app = create_app(_StubPipeline(not_confident))
    client = TestClient(app)

    response = client.post("/identify", files={"image": ("query.png", _png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["confident"] is False
    assert body["no_confident_match"] is True
    assert body["matches"] == []


def test_identify_json_b64_body_works():
    pipeline = _real_pipeline(min_cosine=0.0, min_inliers=30, scores={"met:B": 50})
    app = create_app(pipeline)
    client = TestClient(app)

    b64 = base64.b64encode(_png_bytes()).decode("ascii")
    response = client.post("/identify", json={"image_b64": b64})

    assert response.status_code == 200
    body = response.json()
    assert body["confident"] is True
    assert body["matches"][0]["record"]["source_object_id"] == "B"


def test_identify_garbage_bytes_returns_422():
    pipeline = _real_pipeline(min_cosine=0.0, min_inliers=30, scores={"met:B": 50})
    app = create_app(pipeline)
    client = TestClient(app)

    response = client.post(
        "/identify", files={"image": ("query.png", b"not-an-image", "image/png")}
    )

    assert response.status_code == 422
    assert "detail" in response.json()


def test_identify_missing_image_returns_422():
    pipeline = _real_pipeline(min_cosine=0.0, min_inliers=30, scores={"met:B": 50})
    app = create_app(pipeline)
    client = TestClient(app)

    response = client.post("/identify", json={})

    assert response.status_code == 422
