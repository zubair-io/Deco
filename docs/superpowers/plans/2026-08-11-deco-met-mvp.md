# Deco — Met MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reverse-image artwork identification: ingest The Met Open Access corpus, embed it, and serve `POST /identify` that matches an arbitrary photo back to a canonical catalog record or returns "no confident match."

**Architecture:** A pipeline of narrow interfaces — `IngestAdapter` → `RecordStore` (MongoDB) + image cache → `Embedder` (DINOv2/ONNX) → `VectorIndex` (Meilisearch, swappable) → `Reranker` (ALIKED+LightGlue+RANSAC) → confidence gate → FastAPI. Every heavy/external dependency (Mongo, Meili, ONNX, GPU) sits behind a Protocol with an in-memory/fake implementation, so the entire pipeline is unit-testable in CI with no services and no model weights. Real backends are integration-tested behind pytest markers.

**Tech Stack:** Python 3.12, uv, ruff, pytest, pydantic, httpx, numpy, Pillow, FastAPI, pymongo, meilisearch, onnxruntime, opencv-python-headless. GitHub Actions CI.

## Global Constraints

- Repo: `github.com/zubair-io/Deco`. Tracking issues #1–#16, milestones M0 → Phase 4.
- Record schema fields (verbatim from spec): `source`, `source_object_id`, `title`, `artist`, `date`, `medium`, `accession_number`, `image_url`, `license`, `wikidata_id` (nullable), `dedupe_key` (nullable). For Met: `source="met"`, `license="CC0"`.
- Only rows with `Is Public Domain = True` and a resolvable primary image become records.
- Multi-collection by construction: adding a collection = one new `IngestAdapter` implementation. No schema migration.
- Vector retrieval is behind a backend-agnostic interface: swapping Meilisearch → FAISS/Qdrant (P1-2) must be a backend change, not a caller change.
- Confidence-gate thresholds live in `config/deco.toml`, never hardcoded.
- Embeddings are L2-normalized; every stored vector records its `model_version`.
- Ingest is resumable and idempotent; image bytes are fetched once and cached (TrueNAS path is just a configured directory).
- Rate limiting: respect Met API limits, exponential backoff on 429/5xx.
- Targets (measured by the eval harness, Task 10): top-1 ≥85%, top-5 ≥95%, out-of-corpus FP ≤5%, p95 <2s on `phoebe`.
- CI must pass with no GPU, no network, no Mongo/Meili: unit tests use fakes; integration tests are marked `@pytest.mark.integration` and skipped by default.
- Package layout: `src/deco/`. Lint: `ruff check` + `ruff format --check`. Tests: `pytest`.
- Commit style: conventional commits (`feat:`, `test:`, `chore:`). Commit after every green task.

---

### Task 1: Repo scaffold (issue #1, M0)

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md` (replace stub), `config/deco.toml`, `src/deco/__init__.py`, `src/deco/config.py`, `tests/__init__.py`, `tests/test_config.py`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `deco.config.load_config(path: Path | None = None) -> DecoConfig` where `DecoConfig` has `.gate.min_cosine: float`, `.gate.min_inliers: int`, `.retrieval.top_k: int`, `.retrieval.rerank_top_n: int`, `.ingest.rate_limit_rps: float`, `.ingest.max_backoff_s: float`, `.embedding.model_version: str`, `.paths.image_cache_dir: Path`. Default config path: `config/deco.toml` at repo root, overridable via env `DECO_CONFIG`.

- [ ] **Step 1: `pyproject.toml`**

```toml
[project]
name = "deco"
version = "0.1.0"
description = "Artwork identification service — reverse-image lookup over open-access museum corpora"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "httpx>=0.27",
    "numpy>=1.26",
    "pillow>=10.3",
]

[project.optional-dependencies]
server = ["fastapi>=0.111", "uvicorn>=0.30", "python-multipart>=0.0.9"]
mongo = ["pymongo>=4.7"]
meili = ["meilisearch>=0.31"]
ml = ["onnxruntime>=1.18", "opencv-python-headless>=4.10"]

[dependency-groups]
dev = ["pytest>=8.2", "ruff>=0.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/deco"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: needs live Mongo/Meilisearch/model weights; skipped in CI"]
addopts = "-m 'not integration'"
```

- [ ] **Step 2: `config/deco.toml`**

```toml
[gate]
min_cosine = 0.55
min_inliers = 30

[retrieval]
top_k = 50
rerank_top_n = 5

[ingest]
rate_limit_rps = 2.0
max_backoff_s = 60.0

[embedding]
model_version = "dinov2-vitl14-onnx-v1"

[paths]
image_cache_dir = "/mnt/truenas/deco/images"
```

- [ ] **Step 3: failing test `tests/test_config.py`**

```python
from pathlib import Path

from deco.config import load_config


def test_load_config_from_repo_default():
    cfg = load_config()
    assert cfg.gate.min_cosine == 0.55
    assert cfg.gate.min_inliers == 30
    assert cfg.retrieval.top_k == 50
    assert cfg.embedding.model_version.startswith("dinov2")


def test_load_config_explicit_path(tmp_path: Path):
    p = tmp_path / "deco.toml"
    p.write_text(
        "[gate]\nmin_cosine = 0.9\nmin_inliers = 5\n"
        "[retrieval]\ntop_k = 10\nrerank_top_n = 3\n"
        "[ingest]\nrate_limit_rps = 1.0\nmax_backoff_s = 10.0\n"
        '[embedding]\nmodel_version = "test-v0"\n'
        '[paths]\nimage_cache_dir = "/tmp/imgs"\n'
    )
    cfg = load_config(p)
    assert cfg.gate.min_cosine == 0.9
    assert cfg.paths.image_cache_dir == Path("/tmp/imgs")
```

- [ ] **Step 4: run `uv run pytest tests/test_config.py -v` → FAIL (no module `deco.config`)**

- [ ] **Step 5: implement `src/deco/config.py`**

```python
"""Deco configuration. Thresholds are config, never hardcoded (P0-6)."""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel


class GateConfig(BaseModel):
    min_cosine: float
    min_inliers: int


class RetrievalConfig(BaseModel):
    top_k: int
    rerank_top_n: int


class IngestConfig(BaseModel):
    rate_limit_rps: float
    max_backoff_s: float


class EmbeddingConfig(BaseModel):
    model_version: str


class PathsConfig(BaseModel):
    image_cache_dir: Path


class DecoConfig(BaseModel):
    gate: GateConfig
    retrieval: RetrievalConfig
    ingest: IngestConfig
    embedding: EmbeddingConfig
    paths: PathsConfig


def _default_config_path() -> Path:
    if env := os.environ.get("DECO_CONFIG"):
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config" / "deco.toml"


def load_config(path: Path | None = None) -> DecoConfig:
    with open(path or _default_config_path(), "rb") as f:
        return DecoConfig.model_validate(tomllib.load(f))
```

- [ ] **Step 6: `.github/workflows/ci.yml`**

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --all-extras --dev
      - run: uv run ruff check
      - run: uv run ruff format --check
      - run: uv run pytest -v
```

- [ ] **Step 7: `.gitignore` (standard Python: `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `dist/`, `*.onnx`, `data/`), README describing the project + architecture diagram from the spec**

- [ ] **Step 8: `uv sync --all-extras --dev`, run `uv run pytest -v` → PASS; `uv run ruff check` → clean**

- [ ] **Step 9: Commit `feat: scaffold deco package, config, and CI (closes #1)`**

---

### Task 2: Record model + RecordStore

**Files:**
- Create: `src/deco/records.py`, `src/deco/stores/__init__.py`, `src/deco/stores/memory.py`, `src/deco/stores/mongo.py`, `tests/test_records.py`

**Interfaces:**
- Produces:
  - `deco.records.Record` — pydantic model, fields exactly: `source: str`, `source_object_id: str`, `title: str`, `artist: str`, `date: str`, `medium: str`, `accession_number: str`, `image_url: str | None`, `license: str`, `wikidata_id: str | None = None`, `dedupe_key: str | None = None`; property `record_id -> str` returning `f"{source}:{source_object_id}"`.
  - `deco.records.RecordStore` — `typing.Protocol` with `upsert(record: Record) -> None`, `get(record_id: str) -> Record | None`, `count() -> int`, `iter_records() -> Iterator[Record]`, `set_image_url(record_id: str, url: str) -> None`.
  - `deco.stores.memory.InMemoryRecordStore` — dict-backed, used by all unit tests.
  - `deco.stores.mongo.MongoRecordStore(collection)` — same contract over a pymongo collection, `_id = record_id`; `upsert` is a `replace_one(..., upsert=True)` so re-ingest is idempotent (resumability). Integration-marked test only.

- [ ] **Step 1: failing test `tests/test_records.py`**

```python
from deco.records import Record
from deco.stores.memory import InMemoryRecordStore


def make_record(oid: str = "436535") -> Record:
    return Record(
        source="met",
        source_object_id=oid,
        title="Wheat Field with Cypresses",
        artist="Vincent van Gogh",
        date="1889",
        medium="Oil on canvas",
        accession_number="93.1.2",
        image_url=None,
        license="CC0",
    )


def test_record_id():
    assert make_record().record_id == "met:436535"


def test_upsert_is_idempotent():
    store = InMemoryRecordStore()
    store.upsert(make_record())
    store.upsert(make_record())
    assert store.count() == 1


def test_get_and_iter():
    store = InMemoryRecordStore()
    store.upsert(make_record("1"))
    store.upsert(make_record("2"))
    assert store.get("met:1").source_object_id == "1"
    assert store.get("met:missing") is None
    assert {r.record_id for r in store.iter_records()} == {"met:1", "met:2"}


def test_set_image_url():
    store = InMemoryRecordStore()
    store.upsert(make_record())
    store.set_image_url("met:436535", "https://images.metmuseum.org/x.jpg")
    assert store.get("met:436535").image_url == "https://images.metmuseum.org/x.jpg"
```

- [ ] **Step 2: run → FAIL. Step 3: implement `records.py` (pydantic model + Protocol), `stores/memory.py`. Step 4: run → PASS.**

- [ ] **Step 5: implement `stores/mongo.py` with the same five methods over pymongo (`replace_one({"_id": rid}, doc, upsert=True)`; `set_image_url` = `update_one` on `image_url`). Guard the pymongo import so the module errors helpfully if the `mongo` extra is missing. Add `@pytest.mark.integration` test that round-trips against `MONGO_URL` env var.**

- [ ] **Step 6: ruff + pytest → PASS. Commit `feat: record schema and record stores (in-memory + mongo)`**

---

### Task 3: IngestAdapter interface + Met CSV adapter (issue #5, part 1)

**Files:**
- Create: `src/deco/ingest/__init__.py`, `src/deco/ingest/adapter.py`, `src/deco/ingest/met.py`, `tests/test_met_adapter.py`, `tests/fixtures/met_sample.csv`

**Interfaces:**
- Consumes: `Record` from Task 2.
- Produces:
  - `deco.ingest.adapter.IngestAdapter` — `Protocol`: attribute `source: str`; `records() -> Iterator[Record]` (metadata only; `image_url` may be `None` at this stage); `resolve_image_url(source_object_id: str) -> str | None` (network call; returns primary image URL or None).
  - `deco.ingest.met.MetAdapter(csv_path: Path, client: httpx.Client)` implementing it. `records()` streams the CSV (it is ~300MB — never load it whole), yields only rows where `Is Public Domain` is `True`, mapping columns: `Object ID`→`source_object_id`, `Title`→`title`, `Artist Display Name`→`artist`, `Object Date`→`date`, `Medium`→`medium`, `Object Number`→`accession_number`; `source="met"`, `license="CC0"`, `wikidata_id` from `Wikidata URL` column if present else None. `resolve_image_url` GETs `https://collectionapi.metmuseum.org/public/collection/v1/objects/{id}` and returns `primaryImage` or None if empty.

- [ ] **Step 1: fixture `tests/fixtures/met_sample.csv`** — header row with the real Met column names (`Object Number,Is Highlight,Is Timeline Work,Is Public Domain,Object ID,...,Title,...,Artist Display Name,...,Object Date,Medium,...`) plus 3 rows: two public-domain, one not.

- [ ] **Step 2: failing test `tests/test_met_adapter.py`**

```python
from pathlib import Path

import httpx

from deco.ingest.met import MetAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "met_sample.csv"


def test_records_filters_public_domain():
    adapter = MetAdapter(FIXTURE, client=httpx.Client())
    records = list(adapter.records())
    assert len(records) == 2  # third row is Is Public Domain=False
    assert all(r.source == "met" and r.license == "CC0" for r in records)


def test_resolve_image_url_uses_object_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/objects/436535")
        return httpx.Response(200, json={"primaryImage": "https://images.metmuseum.org/a.jpg"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = MetAdapter(FIXTURE, client=client)
    assert adapter.resolve_image_url("436535") == "https://images.metmuseum.org/a.jpg"


def test_resolve_image_url_none_when_empty():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"primaryImage": ""}))
    )
    assert MetAdapter(FIXTURE, client=client).resolve_image_url("1") is None
```

- [ ] **Step 3: run → FAIL. Step 4: implement (use `csv.DictReader` streaming). Step 5: run → PASS, ruff clean. Step 6: Commit `feat: ingest adapter contract + Met CSV adapter`**

---

### Task 4: Rate-limited fetcher, image cache, resumable ingest runner (issue #5, part 2)

**Files:**
- Create: `src/deco/ingest/fetch.py`, `src/deco/ingest/run.py`, `tests/test_fetch.py`, `tests/test_ingest_run.py`

**Interfaces:**
- Consumes: `IngestAdapter`, `RecordStore`, `IngestConfig`.
- Produces:
  - `deco.ingest.fetch.RateLimiter(rps: float, sleep=time.sleep, clock=time.monotonic)` — `.wait()` blocks to enforce max `rps` calls/sec (injectable clock/sleep for tests).
  - `deco.ingest.fetch.fetch_with_backoff(client, url, *, limiter, max_backoff_s, retries=6) -> httpx.Response` — retries 429/5xx with exponential backoff (1s, 2s, 4s… capped at `max_backoff_s`), raises `httpx.HTTPStatusError` after `retries`.
  - `deco.ingest.fetch.ImageCache(root: Path)` — `path_for(record_id) -> Path` (`root/<source>/<source_object_id>.jpg`), `has(record_id) -> bool`, `put(record_id, data: bytes) -> Path` (atomic write via temp file + rename).
  - `deco.ingest.run.run_ingest(adapter, store, cache, client, cfg: IngestConfig) -> IngestStats` — for each adapter record: upsert metadata; if `image_url` missing, `resolve_image_url` (rate-limited); skip records with no image (spec: only rows with a resolvable primary image are kept — delete/skip, count as `no_image`); if `cache.has(record_id)` skip download (never re-hit the API for an image we have); else fetch bytes with backoff and `cache.put`. Returns `IngestStats(records=int, images_fetched=int, images_cached=int, no_image=int)`. Resumability falls out of idempotent upsert + cache-hit skip: rerunning continues where it left off.

- [ ] **Step 1: failing tests `tests/test_fetch.py`** — RateLimiter: with a fake clock, 3 `.wait()` calls at rps=2 sleep a total of ≥1.0s. Backoff: MockTransport returns 429, 500, then 200 → result is the 200 and recorded sleeps are [1, 2]; returns 429 seven times → raises. ImageCache: `put` then `has` true, `path_for` layout correct, file bytes round-trip.

- [ ] **Step 2: failing test `tests/test_ingest_run.py`** — build `MetAdapter` over the fixture CSV with a MockTransport that serves object JSON + image bytes; run `run_ingest` twice into `InMemoryRecordStore` + `ImageCache(tmp_path)`; assert second run fetches 0 images (`images_cached == records`), store count unchanged, and a record whose API `primaryImage` is empty is not in the store.

- [ ] **Step 3: run → FAIL. Step 4: implement. Step 5: run → PASS, ruff clean. Step 6: add CLI entry `python -m deco.ingest.run --source met --csv <path>` (argparse, wires real Mongo store + config paths; thin, untested glue). Step 7: Commit `feat: rate-limited resumable ingest with image cache (closes #5 scaffolding)`**

---

### Task 5: Embedder + embedding pass (issue #6)

**Files:**
- Create: `src/deco/embed/__init__.py`, `src/deco/embed/embedder.py`, `src/deco/embed/onnx_dinov2.py`, `src/deco/embed/run.py`, `tests/test_embedder.py`, `tests/test_embed_run.py`

**Interfaces:**
- Consumes: `RecordStore`, `ImageCache`, `EmbeddingConfig`, `VectorIndex` (Task 6 — see Produces there; to avoid a hard dependency ordering, this task defines the `VectorSink` it needs and Task 6's index satisfies it structurally).
- Produces:
  - `deco.embed.embedder.Embedder` — `Protocol`: `embed(image: PIL.Image.Image) -> np.ndarray` (1-D float32), `model_version: str`.
  - `deco.embed.embedder.l2_normalize(v: np.ndarray) -> np.ndarray`.
  - `deco.embed.embedder.FakeEmbedder(dim: int = 8, model_version: str = "fake-v1")` — deterministic: hashes image bytes into a seeded RNG vector, L2-normalized. Used by all downstream tests.
  - `deco.embed.onnx_dinov2.DinoV2OnnxEmbedder(model_path: Path, model_version: str)` — loads an ONNX session (CoreML/CUDA EP if available, CPU fallback), resizes/center-crops to 224, ImageNet-normalizes, runs, mean-pools patch tokens if the export emits them, L2-normalizes. Import-guarded on the `ml` extra; integration-marked test only.
  - `deco.embed.run.run_embed_pass(store, cache, embedder, sink: VectorSink) -> EmbedStats` where `VectorSink` is a `Protocol` with `add(record_id: str, vector: np.ndarray, model_version: str) -> None` and `has(record_id: str, model_version: str) -> bool`. Skips records already embedded under this `model_version` (resumable); a new `model_version` re-embeds everything without touching records (spec P0-2).

- [ ] **Step 1: failing test `tests/test_embedder.py`** — `l2_normalize` returns unit norm and handles the zero vector (returns it unchanged); `FakeEmbedder` is deterministic per image and unit-norm.

- [ ] **Step 2: failing test `tests/test_embed_run.py`** — store with 2 records + cached images, in-memory sink (a dict in the test implementing `VectorSink`); run pass twice: second run adds 0 vectors under same version; changing `model_version` re-adds both.

- [ ] **Step 3: run → FAIL. Step 4: implement (including `onnx_dinov2.py`, exercised only by an integration-marked test gated on `DECO_DINOV2_ONNX` env pointing at a local export). Step 5: run → PASS. Step 6: CLI `python -m deco.embed.run`. Step 7: Commit `feat: embedder contract, fake + DINOv2 ONNX embedders, resumable embed pass (closes #6)`**

---

### Task 6: VectorIndex + retrieval (issue #7)

**Files:**
- Create: `src/deco/index/__init__.py`, `src/deco/index/base.py`, `src/deco/index/memory.py`, `src/deco/index/meili.py`, `tests/test_index.py`

**Interfaces:**
- Consumes: nothing beyond numpy.
- Produces:
  - `deco.index.base.VectorIndex` — `Protocol`: `add(record_id: str, vector: np.ndarray, model_version: str) -> None`, `has(record_id: str, model_version: str) -> bool`, `search(vector: np.ndarray, k: int) -> list[Hit]` with `Hit(record_id: str, score: float)` (cosine, descending). Note this Protocol is a superset of Task 5's `VectorSink` — one object serves both.
  - `deco.index.memory.InMemoryVectorIndex` — numpy matrix, exact cosine top-K. Doubles as the test fake and a small-corpus dev backend.
  - `deco.index.meili.MeiliVectorIndex(index, embedder_name="deco")` — Meilisearch userProvided-vectors backend, same contract. Import-guarded on the `meili` extra; integration-marked test against `MEILI_URL`.

- [ ] **Step 1: failing test `tests/test_index.py`** — add 3 unit vectors; `search(query, k=2)` returns the 2 nearest by cosine in order with scores in [−1, 1]; `has` respects model_version; re-`add` of same record_id replaces, not duplicates.

- [ ] **Step 2: run → FAIL. Step 3: implement memory backend, then meili backend mirroring it. Step 4: run → PASS, ruff clean. Step 5: Commit `feat: vector index contract with in-memory and Meilisearch backends (closes #7)`**

---

### Task 7: Geometric re-rank (issue #9)

**Files:**
- Create: `src/deco/rerank/__init__.py`, `src/deco/rerank/base.py`, `src/deco/rerank/aliked.py`, `tests/test_rerank.py`

**Interfaces:**
- Consumes: `ImageCache` (candidate images), `RetrievalConfig`.
- Produces:
  - `deco.rerank.base.MatchResult` — dataclass: `inliers: int`, `total_matches: int`.
  - `deco.rerank.base.Reranker` — `Protocol`: `match(query: PIL.Image.Image, candidate: PIL.Image.Image) -> MatchResult`.
  - `deco.rerank.base.rerank(query_img, hits: list[Hit], images: Callable[[str], PIL.Image.Image | None], reranker, top_n: int) -> list[RankedCandidate]` — pure orchestration: run `reranker.match` on each hit (missing image → inliers 0), sort by `(inliers, cosine)` descending, return top_n `RankedCandidate(record_id, cosine, inliers)`.
  - `deco.rerank.base.FakeReranker(scores: dict[str, int])` — returns scripted inlier counts keyed by a candidate id the test controls (implemented as a callable wrapper the orchestration passes the record_id to; concretely `rerank` calls `reranker.match(query, candidate_img)` — the fake identifies candidates by image pixel tag the test bakes in, or more simply `FakeReranker` takes `match_fn: Callable[[str], int]` and `rerank` passes record_id via a `match_for(record_id, query, img)` optional protocol method — keep it simple: make the Protocol `match(query, candidate, candidate_id: str) -> MatchResult` so fakes can script by id).
  - `deco.rerank.aliked.AlikedLightGlueReranker(aliked_path: Path, lightglue_path: Path)` — ONNX ALIKED keypoints+descriptors on both images, LightGlue matching, `cv2.findHomography(..., cv2.RANSAC)` → inlier count. Import-guarded on `ml` extra; integration-marked test gated on `DECO_ALIKED_ONNX`/`DECO_LIGHTGLUE_ONNX` env vars. (Reuses the Maple pano-stitching model exports.)

- [ ] **Step 1: failing test `tests/test_rerank.py`** — given hits `[A(cos .9), B(cos .8), C(cos .7)]` and scripted inliers `{A: 5, B: 120, C: 0}`, `rerank(..., top_n=2)` returns `[B, A]`; missing candidate image → inliers 0 but candidate still ranked (by cosine among zero-inlier peers); only `top_n` returned.

- [ ] **Step 2: run → FAIL. Step 3: implement base + aliked. Step 4: run → PASS. Step 5: Commit `feat: geometric re-rank contract with ALIKED+LightGlue backend (closes #9)`**

---

### Task 8: Confidence gate (issue #10)

**Files:**
- Create: `src/deco/gate.py`, `tests/test_gate.py`

**Interfaces:**
- Consumes: `GateConfig` (Task 1), `RankedCandidate` (Task 7).
- Produces: `deco.gate.apply_gate(candidates: list[RankedCandidate], cfg: GateConfig) -> GateDecision` where `GateDecision` is a dataclass: `confident: bool`, `candidates: list[RankedCandidate]`. A result is confident iff the top candidate has `cosine >= cfg.min_cosine` **and** `inliers >= cfg.min_inliers`. Not confident → `confident=False` and candidates still returned (the API decides to hide/show them; the flag is authoritative).

- [ ] **Step 1: failing test `tests/test_gate.py`** — top candidate (cos .8, inliers 100) with cfg (.55, 30) → confident; (cos .8, inliers 10) → not confident; (cos .4, inliers 100) → not confident; empty candidate list → not confident.

- [ ] **Step 2: run → FAIL. Step 3: implement (pure function, ~15 lines). Step 4: PASS. Step 5: Commit `feat: confidence gate (closes #10)`**

---

### Task 9: Query service — `POST /identify` (issue #8)

**Files:**
- Create: `src/deco/api/__init__.py`, `src/deco/api/pipeline.py`, `src/deco/api/app.py`, `src/deco/api/serve.py`, `tests/test_pipeline.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `deco.api.pipeline.IdentifyPipeline(embedder, index, store, image_loader: Callable[[str], PIL.Image | None], reranker, cfg: DecoConfig)` with `identify(image: PIL.Image.Image) -> IdentifyResult`; `IdentifyResult`: `confident: bool`, `matches: list[Match]`; `Match`: `record: Record`, `cosine: float`, `inliers: int`. Flow: embed → `index.search(top_k)` → `rerank(top_n)` → `apply_gate` → join records from store.
  - `deco.api.app.create_app(pipeline) -> FastAPI` — `POST /identify` accepting either multipart file field `image` or JSON `{"image_b64": ...}`; response `{"confident": bool, "no_confident_match": bool, "matches": [{"record": {...}, "score": float, "inliers": int}]}` where `no_confident_match = not confident` and `matches` is `[]` when not confident (spec: no low-confidence wrong record). `GET /healthz` → `{"status": "ok"}`.
  - `deco.api.serve.main()` — wires real config, Mongo store, Meili index, ONNX embedder + reranker, uvicorn. Thin untested glue, `python -m deco.api.serve`.

- [ ] **Step 1: failing test `tests/test_pipeline.py`** — build a 3-record corpus with `InMemoryRecordStore` + `FakeEmbedder` + `InMemoryVectorIndex` (index each record's cached fixture image); scripted `FakeReranker` makes record B the geometric winner; `identify(query_img)` → confident, `matches[0].record.record_id == "met:B"`. Second test: reranker scripts all inliers 0 → `confident is False`.

- [ ] **Step 2: failing test `tests/test_api.py`** — `fastapi.testclient.TestClient` over `create_app` with the same fake pipeline components: multipart POST of a PNG → 200, `confident=True`, one match with record fields present; not-confident pipeline → `no_confident_match=True`, `matches == []`; b64 JSON body works; garbage bytes → 422.

- [ ] **Step 3: run → FAIL. Step 4: implement. Step 5: run → PASS, ruff clean. Step 6: Commit `feat: identify pipeline and POST /identify API (closes #8)`**

---

### Task 10: Eval harness (issue #4, harness half)

**Files:**
- Create: `src/deco/eval/__init__.py`, `src/deco/eval/harness.py`, `src/deco/eval/run.py`, `tests/test_eval.py`, `docs/eval-set.md`

**Interfaces:**
- Consumes: `IdentifyPipeline`.
- Produces:
  - Manifest format (documented in `docs/eval-set.md`): JSONL, one line per query photo: `{"photo": "photos/img_0042.jpg", "ground_truth": "met:436535", "condition": "oblique"}`; out-of-corpus negatives use `"ground_truth": null`.
  - `deco.eval.harness.evaluate(pipeline, cases: list[EvalCase], load_image) -> EvalReport` — `EvalCase(photo: str, ground_truth: str | None, condition: str | None)`; `EvalReport`: `top1: float`, `top5: float`, `false_positive_rate: float` (fraction of `ground_truth=None` cases returning `confident=True`), `p95_latency_s: float`, `n_in_corpus: int`, `n_out_of_corpus: int`, plus per-condition top1 breakdown `by_condition: dict[str, float]`. Latency measured around `pipeline.identify` with an injectable clock.
  - `deco.eval.run.main()` — CLI: `python -m deco.eval.run --manifest eval/manifest.jsonl` printing the report as a table + JSON; exits non-zero if top1 < 0.85 or top5 < 0.95 or FP > 0.05 (the spec's bars), overridable with `--no-gate`.

- [ ] **Step 1: failing test `tests/test_eval.py`** — a stub pipeline scripted per photo name (photo1→correct@1, photo2→correct@3, photo3 in-corpus→miss, neg1→not confident, neg2→confident wrong): report has `top1 == 1/3`, `top5 == 2/3`, `false_positive_rate == 0.5`, latencies from fake clock give a sane p95, condition breakdown groups correctly.

- [ ] **Step 2: run → FAIL. Step 3: implement. Step 4: PASS. Step 5: write `docs/eval-set.md` — manifest format, directory layout (`eval/photos/`, `eval/manifest.jsonl`), shooting protocol (per work: clean frontal, ~30° oblique, tight crop, glare, screen re-photo, print re-photo; 40–80 works ⇒ 240–480 photos; plus ≥50 out-of-corpus negatives). Step 6: Commit `feat: eval harness and eval-set spec (closes #4 harness)`**

---

### Spike A (issue #2, research-only): Bulk image acquisition
Deliverable `docs/spikes/bulk-image-acquisition.md`: evaluate per-object Collection API crawl vs. any bulk path (IIIF, mirrors, dataset dumps — verify against current Met docs/policies, cite sources with dates); wall-clock estimate for ~470K objects at the confirmed rate limit; recommendation. **Blocks final P0-1 sizing but not the Task 3/4 code, whose interfaces absorb either answer.**

### Spike B (issue #3, research-only): Met API rate limits
Deliverable `docs/spikes/met-rate-limits.md`: current documented limits (met docs state ~80 req/s ceiling historically — verify), whether a key changes them, recommended `rate_limit_rps` + backoff for `config/deco.toml`, plus image-CDN etiquette. Feeds the `[ingest]` config values.

---

## Self-Review (done at planning time)

- **Spec coverage:** P0-1→Tasks 3+4 (+Spikes A/B), P0-2→Task 5, P0-3→Task 6, P0-4→Task 9, P0-5→Task 7, P0-6→Tasks 1+8, M0→Task 1, eval metrics→Task 10. P1/P2 tracked as GitHub issues #11–#16, deliberately unplanned (YAGNI).
- **Type consistency:** `Hit(record_id, score)` produced by Task 6, consumed by Task 7; `RankedCandidate(record_id, cosine, inliers)` produced by Task 7, consumed by Tasks 8/9; `VectorSink` (Task 5) is a structural subset of `VectorIndex` (Task 6) — one object serves both. `Reranker.match(query, candidate, candidate_id)` includes `candidate_id` so fakes can script by id.
- **Latency/accuracy targets** are eval-harness-enforced (Task 10 CLI gates), not asserted in unit tests — they need real hardware + eval set.
