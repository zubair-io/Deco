# Deco

Artwork identification service — reverse-image lookup over open-access museum corpora.

## Overview

Deco enables reverse-image search for artwork. Submit a photograph and receive catalog records from open-access museum collections, ranked by relevance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Image Upload                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                       Ingest Pipeline                                │
│  (Rate-limited image download and storage)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    Embedding (DINOv2/ONNX)                          │
│  (Generate feature vectors for image similarity)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│               Vector Index (Meilisearch)                             │
│  (Fast approximate nearest-neighbor search)                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│         Geometric Re-ranking (ALIKED + LightGlue)                    │
│  (Verify matches with keypoint-based geometric alignment)           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│              Confidence Gating                                       │
│  (Filter results by cosine similarity and inlier count)             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    FastAPI Server                                    │
│  (REST endpoints + catalog record lookup)                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Pipeline Components

- **Ingest**: Download and cache images from source URLs (rate-limited)
- **Embedding**: Generate vector representations using DINOv2 ONNX model
- **Index**: Store and retrieve similar items via Meilisearch vector search
- **Rerank**: Refine top candidates using ALIKED keypoint detector + LightGlue matcher
- **Gate**: Apply confidence thresholds (cosine similarity, inlier count)
- **API**: FastAPI server for HTTP access

## Configuration

All configuration is in `config/deco.toml`. Key settings:

- `gate.min_cosine`: Minimum cosine similarity to accept (0.55 default)
- `gate.min_inliers`: Minimum matched keypoints for re-rank pass (30 default)
- `retrieval.top_k`: Candidates to re-rank (50 default)
- `retrieval.rerank_top_n`: Final ranked results (5 default)
- `embedding.model_version`: DINOv2 model variant
- `ingest.rate_limit_rps`: Images per second (2.0 default)

Environment variable `DECO_CONFIG` overrides config path.

## Development

### Setup

```bash
uv sync --all-extras --dev
```

### Test

```bash
uv run pytest -v
```

### Format & Lint

```bash
uv run ruff format .
uv run ruff check
```

## Optional Dependencies

- `server`: FastAPI, Uvicorn, multipart support
- `meili`: Meilisearch Python client
- `mongo`: MongoDB driver
- `ml`: ONNX Runtime, OpenCV
