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
