"""Embedder contract, l2 normalization, and a deterministic fake for tests."""

import hashlib
from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    model_version: str

    def embed(self, image: Image.Image) -> np.ndarray: ...


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return `v` scaled to unit L2 norm. A zero vector is returned unchanged."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class FakeEmbedder:
    """Deterministic embedder for tests: image bytes seed a vector, L2-normalized.

    Same image content always produces the same vector; no model weights needed.
    """

    def __init__(self, dim: int = 8, model_version: str = "fake-v1") -> None:
        self.dim = dim
        self.model_version = model_version

    def embed(self, image: Image.Image) -> np.ndarray:
        digest = hashlib.sha256(image.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim).astype(np.float32)
        return l2_normalize(vector)
