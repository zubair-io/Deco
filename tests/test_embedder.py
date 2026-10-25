"""Tests for deco.embed.embedder: Embedder protocol, l2_normalize, FakeEmbedder."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deco.embed.embedder import FakeEmbedder, l2_normalize


def test_l2_normalize_returns_unit_norm():
    v = np.array([3.0, 4.0], dtype=np.float32)
    normalized = l2_normalize(v)
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_l2_normalize_zero_vector_returns_unchanged():
    v = np.zeros(4, dtype=np.float32)
    normalized = l2_normalize(v)
    assert np.array_equal(normalized, v)


def _make_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (16, 16), color)


def test_fake_embedder_is_deterministic_per_image():
    embedder = FakeEmbedder()
    v1 = embedder.embed(_make_image((10, 20, 30)))
    v2 = embedder.embed(_make_image((10, 20, 30)))
    assert np.array_equal(v1, v2)


def test_fake_embedder_differs_for_different_images():
    embedder = FakeEmbedder()
    v1 = embedder.embed(_make_image((10, 20, 30)))
    v2 = embedder.embed(_make_image((200, 100, 50)))
    assert not np.array_equal(v1, v2)


def test_fake_embedder_is_unit_norm_and_has_requested_dim():
    embedder = FakeEmbedder(dim=16)
    v = embedder.embed(_make_image((1, 2, 3)))
    assert v.shape == (16,)
    assert v.dtype == np.float32
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_fake_embedder_default_model_version():
    embedder = FakeEmbedder()
    assert embedder.model_version == "fake-v1"


def test_fake_embedder_custom_model_version():
    embedder = FakeEmbedder(model_version="fake-v2")
    assert embedder.model_version == "fake-v2"


@pytest.mark.integration
def test_dinov2_onnx_embedder_embeds_image():
    model_path = os.environ.get("DECO_DINOV2_ONNX")
    if not model_path:
        pytest.skip("DECO_DINOV2_ONNX not set")

    from deco.embed.onnx_dinov2 import DinoV2OnnxEmbedder

    embedder = DinoV2OnnxEmbedder(Path(model_path), model_version="dinov2-vits14-onnx")
    image = _make_image((128, 64, 200))

    vector = embedder.embed(image)

    assert vector.ndim == 1
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert embedder.model_version == "dinov2-vits14-onnx"
