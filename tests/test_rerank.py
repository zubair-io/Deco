"""Tests for deco.rerank: geometric re-rank orchestration and FakeReranker."""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deco.index.base import Hit
from deco.rerank.base import FakeReranker, RankedCandidate, rerank

QUERY_IMG = Image.new("RGB", (4, 4))
CANDIDATE_IMG = Image.new("RGB", (4, 4))

HITS = [
    Hit(record_id="A", score=0.9),
    Hit(record_id="B", score=0.8),
    Hit(record_id="C", score=0.7),
]


def _images_for(ids: set[str]):
    def images(record_id: str) -> Image.Image | None:
        return CANDIDATE_IMG if record_id in ids else None

    return images


def test_rerank_sorts_by_inliers_then_cosine_and_respects_top_n():
    reranker = FakeReranker(scores={"A": 5, "B": 120, "C": 0})
    images = _images_for({"A", "B", "C"})

    result = rerank(QUERY_IMG, HITS, images, reranker, top_n=2)

    assert [c.record_id for c in result] == ["B", "A"]
    assert result[0] == RankedCandidate(record_id="B", cosine=0.8, inliers=120)
    assert result[1] == RankedCandidate(record_id="A", cosine=0.9, inliers=5)
    assert len(result) == 2


def test_rerank_missing_candidate_image_gets_zero_inliers_but_still_ranked():
    # Only "A" and "C" have images available; "B" is missing from the cache.
    reranker = FakeReranker(scores={"A": 0, "B": 999, "C": 0})
    images = _images_for({"A", "C"})

    result = rerank(QUERY_IMG, HITS, images, reranker, top_n=3)

    assert [c.record_id for c in result] == ["A", "B", "C"]
    ranked_by_id = {c.record_id: c for c in result}
    # B's scripted inliers are never applied because its image is missing.
    assert ranked_by_id["B"].inliers == 0
    assert ranked_by_id["B"].cosine == 0.8
    assert ranked_by_id["A"].inliers == 0
    assert ranked_by_id["C"].inliers == 0


def test_rerank_only_top_n_returned():
    reranker = FakeReranker(scores={"A": 1, "B": 1, "C": 1})
    images = _images_for({"A", "B", "C"})

    result = rerank(QUERY_IMG, HITS, images, reranker, top_n=1)

    assert len(result) == 1
    assert result[0].record_id == "A"  # tie on inliers, highest cosine wins


def test_truncate_keypoints_keeps_top_scores_aligned():
    # Needs the ml extra (cv2/onnxruntime) to import the module, but no model weights.
    aliked = pytest.importorskip("deco.rerank.aliked")

    kpts = np.arange(20, dtype=np.float32).reshape(10, 2)  # row i = [2i, 2i+1]
    descs = np.arange(30, dtype=np.float32).reshape(10, 3)  # row i = [3i, 3i+1, 3i+2]
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.05, 0.6, 0.4, 0.5], dtype=np.float32)

    out_kpts, out_descs = aliked._truncate_keypoints(kpts, descs, scores, max_keypoints=4)

    # Rows 1, 3, 5, 7 hold the four highest scores (0.9, 0.8, 0.7, 0.6).
    assert out_kpts.tolist() == kpts[[1, 3, 5, 7]].tolist()
    assert out_descs.tolist() == descs[[1, 3, 5, 7]].tolist()


def test_truncate_keypoints_without_scores_keeps_first_rows():
    aliked = pytest.importorskip("deco.rerank.aliked")

    kpts = np.arange(20, dtype=np.float32).reshape(10, 2)
    descs = np.arange(30, dtype=np.float32).reshape(10, 3)

    out_kpts, out_descs = aliked._truncate_keypoints(kpts, descs, None, max_keypoints=4)

    assert out_kpts.tolist() == kpts[:4].tolist()
    assert out_descs.tolist() == descs[:4].tolist()


def test_truncate_keypoints_noop_when_under_limit():
    aliked = pytest.importorskip("deco.rerank.aliked")

    kpts = np.arange(6, dtype=np.float32).reshape(3, 2)
    descs = np.arange(9, dtype=np.float32).reshape(3, 3)
    scores = np.array([0.3, 0.2, 0.1], dtype=np.float32)

    out_kpts, out_descs = aliked._truncate_keypoints(kpts, descs, scores, max_keypoints=4)

    assert out_kpts.tolist() == kpts.tolist()
    assert out_descs.tolist() == descs.tolist()


@pytest.mark.integration
def test_aliked_lightglue_reranker_matches_rotated_self_copy():
    aliked_path = os.environ.get("DECO_ALIKED_ONNX")
    lightglue_path = os.environ.get("DECO_LIGHTGLUE_ONNX")
    if not aliked_path or not lightglue_path:
        pytest.skip("DECO_ALIKED_ONNX / DECO_LIGHTGLUE_ONNX not set")

    from deco.rerank.aliked import AlikedLightGlueReranker

    reranker = AlikedLightGlueReranker(Path(aliked_path), Path(lightglue_path))

    # Synthetic textured image (random noise gives ALIKED plenty of corners to
    # detect) matched against a rotated copy of itself.
    rng = np.random.default_rng(seed=0)
    array = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    query = Image.fromarray(array, mode="RGB")
    rotated = query.rotate(15, expand=True)

    result = reranker.match(query, rotated, "self")

    # Export behavior varies across ONNX builds; keep the assertion modest.
    assert result.inliers > 0
