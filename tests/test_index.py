"""Tests for deco.index: VectorIndex contract, in-memory and Meilisearch backends."""

import os
import uuid

import numpy as np
import pytest

from deco.index.base import Hit
from deco.index.memory import InMemoryVectorIndex

# Three unit vectors in 2D: "a" and "b" are close together near 0 degrees,
# "c" points the opposite way. Query is aligned with "a".
VEC_A = np.array([1.0, 0.0], dtype=np.float32)
VEC_B = np.array([0.9961947, 0.0871557], dtype=np.float32)  # ~5 degrees from A
VEC_C = np.array([-1.0, 0.0], dtype=np.float32)  # 180 degrees from A


def _index_with_three_vectors() -> InMemoryVectorIndex:
    index = InMemoryVectorIndex()
    index.add("a", VEC_A, "v1")
    index.add("b", VEC_B, "v1")
    index.add("c", VEC_C, "v1")
    return index


def test_search_returns_top_k_nearest_by_cosine_in_order():
    index = _index_with_three_vectors()

    hits = index.search(VEC_A, k=2)

    assert [hit.record_id for hit in hits] == ["a", "b"]
    assert isinstance(hits[0], Hit)
    for hit in hits:
        assert -1.0 <= hit.score <= 1.0
    # "a" is an exact match for itself.
    assert hits[0].score > hits[1].score
    assert hits[0].score == 1.0


def test_search_returns_fewer_than_k_when_index_smaller_than_k():
    index = InMemoryVectorIndex()
    index.add("a", VEC_A, "v1")

    hits = index.search(VEC_A, k=5)

    assert len(hits) == 1
    assert hits[0].record_id == "a"


def test_search_on_empty_index_returns_no_hits():
    index = InMemoryVectorIndex()

    hits = index.search(VEC_A, k=2)

    assert hits == []


def test_has_respects_model_version():
    index = InMemoryVectorIndex()
    index.add("a", VEC_A, "v1")

    assert index.has("a", "v1") is True
    assert index.has("a", "v2") is False
    assert index.has("missing", "v1") is False


def test_readd_same_record_id_replaces_not_duplicates():
    index = InMemoryVectorIndex()
    index.add("a", VEC_A, "v1")
    index.add("a", VEC_C, "v2")

    # Old model_version no longer applies to this record.
    assert index.has("a", "v1") is False
    assert index.has("a", "v2") is True

    hits = index.search(VEC_C, k=5)

    # Only one hit for "a" — the replaced vector, not a duplicate entry.
    assert len(hits) == 1
    assert hits[0].record_id == "a"
    assert hits[0].score == 1.0


@pytest.mark.integration
def test_meili_vector_index_add_has_search():
    meili_url = os.environ.get("MEILI_URL")
    if not meili_url:
        pytest.skip("MEILI_URL not set")

    import meilisearch

    from deco.index.meili import MeiliVectorIndex

    client = meilisearch.Client(meili_url, os.environ.get("MEILI_API_KEY"))
    index_uid = f"deco-test-{uuid.uuid4().hex}"
    raw_index = client.create_index(index_uid, {"primaryKey": "id"})
    client.wait_for_task(raw_index.task_uid)
    index = client.index(index_uid)
    try:
        task = index.update_embedders({"deco": {"source": "userProvided", "dimensions": 2}})
        index.wait_for_task(task.task_uid)

        target = MeiliVectorIndex(index)
        target.add("met:1", VEC_A, "v1")
        target.add("met:2", VEC_B, "v1")
        target.add("met:3", VEC_C, "v1")
        # Meilisearch indexing is asynchronous; give the last task time to land.
        tasks = index.get_tasks()
        for task_info in tasks.results:
            index.wait_for_task(task_info.uid)

        assert target.has("met:1", "v1") is True
        assert target.has("met:1", "v2") is False
        assert target.has("met:missing", "v1") is False

        hits = target.search(VEC_A, k=2)

        assert [hit.record_id for hit in hits] == ["met:1", "met:2"]
        for hit in hits:
            assert -1.0 <= hit.score <= 1.0
    finally:
        delete_task = client.delete_index(index_uid)
        client.wait_for_task(delete_task.task_uid)
