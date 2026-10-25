"""In-memory VectorIndex: dict-backed, exact cosine top-K.

Doubles as the test fake and a small-corpus dev backend. Assumes stored and
query vectors are already L2-normalized, so cosine similarity reduces to a
plain dot product — no normalization is performed here.
"""

import numpy as np

from deco.index.base import Hit


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        self._versions: dict[str, str] = {}

    def add(self, record_id: str, vector: np.ndarray, model_version: str) -> None:
        self._vectors[record_id] = np.asarray(vector, dtype=np.float32)
        self._versions[record_id] = model_version

    def has(self, record_id: str, model_version: str) -> bool:
        return self._versions.get(record_id) == model_version

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if not self._vectors or k <= 0:
            return []

        query = np.asarray(vector, dtype=np.float32)
        record_ids = list(self._vectors.keys())
        matrix = np.stack([self._vectors[record_id] for record_id in record_ids])
        scores = matrix @ query

        top_k = min(k, len(record_ids))
        order = np.argsort(-scores)[:top_k]
        return [Hit(record_id=record_ids[i], score=float(scores[i])) for i in order]
