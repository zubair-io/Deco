"""Meilisearch-backed VectorIndex. Requires the `meili` extra (meilisearch).

Meilisearch stores vectors as "userProvided" embeddings on a document under
`_vectors.<embedder_name>`. Document ids in Meilisearch may not contain `:`,
but Deco `record_id`s are `<source>:<source_object_id>` (e.g. `met:436535`),
so the Meilisearch document id is a sanitized copy (`:` -> `_`) while the raw
`record_id` is kept as a separate field and returned from `search`.

Two semantics differ from `InMemoryVectorIndex` and are worth calling out:
- `add` issues an asynchronous indexing task; Meilisearch does not guarantee
  the document is immediately visible to `has`/`search` afterwards.
- `search` scores come from Meilisearch's semantic ranking (`_semanticScore`,
  falling back to `_rankingScore`), which approximates but is not guaranteed
  to be bit-identical to a plain cosine dot product.
"""

from typing import Any

import numpy as np

from deco.index.base import Hit

try:
    from meilisearch.errors import MeilisearchApiError
    from meilisearch.index import Index
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "MeiliVectorIndex requires the 'meili' extra. Install it with: uv sync --extra meili"
    ) from exc


def _doc_id(record_id: str) -> str:
    """Meilisearch document ids reject `:`; substitute it with `_`."""
    return record_id.replace(":", "_")


class MeiliVectorIndex:
    def __init__(self, index: Index, embedder_name: str = "deco") -> None:
        self._index = index
        self._embedder_name = embedder_name

    def add(self, record_id: str, vector: np.ndarray, model_version: str) -> None:
        document = {
            "id": _doc_id(record_id),
            "record_id": record_id,
            "model_version": model_version,
            "_vectors": {self._embedder_name: np.asarray(vector, dtype=np.float32).tolist()},
        }
        self._index.add_documents([document])

    def has(self, record_id: str, model_version: str) -> bool:
        try:
            document = self._index.get_document(_doc_id(record_id))
        except MeilisearchApiError:
            return False
        return getattr(document, "model_version", None) == model_version

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        if k <= 0:
            return []

        response = self._index.search(
            "",
            {
                "vector": np.asarray(vector, dtype=np.float32).tolist(),
                "limit": k,
                "retrieveVectors": False,
                "hybrid": {"embedder": self._embedder_name, "semanticRatio": 1.0},
                "showRankingScore": True,
            },
        )

        hits: list[Hit] = []
        for hit in response.get("hits", []):
            hits.append(Hit(record_id=hit["record_id"], score=_score(hit)))
        return hits


def _score(hit: dict[str, Any]) -> float:
    """Prefer Meilisearch's semantic (vector) score; fall back to the overall ranking score."""
    if "_semanticScore" in hit:
        return float(hit["_semanticScore"])
    return float(hit.get("_rankingScore", 0.0))
