"""VectorIndex contract: nearest-neighbour search over embedded vectors.

`VectorIndex` is a structural superset of `deco.embed.run.VectorSink` — it
adds `search` to the same `add`/`has` signatures, so a single backend object
(e.g. `InMemoryVectorIndex`) can serve as both the embed pass's sink and the
retrieval-time index. There is no shared base class; the relationship holds
by structural typing alone, so `VectorSink` is never imported here.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Hit:
    """A single search result: a record id and its similarity score.

    Contract: `score` is cosine similarity in [-1, 1] (it flows unchanged into
    `RankedCandidate.cosine` and is compared against `gate.min_cosine`).
    Backends whose native score is on a different scale (e.g. Meilisearch's
    [0, 1] ranking score) must map it to cosine before returning a `Hit`.
    """

    record_id: str
    score: float


class VectorIndex(Protocol):
    def add(self, record_id: str, vector: np.ndarray, model_version: str) -> None:
        """Store `vector` for `record_id` under `model_version`.

        Re-adding an existing `record_id` replaces its vector and version;
        it does not create a duplicate entry.
        """
        ...

    def has(self, record_id: str, model_version: str) -> bool:
        """True only if `record_id` is currently stored under exactly `model_version`."""
        ...

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        """Return up to `k` nearest stored vectors to `vector`, sorted by score descending."""
        ...
