"""Identify pipeline: embed -> retrieve -> rerank -> gate -> join records.

Orchestrates a single query image through the full identify flow, wiring
together the protocols established by earlier tasks (`Embedder`,
`VectorIndex`, `RecordStore`, `Reranker`) plus the gate. The API layer
(`deco.api.app`) is the only thing that decides what to expose to callers
when the result is not confident; `IdentifyResult.matches` always carries the
(joined) ranked candidates regardless of `confident`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

from deco.config import DecoConfig
from deco.embed.embedder import Embedder
from deco.gate import apply_gate
from deco.index.base import VectorIndex
from deco.records import Record, RecordStore
from deco.rerank.base import Reranker, rerank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    """A ranked candidate joined with its full record."""

    record: Record
    cosine: float
    inliers: int


@dataclass(frozen=True)
class IdentifyResult:
    """Outcome of `IdentifyPipeline.identify`."""

    confident: bool
    matches: list[Match]


class IdentifyPipeline:
    """Runs a query image through embed -> retrieve -> rerank -> gate -> join."""

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        store: RecordStore,
        image_loader: Callable[[str], Image.Image | None],
        reranker: Reranker,
        cfg: DecoConfig,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._store = store
        self._image_loader = image_loader
        self._reranker = reranker
        self._cfg = cfg

    def identify(self, image: Image.Image) -> IdentifyResult:
        vector = self._embedder.embed(image)
        hits = self._index.search(vector, self._cfg.retrieval.top_k)
        ranked = rerank(
            image,
            hits,
            self._image_loader,
            self._reranker,
            self._cfg.retrieval.rerank_top_n,
        )
        decision = apply_gate(ranked, self._cfg.gate)

        matches: list[Match] = []
        for candidate in decision.candidates:
            record = self._store.get(candidate.record_id)
            if record is None:
                logger.warning(
                    "dropping candidate %s: record missing from store", candidate.record_id
                )
                continue
            matches.append(Match(record=record, cosine=candidate.cosine, inliers=candidate.inliers))

        return IdentifyResult(confident=decision.confident, matches=matches)
