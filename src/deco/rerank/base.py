"""Geometric re-rank contract: orchestration over a `Reranker` backend.

`rerank` takes the top cosine-similarity hits from a `VectorIndex` search and
refines the ranking using geometric feature matching (e.g. ALIKED + LightGlue
keypoint matching, see `deco.rerank.aliked`). Candidates are sorted by
`(inliers, cosine)` descending — geometric agreement wins ties, cosine breaks
ties among candidates with equal inlier counts (including all-zero-inlier
candidates, e.g. when an image is missing from the cache).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from deco.index.base import Hit


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching a query image against one candidate image."""

    inliers: int
    total_matches: int


@dataclass(frozen=True)
class RankedCandidate:
    """A hit after geometric re-ranking."""

    record_id: str
    cosine: float
    inliers: int


class Reranker(Protocol):
    def match(self, query: Image.Image, candidate: Image.Image, candidate_id: str) -> MatchResult:
        """Match `query` against `candidate`.

        `candidate_id` lets fakes script per-candidate results; real
        implementations may ignore it.
        """
        ...


class FakeReranker:
    """Test double: returns scripted inlier counts keyed by candidate id."""

    def __init__(self, scores: dict[str, int]) -> None:
        self._scores = scores

    def match(self, query: Image.Image, candidate: Image.Image, candidate_id: str) -> MatchResult:
        score = self._scores.get(candidate_id, 0)
        return MatchResult(inliers=score, total_matches=score)


def rerank(
    query_img: Image.Image,
    hits: list[Hit],
    images: Callable[[str], Image.Image | None],
    reranker: Reranker,
    top_n: int,
) -> list[RankedCandidate]:
    """Re-rank `hits` by geometric match against `query_img`.

    For each hit, `images(hit.record_id)` supplies the candidate image; if it
    returns `None` (e.g. not in the cache), the candidate gets `inliers=0`
    without calling `reranker.match`. Results are sorted by
    `(inliers, cosine)` descending and truncated to `top_n`.
    """
    candidates = []
    for hit in hits:
        candidate_img = images(hit.record_id)
        if candidate_img is None:
            inliers = 0
        else:
            inliers = reranker.match(query_img, candidate_img, hit.record_id).inliers
        candidates.append(
            RankedCandidate(record_id=hit.record_id, cosine=hit.score, inliers=inliers)
        )

    candidates.sort(key=lambda c: (c.inliers, c.cosine), reverse=True)
    return candidates[:top_n]
