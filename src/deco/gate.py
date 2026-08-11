"""Confidence gate for filtering ranked candidates."""

from dataclasses import dataclass

from deco.config import GateConfig
from deco.rerank.base import RankedCandidate


@dataclass(frozen=True)
class GateDecision:
    """Result of applying a confidence gate to ranked candidates."""

    confident: bool
    candidates: list[RankedCandidate]


def apply_gate(candidates: list[RankedCandidate], cfg: GateConfig) -> GateDecision:
    """Apply confidence gate to ranked candidates.

    A result is confident iff the top candidate has cosine >= cfg.min_cosine
    AND inliers >= cfg.min_inliers. Candidates are returned unchanged either way.
    """
    confident = (
        len(candidates) > 0
        and candidates[0].cosine >= cfg.min_cosine
        and candidates[0].inliers >= cfg.min_inliers
    )
    return GateDecision(confident=confident, candidates=candidates)
