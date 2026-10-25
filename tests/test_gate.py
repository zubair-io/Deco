"""Tests for confidence gate."""

import pytest

from deco.config import GateConfig
from deco.gate import apply_gate
from deco.rerank.base import RankedCandidate


@pytest.fixture
def gate_cfg() -> GateConfig:
    """Gate config with min_cosine=0.55, min_inliers=30."""
    return GateConfig(min_cosine=0.55, min_inliers=30)


def test_confident_top_candidate_exceeds_both_thresholds(gate_cfg: GateConfig) -> None:
    """Top candidate (cos 0.8, inliers 100) passes both thresholds."""
    candidates = [RankedCandidate(record_id="a", cosine=0.8, inliers=100)]
    result = apply_gate(candidates, gate_cfg)
    assert result.confident is True
    assert result.candidates == candidates


def test_not_confident_inliers_below_threshold(gate_cfg: GateConfig) -> None:
    """Top candidate (cos 0.8, inliers 10) fails inliers threshold."""
    candidates = [RankedCandidate(record_id="a", cosine=0.8, inliers=10)]
    result = apply_gate(candidates, gate_cfg)
    assert result.confident is False
    assert result.candidates == candidates


def test_not_confident_cosine_below_threshold(gate_cfg: GateConfig) -> None:
    """Top candidate (cos 0.4, inliers 100) fails cosine threshold."""
    candidates = [RankedCandidate(record_id="a", cosine=0.4, inliers=100)]
    result = apply_gate(candidates, gate_cfg)
    assert result.confident is False
    assert result.candidates == candidates


def test_not_confident_empty_candidates(gate_cfg: GateConfig) -> None:
    """Empty candidate list is never confident."""
    candidates: list[RankedCandidate] = []
    result = apply_gate(candidates, gate_cfg)
    assert result.confident is False
    assert result.candidates == candidates
