"""Eval harness: score an IdentifyPipeline against a labelled manifest.

Depends only on the `identify(image) -> IdentifyResult`-shaped duck type
(`Pipeline` below), not on the concrete `deco.api.pipeline.IdentifyPipeline`
class, so tests can pass a stub.
"""

import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

TARGET_TOP1 = 0.85
TARGET_TOP5 = 0.95
TARGET_FALSE_POSITIVE_RATE = 0.05


@dataclass(frozen=True)
class EvalCase:
    """One labelled query photo from the eval manifest."""

    photo: str
    ground_truth: str | None
    condition: str | None = None


@dataclass(frozen=True)
class EvalReport:
    """Aggregate accuracy/latency metrics from an eval run."""

    top1: float
    top5: float
    false_positive_rate: float
    p95_latency_s: float
    n_in_corpus: int
    n_out_of_corpus: int
    by_condition: dict[str, float]


class _IdentifyResultLike(Protocol):
    confident: bool
    matches: list[Any]


class Pipeline(Protocol):
    """Structural type for anything shaped like `IdentifyPipeline.identify`."""

    def identify(self, image: Any) -> _IdentifyResultLike: ...


def load_manifest(path: Path) -> list[EvalCase]:
    """Parse a JSONL eval manifest into `EvalCase`s, skipping blank lines."""
    cases: list[EvalCase] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(
                    photo=data["photo"],
                    ground_truth=data.get("ground_truth"),
                    condition=data.get("condition"),
                )
            )
    return cases


def _p95(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    ordered = sorted(latencies)
    idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[idx]


def evaluate(
    pipeline: Pipeline,
    cases: Iterable[EvalCase],
    load_image: Callable[[str], Any],
    clock: Callable[[], float] = time.perf_counter,
) -> EvalReport:
    """Run every case through `pipeline.identify` and score the results.

    In-corpus cases (`ground_truth is not None`) count a top-1 hit only when
    the top match's record id equals `ground_truth` AND `result.confident` is
    True; a top-5 hit checks membership in the first five record ids
    regardless of confidence (top5 measures retrieval quality, not the gate).
    Out-of-corpus cases (`ground_truth is None`) count a false positive iff
    `result.confident` is True.
    """
    latencies: list[float] = []
    top1_hits = 0
    top5_hits = 0
    false_positives = 0
    n_in_corpus = 0
    n_out_of_corpus = 0
    condition_hits: dict[str, int] = {}
    condition_totals: dict[str, int] = {}

    for case in cases:
        image = load_image(case.photo)
        t0 = clock()
        result = pipeline.identify(image)
        latencies.append(clock() - t0)

        record_ids = [m.record.record_id for m in result.matches]

        if case.ground_truth is not None:
            n_in_corpus += 1
            top1_hit = (
                bool(result.matches) and record_ids[0] == case.ground_truth and result.confident
            )
            top5_hit = case.ground_truth in record_ids[:5]
            top1_hits += int(top1_hit)
            top5_hits += int(top5_hit)

            condition = case.condition or "uncategorized"
            condition_totals[condition] = condition_totals.get(condition, 0) + 1
            condition_hits[condition] = condition_hits.get(condition, 0) + int(top1_hit)
        else:
            n_out_of_corpus += 1
            false_positives += int(result.confident)

    top1 = top1_hits / n_in_corpus if n_in_corpus else 0.0
    top5 = top5_hits / n_in_corpus if n_in_corpus else 0.0
    false_positive_rate = false_positives / n_out_of_corpus if n_out_of_corpus else 0.0
    by_condition = {
        condition: condition_hits[condition] / total
        for condition, total in condition_totals.items()
    }

    return EvalReport(
        top1=top1,
        top5=top5,
        false_positive_rate=false_positive_rate,
        p95_latency_s=_p95(latencies),
        n_in_corpus=n_in_corpus,
        n_out_of_corpus=n_out_of_corpus,
        by_condition=by_condition,
    )


def meets_targets(report: EvalReport) -> bool:
    """Pure threshold check against the spec's accuracy/FP bars.

    p95 latency is reported but not gated here (no fixed hardware baseline
    to gate against in a unit test); the CLI still prints it for eyeballing.
    """
    return (
        report.top1 >= TARGET_TOP1
        and report.top5 >= TARGET_TOP5
        and report.false_positive_rate <= TARGET_FALSE_POSITIVE_RATE
    )
