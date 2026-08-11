"""Tests for deco.eval.harness: the eval harness scores a pipeline against a manifest."""

import dataclasses
import json
import math

import pytest
from PIL import Image

from deco.api.pipeline import IdentifyResult, Match
from deco.eval.harness import EvalCase, EvalReport, evaluate, load_manifest, meets_targets
from deco.records import Record


def _record(record_id: str) -> Record:
    source, source_object_id = record_id.split(":", 1)
    return Record(
        source=source,
        source_object_id=source_object_id,
        title=f"Title {record_id}",
        artist="Artist",
        date="1900",
        medium="Oil on canvas",
        accession_number=f"acc-{record_id}",
        image_url=f"https://example.com/{record_id}.jpg",
        license="CC0",
    )


class StubPipeline:
    """Pipeline duck-type stub: scripts a canned IdentifyResult per photo name."""

    def __init__(self, results: dict[str, IdentifyResult]) -> None:
        self._results = results
        self.calls: list[str] = []
        # each call records the photo name via a sentinel embedded on the "image"
        # (we pass Image objects tagged via a wrapper dict keyed by identity)

    def identify(self, image) -> IdentifyResult:
        # `image` here is actually the photo name string, since our fake
        # `load_image` just returns the name itself for simplicity.
        self.calls.append(image)
        return self._results[image]


def _match(record_id: str, cosine: float = 0.9, inliers: int = 50) -> Match:
    return Match(record=_record(record_id), cosine=cosine, inliers=inliers)


def _make_scripted_results() -> dict[str, IdentifyResult]:
    return {
        # photo1: correct match at rank 1, confident -> top1 hit, top5 hit
        "photo1": IdentifyResult(
            confident=True,
            matches=[_match("met:1"), _match("met:2"), _match("met:3")],
        ),
        # photo2: correct match at rank 3, confident -> top1 miss, top5 hit
        "photo2": IdentifyResult(
            confident=True,
            matches=[_match("met:9"), _match("met:8"), _match("met:2")],
        ),
        # photo3 (in-corpus): complete miss -> top1 miss, top5 miss
        "photo3": IdentifyResult(
            confident=False,
            matches=[_match("met:9"), _match("met:8")],
        ),
        # neg1 (out-of-corpus): not confident -> not a false positive
        "neg1": IdentifyResult(confident=False, matches=[_match("met:5")]),
        # neg2 (out-of-corpus): confident but wrong -> false positive
        "neg2": IdentifyResult(confident=True, matches=[_match("met:7")]),
    }


def _make_cases() -> list[EvalCase]:
    return [
        EvalCase(photo="photo1", ground_truth="met:1", condition="frontal"),
        EvalCase(photo="photo2", ground_truth="met:2", condition="oblique"),
        EvalCase(photo="photo3", ground_truth="met:3", condition="frontal"),
        EvalCase(photo="neg1", ground_truth=None, condition=None),
        EvalCase(photo="neg2", ground_truth=None, condition=None),
    ]


def _identity_load_image(photo: str) -> str:
    return photo


def _fake_clock():
    """Returns a clock() callable that advances by a fixed step on each call."""
    ticks = iter(
        [
            0.0,
            0.10,  # photo1: 0.10s
            0.10,
            0.30,  # photo2: 0.20s
            0.30,
            0.90,  # photo3: 0.60s
            0.90,
            1.00,  # neg1: 0.10s
            1.00,
            1.50,  # neg2: 0.50s
        ]
    )

    def clock() -> float:
        return next(ticks)

    return clock


def test_evaluate_computes_top1_top5_fp_and_by_condition():
    pipeline = StubPipeline(_make_scripted_results())
    cases = _make_cases()

    report = evaluate(pipeline, cases, _identity_load_image, clock=_fake_clock())

    assert report.top1 == pytest.approx(1 / 3)
    assert report.top5 == pytest.approx(2 / 3)
    assert report.false_positive_rate == pytest.approx(0.5)
    assert report.n_in_corpus == 3
    assert report.n_out_of_corpus == 2


def test_evaluate_p95_latency_from_fake_clock():
    pipeline = StubPipeline(_make_scripted_results())
    cases = _make_cases()

    report = evaluate(pipeline, cases, _identity_load_image, clock=_fake_clock())

    # latencies: [0.10, 0.20, 0.60, 0.10, 0.50] -> sorted [0.10, 0.10, 0.20, 0.50, 0.60]
    # p95 index = min(len-1, ceil(0.95*5)-1) = min(4, 4) = 4 -> 0.60
    assert report.p95_latency_s == pytest.approx(0.60)


def test_evaluate_by_condition_groups_correctly():
    pipeline = StubPipeline(_make_scripted_results())
    cases = _make_cases()

    report = evaluate(pipeline, cases, _identity_load_image, clock=_fake_clock())

    # frontal: photo1 (hit) + photo3 (miss) -> 1/2
    assert report.by_condition["frontal"] == pytest.approx(0.5)
    # oblique: photo2 (miss for top1) -> 0/1
    assert report.by_condition["oblique"] == pytest.approx(0.0)
    # out-of-corpus cases are excluded from by_condition entirely (no "uncategorized"
    # entry here since only the negatives have condition=None and they're out-of-corpus)
    assert "uncategorized" not in report.by_condition


def test_evaluate_uncategorized_condition_for_in_corpus_case_without_condition():
    results = {"p1": IdentifyResult(confident=True, matches=[_match("met:1")])}
    cases = [EvalCase(photo="p1", ground_truth="met:1", condition=None)]

    def clock():
        return 0.0

    report = evaluate(StubPipeline(results), cases, _identity_load_image, clock=clock)

    assert report.by_condition == {"uncategorized": 1.0}


def test_evaluate_not_confident_in_corpus_counts_as_top1_miss_but_can_be_top5_hit():
    results = {"p1": IdentifyResult(confident=False, matches=[_match("met:1"), _match("met:2")])}
    cases = [EvalCase(photo="p1", ground_truth="met:1", condition=None)]

    def clock():
        return 0.0

    report = evaluate(StubPipeline(results), cases, _identity_load_image, clock=clock)

    assert report.top1 == 0.0
    assert report.top5 == 1.0


def test_evaluate_empty_cases_yields_zero_rates_no_errors():
    report = evaluate(StubPipeline({}), [], _identity_load_image, clock=lambda: 0.0)

    assert report.top1 == 0.0
    assert report.top5 == 0.0
    assert report.false_positive_rate == 0.0
    assert report.p95_latency_s == 0.0
    assert report.n_in_corpus == 0
    assert report.n_out_of_corpus == 0
    assert report.by_condition == {}


def test_evaluate_only_out_of_corpus_cases_yields_zero_top1_top5():
    results = {
        "neg1": IdentifyResult(confident=False, matches=[]),
        "neg2": IdentifyResult(confident=True, matches=[_match("met:1")]),
    }
    cases = [
        EvalCase(photo="neg1", ground_truth=None, condition=None),
        EvalCase(photo="neg2", ground_truth=None, condition=None),
    ]

    report = evaluate(StubPipeline(results), cases, _identity_load_image, clock=lambda: 0.0)

    assert report.top1 == 0.0
    assert report.top5 == 0.0
    assert report.false_positive_rate == pytest.approx(0.5)


def test_load_manifest_parses_jsonl(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {"photo": "photos/a.jpg", "ground_truth": "met:1", "condition": "oblique"}
                ),
                json.dumps({"photo": "photos/b.jpg", "ground_truth": None}),
                "",
            ]
        )
    )

    cases = load_manifest(manifest)

    assert cases == [
        EvalCase(photo="photos/a.jpg", ground_truth="met:1", condition="oblique"),
        EvalCase(photo="photos/b.jpg", ground_truth=None, condition=None),
    ]


def test_load_manifest_skips_blank_lines(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"photo": "a.jpg", "ground_truth": "met:1"})
        + "\n\n\n"
        + json.dumps({"photo": "b.jpg", "ground_truth": None})
        + "\n"
    )

    cases = load_manifest(manifest)

    assert len(cases) == 2


def test_eval_case_is_frozen():
    case = EvalCase(photo="a.jpg", ground_truth="met:1", condition=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        case.photo = "b.jpg"  # type: ignore[misc]


def test_meets_targets_passes_when_all_thresholds_hit():
    report = EvalReport(
        top1=0.85,
        top5=0.95,
        false_positive_rate=0.05,
        p95_latency_s=1.5,
        n_in_corpus=10,
        n_out_of_corpus=5,
        by_condition={},
    )
    assert meets_targets(report) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("top1", 0.84),
        ("top5", 0.94),
        ("false_positive_rate", 0.06),
    ],
)
def test_meets_targets_fails_when_a_bar_is_missed(field, value):
    kwargs = {
        "top1": 0.9,
        "top5": 0.97,
        "false_positive_rate": 0.02,
        "p95_latency_s": 1.0,
        "n_in_corpus": 10,
        "n_out_of_corpus": 5,
        "by_condition": {},
    }
    kwargs[field] = value
    report = EvalReport(**kwargs)
    assert meets_targets(report) is False


def test_evaluate_passes_loaded_image_through_to_pipeline():
    """`evaluate` should call `load_image(case.photo)` then pass the result to `identify`."""

    loaded = {}

    class RecordingPipeline:
        def identify(self, image) -> IdentifyResult:
            loaded["image"] = image
            return IdentifyResult(confident=False, matches=[])

    def load_image(photo: str) -> Image.Image:
        assert photo == "photo.jpg"
        return Image.new("RGB", (2, 2))

    cases = [EvalCase(photo="photo.jpg", ground_truth=None, condition=None)]
    evaluate(RecordingPipeline(), cases, load_image, clock=lambda: 0.0)

    assert isinstance(loaded["image"], Image.Image)


def test_p95_matches_manual_formula_for_various_lengths():
    # Sanity-check the documented p95 formula directly via evaluate with an
    # increasing sequence of latencies.
    n = 7
    results = {f"p{i}": IdentifyResult(confident=False, matches=[]) for i in range(n)}
    cases = [EvalCase(photo=f"p{i}", ground_truth=None, condition=None) for i in range(n)]

    ticks = []
    t = 0.0
    for i in range(n):
        ticks.append(t)
        t += (i + 1) * 0.1
        ticks.append(t)

    it = iter(ticks)

    def clock():
        return next(it)

    report = evaluate(StubPipeline(results), cases, _identity_load_image, clock=clock)

    latencies = sorted((i + 1) * 0.1 for i in range(n))
    expected_idx = min(n - 1, math.ceil(0.95 * n) - 1)
    expected = latencies[expected_idx]
    assert report.p95_latency_s == pytest.approx(expected)
