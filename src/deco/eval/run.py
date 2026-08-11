"""Eval harness entry point: scores a real pipeline against a labelled manifest.

    python -m deco.eval.run --manifest eval/manifest.jsonl \\
        --model <dinov2.onnx> --aliked <aliked.onnx> --lightglue <lightglue.onnx>

Thin, untested glue — real component wiring only. Behaviour lives in
`deco.eval.harness`, covered by tests with a stub pipeline.
"""

import argparse
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from deco.eval.harness import EvalReport, evaluate, load_manifest, meets_targets

logger = logging.getLogger(__name__)


def _format_report(report: EvalReport) -> str:
    rows: list[tuple[str, str]] = [
        ("top1", f"{report.top1:.3f}"),
        ("top5", f"{report.top5:.3f}"),
        ("false_positive_rate", f"{report.false_positive_rate:.3f}"),
        ("p95_latency_s", f"{report.p95_latency_s:.3f}"),
        ("n_in_corpus", str(report.n_in_corpus)),
        ("n_out_of_corpus", str(report.n_out_of_corpus)),
    ]
    for condition, score in sorted(report.by_condition.items()):
        rows.append((f"by_condition[{condition}]", f"{score:.3f}"))

    width = max(len(label) for label, _ in rows)
    table = "\n".join(f"{label:<{width}}  {value}" for label, value in rows)

    payload = {
        "top1": report.top1,
        "top5": report.top5,
        "false_positive_rate": report.false_positive_rate,
        "p95_latency_s": report.p95_latency_s,
        "n_in_corpus": report.n_in_corpus,
        "n_out_of_corpus": report.n_out_of_corpus,
        "by_condition": report.by_condition,
    }
    return f"{table}\n{json.dumps(payload)}"


def _build_query_image_loader(photos_root: Path):
    def load(photo: str) -> Image.Image:
        with Image.open(photos_root / photo) as raw:
            return raw.convert("RGB")

    return load


def _build_candidate_image_loader(cache) -> Callable[[str], Image.Image | None]:
    def load(record_id: str) -> Image.Image | None:
        if not cache.has(record_id):
            return None
        try:
            with Image.open(cache.path_for(record_id)) as raw:
                return raw.convert("RGB")
        except (UnidentifiedImageError, OSError):
            logger.warning("failed to decode cached image for %s", record_id, exc_info=True)
            return None

    return load


def main() -> None:
    from deco.config import load_config

    parser = argparse.ArgumentParser(description="Run the Deco offline eval harness.")
    parser.add_argument("--manifest", required=True, type=Path, help="path to eval manifest.jsonl")
    parser.add_argument(
        "--photos-root",
        type=Path,
        default=None,
        help="root dir photo paths are relative to (default: manifest's parent dir)",
    )
    parser.add_argument(
        "--no-gate", action="store_true", help="exit 0 regardless of whether targets are met"
    )
    parser.add_argument("--model", required=True, type=Path, help="path to the DINOv2 ONNX export")
    parser.add_argument("--aliked", required=True, type=Path, help="path to the ALIKED ONNX export")
    parser.add_argument(
        "--lightglue", required=True, type=Path, help="path to the LightGlue ONNX export"
    )
    parser.add_argument("--config", type=Path, default=None, help="path to deco.toml")
    parser.add_argument(
        "--mongo-url", default=os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    )
    parser.add_argument("--mongo-db", default="deco")
    parser.add_argument("--mongo-collection", default="records")
    parser.add_argument("--meili-url", default=os.environ.get("MEILI_URL", "http://localhost:7700"))
    parser.add_argument("--meili-key", default=os.environ.get("MEILI_KEY"))
    parser.add_argument("--meili-index", default="deco-vectors")
    args = parser.parse_args()

    cfg = load_config(args.config)
    photos_root = args.photos_root or args.manifest.resolve().parent

    import meilisearch
    from pymongo import MongoClient

    from deco.api.pipeline import IdentifyPipeline
    from deco.embed.onnx_dinov2 import DinoV2OnnxEmbedder
    from deco.index.meili import MeiliVectorIndex
    from deco.ingest.fetch import ImageCache
    from deco.rerank.aliked import AlikedLightGlueReranker
    from deco.stores.mongo import MongoRecordStore

    mongo_client = MongoClient(args.mongo_url)
    store = MongoRecordStore(mongo_client[args.mongo_db][args.mongo_collection])

    meili_client = meilisearch.Client(args.meili_url, args.meili_key)
    index = MeiliVectorIndex(meili_client.index(args.meili_index))

    embedder = DinoV2OnnxEmbedder(args.model, cfg.embedding.model_version)
    reranker = AlikedLightGlueReranker(args.aliked, args.lightglue)
    cache = ImageCache(cfg.paths.image_cache_dir)

    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        image_loader=_build_candidate_image_loader(cache),
        reranker=reranker,
        cfg=cfg,
    )

    cases = load_manifest(args.manifest)
    report = evaluate(pipeline, cases, _build_query_image_loader(photos_root))

    print(_format_report(report))

    if not args.no_gate and not meets_targets(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
