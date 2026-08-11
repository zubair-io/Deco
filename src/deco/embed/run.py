"""Resumable embedding pass over cached images, plus a thin CLI entry point.

python -m deco.embed.run --model <path/to/dinov2.onnx>
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from deco.embed.embedder import Embedder
from deco.ingest.fetch import ImageCache
from deco.records import RecordStore

logger = logging.getLogger(__name__)


class VectorSink(Protocol):
    def add(self, record_id: str, vector: np.ndarray, model_version: str) -> None: ...

    def has(self, record_id: str, model_version: str) -> bool: ...


@dataclass
class EmbedStats:
    embedded: int = 0
    skipped_existing: int = 0
    no_image: int = 0
    failed: int = 0


def run_embed_pass(
    store: RecordStore,
    cache: ImageCache,
    embedder: Embedder,
    sink: VectorSink,
) -> EmbedStats:
    """Embed every cached image not yet embedded under `embedder.model_version`.

    Resumable: a record already embedded under this `model_version` is skipped
    (`skipped_existing`); switching to a new `model_version` re-embeds every
    record without touching the underlying records (spec P0-2). Records with
    no cached image are counted (`no_image`) and skipped. A cached image file
    that fails to decode is counted (`failed`) and does not abort the pass.
    """
    stats = EmbedStats()

    for record in store.iter_records():
        record_id = record.record_id

        if not cache.has(record_id):
            stats.no_image += 1
            continue

        if sink.has(record_id, embedder.model_version):
            stats.skipped_existing += 1
            continue

        try:
            with Image.open(cache.path_for(record_id)) as raw:
                image = raw.convert("RGB")
        except (UnidentifiedImageError, OSError):
            logger.warning("failed to decode cached image for %s", record_id, exc_info=True)
            stats.failed += 1
            continue

        vector = embedder.embed(image)
        sink.add(record_id, vector, embedder.model_version)
        stats.embedded += 1

    return stats


class MongoVectorSink:
    """VectorSink backed by a Mongo collection; `_id` is `record_id:model_version`."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def add(self, record_id: str, vector: np.ndarray, model_version: str) -> None:
        doc = {
            "_id": f"{record_id}:{model_version}",
            "record_id": record_id,
            "model_version": model_version,
            "vector": vector.tolist(),
        }
        self._collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def has(self, record_id: str, model_version: str) -> bool:
        doc = self._collection.find_one({"_id": f"{record_id}:{model_version}"})
        return doc is not None


def main() -> None:
    from deco.config import load_config

    parser = argparse.ArgumentParser(description="Run a Deco embedding pass.")
    parser.add_argument("--model", required=True, type=Path, help="path to the ONNX model export")
    parser.add_argument("--config", type=Path, default=None, help="path to deco.toml")
    parser.add_argument(
        "--mongo-url", default=os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    )
    parser.add_argument("--mongo-db", default="deco")
    parser.add_argument("--mongo-records-collection", default="records")
    parser.add_argument("--mongo-vectors-collection", default="vectors")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from pymongo import MongoClient

    from deco.embed.onnx_dinov2 import DinoV2OnnxEmbedder
    from deco.stores.mongo import MongoRecordStore

    mongo_client = MongoClient(args.mongo_url)
    db = mongo_client[args.mongo_db]
    store = MongoRecordStore(db[args.mongo_records_collection])
    cache = ImageCache(cfg.paths.image_cache_dir)
    embedder = DinoV2OnnxEmbedder(args.model, cfg.embedding.model_version)
    sink = MongoVectorSink(db[args.mongo_vectors_collection])

    stats = run_embed_pass(store, cache, embedder, sink)

    print(
        f"embedded={stats.embedded} skipped_existing={stats.skipped_existing} "
        f"no_image={stats.no_image} failed={stats.failed}"
    )


if __name__ == "__main__":
    main()
