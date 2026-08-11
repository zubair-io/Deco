"""Resumable ingest orchestration: metadata upsert + rate-limited image fetch/cache.

Also doubles as a thin CLI entry point:

    python -m deco.ingest.run --source met --csv <path/to/MetObjects.csv>
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from deco.config import IngestConfig
from deco.ingest.adapter import IngestAdapter
from deco.ingest.fetch import ImageCache, RateLimiter, fetch_with_backoff
from deco.records import RecordStore

USER_AGENT = "deco-ingest/0.1 (github.com/zubair-io/Deco)"


@dataclass
class IngestStats:
    records: int = 0
    images_fetched: int = 0
    images_cached: int = 0
    no_image: int = 0


def run_ingest(
    adapter: IngestAdapter,
    store: RecordStore,
    cache: ImageCache,
    client: httpx.Client,
    cfg: IngestConfig,
) -> IngestStats:
    """Ingest every record from `adapter` into `store`, caching its image via `cache`.

    Idempotent and resumable: a record already in `store` with an
    `image_url` is never re-resolved against the source API, and an image
    already present in `cache` is never re-downloaded. Records whose
    primary image never resolves are skipped (not upserted) and counted in
    `no_image`.
    """
    limiter = RateLimiter(cfg.rate_limit_rps)
    stats = IngestStats()

    for record in adapter.records():
        record_id = record.record_id
        existing = store.get(record_id)

        if existing is not None and existing.image_url:
            image_url = existing.image_url
        else:
            limiter.wait()
            image_url = adapter.resolve_image_url(record.source_object_id)
            if not image_url:
                stats.no_image += 1
                continue
            record = record.model_copy(update={"image_url": image_url})
            store.upsert(record)

        if cache.has(record_id):
            stats.images_cached += 1
        else:
            response = fetch_with_backoff(
                client, image_url, limiter=limiter, max_backoff_s=cfg.max_backoff_s
            )
            cache.put(record_id, response.content)
            stats.images_fetched += 1

    stats.records = store.count()
    return stats


def _build_adapter(source: str, csv_path: Path, client: httpx.Client) -> IngestAdapter:
    if source == "met":
        from deco.ingest.met import MetAdapter

        return MetAdapter(csv_path, client)
    raise ValueError(f"unknown source: {source!r}")


def main() -> None:
    from deco.config import load_config

    parser = argparse.ArgumentParser(description="Run a Deco museum-corpus ingest pass.")
    parser.add_argument("--source", required=True, choices=["met"], help="ingest adapter to use")
    parser.add_argument("--csv", required=True, type=Path, help="path to the source CSV export")
    parser.add_argument("--config", type=Path, default=None, help="path to deco.toml")
    parser.add_argument(
        "--mongo-url", default=os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    )
    parser.add_argument("--mongo-db", default="deco")
    parser.add_argument("--mongo-collection", default="records")
    args = parser.parse_args()

    cfg = load_config(args.config)

    from pymongo import MongoClient

    from deco.stores.mongo import MongoRecordStore

    mongo_client = MongoClient(args.mongo_url)
    collection = mongo_client[args.mongo_db][args.mongo_collection]
    store = MongoRecordStore(collection)

    cache = ImageCache(cfg.paths.image_cache_dir)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        adapter = _build_adapter(args.source, args.csv, client)
        stats = run_ingest(adapter, store, cache, client, cfg.ingest)

    print(
        f"records={stats.records} images_fetched={stats.images_fetched} "
        f"images_cached={stats.images_cached} no_image={stats.no_image}"
    )


if __name__ == "__main__":
    main()
