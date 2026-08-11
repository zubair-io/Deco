"""Query service entry point: wires real backends and runs the identify API.

    python -m deco.api.serve --model <dinov2.onnx> --aliked <aliked.onnx> --lightglue <lightglue.onnx>

Thin, untested glue — real component wiring only. Behaviour lives in
`deco.api.pipeline` and `deco.api.app`, both covered by tests with fakes.
"""

import argparse
import logging
import os
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from deco.ingest.fetch import ImageCache

logger = logging.getLogger(__name__)


def _build_image_loader(cache: ImageCache):
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
    import uvicorn

    from deco.config import load_config

    parser = argparse.ArgumentParser(description="Run the Deco identify query service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
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

    import meilisearch
    from pymongo import MongoClient

    from deco.api.app import create_app
    from deco.api.pipeline import IdentifyPipeline
    from deco.embed.onnx_dinov2 import DinoV2OnnxEmbedder
    from deco.index.meili import MeiliVectorIndex
    from deco.rerank.aliked import AlikedLightGlueReranker
    from deco.stores.mongo import MongoRecordStore

    mongo_client = MongoClient(args.mongo_url)
    store = MongoRecordStore(mongo_client[args.mongo_db][args.mongo_collection])

    meili_client = meilisearch.Client(args.meili_url, args.meili_key)
    index = MeiliVectorIndex(meili_client.index(args.meili_index))

    embedder = DinoV2OnnxEmbedder(args.model, cfg.embedding.model_version)
    reranker = AlikedLightGlueReranker(args.aliked, args.lightglue)
    cache = ImageCache(cfg.paths.image_cache_dir)
    image_loader = _build_image_loader(cache)

    pipeline = IdentifyPipeline(
        embedder=embedder,
        index=index,
        store=store,
        image_loader=image_loader,
        reranker=reranker,
        cfg=cfg,
    )

    app = create_app(pipeline)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
