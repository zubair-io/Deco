"""MongoDB-backed RecordStore. Requires the `mongo` extra (pymongo)."""

from collections.abc import Iterator
from typing import Any

from deco.records import Record

try:
    from pymongo.collection import Collection
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "MongoRecordStore requires the 'mongo' extra. Install it with: uv sync --extra mongo"
    ) from exc


class MongoRecordStore:
    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    def upsert(self, record: Record) -> None:
        doc: dict[str, Any] = record.model_dump()
        record_id = record.record_id
        doc["_id"] = record_id
        self._collection.replace_one({"_id": record_id}, doc, upsert=True)

    def get(self, record_id: str) -> Record | None:
        doc = self._collection.find_one({"_id": record_id})
        if doc is None:
            return None
        doc = dict(doc)
        doc.pop("_id", None)
        return Record.model_validate(doc)

    def count(self) -> int:
        return self._collection.count_documents({})

    def iter_records(self) -> Iterator[Record]:
        for doc in self._collection.find():
            doc = dict(doc)
            doc.pop("_id", None)
            yield Record.model_validate(doc)

    def set_image_url(self, record_id: str, url: str) -> None:
        self._collection.update_one({"_id": record_id}, {"$set": {"image_url": url}})
