"""In-memory RecordStore used by all unit tests."""

from collections.abc import Iterator

from deco.records import Record


class InMemoryRecordStore:
    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def upsert(self, record: Record) -> None:
        self._records[record.record_id] = record

    def get(self, record_id: str) -> Record | None:
        return self._records.get(record_id)

    def count(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterator[Record]:
        return iter(self._records.values())

    def set_image_url(self, record_id: str, url: str) -> None:
        record = self._records[record_id]
        self._records[record_id] = record.model_copy(update={"image_url": url})
