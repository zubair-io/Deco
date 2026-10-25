"""Record schema and storage protocol shared by ingest, embed, and API layers."""

from collections.abc import Iterator
from typing import Protocol

from pydantic import BaseModel


class Record(BaseModel):
    source: str
    source_object_id: str
    title: str
    artist: str
    date: str
    medium: str
    accession_number: str
    image_url: str | None
    license: str
    wikidata_id: str | None = None
    dedupe_key: str | None = None

    @property
    def record_id(self) -> str:
        return f"{self.source}:{self.source_object_id}"


class RecordStore(Protocol):
    def upsert(self, record: Record) -> None: ...

    def get(self, record_id: str) -> Record | None: ...

    def count(self) -> int: ...

    def iter_records(self) -> Iterator[Record]: ...

    def set_image_url(self, record_id: str, url: str) -> None: ...
