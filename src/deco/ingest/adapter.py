"""Source-ingestion contract implemented by each museum-specific adapter."""

from collections.abc import Iterator
from typing import Protocol

from deco.records import Record


class IngestAdapter(Protocol):
    source: str

    def records(self) -> Iterator[Record]: ...

    def resolve_image_url(self, source_object_id: str) -> str | None: ...
