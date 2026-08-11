"""Ingest adapter for the Metropolitan Museum of Art open-access CSV + Collection API."""

import csv
from collections.abc import Iterator
from pathlib import Path

import httpx

from deco.records import Record

OBJECT_API_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"


class MetAdapter:
    source = "met"

    def __init__(self, csv_path: Path, client: httpx.Client) -> None:
        self.csv_path = csv_path
        self.client = client

    def records(self) -> Iterator[Record]:
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Is Public Domain") != "True":
                    continue
                wikidata_id = row.get("Wikidata URL") or None
                yield Record(
                    source=self.source,
                    source_object_id=row["Object ID"],
                    title=row["Title"],
                    artist=row["Artist Display Name"],
                    date=row["Object Date"],
                    medium=row["Medium"],
                    accession_number=row["Object Number"],
                    image_url=None,
                    license="CC0",
                    wikidata_id=wikidata_id,
                )

    def resolve_image_url(self, source_object_id: str) -> str | None:
        url = OBJECT_API_URL.format(object_id=source_object_id)
        response = self.client.get(url)
        response.raise_for_status()
        primary_image = response.json().get("primaryImage")
        return primary_image or None
