import os

import pytest

from deco.records import Record
from deco.stores.memory import InMemoryRecordStore


def make_record(oid: str = "436535") -> Record:
    return Record(
        source="met",
        source_object_id=oid,
        title="Wheat Field with Cypresses",
        artist="Vincent van Gogh",
        date="1889",
        medium="Oil on canvas",
        accession_number="93.1.2",
        image_url=None,
        license="CC0",
    )


def test_record_id():
    assert make_record().record_id == "met:436535"


def test_upsert_is_idempotent():
    store = InMemoryRecordStore()
    store.upsert(make_record())
    store.upsert(make_record())
    assert store.count() == 1


def test_get_and_iter():
    store = InMemoryRecordStore()
    store.upsert(make_record("1"))
    store.upsert(make_record("2"))
    assert store.get("met:1").source_object_id == "1"
    assert store.get("met:missing") is None
    assert {r.record_id for r in store.iter_records()} == {"met:1", "met:2"}


def test_set_image_url():
    store = InMemoryRecordStore()
    store.upsert(make_record())
    store.set_image_url("met:436535", "https://images.metmuseum.org/x.jpg")
    assert store.get("met:436535").image_url == "https://images.metmuseum.org/x.jpg"


@pytest.mark.integration
def test_mongo_record_store_roundtrip():
    mongo_url = os.environ.get("MONGO_URL")
    if not mongo_url:
        pytest.skip("MONGO_URL not set")

    from pymongo import MongoClient

    from deco.stores.mongo import MongoRecordStore

    client = MongoClient(mongo_url)
    collection = client["deco_test"]["records_test"]
    collection.delete_many({})
    try:
        store = MongoRecordStore(collection)
        store.upsert(make_record())
        store.upsert(make_record())
        assert store.count() == 1
        assert store.get("met:436535").title == "Wheat Field with Cypresses"
        assert store.get("met:missing") is None

        store.set_image_url("met:436535", "https://images.metmuseum.org/x.jpg")
        assert store.get("met:436535").image_url == "https://images.metmuseum.org/x.jpg"

        assert {r.record_id for r in store.iter_records()} == {"met:436535"}
    finally:
        collection.delete_many({})
        client.close()
