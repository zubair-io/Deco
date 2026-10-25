from pathlib import Path

import httpx

from deco.ingest.met import MetAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "met_sample.csv"


def test_records_filters_public_domain():
    adapter = MetAdapter(FIXTURE, client=httpx.Client())
    records = list(adapter.records())
    assert len(records) == 2  # third row is Is Public Domain=False
    assert all(r.source == "met" and r.license == "CC0" for r in records)


def test_resolve_image_url_uses_object_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/objects/436535")
        return httpx.Response(200, json={"primaryImage": "https://images.metmuseum.org/a.jpg"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = MetAdapter(FIXTURE, client=client)
    assert adapter.resolve_image_url("436535") == "https://images.metmuseum.org/a.jpg"


def test_records_skips_malformed_rows(tmp_path: Path):
    ragged_csv = tmp_path / "ragged.csv"
    ragged_csv.write_text(
        "Object Number,Is Highlight,Is Timeline Work,Is Public Domain,Object ID,Department,"
        "Title,Artist Display Name,Object Date,Medium,Wikidata URL\n"
        "64.4,True,False,True,436535,European Paintings,Wheat Field with Cypresses,"
        "Vincent van Gogh,1889,Oil on canvas,\n"
        "99.9,False,False,True,111\n"  # ragged: truncated after Object ID
        "67.187.108,False,False,True,436532,European Paintings,The Roman Campagna,"
        "Vincent van Gogh,1888,Oil on canvas,\n"
    )
    adapter = MetAdapter(ragged_csv, client=httpx.Client())
    records = list(adapter.records())
    assert [r.source_object_id for r in records] == ["436535", "436532"]


def test_resolve_image_url_none_when_empty():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"primaryImage": ""}))
    )
    assert MetAdapter(FIXTURE, client=client).resolve_image_url("1") is None
