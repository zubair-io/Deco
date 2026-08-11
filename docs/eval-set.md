# Eval set

Offline accuracy/latency evaluation for the identify pipeline. The harness
(`deco.eval.harness`, driven by `python -m deco.eval.run`) scores a pipeline
against a hand-shot eval set of query photos with known ground truth.

## Directory layout

```
eval/
  photos/           # query photos, arbitrary sub-structure under this root
  manifest.jsonl     # one line per query photo
```

`--photos-root` defaults to the manifest's parent directory, so a manifest at
`eval/manifest.jsonl` referencing `photos/img_0042.jpg` resolves to
`eval/photos/img_0042.jpg` by default. Pass `--photos-root` explicitly to
point at photos stored elsewhere.

## Manifest format

`eval/manifest.jsonl` is JSONL: one JSON object per line, no wrapping array.
Each object has:

- `photo` (string, required) — path to the query photo, relative to
  `--photos-root`.
- `ground_truth` (string or `null`, required) — the expected `record_id`
  (e.g. `"met:436535"`) for an in-corpus photo, or `null` for an
  out-of-corpus negative (a photo of a work that is not in the corpus at
  all, used to measure the false-positive rate).
- `condition` (string, optional) — the shooting condition, used for the
  per-condition top-1 breakdown. Omitted or `null` cases are grouped under
  `"uncategorized"` in the report.

Example lines:

```jsonl
{"photo": "photos/img_0042.jpg", "ground_truth": "met:436535", "condition": "oblique"}
{"photo": "photos/img_0099.jpg", "ground_truth": null}
```

Blank lines are ignored.

## Shooting protocol

Target 40-80 distinct works from the corpus, each photographed under six
conditions, for roughly 240-480 in-corpus query photos total:

1. **Clean frontal** — well-lit, straight-on, minimal distortion.
2. **~30° oblique** — camera angled roughly 30° off the frontal axis.
3. **Tight crop** — framed close enough that some of the work's edge is
   cropped out of frame.
4. **Glare** — a real specular highlight or reflection crossing the work
   (e.g. from glass/varnish under gallery lighting).
5. **Screen re-photo** — a photo of the work as displayed on a phone/laptop
   screen (moiré, screen glare, aspect-ratio letterboxing all fair game).
6. **Print re-photo** — a photo of a printed reproduction of the work (postcard,
   catalog page, poster).

In addition, collect **at least 50 out-of-corpus negatives**: photos of
artworks, objects, or scenes that are not in the ingested corpus at all
(`ground_truth: null`). These measure the false-positive rate — how often the
pipeline confidently (and wrongly) matches something that was never
ingested.

## Targets

Run via:

```
python -m deco.eval.run --manifest eval/manifest.jsonl \
    --model <dinov2.onnx> --aliked <aliked.onnx> --lightglue <lightglue.onnx>
```

The CLI prints an aligned text table and a single-line JSON summary, then
exits non-zero (unless `--no-gate` is passed) if any of the spec's bars are
missed:

| metric | target |
| --- | --- |
| top-1 accuracy | >= 85% |
| top-5 accuracy | >= 95% |
| false-positive rate | <= 5% |
| p95 latency | < 2s, measured on the target deployment hardware (phoebe) |

Top-1 requires both the correct record ranked first *and* `confident=True`;
a not-confident result is always a top-1 miss even if the correct record was
ranked first. Top-5 measures retrieval quality independent of the
confidence gate: it's a hit whenever the correct record appears anywhere in
the first five matches, confident or not. The false-positive rate and top-1/
top-5 accuracy are gated by the CLI; p95 latency is reported for eyeballing
against the phoebe target but is not gated in the CLI's exit code, since it
depends on the hardware the harness happens to run on.
