# Spike: Bulk Acquisition of ~470K Met Open Access Images

Issue: zubair-io/Deco#2
Date: 2026-08-11
Status: Research only — no code/infra changes in this spike.

## Summary / Recommendation

**Do not do 470,000 sequential calls to `collectionapi.metmuseum.org/.../objects/{id}`.** There is no
official bulk/batch image endpoint and no IIIF manifest support, but two faster paths exist and should be
combined:

1. **Primary bulk source — Wikimedia Commons mirror.** The Met uploaded its CC0 public-domain images to
   Wikimedia Commons as part of the 2017 Open Access launch. The `Category:Images from Metropolitan Museum
   of Art` tree currently reports **389,030 files** (fetched 2026-08-11). These are the same CC0-licensed
   originals, enumerable and fetchable through the MediaWiki Action API, which supports **batched metadata
   lookups** (up to 50 titles per `action=query&prop=imageinfo` call, more for bots) — roughly **~7,800
   metadata calls** instead of 470,000. This is the fastest legitimate route to the bulk of the corpus.
2. **Gap-fill crawler — Met Collection API.** Wikimedia's ~389K does not cover 100% of the current
   ~470K-object open-access set (coverage gap of roughly 80K–100K objects, see caveats below), and the
   Commons mirror was seeded in 2017 so items added/flagged CC0 more recently may be missing. For the
   remainder, run a **rate-limited crawler directly against the Met Collection API**, respecting the
   documented **80 requests/second** cap (throttle well under that, e.g. 10–20 req/s, for good citizenship
   and to avoid retries). This is unavoidable for the delta since the Met API has no bulk endpoint.
3. **Do not rely on the Google BigQuery/GCS "the_met" dataset** as a primary source — it is a **2017
   snapshot covering only ~200,000 items**, predates the full 470K catalog, and its current availability/
   maintenance status could not be verified (see Option C). Treat as a secondary cross-check only.
4. **Consider the official Hugging Face `metmuseum/openaccess` parquet dataset** as a fast bootstrap for
   prototyping — it ships ~260K rows with images already embedded via `img2dataset`, no crawling needed —
   but the images are **`primaryImageSmall` (reduced resolution, compressed)**, not the full-res
   `primaryImage`, so it's not sufficient for a production-quality corpus by itself.
5. **IIIF is not a viable path today.** The Met has not documented or confirmed IIIF Image API / manifest
   support (open GitHub issue asking about it, closed without a documented resolution).

**Net effect on timeline:** instead of ~470K + 470K = ~940K rate-limited HTTP calls against the Met API,
the realistic plan is ~7,800 batched Wikimedia metadata calls + ~389K image downloads from Wikimedia's CDN
(not subject to the Met's 80 req/s cap) + a much smaller rate-limited Met API crawl (tens of thousands of
calls, not hundreds of thousands) for the gap and for future incremental updates. This turns a multi-day
crawl into roughly a one-to-few-day job, most of which is bandwidth-bound image download, not request-count
bound.

---

## 1. Met's official API docs and terms

- **Base URL / endpoints:** `https://collectionapi.metmuseum.org/public/collection/v1/objects` (list of
  all valid object IDs, optionally filtered by `metadataDate`/`departmentIds`), `.../objects/{objectID}`
  (single object, full metadata incl. `primaryImage`, `primaryImageSmall`, `additionalImages`),
  `.../departments`, and `.../search` (query-based, supports `hasImages=true` filter but requires a search
  term `q`). No batch/multi-ID endpoint exists.
  Source: [Latest Updates | The Metropolitan Museum of Art Collection API](https://metmuseum.github.io/) (accessed 2026-08-11).
- **Rate limit:** documented as **"Please limit request rate to 80 requests per second."**
  Source: [metmuseum.github.io](https://metmuseum.github.io/) (accessed 2026-08-11); corroborated by
  [PublicAPI.dev listing](https://publicapi.dev/metropolitan-museum-of-art-api) (accessed 2026-08-11).
- **No API key / registration required.**
  Source: [metmuseum.github.io](https://metmuseum.github.io/) (accessed 2026-08-11).
- **Images are not bundled in the CSV/metadata dataset.** The `metmuseum/openaccess` GitHub repo (CSV of
  ~470K objects) explicitly ships metadata only; images must be fetched via the Collection API or the
  museum's image-resources channel.
  Source: [github.com/metmuseum/openaccess](https://github.com/metmuseum/openaccess) (accessed 2026-08-11).
- **License / usage terms:** Open Access data and public-domain images are CC0 — free to download, share,
  modify, and redistribute for any purpose without permission. For images not already published or for
  higher resolution than what's on the site, the Met directs requesters to a paid image-licensing partner
  (Art Resource) — i.e., anything not already CC0-flagged in the API is out of scope for this project.
  Source: search-result synthesis citing [metmuseum.org/policies/image-resources](https://www.metmuseum.org/policies/image-resources)
  and [metmuseum.org FAQ](https://www.metmuseum.org/policies/frequently-asked-questions-image-and-data-resources)
  (accessed 2026-08-11; the FAQ page itself returned HTTP 429 on direct fetch during this spike — figures
  above are from search-engine cached snippets, **treat exact FAQ wording as UNVERIFIED** pending a direct
  successful fetch).
- **No documented hotlinking/crawling prohibition** was found for the image host (`images.metmuseum.org`)
  beyond the general 80 req/s API guidance, and no explicit statement on whether that 80 req/s figure
  applies to the JSON Collection API only or also to direct image downloads. **UNVERIFIED**: whether
  image-host requests are counted against the same limit or governed separately — recommend treating them
  as if the same limit applies, to be safe, unless the Met clarifies (contact: openaccess@metmuseum.org,
  per repo docs).
- **Object/image counts vary by source and date** (the collection grows over time): commonly cited figures
  range from **~406,000 images** (older Open Access announcements) to **~492,000 Open Access works**
  (November 2024 press coverage) against a **~470,000+ object** metadata set. Treat the exact current
  count as a moving target; use the live `/objects` endpoint count at crawl time as ground truth.
  Sources: [mymodernmet.com](https://mymodernmet.com/met-online-database-free-art-images/),
  [openculture.com (Nov 2024)](https://www.openculture.com/2024/11/the-metropolitan-museum-of-art-puts-490000-high-res-images-online.html)
  (accessed 2026-08-11).

## 2. IIIF

- A 2018 GitHub issue (`metmuseum/openaccess#14`) asked whether the Met serves IIIF Image API resources —
  the reporter found an image URL pattern resembling an IIIF image path but could not locate a
  corresponding `info.json`, and asked if manifests would be published. **The issue is closed, but the
  fetched thread did not surface a resolution comment confirming IIIF support was ever shipped.**
  Source: [github.com/metmuseum/openaccess/issues/14](https://github.com/metmuseum/openaccess/issues/14) (accessed 2026-08-11).
- No current (2023–2026) documentation, blog post, or changelog entry confirming a live Met IIIF Image API
  or Presentation API manifest endpoint was found via search.
- **Conclusion: IIIF is UNVERIFIED / most likely not available for bulk enumeration.** Do not plan around
  it; if engineering time is cheap, a single manual check of a known object's asset URL for an
  `info.json` sibling would definitively confirm/deny this, but it does not change the recommended plan
  either way.

## 3. Known mirrors / dumps

### 3a. Wikimedia Commons (recommended primary bulk source)

- The Met–Wikimedia partnership (announced Feb 2017) uploaded the initial ~375,000 CC0 public-domain
  images to Wikimedia Commons, with a dedicated Wikipedian-in-Residence continuing the work.
  Sources: [Creative Commons announcement](https://creativecommons.org/2017/02/07/met-announcement/),
  [Wikimedia Foundation announcement](https://wikimediafoundation.org/news/2017/02/07/the-met-public-art-creative-commons/),
  [Wikimedia Diff, July 2017 follow-up](https://diff.wikimedia.org/2017/07/25/met-open-access-initiative/) (accessed 2026-08-11).
- **Current coverage (live count):** `Category:Images from Metropolitan Museum of Art` on Commons reports
  **389,030 files** as of 2026-08-11 (paginated category view; subcategories included).
  Source: [commons.wikimedia.org/wiki/Category:Images_from_Metropolitan_Museum_of_Art](https://commons.wikimedia.org/wiki/Category:Images_from_Metropolitan_Museum_of_Art) (accessed 2026-08-11).
  This is **less than** the ~406K–492K images the Met itself now reports as open-access, implying a
  coverage gap of roughly **17K–100K+ images** (exact delta UNVERIFIED — depends on which Met figure and
  Commons subcategory scope you use; some of the delta may also be non-artwork installation/press photos
  counted in one corpus but not the other).
- **Resolution:** samples observed on the category page ranged roughly **750px–4000px** on the long edge —
  i.e., these appear to be the same full-resolution CC0 files the Met itself publishes, not thumbnails.
  Exact per-file resolution should be verified per-object at download time (Commons `imageinfo` API returns
  width/height). **UNVERIFIED at the aggregate level** — no single authoritative statement found that 100%
  of Commons copies match the Met's `primaryImage` (full-res) rather than `primaryImageSmall`.
- **Bulk enumeration/download mechanics:**
  - MediaWiki Action API (`action=query&list=categorymembers&cmtitle=Category:Images_from_Metropolitan_Museum_of_Art&cmlimit=500`)
    to page through file titles, then `action=query&prop=imageinfo&iiprop=url|size&titles=A|B|C` to batch
    up to 50 titles (more for authenticated/bot accounts) per call for direct file URLs and dimensions —
    this reduces metadata calls from ~470K to roughly **~7,800 calls** for 389K files (389,030 / 50 ≈
    7,781).
    Source: [MediaWiki API:Imageinfo](https://www.mediawiki.org/wiki/API:Imageinfo) (accessed 2026-08-11).
  - No documented hard per-second rate limit for the Action API; official guidance instead calls for
    **serial (not parallel) requests, a contactable `User-Agent`, use of the `maxlag` parameter for
    non-interactive/bot jobs, and exponential backoff on `ratelimited` errors.**
    Source: [MediaWiki API:Etiquette](https://www.mediawiki.org/wiki/API:Etiquette),
    [Wikimedia APIs/Rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits) (accessed 2026-08-11).
  - Tools like **PetScan** and Glamtools statistics/export are also referenced on the category page for
    category-based bulk export, though not deeply investigated in this spike.
    Source: [commons.wikimedia.org/wiki/Commons:Met](https://commons.wikimedia.org/wiki/Commons:Met) (accessed 2026-08-11).
- **License:** files are CC0/public domain per file page (Commons structured data is CC0; surrounding wiki
  text is CC-BY-SA, irrelevant to the images themselves).
  Source: [commons.wikimedia.org/wiki/Commons:Met](https://commons.wikimedia.org/wiki/Commons:Met) (accessed 2026-08-11).

### 3b. Google Cloud (BigQuery + GCS) — secondary/backup only, likely stale

- Google Cloud published a public dataset in **August 2017**: BigQuery tables
  `bigquery-public-data.the_met.objects`, `.images`, and `.vision_api_data`, backed by a GCS bucket
  (`gcs-public-data--met/{object_id}/0.jpg` per the blog's example query).
  Source: [Google Cloud Blog, "When art meets big data" (Aug 7, 2017)](https://cloud.google.com/blog/products/gcp/when-art-meets-big-data-analyzing-200000-items-from-the-met-collection-in-bigquery) (accessed 2026-08-11).
- **Coverage: only ~200,000 items** at publication — well under half of today's ~470K catalog, and the
  post gives no indication the dataset has been refreshed since 2017.
- **Current availability/maintenance status could not be verified** in this spike — recommend a live
  `bq ls bigquery-public-data:the_met` / bucket-existence check before relying on it at all. **Mark
  UNVERIFIED.** Given the age and partial coverage, this should not be a primary path even if still live.
- The Kaggle mirror "[The Met Public Domain Art Works](https://www.kaggle.com/datasets/metmuseum/the-met)"
  page could not be fully fetched (content not rendered by the fetch tool) — description text alone was
  retrieved via search snippet ("select datasets... for unrestricted... use," images "not included" in the
  CSV variant). **Treat Kaggle's exact current contents as UNVERIFIED** without a logged-in direct check.

### 3c. Hugging Face `metmuseum/openaccess` — useful for prototyping, not production-sufficient

- Official Met-org Hugging Face dataset, distributed as parquet, includes an embedded `image` column
  populated via `img2dataset` from the **`primaryImageSmall`** API field (explicitly documented as
  "not full-res, and have some compression"; README points users back to the Collection API for full-res
  or additional views).
  Source: [huggingface.co/datasets/metmuseum/openaccess/blob/main/README.md](https://huggingface.co/datasets/metmuseum/openaccess/blob/main/README.md) (accessed 2026-08-11).
- **Row count: ~260,000** (per Hugging Face dataset viewer), i.e. a subset of the full ~470K objects —
  likely objects for which `primaryImageSmall` successfully resolved at dataset-build time.
  Source: [huggingface.co/datasets/metmuseum/openaccess](https://huggingface.co/datasets/metmuseum/openaccess) (accessed 2026-08-11).
- No confirmed last-updated date was retrievable in this spike (**UNVERIFIED** how current the snapshot is).
- A related `metmuseum/openaccess_embeddings` dataset provides CLIP embeddings over the collection, which
  could be directly useful for Deco's artwork-identification use case (skip re-embedding), but that's a
  separate follow-up, not part of this acquisition question.
- Third-party mirror `miccull/met_museum` (~244K rows, American Wing subset only, variable low-ish
  resolution ~444–625px) is not a full-collection option and not recommended.
  Source: [huggingface.co/datasets/miccull/met_museum](https://huggingface.co/datasets/miccull/met_museum) (accessed 2026-08-11).

## 4. Practical crawl sizing (if/where a direct Met API crawl is still needed)

Assumptions, all sourced from the docs above:
- Documented safe rate: **80 requests/second** against the Collection API (metadata calls); no separately
  documented limit for the static image host — treated conservatively as the same budget.
- Target: up to ~470,000 objects if crawling everything from scratch (worst case, ignoring Wikimedia
  gap-fill); realistically only the **gap not covered by Wikimedia** needs crawling (order of tens of
  thousands, not hundreds of thousands — exact number UNVERIFIED until a live diff of Met object IDs vs.
  Commons file inventory is run).
- Image size: ~1–4MB per primary image (typical JPEG photography of 2D/3D museum objects at web-delivery
  resolution); using a **~2.5MB midpoint estimate**.

**Full from-scratch crawl (470K metadata calls + 470K image downloads), worst case with no Wikimedia use:**
- Theoretical minimum at the documented 80 req/s cap, back-to-back: 940,000 requests ÷ 80 req/s ≈
  **11,750s ≈ 3.3 hours**. This assumes the limit is achievable end-to-end with no backoff, no failures,
  and that image downloads are gated by the same budget — unrealistic for a "safe/respectful" run.
- Realistic conservative crawl at **10–20 req/s** (headroom under the documented cap, room for retries/
  backoff, concurrent writers to storage): 940,000 ÷ 10 to 940,000 ÷ 20 ≈ **13–26 hours** of wall-clock
  time for a single-threaded-equivalent budget. This is very achievable as an unattended overnight/
  weekend batch job with a resumable job queue and per-object retry/backoff.
- Total storage: 470,000 × ~2.5MB ≈ **~1.15TB** (range: 470,000 × 1MB ≈ 470GB to 470,000 × 4MB ≈ 1.83TB).

**Recommended hybrid crawl (Wikimedia bulk + Met API gap-fill):**
- Wikimedia metadata: ~7,800 batched Action API calls (well within Wikimedia's informal etiquette budget)
  — effectively a few minutes to tens of minutes depending on `maxlag` back-off behavior, not
  request-count bound.
- Wikimedia image downloads: ~389,000 files from Wikimedia's CDN, not subject to the Met's 80 req/s limit
  — bandwidth-bound; at even a modest 20 concurrent downloads and ~2.5MB average, wall-clock time is on
  the order of a few hours, dominated by total bytes (~389,000 × 2.5MB ≈ **~950GB**) and available egress
  bandwidth rather than request count.
- Met API gap-fill: crawl only the delta object IDs (estimate tens of thousands) at the same 10–20 req/s
  conservative budget — well under an hour of request time, likely a few hours including image downloads
  for that subset.
- **Combined estimate: a one-time acquisition job completing well within a single day**, versus the
  multi-day worst case of a naive 470K-object-by-object crawl.
- Total storage for the full corpus regardless of source mix: still **~1–1.9TB** (same per-image size
  assumption), so storage provisioning should plan for **~1.2TB** as a working estimate, with headroom to
  1.9TB.

## 5. Open risks / follow-ups

1. **Exact coverage delta between Wikimedia Commons and the Met's current open-access set is UNVERIFIED.**
   Before committing to the hybrid plan, run a one-off diff: pull all current object IDs with
   `hasImages` truthy from the Met API's `/objects` (or `/search?hasImages=true`) list, and diff against
   the Commons category file inventory (matched via the Met's own `objectID`-in-filename convention or the
   Commons "Institution/accession number" structured-data field) to size the actual gap-fill crawl.
2. **Whether images.metmuseum.org enforces its own rate limit separately from the 80 req/s JSON API limit
   is UNVERIFIED.** Confirm empirically with a small conservative test batch, and/or ask
   openaccess@metmuseum.org directly, before assuming a higher safe rate for image-only downloads.
3. **Whether Commons copies are uniformly full-resolution (matching the Met's `primaryImage`) rather than a
   mix of resolutions is UNVERIFIED at scale.** Spot-check a sample against the Met API's `primaryImage`
   dimensions before trusting Commons as the resolution-authoritative source; fall back to the Met API for
   any object where Commons resolution is insufficient.
4. **`bigquery-public-data.the_met` current live/deprecated status is UNVERIFIED** — a 5-minute
   `bq ls`/console check would resolve this and is worth doing opportunistically, but the plan above does
   not depend on it.
5. **Met FAQ page on image/data resources returned HTTP 429 on direct fetch during this spike** — the
   bulk-download and hotlinking guidance cited above comes from search-engine snippets, not a verified
   direct read of the source page. Recommend a follow-up direct fetch (or manual browser check) before
   finalizing any legal/compliance sign-off on the crawl approach.
6. **The Met's own reported image counts differ across sources and time (406K vs. 492K vs. ~470K objects
   in the issue's own framing)** — use a live count from the `/objects` endpoint at execution time as the
   authoritative number rather than any figure in this document.
7. **Licensing scope:** confirm at execution time that every image pulled (from Wikimedia, Hugging Face, or
   the Met API) is CC0/public-domain-flagged in its *current* metadata — the Met's open-access flag is
   per-object and can in principle change; don't assume a 2017 Wikimedia upload snapshot reflects 2026
   rights status without a metadata cross-check against the live Met CSV/API.
