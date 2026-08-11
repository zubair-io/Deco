# Spike: The Met Collection API — Rate Limits (2026-08-11)

Issue: zubair-io/Deco#3

## Recommendation

The documented "80 requests/second" figure is **stale and not enforced in practice**. The API sits behind Imperva/Incapsula bot protection which throttles/blocks well below that, and multiple 2025–2026 reports confirm this. Do not target 80 rps.

Use these values in `config/deco.toml` `[ingest]`:

**Conservative (recommended default):**
```toml
[ingest]
rate_limit_rps = 1.0      # 1 request/sec, single worker, sequential
max_backoff_s = 60.0
```
- Initial backoff on 403/429: `2s`
- Backoff multiplier: `x2` (exponential), capped at `max_backoff_s = 60s`
- Jitter: full jitter (`random(0, backoff)`) to avoid synchronized retries
- Retry budget: 5 attempts, then dead-letter the object id for a later pass
- Concurrency: 1 in-flight request (no parallel workers hitting `collectionapi.metmuseum.org`)

**Aggressive (only if the conservative profile proves reliable and you want faster crawl throughput; monitor 403 rate closely and fall back immediately if it climbs):**
```toml
[ingest]
rate_limit_rps = 2.0
max_backoff_s = 90.0
```
- Same backoff shape (2s initial, x2 multiplier, full jitter), just a higher steady-state rate.
- This matches the value already checked into `config/deco.toml` today (`rate_limit_rps = 2.0`, `max_backoff_s = 60.0`) — treat that as the ceiling, not a floor, and be ready to drop to 1.0 rps if 403s appear.

Both profiles are well under the ~30 requests/short-window threshold reported in the most recent (Sept 2025) community issue, and use a real browser-like `User-Agent` header (avoid default `curl/x.y` or bare Python `requests` UAs, which correlate with reports of blocking — see below; note this specific causal claim is **UNVERIFIED**, the Met has not published what UA/heuristics trigger the block).

No API key exists for this service, so there is nothing to acquire or configure for auth — rate control is entirely client-side self-throttling plus respecting 403/429 responses with backoff.

## Findings

### 1. Official docs (fetched today, 2026-08-11)

Source: [metmuseum.github.io](https://metmuseum.github.io/) (fetched via WebFetch 2026-08-11)

- **Registration & Authentication**: "No API key or registration is required to access the service."
- **Rate limit text, verbatim, still present today**: "Please limit request rate to 80 requests per second."
- No additional etiquette/backoff guidance is published on the docs site beyond that one line.

This confirms the historically-cited 80 rps figure is still what the docs *say*, unchanged as of today's fetch. It does **not** reflect what the edge/WAF actually allows (see below).

### 2. Community evidence of actual enforced limits

All dates below are as reported in the source; treat GitHub issue/comment timestamps as authoritative for when each report was made.

- **[Issue #60 — "API requests returning 403"](https://github.com/metmuseum/openaccess/issues/60)**, opened **2025-09-01**, still open as of this spike (2026-08-11):
  - Reporter: "Making more than 30 requests results in 403 response on any further requests for an extended period of time (30 seconds or more), which makes it pretty much unusable." Expected 80 rps per the docs; actual behavior is a 403 from **Imperva Incapsula DDoS protection** after ~30 requests, with a captcha/challenge HTML page (`X-Iinfo`, `Set-Cookie: visid_incap_...`, "Incapsula incident ID") returned instead of JSON.
  - Comment, **2025-09-04** (user `ewg118`): "I'm experiencing the same issue."
  - Comment, **2025-09-13** (user `kswope`): "I just managed to download the Egypt objects at one request per second overnight, looks like they are trying to fight off bots." — i.e., ~1 rps sustained overnight worked.
  - Comment, **2026-04-02** (user `amolghode1981`): "I was facing a similar issue. As @kswope mentioned, I added a 0.5-second sleep between the calls [i.e. ~2 rps], and it seems to be working fine."
  - One later comment on this issue (2026-06-14, user `fjcobu14`) contains an off-topic "research verification note" about an unrelated object ID that reads like injected/spam content unrelated to rate limiting; it was disregarded as not credible/relevant.

- **[Issue #18 — "API is blocked"](https://github.com/metmuseum/openaccess/issues/18)**, opened **2019-03-05**: a plain Python `requests`-library script triggered an Incapsula block page (same signature: `_Incapsula_Resource`, "Request unsuccessful. Incapsula incident ID"). Follow-up comment **2019-04-10** describes working around it by copying a browser session's cookies into the harvester script — a fragile, non-API-contract workaround, not something to replicate in production.

- **[Issue #45 — "API currently down?"](https://github.com/metmuseum/openaccess/issues/45)**, **2023-09-19**: maintainer confirms "yes the API is currently down and will not be back up today" — evidence of outright outages independent of client-side rate behavior (not a rate-limit issue per se, but relevant to building resilient retry/backoff and not treating repeated failures as solely a rate problem).

- General pattern across 2019 and 2025–2026 reports: the enforcement mechanism is **Imperva/Incapsula WAF-level bot mitigation**, not a documented token-bucket API rate limiter. It appears to trigger on burst rate (reports converge around "more than ~30 requests quickly" as the trigger) rather than a steady 80 rps figure, and once triggered, imposes a cooldown period (~30s+) during which further requests also 403. This is consistent with WAF/anti-bot behavior generally (background via [ZenRows](https://www.zenrows.com/blog/imperva-incapsula-403) and [ScrapFly](https://scrapfly.io/blog/posts/how-to-bypass-imperva-incapsula-anti-scraping) write-ups on Incapsula 403 behavior, general knowledge not Met-specific).
- **UNVERIFIED**: whether a specific `User-Agent` string (e.g., default `curl` or `python-requests` UA) is itself a triggering signal, independent of request rate. No Met-official source confirms this; it's a plausible inference from how Incapsula/Imperva bot rules commonly work, and from the fact that reports use bare `curl`/`requests` clients, but no one in the threads reviewed tested UA-only as a variable.
- **UNVERIFIED**: exact numeric threshold. "~30 requests" and "works at ~1 rps overnight" / "~2 rps (0.5s sleep) works" are the only concrete data points found, all self-reported, none from Met engineering.

### 3. Image CDN (`images.metmuseum.org`) etiquette

- **[Issue #26 — "Request: remove same origin requirement for API images"](https://github.com/metmuseum/openaccess/issues/26)**, opened **2019-07-22**: reports that API-served images enforce a same-origin/referrer restriction that blocks client-side (browser) apps from loading them cross-origin. This is a **referrer/origin restriction, not a documented rate limit**, and is specific to browser-based hotlinking scenarios, not server-side crawling. No separate numeric rate limit for the image CDN was found in docs or community reports.
- No dedicated rate-limit documentation exists for `images.metmuseum.org` distinct from the collection API's general guidance. **UNVERIFIED**: whether images.metmuseum.org sits behind the same Incapsula layer as collectionapi.metmuseum.org or has independent limits — no direct report tested this in isolation.

### 4. Official Met communications

- The Met's own blog post "[Scaling the Mission: The Met Collection API](https://www.metmuseum.org/perspectives/met-collection-api-2)" (found via search; direct WebFetch returned **HTTP 429 "Too Many Requests" from metmuseum.org itself during this research**, so full text could not be retrieved and is not quoted here) discusses the API's mission (CC0 data, nightly refresh, no-auth access, partnerships with third parties like Wikipedia/Google) per search-result summaries, but no rate-limit-specific numbers were surfaced from what was retrievable. Treat any specific claims from this article as **UNVERIFIED** until it can be fetched directly (retry later; it was itself rate-limiting the fetch attempt, which is at least anecdotally consistent with the museum's web properties being aggressively protected against automated traffic in general).

## Summary for the crawler build

- Ignore the published "80 rps" figure for capacity planning; it does not reflect enforced behavior.
- Build in mandatory backoff/retry handling for 403 (Incapsula block) and 429 responses, not just 5xx — a 403 here is not necessarily a hard auth failure, it may be a transient throttle.
- Prefer a realistic browser-like User-Agent header on the crawler as a low-cost precaution, even though the causal link to blocking is unverified.
- Start at 1 rps single-threaded; this has direct community confirmation (2025-09-13 report) of working reliably over a sustained overnight run. Consider 2 rps as a monitored upper bound only, matching the existing `config/deco.toml` value.
