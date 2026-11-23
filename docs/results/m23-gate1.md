# M23 gate 1: far bounce hits lit by rays — not run, parked

**Not run.** Development showed the candidate does not touch what keeps
the field from settling, so criterion 1 (the field settles) would fail;
per the gate workflow that is more development, not a gate run.

Stub: `326ad69`. Candidate: `CacheOptions::far_rays` (kept, off by
default, experimental): with one bounce, a hit whose footprint asks for its
chart's top mip reads no page; its direct light is the unshadowed light
times one hardware shadow ray toward each light, its sky one ray
cosine-distributed about its normal.

## Development (`viewer_repro`, the viewer's field at its start view,
1470×923, fixed exposure)

| | Without far rays | With |
|---|---|---|
| With the frame cap (11 ms): bake | at its floor (1,000 units, 13 pages a frame) for 1,200 frames | the same |
| bounce samples incomplete | all (117 a frame) | all (117 a frame) |
| Without the cap: frames 1–500 | resident 25,511 → 111,437; every sample incomplete | identical to the digit |
| frames 1,000–1,450 | settled no; incomplete 0–9, then bursts of ~2,400 | the same |

The far branch is barely taken here: a ground sample's rays rise to
boulders within a few metres (not far) and boulders' rays fall to the
ground (one chart 2 km across, never far), so the misses are near pages.
Without the cap, once every camera pixel has its wanted page (42,320 of
42,320 samples at their mip), bounce samples still re-bake in bursts every
~250 frames (about 2,400 samples, nearly all incomplete, 4,000–10,000
requests), resident pages climbing past 200,000 (above `EVICT_ABOVE`'s
131,072, where pages unused for 64 frames are evicted again: 672 in one
frame). So the field's lack of settling is a cycle among bounce samples'
re-bakes, their near hits' pages and eviction, not distant boulders' pages
(the earlier diagnosis). Next: find what re-bakes complete samples at rest
(the bounce ledger's drain, `sky_lag`'s re-bakes, retries of samples whose
hit pages were evicted), then design for it.
