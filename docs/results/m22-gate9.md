# M22 gate 9: coarse texels average their area — not run, parked

**Not run.** Development (before any rehearsal) showed that criterion 1's
mean bar measured something the candidate does not touch, so the gate
would miss it; per the gate workflow a bar missed in development is not a
gate run. Replaced by gate 10, which measures the same candidate with
direct light and the sky alone (`docs/results/m22-gate10.md`).

Stub: `692cff7` (criteria fixed before the implementation). Candidate:
`CacheOptions::area_texels` (texels in the top three mips of their chart
lit from 16 points over their area by hardware rays, the sky's rays
spread over the same points).

## Development numbers (`altitude_dev`, the scorecard's field 40 m higher,
15° down; 640×360, 4×4 supersampled, fixed bake budget, 600 frames)

| Measure | Current | Candidate |
|---|---|---|
| Against Cycles at 4 bounces (ours one, never settling) | 14.23%, mean 0.0318 / 0.0337 (−5.6%) | 11.32% (0.795×), mean 0.0324 (−3.9%) |
| The same, top five mips averaged instead of three | | 10.82%, mean 0.0324 (−3.9%) |
| Without bounce light here and in Cycles | 13.83%, mean 0.0301 / 0.0313 (−3.8%) | 10.42% (0.753×), mean 0.0304 (−2.9%) |

By distance (rows of the image, ours over Cycles at 4 bounces), current
then candidate: rows 80–120 0.888 → 0.935, 120–160 0.879 → 0.882, the
nearer rows 0.94–0.97 either way. Without our bounce light the rows fall
to 0.81–0.92 of Cycles: bounce light is 6–13% of ours there, one bounce
that never settles against Cycles' four. So the mean within 3% of a
four-bounce reference is a bar on bounce depth and the field's settling
(basics item 2), not on texels.

The field's start view (`viewer_repro`, 1470×923, 600 frames): the flat
dark boulders (direct light zero, their top texel's point on their side
facing partly down) all lit and shaded with the candidate.
