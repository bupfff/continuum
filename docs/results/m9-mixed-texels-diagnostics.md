# Moving lights' mixed texels: what they hold (diagnostics, not a gate)

Date: 2026-11-07. Candidate 4 of DESIGN §3.8 (M9 gate 4's follow-up)
proposed a tight bound for mixed texels: the window's true perimeter, no
doubled term, and snapping's risk counted only for ratios near 0 or 1.
Development measurements first, on M9 gate 4's scene (the million-boulder
field, three colored rect lights 1 km away, light 0 moving), at 640×360.

## Almost every texel is stored as exactly 0 or 1

`ledger_dev`: warmed at the standing viewpoint, light 0's channel read back
(8,335 pages, 287,446 texels inside their charts):

| Stored ratio | Share |
|---|---|
| exactly 0 | 37.2% |
| exactly 1 | 60.3% |
| between | 2.5% (below 0.05: 0.6%; above 0.95: 0.4%) |

Moving light 0 at a real sun's speed for 60 frames: the ledger re-baked
71,226 texel jobs a frame, on 3,974 of 6,079 pages checked with mixed
texels (1,612 passed whole).

The proposed bound tightens only texels strictly between 0 and 1. Snapped
texels keep ε in their bound, so it would leave nearly all of these
re-bakes. Not built.

## A "dark" state fails soundness

The bulk is texels in shadow, which count as mixed: M9 gate 2 left out a
dark state, after a re-bake 0.37° later saw 1% of the light in one test.
Tried again: a texel is dark (fixed while the light stays within its guard,
as a clear one) when every bin of its guard cone is wholly covered and its
ratio is 0. M9 gate 4 in development mode (`M9_DEV=1`, 640×360):

- **Speed improved.** At a real sun's speed the flight averaged 14.95 ms
  against 11.94 ms still (1.25×; M9 gate 4's full-size run was 1.6×), with
  934 pages a frame holding mixed texels.
- **Soundness failed.** 8 pages went beyond their bound (0) and beyond ε in
  6 checks, with changes of 1.03–5.35%.

A likely cause, not established: within one mesh, coverage is a signed
running sum of edges, clamped at 1. Overlapping front-facing layers of a
bumpy boulder can therefore sum to "whole" in a bin that still has gaps, and
the light's narrower cone, with finer bins, sees light through them. Taking
"dark" from the guard's bins is then unsafe.

Reverted. What would help: coverage that cannot over-count, such as one
mesh's union instead of its sum where it decides a dark state, or a dark
test at the light cone's own resolution with a margin. Both are for a new
gate.
