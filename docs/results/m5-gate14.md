# M5 gate 14: wide light cones (no stall near a large light)

**FAIL**: the stall is gone (criterion 1) and ordinary lights are
untouched (criterion 3), but accuracy near the light fails in one view
(criterion 2).

| | Near (8 m up, 15° down) | 20 m up, 20° down |
|---|---|---|
| 1. Frames from an empty cache (after 10) | **PASS**: max 32.8 ms, 99th percentile 18.7 ms, mean 10.9 ms | **PASS**: max 24.9 ms, 99th percentile 19.2 ms, mean 11.3 ms |
| 2. Per-pixel cones vs Cycles, surfaces | **FAIL**: 4.10% (bar 3%; 10,796 pixels apart) | **PASS**: 1.52% (562 pixels apart) |
| Settled | 2,826 frames after the 600 | 2,748 frames after the 600 |
| Reported: the cache's settled image vs per-pixel cones | 7.25% | 13.03% |

- **Criterion 1** bars: no frame over 100 ms after the first 10, and the
  99th percentile at most 33.3 ms. First frames, near: 31.3, 6.1, 27.9 ms;
  20 m up: 54.2, 9.1, 9.5 ms.
- **Criterion 3: PASS.** M5 gate 6's three coloured lights at the field's
  camera, per-pixel cones, 1280×720: no pixel differs from a second kernel
  with every guard off.
- **Reported:** the panel alone against Cycles, near: 648.48% (see "What
  fails").

Before the guards, the same scene gave frames of 12–24 s on 10,000
boulders, and at full size a development run waited 12 hours.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-15 (a million boulders), Cycles 4,096 samples a pixel at
320×180, direct light only. The run's lines are in
`docs/results/m5-gate14-run.txt`.

## The problem (DESIGN §8, "Wide light cones")

A rectangle light's cone projects the light onto a plane parallel to it
at the receiver's distance h, with 8×8 bins uniform in that plane. A light
s across gives a window about s/h wide. For a 40 m panel 5 cm above the
ground under the boulders, that is ±2,000: bins no longer prefilter, and a
cone walks to triangles (about 13 ms a cone). Points just outside the
light, near its plane, see grazing slivers with walks just as long.

## The guards (as run)

- **Clip:** a window wider than 8 units of h and wider than 30° in angle
  (a corner more than 30° from its centre) keeps the 8×8 square nearest
  the light's axis (the foot point clamped into it). Its visible share of
  its own unshadowed light stands for the whole light's (`wide_visible`).
  A sphere light's window is capped the same way, all of it the disk.
- **Cap:** each light cone's walk stops after 4,096 nodes and triangles,
  512 where its window is wider than 8 units in the plane. A long walk is
  one invocation's serial work, so a single long cone sets the frame's
  length however few are baked. A grazing sliver stopped at its cap counts
  as blocked, a clipped cone as open. Sky cones are not capped.
- **The caller's wait:** time spent in `InstancedRenderer::wait` counts
  toward the frame period the cache's bake budget adapts to, as a frame
  call's own wait does. With harnesses that wait after each frame, the
  budget otherwise stayed at its largest: this device's pass timestamps
  mostly read zero.

All three are pipeline constants plus one renderer change. A
development setter (`MaskConeKernel::set_wide_cones(false)`) turns the
first two off.

## What fails, and why

In development, a probe of the panel alone (development view, 4×4 a
pixel) beside Cycles showed two errors:
- **Boulders resting on the panel come out dark.** Cycles shows them lit.
  A receiver's light is densest just beneath it, inside its own boulder's
  footprint, so the kept window is blocked by the boulder itself. The
  rest of the panel, which lights it in Cycles, is outside the window.
- **Some far boulders come out lit.** Cycles has them shadowed by the
  field. Their sliver cones walk fully, under the cap, and find too
  little cover at grazing angles.

Totals over surfaces: ours 70.4, Cycles 99.7.

**Tried in development:**

| Variant | Panel alone, surfaces' sum (Cycles 99.7) | Frames on 10,000 boulders | Gate 6's scene |
|---|---|---|---|
| Wider kept window (32 units) | dark boulders stayed dark | — | — |
| Caps counted as blocked (4,096) | 0.2 | — | — |
| Caps counted as open (4,096) | 65.7 | worst 95–101 ms | — |
| Lower cap (1,024) for every cone | — | worst 61–67 ms | 2,506 pixels changed |
| Coarser prefilter for clipped cones | — | slower: bins never fill, so walks never stop early | — |

Near-light accuracy and bounded frame time pull against each other
within mask cones: these receivers need walks of thousands of nodes at
this prefilter.

**Next:** rays. Texels whose light window is wide take that light's
visibility from hardware rays to points of the light, converging over
frames as the sky's samples do (M5 G13). This is "a slower path inside
the paradigm". The guards stay as the stall's backstop.

## Scene and views

The million-boulder field, its sun, and a light panel: 40 m square, 5 cm
above the ground under the boulders, facing up, 130–170 m ahead of the
field's camera along its line, radiance (0.5, 0.45, 0.35). No sky.
Prefilter 1 bin with object-level prefiltering, sky prefilter ×8,
hardware rays, cache defaults (M5 gate 6's settings).

## Disclosed

- **Amended before any run:**
  - the method grew the angular condition, the caps and the caller's wait
    (development after the stub);
  - criterion 3's second kernel turns every guard off, not only the clip.
- **The 20 m view is where the stall was found** (development). The
  near-view failure was expected from development on other views (4.72%
  there with the caps counted open).
- **The machine:** the run went alone. The Claude desktop app used about
  a quarter of a core.
