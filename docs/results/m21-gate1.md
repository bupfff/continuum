# M21 gate 1: motion blur

**PASS** on all four criteria, in every view; the view not used in
development passed narrowly.

| Criterion | (a) trucking sideways | (b) turning | (c) boulders crossing | (d) dollying while turning, foliage (new) |
|---|---|---|---|---|
| 1. Cost at 1280×720 (bar) | **+1.426 ms** (1.7) PASS | **+1.301 ms** (1.7) PASS | **+0.553 ms** (0.8) PASS | +1.233 ms (reported) |
| 2. From the sampled shutter: ours / the still image (bar 0.35×) | 1.04% / 14.35% (**0.072×**) PASS | 2.12% / 11.22% (**0.189×**) PASS | 1.30% / 4.78% (**0.271×**) PASS | 5.42% / 15.82% (**0.343×**) PASS |
| 3. From Cycles: ours / the sampled shutter (bar 1.12×) | 8.22% / 8.24% (**0.998×**) PASS | 8.39% / 8.15% (**1.030×**) PASS | 7.46% / 7.28% (**1.026×**) PASS | 9.71% / 8.85% (**1.097×**) PASS |
| 4. The mean of ours against the sampled shutter's (bar 1%) | **−0.004%** PASS | **−0.057%** PASS | **−0.025%** PASS | **+0.112%** PASS |

The still images against Cycles: 17.40%, 15.50%, 9.28%, 19.20%. Means
against Cycles (ours, the sampled shutter, still): (a) −2.41%, −2.41%,
−2.41%; (b) −2.46%, −2.41%, −2.40%; (c) −2.46%, −2.43%, −2.44%; (d)
−2.86%, −2.97%, −2.97%. Every view settled. Cycles took 17, 14, 14 and 28
s (Metal, 4,096 samples). Timing rounds: (a) 4.170–4.197 against
2.723–2.773 ms; (b) 4.070–4.085 against 2.772–2.778; (c) 3.442–3.456
against 2.900–2.918; (d) 4.116–4.137 against 2.883–2.892.

Apple M4 (fanless MacBook Air), 640×360 (timed at 1280×720), release
build, run once at full size on 2026-11-16 (commit `0cce509`; binary
`experiments/room3d/src/bin/m21_gate1.rs`, stub committed in `2368c25`;
output `docs/results/m21-gate1-run.txt`), after three minutes with no
game or indexing running and one more minute idle. Timings first.

## Method

DESIGN §3.4, "Motion blur": `MotionBlur` (shutter 0.5 centred on the
frame, the longest smear 10% of the height). Each pixel's segment from its
hit point's motion since the previous camera and movers' transforms; a
gather along the tile's fastest motion and the pixel's own, a tap covering
the pixel when the pixel lies on its segment, in three groups by depth.
Ours and the still image at 4×4 sub-pixel offsets, each settled.
References: the shutter sampled outright in our renderer (64 instants
evenly across it, the camera and movers placed at each, settled, sub-pixel
offsets cycling through 4 × 4); Cycles' motion blur (the camera and moving
instances keyed before, now and as far again, linearly).

## Disclosures

- The bars were set with development numbers known (0.072×, 0.189×,
  0.271×; 0.998×, 1.029×, 1.025×; means within 0.06%; +1.44, +1.30,
  +0.54 ms). View (d) was not used in development and passed within 2%
  and 2% of criteria 2 and 3's bars: the near plant, whose motion fans out
  as the camera dollies and turns, keeps more of its leaves than the
  sampled shutter does (the gather samples two lines; motion across them
  is missed) (`experiments/room3d/out/m21_gate1_d_*.png`).
- In view (c) the boulders' shadows on the ground do not smear: the
  ground does not move. Image-space motion blur cannot see that.
- The floor against Cycles is the shading's (our still image 7.08% from
  Cycles' on this scene without motion); criterion 3 measures against the
  sampled shutter's own distance from Cycles for that reason.
- Not measured: deforming geometry, moving lights, motion through glass
  layers, motion blur with depth of field.
