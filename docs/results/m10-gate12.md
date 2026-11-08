# M10 gate 12: emissive surfaces light their surroundings (scored over surfaces)

**FAIL on the absolute bound; the relative bound and speed pass.**
Emission lighting cuts the error over surfaces from 18.00% to 7.46%
(0.41×), at no cost to the frame, but 7.46% is above 5%. Kept, on by
default.

| Criterion | Result | Verdict |
|---|---|---|
| 1. Over surfaces: within 5% of Cycles, and at most 0.5× the error without emission lighting | **7.46%** (at most 5%); without 18.00% (**0.41×**, at most 0.5×); 202 pixels set apart | FAIL |
| 2. Warm frame at most 1.1× without emission lighting | 2.87 ms against 2.88 ms (**1.00×**) | PASS |

Reported: over all pixels 109.67% and 109.81% (gate 11's numbers). Means
−0.19% (ours) and −1.61% (without) from Cycles'. Settled in 616 and 613
frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (binary `experiments/room3d/src/bin/m10_gate12.rs`,
committed in `dcab7db` before the run). Method and scene are gate 11's
(`m10-gate11.md`), and so is the Cycles reference, reused.

**Over surfaces** follows M5 gate 8, applied to the reference: pixels where
Cycles exceeds 1% of the brightest light's radiance in any channel, and
their eight neighbours, are scored apart. Here that is the lights' disks:
202 pixels.

## Where the rest is

From an error map of the saved images (8-bit, so only indicative):

| Region | Share of the squared error |
|---|---|
| Ground near the panel | 43% |
| Panel's face | 28% (both clip at 1 in 8 bits, so this share is unreliable) |
| Field to the right | 16% |
| Brick wall | 9% |

- **The ground near the panel shows blotches** at the scale of pages,
  while its mean matches Cycles to 0.0003. Bounce light samples 3×3 points
  a page and interpolates between them, with 192 directions each.
- **Near-field light from a large emitter changes faster than that.** Here
  the panel is the main light near it, so the bounce's spatial sampling
  shows, where in daylight (M10 G1, 1.79%) bounce is a small share.

Next, for a new gate: denser bounce samples where an emitter is near, or
emissive quads turned into rect lights (exact shadows through the light
cones).
