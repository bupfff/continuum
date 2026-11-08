# M10 gate 10: materials accuracy (M10 gate 2 re-run)

**PASS.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Within 6% of Cycles, and at most 0.6× the Lambertian render's error | **5.85%** (gate 2: 8.94%); Lambertian 119.37% (0.05×) | PASS |
| 2. Warm frame at most 2× the Lambertian scene's | 4.85 ms against 2.89 ms (**1.68×**) | PASS |

Reported: without reflection rays 10.72% (mean −3.76%); materials change
the image's mean by −2.08%; ours +0.94% on the mean against Cycles. The view
settled in 597 frames (gate 2: 4,266; the eviction fix after gate 2 and the
sky later, M5 gate 9, came since).

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-07 (commit `a695a9b`; binary
`experiments/room3d/src/bin/m10_gate10.rs`, committed in `081c9d5` before
the fix; criteria and scene are gate 2's, pre-registered in `e6aafb2`). The
Cycles reference is gate 2's, reused (same job).

## Diagnosis (disclosed: seen before this gate was written)

Diagnostic runs of `m10_gate2` (`M10_DEV=ss`) at full size, accuracy only,
against gate 2's cached reference:

| Code | From Cycles |
|---|---|
| Today's, before any change | 9.04% |
| Reflection rays kept above the geometric surface | 9.04% |
| Eight rays across the glossy lobe (not kept) | 8.85% |
| D and V at the material's roughness, normalization once | **5.85%** |

- **Where the error was.** Pixels where Cycles exceeds 1 (sun highlights on
  the metal boulders, 0.24% of pixels) carried 75% of the squared error.
  The rest of the image was 4.67% from Cycles.
- **What was wrong with the highlights.** Their positions agreed with
  Cycles (2,177 of 2,217 pixels). Their energy was 3.5% low overall and
  5.4% low where Cycles exceeds 10.
- **The cause.** `specular_light` evaluated GGX's D and V at the widened
  roughness α′ *and* multiplied by Karis's energy normalization (α/α′)².
  That lowers a small light's peak twice. Karis (2013) evaluates D at α and
  normalizes once.
- **After the fix**, highlights are 1.9% low and carry 40% of the squared
  error.

Also kept: a reflection ray that a shading or mapped normal turns below the
geometric surface is kept just above it (as Cycles'
`ensure_valid_reflection`). It made no measurable difference here.

## Caveats

- One scene, one sun; rect lights larger than the sun are not measured
  again.
- M10 gate 5 (normal maps, 3.63%) was measured before this fix.
- What remains (4.7% outside highlights) includes reflections: one ray,
  sharp, with black streaks in the boulders' crevices that Cycles does not
  show. Integrating the lobe with eight rays changed little.

## Prior art

GGX (Walter et al. 2007), the representative point and its normalization
(Karis 2013). No novelty claim.
