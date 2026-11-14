# M13 gate 12 in development: the sub-pixel path lit by the renderer's cache

Development (not the gate) on the million-boulder field with its light,
1280×720, other views than the gate's (standing turned 20°, 120 m up at 40°
turned −20°). Reference: per pixel the mean of 16×16 samples shaded by the
renderer's own path (one ray a pixel, lighting cache settled).

## What is built

- **A variant of the hardware-ray module** (`MaskConeKernel::hw_vc_source`,
  `entry_vc`): the sub-pixel passes with their large hits lit by the cache
  (`vc_hit_color`), its sky and colour scale. Binding lists come from naga's
  analysis of the source (`MaskConeKernel::entry_bindings`), not by hand.
- **Deferred lighting of large hits:** the cache's shading needs most of
  the device's 29 buffers a stage, too many beside a pass's own (corners
  needed 34). A pass that shades a large hit records it (the hit, where its
  ray went, where its colour goes); a lighting pass lights the records from
  the cache and a scatter pass writes the colours to their targets, once
  after the corners and once after the bins' shading.
- **The frames' stamp mirrored** into the colour words, and colour always on
  in the variant (two storage bindings fewer).
- **Small instances' ambient from the sky** (a constant open share, 0.75);
  their direct light is still the model's (the first light's).
- The renderer takes this path with the cache, pixel cones and profile
  stage 20 (`InstancedRenderer`; not yet a `RendererOptions` field).

## Numbers so far

| Configuration | Colour (standing / 120 m) | Settled (standing / 120 m) |
|---|---|---|
| 10,000 boulders, sky, bins lit one by one | 0.391× / 0.442× | 12.1 / 13.1 ms |
| Full size, sky, bins lit one by one | 0.409× / 0.460× | 16.4 / 27.9 ms (120 m unsettled) |
| Full size, sky, bins lit only where corners do not suffice | 0.460× / 0.489× | 16.0 / 16.2–27.1 ms |
| Full size, no sky, the same | — | 15.6 / 16.2 ms |
| The renderer's own path (one ray a pixel, cache), no sky | — | 2.65 / 3.46 ms |

Mean colours agree within 0.5% standing and 2.8% from 120 m.

## Findings

- **Every large hit lit by the cache costs its full shading:** lighting all
  16 bins of every pixel with bin rays (1.6 million calls a frame from
  120 m) took the frame from 16 to 28 ms.
- **With a sky the path may not settle:** from 120 m one page is "baked"
  every frame with no texels and 81 sky samples, for thousands of frames,
  and frames then run 27 ms. The renderer's own path settles in 187 frames
  on the same view. Not yet understood (cache side).
- **Anti-aliasing costs most of the frame:** the renderer's own path takes
  2.7–3.5 ms; the sub-pixel path about 16 ms.

## Next

Lighting for bins that is cheaper than a full cache shading per bin (the
cache's irradiance interpolated from the corners with each bin's own normal
and albedo), the sky's settling with this path, then accuracy and speed
measured again before gate 12's run.
