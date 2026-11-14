# M13 gate 11 in development: colour, third

Development at full size (`M13_DEV=1M`) on gate 10's pass before gate 11's
run. Speed on the gate's views (settled frames, median of 10); accuracy only
on other views (standing turned 20°, 120 m up at 40° turned −20°), as one
ray's multiple.

| Step | Accuracy (standing / 120 m) | Settled (standing / 150 m) |
|---|---|---|
| Gate 10 as run (gate views) | 0.383× / 0.465× | 16.05 / 18.03 ms |
| Buried test without a branch; shading constants per record | 0.384× / 0.463× | 15.94 / 16.73 |
| Sun visibility cached in world space (corners, bin rays, records) | 0.383× / 0.463× | 14.67 / 15.15 |
| Background bins shaded one by one through the cache (lean pass, misses listed) | 0.317× / 0.366× | +6 ms from 150 m (off) |
| Same, a visibility a quarter | 0.342× / 0.406× | +2.4 ms from 150 m (off) |
| Visibility at the pixel's centre where the background's corners disagree | 0.329× / 0.389× | 14.99 / 16.05 |
| One compute pass; corner colours in a word; claims cleared; resolve fast paths; fewer writes | 0.330× / 0.389× | 14.16 / 14.74 |
| Corners whose hit is the background trace no background ray; spread 0.15 | 0.332× / 0.391× | 13.85–13.92 / 14.70–14.73 |

Three repeats of the settled timing before the last step: 14.04–14.06 ms
standing, 14.69–14.87 from 150 m.

Lessons:
- **Rays are most of colour's cost**, and the ground's shadow rays at
  every corner were the largest part from 150 m; caching visibility in
  world space removed them in settled frames.
- **A cache lookup costs about as much as a sample of work** (hash, a
  bucket's line from a 64 MB table): five million a frame (every
  background bin at shadow edges) cost 4–6 ms even with no miss. One a
  pixel at shadow edges (400,000) costs about 0.3 ms and buys most of the
  accuracy (0.46× → 0.39× from 120 m).
- **Threadgroup memory costs occupancy:** the bins' transmittance or the
  background's coefficients moved there made the pixels pass slower (+2 ms
  and +0.3 ms); dead code that names a workgroup array still reserves it.
- **The resolve costs by the general case:** summing the common cases at
  once (every bin the background) saved more than vectorising the loop.
- Tried and dropped: a bounding-disk test before map fetches (slower);
  shading normals without barycentrics on flat faces (no change); 16 boxes
  a tile for corner rays (no change); bin rays shading only objects the
  corners do not suffice for (worse accuracy, no speed).
