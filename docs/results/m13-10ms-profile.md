# M13: where the coverage pass's time goes, for the 10 ms bar (development)

Not a gate: a design step's profile after M13 gates 5–7, at full size (the
million-boulder field without its ground, 1280×720), on the gates'
headings, each view settled (every wanted page baked). Each pass submitted
and waited for alone (`frame_profiled`; each carries about 0.15–0.2 ms of
submission and waiting), median of 10. `M13_DEV=1M M13_PROFILE=1 cargo run
--release -p room3d --bin m13_gate7`.

| View | Settled frame | Corners | Roots | Bins | Bin args | Bin rays | Pixels' pass | Page args | Bakes |
|---|---|---|---|---|---|---|---|---|---|
| Standing (1.7 m) | 16.14 ms (10,224 pages) | 1.13 | 0.90 | 0.56 | 0.18 | 1.45 | **12.29** | 0.22 | 0.18 |
| 150 m up, 45° down | 17.23 (no pages) | 1.48 | 0.90 | 0.71 | 0.23 | 1.83 | **13.15** | 0.24 | 0.21 |
| 20 m up, 10° down | 25.38 (13,805 pages) | 1.80 | 1.18 | 0.62 | 0.20 | 4.29 | **17.48** | 0.24 | 0.21 |

- **The pixels' pass is 72–76% of every view.** At 10 ms, it would have to
  fall to about 5 ms with everything else trimmed to about 5 (today 4–8).
- From 150 m up (earlier skipping parts, gate 6's development): within the
  pixels' pass about 4 ms is the walk, 3 the instances' tests, 5 their
  maps' set-up and samples. Each instance's set-up runs once per subgroup
  (32 pixels) that touches it.
- 20 m up, the bin rays cost 4.3 ms: many pixels are near large boulders
  at that height.
- Settled frames vary by about 15% between measurements of the same view
  (gate 7: 21.61 and 25.09 ms at its start).

Levers to weigh in the design (none tried): an instance's set-up shared by
a tile (64 pixels) rather than a subgroup; fewer bins where coverage is
smooth; corner and bin rays at a coarser rate where nothing is large; the
quality dial (§3.3) trading the 4×4 bins for speed.
