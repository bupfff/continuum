# M13 gate 8 in development: the coverage pass toward 10 ms

Development at full size before gate 8's run (`M13_DEV=1M`), on gate 7's
pass (profile stage 18). Speed: settled frames (every wanted page baked,
median of 10) on the gate's headings, through `m13_gate7`'s `M13_PROFILE`
mode; between runs they vary by about ±0.4 ms. Accuracy only on other views
(`M13_ACC=1`: standing turned 20°, 120 m up at 40° turned −20°), as a
multiple of one ray's error.

| Step | Standing | 150 m up | 20 m up |
|---|---|---|---|
| Gate 7's pass | 16.1 ms | 17.2 | 25.4 |
| Instance records once a frame (64 bytes, every object) | 16.3 | 14.5 | 21.5 |
| Records only for objects read (a frame stamp); page records (validity, a homography) | 15.9 | 13.9 | 21.6 |
| Every node's projection once a frame | 13.3 | 13.4 | 20.1 |
| Maps of 32×32 directions, the nearest read (accuracy 0.197×, 0.127×; two of 16×16: 0.197×, 0.134×) | — | — | — |
| Records of 32 bytes | 11.9 | 12.4 | 18.9 |
| Static object data (sphere, quaternion, first layer); one merged pass; each node's largest instance | 11.6 | 12.2 | 18.5 |
| Each tile's leaf instances listed once in workgroup memory; leaves never want pages | 11.9–12.4 | 9.8–10.2 | 16.8 |
| Pages of 1,024 texels (3,723 pages standing) | 10.5–10.7 | 10.2 | 15.2 |
| Pages of 2,048 texels (2,828 pages standing) | 8.8–9.6 | 10.2 | 13.1 |
| Records tell rays whether a hit is large; plain opaque queries for corner and bin rays | 8.8–9.4 | 10.0 | 12.8 |
| Bin rays where the pixel's own corners hit a large object (accuracy 0.183×, 0.127×) | 8.8 | 9.4 | 12.0 |
| Map reads skip bins outside the map's rectangle of non-zero texels (accuracy unchanged) | 9.0–9.4 | 8.9–9.2 | 11.9 |

Tried and not kept: one sample a 2×2 block of bins for instances under 2
pixels' radius (0.159× from 120 m, over the bar; 12.5 ms); 127 roots a tile
(no change); skipping corner rays outside large-capable roots' boxes (no
change: those boxes cover most of the near field); a stack of 24 and a tile
list of 64 (no change); without the nodes' projections 9.37 / 9.08 / 12.33
(standing loses what 150 m gains; kept).

Where the time went (development tests): from 150 m, reading two
directions' maps cost about 2.7 ms more than one; with records and one
direction, texture samples are most of what remains in the pixels' pass
(the frame 9.4 ms with them, 5.6 ms with records tested but not sampled);
corner rays about 0.75 ms and bin rays 1.1–1.8. Standing, the walk was
nearly all of the pixels' pass until pages grew to 2,048 texels.

Pass-boundary GPU timestamps were not used (gate 7: they did not bracket
the work on this device); per-pass times come from `frame_profiled` and no
longer add up to the frame (passes overlap), so decisions used frames.
