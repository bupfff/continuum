# M10 gate 4: free re-coloring with bounce light, corrected settle

**PASS.** Re-coloring the sun on a settled view with bounce light baked
**0 pages** in the next 30 frames, using gate 1's scene and procedure
(settle, supersample 4×4, settle, re-color).

Apple M4, 1280×720, 2026-11-07, run once (binary
`experiments/room3d/src/bin/m10_gate4.rs`, criterion in `21a4cf4`).

## What was wrong before

Gates 1 and 3 failed this check with 794 and 522 pages. Their `settle`
looked at the last 32 frames already read back. Right after another settle,
those were that settle's own quiet frames, so it returned after one frame,
and the new view's requests arrived inside the re-coloring window. They
were requests for pages evicted while the view sat at other sub-pixel
offsets.

The same flaw rendered each sub-pixel offset of gates 1–3's supersampling
one frame after the switch. Now `settle` waits for 32 quiet frames rendered
after it is called.

## Reported

Supersampled with the corrected settle, against the same Cycles references:

- gate 1's scene: **1.44%** with one bounce (gate 1 measured 1.79%);
- gate 2's materials: **9.01%** (gate 2 measured 8.94%).
