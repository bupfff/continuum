# M22 gate 4: bounce light on by default in `Renderer`

**FAIL** on criterion 3 (settling) in the new view: 3.18× the frames
without bounce against 3×. Cost (1.057–1.117×), accuracy (0.388× the error
without bounce) and the control (M14 gate 11's flights hold the frame
budget) pass. Kept: bounce light on by default in `Renderer`; settling
goes to gate 5 (the frame-period cap starves the bake while pages are
missing, DESIGN §3.4).

| View | 1. Warm frame, bounce on / off (bar 1.2×) | 3. Frames to settle, on / off (bar 3×) | 2. From Cycles, on / off (bar 0.5×) |
|---|---|---|---|
| (a) day | 4.14 / 3.87 ms (**1.069×**) | 290 / 120 (**2.42×**; 1.8 s / 0.6 s) | |
| (b) night | 4.51 / 4.04 ms (**1.117×**) | 1,570 / 898 (**1.75×**; 11.9 s / 6.3 s) | 7.43% / 19.13% (**0.388×**) |
| (c) materials, far side (new) | 8.79 / 8.31 ms (**1.057×**) | 817 / 257 (**3.18×**, FAIL; 8.2 s / 2.5 s) | |

GPU memory with bounce: +188–189 MB (657 against 469 MB, 1,337 against
1,148 MB).

**Control** (criterion 4; M14 gate 11 re-run with this code,
`docs/results/m22-gate4-control.txt`): its criterion 1 holds: means 9.15,
8.94, 9.06 ms (then 9.15, 9.90, 9.41), 99th percentiles 15.60, 14.45, 14.35
ms (then 15.02, 15.32, 14.42), worst 20.42, 18.76, 18.95 ms (then 16.98,
18.06, 17.61; bar 33.3), 8, 3 and 2 frames over 16.7 ms (then 1, 1, 2).
**Its criterion 2 no longer holds:** reading back every 20th frame, views
warm in 1.38–1.50× an undisturbed view's frames (bar 1.25×; then
0.96–1.05×). Undisturbed views warm 20–30% faster than then (143, 280, 320,
195 frames against 171, 357, 415, 274), and reading back 10–18% slower (198,
400, 479, 291 against 179, 359, 405, 264): the frame period, not the GPU
model, now steers the budget on this device (below), and a readback's stall
shows in it.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-20 (binary `experiments/room3d/src/bin/m22_gate4.rs`, stub
and code `377d331`; output `docs/results/m22-gate4-run.txt`). The rehearsal
(views (a) and (b)) passed before the run.

## Method

`RendererOptions::bounce` (on by default) turns on the cache's bounce
light with the ledger for movers and settling, except with moving lights,
the sub-pixel path or movers without hardware ray queries;
`Renderer::is_settled` waits for bounce samples too. The time budget's GPU
model now steers only while a frame's timestamps arrive in order (the
bake's span before the camera passes', each start before its end), at least
one frame in 32; on this device they mostly do not, so the frame period
steers. Each view through `Renderer` with its defaults against the same
with `bounce: false`; warm frames: five rounds of 30 frames each waited
for, off then on, the median ratio; accuracy: the settled linear image over
M10 gate 22's kept pixels against gate 21's Cycles reference (one bounce,
4,096 samples per pixel; not supersampled, so edges add to both errors).

## Disclosures

- Bars set with development known (views (a) and (b): 1.075× and 1.119×,
  0.39×, settling 2.5× and 1.84×).
- Settling with the default time budget depends on the controller; the
  failed view's frames are dominated by the frame-period cap holding the
  bake at its floor while rendering with missing pages overruns 11 ms
  (development without the cap: the night scene 6.4 s with bounce against
  12.0 s).
