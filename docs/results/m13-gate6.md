# M13 gate 6: hierarchy nodes holding coverage, as view-dependent coverage pages (coverage only)

**PASS** on all four criteria.

| View | Coverage error, ours | One ray | Ratio | Frame, settled | Pages wanted |
|---|---|---|---|---|---|
| Standing | 1.84% | 8.69% | **0.212× (at most 0.25×): PASS** | **16.12 ms (at most 20): PASS** | **10,224 (at most 32,768): PASS** |
| 150 m up | 7.81% | 57.99% | **0.135× (at most 0.15×): PASS** | **17.11 ms (at most 20): PASS** | **0: PASS** |

**4. Reuse in motion: PASS.** A 10 s walk at 1.5 m/s at standing height
along the view, 60 frames a second, from a cold pool, at most 20 pages
baked a frame: at the ten checks, 0.213×–0.224× the error of each frame's
one-ray image from its own 16×16 reference (bar 0.3×).

| Walk frame | 60 | 114 | 168 | 222 | 276 | 330 | 384 | 438 | 492 | 546 |
|---|---|---|---|---|---|---|---|---|---|---|
| Pages read | 1,184 | 2,250 | 3,319 | 4,385 | 5,424 | 6,471 | 7,485 | 8,453 | 9,384 | 10,227 |
| Ratio | 0.216× | 0.224× | 0.224× | 0.217× | 0.219× | 0.215× | 0.213× | 0.214× | 0.216× | 0.216× |

Gate 5 on the same views: 0.219× and 0.135×, 38.25 and 20.35 ms. Mean
coverage: standing, reference 0.2951, ours 0.2953, one ray 0.2956; 150 m up,
0.2909, 0.2924, 0.2910.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (stub with the criteria committed in the tenth session;
the method in `679ba41`, before the run). Speed timed after settling and
before the references (median of 10 frames, each waited for). Machine free
of other work; control just before the run, gate 4's path at full size:
118.9 ms standing, 105.9 ms from 150 m (122.0 and 105.8 before gate 5's run).

Reported: standing settled in 25 frames at 512 bakes a frame (10,224 pages,
84 million rays; no node failed to fit its page); object nodes a walking
pixel 15.9 (at most 940; gate 5: 34.1, at most 3,275); from 150 m no node
fits a page. On the walk: 11,100 pages baked (18.5 a frame, about 152,000
rays a frame), 1.70 million requests in all (most repeated each frame while
waiting); the largest shift read 0.2499 bin (bound 0.25: pages are read only
within it, so every frame read no stale page); frames 34.0 ms median, 141
ms worst (pages lag demand while walking; unbaked nodes are walked, as in
gate 5). Memory: 16 MB of texels and 2.6 MB of headers for the pool, plus
the page table and request marks (about 9 MB for the field's 1.2 million nodes)
and the item ranges (13 MB).

## Method (DESIGN §3.4, part D, on gate 5's pass)

- **A page** holds a hierarchy node's coverage seen from the bake eye, on
  the plane through the node's centre facing it, a texel a bin at its
  distance: 512 8-bit texels (32×16 to 4×128, the smallest width that fits
  the node's projection, centred), each the share of 4×4 rays through it
  that meet the node's own instances.
- **Which nodes:** the highest node on a path whose projection fits a page
  shape and that holds at least 8 instances, found in the tiles' roots pass
  and in the pixels' walks. If its page is resident and valid it is read
  (per open bin, bilinearly where the bin's ray crosses the plane, as an
  independent occluder); otherwise it is asked for (a mark per node on the
  GPU) and walked as in gate 5, and nodes below it want no page.
- **Valid** while its contents shift by at most a quarter of a bin since
  the bake (2 f |ΔE⊥| L / D²) and its texels are 0.7–1.4 bins wide.
- **Bakes:** after the pixels' pass, one invocation takes at most the
  budget's requests and gives each a page (its own, or the next in a ring
  over the 32,768-page pool, evicting the owner); eight workgroups bake a
  page, a texel a lane, by hardware rays from the node box's entry, opaque
  queries stepping past other nodes' instances. Budget 512 pages a frame
  while settling a view, 20 on the walk (development: 0.087 ms a page, so
  1.75 ms a frame).

## Disclosed

- Developed at full size on the gate's scene; speed on the gate's views,
  accuracy and the walk only on other views
  (`docs/results/m13-gate6-development.md`). Changes from the stub's
  method before the run are listed in the binary's doc comment.
- The bake budget is a page count measured to fit 2 ms in development, not
  a time measured on each frame of the walk.
- Gate 5's pass also got faster for this gate before the run (bin rays
  compacted, bounding spheres, stored root boxes): 150 m up 20.35 → 16.5
  ms in development.

## What the numbers say

- **The horizon is solved when settled.** Pages cut the standing frame from
  38 to 16 ms; the longest walk from 3,275 nodes to 940, and accuracy is
  slightly better (0.212× against 0.219×).
- **In motion pages lag.** Walking at 1.5 m/s, the view asks for thousands
  of pages a frame (mid-field pages of 8 boulders go stale within metres)
  and 20 bakes a frame keep up with a fraction; frames ran 34 ms. Accuracy
  holds because unbaked nodes are walked, but a flight at frame rate needs
  cheaper bakes, bakes ordered by what they save, or pages that stay valid
  longer (bigger, farther nodes first).
- **150 m up** uses no pages and runs 17 ms; the 10 ms bar needs more there
  (corner and bin rays about 5 ms, map reads and the walk the rest).
