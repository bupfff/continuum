# M14 gate 13: gate 12 with a fixed bake budget

**PASS.** Coarser rock levels carrying the finest surface's normals (gate
12's change), measured again with every renderer at a fixed bake budget so
the equality check does not depend on the machine's timing.

| Criterion | View 0 | View 1 | View 2 | Bar | Result |
|---|---|---|---|---|---|
| 1. Streamed against resident, finest's normals / levels' own | 1.484% / 2.287% (0.649×) | 1.246% / 1.945% (0.641×) | 1.425% / 2.538% (0.561×) | at most 0.75× | PASS |
| 2. Levels 1 and 2 produced, per rock | 2.31 ms against 2.66 ms (0.87×) | | | at most 1.5× | PASS |
| 3. Resident images apart | 0.000000% | 0.000000% | 0.000000% | identical | PASS |

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-13 (criteria and binary `7ec1c71`, before the run). Gate
12's scene and renderers (256 unique rocks, sun only, a 2.6-million-triangle
streaming budget against every finest level resident, the three flights'
last views averaged over 4×4 sub-pixel offsets, a first pass discarded),
with `frame_target_ms` 0 (the default 20,000-unit budget each frame).

**Disclosed (before the run):** gate 12's diagnostic with the fixed budget
gave the same numbers; the development mode (36 rocks) passed (0.455–0.499×,
resident images identical).

## What the numbers say

- Streamed rocks now shade a third to nearly half closer to their finest
  levels, at no production cost; the finest levels are untouched.
- Left, as DESIGN §3.8 item 4 named: a level choice that also bounds the
  normals' error, and the same for imported meshes' levels (only the rocks'
  analytic shapes carry the finest normals today).
