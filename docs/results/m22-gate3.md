# M22 gate 3: bounce light that settles

**PASS** on all three criteria in all four views (one not used in
development). A settled still scene's bounce light now matches a fresh
bake: 0.09–0.77% of pages over ε against 2.39–7.09% before, the largest
change 1.38–2.70% of the largest bounce value against 45–95%; settling
takes 1.50–2.00× the frames; accuracy against Cycles unchanged.

| View | 1. Pages over ε (bar 1%), largest change (bar 5%) | Before | 2. Frames to settle (bar 3×) | 3. From Cycles (bar 5%, before + 0.2 points) |
|---|---|---|---|---|
| (a) sun | 13 of 8,277 (**0.16%**), **1.38%** | 212 of 8,862 (2.39%), 68.16% | 181 against 121 (**1.50×**; 0.9 s against 0.5 s) | |
| (b) lamps | 7 of 7,738 (**0.09%**), **2.25%** | 263 of 8,132 (3.23%), 45.16% | 603 against 302 (**2.00×**; 5.1 s against 2.5 s) | **3.080%**, before 3.081% |
| (c) sky | 44 of 7,892 (**0.56%**), **2.70%** | 286 of 8,786 (3.26%), 67.77% | 281 against 158 (**1.78×**; 1.5 s against 0.8 s) | |
| (d) sky, far side (new) | 76 of 9,861 (**0.77%**), **2.06%** | 747 of 10,534 (7.09%), 94.90% | 314 against 179 (**1.75×**; 1.8 s against 0.9 s) | |

Apple M4 (fanless MacBook Air), 1280×720, a fixed bake budget, release
build, run once at full size on 2026-11-20 (binary
`experiments/room3d/src/bin/m22_gate3.rs`, stub committed in `e4b89e1`,
the run's code in `1b6738e`; output `docs/results/m22-gate3-run.txt`). The
rehearsal (views (a)–(c) as measured here) passed before the run with the
same numbers within a few pages.

## Method

DESIGN §3.4, "Bounce light that settles": bounce rays read their hits'
direct light one mip coarser than a bin's footprint; a bin that finds only
a coarser page asks for its page and the sample re-bakes once it may have
arrived (`CacheOptions::bounce_settles`, on by default; "before" is the
same renderer with it off, settled from empty in the same run).
`verify_bounce` re-bakes every resident page's bounce samples in place and
counts pages that change by more than ε (0.01) of the largest bounce value.
Criterion 3 compares the settled image, supersampled 4×4, with M10 gate
21's Cycles reference (one bounce, 4,096 samples per pixel) over the
pixels gate 22 kept (not the lamps' disks and their neighbours).

## Disclosures

- Bars set with development known (views (a)–(c) at this size: 0.11–0.56%
  of pages over ε, changes 1.4–2.7%, settling 1.47–1.96×).
- Settling at a fixed budget. With the default time budget at 1280×720 the
  night scene settles in about 1,200 or about 14,800 frames run to run,
  with or without this change: the frame-period cap holds the bake at its
  floor while frames take longer than 11 ms (DESIGN §3.4).
- Settling bakes more pages (about 16,000 resident against 8,000–9,000)
  and 1.5–3× the bounce samples: hits' pages are now asked for.
- Accuracy against Cycles is unchanged (3.080% against 3.081%): the gate
  is about a settled scene matching its own fresh bake, not about Cycles.
