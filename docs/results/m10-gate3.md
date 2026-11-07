# M10 gate 3: bounce light and reflections without request cascades

**FAIL on criterion 1 (re-coloring), which gate 4 showed was the check's own
procedure. Criteria 2 and 3 pass.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Re-coloring bakes nothing (gate 1's procedure) | 522 pages baked | FAIL (a procedure bug, see gate 4) |
| 2. Materials and bounce light settle in at most 1.5× the frames of Lambertian without | 584 against 523 (**1.12×**; it was 4,266 against 621) | PASS |
| 3. No loss of accuracy | gate 1's scene **1.45%** (was 1.79%), gate 2's **9.01%** (was 8.94%) | PASS |

Apple M4, 1280×720, 2026-11-07, run once (binary
`experiments/room3d/src/bin/m10_gate3.rs`, criteria in `bca6639`).

## The fix

Pages requested only for light at bounce or reflection hits were baked with
bounce samples of their own. Those samples' rays requested more pages, and
so on through the scene. Now:

- Such requests are secondary: planned after pixels' requests, and baked
  without bounce samples. The fill clears what a previous page in the slot
  left.
- A page a pixel shows without samples asks for them, and the CPU
  schedules them.
- Pages read through reflections are marked as used.

Disclosed: the fix and a development check (`M10_DEV=1` of gates 1 and 2,
settling in 602 and 585 frames) came before the stub.

Gate 1's reference was rendered again: its job file changed when materials
were added, though not its content.
