# M10 gate 18: vertex colors' speed (gate 17 re-measured)

**PASS.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Median over five interleaved rounds of the warm-frame ratio, colored to colors ignored, at most 1.1× | **1.004×** | PASS |

| Round | Colors ignored | Colored | Ratio |
|---|---|---|---|
| 0 | 3.02 ms | 3.04 ms | 1.007× |
| 1 | 3.03 ms | 3.04 ms | 1.002× |
| 2 | 3.02 ms | 3.03 ms | 1.004× |
| 3 | 3.03 ms | 3.04 ms | 1.004× |
| 4 | 3.03 ms | 3.04 ms | 1.004× |

Each figure is the median of 30 warm frames, each waited for. Both views
settled in 606 frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criterion committed in `bf7cb77`; the
binary `experiments/room3d/src/bin/m10_gate18.rs` is committed with this
result). Warm machine: it followed `vcolor_dev` runs and a test run; no
stray processes. The kernel is unchanged since gate 17, whose accuracy
result (1.29% from Cycles, 0.02× the error with the colors ignored) stands.

Disclosed: development runs after gate 17 (`m10-gate17.md`) had measured
0.998–1.009× before this gate was written. Gate 17's 1.15× was a transient
during its single measurement of the colored view.
