# M10 gate 15: emissive textures (gate 14 re-run with a one-sided sign)

**PASS.**

| Criterion | Result | Verdict |
|---|---|---|
| 1. Over surfaces: within 10% of Cycles, and at most 0.5× the error with the texture ignored | **6.35%**; ignored 491.34% (**0.01×**); 221 pixels apart | PASS |
| 2. Warm frame at most 1.1× the texture ignored | 2.92 ms against 2.92 ms (**1.00×**) | PASS |

Means: ours −0.32%, ignored +26.18% from Cycles'. All pixels: 116.64% and
184.97%, dominated by the night lights' visible disks. Settled in 604 and
594 frames.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (commit `63f85fc`, which holds the binary
`experiments/room3d/src/bin/m10_gate15.rs`). Cycles: one bounce, 4,096
samples per pixel, 83 s on Metal.

Method and criteria are gate 14's (`m10-gate14.md`). The only change: the
sign has front faces only (`Textured::with_sign_sided(false)`). Gate 14's
double-sided sign had coincident back faces, from which Cycles lit the
sign's face, about 1.22× our emission there. Without them, the means agree
to 0.3%.

What remains is mostly the far ground's undersampled bounce from the sign
(M10 gates 12 and 14).
