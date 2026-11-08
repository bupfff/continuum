# M10 gate 21: bounce light with more than three lights

**FAIL: no criterion has a result.** The run crashed at criterion 3, and
the lines for criteria 1 and 2 were lost.

Apple M4 (fanless MacBook Air), 1280×720, release build, run once at full
size on 2026-11-08 (stub with the criteria committed as `a413582`; binary
`experiments/room3d/src/bin/m10_gate21.rs`).

- **The crash.** Re-coloring the fourth light clears the bounce samples
  (`encoder.clear_buffer`), and the buffer lacked the copy-destination
  usage that clearing needs: a wgpu validation error, then a panic.
- **The lost lines.** Criteria 1 and 2 printed their results before the
  crash, but the command showed only the output's last ten lines, and the
  binary writes its report file at the end. Nothing was kept but the
  Cycles reference (`out/m10_gate21_cycles`).

The fix (the usage added) and a re-run with unchanged criteria are M10
gate 22 (`m10-gate22.md`), with its output saved as it runs.
