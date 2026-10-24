# Continuum

A real-time renderer built on an original paradigm: certified incremental cone
rendering. Every visibility question is asked by a cone that answers in bits,
and cached answers are recomputed only when they could have changed by more
than a chosen, invisible error bound.

Status: research stage. The core bets were tested in 2D (M0) and in 3D on the
CPU (M1). The GPU renderer (M2, M3) renders a procedural showcase scene in real
time with moving objects, exact soft shadows and one bounce of indirect light,
and is measured against Blender's Cycles. See [docs/DESIGN.md](docs/DESIGN.md)
for the design and an honest status (§3.6), and [docs/results/](docs/results/)
for measured results.

## Try it: the real-time viewer

```bash
cargo run --release -p viewer
```

Fly through the showcase: a gallery with a colonnade, an arch and a torus-knot
sculpture, lit by one skylight, with a bouncing ball and a spinning cube.

Controls:

- WASD to move, Q/E down and up, Shift to move faster.
- Drag with the left mouse button to look.
- C plays the camera path, P pauses the moving objects.
- V toggles vsync, Esc quits.

The title bar shows the frame rate. `--scene room` opens the M2 test room
instead.

How it renders:

- Direct light from the area light is baked at startup into a world-space
  lighting cache: one mask cone per texel, on every mesh's charts.
- Each frame is a camera ray per pixel plus a cache lookup.
- When objects move, each frame re-bakes exactly the cache blocks they can
  reach, adding only the moving meshes to each texel's cached static coverage.
  Their soft shadows match a full re-bake (M2 gate 5).
- Probes add one bounce of indirect light. Each caches which static surface it
  sees in 768 directions and re-gathers their radiance every frame (M2 gate
  6). Probes on the moving objects are traced from scratch.

On an M4 MacBook Air, on battery, it runs at about 21 ms per frame at
2560×1440. At 1080p the frame is 17.8 ms, just over the 16.7 ms target
(M3 gate 3).

![Continuum (left), Cycles at 4096 samples per pixel (middle), difference ×4 (right)](docs/images/m3-gate3-vs-cycles.png)

The same frame from Continuum and from Cycles, configured to compute the same
thing (Lambertian materials, direct light plus one diffuse bounce): 3.8%
relative RMSE (M3 gate 3).

## Comparing with other renderers

- `cargo run --release -p room3d --bin export_showcase` writes the showcase to
  `experiments/room3d/out/showcase/`, ready to rebuild in another renderer (for
  a UE5 side-by-side, for example):
  - `showcase.obj`: one object per mesh, Z up, meters, per-vertex normals.
  - `showcase.json`: linear albedos, the light (a one-sided rectangle with its
    radiance), and the camera path and movers sampled every 30 frames.
  - When rebuilding it, use non-metallic, fully rough materials with no
    specular, and fix exposure (no auto-exposure), so differences come from
    lighting.
- `cargo run --release -p room3d --bin cycles_calibration` checks the Cycles
  pipeline on a scene with a known answer. The M3 gate 3 binary renders the
  references itself. Both need Blender 4.2 or later; set `BLENDER` if it is
  not in `/Applications`.

## Layout

- `crates/continuum`: the library: direction masks (binary and continuous), the
  error ledger, 3D pyramid cones with exact per-bin rasterization, a BVH, and
  meshes with charts and smooth normals (`mesh::Surface`).
- `crates/continuum-gpu`: wgpu compute backend (mask cones in WGSL, lighting cache, live renderer).
- `apps/viewer`: the real-time viewer.
- `experiments/flatland`: 2D experiments that test the core bets (milestone M0).
- `experiments/room3d`: 3D experiments on triangle scenes (milestones M1–M3),
  the showcase scene, and the Cycles reference pipeline (`cycles/`).
- `docs/`: design document and experiment results.

## Running

```bash
cargo test --release
cargo run --release -p flatland --bin gate1
cargo run --release -p flatland --bin ledger
cargo run --release -p flatland --bin gate4
cargo run --release -p flatland --bin gate5
cargo run --release -p flatland --bin gate6
cargo run --release -p flatland --bin gate6b
cargo run --release -p flatland --bin gate7
cargo run --release -p room3d --bin m1_gate1
cargo run --release -p room3d --bin m1_gate2
cargo run --release -p room3d --bin m1_gate3
cargo run --release -p room3d --bin m1_gate4
cargo run --release -p room3d --bin render
cargo run --release -p room3d --bin m2_gate1
cargo run --release -p room3d --bin render_gpu
cargo run --release -p room3d --bin m2_gate2
cargo run --release -p room3d --bin m2_gate3 -- capture   # once, before kernel changes
cargo run --release -p room3d --bin m2_gate3
cargo run --release -p room3d --bin m2_gate4
cargo run --release -p room3d --bin m2_gate5
cargo run --release -p room3d --bin m2_gate6
cargo run --release -p room3d --bin render_indirect   # side-by-side image
cargo run --release -p room3d --bin m3_gate1
cargo run --release -p room3d --bin m3_gate2
cargo run --release -p room3d --bin m3_gate3           # needs Blender
cargo run --release -p room3d --bin render_showcase    # three frames of the path
```

![GPU render at 1080p](docs/images/m2-gpu-render-1080p.png)

The same scene rendered on the GPU at 1080p (Apple M4, 189 ms, brute force with
no hierarchy yet; see `docs/results/m2-gpu-render.md`).

![First 3D render](docs/images/m1-first-render.png)

First 3D render (direct light only, CPU, 480×360). Left: mask cones,
deterministic and noise-free. Right: 16 jittered light samples per pixel at
about the same cost. Shadows are black because there is no indirect light yet.

Each run prints a results table and writes images to `experiments/flatland/out/`:

- `sheet.png` (gate 1): top row renders, bottom row signed error maps (red too
  dark, blue too bright).
- `ledger_<scenario>.png` (gates 2–3): one row per key frame; columns are the
  lit result, tiles recomputed that frame (orange), the certified error bound,
  and the actual error. `_stress` versions use a deliberately huge ε so the
  bound and error columns have something to show.
- `gate5_<visibility>_<charging>.png` (gate 5): the same four columns for the
  sky scenario at ε = 1%.
- `gate6_bounce.png` (gate 6): the same four columns for one-bounce indirect
  light (attempt 3 charges, ε = 1%).
