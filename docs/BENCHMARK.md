# Benchmarking on another GPU (NVIDIA)

`experiments/room3d/src/bin/bench.rs` repeats the timing parts of the
gates in one headless run. It needs no window, no Cycles and no assets,
and writes a short report. The M4 baseline is in
`docs/results/bench-m4.md`.

## What it measures (1280×720)

1. Flights over the million-boulder field (M5 gate 8's path, 600 frames
   from an empty cache, three runs each): with the sun; with the sun and
   sky; with three colored lights. Mean, 99th percentile, worst frame,
   fallback, GPU bake and camera time.
2. Warm frames at M5 gate 5's four viewpoints, sun and sky.
3. The sky's warm-up at a fixed budget, with and without the sky later.
4. The textured scene with materials and bounce light (M10 gate 10).
5. Silhouette rays from 150 m up (M7 gate 4): one ray, four quarters, two
   quarters.

About 15 minutes on the M4, plus a first build of 5–10 minutes.
`--quick` runs one flight of each kind and skips part 3.

## Which GPU

Camera rays use Vulkan ray queries (`VK_KHR_ray_query`). These need an
NVIDIA GPU with ray-tracing cores:

- **Suitable:** GeForce RTX 20-series and later, RTX A-series (A4000–A6000,
  RTX 6000 Ada), and the data-center L4, L40/L40S, A10/A10G and A40. The
  T4 (Turing) has RT cores too, but is the slowest of these.
- **Not suitable for the ray-query numbers:** A100, H100 and V100 have no
  ray-tracing cores. The benchmark still runs, through the compute
  fallback, but its camera-ray parts are not comparable.

Typical rentals:

- AWS `g6.xlarge` (L4) or `g5.xlarge` (A10G);
- Google Cloud `g2-standard-4` (L4);
- RunPod, Vast.ai or Lambda instances with an RTX 4090, RTX A6000 or L40S.

The L4 is the common, cheap choice. Check current prices and quotas in the
provider's console; an hour is enough for a full run.

## Steps

1. **Launch** an Ubuntu 22.04 or 24.04 instance with the NVIDIA driver.
   - AWS: a "Deep Learning Base … GPU AMI (Ubuntu)".
   - Google Cloud: a "Deep Learning VM", or install the driver yourself.
   - Container providers (RunPod, Vast.ai): an Ubuntu CUDA image, with
     `NVIDIA_DRIVER_CAPABILITIES=all`. Vulkan needs the driver's graphics
     part, which the default `compute,utility` leaves out.
2. **Copy the repository.** It is private, so copy the working tree rather
   than cloning it (or clone with your own GitHub credentials on the
   machine):

   ```bash
   rsync -az --exclude target --exclude experiments/room3d/out ./ USER@HOST:continuum/
   ```

3. **Run.** The script checks the driver and Vulkan, installs the build
   tools and Rust, builds, and runs:

   ```bash
   ssh USER@HOST 'cd continuum && bash scripts/cloud_bench.sh'
   ```

4. **Fetch the report:**

   ```bash
   scp 'USER@HOST:continuum/experiments/room3d/out/bench_*.txt' experiments/room3d/out/
   ```

5. **Stop or terminate the instance.**

## Troubleshooting

- **"No GPU adapter available"** or a CPU adapter: Vulkan does not see the
  GPU. Check `vulkaninfo --summary`. The script installs
  `libnvidia-gl-<driver major>` when the NVIDIA Vulkan driver is missing.
  In a container, set `NVIDIA_DRIVER_CAPABILITIES=all`.
- **"ray queries: false"** in the report's first line: the GPU or driver
  lacks `VK_KHR_ray_query`. Camera rays then use the compute path, much
  slower on the field.
- The build needs Rust 1.88 or later (edition 2024 with let chains). The
  script installs or updates stable Rust.
