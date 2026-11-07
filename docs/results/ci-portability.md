# CI: the first runs off Apple Silicon

Everything before 2026-11-06 ran only on an Apple M4 (Metal). The first CI
runs (GitHub Actions, `.github/workflows/ci.yml`) put the library on Vulkan
and DX12. Both are software adapters, so this is about correctness, not
speed. Not a gate: these record what the runs found.

## Linux, Vulkan on lavapipe

lavapipe (llvmpipe, LLVM 20.1.2, 256 bits) passes every test, the GPU tests
included:

| Test | Result on lavapipe | On Apple M4 |
|---|---|---|
| Kernel cones against the CPU reference | 1.64e-5 of the light's weight (bound 1e-4) | 3.29e-6 |
| Per-pixel render against the golden image from Apple M4 | 0.091% relative RMSE, no pixel off by more than 2% of the mean | (the golden image) |
| Cache, settled, against per-pixel cones | 10.01% (threshold 13%; settled after 36 frames) | 10.01% |
| `.glb` round trip | 0.0076% | 0.0044% |
| Compute rays against hardware rays | 0.0023% (lavapipe has ray queries) | 0.0082% |

Kernel build: 8.5 s, every pipeline between 0.00 and 1.89 s
(`kernel_setup` example).

## Two portability bugs found and fixed

1. **Naga's SPIR-V output panicked on the kernel** ("Expression [69] is not
   cached"), so no entry point with light cones compiled for Vulkan.
   - The cause: `bound_cone(&out.c, …)` passed a pointer to a struct's
     member. WGSL allows pointer arguments only to whole variables; naga's
     Metal output took it.
   - The fix: `bound_cone` takes and returns the cone by value. The golden
     image is unchanged.
   - `continuum-gpu/tests/shaders.rs` now translates every entry point to
     SPIR-V, HLSL and MSL on the CPU, so any machine catches this. It fails
     on the old source.
2. **A Windows checkout broke the kernel's source patch.** Git converted
   the shader's line endings to CRLF, so the `\n` patterns of the subgroup
   patch did not match.
   - `.gitattributes` keeps `*.wgsl` as LF.
   - `kernel_source` converts CRLF, for users who vendor the crate on
     Windows.

## Windows, DX12

WARP (the "Microsoft Basic Render Driver" software adapter) cannot build the
kernel:

| Pipeline | Build time on WARP |
|---|---|
| `main` | 31.85 s (run 4), 33.23 s (run 5) |
| `render` | 35.84 s, 41.23 s |
| `render_cached` | 30.59 s, 35.71 s |
| `mark_blocks` | 0.02 s |
| `rebake` | out of memory (`0x8007000E`) after 2.2 and 3.0 minutes |

DXC alone, on naga's HLSL for the same entry points (run 5, HLSL 2021):

| Entry point | DXC time |
|---|---|
| `main` | 2.34 s |
| `render` | 5.18 s |
| `rebake` | 0.67 s |
| `vc_bake` | 1.73 s |

`vc_shade` failed in that run, because HLSL 2021 rejects a ternary operator
on vectors. wgpu-hal calls DXC with HLSL 2018, and CI now does the same.

Run 6 compiled all 20 entry points with DXC at HLSL 2018, with no errors.
The total was 23.3 s; `render_cones` was slowest at 4.76 s, then `render`
at 3.77 s and `shade` at 3.54 s. Half the entry points took under 1 s.

So the time and the memory go to WARP's own shader compilation, not to DXC.
Real Windows GPUs remain untested. Their drivers compile DXIL themselves,
so their build times are unknown. CI now checks that every entry point
compiles with DXC at HLSL 2018, as wgpu calls it. The GPU tests run on
Linux only.

## Caveats

- Software adapters say nothing about speed, or about vendor-specific
  driver bugs.
- NVIDIA and AMD remain untested. That needs the cloud GPU benchmark
  (`docs/DESIGN.md` §3.7, item 6).
