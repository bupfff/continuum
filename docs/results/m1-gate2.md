# M1 gate 2: hierarchy culling and scaling

Date: 2026-10-23 (overnight run). Each receiver's cone must find the triangles
that can touch it. Testing every triangle costs time linear in the scene; a
bounding volume hierarchy (`continuum::bvh`) should not.

Pass criterion, fixed before the first run: (1) hierarchy culling gives the same
mask results as testing every triangle, within 1e-6 of the light's weight, at
every receiver; (2) hierarchy culling time per receiver in the largest scene is
at most 10× that of the gate 1 room (for about 100× the triangles). The mask's
own cost is reported, not judged.

**Verdict: FAIL on the letter of (1); (2) passes.** The hierarchy finds exactly
the same candidate triangles as brute force at every receiver (zero
candidate-set mismatches). The 2.5e-6 difference in the sphere scene comes from
summing about 800 triangles' coverage in a different order. My 1e-6 tolerance
should have scaled with the number of summed triangles. It is recorded as a fail
regardless.

## Results

64×64 floor receivers, single-threaded. Per-receiver times. Reproduce with
`cargo run --release -p room3d --bin m1_gate2`.

| Scene | Triangles | BVH build | Brute-force culling | BVH culling | Nodes visited | Candidates | Mask cost | Max difference | Set mismatches |
|---|---|---|---|---|---|---|---|---|---|
| Gate 1 room | 408 | 0.1 ms | 10.1 µs | 4.8 µs | 50 | 28 | 1.3 µs | 1.7e-7 | 0 |
| + sphere (40k triangles) | 40,008 | 7.5 ms | 183.7 µs | 25.3 µs | 782 | 827 | 16.9 µs | 2.5e-6 | 0 |
| + 3,000 boxes | 36,408 | 5.8 ms | 218.5 µs | 30.4 µs | 661 | 473 | 19.3 µs | 1.1e-7 | 0 |

## What this shows

1. **Culling scales**: about 90–100× more triangles costs 5–6× more culling
   time (6.3× worst), and 7× less than brute force on the big scenes.
2. **The mask's cost now grows with the triangles inside each cone**, 1.3 →
   17–19 µs as candidates go from 28 to 470–830 (about 20–40 ns per triangle).
   For dense geometry that is too much per receiver. This is what the
   prefiltered hierarchy (pillar 3.4) is for: when a hierarchy node is smaller
   than a bin, use a precomputed coverage summary instead of rasterizing every
   triangle inside it.
3. **Culling still dominates** (25–30 µs vs 17–19 µs for the mask). Neighboring
   receivers' cones are nearly identical, so a tile of receivers should traverse
   the hierarchy once with a frustum enclosing all their cones, then filter per
   receiver. This is the tile-frustum traversal from the original design.

## Next

- Tile-shared traversal (one hierarchy walk per 8×8 receiver tile).
- Prefiltered hierarchy nodes for dense geometry.
- Use a tolerance that scales with summation length when comparing float results.
