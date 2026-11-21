"""The split-sum table for GGX specular (Smith height-correlated
visibility): for n·v and roughness on a 32 × 32 grid (both 0 to 1; α =
roughness²; n·v at least 0.001, roughness at least 0.02), the directional
albedo's two terms A and B (albedo = F0 × A + F90 × B, Schlick's Fresnel),
by importance sampling the GGX distribution with 4,096 Hammersley samples.
Prints WGSL for `mask_cones.wgsl`'s `DFG` table.

    python3 scripts/dfg_table.py > table.wgsl
"""

import math

N = 32
SAMPLES = 4096


def radical_inverse(i):
    i = ((i << 16) | (i >> 16)) & 0xFFFFFFFF
    i = ((i & 0x55555555) << 1) | ((i & 0xAAAAAAAA) >> 1)
    i = ((i & 0x33333333) << 2) | ((i & 0xCCCCCCCC) >> 2)
    i = ((i & 0x0F0F0F0F) << 4) | ((i & 0xF0F0F0F0) >> 4)
    i = ((i & 0x00FF00FF) << 8) | ((i & 0xFF00FF00) >> 8)
    return i / 4294967296.0


def dfg(nv, roughness, n=SAMPLES):
    a = roughness * roughness
    a2 = a * a
    vx, vz = math.sqrt(max(0.0, 1.0 - nv * nv)), nv
    sum_a = sum_b = 0.0
    for i in range(n):
        u1, u2 = (i + 0.5) / n, radical_inverse(i)
        phi = 2.0 * math.pi * u2
        cos_t = math.sqrt((1.0 - u1) / (1.0 + (a2 - 1.0) * u1))
        sin_t = math.sqrt(1.0 - cos_t * cos_t)
        hx, hz = sin_t * math.cos(phi), cos_t
        vh = vx * hx + vz * hz
        lz = 2.0 * vh * hz - vz
        if lz <= 0.0 or vh <= 0.0:
            continue
        vis = 0.5 / (lz * math.sqrt(nv * nv * (1.0 - a2) + a2) + nv * math.sqrt(lz * lz * (1.0 - a2) + a2))
        g = vis * lz * 4.0 * vh / hz
        fc = (1.0 - vh) ** 5
        sum_a += (1.0 - fc) * g
        sum_b += fc * g
    return sum_a / n, sum_b / n


if __name__ == "__main__":
    rows = []
    for j in range(N):
        r = max(j / (N - 1), 0.02)
        row = []
        for i in range(N):
            nv = max(i / (N - 1), 1e-3)
            a, b = dfg(nv, r)
            row.append("vec2<f32>(%.5f, %.5f)" % (a, b))
        rows.append("    " + ", ".join(row))
    print("const DFG: array<vec2<f32>, %d> = array<vec2<f32>, %d>(" % (N * N, N * N))
    print(",\n".join(rows))
    print(");")
