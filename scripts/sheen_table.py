"""The directional albedo of glTF's sheen (KHR_materials_sheen: the Charlie
distribution of Estevez and Kulla 2017 with their visibility's fitted
lambda, as the extension's specification gives them) for a white sheen,
on a 32 × 32 grid of n·v (columns, 0 to 1, at least 0.001) and sheen
roughness (rows, 0 to 1, at least 0.02): the integral over the hemisphere
of f(l, v) cos θ_l, by stratified uniform sampling (64 × 64 cells, each
sample at its cell's centre), at most 1. Prints WGSL for `mask_cones.wgsl`'s
`SHEEN_E`.

    python3 scripts/sheen_table.py > table.wgsl
"""

import math

N = 32
CELLS = 64


def lambda_helper(x, alpha):
    t = (1.0 - alpha) ** 2
    a = 21.5473 + (25.3245 - 21.5473) * t
    b = 3.82987 + (3.32435 - 3.82987) * t
    c = 0.19823 + (0.16801 - 0.19823) * t
    d = -1.97760 + (-1.27393 + 1.97760) * t
    e = -4.32054 + (-4.85967 + 4.32054) * t
    return a / (1.0 + b * x ** c) + d * x + e


def lambda_sheen(cos_theta, alpha):
    if abs(cos_theta) < 0.5:
        return math.exp(lambda_helper(cos_theta, alpha))
    return math.exp(2.0 * lambda_helper(0.5, alpha) - lambda_helper(1.0 - cos_theta, alpha))


def visibility(nl, nv, roughness):
    alpha = max(roughness, 1e-6) ** 2
    v = 1.0 / ((1.0 + lambda_sheen(nv, alpha) + lambda_sheen(nl, alpha)) * (4.0 * nv * nl))
    return min(max(v, 0.0), 1.0)


def distribution(roughness, nh):
    alpha = max(roughness, 1e-6) ** 2
    inv = 1.0 / alpha
    sin2 = max(1.0 - nh * nh, 0.0)
    return (2.0 + inv) * sin2 ** (0.5 * inv) / (2.0 * math.pi)


def albedo(nv, roughness):
    vx, vz = math.sqrt(max(0.0, 1.0 - nv * nv)), nv
    total = 0.0
    for i in range(CELLS):
        # Uniform in cos θ, so each cell has the same solid angle.
        lz = (i + 0.5) / CELLS
        lr = math.sqrt(1.0 - lz * lz)
        for j in range(CELLS):
            phi = 2.0 * math.pi * (j + 0.5) / CELLS
            lx, ly = lr * math.cos(phi), lr * math.sin(phi)
            hx, hy, hz = lx + vx, ly, lz + vz
            hn = math.sqrt(hx * hx + hy * hy + hz * hz)
            nh = hz / hn
            total += distribution(roughness, nh) * visibility(lz, nv, roughness) * lz
    return total * 2.0 * math.pi / (CELLS * CELLS)


if __name__ == "__main__":
    rows = []
    for j in range(N):
        r = max(j / (N - 1), 0.02)
        # At most 1: the grid undersamples the sharpest lobes (roughness
        # near 0) at grazing angles.
        rows.append(
            "    " + ", ".join("%.5f" % min(albedo(max(i / (N - 1), 1e-3), r), 1.0) for i in range(N))
        )
    print("const SHEEN_E: array<f32, %d> = array<f32, %d>(" % (N * N, N * N))
    print(",\n".join(rows))
    print(");")
