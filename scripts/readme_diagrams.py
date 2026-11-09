"""Draws the README's explainer diagrams (SVG) into docs/images/how/.

    python3 scripts/readme_diagrams.py

- mask-cone.svg (animated): one cone from a surface point to an area light;
  an occluder slides through it; the cone's 64 direction bits flip, and the
  share of light that arrives follows. The floor's soft shadow moves with it.
- three-ways.svg: a soft shadow by a shadow map, by sampled rays, and by a
  mask cone.
- cache.svg: lighting cached on surface pages; a move re-bakes only the
  pages it can change.

Geometry is computed here, so the animation's parts stay in step.
"""

import math
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "how")
os.makedirs(OUT, exist_ok=True)

BG = "#0f141c"
INK = "#f2eee6"
DIM = "#a9aeb8"
SUN = "#ffc15e"
SHADE = "#5b6577"
BLOCK = "#e5674e"
FONT = "font-family=\"Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif\""


def card(w, h, body, title):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}">
<title>{title}</title>
<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>
{body}
</svg>
"""


def text(x, y, s, size=15, fill=INK, weight=400, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{s}</text>'
    )


# ---------------------------------------------------------------- mask cone
def mask_cone():
    W, H = 900, 420
    period = 10.0  # seconds per loop
    n = 200  # samples per loop
    light = (140.0, 340.0, 50.0)  # x0, x1, y
    floor_y = 340.0
    px = 240.0  # the surface point
    occ_y, occ_w = 205.0, 70.0
    s = (floor_y - occ_y) / (floor_y - light[2])  # occluder's height share
    amp = 160.0

    def xo(t):  # occluder centre over the loop
        return 240.0 + amp * math.sin(2 * math.pi * t)

    # The occluder's projection from the point onto the light line.
    def band(t):
        c = 240.0 + (xo(t) - px) / s
        half = occ_w / 2 / s
        return c - half, c + half

    cols = 8
    gx, gy, cell = 590.0, 112.0, 24.0

    def column_open(t):
        lo, hi = band(t)
        out = []
        for c in range(cols):
            centre = light[0] + (c + 0.5) * (light[1] - light[0]) / cols
            out.append(not (lo <= centre <= hi))
        return out

    times = [k / n for k in range(n)]
    # Keyframes, held between samples (steps), so bits flip like bits.
    css = []
    css.append(
        f".occ{{animation:occ {period}s linear infinite}}"
        f".sh{{animation:sh {period}s linear infinite}}"
        f".gauge{{animation:gauge {period}s step-end infinite;transform-origin:{gx}px 0}}"
    )
    occ_k = "".join(
        f"{100 * t:.2f}%{{transform:translateX({xo(t) - 240:.2f}px)}}" for t in times
    ) + f"100%{{transform:translateX(0px)}}"
    sh_k = "".join(
        f"{100 * t:.2f}%{{transform:translateX({(xo(t) - 240) / (1 - s):.2f}px)}}"
        for t in times
    ) + f"100%{{transform:translateX(0px)}}"
    css.append(f"@keyframes occ{{{occ_k}}}")
    css.append(f"@keyframes sh{{{sh_k}}}")
    gauge_k = ""
    last = None
    for t in times:
        v = sum(column_open(t)) / cols
        if v != last:
            gauge_k += f"{100 * t:.2f}%{{transform:scaleX({v:.3f})}}"
            last = v
    gauge_k += f"100%{{transform:scaleX({sum(column_open(0)) / cols:.3f})}}"
    css.append(f"@keyframes gauge{{{gauge_k}}}")
    for c in range(cols):
        frames = ""
        last = None
        for t in times:
            v = 0 if column_open(t)[c] else 1
            if v != last:
                frames += f"{100 * t:.2f}%{{opacity:{v}}}"
                last = v
        frames += f"100%{{opacity:{0 if column_open(0)[c] else 1}}}"
        css.append(
            f".c{c}{{animation:c{c} {period}s step-end infinite}}@keyframes c{c}{{{frames}}}"
        )

    # The floor's shadow at the occluder's centre position: the share of
    # the light each floor point cannot see.
    xs = [40 + 4 * k for k in range(101)]

    def blocked(x):
        a = x + (240 - occ_w / 2 - x) / s
        b = x + (240 + occ_w / 2 - x) / s
        lo, hi = min(a, b), max(a, b)
        return max(0.0, min(hi, light[1]) - max(lo, light[0])) / (light[1] - light[0])

    # Wide enough to slide: the shadow moves 1 / (1 − s) times the occluder.
    span = amp / (1 - s) + 200
    gxs = [240 - span + 2 * span * k / 120 for k in range(121)]
    stops = "".join(
        f'<stop offset="{k / 120:.4f}" stop-color="#000" stop-opacity="{0.85 * blocked(x):.3f}"/>'
        for k, x in enumerate(gxs)
    )

    body = [f"<style>{''.join(css)}</style>"]
    body.append(
        "<defs>"
        f'<linearGradient id="shadow" x1="{240 - span}" x2="{240 + span}" gradientUnits="userSpaceOnUse">{stops}</linearGradient>'
        '<filter id="glow" x="-50%" y="-200%" width="200%" height="500%"><feGaussianBlur stdDeviation="8"/></filter>'
        '<clipPath id="scene"><rect x="30" y="30" width="430" height="360"/></clipPath>'
        f'<clipPath id="grid"><rect x="{gx}" y="{gy}" width="{cols * cell}" height="{8 * cell}"/></clipPath>'
        "</defs>"
    )
    # Left: the scene.
    body.append(text(30, 30, "One cone, from a point to the whole light", 17, INK, 600))
    body.append('<g clip-path="url(#scene)">')
    body.append(
        f'<rect x="{light[0]}" y="{light[2] - 6}" width="{light[1] - light[0]}" height="12" fill="{SUN}" filter="url(#glow)" opacity="0.8"/>'
    )
    body.append(
        f'<rect x="{light[0]}" y="{light[2] - 5}" width="{light[1] - light[0]}" height="10" rx="3" fill="{SUN}"/>'
    )
    body.append(
        f'<polygon points="{px},{floor_y} {light[0]},{light[2] + 5} {light[1]},{light[2] + 5}" fill="{SUN}" fill-opacity="0.10" stroke="{SUN}" stroke-opacity="0.45" stroke-width="1.5"/>'
    )
    body.append(f'<rect x="40" y="{floor_y}" width="420" height="40" fill="#1c2431"/>')
    body.append(
        f'<rect class="sh" x="{240 - span}" y="{floor_y}" width="{2 * span}" height="40" fill="url(#shadow)"/>'
    )
    body.append(f'<line x1="40" y1="{floor_y}" x2="460" y2="{floor_y}" stroke="{DIM}" stroke-opacity="0.5"/>')
    body.append(
        f'<rect class="occ" x="{240 - occ_w / 2}" y="{occ_y - 6}" width="{occ_w}" height="12" rx="2" fill="{SHADE}" stroke="{DIM}" stroke-opacity="0.6"/>'
    )
    body.append(f'<circle cx="{px}" cy="{floor_y}" r="6" fill="{INK}"/>')
    body.append("</g>")
    body.append(text(350, light[2] + 4, "area light", 13, DIM))
    body.append(text(250, floor_y + 28, "surface point", 13, DIM))
    body.append(text(48, 300, "its soft shadow", 13, DIM))
    # Right: the bits.
    body.append(text(gx - 20, 30, "What the cone records", 17, INK, 600))
    body.append(text(gx - 20, 54, "Each bit is a direction toward the light:", 13, DIM))
    body.append(text(gx - 20, 72, "open, or blocked by something in between.", 13, DIM))
    for c in range(cols):
        x = gx + c * cell
        body.append(f'<rect x="{x}" y="{gy}" width="{cell}" height="{8 * cell}" fill="{SUN}" fill-opacity="0.85"/>')
        body.append(f'<rect class="c{c}" x="{x}" y="{gy}" width="{cell}" height="{8 * cell}" fill="{SHADE}"/>')
    for k in range(9):
        body.append(
            f'<line x1="{gx + k * cell}" y1="{gy}" x2="{gx + k * cell}" y2="{gy + 8 * cell}" stroke="{BG}" stroke-width="2"/>'
        )
        body.append(
            f'<line x1="{gx}" y1="{gy + k * cell}" x2="{gx + cols * cell}" y2="{gy + k * cell}" stroke="{BG}" stroke-width="2"/>'
        )
    gy2 = gy + 8 * cell + 34
    body.append(text(gx - 20, gy2 - 10, "Light that arrives = open bits ÷ 64", 14, INK, 500))
    body.append(f'<rect x="{gx}" y="{gy2}" width="{cols * cell}" height="14" rx="7" fill="#1c2431"/>')
    body.append(f'<rect class="gauge" x="{gx}" y="{gy2}" width="{cols * cell}" height="14" rx="7" fill="{SUN}"/>')
    body.append(text(gx - 20, gy2 + 44, "One query, no random samples: no noise,", 13, DIM))
    body.append(text(gx - 20, gy2 + 62, "and the same answer every frame.", 13, DIM))
    return card(W, H, "\n".join(body), "How a mask cone finds a soft shadow")


# ---------------------------------------------------------------- three ways
def three_ways():
    W, H = 900, 330
    body = []
    panels = [
        ("Shadow map", "Most real-time engines", 0),
        ("Sampled rays", "Ray-traced shadows", 300),
        ("Mask cone", "Continuum", 600),
    ]
    for title, who, x0 in panels:
        body.append(text(x0 + 24, 34, title, 17, INK, 600))
        body.append(text(x0 + 24, 54, who, 13, DIM))
        lx0, lx1, ly = x0 + 90, x0 + 210, 84
        body.append(f'<rect x="{lx0}" y="{ly - 4}" width="{lx1 - lx0}" height="8" rx="3" fill="{SUN}"/>')
        body.append(f'<rect x="{x0 + 128}" y="150" width="44" height="9" rx="2" fill="{SHADE}"/>')
        body.append(f'<line x1="{x0 + 30}" y1="236" x2="{x0 + 270}" y2="236" stroke="{DIM}" stroke-opacity="0.5"/>')
        p = (x0 + 165, 236)
        body.append(f'<circle cx="{p[0]}" cy="{p[1]}" r="5" fill="{INK}"/>')
    # Shadow map: a depth image from the light's centre; softness by blur.
    x0 = 0
    for k in range(9):
        x = x0 + 96 + 13.5 * k
        body.append(f'<rect x="{x}" y="100" width="11" height="11" fill="none" stroke="{DIM}" stroke-opacity="0.6"/>')
    body.append(f'<line x1="{x0 + 150}" y1="{84}" x2="{x0 + 165}" y2="236" stroke="{SUN}" stroke-opacity="0.6" stroke-dasharray="4 4"/>')
    body.append(text(x0 + 24, 270, "One depth picture from the light; the soft", 13, DIM))
    body.append(text(x0 + 24, 288, "edge is a blur, so its width is a guess.", 13, DIM))
    body.append(text(x0 + 24, 306, "Fast. Leaks and shimmers at distance.", 13, DIM))
    # Sampled rays: random rays, some blocked.
    x0 = 300
    import random

    rnd = random.Random(7)
    for k in range(8):
        lx = x0 + 90 + 120 * (k + 0.2 + 0.6 * rnd.random()) / 8
        px, py = x0 + 165, 236
        hit = x0 + 128 <= px + (lx - px) * (236 - 154) / (236 - 84) <= x0 + 172
        if hit:
            t = (236 - 154) / (236 - 84)
            body.append(f'<line x1="{px}" y1="{py}" x2="{px + (lx - px) * t:.1f}" y2="154" stroke="{BLOCK}" stroke-width="1.6"/>')
        else:
            body.append(f'<line x1="{px}" y1="{py}" x2="{lx:.1f}" y2="88" stroke="{SUN}" stroke-opacity="0.75" stroke-width="1.6"/>')
    body.append(text(x0 + 24, 270, "A few random rays per pixel, then a", 13, DIM))
    body.append(text(x0 + 24, 288, "denoiser and past frames to hide the noise.", 13, DIM))
    body.append(text(x0 + 24, 306, "Accurate in the limit. Lag and smearing.", 13, DIM))
    # Mask cone: the whole light at once, bits.
    x0 = 600
    body.append(
        f'<polygon points="{x0 + 165},236 {x0 + 90},88 {x0 + 210},88" fill="{SUN}" fill-opacity="0.12" stroke="{SUN}" stroke-opacity="0.5"/>'
    )
    for c in range(8):
        covered = 2 <= c <= 4
        body.append(
            f'<rect x="{x0 + 214 + 8 * c}" y="104" width="7" height="30" fill="{SHADE if covered else SUN}"/>'
        )
    body.append(text(x0 + 226, 150, "bits", 11, DIM))
    body.append(text(x0 + 24, 270, "One cone covers the whole light; what blocks", 13, DIM))
    body.append(text(x0 + 24, 288, "it is recorded as bits. The open share is the", 13, DIM))
    body.append(text(x0 + 24, 306, "light that arrives: no blur, no noise.", 13, DIM))
    return card(W, H, "\n".join(body), "Three ways to get a soft shadow")


# ---------------------------------------------------------------- the cache
def cache():
    W, H = 900, 384
    body = []
    body.append(text(30, 36, "Light is cached on surfaces, and a move re-bakes only what it can change", 17, INK, 600))
    gx, gy, n, cell = 40, 70, 12, 22
    old, new = (120, 210), (230, 170)
    # The region the move can change: both positions, grown by the shadow's
    # reach (a coarse picture of the conservative bound).
    rx0, ry0, rx1, ry1 = 80, 120, 320, 260
    for i in range(n):
        for j in range(n):
            x, y = gx + i * cell, gy + j * cell
            hit = x + cell > rx0 and x < rx1 and y + cell > ry0 and y < ry1
            fill = BLOCK if hit else "#26303f"
            op = 0.55 if hit else 1.0
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="2" fill="{fill}" fill-opacity="{op}"/>')
    body.append(f'<circle cx="{old[0]}" cy="{old[1]}" r="18" fill="none" stroke="{INK}" stroke-dasharray="4 4" stroke-opacity="0.7"/>')
    body.append(f'<circle cx="{new[0]}" cy="{new[1]}" r="18" fill="{SHADE}" stroke="{INK}"/>')
    body.append(
        f'<line x1="{old[0] + 20}" y1="{old[1] - 7}" x2="{new[0] - 24}" y2="{new[1] + 9}" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)"/>'
    )
    body.insert(
        0,
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>',
    )
    body.append(f'<rect x="{gx}" y="{gy + n * cell + 12}" width="14" height="14" rx="2" fill="{BLOCK}" fill-opacity="0.55"/>')
    body.append(text(gx + 22, gy + n * cell + 24, "re-baked this frame", 13, DIM))
    body.append(f'<rect x="{gx + 170}" y="{gy + n * cell + 12}" width="14" height="14" rx="2" fill="#26303f"/>')
    body.append(text(gx + 192, gy + n * cell + 24, "kept: cannot have changed", 13, DIM))
    tx = 360
    lines = [
        ("Pages of light on every surface", 600, INK),
        ("Each page holds the light from each lamp per unit", 400, DIM),
        ("brightness, so changing a light's color or power", 400, DIM),
        ("re-weights the cache and bakes nothing.", 400, DIM),
        ("", 400, DIM),
        ("Only what can change is recomputed", 600, INK),
        ("When an object moves, the pages whose light it could", 400, DIM),
        ("touch are re-baked in the same frame; everything else is", 400, DIM),
        ("provably unchanged and kept. No light is ever shown stale.", 400, DIM),
        ("", 400, DIM),
        ("No blending of past frames", 600, INK),
        ("Each frame is computed from the cache as it is now, so", 400, DIM),
        ("there is no ghosting or lag behind moving things.", 400, DIM),
    ]
    y = 84
    for s, wgt, col in lines:
        if s:
            body.append(text(tx + 40, y, s, 15 if wgt == 600 else 14, col, wgt))
        y += 21
    return card(W, H, "\n".join(body), "The lighting cache")


for name, svg in [("mask-cone.svg", mask_cone()), ("three-ways.svg", three_ways()), ("cache.svg", cache())]:
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)
    print(name, len(svg) // 1024, "KB")
