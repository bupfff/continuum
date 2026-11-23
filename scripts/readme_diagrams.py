"""Draws the README's explainer diagrams (SVG) into docs/images/how/.

    python3 scripts/readme_diagrams.py

- mask-cone.svg (animated): one cone from a surface point to an area light;
  an occluder slides through it; the cone's 64 direction bits flip, and the
  share of light that arrives follows. The floor's soft shadow moves with it.
- three-ways.svg: a soft shadow by a shadow map, by sampled rays, and by a
  mask cone.
- cache.svg: lighting cached on surface pages; a move re-bakes only the
  pages it can change.
- scorecard.svg: the basics scorecard against Cycles, at the start of
  phase 1 and now.
- texels.svg: a coarse texel lit from one point, and from points over its
  area (M22 gate 10).
- frame.svg: what one frame does.

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
    tx = 340
    lines = [
        ("Pages of light on every surface", 600, INK),
        ("Each page holds each lamp's light per unit of", 400, DIM),
        ("brightness: changing a light's color or power", 400, DIM),
        ("re-weights the cache and bakes nothing.", 400, DIM),
        ("", 400, DIM),
        ("Only what can change is recomputed", 600, INK),
        ("When an object moves, the pages whose light it", 400, DIM),
        ("could touch re-bake in the same frame; the rest", 400, DIM),
        ("is provably unchanged. Nothing is shown stale.", 400, DIM),
        ("", 400, DIM),
        ("No blending of past frames", 600, INK),
        ("Each frame comes from the cache as it is now:", 400, DIM),
        ("no ghosting or lag behind moving things.", 400, DIM),
    ]
    y = 84
    for s, wgt, col in lines:
        if s:
            body.append(text(tx + 40, y, s, 15 if wgt == 600 else 14, col, wgt))
        y += 21
    return card(W, H, "\n".join(body), "The lighting cache")


# ---------------------------------------------------------------- scorecard
def scorecard():
    rows = [
        ("Sponza", "3 million triangles, sun and sky", 30.43, 25.35),
        ("Closed room", "one skylight panel, light bouncing", 46.55, 6.93),
        ("Room with movers", "a ball and a cube moving", 44.46, 5.40),
        ("Boulder field", "a million boulders, sun and sky", 1.68, 1.29),
        ("DamagedHelmet", "metal, normal maps, emission", 4.46, 4.46),
        ("Fox", "a skinned character", 8.73, 2.17),
        ("BrainStem", "59 skinned meshes", 9.47, 3.28),
    ]
    W, row = 900, 50
    top = 96
    H = top + len(rows) * row + 56
    x0, x1, vmax = 300, 800, 50.0
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    body = [text(30, 36, "The basics scorecard: seven scenes against Cycles", 17, INK, 600)]
    body.append(text(30, 60, "Error from Blender's Cycles (clamped relative RMSE, lower is better), at the start of phase 1 and now", 13, DIM))
    for v in (0, 10, 20, 30, 40, 50):
        x = sx(v)
        body.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{top + len(rows) * row - 8}" stroke="#26303f" stroke-width="1"/>')
        body.append(text(x, top + len(rows) * row + 8, f"{v}%", 12, DIM, 400, "middle"))
    for k, (name, what, before, now) in enumerate(rows):
        y = top + k * row
        body.append(text(30, y + 10, name, 15, INK, 600))
        body.append(text(30, y + 28, what, 12, DIM))
        body.append(f'<rect x="{x0}" y="{y - 2}" width="{sx(before) - x0:.1f}" height="12" rx="3" fill="{SHADE}"/>')
        body.append(text(sx(before) + 6, y + 9, f"{before:.2f}%", 12, DIM))
        body.append(f'<rect x="{x0}" y="{y + 14}" width="{max(sx(now) - x0, 2):.1f}" height="12" rx="3" fill="{SUN}"/>')
        body.append(text(sx(now) + 6, y + 25, f"{now:.2f}%", 12, INK, 600))
    ly = H - 22
    body.append(f'<rect x="30" y="{ly - 11}" width="14" height="12" rx="3" fill="{SHADE}"/>')
    body.append(text(52, ly, "start of phase 1 (2026-11-22)", 13, DIM))
    body.append(f'<rect x="260" y="{ly - 11}" width="14" height="12" rx="3" fill="{SUN}"/>')
    body.append(text(282, ly, "now", 13, DIM))
    body.append(text(W - 30, ly, "Apple M4 · docs/results/basics-scorecard.md", 12, DIM, 400, "end"))
    return card(W, H, "\n".join(body), "The basics scorecard against Cycles")


# ---------------------------------------------------------------- texels
def blob(cx, cy, s):
    """A boulder's outline: a lumpy dome on the ground, scale s."""
    pts = []
    for i in range(65):
        a = math.pi * i / 64
        r = s * (1 + 0.08 * math.sin(5 * a) + 0.05 * math.cos(3 * a))
        pts.append((cx + 1.25 * r * math.cos(a), cy - 0.85 * r * math.sin(a)))
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"
    return d, pts


def texels():
    W, H = 900, 380
    body = [text(30, 36, "A coarse texel lights its whole area, not one point", 17, INK, 600)]
    body.insert(
        0,
        f'<defs><radialGradient id="lit" cx="0.72" cy="0.2" r="0.95"><stop offset="0" stop-color="#e9dcc2"/><stop offset="0.55" stop-color="#a99f8c"/><stop offset="1" stop-color="#3b3a37"/></radialGradient></defs>',
    )
    sun = lambda x, y: (
        f'<circle cx="{x}" cy="{y}" r="16" fill="{SUN}"/>'
        + "".join(
            f'<line x1="{x + 22 * math.cos(a):.1f}" y1="{y + 22 * math.sin(a):.1f}" x2="{x + 30 * math.cos(a):.1f}" y2="{y + 30 * math.sin(a):.1f}" stroke="{SUN}" stroke-width="2"/>'
            for a in [k * math.pi / 4 for k in range(8)]
        )
    )
    for panel in (0, 1):
        ox = 30 + panel * 440
        cx, cy, sc = ox + 150, 250, 70
        sx, sy = ox + 360, 92
        body.append(text(ox, 70, ["One point (before)", "16 points over it (now)"][panel], 15, INK, 600))
        body.append(sun(sx, sy))
        body.append(f'<line x1="{ox}" y1="{cy}" x2="{ox + 400}" y2="{cy}" stroke="{SHADE}" stroke-width="2"/>')
        d, pts = blob(cx, cy, sc)
        body.append(f'<path d="{d}" fill="#2b313d" stroke="{DIM}" stroke-width="1.5"/>')
        # The texel: the whole boulder.
        body.append(f'<rect x="{cx - 1.25 * sc * 1.1:.1f}" y="{cy - 0.85 * sc * 1.25:.1f}" width="{2.5 * sc * 1.1:.1f}" height="{0.85 * sc * 1.25:.1f}" fill="none" stroke="{SUN}" stroke-opacity="0.45" stroke-dasharray="5 5"/>')
        if panel == 0:
            # The texel's centre lands on the far side, low, facing away: its
            # ray to the sun runs through the boulder.
            px, py = pts[61]
            # Where the ray leaves the boulder (first outline point past it).
            ex, ey = sx - 18, sy + 12
            body.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{BLOCK}" stroke-width="2" stroke-dasharray="6 4" stroke-opacity="0.85"/>')
            body.append(text(px - 4, py + 22, "ray blocked by the boulder", 12, BLOCK))
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{BLOCK}"/>')
        else:
            for k in range(4, 61, 4):
                px, py = pts[k]
                nx, ny = px - cx, (py - cy) / 0.85 * 1.25
                l = math.hypot(nx, ny) or 1
                face = (nx * (sx - px) + ny * (sy - py)) > 0
                if face:
                    body.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{sx - 14:.1f}" y2="{sy + 10:.1f}" stroke="{SUN}" stroke-opacity="0.35" stroke-width="1.2"/>')
                body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{SUN if face else SHADE}"/>')
        # What the boulder looks like with that texel.
        tx, ty = ox + 300, 210
        d2, _ = blob(tx, ty + 40, 34)
        fill = "#2a2722" if panel == 0 else "url(#lit)"
        body.append(f'<path d="{d2}" fill="{fill}" stroke="{SHADE}" stroke-width="1"/>')
        body.append(text(tx, ty + 58, "shown", 12, DIM, 400, "middle"))
        cap = [
            ("Each boulder is one chart, so far away one texel covers it.", "Its centre faces down: no light, and it renders unlit."),
            ("Each point that faces the sun counts what it sees, weighted", "by its light: the texel holds the boulder's average."),
        ][panel]
        body.append(text(ox, 298, cap[0], 13, DIM))
        body.append(text(ox, 318, cap[1], 13, DIM))
    body.append(text(30, H - 22, "Texels in the top three mips of a chart that curves (M22 gate 10): the boulder field from above, 13.83% → 10.41% from Cycles.", 12, DIM))
    return card(W, H, "\n".join(body), "Coarse texels average their area")


# ---------------------------------------------------------------- frame
def frame():
    W, H = 900, 300
    body = [text(30, 36, "One frame", 17, INK, 600)]
    body.insert(
        0,
        f'<defs><marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{DIM}"/></marker></defs>',
    )
    steps = [
        ("Camera rays", ["Hardware ray queries", "find what each pixel", "sees."]),
        ("Pages", ["Each pixel finds its", "page of light; missing", "pages are requested."]),
        ("Bake", ["Within a time budget:", "a mask cone per light,", "rays for sky, bounce."]),
        ("Shade", ["Pixels read their pages;", "reflections and glass", "trace on."]),
        ("Display", ["Anti-aliased edges,", "tone map, bloom,", "exposure."]),
    ]
    bw, gap, y0 = 150, 22, 66
    for k, (title, lines) in enumerate(steps):
        x = 30 + k * (bw + gap)
        col = SUN if k == 2 else INK
        body.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="128" rx="10" fill="#1a212c" stroke="{SUN if k == 2 else "#26303f"}" stroke-width="1.5"/>')
        body.append(text(x + 14, y0 + 30, f"{k + 1}. {title}", 15, col, 600))
        for j, l in enumerate(lines):
            body.append(text(x + 14, y0 + 58 + 20 * j, l, 13, DIM))
        if k < len(steps) - 1:
            body.append(f'<line x1="{x + bw + 3}" y1="{y0 + 64}" x2="{x + bw + gap - 4}" y2="{y0 + 64}" stroke="{DIM}" stroke-width="2" marker-end="url(#arr2)"/>')
    body.append(text(30, 236, "What the budget cannot bake yet shows its page's nearest coarser version; nothing is blended from past frames,", 13, DIM))
    body.append(text(30, 256, "so a frame is the scene as it is now. Pages stay while they can still be right, and a move re-bakes only what it can change.", 13, DIM))
    return card(W, H, "\n".join(body), "One frame")


for name, svg in [
    ("mask-cone.svg", mask_cone()),
    ("three-ways.svg", three_ways()),
    ("cache.svg", cache()),
    ("scorecard.svg", scorecard()),
    ("texels.svg", texels()),
    ("frame.svg", frame()),
]:
    with open(os.path.join(OUT, name), "w") as f:
        f.write(svg)
    print(name, len(svg) // 1024, "KB")
