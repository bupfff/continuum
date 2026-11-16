// Depth of field (M20): a thin lens's blur, from the pinhole image and its
// depths, gathered per pixel each frame (nothing kept between frames).
//
// A point at view depth z spreads over a disc of radius
// c = f · A · (1/F − 1/z) pixels (f the focal length in pixels, A the
// aperture's radius, F the focus distance; negative in front of the plane
// in focus). Each pixel gathers the points whose discs cover it, in three
// groups by their signed radius against its own: in front (they cover what
// is behind them), its own surface, and behind (seen where its own surface
// does not cover the pixel's lens; where none covers it, the nearest behind
// stand in for what the pinhole image hides).

struct Camera {
    eye: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // x: width, y: height (pixels); z: tan(half vertical fov)
    params: vec4<f32>,
};

struct DofParams {
    // x: f · A (pixels times scene units); y: 1 / F; z: the largest radius
    // (pixels); w: rings (the levels below the image)
    lens: vec4<f32>,
    // x, y: 16 × 16 tiles across and down; z: the dilation's reach (tiles)
    tiles: vec4<u32>,
    // x, y: the levels' padded size (pixels, a multiple of 2^rings)
    padded: vec4<f32>,
    // A ring's taps for an inner radius of 1: x, y, their length.
    taps: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vis: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> dof: DofParams;
// Each tile's largest |c|, then (after the first tiles.x · tiles.y) the
// largest that can reach it.
@group(0) @binding(4) var<storage, read_write> tiles: array<f32>;
// Levels 1 and below (mip m is level m + 1).
@group(0) @binding(5) var level_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var level_in: texture_2d<f32>;
@group(0) @binding(7) var level_sampler: sampler;
// Level 0 (the padded size), as halves: (r, g), (b, c).
@group(0) @binding(8) var<storage, read_write> base: array<vec2<u32>>;
// The pixels done so far (even coordinates), as halves: (r, g), (b, 0).
@group(0) @binding(9) var<storage, read_write> out: array<vec2<u32>>;

fn base_at(q: vec2<u32>) -> vec4<f32> {
    let v = base[q.y * u32(dof.padded.x) + q.x];
    return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y));
}

const PI: f32 = 3.14159265;
// The radius whose disc is one pixel's area.
const R_PIXEL: f32 = 0.5641896;
// The inner disc's radius (pixels), gathered from the image itself; ring k
// covers [R_INNER · 2^(k−1), R_INNER · 2^k) from level k.
const R_INNER: f32 = 3.5;
const RING_TAPS: u32 = 16u;

// Pixel `p`'s signed radius (pixels): from its hit's distance along its
// camera ray (the sky at infinity), clamped to the largest.
fn radius_at(p: vec2<u32>) -> f32 {
    let w = u32(camera.params.x);
    let v = vis[p.y * w + p.x];
    let aspect = camera.params.x / camera.params.y;
    let sx = ((f32(p.x) + 0.5 + camera.right.w) / camera.params.x * 2.0 - 1.0) * camera.params.z * aspect;
    let sy = (1.0 - (f32(p.y) + 0.5 + camera.up.w) / camera.params.y * 2.0) * camera.params.z;
    var inv_z = 0.0;
    if (v.x != 0xffffffffu) {
        inv_z = sqrt(1.0 + sx * sx + sy * sy) / max(bitcast<f32>(v.z), 1e-6);
    }
    return clamp(dof.lens.x * (dof.lens.y - inv_z), -dof.lens.z, dof.lens.z);
}

// Four texels as one: the colours' mean, and the radius of the least
// blurred (a texel across an edge spreads no farther than its sharper side).
fn merge4(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>) -> vec4<f32> {
    var m = a.w;
    if (abs(b.w) < abs(m)) { m = b.w; }
    if (abs(c.w) < abs(m)) { m = c.w; }
    if (abs(d.w) < abs(m)) { m = d.w; }
    return vec4<f32>((a.xyz + b.xyz + c.xyz + d.xyz) * 0.25, m);
}

// Levels 1 to 4 as the prepare pass writes them (mips 0 to 3).
@group(0) @binding(10) var level1: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var level2: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var level3: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var level4: texture_storage_2d<rgba16float, write>;

// Highlights: pixels HL_RATIO times brighter than their tile's mean,
// blurred by at least HL_BLUR pixels. Their excess over the tile's mean
// colour leaves the levels (whose sparse taps would scatter it into
// blotches) and every pixel its disc covers adds it back exactly: from a
// list of at most HL_ALL, binned into lists per tile of at most HL_CAP (a
// tile with more reads the whole list).
const HL_RATIO: f32 = 8.0;
const HL_BLUR: f32 = 2.0;
const HL_ALL: u32 = 4096u;
const HL_CAP: u32 = 64u;
// The list's length, then each tile's.
@group(0) @binding(14) var<storage, read_write> hl_count: array<atomic<u32>>;
// The list: pairs (x, y, c, 0) and (excess, 0).
@group(0) @binding(15) var<storage, read_write> hl: array<vec4<f32>>;
// Each tile's highlights (indices into the list).
@group(0) @binding(16) var<storage, read_write> hl_tiles: array<u32>;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

var<workgroup> wg_a: array<vec4<f32>, 256>;
var<workgroup> wg_b: array<vec4<f32>, 64>;
var<workgroup> wg_reach: array<vec2<f32>, 256>;
var<workgroup> wg_sum: array<vec4<f32>, 256>;

// 2 × 2 of a `n`-wide square in `wg_a` (or `wg_b`) merged, for texel
// (x, y) of the level below.
fn merged_a(x: u32, y: u32, n: u32) -> vec4<f32> {
    let k = 2u * y * n + 2u * x;
    return merge4(wg_a[k], wg_a[k + 1u], wg_a[k + n], wg_a[k + n + 1u]);
}

fn merged_b(x: u32, y: u32, n: u32) -> vec4<f32> {
    let k = 2u * y * n + 2u * x;
    return merge4(wg_b[k], wg_b[k + 1u], wg_b[k + n], wg_b[k + n + 1u]);
}

// Level 0 (the padded size; outside the image, its nearest pixel): each
// pixel's colour and signed radius; levels 1 to 4 of its 16 × 16 tile; the
// tile's farthest reach in front of the plane in focus (−c) and behind it
// (c).
@compute @workgroup_size(16, 16)
fn dof_prepare(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let size = vec2<u32>(dof.padded.xy);
    let p = min(gid.xy, vec2<u32>(w - 1u, h - 1u));
    let c = radius_at(p);
    var col = min(colors[p.y * w + p.x].xyz, vec3<f32>(65000.0));
    let inside = gid.x < w && gid.y < h;
    var reach = vec2<f32>(0.0);
    if (inside) { reach = vec2<f32>(max(-c, 0.0), max(c, 0.0)); }
    wg_reach[li] = reach;
    wg_sum[li] = select(vec4<f32>(0.0), vec4<f32>(col, 1.0), inside);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            wg_reach[li] = max(wg_reach[li], wg_reach[li + s]);
            wg_sum[li] += wg_sum[li + s];
        }
        workgroupBarrier();
    }
    // A highlight against its tile's mean.
    let mean = wg_sum[0].xyz / max(wg_sum[0].w, 1.0);
    if (inside && abs(c) >= HL_BLUR && luminance(col) > HL_RATIO * luminance(mean) + 1e-6) {
        let slot = atomicAdd(&hl_count[0], 1u);
        if (slot < HL_ALL) {
            let excess = max(col - mean, vec3<f32>(0.0));
            col -= excess;
            hl[2u * slot] = vec4<f32>(vec2<f32>(gid.xy), c, 0.0);
            hl[2u * slot + 1u] = vec4<f32>(excess, 0.0);
        }
    }
    wg_a[li] = vec4<f32>(col, c);
    base[gid.y * size.x + gid.x] = vec2<u32>(pack2x16float(col.xy), pack2x16float(vec2<f32>(col.z, c)));
    workgroupBarrier();
    if (li < 64u) {
        let t = merged_a(li % 8u, li / 8u, 16u);
        textureStore(level1, wid.xy * 8u + vec2<u32>(li % 8u, li / 8u), t);
        wg_b[li] = t;
    }
    workgroupBarrier();
    if (li < 16u) {
        let t = merged_b(li % 4u, li / 4u, 8u);
        textureStore(level2, wid.xy * 4u + vec2<u32>(li % 4u, li / 4u), t);
        wg_a[li] = t;
    }
    workgroupBarrier();
    if (li < 4u) {
        let t = merged_a(li % 2u, li / 2u, 4u);
        textureStore(level3, wid.xy * 2u + vec2<u32>(li % 2u, li / 2u), t);
        wg_b[li] = t;
    }
    workgroupBarrier();
    if (li == 0u) {
        textureStore(level4, wid.xy, merged_b(0u, 0u, 2u));
        if (wid.x < dof.tiles.x && wid.y < dof.tiles.y) {
            let n = dof.tiles.x * dof.tiles.y;
            let k = wid.y * dof.tiles.x + wid.x;
            tiles[k] = wg_reach[0].x;
            tiles[n + k] = wg_reach[0].y;
        }
    }
}

// Level k from level k − 1 (k ≥ 2): merged 2 × 2.
@compute @workgroup_size(8, 8)
fn dof_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(level_out);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let q = gid.xy * 2u;
    textureStore(level_out, gid.xy, merge4(textureLoad(level_in, q, 0), textureLoad(level_in, q + vec2<u32>(1u, 0u), 0), textureLoad(level_in, q + vec2<u32>(0u, 1u), 0), textureLoad(level_in, q + vec2<u32>(1u, 1u), 0)));
}

// Each highlight into the lists of the tiles its disc (and soft edge)
// touches.
@compute @workgroup_size(64)
fn dof_bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= min(atomicLoad(&hl_count[0]), HL_ALL)) { return; }
    let at = hl[2u * k];
    let r = abs(at.z) + 1.0;
    let t0 = vec2<i32>(floor((at.xy + 0.5 - r) / 16.0));
    let t1 = vec2<i32>(floor((at.xy + 0.5 + r) / 16.0));
    for (var ty = max(t0.y, 0); ty <= min(t1.y, i32(dof.tiles.y) - 1); ty++) {
        for (var tx = max(t0.x, 0); tx <= min(t1.x, i32(dof.tiles.x) - 1); tx++) {
            let tile = u32(ty) * dof.tiles.x + u32(tx);
            let slot = atomicAdd(&hl_count[1u + tile], 1u);
            if (slot < HL_CAP) { hl_tiles[tile * HL_CAP + slot] = k; }
        }
    }
}

// Each tile's reach in front and behind: the largest of the tiles near
// enough for it to reach (their nearest pixels at least (Chebyshev steps −
// 1) · 16 apart).
@compute @workgroup_size(8, 8)
fn dof_dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx = dof.tiles.x;
    let ty = dof.tiles.y;
    if (gid.x >= tx || gid.y >= ty) { return; }
    let n = tx * ty;
    let reach = i32(dof.tiles.z);
    var m = vec2<f32>(0.0);
    for (var j = -reach; j <= reach; j++) {
        for (var i = -reach; i <= reach; i++) {
            let x = i32(gid.x) + i;
            let y = i32(gid.y) + j;
            if (x < 0 || y < 0 || x >= i32(tx) || y >= i32(ty)) { continue; }
            let k = u32(y) * tx + u32(x);
            let t = vec2<f32>(tiles[k], tiles[n + k]);
            let gap = f32(max(max(abs(i), abs(j)) - 1, 0)) * 16.0;
            m = max(m, select(vec2<f32>(0.0), t, t >= vec2<f32>(gap)));
        }
    }
    let k = gid.y * tx + gid.x;
    tiles[2u * n + k] = m.x;
    tiles[3u * n + k] = m.y;
}

struct Gather {
    front: vec4<f32>,
    own: vec4<f32>,
    back: vec4<f32>,
    fill: vec4<f32>,
};

// One tap: colour `col` with signed radius `c`, `d` pixels away, standing
// for `area` pixels spaced 1 / `inv_s` apart (the soft edge of its disc).
// Without branches: each group takes its share by a 0/1 factor.
fn gather_tap(g: ptr<function, Gather>, col: vec3<f32>, c: f32, d: f32, area: f32, inv_s: f32, cp: f32, tol: f32) {
    let r = max(abs(c), R_PIXEL);
    let cover = clamp((r - d) * inv_s + 0.5, 0.0, 1.0);
    let wgt = area * cover / (PI * r * r);
    let in_front = select(0.0, 1.0, c < cp - tol);
    let behind = select(0.0, 1.0, c > cp + tol);
    let t = vec4<f32>(col * wgt, wgt);
    (*g).front += in_front * t;
    (*g).back += behind * t;
    (*g).own += (1.0 - in_front - behind) * t;
    let f = behind * area / (1.0 + d * d);
    (*g).fill += vec4<f32>(col * f, f);
}

// Pixel `p`'s colour through the lens, its tile reaching `reach` (in
// front, behind). Nothing behind shows through an in-focus pixel (its own
// tap covers it), so it gathers only as far as what is in front reaches.
fn gather_at(p: vec2<u32>, reach: vec2<f32>) -> vec3<f32> {
    let w = u32(camera.params.x);
    let own_texel = base_at(p);
    let cp = own_texel.w;
    // In focus, the exact colour; blurred, level 0's (a highlight's excess
    // comes from its list).
    var centre = colors[p.y * w + p.x].xyz;
    if (abs(cp) >= HL_BLUR) { centre = own_texel.xyz; }
    var r_max = max(reach.x, reach.y);
    if (abs(cp) < R_PIXEL) { r_max = max(reach.x, 1.0); }
    let tol = max(1.0, 0.15 * abs(cp));
    var g = Gather(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    gather_tap(&g, centre, cp, 0.0, 1.0, 2.0, cp, tol);
    // The inner disc: the 36 pixels within R_INNER, sharing its area less
    // the centre's.
    let inner_area = (PI * R_INNER * R_INNER - 1.0) / 36.0;
    let last = vec2<i32>(i32(dof.padded.x) - 1, i32(dof.padded.y) - 1);
    for (var j = -3; j <= 3; j++) {
        for (var i = -3; i <= 3; i++) {
            let d2 = i * i + j * j;
            if (d2 == 0 || d2 > 12) { continue; }
            let q = clamp(vec2<i32>(p) + vec2<i32>(i, j), vec2<i32>(0), last);
            let t = base_at(vec2<u32>(q));
            gather_tap(&g, t.xyz, t.w, sqrt(f32(d2)), inner_area, 1.0, cp, tol);
        }
    }
    // The rings, each from its level: taps on a golden-angle spiral (the
    // pattern turned 0.7 radians a ring), equal areas, read bilinearly.
    let rings = u32(dof.lens.w);
    let uv = (vec2<f32>(p) + 0.5) / dof.padded.xy;
    let to_uv = 1.0 / dof.padded.xy;
    for (var k = 1u; k <= rings; k++) {
        let r_in = R_INNER * f32(1u << (k - 1u));
        let area = 3.0 * PI * r_in * r_in / f32(RING_TAPS);
        // Sharper than the taps' spacing: the bilinear reads blur already.
        let inv_s = 2.0 * inverseSqrt(area);
        if (r_in > r_max + 0.5 / inv_s) { break; }
        let turn = f32(k) * 0.7;
        let rot = r_in * vec2<f32>(cos(turn), sin(turn));
        let lod = f32(k - 1u);
        for (var n = 0u; n < RING_TAPS; n++) {
            let tp = dof.taps[n];
            let off = vec2<f32>(tp.x * rot.x - tp.y * rot.y, tp.x * rot.y + tp.y * rot.x);
            let t = textureSampleLevel(level_in, level_sampler, uv + off * to_uv, lod);
            gather_tap(&g, t.xyz, t.w, tp.z * r_in, area, inv_s, cp, tol);
        }
    }
    // The highlights whose discs reach this tile, each exactly (a tile
    // with more than its list holds reads the whole list).
    let tile = (p.y / 16u) * dof.tiles.x + p.x / 16u;
    let listed = atomicLoad(&hl_count[1u + tile]);
    let all = min(atomicLoad(&hl_count[0]), HL_ALL);
    let n_hl = select(listed, all, listed > HL_CAP);
    for (var k = 0u; k < n_hl; k++) {
        var e = k;
        if (listed <= HL_CAP) { e = hl_tiles[tile * HL_CAP + k]; }
        let at = hl[2u * e];
        let d = distance(vec2<f32>(p), at.xy);
        if (d <= abs(at.z) + 1.0) {
            gather_tap(&g, hl[2u * e + 1u].xyz, at.z, d, 1.0, 1.0, cp, tol);
        }
    }
    // In front over its own surface over what is behind.
    var front = g.front;
    if (front.w > 1.0) { front /= front.w; }
    var own = g.own;
    if (own.w > 1.0) { own /= own.w; }
    var rest = centre;
    if (g.fill.w > 0.0) {
        rest = g.fill.xyz / g.fill.w;
    } else if (own.w > 0.0) {
        rest = own.xyz / own.w;
    }
    var back = g.back;
    if (back.w > 1.0) { back /= back.w; }
    let behind = back.xyz + (1.0 - back.w) * rest;
    return front.xyz + (1.0 - front.w) * (own.xyz + (1.0 - own.w) * behind);
}

fn reach_of(p: vec2<u32>) -> vec2<f32> {
    let n = dof.tiles.x * dof.tiles.y;
    let k = (p.y / 16u) * dof.tiles.x + p.x / 16u;
    return vec2<f32>(tiles[2u * n + k], tiles[3u * n + k]);
}

fn out_at(q: vec2<u32>) -> vec3<f32> {
    let v = out[q.y * u32(camera.params.x) + q.x];
    return vec3<f32>(unpack2x16float(v.x), unpack2x16float(v.y).x);
}

fn put_out(q: vec2<u32>, c: vec3<f32>) {
    let k = min(c, vec3<f32>(65000.0));
    out[q.y * u32(camera.params.x) + q.x] = vec2<u32>(pack2x16float(k.xy), pack2x16float(vec2<f32>(k.z, 0.0)));
}

// The pixels at coordinates that are multiples of 4, gathered into `out`.
@compute @workgroup_size(8, 8)
fn dof_gather_coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let p = gid.xy * 4u;
    if (p.x >= w || p.y >= h) { return; }
    let r_max = reach_of(p);
    if (max(r_max.x, r_max.y) < 0.5) { return; }
    put_out(p, gather_at(p, r_max));
}

// Pixel `p` at a multiple of `step` (2, then 1) and not of 2 · step:
// between those done before (in `out`) of the same layer, blurred by at
// least SMOOTH · step pixels and within CONTRAST of their mean, their mean
// (the image through the lens varies slowly there); elsewhere gathered (a
// bright light's disc has a sharp edge however wide it is).
const SMOOTH: f32 = 3.0;
const CONTRAST: f32 = 0.25;

fn refine(p: vec2<u32>, step: u32, r_max: vec2<f32>) -> vec3<f32> {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let ox = (p.x / step) % 2u;
    let oy = (p.y / step) % 2u;
    let cp = base_at(p).w;
    let tol = 0.5 * max(1.0, 0.15 * abs(cp));
    var even = abs(cp) >= SMOOTH * f32(step);
    let si = i32(step);
    var sum = vec3<f32>(0.0);
    var lo = vec3<f32>(3.4e38);
    var hi = vec3<f32>(0.0);
    var n = 0.0;
    for (var k = 0u; k < 4u; k++) {
        var q = vec2<i32>(p);
        if (ox == 1u && oy == 1u) {
            q += vec2<i32>(select(-si, si, (k & 1u) == 1u), select(-si, si, (k & 2u) == 2u));
        } else if (k < 2u) {
            let s = select(-si, si, k == 1u);
            q += select(vec2<i32>(0, s), vec2<i32>(s, 0), ox == 1u);
        } else {
            continue;
        }
        if (q.x < 0 || q.y < 0 || q.x >= i32(w) || q.y >= i32(h)) { continue; }
        let cq = base_at(vec2<u32>(q)).w;
        even = even && abs(cq - cp) <= tol;
        let o = out_at(vec2<u32>(q));
        sum += o;
        lo = min(lo, o);
        hi = max(hi, o);
        n += 1.0;
    }
    let mean = sum / max(n, 1.0);
    if (even && n > 0.0 && all(hi - lo <= CONTRAST * mean + 1e-4)) {
        return mean;
    }
    return gather_at(p, r_max);
}

// The even pixels not done, into `out`.
@compute @workgroup_size(8, 8)
fn dof_refine_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy * 2u;
    if (p.x >= u32(camera.params.x) || p.y >= u32(camera.params.y) || (p.x % 4u == 0u && p.y % 4u == 0u)) { return; }
    let r_max = reach_of(p);
    if (max(r_max.x, r_max.y) < 0.5) { return; }
    put_out(p, refine(p, 2u, r_max));
}

// Every pixel's colour: the even ones from `out`, the rest refined.
@compute @workgroup_size(8, 8)
fn dof_refine_full(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let p = gid.xy;
    if (p.x >= w || p.y >= u32(camera.params.y)) { return; }
    let r_max = reach_of(p);
    if (max(r_max.x, r_max.y) < 0.5) { return; }
    let pixel = p.y * w + p.x;
    var c: vec3<f32>;
    if (p.x % 2u == 0u && p.y % 2u == 0u) {
        c = out_at(p);
    } else {
        c = refine(p, 1u, r_max);
    }
    colors[pixel] = vec4<f32>(c, colors[pixel].w);
}
