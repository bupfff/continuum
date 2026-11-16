// Motion blur (M21): what a shutter open for part of the frame sees,
// gathered each frame from the image, its depths and each pixel's motion
// (nothing kept between frames).
//
// A surface point moving at constant velocity crosses the image along a
// segment V pixels long during the shutter, centred on where the frame
// sees it (V: the shutter times its motion since the previous camera and
// transforms). Each pixel gathers the points whose segments cover it, along
// two lines (the fastest motion near it, and its own), in three groups by
// depth against its own: in front (they cover what is behind them), its
// own surface, and behind (seen where its own surface does not cover it;
// where none covers it, the nearest behind stand in for what the image
// hides). Taps are whole pixels, a pixel's step apart along the line's
// major axis.

struct Camera {
    eye: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // x: width, y: height (pixels); z: tan(half vertical fov)
    params: vec4<f32>,
};

struct MotionParams {
    // x: the shutter (a share of the motion since the previous pose); y:
    // the longest segment (pixels); z: 1 if any transform moved; w: depth
    // tolerance (relative)
    p: vec4<f32>,
    // x, y: 16 × 16 tiles across and down; z: the dilation's reach (tiles)
    tiles: vec4<u32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vis: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> motion: MotionParams;
@group(0) @binding(4) var<uniform> before: Camera;
// The scene's objects (binding 3 of the camera passes: `w`, the object's
// transform) and transforms (rows 0-2 with the translation in w, then
// (scale, 1 / scale, 0, 0)).
@group(0) @binding(5) var<storage, read> meshes: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> xforms: array<vec4<f32>>;
// Per transform, its mover's slot in `moved` plus one (0: it never moves);
// per slot, the mover's transform before (four rows as `xforms`).
@group(0) @binding(7) var<storage, read> mover_of: array<u32>;
@group(0) @binding(8) var<storage, read> moved: array<vec4<f32>>;
// Per pixel, as halves: (r, g), (b, view depth), (Vx, Vy), 0.
@group(0) @binding(9) var<storage, read_write> src: array<vec4<u32>>;
// Each tile's longest V, then (after the first tiles.x · tiles.y) the
// longest of those that can reach it.
@group(0) @binding(10) var<storage, read_write> tiles: array<vec2<f32>>;

const NONE: u32 = 0xffffffffu;

// Continuous image coordinates (pixels, the pixel offset left out) of a
// point seen by camera `c`, and its view depth.
fn project(c: Camera, p: vec3<f32>) -> vec3<f32> {
    let d = p - c.eye.xyz;
    let z = dot(d, c.forward.xyz);
    let aspect = c.params.x / c.params.y;
    let sx = dot(d, c.right.xyz) / z / (c.params.z * aspect);
    let sy = dot(d, c.up.xyz) / z / c.params.z;
    return vec3<f32>((sx + 1.0) * 0.5 * c.params.x, (1.0 - sy) * 0.5 * c.params.y, z);
}

// A point's position before: through its object's transform before, if it
// is a mover's.
fn point_before(m: u32, p: vec3<f32>) -> vec3<f32> {
    if (motion.p.z < 0.5 || m == NONE || m >= arrayLength(&meshes)) { return p; }
    let x = meshes[m].w;
    if (x == NONE || x >= arrayLength(&mover_of)) { return p; }
    let slot = mover_of[x];
    if (slot == 0u) { return p; }
    let t = vec3<f32>(xforms[4u * x].w, xforms[4u * x + 1u].w, xforms[4u * x + 2u].w);
    let q = p - t;
    let local = (xforms[4u * x].xyz * q.x + xforms[4u * x + 1u].xyz * q.y + xforms[4u * x + 2u].xyz * q.z) * xforms[4u * x + 3u].y;
    let k = 4u * (slot - 1u);
    let s = moved[k + 3u].x;
    return vec3<f32>(dot(moved[k].xyz, local), dot(moved[k + 1u].xyz, local), dot(moved[k + 2u].xyz, local)) * s
        + vec3<f32>(moved[k].w, moved[k + 1u].w, moved[k + 2u].w);
}

var<workgroup> wg_v: array<vec2<f32>, 256>;

// Each pixel's colour, view depth and segment V; each tile's longest V.
@compute @workgroup_size(16, 16)
fn motion_prepare(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    var v = vec2<f32>(0.0);
    if (gid.x < w && gid.y < h) {
        let pixel = gid.y * w + gid.x;
        let hit = vis[pixel];
        let aspect = camera.params.x / camera.params.y;
        let px = vec2<f32>(f32(gid.x) + 0.5 + camera.right.w, f32(gid.y) + 0.5 + camera.up.w);
        let sx = (px.x / camera.params.x * 2.0 - 1.0) * camera.params.z * aspect;
        let sy = (1.0 - px.y / camera.params.y * 2.0) * camera.params.z;
        let dir = camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy;
        var z = 1e4;
        if (hit.x != NONE) {
            let t = bitcast<f32>(hit.z);
            let len = length(dir);
            z = t / len;
            let p = camera.eye.xyz + dir / len * t;
            let then = project(before, point_before(hit.x, p));
            let now = project(camera, p);
            if (then.z > 1e-4) { v = (now.xy - then.xy) * motion.p.x; }
        } else {
            // The sky: directions only (at infinity).
            let far = camera.eye.xyz + dir * 1e6;
            let then = project(before, far - camera.eye.xyz + before.eye.xyz);
            let now = project(camera, far);
            if (then.z > 0.0) { v = (now.xy - then.xy) * motion.p.x; }
        }
        let l = length(v);
        if (l > motion.p.y) { v *= motion.p.y / l; }
        let col = min(colors[pixel].xyz, vec3<f32>(65000.0));
        src[pixel] = vec4<u32>(pack2x16float(col.xy), pack2x16float(vec2<f32>(col.z, min(z, 65000.0))), pack2x16float(v), 0u);
    }
    wg_v[li] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s && dot(wg_v[li + s], wg_v[li + s]) > dot(wg_v[li], wg_v[li])) { wg_v[li] = wg_v[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u && wid.x < motion.tiles.x && wid.y < motion.tiles.y) {
        tiles[wid.y * motion.tiles.x + wid.x] = wg_v[0];
    }
}

// Each tile's longest segment among the tiles near enough to reach it (half
// a segment reaches from its tile; their nearest pixels at least
// (Chebyshev steps − 1) · 16 apart).
@compute @workgroup_size(8, 8)
fn motion_dilate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tx = motion.tiles.x;
    let ty = motion.tiles.y;
    if (gid.x >= tx || gid.y >= ty) { return; }
    let reach = i32(motion.tiles.z);
    var m = vec2<f32>(0.0);
    for (var j = -reach; j <= reach; j++) {
        for (var i = -reach; i <= reach; i++) {
            let x = i32(gid.x) + i;
            let y = i32(gid.y) + j;
            if (x < 0 || y < 0 || x >= i32(tx) || y >= i32(ty)) { continue; }
            let t = tiles[u32(y) * tx + u32(x)];
            let gap = f32(max(max(abs(i), abs(j)) - 1, 0)) * 16.0;
            if (0.5 * length(t) >= gap && dot(t, t) > dot(m, m)) { m = t; }
        }
    }
    tiles[tx * ty + gid.y * tx + gid.x] = m;
}

struct Source {
    col: vec3<f32>,
    z: f32,
    v: vec2<f32>,
};

fn source_at(q: vec2<i32>) -> Source {
    let w = i32(camera.params.x);
    let h = i32(camera.params.y);
    let c = clamp(q, vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    let s = src[u32(c.y * w + c.x)];
    let rg = unpack2x16float(s.x);
    let bz = unpack2x16float(s.y);
    return Source(vec3<f32>(rg, bz.x), bz.y, unpack2x16float(s.z));
}

struct Gather {
    front: vec4<f32>,
    own: vec4<f32>,
    back: vec4<f32>,
    fill: vec4<f32>,
};

// One tap: source `s` at offset `o` from the pixel, standing for `step`
// pixels of the line it lies on.
fn gather_tap(g: ptr<function, Gather>, s: Source, o: vec2<f32>, step: f32, zp: f32) {
    let len = length(s.v);
    var cover = 0.0;
    var wgt = 0.0;
    if (len < 1.0) {
        // Still (or nearly): its own pixel only.
        cover = select(0.0, 1.0, dot(o, o) < 0.25);
        wgt = cover;
    } else {
        let dir = s.v / len;
        let along = abs(dot(o, dir));
        let across = abs(o.x * dir.y - o.y * dir.x);
        cover = clamp((0.5 * len - along) / step + 0.5, 0.0, 1.0) * clamp(1.0 - across, 0.0, 1.0);
        wgt = step * cover / len;
    }
    let tol = motion.p.w * zp;
    let in_front = select(0.0, 1.0, s.z < zp - tol);
    let behind = select(0.0, 1.0, s.z > zp + tol);
    let t = vec4<f32>(s.col * wgt, wgt);
    (*g).front += in_front * t;
    (*g).back += behind * t;
    (*g).own += (1.0 - in_front - behind) * t;
    let f = behind / (1.0 + dot(o, o));
    (*g).fill += vec4<f32>(s.col * f, f);
}

// The taps along a line through the pixel in direction `dir` out to half
// of `extent` pixels each way, a pixel's step apart along its major axis;
// `skip_near` leaves out what the first line took (the pixel itself and the
// taps along the same direction).
fn gather_line(g: ptr<function, Gather>, p: vec2<i32>, dir: vec2<f32>, extent: f32, zp: f32) {
    let major = max(abs(dir.x), abs(dir.y));
    if (major < 1e-6) { return; }
    let step = 1.0 / major;
    let n = i32(ceil(0.5 * extent / step));
    for (var k = 1; k <= n; k++) {
        for (var side = -1; side <= 1; side += 2) {
            let o = round(dir * (f32(k * side) * step));
            gather_tap(g, source_at(p + vec2<i32>(o)), o, step, zp);
        }
    }
}

// Each pixel's colour through the shutter.
@compute @workgroup_size(8, 8)
fn motion_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let n = motion.tiles.x * motion.tiles.y;
    let fastest = tiles[n + (gid.y / 16u) * motion.tiles.x + gid.x / 16u];
    let reach = length(fastest);
    if (reach < 1.0) { return; }
    let pixel = gid.y * w + gid.x;
    let p = vec2<i32>(gid.xy);
    let me = source_at(p);
    var g = Gather(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    gather_tap(&g, me, vec2<f32>(0.0), 1.0, me.z);
    // Along the fastest motion near it; and along its own where that
    // differs (more than about 11°) and is long enough to matter.
    let d1 = fastest / reach;
    gather_line(&g, p, d1, reach, me.z);
    let own = length(me.v);
    if (own >= 1.0) {
        let d2 = me.v / own;
        if (abs(d1.x * d2.y - d1.y * d2.x) > 0.2) {
            gather_line(&g, p, d2, own, me.z);
        }
    }
    // In front over its own surface over what is behind.
    var front = g.front;
    if (front.w > 1.0) { front /= front.w; }
    var mine = g.own;
    if (mine.w > 1.0) { mine /= mine.w; }
    var rest = me.col;
    if (g.fill.w > 0.0) {
        rest = g.fill.xyz / g.fill.w;
    } else if (mine.w > 0.0) {
        rest = mine.xyz / mine.w;
    }
    var back = g.back;
    if (back.w > 1.0) { back /= back.w; }
    let behind = back.xyz + (1.0 - back.w) * rest;
    let out = front.xyz + (1.0 - front.w) * (mine.xyz + (1.0 - mine.w) * behind);
    colors[pixel] = vec4<f32>(out, colors[pixel].w);
}
