// Mask cones on the GPU: one invocation per receiver.
//
// A line-for-line port of `continuum::cone3`: a pyramid cone from the receiver to
// a rectangular light, 8x8 bins weighted by projected solid angle (Lambert),
// triangles clipped to the cone's slab and window and rasterized into an
// accumulation buffer for exact per-bin area, coverage summed within a mesh and
// combined by max across meshes. The output is irradiance per unit radiance.

const GRID: u32 = 8u;
const BINS: u32 = 64u;
const MAX_POLY: u32 = 10u;

struct Light {
    corner: vec4<f32>,
    edge_u: vec4<f32>,
    edge_v: vec4<f32>,
    // x: mesh count; y: 1 to write per-bin coverage to `debug`; z: 0 to test
    // every triangle, 1 to traverse the mesh hierarchy, 2 to also traverse each
    // mesh's triangle hierarchy; w: 1 if the light is one-sided (it emits only
    // toward edge_u × edge_v and camera rays see its front)
    counts: vec4<u32>,
    // x: prefilter threshold in bins (0: exact; see `node_cone_test`); y: 1 for
    // nearest-first ray traversal, 0 for a fixed child order; z: profiling
    // stage, cutting cones short to time their parts (0: off, 1: no cone,
    // 2: setup and weights only, 3: plus the object hierarchy, 4: plus triangle
    // hierarchies but no rasterization); w: sky flags: bit 1 to combine
    // coverage across meshes by the maximum (a diagnostic), bit 2 to walk
    // geometry below the receiver's horizon too, bit 4 for per-pixel sky
    // faces in the hit object's frame
    params: vec4<f32>,
    // A uniform sky: radiance (linear RGB) over the whole sphere of directions,
    // lit through six cube-face mask cones per receiver (zero: no sky).
    // Per-pixel shading only so far.
    sky: vec4<f32>,
    // x: the object hierarchy's first node in `blas_cov` (NONE: no object-
    // level prefiltering; see `splat_object_node`)
    objects: vec4<u32>,
};

struct Receiver {
    point: vec4<f32>,
    normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> light: Light;
@group(0) @binding(1) var<storage, read> receivers: array<Receiver>;
@group(0) @binding(2) var<storage, read> tris: array<vec4<f32>>;
// Per object: x: first triangle, y: triangle count, z: root of the triangle
// hierarchy in `blas` (NONE for an empty mesh), w: transform in `xforms` (NONE:
// the triangles are already in world space). Objects with transforms share
// geometry (instancing).
@group(0) @binding(3) var<storage, read> meshes: array<vec4<u32>>;
// Per transform: rows 0-2 of the rotation with the translation in w, then
// (scale, 1 / scale, 0, 0): world = scale · rotation · p + translation, with
// positions relative to the render origin.
@group(0) @binding(30) var<storage, read> xforms: array<vec4<f32>>;

fn rotate_to_object(x: u32, d: vec3<f32>) -> vec3<f32> {
    return xforms[4u * x].xyz * d.x + xforms[4u * x + 1u].xyz * d.y + xforms[4u * x + 2u].xyz * d.z;
}

fn rotate_to_world(x: u32, d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(xforms[4u * x].xyz, d), dot(xforms[4u * x + 1u].xyz, d), dot(xforms[4u * x + 2u].xyz, d));
}

fn translation(x: u32) -> vec3<f32> {
    return vec3<f32>(xforms[4u * x].w, xforms[4u * x + 1u].w, xforms[4u * x + 2u].w);
}

// A world point in the space of object `m` (unchanged without a transform).
fn point_to_object(m: u32, p: vec3<f32>) -> vec3<f32> {
    let x = meshes[m].w;
    if (x == NONE) { return p; }
    return rotate_to_object(x, p - translation(x)) * xforms[4u * x + 3u].y;
}

// A world direction or normal in the space of object `m`, as a unit vector.
fn normal_to_world(m: u32, n: vec3<f32>) -> vec3<f32> {
    let x = meshes[m].w;
    if (x == NONE) { return n; }
    return rotate_to_world(x, n);
}
@group(0) @binding(4) var<storage, read_write> result: array<f32>;
@group(0) @binding(5) var<storage, read_write> debug: array<f32>;

// Mesh-level hierarchy: nodes hold bounds; `info.y > 0` marks a leaf whose meshes
// are `items[info.x .. info.x + info.y]`; interior nodes' children are at
// `info.x` and `info.x + 1`.
struct Node {
    lo: vec4<f32>,
    hi: vec4<f32>,
    info: vec4<u32>,
};
@group(0) @binding(9) var<storage, read> nodes: array<Node>;
@group(0) @binding(10) var<storage, read> items: array<u32>;
// Per-mesh triangle hierarchies, same layout; leaves hold `blas_items[info.x ..
// info.x + info.y]`, which are triangle indices.
@group(0) @binding(25) var<storage, read> blas: array<Node>;
@group(0) @binding(26) var<storage, read> blas_items: array<u32>;
// Per `blas` node, 27 samples: the projected area of its front-facing
// triangles seen from direction (i, j, k), i, j, k in {-1, 0, 1}, at index
// 9(i+1) + 3(j+1) + (k+1).
@group(0) @binding(29) var<storage, read> blas_cov: array<f32>;

const NONE: u32 = 0xffffffffu;
// Traversal stack: the mesh hierarchy's entries, then (above them) the current
// mesh's triangle hierarchy.
const STACK: u32 = 48u;

// Traversal stacks: a per-thread array, or with WORKGROUP_STACKS a slice of a
// workgroup array (interleaved across the 64 invocations of a workgroup). M4
// gate 5 found the workgroup version 2.6× slower on the M4: 12 KB per workgroup
// limits how many workgroups run at once. Every entry point that traverses
// sets `lane` and runs 64 invocations per workgroup. (The kernel shrinks
// `wg_stacks` when it is unused.)
override WORKGROUP_STACKS: bool = false;
var<workgroup> wg_stacks: array<u32, 3072>;
var<private> thread_stack: array<u32, STACK>;
var<private> lane: u32 = 0u;

fn stack_get(i: u32) -> u32 {
    if (WORKGROUP_STACKS) { return wg_stacks[i * 64u + lane]; }
    return thread_stack[i];
}

fn stack_set(i: u32, v: u32) {
    if (WORKGROUP_STACKS) { wg_stacks[i * 64u + lane] = v; } else { thread_stack[i] = v; }
}

// Debug mode 3: clipped polygons are appended to `debug` from this offset.
var<private> debug_cursor: u32 = 0u;

struct Cone {
    apex: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    forward: vec3<f32>,
    window: vec4<f32>, // u0, v0, u1, v1
    far_normal: vec3<f32>,
    far_offset: f32,
    // The receiver's normal, for sky cones (zero: none). Geometry wholly
    // below the receiver's tangent plane can only block directions below its
    // horizon, whose light it never receives, so the walk skips it.
    horizon: vec3<f32>,
};

fn lambert_edge(n: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let an = normalize(a);
    let bn = normalize(b);
    let g = cross(an, bn);
    let len = length(g);
    if (len > 1e-12) {
        // atan2 rather than acos(dot): stable for the tiny angles of far, small bins.
        return atan2(len, dot(an, bn)) * dot(n, g / len);
    }
    return 0.0;
}

fn direction(c: Cone, s: f32, t: f32) -> vec3<f32> {
    return c.forward + c.right * (c.window.x + s * (c.window.z - c.window.x))
        + c.up * (c.window.y + t * (c.window.w - c.window.y));
}

fn to_local(c: Cone, p: vec3<f32>) -> vec3<f32> {
    let d = p - c.apex;
    return vec3<f32>(dot(c.right, d), dot(c.up, d), dot(c.forward, d));
}

const NEAR: f32 = 1e-4;

fn may_touch(c: Cone, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    let f0 = c.far_offset - dot(c.far_normal, v0);
    let f1 = c.far_offset - dot(c.far_normal, v1);
    let f2 = c.far_offset - dot(c.far_normal, v2);
    if (f0 < 0.0 && f1 < 0.0 && f2 < 0.0) { return false; }
    if (dot(c.horizon, v0 - c.apex) < 0.0 && dot(c.horizon, v1 - c.apex) < 0.0
        && dot(c.horizon, v2 - c.apex) < 0.0) { return false; }
    let p0 = to_local(c, v0);
    let p1 = to_local(c, v1);
    let p2 = to_local(c, v2);
    if (p0.z < NEAR && p1.z < NEAR && p2.z < NEAR) { return false; }
    let w = c.window;
    if (p0.x - w.x * p0.z < 0.0 && p1.x - w.x * p1.z < 0.0 && p2.x - w.x * p2.z < 0.0) { return false; }
    if (w.z * p0.z - p0.x < 0.0 && w.z * p1.z - p1.x < 0.0 && w.z * p2.z - p2.x < 0.0) { return false; }
    if (p0.y - w.y * p0.z < 0.0 && p1.y - w.y * p1.z < 0.0 && p2.y - w.y * p2.z < 0.0) { return false; }
    if (w.w * p0.z - p0.y < 0.0 && w.w * p1.z - p1.y < 0.0 && w.w * p2.z - p2.y < 0.0) { return false; }
    return true;
}

fn may_touch_box(c: Cone, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    var far_out = true;
    var near_out = true;
    var out_u0 = true;
    var out_u1 = true;
    var out_v0 = true;
    var out_v1 = true;
    var below = true;
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u), select(lo.z, hi.z, (k & 4u) != 0u));
        if (c.far_offset - dot(c.far_normal, corner) >= 0.0) { far_out = false; }
        if (dot(c.horizon, corner - c.apex) >= 0.0) { below = false; }
        let p = to_local(c, corner);
        if (p.z >= NEAR) { near_out = false; }
        if (p.x - c.window.x * p.z >= 0.0) { out_u0 = false; }
        if (c.window.z * p.z - p.x >= 0.0) { out_u1 = false; }
        if (p.y - c.window.y * p.z >= 0.0) { out_v0 = false; }
        if (c.window.w * p.z - p.y >= 0.0) { out_v1 = false; }
    }
    return !(far_out || near_out || out_u0 || out_u1 || out_v0 || out_v1 || below);
}

struct Poly3 { v: array<vec3<f32>, 10>, n: u32 };
struct Poly2 { v: array<vec2<f32>, 10>, n: u32 };

// Keeps the part of `p` where dot(normal, x) <= offset.
fn clip_plane3(p: Poly3, normal: vec3<f32>, offset: f32) -> Poly3 {
    var out: Poly3;
    out.n = 0u;
    for (var i = 0u; i < p.n; i++) {
        let a = p.v[i];
        let b = p.v[(i + 1u) % p.n];
        let da = offset - dot(normal, a);
        let db = offset - dot(normal, b);
        if (da >= 0.0) { out.v[out.n] = a; out.n++; }
        if ((da >= 0.0) != (db >= 0.0)) { out.v[out.n] = a + (b - a) * (da / (da - db)); out.n++; }
    }
    return out;
}

// Keeps the part of `p` where dot(normal, x) <= offset, in the projected plane.
fn clip_line2(p: Poly2, normal: vec2<f32>, offset: f32) -> Poly2 {
    var out: Poly2;
    out.n = 0u;
    for (var i = 0u; i < p.n; i++) {
        let a = p.v[i];
        let b = p.v[(i + 1u) % p.n];
        let da = offset - dot(normal, a);
        let db = offset - dot(normal, b);
        if (da >= 0.0) { out.v[out.n] = a; out.n++; }
        if ((da >= 0.0) != (db >= 0.0)) { out.v[out.n] = a + (b - a) * (da / (da - db)); out.n++; }
    }
    return out;
}

// Adds an edge's signed swept area to the cells it crosses (accumulation rasterizer).
fn draw_edge(acc: ptr<function, array<f32, 68>>, span: ptr<function, vec2<u32>>, q0: vec2<f32>, q1: vec2<f32>, weight: f32) {
    if (abs(q0.y - q1.y) <= 1.1920929e-7) { return; }
    var dir = 1.0;
    var p0 = q0;
    var p1 = q1;
    if (q0.y > q1.y) { dir = -1.0; p0 = q1; p1 = q0; }
    let dxdy = (p1.x - p0.x) / (p1.y - p0.y);
    var x = p0.x;
    let y_end = min(u32(ceil(p1.y)), GRID);
    for (var y = u32(p0.y); y < y_end; y++) {
        let row = y * GRID;
        // Every write this row lands in cells row .. row + GRID + 1.
        (*span).x = min((*span).x, row);
        (*span).y = max((*span).y, row + GRID + 1u);
        let dy = min(f32(y + 1u), p1.y) - max(f32(y), p0.y);
        // Clamp: stepping along an edge on the grid boundary can drift a hair
        // outside it, and floor(-1e-9) = -1 would credit the row to the wrong cell.
        let x_next = clamp(x + dxdy * dy, 0.0, f32(GRID));
        let d = dy * dir * weight;
        let x0 = min(x, x_next);
        let x1 = max(x, x_next);
        let x0_floor = floor(x0);
        let x0i = u32(x0_floor);
        let x1_ceil = ceil(x1);
        let x1i = u32(x1_ceil);
        if (x1i <= x0i + 1u) {
            let xmf = 0.5 * (x + x_next) - x0_floor;
            (*acc)[row + x0i] += d - d * xmf;
            (*acc)[row + x0i + 1u] += d * xmf;
        } else {
            let s = 1.0 / (x1 - x0);
            let x0f = x0 - x0_floor;
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
            let x1f = x1 - x1_ceil + 1.0;
            let am = 0.5 * s * x1f * x1f;
            (*acc)[row + x0i] += d * a0;
            if (x1i == x0i + 2u) {
                (*acc)[row + x0i + 1u] += d * (1.0 - a0 - am);
            } else {
                let a1 = s * (1.5 - x0f);
                (*acc)[row + x0i + 1u] += d * (a1 - a0);
                for (var xi = x0i + 2u; xi < x1i - 1u; xi++) {
                    (*acc)[row + xi] += d * s;
                }
                let a2 = a1 + f32(x1i - x0i - 3u) * s;
                (*acc)[row + x1i - 1u] += d * (1.0 - a2 - am);
            }
            (*acc)[row + x1i] += d * am;
        }
        x = x_next;
    }
}

fn rasterize(c: Cone, acc: ptr<function, array<f32, 68>>, span: ptr<function, vec2<u32>>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    var p: Poly3;
    p.v[0] = v0;
    p.v[1] = v1;
    p.v[2] = v2;
    p.n = 3u;
    // Clip only when needed: with every vertex inside, clipping returns the same
    // polygon, so skipping it changes nothing.
    let f0 = c.far_offset - dot(c.far_normal, v0);
    let f1 = c.far_offset - dot(c.far_normal, v1);
    let f2 = c.far_offset - dot(c.far_normal, v2);
    if (f0 < 0.0 || f1 < 0.0 || f2 < 0.0) {
        p = clip_plane3(p, c.far_normal, c.far_offset);
    }
    var all_near_ok = true;
    for (var i = 0u; i < p.n; i++) {
        p.v[i] = to_local(c, p.v[i]);
        if (p.v[i].z < NEAR) { all_near_ok = false; }
    }
    // Keep z >= NEAR, i.e. dot(-z, x) <= -NEAR.
    if (!all_near_ok) {
        p = clip_plane3(p, vec3<f32>(0.0, 0.0, -1.0), -NEAR);
    }
    if (p.n < 3u) { return false; }
    let su = f32(GRID) / (c.window.z - c.window.x);
    let sv = f32(GRID) / (c.window.w - c.window.y);
    var q: Poly2;
    q.n = p.n;
    for (var i = 0u; i < p.n; i++) {
        q.v[i] = vec2<f32>((p.v[i].x / p.v[i].z - c.window.x) * su, (p.v[i].y / p.v[i].z - c.window.y) * sv);
    }
    let g = f32(GRID);
    var inside = true;
    for (var i = 0u; i < q.n; i++) {
        if (any(q.v[i] < vec2<f32>(0.0)) || any(q.v[i] > vec2<f32>(g))) { inside = false; }
    }
    if (!inside) {
        q = clip_line2(q, vec2<f32>(-1.0, 0.0), 0.0); // x >= 0
        q = clip_line2(q, vec2<f32>(1.0, 0.0), g);    // x <= g
        q = clip_line2(q, vec2<f32>(0.0, -1.0), 0.0); // y >= 0
        q = clip_line2(q, vec2<f32>(0.0, 1.0), g);    // y <= g
    }
    if (light.counts.y == 3u) {
        debug[debug_cursor] = f32(q.n);
        for (var i = 0u; i < q.n; i++) {
            debug[debug_cursor + 1u + 2u * i] = q.v[i].x;
            debug[debug_cursor + 2u + 2u * i] = q.v[i].y;
        }
        debug_cursor += 21u;
    }
    if (q.n < 3u) { return false; }
    for (var i = 0u; i < q.n; i++) {
        let a = clamp(q.v[i], vec2<f32>(0.0), vec2<f32>(g));
        let b = clamp(q.v[(i + 1u) % q.n], vec2<f32>(0.0), vec2<f32>(g));
        draw_edge(acc, span, a, b, 1.0);
    }
    return true;
}

// --- Prefiltered cones: splat small hierarchy nodes instead of descending. ---

fn cov_sample(n: u32, i: i32, j: i32, k: i32) -> f32 {
    return blas_cov[n * 27u + u32(9 * (i + 1) + 3 * (j + 1) + (k + 1))];
}

// Projected area of node `n`'s front-facing triangles seen from direction `v`
// (unit, pointing toward the viewer), interpolated bilinearly on the face of
// the cube of sample directions that `v` points through.
fn node_front_area(n: u32, v: vec3<f32>) -> f32 {
    let a = abs(v);
    var axis = 0u;
    if (a.y > a.x && a.y >= a.z) { axis = 1u; } else if (a.z > a.x && a.z > a.y) { axis = 2u; }
    var main: f32;
    var s: f32;
    var t: f32;
    if (axis == 0u) { main = v.x; s = v.y / a.x; t = v.z / a.x; }
    else if (axis == 1u) { main = v.y; s = v.x / a.y; t = v.z / a.y; }
    else { main = v.z; s = v.x / a.z; t = v.y / a.z; }
    let m = select(-1, 1, main >= 0.0);
    let s0 = select(0, -1, s < 0.0);
    let t0 = select(0, -1, t < 0.0);
    let fs = s - f32(s0);
    let ft = t - f32(t0);
    var f: array<f32, 4>;
    for (var c = 0; c < 4; c++) {
        let gs = s0 + (c & 1);
        let gt = t0 + (c >> 1);
        if (axis == 0u) { f[c] = cov_sample(n, m, gs, gt); }
        else if (axis == 1u) { f[c] = cov_sample(n, gs, m, gt); }
        else { f[c] = cov_sample(n, gs, gt, m); }
    }
    return mix(mix(f[0], f[1], fs), mix(f[2], f[3], fs), ft);
}

// Clockwise convex hull of 8 points (the winding front-facing triangles have
// in bin space, so splats add to them instead of cancelling).
fn hull8(pts: array<vec2<f32>, 8>) -> Poly2 {
    var start = 0u;
    for (var k = 1u; k < 8u; k++) {
        if (pts[k].x < pts[start].x || (pts[k].x == pts[start].x && pts[k].y < pts[start].y)) { start = k; }
    }
    var out: Poly2;
    out.n = 0u;
    var cur = start;
    loop {
        out.v[out.n] = pts[cur];
        out.n++;
        var next = (cur + 1u) % 8u;
        for (var k = 0u; k < 8u; k++) {
            let e = pts[next] - pts[cur];
            let f = pts[k] - pts[cur];
            let cr = e.x * f.y - e.y * f.x;
            if (cr > 0.0 || (cr == 0.0 && dot(f, f) > dot(e, e))) { next = k; }
        }
        cur = next;
        if (cur == start || out.n >= 8u) { break; }
    }
    return out;
}

// A cone against hierarchy node `n` (box lo..hi), projecting the box's corners
// once for both questions: may the box touch the cone, and (with prefiltering)
// is it small enough to splat? Returns 0 if the box cannot touch the cone, 1
// if it was splatted with nothing to draw, 2 if splatted and drawn, 3 if the
// cone must descend into it.
fn node_cone_test(c: Cone, acc: ptr<function, array<f32, 68>>, span: ptr<function, vec2<u32>>, n: u32, lo: vec3<f32>, hi: vec3<f32>) -> u32 {
    let g = f32(GRID);
    var pts: array<vec2<f32>, 8>;
    var mn = vec2<f32>(3.4e38);
    var mx = vec2<f32>(-3.4e38);
    var far_out = true;
    var near_out = true;
    var out_u0 = true;
    var out_u1 = true;
    var out_v0 = true;
    var out_v1 = true;
    var below = true;
    // Splattable: every corner between the near and far planes.
    var inside = prefilter_bins > 0.0;
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u), select(lo.z, hi.z, (k & 4u) != 0u));
        if (dot(c.horizon, corner - c.apex) >= 0.0) { below = false; }
        let in_far = c.far_offset - dot(c.far_normal, corner) >= 0.0;
        if (in_far) { far_out = false; } else { inside = false; }
        let p = to_local(c, corner);
        if (p.z >= NEAR) { near_out = false; } else { inside = false; }
        if (p.x - c.window.x * p.z >= 0.0) { out_u0 = false; }
        if (c.window.z * p.z - p.x >= 0.0) { out_u1 = false; }
        if (p.y - c.window.y * p.z >= 0.0) { out_v0 = false; }
        if (c.window.w * p.z - p.y >= 0.0) { out_v1 = false; }
        if (inside) {
            let q = vec2<f32>((p.x / p.z - c.window.x) / (c.window.z - c.window.x) * g,
                              (p.y / p.z - c.window.y) / (c.window.w - c.window.y) * g);
            pts[k] = q;
            mn = min(mn, q);
            mx = max(mx, q);
        }
    }
    if (far_out || near_out || out_u0 || out_u1 || out_v0 || out_v1 || below) { return 0u; }
    if (!inside || max(mx.x - mn.x, mx.y - mn.y) > prefilter_bins) { return 3u; }
    if (mx.x <= 0.0 || mx.y <= 0.0 || mn.x >= g || mn.y >= g) { return 1u; }
    let e = hi - lo;
    let v = normalize(c.apex - (lo + hi) * 0.5);
    let silhouette = abs(v.x) * e.y * e.z + abs(v.y) * e.x * e.z + abs(v.z) * e.x * e.y;
    if (silhouette <= 1e-30) { return 1u; }
    let alpha = node_front_area(n, v) / silhouette;
    if (alpha <= 0.0) { return 1u; }
    var q = hull8(pts);
    q = clip_line2(q, vec2<f32>(-1.0, 0.0), 0.0);
    q = clip_line2(q, vec2<f32>(1.0, 0.0), g);
    q = clip_line2(q, vec2<f32>(0.0, -1.0), 0.0);
    q = clip_line2(q, vec2<f32>(0.0, 1.0), g);
    if (q.n < 3u) { return 1u; }
    for (var i = 0u; i < q.n; i++) {
        let a = clamp(q.v[i], vec2<f32>(0.0), vec2<f32>(g));
        let b = clamp(q.v[(i + 1u) % q.n], vec2<f32>(0.0), vec2<f32>(g));
        draw_edge(acc, span, a, b, alpha);
    }
    return 2u;
}

// Rasterizes triangle `t` into `acc` if it faces the apex and may touch the cone.
fn cover_triangle(c: Cone, acc: ptr<function, array<f32, 68>>, span: ptr<function, vec2<u32>>, t: u32) -> bool {
    if (u32(light.params.z) == 4u) { return false; }
    let v0 = tris[3u * t].xyz;
    let v1 = tris[3u * t + 1u].xyz;
    let v2 = tris[3u * t + 2u].xyz;
    let n = cross(v1 - v0, v2 - v0);
    if (dot(n, c.apex - v0) <= 0.0) { return false; }
    // A triangle facing the apex can only block directions going into it. If
    // every corner direction of the window points away from its plane, it
    // cannot touch the cone. (This also keeps a receiver's own huge supporting
    // triangle, nearly coplanar with the apex, out of f32 clipping.)
    if (dot(n, direction(c, 0.0, 0.0)) >= 0.0 && dot(n, direction(c, 1.0, 0.0)) >= 0.0
        && dot(n, direction(c, 0.0, 1.0)) >= 0.0 && dot(n, direction(c, 1.0, 1.0)) >= 0.0) { return false; }
    if (!may_touch(c, v0, v1, v2)) { return false; }
    return rasterize(c, acc, span, v0, v1, v2);
}

// Rasterizes one mesh's front-facing triangles into a fresh accumulation buffer
// and combines the resulting per-bin coverage into `cov` by max. Triangle
// hierarchy traversal uses `stack` from `base` up.
// The cone in the space of object `m`. Rotation and uniform scale preserve the
// cone's shape, so its window and bins are unchanged.
fn cone_to_object(c: Cone, m: u32) -> Cone {
    let x = meshes[m].w;
    if (x == NONE) { return c; }
    var o: Cone;
    o.apex = point_to_object(m, c.apex);
    o.right = rotate_to_object(x, c.right);
    o.up = rotate_to_object(x, c.up);
    o.forward = rotate_to_object(x, c.forward);
    o.window = c.window;
    o.far_normal = rotate_to_object(x, c.far_normal);
    o.far_offset = (c.far_offset - dot(c.far_normal, translation(x))) * xforms[4u * x + 3u].y;
    o.horizon = rotate_to_object(x, c.horizon);
    return o;
}

// The prefilter threshold (bins) of the cone being traced: the light's
// (`light.params.x`), or for sky cones that times `light.objects.y` (as f32
// bits; 0 means 1).
var<private> prefilter_bins: f32 = 0.0;

// How a bin's coverage by one mesh combines with other meshes': the maximum
// (sun cones, whose tiny bins rarely hold two partial occluders), or as
// independent occluders, 1 − Π(1 − c) (sky cones, whose 11° bins often do).
var<private> independent_occluders: bool = false;

// Diagnostics (`KernelOptions::count_work`): the work a thread's cones did.
// work_a: object nodes tested, object nodes splatted, meshes entered,
// triangle-hierarchy nodes tested; work_b: of those, splatted, triangles
// tested, triangles rasterized, cones traced. Compiled away when off.
override COUNT_WORK: bool = false;
var<private> work_a: vec4<u32> = vec4<u32>(0u);
var<private> work_b: vec4<u32> = vec4<u32>(0u);
// work_c: object nodes and triangle-hierarchy nodes the packet visited; of
// the triangle-hierarchy nodes tested, those of the receiver's own object;
// meshes entered that were the receiver's own.
var<private> work_c: vec4<u32> = vec4<u32>(0u);
// The object the traced cones start on (diagnostics only).
var<private> own_object: u32 = 0xffffffffu;

// Packet traversal (M5 gate 5): the invocations of a subgroup that trace
// cones together walk each hierarchy together. A node is visited when any of
// them needs it, and each acts only on the nodes its own walk would visit, in
// the same order, so every result equals walking alone; but the subgroup no
// longer runs 32 different walks one after another. Where the device has
// subgroups, the kernel makes `packet_any` a subgroupAny in pipelines with
// PACKETS; otherwise each invocation is its own packet. Traversal loops leave
// only on uniform conditions, so the invocations stay together. Packets pay
// where neighbouring invocations walk alike (pixels, M5 gate 5) and cost
// where they do not (the view cache's bakes, whose texels can be far apart).
override PACKETS: bool = true;

fn packet_any(x: bool) -> bool {
    return x;
}

// Per invocation: which traversal stack entries it needs (bit i: entry i).
var<private> need_lo: u32 = 0u;
var<private> need_hi: u32 = 0u;

fn need_get(i: u32) -> bool {
    return ((select(need_lo, need_hi, i >= 32u) >> (i & 31u)) & 1u) != 0u;
}

fn need_put(i: u32, v: bool) {
    let bit = 1u << (i & 31u);
    if (i < 32u) {
        need_lo = select(need_lo & ~bit, need_lo | bit, v);
    } else {
        need_hi = select(need_hi & ~bit, need_hi | bit, v);
    }
}

fn add_mesh_coverage(world_cone: Cone, m: u32, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>, i: u32, base: u32) {
    add_mesh_coverage_if(true, world_cone, m, cov, full, i, base);
}

// `add_mesh_coverage` for the invocations with `on` set; the others follow
// the packet's walk without acting.
fn add_mesh_coverage_if(on: bool, world_cone: Cone, m: u32, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>, i: u32, base: u32) {
    if (u32(light.params.z) == 3u) { return; }
    if (!packet_any(on)) { return; }
    if (COUNT_WORK && on) {
        work_a.z++;
        if (m == own_object) { work_c.w++; }
    }
    let c = cone_to_object(world_cone, m);
    let range = meshes[m];
    var acc: array<f32, 68>;
    for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
    // Cells written by this mesh's edges; the running sum is zero outside them.
    var span = vec2<u32>(68u, 0u);
    var touched = false;
    if (light.counts.z == 2u) {
        if (range.z != NONE) {
            var sp = base + 1u;
            stack_set(base, range.z);
            need_put(base, on);
            while (sp > base) {
                sp--;
                let n = stack_get(sp);
                let mine = need_get(sp);
                if (!packet_any(mine)) { continue; }
                if (COUNT_WORK) { work_c.y++; }
                let node = blas[n];
                var descend = false;
                if (mine) {
                    let test = node_cone_test(c, &acc, &span, n, node.lo.xyz, node.hi.xyz);
                    if (COUNT_WORK) {
                        work_a.w++;
                        if (m == own_object) { work_c.z++; }
                        if (test == 1u || test == 2u) { work_b.x++; }
                    }
                    if (test == 2u) { touched = true; }
                    descend = test == 3u;
                }
                if (!packet_any(descend)) { continue; }
                if (node.info.y > 0u) {
                    if (descend) {
                        for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                            let drawn = cover_triangle(c, &acc, &span, blas_items[k]);
                            if (COUNT_WORK) {
                                work_b.y++;
                                if (drawn) { work_b.z++; }
                            }
                            if (drawn) { touched = true; }
                        }
                    }
                } else {
                    stack_set(sp, node.info.x);
                    stack_set(sp + 1u, node.info.x + 1u);
                    need_put(sp, descend);
                    need_put(sp + 1u, descend);
                    sp += 2u;
                }
            }
        }
    } else if (on) {
        for (var t = range.x; t < range.x + range.y; t++) {
            if (cover_triangle(c, &acc, &span, t)) { touched = true; }
        }
    }
    if (light.counts.y == 2u && on) {
        for (var b = 0u; b < 64u; b++) { debug[i * BINS + b] = acc[b]; }
    }
    if (touched) {
        var running = 0.0;
        for (var b = span.x; b < min(span.y, BINS); b++) {
            running += acc[b];
            let covered = abs(running);
            if (covered > 1e-6) {
                let before = (*cov)[b];
                if (independent_occluders) {
                    (*cov)[b] = 1.0 - (1.0 - before) * (1.0 - min(covered, 1.0));
                } else {
                    (*cov)[b] = max(before, min(covered, 1.0));
                }
                if (before < 1.0 && (*cov)[b] >= 1.0) { (*full)++; }
            }
        }
    }
}

// A cone from `apex` to the light, if the light can shine on it at all.
struct LightCone {
    c: Cone,
    ok: bool,
};

fn light_cone(apex: vec3<f32>) -> LightCone {
    var out: LightCone;
    out.ok = false;
    // Light cones prefilter at the light's threshold (sky cones set theirs).
    prefilter_bins = light.params.x;
    var forward = normalize(cross(light.edge_u.xyz, light.edge_v.xyz));
    // Behind a one-sided light: no direct light.
    if (light.counts.w == 1u && dot(forward, apex - light.corner.xyz) <= 0.0) { return out; }
    if (dot(forward, light.corner.xyz - apex) < 0.0) { forward = -forward; }
    let dist = dot(forward, light.corner.xyz - apex);
    if (dist <= 1e-6) { return out; }
    let right = normalize(light.edge_u.xyz);
    let up = cross(forward, right);
    let rel = light.corner.xyz - apex;
    var lo = vec2<f32>(3.4e38);
    var hi = vec2<f32>(-3.4e38);
    for (var k = 0u; k < 4u; k++) {
        let corner = rel + light.edge_u.xyz * f32(k & 1u) + light.edge_v.xyz * f32((k >> 1u) & 1u);
        let uv = vec2<f32>(dot(right, corner), dot(up, corner)) / dist;
        lo = min(lo, uv);
        hi = max(hi, uv);
    }
    out.c.apex = apex;
    out.c.right = right;
    out.c.up = up;
    out.c.forward = forward;
    out.c.window = vec4<f32>(lo, hi);
    out.c.far_normal = forward;
    out.c.far_offset = dot(forward, apex) + dist * (1.0 - 1e-5);
    out.ok = true;
    return out;
}

// Object-level prefiltering (M5 gate 3): if object-hierarchy node `n` (box
// lo..hi) lies wholly between the cone's near and far planes and projects to
// at most the prefilter threshold in bins, splats its hull into `cov` as one
// more mesh, with opacity 1 − exp(−front area / silhouette) (its objects as
// independent occluders, so overlap saturates), and returns true. Otherwise
// returns false: descend as usual.
fn splat_object_node(c: Cone, n: u32, lo: vec3<f32>, hi: vec3<f32>, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>) -> bool {
    if (light.objects.x == NONE || prefilter_bins <= 0.0) { return false; }
    let g = f32(GRID);
    var pts: array<vec2<f32>, 8>;
    var mn = vec2<f32>(3.4e38);
    var mx = vec2<f32>(-3.4e38);
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u), select(lo.z, hi.z, (k & 4u) != 0u));
        if (c.far_offset - dot(c.far_normal, corner) < 0.0) { return false; }
        let p = to_local(c, corner);
        if (p.z < NEAR) { return false; }
        let q = vec2<f32>((p.x / p.z - c.window.x) / (c.window.z - c.window.x) * g,
                          (p.y / p.z - c.window.y) / (c.window.w - c.window.y) * g);
        pts[k] = q;
        mn = min(mn, q);
        mx = max(mx, q);
    }
    if (max(mx.x - mn.x, mx.y - mn.y) > prefilter_bins) { return false; }
    if (mx.x <= 0.0 || mx.y <= 0.0 || mn.x >= g || mn.y >= g) { return true; }
    let e = hi - lo;
    let v = normalize(c.apex - (lo + hi) * 0.5);
    let silhouette = abs(v.x) * e.y * e.z + abs(v.y) * e.x * e.z + abs(v.z) * e.x * e.y;
    if (silhouette <= 1e-30) { return true; }
    let alpha = 1.0 - exp(-node_front_area(light.objects.x + n, v) / silhouette);
    if (alpha <= 0.0) { return true; }
    var q = hull8(pts);
    q = clip_line2(q, vec2<f32>(-1.0, 0.0), 0.0);
    q = clip_line2(q, vec2<f32>(1.0, 0.0), g);
    q = clip_line2(q, vec2<f32>(0.0, -1.0), 0.0);
    q = clip_line2(q, vec2<f32>(0.0, 1.0), g);
    if (q.n < 3u) { return true; }
    var acc: array<f32, 68>;
    for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
    var span = vec2<u32>(68u, 0u);
    for (var i = 0u; i < q.n; i++) {
        let a = clamp(q.v[i], vec2<f32>(0.0), vec2<f32>(g));
        let b = clamp(q.v[(i + 1u) % q.n], vec2<f32>(0.0), vec2<f32>(g));
        draw_edge(&acc, &span, a, b, alpha);
    }
    var running = 0.0;
    for (var b = span.x; b < min(span.y, BINS); b++) {
        running += acc[b];
        let covered = min(abs(running), 1.0);
        if (covered > 1e-6) {
            let before = (*cov)[b];
            if (independent_occluders) {
                (*cov)[b] = 1.0 - (1.0 - before) * (1.0 - covered);
            } else {
                (*cov)[b] = max(before, covered);
            }
            if (before < 1.0 && (*cov)[b] >= 1.0) { (*full)++; }
        }
    }
    return true;
}

// Adds the coverage of every mesh in the bound hierarchy (or every mesh) to `cov`.
fn scene_coverage(c: Cone, i: u32, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>) {
    scene_coverage_if(true, c, i, cov, full);
}

// `scene_coverage` for the invocations with `on` set; the others follow
// the packet's walk without acting. An invocation stops needing nodes once
// every bin is fully covered, as nothing further can change its result.
fn scene_coverage_if(on: bool, c: Cone, i: u32, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>) {
    if (COUNT_WORK && on) { work_b.w++; }
    if (light.counts.z >= 1u) {
        var sp = 1u;
        stack_set(0, 0u);
        need_put(0u, on);
        while (sp > 0u) {
            sp--;
            let n = stack_get(sp);
            let mine = need_get(sp) && *full < BINS;
            if (!packet_any(mine)) { continue; }
            if (COUNT_WORK) { work_c.x++; }
            let node = nodes[n];
            var descend = false;
            if (mine) {
                if (COUNT_WORK) { work_a.x++; }
                if (may_touch_box(c, node.lo.xyz, node.hi.xyz)) {
                    if (splat_object_node(c, n, node.lo.xyz, node.hi.xyz, cov, full)) {
                        if (COUNT_WORK) { work_a.y++; }
                    } else {
                        descend = true;
                    }
                }
            }
            if (!packet_any(descend)) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    add_mesh_coverage_if(descend && items[k] != skip_object, c, items[k], cov, full, i, sp);
                }
            } else {
                stack_set(sp, node.info.x);
                stack_set(sp + 1u, node.info.x + 1u);
                need_put(sp, descend);
                need_put(sp + 1u, descend);
                sp += 2u;
            }
        }
    } else {
        for (var m = 0u; m < light.counts.x; m++) {
            add_mesh_coverage_if(on && *full < BINS && m != skip_object, c, m, cov, full, i, 0u);
        }
    }
}

// Irradiance through the uncovered part of the cone. Bin weights (projected
// solid angle, Lambert) are computed only now, as they are summed, so no
// per-thread weights array is needed.
fn bin_weight(c: Cone, normal: vec3<f32>, b: u32) -> f32 {
    let col = f32(b % GRID);
    let row = f32(b / GRID);
    let g = f32(GRID);
    let d0 = direction(c, col / g, row / g);
    let d1 = direction(c, (col + 1.0) / g, row / g);
    let d2 = direction(c, (col + 1.0) / g, (row + 1.0) / g);
    let d3 = direction(c, col / g, (row + 1.0) / g);
    return abs(0.5 * (lambert_edge(normal, d0, d1) + lambert_edge(normal, d1, d2)
        + lambert_edge(normal, d2, d3) + lambert_edge(normal, d3, d0)));
}

fn cone_visible(c: Cone, normal: vec3<f32>, cov: ptr<function, array<f32, 64>>) -> f32 {
    // With the whole window above the receiver's horizon, every bin's weight is
    // positive and the bins' weights add up to the window's: the visible part
    // is the window's weight (one quad) minus the weights of the bins with
    // coverage, so an open cone needs one quad instead of 64.
    let d00 = direction(c, 0.0, 0.0);
    let d10 = direction(c, 1.0, 0.0);
    let d11 = direction(c, 1.0, 1.0);
    let d01 = direction(c, 0.0, 1.0);
    if (dot(normal, d00) > 0.0 && dot(normal, d10) > 0.0 && dot(normal, d11) > 0.0 && dot(normal, d01) > 0.0) {
        var blocked = 0.0;
        for (var b = 0u; b < BINS; b++) {
            let covered = (*cov)[b];
            if (covered > 0.0) { blocked += covered * bin_weight(c, normal, b); }
        }
        let total = abs(0.5 * (lambert_edge(normal, d00, d10) + lambert_edge(normal, d10, d11)
            + lambert_edge(normal, d11, d01) + lambert_edge(normal, d01, d00)));
        return max(total - blocked, 0.0);
    }
    var visible = 0.0;
    for (var b = 0u; b < BINS; b++) {
        visible += (1.0 - (*cov)[b]) * bin_weight(c, normal, b);
    }
    return visible;
}

// Irradiance per unit radiance at `apex` (surface normal `normal`) from the
// light, through a mask cone. `i` only selects where debug output goes.
fn cone_irradiance(i: u32, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    return cone_irradiance_if(true, i, apex, normal);
}

// `cone_irradiance` where `on` (0 elsewhere), keeping the packet together.
fn cone_irradiance_if(on: bool, i: u32, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    let stage = u32(light.params.z);
    if (stage == 1u) { return 0.0; }
    let lc = light_cone(apex);
    let traced = on && lc.ok;
    // Explicit zeroing: don't rely on loop-scoped `var`s being re-initialized.
    var cov: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
    var full = 0u;
    if (stage != 2u) { scene_coverage_if(traced, lc.c, i, &cov, &full); }
    if (!traced) { return 0.0; }
    let visible = cone_visible(lc.c, normal, &cov);
    if (light.counts.y == 1u) {
        for (var b = 0u; b < BINS; b++) { debug[i * BINS + b] = cov[b]; }
    }
    return visible;
}

// --- Sky light: a uniform sky through six cube-face mask cones. ---

// Far enough that every scene lies inside a sky cone (relative to the origin).
const SKY_FAR: f32 = 1e7;

// The projected solid angle (Lambert) of bin `b` of cone `c` above the
// horizon of `normal`: the bin's quad of directions clipped to dot(normal, d)
// >= 0, then Lambert's edge integrals.
fn clipped_bin_weight(c: Cone, normal: vec3<f32>, b: u32) -> f32 {
    let col = f32(b % GRID);
    let row = f32(b / GRID);
    let g = f32(GRID);
    var d: array<vec3<f32>, 4>;
    d[0] = direction(c, col / g, row / g);
    d[1] = direction(c, (col + 1.0) / g, row / g);
    d[2] = direction(c, (col + 1.0) / g, (row + 1.0) / g);
    d[3] = direction(c, col / g, (row + 1.0) / g);
    var above = 0u;
    for (var k = 0u; k < 4u; k++) {
        if (dot(normal, d[k]) > 0.0) { above++; }
    }
    if (above == 4u) { return bin_weight(c, normal, b); }
    if (above == 0u) { return 0.0; }
    // Directions interpolate linearly along a bin's edges (the edges are
    // great-circle arcs), so the plane clip of the quad is exact.
    var poly: array<vec3<f32>, 6>;
    var n = 0u;
    for (var k = 0u; k < 4u; k++) {
        let a = d[k];
        let e = d[(k + 1u) % 4u];
        let sa = dot(normal, a);
        let se = dot(normal, e);
        if (sa >= 0.0) {
            poly[n] = a;
            n++;
        }
        if ((sa >= 0.0) != (se >= 0.0)) {
            poly[n] = a + (e - a) * (sa / (sa - se));
            n++;
        }
    }
    var sum = 0.0;
    for (var k = 0u; k < n; k++) {
        sum += lambert_edge(normal, poly[k], poly[(k + 1u) % n]);
    }
    return abs(0.5 * sum);
}

// The projected solid angle (Lambert) of cone `c`'s whole window above the
// horizon of `normal`: the window's quad of directions clipped to
// dot(normal, d) >= 0, then Lambert's edge integrals. Bins partition the
// window, so this is the sum of their `clipped_bin_weight`s.
fn clipped_window_weight(c: Cone, normal: vec3<f32>) -> f32 {
    var d: array<vec3<f32>, 4>;
    d[0] = direction(c, 0.0, 0.0);
    d[1] = direction(c, 1.0, 0.0);
    d[2] = direction(c, 1.0, 1.0);
    d[3] = direction(c, 0.0, 1.0);
    var poly: array<vec3<f32>, 6>;
    var n = 0u;
    for (var k = 0u; k < 4u; k++) {
        let a = d[k];
        let e = d[(k + 1u) % 4u];
        let sa = dot(normal, a);
        let se = dot(normal, e);
        if (sa >= 0.0) {
            poly[n] = a;
            n++;
        }
        if ((sa >= 0.0) != (se >= 0.0)) {
            poly[n] = a + (e - a) * (sa / (sa - se));
            n++;
        }
    }
    if (n < 3u) { return 0.0; }
    var sum = 0.0;
    for (var k = 0u; k < n; k++) {
        sum += lambert_edge(normal, poly[k], poly[(k + 1u) % n]);
    }
    return abs(0.5 * sum);
}

// Irradiance per unit sky radiance at `apex` (surface normal `normal`): the
// uncovered projected solid angle of the six faces of a cube of directions
// around it, each a mask cone reaching beyond the scene. Each face's light is
// its whole clipped window weight minus the weights of the bins with coverage
// (the same sum rearranged, as `cone_visible`'s fast path), so open bins cost
// nothing.
fn sky_irradiance(apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    return sky_irradiance_if(true, apex, normal);
}

// `sky_irradiance` where `on` (0 elsewhere), keeping the packet together.
fn sky_irradiance_if(on: bool, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    // Profiling stages, as for sun cones: 1 skips the cones, 2 keeps their
    // setup and weights but finds no coverage.
    let stage = u32(light.params.z);
    if (stage == 1u) { return 0.0; }
    var total = 0.0;
    for (var f = 0u; f < 6u; f++) {
        total += sky_face_irradiance(on, f, apex, normal);
    }
    return total;
}

// Sky faces in an object's frame (M5 gate 5): the transform whose rotation's
// columns are the cube's axes, or NONE for world axes. A uniform sky's light
// does not depend on the cube's orientation; an object's own sky occlusion,
// seen through faces in its frame, is the same for all its instances.
var<private> sky_frame: u32 = 0xffffffffu;
// An object the walks skip (NONE: none): the receiver's own, when its
// coverage comes from a precomputed mask.
var<private> skip_object: u32 = 0xffffffffu;

// Cube face `f` of the sky around `apex`, and whether any of its directions
// lies above the horizon of `normal` (and `on`).
struct SkyFace {
    c: Cone,
    above: bool,
};

fn sky_face(on: bool, f: u32, apex: vec3<f32>, normal: vec3<f32>) -> SkyFace {
    var scale = bitcast<f32>(light.objects.y);
    if (scale <= 0.0) { scale = 1.0; }
    prefilter_bins = light.params.x * scale;
    let axis = f / 2u;
    let sign = select(1.0, -1.0, (f & 1u) == 1u);
    var forward = vec3<f32>(0.0);
    forward[axis] = sign;
    var right = vec3<f32>(0.0);
    right[(axis + 1u) % 3u] = 1.0;
    if (sky_frame != NONE) {
        forward = rotate_to_world(sky_frame, forward);
        right = rotate_to_world(sky_frame, right);
    }
    var face: SkyFace;
    face.c.apex = apex;
    face.c.right = right;
    face.c.up = cross(forward, right);
    face.c.forward = forward;
    face.c.window = vec4<f32>(-1.0, -1.0, 1.0, 1.0);
    face.c.far_normal = forward;
    face.c.far_offset = dot(forward, apex) + SKY_FAR;
    // params.w bit 2: walk geometry below the horizon too (as before M5
    // gate 5).
    if ((u32(light.params.w) & 2u) == 0u) { face.c.horizon = normal; }
    // The face's directions are linear in the window, so its corners decide
    // whether any lies above the horizon.
    var above = false;
    for (var k = 0u; k < 4u; k++) {
        if (dot(normal, direction(face.c, f32(k & 1u), f32(k >> 1u))) > 0.0) { above = true; }
    }
    face.above = above && on;
    return face;
}

// The scene's coverage of a sky face (skipping `skip_object`), with meshes as
// independent occluders.
fn sky_face_coverage(face: SkyFace, cov: ptr<function, array<f32, 64>>) {
    for (var b = 0u; b < BINS; b++) { (*cov)[b] = 0.0; }
    var full = 0u;
    // params.w bit 1: combine by the maximum instead (a diagnostic).
    independent_occluders = (u32(light.params.w) & 1u) == 0u;
    if (u32(light.params.z) != 2u) { scene_coverage_if(face.above, face.c, 0u, cov, &full); }
    independent_occluders = false;
}

// Irradiance per unit sky radiance through a face with coverage `cov`.
fn sky_face_light(face: SkyFace, normal: vec3<f32>, cov: ptr<function, array<f32, 64>>) -> f32 {
    if (!face.above) { return 0.0; }
    var blocked = 0.0;
    for (var b = 0u; b < BINS; b++) {
        if ((*cov)[b] > 0.0) {
            blocked += (*cov)[b] * clipped_bin_weight(face.c, normal, b);
        }
    }
    return max(clipped_window_weight(face.c, normal) - blocked, 0.0);
}

// Irradiance per unit sky radiance through cube face `f` alone (0 if the
// face lies wholly below the horizon, or without `on`).
fn sky_face_irradiance(on: bool, f: u32, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    if (u32(light.params.z) == 1u) { return 0.0; }
    let face = sky_face(on, f, apex, normal);
    var cov: array<f32, 64>;
    sky_face_coverage(face, &cov);
    return sky_face_light(face, normal, &cov);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let i = gid.x;
    if (i >= arrayLength(&receivers)) { return; }
    // Same rule as camera hits: a receiver facing away from the light gets none.
    result[i] = hit_irradiance(i, receivers[i].point.xyz, receivers[i].normal.xyz);
}

// --- Camera rendering: ray cast to the nearest triangle, then a mask cone. ---

struct Camera {
    eye: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // x: width, y: height (pixels); z: tan(half vertical fov); w: radiance
    params: vec4<f32>,
};

@group(0) @binding(6) var<uniform> camera: Camera;
@group(0) @binding(7) var<storage, read_write> colors: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> albedo: array<vec4<f32>>;

const EDGE_TOLERANCE: f32 = 1e-5;

fn ray_triangle(o: vec3<f32>, d: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-12) { return -1.0; }
    let inv = 1.0 / det;
    let s = o - v0;
    // A small tolerance on the edges keeps meshes watertight: in f32, a ray
    // through a shared edge can otherwise miss both triangles.
    let u = dot(s, p) * inv;
    if (u < -EDGE_TOLERANCE || u > 1.0 + EDGE_TOLERANCE) { return -1.0; }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if (v < -EDGE_TOLERANCE || u + v > 1.0 + EDGE_TOLERANCE) { return -1.0; }
    return dot(e2, q) * inv;
}

struct Hit {
    distance: f32,
    mesh: u32,
    // Global triangle index.
    tri: u32,
    normal: vec3<f32>,
};

// Camera rays see front faces only (meshes are closed, and the showcase's room
// shell faces inward). Without this, a ray grazing a silhouette can hit the
// back of the adjacent face within the edge tolerance, which the lighting
// cache, whose texels lie on front faces, shades as unlit. Probe rays keep back
// faces: they flag them to detect probes inside geometry.
var<private> cull_back: bool = false;

fn hit_triangle(eye: vec3<f32>, dir: vec3<f32>, m: u32, t: u32, best: Hit) -> Hit {
    let v0 = tris[3u * t].xyz;
    let v1 = tris[3u * t + 1u].xyz;
    let v2 = tris[3u * t + 2u].xyz;
    if (cull_back && dot(cross(v1 - v0, v2 - v0), dir) > 0.0) { return best; }
    let d = ray_triangle(eye, dir, v0, v1, v2);
    // Ties (coplanar faces, shared edges) go to the lower mesh, then the lower
    // triangle, so the result does not depend on traversal order.
    if (d > 0.0 && (d < best.distance || (d == best.distance
        && (m < best.mesh || (m == best.mesh && t < best.tri))))) {
        return Hit(d, m, t, normalize(cross(v1 - v0, v2 - v0)));
    }
    return best;
}

// Slab test of a ray against a box: whether it may hold a hit nearer than `limit`.
fn ray_box(eye: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, limit: f32) -> bool {
    return ray_box_entry(eye, inv, lo, hi, limit) >= 0.0;
}

// Where a ray enters a box (0 if it starts inside), or -1 if it misses it or
// enters beyond `limit`.
fn ray_box_entry(eye: vec3<f32>, inv: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, limit: f32) -> f32 {
    let t0 = (lo - eye) * inv;
    let t1 = (hi - eye) * inv;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (far < max(near, 0.0) || near > limit) { return -1.0; }
    return max(near, 0.0);
}

// Pushes the children of an interior node for ray traversal: nearest first
// (the one the ray enters first is popped first, and a child the ray misses is
// skipped), or both in a fixed order.
fn push_children(eye: vec3<f32>, inv: vec3<f32>, left: u32, lo_l: vec3<f32>, hi_l: vec3<f32>,
                 lo_r: vec3<f32>, hi_r: vec3<f32>, limit: f32, sp: ptr<function, u32>) {
    if (light.params.y < 0.5) {
        stack_set(*sp, left);
        stack_set(*sp + 1u, left + 1u);
        *sp += 2u;
        return;
    }
    let tl = ray_box_entry(eye, inv, lo_l, hi_l, limit);
    let tr = ray_box_entry(eye, inv, lo_r, hi_r, limit);
    if (tl >= 0.0 && tr >= 0.0) {
        if (tl <= tr) {
            stack_set(*sp, left + 1u);
            stack_set(*sp + 1u, left);
        } else {
            stack_set(*sp, left);
            stack_set(*sp + 1u, left + 1u);
        }
        *sp += 2u;
    } else if (tl >= 0.0) {
        stack_set(*sp, left);
        *sp += 1u;
    } else if (tr >= 0.0) {
        stack_set(*sp, left + 1u);
        *sp += 1u;
    }
}

// The nearest hit in mesh `m`, if nearer than `best`. Triangle hierarchy
// traversal uses `stack` from `base` up.
fn nearest_in_mesh(world_eye: vec3<f32>, world_dir: vec3<f32>, m: u32, best: Hit, base: u32) -> Hit {
    var hit = best;
    let range = meshes[m];
    // In the object's space with the same ray parameter: distances stay world distances.
    var eye = world_eye;
    var dir = world_dir;
    if (range.w != NONE) {
        eye = point_to_object(m, world_eye);
        dir = rotate_to_object(range.w, world_dir) * xforms[4u * range.w + 3u].y;
    }
    if (light.counts.z == 2u) {
        if (range.z == NONE) { return hit; }
        let inv = 1.0 / dir;
        var sp = base + 1u;
        stack_set(base, range.z);
        while (sp > base) {
            sp--;
            let node = blas[stack_get(sp)];
            if (!ray_box(eye, inv, node.lo.xyz, node.hi.xyz, hit.distance)) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    hit = hit_triangle(eye, dir, m, blas_items[k], hit);
                }
            } else {
                let l = blas[node.info.x];
                let r = blas[node.info.x + 1u];
                push_children(eye, inv, node.info.x, l.lo.xyz, l.hi.xyz, r.lo.xyz, r.hi.xyz, hit.distance, &sp);
            }
        }
    } else {
        for (var t = range.x; t < range.x + range.y; t++) {
            hit = hit_triangle(eye, dir, m, t, hit);
        }
    }
    if (range.w != NONE && hit.mesh == m && best.mesh != m) {
        hit.normal = normalize(rotate_to_world(range.w, hit.normal));
    }
    return hit;
}

fn camera_dir(gid: vec3<u32>) -> vec3<f32> {
    let aspect = camera.params.x / camera.params.y;
    let sx = ((f32(gid.x) + 0.5) / camera.params.x * 2.0 - 1.0) * camera.params.z * aspect;
    let sy = (1.0 - (f32(gid.y) + 0.5) / camera.params.y * 2.0) * camera.params.z;
    return normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
}

// Nearest triangle along a ray, through the mesh hierarchy or by testing every mesh.
fn primary_hit(eye: vec3<f32>, dir: vec3<f32>) -> Hit {
    var hit = Hit(3.4e38, 0xffffffffu, 0xffffffffu, vec3<f32>(0.0));
    if (light.counts.z >= 1u) {
        let inv = 1.0 / dir;
        var sp = 1u;
        stack_set(0, 0u);
        while (sp > 0u) {
            sp--;
            let node = nodes[stack_get(sp)];
            // Slab test against the node's box, closer than the best hit so far.
            if (!ray_box(eye, inv, node.lo.xyz, node.hi.xyz, hit.distance)) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    hit = nearest_in_mesh(eye, dir, items[k], hit, sp);
                }
            } else {
                let l = nodes[node.info.x];
                let r = nodes[node.info.x + 1u];
                push_children(eye, inv, node.info.x, l.lo.xyz, l.hi.xyz, r.lo.xyz, r.hi.xyz, hit.distance, &sp);
            }
        }
    } else {
        for (var m = 0u; m < light.counts.x; m++) {
            hit = nearest_in_mesh(eye, dir, m, hit, 0u);
        }
    }
    if (dot(hit.normal, dir) > 0.0) { hit.normal = -hit.normal; }
    return hit;
}

// Barycentric coordinates of world point `p` (on the triangle's plane) in
// triangle `t` of object `m`.
fn barycentric(m: u32, t: u32, world_p: vec3<f32>) -> vec3<f32> {
    let p = point_to_object(m, world_p);
    let v0 = tris[3u * t].xyz;
    let e1 = tris[3u * t + 1u].xyz - v0;
    let e2 = tris[3u * t + 2u].xyz - v0;
    let d = p - v0;
    let d11 = dot(e1, e1);
    let d12 = dot(e1, e2);
    let d22 = dot(e2, e2);
    let denom = d11 * d22 - d12 * d12;
    let s = (d22 * dot(d, e1) - d12 * dot(d, e2)) / denom;
    let u = (d11 * dot(d, e2) - d12 * dot(d, e1)) / denom;
    return vec3<f32>(1.0 - s - u, s, u);
}

// Interpolated shading normal (world space) at world point `p` on triangle `t`
// of object `m`, on the side `toward` points to.
fn shading_normal(m: u32, t: u32, p: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let b = barycentric(m, t, p);
    var n = normalize(normal_to_world(m, normals[3u * t].xyz * b.x + normals[3u * t + 1u].xyz * b.y + normals[3u * t + 2u].xyz * b.z));
    let v0 = tris[3u * t].xyz;
    let geometric = normal_to_world(m, cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0));
    // Seen from behind (back face): flip, like the geometric normal.
    if (dot(geometric, toward) < 0.0) { n = -n; }
    return n;
}

// How far to lift a point off a surface hit at `distance` from the ray's origin:
// 0.1 mm, or more far away, where f32 positions are coarser.
fn surface_offset(distance: f32) -> f32 {
    return max(1e-4, 2e-6 * distance);
}

// Distance along a camera ray to the front of a one-sided light (or 3.4e38).
fn light_distance(eye: vec3<f32>, dir: vec3<f32>) -> f32 {
    if (light.counts.w != 1u) { return 3.4e38; }
    let u = light.edge_u.xyz;
    let v = light.edge_v.xyz;
    let n = cross(u, v);
    let facing = dot(dir, n);
    if (facing >= 0.0) { return 3.4e38; }
    let t = dot(light.corner.xyz - eye, n) / facing;
    if (t <= 0.0) { return 3.4e38; }
    let q = eye + dir * t - light.corner.xyz;
    let s = dot(q, u) / dot(u, u);
    let r = dot(q, v) / dot(v, v);
    if (s < 0.0 || s > 1.0 || r < 0.0 || r > 1.0) { return 3.4e38; }
    return t;
}

// Direct irradiance at a hit through a mask cone (0 if the surface faces away).
fn hit_irradiance(pixel: u32, point: vec3<f32>, normal: vec3<f32>) -> f32 {
    return hit_irradiance_if(true, pixel, point, normal);
}

fn hit_irradiance_if(on: bool, pixel: u32, point: vec3<f32>, normal: vec3<f32>) -> f32 {
    let to_light = light.corner.xyz + (light.edge_u.xyz + light.edge_v.xyz) * 0.5 - point;
    return cone_irradiance_if(on && dot(normal, to_light) > 0.0, pixel, point, normal);
}

@group(0) @binding(24) var<storage, read> normals: array<vec4<f32>>;

// --- Camera passes. Either one pass casts the camera ray and shades, or a
// visibility pass writes each pixel's nearest hit to `vis` and a shading pass
// reads it back. Both shade with the same functions, so their output is the
// same; the split keeps the ray cast free of the cone code's register cost.

// Per pixel: object, triangle, the hit distance's bits, and 1 if the front of
// a one-sided light is nearer than the hit.
@group(0) @binding(31) var<storage, read_write> vis: array<vec4<u32>>;

// The camera ray's nearest hit, as one pass computes it.
fn camera_hit(eye: vec3<f32>, dir: vec3<f32>) -> Hit {
    cull_back = true;
    return primary_hit(eye, dir);
}

@compute @workgroup_size(8, 8)
fn visibility(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    let hit = camera_hit(eye, dir);
    let sees_light = select(0u, 1u, light_distance(eye, dir) < hit.distance);
    vis[gid.y * w + gid.x] = vec4<u32>(hit.mesh, hit.tri, bitcast<u32>(hit.distance), sees_light);
}

// The hit a visibility pass recorded, rebuilt exactly as `camera_hit` returns
// it (the normal is recomputed the same way). `w` of the result: 1 if the
// pixel sees the light.
fn recorded_hit(pixel: u32, dir: vec3<f32>) -> Hit {
    let v = vis[pixel];
    var hit = Hit(bitcast<f32>(v.z), v.x, v.y, vec3<f32>(0.0));
    if (v.x != 0xffffffffu) {
        let v0 = tris[3u * v.y].xyz;
        hit.normal = normalize(cross(tris[3u * v.y + 1u].xyz - v0, tris[3u * v.y + 2u].xyz - v0));
        let x = meshes[v.x].w;
        if (x != NONE) { hit.normal = normalize(rotate_to_world(x, hit.normal)); }
        if (dot(hit.normal, dir) > 0.0) { hit.normal = -hit.normal; }
    }
    return hit;
}

// Per-pixel shading: a mask cone at the hit.
fn shade_per_pixel(pixel: u32, eye: vec3<f32>, dir: vec3<f32>, hit: Hit, sees_light: bool) {
    if (sees_light) { colors[pixel] = vec4<f32>(vec3<f32>(camera.params.w), 1.0); return; }
    // A ray that misses everything sees the sky.
    if (hit.mesh == 0xffffffffu) { colors[pixel] = vec4<f32>(light.sky.xyz, 1.0); return; }
    let surface = eye + dir * hit.distance;
    let point = surface + hit.normal * surface_offset(hit.distance);
    let normal = shading_normal(hit.mesh, hit.tri, surface, -dir);
    let irradiance = hit_irradiance(pixel, point, normal);
    var e = vec3<f32>(camera.params.w * irradiance);
    if (any(light.sky.xyz > vec3<f32>(0.0))) {
        // params.w bit 4: faces in the hit object's frame, as the cache's
        // texels with own-occlusion masks use (and, with this bit, those
        // without).
        if ((u32(light.params.w) & 4u) != 0u) { sky_frame = meshes[hit.mesh].w; }
        e += light.sky.xyz * sky_irradiance(point, normal);
        sky_frame = NONE;
    }
    colors[pixel] = vec4<f32>(albedo[hit.mesh].xyz * e / 3.14159265, 1.0);
}

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    let hit = camera_hit(eye, dir);
    shade_per_pixel(gid.y * w + gid.x, eye, dir, hit, light_distance(eye, dir) < hit.distance);
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let pixel = gid.y * w + gid.x;
    shade_per_pixel(pixel, camera.eye.xyz, dir, recorded_hit(pixel, dir), vis[pixel].w == 1u);
}

// --- World-space lighting cache: lightmap texels on static faces. ---

// A chart: a rectangle in a surface's parameter space with an `nu × nv` texel
// grid starting at texel `info.x`, stored as 8×8 blocks of 64 consecutive
// texels, `info.w` blocks per row, and a coarser `pu × pv` grid of
// indirect-light probes (row-major).
struct Face {
    // x: first texel, y: nu, z: nv, w: blocks per row
    info: vec4<u32>,
    // x: first probe, y: pu, z: pv (0: no probes), w: mesh
    probes: vec4<u32>,
};

// Chart of each triangle, with DYNAMIC set for moving meshes (shaded per pixel;
// their charts hold probes but no texels).
@group(0) @binding(11) var<storage, read> tri_face: array<u32>;
const DYNAMIC: u32 = 0x80000000u;
@group(0) @binding(12) var<storage, read> faces: array<Face>;
@group(0) @binding(13) var<storage, read> texels: array<f32>;
// Chart coordinates of each triangle's vertices, three per triangle.
@group(0) @binding(23) var<storage, read> tri_uv: array<vec2<f32>>;

// Chart coordinates of point `p` on triangle `t`.
fn chart_uv(m: u32, t: u32, p: vec3<f32>) -> vec2<f32> {
    let b = barycentric(m, t, p);
    return tri_uv[3u * t] * b.x + tri_uv[3u * t + 1u] * b.y + tri_uv[3u * t + 2u] * b.z;
}

fn texel(face: Face, i: u32, j: u32) -> f32 {
    return texels[face.info.x + ((j / 8u) * face.info.w + i / 8u) * 64u + (j % 8u) * 8u + (i % 8u)];
}

// Bilinear lookup of cached irradiance at chart coordinates `uv` (texel centers
// at half-integer grid coordinates, clamped at the edges).
fn lookup(face: Face, uv: vec2<f32>) -> f32 {
    let nu = face.info.y;
    let nv = face.info.z;
    let s = clamp(uv.x * f32(nu) - 0.5, 0.0, f32(nu - 1u));
    let t = clamp(uv.y * f32(nv) - 0.5, 0.0, f32(nv - 1u));
    let i0 = u32(s);
    let j0 = u32(t);
    let i1 = min(i0 + 1u, nu - 1u);
    let j1 = min(j0 + 1u, nv - 1u);
    let fs = s - f32(i0);
    let ft = t - f32(j0);
    let a = mix(texel(face, i0, j0), texel(face, i1, j0), fs);
    let b = mix(texel(face, i0, j1), texel(face, i1, j1), fs);
    return mix(a, b, ft);
}

// Cached shading: lightmap and probe lookups, or a mask cone on moving meshes.
fn shade_cached(pixel: u32, eye: vec3<f32>, dir: vec3<f32>, hit: Hit, sees_light: bool) {
    if (sees_light) { colors[pixel] = vec4<f32>(vec3<f32>(camera.params.w), 1.0); return; }
    if (hit.mesh == 0xffffffffu) { colors[pixel] = vec4<f32>(0.0, 0.0, 0.0, 1.0); return; }
    let surface = eye + dir * hit.distance;
    let f = tri_face[hit.tri];
    let uv = chart_uv(hit.mesh, hit.tri, surface);
    var irradiance: vec3<f32>;
    if ((f & DYNAMIC) != 0u) {
        let point = surface + hit.normal * surface_offset(hit.distance);
        irradiance = vec3<f32>(hit_irradiance(pixel, point, shading_normal(hit.mesh, hit.tri, surface, -dir)))
            + indirect_lookup(faces[f & ~DYNAMIC], uv);
    } else {
        irradiance = lookup(faces[f], uv) + indirect_lookup(faces[f], uv);
    }
    let c = albedo[hit.mesh].xyz * irradiance * (camera.params.w / 3.14159265);
    colors[pixel] = vec4<f32>(c, 1.0);
}

@compute @workgroup_size(8, 8)
fn render_cached(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    let hit = camera_hit(eye, dir);
    shade_cached(gid.y * w + gid.x, eye, dir, hit, light_distance(eye, dir) < hit.distance);
}

@compute @workgroup_size(8, 8)
fn shade_cached_pass(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let pixel = gid.y * w + gid.x;
    shade_cached(pixel, camera.eye.xyz, dir, recorded_hit(pixel, dir), vis[pixel].w == 1u);
}

// --- Moving geometry: re-bake exactly the lightmap blocks a change can reach. ---
//
// Each 8×8 texel block has up to 16 planes (`n · p <= w` inside) of a convex
// volume enclosing every one of its texels' cones: the hull of the block and the
// light. A changed mesh's swept bounds (old ∪ new) can alter a texel only if they
// reach inside that volume, so only blocks no plane separates are re-baked.

const BLOCK_PLANES: u32 = 16u;

struct Motion {
    // x: swept boxes, y: blocks
    counts: vec4<u32>,
    // Box k is `boxes[2k] .. boxes[2k + 1]`.
    boxes: array<vec4<f32>, 32>,
};

@group(0) @binding(14) var<uniform> motion: Motion;
@group(0) @binding(15) var<storage, read> block_planes: array<vec4<f32>>;
// Indirect dispatch arguments for `rebake`; x counts the marked blocks.
@group(0) @binding(16) var<storage, read_write> rebake_args: array<atomic<u32>, 3>;
@group(0) @binding(17) var<storage, read_write> dirty: array<u32>;

@compute @workgroup_size(64)
fn mark_blocks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= motion.counts.y) { return; }
    for (var k = 0u; k < motion.counts.x; k++) {
        let lo = motion.boxes[2u * k].xyz;
        let hi = motion.boxes[2u * k + 1u].xyz;
        let center = (lo + hi) * 0.5;
        let extent = (hi - lo) * 0.5;
        var separated = false;
        for (var p = 0u; p < BLOCK_PLANES; p++) {
            let plane = block_planes[b * BLOCK_PLANES + p];
            if (dot(plane.xyz, center) - dot(abs(plane.xyz), extent) > plane.w) {
                separated = true;
                break;
            }
        }
        if (!separated) {
            dirty[atomicAdd(&rebake_args[0], 1u)] = b;
            return;
        }
    }
}

// --- Static coverage cache: each texel of a block that can receive direct
// light keeps the per-bin coverage of static meshes in its cone (built once,
// through a hierarchy of static meshes), so a re-bake only adds the moving
// meshes. Coverage combines by max, which does not depend on order, so this
// equals tracing the whole scene.

// Per texel slot: 64 bins of static coverage.
@group(0) @binding(27) var<storage, read_write> static_cov: array<f32>;
// Per block: its slot (in blocks of 64 texels), or NONE if it never receives
// direct light.
@group(0) @binding(28) var<storage, read> block_slot: array<u32>;

// One workgroup per block, one invocation per texel; the bound hierarchy holds
// the static meshes only.
@compute @workgroup_size(64)
fn bake_static(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let slot = block_slot[wg.x];
    if (slot == NONE) { return; }
    let i = wg.x * 64u + li;
    let r = receivers[i];
    var cov: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
    let to_light = light.corner.xyz + (light.edge_u.xyz + light.edge_v.xyz) * 0.5 - r.point.xyz;
    if (dot(r.normal.xyz, to_light) > 0.0) {
        let lc = light_cone(r.point.xyz);
        if (lc.ok) {
            var full = 0u;
            scene_coverage(lc.c, i, &cov, &full);
        }
    }
    let base = (slot * 64u + li) * BINS;
    for (var b = 0u; b < BINS; b++) { static_cov[base + b] = cov[b]; }
}

// Direct irradiance at texel `li` of a block from its cached static coverage
// plus the moving meshes: equal to `hit_irradiance` through the whole scene.
fn cached_irradiance(i: u32, apex: vec3<f32>, normal: vec3<f32>, slot: u32, li: u32) -> f32 {
    if (slot == NONE) { return 0.0; }
    let to_light = light.corner.xyz + (light.edge_u.xyz + light.edge_v.xyz) * 0.5 - apex;
    if (dot(normal, to_light) <= 0.0) { return 0.0; }
    let lc = light_cone(apex);
    if (!lc.ok) { return 0.0; }
    var cov: array<f32, 64>;
    var full = 0u;
    let base = (slot * 64u + li) * BINS;
    for (var b = 0u; b < BINS; b++) {
        cov[b] = static_cov[base + b];
        if (cov[b] >= 1.0) { full++; }
    }
    for (var k = 0u; k < probe_params.counts.w; k++) {
        if (full == BINS) { break; }
        if (!may_touch_box(lc.c, probe_params.bounds[2u * k].xyz, probe_params.bounds[2u * k + 1u].xyz)) { continue; }
        add_mesh_coverage(lc.c, probe_params.moving[k].x, &cov, &full, i, 0u);
    }
    return cone_visible(lc.c, normal, &cov);
}

// One workgroup per marked block, one invocation per texel; `result` is the
// lightmap's texel buffer and `receivers` its texels.
@compute @workgroup_size(64)
fn rebake(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let block = dirty[wg.x];
    let i = block * 64u + li;
    result[i] = cached_irradiance(i, receivers[i].point.xyz, receivers[i].normal.xyz, block_slot[block], li);
}

// --- Indirect light: probes with cone G-buffers. ---
//
// Probes sit on a coarse grid on every static face. Each caches a hemicube
// G-buffer: for each bin (R×R on the top face, R×R/2 on each side face), the
// nearest static triangle along the bin's center direction and its distance.
// Static geometry does not change, so the G-buffer is built once. Each frame a
// probe gathers bounce radiance: per bin, the hit surface's albedo times its
// cached direct irradiance (the lightmap, which tracks moving shadows exactly),
// with moving meshes composited on top by depth as black occluders.

const NO_HIT: u32 = 0xffffffffu;
const BLACK: u32 = 0xfffffffeu;
const BACK: u32 = 0x80000000u;

struct ProbeParams {
    // x: probes; y: R, bins per hemicube face side; z: mode (0: cached
    // G-buffer with moving meshes composited; 1: trace every bin through the
    // bound hierarchy, moving meshes black); w: moving meshes
    counts: vec4<u32>,
    // x: probes on static charts (the G-buffer's); those after it sit on moving
    // meshes and are traced every gather
    extra: vec4<u32>,
    // Moving mesh k is `moving[k].x`, with bounds `bounds[2k] .. bounds[2k + 1]`.
    moving: array<vec4<u32>, 16>,
    bounds: array<vec4<f32>, 32>,
};

@group(0) @binding(18) var<storage, read> probe_points: array<Receiver>;
// Per probe and bin: triangle (BACK set if its back faces the probe), NO_HIT
// or BLACK; and the distance's bits.
@group(0) @binding(19) var<storage, read_write> gbuffer: array<vec2<u32>>;
@group(0) @binding(20) var<storage, read> bin_weights: array<f32>;
// Per probe: indirect irradiance per unit light radiance (RGB), and 1 if valid.
@group(0) @binding(21) var<storage, read_write> probe_out: array<vec4<f32>>;
@group(0) @binding(22) var<uniform> probe_params: ProbeParams;

fn bins_per_probe() -> u32 {
    let r = probe_params.counts.y;
    return 3u * r * r;
}

// A bin's center direction in the probe's frame (z along the normal), not normalized.
fn bin_local(bin: u32, r: u32) -> vec3<f32> {
    let g = f32(r);
    if (bin < r * r) {
        return vec3<f32>(-1.0 + (2.0 * f32(bin % r) + 1.0) / g, -1.0 + (2.0 * f32(bin / r) + 1.0) / g, 1.0);
    }
    let half = r * r / 2u;
    let side = (bin - r * r) / half;
    let k = (bin - r * r) % half;
    let a = -1.0 + (2.0 * f32(k % r) + 1.0) / g;
    let z = (2.0 * f32(k / r) + 1.0) / g;
    switch side {
        case 0u: { return vec3<f32>(1.0, a, z); }
        case 1u: { return vec3<f32>(-1.0, a, z); }
        case 2u: { return vec3<f32>(a, 1.0, z); }
        default: { return vec3<f32>(a, -1.0, z); }
    }
}

fn bin_world(n: vec3<f32>, local: vec3<f32>) -> vec3<f32> {
    var a = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(n.z) > 0.9) { a = vec3<f32>(1.0, 0.0, 0.0); }
    let t = normalize(cross(n, a));
    let b = cross(n, t);
    return t * local.x + b * local.y + n * local.z;
}

// The nearest surface along a probe ray through the bound hierarchy, encoded as
// a G-buffer entry. Hits on moving meshes are BLACK.
fn trace_bin(origin: vec3<f32>, dir: vec3<f32>) -> vec2<u32> {
    let hit = primary_hit(origin, dir);
    if (hit.mesh == 0xffffffffu) { return vec2<u32>(NO_HIT, 0u); }
    if ((tri_face[hit.tri] & DYNAMIC) != 0u) { return vec2<u32>(BLACK, bitcast<u32>(hit.distance)); }
    let v0 = tris[3u * hit.tri].xyz;
    let v1 = tris[3u * hit.tri + 1u].xyz;
    let v2 = tris[3u * hit.tri + 2u].xyz;
    var code = hit.tri;
    if (dot(cross(v1 - v0, v2 - v0), dir) > 0.0) { code |= BACK; }
    return vec2<u32>(code, bitcast<u32>(hit.distance));
}

// A cached static entry with the moving meshes composited by depth: BLACK if
// one is nearer (same tie rule as tracing the whole scene).
fn composite(origin: vec3<f32>, dir: vec3<f32>, g: vec2<u32>) -> vec2<u32> {
    var best = Hit(3.4e38, 0xffffffffu, 0xffffffffu, vec3<f32>(0.0));
    if (g.x != NO_HIT) {
        best.distance = bitcast<f32>(g.y);
        best.mesh = faces[tri_face[g.x & ~BACK]].probes.w;
    }
    let inv = 1.0 / dir;
    var blocked = false;
    for (var k = 0u; k < probe_params.counts.w; k++) {
        let lo = probe_params.bounds[2u * k].xyz;
        let hi = probe_params.bounds[2u * k + 1u].xyz;
        if (!ray_box(origin, inv, lo, hi, best.distance)) { continue; }
        let m = probe_params.moving[k].x;
        let hit = nearest_in_mesh(origin, dir, m, best, 0u);
        if (hit.mesh == m) {
            best = hit;
            blocked = true;
        }
    }
    if (blocked) { return vec2<u32>(BLACK, bitcast<u32>(best.distance)); }
    return g;
}

// Bounce radiance per unit light radiance (RGB) from a G-buffer entry; w is 1
// for a back face (its radiance is unknown and taken as zero).
fn bin_radiance(origin: vec3<f32>, dir: vec3<f32>, g: vec2<u32>) -> vec4<f32> {
    if (g.x == NO_HIT || g.x == BLACK) { return vec4<f32>(0.0); }
    if ((g.x & BACK) != 0u) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let face = faces[tri_face[g.x]];
    let e = lookup(face, chart_uv(face.probes.w, g.x, origin + dir * bitcast<f32>(g.y)));
    return vec4<f32>(albedo[face.probes.w].xyz * (e / 3.14159265), 0.0);
}

// One invocation per (bin, probe): dispatched as (bins / 64, probes).
@compute @workgroup_size(64)
fn probe_build(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let bins = bins_per_probe();
    let p = gid.y;
    if (gid.x >= bins || p >= probe_params.counts.x) { return; }
    let r = probe_points[p];
    let dir = bin_world(r.normal.xyz, bin_local(gid.x, probe_params.counts.y));
    gbuffer[p * bins + gid.x] = trace_bin(r.point.xyz, dir);
}

var<workgroup> partial: array<vec4<f32>, 64>;

// One workgroup per probe: each invocation sums every 64th bin, then a fixed
// tree reduction, so the sum order never depends on scheduling.
@compute @workgroup_size(64)
fn probe_gather(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let p = wg.x;
    let bins = bins_per_probe();
    var sum = vec4<f32>(0.0);
    if (p < probe_params.counts.x) {
        let r = probe_points[p];
        let origin = r.point.xyz;
        for (var b = li; b < bins; b += 64u) {
            let dir = bin_world(r.normal.xyz, bin_local(b, probe_params.counts.y));
            var g: vec2<u32>;
            if (probe_params.counts.z == 0u && p < probe_params.extra.x) {
                g = composite(origin, dir, gbuffer[p * bins + b]);
            } else {
                g = trace_bin(origin, dir);
            }
            sum += bin_weights[b] * bin_radiance(origin, dir, g);
        }
    }
    partial[li] = sum;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (li < s) { partial[li] += partial[li + s]; }
        workgroupBarrier();
    }
    if (li == 0u && p < probe_params.counts.x) {
        let total = partial[0];
        // A probe that sees mostly back faces sits inside geometry; lookups skip it.
        var valid = 0.0;
        if (total.w < 0.1 * 3.14159265) { valid = 1.0; }
        probe_out[p] = vec4<f32>(total.xyz, valid);
    }
}

fn probe_at(face: Face, i: u32, j: u32) -> vec4<f32> {
    return probe_out[face.probes.x + j * face.probes.y + i];
}

// Bilinear lookup of probe irradiance at chart coordinates `uv`, skipping
// invalid probes.
fn indirect_lookup(face: Face, uv: vec2<f32>) -> vec3<f32> {
    let pu = face.probes.y;
    let pv = face.probes.z;
    if (pu == 0u) { return vec3<f32>(0.0); }
    let s = clamp(uv.x * f32(pu) - 0.5, 0.0, f32(pu - 1u));
    let t = clamp(uv.y * f32(pv) - 0.5, 0.0, f32(pv - 1u));
    let i0 = u32(s);
    let j0 = u32(t);
    let i1 = min(i0 + 1u, pu - 1u);
    let j1 = min(j0 + 1u, pv - 1u);
    let fs = s - f32(i0);
    let ft = t - f32(j0);
    let c00 = probe_at(face, i0, j0);
    let c10 = probe_at(face, i1, j0);
    let c01 = probe_at(face, i0, j1);
    let c11 = probe_at(face, i1, j1);
    let w00 = (1.0 - fs) * (1.0 - ft) * c00.w;
    let w10 = fs * (1.0 - ft) * c10.w;
    let w01 = (1.0 - fs) * ft * c01.w;
    let w11 = fs * ft * c11.w;
    let total = w00 + w10 + w01 + w11;
    if (total <= 1e-6) { return vec3<f32>(0.0); }
    return (c00.xyz * w00 + c10.xyz * w10 + c01.xyz * w01 + c11.xyz * w11) / total;
}

// --- View-dependent lighting cache, for instanced scenes. ---
//
// Pages of 8×8 texels, stored 9×9 (a one-texel border on the high side, so a
// bilinear lookup stays inside one page), on each object's charts at
// power-of-two resolutions: texels at mip m are the base size × 2^m × the
// object's scale. A page table maps (object, chart | mip << 16, page x | page y
// << 16) to a slot of the pool. The shading pass reports pages it lacked, a
// CPU page manager allocates them and lists their texels, and the bake passes
// compute those texels with mask cones. A texel is placed on the surface
// through its chart's cell grid, shared by every object with that geometry.

struct VChart {
    // Object-space size along u and v.
    size: vec2<f32>,
    grid: vec2<u32>,
    // First cell in `cells`.
    first_cell: u32,
    pad: u32,
};

struct VcParams {
    // x: page table mask, y: frame, z: request tile size (pixels), w: request tiles per row
    counts: vec4<u32>,
    // x: texels to bake, y: pages baked, z: page table updates, w: request
    // set mask
    jobs: vec4<u32>,
    // x: 1 / base texel size, y: radians per pixel (camera), z: pixels per texel, w: 0
    params: vec4<f32>,
    // x: sky samples to bake (each as six face jobs), y: 1 with a sky, z, w: 0
    sky: vec4<u32>,
};

const PAGE: u32 = 8u;
const PAGE_STRIDE: u32 = 9u;
const PAGE_TEXELS: u32 = 81u;
const EMPTY_KEY: u32 = 0xffffffffu;

@group(0) @binding(33) var<storage, read> vcharts: array<VChart>;
// Per chart cell: up to two triangles (NONE if absent).
@group(0) @binding(34) var<storage, read> cells: array<vec2<u32>>;
// Chart of each geometry triangle.
@group(0) @binding(35) var<storage, read> tri_vchart: array<u32>;
// Open addressing with linear probing: three key words, then the slot. x is
// EMPTY_KEY for a free entry (deletion moves entries back: no tombstones).
@group(0) @binding(36) var<storage, read_write> page_table: array<vec4<u32>>;
@group(0) @binding(37) var<storage, read_write> pool: array<f32>;
// Pages baked this frame: key words, then the slot.
@group(0) @binding(38) var<storage, read> bake_list: array<vec4<u32>>;
// Pages requested this frame, each once: key words, then the priority (1 for
// a pixel with no page at all; then 2 + 30 − the gap between the found and
// the wanted mip, so the blurriest pixels come first). Count in vc_stats[1].
@group(0) @binding(39) var<storage, read_write> requests: array<vec4<u32>>;
// Per slot: the last frame a pixel read it (or it was baked).
@group(0) @binding(40) var<storage, read_write> slot_used: array<u32>;
@group(0) @binding(41) var<uniform> vc: VcParams;
// x: pixels shaded without any resident page (unshadowed fallback), y:
// requests listed, z: sample pixels on a surface, w: of those, the ones that
// found their wanted mip. 4–15: the bake's work counters (`work_a`,
// `work_b`, `work_c`), with COUNT_WORK.
@group(0) @binding(42) var<storage, read_write> vc_stats: array<atomic<u32>, 16>;
// Texels to bake: entry in `bake_list`, and the texel's index in the 9×9 page.
@group(0) @binding(43) var<storage, read> bake_jobs: array<vec2<u32>>;
// Page table writes: (entry index, 0, 0, 0), then the entry.
@group(0) @binding(44) var<storage, read> table_updates: array<vec4<u32>>;
// This frame's requested pages, as fingerprints (open addressing; 0: free),
// so each page is listed once.
@group(0) @binding(45) var<storage, read_write> request_set: array<atomic<u32>>;
// Sky samples to bake (M5 gate 5): entry in `bake_list`, and the texel's index
// in the 9×9 page. Sky light varies slowly, so it is sampled every 1, 2, 4
// or 8 texels (the page's stride) and interpolated; each sample is six jobs,
// one per cube face, whose results land in `sky_partial` (six per sample) for
// `vc_fill` to sum.
@group(0) @binding(46) var<storage, read> sky_samples: array<vec2<u32>>;
@group(0) @binding(47) var<storage, read_write> sky_partial: array<f32>;
// Per baked page: its first sky sample, and how many it has (low 24 bits)
// with its sky stride (high 8 bits).
@group(0) @binding(48) var<storage, read> page_sky: array<vec2<u32>>;
// Own-object sky occlusion (M5 gate 5). An instance's coverage of its own sky,
// seen through cube faces in its frame, is the same for every instance of its
// geometry; it is baked once per geometry on a grid over each chart, as 64 bins
// of 8-bit coverage per face (16 words), and read bilinearly.
// Per global chart: first self texel, grid size along u and v, and 1 if the
// chart's geometry has masks (0: its texels walk their own object).
@group(0) @binding(49) var<storage, read> self_charts: array<vec4<u32>>;
// Self texels to bake: an object of the geometry, its chart, and the texel
// (x | y << 16).
@group(0) @binding(50) var<storage, read> self_list: array<vec4<u32>>;
@group(0) @binding(51) var<storage, read_write> self_masks: array<u32>;

fn vc_hash(k: vec3<u32>) -> u32 {
    var h = (k.x * 0x9E3779B1u) ^ (k.y * 0x85EBCA77u) ^ (k.z * 0xC2B2AE3Du);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    return h;
}

fn vc_find(k: vec3<u32>) -> u32 {
    let mask = vc.counts.x;
    var i = vc_hash(k) & mask;
    for (var probe = 0u; probe < 64u; probe++) {
        let e = page_table[i];
        if (e.x == EMPTY_KEY) { return NONE; }
        if (e.x == k.x && e.y == k.y && e.z == k.z) { return e.w; }
        i = (i + 1u) & mask;
    }
    return NONE;
}

fn object_scale(m: u32) -> f32 {
    let x = meshes[m].w;
    if (x == NONE) { return 1.0; }
    return xforms[4u * x + 3u].x;
}

// Texels along u and v of chart `c` on object `m` at `mip`. Products only (no
// division), so the CPU page manager computes the same grid.
fn vc_grid(c: u32, m: u32, mip: u32) -> vec2<u32> {
    let size = vcharts[c].size * object_scale(m);
    return max(vec2<u32>(ceil(size * ldexp(vc.params.x, -i32(mip)))), vec2<u32>(1u));
}

// The coarsest mip of chart `c` on object `m`: the first with one texel.
fn vc_top_mip(c: u32, m: u32) -> u32 {
    let ratio = max(vcharts[c].size.x, vcharts[c].size.y) * object_scale(m) * vc.params.x;
    var top = 0u;
    if (ratio > 1.0) { top = min(u32(ceil(log2(ratio))), 30u); }
    // log2 is approximate: settle on the exact first mip.
    if (top > 0u && all(vc_grid(c, m, top - 1u) == vec2<u32>(1u))) { top -= 1u; }
    if (any(vc_grid(c, m, top) != vec2<u32>(1u))) { top += 1u; }
    return top;
}

// The mip whose texels span about `vc.params.z` pixels at `distance`.
fn vc_mip(distance: f32) -> u32 {
    let ratio = distance * vc.params.y * vc.params.z * vc.params.x;
    if (ratio <= 1.0) { return 0u; }
    return min(u32(round(log2(ratio))), 30u);
}

// The page holding the texels around chart coordinates `uv`, with the
// bilinear lookup's texel (within the page) and fraction.
struct PageRef {
    key: vec3<u32>,
    base: vec2<u32>,
    frac: vec2<f32>,
    coarsest: bool,
};

fn vc_page(m: u32, c: u32, uv: vec2<f32>, mip: u32) -> PageRef {
    let n = vc_grid(c, m, mip);
    let t = clamp(uv * vec2<f32>(n) - vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(n - vec2<u32>(1u)));
    let base = vec2<u32>(floor(t));
    let page = base / PAGE;
    var r: PageRef;
    r.key = vec3<u32>(m, c | (mip << 16u), page.x | (page.y << 16u));
    r.base = base - page * PAGE;
    r.frac = t - floor(t);
    r.coarsest = n.x == 1u && n.y == 1u;
    return r;
}

// A texel's two channels: irradiance per unit sun radiance, and per unit sky
// radiance (M5).
fn vc_texel_value(i: u32) -> vec2<f32> {
    return vec2<f32>(pool[2u * i], pool[2u * i + 1u]);
}

fn vc_bilinear(slot: u32, r: PageRef) -> vec2<f32> {
    let o = slot * PAGE_TEXELS + r.base.y * PAGE_STRIDE + r.base.x;
    let a = mix(vc_texel_value(o), vc_texel_value(o + 1u), r.frac.x);
    let b = mix(vc_texel_value(o + PAGE_STRIDE), vc_texel_value(o + PAGE_STRIDE + 1u), r.frac.x);
    return mix(a, b, r.frac.y);
}

// Barycentric coordinates of chart coordinates `uv` in geometry triangle `t`.
fn uv_barycentric(t: u32, uv: vec2<f32>) -> vec3<f32> {
    let a = tri_uv[3u * t];
    let e1 = tri_uv[3u * t + 1u] - a;
    let e2 = tri_uv[3u * t + 2u] - a;
    let d = uv - a;
    let det = e1.x * e2.y - e1.y * e2.x;
    if (abs(det) < 1e-30) { return vec3<f32>(-1.0); }
    let s = (d.x * e2.y - d.y * e2.x) / det;
    let u = (e1.x * d.y - e1.y * d.x) / det;
    return vec3<f32>(1.0 - s - u, s, u);
}

// Where texel `texel` of chart `c` on object `m` (grid `n`) lies: its center
// placed on the surface through the chart's cells (the nearest point of the
// cell's most nearly containing triangle), as a cone apex and a shading
// normal. `ok` is false without a triangle.
struct TexelSurface {
    apex: vec3<f32>,
    normal: vec3<f32>,
    ok: bool,
};

fn vc_surface(m: u32, c: u32, n: vec2<u32>, texel: vec2<u32>) -> TexelSurface {
    let ch = vcharts[c];
    let uv = (vec2<f32>(texel) + vec2<f32>(0.5)) / vec2<f32>(n);
    let cell = min(vec2<u32>(uv * vec2<f32>(ch.grid)), ch.grid - vec2<u32>(1u));
    let pair = cells[ch.first_cell + cell.y * ch.grid.x + cell.x];
    var best_t = NONE;
    var best_b = vec3<f32>(0.0);
    var best_score = -3.4e38;
    for (var q = 0u; q < 2u; q++) {
        let t = select(pair.x, pair.y, q == 1u);
        if (t == NONE) { continue; }
        let b = uv_barycentric(t, uv);
        let score = min(b.x, min(b.y, b.z));
        if (score > best_score) {
            best_score = score;
            best_t = t;
            best_b = b;
        }
    }
    let b0 = clamp(best_b, vec3<f32>(0.0), vec3<f32>(1.0));
    let b = b0 / max(b0.x + b0.y + b0.z, 1e-20);
    let t = select(0u, best_t, best_t != NONE);
    let p = tris[3u * t].xyz * b.x + tris[3u * t + 1u].xyz * b.y + tris[3u * t + 2u].xyz * b.z;
    let nrm = normals[3u * t].xyz * b.x + normals[3u * t + 1u].xyz * b.y + normals[3u * t + 2u].xyz * b.z;
    var pw = p;
    var nw = normalize(nrm);
    let x = meshes[m].w;
    if (x != NONE) {
        pw = rotate_to_world(x, p) * xforms[4u * x + 3u].x + translation(x);
        nw = normalize(rotate_to_world(x, nrm));
    }
    let geometric = normal_to_world(m, cross(tris[3u * t + 1u].xyz - tris[3u * t].xyz, tris[3u * t + 2u].xyz - tris[3u * t].xyz));
    let apex = pw + normalize(geometric) * surface_offset(length(pw));
    return TexelSurface(apex, nw, best_t != NONE);
}

// A baked page's texel: the job's page, and the texel's place in the chart.
struct PageTexel {
    e: vec4<u32>,
    m: u32,
    c: u32,
    n: vec2<u32>,
    texel: vec2<u32>,
    inside: bool,
};

fn page_texel(page_index: u32, k: u32) -> PageTexel {
    var t: PageTexel;
    t.e = bake_list[page_index];
    t.m = t.e.x;
    t.c = t.e.y & 0xffffu;
    let mip = t.e.y >> 16u;
    let page = vec2<u32>(t.e.z & 0xffffu, t.e.z >> 16u);
    t.n = vc_grid(t.c, t.m, mip);
    t.texel = page * PAGE + vec2<u32>(k % PAGE_STRIDE, k / PAGE_STRIDE);
    t.inside = t.texel.x < t.n.x && t.texel.y < t.n.y;
    return t;
}

@compute @workgroup_size(64)
fn vc_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= vc.jobs.z) { return; }
    page_table[table_updates[2u * i].x] = table_updates[2u * i + 1u];
}

// One thread per listed texel. The list may include a texel just outside the
// chart's grid (the CPU's grid can differ by rounding); it is skipped here and
// filled by `vc_fill`.
@compute @workgroup_size(64)
fn vc_bake(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let j = gid.x;
    // No early return: every invocation follows its packet's walks.
    let listed = j < vc.jobs.x;
    let job = bake_jobs[min(j, vc.jobs.x - 1u)];
    let t = page_texel(job.x, job.y);
    let s = vc_surface(t.m, t.c, t.n, t.texel);
    let on = listed && t.inside && s.ok;
    let sun = hit_irradiance_if(on, 0u, s.apex, s.normal);
    if (listed && t.inside) {
        pool[2u * (t.e.w * PAGE_TEXELS + job.y)] = select(0.0, sun, on);
    }
    if (COUNT_WORK) {
        for (var k = 0u; k < 4u; k++) {
            if (work_a[k] > 0u) { atomicAdd(&vc_stats[4u + k], work_a[k]); }
            if (work_b[k] > 0u) { atomicAdd(&vc_stats[8u + k], work_b[k]); }
            if (work_c[k] > 0u) { atomicAdd(&vc_stats[12u + k], work_c[k]); }
        }
    }
}

// Bin `b`'s 8-bit coverage in face `f` of self texel `i`.
fn self_bin(i: u32, f: u32, b: u32) -> f32 {
    let word = self_masks[16u * (6u * i + f) + b / 4u];
    return f32((word >> (8u * (b % 4u))) & 0xffu) / 255.0;
}

// Combines the own object's coverage of face `f` at chart coordinates `uv`
// (bilinear between the chart's self texels) into `cov`, as one more
// independent occluder.
fn self_combine(chart: vec4<u32>, uv: vec2<f32>, f: u32, cov: ptr<function, array<f32, 64>>) {
    let n = chart.yz;
    let p = clamp(uv * vec2<f32>(n) - vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(n - vec2<u32>(1u)));
    let a = vec2<u32>(floor(p));
    let b = min(a + vec2<u32>(1u), n - vec2<u32>(1u));
    let w = p - floor(p);
    let i00 = chart.x + a.y * n.x + a.x;
    let i10 = chart.x + a.y * n.x + b.x;
    let i01 = chart.x + b.y * n.x + a.x;
    let i11 = chart.x + b.y * n.x + b.x;
    for (var k = 0u; k < BINS; k++) {
        let own = mix(mix(self_bin(i00, f, k), self_bin(i10, f, k), w.x),
                      mix(self_bin(i01, f, k), self_bin(i11, f, k), w.x), w.y);
        if (own > 0.0) {
            (*cov)[k] = 1.0 - (1.0 - (*cov)[k]) * (1.0 - own);
        }
    }
}

// One thread per self texel and cube face, face by face: the object's
// coverage of its own sky through the face in its frame (every face, above
// the horizon or not, as a receiver nearby may see it).
@compute @workgroup_size(64)
fn vc_self_bake(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let count = arrayLength(&self_list);
    // Dispatched in rows of 65,535 workgroups.
    let j = gid.x + gid.y * 65535u * 64u;
    let listed = j < 6u * count;
    let f = min(j / count, 5u);
    let k = j % count;
    let e = self_list[k];
    let m = e.x;
    let chart = self_charts[e.y];
    let texel = vec2<u32>(e.z & 0xffffu, e.z >> 16u);
    let s = vc_surface(m, e.y, chart.yz, texel);
    let on = listed && s.ok;
    sky_frame = meshes[m].w;
    var face = sky_face(on, f, s.apex, s.normal);
    face.above = on;
    var cov: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
    var full = 0u;
    add_mesh_coverage_if(on, face.c, m, &cov, &full, 0u, 0u);
    if (listed) {
        for (var w = 0u; w < 16u; w++) {
            var word = 0u;
            for (var q = 0u; q < 4u; q++) {
                let v = select(0.0, clamp(cov[4u * w + q], 0.0, 1.0), on);
                word |= u32(round(v * 255.0)) << (8u * q);
            }
            self_masks[16u * (6u * k + f) + w] = word;
        }
    }
}

// Does nothing: a pass after the camera passes whose beginning timestamps
// their end (an end-of-pass timestamp followed only by copies can land too
// late on Metal).
@compute @workgroup_size(1)
fn mark() {}

// One thread per sky sample and cube face, face by face, so neighbouring
// invocations trace the same face from nearby texels.
@compute @workgroup_size(64)
fn vc_bake_sky(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let count = vc.sky.x;
    let j = gid.x;
    // No early return: every invocation follows its packet's walks.
    let listed = j < 6u * count;
    let f = min(j / max(count, 1u), 5u);
    let k = j % max(count, 1u);
    let sample = sky_samples[k];
    let t = page_texel(sample.x, sample.y);
    let s = vc_surface(t.m, t.c, t.n, t.texel);
    let on = listed && t.inside && s.ok;
    own_object = t.m;
    var value = 0.0;
    if (u32(light.params.z) != 1u) {
        // With its geometry's masks, the face turns with the object and its
        // own coverage comes from them.
        let chart = self_charts[t.c];
        let masked = chart.w == 1u;
        if (masked) {
            sky_frame = meshes[t.m].w;
            skip_object = t.m;
        } else if ((u32(light.params.w) & 4u) != 0u) {
            sky_frame = meshes[t.m].w;
        }
        let face = sky_face(on, f, s.apex, s.normal);
        var cov: array<f32, 64>;
        sky_face_coverage(face, &cov);
        if (masked && face.above) {
            let uv = (vec2<f32>(t.texel) + vec2<f32>(0.5)) / vec2<f32>(t.n);
            self_combine(chart, uv, f, &cov);
        }
        value = sky_face_light(face, s.normal, &cov);
        sky_frame = NONE;
        skip_object = NONE;
    }
    if (listed) { sky_partial[6u * k + f] = select(0.0, value, on); }
    if (COUNT_WORK) {
        for (var q = 0u; q < 4u; q++) {
            if (work_a[q] > 0u) { atomicAdd(&vc_stats[4u + q], work_a[q]); }
            if (work_b[q] > 0u) { atomicAdd(&vc_stats[8u + q], work_b[q]); }
            if (work_c[q] > 0u) { atomicAdd(&vc_stats[12u + q], work_c[q]); }
        }
    }
}

// The sky channel of texel `x` of a page along one axis: the samples on
// either side (multiples of the stride inside the grid), and the weight of
// the second; a texel past the last sample takes the last sample's value.
fn sky_span(x: u32, stride: u32, inside: u32) -> vec3<f32> {
    let x0 = (x / stride) * stride;
    let x1 = x0 + stride;
    if (x1 >= inside || x1 >= PAGE_STRIDE) { return vec3<f32>(f32(x0), f32(x0), 0.0); }
    return vec3<f32>(f32(x0), f32(x1), f32(x - x0) / f32(stride));
}

// One thread per baked page: sums each sky sample's faces into the sky
// channel and interpolates it between samples (bilinearly), then fills the
// border beyond the chart's grid with the nearest texel inside it (clamped
// lookups), and marks the page used.
@compute @workgroup_size(64)
fn vc_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= vc.jobs.y) { return; }
    let e = bake_list[p];
    let c = e.y & 0xffffu;
    let mip = e.y >> 16u;
    let page = vec2<u32>(e.z & 0xffffu, e.z >> 16u);
    let n = vc_grid(c, e.x, mip);
    let o = e.w * PAGE_TEXELS;
    let in_grid = min(n - min(page * PAGE, n), vec2<u32>(PAGE_STRIDE));
    let range = page_sky[p];
    let stride = range.y >> 24u;
    if (vc.sky.y == 0u) {
        for (var k = 0u; k < PAGE_TEXELS; k++) { pool[2u * (o + k) + 1u] = 0.0; }
    } else {
        for (var q = range.x; q < range.x + (range.y & 0xffffffu); q++) {
            let k = sky_samples[q].y;
            if (k % PAGE_STRIDE < in_grid.x && k / PAGE_STRIDE < in_grid.y) {
                var total = 0.0;
                for (var f = 0u; f < 6u; f++) { total += sky_partial[6u * q + f]; }
                pool[2u * (o + k) + 1u] = total;
            }
        }
        if (stride > 1u) {
            for (var k = 0u; k < PAGE_TEXELS; k++) {
                let x = k % PAGE_STRIDE;
                let y = k / PAGE_STRIDE;
                if (x >= in_grid.x || y >= in_grid.y || (x % stride == 0u && y % stride == 0u)) { continue; }
                let sx = sky_span(x, stride, in_grid.x);
                let sy = sky_span(y, stride, in_grid.y);
                let at = vec4<u32>(vec2<u32>(sx.xy), vec2<u32>(sy.xy));
                let a = mix(pool[2u * (o + at.z * PAGE_STRIDE + at.x) + 1u], pool[2u * (o + at.z * PAGE_STRIDE + at.y) + 1u], sx.z);
                let b = mix(pool[2u * (o + at.w * PAGE_STRIDE + at.x) + 1u], pool[2u * (o + at.w * PAGE_STRIDE + at.y) + 1u], sx.z);
                pool[2u * (o + k) + 1u] = mix(a, b, sy.z);
            }
        }
    }
    for (var k = 0u; k < PAGE_TEXELS; k++) {
        let texel = page * PAGE + vec2<u32>(k % PAGE_STRIDE, k / PAGE_STRIDE);
        if (texel.x >= n.x || texel.y >= n.y) {
            let inside = min(texel, n - vec2<u32>(1u)) - page * PAGE;
            let nearest = o + inside.y * PAGE_STRIDE + inside.x;
            pool[2u * (o + k)] = pool[2u * nearest];
            pool[2u * (o + k) + 1u] = pool[2u * nearest + 1u];
        }
    }
    slot_used[e.w] = vc.counts.y;
}

var<workgroup> fallback_count: atomic<u32>;

// Lists page `k` with `priority` unless it is already listed this frame. Two
// keys with the same 32-bit fingerprint count as one; the other is requested
// again in a later frame.
fn vc_request(k: vec3<u32>, priority: u32) {
    let fp = vc_hash(k) | 1u;
    let mask = vc.jobs.w;
    var i = vc_hash(vec3<u32>(k.z, k.x, k.y)) & mask;
    for (var probe = 0u; probe < 32u; probe++) {
        let r = atomicCompareExchangeWeak(&request_set[i], 0u, fp);
        if (r.exchanged) {
            let n = atomicAdd(&vc_stats[1], 1u);
            if (n < arrayLength(&requests)) { requests[n] = vec4<u32>(k, priority); }
            return;
        }
        if (r.old_value == fp) { return; }
        // A spurious failure of the weak exchange: try the same entry again.
        if (r.old_value == 0u) { continue; }
        i = (i + 1u) & mask;
    }
}

// Shading from the cache, reading the visibility buffer: the page of the
// wanted mip, or failing that the finest coarser one resident, or failing
// that unshadowed light. One pixel per request tile (a different one each
// frame) requests the page one mip finer than the one it found (or the
// coarsest, if none), so pages arrive coarse to fine and the CPU never walks.
@compute @workgroup_size(8, 8)
fn vc_shade(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let in_bounds = gid.x < w && gid.y < h;
    let tile = vc.counts.z;
    let sample = (vc.counts.y * 7u) % (tile * tile);
    let is_sample = in_bounds && gid.x % tile == sample % tile && gid.y % tile == sample / tile;
    var request_key = vec3<u32>(0u);
    var request_priority = 0u;
    var fell_back = false;
    if (in_bounds) {
        let dir = camera_dir(gid);
        let pixel = gid.y * w + gid.x;
        let hit = recorded_hit(pixel, dir);
        // A ray that misses everything sees the sky.
        var color = light.sky.xyz;
        if (vis[pixel].w == 1u) {
            color = vec3<f32>(camera.params.w);
        } else if (hit.mesh != NONE) {
            let surface = camera.eye.xyz + dir * hit.distance;
            let c = tri_vchart[hit.tri];
            let uv = chart_uv(hit.mesh, hit.tri, surface);
            let wanted = min(vc_mip(hit.distance), vc_top_mip(c, hit.mesh));
            // Sun and sky irradiance per unit radiance.
            var irradiance = vec2<f32>(0.0);
            var found = false;
            var mip = wanted;
            // The key of the page one mip finer than `mip` (the last missing one).
            var finer = vec3<u32>(0u);
            loop {
                let r = vc_page(hit.mesh, c, uv, mip);
                let slot = vc_find(r.key);
                if (slot != NONE) {
                    slot_used[slot] = vc.counts.y;
                    irradiance = vc_bilinear(slot, r);
                    found = true;
                    if (mip > wanted) {
                        request_key = finer;
                        request_priority = 32u - min(mip - wanted, 30u);
                    }
                    break;
                }
                finer = r.key;
                if (r.coarsest || mip >= 30u) { break; }
                mip += 1u;
            }
            if (!found) {
                fell_back = true;
                request_key = finer;
                request_priority = 1u;
                let point = surface + hit.normal * surface_offset(hit.distance);
                let normal = shading_normal(hit.mesh, hit.tri, surface, -dir);
                let to_light = light.corner.xyz + (light.edge_u.xyz + light.edge_v.xyz) * 0.5 - point;
                let lc = light_cone(point);
                if (lc.ok && dot(normal, to_light) > 0.0) {
                    var cov: array<f32, 64>;
                    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
                    irradiance.x = cone_visible(lc.c, normal, &cov);
                }
                // The sky above the horizon, with the ground below it: a
                // guess until a page arrives.
                irradiance.y = 3.14159265 * 0.5 * (1.0 + normal.z);
            }
            if (is_sample) {
                atomicAdd(&vc_stats[2], 1u);
                if (found && mip == wanted) { atomicAdd(&vc_stats[3], 1u); }
            }
            let e = camera.params.w * irradiance.x + light.sky.xyz * irradiance.y;
            color = albedo[hit.mesh].xyz * e / 3.14159265;
        }
        colors[pixel] = vec4<f32>(color, 1.0);
    }
    if (is_sample && request_priority > 0u) {
        vc_request(request_key, request_priority);
    }
    if (fell_back) { atomicAdd(&fallback_count, 1u); }
    workgroupBarrier();
    if (li == 0u) {
        let count = atomicLoad(&fallback_count);
        if (count > 0u) { atomicAdd(&vc_stats[0], count); }
    }
}
