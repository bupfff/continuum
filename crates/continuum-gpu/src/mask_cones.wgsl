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
    // The environment's radiance above the horizon, nine spherical harmonics
    // coefficients (RGB in xyz; objects.z 1 with an environment; M5 gate 11)
    sky_sh: array<vec4<f32>, 9>,
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
    // A circular cone around the pyramid (M5 gate 6): unit axis, and the
    // cosine and sine of its half-angle (cosine below -1: none). Testing a
    // box's bounding sphere against it culls what the window's planes, one at
    // a time, miss when the window lies far off the cone's forward axis (a
    // light seen at a grazing angle).
    bound: vec4<f32>,
    bound_sin: f32,
};

// Whether a sphere (center `p`, radius `r`) may meet the cone's bounding
// circular cone.
fn in_bound(c: Cone, p: vec3<f32>, r: f32) -> bool {
    if (c.bound.w < -1.0) { return true; }
    let d = p - c.apex;
    let len = length(d);
    if (len <= r) { return true; }
    let s = r / len;
    // cos(half + asin(s)): the widest angle from the axis the sphere reaches.
    let limit = c.bound.w * sqrt(1.0 - s * s) - c.bound_sin * s;
    return dot(c.bound.xyz, d) >= limit * len;
}

// The cone with its bounding circular cone set around `axis` (unit), through
// the directions of the window's corners. (By value: WGSL passes pointers
// only to whole variables, and naga's SPIR-V output fails on a pointer to a
// struct's member.)
fn bound_cone(cone: Cone, axis: vec3<f32>) -> Cone {
    var c = cone;
    var cos_half = 1.0;
    for (var k = 0u; k < 4u; k++) {
        let d = normalize(direction(c, f32(k & 1u), f32(k >> 1u)));
        cos_half = min(cos_half, dot(axis, d));
    }
    // A little wider, for rounding.
    let cos_b = clamp(cos_half - 1e-5, -1.0, 1.0);
    c.bound = vec4<f32>(axis, cos_b);
    c.bound_sin = sqrt(max(1.0 - cos_b * cos_b, 0.0));
    return c;
}

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
    if (far_out || near_out || out_u0 || out_u1 || out_v0 || out_v1 || below) { return false; }
    return in_bound(c, (lo + hi) * 0.5, 0.5 * length(hi - lo));
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
        draw_edge(acc, span, a, b, tri_weight);
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
    if (!in_bound(c, (lo + hi) * 0.5, 0.5 * length(hi - lo))) { return 0u; }
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

// The weight `rasterize` draws a triangle with: its object's opacity if it
// is alpha-blended (M10 gate 16; set per mesh by `add_mesh_coverage_if`).
var<private> tri_weight: f32 = 1.0;

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
    o.bound = vec4<f32>(rotate_to_object(x, c.bound.xyz), c.bound.w);
    o.bound_sin = c.bound_sin;
    return o;
}

// The prefilter threshold (bins) of the cone being traced: the light's
// (`light.params.x`), or for sky cones that times `light.objects.y` (as f32
// bits; 0 means 1).
var<private> prefilter_bins: f32 = 0.0;

// Whether the last light cone traced had no coverage at all, or every bin
// fully covered (`cone_irradiance_rect`).
var<private> last_empty: bool = false;
var<private> last_full: bool = false;

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
// Lighting levels streamed (M14 gate 4): an object whose lighting level was
// needed and absent in this invocation's walks (NONE: none). An absent
// level's objects hold NONE as their first triangle and point at a stub,
// their hierarchy's root alone (bounds and prefiltered coverage).
var<private> lit_missing: u32 = 0xffffffffu;

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

// The first invocation's `x`, shared by its packet (each invocation's own
// without packets).
fn packet_first(x: bool) -> bool {
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
    // An alpha-blended object covers by its opacity (M10 gate 16).
    tri_weight = 1.0;
    if (range.w != NONE) { tri_weight = 1.0 - xforms[4u * range.w + 3u].z; }
    var acc: array<f32, 68>;
    for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
    // Cells written by this mesh's edges; the running sum is zero outside them.
    var span = vec2<u32>(68u, 0u);
    var touched = false;
    if (light.counts.z == 2u) {
        // An absent lighting level's stub (M14 gates 4 and 8: its top levels)
        // answers what those levels answer, as the full walk would; a cone
        // that would descend below its frontier needs the level.
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
                if (node.info.w == 1u) {
                    if (descend) { lit_missing = m; }
                    continue;
                }
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
            // Coverage within 1e-5 of whole is whole: a bin covered by
            // several triangles sums to one only up to rounding, and a
            // whole bin lets the walk stop once every bin is.
            var covered = abs(running);
            if (covered >= 1.0 - 1e-5) { covered = 1.0; }
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

// A light from the list (M5), with RGB radiance. A rectangle: its corner
// (w: 0 if two-sided, 1 if one-sided) and edges. A sphere (corner.w: 2): its
// center in corner, its radius in edge_u.x; a spot if edge_v.w is 1, shining
// along edge_v.xyz, its falloff the square of a ramp in the cosine to that
// axis (edge_u.y the ramp's scale, edge_u.z its offset), as glTF's spot
// lights. Paths without a list read the light uniform as one rectangle.
struct LightRect {
    corner: vec4<f32>,
    edge_u: vec4<f32>,
    edge_v: vec4<f32>,
    radiance: vec4<f32>,
};

@group(0) @binding(52) var<storage, read> lights: array<LightRect>;

// The light uniform's rectangle (radiance 1), for paths without a light list.
fn uniform_light() -> LightRect {
    return LightRect(vec4<f32>(light.corner.xyz, f32(light.counts.w)), light.edge_u, light.edge_v, vec4<f32>(1.0));
}

fn light_cone(apex: vec3<f32>) -> LightCone {
    return light_cone_rect(apex, uniform_light());
}

fn is_sphere(r: LightRect) -> bool {
    return r.corner.w > 1.5;
}

// Where light `r` is: a rectangle's middle or a sphere's center.
fn light_center(r: LightRect) -> vec3<f32> {
    if (is_sphere(r)) { return r.corner.xyz; }
    return r.corner.xyz + (r.edge_u.xyz + r.edge_v.xyz) * 0.5;
}

// A spot's falloff in direction `w` from its center (1 for other lights).
fn spot_falloff(r: LightRect, w: vec3<f32>) -> f32 {
    if (!is_sphere(r) || r.edge_v.w < 0.5) { return 1.0; }
    let a = clamp(dot(r.edge_v.xyz, w) * r.edge_u.y + r.edge_u.z, 0.0, 1.0);
    return a * a;
}

// Whether any of light `r` lies above the horizon of `normal` at `point`
// (a rectangle: its middle).
fn faces_light(r: LightRect, point: vec3<f32>, normal: vec3<f32>) -> bool {
    let to_light = light_center(r) - point;
    if (is_sphere(r)) { return dot(normal, to_light) > -r.edge_u.x; }
    return dot(normal, to_light) > 0.0;
}

// The fraction of each bin of a sphere light's cone inside the sphere's
// disk, the circle inscribed in the cone's window.
const DISK: array<f32, 64> = array<f32, 64>(
    0.000000, 0.160355, 0.694958, 0.957934, 0.957934, 0.694958, 0.160355, 0.000000,
    0.160355, 0.939877, 1.000000, 1.000000, 1.000000, 1.000000, 0.939877, 0.160355,
    0.694958, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.694958,
    0.957934, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.957934,
    0.957934, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.957934,
    0.694958, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.694958,
    0.160355, 0.939877, 1.000000, 1.000000, 1.000000, 1.000000, 0.939877, 0.160355,
    0.000000, 0.160355, 0.694958, 0.957934, 0.957934, 0.694958, 0.160355, 0.000000,
);

// A sphere light's cone from `apex`: around the direction to its center,
// the window the square around its disk, as deep as its silhouette (only
// geometry inside the sphere lies between that and its surface).
fn light_cone_sphere(apex: vec3<f32>, r: LightRect) -> LightCone {
    var out: LightCone;
    out.ok = false;
    prefilter_bins = light.params.x;
    let to_center = r.corner.xyz - apex;
    let d = length(to_center);
    let radius = r.edge_u.x;
    if (d <= radius * 1.0001) { return out; }
    let forward = to_center / d;
    let s = radius / d;
    let t = s / sqrt(max(1.0 - s * s, 1e-12));
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(forward.x) > 0.9);
    let right = normalize(cross(helper, forward));
    out.c.apex = apex;
    out.c.right = right;
    out.c.up = cross(forward, right);
    out.c.forward = forward;
    out.c.window = vec4<f32>(-t, -t, t, t);
    out.c.far_normal = forward;
    out.c.far_offset = dot(forward, apex) + (d - radius * s) * (1.0 - 1e-5);
    out.c = bound_cone(out.c, forward);
    out.ok = true;
    return out;
}

// The unshadowed irradiance per unit radiance from a sphere light at
// `point` (normal `normal`): the projected solid angle of its cap, clipped
// to the horizon where it crosses it (Snyder 1996, as in Lagarde and de
// Rousiers 2014). 0 inside the sphere.
fn sphere_unshadowed(point: vec3<f32>, normal: vec3<f32>, r: LightRect) -> f32 {
    let to_center = r.corner.xyz - point;
    let d2 = dot(to_center, to_center);
    let radius = r.edge_u.x;
    if (d2 <= radius * radius) { return 0.0; }
    let sin2 = radius * radius / d2;
    let cos_t = clamp(dot(normal, to_center) * inverseSqrt(d2), -1.0, 1.0);
    if (cos_t * cos_t > sin2) {
        return 3.14159265 * sin2 * max(cos_t, 0.0);
    }
    let sin_t = sqrt(1.0 - cos_t * cos_t);
    let x = sqrt(1.0 / sin2 - 1.0);
    let y = clamp(-x * cos_t / sin_t, -1.0, 1.0);
    let q = sin_t * sqrt(1.0 - y * y);
    return max((cos_t * acos(y) - x * q) * sin2 + atan(q / x), 0.0);
}

// The share of a sphere light's cap that coverage `cov` leaves open: bins
// weighted by their part inside the disk, their solid angle and their
// cosine to `normal` at their centers.
fn sphere_open_share(c: Cone, normal: vec3<f32>, cov: ptr<function, array<f32, 64>>) -> f32 {
    var total = 0.0;
    var open = 0.0;
    let g = f32(GRID);
    for (var b = 0u; b < BINS; b++) {
        let m = DISK[b];
        if (m <= 0.0) { continue; }
        let d = direction(c, (f32(b % GRID) + 0.5) / g, (f32(b / GRID) + 0.5) / g);
        let l2 = dot(d, d);
        let w = m * max(dot(normal, d), 0.0) / (l2 * l2);
        total += w;
        open += w * (1.0 - (*cov)[b]);
    }
    if (total <= 0.0) { return 1.0; }
    return open / total;
}

fn light_cone_rect(apex: vec3<f32>, r: LightRect) -> LightCone {
    if (is_sphere(r)) { return light_cone_sphere(apex, r); }
    var out: LightCone;
    out.ok = false;
    // Light cones prefilter at the light's threshold (sky cones set theirs).
    prefilter_bins = light.params.x;
    let corner0 = r.corner.xyz;
    let edge_u = r.edge_u.xyz;
    let edge_v = r.edge_v.xyz;
    var forward = normalize(cross(edge_u, edge_v));
    // Behind a one-sided light: no direct light.
    if (r.corner.w > 0.5 && dot(forward, apex - corner0) <= 0.0) { return out; }
    if (dot(forward, corner0 - apex) < 0.0) { forward = -forward; }
    let dist = dot(forward, corner0 - apex);
    if (dist <= 1e-6) { return out; }
    let right = normalize(edge_u);
    let up = cross(forward, right);
    let rel = corner0 - apex;
    var lo = vec2<f32>(3.4e38);
    var hi = vec2<f32>(-3.4e38);
    for (var k = 0u; k < 4u; k++) {
        let corner = rel + edge_u * f32(k & 1u) + edge_v * f32((k >> 1u) & 1u);
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
    // Around the direction to the light's center.
    out.c = bound_cone(out.c, normalize(rel + (edge_u + edge_v) * 0.5));
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
                    // info.z 1: no prefiltered coverage (the root and the
                    // movers' subtree, whose boxes change).
                    if (node.info.z == 0u && splat_object_node(c, n, node.lo.xyz, node.hi.xyz, cov, full)) {
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
                // Nearest child first along the cone when meshes combine by
                // the maximum (light cones): the result does not depend on the
                // order, and an occluded cone fills every bin, and stops,
                // sooner (M5 gate 6). Sky cones keep a fixed order, as their
                // products would round differently.
                var near = node.info.x;
                var far = node.info.x + 1u;
                if (!independent_occluders) {
                    let a = nodes[near];
                    let b = nodes[far];
                    // Along the cone's true axis (a light cone's forward axis is
                    // its light's normal).
                    let axis = select(c.forward, c.bound.xyz, c.bound.w >= -1.0 && any(c.bound.xyz != vec3<f32>(0.0)));
                    let da = dot(axis, (a.lo.xyz + a.hi.xyz) * 0.5 - c.apex);
                    let db = dot(axis, (b.lo.xyz + b.hi.xyz) * 0.5 - c.apex);
                    if (packet_first(db < da)) {
                        near = node.info.x + 1u;
                        far = node.info.x;
                    }
                }
                // The near one is popped first.
                stack_set(sp, far);
                stack_set(sp + 1u, near);
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
    // The window crosses the horizon: the window and each covered bin are
    // clipped to it, as for sky faces. (Summing every bin's unclipped
    // weight, as before, counted bins below the horizon as light.)
    var blocked = 0.0;
    for (var b = 0u; b < BINS; b++) {
        let covered = (*cov)[b];
        if (covered > 0.0) { blocked += covered * clipped_bin_weight(c, normal, b); }
    }
    return max(clipped_window_weight(c, normal) - blocked, 0.0);
}

// Irradiance per unit radiance at `apex` (surface normal `normal`) from the
// light, through a mask cone. `i` only selects where debug output goes.
fn cone_irradiance(i: u32, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    return cone_irradiance_if(true, i, apex, normal);
}

// `cone_irradiance` where `on` (0 elsewhere), keeping the packet together.
fn cone_irradiance_if(on: bool, i: u32, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    return cone_irradiance_rect(on, i, apex, normal, uniform_light());
}

// `cone_irradiance_if` from light `r`.
fn cone_irradiance_rect(on: bool, i: u32, apex: vec3<f32>, normal: vec3<f32>, r: LightRect) -> f32 {
    let stage = u32(light.params.z);
    if (stage == 1u) { return 0.0; }
    let lc = light_cone_rect(apex, r);
    let traced = on && lc.ok;
    // Explicit zeroing: don't rely on loop-scoped `var`s being re-initialized.
    var cov: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
    var full = 0u;
    if (stage != 2u) { scene_coverage_if(traced, lc.c, i, &cov, &full); }
    last_empty = true;
    last_full = true;
    for (var b = 0u; b < BINS; b++) {
        if (cov[b] > 0.0) { last_empty = false; }
        if (cov[b] < 1.0) { last_full = false; }
    }
    if (!traced) { return 0.0; }
    if (is_sphere(r)) {
        return sphere_unshadowed(apex, normal, r) * sphere_open_share(lc.c, normal, &cov);
    }
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
    face.c = bound_cone(face.c, forward);
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
    let n = cross(v1 - v0, v2 - v0);
    if (cull_back && dot(n, dir) > 0.0) { return best; }
    let d = ray_triangle(eye, dir, v0, v1, v2);
    // Coincident faces of one mesh (a double-sided card's front and back,
    // whose distances differ only by rounding): the one facing the ray wins,
    // as a back face counts as inside geometry (M10 gate 8).
    if (d > 0.0 && m == best.mesh && abs(d - best.distance) <= 2e-6 * d) {
        let facing = dot(n, dir) < 0.0;
        if (facing != (dot(best.normal, dir) < 0.0)) {
            if (facing) { return Hit(d, m, t, normalize(n)); }
            return best;
        }
    }
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
            // An absent lighting level (M14 gates 4 and 8): a ray into a box
            // on its stub's frontier needs it.
            if (node.info.w == 1u) {
                lit_missing = m;
                continue;
            }
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

// The camera ray through pixel `gid`: through its center, moved by
// (right.w, up.w) pixels (zero but for supersampled references).
// The share of a pixel a camera ray stands for (1; a quarter's ray at a
// silhouette, half a pixel wide: 0.5).
var<private> sub_share: f32 = 1.0;

// A camera ray `(dx, dy)` pixels from pixel `gid`'s (x right, y down).
fn camera_dir_at(gid: vec3<u32>, dx: f32, dy: f32) -> vec3<f32> {
    let aspect = camera.params.x / camera.params.y;
    let sx = ((f32(gid.x) + 0.5 + camera.right.w + dx) / camera.params.x * 2.0 - 1.0) * camera.params.z * aspect;
    let sy = (1.0 - (f32(gid.y) + 0.5 + camera.up.w + dy) / camera.params.y * 2.0) * camera.params.z;
    return normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
}

// Whether pixel `gid` is at a silhouette (M7 gate 3): a neighbour sees
// another object, or a surface more than 5% nearer or farther.
fn is_edge(gid: vec3<u32>, w: u32, h: u32) -> bool {
    let me = vis[gid.y * w + gid.x];
    let d = bitcast<f32>(me.z);
    for (var k = 0u; k < 4u; k++) {
        var x = i32(gid.x);
        var y = i32(gid.y);
        if (k == 0u) { x -= 1; } else if (k == 1u) { x += 1; } else if (k == 2u) { y -= 1; } else { y += 1; }
        if (x < 0 || y < 0 || x >= i32(w) || y >= i32(h)) { continue; }
        let o = vis[u32(y) * w + u32(x)];
        if (o.x != me.x) { return true; }
        if (me.x != NONE && abs(bitcast<f32>(o.z) - d) > 0.05 * d) { return true; }
    }
    return false;
}

fn camera_dir(gid: vec3<u32>) -> vec3<f32> {
    let aspect = camera.params.x / camera.params.y;
    let sx = ((f32(gid.x) + 0.5 + camera.right.w) / camera.params.x * 2.0 - 1.0) * camera.params.z * aspect;
    let sy = (1.0 - (f32(gid.y) + 0.5 + camera.up.w) / camera.params.y * 2.0) * camera.params.z;
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
// Triangle `t`'s vertex color at world point `p` on object `m` (M10 gate
// 17): each corner's code in its position's w (1 plus an sRGB byte per
// channel, red lowest; 0: none, white), decoded to linear and interpolated.
fn vertex_color(m: u32, t: u32, p: vec3<f32>) -> vec3<f32> {
    let c0 = tris[3u * t].w;
    if (c0 == 0.0) { return vec3<f32>(1.0); }
    let b = barycentric(m, t, p);
    return decode_color(c0) * b.x + decode_color(tris[3u * t + 1u].w) * b.y + decode_color(tris[3u * t + 2u].w) * b.z;
}

fn decode_color(code: f32) -> vec3<f32> {
    let c = u32(code) - 1u;
    let s = vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32(c >> 16u)) / 255.0;
    return select(pow((s + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)), s / 12.92, s <= vec3<f32>(0.04045));
}

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

// Base color textures (M6): a layer per texture, each TEXTURE_SIZE square with
// mip levels; per object, its layer (NONE: none); per geometry triangle, its
// corners' texture coordinates.
@group(0) @binding(55) var base_color: texture_2d_array<f32>;
@group(0) @binding(56) var base_sampler: sampler;
// Per object its texture layer, then per object its emission as two words
// of half floats (`bounce_emission`), then per object its texture slots'
// wrap modes (`wrap_uv`).
@group(0) @binding(57) var<storage, read> object_texture: array<u32>;

// Texture slots in the wrap words (M10 gate 20).
const SLOT_BASE: u32 = 0u;
const SLOT_EMISSIVE: u32 = 1u;
const SLOT_NORMAL: u32 = 2u;
const SLOT_MR: u32 = 3u;

// Texture coordinates `uv` under object `m`'s wrap modes for texture slot
// `slot` (glTF samplers, M10 gate 20), for a texture `size` texels across
// read at level of detail `lod` with the repeating sampler: clamping keeps
// them half a texel (at the coarser level read) inside the edges, so
// filtering never reaches the far edge; mirroring folds them into [0, 1]
// every two tiles, then clamps the same way (the edge texel repeats at each
// fold, as glTF's mirrored repeat does).
fn wrap_uv(m: u32, slot: u32, uv: vec2<f32>, size: f32, lod: f32) -> vec2<f32> {
    let modes = wrap_modes(m, slot);
    if (modes == 0u) { return uv; }
    let level = clamp(ceil(lod), 0.0, log2(max(size, 1.0)));
    let half = 0.5 * exp2(level) / max(size, 1.0);
    return vec2<f32>(wrap_axis(uv.x, modes & 3u, half), wrap_axis(uv.y, modes >> 2u, half));
}

// Object `m`'s wrap modes for texture slot `slot`: two bits for u, then two
// for v (0 repeat, 1 clamp, 2 mirror).
fn wrap_modes(m: u32, slot: u32) -> u32 {
    let n = arrayLength(&object_texture) / 4u;
    if (m >= n) { return 0u; }
    return (object_texture[3u * n + m] >> (4u * slot)) & 15u;
}

// Per axis, 0 where `uv` lies past a clamped edge (the texture does not
// vary along that axis there), else 1: a footprint's extent after
// wrapping, for choosing its level of detail.
fn wrap_extent(modes: u32, uv: vec2<f32>) -> vec2<f32> {
    let out_u = (modes & 3u) == 1u && (uv.x < 0.0 || uv.x > 1.0);
    let out_v = (modes >> 2u) == 1u && (uv.y < 0.0 || uv.y > 1.0);
    return vec2<f32>(select(1.0, 0.0, out_u), select(1.0, 0.0, out_v));
}

// Level of detail `lod`, chosen for a footprint at `uv` without wrapping,
// as wrapping leaves it: the base level where both axes lie past clamped
// edges. (With one axis past its edge, the isotropic level stays: only the
// box filter knows the footprint per axis.)
fn wrap_lod(m: u32, slot: u32, uv: vec2<f32>, lod: f32) -> f32 {
    let e = wrap_extent(wrap_modes(m, slot), uv);
    return select(lod, -8.0, e.x == 0.0 && e.y == 0.0);
}

fn wrap_axis(u: f32, mode: u32, half: f32) -> f32 {
    if (mode == 1u) { return clamp(u, half, 1.0 - half); }
    if (mode == 2u) { return clamp(1.0 - abs(fract(u * 0.5) * 2.0 - 1.0), half, 1.0 - half); }
    return u;
}

// Object `m`'s emission, from `object_texture` (for bounce samples, which
// cannot bind `materials` within the buffer limit; M10 gate 11).
fn bounce_emission(m: u32) -> vec3<f32> {
    let i = arrayLength(&object_texture) / 4u + 2u * m;
    if (i + 1u >= arrayLength(&object_texture)) { return vec3<f32>(0.0); }
    return vec3<f32>(unpack2x16float(object_texture[i]), unpack2x16float(object_texture[i + 1u]).x);
}

// `bounce_emission` times its emissive texture at `p` on triangle `t`
// (M10 gate 14), read for `footprint` world units.
fn bounce_emission_at(m: u32, t: u32, p: vec3<f32>, footprint: f32) -> vec3<f32> {
    let i = arrayLength(&object_texture) / 4u + 2u * m;
    if (i + 1u >= arrayLength(&object_texture)) { return vec3<f32>(0.0); }
    return emission_textured(m, t, p, bounce_emission(m), object_texture[i + 1u] >> 16u, footprint);
}
@group(0) @binding(58) var<storage, read> tri_tex: array<vec2<f32>>;
// Linear textures: normal maps (M10 gate 5) and metallic-roughness
// textures, a layer each; objects name theirs in `materials`.
@group(0) @binding(79) var normal_maps: texture_2d_array<f32>;

// Normal `n` (the interpolated shading normal at world point `p` on
// triangle `t` of object `m`, seen at `distance` along camera ray `dir`)
// bent by the object's normal map, if it has one. The tangent frame comes
// from the triangle's texture coordinates' derivatives: tangent along
// increasing u, bitangent toward decreasing v (the image's top), each
// orthogonalized against `n`. Read at the mip matching the pixel's
// footprint.
fn mapped_normal(m: u32, t: u32, p: vec3<f32>, n: vec3<f32>, distance: f32, dir: vec3<f32>) -> vec3<f32> {
    let layer = materials[2u * m].w >> 16u;
    if (layer == 0u) { return n; }
    let b = barycentric(m, t, p);
    let t0 = tri_tex[3u * t];
    let t1 = tri_tex[3u * t + 1u];
    let t2 = tri_tex[3u * t + 2u];
    let uv = t0 * b.x + t1 * b.y + t2 * b.z;
    let v0 = tris[3u * t].xyz;
    let e1 = tris[3u * t + 1u].xyz - v0;
    let e2 = tris[3u * t + 2u].xyz - v0;
    let d1 = t1 - t0;
    let d2 = t2 - t0;
    let det = d1.x * d2.y - d1.y * d2.x;
    if (abs(det) < 1e-20) { return n; }
    let dpdu = normal_to_world(m, (e1 * d2.y - e2 * d1.y) / det);
    let dpdv = normal_to_world(m, (e2 * d1.x - e1 * d2.x) / det);
    var tangent = dpdu - n * dot(n, dpdu);
    if (dot(tangent, tangent) < 1e-20) { return n; }
    tangent = normalize(tangent);
    var bitangent = cross(n, tangent);
    if (dot(bitangent, dpdv) > 0.0) { bitangent = -bitangent; }
    // MikkTSpace's tangents where the geometry has them (M10 gate 13):
    // interpolated, kept orthogonal to the normal; the bitangent is the
    // handedness times normal × tangent, as glTF specifies.
    let c0 = normals[3u * t].w;
    let c1 = normals[3u * t + 1u].w;
    let c2 = normals[3u * t + 2u].w;
    if (c0 != 0.0 && c1 != 0.0 && c2 != 0.0) {
        let k0 = decode_tangent(c0);
        var mt = normal_to_world(m, k0.xyz * b.x + decode_tangent(c1).xyz * b.y + decode_tangent(c2).xyz * b.z);
        mt -= n * dot(n, mt);
        if (dot(mt, mt) > 1e-20) {
            tangent = normalize(mt);
            bitangent = cross(n, tangent) * k0.w;
        }
    }
    // A texel's size on the surface, and the pixel's footprint there.
    let size = f32(textureDimensions(normal_maps).x);
    let scale = object_scale(m);
    let world_area = 0.5 * length(cross(e1, e2)) * scale * scale;
    let texel_area = 0.5 * abs(det) * size * size;
    let texel = sqrt(world_area / max(texel_area, 1e-12));
    let footprint = select(1.0, camera.eye.w, camera.eye.w > 0.0) * sub_share;
    let pixel = footprint * distance * 2.0 * camera.params.z / camera.params.y / max(abs(dot(n, dir)), 0.2);
    let lod = wrap_lod(m, SLOT_NORMAL, uv, log2(max(pixel / max(texel, 1e-12), 1e-6)));
    let s = textureSampleLevel(normal_maps, base_sampler, wrap_uv(m, SLOT_NORMAL, uv, size, lod), i32(layer - 1u), lod).xyz * 2.0 - vec3<f32>(1.0);
    let bent = tangent * s.x + bitangent * s.y + n * s.z;
    if (dot(bent, bent) < 1e-12) { return n; }
    return normalize(bent);
}

// A tangent and handedness from its code in the normals' w (M10 gate 13):
// octahedral coordinates at 11 bits each, plus one, as an exact integer, its
// sign the handedness.
fn decode_tangent(code: f32) -> vec4<f32> {
    let a = abs(code) - 1.0;
    let qu = floor(a / 2048.0);
    let qv = a - qu * 2048.0;
    let p = vec2<f32>(qu, qv) / 2047.0 * 2.0 - vec2<f32>(1.0);
    var d = vec3<f32>(p, 1.0 - abs(p.x) - abs(p.y));
    if (d.z < 0.0) {
        d = vec3<f32>((1.0 - abs(d.y)) * select(-1.0, 1.0, d.x >= 0.0), (1.0 - abs(d.x)) * select(-1.0, 1.0, d.y >= 0.0), d.z);
    }
    return vec4<f32>(normalize(d), sign(code));
}

// The surface's color at world point `p` on triangle `t` of object `m`: its
// albedo times its texture, sampled at the mip level whose texels match the
// pixel's footprint (`distance` along the camera ray `dir`, normal `n`).
fn surface_albedo(m: u32, t: u32, p: vec3<f32>, distance: f32, dir: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let base = albedo[m].xyz * vertex_color(m, t, p);
    let layer = object_texture[m];
    if (layer == NONE) { return base; }
    let b = barycentric(m, t, p);
    let t0 = tri_tex[3u * t];
    let t1 = tri_tex[3u * t + 1u];
    let t2 = tri_tex[3u * t + 2u];
    let uv = t0 * b.x + t1 * b.y + t2 * b.z;
    // A texel's size on the surface: the triangle's area over its area in
    // texels.
    let size = f32(textureDimensions(base_color).x);
    let v0 = tris[3u * t].xyz;
    let scale = object_scale(m);
    let world_area = 0.5 * length(cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0)) * scale * scale;
    let e1 = (t1 - t0) * size;
    let e2 = (t2 - t0) * size;
    let texel_area = 0.5 * abs(e1.x * e2.y - e1.y * e2.x);
    let texel = sqrt(world_area / max(texel_area, 1e-12));
    // forward.w 1: a box filter over the pixel's footprint (M7 gate 2).
    if (camera.forward.w > 0.5) {
        let boxed = texture_box(m, t, dir, distance, uv, size, layer);
        if (boxed.w > 0.0) { return base * boxed.xyz; }
    }
    // A pixel's footprint there, stretched by the viewing angle; times
    // eye.w where set (a reference's sub-pixel samples: their own share of
    // the pixel).
    let footprint = select(1.0, camera.eye.w, camera.eye.w > 0.0) * sub_share;
    let pixel = footprint * distance * 2.0 * camera.params.z / camera.params.y / max(abs(dot(n, dir)), 0.2);
    let lod = wrap_lod(m, SLOT_BASE, uv, log2(max(pixel / max(texel, 1e-12), 1e-6)));
    return base * textureSampleLevel(base_color, base_sampler, wrap_uv(m, SLOT_BASE, uv, size, lod), i32(layer), lod).xyz;
}

// The texture over a pixel's footprint (M7 gate 2): the camera ray's change
// per pixel projected onto triangle `t`'s plane (ray differentials), mapped
// to texture space, and read with four bilinear taps at the centers of the
// footprint's 2×2 quarters, each at the mip of a quarter's longest axis. w
// is 0 where the footprint is degenerate (the caller falls back).
fn texture_box(m: u32, t: u32, dir: vec3<f32>, distance: f32, uv: vec2<f32>, size: f32, layer: u32) -> vec4<f32> {
    let share = select(1.0, camera.eye.w, camera.eye.w > 0.0) * sub_share;
    let per_pixel = 2.0 * camera.params.z / camera.params.y * share;
    let ddx = camera.right.xyz * per_pixel;
    let ddy = -camera.up.xyz * per_pixel;
    // The unnormalized ray direction (its component along forward is 1).
    let along = dot(dir, camera.forward.xyz);
    if (along <= 1e-6) { return vec4<f32>(0.0); }
    let d = dir / along;
    let s = distance / length(d);
    let v0 = tris[3u * t].xyz;
    let e1 = tris[3u * t + 1u].xyz - v0;
    let e2 = tris[3u * t + 2u].xyz - v0;
    let gn = normalize(normal_to_world(m, cross(e1, e2)));
    let dn = dot(d, gn);
    if (abs(dn) < 1e-6) { return vec4<f32>(0.0); }
    var dpx = s * (ddx - d * (dot(ddx, gn) / dn));
    var dpy = s * (ddy - d * (dot(ddy, gn) / dn));
    let x = meshes[m].w;
    if (x != NONE) {
        let inv = xforms[4u * x + 3u].y;
        dpx = rotate_to_object(x, dpx) * inv;
        dpy = rotate_to_object(x, dpy) * inv;
    }
    let g11 = dot(e1, e1);
    let g12 = dot(e1, e2);
    let g22 = dot(e2, e2);
    let det = g11 * g22 - g12 * g12;
    if (abs(det) < 1e-30) { return vec4<f32>(0.0); }
    let u1 = tri_tex[3u * t + 1u] - tri_tex[3u * t];
    let u2 = tri_tex[3u * t + 2u] - tri_tex[3u * t];
    let ax = (g22 * dot(dpx, e1) - g12 * dot(dpx, e2)) / det;
    let bx = (g11 * dot(dpx, e2) - g12 * dot(dpx, e1)) / det;
    let ay = (g22 * dot(dpy, e1) - g12 * dot(dpy, e2)) / det;
    let by = (g11 * dot(dpy, e2) - g12 * dot(dpy, e1)) / det;
    let dux = ax * u1 + bx * u2;
    let duy = ay * u1 + by * u2;
    var lod = log2(max(0.5 * max(length(dux), length(duy)) * size, 1e-6));
    // Past a clamped edge the texture does not vary along that axis: the
    // footprint's extent along it is dropped (M10 gate 20).
    let e = wrap_extent(wrap_modes(m, SLOT_BASE), uv);
    if (e.x == 0.0 || e.y == 0.0) {
        lod = log2(max(0.5 * max(length(dux * e), length(duy * e)) * size, 1e-6));
    }
    var sum = vec3<f32>(0.0);
    for (var q = 0u; q < 4u; q++) {
        let o = (f32(q & 1u) - 0.5) * 0.5 * dux + (f32(q >> 1u) - 0.5) * 0.5 * duy;
        sum += textureSampleLevel(base_color, base_sampler, wrap_uv(m, SLOT_BASE, uv + o, size, lod), i32(layer), lod).xyz;
    }
    return vec4<f32>(sum * 0.25, 1.0);
}

// Interpolated shading normal (world space) at world point `p` on triangle `t`
// of object `m`, on the side `toward` points to.
fn shading_normal(m: u32, t: u32, p: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let b = barycentric(m, t, p);
    let v0 = tris[3u * t].xyz;
    let face = cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0);
    let s = normals[3u * t].xyz * b.x + normals[3u * t + 1u].xyz * b.y + normals[3u * t + 2u].xyz * b.z;
    // A degenerate interpolated normal (zero or NaN): the face's.
    var n = normalize(normal_to_world(m, select(face, s, dot(s, s) > 1e-20)));
    let geometric = normal_to_world(m, face);
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
    return rect_distance(uniform_light(), eye, dir);
}

// Distance along a camera ray to the front of light `l` if it is a
// one-sided rectangle or a sphere (or 3.4e38).
fn rect_distance(l: LightRect, eye: vec3<f32>, dir: vec3<f32>) -> f32 {
    if (is_sphere(l)) {
        // The nearer intersection with the sphere, if the eye is outside it.
        let oc = eye - l.corner.xyz;
        let b = dot(oc, dir);
        let c = dot(oc, oc) - l.edge_u.x * l.edge_u.x;
        let disc = b * b - c;
        if (c <= 0.0 || disc < 0.0) { return 3.4e38; }
        let t = -b - sqrt(disc);
        if (t <= 0.0) { return 3.4e38; }
        return t;
    }
    if (l.corner.w < 0.5) { return 3.4e38; }
    let u = l.edge_u.xyz;
    let v = l.edge_v.xyz;
    let n = cross(u, v);
    let facing = dot(dir, n);
    if (facing >= 0.0) { return 3.4e38; }
    let t = dot(l.corner.xyz - eye, n) / facing;
    if (t <= 0.0) { return 3.4e38; }
    let q = eye + dir * t - l.corner.xyz;
    let s = dot(q, u) / dot(u, u);
    let r = dot(q, v) / dot(v, v);
    if (s < 0.0 || s > 1.0 || r < 0.0 || r > 1.0) { return 3.4e38; }
    return t;
}

// The light a camera ray sees nearer than `limit`, as its index + 1 (0:
// none).
fn seen_light(eye: vec3<f32>, dir: vec3<f32>, limit: f32) -> u32 {
    var seen = 0u;
    var nearest = limit;
    for (var l = 0u; l < arrayLength(&lights); l++) {
        let t = rect_distance(lights[l], eye, dir);
        if (t < nearest) {
            nearest = t;
            seen = l + 1u;
        }
    }
    return seen;
}

// Direct irradiance at a hit through a mask cone (0 if the surface faces away).
fn hit_irradiance(pixel: u32, point: vec3<f32>, normal: vec3<f32>) -> f32 {
    return hit_irradiance_if(true, pixel, point, normal);
}

fn hit_irradiance_if(on: bool, pixel: u32, point: vec3<f32>, normal: vec3<f32>) -> f32 {
    return hit_irradiance_rect(on, pixel, point, normal, uniform_light());
}

// `hit_irradiance_if` from light `r`.
fn hit_irradiance_rect(on: bool, pixel: u32, point: vec3<f32>, normal: vec3<f32>, r: LightRect) -> f32 {
    let spot = spot_falloff(r, normalize(point - light_center(r)));
    return spot * cone_irradiance_rect(on && spot > 0.0 && faces_light(r, point, normal), pixel, point, normal, r);
}

// Direct light at a hit from every light in the list (RGB).
// The unshadowed irradiance per unit radiance from light `r` at `point`
// (normal `normal`): the light's window clipped to the horizon, which is
// `cone_visible` with nothing covered, so a texel no geometry shadows has a
// ratio of one; 0 where `hit_irradiance_rect` traces nothing.
fn unshadowed_rect(point: vec3<f32>, normal: vec3<f32>, r: LightRect) -> f32 {
    if (!faces_light(r, point, normal)) { return 0.0; }
    if (is_sphere(r)) {
        return spot_falloff(r, normalize(point - r.corner.xyz)) * sphere_unshadowed(point, normal, r);
    }
    return rect_open(point, normal, r);
}

// A rectangle's projected solid angle from `point` above the horizon of
// `normal`: Lambert's formula over its corners, clipped to the horizon where
// it crosses it. The same polygon of directions as its light cone's window
// (`clipped_window_weight`), without building the cone. 0 behind a one-sided
// light or in its plane.
fn rect_open(point: vec3<f32>, normal: vec3<f32>, r: LightRect) -> f32 {
    let u = r.edge_u.xyz;
    let v = r.edge_v.xyz;
    let c0 = r.corner.xyz - point;
    let facing = dot(normalize(cross(u, v)), c0);
    if ((r.corner.w > 0.5 && facing >= 0.0) || abs(facing) <= 1e-6) { return 0.0; }
    var d: array<vec3<f32>, 4>;
    d[0] = c0;
    d[1] = c0 + u;
    d[2] = c0 + u + v;
    d[3] = c0 + v;
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

fn lights_irradiance(pixel: u32, point: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var e = vec3<f32>(0.0);
    for (var l = 0u; l < arrayLength(&lights); l++) {
        let r = lights[l];
        e += r.radiance.xyz * hit_irradiance_rect(true, pixel, point, normal, r);
    }
    return e;
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
    vis[gid.y * w + gid.x] = vec4<u32>(hit.mesh, hit.tri, bitcast<u32>(hit.distance), seen_light(eye, dir, hit.distance));
}

// The hit a visibility pass recorded, rebuilt exactly as `camera_hit` returns
// it (the normal is recomputed the same way). `w` of the result: 1 if the
// pixel sees the light.
fn recorded_hit(pixel: u32, dir: vec3<f32>) -> Hit {
    let v = vis[pixel];
    return hit_from(v.xyz, dir);
}

// A hit from its object, triangle and distance (bits), its normal facing
// the ray `dir`.
fn hit_from(v: vec3<u32>, dir: vec3<f32>) -> Hit {
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
// `seen`: the light the camera ray sees in front of the hit, index + 1 (0:
// none).
fn shade_per_pixel(pixel: u32, eye: vec3<f32>, dir: vec3<f32>, hit: Hit, seen: u32) {
    if (seen != 0u) {
        let l = lights[seen - 1u];
        colors[pixel] = vec4<f32>(l.radiance.xyz * spot_falloff(l, -dir), 1.0);
        return;
    }
    // A ray that misses everything sees the sky.
    if (hit.mesh == 0xffffffffu) { colors[pixel] = vec4<f32>(light.sky.xyz, 1.0); return; }
    colors[pixel] = vec4<f32>(per_pixel_hit_color(pixel, eye, dir, hit), 1.0);
}

// The color of surface `hit`, seen along `dir` from `eye`, lit through a
// mask cone per light (and the sky's cones).
fn per_pixel_hit_color(pixel: u32, eye: vec3<f32>, dir: vec3<f32>, hit: Hit) -> vec3<f32> {
    let surface = eye + dir * hit.distance;
    let point = surface + hit.normal * surface_offset(hit.distance);
    let normal = shading_normal(hit.mesh, hit.tri, surface, -dir);
    var e = lights_irradiance(pixel, point, normal);
    if (any(light.sky.xyz > vec3<f32>(0.0))) {
        // params.w bit 4: faces in the hit object's frame, as the cache's
        // texels with own-occlusion masks use (and, with this bit, those
        // without).
        if ((u32(light.params.w) & 4u) != 0u) { sky_frame = meshes[hit.mesh].w; }
        e += light.sky.xyz * sky_irradiance(point, normal);
        sky_frame = NONE;
    }
    return surface_albedo(hit.mesh, hit.tri, surface, hit.distance, dir, hit.normal) * e / 3.14159265;
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
    shade_per_pixel(gid.y * w + gid.x, eye, dir, hit, seen_light(eye, dir, hit.distance));
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let pixel = gid.y * w + gid.x;
    shade_per_pixel(pixel, camera.eye.xyz, dir, recorded_hit(pixel, dir), vis[pixel].w);
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
    shade_cached(pixel, camera.eye.xyz, dir, recorded_hit(pixel, dir), vis[pixel].w != 0u);
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
    // x: sky samples to bake (each as six face jobs), y: 1 with a sky, z: lights
    // (a texel holds one channel per light, then the sky's), w: 1 if light
    // channels hold ratios (`CacheOptions::light_ratio`)
    sky: vec4<u32>,
    // Movers (M9): x: objects moved this frame (in `moved`), y: pages the
    // stale pass may append to the bake, z: 0, w: a page read within this
    // many frames is re-baked when stale (otherwise marked invalid)
    movers: vec4<u32>,
    // Moving lights (M9 gate 2): x: how many (each with ledger records),
    // y: frames in the pose ring, z: ε (f32 bits), w: the most texel jobs
    // the ledger may add to a frame's bake; movers.z's top bit: 1 if a
    // moving light moved this frame
    ledger: vec4<u32>,
    // Bounce light (M10 gate 1): x: samples to bake, y: 1 if on; z: 1 to
    // skip glossy reflection rays (M10 gate 2's diagnostic); w: 1 for four
    // rays at silhouette pixels (M7 gate 3), 2 and 3 through ray queries
    // (M7 gate 4)
    indirect: vec4<u32>,
    // x: 1 if bounce samples gather emission (M10 gate 11)
    extra: vec4<u32>,
};

const PAGE: u32 = 8u;
const PAGE_STRIDE: u32 = 9u;
const PAGE_TEXELS: u32 = 81u;
const EMPTY_KEY: u32 = 0xffffffffu;

@group(0) @binding(33) var<storage, read> vcharts: array<VChart>;
// Per chart cell: its first entry in `cell_tris`, and how many: the triangles
// whose chart coordinates' bounds meet the cell.
@group(0) @binding(34) var<storage, read> cells: array<vec2<u32>>;
@group(0) @binding(54) var<storage, read> cell_tris: array<u32>;
// Chart of each geometry triangle.
@group(0) @binding(35) var<storage, read> tri_vchart: array<u32>;
// Open addressing with linear probing: three key words, then the slot. x is
// EMPTY_KEY for a free entry (deletion moves entries back: no tombstones).
@group(0) @binding(36) var<storage, read_write> page_table: array<vec4<u32>>;
@group(0) @binding(37) var<storage, read_write> pool: array<f32>;
// Pages baked this frame: key words, then the slot.
@group(0) @binding(38) var<storage, read_write> bake_list: array<vec4<u32>>;
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
@group(0) @binding(43) var<storage, read_write> bake_jobs: array<vec2<u32>>;
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
@group(0) @binding(46) var<storage, read_write> sky_samples: array<vec2<u32>>;
@group(0) @binding(47) var<storage, read_write> sky_partial: array<f32>;
// Per baked page: its first sky sample, and how many it has (low 24 bits)
// with its sky stride (high 8 bits).
@group(0) @binding(48) var<storage, read_write> page_sky: array<vec2<u32>>;
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
// Movers (M9). Objects moved this frame, in world space relative to the
// origin (vec4s): per light, the union of the moved boxes' shadow regions
// (lo, hi); the union of the moved boxes (lo, hi); then per object its old
// box (lo with the object's index as bits in w, hi), its new box (lo, hi),
// and per light its old and new boxes' shadow regions (lo, hi each; empty
// if none). A shadow region bounds where a cone toward the light can meet
// the box (`shadow_region` in `hierarchy.rs`).
@group(0) @binding(60) var<storage, read> moved: array<vec4<f32>>;
// Per slot, written when its page is baked: the page's key with 1 in w
// (0: never baked), then the object-space box of its texels' apexes (lo,
// hi; floats as bits).
@group(0) @binding(61) var<storage, read_write> slot_info: array<vec4<u32>>;
// Per slot: 1 if its page is stale and was not re-baked (shading treats it
// as missing until the CPU evicts it, or a bake replaces it); 2 if the
// moving lights' ledger set it aside (missing too, but kept: the ledger
// re-bakes it once a pixel wants it).
@group(0) @binding(62) var<storage, read_write> slot_stale: array<atomic<u32>>;
// Per baked page (as `bake_list`): its texels' object-space box (lo, hi).
@group(0) @binding(63) var<storage, read_write> bake_bounds: array<vec4<f32>>;
// The stale passes' counters: 0 pages appended to the bake, 1 pages marked
// invalid (listed in `invalid_list`), 2 pages read in the frame before
// (visible), 3 visible stale pages that did not fit the bake, 4 candidate
// pages (tested texel by texel), 5 texel jobs appended to the bake, 6 pages
// with any texel re-baked; 7 unused. Moving lights' ledger: 8 pages with
// texels re-baked, 9 texel jobs, 10 pages marked invalid, 11 pages read in
// the frame before, 12 of those with mixed texels, 13 of those passed whole
// (no mixed texel, within the guard), 14 texel jobs the ledger asked for
// (its budget's count); 15 bounce samples incomplete. Movers under a sky
// (M9 gate 3): 7 slots listed for a sky re-bake (`sky_list`), 16 pages
// read in the frame before that sky charges touched, 17 pages the sky
// ledger marked invalid.
@group(0) @binding(64) var<storage, read_write> mover_stats: array<atomic<u32>, 24>;
// Slots marked invalid this frame, for the CPU to evict.
@group(0) @binding(65) var<storage, read_write> invalid_list: array<u32>;
// Indirect dispatch sizes (x, y, z each): the texel test, the bake's texel
// jobs, the fill's pages.
@group(0) @binding(66) var<storage, read_write> dispatch_args: array<u32, 12>;
// Candidate pages: the slot, and its entry in `bake_list` (NONE: not read
// in the frame before, so marked invalid if any texel is stale).
@group(0) @binding(67) var<storage, read_write> candidates: array<vec2<u32>>;
// Moving lights (M9 gate 2). Each light's pose by frame, in a ring:
// `light_poses[(frame % ring) * lights + l]`.
@group(0) @binding(68) var<storage, read> light_poses: array<LightRect>;
// Per slot and moving light, LEDGER_WORDS words: 0 the frame the page was
// baked (its version), 1 the smallest slack of its texels that are not
// mixed (on a lattice, of every texel the light reaches): a texel's guard
// less the light's shift from the version's pose to the pose it was baked
// at (angle, f32 bits; while the light, seen from the page, stays within it
// of the version's pose, every such texel holds), 2 on a lattice (M9 gate
// 6) the largest κ of its mixed texels (f32 bits), 3 the frame its mixed
// texels were last baked, 4 flags (1: a mixed texel, 2: a texel baked after
// the version, 8: a footprint its lattice could not hold),
// then per texel 16 bits (two per word): the frames since the version it
// was baked (10 bits), 4 unused, and its state (2 bits: 2 clear, 3 mixed,
// 0 fixed: facing away, or never shown).
@group(0) @binding(69) var<storage, read_write> ledger: array<atomic<u32>>;
// Per light: its index among the moving lights, or NONE.
@group(0) @binding(70) var<storage, read> moving_index: array<u32>;
// Per slot and moving light: the certified bound on its texels' ratios, as
// of the last ledger pass (diagnostics).
@group(0) @binding(72) var<storage, read_write> ledger_bound: array<f32>;
// Bounce light (M10 gate 1). Per slot, a 3×3 grid of samples (at the page's
// texels 0, 4 and 8 along each axis), each with one entry per channel (a
// light's, then the sky's): bounce irradiance per unit radiance, RGB as
// half floats (r, g in x; b in y).
@group(0) @binding(73) var<storage, read_write> indirect: array<vec2<u32>>;
// Samples to bake this frame: entry in `bake_list`, and the sample (0–8).
@group(0) @binding(74) var<storage, read> indirect_jobs: array<vec2<u32>>;
// Slots whose samples met hits with no page resident (requested), for the
// CPU to re-bake later; count in mover_stats[15].
@group(0) @binding(75) var<storage, read_write> incomplete: array<u32>;
// The incomplete list's capacity (the CPU's `INCOMPLETE_CAP`), then pairs of
// a page's slot and the object whose lighting level its bake lacked (M14
// gate 4; `MISSING_CAP` of them).
const INCOMPLETE_CAP: u32 = 16384u;
// `slot_stale`'s mark for a page baked against an absent lighting level,
// with the frame in its low bits (M14 gate 4).
const LIT_MARK: u32 = 0x10000000u;
const MISSING_CAP: u32 = 16384u;
// Per object, its material (M10 gate 2): metallic and roughness; roughness
// below zero is Lambertian.
// Per object, two entries: metallic and roughness (f32 bits; roughness
// below zero: Lambertian), its emitted radiance as half floats (r, g; b),
// its normal map's layer plus one in w's high half; then its
// metallic-roughness texture's layer plus one (0: none).
@group(0) @binding(76) var<storage, read> materials: array<vec4<u32>>;

fn material_of(m: u32) -> vec2<f32> {
    let v = materials[2u * m];
    return vec2<f32>(bitcast<f32>(v.x), bitcast<f32>(v.y));
}

fn emission_of(m: u32) -> vec3<f32> {
    let v = materials[2u * m];
    return vec3<f32>(unpack2x16float(v.z), unpack2x16float(v.w).x);
}

// Object `m`'s emission at world point `p` on triangle `t`: its factor times
// its emissive texture (layer plus one `layer`, 0: none; M10 gate 14) read
// for a footprint of `footprint` world units.
fn emission_textured(m: u32, t: u32, p: vec3<f32>, factor: vec3<f32>, layer: u32, footprint: f32) -> vec3<f32> {
    if (layer == 0u || max(factor.x, max(factor.y, factor.z)) <= 0.0) { return factor; }
    let b = barycentric(m, t, p);
    let t0 = tri_tex[3u * t];
    let t1 = tri_tex[3u * t + 1u];
    let t2 = tri_tex[3u * t + 2u];
    let uv = t0 * b.x + t1 * b.y + t2 * b.z;
    let size = f32(textureDimensions(base_color).x);
    let v0 = tris[3u * t].xyz;
    let scale = object_scale(m);
    let world_area = 0.5 * length(cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0)) * scale * scale;
    let e1 = (t1 - t0) * size;
    let e2 = (t2 - t0) * size;
    let texel = sqrt(world_area / max(0.5 * abs(e1.x * e2.y - e1.y * e2.x), 1e-12));
    let lod = wrap_lod(m, SLOT_EMISSIVE, uv, log2(max(footprint / max(texel, 1e-12), 1e-6)));
    return factor * textureSampleLevel(base_color, base_sampler, wrap_uv(m, SLOT_EMISSIVE, uv, size, lod), i32(layer - 1u), lod).xyz;
}

// `emission_textured` with the layer from the materials buffer.
fn emission_at(m: u32, t: u32, p: vec3<f32>, footprint: f32) -> vec3<f32> {
    return emission_textured(m, t, p, emission_of(m), materials[2u * m + 1u].y, footprint);
}

// Object `m`'s material at world point `p` on triangle `t` (seen at
// `distance` along `dir`, geometric normal `n`): its factors times its
// metallic-roughness texture (glTF's: roughness in green, metallic in
// blue), read at the mip matching the pixel's footprint.
fn material_at(m: u32, t: u32, p: vec3<f32>, n: vec3<f32>, distance: f32, dir: vec3<f32>) -> vec2<f32> {
    let mat = material_of(m);
    let layer = materials[2u * m + 1u].x;
    if (layer == 0u || mat.y < 0.0) { return mat; }
    let b = barycentric(m, t, p);
    let t0 = tri_tex[3u * t];
    let t1 = tri_tex[3u * t + 1u];
    let t2 = tri_tex[3u * t + 2u];
    let uv = t0 * b.x + t1 * b.y + t2 * b.z;
    let v0 = tris[3u * t].xyz;
    let d1 = t1 - t0;
    let d2 = t2 - t0;
    let size = f32(textureDimensions(normal_maps).x);
    let scale = object_scale(m);
    let world_area = 0.5 * length(cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0)) * scale * scale;
    let texel_area = 0.5 * abs(d1.x * d2.y - d1.y * d2.x) * size * size;
    let texel = sqrt(world_area / max(texel_area, 1e-12));
    let footprint = select(1.0, camera.eye.w, camera.eye.w > 0.0) * sub_share;
    let pixel = footprint * distance * 2.0 * camera.params.z / camera.params.y / max(abs(dot(n, dir)), 0.2);
    let lod = wrap_lod(m, SLOT_MR, uv, log2(max(pixel / max(texel, 1e-12), 1e-6)));
    let s = textureSampleLevel(normal_maps, base_sampler, wrap_uv(m, SLOT_MR, uv, size, lod), i32(layer - 1u), lod);
    return vec2<f32>(mat.x * s.z, mat.y * s.y);
}
// Movers under a sky (M9 gate 3). Per slot: the sky ledger (the charges
// since the page's sky was baked), its threshold (ε × max(the page's
// smallest sky value, 0.05π)), and 1 once listed for a re-bake.
@group(0) @binding(77) var<storage, read_write> sky_ledger: array<vec4<f32>>;
// Slots listed for the CPU to re-bake their sky (top bit: a page on a
// moved object); count in mover_stats[7].
@group(0) @binding(78) var<storage, read_write> sky_list: array<u32>;

fn vc_hash(k: vec3<u32>) -> u32 {
    var h = (k.x * 0x9E3779B1u) ^ (k.y * 0x85EBCA77u) ^ (k.z * 0xC2B2AE3Du);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    return h;
}

// Whether the page `vc_find` last found shows its sky as a guess.
var<private> sky_guess: bool = false;

fn vc_find(k: vec3<u32>) -> u32 {
    let mask = vc.counts.x;
    var i = vc_hash(k) & mask;
    for (var probe = 0u; probe < 64u; probe++) {
        let e = page_table[i];
        if (e.x == EMPTY_KEY) { return NONE; }
        if (e.x == k.x && e.y == k.y && e.z == k.z) {
            // A stale page that was not re-baked is missing (M9). One the
            // ledger set aside (2) is marked wanted, so the next frame
            // re-bakes it.
            let stale = atomicLoad(&slot_stale[e.w]);
            // 3: a page on a moved object, its sky a guess (M9 gate 3).
            sky_guess = stale == 3u;
            if (stale != 0u && stale != 3u) {
                if (stale == 2u) { slot_used[e.w] = vc.counts.y; }
                return NONE;
            }
            return e.w;
        }
        i = (i + 1u) & mask;
    }
    return NONE;
}

// `vc_find` without stale marks, for bounce light (which movers and moving
// lights, the only markers, do not run with): two fewer buffers.
fn vc_find_any(k: vec3<u32>) -> u32 {
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

// A texel's channels (M5): irradiance per unit radiance from each light, then
// per unit sky radiance.
fn vc_channels() -> u32 {
    return vc.sky.z + 1u;
}

// Texel `i`'s channel `ch` in the pool.
fn pool_at(i: u32, ch: u32) -> u32 {
    return vc_channels() * i + ch;
}

fn vc_bilinear(slot: u32, r: PageRef, ch: u32) -> f32 {
    let o = slot * PAGE_TEXELS + r.base.y * PAGE_STRIDE + r.base.x;
    let a = mix(pool[pool_at(o, ch)], pool[pool_at(o + 1u, ch)], r.frac.x);
    let b = mix(pool[pool_at(o + PAGE_STRIDE, ch)], pool[pool_at(o + PAGE_STRIDE + 1u, ch)], r.frac.x);
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
    if (meshes[m].x == NONE) {
        // The object's lighting level is absent (M14 gate 4).
        lit_missing = m;
        return TexelSurface(vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), false);
    }
    let ch = vcharts[c];
    let uv = (vec2<f32>(texel) + vec2<f32>(0.5)) / vec2<f32>(n);
    let cell = min(vec2<u32>(uv * vec2<f32>(ch.grid)), ch.grid - vec2<u32>(1u));
    let list = cells[ch.first_cell + cell.y * ch.grid.x + cell.x];
    var best_t = NONE;
    var best_b = vec3<f32>(0.0);
    var best_score = -3.4e38;
    for (var q = list.x; q < list.x + list.y; q++) {
        let t = cell_tris[q];
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
    let interpolated = normals[3u * t].xyz * b.x + normals[3u * t + 1u].xyz * b.y + normals[3u * t + 2u].xyz * b.z;
    // A degenerate interpolated normal (zero or NaN): the face's (M10 gate 8).
    let face = cross(tris[3u * t + 1u].xyz - tris[3u * t].xyz, tris[3u * t + 2u].xyz - tris[3u * t].xyz);
    let nrm = select(face, interpolated, dot(interpolated, interpolated) > 1e-20);
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

// Marks slot `slot`'s page as baked against object `object`'s absent
// lighting level (M14 gate 4): it shows as missing (`vc_fill` keeps this
// frame's mark; a mark from an earlier frame is a slot's previous page's,
// cleared like any) and waits; the first texel to mark it lists the page with
// the object it lacked. The sun's, the sky's and bounce bakes mark (M14
// gate 8 for the last two).
fn lit_mark_page(slot: u32, object: u32) {
    atomicStore(&slot_stale[slot], LIT_MARK | (vc.counts.y & 0x0fffffffu));
    // Every absent object a page met is listed once a frame (M14 gate 8),
    // so a page needing several levels learns them in one round: pairs
    // seen this frame set a bit in a hash bitset (a collision only delays
    // a pair to a later round).
    let h = (slot * 0x9e3779b1u) ^ (object * 0x85ebca77u);
    let bit = h >> 16u;
    let old = atomicOr(&lit_seen[bit >> 5u], 1u << (bit & 31u));
    if ((old & (1u << (bit & 31u))) == 0u) {
        let q = atomicAdd(&mover_stats[20], 1u);
        if (q < MISSING_CAP) {
            incomplete[INCOMPLETE_CAP + 2u * q] = slot;
            incomplete[INCOMPLETE_CAP + 2u * q + 1u] = object;
        }
    }
}

// Pairs (page, absent object) listed this frame, as a hash bitset of 2^16
// bits (M14 gate 8; cleared each frame).
@group(0) @binding(83) var<storage, read_write> lit_seen: array<atomic<u32>>;

// One thread per listed texel. The list may include a texel just outside the
// chart's grid (the CPU's grid can differ by rounding); it is skipped here and
// filled by `vc_fill`.
@compute @workgroup_size(64)
fn vc_bake(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let j = gid.x;
    // No early return: every invocation follows its packet's walks.
    let total = bake_job_count();
    let listed = j < total;
    let job = bake_jobs[min(j, max(total, 1u) - 1u)];
    // The texel within the page, and the light (M5).
    let k = job.y & 0xffu;
    let l = min(job.y >> 8u, arrayLength(&lights) - 1u);
    let t = page_texel(job.x, k);
    lit_missing = NONE;
    let s = vc_surface(t.m, t.c, t.n, t.texel);
    let on = listed && t.inside && s.ok;
    let open = unshadowed_rect(s.apex, s.normal, lights[l]);
    // A moving rect light bakes on its page's lattice (M9 gate 6), in the
    // frame of the light's pose at the page's version. Both walks run for
    // every invocation (a packet can hold two lights' texels), each acting
    // only where it applies.
    let mj = moving_index[l];
    let lattice = mj != NONE && !is_sphere(lights[l]);
    var then = lights[l];
    if (lattice) {
        let version = atomicLoad(&ledger[ledger_base(t.e.w, mj)]);
        if (version != 0u && vc.counts.y - version < vc.ledger.y) {
            then = light_poses[(version % vc.ledger.y) * arrayLength(&lights) + l];
        }
    }
    // A texel whose bins' weights spread too far (the light near its
    // horizon) re-bakes whenever the light moves, so it gains nothing from
    // the lattice and takes the (smaller) cone around the light.
    var on_lattice = lattice;
    if (lattice && on && open > 0.0) {
        on_lattice = lattice_kappa(lattice_frame(s.apex, then), s.normal) <= LATTICE_KAPPA_MAX;
    }
    let lat = lattice_ratio(on && on_lattice && open > 0.0, s.apex, s.normal, lights[l], then);
    let lat_empty = last_empty;
    let lat_full = last_full;
    // A footprint the lattice cannot hold (never expected: the ledger
    // re-bakes such pages whole first) bakes as other lights do and has its
    // page re-baked whole.
    let overflow = on && on_lattice && open > 0.0 && lat < 0.0;
    let value = hit_irradiance_rect(on && (!on_lattice || overflow), 0u, s.apex, s.normal, lights[l]);
    // With `light_ratio`, the texel keeps its light's shadowed share of the
    // unshadowed light, and shading multiplies that by the unshadowed light
    // at the pixel, with the pixel's own normal (a ratio estimator). Where
    // the light is below the texel's horizon, the texel's own light is zero
    // whatever the ratio; a ratio of one leaves pixels nearby that face the
    // light unshadowed, as on a convex object's terminator.
    var ratio = select(select(1.0, clamp(value / open, 0.0, 1.0), open > 0.0), 0.0, !on);
    // A cone that nothing covers sees all its light, one wholly covered
    // none: exactly, whatever the estimators' rounding at the horizon (M9
    // gate 2: so a ratio cannot drift while its guard holds).
    if (on && open > 0.0 && last_empty) { ratio = 1.0; }
    if (on && open > 0.0 && last_full) { ratio = 0.0; }
    // A moving light's ratio within ε of 0 or 1 is that (M9 gate 2): a
    // nearly covered cone's estimate wavers by a few thousandths as its bins
    // realign, which no bound on the light's motion covers. Mixed texels'
    // bounds add ε for it, so they re-bake whenever the light moves. (Not on
    // a lattice, whose bins do not realign.)
    if (mj != NONE) {
        let snap = bitcast<f32>(vc.ledger.z);
        if (ratio < snap) { ratio = 0.0; }
        if (ratio > 1.0 - snap) { ratio = 1.0; }
    }
    var light_value = value;
    if (on_lattice && !overflow) {
        ratio = select(select(1.0, lat, open > 0.0), 0.0, !on);
        if (on && open > 0.0 && lat_empty) { ratio = 1.0; }
        if (on && open > 0.0 && lat_full) { ratio = 0.0; }
        light_value = ratio * open;
    }
    if (listed && t.inside) {
        pool[pool_at(t.e.w * PAGE_TEXELS + k, l)] = select(select(0.0, light_value, on), ratio, vc.sky.w == 1u);
    }
    // A texel that needed an absent lighting level (M14 gate 4): its page
    // shows as missing (`vc_fill` keeps this frame's mark; a mark from an
    // earlier frame is a slot's previous page's, cleared like any) and
    // waits; the first texel to mark it lists the page with the object it
    // lacked.
    if (listed && t.inside && lit_missing != NONE) { lit_mark_page(t.e.w, lit_missing); }
    // A moving light's ledger record for this texel (M9 gate 2).
    if (mj != NONE && listed && t.inside) {
        ledger_texel(on, t.e.w, mj, l, k, s.apex, s.normal, lights[l], open, then, lattice, overflow);
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
    let count = vc.sky.x + appended_sky();
    let j = gid.x;
    // No early return: every invocation follows its packet's walks.
    let listed = j < 6u * count;
    let f = min(j / max(count, 1u), 5u);
    let k = j % max(count, 1u);
    let sample = sky_samples[k];
    let t = page_texel(sample.x, sample.y);
    lit_missing = NONE;
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
    // A face's walk that met an absent lighting level (M14 gate 8).
    if (listed && t.inside && lit_missing != NONE) { lit_mark_page(t.e.w, lit_missing); }
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
// lookups), and marks a planned page used.
@compute @workgroup_size(64)
fn vc_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= bake_page_count()) { return; }
    let e = bake_list[p];
    // A page whose stale texels did not all fit stays marked invalid (M9).
    if (bake_bounds[2u * p].w != 0.0) { return; }
    // Baked without bounce samples (for light at hits): clear what a
    // previous page in the slot left (M10).
    if (bake_bounds[2u * p + 1u].w == 1.0 && vc.indirect.y == 1u) {
        for (var q = 0u; q < 9u * IND_CH; q++) { indirect[e.w * 9u * IND_CH + q] = vec2<u32>(0u); }
    }
    // The page's key and box, for the stale pass (M9); a bake makes it
    // current.
    slot_info[3u * e.w] = vec4<u32>(e.xyz, 1u);
    slot_info[3u * e.w + 1u] = bitcast<vec4<u32>>(bake_bounds[2u * p]);
    slot_info[3u * e.w + 2u] = bitcast<vec4<u32>>(bake_bounds[2u * p + 1u]);
    // hi.w 2: re-baked for a move, its sky still the guess; hi.w 3 (planned)
    // and 4 (appended): its sky alone re-baked, which clears a sky guess and
    // leaves any other mark to the light channels' bakes (M9 gate 3).
    let kind = bake_bounds[2u * p + 1u].w;
    // Planned with the sky later (M5 gate 9): with movers, its sky ledger
    // says "never baked" (stride 0) until its sky is baked alone (M9 gate 5).
    if (kind == 2.0 && p < vc.jobs.y && arrayLength(&sky_ledger) > 2u * e.w + 1u) {
        sky_ledger[2u * e.w] = vec4<f32>(0.0);
        sky_ledger[2u * e.w + 1u] = vec4<f32>(0.0);
    }
    if (kind == 5.0) {
        // A bounce retry (M14 gate 8): direct light was not baked again, so
        // its mark (a sky still the guess, or a level it lacked) stays.
    } else if (kind >= 3.0) {
        loop {
            let r = atomicCompareExchangeWeak(&slot_stale[e.w], 3u, 0u);
            if (r.exchanged || r.old_value != 3u) { break; }
        }
    } else if (atomicLoad(&slot_stale[e.w]) != (LIT_MARK | (vc.counts.y & 0x0fffffffu))) {
        // This frame's `LIT_MARK`: baked against an absent lighting level
        // (M14 gate 4), missing until baked again.
        atomicStore(&slot_stale[e.w], select(0u, 3u, kind == 2.0));
    }
    let c = e.y & 0xffffu;
    let mip = e.y >> 16u;
    let page = vec2<u32>(e.z & 0xffffu, e.z >> 16u);
    let n = vc_grid(c, e.x, mip);
    let o = e.w * PAGE_TEXELS;
    let in_grid = min(n - min(page * PAGE, n), vec2<u32>(PAGE_STRIDE));
    // hi.w 4: the stale pass appended the page's sky alone (M9 gate 3);
    // its other appended pages re-bake light channels only.
    let range = page_sky[p];
    let stride = range.y >> 24u;
    let sky_ch = vc.sky.z;
    let sky_baked = vc.sky.y == 1u && (p < vc.jobs.y || kind == 4.0) && (range.y & 0xffffffu) > 0u;
    let light_baked = kind != 3.0 && kind != 4.0;
    if (vc.sky.y == 0u) {
        for (var k = 0u; k < PAGE_TEXELS; k++) { pool[pool_at(o + k, sky_ch)] = 0.0; }
    } else if (sky_baked) {
        for (var q = range.x; q < range.x + (range.y & 0xffffffu); q++) {
            let k = sky_samples[q].y;
            if (k % PAGE_STRIDE < in_grid.x && k / PAGE_STRIDE < in_grid.y) {
                var total = 0.0;
                for (var f = 0u; f < 6u; f++) { total += sky_partial[6u * q + f]; }
                pool[pool_at(o + k, sky_ch)] = total;
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
                let a = mix(pool[pool_at(o + at.z * PAGE_STRIDE + at.x, sky_ch)], pool[pool_at(o + at.z * PAGE_STRIDE + at.y, sky_ch)], sx.z);
                let b = mix(pool[pool_at(o + at.w * PAGE_STRIDE + at.x, sky_ch)], pool[pool_at(o + at.w * PAGE_STRIDE + at.y, sky_ch)], sx.z);
                pool[pool_at(o + k, sky_ch)] = mix(a, b, sy.z);
            }
        }
    }
    for (var k = 0u; k < PAGE_TEXELS; k++) {
        let texel = page * PAGE + vec2<u32>(k % PAGE_STRIDE, k / PAGE_STRIDE);
        if (texel.x >= n.x || texel.y >= n.y) {
            let inside = min(texel, n - vec2<u32>(1u)) - page * PAGE;
            let nearest = o + inside.y * PAGE_STRIDE + inside.x;
            for (var ch = 0u; ch < vc_channels(); ch++) {
                if (select(light_baked, sky_baked || vc.sky.y == 0u, ch == sky_ch)) {
                    pool[pool_at(o + k, ch)] = pool[pool_at(nearest, ch)];
                }
            }
        }
    }
    // A page the CPU planned counts as used (its pixels asked for it); one
    // the movers' stale pass re-baked keeps its last use, so it is re-baked
    // again only if pixels read it (M9).
    if (p < vc.jobs.y) { slot_used[e.w] = vc.counts.y; }
    {
        // A sky baked: a fresh ledger, its threshold from the page's
        // smallest sky value, and its sample stride (M9 gate 3).
        if (sky_baked) {
            var least = 3.4e38;
            for (var k = 0u; k < PAGE_TEXELS; k++) {
                if (k % PAGE_STRIDE < in_grid.x && k / PAGE_STRIDE < in_grid.y) {
                    least = min(least, pool[pool_at(o + k, sky_ch)]);
                }
            }
            let eps = bitcast<f32>(vc.ledger.z);
            sky_ledger[2u * e.w] = vec4<f32>(0.0, 0.0, eps * max(least, 0.05 * 3.14159265), 0.0);
            sky_ledger[2u * e.w + 1u] = vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(stride));
        }
    }
}

// --- Movers in the view cache (M9 gate 1). ---
//
// A texel's channel for a light depends on a mover only if the mover's box
// meets the hull of the texel's apex and the light's bounding box: a light
// cone never leaves that hull. Each frame something moves:
// - `vc_stale` tests every resident page's box. A page on a moved object is
//   stale whole. A page whose box's hull meets a moved box is a candidate.
// - `vc_stale_texels` tests each candidate's texels at their apexes, light
//   by light.
// Stale texels of pages read in the frame before are appended to the
// frame's bake (only the lights they depend on; `vc_args` sizes the indirect
// dispatches). Other pages with a stale texel are marked invalid, so stale
// light is never shown.

// Pages and texel jobs the stale passes appended (within the capacities).
fn appended_pages() -> u32 {
    return min(atomicLoad(&mover_stats[0]), vc.movers.y);
}

fn appended_jobs() -> u32 {
    return min(atomicLoad(&mover_stats[5]), vc.movers.z);
}

// The bake's pages and texel jobs: the CPU's plan, then the appended ones.
// Sky samples the stale pass appended (M9 gate 3), after the CPU's.
fn appended_sky() -> u32 {
    if (vc.movers.x == 0u) { return 0u; }
    return min(atomicLoad(&mover_stats[18]), arrayLength(&sky_samples) - vc.sky.x);
}

fn bake_page_count() -> u32 {
    return vc.jobs.y + appended_pages();
}

fn bake_job_count() -> u32 {
    return vc.jobs.x + appended_jobs();
}

struct Box3 {
    lo: vec3<f32>,
    hi: vec3<f32>,
};

// The world box of object `m`'s object-space box (lo, hi), padded by the
// apex offset and rounding.
fn world_box(m: u32, lo: vec3<f32>, hi: vec3<f32>) -> Box3 {
    var b = Box3(lo, hi);
    let x = meshes[m].w;
    if (x != NONE) {
        let s = xforms[4u * x + 3u].x;
        let c = (lo + hi) * 0.5;
        let e = (hi - lo) * 0.5;
        let wc = rotate_to_world(x, c) * s + translation(x);
        let we = vec3<f32>(dot(abs(xforms[4u * x].xyz), e), dot(abs(xforms[4u * x + 1u].xyz), e),
                           dot(abs(xforms[4u * x + 2u].xyz), e)) * s;
        b = Box3(wc - we, wc + we);
    }
    let far = max(max(abs(b.lo), abs(b.hi)), vec3<f32>(1.0));
    let pad = 2.0 * surface_offset(length(far)) + 1e-5 * max(far.x, max(far.y, far.z));
    return Box3(b.lo - vec3<f32>(pad), b.hi + vec3<f32>(pad));
}

// A box around everything a light cone toward light `r` can reach near the
// light: its bounding sphere (a sphere light's cone window, the square
// around its disk, reaches about 1.42 radii from its center).
fn light_box(r: LightRect) -> Box3 {
    var c = light_center(r);
    var radius = 0.0;
    if (is_sphere(r)) {
        radius = 1.5 * r.edge_u.x;
    } else {
        radius = 0.5 * (length(r.edge_u.xyz) + length(r.edge_v.xyz));
    }
    let far = max(abs(c), vec3<f32>(1.0));
    radius = radius * 1.001 + 1e-5 * max(far.x, max(far.y, far.z)) + 1e-4;
    return Box3(c - vec3<f32>(radius), c + vec3<f32>(radius));
}

// Whether box `m` meets the convex hull of boxes `p` and `l`. The hull is
// the union over s in [0, 1] of the boxes (1 − s) p + s l, whose bounds are
// linear in s, so each axis bounds s to an interval.
fn hull_meets(p: Box3, l: Box3, m: Box3) -> bool {
    var s0 = 0.0;
    var s1 = 1.0;
    for (var a = 0u; a < 3u; a++) {
        // lo(s) = p.lo + s (l.lo − p.lo) <= m.hi
        let b0 = l.lo[a] - p.lo[a];
        let c0 = m.hi[a] - p.lo[a];
        if (b0 > 0.0) { s1 = min(s1, c0 / b0); } else if (b0 < 0.0) { s0 = max(s0, c0 / b0); } else if (c0 < 0.0) { return false; }
        // hi(s) = p.hi + s (l.hi − p.hi) >= m.lo
        let b1 = l.hi[a] - p.hi[a];
        let c1 = m.lo[a] - p.hi[a];
        if (b1 > 0.0) { s0 = max(s0, c1 / b1); } else if (b1 < 0.0) { s1 = min(s1, c1 / b1); } else if (c1 > 0.0) { return false; }
    }
    return s0 <= s1 + 1e-6;
}

fn boxes_meet(a: Box3, b: Box3) -> bool {
    return all(a.lo <= b.hi) && all(b.lo <= a.hi);
}

// Entry `i` of `moved` and the next as a box.
fn moved_box(i: u32) -> Box3 {
    return Box3(moved[i].xyz, moved[i + 1u].xyz);
}

// Whether a cone toward light `l` from anywhere in box `p` may meet a box
// moved this frame (old or new place).
fn may_be_shadowed(p: Box3, l: u32) -> bool {
    let lights_n = vc.sky.z;
    if (!boxes_meet(p, moved_box(2u * l))) { return false; }
    let lb = light_box(lights[l]);
    let first = 2u * lights_n + 2u;
    let stride = 4u + 4u * lights_n;
    for (var k = 0u; k < vc.movers.x; k++) {
        let e = first + k * stride;
        let r = e + 4u + 4u * l;
        if (boxes_meet(p, moved_box(r)) && hull_meets(p, lb, moved_box(e))) { return true; }
        if (boxes_meet(p, moved_box(r + 2u)) && hull_meets(p, lb, moved_box(e + 2u))) { return true; }
    }
    return false;
}

// Marks slot `s` invalid (once) and lists it for the CPU, which evicts it.
// A page shown (0) or hidden for a sky still the guess (3) or set aside by
// the ledger (2) is listed too: left resident with this mark, a later
// partial re-bake would clear it while texels made stale now stayed stale.
fn mark_invalid(s: u32) {
    let old = atomicExchange(&slot_stale[s], 1u);
    if (old == 0u || old == 2u || old == 3u) {
        let q = atomicAdd(&mover_stats[1], 1u);
        if (q < arrayLength(&invalid_list)) { invalid_list[q] = s; }
    }
}

// An entry in the frame's bake for slot `s`'s page (key `key`, box lo..hi),
// or NONE if the bake is full.
fn append_page(s: u32, key: vec3<u32>, lo: vec3<f32>, hi: vec3<f32>) -> u32 {
    let p = atomicAdd(&mover_stats[0], 1u);
    if (p >= vc.movers.y) {
        atomicAdd(&mover_stats[3], 1u);
        return NONE;
    }
    let entry = vc.jobs.y + p;
    bake_list[entry] = vec4<u32>(key, s);
    bake_bounds[2u * entry] = vec4<f32>(lo, 0.0);
    bake_bounds[2u * entry + 1u] = vec4<f32>(hi, 0.0);
    return entry;
}

// `count` texel jobs appended to the bake: the first's index, or NONE if
// they do not fit. The count never passes the capacity, so every job the
// bake runs was written.
fn append_jobs(count: u32) -> u32 {
    var old = atomicLoad(&mover_stats[5]);
    loop {
        if (old + count > vc.movers.z) { return NONE; }
        let r = atomicCompareExchangeWeak(&mover_stats[5], old, old + count);
        if (r.exchanged) { return vc.jobs.x + old; }
        old = r.old_value;
    }
    return NONE;
}

// A page appended to the bake whose stale texels did not all fit: the fill
// leaves it marked invalid (bounds' w: 1), and it is listed for eviction.
fn fail_entry(entry: u32, s: u32) {
    bake_bounds[2u * entry].w = 1.0;
    atomicStore(&slot_stale[s], 1u);
    let q = atomicAdd(&mover_stats[1], 1u);
    if (q < arrayLength(&invalid_list)) { invalid_list[q] = s; }
}

// Lists slot `s` for the CPU to re-bake its sky (once; `own`: a page on a
// moved object).
fn list_sky(s: u32, own: bool) {
    if (sky_ledger[2u * s].w != 0.0) { return; }
    sky_ledger[2u * s].w = 1.0;
    let q = atomicAdd(&mover_stats[7], 1u);
    if (q < arrayLength(&sky_list)) { sky_list[q] = s | select(0u, 0x80000000u, own); }
}

// The solid angle a sphere of radius `r` around `c` subtends from the
// nearest point of box `b` (π if it reaches the box): a bound on the sky
// light per unit radiance it can block there.
fn sphere_charge(b: Box3, c: vec3<f32>, r: f32) -> f32 {
    let gap = max(vec3<f32>(0.0), max(b.lo - c, c - b.hi));
    let d = length(gap);
    if (d <= r) { return 3.14159265; }
    let s = r / d;
    return min(2.0 * 3.14159265 * (1.0 - sqrt(max(1.0 - s * s, 0.0))), 3.14159265);
}

// Charges slot `s` (box `page`) for the objects moved this frame (M9
// gate 3). Since the page's bake, its sky can have changed only where a
// mover covered it then or covers it now, so the charge is, per mover moved
// since, its sphere's solid angle at the bake (fixed, x) plus now (current,
// y; ledger z is the threshold, w the listed flag). Movers 96 and on are
// charged both spheres every frame. Past the threshold, a page read in the
// frame before re-bakes its sky in this frame's bake; any other is marked
// invalid (and false returned).
fn charge_sky(s: u32, key: vec3<u32>, lo: vec3<f32>, hi: vec3<f32>, page: Box3, visible: bool) -> bool {
    let lights_n = vc.sky.z;
    let first = 2u * lights_n + 2u;
    let stride = 4u + 4u * lights_n;
    var ledger = sky_ledger[2u * s];
    var mask = bitcast<vec4<u32>>(sky_ledger[2u * s + 1u]);
    for (var k = 0u; k < vc.movers.x; k++) {
        let e = first + k * stride;
        let old = moved_box(e);
        let now = moved_box(e + 2u);
        let o = sphere_charge(page, (old.lo + old.hi) * 0.5, 0.5 * length(old.hi - old.lo));
        let n = sphere_charge(page, (now.lo + now.hi) * 0.5, 0.5 * length(now.hi - now.lo));
        let id = bitcast<u32>(moved[e + 1u].w);
        if (id < 96u) {
            let bit = 1u << (id % 32u);
            if ((mask[id / 32u] & bit) == 0u) {
                mask[id / 32u] |= bit;
                ledger.x += o;
                ledger.y += n;
            } else {
                ledger.y += n - o;
            }
        } else {
            ledger.x += o + n;
        }
    }
    sky_ledger[2u * s] = ledger;
    sky_ledger[2u * s + 1u] = bitcast<vec4<f32>>(mask);
    if (visible) { atomicAdd(&mover_stats[16], 1u); }
    let charge = ledger.x + max(ledger.y, 0.0);
    if (charge > ledger.z) {
        atomicAdd(&mover_stats[17], 1u);
        if (visible && sky_rebake(s, key, lo, hi)) { return true; }
        mark_invalid(s);
        return false;
    }
    return true;
}

// Appends slot `s`'s page (key `key`, box lo..hi) to the frame's bake for
// its sky alone, with its sky samples at the stride of its last bake; false
// if the bake or the sample list is full.
fn sky_rebake(s: u32, key: vec3<u32>, lo: vec3<f32>, hi: vec3<f32>) -> bool {
    let stride = bitcast<u32>(sky_ledger[2u * s + 1u].w);
    if (stride == 0u) { return false; }
    let mip = key.y >> 16u;
    let page = vec2<u32>(key.z & 0xffffu, key.z >> 16u);
    let n = vc_grid(key.y & 0xffffu, key.x, mip);
    let inside = min(n - min(page * PAGE, n), vec2<u32>(PAGE_STRIDE));
    let listed = min(inside + vec2<u32>(1u), vec2<u32>(PAGE_STRIDE));
    let count = ((listed.x + stride - 1u) / stride) * ((listed.y + stride - 1u) / stride);
    let first = atomicAdd(&mover_stats[18], count);
    if (first + count > arrayLength(&sky_samples) - vc.sky.x) { return false; }
    let entry = append_page(s, key, lo, hi);
    if (entry == NONE) { return false; }
    var q = vc.sky.x + first;
    for (var y = 0u; y < listed.y; y += stride) {
        for (var x = 0u; x < listed.x; x += stride) {
            sky_samples[q] = vec2<u32>(entry, y * PAGE_STRIDE + x);
            q++;
        }
    }
    page_sky[entry] = vec2<u32>(vc.sky.x + first, count | (stride << 24u));
    bake_bounds[2u * entry + 1u].w = 4.0;
    atomicAdd(&mover_stats[19], 1u);
    return true;
}

// One thread per slot (dispatched in rows of 65,535 workgroups).
@compute @workgroup_size(64)
fn vc_stale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x + gid.y * 65535u * 64u;
    if (s >= arrayLength(&slot_stale)) { return; }
    let info = slot_info[3u * s];
    if (info.w == 0u) { return; }
    // Marked invalid in an earlier frame and waiting to be evicted: its
    // texels made stale then must not be shown after a partial re-bake now.
    let mark = atomicLoad(&slot_stale[s]);
    if (mark == 1u) { return; }
    let m = info.x;
    let visible = slot_used[s] + vc.movers.w >= vc.counts.y;
    if (visible) { atomicAdd(&mover_stats[2], 1u); }
    let lo = bitcast<vec3<f32>>(slot_info[3u * s + 1u].xyz);
    let hi = bitcast<vec3<f32>>(slot_info[3u * s + 2u].xyz);
    let page = world_box(m, lo, hi);
    let lights_n = vc.sky.z;
    let first = 2u * lights_n + 2u;
    let stride = 4u + 4u * lights_n;
    // A page on a moved object (it lies within the moved boxes' union) is
    // stale whole.
    var own = false;
    if (boxes_meet(page, moved_box(2u * lights_n))) {
        for (var k = 0u; k < vc.movers.x; k++) {
            if (bitcast<u32>(moved[first + k * stride].w) == m) { own = true; }
        }
    }
    if (own) {
        if (!visible) {
            mark_invalid(s);
            return;
        }
        let entry = append_page(s, info.xyz, lo, hi);
        if (entry == NONE) {
            mark_invalid(s);
            return;
        }
        let j = append_jobs(PAGE_TEXELS * lights_n);
        if (j == NONE) {
            atomicAdd(&mover_stats[3], 1u);
            fail_entry(entry, s);
            return;
        }
        atomicAdd(&mover_stats[6], 1u);
        for (var l = 0u; l < lights_n; l++) {
            for (var k = 0u; k < PAGE_TEXELS; k++) {
                bake_jobs[j + l * PAGE_TEXELS + k] = vec2<u32>(entry, k | (l << 8u));
            }
        }
        // With a sky: the page moved with its object, so its sky shows the
        // guess until re-baked (listed; the CPU waits until it stops).
        if (vc.sky.y == 1u) {
            bake_bounds[2u * entry + 1u].w = 2.0;
            list_sky(s, true);
        }
        return;
    }
    // A page whose sky is still a guess (the sky later, M9 gate 5: stride 0
    // in its sky ledger) shows no sky to charge. If a mover may change its
    // light, it is marked invalid and baked anew: re-baking its light in
    // place would clear the guess's mark.
    let unknown_sky = vc.sky.y == 1u && bitcast<u32>(sky_ledger[2u * s + 1u].w) == 0u;
    if (vc.sky.y == 1u && !unknown_sky && !charge_sky(s, info.xyz, lo, hi, page, visible)) { return; }
    var candidate = false;
    for (var l = 0u; l < lights_n && !candidate; l++) {
        candidate = may_be_shadowed(page, l);
    }
    if (!candidate) { return; }
    // A page hidden for another reason (its sky the guess, set aside by the
    // ledger) is not re-baked in part: the fill would show it.
    if (unknown_sky || mark != 0u) {
        mark_invalid(s);
        return;
    }
    let c = atomicAdd(&mover_stats[4], 1u);
    if (c >= arrayLength(&candidates)) {
        mark_invalid(s);
        return;
    }
    var entry = NONE;
    if (visible) {
        entry = append_page(s, info.xyz, lo, hi);
        if (entry == NONE) {
            candidates[c] = vec2<u32>(NONE, NONE);
            mark_invalid(s);
            return;
        }
    }
    candidates[c] = vec2<u32>(s, entry);
}

// One thread per candidate page's texel (dispatched in rows of 65,535
// workgroups): its apex tested against each light's moved boxes. A stale
// texel of a visible page is re-baked for the lights it depends on; one of
// another page marks the page invalid.
@compute @workgroup_size(64)
fn vc_stale_texels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x + gid.y * 65535u * 64u;
    let c = j / PAGE_TEXELS;
    let k = j % PAGE_TEXELS;
    if (c >= min(atomicLoad(&mover_stats[4]), arrayLength(&candidates))) { return; }
    let cand = candidates[c];
    let s = cand.x;
    if (s == NONE) { return; }
    let info = slot_info[3u * s];
    let m = info.x;
    let chart = info.y & 0xffffu;
    let mip = info.y >> 16u;
    let page = vec2<u32>(info.z & 0xffffu, info.z >> 16u);
    let n = vc_grid(chart, m, mip);
    let texel = page * PAGE + vec2<u32>(k % PAGE_STRIDE, k / PAGE_STRIDE);
    if (texel.x >= n.x || texel.y >= n.y) { return; }
    let surface = vc_surface(m, chart, n, texel);
    if (!surface.ok) { return; }
    let far = max(abs(surface.apex), vec3<f32>(1.0));
    let pad = 1e-5 * max(far.x, max(far.y, far.z));
    let apex = Box3(surface.apex - vec3<f32>(pad), surface.apex + vec3<f32>(pad));
    let lights_n = vc.sky.z;
    for (var l = 0u; l < lights_n; l++) {
        if (!may_be_shadowed(apex, l)) { continue; }
        if (cand.y == NONE) {
            mark_invalid(s);
            return;
        }
        let at = append_jobs(1u);
        if (at == NONE) {
            // The bake is full: the page cannot be made current.
            atomicAdd(&mover_stats[3], 1u);
            fail_entry(cand.y, s);
            return;
        }
        bake_jobs[at] = vec2<u32>(cand.y, k | (l << 8u));
        // The first stale texel counts the page (the fill clears the mark).
        if (atomicExchange(&slot_stale[s], 1u) == 0u) { atomicAdd(&mover_stats[6], 1u); }
    }
}

// After `vc_stale`: the texel test's dispatch size.
@compute @workgroup_size(1)
fn vc_args() {
    let groups = (min(atomicLoad(&mover_stats[4]), arrayLength(&candidates)) * PAGE_TEXELS + 63u) / 64u;
    dispatch_args[0] = min(groups, 65535u);
    dispatch_args[1] = (groups + 65534u) / 65535u;
    dispatch_args[2] = 1u;
}

// After `vc_stale_texels`: the bake's and the fill's dispatch sizes.
@compute @workgroup_size(1)
fn vc_args_bake() {
    dispatch_args[3] = (bake_job_count() + 63u) / 64u;
    dispatch_args[4] = 1u;
    dispatch_args[5] = 1u;
    dispatch_args[6] = (bake_page_count() + 63u) / 64u;
    dispatch_args[7] = 1u;
    dispatch_args[8] = 1u;
    dispatch_args[9] = (6u * (vc.sky.x + appended_sky()) + 63u) / 64u;
    dispatch_args[10] = 1u;
    dispatch_args[11] = 1u;
}

// --- Moving lights through the error ledger (M9 gate 2). ---
//
// Texels keep a light's shadowed share of its unshadowed light; shading
// multiplies by the unshadowed light where the light is now. When a light
// moves, a cached ratio is stale by at most what the ledger certifies:
// - Each texel also traces a guard cone, three times the light's angular
//   radius around its center. If nothing meets it (clear), the ratio stays
//   exactly 1 while the light, seen from the texel, stays inside it: while
//   its shift stays within the guard (twice the light's radius). A texel
//   facing away from the light by at least twice its radius (fixed) cannot
//   change while the light's center stays below its horizon. Only texels
//   whose whole guard lies well above their horizon can be clear.
// - Otherwise a texel is mixed. Its bound, the band of width δ the light's
//   edge sweeps (at most 4π δ α + 4π δ² of solid angle) over its unshadowed
//   light, plus ε for snapping (a moving light's ratios within ε of 0 or 1
//   are stored as 0 or 1), exceeds ε once the light moves: mixed texels
//   re-bake, alone and without a guard cone, whenever it does.
// - Each page keeps its slack: the smallest guard of its clear and fixed
//   texels, less the light's shift since the page's version. Past a hashed
//   share (a half to all) of it the page re-bakes whole, with fresh guards.
// Pages not read in the frame before that need a re-bake are set aside:
// kept, but shown as missing until a pixel wants one, when the next frame
// re-bakes it.

const LEDGER_WORDS: u32 = 48u;
const GUARD: f32 = 3.0;
const MAX_OFFSET: u32 = 1023u;

fn ledger_base(slot: u32, j: u32) -> u32 {
    return (slot * vc.ledger.x + j) * LEDGER_WORDS;
}

struct GuardCone {
    c: Cone,
    // The light's angular radius around its center, seen from the apex.
    alpha: f32,
};

// The guard cone from `apex` toward light `r`: around the direction to its
// center, its window the square around a circle of GUARD times the light's
// angular radius, reaching beyond the scene (geometry behind the light
// counts too, so a light moving away stays covered).
fn guard_cone(apex: vec3<f32>, r: LightRect) -> GuardCone {
    var out: GuardCone;
    prefilter_bins = light.params.x;
    let to = light_center(r) - apex;
    let d = max(length(to), 1e-6);
    let forward = to / d;
    var alpha = 0.0;
    if (is_sphere(r)) {
        alpha = asin(min(r.edge_u.x / d, 1.0));
    } else {
        for (var k = 0u; k < 4u; k++) {
            let corner = r.corner.xyz + r.edge_u.xyz * f32(k & 1u) + r.edge_v.xyz * f32((k >> 1u) & 1u);
            alpha = max(alpha, acos(clamp(dot(forward, normalize(corner - apex)), -1.0, 1.0)));
        }
    }
    let wide = min(GUARD * alpha, 1.4);
    let t = tan(wide);
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(forward.x) > 0.9);
    let right = normalize(cross(helper, forward));
    out.c.apex = apex;
    out.c.right = right;
    out.c.up = cross(forward, right);
    out.c.forward = forward;
    out.c.window = vec4<f32>(-t, -t, t, t);
    out.c.far_normal = forward;
    out.c.far_offset = dot(forward, apex) + SKY_FAR;
    out.c = bound_cone(out.c, forward);
    out.alpha = alpha;
    return out;
}

// How far below `apex`'s horizon (normal `n`) light `r`'s center lies, as
// an angle (`faces_light` is false while it is positive).
fn below_horizon(apex: vec3<f32>, n: vec3<f32>, r: LightRect) -> f32 {
    let to = light_center(r) - apex;
    let d = max(length(to), 1e-6);
    var below = -dot(n, to);
    if (is_sphere(r)) { below -= r.edge_u.x; }
    return asin(clamp(below / d, -1.0, 1.0));
}

// Records texel `k` of slot `slot` for moving light `j` (bake pass): its
// state and frame, and the page's guard, unshadowed light and radius. On a
// lattice (M9 gate 6, `then` the light's pose at the page's version), every
// texel the light reaches keeps a slack: its bins stay within the guard's
// reach while the light, seen from it, moves less than the reach less the
// light's radius and a bin's diagonal; and a mixed texel's κ, the spread of
// its bins' weights over that reach, bounds how its ratio can move.
fn ledger_texel(on: bool, slot: u32, j: u32, light_index: u32, k: u32, apex: vec3<f32>, n: vec3<f32>, r: LightRect, open: f32, then: LightRect, lattice: bool, overflow: bool) {
    let base = ledger_base(slot, j);
    let version = atomicLoad(&ledger[base]);
    let offset = min(vc.counts.y - version, MAX_OFFSET);
    var state = 0u;
    var guard = 3.4e38;
    // On a lattice: the reach (the guard's half-angle), the light's radius
    // plus a bin's diagonal, and κ.
    var reach = 0.0;
    var margin = 3.4e38;
    var bin = 0.0;
    var kappa = 3.0e38;
    if (lattice && on && open > 0.0) {
        let f = lattice_frame(apex, then);
        if (f.ok) {
            reach = min(GUARD * f.alpha, 1.4);
            bin = length(f.beta);
            margin = f.alpha + bin;
            kappa = lattice_kappa(f, n);
        }
    }
    if (on && open > 0.0 && offset != 0u) {
        // Re-baked alone after the page's version: only mixed texels are
        // (the ledger re-bakes a page whole once its slack runs out), and
        // it stays mixed, with no guard cone traced.
        state = 3u;
        if (lattice) { guard = reach - margin; }
    } else if (on && open > 0.0) {
        let g = guard_cone(apex, r);
        var cov: array<f32, 64>;
        for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
        var full = 0u;
        scene_coverage_if(true, g.c, 0u, &cov, &full);
        var any = false;
        for (var b = 0u; b < BINS; b++) {
            if (cov[b] > 0.0) { any = true; }
        }
        // Clear if nothing meets the guard; otherwise mixed. (Dark, a guard
        // wholly covered, is not used: with one mesh covering the whole
        // guard and the light's own cone wholly covered at the bake, a
        // re-bake 0.37° later still saw 1% of the light in a development
        // test, so shadowed texels count as mixed.)
        state = select(2u, 3u, any);
        // Clear or dark only with the whole guard above the horizon: near
        // it, the estimates of grazing light waver as bins realign.
        if (-below_horizon(apex, n, r) <= max(2.0 * min(GUARD * g.alpha, 1.4), 0.1)) { state = 3u; }
        guard = min(GUARD * g.alpha, 1.4) - g.alpha;
        // On a lattice, the guard (traced around the light's center, as the
        // lattice's forward is) must also hold the bins the footprint meets.
        if (lattice) { guard = min(min(GUARD * g.alpha, 1.4), reach) - max(margin, g.alpha + bin); }
    } else if (on) {
        // Facing away: fixed while the light's center stays below the
        // horizon, if it lies well below (twice the light's radius);
        // otherwise mixed, so a page's slack is not a sliver.
        let margin = below_horizon(apex, n, r);
        let alpha = guard_cone(apex, r).alpha;
        if (margin >= 2.0 * alpha) {
            guard = margin;
        } else {
            state = 3u;
        }
    }
    if ((state != 3u || lattice) && guard < 3.0e38) {
        // Its slack: the guard less the light's shift since the version, as
        // seen from the apex.
        var slack = guard;
        if (offset != 0u) {
            let point = Box3(apex, apex);
            let then_v = light_poses[(version % vc.ledger.y) * arrayLength(&lights) + light_index];
            slack = guard - light_shift(then_v, r, point);
        }
        atomicMin(&ledger[base + 1u], bitcast<u32>(max(slack, 0.0)));
    }
    // A mixed texel whose bins' weights are unbounded (the light near its
    // horizon, or below it) is marked (code bit 1) to re-bake whenever the
    // light moves; the others' κ bounds the page's.
    var code = 0u;
    if (lattice && state == 3u) {
        if (on && open > 0.0 && kappa <= LATTICE_KAPPA_MAX) {
            atomicMax(&ledger[base + 2u], bitcast<u32>(kappa));
        } else {
            code = 4u;
        }
    }
    let shift = 16u * (k % 2u);
    let word = base + 5u + k / 2u;
    atomicAnd(&ledger[word], ~(0xffffu << shift));
    atomicOr(&ledger[word], ((offset << 6u) | code | state) << shift);
    // Page flags: 1 a mixed texel, 2 a texel baked after the version, 8 a
    // footprint its lattice could not hold (M9 gate 6).
    let flags = select(0u, 1u, state == 3u) | select(0u, 2u, offset != 0u) | select(0u, 8u, overflow);
    if (flags != 0u) { atomicOr(&ledger[base + 4u], flags); }
}

// Resets slot `slot`'s records for moving light `j`: version `frame`, no
// texel recorded.
fn ledger_reset(slot: u32, j: u32) {
    let base = ledger_base(slot, j);
    atomicStore(&ledger[base], vc.counts.y);
    atomicStore(&ledger[base + 1u], 0x7f7fffffu);
    atomicStore(&ledger[base + 2u], 0u);
    atomicStore(&ledger[base + 3u], vc.counts.y);
    atomicStore(&ledger[base + 4u], 0u);
    for (var w = 5u; w < LEDGER_WORDS; w++) { atomicStore(&ledger[base + w], 0u); }
    ledger_bound[slot * vc.ledger.x + j] = 0.0;
}

// One thread per page in the CPU's plan: fresh ledger records for it.
@compute @workgroup_size(64)
fn vc_ledger_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= vc.jobs.y) { return; }
    let slot = bake_list[p].w;
    for (var j = 0u; j < vc.ledger.x; j++) { ledger_reset(slot, j); }
}

// The largest angle a point of light pose `a` moves to reach pose `b`, seen
// from anywhere in box `page`: twice the arcsine of half the largest
// displacement over the distance to the light (4: unbounded).
fn light_shift(a: LightRect, b: LightRect, page: Box3) -> f32 {
    var moved = 0.0;
    if (is_sphere(a)) {
        moved = length(b.corner.xyz - a.corner.xyz) + abs(b.edge_u.x - a.edge_u.x);
    } else {
        for (var k = 0u; k < 4u; k++) {
            let u = f32(k & 1u);
            let v = f32((k >> 1u) & 1u);
            let pa = a.corner.xyz + a.edge_u.xyz * u + a.edge_v.xyz * v;
            let pb = b.corner.xyz + b.edge_u.xyz * u + b.edge_v.xyz * v;
            moved = max(moved, length(pb - pa));
        }
    }
    if (moved <= 0.0) { return 0.0; }
    let la = light_box(a);
    let lb = light_box(b);
    let span = Box3(min(la.lo, lb.lo), max(la.hi, lb.hi));
    let gap = max(vec3<f32>(0.0), max(page.lo - span.hi, span.lo - page.hi));
    let d = length(gap);
    if (moved >= 2.0 * d) { return 4.0; }
    return 2.0 * asin(moved / (2.0 * d)) * 1.0001;
}

// --- Moving rect lights on a lattice (M9 gate 6). ---
//
// Cones re-centred on a moving light waver by a few thousandths as their
// bins realign with it (gate 4), which no bound on the light's motion
// covers. A moving rect light's cones instead keep their bins on a lattice
// fixed in the frame of the light's pose at the page's version: forward
// toward the light's center then (from the texel), right along its first
// edge, bins a seventh of that footprint (plus a margin) on a side. The
// window is the lattice's 8×8 bins holding the footprint now, and each bin
// counts by the footprint's area in it times its projected solid angle per
// area at its center. A bin's coverage does not depend on where the window
// sits, so the ratio, an average of the bins' openness under the
// footprint's weighting, moves only as that weighting does
// (`lattice_bound`).

const LATTICE_MARGIN: f32 = 0.03;
// A mixed texel whose bins' weights spread more than this (the light near
// its horizon) re-bakes whenever the light moves instead of widening its
// page's bound.
const LATTICE_KAPPA_MAX: f32 = 8.0;

// Corner `k` of rect light `r` (k's bits: along edge_u, along edge_v).
fn rect_corner(r: LightRect, k: u32) -> vec3<f32> {
    return r.corner.xyz + r.edge_u.xyz * f32(k & 1u) + r.edge_v.xyz * f32((k >> 1u) & 1u);
}

struct LatticeFrame {
    forward: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
    // A bin's size in the window's plane (tangents), per axis.
    beta: vec2<f32>,
    // The light's largest angle from forward (radians).
    alpha: f32,
    ok: bool,
};

// The lattice of rect light pose `then` seen from `apex`.
fn lattice_frame(apex: vec3<f32>, then: LightRect) -> LatticeFrame {
    var f: LatticeFrame;
    f.ok = false;
    let to = light_center(then) - apex;
    let d = length(to);
    if (d <= 1e-6) { return f; }
    let forward = to / d;
    var right = then.edge_u.xyz - forward * dot(forward, then.edge_u.xyz);
    if (length(right) <= 1e-6 * length(then.edge_u.xyz)) {
        right = then.edge_v.xyz - forward * dot(forward, then.edge_v.xyz);
    }
    if (length(right) <= 0.0) { return f; }
    right = normalize(right);
    let up = cross(forward, right);
    var lo = vec2<f32>(3.4e38);
    var hi = vec2<f32>(-3.4e38);
    var alpha = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let p = rect_corner(then, k) - apex;
        let z = dot(forward, p);
        if (z <= 1e-6 * d) { return f; }
        let uv = vec2<f32>(dot(right, p), dot(up, p)) / z;
        lo = min(lo, uv);
        hi = max(hi, uv);
        alpha = max(alpha, atan2(length(cross(forward, p)), z));
    }
    f.forward = forward;
    f.right = right;
    f.up = up;
    f.beta = max(hi - lo, vec2<f32>(1e-12)) * ((1.0 + LATTICE_MARGIN) / 7.0);
    f.alpha = alpha * 1.0001;
    f.ok = true;
    return f;
}

// The spread κ (max over min) of the bins' weights for a texel with normal
// `n` on lattice `f`, over the guard's reach around forward: the cosine to
// the normal and the window's solid angle per area (3e38 where the reach
// meets the horizon).
fn lattice_kappa(f: LatticeFrame, n: vec3<f32>) -> f32 {
    if (!f.ok) { return 3.0e38; }
    let reach = min(GUARD * f.alpha, 1.4);
    let theta = atan2(length(cross(n, f.forward)), dot(n, f.forward));
    if (theta + reach >= 1.5706) { return 3.0e38; }
    let t = tan(reach);
    return cos(max(theta - reach, 0.0)) / cos(theta + reach) * pow(1.0 + t * t, 1.5) * 1.0001;
}

struct Lattice {
    c: Cone,
    // The light's footprint in bins, its corners in order around it.
    q: Poly2,
    ok: bool,
};

// The lattice cone from `apex` toward rect light `now` on the lattice of
// pose `then`: the 8×8 bins holding the footprint now, as deep as the
// light's plane. Not ok behind a one-sided light or where the footprint
// does not fit.
fn lattice_cone(apex: vec3<f32>, now: LightRect, then: LightRect) -> Lattice {
    var out: Lattice;
    out.ok = false;
    prefilter_bins = light.params.x;
    let f = lattice_frame(apex, then);
    if (!f.ok) { return out; }
    let n = normalize(cross(now.edge_u.xyz, now.edge_v.xyz));
    if (now.corner.w > 0.5 && dot(n, apex - now.corner.xyz) <= 0.0) { return out; }
    var pts: array<vec2<f32>, 4>;
    var lo = vec2<f32>(3.4e38);
    var hi = vec2<f32>(-3.4e38);
    for (var k = 0u; k < 4u; k++) {
        let p = rect_corner(now, k) - apex;
        let z = dot(f.forward, p);
        if (z <= 1e-6) { return out; }
        pts[k] = vec2<f32>(dot(f.right, p), dot(f.up, p)) / z;
        lo = min(lo, pts[k]);
        hi = max(hi, pts[k]);
    }
    let origin = floor(lo / f.beta) * f.beta;
    let end = origin + 8.0 * f.beta;
    if (any(hi > end)) { return out; }
    var far_n = n;
    if (dot(far_n, now.corner.xyz - apex) < 0.0) { far_n = -far_n; }
    let dist = dot(far_n, now.corner.xyz - apex);
    if (dist <= 1e-6) { return out; }
    out.c.apex = apex;
    out.c.right = f.right;
    out.c.up = f.up;
    out.c.forward = f.forward;
    out.c.window = vec4<f32>(origin, end);
    out.c.far_normal = far_n;
    out.c.far_offset = dot(far_n, apex) + dist * (1.0 - 1e-5);
    out.c = bound_cone(out.c, normalize(light_center(now) - apex));
    out.q.n = 4u;
    out.q.v[0] = (pts[0] - origin) / f.beta;
    out.q.v[1] = (pts[1] - origin) / f.beta;
    out.q.v[2] = (pts[3] - origin) / f.beta;
    out.q.v[3] = (pts[2] - origin) / f.beta;
    out.ok = true;
    return out;
}

// The footprint `q`'s area in each bin (in bins).
fn lattice_areas(q: Poly2) -> array<f32, 64> {
    var acc: array<f32, 68>;
    for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
    var span = vec2<u32>(68u, 0u);
    let g = f32(GRID);
    for (var i = 0u; i < q.n; i++) {
        let a = clamp(q.v[i], vec2<f32>(0.0), vec2<f32>(g));
        let e = clamp(q.v[(i + 1u) % q.n], vec2<f32>(0.0), vec2<f32>(g));
        draw_edge(&acc, &span, a, e, 1.0);
    }
    var m: array<f32, 64>;
    var running = 0.0;
    for (var b = 0u; b < BINS; b++) {
        running += acc[b];
        m[b] = min(abs(running), 1.0);
    }
    return m;
}

// The share of the footprint (its area `m` in each bin) that coverage
// `cov` leaves open: each bin weighted by the footprint's area in it and its
// projected solid angle per area at its center (0 below the horizon of
// `normal`).
fn lattice_share(c: Cone, normal: vec3<f32>, m: ptr<function, array<f32, 64>>, cov: ptr<function, array<f32, 64>>) -> f32 {
    let g = f32(GRID);
    var total = 0.0;
    var open = 0.0;
    for (var b = 0u; b < BINS; b++) {
        let area = (*m)[b];
        if (area <= 0.0) { continue; }
        let d = direction(c, (f32(b % GRID) + 0.5) / g, (f32(b / GRID) + 0.5) / g);
        let l2 = dot(d, d);
        let w = area * max(dot(normal, d), 0.0) / (l2 * l2);
        total += w;
        open += w * (1.0 - (*cov)[b]);
    }
    if (total <= 0.0) { return 1.0; }
    return open / total;
}

// A moving rect light's ratio at `apex` (normal `normal`) on the lattice
// of pose `then`, setting `last_empty` and `last_full` as
// `cone_irradiance_rect` does; −1 where the lattice cannot hold it (or
// `on` is false). The window's bins outside the footprint start covered:
// they do not count, and the walk can stop once the footprint is covered.
fn lattice_ratio(on: bool, apex: vec3<f32>, normal: vec3<f32>, now: LightRect, then: LightRect) -> f32 {
    let lc = lattice_cone(apex, now, then);
    let traced = on && lc.ok;
    var m = lattice_areas(lc.q);
    var cov: array<f32, 64>;
    var full = 0u;
    for (var b = 0u; b < BINS; b++) {
        cov[b] = select(0.0, 1.0, traced && m[b] <= 0.0);
        if (cov[b] >= 1.0) { full++; }
    }
    scene_coverage_if(traced, lc.c, 0u, &cov, &full);
    last_empty = true;
    last_full = true;
    for (var b = 0u; b < BINS; b++) {
        if (cov[b] > 0.0 && m[b] > 0.0) { last_empty = false; }
        if (cov[b] < 1.0) { last_full = false; }
    }
    if (!traced) { return -1.0; }
    return lattice_share(lc.c, normal, &m, &cov);
}

// Distances and directions from page box `page` to rect lights `a` and `b`
// (M9 gate 6): the nearest and farthest, the smallest cosine between a
// light's normal and a direction from the page to it, and the largest half
// diagonal.
struct LatticeReach {
    near: f32,
    far: f32,
    cos_min: f32,
    half_diagonal: f32,
};

fn lattice_reach(a: LightRect, b: LightRect, page: Box3) -> LatticeReach {
    var out: LatticeReach;
    let la = light_box(a);
    let lb = light_box(b);
    let span = Box3(min(la.lo, lb.lo), max(la.hi, lb.hi));
    let gap = max(vec3<f32>(0.0), max(page.lo - span.hi, span.lo - page.hi));
    out.near = length(gap);
    out.half_diagonal = 0.5 * max(length(a.edge_u.xyz + a.edge_v.xyz), length(b.edge_u.xyz + b.edge_v.xyz));
    out.half_diagonal = max(out.half_diagonal, 0.5 * max(length(a.edge_u.xyz - a.edge_v.xyz), length(b.edge_u.xyz - b.edge_v.xyz)));
    // From the centers: any point of the page lies within half its diagonal
    // of its center, any point of a light within its half diagonal of its.
    let pc = (page.lo + page.hi) * 0.5;
    let ph = 0.5 * length(page.hi - page.lo);
    let ca = light_center(a) - pc;
    let cb = light_center(b) - pc;
    out.far = max(length(ca), length(cb)) + ph + out.half_diagonal;
    let na = normalize(cross(a.edge_u.xyz, a.edge_v.xyz));
    let nb = normalize(cross(b.edge_u.xyz, b.edge_v.xyz));
    let cos_c = min(abs(dot(na, normalize(ca))), abs(dot(nb, normalize(cb))));
    // A direction between such points turns from the centers' by at most
    // the arcsine of their offsets over the nearest distance.
    if (out.near > 0.0) {
        out.cos_min = cos_c - asin(min((ph + out.half_diagonal) / out.near, 1.0));
    } else {
        out.cos_min = -1.0;
    }
    return out;
}

// The largest change of a ratio baked on a lattice with rect light pose `a`
// and re-baked with pose `b`, for texels in page box `page` whose bins'
// weights spread by at most `kappa` (max over min) over their guard's
// reach (M9 gate 6). Every point of the light moves at most δ seen from the
// page (`light_shift`), at most sec²ρ δ in the window's plane within the
// reach ρ, so the footprint A' lies within δ' of A, and |A' \ A| (and
// |A \ A'|) is at most P δ' + π δ'² (Steiner, A convex, P its perimeter).
// The bins' weighting is a density between g and κ g on the footprint, so
// the two weightings differ in total variation by at most
// ½(κ a/|A| + κ b/|A'| + κ max(a, b)/max(|A|, |A'|)) ≤ 1.5 κ S / Ω for
// S = P δ' + π δ'² and Ω ≤ either footprint's area (a plane area is at
// least its solid angle); a ratio, an average of bins' openness in [0, 1],
// moves by at most that.
fn lattice_bound(a: LightRect, b: LightRect, page: Box3, kappa: f32) -> f32 {
    let delta = light_shift(a, b, page);
    if (delta <= 0.0) { return 0.0; }
    if (delta >= 1.0 || kappa >= 1.0e30) { return 1.0; }
    let r = lattice_reach(a, b, page);
    if (r.cos_min <= 0.0 || r.near <= 2.0 * r.half_diagonal) { return 1.0; }
    let area = min(length(cross(a.edge_u.xyz, a.edge_v.xyz)), length(cross(b.edge_u.xyz, b.edge_v.xyz)));
    let omega = area * r.cos_min / (r.far * r.far);
    let lu = max(length(a.edge_u.xyz), length(b.edge_u.xyz));
    let lv = max(length(a.edge_v.xyz), length(b.edge_v.xyz));
    let perimeter = 4.0 * (asin(min(lu / (2.0 * r.near), 1.0)) + asin(min(lv / (2.0 * r.near), 1.0)));
    let rho = min(GUARD * asin(min(r.half_diagonal / r.near, 1.0)) * 1.0001, 1.4);
    let t = tan(rho);
    let sec2 = 1.0 + t * t;
    let moved = sec2 * delta;
    let s = sec2 * perimeter * moved + 3.14159265 * moved * moved;
    return min(1.5 * kappa * s / omega * 1.0001, 1.0);
}

// Whether a footprint on the lattice of rect light pose `a` (the page's
// version) may no longer fit its window with the light at pose `b`, for
// texels in page box `page` (M9 gate 6). The window holds a footprint up to
// (1 + LATTICE_MARGIN) times the version's extent along each axis. An
// extent grows by at most twice the largest difference between two of the
// footprint's points' moves in the plane: the light's center moves by t
// and its points by at most r more, so seen from the page the light's
// points' moves differ by at most 3 (|t|/d)(diagonal/d) + r/d (the
// direction map's Jacobian varies by 3/d² per unit length), then by sec²ρ
// in the plane, plus the plane map's own variation (at most 3 tanρ sec⁴ρ
// per radian) times the light's angular diameter and its move. An extent is
// at least the footprint's smallest width, at least its area over its
// diameter.
fn lattice_grew(a: LightRect, b: LightRect, page: Box3) -> bool {
    let r = lattice_reach(a, b, page);
    if (r.cos_min <= 0.0 || r.near <= 2.0 * r.half_diagonal) { return true; }
    let ca = light_center(a);
    let cb = light_center(b);
    let t = length(cb - ca);
    var rest = 0.0;
    for (var k = 0u; k < 4u; k++) {
        rest = max(rest, length((rect_corner(b, k) - cb) - (rect_corner(a, k) - ca)));
    }
    let diameter = 2.0 * asin(min(r.half_diagonal / r.near, 1.0));
    let rho = min(GUARD * 0.5 * diameter * 1.0001, 1.4);
    let tr = tan(rho);
    let sec2 = 1.0 + tr * tr;
    let delta = light_shift(a, b, page);
    let spread = sec2 * (3.0 * (t / r.near) * (2.0 * r.half_diagonal / r.near) + rest / r.near)
        + 3.0 * tr * sec2 * sec2 * diameter * delta;
    let area = min(length(cross(a.edge_u.xyz, a.edge_v.xyz)), length(cross(b.edge_u.xyz, b.edge_v.xyz)));
    let width = area * r.cos_min / (r.far * r.far) / (sec2 * diameter);
    return 2.0 * spread * 1.0001 > LATTICE_MARGIN * width;
}

// One thread per slot (rows of 65,535 workgroups): checks each moving
// light's ledger. A page read in the frame before re-bakes its texels whose
// bound exceeds ε (the whole page past its guard or its ring); another page
// with such a texel is marked invalid.
@compute @workgroup_size(64)
fn vc_ledger(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x + gid.y * 65535u * 64u;
    if (s >= arrayLength(&slot_stale)) { return; }
    let info = slot_info[3u * s];
    let set_aside = atomicLoad(&slot_stale[s]);
    if (info.w == 0u || set_aside == 1u) { return; }
    let visible = slot_used[s] + 1u >= vc.counts.y;
    // A page set aside is checked once wanted; nothing else changes unless a
    // light moved this frame.
    let lit = vc.ledger.w >> 31u == 1u;
    if (set_aside == 2u && !visible) { return; }
    if (set_aside == 0u && !lit) { return; }
    if (visible) { atomicAdd(&mover_stats[11], 1u); }
    let lo = bitcast<vec3<f32>>(slot_info[3u * s + 1u].xyz);
    let hi = bitcast<vec3<f32>>(slot_info[3u * s + 2u].xyz);
    let page = world_box(info.x, lo, hi);
    let ring = vc.ledger.y;
    let lights_n = arrayLength(&lights);
    var entry = NONE;
    var counted = false;
    for (var l = 0u; l < lights_n; l++) {
        let j = moving_index[l];
        if (j == NONE) { continue; }
        let base = ledger_base(s, j);
        let version = atomicLoad(&ledger[base]);
        if (version == 0u) { continue; }
        let now = lights[l];
        let slack = bitcast<f32>(atomicLoad(&ledger[base + 1u]));
        let page_flags = atomicLoad(&ledger[base + 4u]);
        let mixed_page = (page_flags & 1u) != 0u;
        let age = vc.counts.y - version;
        // A page set aside and now wanted is checked as any other: its
        // records were kept while it was hidden, so only what is stale
        // re-bakes (M9 gate 6; before, it re-baked whole).
        var whole = age + 1u >= min(ring, MAX_OFFSET);
        // On a lattice (M9 gate 6): a footprint its lattice could not hold,
        // or may not hold now, re-bakes the page whole.
        let lattice = !is_sphere(now);
        if (lattice && !whole && (page_flags & 8u) != 0u) {
            whole = true;
            if (visible) { atomicAdd(&mover_stats[22], 1u); }
        }
        var stale = array<u32, 3>(0u, 0u, 0u);
        var count = 0u;
        var bound = 0.0;
        // On a lattice: mixed texels held within their bound (M9 gate 6).
        var held = false;
        if (!whole) {
            let d0 = light_shift(light_poses[(version % ring) * lights_n + l], now, page);
            // Within the slack every texel that is not mixed holds; past it,
            // the page re-bakes whole, with fresh guards. Pages re-bake at a
            // share of their slack between a half and all of it, hashed, so
            // pages baked together do not all re-bake in one frame (early is
            // always allowed).
            let share = 0.5 + 0.5 * f32(vc_hash(vec3<u32>(s, version, 0x5bd1e995u)) & 0xffffu) / 65535.0;
            if (d0 > slack * share) {
                whole = true;
            } else if (!mixed_page) {
                if (visible) { atomicAdd(&mover_stats[13], 1u); }
                ledger_bound[s * vc.ledger.x + j] = 0.0;
                continue;
            } else {
                if (visible) { atomicAdd(&mover_stats[12], 1u); }
                // On a lattice, mixed texels hold while their bound since
                // their last bake stays within a hashed share of ε (a half to
                // all, so pages baked together spread their re-bakes); those
                // marked unbounded re-bake whenever the light moves.
                if (lattice) {
                    let since = atomicLoad(&ledger[base + 3u]);
                    let kappa = bitcast<f32>(atomicLoad(&ledger[base + 2u]));
                    if (vc.counts.y - since < ring) {
                        let b = lattice_bound(light_poses[(since % ring) * lights_n + l], now, page, kappa);
                        let held_share = 0.5 + 0.5 * f32(vc_hash(vec3<u32>(s, since, 0x27d4eb2du)) & 0xffffu) / 65535.0;
                        held = b <= bitcast<f32>(vc.ledger.z) * held_share;
                        // Held texels must still fit their lattice's window
                        // (clear ones need not: either cone sees nothing).
                        if (held && lattice_grew(light_poses[(version % ring) * lights_n + l], now, page)) {
                            held = false;
                            if (visible) { atomicAdd(&mover_stats[22], 1u); }
                        }
                        if (held) { bound = b; }
                    }
                }
                // Mixed texels: their bound, the band the light's edge swept
                // over the unshadowed light plus ε for snapping, exceeds ε
                // once the light has moved (this pass runs only then), so
                // they re-bake (on a lattice, past their share of ε).
                let want = select(3u, 7u, held);
                for (var k = 0u; k < PAGE_TEXELS; k++) {
                    let half = (atomicLoad(&ledger[base + 5u + k / 2u]) >> (16u * (k % 2u))) & 0xffffu;
                    if ((half & want) == want) {
                        stale[k / 32u] |= 1u << (k % 32u);
                        count++;
                    }
                }
                if (held && visible) {
                    atomicAdd(&mover_stats[21], 1u);
                    atomicAdd(&mover_stats[23], count);
                }
                whole = count == PAGE_TEXELS && !held;
            }
        }
        if (whole) { count = PAGE_TEXELS; }
        ledger_bound[s * vc.ledger.x + j] = bound;
        if (count == 0u) { continue; }
        if (!visible) {
            // Set aside, kept: re-baked once a pixel wants it.
            if (atomicExchange(&slot_stale[s], 2u) == 0u) { atomicAdd(&mover_stats[10], 1u); }
            return;
        }
        if (entry == NONE) {
            entry = append_page(s, info.xyz, lo, hi);
            if (entry == NONE) {
                atomicStore(&slot_stale[s], 2u);
                atomicAdd(&mover_stats[10], 1u);
                return;
            }
        }
        // Within the safety limit on the ledger's work; past it, the page
        // is set aside (its pixels fall back for a frame).
        var first = NONE;
        if (atomicAdd(&mover_stats[14], count) + count <= (vc.ledger.w & 0x7fffffffu)) {
            first = append_jobs(count);
        }
        if (first == NONE) {
            atomicAdd(&mover_stats[10], 1u);
            bake_bounds[2u * entry].w = 1.0;
            atomicStore(&slot_stale[s], 2u);
            return;
        }
        if (whole) {
            ledger_reset(s, j);
            for (var k = 0u; k < PAGE_TEXELS; k++) {
                bake_jobs[first + k] = vec2<u32>(entry, k | (l << 8u));
            }
        } else {
            var n = 0u;
            for (var k = 0u; k < PAGE_TEXELS; k++) {
                if ((stale[k / 32u] & (1u << (k % 32u))) != 0u) {
                    bake_jobs[first + n] = vec2<u32>(entry, k | (l << 8u));
                    n++;
                }
            }
            // Its mixed texels baked now (M9 gate 6), unless they held and
            // only the unbounded ones re-bake.
            if (!held) { atomicStore(&ledger[base + 3u], vc.counts.y); }
        }
        atomicAdd(&mover_stats[9], count);
        if (!counted) {
            counted = true;
            atomicAdd(&mover_stats[8], 1u);
        }
    }
    // A page set aside with nothing stale left to re-bake is shown again
    // (one with texels re-baked is shown by the fill).
    if (set_aside == 2u && entry == NONE) { atomicStore(&slot_stale[s], 0u); }
}

// --- Bounce light in the view cache (M10 gate 1). ---
//
// A page's 3×3 samples each gather a hemicube (192 bins, M2 gate 6's) of
// rays: a hit's light is its albedo over π times its direct light, read
// from the cache's page there at a mip matched to the bin's footprint. A
// hit with no page resident requests one and counts as missing; the CPU
// re-bakes such pages' samples once the pages arrive.

const IND_R: u32 = 8u;
// Request priorities' flags: a page wanted only for light at bounce or
// reflection hits (baked without bounce samples of its own, so requests do
// not cascade through the scene), and a resident page shown without its
// bounce samples (the CPU schedules them).
const SECONDARY: u32 = 0x80000000u;
const WANTS_SAMPLES: u32 = 0x40000000u;
const IND_BINS: u32 = 192u;
// Channels kept per sample: up to three lights, the sky, and emission
// (absolute radiance, M10 gate 11), which also holds the bounce of lights
// past the third at their radiance (M10 gate 21).
const IND_CH: u32 = 5u;
const IND_SKY: u32 = 3u;
const IND_GLOW: u32 = 4u;
// A bin's width, in radians, about: its footprint at a hit is this times
// the distance.
const IND_BIN_ANGLE: f32 = 0.25;

fn indirect_at(slot: u32, sample: u32, ch: u32) -> u32 {
    return (slot * 9u + sample) * IND_CH + ch;
}

// A hit's base color, its texture read at a footprint of `footprint` meters.
fn bounce_albedo(m: u32, t: u32, p: vec3<f32>, footprint: f32) -> vec3<f32> {
    let base = albedo[m].xyz * vertex_color(m, t, p);
    let layer = object_texture[m];
    if (layer == NONE) { return base; }
    let b = barycentric(m, t, p);
    let t0 = tri_tex[3u * t];
    let t1 = tri_tex[3u * t + 1u];
    let t2 = tri_tex[3u * t + 2u];
    let uv = t0 * b.x + t1 * b.y + t2 * b.z;
    let size = f32(textureDimensions(base_color).x);
    let v0 = tris[3u * t].xyz;
    let scale = object_scale(m);
    let world_area = 0.5 * length(cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0)) * scale * scale;
    let e1 = (t1 - t0) * size;
    let e2 = (t2 - t0) * size;
    let texel_area = 0.5 * abs(e1.x * e2.y - e1.y * e2.x);
    let texel = sqrt(world_area / max(texel_area, 1e-12));
    let lod = wrap_lod(m, SLOT_BASE, uv, log2(max(footprint / max(texel, 1e-12), 1e-6)));
    return base * textureSampleLevel(base_color, base_sampler, wrap_uv(m, SLOT_BASE, uv, size, lod), i32(layer), lod).xyz;
}

// Bin `b`'s corner `q` (0–3, around it) in the hemicube's frame (z along
// the normal), as `bin_local` lays bins out.
fn hemicube_corner(b: u32, q: u32) -> vec3<f32> {
    let g = f32(IND_R);
    let du = f32(q == 1u || q == 2u);
    let dv = f32(q >= 2u);
    if (b < IND_R * IND_R) {
        return vec3<f32>(-1.0 + 2.0 * (f32(b % IND_R) + du) / g, -1.0 + 2.0 * (f32(b / IND_R) + dv) / g, 1.0);
    }
    let half = IND_R * IND_R / 2u;
    let side = (b - IND_R * IND_R) / half;
    let k = (b - IND_R * IND_R) % half;
    let a = -1.0 + 2.0 * (f32(k % IND_R) + du) / g;
    let z = 2.0 * (f32(k / IND_R) + dv) / g;
    switch side {
        case 0u: { return vec3<f32>(1.0, a, z); }
        case 1u: { return vec3<f32>(-1.0, a, z); }
        case 2u: { return vec3<f32>(a, 1.0, z); }
        default: { return vec3<f32>(a, -1.0, z); }
    }
}

// Bin `b`'s projected solid angle (Lambert's edge integrals; the 192 sum
// to π), as `hemicube_weights` on the CPU.
fn hemicube_weight(b: u32) -> f32 {
    let n = vec3<f32>(0.0, 0.0, 1.0);
    var sum = 0.0;
    for (var q = 0u; q < 4u; q++) {
        sum += lambert_edge(n, hemicube_corner(b, q), hemicube_corner(b, (q + 1u) % 4u));
    }
    return abs(0.5 * sum);
}

var<workgroup> ind_partial: array<vec4<f32>, 320>;
var<workgroup> ind_missing: array<f32, 64>;
// More bounces (M10 gate 25): per lane, the fewest bounces its hits held
// and its share whose hits held too few.
var<workgroup> ind_gen_wg: array<f32, 64>;
var<workgroup> ind_short_wg: array<f32, 64>;

// One workgroup per sample (rows of 65,535 workgroups); each invocation
// traces three of the 192 bins, then a fixed tree reduction.
// The absent lighting level a bounce sample's rays met (M14 gate 8; NONE:
// none), the least object if several.
var<workgroup> ind_lit_missing: atomic<u32>;

// Marks a missing list's slot as a bounce sample's want: the page stays
// shown (M14 gate 8).
const BOUNCE_WANT: u32 = 0x80000000u;

// One bounce bin's light at a hit along `dir` from `apex` (M10): the hit
// page's direct light reflected, its emission, into `sum`; the bin's weight
// `w` if the page is missing (requested), else 0. Shared by the compute and
// the hardware bounce bakes (M14 gate 10).
// More bounces (M10 gate 25): the bounces the lane's hits hold (the
// fewest), and the hemisphere's share whose hits hold too few.
var<private> ind_gen: f32 = 15.0;
var<private> ind_short: f32 = 0.0;

// Adds the bounce light of the page at a hit (slot `slot`, lookup `r`), its
// 3×3 samples bilinearly, times `reflect`, to each channel's `sum`; returns
// the fewest bounces those samples hold (0 where never baked).
fn bounce_more(slot: u32, r: PageRef, reflect: vec3<f32>, sum: ptr<function, array<vec3<f32>, 5>>) -> f32 {
    let g = (vec2<f32>(r.base) + r.frac) * 0.25;
    let i0 = min(vec2<u32>(floor(g)), vec2<u32>(1u));
    let f = clamp(g - vec2<f32>(i0), vec2<f32>(0.0), vec2<f32>(1.0));
    var held = 15.0;
    for (var q = 0u; q < 4u; q++) {
        let ix = i0.x + (q & 1u);
        let iy = i0.y + (q >> 1u);
        let wq = select(1.0 - f.x, f.x, (q & 1u) == 1u) * select(1.0 - f.y, f.y, (q >> 1u) == 1u);
        held = min(held, unpack2x16float(indirect[indirect_at(slot, iy * 3u + ix, 0u)].y).y);
        for (var ch = 0u; ch < IND_CH; ch++) {
            let raw = indirect[indirect_at(slot, iy * 3u + ix, ch)];
            (*sum)[ch] += reflect * (wq * vec3<f32>(unpack2x16float(raw.x), unpack2x16float(raw.y).x));
        }
    }
    return held;
}

fn bounce_gather(apex: vec3<f32>, dir: vec3<f32>, hit: Hit, w: f32, sum: ptr<function, array<vec3<f32>, 5>>, bounces: u32) -> f32 {
    if (hit.mesh == NONE) { return 0.0; }
    let lights_n = min(arrayLength(&lights), IND_SKY);
    let sky_ch = vc.sky.z;
    let p = apex + dir * hit.distance;
    // A back face: the sample sees inside geometry; no light.
    let tv = 3u * hit.tri;
    let geometric = normal_to_world(hit.mesh, cross(tris[tv + 1u].xyz - tris[tv].xyz, tris[tv + 2u].xyz - tris[tv].xyz));
    if (dot(geometric, dir) > 0.0) { return 0.0; }
    // The hit's own emission (M10 gate 11): radiance over the bin's
    // projected solid angle; no page needed.
    if (vc.extra.x == 1u) {
        (*sum)[IND_GLOW] += bounce_emission_at(hit.mesh, hit.tri, p, hit.distance * IND_BIN_ANGLE) * w;
    }
    // The page at the hit, at the mip of the bin's footprint or the finest
    // coarser one resident.
    let footprint = hit.distance * IND_BIN_ANGLE;
    let hc = tri_vchart[hit.tri];
    let uv = chart_uv(hit.mesh, hit.tri, p);
    let top = vc_top_mip(hc, hit.mesh);
    let ratio = footprint * vc.params.x / object_scale(hit.mesh);
    var want = 0u;
    if (ratio > 1.0) { want = min(u32(round(log2(ratio))), top); }
    var slot = NONE;
    var r: PageRef;
    for (var q = want; q <= top; q++) {
        r = vc_page(hit.mesh, hc, uv, q);
        slot = vc_find_any(r.key);
        if (slot != NONE) { break; }
    }
    // With more bounces (M10 gate 25), the page hit should hold one fewer.
    let wanted_bits = (bounces - 1u) << 24u;
    if (slot == NONE) {
        vc_request(vc_page(hit.mesh, hc, uv, want).key, 1u | SECONDARY | wanted_bits);
        return w;
    }
    let hn = shading_normal(hit.mesh, hit.tri, p, -dir);
    let hp = p + hit.normal * surface_offset(length(p));
    let reflect = bounce_albedo(hit.mesh, hit.tri, p, footprint) * (w / 3.14159265);
    for (var l = 0u; l < lights_n; l++) {
        var direct = vc_bilinear(slot, r, l);
        if (vc.sky.w == 1u && direct > 0.0) { direct *= unshadowed_rect(hp, hn, lights[l]); }
        (*sum)[l] += reflect * direct;
    }
    // Lights past the third (M10 gate 21): at their radiance, in the
    // emission channel (absolute radiance).
    for (var l = lights_n; l < arrayLength(&lights); l++) {
        var direct = vc_bilinear(slot, r, l);
        if (vc.sky.w == 1u && direct > 0.0) { direct *= unshadowed_rect(hp, hn, lights[l]); }
        (*sum)[IND_GLOW] += reflect * direct * lights[l].radiance.xyz;
    }
    (*sum)[IND_SKY] += reflect * vc_bilinear(slot, r, sky_ch);
    if (bounces > 1u) {
        let held = bounce_more(slot, r, reflect, sum);
        ind_gen = min(ind_gen, held);
        if (held < f32(bounces - 1u)) {
            ind_short += w;
            vc_request(r.key, 1u | SECONDARY | WANTS_SAMPLES | wanted_bits);
        }
    }
    return 0.0;
}

// A bounce sample's job: its page, texel and surface.
struct BounceJob {
    e: vec4<u32>,
    k: u32,
    listed: bool,
    surf: TexelSurface,
    // The bounces its samples should hold (M10 gate 25; at least 1).
    bounces: u32,
}

fn bounce_job(wg: vec3<u32>) -> BounceJob {
    let j = wg.x + wg.y * 65535u;
    let listed = j < vc.indirect.x;
    let job = indirect_jobs[min(j, max(vc.indirect.x, 1u) - 1u)];
    let e = bake_list[job.x];
    let m = e.x;
    let c = e.y & 0xffffu;
    let mip = e.y >> 16u;
    let page = vec2<u32>(e.z & 0xffffu, e.z >> 16u);
    let n = vc_grid(c, m, mip);
    let k = job.y & 0xffu;
    let texel = min(page * PAGE + vec2<u32>(4u * (k % 3u), 4u * (k / 3u)), n - vec2<u32>(1u));
    ind_gen = 15.0;
    ind_short = 0.0;
    return BounceJob(e, k, listed, vc_surface(m, c, n, texel), max(job.y >> 8u, 1u));
}

// Sums the workgroup's bins and stores the sample (lane 0), listing its page
// for a retry if more than a hundredth of the hemisphere was missing, and
// any absent lighting level a ray met (M14 gate 8).
fn bounce_finish(li: u32, job: BounceJob, sum: array<vec3<f32>, 5>, missing: f32) {
    for (var ch = 0u; ch < IND_CH; ch++) { ind_partial[li * IND_CH + ch] = vec4<f32>(sum[ch], 0.0); }
    ind_missing[li] = missing;
    ind_gen_wg[li] = ind_gen;
    ind_short_wg[li] = ind_short;
    workgroupBarrier();
    for (var h = 32u; h > 0u; h >>= 1u) {
        if (li < h) {
            for (var ch = 0u; ch < IND_CH; ch++) {
                ind_partial[li * IND_CH + ch] += ind_partial[(li + h) * IND_CH + ch];
            }
            ind_missing[li] += ind_missing[li + h];
            ind_gen_wg[li] = min(ind_gen_wg[li], ind_gen_wg[li + h]);
            ind_short_wg[li] += ind_short_wg[li + h];
        }
        workgroupBarrier();
    }
    if (li == 0u && job.listed) {
        // The bounces the sample holds (y's second half; 0: never baked):
        // one, or with more bounces one more than the fewest its hits held
        // (M10 gate 25).
        var held = 1.0;
        if (job.bounces > 1u) { held = min(1.0 + ind_gen_wg[0], 15.0); }
        for (var ch = 0u; ch < IND_CH; ch++) {
            let v = ind_partial[ch].xyz;
            indirect[indirect_at(job.e.w, job.k, ch)] = vec2<u32>(pack2x16float(v.xy), pack2x16float(vec2<f32>(v.z, held)));
        }
        // Retried while hits were missing, or held too few bounces.
        if (ind_missing[0] + ind_short_wg[0] > 0.01 * 3.14159265) {
            let q = atomicAdd(&mover_stats[15], 1u);
            if (q < INCOMPLETE_CAP) { incomplete[q] = job.e.w; }
        }
        // An absent lighting level a ray met: listed with the page flagged
        // as wanting it only (the page stays shown; its samples retry).
        let absent = atomicLoad(&ind_lit_missing);
        if (absent != NONE) {
            let q = atomicAdd(&mover_stats[20], 1u);
            if (q < MISSING_CAP) {
                incomplete[INCOMPLETE_CAP + 2u * q] = job.e.w | BOUNCE_WANT;
                incomplete[INCOMPLETE_CAP + 2u * q + 1u] = absent;
            }
        }
    }
}

@compute @workgroup_size(64)
fn vc_bake_indirect(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    if (li == 0u) { atomicStore(&ind_lit_missing, NONE); }
    workgroupBarrier();
    let job = bounce_job(wg);
    var sum = array<vec3<f32>, 5>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var missing = 0.0;
    if (job.listed && job.surf.ok) {
        for (var b = li; b < IND_BINS; b += 64u) {
            let dir = normalize(bin_world(job.surf.normal, bin_local(b, IND_R)));
            lit_missing = NONE;
            let hit = primary_hit(job.surf.apex, dir);
            // A ray that met an absent lighting level (M14 gate 8): its bin
            // is missing, as with an absent page at the hit; the level is
            // wanted and the sample retried.
            if (lit_missing != NONE) {
                atomicMin(&ind_lit_missing, lit_missing);
                missing += hemicube_weight(b);
                continue;
            }
            missing += bounce_gather(job.surf.apex, dir, hit, hemicube_weight(b), &sum, job.bounces);
        }
    }
    bounce_finish(li, job, sum, missing);
}

// Bounce irradiance (RGB, scaled by each channel's radiance) at a cached
// lookup `r` in slot `slot`: the page's 3×3 samples, bilinearly.
fn indirect_light(slot: u32, r: PageRef) -> vec3<f32> {
    let g = (vec2<f32>(r.base) + r.frac) * 0.25;
    let i0 = min(vec2<u32>(floor(g)), vec2<u32>(1u));
    let f = clamp(g - vec2<f32>(i0), vec2<f32>(0.0), vec2<f32>(1.0));
    let lights_n = min(arrayLength(&lights), IND_SKY);
    var total = vec3<f32>(0.0);
    for (var ch = 0u; ch < IND_CH; ch++) {
        var radiance = light.sky.xyz;
        if (ch == IND_GLOW) {
            // Emission is stored as absolute radiance.
            radiance = vec3<f32>(1.0);
        } else if (ch < lights_n) {
            radiance = lights[ch].radiance.xyz;
        } else if (ch < IND_SKY) {
            continue;
        }
        var v = vec3<f32>(0.0);
        for (var q = 0u; q < 4u; q++) {
            let ix = i0.x + (q & 1u);
            let iy = i0.y + (q >> 1u);
            let wq = select(1.0 - f.x, f.x, (q & 1u) == 1u) * select(1.0 - f.y, f.y, (q >> 1u) == 1u);
            let raw = indirect[indirect_at(slot, iy * 3u + ix, ch)];
            v += wq * vec3<f32>(unpack2x16float(raw.x), unpack2x16float(raw.y).x);
        }
        total += radiance * v;
    }
    return total;
}

// Extra samples (M7 gate 4's silhouette rays, M10 gate 16's layers behind
// blended surfaces): the records `hw_edges` lists, counted in the stats'
// last counter (so not with COUNT_WORK), five words each (see `hw_edges`).
@group(0) @binding(81) var<storage, read_write> edge_hits: array<vec4<u32>>;

// After `vc_shade`: each listed edge pixel averages its four quarter-rays,
// each shaded from the cache with textures read for half a pixel, as
// `vc_shade`'s compute path does. One thread per list entry (over the
// screen's grid).
// Before `vc_shade_edges` (M10 gate 19): its dispatch, a workgroup per 64
// records of the extra samples' list, in the dispatch arguments' last three
// words.
@compute @workgroup_size(1)
fn vc_args_edges() {
    let count = min(atomicLoad(&vc_stats[15]), arrayLength(&edge_hits) / 5u);
    dispatch_args[9] = (count + 63u) / 64u;
    dispatch_args[10] = 1u;
    dispatch_args[11] = 1u;
}

@compute @workgroup_size(8, 8)
fn vc_shade_edges(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    // Dispatched for the list alone (`vc_args_edges`, M10 gate 19).
    let i = wid.x * 64u + li;
    let count = min(atomicLoad(&vc_stats[15]), arrayLength(&edge_hits) / 5u);
    if (i >= count) { return; }
    let head = edge_hits[5u * i + 4u];
    let pixel = head.x;
    let at = vec3<u32>(pixel % w, pixel / w, 0u);
    // The pixel's own sample, then each listed one, by its weight.
    var sum = colors[pixel].xyz * bitcast<f32>(head.y);
    for (var j = 0u; j < min(head.z, 4u); j++) {
        let e = edge_hits[5u * i + j];
        let q = (head.w >> (4u * j)) & 0xfu;
        // A quarter's ray reads textures for half a pixel; a layer behind a
        // blended surface, the pixel's centre, for the whole pixel.
        var sd = camera_dir_at(at, 0.0, 0.0);
        sub_share = 1.0;
        if (q != 0xfu) {
            sd = camera_dir_at(at, (f32(q & 1u) - 0.5) * 0.5, (f32(q >> 1u) - 0.5) * 0.5);
            sub_share = 0.5;
        }
        let weight = bitcast<f32>(e.w);
        if (e.x == NONE) {
            sum += sky_seen(sd) * weight;
        } else {
            sum += vc_hit_color(sd, hit_from(e.xyz, sd), false) * weight;
        }
    }
    colors[pixel] = vec4<f32>(sum, 1.0);
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
    vc_request_priority = 0u;
    vc_fell_back = false;
    if (in_bounds) {
        let dir = camera_dir(gid);
        let pixel = gid.y * w + gid.x;
        let hit = recorded_hit(pixel, dir);
        // A ray that misses everything sees the sky.
        var color = sky_seen(dir);
        if (vis[pixel].w != 0u) {
            let l = lights[vis[pixel].w - 1u];
            color = l.radiance.xyz * spot_falloff(l, -dir);
        } else if (hit.mesh != NONE) {
            color = vc_hit_color(dir, hit, is_sample);
        }
        // At a silhouette, four rays through the pixel's quarters (M7 gate
        // 3), textures read for half a pixel.
        // (w 2: hardware silhouette rays, shaded by `vc_shade_edges`.)
        if (vc.indirect.w == 1u && is_edge(gid, w, h)) {
            // Counted in the last work counter (unused without
            // COUNT_WORK).
            if (!COUNT_WORK) { atomicAdd(&vc_stats[15], 1u); }
            let key = vc_request_key;
            let priority = vc_request_priority;
            sub_share = 0.5;
            var sum = vec3<f32>(0.0);
            for (var q = 0u; q < 4u; q++) {
                let sd = camera_dir_at(gid, (f32(q & 1u) - 0.5) * 0.5, (f32(q >> 1u) - 0.5) * 0.5);
                let sh = primary_hit(camera.eye.xyz, sd);
                if (sh.mesh == NONE) {
                    sum += sky_seen(sd);
                } else {
                    sum += vc_hit_color(sd, sh, false);
                }
            }
            sub_share = 1.0;
            vc_request_key = key;
            vc_request_priority = priority;
            color = sum * 0.25;
        }
        colors[pixel] = vec4<f32>(color, 1.0);
    }
    if (is_sample && vc_request_priority > 0u) {
        vc_request(vc_request_key, vc_request_priority);
    }
    if (vc_fell_back) { atomicAdd(&fallback_count, 1u); }
    workgroupBarrier();
    if (li == 0u) {
        let count = atomicLoad(&fallback_count);
        if (count > 0u) { atomicAdd(&vc_stats[0], count); }
    }
}

// What shading from the cache asks of the frame: the page a sample pixel
// requests (key and priority; 0: none), and whether a pixel fell back to
// unshadowed light.
var<private> vc_request_key: vec3<u32> = vec3<u32>(0u);
var<private> vc_request_priority: u32 = 0u;
var<private> vc_fell_back: bool = false;

// The color of surface `hit` seen along camera ray `dir`, lit from the
// cache: the page of the wanted mip, or failing that the finest coarser one
// resident, or failing that unshadowed light. `sample`: the pixel samples
// this frame (its request and statistics count).
fn vc_hit_color(dir: vec3<f32>, hit: Hit, sample: bool) -> vec3<f32> {
    let surface = camera.eye.xyz + dir * hit.distance;
    let c = tri_vchart[hit.tri];
    let uv = chart_uv(hit.mesh, hit.tri, surface);
    let wanted = min(vc_mip(hit.distance), vc_top_mip(c, hit.mesh));
    // Irradiance per unit radiance from the found page: lights, then sky.
    var found_slot = NONE;
    var found_ref: PageRef;
    var found = false;
    var found_guess = false;
    var mip = wanted;
    // The keys of the pages one to four mips finer than `mip` (the last
    // missing ones).
    var finer = vec3<u32>(0u);
    var finer2 = vec3<u32>(0u);
    var finer3 = vec3<u32>(0u);
    var finer4 = vec3<u32>(0u);
    loop {
        let r = vc_page(hit.mesh, c, uv, mip);
        let slot = vc_find(r.key);
        if (slot != NONE) {
            slot_used[slot] = vc.counts.y;
            found_guess = sky_guess;
            found_slot = slot;
            found_ref = r;
            found = true;
            if (mip > wanted) {
                // Up to `refine_levels` finer (M4 gate 12): fewer rounds
                // from coarse to fine.
                let step = u32(vc.params.w);
                vc_request_key = finer;
                if (step >= 2u && mip >= wanted + 2u) { vc_request_key = finer2; }
                if (step >= 3u && mip >= wanted + 3u) { vc_request_key = finer3; }
                if (step >= 4u && mip >= wanted + 4u) { vc_request_key = finer4; }
                vc_request_priority = 32u - min(mip - wanted, 30u);
            }
            break;
        }
        finer4 = finer3;
        finer3 = finer2;
        finer2 = finer;
        finer = r.key;
        if (r.coarsest || mip >= 30u) { break; }
        mip += 1u;
    }
    var e = vec3<f32>(0.0);
    let count = arrayLength(&lights);
    let point = surface + hit.normal * surface_offset(hit.distance);
    let normal = mapped_normal(hit.mesh, hit.tri, surface, shading_normal(hit.mesh, hit.tri, surface, -dir), hit.distance, dir);
    let base = surface_albedo(hit.mesh, hit.tri, surface, hit.distance, dir, hit.normal);
    // A material (M10 gate 2): roughness below zero is Lambertian.
    let mat = material_at(hit.mesh, hit.tri, surface, hit.normal, hit.distance, dir);
    let pbr = mat.y >= 0.0;
    // Emission (with its texture, M10 gate 14) for the pixel's footprint.
    let glow_footprint = select(1.0, camera.eye.w, camera.eye.w > 0.0) * sub_share * hit.distance * 2.0 * camera.params.z / camera.params.y / max(abs(dot(hit.normal, dir)), 0.2);
    let glow = emission_at(hit.mesh, hit.tri, surface, glow_footprint);
    var spec = vec3<f32>(0.0);
    if (found) {
        // Each light's cached shadowed share, times its unshadowed
        // light here.
        for (var l = 0u; l < count; l++) {
            if (vc.sky.w == 1u) {
                // Shadowed pixels need no unshadowed light.
                let share = vc_bilinear(found_slot, found_ref, l);
                if (share > 0.0) {
                    let open = unshadowed_rect(point, normal, lights[l]);
                    e += lights[l].radiance.xyz * share * open;
                    if (pbr) {
                        spec += lights[l].radiance.xyz * share * open * specular_light(point, normal, -dir, lights[l], base, mat);
                    }
                }
            } else {
                e += lights[l].radiance.xyz * vc_bilinear(found_slot, found_ref, l);
            }
        }
        if (found_guess) {
            vc_fell_back = true;
            e += sky_open(normal);
        } else {
            e += sky_scale(normal) * vc_bilinear(found_slot, found_ref, count);
        }
        if (vc.indirect.y == 1u) {
            e += indirect_light(found_slot, found_ref);
            // A page shown without bounce samples (baked for a hit): ask for
            // them.
            if (unpack2x16float(indirect[indirect_at(found_slot, 0u, 0u)].y).y == 0.0) {
                vc_request(vc_page(hit.mesh, c, uv, mip).key, 1u | WANTS_SAMPLES);
            }
        }
    } else {
        vc_fell_back = true;
        vc_request_key = finer;
        vc_request_priority = 1u;
        // Unshadowed light from each light.
        for (var l = 0u; l < count; l++) {
            e += lights[l].radiance.xyz * unshadowed_rect(point, normal, lights[l]);
        }
        // The sky above the horizon, with the ground below it: a
        // guess until a page arrives.
        e += sky_open(normal);
    }
    if (sample) {
        atomicAdd(&vc_stats[2], 1u);
        if (found && mip == wanted) { atomicAdd(&vc_stats[3], 1u); }
    }
    if (!pbr) { return base * e / 3.14159265 + glow; }
    // Metallic-roughness: the specular layer's directional albedo (the
    // split sum), glossy reflections along the mirror direction, and the
    // diffuse part less what the specular layer reflects.
    let nv = max(dot(normal, -dir), 1e-4);
    let ab = env_brdf(nv, mat.y);
    let f0 = mix(vec3<f32>(0.04), base, mat.x);
    let albedo_spec = f0 * ab.x + ab.y;
    if (vc.indirect.z == 0u && max(albedo_spec.x, max(albedo_spec.y, albedo_spec.z)) > 0.005) {
        // A shading or mapped normal can turn the mirror direction below the
        // surface itself; it is kept just above it instead (as Cycles'
        // ensure_valid_reflection does).
        // A shading or mapped normal can turn the mirror direction below the
        // surface itself; it is kept just above it instead (as Cycles'
        // ensure_valid_reflection does).
        var refl = reflect(dir, normal);
        let up = dot(refl, hit.normal);
        if (up < 0.01) { refl = normalize(refl + hit.normal * (0.01 - up)); }
        spec += albedo_spec * reflected_light(point, refl, mat.y);
    }
    let diffuse = base * (1.0 - mat.x) * (1.0 - (0.04 * ab.x + ab.y));
    return diffuse * e / 3.14159265 + spec + glow;
}

// --- Environment light (M5 gate 11): sky radiance varying with direction. ---

fn sh9(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y));
}

fn environment_on() -> bool {
    return light.objects.z == 1u;
}

// The environment's irradiance on a normal `n` (its radiance above the
// horizon convolved with the clamped cosine: bands weighted π, 2π/3, π/4),
// unrolled (M5 gate 12).
fn environment_irradiance(n: vec3<f32>) -> vec3<f32> {
    let s = light.sky_sh;
    var e = s[0].xyz * (3.14159265 * 0.282095);
    e += (s[1].xyz * n.y + s[2].xyz * n.z + s[3].xyz * n.x) * (2.0943951 * 0.488603);
    e += (s[4].xyz * (n.x * n.y) + s[5].xyz * (n.y * n.z) + s[7].xyz * (n.x * n.z)) * (0.78539816 * 1.092548);
    e += s[6].xyz * (0.78539816 * 0.315392 * (3.0 * n.z * n.z - 1.0));
    e += s[8].xyz * (0.78539816 * 0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// The sky's irradiance on normal `n` with nothing in the way: the
// environment's, or the uniform sky's above the horizon.
fn sky_open(n: vec3<f32>) -> vec3<f32> {
    if (environment_on()) { return environment_irradiance(n); }
    return light.sky.xyz * (3.14159265 * 0.5 * (1.0 + n.z));
}

// What a cached sky value (irradiance per unit radiance of a uniform sky)
// is multiplied by at normal `n`: the uniform sky's radiance, or the
// environment's open irradiance over the uniform sky's (its ratio).
fn sky_scale(n: vec3<f32>) -> vec3<f32> {
    if (environment_on()) { return environment_irradiance(n) / max(3.14159265 * 0.5 * (1.0 + n.z), 1e-3); }
    return light.sky.xyz;
}

// The sky's radiance seen along `d`: the uniform sky's, or the
// environment's: from its image where the base color array holds it (M5
// gate 12; light.objects.w its layer, sky_sh[0].w its peak), else through
// its spherical harmonics (none below the horizon).
fn sky_seen(d: vec3<f32>) -> vec3<f32> {
    if (!environment_on()) { return light.sky.xyz; }
    if (light.objects.w != NONE) {
        let u = (3.14159265 - atan2(d.y, d.x)) / 6.2831853;
        let v = 0.5 - asin(clamp(d.z, -1.0, 1.0)) / 3.14159265;
        return textureSampleLevel(base_color, base_sampler, vec2<f32>(u, v), i32(light.objects.w), 0.0).xyz * light.sky_sh[0].w;
    }
    if (d.z < 0.0) { return vec3<f32>(0.0); }
    let y = sh9(d);
    var l = vec3<f32>(0.0);
    for (var k = 0u; k < 9u; k++) { l += light.sky_sh[k].xyz * y[k]; }
    return max(l, vec3<f32>(0.0));
}

// --- Materials (M10 gate 2): metallic-roughness, glossy reflections. ---

fn ggx_d(nh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

// Smith's height-correlated visibility, G / (4 n·l n·v).
fn smith_visibility(nv: f32, nl: f32, a: f32) -> f32 {
    let a2 = a * a;
    let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-7);
}

fn fresnel(f0: vec3<f32>, vh: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - vh, 0.0, 1.0), 5.0);
}

// The split sum's scale and bias on F0 (Karis's analytic fit, 2014).
fn env_brdf(nv: f32, roughness: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// GGX specular per unit of the light's unshadowed irradiance at `p`
// (normal `n`, toward the viewer `v`): toward light `r`'s point nearest the
// mirror direction (Karis 2013's representative point), D and V at the
// material's roughness, times the energy normalization (α/α′)² with α′
// widened by the light's angular radius. (Evaluating D and V at α′ as well
// lowered a small light's peak twice: M10 gate 10.)
fn specular_light(p: vec3<f32>, n: vec3<f32>, v: vec3<f32>, r: LightRect, base: vec3<f32>, mat: vec2<f32>) -> vec3<f32> {
    let refl = reflect(-v, n);
    var aim = light_center(r);
    var radius = 0.0;
    if (is_sphere(r)) {
        let c = r.corner.xyz - p;
        let along = c - dot(c, refl) * refl;
        aim = r.corner.xyz - along * clamp(r.edge_u.x / max(length(along), 1e-9), 0.0, 1.0);
        radius = r.edge_u.x;
    } else {
        let nr = normalize(cross(r.edge_u.xyz, r.edge_v.xyz));
        let denom = dot(nr, refl);
        var q = light_center(r);
        if (abs(denom) > 1e-6) {
            let t = dot(nr, r.corner.xyz - p) / denom;
            if (t > 0.0) { q = p + refl * t; }
        }
        let d = q - r.corner.xyz;
        let s = clamp(dot(d, r.edge_u.xyz) / dot(r.edge_u.xyz, r.edge_u.xyz), 0.0, 1.0);
        let u = clamp(dot(d, r.edge_v.xyz) / dot(r.edge_v.xyz, r.edge_v.xyz), 0.0, 1.0);
        aim = r.corner.xyz + r.edge_u.xyz * s + r.edge_v.xyz * u;
        radius = 0.5 * length(r.edge_u.xyz + r.edge_v.xyz);
    }
    let to = aim - p;
    let dist = max(length(to), 1e-6);
    let l = to / dist;
    let nl = dot(n, l);
    let nv = max(dot(n, v), 1e-4);
    if (nl <= 0.0) { return vec3<f32>(0.0); }
    let h = normalize(v + l);
    let a = max(mat.y * mat.y, 1e-3);
    let wide = min(a + 0.5 * radius / max(length(light_center(r) - p), 1e-6), 1.0);
    let norm = (a / wide) * (a / wide);
    let f0 = mix(vec3<f32>(0.04), base, mat.x);
    return ggx_d(max(dot(n, h), 0.0), a) * smith_visibility(nv, nl, a) * fresnel(f0, max(dot(v, h), 0.0)) * norm;
}

// The light arriving at `p` along `refl`: what the cache holds for the
// surface it meets (its diffuse light, read at the mip matching the lobe's
// width there), or the sky.
fn reflected_light(p: vec3<f32>, refl: vec3<f32>, roughness: f32) -> vec3<f32> {
    let hit = primary_hit(p, refl);
    if (hit.mesh == NONE) { return sky_seen(refl); }
    let q = p + refl * hit.distance;
    // The lobe's width there: about α radians.
    let footprint = hit.distance * max(roughness * roughness, 0.002);
    let hc = tri_vchart[hit.tri];
    let uv = chart_uv(hit.mesh, hit.tri, q);
    let top = vc_top_mip(hc, hit.mesh);
    let ratio = footprint * vc.params.x / object_scale(hit.mesh);
    var want = 0u;
    if (ratio > 1.0) { want = min(u32(round(log2(ratio))), top); }
    var slot = NONE;
    var r: PageRef;
    for (var m = want; m <= top; m++) {
        r = vc_page(hit.mesh, hc, uv, m);
        slot = vc_find(r.key);
        if (slot != NONE) { break; }
    }
    let hn = shading_normal(hit.mesh, hit.tri, q, -refl);
    let hp = q + hit.normal * surface_offset(length(q));
    let count = arrayLength(&lights);
    var e = vec3<f32>(0.0);
    if (slot == NONE) {
        vc_request(vc_page(hit.mesh, hc, uv, want).key, 1u | SECONDARY);
        for (var l = 0u; l < count; l++) {
            e += lights[l].radiance.xyz * unshadowed_rect(hp, hn, lights[l]);
        }
        e += sky_open(hn);
    } else {
        // A page seen in reflections is in use: kept, not evicted.
        slot_used[slot] = vc.counts.y;
        for (var l = 0u; l < count; l++) {
            var direct = vc_bilinear(slot, r, l);
            if (vc.sky.w == 1u && direct > 0.0) { direct *= unshadowed_rect(hp, hn, lights[l]); }
            e += lights[l].radiance.xyz * direct;
        }
        if (sky_guess) {
            e += sky_open(hn);
        } else {
            e += sky_scale(hn) * vc_bilinear(slot, r, count);
        }
        if (vc.indirect.y == 1u) { e += indirect_light(slot, r); }
    }
    let hm = material_of(hit.mesh);
    let metal = select(0.0, hm.x, hm.y >= 0.0);
    return bounce_albedo(hit.mesh, hit.tri, q, footprint) * (1.0 - metal) * e / 3.14159265 + emission_at(hit.mesh, hit.tri, q, footprint);
}


// --- Pixel cones (M7): primary visibility through each pixel's square. ---
//
// A pixel's cone is its square seen from the eye, split into 8×8 bins of
// equal image-plane area. Front to back through the object hierarchy, each
// object it meets becomes a fragment: its mean coverage of the bins (its
// front-facing triangles summed per bin, prefiltered below a bin, as light
// cones do), a mask of the bins it covers at least half of, and its depth
// along a ray through its coverage. Fragments composite by mask, nearest
// first, so nearer ones hide farther ones bin by bin; each visible one is
// shaded once, at a ray through its visible bins; what none covers sees the
// sky.

const PIXEL_FRAGMENTS: u32 = 4u;

struct Fragment {
    mesh: u32,
    // Bins it covers at least half of (bit b of x for b < 32, of y above).
    mask: vec2<u32>,
    // Mean coverage over the 64 bins.
    cover: f32,
    // Distance along the ray through `st` to its surface.
    depth: f32,
    // Its coverage's centroid in the pixel, in [0, 1]².
    st: vec2<f32>,
};

// The pixel's fragments, nearest first.
var<private> frags: array<Fragment, PIXEL_FRAGMENTS>;
var<private> frag_count: u32 = 0u;
// Diagnostics (profile stage 9: `render_cones` writes them as the pixel's
// color): object nodes visited, meshes entered, triangle-hierarchy nodes
// tested.
var<private> pc_work: vec3<u32> = vec3<u32>(0u);


// Bins the last `mesh_fragment` covered wholly.
var<private> frag_solid: vec2<u32> = vec2<u32>(0u);

fn bit_count(m: vec2<u32>) -> u32 {
    return countOneBits(m.x) + countOneBits(m.y);
}

// The cone through pixel `gid`'s square.
fn pixel_cone(gid: vec3<u32>) -> Cone {
    let aspect = camera.params.x / camera.params.y;
    let tz = camera.params.z;
    var c: Cone;
    c.apex = camera.eye.xyz;
    c.right = camera.right.xyz;
    c.up = camera.up.xyz;
    c.forward = camera.forward.xyz;
    c.window = vec4<f32>(
        (f32(gid.x) / camera.params.x * 2.0 - 1.0) * tz * aspect,
        (1.0 - f32(gid.y + 1u) / camera.params.y * 2.0) * tz,
        (f32(gid.x + 1u) / camera.params.x * 2.0 - 1.0) * tz * aspect,
        (1.0 - f32(gid.y) / camera.params.y * 2.0) * tz,
    );
    c.far_normal = camera.forward.xyz;
    c.far_offset = 1e30;
    c.horizon = vec3<f32>(0.0);
    c.bound = vec4<f32>(0.0, 0.0, 0.0, -2.0);
    c.bound_sin = 0.0;
    return c;
}

// The center of bin `b` in the pixel, in [0, 1]².
fn bin_center(b: u32) -> vec2<f32> {
    return (vec2<f32>(f32(b % GRID), f32(b / GRID)) + 0.5) / f32(GRID);
}

// The mean center of the bins in mask `m` (the pixel's center if empty).
fn mask_centroid(m: vec2<u32>) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    var n = 0.0;
    for (var b = 0u; b < BINS; b++) {
        let word = select(m.x, m.y, b >= 32u);
        if (((word >> (b & 31u)) & 1u) != 0u) {
            sum += bin_center(b);
            n += 1.0;
        }
    }
    if (n == 0.0) { return vec2<f32>(0.5); }
    return sum / n;
}

// Mesh `m`'s coverage of pixel cone `world_cone` as a fragment (depth not
// yet set); `frag_solid` gets the bins it covers wholly. Triangle hierarchy
// traversal uses `stack` from `base` up.
fn mesh_fragment(world_cone: Cone, m: u32, base: u32) -> Fragment {
    var f = Fragment(m, vec2<u32>(0u), 0.0, 3.4e38, vec2<f32>(0.5));
    frag_solid = vec2<u32>(0u);
    let c = cone_to_object(world_cone, m);
    let range = meshes[m];
    // An alpha-blended object covers by its opacity (M10 gate 16).
    tri_weight = 1.0;
    if (range.w != NONE) { tri_weight = 1.0 - xforms[4u * range.w + 3u].z; }
    var acc: array<f32, 68>;
    for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
    var span = vec2<u32>(68u, 0u);
    var touched = false;
    if (range.z != NONE) {
        var sp = base + 1u;
        stack_set(base, range.z);
        while (sp > base) {
            sp--;
            let n = stack_get(sp);
            let node = blas[n];
            pc_work.z++;
            let test = node_cone_test(c, &acc, &span, n, node.lo.xyz, node.hi.xyz);
            if (test == 2u) { touched = true; }
            if (test != 3u) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    if (cover_triangle(c, &acc, &span, blas_items[k])) { touched = true; }
                }
            } else {
                stack_set(sp, node.info.x);
                stack_set(sp + 1u, node.info.x + 1u);
                sp += 2u;
            }
        }
    } else {
        for (var t = range.x; t < range.x + range.y; t++) {
            if (cover_triangle(c, &acc, &span, t)) { touched = true; }
        }
    }
    if (!touched) { return f; }
    var running = 0.0;
    var sum = 0.0;
    var centroid = vec2<f32>(0.0);
    for (var b = span.x; b < min(span.y, BINS); b++) {
        running += acc[b];
        var covered = min(abs(running), 1.0);
        if (covered >= 1.0 - 1e-5) { covered = 1.0; }
        if (covered <= 1e-6) { continue; }
        sum += covered;
        centroid += covered * bin_center(b);
        let bit = 1u << (b & 31u);
        if (covered >= 0.5) {
            if (b < 32u) { f.mask.x |= bit; } else { f.mask.y |= bit; }
        }
        if (covered >= 1.0) {
            if (b < 32u) { frag_solid.x |= bit; } else { frag_solid.y |= bit; }
        }
    }
    f.cover = sum / f32(BINS);
    if (sum > 0.0) { f.st = centroid / sum; }
    return f;
}

const MISS: Hit = Hit(3.4e38, 0xffffffffu, 0xffffffffu, vec3<f32>(0.0));

// The nearest hit on mesh `m` along the ray through `st` in pixel cone `c`,
// its normal facing the ray. Triangle traversal uses `stack` from `base` up.
fn fragment_hit(c: Cone, m: u32, st: vec2<f32>, base: u32) -> Hit {
    let dir = normalize(direction(c, st.x, st.y));
    var hit = nearest_in_mesh(c.apex, dir, m, MISS, base);
    if (hit.mesh == m && dot(hit.normal, dir) > 0.0) { hit.normal = -hit.normal; }
    return hit;
}

// Inserts `f` among the pixel's fragments, nearest first. When they are
// full, the one with the least coverage (`f` or a kept one) is dropped.
fn insert_fragment(f: Fragment) {
    if (frag_count == PIXEL_FRAGMENTS) {
        var least = PIXEL_FRAGMENTS;
        var least_cover = f.cover;
        for (var k = 0u; k < PIXEL_FRAGMENTS; k++) {
            if (frags[k].cover < least_cover) {
                least = k;
                least_cover = frags[k].cover;
            }
        }
        if (least == PIXEL_FRAGMENTS) { return; }
        for (var k = least; k + 1u < PIXEL_FRAGMENTS; k++) { frags[k] = frags[k + 1u]; }
        frag_count--;
    }
    var at = frag_count;
    while (at > 0u && frags[at - 1u].depth > f.depth) {
        frags[at] = frags[at - 1u];
        at--;
    }
    frags[at] = f;
    frag_count++;
}

// The least distance along the cone's forward axis of a box's corners: no
// point of the box is nearer along any of the cone's rays.
fn box_near(c: Cone, lo: vec3<f32>, hi: vec3<f32>) -> f32 {
    var near = 3.4e38;
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u), select(lo.z, hi.z, (k & 4u) != 0u));
        near = min(near, dot(c.forward, corner - c.apex));
    }
    return near;
}

// Collects pixel cone `c`'s fragments: objects nearest first, until the
// fragments cover every bin wholly; then only objects that may lie in front
// of the farthest fragment.
fn pixel_fragments(c: Cone) {
    // Prefiltered as light cones are: nodes within the threshold (in bins)
    // splat their coverage.
    prefilter_bins = light.params.x;
    frag_count = 0u;
    var solid = vec2<u32>(0u);
    var limit = 3.4e38;
    var sp = 1u;
    stack_set(0, 0u);
    while (sp > 0u) {
        sp--;
        let node = nodes[stack_get(sp)];
        pc_work.x++;
        if (!may_touch_box(c, node.lo.xyz, node.hi.xyz)) { continue; }
        if (limit < 3.4e38 && box_near(c, node.lo.xyz, node.hi.xyz) > limit) { continue; }
        if (node.info.y > 0u) {
            for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                let m = items[k];
                pc_work.y++;
                var f = mesh_fragment(c, m, sp);
                if (f.cover <= 1e-4) { continue; }
                let mesh_solid = frag_solid;
                // Its depth: along a ray through its coverage, else through
                // its most covered bins, else the box's near side.
                var hit = fragment_hit(c, m, f.st, sp);
                if (hit.mesh != m && bit_count(f.mask) > 0u) {
                    hit = fragment_hit(c, m, mask_centroid(f.mask), sp);
                }
                f.depth = select(box_near(c, node.lo.xyz, node.hi.xyz), hit.distance, hit.mesh == m);
                insert_fragment(f);
                solid |= mesh_solid;
                if (bit_count(solid) == BINS) {
                    // A little beyond the farthest kept fragment: a
                    // fragment's surface can reach past the depth of its
                    // one ray.
                    var far = 0.0;
                    for (var j = 0u; j < frag_count; j++) { far = max(far, frags[j].depth); }
                    limit = far * 1.02;
                }
            }
        } else {
            var near = node.info.x;
            var far = node.info.x + 1u;
            let a = nodes[near];
            let b = nodes[far];
            // The one whose box begins nearer along the cone is popped
            // first: a large surface behind (the ground) is met early and
            // bounds the walk.
            if (box_near(c, b.lo.xyz, b.hi.xyz) < box_near(c, a.lo.xyz, a.hi.xyz)) {
                near = node.info.x + 1u;
                far = node.info.x;
            }
            stack_set(sp, far);
            stack_set(sp + 1u, near);
            sp += 2u;
        }
    }
}

// The fragments' visible coverage (nearest first), the sum of it, and the
// most visible fragment.
var<private> frag_weight: array<f32, PIXEL_FRAGMENTS>;
var<private> frag_visible: array<vec2<u32>, PIXEL_FRAGMENTS>;
var<private> frag_total: f32 = 0.0;
var<private> frag_best: u32 = 0u;

// Finds pixel cone `c`'s fragments (pixel `gid`) and their visible
// coverage: bins hidden by nearer fragments' masks do not count, and a
// fragment without a mask (every bin under half covered) is hidden in
// proportion to the bins already covered. Returns the color of a light the
// pixel's center sees in front of every fragment (w: 1), shown as with one
// ray, or w: 0.
fn cone_pixel_setup(gid: vec3<u32>, c: Cone) -> vec4<f32> {
    cull_back = true;
    pixel_fragments(c);
    let center = camera_dir(gid);
    var nearest = 3.4e38;
    if (frag_count > 0u) { nearest = frags[0].depth; }
    let seen = seen_light(c.apex, center, nearest);
    if (seen != 0u) {
        let l = lights[seen - 1u];
        return vec4<f32>(l.radiance.xyz * spot_falloff(l, -center), 1.0);
    }
    var covered = vec2<u32>(0u);
    frag_total = 0.0;
    frag_best = 0u;
    for (var k = 0u; k < frag_count; k++) {
        let f = frags[k];
        let vis = f.mask & ~covered;
        let bits = bit_count(f.mask);
        var w = f.cover * (1.0 - f32(bit_count(covered)) / f32(BINS));
        if (bits > 0u) { w = f.cover * f32(bit_count(vis)) / f32(bits); }
        w = min(w, max(1.0 - frag_total, 0.0));
        frag_weight[k] = w;
        frag_visible[k] = vis;
        frag_total += w;
        covered |= f.mask;
        if (w > frag_weight[frag_best]) { frag_best = k; }
    }
    return vec4<f32>(0.0);
}

// Where fragment `k` of pixel cone `c` is shaded: a ray through its visible
// bins, else through its coverage. `hit.mesh` is not the fragment's if
// neither ray hits it (its coverage too thin).
struct FragmentRay {
    hit: Hit,
    dir: vec3<f32>,
};

fn fragment_ray(c: Cone, k: u32) -> FragmentRay {
    let m = frags[k].mesh;
    var st = mask_centroid(frag_visible[k]);
    var hit = MISS;
    if (bit_count(frag_visible[k]) > 0u) { hit = fragment_hit(c, m, st, 0u); }
    if (hit.mesh != m) {
        st = frags[k].st;
        hit = fragment_hit(c, m, st, 0u);
    }
    return FragmentRay(hit, normalize(direction(c, st.x, st.y)));
}

// The pixel's color from the weighted sum of its shaded fragments' colors
// (`shaded`: their weights): the share of fragments no ray hit goes to the
// others, and the sky fills what no fragment covers.
fn cone_pixel_finish(color: vec3<f32>, shaded: f32) -> vec3<f32> {
    var c = color;
    if (shaded > 0.0) { c *= frag_total / shaded; }
    return c + max(1.0 - frag_total, 0.0) * light.sky.xyz;
}

@compute @workgroup_size(8, 8)
fn render_cones(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let pixel = gid.y * w + gid.x;
    let c = pixel_cone(gid);
    // M13 gate 1 (profile stages 10 and 11): the share of the pixel covered
    // by objects other than object 0 (the ground). Stage 10: the pixel's
    // cone walked as light cones are, prefiltered at the light's threshold
    // (bins), objects as independent occluders; x the coverage, y object
    // nodes tested, z triangle-hierarchy nodes tested (with `COUNT_WORK`).
    // Stage 11: one ray at the pixel offset, 1 if it hits such an object.
    if (u32(light.params.z) == 10u) {
        prefilter_bins = light.params.x;
        independent_occluders = true;
        skip_object = 0u;
        var cov: array<f32, 64>;
        for (var b = 0u; b < 64u; b++) { cov[b] = 0.0; }
        var full = 0u;
        scene_coverage(c, 0u, &cov, &full);
        var sum = 0.0;
        for (var b = 0u; b < 64u; b++) { sum += cov[b]; }
        colors[pixel] = vec4<f32>(sum / 64.0, f32(work_a.x), f32(work_a.w), 1.0);
        return;
    }
    // M13 gate 2: stage 13, one ray at the pixel offset, 1 if it hits
    // anything.
    if (u32(light.params.z) == 13u) {
        let hit = primary_hit(camera.eye.xyz, camera_dir(gid));
        colors[pixel] = vec4<f32>(select(0.0, 1.0, hit.mesh != NONE), 0.0, 0.0, 1.0);
        return;
    }
    if (u32(light.params.z) == 12u) {
        colors[pixel] = vec4<f32>(scalar_coverage(gid), 1.0);
        return;
    }
    if (u32(light.params.z) == 11u) {
        let hit = primary_hit(camera.eye.xyz, camera_dir(gid));
        colors[pixel] = vec4<f32>(select(0.0, 1.0, hit.mesh != NONE && hit.mesh != 0u), 0.0, 0.0, 1.0);
        return;
    }
    let seen = cone_pixel_setup(gid, c);
    if (u32(light.params.z) == 9u) {
        colors[pixel] = vec4<f32>(vec3<f32>(pc_work), f32(frag_count));
        return;
    }

    if (seen.w > 0.0) {
        colors[pixel] = vec4<f32>(seen.xyz, 1.0);
        return;
    }
    var color = vec3<f32>(0.0);
    var shaded = 0.0;
    for (var k = 0u; k < frag_count; k++) {
        if (frag_weight[k] <= 0.0) { continue; }
        let r = fragment_ray(c, k);
        if (r.hit.mesh != frags[k].mesh) { continue; }
        color += frag_weight[k] * per_pixel_hit_color(pixel, c.apex, r.dir, r.hit);
        shaded += frag_weight[k];
    }
    colors[pixel] = vec4<f32>(cone_pixel_finish(color, shaded), 1.0);
}

// `vc_shade` with pixel cones for visibility (no visibility pass).
@compute @workgroup_size(8, 8)
fn vc_shade_cones(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let in_bounds = gid.x < w && gid.y < h;
    let tile = vc.counts.z;
    let sample = (vc.counts.y * 7u) % (tile * tile);
    let is_sample = in_bounds && gid.x % tile == sample % tile && gid.y % tile == sample / tile;
    vc_request_priority = 0u;
    vc_fell_back = false;
    if (in_bounds) {
        let pixel = gid.y * w + gid.x;
        let c = pixel_cone(gid);
        var color = cone_pixel_setup(gid, c);
        if (color.w == 0.0) {
            var sum = vec3<f32>(0.0);
            var shaded = 0.0;
            for (var k = 0u; k < frag_count; k++) {
                if (frag_weight[k] <= 0.0) { continue; }
                let r = fragment_ray(c, k);
                if (r.hit.mesh != frags[k].mesh) { continue; }
                // The most visible fragment requests pages for the pixel.
                sum += frag_weight[k] * vc_hit_color(r.dir, r.hit, is_sample && k == frag_best);
                shaded += frag_weight[k];
            }
            color = vec4<f32>(cone_pixel_finish(sum, shaded), 1.0);
        }
        colors[pixel] = vec4<f32>(color.xyz, 1.0);
    }
    if (is_sample && vc_request_priority > 0u) {
        vc_request(vc_request_key, vc_request_priority);
    }
    if (vc_fell_back) { atomicAdd(&fallback_count, 1u); }
    workgroupBarrier();
    if (li == 0u) {
        let count = atomicLoad(&fallback_count);
        if (count > 0u) { atomicAdd(&vc_stats[0], count); }
    }
}

// --- M13 gate 2: a pixel's coverage by geometry, a scalar test per node. ---
//
// Boxes are tested in screen space (pixels): a box within one pixel adds the
// share of the pixel its projected box covers times its prefiltered opacity
// (no bins); larger boxes are opened, nearer child first, and triangles add
// the exact area of the pixel they cover. Within an object, contributions add
// up to 1; objects combine as independent occluders; the walk stops once the
// pixel is covered. Points go to camera space with one transform per object
// (`View3`). Returns (coverage, object nodes visited, triangle-hierarchy
// nodes visited).

// Camera space (x right, y up, z along the view) from a space: cam = (r0·p,
// r1·p, r2·p) + t.
struct View3 {
    r0: vec3<f32>,
    r1: vec3<f32>,
    r2: vec3<f32>,
    t: vec3<f32>,
};

fn view_world() -> View3 {
    let e = camera.eye.xyz;
    return View3(camera.right.xyz, camera.up.xyz, camera.forward.xyz,
        -vec3<f32>(dot(camera.right.xyz, e), dot(camera.up.xyz, e), dot(camera.forward.xyz, e)));
}

// The view of object space for transform `x` (NONE: world).
fn view_object(x: u32) -> View3 {
    let w = view_world();
    if (x == NONE) { return w; }
    let s = xforms[4u * x + 3u].x;
    let a = xforms[4u * x].xyz;
    let b = xforms[4u * x + 1u].xyz;
    let c = xforms[4u * x + 2u].xyz;
    let tr = translation(x);
    // world = s (a·p, b·p, c·p) + tr.
    let r0 = s * (w.r0.x * a + w.r0.y * b + w.r0.z * c);
    let r1 = s * (w.r1.x * a + w.r1.y * b + w.r1.z * c);
    let r2 = s * (w.r2.x * a + w.r2.y * b + w.r2.z * c);
    return View3(r0, r1, r2, vec3<f32>(dot(w.r0, tr), dot(w.r1, tr), dot(w.r2, tr)) + w.t);
}

fn to_camera(v: View3, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(v.r0, p), dot(v.r1, p), dot(v.r2, p)) + v.t;
}

// A camera-space point in pixels (x right, y down).
fn cam_to_pixels(q: vec3<f32>) -> vec2<f32> {
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    return vec2<f32>((q.x / q.z / (tz * aspect) + 1.0) * 0.5 * camera.params.x, (1.0 - q.y / q.z / tz) * 0.5 * camera.params.y);
}

struct ScreenBox {
    lo: vec2<f32>,
    hi: vec2<f32>,
    // The box's centre's depth along the view.
    depth: f32,
    // 0: every corner in front of the eye; 1: some behind; 2: all behind.
    behind: u32,
};

// The pixel's frustum, as tangent bounds (x left, x right, y bottom, y top)
// of its four side planes through the eye.
var<private> sc_frustum: vec4<f32> = vec4<f32>(0.0);

fn set_pixel_frustum(pix: vec2<f32>) {
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    sc_frustum = vec4<f32>(
        (2.0 * pix.x / camera.params.x - 1.0) * tz * aspect,
        (2.0 * (pix.x + 1.0) / camera.params.x - 1.0) * tz * aspect,
        (1.0 - 2.0 * (pix.y + 1.0) / camera.params.y) * tz,
        (1.0 - 2.0 * pix.y / camera.params.y) * tz,
    );
}

// `behind` 2 also when the box lies wholly outside one of the pixel
// frustum's side planes (they hold on both sides of the eye).
fn screen_box(lo: vec3<f32>, hi: vec3<f32>, v: View3) -> ScreenBox {
    var sb = ScreenBox(vec2<f32>(3.4e38), vec2<f32>(-3.4e38), 0.0, 0u);
    var behind = 0u;
    var out = vec4<u32>(0u);
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u), select(lo.z, hi.z, (k & 4u) != 0u));
        let q = to_camera(v, corner);
        let left = sc_frustum.x * q.z - q.x;
        let right = q.x - sc_frustum.y * q.z;
        let bottom = sc_frustum.z * q.z - q.y;
        let top = q.y - sc_frustum.w * q.z;
        out += select(vec4<u32>(0u), vec4<u32>(1u), vec4<f32>(left, right, bottom, top) > vec4<f32>(0.0));
        if (q.z <= 1e-3) { behind++; continue; }
        let s = cam_to_pixels(q);
        sb.lo = min(sb.lo, s);
        sb.hi = max(sb.hi, s);
    }
    sb.behind = select(select(0u, 1u, behind > 0u), 2u, behind == 8u || any(out == vec4<u32>(8u)));
    sb.depth = to_camera(v, (lo + hi) * 0.5).z;
    return sb;
}

// The area of pixel `pix`'s square inside the box's rectangle.
fn rect_overlap(sb: ScreenBox, pix: vec2<f32>) -> f32 {
    let lo = max(sb.lo, pix);
    let hi = min(sb.hi, pix + vec2<f32>(1.0));
    let d = max(hi - lo, vec2<f32>(0.0));
    return d.x * d.y;
}

// A node's splat: opacity `alpha` over its silhouette `sil` (world area
// across the view at its centre), spread evenly over its projected box.
fn box_splat(sb: ScreenBox, pix: vec2<f32>, alpha: f32, sil: f32) -> f32 {
    let f = camera.params.y / (2.0 * camera.params.z) / max(sb.depth, 1e-3);
    let area = max((sb.hi.x - sb.lo.x) * (sb.hi.y - sb.lo.y), 1e-12);
    return clamp(alpha * sil * f * f * rect_overlap(sb, pix) / area, 0.0, 1.0);
}

fn box_silhouette(lo: vec3<f32>, hi: vec3<f32>, v: vec3<f32>) -> f32 {
    let e = hi - lo;
    return abs(v.x) * e.y * e.z + abs(v.y) * e.x * e.z + abs(v.z) * e.x * e.y;
}

// The area of pixel `pix`'s square covered by triangle `t` seen through
// view `v` (front faces; none if a corner is behind the eye).
fn triangle_cover(t: u32, v: View3, pix: vec2<f32>) -> f32 {
    let c0 = to_camera(v, tris[3u * t].xyz);
    let c1 = to_camera(v, tris[3u * t + 1u].xyz);
    let c2 = to_camera(v, tris[3u * t + 2u].xyz);
    if (c0.z <= 1e-3 || c1.z <= 1e-3 || c2.z <= 1e-3) { return 0.0; }
    if (dot(cross(c1 - c0, c2 - c0), -c0) <= 0.0) { return 0.0; }
    let a = cam_to_pixels(c0);
    let b = cam_to_pixels(c1);
    let c = cam_to_pixels(c2);
    let lo = min(a, min(b, c));
    let hi = max(a, max(b, c));
    if (hi.x <= pix.x || hi.y <= pix.y || lo.x >= pix.x + 1.0 || lo.y >= pix.y + 1.0) { return 0.0; }
    let orient = sign((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    if (orient == 0.0) { return 0.0; }
    var poly: array<vec2<f32>, 8>;
    var n = 4u;
    poly[0] = pix;
    poly[1] = pix + vec2<f32>(1.0, 0.0);
    poly[2] = pix + vec2<f32>(1.0, 1.0);
    poly[3] = pix + vec2<f32>(0.0, 1.0);
    let edges = array<vec2<f32>, 3>(a, b, c);
    for (var e = 0u; e < 3u; e++) {
        let p0 = edges[e];
        let p1 = edges[(e + 1u) % 3u];
        var out: array<vec2<f32>, 8>;
        var m = 0u;
        for (var i = 0u; i < n; i++) {
            let u = poly[i];
            let w = poly[(i + 1u) % n];
            let du = orient * ((p1.x - p0.x) * (u.y - p0.y) - (p1.y - p0.y) * (u.x - p0.x));
            let dw = orient * ((p1.x - p0.x) * (w.y - p0.y) - (p1.y - p0.y) * (w.x - p0.x));
            if (du >= 0.0 && m < 8u) { out[m] = u; m++; }
            if ((du >= 0.0) != (dw >= 0.0) && m < 8u) { out[m] = u + (w - u) * (du / (du - dw)); m++; }
        }
        n = m;
        poly = out;
        if (n < 3u) { return 0.0; }
    }
    var area = 0.0;
    for (var i = 0u; i < n; i++) {
        let u = poly[i];
        let w = poly[(i + 1u) % n];
        area += u.x * w.y - w.x * u.y;
    }
    return abs(area) * 0.5;
}

var<private> sc_tri_nodes: u32 = 0u;

// Object `m`'s coverage of pixel `pix`, its triangle hierarchy walked with
// the stack from `base` up, nearer child first.
fn mesh_scalar_cover(m: u32, pix: vec2<f32>, base: u32) -> f32 {
    let range = meshes[m];
    let x = range.w;
    let v = view_object(x);
    var cm = 0.0;
    if (range.z == NONE) {
        for (var t = range.x; t < range.x + range.y; t++) { cm += triangle_cover(t, v, pix); }
        return min(cm, 1.0);
    }
    // The eye in object space, for nodes' view directions.
    let eye = point_to_object(m, camera.eye.xyz);
    let scale = select(1.0, xforms[4u * x + 3u].x, x != NONE);
    var sp = base + 1u;
    stack_set(base, range.z);
    while (sp > base && cm < 0.9999) {
        sp--;
        let n = stack_get(sp);
        let node = blas[n];
        sc_tri_nodes++;
        let sb = screen_box(node.lo.xyz, node.hi.xyz, v);
        if (sb.behind == 2u) { continue; }
        if (sb.behind == 0u && rect_overlap(sb, pix) <= 0.0) { continue; }
        if (sb.behind == 0u && max(sb.hi.x - sb.lo.x, sb.hi.y - sb.lo.y) <= 1.0) {
            let dir = normalize(eye - (node.lo.xyz + node.hi.xyz) * 0.5);
            let sil = box_silhouette(node.lo.xyz, node.hi.xyz, dir);
            if (sil > 1e-30) {
                let alpha = min(node_front_area(n, dir) / sil, 1.0);
                cm += box_splat(sb, pix, alpha, sil * scale * scale);
            }
            continue;
        }
        if (node.info.y > 0u) {
            for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                cm += triangle_cover(blas_items[k], v, pix);
            }
        } else if (sp + 2u <= STACK) {
            var near = node.info.x;
            var far = node.info.x + 1u;
            let a = blas[near];
            let b = blas[far];
            if (to_camera(v, (b.lo.xyz + b.hi.xyz) * 0.5).z < to_camera(v, (a.lo.xyz + a.hi.xyz) * 0.5).z) {
                near = node.info.x + 1u;
                far = node.info.x;
            }
            stack_set(sp, far);
            stack_set(sp + 1u, near);
            sp += 2u;
        }
    }
    return min(cm, 1.0);
}

fn scalar_coverage(gid: vec3<u32>) -> vec3<f32> {
    let pix = vec2<f32>(f32(gid.x), f32(gid.y));
    set_pixel_frustum(pix);
    let v = view_world();
    var total = 0.0;
    var object_nodes = 0u;
    sc_tri_nodes = 0u;
    var sp = 1u;
    stack_set(0u, 0u);
    while (sp > 0u && total < 0.9999) {
        sp--;
        let n = stack_get(sp);
        let node = nodes[n];
        object_nodes++;
        let sb = screen_box(node.lo.xyz, node.hi.xyz, v);
        if (sb.behind == 2u) { continue; }
        if (sb.behind == 0u && rect_overlap(sb, pix) <= 0.0) { continue; }
        if (sb.behind == 0u && node.info.z == 0u && light.objects.x != NONE
            && max(sb.hi.x - sb.lo.x, sb.hi.y - sb.lo.y) <= 1.0) {
            let dir = normalize(camera.eye.xyz - (node.lo.xyz + node.hi.xyz) * 0.5);
            let sil = box_silhouette(node.lo.xyz, node.hi.xyz, dir);
            if (sil > 1e-30) {
                let alpha = 1.0 - exp(-node_front_area(light.objects.x + n, dir) / sil);
                let c = box_splat(sb, pix, alpha, sil);
                total = 1.0 - (1.0 - total) * (1.0 - c);
            }
            continue;
        }
        if (node.info.y > 0u) {
            for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                let c = mesh_scalar_cover(items[k], pix, sp);
                total = 1.0 - (1.0 - total) * (1.0 - c);
            }
        } else if (sp + 2u <= STACK) {
            var near = node.info.x;
            var far = node.info.x + 1u;
            let a = nodes[near];
            let b = nodes[far];
            if (to_camera(v, (b.lo.xyz + b.hi.xyz) * 0.5).z < to_camera(v, (a.lo.xyz + a.hi.xyz) * 0.5).z) {
                near = node.info.x + 1u;
                far = node.info.x;
            }
            stack_set(sp, far);
            stack_set(sp + 1u, near);
            sp += 2u;
        }
    }
    return vec3<f32>(total, f32(object_nodes), f32(sc_tri_nodes));
}

// M13: `scalar_coverage` in its own entry point (profile stage 14 selects
// it), writing (coverage, object nodes, triangle nodes).
@compute @workgroup_size(8, 8)
fn m13_coverage(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    colors[gid.y * w + gid.x] = vec4<f32>(scalar_coverage(gid), 1.0);
}

// --- M13 gate 3: coverage through 4×4 sub-pixel bins and directional
// coverage maps (`coverage_maps.rs` builds them; same conventions). ---

@group(0) @binding(82) var<storage, read> cov_maps: array<u32>;

// Object space (transform `x`, NONE: world) to world.
fn m13_to_world(x: u32, p: vec3<f32>) -> vec3<f32> {
    if (x == NONE) { return p; }
    return rotate_to_world(x, p * xforms[4u * x + 3u].x) + translation(x);
}

const MAP_DIRS: u32 = 16u;
const MAP_TEXELS: u32 = 16u;
const MAP_PER_DIR: u32 = 341u;
// Instances at most this many pixels across read their map.
const MAP_LIMIT_PX: f32 = 8.0;

// The pixel's bins, and the bins of the object being walked.
var<private> pbin: array<f32, 16>;
var<private> mbin: array<f32, 16>;

fn bins_full(m: bool) -> bool {
    for (var b = 0u; b < 16u; b++) {
        if (select(pbin[b], mbin[b], m) < 0.9999) { return false; }
    }
    return true;
}

// Whether the box's projected rectangle overlaps a bin not yet full.
fn touches_open_bin(sb: ScreenBox, pix: vec2<f32>) -> bool {
    for (var b = 0u; b < 16u; b++) {
        if (pbin[b] >= 0.9999) { continue; }
        let lo = bin_lo(pix, b);
        if (sb.hi.x > lo.x && sb.hi.y > lo.y && sb.lo.x < lo.x + 0.25 && sb.lo.y < lo.y + 0.25) { return true; }
    }
    return false;
}

fn bin_lo(pix: vec2<f32>, b: u32) -> vec2<f32> {
    return pix + vec2<f32>(f32(b % 4u), f32(b / 4u)) * 0.25;
}

// A node's splat into the bins: opacity `alpha` over silhouette `sil`
// (world area at its centre), spread evenly over its projected box; into
// the object's bins (summed) or the pixel's (independent).
fn splat_bins(sb: ScreenBox, pix: vec2<f32>, alpha: f32, sil: f32, into_mesh: bool) {
    let f = camera.params.y / (2.0 * camera.params.z) / max(sb.depth, 1e-3);
    let area = max((sb.hi.x - sb.lo.x) * (sb.hi.y - sb.lo.y), 1e-12);
    let k = alpha * sil * f * f / area * 16.0;
    for (var b = 0u; b < 16u; b++) {
        let lo = bin_lo(pix, b);
        let d = max(min(sb.hi, lo + vec2<f32>(0.25)) - max(sb.lo, lo), vec2<f32>(0.0));
        let o = d.x * d.y;
        if (o <= 0.0) { continue; }
        let c = clamp(k * o, 0.0, 1.0);
        if (into_mesh) { mbin[b] = min(mbin[b] + c, 1.0); } else { pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - c); }
    }
}

// Triangle `t` (view `v`) covers the bins whose centres it contains (front
// faces, every corner in front of the eye).
fn triangle_bins(t: u32, v: View3, pix: vec2<f32>) {
    let c0 = to_camera(v, tris[3u * t].xyz);
    let c1 = to_camera(v, tris[3u * t + 1u].xyz);
    let c2 = to_camera(v, tris[3u * t + 2u].xyz);
    if (c0.z <= 1e-3 || c1.z <= 1e-3 || c2.z <= 1e-3) { return; }
    if (dot(cross(c1 - c0, c2 - c0), -c0) <= 0.0) { return; }
    let a = cam_to_pixels(c0);
    let b = cam_to_pixels(c1);
    let c = cam_to_pixels(c2);
    let lo = min(a, min(b, c));
    let hi = max(a, max(b, c));
    if (hi.x <= pix.x || hi.y <= pix.y || lo.x >= pix.x + 1.0 || lo.y >= pix.y + 1.0) { return; }
    let orient = sign((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    if (orient == 0.0) { return; }
    for (var k = 0u; k < 16u; k++) {
        let p = bin_lo(pix, k) + vec2<f32>(0.125);
        let e0 = orient * ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x));
        let e1 = orient * ((c.x - b.x) * (p.y - b.y) - (c.y - b.y) * (p.x - b.x));
        let e2 = orient * ((a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x));
        if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) { mbin[k] = 1.0; }
    }
}

// Object `m` walked by its triangle hierarchy into `mbin` (as
// `mesh_scalar_cover`, per bin), then combined into the pixel's bins.
fn mesh_bins(m: u32, pix: vec2<f32>, base: u32) {
    let range = meshes[m];
    let x = range.w;
    let v = view_object(x);
    for (var b = 0u; b < 16u; b++) { mbin[b] = 0.0; }
    if (range.z == NONE) {
        for (var t = range.x; t < range.x + range.y; t++) { triangle_bins(t, v, pix); }
    } else {
        let eye = point_to_object(m, camera.eye.xyz);
        let scale = select(1.0, xforms[4u * x + 3u].x, x != NONE);
        var sp = base + 1u;
        stack_set(base, range.z);
        while (sp > base) {
            sp--;
            let n = stack_get(sp);
            let node = blas[n];
            sc_tri_nodes++;
            let sb = screen_box(node.lo.xyz, node.hi.xyz, v);
            if (sb.behind == 2u) { continue; }
            if (sb.behind == 0u && rect_overlap(sb, pix) <= 0.0) { continue; }
            if (sb.behind == 0u && max(sb.hi.x - sb.lo.x, sb.hi.y - sb.lo.y) <= 1.0) {
                let dir = normalize(eye - (node.lo.xyz + node.hi.xyz) * 0.5);
                let sil = box_silhouette(node.lo.xyz, node.hi.xyz, dir);
                if (sil > 1e-30) {
                    splat_bins(sb, pix, min(node_front_area(n, dir) / sil, 1.0), sil * scale * scale, true);
                }
                continue;
            }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    triangle_bins(blas_items[k], v, pix);
                }
                if (bins_full(true)) { break; }
            } else if (sp + 2u <= STACK) {
                var near = node.info.x;
                var far = node.info.x + 1u;
                let a = blas[near];
                let b = blas[far];
                if (to_camera(v, (b.lo.xyz + b.hi.xyz) * 0.5).z < to_camera(v, (a.lo.xyz + a.hi.xyz) * 0.5).z) {
                    near = node.info.x + 1u;
                    far = node.info.x;
                }
                stack_set(sp, far);
                stack_set(sp + 1u, near);
                sp += 2u;
            }
        }
    }
    for (var b = 0u; b < 16u; b++) { pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - mbin[b]); }
}

// The header of the maps for triangle-hierarchy root `root`, or NONE.
fn map_header(root: u32) -> u32 {
    let n = cov_maps[0];
    for (var k = 0u; k < n; k++) {
        if (cov_maps[1u + 8u * k] == root) { return 1u + 8u * k; }
    }
    return NONE;
}

fn oct_encode(d: vec3<f32>) -> vec2<f32> {
    var p = d.xy / (abs(d.x) + abs(d.y) + abs(d.z));
    if (d.z < 0.0) {
        p = (vec2<f32>(1.0) - abs(p.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    }
    return p;
}

fn oct_decode(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (d.z < 0.0) {
        let x = d.x;
        let y = d.y;
        d.x = (1.0 - abs(y)) * select(-1.0, 1.0, x >= 0.0);
        d.y = (1.0 - abs(x)) * select(-1.0, 1.0, y >= 0.0);
    }
    return normalize(d);
}

// The word where level `level` of map `di` begins.
fn map_level_at(base: u32, di: u32, level: u32) -> u32 {
    var off = 0u;
    for (var l = 0u; l < level; l++) { off += (MAP_TEXELS >> l) * (MAP_TEXELS >> l); }
    return base + di * MAP_PER_DIR + off;
}

fn map_texel(at: u32, side: i32, x: i32, y: i32) -> f32 {
    if (x < 0 || y < 0 || x >= side || y >= side) { return 0.0; }
    return bitcast<f32>(cov_maps[at + u32(y * side + x)]);
}

// The map level beginning at word `at` (`side` texels across), bilinearly,
// at `uv` in [−1, 1]² (0 outside).
fn map_sample(at: u32, side: i32, uv: vec2<f32>) -> f32 {
    let g = (uv + vec2<f32>(1.0)) * 0.5 * f32(side) - vec2<f32>(0.5);
    let i0 = floor(g);
    let f = g - i0;
    let x = i32(i0.x);
    let y = i32(i0.y);
    let t00 = map_texel(at, side, x, y);
    let t10 = map_texel(at, side, x + 1, y);
    let t01 = map_texel(at, side, x, y + 1);
    let t11 = map_texel(at, side, x + 1, y + 1);
    return mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
}

// Object `m`, at most MAP_LIMIT_PX across, read from its maps (header
// `hdr`) into the pixel's bins. False if it is too large or too near (walk
// it instead).
fn instance_map_bins(m: u32, hdr: u32, pix: vec2<f32>) -> bool {
    let x = meshes[m].w;
    let c_obj = vec3<f32>(bitcast<f32>(cov_maps[hdr + 2u]), bitcast<f32>(cov_maps[hdr + 3u]), bitcast<f32>(cov_maps[hdr + 4u]));
    let r_obj = bitcast<f32>(cov_maps[hdr + 5u]);
    let base = cov_maps[hdr + 1u];
    let scale = select(1.0, xforms[4u * x + 3u].x, x != NONE);
    let centre = m13_to_world(x, c_obj);
    let radius = r_obj * scale;
    let rel = centre - camera.eye.xyz;
    let depth = dot(rel, camera.forward.xyz);
    if (depth - radius <= 1e-2) { return false; }
    let f = camera.params.y / (2.0 * camera.params.z);
    let rad = radius * f / depth;
    if (2.0 * rad > MAP_LIMIT_PX) { return false; }
    let cp = cam_to_pixels(to_camera(view_world(), centre));
    if (cp.x + rad <= pix.x || cp.y + rad <= pix.y || cp.x - rad >= pix.x + 1.0 || cp.y - rad >= pix.y + 1.0) { return true; }
    let dw = normalize(rel);
    var d = dw;
    if (x != NONE) { d = normalize(rotate_to_object(x, dw)); }
    let g = (oct_encode(d) + vec2<f32>(1.0)) * 0.5 * f32(MAP_DIRS) - vec2<f32>(0.5);
    let i0 = floor(g);
    let fr = g - i0;
    // The level whose texels match a bin's footprint.
    let texels_per_bin = 0.25 * depth / f / (2.0 * radius / f32(MAP_TEXELS));
    let level = u32(clamp(round(log2(max(texels_per_bin, 1e-6))), 0.0, 4.0));
    var dirs: array<u32, 4>;
    let side = i32(MAP_TEXELS >> level);
    var weights: array<f32, 4>;
    var us: array<vec3<f32>, 4>;
    var vs: array<vec3<f32>, 4>;
    for (var k = 0u; k < 4u; k++) {
        let ix = u32(clamp(i32(i0.x) + i32(k & 1u), 0, i32(MAP_DIRS) - 1));
        let iy = u32(clamp(i32(i0.y) + i32(k >> 1u), 0, i32(MAP_DIRS) - 1));
        dirs[k] = map_level_at(base, iy * MAP_DIRS + ix, level);
        weights[k] = select(1.0 - fr.x, fr.x, (k & 1u) == 1u) * select(1.0 - fr.y, fr.y, (k >> 1u) == 1u);
        let dk = oct_decode((vec2<f32>(f32(ix), f32(iy)) + vec2<f32>(0.5)) / f32(MAP_DIRS) * 2.0 - vec2<f32>(1.0));
        let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(dk.z) < 0.999);
        us[k] = normalize(cross(helper, dk));
        vs[k] = cross(dk, us[k]);
    }
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    for (var b = 0u; b < 16u; b++) {
        let p = bin_lo(pix, b) + vec2<f32>(0.125);
        if (abs(p.x - cp.x) > rad || abs(p.y - cp.y) > rad) { continue; }
        let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
        let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
        let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
        let qw = camera.eye.xyz + dir * dot(rel, dir) - centre;
        var q = qw / r_obj;
        if (x != NONE) { q = rotate_to_object(x, qw) / (scale * r_obj); }
        var cov = 0.0;
        for (var k = 0u; k < 4u; k++) {
            if (weights[k] <= 0.0) { continue; }
            cov += weights[k] * map_sample(dirs[k], side, vec2<f32>(dot(q, us[k]), dot(q, vs[k])));
        }
        pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - clamp(cov, 0.0, 1.0));
    }
    return true;
}

fn map_coverage(gid: vec3<u32>) -> vec3<f32> {
    for (var b = 0u; b < 16u; b++) { pbin[b] = 0.0; }
    return map_coverage_from(gid, false);
}

// `map_coverage` from the pixel's bins as they are; with `rays_did_large`
// (M13 gate 4) large instances are left out of the walk: the bins' hardware
// rays covered them.
fn map_coverage_from(gid: vec3<u32>, rays_did_large: bool) -> vec3<f32> {
    let pix = vec2<f32>(f32(gid.x), f32(gid.y));
    set_pixel_frustum(pix);
    let v = view_world();
    if (bins_full(false)) { return vec3<f32>(1.0, 0.0, 0.0); }
    var object_nodes = 0u;
    sc_tri_nodes = 0u;
    var sp = 1u;
    stack_set(0u, 0u);
    while (sp > 0u) {
        sp--;
        let n = stack_get(sp);
        let node = nodes[n];
        object_nodes++;
        let sb = screen_box(node.lo.xyz, node.hi.xyz, v);
        if (sb.behind == 2u) { continue; }
        if (sb.behind == 0u && rect_overlap(sb, pix) <= 0.0) { continue; }
        // A node over full bins only cannot change the pixel.
        if (sb.behind == 0u && !touches_open_bin(sb, pix)) { continue; }
        // Within a pixel, or (far, at grazing views) no taller than a bin:
        // splat, spread over its projected box.
        let extent = sb.hi - sb.lo;
        if (sb.behind == 0u && node.info.z == 0u && light.objects.x != NONE
            && (max(extent.x, extent.y) <= 1.0 || min(extent.x, extent.y) <= 0.25)) {
            let dir = normalize(camera.eye.xyz - (node.lo.xyz + node.hi.xyz) * 0.5);
            let sil = box_silhouette(node.lo.xyz, node.hi.xyz, dir);
            if (sil > 1e-30) {
                splat_bins(sb, pix, 1.0 - exp(-node_front_area(light.objects.x + n, dir) / sil), sil, false);
                if (bins_full(false)) { break; }
            }
            continue;
        }
        if (node.info.y > 0u) {
            for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                let m = items[k];
                if (rays_did_large && m13_large(m)) { continue; }
                let hdr = map_header(meshes[m].z);
                if (!(hdr != NONE && instance_map_bins(m, hdr, pix)) && !rays_did_large) { mesh_bins(m, pix, sp); }
            }
            if (bins_full(false)) { break; }
        } else if (sp + 2u <= STACK) {
            var near = node.info.x;
            var far = node.info.x + 1u;
            let a = nodes[near];
            let b = nodes[far];
            if (to_camera(v, (b.lo.xyz + b.hi.xyz) * 0.5).z < to_camera(v, (a.lo.xyz + a.hi.xyz) * 0.5).z) {
                near = node.info.x + 1u;
                far = node.info.x;
            }
            stack_set(sp, far);
            stack_set(sp + 1u, near);
            sp += 2u;
        }
    }
    var sum = 0.0;
    for (var b = 0u; b < 16u; b++) { sum += pbin[b]; }
    return vec3<f32>(sum / 16.0, f32(object_nodes), f32(sc_tri_nodes));
}

// M13 gate 3 (profile stage 15 selects it): `map_coverage`, writing
// (coverage, object nodes, triangle nodes).
@compute @workgroup_size(8, 8)
fn m13_coverage_maps(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    colors[gid.y * w + gid.x] = vec4<f32>(map_coverage(gid), 1.0);
}

// --- M13 gate 4: near surfaces resolved by the corner rays. ---

// Whether object `m` is too large (or too near, or without maps) to be read
// from its coverage map: `instance_map_bins` would walk it.
fn m13_large(m: u32) -> bool {
    let hdr = map_header(meshes[m].z);
    if (hdr == NONE) { return true; }
    let x = meshes[m].w;
    let c_obj = vec3<f32>(bitcast<f32>(cov_maps[hdr + 2u]), bitcast<f32>(cov_maps[hdr + 3u]), bitcast<f32>(cov_maps[hdr + 4u]));
    let scale = select(1.0, xforms[4u * x + 3u].x, x != NONE);
    let radius = bitcast<f32>(cov_maps[hdr + 5u]) * scale;
    let depth = dot(m13_to_world(x, c_obj) - camera.eye.xyz, camera.forward.xyz);
    if (depth - radius <= 1e-2) { return true; }
    let f = camera.params.y / (2.0 * camera.params.z);
    return 2.0 * radius * f / depth > MAP_LIMIT_PX;
}
