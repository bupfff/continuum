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
};

struct Receiver {
    point: vec4<f32>,
    normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> light: Light;
@group(0) @binding(1) var<storage, read> receivers: array<Receiver>;
@group(0) @binding(2) var<storage, read> tris: array<vec4<f32>>;
// x: first triangle, y: triangle count, z: root of the mesh's triangle
// hierarchy in `blas` (NONE for an empty mesh)
@group(0) @binding(3) var<storage, read> meshes: array<vec4<u32>>;
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

const NONE: u32 = 0xffffffffu;
// Traversal stack: the mesh hierarchy's entries, then (above them) the current
// mesh's triangle hierarchy.
const STACK: u32 = 48u;

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
    for (var k = 0u; k < 8u; k++) {
        let corner = vec3<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u), select(lo.z, hi.z, (k & 4u) != 0u));
        if (c.far_offset - dot(c.far_normal, corner) >= 0.0) { far_out = false; }
        let p = to_local(c, corner);
        if (p.z >= NEAR) { near_out = false; }
        if (p.x - c.window.x * p.z >= 0.0) { out_u0 = false; }
        if (c.window.z * p.z - p.x >= 0.0) { out_u1 = false; }
        if (p.y - c.window.y * p.z >= 0.0) { out_v0 = false; }
        if (c.window.w * p.z - p.y >= 0.0) { out_v1 = false; }
    }
    return !(far_out || near_out || out_u0 || out_u1 || out_v0 || out_v1);
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
fn draw_edge(acc: ptr<function, array<f32, 68>>, span: ptr<function, vec2<u32>>, q0: vec2<f32>, q1: vec2<f32>) {
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
        let d = dy * dir;
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
        draw_edge(acc, span, a, b);
    }
    return true;
}

// Rasterizes triangle `t` into `acc` if it faces the apex and may touch the cone.
fn cover_triangle(c: Cone, acc: ptr<function, array<f32, 68>>, span: ptr<function, vec2<u32>>, t: u32) -> bool {
    let v0 = tris[3u * t].xyz;
    let v1 = tris[3u * t + 1u].xyz;
    let v2 = tris[3u * t + 2u].xyz;
    if (dot(cross(v1 - v0, v2 - v0), c.apex - v0) <= 0.0) { return false; }
    if (!may_touch(c, v0, v1, v2)) { return false; }
    return rasterize(c, acc, span, v0, v1, v2);
}

// Rasterizes one mesh's front-facing triangles into a fresh accumulation buffer
// and combines the resulting per-bin coverage into `cov` by max. Triangle
// hierarchy traversal uses `stack` from `base` up.
fn add_mesh_coverage(c: Cone, m: u32, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>, i: u32,
                     stack: ptr<function, array<u32, STACK>>, base: u32) {
    let range = meshes[m];
    var acc: array<f32, 68>;
    for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
    // Cells written by this mesh's edges; the running sum is zero outside them.
    var span = vec2<u32>(68u, 0u);
    var touched = false;
    if (light.counts.z == 2u) {
        if (range.z != NONE) {
            var sp = base + 1u;
            (*stack)[base] = range.z;
            while (sp > base) {
                sp--;
                let node = blas[(*stack)[sp]];
                if (!may_touch_box(c, node.lo.xyz, node.hi.xyz)) { continue; }
                if (node.info.y > 0u) {
                    for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                        if (cover_triangle(c, &acc, &span, blas_items[k])) { touched = true; }
                    }
                } else {
                    (*stack)[sp] = node.info.x;
                    (*stack)[sp + 1u] = node.info.x + 1u;
                    sp += 2u;
                }
            }
        }
    } else {
        for (var t = range.x; t < range.x + range.y; t++) {
            if (cover_triangle(c, &acc, &span, t)) { touched = true; }
        }
    }
    if (light.counts.y == 2u) {
        for (var b = 0u; b < 64u; b++) { debug[i * BINS + b] = acc[b]; }
    }
    if (touched) {
        var running = 0.0;
        for (var b = span.x; b < min(span.y, BINS); b++) {
            running += acc[b];
            let covered = abs(running);
            if (covered > 1e-6) {
                let before = (*cov)[b];
                (*cov)[b] = max(before, min(covered, 1.0));
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

// Adds the coverage of every mesh in the bound hierarchy (or every mesh) to `cov`.
fn scene_coverage(c: Cone, i: u32, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>) {
    var stack: array<u32, STACK>;
    if (light.counts.z >= 1u) {
        var sp = 1u;
        stack[0] = 0u;
        while (sp > 0u) {
            sp--;
            let node = nodes[stack[sp]];
            if (!may_touch_box(c, node.lo.xyz, node.hi.xyz)) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    add_mesh_coverage(c, items[k], cov, full, i, &stack, sp);
                }
                // Every bin fully covered: nothing further can change the result.
                if (*full == BINS) { break; }
            } else {
                stack[sp] = node.info.x;
                stack[sp + 1u] = node.info.x + 1u;
                sp += 2u;
            }
        }
    } else {
        for (var m = 0u; m < light.counts.x; m++) {
            add_mesh_coverage(c, m, cov, full, i, &stack, 0u);
            if (*full == BINS) { break; }
        }
    }
}

// Irradiance through the uncovered part of the cone. Bin weights (projected
// solid angle, Lambert) are computed only now, as they are summed, so no
// per-thread weights array is needed.
fn cone_visible(c: Cone, normal: vec3<f32>, cov: ptr<function, array<f32, 64>>) -> f32 {
    var visible = 0.0;
    for (var b = 0u; b < BINS; b++) {
        let col = f32(b % GRID);
        let row = f32(b / GRID);
        let g = f32(GRID);
        let d0 = direction(c, col / g, row / g);
        let d1 = direction(c, (col + 1.0) / g, row / g);
        let d2 = direction(c, (col + 1.0) / g, (row + 1.0) / g);
        let d3 = direction(c, col / g, (row + 1.0) / g);
        let weight = abs(0.5 * (lambert_edge(normal, d0, d1) + lambert_edge(normal, d1, d2)
            + lambert_edge(normal, d2, d3) + lambert_edge(normal, d3, d0)));
        visible += (1.0 - (*cov)[b]) * weight;
    }
    return visible;
}

// Irradiance per unit radiance at `apex` (surface normal `normal`) from the
// light, through a mask cone. `i` only selects where debug output goes.
fn cone_irradiance(i: u32, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    let lc = light_cone(apex);
    if (!lc.ok) { return 0.0; }
    // Explicit zeroing: don't rely on loop-scoped `var`s being re-initialized.
    var cov: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
    var full = 0u;
    scene_coverage(lc.c, i, &cov, &full);
    let visible = cone_visible(lc.c, normal, &cov);
    if (light.counts.y == 1u) {
        for (var b = 0u; b < BINS; b++) { debug[i * BINS + b] = cov[b]; }
    }
    return visible;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

fn ray_triangle(o: vec3<f32>, d: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-12) { return -1.0; }
    let inv = 1.0 / det;
    let s = o - v0;
    let u = dot(s, p) * inv;
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    return dot(e2, q) * inv;
}

struct Hit {
    distance: f32,
    mesh: u32,
    // Global triangle index.
    tri: u32,
    normal: vec3<f32>,
};

fn hit_triangle(eye: vec3<f32>, dir: vec3<f32>, m: u32, t: u32, best: Hit) -> Hit {
    let v0 = tris[3u * t].xyz;
    let v1 = tris[3u * t + 1u].xyz;
    let v2 = tris[3u * t + 2u].xyz;
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
    let t0 = (lo - eye) * inv;
    let t1 = (hi - eye) * inv;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return !(far < max(near, 0.0) || near > limit);
}

// The nearest hit in mesh `m`, if nearer than `best`. Triangle hierarchy
// traversal uses `stack` from `base` up.
fn nearest_in_mesh(eye: vec3<f32>, dir: vec3<f32>, m: u32, best: Hit,
                   stack: ptr<function, array<u32, STACK>>, base: u32) -> Hit {
    var hit = best;
    let range = meshes[m];
    if (light.counts.z == 2u) {
        if (range.z == NONE) { return hit; }
        let inv = 1.0 / dir;
        var sp = base + 1u;
        (*stack)[base] = range.z;
        while (sp > base) {
            sp--;
            let node = blas[(*stack)[sp]];
            if (!ray_box(eye, inv, node.lo.xyz, node.hi.xyz, hit.distance)) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    hit = hit_triangle(eye, dir, m, blas_items[k], hit);
                }
            } else {
                (*stack)[sp] = node.info.x;
                (*stack)[sp + 1u] = node.info.x + 1u;
                sp += 2u;
            }
        }
    } else {
        for (var t = range.x; t < range.x + range.y; t++) {
            hit = hit_triangle(eye, dir, m, t, hit);
        }
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
    var stack: array<u32, STACK>;
    if (light.counts.z >= 1u) {
        let inv = 1.0 / dir;
        var sp = 1u;
        stack[0] = 0u;
        while (sp > 0u) {
            sp--;
            let node = nodes[stack[sp]];
            // Slab test against the node's box, closer than the best hit so far.
            if (!ray_box(eye, inv, node.lo.xyz, node.hi.xyz, hit.distance)) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    hit = nearest_in_mesh(eye, dir, items[k], hit, &stack, sp);
                }
            } else {
                stack[sp] = node.info.x;
                stack[sp + 1u] = node.info.x + 1u;
                sp += 2u;
            }
        }
    } else {
        for (var m = 0u; m < light.counts.x; m++) {
            hit = nearest_in_mesh(eye, dir, m, hit, &stack, 0u);
        }
    }
    if (dot(hit.normal, dir) > 0.0) { hit.normal = -hit.normal; }
    return hit;
}

// Barycentric coordinates of `p` (on the triangle's plane) in triangle `t`.
fn barycentric(t: u32, p: vec3<f32>) -> vec3<f32> {
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

// Interpolated shading normal at `p` on triangle `t`, on the side `toward` points to.
fn shading_normal(t: u32, p: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let b = barycentric(t, p);
    var n = normalize(normals[3u * t].xyz * b.x + normals[3u * t + 1u].xyz * b.y + normals[3u * t + 2u].xyz * b.z);
    let v0 = tris[3u * t].xyz;
    let geometric = cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0);
    // Seen from behind (back face): flip, like the geometric normal.
    if (dot(geometric, toward) < 0.0) { n = -n; }
    return n;
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
    let to_light = light.corner.xyz + (light.edge_u.xyz + light.edge_v.xyz) * 0.5 - point;
    if (dot(normal, to_light) > 0.0) {
        return cone_irradiance(pixel, point, normal);
    }
    return 0.0;
}

@group(0) @binding(24) var<storage, read> normals: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    let hit = primary_hit(eye, dir);
    let pixel = gid.y * w + gid.x;
    if (light_distance(eye, dir) < hit.distance) { colors[pixel] = vec4<f32>(vec3<f32>(camera.params.w), 1.0); return; }
    if (hit.mesh == 0xffffffffu) { colors[pixel] = vec4<f32>(0.0, 0.0, 0.0, 1.0); return; }
    let surface = eye + dir * hit.distance;
    let point = surface + hit.normal * 1e-4;
    let irradiance = hit_irradiance(pixel, point, shading_normal(hit.tri, surface, -dir));
    let c = albedo[hit.mesh].xyz * (camera.params.w * irradiance / 3.14159265);
    colors[pixel] = vec4<f32>(c, 1.0);
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

// Chart of each triangle, or 0xffffffff for dynamic geometry (shaded per pixel).
@group(0) @binding(11) var<storage, read> tri_face: array<u32>;
@group(0) @binding(12) var<storage, read> faces: array<Face>;
@group(0) @binding(13) var<storage, read> texels: array<f32>;
// Chart coordinates of each triangle's vertices, three per triangle.
@group(0) @binding(23) var<storage, read> tri_uv: array<vec2<f32>>;

// Chart coordinates of point `p` on triangle `t`.
fn chart_uv(t: u32, p: vec3<f32>) -> vec2<f32> {
    let b = barycentric(t, p);
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

@compute @workgroup_size(8, 8)
fn render_cached(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    let hit = primary_hit(eye, dir);
    let pixel = gid.y * w + gid.x;
    if (light_distance(eye, dir) < hit.distance) { colors[pixel] = vec4<f32>(vec3<f32>(camera.params.w), 1.0); return; }
    if (hit.mesh == 0xffffffffu) { colors[pixel] = vec4<f32>(0.0, 0.0, 0.0, 1.0); return; }
    let surface = eye + dir * hit.distance;
    let f = tri_face[hit.tri];
    var irradiance: vec3<f32>;
    if (f == 0xffffffffu) {
        let point = surface + hit.normal * 1e-4;
        irradiance = vec3<f32>(hit_irradiance(pixel, point, shading_normal(hit.tri, surface, -dir)));
    } else {
        let uv = chart_uv(hit.tri, surface);
        irradiance = lookup(faces[f], uv) + indirect_lookup(faces[f], uv);
    }
    let c = albedo[hit.mesh].xyz * irradiance * (camera.params.w / 3.14159265);
    colors[pixel] = vec4<f32>(c, 1.0);
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
    var stack: array<u32, STACK>;
    for (var k = 0u; k < probe_params.counts.w; k++) {
        if (full == BINS) { break; }
        if (!may_touch_box(lc.c, probe_params.bounds[2u * k].xyz, probe_params.bounds[2u * k + 1u].xyz)) { continue; }
        add_mesh_coverage(lc.c, probe_params.moving[k].x, &cov, &full, i, &stack, 0u);
    }
    return cone_visible(lc.c, normal, &cov);
}

// One workgroup per marked block, one invocation per texel; `result` is the
// lightmap's texel buffer and `receivers` its texels.
@compute @workgroup_size(64)
fn rebake(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
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
    if (tri_face[hit.tri] == 0xffffffffu) { return vec2<u32>(BLACK, bitcast<u32>(hit.distance)); }
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
    var stack: array<u32, STACK>;
    for (var k = 0u; k < probe_params.counts.w; k++) {
        let lo = probe_params.bounds[2u * k].xyz;
        let hi = probe_params.bounds[2u * k + 1u].xyz;
        if (!ray_box(origin, inv, lo, hi, best.distance)) { continue; }
        let m = probe_params.moving[k].x;
        let hit = nearest_in_mesh(origin, dir, m, best, &stack, 0u);
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
    let e = lookup(face, chart_uv(g.x, origin + dir * bitcast<f32>(g.y)));
    return vec4<f32>(albedo[face.probes.w].xyz * (e / 3.14159265), 0.0);
}

// One invocation per (bin, probe): dispatched as (bins / 64, probes).
@compute @workgroup_size(64)
fn probe_build(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let p = wg.x;
    let bins = bins_per_probe();
    var sum = vec4<f32>(0.0);
    if (p < probe_params.counts.x) {
        let r = probe_points[p];
        let origin = r.point.xyz;
        for (var b = li; b < bins; b += 64u) {
            let dir = bin_world(r.normal.xyz, bin_local(b, probe_params.counts.y));
            var g: vec2<u32>;
            if (probe_params.counts.z == 0u) {
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
