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
    // x: mesh count; y: 1 to write per-bin coverage to `debug`;
    // z: 1 to traverse the mesh hierarchy instead of testing every mesh
    counts: vec4<u32>,
};

struct Receiver {
    point: vec4<f32>,
    normal: vec4<f32>,
};

@group(0) @binding(0) var<uniform> light: Light;
@group(0) @binding(1) var<storage, read> receivers: array<Receiver>;
@group(0) @binding(2) var<storage, read> tris: array<vec4<f32>>;
// x: first triangle, y: triangle count
@group(0) @binding(3) var<storage, read> meshes: array<vec2<u32>>;
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

// Rasterizes one mesh's front-facing triangles into a fresh accumulation buffer
// and combines the resulting per-bin coverage into `cov` by max.
fn add_mesh_coverage(c: Cone, m: u32, cov: ptr<function, array<f32, 64>>, full: ptr<function, u32>, i: u32) {
    let range = meshes[m];
    var acc: array<f32, 68>;
    for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
    // Cells written by this mesh's edges; the running sum is zero outside them.
    var span = vec2<u32>(68u, 0u);
    var touched = false;
    for (var t = range.x; t < range.x + range.y; t++) {
        let v0 = tris[3u * t].xyz;
        let v1 = tris[3u * t + 1u].xyz;
        let v2 = tris[3u * t + 2u].xyz;
        if (dot(cross(v1 - v0, v2 - v0), c.apex - v0) <= 0.0) { continue; }
        if (!may_touch(c, v0, v1, v2)) { continue; }
        if (rasterize(c, &acc, &span, v0, v1, v2)) { touched = true; }
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

// Irradiance per unit radiance at `apex` (surface normal `normal`) from the
// light, through a mask cone. `i` only selects where debug output goes.
fn cone_irradiance(i: u32, apex: vec3<f32>, normal: vec3<f32>) -> f32 {
    var forward = normalize(cross(light.edge_u.xyz, light.edge_v.xyz));
    if (dot(forward, light.corner.xyz - apex) < 0.0) { forward = -forward; }
    let dist = dot(forward, light.corner.xyz - apex);
    if (dist <= 1e-6) { return 0.0; }
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
    var c: Cone;
    c.apex = apex;
    c.right = right;
    c.up = up;
    c.forward = forward;
    c.window = vec4<f32>(lo, hi);
    c.far_normal = forward;
    c.far_offset = dot(forward, apex) + dist * (1.0 - 1e-5);

    // Explicit zeroing: don't rely on loop-scoped `var`s being re-initialized.
    var cov: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
    var full = 0u;
    if (light.counts.z == 1u) {
        var stack: array<u32, 32>;
        var sp = 1u;
        stack[0] = 0u;
        while (sp > 0u) {
            sp--;
            let node = nodes[stack[sp]];
            if (!may_touch_box(c, node.lo.xyz, node.hi.xyz)) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    add_mesh_coverage(c, items[k], &cov, &full, i);
                }
                // Every bin fully covered: nothing further can change the result.
                if (full == BINS) { break; }
            } else {
                stack[sp] = node.info.x;
                stack[sp + 1u] = node.info.x + 1u;
                sp += 2u;
            }
        }
    } else {
        for (var m = 0u; m < light.counts.x; m++) {
            add_mesh_coverage(c, m, &cov, &full, i);
            if (full == BINS) { break; }
        }
    }
    // Bin weights (projected solid angle, Lambert) computed only now, as they are
    // summed, so no per-thread weights array is needed.
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
        visible += (1.0 - cov[b]) * weight;
    }
    if (light.counts.y == 1u) {
        for (var b = 0u; b < BINS; b++) { debug[i * BINS + b] = cov[b]; }
    }
    return visible;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&receivers)) { return; }
    result[i] = cone_irradiance(i, receivers[i].point.xyz, receivers[i].normal.xyz);
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
    normal: vec3<f32>,
};

fn nearest_in_mesh(eye: vec3<f32>, dir: vec3<f32>, m: u32, best: Hit) -> Hit {
    var hit = best;
    let range = meshes[m];
    for (var t = range.x; t < range.x + range.y; t++) {
        let v0 = tris[3u * t].xyz;
        let v1 = tris[3u * t + 1u].xyz;
        let v2 = tris[3u * t + 2u].xyz;
        let d = ray_triangle(eye, dir, v0, v1, v2);
        // Ties (coplanar faces of different meshes) go to the lower mesh index,
        // so the result does not depend on traversal order.
        if (d > 0.0 && (d < hit.distance || (d == hit.distance && m < hit.mesh))) {
            hit = Hit(d, m, normalize(cross(v1 - v0, v2 - v0)));
        }
    }
    return hit;
}

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let aspect = camera.params.x / camera.params.y;
    let sx = ((f32(gid.x) + 0.5) / camera.params.x * 2.0 - 1.0) * camera.params.z * aspect;
    let sy = (1.0 - (f32(gid.y) + 0.5) / camera.params.y * 2.0) * camera.params.z;
    let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
    let eye = camera.eye.xyz;
    var hit = Hit(3.4e38, 0xffffffffu, vec3<f32>(0.0));
    if (light.counts.z == 1u) {
        let inv = 1.0 / dir;
        var stack: array<u32, 32>;
        var sp = 1u;
        stack[0] = 0u;
        while (sp > 0u) {
            sp--;
            let node = nodes[stack[sp]];
            // Slab test against the node's box, closer than the best hit so far.
            let t0 = (node.lo.xyz - eye) * inv;
            let t1 = (node.hi.xyz - eye) * inv;
            let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
            let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
            if (far < max(near, 0.0) || near > hit.distance) { continue; }
            if (node.info.y > 0u) {
                for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                    hit = nearest_in_mesh(eye, dir, items[k], hit);
                }
            } else {
                stack[sp] = node.info.x;
                stack[sp + 1u] = node.info.x + 1u;
                sp += 2u;
            }
        }
    } else {
        for (var m = 0u; m < light.counts.x; m++) {
            hit = nearest_in_mesh(eye, dir, m, hit);
        }
    }
    let best = hit.distance;
    let hit_mesh = hit.mesh;
    var hit_normal = hit.normal;
    let pixel = gid.y * w + gid.x;
    if (hit_mesh == 0xffffffffu) { colors[pixel] = vec4<f32>(0.0, 0.0, 0.0, 1.0); return; }
    if (dot(hit_normal, dir) > 0.0) { hit_normal = -hit_normal; }
    let point = eye + dir * best + hit_normal * 1e-4;
    let to_light = light.corner.xyz + (light.edge_u.xyz + light.edge_v.xyz) * 0.5 - point;
    var irradiance = 0.0;
    if (dot(hit_normal, to_light) > 0.0) {
        irradiance = cone_irradiance(pixel, point, hit_normal);
    }
    let c = albedo[hit_mesh].xyz * (camera.params.w * irradiance / 3.14159265);
    colors[pixel] = vec4<f32>(c, 1.0);
}
