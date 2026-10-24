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
    // x: mesh count; y: 1 to write per-bin coverage to `debug`
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
fn draw_edge(acc: ptr<function, array<f32, 68>>, q0: vec2<f32>, q1: vec2<f32>) {
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

fn rasterize(c: Cone, acc: ptr<function, array<f32, 68>>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> bool {
    var p: Poly3;
    p.v[0] = v0;
    p.v[1] = v1;
    p.v[2] = v2;
    p.n = 3u;
    p = clip_plane3(p, c.far_normal, c.far_offset);
    for (var i = 0u; i < p.n; i++) { p.v[i] = to_local(c, p.v[i]); }
    // Keep z >= NEAR, i.e. dot(-z, x) <= -NEAR.
    p = clip_plane3(p, vec3<f32>(0.0, 0.0, -1.0), -NEAR);
    if (p.n < 3u) { return false; }
    let su = f32(GRID) / (c.window.z - c.window.x);
    let sv = f32(GRID) / (c.window.w - c.window.y);
    var q: Poly2;
    q.n = p.n;
    for (var i = 0u; i < p.n; i++) {
        q.v[i] = vec2<f32>((p.v[i].x / p.v[i].z - c.window.x) * su, (p.v[i].y / p.v[i].z - c.window.y) * sv);
    }
    let g = f32(GRID);
    q = clip_line2(q, vec2<f32>(-1.0, 0.0), 0.0); // x >= 0
    q = clip_line2(q, vec2<f32>(1.0, 0.0), g);    // x <= g
    q = clip_line2(q, vec2<f32>(0.0, -1.0), 0.0); // y >= 0
    q = clip_line2(q, vec2<f32>(0.0, 1.0), g);    // y <= g
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
        draw_edge(acc, a, b);
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&receivers)) { return; }
    let apex = receivers[i].point.xyz;
    let normal = receivers[i].normal.xyz;

    var forward = normalize(cross(light.edge_u.xyz, light.edge_v.xyz));
    if (dot(forward, light.corner.xyz - apex) < 0.0) { forward = -forward; }
    let dist = dot(forward, light.corner.xyz - apex);
    if (dist <= 1e-6) { result[i] = 0.0; return; }
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

    var weights: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) {
        let col = f32(b % GRID);
        let row = f32(b / GRID);
        let g = f32(GRID);
        let d0 = direction(c, col / g, row / g);
        let d1 = direction(c, (col + 1.0) / g, row / g);
        let d2 = direction(c, (col + 1.0) / g, (row + 1.0) / g);
        let d3 = direction(c, col / g, (row + 1.0) / g);
        weights[b] = abs(0.5 * (lambert_edge(normal, d0, d1) + lambert_edge(normal, d1, d2)
            + lambert_edge(normal, d2, d3) + lambert_edge(normal, d3, d0)));
    }

    // Explicit zeroing: don't rely on loop-scoped `var`s being re-initialized.
    var cov: array<f32, 64>;
    for (var b = 0u; b < BINS; b++) { cov[b] = 0.0; }
    var acc: array<f32, 68>;
    for (var m = 0u; m < light.counts.x; m++) {
        let range = meshes[m];
        for (var b = 0u; b < 68u; b++) { acc[b] = 0.0; }
        var touched = false;
        for (var t = range.x; t < range.x + range.y; t++) {
            let v0 = tris[3u * t].xyz;
            let v1 = tris[3u * t + 1u].xyz;
            let v2 = tris[3u * t + 2u].xyz;
            if (dot(cross(v1 - v0, v2 - v0), apex - v0) <= 0.0) { continue; }
            if (!may_touch(c, v0, v1, v2)) { continue; }
            if (rasterize(c, &acc, v0, v1, v2)) { touched = true; }
        }
        if (light.counts.y == 2u) {
            for (var b = 0u; b < 64u; b++) { debug[i * BINS + b] = acc[b]; }
        }
        if (touched) {
            var running = 0.0;
            for (var b = 0u; b < BINS; b++) {
                running += acc[b];
                let covered = abs(running);
                if (covered > 1e-6) { cov[b] = max(cov[b], min(covered, 1.0)); }
            }
        }
    }
    var visible = 0.0;
    for (var b = 0u; b < BINS; b++) {
        visible += (1.0 - cov[b]) * weights[b];
    }
    result[i] = visible;
    if (light.counts.y == 1u) {
        for (var b = 0u; b < BINS; b++) { debug[i * BINS + b] = cov[b]; }
    }
}
