// M13 gate 12 (development): the sub-pixel pass's large hits lit by the
// renderer's lighting cache. Appended to the hardware-ray module, whose own
// `g9_shade_hit`, `g10_shade`, `g9_sky`, `g9_scale`, `g9_colour_on`,
// `g8_stamp` and hooks are renamed (`*_model`); see
// `MaskConeKernel::hw_vc_source`.
//
// A pass that shades a large hit records it instead (the hit, where its ray
// went, where its colour goes) and a pass of its own lights the records from
// the cache: the cache's shading needs most of the device's 29 buffers a
// stage, too many beside a pass's own.

// The records: word 0 their count, word 1 the first shading pass's share;
// from word 8, five words each: object, triangle, distance, the ray (index
// in bits 0–23, bin in 24–27, a page request in 28, its kind in 30–31), the
// colour's word (bits 0–27) and target (28–29).
@group(0) @binding(105) var<storage, read_write> g12_hits: array<atomic<u32>>;
const G12_HITS: u32 = 3000000u;

// The eye a hit's distance is measured from.
var<private> g12_eye: vec3<f32> = vec3<f32>(0.0);
// Whether this invocation's hits ask the cache for pages.
var<private> g12_request: bool = false;
// The next shading's target (NONE: shaded here by the model, as page bakes
// from their own eye are), word and ray.
var<private> g12_to: u32 = 0xffffffffu;
var<private> g12_word: u32 = 0u;
var<private> g12_pos: u32 = 0u;

fn g12_target(to: u32, word: u32, kind: u32, index: u32, sub: u32) {
    g12_to = to;
    g12_word = word;
    g12_pos = index | (sub << 24u) | (kind << 30u);
}

fn g12_defer(m: u32, t: u32, dist: f32) {
    let i = atomicAdd(&g12_hits[0], 1u);
    if (i < G12_HITS) {
        let r = 8u + 5u * i;
        atomicStore(&g12_hits[r], m);
        atomicStore(&g12_hits[r + 1u], t);
        atomicStore(&g12_hits[r + 2u], bitcast<u32>(dist));
        atomicStore(&g12_hits[r + 3u], g12_pos | select(0u, 1u << 28u, g12_request));
        atomicStore(&g12_hits[r + 4u], g12_word | (g12_to << 28u));
    }
    g12_to = 0xffffffffu;
}

fn g9_shade_hit(m: u32, t: u32, dist: f32, e: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    if (g12_to == 0xffffffffu) { return g9_shade_hit_model(m, t, dist, e, dir); }
    g9_last_v = 1.0;
    g12_defer(m, t, dist);
    return vec3<f32>(0.0);
}

fn g10_shade(m: u32, t: u32, p: vec3<f32>, dist: f32, dir: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v_in: f32) -> vec3<f32> {
    if (g12_to == 0xffffffffu) { return g10_shade_model(m, t, p, dist, dir, ng, n, v_in); }
    g12_defer(m, t, dist);
    return vec3<f32>(0.0);
}

fn g9_sky() -> vec3<f32> {
    return light.sky.xyz;
}

// Colours packed as square roots of a share of this: the model's scale with
// room for several lights, a sky and bounce light.
fn g9_scale() -> f32 {
    return 4.0 * g9_scale_model();
}

// Colour is always on here (no state word read: a storage binding fewer).
fn g9_colour_on() -> bool {
    return true;
}

// A corner of the tile's sample pixel (as the cached shade pass picks it)
// asks the cache for the pages its hit shows.
fn g12_corner_request(gid: vec3<u32>) {
    let tile = vc.counts.z;
    let sample = (vc.counts.y * 7u) % (tile * tile);
    g12_request = gid.x % tile == sample % tile && gid.y % tile == sample / tile;
}

fn g12_corner_done() {
    g12_request = false;
}

// The frames' stamp mirrored after the colour words (written by the page
// arguments' pass as it advances): passes read it without the pages' state.
fn g12_stamp_word() -> u32 {
    return 28u * g9_p();
}

fn g8_stamp() -> u32 {
    return m13_plain[g12_stamp_word()] & 0x1fffu;
}

fn g12_stamp_advanced() {
    m13_plain[g12_stamp_word()] = atomicLoad(&g6_state[3]);
}

// The records' shading passes' arguments: the first takes the records so far
// (after the corners), the second the rest (after the shading of bins).
@compute @workgroup_size(1)
fn hw_m13_hits_args_a() {
    let n = min(atomicLoad(&g12_hits[0]), G12_HITS);
    atomicStore(&g12_hits[1], n);
    g10_args(4u, n);
}

@compute @workgroup_size(1)
fn hw_m13_hits_args_b() {
    let n = min(atomicLoad(&g12_hits[0]), G12_HITS);
    g10_args(8u, n - min(atomicLoad(&g12_hits[1]), n));
}

@compute @workgroup_size(64)
fn hw_m13_shade_hits_a(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = (wg.y * nwg.x + wg.x) * 64u + li;
    if (t >= atomicLoad(&g12_hits[1])) { return; }
    g12_shade_record(t);
}

@compute @workgroup_size(64)
fn hw_m13_shade_hits_b(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let first = atomicLoad(&g12_hits[1]);
    let t = first + (wg.y * nwg.x + wg.x) * 64u + li;
    if (t >= min(atomicLoad(&g12_hits[0]), G12_HITS)) { return; }
    g12_shade_record(t);
}

// Record `i` lit by the cache into its target.
fn g12_shade_record(i: u32) {
    let r = 8u + 5u * i;
    let m = atomicLoad(&g12_hits[r]);
    let t = atomicLoad(&g12_hits[r + 1u]);
    let dist = bitcast<f32>(atomicLoad(&g12_hits[r + 2u]));
    let pos = atomicLoad(&g12_hits[r + 3u]);
    let dest = atomicLoad(&g12_hits[r + 4u]);
    let w = u32(camera.params.x);
    let index = pos & 0xffffffu;
    let sub = (pos >> 24u) & 15u;
    let kind = pos >> 30u;
    // Where the ray went, as the pass that traced it aimed it.
    var q = vec2<f32>(f32(index % w), f32(index / w));
    if (kind == G12_AT_BIN) {
        q += (vec2<f32>(f32(sub % 4u), f32(sub / 4u)) + vec2<f32>(0.5)) * 0.25;
    } else if (kind == G12_AT_QUARTER) {
        q += (vec2<f32>(f32(sub % 4u), f32(sub / 4u)) + vec2<f32>(1.0)) * 0.25;
    } else if (kind == G12_AT_CENTRE) {
        q += vec2<f32>(0.5);
    }
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    let dir = normalize(camera.forward.xyz + camera.right.xyz * ((q.x / camera.params.x * 2.0 - 1.0) * tz * aspect) + camera.up.xyz * ((1.0 - q.y / camera.params.y * 2.0) * tz));
    g12_eye = camera.eye.xyz;
    let sample = ((pos >> 28u) & 1u) != 0u;
    vc_request_priority = 0u;
    g12_share0 = 1.0;
    g12_sky_value = -1.0;
    let c = vc_hit_color(dir, hit_from(vec3<u32>(m, t, bitcast<u32>(dist)), dir), sample);
    if (sample && vc_request_priority > 0u) { vc_request(vc_request_key, vc_request_priority); }
    // The colour into the record (its distance is spent), and the cache's
    // shares at the hit (its object is spent), for the scatter pass: the
    // lighting pass has no buffer to spare for the targets.
    atomicStore(&g12_hits[r + 2u], g10_pack(c, g9_scale()));
    atomicStore(&g12_hits[r], pack2x16float(vec2<f32>(g12_share0, g12_sky_value)));
}

// The records' colours to their targets (after each lighting pass).
@compute @workgroup_size(64)
fn hw_m13_scatter_hits_a(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = (wg.y * nwg.x + wg.x) * 64u + li;
    if (t >= atomicLoad(&g12_hits[1])) { return; }
    g12_scatter_record(t);
}

@compute @workgroup_size(64)
fn hw_m13_scatter_hits_b(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let first = atomicLoad(&g12_hits[1]);
    let t = first + (wg.y * nwg.x + wg.x) * 64u + li;
    if (t >= min(atomicLoad(&g12_hits[0]), G12_HITS)) { return; }
    g12_scatter_record(t);
}

fn g12_scatter_record(i: u32) {
    let r = 8u + 5u * i;
    let packed = atomicLoad(&g12_hits[r + 2u]);
    let dest = atomicLoad(&g12_hits[r + 4u]);
    let word = dest & 0xfffffffu;
    let to = dest >> 28u;
    let c = g10_unpack(packed, g9_scale());
    if (to == G12_TO_CORNER) {
        m13_plain[word] = g12_cpack(c);
        // The corner's shares, in its spare word.
        m13_plain[word + g9_p()] = atomicLoad(&g12_hits[r]);
    } else if (to == G12_TO_BIN) {
        m13_shaded[word] = packed | G10_SHADED_BIT;
    } else if (to == G12_TO_QUAD) {
        let v = packed | G10_SHADED_BIT;
        m13_shaded[word] = v;
        m13_shaded[word + 1u] = v;
        m13_shaded[word + 4u] = v;
        m13_shaded[word + 5u] = v;
    } else {
        let p = g9_p();
        m13_plain[word] = bitcast<u32>(c.x);
        m13_plain[word + p] = bitcast<u32>(c.y);
        m13_plain[word + 2u * p] = bitcast<u32>(c.z);
    }
}

// A small instance's ambient light: the sky's radiance over the open share
// of its points' hemispheres, as albedo × E / π with E = π L × open. The
// share is a sphere's resting on a plane, averaged over its surface (each
// point sees (1 + cos θ) / 2 of the sky): a half (development constant).
const G12_SKY_OPEN: f32 = 0.5;
fn g12_ambient(albedo: vec3<f32>) -> vec3<f32> {
    return albedo * light.sky.xyz * G12_SKY_OPEN;
}

fn g12_untarget() {
    g12_to = 0xffffffffu;
}

// The cache's shares at the last hit lit (`vc_light_terms`): the first
// light's shadowed share and the sky's cached value (-1: none, a guess).
var<private> g12_share0: f32 = 1.0;
var<private> g12_sky_value: f32 = -1.0;

// A bin of object `m` lit from its pixel's corners that show it: their
// cached shares (the first light's, the sky's) bilinearly at the bin, with
// the bin's own point, normal and albedo, as the cache's shading does with
// its texels' shares (one light; otherwise w 0: lit in full).
fn g12_lean_bin(m: u32, t: u32, p: vec3<f32>, dist: f32, dir: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, pixel: u32, b: u32) -> vec4<f32> {
    if (arrayLength(&lights) != 1u || !G12_LEAN) { return vec4<f32>(0.0); }
    let cws = g12_cws(pixel);
    let pp = g9_p();
    let co = vec4<u32>(m13_plain[cws.x + 3u * pp], m13_plain[cws.y + 3u * pp], m13_plain[cws.z + 3u * pp], m13_plain[cws.w + 3u * pp]);
    let mine = co == vec4<u32>(m);
    if (!any(mine)) { return vec4<f32>(0.0); }
    let fx = (f32(b % 4u) + 0.5) * 0.25;
    let fy = (f32(b / 4u) + 0.5) * 0.25;
    let wk = vec4<f32>((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy) * select(vec4<f32>(0.0), vec4<f32>(1.0), mine);
    let wsum = wk.x + wk.y + wk.z + wk.w;
    if (wsum < 1e-6) { return vec4<f32>(0.0); }
    var share = 0.0;
    var sky = 0.0;
    var guess = false;
    for (var k = 0u; k < 4u; k++) {
        if (wk[k] <= 0.0) { continue; }
        let sv = unpack2x16float(m13_plain[cws[k] + pp]);
        share += wk[k] * sv.x;
        if (sv.y < 0.0) { guess = true; }
        sky += wk[k] * sv.y;
    }
    share /= wsum;
    sky /= wsum;
    let point = p + ng * surface_offset(dist);
    let base = surface_albedo(m, t, p, dist, dir, ng);
    var e = lights[0].radiance.xyz * share * unshadowed_rect(point, n, lights[0]);
    if (guess) { e += sky_open(n); } else { e += sky_scale(n) * sky; }
    return vec4<f32>(base * e / 3.14159265, 1.0);
}

// Development: bins lit from their corners' shares.
const G12_LEAN: bool = true;
