// M13 gate 12 (development): the sub-pixel pass's large hits lit by the
// renderer's lighting cache. Appended to the hardware-ray module, whose own
// `g9_shade_hit`, `g10_shade`, `g9_sky`, `g9_scale` and corner hooks are
// renamed (`*_model`); see `MaskConeKernel::hw_vc_source`.

// The eye a hit's distance is measured from (a page bake's own eye).
var<private> g12_eye: vec3<f32> = vec3<f32>(0.0);
// Whether this invocation's shading asks the cache for pages.
var<private> g12_request: bool = false;

fn g9_shade_hit(m: u32, t: u32, dist: f32, e: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    g12_eye = e;
    g9_last_v = 1.0;
    return vc_hit_color(dir, hit_from(vec3<u32>(m, t, bitcast<u32>(dist)), dir), g12_request);
}

fn g10_shade(m: u32, t: u32, p: vec3<f32>, dist: f32, dir: vec3<f32>, ng: vec3<f32>, n: vec3<f32>, v_in: f32) -> vec3<f32> {
    g12_eye = camera.eye.xyz;
    return vc_hit_color(dir, hit_from(vec3<u32>(m, t, bitcast<u32>(dist)), dir), false);
}

fn g9_sky() -> vec3<f32> {
    return light.sky.xyz;
}

// Colours packed as square roots of a share of this: the model's scale with
// room for several lights, a sky and bounce light.
fn g9_scale() -> f32 {
    return 4.0 * g9_scale_model();
}

// A corner of the tile's sample pixel (as the cached shade pass picks it)
// asks the cache for the pages it shows.
fn g12_corner_request(gid: vec3<u32>) {
    let tile = vc.counts.z;
    let sample = (vc.counts.y * 7u) % (tile * tile);
    g12_request = gid.x % tile == sample % tile && gid.y % tile == sample / tile;
    vc_request_priority = 0u;
}

fn g12_corner_done() {
    if (g12_request && vc_request_priority > 0u) { vc_request(vc_request_key, vc_request_priority); }
    g12_request = false;
}

// Colour is always on here (no state word read: a storage binding fewer;
// the device allows 29 a stage).
fn g9_colour_on() -> bool {
    return true;
}

// The frames' stamp mirrored after the colour words (written by the page
// arguments' pass as it advances): passes read it without the pages' state
// (a storage binding fewer).
fn g12_stamp_word() -> u32 {
    return 28u * g9_p();
}

fn g8_stamp() -> u32 {
    return m13_plain[g12_stamp_word()] & 0x1fffu;
}

fn g12_stamp_advanced() {
    m13_plain[g12_stamp_word()] = atomicLoad(&g6_state[3]);
}
