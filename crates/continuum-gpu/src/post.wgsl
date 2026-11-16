// Bloom (M19): the image at a quarter of its size (each texel the mean of
// 4 × 4 pixels, `bloom_down`), blurred by a Gaussian across and down
// (`bloom_blur`, its weights renormalized where it meets the image's edge).
// The display mixes it in, energy kept: (1 − strength) of each pixel plus
// strength of the blur there.

struct BloomParams {
    // The image's size; the quarter's.
    size: vec2<u32>,
    quarter: vec2<u32>,
    // x: σ in quarter texels; y: taps each side; z: 0 across, 1 down.
    p: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: BloomParams;

@compute @workgroup_size(8, 8)
fn bloom_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = params.quarter;
    if (gid.x >= q.x || gid.y >= q.y) { return; }
    var sum = vec3<f32>(0.0);
    var n = 0.0;
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            let x = 4u * gid.x + i;
            let y = 4u * gid.y + j;
            if (x < params.size.x && y < params.size.y) {
                sum += colors[y * params.size.x + x].xyz;
                n += 1.0;
            }
        }
    }
    dst[gid.y * q.x + gid.x] = vec4<f32>(sum / max(n, 1.0), 1.0);
}

@compute @workgroup_size(8, 8)
fn bloom_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = params.quarter;
    if (gid.x >= q.x || gid.y >= q.y) { return; }
    let sigma = params.p.x;
    let taps = i32(params.p.y);
    let down = params.p.z > 0.5;
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var k = -taps; k <= taps; k++) {
        var x = i32(gid.x);
        var y = i32(gid.y);
        if (down) { y += k; } else { x += k; }
        if (x < 0 || y < 0 || x >= i32(q.x) || y >= i32(q.y)) { continue; }
        let w = exp(-0.5 * f32(k * k) / (sigma * sigma));
        sum += w * src[u32(y) * q.x + u32(x)].xyz;
        weight += w;
    }
    dst[gid.y * q.x + gid.x] = vec4<f32>(sum / max(weight, 1e-12), 1.0);
}
