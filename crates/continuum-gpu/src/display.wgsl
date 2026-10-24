// Draws the compute kernel's linear colors to the window: one fullscreen
// triangle; each fragment reads its pixel and applies the same tone curve as the
// offline images (Reinhard, then display gamma, done by the sRGB target or here).

struct Dims {
    size: vec2<u32>,
    // x: 1 if the target is not sRGB and gamma must be applied here
    flags: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> dims: Dims;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = min(vec2<u32>(pos.xy), dims.size - vec2<u32>(1u));
    let c = colors[p.y * dims.size.x + p.x].rgb;
    var t = c / (c + vec3<f32>(0.6));
    if (dims.flags.x == 1u) {
        t = pow(t, vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(t, 1.0);
}
