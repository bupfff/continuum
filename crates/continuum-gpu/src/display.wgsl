// Draws the compute kernel's linear colors to the window: one fullscreen
// triangle; each fragment reads its pixel, scales it by the exposure (by
// hand, times the auto-exposure meter's where on) and maps it to display
// light with the tone map (M17), then display gamma, done by the sRGB
// target or here.

struct Dims {
    // The rendered image's size.
    size: vec2<u32>,
    // x: 0 tone map (an sRGB target applies gamma), 1 tone map and gamma
    // here, 2 linear light (float targets, which an engine tone-maps);
    // y: the render scale (window pixels per rendered pixel; 0 means 1)
    flags: vec2<u32>,
    // x: the exposure, a factor on the colors before the tone map (0 means
    // 1); y: the tone map (0 Reinhard, 1 Khronos PBR Neutral, 2 AgX's
    // approximation); z: 1 to multiply by the meter's exposure.
    exposure: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> dims: Dims;
// The auto-exposure meter (`meter.wgsl`): its exposure first.
@group(0) @binding(2) var<storage, read> meter: array<f32>;

// Khronos PBR Neutral (the published reference implementation): linear
// display light.
fn pbr_neutral(c0: vec3<f32>) -> vec3<f32> {
    var c = c0;
    let x = min(c.r, min(c.g, c.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    c -= vec3<f32>(offset);
    let peak = max(c.r, max(c.g, c.b));
    if (peak < 0.76) { return c; }
    let d = 0.24;
    let new_peak = 1.0 - d * d / (peak + d - 0.76);
    c *= new_peak / peak;
    let g = 1.0 - 1.0 / (0.15 * (peak - new_peak) + 1.0);
    return mix(c, vec3<f32>(new_peak), g);
}

// AgX, Benjamin Wrensch's approximation of Troy Sobotka's original: the
// inset matrix, 16.5 stops in log2, a sixth-order fit of the default
// contrast curve, the outset matrix, then its 2.2 display power (linear
// display light).
fn agx(c: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    var v = inset * c;
    v = (clamp(log2(max(v, vec3<f32>(1e-10))), vec3<f32>(-12.47393), vec3<f32>(4.026069)) + 12.47393) / 16.499999;
    let x2 = v * v;
    let x4 = x2 * x2;
    v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
    v = outset * v;
    return pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.2));
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = min(vec2<u32>(pos.xy) / max(dims.flags.y, 1u), dims.size - vec2<u32>(1u));
    var e = select(1.0, dims.exposure.x, dims.exposure.x > 0.0);
    if (dims.exposure.z > 0.5) { e *= meter[0]; }
    let c = colors[p.y * dims.size.x + p.x].rgb * e;
    if (dims.flags.x == 2u) {
        return vec4<f32>(c, 1.0);
    }
    let mode = u32(dims.exposure.y + 0.5);
    var t: vec3<f32>;
    if (mode == 1u) {
        t = clamp(pbr_neutral(c), vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (mode == 2u) {
        t = clamp(agx(c), vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        t = c / (c + vec3<f32>(0.6));
    }
    if (dims.flags.x == 1u) {
        t = pow(t, vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(t, 1.0);
}
