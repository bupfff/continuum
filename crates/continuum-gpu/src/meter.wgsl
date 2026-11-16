// Auto-exposure (M17): a histogram of the frame's log2 luminance
// (`meter_histogram`), then the exposure that brings the mean of its
// middle (between two percentiles) to the key, adapted over time toward
// that (`meter_resolve`). The display multiplies colours by it.

struct MeterParams {
    size: vec2<u32>,
    pad: vec2<u32>,
    // x: the key (the luminance the trimmed mean is shown at); y: this
    // frame's adaptation, 1 − exp(−dt / τ) (1: at once); z, w: the
    // percentiles bounding the mean.
    p: vec4<f32>,
    // x, y: the exposure's bounds (factors).
    bounds: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>, 64>;
// 0: the exposure; 1: the frames metered.
@group(0) @binding(2) var<storage, read_write> meter: array<f32, 4>;
@group(0) @binding(3) var<uniform> params: MeterParams;

// 64 bins of half a stop over log2 luminance from −16 to +16.
fn bin_of(l: f32) -> u32 {
    if (l <= 0.0) { return 0u; }
    return u32(clamp((log2(l) + 16.0) * 2.0, 0.0, 63.0));
}

var<workgroup> local_hist: array<atomic<u32>, 64>;

@compute @workgroup_size(16, 16)
fn meter_histogram(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li < 64u) { atomicStore(&local_hist[li], 0u); }
    workgroupBarrier();
    if (gid.x < params.size.x && gid.y < params.size.y) {
        let c = colors[gid.y * params.size.x + gid.x].xyz;
        let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
        atomicAdd(&local_hist[bin_of(l)], 1u);
    }
    workgroupBarrier();
    if (li < 64u) {
        let v = atomicLoad(&local_hist[li]);
        if (v > 0u) { atomicAdd(&hist[li], v); }
    }
}

@compute @workgroup_size(1)
fn meter_resolve() {
    var total = 0u;
    for (var b = 0u; b < 64u; b++) { total += atomicLoad(&hist[b]); }
    let lo = f32(total) * params.p.z;
    let hi = f32(total) * params.p.w;
    var seen = 0.0;
    var weight = 0.0;
    var sum = 0.0;
    for (var b = 0u; b < 64u; b++) {
        let n = f32(atomicLoad(&hist[b]));
        // The part of this bin between the percentiles.
        let take = max(min(seen + n, hi) - max(seen, lo), 0.0);
        sum += take * ((f32(b) + 0.5) * 0.5 - 16.0);
        weight += take;
        seen += n;
        atomicStore(&hist[b], 0u);
    }
    var target_e = 1.0;
    if (weight > 0.0) { target_e = params.p.x / exp2(sum / weight); }
    target_e = clamp(target_e, params.bounds.x, params.bounds.y);
    var e = target_e;
    if (meter[1] > 0.0 && params.p.y < 1.0) {
        e = exp2(mix(log2(max(meter[0], 1e-12)), log2(target_e), params.p.y));
    }
    meter[0] = e;
    meter[1] += 1.0;
}
