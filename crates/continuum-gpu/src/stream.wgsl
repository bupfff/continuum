// Streaming feedback (M14 gate 1): after the camera's visibility pass, each
// pixel turns its hit's distance into the camera level its object's
// geometry needs, and the highest per geometry is kept.

struct StreamParams {
    // Rendered pixels across and down; objects; entries per geometry in
    // `errors` (the lighting level's, then each finer level's).
    width: u32,
    height: u32,
    objects: u32,
    levels: u32,
    // The largest error allowed per meter of distance: the pixel error times
    // one pixel's height per meter.
    per_meter: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: StreamParams;
@group(0) @binding(1) var<storage, read> vis: array<vec4<u32>>;
// Per object: its streamed geometry (NONE: none), and its scale's bits.
@group(0) @binding(2) var<storage, read> object_stream: array<vec2<u32>>;
// Per streamed geometry, `levels` errors in its own units, coarse to fine.
@group(0) @binding(3) var<storage, read> errors: array<f32>;
// Per streamed geometry: 1 plus the level wanted (0: not seen).
@group(0) @binding(4) var<storage, read_write> wanted: array<atomic<u32>>;

const STREAM_NONE: u32 = 0xffffffffu;

// One pixel of each 4×4: an object it misses is under four pixels across,
// where a level's error is rarely a visible share of it.
@compute @workgroup_size(8, 8)
fn stream_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = 4u * gid.xy;
    if (p.x >= params.width || p.y >= params.height) { return; }
    let v = vis[p.y * params.width + p.x];
    if (v.x == STREAM_NONE || v.x >= params.objects) { return; }
    let o = object_stream[v.x];
    if (o.x == STREAM_NONE) { return; }
    let allowed = params.per_meter * bitcast<f32>(v.z) / bitcast<f32>(o.y);
    var level = params.levels - 1u;
    for (var l = 0u; l < params.levels; l += 1u) {
        if (errors[o.x * params.levels + l] <= allowed) {
            level = l;
            break;
        }
    }
    if (atomicLoad(&wanted[o.x]) < level + 1u) {
        atomicMax(&wanted[o.x], level + 1u);
    }
}
