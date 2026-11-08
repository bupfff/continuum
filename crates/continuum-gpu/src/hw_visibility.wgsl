// Hardware camera rays: appended to `mask_cones.wgsl` (with `enable
// wgpu_ray_query;` in front) when the device supports ray queries. Writes the
// same visibility buffer as the compute `visibility` pass, so the compute
// shading passes run unchanged.

// One top-level instance per object: its transform, and the object index as
// custom data. Each object's geometry is a bottom-level structure whose
// primitive index is the triangle's index within the geometry.
@group(0) @binding(32) var scene_as: acceleration_structure;

// Camera rays see front faces only, as `camera_hit` does.
const HW_RAY_FLAGS: u32 = 0x1u | 0x10u; // FORCE_OPAQUE | CULL_BACK_FACING
// The same through candidates tested here: FORCE_NO_OPAQUE | CULL_BACK_FACING.
const HW_FRONT_FLAGS: u32 = 0x2u | 0x10u;

// Whether a camera ray along `dir` sees the front of object `mesh`'s
// triangle at primitive `primitive`.
fn hw_front(mesh: u32, primitive: u32, dir: vec3<f32>) -> bool {
    let t = meshes[mesh].x + primitive;
    let v0 = tris[3u * t].xyz;
    let n = normal_to_world(mesh, cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0));
    return dot(n, dir) < 0.0;
}

// Metal ignores CULL_BACK_FACING (naga 30 sets no triangle cull mode), so a
// committed hit can be a back face, and of a double-sided card's coincident
// front and back faces either one (M10 gate 8). A pixel whose hit is a back
// face is traced again here, its candidates tested for facing: the nearest
// front face, as `camera_hit` finds. Other pixels keep the opaque query.

@compute @workgroup_size(8, 8)
fn hw_visibility(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_as, RayDesc(HW_RAY_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
    rayQueryProceed(&rq);
    var hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind != 0u && !hw_front(hit.instance_custom_data, hit.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, scene_as, RayDesc(HW_FRONT_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        hit = rayQueryGetCommittedIntersection(&fq);
    }
    var mesh = 0xffffffffu;
    var tri = 0xffffffffu;
    var dist = 3.4e38;
    if (hit.kind != 0u) {
        mesh = hit.instance_custom_data;
        tri = meshes[mesh].x + hit.primitive_index;
        dist = hit.t;
    }
    vis[gid.y * w + gid.x] = vec4<u32>(mesh, tri, bitcast<u32>(dist), seen_light(eye, dir, dist));
}

// The movable objects' own top-level structure (M9): they are left out of
// `scene_as` and rebuilt here as they move.
@group(0) @binding(59) var movers_as: acceleration_structure;

// `hw_visibility` for scenes with movers: the nearer hit of the static
// structure and the movers'.
@compute @workgroup_size(8, 8)
fn hw_visibility_movers(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    var mesh = 0xffffffffu;
    var tri = 0xffffffffu;
    var dist = 3.4e38;
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_as, RayDesc(HW_RAY_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
    rayQueryProceed(&rq);
    var hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind != 0u && !hw_front(hit.instance_custom_data, hit.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, scene_as, RayDesc(HW_FRONT_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        hit = rayQueryGetCommittedIntersection(&fq);
    }
    if (hit.kind != 0u) {
        mesh = hit.instance_custom_data;
        tri = meshes[mesh].x + hit.primitive_index;
        dist = hit.t;
    }
    var mq: ray_query;
    rayQueryInitialize(&mq, movers_as, RayDesc(HW_RAY_FLAGS, 0xffu, 0.0, dist, eye, dir));
    rayQueryProceed(&mq);
    var moved = rayQueryGetCommittedIntersection(&mq);
    if (moved.kind != 0u && !hw_front(moved.instance_custom_data, moved.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, movers_as, RayDesc(HW_FRONT_FLAGS, 0xffu, 0.0, dist, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        moved = rayQueryGetCommittedIntersection(&fq);
    }
    if (moved.kind != 0u && moved.t < dist) {
        mesh = moved.instance_custom_data;
        tri = meshes[mesh].x + moved.primitive_index;
        dist = moved.t;
    }
    vis[gid.y * w + gid.x] = vec4<u32>(mesh, tri, bitcast<u32>(dist), seen_light(eye, dir, dist));
}

// The nearest front face along a camera ray through `scene_as`: object,
// triangle and distance bits (object NONE: a miss), back-face hits traced
// again as `hw_visibility` does.
fn hw_trace(eye: vec3<f32>, dir: vec3<f32>) -> vec3<u32> {
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_as, RayDesc(HW_RAY_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
    rayQueryProceed(&rq);
    var hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind != 0u && !hw_front(hit.instance_custom_data, hit.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, scene_as, RayDesc(HW_FRONT_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        hit = rayQueryGetCommittedIntersection(&fq);
    }
    if (hit.kind == 0u) { return vec3<u32>(0xffffffffu, 0xffffffffu, bitcast<u32>(3.4e38)); }
    let mesh = hit.instance_custom_data;
    return vec3<u32>(mesh, meshes[mesh].x + hit.primitive_index, bitcast<u32>(hit.t));
}

// After `hw_visibility` (M7 gate 4): an edge pixel (a neighbour sees
// another object or a depth 5% away, as `vc_shade`'s test) traces its four
// quarter-rays and is listed for `vc_shade_edges`, at the stats' last
// counter (which counts edge pixels, as the compute path does).
@compute @workgroup_size(8, 8)
fn hw_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h || !is_edge(gid, w, h)) { return; }
    let i = atomicAdd(&vc_stats[15], 1u);
    if (4u * i + 3u >= arrayLength(&edge_hits)) { return; }
    let pixel = gid.y * w + gid.x;
    let eye = camera.eye.xyz;
    for (var q = 0u; q < 4u; q++) {
        let sd = camera_dir_at(gid, (f32(q & 1u) - 0.5) * 0.5, (f32(q >> 1u) - 0.5) * 0.5);
        edge_hits[4u * i + q] = vec4<u32>(hw_trace(eye, sd), pixel);
    }
}
