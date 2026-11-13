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

// The triangle a camera ray's hit names: object `mesh`'s geometry at its
// resident camera level (M14), whose bottom-level structure has one
// geometry per pool page; the transform's spare word holds, flagged, the
// level's page-table entry (0: none, the lighting level), and the table
// (after the objects' ranges) each page's first triangle.
fn cam_tri(mesh: u32, geometry: u32, primitive: u32) -> u32 {
    let x = meshes[mesh].w;
    if (x != NONE) {
        let word = bitcast<u32>(xforms[4u * x + 3u].w);
        if ((word & 0x40000000u) != 0u) {
            let e = (word & 0x3fffffffu) + geometry;
            return meshes[e / 4u][e % 4u] + primitive;
        }
    }
    return meshes[mesh].x + primitive;
}

// The object and geometry a hit names (M14 gate 6): an instance's custom
// data is its object, or on a cluster of proxies the word where the
// cluster's member list starts (one member per geometry); a member shows
// its proxy, its only geometry.
fn hw_object(custom: u32, geometry: u32) -> vec2<u32> {
    if ((custom & 0x800000u) != 0u) {
        let w = (custom & 0x7fffffu) + geometry;
        return vec2<u32>(meshes[w / 4u][w % 4u], 0u);
    }
    return vec2<u32>(custom, geometry);
}

// Whether a camera ray along `dir` sees the front of the triangle a hit
// names (its instance's custom data, `geometry`, `primitive`).
fn hw_front(custom: u32, geometry: u32, primitive: u32, dir: vec3<f32>) -> bool {
    let o = hw_object(custom, geometry);
    let t = cam_tri(o.x, o.y, primitive);
    let v0 = tris[3u * t].xyz;
    let n = normal_to_world(o.x, cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0));
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
    if (hit.kind != 0u && !hw_front(hit.instance_custom_data, hit.geometry_index, hit.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, scene_as, RayDesc(HW_FRONT_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.geometry_index, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        hit = rayQueryGetCommittedIntersection(&fq);
    }
    var mesh = 0xffffffffu;
    var tri = 0xffffffffu;
    var dist = 3.4e38;
    if (hit.kind != 0u) {
        let o = hw_object(hit.instance_custom_data, hit.geometry_index);
        mesh = o.x;
        tri = cam_tri(o.x, o.y, hit.primitive_index);
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
    if (hit.kind != 0u && !hw_front(hit.instance_custom_data, hit.geometry_index, hit.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, scene_as, RayDesc(HW_FRONT_FLAGS, 0xffu, 0.0, 3.4e38, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.geometry_index, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        hit = rayQueryGetCommittedIntersection(&fq);
    }
    if (hit.kind != 0u) {
        let o = hw_object(hit.instance_custom_data, hit.geometry_index);
        mesh = o.x;
        tri = cam_tri(o.x, o.y, hit.primitive_index);
        dist = hit.t;
    }
    var mq: ray_query;
    rayQueryInitialize(&mq, movers_as, RayDesc(HW_RAY_FLAGS, 0xffu, 0.0, dist, eye, dir));
    rayQueryProceed(&mq);
    var moved = rayQueryGetCommittedIntersection(&mq);
    if (moved.kind != 0u && !hw_front(moved.instance_custom_data, moved.geometry_index, moved.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, movers_as, RayDesc(HW_FRONT_FLAGS, 0xffu, 0.0, dist, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.geometry_index, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        moved = rayQueryGetCommittedIntersection(&fq);
    }
    if (moved.kind != 0u && moved.t < dist) {
        mesh = moved.instance_custom_data;
        tri = cam_tri(mesh, moved.geometry_index, moved.primitive_index);
        dist = moved.t;
    }
    vis[gid.y * w + gid.x] = vec4<u32>(mesh, tri, bitcast<u32>(dist), seen_light(eye, dir, dist));
}

// The nearest front face along a camera ray through `scene_as` beyond
// `t_min`: object, triangle and distance bits (object NONE: a miss),
// back-face hits traced again as `hw_visibility` does.
fn hw_trace(eye: vec3<f32>, dir: vec3<f32>, t_min: f32) -> vec3<u32> {
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_as, RayDesc(HW_RAY_FLAGS, 0xffu, t_min, 3.4e38, eye, dir));
    rayQueryProceed(&rq);
    var hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind != 0u && !hw_front(hit.instance_custom_data, hit.geometry_index, hit.primitive_index, dir)) {
        var fq: ray_query;
        rayQueryInitialize(&fq, scene_as, RayDesc(HW_FRONT_FLAGS, 0xffu, t_min, 3.4e38, eye, dir));
        while (rayQueryProceed(&fq)) {
            let c = rayQueryGetCandidateIntersection(&fq);
            if (hw_front(c.instance_custom_data, c.geometry_index, c.primitive_index, dir)) { rayQueryConfirmIntersection(&fq); }
        }
        hit = rayQueryGetCommittedIntersection(&fq);
    }
    if (hit.kind == 0u) { return vec3<u32>(0xffffffffu, 0xffffffffu, bitcast<u32>(3.4e38)); }
    let o = hw_object(hit.instance_custom_data, hit.geometry_index);
    return vec3<u32>(o.x, cam_tri(o.x, o.y, hit.primitive_index), bitcast<u32>(hit.t));
}

// The opacity at world point `p` on triangle `t` of object `m`: 1 unless
// alpha-blended (M10 gate 16; its factor in its material's second word's
// z, w set), then the factor times its base color texture's alpha.
fn hw_opacity(m: u32, t: u32, p: vec3<f32>) -> f32 {
    let mat = materials[2u * m + 1u];
    if (mat.w == 0u) { return 1.0; }
    var a = bitcast<f32>(mat.z);
    let layer = object_texture[m];
    if (layer != NONE) {
        let b = barycentric(m, t, p);
        let uv = tri_tex[3u * t] * b.x + tri_tex[3u * t + 1u] * b.y + tri_tex[3u * t + 2u] * b.z;
        let size = f32(textureDimensions(base_color).x);
        a *= textureSampleLevel(base_color, base_sampler, wrap_uv(m, SLOT_BASE, uv, size, 0.0), i32(layer), 0.0).w;
    }
    return clamp(a, 0.0, 1.0);
}

// Appends a record to the extra samples' list for `vc_shade_edges`, at the
// stats' last counter, five words a record: up to four samples (object,
// triangle, distance bits, weight bits), then the pixel, its own sample's
// weight bits, the samples' count, and each sample's quarter in four bits
// (15: the pixel's centre).
fn list_extra(pixel: u32, samples: array<vec4<u32>, 4>, n: u32, own_weight: f32, quarters: u32) {
    let i = atomicAdd(&vc_stats[15], 1u);
    if (5u * i + 4u >= arrayLength(&edge_hits)) { return; }
    for (var j = 0u; j < n; j++) { edge_hits[5u * i + j] = samples[j]; }
    edge_hits[5u * i + 4u] = vec4<u32>(pixel, bitcast<u32>(own_weight), n, quarters);
}

// The layers behind camera hit `own` (object, triangle, distance bits) if
// it is alpha-blended (M10 gate 16), listed for `vc_shade_edges`: the ray
// traces on through blended surfaces, front faces only, each layer weighing
// its opacity times what the layers before let through; the fourth takes
// what is left. Whether the pixel was listed.
fn list_layers(pixel: u32, eye: vec3<f32>, dir: vec3<f32>, own: vec3<u32>) -> bool {
    if (own.x == NONE) { return false; }
    let first = hw_opacity(own.x, own.y, eye + dir * bitcast<f32>(own.z));
    if (first >= 1.0) { return false; }
    var samples: array<vec4<u32>, 4>;
    var n = 0u;
    var through = 1.0 - first;
    var t = bitcast<f32>(own.z);
    loop {
        if (n >= 4u || through <= 1e-4) { break; }
        let hit = hw_trace(eye, dir, t * (1.0 + 1e-5) + 1e-4);
        var a = 1.0;
        if (hit.x != NONE && n < 3u) { a = hw_opacity(hit.x, hit.y, eye + dir * bitcast<f32>(hit.z)); }
        let weight = through * a;
        samples[n] = vec4<u32>(hit, bitcast<u32>(weight));
        n++;
        through -= weight;
        if (hit.x == NONE || a >= 1.0) { break; }
        t = bitcast<f32>(hit.z);
    }
    list_extra(pixel, samples, n, first, 0xffffu);
    return true;
}

// `hw_visibility` with alpha blending (M10 gate 19): a pixel whose hit is
// blended lists its layers here, in the pass that found the hit, rather
// than in a full-screen pass after it.
@compute @workgroup_size(8, 8)
fn hw_visibility_blend(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let eye = camera.eye.xyz;
    let hit = hw_trace(eye, dir, 0.0);
    let pixel = gid.y * w + gid.x;
    vis[pixel] = vec4<u32>(hit, seen_light(eye, dir, bitcast<f32>(hit.z)));
    list_layers(pixel, eye, dir, hit);
}

// After `hw_visibility` (or `hw_visibility_blend`), with silhouette rays
// (`vc.indirect.w` 2 or 3, M7 gate 4): an edge pixel (a neighbour sees
// another object or a depth 5% away) traces four quarter-rays (or one
// diagonal pair, averaged with its own ray), listed for `vc_shade_edges`.
// A pixel whose own hit is blended was listed with its layers instead.
@compute @workgroup_size(8, 8)
fn hw_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    if (vc.indirect.w != 2u && vc.indirect.w != 3u) { return; }
    let pixel = gid.y * w + gid.x;
    let eye = camera.eye.xyz;
    let own = vis[pixel];
    let dir = camera_dir(gid);
    if (own.x != NONE && hw_opacity(own.x, own.y, eye + dir * bitcast<f32>(own.z)) < 1.0) { return; }
    if (!is_edge(gid, w, h)) { return; }
    // Two samples (`vc.indirect.w` 3): one diagonal pair of quarters,
    // alternating from pixel to pixel, with the pixel's own ray; four
    // otherwise.
    let two = vc.indirect.w == 3u;
    let n = select(4u, 2u, two);
    let weight = select(0.25, 1.0 / 3.0, two);
    let own_weight = select(0.0, 1.0 / 3.0, two);
    let flip = ((gid.x + gid.y) & 1u) == 1u;
    var samples: array<vec4<u32>, 4>;
    var quarters = 0u;
    for (var j = 0u; j < n; j++) {
        var q = j;
        if (two) { q = select(select(0u, 3u, j == 1u), select(1u, 2u, j == 1u), flip); }
        let sd = camera_dir_at(gid, (f32(q & 1u) - 0.5) * 0.5, (f32(q >> 1u) - 0.5) * 0.5);
        samples[j] = vec4<u32>(hw_trace(eye, sd, 0.0), bitcast<u32>(weight));
        quarters |= q << (4u * j);
    }
    quarters |= (0xffffu << (4u * n)) & 0xffffu;
    list_extra(pixel, samples, n, own_weight, quarters);
}

// M13 gate 4: per pixel, a ray through its top-left corner, recording
// (the object hit or NONE, 1 if that object is large for the coverage maps,
// its distance's bits, 0) for `m13_coverage_near`.
@compute @workgroup_size(8, 8)
fn hw_m13_corners(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    let sx = (f32(gid.x) / camera.params.x * 2.0 - 1.0) * tz * aspect;
    let sy = (1.0 - f32(gid.y) / camera.params.y * 2.0) * tz;
    let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
    let hit = hw_trace(camera.eye.xyz, dir, 0.0);
    var large = 0u;
    if (hit.x != NONE && m13_large(hit.x)) { large = 1u; }
    vis[gid.y * w + gid.x] = vec4<u32>(hit.x, large, hit.z, 0u);
}

// After `hw_m13_corners` (M13 gate 4): a pixel whose four corners hit the
// same large object is wholly covered by it. Every other pixel casts a
// hardware ray through each of its 4×4 bins' centres: a bin whose ray meets
// a large object is covered; then the walk adds the small instances (their
// maps) and far object nodes, leaving large instances out. Writes
// (coverage, object nodes, triangle nodes): no nodes for a pixel its
// corners resolved.
@compute @workgroup_size(8, 8)
fn hw_m13_near(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let pixel = gid.y * w + gid.x;
    if (gid.x + 1u < w && gid.y + 1u < h) {
        let a = vis[pixel];
        let b = vis[pixel + 1u];
        let c = vis[pixel + w];
        let d = vis[pixel + w + 1u];
        if (a.x != NONE && a.y == 1u && b.x == a.x && c.x == a.x && d.x == a.x) {
            colors[pixel] = vec4<f32>(1.0, 0.0, 0.0, 1.0);
            return;
        }
    }
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    for (var k = 0u; k < 16u; k++) {
        let p = vec2<f32>(f32(gid.x), f32(gid.y)) + (vec2<f32>(f32(k % 4u), f32(k / 4u)) + vec2<f32>(0.5)) * 0.25;
        let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
        let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
        let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
        let hit = hw_trace(camera.eye.xyz, dir, 0.0);
        pbin[k] = select(0.0, 1.0, hit.x != NONE && m13_large(hit.x));
    }
    colors[pixel] = vec4<f32>(map_coverage_from(gid, true), 1.0);
}


// Bounce light's rays through hardware ray queries (M14 gate 10): the
// nearest hit among the camera's structures (camera levels, clusters of
// proxies), so a ray never needs a lighting level. Where the sample's own
// object shows a camera level or a proxy, hits on it are skipped: those
// surfaces differ from the lighting level the sample sits on by up to the
// level's error, so near hits would be its own surface.
fn hw_bounce_hit(eye: vec3<f32>, dir: vec3<f32>, own: u32) -> Hit {
    var hit = Hit(3.4e38, NONE, NONE, vec3<f32>(0.0));
    let x = meshes[own].w;
    let streamed = x != NONE && (bitcast<u32>(xforms[4u * x + 3u].w) & 0x40000000u) != 0u;
    var rq: ray_query;
    if (streamed) {
        // Through candidates: FORCE_NO_OPAQUE.
        rayQueryInitialize(&rq, scene_as, RayDesc(0x2u, 0xffu, 0.0, 3.4e38, eye, dir));
        while (rayQueryProceed(&rq)) {
            let c = rayQueryGetCandidateIntersection(&rq);
            if (hw_object(c.instance_custom_data, c.geometry_index).x != own) {
                rayQueryConfirmIntersection(&rq);
            }
        }
    } else {
        // FORCE_OPAQUE: the nearest face, front or back (the gather skips
        // back faces, as the compute walk's hits).
        rayQueryInitialize(&rq, scene_as, RayDesc(0x1u, 0xffu, 0.0, 3.4e38, eye, dir));
        rayQueryProceed(&rq);
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind != 0u) {
        let o = hw_object(h.instance_custom_data, h.geometry_index);
        let t = cam_tri(o.x, o.y, h.primitive_index);
        let v0 = tris[3u * t].xyz;
        let n = normalize(normal_to_world(o.x, cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0)));
        hit = Hit(h.t, o.x, t, select(n, -n, dot(n, dir) > 0.0));
    }
    return hit;
}

// `vc_bake_indirect` with its rays through hardware ray queries (M14 gate
// 10).
@compute @workgroup_size(64)
fn hw_vc_bake_indirect(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    if (li == 0u) { atomicStore(&ind_lit_missing, NONE); }
    workgroupBarrier();
    let job = bounce_job(wg);
    var sum = array<vec3<f32>, 5>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var missing = 0.0;
    if (job.listed && job.surf.ok) {
        for (var b = li; b < IND_BINS; b += 64u) {
            let dir = normalize(bin_world(job.surf.normal, bin_local(b, IND_R)));
            let hit = hw_bounce_hit(job.surf.apex, dir, job.e.x);
            missing += bounce_gather(job.surf.apex, dir, hit, hemicube_weight(b), &sum, job.bounces);
        }
    }
    bounce_finish(li, job, sum, missing);
}

// Whether a ray from `eye` along `dir` meets nothing (FORCE_OPAQUE |
// TERMINATE_ON_FIRST_HIT: visibility only).
fn hw_sky_open(eye: vec3<f32>, dir: vec3<f32>) -> bool {
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_as, RayDesc(0x5u, 0xffu, 0.0, 3.4e38, eye, dir));
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq).kind == 0u;
}

// Sky samples through hardware ray queries (M5 gate 13): each of a sky
// sample's six jobs traces 32 of the hemicube's 192 bins around the texel's
// normal, one ray through each bin's center, and adds the projected solid
// angle of the bins whose ray meets nothing (a uniform sky's light per unit
// radiance, as `vc_bake_sky`'s faces add theirs).
@compute @workgroup_size(64)
fn hw_vc_bake_sky(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let count = vc.sky.x + appended_sky();
    let j = gid.x;
    let listed = j < 6u * count;
    let f = min(j / max(count, 1u), 5u);
    let k = j % max(count, 1u);
    let sample = sky_samples[k];
    let t = page_texel(sample.x, sample.y);
    let s = vc_surface(t.m, t.c, t.n, t.texel);
    let on = listed && t.inside && s.ok;
    var value = 0.0;
    if (on) {
        for (var q = 0u; q < 32u; q++) {
            let b = 32u * f + q;
            let dir = normalize(bin_world(s.normal, bin_local(b, IND_R)));
            // The uniform sky lies above the horizon.
            if (dir.z > 0.0 && hw_sky_open(s.apex, dir)) { value += hemicube_weight(b); }
        }
    }
    if (listed) { sky_partial[6u * k + f] = select(0.0, value, on); }
}

// --- The error while flying, reported (M4 gate 13). ---

// Whether the segment from `p` along unit `w` for `t` meets nothing
// (FORCE_OPAQUE | TERMINATE_ON_FIRST_HIT).
fn hw_segment_open(p: vec3<f32>, w: vec3<f32>, t: f32) -> bool {
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_as, RayDesc(0x5u, 0xffu, 0.0, t, p, w));
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq).kind == 0u;
}

// Per lane: each light's visible and whole weight, then the sky's open
// weight (17 values a lane).
var<workgroup> err_red: array<f32, 1088>;

// One workgroup per probe pixel, before `vc_error_probes`: each light's
// shadowed share (the first eight) and the sky's at the pixel's own point,
// into the probe's words 0–8, by hardware rays spread over the lanes. A
// light's share is the visible part of its irradiance over 64 points (an
// 8×8 grid on a rectangle, or across a sphere's disk), one ray each; the
// sky's is `hw_vc_bake_sky`'s (the projected solid angle of the hemicube's
// 192 bins whose ray meets nothing), three bins a lane. One where the light
// is below the horizon, as the bake.
@compute @workgroup_size(64)
fn hw_vc_error_shares(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let k = wg.x;
    let listed = k < vc.extra.z;
    let gp = err_probe_pixel(k);
    let w = u32(camera.params.x);
    let pixel = gp.y * w + gp.x;
    let dir = camera_dir(gp);
    let hit = recorded_hit(pixel, dir);
    let surf = listed && hit.mesh != NONE && vis[pixel].w == 0u;
    let surface = camera.eye.xyz + dir * hit.distance;
    let p = surface + hit.normal * surface_offset(hit.distance);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (surf) { n = shading_normal(hit.mesh, hit.tri, surface, -dir); }
    let count = min(arrayLength(&lights), 8u);
    for (var i = 0u; i < 17u; i++) { err_red[17u * li + i] = 0.0; }
    if (surf) {
        let gu = (f32(li % 8u) + 0.5) / 8.0;
        let gv = (f32(li / 8u) + 0.5) / 8.0;
        for (var l = 0u; l < count; l++) {
            let r = lights[l];
            var y = r.corner.xyz + r.edge_u.xyz * gu + r.edge_v.xyz * gv;
            var facing = 1.0;
            if (is_sphere(r)) {
                // Across the disk the sphere shows the point (a square grid
                // mapped to the disk).
                let c = r.corner.xyz;
                let to = normalize(c - p);
                let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(to.z) < 0.9);
                let a = normalize(cross(helper, to));
                let b = cross(to, a);
                let q = vec2<f32>(gu, gv) * 2.0 - vec2<f32>(1.0);
                let ang = 6.2831853 * gv;
                y = c + (a * cos(ang) + b * sin(ang)) * (r.edge_u.x * sqrt(gu));
                facing = 1.0 + 0.0 * q.x;
            } else {
                facing = abs(dot(normalize(cross(r.edge_u.xyz, r.edge_v.xyz)), normalize(y - p)));
            }
            let to_y = y - p;
            let t = length(to_y);
            let wy = to_y / max(t, 1e-20);
            let g = max(dot(n, wy), 0.0) * facing / max(t * t, 1e-20);
            if (g > 0.0) {
                err_red[17u * li + 2u * l + 1u] = g;
                if (hw_segment_open(p, wy, t * (1.0 - 1e-4))) { err_red[17u * li + 2u * l] = g; }
            }
        }
        if (vc.sky.y == 1u) {
            var sky = 0.0;
            for (var q = 0u; q < 3u; q++) {
                let b = li + 64u * q;
                let d = normalize(bin_world(n, bin_local(b, IND_R)));
                if (d.z > 0.0 && hw_sky_open(p, d)) { sky += hemicube_weight(b); }
            }
            err_red[17u * li + 16u] = sky;
        }
    }
    workgroupBarrier();
    if (li == 0u && listed) {
        var sums: array<f32, 17>;
        for (var i = 0u; i < 17u; i++) { sums[i] = 0.0; }
        for (var j = 0u; j < 64u; j++) {
            for (var i = 0u; i < 17u; i++) { sums[i] += err_red[17u * j + i]; }
        }
        for (var l = 0u; l < 8u; l++) {
            let whole = sums[2u * l + 1u];
            err_probes[16u * k + l] = select(1.0, sums[2u * l] / max(whole, 1e-30), whole > 0.0);
        }
        err_probes[16u * k + 8u] = sums[16];
    }
}

// --- M13 gate 5: the coverage pass with exact culling, bin rays only near
// large objects, one walk per tile and the maps through a texture. ---

@group(0) @binding(85) var cov_tex: texture_2d_array<f32>;
@group(0) @binding(86) var cov_sampler: sampler;

// `instance_map_bins` reading the maps' texture (8 bits, filtered by the
// hardware): object `m`, at most MAP_LIMIT_PX across, into the pixel's
// bins. False if it is too large or too near.
fn instance_map_tex(m: u32, hdr: u32, pix: vec2<f32>) -> bool {
    let x = meshes[m].w;
    let c_obj = vec3<f32>(bitcast<f32>(cov_maps[hdr + 2u]), bitcast<f32>(cov_maps[hdr + 3u]), bitcast<f32>(cov_maps[hdr + 4u]));
    let r_obj = bitcast<f32>(cov_maps[hdr + 5u]);
    let scale = select(1.0, xforms[4u * x + 3u].x, x != NONE);
    let centre = m13_to_world(x, c_obj);
    let radius = r_obj * scale;
    let rel = centre - camera.eye.xyz;
    let depth = dot(rel, camera.forward.xyz);
    if (depth - radius <= 1e-2) { return false; }
    let f = camera.params.y / (2.0 * camera.params.z);
    let rad = radius * f / depth;
    if (2.0 * rad > MAP_LIMIT_PX) { return false; }
    let cp = cam_to_pixels(to_camera(view_world(), centre));
    if (cp.x + rad <= pix.x || cp.y + rad <= pix.y || cp.x - rad >= pix.x + 1.0 || cp.y - rad >= pix.y + 1.0) { return true; }
    let dw = normalize(rel);
    var d = dw;
    if (x != NONE) { d = normalize(rotate_to_object(x, dw)); }
    let g = (oct_encode(d) + vec2<f32>(1.0)) * 0.5 * f32(MAP_DIRS) - vec2<f32>(0.5);
    let i0 = floor(g);
    let fr = g - i0;
    let texels_per_bin = 0.25 * depth / f / (2.0 * radius / f32(MAP_TEXELS));
    let level = clamp(round(log2(max(texels_per_bin, 1e-6))), 0.0, 4.0);
    let first = (hdr - 1u) / 8u * MAP_DIRS * MAP_DIRS;
    var layers: array<u32, 4>;
    var weights: array<f32, 4>;
    var us: array<vec3<f32>, 4>;
    var vs: array<vec3<f32>, 4>;
    for (var k = 0u; k < 4u; k++) {
        let ix = u32(clamp(i32(i0.x) + i32(k & 1u), 0, i32(MAP_DIRS) - 1));
        let iy = u32(clamp(i32(i0.y) + i32(k >> 1u), 0, i32(MAP_DIRS) - 1));
        layers[k] = first + iy * MAP_DIRS + ix;
        weights[k] = select(1.0 - fr.x, fr.x, (k & 1u) == 1u) * select(1.0 - fr.y, fr.y, (k >> 1u) == 1u);
        let dk = oct_decode((vec2<f32>(f32(ix), f32(iy)) + vec2<f32>(0.5)) / f32(MAP_DIRS) * 2.0 - vec2<f32>(1.0));
        let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(dk.z) < 0.999);
        us[k] = normalize(cross(helper, dk));
        vs[k] = cross(dk, us[k]);
    }
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    for (var b = 0u; b < 16u; b++) {
        let p = bin_lo(pix, b) + vec2<f32>(0.125);
        if (abs(p.x - cp.x) > rad || abs(p.y - cp.y) > rad) { continue; }
        let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
        let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
        let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
        let qw = camera.eye.xyz + dir * dot(rel, dir) - centre;
        var q = qw / r_obj;
        if (x != NONE) { q = rotate_to_object(x, qw) / (scale * r_obj); }
        var cov = 0.0;
        for (var k = 0u; k < 4u; k++) {
            if (weights[k] <= 0.0) { continue; }
            let uv = (vec2<f32>(dot(q, us[k]), dot(q, vs[k])) + vec2<f32>(1.0)) * 0.5;
            if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { continue; }
            cov += weights[k] * textureSampleLevel(cov_tex, cov_sampler, uv, layers[k], level).x;
        }
        pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - clamp(cov, 0.0, 1.0));
    }
    return true;
}

// Entries a tile's walk lists per chunk: screen rectangles, and (bits:
// the node with the top bit, or the instance; then the splat's opacity,
// silhouette and depth, or the instance's map header).
const G5_CHUNK: u32 = 128u;
var<workgroup> g5_box: array<vec4<f32>, 128>;
var<workgroup> g5_info: array<vec4<f32>, 128>;
var<workgroup> g5_n: u32;
var<workgroup> g5_go: u32;
var<workgroup> g5_visits: u32;
var<workgroup> g5_open: atomic<u32>;
// Each pixel's open bins' bounding rectangle (empty: lo above hi), for the
// walk to skip nodes over full bins only.
var<workgroup> g5_rects: array<vec4<f32>, 64>;
// The walk's stack depth and visits (the first invocation's).
var<private> g5_sp: u32 = 0u;
var<private> g5_count: u32 = 0u;
// Development switches (bit 0: no bin rays, 1: no reading of entries, 2: no
// walk).
const G5_SKIP: u32 = 0u;

// Whether the screen rectangle [a, b) meets any of the tile's (corner at
// `lo`) pixels' open bins' rectangles; boxes over more than 16 pixels are
// taken as meeting one (testing them costs more than it saves).
fn g5_touches_open(a: vec2<f32>, b: vec2<f32>, lo: vec2<f32>) -> bool {
    let p0 = vec2<i32>(clamp(floor(a - lo), vec2<f32>(0.0), vec2<f32>(7.0)));
    let p1 = vec2<i32>(clamp(ceil(b - lo) - vec2<f32>(1.0), vec2<f32>(0.0), vec2<f32>(7.0)));
    if ((p1.x - p0.x + 1) * (p1.y - p0.y + 1) > 16) { return true; }
    for (var y = p0.y; y <= p1.y; y++) {
        for (var x = p0.x; x <= p1.x; x++) {
            let r = g5_rects[u32(y * 8 + x)];
            if (b.x > r.x && b.y > r.y && a.x < r.z && a.y < r.w) { return true; }
        }
    }
    return false;
}

// This pixel's open bins' bounding rectangle (empty if none).
fn g5_open_rect(pix: vec2<f32>, open: bool) -> vec4<f32> {
    var r = vec4<f32>(3.4e38, 3.4e38, -3.4e38, -3.4e38);
    if (!open) { return r; }
    for (var b = 0u; b < 16u; b++) {
        if (pbin[b] >= 0.9999) { continue; }
        let l = bin_lo(pix, b);
        r = vec4<f32>(min(r.xy, l), max(r.zw, l + vec2<f32>(0.25)));
    }
    return r;
}

// The first invocation walks the object hierarchy against the tile's
// frustum (`set_rect_frustum`, the exact box-face test included), nearest
// first, listing up to a chunk of entries: object nodes small enough to
// splat (gate 4's rule), and instances small enough for their maps (large
// ones the bins' rays cover).
fn g5_walk(lo: vec2<f32>, hi: vec2<f32>) {
    let v = view_world();
    var n = 0u;
    loop {
        if (g5_sp == 0u || n + 2u > G5_CHUNK) { break; }
        g5_sp--;
        let id = stack_get(g5_sp);
        let node = nodes[id];
        g5_count++;
        let sb = screen_box(node.lo.xyz, node.hi.xyz, v);
        if (sb.behind == 2u || frustum_misses_box(node.lo.xyz, node.hi.xyz)) { continue; }
        if (sb.behind == 0u && (sb.hi.x <= lo.x || sb.hi.y <= lo.y || sb.lo.x >= hi.x || sb.lo.y >= hi.y)) { continue; }
        if (sb.behind == 0u && !g5_touches_open(sb.lo, sb.hi, lo)) { continue; }
        let extent = sb.hi - sb.lo;
        if (sb.behind == 0u && node.info.z == 0u && light.objects.x != NONE
            && (max(extent.x, extent.y) <= 1.0 || min(extent.x, extent.y) <= 0.25)) {
            let dir = normalize(camera.eye.xyz - (node.lo.xyz + node.hi.xyz) * 0.5);
            let sil = box_silhouette(node.lo.xyz, node.hi.xyz, dir);
            if (sil > 1e-30) {
                g5_box[n] = vec4<f32>(sb.lo, sb.hi);
                g5_info[n] = vec4<f32>(bitcast<f32>(0x80000000u | id), 1.0 - exp(-node_front_area(light.objects.x + id, dir) / sil), sil, sb.depth);
                n++;
            }
        } else if (node.info.y > 0u) {
            for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                let m = items[k];
                if (m13_large(m)) { continue; }
                let hdr = map_header(meshes[m].z);
                let x = meshes[m].w;
                let c_obj = vec3<f32>(bitcast<f32>(cov_maps[hdr + 2u]), bitcast<f32>(cov_maps[hdr + 3u]), bitcast<f32>(cov_maps[hdr + 4u]));
                let scale = select(1.0, xforms[4u * x + 3u].x, x != NONE);
                let centre = m13_to_world(x, c_obj);
                let depth = dot(centre - camera.eye.xyz, camera.forward.xyz);
                let rad = bitcast<f32>(cov_maps[hdr + 5u]) * scale * camera.params.y / (2.0 * camera.params.z) / depth;
                let cp = cam_to_pixels(to_camera(v, centre));
                if (cp.x + rad <= lo.x || cp.y + rad <= lo.y || cp.x - rad >= hi.x || cp.y - rad >= hi.y) { continue; }
                if (!g5_touches_open(cp - vec2<f32>(rad), cp + vec2<f32>(rad), lo)) { continue; }
                g5_box[n] = vec4<f32>(cp - vec2<f32>(rad), cp + vec2<f32>(rad));
                g5_info[n] = vec4<f32>(bitcast<f32>(m), bitcast<f32>(hdr), 0.0, 0.0);
                n++;
            }
        } else if (g5_sp + 2u <= STACK) {
            var near = node.info.x;
            var far = node.info.x + 1u;
            let a = nodes[near];
            let b = nodes[far];
            if (to_camera(v, (b.lo.xyz + b.hi.xyz) * 0.5).z < to_camera(v, (a.lo.xyz + a.hi.xyz) * 0.5).z) {
                near = node.info.x + 1u;
                far = node.info.x;
            }
            stack_set(g5_sp, far);
            stack_set(g5_sp + 1u, near);
            g5_sp += 2u;
        }
    }
    g5_n = n;
}

// M13 gate 5 (profile stage 17), one workgroup per 8×8 tile after
// `hw_m13_corners`. A pixel whose four corners hit the same large object is
// covered. Another traces its 16 bin rays (a bin whose ray meets a large
// object is covered) only if one of the 16 corners of its 3×3 block hits a
// large object. Then the tile's walk lists its entries in chunks; each pixel
// adds those over its open bins (instances through the maps' texture, nodes
// splatted), until every pixel's bins are full or the walk ends. Writes
// (coverage, the tile's node visits (0 where the corners resolved the
// pixel), 2 × entries evaluated + 1 if it traced bin rays).
@compute @workgroup_size(8, 8)
fn hw_m13_tiles(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let inside = gid.x < w && gid.y < h;
    let pixel = gid.y * w + gid.x;
    let pix = vec2<f32>(f32(gid.x), f32(gid.y));
    for (var b = 0u; b < 16u; b++) { pbin[b] = 0.0; }
    var resolved = false;
    var traced = false;
    if (inside) {
        if (gid.x + 1u < w && gid.y + 1u < h) {
            let a = vis[pixel];
            if (a.x != NONE && a.y == 1u && vis[pixel + 1u].x == a.x && vis[pixel + w].x == a.x && vis[pixel + w + 1u].x == a.x) {
                resolved = true;
                for (var b = 0u; b < 16u; b++) { pbin[b] = 1.0; }
            }
        }
        if (!resolved && (G5_SKIP & 1u) == 0u) {
            var near_large = false;
            for (var dy = -1; dy <= 2; dy++) {
                for (var dx = -1; dx <= 2; dx++) {
                    let cx = u32(clamp(i32(gid.x) + dx, 0, i32(w) - 1));
                    let cy = u32(clamp(i32(gid.y) + dy, 0, i32(h) - 1));
                    if (vis[cy * w + cx].y == 1u) { near_large = true; }
                }
            }
            if (near_large) {
                traced = true;
                let tz = camera.params.z;
                let aspect = camera.params.x / camera.params.y;
                for (var k = 0u; k < 16u; k++) {
                    let p = pix + (vec2<f32>(f32(k % 4u), f32(k / 4u)) + vec2<f32>(0.5)) * 0.25;
                    let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
                    let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
                    let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
                    let hit = hw_trace(camera.eye.xyz, dir, 0.0);
                    pbin[k] = select(0.0, 1.0, hit.x != NONE && m13_large(hit.x));
                }
            }
        }
    }
    var open_bins = inside && !resolved && !bins_full(false);
    g5_rects[li] = g5_open_rect(pix, open_bins);
    workgroupBarrier();
    let tile_lo = vec2<f32>(f32(wid.x * 8u), f32(wid.y * 8u));
    let tile_hi = tile_lo + vec2<f32>(8.0);
    if (li == 0u) {
        set_rect_frustum(tile_lo, tile_hi);
        g5_sp = select(1u, 0u, (G5_SKIP & 4u) != 0u);
        g5_count = 0u;
        stack_set(0u, 0u);
    }
    var evaluated = 0u;
    loop {
        if (li == 0u) {
            g5_walk(tile_lo, tile_hi);
            atomicStore(&g5_open, 0u);
        }
        let n = workgroupUniformLoad(&g5_n);
        if (open_bins && (G5_SKIP & 2u) == 0u) {
            for (var e = 0u; e < n; e++) {
                let bx = g5_box[e];
                if (bx.z <= pix.x || bx.w <= pix.y || bx.x >= pix.x + 1.0 || bx.y >= pix.y + 1.0) { continue; }
                let info = g5_info[e];
                let bits = bitcast<u32>(info.x);
                evaluated++;
                let sb = ScreenBox(bx.xy, bx.zw, info.w, 0u);
                if (!touches_open_bin(sb, pix)) { continue; }
                if ((bits & 0x80000000u) != 0u) {
                    splat_bins(sb, pix, info.y, info.z, false);
                } else {
                    instance_map_tex(bits, bitcast<u32>(info.y), pix);
                }
                if (bins_full(false)) {
                    open_bins = false;
                    break;
                }
            }
        }
        if (open_bins) { atomicAdd(&g5_open, 1u); }
        g5_rects[li] = g5_open_rect(pix, open_bins);
        workgroupBarrier();
        if (li == 0u) {
            g5_go = select(0u, 1u, g5_sp > 0u && atomicLoad(&g5_open) > 0u);
            g5_visits = g5_count;
        }
        if (workgroupUniformLoad(&g5_go) == 0u) { break; }
    }
    let visits = workgroupUniformLoad(&g5_visits);
    if (inside) {
        var sum = 0.0;
        for (var b = 0u; b < 16u; b++) { sum += pbin[b]; }
        colors[pixel] = vec4<f32>(sum / 16.0, select(f32(visits), 0.0, resolved), f32(2u * evaluated + select(0u, 1u, traced)), 1.0);
    }
}
