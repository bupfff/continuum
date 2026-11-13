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

// Directions' maps read per bin: 4 (the four nearest, blended
// bilinearly), 2 (the nearest along one axis of the directions' grid, the
// two nearest along the axis whose fraction is nearer a half, blended), or
// 1 (the nearest).
const G5_DIRS: u32 = 2u;

// The pixel's bins not yet full (bit b: bin b), kept by the walk.
var<private> g5_open: u32 = 0xffffu;

fn g5_refresh_open() {
    var m = 0u;
    for (var b = 0u; b < 16u; b++) {
        if (pbin[b] < 0.9999) { m |= 1u << b; }
    }
    g5_open = m;
}

// The pixel's bins whose squares the screen rectangle [lo, hi) overlaps.
fn g5_rect_bins(lo: vec2<f32>, hi: vec2<f32>, pix: vec2<f32>) -> u32 {
    let a = vec2<i32>(max(floor((lo - pix) * 4.0), vec2<f32>(0.0)));
    let b = vec2<i32>(min(ceil((hi - pix) * 4.0) - vec2<f32>(1.0), vec2<f32>(3.0)));
    if (any(b < a)) { return 0u; }
    let row = (0xfu >> u32(3 - b.x)) & (0xfu << u32(a.x));
    // Rows a.y through b.y: the row's bits repeated every four bits.
    let rows = (0xffffu >> u32(4 * (3 - b.y))) & (0xffffu << u32(4 * a.y));
    return rows & (row * 0x1111u);
}

// `instance_map_bins` reading the maps' texture (8 bits, filtered by the
// hardware): object `m`, at most MAP_LIMIT_PX across, into the pixel's
// bins. False if it is too large or too near. Where a bin's ray passes the
// object's centre (its closest approach, in the object's frame over its
// radius) is taken affine in the bin's screen position about the centre's
// ray: the ray turns by `(I - c cᵀ) dD` (`c` the centre's direction, `D`
// the ray's unnormalized direction) and its closest approach moves by
// the distance times that; the next term is about rad / 2f of the radius
// (under 0.3% at 8 pixels across). Each bin's coordinates in a direction's
// map then cost two multiply-adds.
fn instance_map_tex(m: u32, pix: vec2<f32>) -> bool {
    let sph = m13_spheres[m];
    let radius = sph.w;
    if (radius <= 0.0) { return false; }
    let centre = sph.xyz;
    let rel = centre - camera.eye.xyz;
    let depth = dot(rel, camera.forward.xyz);
    if (depth - radius <= 1e-2) { return false; }
    let f = camera.params.y / (2.0 * camera.params.z);
    let rad = radius * f / depth;
    if (2.0 * rad > MAP_LIMIT_PX) { return false; }
    let cp = cam_to_pixels(to_camera(view_world(), centre));
    // Its square's open bins (by their squares; a bin is read where its
    // centre lies in the object's square).
    var todo = g5_rect_bins(cp - vec2<f32>(rad), cp + vec2<f32>(rad), pix) & g5_open;
    if (todo == 0u) { return true; }
    let mesh = meshes[m];
    let hdr = map_header(mesh.z);
    let x = mesh.w;
    if ((G5_SKIP & 16u) != 0u) {
        for (var b = 0u; b < 16u; b++) {
            let p = pix + vec2<f32>(f32(b % 4u), f32(b / 4u)) * 0.25 + vec2<f32>(0.125);
            if (abs(p.x - cp.x) > rad || abs(p.y - cp.y) > rad) { continue; }
            pbin[b] = 1.0 - (1.0 - pbin[b]) * 0.5;
        }
        return true;
    }
    // The object's rotation (world to object), loaded once.
    var r0 = vec3<f32>(1.0, 0.0, 0.0);
    var r1 = vec3<f32>(0.0, 1.0, 0.0);
    var r2 = vec3<f32>(0.0, 0.0, 1.0);
    if (x != NONE) {
        r0 = xforms[4u * x].xyz;
        r1 = xforms[4u * x + 1u].xyz;
        r2 = xforms[4u * x + 2u].xyz;
    }
    let len = length(rel);
    let c = rel / len;
    let d = normalize(r0 * c.x + r1 * c.y + r2 * c.z);
    let g = (oct_encode(d) + vec2<f32>(1.0)) * 0.5 * f32(MAP_DIRS) - vec2<f32>(0.5);
    let i0 = floor(g);
    let fr = g - i0;
    let texels_per_bin = 0.25 * depth / f / (2.0 * radius / f32(MAP_TEXELS));
    let level = clamp(round(log2(max(texels_per_bin, 1e-6))), 0.0, 4.0);
    let first = (hdr - 1u) / 8u * MAP_DIRS * MAP_DIRS;
    // The closest approach's change per pixel across and down, world then
    // object frame, over the radius.
    let tz = camera.params.z;
    let k = dot(c, camera.forward.xyz) * len / radius;
    let ex = camera.right.xyz * (2.0 * tz / camera.params.y);
    let ey = camera.up.xyz * (-2.0 * tz / camera.params.y);
    let wx = (ex - c * dot(c, ex)) * k;
    let wy = (ey - c * dot(c, ey)) * k;
    let qx = r0 * wx.x + r1 * wx.y + r2 * wx.z;
    let qy = r0 * wy.x + r1 * wy.y + r2 * wy.z;
    var layers: array<u32, 4>;
    var weights: array<f32, 4>;
    var uvx: array<vec2<f32>, 4>;
    var uvy: array<vec2<f32>, 4>;
    let dirs = G5_DIRS;
    let along_x = abs(fr.x - 0.5) <= abs(fr.y - 0.5);
    for (var k = 0u; k < dirs; k++) {
        var ix = u32(clamp(i32(i0.x) + i32(k & 1u), 0, i32(MAP_DIRS) - 1));
        var iy = u32(clamp(i32(i0.y) + i32(k >> 1u), 0, i32(MAP_DIRS) - 1));
        weights[k] = select(1.0 - fr.x, fr.x, (k & 1u) == 1u) * select(1.0 - fr.y, fr.y, (k >> 1u) == 1u);
        if (dirs == 1u) {
            ix = u32(clamp(round(g.x), 0.0, f32(MAP_DIRS) - 1.0));
            iy = u32(clamp(round(g.y), 0.0, f32(MAP_DIRS) - 1.0));
            weights[k] = 1.0;
        } else if (dirs == 2u) {
            if (along_x) {
                ix = u32(clamp(i32(i0.x) + i32(k), 0, i32(MAP_DIRS) - 1));
                iy = u32(clamp(round(g.y), 0.0, f32(MAP_DIRS) - 1.0));
                weights[k] = select(1.0 - fr.x, fr.x, k == 1u);
            } else {
                ix = u32(clamp(round(g.x), 0.0, f32(MAP_DIRS) - 1.0));
                iy = u32(clamp(i32(i0.y) + i32(k), 0, i32(MAP_DIRS) - 1));
                weights[k] = select(1.0 - fr.y, fr.y, k == 1u);
            }
        }
        layers[k] = first + iy * MAP_DIRS + ix;
        let dk = oct_decode((vec2<f32>(f32(ix), f32(iy)) + vec2<f32>(0.5)) / f32(MAP_DIRS) * 2.0 - vec2<f32>(1.0));
        let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(dk.z) < 0.999);
        let u = normalize(cross(helper, dk));
        let v = cross(dk, u);
        uvx[k] = vec2<f32>(dot(qx, u), dot(qx, v)) * 0.5;
        uvy[k] = vec2<f32>(dot(qy, u), dot(qy, v)) * 0.5;
    }
    let p0 = pix + vec2<f32>(0.125) - cp;
    while (todo != 0u) {
        let b = countTrailingZeros(todo);
        todo &= todo - 1u;
        let o = p0 + vec2<f32>(f32(b % 4u), f32(b / 4u)) * 0.25;
        if (abs(o.x) > rad || abs(o.y) > rad) { continue; }
        var cov = 0.0;
        for (var k = 0u; k < dirs; k++) {
            if (weights[k] <= 0.0) { continue; }
            let uv = vec2<f32>(0.5) + uvx[k] * o.x + uvy[k] * o.y;
            if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { continue; }
            if ((G5_SKIP & 8u) != 0u) { cov += weights[k] * 0.5; continue; }
            cov += weights[k] * textureSampleLevel(cov_tex, cov_sampler, uv, layers[k], level).x;
        }
        pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - clamp(cov, 0.0, 1.0));
    }
    return true;
}

// --- The tiles' subtree roots (development after the first tile walk). ---

@group(0) @binding(87) var<storage, read_write> m13_roots: array<u32>;
// Per frame: the count of pixels tracing bin rays, then from word 8 the
// pixels' list and after it every pixel's word: covered bins (bit b: bin
// b), bit 16 if it traced bin rays, bit 17 if its corners resolved it.
@group(0) @binding(88) var<storage, read_write> m13_work: array<atomic<u32>>;
// The bin rays' dispatch arguments.
@group(0) @binding(89) var<storage, read_write> m13_args: array<u32>;
// Each object's world bounding sphere about its coverage map's centre
// (radius 0: no map).
@group(0) @binding(90) var<storage, read> m13_spheres: array<vec4<f32>>;

fn m13_mask_word(pixel: u32) -> u32 {
    return 8u + u32(camera.params.x) * u32(camera.params.y) + pixel;
}

// Roots a tile lists at most. A tile's words: the count, then six per
// root: its node (bit 31 if it straddles the eye's plane, bit 30 if it is
// read from its coverage page, bit 29 if it lies below a node waiting for
// its page), its projected box in pixels as f32 bits (lo,
// hi; unless it straddles), so its pixels test it without loading or
// projecting the node, and its page.
const G5_ROOTS: u32 = 63u;
const G5_TILE_WORDS: u32 = 384u;
// A node is listed once its projected box is at most this many pixels
// across: the pixels walk below it.
const G5_CUT_PX: f32 = 8.0;

// M13 gate 5's roots pass (profile stage 17), one invocation per 8×8 tile
// after `hw_m13_corners`: walks the object hierarchy against the tile's
// frustum (the exact box-face test included), nearest first, and lists the
// roots of the subtrees its pixels walk: leaves, nodes at most G5_CUT_PX
// across, and, once the list would overflow, the nodes it would have
// descended. The listed subtrees hold every node the tile's frustum meets.
@compute @workgroup_size(8, 8)
fn hw_m13_tile_roots(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    lane = li;
    let tw = (u32(camera.params.x) + 7u) / 8u;
    let th = (u32(camera.params.y) + 7u) / 8u;
    if (gid.x >= tw || gid.y >= th) { return; }
    // The bin rays' list starts empty (it fills in `hw_m13_bins`).
    if (gid.x == 0u && gid.y == 0u) { atomicStore(&m13_work[0], 0u);}
    let base = (gid.y * tw + gid.x) * G5_TILE_WORDS;
    let lo = vec2<f32>(f32(gid.x * 8u), f32(gid.y * 8u));
    let hi = lo + vec2<f32>(8.0);
    set_rect_frustum(lo, hi);
    let v = view_world();
    var n = 0u;
    var sp = 1u;
    stack_set(0u, 0u);
    let pages = atomicLoad(&g6_state[9]) != 0u;
    let frame = atomicLoad(&g6_state[3]) + 1u;
    // At each pop, n + sp is at most G5_ROOTS. A stack entry's bit 31: below
    // a node that wants a page (no page is wanted under it).
    while (sp > 0u) {
        sp--;
        let raw = stack_get(sp);
        let id = raw & 0x7fffffffu;
        var below = (raw & 0x80000000u) != 0u;
        let node = nodes[id];
        let sb = screen_box(node.lo.xyz, node.hi.xyz, v);
        if (sb.behind == 2u || frustum_misses_box(node.lo.xyz, node.hi.xyz)) { continue; }
        if (sb.behind == 0u && (sb.hi.x <= lo.x || sb.hi.y <= lo.y || sb.lo.x >= hi.x || sb.lo.y >= hi.y)) { continue; }
        let extent = sb.hi - sb.lo;
        // M13 gate 6: the highest node on the path that fits a page at one
        // texel per bin and holds enough instances is read from its page if
        // it is resident and valid from here; otherwise it is requested and
        // walked as before.
        if (pages && !below && sb.behind == 0u && node.info.z == 0u && node.info.w == 0u
            && g6_range(id).y >= G6_MIN_ITEMS && g6_shape(extent * 4.0).x != 0u) {
            let page = atomicLoad(&g6_state[g6_table(id)]);
            if (page < G6_POOL && g6_valid(page, (node.lo.xyz + node.hi.xyz) * 0.5)) {
                let at = base + 1u + 6u * n;
                m13_roots[at] = id | 0x40000000u;
                m13_roots[at + 1u] = bitcast<u32>(sb.lo.x);
                m13_roots[at + 2u] = bitcast<u32>(sb.lo.y);
                m13_roots[at + 3u] = bitcast<u32>(sb.hi.x);
                m13_roots[at + 4u] = bitcast<u32>(sb.hi.y);
                m13_roots[at + 5u] = page;
                n++;
                atomicAdd(&g6_state[5], 1u);
                g6_mark_used(page, frame);
                continue;
            }
            if (page != G6_FAIL) {
                g6_request(id);
                below = true;
            }
        }
        if (node.info.y > 0u || (sb.behind == 0u && max(extent.x, extent.y) <= G5_CUT_PX)
            || n + sp + 2u > G5_ROOTS || sp + 2u > STACK) {
            let at = base + 1u + 6u * n;
            m13_roots[at] = id | select(0u, 0x80000000u, sb.behind != 0u) | select(0u, 0x20000000u, below);
            m13_roots[at + 1u] = bitcast<u32>(sb.lo.x);
            m13_roots[at + 2u] = bitcast<u32>(sb.lo.y);
            m13_roots[at + 3u] = bitcast<u32>(sb.hi.x);
            m13_roots[at + 4u] = bitcast<u32>(sb.hi.y);
            n++;
            continue;
        }
        var near = node.info.x;
        var far = node.info.x + 1u;
        let a = nodes[near];
        let b = nodes[far];
        if (to_camera(v, (b.lo.xyz + b.hi.xyz) * 0.5).z < to_camera(v, (a.lo.xyz + a.hi.xyz) * 0.5).z) {
            near = node.info.x + 1u;
            far = node.info.x;
        }
        let flag = select(0u, 0x80000000u, below);
        stack_set(sp, far | flag);
        stack_set(sp + 1u, near | flag);
        sp += 2u;
    }
    m13_roots[base] = n;
}

// Development switches (bit 0: no bin rays, 1: no walks, 2: no map reads,
// 3: no texture samples in them, 4: no set-up or samples either).
const G5_SKIP: u32 = 0u;
// Whether the pixels' walks go as packets (a subgroup walking the union of
// its pixels' walks) or each on its own.
const G5_PACKETS: bool = true;

fn g5_any(x: bool) -> bool {
    if (G5_PACKETS) { return packet_any(x); }
    return x;
}

fn g5_first(x: bool) -> bool {
    if (G5_PACKETS) { return packet_first(x); }
    return x;
}

// The pixel's walk of the subtree under `root` (its frustum set), nearest
// first, as a packet: `map_coverage_from`'s walk with large instances left
// out (the bins' rays covered them), the maps read through the texture and
// the exact box-face test. `on` turns false once the bins are full. Returns
// the object nodes it visited.
fn g5_walk_subtree(root: u32, pix: vec2<f32>, on: ptr<function, bool>, root_box: vec4<f32>, root_known: bool, pages: bool) -> u32 {
    let v = view_world();
    var visits = 0u;
    var sp = 1u;
    stack_set(0u, root);
    need_put(0u, *on);
    var first = true;
    while (sp > 0u) {
        sp--;
        let raw = stack_get(sp);
        let n = raw & 0x7fffffffu;
        // Bit 31: below a node waiting for its page (M13 gate 6).
        let below = (raw & 0x80000000u) != 0u;
        let mine = need_get(sp) && *on;
        let known = first && root_known;
        first = false;
        if (!g5_any(mine)) { continue; }
        let node = nodes[n];
        var descend = false;
        var waiting = false;
        if (mine) {
            visits++;
            // The root's box as the roots pass projected it (in front of
            // the eye, inside the tile's frustum).
            var sb: ScreenBox;
            var live = true;
            if (known) {
                sb = ScreenBox(root_box.xy, root_box.zw, to_camera(v, (node.lo.xyz + node.hi.xyz) * 0.5).z, 0u);
            } else {
                sb = screen_box(node.lo.xyz, node.hi.xyz, v);
                live = sb.behind != 2u && !frustum_misses_box(node.lo.xyz, node.hi.xyz);
            }
            // A node over full bins only (or none of the pixel's) cannot
            // change the pixel.
            if (live && sb.behind == 0u && (g5_rect_bins(sb.lo, sb.hi, pix) & g5_open) == 0u) { live = false; }
            // M13 gate 6: the highest node on the path that fits a page and
            // holds enough instances is read from its page if resident and
            // valid; otherwise it is asked for and walked as before.
            var paged = false;
            if (live && pages && !below && sb.behind == 0u && node.info.z == 0u && node.info.w == 0u
                && g6_shape((sb.hi - sb.lo) * 4.0).x != 0u && g6_range(n).y >= G6_MIN_ITEMS) {
                let page = atomicLoad(&g6_state[g6_table(n)]);
                if (page < G6_POOL && g6_valid(page, (node.lo.xyz + node.hi.xyz) * 0.5)) {
                    g6_mark_used(page, atomicLoad(&g6_state[3]) + 1u);
                    g6_read_page(page, pix, vec4<f32>(sb.lo, sb.hi));
                    g5_refresh_open();
                    if (g5_open == 0u) { *on = false; }
                    paged = true;
                } else if (page != G6_FAIL) {
                    g6_request(n);
                    waiting = true;
                }
            }
            if (live && !paged) {
                let extent = sb.hi - sb.lo;
                if (sb.behind == 0u && node.info.z == 0u && light.objects.x != NONE
                    && (max(extent.x, extent.y) <= 1.0 || min(extent.x, extent.y) <= 0.25)) {
                    let dir = normalize(camera.eye.xyz - (node.lo.xyz + node.hi.xyz) * 0.5);
                    let sil = box_silhouette(node.lo.xyz, node.hi.xyz, dir);
                    if (sil > 1e-30) {
                        splat_bins(sb, pix, 1.0 - exp(-node_front_area(light.objects.x + n, dir) / sil), sil, false);
                        g5_refresh_open();
                        if (g5_open == 0u) { *on = false; }
                    }
                } else if (node.info.y > 0u) {
                    for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                        let m = items[k];
                        // Large instances (too large or near for their maps,
                        // which `instance_map_tex` skips) the bins' rays covered.
                        if ((G5_SKIP & 4u) != 0u) { continue; }
                        instance_map_tex(m, pix);
                    }
                    g5_refresh_open();
                    if (g5_open == 0u) { *on = false; }
                } else {
                    descend = true;
                }
            }
        }
        if (!g5_any(descend) || node.info.y > 0u || sp + 2u > STACK) { continue; }
        let flag = select(0u, 0x80000000u, below || g5_any(waiting));
        var near = node.info.x;
        var far = node.info.x + 1u;
        let a = nodes[near];
        let b = nodes[far];
        if (g5_first(to_camera(v, (b.lo.xyz + b.hi.xyz) * 0.5).z < to_camera(v, (a.lo.xyz + a.hi.xyz) * 0.5).z)) {
            near = node.info.x + 1u;
            far = node.info.x;
        }
        stack_set(sp, far | flag);
        stack_set(sp + 1u, near | flag);
        need_put(sp, descend);
        need_put(sp + 1u, descend);
        sp += 2u;
    }
    return visits;
}


// M13 gate 5's bins pass (profile stage 17), after `hw_m13_corners`: a
// pixel whose four corners hit the same large object is covered. Another
// is listed to trace its 16 bin rays (a bin whose ray meets a large object
// is covered) only if one of the 16 corners of its 3×3 block hits a large
// object. Starts each pixel's word.
@compute @workgroup_size(8, 8)
fn hw_m13_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let pixel = gid.y * w + gid.x;
    if (gid.x + 1u < w && gid.y + 1u < h) {
        let a = vis[pixel];
        if (a.x != NONE && a.y == 1u && vis[pixel + 1u].x == a.x && vis[pixel + w].x == a.x && vis[pixel + w + 1u].x == a.x) {
            atomicStore(&m13_work[m13_mask_word(pixel)], 0xffffu | (1u << 17u));
            return;
        }
    }
    var near_large = false;
    for (var dy = -1; dy <= 2; dy++) {
        for (var dx = -1; dx <= 2; dx++) {
            let cx = u32(clamp(i32(gid.x) + dx, 0, i32(w) - 1));
            let cy = u32(clamp(i32(gid.y) + dy, 0, i32(h) - 1));
            if (vis[cy * w + cx].y == 1u) { near_large = true; }
        }
    }
    var word = 0u;
    if (near_large && (G5_SKIP & 1u) == 0u) {
        word = 1u << 16u;
        let i = atomicAdd(&m13_work[0], 1u);
        atomicStore(&m13_work[8u + i], pixel);
    }
    atomicStore(&m13_work[m13_mask_word(pixel)], word);
}

// The bin rays' dispatch arguments: 16 rays a listed pixel, 64 a workgroup.
@compute @workgroup_size(1)
fn hw_m13_bin_args() {
    let groups = (atomicLoad(&m13_work[0]) * 16u + 63u) / 64u;
    let x = min(groups, 65535u);
    m13_args[0] = x;
    m13_args[1] = select(1u, (groups + x - 1u) / x, x > 0u);
    m13_args[2] = 1u;
}

// One invocation per bin ray of a listed pixel: through the bin's centre;
// a bin whose ray meets a large object is covered.
@compute @workgroup_size(64)
fn hw_m13_bin_rays(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = (wg.y * nwg.x + wg.x) * 64u + li;
    if (t >= atomicLoad(&m13_work[0]) * 16u) { return; }
    let pixel = atomicLoad(&m13_work[8u + t / 16u]);
    let k = t % 16u;
    let w = u32(camera.params.x);
    let p = vec2<f32>(f32(pixel % w), f32(pixel / w)) + (vec2<f32>(f32(k % 4u), f32(k / 4u)) + vec2<f32>(0.5)) * 0.25;
    let tz = camera.params.z;
    let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * camera.params.x / camera.params.y;
    let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
    let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
    let hit = hw_trace(camera.eye.xyz, dir, 0.0);
    if (hit.x != NONE && m13_large(hit.x)) { atomicOr(&m13_work[m13_mask_word(pixel)], 1u << k); }
}

// M13 gate 5 (profile stage 17), one workgroup per 8×8 tile after
// `hw_m13_tile_roots` and the bin rays: each pixel not covered walks its
// tile's listed subtrees in order (instances through the maps' texture, far
// nodes splatted) until its bins are full. Writes (coverage, object nodes
// the pixel visited (0 where the corners resolved it), 2 × the tile's roots
// + 1 if it traced bin rays).
@compute @workgroup_size(8, 8)
fn hw_m13_tiles(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    lane = li;
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let inside = gid.x < w && gid.y < h;
    let pixel = gid.y * w + gid.x;
    let pix = vec2<f32>(f32(gid.x), f32(gid.y));
    var word = 0u;
    if (inside) { word = atomicLoad(&m13_work[m13_mask_word(pixel)]); }
    for (var b = 0u; b < 16u; b++) { pbin[b] = select(0.0, 1.0, (word & (1u << b)) != 0u); }
    let resolved = (word & (1u << 17u)) != 0u;
    let traced = (word & (1u << 16u)) != 0u;
    g5_open = ~word & 0xffffu;
    var on = inside && g5_open != 0u && (G5_SKIP & 2u) == 0u;
    let base = (wid.y * ((w + 7u) / 8u) + wid.x) * G5_TILE_WORDS;
    let count = m13_roots[base];
    let pages = atomicLoad(&g6_state[9]) != 0u;
    set_pixel_frustum(pix);
    var visits = 0u;
    for (var r = 0u; r < count; r++) {
        if (!g5_any(on)) { break; }
        let at = base + 1u + 6u * r;
        let word = m13_roots[at];
        let id = word & 0x1fffffffu;
        let known = (word & 0x80000000u) == 0u;
        let rb = vec4<f32>(bitcast<f32>(m13_roots[at + 1u]), bitcast<f32>(m13_roots[at + 2u]), bitcast<f32>(m13_roots[at + 3u]), bitcast<f32>(m13_roots[at + 4u]));
        // A root in front of the eye over none of the pixel's open bins is
        // skipped from its stored box.
        var mine = on;
        if (known) { mine = mine && (g5_rect_bins(rb.xy, rb.zw, pix) & g5_open) != 0u; }
        if (!g5_any(mine)) { continue; }
        // M13 gate 6: a node read from its coverage page.
        if ((word & 0x40000000u) != 0u) {
            if (mine) {
                g6_read_page(m13_roots[at + 5u], pix, rb);
                g5_refresh_open();
                if (g5_open == 0u) { on = false; }
            }
            continue;
        }
        // A pixel whose bins are full or that the root misses takes no part.
        var walking = mine;
        visits += g5_walk_subtree(id | select(0u, 0x80000000u, (word & 0x20000000u) != 0u), pix, &walking, rb, known, pages);
        if (mine && !walking) { on = false; }
    }
    if (inside) {
        var sum = 0.0;
        for (var b = 0u; b < 16u; b++) { sum += pbin[b]; }
        colors[pixel] = vec4<f32>(sum / 16.0, select(f32(visits), 0.0, resolved), f32(2u * count + select(0u, 1u, traced)), 1.0);
    }
}

// --- M13 gate 6: hierarchy nodes holding coverage, as view-dependent
// coverage pages (DESIGN §3.4, part D). ---

// The hierarchy's item ranges: word 0 the node count, then per node (first
// item, items in its subtree), then per object its place among the items.
@group(0) @binding(91) var<storage, read> g6_fixed: array<u32>;
// The pages' state: G6_META counters (0 requests this frame, 1 the pool's
// ring, 2 bakes this frame, 3 frames, 4 bakes a frame at most, 5 page roots
// listed, 6 requests last frame, 7 pages used, 8 pages used last frame,
// 9 pages on, 10 pages baked, 11 nodes that fit no page, 12 page roots
// listed last frame, 13 the largest shift read in 1/10,000 bin, 14 last
// frame's), the requests' list, the pages the bakes write, then per node
// its page (NONE, or
// G6_FAIL: it fits none), per node its request mark, per page the frame it
// was last used.
@group(0) @binding(92) var<storage, read_write> g6_state: array<atomic<u32>>;
// The bakes' dispatch arguments.
@group(0) @binding(93) var<storage, read_write> g6_args: array<u32>;
// Per page five words: (plane origin, bake distance), (texel axis across /
// texel, depth extent), (texel axis down / texel, shape bits: width | height
// << 16), (plane normal, node bits), (bake eye, bake focal length).
@group(0) @binding(94) var<storage, read_write> g6_hdr: array<vec4<f32>>;
// Per page 512 8-bit texels: the share of 4×4 rays through each that meet
// the node's contents.
@group(0) @binding(95) var<storage, read_write> g6_texels: array<u32>;

const G6_POOL: u32 = 32768u;
const G6_REQUESTS: u32 = 4096u;
const G6_META: u32 = 16u;
// Nodes with fewer instances read the instances' shared maps instead.
const G6_MIN_ITEMS: u32 = 8u;
const G6_FAIL: u32 = 0xfffffffeu;

fn g6_nodes() -> u32 { return g6_fixed[0]; }
fn g6_range(n: u32) -> vec2<u32> { return vec2<u32>(g6_fixed[1u + 2u * n], g6_fixed[2u + 2u * n]); }
fn g6_place(m: u32) -> u32 { return g6_fixed[1u + 2u * g6_nodes() + m]; }
fn g6_slot(i: u32) -> u32 { return G6_META + G6_REQUESTS + i; }
fn g6_table(n: u32) -> u32 { return G6_META + 2u * G6_REQUESTS + n; }
fn g6_mark(n: u32) -> u32 { return G6_META + 2u * G6_REQUESTS + g6_nodes() + n; }
fn g6_used(page: u32) -> u32 { return G6_META + 2u * G6_REQUESTS + 2u * g6_nodes() + page; }

// The page shape (width, height in texels; 512 texels, width 4 to 128) a
// node `extent` bins across fits, or zero.
fn g6_shape(extent: vec2<f32>) -> vec2<u32> {
    let w = max(ceil(extent.x), 1.0);
    var sw = 4u;
    loop {
        if (f32(sw) >= w || sw >= 128u) { break; }
        sw *= 2u;
    }
    let sh = 512u / sw;
    if (f32(sw) < w || ceil(extent.y) > f32(sh)) { return vec2<u32>(0u); }
    return vec2<u32>(sw, sh);
}

// Whether `page`, of a node centred at `centre`, may be read from the
// current eye: its contents shift by at most a quarter of a bin since its
// bake (2 f |ΔE⊥| L / D²), and its texels are 0.7–1.4 bins wide.
fn g6_valid(page: u32, centre: vec3<f32>) -> bool {
    let h0 = g6_hdr[5u * page];
    let h1 = g6_hdr[5u * page + 1u];
    let h3 = g6_hdr[5u * page + 3u];
    let h4 = g6_hdr[5u * page + 4u];
    let f = camera.params.y / (2.0 * camera.params.z);
    let de = camera.eye.xyz - h4.xyz;
    let perp = de - h3.xyz * dot(de, h3.xyz);
    let shift = 2.0 * f * length(perp) * h1.w / (h0.w * h0.w);
    if (shift > 0.25) { return false; }
    let ratio = (h0.w / h4.w) / (length(centre - camera.eye.xyz) / f);
    if (ratio < 0.7 || ratio > 1.4) { return false; }
    // The largest shift read this frame, in ten-thousandths of a bin.
    let q = u32(shift * 1e4);
    if (q > atomicLoad(&g6_state[13])) { atomicMax(&g6_state[13], q); }
    return true;
}

// Counts `page` among the pages read this frame (`frame`: the frame
// counter plus one), once.
fn g6_mark_used(page: u32, frame: u32) {
    if (atomicLoad(&g6_state[g6_used(page)]) == frame) { return; }
    if (atomicExchange(&g6_state[g6_used(page)], frame) != frame) { atomicAdd(&g6_state[7], 1u); }
}

// Asks for node `n`'s page, once a frame.
fn g6_request(n: u32) {
    if (atomicLoad(&g6_state[g6_mark(n)]) != 0u) { return; }
    if (atomicExchange(&g6_state[g6_mark(n)], 1u) != 0u) { return; }
    let i = atomicAdd(&g6_state[0], 1u);
    if (i < G6_REQUESTS) {
        atomicStore(&g6_state[G6_META + i], n);
    } else {
        atomicStore(&g6_state[g6_mark(n)], 0u);
    }
}

// Where the ray through screen point `p` crosses the plane through `o`
// with normal `nrm`, in texels along `u` and `v` (axes over the texel).
fn g6_plane_texels(p: vec2<f32>, o: vec3<f32>, u: vec3<f32>, v: vec3<f32>, nrm: vec3<f32>) -> vec2<f32> {
    let tz = camera.params.z;
    let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * camera.params.x / camera.params.y;
    let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
    let dir = camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy;
    let e = camera.eye.xyz;
    let q = e + dir * (dot(o - e, nrm) / dot(dir, nrm)) - o;
    return vec2<f32>(dot(q, u), dot(q, v));
}

fn g6_texel(page: u32, i: i32, j: i32, sw: i32, sh: i32) -> f32 {
    if (i < 0 || j < 0 || i >= sw || j >= sh) { return 0.0; }
    let k = u32(j * sw + i);
    return f32((g6_texels[128u * page + k / 4u] >> (8u * (k % 4u))) & 0xffu) / 255.0;
}

// Adds `page` (its node projected over `bx`) to the pixel's open bins:
// per bin, bilinearly where the bin's centre ray crosses the page's plane
// (affine across the pixel), combined as an independent occluder.
fn g6_read_page(page: u32, pix: vec2<f32>, bx: vec4<f32>) {
    var todo = g5_rect_bins(bx.xy, bx.zw, pix) & g5_open;
    if (todo == 0u) { return; }
    let o = g6_hdr[5u * page].xyz;
    let u = g6_hdr[5u * page + 1u].xyz;
    let h2 = g6_hdr[5u * page + 2u];
    let nrm = g6_hdr[5u * page + 3u].xyz;
    let shape = bitcast<u32>(h2.w);
    let sw = i32(shape & 0xffffu);
    let sh = i32(shape >> 16u);
    let p0 = pix + vec2<f32>(0.125);
    let c0 = g6_plane_texels(p0, o, u, h2.xyz, nrm) - vec2<f32>(0.5);
    let cx = (g6_plane_texels(p0 + vec2<f32>(0.75, 0.0), o, u, h2.xyz, nrm) - vec2<f32>(0.5) - c0) / 3.0;
    let cy = (g6_plane_texels(p0 + vec2<f32>(0.0, 0.75), o, u, h2.xyz, nrm) - vec2<f32>(0.5) - c0) / 3.0;
    while (todo != 0u) {
        let b = countTrailingZeros(todo);
        todo &= todo - 1u;
        let c = c0 + cx * f32(b % 4u) + cy * f32(b / 4u);
        let f0 = floor(c);
        let fr = c - f0;
        let i = vec2<i32>(f0);
        let top = mix(g6_texel(page, i.x, i.y, sw, sh), g6_texel(page, i.x + 1, i.y, sw, sh), fr.x);
        let bottom = mix(g6_texel(page, i.x, i.y + 1, sw, sh), g6_texel(page, i.x + 1, i.y + 1, sw, sh), fr.x);
        pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - clamp(mix(top, bottom, fr.y), 0.0, 1.0));
    }
}

// After the pixels' pass, one invocation: the bakes this frame (at most
// the budget, the rest dropped and asked again), each a page (its own, or
// the next in the pool's ring, evicting its owner), their dispatch (eight
// workgroups a page), the counters' turn to the next frame.
@compute @workgroup_size(1)
fn hw_m13_page_args() {
    let c = min(atomicLoad(&g6_state[0]), G6_REQUESTS);
    let nb = min(c, atomicLoad(&g6_state[4]));
    for (var i = 0u; i < nb; i++) {
        let n = atomicLoad(&g6_state[G6_META + i]);
        var page = atomicLoad(&g6_state[g6_table(n)]);
        if (page >= G6_POOL) {
            page = atomicAdd(&g6_state[1], 1u) % G6_POOL;
            let old = bitcast<u32>(g6_hdr[5u * page + 3u].w);
            if (old < g6_nodes() && atomicLoad(&g6_state[g6_table(old)]) == page) {
                atomicStore(&g6_state[g6_table(old)], NONE);
            }
            // The new owner, so a later eviction finds it.
            g6_hdr[5u * page + 3u].w = bitcast<f32>(n);
        }
        atomicStore(&g6_state[g6_slot(i)], page);
    }
    for (var i = nb; i < c; i++) {
        atomicStore(&g6_state[g6_mark(atomicLoad(&g6_state[G6_META + i]))], 0u);
    }
    atomicStore(&g6_state[2], nb);
    atomicStore(&g6_state[6], c);
    atomicStore(&g6_state[0], 0u);
    atomicStore(&g6_state[8], atomicLoad(&g6_state[7]));
    atomicStore(&g6_state[7], 0u);
    atomicStore(&g6_state[12], atomicLoad(&g6_state[5]));
    atomicStore(&g6_state[5], 0u);
    atomicStore(&g6_state[14], atomicLoad(&g6_state[13]));
    atomicStore(&g6_state[13], 0u);
    atomicAdd(&g6_state[3], 1u);
    g6_args[0] = 8u * nb;
    g6_args[1] = 1u;
    g6_args[2] = 1u;
}

// Whether a ray from `e` along `dir` meets one of the instances in item
// range `r` inside the box [lo, hi]: opaque queries, stepping past hits on
// other instances (in at most 16 queries).
fn g6_ray_hits(e: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, r: vec2<u32>) -> bool {
    let inv = vec3<f32>(1.0) / dir;
    let ta = (lo - e) * inv;
    let tb = (hi - e) * inv;
    var t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
    let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    // From the box's entry: traversal from the eye would meet everything
    // the ray passes before it.
    let start = e + dir * t0;
    let t_end = t1 - t0;
    t0 = 0.0;
    for (var k = 0u; k < 16u; k++) {
        if (t_end <= t0) { return false; }
        let hit = g6_first_hit(start, dir, t0, t_end);
        if (hit.y == NONE) { return false; }
        let p = g6_place(hit.y);
        if (p >= r.x && p < r.x + r.y) { return true; }
        t0 = bitcast<f32>(hit.x) * (1.0 + 1e-6) + 1e-4;
    }
    return false;
}

// The nearest hit on [t0, t1] (front and back faces alike): its distance's
// bits and object (NONE: none).
fn g6_first_hit(e: vec3<f32>, dir: vec3<f32>, t0: f32, t1: f32) -> vec2<u32> {
    var rq: ray_query;
    // FORCE_OPAQUE.
    rayQueryInitialize(&rq, scene_as, RayDesc(0x1u, 0xffu, t0, t1, e, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == 0u) { return vec2<u32>(0u, NONE); }
    return vec2<u32>(bitcast<u32>(hit.t), hw_object(hit.instance_custom_data, hit.geometry_index).x);
}

var<workgroup> g6_bake: array<vec4<f32>, 5>;
var<workgroup> g6_bake_ok: u32;
var<workgroup> g6_bytes: array<u32, 64>;

// Eight workgroups per page baked this frame, 64 texels each: the
// requested node's page from the current eye, on the plane through its
// centre facing the eye, a texel a bin at its distance, its projection
// centred in the smallest shape that fits; each texel the share of 4×4 rays
// through it that meet the node's contents (8,192 rays a page). The first
// workgroup writes the header.
@compute @workgroup_size(64)
fn hw_m13_page_bake(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = wg.x / 8u;
    let part = wg.x % 8u;
    let n = atomicLoad(&g6_state[G6_META + i]);
    let page = atomicLoad(&g6_state[g6_slot(i)]);
    let node = nodes[n];
    let lo = node.lo.xyz;
    let hi = node.hi.xyz;
    let e = camera.eye.xyz;
    if (li == 0u) {
        let centre = (lo + hi) * 0.5;
        let rel = centre - e;
        let dist = length(rel);
        let d = rel / dist;
        let f = camera.params.y / (2.0 * camera.params.z);
        let t = dist / (4.0 * f);
        let u = normalize(camera.right.xyz - d * dot(d, camera.right.xyz));
        let v = cross(u, d);
        var amin = vec2<f32>(3.4e38);
        var amax = vec2<f32>(-3.4e38);
        var zmin = 3.4e38;
        var zmax = -3.4e38;
        for (var k = 0u; k < 8u; k++) {
            let c = vec3<f32>(select(lo.x, hi.x, (k & 1u) != 0u), select(lo.y, hi.y, (k & 2u) != 0u), select(lo.z, hi.z, (k & 4u) != 0u));
            let w = c - e;
            let z = dot(w, d);
            zmin = min(zmin, z);
            zmax = max(zmax, z);
            let p = w * (dist / max(z, 1e-3)) - rel;
            let ab = vec2<f32>(dot(p, u), dot(p, v)) / t;
            amin = min(amin, ab);
            amax = max(amax, ab);
        }
        let shape = g6_shape(amax - amin);
        let ok = shape.x != 0u && zmin > 1e-2;
        let a0 = (amin + amax) * 0.5 - vec2<f32>(f32(shape.x), f32(shape.y)) * 0.5;
        g6_bake[0] = vec4<f32>(centre + u * (t * a0.x) + v * (t * a0.y), dist);
        g6_bake[1] = vec4<f32>(u / t, zmax - zmin);
        g6_bake[2] = vec4<f32>(v / t, bitcast<f32>(shape.x | (shape.y << 16u)));
        g6_bake[3] = vec4<f32>(d, bitcast<f32>(n));
        g6_bake[4] = vec4<f32>(e, f);
        g6_bake_ok = select(0u, 1u, ok);
        if (part == 0u) {
            if (ok) {
                for (var k = 0u; k < 5u; k++) { g6_hdr[5u * page + k] = g6_bake[k]; }
                atomicStore(&g6_state[g6_table(n)], page);
                atomicAdd(&g6_state[10], 1u);
            } else {
                atomicStore(&g6_state[g6_table(n)], G6_FAIL);
                atomicAdd(&g6_state[11], 1u);
            }
            atomicStore(&g6_state[g6_mark(n)], 0u);
        }
    }
    if (workgroupUniformLoad(&g6_bake_ok) == 0u) { return; }
    let o = g6_bake[0].xyz;
    // Back from texels to world: the axes over the texel's square.
    let tu = g6_bake[1].xyz;
    let tv = g6_bake[2].xyz;
    let u = tu / dot(tu, tu);
    let v = tv / dot(tv, tv);
    let sw = bitcast<u32>(g6_bake[2].w) & 0xffffu;
    let r = g6_range(n);
    let k = part * 64u + li;
    let ti = f32(k % sw);
    let tj = f32(k / sw);
    var hits = 0u;
    for (var ray = 0u; ray < 16u; ray++) {
        let aim = o + u * (ti + (f32(ray % 4u) + 0.5) * 0.25) + v * (tj + (f32(ray / 4u) + 0.5) * 0.25);
        if (g6_ray_hits(e, normalize(aim - e), lo, hi, r)) { hits++; }
    }
    g6_bytes[li] = u32(round(f32(hits) / 16.0 * 255.0));
    workgroupBarrier();
    if (li < 16u) {
        let b = 4u * li;
        g6_texels[128u * page + part * 16u + li] = g6_bytes[b] | (g6_bytes[b + 1u] << 8u) | (g6_bytes[b + 2u] << 16u) | (g6_bytes[b + 3u] << 24u);
    }
}
