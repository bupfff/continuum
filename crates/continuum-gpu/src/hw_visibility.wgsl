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
            missing += bounce_gather(job.surf.apex, dir, hit, hemicube_weight(b), &sum);
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
