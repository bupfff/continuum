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
    hw_m13_corner(gid);
}

fn hw_m13_corner(gid: vec3<u32>) {
    let w = u32(camera.params.x);
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

// `hw_m13_corners` after the roots pass (M13 gate 8): in a tile whose roots
// are all at most MAP_LIMIT_PX across, no instance is large, so its pixels'
// corners record no large hit without a ray.
@compute @workgroup_size(8, 8)
fn hw_m13_corners_tiles(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let base = (wid.y * ((w + 7u) / 8u) + wid.x) * G5_TILE_WORDS;
    // The corner (the pixel's top-left) is traced only inside a box of a
    // root that may hold a large instance (half a pixel's margin).
    let n_large = m13_roots[base + G5_TILE_WORDS - 1u];
    var inside = n_large == 9u && (G5_SKIP & 256u) == 0u;
    let pt = vec2<f32>(f32(gid.x), f32(gid.y));
    for (var k = 0u; k < min(n_large, 8u); k++) {
        let at = base + 380u + 4u * k;
        let lo = vec2<f32>(bitcast<f32>(m13_roots[at]), bitcast<f32>(m13_roots[at + 1u])) - vec2<f32>(0.5);
        let hi = vec2<f32>(bitcast<f32>(m13_roots[at + 2u]), bitcast<f32>(m13_roots[at + 3u])) + vec2<f32>(0.5);
        if (all(pt >= lo) && all(pt <= hi) && (G5_SKIP & 256u) == 0u) { inside = true; }
    }
    if (!inside) {
        vis[gid.y * w + gid.x] = vec4<u32>(NONE, 0u, bitcast<u32>(3.4e38), 0u);
        if (g9_colour_on()) {
            let at = g9_corner_word(gid.y * w + gid.x);
            atomicStore(&m13_work[at + 3u], NONE);
            // Tiles that hold no large instance hold no background object.
            atomicStore(&m13_work[at + 5u], NONE);
        }
        return;
    }
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    let sx = (f32(gid.x) / camera.params.x * 2.0 - 1.0) * tz * aspect;
    let sy = (1.0 - f32(gid.y) / camera.params.y * 2.0) * tz;
    let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
    if (g9_colour_on()) {
        // M13 gate 9: a large hit shaded at the corner, for its pixels'
        // large bins.
        let full = hw_trace(camera.eye.xyz, dir, 0.0);
        var large = 0u;
        let at = g9_corner_word(gid.y * w + gid.x);
        atomicStore(&m13_work[at + 3u], NONE);
        // What lies behind at the corner: the background object a ray seeing
        // only those meets (NONE: the sky).
        atomicStore(&m13_work[at + 5u], g9_background(camera.eye.xyz, dir).x);
        if (full.x != NONE && g8_large(full.x)) {
            // Background objects are not large for bin rays (the pixels'
            // background ray finds them), but their corners are shaded.
            large = select(1u, 0u, g9_is_background(full.x) && G9_BG_NOT_LARGE);
            let c = g9_shade_hit(full.x, full.y, bitcast<f32>(full.z), camera.eye.xyz, dir);
            atomicStore(&m13_work[at], bitcast<u32>(c.x));
            atomicStore(&m13_work[at + 1u], bitcast<u32>(c.y));
            atomicStore(&m13_work[at + 2u], bitcast<u32>(c.z));
            atomicStore(&m13_work[at + 3u], full.x);
            atomicStore(&m13_work[at + 4u], full.z);
        }
        vis[gid.y * w + gid.x] = vec4<u32>(full.x, large, 0u, 0u);
        return;
    }
    let hit = g8_first_object(camera.eye.xyz, dir);
    var large = 0u;
    if (hit != NONE && g8_large(hit)) { large = 1u; }
    vis[gid.y * w + gid.x] = vec4<u32>(hit, large, 0u, 0u);
}

// The object a ray from `e` along `dir` meets first, any face (coverage asks
// only whether a ray is blocked; M13 gate 8), or NONE.
fn g8_first_object(e: vec3<f32>, dir: vec3<f32>) -> u32 {
    var rq: ray_query;
    // FORCE_OPAQUE.
    rayQueryInitialize(&rq, scene_as, RayDesc(0x1u, 0xffu, 0.0, 3.4e38, e, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == 0u) { return NONE; }
    return hw_object(hit.instance_custom_data, hit.geometry_index).x;
}

// Whether object `m`, met by a ray this frame, is large (M13 gate 8): it
// has no current record (too large or near for its maps, or mapless; a
// met object lies in the view).
fn g8_large(m: u32) -> bool {
    return ((bitcast<u32>(m13_records[3u * m].w) >> 3u) & 0x1fffu) != g8_stamp();
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

// --- M13 gate 8: instance records, once a frame. ---

// Development: a sink for timing tests (G5_SKIP bit 6).
var<private> g8_dummy: f32 = 0.0;

// Development: whether the once-a-frame pass projects the nodes' boxes.
const G8_NODE_BOXES: bool = true;

// Development: bin rays only where one of the pixel's own corners hits a
// large object.
const G8_OWN_CORNERS: bool = true;

// This frame's stamp for records (the pages' frame counter, which turns at
// the frame's end, modulo 2^13).
fn g8_stamp() -> u32 {
    return atomicLoad(&g6_state[3]) & 0x1fffu;
}

// `v` rotated by unit quaternion `q`.
fn g8_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Per object two words: (projected centre, radius in pixels, bits: the
// maps' level (3 bits), the frame's stamp (13 bits), the nearest
// direction's layer (16 bits)) and the change of that direction's map
// coordinates per pixel across and down. Only objects read this frame are
// written (small, in front of the eye, inside the view); a stale stamp
// means not read (outside the view, behind the eye, too large or near for
// its maps, or mapless).
@group(0) @binding(96) var<storage, read_write> m13_records: array<vec4<f32>>;

// One invocation per object (M13 gate 8): object `m`'s record, as
// `instance_map_bins` would set it up for any pixel: where a bin's ray
// passes the object's centre is affine in the bin's screen position about
// the centre's ray (the ray turns by `(I - c cᵀ) dD`, the closest approach
// moves by the distance times that; the next term is about rad / 2f of the
// radius, under 0.3% at 8 pixels across), and the two directions read are
// the nearest along the axis of the directions' grid whose fraction is
// nearer a half.
fn g8_instance(m: u32) {
    let sph = m13_spheres[2u * m];
    let radius = sph.w;
    if (radius <= 0.0) { return; }
    let centre = sph.xyz;
    let rel = centre - camera.eye.xyz;
    let depth = dot(rel, camera.forward.xyz);
    if (depth - radius <= 1e-2) { return; }
    let f = camera.params.y / (2.0 * camera.params.z);
    let rad = radius * f / depth;
    if (2.0 * rad > MAP_LIMIT_PX) { return; }
    let cp = cam_to_pixels(to_camera(view_world(), centre));
    if (cp.x + rad <= 0.0 || cp.y + rad <= 0.0 || cp.x - rad >= camera.params.x || cp.y - rad >= camera.params.y) {
        return;
    }
    // The rotation from world to object, and the maps' first layer.
    let rot = m13_spheres[2u * m + 1u];
    let q = vec4<f32>(rot.xyz, sqrt(max(0.0, 1.0 - dot(rot.xyz, rot.xyz))));
    let first = bitcast<u32>(rot.w) & 0x7fffffffu;
    let len = length(rel);
    let c = rel / len;
    let d = normalize(g8_rotate(q, c));
    let g = (oct_encode(d) + vec2<f32>(1.0)) * 0.5 * f32(MAP_DIRS) - vec2<f32>(0.5);
    let texels_per_bin = 0.25 * depth / f / (2.0 * radius / f32(MAP_TEXELS));
    let level = clamp(round(log2(max(texels_per_bin, 1e-6))), 0.0, 4.0);
    // The closest approach's change per pixel across and down, world then
    // object frame, over the radius.
    let tz = camera.params.z;
    let k = dot(c, camera.forward.xyz) * len / radius;
    let ex = camera.right.xyz * (2.0 * tz / camera.params.y);
    let ey = camera.up.xyz * (-2.0 * tz / camera.params.y);
    let wx = (ex - c * dot(c, ex)) * k;
    let wy = (ey - c * dot(c, ey)) * k;
    let qx = g8_rotate(q, wx);
    let qy = g8_rotate(q, wy);
    // The nearest direction (32×32 of them: as accurate as two of 16×16
    // blended, development).
    let ix = u32(clamp(round(g.x), 0.0, f32(MAP_DIRS) - 1.0));
    let iy = u32(clamp(round(g.y), 0.0, f32(MAP_DIRS) - 1.0));
    let layer = first + iy * MAP_DIRS + ix;
    let dk = oct_decode((vec2<f32>(f32(ix), f32(iy)) + vec2<f32>(0.5)) / f32(MAP_DIRS) * 2.0 - vec2<f32>(1.0));
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(dk.z) < 0.999);
    let u = normalize(cross(helper, dk));
    let v = cross(dk, u);
    let bits = u32(level) | (g8_stamp() << 3u) | (layer << 16u);
    m13_records[3u * m] = vec4<f32>(cp, rad, bitcast<f32>(bits));
    m13_records[3u * m + 1u] = vec4<f32>(dot(qx, u), dot(qx, v), dot(qy, u), dot(qy, v)) * 0.5;
    // M13 gate 9: the nearest distance, the light's direction in the
    // object's frame, the albedo and the sun's visibility (one ray from the
    // sphere's sunward point).
    if (g9_colour_on()) {
        let sun = g9_sun(centre);
        let lo = oct_encode(normalize(g8_rotate(q, sun.l))) * 0.5 + vec2<f32>(0.5);
        var vis = 1.0;
        if (G9_SHADOWS) { vis = select(0.0, 1.0, hw_segment_open(centre + sun.l * (radius * 1.05), sun.l, sun.dist)); }
        // The light's irradiance where the instance is over the eye's (the
        // light is near enough to vary across the scene), carried in the
        // albedo, so pixels light it by the eye's.
        let at_eye = g9_sun(camera.eye.xyz).e;
        let k = length(sun.e) / max(length(at_eye), 1e-20);
        let a = clamp(albedo[m].xyz * k, vec3<f32>(0.0), vec3<f32>(1.0));
        m13_records[3u * m + 2u] = vec4<f32>(len - radius, bitcast<f32>(pack2x16unorm(lo)), bitcast<f32>(pack4x8unorm(vec4<f32>(a, vis))), radius);
    }
}

// Object `m`'s map into the pixel's open bins, from its record (M13 gate
// 8): a bin is read where its coordinates fall in the map, sampled there
// (filtered by the hardware).
fn instance_map_rec(m: u32, pix: vec2<f32>, stamp: u32) {
    let r0 = m13_records[3u * m];
    let bits = bitcast<u32>(r0.w);
    if (((bits >> 3u) & 0x1fffu) != stamp) { return; }
    let rad = r0.z;
    let cp = r0.xy;
    let todo = g5_rect_bins(cp - vec2<f32>(rad), cp + vec2<f32>(rad), pix) & g5_open;
    if (todo == 0u) { return; }
    g8_sample_record(r0, m13_records[3u * m + 1u], m13_records[3u * m + 2u], todo, pix);
}

// A record's map (`r0`, `uv`: the record's two words) into the pixel's bins
// `todo`.
fn g8_sample_record(r0: vec4<f32>, uv: vec4<f32>, r2: vec4<f32>, bins: u32, pix: vec2<f32>) {
    var todo = bins;
    // M13 gate 9: behind the pixel's large object only its other bins.
    var l_obj = vec3<f32>(0.0);
    var albedo = vec4<f32>(0.0);
    if (g9_on) {
        todo &= g9_in_front(r2.x);
        l_obj = oct_decode(unpack2x16unorm(bitcast<u32>(r2.y)) * 2.0 - vec2<f32>(1.0));
        albedo = unpack4x8unorm(bitcast<u32>(r2.z));
    }
    let bits = bitcast<u32>(r0.w);
    let cp = r0.xy;
    if ((G5_SKIP & 16u) != 0u) { return; }
    let level = f32(bits & 7u);
    let layer = bits >> 16u;
    // Where the map can read other than zero: its rectangle of texels not
    // zero, widened for the level's coarser texels and bilinear reads
    // (1.5 × 2^level − 1 finest texels).
    let rect = g8_rects[layer];
    let margin = (1.5 * exp2(level) - 1.0) / f32(MAP_TEXELS);
    let rlo = max(vec2<f32>(f32(rect & 0xffu), f32((rect >> 8u) & 0xffu)) / f32(MAP_TEXELS) - vec2<f32>(margin), vec2<f32>(0.0));
    let rhi = min(vec2<f32>(f32((rect >> 16u) & 0xffu), f32(rect >> 24u)) / f32(MAP_TEXELS) + vec2<f32>(margin), vec2<f32>(1.0));
    let p0 = pix + vec2<f32>(0.125) - cp;
    while (todo != 0u) {
        let b = countTrailingZeros(todo);
        todo &= todo - 1u;
        let o = p0 + vec2<f32>(f32(b % 4u), f32(b / 4u)) * 0.25;
        let a = vec2<f32>(0.5) + uv.xy * o.x + uv.zw * o.y;
        if (any(a < rlo) || any(a > rhi)) { continue; }
        let texel = textureSampleLevel(cov_tex, cov_sampler, a, layer, level);
        let cov = texel.x;
        if ((G5_SKIP & 64u) != 0u) { g8_dummy += cov; continue; }
        // M13 gate 9: a bin showing the background skips the part of the
        // instance behind it (a boulder's buried part), by the map's depth.
        if (G9_BURIED && g9_on && g9_dbg < 3.0e38 && (((g9_l | g9_l2) >> b) & 1u) == 0u && cov > 0.0) {
            let surface = r2.x + r2.w * (1.0 + (texel.w * 255.0 / 254.0 * 2.0 - 1.0));
            if (surface > g9_bg_at(b) || G9_PROBE) { continue; }
        }
        if (g9_on) {
            let n = oct_decode(texel.yz * 2.0 - vec2<f32>(1.0));
            var c = albedo.xyz * (g9_e * (max(dot(n, l_obj), 0.0) * albedo.w + G9_AMBIENT)) / 3.14159265;
            if (G9_FLAT) { c = albedo.xyz * g9_e * G9_AMBIENT / 3.14159265; }
            if (G9_CONST_SMALL) { c = albedo.xyz * g9_e * (0.5 + G9_AMBIENT) / 3.14159265; }
            g9_c += (1.0 - pbin[b]) * cov * c / 16.0;
        }
        pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - cov);
    }
}

// --- The tiles' subtree roots (development after the first tile walk). ---

@group(0) @binding(87) var<storage, read_write> m13_roots: array<u32>;
// Per frame: the count of pixels tracing bin rays, then from word 8 the
// pixels' list and after it every pixel's word: covered bins (bit b: bin
// b), bit 16 if it traced bin rays, bit 17 if its corners resolved it.
@group(0) @binding(88) var<storage, read_write> m13_work: array<atomic<u32>>;
// The bin rays' dispatch arguments.
@group(0) @binding(89) var<storage, read_write> m13_args: array<u32>;
// Per object two words: its world bounding sphere about its coverage map's
// centre (radius 0: no map), and its rotation from world to object as a
// quaternion's x, y, z (w at least 0) with its maps' first layer's bits.
@group(0) @binding(90) var<storage, read> m13_spheres: array<vec4<f32>>;

fn m13_mask_word(pixel: u32) -> u32 {
    return 8u + u32(camera.params.x) * u32(camera.params.y) + pixel;
}

// M13 gate 9: the pixel's objects behind small ones, after the masks, 16
// words: its background's colour (three), distance and object (the first
// background object its centre's ray meets); then two large objects that
// are not background, each colour (three), distance and object; then their
// bins (the first's in the low 16 bits, the second's in the high); then the
// background hit's face normal (three words).
fn g9_large_word(pixel: u32) -> u32 {
    return 8u + 2u * u32(camera.params.x) * u32(camera.params.y) + 20u * pixel;
}

// M13 gate 9: each pixel corner's large hit shaded by the model (colour,
// then its object and distance), after the large objects.
fn g9_corner_word(pixel: u32) -> u32 {
    return 8u + 22u * u32(camera.params.x) * u32(camera.params.y) + 8u * pixel;
}

// Roots a tile lists at most. A tile's words: the count, then six per
// root: its node (bit 31 if it straddles the eye's plane, bit 30 if it is
// read from its coverage page, bit 29 if it lies below a node waiting for
// its page), its projected box in pixels as f32 bits (lo,
// hi; unless it straddles), so its pixels test it without loading or
// projecting the node, and its page; then (M13 gate 8, from word 380) up
// to 8 boxes of roots that may hold a large instance (4 words each), and in
// the last word their count, or 9 for the whole tile.
const G5_ROOTS: u32 = 63u;
const G5_TILE_WORDS: u32 = 416u;
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
    var n_large = 0u;
    // At each pop, n + sp is at most G5_ROOTS. A stack entry's bit 31: below
    // a node that wants a page (no page is wanted under it).
    while (sp > 0u) {
        sp--;
        let raw = stack_get(sp);
        let id = raw & 0x7fffffffu;
        var below = (raw & 0x80000000u) != 0u;
        let node = nodes[id];
        let stored = g8_boxes[id];
        var sb: ScreenBox;
        if (stored.x < 3.0e38) {
            sb = ScreenBox(stored.xy, stored.zw, 0.0, 0u);
        } else {
            sb = screen_box(node.lo.xyz, node.hi.xyz, v);
            if (sb.behind == 2u || frustum_misses_box(node.lo.xyz, node.hi.xyz)) { continue; }
        }
        if (sb.behind == 0u && (sb.hi.x <= lo.x || sb.hi.y <= lo.y || sb.lo.x >= hi.x || sb.lo.y >= hi.y)) { continue; }
        let extent = sb.hi - sb.lo;
        // M13 gate 6: the highest node on the path that fits a page at one
        // texel per bin and holds enough instances is read from its page if
        // it is resident and valid from here; otherwise it is requested and
        // walked as before.
        if (pages && !below && sb.behind == 0u && node.info.y == 0u && node.info.z == 0u && node.info.w == 0u
            && g6_range(id).y >= G6_MIN_ITEMS && g6_shape(extent * 4.0).x != 0u) {
            let page = atomicLoad(&g6_state[g6_table(id)]);
            if (page < G6_POOL && g8_page_valid(page)) {
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
            // An instance over MAP_LIMIT_PX across (or seen across the eye's
            // plane) can lie only under a root at least that wide (M13 gate
            // 8: the tile's corner rays are needed only then).
            if (g8_may_hold_large(id, node, sb.behind != 0u)) {
                if (sb.behind != 0u || n_large >= 8u) {
                    n_large = 9u;
                } else {
                    let at = base + 380u + 4u * n_large;
                    m13_roots[at] = bitcast<u32>(sb.lo.x);
                    m13_roots[at + 1u] = bitcast<u32>(sb.lo.y);
                    m13_roots[at + 2u] = bitcast<u32>(sb.hi.x);
                    m13_roots[at + 3u] = bitcast<u32>(sb.hi.y);
                    n_large++;
                }
            }
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
    m13_roots[base + G5_TILE_WORDS - 1u] = n_large;
}

// Per map (layer) the bounding rectangle of its finest texels not zero
// (bytes: u0, v0, u1, v1 past the last), M13 gate 8.
@group(0) @binding(100) var<storage, read> g8_rects: array<u32>;

// Development: the tile lists on.
const G8_LIST_ON: bool = true;

// The tile's list of leaf roots' instances (M13 gate 8): their records'
// two words, the count, and which roots it holds.
const G8_LIST: u32 = 128u;
var<workgroup> g8_list_a: array<vec4<f32>, 128>;
var<workgroup> g8_list_b: array<vec4<f32>, 128>;
var<workgroup> g8_list_c: array<vec4<f32>, 128>;
var<workgroup> g8_list_n: atomic<u32>;
var<workgroup> g8_listed: array<atomic<u32>, 2>;

// Development switches (bit 0: no bin rays, 1: no walks, 2: no map reads,
// 4: the maps' records tested but not read, 5: pages not read).
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
    let stamp = g8_stamp();
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
            let stored = g8_boxes[n];
            if (known) {
                sb = ScreenBox(root_box.xy, root_box.zw, to_camera(v, (node.lo.xyz + node.hi.xyz) * 0.5).z, 0u);
            } else if (stored.x < 3.0e38) {
                // In front of the eye: its projection, stored this frame (M13
                // gate 8); the pixel's side planes cut it as its rectangle does.
                sb = ScreenBox(stored.xy, stored.zw, to_camera(v, (node.lo.xyz + node.hi.xyz) * 0.5).z, 0u);
            } else {
                sb = screen_box(node.lo.xyz, node.hi.xyz, v);
                live = sb.behind != 2u && !frustum_misses_box(node.lo.xyz, node.hi.xyz);
            }
            // A node over full bins only (or none of the pixel's) cannot
            // change the pixel.
            if (live && sb.behind == 0u && (g5_rect_bins(sb.lo, sb.hi, pix) & g5_open) == 0u) { live = false; }
            // M13 gate 9: nor one wholly behind the pixel's large object
            // over its large bins only.
            if (live && g9_on && sb.behind == 0u) {
                let e = camera.eye.xyz;
                let near = length(max(node.lo.xyz - e, vec3<f32>(0.0)) + max(e - node.hi.xyz, vec3<f32>(0.0)));
                if ((g5_rect_bins(sb.lo, sb.hi, pix) & g5_open & g9_in_front(near)) == 0u) { live = false; }
            }
            // M13 gate 6: the highest node on the path that fits a page and
            // holds enough instances is read from its page if resident and
            // valid; otherwise it is asked for and walked as before.
            var paged = false;
            if (live && pages && !below && sb.behind == 0u && node.info.y == 0u && node.info.z == 0u && node.info.w == 0u
                && g6_shape((sb.hi - sb.lo) * 4.0).x != 0u && g6_range(n).y >= G6_MIN_ITEMS) {
                let page = atomicLoad(&g6_state[g6_table(n)]);
                if (page < G6_POOL && g8_page_valid(page)) {
                    g6_mark_used(page, atomicLoad(&g6_state[3]) + 1u);
                    g8_read_page(page, pix, vec4<f32>(sb.lo, sb.hi));
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
                        var before = 0.0;
                        if (g9_on) { for (var b = 0u; b < 16u; b++) { before += pbin[b]; } }
                        splat_bins(sb, pix, 1.0 - exp(-node_front_area(light.objects.x + n, dir) / sil), sil, false);
                        if (g9_on) {
                            // A splat's colour: its first instance's albedo
                            // by the model, facing up, unshadowed.
                            var after = 0.0;
                            for (var b = 0u; b < 16u; b++) { after += pbin[b]; }
                            let a = albedo[items[g6_range(n).x]].xyz;
                            let c = g9_colour(a, vec3<f32>(0.0, 0.0, 1.0), g9_sun(camera.eye.xyz), 1.0);
                            g9_c += (after - before) * c / 16.0;
                            g9_splats += 1u;
                        }
                        g5_refresh_open();
                        if (g5_open == 0u) { *on = false; }
                    }
                } else if (node.info.y > 0u) {
                    for (var k = node.info.x; k < node.info.x + node.info.y; k++) {
                        let m = items[k];
                        // Large instances (too large or near for their maps,
                        // which `instance_map_tex` skips) the bins' rays covered.
                        if ((G5_SKIP & 4u) != 0u) { continue; }
                        instance_map_rec(m, pix, stamp);
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
    let colour = g9_colour_on();
    if (gid.x + 1u < w && gid.y + 1u < h) {
        let a = vis[pixel];
        if (a.x != NONE && a.y == 1u && vis[pixel + 1u].x == a.x && vis[pixel + w].x == a.x && vis[pixel + w + 1u].x == a.x) {
            atomicStore(&m13_work[m13_mask_word(pixel)], 0xffffu | (1u << 17u));
            if (colour) { g9_large_hit(gid, pixel, true); }
            return;
        }
    }
    var near_large = false;
    // B's neighbourhood: the 16 corners of the pixel's 3×3 block, or with
    // G8_OWN_CORNERS its own four.
    let reach = select(1, 0, G8_OWN_CORNERS);
    for (var dy = -reach; dy <= 1 + reach; dy++) {
        for (var dx = -reach; dx <= 1 + reach; dx++) {
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
    if (colour) { g9_large_hit(gid, pixel, false); }
}

// A bin rays' claim of the sky (M13 gate 9).
const G9_SKY: u32 = 0xfffffffdu;

// Whether object `m` is part of the background (far larger than most;
// M13 gate 9).
fn g9_is_background(m: u32) -> bool {
    return (bitcast<u32>(m13_spheres[2u * m + 1u].w) & 0x80000000u) != 0u;
}

// M13 gate 9: the pixel's background and, if its corners resolved it to a
// large object, that object; each shaded once by the model at its hit on
// the pixel's centre ray (the background by a ray seeing only background
// objects).
fn g9_large_hit(gid: vec3<u32>, pixel: u32, resolved: bool) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    let e = camera.eye.xyz;
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    let p = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
    let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
    let dir = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
    let at = g9_large_word(pixel);
    for (var k = 3u; k < 15u; k += 5u) {
        atomicStore(&m13_work[at + k], bitcast<u32>(3.4e38));
        atomicStore(&m13_work[at + k + 1u], NONE);
    }
    atomicStore(&m13_work[at + 15u], 0u);
    let bg = g9_background(e, dir);
    if (bg.x != NONE) {
        let dist = bitcast<f32>(bg.z);
        // Shaded here only where none of the pixel's corners shows it.
        var shown = false;
        for (var k = 0u; k < 4u; k++) {
            let cx = min(gid.x + (k & 1u), w - 1u);
            let cy = min(gid.y + (k >> 1u), h - 1u);
            if (atomicLoad(&m13_work[g9_corner_word(cy * w + cx) + 3u]) == bg.x) { shown = true; }
        }
        var c = vec3<f32>(0.0);
        if (!shown) { c = g9_shade_hit(bg.x, bg.y, dist, e, dir); }
        g9_store_large(at, c, dist, bg.x);
        // Its face's plane, for each bin's distance to it.
        let v0 = tris[3u * bg.y].xyz;
        let n = normalize(normal_to_world(bg.x, cross(tris[3u * bg.y + 1u].xyz - v0, tris[3u * bg.y + 2u].xyz - v0)));
        atomicStore(&m13_work[at + 16u], bitcast<u32>(n.x));
        atomicStore(&m13_work[at + 17u], bitcast<u32>(n.y));
        atomicStore(&m13_work[at + 18u], bitcast<u32>(n.z));
    }
    // Where the pixel's corners disagree on what lies behind (the ground's
    // edge, the horizon), each bin's background ray: bins whose background
    // differs from the centre's claim it (the sky as G9_SKY).
    {
        var differs = false;
        for (var k = 0u; k < 4u; k++) {
            let cx = min(gid.x + (k & 1u), w - 1u);
            let cy = min(gid.y + (k >> 1u), h - 1u);
            if (atomicLoad(&m13_work[g9_corner_word(cy * w + cx) + 5u]) != bg.x) { differs = true; }
        }
        if (differs) {
            for (var b = 0u; b < 16u; b++) {
                let q = vec2<f32>(f32(gid.x), f32(gid.y)) + (vec2<f32>(f32(b % 4u), f32(b / 4u)) + vec2<f32>(0.5)) * 0.25;
                let qx = (q.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
                let qy = (1.0 - q.y / camera.params.y * 2.0) * tz;
                let bdir = normalize(camera.forward.xyz + camera.right.xyz * qx + camera.up.xyz * qy);
                let bb = g9_background(e, bdir);
                if (bb.x == bg.x) { continue; }
                var claimed = bb.x;
                if (bb.x == NONE) { claimed = G9_SKY; }
                for (var slot = 5u; slot < 15u; slot += 5u) {
                    let old = atomicLoad(&m13_work[at + slot + 4u]);
                    if (old == NONE) {
                        var c = g9_sky();
                        var dist = 3.4e38;
                        if (claimed != G9_SKY) {
                            dist = bitcast<f32>(bb.z);
                            c = g9_shade_hit(bb.x, bb.y, dist, e, bdir);
                        }
                        g9_store_large(at + slot, c, dist, claimed);
                    } else if (old != claimed) {
                        continue;
                    }
                    atomicOr(&m13_work[at + 15u], 1u << (b + select(16u, 0u, slot == 5u)));
                    break;
                }
            }
        }
    }
    if (resolved) {
        let hit = hw_trace(e, dir, 0.0);
        if (hit.x != NONE && hit.x != bg.x && g8_large(hit.x)) {
            let dist = bitcast<f32>(hit.z);
            g9_store_large(at + 5u, g9_shade_hit(hit.x, hit.y, dist, e, dir), dist, hit.x);
            atomicStore(&m13_work[at + 15u], 0xffffu);
        }
    }
}

fn g9_store_large(at: u32, c: vec3<f32>, dist: f32, object: u32) {
    atomicStore(&m13_work[at], bitcast<u32>(c.x));
    atomicStore(&m13_work[at + 1u], bitcast<u32>(c.y));
    atomicStore(&m13_work[at + 2u], bitcast<u32>(c.z));
    atomicStore(&m13_work[at + 3u], bitcast<u32>(dist));
    atomicStore(&m13_work[at + 4u], object);
}

// The nearest front face a ray meets among background objects only (TLAS
// mask bit 0, M13 gate 9): object, global triangle, distance bits.
fn g9_background(e: vec3<f32>, dir: vec3<f32>) -> vec3<u32> {
    var rq: ray_query;
    rayQueryInitialize(&rq, scene_as, RayDesc(HW_RAY_FLAGS, 0x1u, 0.0, 3.4e38, e, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == 0u) { return vec3<u32>(NONE, NONE, bitcast<u32>(3.4e38)); }
    let o = hw_object(hit.instance_custom_data, hit.geometry_index);
    return vec3<u32>(o.x, cam_tri(o.x, o.y, hit.primitive_index), bitcast<u32>(hit.t));
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
    if (g9_colour_on()) {
        // M13 gate 9: a bin whose nearest hit is a large object other than
        // the pixel's background shows it: the first two such objects, each
        // shaded once by the first ray to claim it.
        let hit = hw_trace(camera.eye.xyz, dir, 0.0);
        let at = g9_large_word(pixel);
        let bg = atomicLoad(&m13_work[at + 4u]);
        // A bin that sees nothing where the pixel's background is an object
        // shows the sky (as a claimed object of its own).
        var claimed = hit.x;
        if (hit.x == NONE) {
            if (bg == NONE) { return; }
            claimed = G9_SKY;
        } else if (!g8_large(hit.x) || hit.x == bg) {
            return;
        }
        for (var slot = 5u; slot < 15u; slot += 5u) {
            let claim = atomicCompareExchangeWeak(&m13_work[at + slot + 4u], NONE, claimed);
            if (claim.exchanged) {
                var dist = 3.4e38;
                var c = g9_sky();
                if (claimed != G9_SKY) {
                    dist = bitcast<f32>(hit.z);
                    c = g9_shade_hit(hit.x, hit.y, dist, camera.eye.xyz, dir);
                }
                atomicStore(&m13_work[at + slot], bitcast<u32>(c.x));
                atomicStore(&m13_work[at + slot + 1u], bitcast<u32>(c.y));
                atomicStore(&m13_work[at + slot + 2u], bitcast<u32>(c.z));
                atomicStore(&m13_work[at + slot + 3u], bitcast<u32>(dist));
            }
            if (claim.exchanged || claim.old_value == claimed) {
                // Bins: the mask word's low bits for the first, word 15's
                // for the second.
                atomicOr(&m13_work[at + 15u], 1u << (k + select(16u, 0u, slot == 5u)));
                return;
            }
        }
        return;
    }
    let hit = g8_first_object(camera.eye.xyz, dir);
    if (hit != NONE && g8_large(hit)) { atomicOr(&m13_work[m13_mask_word(pixel)], 1u << k); }
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
    // M13 gate 9: with colour, large bins are a background with a depth:
    // open to what lies in front of the pixel's large object.
    g9_on = g9_colour_on();
    var large_colour = vec3<f32>(0.0);
    var second_colour = vec3<f32>(0.0);
    var bg_colour = vec3<f32>(0.0);
    if (g9_on) {
        g9_l = word & 0xffffu;
        for (var b = 0u; b < 16u; b++) { pbin[b] = 0.0; }
        g5_open = 0xffffu;
        g9_e = g9_sun(camera.eye.xyz).e;
        if (inside) {
            // The background's distance at the pixel's corners that show it
            // (weights 0 elsewhere), for bins reading maps.
            let bg_obj = atomicLoad(&m13_work[g9_large_word(pixel) + 4u]);
            for (var k = 0u; k < 4u; k++) {
                let cx = min(gid.x + (k & 1u), w - 1u);
                let cy = min(gid.y + (k >> 1u), h - 1u);
                let cw = g9_corner_word(cy * w + cx);
                if (bg_obj != NONE && atomicLoad(&m13_work[cw + 3u]) == bg_obj) {
                    g9_cd[k] = bitcast<f32>(atomicLoad(&m13_work[cw + 4u]));
                    g9_cw[k] = 1.0;
                }
            }
            let at = g9_large_word(pixel);
            g9_bg_n = vec3<f32>(bitcast<f32>(atomicLoad(&m13_work[at + 16u])), bitcast<f32>(atomicLoad(&m13_work[at + 17u])), bitcast<f32>(atomicLoad(&m13_work[at + 18u])));
            g9_pix = pix;
            bg_colour = vec3<f32>(bitcast<f32>(atomicLoad(&m13_work[at])), bitcast<f32>(atomicLoad(&m13_work[at + 1u])), bitcast<f32>(atomicLoad(&m13_work[at + 2u])));
            g9_dbg = bitcast<f32>(atomicLoad(&m13_work[at + 3u]));
            large_colour = vec3<f32>(bitcast<f32>(atomicLoad(&m13_work[at + 5u])), bitcast<f32>(atomicLoad(&m13_work[at + 6u])), bitcast<f32>(atomicLoad(&m13_work[at + 7u])));
            g9_dl = bitcast<f32>(atomicLoad(&m13_work[at + 8u]));
            second_colour = vec3<f32>(bitcast<f32>(atomicLoad(&m13_work[at + 10u])), bitcast<f32>(atomicLoad(&m13_work[at + 11u])), bitcast<f32>(atomicLoad(&m13_work[at + 12u])));
            g9_d2 = bitcast<f32>(atomicLoad(&m13_work[at + 13u]));
            let masks = atomicLoad(&m13_work[at + 15u]);
            g9_l = masks & 0xffffu;
            g9_l2 = (masks >> 16u) & ~g9_l;
        }
    }
    var on = inside && g5_open != 0u && (G5_SKIP & 2u) == 0u;
    let base = (wid.y * ((w + 7u) / 8u) + wid.x) * G5_TILE_WORDS;
    let count = m13_roots[base];
    let pages = atomicLoad(&g6_state[9]) != 0u;
    set_pixel_frustum(pix);
    var visits = 0u;
    // M13 gate 8: the tile's leaf roots' instances, their records loaded
    // once by the tile's lanes (a root each) into a list its pixels read;
    // a leaf whose instances do not fit is walked as before.
    if (li == 0u) {
        atomicStore(&g8_list_n, 0u);
        for (var k = 0u; k < 2u; k++) { atomicStore(&g8_listed[k], 0u); }
    }
    workgroupBarrier();
    let stamp = g8_stamp();
    let tile_lo = vec2<f32>(f32(wid.x * 8u), f32(wid.y * 8u));
    for (var r = li; r < count; r += 64u) {
        let rw = m13_roots[base + 1u + 6u * r];
        if ((rw & 0xc0000000u) != 0u || !G8_LIST_ON) { continue; }
        let leaf = nodes[rw & 0x1fffffffu].info;
        if (leaf.y == 0u) { continue; }
        // The leaf's instances read this frame and over the tile.
        var take = 0u;
        for (var k = 0u; k < leaf.y; k++) {
            let r0 = m13_records[3u * items[leaf.x + k]];
            let b = bitcast<u32>(r0.w);
            if (((b >> 3u) & 0x1fffu) != stamp) { continue; }
            if (r0.x + r0.z <= tile_lo.x || r0.y + r0.z <= tile_lo.y || r0.x - r0.z >= tile_lo.x + 8.0 || r0.y - r0.z >= tile_lo.y + 8.0) { continue; }
            take |= 1u << k;
        }
        let n_take = countOneBits(take);
        let at = atomicAdd(&g8_list_n, n_take);
        var i = 0u;
        for (var k = 0u; k < leaf.y; k++) {
            if ((take & (1u << k)) == 0u) { continue; }
            if (at + n_take <= G8_LIST) {
                let m = items[leaf.x + k];
                g8_list_a[at + i] = m13_records[3u * m];
                g8_list_b[at + i] = m13_records[3u * m + 1u];
                g8_list_c[at + i] = m13_records[3u * m + 2u];
            } else if (at + i < G8_LIST) {
                // Reserved but not used: a leaf that did not fit is walked.
                g8_list_a[at + i] = vec4<f32>(0.0, 0.0, -1.0, 0.0);
            }
            i++;
        }
        if (at + n_take <= G8_LIST) { atomicOr(&g8_listed[r / 32u], 1u << (r % 32u)); }
    }
    workgroupBarrier();
    let listed = min(atomicLoad(&g8_list_n), G8_LIST);
    if (on && (G5_SKIP & 4u) == 0u) {
        for (var i = 0u; i < listed; i++) {
            let r0 = g8_list_a[i];
            if (r0.z <= 0.0) { continue; }
            let todo = g5_rect_bins(r0.xy - vec2<f32>(r0.z), r0.xy + vec2<f32>(r0.z), pix) & g5_open;
            if (todo == 0u) { continue; }
            g8_sample_record(r0, g8_list_b[i], g8_list_c[i], todo, pix);
        }
        g5_refresh_open();
        if (g5_open == 0u) { on = false; }
    }
    let listed_roots = vec2<u32>(atomicLoad(&g8_listed[0]), atomicLoad(&g8_listed[1]));
    for (var r = 0u; r < count; r++) {
        if (!g5_any(on)) { break; }
        if ((listed_roots[r / 32u] & (1u << (r % 32u))) != 0u) { continue; }
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
                g8_read_page(m13_roots[at + 5u], pix, rb);
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
    if (inside && g9_on) {
        // What each bin leaves shows its large object or the sky.
        var sky = g9_sky();
        if (G9_RED_SKY) { sky = vec3<f32>(sky.x * 20.0, 0.0, 0.0); }
        // A bin shows its large object, else the pixel's background, else
        // the sky; a large object's or the background's colour is its
        // corners' colours where they show it, bilinearly at the bin's
        // centre (its own shading where none does).
        let at = g9_large_word(pixel);
        let objs = vec3<u32>(atomicLoad(&m13_work[at + 4u]), atomicLoad(&m13_work[at + 9u]), atomicLoad(&m13_work[at + 14u]));
        var cc: array<vec3<f32>, 4>;
        var co: array<u32, 4>;
        for (var k = 0u; k < 4u; k++) {
            let cx = min(gid.x + (k & 1u), w - 1u);
            let cy = min(gid.y + (k >> 1u), h - 1u);
            let cw = g9_corner_word(cy * w + cx);
            co[k] = atomicLoad(&m13_work[cw + 3u]);
            cc[k] = vec3<f32>(bitcast<f32>(atomicLoad(&m13_work[cw])), bitcast<f32>(atomicLoad(&m13_work[cw + 1u])), bitcast<f32>(atomicLoad(&m13_work[cw + 2u])));
        }
        for (var b = 0u; b < 16u; b++) {
            var back = sky;
            var obj = NONE;
            if (((g9_l >> b) & 1u) != 0u) {
                back = large_colour;
                obj = objs.y;
            } else if (((g9_l2 >> b) & 1u) != 0u) {
                back = second_colour;
                obj = objs.z;
            } else if (objs.x != NONE) {
                back = bg_colour;
                obj = objs.x;
            }
            if (obj != NONE && G9_CORNERS) {
                let f = (vec2<f32>(f32(b % 4u), f32(b / 4u)) + vec2<f32>(0.5)) * 0.25;
                var sum = vec3<f32>(0.0);
                var wsum = 0.0;
                for (var k = 0u; k < 4u; k++) {
                    if (co[k] != obj) { continue; }
                    let wk = select(1.0 - f.x, f.x, (k & 1u) != 0u) * select(1.0 - f.y, f.y, (k >> 1u) != 0u);
                    sum += wk * cc[k];
                    wsum += wk;
                }
                if (wsum > 1e-3) { back = sum / wsum; }
            }
            g9_c += (1.0 - pbin[b]) * back / 16.0;
        }
        colors[pixel] = vec4<f32>(g9_c, 1.0);
    } else if (inside) {
        var sum = 0.0;
        for (var b = 0u; b < 16u; b++) { sum += pbin[b]; }
        colors[pixel] = vec4<f32>(sum / 16.0 + g8_dummy * 1e-30, select(f32(visits), 0.0, resolved), f32(2u * count + select(0u, 1u, traced)), 1.0);
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
// frame's; 16–18 the last frame's eye, 20–22 the eye's last step, 24 1
// once an eye is kept, 25 refreshes asked this frame, 26 refreshes baked
// last frame), the requests' list, the pages the bakes write, the
// refreshes' list, then per node its page (NONE, or
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

const G6_POOL: u32 = 8192u;
// Texels a page (M13 gate 6: 512; gate 8: 1,024, shapes 4×256 to 256×4).
const G6_TEXELS: u32 = 2048u;
const G6_REQUESTS: u32 = 4096u;
const G6_META: u32 = 32u;
// Nodes with fewer instances read the instances' shared maps instead.
const G6_MIN_ITEMS: u32 = 8u;
const G6_FAIL: u32 = 0xfffffffeu;

fn g6_nodes() -> u32 { return g6_fixed[0]; }
fn g6_range(n: u32) -> vec2<u32> { return vec2<u32>(g6_fixed[1u + 2u * n], g6_fixed[2u + 2u * n]); }
fn g6_place(m: u32) -> u32 { return g6_fixed[1u + 2u * g6_nodes() + m]; }
fn g6_slot(i: u32) -> u32 { return G6_META + G6_REQUESTS + i; }
fn g6_low(i: u32) -> u32 { return G6_META + 2u * G6_REQUESTS + i; }
fn g6_table(n: u32) -> u32 { return G6_META + 3u * G6_REQUESTS + n; }
fn g6_mark(n: u32) -> u32 { return G6_META + 3u * G6_REQUESTS + g6_nodes() + n; }
fn g6_used(page: u32) -> u32 { return G6_META + 3u * G6_REQUESTS + 2u * g6_nodes() + page; }

// The page shape (width, height in texels; 512 texels, width 4 to 128) a
// node `extent` bins across fits, or zero.
fn g6_shape(extent: vec2<f32>) -> vec2<u32> {
    let w = max(ceil(extent.x), 1.0);
    var sw = 4u;
    loop {
        if (f32(sw) >= w || sw >= G6_TEXELS / 4u) { break; }
        sw *= 2u;
    }
    let sh = G6_TEXELS / sw;
    if (f32(sw) < w || ceil(extent.y) > f32(sh)) { return vec2<u32>(0u); }
    return vec2<u32>(sw, sh);
}

// How far `page`'s contents (a node centred at `centre`) have shifted
// since its bake, in bins (2 f |ΔE⊥| L / D²), or 1e9 if its texels are no
// longer 0.7–1.4 bins wide: it may be read while at most a quarter.
fn g6_shift(page: u32, centre: vec3<f32>) -> f32 {
    let h0 = g6_hdr[5u * page];
    let h1 = g6_hdr[5u * page + 1u];
    let h3 = g6_hdr[5u * page + 3u];
    let h4 = g6_hdr[5u * page + 4u];
    let f = camera.params.y / (2.0 * camera.params.z);
    let de = camera.eye.xyz - h4.xyz;
    let perp = de - h3.xyz * dot(de, h3.xyz);
    let shift = 2.0 * f * length(perp) * h1.w / (h0.w * h0.w);
    let ratio = (h0.w / h4.w) / (length(centre - camera.eye.xyz) / f);
    if (ratio < 0.7 || ratio > 1.4) { return 1e9; }
    return shift;
}

// A page read past this shift (bins) is asked again on the refreshes'
// list (M13 gate 7), baked with the budget the frame's requests leave.
const G6_REFRESH: f32 = 0.2;

// Counts `page` among the pages read this frame (`frame`: the frame
// counter plus one), once.
fn g6_mark_used(page: u32, frame: u32) {
    if (atomicLoad(&g6_state[g6_used(page)]) == frame) { return; }
    if (atomicExchange(&g6_state[g6_used(page)], frame) != frame) { atomicAdd(&g6_state[7], 1u); }
}

// Asks for node `n`'s page, once a frame.
fn g6_request(n: u32) {
    g6_request_on(n, 0u, 0u);
}

// Asks for node `n`'s page on the list counted in meta word `counter`, its
// entries from `G6_META + offset`, once a frame across both lists.
fn g6_request_on(n: u32, counter: u32, offset: u32) {
    if (atomicLoad(&g6_state[g6_mark(n)]) != 0u) { return; }
    if (atomicExchange(&g6_state[g6_mark(n)], 1u) != 0u) { return; }
    let i = atomicAdd(&g6_state[counter], 1u);
    if (i < G6_REQUESTS) {
        atomicStore(&g6_state[G6_META + offset + i], n);
    } else {
        atomicStore(&g6_state[g6_mark(n)], 0u);
    }
}

fn g6_texel(page: u32, i: i32, j: i32, sw: i32, sh: i32) -> f32 {
    return g9_texel(page, i, j, sw, sh).w;
}

// A page texel (a word: coverage, then the mean of its hits' colours as
// square roots of the page scale's share, bytes): (colour times coverage,
// coverage) (M13 gate 9).
fn g9_texel(page: u32, i: i32, j: i32, sw: i32, sh: i32) -> vec4<f32> {
    if (i < 0 || j < 0 || i >= sw || j >= sh) { return vec4<f32>(0.0); }
    let t = unpack4x8unorm(g6_texels[G6_TEXELS * page + u32(j * sw + i)]);
    return vec4<f32>(t.yzw * t.yzw * t.x, t.x);
}

// After the pixels' pass, one workgroup: the bakes this frame (at most the
// budget, the rest dropped and asked again), each a page (its own, or the
// next in the pool's ring, evicting its owner), their dispatch (128
// workgroups a page), the counters' turn to the next frame, the eye's step.
@compute @workgroup_size(256)
fn hw_m13_page_args(@builtin(local_invocation_index) li: u32) {
    let c = min(atomicLoad(&g6_state[0]), G6_REQUESTS);
    let nh = min(c, atomicLoad(&g6_state[4]));
    let cl = min(atomicLoad(&g6_state[25]), G6_REQUESTS);
    let nl = min(cl, atomicLoad(&g6_state[4]) - nh);
    let nb = nh + nl;
    // Requests dropped (past the budget) are asked again.
    for (var i = nh + li; i < c; i += 256u) {
        atomicStore(&g6_state[g6_mark(atomicLoad(&g6_state[G6_META + i]))], 0u);
    }
    for (var i = nl + li; i < cl; i += 256u) {
        atomicStore(&g6_state[g6_mark(atomicLoad(&g6_state[g6_low(i)]))], 0u);
    }
    storageBarrier();
    workgroupBarrier();
    // Refreshes after the requests.
    for (var i = li; i < nl; i += 256u) {
        atomicStore(&g6_state[G6_META + nh + i], atomicLoad(&g6_state[g6_low(i)]));
    }
    storageBarrier();
    workgroupBarrier();
    for (var i = li; i < nb; i += 256u) {
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
    workgroupBarrier();
    if (li != 0u) { return; }
    atomicStore(&g6_state[2], nb);
    atomicStore(&g6_state[6], c);
    atomicStore(&g6_state[0], 0u);
    atomicStore(&g6_state[26], nl);
    atomicStore(&g6_state[25], 0u);
    atomicStore(&g6_state[8], atomicLoad(&g6_state[7]));
    atomicStore(&g6_state[7], 0u);
    atomicStore(&g6_state[12], atomicLoad(&g6_state[5]));
    atomicStore(&g6_state[5], 0u);
    atomicStore(&g6_state[14], atomicLoad(&g6_state[13]));
    atomicStore(&g6_state[13], 0u);
    atomicAdd(&g6_state[3], 1u);
    // The eye's step since the last frame, for the bakes' lead.
    let e = camera.eye.xyz;
    var step = vec3<f32>(0.0);
    if (atomicLoad(&g6_state[24]) != 0u) {
        step = e - vec3<f32>(bitcast<f32>(atomicLoad(&g6_state[16])), bitcast<f32>(atomicLoad(&g6_state[17])), bitcast<f32>(atomicLoad(&g6_state[18])));
    }
    for (var k = 0u; k < 3u; k++) {
        atomicStore(&g6_state[16u + k], bitcast<u32>(e[k]));
        atomicStore(&g6_state[20u + k], bitcast<u32>(step[k]));
    }
    atomicStore(&g6_state[24], 1u);
    // A workgroup per page and 4 texels (64 rays).
    g6_args[0] = nb;
    g6_args[1] = G6_TEXELS / 4u;
    g6_args[2] = 1u;
}

// Whether a ray from `e` along `dir` meets one of the instances in item
// range `r` inside the box [lo, hi]: opaque queries, stepping past hits on
// other instances (in at most 16 queries).
fn g6_ray_hits(e: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, r: vec2<u32>) -> bool {
    return g9_ray_hit(e, dir, lo, hi, r).x != NONE;
}

// `g6_ray_hits`' hit (M13 gate 9): object, global triangle and distance
// from `e` as bits (object NONE: none).
fn g9_ray_hit(e: vec3<f32>, dir: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, r: vec2<u32>) -> vec3<u32> {
    let none = vec3<u32>(NONE, NONE, 0u);
    let inv = vec3<f32>(1.0) / dir;
    let ta = (lo - e) * inv;
    let tb = (hi - e) * inv;
    let entry = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
    let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    // From the box's entry: traversal from the eye would meet everything
    // the ray passes before it.
    let start = e + dir * entry;
    let t_end = t1 - entry;
    var t0 = 0.0;
    for (var k = 0u; k < 16u; k++) {
        if (t_end <= t0) { return none; }
        let hit = g6_first_hit(start, dir, t0, t_end);
        if (hit.y == NONE) { return none; }
        let p = g6_place(hit.y);
        if (p >= r.x && p < r.x + r.y) {
            return vec3<u32>(hit.y, hit.z, bitcast<u32>(entry + bitcast<f32>(hit.x)));
        }
        t0 = bitcast<f32>(hit.x) * (1.0 + 1e-6) + 1e-4;
    }
    return none;
}

// The nearest hit on [t0, t1] (front and back faces alike): its distance's
// bits, object (NONE: none) and global triangle.
fn g6_first_hit(e: vec3<f32>, dir: vec3<f32>, t0: f32, t1: f32) -> vec3<u32> {
    var rq: ray_query;
    // FORCE_OPAQUE.
    rayQueryInitialize(&rq, scene_as, RayDesc(0x1u, 0xffu, t0, t1, e, dir));
    rayQueryProceed(&rq);
    let hit = rayQueryGetCommittedIntersection(&rq);
    if (hit.kind == 0u) { return vec3<u32>(0u, NONE, NONE); }
    let o = hw_object(hit.instance_custom_data, hit.geometry_index);
    return vec3<u32>(bitcast<u32>(hit.t), o.x, cam_tri(o.x, o.y, hit.primitive_index));
}

var<workgroup> g6_bake: array<vec4<f32>, 5>;
var<workgroup> g6_bake_ok: u32;
var<workgroup> g6_hits: array<atomic<u32>, 4>;
// M13 gate 9: the 4 texels' colour sums (8-bit fixed point a ray, of the
// page scale), red, green, blue.
var<workgroup> g9_sums: array<atomic<u32>, 12>;

// A page's header for the node box [lo, hi] baked from eye `e`, into
// `g6_bake`: on the plane through the node's centre facing `e`, a texel a
// bin at its distance, the projection centred in the smallest shape that
// fits. False if none fits.
fn g6_header(lo: vec3<f32>, hi: vec3<f32>, e: vec3<f32>, n: u32) -> bool {
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
    let a0 = (amin + amax) * 0.5 - vec2<f32>(f32(shape.x), f32(shape.y)) * 0.5;
    g6_bake[0] = vec4<f32>(centre + u * (t * a0.x) + v * (t * a0.y), dist);
    g6_bake[1] = vec4<f32>(u / t, zmax - zmin);
    g6_bake[2] = vec4<f32>(v / t, bitcast<f32>(shape.x | (shape.y << 16u)));
    g6_bake[3] = vec4<f32>(d, bitcast<f32>(n));
    g6_bake[4] = vec4<f32>(e, f);
    return shape.x != 0u && zmin > 1e-2;
}

// M13 gate 7: where to bake node box [lo, hi]'s page from, led along the
// eye's last step `step` by half the steps after which its contents would
// shift a quarter of a bin (`D² / 8 f L` across the line of sight), so the
// page is valid at once and about twice as long along the path; the lead
// is cut so the distance changes by at most 15%.
fn g6_lead(lo: vec3<f32>, hi: vec3<f32>, e: vec3<f32>, step: vec3<f32>) -> vec3<f32> {
    let rel = (lo + hi) * 0.5 - e;
    let dist = length(rel);
    let d = rel / dist;
    let across = length(step - d * dot(step, d));
    if (across < 1e-7) { return e; }
    // The box's depth along the line of sight bounds L.
    let ext = (hi - lo) * 0.5;
    let l = 2.0 * dot(ext, abs(d));
    let f = camera.params.y / (2.0 * camera.params.z);
    let reach = dist * dist / (8.0 * f * max(l, 1e-3));
    var steps = 0.5 * reach / across;
    let along = abs(dot(step, d));
    if (along > 0.0) { steps = min(steps, 0.15 * dist / along); }
    return e + step * steps;
}

// 128 workgroups per page baked this frame, 4 texels (64 rays, one an
// invocation) each: the requested node's page (`g6_header`) from the eye
// led along its motion (`g6_lead`; from the current eye if the led page
// would fit no shape); each texel the share of 4×4 rays through it that
// meet the node's contents (8,192 rays a page). The first workgroup writes
// the header.
@compute @workgroup_size(64)
fn hw_m13_page_bake(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = wg.x;
    let part = wg.y;
    let n = atomicLoad(&g6_state[G6_META + i]);
    let page = atomicLoad(&g6_state[g6_slot(i)]);
    let node = nodes[n];
    let lo = node.lo.xyz;
    let hi = node.hi.xyz;
    if (li < 4u) { atomicStore(&g6_hits[li], 0u); }
    if (li < 12u) { atomicStore(&g9_sums[li], 0u); }
    if (li == 0u) {
        let step = vec3<f32>(bitcast<f32>(atomicLoad(&g6_state[20])), bitcast<f32>(atomicLoad(&g6_state[21])), bitcast<f32>(atomicLoad(&g6_state[22])));
        var ok = g6_header(lo, hi, g6_lead(lo, hi, camera.eye.xyz, step), n);
        if (!ok) { ok = g6_header(lo, hi, camera.eye.xyz, n); }
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
    let e = g6_bake[4].xyz;
    // Back from texels to world: the axes over the texel's square.
    let tu = g6_bake[1].xyz;
    let tv = g6_bake[2].xyz;
    let u = tu / dot(tu, tu);
    let v = tv / dot(tv, tv);
    let sw = bitcast<u32>(g6_bake[2].w) & 0xffffu;
    let k = part * 4u + li / 16u;
    let ray = li % 16u;
    let aim = o + u * (f32(k % sw) + (f32(ray % 4u) + 0.5) * 0.25) + v * (f32(k / sw) + (f32(ray / 4u) + 0.5) * 0.25);
    let ray_dir = normalize(aim - e);
    var hit = g9_ray_hit(e, ray_dir, lo, hi, g6_range(n));
    // M13 gate 9: a hit behind the background (a boulder's buried part)
    // does not count.
    if (hit.x != NONE && g9_colour_on() && bitcast<f32>(g9_background(e, ray_dir).z) < bitcast<f32>(hit.z)) {
        hit.x = NONE;
    }
    if (hit.x != NONE) {
        atomicAdd(&g6_hits[li / 16u], 1u);
        // M13 gate 9: the hit shaded by the model, in the page's scale.
        if (g9_colour_on()) {
            // Square roots of the page scale's share, for dark colours'
            // precision in a byte.
            let c = sqrt(clamp(g9_shade_hit(hit.x, hit.y, bitcast<f32>(hit.z), e, ray_dir) / g9_scale(), vec3<f32>(0.0), vec3<f32>(1.0)));
            let q = li / 16u;
            atomicAdd(&g9_sums[3u * q], u32(round(c.x * 255.0)));
            atomicAdd(&g9_sums[3u * q + 1u], u32(round(c.y * 255.0)));
            atomicAdd(&g9_sums[3u * q + 2u], u32(round(c.z * 255.0)));
        }
    }
    workgroupBarrier();
    // A texel a word: its coverage, then its hits' mean colour (bytes).
    if (li < 4u) {
        let hits = atomicLoad(&g6_hits[li]);
        var word = u32(round(f32(hits) / 16.0 * 255.0));
        if (hits > 0u) {
            for (var c = 0u; c < 3u; c++) {
                word |= u32(round(f32(atomicLoad(&g9_sums[3u * li + c])) / f32(hits))) << (8u * (c + 1u));
            }
        }
        g6_texels[G6_TEXELS * page + part * 4u + li] = word;
    }
}

// --- M13 gate 8: page records, once a frame. ---

// Per page three words: the homography from screen position (x, y, 1) to
// its texels (across, down, and the denominator, each a row of three), the
// first two rows' fourth word its shape (width, height), the third's 1 if
// the page may be read this frame.
@group(0) @binding(97) var<storage, read_write> g8_pages: array<vec4<f32>>;

fn g8_page_valid(page: u32) -> bool {
    return g8_pages[3u * page + 2u].w > 0.5;
}

// One invocation per page of the pool, before the roots pass (M13 gate 8):
// whether the page, if it is its node's, may be read from the eye
// (`g6_shift`); a page read last frame past G6_REFRESH is asked again on
// the refreshes' list; and the homography from screen position to texels:
// a texel coordinate along axis `A` is `(α n + β A)·D / n·D` for the ray
// `D(p)`, affine in the pixel `p` (`α = (e − O)·A`, `β = (O − e)·n`).
fn g8_page(page: u32) {
    g8_pages[3u * page + 2u].w = 0.0;
    let n = bitcast<u32>(g6_hdr[5u * page + 3u].w);
    if (n >= g6_nodes() || atomicLoad(&g6_state[g6_table(n)]) != page) { return; }
    let node = nodes[n];
    let shift = g6_shift(page, (node.lo.xyz + node.hi.xyz) * 0.5);
    if (shift > 0.25) { return; }
    // Read last frame (its use mark is the frame counter now).
    if (atomicLoad(&g6_state[g6_used(page)]) == atomicLoad(&g6_state[3])) {
        let q = u32(shift * 1e4);
        if (q > atomicLoad(&g6_state[13])) { atomicMax(&g6_state[13], q); }
        if (shift > G6_REFRESH) { g6_request_on(n, 25u, 2u * G6_REQUESTS); }
    }
    let o = g6_hdr[5u * page].xyz;
    let ua = g6_hdr[5u * page + 1u].xyz;
    let h2 = g6_hdr[5u * page + 2u];
    let va = h2.xyz;
    let nrm = g6_hdr[5u * page + 3u].xyz;
    let e = camera.eye.xyz;
    let tz = camera.params.z;
    let aspect = camera.params.x / camera.params.y;
    // D(p) = d0 + dx p.x + dy p.y.
    let dx = camera.right.xyz * (2.0 * tz * aspect / camera.params.x);
    let dy = camera.up.xyz * (-2.0 * tz / camera.params.y);
    let d0 = camera.forward.xyz - camera.right.xyz * (tz * aspect) + camera.up.xyz * tz;
    let beta = dot(o - e, nrm);
    let au = dot(e - o, ua) * nrm + beta * ua;
    let av = dot(e - o, va) * nrm + beta * va;
    let shape = bitcast<u32>(h2.w);
    g8_pages[3u * page] = vec4<f32>(dot(au, dx), dot(au, dy), dot(au, d0), f32(shape & 0xffffu));
    g8_pages[3u * page + 1u] = vec4<f32>(dot(av, dx), dot(av, dy), dot(av, d0), f32(shape >> 16u));
    g8_pages[3u * page + 2u] = vec4<f32>(dot(nrm, dx), dot(nrm, dy), dot(nrm, d0), 1.0);
}

// `g6_read_page` from the page's record: per open bin, its texel
// coordinates through the homography, bilinearly.
fn g8_read_page(page: u32, pix: vec2<f32>, bx: vec4<f32>) {
    var todo = g5_rect_bins(bx.xy, bx.zw, pix) & g5_open;
    if (todo == 0u || (G5_SKIP & 32u) != 0u) { return; }
    // M13 gate 9: behind the pixel's large object only its other bins.
    var scale = 0.0;
    if (g9_on) {
        todo &= g9_in_front(g6_hdr[5u * page].w - 0.5 * g6_hdr[5u * page + 1u].w);
        scale = g9_scale();
    }
    let ha = g8_pages[3u * page];
    let hb = g8_pages[3u * page + 1u];
    let hw = g8_pages[3u * page + 2u];
    let sw = i32(ha.w);
    let sh = i32(hb.w);
    while (todo != 0u) {
        let b = countTrailingZeros(todo);
        todo &= todo - 1u;
        let p = vec3<f32>(pix + vec2<f32>(f32(b % 4u), f32(b / 4u)) * 0.25 + vec2<f32>(0.125), 1.0);
        let c = vec2<f32>(dot(ha.xyz, p), dot(hb.xyz, p)) / dot(hw.xyz, p) - vec2<f32>(0.5);
        let f0 = floor(c);
        let fr = c - f0;
        let i = vec2<i32>(f0);
        let top = mix(g9_texel(page, i.x, i.y, sw, sh), g9_texel(page, i.x + 1, i.y, sw, sh), fr.x);
        let bottom = mix(g9_texel(page, i.x, i.y + 1, sw, sh), g9_texel(page, i.x + 1, i.y + 1, sw, sh), fr.x);
        let t = mix(top, bottom, fr.y);
        let cov = clamp(t.w, 0.0, 1.0);
        if (g9_on && cov > 0.0) {
            g9_c += (1.0 - pbin[b]) * t.xyz * scale / 16.0;
        }
        pbin[b] = 1.0 - (1.0 - pbin[b]) * (1.0 - cov);
    }
}

// --- M13 gate 8: the hierarchy's projected boxes, once a frame. ---

// Per object-hierarchy node the radius of its largest instance (1e30 if it
// holds one without a map, always large).
@group(0) @binding(99) var<storage, read> g8_largest: array<f32>;

// Whether node `n` (its box's projection straddling the eye's plane if
// `straddles`) may hold an instance over MAP_LIMIT_PX across: its largest
// instance at the box's nearest depth.
fn g8_may_hold_large(n: u32, node: Node, straddles: bool) -> bool {
    if (straddles) { return true; }
    let c = (node.lo.xyz + node.hi.xyz) * 0.5 - camera.eye.xyz;
    let h = (node.hi.xyz - node.lo.xyz) * 0.5;
    let near = dot(c, camera.forward.xyz) - dot(h, abs(camera.forward.xyz));
    if (near <= 1e-2) { return true; }
    let f = camera.params.y / (2.0 * camera.params.z);
    return 2.0 * g8_largest[n] * f / near > MAP_LIMIT_PX;
}

// Per object-hierarchy node its box's projection in pixels (lo, hi), or lo.x
// 3.4e38 if a corner lies behind the eye (walks then project it themselves,
// with the pixel's or tile's frustum).
@group(0) @binding(98) var<storage, read_write> g8_boxes: array<vec4<f32>>;

// Node `n`'s projection (M13 gate 8).
fn g8_node_box(n: u32) {
    let node = nodes[n];
    let v = view_world();
    var lo = vec2<f32>(3.4e38);
    var hi = vec2<f32>(-3.4e38);
    for (var k = 0u; k < 8u; k++) {
        let c = vec3<f32>(select(node.lo.x, node.hi.x, (k & 1u) != 0u), select(node.lo.y, node.hi.y, (k & 2u) != 0u), select(node.lo.z, node.hi.z, (k & 4u) != 0u));
        let q = to_camera(v, c);
        if (q.z <= 1e-3) {
            g8_boxes[n] = vec4<f32>(3.4e38);
            return;
        }
        let p = cam_to_pixels(q);
        lo = min(lo, p);
        hi = max(hi, p);
    }
    g8_boxes[n] = vec4<f32>(lo, hi);
}

// M13 gate 8's once-a-frame pass, one invocation per page of the pool, then
// per node of the object hierarchy, then per object: page records, the
// nodes' projections, instance records.
@compute @workgroup_size(64)
fn hw_m13_prep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = (wg.y * nwg.x + wg.x) * 64u + li;
    let nodes_count = arrayLength(&g8_boxes);
    if (t < G6_POOL) {
        g8_page(t);
    } else if (t < G6_POOL + nodes_count) {
        if (G8_NODE_BOXES) { g8_node_box(t - G6_POOL); } else { g8_boxes[t - G6_POOL] = vec4<f32>(3.4e38); }
    } else if (t < G6_POOL + nodes_count + arrayLength(&m13_spheres) / 2u) {
        g8_instance(t - G6_POOL - nodes_count);
    }
}

// --- M13 gate 9: appearance, with one shading model everywhere. ---

// Whether the pass shades (M13 gate 9; the pages' state word 27).
fn g9_colour_on() -> bool {
    return atomicLoad(&g6_state[27]) != 0u;
}

// The pixel's colour so far, its large bins, its large object's distance,
// and the light's normal irradiance (M13 gate 9).
var<private> g9_on: bool = false;
var<private> g9_c: vec3<f32> = vec3<f32>(0.0);
var<private> g9_l: u32 = 0u;
var<private> g9_dl: f32 = 3.4e38;
// The pixel's second large object's bins and distance.
var<private> g9_l2: u32 = 0u;
var<private> g9_d2: f32 = 3.4e38;
// Development: splats this pixel read.
var<private> g9_splats: u32 = 0u;
// The pixel's background's distance (3.4e38: the sky).
var<private> g9_dbg: f32 = 3.4e38;
// The background's distance at the pixel's corners (weight 1 where a corner
// shows it).
var<private> g9_cd: vec4<f32> = vec4<f32>(0.0);
var<private> g9_cw: vec4<f32> = vec4<f32>(0.0);

// The background hit's face normal, and the pixel (M13 gate 9).
var<private> g9_bg_n: vec3<f32> = vec3<f32>(0.0);
var<private> g9_pix: vec2<f32> = vec2<f32>(0.0);

// The background's distance along bin `b`'s ray: to the plane of its hit
// at the pixel's centre (exact for a flat background).
fn g9_bg_at(b: u32) -> f32 {
    if (G9_PLANE && dot(g9_bg_n, g9_bg_n) > 0.5) {
        let tz = camera.params.z;
        let aspect = camera.params.x / camera.params.y;
        let pc = g9_pix + vec2<f32>(0.5);
        let sxc = (pc.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
        let syc = (1.0 - pc.y / camera.params.y * 2.0) * tz;
        let dc = normalize(camera.forward.xyz + camera.right.xyz * sxc + camera.up.xyz * syc);
        let p = g9_pix + (vec2<f32>(f32(b % 4u), f32(b / 4u)) + vec2<f32>(0.5)) * 0.25;
        let sx = (p.x / camera.params.x * 2.0 - 1.0) * tz * aspect;
        let sy = (1.0 - p.y / camera.params.y * 2.0) * tz;
        let d = normalize(camera.forward.xyz + camera.right.xyz * sx + camera.up.xyz * sy);
        let den = dot(d, g9_bg_n);
        if (abs(den) > 1e-6) {
            let t = g9_dbg * dot(dc, g9_bg_n) / den;
            if (t > 0.0) { return t; }
        }
    }
    return g9_bg_corners(b);
}

// The background's distance at bin `b`: its corners' bilinearly, else the
// pixel's centre's.
fn g9_bg_corners(b: u32) -> f32 {
    let f = (vec2<f32>(f32(b % 4u), f32(b / 4u)) + vec2<f32>(0.5)) * 0.25;
    let wk = vec4<f32>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y) * g9_cw;
    let total = wk.x + wk.y + wk.z + wk.w;
    if (total < 1e-3) { return g9_dbg; }
    return dot(wk, g9_cd) / total;
}
var<private> g9_e: vec3<f32> = vec3<f32>(0.0);

// The pixel's bins in which something `near` away lies in front of what
// the bin shows behind small entries (its large object or background).
fn g9_in_front(near: f32) -> u32 {
    var bins = 0xffffu;
    if (near > g9_dl) { bins &= ~g9_l; }
    if (near > g9_d2) { bins &= ~g9_l2; }
    if (near > g9_dbg) { bins &= g9_l | g9_l2; }
    return bins;
}

// The largest colour a page texel holds (albedo 1 lit fully), for its
// bytes.
fn g9_scale() -> f32 {
    let e = g9_sun(camera.eye.xyz).e * (1.0 + G9_AMBIENT) / 3.14159265;
    return max(max(e.x, e.y), max(e.z, 1e-6));
}

// The ambient's irradiance as a share of the light's normal irradiance.
const G9_AMBIENT: f32 = 0.2;
// Development: the model's shadows (off: every point sees the light).
const G9_SHADOWS: bool = true;
// Large bins' colours from their object's corner colours, bilinearly.
const G9_CORNERS: bool = true;
// Development: the sky shown red in our image (where bins fall through).
const G9_RED_SKY: bool = false;
// Development: map reads skip instances' parts behind the background.
const G9_BURIED: bool = true;
// Development: background objects not large for corner rays.
const G9_BG_NOT_LARGE: bool = true;
// Development: the buried test skips every bin it reaches (a probe).
const G9_PROBE: bool = false;
// Development: bins' background distance by the background hit's plane.
const G9_PLANE: bool = true;
// Development: small instances a constant half-lit colour (a test).
const G9_CONST_SMALL: bool = false;
// Development: ambient light alone (colour then measures coverage).
const G9_FLAT: bool = false;

// The light (the first in `lights`) seen from a point: the direction toward
// its rectangle's centre, its distance, and its normal irradiance (radiance
// times the rectangle's solid angle; none behind a one-sided light).
struct G9Sun {
    l: vec3<f32>,
    e: vec3<f32>,
    dist: f32,
};

fn g9_sun(p: vec3<f32>) -> G9Sun {
    let r = lights[0];
    let c = r.corner.xyz + 0.5 * (r.edge_u.xyz + r.edge_v.xyz);
    let to = c - p;
    let d = length(to);
    let l = to / d;
    let nr = cross(r.edge_u.xyz, r.edge_v.xyz);
    let area = length(nr);
    var cosl = dot(nr / area, -l);
    if (r.corner.w < 0.5) { cosl = abs(cosl); }
    return G9Sun(l, r.radiance.xyz * (area * max(cosl, 0.0) / (d * d)), d);
}

// The model: albedo × (E_sun · max(0, n·l) · v + E_amb) / π.
fn g9_colour(albedo: vec3<f32>, n: vec3<f32>, sun: G9Sun, v: f32) -> vec3<f32> {
    if (G9_FLAT) { return albedo * sun.e * G9_AMBIENT / 3.14159265; }
    return albedo * (sun.e * (max(dot(n, sun.l), 0.0) * v + G9_AMBIENT)) / 3.14159265;
}

// What rays that miss see: the ambient's radiance.
fn g9_sky() -> vec3<f32> {
    return G9_AMBIENT * g9_sun(camera.eye.xyz).e / 3.14159265;
}

// The model at a camera ray's hit (object `m`, global triangle `t`,
// distance `dist` along `dir` from `e`): shading normal, albedo, and a
// hard shadow toward the light's centre.
fn g9_shade_hit(m: u32, t: u32, dist: f32, e: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let p = e + dir * dist;
    let v0 = tris[3u * t].xyz;
    var ng = normalize(normal_to_world(m, cross(tris[3u * t + 1u].xyz - v0, tris[3u * t + 2u].xyz - v0)));
    if (dot(ng, dir) > 0.0) { ng = -ng; }
    let n = shading_normal(m, t, p, -dir);
    let albedo = surface_albedo(m, t, p, dist, dir, ng);
    let sun = g9_sun(p);
    var v = 1.0;
    if (G9_SHADOWS) { v = select(0.0, 1.0, hw_segment_open(p + ng * surface_offset(dist), sun.l, sun.dist)); }
    return g9_colour(albedo, n, sun, v);
}

// M13 gate 9's reference (profile stage 19): one ray through the pixel at
// the camera's sample offset, its front hit shaded by the model (the sky
// where it misses).
@compute @workgroup_size(8, 8)
fn hw_m13_reference(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(camera.params.x);
    let h = u32(camera.params.y);
    if (gid.x >= w || gid.y >= h) { return; }
    let dir = camera_dir(gid);
    let e = camera.eye.xyz;
    let hit = hw_trace(e, dir, 0.0);
    var c = g9_sky();
    if (hit.x != NONE) { c = g9_shade_hit(hit.x, hit.y, bitcast<f32>(hit.z), e, dir); }
    colors[gid.y * w + gid.x] = vec4<f32>(c, 1.0);
}
