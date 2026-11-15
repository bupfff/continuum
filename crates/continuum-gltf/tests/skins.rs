//! Skins and morph targets (M12 gate 3), on a small file made in Blender
//! by `experiments/room3d/cycles/make_skinned_test.py`: a tube skinned to
//! three bones with a shape key, animated (a rig action and a shape key
//! action). M12 gate 3 checked its poses against Blender's own evaluation
//! (within 0.0004 mm); this keeps the import's shape.

use std::path::Path;

use continuum::glam::Vec3;

fn tube() -> continuum_gltf::Imported {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/skinned_tube.glb");
    continuum_gltf::load(&path).expect("the skinned tube")
}

#[test]
fn skinned_tube_imports_as_deforming_geometry() {
    let imported = tube();
    let deforming = imported.deforming();
    assert_eq!(deforming.len(), 1, "one deforming geometry");
    assert_eq!(imported.animations.list.len(), 2);
    assert_eq!(imported.stats.animation_channels_skipped, 0);
    let g = deforming[0];
    let rest = &imported.scene.geometries[g];
    let all: Vec<usize> = (0..imported.animations.list.len()).collect();
    // The rest pose (before the first keyframe) is the geometry's.
    let pose = imported
        .animations
        .deform(&all, 0.0)
        .into_iter()
        .find(|p| p.geometry == g)
        .expect("a pose");
    assert_eq!(pose.triangles.len(), rest.triangles.len());
    let largest = |a: &[[Vec3; 3]], b: &[[Vec3; 3]]| {
        a.iter()
            .zip(b)
            .flat_map(|(s, t)| (0..3).map(move |k| (s[k] - t[k]).length()))
            .fold(0.0f32, f32::max)
    };
    assert!(
        largest(&pose.triangles, &rest.triangles) < 1e-4,
        "the rest pose moved"
    );
    // At 1 s the middle bone is bent 60° and the shape key full: the top
    // moves by metres' fractions, the tube stays 2 m long (its corners
    // within 2.3 m of its foot).
    let bent = imported
        .animations
        .deform(&all, 1.0)
        .into_iter()
        .find(|p| p.geometry == g)
        .expect("a pose");
    let moved = largest(&bent.triangles, &rest.triangles);
    assert!(moved > 0.3, "the pose barely moved: {moved}");
    let reach = bent
        .triangles
        .iter()
        .flatten()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    assert!(reach < 2.3, "the tube stretched: {reach}");
}
