//! M10 gate 24: the cache settles with bounce light (M10 gate 23 re-run).
//!
//! M10 gate 23 (emissive faces as lights) never reached its accuracy
//! criterion: with bounce light on, the view never settled at a sub-pixel
//! offset (50,000 frames). Diagnosis after it (`emitters_dev` with
//! `CONTINUUM_TRACE_BAKES`): a page's bounce samples complete only when
//! every one of their hits finds a resident page at once; hit pages off
//! screen, read by no pixel, aged out after 64 frames before the rest had
//! arrived (the bake budget at its floor bakes few a frame), so the
//! waiting page was retried forever (one page 552 times in 3,000 frames).
//! A liveness bug in the certified cache, not specific to emissive faces.
//!
//! Fix (made while diagnosing, before this stub): pages baked for bounce
//! hits (secondary requests) stay resident while any page's bounce samples
//! wait, at most a quarter of the pool; once none wait they age out as
//! before. In development the same view settled about 1,000 frames after
//! the offset and stayed quiet.
//!
//! Scene, renderers, Cycles reference and accuracy criterion: gate 23's
//! (M10 gate 12's scene, the textured scene at night with a glowing panel
//! whose six faces become lights, against emission through bounce light;
//! Cycles at 4,096 samples per pixel, one bounce, gate 11's cached render
//! of the same job).
//!
//! Pass criterion, fixed before the run:
//! 1. **Liveness:** with bounce light on, both renderers (faces as lights,
//!    and emission through bounce) settle (32 frames asking for and baking
//!    nothing) at the first view and then at each of the 16 sub-pixel
//!    offsets of the 4×4 supersampling (textures read for a quarter of a
//!    pixel), each within 20,000 frames.
//! 2. **Accuracy over surfaces** (gate 23's criterion 2): the settled cached
//!    image with the faces as lights, supersampled 4×4, within 5% relative
//!    RMSE of Cycles, and at most 0.5× the error of emission through
//!    bounce, over the pixels not set apart (gate 12's rule).
//!
//! Reported: frames to settle per offset; the pages resident when settled.
//!
//! `cargo run --release -p room3d --bin m10_gate24`
//!
//! `M10_DEV=1` runs a development check (the first view and two offsets,
//! no Cycles), whose numbers are not the gate's.

fn main() {
    unimplemented!("M10 gate 24: stub committed with its criteria before the run");
}
