//! M13 gate 9: colour in the coverage pass (appearance, first).
//!
//! M13 gate 8 brought the coverage pass to about 9 ms settled
//! (`docs/results/m13-gate8.md`), coverage only and without the field's
//! ground. The design (DESIGN §3.4, "Appearance") adds colour with one
//! shading model evaluated everywhere, the reference included, and the
//! ground, which makes large objects a background with a depth.
//!
//! The model: colour = albedo × (E_sun · max(0, n·l) · V + E_amb) / π, l
//! and E_sun from the light's rectangle (toward its centre; its radiance
//! times its solid angle), V a hard shadow (one hardware ray toward the
//! light's centre), E_amb a fifth of the sun's normal irradiance, shading
//! normals; rays that miss see a constant sky.
//!
//! Method, on gate 8's pass:
//! - Large objects (mapless or over 8 pixels) as a background: a pixel with
//!   large bins shades one large hit by the model and keeps its depth; walks
//!   cull nodes behind it; small entries in front composite over its bins;
//!   what is left of each bin shows the large colour or the sky.
//! - Per entry and bin the pixel's colour accumulates the bin's
//!   transmittance times the entry's coverage and colour, over 16.
//! - Instances: maps hold coverage and the texels' mean normal; records add
//!   the light's direction in the object's frame, the instance's sun
//!   visibility (one ray a frame) and its albedo.
//! - Pages: texels add the mean colour of their rays' hits, shaded by the
//!   model at bake (a shadow ray a hit).
//!
//! Scene: the million-boulder field with its ground, its light, 1280×720;
//! views standing at 1.7 m and 150 m up looking 45° down (gates 4–8's).
//! Reference: per pixel the mean of 16×16 samples (offsets on a regular
//! grid), each one ray shaded by the model. One ray: the centre sample.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Colour:** relative RMSE of the RGB image from the reference at most
//!    0.5× one ray's, standing and from 150 m.
//! 2. **Speed:** settled (every wanted page baked), at most 15 ms a frame
//!    in both views (median of 10 frames each waited for), timed first.
//!
//! Reported: the 20 m view (gate 7's flight view) settled; pixels with large
//! bins; mean colour against the reference's.
//!
//! `cargo run --release -p room3d --bin m13_gate9`
//!
//! `M13_DEV=1` runs a development check (10,000 boulders; `M13_DEV=1M` at
//! full size), whose numbers are not the gate's.

fn main() {
    unimplemented!("M13 gate 9: written before the implementation");
}
