//! M17 gate 1: the display's tone map and auto-exposure.
//!
//! Method (DESIGN §3.4, "Display"): the display pass maps the renderer's
//! linear light with Khronos PBR Neutral (`ToneMap::PbrNeutral`, the
//! published formula; `Renderer`'s default) to an sRGB target; the
//! auto-exposure meter (`AutoExposure`) histograms the frame's log2
//! luminance in half stops and sets the exposure that shows the mean
//! between its 50th and 95th percentiles at 0.18, approached with a time
//! constant (0.5 s), only that number carried between frames.
//!
//! Views: M6's textured scene by day (sun and sky) and at night with its
//! five lamps (`with_lamps`), 1280×720, cache settled. For the tone map,
//! each view's exposure is fixed at the meter's rule computed on the CPU
//! from its linear image; Blender 5.2 applies its own "Khronos PBR
//! Neutral" view (and AgX and Standard, reported) to the same linear
//! image times that exposure (`cycles/view_transform.py`), against our
//! display's output read from an `Rgba8UnormSrgb` target.
//!
//! Pass criterion, fixed before the first run:
//! 1. **Cost** (first): on the day view, the warm frame with
//!    auto-exposure at most 0.15 ms more than without (medians of five
//!    interleaved rounds of 30 frames, each waited for).
//! 2. **Tone map:** in both views, our PBR Neutral display values within
//!    1/255 of Blender's on average and 3/255 at the 99th percentile (in
//!    sRGB-encoded units, every channel of every pixel; ours are 8-bit).
//! 3. **Adaptation:** on the day view settled with auto-exposure, the
//!    sun's radiance drops to a sixteenth; the meter's exposure each
//!    frame, against its value 4 s later, never passes it by more than
//!    0.5% (no overshoot) and is within 2% of it 2.5 s after the drop
//!    (five time constants).
//!
//! Reported: our AgX (an approximation of the original) against Blender's
//! AgX; the maximum differences; the exposures.
//!
//! Development mode (`M17_DEV=1`): 640×360.

fn main() {
    todo!("M17 gate 1: stub committed before the run")
}
