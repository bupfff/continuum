//! Times building the kernel on the adapter wgpu picks (see
//! `MaskConeKernel::new` for the environment variables that choose it).
//! With `CONTINUUM_PIPELINE_TIMES=1`, each pipeline's build time too.
//!
//! `cargo run --release -p continuum-gpu --example kernel_setup`

use std::time::Instant;

fn main() {
    let start = Instant::now();
    let kernel = continuum_gpu::MaskConeKernel::new().expect("a GPU adapter");
    // Pipelines are otherwise built on first use.
    kernel.build_all_pipelines();
    println!(
        "{}: kernel built in {:.2} s; hardware rays: {}",
        kernel.adapter_name,
        start.elapsed().as_secs_f64(),
        kernel.has_hardware_rays()
    );
}
