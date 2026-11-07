//! The kernel's shaders translate for every backend wgpu drives (SPIR-V for
//! Vulkan, HLSL for DX12, MSL for Metal), entry point by entry point, as
//! wgpu translates them. This runs on the CPU, so a shader that one backend
//! cannot take fails here on any machine, not only on that backend's GPU.
//! (Naga's SPIR-V output once failed on a pointer to a struct's member that
//! Metal took.)

use naga::back::PipelineConstants;
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn constants(packets: bool) -> PipelineConstants {
    let mut c = PipelineConstants::default();
    c.insert("WORKGROUP_STACKS".to_string(), 0.0);
    c.insert("COUNT_WORK".to_string(), 0.0);
    c.insert("PACKETS".to_string(), if packets { 1.0 } else { 0.0 });
    c
}

/// Translates every entry point of `source` (or only `only`); returns the
/// failures.
fn translate(label: &str, source: &str, packets: bool, only: Option<&str>) -> Vec<String> {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
    let mut failures = Vec::new();
    for ep in &module.entry_points {
        if only.is_some_and(|o| o != ep.name) {
            continue;
        }
        let name = ep.name.clone();
        let stage = ep.stage;
        let result = std::panic::catch_unwind(|| -> Result<(), String> {
            let (m, i) = naga::back::pipeline_constants::process_overrides(
                &module,
                &info,
                Some((stage, &name)),
                &constants(packets),
            )
            .map_err(|e| format!("overrides: {e}"))?;
            naga::back::spv::write_vec(
                &m,
                &i,
                &naga::back::spv::Options::default(),
                Some(&naga::back::spv::PipelineOptions {
                    shader_stage: stage,
                    entry_point: name.clone(),
                }),
            )
            .map_err(|e| format!("SPIR-V: {e}"))?;
            let hlsl_options = naga::back::hlsl::Options {
                shader_model: naga::back::hlsl::ShaderModel::V6_0,
                fake_missing_bindings: true,
                ..Default::default()
            };
            let pipeline = naga::back::hlsl::PipelineOptions {
                entry_point: Some((stage, name.clone())),
            };
            let mut hlsl = String::new();
            naga::back::hlsl::Writer::new(&mut hlsl, &hlsl_options, &pipeline)
                .write(&m, &i, None)
                .map_err(|e| format!("HLSL: {e}"))?;
            naga::back::msl::write_string(
                &m,
                &i,
                // Ray queries need MSL 2.4 (Apple GPUs with them have it).
                &naga::back::msl::Options {
                    lang_version: (2, 4),
                    ..Default::default()
                },
                &naga::back::msl::PipelineOptions {
                    entry_point: Some((stage, name.clone())),
                    ..Default::default()
                },
            )
            .map_err(|e| format!("MSL: {e}"))?;
            Ok(())
        });
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => failures.push(format!("{label} {name}: {e}")),
            Err(_) => failures.push(format!("{label} {name}: the translator panicked")),
        }
    }
    failures
}

#[test]
fn kernel_translates_for_every_backend() {
    let mut failures = Vec::new();
    for packets in [false, true] {
        let source = continuum_gpu::kernel_source(packets, false);
        failures.extend(translate(
            &format!("kernel (packets {packets})"),
            &source,
            packets,
            None,
        ));
        let hw = format!(
            "enable wgpu_ray_query;\n{source}\n{}",
            include_str!("../src/hw_visibility.wgsl")
        );
        failures.extend(translate(
            &format!("hardware rays (packets {packets})"),
            &hw,
            packets,
            Some("hw_visibility"),
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
