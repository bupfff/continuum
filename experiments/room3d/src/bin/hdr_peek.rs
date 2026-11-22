//! Development: an HDR image's brightest regions and its light by quarter
//! of the sphere, and a tone-mapped PNG of it (`HP_OUT`). Not a gate.
fn main() {
    let path = std::env::args().nth(1).expect("an .hdr");
    let env = continuum_gltf::load_environment(std::path::Path::new(&path)).expect("the image");
    let (w, h) = (env.width, env.height);
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let cell = std::f32::consts::TAU / w as f32 * std::f32::consts::PI / h as f32;
    let mut total = 0.0;
    let mut upper = 0.0;
    let mut values: Vec<(f32, u32)> = Vec::new();
    for y in 0..h {
        let e = (0.5 - (y as f32 + 0.5) / h as f32) * std::f32::consts::PI;
        for x in 0..w {
            let l = lum(env.rgb[(y * w + x) as usize]);
            let dw = cell * e.cos();
            total += l * dw;
            if e > 0.0 {
                upper += l * dw;
            }
            values.push((l, y * w + x));
        }
    }
    values.sort_by(|a, b| b.0.total_cmp(&a.0));
    let median = values[values.len() / 2].0;
    println!(
        "{w}x{h}: peak {:.1}, median {:.3}, light above the horizon {:.0}% of {:.2}",
        values[0].0,
        median,
        100.0 * upper / total,
        total
    );
    // The share of light in the brightest 1%, 5%, 10% of pixels (by count).
    for share in [0.01, 0.05, 0.1] {
        let n = (values.len() as f32 * share) as usize;
        let e: f32 = values[..n]
            .iter()
            .map(|&(l, i)| {
                let y = i / w;
                let el = (0.5 - (y as f32 + 0.5) / h as f32) * std::f32::consts::PI;
                l * cell * el.cos()
            })
            .sum();
        println!(
            "brightest {:.0}% of pixels hold {:.0}% of the light",
            share * 100.0,
            100.0 * e / total
        );
    }
    if let Some(out) = std::env::var_os("HP_OUT") {
        let k = 1.0 / (4.0 * median.max(1e-6));
        let bytes: Vec<u8> = env
            .rgb
            .iter()
            .flat_map(|c| {
                c.map(|v| {
                    let v = v * k / (1.0 + v * k);
                    (v.powf(1.0 / 2.2) * 255.0) as u8
                })
            })
            .collect();
        let file = std::fs::File::create(out).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .unwrap()
            .write_image_data(&bytes)
            .unwrap();
    }
}
