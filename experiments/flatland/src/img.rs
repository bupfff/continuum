//! Minimal RGB images: tone mapping, error maps, scene overlays, PNG output.

use std::io;
use std::path::Path;

use crate::geom::Segment;
use crate::scene::Scene;

pub type Rgb = [u8; 3];

pub struct Image {
    pub w: usize,
    pub h: usize,
    data: Vec<Rgb>,
}

impl Image {
    pub fn new(w: usize, h: usize, fill: Rgb) -> Self {
        Image {
            w,
            h,
            data: vec![fill; w * h],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, c: Rgb) {
        if x < self.w && y < self.h {
            self.data[y * self.w + x] = c;
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Rgb {
        self.data[y * self.w + x]
    }

    pub fn blit(&mut self, src: &Image, ox: usize, oy: usize) {
        for y in 0..src.h {
            for x in 0..src.w {
                self.set(ox + x, oy + y, src.data[y * src.w + x]);
            }
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut encoder = png::Encoder::new(io::BufWriter::new(file), self.w as u32, self.h as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(io::Error::other)?;
        let bytes: Vec<u8> = self.data.iter().flatten().copied().collect();
        writer.write_image_data(&bytes).map_err(io::Error::other)
    }
}

/// Reinhard tone curve plus display gamma.
pub fn tonemap(fluence: &[f32], size: usize) -> Image {
    let mut img = Image::new(size, size, [0; 3]);
    for (i, &e) in fluence.iter().enumerate() {
        let v = (e / (e + 0.35)).powf(1.0 / 2.2);
        let c = (v * 255.0).round() as u8;
        img.set(i % size, i / size, [c, c, c]);
    }
    img
}

/// Red where the estimate is too dark, blue where it is too bright (light leaking).
/// Full intensity at an absolute fluence error of `scale`.
pub fn signed_error(estimate: &[f32], reference: &[f32], size: usize, scale: f32) -> Image {
    let mut img = Image::new(size, size, [0; 3]);
    for (i, (&e, &r)) in estimate.iter().zip(reference).enumerate() {
        let d = e - r;
        let c = (d.abs() / scale).min(1.0).sqrt() * 255.0;
        let rgb = if d < 0.0 {
            [c, c * 0.25, c * 0.15]
        } else {
            [c * 0.2, c * 0.5, c]
        };
        img.set(i % size, i / size, rgb.map(|v| v as u8));
    }
    img
}

pub const OCCLUDER_COLOR: Rgb = [120, 220, 140];
pub const LIGHT_COLOR: Rgb = [255, 220, 90];

/// Black at 0, rising through orange to pale yellow at `scale` and above.
pub fn magnitude(values: &[f32], size: usize, scale: f32) -> Image {
    let mut img = Image::new(size, size, [0; 3]);
    for (i, &v) in values.iter().enumerate() {
        let m = (v / scale).clamp(0.0, 1.0).sqrt();
        let rgb = [255.0 * m, 200.0 * m * m, 120.0 * m.powi(4)];
        img.set(i % size, i / size, rgb.map(|c| c as u8));
    }
    img
}

/// Draws segments over an image of the unit square (y up).
pub fn draw_segments(img: &mut Image, segments: &[Segment], color: Rgb) {
    let size = img.w as f32;
    for s in segments {
        let steps = ((s.b - s.a).length() * size * 2.0).ceil().max(1.0) as usize;
        for k in 0..=steps {
            let q = s.a + (s.b - s.a) * (k as f32 / steps as f32);
            let (x, y) = (q.x * size, (1.0 - q.y) * size);
            if x >= 0.0 && y >= 0.0 {
                img.set(x as usize, y as usize, color);
            }
        }
    }
}

/// Draws occluders (green) and the light (yellow) over an image of the unit square.
pub fn draw_scene(img: &mut Image, scene: &Scene) {
    draw_segments(img, &scene.occluders, OCCLUDER_COLOR);
    draw_segments(img, &[scene.light], LIGHT_COLOR);
}
