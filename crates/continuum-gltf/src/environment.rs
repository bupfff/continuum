//! Environment images: an equirectangular sky as an [`Environment`], from a
//! Radiance HDR file (`.hdr`, the usual format for HDRIs) or a PNG.

use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use continuum_gpu::Environment;

/// Loads an equirectangular environment image, recognized by its contents:
///
/// - a Radiance HDR file (RGBE, flat or run-length encoded, in the standard
///   `-Y height +X width` orientation), as the linear radiances it stores.
///   Like Blender, the header's `EXPOSURE` is ignored;
/// - a PNG (8 or 16 bits, grey or RGB, alpha ignored), decoded from sRGB to
///   linear, so its brightest white is radiance 1.
///
/// Its orientation is Blender's ([`Environment`]): the top row is the zenith
/// and column `u` looks toward azimuth `π − 2π u` (z up), so an HDRI looks the
/// same here as in Blender's world shader.
pub fn load_environment(path: &Path) -> Result<Environment> {
    decode_environment(&std::fs::read(path)?)
}

/// [`load_environment`] from the file's bytes.
pub fn decode_environment(bytes: &[u8]) -> Result<Environment> {
    if bytes.starts_with(b"#?") {
        decode_hdr(bytes)
    } else if bytes.starts_with(b"\x89PNG") {
        decode_png(bytes)
    } else {
        Err(invalid("not a Radiance HDR or PNG image"))
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

fn decode_hdr(bytes: &[u8]) -> Result<Environment> {
    // The header: lines up to an empty one, then the resolution line.
    let mut pos = 0;
    let mut line = || -> Result<&[u8]> {
        let start = pos;
        let len = bytes[start..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| invalid("truncated HDR header"))?;
        pos = start + len + 1;
        Ok(&bytes[start..start + len])
    };
    loop {
        let l = line()?;
        if l.is_empty() {
            break;
        }
        if let Some(format) = l.strip_prefix(b"FORMAT=")
            && format != b"32-bit_rle_rgbe"
        {
            return Err(invalid("HDR format is not 32-bit_rle_rgbe"));
        }
    }
    let resolution = String::from_utf8_lossy(line()?).into_owned();
    let fields: Vec<&str> = resolution.split_whitespace().collect();
    let (height, width) = match fields.as_slice() {
        ["-Y", h, "+X", w] => (
            h.parse::<u32>().map_err(|_| invalid("bad HDR height"))?,
            w.parse::<u32>().map_err(|_| invalid("bad HDR width"))?,
        ),
        _ => return Err(invalid("HDR orientation is not -Y height +X width")),
    };
    let (w, h) = (width as usize, height as usize);
    let mut data = &bytes[pos..];
    let mut rgbe = vec![[0u8; 4]; w * h];
    let mut take = |n: usize| -> Result<&[u8]> {
        if data.len() < n {
            return Err(invalid("truncated HDR pixels"));
        }
        let (head, rest) = data.split_at(n);
        data = rest;
        Ok(head)
    };
    for row in rgbe.chunks_mut(w.max(1)).take(h) {
        let start = take(4.min(w * 4))?;
        let rle = (8..32768).contains(&w)
            && start[0] == 2
            && start[1] == 2
            && ((start[2] as usize) << 8 | start[3] as usize) == w;
        if rle {
            // New run-length encoding: each channel in turn, as runs
            // (count > 128) and literals.
            for c in 0..4 {
                let mut x = 0;
                while x < w {
                    let count = take(1)?[0] as usize;
                    if count > 128 {
                        let n = count - 128;
                        let v = take(1)?[0];
                        if x + n > w {
                            return Err(invalid("HDR run overflows its scanline"));
                        }
                        row[x..x + n].iter_mut().for_each(|p| p[c] = v);
                        x += n;
                    } else {
                        if count == 0 || x + count > w {
                            return Err(invalid("bad HDR literal"));
                        }
                        for (p, &v) in row[x..x + count].iter_mut().zip(take(count)?) {
                            p[c] = v;
                        }
                        x += count;
                    }
                }
            }
        } else {
            // Flat pixels, with the old encoding's repeats ((1, 1, 1, n):
            // the previous pixel n times, shifted by 8 bits per repeat in a
            // row).
            let mut first = Some(start);
            let (mut x, mut shift) = (0usize, 0u32);
            while x < w {
                let p: [u8; 4] = match first.take() {
                    Some(s) if s.len() == 4 => [s[0], s[1], s[2], s[3]],
                    _ => take(4)?.try_into().unwrap(),
                };
                if p[0] == 1 && p[1] == 1 && p[2] == 1 {
                    if x == 0 {
                        return Err(invalid("HDR repeat with no pixel before it"));
                    }
                    let n = (p[3] as usize) << shift;
                    if x + n > w {
                        return Err(invalid("HDR repeat overflows its scanline"));
                    }
                    let prev = row[x - 1];
                    row[x..x + n].fill(prev);
                    x += n;
                    shift += 8;
                } else {
                    row[x] = p;
                    x += 1;
                    shift = 0;
                }
            }
        }
    }
    // As Blender and Radiance decode RGBE: the mantissa's centre.
    let rgb = rgbe
        .iter()
        .map(|p| {
            if p[3] == 0 {
                [0.0; 3]
            } else {
                let f = 2f32.powi(p[3] as i32 - 136);
                [0, 1, 2].map(|c| (p[c] as f32 + 0.5) * f)
            }
        })
        .collect();
    Ok(Environment { width, height, rgb })
}

fn decode_png(bytes: &[u8]) -> Result<Environment> {
    let mut decoder = png::Decoder::new(bytes);
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(Error::other)?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(Error::other)?;
    let buf = &buf[..info.buffer_size()];
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(invalid("PNG palette was not expanded")),
    };
    let values: Vec<f32> = match info.bit_depth {
        png::BitDepth::Sixteen => buf
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_be_bytes(*b) as f32 / 65535.0)
            .collect(),
        _ => buf.iter().map(|&b| b as f32 / 255.0).collect(),
    };
    let linear = |s: f32| {
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let rgb = values
        .chunks_exact(channels)
        .map(|p| {
            if channels < 3 {
                [linear(p[0]); 3]
            } else {
                [linear(p[0]), linear(p[1]), linear(p[2])]
            }
        })
        .collect();
    Ok(Environment {
        width: info.width,
        height: info.height,
        rgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(w: u32, h: u32) -> Vec<u8> {
        format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nEXPOSURE=2\n\n-Y {h} +X {w}\n").into_bytes()
    }

    fn rgbe(p: [u8; 4]) -> [f32; 3] {
        let f = 2f32.powi(p[3] as i32 - 136);
        [0, 1, 2].map(|c| (p[c] as f32 + 0.5) * f)
    }

    #[test]
    fn flat_hdr_with_old_repeats() {
        // 3×2: a row of three pixels, then a pixel repeated twice.
        let (a, b, c, d) = (
            [128, 64, 32, 129],
            [200, 100, 50, 140],
            [0, 0, 0, 0],
            [255, 1, 9, 100],
        );
        let mut bytes = header(3, 2);
        for p in [a, b, c, d, [1, 1, 1, 2]] {
            bytes.extend(p);
        }
        let e = decode_environment(&bytes).unwrap();
        assert_eq!((e.width, e.height), (3, 2));
        assert_eq!(e.rgb[0], rgbe(a));
        assert_eq!(e.rgb[1], rgbe(b));
        assert_eq!(e.rgb[2], [0.0; 3]);
        assert_eq!(&e.rgb[3..], &[rgbe(d); 3]);
        // 128 · 2^(129 − 136) + half a step: 1.0 and a little.
        assert!((e.rgb[0][0] - 1.0039).abs() < 1e-3);
    }

    #[test]
    fn run_length_encoded_hdr() {
        // One 10-pixel scanline: red a run of 10, green 10 literals, blue a
        // run of 4 then 6 literals, exponent a run of 10.
        let mut bytes = header(10, 1);
        bytes.extend([2, 2, 0, 10]);
        bytes.extend([128 + 10, 7]);
        bytes.push(10);
        bytes.extend(0..10u8);
        bytes.extend([128 + 4, 50, 6]);
        bytes.extend(20..26u8);
        bytes.extend([128 + 10, 130]);
        let e = decode_environment(&bytes).unwrap();
        assert_eq!((e.width, e.height), (10, 1));
        assert_eq!(e.rgb.len(), 10);
        for (x, &pixel) in e.rgb.iter().enumerate() {
            let blue = if x < 4 { 50 } else { 20 + x as u8 - 4 };
            assert_eq!(pixel, rgbe([7, x as u8, blue, 130]), "pixel {x}");
        }
    }

    #[test]
    fn rejects_bad_files() {
        let mut bytes = header(10, 1);
        bytes.extend([2, 2, 0, 10, 128 + 11, 7]);
        assert!(decode_environment(&bytes).is_err());
        let mut rotated = b"#?RADIANCE\n\n+Y 1 +X 1\n".to_vec();
        rotated.extend([1, 2, 3, 4]);
        assert!(decode_environment(&rotated).is_err());
        assert!(decode_environment(b"GIF89a").is_err());
    }

    #[test]
    fn png_is_decoded_from_srgb() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 3, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&[255, 0, 188, 0, 0, 0, 10, 10, 10])
                .unwrap();
        }
        let e = decode_environment(&bytes).unwrap();
        assert_eq!((e.width, e.height), (3, 1));
        assert_eq!(e.rgb[0][0], 1.0);
        assert_eq!(e.rgb[0][1], 0.0);
        assert!((e.rgb[0][2] - 0.5029).abs() < 1e-3);
        assert_eq!(e.rgb[1], [0.0; 3]);
        assert!((e.rgb[2][0] - 10.0 / 255.0 / 12.92).abs() < 1e-6);
    }
}
