use crate::image::ImageBuf;
use crate::precision::from_f64;
use rayon::prelude::*;

/// Trilinear 3D LUT interpolation.
///
/// `data`: flat RGB data, size^3 * 3 floats, indexed as data[(r*size*size + g*size + b)*3 + c].
/// `size`: side length of the cube.
/// `r`, `g`, `b`: input values in [0, 1].
///
/// Returns interpolated [R, G, B].
#[inline]
pub fn trilinear_3d(data: &[f32], size: usize, r: f32, g: f32, b: f32) -> [f32; 3] {
    let max_idx = (size - 1) as f32;

    let rf = (r.clamp(0.0, 1.0) * max_idx).min(max_idx);
    let gf = (g.clamp(0.0, 1.0) * max_idx).min(max_idx);
    let bf = (b.clamp(0.0, 1.0) * max_idx).min(max_idx);

    let r0 = rf as usize;
    let g0 = gf as usize;
    let b0 = bf as usize;

    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let fr = rf - r0 as f32;
    let fg = gf - g0 as f32;
    let fb = bf - b0 as f32;

    let idx = |r: usize, g: usize, b: usize, c: usize| -> usize {
        (r * size * size + g * size + b) * 3 + c
    };

    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let c000 = data[idx(r0, g0, b0, c)];
        let c100 = data[idx(r1, g0, b0, c)];
        let c010 = data[idx(r0, g1, b0, c)];
        let c110 = data[idx(r1, g1, b0, c)];
        let c001 = data[idx(r0, g0, b1, c)];
        let c101 = data[idx(r1, g0, b1, c)];
        let c011 = data[idx(r0, g1, b1, c)];
        let c111 = data[idx(r1, g1, b1, c)];

        let c00 = c000 + fr * (c100 - c000);
        let c10 = c010 + fr * (c110 - c010);
        let c01 = c001 + fr * (c101 - c001);
        let c11 = c011 + fr * (c111 - c011);

        let c0 = c00 + fg * (c10 - c00);
        let c1 = c01 + fg * (c11 - c01);

        out[c] = c0 + fb * (c1 - c0);
    }
    out
}

/// Tetrahedral 3D LUT interpolation on f64 data.
///
/// `r`, `g`, and `b` are LUT-grid coordinates in `[0, size - 1]`. The
/// tetrahedron selection and interpolation order match Python's
/// `apply_lut_tetrahedral_3d`.
#[inline]
pub fn tetrahedral_3d_f64(data: &[f64], size: usize, r: f64, g: f64, b: f64) -> [f64; 3] {
    assert!(size >= 2, "tetrahedral LUT size must be at least 2");
    assert_eq!(data.len(), size * size * size * 3);

    let coordinate = |value: f64| {
        let clamped = value.clamp(0.0, (size - 1) as f64);
        let base = if clamped >= (size - 1) as f64 {
            size - 2
        } else {
            clamped.floor() as usize
        };
        (base, clamped - base as f64)
    };
    let (i, fr) = coordinate(r);
    let (j, fg) = coordinate(g);
    let (k, fb) = coordinate(b);
    let index = |ri: usize, gi: usize, bi: usize, channel: usize| {
        data[((ri * size + gi) * size + bi) * 3 + channel]
    };

    let mut out = [0.0; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let c000 = index(i, j, k, channel);
        let c111 = index(i + 1, j + 1, k + 1, channel);
        *value = if fr > fg {
            if fg > fb {
                let c100 = index(i + 1, j, k, channel);
                let c110 = index(i + 1, j + 1, k, channel);
                c000 + fr * (c100 - c000) + fg * (c110 - c100) + fb * (c111 - c110)
            } else if fr > fb {
                let c100 = index(i + 1, j, k, channel);
                let c101 = index(i + 1, j, k + 1, channel);
                c000 + fr * (c100 - c000) + fb * (c101 - c100) + fg * (c111 - c101)
            } else {
                let c001 = index(i, j, k + 1, channel);
                let c101 = index(i + 1, j, k + 1, channel);
                c000 + fb * (c001 - c000) + fr * (c101 - c001) + fg * (c111 - c101)
            }
        } else if fb > fg {
            let c001 = index(i, j, k + 1, channel);
            let c011 = index(i, j + 1, k + 1, channel);
            c000 + fb * (c001 - c000) + fg * (c011 - c001) + fr * (c111 - c011)
        } else if fb > fr {
            let c010 = index(i, j + 1, k, channel);
            let c011 = index(i, j + 1, k + 1, channel);
            c000 + fg * (c010 - c000) + fb * (c011 - c010) + fr * (c111 - c011)
        } else {
            let c010 = index(i, j + 1, k, channel);
            let c110 = index(i + 1, j + 1, k, channel);
            c000 + fg * (c010 - c000) + fr * (c110 - c010) + fb * (c111 - c110)
        };
    }
    out
}

/// Apply an f64 tetrahedral LUT to an image whose channel values use
/// independent input bounds.
pub fn apply_tetrahedral_3d_f64(
    image: &ImageBuf,
    data: &[f64],
    size: usize,
    data_min: [f64; 3],
    data_max: [f64; 3],
) -> ImageBuf {
    let scale = (size - 1) as f64;
    let inverse_range = [
        scale / (data_max[0] - data_min[0]),
        scale / (data_max[1] - data_min[1]),
        scale / (data_max[2] - data_min[2]),
    ];
    let mut output = image.clone();
    output
        .data
        .par_chunks_exact_mut(3)
        .zip(image.data.par_chunks_exact(3))
        .for_each(|(dst, src)| {
            let coordinates = [
                (src[0] as f64 - data_min[0]) * inverse_range[0],
                (src[1] as f64 - data_min[1]) * inverse_range[1],
                (src[2] as f64 - data_min[2]) * inverse_range[2],
            ];
            let rgb =
                tetrahedral_3d_f64(data, size, coordinates[0], coordinates[1], coordinates[2]);
            dst[0] = from_f64(rgb[0]);
            dst[1] = from_f64(rgb[1]);
            dst[2] = from_f64(rgb[2]);
        });
    output
}

/// Mitchell-Netravali bicubic 2D interpolation on f64 data.
///
/// Bit-identical port of Python `_cubic_interp_lut_at_2d`. Argument
/// naming follows Python: `x` indexes the first LUT axis, `y` the
/// second. The data buffer must be laid out so offset
/// `(x_idx * second_dim + y_idx) * channels + c` retrieves
/// `lut[x_idx, y_idx, c]` — that is what `compute_tc_lut` produces.
pub fn bicubic_2d_f64(
    data: &[f64],
    first_dim: usize,
    second_dim: usize,
    channels: usize,
    x: f64,
    y: f64,
) -> Vec<f64> {
    let max_x = (first_dim - 1) as f64;
    let max_y = (second_dim - 1) as f64;
    let xf = x.clamp(0.0, max_x);
    let yf = y.clamp(0.0, max_y);
    let xi = if xf >= max_x {
        first_dim - 2
    } else {
        xf as usize
    };
    let yi = if yf >= max_y {
        second_dim - 2
    } else {
        yf as usize
    };
    let fx = xf - xi as f64;
    let fy = yf - yi as f64;
    let wx = [
        mitchell_weight(fx + 1.0),
        mitchell_weight(fx),
        mitchell_weight(fx - 1.0),
        mitchell_weight(fx - 2.0),
    ];
    let wy = [
        mitchell_weight(fy + 1.0),
        mitchell_weight(fy),
        mitchell_weight(fy - 1.0),
        mitchell_weight(fy - 2.0),
    ];
    let mut result = vec![0.0f64; channels];
    let mut weight_sum = 0.0f64;
    for i in 0..4usize {
        let xi_idx = reflect_index(xi as isize + i as isize - 1, first_dim);
        for j in 0..4usize {
            let yj_idx = reflect_index(yi as isize + j as isize - 1, second_dim);
            let weight = wx[i] * wy[j];
            weight_sum += weight;
            let base = (xi_idx * second_dim + yj_idx) * channels;
            for c in 0..channels {
                result[c] += weight * data[base + c];
            }
        }
    }
    if weight_sum != 0.0 {
        for c in 0..channels {
            result[c] /= weight_sum;
        }
    }
    result
}

/// Mitchell-Netravali bicubic 2D interpolation on a regular grid.
///
/// Uses B=1/3, C=1/3 kernel matching Python's `fast_interp_lut._cubic_interp_lut_at_2d`.
/// Reflected boundary handling. f64 accumulation for precision.
pub fn bicubic_2d(
    data: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    x: f32,
    y: f32,
) -> Vec<f32> {
    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;

    let xf = (x as f64).clamp(0.0, max_x);
    let yf = (y as f64).clamp(0.0, max_y);

    // Base cell (clamped to allow cubic stencil)
    let xi = if xf >= max_x { width - 2 } else { xf as usize };
    let yi = if yf >= max_y { height - 2 } else { yf as usize };

    let fx = xf - xi as f64;
    let fy = yf - yi as f64;

    // Mitchell-Netravali weights
    let wx = [
        mitchell_weight(fx + 1.0),
        mitchell_weight(fx),
        mitchell_weight(fx - 1.0),
        mitchell_weight(fx - 2.0),
    ];
    let wy = [
        mitchell_weight(fy + 1.0),
        mitchell_weight(fy),
        mitchell_weight(fy - 1.0),
        mitchell_weight(fy - 2.0),
    ];

    let mut result = vec![0.0f32; channels];

    for c in 0..channels {
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for dy in 0..4usize {
            let sy = reflect_index(yi as isize + dy as isize - 1, height);
            for dx in 0..4usize {
                let sx = reflect_index(xi as isize + dx as isize - 1, width);
                let w = wx[dx] * wy[dy];
                weight_sum += w;
                sum += w * data[(sy * width + sx) * channels + c] as f64;
            }
        }
        result[c] = if weight_sum != 0.0 {
            (sum / weight_sum) as f32
        } else {
            0.0
        };
    }

    result
}

/// Mitchell-Netravali kernel weight (B=1/3, C=1/3).
/// Matches Python's `mitchell_weight` in `fast_interp_lut.py`.
#[inline]
fn mitchell_weight(t: f64) -> f64 {
    const B: f64 = 1.0 / 3.0;
    const C: f64 = 1.0 / 3.0;
    let x = t.abs();
    if x < 1.0 {
        (1.0 / 6.0)
            * ((12.0 - 9.0 * B - 6.0 * C) * x * x * x
                + (-18.0 + 12.0 * B + 6.0 * C) * x * x
                + (6.0 - 2.0 * B))
    } else if x < 2.0 {
        (1.0 / 6.0)
            * ((-B - 6.0 * C) * x * x * x
                + (6.0 * B + 30.0 * C) * x * x
                + (-12.0 * B - 48.0 * C) * x
                + (8.0 * B + 24.0 * C))
    } else {
        0.0
    }
}

/// Reflect index into valid range [0, L-1] using symmetric reflection.
/// Matches Python's `safe_index`.
#[inline]
fn reflect_index(idx: isize, len: usize) -> usize {
    if idx < 0 {
        (-idx) as usize
    } else if idx >= len as isize {
        (2 * (len as isize - 1) - idx) as usize
    } else {
        idx as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetrahedral_3d_f64_preserves_linear_lut() {
        let size = 3;
        let mut data = vec![0.0; size * size * size * 3];
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let base = ((r * size + g) * size + b) * 3;
                    data[base] = r as f64 / 2.0;
                    data[base + 1] = g as f64 / 2.0;
                    data[base + 2] = b as f64 / 2.0;
                }
            }
        }
        for (r, g, b) in [
            (0.2, 0.6, 0.9),
            (0.9, 0.6, 0.2),
            (2.0, 1.0, 0.0),
            (-1.0, 0.5, 4.0),
        ] {
            let output = tetrahedral_3d_f64(&data, size, r, g, b);
            let expected = [
                r.clamp(0.0, 2.0) / 2.0,
                g.clamp(0.0, 2.0) / 2.0,
                b.clamp(0.0, 2.0) / 2.0,
            ];
            for channel in 0..3 {
                assert!(
                    (output[channel] - expected[channel]).abs() < 1e-12,
                    "input=({r}, {g}, {b}), got {output:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn tetrahedral_3d_f64_matches_python_vertex_ordering() {
        let size = 2;
        let mut data = vec![0.0; size * size * size * 3];
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let f =
                        (r + 2 * g + 4 * b + 8 * r * g + 16 * g * b + 32 * r * b + 64 * r * g * b)
                            as f64;
                    let base = ((r * size + g) * size + b) * 3;
                    data[base] = f;
                    data[base + 1] = 2.0 * f + 1.0;
                    data[base + 2] = -f;
                }
            }
        }
        let cases = [
            ((0.2, 0.6, 0.9), 35.4),
            ((0.6, 0.2, 0.9), 41.4),
            ((0.6, 0.9, 0.2), 30.4),
            ((0.2, 0.9, 0.6), 34.8),
            ((0.9, 0.2, 0.6), 40.5),
            ((0.9, 0.6, 0.2), 30.1),
        ];
        for ((r, g, b), expected) in cases {
            let output = tetrahedral_3d_f64(&data, size, r, g, b);
            assert!((output[0] - expected).abs() < 1e-12);
            assert!((output[1] - (2.0 * expected + 1.0)).abs() < 1e-12);
            assert!((output[2] + expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_trilinear_corners() {
        // 2x2x2 identity LUT
        let size = 2;
        let mut data = vec![0.0f32; size * size * size * 3];
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let idx = (r * size * size + g * size + b) * 3;
                    data[idx] = r as f32;
                    data[idx + 1] = g as f32;
                    data[idx + 2] = b as f32;
                }
            }
        }
        let v = trilinear_3d(&data, size, 0.0, 0.0, 0.0);
        assert_eq!(v, [0.0, 0.0, 0.0]);
        let v = trilinear_3d(&data, size, 1.0, 1.0, 1.0);
        assert_eq!(v, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_trilinear_midpoint() {
        let size = 2;
        let mut data = vec![0.0f32; size * size * size * 3];
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let idx = (r * size * size + g * size + b) * 3;
                    data[idx] = r as f32;
                    data[idx + 1] = g as f32;
                    data[idx + 2] = b as f32;
                }
            }
        }
        let v = trilinear_3d(&data, size, 0.5, 0.5, 0.5);
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[1] - 0.5).abs() < 1e-6);
        assert!((v[2] - 0.5).abs() < 1e-6);
    }
}
