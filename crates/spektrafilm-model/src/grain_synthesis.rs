//! Aperture-integrated inhomogeneous Boolean grain model ("Monte Carlo" grain).
//!
//! Silver grains are disks whose centres form a Poisson process in the film
//! plane with lognormal radii. Each grain develops with the local development
//! probability `p = (D + d_min) / (D_max + d_min)` (thinning), and the density
//! seen through a Gaussian observation aperture is `-log10(1 - coverage)`.
//! Calibrating `lambda * E[pi R^2] = D_max * ln(10)` makes the expected density
//! equal the input density, so the texture is generated around the image.
//!
//! The coverage under the aperture is integrated deterministically instead of
//! being estimated from a handful of random sample points. A binomial estimate
//! from 16 samples has a standard deviation several times larger than the real
//! grain fluctuation, which showed up as per-pixel colour speckle, a
//! convexity bias in `-log10` (washed-out tones) and a dependence on the epsilon
//! clamp. Here every strip rasterises the developed grains onto a fine grid,
//! reduces them to aperture-sized cells and applies the exact Gaussian
//! box-integral weights, so only the physical grain structure remains.
//!
//! Real emulsions stack several grain layers whose densities add. With
//! `layered` the density budget is split over layers of different grain sizes,
//! which gives a multi-scale texture and keeps deep-shadow holes from
//! dominating the noise.
//!
//! Every random number is derived from a counter-based hash of absolute cell
//! coordinates, so the output is independent of thread scheduling and of how
//! the image is split into strips.

use rayon::prelude::*;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{from_f64, ZERO};
use spektrafilm_math::progress;
use std::f64::consts::{LN_10, PI};

const STRIP_PIXEL_ROWS: usize = 16;
const FINE_BAND_ROWS: usize = 64;
const LOG_QUANTILE_Z: f64 = 3.09; // 0.999 quantile of the lognormal radius

fn mix64(value: u64) -> u64 {
    let value = (value ^ (value >> 33)).wrapping_mul(0xff51afd7ed558ccd);
    let value = (value ^ (value >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
    value ^ (value >> 33)
}

fn hash01(a: u64, b: u64, c: u64, seed: u64) -> f64 {
    let hash = mix64(
        a.wrapping_mul(0x9e3779b97f4a7c15)
            ^ b.wrapping_mul(0xbf58476d1ce4e5b9)
            ^ c.wrapping_mul(0x94d049bb133111eb)
            ^ seed,
    );
    ((hash >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
}

fn hash_pair(a: u64, b: u64, c: u64, seed: u64) -> (f64, f64) {
    (hash01(a, b, c, seed), hash01(a, b, c.wrapping_add(1), seed))
}

pub(crate) fn cell_poisson(cell_x: isize, cell_y: isize, lambda: f64, cap: u32, seed: u64) -> u32 {
    let threshold = (-lambda).exp();
    let mut product = 1.0;
    let mut count = 0;
    while count <= cap {
        product *= hash01(cell_x as u64, cell_y as u64, 900_000 + count as u64, seed);
        if product <= threshold {
            break;
        }
        count += 1;
    }
    count.min(cap)
}

/// Complementary error function (Numerical Recipes Chebyshev fit, |err| < 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87
                                    + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let value = t * poly.exp();
    if x >= 0.0 {
        value
    } else {
        2.0 - value
    }
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Observation aperture discretisation shared by every layer of a channel.
struct Aperture {
    /// Aperture cells per pixel edge; a cell is `cell_um = pixel_um / cells`.
    cells: usize,
    cell_um: f64,
    /// First and last aperture-cell offsets relative to a pixel's first cell.
    first_offset: isize,
    /// Cells needed beyond the image edge on every side.
    pad: isize,
    /// Exact Gaussian box-integral weights, one per tap.
    weights: Vec<f64>,
}

impl Aperture {
    fn new(pixel_um: f64, sigma_um: f64) -> Self {
        let cells = ((pixel_um / (0.5 * sigma_um)).ceil() as usize).clamp(1, 16);
        let cell_um = pixel_um / cells as f64;
        let sigma_cells = (sigma_um / cell_um).max(1e-6);
        let half = 0.5 * cells as f64;
        let first_offset = (half - 4.0 * sigma_cells).floor() as isize;
        let last_offset = (half + 4.0 * sigma_cells).ceil() as isize - 1;
        let mut weights: Vec<f64> = (first_offset..=last_offset)
            .map(|d| {
                normal_cdf(((d + 1) as f64 - half) / sigma_cells)
                    - normal_cdf((d as f64 - half) / sigma_cells)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        weights.iter_mut().for_each(|w| *w /= total);
        let pad = (-first_offset).max(last_offset + 1 - cells as isize).max(0);
        Self {
            cells,
            cell_um,
            first_offset,
            pad,
            weights,
        }
    }
}

/// One grain layer of one channel.
struct Layer {
    mu_r: f64,
    sigma_r: f64,
    r_max: f64,
    cell_um: f64,
    lambda_cell: f64,
    cap: u32,
    /// Raster cells per aperture cell edge.
    subdivisions: usize,
    /// Smallest uncovered fraction that is reported; bounds the density.
    floor_uncovered: f64,
    seed: u64,
}

fn render_layer(
    p_map: &[f64],
    width: usize,
    height: usize,
    pixel_um: f64,
    aperture: &Aperture,
    layer: &Layer,
    strip_rows: usize,
) -> Vec<f64> {
    let mut field = vec![0.0; width * height];
    field
        .par_chunks_mut(strip_rows * width)
        .enumerate()
        .for_each(|(strip, out)| {
            let row0 = strip * strip_rows;
            let rows = out.len() / width;
            render_strip(
                p_map, width, height, pixel_um, aperture, layer, row0, rows, out,
            );
        });
    field
}

#[allow(clippy::too_many_arguments)]
fn render_strip(
    p_map: &[f64],
    width: usize,
    height: usize,
    pixel_um: f64,
    aperture: &Aperture,
    layer: &Layer,
    row0: usize,
    rows: usize,
    out: &mut [f64],
) {
    let k = aperture.cells as isize;
    let pad = aperture.pad;
    let sub = layer.subdivisions;
    let fine_um = aperture.cell_um / sub as f64;
    let y_lo = row0 as isize * k - pad;
    let y_hi = (row0 + rows) as isize * k + pad;
    let x_lo = -pad;
    let x_hi = width as isize * k + pad;
    let cols = (x_hi - x_lo) as usize;
    let coarse_rows = (y_hi - y_lo) as usize;
    let fine_cols = cols * sub;
    let fine_x0 = x_lo * sub as isize;
    let fine_x1 = x_hi * sub as isize;

    let mut counts = vec![0u16; cols * coarse_rows];
    let band_rows = (FINE_BAND_ROWS / sub).max(1) as isize;
    let mut fine = vec![0u8; band_rows as usize * sub * fine_cols];

    let mut band_start = y_lo;
    while band_start < y_hi {
        let band_end = (band_start + band_rows).min(y_hi);
        let fine_y0 = band_start * sub as isize;
        let fine_y1 = band_end * sub as isize;
        fine[..(fine_y1 - fine_y0) as usize * fine_cols].fill(0);

        let band_y0_um = band_start as f64 * aperture.cell_um;
        let band_y1_um = band_end as f64 * aperture.cell_um;
        let cell_y_first = ((band_y0_um - layer.r_max) / layer.cell_um).floor() as isize;
        let cell_y_last = ((band_y1_um + layer.r_max) / layer.cell_um).floor() as isize;
        let cell_x_first =
            ((x_lo as f64 * aperture.cell_um - layer.r_max) / layer.cell_um).floor() as isize;
        let cell_x_last =
            ((x_hi as f64 * aperture.cell_um + layer.r_max) / layer.cell_um).floor() as isize;

        for cell_y in cell_y_first..=cell_y_last {
            for cell_x in cell_x_first..=cell_x_last {
                let count =
                    cell_poisson(cell_x, cell_y, layer.lambda_cell, layer.cap, layer.seed);
                for grain in 0..count {
                    let g = grain as u64 * 8;
                    let (ux, uy) = hash_pair(cell_x as u64, cell_y as u64, g, layer.seed);
                    let grain_x = (cell_x as f64 + ux) * layer.cell_um;
                    let grain_y = (cell_y as f64 + uy) * layer.cell_um;
                    if grain_y + layer.r_max < band_y0_um || grain_y - layer.r_max >= band_y1_um {
                        continue;
                    }
                    let pixel_x = ((grain_x / pixel_um) as isize).clamp(0, width as isize - 1);
                    let pixel_y = ((grain_y / pixel_um) as isize).clamp(0, height as isize - 1);
                    let develop = hash01(cell_x as u64, cell_y as u64, g + 4, layer.seed);
                    if develop >= p_map[pixel_y as usize * width + pixel_x as usize] {
                        continue;
                    }
                    let (v1, v2) = hash_pair(cell_x as u64, cell_y as u64, g + 2, layer.seed);
                    let normal = (-2.0 * v1.ln()).sqrt() * (2.0 * PI * v2).cos();
                    let radius = (layer.mu_r + layer.sigma_r * normal).exp().min(layer.r_max);
                    let radius_sq = radius * radius;

                    let row_first = (((grain_y - radius) / fine_um - 0.5).ceil() as isize)
                        .max(fine_y0);
                    let row_last = (((grain_y + radius) / fine_um - 0.5).floor() as isize)
                        .min(fine_y1 - 1);
                    for fine_row in row_first..=row_last {
                        let dy = (fine_row as f64 + 0.5) * fine_um - grain_y;
                        let chord_sq = radius_sq - dy * dy;
                        if chord_sq < 0.0 {
                            continue;
                        }
                        let half_chord = chord_sq.sqrt();
                        let col_first = (((grain_x - half_chord) / fine_um - 0.5).ceil() as isize)
                            .max(fine_x0);
                        let col_last = (((grain_x + half_chord) / fine_um - 0.5).floor() as isize)
                            .min(fine_x1 - 1);
                        if col_last < col_first {
                            continue;
                        }
                        let row_base = (fine_row - fine_y0) as usize * fine_cols;
                        fine[row_base + (col_first - fine_x0) as usize
                            ..=row_base + (col_last - fine_x0) as usize]
                            .fill(1);
                    }
                }
            }
        }

        for row in 0..(band_end - band_start) as usize {
            let target =
                &mut counts[((band_start - y_lo) as usize + row) * cols..][..cols];
            for s in 0..sub {
                let source = &fine[(row * sub + s) * fine_cols..][..fine_cols];
                for (acc, chunk) in target.iter_mut().zip(source.chunks_exact(sub)) {
                    *acc += chunk.iter().map(|&v| v as u16).sum::<u16>();
                }
            }
        }
        band_start = band_end;
    }

    let area_norm = 1.0 / (sub * sub) as f64;
    let weights = &aperture.weights;
    let mut horizontal = vec![0.0f64; coarse_rows * width];
    for (row, destination) in horizontal.chunks_exact_mut(width).enumerate() {
        let source = &counts[row * cols..][..cols];
        for (px, value) in destination.iter_mut().enumerate() {
            let base = (px as isize * k + aperture.first_offset + pad) as usize;
            let mut acc = 0.0;
            for (tap, weight) in weights.iter().enumerate() {
                acc += weight * source[base + tap] as f64;
            }
            *value = acc * area_norm;
        }
    }
    for py in 0..rows {
        let base_row = (py as isize * k + aperture.first_offset + pad) as usize;
        for px in 0..width {
            let mut coverage = 0.0;
            for (tap, weight) in weights.iter().enumerate() {
                coverage += weight * horizontal[(base_row + tap) * width + px];
            }
            let uncovered = (1.0 - coverage).max(layer.floor_uncovered);
            out[py * width + px] = -uncovered.log10();
        }
    }
}

fn render_gross(
    p_map: &[f64],
    width: usize,
    height: usize,
    pixel_um: f64,
    aperture: &Aperture,
    layers: &[Layer],
    strip_rows: usize,
    on_layer_done: &dyn Fn(usize),
) -> Vec<f64> {
    let mut gross = vec![0.0f64; width * height];
    for (index, layer) in layers.iter().enumerate() {
        let field = render_layer(p_map, width, height, pixel_um, aperture, layer, strip_rows);
        gross.iter_mut().zip(field).for_each(|(g, f)| *g += f);
        on_layer_done(index);
    }
    gross
}

/// Mean summed layer density of a flat patch as a function of the
/// development probability, sampled on a regular grid.
struct ExpectedDensity {
    mean: Vec<f64>,
}

const TABLE_POINTS: usize = 17;
const TABLE_PATCH_PIXELS: usize = 96;

impl ExpectedDensity {
    fn lookup(&self, p: f64) -> f64 {
        let position = p.clamp(0.0, 1.0) * (TABLE_POINTS - 1) as f64;
        let low = (position.floor() as usize).min(TABLE_POINTS - 2);
        let frac = position - low as f64;
        self.mean[low] * (1.0 - frac) + self.mean[low + 1] * frac
    }
}

fn expected_density_table(
    pixel_um: f64,
    aperture: &Aperture,
    layers: &[Layer],
    strip_rows: usize,
) -> ExpectedDensity {
    let side = TABLE_PATCH_PIXELS;
    let mean = (0..TABLE_POINTS)
        .map(|i| {
            let p = (i as f64 / (TABLE_POINTS - 1) as f64).clamp(1e-6, 1.0 - 1e-6);
            let flat = vec![p; side * side];
            let gross = render_gross(&flat, side, side, pixel_um, aperture, layers, strip_rows, &|_| {});
            gross.iter().sum::<f64>() / gross.len() as f64
        })
        .collect();
    ExpectedDensity { mean }
}

/// Render the grain field for a density image and blend it with the input.
#[allow(clippy::too_many_arguments)]
pub fn apply_grain_monte_carlo(
    density: &ImageBuf,
    pixel_um: f32,
    density_min: [f64; 3],
    density_max: [f64; 3],
    amount: f64,
    samples: u32,
    mean_radius_um: f64,
    radius_stddev_ratio: f64,
    cell_ratio: f64,
    max_grains: u32,
    layered: bool,
    layer_scales: [f64; 3],
    monochrome: bool,
    _backend: &dyn ComputeBackend,
) -> ImageBuf {
    render(
        density,
        pixel_um as f64,
        density_min,
        density_max,
        amount,
        samples,
        mean_radius_um,
        radius_stddev_ratio,
        cell_ratio,
        max_grains,
        layered,
        layer_scales,
        monochrome,
        STRIP_PIXEL_ROWS,
    )
}

#[allow(clippy::too_many_arguments)]
fn render(
    density: &ImageBuf,
    pixel_um: f64,
    density_min: [f64; 3],
    density_max: [f64; 3],
    amount: f64,
    samples: u32,
    mean_radius_um: f64,
    radius_stddev_ratio: f64,
    cell_ratio: f64,
    max_grains: u32,
    layered: bool,
    layer_scales: [f64; 3],
    monochrome: bool,
    strip_rows: usize,
) -> ImageBuf {
    if amount == 0.0 {
        return density.clone();
    }
    let width = density.width as usize;
    let height = density.height as usize;
    let sigma_um = 0.5 * pixel_um;
    let aperture = Aperture::new(pixel_um, sigma_um);
    let ratio = radius_stddev_ratio.max(1e-3);
    let sigma_r = (1.0 + ratio * ratio).ln().sqrt();
    let scales: Vec<f64> = if layered {
        layer_scales.iter().map(|s| s.max(1e-3)).collect()
    } else {
        vec![1.0]
    };
    // More requested samples refine the raster that resolves the grain edges.
    let refinement = (samples.max(1) as f64 / 16.0).sqrt().max(0.5);
    let channel_count = if monochrome { 1 } else { 3 };
    let mut out = ImageBuf::new(density.width, density.height);

    for channel in 0..channel_count {
        let dmin = density_min[channel];
        let dmax = (density_max[channel] + dmin).max(1e-9);
        let p_map: Vec<f64> = (0..width * height)
            .map(|i| ((density.data[i * 3 + channel] as f64 + dmin) / dmax).clamp(1e-6, 1.0 - 1e-6))
            .collect();

        let share = dmax / scales.len() as f64;
        let layers: Vec<Layer> = scales.iter().enumerate().map(|(index, scale)| {
            let mean_radius = (mean_radius_um * scale).max(1e-3);
            let mu_r = mean_radius.ln() - 0.5 * sigma_r * sigma_r;
            let r_max = (mu_r + LOG_QUANTILE_Z * sigma_r).exp();
            let expected_area = PI * mean_radius * mean_radius * (1.0 + ratio * ratio);
            let cell_um = (cell_ratio.max(0.05) * mean_radius).max(1e-2);
            let lambda_cell = share * LN_10 / expected_area * cell_um * cell_um;
            // The user cap must not truncate the Poisson tail for dense
            // emulsions, which would silently remove coverage.
            let cap = max_grains
                .max(1)
                .max((lambda_cell + 10.0 * lambda_cell.sqrt() + 10.0).ceil() as u32);
            let subdivisions = (((aperture.cell_um / (mean_radius / 2.5)) * refinement).ceil()
                as usize)
                .clamp(1, 16);
            Layer {
                mu_r,
                sigma_r,
                r_max,
                cell_um,
                lambda_cell,
                cap,
                subdivisions,
                floor_uncovered: 10f64.powf(-(share + 1.0)),
                seed: 11 + channel as u64 + 1000 * index as u64,
            }
        }).collect();

        let table = expected_density_table(pixel_um, &aperture, &layers, strip_rows);
        let layer_count = layers.len();
        let gross = render_gross(
            &p_map, width, height, pixel_um, &aperture, &layers, strip_rows,
            &|index| {
                let done = channel * layer_count + index + 1;
                progress::report_in_span(
                    done as f32 / (channel_count * layer_count) as f32,
                    "Rendering grain",
                );
            },
        );

        // The -log10 of a noisy coverage is biased upward. Measure the mean
        // grain density of flat patches at the same settings and subtract it,
        // so the grain adds zero-mean texture around the input density.
        let table = expected_density_table(pixel_um, &aperture, &layers, strip_rows);

        let mut channel_data = vec![ZERO; width * height];
        for (i, value) in channel_data.iter_mut().enumerate() {
            let original = density.data[i * 3 + channel] as f64;
            let expected = table.lookup(p_map[i]);
            let texture = gross[i] - expected;
            *value = from_f64((original + amount * texture).max(-dmin));
        }
        out.write_channel(channel, &channel_data);
    }
    if monochrome {
        let gray = out.extract_channel(0);
        out.write_channel(1, &gray);
        out.write_channel(2, &gray);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use spektrafilm_gpu::cpu_backend::CpuBackend;

    fn flat(width: u32, height: u32, value: f64) -> ImageBuf {
        ImageBuf::from_data(
            width,
            height,
            (0..width as usize * height as usize)
                .flat_map(|_| [from_f64(value); 3])
                .collect(),
        )
    }

    fn stats(image: &ImageBuf, channel: usize) -> (f64, f64) {
        let values: Vec<f64> = image
            .extract_channel(channel)
            .iter()
            .map(|&v| v as f64)
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        (mean, var.sqrt())
    }

    fn run(image: &ImageBuf, layered: bool, strip_rows: usize) -> ImageBuf {
        render(
            image, 3.9167, [0.03; 3], [2.5; 3], 1.0, 16, 0.6, 0.4, 4.0, 32, layered,
            [2.0, 1.0, 0.5], false, strip_rows,
        )
    }

    #[test]
    fn poisson_cell_counts_have_the_requested_mean() {
        let counts: f64 = (0..10_000)
            .map(|x| cell_poisson(x, x.wrapping_mul(37), 5.0, 32, 11) as f64)
            .sum();
        assert!((counts / 10_000.0 - 5.0).abs() < 0.15);
    }

    #[test]
    fn aperture_weights_are_normalised_and_symmetric() {
        let aperture = Aperture::new(3.9167, 0.5 * 3.9167);
        let sum: f64 = aperture.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        let n = aperture.weights.len();
        let peak = aperture.weights.iter().cloned().fold(0.0, f64::max);
        assert!(peak < 0.2);
        assert!(aperture.weights[0] < 1e-3 && aperture.weights[n - 1] < 1e-3);
    }

    #[test]
    fn mean_density_is_preserved_across_the_tone_scale() {
        for layered in [false, true] {
            for target in [0.15, 0.6, 1.2, 2.0] {
                let result = run(&flat(48, 48, target), layered, STRIP_PIXEL_ROWS);
                let (mean, std) = stats(&result, 0);
                // A single dense layer is Jensen-biased by the convex log; the
                // layered default is the accuracy-critical path.
                let tolerance = 0.03 + 0.02 * target;
                assert!(
                    (mean - target).abs() < tolerance,
                    "layered={layered} density {target} drifted to {mean}",
                );
                let std_limit = if target > 1.5 { 0.6 } else { 0.25 };
                assert!(std < std_limit, "layered={layered} density {target} std {std}");
            }
        }
    }

    #[test]
    fn output_is_independent_of_strip_partitioning() {
        let image = flat(20, 23, 0.7);
        let whole = run(&image, true, 32);
        let split = run(&image, true, 5);
        assert_eq!(whole.data, split.data);
    }

    #[test]
    fn channels_are_decorrelated_but_statistically_equal() {
        let result = run(&flat(48, 48, 0.8), true, STRIP_PIXEL_ROWS);
        let a = result.extract_channel(0);
        let b = result.extract_channel(1);
        assert_ne!(a, b);
        let (ma, sa) = stats(&result, 0);
        let (mb, sb) = stats(&result, 1);
        assert!((ma - mb).abs() < 0.03 && (sa - sb).abs() < 0.03);
    }

    #[test]
    fn zero_amount_keeps_the_input_unchanged() {
        let density = flat(2, 2, 0.2);
        let result = apply_grain_monte_carlo(
            &density, 1.0, [0.03; 3], [1.0; 3], 0.0, 16, 0.6, 0.4, 4.0, 32, true,
            [2.0, 1.0, 0.5], false, &CpuBackend,
        );
        assert_eq!(result.data, density.data);
    }
}
