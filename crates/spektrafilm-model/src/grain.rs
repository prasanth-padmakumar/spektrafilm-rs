// Stochastic grain generation.
// Poisson-binomial particle model with dye cloud blur and lognormal micro-structure.

use rayon::prelude::*;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::gaussian;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{from_f64, Scalar, ZERO};

/// Apply the Poisson-binomial grain particle model to a single-channel density image.
///
/// Port of Python `layer_particle_model`. Density values flow through in `Scalar`
/// (full f64 precision in `precision-f64` mode); RNG sampling uses f64 directly.
pub fn layer_particle_model(
    density: &[Scalar],
    width: u32,
    height: u32,
    density_max: f64,
    n_particles_per_pixel: f64,
    grain_uniformity: f64,
    seed: u64,
    blur_particle: f32,
) -> Vec<Scalar> {
    // Inputs are f64 to match Python's `layer_particle_model` —
    // density_max/uniformity/agx-area come from JSON profiles at full
    // f64 precision in Python. Passing them as f32 in Rust truncated
    // ~7-decimal noise into `od_particle` and `saturation` which then
    // shifted every Poisson lambda by ~5e-8, producing a different
    // RNG stream and thus visibly different grain patterns.
    let dmax = density_max;
    let od_particle = dmax / n_particles_per_pixel;
    let gu = grain_uniformity;
    let npp = n_particles_per_pixel;

    // Python-bit-exact RNG: numpy's MT19937 + RandomState distribution
    // algorithms (`rk_poisson` / `rk_binomial`). Numpy's seed-to-stream
    // takes a `u32`, so truncate the 64-bit seed — for the CPU export
    // path we never need more than 24 distinct streams (3 channels × N
    // sub-layers). Iteration is single-threaded so the RNG state advances
    // in row-major order exactly like `numpy.random.RandomState(seed)
    // .poisson(lambda_array)` does internally.
    //
    // CRITICAL — match Python's call order: it does
    //   seeds = np.random.poisson(lam_array)   # whole array, then
    //   grain = np.random.binomial(seeds, p)   # whole array.
    // So the RNG must draw ALL Poisson samples first, THEN ALL Binomial
    // samples. Interleaving Poisson/Binomial per pixel (what we used to
    // do) gives a different RNG state at each draw and produces a
    // completely different grain pattern.
    let n = density.len();
    let mut rng = rand_mt::Mt::new(seed as u32);
    let mut p_arr = Vec::with_capacity(n);
    let mut sat_arr = Vec::with_capacity(n);
    for &d in density.iter() {
        let p = ((d as f64) / dmax).clamp(1e-6, 1.0 - 1e-6);
        let saturation = 1.0 - p * gu * (1.0 - 1e-6);
        p_arr.push(p);
        sat_arr.push(saturation);
    }
    let mut seeds = Vec::with_capacity(n);
    for sat in &sat_arr {
        seeds.push(spektrafilm_math::numpy_rng::rk_poisson(&mut rng, npp / sat));
    }
    let mut grain = vec![ZERO; n];
    for (i, slot) in grain.iter_mut().enumerate() {
        let developed = spektrafilm_math::numpy_rng::rk_binomial(&mut rng, seeds[i], p_arr[i]);
        *slot = from_f64((developed as f64) * od_particle * sat_arr[i]);
    }

    if blur_particle > 0.4 {
        // Match Python: `sigma = blur_particle * np.sqrt(od_particle)`
        // — keep the multiplication in f64 and narrow at the end.
        let sigma = (blur_particle as f64 * od_particle.sqrt()) as f32;
        if sigma > 0.1 {
            let mut img = ImageBuf::from_data(
                width,
                height,
                grain.iter().flat_map(|&v| [v, v, v]).collect(),
            );
            img = gaussian::gaussian_blur(&img, sigma);
            grain = img.extract_channel(0);
        }
    }

    grain
}

/// Apply grain to a CMY density image.
///
/// Port of Python `apply_grain_to_density`.
#[allow(clippy::too_many_arguments)]
pub fn apply_grain_to_density(
    density_cmy: &ImageBuf,
    pixel_size_um: f32,
    agx_particle_area_um2: f64,
    agx_particle_scale: [f64; 3],
    density_min: [f64; 3],
    density_max_curves: [f64; 3],
    grain_uniformity: [f64; 3],
    grain_blur: f32,
    n_sub_layers: u32,
    monochrome: bool,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    let w = density_cmy.width;
    let h = density_cmy.height;
    let pixel_area = (pixel_size_um as f64) * (pixel_size_um as f64);
    let density_max: [f64; 3] = [
        density_max_curves[0] + density_min[0],
        density_max_curves[1] + density_min[1],
        density_max_curves[2] + density_min[2],
    ];

    let channels: Vec<(usize, Vec<Scalar>)> = (0..3)
        .into_par_iter()
        .map(|ch| {
            let particle_area = agx_particle_area_um2 * agx_particle_scale[ch];
            let mut n_particles = pixel_area / particle_area;
            if n_sub_layers > 1 {
                n_particles /= n_sub_layers as f64;
            }

            // Add density_min to input (kept in Scalar precision)
            let dmin_s = from_f64(density_min[ch]);
            let density_ch: Vec<Scalar> = density_cmy.pixels().map(|px| px[ch] + dmin_s).collect();

            let mut grain_sum = vec![ZERO; density_ch.len()];

            for sl in 0..n_sub_layers {
                // Monochrome (B&W): one shared noise field — every lane draws
                // from channel 0's RNG stream (upstream n_channels==1 has a
                // single emulsion, seed ch=0).
                let seed_ch = if monochrome { 0 } else { ch as u64 };
                let seed = seed_ch + (sl as u64) * 10;
                let g = layer_particle_model(
                    &density_ch,
                    w,
                    h,
                    density_max[ch],
                    n_particles,
                    grain_uniformity[ch],
                    seed,
                    0.0,
                );
                for (s, &v) in grain_sum.iter_mut().zip(g.iter()) {
                    *s += v;
                }
            }

            // Average sub-layers
            let scale = from_f64(1.0 / n_sub_layers as f64);
            for v in &mut grain_sum {
                *v = *v * scale - dmin_s;
            }

            (ch, grain_sum)
        })
        .collect();

    let mut out = ImageBuf::new(w, h);
    for (ch, grain_sum) in channels {
        out.write_channel(ch, &grain_sum);
    }

    // Final blur — typically a few px sigma at 1–6 MP, big enough that the
    // GPU separable kernel wins.
    if grain_blur > 0.4 {
        out = backend.gaussian_blur(&out, grain_blur);
    }

    out
}

pub use crate::grain_synthesis::apply_grain_monte_carlo;

/// Apply the Python RMS-granularity model using the measured per-emulsion
/// density curves. `density_curves_layers` is `[exposure][layer][channel]`.
#[allow(clippy::too_many_arguments)]
pub fn apply_grain_rms_to_density(
    density_cmy: &ImageBuf,
    pixel_size_um: f32,
    rms_granularity: [f64; 3],
    density_min: [f64; 3],
    grain_uniformity: [f64; 3],
    particle_scale_sublayers: [f64; 3],
    density_curves: &[[f64; 3]],
    density_curves_layers: &[Vec<Vec<f64>>],
    positive_film: bool,
    grain_blur: f32,
    blur_dye_clouds_um: f32,
    mult_usm_sigma: f64,
    mult_usm_amount: f64,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    const APERTURE_AREA_UM2: f64 = std::f64::consts::PI * 24.0 * 24.0;
    let width = density_cmy.width as usize;
    let height = density_cmy.height as usize;
    let pixel_area = (pixel_size_um as f64).powi(2);
    let curves = spektrafilm_model_curves_by_layer(density_curves, density_curves_layers);
    let n_layers = curves.len();

    let channels: Vec<(usize, Vec<Scalar>)> = (0..3)
        .into_par_iter()
        .map(|channel| {
            let mut layer_max = Vec::with_capacity(n_layers);
            for layer in &curves {
                layer_max.push(
                    layer
                        .iter()
                        .map(|point| point[channel])
                        .filter(|value| value.is_finite())
                        .fold(f64::NEG_INFINITY, f64::max),
                );
            }
            let total_max: f64 = layer_max.iter().sum();
            let fractions: Vec<f64> = layer_max.iter().map(|value| value / total_max).collect();
            let layer_min: Vec<f64> = fractions
                .iter()
                .map(|fraction| fraction * density_min[channel])
                .collect();
            let layer_max_absolute: Vec<f64> = layer_max
                .iter()
                .zip(&layer_min)
                .map(|(maximum, minimum)| maximum + minimum)
                .collect();
            let mut peak_variance = 0.0_f64;
            for exposure_index in 0..curves[0].len() {
                let mut summed_variance = 0.0;
                for layer_index in 0..n_layers {
                    let absolute_density =
                        curves[layer_index][exposure_index][channel] + layer_min[layer_index];
                    let scale = particle_scale_sublayers
                        .get(layer_index)
                        .copied()
                        .unwrap_or(*particle_scale_sublayers.last().unwrap_or(&1.0));
                    let fraction = fractions[layer_index];
                    summed_variance += (scale / fraction)
                        * absolute_density
                        * (layer_max_absolute[layer_index]
                            - grain_uniformity[channel] * absolute_density);
                }
                peak_variance = peak_variance.max(summed_variance);
            }
            let particle_area = (rms_granularity[channel] / 1000.0).powi(2) * APERTURE_AREA_UM2
                / peak_variance.max(1e-9);
            let mut channel_grain = vec![ZERO; width * height];

            for layer_index in 0..n_layers {
                let scale = particle_scale_sublayers
                    .get(layer_index)
                    .copied()
                    .unwrap_or(*particle_scale_sublayers.last().unwrap_or(&1.0));
                let particles_per_pixel =
                    pixel_area * fractions[layer_index] / (particle_area * scale);
                let mut layer_density = Vec::with_capacity(width * height);
                for pixel in density_cmy.pixels() {
                    let target = pixel[channel] as f64;
                    let value = interp_layer_density(
                        target,
                        channel,
                        density_curves,
                        &curves[layer_index],
                        positive_film,
                    ) + layer_min[layer_index];
                    layer_density.push(from_f64(value));
                }

                let seed = (channel as u64) + (layer_index as u64) * 10;
                let layer = layer_particle_model(
                    &layer_density,
                    density_cmy.width,
                    density_cmy.height,
                    layer_max_absolute[layer_index],
                    particles_per_pixel,
                    grain_uniformity[channel],
                    seed,
                    blur_dye_clouds_um,
                );
                for (sum, value) in channel_grain.iter_mut().zip(layer) {
                    *sum += value;
                }
            }

            (channel, channel_grain)
        })
        .collect();

    let mut result = ImageBuf::new(density_cmy.width, density_cmy.height);
    for (channel, channel_grain) in channels {
        result.write_channel(channel, &channel_grain);
    }

    if grain_blur > 0.0 {
        result = backend.gaussian_blur(&result, grain_blur);
    }
    if mult_usm_sigma > 0.0 && mult_usm_amount > 0.0 {
        let blurred = backend.gaussian_blur(&result, mult_usm_sigma as f32);
        let mut sums_before = [0.0; 3];
        let mut sums_after = [0.0; 3];
        for (index, pixel) in result.data.iter_mut().enumerate() {
            let channel = index % 3;
            let density = (*pixel as f64).max(0.0);
            let blur = (blurred.data[index] as f64).max(1e-6);
            sums_before[channel] += density;
            let sharpened = if density <= 0.0 {
                0.0
            } else {
                density.max(1e-6).powf(1.0 + mult_usm_amount) / blur.powf(mult_usm_amount)
            };
            *pixel = from_f64(sharpened);
            sums_after[channel] += sharpened;
        }
        for (index, pixel) in result.data.iter_mut().enumerate() {
            let channel = index % 3;
            let scale = if sums_after[channel] > 0.0 {
                sums_before[channel] / sums_after[channel]
            } else {
                1.0
            };
            *pixel = from_f64(*pixel as f64 * scale - density_min[channel]);
        }
    } else {
        for (index, pixel) in result.data.iter_mut().enumerate() {
            *pixel = from_f64(*pixel as f64 - density_min[index % 3]);
        }
    }

    result
}

fn spektrafilm_model_curves_by_layer(
    total_curves: &[[f64; 3]],
    layer_curves: &[Vec<Vec<f64>>],
) -> Vec<Vec<[f64; 3]>> {
    let n_layers = layer_curves.iter().map(Vec::len).max().unwrap_or(0);
    if n_layers == 0 {
        return vec![total_curves.to_vec()];
    }

    (0..n_layers)
        .map(|layer_index| {
            layer_curves
                .iter()
                .map(|exposure| {
                    let layer = exposure.get(layer_index);
                    [
                        layer
                            .and_then(|values| values.first())
                            .copied()
                            .unwrap_or(0.0),
                        layer
                            .and_then(|values| values.get(1))
                            .copied()
                            .unwrap_or(0.0),
                        layer
                            .and_then(|values| values.get(2))
                            .copied()
                            .unwrap_or(0.0),
                    ]
                })
                .collect()
        })
        .collect()
}

fn interp_layer_density(
    density: f64,
    channel: usize,
    total_curves: &[[f64; 3]],
    layer_curve: &[[f64; 3]],
    positive_film: bool,
) -> f64 {
    if total_curves.is_empty() || layer_curve.is_empty() {
        return 0.0;
    }
    let query = if positive_film { -density } else { density };
    if query.is_nan() {
        return f64::NAN;
    }
    let get_x = |index: usize| {
        let value = total_curves[index.min(total_curves.len() - 1)][channel];
        if positive_film {
            -value
        } else {
            value
        }
    };
    if query <= get_x(0) {
        return layer_curve[0][channel];
    }
    let last = total_curves.len() - 1;
    if query >= get_x(last) {
        return layer_curve[layer_curve.len() - 1][channel];
    }

    // Match numpy.searchsorted(..., side="right") used by Python's
    // fast_interp, but avoid a linear scan of the exposure curve per pixel.
    let mut lower = 0;
    let mut upper = total_curves.len();
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if get_x(middle) <= query {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    let upper = lower;
    let lower = upper - 1;
    let x0 = get_x(lower);
    let x1 = get_x(upper);
    let y0 = layer_curve[lower.min(layer_curve.len() - 1)][channel];
    let y1 = layer_curve[upper.min(layer_curve.len() - 1)][channel];
    if (x1 - x0).abs() <= f64::EPSILON {
        y0
    } else {
        y0 + (y1 - y0) * ((query - x0) / (x1 - x0)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod rms_tests {
    use super::interp_layer_density;

    #[test]
    fn layer_density_interpolation_matches_negative_and_positive_curve_order() {
        let negative_total = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]];
        let negative_layer = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4]];
        assert!(
            (interp_layer_density(0.25, 0, &negative_total, &negative_layer, false) - 0.1).abs()
                < 1e-12
        );

        let positive_total = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]];
        let positive_layer = [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7]];
        assert!(
            (interp_layer_density(0.75, 0, &positive_total, &positive_layer, true) - 0.15).abs()
                < 1e-12
        );
    }

    #[test]
    fn layer_density_interpolation_uses_rightmost_duplicate_curve_point() {
        let total = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ];
        let layer = [
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2],
            [0.6, 0.6, 0.6],
            [1.0, 1.0, 1.0],
        ];
        assert_eq!(interp_layer_density(0.5, 0, &total, &layer, false), 0.6);
    }
}
