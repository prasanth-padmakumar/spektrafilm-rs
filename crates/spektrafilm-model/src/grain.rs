// Stochastic grain generation.
// Poisson-binomial particle model with dye cloud blur and lognormal micro-structure.

use rayon::prelude::*;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::gaussian;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{from_f64, Scalar, ZERO};
use std::time::Instant;

fn stage_timings_enabled() -> bool {
    std::env::var_os("SPEKTRAFILM_STAGE_TIMINGS").is_some()
        || std::env::var_os("SPEKTRAFILM_PARITY_STATS").is_some()
}

fn print_stage_timing(enabled: bool, stage: &str, start: Instant) {
    if enabled {
        eprintln!("stage {stage}: {} ms", start.elapsed().as_millis());
    }
}

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
    let stage_timings = stage_timings_enabled();
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
            let t_ch = Instant::now();
            let particle_area = agx_particle_area_um2 * agx_particle_scale[ch];
            let mut n_particles = pixel_area / particle_area;
            if n_sub_layers > 1 {
                n_particles /= n_sub_layers as f64;
            }

            // Add density_min to input (kept in Scalar precision)
            let dmin_s = from_f64(density_min[ch]);
            let t = Instant::now();
            let density_ch: Vec<Scalar> = density_cmy.pixels().map(|px| px[ch] + dmin_s).collect();
            print_stage_timing(stage_timings, "grain.extract_channel", t);

            let mut grain_sum = vec![ZERO; density_ch.len()];

            for sl in 0..n_sub_layers {
                let t = Instant::now();
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
                print_stage_timing(stage_timings, "grain.layer_particle_model", t);
                for (s, &v) in grain_sum.iter_mut().zip(g.iter()) {
                    *s += v;
                }
            }

            // Average sub-layers
            let scale = from_f64(1.0 / n_sub_layers as f64);
            for v in &mut grain_sum {
                *v = *v * scale - dmin_s;
            }

            print_stage_timing(stage_timings, "grain.channel_total", t_ch);
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
        let t = Instant::now();
        out = backend.gaussian_blur(&out, grain_blur);
        print_stage_timing(stage_timings, "grain.final_blur", t);
    }

    out
}

/// High-quality inhomogeneous Boolean grain field. Grain centres are sampled
/// from a spatial Poisson process, radii are lognormal, and each observation
/// point is tested through a small Gaussian aperture. The counter-derived
/// hash makes the result deterministic and thread-safe.
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
    monochrome: bool,
    _backend: &dyn ComputeBackend,
) -> ImageBuf {
    if amount == 0.0 {
        return density.clone();
    }
    let width = density.width as usize;
    let height = density.height as usize;
    let samples = samples.max(1) as usize;
    let pixel_um = pixel_um as f64;
    let mean_radius_um = mean_radius_um.max(1e-3);
    let radius_stddev_ratio = radius_stddev_ratio.max(1e-3);
    let sigma_r = (1.0 + radius_stddev_ratio.powi(2)).ln().sqrt();
    let mu_r = mean_radius_um.ln() - 0.5 * sigma_r.powi(2);
    let cell_um = (cell_ratio.max(1e-3) * mean_radius_um).max(1e-3);
    let max_grains = max_grains.max(1);
    let aperture_sigma_um = 0.5 * pixel_um;

    // Python clamps the lognormal tail at the configured quantile. Rust's
    // current public settings do not expose that control; use the same
    // practical default quantile (0.999), approximately 3.09 sigma.
    let max_radius_um = (mu_r + 3.09 * sigma_r).exp();
    let rings = 1 + (max_radius_um / cell_um) as isize;
    let lambda_cell_density = std::f64::consts::LN_10
        / (std::f64::consts::PI * mean_radius_um.powi(2) * (1.0 + radius_stddev_ratio.powi(2)));
    let channel_count = if monochrome { 1 } else { 3 };
    let mut out = ImageBuf::new(density.width, density.height);

    for channel in 0..channel_count {
        let dmin = density_min[channel];
        // density_max passed by the film pipeline is the normalized curve
        // maximum; Python adds density_min before calibration and p-map use.
        let dmax = (density_max[channel] + dmin).max(1e-9);
        let lambda_area = dmax * lambda_cell_density;
        let mut density_channel = Vec::with_capacity(width * height);
        for pixel in density.pixels() {
            density_channel.push(((pixel[channel] as f64 + dmin) / dmax).clamp(1e-6, 1.0 - 1e-6));
        }

        let mut grain_channel = vec![ZERO; width * height];
        const ROW_BLOCK: usize = 8;
        for row_start in (0..height).step_by(ROW_BLOCK) {
            let row_end = (row_start + ROW_BLOCK).min(height);
            let max_sample_offset = 8.6 * aperture_sigma_um;
            let min_cell_x =
                ((0.5 * pixel_um - max_sample_offset) / cell_um).floor() as isize - rings;
            let max_cell_x = (((width as f64 - 0.5) * pixel_um + max_sample_offset) / cell_um)
                .floor() as isize
                + rings;
            let min_cell_y = (((row_start as f64 + 0.5) * pixel_um - max_sample_offset) / cell_um)
                .floor() as isize
                - rings;
            let max_cell_y = (((row_end as f64 - 0.5) * pixel_um + max_sample_offset) / cell_um)
                .floor() as isize
                + rings;
            let grid_width = (max_cell_x - min_cell_x + 1) as usize;
            let grid_height = (max_cell_y - min_cell_y + 1) as usize;
            let mut cell_ranges = Vec::with_capacity(grid_width * grid_height);
            let mut grains = Vec::new();

            for cell_y in min_cell_y..=max_cell_y {
                for cell_x in min_cell_x..=max_cell_x {
                    let start = grains.len();
                    let count = monte_carlo_cell_poisson(
                        cell_x,
                        cell_y,
                        lambda_area * cell_um * cell_um,
                        max_grains,
                        11 + channel as u64,
                    );
                    for grain_index in 0..count {
                        let (x_uniform, y_uniform) = monte_carlo_hash_pair(
                            cell_x as u64,
                            cell_y as u64,
                            grain_index as u64 * 8,
                            11 + channel as u64,
                        );
                        let grain_x = (cell_x as f64 + x_uniform) * cell_um;
                        let grain_y = (cell_y as f64 + y_uniform) * cell_um;
                        let (radius_u1, radius_u2) = monte_carlo_hash_pair(
                            cell_x as u64,
                            cell_y as u64,
                            grain_index as u64 * 8 + 2,
                            11 + channel as u64,
                        );
                        let normal = (-2.0 * radius_u1.ln()).sqrt()
                            * (2.0 * std::f64::consts::PI * radius_u2).cos();
                        let radius = (mu_r + sigma_r * normal).exp().min(max_radius_um);
                        let pixel_x = ((grain_x / pixel_um) as isize)
                            .clamp(0, width.saturating_sub(1) as isize)
                            as usize;
                        let pixel_y = ((grain_y / pixel_um) as isize)
                            .clamp(0, height.saturating_sub(1) as isize)
                            as usize;
                        let probability = density_channel[pixel_y * width + pixel_x];
                        let (develop_u, _) = monte_carlo_hash_pair(
                            cell_x as u64,
                            cell_y as u64,
                            grain_index as u64 * 8 + 4,
                            11 + channel as u64,
                        );
                        if develop_u < probability {
                            grains.push(MonteCarloGrain {
                                x: grain_x,
                                y: grain_y,
                                radius_squared: radius * radius,
                            });
                        }
                    }
                    cell_ranges.push((start, grains.len()));
                }
            }

            let output_block = &mut grain_channel[row_start * width..row_end * width];
            output_block
                .par_iter_mut()
                .enumerate()
                .for_each(|(local_index, output)| {
                    let y = row_start + local_index / width;
                    let x = local_index % width;
                    let mut covered = 0usize;
                    for sample in 0..samples {
                        let (u1, u2) = monte_carlo_hash_pair(
                            x as u64,
                            y as u64,
                            (sample * 2) as u64,
                            11 + channel as u64 + 7,
                        );
                        let normal_radius = (-2.0 * u1.ln()).sqrt();
                        let angle = 2.0 * std::f64::consts::PI * u2;
                        let sample_x = (x as f64 + 0.5) * pixel_um
                            + normal_radius * angle.cos() * aperture_sigma_um;
                        let sample_y = (y as f64 + 0.5) * pixel_um
                            + normal_radius * angle.sin() * aperture_sigma_um;
                        let sample_cell_x = (sample_x / cell_um).floor() as isize;
                        let sample_cell_y = (sample_y / cell_um).floor() as isize;
                        let mut hit = false;

                        'cells: for dx in -rings..=rings {
                            for dy in -rings..=rings {
                                let cell_x = sample_cell_x + dx;
                                let cell_y = sample_cell_y + dy;
                                let cell_left = cell_x as f64 * cell_um;
                                let cell_top = cell_y as f64 * cell_um;
                                let edge_distance_x = if sample_x < cell_left {
                                    cell_left - sample_x
                                } else if sample_x > cell_left + cell_um {
                                    sample_x - (cell_left + cell_um)
                                } else {
                                    0.0
                                };
                                let edge_distance_y = if sample_y < cell_top {
                                    cell_top - sample_y
                                } else if sample_y > cell_top + cell_um {
                                    sample_y - (cell_top + cell_um)
                                } else {
                                    0.0
                                };
                                if edge_distance_x * edge_distance_x
                                    + edge_distance_y * edge_distance_y
                                    > max_radius_um * max_radius_um
                                {
                                    continue;
                                }
                                let grid_x = (cell_x - min_cell_x) as usize;
                                let grid_y = (cell_y - min_cell_y) as usize;
                                let index = grid_y * grid_width + grid_x;
                                let (start, end) = cell_ranges[index];
                                for grain in &grains[start..end] {
                                    let offset_x = sample_x - grain.x;
                                    let offset_y = sample_y - grain.y;
                                    if offset_x * offset_x + offset_y * offset_y
                                        <= grain.radius_squared
                                    {
                                        hit = true;
                                        break 'cells;
                                    }
                                }
                            }
                        }
                        covered += usize::from(hit);
                    }

                    let index = y * width + x;
                    let original = density_channel_value(density, index, channel);
                    let expected_uncovered = 10.0f64.powf(-(density_channel[index] * dmax));
                    let uncovered_samples = (samples - covered) as f64;
                    // Finite aperture sampling can otherwise report exactly
                    // 100% coverage and turn a modest density into the fixed
                    // epsilon ceiling. Use the analytically calibrated
                    // Poisson coverage as a 16-sample prior (a control
                    // variate) to stabilize low sample counts without
                    // changing the expected coverage.
                    let prior_samples = 16.0;
                    let uncovered = (uncovered_samples + prior_samples * expected_uncovered)
                        / (samples as f64 + prior_samples);
                    let synthesized = -uncovered.max(1e-9).log10() - dmin;
                    *output = from_f64((original + amount * (synthesized - original)).max(-dmin));
                });
        }
        out.write_channel(channel, &grain_channel);
    }
    if monochrome {
        let gray = out.extract_channel(0);
        out.write_channel(1, &gray);
        out.write_channel(2, &gray);
    }
    out
}

#[derive(Clone, Copy)]
struct MonteCarloGrain {
    x: f64,
    y: f64,
    radius_squared: f64,
}

fn density_channel_value(density: &ImageBuf, index: usize, channel: usize) -> f64 {
    density.data[index * 3 + channel] as f64
}

fn monte_carlo_mix64(value: u64) -> u64 {
    let value = (value ^ (value >> 33)).wrapping_mul(0xff51afd7ed558ccd);
    let value = (value ^ (value >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
    value ^ (value >> 33)
}

fn monte_carlo_hash01(a: u64, b: u64, c: u64, seed: u64) -> f64 {
    let hash = monte_carlo_mix64(
        a.wrapping_mul(0x9e3779b97f4a7c15)
            ^ b.wrapping_mul(0xbf58476d1ce4e5b9)
            ^ c.wrapping_mul(0x94d049bb133111eb)
            ^ seed,
    );
    ((hash >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
}

fn monte_carlo_hash_pair(a: u64, b: u64, c: u64, seed: u64) -> (f64, f64) {
    (
        monte_carlo_hash01(a, b, c, seed),
        monte_carlo_hash01(a, b, c.wrapping_add(1), seed),
    )
}

fn monte_carlo_cell_poisson(
    cell_x: isize,
    cell_y: isize,
    lambda: f64,
    max_count: u32,
    seed: u64,
) -> u32 {
    let threshold = (-lambda).exp();
    let mut product = 1.0;
    let mut count = 0;
    while count <= max_count {
        product *= monte_carlo_hash01(cell_x as u64, cell_y as u64, 900_000 + count as u64, seed);
        if product <= threshold {
            break;
        }
        count += 1;
    }
    count.min(max_count)
}

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
    let stage_timings = stage_timings_enabled();
    const APERTURE_AREA_UM2: f64 = std::f64::consts::PI * 24.0 * 24.0;
    let width = density_cmy.width as usize;
    let height = density_cmy.height as usize;
    let pixel_area = (pixel_size_um as f64).powi(2);
    let curves = spektrafilm_model_curves_by_layer(density_curves, density_curves_layers);
    let n_layers = curves.len();

    let channels: Vec<(usize, Vec<Scalar>)> = (0..3)
        .into_par_iter()
        .map(|channel| {
            let channel_start = Instant::now();
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
                let t = Instant::now();
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
                print_stage_timing(stage_timings, "grain.rms.layer_density", t);

                let seed = (channel as u64) + (layer_index as u64) * 10;
                let t = Instant::now();
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
                print_stage_timing(stage_timings, "grain.rms.particle_model", t);
                for (sum, value) in channel_grain.iter_mut().zip(layer) {
                    *sum += value;
                }
            }

            print_stage_timing(stage_timings, "grain.rms.channel", channel_start);
            (channel, channel_grain)
        })
        .collect();

    let mut result = ImageBuf::new(density_cmy.width, density_cmy.height);
    for (channel, channel_grain) in channels {
        result.write_channel(channel, &channel_grain);
    }

    if grain_blur > 0.0 {
        let t = Instant::now();
        result = backend.gaussian_blur(&result, grain_blur);
        print_stage_timing(stage_timings, "grain.rms.blur", t);
    }
    if mult_usm_sigma > 0.0 && mult_usm_amount > 0.0 {
        let t = Instant::now();
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
        print_stage_timing(stage_timings, "grain.rms.mult_usm", t);
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

#[cfg(test)]
mod monte_carlo_tests {
    use super::{apply_grain_monte_carlo, monte_carlo_cell_poisson};
    use spektrafilm_gpu::cpu_backend::CpuBackend;
    use spektrafilm_math::image::ImageBuf;
    use spektrafilm_math::precision::from_f64;

    #[test]
    fn poisson_cell_counts_have_the_requested_mean() {
        let counts: f64 = (0..10_000)
            .map(|x| monte_carlo_cell_poisson(x, x.wrapping_mul(37), 5.0, 32, 11) as f64)
            .sum();
        let mean = counts / 10_000.0;
        assert!((mean - 5.0).abs() < 0.15, "Poisson sample mean was {mean}");
    }

    #[test]
    fn monte_carlo_grain_preserves_mean_density_without_saturation() {
        let width = 64;
        let height = 64;
        let density = ImageBuf::from_data(
            width,
            height,
            (0..width as usize * height as usize)
                .flat_map(|_| [from_f64(0.4); 3])
                .collect(),
        );
        let result = apply_grain_monte_carlo(
            &density,
            3.9167,
            [0.03; 3],
            [1.0; 3],
            1.0,
            16,
            0.6,
            0.4,
            4.0,
            32,
            false,
            &CpuBackend,
        );
        let channel = result.extract_channel(0);
        let mean = channel.iter().map(|&value| value as f64).sum::<f64>() / channel.len() as f64;
        let maximum = channel
            .iter()
            .map(|&value| value as f64)
            .fold(0.0, f64::max);
        assert!(
            (mean - 0.4).abs() < 0.15,
            "mean density drifted from 0.4 to {mean}",
        );
        assert!(
            maximum < 2.0,
            "coverage estimator saturated at density {maximum}",
        );
        assert!(channel
            .iter()
            .all(|value| value.is_finite() && *value >= from_f64(-0.03)),);
        for channel_index in 1..3 {
            let channel = result.extract_channel(channel_index);
            let channel_mean =
                channel.iter().map(|&value| value as f64).sum::<f64>() / channel.len() as f64;
            assert!(
                (channel_mean - mean).abs() < 0.05,
                "neutral input developed a channel cast: means {mean} and {channel_mean}",
            );
        }
        let repeated = apply_grain_monte_carlo(
            &density,
            3.9167,
            [0.03; 3],
            [1.0; 3],
            1.0,
            16,
            0.6,
            0.4,
            4.0,
            32,
            false,
            &CpuBackend,
        );
        assert_eq!(result.data, repeated.data);
    }

    #[test]
    fn zero_amount_keeps_the_input_unchanged() {
        let density = ImageBuf::from_data(2, 2, vec![from_f64(0.2); 12]);
        let result = apply_grain_monte_carlo(
            &density,
            1.0,
            [0.03; 3],
            [1.0; 3],
            0.0,
            16,
            0.6,
            0.4,
            4.0,
            32,
            false,
            &CpuBackend,
        );
        assert_eq!(result.data, density.data);
    }
}
