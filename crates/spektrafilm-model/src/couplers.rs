// DIR (Development Inhibitor Release) coupler model.
// Handles same-layer and inter-layer inhibition with spatial diffusion.

use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::from_f64;
use rayon::prelude::*;

/// DIR couplers matrix [3][3]. Row = donor layer, Column = receiver layer.
pub fn compute_dir_couplers_matrix(
    gamma_samelayer: [f64; 3],
    gamma_r_to_gb: [f64; 2],
    gamma_g_to_rb: [f64; 2],
    gamma_b_to_rg: [f64; 2],
    inhibition_samelayer: f64,
    inhibition_interlayer: f64,
) -> [[f64; 3]; 3] {
    let mut m = [[0.0f64; 3]; 3];
    m[0][0] = gamma_samelayer[0] * inhibition_samelayer;
    m[1][1] = gamma_samelayer[1] * inhibition_samelayer;
    m[2][2] = gamma_samelayer[2] * inhibition_samelayer;
    m[0][1] = gamma_r_to_gb[0] * inhibition_interlayer;
    m[0][2] = gamma_r_to_gb[1] * inhibition_interlayer;
    m[1][0] = gamma_g_to_rb[0] * inhibition_interlayer;
    m[1][2] = gamma_g_to_rb[1] * inhibition_interlayer;
    m[2][0] = gamma_b_to_rg[0] * inhibition_interlayer;
    m[2][1] = gamma_b_to_rg[1] * inhibition_interlayer;
    m
}

/// Apply exposure correction from DIR couplers.
///
/// density_cmy contributes inhibitor to other layers through the couplers matrix.
/// The inhibitor diffuses spatially (Gaussian + exponential tail).
#[allow(clippy::too_many_arguments)]
pub fn compute_exposure_correction(
    log_raw: &ImageBuf,
    density_cmy: &ImageBuf,
    density_max: [f64; 3],
    couplers_matrix: &[[f64; 3]; 3],
    diffusion_size_pixel: f32,
    diffusion_tail_pixel: f32,
    diffusion_tail_weight: f64,
    positive: bool,
    backend: &dyn ComputeBackend,
    donor_k: [f64; 3],
    donor_ref: [f64; 3],
) -> ImageBuf {
    let mut density_silver = density_cmy.clone();

    if positive {
        let dmax = [
            from_f64(density_max[0]),
            from_f64(density_max[1]),
            from_f64(density_max[2]),
        ];
        density_silver.par_pixels_mut().for_each(|px| {
            for c in 0..3 {
                px[c] = dmax[c] - px[c];
            }
        });
    }

    // log_raw_correction = einsum('ijk, km->ijm', density_silver, couplers_matrix)
    let cm: [[spektrafilm_math::precision::Scalar; 3]; 3] = [
        [
            from_f64(couplers_matrix[0][0]),
            from_f64(couplers_matrix[0][1]),
            from_f64(couplers_matrix[0][2]),
        ],
        [
            from_f64(couplers_matrix[1][0]),
            from_f64(couplers_matrix[1][1]),
            from_f64(couplers_matrix[1][2]),
        ],
        [
            from_f64(couplers_matrix[2][0]),
            from_f64(couplers_matrix[2][1]),
            from_f64(couplers_matrix[2][2]),
        ],
    ];
    let mut correction = ImageBuf::new(log_raw.width, log_raw.height);
    let mut donor_image = ImageBuf::new(log_raw.width, log_raw.height);
    donor_image
        .data
        .par_chunks_exact_mut(3)
        .zip(density_silver.data.par_chunks_exact(3))
        .for_each(|(out, px)| {
            for c in 0..3 {
                out[c] = px[c] * from_f64(donor_k[c] + donor_ref[c])
                    / (from_f64(donor_k[c]) + px[c]);
            }
        });
    correction
        .data
        .par_chunks_exact_mut(3)
        .zip(donor_image.data.par_chunks_exact(3))
        .for_each(|(out, donor_px)| {
            for m in 0..3 {
                out[m] = 0.0.into();
            }
            for c in 0..3 {
                for m in 0..3 {
                    // Match Python's einsum('ijk,km->ijm', donor, matrix):
                    // donor channel c is multiplied by matrix[c][m].
                    out[m] += donor_px[c] * cm[c][m];
                }
            }
        });
    tracing::info!(
        matrix = ?couplers_matrix,
        donor_k = ?donor_k,
        donor_ref = ?donor_ref,
        donor_first = ?donor_image.data.get(0..3).map(|px| [px[0] as f64, px[1] as f64, px[2] as f64]),
        correction_first = ?correction.data.get(0..3).map(|px| [px[0] as f64, px[1] as f64, px[2] as f64]),
        "DIR CPU f64 matrix diagnostic"
    );

    if diffusion_size_pixel > 0.0 {
        // Python: `(1 - w) * fast_gaussian_filter(corr, σ_size)
        //         + w * fast_exponential_filter(corr, σ_tail)`.
        // The tail kernel is EXPONENTIAL (3-Gaussian mixture), not
        // another Gaussian — using two Gaussians here was a structural
        // diff vs Python.
        use spektrafilm_math::gaussian::{exponential_filter_channel, gaussian_blur_channel};
        let w_img = correction.width;
        let h_img = correction.height;
        let n_pix = (w_img as usize) * (h_img as usize);
        let w = from_f64(diffusion_tail_weight);
        let one = from_f64(1.0);
        let mut blended_channels: [Vec<spektrafilm_math::precision::Scalar>; 3] =
            [vec![one; n_pix], vec![one; n_pix], vec![one; n_pix]];
        for c in 0..3 {
            let ch = correction.extract_channel(c);
            let g = gaussian_blur_channel(&ch, w_img, h_img, diffusion_size_pixel);
            let t = exponential_filter_channel(&ch, w_img, h_img, diffusion_tail_pixel);
            for i in 0..n_pix {
                blended_channels[c][i] = (one - w) * g[i] + w * t[i];
            }
        }
        for c in 0..3 {
            correction.write_channel(c, &blended_channels[c]);
        }
    }

    let mut result = log_raw.clone();
    result.data.par_iter_mut().zip(correction.data.par_iter()).for_each(|(r, c)| {
        *r -= c;
    });
    result
}

/// Full DIR coupler density correction pipeline.
///
/// Port of Python `apply_density_correction_dir_couplers`.
#[allow(clippy::too_many_arguments)]
pub fn apply_density_correction(
    density_cmy: &ImageBuf,
    log_raw: &ImageBuf,
    pixel_size_um: f32,
    log_exposure: &[f64],
    density_curves: &[[f64; 3]],
    couplers_matrix: &[[f64; 3]; 3],
    amount: f64,
    diffusion_size_um: f64,
    diffusion_tail_um: f64,
    diffusion_tail_weight: f64,
    positive: bool,
    gamma_factor: f64,
    backend: &dyn ComputeBackend,
    langmuir_donor_k_rgb: [f64; 3],
) -> ImageBuf {
    let mut matrix_scaled = *couplers_matrix;
    for row in &mut matrix_scaled {
        for v in row.iter_mut() {
            *v *= amount;
        }
    }

    let mut norm_curves = density_curves.to_vec();
    let mins = [
        norm_curves.iter().map(|r| r[0]).fold(f64::INFINITY, f64::min),
        norm_curves.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min),
        norm_curves.iter().map(|r| r[2]).fold(f64::INFINITY, f64::min),
    ];
    for row in &mut norm_curves { for c in 0..3 { row[c] -= mins[c]; } }
    let density_max = [
        norm_curves.iter().map(|r| r[0]).fold(0.0, f64::max),
        norm_curves.iter().map(|r| r[1]).fold(0.0, f64::max),
        norm_curves.iter().map(|r| r[2]).fold(0.0, f64::max),
    ];
    let donor_k = [langmuir_donor_k_rgb[0] * density_max[0], langmuir_donor_k_rgb[1] * density_max[1], langmuir_donor_k_rgb[2] * density_max[2]];
    let donor_ref = [density_max[0] * 0.5, density_max[1] * 0.5, density_max[2] * 0.5];
    let curve_count = norm_curves.len();
    let curve_min = [
        norm_curves
            .iter()
            .map(|row| row[0])
            .fold(f64::INFINITY, f64::min),
        norm_curves
            .iter()
            .map(|row| row[1])
            .fold(f64::INFINITY, f64::min),
        norm_curves
            .iter()
            .map(|row| row[2])
            .fold(f64::INFINITY, f64::min),
    ];
    let sample_indices: Vec<usize> = if curve_count == 0 {
        Vec::new()
    } else {
        [
            0,
            curve_count / 4,
            curve_count / 2,
            (3 * curve_count) / 4,
            curve_count - 1,
        ]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
    };
    let curve_samples: Vec<[f64; 3]> = sample_indices
        .iter()
        .map(|&index| norm_curves[index])
        .collect();
    tracing::info!(
        curve_count,
        curve_min = ?curve_min,
        curve_max = ?density_max,
        curve_sample_indices = ?sample_indices,
        curve_samples = ?curve_samples,
        density_max = ?density_max,
        donor_k = ?donor_k,
        donor_ref = ?donor_ref,
        "DIR normalized density curve diagnostics"
    );
    let density_curves_0 = compute_curves_before_dir(&norm_curves, log_exposure, &matrix_scaled, positive, donor_k, donor_ref);

    let diffusion_size_px = (diffusion_size_um / pixel_size_um as f64) as f32;
    let diffusion_tail_px = (diffusion_tail_um / pixel_size_um as f64) as f32;

    let log_raw_corrected = compute_exposure_correction(
        log_raw,
        density_cmy,
        density_max,
        &matrix_scaled,
        diffusion_size_px,
        diffusion_tail_px,
        diffusion_tail_weight,
        positive,
        backend,
        donor_k,
        donor_ref,
    );

    backend.density_curve_interp(
        &log_raw_corrected,
        log_exposure,
        &density_curves_0,
        gamma_factor,
    )
}

/// Compute density curves before DIR coupler effects.
pub fn compute_curves_before_dir(
    density_curves: &[[f64; 3]],
    log_exposure: &[f64],
    couplers_matrix: &[[f64; 3]; 3],
    positive: bool,
    donor_k: [f64; 3],
    donor_ref: [f64; 3],
) -> Vec<[f64; 3]> {
    let k = density_curves.len();
    let mut dc_silver = density_curves.to_vec();

    if positive {
        let max_d = [
            density_curves.iter().map(|r| r[0]).fold(0.0, f64::max),
            density_curves.iter().map(|r| r[1]).fold(0.0, f64::max),
            density_curves.iter().map(|r| r[2]).fold(0.0, f64::max),
        ];
        for row in &mut dc_silver {
            for c in 0..3 {
                row[c] = max_d[c] - row[c];
            }
        }
    }
    for row in &mut dc_silver {
        for c in 0..3 {
            row[c] = row[c] * (donor_k[c] + donor_ref[c]) / (donor_k[c] + row[c]);
        }
    }

    // couplers_amount = dc_silver @ couplers_matrix
    let mut couplers_amount = vec![[0.0f64; 3]; k];
    for j in 0..k {
        for m in 0..3 {
            couplers_amount[j][m] = dc_silver[j][0] * couplers_matrix[0][m]
                + dc_silver[j][1] * couplers_matrix[1][m]
                + dc_silver[j][2] * couplers_matrix[2][m];
        }
    }

    // log_exposure_0 = log_exposure - couplers_amount
    // Then re-interpolate density curves on the shifted axis
    let mut corrected = vec![[0.0f64; 3]; k];
    for c in 0..3 {
        let le_shifted: Vec<f64> = (0..k)
            .map(|j| log_exposure[j] - couplers_amount[j][c])
            .collect();
        let dc_col: Vec<f64> = density_curves.iter().map(|row| row[c]).collect();
        for j in 0..k {
            corrected[j][c] =
                interp_1d_f64(&le_shifted, &dc_col, log_exposure[j]);
        }
    }

    corrected
}

fn interp_1d_f64(x: &[f64], y: &[f64], xq: f64) -> f64 {
    if x.is_empty() { return 0.0; }
    if xq <= x[0] { return y[0]; }
    if xq >= x[x.len() - 1] { return y[y.len() - 1]; }
    let mut hi = x.partition_point(|&v| v < xq);
    if hi == 0 { hi = 1; }
    let lo = hi - 1;
    let span = x[hi] - x[lo];
    if span == 0.0 { y[lo] } else { y[lo] + (y[hi] - y[lo]) * (xq - x[lo]) / span }
}
