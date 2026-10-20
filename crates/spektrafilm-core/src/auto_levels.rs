use rayon::prelude::*;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{from_f64, to_f64, Scalar};

const CLIP_FRACTION: f64 = 0.005;
const NEUTRAL_LUMINANCE_MIN: f64 = 0.1;
const NEUTRAL_LUMINANCE_MAX: f64 = 0.8;
const NEUTRAL_RELATIVE_TOLERANCE: f64 = 0.05;
const NEUTRAL_ABSOLUTE_TOLERANCE: f64 = 0.01;
const HIGHLIGHT_SHOULDER_START: f64 = 0.97;
const HIGHLIGHT_OUTPUT_CEILING: f64 = 0.995;

pub(crate) fn apply(image: &mut ImageBuf, mode: &str) {
    if mode == "Off" {
        return;
    }
    if mode != "Enhance Monochromatic Contrast"
        && mode != "Per Channel Contrast with Snap Neutral Midtones"
    {
        tracing::error!(mode, "unsupported auto-levels mode; leaving scan unchanged");
        return;
    }
    if image.data.is_empty() || image.data.len() % 3 != 0 {
        tracing::error!(
            width = image.width,
            height = image.height,
            data_len = image.data.len(),
            "auto-levels requires a non-empty RGB image; leaving scan unchanged"
        );
        return;
    }

    let (low, high) = if mode == "Enhance Monochromatic Contrast" {
        let mut values: Vec<f64> = image
            .data
            .iter()
            .map(|&value| to_f64(value))
            .filter(|value| !value.is_nan())
            .collect();
        if values.is_empty() {
            tracing::error!("auto-levels found no non-NaN samples; leaving scan unchanged");
            return;
        }
        let low = percentile(&mut values, CLIP_FRACTION);
        let high = percentile(&mut values, 1.0 - CLIP_FRACTION);
        ([low; 3], [high; 3])
    } else {
        let Some(bounds) = channel_percentile_bounds(&image.data) else {
            return;
        };
        bounds
    };

    let neutral_sums = if mode == "Per Channel Contrast with Snap Neutral Midtones" {
        image
            .data
            .par_chunks_exact(3)
            .fold(
                || ([0.0_f64; 3], 0.0_f64),
                |(mut sums, mut count), pixel| {
                    let source = [to_f64(pixel[0]), to_f64(pixel[1]), to_f64(pixel[2])];
                    let mean = (source[0] + source[1] + source[2]) / 3.0;
                    let span = source[0].max(source[1]).max(source[2])
                        - source[0].min(source[1]).min(source[2]);
                    let tolerance =
                        NEUTRAL_ABSOLUTE_TOLERANCE.max(mean.abs() * NEUTRAL_RELATIVE_TOLERANCE);
                    if mean >= NEUTRAL_LUMINANCE_MIN
                        && mean <= NEUTRAL_LUMINANCE_MAX
                        && span <= tolerance
                    {
                        for channel in 0..3 {
                            sums[channel] += stretch(source[channel], low[channel], high[channel]);
                        }
                        count += 1.0;
                    }
                    (sums, count)
                },
            )
            .reduce(
                || ([0.0_f64; 3], 0.0_f64),
                |(mut a, count_a), (b, count_b)| {
                    for channel in 0..3 {
                        a[channel] += b[channel];
                    }
                    (a, count_a + count_b)
                },
            )
    } else {
        ([0.0; 3], 0.0)
    };

    let gains = if neutral_sums.1 > 0.0 {
        let means = neutral_sums.0.map(|sum| sum / neutral_sums.1);
        if means.iter().any(|mean| *mean <= 1e-12) {
            [1.0; 3]
        } else {
            let target = (means[0] + means[1] + means[2]) / 3.0;
            means.map(|mean| target / mean)
        }
    } else {
        [1.0; 3]
    };

    image.data.par_chunks_exact_mut(3).for_each(|pixel| {
        for channel in 0..3 {
            let value =
                stretch(to_f64(pixel[channel]), low[channel], high[channel]) * gains[channel];
            let value = if value < 0.0 { 0.0 } else { value };
            let value = roll_off_highlights(value);
            pixel[channel] = from_f64(value);
        }
    });
}

fn channel_percentile_bounds(data: &[Scalar]) -> Option<([f64; 3], [f64; 3])> {
    let mut channels: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(data.len() / 3));
    for pixel in data.chunks_exact(3) {
        for channel in 0..3 {
            let value = to_f64(pixel[channel]);
            if !value.is_nan() {
                channels[channel].push(value);
            }
        }
    }

    let mut low = [f64::NAN; 3];
    let mut high = [f64::NAN; 3];
    for channel in 0..3 {
        if channels[channel].is_empty() {
            tracing::error!(
                channel,
                "auto-levels found no non-NaN samples for channel; leaving scan unchanged"
            );
            return None;
        }
        low[channel] = percentile(&mut channels[channel], CLIP_FRACTION);
        high[channel] = percentile(&mut channels[channel], 1.0 - CLIP_FRACTION);
    }
    Some((low, high))
}

fn percentile(values: &mut [f64], quantile: f64) -> f64 {
    let position = quantile * (values.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let fraction = position - lower_index as f64;
    let compare = f64::total_cmp;
    let lower = *values.select_nth_unstable_by(lower_index, compare).1;
    if upper_index == lower_index {
        return lower;
    }
    let upper = *values[lower_index + 1..]
        .select_nth_unstable_by(upper_index - lower_index - 1, compare)
        .1;
    lower + (upper - lower) * fraction
}

fn stretch(value: f64, low: f64, high: f64) -> f64 {
    let span = high - low;
    if span > 1e-12 {
        (value - low) / span
    } else {
        value
    }
}

fn roll_off_highlights(value: f64) -> f64 {
    if value <= HIGHLIGHT_SHOULDER_START {
        return value;
    }
    let shoulder_range = HIGHLIGHT_OUTPUT_CEILING - HIGHLIGHT_SHOULDER_START;
    HIGHLIGHT_SHOULDER_START
        + shoulder_range * (-(-(value - HIGHLIGHT_SHOULDER_START) / shoulder_range).exp_m1())
}

#[cfg(test)]
mod tests {
    use super::{apply, percentile};
    use spektrafilm_math::image::ImageBuf;
    use spektrafilm_math::precision::from_f64;

    #[test]
    fn percentile_uses_linear_interpolation() {
        let mut values = [4.0, 1.0, 3.0, 2.0];
        assert!((percentile(&mut values, 0.25) - 1.75).abs() < 1e-12);
    }

    #[test]
    fn off_preserves_values_and_per_channel_mode_levels_channels() {
        let mut off = ImageBuf::from_data(
            4,
            1,
            [0.1, 0.2, 0.3, 0.2, 0.4, 0.6, 0.3, 0.6, 0.9, 0.4, 0.8, 1.2]
                .map(from_f64)
                .to_vec(),
        );
        let original = off.data.clone();
        apply(&mut off, "Off");
        assert_eq!(off.data, original);

        let mut image = ImageBuf::from_data(
            4,
            1,
            [0.1, 0.2, 0.3, 0.2, 0.4, 0.6, 0.3, 0.6, 0.9, 0.4, 0.8, 1.2]
                .map(from_f64)
                .to_vec(),
        );
        apply(&mut image, "Enhance Monochromatic Contrast");
        let first = image.pixels().next().unwrap();
        let last = image.pixels().last().unwrap();
        assert!((first[0] as f64).abs() < 1e-6);
        assert!((last[2] as f64) > 0.97);
        assert!((last[2] as f64) < 0.995);
    }

    #[test]
    fn per_channel_mode_snaps_neutral_midtones_and_rolls_off_highlights() {
        let mut data = Vec::with_capacity(20 * 20 * 3);
        for index in 0..400 {
            let value = 0.2 + 0.5 * index as f64 / 399.0;
            data.extend([from_f64(value); 3]);
        }
        data[0..3].copy_from_slice(&[0.95, 0.15, 0.1].map(from_f64));
        data[3..6].copy_from_slice(&[0.1, 0.95, 0.15].map(from_f64));
        data[6..9].copy_from_slice(&[0.15, 0.1, 0.95].map(from_f64));
        let source = data.clone();
        let mut image = ImageBuf::from_data(20, 20, data);

        apply(
            &mut image,
            "Per Channel Contrast with Snap Neutral Midtones",
        );

        let mut sums = [0.0_f64; 3];
        let mut count = 0.0;
        let mut output_max = f64::NEG_INFINITY;
        for (input, output) in source.chunks_exact(3).zip(image.data.chunks_exact(3)) {
            let input = [input[0] as f64, input[1] as f64, input[2] as f64];
            let mean = (input[0] + input[1] + input[2]) / 3.0;
            let span = input[0].max(input[1]).max(input[2]) - input[0].min(input[1]).min(input[2]);
            if (0.1..=0.8).contains(&mean) && span <= 0.01 {
                for channel in 0..3 {
                    sums[channel] += output[channel] as f64;
                }
                count += 1.0;
            }
            output_max = output_max.max(
                (output[0] as f64)
                    .max(output[1] as f64)
                    .max(output[2] as f64),
            );
        }
        let means = sums.map(|sum| sum / count);
        assert!(means[0].max(means[1]).max(means[2]) - means[0].min(means[1]).min(means[2]) < 1e-5);
        assert!(output_max > 0.98 && output_max < 1.0);
    }
}
