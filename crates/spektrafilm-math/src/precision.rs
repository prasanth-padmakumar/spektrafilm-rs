/// Precision type alias for the pixel pipeline.
///
/// Default: f32 for GPU speed and memory efficiency.
/// With `--features precision-f64`: f64 for 1:1 parity with Python.
///
/// The calibration chain (Hanatos2025 tc/b, CAT02 adaptation, exposure factor)
/// always uses f64 regardless of this flag.

#[cfg(feature = "precision-f64")]
pub type Scalar = f64;
#[cfg(not(feature = "precision-f64"))]
pub type Scalar = f32;

/// Convert f64 to Scalar (no-op in f64 mode, truncation in f32 mode).
#[inline(always)]
pub fn from_f64(v: f64) -> Scalar {
    v as Scalar
}

/// Convert Scalar to f64 (no-op in f64 mode, promotion in f32 mode).
#[inline(always)]
pub fn to_f64(v: Scalar) -> f64 {
    v as f64
}

/// Convert f32 to Scalar.
#[inline(always)]
pub fn from_f32(v: f32) -> Scalar {
    v as Scalar
}

/// Convert Scalar to f32.
#[inline(always)]
pub fn to_f32(v: Scalar) -> f32 {
    v as f32
}

/// Zero constant.
pub const ZERO: Scalar = 0.0;
/// One constant.
pub const ONE: Scalar = 1.0;

/// Scalar-aware pow(10, x).
#[inline(always)]
pub fn pow10(x: Scalar) -> Scalar {
    from_f64(10.0).powf(x)
}

/// Scalar-aware log10.
#[inline(always)]
pub fn log10(x: Scalar) -> Scalar {
    x.log10()
}

/// Scalar-aware powf.
#[inline(always)]
pub fn powf(base: Scalar, exp: Scalar) -> Scalar {
    base.powf(exp)
}

/// sRGB OETF (linear → gamma) for Scalar.
#[inline(always)]
pub fn srgb_encode(x: Scalar) -> Scalar {
    let threshold = from_f64(0.0031308);
    if x <= threshold {
        from_f64(12.92) * x
    } else {
        from_f64(1.055) * powf(x, from_f64(1.0 / 2.4)) - from_f64(0.055)
    }
}

/// ProPhoto RGB (ROMM RGB) OETF for linear values.
#[inline(always)]
pub fn prophoto_encode(x: Scalar) -> Scalar {
    let threshold = from_f64(1.0 / 512.0);
    if x < threshold {
        from_f64(16.0) * x
    } else {
        powf(x, from_f64(1.0 / 1.8))
    }
}

/// Return the encoding CCTF for an output RGB color space.
pub fn output_cctf_encoder(color_space: &str) -> fn(f64) -> f64 {
    match color_space {
        "ProPhoto RGB" => encode_prophoto_f64,
        "Adobe RGB (1998)" => encode_adobe_rgb_f64,
        "Rec. 2020" | "Rec2020" | "ITU-R BT.2020" => encode_bt2020_f64,
        "ACEScg" | "ACES2065-1" => |value| value,
        _ => encode_srgb_f64,
    }
}

fn encode_srgb_f64(value: f64) -> f64 {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn encode_prophoto_f64(value: f64) -> f64 {
    if value < 1.0 / 512.0 {
        16.0 * value
    } else {
        value.powf(1.0 / 1.8)
    }
}

fn encode_adobe_rgb_f64(value: f64) -> f64 {
    value.signum() * value.abs().powf(1.0 / 2.19921875)
}

fn encode_bt2020_f64(value: f64) -> f64 {
    const ALPHA: f64 = 1.099;
    const BETA: f64 = 0.018;
    if value < BETA {
        4.5 * value
    } else {
        ALPHA * value.powf(0.45) - (ALPHA - 1.0)
    }
}

/// sRGB EOTF (gamma → linear) for Scalar.
#[inline(always)]
pub fn srgb_decode(x: Scalar) -> Scalar {
    let threshold = from_f64(0.04045);
    if x <= threshold {
        x / from_f64(12.92)
    } else {
        powf((x + from_f64(0.055)) / from_f64(1.055), from_f64(2.4))
    }
}

/// Return the decoding CCTF for an input RGB color space.
///
/// The transfer function is applied to encoded RGB values before the RGB→XYZ
/// matrix, matching `colour.RGB_to_XYZ(..., apply_cctf_decoding=True)`.
pub fn input_cctf_decoder(color_space: &str) -> fn(f64) -> f64 {
    match color_space {
        "sRGB" | "Display P3" => decode_srgb_f64,
        "ProPhoto RGB" => decode_prophoto_f64,
        "Adobe RGB (1998)" => decode_adobe_rgb_f64,
        "Rec. 2020" | "Rec2020" | "ITU-R BT.2020" => decode_bt2020_f64,
        "ACEScg" | "ACES2065-1" => |value| value,
        _ => |value| value,
    }
}

fn decode_srgb_f64(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn decode_prophoto_f64(value: f64) -> f64 {
    if value < 1.0 / 32.0 {
        value / 16.0
    } else {
        value.powf(1.8)
    }
}

fn decode_adobe_rgb_f64(value: f64) -> f64 {
    value.signum() * value.abs().powf(2.19921875)
}

fn decode_bt2020_f64(value: f64) -> f64 {
    const ALPHA: f64 = 1.099;
    const BETA: f64 = 0.018;
    if value < 4.5 * BETA {
        value / 4.5
    } else {
        ((value + (ALPHA - 1.0)) / ALPHA).powf(1.0 / 0.45)
    }
}

#[cfg(test)]
mod tests {
    use super::{input_cctf_decoder, output_cctf_encoder, prophoto_encode};
    use crate::precision::from_f64;

    #[test]
    fn prophoto_decode_matches_romm_rgb_cctf() {
        let decode = input_cctf_decoder("ProPhoto RGB");
        for (encoded, expected) in [
            (0.0, 0.0),
            (0.01, 0.000625),
            (1.0 / 32.0, (1.0_f64 / 32.0).powf(1.8)),
            (0.5, 0.2871745887492587),
            (1.0, 1.0),
        ] {
            assert!(
                (decode(encoded) - expected).abs() < 1e-14,
                "encoded={encoded}: got {}, expected {expected}",
                decode(encoded),
            );
        }
    }

    #[test]
    fn prophoto_encode_matches_romm_rgb_cctf() {
        for (linear, expected) in [
            (0.0, 0.0),
            (1.0 / 512.0, 1.0 / 32.0),
            (0.01, 0.01_f64.powf(1.0 / 1.8)),
            (0.5, 0.5_f64.powf(1.0 / 1.8)),
            (1.0, 1.0),
        ] {
            let actual = prophoto_encode(from_f64(linear)) as f64;
            assert!(
                (actual - expected).abs() < 1e-6,
                "linear={linear}: got {actual}, expected {expected}",
            );
        }
    }

    #[test]
    fn output_cctf_encoder_selects_prophoto_and_linear_spaces() {
        let prophoto = output_cctf_encoder("ProPhoto RGB");
        assert!((prophoto(0.5) - 0.5_f64.powf(1.0 / 1.8)).abs() < 1e-14);
        assert!((prophoto(1.0 / 512.0) - 1.0 / 32.0).abs() < 1e-14);
        assert_eq!(output_cctf_encoder("ACES2065-1")(0.5), 0.5);
        assert!((output_cctf_encoder("sRGB")(0.5) - 0.7353569830524495).abs() < 1e-14);
    }

    #[test]
    fn input_cctf_decoder_matches_supported_color_spaces() {
        let srgb = input_cctf_decoder("sRGB");
        let p3 = input_cctf_decoder("Display P3");
        let adobe = input_cctf_decoder("Adobe RGB (1998)");
        let rec2020 = input_cctf_decoder("Rec. 2020");
        let linear = input_cctf_decoder("ACES2065-1");

        assert!((srgb(0.5) - 0.21404114048223255).abs() < 1e-14);
        assert_eq!(p3(0.5), srgb(0.5));
        assert!((adobe(0.5) - 0.21775552814439456).abs() < 1e-14);
        assert!((rec2020(0.5) - 0.25958940050628576).abs() < 1e-14);
        assert_eq!(linear(0.5), 0.5);
    }
}
