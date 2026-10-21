//! Output gamut compression — port of upstream spektrafilm's
//! `utils/gamut_compression.compress_rgb` (oklch / oklrab / cam16ucs / aces_rgc).
//!
//! The scan produces linear RGB in its configured output color space. Colors
//! outside the output cube would otherwise be hard-clipped to `[0, 1]` at
//! encode. The `oklch` compressor instead reduces *perceptual chroma*
//! smoothly: linear RGB → XYZ → OkLab → OkLch `(L, C, h)`, look up the
//! max in-gamut chroma `C_max(L, h)` for the output cube, apply a Reinhard
//! knee to `C / C_max`, and reconstruct — preserving OkLab hue and
//! lightness. A one-sided lightness compression first rolls super-bright
//! highlights (`L > white`) back into range.
//!
//! `oklch`, `oklrab`, and `cam16ucs` use target-gamut tables for sRGB and
//! ProPhoto RGB; `aces_rgc` is per-channel. `jzazbz` remains a follow-up.
//! Gated behind
//! `io.output_gamut_compress` (default off → no change).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::params::OutputGamutCompressParams;

// sRGB linear RGB ↔ XYZ (colour-science's rounded sRGB matrices, D65; no CAT
// since the conversion uses the colourspace's own whitepoint).
const SRGB_RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
];
const SRGB_XYZ_TO_RGB: [[f64; 3]; 3] = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.204, 1.057],
];
// sRGB (D65) whitepoint as XYZ at Y=1, from the exact xy = (0.3127, 0.329)
// — matches colour's `_xy_to_xyz_unit_y(whitepoint)`.
const SRGB_WHITE_XYZ: [f64; 3] = [0.3127 / 0.329, 1.0, (1.0 - 0.3127 - 0.329) / 0.329];
// colour-science's ProPhoto RGB (linear, D50) matrices.
const PROPHOTO_RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.7977, 0.1352, 0.0313],
    [0.2880, 0.7119, 0.0001],
    [0.0, 0.0, 0.8249],
];
const PROPHOTO_WHITE_XYZ: [f64; 3] = [0.3457 / 0.3585, 1.0, (1.0 - 0.3457 - 0.3585) / 0.3585];
const XYZ_TO_PROPHOTO_RGB: [[f64; 3]; 3] = spektrafilm_math::colorspace::XYZ_TO_PROPHOTO_F64;
// OkLab (Ottosson) matrices, exactly as colour-science stores them.
const M1_XYZ_TO_LMS: [[f64; 3]; 3] = [
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.633851707],
];
const M1_LMS_TO_XYZ: [[f64; 3]; 3] = [
    [1.2270138511035211, -0.5577999806518222, 0.2812561489664678],
    [-0.0405801784232806, 1.11225686961683, -0.07167667866560119],
    [-0.0763812845057069, -0.4214819784180127, 1.5861632204407947],
];
const M2_LMS_TO_LAB: [[f64; 3]; 3] = [
    [0.2104542553, 0.793617785, -0.0040720468],
    [1.9779984951, -2.428592205, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.808675766],
];
const M2_LAB_TO_LMS: [[f64; 3]; 3] = [
    [0.9999999984505196, 0.3963377921737678, 0.21580375806075877],
    [
        1.0000000088817607,
        -0.10556134232365633,
        -0.0638541747717059,
    ],
    [
        1.0000000546724108,
        -0.08948418209496574,
        -1.2914855378640917,
    ],
];

// C_max(L, h) table grid dims, matching `_OKLCH_CMAX_TABLE_N_*`. The L-axis
// bounds and the bisection's chroma ceiling are per-space (see `build`).
const N_L: usize = 64;
const N_H: usize = 720;
const N_BISECT: usize = 18;

#[inline]
fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn xyz_to_oklab(xyz: [f64; 3]) -> [f64; 3] {
    let lms = mat_vec(&M1_XYZ_TO_LMS, xyz);
    let lms_ = [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()];
    mat_vec(&M2_LMS_TO_LAB, lms_)
}

#[inline]
fn oklab_to_xyz(lab: [f64; 3]) -> [f64; 3] {
    let lms_ = mat_vec(&M2_LAB_TO_LMS, lab);
    let lms = [lms_[0].powi(3), lms_[1].powi(3), lms_[2].powi(3)];
    mat_vec(&M1_LMS_TO_XYZ, lms)
}

// Ottosson 2023 "Lr": a 1D monotonic remap of OkLab L so the lightness scale
// tracks CIELAB L* more closely. Lr(0)=0, Lr(1)=1.
const OKLRAB_K1: f64 = 0.206;
const OKLRAB_K2: f64 = 0.03;
const OKLRAB_K3: f64 = (1.0 + OKLRAB_K1) / (1.0 + OKLRAB_K2);

#[inline]
fn oklab_l_to_lr(l: f64) -> f64 {
    let t = OKLRAB_K3 * l - OKLRAB_K1;
    0.5 * (t + (t * t + 4.0 * OKLRAB_K2 * OKLRAB_K3 * l).sqrt())
}

#[inline]
fn oklrab_lr_to_l(lr: f64) -> f64 {
    (lr * (lr + OKLRAB_K1)) / (OKLRAB_K3 * (lr + OKLRAB_K2))
}

/// Smooth Reinhard knee on normalized distance: identity below `threshold`,
/// asymptotic at `limit` above it. Matches the ACES RGC v1.3 reference.
#[inline]
pub(crate) fn reinhard_knee(d: f64, threshold: f64, limit: f64, power: f64) -> f64 {
    if d > threshold {
        let scale = limit - threshold;
        let x = (d - threshold) / scale;
        let y = x / (1.0 + x.powf(power)).powf(1.0 / power);
        threshold + scale * y
    } else {
        d
    }
}

#[inline]
fn h_grid(j: usize) -> f64 {
    // linspace(-pi, pi, N_H, endpoint=False)
    -std::f64::consts::PI + (2.0 * std::f64::consts::PI) * (j as f64) / (N_H as f64)
}

/// Perceptual space the chroma reduction runs in.
#[derive(Clone, Copy, PartialEq)]
enum Space {
    Oklch,
    /// OkLab with the Ottosson Lr-rebased lightness on the L axis.
    Oklrab,
    OklchProPhoto,
    OklrabProPhoto,
    Cam16ucs,
    Cam16ucsProPhoto,
}

impl Space {
    /// Linear RGB in the space's target gamut → perceptual coords `[L, a, b]`.
    #[inline]
    fn from_rgb(self, rgb: [f64; 3]) -> [f64; 3] {
        match self {
            // oklrab shares OkLab coords; only the C_max lookup index differs.
            Space::Oklch | Space::Oklrab => xyz_to_oklab(mat_vec(&SRGB_RGB_TO_XYZ, rgb)),
            Space::OklchProPhoto | Space::OklrabProPhoto => {
                xyz_to_oklab(mat_vec(&PROPHOTO_RGB_TO_XYZ, rgb))
            }
            Space::Cam16ucs => xyz_to_cam16ucs(mat_vec(&SRGB_RGB_TO_XYZ, rgb)),
            Space::Cam16ucsProPhoto => {
                let xyz = mat_vec(&PROPHOTO_RGB_TO_XYZ, rgb);
                xyz_to_cam16ucs_with(xyz, cam16_prophoto_params())
            }
        }
    }
    /// Perceptual coords `[L, a, b]` → linear RGB in the target gamut.
    #[inline]
    fn to_rgb(self, lab: [f64; 3]) -> [f64; 3] {
        let xyz = match self {
            Space::Oklch | Space::Oklrab => oklab_to_xyz(lab),
            Space::OklchProPhoto | Space::OklrabProPhoto => {
                return mat_vec(&XYZ_TO_PROPHOTO_RGB, oklab_to_xyz(lab));
            }
            Space::Cam16ucs => cam16ucs_to_xyz(lab),
            Space::Cam16ucsProPhoto => {
                let xyz = cam16ucs_to_xyz_with(lab, cam16_prophoto_params());
                return mat_vec(&XYZ_TO_PROPHOTO_RGB, xyz);
            }
        };
        mat_vec(&SRGB_XYZ_TO_RGB, xyz)
    }
    /// Reconstruction lightness `L` → C_max-table lookup index. Identity except
    /// oklrab, which indexes the table by Ottosson's rebased `Lr`.
    #[inline]
    fn lookup_lightness(self, l: f64) -> f64 {
        match self {
            Space::Oklrab | Space::OklrabProPhoto => oklab_l_to_lr(l),
            _ => l,
        }
    }
    /// C_max-table lookup index → reconstruction lightness `L` (for the bake).
    #[inline]
    fn recon_lightness(self, lookup: f64) -> f64 {
        match self {
            Space::Oklrab | Space::OklrabProPhoto => oklrab_lr_to_l(lookup),
            _ => lookup,
        }
    }
    /// `(l_min, l_max, chroma_upper)` — the C_max table's lightness-axis
    /// bounds and the bisection's chroma ceiling, fixed per space and target
    /// gamut. Owned by `Space` so the table cache can key by
    /// space alone.
    fn table_geometry(self) -> (f64, f64, f64) {
        match self {
            Space::Oklch | Space::Oklrab | Space::OklchProPhoto | Space::OklrabProPhoto => {
                (0.02, 1.0, 0.5)
            }
            Space::Cam16ucs | Space::Cam16ucsProPhoto => (1.0, 110.0, 150.0),
        }
    }
}

/// Which algorithm the compressor runs.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Off,
    /// ACES Reference Gamut Compression — per-channel knee in RGB, no table.
    AcesRgc,
    /// Perceptual chroma reduction against a `C_max(L, h)` table.
    Perceptual(Space),
}

/// Pre-computed output gamut compressor for a fixed output color space.
#[derive(Clone)]
pub struct OutputGamutCompress {
    kind: Kind,
    knee: (f64, f64, f64),
    lightness: Option<(f64, f64, f64)>,
    /// Perceptual lightness of the output white (1.0 for OkLab, ~100 for CAM16-UCS).
    l_white: f64,
    /// Lightness-axis bounds of the `C_max` table grid.
    l_min: f64,
    l_max: f64,
    /// `C_max(L, h)` table, row-major `[N_L][N_H]`. Empty for off/aces_rgc.
    /// Shared via the process-wide cache — the table depends only on the
    /// perceptual space (the knee/lightness params don't enter the bake),
    /// and the ~105 ms bisection would otherwise re-run on every pipeline
    /// rebuild (the GUI rebuilds the pipeline per slider change).
    cmax: Arc<Vec<f64>>,
    output_color_space: String,
}

impl OutputGamutCompress {
    /// No-op (identity) compressor.
    pub fn identity() -> Self {
        Self {
            kind: Kind::Off,
            knee: (0.0, 1.0, 1.0),
            lightness: None,
            l_white: 1.0,
            l_min: 0.0,
            l_max: 1.0,
            cmax: Arc::new(Vec::new()),
            output_color_space: "sRGB".into(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.kind != Kind::Off
    }

    /// Build from params for the given output color space. `off` is a silent
    /// identity; perceptual algorithms build their target-gamut `C_max` table.
    /// Unsupported algorithms, output spaces other than sRGB/ProPhoto RGB, or
    /// unknown algorithms log an error and fall back to identity.
    pub fn build(params: &OutputGamutCompressParams, output_color_space: &str) -> Self {
        let knee = (
            params.knee[0] as f64,
            params.knee[1] as f64,
            params.knee[2] as f64,
        );
        let lightness = params
            .lightness_compression
            .map(|l| (l[0] as f64, l[1] as f64, l[2] as f64));
        // aces_rgc is per-channel in RGB (no perceptual space / table / lightness).
        if params.algorithm == "aces_rgc" {
            return Self {
                kind: Kind::AcesRgc,
                knee,
                lightness: None,
                l_white: 1.0,
                l_min: 0.0,
                l_max: 1.0,
                cmax: Arc::new(Vec::new()),
                output_color_space: output_color_space.to_string(),
            };
        }
        let space = match params.algorithm.as_str() {
            "off" => return Self::identity(),
            "oklch" if output_color_space == "sRGB" => Space::Oklch,
            "oklrab" if output_color_space == "sRGB" => Space::Oklrab,
            "cam16ucs" if output_color_space == "sRGB" => Space::Cam16ucs,
            "oklch" | "oklrab" | "cam16ucs" if output_color_space == "ProPhoto RGB" => {
                match params.algorithm.as_str() {
                    "oklch" => Space::OklchProPhoto,
                    "oklrab" => Space::OklrabProPhoto,
                    _ => Space::Cam16ucsProPhoto,
                }
            }
            algo @ ("oklch" | "oklrab" | "cam16ucs") => {
                tracing::error!(
                    algorithm = algo,
                    output = output_color_space,
                    "output gamut compression: only sRGB and ProPhoto RGB outputs are supported; disabling"
                );
                return Self::identity();
            }
            other @ "jzazbz" => {
                tracing::error!(
                    algorithm = other,
                    "output gamut compression: jzazbz not yet ported; disabling"
                );
                return Self::identity();
            }
            other => {
                tracing::error!(
                    algorithm = other,
                    "output gamut compression: unknown algorithm; disabling"
                );
                return Self::identity();
            }
        };
        // Perceptual white: OkLab white sits at L=1; CAM16-UCS Jp is measured
        // at the output white under the configured viewing conditions.
        let l_white = match space {
            // OkLab white sits at L=1, and Lr(1)=1, so both are 1.0.
            Space::Oklch | Space::Oklrab | Space::OklchProPhoto | Space::OklrabProPhoto => 1.0,
            Space::Cam16ucs => xyz_to_cam16ucs(SRGB_WHITE_XYZ)[0],
            Space::Cam16ucsProPhoto => {
                xyz_to_cam16ucs_with(PROPHOTO_WHITE_XYZ, cam16_prophoto_params())[0]
            }
        };
        let (l_min, l_max, _) = space.table_geometry();
        let cmax = cmax_table_cached(space);
        Self {
            kind: Kind::Perceptual(space),
            knee,
            lightness,
            l_white,
            l_min,
            l_max,
            cmax,
            output_color_space: output_color_space.to_string(),
        }
    }

    /// GPU-pass inputs for the active compressor (mode id matching
    /// `gamut_compress.wgsl`, knee/lightness coefficients, table geometry
    /// and the baked `C_max` table). `None` when the compressor is off —
    /// callers skip the pass entirely.
    pub fn gpu_params(&self) -> Option<spektrafilm_gpu::GamutGpuParams<'_>> {
        if self.output_color_space == "ProPhoto RGB" {
            return None;
        }
        let mode = match self.kind {
            Kind::Off => return None,
            Kind::AcesRgc => 0,
            Kind::Perceptual(Space::Oklch) => 1,
            Kind::Perceptual(Space::Oklrab) => 2,
            Kind::Perceptual(Space::OklchProPhoto) => 1,
            Kind::Perceptual(Space::OklrabProPhoto) => 2,
            Kind::Perceptual(Space::Cam16ucs) => 3,
            Kind::Perceptual(Space::Cam16ucsProPhoto) => 3,
        };
        Some(spektrafilm_gpu::GamutGpuParams {
            mode,
            knee: [self.knee.0 as f32, self.knee.1 as f32, self.knee.2 as f32],
            lightness: self
                .lightness
                .map(|(t, l, p)| [t as f32, l as f32, p as f32]),
            l_white: self.l_white as f32,
            l_min: self.l_min as f32,
            l_max: self.l_max as f32,
            cmax: &self.cmax,
            n_l: N_L as u32,
            n_h: N_H as u32,
        })
    }

    /// Bilinear `C_max(L, h)` lookup (L clamped, h wraps).
    #[inline]
    fn c_max_lookup(&self, l: f64, h: f64) -> f64 {
        let l = l.clamp(self.l_min, self.l_max);
        let h_step = h_grid(1) - h_grid(0);
        let h_idx = (h - h_grid(0)) / h_step;
        let h_floor = h_idx.floor();
        let h_lo = (h_floor as i64).rem_euclid(N_H as i64) as usize;
        let h_hi = (h_lo + 1) % N_H;
        let h_frac = h_idx - h_floor;

        let l_idx = (l - self.l_min) / (self.l_max - self.l_min) * ((N_L - 1) as f64);
        let l_lo = (l_idx.floor() as i64).clamp(0, (N_L - 2) as i64) as usize;
        let l_hi = l_lo + 1;
        let l_frac = l_idx - l_lo as f64;

        let t = &self.cmax;
        let v00 = t[l_lo * N_H + h_lo];
        let v01 = t[l_lo * N_H + h_hi];
        let v10 = t[l_hi * N_H + h_lo];
        let v11 = t[l_hi * N_H + h_hi];
        v00 * (1.0 - l_frac) * (1.0 - h_frac)
            + v01 * (1.0 - l_frac) * h_frac
            + v10 * l_frac * (1.0 - h_frac)
            + v11 * l_frac * h_frac
    }

    /// Compress a single linear output-RGB pixel. Identity when inactive.
    #[inline]
    pub fn compress(&self, rgb: [f64; 3]) -> [f64; 3] {
        let space = match self.kind {
            Kind::Off => return rgb,
            Kind::AcesRgc => return self.compress_aces_rgc(rgb),
            Kind::Perceptual(s) => s,
        };
        let lab = space.from_rgb(rgb);
        // One-sided lightness compression first (so C_max is looked up at the
        // corrected L), normalized by the perceptual white.
        let mut l = lab[0];
        if let Some((t, lim, p)) = self.lightness {
            l = reinhard_knee(l / self.l_white, t, lim, p) * self.l_white;
        }
        let a = lab[1];
        let b = lab[2];
        let c = a.hypot(b);
        let h = b.atan2(a);
        // C_max is looked up at the lookup-lightness (Lr for oklrab); the
        // reconstruction below keeps the reconstruction lightness `l`.
        let c_max = self.c_max_lookup(space.lookup_lightness(l), h);
        let safe = c_max.max(1e-9);
        let d_comp = reinhard_knee(c / safe, self.knee.0, self.knee.1, self.knee.2);
        let c_new = d_comp * safe;
        space.to_rgb([l, c_new * h.cos(), c_new * h.sin()])
    }

    /// ACES Reference Gamut Compression v1.3: per-channel Reinhard knee on the
    /// achromatic distance `d = (max - c)/max`. Mirrors `compress_rgb_aces_rgc`.
    #[inline]
    fn compress_aces_rgc(&self, rgb: [f64; 3]) -> [f64; 3] {
        let ach = rgb[0].max(rgb[1]).max(rgb[2]);
        if ach <= 1e-12 {
            return rgb;
        }
        let mut out = [0.0f64; 3];
        for c in 0..3 {
            let d = (ach - rgb[c]) / ach;
            let dc = reinhard_knee(d, self.knee.0, self.knee.1, self.knee.2);
            out[c] = ach * (1.0 - dc);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────
// CAM16-UCS (upstream's default `cam16ucs` method).
//
// Fixed viewing conditions: XYZ_w = output (sRGB/D65) white, L_A = 64 cd/m²,
// Y_b = 20, Average surround. Every viewing-condition-dependent quantity is
// therefore constant and precomputed (extracted from colour-science). The
// per-pixel path is the CIECAM16 forward/inverse + the Luo 2006 UCS transform.
// XYZ here is at Y=1; CIECAM16 works in the Y=100 domain, hence the ×/÷100.
// ─────────────────────────────────────────────────────────────────────────
const CAM16_M16: [[f64; 3]; 3] = [
    [0.401288, 0.650173, -0.051461],
    [-0.250268, 1.204414, 0.045854],
    [-0.002079, 0.048952, 0.953127],
];
const CAM16_M16I: [[f64; 3]; 3] = [
    [1.8620678550872327, -1.0112546305316843, 0.14918677544445175],
    [
        0.3875265432361372,
        0.6214474419314753,
        -0.008973985167612516,
    ],
    [
        -0.015841498849333863,
        -0.03412293802851557,
        1.0499644368778496,
    ],
];
const CAM16_D_RGB: [f64; 3] = [1.0228770275436545, 0.9852074782801457, 0.9285450586783286];
const CAM16_F_L: f64 = 0.6839903845696502;
const CAM16_N: f64 = 0.2;
const CAM16_N_BB: f64 = 1.0003040045593807;
const CAM16_N_CB: f64 = 1.0003040045593807;
const CAM16_Z: f64 = 1.9272135954999579;
const CAM16_A_W: f64 = 37.16907530221132;
const CAM16_C: f64 = 0.69;
const CAM16_N_C: f64 = 1.0;
// Luo 2006 CAM16-UCS coefficients (K_L=1.0, c1, c2).
const UCS_C1: f64 = 0.007;
const UCS_C2: f64 = 0.0228;

#[derive(Clone, Copy)]
struct Cam16Params {
    d_rgb: [f64; 3],
    f_l: f64,
    n: f64,
    n_bb: f64,
    n_cb: f64,
    z: f64,
    a_w: f64,
    c: f64,
    n_c: f64,
}

fn cam16_prophoto_params() -> &'static Cam16Params {
    static PARAMS: OnceLock<Cam16Params> = OnceLock::new();
    PARAMS.get_or_init(|| {
        let y_w = 100.0 * PROPHOTO_WHITE_XYZ[1];
        let y_b = 20.0;
        let n = y_b / y_w;
        let k: f64 = 1.0 / (5.0 * 64.0 + 1.0);
        let k4 = k.powi(4);
        let f_l = 0.2 * k4 * (5.0 * 64.0) + 0.1 * (1.0 - k4).powi(2) * (5.0 * 64.0_f64).cbrt();
        let d = (1.0 - ((-64.0_f64 - 42.0) / 92.0).exp() / 3.6).clamp(0.0, 1.0);
        let rgb_w = mat_vec(
            &CAM16_M16,
            [
                PROPHOTO_WHITE_XYZ[0] * 100.0,
                PROPHOTO_WHITE_XYZ[1] * 100.0,
                PROPHOTO_WHITE_XYZ[2] * 100.0,
            ],
        );
        let d_rgb = [
            d * y_w / rgb_w[0] + 1.0 - d,
            d * y_w / rgb_w[1] + 1.0 - d,
            d * y_w / rgb_w[2] + 1.0 - d,
        ];
        let rgb_c = [
            rgb_w[0] * d_rgb[0],
            rgb_w[1] * d_rgb[1],
            rgb_w[2] * d_rgb[2],
        ];
        let mut adapted = [0.0; 3];
        for i in 0..3 {
            let flr = (f_l * rgb_c[i].abs() / 100.0).powf(0.42);
            adapted[i] = 400.0 * rgb_c[i].signum() * flr / (27.13 + flr) + 0.1;
        }
        let n_bb = 0.725 * (1.0 / n).powf(0.2);
        Cam16Params {
            d_rgb,
            f_l,
            n,
            n_bb,
            n_cb: n_bb,
            z: 1.48 + n.sqrt(),
            a_w: (2.0 * adapted[0] + adapted[1] + adapted[2] / 20.0 - 0.305) * n_bb,
            c: 0.69,
            n_c: 1.0,
        }
    })
}

#[inline]
fn atan2_deg(y: f64, x: f64) -> f64 {
    y.atan2(x).to_degrees().rem_euclid(360.0)
}

#[inline]
fn cam16_padc_forward(rgb: [f64; 3], f_l: f64) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for i in 0..3 {
        let flr = (f_l * rgb[i].abs() / 100.0).powf(0.42);
        out[i] = 400.0 * rgb[i].signum() * flr / (27.13 + flr) + 0.1;
    }
    out
}

#[inline]
fn cam16_padc_inverse(rgb: [f64; 3], f_l: f64) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for i in 0..3 {
        let d = rgb[i] - 0.1;
        let base = (27.13 * d.abs()) / (400.0 - d.abs());
        out[i] = d.signum() * 100.0 / f_l * base.powf(1.0 / 0.42);
    }
    out
}

/// XYZ (Y=1) → CAM16-UCS `(Jp, ap, bp)`.
fn xyz_to_cam16ucs(xyz: [f64; 3]) -> [f64; 3] {
    xyz_to_cam16ucs_with(
        xyz,
        &Cam16Params {
            d_rgb: CAM16_D_RGB,
            f_l: CAM16_F_L,
            n: CAM16_N,
            n_bb: CAM16_N_BB,
            n_cb: CAM16_N_CB,
            z: CAM16_Z,
            a_w: CAM16_A_W,
            c: CAM16_C,
            n_c: CAM16_N_C,
        },
    )
}

fn xyz_to_cam16ucs_with(xyz: [f64; 3], params: &Cam16Params) -> [f64; 3] {
    let xyz100 = [xyz[0] * 100.0, xyz[1] * 100.0, xyz[2] * 100.0];
    let rgb = mat_vec(&CAM16_M16, xyz100);
    let rgb_c = [
        rgb[0] * params.d_rgb[0],
        rgb[1] * params.d_rgb[1],
        rgb[2] * params.d_rgb[2],
    ];
    let [ra, ga, ba] = cam16_padc_forward(rgb_c, params.f_l);
    let a = ra - 12.0 * ga / 11.0 + ba / 11.0;
    let b = (ra + ga - 2.0 * ba) / 9.0;
    let h = atan2_deg(b, a);
    let e_t = 0.25 * ((2.0 + h * std::f64::consts::PI / 180.0).cos() + 3.8);
    let a_resp = (2.0 * ra + ga + ba / 20.0 - 0.305) * params.n_bb;
    let jj = 100.0 * (a_resp / params.a_w).powf(params.c * params.z);
    let denom = ra + ga + 21.0 * ba / 20.0;
    let t = if denom != 0.0 {
        (50000.0 / 13.0) * params.n_c * params.n_cb * (e_t * (a * a + b * b).sqrt()) / denom
    } else {
        0.0
    };
    let cc = t.powf(0.9) * (jj / 100.0).powf(0.5) * (1.64 - 0.29f64.powf(params.n)).powf(0.73);
    let m = cc * params.f_l.powf(0.25);
    let jp = (1.0 + 100.0 * UCS_C1) * jj / (1.0 + UCS_C1 * jj);
    let mp = (1.0 / UCS_C2) * (1.0 + UCS_C2 * m).ln();
    let hr = h * std::f64::consts::PI / 180.0;
    [jp, mp * hr.cos(), mp * hr.sin()]
}

/// CAM16-UCS `(Jp, ap, bp)` → XYZ (Y=1).
fn cam16ucs_to_xyz(jab: [f64; 3]) -> [f64; 3] {
    cam16ucs_to_xyz_with(
        jab,
        &Cam16Params {
            d_rgb: CAM16_D_RGB,
            f_l: CAM16_F_L,
            n: CAM16_N,
            n_bb: CAM16_N_BB,
            n_cb: CAM16_N_CB,
            z: CAM16_Z,
            a_w: CAM16_A_W,
            c: CAM16_C,
            n_c: CAM16_N_C,
        },
    )
}

fn cam16ucs_to_xyz_with(jab: [f64; 3], params: &Cam16Params) -> [f64; 3] {
    let [jp, ap, bp] = jab;
    let mp = (ap * ap + bp * bp).sqrt();
    let h = atan2_deg(bp, ap);
    let jj = jp / ((1.0 + 100.0 * UCS_C1) - UCS_C1 * jp);
    let m = ((UCS_C2 * mp).exp() - 1.0) / UCS_C2;
    let cc = m / params.f_l.powf(0.25);
    let j_prime = jj.max(f64::EPSILON);
    let t = (cc / ((j_prime / 100.0).sqrt() * (1.64 - 0.29f64.powf(params.n)).powf(0.73)))
        .powf(1.0 / 0.9);
    let e_t = 0.25 * ((2.0 + h * std::f64::consts::PI / 180.0).cos() + 3.8);
    let a_resp = params.a_w * (jj / 100.0).powf(1.0 / (params.c * params.z));
    let p1 = if t != 0.0 {
        (50000.0 / 13.0) * params.n_c * params.n_cb * e_t / t
    } else {
        0.0
    };
    let p2 = a_resp / params.n_bb + 0.305;
    let p3 = 21.0 / 20.0;
    let (mut a, mut b) = cam16_opponent_inverse(p1, p2, p3, h);
    // Achromatic guard, matching colour's `ab * np.where(t == 0, 0, 1)`:
    // when t == 0 the hue is undefined and the opponent inverse must be zeroed.
    if t == 0.0 {
        a = 0.0;
        b = 0.0;
    }
    let ra = (460.0 * p2 + 451.0 * a + 288.0 * b) / 1403.0;
    let ga = (460.0 * p2 - 891.0 * a - 261.0 * b) / 1403.0;
    let ba = (460.0 * p2 - 220.0 * a - 6300.0 * b) / 1403.0;
    let rgb_c = cam16_padc_inverse([ra, ga, ba], params.f_l);
    let rgb = [
        rgb_c[0] / params.d_rgb[0],
        rgb_c[1] / params.d_rgb[1],
        rgb_c[2] / params.d_rgb[2],
    ];
    let xyz100 = mat_vec(&CAM16_M16I, rgb);
    [xyz100[0] / 100.0, xyz100[1] / 100.0, xyz100[2] / 100.0]
}

/// Inverse opponent dimensions (CIECAM02/16 `opponent_colour_dimensions_inverse`).
fn cam16_opponent_inverse(p1: f64, p2: f64, p3: f64, h: f64) -> (f64, f64) {
    let hr = h * std::f64::consts::PI / 180.0;
    let s = hr.sin();
    let c = hr.cos();
    let nn = p2 * (2.0 + p3) * (460.0 / 1403.0);
    if s.abs() >= c.abs() {
        let p4 = if s != 0.0 { p1 / s } else { 0.0 };
        let b = nn
            / (p4 + (2.0 + p3) * (220.0 / 1403.0) * (c / s) - (27.0 / 1403.0)
                + p3 * (6300.0 / 1403.0));
        (b * (c / s), b)
    } else {
        let p5 = if c != 0.0 { p1 / c } else { 0.0 };
        let a = nn
            / (p5 + (2.0 + p3) * (220.0 / 1403.0)
                - ((27.0 / 1403.0) - p3 * (6300.0 / 1403.0)) * (s / c));
        (a, a * (s / c))
    }
}

/// Process-wide `C_max` table cache, keyed by perceptual space. The bake's
/// other inputs come from `Space::table_geometry`, so the space alone
/// identifies the table. The build runs outside the lock — a racing
/// double-build is harmless and cheaper than serializing on it.
fn cmax_table_cached(space: Space) -> Arc<Vec<f64>> {
    static CACHE: OnceLock<Mutex<HashMap<u8, Arc<Vec<f64>>>>> = OnceLock::new();
    let key = match space {
        Space::Oklch => 0u8,
        Space::Oklrab => 1,
        Space::Cam16ucs => 2,
        Space::Cam16ucsProPhoto => 3,
        Space::OklchProPhoto => 4,
        Space::OklrabProPhoto => 5,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(table) = cache.lock().unwrap().get(&key) {
        return table.clone();
    }
    let table = Arc::new(build_cmax_table(space));
    cache.lock().unwrap().insert(key, table.clone());
    table
}

/// Bisect the max in-gamut chroma at each `(L, h)` grid node for the given
/// perceptual space, mirroring `_build_polar_perceptual_c_max_table`. Built
/// once per space (see `cmax_table_cached`).
fn build_cmax_table(space: Space) -> Vec<f64> {
    use rayon::prelude::*;
    let (l_min, l_max, chroma_upper) = space.table_geometry();
    let mut table = vec![0.0f64; N_L * N_H];
    table
        .par_chunks_exact_mut(N_H)
        .enumerate()
        .for_each(|(i, row)| {
            // Grid value is the lookup-lightness; map to reconstruction
            // lightness (Lr→L for oklrab) before the gamut check.
            let lookup_l = l_min + (l_max - l_min) * (i as f64) / ((N_L - 1) as f64);
            let l = space.recon_lightness(lookup_l);
            for (j, out) in row.iter_mut().enumerate() {
                let h = h_grid(j);
                let (cos_h, sin_h) = (h.cos(), h.sin());
                let mut lo = 0.0f64;
                let mut hi = chroma_upper;
                for _ in 0..N_BISECT {
                    let mid = (lo + hi) * 0.5;
                    let rgb = space.to_rgb([l, mid * cos_h, mid * sin_h]);
                    let in_gamut = rgb.iter().all(|&v| v >= -1e-6 && v <= 1.0 + 1e-6);
                    if in_gamut {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                *out = lo;
            }
        });
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_for(algorithm: &str) -> OutputGamutCompress {
        let params = OutputGamutCompressParams {
            algorithm: algorithm.into(),
            knee: [0.0, 1.0, 6.0],
            lightness_compression: Some([0.7, 1.0, 2.2]),
        };
        OutputGamutCompress::build(&params, "sRGB")
    }

    // Full cross-checks against upstream (tmp/gen_oklch_ref.py,
    // tmp/gen_cam16_ref.py) confirmed bit-identical C_max tables and a
    // per-pixel compress matching Python to ~1.6e-8 over 100+ in/out-of-gamut
    // rows for both spaces, and the CAM16-UCS conversion matching colour to
    // ~3e-15 (forward) with a ~4e-15 roundtrip. The reference rows below pin
    // that in CI without depending on the generated dumps.

    fn check_reference(algo: &str, cases: &[([f64; 3], [f64; 3])]) {
        let comp = build_for(algo);
        check_compressor_reference(algo, &comp, cases);
    }

    fn check_compressor_reference(
        algo: &str,
        comp: &OutputGamutCompress,
        cases: &[([f64; 3], [f64; 3])],
    ) {
        for (rgb, want) in cases {
            let got = comp.compress(*rgb);
            for c in 0..3 {
                // 1e-6 bound: bit-faithful tables; the per-pixel chain agrees
                // with Python to ~1.6e-8 at extreme out-of-gamut points
                // (release-build FMA reordering), far below the threshold.
                assert!(
                    (got[c] - want[c]).abs() < 1e-6,
                    "{algo} rgb {rgb:?} ch {c}: got {} want {}",
                    got[c],
                    want[c]
                );
            }
        }
    }

    fn build_for_prophoto(algorithm: &str) -> OutputGamutCompress {
        let params = OutputGamutCompressParams {
            algorithm: algorithm.into(),
            knee: [0.0, 1.0, 6.0],
            lightness_compression: Some([0.7, 1.0, 2.2]),
        };
        OutputGamutCompress::build(&params, "ProPhoto RGB")
    }

    #[test]
    fn prophoto_oklch_matches_python_reference() {
        check_compressor_reference(
            "ProPhoto oklch",
            &build_for_prophoto("oklch"),
            &[
                (
                    [0.5, 0.4, 0.3],
                    [0.4985381706538354, 0.39874271720967963, 0.2989796528305229],
                ),
                (
                    [1.4, 0.2, 0.1],
                    [0.9963083364126335, 0.37365513926763877, 0.3208137542477491],
                ),
                (
                    [0.1, 0.3, 1.2],
                    [0.10350153873410428, 0.29397512666912284, 0.9548831145164554],
                ),
            ],
        );
    }

    #[test]
    fn prophoto_oklrab_matches_python_reference() {
        check_compressor_reference(
            "ProPhoto oklrab",
            &build_for_prophoto("oklrab"),
            &[
                (
                    [0.5, 0.4, 0.3],
                    [0.4985381695524968, 0.3987427178664791, 0.29897965595762577],
                ),
                (
                    [1.4, 0.2, 0.1],
                    [0.9963053115700281, 0.37365691201865087, 0.3208163672531871],
                ),
                (
                    [0.1, 0.3, 1.2],
                    [0.10350190199091726, 0.2939747497778835, 0.9548733005279845],
                ),
            ],
        );
    }

    /// Regression guard: oklch reference rows from `compress_rgb_oklch_chroma`
    /// (knee (0,1,6), lightness (0.7,1,2.2)) — in-gamut, super-red, super-cyan.
    #[test]
    fn oklch_matches_python_reference() {
        check_reference(
            "oklch",
            &[
                (
                    [0.5, 0.4, 0.3],
                    [0.49934573392726778, 0.3994604511652195, 0.29954075929970825],
                ),
                (
                    [2.0, 0.0, 0.0],
                    [0.9999165297352135, 0.32648626944154618, 0.2608847299103626],
                ),
                (
                    [0.0, 2.0, 2.0],
                    [0.50013028382005009, 1.0000925606064377, 0.98946014302066698],
                ),
            ],
        );
    }

    /// Regression guard: cam16ucs reference rows from
    /// `compress_rgb_cam16ucs_chroma` (same knee/lightness).
    #[test]
    fn cam16ucs_matches_python_reference() {
        check_reference(
            "cam16ucs",
            &[
                (
                    [0.5, 0.4, 0.3],
                    [
                        0.49974636928661886,
                        0.39981273339722234,
                        0.29983897633446865,
                    ],
                ),
                (
                    [2.0, 0.0, 0.0],
                    [0.9987900431158222, 0.32032525739267781, 0.25154805990152163],
                ),
                (
                    [0.0, 2.0, 2.0],
                    [
                        0.46476658163581974,
                        0.99783501836142674,
                        0.99071804859295698,
                    ],
                ),
            ],
        );
    }

    #[test]
    fn prophoto_cam16_matches_python_colour_reference() {
        let rgb = [0.5, 0.4, 0.3];
        let jab = Space::Cam16ucsProPhoto.from_rgb(rgb);
        let expected_jab = [74.94011665199008, 10.356644228514293, 10.724511069384786];
        for i in 0..3 {
            assert!(
                (jab[i] - expected_jab[i]).abs() < 1e-6,
                "CAM16-UCS component {i}: got {}, expected {}",
                jab[i],
                expected_jab[i]
            );
        }

        let roundtrip = Space::Cam16ucsProPhoto.to_rgb(jab);
        let expected_rgb = [0.500038279, 0.399994741, 0.300007881];
        for i in 0..3 {
            assert!(
                (roundtrip[i] - expected_rgb[i]).abs() < 1e-6,
                "ProPhoto roundtrip component {i}: got {}, expected {}",
                roundtrip[i],
                expected_rgb[i]
            );
        }
    }

    #[test]
    fn prophoto_gamut_compression_matches_python_reference() {
        let params = OutputGamutCompressParams {
            algorithm: "cam16ucs".into(),
            knee: [0.0, 1.0, 6.0],
            lightness_compression: Some([0.7, 1.0, 2.2]),
        };
        let compressor = OutputGamutCompress::build(&params, "ProPhoto RGB");
        let cases = [
            (
                [0.5, 0.4, 0.3],
                [0.4993171250855837, 0.3993929699943089, 0.29952752410106853],
            ),
            (
                [1.4, 0.2, 0.1],
                [0.9901349832692647, 0.3737128755954074, 0.2773546123820442],
            ),
            (
                [0.1, 0.3, 1.2],
                [0.12370000587520902, 0.27920616695876566, 0.8154081134208891],
            ),
        ];
        for (rgb, expected) in cases {
            let actual = compressor.compress(rgb);
            for i in 0..3 {
                assert!(
                    (actual[i] - expected[i]).abs() < 1e-6,
                    "ProPhoto CAM16 sample {rgb:?} channel {i}: got {}, expected {}",
                    actual[i],
                    expected[i]
                );
            }
        }
    }

    /// Regression guard: oklrab reference rows from `compress_rgb_oklrab_chroma`.
    #[test]
    fn oklrab_matches_python_reference() {
        check_reference(
            "oklrab",
            &[
                (
                    [0.5, 0.4, 0.3],
                    [0.4993457339279313, 0.39946045116508211, 0.29954075929887197],
                ),
                (
                    [2.0, 0.0, 0.0],
                    [
                        0.99981431200564952,
                        0.32651963747538049,
                        0.26092249322038208,
                    ],
                ),
                (
                    [0.0, 2.0, 2.0],
                    [0.49984508410893458, 1.0001915827542378, 0.98955691468555185],
                ),
            ],
        );
    }

    /// Regression guard: aces_rgc reference rows from `compress_rgb_aces_rgc`
    /// (per-channel knee, no lightness/table).
    #[test]
    fn aces_rgc_matches_python_reference() {
        check_reference(
            "aces_rgc",
            &[
                (
                    [0.5, 0.4, 0.3],
                    [0.5, 0.40000106662684631, 0.30013620807161223],
                ),
                (
                    [2.0, 0.0, 0.0],
                    [2.0, 0.21820256371932145, 0.21820256371932145],
                ),
                (
                    [1.2, -0.1, 0.4],
                    [1.2, 0.092551484339614021, 0.41114161919196679],
                ),
            ],
        );
    }

    /// `off` (and unported algorithms) must be a true identity no-op.
    #[test]
    fn off_is_identity() {
        let params = OutputGamutCompressParams {
            algorithm: "off".into(),
            ..Default::default()
        };
        let comp = OutputGamutCompress::build(&params, "sRGB");
        assert!(!comp.is_active());
        assert_eq!(comp.compress([1.7, 0.0, 0.3]), [1.7, 0.0, 0.3]);
    }
}
