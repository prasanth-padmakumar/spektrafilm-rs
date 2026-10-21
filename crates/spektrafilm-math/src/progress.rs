//! Process-wide progress reporting for long renders.
//!
//! The CLI enables it and the GUI parses `SCAN_PROGRESS <percent> <label>`
//! lines from stderr. Percentages never move backwards.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static LAST_TENTHS: AtomicU32 = AtomicU32::new(0);
static SPAN_LO: AtomicU32 = AtomicU32::new(0);
static SPAN_HI: AtomicU32 = AtomicU32::new(0);

pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
    LAST_TENTHS.store(0, Ordering::Relaxed);
}

/// Absolute percentage range that the next long stage maps its own 0..1
/// fraction onto (see [`report_in_span`]).
pub fn set_span(low_percent: f32, high_percent: f32) {
    SPAN_LO.store(low_percent.to_bits(), Ordering::Relaxed);
    SPAN_HI.store(high_percent.to_bits(), Ordering::Relaxed);
}

pub fn report(percent: f32, label: &str) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let tenths = (percent.clamp(0.0, 100.0) * 10.0).round() as u32;
    let previous = LAST_TENTHS.fetch_max(tenths, Ordering::Relaxed);
    if tenths > previous || percent >= 100.0 {
        eprintln!("SCAN_PROGRESS {:.1} {label}", tenths as f32 / 10.0);
    }
}

pub fn report_in_span(fraction: f32, label: &str) {
    let low = f32::from_bits(SPAN_LO.load(Ordering::Relaxed));
    let high = f32::from_bits(SPAN_HI.load(Ordering::Relaxed));
    report(low + (high - low) * fraction.clamp(0.0, 1.0), label);
}
