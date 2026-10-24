//! Continuous direction coverage: the mask primitive with fractional edge bins.
//!
//! [`ConeMask`] marks a bin blocked when an occluder covers its center, so
//! visibility jumps by a whole bin as an edge crosses a center. That jump is
//! why a ledger can never defer small changes with binary masks: any motion may
//! cost a full bin. `CoverageMask` instead keeps, per bin, the largest fraction
//! of that bin covered by any single occluder. Fully covered bins are still exact
//! unions (bits); a partially covered bin takes the max, which is exact when one
//! edge falls in it and slightly underestimates coverage when several do.
//!
//! In exchange, visibility is continuous and 1-Lipschitz in each occluder: moving
//! one occluder changes the visible fraction by at most the measure its edges
//! sweep, with no one-bin floor.

use crate::ConeMask;

const BINS: usize = ConeMask::BINS as usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoverageMask {
    full: ConeMask,
    partial: [f32; BINS],
}

impl Default for CoverageMask {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl CoverageMask {
    pub const EMPTY: Self = Self {
        full: ConeMask::EMPTY,
        partial: [0.0; BINS],
    };

    /// Adds an occluder covering normalized positions `[t0, t1]` across the cone
    /// (0 = first edge, 1 = second edge).
    pub fn add_interval(&mut self, t0: f32, t1: f32) {
        let n = BINS as f32;
        let (a, b) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        let (x0, x1) = ((a * n).clamp(0.0, n), (b * n).clamp(0.0, n));
        if x1 <= x0 {
            return;
        }
        let first = (x0.floor() as usize).min(BINS - 1);
        let last = (x1.ceil() as usize).saturating_sub(1).min(BINS - 1);
        if first == last {
            self.cover(first, x1 - x0);
            return;
        }
        self.cover(first, (first + 1) as f32 - x0);
        self.cover(last, x1 - last as f32);
        if last > first + 1 {
            self.full |= ConeMask::range(first as u32 + 1, last as u32 - 1);
        }
    }

    fn cover(&mut self, bin: usize, fraction: f32) {
        if fraction >= 1.0 {
            self.full |= ConeMask::range(bin as u32, bin as u32);
        } else if fraction > self.partial[bin] {
            self.partial[bin] = fraction;
        }
    }

    pub fn is_full(&self) -> bool {
        self.full.is_full()
    }

    /// Bins fully covered so far.
    pub fn full_bins(&self) -> ConeMask {
        self.full
    }

    /// Length of `[t0, t1]` (normalized) that falls outside fully covered bins.
    ///
    /// A fully covered bin stays fully covered whatever else moves, so when this
    /// mask holds static occluders, only open length can change visibility: it
    /// bounds what a moving occluder's edge sweep can do (occlusion-aware charging).
    pub fn open_length(&self, t0: f32, t1: f32) -> f32 {
        let n = BINS as f32;
        let (a, b) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        let (x0, x1) = ((a * n).clamp(0.0, n), (b * n).clamp(0.0, n));
        if x1 <= x0 {
            return 0.0;
        }
        let first = (x0.floor() as usize).min(BINS - 1);
        let last = (x1.ceil() as usize).saturating_sub(1).min(BINS - 1);
        let mut open = 0.0;
        for bin in (first..=last).filter(|&bin| self.full.0 & (1 << bin) == 0) {
            open += (x1.min((bin + 1) as f32) - x0.max(bin as f32)).max(0.0);
        }
        open / n
    }

    /// Fraction of the cone that is unblocked, in `[0, 1]`.
    pub fn visible_fraction(&self) -> f32 {
        let mut visible = 0.0;
        for (bin, &covered) in self.partial.iter().enumerate() {
            if self.full.0 & (1 << bin) == 0 {
                visible += 1.0 - covered;
            }
        }
        visible / BINS as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible(intervals: &[(f32, f32)]) -> f32 {
        let mut m = CoverageMask::EMPTY;
        for &(a, b) in intervals {
            m.add_interval(a, b);
        }
        m.visible_fraction()
    }

    #[test]
    fn single_occluder_is_exact() {
        for &(a, b) in &[
            (0.1, 0.37),
            (0.0, 1.0),
            (0.505, 0.507),
            (0.25, 0.5),
            (0.99, 1.0),
        ] {
            assert!(
                (visible(&[(a, b)]) - (1.0 - (b - a))).abs() < 1e-6,
                "[{a}, {b}]"
            );
        }
    }

    #[test]
    fn correlated_occluders_do_not_double_count() {
        assert!((visible(&[(0.0, 0.5), (0.0, 0.5)]) - 0.5).abs() < 1e-6);
        assert!(visible(&[(0.0, 0.5), (0.5, 1.0)]).abs() < 1e-6);
    }

    #[test]
    fn order_does_not_matter() {
        let a = visible(&[(0.1, 0.3013), (0.3, 0.6), (0.2, 0.25)]);
        let b = visible(&[(0.2, 0.25), (0.3, 0.6), (0.1, 0.3013)]);
        assert_eq!(a, b);
    }

    /// The property the ledger relies on: moving one occluder's edges by `da`
    /// and `db` changes visibility by at most `|da| + |db|`, whatever else is there.
    #[test]
    fn moving_one_occluder_is_one_lipschitz() {
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / 16_777_216.0
        };
        for _ in 0..2000 {
            let others: Vec<(f32, f32)> = (0..4)
                .map(|_| {
                    let a = next();
                    (a, (a + next() * 0.2).min(1.0))
                })
                .collect();
            let (a0, b0) = {
                let a = next();
                (a, (a + next() * 0.3).min(1.0))
            };
            let (da, db) = ((next() - 0.5) * 0.02, (next() - 0.5) * 0.02);
            let (a1, b1) = ((a0 + da).clamp(0.0, 1.0), (b0 + db).clamp(0.0, 1.0));
            let mut before = others.clone();
            before.push((a0, b0));
            let mut after = others.clone();
            after.push((a1.min(b1), b1.max(a1)));
            let change = (visible(&before) - visible(&after)).abs();
            let bound = (a1 - a0).abs() + (b1 - b0).abs();
            assert!(change <= bound + 1e-5, "change {change} > bound {bound}");
        }
    }

    #[test]
    fn open_length_skips_fully_covered_bins() {
        let mut m = CoverageMask::EMPTY;
        assert!((m.open_length(0.1, 0.3) - 0.2).abs() < 1e-6);
        m.add_interval(0.0, 0.5); // bins 0..32 full
        assert!((m.open_length(0.25, 0.75) - 0.25).abs() < 1e-6);
        assert!(m.open_length(0.1, 0.4).abs() < 1e-6);
        m.add_interval(0.5, 0.505); // a partial bin stays open
        assert!((m.open_length(0.5, 0.52) - 0.02).abs() < 1e-6);
    }

    #[test]
    fn binary_mask_is_not_lipschitz() {
        // An edge crossing a bin center by a hair flips a whole bin in ConeMask.
        let center = 0.5 / ConeMask::BINS as f32;
        let a = ConeMask::from_interval(0.0, center - 1e-4).visible_fraction();
        let b = ConeMask::from_interval(0.0, center + 1e-4).visible_fraction();
        assert!((a - b).abs() > 1.0 / 64.0 - 1e-6);
    }
}
