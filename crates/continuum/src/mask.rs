//! Direction masks: the visibility primitive.
//!
//! A cone's cross-section is split into [`ConeMask::BINS`] direction bins. A set
//! bit means that direction is blocked. Occluders at any depth set the bins
//! they cover, so combining occluders is a bitwise OR, which keeps correlated
//! occlusion correct (two occluders hiding the same half of a cone block 50%,
//! not 75%).

use std::ops::{BitOr, BitOrAssign};

/// Visibility across a cone, one bit per direction bin (1 = blocked).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConeMask(pub u64);

impl ConeMask {
    pub const BINS: u32 = 64;
    pub const EMPTY: Self = Self(0);
    pub const FULL: Self = Self(u64::MAX);

    /// Mask with bins `lo..=hi` set. Empty when `lo > hi`; `hi` is clamped to the last bin.
    pub fn range(lo: u32, hi: u32) -> Self {
        if lo > hi || lo >= Self::BINS {
            return Self::EMPTY;
        }
        let hi = hi.min(Self::BINS - 1);
        let width = hi - lo + 1;
        if width == Self::BINS {
            Self::FULL
        } else {
            Self(((1u64 << width) - 1) << lo)
        }
    }

    /// Sets the bins whose centers lie in `[t0, t1]`, where `t` is the normalized
    /// position across the cone (0 = first edge, 1 = second edge).
    ///
    /// Using bin centers makes a mask exactly equivalent to one ray per bin, so any
    /// gain over rays comes from doing one test per occluder instead of one per ray.
    pub fn from_interval(t0: f32, t1: f32) -> Self {
        let (a, b) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        let n = Self::BINS as f32;
        let lo = (a * n - 0.5).ceil().max(0.0);
        let hi = (b * n - 0.5).floor().min(n - 1.0);
        if hi < lo {
            return Self::EMPTY;
        }
        Self::range(lo as u32, hi as u32)
    }

    pub fn blocked_bins(self) -> u32 {
        self.0.count_ones()
    }

    pub fn is_full(self) -> bool {
        self.0 == u64::MAX
    }

    /// Fraction of the cone that is unblocked, in `[0, 1]`.
    pub fn visible_fraction(self) -> f32 {
        1.0 - self.blocked_bins() as f32 / Self::BINS as f32
    }
}

impl BitOr for ConeMask {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for ConeMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_edges() {
        assert_eq!(ConeMask::range(0, 0).0, 1);
        assert_eq!(ConeMask::range(63, 63).0, 1 << 63);
        assert_eq!(ConeMask::range(0, 63), ConeMask::FULL);
        assert_eq!(ConeMask::range(5, 4), ConeMask::EMPTY);
        assert_eq!(ConeMask::range(64, 70), ConeMask::EMPTY);
        assert_eq!(ConeMask::range(60, 99).blocked_bins(), 4);
    }

    #[test]
    fn interval_uses_bin_centers() {
        // Bin 0's center is at 0.5/64; an interval ending just before it sets nothing.
        assert_eq!(ConeMask::from_interval(0.0, 0.4 / 64.0), ConeMask::EMPTY);
        assert_eq!(ConeMask::from_interval(0.0, 0.6 / 64.0).0, 1);
        assert_eq!(ConeMask::from_interval(0.0, 1.0), ConeMask::FULL);
        assert_eq!(ConeMask::from_interval(0.5, 0.0).blocked_bins(), 32);
        assert_eq!(ConeMask::from_interval(-1.0, 2.0), ConeMask::FULL);
    }

    #[test]
    fn correlated_occluders_do_not_double_count() {
        let top_half = ConeMask::from_interval(0.0, 0.5);
        assert_eq!((top_half | top_half).visible_fraction(), 0.5);
        let bottom_half = ConeMask::from_interval(0.5, 1.0);
        assert_eq!((top_half | bottom_half).visible_fraction(), 0.0);
    }
}
