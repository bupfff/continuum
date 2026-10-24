//! Error ledgers: conservative accounting of how stale a cached result may be.
//!
//! A cached result is valid when it was computed. Every later scene change that
//! could affect it charges the ledger a worst-case bound on how much that change
//! can move the result. By the triangle inequality the balance bounds the total
//! drift from a fresh recompute, so a result whose weighted balance stays under
//! ε can be reused with a guarantee instead of a heuristic.

/// Accumulated worst-case error of one cached result since it was last computed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ledger(f32);

impl Ledger {
    /// Adds a worst-case bound on how much one change can move the result.
    pub fn charge(&mut self, bound: f32) {
        debug_assert!(bound >= 0.0, "charges are magnitudes");
        self.0 += bound;
    }

    /// Guaranteed upper bound on |cached - fresh|, in the result's own units.
    pub fn balance(self) -> f32 {
        self.0
    }

    /// Call after recomputing the result.
    pub fn settle(&mut self) {
        self.0 = 0.0;
    }

    /// Whether the balance, scaled by how much the result contributes to the
    /// output (`weight`), exceeds the allowed error.
    pub fn exceeds(self, weight: f32, epsilon: f32) -> bool {
        self.0 * weight > epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_accumulate_until_settled() {
        let mut l = Ledger::default();
        l.charge(0.01);
        l.charge(0.02);
        assert!((l.balance() - 0.03).abs() < 1e-7);
        assert!(l.exceeds(1.0, 0.025));
        assert!(!l.exceeds(0.5, 0.025));
        l.settle();
        assert_eq!(l.balance(), 0.0);
    }

    #[test]
    fn zero_weight_never_needs_recompute() {
        let mut l = Ledger::default();
        l.charge(10.0);
        assert!(!l.exceeds(0.0, 0.0));
    }
}
