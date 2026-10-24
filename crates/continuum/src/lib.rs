//! Continuum: certified incremental cone rendering.
//!
//! Every visibility question is asked by a cone carrying a direction mask
//! ([`ConeMask`], or [`CoverageMask`] when visibility must be continuous).
//! Cached answers carry an error [`Ledger`] so they are only recomputed when
//! they could have changed by more than a chosen bound.
//! See `docs/DESIGN.md` for the full design.

pub mod bvh;
pub mod cone3;
pub mod coverage;
pub mod ledger;
pub mod mask;

pub use coverage::CoverageMask;
pub use glam;
pub use ledger::Ledger;
pub use mask::ConeMask;
