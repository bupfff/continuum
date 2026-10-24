//! Continuum: certified incremental cone rendering.
//!
//! Every visibility question is asked by a cone carrying a direction mask
//! ([`ConeMask`]). Cached answers carry an error [`Ledger`] so they are only
//! recomputed when they could have changed by more than a chosen bound.
//! See `docs/DESIGN.md` for the full design.

pub mod ledger;
pub mod mask;

pub use ledger::Ledger;
pub use mask::ConeMask;
