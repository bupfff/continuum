//! Continuum: certified incremental cone rendering.
//!
//! Every visibility question is asked by a cone carrying a direction mask
//! ([`ConeMask`]). Cached answers will carry an error ledger so they are only
//! recomputed when they could have changed by more than a chosen bound.
//! See `docs/DESIGN.md` for the full design.

pub mod mask;

pub use mask::ConeMask;
