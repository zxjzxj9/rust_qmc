//! Wavefunction module - traits and basis functions for QMC calculations.

mod traits;
mod slater;

pub use traits::{SingleWfn, MultiWfn};
pub use slater::{STO, Slater1s, STOSlaterDet, init_li_sto};
