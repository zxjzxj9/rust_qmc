//! Potential-energy traits and concrete potentials for PIMC/PIMD.

pub mod traits;
pub mod scalar;
pub mod molecular;
pub mod dftb;

pub use traits::{
    Potential, MolecularPotential,
    SplittablePotential, SplittableMolecularPotential,
};
pub use scalar::{
    HarmonicPotential, SombreroPotential, DoubleWellPotential, ProtonTransferPotential,
};
pub use molecular::{BifluoridePES, ZundelPES};
pub use dftb::ToyDFTB;
