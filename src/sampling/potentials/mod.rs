//! Potential-energy traits and concrete potentials for PIMC/PIMD.

pub mod traits;

pub use traits::{
    Potential, MolecularPotential,
    SplittablePotential, SplittableMolecularPotential,
};
