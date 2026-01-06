//! Sampling module - Monte Carlo sampling methods for QMC.

mod traits;
mod vmc;
mod dmc;
mod pimc;

pub use traits::{EnergyCalculator, Walker, BranchingResult, VmcWalker};
pub use vmc::{MCMCParams, MCMCState, MCMCResults, MCMCSimulation};
pub use dmc::{run_dmc_sampling, HarmonicWalker, HydrogenAtomWalker, HydrogenMoleculeWalker};
pub use pimc::{
    QuantumPath, PIMCSimulation, run_pimc_harmonic,
    // Generalized potentials
    Potential, HarmonicPotential, SombreroPotential, DoubleWellPotential,
    GeneralPath, GeneralPIMC, run_pimc_sombrero,
};
