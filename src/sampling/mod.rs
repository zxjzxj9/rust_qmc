//! Sampling module - Monte Carlo sampling methods for QMC.

// --- Shared building blocks ---
mod traits;
pub mod potentials;
mod pimd_core;

// --- Electronic-structure Monte Carlo ---
mod vmc;
mod dmc;
mod is_dmc;
mod optimize;
mod sr_optimize;
pub mod geometry_opt;
pub mod force_variance;

// --- Path-integral methods ---
mod pimc;
mod pimc_fermion;
mod pimd;
mod pimd_molecular;
mod pimd_rpc;
mod piglet;

// --- Enhanced sampling / free energy ---
mod enhanced;

// ======================================================================
// Public re-exports
// ======================================================================

// Core traits
pub use traits::{EnergyCalculator, ForceCalculator, Walker, BranchingResult, VmcWalker};

// Potentials (traits + concrete surfaces)
pub use potentials::{
    Potential, MolecularPotential, SplittablePotential, SplittableMolecularPotential,
    HarmonicPotential, SombreroPotential, DoubleWellPotential, ProtonTransferPotential,
    BifluoridePES, ZundelPES, ToyDFTB,
};

// Ring-polymer core (normal modes + thermostat)
pub use pimd_core::{NormalModeTransform, PILEThermostat};

// Electronic-structure Monte Carlo
pub use vmc::{MCMCParams, MCMCState, MCMCResults, MCMCSimulation,
              DDVMCParams, DDVMCResults, DriftDiffusionVMC};
pub use dmc::{run_dmc_sampling, HarmonicWalker, HydrogenAtomWalker, HydrogenMoleculeWalker};
pub use is_dmc::{ISDMCParams, ISDMCResults, ImportanceSampledDMC};
pub use optimize::{JastrowOptimizer, OptimizationResult, SamplingStats};
pub use sr_optimize::{SROptimizer, SRResult};
pub use geometry_opt::{GeometryOptimizer, GeometryOptResult};
pub use force_variance::ForceEstimator;

// Path-integral Monte Carlo
pub use pimc::{
    QuantumPath, PIMCSimulation, run_pimc_harmonic,
    GeneralPath, GeneralPIMC, run_pimc_sombrero,
};
pub use pimc_fermion::{
    TrialWavefunction, Hydrogen1s, FermionPath, FermionPIMC, run_pimc_hydrogen,
};

// Path-integral molecular dynamics
pub use pimd::{RingPolymer, PIMDSimulation, run_pimd_proton_transfer};
pub use pimd_molecular::{
    MolecularRingPolymer, MolecularPILE, MolecularPIMD,
    run_pimd_bifluoride, run_pimd_zundel, free_energy_profile,
};
pub use pimd_rpc::{
    RPContraction, SplittableDoubleWell, RPCRingPolymer, RPCSimulation,
};
pub use piglet::{
    PIQTBThermostat, MolecularPIQTB, PIGLETThermostat,
    matrix_exponential, cholesky_decompose, load_piglet_matrices,
};

// Enhanced sampling / free energy
pub use enhanced::{
    UmbrellaBias, BiasedMolecularPotential, UmbrellaWindow, WHAMSolver,
    run_pimd_umbrella_sampling, run_zundel_umbrella_sampling,
    GaussianHill, MetadynamicsBias, MetadynamicsPotential, MetadynamicsResult,
    run_pimd_metadynamics, run_zundel_metadynamics,
};
