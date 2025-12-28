//! Traits for Monte Carlo sampling.

use nalgebra::Vector3;

/// Trait for computing local energy from electron positions.
pub trait EnergyCalculator {
    fn local_energy(&self, positions: &[Vector3<f64>]) -> f64;
}

/// Define an enum for branching decisions (DMC)
pub enum BranchingResult {
    Clone { n: usize }, // n is the number of clones
    Keep,               // The walker continues as is
    Kill,               // The walker should be removed
}

/// Define a trait for DMC walker behavior
pub trait Walker {
    fn new(dt: f64, eref: f64) -> Self;
    /// Move the walker to a new position
    fn move_walker(&mut self);
    /// Calculate local properties like energy
    fn calculate_local_energy(&mut self);
    /// Get the local energy
    fn local_energy(&self) -> f64;
    /// Update the walker's weight
    fn update_weight(&mut self, e_ref: f64);
    /// Decide whether to branch (clone) or die
    fn branching_decision(&mut self) -> BranchingResult;
    /// Check if the walker should be deleted
    fn should_be_deleted(&self) -> bool;
    /// Mark the walker for deletion
    fn mark_for_deletion(&mut self);
}

/// Trait for VMC walker behavior
pub trait VmcWalker: Sized {
    fn new() -> Self;
    fn move_walker(&mut self) -> (bool, f64);
    fn calculate_local_energy(&mut self) -> f64;
    fn get_positions(&self) -> &Vec<Vector3<f64>>;
}
