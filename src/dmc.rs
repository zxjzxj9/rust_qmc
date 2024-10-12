
use nalgebra::{DMatrix, Matrix, Vector3};
use rand_distr::Normal;

// Define a trait for walker behavior
pub trait Walker {
    // Move the walker to a new position
    fn move_walker(&mut self);

    // Calculate local properties like energy
    fn calculate_local_energy(&mut self);

    // Update the walker's weight
    fn update_weight(&mut self);

    // Decide whether to branch (clone) or die
    fn branching_decision(&self) -> BranchingResult;

    // Clone the walker (for branching)
    fn clone_walker(&self) -> Self
    where
        Self: Sized;

    // Check if the walker should be deleted
    fn should_be_deleted(&self) -> bool;

    // Mark the walker for deletion
    fn mark_for_deletion(&mut self);
}

// Define an enum for branching decisions
pub enum BranchingResult {
    Clone, // The walker should be cloned
    Keep,  // The walker continues as is
    Kill,  // The walker should be removed
}

