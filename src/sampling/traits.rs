//! Traits for Monte Carlo sampling.

use nalgebra::Vector3;

/// Trait for computing local energy from electron positions.
pub trait EnergyCalculator {
    fn local_energy(&self, positions: &[Vector3<f64>]) -> f64;
}

/// Trait for computing nuclear forces and supporting mutable geometry.
///
/// Forces are estimated via the Hellmann-Feynman theorem:
///   F_I = -∂E/∂R_I
///
/// The electrostatic (bare) HF force on nucleus I is:
///   F_I = Z_I Σ_i (r_i - R_I)/|r_i - R_I|³
///       - Σ_{J≠I} Z_I Z_J (R_I - R_J)/|R_I - R_J|³
///
/// VMC forces are obtained by averaging over sampled electron configurations.
pub trait ForceCalculator: EnergyCalculator {
    /// Number of nuclei.
    fn num_nuclei(&self) -> usize;

    /// Get current nuclear positions.
    fn get_nuclei(&self) -> Vec<Vector3<f64>>;

    /// Get nuclear charges.
    fn get_charges(&self) -> Vec<f64>;

    /// Set nuclear positions and rebuild internal state (basis, Jastrow, etc.).
    fn set_nuclei(&mut self, nuclei: &[Vector3<f64>]);

    /// Compute the Hellmann-Feynman force on each nucleus for a given
    /// electron configuration.
    ///
    /// Includes electron-nucleus attraction gradient and
    /// nuclear-nuclear repulsion gradient.
    fn hellmann_feynman_force(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>>;

    /// Compute ∂ ln|Ψ_T| / ∂R_I for each nucleus I.
    ///
    /// This is the derivative of the log-wavefunction with respect to
    /// nuclear positions. It is the key ingredient for the zero-variance
    /// (Assaraf-Caffarel) force estimator.
    ///
    /// For a Slater-Jastrow wavefunction Ψ = D × J:
    ///   ∂ ln|Ψ| / ∂R_I = ∂ ln|D| / ∂R_I + ∂u / ∂R_I
    ///
    /// Default implementation panics — override for ZV force support.
    fn wfn_nuclear_gradient(&self, _r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        unimplemented!(
            "wfn_nuclear_gradient not implemented for this system. \
             Required for zero-variance force estimation."
        )
    }
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
