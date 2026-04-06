//! Force variance reduction methods for VMC.
//!
//! The bare Hellmann-Feynman (HF) force estimator suffers from infinite variance
//! due to the 1/r² singularity at electron-nucleus coalescence. This module
//! provides variance-reduced estimators following established QMC literature.
//!
//! # Zero-Variance (ZV) Estimator
//!
//! The Assaraf-Caffarel zero-variance principle constructs a renormalized
//! force estimator by adding a Pulay correction term:
//!
//! ```text
//! F_I^ZV = F_I^HF - 2(E_L - E_ref) × ∂ln|Ψ_T|/∂R_I
//! ```
//!
//! The Pulay term `(E_L - E_ref) × ∂ln|Ψ|/∂R_I` has zero expectation value
//! but is correlated with the HF force fluctuations, acting as a control
//! variate. As Ψ_T → Ψ_exact, the variance of F^ZV → 0.
//!
//! # References
//!
//! - R. Assaraf & M. Caffarel, J. Chem. Phys. 113, 4028 (2000)
//! - R. Assaraf & M. Caffarel, J. Chem. Phys. 119, 10536 (2003)

/// Selects which force estimator to use in VMC sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceEstimator {
    /// Bare Hellmann-Feynman force estimator.
    ///
    /// Simple electrostatic force, high variance due to 1/r² divergence.
    Bare,

    /// Zero-variance (Assaraf-Caffarel) force estimator.
    ///
    /// Adds a Pulay correction using wavefunction nuclear derivatives.
    /// Requires `ForceCalculator::wfn_nuclear_gradient()` implementation.
    /// Dramatically reduces variance at the cost of computing ∂ln|Ψ|/∂R_I.
    ZeroVariance,
}

impl Default for ForceEstimator {
    fn default() -> Self {
        ForceEstimator::Bare
    }
}

impl std::fmt::Display for ForceEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForceEstimator::Bare => write!(f, "Bare Hellmann-Feynman"),
            ForceEstimator::ZeroVariance => write!(f, "Zero-Variance (Assaraf-Caffarel)"),
        }
    }
}
