//! Simple QMC implementation for the H₂ molecule
//!
//! Supports both Valence Bond (VB) and Molecular Orbital (MO) wavefunctions.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use crate::jastrow::Jastrow1;
use crate::mcmc::EnergyCalculator;
use crate::wfn::{MultiWfn, SingleWfn};

/// Slater 1s orbital centered at position `center` with exponent `alpha`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Slater1s {
    /// Orbital exponent
    pub alpha: f64,
    /// Center position of the orbital
    pub center: Vector3<f64>,
}

impl SingleWfn for Slater1s {
    fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        let dr = r - self.center;
        (-self.alpha * dr.norm()).exp()
    }

    fn derivative(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let dr = r - self.center;
        let r_norm = dr.norm();
        if r_norm == 0.0 {
            return Vector3::zeros();
        }
        let scalar = -self.alpha / r_norm * (-self.alpha * r_norm).exp();
        dr * scalar
    }

    fn laplacian(&self, r: &Vector3<f64>) -> f64 {
        let dr = r - self.center;
        let r_norm = dr.norm();
        if r_norm == 0.0 {
            return f64::NEG_INFINITY;
        }
        let exp_part = (-self.alpha * r_norm).exp();
        (self.alpha.powi(2) - 2.0 * self.alpha / r_norm) * exp_part
    }
}

// Re-export Jastrow1 for backwards compatibility
pub use crate::jastrow::Jastrow1;

/// H₂ molecule with Valence Bond (VB) wavefunction.
///
/// The VB wavefunction is: Ψ = (φ₁(r₁)φ₂(r₁)) × (φ₁(r₂)φ₂(r₂)) × J(r₁₂)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct H2MoleculeVB {
    /// First hydrogen atom orbital
    pub orbital1: Slater1s,
    /// Second hydrogen atom orbital
    pub orbital2: Slater1s,
    /// Jastrow correlation factor
    pub jastrow: Jastrow1,
}

/// H₂ molecule with Molecular Orbital (MO/LCAO) wavefunction.
///
/// The MO wavefunction is: Ψ = (φ₁(r₁)+φ₂(r₁)) × (φ₁(r₂)+φ₂(r₂)) × J(r₁₂)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct H2MoleculeMO {
    /// First hydrogen atom orbital
    pub orbital1: Slater1s,
    /// Second hydrogen atom orbital  
    pub orbital2: Slater1s,
    /// Jastrow correlation factor
    pub jastrow: Jastrow1,
}

impl MultiWfn for H2MoleculeVB {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        self.jastrow.initialize()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        let psi_1 = self.orbital1.evaluate(&r[0]) * self.orbital2.evaluate(&r[0]);
        let psi_2 = self.orbital1.evaluate(&r[1]) * self.orbital2.evaluate(&r[1]);
        let j = self.jastrow.evaluate(r);
        psi_1 * psi_2 * j
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let psi_1 = self.orbital1.evaluate(&r[0]) * self.orbital2.evaluate(&r[0]);
        let psi_2 = self.orbital1.evaluate(&r[1]) * self.orbital2.evaluate(&r[1]);
        let j = self.jastrow.evaluate(r);

        // Derivatives at r[0]
        let h1_eval_r0 = self.orbital1.evaluate(&r[0]);
        let h2_eval_r0 = self.orbital2.evaluate(&r[0]);
        let h1_deriv_r0 = self.orbital1.derivative(&r[0]);
        let h2_deriv_r0 = self.orbital2.derivative(&r[0]);
        let psi_1_derivative = h1_deriv_r0 * h2_eval_r0 + h1_eval_r0 * h2_deriv_r0;

        // Derivatives at r[1]
        let h1_eval_r1 = self.orbital1.evaluate(&r[1]);
        let h2_eval_r1 = self.orbital2.evaluate(&r[1]);
        let h1_deriv_r1 = self.orbital1.derivative(&r[1]);
        let h2_deriv_r1 = self.orbital2.derivative(&r[1]);
        let psi_2_derivative = h1_deriv_r1 * h2_eval_r1 + h1_eval_r1 * h2_deriv_r1;

        let j_derivative = self.jastrow.derivative(r);

        vec![
            psi_2 * (psi_1_derivative * j + psi_1 * j_derivative[0]),
            psi_1 * (psi_2_derivative * j + psi_2 * j_derivative[1]),
        ]
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let psi_1 = self.orbital1.evaluate(&r[0]) * self.orbital2.evaluate(&r[0]);
        let psi_2 = self.orbital1.evaluate(&r[1]) * self.orbital2.evaluate(&r[1]);
        let j = self.jastrow.evaluate(r);

        // Derivatives and Laplacians at r[0]
        let h1_eval_r0 = self.orbital1.evaluate(&r[0]);
        let h2_eval_r0 = self.orbital2.evaluate(&r[0]);
        let h1_deriv_r0 = self.orbital1.derivative(&r[0]);
        let h2_deriv_r0 = self.orbital2.derivative(&r[0]);
        let h1_lap_r0 = self.orbital1.laplacian(&r[0]);
        let h2_lap_r0 = self.orbital2.laplacian(&r[0]);
        let psi_1_derivative = h1_deriv_r0 * h2_eval_r0 + h1_eval_r0 * h2_deriv_r0;
        let psi_1_laplacian = h1_lap_r0 * h2_eval_r0
            + 2.0 * h1_deriv_r0.dot(&h2_deriv_r0)
            + h1_eval_r0 * h2_lap_r0;

        // Derivatives and Laplacians at r[1]
        let h1_eval_r1 = self.orbital1.evaluate(&r[1]);
        let h2_eval_r1 = self.orbital2.evaluate(&r[1]);
        let h1_deriv_r1 = self.orbital1.derivative(&r[1]);
        let h2_deriv_r1 = self.orbital2.derivative(&r[1]);
        let h1_lap_r1 = self.orbital1.laplacian(&r[1]);
        let h2_lap_r1 = self.orbital2.laplacian(&r[1]);
        let psi_2_derivative = h1_deriv_r1 * h2_eval_r1 + h1_eval_r1 * h2_deriv_r1;
        let psi_2_laplacian = h1_lap_r1 * h2_eval_r1
            + 2.0 * h1_deriv_r1.dot(&h2_deriv_r1)
            + h1_eval_r1 * h2_lap_r1;

        let j_derivative = self.jastrow.derivative(r);
        let j_laplacian = self.jastrow.laplacian(r);

        vec![
            psi_2 * (psi_1_laplacian * j + 2.0 * psi_1_derivative.dot(&j_derivative[0]) + psi_1 * j_laplacian[0]),
            psi_1 * (psi_2_laplacian * j + 2.0 * psi_2_derivative.dot(&j_derivative[1]) + psi_2 * j_laplacian[1]),
        ]
    }
}

impl MultiWfn for H2MoleculeMO {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        self.jastrow.initialize()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        let psi_1 = self.orbital1.evaluate(&r[0]) + self.orbital2.evaluate(&r[0]);
        let psi_2 = self.orbital1.evaluate(&r[1]) + self.orbital2.evaluate(&r[1]);
        psi_1 * psi_2 * self.jastrow.evaluate(r)
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let psi_sum_r1 = self.orbital1.evaluate(&r[0]) + self.orbital2.evaluate(&r[0]);
        let psi_sum_r2 = self.orbital1.evaluate(&r[1]) + self.orbital2.evaluate(&r[1]);
        let grad_sum_r1 = self.orbital1.derivative(&r[0]) + self.orbital2.derivative(&r[0]);
        let grad_sum_r2 = self.orbital1.derivative(&r[1]) + self.orbital2.derivative(&r[1]);

        let j = self.jastrow.evaluate(r);
        let grad_j = self.jastrow.derivative(r);

        vec![
            grad_sum_r1 * psi_sum_r2 * j + psi_sum_r1 * psi_sum_r2 * grad_j[0],
            grad_sum_r2 * psi_sum_r1 * j + psi_sum_r1 * psi_sum_r2 * grad_j[1],
        ]
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let psi_sum_r1 = self.orbital1.evaluate(&r[0]) + self.orbital2.evaluate(&r[0]);
        let psi_sum_r2 = self.orbital1.evaluate(&r[1]) + self.orbital2.evaluate(&r[1]);
        let grad_sum_r1 = self.orbital1.derivative(&r[0]) + self.orbital2.derivative(&r[0]);
        let grad_sum_r2 = self.orbital1.derivative(&r[1]) + self.orbital2.derivative(&r[1]);
        let lap_sum_r1 = self.orbital1.laplacian(&r[0]) + self.orbital2.laplacian(&r[0]);
        let lap_sum_r2 = self.orbital1.laplacian(&r[1]) + self.orbital2.laplacian(&r[1]);

        let j = self.jastrow.evaluate(r);
        let grad_j = self.jastrow.derivative(r);
        let lap_j = self.jastrow.laplacian(r);

        vec![
            lap_sum_r1 * psi_sum_r2 * j
                + 2.0 * grad_sum_r1.dot(&grad_j[0]) * psi_sum_r2
                + psi_sum_r1 * psi_sum_r2 * lap_j[0],
            lap_sum_r2 * psi_sum_r1 * j
                + 2.0 * grad_sum_r2.dot(&grad_j[1]) * psi_sum_r1
                + psi_sum_r1 * psi_sum_r2 * lap_j[1],
        ]
    }
}

impl EnergyCalculator for H2MoleculeVB {
    fn local_energy(&self, positions: &[Vector3<f64>]) -> f64 {
        let r1 = &positions[0];
        let r2 = &positions[1];
        let r12_norm = (r1 - r2).norm();
        let psi = self.evaluate(positions);
        let lap: f64 = self.laplacian(positions).into_iter().sum();

        let kinetic = -0.5 * lap / psi;
        let potential = 1.0 / r12_norm
            - 1.0 / (r1 - self.orbital1.center).norm()
            - 1.0 / (r2 - self.orbital2.center).norm()
            - 1.0 / (r1 - self.orbital2.center).norm()
            - 1.0 / (r2 - self.orbital1.center).norm()
            + 1.0 / (self.orbital1.center - self.orbital2.center).norm();
        kinetic + potential
    }
}

impl EnergyCalculator for H2MoleculeMO {
    fn local_energy(&self, positions: &[Vector3<f64>]) -> f64 {
        let r1 = &positions[0];
        let r2 = &positions[1];
        let r12_norm = (r1 - r2).norm();
        let psi = self.evaluate(positions);
        let lap: f64 = self.laplacian(positions).into_iter().sum();

        let kinetic = -0.5 * lap / psi;
        let potential = 1.0 / r12_norm
            - 1.0 / (r1 - self.orbital1.center).norm()
            - 1.0 / (r2 - self.orbital2.center).norm()
            - 1.0 / (r1 - self.orbital2.center).norm()
            - 1.0 / (r2 - self.orbital1.center).norm()
            + 1.0 / (self.orbital1.center - self.orbital2.center).norm();
        kinetic + potential
    }
}