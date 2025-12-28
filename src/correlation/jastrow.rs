//! Jastrow correlation factors for QMC calculations.
//!
//! Jastrow factors capture electron-electron correlations that are
//! difficult to represent with single-particle orbitals.

use nalgebra::Vector3;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::wavefunction::MultiWfn;

/// Two-electron Jastrow factor: J(r₁₂) = exp(-F / (2(1 + r₁₂/F)))
///
/// Used for H₂ molecule and similar two-electron systems.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Jastrow1 {
    /// Correlation parameter controlling electron-electron cusp
    pub cusp_param: f64,
}

impl MultiWfn for Jastrow1 {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        vec![
            Vector3::<f64>::from_distribution(&dist, &mut rng),
            Vector3::<f64>::from_distribution(&dist, &mut rng),
        ]
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        let r12_norm = (r[0] - r[1]).norm();
        (-self.cusp_param / (2.0 * (1.0 + r12_norm / self.cusp_param))).exp()
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let r12 = r[0] - r[1];
        let r12_norm = r12.norm();
        let psi = self.evaluate(r);
        let denom = 2.0 * (1.0 + r12_norm / self.cusp_param).powi(2) * r12_norm;
        let grad_factor = psi / denom;
        vec![grad_factor * r12, -grad_factor * r12]
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let r12_norm = (r[0] - r[1]).norm();
        let factor = 1.0 + r12_norm / self.cusp_param;
        let psi = self.evaluate(r);
        let denom = 2.0 * factor.powi(2);
        let grad_square = denom.powi(-2);
        let laplacian_factor = 1.0 / (r12_norm * factor.powi(3));
        let comp = (grad_square + laplacian_factor) * psi;
        vec![comp, comp]
    }
}

/// Multi-electron Jastrow factor for N electrons.
///
/// J(r) = exp(-Σᵢ<ⱼ F / (2(1 + rᵢⱼ/F)))
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Jastrow2 {
    /// Correlation parameter
    pub cusp_param: f64,
    /// Number of electrons
    pub num_electrons: usize,
}

impl Jastrow2 {
    /// Generate all unique electron pairs (i, j) with i < j.
    fn unique_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.num_electrons).flat_map(move |i| {
            ((i + 1)..self.num_electrons).map(move |j| (i, j))
        })
    }
}

impl MultiWfn for Jastrow2 {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        (0..self.num_electrons)
            .map(|_| Vector3::<f64>::from_distribution(&dist, &mut rng))
            .collect()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        let sum: f64 = self.unique_pairs()
            .map(|(i, j)| {
                let r_ij_norm = (r[i] - r[j]).norm();
                -self.cusp_param / (2.0 * (1.0 + r_ij_norm / self.cusp_param))
            })
            .sum();
        sum.exp()
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let mut gradients = vec![Vector3::zeros(); self.num_electrons];
        let psi = self.evaluate(r);
        
        for (i, j) in self.unique_pairs() {
            let r_ij = r[i] - r[j];
            let r_ij_norm = r_ij.norm();
            let factor = self.cusp_param / (2.0 * (1.0 + r_ij_norm / self.cusp_param).powi(2) * r_ij_norm) * psi;
            let grad = factor * r_ij;
            gradients[i] += grad;
            gradients[j] -= grad;
        }
        gradients
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let mut laplacians = vec![0.0; self.num_electrons];
        let psi = self.evaluate(r);
        
        for (i, j) in self.unique_pairs() {
            let r_ij_norm = (r[i] - r[j]).norm();
            let factor = 1.0 + r_ij_norm / self.cusp_param;
            let denom = 2.0 * factor.powi(2);
            let grad_square = denom.powi(-2);
            let laplacian_factor = 1.0 / (r_ij_norm * factor.powi(3));
            let comp = (grad_square + laplacian_factor) * psi;
            laplacians[i] += comp;
            laplacians[j] += comp;
        }
        laplacians
    }
}
