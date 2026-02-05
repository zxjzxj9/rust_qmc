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

/// Advanced Jastrow factor with both electron-electron and electron-nucleus terms.
///
/// J(r) = exp(u_ee + u_en)
/// 
/// where:
/// - u_ee = Σᵢ<ⱼ a_σ rᵢⱼ / (1 + b rᵢⱼ)  (electron-electron, spin-dependent)
/// - u_en = Σᵢ,I -Z_I aₑₙ rᵢI / (1 + bₑₙ rᵢI)  (electron-nucleus)
///
/// The e-n term satisfies the Kato cusp condition when aₑₙ = Z.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Jastrow3 {
    /// Number of electrons
    pub num_electrons: usize,
    /// Electron-electron cusp parameter (a) for antiparallel spins
    pub a_ee_anti: f64,
    /// Electron-electron cusp parameter (a) for parallel spins
    pub a_ee_para: f64,
    /// Electron-electron decay parameter (b)
    pub b_ee: f64,
    /// Electron-nucleus decay parameter (b)
    pub b_en: f64,
    /// Nucleus positions
    pub nuclei: Vec<Vector3<f64>>,
    /// Nuclear charges
    pub charges: Vec<f64>,
    /// Spin assignments (+1 or -1)
    pub spins: Vec<i32>,
}

impl Jastrow3 {
    /// Create a new Jastrow3 factor for CH4.
    pub fn new_ch4(b_ee: f64, b_en: f64) -> Self {
        // CH4 geometry
        let d = 2.05 / (3.0_f64).sqrt();
        let nuclei = vec![
            Vector3::zeros(),              // Carbon
            Vector3::new(d, d, d),         // H1
            Vector3::new(d, -d, -d),       // H2
            Vector3::new(-d, d, -d),       // H3
            Vector3::new(-d, -d, d),       // H4
        ];
        let charges = vec![6.0, 1.0, 1.0, 1.0, 1.0];
        
        // 5 spin-up, 5 spin-down
        let spins = vec![1, 1, 1, 1, 1, -1, -1, -1, -1, -1];
        
        Self {
            num_electrons: 10,
            // Antiparallel cusp: 1/2 (opposite spin electrons)
            a_ee_anti: 0.5,
            // Parallel cusp: 1/4 (same spin electrons)
            a_ee_para: 0.25,
            b_ee,
            b_en,
            nuclei,
            charges,
            spins,
        }
    }
    
    /// u_ee contribution: electron-electron correlation
    fn u_ee(&self, r: &[Vector3<f64>]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.num_electrons {
            for j in (i + 1)..self.num_electrons {
                let r_ij = (r[i] - r[j]).norm();
                // Spin-dependent cusp parameter
                let a = if self.spins[i] == self.spins[j] {
                    self.a_ee_para
                } else {
                    self.a_ee_anti
                };
                sum += a * r_ij / (1.0 + self.b_ee * r_ij);
            }
        }
        sum
    }
    
    /// u_en contribution: electron-nucleus correlation
    fn u_en(&self, r: &[Vector3<f64>]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.num_electrons {
            for (n, nuc) in self.nuclei.iter().enumerate() {
                let r_in = (r[i] - nuc).norm();
                // Kato cusp: coefficient proportional to Z
                let a_en = self.charges[n];
                sum -= a_en * r_in / (1.0 + self.b_en * r_in);
            }
        }
        sum
    }
    
    /// Gradient of u_ee with respect to electron i
    fn grad_u_ee(&self, r: &[Vector3<f64>], i: usize) -> Vector3<f64> {
        let mut grad = Vector3::zeros();
        for j in 0..self.num_electrons {
            if i == j { continue; }
            let r_ij_vec = r[i] - r[j];
            let r_ij = r_ij_vec.norm();
            if r_ij < 1e-10 { continue; }
            
            let a = if self.spins[i] == self.spins[j] {
                self.a_ee_para
            } else {
                self.a_ee_anti
            };
            let denom = 1.0 + self.b_ee * r_ij;
            // d/dr_i [a * r_ij / (1 + b * r_ij)]
            let factor = a / (denom * denom * r_ij);
            grad += factor * r_ij_vec;
        }
        grad
    }
    
    /// Gradient of u_en with respect to electron i
    fn grad_u_en(&self, r: &[Vector3<f64>], i: usize) -> Vector3<f64> {
        let mut grad = Vector3::zeros();
        for (n, nuc) in self.nuclei.iter().enumerate() {
            let r_in_vec = r[i] - nuc;
            let r_in = r_in_vec.norm();
            if r_in < 1e-10 { continue; }
            
            let a_en = self.charges[n];
            let denom = 1.0 + self.b_en * r_in;
            // d/dr_i [-a_en * r_in / (1 + b_en * r_in)]
            let factor = -a_en / (denom * denom * r_in);
            grad += factor * r_in_vec;
        }
        grad
    }
    
    /// Laplacian of u_ee with respect to electron i
    fn lap_u_ee(&self, r: &[Vector3<f64>], i: usize) -> f64 {
        let mut lap = 0.0;
        for j in 0..self.num_electrons {
            if i == j { continue; }
            let r_ij = (r[i] - r[j]).norm();
            if r_ij < 1e-10 { continue; }
            
            let a = if self.spins[i] == self.spins[j] {
                self.a_ee_para
            } else {
                self.a_ee_anti
            };
            let denom = 1.0 + self.b_ee * r_ij;
            // ∇² [a * r / (1 + br)] = 2a / (r(1+br)²) - 2ab / (1+br)³
            lap += 2.0 * a / (r_ij * denom * denom) - 2.0 * a * self.b_ee / (denom * denom * denom);
        }
        lap
    }
    
    /// Laplacian of u_en with respect to electron i
    fn lap_u_en(&self, r: &[Vector3<f64>], i: usize) -> f64 {
        let mut lap = 0.0;
        for (n, nuc) in self.nuclei.iter().enumerate() {
            let r_in = (r[i] - nuc).norm();
            if r_in < 1e-10 { continue; }
            
            let a_en = self.charges[n];
            let denom = 1.0 + self.b_en * r_in;
            // ∇² [-a * r / (1 + br)] = -2a / (r(1+br)²) + 2ab / (1+br)³
            lap += -2.0 * a_en / (r_in * denom * denom) + 2.0 * a_en * self.b_en / (denom * denom * denom);
        }
        lap
    }
}

impl MultiWfn for Jastrow3 {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 0.5).unwrap();
        use rand_distr::Distribution;
        
        (0..self.num_electrons)
            .map(|i| {
                // Place electrons near nuclei
                let center = if i < 6 {
                    self.nuclei[0] // near Carbon
                } else {
                    self.nuclei[i - 5] // near H atoms
                };
                center + Vector3::new(
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                )
            })
            .collect()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        (self.u_ee(r) + self.u_en(r)).exp()
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let j = self.evaluate(r);
        (0..self.num_electrons)
            .map(|i| j * (self.grad_u_ee(r, i) + self.grad_u_en(r, i)))
            .collect()
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let j = self.evaluate(r);
        (0..self.num_electrons)
            .map(|i| {
                let grad_u = self.grad_u_ee(r, i) + self.grad_u_en(r, i);
                let lap_u = self.lap_u_ee(r, i) + self.lap_u_en(r, i);
                // ∇²J = J * (∇²u + |∇u|²)
                j * (lap_u + grad_u.norm_squared())
            })
            .collect()
    }
}

