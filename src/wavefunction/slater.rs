//! Slater-Type Orbital (STO) and Slater 1s basis functions.

use nalgebra::{DMatrix, Vector3};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use super::traits::{MultiWfn, SingleWfn};

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

/// Slater-Type Orbital (STO) basis function.
///
/// ψₙ(r) = Σᵥ φᵥₙ r^pᵥ exp(-ζᵥ r)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct STO {
    /// Number of primitive functions
    pub num_primitives: usize,
    /// Contraction coefficients
    pub coefficients: Vec<f64>,
    /// Radial powers
    pub powers: Vec<i32>,
    /// Orbital exponents
    pub exponents: Vec<f64>,
    /// Center position
    pub center: Vector3<f64>,
}

/// Create an STO for Lithium atom.
///
/// # Arguments
/// * `center` - Position of the nucleus
/// * `n` - Principal quantum number (1 or 2)
/// * `_l` - Angular momentum quantum number (unused, for future extension)
/// * `_m` - Magnetic quantum number (unused, for future extension)
pub fn init_li_sto(center: Vector3<f64>, n: i32, _l: i32, _m: i32) -> STO {
    // Optimized STO-7G basis for Lithium
    let exponents = vec![
        0.72089388, 2.61691643, 0.69257443, 1.37137558, 
        3.97864549, 13.52900016, 19.30801440
    ];
    let powers = vec![1, 1, 2, 2, 2, 2, 3];
    
    let coefficients = match n {
        1 => vec![-0.12220686, 1.11273225, 0.04125378, 0.09306499, -0.10260021, -0.00034191, 0.00021963],
        2 => vec![0.47750469, 0.11140449, -1.25954273, -0.18475003, -0.02736293, -0.00025064, 0.00057962],
        _ => vec![0.0; 7],
    };
    
    STO {
        num_primitives: 7,
        coefficients,
        powers,
        exponents,
        center,
    }
}

impl SingleWfn for STO {
    fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        let s = (r - self.center).norm();
        self.coefficients.iter()
            .zip(self.powers.iter())
            .zip(self.exponents.iter())
            .map(|((&c, &p), &z)| c * s.powi(p) * (-z * s).exp())
            .sum()
    }

    fn derivative(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let s = (r - self.center).norm();
        if s == 0.0 {
            return Vector3::zeros();
        }
        
        let radial_dir = (r - self.center) / s;
        let radial_deriv: f64 = self.coefficients.iter()
            .zip(self.powers.iter())
            .zip(self.exponents.iter())
            .map(|((&c, &p), &z)| {
                let pf = p as f64;
                let exp_factor = (-z * s).exp();
                let s_p = s.powi(p);
                let s_p_m1 = s_p / s;
                c * exp_factor * (-z * s_p + pf * s_p_m1)
            })
            .sum();
        
        radial_dir * radial_deriv
    }

    fn laplacian(&self, r: &Vector3<f64>) -> f64 {
        let s = (r - self.center).norm();
        if s == 0.0 {
            return 0.0;
        }
        
        self.coefficients.iter()
            .zip(self.powers.iter())
            .zip(self.exponents.iter())
            .map(|((&c, &p), &z)| {
                let pf = p as f64;
                let exp_factor = (-z * s).exp();
                let s_p = s.powi(p);
                let s_p_m1 = s_p / s;
                let s_p_m2 = s_p_m1 / s;
                
                let d2_radial = z * z * s_p - 2.0 * z * pf * s_p_m1 + pf * (pf - 1.0) * s_p_m2;
                let d1_radial = 2.0 * (-z * s_p + pf * s_p_m1) / s;
                c * exp_factor * (d2_radial + d1_radial)
            })
            .sum()
    }
}

/// Slater determinant constructed from STOs.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct STOSlaterDet {
    /// Number of electrons
    pub num_electrons: usize,
    /// STO basis functions (one per electron)
    pub orbitals: Vec<STO>,
    /// Spin of each electron (+1 or -1)
    pub spins: Vec<i32>,
    /// Slater matrix (cached)
    #[serde(skip)]
    pub slater_matrix: DMatrix<f64>,
    /// Inverse Slater matrix (cached)
    #[serde(skip)]
    pub inv_slater_matrix: DMatrix<f64>,
}

impl STOSlaterDet {
    /// Create a new Slater determinant.
    pub fn new(orbitals: Vec<STO>, spins: Vec<i32>) -> Self {
        let n = orbitals.len();
        Self {
            num_electrons: n,
            orbitals,
            spins,
            slater_matrix: DMatrix::zeros(n, n),
            inv_slater_matrix: DMatrix::zeros(n, n),
        }
    }

    /// Update the Slater matrix and its inverse for given positions.
    #[allow(dead_code)]
    fn update_matrices(&mut self, r: &[Vector3<f64>]) {
        let n = self.num_electrons;
        self.slater_matrix = DMatrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                if self.spins[i] == self.spins[j] {
                    self.slater_matrix[(i, j)] = self.orbitals[j].evaluate(&r[i]);
                }
            }
        }
        
        self.inv_slater_matrix = self.slater_matrix.clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(n, n));
    }
}

impl MultiWfn for STOSlaterDet {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        (0..self.num_electrons)
            .map(|_| Vector3::<f64>::from_distribution(&dist, &mut rng))
            .collect()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        let n = self.num_electrons;
        let mut s = DMatrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                if self.spins[i] == self.spins[j] {
                    s[(i, j)] = self.orbitals[j].evaluate(&r[i]);
                }
            }
        }
        s.determinant()
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = self.num_electrons;
        let psi = self.evaluate(r);
        
        // Build and invert Slater matrix
        let mut s = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if self.spins[i] == self.spins[j] {
                    s[(i, j)] = self.orbitals[j].evaluate(&r[i]);
                }
            }
        }
        let inv_s = s.try_inverse().unwrap_or_else(|| DMatrix::identity(n, n));
        
        (0..n)
            .map(|i| {
                let grad: Vector3<f64> = (0..n)
                    .filter(|&j| self.spins[i] == self.spins[j])
                    .map(|j| inv_s[(j, i)] * self.orbitals[j].derivative(&r[i]))
                    .sum();
                psi * grad
            })
            .collect()
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let n = self.num_electrons;
        let psi = self.evaluate(r);
        
        // Build and invert Slater matrix
        let mut s = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if self.spins[i] == self.spins[j] {
                    s[(i, j)] = self.orbitals[j].evaluate(&r[i]);
                }
            }
        }
        let inv_s = s.try_inverse().unwrap_or_else(|| DMatrix::identity(n, n));
        
        (0..n)
            .map(|i| {
                let lap_sum: f64 = (0..n)
                    .filter(|&j| self.spins[i] == self.spins[j])
                    .map(|j| inv_s[(j, i)] * self.orbitals[j].laplacian(&r[i]))
                    .sum();
                psi * lap_sum
            })
            .collect()
    }
}
