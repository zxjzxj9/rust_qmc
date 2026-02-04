//! Methane (CH4) molecule wavefunction for QMC calculations.
//!
//! Implements a Slater-Jastrow trial wavefunction for methane with 10 electrons
//! (6 from Carbon, 4 from Hydrogen atoms) in a tetrahedral geometry.

use nalgebra::Vector3;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::correlation::Jastrow2;
use crate::sampling::EnergyCalculator;
use crate::wavefunction::{MultiWfn, SingleWfn};

/// Tetrahedral bond length in Bohr (1.087 Å ≈ 2.05 Bohr)
const CH_BOND_LENGTH: f64 = 2.05;

/// STO-3G contracted Gaussian (approximated as STO for simplicity)
/// For a more accurate calculation, one could use GTO basis sets.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MethaneOrbitals {
    /// Orbital centers (nucleus positions)
    pub centers: Vec<Vector3<f64>>,
    /// Orbital exponents (zeta values)
    pub exponents: Vec<f64>,
    /// Orbital types: 0 = 1s, 1 = 2s, 2 = 2px, 3 = 2py, 4 = 2pz
    pub orbital_types: Vec<u8>,
    /// Molecular orbital coefficients (MO index, AO index)
    pub mo_coefficients: Vec<Vec<f64>>,
}

/// Simple STO-like basis for CH4.
/// 
/// For Carbon: 1s (core), 2s, 2px, 2py, 2pz
/// For each Hydrogen: 1s
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CH4Basis {
    /// Carbon position (origin)
    pub carbon: Vector3<f64>,
    /// Hydrogen positions (4 atoms in tetrahedral arrangement)
    pub hydrogens: [Vector3<f64>; 4],
    /// Carbon 1s exponent
    pub c_1s_exp: f64,
    /// Carbon 2s exponent  
    pub c_2s_exp: f64,
    /// Carbon 2p exponent
    pub c_2p_exp: f64,
    /// Hydrogen 1s exponent
    pub h_1s_exp: f64,
}

impl CH4Basis {
    /// Create a new CH4 basis with default STO-3G-like exponents.
    pub fn new() -> Self {
        // Tetrahedral geometry: C at origin, H at corners of tetrahedron
        let d = CH_BOND_LENGTH / (3.0_f64).sqrt(); // ~1.185 Bohr
        let hydrogens = [
            Vector3::new(d, d, d),
            Vector3::new(d, -d, -d),
            Vector3::new(-d, d, -d),
            Vector3::new(-d, -d, d),
        ];
        
        Self {
            carbon: Vector3::zeros(),
            hydrogens,
            // STO-3G-like exponents (optimized for molecular calculations)
            c_1s_exp: 5.67,  // Carbon 1s
            c_2s_exp: 1.72,  // Carbon 2s
            c_2p_exp: 1.72,  // Carbon 2p
            h_1s_exp: 1.24,  // Hydrogen 1s
        }
    }
    
    /// Evaluate Carbon 1s orbital
    pub fn c_1s(&self, r: &Vector3<f64>) -> f64 {
        let dr = (r - self.carbon).norm();
        (-self.c_1s_exp * dr).exp()
    }
    
    /// Evaluate Carbon 2s orbital
    pub fn c_2s(&self, r: &Vector3<f64>) -> f64 {
        let dr = (r - self.carbon).norm();
        (1.0 - self.c_2s_exp * dr / 2.0) * (-self.c_2s_exp * dr / 2.0).exp()
    }
    
    /// Evaluate Carbon 2px orbital
    pub fn c_2px(&self, r: &Vector3<f64>) -> f64 {
        let dr_vec = r - self.carbon;
        let dr = dr_vec.norm();
        if dr < 1e-10 { return 0.0; }
        dr_vec.x * (-self.c_2p_exp * dr / 2.0).exp()
    }
    
    /// Evaluate Carbon 2py orbital
    pub fn c_2py(&self, r: &Vector3<f64>) -> f64 {
        let dr_vec = r - self.carbon;
        let dr = dr_vec.norm();
        if dr < 1e-10 { return 0.0; }
        dr_vec.y * (-self.c_2p_exp * dr / 2.0).exp()
    }
    
    /// Evaluate Carbon 2pz orbital
    pub fn c_2pz(&self, r: &Vector3<f64>) -> f64 {
        let dr_vec = r - self.carbon;
        let dr = dr_vec.norm();
        if dr < 1e-10 { return 0.0; }
        dr_vec.z * (-self.c_2p_exp * dr / 2.0).exp()
    }
    
    /// Evaluate Hydrogen 1s orbital for given H index (0-3)
    pub fn h_1s(&self, h_idx: usize, r: &Vector3<f64>) -> f64 {
        let dr = (r - self.hydrogens[h_idx]).norm();
        (-self.h_1s_exp * dr).exp()
    }
}

impl Default for CH4Basis {
    fn default() -> Self {
        Self::new()
    }
}

/// Methane molecule wavefunction: Slater determinant × Jastrow factor.
/// 
/// Uses 5 doubly-occupied molecular orbitals for 10 electrons:
/// - MO1: Core (mostly C 1s)
/// - MO2: Bonding (C 2s + H 1s)
/// - MO3-5: Bonding (C 2p + H 1s combinations)
#[derive(Clone)]
pub struct Methane {
    /// Basis set
    pub basis: CH4Basis,
    /// Jastrow correlation factor
    pub jastrow: Jastrow2,
    /// Number of electrons (always 10)
    pub num_electrons: usize,
    /// Spin assignments: +1 for up, -1 for down (5 each)
    pub spins: Vec<i32>,
}

impl Methane {
    /// Create a new Methane wavefunction.
    pub fn new(cusp_param: f64) -> Self {
        let basis = CH4Basis::new();
        let jastrow = Jastrow2 {
            cusp_param,
            num_electrons: 10,
        };
        
        // 5 spin-up, 5 spin-down electrons
        let spins = vec![1, 1, 1, 1, 1, -1, -1, -1, -1, -1];
        
        Self {
            basis,
            jastrow,
            num_electrons: 10,
            spins,
        }
    }
    
    /// Evaluate molecular orbital at position r.
    /// 
    /// MO indices:
    /// 0: Core (C 1s)
    /// 1: σ bonding (C 2s + H 1s sum)
    /// 2: σ bonding (C 2px + H combinations)
    /// 3: σ bonding (C 2py + H combinations)
    /// 4: σ bonding (C 2pz + H combinations)
    fn mo_evaluate(&self, mo_idx: usize, r: &Vector3<f64>) -> f64 {
        match mo_idx {
            0 => {
                // Core: mostly C 1s
                self.basis.c_1s(r)
            }
            1 => {
                // σ bonding: C 2s + all H 1s
                let c2s = self.basis.c_2s(r);
                let h_sum: f64 = (0..4).map(|i| self.basis.h_1s(i, r)).sum();
                0.5 * (c2s + 0.5 * h_sum)
            }
            2 => {
                // C 2px + H1 + H2 - H3 - H4 (along x-like direction)
                let c2px = self.basis.c_2px(r);
                let h_comb = self.basis.h_1s(0, r) + self.basis.h_1s(1, r)
                          - self.basis.h_1s(2, r) - self.basis.h_1s(3, r);
                0.5 * (c2px + 0.3 * h_comb)
            }
            3 => {
                // C 2py + H1 - H2 + H3 - H4 (along y-like direction)
                let c2py = self.basis.c_2py(r);
                let h_comb = self.basis.h_1s(0, r) - self.basis.h_1s(1, r)
                          + self.basis.h_1s(2, r) - self.basis.h_1s(3, r);
                0.5 * (c2py + 0.3 * h_comb)
            }
            4 => {
                // C 2pz + H1 - H2 - H3 + H4 (along z-like direction)
                let c2pz = self.basis.c_2pz(r);
                let h_comb = self.basis.h_1s(0, r) - self.basis.h_1s(1, r)
                          - self.basis.h_1s(2, r) + self.basis.h_1s(3, r);
                0.5 * (c2pz + 0.3 * h_comb)
            }
            _ => 0.0,
        }
    }
    
    /// Numerical gradient of molecular orbital.
    fn mo_derivative(&self, mo_idx: usize, r: &Vector3<f64>) -> Vector3<f64> {
        let h = 1e-5;
        let mut grad = Vector3::zeros();
        for axis in 0..3 {
            let mut r_fwd = *r;
            let mut r_bwd = *r;
            r_fwd[axis] += h;
            r_bwd[axis] -= h;
            grad[axis] = (self.mo_evaluate(mo_idx, &r_fwd) - self.mo_evaluate(mo_idx, &r_bwd)) / (2.0 * h);
        }
        grad
    }
    
    /// Numerical Laplacian of molecular orbital.
    fn mo_laplacian(&self, mo_idx: usize, r: &Vector3<f64>) -> f64 {
        let h = 1e-5;
        let psi = self.mo_evaluate(mo_idx, r);
        let mut laplacian = 0.0;
        for axis in 0..3 {
            let mut r_fwd = *r;
            let mut r_bwd = *r;
            r_fwd[axis] += h;
            r_bwd[axis] -= h;
            laplacian += (self.mo_evaluate(mo_idx, &r_fwd) - 2.0 * psi + self.mo_evaluate(mo_idx, &r_bwd)) / (h * h);
        }
        laplacian
    }
    
    /// Build Slater matrix for one spin sector.
    /// Returns (determinant, inverse matrix)
    fn slater_sector(&self, r: &[Vector3<f64>], spin: i32) -> (f64, nalgebra::DMatrix<f64>) {
        let indices: Vec<usize> = self.spins.iter()
            .enumerate()
            .filter(|(_, &s)| s == spin)
            .map(|(i, _)| i)
            .collect();
        
        let n = indices.len();
        let mut s = nalgebra::DMatrix::zeros(n, n);
        
        // Fill Slater matrix: S[i][j] = MO_j(r_i)
        for (row, &elec_idx) in indices.iter().enumerate() {
            for col in 0..n {
                s[(row, col)] = self.mo_evaluate(col, &r[elec_idx]);
            }
        }
        
        let det = s.determinant();
        let inv = s.try_inverse().unwrap_or_else(|| nalgebra::DMatrix::identity(n, n));
        (det, inv)
    }
    
    /// Evaluate Slater determinant part.
    fn slater_evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        let (det_up, _) = self.slater_sector(r, 1);
        let (det_down, _) = self.slater_sector(r, -1);
        det_up * det_down
    }
    
    /// Gradient of Slater determinant.
    /// Returns ∇S (not psi × ∇log(S))
    fn slater_derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let (det_up, inv_up) = self.slater_sector(r, 1);
        let (det_down, inv_down) = self.slater_sector(r, -1);
        
        let up_indices: Vec<usize> = self.spins.iter()
            .enumerate()
            .filter(|(_, &s)| s == 1)
            .map(|(i, _)| i)
            .collect();
        let down_indices: Vec<usize> = self.spins.iter()
            .enumerate()
            .filter(|(_, &s)| s == -1)
            .map(|(i, _)| i)
            .collect();
        
        let mut gradients = vec![Vector3::zeros(); 10];
        
        // Spin-up electrons: ∂S/∂r_i = S × Σ_j (S^-1)_ji × ∇φ_j(r_i)
        // For product D_up × D_down, derivative w.r.t. up electron gives D_down × ∇D_up
        for (row, &elec_idx) in up_indices.iter().enumerate() {
            let grad_log: Vector3<f64> = (0..5)
                .map(|col| inv_up[(col, row)] * self.mo_derivative(col, &r[elec_idx]))
                .sum();
            gradients[elec_idx] = det_up * det_down * grad_log;
        }
        
        // Spin-down electrons
        for (row, &elec_idx) in down_indices.iter().enumerate() {
            let grad_log: Vector3<f64> = (0..5)
                .map(|col| inv_down[(col, row)] * self.mo_derivative(col, &r[elec_idx]))
                .sum();
            gradients[elec_idx] = det_up * det_down * grad_log;
        }
        
        gradients
    }
    
    /// Laplacian of Slater determinant.
    /// Returns ∇²S (not psi × ∇²log(S))
    fn slater_laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let (det_up, inv_up) = self.slater_sector(r, 1);
        let (det_down, inv_down) = self.slater_sector(r, -1);
        
        let up_indices: Vec<usize> = self.spins.iter()
            .enumerate()
            .filter(|(_, &s)| s == 1)
            .map(|(i, _)| i)
            .collect();
        let down_indices: Vec<usize> = self.spins.iter()
            .enumerate()
            .filter(|(_, &s)| s == -1)
            .map(|(i, _)| i)
            .collect();
        
        let mut laplacians = vec![0.0; 10];
        
        // Spin-up electrons
        for (row, &elec_idx) in up_indices.iter().enumerate() {
            let lap_log: f64 = (0..5)
                .map(|col| inv_up[(col, row)] * self.mo_laplacian(col, &r[elec_idx]))
                .sum();
            laplacians[elec_idx] = det_up * det_down * lap_log;
        }
        
        // Spin-down electrons
        for (row, &elec_idx) in down_indices.iter().enumerate() {
            let lap_log: f64 = (0..5)
                .map(|col| inv_down[(col, row)] * self.mo_laplacian(col, &r[elec_idx]))
                .sum();
            laplacians[elec_idx] = det_up * det_down * lap_log;
        }
        
        laplacians
    }
    
    /// Nuclear-nuclear repulsion energy (constant).
    pub fn nuclear_repulsion(&self) -> f64 {
        let z_c = 6.0;
        let z_h = 1.0;
        let mut v_nn = 0.0;
        
        // C-H repulsion
        for i in 0..4 {
            let r_ch = (self.basis.carbon - self.basis.hydrogens[i]).norm();
            v_nn += z_c * z_h / r_ch;
        }
        
        // H-H repulsion
        for i in 0..4 {
            for j in (i+1)..4 {
                let r_hh = (self.basis.hydrogens[i] - self.basis.hydrogens[j]).norm();
                v_nn += z_h * z_h / r_hh;
            }
        }
        
        v_nn
    }
}

impl MultiWfn for Methane {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        use rand_distr::Distribution;
        
        // Initialize electrons near the nuclei for better sampling
        (0..self.num_electrons)
            .map(|i| {
                // Place some electrons near C, some near H atoms
                let center = if i < 6 {
                    self.basis.carbon
                } else {
                    self.basis.hydrogens[i - 6]
                };
                center + Vector3::new(
                    dist.sample(&mut rng) * 0.5,
                    dist.sample(&mut rng) * 0.5,
                    dist.sample(&mut rng) * 0.5,
                )
            })
            .collect()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        self.slater_evaluate(r) * self.jastrow.evaluate(r)
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        // Ψ = S × J
        // ∇Ψ = (∇S) × J + S × (∇J)
        let s = self.slater_evaluate(r);
        let j = self.jastrow.evaluate(r);
        let grad_s = self.slater_derivative(r);
        let grad_j = self.jastrow.derivative(r);
        
        grad_s.into_iter()
            .zip(grad_j.into_iter())
            .map(|(gs, gj)| gs * j + s * gj)
            .collect()
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        // Ψ = S × J
        // ∇²Ψ = (∇²S) × J + 2(∇S)·(∇J) + S × (∇²J)
        let s = self.slater_evaluate(r);
        let j = self.jastrow.evaluate(r);
        let lap_s = self.slater_laplacian(r);
        let lap_j = self.jastrow.laplacian(r);
        let grad_s = self.slater_derivative(r);
        let grad_j = self.jastrow.derivative(r);
        
        lap_s.into_iter()
            .zip(lap_j.into_iter())
            .zip(grad_s.into_iter().zip(grad_j.into_iter()))
            .map(|((ls, lj), (gs, gj))| {
                ls * j + 2.0 * gs.dot(&gj) + s * lj
            })
            .collect()
    }
}

impl EnergyCalculator for Methane {
    fn local_energy(&self, r: &[Vector3<f64>]) -> f64 {
        let psi = self.evaluate(r);
        let laplacian = self.laplacian(r);
        
        // Kinetic energy: -½ Σᵢ ∇²ψ/ψ
        let kinetic = -0.5 * laplacian.iter().sum::<f64>() / psi;
        
        // Electron-nucleus attraction
        // Carbon (Z=6) at origin
        let z_c = 6.0;
        let v_ec: f64 = r.iter()
            .map(|ri| -z_c / (ri - self.basis.carbon).norm())
            .sum();
        
        // Hydrogen atoms (Z=1)
        let v_eh: f64 = r.iter()
            .flat_map(|ri| {
                self.basis.hydrogens.iter().map(move |h_pos| {
                    -1.0 / (ri - h_pos).norm()
                })
            })
            .sum();
        
        // Electron-electron repulsion
        let n = r.len();
        let v_ee: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| 1.0 / (r[i] - r[j]).norm())
            .sum();
        
        // Nuclear-nuclear repulsion (constant)
        let v_nn = self.nuclear_repulsion();
        
        kinetic + v_ec + v_eh + v_ee + v_nn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_methane_geometry() {
        let basis = CH4Basis::new();
        // Check tetrahedral symmetry: all C-H distances equal
        for i in 0..4 {
            let d = (basis.carbon - basis.hydrogens[i]).norm();
            assert!((d - CH_BOND_LENGTH).abs() < 0.01, "C-H bond length should be ~2.05 Bohr");
        }
        // Check H-H distances (should all be equal)
        let hh_dist = (basis.hydrogens[0] - basis.hydrogens[1]).norm();
        for i in 0..4 {
            for j in (i+1)..4 {
                let d = (basis.hydrogens[i] - basis.hydrogens[j]).norm();
                assert!((d - hh_dist).abs() < 0.01, "All H-H distances should be equal");
            }
        }
    }
    
    #[test]
    fn test_methane_nuclear_repulsion() {
        let ch4 = Methane::new(0.5);
        let v_nn = ch4.nuclear_repulsion();
        // V_nn for CH4 should be positive and around 13-14 Ha
        assert!(v_nn > 10.0 && v_nn < 20.0, "Nuclear repulsion should be ~13 Ha, got {}", v_nn);
    }
    
    #[test]
    fn test_methane_numerical_derivative_and_laplacian() {
        let ch4 = Methane::new(0.5);
        let positions = ch4.initialize();
        
        let analytical_deriv = ch4.derivative(&positions);
        let numerical_deriv = ch4.numerical_derivative(&positions, 1e-5);
        
        for i in 0..10 {
            for axis in 0..3 {
                let diff = (analytical_deriv[i][axis] - numerical_deriv[i][axis]).abs();
                let scale = analytical_deriv[i][axis].abs().max(1e-6);
                assert!(diff / scale < 0.1, 
                    "Derivative mismatch at electron {}, axis {}: analytical={}, numerical={}",
                    i, axis, analytical_deriv[i][axis], numerical_deriv[i][axis]);
            }
        }
        
        let analytical_lap = ch4.laplacian(&positions);
        let numerical_lap = ch4.numerical_laplacian(&positions, 1e-4);
        
        for i in 0..10 {
            let diff = (analytical_lap[i] - numerical_lap[i]).abs();
            let scale = analytical_lap[i].abs().max(1e-6);
            assert!(diff / scale < 0.2,
                "Laplacian mismatch at electron {}: analytical={}, numerical={}",
                i, analytical_lap[i], numerical_lap[i]);
        }
    }
}
