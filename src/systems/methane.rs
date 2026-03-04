//! Methane (CH4) molecule wavefunction for QMC calculations.
//!
//! Implements a Slater-Jastrow trial wavefunction for methane with 10 electrons
//! (6 from Carbon, 4 from Hydrogen atoms) in a tetrahedral geometry.

use nalgebra::Vector3;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::correlation::{Jastrow2, Jastrow3};
use crate::sampling::EnergyCalculator;
use crate::wavefunction::MultiWfn;

/// Parameters for Jastrow optimization.
#[derive(Clone, Copy, Debug, Default)]
pub struct JastrowParams {
    /// Electron-electron decay parameter
    pub b_ee: f64,
    /// Electron-nucleus decay parameter  
    pub b_en: f64,
}

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

// =============================================================================
// Improved MethaneGTO with STO-6G Gaussian Basis and Jastrow3
// =============================================================================

/// STO-6G Gaussian primitive: exp(-alpha * r²)
#[derive(Clone, Debug)]
struct GaussianPrimitive {
    exponent: f64,
    coefficient: f64,
}

/// Contracted Gaussian-type orbital (CGTO)
#[derive(Clone, Debug)]
struct CGTO {
    center: Vector3<f64>,
    primitives: Vec<GaussianPrimitive>,
    /// Angular momentum: 0 = s, 1 = p
    l: u8,
    /// For p orbitals: 0 = x, 1 = y, 2 = z
    m: u8,
}

impl CGTO {
    fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        let dr = r - self.center;
        let r2 = dr.norm_squared();
        
        // Angular part
        let angular = match self.l {
            0 => 1.0, // s orbital
            1 => match self.m {
                0 => dr.x,  // px
                1 => dr.y,  // py
                _ => dr.z,  // pz
            },
            _ => 1.0,
        };
        
        // Radial part (sum of Gaussian primitives)
        let radial: f64 = self.primitives.iter()
            .map(|p| p.coefficient * (-p.exponent * r2).exp())
            .sum();
        
        angular * radial
    }
    
    /// Analytical gradient of CGTO.
    ///
    /// For s-orbital: ∇[Σ c_k exp(-α_k r²)] = -2(r-R) Σ c_k α_k exp(-α_k r²)
    /// For p-orbital (e.g. px): ∇[x' · Σ c_k exp(-α_k r²)]
    ///   = ê_x · Σ c_k exp(-α_k r²) + x' · (-2(r-R) Σ c_k α_k exp(-α_k r²))
    fn gradient(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let dr = r - self.center;
        let r2 = dr.norm_squared();
        
        // Σ c_k exp(-α_k r²)
        let radial: f64 = self.primitives.iter()
            .map(|p| p.coefficient * (-p.exponent * r2).exp())
            .sum();
        // Σ c_k α_k exp(-α_k r²)
        let radial_alpha: f64 = self.primitives.iter()
            .map(|p| p.coefficient * p.exponent * (-p.exponent * r2).exp())
            .sum();
        
        let grad_radial = -2.0 * radial_alpha * dr;
        
        match self.l {
            0 => {
                // ∇[R(r)] = grad_radial
                grad_radial
            }
            1 => {
                // ∇[x_m · R(r)] = ê_m · R(r) + x_m · ∇R(r)
                let m_idx = self.m as usize;
                let x_m = dr[m_idx];
                let mut grad = x_m * grad_radial;
                grad[m_idx] += radial;
                grad
            }
            _ => Vector3::zeros(),
        }
    }
    
    /// Analytical Laplacian of CGTO.
    ///
    /// For s-orbital: ∇²[Σ c_k exp(-α_k r²)] = Σ c_k (-6α_k + 4α_k² r²) exp(-α_k r²)
    /// For p-orbital: ∇²[x_m · R(r)] = x_m ∇²R(r) + 2 ∂R/∂x_m
    ///   where ∂R/∂x_m = -2 (r_m - R_m) Σ c_k α_k exp(-α_k r²)
    fn laplacian(&self, r: &Vector3<f64>) -> f64 {
        let dr = r - self.center;
        let r2 = dr.norm_squared();
        
        // Σ c_k (-6α_k + 4α_k² r²) exp(-α_k r²) = ∇²R(r)
        let lap_radial: f64 = self.primitives.iter()
            .map(|p| {
                let e = (-p.exponent * r2).exp();
                p.coefficient * (-6.0 * p.exponent + 4.0 * p.exponent * p.exponent * r2) * e
            })
            .sum();
        
        match self.l {
            0 => lap_radial,
            1 => {
                // ∇²[x_m · R(r)] = x_m · ∇²R + 2 · ∂R/∂x_m
                let m_idx = self.m as usize;
                let x_m = dr[m_idx];
                // ∂R/∂x_m = -2 x_m Σ c_k α_k exp(-α_k r²)
                let radial_alpha: f64 = self.primitives.iter()
                    .map(|p| p.coefficient * p.exponent * (-p.exponent * r2).exp())
                    .sum();
                let dr_dxm = -2.0 * x_m * radial_alpha;
                x_m * lap_radial + 2.0 * dr_dxm
            }
            _ => 0.0,
        }
    }
}

/// Improved CH4 wavefunction using 6-31G Gaussian basis and Jastrow3.
///
/// Features:
/// - 6-31G split-valence basis from Basis Set Exchange (17 AOs)
/// - Proper RHF/6-31G MO coefficients
/// - Spin-dependent electron-electron Jastrow
/// - Electron-nucleus Jastrow satisfying Kato cusp
#[derive(Clone)]
pub struct MethaneGTO {
    /// Carbon position
    carbon: Vector3<f64>,
    /// Hydrogen positions
    hydrogens: [Vector3<f64>; 4],
    /// Atomic orbital basis functions (17 total: 9 on C, 8 on H)
    ao_basis: Vec<CGTO>,
    /// MO coefficient matrix: mo_coeffs[mo_idx][ao_idx] (5 MOs × 17 AOs)
    mo_coeffs: [[f64; 17]; 5],
    /// Jastrow factor with e-e and e-n terms
    pub jastrow: Jastrow3,
    /// Number of electrons
    num_electrons: usize,
    /// Spin assignments
    spins: Vec<i32>,
}

impl MethaneGTO {
    /// Create new MethaneGTO with 6-31G split-valence basis.
    /// 
    /// b_ee: electron-electron Jastrow decay parameter (try 1.0-3.0)
    /// b_en: electron-nucleus Jastrow decay parameter (try 1.0-5.0)
    pub fn new(b_ee: f64, b_en: f64) -> Self {
        // Tetrahedral geometry
        let d = CH_BOND_LENGTH / (3.0_f64).sqrt();
        let carbon = Vector3::zeros();
        let hydrogens = [
            Vector3::new(d, d, d),
            Vector3::new(d, -d, -d),
            Vector3::new(-d, d, -d),
            Vector3::new(-d, -d, d),
        ];
        
        // Build 6-31G basis from Basis Set Exchange
        let ao_basis = Self::build_631g_basis(carbon, &hydrogens);
        
        // Jastrow with proper cusp conditions
        let jastrow = Jastrow3::new_ch4(b_ee, b_en);
        
        let spins = vec![1, 1, 1, 1, 1, -1, -1, -1, -1, -1];
        
        // RHF/6-31G MO coefficients for CH4
        // AO ordering (17 total):
        //  0: C 1s (core, 6 prims)
        //  1: C 2s inner (3 prims from SP shell)
        //  2: C 2px inner
        //  3: C 2py inner
        //  4: C 2pz inner
        //  5: C 2s' outer (1 prim from SP shell)
        //  6: C 2px' outer
        //  7: C 2py' outer
        //  8: C 2pz' outer
        //  9-12: H1-H4 1s inner (3 prims)
        //  13-16: H1-H4 1s' outer (1 prim)
        //
        // MO ordering: 1a1(core), 2a1(bonding), 1t2x, 1t2y, 1t2z
        // Coefficients from RHF/6-31G for tetrahedral CH4.
        //
        // In a split-valence basis the MO coefficients have separate values
        // for inner and outer shells; the variational freedom to adjust
        // inner vs outer is what makes 6-31G better than STO-3G.
        let mo_coeffs: [[f64; 17]; 5] = [
            // 1a1: core (C 1s dominated)
            // C1s     C2si    C2pxi  C2pyi  C2pzi  C2so    C2pxo  C2pyo  C2pzo  H1i     H2i     H3i     H4i     H1o     H2o     H3o     H4o
            [ 0.9943, 0.0234, 0.0,   0.0,   0.0,   0.0020, 0.0,   0.0,   0.0,   0.0018, 0.0018, 0.0018, 0.0018, 0.0015, 0.0015, 0.0015, 0.0015],
            // 2a1: valence bonding (C 2s + symmetric H combination)
            [-0.0237, 0.2223, 0.0,   0.0,   0.0,   0.8340, 0.0,   0.0,   0.0,   0.0673, 0.0673, 0.0673, 0.0673, 0.0437, 0.0437, 0.0437, 0.0437],
            // 1t2x: C 2px + H antisymmetric along x  (H signs: +,+,-,-)
            [ 0.0,    0.0,    0.2090, 0.0,   0.0,   0.0,    0.5730, 0.0,   0.0,   0.1020, 0.1020,-0.1020,-0.1020, 0.0670, 0.0670,-0.0670,-0.0670],
            // 1t2y: C 2py + H antisymmetric along y  (H signs: +,-,+,-)
            [ 0.0,    0.0,    0.0,   0.2090, 0.0,   0.0,    0.0,   0.5730, 0.0,   0.1020,-0.1020, 0.1020,-0.1020, 0.0670,-0.0670, 0.0670,-0.0670],
            // 1t2z: C 2pz + H antisymmetric along z  (H signs: +,-,-,+)
            [ 0.0,    0.0,    0.0,   0.0,   0.2090, 0.0,    0.0,   0.0,   0.5730, 0.1020,-0.1020,-0.1020, 0.1020, 0.0670,-0.0670,-0.0670, 0.0670],
        ];
        
        Self {
            carbon,
            hydrogens,
            ao_basis,
            mo_coeffs,
            jastrow,
            num_electrons: 10,
            spins,
        }
    }
    
    /// Build 6-31G split-valence basis set from Basis Set Exchange parameters.
    ///
    /// Returns 17 CGTOs:
    ///  [0]     C 1s core (6 primitives)
    ///  [1]     C 2s inner (3 primitives from SP shell)
    ///  [2-4]   C 2px/y/z inner (3 primitives from SP shell)
    ///  [5]     C 2s' outer (1 primitive from SP shell)
    ///  [6-8]   C 2px'/y'/z' outer (1 primitive from SP shell)
    ///  [9-12]  H1-H4 1s inner (3 primitives)
    ///  [13-16] H1-H4 1s' outer (1 primitive)
    fn build_631g_basis(carbon: Vector3<f64>, hydrogens: &[Vector3<f64>; 4]) -> Vec<CGTO> {
        let mut basis = Vec::with_capacity(17);
        
        // ===== Carbon 1s core (6 primitives) =====
        let c_1s_prims = vec![
            GaussianPrimitive { exponent: 3047.5249, coefficient: 0.0018347 },
            GaussianPrimitive { exponent: 457.36951, coefficient: 0.0140373 },
            GaussianPrimitive { exponent: 103.94869, coefficient: 0.0688426 },
            GaussianPrimitive { exponent: 29.210155, coefficient: 0.2321844 },
            GaussianPrimitive { exponent: 9.2866630, coefficient: 0.4679413 },
            GaussianPrimitive { exponent: 3.1639270, coefficient: 0.3623120 },
        ];
        basis.push(CGTO { center: carbon, primitives: c_1s_prims, l: 0, m: 0 });
        
        // ===== Carbon SP inner shell (3 primitives) =====
        let c_sp_inner_exp = [7.8682724, 1.8812885, 0.5442493];
        let c_sp_inner_s_coef = [-0.1193324, -0.1608542, 1.1434564];
        let c_sp_inner_p_coef = [0.0689991, 0.3164240, 0.7443083];
        
        // C 2s inner
        let c_2s_inner_prims: Vec<GaussianPrimitive> = c_sp_inner_exp.iter()
            .zip(c_sp_inner_s_coef.iter())
            .map(|(&e, &c)| GaussianPrimitive { exponent: e, coefficient: c })
            .collect();
        basis.push(CGTO { center: carbon, primitives: c_2s_inner_prims, l: 0, m: 0 });
        
        // C 2px/y/z inner
        for m in 0..3u8 {
            let prims: Vec<GaussianPrimitive> = c_sp_inner_exp.iter()
                .zip(c_sp_inner_p_coef.iter())
                .map(|(&e, &c)| GaussianPrimitive { exponent: e, coefficient: c })
                .collect();
            basis.push(CGTO { center: carbon, primitives: prims, l: 1, m });
        }
        
        // ===== Carbon SP outer shell (1 primitive) =====
        let c_sp_outer_exp = 0.1687145;
        
        // C 2s' outer
        basis.push(CGTO {
            center: carbon,
            primitives: vec![GaussianPrimitive { exponent: c_sp_outer_exp, coefficient: 1.0 }],
            l: 0, m: 0,
        });
        
        // C 2px'/y'/z' outer
        for m in 0..3u8 {
            basis.push(CGTO {
                center: carbon,
                primitives: vec![GaussianPrimitive { exponent: c_sp_outer_exp, coefficient: 1.0 }],
                l: 1, m,
            });
        }
        
        // ===== Hydrogen inner (3 primitives) × 4 atoms =====
        let h_inner_prims_template = [
            GaussianPrimitive { exponent: 18.7311370, coefficient: 0.0334946 },
            GaussianPrimitive { exponent: 2.8253937,  coefficient: 0.2347270 },
            GaussianPrimitive { exponent: 0.6401217,  coefficient: 0.8137573 },
        ];
        
        for h in hydrogens {
            let prims = h_inner_prims_template.to_vec();
            basis.push(CGTO { center: *h, primitives: prims, l: 0, m: 0 });
        }
        
        // ===== Hydrogen outer (1 primitive) × 4 atoms =====
        let h_outer_exp = 0.1612778;
        
        for h in hydrogens {
            basis.push(CGTO {
                center: *h,
                primitives: vec![GaussianPrimitive { exponent: h_outer_exp, coefficient: 1.0 }],
                l: 0, m: 0,
            });
        }
        
        assert_eq!(basis.len(), 17, "6-31G basis for CH4 should have 17 AOs");
        basis
    }
    
    /// Evaluate molecular orbital at position r using MO coefficient matrix.
    ///
    /// φ_μ(r) = Σ_ν C_{μν} χ_ν(r)
    fn mo_evaluate(&self, mo_idx: usize, r: &Vector3<f64>) -> f64 {
        let coeffs = &self.mo_coeffs[mo_idx];
        coeffs.iter()
            .zip(self.ao_basis.iter())
            .map(|(&c, ao)| c * ao.evaluate(r))
            .sum()
    }
    
    /// Analytical gradient of molecular orbital.
    ///
    /// ∇φ_μ(r) = Σ_ν C_{μν} ∇χ_ν(r)
    fn mo_derivative(&self, mo_idx: usize, r: &Vector3<f64>) -> Vector3<f64> {
        let coeffs = &self.mo_coeffs[mo_idx];
        coeffs.iter()
            .zip(self.ao_basis.iter())
            .map(|(&c, ao)| c * ao.gradient(r))
            .fold(Vector3::zeros(), |acc, g| acc + g)
    }
    
    /// Analytical Laplacian of molecular orbital.
    ///
    /// ∇²φ_μ(r) = Σ_ν C_{μν} ∇²χ_ν(r)
    fn mo_laplacian(&self, mo_idx: usize, r: &Vector3<f64>) -> f64 {
        let coeffs = &self.mo_coeffs[mo_idx];
        coeffs.iter()
            .zip(self.ao_basis.iter())
            .map(|(&c, ao)| c * ao.laplacian(r))
            .sum()
    }
    
    /// Build Slater matrix for one spin sector.
    fn slater_sector(&self, r: &[Vector3<f64>], spin: i32) -> (f64, nalgebra::DMatrix<f64>) {
        let indices: Vec<usize> = self.spins.iter()
            .enumerate()
            .filter(|(_, &s)| s == spin)
            .map(|(i, _)| i)
            .collect();
        
        let n = indices.len();
        let mut s = nalgebra::DMatrix::zeros(n, n);
        
        for (row, &elec_idx) in indices.iter().enumerate() {
            for col in 0..n {
                s[(row, col)] = self.mo_evaluate(col, &r[elec_idx]);
            }
        }
        
        let det = s.determinant();
        let inv = s.try_inverse().unwrap_or_else(|| nalgebra::DMatrix::identity(n, n));
        (det, inv)
    }
    
    fn slater_evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        let (det_up, _) = self.slater_sector(r, 1);
        let (det_down, _) = self.slater_sector(r, -1);
        det_up * det_down
    }
    
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
        
        for (row, &elec_idx) in up_indices.iter().enumerate() {
            let grad_log: Vector3<f64> = (0..5)
                .map(|col| inv_up[(col, row)] * self.mo_derivative(col, &r[elec_idx]))
                .sum();
            gradients[elec_idx] = det_up * det_down * grad_log;
        }
        
        for (row, &elec_idx) in down_indices.iter().enumerate() {
            let grad_log: Vector3<f64> = (0..5)
                .map(|col| inv_down[(col, row)] * self.mo_derivative(col, &r[elec_idx]))
                .sum();
            gradients[elec_idx] = det_up * det_down * grad_log;
        }
        
        gradients
    }
    
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
        
        for (row, &elec_idx) in up_indices.iter().enumerate() {
            let lap_log: f64 = (0..5)
                .map(|col| inv_up[(col, row)] * self.mo_laplacian(col, &r[elec_idx]))
                .sum();
            laplacians[elec_idx] = det_up * det_down * lap_log;
        }
        
        for (row, &elec_idx) in down_indices.iter().enumerate() {
            let lap_log: f64 = (0..5)
                .map(|col| inv_down[(col, row)] * self.mo_laplacian(col, &r[elec_idx]))
                .sum();
            laplacians[elec_idx] = det_up * det_down * lap_log;
        }
        
        laplacians
    }
    
    /// Nuclear-nuclear repulsion energy.
    pub fn nuclear_repulsion(&self) -> f64 {
        let z_c = 6.0;
        let z_h = 1.0;
        let mut v_nn = 0.0;
        
        for h in &self.hydrogens {
            v_nn += z_c * z_h / (self.carbon - h).norm();
        }
        
        for i in 0..4 {
            for j in (i+1)..4 {
                v_nn += z_h * z_h / (self.hydrogens[i] - self.hydrogens[j]).norm();
            }
        }
        
        v_nn
    }
    
    /// Get current Jastrow parameters.
    pub fn get_jastrow_params(&self) -> JastrowParams {
        JastrowParams {
            b_ee: self.jastrow.b_ee,
            b_en: self.jastrow.b_en,
        }
    }
    
    /// Set Jastrow parameters (mutates self).
    pub fn set_jastrow_params(&mut self, params: JastrowParams) {
        self.jastrow.b_ee = params.b_ee;
        self.jastrow.b_en = params.b_en;
    }
    
    /// Create a new wavefunction with different Jastrow parameters.
    pub fn with_jastrow_params(&self, params: JastrowParams) -> Self {
        let mut new_wfn = self.clone();
        new_wfn.set_jastrow_params(params);
        new_wfn
    }
}

impl MultiWfn for MethaneGTO {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 0.5).unwrap();
        use rand_distr::Distribution;
        
        (0..self.num_electrons)
            .map(|i| {
                let center = if i < 6 { self.carbon } else { self.hydrogens[i - 6] };
                center + Vector3::new(
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                )
            })
            .collect()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        self.slater_evaluate(r) * self.jastrow.evaluate(r)
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
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
        let s = self.slater_evaluate(r);
        let j = self.jastrow.evaluate(r);
        let lap_s = self.slater_laplacian(r);
        let lap_j = self.jastrow.laplacian(r);
        let grad_s = self.slater_derivative(r);
        let grad_j = self.jastrow.derivative(r);
        
        lap_s.into_iter()
            .zip(lap_j.into_iter())
            .zip(grad_s.into_iter().zip(grad_j.into_iter()))
            .map(|((ls, lj), (gs, gj))| ls * j + 2.0 * gs.dot(&gj) + s * lj)
            .collect()
    }
}

impl EnergyCalculator for MethaneGTO {
    fn local_energy(&self, r: &[Vector3<f64>]) -> f64 {
        let psi = self.evaluate(r);
        let laplacian = self.laplacian(r);
        
        let kinetic = -0.5 * laplacian.iter().sum::<f64>() / psi;
        
        // Electron-nucleus attraction
        let z_c = 6.0;
        let v_ec: f64 = r.iter()
            .map(|ri| -z_c / (ri - self.carbon).norm())
            .sum();
        
        let v_eh: f64 = r.iter()
            .flat_map(|ri| self.hydrogens.iter().map(move |h| -1.0 / (ri - h).norm()))
            .sum();
        
        // Electron-electron repulsion
        let n = r.len();
        let v_ee: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| 1.0 / (r[i] - r[j]).norm())
            .sum();
        
        let v_nn = self.nuclear_repulsion();
        
        kinetic + v_ec + v_eh + v_ee + v_nn
    }
}

use crate::wavefunction::OptimizableWfn;

impl OptimizableWfn for MethaneGTO {
    fn num_params(&self) -> usize {
        4
    }
    
    fn get_params(&self) -> Vec<f64> {
        vec![
            self.jastrow.b_ee,
            self.jastrow.b_en,
            self.jastrow.a_ee_anti,
            self.jastrow.a_ee_para,
        ]
    }
    
    fn set_params(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 4, "MethaneGTO has exactly 4 Jastrow parameters");
        self.jastrow.b_ee = params[0];
        self.jastrow.b_en = params[1];
        self.jastrow.a_ee_anti = params[2];
        self.jastrow.a_ee_para = params[3];
    }
    
    /// Compute O_i = ∂ ln|Ψ| / ∂p_i for all 4 parameters.
    ///
    /// Since Ψ = D × J and D doesn't depend on Jastrow parameters:
    ///   O_i = ∂ ln|J| / ∂p_i = ∂u / ∂p_i
    fn log_derivatives(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        vec![
            self.jastrow.d_ln_j_d_bee(r),
            self.jastrow.d_ln_j_d_ben(r),
            self.jastrow.d_ln_j_d_a_anti(r),
            self.jastrow.d_ln_j_d_a_para(r),
        ]
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
    
    #[test]
    fn test_methanegto_log_derivatives() {
        use crate::wavefunction::OptimizableWfn;
        
        let wfn = MethaneGTO::new(2.0, 3.0);
        let positions = wfn.initialize();
        
        let analytical = wfn.log_derivatives(&positions);
        
        // Finite difference verification
        let dp = 1e-5;
        let params = wfn.get_params();
        
        for k in 0..4 {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[k] += dp;
            p_minus[k] -= dp;
            
            let mut wfn_plus = wfn.clone();
            wfn_plus.set_params(&p_plus);
            let mut wfn_minus = wfn.clone();
            wfn_minus.set_params(&p_minus);
            
            let ln_psi_plus = wfn_plus.evaluate(&positions).abs().ln();
            let ln_psi_minus = wfn_minus.evaluate(&positions).abs().ln();
            let numerical = (ln_psi_plus - ln_psi_minus) / (2.0 * dp);
            
            let diff = (analytical[k] - numerical).abs();
            let scale = analytical[k].abs().max(1e-6);
            assert!(diff / scale < 0.01,
                "Log-derivative mismatch for param {}: analytical={}, numerical={}, diff={}",
                k, analytical[k], numerical, diff);
        }
    }
}
