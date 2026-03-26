//! Benzene (C₆H₆) molecule wavefunction for QMC calculations.
//!
//! Implements a Slater-Jastrow trial wavefunction for benzene with 42 electrons
//! (36 from Carbon, 6 from Hydrogen) in a planar D₆h geometry.
//! Uses the same 6-31G Gaussian basis as MethaneGTO.

use nalgebra::Vector3;
use rand_distr::Normal;
use crate::correlation::Jastrow3;
use crate::sampling::EnergyCalculator;
use crate::wavefunction::{MultiWfn, OptimizableWfn};
use super::methane::{GaussianPrimitive, CGTO};

// =============================================================================
// Geometry constants
// =============================================================================

/// C-C bond length in benzene: 1.397 Å ≈ 2.640 Bohr
const CC_BOND_LENGTH: f64 = 2.640;

/// C-H bond length in benzene: 1.081 Å ≈ 2.042 Bohr
const CH_BOND_LENGTH: f64 = 2.042;

/// Number of electrons in benzene (6×6 + 6×1 = 42)
const NUM_ELECTRONS: usize = 42;

/// Number of occupied molecular orbitals (42/2 = 21)
const NUM_MOS: usize = 21;

/// Number of carbon atoms
const NUM_C: usize = 6;

/// Number of hydrogen atoms
const NUM_H: usize = 6;

// =============================================================================
// BenzeneGTO struct
// =============================================================================

/// Benzene (C₆H₆) wavefunction using 6-31G Gaussian basis and Jastrow3.
///
/// Features:
/// - 6-31G split-valence basis (66 AOs: 9 per C, 2 per H)
/// - Symmetry-adapted MO coefficients for σ and π orbitals
/// - Spin-dependent electron-electron Jastrow
/// - Electron-nucleus Jastrow satisfying Kato cusp
/// - 4 optimizable Jastrow parameters
#[derive(Clone)]
pub struct BenzeneGTO {
    /// Carbon positions (6 atoms in hexagonal ring, z=0 plane)
    carbons: [Vector3<f64>; NUM_C],
    /// Hydrogen positions (6 atoms, coplanar)
    hydrogens: [Vector3<f64>; NUM_H],
    /// Atomic orbital basis functions (66 total: 9 per C + 2 per H)
    ao_basis: Vec<CGTO>,
    /// MO coefficient matrix: mo_coeffs[mo_idx][ao_idx] (21 MOs × 66 AOs)
    mo_coeffs: Vec<Vec<f64>>,
    /// Jastrow factor with e-e and e-n terms
    pub jastrow: Jastrow3,
    /// Number of electrons (always 42)
    num_electrons: usize,
    /// Spin assignments: 21 up, 21 down
    spins: Vec<i32>,
}

impl BenzeneGTO {
    /// Create new BenzeneGTO with 6-31G split-valence basis.
    ///
    /// b_ee: electron-electron Jastrow decay parameter
    /// b_en: electron-nucleus Jastrow decay parameter
    pub fn new(b_ee: f64, b_en: f64) -> Self {
        // Build D₆h geometry: all atoms in z=0 plane
        let (carbons, hydrogens) = Self::build_geometry();
        
        // Build 6-31G basis
        let ao_basis = Self::build_631g_basis(&carbons, &hydrogens);
        
        // Build MO coefficients
        let mo_coeffs = Self::build_mo_coefficients(&carbons, &hydrogens);
        
        // Spin assignments: 21 up, 21 down
        let mut spins = Vec::with_capacity(NUM_ELECTRONS);
        for _ in 0..NUM_MOS { spins.push(1); }
        for _ in 0..NUM_MOS { spins.push(-1); }
        
        // Jastrow factor
        let mut nuclei = Vec::with_capacity(NUM_C + NUM_H);
        let mut charges = Vec::with_capacity(NUM_C + NUM_H);
        for c in &carbons {
            nuclei.push(*c);
            charges.push(6.0);
        }
        for h in &hydrogens {
            nuclei.push(*h);
            charges.push(1.0);
        }
        
        let jastrow = Jastrow3::new_general(
            NUM_ELECTRONS, b_ee, b_en,
            nuclei, charges, spins.clone(),
        );
        
        Self {
            carbons,
            hydrogens,
            ao_basis,
            mo_coeffs,
            jastrow,
            num_electrons: NUM_ELECTRONS,
            spins,
        }
    }
    
    /// Build D₆h hexagonal geometry for benzene.
    ///
    /// Carbon atoms are placed in a regular hexagon in the z=0 plane,
    /// centered at the origin. Hydrogen atoms are placed radially
    /// outward from each carbon.
    fn build_geometry() -> ([Vector3<f64>; NUM_C], [Vector3<f64>; NUM_H]) {
        let mut carbons = [Vector3::zeros(); NUM_C];
        let mut hydrogens = [Vector3::zeros(); NUM_H];
        
        for i in 0..NUM_C {
            let angle = std::f64::consts::PI / 3.0 * i as f64;
            carbons[i] = Vector3::new(
                CC_BOND_LENGTH * angle.cos(),
                CC_BOND_LENGTH * angle.sin(),
                0.0,
            );
            // Hydrogen radially outward from carbon
            let rh = CC_BOND_LENGTH + CH_BOND_LENGTH;
            hydrogens[i] = Vector3::new(
                rh * angle.cos(),
                rh * angle.sin(),
                0.0,
            );
        }
        
        (carbons, hydrogens)
    }
    
    /// Build 6-31G split-valence basis set for benzene.
    ///
    /// Returns 66 CGTOs:
    ///   [0..8]   C1: 1s(6p) + 2si(3p) + 2px/y/z_i(3p each) + 2so(1p) + 2px/y/z_o(1p each)
    ///   [9..17]  C2: same
    ///   ...
    ///   [45..53] C6: same
    ///   [54..55] H1: 1s_inner(3p) + 1s_outer(1p)
    ///   ...
    ///   [64..65] H6: same
    fn build_631g_basis(
        carbons: &[Vector3<f64>; NUM_C],
        hydrogens: &[Vector3<f64>; NUM_H],
    ) -> Vec<CGTO> {
        let mut basis = Vec::with_capacity(66);
        
        // ===== 6-31G Carbon basis (same exponents as methane) =====
        
        // Carbon 1s core (6 primitives)
        let c_1s_template = vec![
            GaussianPrimitive { exponent: 3047.5249, coefficient: 0.0018347 },
            GaussianPrimitive { exponent: 457.36951, coefficient: 0.0140373 },
            GaussianPrimitive { exponent: 103.94869, coefficient: 0.0688426 },
            GaussianPrimitive { exponent: 29.210155, coefficient: 0.2321844 },
            GaussianPrimitive { exponent: 9.2866630, coefficient: 0.4679413 },
            GaussianPrimitive { exponent: 3.1639270, coefficient: 0.3623120 },
        ];
        
        // Carbon SP inner shell (3 primitives)
        let c_sp_inner_exp = [7.8682724, 1.8812885, 0.5442493];
        let c_sp_inner_s_coef = [-0.1193324, -0.1608542, 1.1434564];
        let c_sp_inner_p_coef = [0.0689991, 0.3164240, 0.7443083];
        
        // Carbon SP outer shell exponent
        let c_sp_outer_exp = 0.1687145;
        
        for carbon in carbons {
            // C 1s core
            basis.push(CGTO {
                center: *carbon,
                primitives: c_1s_template.clone(),
                l: 0, m: 0,
            });
            
            // C 2s inner
            let c_2s_inner: Vec<GaussianPrimitive> = c_sp_inner_exp.iter()
                .zip(c_sp_inner_s_coef.iter())
                .map(|(&e, &c)| GaussianPrimitive { exponent: e, coefficient: c })
                .collect();
            basis.push(CGTO { center: *carbon, primitives: c_2s_inner, l: 0, m: 0 });
            
            // C 2px/y/z inner
            for m in 0..3u8 {
                let prims: Vec<GaussianPrimitive> = c_sp_inner_exp.iter()
                    .zip(c_sp_inner_p_coef.iter())
                    .map(|(&e, &c)| GaussianPrimitive { exponent: e, coefficient: c })
                    .collect();
                basis.push(CGTO { center: *carbon, primitives: prims, l: 1, m });
            }
            
            // C 2s' outer
            basis.push(CGTO {
                center: *carbon,
                primitives: vec![GaussianPrimitive { exponent: c_sp_outer_exp, coefficient: 1.0 }],
                l: 0, m: 0,
            });
            
            // C 2px'/y'/z' outer
            for m in 0..3u8 {
                basis.push(CGTO {
                    center: *carbon,
                    primitives: vec![GaussianPrimitive { exponent: c_sp_outer_exp, coefficient: 1.0 }],
                    l: 1, m,
                });
            }
        }
        
        // ===== 6-31G Hydrogen basis =====
        
        let h_inner_template = vec![
            GaussianPrimitive { exponent: 18.7311370, coefficient: 0.0334946 },
            GaussianPrimitive { exponent: 2.8253937,  coefficient: 0.2347270 },
            GaussianPrimitive { exponent: 0.6401217,  coefficient: 0.8137573 },
        ];
        let h_outer_exp = 0.1612778;
        
        for hydrogen in hydrogens {
            // H 1s inner
            basis.push(CGTO {
                center: *hydrogen,
                primitives: h_inner_template.clone(),
                l: 0, m: 0,
            });
            // H 1s' outer
            basis.push(CGTO {
                center: *hydrogen,
                primitives: vec![GaussianPrimitive { exponent: h_outer_exp, coefficient: 1.0 }],
                l: 0, m: 0,
            });
        }
        
        assert_eq!(basis.len(), 66, "6-31G basis for C6H6 should have 66 AOs");
        basis
    }
    
    /// Build approximate RHF/6-31G MO coefficients for benzene.
    ///
    /// AO ordering (66 total, 9 per carbon + 2 per hydrogen):
    ///   C_n: [1s, 2si, 2pxi, 2pyi, 2pzi, 2so, 2pxo, 2pyo, 2pzo]
    ///   H_n: [1si, 1so]
    ///
    /// Carbon AO offsets: C_n starts at index 9*n
    /// Hydrogen AO offsets: H_n starts at index 54 + 2*n
    ///
    /// MO ordering (21 doubly-occupied):
    ///   0-5:   C 1s core (localized on each carbon)
    ///   6-8:   C-C σ bonding (Hückel-like symmetric combinations)
    ///   9-14:  C-H σ bonding (each C-H bond)
    ///   15-17: C-C π bonding (out-of-plane pz)
    ///   18-20: Additional C-C σ bonding
    fn build_mo_coefficients(
        carbons: &[Vector3<f64>; NUM_C],
        hydrogens: &[Vector3<f64>; NUM_H],
    ) -> Vec<Vec<f64>> {
        let n_ao = 66;
        let mut mos: Vec<Vec<f64>> = Vec::with_capacity(NUM_MOS);
        
        // Helper: carbon AO offset for atom n
        let c_offset = |n: usize| 9 * n;
        // Helper: hydrogen AO offset for atom n  
        let h_offset = |n: usize| 54 + 2 * n;
        
        // -----------------------------------------------------------------
        // MO 0-5: Core 1s orbitals (localized on each carbon)
        // -----------------------------------------------------------------
        for n in 0..NUM_C {
            let mut mo = vec![0.0; n_ao];
            mo[c_offset(n)] = 0.9943;  // C_n 1s
            // Small tails from neighboring C 2s inner
            mo[c_offset(n) + 1] = 0.0234;  // C_n 2si
            mo[c_offset(n) + 5] = 0.0020;  // C_n 2so
            mos.push(mo);
        }
        
        // -----------------------------------------------------------------
        // MO 6-8: C-C σ bonding — delocalized ring σ orbitals
        //
        // These are formed from C 2s + in-plane C 2p orbitals pointing
        // along C-C bonds, using Hückel-like phase patterns.
        // -----------------------------------------------------------------
        
        // For in-plane σ bonding, each carbon contributes its 2s (inner+outer)
        // and the in-plane 2p component directed along the bisector of its
        // two C-C bonds (which points radially outward from the ring center).
        
        // MO 6: σ bonding — fully symmetric (a1g): all C 2s in phase
        {
            let mut mo = vec![0.0; n_ao];
            let c_s = 1.0 / (6.0_f64).sqrt();
            for n in 0..NUM_C {
                mo[c_offset(n) + 1] = 0.20 * c_s;  // C 2si
                mo[c_offset(n) + 5] = 0.80 * c_s;  // C 2so
            }
            mos.push(mo);
        }
        
        // MO 7: σ bonding — e1u (x-like node pattern)
        {
            let mut mo = vec![0.0; n_ao];
            // Phase pattern: cos(angle_n) for each carbon
            for n in 0..NUM_C {
                let angle = std::f64::consts::PI / 3.0 * n as f64;
                let phase = angle.cos();
                let weight = phase / 3.0_f64.sqrt();
                mo[c_offset(n) + 1] = 0.20 * weight;
                mo[c_offset(n) + 5] = 0.80 * weight;
                // In-plane p radical component (px)
                mo[c_offset(n) + 2] = 0.50 * weight;  // 2pxi
                mo[c_offset(n) + 6] = 0.55 * weight;  // 2pxo
            }
            mos.push(mo);
        }
        
        // MO 8: σ bonding — e1u (y-like node pattern)
        {
            let mut mo = vec![0.0; n_ao];
            for n in 0..NUM_C {
                let angle = std::f64::consts::PI / 3.0 * n as f64;
                let phase = angle.sin();
                let weight = phase / 3.0_f64.sqrt();
                mo[c_offset(n) + 1] = 0.20 * weight;
                mo[c_offset(n) + 5] = 0.80 * weight;
                // In-plane p (py)
                mo[c_offset(n) + 3] = 0.50 * weight;  // 2pyi
                mo[c_offset(n) + 7] = 0.55 * weight;  // 2pyo
            }
            mos.push(mo);
        }
        
        // -----------------------------------------------------------------
        // MO 9-14: C-H σ bonding orbitals
        //
        // Each MO is localized on one C-H bond: carbon radial 2p + H 1s
        // -----------------------------------------------------------------
        for n in 0..NUM_H {
            let mut mo = vec![0.0; n_ao];
            
            // Direction from carbon to hydrogen (radially outward)
            let dir = (hydrogens[n] - carbons[n]).normalize();
            
            // Carbon contributions: 2s + directed 2p
            mo[c_offset(n) + 1] = 0.15;   // C 2si
            mo[c_offset(n) + 5] = 0.30;   // C 2so
            // px, py, pz components along C-H direction
            mo[c_offset(n) + 2] = 0.20 * dir.x;  // 2pxi
            mo[c_offset(n) + 3] = 0.20 * dir.y;  // 2pyi
            mo[c_offset(n) + 4] = 0.20 * dir.z;  // 2pzi
            mo[c_offset(n) + 6] = 0.55 * dir.x;  // 2pxo
            mo[c_offset(n) + 7] = 0.55 * dir.y;  // 2pyo
            mo[c_offset(n) + 8] = 0.55 * dir.z;  // 2pzo
            
            // Hydrogen contributions
            mo[h_offset(n)]     = 0.25;  // H 1si
            mo[h_offset(n) + 1] = 0.15;  // H 1so
            
            mos.push(mo);
        }
        
        // -----------------------------------------------------------------
        // MO 15-17: C-C π bonding (out-of-plane C 2pz)
        //
        // Hückel π orbitals with energies 2β, β, β (3 occupied levels)
        // Phase patterns: all-in-phase (a2u), and degenerate e1g pair
        // -----------------------------------------------------------------
        
        // MO 15: π bonding — a2u (fully symmetric, all pz in phase)
        {
            let mut mo = vec![0.0; n_ao];
            let c_pz = 1.0 / (6.0_f64).sqrt();
            for n in 0..NUM_C {
                mo[c_offset(n) + 4] = 0.30 * c_pz;  // 2pzi
                mo[c_offset(n) + 8] = 0.70 * c_pz;  // 2pzo
            }
            mos.push(mo);
        }
        
        // MO 16: π bonding — e1g (x-like: cos pattern)
        {
            let mut mo = vec![0.0; n_ao];
            for n in 0..NUM_C {
                let angle = std::f64::consts::PI / 3.0 * n as f64;
                let phase = angle.cos() / 3.0_f64.sqrt();
                mo[c_offset(n) + 4] = 0.30 * phase;  // 2pzi
                mo[c_offset(n) + 8] = 0.70 * phase;  // 2pzo
            }
            mos.push(mo);
        }
        
        // MO 17: π bonding — e1g (y-like: sin pattern)
        {
            let mut mo = vec![0.0; n_ao];
            for n in 0..NUM_C {
                let angle = std::f64::consts::PI / 3.0 * n as f64;
                let phase = angle.sin() / 3.0_f64.sqrt();
                mo[c_offset(n) + 4] = 0.30 * phase;  // 2pzi
                mo[c_offset(n) + 8] = 0.70 * phase;  // 2pzo
            }
            mos.push(mo);
        }
        
        // -----------------------------------------------------------------
        // MO 18-20: Additional C-C σ bonding
        //
        // In-plane bonding MOs using tangential 2p components
        // (these complete the σ framework of the ring)
        // -----------------------------------------------------------------
        
        // MO 18: σ ring — tangential, fully symmetric
        {
            let mut mo = vec![0.0; n_ao];
            let norm = 1.0 / (6.0_f64).sqrt();
            for n in 0..NUM_C {
                let angle = std::f64::consts::PI / 3.0 * n as f64;
                // Tangent direction (-sin, cos, 0) perpendicular to radial
                let tx = -angle.sin();
                let ty = angle.cos();
                mo[c_offset(n) + 2] = 0.25 * tx * norm;  // 2pxi
                mo[c_offset(n) + 3] = 0.25 * ty * norm;  // 2pyi
                mo[c_offset(n) + 6] = 0.60 * tx * norm;  // 2pxo
                mo[c_offset(n) + 7] = 0.60 * ty * norm;  // 2pyo
            }
            mos.push(mo);
        }
        
        // MO 19: σ ring — tangential, e2g (cos 2θ pattern)
        {
            let mut mo = vec![0.0; n_ao];
            for n in 0..NUM_C {
                let angle = std::f64::consts::PI / 3.0 * n as f64;
                let phase = (2.0 * angle).cos() / 3.0_f64.sqrt();
                let tx = -angle.sin();
                let ty = angle.cos();
                mo[c_offset(n) + 2] = 0.25 * tx * phase;
                mo[c_offset(n) + 3] = 0.25 * ty * phase;
                mo[c_offset(n) + 6] = 0.60 * tx * phase;
                mo[c_offset(n) + 7] = 0.60 * ty * phase;
            }
            mos.push(mo);
        }
        
        // MO 20: σ ring — tangential, e2g (sin 2θ pattern)
        {
            let mut mo = vec![0.0; n_ao];
            for n in 0..NUM_C {
                let angle = std::f64::consts::PI / 3.0 * n as f64;
                let phase = (2.0 * angle).sin() / 3.0_f64.sqrt();
                let tx = -angle.sin();
                let ty = angle.cos();
                mo[c_offset(n) + 2] = 0.25 * tx * phase;
                mo[c_offset(n) + 3] = 0.25 * ty * phase;
                mo[c_offset(n) + 6] = 0.60 * tx * phase;
                mo[c_offset(n) + 7] = 0.60 * ty * phase;
            }
            mos.push(mo);
        }
        
        assert_eq!(mos.len(), NUM_MOS, "Benzene should have {} MOs", NUM_MOS);
        mos
    }
    
    /// Evaluate molecular orbital at position r.
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
            .fold(Vector3::zeros(), |acc, (&c, ao)| acc + c * ao.gradient(r))
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
    
    /// Build Slater matrix for one spin sector and return (det, inverse).
    fn slater_sector(&self, r: &[Vector3<f64>], spin: i32) -> (f64, nalgebra::DMatrix<f64>) {
        let indices: Vec<usize> = self.spins.iter()
            .enumerate()
            .filter(|(_, &s)| s == spin)
            .map(|(i, _)| i)
            .collect();
        
        let n = indices.len(); // 21
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
        
        let mut gradients = vec![Vector3::zeros(); NUM_ELECTRONS];
        
        for (row, &elec_idx) in up_indices.iter().enumerate() {
            let grad_log: Vector3<f64> = (0..NUM_MOS)
                .map(|col| inv_up[(col, row)] * self.mo_derivative(col, &r[elec_idx]))
                .sum();
            gradients[elec_idx] = det_up * det_down * grad_log;
        }
        
        for (row, &elec_idx) in down_indices.iter().enumerate() {
            let grad_log: Vector3<f64> = (0..NUM_MOS)
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
        
        let mut laplacians = vec![0.0; NUM_ELECTRONS];
        
        for (row, &elec_idx) in up_indices.iter().enumerate() {
            let lap_log: f64 = (0..NUM_MOS)
                .map(|col| inv_up[(col, row)] * self.mo_laplacian(col, &r[elec_idx]))
                .sum();
            laplacians[elec_idx] = det_up * det_down * lap_log;
        }
        
        for (row, &elec_idx) in down_indices.iter().enumerate() {
            let lap_log: f64 = (0..NUM_MOS)
                .map(|col| inv_down[(col, row)] * self.mo_laplacian(col, &r[elec_idx]))
                .sum();
            laplacians[elec_idx] = det_up * det_down * lap_log;
        }
        
        laplacians
    }
    
    /// Nuclear-nuclear repulsion energy (constant for fixed geometry).
    pub fn nuclear_repulsion(&self) -> f64 {
        let z_c = 6.0;
        let z_h = 1.0;
        let mut v_nn = 0.0;
        
        // C-C repulsion
        for i in 0..NUM_C {
            for j in (i + 1)..NUM_C {
                v_nn += z_c * z_c / (self.carbons[i] - self.carbons[j]).norm();
            }
        }
        
        // C-H repulsion
        for c in &self.carbons {
            for h in &self.hydrogens {
                v_nn += z_c * z_h / (c - h).norm();
            }
        }
        
        // H-H repulsion
        for i in 0..NUM_H {
            for j in (i + 1)..NUM_H {
                v_nn += z_h * z_h / (self.hydrogens[i] - self.hydrogens[j]).norm();
            }
        }
        
        v_nn
    }
}

// =============================================================================
// Trait implementations
// =============================================================================

impl MultiWfn for BenzeneGTO {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 0.5).unwrap();
        use rand_distr::Distribution;
        
        // Distribute electrons: 6 near each C (36 total), 1 near each H (6 total)
        (0..self.num_electrons)
            .map(|i| {
                let center = if i < 36 {
                    self.carbons[i / 6]
                } else {
                    self.hydrogens[i - 36]
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

impl EnergyCalculator for BenzeneGTO {
    fn local_energy(&self, r: &[Vector3<f64>]) -> f64 {
        let psi = self.evaluate(r);
        let laplacian = self.laplacian(r);
        
        // Kinetic energy: -½ Σᵢ ∇²ψ/ψ
        let kinetic = -0.5 * laplacian.iter().sum::<f64>() / psi;
        
        // Electron-nucleus attraction
        let z_c = 6.0;
        let z_h = 1.0;
        
        // Electron-Carbon attraction
        let v_ec: f64 = r.iter()
            .flat_map(|ri| self.carbons.iter().map(move |c| -z_c / (ri - c).norm()))
            .sum();
        
        // Electron-Hydrogen attraction
        let v_eh: f64 = r.iter()
            .flat_map(|ri| self.hydrogens.iter().map(move |h| -z_h / (ri - h).norm()))
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

impl OptimizableWfn for BenzeneGTO {
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
        assert_eq!(params.len(), 4, "BenzeneGTO has exactly 4 Jastrow parameters");
        self.jastrow.b_ee = params[0];
        self.jastrow.b_en = params[1];
        self.jastrow.a_ee_anti = params[2];
        self.jastrow.a_ee_para = params[3];
    }
    
    fn log_derivatives(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        vec![
            self.jastrow.d_ln_j_d_bee(r),
            self.jastrow.d_ln_j_d_ben(r),
            self.jastrow.d_ln_j_d_a_anti(r),
            self.jastrow.d_ln_j_d_a_para(r),
        ]
    }
}

// =============================================================================
// Mutable geometry and ForceCalculator
// =============================================================================

impl BenzeneGTO {
    /// Get nuclear positions: [C1..C6, H1..H6] (12 positions).
    pub fn get_nuclei(&self) -> Vec<Vector3<f64>> {
        let mut nuclei = Vec::with_capacity(NUM_C + NUM_H);
        for c in &self.carbons {
            nuclei.push(*c);
        }
        for h in &self.hydrogens {
            nuclei.push(*h);
        }
        nuclei
    }
    
    /// Get nuclear charges: [6.0 × 6, 1.0 × 6].
    pub fn get_charges(&self) -> Vec<f64> {
        let mut charges = Vec::with_capacity(NUM_C + NUM_H);
        for _ in 0..NUM_C { charges.push(6.0); }
        for _ in 0..NUM_H { charges.push(1.0); }
        charges
    }
    
    /// Set nuclear positions and rebuild basis and Jastrow.
    ///
    /// nuclei: [C1..C6, H1..H6] (12 positions)
    pub fn set_nuclei(&mut self, nuclei: &[Vector3<f64>]) {
        assert_eq!(nuclei.len(), NUM_C + NUM_H,
            "BenzeneGTO requires exactly {} nuclei: [C1..C6, H1..H6]", NUM_C + NUM_H);
        for i in 0..NUM_C {
            self.carbons[i] = nuclei[i];
        }
        for i in 0..NUM_H {
            self.hydrogens[i] = nuclei[NUM_C + i];
        }
        // Rebuild AO basis at new centers
        self.ao_basis = Self::build_631g_basis(&self.carbons, &self.hydrogens);
        // Rebuild MO coefficients (they depend on C-H directions)
        self.mo_coeffs = Self::build_mo_coefficients(&self.carbons, &self.hydrogens);
        // Update Jastrow nucleus positions
        for i in 0..NUM_C {
            self.jastrow.nuclei[i] = self.carbons[i];
        }
        for i in 0..NUM_H {
            self.jastrow.nuclei[NUM_C + i] = self.hydrogens[i];
        }
    }
}

use crate::sampling::ForceCalculator;

impl ForceCalculator for BenzeneGTO {
    fn num_nuclei(&self) -> usize {
        NUM_C + NUM_H // 12
    }

    fn get_nuclei(&self) -> Vec<Vector3<f64>> {
        BenzeneGTO::get_nuclei(self)
    }

    fn get_charges(&self) -> Vec<f64> {
        BenzeneGTO::get_charges(self)
    }

    fn set_nuclei(&mut self, nuclei: &[Vector3<f64>]) {
        BenzeneGTO::set_nuclei(self, nuclei)
    }

    fn hellmann_feynman_force(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let nuclei = self.get_nuclei();
        let charges = self.get_charges();
        let n_nuc = nuclei.len();
        let mut forces = vec![Vector3::zeros(); n_nuc];

        // Electron-nucleus attraction: F_I += Z_I Σ_i (r_i - R_I) / |r_i - R_I|³
        for (nuc_idx, nuc_pos) in nuclei.iter().enumerate() {
            let z_i = charges[nuc_idx];
            for elec_pos in r.iter() {
                let dr = elec_pos - nuc_pos;
                let dist = dr.norm();
                if dist > 1e-10 {
                    forces[nuc_idx] += z_i * dr / (dist * dist * dist);
                }
            }
        }

        // Nuclear-nuclear repulsion: F_I -= Σ_{J≠I} Z_I Z_J (R_I - R_J) / |R_I - R_J|³
        for i in 0..n_nuc {
            for j in 0..n_nuc {
                if i == j { continue; }
                let dr = nuclei[i] - nuclei[j];
                let dist = dr.norm();
                if dist > 1e-10 {
                    forces[i] -= charges[i] * charges[j] * dr / (dist * dist * dist);
                }
            }
        }

        forces
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benzene_geometry() {
        let (carbons, hydrogens) = BenzeneGTO::build_geometry();
        
        // All C atoms should be at distance CC_BOND_LENGTH from origin
        for c in &carbons {
            assert!((c.norm() - CC_BOND_LENGTH).abs() < 1e-10,
                "Carbon should be at R={} from origin, got {}", CC_BOND_LENGTH, c.norm());
        }
        
        // All C-C nearest-neighbor distances should be CC_BOND_LENGTH
        for i in 0..NUM_C {
            let j = (i + 1) % NUM_C;
            let d = (carbons[i] - carbons[j]).norm();
            assert!((d - CC_BOND_LENGTH).abs() < 0.01,
                "C{}-C{} distance should be ~{}, got {}", i, j, CC_BOND_LENGTH, d);
        }
        
        // C-C-C angles should be 120°
        for i in 0..NUM_C {
            let prev = (i + NUM_C - 1) % NUM_C;
            let next = (i + 1) % NUM_C;
            let v1 = (carbons[prev] - carbons[i]).normalize();
            let v2 = (carbons[next] - carbons[i]).normalize();
            let angle_deg = v1.dot(&v2).acos().to_degrees();
            assert!((angle_deg - 120.0).abs() < 0.1,
                "C-C-C angle at C{} should be 120°, got {:.1}°", i, angle_deg);
        }
        
        // All atoms should be in z=0 plane
        for c in &carbons {
            assert!(c.z.abs() < 1e-10, "Carbon z should be 0");
        }
        for h in &hydrogens {
            assert!(h.z.abs() < 1e-10, "Hydrogen z should be 0");
        }
        
        // Each C-H distance should be CH_BOND_LENGTH
        for i in 0..NUM_H {
            let d = (carbons[i] - hydrogens[i]).norm();
            assert!((d - CH_BOND_LENGTH).abs() < 0.01,
                "C-H bond {} length should be ~{}, got {}", i, CH_BOND_LENGTH, d);
        }
    }
    
    #[test]
    fn test_benzene_basis_size() {
        let wfn = BenzeneGTO::new(1.5, 2.0);
        assert_eq!(wfn.ao_basis.len(), 66, "6-31G basis should have 66 AOs");
        assert_eq!(wfn.mo_coeffs.len(), NUM_MOS, "Should have {} MOs", NUM_MOS);
        for (i, mo) in wfn.mo_coeffs.iter().enumerate() {
            assert_eq!(mo.len(), 66, "MO {} should have 66 coefficients", i);
        }
    }
    
    #[test]
    fn test_benzene_nuclear_repulsion() {
        let wfn = BenzeneGTO::new(1.5, 2.0);
        let v_nn = wfn.nuclear_repulsion();
        // V_nn for benzene should be ~200-210 Ha
        assert!(v_nn > 180.0 && v_nn < 230.0,
            "Nuclear repulsion should be ~200-210 Ha, got {:.2}", v_nn);
    }
    
    #[test]
    fn test_benzene_electron_count() {
        let wfn = BenzeneGTO::new(1.5, 2.0);
        assert_eq!(wfn.num_electrons, 42);
        assert_eq!(wfn.spins.len(), 42);
        let n_up = wfn.spins.iter().filter(|&&s| s == 1).count();
        let n_down = wfn.spins.iter().filter(|&&s| s == -1).count();
        assert_eq!(n_up, 21);
        assert_eq!(n_down, 21);
    }
    
    #[test]
    fn test_benzene_wfn_evaluates() {
        let wfn = BenzeneGTO::new(1.5, 2.0);
        let positions = wfn.initialize();
        let psi = wfn.evaluate(&positions);
        assert!(psi.is_finite(), "Wavefunction should be finite, got {}", psi);
    }
    
    #[test]
    fn test_benzene_numerical_derivative() {
        let wfn = BenzeneGTO::new(1.5, 2.0);
        let positions = wfn.initialize();
        
        let analytical_deriv = wfn.derivative(&positions);
        let numerical_deriv = wfn.numerical_derivative(&positions, 1e-5);
        
        // Check a subset of electrons to keep test fast
        for i in [0, 10, 20, 30, 40].iter().copied().filter(|&i| i < NUM_ELECTRONS) {
            for axis in 0..3 {
                let a = analytical_deriv[i][axis];
                let n = numerical_deriv[i][axis];
                let diff = (a - n).abs();
                let scale = a.abs().max(n.abs()).max(1e-8);
                assert!(diff / scale < 0.15,
                    "Derivative mismatch at electron {}, axis {}: analytical={:.6e}, numerical={:.6e}",
                    i, axis, a, n);
            }
        }
    }
    
    #[test]
    fn test_benzene_log_derivatives() {
        let wfn = BenzeneGTO::new(2.0, 3.0);
        let positions = wfn.initialize();
        
        let analytical = wfn.log_derivatives(&positions);
        
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
