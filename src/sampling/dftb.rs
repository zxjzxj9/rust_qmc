//! Simplified Empirical Tight-Binding (ETB) Engine for Path Integral Molecular Dynamics
//!
//! This module implements a toy quantum mechanical tight-binding model
//! (similar to Extended Hückel or simplified DFTB) for the Zundel cation (H5O2+).
//!
//! It solves the electronic Schrödinger equation HC = EC in a minimal basis set
//! to obtain the electronic energy, and adds an empirical pairwise repulsive
//! potential to stabilize the nuclei.
//!
//! # Method Details
//! - **Basis set**: Minimal atomic orbital basis (s and p for Oxygen, s for Hydrogen)
//!   - O1 (atoms 0, 4): 2s, 2px, 2py, 2pz (4 orbitals)
//!   - H (atoms 1, 2, 3, 5, 6): 1s (1 orbital)
//!   - Total basis size: 13 orbitals (13x13 Hamiltonian matrix)
//! - **Orthogonal approximation**: S = I (Standard symmetric eigenvalue problem)
//! - **Matrix elements**: On-site energies on the diagonal, Slater-Koster hopping
//!   integrals on the off-diagonals with exponential distance decay.
//! - **Forces**: Computed via numerical finite-difference, which is feasible
//!   because diagonalizing a 13x13 symmetric matrix is extremely fast (~μs).

use std::f64;
use nalgebra::{DMatrix, SymmetricEigen};
use super::pimd_molecular::MolecularPotential;

// =============================================================================
// Slater-Koster Parameters
// =============================================================================

#[derive(Clone, Debug)]
pub struct ETBParameters {
    // On-site energies (Hartree)
    pub e_o_s: f64,
    pub e_o_p: f64,
    pub e_h_s: f64,

    // Hopping amplitude prefactors (Hartree)
    pub v_ss_sigma: f64,
    pub v_sp_sigma: f64,
    pub v_pp_sigma: f64,
    pub v_pp_pi: f64,
    
    // Distance decay factor (1/Bohr) for electronic hopping
    pub alpha_hop: f64,
    // Reference distance for hopping parameters (Bohr)
    pub r0_hop: f64,

    // Repulsion parameters: V_rep(r) = A * exp(-B * r)
    pub a_oo: f64,
    pub b_oo: f64,
    pub a_oh: f64,
    pub b_oh: f64,
    pub a_hh: f64,
    pub b_hh: f64,
}

impl ETBParameters {
    /// A set of "toy" parameters that give a stable H5O2+ structure
    /// with a proton transfer double well.
    pub fn new_toy_zundel() -> Self {
        Self {
            // Valence orbital energies (approximate ionization potentials)
            e_o_s: -1.2,
            e_o_p: -0.5,
            e_h_s: -0.4, // Shifted to favor charge transfer

            // Hopping integrals at r0 = 1.8 Bohr (typical O-H bond)
            // V_ss is negative, V_sp is positive, V_pp_sigma positive, V_pp_pi negative
            v_ss_sigma: -0.4,
            v_sp_sigma:  0.4,
            v_pp_sigma:  0.5,
            v_pp_pi:    -0.2,

            alpha_hop: 1.2, // Decay rate
            r0_hop: 1.8,    // Reference distance

            // Repulsion: A * exp(-B * r)
            a_oo: 30.0,
            b_oo: 1.8,
            a_oh: 8.0,
            b_oh: 2.0,
            a_hh: 3.0,
            b_hh: 2.5,
        }
    }

    /// Hopping scaling factor f(r) = exp(-alpha * (r - r0))
    #[inline]
    pub fn hop_scaling(&self, r: f64) -> f64 {
        (-self.alpha_hop * (r - self.r0_hop)).exp()
    }
}

// =============================================================================
// Toy DFTB / ETB Engine
// =============================================================================

#[derive(Clone)]
pub struct ToyDFTB {
    pub params: ETBParameters,
    /// Reference geometry for Zundel (from EVB PES)
    pub ref_geom: Vec<f64>,
    pub masses: [f64; 7],
}

impl ToyDFTB {
    pub fn new() -> Self {
        let m_o = 29156.95;
        let m_h = 1836.15;
        
        let mut t = Self {
            params: ETBParameters::new_toy_zundel(),
            ref_geom: vec![0.0; 21],
            masses: [m_o, m_h, m_h, m_h, m_o, m_h, m_h],
        };
        
        // Use a reasonable starting geometry (similar to Zundel PES)
        let r_oo = 4.5;
        let r_oh = 1.8;
        let theta = 109.5_f64.to_radians() / 2.0;
        let hy = r_oh * theta.sin();
        let hx = r_oh * theta.cos();
        
        t.ref_geom = vec![
            0.0, 0.0, 0.0,             // O1
            -hx, hy, 0.0,              // H1a
            -hx, -hy, 0.0,             // H1b
            r_oo/2.0, 0.0, 0.0,        // H* (centered)
            r_oo, 0.0, 0.0,            // O2
            r_oo+hx, 0.0, hy,          // H2a
            r_oo+hx, 0.0, -hy,         // H2b
        ];
        
        t
    }

    /// Map atom index to basis function starting index
    fn basis_offset(atom: usize) -> usize {
        match atom {
            0 => 0,          // O1 (4 orbitals: s, px, py, pz)
            1 => 4,          // H1a (1 orbital)
            2 => 5,          // H1b
            3 => 6,          // H*
            4 => 7,          // O2 (4 orbitals)
            5 => 11,         // H2a
            6 => 12,         // H2b
            _ => panic!("Invalid atom index"),
        }
    }

    /// Number of orbitals for a given atom
    fn n_orbitals(atom: usize) -> usize {
        if atom == 0 || atom == 4 { 4 } else { 1 }
    }

    /// Compute the 13x13 Hamiltonian matrix for a given geometry
    pub fn build_hamiltonian(&self, coords: &[f64]) -> DMatrix<f64> {
        let n_basis = 13;
        let mut h = DMatrix::zeros(n_basis, n_basis);

        // 1. On-site energies (diagonal elements)
        for atom in 0..7 {
            let offset = Self::basis_offset(atom);
            if atom == 0 || atom == 4 {
                // Oxygen: s, px, py, pz
                h[(offset, offset)] = self.params.e_o_s;
                h[(offset+1, offset+1)] = self.params.e_o_p;
                h[(offset+2, offset+2)] = self.params.e_o_p;
                h[(offset+3, offset+3)] = self.params.e_o_p;
            } else {
                // Hydrogen: s
                h[(offset, offset)] = self.params.e_h_s;
            }
        }

        // 2. Off-diagonal hopping (Slater-Koster)
        for a in 0..7 {
            for b in (a+1)..7 {
                // Distance and direction cosines
                let dx = coords[3*b + 0] - coords[3*a + 0];
                let dy = coords[3*b + 1] - coords[3*a + 1];
                let dz = coords[3*b + 2] - coords[3*a + 2];
                let r2 = dx*dx + dy*dy + dz*dz;
                let r = r2.sqrt();
                
                if r > 6.0 || r < 0.1 { continue; } // Cutoff
                
                let l = dx / r;
                let m = dy / r;
                let n = dz / r;
                
                let scale = self.params.hop_scaling(r);
                let v_ss = self.params.v_ss_sigma * scale;
                let v_sp = self.params.v_sp_sigma * scale;
                let v_pps = self.params.v_pp_sigma * scale;
                let v_ppp = self.params.v_pp_pi * scale;

                let off_a = Self::basis_offset(a);
                let off_b = Self::basis_offset(b);
                let is_a_oxy = a == 0 || a == 4;
                let is_b_oxy = b == 0 || b == 4;

                if !is_a_oxy && !is_b_oxy {
                    // H-H (s-s only)
                    let h_ab = v_ss;
                    h[(off_a, off_b)] = h_ab;
                    h[(off_b, off_a)] = h_ab;
                } else if is_a_oxy && !is_b_oxy {
                    // O-H (a=O, b=H)
                    // s-s
                    h[(off_a, off_b)] = v_ss;
                    h[(off_b, off_a)] = v_ss;
                    // px-s, py-s, pz-s (using l,m,n)
                    // Note: SK rule for p_a - s_b with vector pointing A->B is l*V_sp
                    h[(off_a+1, off_b)] = l * v_sp;
                    h[(off_b, off_a+1)] = l * v_sp;
                    
                    h[(off_a+2, off_b)] = m * v_sp;
                    h[(off_b, off_a+2)] = m * v_sp;
                    
                    h[(off_a+3, off_b)] = n * v_sp;
                    h[(off_b, off_a+3)] = n * v_sp;
                    
                } else if !is_a_oxy && is_b_oxy {
                    // H-O (a=H, b=O)
                    // s-s
                    h[(off_a, off_b)] = v_ss;
                    h[(off_b, off_a)] = v_ss;
                    // s_a - p_b with vector pointing A->B is -l*V_sp
                    h[(off_a, off_b+1)] = -l * v_sp;
                    h[(off_b+1, off_a)] = -l * v_sp;
                    
                    h[(off_a, off_b+2)] = -m * v_sp;
                    h[(off_b+2, off_a)] = -m * v_sp;
                    
                    h[(off_a, off_b+3)] = -n * v_sp;
                    h[(off_b+3, off_a)] = -n * v_sp;
                } else {
                    // O-O (a=O, b=O)
                    // s-s
                    h[(off_a, off_b)] = v_ss;
                    
                    // s_a - p_b
                    h[(off_a, off_b+1)] = l * v_sp;
                    h[(off_a, off_b+2)] = m * v_sp;
                    h[(off_a, off_b+3)] = n * v_sp;
                    
                    // p_a - s_b
                    h[(off_a+1, off_b)] = -l * v_sp;
                    h[(off_a+2, off_b)] = -m * v_sp;
                    h[(off_a+3, off_b)] = -n * v_sp;
                    
                    // p-p
                    h[(off_a+1, off_b+1)] = l*l*v_pps + (1.0-l*l)*v_ppp;
                    h[(off_a+1, off_b+2)] = l*m*(v_pps - v_ppp);
                    h[(off_a+1, off_b+3)] = l*n*(v_pps - v_ppp);
                    
                    h[(off_a+2, off_b+1)] = l*m*(v_pps - v_ppp);
                    h[(off_a+2, off_b+2)] = m*m*v_pps + (1.0-m*m)*v_ppp;
                    h[(off_a+2, off_b+3)] = m*n*(v_pps - v_ppp);
                    
                    h[(off_a+3, off_b+1)] = l*n*(v_pps - v_ppp);
                    h[(off_a+3, off_b+2)] = m*n*(v_pps - v_ppp);
                    h[(off_a+3, off_b+3)] = n*n*v_pps + (1.0-n*n)*v_ppp;
                    
                    // Make symmetric
                    for i in 0..4 {
                        for j in 0..4 {
                            h[(off_b+j, off_a+i)] = h[(off_a+i, off_b+j)];
                        }
                    }
                }
            }
        }

        h
    }

    /// Compute pairwise empirical repulsive energy
    pub fn repulsive_energy(&self, coords: &[f64]) -> f64 {
        let mut e_rep = 0.0;
        for a in 0..7 {
            for b in (a+1)..7 {
                let dx = coords[3*b + 0] - coords[3*a + 0];
                let dy = coords[3*b + 1] - coords[3*a + 1];
                let dz = coords[3*b + 2] - coords[3*a + 2];
                let r = (dx*dx + dy*dy + dz*dz).sqrt();
                
                let is_a_oxy = a == 0 || a == 4;
                let is_b_oxy = b == 0 || b == 4;

                let (A, B) = if is_a_oxy && is_b_oxy {
                    (self.params.a_oo, self.params.b_oo)
                } else if !is_a_oxy && !is_b_oxy {
                    (self.params.a_hh, self.params.b_hh)
                } else {
                    (self.params.a_oh, self.params.b_oh)
                };

                e_rep += A * (-B * r).exp();
            }
        }
        e_rep
    }

    /// Compute total energy (electronic + repulsive)
    pub fn compute_total_energy(&self, coords: &[f64]) -> f64 {
        // 1. Electronic energy
        let h_matrix = self.build_hamiltonian(coords);
        
        // Diagonalize
        let eigen = SymmetricEigen::new(h_matrix);
        let mut eigenvalues = eigen.eigenvalues.as_slice().to_vec();
        
        // Sort eigenvalues (ascending)
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Zundel cation has 16 valence electrons -> 8 doubly occupied orbitals
        let mut e_elec = 0.0;
        for i in 0..8 {
            e_elec += 2.0 * eigenvalues[i];
        }

        // 2. Repulsive energy
        let e_rep = self.repulsive_energy(coords);

        e_elec + e_rep
    }
}

impl MolecularPotential for ToyDFTB {
    fn n_atoms(&self) -> usize { 7 }

    fn energy(&self, coords: &[f64]) -> f64 {
        self.compute_total_energy(coords)
    }

    /// Numerical finite difference for forces.
    /// Since the 13x13 diagonalization takes < 1 μs, 42 evaluations is trivial.
    fn forces(&self, coords: &[f64], forces: &mut [f64]) {
        let h = 1e-5;
        let ndof = 21;
        
        let mut gp = coords.to_vec();
        let mut gm = coords.to_vec();
        
        for d in 0..ndof {
            gp[d] = coords[d] + h;
            gm[d] = coords[d] - h;
            
            let e_p = self.compute_total_energy(&gp);
            let e_m = self.compute_total_energy(&gm);
            
            forces[d] = -(e_p - e_m) / (2.0 * h);
            
            gp[d] = coords[d];
            gm[d] = coords[d];
        }
    }

    fn masses(&self) -> &[f64] {
        &self.masses
    }

    fn reference_geometry(&self) -> Vec<f64> {
        self.ref_geom.clone()
    }

    fn name(&self) -> &'static str {
        "Toy DFTB / ETB"
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dftb_hamiltonian_size() {
        let etb = ToyDFTB::new();
        let h = etb.build_hamiltonian(&etb.reference_geometry());
        assert_eq!(h.nrows(), 13);
        assert_eq!(h.ncols(), 13);
    }

    #[test]
    fn test_dftb_energy_continuity() {
        let etb = ToyDFTB::new();
        let geom = etb.reference_geometry();
        
        let e0 = etb.energy(&geom);
        
        let mut g1 = geom.clone();
        g1[0] += 1e-4; // Move O1 slightly
        
        let e1 = etb.energy(&g1);
        
        // Energy should be continuous
        assert!((e1 - e0).abs() < 1e-3, "Energy jump too large: {} -> {}", e0, e1);
    }

    #[test]
    fn test_dftb_double_well() {
        let etb = ToyDFTB::new();
        let mut geom = etb.reference_geometry();
        let r_oo = 4.5; // O2 is at x=4.5
        
        // Move H* (atom 3, starting at x=3*3=9)
        // O1 is at x=0, O2 is at x=r_oo
        
        // Energy at midpoint (TS)
        geom[9] = r_oo / 2.0;
        geom[10] = 0.0;
        geom[11] = 0.0;
        let e_ts = etb.energy(&geom);
        
        // Energy closer to O1 (minimum)
        geom[9] = 1.8;
        let e_min = etb.energy(&geom);
        
        assert!(e_ts > e_min, "TS energy ({:.4}) should be higher than min ({:.4})", e_ts, e_min);
        
        // Energy closer to O2 (symmetric minimum)
        geom[9] = r_oo - 1.8;
        let e_min2 = etb.energy(&geom);
        
        assert!((e_min - e_min2).abs() < 1e-5, "Wells should be symmetric: {:.4} vs {:.4}", e_min, e_min2);
    }
}
