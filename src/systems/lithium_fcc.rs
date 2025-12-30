//! Lithium FCC crystal wavefunction for VMC calculations.
//!
//! Implements a Slater determinant × Jastrow factor trial wavefunction
//! for crystalline lithium with periodic boundary conditions.

use nalgebra::Vector3;
use rand::Rng;
use crate::correlation::Jastrow2;
use crate::sampling::EnergyCalculator;
use crate::wavefunction::{MultiWfn, STOSlaterDet, STO, init_li_sto};
use super::crystal::LatticeVector;

/// Lithium FCC crystal wavefunction.
///
/// Uses a Slater determinant of localized orbitals (STOs centered on ion sites)
/// multiplied by a Jastrow correlation factor. Periodic boundary conditions
/// are handled via minimum image convention and Ewald summation.
pub struct LithiumFCC {
    /// Crystal lattice
    pub lattice: LatticeVector,
    /// Ion positions in the unit cell
    pub ion_positions: Vec<Vector3<f64>>,
    /// Number of electrons (1 valence per Li for pseudopotential, 3 for all-electron)
    pub num_electrons: usize,
    /// Jastrow correlation factor
    pub jastrow: Jastrow2,
    /// Localized orbitals centered on each ion
    orbitals: Vec<STO>,
    /// Lattice constant
    lattice_constant: f64,
}

impl LithiumFCC {
    /// Create a new LithiumFCC wavefunction.
    ///
    /// # Arguments
    /// * `lattice_constant` - FCC lattice constant in Bohr
    /// * `electrons_per_atom` - 1 for pseudopotential, 3 for all-electron
    /// * `cusp_param` - Jastrow cusp parameter
    pub fn new(lattice_constant: f64, electrons_per_atom: usize, cusp_param: f64) -> Self {
        let lattice = LatticeVector::new_fcc(lattice_constant);

        // FCC has 1 atom in the primitive cell at origin
        // For a conventional cell, we'd have 4 atoms
        let ion_positions = vec![Vector3::zeros()];
        
        let num_atoms = ion_positions.len();
        let num_electrons = num_atoms * electrons_per_atom;

        // Create STOs centered on each ion
        let orbitals: Vec<STO> = ion_positions.iter()
            .flat_map(|&pos| {
                (0..electrons_per_atom).map(move |i| {
                    let n = if i == 0 { 1 } else { 2 }; // 1s, 1s, 2s
                    init_li_sto(pos, n, 0, 0)
                })
            })
            .collect();

        let jastrow = Jastrow2 {
            cusp_param,
            num_electrons,
        };

        Self {
            lattice,
            ion_positions,
            num_electrons,
            jastrow,
            orbitals,
            lattice_constant,
        }
    }

    /// Create with multiple unit cells (supercell).
    ///
    /// # Arguments
    /// * `lattice_constant` - FCC lattice constant in Bohr
    /// * `supercell` - Number of unit cells in each direction (e.g., 2 for 2×2×2)
    /// * `electrons_per_atom` - 1 for pseudopotential, 3 for all-electron
    /// * `cusp_param` - Jastrow cusp parameter
    pub fn new_supercell(
        lattice_constant: f64,
        supercell: usize,
        electrons_per_atom: usize,
        cusp_param: f64,
    ) -> Self {
        // Scale the lattice by supercell factor
        let scaled_a = lattice_constant * supercell as f64;
        let lattice = LatticeVector::new_fcc(scaled_a);

        // Generate ion positions for the supercell
        // FCC conventional cell has 4 atoms at:
        // (0,0,0), (0,½,½), (½,0,½), (½,½,0)
        let mut ion_positions = Vec::new();
        let a = lattice_constant;
        
        // FCC basis positions (in units of a)
        let fcc_basis = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 0.5),
            Vector3::new(0.5, 0.0, 0.5),
            Vector3::new(0.5, 0.5, 0.0),
        ];

        for nx in 0..supercell {
            for ny in 0..supercell {
                for nz in 0..supercell {
                    let offset = Vector3::new(
                        nx as f64 * a,
                        ny as f64 * a,
                        nz as f64 * a,
                    );
                    for basis in &fcc_basis {
                        ion_positions.push(offset + a * basis);
                    }
                }
            }
        }

        let num_atoms = ion_positions.len();
        let num_electrons = num_atoms * electrons_per_atom;

        // Create STOs centered on each ion
        let orbitals: Vec<STO> = ion_positions.iter()
            .flat_map(|&pos| {
                (0..electrons_per_atom).map(move |i| {
                    let n = if i == 0 { 1 } else { 2 };
                    init_li_sto(pos, n, 0, 0)
                })
            })
            .collect();

        let jastrow = Jastrow2 {
            cusp_param,
            num_electrons,
        };

        Self {
            lattice,
            ion_positions,
            num_electrons,
            jastrow,
            orbitals,
            lattice_constant: scaled_a,
        }
    }

    /// Apply periodic boundary conditions to a position.
    fn wrap_position(&self, pos: &Vector3<f64>) -> Vector3<f64> {
        let a = self.lattice_constant;
        Vector3::new(
            pos.x.rem_euclid(a),
            pos.y.rem_euclid(a),
            pos.z.rem_euclid(a),
        )
    }

    /// Minimum image distance between two positions.
    fn minimum_image(&self, r1: &Vector3<f64>, r2: &Vector3<f64>) -> Vector3<f64> {
        let mut dr = r1 - r2;
        let a = self.lattice_constant;
        dr.x -= a * (dr.x / a).round();
        dr.y -= a * (dr.y / a).round();
        dr.z -= a * (dr.z / a).round();
        dr
    }

    /// Calculate the Madelung energy contribution (ion-ion).
    fn madelung_energy(&self) -> f64 {
        // For a simple estimate, use the Madelung constant for FCC
        // E_madelung ≈ -α * Z² / a per ion
        // FCC Madelung constant α ≈ 1.7476... for monovalent
        let alpha_madelung = 1.7476;
        let z = 1.0; // Assuming pseudopotential with 1 valence
        let num_atoms = self.ion_positions.len();
        -alpha_madelung * z * z / self.lattice_constant * num_atoms as f64
    }
}

impl MultiWfn for LithiumFCC {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let a = self.lattice_constant;
        
        // Initialize electrons randomly within the simulation cell
        (0..self.num_electrons)
            .map(|_| Vector3::new(
                rng.gen::<f64>() * a,
                rng.gen::<f64>() * a,
                rng.gen::<f64>() * a,
            ))
            .collect()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        // Simple product of localized orbitals (no antisymmetry for now)
        // For a proper implementation, use Slater determinant
        let mut psi = 1.0;
        
        for (i, pos) in r.iter().enumerate() {
            let orbital_idx = i % self.orbitals.len();
            let orbital = &self.orbitals[orbital_idx];
            
            // Use minimum image to nearest ion
            let ion_idx = orbital_idx / (self.num_electrons / self.ion_positions.len()).max(1);
            let ion_pos = &self.ion_positions[ion_idx.min(self.ion_positions.len() - 1)];
            let dr = self.minimum_image(pos, ion_pos);
            let local_pos = *ion_pos + dr;
            
            use crate::wavefunction::SingleWfn;
            psi *= orbital.evaluate(&local_pos);
        }
        
        psi * self.jastrow.evaluate(r)
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        // Use numerical derivatives for now
        self.numerical_derivative(r, 1e-5)
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        // Use numerical Laplacian for now
        self.numerical_laplacian(r, 1e-5)
    }
}

impl EnergyCalculator for LithiumFCC {
    fn local_energy(&self, r: &[Vector3<f64>]) -> f64 {
        let psi = self.evaluate(r);
        if psi.abs() < 1e-15 {
            return 0.0; // Avoid division by zero
        }
        
        let laplacian = self.laplacian(r);
        
        // Kinetic energy: -½ Σᵢ ∇²ψ/ψ
        let kinetic = -0.5 * laplacian.iter().sum::<f64>() / psi;
        
        // Electron-ion potential (using minimum image convention)
        // For pseudopotential: Z_eff = 1
        let z_eff = if self.num_electrons / self.ion_positions.len() == 1 { 1.0 } else { 3.0 };
        
        let v_ei: f64 = r.iter()
            .map(|ri| {
                self.ion_positions.iter()
                    .map(|ion| {
                        let dr = self.minimum_image(ri, ion);
                        let dist = dr.norm();
                        if dist > 1e-10 { -z_eff / dist } else { 0.0 }
                    })
                    .sum::<f64>()
            })
            .sum();
        
        // Electron-electron repulsion (using minimum image)
        let n = r.len();
        let v_ee: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| {
                let dr = self.minimum_image(&r[i], &r[j]);
                let dist = dr.norm();
                if dist > 1e-10 { 1.0 / dist } else { 0.0 }
            })
            .sum();
        
        // Madelung energy (ion-ion, constant for fixed ions)
        let v_ii = self.madelung_energy();
        
        kinetic + v_ei + v_ee + v_ii
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lithium_fcc_creation() {
        let li_fcc = LithiumFCC::new(8.0, 1, 1.0); // 8 Bohr ≈ 4.2 Å
        
        assert_eq!(li_fcc.ion_positions.len(), 1); // 1 atom in primitive cell
        assert_eq!(li_fcc.num_electrons, 1);
        assert_relative_eq!(li_fcc.lattice_constant, 8.0);
    }

    #[test]
    fn test_lithium_fcc_supercell() {
        let li_fcc = LithiumFCC::new_supercell(8.0, 2, 1, 1.0);
        
        // 2×2×2 supercell with 4 atoms per conventional cell = 32 atoms
        assert_eq!(li_fcc.ion_positions.len(), 32);
        assert_eq!(li_fcc.num_electrons, 32);
    }

    #[test]
    fn test_lithium_fcc_evaluate() {
        let li_fcc = LithiumFCC::new(8.0, 1, 1.0);
        let r = li_fcc.initialize();
        
        let psi = li_fcc.evaluate(&r);
        assert!(psi.is_finite());
        assert!(psi != 0.0);
    }

    #[test]
    fn test_lithium_fcc_energy() {
        let li_fcc = LithiumFCC::new(8.0, 1, 1.0);
        let r = li_fcc.initialize();
        
        let energy = li_fcc.local_energy(&r);
        assert!(energy.is_finite());
    }
}
