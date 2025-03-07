use nalgebra::{Matrix3, Vector3};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::dmc::{Walker, BranchingResult};
use crate::mcmc::EnergyCalculator;
use statrs::function::erf::erfc;


const MAX_CLONES: i32 = 3;

#[derive(Debug, Clone, Copy)]
pub struct LatticeVector {
    pub lattice_vector: Matrix3<f64>,
    pub reciprocal_vector: Matrix3<f64>
}

impl LatticeVector {
    pub fn new_bcc(a: f64) -> Self {
        // BCC lattice vectors
        let lattice = Matrix3::new(
            -a/2.0, a/2.0, a/2.0,
            a/2.0, -a/2.0, a/2.0,
            a/2.0, a/2.0, -a/2.0
        );

        // Calculate reciprocal lattice vectors
        let volume = lattice.determinant();
        let reciprocal = 2.0 * std::f64::consts::PI / volume *
            Matrix3::new(
                0.0, 1.0, 1.0,
                1.0, 0.0, 1.0,
                1.0, 1.0, 0.0
            );

        Self {
            lattice_vector: lattice,
            reciprocal_vector: reciprocal
        }
    }

    pub fn minimum_image_distance(&self, r1: &Vector3<f64>, r2: &Vector3<f64>) -> Vector3<f64> {
        let mut dr = r1 - r2;
        dr = self.lattice_vector * (dr.component_div(&self.lattice_vector.diagonal()).map(|x| x.round()));
        dr
    }
}

pub struct LithiumCrystalWalker {
    pub lattice: LatticeVector,
    pub ion_positions: Vec<Vector3<f64>>,
    pub electron_positions: Vec<Vector3<f64>>,
    pub dt: f64,
    pub sdt: f64,
    pub energy: f64,
    pub weight: f64,
    pub marked_for_deletion: bool,
}

impl LithiumCrystalWalker {
    fn calculate_ewald_sum(&self) -> f64 {
        // Parameters for Ewald summation
        let alpha = 1.0;  // Adjustable parameter for convergence speed
        let k_cutoff = 5; // Maximum k-vector components
        let r_cutoff = 3; // Maximum real-space lattice vectors

        let mut v_ewald = 0.0;
        let a = self.lattice.lattice_vector.diagonal().x;

        // Real-space sum
        for nx in -r_cutoff..=r_cutoff {
            for ny in -r_cutoff..=r_cutoff {
                for nz in -r_cutoff..=r_cutoff {
                    if nx == 0 && ny == 0 && nz == 0 {
                        continue;
                    }

                    let r = a * f64::sqrt((nx*nx + ny*ny + nz*nz) as f64);
                    v_ewald += 1.0/r * erfc(alpha * r);
                }
            }
        }

        // Reciprocal-space sum
        for nx in -k_cutoff..=k_cutoff {
            for ny in -k_cutoff..=k_cutoff {
                for nz in -k_cutoff..=k_cutoff {
                    if nx == 0 && ny == 0 && nz == 0 {
                        continue;
                    }

                    let k_vec = Vector3::new(nx as f64, ny as f64, nz as f64);
                    let k = 2.0 * std::f64::consts::PI / a * k_vec.norm();
                    v_ewald += 4.0 * std::f64::consts::PI / (a*a*a) *
                        f64::exp(-k*k/(4.0*alpha*alpha)) / k*k;
                }
            }
        }

        // Self-interaction correction
        v_ewald -= alpha / f64::sqrt(std::f64::consts::PI);

        v_ewald
    }
}

impl Walker for LithiumCrystalWalker {
    fn new(dt: f64, eref: f64) -> Self {
        let a = 3.51; // Lithium lattice constant in angstroms
        let lattice = LatticeVector::new_bcc(a);

        // BCC has atoms at (0,0,0) and (0.5,0.5,0.5)
        let ion_positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(a/2.0, a/2.0, a/2.0)
        ];

        // Initialize 6 electrons (3 from each Li atom)
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let electron_positions = (0..6)
            .map(|_| Vector3::<f64>::from_distribution(&dist, &mut rng))
            .collect();

        let mut walker = Self {
            lattice,
            ion_positions,
            electron_positions,
            dt,
            sdt: dt.sqrt(),
            energy: 0.0,
            weight: 1.0,
            marked_for_deletion: false,
        };

        walker.calculate_local_energy();
        walker.update_weight(eref);
        walker
    }

    fn move_walker(&mut self) {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, self.sdt).unwrap();

        // Move each electron
        for pos in self.electron_positions.iter_mut() {
            *pos += Vector3::<f64>::from_distribution(&dist, &mut rng);

            // Apply periodic boundary conditions
            *pos = Vector3::new(
                pos.x.rem_euclid(self.lattice.lattice_vector.diagonal().x),
                pos.y.rem_euclid(self.lattice.lattice_vector.diagonal().y),
                pos.z.rem_euclid(self.lattice.lattice_vector.diagonal().z)
            );
        }
    }

    fn calculate_local_energy(&mut self) {
        let mut energy = 0.0;

        // Electron-electron repulsion with minimum image convention
        for i in 0..self.electron_positions.len() {
            for j in (i+1)..self.electron_positions.len() {
                let r_ij = self.lattice.minimum_image_distance(
                    &self.electron_positions[i],
                    &self.electron_positions[j]
                );
                energy += 1.0 / r_ij.norm();
            }

            // Electron-ion attraction with minimum image convention
            for ion_pos in &self.ion_positions {
                let r_ei = self.lattice.minimum_image_distance(
                    &self.electron_positions[i],
                    ion_pos
                );
                energy -= 3.0 / r_ei.norm(); // Li nuclear charge is 3
            }
        }

        // Add ion-ion repulsion using Ewald summation
        energy += self.calculate_ewald_sum();

        self.energy = energy;
    }

    fn local_energy(&self) -> f64 {
        self.energy
    }

    fn update_weight(&mut self, e_ref: f64) {
        self.weight = ((-self.energy + e_ref) * self.dt).exp();
    }

    fn branching_decision(&mut self) -> BranchingResult {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>();
        let cnt = ((self.weight + r).floor() as i32).max(0).min(MAX_CLONES);
        match cnt {
            0 => BranchingResult::Kill,
            1 => BranchingResult::Keep,
            _ => BranchingResult::Clone { n: cnt as usize },
        }
    }

    fn should_be_deleted(&self) -> bool {
        self.marked_for_deletion
    }

    fn mark_for_deletion(&mut self) {
        self.marked_for_deletion = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bcc_lattice_creation() {
        let a = 3.51; // Lithium lattice constant
        let lattice = LatticeVector::new_bcc(a);

        // Check lattice vectors
        assert_relative_eq!(lattice.lattice_vector[(0, 0)], -a/2.0);
        assert_relative_eq!(lattice.lattice_vector[(0, 1)], a/2.0);
        assert_relative_eq!(lattice.lattice_vector[(0, 2)], a/2.0);

        // Check volume is correct for BCC
        let volume = lattice.lattice_vector.determinant().abs();
        let expected_volume = a*a*a / 2.0; // BCC unit cell volume = a³/2
        assert_relative_eq!(volume, expected_volume, epsilon = 1e-10);

        // Check reciprocal vectors
        let det = lattice.lattice_vector.determinant();
        assert_relative_eq!(
            lattice.reciprocal_vector[(0, 0)],
            2.0 * std::f64::consts::PI / det * 0.0
        );
    }

    #[test]
    fn test_minimum_image_distance() {
        let a = 3.51;
        let lattice = LatticeVector::new_bcc(a);

        // Test points inside the same cell
        let r1 = Vector3::new(0.1, 0.2, 0.3);
        let r2 = Vector3::new(0.4, 0.5, 0.6);
        let dr = lattice.minimum_image_distance(&r1, &r2);
        assert_relative_eq!(dr.x, r1.x - r2.x, epsilon = 1e-10);
        assert_relative_eq!(dr.y, r1.y - r2.y, epsilon = 1e-10);
        assert_relative_eq!(dr.z, r1.z - r2.z, epsilon = 1e-10);

        // Test points across periodic boundary
        let r3 = Vector3::new(0.1, 0.2, 0.3);
        let r4 = Vector3::new(a + 0.1, 0.2, 0.3); // Same point, but shifted by one lattice vector
        let dr2 = lattice.minimum_image_distance(&r3, &r4);
        // The minimum image should be much smaller than a
        assert!(dr2.norm() < a/2.0);
    }

    #[test]
    fn test_lithium_crystal_walker_creation() {
        let dt = 0.01;
        let eref = 0.0;
        let walker = LithiumCrystalWalker::new(dt, eref);

        // Check initialization
        assert_eq!(walker.ion_positions.len(), 2); // BCC has 2 atoms per unit cell
        assert_eq!(walker.electron_positions.len(), 6); // 6 electrons (3 from each Li)
        assert_relative_eq!(walker.dt, dt);
        assert_relative_eq!(walker.sdt, dt.sqrt());
        assert!(!walker.marked_for_deletion);
    }

    #[test]
    fn test_ewald_summation() {
        let dt = 0.01;
        let eref = 0.0;
        let walker = LithiumCrystalWalker::new(dt, eref);

        // Ewald sum for a simple case should be finite and non-zero
        let ewald_energy = walker.calculate_ewald_sum();
        assert!(ewald_energy.is_finite());
        assert!(ewald_energy != 0.0);

        // The energy should be positive for repulsive ion-ion interactions
        assert!(ewald_energy > 0.0);
    }

    #[test]
    fn test_walker_movement() {
        let dt = 0.01;
        let eref = 0.0;
        let mut walker = LithiumCrystalWalker::new(dt, eref);

        // Save initial positions
        let initial_positions = walker.electron_positions.clone();

        // Move walker
        walker.move_walker();

        // Positions should change
        let mut changed = false;
        for (i, pos) in walker.electron_positions.iter().enumerate() {
            if (pos - initial_positions[i]).norm() > 1e-10 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Electron positions did not change after move_walker()");

        // All positions should be within the unit cell
        let a = walker.lattice.lattice_vector.diagonal().x;
        for pos in &walker.electron_positions {
            assert!(pos.x >= 0.0 && pos.x < a);
            assert!(pos.y >= 0.0 && pos.y < a);
            assert!(pos.z >= 0.0 && pos.z < a);
        }
    }

    #[test]
    fn test_local_energy_calculation() {
        let dt = 0.01;
        let eref = 0.0;
        let mut walker = LithiumCrystalWalker::new(dt, eref);

        // Energy should be calculated during creation
        let initial_energy = walker.energy;
        assert!(initial_energy.is_finite());

        // Move walker and recalculate energy
        walker.move_walker();
        walker.calculate_local_energy();

        // Energy should change
        assert_ne!(initial_energy, walker.energy);
    }
}