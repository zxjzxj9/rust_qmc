use nalgebra::{Matrix3, Vector3};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::dmc::{Walker, BranchingResult};
use crate::mcmc::EnergyCalculator;

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
                    v_ewald += 1.0/r * f64::erfc(alpha * r);
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