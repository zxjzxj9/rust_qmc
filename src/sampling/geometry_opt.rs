//! Geometry optimization via VMC-estimated Hellmann-Feynman forces.
//!
//! Uses steepest descent with VMC-averaged forces to relax nuclear positions
//! toward the equilibrium geometry on the Born-Oppenheimer surface.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::wavefunction::MultiWfn;
use super::traits::{EnergyCalculator, ForceCalculator};

/// Configuration for the geometry optimizer.
#[derive(Clone, Debug)]
pub struct GeometryOptimizer {
    /// Number of VMC samples per force evaluation
    pub n_samples: usize,
    /// Number of walkers for VMC sampling
    pub n_walkers: usize,
    /// Number of equilibration sweeps before measurement
    pub n_equilibrate: usize,
    /// Steepest-descent step size α (Bohr): R_I += α × F_I
    pub step_size: f64,
    /// Maximum number of optimization steps
    pub max_iterations: usize,
    /// Convergence threshold: stop when max|F_I| < force_threshold (Ha/Bohr)
    pub force_threshold: f64,
    /// Metropolis step size for VMC sampling
    pub mc_step_size: f64,
    /// Whether to print progress
    pub verbose: bool,
}

impl Default for GeometryOptimizer {
    fn default() -> Self {
        Self {
            n_samples: 5000,
            n_walkers: 20,
            n_equilibrate: 500,
            step_size: 0.005,
            max_iterations: 50,
            force_threshold: 0.05,
            mc_step_size: 0.5,
            verbose: true,
        }
    }
}

impl GeometryOptimizer {
    /// Create a new geometry optimizer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }

    pub fn with_n_walkers(mut self, n: usize) -> Self {
        self.n_walkers = n;
        self
    }

    pub fn with_step_size(mut self, s: f64) -> Self {
        self.step_size = s;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_force_threshold(mut self, f: f64) -> Self {
        self.force_threshold = f;
        self
    }

    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Run VMC sampling and compute mean forces and energy.
    ///
    /// Returns (mean_forces, mean_energy, energy_error).
    fn sample_forces<T: MultiWfn + ForceCalculator>(
        &self,
        wfn: &T,
    ) -> (Vec<Vector3<f64>>, f64, f64) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.mc_step_size).unwrap();
        let n_nuc = wfn.num_nuclei();

        // Initialize walkers
        let mut positions: Vec<Vec<Vector3<f64>>> = (0..self.n_walkers)
            .map(|_| wfn.initialize())
            .collect();
        let mut psi_values: Vec<f64> = positions.iter()
            .map(|r| wfn.evaluate(r))
            .collect();

        // Equilibration
        for _ in 0..self.n_equilibrate {
            for (walker_idx, pos) in positions.iter_mut().enumerate() {
                let new_pos: Vec<Vector3<f64>> = pos.iter()
                    .map(|p| p + Vector3::new(
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                    ))
                    .collect();
                let new_psi = wfn.evaluate(&new_pos);
                let ratio = (new_psi / psi_values[walker_idx]).powi(2);
                if rng.gen::<f64>() < ratio {
                    *pos = new_pos;
                    psi_values[walker_idx] = new_psi;
                }
            }
        }

        // Production: collect force and energy samples
        let mut force_accum = vec![Vector3::zeros(); n_nuc];
        let mut energy_accum = 0.0;
        let mut energy_sq_accum = 0.0;
        let mut n_collected = 0usize;
        let steps_per_sample = (self.n_samples / self.n_walkers).max(1);

        for _ in 0..steps_per_sample {
            // Decorrelation steps
            for _ in 0..5 {
                for (walker_idx, pos) in positions.iter_mut().enumerate() {
                    let new_pos: Vec<Vector3<f64>> = pos.iter()
                        .map(|p| p + Vector3::new(
                            normal.sample(&mut rng),
                            normal.sample(&mut rng),
                            normal.sample(&mut rng),
                        ))
                        .collect();
                    let new_psi = wfn.evaluate(&new_pos);
                    let ratio = (new_psi / psi_values[walker_idx]).powi(2);
                    if rng.gen::<f64>() < ratio {
                        *pos = new_pos;
                        psi_values[walker_idx] = new_psi;
                    }
                }
            }

            // Collect measurements
            for pos in positions.iter() {
                let forces = wfn.hellmann_feynman_force(pos);
                let energy = wfn.local_energy(pos);
                for (acc, f) in force_accum.iter_mut().zip(forces.iter()) {
                    *acc += f;
                }
                energy_accum += energy;
                energy_sq_accum += energy * energy;
                n_collected += 1;
            }
        }

        let n = n_collected as f64;
        let mean_forces: Vec<Vector3<f64>> = force_accum.iter()
            .map(|f| f / n)
            .collect();
        let mean_energy = energy_accum / n;
        let variance = energy_sq_accum / n - mean_energy * mean_energy;
        let error = (variance / n).sqrt();

        (mean_forces, mean_energy, error)
    }

    /// Run the geometry optimization.
    ///
    /// Iteratively moves nuclei along VMC-estimated Hellmann-Feynman forces
    /// using steepest descent until convergence.
    pub fn optimize<T: MultiWfn + ForceCalculator>(
        &self,
        wfn: &mut T,
    ) -> GeometryOptResult {
        let n_nuc = wfn.num_nuclei();
        let mut energy_history = Vec::with_capacity(self.max_iterations);
        let mut force_history = Vec::with_capacity(self.max_iterations);
        let mut geometry_history = Vec::with_capacity(self.max_iterations + 1);

        geometry_history.push(wfn.get_nuclei());

        if self.verbose {
            println!("Geometry Optimization (VMC Hellmann-Feynman Forces)");
            println!("===================================================");
            println!("  Nuclei:           {}", n_nuc);
            println!("  Samples/iter:     {}", self.n_samples);
            println!("  Walkers:          {}", self.n_walkers);
            println!("  Step size α:      {:.4} Bohr", self.step_size);
            println!("  Force threshold:  {:.4} Ha/Bohr", self.force_threshold);
            println!("  Max iterations:   {}", self.max_iterations);
            println!();
        }

        for iter in 0..self.max_iterations {
            // Sample forces and energy at current geometry
            let (mean_forces, mean_energy, error) = self.sample_forces(wfn);

            // Compute max force magnitude
            let max_force = mean_forces.iter()
                .map(|f| f.norm())
                .fold(0.0_f64, f64::max);

            energy_history.push(mean_energy);
            force_history.push(max_force);

            if self.verbose {
                println!("  Step {:3}: E = {:10.5} ± {:.4} Ha, max|F| = {:.4} Ha/Bohr",
                    iter + 1, mean_energy, error, max_force);
                if self.verbose && iter < 5 {
                    // Print individual forces for first few steps
                    for (i, f) in mean_forces.iter().enumerate() {
                        println!("           F[{}] = ({:+.4}, {:+.4}, {:+.4}) |F|={:.4}",
                            i, f.x, f.y, f.z, f.norm());
                    }
                }
            }

            // Check convergence
            if max_force < self.force_threshold {
                if self.verbose {
                    println!("\n  Converged after {} steps (max|F| = {:.4} < {:.4} Ha/Bohr)",
                        iter + 1, max_force, self.force_threshold);
                }
                break;
            }

            // Steepest descent: R_I += α × F_I
            let mut nuclei = wfn.get_nuclei();
            let charges = wfn.get_charges();
            
            // Project out center-of-mass force to prevent rigid translation
            let total_mass: f64 = charges.iter().sum();
            let com_force: Vector3<f64> = mean_forces.iter()
                .zip(charges.iter())
                .map(|(f, &m)| f * m)
                .fold(Vector3::zeros(), |acc, f| acc + f)
                / total_mass;
            
            for (i, f) in mean_forces.iter().enumerate() {
                // Remove COM component
                let projected_force = f - com_force;
                // Compute displacement
                let disp = self.step_size * projected_force;
                let disp_norm = disp.norm();
                let max_disp = 0.02; // conservative: max 0.02 Bohr per step
                if disp_norm > max_disp {
                    nuclei[i] += disp * (max_disp / disp_norm);
                } else {
                    nuclei[i] += disp;
                }
            }
            wfn.set_nuclei(&nuclei);
            geometry_history.push(nuclei);
        }

        // Final energy evaluation
        let (_, final_energy, _) = self.sample_forces(wfn);
        let final_max_force = force_history.last().copied().unwrap_or(0.0);

        if self.verbose {
            println!("\nFinal Results:");
            println!("  Energy:     {:10.5} Ha", final_energy);
            println!("  Max |F|:    {:.4} Ha/Bohr", final_max_force);
            let nuclei = wfn.get_nuclei();
            let charges = wfn.get_charges();
            println!("  Geometry:");
            for (i, (pos, z)) in nuclei.iter().zip(charges.iter()).enumerate() {
                let label = if *z > 5.0 { "C" } else { "H" };
                println!("    {}{}  ({:+.4}, {:+.4}, {:+.4})",
                    label, i, pos.x, pos.y, pos.z);
            }
        }

        GeometryOptResult {
            final_nuclei: wfn.get_nuclei(),
            final_energy,
            final_max_force,
            energy_history,
            force_history,
            geometry_history,
        }
    }
}

/// Results from geometry optimization.
#[derive(Clone, Debug)]
pub struct GeometryOptResult {
    /// Final nuclear positions
    pub final_nuclei: Vec<Vector3<f64>>,
    /// Final VMC energy
    pub final_energy: f64,
    /// Final maximum force magnitude
    pub final_max_force: f64,
    /// Energy at each optimization step
    pub energy_history: Vec<f64>,
    /// Max |F| at each optimization step
    pub force_history: Vec<f64>,
    /// Nuclear positions at each step (including initial)
    pub geometry_history: Vec<Vec<Vector3<f64>>>,
}
