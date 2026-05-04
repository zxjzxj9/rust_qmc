//! Geometry optimization via VMC-estimated forces.
//!
//! Uses steepest descent with VMC-averaged forces to relax nuclear positions
//! toward the equilibrium geometry on the Born-Oppenheimer surface.
//!
//! Supports both bare Hellmann-Feynman and zero-variance (Assaraf-Caffarel)
//! force estimators for reduced statistical noise.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::wavefunction::MultiWfn;
use super::traits::{EnergyCalculator, ForceCalculator};
use super::force_variance::ForceEstimator;

/// Configuration for the geometry optimizer.
#[derive(Clone, Debug)]
pub struct GeometryOptimizer {
    /// Number of VMC samples per force evaluation
    pub n_samples: usize,
    /// Number of walkers for VMC sampling
    pub n_walkers: usize,
    /// Number of equilibration sweeps before measurement
    pub n_equilibrate: usize,
    /// Steepest-descent step size alpha (Bohr): R_I += alpha x F_I
    pub step_size: f64,
    /// Maximum number of optimization steps
    pub max_iterations: usize,
    /// Convergence threshold: stop when max|F_I| < force_threshold (Ha/Bohr)
    pub force_threshold: f64,
    /// Metropolis step size for VMC sampling
    pub mc_step_size: f64,
    /// Which force estimator to use
    pub force_estimator: ForceEstimator,
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
            force_estimator: ForceEstimator::Bare,
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

    pub fn with_force_estimator(mut self, e: ForceEstimator) -> Self {
        self.force_estimator = e;
        self
    }

    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Run VMC sampling and compute mean forces and energy.
    ///
    /// Returns (mean_forces, mean_energy, energy_error, force_variance).
    ///
    /// When using the ZV estimator, the force is:
    ///   F_I^ZV = F_I^HF - 2(E_L - E_ref) x ∂ln|Ψ_T|/∂R_I
    ///
    /// The Pulay correction is accumulated as a covariance and applied
    /// after all samples are collected.
    fn sample_forces<T: MultiWfn + ForceCalculator>(
        &self,
        wfn: &T,
    ) -> (Vec<Vector3<f64>>, f64, f64, Vec<f64>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.mc_step_size).unwrap();
        let n_nuc = wfn.num_nuclei();
        let use_zv = self.force_estimator == ForceEstimator::ZeroVariance;

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
        let mut force_accum = vec![Vector3::zeros(); n_nuc];   // Σ F_HF
        let mut force_sq_accum = vec![0.0_f64; n_nuc];          // Σ |F_HF|²
        let mut energy_accum = 0.0;
        let mut energy_sq_accum = 0.0;
        let mut n_collected = 0usize;
        let steps_per_sample = (self.n_samples / self.n_walkers).max(1);

        // ZV-specific accumulators: Σ E_LxΩ_I and Σ Ω_I
        let mut el_omega_accum = vec![Vector3::zeros(); n_nuc]; // Σ E_L x Ω_I
        let mut omega_accum = vec![Vector3::zeros(); n_nuc];    // Σ Ω_I

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
                let forces_hf = wfn.hellmann_feynman_force(pos);
                let energy = wfn.local_energy(pos);

                for (i, f) in forces_hf.iter().enumerate() {
                    force_accum[i] += f;
                    force_sq_accum[i] += f.norm_squared();
                }
                energy_accum += energy;
                energy_sq_accum += energy * energy;

                // ZV: accumulate Ω_I and E_L x Ω_I
                if use_zv {
                    let omega = wfn.wfn_nuclear_gradient(pos);
                    for (i, o) in omega.iter().enumerate() {
                        el_omega_accum[i] += energy * o;
                        omega_accum[i] += *o;
                    }
                }

                n_collected += 1;
            }
        }

        let n = n_collected as f64;
        let mean_energy = energy_accum / n;
        let e_variance = energy_sq_accum / n - mean_energy * mean_energy;
        let error = (e_variance / n).sqrt();

        // Compute mean forces
        let mean_forces: Vec<Vector3<f64>> = if use_zv {
            // ZV force = <F_HF> - 2 x Cov(E_L, Ω)
            //          = <F_HF> - 2 x (<E_L x Ω> - <E_L> x <Ω>)
            (0..n_nuc).map(|i| {
                let mean_f_hf = force_accum[i] / n;
                let mean_el_omega = el_omega_accum[i] / n;
                let mean_omega = omega_accum[i] / n;
                let cov = mean_el_omega - mean_energy * mean_omega;
                mean_f_hf - 2.0 * cov
            }).collect()
        } else {
            force_accum.iter().map(|f| f / n).collect()
        };

        // Force variance per nucleus (of whichever estimator we used)
        // For bare: var(|F_HF|) per nucleus
        let force_variance: Vec<f64> = (0..n_nuc).map(|i| {
            let mean_sq = force_sq_accum[i] / n;
            let sq_mean = (force_accum[i] / n).norm_squared();
            (mean_sq - sq_mean).max(0.0)
        }).collect();

        (mean_forces, mean_energy, error, force_variance)
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
            println!("Geometry Optimization (VMC Forces)");
            println!("===================================================");
            println!("  Nuclei:           {}", n_nuc);
            println!("  Samples/iter:     {}", self.n_samples);
            println!("  Walkers:          {}", self.n_walkers);
            println!("  Step size alpha:      {:.4} Bohr", self.step_size);
            println!("  Force threshold:  {:.4} Ha/Bohr", self.force_threshold);
            println!("  Force estimator:  {}", self.force_estimator);
            println!("  Max iterations:   {}", self.max_iterations);
            println!();
        }

        for iter in 0..self.max_iterations {
            // Sample forces and energy at current geometry
            let (mean_forces, mean_energy, error, force_var) = self.sample_forces(wfn);

            // Compute max force magnitude
            let max_force = mean_forces.iter()
                .map(|f| f.norm())
                .fold(0.0_f64, f64::max);

            energy_history.push(mean_energy);
            force_history.push(max_force);

            if self.verbose {
                println!("  Step {:3}: E = {:10.5} +/- {:.4} Ha, max|F| = {:.4} Ha/Bohr",
                    iter + 1, mean_energy, error, max_force);
                if self.verbose && iter < 5 {
                    // Print individual forces and variance for first few steps
                    for (i, f) in mean_forces.iter().enumerate() {
                        println!("           F[{}] = ({:+.4}, {:+.4}, {:+.4}) |F|={:.4}  σ²={:.4}",
                            i, f.x, f.y, f.z, f.norm(), force_var[i]);
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

            // Steepest descent: R_I += alpha x F_I
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
        let (_, final_energy, _, _) = self.sample_forces(wfn);
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
            force_estimator: self.force_estimator,
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
    /// Which force estimator was used
    pub force_estimator: ForceEstimator,
}
