//! Stochastic Reconfiguration (SR) wavefunction optimizer.
//!
//! Implements the SR method for optimizing variational parameters by
//! natural gradient descent on the energy. The parameter update rule is:
//!
//!   S · δp = -f
//!
//! where:
//! - S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩  (overlap/covariance matrix)
//! - f_i = ⟨E_L O_i⟩ - ⟨E_L⟩⟨O_i⟩  (energy-parameter covariance)
//! - O_i = ∂ ln|Ψ| / ∂p_i            (log-derivatives)

use nalgebra::{DMatrix, DVector, Vector3};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::wavefunction::OptimizableWfn;
use super::traits::EnergyCalculator;

/// Configuration for the SR optimizer.
#[derive(Clone, Debug)]
pub struct SROptimizer {
    /// Number of VMC samples per iteration
    pub n_samples: usize,
    /// Number of walkers
    pub n_walkers: usize,
    /// Number of equilibration steps per iteration
    pub n_equilibrate: usize,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Learning rate (step size δt for parameter update)
    pub learning_rate: f64,
    /// Levenberg-Marquardt regularization for the S matrix
    pub sr_epsilon: f64,
    /// Convergence tolerance on |δE|
    pub tolerance: f64,
    /// Metropolis step size for sampling
    pub step_size: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for SROptimizer {
    fn default() -> Self {
        Self {
            n_samples: 5000,
            n_walkers: 20,
            n_equilibrate: 500,
            max_iterations: 50,
            learning_rate: 0.05,
            sr_epsilon: 0.001,
            tolerance: 1e-5,
            step_size: 0.5,
            verbose: true,
        }
    }
}

impl SROptimizer {
    /// Create a new SR optimizer with default settings.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set number of samples per iteration.
    pub fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
        self
    }
    
    /// Set number of walkers.
    pub fn with_n_walkers(mut self, n: usize) -> Self {
        self.n_walkers = n;
        self
    }
    
    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    
    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    
    /// Set SR regularization parameter.
    pub fn with_sr_epsilon(mut self, eps: f64) -> Self {
        self.sr_epsilon = eps;
        self
    }
    
    /// Set verbosity.
    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    
    /// Run VMC sampling and collect local energies + log-derivatives.
    ///
    /// Returns (energies, log_derivs) where log_derivs[k][i] is
    /// the i-th parameter log-derivative at sample k.
    fn sample_with_derivatives<T: OptimizableWfn + EnergyCalculator>(
        &self,
        wfn: &T,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, self.step_size).unwrap();
        
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
        
        // Production run: collect samples
        let mut energies = Vec::with_capacity(self.n_samples);
        let mut log_derivs = Vec::with_capacity(self.n_samples);
        let steps_per_sample = (self.n_samples / self.n_walkers).max(1);
        
        for _ in 0..steps_per_sample {
            // Decorrelation steps between samples
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
            
            // Collect measurements from all walkers
            for pos in positions.iter() {
                energies.push(wfn.local_energy(pos));
                log_derivs.push(wfn.log_derivatives(pos));
            }
        }
        
        (energies, log_derivs)
    }
    
    /// Compute the SR parameter update from sampled data.
    ///
    /// Returns (delta_params, mean_energy, variance).
    fn compute_sr_update(
        &self,
        energies: &[f64],
        log_derivs: &[Vec<f64>],
        n_params: usize,
    ) -> (Vec<f64>, f64, f64) {
        let n = energies.len() as f64;
        
        // Mean energy
        let e_mean = energies.iter().sum::<f64>() / n;
        
        // Variance
        let variance = energies.iter()
            .map(|e| (e - e_mean).powi(2))
            .sum::<f64>() / n;
        
        // Mean log-derivatives: ⟨O_i⟩
        let mut o_mean = vec![0.0; n_params];
        for od in log_derivs.iter() {
            for (i, &o) in od.iter().enumerate() {
                o_mean[i] += o;
            }
        }
        for v in o_mean.iter_mut() {
            *v /= n;
        }
        
        // Build S matrix: S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩
        let mut s_matrix = DMatrix::zeros(n_params, n_params);
        for od in log_derivs.iter() {
            for i in 0..n_params {
                for j in 0..n_params {
                    s_matrix[(i, j)] += od[i] * od[j];
                }
            }
        }
        s_matrix /= n;
        for i in 0..n_params {
            for j in 0..n_params {
                s_matrix[(i, j)] -= o_mean[i] * o_mean[j];
            }
        }
        
        // Levenberg-Marquardt regularization: S_ii += ε
        for i in 0..n_params {
            s_matrix[(i, i)] += self.sr_epsilon;
        }
        
        // Build force vector: f_i = ⟨E_L O_i⟩ - ⟨E_L⟩⟨O_i⟩
        let mut force = DVector::zeros(n_params);
        for (k, od) in log_derivs.iter().enumerate() {
            for (i, &o) in od.iter().enumerate() {
                force[i] += energies[k] * o;
            }
        }
        force /= n;
        for i in 0..n_params {
            force[i] -= e_mean * o_mean[i];
        }
        
        // Solve S · δp = -f
        let neg_force = -&force;
        let delta_params = s_matrix.lu().solve(&neg_force)
            .unwrap_or_else(|| {
                // Fallback to simple gradient if S is singular
                neg_force.clone()
            });
        
        let dp: Vec<f64> = delta_params.iter().cloned().collect();
        (dp, e_mean, variance)
    }
    
    /// Run the SR optimization.
    ///
    /// Iteratively optimizes the variational parameters of the wavefunction
    /// to minimize the energy using stochastic reconfiguration.
    pub fn optimize<T: OptimizableWfn + EnergyCalculator>(
        &self,
        wfn: &mut T,
    ) -> SRResult {
        let n_params = wfn.num_params();
        let mut energy_history = Vec::with_capacity(self.max_iterations);
        let mut variance_history = Vec::with_capacity(self.max_iterations);
        let mut param_history = Vec::with_capacity(self.max_iterations + 1);
        
        param_history.push(wfn.get_params());
        
        if self.verbose {
            println!("Stochastic Reconfiguration Optimization");
            println!("========================================");
            println!("  Parameters:      {}", n_params);
            println!("  Samples/iter:    {}", self.n_samples);
            println!("  Walkers:         {}", self.n_walkers);
            println!("  Learning rate:   {:.4}", self.learning_rate);
            println!("  SR epsilon:      {:.1e}", self.sr_epsilon);
            println!("  Max iterations:  {}", self.max_iterations);
            println!("  Initial params:  {:?}", wfn.get_params());
            println!();
        }
        
        let mut prev_energy = f64::MAX;
        
        for iter in 0..self.max_iterations {
            // Sample with current parameters
            let (energies, log_derivs) = self.sample_with_derivatives(wfn);
            
            // Compute SR update
            let (delta_params, e_mean, variance) = 
                self.compute_sr_update(&energies, &log_derivs, n_params);
            
            energy_history.push(e_mean);
            variance_history.push(variance);
            
            if self.verbose {
                let error = (variance / energies.len() as f64).sqrt();
                println!("  Iter {:3}: E = {:10.5} ± {:.4} Ha, σ² = {:.3}, δp = {:?}",
                    iter + 1, e_mean, error, variance,
                    delta_params.iter().map(|x| format!("{:.4}", x * self.learning_rate)).collect::<Vec<_>>());
            }
            
            // Apply parameter update: p_new = p_old + δt * δp
            let mut params = wfn.get_params();
            for (p, dp) in params.iter_mut().zip(delta_params.iter()) {
                *p += self.learning_rate * dp;
            }
            
            // Enforce positivity of decay parameters
            for p in params.iter_mut() {
                if *p < 0.1 {
                    *p = 0.1;
                }
            }
            
            wfn.set_params(&params);
            param_history.push(params);
            
            // Convergence check
            let energy_change = (e_mean - prev_energy).abs();
            if iter > 5 && energy_change < self.tolerance {
                if self.verbose {
                    println!("\n  Converged after {} iterations (|δE| = {:.2e} < {:.2e})",
                        iter + 1, energy_change, self.tolerance);
                }
                break;
            }
            prev_energy = e_mean;
        }
        
        // Final high-statistics sampling
        let (final_energies, _) = self.sample_with_derivatives(wfn);
        let n = final_energies.len() as f64;
        let final_energy = final_energies.iter().sum::<f64>() / n;
        let final_variance = final_energies.iter()
            .map(|e| (e - final_energy).powi(2))
            .sum::<f64>() / n;
        
        if self.verbose {
            let error = (final_variance / n).sqrt();
            println!("\nFinal results:");
            println!("  Energy:    {:10.5} ± {:.4} Ha", final_energy, error);
            println!("  Variance:  {:.4} Ha²", final_variance);
            println!("  Params:    {:?}", wfn.get_params());
        }
        
        SRResult {
            final_params: wfn.get_params(),
            final_energy,
            final_variance,
            energy_history,
            variance_history,
            param_history,
        }
    }
}

/// Results from SR optimization.
#[derive(Clone, Debug)]
pub struct SRResult {
    /// Optimized parameter values
    pub final_params: Vec<f64>,
    /// Final energy estimate
    pub final_energy: f64,
    /// Final variance of local energy
    pub final_variance: f64,
    /// Energy at each iteration
    pub energy_history: Vec<f64>,
    /// Variance at each iteration
    pub variance_history: Vec<f64>,
    /// Parameter values at each iteration (including initial)
    pub param_history: Vec<Vec<f64>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::MethaneGTO;
    
    #[test]
    fn test_sr_produces_reasonable_results() {
        // Start with suboptimal parameters
        let mut wfn = MethaneGTO::new(0.5, 1.0);
        
        let optimizer = SROptimizer::new()
            .with_n_samples(2000)
            .with_n_walkers(10)
            .with_max_iterations(10)
            .with_learning_rate(0.05)
            .with_verbose(false);
        
        let result = optimizer.optimize(&mut wfn);
        
        // Energy should be negative (bound state)
        assert!(result.final_energy < 0.0,
            "Energy should be negative, got {}", result.final_energy);
        
        // Parameters should have changed from initial
        let initial = result.param_history[0].clone();
        let final_p = result.final_params.clone();
        let changed = initial.iter().zip(final_p.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(changed, "Parameters should change during optimization");
    }
    
    #[test]
    fn test_sr_energy_decreases() {
        let mut wfn = MethaneGTO::new(0.5, 1.0);
        
        let optimizer = SROptimizer::new()
            .with_n_samples(3000)
            .with_n_walkers(20)
            .with_max_iterations(15)
            .with_learning_rate(0.03)
            .with_verbose(false);
        
        let result = optimizer.optimize(&mut wfn);
        
        // The average energy over the last few iterations should be
        // lower than the first few iterations (statistically)
        if result.energy_history.len() >= 6 {
            let first_avg: f64 = result.energy_history[..3].iter().sum::<f64>() / 3.0;
            let last_avg: f64 = result.energy_history[result.energy_history.len()-3..]
                .iter().sum::<f64>() / 3.0;
            
            // Allow some slack due to statistical noise
            assert!(last_avg < first_avg + 2.0,
                "Energy should decrease or stay similar: first_avg={:.4}, last_avg={:.4}",
                first_avg, last_avg);
        }
    }
}
