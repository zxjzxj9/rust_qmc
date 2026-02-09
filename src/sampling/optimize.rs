//! Jastrow parameter optimization using variance minimization.
//!
//! This module provides tools for optimizing Jastrow correlation factor
//! parameters to minimize the variance of the local energy, which improves
//! the quality of VMC trial wavefunctions.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::wavefunction::MultiWfn;
use crate::systems::{MethaneGTO, JastrowParams};
use super::traits::EnergyCalculator;

/// Results from a single VMC sampling run.
#[derive(Clone, Debug)]
pub struct SamplingStats {
    /// Mean energy
    pub energy: f64,
    /// Variance of local energy
    pub variance: f64,
    /// Standard error of the mean
    pub error: f64,
    /// Number of samples
    pub n_samples: usize,
}

/// Results from Jastrow optimization.
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    /// Optimized parameters
    pub params: JastrowParams,
    /// Final energy
    pub final_energy: f64,
    /// Final variance
    pub final_variance: f64,
    /// History of energies during optimization
    pub energy_history: Vec<f64>,
    /// History of variances during optimization
    pub variance_history: Vec<f64>,
}

/// Jastrow parameter optimizer using variance minimization.
///
/// Uses gradient descent with finite differences to minimize σ²(E_L).
pub struct JastrowOptimizer {
    /// Samples per optimization step
    pub n_samples: usize,
    /// Number of walkers for sampling
    pub n_walkers: usize,
    /// Number of equilibration steps
    pub n_equilibrate: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Finite difference step for gradient estimation
    pub fd_step: f64,
    /// Convergence tolerance for variance change
    pub tolerance: f64,
    /// Verbose output
    pub verbose: bool,
}

impl Default for JastrowOptimizer {
    fn default() -> Self {
        Self {
            n_samples: 5000,
            n_walkers: 10,
            n_equilibrate: 500,
            learning_rate: 0.1,
            max_iterations: 50,
            fd_step: 0.05,
            tolerance: 1e-4,
            verbose: true,
        }
    }
}

impl JastrowOptimizer {
    /// Create a new optimizer with custom settings.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set number of samples per optimization step.
    pub fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = n;
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
    
    /// Set verbosity.
    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    
    /// Run VMC sampling and compute energy statistics.
    pub fn sample(&self, wfn: &MethaneGTO) -> SamplingStats {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.5).unwrap();
        
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
        
        // Production run
        let mut energies = Vec::with_capacity(self.n_samples);
        let steps_per_sample = (self.n_samples / self.n_walkers).max(1);
        
        for _ in 0..steps_per_sample {
            for (walker_idx, pos) in positions.iter_mut().enumerate() {
                // Metropolis move
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
                
                // Sample energy
                energies.push(wfn.local_energy(pos));
            }
        }
        
        // Compute statistics
        let n = energies.len() as f64;
        let energy = energies.iter().sum::<f64>() / n;
        let variance = energies.iter()
            .map(|e| (e - energy).powi(2))
            .sum::<f64>() / n;
        let error = (variance / n).sqrt();
        
        SamplingStats {
            energy,
            variance,
            error,
            n_samples: energies.len(),
        }
    }
    
    /// Compute gradient of variance with respect to Jastrow parameters.
    fn compute_gradient(&self, wfn: &MethaneGTO, current_variance: f64) -> (f64, f64) {
        let params = wfn.get_jastrow_params();
        
        // Gradient w.r.t. b_ee
        let params_plus = JastrowParams { b_ee: params.b_ee + self.fd_step, b_en: params.b_en };
        let params_minus = JastrowParams { b_ee: params.b_ee - self.fd_step, b_en: params.b_en };
        let var_plus = self.sample(&wfn.with_jastrow_params(params_plus)).variance;
        let var_minus = self.sample(&wfn.with_jastrow_params(params_minus)).variance;
        let grad_b_ee = (var_plus - var_minus) / (2.0 * self.fd_step);
        
        // Gradient w.r.t. b_en
        let params_plus = JastrowParams { b_ee: params.b_ee, b_en: params.b_en + self.fd_step };
        let params_minus = JastrowParams { b_ee: params.b_ee, b_en: params.b_en - self.fd_step };
        let var_plus = self.sample(&wfn.with_jastrow_params(params_plus)).variance;
        let var_minus = self.sample(&wfn.with_jastrow_params(params_minus)).variance;
        let grad_b_en = (var_plus - var_minus) / (2.0 * self.fd_step);
        
        (grad_b_ee, grad_b_en)
    }
    
    /// Optimize Jastrow parameters for the given wavefunction.
    pub fn optimize(&self, wfn: &MethaneGTO) -> OptimizationResult {
        let mut current_wfn = wfn.clone();
        let mut energy_history = Vec::new();
        let mut variance_history = Vec::new();
        
        if self.verbose {
            println!("Starting Jastrow optimization...");
            println!("Initial parameters: b_ee={:.4}, b_en={:.4}",
                wfn.get_jastrow_params().b_ee, wfn.get_jastrow_params().b_en);
        }
        
        let mut prev_variance = f64::MAX;
        
        for iter in 0..self.max_iterations {
            // Sample current wavefunction
            let stats = self.sample(&current_wfn);
            energy_history.push(stats.energy);
            variance_history.push(stats.variance);
            
            if self.verbose {
                let params = current_wfn.get_jastrow_params();
                println!("Iter {:3}: E = {:8.4} ± {:.4}, σ² = {:8.4}, b_ee = {:.4}, b_en = {:.4}",
                    iter, stats.energy, stats.error, stats.variance, params.b_ee, params.b_en);
            }
            
            // Check convergence
            let variance_change = (prev_variance - stats.variance).abs() / prev_variance.abs();
            if variance_change < self.tolerance && iter > 5 {
                if self.verbose {
                    println!("Converged after {} iterations", iter + 1);
                }
                break;
            }
            prev_variance = stats.variance;
            
            // Compute gradient
            let (grad_b_ee, grad_b_en) = self.compute_gradient(&current_wfn, stats.variance);
            
            // Update parameters with gradient descent
            let mut params = current_wfn.get_jastrow_params();
            params.b_ee -= self.learning_rate * grad_b_ee;
            params.b_en -= self.learning_rate * grad_b_en;
            
            // Clamp to reasonable values
            params.b_ee = params.b_ee.clamp(0.1, 10.0);
            params.b_en = params.b_en.clamp(0.1, 10.0);
            
            current_wfn.set_jastrow_params(params);
        }
        
        // Final sampling with more samples
        let final_stats = self.sample(&current_wfn);
        
        OptimizationResult {
            params: current_wfn.get_jastrow_params(),
            final_energy: final_stats.energy,
            final_variance: final_stats.variance,
            energy_history,
            variance_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sampling_produces_reasonable_energy() {
        let wfn = MethaneGTO::new(1.0, 2.0);
        let optimizer = JastrowOptimizer::new()
            .with_n_samples(1000)
            .with_verbose(false);
        
        let stats = optimizer.sample(&wfn);
        
        // Energy should be negative for a bound molecule
        assert!(stats.energy < 0.0, "Energy should be negative, got {}", stats.energy);
        // Variance should be positive
        assert!(stats.variance > 0.0, "Variance should be positive");
    }
    
    #[test]
    fn test_variance_calculation() {
        let wfn = MethaneGTO::new(1.0, 2.0);
        let optimizer = JastrowOptimizer::new()
            .with_n_samples(2000)
            .with_verbose(false);
        
        let stats = optimizer.sample(&wfn);
        
        // Standard error should be smaller than energy magnitude
        assert!(stats.error < stats.energy.abs(), 
            "Error {} should be smaller than |energy| {}", stats.error, stats.energy.abs());
    }
}
