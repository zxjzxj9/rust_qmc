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
    /// 
    /// Uses a two-phase approach:
    /// 1. Grid search to find approximate optimum
    /// 2. Local refinement with gradient descent
    pub fn optimize(&self, wfn: &MethaneGTO) -> OptimizationResult {
        let mut energy_history = Vec::new();
        let mut variance_history = Vec::new();
        
        if self.verbose {
            println!("Starting Jastrow optimization...");
            println!("Initial parameters: b_ee={:.4}, b_en={:.4}",
                wfn.get_jastrow_params().b_ee, wfn.get_jastrow_params().b_en);
            println!("\nPhase 1: Grid Search");
        }
        
        // Phase 1: Grid search to find good starting point
        let b_ee_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
        let b_en_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let mut best_params = wfn.get_jastrow_params();
        let mut best_variance = f64::MAX;
        let mut best_energy = 0.0;
        
        for &b_ee in &b_ee_values {
            for &b_en in &b_en_values {
                let params = JastrowParams { b_ee, b_en };
                let test_wfn = wfn.with_jastrow_params(params);
                let stats = self.sample(&test_wfn);
                
                if self.verbose {
                    println!("  b_ee={:.2}, b_en={:.2}: E={:8.4} ± {:.4}, σ²={:.2}",
                        b_ee, b_en, stats.energy, stats.error, stats.variance);
                }
                
                if stats.variance < best_variance {
                    best_variance = stats.variance;
                    best_params = params;
                    best_energy = stats.energy;
                }
            }
        }
        
        if self.verbose {
            println!("\nBest from grid: b_ee={:.4}, b_en={:.4}, σ²={:.4}",
                best_params.b_ee, best_params.b_en, best_variance);
            println!("\nPhase 2: Local Refinement");
        }
        
        // Phase 2: Local refinement around best point
        let mut current_wfn = wfn.with_jastrow_params(best_params);
        let mut prev_variance = best_variance;
        
        // Use smaller learning rate for refinement
        let refine_lr = self.learning_rate * 0.5;
        let refine_steps = (self.max_iterations / 3).max(5);
        
        for iter in 0..refine_steps {
            let stats = self.sample(&current_wfn);
            energy_history.push(stats.energy);
            variance_history.push(stats.variance);
            
            if self.verbose {
                let params = current_wfn.get_jastrow_params();
                println!("Iter {:3}: E = {:8.4} ± {:.4}, σ² = {:8.4}, b_ee = {:.4}, b_en = {:.4}",
                    iter, stats.energy, stats.error, stats.variance, params.b_ee, params.b_en);
            }
            
            // Check convergence
            let variance_change = (prev_variance - stats.variance).abs() / prev_variance.abs().max(1.0);
            if variance_change < self.tolerance && iter > 3 {
                if self.verbose {
                    println!("Converged after {} refinement iterations", iter + 1);
                }
                break;
            }
            prev_variance = stats.variance;
            
            // Compute gradient with averaged sampling (more stable)
            let params = current_wfn.get_jastrow_params();
            let h = self.fd_step * 0.5;  // Smaller step for refinement
            
            // Gradient w.r.t. b_ee (average of two samples)
            let params_plus = JastrowParams { b_ee: params.b_ee + h, b_en: params.b_en };
            let params_minus = JastrowParams { b_ee: params.b_ee - h, b_en: params.b_en };
            let var_plus = self.sample(&wfn.with_jastrow_params(params_plus)).variance;
            let var_minus = self.sample(&wfn.with_jastrow_params(params_minus)).variance;
            let grad_b_ee = (var_plus - var_minus) / (2.0 * h);
            
            // Gradient w.r.t. b_en
            let params_plus = JastrowParams { b_ee: params.b_ee, b_en: params.b_en + h };
            let params_minus = JastrowParams { b_ee: params.b_ee, b_en: params.b_en - h };
            let var_plus = self.sample(&wfn.with_jastrow_params(params_plus)).variance;
            let var_minus = self.sample(&wfn.with_jastrow_params(params_minus)).variance;
            let grad_b_en = (var_plus - var_minus) / (2.0 * h);
            
            // Clip gradients to prevent wild jumps
            let max_grad = 50.0;
            let grad_b_ee = grad_b_ee.clamp(-max_grad, max_grad);
            let grad_b_en = grad_b_en.clamp(-max_grad, max_grad);
            
            // Update parameters
            let mut new_params = params;
            new_params.b_ee -= refine_lr * grad_b_ee;
            new_params.b_en -= refine_lr * grad_b_en;
            
            // Clamp to reasonable values
            new_params.b_ee = new_params.b_ee.clamp(0.3, 8.0);
            new_params.b_en = new_params.b_en.clamp(0.5, 8.0);
            
            current_wfn.set_jastrow_params(new_params);
        }
        
        // Final sampling with more samples for accurate estimate
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
