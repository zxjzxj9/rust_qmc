//! Markov Chain Monte Carlo (MCMC) implementation for Variational Monte Carlo.
//!
//! This module provides the core VMC sampling algorithm using the Metropolis-Hastings algorithm.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::wavefunction::MultiWfn;
use super::traits::EnergyCalculator;

/// Parameters for MCMC simulation.
#[derive(Copy, Clone, Debug)]
pub struct MCMCParams {
    pub n_walkers: usize,
    pub n_steps: usize,
    pub initial_step_size: f64,
    pub max_step_size: f64,
    pub min_step_size: f64,
    pub target_acceptance: f64,
    pub adaptation_interval: usize,
}

/// State of a single walker in the MCMC simulation.
pub struct MCMCState {
    pub positions: Vec<Vector3<f64>>,
    pub wavefunction: f64,
    pub energy: f64,
}

/// Results of an MCMC simulation.
pub struct MCMCResults {
    pub energy: f64,
    pub error: f64,
    pub autocorrelation_time: f64,
}

/// MCMC simulation engine for variational Monte Carlo.
pub struct MCMCSimulation<T: MultiWfn + EnergyCalculator> {
    wavefunction: T,
    params: MCMCParams,
    rng: rand::rngs::ThreadRng,
    step_size: f64,
}

impl<T: MultiWfn + EnergyCalculator> MCMCSimulation<T> {
    pub fn new(wavefunction: T, params: MCMCParams) -> Self {
        let step_size = params.initial_step_size;
        Self {
            wavefunction,
            params,
            rng: rand::thread_rng(),
            step_size,
        }
    }

    /// Initialize all walkers with random positions.
    pub fn initialize(&mut self) -> Vec<MCMCState> {
        (0..self.params.n_walkers)
            .map(|_| {
                let positions = self.wavefunction.initialize();
                let wavefunction = self.wavefunction.evaluate(&positions);
                let energy = self.wavefunction.local_energy(&positions);
                MCMCState { positions, wavefunction, energy }
            })
            .collect()
    }

    /// Run the full MCMC simulation.
    pub fn run(&mut self) -> MCMCResults {
        let mut states = self.initialize();
        let mut energies = Vec::with_capacity(self.params.n_steps);
        let mut acceptance_count = 0;

        for step in 0..self.params.n_steps {
            for walker in states.iter_mut() {
                if self.metropolis_step(walker) {
                    acceptance_count += 1;
                }
            }

            let mean_energy: f64 = states.iter().map(|s| s.energy).sum::<f64>() 
                / self.params.n_walkers as f64;
            energies.push(mean_energy);

            if (step + 1) % self.params.adaptation_interval == 0 {
                self.adapt_step_size(acceptance_count);
                acceptance_count = 0;
            }
        }

        self.compute_results(&energies)
    }

    /// Perform a single Metropolis step for one walker.
    fn metropolis_step(&mut self, state: &mut MCMCState) -> bool {
        let normal = Normal::new(0.0, self.step_size).unwrap();
        
        let new_positions: Vec<Vector3<f64>> = state.positions.iter()
            .map(|pos| {
                Vector3::new(
                    pos[0] + normal.sample(&mut self.rng),
                    pos[1] + normal.sample(&mut self.rng),
                    pos[2] + normal.sample(&mut self.rng),
                )
            })
            .collect();

        let new_wavefunction = self.wavefunction.evaluate(&new_positions);
        let acceptance_ratio = (new_wavefunction / state.wavefunction).powi(2);

        if self.rng.gen::<f64>() < acceptance_ratio {
            state.positions = new_positions;
            state.wavefunction = new_wavefunction;
            state.energy = self.wavefunction.local_energy(&state.positions);
            true
        } else {
            false
        }
    }

    /// Adapt the step size to achieve target acceptance rate.
    fn adapt_step_size(&mut self, acceptance_count: usize) {
        let total_moves = self.params.n_walkers * self.params.adaptation_interval;
        let acceptance_rate = acceptance_count as f64 / total_moves as f64;
        let adjustment = (acceptance_rate / self.params.target_acceptance).sqrt();
        self.step_size = (self.step_size * adjustment)
            .clamp(self.params.min_step_size, self.params.max_step_size);
    }

    /// Compute final statistics from energy samples.
    fn compute_results(&self, energies: &[f64]) -> MCMCResults {
        let n = energies.len() as f64;
        let energy = energies.iter().sum::<f64>() / n;
        let autocorrelation_time = self.compute_autocorrelation_time(energies);
        let error = self.compute_error(energies, autocorrelation_time);

        MCMCResults { energy, error, autocorrelation_time }
    }

    /// Estimate autocorrelation time using initial positive sequence.
    fn compute_autocorrelation_time(&self, energies: &[f64]) -> f64 {
        let n = energies.len();
        let mean = energies.iter().sum::<f64>() / n as f64;
        let var = energies.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        
        if var == 0.0 {
            return 1.0;
        }

        let mut autocorr = 1.0;
        for t in 1..n / 2 {
            let auto_t: f64 = energies[..n - t].iter()
                .zip(energies[t..].iter())
                .map(|(&x, &y)| (x - mean) * (y - mean))
                .sum::<f64>() / ((n - t) as f64 * var);

            if auto_t < 0.0 {
                break;
            }
            autocorr += 2.0 * auto_t;
        }
        autocorr
    }

    /// Compute error using blocking method.
    fn compute_error(&self, energies: &[f64], autocorrelation_time: f64) -> f64 {
        let block_size = (2.0 * autocorrelation_time).ceil() as usize;
        let n_blocks = energies.len() / block_size;
        
        if n_blocks < 2 {
            return 0.0;
        }

        let block_means: Vec<f64> = (0..n_blocks)
            .map(|i| {
                let start = i * block_size;
                let end = start + block_size;
                energies[start..end].iter().sum::<f64>() / block_size as f64
            })
            .collect();

        let mean = block_means.iter().sum::<f64>() / n_blocks as f64;
        let variance = block_means.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n_blocks - 1) as f64;

        (variance / n_blocks as f64).sqrt()
    }
}
