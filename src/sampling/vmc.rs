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

// =============================================================================
// Drift-Diffusion VMC with Single-Electron Moves
// =============================================================================

/// Parameters for drift-diffusion VMC.
#[derive(Copy, Clone, Debug)]
pub struct DDVMCParams {
    /// Number of walkers
    pub n_walkers: usize,
    /// Number of production steps
    pub n_steps: usize,
    /// Number of burn-in steps (discarded)
    pub n_burnin: usize,
    /// Time step τ for Langevin dynamics
    pub time_step: f64,
    /// Maximum drift displacement per electron (prevents node divergence)
    pub max_drift: f64,
    /// Target acceptance rate for time step adaptation
    pub target_acceptance: f64,
    /// Adaptation interval (in sweeps)
    pub adaptation_interval: usize,
}

impl Default for DDVMCParams {
    fn default() -> Self {
        Self {
            n_walkers: 10,
            n_steps: 100_000,
            n_burnin: 1000,
            time_step: 0.01,
            max_drift: 1.0,
            target_acceptance: 0.5,
            adaptation_interval: 100,
        }
    }
}

/// Results from drift-diffusion VMC.
pub struct DDVMCResults {
    /// Mean energy
    pub energy: f64,
    /// Statistical error
    pub error: f64,
    /// Autocorrelation time
    pub autocorrelation_time: f64,
    /// Mean acceptance rate per electron
    pub acceptance_rate: f64,
    /// Final time step
    pub final_time_step: f64,
}

/// Internal state of a drift-diffusion walker.
struct DDWalkerState {
    /// Electron positions
    positions: Vec<Vector3<f64>>,
    /// Wavefunction value ψ(r)
    psi: f64,
    /// Drift forces F_i = 2∇_iψ/ψ for each electron
    drift: Vec<Vector3<f64>>,
}

impl DDWalkerState {
    /// Compute drift forces from wavefunction gradients.
    /// F_i = 2 ∇_i ψ / ψ = 2 ∇_i ln|ψ| (the quantum force)
    fn compute_drift<T: MultiWfn>(wfn: &T, positions: &[Vector3<f64>], psi: f64) -> Vec<Vector3<f64>> {
        if psi.abs() < 1e-30 {
            return vec![Vector3::zeros(); positions.len()];
        }
        let grad = wfn.derivative(positions);
        grad.iter().map(|g| 2.0 * g / psi).collect()
    }

    /// Initialize a walker state.
    fn new<T: MultiWfn>(wfn: &T) -> Self {
        let positions = wfn.initialize();
        let psi = wfn.evaluate(&positions);
        let drift = Self::compute_drift(wfn, &positions, psi);
        Self { positions, psi, drift }
    }

    /// Refresh drift and psi from current positions.
    fn refresh<T: MultiWfn>(&mut self, wfn: &T) {
        self.psi = wfn.evaluate(&self.positions);
        self.drift = Self::compute_drift(wfn, &self.positions, self.psi);
    }
}

/// Drift-diffusion VMC engine with single-electron moves.
///
/// Uses Langevin dynamics to propose moves biased by the quantum force,
/// with Green's function ratio in the acceptance criterion.
pub struct DriftDiffusionVMC<T: MultiWfn + EnergyCalculator> {
    wavefunction: T,
    params: DDVMCParams,
    rng: rand::rngs::ThreadRng,
    time_step: f64,
}

impl<T: MultiWfn + EnergyCalculator> DriftDiffusionVMC<T> {
    /// Create a new drift-diffusion VMC simulation.
    pub fn new(wavefunction: T, params: DDVMCParams) -> Self {
        let time_step = params.time_step;
        Self {
            wavefunction,
            params,
            rng: rand::thread_rng(),
            time_step,
        }
    }

    /// Limit drift displacement to prevent divergence near nodes.
    fn limit_drift(&self, drift: &Vector3<f64>) -> Vector3<f64> {
        let disp = self.time_step * drift;
        let disp_mag = disp.norm();
        if disp_mag > self.params.max_drift {
            disp * (self.params.max_drift / disp_mag)
        } else {
            disp
        }
    }

    /// Propose a single-electron move using Langevin dynamics.
    /// Returns (new_position, forward Green's function exponent, backward Green's function exponent)
    fn propose_move(
        &mut self,
        old_pos: &Vector3<f64>,
        old_drift: &Vector3<f64>,
        new_drift: &Vector3<f64>,
        new_pos: &Vector3<f64>,
    ) -> (f64, f64) {
        let tau = self.time_step;

        // Forward Green's function exponent: -|r' - r - τF(r)|² / (4τ)
        let limited_old_drift = self.limit_drift(old_drift);
        let forward_diff = new_pos - old_pos - limited_old_drift;
        let g_forward = -forward_diff.norm_squared() / (4.0 * tau);

        // Backward Green's function exponent: -|r - r' - τF(r')|² / (4τ)
        let limited_new_drift = self.limit_drift(new_drift);
        let backward_diff = old_pos - new_pos - limited_new_drift;
        let g_backward = -backward_diff.norm_squared() / (4.0 * tau);

        (g_forward, g_backward)
    }

    /// Perform one sweep: attempt to move each electron once.
    /// Returns the number of accepted moves.
    fn sweep(&mut self, state: &mut DDWalkerState) -> usize {
        let n_elec = state.positions.len();
        let tau = self.time_step;
        let sqrt_2tau = (2.0 * tau).sqrt();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut accepted = 0;

        for i in 0..n_elec {
            // Propose new position for electron i using Langevin dynamics
            let limited_drift = self.limit_drift(&state.drift[i]);
            let new_pos_i = state.positions[i] + limited_drift + sqrt_2tau * Vector3::new(
                normal.sample(&mut self.rng),
                normal.sample(&mut self.rng),
                normal.sample(&mut self.rng),
            );

            // Create trial configuration
            let mut trial_positions = state.positions.clone();
            trial_positions[i] = new_pos_i;

            // Evaluate wavefunction at trial position
            let new_psi = self.wavefunction.evaluate(&trial_positions);

            // Skip if wavefunction is essentially zero
            if new_psi.abs() < 1e-30 {
                continue;
            }

            // Compute new drift at trial position
            let new_drift = DDWalkerState::compute_drift(
                &self.wavefunction, &trial_positions, new_psi
            );

            // Green's function ratio
            let (g_forward, g_backward) = self.propose_move(
                &state.positions[i],
                &state.drift[i],
                &new_drift[i],
                &new_pos_i,
            );

            // Acceptance ratio: |ψ(r')/ψ(r)|² × exp(G_backward - G_forward)
            let psi_ratio_sq = (new_psi / state.psi).powi(2);
            let greens_ratio = (g_backward - g_forward).exp();
            let acceptance = psi_ratio_sq * greens_ratio;

            if self.rng.gen::<f64>() < acceptance.min(1.0) {
                state.positions = trial_positions;
                state.psi = new_psi;
                state.drift = new_drift;
                accepted += 1;
            }
        }

        accepted
    }

    /// Adapt time step to achieve target acceptance rate.
    fn adapt_time_step(&mut self, acceptance_rate: f64) {
        let ratio = acceptance_rate / self.params.target_acceptance;
        // Gentle adjustment to avoid oscillation
        let adjustment = ratio.sqrt().clamp(0.8, 1.2);
        self.time_step = (self.time_step * adjustment).clamp(0.001, 0.5);
    }

    /// Run the full drift-diffusion VMC simulation.
    pub fn run(&mut self) -> DDVMCResults {
        let n_elec;

        // Initialize walkers
        let mut states: Vec<DDWalkerState> = (0..self.params.n_walkers)
            .map(|_| DDWalkerState::new(&self.wavefunction))
            .collect();
        n_elec = states[0].positions.len();

        let mut total_accepted: usize = 0;
        let mut total_moves: usize = 0;

        // Burn-in phase
        for step in 0..self.params.n_burnin {
            for state in states.iter_mut() {
                let acc = self.sweep(state);
                total_accepted += acc;
                total_moves += n_elec;
            }

            // Adapt during burn-in
            if (step + 1) % self.params.adaptation_interval == 0 && total_moves > 0 {
                let rate = total_accepted as f64 / total_moves as f64;
                self.adapt_time_step(rate);
                total_accepted = 0;
                total_moves = 0;
            }
        }

        // Production phase
        let mut energies = Vec::with_capacity(self.params.n_steps);
        total_accepted = 0;
        total_moves = 0;

        for step in 0..self.params.n_steps {
            for state in states.iter_mut() {
                let acc = self.sweep(state);
                total_accepted += acc;
                total_moves += n_elec;
            }

            // Record mean energy across walkers
            let mean_energy: f64 = states.iter()
                .map(|s| self.wavefunction.local_energy(&s.positions))
                .sum::<f64>() / self.params.n_walkers as f64;
            energies.push(mean_energy);

            // Continue adapting (gently) during production
            if (step + 1) % self.params.adaptation_interval == 0 && total_moves > 0 {
                let rate = total_accepted as f64 / total_moves as f64;
                self.adapt_time_step(rate);
                total_accepted = 0;
                total_moves = 0;
            }
        }

        let acceptance_rate = if total_moves > 0 {
            total_accepted as f64 / total_moves as f64
        } else {
            0.0
        };

        // Compute statistics
        let n = energies.len() as f64;
        let energy = energies.iter().sum::<f64>() / n;
        let autocorrelation_time = self.compute_autocorrelation_time(&energies);
        let error = self.compute_error(&energies, autocorrelation_time);

        DDVMCResults {
            energy,
            error,
            autocorrelation_time,
            acceptance_rate,
            final_time_step: self.time_step,
        }
    }

    /// Estimate autocorrelation time.
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
