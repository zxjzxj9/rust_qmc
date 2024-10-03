use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64;
use crate::h2_mol::MultiWfn;

#[derive(Copy, Clone)]
pub struct MCMCParams {
    pub n_walkers: usize,
    pub n_steps: usize,
    pub initial_step_size: f64,
    pub max_step_size: f64,
    pub min_step_size: f64,
    pub target_acceptance: f64,
    pub adaptation_interval: usize,
}

pub struct MCMCState {
    pub positions: Vec<Vector3<f64>>,
    pub wavefunction: f64,
    pub energy: f64,
}

pub struct MCMCResults {
    pub energy: f64,
    pub error: f64,
    pub autocorrelation_time: f64,
}

pub trait EnergyCalculator {
    fn local_energy(&self, positions: &Vec<Vector3<f64>>) -> f64;
}

pub struct MCMCSimulation<T: MultiWfn + EnergyCalculator> {
    wavefunction: T,
    params: MCMCParams,
    rng: rand::rngs::ThreadRng,
    step_size: f64,
}

impl<T: MultiWfn + EnergyCalculator> MCMCSimulation<T> {
    pub fn new(wavefunction: T, params: MCMCParams) -> Self {
        Self {
            wavefunction,
            params: params.clone(),
            rng: rand::thread_rng(),
            step_size: params.initial_step_size,
        }
    }

    pub fn initialize(&self) -> Vec<MCMCState> {
        let mut states = Vec::with_capacity(self.params.n_walkers);
        for _ in 0..self.params.n_walkers {
            let positions = self.generate_random_positions();
            let wavefunction = self.wavefunction.evaluate(&positions);
            let energy = self.wavefunction.local_energy(&positions);
            states.push(MCMCState { positions, wavefunction, energy });
        }
        states
    }

    fn generate_random_positions(&self) -> Vec<Vector3<f64>> {
        // Implement based on your system's requirements
        self.wavefunction.initialize()
    }

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

            let mean_energy: f64 = states.iter().map(|s| s.energy).sum::<f64>() / self.params.n_walkers as f64;
            energies.push(mean_energy);

            // Adapt step size
            if (step + 1) % self.params.adaptation_interval == 0 {
                self.adapt_step_size(acceptance_count);
                acceptance_count = 0;
            }
        }

        self.compute_results(energies)
    }

    fn metropolis_step(&mut self, state: &mut MCMCState) -> bool {
        let mut new_positions = state.positions.clone();

        // Propose a move for each particle
        for pos in new_positions.iter_mut() {
            let normal = Normal::new(0.0, self.step_size).unwrap();
            for i in 0..3 {
                pos[i] += normal.sample(&mut self.rng);
            }
        }

        let new_wavefunction = self.wavefunction.evaluate(&new_positions);
        let acceptance_ratio = (new_wavefunction / state.wavefunction).powi(2);

        if self.rng.gen::<f64>() < acceptance_ratio {
            // Accept the move
            state.positions = new_positions;
            state.wavefunction = new_wavefunction;
            state.energy = self.wavefunction.local_energy(&state.positions);
            true
        } else {
            false
        }
    }

    fn adapt_step_size(&mut self, acceptance_count: usize) {
        let acceptance_rate = acceptance_count as f64 / (self.params.n_walkers * self.params.adaptation_interval) as f64;
        let adjustment_factor = (acceptance_rate / self.params.target_acceptance).sqrt();
        self.step_size *= adjustment_factor;
        self.step_size = self.step_size.min(self.params.max_step_size);
        self.step_size = self.step_size.max(self.params.min_step_size);
    }

    fn compute_results(&self, energies: Vec<f64>) -> MCMCResults {
        let energy: f64 = energies.iter().sum::<f64>() / energies.len() as f64;

        // Compute autocorrelation time
        let autocorrelation_time = self.compute_autocorrelation_time(&energies);

        // Compute error using blocking method
        let error = self.compute_error(&energies, autocorrelation_time);

        MCMCResults {
            energy,
            error,
            autocorrelation_time,
        }
    }

    fn compute_autocorrelation_time(&self, energies: &[f64]) -> f64 {
        let mean = energies.iter().sum::<f64>() / energies.len() as f64;
        let var = energies.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / energies.len() as f64;

        let mut autocorr = 1.0;
        let mut t = 0;

        while t < energies.len() / 2 {
            t += 1;
            let auto_t = energies.windows(energies.len() - t)
                .zip(energies[t..].iter())
                .map(|(w, &y)| (w[0] - mean) * (y - mean))
                .sum::<f64>() / ((energies.len() - t) as f64 * var);

            if auto_t < 0.0 {
                break;
            }

            autocorr += 2.0 * auto_t;
        }

        autocorr
    }

    fn compute_error(&self, energies: &[f64], autocorrelation_time: f64) -> f64 {
        let block_size = (2.0 * autocorrelation_time).ceil() as usize;
        let n_blocks = energies.len() / block_size;

        let block_means: Vec<f64> = (0..n_blocks)
            .map(|i| energies[i*block_size..(i+1)*block_size].iter().sum::<f64>() / block_size as f64)
            .collect();

        let mean = block_means.iter().sum::<f64>() / n_blocks as f64;
        let variance = block_means.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_blocks - 1) as f64;

        (variance / n_blocks as f64).sqrt()
    }
}