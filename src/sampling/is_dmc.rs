//! Importance-Sampled Diffusion Monte Carlo (IS-DMC) with force estimation.
//!
//! This module implements DMC with importance sampling using a trial wavefunction
//! Ψ_T to guide the diffusion process. The walkers undergo drift-diffusion moves
//! (Langevin dynamics) biased by the quantum force F = 2∇Ψ_T/Ψ_T, followed by
//! stochastic branching based on the local energy.
//!
//! # Algorithm
//!
//! 1. **Drift-diffusion move**: For each electron i,
//!    r'_i = r_i + τ F_i(r) + η √(2τ)
//!    Accept/reject with Green's function ratio (same as VMC drift-diffusion).
//!
//! 2. **Branching**: After a full sweep, compute the branching weight
//!    W = exp(-τ (E_L(r') + E_L(r))/2 + τ E_ref)
//!    Generate m = floor(W + ξ) copies. Kill if m=0, keep if m=1, clone if m>1.
//!
//! 3. **E_ref feedback**: E_ref is updated to control the walker population:
//!    E_ref += κ ln(N_target / N_current)
//!
//! # Force Estimation
//!
//! Forces are accumulated using the **mixed estimator**:
//!   F_mixed = ⟨Ψ_0|F_HF|Ψ_T⟩ / ⟨Ψ_0|Ψ_T⟩
//!
//! In practice this is the weighted average of F_HF over the DMC population.
//! The **extrapolated estimator** corrects for finite trial-wavefunction bias:
//!   F_extrap = 2 F_mixed - F_VMC
//!
//! # References
//!
//! - W.M.C. Foulkes et al., Rev. Mod. Phys. 73, 33 (2001)
//! - C.J. Umrigar, M.P. Nightingale, K.J. Runge, J. Chem. Phys. 99, 2865 (1993)
//! - R. Assaraf & M. Caffarel, J. Chem. Phys. 113, 4028 (2000)

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::wavefunction::MultiWfn;
use super::traits::{EnergyCalculator, ForceCalculator};

/// Maximum number of clones per walker to prevent population explosion.
const MAX_CLONES: usize = 3;

// =============================================================================
// Configuration
// =============================================================================

/// Parameters for importance-sampled DMC.
#[derive(Clone, Debug)]
pub struct ISDMCParams {
    /// Target number of walkers
    pub n_walkers: usize,
    /// Number of production (measurement) steps
    pub n_steps: usize,
    /// Number of equilibration steps (no measurements)
    pub n_burnin: usize,
    /// Time step τ for drift-diffusion dynamics
    pub time_step: f64,
    /// Maximum drift displacement per electron (prevents node divergence)
    pub max_drift: f64,
    /// Damping parameter for E_ref updates
    pub e_ref_damping: f64,
    /// Number of VMC pre-equilibration sweeps before DMC
    pub n_vmc_equilibrate: usize,
    /// Number of VMC samples for the VMC force estimate (for extrapolated estimator)
    pub n_vmc_force_samples: usize,
    /// Whether to print progress every N steps (0 = silent)
    pub print_interval: usize,
}

impl Default for ISDMCParams {
    fn default() -> Self {
        Self {
            n_walkers: 200,
            n_steps: 5000,
            n_burnin: 1000,
            time_step: 0.005,
            max_drift: 1.0,
            e_ref_damping: 0.5,
            n_vmc_equilibrate: 500,
            n_vmc_force_samples: 2000,
            print_interval: 500,
        }
    }
}

// =============================================================================
// Results
// =============================================================================

/// Results from an IS-DMC simulation.
#[derive(Clone, Debug)]
pub struct ISDMCResults {
    /// DMC mixed-estimator energy
    pub energy: f64,
    /// Statistical error on energy
    pub error: f64,
    /// Mixed-estimator forces per nucleus: ⟨Ψ_0|F_HF|Ψ_T⟩
    pub forces_mixed: Vec<Vector3<f64>>,
    /// VMC forces per nucleus: ⟨Ψ_T|F_HF|Ψ_T⟩
    pub forces_vmc: Vec<Vector3<f64>>,
    /// Extrapolated forces: 2×F_mixed - F_vmc
    pub forces_extrapolated: Vec<Vector3<f64>>,
    /// Mean acceptance rate per electron
    pub acceptance_rate: f64,
    /// Average walker population during production
    pub avg_population: f64,
}

// =============================================================================
// Walker
// =============================================================================

/// Internal state of a single IS-DMC walker.
#[derive(Clone)]
struct ISDMCWalker {
    /// Electron positions
    positions: Vec<Vector3<f64>>,
    /// Wavefunction value Ψ_T(R)
    psi: f64,
    /// Quantum drift F_i = 2∇_iΨ_T/Ψ_T for each electron
    drift: Vec<Vector3<f64>>,
    /// Local energy E_L(R)
    local_energy: f64,
    /// Walker age (steps since last branching event)
    age: usize,
}

impl ISDMCWalker {
    /// Compute drift forces from wavefunction gradients.
    fn compute_drift<T: MultiWfn>(
        wfn: &T,
        positions: &[Vector3<f64>],
        psi: f64,
    ) -> Vec<Vector3<f64>> {
        if psi.abs() < 1e-30 {
            return vec![Vector3::zeros(); positions.len()];
        }
        let grad = wfn.derivative(positions);
        grad.iter().map(|g| 2.0 * g / psi).collect()
    }

    /// Create a new walker at a random initial configuration.
    fn new<T: MultiWfn + EnergyCalculator>(wfn: &T) -> Self {
        let positions = wfn.initialize();
        let psi = wfn.evaluate(&positions);
        let drift = Self::compute_drift(wfn, &positions, psi);
        let local_energy = wfn.local_energy(&positions);
        Self {
            positions,
            psi,
            drift,
            local_energy,
            age: 0,
        }
    }

    /// Refresh all cached quantities from current positions.
    fn refresh<T: MultiWfn + EnergyCalculator>(&mut self, wfn: &T) {
        self.psi = wfn.evaluate(&self.positions);
        self.drift = Self::compute_drift(wfn, &self.positions, self.psi);
        self.local_energy = wfn.local_energy(&self.positions);
    }
}

// =============================================================================
// IS-DMC Engine
// =============================================================================

/// Importance-sampled Diffusion Monte Carlo engine.
///
/// Generic over any wavefunction that implements `MultiWfn`, `EnergyCalculator`,
/// and `ForceCalculator`.
pub struct ImportanceSampledDMC<T: MultiWfn + EnergyCalculator + ForceCalculator> {
    wfn: T,
    params: ISDMCParams,
    rng: rand::rngs::ThreadRng,
    time_step: f64,
}

impl<T: MultiWfn + EnergyCalculator + ForceCalculator + Clone> ImportanceSampledDMC<T> {
    /// Create a new IS-DMC simulation.
    pub fn new(wfn: T, params: ISDMCParams) -> Self {
        let time_step = params.time_step;
        Self {
            wfn,
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

    /// Perform one sweep: attempt to move each electron once via Langevin dynamics.
    /// Returns the number of accepted moves.
    fn drift_diffusion_sweep(&mut self, walker: &mut ISDMCWalker) -> usize {
        let n_elec = walker.positions.len();
        let tau = self.time_step;
        let sqrt_2tau = (2.0 * tau).sqrt();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut accepted = 0;

        for i in 0..n_elec {
            // Propose new position for electron i
            let limited_drift = self.limit_drift(&walker.drift[i]);
            let new_pos_i = walker.positions[i]
                + limited_drift
                + sqrt_2tau
                    * Vector3::new(
                        normal.sample(&mut self.rng),
                        normal.sample(&mut self.rng),
                        normal.sample(&mut self.rng),
                    );

            // Create trial configuration
            let mut trial_positions = walker.positions.clone();
            trial_positions[i] = new_pos_i;

            // Evaluate wavefunction at trial position
            let new_psi = self.wfn.evaluate(&trial_positions);

            if new_psi.abs() < 1e-30 {
                continue;
            }

            // Compute new drift at trial position
            let new_drift =
                ISDMCWalker::compute_drift(&self.wfn, &trial_positions, new_psi);

            // Green's function ratio for asymmetric acceptance
            // Forward: G(r→r') ~ exp(-|r'-r-τF(r)|²/(4τ))
            let limited_old = self.limit_drift(&walker.drift[i]);
            let fwd_diff = new_pos_i - walker.positions[i] - limited_old;
            let g_fwd = -fwd_diff.norm_squared() / (4.0 * tau);

            // Backward: G(r'→r) ~ exp(-|r-r'-τF(r')|²/(4τ))
            let limited_new = self.limit_drift(&new_drift[i]);
            let bwd_diff = walker.positions[i] - new_pos_i - limited_new;
            let g_bwd = -bwd_diff.norm_squared() / (4.0 * tau);

            // Acceptance ratio
            let psi_ratio_sq = (new_psi / walker.psi).powi(2);
            let greens_ratio = (g_bwd - g_fwd).exp();
            let acceptance = (psi_ratio_sq * greens_ratio).min(1.0);

            if self.rng.gen::<f64>() < acceptance {
                walker.positions = trial_positions;
                walker.psi = new_psi;
                walker.drift = new_drift;
                accepted += 1;
            }
        }

        // Update local energy after the full sweep
        walker.local_energy = self.wfn.local_energy(&walker.positions);

        accepted
    }

    /// Perform branching on the walker population.
    ///
    /// Each walker's branching weight is:
    ///   W = exp(-τ (E_L(R_new) + E_L(R_old))/2 + τ E_ref)
    ///
    /// The number of copies is m = floor(W + ξ), capped at MAX_CLONES.
    fn branch(
        &mut self,
        walkers: &mut Vec<ISDMCWalker>,
        old_energies: &[f64],
        e_ref: f64,
    ) {
        let tau = self.time_step;
        let mut new_walkers = Vec::with_capacity(walkers.len());

        for (walker, &e_old) in walkers.iter().zip(old_energies.iter()) {
            let e_avg = 0.5 * (walker.local_energy + e_old);
            let weight = (tau * (e_ref - e_avg)).exp();

            let xi: f64 = self.rng.gen();
            let m = ((weight + xi).floor() as usize).min(MAX_CLONES);

            for _ in 0..m {
                let mut clone = walker.clone();
                clone.age = if m > 1 { 0 } else { walker.age + 1 };
                new_walkers.push(clone);
            }
        }

        *walkers = new_walkers;
    }

    /// Run VMC sampling to compute ⟨F_HF⟩_VMC (needed for extrapolated estimator).
    fn vmc_forces(&mut self) -> Vec<Vector3<f64>> {
        let n_nuc = self.wfn.num_nuclei();
        let n_walkers = self.params.n_walkers.min(50); // Use modest number for VMC
        let normal = Normal::new(0.0, 0.5).unwrap();

        // Initialize VMC walkers
        let mut positions: Vec<Vec<Vector3<f64>>> = (0..n_walkers)
            .map(|_| self.wfn.initialize())
            .collect();
        let mut psi_values: Vec<f64> = positions
            .iter()
            .map(|r| self.wfn.evaluate(r))
            .collect();

        // Equilibrate
        for _ in 0..self.params.n_vmc_equilibrate {
            for (widx, pos) in positions.iter_mut().enumerate() {
                let new_pos: Vec<Vector3<f64>> = pos
                    .iter()
                    .map(|p| {
                        p + Vector3::new(
                            normal.sample(&mut self.rng),
                            normal.sample(&mut self.rng),
                            normal.sample(&mut self.rng),
                        )
                    })
                    .collect();
                let new_psi = self.wfn.evaluate(&new_pos);
                let ratio = (new_psi / psi_values[widx]).powi(2);
                if self.rng.gen::<f64>() < ratio {
                    *pos = new_pos;
                    psi_values[widx] = new_psi;
                }
            }
        }

        // Accumulate forces
        let mut force_accum = vec![Vector3::zeros(); n_nuc];
        let mut n_collected = 0usize;
        let steps = (self.params.n_vmc_force_samples / n_walkers).max(1);

        for _ in 0..steps {
            // Decorrelation
            for _ in 0..5 {
                for (widx, pos) in positions.iter_mut().enumerate() {
                    let new_pos: Vec<Vector3<f64>> = pos
                        .iter()
                        .map(|p| {
                            p + Vector3::new(
                                normal.sample(&mut self.rng),
                                normal.sample(&mut self.rng),
                                normal.sample(&mut self.rng),
                            )
                        })
                        .collect();
                    let new_psi = self.wfn.evaluate(&new_pos);
                    let ratio = (new_psi / psi_values[widx]).powi(2);
                    if self.rng.gen::<f64>() < ratio {
                        *pos = new_pos;
                        psi_values[widx] = new_psi;
                    }
                }
            }

            for pos in positions.iter() {
                let f = self.wfn.hellmann_feynman_force(pos);
                for (i, fi) in f.iter().enumerate() {
                    force_accum[i] += fi;
                }
                n_collected += 1;
            }
        }

        let n = n_collected as f64;
        force_accum.iter().map(|f| f / n).collect()
    }

    /// Run the full IS-DMC simulation.
    ///
    /// Returns energy, forces, and statistics.
    pub fn run(&mut self) -> ISDMCResults {
        let n_nuc = self.wfn.num_nuclei();

        // —— Phase 1: VMC forces ——
        let forces_vmc = self.vmc_forces();

        // —— Phase 2: Initialize DMC walkers ——
        let mut walkers: Vec<ISDMCWalker> = (0..self.params.n_walkers)
            .map(|_| ISDMCWalker::new(&self.wfn))
            .collect();

        // Initial E_ref from VMC local energies
        let mut e_ref: f64 = walkers.iter().map(|w| w.local_energy).sum::<f64>()
            / walkers.len() as f64;

        let mut total_accepted: usize = 0;
        let mut total_moves: usize = 0;

        // —— Phase 3: Burn-in ——
        if self.params.print_interval > 0 {
            println!("IS-DMC burn-in ({} steps, {} walkers, τ={:.4})...",
                self.params.n_burnin, self.params.n_walkers, self.time_step);
        }

        for _step in 0..self.params.n_burnin {
            let n_elec = walkers[0].positions.len();
            let old_energies: Vec<f64> = walkers.iter().map(|w| w.local_energy).collect();

            for walker in walkers.iter_mut() {
                let acc = self.drift_diffusion_sweep(walker);
                total_accepted += acc;
                total_moves += n_elec;
            }

            self.branch(&mut walkers, &old_energies, e_ref);

            // Update E_ref with population feedback
            let n_current = walkers.len() as f64;
            let n_target = self.params.n_walkers as f64;
            let e_mean: f64 = walkers.iter().map(|w| w.local_energy).sum::<f64>() / n_current;
            e_ref = e_mean - self.params.e_ref_damping * (n_current / n_target).ln() / self.time_step;

            // Safety: prevent population collapse or explosion
            if walkers.is_empty() {
                if self.params.print_interval > 0 {
                    println!("  WARNING: Population collapsed! Re-initializing...");
                }
                walkers = (0..self.params.n_walkers)
                    .map(|_| ISDMCWalker::new(&self.wfn))
                    .collect();
                e_ref = walkers.iter().map(|w| w.local_energy).sum::<f64>()
                    / walkers.len() as f64;
            }
        }

        // —— Phase 4: Production ——
        if self.params.print_interval > 0 {
            let rate = if total_moves > 0 {
                total_accepted as f64 / total_moves as f64
            } else {
                0.0
            };
            println!("IS-DMC production ({} steps), burn-in acceptance: {:.1}%",
                self.params.n_steps, rate * 100.0);
        }

        let mut energy_samples = Vec::with_capacity(self.params.n_steps);
        let mut force_accum = vec![Vector3::zeros(); n_nuc];
        let mut n_force_samples = 0usize;
        let mut pop_sum = 0.0_f64;
        let mut prod_accepted: usize = 0;
        let mut prod_moves: usize = 0;

        for step in 0..self.params.n_steps {
            let n_elec = walkers[0].positions.len();
            let old_energies: Vec<f64> = walkers.iter().map(|w| w.local_energy).collect();

            for walker in walkers.iter_mut() {
                let acc = self.drift_diffusion_sweep(walker);
                prod_accepted += acc;
                prod_moves += n_elec;
            }

            self.branch(&mut walkers, &old_energies, e_ref);

            let n_current = walkers.len() as f64;
            pop_sum += n_current;

            // Energy: mixed estimator is simply the mean local energy of the DMC population
            let e_mean: f64 =
                walkers.iter().map(|w| w.local_energy).sum::<f64>() / n_current;
            energy_samples.push(e_mean);

            // Force accumulation: ⟨F_HF⟩ over all walkers
            for walker in walkers.iter() {
                let f = self.wfn.hellmann_feynman_force(&walker.positions);
                for (i, fi) in f.iter().enumerate() {
                    force_accum[i] += fi;
                }
                n_force_samples += 1;
            }

            // Update E_ref
            let n_target = self.params.n_walkers as f64;
            e_ref = e_mean
                - self.params.e_ref_damping * (n_current / n_target).ln() / self.time_step;

            // Safety
            if walkers.is_empty() {
                if self.params.print_interval > 0 {
                    println!("  WARNING: Population collapsed at step {}! Re-initializing...", step);
                }
                walkers = (0..self.params.n_walkers)
                    .map(|_| ISDMCWalker::new(&self.wfn))
                    .collect();
                e_ref = walkers.iter().map(|w| w.local_energy).sum::<f64>()
                    / walkers.len() as f64;
            }

            // Progress
            if self.params.print_interval > 0
                && (step + 1) % self.params.print_interval == 0
            {
                let running_mean =
                    energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
                println!(
                    "  Step {:5}: E = {:10.6}, pop = {:4}, E_ref = {:10.6}",
                    step + 1,
                    running_mean,
                    walkers.len(),
                    e_ref
                );
            }
        }

        // —— Phase 5: Compute results ——
        let n_prod = energy_samples.len() as f64;
        let energy = energy_samples.iter().sum::<f64>() / n_prod;
        let error = self.block_error(&energy_samples);

        let forces_mixed: Vec<Vector3<f64>> = force_accum
            .iter()
            .map(|f| f / n_force_samples as f64)
            .collect();

        let forces_extrapolated: Vec<Vector3<f64>> = forces_mixed
            .iter()
            .zip(forces_vmc.iter())
            .map(|(fm, fv)| 2.0 * fm - fv)
            .collect();

        let acceptance_rate = if prod_moves > 0 {
            prod_accepted as f64 / prod_moves as f64
        } else {
            0.0
        };

        let avg_population = pop_sum / self.params.n_steps as f64;

        ISDMCResults {
            energy,
            error,
            forces_mixed,
            forces_vmc,
            forces_extrapolated,
            acceptance_rate,
            avg_population,
        }
    }

    /// Compute statistical error using blocking analysis.
    fn block_error(&self, samples: &[f64]) -> f64 {
        let n = samples.len();
        if n < 20 {
            return 0.0;
        }

        // Estimate autocorrelation time
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        if var < 1e-30 {
            return 0.0;
        }

        let mut autocorr = 1.0;
        for t in 1..n / 2 {
            let auto_t: f64 = samples[..n - t]
                .iter()
                .zip(samples[t..].iter())
                .map(|(&x, &y)| (x - mean) * (y - mean))
                .sum::<f64>()
                / ((n - t) as f64 * var);
            if auto_t < 0.0 {
                break;
            }
            autocorr += 2.0 * auto_t;
        }

        // Block size from autocorrelation time
        let block_size = (2.0 * autocorr).ceil() as usize;
        let n_blocks = n / block_size;

        if n_blocks < 2 {
            return (var * autocorr / n as f64).sqrt();
        }

        let block_means: Vec<f64> = (0..n_blocks)
            .map(|i| {
                let start = i * block_size;
                let end = start + block_size;
                samples[start..end].iter().sum::<f64>() / block_size as f64
            })
            .collect();

        let block_mean = block_means.iter().sum::<f64>() / n_blocks as f64;
        let block_var = block_means
            .iter()
            .map(|&x| (x - block_mean).powi(2))
            .sum::<f64>()
            / (n_blocks - 1) as f64;

        (block_var / n_blocks as f64).sqrt()
    }
}
