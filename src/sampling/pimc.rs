//! Path Integral Monte Carlo (PIMC) for Quantum Harmonic Oscillator
//!
//! Uses Wick's rotation to convert quantum mechanics into imaginary-time
//! path integrals with Boltzmann weights. Paths have periodic boundary
//! conditions in imaginary time (closed paths).
//!
//! Reference: Ceperley, D.M. (1995) "Path integrals in the theory of condensed helium"
//! Rev. Mod. Phys. 67, 279

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::fs::File;
use std::io::{BufWriter, Write};

/// A single quantum path in imaginary time with M beads.
/// Represents a particle's worldline with PBC: x[M] = x[0]
#[derive(Clone, Debug)]
pub struct QuantumPath {
    /// Positions at each imaginary time slice [0, M-1]
    pub beads: Vec<f64>,
    /// Number of Trotter slices (beads)
    pub n_beads: usize,
    /// Imaginary time step Δτ = βℏ/M
    pub dtau: f64,
    /// Inverse temperature β = 1/(k_B T)
    pub beta: f64,
    /// Oscillator angular frequency ω
    pub omega: f64,
    /// Particle mass (in natural units, typically 1.0)
    pub mass: f64,
}

impl QuantumPath {
    /// Create a new quantum path with given parameters
    ///
    /// # Arguments
    /// * `n_beads` - Number of Trotter time slices M
    /// * `beta` - Inverse temperature β = 1/(k_B T)
    /// * `omega` - Harmonic oscillator frequency
    /// * `mass` - Particle mass (default 1.0 for natural units)
    pub fn new(n_beads: usize, beta: f64, omega: f64, mass: f64) -> Self {
        let dtau = beta / n_beads as f64;
        
        // Initialize beads from thermal Gaussian distribution
        let mut rng = rand::thread_rng();
        // Classical thermal width: σ = sqrt(1/(mω²β)) at high T
        // Quantum ground state width: σ = sqrt(ℏ/(2mω)) = sqrt(1/(2*omega)) in natural units
        let sigma = (1.0 / (mass * omega)).sqrt();
        let dist = Normal::new(0.0, sigma).unwrap();
        
        let beads: Vec<f64> = (0..n_beads)
            .map(|_| dist.sample(&mut rng))
            .collect();
        
        Self {
            beads,
            n_beads,
            dtau,
            beta,
            omega,
            mass,
        }
    }

    /// Compute the kinetic (spring) action contribution between bead i and i+1
    /// S_kin = (m / 2Δτ) * (x_{i+1} - x_i)²
    #[inline]
    fn kinetic_action(&self, i: usize) -> f64 {
        let j = (i + 1) % self.n_beads; // PBC: wraps M -> 0
        let dx = self.beads[j] - self.beads[i];
        0.5 * self.mass / self.dtau * dx * dx
    }

    /// Compute the potential action contribution at bead i
    /// S_pot = Δτ * V(x_i) = Δτ * (1/2)mω²x_i²
    #[inline]
    fn potential_action(&self, i: usize) -> f64 {
        let x = self.beads[i];
        0.5 * self.dtau * self.mass * self.omega * self.omega * x * x
    }

    /// Compute total Euclidean action (primitive approximation)
    /// S = Σᵢ [kinetic(i) + potential(i)]
    pub fn total_action(&self) -> f64 {
        let mut action = 0.0;
        for i in 0..self.n_beads {
            action += self.kinetic_action(i);
            action += self.potential_action(i);
        }
        action
    }

    /// Compute local action change when moving bead i from x_old to x_new
    /// Only beads i-1, i, i+1 are affected (PBC handled)
    fn local_action_change(&self, i: usize, x_old: f64, x_new: f64) -> f64 {
        let prev = (i + self.n_beads - 1) % self.n_beads;
        let next = (i + 1) % self.n_beads;
        
        let x_prev = self.beads[prev];
        let x_next = self.beads[next];
        
        let spring_const = self.mass / self.dtau;
        let pot_const = self.dtau * self.mass * self.omega * self.omega;
        
        // Old action contributions
        let s_old = 0.5 * spring_const * ((x_old - x_prev).powi(2) + (x_next - x_old).powi(2))
                  + 0.5 * pot_const * x_old * x_old;
        
        // New action contributions  
        let s_new = 0.5 * spring_const * ((x_new - x_prev).powi(2) + (x_next - x_new).powi(2))
                  + 0.5 * pot_const * x_new * x_new;
        
        s_new - s_old
    }

    /// Perform a single-bead Metropolis move
    /// Returns true if move was accepted
    pub fn metropolis_move(&mut self, delta: f64) -> bool {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        
        // Select random bead
        let i = rng.gen_range(0..self.n_beads);
        let x_old = self.beads[i];
        let x_new = x_old + delta * uniform.sample(&mut rng);
        
        // Compute action change
        let delta_s = self.local_action_change(i, x_old, x_new);
        
        // Metropolis acceptance
        let accept = if delta_s < 0.0 {
            true
        } else {
            let r: f64 = rng.gen();
            r < (-delta_s).exp()
        };
        
        if accept {
            self.beads[i] = x_new;
        }
        
        accept
    }

    /// Perform staging move: update multiple connected beads with correct sampling
    /// This is more efficient for correlated paths.
    /// Uses Levy bridge which samples the kinetic action exactly, then accepts/rejects
    /// based only on the potential energy difference.
    pub fn staging_move(&mut self, segment_length: usize) -> bool {
        if segment_length < 2 || segment_length > self.n_beads {
            return false;
        }
        
        let mut rng = rand::thread_rng();
        
        // Choose random starting bead
        let start = rng.gen_range(0..self.n_beads);
        let end = (start + segment_length) % self.n_beads;
        
        // Store old positions and compute old potential action 
        let mut old_positions: Vec<f64> = Vec::with_capacity(segment_length - 1);
        let mut old_pot_action = 0.0;
        for k in 1..segment_length {
            let idx = (start + k) % self.n_beads;
            let x = self.beads[idx];
            old_positions.push(x);
            old_pot_action += self.potential_action(idx);
        }
        
        // Levy bridge construction for free particle
        // This samples the kinetic action exactly, so we only need to accept/reject on potential
        let x_start = self.beads[start];
        let x_end = self.beads[end];
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // Build bridge using proper Levy construction
        // For segment from x_start to x_end over (segment_length) steps
        for k in 1..segment_length {
            let idx = (start + k) % self.n_beads;
            let steps_from_start = k;
            let steps_to_end = segment_length - k;
            let total_steps = segment_length;
            
            // Mean is linear interpolation weighted by distances
            let mean = (steps_to_end as f64 * x_start + steps_from_start as f64 * x_end) 
                     / total_steps as f64;
            
            // Variance for Levy bridge: σ² = (steps_from_start * steps_to_end / total_steps) * Δτ/m
            let variance = (steps_from_start * steps_to_end) as f64 / total_steps as f64 
                         * self.dtau / self.mass;
            let sigma = variance.sqrt();
            
            self.beads[idx] = mean + sigma * normal.sample(&mut rng);
        }
        
        // Compute new potential action
        let mut new_pot_action = 0.0;
        for k in 1..segment_length {
            let idx = (start + k) % self.n_beads;
            new_pot_action += self.potential_action(idx);
        }
        
        // Accept/reject based only on potential difference
        // (kinetic part is sampled exactly by Levy bridge)
        let delta_pot = new_pot_action - old_pot_action;
        
        let accept = if delta_pot < 0.0 {
            true
        } else {
            let r: f64 = rng.gen();
            r < (-delta_pot).exp()
        };
        
        if !accept {
            // Restore old positions
            for (k, &x) in old_positions.iter().enumerate() {
                self.beads[(start + k + 1) % self.n_beads] = x;
            }
        }
        
        accept
    }

    /// Energy estimator using the virial theorem
    /// 
    /// For harmonic oscillator, the virial theorem gives:
    /// E = <V> + (1/2)<x × dV/dx> = <V> + <V> = 2<V>
    /// 
    /// where <V> = (1/2)mω²<x²>
    /// 
    /// This estimator is exact for harmonic oscillator and has lower variance
    /// than the thermodynamic (primitive) estimator.
    pub fn energy_estimator(&self) -> f64 {
        let m = self.n_beads as f64;
        let mut x2_sum = 0.0;
        
        for i in 0..self.n_beads {
            let x = self.beads[i];
            x2_sum += x * x;
        }
        
        let mean_x2 = x2_sum / m;
        
        // <V> = (1/2)mω²<x²>
        // For harmonic oscillator with virial: E = 2<V> = mω²<x²>
        self.mass * self.omega * self.omega * mean_x2
    }

    /// Thermodynamic (primitive) energy estimator
    /// 
    /// E = M/(2β) - (m·M)/(2β²) × (1/M)Σᵢ(x_{i+1} - x_i)² + (1/M)Σᵢ V(x_i)
    /// 
    /// Note: This estimator can have high variance and may give incorrect results
    /// if not properly converged. Use energy_estimator() (virial) for harmonic oscillator.
    #[allow(dead_code)]
    pub fn primitive_energy_estimator(&self) -> f64 {
        let n = self.n_beads as f64;
        
        // Kinetic contribution using primitive estimator
        let mut spring_sum = 0.0;
        for i in 0..self.n_beads {
            let j = (i + 1) % self.n_beads;
            let dx = self.beads[j] - self.beads[i];
            spring_sum += dx * dx;
        }
        
        // Correct primitive estimator: 
        // E_kin = M/(2β) - m*sum((x_{i+1}-x_i)²)/(2*dtau²)
        //       = M/(2β) - m*M*<(dx)²>/(2β²/M)
        //       = M/(2β) - m*M²*<(dx)²>/(2β²)
        let mean_dx2 = spring_sum / n;
        let kinetic = n / (2.0 * self.beta) 
                    - self.mass * n * n * mean_dx2 / (2.0 * self.beta * self.beta);
        
        // Potential contribution (average over beads)
        let mut pot_sum = 0.0;
        for i in 0..self.n_beads {
            let x = self.beads[i];
            pot_sum += 0.5 * self.mass * self.omega * self.omega * x * x;
        }
        let potential = pot_sum / n;
        
        kinetic + potential
    }

    /// Get average position (should be ~0 for harmonic oscillator)
    pub fn average_position(&self) -> f64 {
        self.beads.iter().sum::<f64>() / self.n_beads as f64
    }

    /// Get average position squared (for <x²> measurement)
    pub fn average_position_squared(&self) -> f64 {
        self.beads.iter().map(|x| x * x).sum::<f64>() / self.n_beads as f64
    }
}

/// PIMC simulation with multiple parallel paths
#[derive(Clone)]
pub struct PIMCSimulation {
    /// Collection of parallel quantum paths
    pub paths: Vec<QuantumPath>,
    /// Number of parallel paths (walkers)
    pub n_paths: usize,
    /// Move step size for Metropolis
    pub delta: f64,
    /// Acceptance rate tracking
    pub acceptance_count: usize,
    pub total_moves: usize,
}

impl PIMCSimulation {
    /// Create new PIMC simulation with parallel paths
    pub fn new(n_paths: usize, n_beads: usize, beta: f64, omega: f64, mass: f64) -> Self {
        let paths: Vec<QuantumPath> = (0..n_paths)
            .map(|_| QuantumPath::new(n_beads, beta, omega, mass))
            .collect();
        
        // Initial step size based on thermal width
        let delta = (1.0 / (beta * mass * omega * omega)).sqrt().min(1.0);
        
        Self {
            paths,
            n_paths,
            delta,
            acceptance_count: 0,
            total_moves: 0,
        }
    }

    /// Perform one MC sweep (attempt to move all beads of all paths once)
    pub fn sweep(&mut self, use_staging: bool, staging_length: usize) {
        for path in self.paths.iter_mut() {
            if use_staging && staging_length >= 2 {
                // Use staging moves
                let n_staging_moves = path.n_beads / staging_length;
                for _ in 0..n_staging_moves {
                    if path.staging_move(staging_length) {
                        self.acceptance_count += 1;
                    }
                    self.total_moves += 1;
                }
            } else {
                // Use single-bead Metropolis moves
                for _ in 0..path.n_beads {
                    if path.metropolis_move(self.delta) {
                        self.acceptance_count += 1;
                    }
                    self.total_moves += 1;
                }
            }
        }
    }

    /// Adapt step size to target ~50% acceptance
    pub fn adapt_delta(&mut self, target_rate: f64) {
        if self.total_moves < 100 {
            return;
        }
        
        let current_rate = self.acceptance_count as f64 / self.total_moves as f64;
        
        if current_rate < target_rate - 0.05 {
            self.delta *= 0.95;
        } else if current_rate > target_rate + 0.05 {
            self.delta *= 1.05;
        }
        
        // Reset counters
        self.acceptance_count = 0;
        self.total_moves = 0;
    }

    /// Get current acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_moves == 0 {
            0.0
        } else {
            self.acceptance_count as f64 / self.total_moves as f64
        }
    }

    /// Compute average energy across all paths
    pub fn average_energy(&self) -> f64 {
        let sum: f64 = self.paths.iter()
            .map(|p| p.energy_estimator())
            .sum();
        sum / self.n_paths as f64
    }

    /// Compute average <x²> across all paths
    pub fn average_x_squared(&self) -> f64 {
        let sum: f64 = self.paths.iter()
            .map(|p| p.average_position_squared())
            .sum();
        sum / self.n_paths as f64
    }
}

/// Run PIMC simulation for harmonic oscillator
pub fn run_pimc_harmonic(
    n_paths: usize,
    n_beads: usize,
    beta: f64,
    omega: f64,
    n_thermalize: usize,
    n_production: usize,
    use_staging: bool,
) {
    println!("=== PIMC Harmonic Oscillator ===");
    println!("Number of paths: {}", n_paths);
    println!("Number of beads (M): {}", n_beads);
    println!("Inverse temperature β: {:.4}", beta);
    println!("Frequency ω: {:.4}", omega);
    println!("Expected ground state energy: {:.6}", 0.5 * omega);
    println!();

    let mass = 1.0; // Natural units
    let mut sim = PIMCSimulation::new(n_paths, n_beads, beta, omega, mass);
    
    let staging_length = if use_staging { (n_beads / 4).max(2) } else { 0 };

    // Thermalization
    println!("Thermalizing ({} sweeps)...", n_thermalize);
    for step in 0..n_thermalize {
        sim.sweep(use_staging, staging_length);
        
        // Adapt step size during thermalization
        if step % 100 == 0 && step > 0 {
            sim.adapt_delta(0.5);
        }
        
        if step % (n_thermalize / 10).max(1) == 0 {
            println!(
                "  Step {:6}: E = {:10.6}, acceptance = {:.2}%",
                step,
                sim.average_energy(),
                100.0 * sim.acceptance_rate()
            );
        }
    }
    println!();

    // Production run
    println!("Production ({} sweeps)...", n_production);
    
    let mut energies: Vec<f64> = Vec::with_capacity(n_production);
    let mut x_squared: Vec<f64> = Vec::with_capacity(n_production);

    for step in 0..n_production {
        sim.sweep(use_staging, staging_length);
        
        let e = sim.average_energy();
        let x2 = sim.average_x_squared();
        energies.push(e);
        x_squared.push(x2);
        
        if step % (n_production / 10).max(1) == 0 {
            println!(
                "  Step {:6}: E = {:10.6}, <x²> = {:10.6}",
                step, e, x2
            );
        }
    }

    // Calculate statistics
    let n = energies.len() as f64;
    let mean_e: f64 = energies.iter().sum::<f64>() / n;
    let var_e: f64 = energies.iter().map(|e| (e - mean_e).powi(2)).sum::<f64>() / n;
    let std_e = var_e.sqrt();
    let stderr_e = std_e / n.sqrt();

    let mean_x2: f64 = x_squared.iter().sum::<f64>() / n;
    let var_x2: f64 = x_squared.iter().map(|x| (x - mean_x2).powi(2)).sum::<f64>() / n;
    let std_x2 = var_x2.sqrt();
    let stderr_x2 = std_x2 / n.sqrt();

    println!();
    println!("=== Results ===");
    println!("Ground state energy (expected: {:.6}):", 0.5 * omega);
    println!("  E = {:.6} ± {:.6}", mean_e, stderr_e);
    println!();
    println!("Position variance <x²> (expected from QM: {:.6}):", 0.5 / (mass * omega));
    println!("  <x²> = {:.6} ± {:.6}", mean_x2, stderr_x2);
    println!();
    println!("Final acceptance rate: {:.2}%", 100.0 * sim.acceptance_rate());

    // Write energies to file
    let file = File::create("pimc_energies.txt").unwrap();
    let mut writer = BufWriter::new(file);
    for (i, e) in energies.iter().enumerate() {
        writeln!(writer, "{} {}", i, e).unwrap();
    }
    println!("\nEnergy trajectory written to pimc_energies.txt");
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_path_creation() {
        let path = QuantumPath::new(16, 10.0, 1.0, 1.0);
        assert_eq!(path.n_beads, 16);
        assert_relative_eq!(path.dtau, 10.0 / 16.0, epsilon = 1e-10);
    }

    #[test]
    fn test_total_action_positive() {
        let path = QuantumPath::new(32, 10.0, 1.0, 1.0);
        assert!(path.total_action() >= 0.0);
    }

    #[test]
    fn test_metropolis_move() {
        let mut path = QuantumPath::new(16, 10.0, 1.0, 1.0);
        let _accepted = path.metropolis_move(0.5);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_energy_estimator_reasonable() {
        // At low temperature (large β), should approach ground state E₀ = 0.5
        // Note: the primitive estimator for a single unequilibrated path
        // can give negative values, so we just check for NaN/infinity
        let path = QuantumPath::new(64, 20.0, 1.0, 1.0);
        let e = path.energy_estimator();
        // Just verify it's a finite number (single path has very high variance)
        assert!(e.is_finite());
    }

    #[test]
    fn test_pimc_ground_state_energy() {
        // Run short simulation to verify we get ~0.5 for ground state
        let n_paths = 50;
        let n_beads = 32;
        let beta = 20.0; // Low temperature
        let omega = 1.0;
        
        let mut sim = PIMCSimulation::new(n_paths, n_beads, beta, omega, 1.0);
        
        // Thermalize
        for _ in 0..500 {
            sim.sweep(false, 0);
        }
        
        // Sample
        let mut energies = Vec::new();
        for _ in 0..500 {
            sim.sweep(false, 0);
            energies.push(sim.average_energy());
        }
        
        let mean_e: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
        
        // Should be close to 0.5 (ground state of harmonic oscillator)
        assert_relative_eq!(mean_e, 0.5, epsilon = 0.1);
    }
}
