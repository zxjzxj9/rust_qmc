//! Fixed-Node Path Integral Monte Carlo for Fermions
//!
//! Uses the fixed-node approximation to avoid the fermion sign problem:
//! paths that would cross the nodal surface of the trial wavefunction are rejected.
//!
//! Reference: Ceperley, D.M. (1996) "Path integral Monte Carlo methods for fermions"
//! in Monte Carlo and Molecular Dynamics of Condensed Matter Systems

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::fs::File;
use std::io::{BufWriter, Write};

// =============================================================================
// Trial Wavefunction Trait
// =============================================================================

/// Trait for trial wavefunctions that define the nodal surface for fixed-node PIMC
pub trait TrialWavefunction: Clone + Send + Sync {
    /// Evaluate the wavefunction at electron configuration
    fn evaluate(&self, positions: &[Vector3<f64>]) -> f64;
    
    /// Return sign of wavefunction: +1 or -1 (0 if exactly on node)
    fn sign(&self, positions: &[Vector3<f64>]) -> i32 {
        let psi = self.evaluate(positions);
        if psi > 0.0 { 1 } else if psi < 0.0 { -1 } else { 0 }
    }
    
    /// Compute local energy: E_L = HΨ/Ψ for mixed estimator
    /// Default implementation uses numerical derivatives
    fn local_energy(&self, positions: &[Vector3<f64>], nuclear_positions: &[Vector3<f64>], nuclear_charges: &[f64]) -> f64;
    
    /// Number of electrons this wavefunction describes
    fn n_electrons(&self) -> usize;
}

// =============================================================================
// Simple Hydrogen 1s Wavefunction (for testing)
// =============================================================================

/// Simple hydrogen 1s trial wavefunction: Ψ = exp(-α|r - R_nuc|)
#[derive(Clone)]
pub struct Hydrogen1s {
    pub alpha: f64,
    pub nucleus: Vector3<f64>,
}

impl Hydrogen1s {
    pub fn new(alpha: f64, nucleus: Vector3<f64>) -> Self {
        Self { alpha, nucleus }
    }
}

impl TrialWavefunction for Hydrogen1s {
    fn evaluate(&self, positions: &[Vector3<f64>]) -> f64 {
        let r = (positions[0] - self.nucleus).norm();
        (-self.alpha * r).exp()
    }
    
    fn sign(&self, _positions: &[Vector3<f64>]) -> i32 {
        1  // 1s is always positive - no nodes!
    }
    
    fn local_energy(&self, positions: &[Vector3<f64>], nuclear_positions: &[Vector3<f64>], nuclear_charges: &[f64]) -> f64 {
        let r_vec = positions[0] - self.nucleus;
        let r = r_vec.norm();
        
        if r < 1e-10 {
            return 0.0; // Avoid singularity
        }
        
        // For Ψ = exp(-αr):
        // ∇²Ψ/Ψ = α² - 2α/r
        // K.E. = -½∇²Ψ/Ψ = -α²/2 + α/r
        let kinetic = -0.5 * self.alpha * self.alpha + self.alpha / r;
        
        // Coulomb potential: -Z/r for each nucleus
        let mut potential = 0.0;
        for (nuc_pos, &z) in nuclear_positions.iter().zip(nuclear_charges.iter()) {
            let r_nuc = (positions[0] - nuc_pos).norm();
            if r_nuc > 1e-10 {
                potential -= z / r_nuc;
            }
        }
        
        kinetic + potential
    }
    
    fn n_electrons(&self) -> usize {
        1
    }
}

// =============================================================================
// Fermion Path (3D multi-particle with nodal constraint)
// =============================================================================

/// 3D multi-particle path for fermion PIMC with fixed-node constraint
#[derive(Clone)]
pub struct FermionPath<T: TrialWavefunction> {
    /// Beads: beads[i] = positions of all electrons at imaginary time slice i
    /// beads[i][j] = Vector3 position of electron j at time slice i
    pub beads: Vec<Vec<Vector3<f64>>>,
    /// Number of Trotter time slices
    pub n_beads: usize,
    /// Number of electrons
    pub n_electrons: usize,
    /// Imaginary time step Δτ = β/M
    pub dtau: f64,
    /// Inverse temperature
    pub beta: f64,
    /// Particle mass
    pub mass: f64,
    /// Trial wavefunction defining nodal surface
    pub trial_wfn: T,
    /// Sign of trial wavefunction at τ=0 (reference sign)
    pub reference_sign: i32,
    /// Nuclear positions (for Coulomb)
    pub nuclear_positions: Vec<Vector3<f64>>,
    /// Nuclear charges
    pub nuclear_charges: Vec<f64>,
}

impl<T: TrialWavefunction> FermionPath<T> {
    /// Create new fermion path
    pub fn new(
        n_beads: usize,
        beta: f64,
        mass: f64,
        trial_wfn: T,
        nuclear_positions: Vec<Vector3<f64>>,
        nuclear_charges: Vec<f64>,
    ) -> Self {
        let dtau = beta / n_beads as f64;
        let n_electrons = trial_wfn.n_electrons();
        
        // Initialize beads near the nucleus with Gaussian distribution
        let mut rng = rand::thread_rng();
        let sigma = 1.0 / mass.sqrt(); // Thermal width
        let dist = Normal::new(0.0, sigma).unwrap();
        
        // Start near first nucleus
        let center = if nuclear_positions.is_empty() {
            Vector3::zeros()
        } else {
            nuclear_positions[0]
        };
        
        let mut beads = Vec::with_capacity(n_beads);
        for _ in 0..n_beads {
            let mut electron_positions = Vec::with_capacity(n_electrons);
            for _ in 0..n_electrons {
                let pos = center + Vector3::new(
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                );
                electron_positions.push(pos);
            }
            beads.push(electron_positions);
        }
        
        // Compute reference sign from first bead
        let reference_sign = trial_wfn.sign(&beads[0]);
        
        Self {
            beads,
            n_beads,
            n_electrons,
            dtau,
            beta,
            mass,
            trial_wfn,
            reference_sign,
            nuclear_positions,
            nuclear_charges,
        }
    }
    
    /// Compute kinetic (spring) action contribution for electron e between beads i and i+1
    #[inline]
    fn kinetic_action_single(&self, i: usize, e: usize) -> f64 {
        let j = (i + 1) % self.n_beads;  // PBC in imaginary time
        let dr = self.beads[j][e] - self.beads[i][e];
        0.5 * self.mass / self.dtau * dr.norm_squared()
    }
    
    /// Compute Coulomb action at bead i (primitive approximation)
    fn coulomb_action(&self, i: usize) -> f64 {
        let positions = &self.beads[i];
        let mut action = 0.0;
        
        // Electron-nucleus attraction
        for e in 0..self.n_electrons {
            for (nuc_pos, &z) in self.nuclear_positions.iter().zip(self.nuclear_charges.iter()) {
                let r = (positions[e] - nuc_pos).norm();
                if r > 1e-10 {
                    action -= self.dtau * z / r;
                }
            }
        }
        
        // Electron-electron repulsion
        for e1 in 0..self.n_electrons {
            for e2 in (e1 + 1)..self.n_electrons {
                let r12 = (positions[e1] - positions[e2]).norm();
                if r12 > 1e-10 {
                    action += self.dtau / r12;
                }
            }
        }
        
        action
    }
    
    /// Compute total action of the path
    pub fn total_action(&self) -> f64 {
        let mut action = 0.0;
        
        for i in 0..self.n_beads {
            // Kinetic (springs)
            for e in 0..self.n_electrons {
                action += self.kinetic_action_single(i, e);
            }
            // Coulomb
            action += self.coulomb_action(i);
        }
        
        action
    }
    
    /// Compute change in action when moving one electron at one bead
    fn local_action_change(&self, bead_idx: usize, elec_idx: usize, old_pos: Vector3<f64>, new_pos: Vector3<f64>) -> f64 {
        let prev = (bead_idx + self.n_beads - 1) % self.n_beads;
        let next = (bead_idx + 1) % self.n_beads;
        
        let spring_const = self.mass / self.dtau;
        
        // Old spring contributions
        let dr_prev_old = old_pos - self.beads[prev][elec_idx];
        let dr_next_old = self.beads[next][elec_idx] - old_pos;
        let s_kin_old = 0.5 * spring_const * (dr_prev_old.norm_squared() + dr_next_old.norm_squared());
        
        // New spring contributions
        let dr_prev_new = new_pos - self.beads[prev][elec_idx];
        let dr_next_new = self.beads[next][elec_idx] - new_pos;
        let s_kin_new = 0.5 * spring_const * (dr_prev_new.norm_squared() + dr_next_new.norm_squared());
        
        // Old Coulomb
        let mut s_coul_old = 0.0;
        // Electron-nucleus
        for (nuc_pos, &z) in self.nuclear_positions.iter().zip(self.nuclear_charges.iter()) {
            let r = (old_pos - nuc_pos).norm();
            if r > 1e-10 {
                s_coul_old -= self.dtau * z / r;
            }
        }
        // Electron-electron
        for e2 in 0..self.n_electrons {
            if e2 != elec_idx {
                let r12 = (old_pos - self.beads[bead_idx][e2]).norm();
                if r12 > 1e-10 {
                    s_coul_old += self.dtau / r12;
                }
            }
        }
        
        // New Coulomb
        let mut s_coul_new = 0.0;
        for (nuc_pos, &z) in self.nuclear_positions.iter().zip(self.nuclear_charges.iter()) {
            let r = (new_pos - nuc_pos).norm();
            if r > 1e-10 {
                s_coul_new -= self.dtau * z / r;
            }
        }
        for e2 in 0..self.n_electrons {
            if e2 != elec_idx {
                let r12 = (new_pos - self.beads[bead_idx][e2]).norm();
                if r12 > 1e-10 {
                    s_coul_new += self.dtau / r12;
                }
            }
        }
        
        (s_kin_new + s_coul_new) - (s_kin_old + s_coul_old)
    }
    
    /// Perform Metropolis move with nodal constraint
    /// Returns true if accepted
    pub fn metropolis_move(&mut self, delta: f64) -> bool {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        
        // Pick random bead and electron
        let bead_idx = rng.gen_range(0..self.n_beads);
        let elec_idx = rng.gen_range(0..self.n_electrons);
        
        let old_pos = self.beads[bead_idx][elec_idx];
        let displacement = Vector3::new(
            delta * uniform.sample(&mut rng),
            delta * uniform.sample(&mut rng),
            delta * uniform.sample(&mut rng),
        );
        let new_pos = old_pos + displacement;
        
        // FIXED-NODE CONSTRAINT: Check if move crosses nodal surface
        let mut test_config = self.beads[bead_idx].clone();
        test_config[elec_idx] = new_pos;
        
        let new_sign = self.trial_wfn.sign(&test_config);
        if new_sign != self.reference_sign {
            // Would cross nodal surface - REJECT
            return false;
        }
        
        // Compute action change
        let delta_s = self.local_action_change(bead_idx, elec_idx, old_pos, new_pos);
        
        // Metropolis acceptance
        let accept = if delta_s < 0.0 {
            true
        } else {
            rng.gen::<f64>() < (-delta_s).exp()
        };
        
        if accept {
            self.beads[bead_idx][elec_idx] = new_pos;
        }
        
        accept
    }
    
    /// Mixed energy estimator using trial wavefunction
    /// E_mixed = (1/M) Σᵢ E_L(Rᵢ) where E_L = HΨ_T/Ψ_T
    pub fn mixed_energy_estimator(&self) -> f64 {
        let mut energy_sum = 0.0;
        
        for i in 0..self.n_beads {
            energy_sum += self.trial_wfn.local_energy(
                &self.beads[i],
                &self.nuclear_positions,
                &self.nuclear_charges,
            );
        }
        
        energy_sum / self.n_beads as f64
    }
    
    /// Thermodynamic energy estimator (primitive)
    pub fn primitive_energy_estimator(&self) -> f64 {
        let n = self.n_beads as f64;
        
        // Kinetic from spring terms
        let mut spring_sum = 0.0;
        for i in 0..self.n_beads {
            let j = (i + 1) % self.n_beads;
            for e in 0..self.n_electrons {
                let dr = self.beads[j][e] - self.beads[i][e];
                spring_sum += dr.norm_squared();
            }
        }
        let mean_dr2 = spring_sum / n;
        
        // 3D kinetic: 3N/(2β) - (m·M²/2β²)·<dr²>
        let kinetic = 3.0 * self.n_electrons as f64 / (2.0 * self.beta)
                    - self.mass * n * n * mean_dr2 / (2.0 * self.beta * self.beta);
        
        // Potential average
        let mut pot_sum = 0.0;
        for i in 0..self.n_beads {
            let positions = &self.beads[i];
            
            // Electron-nucleus
            for e in 0..self.n_electrons {
                for (nuc_pos, &z) in self.nuclear_positions.iter().zip(self.nuclear_charges.iter()) {
                    let r = (positions[e] - nuc_pos).norm();
                    if r > 1e-10 {
                        pot_sum -= z / r;
                    }
                }
            }
            
            // Electron-electron
            for e1 in 0..self.n_electrons {
                for e2 in (e1 + 1)..self.n_electrons {
                    let r12 = (positions[e1] - positions[e2]).norm();
                    if r12 > 1e-10 {
                        pot_sum += 1.0 / r12;
                    }
                }
            }
        }
        
        kinetic + pot_sum / n
    }
    
    /// Average distance from nucleus (for <r> measurement)
    pub fn average_r(&self) -> f64 {
        let mut r_sum = 0.0;
        let nuc = if self.nuclear_positions.is_empty() {
            Vector3::zeros()
        } else {
            self.nuclear_positions[0]
        };
        
        for i in 0..self.n_beads {
            for e in 0..self.n_electrons {
                r_sum += (self.beads[i][e] - nuc).norm();
            }
        }
        
        r_sum / (self.n_beads * self.n_electrons) as f64
    }
}

// =============================================================================
// Fermion PIMC Simulation
// =============================================================================

/// Fixed-Node PIMC simulation for fermions
pub struct FermionPIMC<T: TrialWavefunction> {
    pub paths: Vec<FermionPath<T>>,
    pub n_paths: usize,
    pub delta: f64,
    pub acceptance_count: usize,
    pub total_moves: usize,
    pub nodal_rejections: usize,
}

impl<T: TrialWavefunction> FermionPIMC<T> {
    pub fn new(
        n_paths: usize,
        n_beads: usize,
        beta: f64,
        mass: f64,
        trial_wfn: T,
        nuclear_positions: Vec<Vector3<f64>>,
        nuclear_charges: Vec<f64>,
    ) -> Self {
        let paths: Vec<FermionPath<T>> = (0..n_paths)
            .map(|_| FermionPath::new(
                n_beads, beta, mass, 
                trial_wfn.clone(),
                nuclear_positions.clone(),
                nuclear_charges.clone(),
            ))
            .collect();
        
        Self {
            paths,
            n_paths,
            delta: 0.5,
            acceptance_count: 0,
            total_moves: 0,
            nodal_rejections: 0,
        }
    }
    
    pub fn sweep(&mut self) {
        for path in self.paths.iter_mut() {
            let n_moves = path.n_beads * path.n_electrons;
            for _ in 0..n_moves {
                if path.metropolis_move(self.delta) {
                    self.acceptance_count += 1;
                }
                self.total_moves += 1;
            }
        }
    }
    
    pub fn adapt_delta(&mut self, target_rate: f64) {
        if self.total_moves < 100 {
            return;
        }
        let current_rate = self.acceptance_count as f64 / self.total_moves as f64;
        
        // Stronger adjustment for better convergence
        if current_rate < target_rate - 0.1 {
            self.delta *= 0.9;
        } else if current_rate < target_rate - 0.05 {
            self.delta *= 0.95;
        } else if current_rate > target_rate + 0.1 {
            self.delta *= 1.1;
        } else if current_rate > target_rate + 0.05 {
            self.delta *= 1.05;
        }
        
        // Bound delta to prevent collapse or explosion
        self.delta = self.delta.max(0.05).min(3.0);
        
        self.acceptance_count = 0;
        self.total_moves = 0;
    }
    
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_moves == 0 { 0.0 }
        else { self.acceptance_count as f64 / self.total_moves as f64 }
    }
    
    pub fn average_energy_mixed(&self) -> f64 {
        self.paths.iter().map(|p| p.mixed_energy_estimator()).sum::<f64>() / self.n_paths as f64
    }
    
    pub fn average_energy_primitive(&self) -> f64 {
        self.paths.iter().map(|p| p.primitive_energy_estimator()).sum::<f64>() / self.n_paths as f64
    }
    
    pub fn average_r(&self) -> f64 {
        self.paths.iter().map(|p| p.average_r()).sum::<f64>() / self.n_paths as f64
    }
}

// =============================================================================
// Simulation Driver
// =============================================================================

/// Run fixed-node PIMC for hydrogen atom
pub fn run_pimc_hydrogen(
    n_paths: usize,
    n_beads: usize,
    beta: f64,
    alpha: f64,
    n_thermalize: usize,
    n_production: usize,
) {
    let nucleus = Vector3::zeros();
    let trial_wfn = Hydrogen1s::new(alpha, nucleus);
    let nuclear_positions = vec![nucleus];
    let nuclear_charges = vec![1.0];  // Z = 1 for hydrogen
    
    println!("=== Fixed-Node PIMC: Hydrogen Atom ===");
    println!("Number of paths: {}", n_paths);
    println!("Number of beads (M): {}", n_beads);
    println!("Inverse temperature β: {:.4}", beta);
    println!("Trial wavefunction: Ψ = exp(-{:.4}r)", alpha);
    println!("Expected ground state: E₀ = -0.5 Hartree");
    println!();

    let mass = 1.0;  // Atomic units
    let mut sim = FermionPIMC::new(
        n_paths, n_beads, beta, mass,
        trial_wfn, nuclear_positions, nuclear_charges
    );

    // Thermalization
    println!("Thermalizing ({} sweeps)...", n_thermalize);
    for step in 0..n_thermalize {
        sim.sweep();
        if step % 100 == 0 && step > 0 {
            sim.adapt_delta(0.4);
        }
        if step % (n_thermalize / 5).max(1) == 0 {
            println!(
                "  Step {:6}: E_mixed = {:10.6}, <r> = {:8.4}, accept = {:.2}%",
                step, sim.average_energy_mixed(), sim.average_r(), 100.0 * sim.acceptance_rate()
            );
        }
    }
    println!();

    // Production
    println!("Production ({} sweeps)...", n_production);
    let mut energies_mixed = Vec::with_capacity(n_production);
    let mut energies_prim = Vec::with_capacity(n_production);
    let mut r_values = Vec::with_capacity(n_production);

    for step in 0..n_production {
        sim.sweep();
        
        let e_mix = sim.average_energy_mixed();
        let e_prim = sim.average_energy_primitive();
        let r_avg = sim.average_r();
        
        energies_mixed.push(e_mix);
        energies_prim.push(e_prim);
        r_values.push(r_avg);
        
        if step % (n_production / 10).max(1) == 0 {
            println!(
                "  Step {:6}: E_mixed = {:10.6}, E_prim = {:10.6}, <r> = {:8.4}",
                step, e_mix, e_prim, r_avg
            );
        }
    }

    // Statistics
    let n = energies_mixed.len() as f64;
    let mean_e_mix = energies_mixed.iter().sum::<f64>() / n;
    let var_e_mix = energies_mixed.iter().map(|e| (e - mean_e_mix).powi(2)).sum::<f64>() / n;
    let stderr_e_mix = var_e_mix.sqrt() / n.sqrt();
    
    let mean_e_prim = energies_prim.iter().sum::<f64>() / n;
    let mean_r = r_values.iter().sum::<f64>() / n;

    println!();
    println!("=== Results ===");
    println!("Mixed estimator (expected: -0.5 Hartree):");
    println!("  E_mixed = {:.6} ± {:.6} Hartree", mean_e_mix, stderr_e_mix);
    println!("Primitive estimator:");
    println!("  E_prim  = {:.6} Hartree", mean_e_prim);
    println!();
    println!("Average radius (expected: 1.5 a₀):");
    println!("  <r> = {:.4} a₀", mean_r);
    println!();
    println!("Acceptance rate: {:.2}%", 100.0 * sim.acceptance_rate());

    // Write output
    let file = File::create("hydrogen_pimc_energies.txt").unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# step E_mixed E_primitive r").unwrap();
    for (i, ((e_mix, e_prim), r)) in energies_mixed.iter().zip(energies_prim.iter()).zip(r_values.iter()).enumerate() {
        writeln!(writer, "{} {:.6} {:.6} {:.6}", i, e_mix, e_prim, r).unwrap();
    }
    println!("\nData written to hydrogen_pimc_energies.txt");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydrogen_1s_sign() {
        let wfn = Hydrogen1s::new(1.0, Vector3::zeros());
        let pos = vec![Vector3::new(1.0, 0.0, 0.0)];
        assert_eq!(wfn.sign(&pos), 1);  // 1s is always positive
    }
    
    #[test]
    fn test_hydrogen_local_energy_at_optimal() {
        // For hydrogen with α = 1.0, local energy should be -0.5 everywhere
        let wfn = Hydrogen1s::new(1.0, Vector3::zeros());
        let pos = vec![Vector3::new(1.0, 0.0, 0.0)];
        let nuc = vec![Vector3::zeros()];
        let charges = vec![1.0];
        
        let e_local = wfn.local_energy(&pos, &nuc, &charges);
        assert!((e_local + 0.5).abs() < 0.01);  // Should be -0.5
    }
    
    #[test]
    fn test_fermion_path_creation() {
        let wfn = Hydrogen1s::new(1.0, Vector3::zeros());
        let path = FermionPath::new(
            16, 10.0, 1.0, wfn,
            vec![Vector3::zeros()],
            vec![1.0],
        );
        assert_eq!(path.n_beads, 16);
        assert_eq!(path.n_electrons, 1);
    }
}
