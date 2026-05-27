//! Ring Polymer Contraction (RPC) for PIMD
//!
//! Evaluates expensive potential energy forces on a reduced number of
//! "contracted" beads P' < P, while keeping the full ring polymer with
//! P beads for the cheap/fast forces and spring interactions.
//!
//! The contraction projects the full ring polymer onto its lowest P'
//! normal modes, constructs P' contracted bead positions, evaluates
//! forces there, and expands the forces back to all P beads.
//!
//! Key equations:
//!   Contracted positions: R' = T · R   (T is P' × P contraction matrix)
//!   Expanded forces:      F  = T^T · F'  (T^T is P × P' expansion matrix)
//!
//! The contraction matrix is built from the normal mode transform:
//!   T[j'][i] = Σ_{k=0}^{P'-1} C_{P'}[k][j'] · C_P[k][i]
//! where C_P is the P-bead normal mode transform matrix.
//!
//! Reference: Markland & Manolopoulos, JCP 129, 024105 (2008)

use super::pimc::Potential;
use super::pimd::{NormalModeTransform, PILEThermostat};
use super::pimd_molecular::MolecularPotential;

use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::f64::consts::PI;

// =============================================================================
// Normal Mode Transform Matrix Element
// =============================================================================

/// Compute normal mode transform matrix element C[k][i] for P beads.
///
/// Uses the same convention as `NormalModeTransform::to_normal_modes`:
/// - k=0: centroid, C[0][i] = 1/√P
/// - k < P/2: cos mode, C[k][i] = √(2/P) cos(2πki/P)
/// - k = P/2 (P even): C[P/2][i] = cos(πi) / √P = (-1)^i / √P
/// - k > P/2: sin mode, C[k][i] = √(2/P) sin(2πki/P)
fn nm_element(k: usize, i: usize, p: usize) -> f64 {
    let pf = p as f64;
    let angle = 2.0 * PI * k as f64 * i as f64 / pf;
    if k == 0 {
        1.0 / pf.sqrt()
    } else if k == p / 2 && p % 2 == 0 {
        angle.cos() / pf.sqrt()
    } else if k <= p / 2 {
        angle.cos() * (2.0 / pf).sqrt()
    } else {
        angle.sin() * (2.0 / pf).sqrt()
    }
}

// =============================================================================
// Ring Polymer Contraction Engine
// =============================================================================

/// Ring Polymer Contraction (RPC) engine.
///
/// Projects P bead positions to P' < P contracted beads via the lowest
/// normal modes, enabling expensive forces to be evaluated on fewer beads.
///
/// The contraction keeps the lowest P' normal modes and discards higher ones.
/// Forces computed on contracted beads are expanded back to all P beads
/// using the transpose of the contraction matrix.
///
/// Special case: P' = 1 is "centroid contraction" where all beads get the
/// same force evaluated at the centroid position.
///
/// Reference: Markland & Manolopoulos, JCP 129, 024105 (2008)
#[derive(Clone)]
pub struct RPContraction {
    /// Full bead count P
    pub n_full: usize,
    /// Contracted bead count P'
    pub n_contracted: usize,
    /// Contraction matrix T: P' × P
    /// T[j'][i] maps full bead i to contracted bead j'
    contraction_matrix: Vec<Vec<f64>>,
    /// Expansion matrix T^T: P × P'
    /// expansion[i][j'] maps contracted bead j' force to full bead i
    expansion_matrix: Vec<Vec<f64>>,
}

impl RPContraction {
    /// Create a new ring polymer contraction.
    ///
    /// # Arguments
    /// * `n_full` - Full number of beads P
    /// * `n_contracted` - Contracted number of beads P' (must be ≤ P, ≥ 1)
    /// * `_beta` - Physical inverse temperature (for consistency; not used in matrix construction)
    pub fn new(n_full: usize, n_contracted: usize, _beta: f64) -> Self {
        assert!(n_contracted >= 1, "Need at least 1 contracted bead");
        assert!(n_contracted <= n_full, "P' must be ≤ P");

        // Build contraction matrix T: P' × P
        // T[j'][i] = Σ_{k=0}^{P'-1} C_{P'}^T[j'][k] · C_P[k][i]
        // Since normal mode matrix is orthogonal: C^T = C^{-1}
        // So C_{P'}^T[j'][k] = C_{P'}[k][j'] (the k-th mode evaluated at bead j')
        let mut contraction = vec![vec![0.0; n_full]; n_contracted];

        for jp in 0..n_contracted {
            for i in 0..n_full {
                let mut sum = 0.0;
                for k in 0..n_contracted {
                    let c_pc = nm_element(k, jp, n_contracted);
                    let c_pf = nm_element(k, i, n_full);
                    sum += c_pc * c_pf;
                }
                contraction[jp][i] = sum;
            }
        }

        // Expansion = T^T: P × P'
        let mut expansion = vec![vec![0.0; n_contracted]; n_full];
        for i in 0..n_full {
            for jp in 0..n_contracted {
                expansion[i][jp] = contraction[jp][i];
            }
        }

        Self {
            n_full,
            n_contracted,
            contraction_matrix: contraction,
            expansion_matrix: expansion,
        }
    }

    /// Check if this is a centroid contraction (P' = 1).
    pub fn is_centroid_contraction(&self) -> bool {
        self.n_contracted == 1
    }

    /// Contract P bead positions → P' contracted positions (1D).
    pub fn contract(&self, full: &[f64]) -> Vec<f64> {
        let mut contracted = vec![0.0; self.n_contracted];
        for jp in 0..self.n_contracted {
            for i in 0..self.n_full {
                contracted[jp] += self.contraction_matrix[jp][i] * full[i];
            }
        }
        contracted
    }

    /// Expand P' contracted forces → P full-bead forces (1D).
    pub fn expand(&self, contracted_forces: &[f64]) -> Vec<f64> {
        let mut expanded = vec![0.0; self.n_full];
        for i in 0..self.n_full {
            for jp in 0..self.n_contracted {
                expanded[i] += self.expansion_matrix[i][jp] * contracted_forces[jp];
            }
        }
        expanded
    }

    /// Contract molecular positions: positions[bead][dof] → contracted[bead'][dof].
    pub fn contract_molecular(&self, positions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let ndof = positions[0].len();
        let mut contracted = vec![vec![0.0; ndof]; self.n_contracted];
        for d in 0..ndof {
            for jp in 0..self.n_contracted {
                for i in 0..self.n_full {
                    contracted[jp][d] += self.contraction_matrix[jp][i] * positions[i][d];
                }
            }
        }
        contracted
    }

    /// Expand molecular forces: forces[bead'][dof] → expanded[bead][dof].
    pub fn expand_molecular(&self, contracted_forces: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let ndof = contracted_forces[0].len();
        let mut expanded = vec![vec![0.0; ndof]; self.n_full];
        for d in 0..ndof {
            for i in 0..self.n_full {
                for jp in 0..self.n_contracted {
                    expanded[i][d] += self.expansion_matrix[i][jp] * contracted_forces[jp][d];
                }
            }
        }
        expanded
    }

    /// Fast centroid contraction: compute centroid (P'=1 special case).
    pub fn centroid_contract(&self, full: &[f64]) -> f64 {
        full.iter().sum::<f64>() / self.n_full as f64
    }
}

// =============================================================================
// Splittable Potential Traits
// =============================================================================

/// A 1D potential that can be split into fast (cheap) and slow (expensive) parts.
///
/// V(x) = V_fast(x) + V_slow(x)
///
/// Fast forces are evaluated on all P beads.
/// Slow forces are evaluated on P' contracted beads and expanded back.
pub trait SplittablePotential: Potential {
    /// Fast (cheap) potential energy component
    fn energy_fast(&self, x: f64) -> f64;
    /// Slow (expensive) potential energy component
    fn energy_slow(&self, x: f64) -> f64;
    /// Fast force: F_fast = -dV_fast/dx
    fn force_fast(&self, x: f64) -> f64;
    /// Slow force: F_slow = -dV_slow/dx
    fn force_slow(&self, x: f64) -> f64;
}

/// A molecular potential that can be split into fast and slow parts.
pub trait SplittableMolecularPotential: MolecularPotential {
    /// Fast potential energy component
    fn energy_fast(&self, coords: &[f64]) -> f64;
    /// Slow potential energy component
    fn energy_slow(&self, coords: &[f64]) -> f64;
    /// Fast forces
    fn forces_fast(&self, coords: &[f64], forces: &mut [f64]);
    /// Slow forces
    fn forces_slow(&self, coords: &[f64], forces: &mut [f64]);
}

// =============================================================================
// Example: Splittable Double Well
// =============================================================================

/// A double-well potential with harmonic (fast) + barrier (slow) splitting.
///
/// V(x) = V_fast(x) + V_slow(x) where:
///   V_fast(x) = -½ k x² + ¼ k x⁴/x₀²  (dominant harmonic + quartic wells)
///   V_slow(x) = barrier_height · exp(-x²/(2σ²))  (barrier, "expensive")
///
/// This splitting lets the cheap harmonic part be evaluated on all P beads
/// while the barrier is evaluated on P' contracted beads.
#[derive(Clone)]
pub struct SplittableDoubleWell {
    /// Well stiffness parameter
    pub k: f64,
    /// Well position parameter x₀
    pub x0: f64,
    /// Barrier height
    pub barrier_height: f64,
    /// Barrier width (Gaussian σ)
    pub barrier_sigma: f64,
}

impl SplittableDoubleWell {
    /// Create with default parameters.
    pub fn new(barrier_height: f64, well_distance: f64) -> Self {
        Self {
            k: 4.0 * barrier_height / (well_distance * well_distance),
            x0: well_distance,
            barrier_height,
            barrier_sigma: well_distance * 0.5,
        }
    }
}

impl Potential for SplittableDoubleWell {
    fn evaluate(&self, x: f64) -> f64 {
        self.energy_fast(x) + self.energy_slow(x)
    }

    fn force(&self, x: f64) -> f64 {
        self.force_fast(x) + self.force_slow(x)
    }

    fn name(&self) -> &'static str {
        "SplittableDoubleWell"
    }

    fn init_width(&self) -> f64 {
        self.x0
    }
}

impl SplittablePotential for SplittableDoubleWell {
    fn energy_fast(&self, x: f64) -> f64 {
        // Double well: -½kx² + ¼k x⁴/x₀²
        let x2 = x * x;
        -0.5 * self.k * x2 + 0.25 * self.k * x2 * x2 / (self.x0 * self.x0)
    }

    fn energy_slow(&self, x: f64) -> f64 {
        // Gaussian barrier
        self.barrier_height * (-x * x / (2.0 * self.barrier_sigma * self.barrier_sigma)).exp()
    }

    fn force_fast(&self, x: f64) -> f64 {
        // -dV_fast/dx = kx - k x³/x₀²
        self.k * x - self.k * x * x * x / (self.x0 * self.x0)
    }

    fn force_slow(&self, x: f64) -> f64 {
        // -dV_slow/dx = barrier * x/(σ²) exp(-x²/(2σ²))
        let s2 = self.barrier_sigma * self.barrier_sigma;
        self.barrier_height * x / s2 * (-x * x / (2.0 * s2)).exp()
    }
}

// =============================================================================
// RPC Ring Polymer (1D)
// =============================================================================

/// Ring polymer with Ring Polymer Contraction for split potentials.
///
/// Evaluates fast forces on all P beads and slow forces on P' contracted beads.
#[derive(Clone)]
pub struct RPCRingPolymer<P: SplittablePotential> {
    /// Bead positions
    pub positions: Vec<f64>,
    /// Bead velocities
    pub velocities: Vec<f64>,
    /// Forces on each bead
    pub forces: Vec<f64>,
    /// Number of beads P
    pub n_beads: usize,
    /// Particle mass
    pub mass: f64,
    /// Physical inverse temperature beta
    pub beta: f64,
    /// Imaginary time step dτ = beta/P
    pub dtau: f64,
    /// Spring constant κ = m/dτ²
    pub spring_constant: f64,
    /// The splittable potential
    pub potential: P,
    /// Ring polymer contraction engine
    pub contraction: RPContraction,
    /// Number of contracted beads (stored for convenience)
    pub n_contracted: usize,
}

impl<P: SplittablePotential> RPCRingPolymer<P> {
    /// Create a new RPC ring polymer.
    pub fn new(
        n_beads: usize,
        n_contracted: usize,
        beta: f64,
        mass: f64,
        potential: P,
    ) -> Self {
        let dtau = beta / n_beads as f64;
        let spring_constant = mass / (dtau * dtau);
        let contraction = RPContraction::new(n_beads, n_contracted, beta);

        let mut rng = rand::thread_rng();
        let sigma_x = potential.init_width();
        let sigma_v = (1.0 / (beta * mass)).sqrt();
        let pos_dist = Normal::new(0.0, sigma_x).unwrap();
        let vel_dist = Normal::new(0.0, sigma_v).unwrap();

        let x0 = sigma_x;
        let positions: Vec<f64> = (0..n_beads)
            .map(|_| x0 + 0.1 * pos_dist.sample(&mut rng))
            .collect();
        let velocities: Vec<f64> = (0..n_beads)
            .map(|_| vel_dist.sample(&mut rng))
            .collect();

        let mut rp = Self {
            positions,
            velocities,
            forces: vec![0.0; n_beads],
            n_beads,
            mass,
            beta,
            dtau,
            spring_constant,
            potential,
            contraction,
            n_contracted,
        };
        rp.compute_forces();
        rp
    }

    /// Compute forces with RPC:
    /// F[i] = F_spring[i] + F_fast(x_i) + expand(F_slow(contract(x)))[i]
    pub fn compute_forces(&mut self) {
        // Spring forces + fast forces on all P beads
        for i in 0..self.n_beads {
            let prev = if i == 0 { self.n_beads - 1 } else { i - 1 };
            let next = (i + 1) % self.n_beads;

            let f_spring = -self.spring_constant
                * (2.0 * self.positions[i] - self.positions[prev] - self.positions[next]);

            let f_fast = self.potential.force_fast(self.positions[i]);

            self.forces[i] = f_spring + f_fast;
        }

        // Slow forces on contracted beads
        if self.contraction.is_centroid_contraction() {
            // Fast path: centroid contraction
            let centroid = self.positions.iter().sum::<f64>() / self.n_beads as f64;
            let f_slow = self.potential.force_slow(centroid);
            for i in 0..self.n_beads {
                self.forces[i] += f_slow;
            }
        } else {
            // General contraction
            let contracted_pos = self.contraction.contract(&self.positions);
            let contracted_forces: Vec<f64> = contracted_pos.iter()
                .map(|&x| self.potential.force_slow(x))
                .collect();
            let expanded = self.contraction.expand(&contracted_forces);
            for i in 0..self.n_beads {
                self.forces[i] += expanded[i];
            }
        }
    }

    /// Centroid position
    pub fn centroid(&self) -> f64 {
        self.positions.iter().sum::<f64>() / self.n_beads as f64
    }

    /// Physical potential energy (full, not split) averaged over beads
    pub fn potential_energy(&self) -> f64 {
        self.positions.iter()
            .map(|&x| self.potential.evaluate(x))
            .sum::<f64>() / self.n_beads as f64
    }

    /// Virial energy estimator
    pub fn virial_energy_estimator(&self) -> f64 {
        let xbar = self.centroid();
        let mut virial_sum = 0.0;

        for i in 0..self.n_beads {
            let dv_dx = -self.potential.force(self.positions[i]);
            virial_sum += (self.positions[i] - xbar) * dv_dx;
        }

        1.0 / (2.0 * self.beta) + self.potential_energy()
            + 0.5 * virial_sum / self.n_beads as f64
    }

    /// Radius of gyration
    pub fn radius_of_gyration(&self) -> f64 {
        let xbar = self.centroid();
        let rg2 = self.positions.iter()
            .map(|&x| (x - xbar).powi(2))
            .sum::<f64>() / self.n_beads as f64;
        rg2.sqrt()
    }
}

// =============================================================================
// RPC Simulation Engine (1D)
// =============================================================================

/// PIMD simulation with Ring Polymer Contraction.
///
/// Uses the same OBABO Langevin integration as `PIMDSimulation`
/// but with split force evaluation via RPC.
pub struct RPCSimulation<P: SplittablePotential> {
    /// Ring polymers with RPC
    pub polymers: Vec<RPCRingPolymer<P>>,
    /// Normal mode transformer
    pub nm_transform: NormalModeTransform,
    /// PILE thermostat
    pub thermostat: PILEThermostat,
    /// Time step
    pub dt: f64,
    /// Current step
    pub step: usize,
}

impl<P: SplittablePotential> RPCSimulation<P> {
    /// Create a new RPC simulation.
    ///
    /// # Arguments
    /// * `n_polymers` - Number of parallel ring polymer replicas
    /// * `n_beads` - Full bead count P
    /// * `n_contracted` - Contracted bead count P'
    /// * `beta` - Physical inverse temperature
    /// * `mass` - Particle mass
    /// * `dt` - Integration time step
    /// * `gamma_centroid` - Centroid friction
    /// * `potential` - Splittable potential
    pub fn new(
        n_polymers: usize,
        n_beads: usize,
        n_contracted: usize,
        beta: f64,
        mass: f64,
        dt: f64,
        gamma_centroid: f64,
        potential: P,
    ) -> Self {
        let nm_transform = NormalModeTransform::new(n_beads, beta);
        let thermostat = PILEThermostat::new(
            n_beads, beta, dt, mass, gamma_centroid, &nm_transform,
        );

        let polymers: Vec<RPCRingPolymer<P>> = (0..n_polymers)
            .map(|_| RPCRingPolymer::new(n_beads, n_contracted, beta, mass, potential.clone()))
            .collect();

        Self {
            polymers,
            nm_transform,
            thermostat,
            dt,
            step: 0,
        }
    }

    /// OBABO step with RPC force evaluation.
    pub fn step_obabo(&mut self) {
        let dt = self.dt;
        let half_dt = dt / 2.0;
        let nm_transform = &self.nm_transform;
        let thermostat = &self.thermostat;

        self.polymers.par_iter_mut().for_each(|polymer| {
            // O step
            let mut mode_v = nm_transform.velocities_to_normal_modes(&polymer.velocities);
            thermostat.apply_o_step(&mut mode_v);
            polymer.velocities = nm_transform.velocities_to_beads(&mode_v);

            // B step
            for i in 0..polymer.n_beads {
                polymer.velocities[i] += half_dt * polymer.forces[i] / polymer.mass;
            }

            // A step
            for i in 0..polymer.n_beads {
                polymer.positions[i] += dt * polymer.velocities[i];
            }

            // Recompute forces (with RPC)
            polymer.compute_forces();

            // B step
            for i in 0..polymer.n_beads {
                polymer.velocities[i] += half_dt * polymer.forces[i] / polymer.mass;
            }

            // O step
            let mut mode_v = nm_transform.velocities_to_normal_modes(&polymer.velocities);
            thermostat.apply_o_step(&mut mode_v);
            polymer.velocities = nm_transform.velocities_to_beads(&mode_v);
        });

        self.step += 1;
    }

    /// Average virial energy across all polymers
    pub fn average_virial_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.virial_energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average centroid position
    pub fn average_centroid(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.centroid())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average radius of gyration
    pub fn average_radius_of_gyration(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.radius_of_gyration())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average potential energy
    pub fn average_potential_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.potential_energy())
            .sum::<f64>() / self.polymers.len() as f64
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pimc::HarmonicPotential;
    use approx::assert_relative_eq;

    /// Harmonic potential as a trivially-splittable potential (all fast, no slow).
    impl SplittablePotential for HarmonicPotential {
        fn energy_fast(&self, x: f64) -> f64 {
            self.evaluate(x)
        }
        fn energy_slow(&self, _x: f64) -> f64 {
            0.0
        }
        fn force_fast(&self, x: f64) -> f64 {
            self.force(x)
        }
        fn force_slow(&self, _x: f64) -> f64 {
            0.0
        }
    }

    #[test]
    fn test_contraction_centroid() {
        // P'=1: contraction should give the centroid
        let rpc = RPContraction::new(8, 1, 10.0);
        assert!(rpc.is_centroid_contraction());

        let positions = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let contracted = rpc.contract(&positions);
        let expected_centroid = positions.iter().sum::<f64>() / 8.0;

        assert_eq!(contracted.len(), 1);
        assert_relative_eq!(contracted[0], expected_centroid, epsilon = 1e-10);
    }

    #[test]
    fn test_contraction_identity() {
        // P'=P: contraction should be identity (positions unchanged)
        let rpc = RPContraction::new(8, 8, 10.0);
        let positions = vec![1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5];

        let contracted = rpc.contract(&positions);

        for (i, (&orig, &contr)) in positions.iter().zip(contracted.iter()).enumerate() {
            assert_relative_eq!(orig, contr, epsilon = 1e-10,);
            let _ = i; // suppress unused warning
        }
    }

    #[test]
    fn test_contraction_expansion_low_freq() {
        // For smooth (low frequency) data, contract→expand should preserve well
        let n_full = 16;
        let n_contracted = 4;
        let rpc = RPContraction::new(n_full, n_contracted, 10.0);

        // Smooth signal: only low-frequency components
        let positions: Vec<f64> = (0..n_full)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / n_full as f64;
                1.0 + 0.5 * t.cos() // Only k=0 and k=1 modes
            })
            .collect();

        let contracted = rpc.contract(&positions);
        let forces = contracted.clone(); // Use positions as "forces" for testing
        let expanded = rpc.expand(&forces);

        // The expanded signal should match since it only has low-freq components
        // that are preserved by the contraction
        let contracted_2 = rpc.contract(&expanded);
        for (j, (&c1, &c2)) in contracted.iter().zip(contracted_2.iter()).enumerate() {
            assert_relative_eq!(c1, c2, epsilon = 1e-10,);
            let _ = j;
        }
    }

    #[test]
    fn test_rpc_harmonic_oscillator() {
        // RPC with all-fast potential should give correct harmonic oscillator energy
        let pot = HarmonicPotential { mass: 1.0, omega: 1.0 };
        let n_beads = 32;
        let n_contracted = 4; // Contract to 4 beads (but slow force is 0)
        let beta = 20.0;
        let mass = 1.0;
        let dt = 0.1;
        let gamma = 1.0;

        let mut sim = RPCSimulation::new(
            20, n_beads, n_contracted, beta, mass, dt, gamma, pot,
        );

        // Equilibrate
        for _ in 0..2000 {
            sim.step_obabo();
        }

        // Sample
        let mut energies = Vec::new();
        for _ in 0..2000 {
            sim.step_obabo();
            energies.push(sim.average_virial_energy());
        }

        let mean_e = energies.iter().sum::<f64>() / energies.len() as f64;
        // E0 = 0.5 for ω=1, m=1
        assert_relative_eq!(mean_e, 0.5, epsilon = 0.15);
    }

    #[test]
    fn test_splittable_double_well() {
        // Test that V = V_fast + V_slow
        let pot = SplittableDoubleWell::new(0.01, 0.75);

        for &x in &[-1.0, -0.5, 0.0, 0.3, 0.75, 1.5] {
            let v_total = pot.evaluate(x);
            let v_fast = pot.energy_fast(x);
            let v_slow = pot.energy_slow(x);
            assert_relative_eq!(v_total, v_fast + v_slow, epsilon = 1e-12);

            let f_total = pot.force(x);
            let f_fast = pot.force_fast(x);
            let f_slow = pot.force_slow(x);
            assert_relative_eq!(f_total, f_fast + f_slow, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_rpc_double_well_centroid() {
        // Centroid contraction (P'=1) with splittable double well
        let pot = SplittableDoubleWell::new(0.01, 0.75);
        let n_beads = 16;
        let n_contracted = 1; // Centroid contraction
        let beta = 10.0;
        let mass = 1836.15; // Proton mass
        let dt = 1.0;
        let gamma = 0.001;

        let mut sim = RPCSimulation::new(
            10, n_beads, n_contracted, beta, mass, dt, gamma, pot,
        );

        // Just check it runs without panicking and gives finite energy
        for _ in 0..100 {
            sim.step_obabo();
        }
        let e = sim.average_virial_energy();
        assert!(e.is_finite(), "Energy should be finite, got {}", e);
    }
}
