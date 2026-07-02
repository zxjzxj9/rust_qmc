//! Multi-Dimensional Path Integral Molecular Dynamics (PIMD) for Molecular Systems
//!
//! Extends the 1D PIMD engine to handle multi-atom, 3D molecular systems.
//! Each atom is represented as a ring polymer of P beads, with per-atom
//! masses and spring constants. The PILE thermostat operates in normal
//! mode space with mass-dependent velocity noise.
//!
//! Key features:
//! - Multi-atom ring polymers with per-atom masses
//! - Per-DOF normal mode transform and PILE thermostat
//! - Centroid virial energy estimator for N-D
//! - Supports any molecular potential via the `MolecularPotential` trait
//!
//! Reference:
//!   Ceriotti, Parrinello, Markland, Manolopoulos (2010)
//!   "Efficient stochastic thermostatting of path integral molecular dynamics"
//!   J. Chem. Phys. 133, 124104

use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

use super::potentials::MolecularPotential;
use super::potentials::molecular::{BifluoridePES, ZundelPES};

// The `MolecularPotential` trait and the concrete molecular potentials now live
// in `super::potentials`. They are imported above.

// =============================================================================
// Multi-Atom Ring Polymer
// =============================================================================

/// Ring polymer for a multi-atom molecular system.
///
/// Each bead stores a full molecular geometry (ndof = 3 x N_atoms coordinates).
/// Spring constants are per-atom: κ_a = m_a / dτ² for atom a.
///
/// Optionally supports Takahashi-Imada fourth-order factorization:
///   V_TI(R) = V(R) + (dτ²/24) Σ_a |F_a(R)|²/m_a
/// which achieves O(1/P⁴) convergence.
pub struct MolecularRingPolymer<P: MolecularPotential> {
    /// Bead positions: positions[bead][dof], each bead has ndof entries
    pub positions: Vec<Vec<f64>>,
    /// Bead velocities: velocities[bead][dof]
    pub velocities: Vec<Vec<f64>>,
    /// Forces on each bead: forces[bead][dof]
    pub forces: Vec<Vec<f64>>,
    /// Number of beads P
    pub n_beads: usize,
    /// Number of atoms
    pub n_atoms: usize,
    /// Number of DOF (3 * n_atoms)
    pub ndof: usize,
    /// Per-atom masses [m0, m1, ...]
    pub masses: Vec<f64>,
    /// Per-DOF mass (maps dof index -> atom mass): mass_dof[d] = masses[d/3]
    pub mass_dof: Vec<f64>,
    /// Per-atom spring constants κ_a = m_a / dτ²
    pub spring_constants: Vec<f64>,
    /// Per-DOF spring constants (maps dof -> spring constant)
    pub spring_dof: Vec<f64>,
    /// Physical inverse temperature beta
    pub beta: f64,
    /// Imaginary time step dτ = beta/P
    pub dtau: f64,
    /// The molecular potential
    pub potential: P,
    /// Whether to use Takahashi-Imada fourth-order correction
    pub use_ti: bool,
}

impl<P: MolecularPotential> MolecularRingPolymer<P> {
    /// Create a new multi-atom ring polymer.
    ///
    /// Initializes all beads near the reference geometry with small
    /// thermal fluctuations.
    pub fn new(n_beads: usize, beta: f64, potential: P) -> Self {
        let n_atoms = potential.n_atoms();
        let ndof = potential.ndof();
        let dtau = beta / n_beads as f64;
        let masses = potential.masses().to_vec();
        let ref_geom = potential.reference_geometry();

        // Per-atom spring constants
        let spring_constants: Vec<f64> = masses.iter()
            .map(|&m| m / (dtau * dtau))
            .collect();

        // Per-DOF mass and spring constant
        let mass_dof: Vec<f64> = (0..ndof).map(|d| masses[d / 3]).collect();
        let spring_dof: Vec<f64> = (0..ndof).map(|d| spring_constants[d / 3]).collect();

        let mut rng = rand::thread_rng();
        let bead_kbt = n_beads as f64 / beta; // Bead temperature

        // Initialize beads near reference geometry
        let mut positions = Vec::with_capacity(n_beads);
        let mut velocities = Vec::with_capacity(n_beads);

        for _bead in 0..n_beads {
            let mut pos = ref_geom.clone();
            let mut vel = vec![0.0; ndof];

            for d in 0..ndof {
                let m = mass_dof[d];
                // Small position perturbation
                let sigma_x = 0.01; // Bohr -- small perturbation
                let pos_noise = Normal::new(0.0, sigma_x).unwrap();
                pos[d] += pos_noise.sample(&mut rng);

                // Thermal velocity at bead temperature
                let sigma_v = (bead_kbt / m).sqrt();
                let vel_noise = Normal::new(0.0, sigma_v).unwrap();
                vel[d] = vel_noise.sample(&mut rng);
            }

            positions.push(pos);
            velocities.push(vel);
        }

        let forces = vec![vec![0.0; ndof]; n_beads];

        let mut rp = Self {
            positions,
            velocities,
            forces,
            n_beads,
            n_atoms,
            ndof,
            masses,
            mass_dof,
            spring_constants,
            spring_dof,
            beta,
            dtau,
            potential,
            use_ti: false,
        };
        rp.compute_forces();
        rp
    }

    /// Compute total forces on each bead: F = F_spring + F_physical
    ///
    /// Phase 1: physical forces computed in parallel across beads (independent)
    /// Phase 2: spring forces added sequentially (depends on neighbors)
    /// Phase 3: (if use_ti) TI correction added via numerical force gradient
    pub fn compute_forces(&mut self) {
        let ndof = self.ndof;
        let potential = &self.potential;

        // Phase 1: compute physical forces in parallel (each bead is independent)
        self.forces.par_iter_mut()
            .zip(self.positions.par_iter())
            .for_each(|(f, pos)| {
                potential.forces(pos, f);
            });

        // Phase 1b: if TI, add the force gradient correction
        // F_TI_d = F_d - (dτ²/12m_d) Σ_d' F_d'·(∂F_d'/∂R_d) / m_d'
        // Simplified: use numerical gradient of |F|²/2m contribution
        if self.use_ti {
            let dtau2 = self.dtau * self.dtau;
            let h = 1e-6;
            let mass_dof = &self.mass_dof;

            // For each bead, compute the TI force correction numerically:
            // -∇_d [(dτ²/24) Σ_d' |F_d'|²/m_d']
            // = -(dτ²/24) * d/dR_d [Σ_d' F_d'² / m_d']
            // via central difference
            for i in 0..self.n_beads {
                let mut pos_plus = self.positions[i].clone();
                let mut pos_minus = self.positions[i].clone();
                let mut f_plus = vec![0.0; ndof];
                let mut f_minus = vec![0.0; ndof];

                for d in 0..ndof {
                    pos_plus[d] = self.positions[i][d] + h;
                    pos_minus[d] = self.positions[i][d] - h;

                    potential.forces(&pos_plus, &mut f_plus);
                    potential.forces(&pos_minus, &mut f_minus);

                    // Σ_d' |F_d'|²/m_d' at pos+h and pos-h
                    let sum_plus: f64 = f_plus.iter().enumerate()
                        .map(|(dp, &fp)| fp * fp / mass_dof[dp])
                        .sum();
                    let sum_minus: f64 = f_minus.iter().enumerate()
                        .map(|(dp, &fm)| fm * fm / mass_dof[dp])
                        .sum();

                    let d_sum_dr = (sum_plus - sum_minus) / (2.0 * h);
                    self.forces[i][d] -= (dtau2 / 24.0) * d_sum_dr;

                    pos_plus[d] = self.positions[i][d];
                    pos_minus[d] = self.positions[i][d];
                }
            }
        }

        // Phase 2: add spring forces (depends on neighboring bead positions)
        for i in 0..self.n_beads {
            let prev = if i == 0 { self.n_beads - 1 } else { i - 1 };
            let next = (i + 1) % self.n_beads;
            for d in 0..ndof {
                let f_spring = -self.spring_dof[d]
                    * (2.0 * self.positions[i][d]
                       - self.positions[prev][d]
                       - self.positions[next][d]);
                self.forces[i][d] += f_spring;
            }
        }
    }

    /// Centroid position: R̄ = (1/P) Σ_i R_i
    pub fn centroid(&self) -> Vec<f64> {
        let mut cent = vec![0.0; self.ndof];
        for bead in &self.positions {
            for d in 0..self.ndof {
                cent[d] += bead[d];
            }
        }
        let p = self.n_beads as f64;
        for c in &mut cent {
            *c /= p;
        }
        cent
    }

    /// Bead-averaged potential energy: (1/P) Σ_i V(R_i)
    pub fn potential_energy(&self) -> f64 {
        self.positions.par_iter()
            .map(|pos| self.potential.energy(pos))
            .sum::<f64>() / self.n_beads as f64
    }

    /// Centroid virial energy estimator for N-D:
    ///
    ///   E_cv = N_dof/(2beta) + (1/P) Σ_i [V(R_i) + 0.5 Σ_d (R_i[d] - R̄[d]) x (∂V/∂R_i[d])]
    ///
    /// Here N_dof is the number of degrees of freedom.
    pub fn virial_energy_estimator(&self) -> f64 {
        let cent = self.centroid();
        let ndof = self.ndof;
        let p = self.n_beads as f64;

        // Compute per-bead virial contributions in parallel
        let sum: f64 = self.positions.par_iter()
            .map(|pos| {
                let mut phys_forces = vec![0.0; ndof];
                self.potential.forces(pos, &mut phys_forces);
                let v_i = self.potential.energy(pos);
                let mut virial = 0.0;
                for d in 0..ndof {
                    // ∂V/∂R = -F_phys, so (R-R̄).(∂V/∂R) = -(R-R̄).F
                    virial += (pos[d] - cent[d]) * (-phys_forces[d]);
                }
                v_i + 0.5 * virial
            })
            .sum();

        // N_dof/(2beta) + bead-averaged [ V + virial_correction ]
        ndof as f64 / (2.0 * self.beta) + sum / p
    }

    /// Takahashi-Imada virial energy estimator for multi-atom systems.
    ///
    /// Adds the TI correction to the standard virial estimator:
    ///   E_TI = E_vir + (1/P) Σ_i (dτ²/12) Σ_a |F_a(R_i)|²/m_a
    ///
    /// Reference: Yamamoto, JCP 123, 104101 (2005)
    pub fn ti_virial_energy_estimator(&self) -> f64 {
        let base = self.virial_energy_estimator();
        let dtau2 = self.dtau * self.dtau;
        let ndof = self.ndof;

        let ti_correction: f64 = self.positions.par_iter()
            .map(|pos| {
                let mut phys_forces = vec![0.0; ndof];
                self.potential.forces(pos, &mut phys_forces);
                let mut sum_f2_m: f64 = 0.0;
                for d in 0..ndof {
                    sum_f2_m += phys_forces[d] * phys_forces[d] / self.mass_dof[d];
                }
                (dtau2 / 12.0) * sum_f2_m
            })
            .sum::<f64>() / self.n_beads as f64;

        base + ti_correction
    }

    /// Get the appropriate energy estimator based on whether TI is enabled.
    pub fn energy_estimator(&self) -> f64 {
        if self.use_ti {
            self.ti_virial_energy_estimator()
        } else {
            self.virial_energy_estimator()
        }
    }

    /// Radius of gyration for a specific atom a:
    ///   R_g²(a) = (1/P) Σ_i |r_a^(i) - r̄_a|²
    pub fn atom_radius_of_gyration(&self, atom: usize) -> f64 {
        let cent = self.centroid();
        let mut rg2 = 0.0;
        for bead in &self.positions {
            for xyz in 0..3 {
                let d = 3 * atom + xyz;
                let dr = bead[d] - cent[d];
                rg2 += dr * dr;
            }
        }
        (rg2 / self.n_beads as f64).sqrt()
    }

    /// Distance between two atoms in the centroid geometry
    pub fn centroid_distance(&self, atom_a: usize, atom_b: usize) -> f64 {
        let cent = self.centroid();
        let mut dist2 = 0.0;
        for xyz in 0..3 {
            let da = cent[3 * atom_a + xyz] - cent[3 * atom_b + xyz];
            dist2 += da * da;
        }
        dist2.sqrt()
    }

    /// Proton transfer coordinate: difference of distances d(A-H) - d(B-H)
    /// for atom indices (donor, proton, acceptor).
    /// Negative = proton closer to donor, positive = closer to acceptor.
    pub fn transfer_coordinate(&self, donor: usize, proton: usize, acceptor: usize) -> f64 {
        let cent = self.centroid();
        let mut dah2 = 0.0;
        let mut dbh2 = 0.0;
        for xyz in 0..3 {
            let rh = cent[3 * proton + xyz];
            let ra = cent[3 * donor + xyz];
            let rb = cent[3 * acceptor + xyz];
            dah2 += (rh - ra).powi(2);
            dbh2 += (rh - rb).powi(2);
        }
        dah2.sqrt() - dbh2.sqrt()
    }

    /// Bead-resolved transfer coordinate: for each bead, compute d(A-H) - d(B-H)
    pub fn bead_transfer_coordinates(&self, donor: usize, proton: usize, acceptor: usize) -> Vec<f64> {
        self.positions.iter().map(|pos| {
            let mut dah2 = 0.0;
            let mut dbh2 = 0.0;
            for xyz in 0..3 {
                let rh = pos[3 * proton + xyz];
                let ra = pos[3 * donor + xyz];
                let rb = pos[3 * acceptor + xyz];
                dah2 += (rh - ra).powi(2);
                dbh2 += (rh - rb).powi(2);
            }
            dah2.sqrt() - dbh2.sqrt()
        }).collect()
    }
}

// =============================================================================
// Per-Atom PILE Thermostat
// =============================================================================

/// PILE thermostat adapted for multi-atom molecular systems.
///
/// Each degree of freedom is thermostatted independently in normal mode space.
/// The per-atom mass enters through the velocity noise width:
///   σ_{a,k} = sqrt(kBT_P / m_a)
///
/// Mode frequencies w_k are the same for all atoms (they depend only on beta and P).
pub struct MolecularPILE {
    /// Number of beads P
    pub n_beads: usize,
    /// Number of DOF
    pub ndof: usize,
    /// Bead temperature kBT_P = P/beta
    pub kbt_p: f64,
    /// Per-DOF velocity noise width: σ[d] = sqrt(kBT_P / m_d)
    pub sigma: Vec<f64>,
    /// Propagator coefficients c1[k] = exp(-γ_k dt/2)
    pub c1: Vec<f64>,
    /// Noise coefficients c2[k] = sqrt(1 - c1[k]²)
    pub c2: Vec<f64>,
}

impl MolecularPILE {
    /// Create a molecular PILE thermostat.
    pub fn new(
        n_beads: usize,
        beta: f64,
        dt: f64,
        masses: &[f64],   // per-atom masses
        gamma_centroid: f64,
        nm_frequencies: &[f64],  // normal mode frequencies w_k
    ) -> Self {
        let n_atoms = masses.len();
        let ndof = 3 * n_atoms;
        let kbt_p = n_beads as f64 / beta;

        // Per-DOF velocity noise width
        let sigma: Vec<f64> = (0..ndof)
            .map(|d| (kbt_p / masses[d / 3]).sqrt())
            .collect();

        // Friction coefficients per mode
        let mut gamma = vec![0.0; n_beads];
        gamma[0] = gamma_centroid;
        for k in 1..n_beads {
            gamma[k] = 2.0 * nm_frequencies[k.min(n_beads - k)];
        }

        // O-step propagator
        let half_dt = dt / 2.0;
        let c1: Vec<f64> = gamma.iter().map(|&g| (-g * half_dt).exp()).collect();
        let c2: Vec<f64> = c1.iter().map(|&c| (1.0 - c * c).sqrt()).collect();

        Self { n_beads, ndof, kbt_p, sigma, c1, c2 }
    }

    /// Apply O step to normal mode velocities for all DOF.
    ///
    /// `mode_vel` is [mode_k][dof_d] -- mode k's velocity for each DOF.
    pub fn apply_o_step(&self, mode_vel: &mut Vec<Vec<f64>>) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for k in 0..self.n_beads {
            for d in 0..self.ndof {
                let noise: f64 = normal.sample(&mut rng);
                mode_vel[k][d] = self.c1[k] * mode_vel[k][d]
                    + self.c2[k] * self.sigma[d] * noise;
            }
        }
    }
}

// =============================================================================
// Molecular PIMD Simulation
// =============================================================================

/// Multi-atom PIMD simulation with PILE thermostat.
pub struct MolecularPIMD<P: MolecularPotential> {
    /// Ring polymers (parallel replicas)
    pub polymers: Vec<MolecularRingPolymer<P>>,
    /// Normal mode transform (shared, dimension-independent)
    pub nm_transform: super::pimd_core::NormalModeTransform,
    /// PILE thermostat
    pub thermostat: MolecularPILE,
    /// Time step
    pub dt: f64,
    /// Current step
    pub step: usize,
    /// Number of DOF
    pub ndof: usize,
}

impl<P: MolecularPotential> MolecularPIMD<P> {
    /// Create a new multi-atom PIMD simulation.
    pub fn new(
        n_polymers: usize,
        n_beads: usize,
        beta: f64,
        dt: f64,
        gamma_centroid: f64,
        potential: P,
    ) -> Self {
        let ndof = potential.ndof();
        let nm_transform = super::pimd_core::NormalModeTransform::new(n_beads, beta);
        let thermostat = MolecularPILE::new(
            n_beads, beta, dt,
            potential.masses(),
            gamma_centroid,
            &nm_transform.frequencies,
        );

        let polymers: Vec<MolecularRingPolymer<P>> = (0..n_polymers)
            .map(|_| MolecularRingPolymer::new(n_beads, beta, potential.clone()))
            .collect();

        Self {
            polymers,
            nm_transform,
            thermostat,
            dt,
            step: 0,
            ndof,
        }
    }

    /// Create a new multi-atom PIMD simulation with Takahashi-Imada correction.
    ///
    /// Same as `new()` but enables TI fourth-order correction on all ring polymers.
    ///
    /// Reference: Takahashi & Imada, J. Phys. Soc. Jpn. 53, 3765 (1984)
    pub fn new_with_ti(
        n_polymers: usize,
        n_beads: usize,
        beta: f64,
        dt: f64,
        gamma_centroid: f64,
        potential: P,
    ) -> Self {
        let mut sim = Self::new(n_polymers, n_beads, beta, dt, gamma_centroid, potential);
        for polymer in &mut sim.polymers {
            polymer.use_ti = true;
            polymer.compute_forces();
        }
        sim
    }

    /// Normal mode transform for a single DOF across all beads.
    /// Extracts bead values for DOF d, transforms, and returns mode values.
    #[allow(dead_code)]
    fn to_modes_1d(&self, bead_values: &[Vec<f64>], dof: usize) -> Vec<f64> {
        let vals: Vec<f64> = bead_values.iter().map(|b| b[dof]).collect();
        self.nm_transform.to_normal_modes(&vals)
    }

    /// Inverse: mode values for DOF d -> bead values
    #[allow(dead_code)]
    fn to_beads_1d(&self, mode_values: &[f64]) -> Vec<f64> {
        self.nm_transform.to_beads(mode_values)
    }

    /// OBABO Langevin step for multi-atom system.
    ///
    /// All polymer replicas are stepped in parallel using Rayon.
    /// Each polymer's O-step thermostat uses thread-local RNG.
    pub fn step_obabo(&mut self) {
        let dt = self.dt;
        let half_dt = dt / 2.0;
        let n_beads = self.nm_transform.n_beads;
        let ndof = self.ndof;
        let nm_transform = &self.nm_transform;
        let thermostat = &self.thermostat;

        self.polymers.par_iter_mut().for_each(|polymer| {
            // === O step: thermostat in normal mode space ===
            // Transform velocities to normal modes (per DOF)
            let mut mode_vel: Vec<Vec<f64>> = vec![vec![0.0; ndof]; n_beads];
            for d in 0..ndof {
                let bead_v: Vec<f64> = polymer.velocities.iter().map(|v| v[d]).collect();
                let modes = nm_transform.to_normal_modes(&bead_v);
                for k in 0..n_beads {
                    mode_vel[k][d] = modes[k];
                }
            }

            thermostat.apply_o_step(&mut mode_vel);

            // Transform back to bead space
            for d in 0..ndof {
                let modes: Vec<f64> = mode_vel.iter().map(|m| m[d]).collect();
                let bead_v = nm_transform.to_beads(&modes);
                for i in 0..n_beads {
                    polymer.velocities[i][d] = bead_v[i];
                }
            }

            // === B step: half-step velocity from forces ===
            for i in 0..n_beads {
                for d in 0..ndof {
                    polymer.velocities[i][d] += half_dt * polymer.forces[i][d] / polymer.mass_dof[d];
                }
            }

            // === A step: full-step position update ===
            for i in 0..n_beads {
                for d in 0..ndof {
                    polymer.positions[i][d] += dt * polymer.velocities[i][d];
                }
            }

            // === Recompute forces ===
            polymer.compute_forces();

            // === B step: half-step velocity from forces ===
            for i in 0..n_beads {
                for d in 0..ndof {
                    polymer.velocities[i][d] += half_dt * polymer.forces[i][d] / polymer.mass_dof[d];
                }
            }

            // === O step: thermostat in normal mode space ===
            let mut mode_vel: Vec<Vec<f64>> = vec![vec![0.0; ndof]; n_beads];
            for d in 0..ndof {
                let bead_v: Vec<f64> = polymer.velocities.iter().map(|v| v[d]).collect();
                let modes = nm_transform.to_normal_modes(&bead_v);
                for k in 0..n_beads {
                    mode_vel[k][d] = modes[k];
                }
            }

            thermostat.apply_o_step(&mut mode_vel);

            for d in 0..ndof {
                let modes: Vec<f64> = mode_vel.iter().map(|m| m[d]).collect();
                let bead_v = nm_transform.to_beads(&modes);
                for i in 0..n_beads {
                    polymer.velocities[i][d] = bead_v[i];
                }
            }
        });

        self.step += 1;
    }

    /// OBABO step with PIQTB quantum thermal bath thermostat.
    ///
    /// Uses the PIQTB thermostat which enforces quantum harmonic oscillator
    /// energy distributions per mode, enabling faster convergence with
    /// fewer beads for molecular systems.
    pub fn step_obabo_piqtb(&mut self, piqtb: &super::piglet::MolecularPIQTB) {
        let dt = self.dt;
        let half_dt = dt / 2.0;
        let nm_transform = &self.nm_transform;
        let ndof = self.ndof;
        let n_beads = self.polymers[0].n_beads;

        self.polymers.par_iter_mut().for_each(|polymer| {
            // O step with PIQTB
            let mut mode_vel: Vec<Vec<f64>> = vec![vec![0.0; ndof]; n_beads];
            for d in 0..ndof {
                let bead_v: Vec<f64> = polymer.velocities.iter().map(|v| v[d]).collect();
                let modes = nm_transform.to_normal_modes(&bead_v);
                for k in 0..n_beads {
                    mode_vel[k][d] = modes[k];
                }
            }
            piqtb.apply_o_step(&mut mode_vel);
            for d in 0..ndof {
                let modes: Vec<f64> = mode_vel.iter().map(|m| m[d]).collect();
                let bead_v = nm_transform.to_beads(&modes);
                for i in 0..n_beads {
                    polymer.velocities[i][d] = bead_v[i];
                }
            }

            // B step
            for i in 0..n_beads {
                for d in 0..ndof {
                    polymer.velocities[i][d] += half_dt * polymer.forces[i][d] / polymer.mass_dof[d];
                }
            }

            // A step
            for i in 0..n_beads {
                for d in 0..ndof {
                    polymer.positions[i][d] += dt * polymer.velocities[i][d];
                }
            }

            // Recompute forces
            polymer.compute_forces();

            // B step
            for i in 0..n_beads {
                for d in 0..ndof {
                    polymer.velocities[i][d] += half_dt * polymer.forces[i][d] / polymer.mass_dof[d];
                }
            }

            // O step with PIQTB
            let mut mode_vel: Vec<Vec<f64>> = vec![vec![0.0; ndof]; n_beads];
            for d in 0..ndof {
                let bead_v: Vec<f64> = polymer.velocities.iter().map(|v| v[d]).collect();
                let modes = nm_transform.to_normal_modes(&bead_v);
                for k in 0..n_beads {
                    mode_vel[k][d] = modes[k];
                }
            }
            piqtb.apply_o_step(&mut mode_vel);
            for d in 0..ndof {
                let modes: Vec<f64> = mode_vel.iter().map(|m| m[d]).collect();
                let bead_v = nm_transform.to_beads(&modes);
                for i in 0..n_beads {
                    polymer.velocities[i][d] = bead_v[i];
                }
            }
        });

        self.step += 1;
    }

    /// Average virial energy across all polymers
    pub fn average_virial_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.virial_energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average Takahashi-Imada virial energy across all polymers.
    ///
    /// Uses the TI-corrected virial estimator which includes the
    /// (dτ²/12) Σ_a |F_a|²/m_a correction term.
    pub fn average_ti_virial_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.ti_virial_energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average energy using the appropriate estimator (TI if enabled, virial otherwise).
    pub fn average_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average potential energy across all polymers
    pub fn average_potential_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.potential_energy())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average transfer coordinate across all polymers
    pub fn average_transfer_coordinate(&self, donor: usize, proton: usize, acceptor: usize) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.transfer_coordinate(donor, proton, acceptor))
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average R_g for a specific atom across all polymers
    pub fn average_atom_rg(&self, atom: usize) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.atom_radius_of_gyration(atom))
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Accumulate transfer coordinate histogram from all beads of all polymers
    pub fn accumulate_transfer_histogram(
        &self,
        counts: &mut [f64],
        x_min: f64,
        x_max: f64,
        donor: usize,
        proton: usize,
        acceptor: usize,
    ) {
        let n_bins = counts.len();
        let bin_width = (x_max - x_min) / n_bins as f64;

        for polymer in &self.polymers {
            let tc = polymer.bead_transfer_coordinates(donor, proton, acceptor);
            for &x in &tc {
                if x >= x_min && x < x_max {
                    let bin = ((x - x_min) / bin_width) as usize;
                    if bin < n_bins {
                        counts[bin] += 1.0;
                    }
                }
            }
        }
    }

    /// Fraction of beads where the proton is closer to the acceptor (transfer coord > 0)
    pub fn tunneling_fraction(&self, donor: usize, proton: usize, acceptor: usize) -> f64 {
        let (tunneled, total) = self.polymers.par_iter()
            .map(|polymer| {
                let tc = polymer.bead_transfer_coordinates(donor, proton, acceptor);
                let t = tc.iter().filter(|&&x| x > 0.0).count();
                (t, tc.len())
            })
            .reduce(|| (0, 0), |(t1, n1), (t2, n2)| (t1 + t2, n1 + n2));
        tunneled as f64 / total as f64
    }

    /// Average R_OO (donor-acceptor distance) across all polymers
    pub fn average_roo(&self, donor: usize, acceptor: usize) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.centroid_distance(donor, acceptor))
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Proton R_g decomposed into parallel and perpendicular components
    /// relative to the donor-acceptor (O...O) axis.
    /// Returns (R_g_parallel, R_g_perpendicular) averaged over all polymers.
    pub fn average_proton_rg_decomposed(
        &self, donor: usize, proton: usize, acceptor: usize,
    ) -> (f64, f64) {
        let n = self.polymers.len() as f64;

        let (rg_par_sum, rg_perp_sum) = self.polymers.par_iter()
            .map(|polymer| {
                let cent = polymer.centroid();
                // O-O axis direction
                let mut axis = [0.0; 3];
                let mut axis_len = 0.0;
                for xyz in 0..3 {
                    let d = cent[3*acceptor+xyz] - cent[3*donor+xyz];
                    axis[xyz] = d;
                    axis_len += d * d;
                }
                axis_len = axis_len.sqrt().max(1e-15);
                for xyz in 0..3 { axis[xyz] /= axis_len; }

                // Centroid of the proton
                let mut h_cent = [0.0; 3];
                for xyz in 0..3 {
                    h_cent[xyz] = cent[3*proton+xyz];
                }

                let mut par2 = 0.0;
                let mut perp2 = 0.0;
                for bead in &polymer.positions {
                    let mut dh = [0.0; 3];
                    for xyz in 0..3 {
                        dh[xyz] = bead[3*proton+xyz] - h_cent[xyz];
                    }
                    let proj = dh[0]*axis[0] + dh[1]*axis[1] + dh[2]*axis[2];
                    par2 += proj * proj;
                    perp2 += dh[0]*dh[0] + dh[1]*dh[1] + dh[2]*dh[2] - proj*proj;
                }
                let p = polymer.n_beads as f64;
                ((par2 / p).sqrt(), (perp2.max(0.0) / p).sqrt())
            })
            .reduce(|| (0.0, 0.0), |(a1, b1), (a2, b2)| (a1 + a2, b1 + b2));

        (rg_par_sum / n, rg_perp_sum / n)
    }

    /// Accumulate 2D histogram of (R_OO, transfer_coordinate) for correlation analysis
    pub fn accumulate_roo_tc_histogram(
        &self,
        counts: &mut Vec<Vec<f64>>,
        roo_min: f64, roo_max: f64, n_roo_bins: usize,
        tc_min: f64, tc_max: f64, n_tc_bins: usize,
        donor: usize, proton: usize, acceptor: usize,
    ) {
        let roo_bw = (roo_max - roo_min) / n_roo_bins as f64;
        let tc_bw = (tc_max - tc_min) / n_tc_bins as f64;

        for polymer in &self.polymers {
            for bead in &polymer.positions {
                // R_OO for this bead
                let mut roo2 = 0.0;
                for xyz in 0..3 {
                    let d = bead[3*acceptor+xyz] - bead[3*donor+xyz];
                    roo2 += d * d;
                }
                let roo = roo2.sqrt();

                // Transfer coordinate for this bead
                let mut dah2 = 0.0;
                let mut dbh2 = 0.0;
                for xyz in 0..3 {
                    let rh = bead[3*proton+xyz];
                    let ra = bead[3*donor+xyz];
                    let rb = bead[3*acceptor+xyz];
                    dah2 += (rh - ra).powi(2);
                    dbh2 += (rh - rb).powi(2);
                }
                let tc = dah2.sqrt() - dbh2.sqrt();

                if roo >= roo_min && roo < roo_max && tc >= tc_min && tc < tc_max {
                    let ri = ((roo - roo_min) / roo_bw) as usize;
                    let ti = ((tc - tc_min) / tc_bw) as usize;
                    if ri < n_roo_bins && ti < n_tc_bins {
                        counts[ri][ti] += 1.0;
                    }
                }
            }
        }
    }
}

/// Convert a probability histogram P(x) to free energy W(x) = -k_BT * ln(P(x)).
/// Returns W(x) shifted so that the minimum is zero.
pub fn free_energy_profile(histogram: &[f64], bin_width: f64, kbt: f64) -> Vec<f64> {
    let total: f64 = histogram.iter().sum();
    if total <= 0.0 {
        return vec![0.0; histogram.len()];
    }
    let prob: Vec<f64> = histogram.iter().map(|&c| c / (total * bin_width)).collect();
    let max_p = prob.iter().cloned().fold(0.0_f64, f64::max);
    if max_p <= 0.0 {
        return vec![0.0; histogram.len()];
    }
    let mut w: Vec<f64> = prob.iter().map(|&p| {
        if p > 1e-30 {
            -kbt * (p / max_p).ln()
        } else {
            10.0 * kbt // Large but finite for empty bins
        }
    }).collect();
    let w_min = w.iter().cloned().fold(f64::INFINITY, f64::min);
    for v in &mut w { *v -= w_min; }
    w
}
// The BifluoridePES potential now lives in super::potentials::molecular.


// =============================================================================
// Simulation Driver
// =============================================================================

/// Run PIMD simulation for proton transfer in bifluoride HF2-.
///
/// Compares classical (P=1) and quantum (P=n_beads) behavior.
/// Atom ordering: F1(0), H(1), F2(2) -- proton transfers between two fluorines.
pub fn run_pimd_bifluoride(
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
) {
    let pes = BifluoridePES::new();
    let m_h = pes.masses()[1];

    // Indices: F1=0, H=1, F2=2
    let donor = 0_usize;
    let proton = 1_usize;
    let acceptor = 2_usize;

    println!("==============================================================");
    println!("  Multi-Atom PIMD -- Bifluoride HF2- Proton Transfer");
    println!("  3D Ring Polymer with PILE Thermostat");
    println!("==============================================================");
    println!();
    println!("System: F-H...F (3 atoms, 9 DOF)");
    println!("  F mass:   {:.2} a.u.", pes.masses()[0]);
    println!("  H mass:   {:.2} a.u.", m_h);
    println!("  F...F eq: {:.4} Bohr ({:.4} A)", pes.r_ff_eq, pes.r_ff_eq * 0.529177);
    println!("  F-H eq:   {:.4} Bohr ({:.4} A)", pes.r_fh_eq, pes.r_fh_eq * 0.529177);
    println!("  Barrier:  {:.6} Hartree ({:.2} kcal/mol)", pes.barrier_height, pes.barrier_height * 627.509);
    println!("  Temp:     {:.1} K (beta = {:.2} a.u.)", 315774.65 / beta, beta);
    println!();
    println!("Simulation:");
    println!("  Beads: {}", n_beads);
    println!("  Replicas: {}", n_polymers);
    println!("  dt: {:.4} a.u.", dt);
    println!("  Equilibration: {} steps", n_equilibrate);
    println!("  Production: {} steps", n_production);
    println!();

    let gamma_centroid = 0.001; // Light centroid friction

    // =========================================================================
    // Classical (P=1)
    // =========================================================================
    println!("-----------------------------------------------------------");
    println!("  Running CLASSICAL simulation (P = 1)...");
    println!("-----------------------------------------------------------");

    let mut cl_sim = MolecularPIMD::new(n_polymers, 1, beta, dt, gamma_centroid, pes.clone());

    for step in 0..n_equilibrate {
        cl_sim.step_obabo();
        if step % (n_equilibrate / 5).max(1) == 0 {
            let tc = cl_sim.average_transfer_coordinate(donor, proton, acceptor);
            println!("  Equil {:6}: E_vir = {:10.6}, d = {:8.5}",
                     step, cl_sim.average_virial_energy(), tc);
        }
    }
    println!();

    let sample_interval = 10;
    let n_hist_bins = 200;
    let hist_min = -2.0;
    let hist_max = 2.0;
    let hist_bw = (hist_max - hist_min) / n_hist_bins as f64;

    let mut cl_energies = Vec::new();
    let mut cl_tc = Vec::new();
    let mut cl_hist = vec![0.0_f64; n_hist_bins];
    let mut cl_tunnel_sum = 0.0_f64;
    let mut cl_tunnel_n = 0_usize;

    for step in 0..n_production {
        cl_sim.step_obabo();
        if step % sample_interval == 0 {
            cl_energies.push(cl_sim.average_virial_energy());
            cl_tc.push(cl_sim.average_transfer_coordinate(donor, proton, acceptor));
            cl_tunnel_sum += cl_sim.tunneling_fraction(donor, proton, acceptor);
            cl_tunnel_n += 1;
            cl_sim.accumulate_transfer_histogram(&mut cl_hist, hist_min, hist_max, donor, proton, acceptor);
        }
        if step % (n_production / 5).max(1) == 0 {
            let tc = cl_sim.average_transfer_coordinate(donor, proton, acceptor);
            println!("  Prod {:6}: E_vir = {:10.6}, d = {:8.5}, tunnel = {:.2}%",
                     step, cl_sim.average_virial_energy(), tc,
                     100.0 * cl_sim.tunneling_fraction(donor, proton, acceptor));
        }
    }

    let cl_n = cl_energies.len() as f64;
    let cl_mean_e = cl_energies.iter().sum::<f64>() / cl_n;
    let cl_mean_tc = cl_tc.iter().sum::<f64>() / cl_n;
    let cl_tunnel = cl_tunnel_sum / cl_tunnel_n as f64;

    println!();
    println!("  Classical: E = {:.6} Ha, <d> = {:.5}, tunnel = {:.2}%",
             cl_mean_e, cl_mean_tc, 100.0 * cl_tunnel);

    // =========================================================================
    // Quantum (P = n_beads)
    // =========================================================================
    println!();
    println!("-----------------------------------------------------------");
    println!("  Running QUANTUM simulation (P = {})...", n_beads);
    println!("-----------------------------------------------------------");

    let mut q_sim = MolecularPIMD::new(n_polymers, n_beads, beta, dt, gamma_centroid, pes.clone());

    for step in 0..n_equilibrate {
        q_sim.step_obabo();
        if step % (n_equilibrate / 5).max(1) == 0 {
            let tc = q_sim.average_transfer_coordinate(donor, proton, acceptor);
            let rg_h = q_sim.average_atom_rg(proton);
            println!("  Equil {:6}: E_vir = {:10.6}, d = {:8.5}, R_g(H) = {:8.5}",
                     step, q_sim.average_virial_energy(), tc, rg_h);
        }
    }
    println!();

    let mut q_energies = Vec::new();
    let mut q_tc = Vec::new();
    let mut q_rg_h = Vec::new();
    let mut q_hist = vec![0.0_f64; n_hist_bins];
    let mut q_tunnel_sum = 0.0_f64;
    let mut q_tunnel_n = 0_usize;

    for step in 0..n_production {
        q_sim.step_obabo();
        if step % sample_interval == 0 {
            q_energies.push(q_sim.average_virial_energy());
            q_tc.push(q_sim.average_transfer_coordinate(donor, proton, acceptor));
            q_rg_h.push(q_sim.average_atom_rg(proton));
            q_tunnel_sum += q_sim.tunneling_fraction(donor, proton, acceptor);
            q_tunnel_n += 1;
            q_sim.accumulate_transfer_histogram(&mut q_hist, hist_min, hist_max, donor, proton, acceptor);
        }
        if step % (n_production / 5).max(1) == 0 {
            let tc = q_sim.average_transfer_coordinate(donor, proton, acceptor);
            let rg_h = q_sim.average_atom_rg(proton);
            println!("  Prod {:6}: E_vir = {:10.6}, d = {:8.5}, R_g(H) = {:8.5}, tunnel = {:.2}%",
                     step, q_sim.average_virial_energy(), tc, rg_h,
                     100.0 * q_sim.tunneling_fraction(donor, proton, acceptor));
        }
    }

    let q_n = q_energies.len() as f64;
    let q_mean_e = q_energies.iter().sum::<f64>() / q_n;
    let q_var_e = q_energies.iter().map(|e| (e - q_mean_e).powi(2)).sum::<f64>() / q_n;
    let q_stderr_e = q_var_e.sqrt() / q_n.sqrt();
    let q_mean_tc = q_tc.iter().sum::<f64>() / q_n;
    let q_mean_rg = q_rg_h.iter().sum::<f64>() / q_n;
    let q_tunnel = q_tunnel_sum / q_tunnel_n as f64;

    println!();
    println!("  Quantum: E = {:.6} +/- {:.6} Ha, <d> = {:.5}, R_g(H) = {:.5}, tunnel = {:.2}%",
             q_mean_e, q_stderr_e, q_mean_tc, q_mean_rg, 100.0 * q_tunnel);

    // =========================================================================
    // Summary
    // =========================================================================
    println!();
    println!("==============================================================");
    println!("                    COMPARISON SUMMARY");
    println!("==============================================================");
    println!("  {:>18} | {:>14} | {:>14}", "Property", "Classical", "Quantum");
    println!("  {:>18} | {:>14} | {:>14}", "------------------", "--------------", "--------------");
    println!("  {:>18} | {:>14.6} | {:>14.6}", "E_virial (Ha)", cl_mean_e, q_mean_e);
    println!("  {:>18} | {:>14.5} | {:>14.5}", "d (Bohr)", cl_mean_tc, q_mean_tc);
    println!("  {:>18} | {:>14} | {:>14.5}", "R_g(H) (Bohr)", "N/A", q_mean_rg);
    println!("  {:>18} | {:>13.2}% | {:>13.2}%", "Tunneling", 100.0 * cl_tunnel, 100.0 * q_tunnel);
    println!("==============================================================");
    println!();

    if q_tunnel > cl_tunnel + 0.01 {
        println!("  * Quantum tunneling OBSERVED in HF2-!");
        println!("    Proton ring polymer delocalizes between the two fluorines.");
    }
    let zpe_diff = q_mean_e - cl_mean_e;
    if zpe_diff > 0.001 {
        println!("  * Zero-point energy: +{:.4} Ha ({:.1} kcal/mol)",
                 zpe_diff, zpe_diff * 627.509);
    }
    if q_mean_rg > 0.05 {
        println!("  * Proton R_g = {:.4} Bohr -> significant quantum delocalization", q_mean_rg);
    }

    // =========================================================================
    // Output files
    // =========================================================================
    // Transfer coordinate distribution
    {
        let file = File::create("pimd_bifluoride_distribution.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# delta classical_P(delta) quantum_P(delta)").unwrap();

        let cl_total: f64 = cl_hist.iter().sum();
        let q_total: f64 = q_hist.iter().sum();

        for i in 0..n_hist_bins {
            let x = hist_min + (i as f64 + 0.5) * hist_bw;
            let cl_p = if cl_total > 0.0 { cl_hist[i] / (cl_total * hist_bw) } else { 0.0 };
            let q_p = if q_total > 0.0 { q_hist[i] / (q_total * hist_bw) } else { 0.0 };
            writeln!(w, "{:.6} {:.6} {:.6}", x, cl_p, q_p).unwrap();
        }
        println!();
        println!("  Transfer coordinate distribution -> pimd_bifluoride_distribution.txt");
    }

    // Energy trajectory
    {
        let file = File::create("pimd_bifluoride_energy.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# sample E_classical E_quantum delta_cl delta_q R_g_H").unwrap();
        let n = q_energies.len().min(cl_energies.len());
        for i in 0..n {
            writeln!(w, "{} {:.6} {:.6} {:.6} {:.6} {:.6}",
                     i, cl_energies[i], q_energies[i],
                     cl_tc[i], q_tc[i], q_rg_h[i]).unwrap();
        }
        println!("  Energy trajectory -> pimd_bifluoride_energy.txt");
    }

    // Bead snapshot
    {
        let file = File::create("pimd_bifluoride_beads.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# polymer bead atom x y z").unwrap();
        for (pi, polymer) in q_sim.polymers.iter().enumerate() {
            for (bi, bead) in polymer.positions.iter().enumerate() {
                for a in 0..3 {
                    writeln!(w, "{} {} {} {:.6} {:.6} {:.6}",
                             pi, bi, a, bead[3*a], bead[3*a+1], bead[3*a+2]).unwrap();
                }
            }
        }
        println!("  Bead snapshot -> pimd_bifluoride_beads.txt");
    }
}

// The ZundelPES potential now lives in super::potentials::molecular.


// =============================================================================
// Zundel Cation Simulation Driver
// =============================================================================

/// Run PIMD simulation for proton transfer in the Zundel cation H5O2+.
///
/// Compares classical (P=1) and quantum (P=n_beads) behavior.
/// The shared proton (atom 3) transfers between O1 (atom 0) and O2 (atom 4).
pub fn run_pimd_zundel(
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
) {
    let pes = ZundelPES::new();

    // Transfer coordinate: O1 is donor (0), H* is proton (3), O2 is acceptor (4)
    let donor = 0_usize;
    let proton = 3_usize;
    let acceptor = 4_usize;

    let temp_k = 315774.65 / beta;

    println!("==============================================================");
    println!("  Multi-Atom PIMD -- Zundel Cation H5O2+");
    println!("  H2O - H+ ... OH2  <->  H2O ... H+ - OH2");
    println!("  7 atoms, 21 DOF, PILE Thermostat");
    println!("==============================================================");
    println!();
    println!("System: {} -- 7 atoms, 21 DOF", pes.name());
    println!("  O mass:  {:.2} a.u.", pes.masses_arr[0]);
    println!("  H mass:  {:.2} a.u.", pes.masses_arr[1]);
    println!("  O...O eq: {:.4} Bohr ({:.4} A)", pes.r_oo_eq, pes.r_oo_eq * 0.529177);
    println!("  O-H(H2O):  {:.4} Bohr ({:.4} A)", pes.r_oh_h2o, pes.r_oh_h2o * 0.529177);
    println!("  O-H(H3O+): {:.4} Bohr ({:.4} A)", pes.r_oh_h3o, pes.r_oh_h3o * 0.529177);
    println!("  EVB coupling: A={:.4} Ha, mu={:.3} /Bohr", pes.coupling_a, pes.coupling_mu);
    println!("  PES type: Empirical Valence Bond (2-state)");
    println!("  Temp:     {:.1} K (beta = {:.2} a.u.)", temp_k, beta);
    println!();
    println!("Simulation: P={}, replicas={}, dt={:.4}, equil={}, prod={}",
             n_beads, n_polymers, dt, n_equilibrate, n_production);
    println!();

    let gamma = 0.001;

    // =========================================================================
    // Classical (P=1)
    // =========================================================================
    println!("-----------------------------------------------------------");
    println!("  Running CLASSICAL simulation (P = 1)...");
    println!("-----------------------------------------------------------");

    let mut cl_sim = MolecularPIMD::new(n_polymers, 1, beta, dt, gamma, pes.clone());

    for step in 0..n_equilibrate {
        cl_sim.step_obabo();
        if step % (n_equilibrate / 4).max(1) == 0 {
            let tc = cl_sim.average_transfer_coordinate(donor, proton, acceptor);
            println!("  Equil {:6}: E = {:10.6}, d = {:8.5}", step, cl_sim.average_virial_energy(), tc);
        }
    }
    println!();

    let sample_interval = 10;
    let n_hist_bins = 200;
    let hist_min = -2.5;
    let hist_max = 2.5;
    let hist_bw = (hist_max - hist_min) / n_hist_bins as f64;

    let mut cl_e = Vec::new();
    let mut cl_tc = Vec::new();
    let mut cl_hist = vec![0.0; n_hist_bins];
    let mut cl_tun_sum = 0.0;
    let mut cl_tun_n = 0;

    for step in 0..n_production {
        cl_sim.step_obabo();
        if step % sample_interval == 0 {
            cl_e.push(cl_sim.average_virial_energy());
            cl_tc.push(cl_sim.average_transfer_coordinate(donor, proton, acceptor));
            cl_tun_sum += cl_sim.tunneling_fraction(donor, proton, acceptor);
            cl_tun_n += 1;
            cl_sim.accumulate_transfer_histogram(&mut cl_hist, hist_min, hist_max, donor, proton, acceptor);
        }
        if step % (n_production / 5).max(1) == 0 {
            let tc = cl_sim.average_transfer_coordinate(donor, proton, acceptor);
            println!("  Prod {:6}: E = {:10.6}, d = {:8.5}, tunnel = {:.2}%",
                     step, cl_sim.average_virial_energy(), tc,
                     100.0 * cl_sim.tunneling_fraction(donor, proton, acceptor));
        }
    }

    let cl_n = cl_e.len() as f64;
    let cl_mean_e = cl_e.iter().sum::<f64>() / cl_n;
    let cl_mean_tc = cl_tc.iter().sum::<f64>() / cl_n;
    let cl_tunnel = cl_tun_sum / cl_tun_n as f64;

    println!();
    println!("  Classical: E = {:.6} Ha, <d> = {:.5}, tunnel = {:.2}%",
             cl_mean_e, cl_mean_tc, 100.0 * cl_tunnel);

    // =========================================================================
    // Quantum (P = n_beads)
    // =========================================================================
    println!();
    println!("-----------------------------------------------------------");
    println!("  Running QUANTUM simulation (P = {})...", n_beads);
    println!("-----------------------------------------------------------");

    let mut q_sim = MolecularPIMD::new(n_polymers, n_beads, beta, dt, gamma, pes.clone());

    for step in 0..n_equilibrate {
        q_sim.step_obabo();
        if step % (n_equilibrate / 4).max(1) == 0 {
            let tc = q_sim.average_transfer_coordinate(donor, proton, acceptor);
            let rg = q_sim.average_atom_rg(proton);
            println!("  Equil {:6}: E = {:10.6}, d = {:8.5}, R_g(H*) = {:8.5}",
                     step, q_sim.average_virial_energy(), tc, rg);
        }
    }
    println!();

    let mut q_e = Vec::new();
    let mut q_tc = Vec::new();
    let mut q_rg = Vec::new();
    let mut q_rg_par = Vec::new();
    let mut q_rg_perp = Vec::new();
    let mut q_roo = Vec::new();
    let mut q_hist = vec![0.0; n_hist_bins];
    let mut q_tun_sum = 0.0;
    let mut q_tun_n = 0;

    // 2D R_OO vs d correlation histogram
    let n_roo_bins = 50;
    let roo_min = 3.5;
    let roo_max = 6.0;
    let n_tc_bins_2d = 50;
    let mut roo_tc_hist = vec![vec![0.0; n_tc_bins_2d]; n_roo_bins];

    for step in 0..n_production {
        q_sim.step_obabo();
        if step % sample_interval == 0 {
            q_e.push(q_sim.average_virial_energy());
            q_tc.push(q_sim.average_transfer_coordinate(donor, proton, acceptor));
            q_rg.push(q_sim.average_atom_rg(proton));
            q_roo.push(q_sim.average_roo(donor, acceptor));
            let (rg_p, rg_pp) = q_sim.average_proton_rg_decomposed(donor, proton, acceptor);
            q_rg_par.push(rg_p);
            q_rg_perp.push(rg_pp);
            q_tun_sum += q_sim.tunneling_fraction(donor, proton, acceptor);
            q_tun_n += 1;
            q_sim.accumulate_transfer_histogram(&mut q_hist, hist_min, hist_max, donor, proton, acceptor);
            q_sim.accumulate_roo_tc_histogram(
                &mut roo_tc_hist, roo_min, roo_max, n_roo_bins,
                hist_min, hist_max, n_tc_bins_2d, donor, proton, acceptor,
            );
        }
        if step % (n_production / 5).max(1) == 0 {
            let tc = q_sim.average_transfer_coordinate(donor, proton, acceptor);
            let rg = q_sim.average_atom_rg(proton);
            println!("  Prod {:6}: E = {:10.6}, d = {:8.5}, R_g(H*) = {:8.5}, tunnel = {:.2}%",
                     step, q_sim.average_virial_energy(), tc, rg,
                     100.0 * q_sim.tunneling_fraction(donor, proton, acceptor));
        }
    }

    let q_n = q_e.len() as f64;
    let q_mean_e = q_e.iter().sum::<f64>() / q_n;
    let q_var_e = q_e.iter().map(|e| (e - q_mean_e).powi(2)).sum::<f64>() / q_n;
    let q_stderr_e = q_var_e.sqrt() / q_n.sqrt();
    let q_mean_tc = q_tc.iter().sum::<f64>() / q_n;
    let q_mean_rg = q_rg.iter().sum::<f64>() / q_n;
    let q_mean_rg_par = q_rg_par.iter().sum::<f64>() / q_n;
    let q_mean_rg_perp = q_rg_perp.iter().sum::<f64>() / q_n;
    let q_mean_roo = q_roo.iter().sum::<f64>() / q_n;
    let q_tunnel = q_tun_sum / q_tun_n as f64;

    println!();
    println!("  Quantum: E = {:.6} +/- {:.6} Ha, <d> = {:.5}, R_g(H*) = {:.5}, tunnel = {:.2}%",
             q_mean_e, q_stderr_e, q_mean_tc, q_mean_rg, 100.0 * q_tunnel);

    // =========================================================================
    // H/D Isotope Comparison (Deuterium run)
    // =========================================================================
    println!();
    println!("-----------------------------------------------------------");
    println!("  Running DEUTERIUM simulation (P = {})...", n_beads);
    println!("-----------------------------------------------------------");

    let mut pes_d = pes.clone();
    let m_d = 3672.30; // Deuterium mass in a.u.
    pes_d.masses_arr[3] = m_d; // Only H* is deuterated

    let mut d_sim = MolecularPIMD::new(n_polymers, n_beads, beta, dt, gamma, pes_d);

    // Shorter equilibration/production for isotope comparison
    let d_equil = n_equilibrate / 2;
    let d_prod = n_production / 2;

    for step in 0..d_equil {
        d_sim.step_obabo();
        if step % (d_equil / 3).max(1) == 0 {
            println!("  D equil {:6}: E = {:10.6}", step, d_sim.average_virial_energy());
        }
    }
    println!();

    let mut d_e = Vec::new();
    let mut d_hist = vec![0.0; n_hist_bins];
    let mut d_tun_sum = 0.0;
    let mut d_tun_n = 0;
    let mut d_rg = Vec::new();

    for step in 0..d_prod {
        d_sim.step_obabo();
        if step % sample_interval == 0 {
            d_e.push(d_sim.average_virial_energy());
            d_rg.push(d_sim.average_atom_rg(proton));
            d_tun_sum += d_sim.tunneling_fraction(donor, proton, acceptor);
            d_tun_n += 1;
            d_sim.accumulate_transfer_histogram(&mut d_hist, hist_min, hist_max, donor, proton, acceptor);
        }
        if step % (d_prod / 3).max(1) == 0 {
            println!("  D prod {:6}: E = {:10.6}, tunnel = {:.2}%",
                     step, d_sim.average_virial_energy(),
                     100.0 * d_sim.tunneling_fraction(donor, proton, acceptor));
        }
    }

    let d_n = d_e.len() as f64;
    let d_mean_e = d_e.iter().sum::<f64>() / d_n;
    let d_mean_rg = d_rg.iter().sum::<f64>() / d_n;
    let d_tunnel = d_tun_sum / d_tun_n as f64;

    // =========================================================================
    // Comparison Summary
    // =========================================================================
    let cl_roo = cl_sim.average_roo(donor, acceptor);
    let kbt = 1.0 / beta; // k_B T in Hartree

    println!();
    println!("======================================================================");
    println!("                  ZUNDEL CATION -- FULL COMPARISON");
    println!("======================================================================");
    println!("  {:>18} | {:>12} | {:>12} | {:>12}", "Property", "Classical", "Quantum(H)", "Quantum(D)");
    println!("  {:>18} | {:>12} | {:>12} | {:>12}", "------------------", "------------", "------------", "------------");
    println!("  {:>18} | {:>12.6} | {:>12.6} | {:>12.6}", "E_virial (Ha)", cl_mean_e, q_mean_e, d_mean_e);
    println!("  {:>18} | {:>12.5} | {:>12.5} | {:>12}", "<d> (Bohr)", cl_mean_tc, q_mean_tc, "--");
    println!("  {:>18} | {:>12} | {:>12.5} | {:>12.5}", "R_g(H*) (Bohr)", "N/A", q_mean_rg, d_mean_rg);
    println!("  {:>18} | {:>12} | {:>12.5} | {:>12}", "R_g_par (Bohr)", "N/A", q_mean_rg_par, "--");
    println!("  {:>18} | {:>12} | {:>12.5} | {:>12}", "R_g_perp (Bohr)", "N/A", q_mean_rg_perp, "--");
    println!("  {:>18} | {:>12.4} | {:>12.4} | {:>12}", "R_OO (Bohr)", cl_roo, q_mean_roo, "--");
    println!("  {:>18} | {:>11.2}% | {:>11.2}% | {:>11.2}%", "Tunneling", 100.0*cl_tunnel, 100.0*q_tunnel, 100.0*d_tunnel);
    println!("======================================================================");
    println!();

    // Physical interpretation
    if q_tunnel > cl_tunnel + 0.01 {
        println!("  * Quantum proton transfer OBSERVED in Zundel cation!");
        if q_tunnel > 0.35 {
            println!("    -> Strong tunneling: proton significantly shared between both oxygens!");
        } else if q_tunnel > 0.15 {
            println!("    -> Moderate tunneling: clear quantum enhancement.");
        } else {
            println!("    -> Weak but observable quantum effects.");
        }
    } else {
        println!("  o Quantum effects are modest -- try lower temperature or lower barrier.");
    }

    let zpe_diff = q_mean_e - cl_mean_e;
    if zpe_diff > 0.001 {
        println!("  * Zero-point energy: +{:.4} Ha ({:.1} kcal/mol)", zpe_diff, zpe_diff * 627.509);
    }
    if q_mean_rg > 0.05 {
        println!("  * H* R_g = {:.4} Bohr -> quantum delocalization of the shared proton", q_mean_rg);
        println!("    R_g_par = {:.4} (along O-O), R_g_perp = {:.4} (perpendicular)", q_mean_rg_par, q_mean_rg_perp);
    }

    // Isotope effect
    if q_tunnel > 0.01 && d_tunnel > 0.01 {
        let kie = q_tunnel / d_tunnel;
        println!("  * Kinetic isotope effect (H/D): {:.2}x", kie);
        if kie > 1.5 {
            println!("    -> Strong isotope effect: quantum tunneling dominates!");
        }
    }

    // R_OO comparison
    println!("  O...O: cl = {:.4} Bohr ({:.4} A), qm = {:.4} Bohr ({:.4} A)",
             cl_roo, cl_roo * 0.529177, q_mean_roo, q_mean_roo * 0.529177);
    if q_mean_roo < cl_roo - 0.01 {
        println!("    -> Quantum shortening of O...O distance (geometric isotope effect)");
    }

    // =========================================================================
    // Output files
    // =========================================================================
    // Transfer coordinate distribution + free energy
    {
        let file = File::create("pimd_zundel_distribution.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# delta classical_P quantum_H_P quantum_D_P W_cl(Ha) W_qm_H(Ha) W_qm_D(Ha)").unwrap();

        let w_cl = free_energy_profile(&cl_hist, hist_bw, kbt);
        let w_q = free_energy_profile(&q_hist, hist_bw, kbt);
        let w_d = free_energy_profile(&d_hist, hist_bw, kbt);

        let cl_total: f64 = cl_hist.iter().sum();
        let q_total: f64 = q_hist.iter().sum();
        let d_total: f64 = d_hist.iter().sum();

        for i in 0..n_hist_bins {
            let x = hist_min + (i as f64 + 0.5) * hist_bw;
            let cp = if cl_total > 0.0 { cl_hist[i] / (cl_total * hist_bw) } else { 0.0 };
            let qp = if q_total > 0.0 { q_hist[i] / (q_total * hist_bw) } else { 0.0 };
            let dp = if d_total > 0.0 { d_hist[i] / (d_total * hist_bw) } else { 0.0 };
            writeln!(w, "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                     x, cp, qp, dp, w_cl[i], w_q[i], w_d[i]).unwrap();
        }
        println!();
        println!("  Transfer coord + free energy -> pimd_zundel_distribution.txt");
    }
    // Energy trajectory
    {
        let file = File::create("pimd_zundel_energy.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# sample E_cl E_q delta_cl delta_q R_g_H R_g_par R_g_perp R_OO").unwrap();
        let n = q_e.len().min(cl_e.len());
        for i in 0..n {
            writeln!(w, "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                     i, cl_e[i], q_e[i], cl_tc[i], q_tc[i],
                     q_rg[i], q_rg_par[i], q_rg_perp[i], q_roo[i]).unwrap();
        }
        println!("  Energy trajectory -> pimd_zundel_energy.txt");
    }
    // R_OO vs d 2D correlation
    {
        let file = File::create("pimd_zundel_roo_correlation.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# R_OO delta counts (2D histogram)").unwrap();
        let roo_bw = (roo_max - roo_min) / n_roo_bins as f64;
        let tc_bw_2d = (hist_max - hist_min) / n_tc_bins_2d as f64;
        for ri in 0..n_roo_bins {
            for ti in 0..n_tc_bins_2d {
                let roo = roo_min + (ri as f64 + 0.5) * roo_bw;
                let tc = hist_min + (ti as f64 + 0.5) * tc_bw_2d;
                if roo_tc_hist[ri][ti] > 0.0 {
                    writeln!(w, "{:.4} {:.4} {:.1}", roo, tc, roo_tc_hist[ri][ti]).unwrap();
                }
            }
        }
        println!("  R_OO vs d correlation -> pimd_zundel_roo_correlation.txt");
    }
    // Bead snapshot
    {
        let file = File::create("pimd_zundel_beads.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# polymer bead atom x y z").unwrap();
        for (pi, polymer) in q_sim.polymers.iter().enumerate() {
            for (bi, bead) in polymer.positions.iter().enumerate() {
                for a in 0..7 {
                    writeln!(w, "{} {} {} {:.6} {:.6} {:.6}",
                             pi, bi, a, bead[3*a], bead[3*a+1], bead[3*a+2]).unwrap();
                }
            }
        }
        println!("  Bead snapshot -> pimd_zundel_beads.txt");
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;


    /// Simple 3D harmonic potential for testing
    #[derive(Clone)]
    struct Harmonic3D {
        omega: f64,
        mass: f64,
    }

    impl MolecularPotential for Harmonic3D {
        fn n_atoms(&self) -> usize { 1 }

        fn energy(&self, coords: &[f64]) -> f64 {
            let r2: f64 = coords.iter().map(|&x| x * x).sum();
            0.5 * self.mass * self.omega * self.omega * r2
        }

        fn forces(&self, coords: &[f64], forces: &mut [f64]) {
            let k = self.mass * self.omega * self.omega;
            for d in 0..3 {
                forces[d] = -k * coords[d];
            }
        }

        fn masses(&self) -> &[f64] { std::slice::from_ref(&self.mass) }

        fn reference_geometry(&self) -> Vec<f64> { vec![0.0, 0.0, 0.0] }

        fn name(&self) -> &'static str { "3D Harmonic" }
    }

    #[test]
    fn test_3d_harmonic_energy() {
        // 3D harmonic oscillator: E0 = 3/2 hbarw = 1.5 for w=1, m=1
        let pot = Harmonic3D { omega: 1.0, mass: 1.0 };
        let n_beads = 32;
        let beta = 20.0;
        let dt = 0.05;

        let mut sim = MolecularPIMD::new(20, n_beads, beta, dt, 1.0, pot);

        // Equilibrate
        for _ in 0..5000 {
            sim.step_obabo();
        }

        // Sample
        let mut energies = Vec::new();
        for _ in 0..5000 {
            sim.step_obabo();
            energies.push(sim.average_virial_energy());
        }

        let mean_e = energies.iter().sum::<f64>() / energies.len() as f64;
        // 3D ground state energy = 3 x 0.5 = 1.5
        assert!((mean_e - 1.5).abs() < 0.25,
                "3D harmonic energy {:.4} should be near 1.5", mean_e);
    }

    #[test]
    fn test_bifluoride_force_consistency() {
        // Verify numerical forces match finite difference of energy
        let pes = BifluoridePES::new();
        let geom = pes.reference_geometry();
        let ndof = pes.ndof();

        let mut forces = vec![0.0; ndof];
        pes.forces(&geom, &mut forces);

        let h = 1e-5;
        let mut gp = geom.clone();
        let mut gm = geom.clone();

        for d in 0..ndof {
            gp[d] = geom[d] + h;
            gm[d] = geom[d] - h;
            let f_num = -(pes.energy(&gp) - pes.energy(&gm)) / (2.0 * h);
            gp[d] = geom[d];
            gm[d] = geom[d];

            assert!((forces[d] - f_num).abs() < 1e-4,
                    "Force mismatch at dof {}: analytical {:.8} vs numerical {:.8}",
                    d, forces[d], f_num);
        }
    }

    #[test]
    fn test_bifluoride_double_well() {
        // Verify the PES has a double-well structure along x for the proton
        let pes = BifluoridePES::new();

        // Place H at various positions along F1-F2 axis
        let r_ff = pes.r_ff_eq;
        let midpoint = r_ff / 2.0;

        // Energy at midpoint (barrier)
        let geom_ts = vec![0.0, 0.0, 0.0, midpoint, 0.0, 0.0, r_ff, 0.0, 0.0];
        let e_ts = pes.energy(&geom_ts);

        // Energy at left minimum
        let geom_left = vec![0.0, 0.0, 0.0, pes.r_fh_eq, 0.0, 0.0, r_ff, 0.0, 0.0];
        let e_left = pes.energy(&geom_left);

        // Energy at right minimum
        let geom_right = vec![0.0, 0.0, 0.0, r_ff - pes.r_fh_eq, 0.0, 0.0, r_ff, 0.0, 0.0];
        let e_right = pes.energy(&geom_right);

        // Barrier should be above both minima
        assert!(e_ts > e_left, "TS energy ({:.6}) should be above left min ({:.6})", e_ts, e_left);
        assert!(e_ts > e_right, "TS energy ({:.6}) should be above right min ({:.6})", e_ts, e_right);

        // Symmetric: left ~ right
        assert!((e_left - e_right).abs() < 0.001,
                "Wells should be symmetric: left={:.6}, right={:.6}", e_left, e_right);
    }

    #[test]
    fn test_molecular_ring_polymer_centroid() {
        let pes = BifluoridePES::new();
        let rp = MolecularRingPolymer::new(16, 1000.0, pes);
        let cent = rp.centroid();
        // Centroid should exist and be finite
        assert!(cent.iter().all(|&c| c.is_finite()), "Centroid should be finite");
        assert_eq!(cent.len(), 9); // 3 atoms x 3D
    }

    #[test]
    fn test_zundel_double_well() {
        let pes = ZundelPES::new();
        let r_oo = pes.r_oo_eq;
        let ref_geom = pes.reference_geometry();

        // Energy at reference (H* near O1)
        let e_left = pes.energy(&ref_geom);

        // Energy with H* at midpoint (transition state)
        let mut geom_ts = ref_geom.clone();
        geom_ts[3 * 3] = r_oo / 2.0; // H* x = midpoint
        let e_ts = pes.energy(&geom_ts);

        // Energy with H* near O2 (right well)
        let mut geom_right = ref_geom.clone();
        geom_right[3 * 3] = r_oo - pes.r_oh_h3o;
        let e_right = pes.energy(&geom_right);

        // Barrier should be above both minima
        assert!(e_ts > e_left, "TS ({:.6}) should be above left ({:.6})", e_ts, e_left);
        assert!(e_ts > e_right, "TS ({:.6}) should be above right ({:.6})", e_ts, e_right);

        // Wells should be roughly symmetric
        assert!((e_left - e_right).abs() < 0.01,
                "Zundel wells should be ~symmetric: left={:.6}, right={:.6}", e_left, e_right);
    }

    #[test]
    fn test_zundel_force_consistency() {
        let pes = ZundelPES::new();
        let geom = pes.reference_geometry();
        let ndof = pes.ndof();

        let mut forces = vec![0.0; ndof];
        pes.forces(&geom, &mut forces);

        let h = 1e-5;
        let mut gp = geom.clone();
        let mut gm = geom.clone();
        for d in 0..ndof {
            gp[d] = geom[d] + h;
            gm[d] = geom[d] - h;
            let f_num = -(pes.energy(&gp) - pes.energy(&gm)) / (2.0 * h);
            gp[d] = geom[d];
            gm[d] = geom[d];

            assert!((forces[d] - f_num).abs() < 1e-3,
                    "Zundel force mismatch at dof {}: {:.8} vs {:.8}", d, forces[d], f_num);
        }
    }
}
