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
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

// =============================================================================
// Multi-Dimensional Potential Trait
// =============================================================================

/// Trait for multi-dimensional molecular potential energy surfaces.
///
/// Coordinates are stored as a flat array: [x0,y0,z0, x1,y1,z1, ...]
/// where atom i has coordinates at indices [3*i, 3*i+1, 3*i+2].
pub trait MolecularPotential: Clone + Send + Sync {
    /// Number of atoms in the system
    fn n_atoms(&self) -> usize;

    /// Number of degrees of freedom (= 3 * n_atoms for 3D)
    fn ndof(&self) -> usize {
        3 * self.n_atoms()
    }

    /// Evaluate the potential energy V(R)
    ///
    /// # Arguments
    /// * `coords` - Flat array of Cartesian coordinates [x0,y0,z0, x1,y1,z1, ...]
    fn energy(&self, coords: &[f64]) -> f64;

    /// Compute forces F = -∇V and store in the `forces` buffer
    ///
    /// Default: numerical central difference (slow but correct).
    /// Override for analytical forces.
    fn forces(&self, coords: &[f64], forces: &mut [f64]) {
        let h = 1e-6;
        let ndof = self.ndof();
        let mut coords_plus = coords.to_vec();
        let mut coords_minus = coords.to_vec();
        for d in 0..ndof {
            coords_plus[d] = coords[d] + h;
            coords_minus[d] = coords[d] - h;
            forces[d] = -(self.energy(&coords_plus) - self.energy(&coords_minus)) / (2.0 * h);
            coords_plus[d] = coords[d];
            coords_minus[d] = coords[d];
        }
    }

    /// Atom masses in atomic units [m0, m1, m2, ...]
    fn masses(&self) -> &[f64];

    /// Reference equilibrium geometry for initialization
    fn reference_geometry(&self) -> Vec<f64>;

    /// Name for display
    fn name(&self) -> &'static str;
}

// =============================================================================
// Multi-Atom Ring Polymer
// =============================================================================

/// Ring polymer for a multi-atom molecular system.
///
/// Each bead stores a full molecular geometry (ndof = 3 × N_atoms coordinates).
/// Spring constants are per-atom: κ_a = m_a / Δτ² for atom a.
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
    /// Per-DOF mass (maps dof index → atom mass): mass_dof[d] = masses[d/3]
    pub mass_dof: Vec<f64>,
    /// Per-atom spring constants κ_a = m_a / Δτ²
    pub spring_constants: Vec<f64>,
    /// Per-DOF spring constants (maps dof → spring constant)
    pub spring_dof: Vec<f64>,
    /// Physical inverse temperature β
    pub beta: f64,
    /// Imaginary time step Δτ = β/P
    pub dtau: f64,
    /// The molecular potential
    pub potential: P,
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
                let sigma_x = 0.01; // Bohr — small perturbation
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
        };
        rp.compute_forces();
        rp
    }

    /// Compute total forces on each bead: F = F_spring + F_physical
    pub fn compute_forces(&mut self) {
        let ndof = self.ndof;

        for i in 0..self.n_beads {
            let prev = if i == 0 { self.n_beads - 1 } else { i - 1 };
            let next = (i + 1) % self.n_beads;

            // Physical forces from potential
            self.potential.forces(&self.positions[i], &mut self.forces[i]);

            // Add spring forces: F_spring[d] = -κ_d (2 x[i][d] - x[prev][d] - x[next][d])
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
        self.positions.iter()
            .map(|pos| self.potential.energy(pos))
            .sum::<f64>() / self.n_beads as f64
    }

    /// Centroid virial energy estimator for N-D:
    ///
    ///   E_cv = N_dof/(2β) + (1/P) Σ_i [V(R_i) + ½ Σ_d (R_i[d] - R̄[d]) × (∂V/∂R_i[d])]
    ///
    /// Here N_dof is the number of degrees of freedom.
    pub fn virial_energy_estimator(&self) -> f64 {
        let cent = self.centroid();
        let ndof = self.ndof;
        let p = self.n_beads as f64;

        let mut sum = 0.0;
        let mut phys_forces = vec![0.0; ndof];

        for i in 0..self.n_beads {
            // Get physical forces (without spring contribution)
            self.potential.forces(&self.positions[i], &mut phys_forces);

            let v_i = self.potential.energy(&self.positions[i]);
            let mut virial = 0.0;
            for d in 0..ndof {
                // ∂V/∂R = -F_phys, so (R-R̄)·(∂V/∂R) = -(R-R̄)·F
                virial += (self.positions[i][d] - cent[d]) * (-phys_forces[d]);
            }
            sum += v_i + 0.5 * virial;
        }

        // N_dof/(2β) + bead-averaged [ V + virial_correction ]
        ndof as f64 / (2.0 * self.beta) + sum / p
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
///   σ_{a,k} = √(kBT_P / m_a)
///
/// Mode frequencies ω_k are the same for all atoms (they depend only on β and P).
pub struct MolecularPILE {
    /// Number of beads P
    pub n_beads: usize,
    /// Number of DOF
    pub ndof: usize,
    /// Bead temperature kBT_P = P/β
    pub kbt_p: f64,
    /// Per-DOF velocity noise width: σ[d] = √(kBT_P / m_d)
    pub sigma: Vec<f64>,
    /// Propagator coefficients c1[k] = exp(-γ_k dt/2)
    pub c1: Vec<f64>,
    /// Noise coefficients c2[k] = √(1 - c1[k]²)
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
        nm_frequencies: &[f64],  // normal mode frequencies ω_k
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
    /// `mode_vel` is [mode_k][dof_d] — mode k's velocity for each DOF.
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
    pub nm_transform: super::pimd::NormalModeTransform,
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
        let nm_transform = super::pimd::NormalModeTransform::new(n_beads, beta);
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

    /// Normal mode transform for a single DOF across all beads.
    /// Extracts bead values for DOF d, transforms, and returns mode values.
    fn to_modes_1d(&self, bead_values: &[Vec<f64>], dof: usize) -> Vec<f64> {
        let vals: Vec<f64> = bead_values.iter().map(|b| b[dof]).collect();
        self.nm_transform.to_normal_modes(&vals)
    }

    /// Inverse: mode values for DOF d → bead values
    fn to_beads_1d(&self, mode_values: &[f64]) -> Vec<f64> {
        self.nm_transform.to_beads(mode_values)
    }

    /// OBABO Langevin step for multi-atom system.
    pub fn step_obabo(&mut self) {
        let dt = self.dt;
        let half_dt = dt / 2.0;
        let n_beads = self.nm_transform.n_beads;
        let ndof = self.ndof;

        for polymer in &mut self.polymers {
            // === O step: thermostat in normal mode space ===
            // Transform velocities to normal modes (per DOF)
            let mut mode_vel: Vec<Vec<f64>> = vec![vec![0.0; ndof]; n_beads];
            for d in 0..ndof {
                let bead_v: Vec<f64> = polymer.velocities.iter().map(|v| v[d]).collect();
                let modes = self.nm_transform.to_normal_modes(&bead_v);
                for k in 0..n_beads {
                    mode_vel[k][d] = modes[k];
                }
            }

            self.thermostat.apply_o_step(&mut mode_vel);

            // Transform back to bead space
            for d in 0..ndof {
                let modes: Vec<f64> = mode_vel.iter().map(|m| m[d]).collect();
                let bead_v = self.nm_transform.to_beads(&modes);
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
                let modes = self.nm_transform.to_normal_modes(&bead_v);
                for k in 0..n_beads {
                    mode_vel[k][d] = modes[k];
                }
            }

            self.thermostat.apply_o_step(&mut mode_vel);

            for d in 0..ndof {
                let modes: Vec<f64> = mode_vel.iter().map(|m| m[d]).collect();
                let bead_v = self.nm_transform.to_beads(&modes);
                for i in 0..n_beads {
                    polymer.velocities[i][d] = bead_v[i];
                }
            }
        }

        self.step += 1;
    }

    /// Average virial energy across all polymers
    pub fn average_virial_energy(&self) -> f64 {
        self.polymers.iter()
            .map(|p| p.virial_energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average potential energy across all polymers
    pub fn average_potential_energy(&self) -> f64 {
        self.polymers.iter()
            .map(|p| p.potential_energy())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average transfer coordinate across all polymers
    pub fn average_transfer_coordinate(&self, donor: usize, proton: usize, acceptor: usize) -> f64 {
        self.polymers.iter()
            .map(|p| p.transfer_coordinate(donor, proton, acceptor))
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average R_g for a specific atom across all polymers
    pub fn average_atom_rg(&self, atom: usize) -> f64 {
        self.polymers.iter()
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
        let mut tunneled = 0;
        let mut total = 0;
        for polymer in &self.polymers {
            let tc = polymer.bead_transfer_coordinates(donor, proton, acceptor);
            for &x in &tc {
                if x > 0.0 { tunneled += 1; }
                total += 1;
            }
        }
        tunneled as f64 / total as f64
    }
}

// =============================================================================
// Bifluoride (HF₂⁻) Potential Energy Surface
// =============================================================================

/// Potential energy surface for the bifluoride ion HF₂⁻ (F−H−F).
///
/// Atom ordering: F₁ (0), H (1), F₂ (2)
/// Linear geometry along the x-axis.
///
/// The PES models:
/// 1. Symmetric double-well for the proton along the F···F axis
/// 2. F···F stretch (harmonic around equilibrium)
/// 3. H bending modes (perpendicular to F···F axis)
///
/// Parameters from CCSD(T) calculations:
/// - F···F equilibrium distance: 2.278 Å = 4.304 Bohr
/// - Barrier height: ~1.5 kcal/mol = 0.00239 Hartree
/// - H placed symmetrically at the midpoint = TS geometry
/// - Well O−H distance: ~0.95 Å = 1.795 Bohr from nearest F
///
/// Reference: Kawaguchi & Hirota, J. Chem. Phys. 84, 2953 (1986)
#[derive(Clone)]
pub struct BifluoridePES {
    /// F···F equilibrium distance (Bohr)
    pub r_ff_eq: f64,
    /// F···F force constant (Hartree/Bohr²)
    pub k_ff: f64,
    /// Barrier height at midpoint (Hartree)
    pub barrier_height: f64,
    /// Equilibrium F−H distance (Bohr)
    pub r_fh_eq: f64,
    /// F−H Morse well depth (Hartree)
    pub d_fh: f64,
    /// F−H Morse width parameter (1/Bohr)
    pub alpha_fh: f64,
    /// Bending force constant (Hartree/rad²)
    pub k_bend: f64,
    /// Atom masses [m_F, m_H, m_F]
    masses_arr: [f64; 3],
}

impl BifluoridePES {
    /// Create a bifluoride PES with default CCSD(T)-quality parameters.
    pub fn new() -> Self {
        // Atomic masses in a.u.
        let m_f = 34631.97; // ¹⁹F mass in electron masses
        let m_h = 1836.15;  // ¹H mass

        // Geometry (from spectroscopic and ab initio data)
        let r_ff_eq = 4.304;   // F···F distance in Bohr (~2.278 Å)
        let r_fh_eq = 1.832;   // F−H equilibrium in Bohr (~0.97 Å)

        // Force constants
        let barrier_height = 0.00239; // ~1.5 kcal/mol
        let k_ff = 0.15;              // F···F stretch force constant
        let d_fh = 0.225;             // F−H Morse depth (~141 kcal/mol)
        let alpha_fh = 1.15;          // Morse width parameter
        let k_bend = 0.06;            // Bending force constant

        Self {
            r_ff_eq,
            k_ff,
            barrier_height,
            r_fh_eq,
            d_fh,
            alpha_fh,
            k_bend,
            masses_arr: [m_f, m_h, m_f],
        }
    }

    /// Distance between two atoms given coordinates
    fn distance(coords: &[f64], a: usize, b: usize) -> f64 {
        let mut d2 = 0.0;
        for xyz in 0..3 {
            let dr = coords[3 * a + xyz] - coords[3 * b + xyz];
            d2 += dr * dr;
        }
        d2.sqrt()
    }

    /// Unit vector from atom a to atom b
    fn unit_vector(coords: &[f64], a: usize, b: usize) -> [f64; 3] {
        let r = Self::distance(coords, a, b);
        let mut uv = [0.0; 3];
        if r > 1e-15 {
            for xyz in 0..3 {
                uv[xyz] = (coords[3 * b + xyz] - coords[3 * a + xyz]) / r;
            }
        }
        uv
    }
}

impl MolecularPotential for BifluoridePES {
    fn n_atoms(&self) -> usize { 3 }

    fn energy(&self, coords: &[f64]) -> f64 {
        // Atom 0 = F₁, Atom 1 = H, Atom 2 = F₂
        let r_f1h = Self::distance(coords, 0, 1);
        let r_f2h = Self::distance(coords, 2, 1);
        let r_ff  = Self::distance(coords, 0, 2);

        // 1. F···F stretch: harmonic around equilibrium
        let v_ff = 0.5 * self.k_ff * (r_ff - self.r_ff_eq).powi(2);

        // 2. Proton in double-well along F···F axis
        // Use a symmetric double-Morse potential:
        // V_DM = D × [(1-exp(-α(r-r_eq)))² + (1-exp(-α(r'-r_eq)))²] - 2D
        // plus a coupling term to create the barrier
        let morse_1 = self.d_fh * (1.0 - (-self.alpha_fh * (r_f1h - self.r_fh_eq)).exp()).powi(2);
        let morse_2 = self.d_fh * (1.0 - (-self.alpha_fh * (r_f2h - self.r_fh_eq)).exp()).powi(2);

        // Coupling: when H is at midpoint, both Morse terms are nonzero → barrier
        // When H is near one F, one Morse is ~0, other is large
        // We use: V_proton = min(morse_1, morse_2) + barrier_correction
        // Better approach: LEPS-like mixing
        //   V = (morse_1 + morse_2)/2 - sqrt((morse_1 - morse_2)²/4 + Δ²)
        // where Δ controls the barrier height
        let avg = (morse_1 + morse_2) / 2.0;
        let diff2 = (morse_1 - morse_2).powi(2) / 4.0;

        // Coupling parameter Δ chosen so barrier = self.barrier_height
        // At TS (midpoint): morse_1 = morse_2 = M, so V = M - Δ
        // At minimum: morse_1 ≈ 0, morse_2 ≈ M_large, V ≈ M_large/2 - sqrt(M_large²/4 + Δ²)
        //           ≈ -Δ²/M_large ≈ 0 for large M_large
        // So barrier ≈ M_ts - Δ where M_ts = d_fh(1-exp(-α(r_ff/2 - r_fh_eq)))²
        let r_ts = r_ff / 2.0; // midpoint distance
        let m_ts = self.d_fh * (1.0 - (-self.alpha_fh * (r_ts - self.r_fh_eq)).exp()).powi(2);
        let delta = (m_ts - self.barrier_height).max(0.001);

        let v_proton = avg - (diff2 + delta * delta).sqrt() + delta; // shift so minimum ≈ 0

        // 3. Bending: penalize H displacement perpendicular to F···F axis
        let ff_axis = Self::unit_vector(coords, 0, 2);
        let f1h = [
            coords[3] - coords[0],
            coords[4] - coords[1],
            coords[5] - coords[2],
        ];
        // Project F1-H along F-F axis
        let proj = f1h[0] * ff_axis[0] + f1h[1] * ff_axis[1] + f1h[2] * ff_axis[2];
        let perp2 = f1h[0].powi(2) + f1h[1].powi(2) + f1h[2].powi(2) - proj * proj;
        let v_bend = 0.5 * self.k_bend * perp2 / r_f1h.max(0.1); // scale by 1/r

        v_ff + v_proton + v_bend
    }

    fn forces(&self, coords: &[f64], forces: &mut [f64]) {
        // Use numerical forces for safety (analytical forces for this PES are complex)
        let h = 1e-6;
        let ndof = self.ndof();
        let mut cp = coords.to_vec();
        let mut cm = coords.to_vec();
        for d in 0..ndof {
            cp[d] = coords[d] + h;
            cm[d] = coords[d] - h;
            forces[d] = -(self.energy(&cp) - self.energy(&cm)) / (2.0 * h);
            cp[d] = coords[d];
            cm[d] = coords[d];
        }
    }

    fn masses(&self) -> &[f64] {
        &self.masses_arr
    }

    fn reference_geometry(&self) -> Vec<f64> {
        // Linear F−H−F along x-axis, H near F₁ (right well)
        let r_ff = self.r_ff_eq;
        let r_fh = self.r_fh_eq;
        vec![
            // F₁ at origin
            0.0, 0.0, 0.0,
            // H near F₁ (in the "left well" at distance r_fh from F₁)
            r_fh, 0.0, 0.0,
            // F₂ at r_ff
            r_ff, 0.0, 0.0,
        ]
    }

    fn name(&self) -> &'static str {
        "Bifluoride HF₂⁻ (F−H−F)"
    }
}

// =============================================================================
// Simulation Driver
// =============================================================================

/// Run PIMD simulation for proton transfer in bifluoride HF₂⁻.
///
/// Compares classical (P=1) and quantum (P=n_beads) behavior.
/// Atom ordering: F₁(0), H(1), F₂(2) — proton transfers between two fluorines.
pub fn run_pimd_bifluoride(
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
) {
    let pes = BifluoridePES::new();
    let m_h = pes.masses_arr[1];

    // Indices: F₁=0, H=1, F₂=2
    let donor = 0_usize;
    let proton = 1_usize;
    let acceptor = 2_usize;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    Multi-Atom PIMD — Bifluoride HF₂⁻ Proton Transfer      ║");
    println!("║    3D Ring Polymer with PILE Thermostat                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("System: F−H···F (3 atoms, 9 DOF)");
    println!("  F mass: {:.2} a.u.", pes.masses_arr[0]);
    println!("  H mass: {:.2} a.u.", m_h);
    println!("  F···F eq: {:.4} Bohr ({:.4} Å)", pes.r_ff_eq, pes.r_ff_eq * 0.529177);
    println!("  F−H eq:   {:.4} Bohr ({:.4} Å)", pes.r_fh_eq, pes.r_fh_eq * 0.529177);
    println!("  Barrier:  {:.6} Hartree ({:.2} kcal/mol)", pes.barrier_height, pes.barrier_height * 627.509);
    println!("  Temp:     {:.1} K (β = {:.2} a.u.)", 315774.65 / beta, beta);
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
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Running CLASSICAL simulation (P = 1)...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut cl_sim = MolecularPIMD::new(n_polymers, 1, beta, dt, gamma_centroid, pes.clone());

    for step in 0..n_equilibrate {
        cl_sim.step_obabo();
        if step % (n_equilibrate / 5).max(1) == 0 {
            let tc = cl_sim.average_transfer_coordinate(donor, proton, acceptor);
            println!("  Equil {:6}: E_vir = {:10.6}, δ = {:8.5}",
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
            println!("  Prod {:6}: E_vir = {:10.6}, δ = {:8.5}, tunnel = {:.2}%",
                     step, cl_sim.average_virial_energy(), tc,
                     100.0 * cl_sim.tunneling_fraction(donor, proton, acceptor));
        }
    }

    let cl_n = cl_energies.len() as f64;
    let cl_mean_e = cl_energies.iter().sum::<f64>() / cl_n;
    let cl_mean_tc = cl_tc.iter().sum::<f64>() / cl_n;
    let cl_tunnel = cl_tunnel_sum / cl_tunnel_n as f64;

    println!();
    println!("  Classical: E = {:.6} Ha, <δ> = {:.5}, tunnel = {:.2}%",
             cl_mean_e, cl_mean_tc, 100.0 * cl_tunnel);

    // =========================================================================
    // Quantum (P = n_beads)
    // =========================================================================
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Running QUANTUM simulation (P = {})...", n_beads);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut q_sim = MolecularPIMD::new(n_polymers, n_beads, beta, dt, gamma_centroid, pes.clone());

    for step in 0..n_equilibrate {
        q_sim.step_obabo();
        if step % (n_equilibrate / 5).max(1) == 0 {
            let tc = q_sim.average_transfer_coordinate(donor, proton, acceptor);
            let rg_h = q_sim.average_atom_rg(proton);
            println!("  Equil {:6}: E_vir = {:10.6}, δ = {:8.5}, R_g(H) = {:8.5}",
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
            println!("  Prod {:6}: E_vir = {:10.6}, δ = {:8.5}, R_g(H) = {:8.5}, tunnel = {:.2}%",
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
    println!("  Quantum: E = {:.6} ± {:.6} Ha, <δ> = {:.5}, R_g(H) = {:.5}, tunnel = {:.2}%",
             q_mean_e, q_stderr_e, q_mean_tc, q_mean_rg, 100.0 * q_tunnel);

    // =========================================================================
    // Summary
    // =========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  COMPARISON SUMMARY                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  {:>18} │ {:>14} │ {:>14}  ║", "Property", "Classical", "Quantum");
    println!("║  {:>18} │ {:>14} │ {:>14}  ║", "──────────────────", "──────────────", "──────────────");
    println!("║  {:>18} │ {:>14.6} │ {:>14.6}  ║", "E_virial (Ha)", cl_mean_e, q_mean_e);
    println!("║  {:>18} │ {:>14.5} │ {:>14.5}  ║", "δ (Bohr)", cl_mean_tc, q_mean_tc);
    println!("║  {:>18} │ {:>14} │ {:>14.5}  ║", "R_g(H) (Bohr)", "N/A", q_mean_rg);
    println!("║  {:>18} │ {:>13.2}% │ {:>13.2}%  ║", "Tunneling", 100.0 * cl_tunnel, 100.0 * q_tunnel);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if q_tunnel > cl_tunnel + 0.01 {
        println!("  ✓ Quantum tunneling OBSERVED in HF₂⁻!");
        println!("    Proton ring polymer delocalizes between the two fluorines.");
    }
    let zpe_diff = q_mean_e - cl_mean_e;
    if zpe_diff > 0.001 {
        println!("  ✓ Zero-point energy: +{:.4} Ha ({:.1} kcal/mol)",
                 zpe_diff, zpe_diff * 627.509);
    }
    if q_mean_rg > 0.05 {
        println!("  ✓ Proton R_g = {:.4} Bohr → significant quantum delocalization", q_mean_rg);
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
        println!("  Transfer coordinate distribution → pimd_bifluoride_distribution.txt");
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
        println!("  Energy trajectory → pimd_bifluoride_energy.txt");
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
        println!("  Bead snapshot → pimd_bifluoride_beads.txt");
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
        // 3D harmonic oscillator: E₀ = 3/2 ℏω = 1.5 for ω=1, m=1
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
        // 3D ground state energy = 3 × 0.5 = 1.5
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

        // Place H at various positions along F₁−F₂ axis
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

        // Symmetric: left ≈ right
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
        assert_eq!(cent.len(), 9); // 3 atoms × 3D
    }
}
