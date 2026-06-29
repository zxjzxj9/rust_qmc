//! Well-Tempered Metadynamics with Path Integral Molecular Dynamics
//!
//! Implements adaptive biasing along a collective variable (CV) by depositing
//! Gaussian hills during the simulation. The bias potential progressively
//! fills free energy wells, enabling barrier crossing and reconstruction
//! of the free energy surface (FES).
//!
//! Well-tempered variant: hill heights decay as bias accumulates,
//! ensuring convergence to the true FES:
//!   w_k = w₀ × exp(-V_bias(s_k) / (kT × (γ - 1)))
//!
//! The FES is recovered via:  W(s) = -(γ/(γ-1)) × V_bias(s)
//!
//! Combined with PIMD, the metadynamics bias acts on the centroid CV
//! but applies forces to each bead via the chain rule, exactly as in
//! the umbrella sampling implementation.
//!
//! References:
//!   - Laio & Parrinello, PNAS 99, 12562 (2002)  -- Metadynamics
//!   - Barducci, Bussi, Parrinello, PRL 100, 020603 (2008)  -- Well-tempered
//!   - Marx & Parrinello, JCP 104, 4077 (1996) -- PIMD + biased sampling

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, RwLock};

use super::potentials::MolecularPotential;
use super::pimd_molecular::MolecularPIMD;

// =============================================================================
// Gaussian Hill
// =============================================================================

/// A single Gaussian hill deposited during metadynamics.
#[derive(Clone, Debug)]
pub struct GaussianHill {
    /// CV value where the hill was deposited
    pub center: f64,
    /// Effective height (decays for well-tempered metadynamics)
    pub height: f64,
    /// Width parameter σ
    pub sigma: f64,
}

impl GaussianHill {
    /// Evaluate the Gaussian hill at CV value s:
    ///   h × exp(-(s - s₀)² / (2σ²))
    #[inline]
    pub fn energy(&self, s: f64) -> f64 {
        let ds = s - self.center;
        self.height * (-ds * ds / (2.0 * self.sigma * self.sigma)).exp()
    }

    /// Force (negative derivative) of the Gaussian hill:
    ///   -dV/ds = h × (s - s₀)/σ² × exp(-(s - s₀)² / (2σ²))
    #[inline]
    pub fn force(&self, s: f64) -> f64 {
        let ds = s - self.center;
        let sig2 = self.sigma * self.sigma;
        self.height * ds / sig2 * (-ds * ds / (2.0 * sig2)).exp()
    }
}

// =============================================================================
// Metadynamics Bias
// =============================================================================

/// Adaptive metadynamics bias potential.
///
/// Stores deposited Gaussian hills and provides energy/force evaluation.
/// Supports well-tempered variant with decaying hill heights.
///
/// For grid-accelerated evaluation, the bias is periodically rebuilt on
/// a dense grid for O(1) lookup instead of O(N_hills).
#[derive(Clone)]
pub struct MetadynamicsBias {
    /// All deposited Gaussian hills
    pub hills: Vec<GaussianHill>,
    /// Initial hill height w₀ (Hartree)
    pub initial_height: f64,
    /// Hill width σ (Bohr)
    pub sigma: f64,
    /// Well-tempered bias factor γ (dimensionless, > 1)
    /// γ → ∞ recovers standard metadynamics
    pub bias_factor: f64,
    /// Thermal energy kT (Hartree) = 1/β
    pub kt: f64,

    // --- Grid acceleration ---
    /// Grid values for fast bias evaluation
    grid_energy: Vec<f64>,
    /// Grid forces for fast bias evaluation
    grid_force: Vec<f64>,
    /// Grid minimum CV value
    grid_min: f64,
    /// Grid maximum CV value
    grid_max: f64,
    /// Grid spacing
    grid_dx: f64,
    /// Number of grid points
    grid_n: usize,
    /// Number of hills when grid was last rebuilt
    grid_hills_count: usize,
    /// Whether to use grid acceleration
    use_grid: bool,
}

impl MetadynamicsBias {
    /// Create a new metadynamics bias.
    ///
    /// # Arguments
    /// * `initial_height` - Height w₀ of the first hill (Hartree)
    /// * `sigma` - Width σ of Gaussian hills (Bohr)
    /// * `bias_factor` - Well-tempered parameter γ (typically 5-30)
    /// * `kt` - Thermal energy kT = 1/β (Hartree)
    /// * `grid_min`, `grid_max` - CV range for grid acceleration
    /// * `grid_n` - Number of grid points (0 to disable grid)
    pub fn new(
        initial_height: f64,
        sigma: f64,
        bias_factor: f64,
        kt: f64,
        grid_min: f64,
        grid_max: f64,
        grid_n: usize,
    ) -> Self {
        let use_grid = grid_n > 1;
        let grid_dx = if use_grid {
            (grid_max - grid_min) / (grid_n - 1) as f64
        } else {
            1.0
        };

        Self {
            hills: Vec::new(),
            initial_height,
            sigma,
            bias_factor,
            kt,
            grid_energy: vec![0.0; grid_n],
            grid_force: vec![0.0; grid_n],
            grid_min,
            grid_max,
            grid_dx,
            grid_n,
            grid_hills_count: 0,
            use_grid,
        }
    }

    /// Deposit a new Gaussian hill at the current CV value.
    ///
    /// For well-tempered metadynamics, the height is scaled:
    ///   w = w₀ × exp(-V_bias(s) / (kT × (γ - 1)))
    pub fn deposit_hill(&mut self, s: f64) {
        let current_bias = self.energy_direct(s);

        // Well-tempered scaling
        let height = if self.bias_factor > 1.0 + 1e-10 {
            let delta_t = self.kt * (self.bias_factor - 1.0);
            self.initial_height * (-current_bias / delta_t).exp()
        } else {
            // Standard metadynamics (γ → ∞ limit)
            self.initial_height
        };

        let hill = GaussianHill {
            center: s,
            height,
            sigma: self.sigma,
        };

        // Update grid incrementally (add just this hill)
        if self.use_grid {
            for i in 0..self.grid_n {
                let s_grid = self.grid_min + i as f64 * self.grid_dx;
                self.grid_energy[i] += hill.energy(s_grid);
                self.grid_force[i] += hill.force(s_grid);
            }
            self.grid_hills_count = self.hills.len() + 1;
        }

        self.hills.push(hill);
    }

    /// Evaluate the total bias energy by direct summation over all hills.
    /// O(N_hills) cost.
    pub fn energy_direct(&self, s: f64) -> f64 {
        self.hills.iter().map(|h| h.energy(s)).sum()
    }

    /// Evaluate the total bias force by direct summation.
    /// O(N_hills) cost.
    pub fn force_direct(&self, s: f64) -> f64 {
        self.hills.iter().map(|h| h.force(s)).sum()
    }

    /// Evaluate bias energy using grid interpolation (linear).
    /// O(1) cost.
    fn energy_grid(&self, s: f64) -> f64 {
        if s <= self.grid_min {
            return self.grid_energy[0];
        }
        if s >= self.grid_max {
            return self.grid_energy[self.grid_n - 1];
        }
        let x = (s - self.grid_min) / self.grid_dx;
        let i = x as usize;
        let frac = x - i as f64;
        if i + 1 < self.grid_n {
            self.grid_energy[i] * (1.0 - frac) + self.grid_energy[i + 1] * frac
        } else {
            self.grid_energy[i]
        }
    }

    /// Evaluate bias force using grid interpolation.
    fn force_grid(&self, s: f64) -> f64 {
        if s <= self.grid_min {
            return self.grid_force[0];
        }
        if s >= self.grid_max {
            return self.grid_force[self.grid_n - 1];
        }
        let x = (s - self.grid_min) / self.grid_dx;
        let i = x as usize;
        let frac = x - i as f64;
        if i + 1 < self.grid_n {
            self.grid_force[i] * (1.0 - frac) + self.grid_force[i + 1] * frac
        } else {
            self.grid_force[i]
        }
    }

    /// Evaluate bias energy (auto-selects grid vs direct).
    pub fn energy(&self, s: f64) -> f64 {
        if self.use_grid && self.grid_hills_count == self.hills.len() {
            self.energy_grid(s)
        } else {
            self.energy_direct(s)
        }
    }

    /// Evaluate bias force = -dV_bias/ds (auto-selects grid vs direct).
    pub fn force(&self, s: f64) -> f64 {
        if self.use_grid && self.grid_hills_count == self.hills.len() {
            self.force_grid(s)
        } else {
            self.force_direct(s)
        }
    }

    /// Reconstruct the free energy surface from the well-tempered bias.
    ///
    /// W(s) = -(γ/(γ-1)) × V_bias(s)
    ///
    /// Returns (cv_values, fes_values) on a uniform grid.
    pub fn reconstruct_fes(&self, s_min: f64, s_max: f64, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        let ds = (s_max - s_min) / (n_points - 1) as f64;
        let factor = if self.bias_factor > 1.0 + 1e-10 {
            -self.bias_factor / (self.bias_factor - 1.0)
        } else {
            -1.0 // Standard metadynamics
        };

        let mut cv_values = Vec::with_capacity(n_points);
        let mut fes_values = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let s = s_min + i as f64 * ds;
            cv_values.push(s);
            fes_values.push(factor * self.energy_direct(s));
        }

        // Shift so minimum is zero
        let min_val = fes_values.iter().cloned().fold(f64::INFINITY, f64::min);
        for v in &mut fes_values {
            *v -= min_val;
        }

        (cv_values, fes_values)
    }

    /// Number of deposited hills
    pub fn n_hills(&self) -> usize {
        self.hills.len()
    }

    /// Current effective hill height (would-be height if a hill were deposited at s)
    pub fn current_height_at(&self, s: f64) -> f64 {
        if self.bias_factor > 1.0 + 1e-10 {
            let delta_t = self.kt * (self.bias_factor - 1.0);
            self.initial_height * (-self.energy_direct(s) / delta_t).exp()
        } else {
            self.initial_height
        }
    }
}

// =============================================================================
// Metadynamics Potential Wrapper
// =============================================================================

/// Wraps a `MolecularPotential` with an adaptive metadynamics bias
/// along the proton transfer coordinate δ = d(donor-H*) - d(acceptor-H*).
///
/// The bias is stored in a shared `Arc<RwLock<MetadynamicsBias>>` so that
/// all clones (one per bead per polymer) share the same growing bias.
/// Hills are deposited externally between PIMD steps, so the `RwLock`
/// is only read-locked during force evaluation.
pub struct MetadynamicsPotential<P: MolecularPotential> {
    /// The underlying physical potential
    pub inner: P,
    /// Shared metadynamics bias (read during force eval, written during hill deposit)
    pub bias: Arc<RwLock<MetadynamicsBias>>,
    /// Index of the donor atom (e.g., O₁)
    pub donor: usize,
    /// Index of the proton being transferred (e.g., H*)
    pub proton: usize,
    /// Index of the acceptor atom (e.g., O₂)
    pub acceptor: usize,
}

impl<P: MolecularPotential> Clone for MetadynamicsPotential<P> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            bias: Arc::clone(&self.bias),
            donor: self.donor,
            proton: self.proton,
            acceptor: self.acceptor,
        }
    }
}

// Safety: MetadynamicsPotential is Send + Sync because:
// - P: MolecularPotential requires Send + Sync
// - Arc<RwLock<T>> is Send + Sync when T: Send + Sync
// - MetadynamicsBias: Send + Sync (only contains Vec, f64, etc.)
unsafe impl<P: MolecularPotential> Send for MetadynamicsPotential<P> {}
unsafe impl<P: MolecularPotential> Sync for MetadynamicsPotential<P> {}

impl<P: MolecularPotential> MetadynamicsPotential<P> {
    /// Create a new metadynamics-biased potential.
    pub fn new(
        inner: P,
        bias: Arc<RwLock<MetadynamicsBias>>,
        donor: usize,
        proton: usize,
        acceptor: usize,
    ) -> Self {
        Self { inner, bias, donor, proton, acceptor }
    }

    /// Compute the transfer coordinate δ = d(donor-proton) - d(acceptor-proton)
    pub fn transfer_coordinate(&self, coords: &[f64]) -> f64 {
        let d_ah = Self::atom_distance(coords, self.donor, self.proton);
        let d_bh = Self::atom_distance(coords, self.acceptor, self.proton);
        d_ah - d_bh
    }

    /// Distance between two atoms
    fn atom_distance(coords: &[f64], a: usize, b: usize) -> f64 {
        let mut d2 = 0.0;
        for xyz in 0..3 {
            let dr = coords[3 * a + xyz] - coords[3 * b + xyz];
            d2 += dr * dr;
        }
        d2.sqrt()
    }

    /// Compute ∂δ/∂R_i for all coordinates (same as umbrella sampling).
    ///
    /// δ = r_AH - r_BH
    ///   ∂δ/∂R_A = (R_A - R_H) / r_AH
    ///   ∂δ/∂R_B = -(R_B - R_H) / r_BH
    ///   ∂δ/∂R_H = (R_H - R_A)/r_AH - (R_H - R_B)/r_BH
    fn transfer_coordinate_gradient(&self, coords: &[f64], grad: &mut [f64]) {
        let ndof = self.inner.ndof();
        for g in grad[..ndof].iter_mut() {
            *g = 0.0;
        }

        let r_ah = Self::atom_distance(coords, self.donor, self.proton);
        let r_bh = Self::atom_distance(coords, self.acceptor, self.proton);

        if r_ah < 1e-15 || r_bh < 1e-15 {
            return;
        }

        let a = self.donor;
        let h = self.proton;
        let b = self.acceptor;

        for xyz in 0..3 {
            let ra = coords[3 * a + xyz];
            let rh = coords[3 * h + xyz];
            let rb = coords[3 * b + xyz];

            grad[3 * a + xyz] = (ra - rh) / r_ah;
            grad[3 * b + xyz] = -(rb - rh) / r_bh;
            grad[3 * h + xyz] = (rh - ra) / r_ah - (rh - rb) / r_bh;
        }
    }
}

impl<P: MolecularPotential> MolecularPotential for MetadynamicsPotential<P> {
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    fn energy(&self, coords: &[f64]) -> f64 {
        let e_phys = self.inner.energy(coords);
        let delta = self.transfer_coordinate(coords);
        let bias = self.bias.read().unwrap();
        let e_bias = bias.energy(delta);
        e_phys + e_bias
    }

    /// Forces: F_total = F_physical + F_bias
    ///
    /// F_bias_i = -∂V_bias/∂R_i = -dV_bias/dδ × ∂δ/∂R_i
    ///
    /// Note: dV_bias/dδ is the *derivative*, so bias.force() = -dV/dδ,
    /// and we compute F_bias_i = force_along_cv × ∂δ/∂R_i.
    /// Wait -- bias.force() returns the positive force = -(dV/ds),
    /// so the contribution to Cartesian forces is:
    ///   F_bias_i = -(dV_bias/ds) × (∂s/∂R_i) = bias.force(s) × (∂δ/∂R_i)
    /// This is WRONG — we need:
    ///   F_i = -∂V_bias/∂R_i = -(dV_bias/ds) × (∂s/∂R_i)
    /// The GaussianHill::force returns +(dV/ds)×(s-s_k)/σ² which is NOT -dV/ds.
    /// Let me reconsider:
    ///   GaussianHill::force(s) = h × (s-s₀)/σ² × exp(...) = -(-dV/ds) ... no.
    ///   V = h × exp(-(s-s₀)²/(2σ²))
    ///   dV/ds = h × (-(s-s₀)/σ²) × exp(-(s-s₀)²/(2σ²)) = -h(s-s₀)/σ² × exp(...)
    ///   force = -dV/ds = h(s-s₀)/σ² × exp(...)
    /// So GaussianHill::force IS -dV/ds (the 1D force). Good.
    /// Then: F_bias_i = -(dV_bias/dδ) × (∂δ/∂R_i) = bias.force(δ) × (∂δ/∂R_i)
    /// Wait:  ∂V_bias/∂R_i = (dV_bias/dδ) × (∂δ/∂R_i)
    ///        F_i = -∂V_bias/∂R_i = -(dV_bias/dδ) × (∂δ/∂R_i)
    ///        bias.force(δ) = -(dV_bias/dδ)
    ///        So: F_i = bias.force(δ) × (∂δ/∂R_i)
    /// Wait no, this gives forces[d] += f_cv * grad[d] which adds the bias force.
    /// But the umbrella code uses forces[d] -= dv_ddelta * grad[d], where
    /// dv_ddelta = κ(δ-δ₀) = dV/dδ.
    /// So forces[d] -= (dV/dδ) × (∂δ/∂R_d) which is F = -∇V. Correct.
    /// Here: dV_bias/dδ = -bias.force(δ), so:
    ///   forces[d] -= (-bias.force(δ)) × grad[d]
    ///   forces[d] += bias.force(δ) × grad[d]
    fn forces(&self, coords: &[f64], forces: &mut [f64]) {
        // Physical forces
        self.inner.forces(coords, forces);

        // Bias forces via chain rule
        let ndof = self.inner.ndof();
        let delta = self.transfer_coordinate(coords);

        let bias = self.bias.read().unwrap();
        // dV_bias/dδ = -(bias.force(δ))
        let dv_ddelta = -bias.force(delta);
        drop(bias); // Release lock early

        let mut ddelta_dr = vec![0.0; ndof];
        self.transfer_coordinate_gradient(coords, &mut ddelta_dr);

        for d in 0..ndof {
            forces[d] -= dv_ddelta * ddelta_dr[d];
        }
    }

    fn masses(&self) -> &[f64] {
        self.inner.masses()
    }

    fn reference_geometry(&self) -> Vec<f64> {
        self.inner.reference_geometry()
    }

    fn name(&self) -> &'static str {
        "Metadynamics-Biased"
    }
}

// =============================================================================
// Metadynamics Result
// =============================================================================

/// Results from a PIMD metadynamics simulation.
pub struct MetadynamicsResult {
    /// Deposited hills
    pub hills: Vec<GaussianHill>,
    /// CV time series (centroid δ at each sample point)
    pub cv_trajectory: Vec<f64>,
    /// Energy time series
    pub energy_trajectory: Vec<f64>,
    /// Step indices corresponding to trajectory entries
    pub step_indices: Vec<usize>,
    /// Reconstructed FES: (cv_values, fes_values) in Hartree
    pub fes_cv: Vec<f64>,
    pub fes_values: Vec<f64>,
    /// Label for this run
    pub label: String,
}

// =============================================================================
// Driver Functions
// =============================================================================

/// Run PIMD metadynamics along the proton transfer coordinate.
///
/// # Arguments
/// * `potential` - Physical potential energy surface
/// * `n_polymers` - Number of parallel ring polymer replicas
/// * `n_beads` - Number of beads per ring polymer (1 = classical)
/// * `beta` - Inverse temperature in a.u.
/// * `dt` - Time step in a.u.
/// * `n_equilibrate` - Equilibration steps (no hill deposition)
/// * `n_production` - Production steps (hills deposited)
/// * `initial_height` - Initial hill height w₀ (Hartree)
/// * `sigma` - Hill width σ (Bohr)
/// * `bias_factor` - Well-tempered bias factor γ
/// * `deposit_stride` - Steps between hill depositions
/// * `donor, proton, acceptor` - Atom indices for the CV
/// * `label` - Label for output
/// * `use_ti` - Enable Takahashi-Imada fourth-order correction
pub fn run_pimd_metadynamics<P: MolecularPotential>(
    potential: P,
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
    initial_height: f64,
    sigma: f64,
    bias_factor: f64,
    deposit_stride: usize,
    donor: usize,
    proton: usize,
    acceptor: usize,
    label: &str,
    use_ti: bool,
) -> MetadynamicsResult {
    let kt = 1.0 / beta;
    let gamma_centroid = 0.001;
    let sample_interval = 10;

    // Grid for bias acceleration
    let grid_min = -3.0;
    let grid_max = 3.0;
    let grid_n = 1000;

    // Create shared bias
    let bias = Arc::new(RwLock::new(MetadynamicsBias::new(
        initial_height, sigma, bias_factor, kt, grid_min, grid_max, grid_n,
    )));

    // Create metadynamics potential
    let meta_pot = MetadynamicsPotential::new(
        potential, Arc::clone(&bias), donor, proton, acceptor,
    );

    // Create PIMD simulation
    let mut sim = if use_ti {
        MolecularPIMD::new_with_ti(
            n_polymers, n_beads, beta, dt, gamma_centroid, meta_pot,
        )
    } else {
        MolecularPIMD::new(
            n_polymers, n_beads, beta, dt, gamma_centroid, meta_pot,
        )
    };

    println!("  {} metadynamics: w₀={:.6} Ha, σ={:.3} Bohr, γ={:.1}{}",
             label, initial_height, sigma, bias_factor,
             if use_ti { " [TI]" } else { "" });
    println!("  Beads: {}, replicas: {}, dt: {:.4}, deposit every {} steps",
             n_beads, n_polymers, dt, deposit_stride);
    println!("  {} equil + {} production steps", n_equilibrate, n_production);
    println!();

    // --- Equilibration (no hills) ---
    for _step in 0..n_equilibrate {
        sim.step_obabo();
    }

    // --- Production ---
    let mut cv_trajectory = Vec::new();
    let mut energy_trajectory = Vec::new();
    let mut step_indices = Vec::new();
    let mut n_crossings = 0usize;
    let mut last_sign = 0i32;

    for step in 0..n_production {
        sim.step_obabo();

        // Deposit hill at centroid CV
        if step % deposit_stride == 0 {
            let avg_delta = sim.average_transfer_coordinate(donor, proton, acceptor);
            let mut bias_w = bias.write().unwrap();
            bias_w.deposit_hill(avg_delta);
        }

        // Sample observables
        if step % sample_interval == 0 {
            let avg_delta = sim.average_transfer_coordinate(donor, proton, acceptor);
            let avg_energy = sim.average_energy();

            cv_trajectory.push(avg_delta);
            energy_trajectory.push(avg_energy);
            step_indices.push(step);

            // Count barrier crossings (sign changes of δ)
            let current_sign = if avg_delta > 0.05 {
                1
            } else if avg_delta < -0.05 {
                -1
            } else {
                0
            };
            if current_sign != 0 && last_sign != 0 && current_sign != last_sign {
                n_crossings += 1;
            }
            if current_sign != 0 {
                last_sign = current_sign;
            }
        }

        // Progress report
        if (step + 1) % (n_production / 5).max(1) == 0 {
            let bias_r = bias.read().unwrap();
            let avg_delta = sim.average_transfer_coordinate(donor, proton, acceptor);
            let latest_h = if let Some(h) = bias_r.hills.last() {
                h.height
            } else {
                initial_height
            };
            println!("    Step {:>7}/{}: δ = {:+.4}, {} hills, h_eff = {:.6} Ha, crossings = {}",
                     step + 1, n_production,
                     avg_delta, bias_r.n_hills(), latest_h, n_crossings);
        }
    }

    // Reconstruct FES
    let bias_r = bias.read().unwrap();
    let (fes_cv, fes_values) = bias_r.reconstruct_fes(-2.5, 2.5, 200);
    let n_hills = bias_r.n_hills();
    let hills_clone = bias_r.hills.clone();
    drop(bias_r);

    println!("  Done: {} hills deposited, {} barrier crossings", n_hills, n_crossings);
    println!();

    MetadynamicsResult {
        hills: hills_clone,
        cv_trajectory,
        energy_trajectory,
        step_indices,
        fes_cv,
        fes_values,
        label: label.to_string(),
    }
}

/// Find the free energy barrier in the reconstructed FES.
fn find_barrier(fes: &[f64], cv: &[f64]) -> f64 {
    if fes.is_empty() || cv.is_empty() {
        return 0.0;
    }

    let n = fes.len();
    let mid_idx = cv.iter().position(|&x| x >= 0.0).unwrap_or(n / 2);

    // Find minimum on left side (δ < 0)
    let left_min = if mid_idx > 0 {
        fes[..mid_idx].iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        fes[0]
    };

    // Find minimum on right side (δ > 0)
    let right_min = if mid_idx < n - 1 {
        fes[(mid_idx + 1)..].iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        fes[n - 1]
    };

    // Find maximum near center
    let search_range = (n / 10).max(5);
    let start = mid_idx.saturating_sub(search_range);
    let end = (mid_idx + search_range).min(n);
    let center_max = fes[start..end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let well_avg = (left_min + right_min) / 2.0;
    (center_max - well_avg).max(0.0)
}

/// Run PIMD metadynamics on the Zundel cation with three-way comparison.
///
/// Compares:
///   1. Classical metadynamics (P=1)
///   2. Quantum PIMD metadynamics (P=n_beads, primitive)
///   3. Quantum+TI PIMD metadynamics (P=n_beads, fourth-order)
///
/// Outputs FES, hills history, and CV trajectories to files.
pub fn run_zundel_metadynamics(
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
    initial_height: f64,
    sigma: f64,
    bias_factor: f64,
    deposit_stride: usize,
) {
    use super::pimd_molecular::ZundelPES;

    let pes = ZundelPES::new();
    let kt = 1.0 / beta;
    let temp_k = 315774.65 / beta;

    // Zundel atom indices:
    //   O1(0), H1a(1), H1b(2), H*(3), O2(4), H2a(5), H2b(6)
    let donor = 0;
    let proton = 3;
    let acceptor = 4;

    println!("================================================================");
    println!("|  PIMD + Well-Tempered Metadynamics -- Zundel Cation H5O2+    |");
    println!("|                                                              |");
    println!("|  Free Energy Surface for Proton Transfer                     |");
    println!("|  CV: δ = d(O₁-H*) - d(O₂-H*)                               |");
    println!("================================================================");
    println!();
    println!("System: {} -- {} atoms, {} DOF", pes.name(), pes.n_atoms(), pes.ndof());
    println!("  Temperature: {:.1} K (β = {:.2} a.u., kT = {:.6} Ha)",
             temp_k, beta, kt);
    println!();
    println!("Metadynamics parameters:");
    println!("  Initial hill height: w₀ = {:.6} Ha ({:.3} kcal/mol)",
             initial_height, initial_height * 627.509);
    println!("  Hill width:          σ  = {:.3} Bohr", sigma);
    println!("  Bias factor:         γ  = {:.1}", bias_factor);
    println!("  Deposit stride:      every {} steps ({:.4} a.u.)",
             deposit_stride, deposit_stride as f64 * dt);
    println!("  Expected hills:      ~{}", n_production / deposit_stride);
    println!();

    // =========================================================================
    // Classical metadynamics (P = 1)
    // =========================================================================
    println!("============================================================");
    println!("  CLASSICAL Metadynamics (P = 1)");
    println!("============================================================");

    let cl_result = run_pimd_metadynamics(
        pes.clone(), n_polymers, 1, beta, dt,
        n_equilibrate, n_production,
        initial_height, sigma, bias_factor, deposit_stride,
        donor, proton, acceptor,
        "Classical", false,
    );

    // =========================================================================
    // Quantum metadynamics (P = n_beads, primitive)
    // =========================================================================
    println!("============================================================");
    println!("  QUANTUM Metadynamics (P = {}, primitive)", n_beads);
    println!("============================================================");

    let q_result = run_pimd_metadynamics(
        pes.clone(), n_polymers, n_beads, beta, dt,
        n_equilibrate, n_production,
        initial_height, sigma, bias_factor, deposit_stride,
        donor, proton, acceptor,
        "Quantum", false,
    );

    // =========================================================================
    // Quantum+TI metadynamics (P = n_beads, fourth-order)
    // =========================================================================
    println!("============================================================");
    println!("  QUANTUM+TI Metadynamics (P = {}, fourth-order)", n_beads);
    println!("============================================================");

    let ti_result = run_pimd_metadynamics(
        pes.clone(), n_polymers, n_beads, beta, dt,
        n_equilibrate, n_production,
        initial_height, sigma, bias_factor, deposit_stride,
        donor, proton, acceptor,
        "Quantum+TI", true,
    );

    // =========================================================================
    // Comparison
    // =========================================================================
    let cl_barrier = find_barrier(&cl_result.fes_values, &cl_result.fes_cv);
    let q_barrier = find_barrier(&q_result.fes_values, &q_result.fes_cv);
    let ti_barrier = find_barrier(&ti_result.fes_values, &ti_result.fes_cv);

    println!("========================================================================");
    println!("                 METADYNAMICS FES COMPARISON SUMMARY");
    println!("========================================================================");
    println!("  {:>20} | {:>14} | {:>14} | {:>14}", "Property", "Classical", "Quantum", "Quantum+TI");
    println!("  {:>20} | {:>14} | {:>14} | {:>14}", "--------------------", "--------------", "--------------", "--------------");
    println!("  {:>20} | {:>14.6} | {:>14.6} | {:>14.6}", "Barrier (Ha)", cl_barrier, q_barrier, ti_barrier);
    println!("  {:>20} | {:>14.2} | {:>14.2} | {:>14.2}", "Barrier (kcal/mol)",
             cl_barrier * 627.509, q_barrier * 627.509, ti_barrier * 627.509);
    println!("  {:>20} | {:>14} | {:>14} | {:>14}", "Hills deposited",
             format!("{}", cl_result.hills.len()),
             format!("{}", q_result.hills.len()),
             format!("{}", ti_result.hills.len()));
    println!("========================================================================");
    println!();

    if q_barrier < cl_barrier - 0.0001 {
        let reduction = (cl_barrier - q_barrier) * 627.509;
        println!("  * Quantum tunneling REDUCES the effective barrier by {:.2} kcal/mol!", reduction);
    }
    if (ti_barrier - q_barrier).abs() > 0.0001 {
        println!("  * TI correction shifts barrier by {:.2} kcal/mol vs primitive",
                 (ti_barrier - q_barrier) * 627.509);
    }
    println!();

    // =========================================================================
    // Output files
    // =========================================================================

    // FES comparison
    {
        let file = File::create("metadynamics_fes.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# delta W_classical(Ha) W_quantum(Ha) W_quantum_TI(Ha) W_cl(kcal/mol) W_q(kcal/mol) W_ti(kcal/mol)").unwrap();
        let n = cl_result.fes_cv.len();
        for i in 0..n {
            writeln!(w, "{:.6} {:.8} {:.8} {:.8} {:.4} {:.4} {:.4}",
                     cl_result.fes_cv[i],
                     cl_result.fes_values[i], q_result.fes_values[i], ti_result.fes_values[i],
                     cl_result.fes_values[i] * 627.509,
                     q_result.fes_values[i] * 627.509,
                     ti_result.fes_values[i] * 627.509).unwrap();
        }
        println!("  FES                     -> metadynamics_fes.txt");
    }

    // Hills history
    for (result, suffix) in [(&cl_result, "cl"), (&q_result, "q"), (&ti_result, "ti")] {
        let filename = format!("metadynamics_hills_{}.txt", suffix);
        let file = File::create(&filename).unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# hill_index center(Bohr) height(Ha) sigma(Bohr)").unwrap();
        for (i, hill) in result.hills.iter().enumerate() {
            writeln!(w, "{} {:.6} {:.8} {:.6}", i, hill.center, hill.height, hill.sigma).unwrap();
        }
        println!("  Hills ({:>10})      -> {}", result.label, filename);
    }

    // CV trajectories
    for (result, suffix) in [(&cl_result, "cl"), (&q_result, "q"), (&ti_result, "ti")] {
        let filename = format!("metadynamics_cv_{}.txt", suffix);
        let file = File::create(&filename).unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# step delta(Bohr) energy(Ha)").unwrap();
        for i in 0..result.cv_trajectory.len() {
            writeln!(w, "{} {:.6} {:.8}",
                     result.step_indices[i],
                     result.cv_trajectory[i],
                     result.energy_trajectory[i]).unwrap();
        }
        println!("  CV trajectory ({:>6}) -> {}", result.label, filename);
    }

    println!();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::potentials::MolecularPotential;

    /// Simple harmonic potential for testing (single atom, 3 DOF)
    #[derive(Clone)]
    struct TestPotential {
        omega: f64,
        mass: f64,
    }

    impl MolecularPotential for TestPotential {
        fn n_atoms(&self) -> usize { 3 }
        fn energy(&self, coords: &[f64]) -> f64 {
            let r2: f64 = coords[..3].iter().map(|&x| x * x).sum();
            0.5 * self.mass * self.omega * self.omega * r2
        }
        fn forces(&self, coords: &[f64], forces: &mut [f64]) {
            let k = self.mass * self.omega * self.omega;
            for d in 0..self.ndof() {
                if d < 3 { forces[d] = -k * coords[d]; }
                else { forces[d] = 0.0; }
            }
        }
        fn masses(&self) -> &[f64] { &[1836.15, 1836.15, 1836.15] }
        fn reference_geometry(&self) -> Vec<f64> {
            vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        }
        fn name(&self) -> &'static str { "Test Harmonic" }
    }

    #[test]
    fn test_gaussian_hill_energy_force() {
        let hill = GaussianHill { center: 0.5, height: 0.01, sigma: 0.2 };

        // At the center, energy = height
        assert!((hill.energy(0.5) - 0.01).abs() < 1e-12);

        // Force at center should be zero
        assert!(hill.force(0.5).abs() < 1e-12);

        // Numerical force check
        let h = 1e-7;
        let f_num = -(hill.energy(0.5 + h) - hill.energy(0.5 - h)) / (2.0 * h);
        let f_ana = hill.force(0.5);
        assert!((f_ana - f_num).abs() < 1e-6,
                "Force at center: analytical {:.10} vs numerical {:.10}", f_ana, f_num);

        // Away from center
        let s = 0.8;
        let f_num = -(hill.energy(s + h) - hill.energy(s - h)) / (2.0 * h);
        let f_ana = hill.force(s);
        assert!((f_ana - f_num).abs() < 1e-6,
                "Force at s={}: analytical {:.10} vs numerical {:.10}", s, f_ana, f_num);
    }

    #[test]
    fn test_well_tempered_height_decay() {
        let kt = 0.001;
        let gamma = 10.0;
        let mut bias = MetadynamicsBias::new(0.01, 0.2, gamma, kt, -3.0, 3.0, 100);

        // First hill has full height
        bias.deposit_hill(0.0);
        assert!((bias.hills[0].height - 0.01).abs() < 1e-12,
                "First hill should have full height");

        // Deposit many hills at the same point — heights should decay
        for _ in 0..20 {
            bias.deposit_hill(0.0);
        }

        // Later hills should be smaller
        let first_h = bias.hills[0].height;
        let last_h = bias.hills.last().unwrap().height;
        assert!(last_h < first_h * 0.5,
                "Well-tempered hills should decay: first={:.6}, last={:.6}", first_h, last_h);
    }

    #[test]
    fn test_metadynamics_bias_energy_force_consistency() {
        let kt = 0.001;
        let mut bias = MetadynamicsBias::new(0.005, 0.3, 15.0, kt, -3.0, 3.0, 500);

        // Deposit some hills at various positions
        for &s in &[-0.5, 0.0, 0.3, -0.2, 0.1, -0.8, 0.6] {
            bias.deposit_hill(s);
        }

        // Check force = -dV/ds numerically
        let h = 1e-7;
        for &s in &[-1.0, -0.5, 0.0, 0.3, 0.7, 1.0] {
            let f_ana = bias.force_direct(s);
            let f_num = -(bias.energy_direct(s + h) - bias.energy_direct(s - h)) / (2.0 * h);
            assert!((f_ana - f_num).abs() < 1e-5,
                    "Bias force mismatch at s={}: analytical {:.8} vs numerical {:.8}",
                    s, f_ana, f_num);
        }
    }

    #[test]
    fn test_grid_matches_direct() {
        let kt = 0.001;
        let mut bias = MetadynamicsBias::new(0.005, 0.2, 10.0, kt, -3.0, 3.0, 1000);

        for &s in &[-0.5, 0.0, 0.3, -0.2, 0.7] {
            bias.deposit_hill(s);
        }

        // Grid should match direct summation within interpolation error
        for &s in &[-1.0, -0.3, 0.0, 0.15, 0.5, 1.2] {
            let e_direct = bias.energy_direct(s);
            let e_grid = bias.energy_grid(s);
            let rel_err = if e_direct.abs() > 1e-10 {
                (e_grid - e_direct).abs() / e_direct.abs()
            } else {
                (e_grid - e_direct).abs()
            };
            assert!(rel_err < 0.01,
                    "Grid energy mismatch at s={}: direct={:.8}, grid={:.8}, rel_err={:.6}",
                    s, e_direct, e_grid, rel_err);
        }
    }

    #[test]
    fn test_fes_reconstruction() {
        let kt = 0.001;
        let mut bias = MetadynamicsBias::new(0.01, 0.2, 10.0, kt, -3.0, 3.0, 100);

        // Deposit many hills at δ=0 — this fills the well there,
        // so the FES should show the minimum at δ=0 (most explored region)
        for _ in 0..30 {
            bias.deposit_hill(0.0);
        }

        let (cv, fes) = bias.reconstruct_fes(-2.0, 2.0, 100);

        // FES should be non-negative (shifted so minimum = 0)
        for &v in &fes {
            assert!(v >= -1e-10, "FES should be non-negative, got {:.8}", v);
        }

        // FES minimum should be at δ=0 (where all hills were deposited)
        let min_idx = fes.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        assert!(cv[min_idx].abs() < 0.5,
                "FES minimum should be near δ=0, got δ={:.3}", cv[min_idx]);

        // FES should be larger away from δ=0
        // Find value at the edges
        let edge_val = fes[0]; // at δ=-2.0
        let center_val = fes[min_idx]; // at minimum
        assert!(edge_val > center_val + 1e-6,
                "FES should be higher at edges ({:.6}) than center ({:.6})",
                edge_val, center_val);
    }

    #[test]
    fn test_metadynamics_potential_forces() {
        // Verify that MetadynamicsPotential forces match numerical derivatives
        use super::super::pimd_molecular::ZundelPES;

        let pes = ZundelPES::new();
        let kt = 0.001;
        let bias = Arc::new(RwLock::new(
            MetadynamicsBias::new(0.005, 0.3, 10.0, kt, -3.0, 3.0, 500)
        ));

        // Deposit a few hills
        {
            let mut b = bias.write().unwrap();
            b.deposit_hill(0.0);
            b.deposit_hill(0.3);
            b.deposit_hill(-0.2);
        }

        let meta_pot = MetadynamicsPotential::new(pes, Arc::clone(&bias), 0, 3, 4);
        let geom = meta_pot.inner.reference_geometry();
        let ndof = meta_pot.ndof();

        let mut forces = vec![0.0; ndof];
        meta_pot.forces(&geom, &mut forces);

        // Numerical check
        let h = 1e-6;
        let mut gp = geom.clone();
        let mut gm = geom.clone();
        for d in 0..ndof {
            gp[d] = geom[d] + h;
            gm[d] = geom[d] - h;
            let f_num = -(meta_pot.energy(&gp) - meta_pot.energy(&gm)) / (2.0 * h);
            gp[d] = geom[d];
            gm[d] = geom[d];

            assert!((forces[d] - f_num).abs() < 1e-3,
                    "Metadynamics force mismatch at dof {}: analytical {:.8} vs numerical {:.8}",
                    d, forces[d], f_num);
        }
    }
}
