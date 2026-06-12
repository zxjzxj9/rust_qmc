//! Umbrella Sampling with WHAM for Path Integral Molecular Dynamics
//!
//! Implements biased PIMD simulations along a reaction coordinate (the proton
//! transfer coordinate δ = d(O₁-H*) - d(O₂-H*)) with harmonic restraining
//! potentials, and the Weighted Histogram Analysis Method (WHAM) for
//! reconstructing the unbiased potential of mean force (PMF).
//!
//! The umbrella sampling workflow:
//!   1. Define a set of windows along the reaction coordinate
//!   2. In each window, run PIMD with an additional harmonic bias V_bias = ½κ(δ-δ₀)²
//!   3. Collect histograms of the centroid transfer coordinate in each window
//!   4. Use WHAM to combine biased histograms into the unbiased PMF
//!
//! The bias potential and forces are applied at the MolecularPotential level,
//! so the existing MolecularPIMD integrator is used without modification.
//!
//! References:
//!   - Torrie & Valleau, J. Comput. Phys. 23, 187 (1977) -- Umbrella sampling
//!   - Kumar et al., J. Comput. Chem. 13, 1011 (1992) -- WHAM
//!   - Marx & Parrinello, J. Chem. Phys. 104, 4077 (1996) -- PIMD + umbrella

use std::fs::File;
use std::io::{BufWriter, Write};

use super::pimd_molecular::{MolecularPotential, MolecularPIMD};

// =============================================================================
// Umbrella Bias
// =============================================================================

/// Harmonic umbrella bias potential: V_bias(δ) = ½κ(δ - δ₀)²
///
/// Applied along the proton transfer coordinate
/// δ = d(donor-proton) - d(acceptor-proton).
#[derive(Clone, Debug)]
pub struct UmbrellaBias {
    /// Window center δ₀ (Bohr)
    pub center: f64,
    /// Spring constant κ (Hartree/Bohr²)
    pub spring_constant: f64,
}

impl UmbrellaBias {
    /// Create a new umbrella bias window
    pub fn new(center: f64, spring_constant: f64) -> Self {
        Self { center, spring_constant }
    }

    /// Evaluate the bias energy: ½κ(δ - δ₀)²
    pub fn energy(&self, delta: f64) -> f64 {
        0.5 * self.spring_constant * (delta - self.center).powi(2)
    }

    /// Bias force along the reaction coordinate: -dV/dδ = -κ(δ - δ₀)
    pub fn force_along_rc(&self, delta: f64) -> f64 {
        -self.spring_constant * (delta - self.center)
    }
}

// =============================================================================
// Biased Molecular Potential
// =============================================================================

/// Wraps a `MolecularPotential` with an umbrella bias along the proton
/// transfer coordinate δ = d(donor-H*) - d(acceptor-H*).
///
/// The total potential is V_total = V_physical + V_bias(δ(R)).
/// Forces include the analytical gradient of V_bias through the chain rule:
///   F_bias_i = -∂V_bias/∂R_i = -κ(δ-δ₀) × ∂δ/∂R_i
#[derive(Clone)]
pub struct BiasedMolecularPotential<P: MolecularPotential> {
    /// The underlying physical potential
    pub inner: P,
    /// Umbrella bias
    pub bias: UmbrellaBias,
    /// Index of the donor atom (e.g., O₁)
    pub donor: usize,
    /// Index of the proton being transferred (e.g., H*)
    pub proton: usize,
    /// Index of the acceptor atom (e.g., O₂)
    pub acceptor: usize,
}

impl<P: MolecularPotential> BiasedMolecularPotential<P> {
    /// Create a new biased potential
    pub fn new(
        inner: P,
        bias: UmbrellaBias,
        donor: usize,
        proton: usize,
        acceptor: usize,
    ) -> Self {
        Self { inner, bias, donor, proton, acceptor }
    }

    /// Compute the transfer coordinate δ = d(donor-proton) - d(acceptor-proton)
    fn transfer_coordinate(&self, coords: &[f64]) -> f64 {
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

    /// Compute ∂δ/∂R_i for all coordinates.
    ///
    /// δ = r_AH - r_BH where A=donor, H=proton, B=acceptor
    ///
    /// ∂r_AH/∂R_A = (R_A - R_H) / r_AH   (unit vector A→H, but from A)
    /// ∂r_AH/∂R_H = (R_H - R_A) / r_AH   (unit vector H→A, but from H)
    /// ∂r_BH/∂R_B = (R_B - R_H) / r_BH
    /// ∂r_BH/∂R_H = (R_H - R_B) / r_BH
    ///
    /// So: ∂δ/∂R_A = (R_A - R_H) / r_AH
    ///     ∂δ/∂R_H = (R_H - R_A) / r_AH - (R_H - R_B) / r_BH
    ///     ∂δ/∂R_B = -(R_B - R_H) / r_BH = (R_H - R_B) / r_BH  ... wait
    ///
    /// Let me be more careful:
    ///   δ = r_AH - r_BH
    ///   ∂δ/∂R = ∂r_AH/∂R - ∂r_BH/∂R
    ///
    /// For r_AH = |R_A - R_H|:
    ///   ∂r_AH/∂R_{A,xyz} = (R_{A,xyz} - R_{H,xyz}) / r_AH
    ///   ∂r_AH/∂R_{H,xyz} = -(R_{A,xyz} - R_{H,xyz}) / r_AH = (R_{H,xyz} - R_{A,xyz}) / r_AH
    ///
    /// For r_BH = |R_B - R_H|:
    ///   ∂r_BH/∂R_{B,xyz} = (R_{B,xyz} - R_{H,xyz}) / r_BH
    ///   ∂r_BH/∂R_{H,xyz} = (R_{H,xyz} - R_{B,xyz}) / r_BH
    fn transfer_coordinate_gradient(&self, coords: &[f64], grad: &mut [f64]) {
        let ndof = self.inner.ndof();
        for g in grad[..ndof].iter_mut() {
            *g = 0.0;
        }

        let r_ah = Self::atom_distance(coords, self.donor, self.proton);
        let r_bh = Self::atom_distance(coords, self.acceptor, self.proton);

        if r_ah < 1e-15 || r_bh < 1e-15 {
            return; // Degenerate geometry, skip
        }

        let a = self.donor;
        let h = self.proton;
        let b = self.acceptor;

        for xyz in 0..3 {
            let ra = coords[3 * a + xyz];
            let rh = coords[3 * h + xyz];
            let rb = coords[3 * b + xyz];

            // ∂δ/∂R_A = ∂r_AH/∂R_A = (R_A - R_H) / r_AH
            grad[3 * a + xyz] = (ra - rh) / r_ah;

            // ∂δ/∂R_B = -∂r_BH/∂R_B = -(R_B - R_H) / r_BH
            grad[3 * b + xyz] = -(rb - rh) / r_bh;

            // ∂δ/∂R_H = ∂r_AH/∂R_H - ∂r_BH/∂R_H
            //         = (R_H - R_A)/r_AH - (R_H - R_B)/r_BH
            grad[3 * h + xyz] = (rh - ra) / r_ah - (rh - rb) / r_bh;
        }
    }
}

impl<P: MolecularPotential> MolecularPotential for BiasedMolecularPotential<P> {
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    fn energy(&self, coords: &[f64]) -> f64 {
        let e_phys = self.inner.energy(coords);
        let delta = self.transfer_coordinate(coords);
        let e_bias = self.bias.energy(delta);
        e_phys + e_bias
    }

    /// Analytical forces: F_total = F_physical + F_bias
    ///
    /// F_bias_i = -∂V_bias/∂R_i = -κ(δ-δ₀) × ∂δ/∂R_i
    fn forces(&self, coords: &[f64], forces: &mut [f64]) {
        // Physical forces
        self.inner.forces(coords, forces);

        // Bias forces via chain rule
        let ndof = self.inner.ndof();
        let delta = self.transfer_coordinate(coords);
        let dv_ddelta = self.bias.spring_constant * (delta - self.bias.center);

        let mut ddelta_dr = vec![0.0; ndof];
        self.transfer_coordinate_gradient(coords, &mut ddelta_dr);

        for d in 0..ndof {
            forces[d] -= dv_ddelta * ddelta_dr[d]; // F = -dV/dR
        }
    }

    fn masses(&self) -> &[f64] {
        self.inner.masses()
    }

    fn reference_geometry(&self) -> Vec<f64> {
        self.inner.reference_geometry()
    }

    fn name(&self) -> &'static str {
        "Biased (Umbrella Sampling)"
    }
}

// =============================================================================
// Umbrella Window Results
// =============================================================================

/// Results from a single umbrella sampling window
#[derive(Clone, Debug)]
pub struct UmbrellaWindow {
    /// Window center δ₀
    pub center: f64,
    /// Bias spring constant κ
    pub spring_constant: f64,
    /// Histogram of the transfer coordinate (centroid values)
    pub histogram: Vec<f64>,
    /// Histogram bin edges: [min, max)
    pub hist_min: f64,
    pub hist_max: f64,
    /// Total number of samples collected
    pub n_samples: usize,
    /// Mean energy in this window
    pub mean_energy: f64,
    /// Mean transfer coordinate in this window
    pub mean_delta: f64,
}

// =============================================================================
// WHAM Solver
// =============================================================================

/// Weighted Histogram Analysis Method (WHAM) solver.
///
/// Iteratively solves the self-consistent equations to combine biased
/// histograms from multiple umbrella windows into an unbiased probability
/// distribution (and hence the potential of mean force).
///
/// The WHAM equations:
///
///   P_unbias(δ_j) = Σ_i n_i(δ_j) / Σ_i N_i × exp[-β(V_i(δ_j) - F_i)]
///
///   exp(-βF_i) = Σ_j P_unbias(δ_j) × exp[-βV_i(δ_j)] × Δδ
///
/// where:
///   n_i(δ_j) = count in bin j from window i
///   N_i = total samples in window i
///   V_i(δ) = ½κ_i(δ - δ₀_i)² = bias potential in window i
///   F_i = free energy offset for window i
///
/// Reference: Kumar et al., J. Comput. Chem. 13, 1011 (1992)
pub struct WHAMSolver {
    /// Inverse temperature β (a.u.)
    pub beta: f64,
    /// Number of histogram bins
    pub n_bins: usize,
    /// Bin centers
    pub bin_centers: Vec<f64>,
    /// Bin width
    pub bin_width: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance on free energies
    pub tolerance: f64,
}

impl WHAMSolver {
    /// Create a WHAM solver
    pub fn new(
        beta: f64,
        hist_min: f64,
        hist_max: f64,
        n_bins: usize,
        max_iter: usize,
        tolerance: f64,
    ) -> Self {
        let bin_width = (hist_max - hist_min) / n_bins as f64;
        let bin_centers: Vec<f64> = (0..n_bins)
            .map(|i| hist_min + (i as f64 + 0.5) * bin_width)
            .collect();

        Self {
            beta,
            n_bins,
            bin_centers,
            bin_width,
            max_iter,
            tolerance,
        }
    }

    /// Solve the WHAM equations to obtain the unbiased PMF.
    ///
    /// Returns (pmf, free_energies, n_iterations):
    /// - pmf: W(δ) = -kT ln P(δ) for each bin, shifted so minimum = 0
    /// - free_energies: F_i for each window
    /// - n_iterations: number of iterations until convergence
    pub fn solve(&self, windows: &[UmbrellaWindow]) -> (Vec<f64>, Vec<f64>, usize) {
        let n_windows = windows.len();
        let n_bins = self.n_bins;
        let beta = self.beta;
        let kbt = 1.0 / beta;

        // Precompute bias energies: bias_energy[window][bin]
        let bias_energy: Vec<Vec<f64>> = windows.iter()
            .map(|w| {
                self.bin_centers.iter()
                    .map(|&delta| 0.5 * w.spring_constant * (delta - w.center).powi(2))
                    .collect()
            })
            .collect();

        // Total samples per window
        let n_samples: Vec<f64> = windows.iter()
            .map(|w| w.n_samples as f64)
            .collect();

        // Total counts per bin across all windows
        let total_counts: Vec<f64> = (0..n_bins)
            .map(|j| windows.iter().map(|w| w.histogram[j]).sum())
            .collect();

        // Initialize free energies to zero
        let mut free_energies = vec![0.0_f64; n_windows];
        let mut p_unbias = vec![0.0_f64; n_bins];

        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Step 1: Compute P_unbias(δ_j)
            for j in 0..n_bins {
                if total_counts[j] < 0.5 {
                    p_unbias[j] = 0.0;
                    continue;
                }
                let mut denom = 0.0;
                for i in 0..n_windows {
                    // N_i × exp[-β(V_i(δ_j) - F_i)]
                    denom += n_samples[i] * (-beta * (bias_energy[i][j] - free_energies[i])).exp();
                }
                p_unbias[j] = total_counts[j] / denom;
            }

            // Normalize P_unbias (for numerical stability)
            let p_total: f64 = p_unbias.iter().sum::<f64>() * self.bin_width;
            if p_total > 0.0 {
                for p in &mut p_unbias {
                    *p /= p_total;
                }
            }

            // Step 2: Update free energies
            let mut new_free_energies = vec![0.0_f64; n_windows];
            for i in 0..n_windows {
                let mut sum = 0.0;
                for j in 0..n_bins {
                    if p_unbias[j] > 1e-300 {
                        sum += p_unbias[j] * (-beta * bias_energy[i][j]).exp() * self.bin_width;
                    }
                }
                if sum > 1e-300 {
                    new_free_energies[i] = -kbt * sum.ln();
                } else {
                    new_free_energies[i] = free_energies[i]; // Keep old value
                }
            }

            // Shift so F[0] = 0 (arbitrary reference)
            let f0 = new_free_energies[0];
            for f in &mut new_free_energies {
                *f -= f0;
            }

            // Check convergence
            let max_change: f64 = free_energies.iter()
                .zip(new_free_energies.iter())
                .map(|(&old, &new)| (old - new).abs())
                .fold(0.0_f64, f64::max);

            free_energies = new_free_energies;

            if max_change < self.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            println!("  WARNING: WHAM did not converge after {} iterations", self.max_iter);
        }

        // Compute PMF: W(δ) = -kT ln P(δ)
        let p_max = p_unbias.iter().cloned().fold(0.0_f64, f64::max);
        let pmf: Vec<f64> = p_unbias.iter()
            .map(|&p| {
                if p > 1e-300 && p_max > 1e-300 {
                    -kbt * (p / p_max).ln()
                } else {
                    10.0 * kbt // Large but finite for empty bins
                }
            })
            .collect();

        // Shift PMF so minimum is zero
        let pmf_min = pmf.iter().cloned().fold(f64::INFINITY, f64::min);
        let pmf_shifted: Vec<f64> = pmf.iter().map(|&w| w - pmf_min).collect();

        (pmf_shifted, free_energies, n_iter)
    }
}

// =============================================================================
// Umbrella Sampling Driver for the Zundel Cation
// =============================================================================

/// Run PIMD + umbrella sampling for the Zundel cation proton transfer.
///
/// Performs biased PIMD simulations in multiple windows along the proton
/// transfer coordinate δ, then uses WHAM to reconstruct the unbiased PMF.
///
/// # Arguments
/// * `n_polymers` - Number of ring polymer replicas per window
/// * `n_beads` - Number of beads per ring polymer (1 for classical)
/// * `beta` - Physical inverse temperature (a.u.)
/// * `dt` - Integration time step (a.u.)
/// * `n_equilibrate` - Equilibration steps per window
/// * `n_production` - Production steps per window
/// * `window_centers` - Centers δ₀ of umbrella windows (Bohr)
/// * `bias_spring_constant` - κ for the harmonic bias (Hartree/Bohr²)
/// * `n_hist_bins` - Number of histogram bins for WHAM
/// * `hist_min` - Minimum δ for histogram
/// * `hist_max` - Maximum δ for histogram
/// * `label` - Label for output (e.g., "quantum" or "classical")
///
/// # Returns
/// Vector of `UmbrellaWindow` results for further analysis
pub fn run_pimd_umbrella_sampling<P: MolecularPotential>(
    potential: P,
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
    window_centers: &[f64],
    bias_spring_constant: f64,
    donor: usize,
    proton: usize,
    acceptor: usize,
    n_hist_bins: usize,
    hist_min: f64,
    hist_max: f64,
    label: &str,
    use_ti: bool,
) -> Vec<UmbrellaWindow> {
    let n_windows = window_centers.len();
    let hist_bw = (hist_max - hist_min) / n_hist_bins as f64;
    let sample_interval = 10;
    let gamma_centroid = 0.001;

    println!("  {} umbrella sampling: {} windows, κ = {:.4} Ha/Bohr²{}",
             label, n_windows, bias_spring_constant,
             if use_ti { " [TI fourth-order]" } else { "" });
    println!("  Beads: {}, replicas: {}, dt: {:.4}", n_beads, n_polymers, dt);
    println!("  Per window: {} equil + {} production steps", n_equilibrate, n_production);
    println!();

    let mut windows: Vec<UmbrellaWindow> = Vec::with_capacity(n_windows);

    for (win_idx, &center) in window_centers.iter().enumerate() {
        let bias = UmbrellaBias::new(center, bias_spring_constant);
        let biased_pot = BiasedMolecularPotential::new(
            potential.clone(), bias, donor, proton, acceptor,
        );

        let mut sim = if use_ti {
            MolecularPIMD::new_with_ti(
                n_polymers, n_beads, beta, dt, gamma_centroid, biased_pot,
            )
        } else {
            MolecularPIMD::new(
                n_polymers, n_beads, beta, dt, gamma_centroid, biased_pot,
            )
        };

        // Equilibrate
        for _step in 0..n_equilibrate {
            sim.step_obabo();
        }

        // Production: collect histogram of centroid transfer coordinate
        let mut histogram = vec![0.0_f64; n_hist_bins];
        let mut energy_sum = 0.0_f64;
        let mut delta_sum = 0.0_f64;
        let mut n_samples = 0_usize;

        for step in 0..n_production {
            sim.step_obabo();

            if step % sample_interval == 0 {
                // Collect centroid transfer coordinate from each polymer
                for polymer in &sim.polymers {
                    let delta = polymer.transfer_coordinate(donor, proton, acceptor);

                    // Bin the centroid transfer coordinate
                    if delta >= hist_min && delta < hist_max {
                        let bin = ((delta - hist_min) / hist_bw) as usize;
                        if bin < n_hist_bins {
                            histogram[bin] += 1.0;
                        }
                    }

                    delta_sum += delta;
                    n_samples += 1;
                }

                energy_sum += sim.average_energy();
            }
        }

        let n_energy_samples = n_production / sample_interval;
        let mean_energy = if n_energy_samples > 0 {
            energy_sum / n_energy_samples as f64
        } else {
            0.0
        };
        let mean_delta = if n_samples > 0 {
            delta_sum / n_samples as f64
        } else {
            0.0
        };

        println!("  Window {:2}/{}: δ₀ = {:+6.3}, <δ> = {:+6.3}, <E> = {:.6}, samples = {}",
                 win_idx + 1, n_windows, center, mean_delta, mean_energy, n_samples);

        windows.push(UmbrellaWindow {
            center,
            spring_constant: bias_spring_constant,
            histogram,
            hist_min,
            hist_max,
            n_samples,
            mean_energy,
            mean_delta,
        });
    }

    println!();
    windows
}

/// Run the full umbrella sampling workflow for the Zundel cation:
/// classical (P=1) and quantum (P=n_beads) umbrella sampling + WHAM.
///
/// This is the main entry point for the example.
pub fn run_zundel_umbrella_sampling(
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
    window_centers: &[f64],
    bias_spring_constant: f64,
) {
    use super::pimd_molecular::ZundelPES;

    let pes = ZundelPES::new();
    let temp_k = 315774.65 / beta;
    let kbt = 1.0 / beta;

    // Atom indices for the Zundel cation
    let donor = 0_usize;   // O1
    let proton = 3_usize;  // H* (shared proton)
    let acceptor = 4_usize; // O2

    let n_hist_bins = 200;
    let hist_min = -2.5;
    let hist_max = 2.5;

    println!("================================================================");
    println!("|  PIMD + Umbrella Sampling -- Zundel Cation H5O2+            |");
    println!("|                                                              |");
    println!("|  Potential of Mean Force for Proton Transfer                 |");
    println!("|  W(δ) along δ = d(O1-H*) - d(O2-H*)                        |");
    println!("================================================================");
    println!();
    println!("System: {} -- 7 atoms, 21 DOF", pes.name());
    println!("  Temperature: {:.1} K (β = {:.2} a.u., kT = {:.6} Ha)", temp_k, beta, kbt);
    println!("  O...O equilibrium: {:.4} Bohr ({:.4} Å)",
             pes.r_oo_eq, pes.r_oo_eq * 0.529177);
    println!("  EVB coupling: A = {:.4} Ha, μ = {:.3} /Bohr",
             pes.coupling_a, pes.coupling_mu);
    println!();
    println!("Umbrella sampling parameters:");
    println!("  Windows: {} from δ = {:.3} to {:.3} Bohr",
             window_centers.len(),
             window_centers.first().unwrap_or(&0.0),
             window_centers.last().unwrap_or(&0.0));
    println!("  Bias spring constant: κ = {:.4} Ha/Bohr² ({:.2} kcal/mol/Bohr²)",
             bias_spring_constant, bias_spring_constant * 627.509);
    println!("  Histogram: {} bins, δ ∈ [{:.1}, {:.1}]", n_hist_bins, hist_min, hist_max);
    println!();

    // =========================================================================
    // Classical umbrella sampling (P = 1)
    // =========================================================================
    println!("============================================================");
    println!("  CLASSICAL Umbrella Sampling (P = 1)");
    println!("============================================================");

    let cl_windows = run_pimd_umbrella_sampling(
        pes.clone(),
        n_polymers,
        1,       // P = 1 for classical
        beta,
        dt,
        n_equilibrate,
        n_production,
        window_centers,
        bias_spring_constant,
        donor, proton, acceptor,
        n_hist_bins,
        hist_min, hist_max,
        "Classical",
        false,
    );

    // =========================================================================
    // Quantum umbrella sampling (P = n_beads, primitive)
    // =========================================================================
    println!("============================================================");
    println!("  QUANTUM Umbrella Sampling (P = {}, primitive)", n_beads);
    println!("============================================================");

    let q_windows = run_pimd_umbrella_sampling(
        pes.clone(),
        n_polymers,
        n_beads,
        beta,
        dt,
        n_equilibrate,
        n_production,
        window_centers,
        bias_spring_constant,
        donor, proton, acceptor,
        n_hist_bins,
        hist_min, hist_max,
        "Quantum",
        false,
    );

    // =========================================================================
    // Quantum umbrella sampling with TI (P = n_beads, fourth-order)
    // =========================================================================
    println!("============================================================");
    println!("  QUANTUM+TI Umbrella Sampling (P = {}, fourth-order)", n_beads);
    println!("============================================================");

    let ti_windows = run_pimd_umbrella_sampling(
        pes.clone(),
        n_polymers,
        n_beads,
        beta,
        dt,
        n_equilibrate,
        n_production,
        window_centers,
        bias_spring_constant,
        donor, proton, acceptor,
        n_hist_bins,
        hist_min, hist_max,
        "Quantum+TI",
        true,
    );

    // =========================================================================
    // WHAM analysis
    // =========================================================================
    println!("============================================================");
    println!("  WHAM Free Energy Reconstruction");
    println!("============================================================");
    println!();

    let wham = WHAMSolver::new(beta, hist_min, hist_max, n_hist_bins, 1000, 1e-7);

    // Classical WHAM
    let (cl_pmf, cl_free_energies, cl_wham_iter) = wham.solve(&cl_windows);
    println!("  Classical WHAM converged in {} iterations", cl_wham_iter);
    println!("    Window free energies (Ha): {:?}",
             cl_free_energies.iter().map(|f| format!("{:.4}", f)).collect::<Vec<_>>());

    // Quantum WHAM (primitive)
    let (q_pmf, q_free_energies, q_wham_iter) = wham.solve(&q_windows);
    println!("  Quantum (primitive) WHAM converged in {} iterations", q_wham_iter);
    println!("    Window free energies (Ha): {:?}",
             q_free_energies.iter().map(|f| format!("{:.4}", f)).collect::<Vec<_>>());

    // Quantum+TI WHAM
    let (ti_pmf, ti_free_energies, ti_wham_iter) = wham.solve(&ti_windows);
    println!("  Quantum+TI WHAM converged in {} iterations", ti_wham_iter);
    println!("    Window free energies (Ha): {:?}",
             ti_free_energies.iter().map(|f| format!("{:.4}", f)).collect::<Vec<_>>());
    println!();

    // =========================================================================
    // Analysis: extract barrier heights
    // =========================================================================
    let bin_centers = &wham.bin_centers;

    // Find barrier height: max PMF near δ=0 minus min PMF
    let cl_barrier = find_barrier(&cl_pmf, bin_centers);
    let q_barrier = find_barrier(&q_pmf, bin_centers);
    let ti_barrier = find_barrier(&ti_pmf, bin_centers);

    println!("========================================================================");
    println!("                        PMF COMPARISON SUMMARY");
    println!("========================================================================");
    println!("  {:>20} | {:>14} | {:>14} | {:>14}", "Property", "Classical", "Quantum", "Quantum+TI");
    println!("  {:>20} | {:>14} | {:>14} | {:>14}", "--------------------", "--------------", "--------------", "--------------");
    println!("  {:>20} | {:>14.6} | {:>14.6} | {:>14.6}", "Barrier (Ha)", cl_barrier, q_barrier, ti_barrier);
    println!("  {:>20} | {:>14.2} | {:>14.2} | {:>14.2}", "Barrier (kcal/mol)",
             cl_barrier * 627.509, q_barrier * 627.509, ti_barrier * 627.509);
    println!("  {:>20} | {:>14} | {:>14} | {:>14}", "WHAM iterations",
             format!("{}", cl_wham_iter), format!("{}", q_wham_iter), format!("{}", ti_wham_iter));
    println!("========================================================================");
    println!();

    if q_barrier < cl_barrier - 0.0001 {
        let reduction = (cl_barrier - q_barrier) * 627.509;
        println!("  * Quantum tunneling REDUCES the effective barrier by {:.2} kcal/mol!", reduction);
        println!("    Classical barrier: {:.2} kcal/mol", cl_barrier * 627.509);
        println!("    Quantum barrier:   {:.2} kcal/mol", q_barrier * 627.509);
    }
    if ti_barrier < q_barrier - 0.0001 {
        let ti_diff = (q_barrier - ti_barrier) * 627.509;
        println!("  * TI correction shifts barrier by {:.2} kcal/mol vs primitive", ti_diff);
    } else if (ti_barrier - q_barrier).abs() < 0.0001 {
        println!("  * TI and primitive quantum barriers agree (well-converged with P={}).", n_beads);
    }
    println!();

    // =========================================================================
    // Output files
    // =========================================================================

    // PMF comparison (now includes TI)
    {
        let file = File::create("pimd_umbrella_pmf.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# delta W_classical(Ha) W_quantum(Ha) W_quantum_TI(Ha) W_cl(kcal/mol) W_q(kcal/mol) W_ti(kcal/mol)").unwrap();
        for i in 0..n_hist_bins {
            writeln!(w, "{:.6} {:.8} {:.8} {:.8} {:.4} {:.4} {:.4}",
                     bin_centers[i], cl_pmf[i], q_pmf[i], ti_pmf[i],
                     cl_pmf[i] * 627.509, q_pmf[i] * 627.509, ti_pmf[i] * 627.509).unwrap();
        }
        println!("  PMF (WHAM)              -> pimd_umbrella_pmf.txt");
    }

    // Per-window histograms (includes TI)
    {
        let file = File::create("pimd_umbrella_histograms.txt").unwrap();
        let mut w = BufWriter::new(file);
        write!(w, "# delta").unwrap();
        for (i, center) in window_centers.iter().enumerate() {
            write!(w, " cl_win{}(d0={:.3}) q_win{}(d0={:.3}) ti_win{}(d0={:.3})",
                   i, center, i, center, i, center).unwrap();
        }
        writeln!(w).unwrap();

        let hist_bw = (hist_max - hist_min) / n_hist_bins as f64;
        for j in 0..n_hist_bins {
            let delta = hist_min + (j as f64 + 0.5) * hist_bw;
            write!(w, "{:.6}", delta).unwrap();
            for i in 0..window_centers.len() {
                let cl_p = if cl_windows[i].n_samples > 0 {
                    cl_windows[i].histogram[j] / (cl_windows[i].n_samples as f64 * hist_bw)
                } else {
                    0.0
                };
                let q_p = if q_windows[i].n_samples > 0 {
                    q_windows[i].histogram[j] / (q_windows[i].n_samples as f64 * hist_bw)
                } else {
                    0.0
                };
                let ti_p = if ti_windows[i].n_samples > 0 {
                    ti_windows[i].histogram[j] / (ti_windows[i].n_samples as f64 * hist_bw)
                } else {
                    0.0
                };
                write!(w, " {:.6} {:.6} {:.6}", cl_p, q_p, ti_p).unwrap();
            }
            writeln!(w).unwrap();
        }
        println!("  Per-window histograms   -> pimd_umbrella_histograms.txt");
    }

    // WHAM convergence info (includes TI)
    {
        let file = File::create("pimd_umbrella_convergence.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# WHAM convergence summary").unwrap();
        writeln!(w, "# Classical:  {} iterations", cl_wham_iter).unwrap();
        writeln!(w, "# Quantum:    {} iterations", q_wham_iter).unwrap();
        writeln!(w, "# Quantum+TI: {} iterations", ti_wham_iter).unwrap();
        writeln!(w, "# window center cl_F(Ha) q_F(Ha) ti_F(Ha)").unwrap();
        for i in 0..window_centers.len() {
            writeln!(w, "{:.6} {:.8} {:.8} {:.8}",
                     window_centers[i], cl_free_energies[i], q_free_energies[i],
                     ti_free_energies[i]).unwrap();
        }
        println!("  WHAM convergence        -> pimd_umbrella_convergence.txt");
    }
}

/// Find the free energy barrier in the PMF.
///
/// Strategy: find the minimum on the left (δ < 0), minimum on the right (δ > 0),
/// and maximum near δ = 0. The barrier is max - average(min_left, min_right).
fn find_barrier(pmf: &[f64], bin_centers: &[f64]) -> f64 {
    let n = pmf.len();
    if n == 0 {
        return 0.0;
    }

    // Find index closest to δ = 0
    let mid_idx = bin_centers.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(n / 2);

    // Search for minimum on left side (δ < 0)
    let left_min = if mid_idx > 0 {
        pmf[..mid_idx].iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        pmf[0]
    };

    // Search for minimum on right side (δ > 0)
    let right_min = if mid_idx < n - 1 {
        pmf[(mid_idx + 1)..].iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        pmf[n - 1]
    };

    // Search for maximum near the center (within a few bins)
    let search_range = (n / 10).max(5);
    let start = mid_idx.saturating_sub(search_range);
    let end = (mid_idx + search_range).min(n);
    let center_max = pmf[start..end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Barrier = TS energy - average of the two wells
    let well_avg = (left_min + right_min) / 2.0;
    (center_max - well_avg).max(0.0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pimd_molecular::MolecularPotential;

    /// Simple 1D-like potential for testing: harmonic in x, flat in y/z.
    /// Single atom, 3 DOF.
    #[derive(Clone)]
    struct SimpleTestPotential {
        omega: f64,
        mass: f64,
    }

    impl MolecularPotential for SimpleTestPotential {
        fn n_atoms(&self) -> usize { 3 } // Need at least 3 for donor/proton/acceptor

        fn energy(&self, coords: &[f64]) -> f64 {
            // Simple: just use harmonic on atom 0
            let r2: f64 = coords[..3].iter().map(|&x| x * x).sum();
            0.5 * self.mass * self.omega * self.omega * r2
        }

        fn forces(&self, coords: &[f64], forces: &mut [f64]) {
            let k = self.mass * self.omega * self.omega;
            for d in 0..self.ndof() {
                if d < 3 {
                    forces[d] = -k * coords[d];
                } else {
                    forces[d] = 0.0;
                }
            }
        }

        fn masses(&self) -> &[f64] {
            // Static masses for 3 atoms
            &[1836.15, 1836.15, 1836.15]
        }

        fn reference_geometry(&self) -> Vec<f64> {
            vec![
                -1.0, 0.0, 0.0, // "donor" at x=-1
                 0.0, 0.0, 0.0, // "proton" at origin
                 1.0, 0.0, 0.0, // "acceptor" at x=+1
            ]
        }

        fn name(&self) -> &'static str { "Simple Test" }
    }

    #[test]
    fn test_umbrella_bias_energy() {
        let bias = UmbrellaBias::new(0.5, 0.1);

        // At the center, energy should be zero
        assert!((bias.energy(0.5) - 0.0).abs() < 1e-15);

        // Away from center: V = 0.5 * 0.1 * (1.0 - 0.5)^2 = 0.5 * 0.1 * 0.25 = 0.0125
        assert!((bias.energy(1.0) - 0.0125).abs() < 1e-10);

        // Force at center should be zero
        assert!((bias.force_along_rc(0.5) - 0.0).abs() < 1e-15);

        // Force away from center: F = -0.1 * (1.0 - 0.5) = -0.05
        assert!((bias.force_along_rc(1.0) - (-0.05)).abs() < 1e-10);
    }

    #[test]
    fn test_biased_potential_energy() {
        let pot = SimpleTestPotential { omega: 1.0, mass: 1.0 };
        let bias = UmbrellaBias::new(0.0, 0.1);
        let biased = BiasedMolecularPotential::new(pot.clone(), bias.clone(), 0, 1, 2);

        let coords = vec![
            -1.0, 0.0, 0.0,  // donor
             0.0, 0.0, 0.0,  // proton
             1.0, 0.0, 0.0,  // acceptor
        ];

        let e_phys = pot.energy(&coords);
        let delta = 1.0 - 1.0; // d(donor-proton) - d(acceptor-proton) = 1.0 - 1.0 = 0.0
        let e_bias = bias.energy(delta);
        let e_total = biased.energy(&coords);

        assert!((e_total - (e_phys + e_bias)).abs() < 1e-10,
                "Biased energy {} != physical {} + bias {}",
                e_total, e_phys, e_bias);
    }

    #[test]
    fn test_biased_force_consistency() {
        // Verify that analytical bias forces match numerical finite differences
        use super::super::pimd_molecular::ZundelPES;

        let pes = ZundelPES::new();
        let bias = UmbrellaBias::new(0.3, 0.08);
        let biased = BiasedMolecularPotential::new(pes, bias, 0, 3, 4);

        let geom = biased.inner.reference_geometry();
        let ndof = biased.ndof();

        let mut forces = vec![0.0; ndof];
        biased.forces(&geom, &mut forces);

        let h = 1e-6;
        let mut gp = geom.clone();
        let mut gm = geom.clone();

        for d in 0..ndof {
            gp[d] = geom[d] + h;
            gm[d] = geom[d] - h;
            let f_num = -(biased.energy(&gp) - biased.energy(&gm)) / (2.0 * h);
            gp[d] = geom[d];
            gm[d] = geom[d];

            assert!((forces[d] - f_num).abs() < 1e-3,
                    "Biased force mismatch at dof {}: analytical {:.8} vs numerical {:.8}",
                    d, forces[d], f_num);
        }
    }

    #[test]
    fn test_transfer_coordinate_gradient() {
        // Verify ∂δ/∂R against numerical finite differences
        use super::super::pimd_molecular::ZundelPES;

        let pes = ZundelPES::new();
        let bias = UmbrellaBias::new(0.0, 0.1);
        let biased = BiasedMolecularPotential::new(pes, bias, 0, 3, 4);

        let geom = biased.inner.reference_geometry();
        let ndof = biased.ndof();

        let mut grad = vec![0.0; ndof];
        biased.transfer_coordinate_gradient(&geom, &mut grad);

        let h = 1e-7;
        let mut gp = geom.clone();
        let mut gm = geom.clone();

        for d in 0..ndof {
            gp[d] = geom[d] + h;
            gm[d] = geom[d] - h;
            let delta_p = biased.transfer_coordinate(&gp);
            let delta_m = biased.transfer_coordinate(&gm);
            let grad_num = (delta_p - delta_m) / (2.0 * h);
            gp[d] = geom[d];
            gm[d] = geom[d];

            assert!((grad[d] - grad_num).abs() < 1e-5,
                    "TC gradient mismatch at dof {}: analytical {:.8} vs numerical {:.8}",
                    d, grad[d], grad_num);
        }
    }

    #[test]
    fn test_wham_flat_potential() {
        // For a flat potential (no bias effectively), WHAM should recover uniform distribution
        // Use 3 windows with overlapping histograms of a uniform distribution
        let beta = 1.0;
        let n_bins = 50;
        let hist_min = -2.0;
        let hist_max = 2.0;
        let bin_width = (hist_max - hist_min) / n_bins as f64;

        let window_centers = vec![-0.5, 0.0, 0.5];
        let kappa = 0.5; // Moderate bias

        let mut windows = Vec::new();
        for &center in &window_centers {
            let mut histogram = vec![0.0; n_bins];

            // Generate counts from a biased Gaussian centered at `center`
            // P_biased(x) ∝ exp(-β × 0.5κ(x - center)²) for a flat underlying potential
            let sigma = (1.0_f64 / (beta * kappa)).sqrt();
            for j in 0..n_bins {
                let x = hist_min + (j as f64 + 0.5) * bin_width;
                let bias_e = 0.5 * kappa * (x - center).powi(2);
                histogram[j] = 1000.0 * (-beta * bias_e).exp(); // Synthetic counts
            }

            let n_samples: usize = histogram.iter().map(|&c| c as usize).sum();
            windows.push(UmbrellaWindow {
                center,
                spring_constant: kappa,
                histogram,
                hist_min,
                hist_max,
                n_samples,
                mean_energy: 0.0,
                mean_delta: center,
            });
        }

        let wham = WHAMSolver::new(beta, hist_min, hist_max, n_bins, 500, 1e-8);
        let (pmf, _free_energies, n_iter) = wham.solve(&windows);

        // For a flat underlying potential, the PMF should be approximately flat
        // (constant) across the well-sampled region
        let sampled_region: Vec<f64> = pmf.iter()
            .zip(wham.bin_centers.iter())
            .filter(|(_, &x)| x > -1.0 && x < 1.0) // Focus on well-sampled region
            .map(|(&w, _)| w)
            .collect();

        if !sampled_region.is_empty() {
            let mean_pmf: f64 = sampled_region.iter().sum::<f64>() / sampled_region.len() as f64;
            let max_deviation = sampled_region.iter()
                .map(|&w| (w - mean_pmf).abs())
                .fold(0.0_f64, f64::max);

            // PMF should be flat to within ~0.1 kT
            assert!(max_deviation < 0.1,
                    "PMF should be flat for uniform potential, max deviation = {:.4}", max_deviation);
        }

        assert!(n_iter < 500, "WHAM should converge within 500 iterations");
    }

    #[test]
    fn test_wham_symmetric_double_well() {
        // For a symmetric double well with symmetric windows, the PMF should be symmetric
        let beta = 10.0;
        let n_bins = 100;
        let hist_min = -3.0;
        let hist_max = 3.0;
        let bin_width = (hist_max - hist_min) / n_bins as f64;

        let window_centers = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let kappa = 0.3;

        // Underlying double-well: V(x) = (x² - 1)²
        let mut windows = Vec::new();
        for &center in &window_centers {
            let mut histogram = vec![0.0; n_bins];

            for j in 0..n_bins {
                let x = hist_min + (j as f64 + 0.5) * bin_width;
                let v_phys = (x * x - 1.0).powi(2);
                let v_bias = 0.5 * kappa * (x - center).powi(2);
                histogram[j] = 1000.0 * (-beta * (v_phys + v_bias)).exp();
            }

            let n_samples: usize = histogram.iter().map(|&c| c.max(1.0) as usize).sum();
            windows.push(UmbrellaWindow {
                center,
                spring_constant: kappa,
                histogram,
                hist_min,
                hist_max,
                n_samples,
                mean_energy: 0.0,
                mean_delta: center,
            });
        }

        let wham = WHAMSolver::new(beta, hist_min, hist_max, n_bins, 500, 1e-8);
        let (pmf, _free_energies, n_iter) = wham.solve(&windows);

        // Check symmetry: W(x) ≈ W(-x)
        for j in 0..n_bins / 2 {
            let mirror = n_bins - 1 - j;
            let diff = (pmf[j] - pmf[mirror]).abs();
            // Allow tolerance — symmetry may not be perfect due to binning
            if pmf[j] < 5.0 && pmf[mirror] < 5.0 {
                assert!(diff < 0.5,
                        "PMF asymmetry at bin {}: W({:.2})={:.4} vs W({:.2})={:.4}",
                        j, hist_min + (j as f64 + 0.5) * bin_width, pmf[j],
                        hist_min + (mirror as f64 + 0.5) * bin_width, pmf[mirror]);
            }
        }

        assert!(n_iter < 500, "WHAM should converge for symmetric double well");
    }

    #[test]
    fn test_find_barrier() {
        // Simple test: PMF with a barrier
        let bin_centers: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
        let pmf: Vec<f64> = bin_centers.iter()
            .map(|&x| (x * x - 1.0).powi(2))
            .collect();
        // Minima at x = ±1 (PMF = 0), barrier at x = 0 (PMF = 1.0)

        let barrier = find_barrier(&pmf, &bin_centers);
        assert!((barrier - 1.0).abs() < 0.1,
                "Expected barrier ≈ 1.0, got {:.4}", barrier);
    }
}
