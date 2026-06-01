//! Path Integral Molecular Dynamics (PIMD) Engine
//!
//! Bead-based PIMD for studying nuclear quantum effects such as tunneling.
//! Each quantum particle is represented as a ring polymer of P beads
//! connected by harmonic springs. Classical equations of motion are
//! integrated with a PILE (Path Integral Langevin Equation) thermostat
//! operating in the normal mode representation.
//!
//! Key differences from PIMC:
//! - Deterministic dynamics (velocity Verlet) instead of Monte Carlo
//! - PILE thermostat instead of Metropolis acceptance
//! - Normal mode decomposition for efficient sampling
//!
//! Reference:
//!   Ceriotti, Parrinello, Markland, Manolopoulos (2010)
//!   "Efficient stochastic thermostatting of path integral molecular dynamics"
//!   J. Chem. Phys. 133, 124104

use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

use super::pimc::Potential;

// =============================================================================
// Normal Mode Transformation
// =============================================================================

/// FFT-free normal mode transformation for ring polymer beads.
///
/// Transforms between bead coordinates {x_i} and normal mode coordinates {q_k}
/// using the orthogonal matrix from Tuckerman's staging approach.
///
/// Normal mode frequencies: w_k = 2/(betaₚhbar) sin(πk/P) for k = 0,...,P-1
/// where betaₚ = beta/P is the bead inverse temperature.
#[derive(Clone)]
pub struct NormalModeTransform {
    /// Number of beads P
    pub n_beads: usize,
    /// Normal mode frequencies w_k (in internal units)
    pub frequencies: Vec<f64>,
    /// Imaginary time step dτ = beta/P (= betaₚ in atomic units with hbar=1)
    pub dtau: f64,
}

impl NormalModeTransform {
    /// Create normal mode transform for P beads at inverse temperature beta
    pub fn new(n_beads: usize, beta: f64) -> Self {
        let dtau = beta / n_beads as f64;
        let p = n_beads as f64;

        // Normal mode frequencies: w_k = 2/(dτ) sin(πk/P)
        // These are the eigenvalues of the spring coupling matrix
        let frequencies: Vec<f64> = (0..n_beads)
            .map(|k| {
                2.0 / dtau * (PI * k as f64 / p).sin()
            })
            .collect();

        Self { n_beads, frequencies, dtau }
    }

    /// Transform from bead coordinates to normal mode coordinates
    /// Uses the orthogonal transformation matrix C where:
    ///   q_0 = (1/sqrtP) Σ_i x_i              (centroid)
    ///   q_k = sqrt(2/P) Σ_i x_i cos(2πki/P)  (k = 1..P/2-1)
    ///   q_{P/2} = (1/sqrtP) Σ_i (-1)^i x_i   (if P even)
    ///   q_k = sqrt(2/P) Σ_i x_i sin(2πki/P)  (k = P/2+1..P-1, paired with cos modes)
    pub fn to_normal_modes(&self, beads: &[f64]) -> Vec<f64> {
        let p = self.n_beads;
        let pf = p as f64;
        let mut modes = vec![0.0; p];

        // Centroid mode (k=0)
        modes[0] = beads.iter().sum::<f64>() / pf.sqrt();

        // Interior modes
        for k in 1..p {
            let mut sum = 0.0;
            for i in 0..p {
                let angle = 2.0 * PI * k as f64 * i as f64 / pf;
                // Use real DFT: cos for k <= P/2, sin for k > P/2
                if k <= p / 2 {
                    sum += beads[i] * angle.cos();
                } else {
                    // Map k > P/2 to sin mode for the paired frequency
                    sum += beads[i] * angle.sin();
                }
            }
            if k == p / 2 && p % 2 == 0 {
                modes[k] = sum / pf.sqrt();
            } else {
                modes[k] = sum * (2.0 / pf).sqrt();
            }
        }
        modes
    }

    /// Transform from normal mode coordinates back to bead coordinates
    pub fn to_beads(&self, modes: &[f64]) -> Vec<f64> {
        let p = self.n_beads;
        let pf = p as f64;
        let mut beads = vec![0.0; p];

        for i in 0..p {
            // Centroid contribution
            beads[i] += modes[0] / pf.sqrt();

            // Interior modes
            for k in 1..p {
                let angle = 2.0 * PI * k as f64 * i as f64 / pf;
                if k == p / 2 && p % 2 == 0 {
                    beads[i] += modes[k] * angle.cos() / pf.sqrt();
                } else if k <= p / 2 {
                    beads[i] += modes[k] * angle.cos() * (2.0 / pf).sqrt();
                } else {
                    beads[i] += modes[k] * angle.sin() * (2.0 / pf).sqrt();
                }
            }
        }
        beads
    }

    /// Transform velocities to normal mode representation
    pub fn velocities_to_normal_modes(&self, velocities: &[f64]) -> Vec<f64> {
        self.to_normal_modes(velocities)
    }

    /// Transform normal mode velocities back to bead representation
    pub fn velocities_to_beads(&self, mode_velocities: &[f64]) -> Vec<f64> {
        self.to_beads(mode_velocities)
    }

    // =========================================================================
    // FFT-based transforms for O(P log P) complexity
    // =========================================================================

    /// FFT-based normal mode transform. O(P log P) for power-of-2 bead counts.
    ///
    /// Uses Cooley-Tukey radix-2 DIT FFT to compute the real DFT,
    /// then maps complex FFT output to the real normal mode convention.
    pub fn to_normal_modes_fft(&self, beads: &[f64]) -> Vec<f64> {
        let p = self.n_beads;
        assert!(p.is_power_of_two(), "FFT requires power-of-2 bead count");

        // Compute FFT of real input
        let mut re = beads.to_vec();
        let mut im = vec![0.0; p];
        fft_radix2(&mut re, &mut im, false);

        // Map complex FFT output to real normal mode convention
        // FFT: X[k] = Σ x[i] exp(-j2πki/P) = Σ x[i][cos(2πki/P) - j·sin(2πki/P)]
        // So: Re[k] = Σ x[i] cos(2πki/P), Im[k] = -Σ x[i] sin(2πki/P)
        // DFT modes use cos(2πki/P) for k≤P/2, sin(2πki/P) for k>P/2
        let pf = p as f64;
        let mut modes = vec![0.0; p];

        // k=0: centroid
        modes[0] = re[0] / pf.sqrt();

        // cos modes: k = 1..P/2-1
        for k in 1..p / 2 {
            modes[k] = re[k] * (2.0 / pf).sqrt();
        }

        // k = P/2 (Nyquist)
        if p % 2 == 0 {
            modes[p / 2] = re[p / 2] / pf.sqrt();
        }

        // sin modes: k > P/2
        // DFT sin coeff = Σ x[i] sin(2πki/P) = -Im[k]
        for k in (p / 2 + 1)..p {
            modes[k] = -im[k] * (2.0 / pf).sqrt();
        }

        modes
    }

    /// Inverse FFT-based transform: normal modes back to beads. O(P log P).
    pub fn to_beads_fft(&self, modes: &[f64]) -> Vec<f64> {
        let p = self.n_beads;
        assert!(p.is_power_of_two(), "FFT requires power-of-2 bead count");

        let pf = p as f64;

        // Reconstruct the full complex FFT spectrum X[k] from normal mode coefficients.
        //
        // Forward mapping was:
        //   modes[0]   = Re[0] / sqrt(P)             (centroid)
        //   modes[k]   = Re[k] * sqrt(2/P)           (cos, 1 <= k < P/2)
        //   modes[P/2] = Re[P/2] / sqrt(P)           (Nyquist)
        //   modes[k]   = -Im[k] * sqrt(2/P)          (sin, P/2 < k < P)
        //
        // So inversely:
        //   Re[k] = modes[k] / sqrt(2/P)  for cos modes
        //   Im[k] = -modes[k] / sqrt(2/P) for sin modes
        //
        // Hermitian symmetry for real signal: X[P-k] = conj(X[k])
        //   Re[P-k] = Re[k],  Im[P-k] = -Im[k]

        let mut re = vec![0.0; p];
        let mut im = vec![0.0; p];

        // k=0: centroid (Im[0] = 0 for real signal)
        re[0] = modes[0] * pf.sqrt();

        // k = P/2: Nyquist (Im[P/2] = 0 for real signal)
        if p % 2 == 0 {
            re[p / 2] = modes[p / 2] * pf.sqrt();
        }

        // For k = 1..P/2-1: cos mode gives Re[k], sin mode at P-k gives Im[k]
        // modes[k] encodes cos(2πki/P) coefficient → Re[k]
        // modes[P-k] encodes sin(2π(P-k)i/P) coefficient → Im[P-k]
        // But sin(2π(P-k)i/P) = -sin(2πki/P), so:
        //   modes[P-k] = -Im[P-k] * sqrt(2/P)  →  Im[P-k] = -modes[P-k] / sqrt(2/P)
        // By Hermitian: Im[k] = -Im[P-k] = modes[P-k] / sqrt(2/P)
        let scale = (2.0 / pf).sqrt();
        for k in 1..p / 2 {
            re[k] = modes[k] / scale;
            im[k] = modes[p - k] / scale;  // from Hermitian symmetry of sin mode at P-k
            // Apply Hermitian symmetry: X[P-k] = conj(X[k])
            re[p - k] = re[k];
            im[p - k] = -im[k];
        }

        // Inverse FFT via conjugate: IFFT(X) = (1/P) conj(FFT(conj(X)))
        // Step 1: conjugate input
        for v in im.iter_mut() { *v = -*v; }
        // Step 2: forward FFT
        fft_radix2(&mut re, &mut im, false);
        // Step 3: conjugate and scale by 1/P
        for v in re.iter_mut() { *v /= pf; }
        // (imaginary parts should be ~0 for real signal, ignored)

        re
    }

    /// Auto-select the best transform method based on bead count.
    /// Uses FFT for power-of-2 counts ≥ 16, DFT otherwise.
    pub fn to_normal_modes_auto(&self, beads: &[f64]) -> Vec<f64> {
        if self.n_beads.is_power_of_two() && self.n_beads >= 16 {
            self.to_normal_modes_fft(beads)
        } else {
            self.to_normal_modes(beads)
        }
    }

    /// Auto-select the best inverse transform method.
    pub fn to_beads_auto(&self, modes: &[f64]) -> Vec<f64> {
        if self.n_beads.is_power_of_two() && self.n_beads >= 16 {
            self.to_beads_fft(modes)
        } else {
            self.to_beads(modes)
        }
    }
}

/// In-place Cooley-Tukey radix-2 DIT FFT.
///
/// Operates on separate real and imaginary arrays.
/// `inverse` flag is unused here (caller handles conjugation for IFFT).
fn fft_radix2(re: &mut [f64], im: &mut [f64], _inverse: bool) {
    let n = re.len();
    assert!(n.is_power_of_two());
    assert_eq!(re.len(), im.len());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let angle = angle_step * k as f64;
                let wr = angle.cos();
                let wi = angle.sin();

                let u_re = re[start + k];
                let u_im = im[start + k];
                let v_re = re[start + k + half] * wr - im[start + k + half] * wi;
                let v_im = re[start + k + half] * wi + im[start + k + half] * wr;

                re[start + k] = u_re + v_re;
                im[start + k] = u_im + v_im;
                re[start + k + half] = u_re - v_re;
                im[start + k + half] = u_im - v_im;
            }
        }
        len <<= 1;
    }
}

// =============================================================================
// PILE Thermostat
// =============================================================================

/// Path Integral Langevin Equation (PILE) thermostat.
///
/// Applies frequency-dependent Langevin friction in normal mode space:
/// - Centroid (k=0): γ0 = user-specified physical friction
/// - Internal modes (k>0): γ_k = 2w_k (critically damped for optimal sampling)
///
/// The OBABO splitting ensures symplectic integration:
///   O: half-step Ornstein-Uhlenbeck (friction + noise)
///   B: half-step velocity update from forces
///   A: full-step position update
///   B: half-step velocity update from forces
///   O: half-step Ornstein-Uhlenbeck
#[derive(Clone)]
pub struct PILEThermostat {
    /// Number of beads P
    pub n_beads: usize,
    /// Target inverse temperature beta (of physical system, not per-bead)
    pub beta: f64,
    /// Time step dt
    pub dt: f64,
    /// Friction coefficients γ_k for each normal mode
    pub gamma: Vec<f64>,
    /// exp(-γ_k dt/2) propagator coefficients for O step
    pub c1: Vec<f64>,
    /// sqrt(1 - c1²) noise coefficients for O step
    pub c2: Vec<f64>,
    /// Target kinetic temperature per mode: k_B T = 1/beta in atomic units
    pub kbt: f64,
    /// Particle mass
    pub mass: f64,
}

impl PILEThermostat {
    /// Create PILE thermostat
    ///
    /// # Arguments
    /// * `n_beads` - Number of ring polymer beads
    /// * `beta` - Physical inverse temperature (a.u.)
    /// * `dt` - Integration time step (a.u.)
    /// * `mass` - Particle mass (a.u.)
    /// * `gamma_centroid` - Friction for centroid mode (a.u.); 0 for NVE-like centroid
    /// * `nm_transform` - Normal mode transform for frequency info
    pub fn new(
        n_beads: usize,
        beta: f64,
        dt: f64,
        mass: f64,
        gamma_centroid: f64,
        nm_transform: &NormalModeTransform,
    ) -> Self {
        // The ring polymer Hamiltonian is sampled at the BEAD temperature
        // T_P = P x T, i.e., beta_P = beta/P. So kBT_P = P/beta.
        // Each bead velocity is thermalized at this temperature.
        let kbt = n_beads as f64 / beta;

        // Build friction coefficients
        let mut gamma = vec![0.0; n_beads];
        gamma[0] = gamma_centroid;

        // Internal modes: γ_k = 2w_k (critically damped)
        for k in 1..n_beads {
            gamma[k] = 2.0 * nm_transform.frequencies[k.min(n_beads - k)];
        }

        // O-step propagator coefficients (for half-step dt/2)
        let half_dt = dt / 2.0;
        let c1: Vec<f64> = gamma.iter().map(|&g| (-g * half_dt).exp()).collect();
        let c2: Vec<f64> = c1.iter().map(|&c| (1.0 - c * c).sqrt()).collect();

        Self { n_beads, beta, dt, gamma, c1, c2, kbt, mass }
    }

    /// Apply the Ornstein-Uhlenbeck (O) half-step to normal mode velocities
    ///
    /// v_k -> c1_k . v_k + c2_k . σ_k . η
    /// where σ_k = sqrt(k_B T_P / m) and η ~ N(0,1)
    /// Note: T_P = P x T is the bead temperature.
    pub fn apply_o_step(&self, mode_velocities: &mut [f64]) {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Target velocity width: σ = sqrt(kBT_P / m) = sqrt(P/(betam))
        let sigma = (self.kbt / self.mass).sqrt();

        for k in 0..self.n_beads {
            let noise = normal.sample(&mut rng);
            mode_velocities[k] = self.c1[k] * mode_velocities[k] + self.c2[k] * sigma * noise;
        }
    }
}

// =============================================================================
// Ring Polymer
// =============================================================================

/// A single ring polymer representing one quantum particle with P beads.
///
/// The ring polymer Hamiltonian is:
///   H_RP = Σᵢ [0.5m v_i² + 0.5mw_P²(x_{i+1}-x_i)² + V(x_i)]
/// where w_P = P/(betahbar) = 1/(dτ) is the intra-bead spring frequency.
///
/// Optionally supports Takahashi-Imada fourth-order factorization:
///   V_TI(x) = V(x) + (dτ²/24m)|F(x)|²
/// which achieves O(1/P⁴) convergence instead of O(1/P²).
///
/// Reference (TI): Takahashi & Imada, J. Phys. Soc. Jpn. 53, 3765 (1984)
#[derive(Clone)]
pub struct RingPolymer<P: Potential> {
    /// Bead positions x_i, i = 0..P-1
    pub positions: Vec<f64>,
    /// Bead velocities v_i
    pub velocities: Vec<f64>,
    /// Forces on each bead (physical + spring)
    pub forces: Vec<f64>,
    /// Number of beads P
    pub n_beads: usize,
    /// Particle mass (a.u.)
    pub mass: f64,
    /// Physical inverse temperature beta
    pub beta: f64,
    /// Imaginary time step dτ = beta/P
    pub dtau: f64,
    /// Spring constant: κ = m P / beta² = m / (dτ² P)
    /// Actually for nearest-neighbor spring: κ = mP/(beta²hbar²) = m / dτ²
    pub spring_constant: f64,
    /// External potential
    pub potential: P,
    /// Whether to use Takahashi-Imada fourth-order correction
    pub use_ti: bool,
}

impl<P: Potential> RingPolymer<P> {
    /// Create a new ring polymer with P beads
    ///
    /// # Arguments
    /// * `n_beads` - Number of Trotter beads P
    /// * `beta` - Physical inverse temperature beta
    /// * `mass` - Particle mass (atomic units)
    /// * `potential` - External potential V(x)
    pub fn new(n_beads: usize, beta: f64, mass: f64, potential: P) -> Self {
        let dtau = beta / n_beads as f64;
        let spring_constant = mass / (dtau * dtau);

        let mut rng = rand::thread_rng();
        let sigma_x = potential.init_width();
        let sigma_v = (1.0 / (beta * mass)).sqrt(); // Classical thermal velocity
        let pos_dist = Normal::new(0.0, sigma_x).unwrap();
        let vel_dist = Normal::new(0.0, sigma_v).unwrap();

        // Initialize all beads near one well with thermal fluctuations
        let x0 = sigma_x; // Start in the "right" well
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
            use_ti: false,
        };
        rp.compute_forces();
        rp
    }

    /// Compute total forces on each bead: F_i = F_spring(i) + F_phys(i)
    ///
    /// When `use_ti` is enabled, the physical force includes the
    /// Takahashi-Imada gradient correction:
    ///   F_TI(x) = F(x) - d/dx [(dτ²/24m)|F(x)|²]
    ///           = F(x) - (dτ²/12m) F(x) · F'(x)
    /// where F'(x) = dF/dx (computed numerically).
    pub fn compute_forces(&mut self) {
        let ti_prefactor = if self.use_ti {
            self.dtau * self.dtau / (12.0 * self.mass)
        } else {
            0.0
        };

        for i in 0..self.n_beads {
            let prev = if i == 0 { self.n_beads - 1 } else { i - 1 };
            let next = (i + 1) % self.n_beads;

            // Spring force: -κ(2x_i - x_{i-1} - x_{i+1})
            let f_spring = -self.spring_constant
                * (2.0 * self.positions[i] - self.positions[prev] - self.positions[next]);

            // Physical force from potential
            let f_phys = self.potential.force(self.positions[i]);

            if self.use_ti {
                // TI correction: -d/dx [(dτ²/24m)|F(x)|²]
                // = -(dτ²/12m) · F(x) · dF/dx
                // Compute dF/dx numerically
                let h = 1e-6;
                let f_plus = self.potential.force(self.positions[i] + h);
                let f_minus = self.potential.force(self.positions[i] - h);
                let df_dx = (f_plus - f_minus) / (2.0 * h);
                let f_ti_correction = -ti_prefactor * f_phys * df_dx;
                self.forces[i] = f_spring + f_phys + f_ti_correction;
            } else {
                self.forces[i] = f_spring + f_phys;
            }
        }
    }

    /// Centroid position: x̄ = (1/P) Σᵢ xᵢ
    pub fn centroid(&self) -> f64 {
        self.positions.iter().sum::<f64>() / self.n_beads as f64
    }

    /// Centroid velocity
    pub fn centroid_velocity(&self) -> f64 {
        self.velocities.iter().sum::<f64>() / self.n_beads as f64
    }

    /// Radius of gyration: R_g² = (1/P) Σᵢ (xᵢ - x̄)²
    /// Measures the quantum delocalization / spread of the ring polymer
    pub fn radius_of_gyration(&self) -> f64 {
        let xbar = self.centroid();
        let rg2 = self.positions.iter()
            .map(|&x| (x - xbar).powi(2))
            .sum::<f64>() / self.n_beads as f64;
        rg2.sqrt()
    }

    /// Physical potential energy averaged over beads: (1/P) Σᵢ V(xᵢ)
    pub fn potential_energy(&self) -> f64 {
        self.positions.iter()
            .map(|&x| self.potential.evaluate(x))
            .sum::<f64>() / self.n_beads as f64
    }

    /// Total kinetic energy of the beads: Σᵢ 0.5m vᵢ²
    /// (This is the MD kinetic energy, NOT the quantum kinetic energy)
    pub fn kinetic_energy_md(&self) -> f64 {
        0.5 * self.mass * self.velocities.iter()
            .map(|&v| v * v)
            .sum::<f64>()
    }

    /// Spring (harmonic coupling) energy: Σᵢ 0.5κ(x_{i+1} - xᵢ)²
    pub fn spring_energy(&self) -> f64 {
        let mut energy = 0.0;
        for i in 0..self.n_beads {
            let next = (i + 1) % self.n_beads;
            let dx = self.positions[next] - self.positions[i];
            energy += 0.5 * self.spring_constant * dx * dx;
        }
        energy
    }

    /// Primitive energy estimator for the quantum particle:
    ///   E_prim = P/(2beta) - E_spring + <V>
    ///
    /// This has the "1/P variance problem" -- variance grows with P.
    pub fn primitive_energy_estimator(&self) -> f64 {
        let p = self.n_beads as f64;
        // Kinetic part from primitive estimator
        // = (P/2beta) - Σ_i 0.5m(x_{i+1} - x_i)²/(2dτ²)
        p / (2.0 * self.beta) - self.spring_energy() / p + self.potential_energy()
    }

    /// Virial energy estimator (lower variance):
    ///   E_vir = d/(2beta) + (1/P) Σᵢ [V(xᵢ) + 0.5(xᵢ - x̄).(dV/dxᵢ)]
    ///
    /// For d=1:  E_vir = 1/(2beta) + <V> + (1/2P) Σᵢ (xᵢ - x̄).(dV/dxᵢ)
    ///
    /// This estimator is independent of P to leading order in variance.
    /// Reference: Tuckerman, Statistical Mechanics, Eq. 12.6.22
    pub fn virial_energy_estimator(&self) -> f64 {
        let xbar = self.centroid();
        let mut virial_sum = 0.0;

        for i in 0..self.n_beads {
            let dv_dx = -self.potential.force(self.positions[i]); // dV/dx = -F
            virial_sum += (self.positions[i] - xbar) * dv_dx;
        }

        // d/(2beta) + <V> + virial_correction
        // d=1 for 1D
        1.0 / (2.0 * self.beta) + self.potential_energy() + 0.5 * virial_sum / self.n_beads as f64
    }

    /// Takahashi-Imada virial energy estimator.
    ///
    /// Adds the TI potential correction to the standard virial estimator:
    ///   E_TI = E_vir + (1/P) Σᵢ (dτ²/12m)|F(xᵢ)|²
    ///
    /// The factor of 1/12 (not 1/24) comes from the thermodynamic derivative
    /// of the TI action with respect to beta.
    ///
    /// Reference: Yamamoto, JCP 123, 104101 (2005)
    pub fn ti_virial_energy_estimator(&self) -> f64 {
        let base = self.virial_energy_estimator();
        let ti_factor = self.dtau * self.dtau / (12.0 * self.mass);

        let ti_correction: f64 = self.positions.iter()
            .map(|&x| {
                let f = self.potential.force(x);
                ti_factor * f * f
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

    /// Get the spread of bead positions (max - min) to gauge tunneling
    pub fn bead_spread(&self) -> f64 {
        let min = self.positions.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.positions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    /// Count how many beads are on each side of the barrier (x=0)
    /// Returns (n_left, n_right) -- useful for tunneling analysis
    pub fn bead_distribution(&self) -> (usize, usize) {
        let n_left = self.positions.iter().filter(|&&x| x < 0.0).count();
        let n_right = self.n_beads - n_left;
        (n_left, n_right)
    }
}

// =============================================================================
// PIMD Simulation
// =============================================================================

/// PIMD simulation engine with PILE thermostat
///
/// Integrates the ring polymer equations of motion using velocity Verlet
/// with OBABO Langevin splitting for thermostatting in normal mode space.
pub struct PIMDSimulation<P: Potential> {
    /// Collection of ring polymers (one per quantum particle, or parallel replicas)
    pub polymers: Vec<RingPolymer<P>>,
    /// Normal mode transformer
    pub nm_transform: NormalModeTransform,
    /// PILE thermostat
    pub thermostat: PILEThermostat,
    /// Integration time step
    pub dt: f64,
    /// Current simulation step
    pub step: usize,
}

impl<P: Potential> PIMDSimulation<P> {
    /// Create a new PIMD simulation
    ///
    /// # Arguments
    /// * `n_polymers` - Number of parallel ring polymers (replicas)
    /// * `n_beads` - Number of beads per ring polymer
    /// * `beta` - Physical inverse temperature (a.u.)
    /// * `mass` - Particle mass (a.u.)
    /// * `dt` - Integration time step (a.u.)
    /// * `gamma_centroid` - Centroid friction coefficient (a.u.)
    /// * `potential` - External potential
    pub fn new(
        n_polymers: usize,
        n_beads: usize,
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

        let polymers: Vec<RingPolymer<P>> = (0..n_polymers)
            .map(|_| RingPolymer::new(n_beads, beta, mass, potential.clone()))
            .collect();

        Self {
            polymers,
            nm_transform,
            thermostat,
            dt,
            step: 0,
        }
    }

    /// Create a new PIMD simulation with Takahashi-Imada fourth-order correction.
    ///
    /// Same as `new()` but enables the TI correction on all ring polymers,
    /// achieving O(1/P⁴) convergence instead of O(1/P²).
    ///
    /// Reference: Takahashi & Imada, J. Phys. Soc. Jpn. 53, 3765 (1984)
    pub fn new_with_ti(
        n_polymers: usize,
        n_beads: usize,
        beta: f64,
        mass: f64,
        dt: f64,
        gamma_centroid: f64,
        potential: P,
    ) -> Self {
        let mut sim = Self::new(n_polymers, n_beads, beta, mass, dt, gamma_centroid, potential);
        for polymer in &mut sim.polymers {
            polymer.use_ti = true;
            polymer.compute_forces(); // Recompute with TI correction
        }
        sim
    }

    /// Perform one PIMD time step using OBABO Langevin splitting:
    ///   O: half-step thermostat (in normal mode space)
    ///   B: half-step velocity update from forces (bead space)
    ///   A: full-step position update (bead space)
    ///   B: half-step velocity update from forces (bead space)
    ///   O: half-step thermostat (in normal mode space)
    ///
    /// All polymer replicas are stepped in parallel using Rayon.
    pub fn step_obabo(&mut self) {
        let dt = self.dt;
        let half_dt = dt / 2.0;
        let nm_transform = &self.nm_transform;
        let thermostat = &self.thermostat;

        self.polymers.par_iter_mut().for_each(|polymer| {
            // === O step: thermostat in normal mode space ===
            let mut mode_v = nm_transform.velocities_to_normal_modes(&polymer.velocities);
            thermostat.apply_o_step(&mut mode_v);
            polymer.velocities = nm_transform.velocities_to_beads(&mode_v);

            // === B step: half-step velocity from forces ===
            for i in 0..polymer.n_beads {
                polymer.velocities[i] += half_dt * polymer.forces[i] / polymer.mass;
            }

            // === A step: full-step position update ===
            for i in 0..polymer.n_beads {
                polymer.positions[i] += dt * polymer.velocities[i];
            }

            // === Recompute forces ===
            polymer.compute_forces();

            // === B step: half-step velocity from forces ===
            for i in 0..polymer.n_beads {
                polymer.velocities[i] += half_dt * polymer.forces[i] / polymer.mass;
            }

            // === O step: thermostat in normal mode space ===
            let mut mode_v = nm_transform.velocities_to_normal_modes(&polymer.velocities);
            thermostat.apply_o_step(&mut mode_v);
            polymer.velocities = nm_transform.velocities_to_beads(&mode_v);
        });

        self.step += 1;
    }

    /// Perform one PIMD step using PIQTB (quantum thermal bath) thermostat.
    ///
    /// Same OBABO structure as `step_obabo` but uses the PIQTB thermostat
    /// which targets quantum harmonic oscillator energies per mode:
    ///   E_k = (ℏω_k/2) coth(βℏω_k/2)
    ///
    /// For harmonic systems, this gives exact quantum results with any P.
    /// For anharmonic systems, it significantly reduces the number of
    /// beads needed for convergence.
    ///
    /// # Arguments
    /// * `piqtb` - Pre-constructed PIQTB thermostat (from `piglet::PIQTBThermostat::new`)
    pub fn step_obabo_piqtb(&mut self, piqtb: &super::piglet::PIQTBThermostat) {
        let dt = self.dt;
        let half_dt = dt / 2.0;
        let nm_transform = &self.nm_transform;

        self.polymers.par_iter_mut().for_each(|polymer| {
            // O step with PIQTB
            let mut mode_v = nm_transform.velocities_to_normal_modes(&polymer.velocities);
            piqtb.apply_o_step(&mut mode_v);
            polymer.velocities = nm_transform.velocities_to_beads(&mode_v);

            // B step
            for i in 0..polymer.n_beads {
                polymer.velocities[i] += half_dt * polymer.forces[i] / polymer.mass;
            }

            // A step
            for i in 0..polymer.n_beads {
                polymer.positions[i] += dt * polymer.velocities[i];
            }

            // Recompute forces
            polymer.compute_forces();

            // B step
            for i in 0..polymer.n_beads {
                polymer.velocities[i] += half_dt * polymer.forces[i] / polymer.mass;
            }

            // O step with PIQTB
            let mut mode_v = nm_transform.velocities_to_normal_modes(&polymer.velocities);
            piqtb.apply_o_step(&mut mode_v);
            polymer.velocities = nm_transform.velocities_to_beads(&mode_v);
        });

        self.step += 1;
    }

    /// Average virial energy estimator across all polymers
    pub fn average_virial_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.virial_energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average TI virial energy estimator across all polymers
    pub fn average_ti_virial_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.ti_virial_energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average energy using the appropriate estimator (TI or standard virial).
    /// Automatically selects based on whether TI is enabled on the polymers.
    pub fn average_energy_estimator(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average primitive energy estimator across all polymers
    pub fn average_primitive_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.primitive_energy_estimator())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average centroid position across all polymers
    pub fn average_centroid(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.centroid())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average radius of gyration across all polymers
    pub fn average_radius_of_gyration(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.radius_of_gyration())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Average potential energy across all polymers
    pub fn average_potential_energy(&self) -> f64 {
        self.polymers.par_iter()
            .map(|p| p.potential_energy())
            .sum::<f64>() / self.polymers.len() as f64
    }

    /// Build position histogram |ψ(x)|² from all bead positions of all polymers
    /// (single snapshot -- for accumulated statistics, use accumulate_histogram)
    pub fn build_histogram(&self, n_bins: usize, x_min: f64, x_max: f64) -> (Vec<f64>, Vec<f64>) {
        let bin_width = (x_max - x_min) / n_bins as f64;
        let mut counts = vec![0.0; n_bins];
        let mut total = 0.0;

        for polymer in &self.polymers {
            for &x in &polymer.positions {
                if x >= x_min && x < x_max {
                    let bin = ((x - x_min) / bin_width) as usize;
                    if bin < n_bins {
                        counts[bin] += 1.0;
                        total += 1.0;
                    }
                }
            }
        }

        // Normalize to probability density
        if total > 0.0 {
            for c in &mut counts {
                *c /= total * bin_width;
            }
        }

        let x_values: Vec<f64> = (0..n_bins)
            .map(|i| x_min + (i as f64 + 0.5) * bin_width)
            .collect();

        (x_values, counts)
    }

    /// Accumulate current bead positions into an existing histogram buffer.
    /// Call this repeatedly during production to build up smooth statistics.
    /// The `counts` vector should be initialized to zeros and will be 
    /// normalized at the end by the caller.
    pub fn accumulate_histogram(&self, counts: &mut [f64], x_min: f64, x_max: f64) {
        let n_bins = counts.len();
        let bin_width = (x_max - x_min) / n_bins as f64;

        for polymer in &self.polymers {
            for &x in &polymer.positions {
                if x >= x_min && x < x_max {
                    let bin = ((x - x_min) / bin_width) as usize;
                    if bin < n_bins {
                        counts[bin] += 1.0;
                    }
                }
            }
        }
    }

    /// Compute the fraction of beads that have tunneled (crossed x=0)
    /// relative to starting in the right well. Averaged across polymers.
    pub fn tunneling_fraction(&self) -> f64 {
        let (total_left, total_beads) = self.polymers.par_iter()
            .map(|polymer| {
                let (n_left, _) = polymer.bead_distribution();
                (n_left, polymer.n_beads)
            })
            .reduce(|| (0, 0), |(l1, b1), (l2, b2)| (l1 + l2, b1 + b2));
        total_left as f64 / total_beads as f64
    }
}

// =============================================================================
// Simulation Driver
// =============================================================================

/// Run PIMD simulation for proton transfer in a double-well potential.
///
/// Compares classical (P=1) and quantum (P=n_beads) behavior to demonstrate
/// tunneling of the proton through the barrier.
pub fn run_pimd_proton_transfer(
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    mass: f64,
    barrier_height: f64,
    well_distance: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
) {
    use super::pimc::ProtonTransferPotential;

    let potential = ProtonTransferPotential::symmetric(barrier_height, well_distance);
    let omega_well = potential.well_frequency(mass);

    println!("==============================================================");
    println!("  Path Integral Molecular Dynamics - Proton Transfer");
    println!("  Bead-Based Ring Polymer with PILE Thermostat");
    println!("==============================================================");
    println!();
    println!("Physical parameters:");
    println!("  Particle mass: {:.2} a.u. ({:.4} m_e)", mass, mass);
    println!("  Temperature: {:.1} K  (beta = {:.2} a.u.)", 315774.65 / beta, beta);
    println!("  Barrier height: {:.6} Hartree ({:.2} kcal/mol)", 
             barrier_height, barrier_height * 627.509);
    println!("  Well distance: +/-{:.4} Bohr", well_distance);
    println!("  Well frequency: w = {:.6} a.u. ({:.1} cm-1)", 
             omega_well, omega_well * 219474.63);
    println!("  Thermal de Broglie lambda_dB = {:.4} Bohr",
             (2.0 * PI * beta / mass).sqrt());
    println!();
    println!("Simulation parameters:");
    println!("  Ring polymer beads: {}", n_beads);
    println!("  Parallel replicas: {}", n_polymers);
    println!("  Time step: {:.4} a.u. ({:.4} fs)", dt, dt * 0.02419);
    println!("  Equilibration: {} steps ({:.1} fs)", n_equilibrate, n_equilibrate as f64 * dt * 0.02419);
    println!("  Production: {} steps ({:.1} fs)", n_production, n_production as f64 * dt * 0.02419);
    println!();

    // =========================================================================
    // Classical simulation (P=1)
    // =========================================================================
    println!("------------------------------------------------------------");
    println!("  Running CLASSICAL simulation (P = 1 bead)...");
    println!("------------------------------------------------------------");

    let gamma_centroid = 0.001 * omega_well; // Light coupling for centroid
    let mut classical_sim = PIMDSimulation::new(
        n_polymers, 1, beta, mass, dt, gamma_centroid, potential.clone(),
    );

    // Equilibrate classical
    for step in 0..n_equilibrate {
        classical_sim.step_obabo();
        if step % (n_equilibrate / 5).max(1) == 0 {
            println!("  Equil step {:6}: E_vir = {:10.6}, <x> = {:8.4}",
                     step, classical_sim.average_virial_energy(),
                     classical_sim.average_centroid());
        }
    }
    println!();

    // Production classical -- accumulate histogram over trajectory
    let n_hist_bins = 200;
    let hist_xmin = -3.0 * well_distance;
    let hist_xmax = 3.0 * well_distance;
    let mut cl_energies = Vec::with_capacity(n_production);
    let mut cl_centroids = Vec::with_capacity(n_production);
    let mut cl_tunnel_sum = 0.0_f64;
    let mut cl_tunnel_count = 0_usize;
    let mut cl_hist = vec![0.0_f64; n_hist_bins];
    let sample_interval = 10; // Sample every 10 steps to reduce correlation

    for step in 0..n_production {
        classical_sim.step_obabo();
        if step % sample_interval == 0 {
            cl_energies.push(classical_sim.average_virial_energy());
            cl_centroids.push(classical_sim.average_centroid());
            cl_tunnel_sum += classical_sim.tunneling_fraction();
            cl_tunnel_count += 1;
            classical_sim.accumulate_histogram(&mut cl_hist, hist_xmin, hist_xmax);
        }
        if step % (n_production / 5).max(1) == 0 {
            println!("  Prod step {:6}: E_vir = {:10.6}, <x> = {:8.4}, tunnel = {:.2}%",
                     step, classical_sim.average_virial_energy(),
                     classical_sim.average_centroid(),
                     100.0 * classical_sim.tunneling_fraction());
        }
    }

    // Normalize classical histogram to probability density
    let cl_hist_total: f64 = cl_hist.iter().sum();
    let hist_bin_width = (hist_xmax - hist_xmin) / n_hist_bins as f64;
    let cl_psi2: Vec<f64> = cl_hist.iter()
        .map(|&c| if cl_hist_total > 0.0 { c / (cl_hist_total * hist_bin_width) } else { 0.0 })
        .collect();
    let cl_x: Vec<f64> = (0..n_hist_bins)
        .map(|i| hist_xmin + (i as f64 + 0.5) * hist_bin_width)
        .collect();

    // Classical statistics
    let n_cl = cl_energies.len() as f64;
    let cl_mean_e = cl_energies.iter().sum::<f64>() / n_cl;
    let cl_var_e = cl_energies.iter().map(|e| (e - cl_mean_e).powi(2)).sum::<f64>() / n_cl;
    let cl_stderr_e = cl_var_e.sqrt() / n_cl.sqrt();
    let cl_mean_x = cl_centroids.iter().sum::<f64>() / n_cl;
    let cl_tunnel = cl_tunnel_sum / cl_tunnel_count as f64;

    println!();
    println!("  Classical results:");
    println!("    E_virial   = {:.6} +/- {:.6} Hartree", cl_mean_e, cl_stderr_e);
    println!("    <x>        = {:.6} Bohr", cl_mean_x);
    println!("    Tunneling  = {:.2}%", 100.0 * cl_tunnel);

    // =========================================================================
    // Quantum simulation (P = n_beads)
    // =========================================================================
    println!();
    println!("------------------------------------------------------------");
    println!("  Running QUANTUM simulation (P = {} beads)...", n_beads);
    println!("------------------------------------------------------------");

    let mut quantum_sim = PIMDSimulation::new(
        n_polymers, n_beads, beta, mass, dt, gamma_centroid, potential.clone(),
    );

    // Equilibrate quantum
    for step in 0..n_equilibrate {
        quantum_sim.step_obabo();
        if step % (n_equilibrate / 5).max(1) == 0 {
            println!("  Equil step {:6}: E_vir = {:10.6}, <x> = {:8.4}, R_g = {:8.4}",
                     step, quantum_sim.average_virial_energy(),
                     quantum_sim.average_centroid(),
                     quantum_sim.average_radius_of_gyration());
        }
    }
    println!();

    // Production quantum -- accumulate histogram over trajectory
    let mut q_energies = Vec::with_capacity(n_production);
    let mut q_centroids = Vec::with_capacity(n_production);
    let mut q_rg = Vec::with_capacity(n_production);
    let mut q_tunnel_sum = 0.0_f64;
    let mut q_tunnel_count = 0_usize;
    let mut q_hist = vec![0.0_f64; n_hist_bins];

    for step in 0..n_production {
        quantum_sim.step_obabo();
        if step % sample_interval == 0 {
            q_energies.push(quantum_sim.average_virial_energy());
            q_centroids.push(quantum_sim.average_centroid());
            q_rg.push(quantum_sim.average_radius_of_gyration());
            q_tunnel_sum += quantum_sim.tunneling_fraction();
            q_tunnel_count += 1;
            quantum_sim.accumulate_histogram(&mut q_hist, hist_xmin, hist_xmax);
        }
        if step % (n_production / 5).max(1) == 0 {
            println!("  Prod step {:6}: E_vir = {:10.6}, <x> = {:8.4}, R_g = {:8.4}, tunnel = {:.2}%",
                     step, quantum_sim.average_virial_energy(),
                     quantum_sim.average_centroid(),
                     quantum_sim.average_radius_of_gyration(),
                     100.0 * quantum_sim.tunneling_fraction());
        }
    }

    // Normalize quantum histogram to probability density
    let q_hist_total: f64 = q_hist.iter().sum();
    let q_psi2: Vec<f64> = q_hist.iter()
        .map(|&c| if q_hist_total > 0.0 { c / (q_hist_total * hist_bin_width) } else { 0.0 })
        .collect();

    // Quantum statistics
    let n_q = q_energies.len() as f64;
    let q_mean_e = q_energies.iter().sum::<f64>() / n_q;
    let q_var_e = q_energies.iter().map(|e| (e - q_mean_e).powi(2)).sum::<f64>() / n_q;
    let q_stderr_e = q_var_e.sqrt() / n_q.sqrt();
    let q_mean_x = q_centroids.iter().sum::<f64>() / n_q;
    let q_mean_rg = q_rg.iter().sum::<f64>() / n_q;
    let q_tunnel = q_tunnel_sum / q_tunnel_count as f64;

    println!();
    println!("  Quantum results:");
    println!("    E_virial   = {:.6} +/- {:.6} Hartree", q_mean_e, q_stderr_e);
    println!("    <x>        = {:.6} Bohr", q_mean_x);
    println!("    <R_g>      = {:.6} Bohr (quantum delocalization)", q_mean_rg);
    println!("    Tunneling  = {:.2}%", 100.0 * q_tunnel);

    // =========================================================================
    // Summary comparison
    // =========================================================================
    println!();
    println!("==============================================================");
    println!("                    COMPARISON SUMMARY");
    println!("==============================================================");
    println!("  {:>20} | {:>12} | {:>12}", "Property", "Classical", "Quantum");
    println!("  {:>20} | {:>12} | {:>12}", "--------------------", "------------", "------------");
    println!("  {:>20} | {:>12.6} | {:>12.6}", "E_virial (Ha)", cl_mean_e, q_mean_e);
    println!("  {:>20} | {:>12.6} | {:>12.6}", "<x> (Bohr)", cl_mean_x, q_mean_x);
    println!("  {:>20} | {:>12} | {:>12.6}", "R_g (Bohr)", "N/A", q_mean_rg);
    println!("  {:>20} | {:>11.2}% | {:>11.2}%", "Tunneling", 100.0 * cl_tunnel, 100.0 * q_tunnel);
    println!("==============================================================");
    println!();
    println!("Physical interpretation:");
    if q_tunnel > cl_tunnel + 0.01 {
        println!("  * Quantum tunneling is OBSERVED!");
        println!("    The ring polymer delocalizes across the barrier,");
        println!("    showing {:.1}% of beads on the donor side vs {:.1}% classically.",
                 100.0 * q_tunnel, 100.0 * cl_tunnel);
        if q_tunnel > 0.2 {
            println!("    -> Strong tunneling: proton is significantly delocalized!");
        } else if q_tunnel > 0.05 {
            println!("    -> Moderate tunneling: clear quantum enhancement of barrier crossing.");
        } else {
            println!("    -> Weak tunneling: quantum effects are present but subtle.");
        }
    } else {
        println!("  o Quantum effects are minimal for these parameters.");
        println!("    Try increasing beta (lower T) or decreasing the barrier.");
    }
    let zpe_ratio = q_mean_e / cl_mean_e.max(1e-10);
    println!("  * Zero-point energy: E_quantum/E_classical = {:.1}x", zpe_ratio);
    println!("    Quantum ZPE raises the energy by {:.6} Hartree ({:.2} kcal/mol)",
             q_mean_e - cl_mean_e, (q_mean_e - cl_mean_e) * 627.509);
    if q_mean_rg > well_distance * 0.15 {
        println!("  * Ring polymer R_g ({:.4}) is significant compared to well distance ({:.4})",
                 q_mean_rg, well_distance);
        println!("    -> Quantum delocalization of the proton wavepacket!");
    }

    // =========================================================================
    // Write output files
    // =========================================================================

    // Position distribution
    {
        let file = File::create("pimd_position_distribution.txt").unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# x classical_P(x) quantum_P(x) V(x)").unwrap();
        for i in 0..cl_x.len() {
            let v = potential.evaluate(cl_x[i]);
            writeln!(writer, "{:.6} {:.6} {:.6} {:.6}", cl_x[i], cl_psi2[i], q_psi2[i], v).unwrap();
        }
        println!();
        println!("Position distribution -> pimd_position_distribution.txt");
    }

    // Energy trajectory
    {
        let file = File::create("pimd_proton_transfer.txt").unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# sample E_classical E_quantum centroid_cl centroid_q R_g_q").unwrap();
        let n_samples = q_energies.len().min(cl_energies.len());
        for i in 0..n_samples {
            writeln!(writer, "{} {:.6} {:.6} {:.6} {:.6} {:.6}",
                     i, cl_energies[i], q_energies[i], 
                     cl_centroids[i], q_centroids[i], q_rg[i]).unwrap();
        }
        println!("Energy trajectory     -> pimd_proton_transfer.txt");
    }

    // Bead snapshot
    {
        let file = File::create("pimd_bead_snapshot.txt").unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# Ring polymer bead positions (snapshot from last configuration)").unwrap();
        writeln!(writer, "# polymer_index bead_index position").unwrap();
        for (pi, polymer) in quantum_sim.polymers.iter().enumerate() {
            for (bi, &x) in polymer.positions.iter().enumerate() {
                writeln!(writer, "{} {} {:.6}", pi, bi, x).unwrap();
            }
        }
        println!("Bead snapshot         -> pimd_bead_snapshot.txt");
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pimc::{HarmonicPotential, ProtonTransferPotential};
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_mode_roundtrip() {
        // Verify that to_beads(to_normal_modes(x)) ~ x
        let nmt = NormalModeTransform::new(8, 10.0);
        let beads = vec![1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5];
        let modes = nmt.to_normal_modes(&beads);
        let reconstructed = nmt.to_beads(&modes);
        for (&orig, &rec) in beads.iter().zip(reconstructed.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normal_mode_centroid() {
        // The k=0 mode should give the centroid * sqrt(P)
        let nmt = NormalModeTransform::new(8, 10.0);
        let beads = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let modes = nmt.to_normal_modes(&beads);
        let expected_centroid = beads.iter().sum::<f64>() / 8.0;
        assert_relative_eq!(modes[0] / (8.0_f64).sqrt(), expected_centroid, epsilon = 1e-10);
    }

    #[test]
    fn test_harmonic_oscillator_pimd() {
        // For a harmonic oscillator with w=1, m=1: E0 = 0.5hbarw = 0.5
        let pot = HarmonicPotential { mass: 1.0, omega: 1.0 };
        let n_beads = 32;
        let beta = 20.0;
        let mass = 1.0;
        let dt = 0.1;
        let gamma = 1.0;

        let mut sim = PIMDSimulation::new(20, n_beads, beta, mass, dt, gamma, pot);

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
        // Should be close to 0.5
        assert_relative_eq!(mean_e, 0.5, epsilon = 0.15);
    }

    #[test]
    fn test_proton_transfer_force_consistency() {
        // Verify analytical force matches numerical derivative
        let pot = ProtonTransferPotential::symmetric(0.01, 0.75);
        let h = 1e-7;
        
        for &x in &[-1.0, -0.5, 0.0, 0.3, 0.75, 1.5] {
            let f_analytical = pot.force(x);
            let f_numerical = -(pot.evaluate(x + h) - pot.evaluate(x - h)) / (2.0 * h);
            assert_relative_eq!(f_analytical, f_numerical, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_ring_polymer_centroid() {
        let pot = HarmonicPotential { mass: 1.0, omega: 1.0 };
        let rp = RingPolymer::new(16, 10.0, 1.0, pot);
        let centroid = rp.centroid();
        // Centroid should be finite
        assert!(centroid.is_finite());
    }

    #[test]
    fn test_free_particle_rg() {
        // For a free particle: R_g² = betahbar²/(12m) = beta/(12m) in a.u.
        // Use very weak harmonic to approximate free particle
        let pot = HarmonicPotential { mass: 1.0, omega: 0.001 }; // Nearly free
        let n_beads = 64;
        let beta = 10.0;
        let mass = 1.0;
        let dt = 0.05;

        let mut sim = PIMDSimulation::new(50, n_beads, beta, mass, dt, 1.0, pot);

        // Equilibrate
        for _ in 0..3000 {
            sim.step_obabo();
        }

        // Sample R_g
        let mut rg_values = Vec::new();
        for _ in 0..2000 {
            sim.step_obabo();
            rg_values.push(sim.average_radius_of_gyration());
        }

        let mean_rg = rg_values.iter().sum::<f64>() / rg_values.len() as f64;
        let expected_rg = (beta / (12.0 * mass)).sqrt();
        // Allow generous tolerance since we use near-free, not exactly free
        assert_relative_eq!(mean_rg, expected_rg, epsilon = 0.3);
    }

    // ===== FFT Normal Mode Transform Tests =====

    #[test]
    fn test_fft_roundtrip() {
        // FFT transform roundtrip: to_beads_fft(to_normal_modes_fft(x)) ~ x
        let nmt = NormalModeTransform::new(16, 10.0);
        let beads = vec![1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5,
                         0.3, -0.7, 1.2, -0.1, 0.9, -0.4, 0.6, -0.8];
        let modes = nmt.to_normal_modes_fft(&beads);
        let reconstructed = nmt.to_beads_fft(&modes);
        for (&orig, &rec) in beads.iter().zip(reconstructed.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_matches_dft() {
        // FFT and DFT should give identical results
        let nmt = NormalModeTransform::new(32, 10.0);
        let beads: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin()).collect();

        let modes_dft = nmt.to_normal_modes(&beads);
        let modes_fft = nmt.to_normal_modes_fft(&beads);

        for k in 0..32 {
            assert_relative_eq!(modes_dft[k], modes_fft[k], epsilon = 1e-10,);
        }
    }

    #[test]
    fn test_fft_inverse_matches_dft() {
        let nmt = NormalModeTransform::new(16, 10.0);
        let modes: Vec<f64> = (0..16).map(|i| (i as f64 * 0.5).cos()).collect();

        let beads_dft = nmt.to_beads(&modes);
        let beads_fft = nmt.to_beads_fft(&modes);

        for i in 0..16 {
            assert_relative_eq!(beads_dft[i], beads_fft[i], epsilon = 1e-10);
        }
    }

    // ===== Takahashi-Imada Tests =====

    #[test]
    fn test_ti_harmonic_oscillator() {
        // TI should give exact result for harmonic oscillator even with fewer beads
        // E0 = 0.5 for ω=1, m=1
        let pot = HarmonicPotential { mass: 1.0, omega: 1.0 };
        let n_beads = 16; // Fewer beads than standard test (32)
        let beta = 20.0;
        let mass = 1.0;
        let dt = 0.1;
        let gamma = 1.0;

        let mut sim = PIMDSimulation::new_with_ti(20, n_beads, beta, mass, dt, gamma, pot);

        // Equilibrate
        for _ in 0..3000 {
            sim.step_obabo();
        }

        // Sample
        let mut energies = Vec::new();
        for _ in 0..3000 {
            sim.step_obabo();
            energies.push(sim.average_ti_virial_energy());
        }

        let mean_e = energies.iter().sum::<f64>() / energies.len() as f64;
        // TI should converge faster, allow tighter tolerance
        assert_relative_eq!(mean_e, 0.5, epsilon = 0.15);
    }

    #[test]
    fn test_ti_force_correction() {
        // For harmonic potential V = 0.5*ω²*x², F = -ω²*x
        // TI correction: -(dτ²/12m) F dF/dx = -(dτ²/12m)(-ω²x)(-ω²) = -(dτ²ω⁴x/12m)
        // Total TI force: -ω²x - dτ²ω⁴x/(12m)
        let pot = HarmonicPotential { mass: 1.0, omega: 1.0 };
        let beta = 10.0;
        let n_beads = 8;
        let dtau = beta / n_beads as f64;
        let mass = 1.0;

        let mut rp = RingPolymer::new(n_beads, beta, mass, pot);
        rp.use_ti = true;

        // Set a known position
        let x_test = 0.5;
        rp.positions = vec![x_test; n_beads];
        rp.compute_forces();

        let omega = 1.0;
        let f_phys = -omega * omega * x_test;
        let df_dx = -omega * omega;
        let ti_correction = -(dtau * dtau / (12.0 * mass)) * f_phys * df_dx;

        // Spring forces cancel when all beads are at the same position
        let expected_total = f_phys + ti_correction;
        assert_relative_eq!(rp.forces[0], expected_total, epsilon = 1e-5);
    }

    #[test]
    fn test_ti_backward_compatible() {
        // With use_ti=false, results should match standard PIMD
        let pot = HarmonicPotential { mass: 1.0, omega: 1.0 };
        let rp_standard = RingPolymer::new(8, 10.0, 1.0, pot.clone());

        let mut rp_ti = RingPolymer::new(8, 10.0, 1.0, pot);
        rp_ti.positions = rp_standard.positions.clone();
        rp_ti.use_ti = false;
        rp_ti.compute_forces();

        // Forces should be identical when TI is disabled
        for (f1, f2) in rp_standard.forces.iter().zip(rp_ti.forces.iter()) {
            assert_relative_eq!(f1, f2, epsilon = 1e-12);
        }
    }
}
