//! Core ring-polymer machinery shared across PIMD samplers:
//! the normal-mode transform and the PILE thermostat.
//!
//! These types were originally defined in `pimd.rs`; they live here so that both
//! the 1D (`pimd`) and molecular (`pimd_molecular`, `pimd_rpc`) drivers can depend
//! on them without reaching across sibling modules.

use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
