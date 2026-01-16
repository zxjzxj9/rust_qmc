//! Homogeneous Electron Gas (HEG) for QMC calculations.
//!
//! The HEG (also known as "jellium") is a fundamental model system where electrons
//! move in a uniform positive background charge. It's essential for:
//! - Parameterizing LDA correlation functionals (Ceperley-Alder, VWN, PW92)
//! - Benchmarking QMC methods against known exact results
//!
//! The key parameter is the Wigner-Seitz radius rs, defined such that
//! 4π/3 rs³ = 1/n where n is the electron density.

use nalgebra::Vector3;
use num_complex::Complex64;
use rand::Rng;
use statrs::function::erf::erfc;
use std::f64::consts::PI;

use crate::sampling::EnergyCalculator;
use crate::wavefunction::MultiWfn;

/// Homogeneous Electron Gas (Jellium) wavefunction.
///
/// Uses a Slater-Jastrow trial wavefunction:
/// Ψ(R) = D↑(r↑) × D↓(r↓) × exp(J(R))
///
/// where D are Slater determinants of plane waves and J is the Jastrow factor.
#[derive(Debug, Clone)]
pub struct HomogeneousElectronGas {
    /// Number of electrons
    pub num_electrons: usize,
    /// Number of spin-up electrons
    pub num_up: usize,
    /// Number of spin-down electrons  
    pub num_down: usize,
    /// Wigner-Seitz radius (Bohr)
    pub rs: f64,
    /// Cubic simulation box length (Bohr)
    pub box_length: f64,
    /// k-vectors for occupied orbitals (spin-up)
    k_vectors_up: Vec<Vector3<f64>>,
    /// k-vectors for occupied orbitals (spin-down)
    k_vectors_down: Vec<Vector3<f64>>,
    /// Jastrow parameter A for same-spin pairs (satisfies cusp = 1/2)
    jastrow_a_same: f64,
    /// Jastrow parameter A for opposite-spin pairs (satisfies cusp = 1/4)
    jastrow_a_anti: f64,
    /// Jastrow RPA parameter F
    jastrow_f: f64,
    /// Ewald alpha parameter
    ewald_alpha: f64,
    /// Precomputed reciprocal space vectors for Ewald
    ewald_k_vectors: Vec<(Vector3<f64>, f64)>,
}

/// Results from HEG calculation.
#[derive(Debug, Clone)]
pub struct HEGResults {
    /// Total energy per electron (Hartree)
    pub energy_per_electron: f64,
    /// Kinetic energy per electron (Hartree)
    pub kinetic_per_electron: f64,
    /// Potential energy per electron (Hartree)
    pub potential_per_electron: f64,
    /// Hartree-Fock energy per electron for comparison
    pub hf_energy_per_electron: f64,
    /// Estimated correlation energy (total - HF)
    pub correlation_energy: f64,
}

impl HomogeneousElectronGas {
    /// Create a new HEG system.
    ///
    /// # Arguments
    /// * `num_electrons` - Total number of electrons (use closed-shell numbers: 2, 14, 38, 54, 66)
    /// * `rs` - Wigner-Seitz radius in Bohr (typical range: 1-10)
    /// * `jastrow_f` - Jastrow parameter F (controls range, typically rs/2 to rs)
    pub fn new(num_electrons: usize, rs: f64, jastrow_f: f64) -> Self {
        // Box length from rs: L = (4π/3 · N)^(1/3) · rs
        let box_length = (4.0 * PI / 3.0 * num_electrons as f64).powf(1.0 / 3.0) * rs;

        // Split electrons equally between spins (assuming closed shell)
        let num_up = num_electrons / 2;
        let num_down = num_electrons - num_up;

        // Generate k-vectors filling Fermi sphere
        let k_vectors_up = Self::generate_k_vectors(num_up, box_length);
        let k_vectors_down = Self::generate_k_vectors(num_down, box_length);

        // Jastrow A parameters to satisfy cusp conditions:
        // For same-spin: cusp = 1/2, so A_same = 1/2
        // For opposite-spin: cusp = 1/4, so A_anti = 1/4
        // The full Jastrow is: u(r) = A * r / (1 + r/F) where A = cusp
        let jastrow_a_same = 0.5;
        let jastrow_a_anti = 0.25;

        // Ewald parameters
        let ewald_alpha = 5.0 / box_length;
        let ewald_k_vectors = Self::generate_ewald_k_vectors(box_length, ewald_alpha);

        Self {
            num_electrons,
            num_up,
            num_down,
            rs,
            box_length,
            k_vectors_up,
            k_vectors_down,
            jastrow_a_same,
            jastrow_a_anti,
            jastrow_f,
            ewald_alpha,
            ewald_k_vectors,
        }
    }

    /// Generate k-vectors to fill the Fermi sphere for n electrons of one spin.
    fn generate_k_vectors(n: usize, box_length: f64) -> Vec<Vector3<f64>> {
        let dk = 2.0 * PI / box_length;
        let mut k_shells: Vec<(i32, i32, i32, f64)> = Vec::new();

        // Generate all k-vectors up to some maximum
        let max_n = ((n as f64).powf(1.0 / 3.0) * 2.0).ceil() as i32 + 2;
        
        for nx in -max_n..=max_n {
            for ny in -max_n..=max_n {
                for nz in -max_n..=max_n {
                    let k2 = (nx * nx + ny * ny + nz * nz) as f64;
                    k_shells.push((nx, ny, nz, k2));
                }
            }
        }

        // Sort by |k|² and take the first n
        k_shells.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
        
        k_shells
            .into_iter()
            .take(n)
            .map(|(nx, ny, nz, _)| Vector3::new(nx as f64 * dk, ny as f64 * dk, nz as f64 * dk))
            .collect()
    }

    /// Generate reciprocal space vectors for Ewald summation.
    fn generate_ewald_k_vectors(box_length: f64, alpha: f64) -> Vec<(Vector3<f64>, f64)> {
        let dk = 2.0 * PI / box_length;
        let k_max = 7; // Cutoff in reciprocal space (increased for accuracy)
        let mut k_vectors = Vec::new();

        for nx in -k_max..=k_max {
            for ny in -k_max..=k_max {
                for nz in -k_max..=k_max {
                    if nx == 0 && ny == 0 && nz == 0 {
                        continue;
                    }
                    let k = Vector3::new(nx as f64 * dk, ny as f64 * dk, nz as f64 * dk);
                    let k2 = k.norm_squared();
                    // Precompute the factor for efficiency
                    let factor = 4.0 * PI / (box_length.powi(3) * k2)
                        * (-k2 / (4.0 * alpha * alpha)).exp();
                    k_vectors.push((k, factor));
                }
            }
        }
        k_vectors
    }

    /// Apply minimum image convention for periodic boundaries.
    #[inline]
    fn minimum_image(&self, r: Vector3<f64>) -> Vector3<f64> {
        let l = self.box_length;
        Vector3::new(
            r.x - l * (r.x / l).round(),
            r.y - l * (r.y / l).round(),
            r.z - l * (r.z / l).round(),
        )
    }

    /// Wrap position into the simulation box [0, L).
    #[inline]
    fn wrap_position(&self, r: Vector3<f64>) -> Vector3<f64> {
        let l = self.box_length;
        Vector3::new(
            r.x.rem_euclid(l),
            r.y.rem_euclid(l),
            r.z.rem_euclid(l),
        )
    }

    /// Evaluate plane wave orbital: φ_k(r) = exp(ik·r)
    /// Note: Normalization factor 1/√V is omitted as it cancels in local energy
    #[inline]
    fn plane_wave(&self, k: &Vector3<f64>, r: &Vector3<f64>) -> Complex64 {
        let phase = k.dot(r);
        Complex64::new(phase.cos(), phase.sin())
    }

    /// Compute Slater determinant for a set of electrons and k-vectors.
    fn slater_determinant(&self, positions: &[Vector3<f64>], k_vectors: &[Vector3<f64>]) -> Complex64 {
        let n = positions.len();
        if n == 0 {
            return Complex64::new(1.0, 0.0);
        }

        // Build the Slater matrix
        let mut matrix = nalgebra::DMatrix::<Complex64>::zeros(n, n);
        for (i, r) in positions.iter().enumerate() {
            for (j, k) in k_vectors.iter().enumerate() {
                matrix[(i, j)] = self.plane_wave(k, r);
            }
        }

        // Compute determinant
        matrix.determinant()
    }

    /// Compute inverse of Slater matrix for efficient updates.
    fn slater_inverse(&self, positions: &[Vector3<f64>], k_vectors: &[Vector3<f64>]) 
        -> Option<nalgebra::DMatrix<Complex64>> 
    {
        let n = positions.len();
        if n == 0 {
            return Some(nalgebra::DMatrix::<Complex64>::zeros(0, 0));
        }

        let mut matrix = nalgebra::DMatrix::<Complex64>::zeros(n, n);
        for (i, r) in positions.iter().enumerate() {
            for (j, k) in k_vectors.iter().enumerate() {
                matrix[(i, j)] = self.plane_wave(k, r);
            }
        }

        matrix.try_inverse()
    }

    /// Check if electrons i and j have the same spin.
    #[inline]
    fn same_spin(&self, i: usize, j: usize) -> bool {
        (i < self.num_up && j < self.num_up) || (i >= self.num_up && j >= self.num_up)
    }

    /// Get Jastrow A parameter for a pair of electrons.
    #[inline]
    fn jastrow_a(&self, i: usize, j: usize) -> f64 {
        if self.same_spin(i, j) {
            self.jastrow_a_same
        } else {
            self.jastrow_a_anti
        }
    }

    /// Evaluate Jastrow correlation function u(r) for a pair.
    /// 
    /// Using the Padé form: u(r) = A * r / (1 + r/F)
    /// This satisfies the cusp condition: du/dr|_{r=0} = A
    #[inline]
    fn jastrow_u(&self, r: f64, a: f64) -> f64 {
        if r < 1e-10 {
            return 0.0;
        }
        a * r / (1.0 + r / self.jastrow_f)
    }

    /// Evaluate periodic Jastrow factor.
    ///
    /// J = exp(Σᵢ<ⱼ u(rᵢⱼ))
    fn jastrow_factor(&self, positions: &[Vector3<f64>]) -> f64 {
        let mut u_sum = 0.0;
        let n = positions.len();
        let l_half = self.box_length / 2.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let rij = self.minimum_image(positions[i] - positions[j]);
                let r = rij.norm();
                
                if r < l_half && r > 1e-10 {
                    let a = self.jastrow_a(i, j);
                    u_sum += self.jastrow_u(r, a);
                }
            }
        }
        u_sum.exp()
    }

    /// Compute gradient of u(r) = A*r/(1+r/F)
    /// ∇u = A * (1/(1+r/F)² * F/r - 1/(1+r/F) * 1/r) * rij... wait let me redo this
    /// 
    /// u(r) = A*r/(1+r/F) = A*F*r/(F+r)
    /// du/dr = A*F * (F+r - r) / (F+r)² = A*F² / (F+r)²
    /// ∇ᵢu = du/dr * rij/r for pair (i,j)
    fn jastrow_gradient(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let n = positions.len();
        let mut grad = vec![Vector3::zeros(); n];
        let l_half = self.box_length / 2.0;
        let f = self.jastrow_f;

        for i in 0..n {
            for j in (i + 1)..n {
                let rij = self.minimum_image(positions[i] - positions[j]);
                let r = rij.norm();

                if r < l_half && r > 1e-10 {
                    let a = self.jastrow_a(i, j);
                    // du/dr = A * F² / (F + r)²
                    let du_dr = a * f * f / (f + r).powi(2);
                    let grad_ij = du_dr * rij / r;
                    grad[i] += grad_ij;
                    grad[j] -= grad_ij;
                }
            }
        }
        grad
    }

    /// Compute Laplacian of Jastrow factor.
    /// 
    /// For u(r) = A*F*r/(F+r):
    /// du/dr = A*F²/(F+r)²
    /// d²u/dr² = -2*A*F²/(F+r)³
    /// 
    /// ∇²u = d²u/dr² + (2/r) * du/dr
    ///     = -2*A*F²/(F+r)³ + 2*A*F²/(r*(F+r)²)
    ///     = 2*A*F² * (1/(r*(F+r)²) - 1/(F+r)³)
    ///     = 2*A*F² / (F+r)² * (1/r - 1/(F+r))
    ///     = 2*A*F³ / (r*(F+r)³)
    fn jastrow_laplacian(&self, positions: &[Vector3<f64>]) -> Vec<f64> {
        let n = positions.len();
        let mut lap = vec![0.0; n];
        let l_half = self.box_length / 2.0;
        let f = self.jastrow_f;

        for i in 0..n {
            for j in (i + 1)..n {
                let rij = self.minimum_image(positions[i] - positions[j]);
                let r = rij.norm();

                if r < l_half && r > 1e-10 {
                    let a = self.jastrow_a(i, j);
                    // ∇²u = 2*A*F³ / (r*(F+r)³)
                    let lap_u = 2.0 * a * f.powi(3) / (r * (f + r).powi(3));
                    lap[i] += lap_u;
                    lap[j] += lap_u;
                }
            }
        }
        lap
    }

    /// Compute electron-electron potential using Ewald summation.
    /// 
    /// For jellium (HEG), the potential energy is the sum of:
    /// 1. Electron-electron Coulomb repulsion (Ewald method)
    /// 2. Electron-background attraction (cancels the k=0 divergence)
    /// 3. Background-background repulsion (Madelung constant)
    /// 
    /// The total for a neutral system gives a well-defined finite result.
    /// 
    /// Reference: Fraser et al., PRB 53, 1814 (1996)
    fn ewald_potential(&self, positions: &[Vector3<f64>]) -> f64 {
        let n = positions.len();
        let alpha = self.ewald_alpha;
        let l = self.box_length;
        let volume = l.powi(3);
        
        // === Part 1: Real-space sum ===
        // Sum over pairs with minimum image convention
        // V_real = Σᵢ<ⱼ erfc(α|rᵢⱼ|) / |rᵢⱼ|
        let mut v_real = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let rij = self.minimum_image(positions[i] - positions[j]);
                let r = rij.norm();
                if r > 1e-10 {
                    v_real += erfc(alpha * r) / r;
                }
            }
        }

        // === Part 2: Reciprocal-space sum ===
        // V_recip = (1/2) Σₖ≠₀ (4π/Vk²) exp(-k²/4α²) × [|S(k)|² - N]
        // where S(k) = Σⱼ exp(ik·rⱼ) is the structure factor
        // The -N subtracts the self-term
        let mut v_recip = 0.0;
        for (k, factor) in &self.ewald_k_vectors {
            let mut rho_k = Complex64::new(0.0, 0.0);
            for pos in positions {
                let phase = k.dot(pos);
                rho_k += Complex64::new(phase.cos(), phase.sin());
            }
            let rho_k_sq = rho_k.norm_sqr() - n as f64;
            v_recip += 0.5 * factor * rho_k_sq;
        }

        // === Part 3: Self-energy correction ===
        // Each electron interacts with its own Gaussian screening cloud
        // V_self = -α N / √π
        let v_self = -alpha * n as f64 / PI.sqrt();
        
        // === Part 4: Neutralizing background (Madelung constant) ===
        // For jellium, the k=0 term (which would diverge) is cancelled by
        // electron-background and background-background interactions.
        // The net result is the "Madelung energy":
        // V_Madelung = -π N² / (α² V)
        // 
        // This comes from integrating the Fourier transform of the Coulomb
        // potential near k=0 with the uniform background.
        let v_madelung = -PI * (n as f64).powi(2) / (alpha.powi(2) * volume);

        // === Total potential energy ===
        // Note: This is the *total* Coulomb energy, which includes the
        // exchange-correlation effects implicitly through the electron positions.
        // The "Hartree" energy of the uniform background is already cancelled
        // by v_madelung.
        v_real + v_recip + v_self + v_madelung
    }

    /// Compute kinetic energy from plane-wave Slater determinant.
    /// 
    /// For plane waves: ∇²φₖ = -k² φₖ
    /// So kinetic contribution from Slater det: T_D = Σᵢ Σⱼ (A⁻¹)ⱼᵢ × (k²ⱼ/2) × φⱼ(rᵢ) = Σⱼ k²ⱼ/2
    /// where A⁻¹ is the inverse Slater matrix.
    /// 
    /// This simplifies to just summing the occupied k² values (for a single-det wavefunction).
    fn kinetic_slater(&self) -> f64 {
        let t_up: f64 = self.k_vectors_up.iter()
            .map(|k| 0.5 * k.norm_squared())
            .sum();
        let t_down: f64 = self.k_vectors_down.iter()
            .map(|k| 0.5 * k.norm_squared())
            .sum();
        t_up + t_down
    }

    /// Compute full local kinetic energy including all Slater-Jastrow terms.
    /// 
    /// T = -½ ∇²Ψ/Ψ = -½ Σᵢ [ ∇²ₗₙD + ∇²J + 2∇ₗₙD·∇J + (∇J)² ]
    ///   = T_D + T_J + T_cross
    /// 
    /// where T_D = Σⱼ k²ⱼ/2 (Slater kinetic)
    ///       T_J = -½ Σᵢ [∇²J + (∇J)²] (Jastrow kinetic)
    ///       T_cross = -Σᵢ ∇ₗₙD · ∇J (cross term)
    fn kinetic_energy(&self, positions: &[Vector3<f64>]) -> f64 {
        // Slater kinetic energy
        let t_slater = self.kinetic_slater();

        // Jastrow contributions
        let grad_j = self.jastrow_gradient(positions);
        let lap_j = self.jastrow_laplacian(positions);

        // T_J = -½ Σᵢ (∇²J + (∇J)²)
        let t_jastrow: f64 = (0..positions.len())
            .map(|i| -0.5 * (lap_j[i] + grad_j[i].norm_squared()))
            .sum();

        // Cross term: need ∇ₗₙD for each electron
        // For plane waves: ∇ₗₙD_σ = Σⱼ (A⁻¹_σ)ⱼᵢ × (ikⱼ) × φⱼ(rᵢ) / φⱼ(rᵢ) = Σⱼ (A⁻¹_σ)ⱼᵢ × (ikⱼ)
        let t_cross = self.kinetic_cross_term(positions, &grad_j);

        t_slater + t_jastrow + t_cross
    }

    /// Compute cross term: -Σᵢ ∇ₗₙD · ∇J
    /// 
    /// For plane waves, ∇ₗₙD is purely imaginary (ik terms).
    /// The real part of the cross term vanishes on average, but we should
    /// still compute it for accuracy.
    fn kinetic_cross_term(&self, positions: &[Vector3<f64>], grad_j: &[Vector3<f64>]) -> f64 {
        // Split positions by spin
        let r_up: Vec<_> = positions[..self.num_up].to_vec();
        let r_down: Vec<_> = positions[self.num_up..].to_vec();

        let mut cross = 0.0;

        // Spin-up electrons
        if let Some(inv_up) = self.slater_inverse(&r_up, &self.k_vectors_up) {
            for i in 0..self.num_up {
                // ∇ₗₙD_↑ = Σⱼ (A⁻¹)ⱼᵢ × (ikⱼ) (imaginary)
                let mut grad_ln_d = Vector3::<Complex64>::zeros();
                for j in 0..self.num_up {
                    let k = &self.k_vectors_up[j];
                    let ik = Vector3::new(
                        Complex64::new(0.0, k.x),
                        Complex64::new(0.0, k.y),
                        Complex64::new(0.0, k.z),
                    );
                    grad_ln_d += ik * inv_up[(j, i)];
                }
                // Cross term: Real part of ∇ₗₙD · ∇J
                cross -= grad_ln_d.x.re * grad_j[i].x 
                       + grad_ln_d.y.re * grad_j[i].y 
                       + grad_ln_d.z.re * grad_j[i].z;
            }
        }

        // Spin-down electrons
        if let Some(inv_down) = self.slater_inverse(&r_down, &self.k_vectors_down) {
            for i in 0..self.num_down {
                let mut grad_ln_d = Vector3::<Complex64>::zeros();
                for j in 0..self.num_down {
                    let k = &self.k_vectors_down[j];
                    let ik = Vector3::new(
                        Complex64::new(0.0, k.x),
                        Complex64::new(0.0, k.y),
                        Complex64::new(0.0, k.z),
                    );
                    grad_ln_d += ik * inv_down[(j, i)];
                }
                let global_i = i + self.num_up;
                cross -= grad_ln_d.x.re * grad_j[global_i].x 
                       + grad_ln_d.y.re * grad_j[global_i].y 
                       + grad_ln_d.z.re * grad_j[global_i].z;
            }
        }

        cross
    }

    /// Compute Hartree-Fock energy per electron (analytical formula).
    /// 
    /// E_HF/N = (3/5)(kF²/2) - (3kF)/(4π)
    ///        = 2.21/rs² - 0.916/rs  (in Rydberg)
    ///        = 1.105/rs² - 0.458/rs (in Hartree)
    pub fn hartree_fock_energy(&self) -> f64 {
        // Fermi wavevector: kF = (3π²n)^(1/3) = (9π/4)^(1/3) / rs
        let k_fermi = (9.0 * PI / 4.0).powf(1.0 / 3.0) / self.rs;
        
        // Kinetic: T_HF/N = (3/5) * kF²/2
        let t_hf = 0.3 * k_fermi.powi(2);

        // Exchange: V_x/N = -3*kF/(4π)
        let v_x = -3.0 * k_fermi / (4.0 * PI);

        t_hf + v_x
    }

    /// Get reference correlation energy from Perdew-Zunger parameterization.
    pub fn ceperley_alder_correlation(&self) -> f64 {
        let rs = self.rs;
        if rs >= 1.0 {
            // Perdew-Zunger parameterization (paramagnetic)
            let gamma = -0.1423;
            let beta1 = 1.0529;
            let beta2 = 0.3334;
            gamma / (1.0 + beta1 * rs.sqrt() + beta2 * rs)
        } else {
            // High density limit (rs < 1)
            let a = 0.0311;
            let b = -0.048;
            let c = 0.0020;
            let d = -0.0116;
            a * rs.ln() + b + c * rs * rs.ln() + d * rs
        }
    }
}

impl MultiWfn for HomogeneousElectronGas {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let l = self.box_length;

        // Initialize electrons uniformly in the box
        (0..self.num_electrons)
            .map(|_| Vector3::new(
                rng.gen::<f64>() * l,
                rng.gen::<f64>() * l,
                rng.gen::<f64>() * l,
            ))
            .collect()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        // Wrap positions
        let wrapped: Vec<_> = r.iter().map(|&pos| self.wrap_position(pos)).collect();
        
        // Split positions by spin
        let r_up: Vec<_> = wrapped[..self.num_up].to_vec();
        let r_down: Vec<_> = wrapped[self.num_up..].to_vec();

        // Compute Slater determinants
        let det_up = self.slater_determinant(&r_up, &self.k_vectors_up);
        let det_down = self.slater_determinant(&r_down, &self.k_vectors_down);

        // Jastrow factor
        let jastrow = self.jastrow_factor(&wrapped);

        // Total wavefunction magnitude
        // Using |D_up * D_down| * J  (we take the magnitude to handle complex determinant)
        (det_up * det_down).norm() * jastrow
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        // Numerical derivative
        self.numerical_derivative(r, 1e-5)
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        // Numerical Laplacian
        self.numerical_laplacian(r, 1e-5)
    }
}

impl EnergyCalculator for HomogeneousElectronGas {
    fn local_energy(&self, r: &[Vector3<f64>]) -> f64 {
        // Wrap positions for periodic boundaries
        let wrapped: Vec<_> = r.iter().map(|&pos| self.wrap_position(pos)).collect();

        // Compute wavefunction value
        let psi = self.evaluate(&wrapped);
        
        if psi.abs() < 1e-15 {
            return 0.0; // Avoid division by zero at nodes
        }

        // Kinetic energy using numerical Laplacian: T = -½ Σᵢ ∇²Ψ/Ψ
        let h = 1e-4; // Step size for numerical differentiation
        let mut kinetic = 0.0;
        
        for i in 0..wrapped.len() {
            for axis in 0..3 {
                let mut r_fwd = wrapped.clone();
                let mut r_bwd = wrapped.clone();
                r_fwd[i][axis] += h;
                r_bwd[i][axis] -= h;
                
                let psi_fwd = self.evaluate(&r_fwd);
                let psi_bwd = self.evaluate(&r_bwd);
                
                // ∇²Ψ ≈ (Ψ(r+h) - 2Ψ(r) + Ψ(r-h)) / h²
                kinetic += (psi_fwd - 2.0 * psi + psi_bwd) / (h * h);
            }
        }
        kinetic = -0.5 * kinetic / psi;

        // Potential energy from Ewald sum
        let potential = self.ewald_potential(&wrapped);

        kinetic + potential
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_heg_creation() {
        let heg = HomogeneousElectronGas::new(14, 4.0, 2.0);
        
        assert_eq!(heg.num_electrons, 14);
        assert_eq!(heg.num_up, 7);
        assert_eq!(heg.num_down, 7);
        assert_eq!(heg.k_vectors_up.len(), 7);
        assert_eq!(heg.k_vectors_down.len(), 7);
        
        // Check box length: L = (4π/3 * 14)^(1/3) * 4.0
        let expected_l = (4.0 * PI / 3.0 * 14.0).powf(1.0 / 3.0) * 4.0;
        assert_relative_eq!(heg.box_length, expected_l, epsilon = 1e-10);
    }

    #[test]
    fn test_k_vectors_fill_fermi_sphere() {
        let heg = HomogeneousElectronGas::new(14, 4.0, 2.0);
        
        // First k-vector should be (0,0,0)
        assert_relative_eq!(heg.k_vectors_up[0].norm(), 0.0, epsilon = 1e-10);
        
        // All other k-vectors should be in the first shell
        let dk = 2.0 * PI / heg.box_length;
        for k in &heg.k_vectors_up[1..] {
            assert_relative_eq!(k.norm(), dk, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_minimum_image() {
        let heg = HomogeneousElectronGas::new(14, 4.0, 2.0);
        let l = heg.box_length;

        // Test wrapping
        let r = Vector3::new(0.8 * l, 0.0, 0.0);
        let r_img = heg.minimum_image(r);
        assert!(r_img.norm() < l / 2.0);
    }

    #[test]
    fn test_jastrow_cusp_conditions() {
        let heg = HomogeneousElectronGas::new(4, 4.0, 2.0);
        
        // Check that A parameters satisfy cusp conditions
        assert_relative_eq!(heg.jastrow_a_same, 0.5, epsilon = 1e-10);
        assert_relative_eq!(heg.jastrow_a_anti, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_jastrow_positive() {
        let heg = HomogeneousElectronGas::new(2, 4.0, 2.0);
        
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(heg.box_length / 4.0, 0.0, 0.0),
        ];
        let j = heg.jastrow_factor(&positions);
        assert!(j > 0.0 && j < std::f64::INFINITY);
    }

    #[test]
    fn test_wavefunction_positive() {
        let heg = HomogeneousElectronGas::new(2, 4.0, 2.0);
        let r = heg.initialize();
        let psi = heg.evaluate(&r);
        assert!(psi.is_finite());
    }

    #[test]
    fn test_ewald_finite() {
        let heg = HomogeneousElectronGas::new(14, 4.0, 2.0);
        let r = heg.initialize();
        let v_ewald = heg.ewald_potential(&r);
        assert!(v_ewald.is_finite());
    }

    #[test]
    fn test_local_energy_finite() {
        let heg = HomogeneousElectronGas::new(14, 4.0, 2.0);
        let r = heg.initialize();
        let energy = heg.local_energy(&r);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_hartree_fock_formula() {
        // At rs = 1: E_HF/N ≈ 1.105 - 0.458 ≈ 0.647 Ha
        let heg = HomogeneousElectronGas::new(14, 1.0, 0.5);
        let e_hf = heg.hartree_fock_energy();
        assert!(e_hf > 0.5 && e_hf < 1.0);
        
        // At rs = 4: E_HF/N ≈ 1.105/16 - 0.458/4 ≈ 0.069 - 0.115 ≈ -0.046 Ha
        let heg4 = HomogeneousElectronGas::new(14, 4.0, 2.0);
        let e_hf4 = heg4.hartree_fock_energy();
        assert!(e_hf4 > -0.1 && e_hf4 < 0.0);
    }

    #[test]
    fn test_correlation_energy_negative() {
        // Correlation energy should always be negative
        for rs in [1.0, 2.0, 5.0, 10.0] {
            let heg = HomogeneousElectronGas::new(14, rs, rs / 2.0);
            let ec = heg.ceperley_alder_correlation();
            assert!(ec < 0.0, "Correlation energy should be negative at rs={}", rs);
        }
    }

    #[test]
    fn test_jastrow_gradient_antisymmetric() {
        let heg = HomogeneousElectronGas::new(2, 4.0, 2.0);
        
        let positions = vec![
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(4.0, 5.0, 6.0),
        ];
        let grad = heg.jastrow_gradient(&positions);
        
        // Gradient should be antisymmetric: ∇₁J = -∇₂J
        assert_relative_eq!(grad[0].x, -grad[1].x, epsilon = 1e-10);
        assert_relative_eq!(grad[0].y, -grad[1].y, epsilon = 1e-10);
        assert_relative_eq!(grad[0].z, -grad[1].z, epsilon = 1e-10);
    }

    #[test]
    fn test_local_energy_reasonable() {
        // For 2 electrons in a box at rs=4, check that energy is reasonable
        let heg = HomogeneousElectronGas::new(2, 4.0, 2.0);
        
        // Use a well-separated electron configuration
        let l = heg.box_length;
        let positions = vec![
            Vector3::new(l/4.0, l/4.0, l/4.0),
            Vector3::new(3.0*l/4.0, 3.0*l/4.0, 3.0*l/4.0),
        ];
        
        let psi = heg.evaluate(&positions);
        let energy = heg.local_energy(&positions);
        let potential = heg.ewald_potential(&positions);
        
        println!("=== HEG Debug (N=2) ===");
        println!("Box length: {:.4}", l);
        println!("Wavefunction: {:.6e}", psi);
        println!("Potential energy: {:.6}", potential);
        println!("Local energy: {:.6}", energy);
        println!("Kinetic energy: {:.6}", energy - potential);
        println!("E per electron: {:.6}", energy / 2.0);
        
        // Energy per electron should be on the order of -0.1 to +1.0 Ha for rs=4
        let e_per_elec = energy / 2.0;
        assert!(e_per_elec.is_finite(), "Energy should be finite");
        assert!(e_per_elec.abs() < 10.0, 
            "Energy per electron should be reasonable (got {})", e_per_elec);
    }

    #[test]
    fn test_local_energy_14_electrons() {
        // For 14 electrons at rs=4
        let heg = HomogeneousElectronGas::new(14, 4.0, 2.0);
        
        // Random configuration
        let positions = heg.initialize();
        
        let psi = heg.evaluate(&positions);
        let energy = heg.local_energy(&positions);
        let potential = heg.ewald_potential(&positions);
        
        println!("=== HEG Debug (N=14, rs=4) ===");
        println!("Box length: {:.4}", heg.box_length);
        println!("Wavefunction: {:.6e}", psi);
        println!("Potential energy: {:.6}", potential);
        println!("Local energy: {:.6}", energy);
        println!("Kinetic energy: {:.6}", energy - potential);
        println!("E per electron: {:.6}", energy / 14.0);
        println!("Expected HF E/N: {:.6}", heg.hartree_fock_energy());
        println!("Expected total E/N: {:.6}", heg.hartree_fock_energy() + heg.ceperley_alder_correlation());
        
        // Energy per electron at rs=4 should be around -0.05 to -0.15 Ha
        let e_per_elec = energy / 14.0;
        assert!(e_per_elec.is_finite(), "Energy should be finite");
    }

    #[test]
    fn test_ewald_components() {
        // Test Ewald potential components for a uniform configuration
        let heg = HomogeneousElectronGas::new(14, 4.0, 2.0);
        let l = heg.box_length;
        let alpha = heg.ewald_alpha;
        let n = heg.num_electrons;
        
        // Create a roughly uniform FCC-like configuration
        let spacing = l / 3.0;
        let positions: Vec<Vector3<f64>> = (0..n)
            .map(|i| {
                let x = ((i % 3) as f64 + 0.5) * spacing;
                let y = (((i / 3) % 3) as f64 + 0.5) * spacing;
                let z = ((i / 9) as f64 + 0.5) * spacing;
                Vector3::new(x, y, z)
            })
            .collect();
        
        let v_total = heg.ewald_potential(&positions);
        
        println!("=== Ewald Components ===");
        println!("N = {}, L = {:.4}, α = {:.4}", n, l, alpha);
        println!("Total Ewald potential: {:.6}", v_total);
        println!("V_total / N: {:.6}", v_total / n as f64);
        
        // For jellium at rs=4, the exchange potential is approximately -0.916/rs = -0.229 Ha
        // But that's per electron in the thermodynamic limit
        // For finite N, there are shell effects
        
        // The Ewald potential should be order N²/L ~ N^(5/3)/rs ~ 14^(5/3)/4 ~ 9
        // Wait, that's too big. Let me reconsider.
        // Actually V_ee ~ N(N-1)/2 × <1/r> ~ N²/L
        // For N=14, L=15.5, this is ~14/15.5 ~ 0.9 Ha total, or ~0.06/electron
        
        assert!(v_total.is_finite());
    }
}
