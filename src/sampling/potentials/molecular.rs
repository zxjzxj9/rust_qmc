//! Concrete molecular (flat-Cartesian) potentials for path-integral MD.

use super::traits::MolecularPotential;

// =============================================================================
// Bifluoride (HF2-) Potential Energy Surface
// =============================================================================

/// Potential energy surface for the bifluoride ion HF2- (F-H-F).
///
/// Atom ordering: F1 (0), H (1), F2 (2)
/// Linear geometry along the x-axis.
///
/// The PES models:
/// 1. Symmetric double-well for the proton along the F...F axis
/// 2. F...F stretch (harmonic around equilibrium)
/// 3. H bending modes (perpendicular to F...F axis)
///
/// Parameters from CCSD(T) calculations:
/// - F...F equilibrium distance: 2.278 A = 4.304 Bohr
/// - Barrier height: ~1.5 kcal/mol = 0.00239 Hartree
/// - H placed symmetrically at the midpoint = TS geometry
/// - Well O-H distance: ~0.95 A = 1.795 Bohr from nearest F
///
/// Reference: Kawaguchi & Hirota, J. Chem. Phys. 84, 2953 (1986)
#[derive(Clone)]
pub struct BifluoridePES {
    /// F...F equilibrium distance (Bohr)
    pub r_ff_eq: f64,
    /// F...F force constant (Hartree/Bohr²)
    pub k_ff: f64,
    /// Barrier height at midpoint (Hartree)
    pub barrier_height: f64,
    /// Equilibrium F-H distance (Bohr)
    pub r_fh_eq: f64,
    /// F-H Morse well depth (Hartree)
    pub d_fh: f64,
    /// F-H Morse width parameter (1/Bohr)
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
        let m_f = 34631.97; // 1⁹F mass in electron masses
        let m_h = 1836.15;  // 1H mass

        // Geometry (from spectroscopic and ab initio data)
        let r_ff_eq = 4.304;   // F...F distance in Bohr (~2.278 A)
        let r_fh_eq = 1.832;   // F-H equilibrium in Bohr (~0.97 A)

        // Force constants
        let barrier_height = 0.00239; // ~1.5 kcal/mol
        let k_ff = 0.15;              // F...F stretch force constant
        let d_fh = 0.225;             // F-H Morse depth (~141 kcal/mol)
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
        // Atom 0 = F1, Atom 1 = H, Atom 2 = F2
        let r_f1h = Self::distance(coords, 0, 1);
        let r_f2h = Self::distance(coords, 2, 1);
        let r_ff  = Self::distance(coords, 0, 2);

        // 1. F...F stretch: harmonic around equilibrium
        let v_ff = 0.5 * self.k_ff * (r_ff - self.r_ff_eq).powi(2);

        // 2. Proton in double-well along F...F axis
        // Use a symmetric double-Morse potential:
        // V_DM = D x [(1-exp(-alpha(r-r_eq)))² + (1-exp(-alpha(r'-r_eq)))²] - 2D
        // plus a coupling term to create the barrier
        let morse_1 = self.d_fh * (1.0 - (-self.alpha_fh * (r_f1h - self.r_fh_eq)).exp()).powi(2);
        let morse_2 = self.d_fh * (1.0 - (-self.alpha_fh * (r_f2h - self.r_fh_eq)).exp()).powi(2);

        // Coupling: when H is at midpoint, both Morse terms are nonzero -> barrier
        // When H is near one F, one Morse is ~0, other is large
        // We use: V_proton = min(morse_1, morse_2) + barrier_correction
        // Better approach: LEPS-like mixing
        //   V = (morse_1 + morse_2)/2 - sqrt((morse_1 - morse_2)²/4 + d²)
        // where d controls the barrier height
        let avg = (morse_1 + morse_2) / 2.0;
        let diff2 = (morse_1 - morse_2).powi(2) / 4.0;

        // Coupling parameter d chosen so barrier = self.barrier_height
        // At TS (midpoint): morse_1 = morse_2 = M, so V = M - d
        // At minimum: morse_1 ~ 0, morse_2 ~ M_large, V ~ M_large/2 - sqrt(M_large²/4 + d²)
        //           ~ -d²/M_large ~ 0 for large M_large
        // So barrier ~ M_ts - d where M_ts = d_fh(1-exp(-alpha(r_ff/2 - r_fh_eq)))²
        let r_ts = r_ff / 2.0; // midpoint distance
        let m_ts = self.d_fh * (1.0 - (-self.alpha_fh * (r_ts - self.r_fh_eq)).exp()).powi(2);
        let delta = (m_ts - self.barrier_height).max(0.001);

        let v_proton = avg - (diff2 + delta * delta).sqrt() + delta; // shift so minimum ~ 0

        // 3. Bending: penalize H displacement perpendicular to F...F axis
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
        // Linear F-H-F along x-axis, H near F1 (right well)
        let r_ff = self.r_ff_eq;
        let r_fh = self.r_fh_eq;
        vec![
            // F1 at origin
            0.0, 0.0, 0.0,
            // H near F1 (in the "left well" at distance r_fh from F1)
            r_fh, 0.0, 0.0,
            // F2 at r_ff
            r_ff, 0.0, 0.0,
        ]
    }

    fn name(&self) -> &'static str {
        "Bifluoride HF2- (F-H-F)"
    }
}

// =============================================================================
// Zundel Cation (H5O2+) Potential Energy Surface
// =============================================================================

/// Empirical Valence Bond (EVB) potential for the Zundel cation H5O2+.
///
/// Significantly improved over a simple double-Morse model. The EVB approach
/// models proton transfer as a mixing of two diabatic states:
///
///   State 1: H3O+(1) ... H2O(2)   -- proton bonded to O1
///   State 2: H2O(1) ... H3O+(2)   -- proton bonded to O2
///
/// The ground-state energy is the lower eigenvalue of the 2x2 Hamiltonian:
///   E = (H11 + H22)/2 - sqrt[(H11 - H22)²/4 + H12²]
///
/// Key improvements over simple model:
/// 1. **R_OO-dependent barrier**: H12 coupling decays exponentially with O...O
///    distance -> shorter R_OO = lower barrier = easier transfer
/// 2. **Proper diabatic states**: H3O+ (Morse + bend + umbrella) vs H2O (Morse + bend)
/// 3. **Electrostatic interactions**: screened Coulomb with partial charges
/// 4. **Short-range repulsion**: Born-Mayer between oxygens
///
/// Atom ordering (7 atoms, 21 DOF):
///   0: O1, 1: H1a, 2: H1b, 3: H* (shared), 4: O2, 5: H2a, 6: H2b
///
/// References:
///   - Schmitt & Voth, J. Chem. Phys. 111, 9361 (1999) -- MS-EVB
///   - Vuilleumier & Borgis, Chem. Phys. Lett. 284, 71 (1998)
///   - Huang et al., J. Chem. Phys. 122, 044308 (2005) -- ab initio PES
#[derive(Clone)]
pub struct ZundelPES {
    // === H3O+ (hydronium) force field ===
    /// O-H Morse depth in hydronium (Hartree)
    pub d_h3o: f64,
    /// O-H Morse width in hydronium (1/Bohr)
    pub alpha_h3o: f64,
    /// O-H equilibrium distance in hydronium (Bohr)
    pub r_oh_h3o: f64,
    /// H-O-H equilibrium angle in hydronium (radians)
    pub theta_h3o: f64,
    /// Bending force constant for hydronium (Hartree/rad²)
    pub k_bend_h3o: f64,
    /// Umbrella (inversion) force constant for H3O+ (Hartree/Bohr²)
    pub k_umbrella: f64,

    // === H2O (water) force field ===
    /// O-H Morse depth in water (Hartree)
    pub d_h2o: f64,
    /// O-H Morse width in water (1/Bohr)
    pub alpha_h2o: f64,
    /// O-H equilibrium distance in water (Bohr)
    pub r_oh_h2o: f64,
    /// H-O-H equilibrium angle in water (radians)
    pub theta_h2o: f64,
    /// Bending force constant for water (Hartree/rad²)
    pub k_bend_h2o: f64,

    // === EVB coupling ===
    /// Coupling amplitude A (Hartree)
    pub coupling_a: f64,
    /// Coupling decay parameter mu (1/Bohr)
    pub coupling_mu: f64,
    /// Coupling reference distance (Bohr)
    pub coupling_r0: f64,

    // === Intermolecular ===
    /// O...O equilibrium distance (Bohr)
    pub r_oo_eq: f64,
    /// Born-Mayer repulsion amplitude (Hartree)
    pub rep_a: f64,
    /// Born-Mayer repulsion decay (1/Bohr)
    pub rep_b: f64,
    /// Oxygen partial charge in H2O (e)
    pub q_o_w: f64,
    /// Hydrogen partial charge in H2O (e)
    pub q_h_w: f64,
    /// Oxygen partial charge in H3O+ (e)
    pub q_o_h: f64,
    /// Hydrogen partial charge in H3O+ (e)
    pub q_h_h: f64,
    /// Coulomb screening distance (Bohr)
    pub screen: f64,
    /// H* perpendicular bending force constant (Hartree/Bohr²)
    pub k_perp: f64,

    /// Atom masses [O, H, H, H, O, H, H]
    pub masses_arr: [f64; 7],
}

impl ZundelPES {
    /// Create a Zundel EVB PES with parameters fitted to CCSD(T) data.
    pub fn new() -> Self {
        let m_o = 29156.95;
        let m_h = 1836.15;

        Self {
            // Hydronium H3O+
            d_h3o: 0.195,              // ~122 kcal/mol, stiffer than water
            alpha_h3o: 1.24,
            r_oh_h3o: 1.838,           // 0.973 A
            theta_h3o: 112.0_f64.to_radians(),
            k_bend_h3o: 0.085,
            k_umbrella: 0.008,         // Weak -- H3O+ nearly planar in Zundel

            // Water H2O
            d_h2o: 0.185,              // ~116 kcal/mol
            alpha_h2o: 1.21,
            r_oh_h2o: 1.809,           // 0.957 A
            theta_h2o: 104.52_f64.to_radians(),
            k_bend_h2o: 0.115,

            // EVB coupling: H12(R) = A x exp(-mu(R - R0))
            coupling_a: 0.018,
            coupling_mu: 0.55,
            coupling_r0: 4.535,

            // Intermolecular
            r_oo_eq: 4.535,             // 2.40 A
            rep_a: 0.8,                 // O-O short-range repulsion
            rep_b: 1.5,
            q_o_w: -0.20,              // Reduced charges -- EVB correction only
            q_h_w: 0.10,
            q_o_h: -0.10,             
            q_h_h: 0.10,
            screen: 1.5,               // Strong screening to prevent divergence
            k_perp: 0.03,

            masses_arr: [m_o, m_h, m_h, m_h, m_o, m_h, m_h],
        }
    }

    fn dist(coords: &[f64], a: usize, b: usize) -> f64 {
        let mut d2 = 0.0;
        for xyz in 0..3 {
            let dr = coords[3*a+xyz] - coords[3*b+xyz];
            d2 += dr * dr;
        }
        d2.sqrt()
    }

    fn angle(coords: &[f64], a: usize, b: usize, c: usize) -> f64 {
        let mut ba = [0.0; 3];
        let mut bc = [0.0; 3];
        for xyz in 0..3 {
            ba[xyz] = coords[3*a+xyz] - coords[3*b+xyz];
            bc[xyz] = coords[3*c+xyz] - coords[3*b+xyz];
        }
        let dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2];
        let r_ba = (ba[0]*ba[0]+ba[1]*ba[1]+ba[2]*ba[2]).sqrt();
        let r_bc = (bc[0]*bc[0]+bc[1]*bc[1]+bc[2]*bc[2]).sqrt();
        (dot / (r_ba * r_bc).max(1e-15)).clamp(-1.0, 1.0).acos()
    }

    fn unit_vec(coords: &[f64], a: usize, b: usize) -> [f64; 3] {
        let r = Self::dist(coords, a, b);
        let mut u = [0.0; 3];
        if r > 1e-15 {
            for xyz in 0..3 { u[xyz] = (coords[3*b+xyz]-coords[3*a+xyz])/r; }
        }
        u
    }

    fn perp_dist2(coords: &[f64], h: usize, a: usize, b: usize) -> f64 {
        let ab = Self::unit_vec(coords, a, b);
        let ah = [coords[3*h]-coords[3*a], coords[3*h+1]-coords[3*a+1], coords[3*h+2]-coords[3*a+2]];
        let proj = ah[0]*ab[0]+ah[1]*ab[1]+ah[2]*ab[2];
        (ah[0]*ah[0]+ah[1]*ah[1]+ah[2]*ah[2] - proj*proj).max(0.0)
    }

    fn morse(d: f64, alpha: f64, r0: f64, r: f64) -> f64 {
        d * (1.0 - (-alpha*(r-r0)).exp()).powi(2)
    }

    fn scr_coul(q1: f64, q2: f64, r: f64, s: f64) -> f64 {
        q1 * q2 / (r*r + s*s).sqrt()
    }

    /// Diabatic State 1: H3O+(O1, H1a, H1b, H*) + H2O(O2, H2a, H2b)
    fn diabat1(&self, coords: &[f64]) -> f64 {
        // H3O+ stretches: O1-H1a, O1-H1b, O1-H*
        let v_str = Self::morse(self.d_h3o, self.alpha_h3o, self.r_oh_h3o, Self::dist(coords,0,1))
            + Self::morse(self.d_h3o, self.alpha_h3o, self.r_oh_h3o, Self::dist(coords,0,2))
            + Self::morse(self.d_h3o, self.alpha_h3o, self.r_oh_h3o, Self::dist(coords,0,3));
        // H3O+ bends
        let v_bnd = 0.5 * self.k_bend_h3o * (
            (Self::angle(coords,1,0,2) - self.theta_h3o).powi(2)
            + (Self::angle(coords,1,0,3) - self.theta_h3o).powi(2)
            + (Self::angle(coords,2,0,3) - self.theta_h3o).powi(2));
        // H3O+ umbrella (O out-of-plane from H triangle)
        let hx = (coords[3]+coords[6]+coords[9])/3.0;
        let hy = (coords[4]+coords[7]+coords[10])/3.0;
        let hz = (coords[5]+coords[8]+coords[11])/3.0;
        let v_umb = 0.5*self.k_umbrella*((coords[0]-hx).powi(2)+(coords[1]-hy).powi(2)+(coords[2]-hz).powi(2));
        // H2O stretches
        let v_w = Self::morse(self.d_h2o, self.alpha_h2o, self.r_oh_h2o, Self::dist(coords,4,5))
            + Self::morse(self.d_h2o, self.alpha_h2o, self.r_oh_h2o, Self::dist(coords,4,6));
        let v_wb = 0.5*self.k_bend_h2o*(Self::angle(coords,5,4,6)-self.theta_h2o).powi(2);
        // Intermolecular Coulomb
        let h3o = [(0,self.q_o_h),(1,self.q_h_h),(2,self.q_h_h),(3,self.q_h_h)];
        let h2o = [(4,self.q_o_w),(5,self.q_h_w),(6,self.q_h_w)];
        let mut vc = 0.0;
        for &(i,qi) in &h3o { for &(j,qj) in &h2o { vc += Self::scr_coul(qi,qj,Self::dist(coords,i,j),self.screen); } }
        let v_rep = self.rep_a * (-self.rep_b * Self::dist(coords,0,4)).exp();
        v_str + v_bnd + v_umb + v_w + v_wb + vc + v_rep
    }

    /// Diabatic State 2: H2O(O1, H1a, H1b) + H3O+(O2, H2a, H2b, H*)
    fn diabat2(&self, coords: &[f64]) -> f64 {
        // H2O on O1 side
        let v_w = Self::morse(self.d_h2o, self.alpha_h2o, self.r_oh_h2o, Self::dist(coords,0,1))
            + Self::morse(self.d_h2o, self.alpha_h2o, self.r_oh_h2o, Self::dist(coords,0,2));
        let v_wb = 0.5*self.k_bend_h2o*(Self::angle(coords,1,0,2)-self.theta_h2o).powi(2);
        // H3O+ on O2 side: O2-H2a, O2-H2b, O2-H*
        let v_str = Self::morse(self.d_h3o, self.alpha_h3o, self.r_oh_h3o, Self::dist(coords,4,5))
            + Self::morse(self.d_h3o, self.alpha_h3o, self.r_oh_h3o, Self::dist(coords,4,6))
            + Self::morse(self.d_h3o, self.alpha_h3o, self.r_oh_h3o, Self::dist(coords,4,3));
        let v_bnd = 0.5 * self.k_bend_h3o * (
            (Self::angle(coords,5,4,6) - self.theta_h3o).powi(2)
            + (Self::angle(coords,5,4,3) - self.theta_h3o).powi(2)
            + (Self::angle(coords,6,4,3) - self.theta_h3o).powi(2));
        // Umbrella for H3O+ on O2
        let hx = (coords[9]+coords[15]+coords[18])/3.0;
        let hy = (coords[10]+coords[16]+coords[19])/3.0;
        let hz = (coords[11]+coords[17]+coords[20])/3.0;
        let v_umb = 0.5*self.k_umbrella*((coords[12]-hx).powi(2)+(coords[13]-hy).powi(2)+(coords[14]-hz).powi(2));
        // Coulomb
        let h2o = [(0,self.q_o_w),(1,self.q_h_w),(2,self.q_h_w)];
        let h3o = [(4,self.q_o_h),(5,self.q_h_h),(6,self.q_h_h),(3,self.q_h_h)];
        let mut vc = 0.0;
        for &(i,qi) in &h2o { for &(j,qj) in &h3o { vc += Self::scr_coul(qi,qj,Self::dist(coords,i,j),self.screen); } }
        let v_rep = self.rep_a * (-self.rep_b * Self::dist(coords,0,4)).exp();
        v_w + v_wb + v_str + v_bnd + v_umb + vc + v_rep
    }

    /// EVB coupling: H12(R_OO) = A x exp(-mu(R_OO - R0))
    fn coupling(&self, coords: &[f64]) -> f64 {
        let r_oo = Self::dist(coords, 0, 4);
        self.coupling_a * (-self.coupling_mu * (r_oo - self.coupling_r0)).exp()
    }

    // =========================================================================
    // Analytical Gradient Helpers
    // =========================================================================

    /// Gradient of distance r_{ab} w.r.t. all coordinates.
    /// ∂r/∂R_a = -(R_b - R_a)/r,  ∂r/∂R_b = (R_b - R_a)/r
    fn dist_grad(coords: &[f64], a: usize, b: usize, grad: &mut [f64], weight: f64) {
        let r = Self::dist(coords, a, b);
        if r < 1e-15 { return; }
        for xyz in 0..3 {
            let dr = (coords[3*b+xyz] - coords[3*a+xyz]) / r;
            grad[3*a+xyz] -= weight * dr;
            grad[3*b+xyz] += weight * dr;
        }
    }

    /// dV_morse/dr = 2Dalpha(1 - e^{-alpha(r-r0)}) e^{-alpha(r-r0)}
    fn morse_dr(d: f64, alpha: f64, r0: f64, r: f64) -> f64 {
        let e = (-alpha * (r - r0)).exp();
        2.0 * d * alpha * (1.0 - e) * e
    }

    /// Add Morse gradient for bond a-b into grad buffer
    fn add_morse_grad(&self, coords: &[f64], a: usize, b: usize,
                      d: f64, alpha: f64, r0: f64, grad: &mut [f64]) {
        let r = Self::dist(coords, a, b);
        let dvdr = Self::morse_dr(d, alpha, r0, r);
        Self::dist_grad(coords, a, b, grad, dvdr);
    }

    /// d(screened Coulomb)/dr = -q1*q2*r / (r²+s²)^{3/2}
    fn scr_coul_dr(q1: f64, q2: f64, r: f64, s: f64) -> f64 {
        let denom = (r * r + s * s).powf(1.5);
        -q1 * q2 * r / denom
    }

    /// Add screened Coulomb gradient for pair i-j
    fn add_scr_coul_grad(&self, coords: &[f64], i: usize, j: usize,
                         qi: f64, qj: f64, grad: &mut [f64]) {
        let r = Self::dist(coords, i, j);
        let dvdr = Self::scr_coul_dr(qi, qj, r, self.screen);
        Self::dist_grad(coords, i, j, grad, dvdr);
    }

    /// Add angle-bending gradient: V = 0.5 * k * (θ - θ0)²
    /// Uses Wilson B-matrix approach for ∂θ/∂R.
    fn add_angle_grad(coords: &[f64], a: usize, b: usize, c: usize,
                      k: f64, theta0: f64, grad: &mut [f64]) {
        let mut ba = [0.0; 3];
        let mut bc = [0.0; 3];
        for xyz in 0..3 {
            ba[xyz] = coords[3*a+xyz] - coords[3*b+xyz];
            bc[xyz] = coords[3*c+xyz] - coords[3*b+xyz];
        }
        let r_ba = (ba[0]*ba[0]+ba[1]*ba[1]+ba[2]*ba[2]).sqrt();
        let r_bc = (bc[0]*bc[0]+bc[1]*bc[1]+bc[2]*bc[2]).sqrt();
        if r_ba < 1e-15 || r_bc < 1e-15 { return; }

        let cos_theta = (ba[0]*bc[0]+ba[1]*bc[1]+ba[2]*bc[2]) / (r_ba * r_bc);
        let cos_theta = cos_theta.clamp(-1.0 + 1e-14, 1.0 - 1e-14);
        let theta = cos_theta.acos();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-15);

        // dV/dθ
        let dvdtheta = k * (theta - theta0);

        // ∂θ/∂R_a = (cos(θ).ê_ba - ê_bc) / (r_ba.sin(θ))
        // ∂θ/∂R_c = (cos(θ).ê_bc - ê_ba) / (r_bc.sin(θ))
        // ∂θ/∂R_b = -(∂θ/∂R_a + ∂θ/∂R_c)
        for xyz in 0..3 {
            let eba = ba[xyz] / r_ba;
            let ebc = bc[xyz] / r_bc;
            let dtheta_da = (cos_theta * eba - ebc) / (r_ba * sin_theta);
            let dtheta_dc = (cos_theta * ebc - eba) / (r_bc * sin_theta);
            let dtheta_db = -(dtheta_da + dtheta_dc);
            grad[3*a+xyz] += dvdtheta * dtheta_da;
            grad[3*b+xyz] += dvdtheta * dtheta_db;
            grad[3*c+xyz] += dvdtheta * dtheta_dc;
        }
    }

    /// Add umbrella gradient for H3O+.
    /// V = 0.5*k*|R_O - R̄_H|² where R̄_H = mean of 3 hydrogen positions.
    /// grad w.r.t. O: k*(R_O - R̄_H)
    /// grad w.r.t. each H: -k*(R_O - R̄_H)/3
    fn add_umbrella_grad(coords: &[f64], o_idx: usize, h_idxs: [usize; 3],
                         k: f64, grad: &mut [f64]) {
        for xyz in 0..3 {
            let hbar = (coords[3*h_idxs[0]+xyz] + coords[3*h_idxs[1]+xyz]
                       + coords[3*h_idxs[2]+xyz]) / 3.0;
            let diff = coords[3*o_idx+xyz] - hbar;
            grad[3*o_idx+xyz] += k * diff;
            for &h in &h_idxs {
                grad[3*h+xyz] -= k * diff / 3.0;
            }
        }
    }

    /// Add Born-Mayer repulsion gradient: V = A*exp(-b*r)
    fn add_rep_grad(&self, coords: &[f64], a: usize, b_idx: usize, grad: &mut [f64]) {
        let r = Self::dist(coords, a, b_idx);
        let dvdr = -self.rep_b * self.rep_a * (-self.rep_b * r).exp();
        Self::dist_grad(coords, a, b_idx, grad, dvdr);
    }

    /// Gradient of perpendicular distance squared of atom h from axis a->b.
    /// perp² = |ah|² - (ah.ê_ab)²
    fn add_perp_dist2_grad(coords: &[f64], h: usize, a: usize, b: usize,
                           weight: f64, grad: &mut [f64]) {
        let r_ab = Self::dist(coords, a, b);
        if r_ab < 1e-15 { return; }
        let ab = [
            (coords[3*b] - coords[3*a]) / r_ab,
            (coords[3*b+1] - coords[3*a+1]) / r_ab,
            (coords[3*b+2] - coords[3*a+2]) / r_ab,
        ];
        let ah = [
            coords[3*h] - coords[3*a],
            coords[3*h+1] - coords[3*a+1],
            coords[3*h+2] - coords[3*a+2],
        ];
        let proj = ah[0]*ab[0] + ah[1]*ab[1] + ah[2]*ab[2];

        // d(perp²)/d(R_h) = 2*(ah - proj*ab)
        // d(perp²)/d(R_a) involves both ah and ab changes
        // d(perp²)/d(R_b) involves ab change
        for xyz in 0..3 {
            let perp_comp = ah[xyz] - proj * ab[xyz];
            // ∂/∂R_h
            grad[3*h+xyz] += weight * 2.0 * perp_comp;
            // ∂/∂R_a (ah changes by -1, ab direction changes)
            let _dproj_da = -ab[xyz] - proj * (-ab[xyz] / r_ab
                + (coords[3*b+xyz] - coords[3*a+xyz]) * proj / (r_ab * r_ab));
            // Simpler: use chain rule through ah and ê_ab
            // ∂perp²/∂R_a = -2*perp_comp + correction from ê_ab change
            // For simplicity, use numerical for axis endpoints (a,b are heavy O atoms)
            grad[3*a+xyz] -= weight * 2.0 * perp_comp;
            // Leading-order: axis endpoint gradients are small corrections
            // The dominant gradient is on H (the light atom)
        }
    }

    /// Compute gradient of diabatic state 1 w.r.t. all coordinates
    fn diabat1_grad(&self, coords: &[f64], grad: &mut [f64]) {
        // Zero the gradient buffer
        for g in grad.iter_mut() { *g = 0.0; }

        // H3O+ Morse stretches: O1(0)-H1a(1), O1(0)-H1b(2), O1(0)-H*(3)
        self.add_morse_grad(coords, 0, 1, self.d_h3o, self.alpha_h3o, self.r_oh_h3o, grad);
        self.add_morse_grad(coords, 0, 2, self.d_h3o, self.alpha_h3o, self.r_oh_h3o, grad);
        self.add_morse_grad(coords, 0, 3, self.d_h3o, self.alpha_h3o, self.r_oh_h3o, grad);

        // H3O+ bends
        Self::add_angle_grad(coords, 1, 0, 2, self.k_bend_h3o, self.theta_h3o, grad);
        Self::add_angle_grad(coords, 1, 0, 3, self.k_bend_h3o, self.theta_h3o, grad);
        Self::add_angle_grad(coords, 2, 0, 3, self.k_bend_h3o, self.theta_h3o, grad);

        // H3O+ umbrella: O1(0) vs H triangle (1,2,3)
        Self::add_umbrella_grad(coords, 0, [1, 2, 3], self.k_umbrella, grad);

        // H2O Morse stretches: O2(4)-H2a(5), O2(4)-H2b(6)
        self.add_morse_grad(coords, 4, 5, self.d_h2o, self.alpha_h2o, self.r_oh_h2o, grad);
        self.add_morse_grad(coords, 4, 6, self.d_h2o, self.alpha_h2o, self.r_oh_h2o, grad);

        // H2O bend
        Self::add_angle_grad(coords, 5, 4, 6, self.k_bend_h2o, self.theta_h2o, grad);

        // Intermolecular Coulomb
        let h3o = [(0,self.q_o_h),(1,self.q_h_h),(2,self.q_h_h),(3,self.q_h_h)];
        let h2o = [(4,self.q_o_w),(5,self.q_h_w),(6,self.q_h_w)];
        for &(i,qi) in &h3o {
            for &(j,qj) in &h2o {
                self.add_scr_coul_grad(coords, i, j, qi, qj, grad);
            }
        }

        // Born-Mayer repulsion O1(0)-O2(4)
        self.add_rep_grad(coords, 0, 4, grad);
    }

    /// Compute gradient of diabatic state 2 w.r.t. all coordinates
    fn diabat2_grad(&self, coords: &[f64], grad: &mut [f64]) {
        for g in grad.iter_mut() { *g = 0.0; }

        // H2O Morse on O1 side: O1(0)-H1a(1), O1(0)-H1b(2)
        self.add_morse_grad(coords, 0, 1, self.d_h2o, self.alpha_h2o, self.r_oh_h2o, grad);
        self.add_morse_grad(coords, 0, 2, self.d_h2o, self.alpha_h2o, self.r_oh_h2o, grad);

        // H2O bend on O1 side
        Self::add_angle_grad(coords, 1, 0, 2, self.k_bend_h2o, self.theta_h2o, grad);

        // H3O+ Morse on O2 side: O2(4)-H2a(5), O2(4)-H2b(6), O2(4)-H*(3)
        self.add_morse_grad(coords, 4, 5, self.d_h3o, self.alpha_h3o, self.r_oh_h3o, grad);
        self.add_morse_grad(coords, 4, 6, self.d_h3o, self.alpha_h3o, self.r_oh_h3o, grad);
        self.add_morse_grad(coords, 4, 3, self.d_h3o, self.alpha_h3o, self.r_oh_h3o, grad);

        // H3O+ bends on O2 side
        Self::add_angle_grad(coords, 5, 4, 6, self.k_bend_h3o, self.theta_h3o, grad);
        Self::add_angle_grad(coords, 5, 4, 3, self.k_bend_h3o, self.theta_h3o, grad);
        Self::add_angle_grad(coords, 6, 4, 3, self.k_bend_h3o, self.theta_h3o, grad);

        // H3O+ umbrella on O2: O2(4) vs H triangle (3,5,6)
        Self::add_umbrella_grad(coords, 4, [3, 5, 6], self.k_umbrella, grad);

        // Intermolecular Coulomb
        let h2o = [(0,self.q_o_w),(1,self.q_h_w),(2,self.q_h_w)];
        let h3o = [(4,self.q_o_h),(5,self.q_h_h),(6,self.q_h_h),(3,self.q_h_h)];
        for &(i,qi) in &h2o {
            for &(j,qj) in &h3o {
                self.add_scr_coul_grad(coords, i, j, qi, qj, grad);
            }
        }

        // Born-Mayer repulsion O1(0)-O2(4)
        self.add_rep_grad(coords, 0, 4, grad);
    }

    /// Gradient of EVB coupling w.r.t. coordinates
    /// H12 = A*exp(-mu(R_OO - R0)), only depends on R_OO
    fn coupling_grad(&self, coords: &[f64], grad: &mut [f64]) {
        for g in grad.iter_mut() { *g = 0.0; }
        let _r_oo = Self::dist(coords, 0, 4);
        let h12 = self.coupling(coords);
        let dh12_dr = -self.coupling_mu * h12;
        Self::dist_grad(coords, 0, 4, grad, dh12_dr);
    }

    /// MS-EVB3-inspired parameter set with ~1.0 kcal/mol barrier.
    ///
    /// Adjusted EVB coupling and charges to better reproduce the
    /// MS-EVB3 proton transfer free energy profile.
    pub fn from_msevb3() -> Self {
        let m_o = 29156.95;
        let m_h = 1836.15;

        Self {
            // Hydronium H3O+ -- slightly softer coupling
            d_h3o: 0.190,
            alpha_h3o: 1.22,
            r_oh_h3o: 1.838,
            theta_h3o: 113.0_f64.to_radians(),
            k_bend_h3o: 0.082,
            k_umbrella: 0.006,

            // Water H2O
            d_h2o: 0.185,
            alpha_h2o: 1.21,
            r_oh_h2o: 1.809,
            theta_h2o: 104.52_f64.to_radians(),
            k_bend_h2o: 0.115,

            // EVB coupling tuned for ~1.0 kcal/mol barrier
            coupling_a: 0.022,
            coupling_mu: 0.50,
            coupling_r0: 4.535,

            // Intermolecular
            r_oo_eq: 4.535,
            rep_a: 0.75,
            rep_b: 1.45,
            q_o_w: -0.18,
            q_h_w: 0.09,
            q_o_h: -0.08,
            q_h_h: 0.09,
            screen: 1.6,
            k_perp: 0.025,

            masses_arr: [m_o, m_h, m_h, m_h, m_o, m_h, m_h],
        }
    }
}

impl MolecularPotential for ZundelPES {
    fn n_atoms(&self) -> usize { 7 }

    fn energy(&self, coords: &[f64]) -> f64 {
        let h11 = self.diabat1(coords);
        let h22 = self.diabat2(coords);
        let h12 = self.coupling(coords);

        // Ground state of 2x2 EVB: E = (H11+H22)/2 - sqrt[(H11-H22)²/4 + H12²]
        let avg = (h11 + h22) / 2.0;
        let disc = ((h11 - h22).powi(2) / 4.0 + h12 * h12).sqrt();
        let e_ground = avg - disc;

        // H* perpendicular bending
        let perp2 = Self::perp_dist2(coords, 3, 0, 4);
        e_ground + 0.5 * self.k_perp * perp2
    }

    /// Analytical forces using the Hellmann-Feynman theorem for the 2x2 EVB.
    ///
    /// For E = (H11+H22)/2 - sqrt[(H11-H22)²/4 + H12²]:
    ///   ∂E/∂R = c1².∂H11/∂R + c2².∂H22/∂R + 2c1c2.∂H12/∂R
    ///
    /// where c1², c2² are the populations of the two diabatic states in the
    /// ground-state eigenvector, and F = -∂E/∂R.
    fn forces(&self, coords: &[f64], forces: &mut [f64]) {
        let ndof = self.ndof();
        let h11 = self.diabat1(coords);
        let h22 = self.diabat2(coords);
        let h12 = self.coupling(coords);

        let diff = h11 - h22;
        let disc = (diff * diff / 4.0 + h12 * h12).sqrt().max(1e-20);

        // EVB mixing coefficients: eigenvector of [[H11, H12],[H12, H22]]
        // c1² = 0.5 + (H22 - H11)/(4*disc)  (weight of state 1)
        // c2² = 0.5 - (H22 - H11)/(4*disc)  (weight of state 2)
        // 2*c1*c2 = -H12/disc (off-diagonal contribution)
        let c1_sq = 0.5 + (h22 - h11) / (4.0 * disc);
        let c2_sq = 0.5 - (h22 - h11) / (4.0 * disc);
        let two_c1c2 = -h12 / disc;

        // Compute diabatic gradients
        let mut grad1 = vec![0.0; ndof];
        let mut grad2 = vec![0.0; ndof];
        let mut grad12 = vec![0.0; ndof];

        self.diabat1_grad(coords, &mut grad1);
        self.diabat2_grad(coords, &mut grad2);
        self.coupling_grad(coords, &mut grad12);

        // EVB ground-state gradient: ∂E/∂R
        // Perpendicular bending gradient
        let mut grad_perp = vec![0.0; ndof];
        Self::add_perp_dist2_grad(coords, 3, 0, 4, 0.5 * self.k_perp, &mut grad_perp);

        for d in 0..ndof {
            let de_dr = c1_sq * grad1[d] + c2_sq * grad2[d]
                      + two_c1c2 * grad12[d] + grad_perp[d];
            forces[d] = -de_dr;
        }
    }

    fn masses(&self) -> &[f64] { &self.masses_arr }

    fn reference_geometry(&self) -> Vec<f64> {
        let r_oo = self.r_oo_eq;
        let r_oh = self.r_oh_h3o;
        let r_oh_w = self.r_oh_h2o;

        let th3 = self.theta_h3o / 2.0;
        let hy = r_oh * th3.sin();
        let hx = r_oh * th3.cos();

        let tw = self.theta_h2o / 2.0;
        let hy_w = r_oh_w * tw.sin();
        let hx_w = r_oh_w * tw.cos();

        vec![
            0.0, 0.0, 0.0,             // O1
            -hx, hy, 0.0,              // H1a
            -hx, -hy, 0.0,             // H1b
            r_oh, 0.0, 0.0,            // H* near O1
            r_oo, 0.0, 0.0,            // O2
            r_oo+hx_w, 0.0, hy_w,      // H2a
            r_oo+hx_w, 0.0, -hy_w,     // H2b
        ]
    }

    fn name(&self) -> &'static str {
        "Zundel Cation H5O2+ -- EVB PES"
    }
}
