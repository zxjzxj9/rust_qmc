//! Potential-energy abstractions shared by all PIMC/PIMD samplers.
//!
//! `Potential` is the 1D scalar form used by path-integral Monte Carlo;
//! `MolecularPotential` is the flat-Cartesian-coordinate form used by molecular
//! path-integral MD. The `Splittable*` variants add fast/slow force splitting
//! for ring-polymer contraction (`pimd_rpc`).

// =============================================================================
// 1D Scalar Potential
// =============================================================================

/// Trait for 1D potentials that can be used with PIMC/PIMD
pub trait Potential: Clone + Send + Sync {
    /// Evaluate the potential V(x) at position x
    fn evaluate(&self, x: f64) -> f64;

    /// Compute the force F(x) = -dV/dx at position x
    /// Default implementation uses numerical central difference
    fn force(&self, x: f64) -> f64 {
        let h = 1e-7;
        -(self.evaluate(x + h) - self.evaluate(x - h)) / (2.0 * h)
    }

    /// Name of the potential for display
    fn name(&self) -> &'static str;

    /// Suggested initialization width for beads
    fn init_width(&self) -> f64;
}

// =============================================================================
// Multi-Dimensional Molecular Potential
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
// Splittable (fast/slow) Potentials for Ring-Polymer Contraction
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
