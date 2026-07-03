//! Concrete 1D scalar potentials for path-integral Monte Carlo.

use super::traits::Potential;

/// Harmonic oscillator potential: V(x) = (1/2)mw²x²
#[derive(Clone)]
pub struct HarmonicPotential {
    pub mass: f64,
    pub omega: f64,
}

impl Potential for HarmonicPotential {
    fn evaluate(&self, x: f64) -> f64 {
        0.5 * self.mass * self.omega * self.omega * x * x
    }

    fn force(&self, x: f64) -> f64 {
        -self.mass * self.omega * self.omega * x
    }

    fn name(&self) -> &'static str {
        "Harmonic Oscillator"
    }

    fn init_width(&self) -> f64 {
        (1.0 / (self.mass * self.omega)).sqrt()
    }
}

/// Sombrero (Mexican Hat) potential: V(x) = -mu²x²/2 + lambdax⁴/4
///
/// This is a double-well potential with minima at x = +/-sqrt(mu²/lambda)
/// The barrier height is V(0) - V(x_min) = mu⁴/(4lambda)
#[derive(Clone)]
pub struct SombreroPotential {
    /// mu² coefficient (controls well depth)
    pub mu_squared: f64,
    /// lambda coefficient (controls quartic term)
    pub lambda: f64,
}

impl SombreroPotential {
    /// Create a sombrero potential with specified barrier height and well positions
    ///
    /// # Arguments
    /// * `well_position` - Location of the minima at x = +/-x_min
    /// * `barrier_height` - Height of the barrier at x=0 above the minima
    pub fn from_geometry(well_position: f64, barrier_height: f64) -> Self {
        // x_min = sqrt(mu²/lambda), so mu²/lambda = x_min²
        // barrier = mu⁴/(4lambda), so mu⁴ = 4lambda x barrier
        // From x_min²: mu² = lambda x x_min², so mu⁴ = lambda² x x_min⁴
        // Therefore: lambda² x x_min⁴ = 4lambda x barrier
        // lambda = 4 x barrier / x_min⁴
        // mu² = lambda x x_min² = 4 x barrier / x_min²
        let lambda = 4.0 * barrier_height / well_position.powi(4);
        let mu_squared = lambda * well_position.powi(2);
        Self { mu_squared, lambda }
    }

    /// Location of the potential minima
    pub fn well_position(&self) -> f64 {
        (self.mu_squared / self.lambda).sqrt()
    }

    /// Height of the barrier at x=0
    pub fn barrier_height(&self) -> f64 {
        self.mu_squared * self.mu_squared / (4.0 * self.lambda)
    }
}

impl Potential for SombreroPotential {
    fn evaluate(&self, x: f64) -> f64 {
        -0.5 * self.mu_squared * x * x + 0.25 * self.lambda * x.powi(4)
    }

    fn force(&self, x: f64) -> f64 {
        // F = -dV/dx = mu²x - lambdax³
        self.mu_squared * x - self.lambda * x.powi(3)
    }

    fn name(&self) -> &'static str {
        "Sombrero (Mexican Hat)"
    }

    fn init_width(&self) -> f64 {
        self.well_position()
    }
}

/// Double-well potential: V(x) = a(x² - b²)²
///
/// Minima at x = +/-b, barrier height = ab⁴
#[derive(Clone)]
pub struct DoubleWellPotential {
    pub a: f64,
    pub b: f64,
}

impl Potential for DoubleWellPotential {
    fn evaluate(&self, x: f64) -> f64 {
        let diff = x * x - self.b * self.b;
        self.a * diff * diff
    }

    fn force(&self, x: f64) -> f64 {
        // V = a(x² - b²)², F = -dV/dx = -4ax(x² - b²)
        -4.0 * self.a * x * (x * x - self.b * self.b)
    }

    fn name(&self) -> &'static str {
        "Double Well"
    }

    fn init_width(&self) -> f64 {
        self.b
    }
}

/// Proton transfer double-well potential for O-H...O hydrogen bond tunneling
///
/// V(x) = V_b x (x²/d² - 1)²
///
/// This models a symmetric double well with:
/// - Minima at x = +/-d (donor and acceptor sites)
/// - Barrier height V_b at x = 0
/// - Typical parameters: d ~ 0.5–1.0 Bohr, V_b ~ 0.01–0.05 Hartree
///
/// The potential is equivalent to a(x² - b²)² with a = V_b/d⁴, b = d,
/// but parameterized in the physically intuitive proton transfer language.
#[derive(Clone)]
pub struct ProtonTransferPotential {
    /// Barrier height at x=0, in Hartree
    pub barrier_height: f64,
    /// Half-distance between wells (+/-d), in Bohr
    pub well_distance: f64,
    /// Optional asymmetry: tilt ε adds a linear term εx to break symmetry
    pub asymmetry: f64,
}

impl ProtonTransferPotential {
    /// Create a symmetric proton transfer potential
    pub fn symmetric(barrier_height: f64, well_distance: f64) -> Self {
        Self { barrier_height, well_distance, asymmetry: 0.0 }
    }

    /// Create an asymmetric proton transfer potential
    /// `asymmetry` > 0 favors the right well (x > 0)
    pub fn asymmetric(barrier_height: f64, well_distance: f64, asymmetry: f64) -> Self {
        Self { barrier_height, well_distance, asymmetry }
    }

    /// Effective frequency at the bottom of a well (harmonic approximation)
    /// w = sqrt(V''(+/-d)/m) where V''(+/-d) = 8 V_b / d²
    pub fn well_frequency(&self, mass: f64) -> f64 {
        (8.0 * self.barrier_height / (self.well_distance * self.well_distance * mass)).sqrt()
    }
}

impl Potential for ProtonTransferPotential {
    fn evaluate(&self, x: f64) -> f64 {
        let d2 = self.well_distance * self.well_distance;
        let ratio = x * x / d2 - 1.0;
        self.barrier_height * ratio * ratio + self.asymmetry * x
    }

    fn force(&self, x: f64) -> f64 {
        let d2 = self.well_distance * self.well_distance;
        // F = -dV/dx = -4 V_b x (x²/d² - 1) / d² - ε
        -4.0 * self.barrier_height * x * (x * x / d2 - 1.0) / d2 - self.asymmetry
    }

    fn name(&self) -> &'static str {
        "Proton Transfer Double-Well"
    }

    fn init_width(&self) -> f64 {
        self.well_distance
    }
}
