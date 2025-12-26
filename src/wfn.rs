//! Wave function traits for QMC calculations.
//!
//! Provides `SingleWfn` for single-center wavefunctions (e.g., atomic orbitals)
//! and `MultiWfn` for multi-electron wavefunctions.

use nalgebra::Vector3;

/// Single-center wavefunction trait (e.g., atomic orbitals).
pub trait SingleWfn {
    /// Evaluate the wavefunction at position `r`.
    fn evaluate(&self, r: &Vector3<f64>) -> f64;
    
    /// Compute the gradient at position `r`.
    fn derivative(&self, r: &Vector3<f64>) -> Vector3<f64>;
    
    /// Compute the Laplacian at position `r`.
    fn laplacian(&self, r: &Vector3<f64>) -> f64;

    /// Numerical gradient using central difference.
    fn numerical_derivative(&self, r: &Vector3<f64>, h: f64) -> Vector3<f64> {
        let mut grad = Vector3::zeros();
        for axis in 0..3 {
            let mut r_fwd = *r;
            let mut r_bwd = *r;
            r_fwd[axis] += h;
            r_bwd[axis] -= h;
            grad[axis] = (self.evaluate(&r_fwd) - self.evaluate(&r_bwd)) / (2.0 * h);
        }
        grad
    }

    /// Numerical Laplacian using central difference.
    fn numerical_laplacian(&self, r: &Vector3<f64>, h: f64) -> f64 {
        let psi = self.evaluate(r);
        let mut laplacian = 0.0;
        for axis in 0..3 {
            let mut r_fwd = *r;
            let mut r_bwd = *r;
            r_fwd[axis] += h;
            r_bwd[axis] -= h;
            laplacian += (self.evaluate(&r_fwd) - 2.0 * psi + self.evaluate(&r_bwd)) / (h * h);
        }
        laplacian
    }
}

/// Multi-electron wavefunction trait.
pub trait MultiWfn {
    /// Generate initial random electron positions.
    fn initialize(&self) -> Vec<Vector3<f64>>;
    
    /// Evaluate the wavefunction at positions `r`.
    fn evaluate(&self, r: &[Vector3<f64>]) -> f64;
    
    /// Compute gradients at all positions.
    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>>;
    
    /// Compute Laplacians at all positions.
    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64>;

    /// Numerical gradients using central difference.
    fn numerical_derivative(&self, r: &[Vector3<f64>], h: f64) -> Vec<Vector3<f64>> {
        let mut grad = vec![Vector3::zeros(); r.len()];
        for i in 0..r.len() {
            for axis in 0..3 {
                let mut r_fwd = r.to_vec();
                let mut r_bwd = r.to_vec();
                r_fwd[i][axis] += h;
                r_bwd[i][axis] -= h;
                grad[i][axis] = (self.evaluate(&r_fwd) - self.evaluate(&r_bwd)) / (2.0 * h);
            }
        }
        grad
    }

    /// Numerical Laplacians using central difference.
    fn numerical_laplacian(&self, r: &[Vector3<f64>], h: f64) -> Vec<f64> {
        let psi = self.evaluate(r);
        let mut laplacian = vec![0.0; r.len()];
        for i in 0..r.len() {
            for axis in 0..3 {
                let mut r_fwd = r.to_vec();
                let mut r_bwd = r.to_vec();
                r_fwd[i][axis] += h;
                r_bwd[i][axis] -= h;
                laplacian[i] += (self.evaluate(&r_fwd) - 2.0 * psi + self.evaluate(&r_bwd)) / (h * h);
            }
        }
        laplacian
    }
}