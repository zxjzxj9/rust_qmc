// this file encodes useful jastrow factors

use nalgebra::Vector3;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::wfn::MultiWfn;

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Jastrow1 {
    pub(crate) F: f64,
}

impl MultiWfn for Jastrow1 {

    /// Initializes the Jastrow 1 wavefunction by sampling two random positions from a normal distribution.
    fn initialize(&mut self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        vec![
            Vector3::<f64>::from_distribution(&dist, &mut rng),
            Vector3::<f64>::from_distribution(&dist, &mut rng),
        ]
    }

    /// Evaluates the Jastrow 1 wavefunction at positions `r`.
    fn evaluate(&mut self, r: &Vec<Vector3<f64>>) -> f64 {
        let r1 = &r[0];
        let r2 = &r[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        (-self.F / (2.0 * (1.0 + r12_norm / self.F))).exp()
    }

    /// Computes the gradient (first derivative) of the Jastrow 1 wavefunction at positions `r`.
    fn derivative(&mut self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        let r1 = &r[0];
        let r2 = &r[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        let psi = self.evaluate(r);
        let denom = 2.0 * (1.0 + r12_norm / self.F).powi(2) * r12_norm;
        let grad_factor = psi / denom; // Removed the negative sign here

        vec![
            grad_factor * r12,
            -grad_factor * r12,
        ]
    }

    /// Computes the Laplacian (second derivative) of the Jastrow 1 wavefunction at positions `r`.
    fn laplacian(&mut self, r: &Vec<Vector3<f64>>) -> Vec<f64> {
        let r1 = &r[0];
        let r2 = &r[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        let factor = 1.0 + r12_norm / self.F;
        let psi = self.evaluate(r);
        let denom = 2.0 * factor.powi(2);
        let grad_square = denom.powi(-2);
        let laplacian_factor = 1.0 / (r12_norm * factor.powi(3));
        let comp = (grad_square + laplacian_factor) * psi ;
        vec![
            comp,
            comp
        ]
    }
}