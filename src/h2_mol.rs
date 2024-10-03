// simple qmc for h2 molecule

// import vec3 library
extern crate nalgebra as na;
use na::Vector3;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::mcmc::EnergyCalculator;
use crate::wfn::{MultiWfn, SingleWfn};

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Slater1s {
    pub(crate) alpha: f64,
    pub(crate) R: Vector3<f64>,
}

impl SingleWfn for Slater1s {
    /// Evaluates the Slater 1s wavefunction at position `r`.
    fn evaluate(&self, r: &Vector3<f64>) -> f64 {
        let r_minus_R = r - self.R;
        let r_norm = r_minus_R.norm();
        (-self.alpha * r_norm).exp()
    }

    /// Computes the gradient (first derivative) of the Slater 1s wavefunction at position `r`.
    fn derivative(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let r_minus_R = r - self.R;
        let r_norm = r_minus_R.norm();
        if r_norm == 0.0 {
            return Vector3::new(0.0, 0.0, 0.0); // Avoid division by zero
        }
        let scalar = -self.alpha / r_norm * (-self.alpha * r_norm).exp();
        r_minus_R * scalar
    }

    /// Computes the Laplacian (second derivative) of the Slater 1s wavefunction at position `r`.
    fn laplacian(&self, r: &Vector3<f64>) -> f64 {
        let r_minus_R = r - self.R;
        let r_norm = r_minus_R.norm();
        if r_norm == 0.0 {
            return f64::NEG_INFINITY; // Handle singularity at r = R
        }
        let exp_part = (-self.alpha * r_norm).exp();
        let laplacian = (self.alpha.powi(2) - 2.0 * self.alpha / r_norm) * exp_part;
        laplacian
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Jastrow1 {
    pub(crate) F: f64,
}

impl MultiWfn for Jastrow1 {

    /// Initializes the Jastrow 1 wavefunction by sampling two random positions from a normal distribution.
    fn initialize(&self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        vec![
            Vector3::<f64>::from_distribution(&dist, &mut rng),
            Vector3::<f64>::from_distribution(&dist, &mut rng),
        ]
    }

    /// Evaluates the Jastrow 1 wavefunction at positions `r`.
    fn evaluate(&self, r: &Vec<Vector3<f64>>) -> f64 {
        let r1 = &r[0];
        let r2 = &r[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        (-self.F / (2.0 * (1.0 + r12_norm / self.F))).exp()
    }

    /// Computes the gradient (first derivative) of the Jastrow 1 wavefunction at positions `r`.
    fn derivative(&self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
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
    fn laplacian(&self, r: &Vec<Vector3<f64>>) -> Vec<f64> {
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

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct H2MoleculeVB {
    pub(crate) H1: Slater1s,
    pub(crate) H2: Slater1s,
    pub(crate) J: Jastrow1,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct H2MoleculeMO {
    pub(crate) H1: Slater1s,
    pub(crate) H2: Slater1s,
    pub(crate) J: Jastrow1,
}

impl MultiWfn for H2MoleculeVB {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        self.J.initialize()
    }
    fn evaluate(&self, r: &Vec<Vector3<f64>>) -> f64 {
        let psi_1 = self.H1.evaluate(&r[0]) * self.H2.evaluate(&r[0]);
        let psi_2 = self.H1.evaluate(&r[1]) * self.H2.evaluate(&r[1]);
        let j = self.J.evaluate(r);
        psi_1 * psi_2 * j
    }

    fn derivative(&self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        // Evaluate psi_1 and psi_2
        let psi_1 = self.H1.evaluate(&r[0]) * self.H2.evaluate(&r[0]);
        let psi_2 = self.H1.evaluate(&r[1]) * self.H2.evaluate(&r[1]);
        let j = self.J.evaluate(r);

        // Derivatives of H1 and H2 at r[0]
        let h1_eval_r0 = self.H1.evaluate(&r[0]);
        let h2_eval_r0 = self.H2.evaluate(&r[0]);
        let h1_deriv_r0 = self.H1.derivative(&r[0]);
        let h2_deriv_r0 = self.H2.derivative(&r[0]);
        // Derivative of psi_1 with respect to r[0]
        let psi_1_derivative = h1_deriv_r0 * h2_eval_r0 + h1_eval_r0 * h2_deriv_r0;

        // Derivatives of H1 and H2 at r[1]
        let h1_eval_r1 = self.H1.evaluate(&r[1]);
        let h2_eval_r1 = self.H2.evaluate(&r[1]);
        let h1_deriv_r1 = self.H1.derivative(&r[1]);
        let h2_deriv_r1 = self.H2.derivative(&r[1]);
        // Derivative of psi_2 with respect to r[1]
        let psi_2_derivative = h1_deriv_r1 * h2_eval_r1 + h1_eval_r1 * h2_deriv_r1;

        // Derivative of J with respect to r[0] and r[1]
        let j_derivative = self.J.derivative(r); // Returns Vec<Vector3<f64>>

        // Compute the derivatives with respect to r[0] and r[1]
        let derivative_0 = psi_2 * (psi_1_derivative * j + psi_1 * j_derivative[0]);
        let derivative_1 = psi_1 * (psi_2_derivative * j + psi_2 * j_derivative[1]);

        vec![derivative_0, derivative_1]
    }


    fn laplacian(&self, r: &Vec<Vector3<f64>>) -> Vec<f64> {
        // Evaluate psi_1 and psi_2
        let psi_1 = self.H1.evaluate(&r[0]) * self.H2.evaluate(&r[0]);
        let psi_2 = self.H1.evaluate(&r[1]) * self.H2.evaluate(&r[1]);
        let j = self.J.evaluate(r);

        // Derivatives and Laplacians at r[0]
        let h1_eval_r0 = self.H1.evaluate(&r[0]);
        let h2_eval_r0 = self.H2.evaluate(&r[0]);
        let h1_deriv_r0 = self.H1.derivative(&r[0]);
        let h2_deriv_r0 = self.H2.derivative(&r[0]);
        let h1_laplacian_r0 = self.H1.laplacian(&r[0]);
        let h2_laplacian_r0 = self.H2.laplacian(&r[0]);

        // Derivative and Laplacian of psi_1
        let psi_1_derivative = h1_deriv_r0 * h2_eval_r0 + h1_eval_r0 * h2_deriv_r0;
        let psi_1_laplacian = h1_laplacian_r0 * h2_eval_r0
            + 2.0 * h1_deriv_r0.dot(&h2_deriv_r0)
            + h1_eval_r0 * h2_laplacian_r0;

        // Derivatives and Laplacians at r[1]
        let h1_eval_r1 = self.H1.evaluate(&r[1]);
        let h2_eval_r1 = self.H2.evaluate(&r[1]);
        let h1_deriv_r1 = self.H1.derivative(&r[1]);
        let h2_deriv_r1 = self.H2.derivative(&r[1]);
        let h1_laplacian_r1 = self.H1.laplacian(&r[1]);
        let h2_laplacian_r1 = self.H2.laplacian(&r[1]);

        // Derivative and Laplacian of psi_2
        let psi_2_derivative = h1_deriv_r1 * h2_eval_r1 + h1_eval_r1 * h2_deriv_r1;
        let psi_2_laplacian = h1_laplacian_r1 * h2_eval_r1
            + 2.0 * h1_deriv_r1.dot(&h2_deriv_r1)
            + h1_eval_r1 * h2_laplacian_r1;

        // Derivative and Laplacian of J
        let j_derivative = self.J.derivative(r); // Vec<Vector3<f64>>
        let j_laplacian = self.J.laplacian(r);   // f64

        // Compute Laplacian contributions
        let laplacian_0 = psi_2
            * (psi_1_laplacian * j + 2.0 * psi_1_derivative.dot(&j_derivative[0]) + psi_1 * j_laplacian[0]);
        let laplacian_1 = psi_1
            * (psi_2_laplacian * j + 2.0 * psi_2_derivative.dot(&j_derivative[1]) + psi_2 * j_laplacian[1]);

        vec![laplacian_0, laplacian_1]
    }
}

impl MultiWfn for H2MoleculeMO {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        self.J.initialize()
    }
    fn evaluate(&self, r: &Vec<Vector3<f64>>) -> f64 {
        let psi_1 = self.H1.evaluate(&r[0]) + self.H2.evaluate(&r[0]);
        let psi_2 = self.H1.evaluate(&r[1]) + self.H2.evaluate(&r[1]);
        let j = self.J.evaluate(r);
        psi_1 * psi_2 * j
    }

    fn derivative(&self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        let psi_sum_r1 = self.H1.evaluate(&r[0]) + self.H2.evaluate(&r[0]);
        let psi_sum_r2 = self.H1.evaluate(&r[1]) + self.H2.evaluate(&r[1]);

        let grad_psi_sum_r1 = self.H1.derivative(&r[0]) + self.H2.derivative(&r[0]);
        let grad_psi_sum_r2 = self.H1.derivative(&r[1]) + self.H2.derivative(&r[1]);

        let j = self.J.evaluate(r);
        let grad_j = self.J.derivative(r);

        let grad_psi_r1 = (grad_psi_sum_r1 * psi_sum_r2 * j) + (psi_sum_r1 * psi_sum_r2 * grad_j[0]);
        let grad_psi_r2 = (grad_psi_sum_r2 * psi_sum_r1 * j) + (psi_sum_r1 * psi_sum_r2 * grad_j[1]);

        vec![grad_psi_r1, grad_psi_r2]
    }


    fn laplacian(&self, r: &Vec<Vector3<f64>>) -> Vec<f64> {
        let psi_sum_r1 = self.H1.evaluate(&r[0]) + self.H2.evaluate(&r[0]);
        let psi_sum_r2 = self.H1.evaluate(&r[1]) + self.H2.evaluate(&r[1]);

        let grad_psi_sum_r1 = self.H1.derivative(&r[0]) + self.H2.derivative(&r[0]);
        let grad_psi_sum_r2 = self.H1.derivative(&r[1]) + self.H2.derivative(&r[1]);

        let laplacian_psi_sum_r1 = self.H1.laplacian(&r[0]) + self.H2.laplacian(&r[0]);
        let laplacian_psi_sum_r2 = self.H1.laplacian(&r[1]) + self.H2.laplacian(&r[1]);

        let j = self.J.evaluate(r);
        let grad_j = self.J.derivative(r);
        let j_laplacian = self.J.laplacian(r);

        let laplacian_j_r1 = j_laplacian[0];
        let laplacian_j_r2 = j_laplacian[1];

        let delta0_psi = laplacian_psi_sum_r1 * psi_sum_r2 * j
            + 2.0 * grad_psi_sum_r1.dot(&grad_j[0]) * psi_sum_r2
            + psi_sum_r1 * psi_sum_r2 * laplacian_j_r1;

        let delta1_psi = laplacian_psi_sum_r2 * psi_sum_r1 * j
            + 2.0 * grad_psi_sum_r2.dot(&grad_j[1]) * psi_sum_r1
            + psi_sum_r1 * psi_sum_r2 * laplacian_j_r2;

        // Total Laplacian
        vec![delta0_psi, delta1_psi]
    }
}

impl EnergyCalculator for H2MoleculeVB {
    fn local_energy(&self, positions: &Vec<Vector3<f64>>) -> f64 {
        let r1 = &positions[0];
        let r2 = &positions[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        let psi = self.evaluate(positions);
        let lap: f64 = self.laplacian(positions).into_iter().sum();

        let kinetic = -0.5 * lap / psi;
        let potential = 1.0 / r12_norm
            - 1.0 / (r1 - self.H1.R).norm() - 1.0 / (r2 - self.H2.R).norm()
            - 1.0 / (r1 - self.H2.R).norm() - 1.0 / (r2 - self.H1.R).norm()
            + 1.0 / (self.H1.R - self.H2.R).norm();
        kinetic + potential
    }
}

impl EnergyCalculator for H2MoleculeMO {
    fn local_energy(&self, positions: &Vec<Vector3<f64>>) -> f64 {
        let r1 = &positions[0];
        let r2 = &positions[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        let psi = self.evaluate(positions);
        let lap: f64 = self.laplacian(positions).into_iter().sum();

        let kinetic = -0.5 * lap / psi;
        let potential = 1.0 / r12_norm
            - 1.0 / (r1 - self.H1.R).norm() - 1.0 / (r2 - self.H2.R).norm()
            - 1.0 / (r1 - self.H2.R).norm() - 1.0 / (r2 - self.H1.R).norm()
            + 1.0 / (self.H1.R - self.H2.R).norm();
        kinetic + potential
    }
}
