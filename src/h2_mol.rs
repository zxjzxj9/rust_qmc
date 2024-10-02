// simple qmc for h2 molecule

// import vec3 library
extern crate nalgebra as na;
use na::{Vector3, Rotation3};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::mcmc::EnergyCalculator;

// single center wave function
pub trait SingleWfn {
    fn evaluate(&self, r: &Vector3<f64>) -> f64;
    fn derivative(&self, r: &Vector3<f64>) -> Vector3<f64>;
    fn laplacian(&self, r: &Vector3<f64>) -> f64;
}

// multi center wave function, input is a vector of vec3 coords
pub trait MultiWfn {
    fn initialize(&self) -> Vec<Vector3<f64>>;
    fn evaluate(&self, r: &Vec<Vector3<f64>>) -> f64;
    fn derivative(&self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>>;
    fn laplacian(&self, r: &Vec<Vector3<f64>>) -> f64;

    fn numerical_derivative(&self, r: &Vec<Vector3<f64>>, h: f64) -> Vec<Vector3<f64>> {
        let mut grad = vec![Vector3::zeros(); r.len()];

        for (i, _) in r.iter().enumerate() {
            let mut grad_i = Vector3::zeros();

            for axis in 0..3 {
                let mut r_forward = r.clone();
                let mut r_backward = r.clone();

                r_forward[i][axis] += h;
                r_backward[i][axis] -= h;

                let psi_forward = self.evaluate(&r_forward);
                let psi_backward = self.evaluate(&r_backward);

                grad_i[axis] = (psi_forward - psi_backward) / (2.0 * h);
            }

            grad[i] = grad_i;
        }

        grad
    }

    fn numerical_laplacian(&self, r: &Vec<Vector3<f64>>, h: f64) -> f64 {
        let mut laplacian = 0.0;

        for (i, _) in r.iter().enumerate() {
            for axis in 0..3 {
                let mut r_forward = r.clone();
                let mut r_backward = r.clone();

                r_forward[i][axis] += h;
                r_backward[i][axis] -= h;

                let psi_forward = self.evaluate(&r_forward);
                let psi = self.evaluate(r);
                let psi_backward = self.evaluate(&r_backward);

                let second_derivative = (psi_forward - 2.0 * psi + psi_backward) / (h * h);

                laplacian += second_derivative;
            }
        }

        laplacian
    }
}

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
            return 0.0; // Handle singularity at r = R
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
    fn laplacian(&self, r: &Vec<Vector3<f64>>) -> f64 {
        let r1 = &r[0];
        let r2 = &r[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        let psi = self.evaluate(r);
        let denom = 2.0 * (1.0 + r12_norm / self.F).powi(2) * r12_norm;
        let grad_factor = psi / denom;
        let grad_square = 1.0 * (grad_factor * r12_norm).powi(2);
        let laplacian_factor = 1.0 / (r12_norm * (1.0 + r12_norm / self.F).powi(3));

        2.0 * (grad_square + laplacian_factor) * psi
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct H2MoleculeVB {
    pub(crate) H1: Slater1s,
    pub(crate) H2: Slater1s,
    pub(crate) J: Jastrow1,
}

impl MultiWfn for H2MoleculeVB {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        self.J.initialize()
    }
    fn evaluate(&self, r: &Vec<Vector3<f64>>) -> f64 {
        let psi_1 = self.H1.evaluate(&r[0]);
        let psi_2 = self.H2.evaluate(&r[1]);
        let j = self.J.evaluate(r);
        psi_1 * psi_2 * j
    }

    fn derivative(&self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        let psi_1 = self.H1.evaluate(&r[0]);
        let psi_2 = self.H2.evaluate(&r[1]);
        let j = self.J.evaluate(r);

        let grad_1 = self.H1.derivative(&r[0]);
        let grad_2 = self.H2.derivative(&r[1]);
        let grad_j = self.J.derivative(r);

        vec![
            grad_1 * psi_2 * j + psi_1 * psi_2 * grad_j[0],
            psi_1 * grad_2 * j + psi_1 * psi_2 * grad_j[1],
        ]
    }

    fn laplacian(&self, r: &Vec<Vector3<f64>>) -> f64 {
        let psi_1 = self.H1.evaluate(&r[0]);
        let psi_2 = self.H2.evaluate(&r[1]);
        let j = self.J.evaluate(r);

        let grad_1 = self.H1.derivative(&r[0]);
        let grad_2 = self.H2.derivative(&r[1]);
        let grad_j = self.J.derivative(r);

        let lap_1 = self.H1.laplacian(&r[0]);
        let lap_2 = self.H2.laplacian(&r[1]);
        let lap_j = self.J.laplacian(r);

        // Compute ∇²ψ using the product rule
        let term1 = lap_1 * psi_2 * j;
        let term2 = psi_1 * lap_2 * j;
        let term3 = psi_1 * psi_2 * lap_j;
        let term4 = 2.0 * (grad_1.dot(&grad_j[0]) * psi_2 +
            psi_1 * grad_2.dot(&grad_j[1]) +
            grad_1.dot(&grad_2) * j);

        term1 + term2 + term3 + term4
    }
}

impl EnergyCalculator for H2MoleculeVB {
    fn local_energy(&self, positions: &Vec<Vector3<f64>>) -> f64 {
        let r1 = &positions[0];
        let r2 = &positions[1];
        let r12 = r1 - r2;
        let r12_norm = r12.norm();
        let psi = self.evaluate(positions);
        let lap = self.laplacian(positions);

        let kinetic = -0.5 * lap / psi;
        let potential = -1.0 / r12_norm
            - 1.0 / (r1 - self.H1.R).norm() - 1.0 / (r2 - self.H2.R).norm()
            + 1.0 / (self.H1.R - self.H2.R).norm();
        kinetic + potential
    }
}
