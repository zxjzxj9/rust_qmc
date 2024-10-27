use nalgebra::{DMatrix, Matrix, Vector3};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::wfn::{MultiWfn, SingleWfn};

// \psi_n(r) = A \sum_{\nu=1}^{m} \phi_{\nu n} r^{p_{\nu}} \exp\left(-\xi_n r\right)
#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct STO {
    // pub(crate) A: f64, no need for MC run
    pub(crate) m: i32,
    pub(crate) phi: Vec<f64>,
    pub(crate) p: Vec<i32>,
    pub(crate) zeta: Vec<f64>,
    pub(crate) R: Vector3<f64>,
}

// initialize STO for Li, keep l, m if needed in the future
pub(crate) fn init_li_sto(r: Vector3<f64>, n: i32, l: i32, m: i32) -> STO {
    // switch case n = 1, 2
    match n {
        1 => STO {
            m: 7,
            zeta: vec![0.72089388, 2.61691643, 0.69257443, 1.37137558, 3.97864549, 13.52900016, 19.30801440],
            p: vec![1, 1, 2, 2, 2, 2, 3],
            phi: vec![-0.12220686, 1.11273225, 0.04125378, 0.09306499, -0.10260021, -0.00034191, 0.00021963],
            R: r,
        },
        2 => STO {
            m: 7,
            zeta: vec![0.72089388, 2.61691643, 0.69257443, 1.37137558, 3.97864549, 13.52900016, 19.30801440],
            p: vec![1, 1, 2, 2, 2, 2, 3],
            phi: vec![0.47750469, 0.11140449, -1.25954273, -0.18475003, -0.02736293, -0.00025064, 0.00057962],
            R: r,
        },
        _ => STO {
            m: 1,
            phi: vec![0.0],
            p: vec![0],
            zeta: vec![0.0],
            R: r,
        },
    }
}

impl SingleWfn for STO {
    fn evaluate(&mut self, r: &Vector3<f64>) -> f64 {
        let mut psi = 0.0;
        let dist = (r - self.R).norm();
        for nu in 0..self.m {
            psi += self.phi[nu as usize] * dist.powi(self.p[nu as usize]) * (-self.zeta[nu as usize] * dist).exp();
        }
        psi
    }

    fn derivative(&mut self, r: &Vector3<f64>) -> Vector3<f64> {
        let dist = (r - self.R).norm();
        let grad_dir = (r - self.R) / dist; // Unit vector in the radial direction
        let mut derivative = Vector3::zeros();

        for nu in 0..self.m as usize {
            let phi_nu = self.phi[nu];
            let p_nu = self.p[nu] as f64;
            let zeta_nu = self.zeta[nu];
            let s = dist;
            let exp_factor = (-zeta_nu * s).exp();
            let s_p = s.powf(p_nu);
            let s_p_minus1 = s_p / s; // s^{p_nu - 1}

            let radial_derivative = phi_nu * exp_factor * (-zeta_nu * s_p + p_nu * s_p_minus1);
            derivative += grad_dir * radial_derivative;
        }
        derivative
    }

    fn laplacian(&mut self, r: &Vector3<f64>) -> f64 {
        let dist = (r - self.R).norm();
        let s = dist;
        let mut laplacian = 0.0;

        for nu in 0..self.m as usize {
            let phi_nu = self.phi[nu];
            let p_nu = self.p[nu] as f64;
            let zeta_nu = self.zeta[nu];
            let exp_factor = (-zeta_nu * s).exp();
            let s_p = s.powf(p_nu);
            let s_p_minus1 = s_p / s; // s^{p_nu - 1}
            let s_p_minus2 = s_p_minus1 / s; // s^{p_nu - 2}

            let grad2_term = zeta_nu * zeta_nu * s_p
                - 2.0 * zeta_nu * p_nu * s_p_minus1
                + p_nu * (p_nu - 1.0) * s_p_minus2;
            let grad_term = 2.0 * (-zeta_nu * s_p + p_nu * s_p_minus1) / dist;
            laplacian += phi_nu * exp_factor * (grad2_term + grad_term);
        }
        laplacian
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct STOSlaterDet {
    // slater determinant for STO
    pub(crate) n: usize,
    pub(crate) sto: Vec<STO>,
    pub(crate) spin: Vec<i32>,
    pub(crate) s: DMatrix<f64>,
    pub(crate) inv_s: DMatrix<f64>,
}


impl STOSlaterDet {
    pub fn init_wfn(&mut self, r: Vec<Vector3<f64>>) -> &Self {
        self.s = DMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                if self.spin[i] == self.spin[j] {
                    self.s[(i, j)] = self.sto[i].evaluate(&r[j]);
                } else {
                    self.s[(i, j)] = 0.0;
                }
            }
        }
        // calculate inverse of self.s
        self.inv_s = self.s.clone().try_inverse().unwrap();
        self
    }
}

impl MultiWfn for STOSlaterDet {
    // initialize the STO slater determinant
    fn initialize(&mut self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        // initialize random positions, self.n is the number of electrons
        let mut r = vec![];
        for _ in 0..self.n {
            r.push(Vector3::<f64>::from_distribution(&dist, &mut rng));
        }
        r
    }

    fn evaluate(&mut self, r: &Vec<Vector3<f64>>) -> f64 {
        // Update the Slater matrix
        self.s = DMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                if self.spin[i] == self.spin[j] {
                    self.s[(i, j)] = self.sto[i].evaluate(&r[j]);
                } else {
                    self.s[(i, j)] = 0.0;
                }
            }
        }
        // Calculate the determinant
        let psi = self.s.determinant();
        // println!("psi = {}", psi);
        // Update the inverse of the Slater matrix
        self.inv_s = self.s.clone().try_inverse().unwrap();
        psi
    }

    fn derivative(&mut self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        // Compute the derivative of the Slater determinant
        let psi = self.evaluate(r);
        let mut derivative = vec![Vector3::zeros(); r.len()];
        for i in 0..self.n {
            let mut sum = Vector3::zeros();
            for j in 0..self.n {
                if self.spin[i] == self.spin[j] {
                    let grad_phi = self.sto[j].derivative(&r[i]);
                    sum += self.inv_s[(i, j)] * grad_phi;
                }
            }
            derivative[i] = psi * sum;
        }
        derivative
    }

    fn laplacian(&mut self, r: &Vec<Vector3<f64>>) -> Vec<f64> {
        // Compute the Laplacian of the Slater determinant
        let psi = self.evaluate(r);
        let mut laplacian = vec![0.0; r.len()];
        // Precompute gradients and Laplacians of orbitals
        let mut grad_phi = vec![vec![Vector3::zeros(); self.n]; self.n];
        let mut lap_phi = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                if self.spin[i] == self.spin[j] {
                    grad_phi[i][j] = self.sto[j].derivative(&r[i]);
                    lap_phi[i][j] = self.sto[j].laplacian(&r[i]);
                }
            }
        }
        for i in 0..self.n {
            let mut sum_lap = 0.0;
            for j in 0..self.n {
                sum_lap += self.inv_s[(i, j)] * lap_phi[i][j];
            }
            let mut sum_grad_dot = 0.0;
            for j in 0..self.n {
                for k in 0..self.n {
                    sum_grad_dot += self.inv_s[(i, j)] * self.inv_s[(i, k)] * grad_phi[i][j].dot(&grad_phi[i][k]);
                }
            }
            laplacian[i] = psi * (sum_lap + sum_grad_dot);
        }
        laplacian
    }
}