use nalgebra::{DMatrix, Matrix, Vector3};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use crate::jastrow::Jastrow2;
use crate::mcmc::EnergyCalculator;
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

/// Parameters for the Jastrow factor.
struct JastrowParameters {
    /// Electron-electron interaction parameters.
    a: f64,
    b: f64,
    /// Electron-nucleus interaction parameters.
    c: f64,
    d: f64,
}

/// Represents the Jastrow factor for multi-center electrons.
struct JastrowFactor {
    /// Positions of nuclei in 3D space.
    nucleus_positions: Vec<Vector3<f64>>,
    /// Parameters for the Jastrow factor.
    parameters: JastrowParameters,
}

impl JastrowFactor {
    /// Creates a new `JastrowFactor`.
    ///
    /// # Arguments
    ///
    /// * `nucleus_positions` - A vector of 3D positions of the nuclei.
    /// * `parameters` - The parameters for the Jastrow factor.
    fn new(nucleus_positions: Vec<Vector3<f64>>, parameters: JastrowParameters) -> Self {
        JastrowFactor {
            nucleus_positions,
            parameters,
        }
    }

    /// Computes the Jastrow factor for given electron positions.
    ///
    /// # Arguments
    ///
    /// * `electron_positions` - A vector of 3D positions of the electrons.
    ///
    /// # Returns
    ///
    /// * The computed Jastrow factor as a `f64`.
    fn compute(&self, electron_positions: &Vec<Vector3<f64>>) -> f64 {
        let mut u = 0.0;

        // Electron-electron interaction terms.
        for i in 0..electron_positions.len() {
            for j in (i + 1)..electron_positions.len() {
                let r_ij = (electron_positions[i]- electron_positions[j]).norm();
                u += self.parameters.a * r_ij / (1.0 + self.parameters.b * r_ij);
            }
        }

        // Electron-nucleus interaction terms.
        for electron_pos in electron_positions {
            for nucleus_pos in &self.nucleus_positions {
                let r_iI = (electron_pos - nucleus_pos).norm();
                u += self.parameters.c * r_iI / (1.0 + self.parameters.d * r_iI);
            }
        }

        (-u).exp()
    }
}

impl MultiWfn for JastrowFactor {
    fn initialize(&mut self) -> Vec<Vector3<f64>> {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
        let mut r = vec![];
        for _ in 0..self.nucleus_positions.len() {
            r.push(Vector3::<f64>::from_distribution(&dist, &mut rng));
        }
        r
    }

    fn evaluate(&mut self, r: &Vec<Vector3<f64>>) -> f64 {
        self.compute(r)
    }

    fn derivative(&mut self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        let mut derivative = vec![Vector3::zeros(); r.len()];

        // Electron-electron interaction terms.
        for i in 0..r.len() {
            for j in (i + 1)..r.len() {
                let r_ij = (r[i] - r[j]).norm();
                let du = self.parameters.a / (1.0 + self.parameters.b * r_ij).powi(2);
                derivative[i] += du * (r[i] - r[j]) / r_ij;
                derivative[j] -= du * (r[i] - r[j]) / r_ij;
            }
        }

        // Electron-nucleus interaction terms.
        for i in 0..r.len() {
            for nucleus_pos in &self.nucleus_positions {
                let r_iI = (r[i] - nucleus_pos).norm();
                let du = self.parameters.c / (1.0 + self.parameters.d * r_iI).powi(2);
                derivative[i] += du * (r[i] - nucleus_pos) / r_iI;
            }
        }

        derivative
    }

    fn laplacian(&mut self, r: &Vec<Vector3<f64>>) -> Vec<f64> {
        let mut laplacian = vec![0.0; r.len()];

        // Electron-electron interaction terms.
        for i in 0..r.len() {
            for j in (i + 1)..r.len() {
                let r_ij = (r[i] - r[j]).norm();
                let du = self.parameters.a / (1.0 + self.parameters.b * r_ij).powi(2);
                let d2 = -2.0 * self.parameters.a * self.parameters.b / (1.0 + self.parameters.b * r_ij).powi(3);
                laplacian[i] += (d2 / r_ij + du) * (r[i] - r[j]).dot(&(r[i] - r[j])) / r_ij;
                laplacian[j] += (d2 / r_ij + du) * (r[i] - r[j]).dot(&(r[i] - r[j])) / r_ij;
            }
        }

        // Electron-nucleus interaction terms.
        for i in 0..r.len() {
            for nucleus_pos in &self.nucleus_positions {
                let r_iI = (r[i] - nucleus_pos).norm();
                let du = self.parameters.c / (1.0 + self.parameters.d * r_iI).powi(2);
                let d2 = -2.0 * self.parameters.c * self.parameters.d / (1.0 + self.parameters.d * r_iI).powi(3);
                laplacian[i] += (d2 / r_iI + du) * (r[i] - nucleus_pos).dot(&(r[i] - nucleus_pos)) / r_iI;
            }
        }

        laplacian
    }
}

fn u(r: f64, a: f64, b: f64) -> f64 {
    a * r / (1.0 + b * r)
}

fn du_dr(r: f64, a: f64, b: f64) -> f64 {
    a / (1.0 + b * r).powi(2)
}

fn d2u_dr2(r: f64, a: f64, b: f64) -> f64 {
    -2.0 * a * b / (1.0 + b * r).powi(3)
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
                    self.s[(i, j)] = self.sto[j].evaluate(&r[i]);
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
        // Calculate the determinant
        let psi_s = self.s.determinant();

        // // Compute the Jastrow factor
        // let mut jastrow = 0.0;
        // for i in 0..self.n {
        //     for j in (i + 1)..self.n {
        //         let r_ij = (r[i] - r[j]).norm();
        //         // You may define a_ij and b as constants or functions of i and j
        //         let a_ij = 0.5; // Example value
        //         let b = 0.5;    // Example value
        //         jastrow += u(r_ij, a_ij, b);
        //     }
        // }
        // let j = f64::exp(jastrow);

        // Update the inverse of the Slater matrix
        self.inv_s = self.s.clone().try_inverse().unwrap();
        // psi_s * j
        psi_s
    }

    fn derivative(&mut self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        let psi = self.evaluate(r);
        let mut derivative = vec![Vector3::zeros(); r.len()];

        // Compute gradients of the Slater determinant
        for i in 0..self.n {
            let mut sum = Vector3::zeros();
            for j in 0..self.n {
                if self.spin[i] == self.spin[j] {
                    let grad_phi = self.sto[j].derivative(&r[i]);
                    sum += self.inv_s[(j, i)] * grad_phi;
                }
            }
            derivative[i] = psi*sum;
        }

        // // Compute gradients of the Jastrow factor
        // let mut grad_ln_j = vec![Vector3::zeros(); self.n];
        // for i in 0..self.n {
        //     for j in 0..self.n {
        //         if i != j {
        //             let r_ij = r[i] - r[j];
        //             let r_ij_norm = r_ij.norm();
        //             let a_ij = 0.5; // Example value
        //             let b = 0.5;    // Example value
        //             let du = du_dr(r_ij_norm, a_ij, b);
        //             grad_ln_j[i] += du * r_ij / r_ij_norm;
        //         }
        //     }
        // }
        //
        // // Combine gradients
        // for i in 0..self.n {
        //     derivative[i] = psi * (derivative[i] + grad_ln_j[i]);
        // }
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
                sum_lap += self.inv_s[(j, i)] * lap_phi[i][j];
            }
            let mut sum_grad_dot = 0.0;
            // for j in 0..self.n {
            //     for k in 0..self.n {
            //         sum_grad_dot += self.inv_s[(j, i)] * self.inv_s[(k, i)] * grad_phi[i][j].dot(&grad_phi[i][k]);
            //     }
            // }
            laplacian[i] = psi * (sum_lap + sum_grad_dot);
        }

        // lapaclian of Jastrow factor
        // for i in 0..self.n {
        //     for j in 0..self.n {
        //         if i != j {
        //             let r_ij = r[i] - r[j];
        //             let r_ij_norm = r_ij.norm();
        //             let a_ij = 0.5; // Example value
        //             let b = 0.5;    // Example value
        //             let du = du_dr(r_ij_norm, a_ij, b);
        //             let d2u = d2u_dr2(r_ij_norm, a_ij, b);
        //             laplacian[i] += 2.0 * (d2u / r_ij_norm + du) * r_ij.dot(&r_ij) / r_ij_norm;
        //         }
        //     }
        // }
        laplacian
    }
}

// combine sto and jastrow together for lithium atom
pub(crate) struct Lithium {
    pub(crate) sto: STOSlaterDet,
    pub(crate) jastrow: Jastrow2,
}

impl MultiWfn for Lithium {
    fn initialize(&mut self) -> Vec<Vector3<f64>> {
        let r = self.sto.initialize();
        r
    }

    fn evaluate(&mut self, r: &Vec<Vector3<f64>>) -> f64 {
        let psi = self.sto.evaluate(r);
        let jastrow = self.jastrow.evaluate(r);
        psi * jastrow
    }

    fn derivative(&mut self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>> {
        let psi = self.sto.evaluate(r);
        let jastrow = self.jastrow.evaluate(r);
        let grad_psi = self.sto.derivative(r);
        let grad_jastrow = self.jastrow.derivative(r);
        let mut grad = vec![Vector3::zeros(); r.len()];
        for i in 0..r.len() {
            grad[i] = psi * grad_jastrow[i] + jastrow * grad_psi[i];
        }
        grad
    }

    fn laplacian(&mut self, r: &Vec<Vector3<f64>>) -> Vec<f64> {
        let psi = self.sto.evaluate(r);
        let jastrow = self.jastrow.evaluate(r);
        let lap_psi = self.sto.laplacian(r);
        let lap_jastrow = self.jastrow.laplacian(r);
        let mut lap = vec![0.0; r.len()];
        for i in 0..r.len() {
            // tbd
            lap[i] = psi * lap_jastrow[i] + jastrow * lap_psi[i] +
                2.0 * self.sto.derivative(r)[i].dot(&self.jastrow.derivative(r)[i]);
        }
        lap
    }
}


// implement energy for STO
impl EnergyCalculator for STOSlaterDet {
    fn local_energy(&mut self, r: &Vec<Vector3<f64>>) -> f64 {
        let psi = self.evaluate(r);
        let mut kinetic_energy = 0.0;
        let mut potential_energy = 0.0;
        let mut laplacian = vec![0.0; r.len()];
        let mut grad = vec![Vector3::zeros(); r.len()];
        for i in 0..self.n {
            let mut sum_lap = 0.0;
            for j in 0..self.n {
                sum_lap += self.inv_s[(j, i)] * self.sto[j].laplacian(&r[i]);
            }
            let mut sum_grad_dot = 0.0;
            for j in 0..self.n {
                for k in 0..self.n {
                    sum_grad_dot += self.inv_s[(j, i)] * self.inv_s[(k, i)] *
                        self.sto[j].derivative(&r[i]).dot(&self.sto[k].derivative(&r[i]));
                }
            }
            laplacian[i] = psi * (sum_lap + sum_grad_dot);
            grad[i] = psi * self.sto[i].derivative(&r[i]);
        }
        // Compute the kinetic energy
        for i in 0..self.n {
            kinetic_energy += -0.5 * laplacian[i];
        }
        // Compute the potential energy
        for i in 0..self.n {
            for j in 0..self.n {
                if self.spin[i] == self.spin[j] {
                    potential_energy += -1.0 / (r[i] - self.sto[j].R).norm();
                }
            }
        }
        kinetic_energy + potential_energy
    }
}

impl EnergyCalculator for Lithium {
    fn local_energy(&mut self, r: &Vec<Vector3<f64>>) -> f64 {
        let psi = self.sto.evaluate(r);
        let jastrow = self.jastrow.evaluate(r);
        let mut kinetic_energy = 0.0;
        let mut potential_energy = 0.0;
        let mut laplacian = vec![0.0; r.len()];
        let mut grad = vec![Vector3::zeros(); r.len()];
        for i in 0..r.len() {
            let mut sum_lap = 0.0;
            for j in 0..self.sto.n {
                sum_lap += self.sto.inv_s[(j, i)] * self.sto.sto[j].laplacian(&r[i]);
            }
            let mut sum_grad_dot = 0.0;
            for j in 0..self.sto.n {
                for k in 0..self.sto.n {
                    sum_grad_dot += self.sto.inv_s[(j, i)] * self.sto.inv_s[(k, i)] *
                        self.sto.sto[j].derivative(&r[i]).dot(&self.sto.sto[k].derivative(&r[i]));
                }
            }
            laplacian[i] = psi * (sum_lap + sum_grad_dot);
            grad[i] = psi * self.sto.sto[i].derivative(&r[i]);
        }
        // Compute the kinetic energy
        for i in 0..r.len() {
            kinetic_energy += -0.5 * laplacian[i];
        }
        // Compute the potential energy
        for i in 0..r.len() {
            for j in 0..self.sto.n {
                if self.sto.spin[i] == self.sto.spin[j] {
                    potential_energy += -1.0 / (r[i] - self.sto.sto[j].R).norm();
                }
            }
        }
        kinetic_energy + potential_energy
    }
}


