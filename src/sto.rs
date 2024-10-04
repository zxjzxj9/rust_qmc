use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

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