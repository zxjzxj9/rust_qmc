use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct STO {
    pub(crate) m: i32,
    pub(crate) phi: Vec<f64>,
    pub(crate) p: Vec<f64>,
    pub(crate) zeta: Vec<f64>,
    pub(crate) R: Vector3<f64>,
}