use nalgebra::Matrix3;

// define lattice vector struct, contains vx vy vz, 3x3 matrix
#[derive(Debug, Clone, Copy)]
pub struct LatticeVector {
    pub lattice_vector: Matrix3<f64>
}

// ewald summation code
