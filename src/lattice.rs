use nalgebra::Matrix3;

// define lattice vector struct, contains vx vy vz, 3x3 matrix
#[derive(Debug, Clone, Copy)]
pub struct LatticeVector {
    pub lattice_vector: Matrix3<f64>
}

// ewald summation code
pub fn ewald() {
    // define the lattice vector
    let lattice_vector = LatticeVector {
        lattice_vector: Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    };
    // print the lattice vector
    println!("{:?}", lattice_vector);
}