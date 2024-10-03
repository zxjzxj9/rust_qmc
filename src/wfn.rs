use nalgebra::Vector3;

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
    fn laplacian(&self, r: &Vec<Vector3<f64>>) -> Vec<f64>;

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

    fn numerical_laplacian(&self, r: &Vec<Vector3<f64>>, h: f64) -> Vec<f64> {
        let mut laplacian = vec![0.0; r.len()];

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

                laplacian[i] += second_derivative;
            }
        }

        laplacian
    }
}