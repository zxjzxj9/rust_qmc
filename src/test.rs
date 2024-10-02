#[cfg(test)]
mod tests {
    extern crate nalgebra as na;

    use approx::assert_relative_eq;
    use na::{Vector3};
    use rand::distributions::Uniform;
    use rand_distr::{Distribution, Normal};
    use crate::h2_mol::{Jastrow1, MultiWfn, H2MoleculeVB, Slater1s};

    #[test]
    fn test_jastrow1_evaluate() {
        let jastrow = Jastrow1 { F: 1.0 };
        let r = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let result = jastrow.evaluate(&r);
        assert_relative_eq!(result, 0.7788007831, epsilon = 1e-8);
    }

    #[test]
    fn test_jastrow1_derivative() {
        let jastrow = Jastrow1 { F: 1.0 };
        let r = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let result = jastrow.derivative(&r);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0].x, -0.09735009789, epsilon = 1e-6);
        assert_relative_eq!(result[0].y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[0].z, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[1].x, 0.09735009789, epsilon = 1e-6);
        assert_relative_eq!(result[1].y, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[1].z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_jastrow1_laplacian() {
        let jastrow = Jastrow1 { F: 1.0 };
        let r = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let result = jastrow.laplacian(&r);
    }

    #[test]
    fn test_jastrow1_consistency() {
        let jastrow = Jastrow1 { F: 1.0 };
        let r = vec![
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(-0.5, -0.5, -0.5),
        ];

        // Test that the wavefunction is symmetric
        let psi1 = jastrow.evaluate(&r);
        let psi2 = jastrow.evaluate(&vec![r[1], r[0]]);
        assert_relative_eq!(psi1, psi2, epsilon = 1e-8);

        // Test that the derivatives sum to zero (due to translational invariance)
        let deriv = jastrow.derivative(&r);
        let sum_deriv = deriv[0] + deriv[1];
        assert_relative_eq!(sum_deriv.norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_jastrow1_numerical_derivative_and_laplacian() {
        let jastrow = Jastrow1 { F: 1.0 };
        let h = 1e-5;

        // define a normal distribution
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();

        // test with random positions between [-1, 1]
        let r = vec![
            Vector3::<f64>::from_distribution(&dist, &mut rng),
            Vector3::<f64>::from_distribution(&dist, &mut rng),
        ];

        let analytical_grad = jastrow.derivative(&r);
        let numerical_grad = jastrow.numerical_derivative(&r, h);

        for i in 0..r.len() {
            assert_relative_eq!(analytical_grad[i].x, numerical_grad[i].x, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].y, numerical_grad[i].y, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].z, numerical_grad[i].z, epsilon = 1e-5);
        }

        let analytical_laplacian = jastrow.laplacian(&r);
        let numerical_laplacian = jastrow.numerical_laplacian(&r, h);

        assert_relative_eq!(analytical_laplacian, numerical_laplacian, epsilon = 1e-2);
    }

    #[test]
    fn test_h2molecule_numerical_derivative_and_laplacian() {
        let h2 = H2MoleculeVB {
            H1: Slater1s { R: Vector3::new(0.0, 0.0, 0.0), alpha: 1.0 },
            H2: Slater1s { R: Vector3::new(1.0, 0.0, 0.0), alpha: 1.0 },
            J: Jastrow1 { F: 1.0 },
        };
        let h = 1e-5;

        // define a normal distribution
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();

        // test with random positions between [-1, 1]
        let r = vec![
            Vector3::<f64>::from_distribution(&dist, &mut rng),
            Vector3::<f64>::from_distribution(&dist, &mut rng),
        ];

        let analytical_grad = h2.derivative(&r);
        let numerical_grad = h2.numerical_derivative(&r, h);

        for i in 0..r.len() {
            assert_relative_eq!(analytical_grad[i].x, numerical_grad[i].x, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].y, numerical_grad[i].y, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].z, numerical_grad[i].z, epsilon = 1e-5);
        }

        let analytical_laplacian = h2.laplacian(&r);
        let numerical_laplacian = h2.numerical_laplacian(&r, h);

        assert_relative_eq!(analytical_laplacian, numerical_laplacian, epsilon = 1e-2);
    }
}
