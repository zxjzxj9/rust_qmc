#[cfg(test)]
mod tests {
    extern crate nalgebra as na;

    use approx::assert_relative_eq;
    use na::{Vector3};
    use rand::distributions::Uniform;
    use rand_distr::{Distribution, Normal};
    use crate::h2_mol::{Jastrow1, H2MoleculeVB, Slater1s};
    use crate::jastrow::Jastrow2;
    use crate::wfn::{MultiWfn, SingleWfn};
    use crate::sto::{STO, STOSlaterDet, init_li_sto};

    #[test]
    fn test_jastrow1_evaluate() {
        let mut jastrow = Jastrow1 { F: 1.0 };
        let r = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let result = jastrow.evaluate(&r);
        assert_relative_eq!(result, 0.7788007831, epsilon = 1e-8);
    }

    #[test]
    fn test_jastrow1_derivative() {
        let mut jastrow = Jastrow1 { F: 1.0 };
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
        let mut jastrow = Jastrow1 { F: 1.0 };
        let r = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let result = jastrow.laplacian(&r);
    }

    #[test]
    fn test_jastrow1_consistency() {
        let mut jastrow = Jastrow1 { F: 1.0 };
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
        let mut jastrow = Jastrow1 { F: 1.0 };
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

        for i in 0..r.len() {
            assert_relative_eq!(analytical_laplacian[i], numerical_laplacian[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_h2molecule_numerical_derivative_and_laplacian() {
        let mut h2 = H2MoleculeVB {
            H1: Slater1s { R: Vector3::new(0.0, 0.0, 0.0), alpha: 0.6 },
            H2: Slater1s { R: Vector3::new(1.0, 0.0, 0.0), alpha: 0.6 },
            J: Jastrow1 { F: 5.0 },
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

        for i in 0..r.len() {
            assert_relative_eq!(analytical_laplacian[i], numerical_laplacian[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_sto_numerical_derivative_and_laplacian() {
        let mut sto = init_li_sto(Vector3::new(1.0, 0.0, 0.0), 1, 0, 0);
        let h = 1e-5;

        // define a normal distribution
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();

        // test with random positions between [-1, 1]
        let r = Vector3::<f64>::from_distribution(&dist, &mut rng);

        let analytical_grad = sto.derivative(&r);
        let numerical_grad = sto.numerical_derivative(&r, h);

        assert_relative_eq!(analytical_grad.x, numerical_grad.x, epsilon = 1e-5);
        assert_relative_eq!(analytical_grad.y, numerical_grad.y, epsilon = 1e-5);
        assert_relative_eq!(analytical_grad.z, numerical_grad.z, epsilon = 1e-5);

        let analytical_laplacian = sto.laplacian(&r);
        let numerical_laplacian = sto.numerical_laplacian(&r, h);

        assert_relative_eq!(analytical_laplacian, numerical_laplacian, epsilon = 1e-5);
    }

    #[test]
    fn test_lithium_numerical_derivative_and_laplacian() {
        let mut sto1 = init_li_sto(Vector3::new(0.0, 0.0, 0.0), 1, 0, 0);
        let mut sto2 = init_li_sto(Vector3::new(0.0, 0.0, 0.0), 1, 0, 0);
        let mut sto3 = init_li_sto(Vector3::new(0.0, 0.0, 0.0), 2, 0, 0);
        let mut stodet = STOSlaterDet {
            n: 3,
            sto: vec![sto1, sto2, sto3],
            spin: vec![1, -1, 1],
            s: Default::default(),
            inv_s: Default::default(),
        };
        let h = 1e-5;

        let r = stodet.initialize();
        // evaluate numerical derivative and laplacian
        let analytical_grad = stodet.derivative(&r);
        let numerical_grad = stodet.numerical_derivative(&r, h);
        // assert they are close enough
        for i in 0..r.len() {
            assert_relative_eq!(analytical_grad[i].x, numerical_grad[i].x, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].y, numerical_grad[i].y, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].z, numerical_grad[i].z, epsilon = 1e-5);
        }

        // evaluate numerical laplacian
        let analytical_laplacian = stodet.laplacian(&r);
        let numerical_laplacian = stodet.numerical_laplacian(&r, h);
        // assert they are close enough
        for i in 0..r.len() {
            assert_relative_eq!(analytical_laplacian[i], numerical_laplacian[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_jastrow2_numerical_derivative_and_laplacian() {
        // add test for jastrow2
        let mut jastrow2 = Jastrow2 {
            num_electrons: 4,
            F: 1.0,
        };

        let r = jastrow2.initialize();

        let h = 1e-5;

        let analytical_grad = jastrow2.derivative(&r);
        let numerical_grad = jastrow2.numerical_derivative(&r, h);

        for i in 0..r.len() {
            assert_relative_eq!(analytical_grad[i].x, numerical_grad[i].x, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].y, numerical_grad[i].y, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].z, numerical_grad[i].z, epsilon = 1e-5);
        }

        // evaluate numerical laplacian
        let analytical_laplacian = jastrow2.laplacian(&r);
        let numerical_laplacian = jastrow2.numerical_laplacian(&r, h);
        // assert they are close enough
        for i in 0..r.len() {
            assert_relative_eq!(analytical_laplacian[i], numerical_laplacian[i], epsilon = 1e-5);
        }
    }

}
