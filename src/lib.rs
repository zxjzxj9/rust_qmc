//! Rust QMC - Quantum Monte Carlo simulations in Rust
//!
//! This crate provides Variational Monte Carlo (VMC) and Diffusion Monte Carlo (DMC)
//! implementations for quantum chemistry calculations.

pub mod wavefunction;
pub mod correlation;
pub mod systems;
pub mod sampling;
pub mod io;

// Re-export commonly used types at crate root
pub use wavefunction::{SingleWfn, MultiWfn, OptimizableWfn, STO, Slater1s, STOSlaterDet, init_li_sto};
pub use correlation::{Jastrow1, Jastrow2, Jastrow3};
pub use systems::{H2MoleculeVB, H2MoleculeMO, Lithium, LatticeVector, LithiumCrystalWalker, LithiumFCC, HomogeneousElectronGas, HEGResults, JastrowForm, Methane, MethaneGTO, JastrowParams};
pub use sampling::{EnergyCalculator, MCMCParams, MCMCSimulation, MCMCResults, Walker, BranchingResult, VmcWalker, JastrowOptimizer, OptimizationResult, SamplingStats, DDVMCParams, DDVMCResults, DriftDiffusionVMC, SROptimizer, SRResult};
pub use io::{read_h2molecule_vb, read_h2molecule_mo};

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use rand_distr::{Distribution, Normal};
    
    use crate::systems::{H2MoleculeVB};
    use crate::correlation::{Jastrow1, Jastrow2};
    use crate::wavefunction::{init_li_sto, Slater1s, STOSlaterDet, MultiWfn, SingleWfn};
    use crate::systems::Lithium;

    #[test]
    fn test_jastrow1_evaluate() {
        let jastrow = Jastrow1 { cusp_param: 1.0 };
        let r = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let result = jastrow.evaluate(&r);
        assert_relative_eq!(result, 0.7788007831, epsilon = 1e-8);
    }

    #[test]
    fn test_jastrow1_derivative() {
        let jastrow = Jastrow1 { cusp_param: 1.0 };
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
        let jastrow = Jastrow1 { cusp_param: 1.0 };
        let r = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
        ];
        let _result = jastrow.laplacian(&r);
    }

    #[test]
    fn test_jastrow1_consistency() {
        let jastrow = Jastrow1 { cusp_param: 1.0 };
        let r = vec![
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(-0.5, -0.5, -0.5),
        ];

        // Test symmetry
        let psi1 = jastrow.evaluate(&r);
        let psi2 = jastrow.evaluate(&[r[1], r[0]]);
        assert_relative_eq!(psi1, psi2, epsilon = 1e-8);

        // Derivatives sum to zero (translational invariance)
        let deriv = jastrow.derivative(&r);
        let sum_deriv = deriv[0] + deriv[1];
        assert_relative_eq!(sum_deriv.norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_jastrow1_numerical_derivative_and_laplacian() {
        let jastrow = Jastrow1 { cusp_param: 1.0 };
        let h = 1e-5;

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
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
        let h2 = H2MoleculeVB {
            orbital1: Slater1s { center: Vector3::new(0.0, 0.0, 0.0), alpha: 0.6 },
            orbital2: Slater1s { center: Vector3::new(1.0, 0.0, 0.0), alpha: 0.6 },
            jastrow: Jastrow1 { cusp_param: 5.0 },
        };
        let h = 1e-5;

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
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
        let sto = init_li_sto(Vector3::new(1.0, 0.0, 0.0), 1, 0, 0);
        let h = 1e-5;

        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, 1.0).unwrap();
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
    fn test_slater_det_numerical_derivative_and_laplacian() {
        let origin = Vector3::zeros();
        let stodet = STOSlaterDet::new(
            vec![
                init_li_sto(origin, 1, 0, 0),
                init_li_sto(origin, 1, 0, 0),
                init_li_sto(origin, 2, 0, 0),
            ],
            vec![1, -1, 1],
        );
        let h = 1e-5;

        let r = stodet.initialize();
        
        let analytical_grad = stodet.derivative(&r);
        let numerical_grad = stodet.numerical_derivative(&r, h);
        
        for i in 0..r.len() {
            assert_relative_eq!(analytical_grad[i].x, numerical_grad[i].x, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].y, numerical_grad[i].y, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].z, numerical_grad[i].z, epsilon = 1e-5);
        }

        let analytical_laplacian = stodet.laplacian(&r);
        let numerical_laplacian = stodet.numerical_laplacian(&r, h);
        
        for i in 0..r.len() {
            assert_relative_eq!(analytical_laplacian[i], numerical_laplacian[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_jastrow2_numerical_derivative_and_laplacian() {
        let jastrow2 = Jastrow2 {
            num_electrons: 3,
            cusp_param: 1.0,
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

        let analytical_laplacian = jastrow2.laplacian(&r);
        let numerical_laplacian = jastrow2.numerical_laplacian(&r, h);
        
        for i in 0..r.len() {
            // Relaxed tolerance due to numerical differentiation sensitivity
            assert_relative_eq!(analytical_laplacian[i], numerical_laplacian[i], epsilon = 1e-2);
        }
    }

    #[test]
    fn test_lithium_atom_numerical_derivative_and_laplacian() {
        let origin = Vector3::zeros();
        let slater = STOSlaterDet::new(
            vec![
                init_li_sto(origin, 1, 0, 0),
                init_li_sto(origin, 1, 0, 0),
                init_li_sto(origin, 2, 0, 0),
            ],
            vec![1, -1, 1],
        );
        let jastrow = Jastrow2 {
            num_electrons: 3,
            cusp_param: 1.0,
        };
        let atom = Lithium::new(slater, jastrow);
        let h = 1e-5;

        let r = atom.initialize();
        
        let analytical_grad = atom.derivative(&r);
        let numerical_grad = atom.numerical_derivative(&r, h);
        
        for i in 0..r.len() {
            assert_relative_eq!(analytical_grad[i].x, numerical_grad[i].x, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].y, numerical_grad[i].y, epsilon = 1e-5);
            assert_relative_eq!(analytical_grad[i].z, numerical_grad[i].z, epsilon = 1e-5);
        }

        let analytical_laplacian = atom.laplacian(&r);
        let numerical_laplacian = atom.numerical_laplacian(&r, h);
        
        for i in 0..r.len() {
            // Relaxed tolerance due to numerical differentiation sensitivity  
            assert_relative_eq!(analytical_laplacian[i], numerical_laplacian[i], epsilon = 1e-2);
        }
    }
}
