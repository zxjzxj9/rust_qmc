//! Lithium atom wavefunction for QMC calculations.

use nalgebra::Vector3;
use crate::correlation::Jastrow2;
use crate::sampling::EnergyCalculator;
use crate::wavefunction::{MultiWfn, STOSlaterDet};

/// Lithium atom wavefunction: Slater determinant × Jastrow factor.
pub struct Lithium {
    pub slater: STOSlaterDet,
    pub jastrow: Jastrow2,
}

impl Lithium {
    /// Create a new Lithium atom wavefunction.
    pub fn new(slater: STOSlaterDet, jastrow: Jastrow2) -> Self {
        Self { slater, jastrow }
    }
}

impl MultiWfn for Lithium {
    fn initialize(&self) -> Vec<Vector3<f64>> {
        self.slater.initialize()
    }

    fn evaluate(&self, r: &[Vector3<f64>]) -> f64 {
        self.slater.evaluate(r) * self.jastrow.evaluate(r)
    }

    fn derivative(&self, r: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        let psi = self.slater.evaluate(r);
        let j = self.jastrow.evaluate(r);
        let grad_psi = self.slater.derivative(r);
        let grad_j = self.jastrow.derivative(r);
        
        grad_psi.into_iter()
            .zip(grad_j.into_iter())
            .map(|(gp, gj)| j * gp + psi * gj)
            .collect()
    }

    fn laplacian(&self, r: &[Vector3<f64>]) -> Vec<f64> {
        let psi = self.slater.evaluate(r);
        let j = self.jastrow.evaluate(r);
        let lap_psi = self.slater.laplacian(r);
        let lap_j = self.jastrow.laplacian(r);
        let grad_psi = self.slater.derivative(r);
        let grad_j = self.jastrow.derivative(r);
        
        lap_psi.into_iter()
            .zip(lap_j.into_iter())
            .zip(grad_psi.into_iter().zip(grad_j.into_iter()))
            .map(|((lp, lj), (gp, gj))| {
                j * lp + psi * lj + 2.0 * gp.dot(&gj)
            })
            .collect()
    }
}

impl EnergyCalculator for Lithium {
    fn local_energy(&self, r: &[Vector3<f64>]) -> f64 {
        let psi = self.evaluate(r);
        let laplacian = self.laplacian(r);
        
        // Kinetic energy
        let kinetic = -0.5 * laplacian.iter().sum::<f64>() / psi;
        
        // Electron-nucleus potential (Li nucleus Z=3 at origin)
        let v_en: f64 = r.iter()
            .map(|ri| -3.0 / ri.norm())
            .sum();
        
        // Electron-electron repulsion
        let n = r.len();
        let v_ee: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| 1.0 / (r[i] - r[j]).norm())
            .sum();
        
        kinetic + v_en + v_ee
    }
}

impl EnergyCalculator for STOSlaterDet {
    fn local_energy(&self, r: &[Vector3<f64>]) -> f64 {
        let psi = self.evaluate(r);
        let laplacian = self.laplacian(r);
        
        // Kinetic energy: -½ Σᵢ ∇²ψ/ψ
        let kinetic = -0.5 * laplacian.iter().sum::<f64>() / psi;
        
        // Potential energy: electron-nucleus attraction (Li nucleus Z=3)
        // All electrons attracted to nucleus at origin
        let potential: f64 = r.iter()
            .map(|ri| -3.0 / ri.norm())
            .sum();
        
        kinetic + potential
    }
}
