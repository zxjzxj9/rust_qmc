//! CH4 Methane Geometry Optimization Example
//!
//! Starts from a slightly distorted geometry and optimizes nuclear positions
//! using VMC-estimated Hellmann-Feynman forces.
//!
//! Usage:
//!   cargo run --example ch4_geom_opt --release

use rust_qmc::{MethaneGTO, GeometryOptimizer};
use nalgebra::Vector3;

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;
/// Bohr to Angstrom conversion factor
const BOHR_TO_ANG: f64 = 0.529177;

fn main() {
    println!("CH4 Methane Geometry Optimization");
    println!("==================================\n");

    // Create wavefunction with reasonable Jastrow parameters
    let mut wfn = MethaneGTO::new(1.5, 3.0);

    // Get the equilibrium geometry
    let eq_nuclei = wfn.get_nuclei();
    println!("Equilibrium geometry:");
    print_geometry(&eq_nuclei, &wfn.get_charges());

    // Mild distortion: stretch all C-H bonds by 5% (uniform scaling of H positions)
    let mut distorted = eq_nuclei.clone();
    for i in 1..5 {
        let dir = (distorted[i] - distorted[0]).normalize();
        let eq_len = (eq_nuclei[i] - eq_nuclei[0]).norm();
        distorted[i] = distorted[0] + dir * eq_len * 1.05;
    }
    wfn.set_nuclei(&distorted);

    println!("\nDistorted geometry (all C-H bonds stretched by 5%):");
    print_geometry(&distorted, &wfn.get_charges());

    println!("\nBond lengths:");
    for i in 1..5 {
        let d = (distorted[i] - distorted[0]).norm();
        println!("  C-H{}: {:.4} Bohr ({:.4} A)", i, d, d * BOHR_TO_ANG);
    }
    println!();

    // Configure optimizer
    let optimizer = GeometryOptimizer::new()
        .with_n_samples(5000)
        .with_n_walkers(20)
        .with_step_size(0.001)
        .with_max_iterations(30)
        .with_force_threshold(0.5)
        .with_verbose(true);

    // Run optimization
    let result = optimizer.optimize(&mut wfn);

    // Summary
    println!("\n=== Optimization Summary ===");
    println!("Final energy:    {:.4} Ha ({:.2} eV)", result.final_energy, result.final_energy * HA_TO_EV);
    println!("Final max |F|:   {:.4} Ha/Bohr", result.final_max_force);
    println!("\nFinal bond lengths:");
    for i in 1..5 {
        let d = (result.final_nuclei[i] - result.final_nuclei[0]).norm();
        println!("  C-H{}: {:.4} Bohr ({:.4} A)", i, d, d * BOHR_TO_ANG);
    }

    if result.energy_history.len() > 1 {
        let e_init = result.energy_history[0];
        let e_final = result.final_energy;
        println!("\nEnergy change: {:.4} Ha ({:.2} eV)",
            e_final - e_init, (e_final - e_init) * HA_TO_EV);
    }

    println!("\nReference: HF/STO-3G ~ -39.7 Ha, Experiment ~ -40.5 Ha");
}

fn print_geometry(nuclei: &[Vector3<f64>], charges: &[f64]) {
    for (i, (pos, z)) in nuclei.iter().zip(charges.iter()).enumerate() {
        let label = if *z > 5.0 { "C" } else { "H" };
        println!("  {}{}: ({:+.4}, {:+.4}, {:+.4}) Bohr",
            label, i, pos.x, pos.y, pos.z);
    }
}
