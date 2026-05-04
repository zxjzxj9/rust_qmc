//! Benzene (C6H6) Geometry Optimization Example
//!
//! Starts from a slightly expanded ring geometry and optimizes nuclear
//! positions using VMC-estimated Hellmann-Feynman forces.
//!
//! Usage:
//!   cargo run --example benzene_geom_opt --release

use rust_qmc::{BenzeneGTO, GeometryOptimizer};
use nalgebra::Vector3;

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;
/// Bohr to Angstrom conversion factor
const BOHR_TO_ANG: f64 = 0.529177;

fn main() {
    println!("Benzene (C6H6) Geometry Optimization");
    println!("======================================\n");

    // Create wavefunction
    let mut wfn = BenzeneGTO::new(1.5, 3.0);

    let eq_nuclei = wfn.get_nuclei();
    let charges = wfn.get_charges();

    println!("Equilibrium geometry (6 carbons + 6 hydrogens):");
    print_geometry(&eq_nuclei, &charges);

    // Distort: expand the ring by 5%
    let mut distorted = eq_nuclei.clone();
    let scale = 1.05;
    // Scale all atom positions radially (they're all in the z=0 plane)
    for pos in distorted.iter_mut() {
        *pos = *pos * scale;
    }
    wfn.set_nuclei(&distorted);

    println!("\nDistorted geometry (ring expanded by 5%):");
    print_geometry(&distorted, &charges);

    // Print some bond lengths
    println!("\nC-C bond lengths:");
    for i in 0..6 {
        let j = (i + 1) % 6;
        let d = (distorted[i] - distorted[j]).norm();
        println!("  C{}-C{}: {:.4} Bohr ({:.4} A)", i, j, d, d * BOHR_TO_ANG);
    }
    println!("C-H bond lengths:");
    for i in 0..6 {
        let d = (distorted[i] - distorted[6 + i]).norm();
        println!("  C{}-H{}: {:.4} Bohr ({:.4} A)", i, i, d, d * BOHR_TO_ANG);
    }
    println!();

    // Configure optimizer — benzene is larger so use more samples
    let optimizer = GeometryOptimizer::new()
        .with_n_samples(2000)
        .with_n_walkers(10)
        .with_step_size(0.003)
        .with_max_iterations(20)
        .with_force_threshold(0.2)
        .with_verbose(true);

    // Run optimization
    let result = optimizer.optimize(&mut wfn);

    // Summary
    println!("\n=== Optimization Summary ===");
    println!("Final energy:    {:.4} Ha ({:.2} eV)", result.final_energy, result.final_energy * HA_TO_EV);
    println!("Final max |F|:   {:.4} Ha/Bohr", result.final_max_force);

    println!("\nFinal C-C bond lengths:");
    for i in 0..6 {
        let j = (i + 1) % 6;
        let d = (result.final_nuclei[i] - result.final_nuclei[j]).norm();
        println!("  C{}-C{}: {:.4} Bohr ({:.4} A)", i, j, d, d * BOHR_TO_ANG);
    }
    println!("Final C-H bond lengths:");
    for i in 0..6 {
        let d = (result.final_nuclei[i] - result.final_nuclei[6 + i]).norm();
        println!("  C{}-H{}: {:.4} Bohr ({:.4} A)", i, i, d, d * BOHR_TO_ANG);
    }
}

fn print_geometry(nuclei: &[Vector3<f64>], charges: &[f64]) {
    for (i, (pos, z)) in nuclei.iter().zip(charges.iter()).enumerate() {
        let label = if *z > 5.0 { "C" } else { "H" };
        println!("  {}{}: ({:+.4}, {:+.4}, {:+.4}) Bohr",
            label, i, pos.x, pos.y, pos.z);
    }
}
