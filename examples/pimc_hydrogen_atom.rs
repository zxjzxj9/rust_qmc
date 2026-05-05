//! Fixed-Node Path Integral Monte Carlo for Hydrogen Atom
//!
//! Run with: cargo run --release --example pimc_hydrogen_atom
//!
//! Uses fixed-node approximation with Slater 1s trial wavefunction.
//! For hydrogen (1 electron), the ground state has no nodes,
//! so the fixed-node approximation is EXACT.

use rust_qmc::sampling::run_pimc_hydrogen;

fn main() {
    // Simulation parameters
    let n_paths = 100;          // Number of parallel walkers
    let n_beads = 64;           // Trotter slices (M)
    let beta = 40.0;            // Inverse temperature (large = ground state)
    let alpha = 1.0;            // Optimal variational parameter for H
    let n_thermalize = 3000;    // Thermalization sweeps
    let n_production = 10000;   // Production sweeps

    println!("================================================================");
    println!("|     Fixed-Node Path Integral Monte Carlo                     |");
    println!("|               Hydrogen Atom (1 electron)                     |");
    println!("================================================================");
    println!();
    println!("Trial wavefunction: Psi_T = exp(-alpha*r) with alpha = {:.2}", alpha);
    println!();
    println!("Physical insight:");
    println!("  - 1s orbital has NO NODES -> fixed-node is EXACT");
    println!("  - Ground state energy: E0 = -0.5 Hartree = -13.6 eV");
    println!("  - Average radius: <r> = 1.5 a0 (Bohr radii)");
    println!();

    run_pimc_hydrogen(
        n_paths,
        n_beads,
        beta,
        alpha,
        n_thermalize,
        n_production,
    );

    println!();
    println!("================================================================");
    println!("|  Exact results for hydrogen ground state:                    |");
    println!("|    E0 = -0.5 Hartree = -13.606 eV                           |");
    println!("|    <r> = 1.5 a0 = 0.794 A                                   |");
    println!("|    <r^2> = 3.0 a0^2                                           |");
    println!("================================================================");
}
