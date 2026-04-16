//! Multi-Atom PIMD — Bifluoride HF₂⁻ Proton Transfer
//!
//! Run with: cargo run --release --example pimd_bifluoride
//!
//! Demonstrates quantum tunneling of a proton between two fluorine atoms
//! in the bifluoride ion (F−H···F ↔ F···H−F) using 3D multi-atom PIMD.
//!
//! This is the simplest molecular proton transfer system:
//! - 3 atoms (F, H, F), 9 degrees of freedom
//! - Symmetric double-well PES (LEPS-style Morse coupling)
//! - Barrier: ~1.5 kcal/mol  
//! - Experimentally: very strong tunneling (~1000 cm⁻¹ splitting)

use rust_qmc::sampling::run_pimd_bifluoride;

fn main() {
    // Physical parameters
    let temperature_k = 300.0;
    let beta = 315774.65 / temperature_k;

    // Simulation parameters  
    let n_beads = 32;
    let n_polymers = 30;
    let dt = 0.3;               // a.u. (~0.007 fs)
    let n_equilibrate = 20_000;
    let n_production = 50_000;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   Multi-Atom PIMD: Bifluoride HF₂⁻ Proton Transfer        ║");
    println!("║                                                             ║");
    println!("║   F − H ··· F  ↔  F ··· H − F                             ║");
    println!("║                                                             ║");
    println!("║   3 atoms, 9 DOF, symmetric double-well                    ║");
    println!("║   Barrier: ~1.5 kcal/mol                                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    run_pimd_bifluoride(
        n_polymers,
        n_beads,
        beta,
        dt,
        n_equilibrate,
        n_production,
    );

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Output files:                                              ║");
    println!("║    pimd_bifluoride_distribution.txt — P(δ) histogram       ║");
    println!("║    pimd_bifluoride_energy.txt       — energy trajectory     ║");
    println!("║    pimd_bifluoride_beads.txt        — 3D bead positions     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
