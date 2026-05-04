//! Path Integral Molecular Dynamics - Proton Transfer Tunneling
//!
//! Run with: cargo run --release --example pimd_proton_transfer
//!
//! This demonstrates quantum tunneling of a proton (H+) through a
//! double-well barrier using bead-based Path Integral Molecular Dynamics.
//!
//! The simulation compares:
//!   P=1 bead  (classical): proton trapped in one well
//!   P=32 beads (quantum):  ring polymer delocalizes -> tunneling!
//!
//! Physical system: symmetric O-H...O hydrogen bond
//!   - Proton mass: 1836.15 a.u.
//!   - Barrier: ~6 kcal/mol (typical H-bond)
//!   - Temperature: 300 K

use rust_qmc::sampling::run_pimd_proton_transfer;

fn main() {
    // --- Physical parameters -----------------------------------------
    let proton_mass = 1836.15;           // Proton mass in atomic units
    let temperature_k = 300.0;           // Temperature in Kelvin
    let beta = 315774.65 / temperature_k; // Inverse temperature in a.u.

    // Double-well parameters (symmetric O-H...O proton transfer)
    let barrier_height = 0.010;          // ~6.3 kcal/mol barrier
    let well_distance = 0.75;            // +/-0.75 Bohr between wells

    // --- Simulation parameters ---------------------------------------
    let n_beads = 32;                    // Trotter number (ring polymer beads)
    let n_polymers = 50;                 // Parallel replicas for statistics
    let dt = 0.5;                        // Time step in a.u. (~0.012 fs)
    let n_equilibrate = 20_000;          // Equilibration steps
    let n_production = 50_000;           // Production steps

    println!("================================================================");
    println!("|       PIMD Proton Transfer Tunneling Demonstration          |");
    println!("|                                                             |");
    println!("|   Quantum particle (ring polymer) vs Classical particle     |");
    println!("|   in a symmetric double-well O-H...O potential              |");
    println!("================================================================");
    println!();
    println!("A proton in a double-well potential:");
    println!("  • Classically (P=1): stays trapped in the starting well");
    println!("  • Quantum (P=32): ring polymer spreads across the barrier");
    println!("    -> beads populate BOTH wells = quantum tunneling!");
    println!();

    run_pimd_proton_transfer(
        n_polymers,
        n_beads,
        beta,
        proton_mass,
        barrier_height,
        well_distance,
        dt,
        n_equilibrate,
        n_production,
    );

    println!();
    println!("================================================================");
    println!("|  Output files:                                              |");
    println!("|    pimd_position_distribution.txt  -- |ψ(x)|² histogram     |");
    println!("|    pimd_proton_transfer.txt        -- energy trajectory      |");
    println!("|    pimd_bead_snapshot.txt           -- bead positions        |");
    println!("================================================================");
}
