//! Path Integral Monte Carlo for Sombrero (Mexican Hat) Potential
//!
//! Run with: cargo run --release --example pimc_sombrero
//!
//! The sombrero potential: V(x) = -μ²x²/2 + λx⁴/4
//! Features double-well structure with quantum tunneling between wells.

use rust_qmc::sampling::run_pimc_sombrero;

fn main() {
    // Simulation parameters
    let n_paths = 100;          // Number of parallel walkers
    let n_beads = 128;          // Trotter slices (M)
    let beta = 20.0;            // Inverse temperature (lower = more classical)
    let well_position: f64 = 1.0;    // Minima at x = ±1
    let barrier_height = 2.0;   // Height of barrier at x=0
    let n_thermalize = 5000;    // Thermalization sweeps
    let n_production = 10000;   // Production sweeps

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Path Integral Monte Carlo - Sombrero Potential           ║");
    println!("║     (Mexican Hat / Double-Well with Quantum Tunneling)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("The sombrero potential: V(x) = -μ²x²/2 + λx⁴/4");
    println!("                            = {}(x² - {}²)²", 
             barrier_height / well_position.powi(4), well_position);
    println!();
    println!("Physical insight:");
    println!("  - At high T (small β): classical, particle in one well");
    println!("  - At low T (large β): quantum, wavefunction spreads over both wells");
    println!("  - Ground state: symmetric, <x> ≈ 0");
    println!("  - First excited: antisymmetric, energy split by tunneling");
    println!();

    run_pimc_sombrero(
        n_paths,
        n_beads,
        beta,
        well_position,
        barrier_height,
        n_thermalize,
        n_production,
    );

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Files written:                                              ║");
    println!("║    sombrero_wavefunction.txt - |ψ(x)|² histogram            ║");
    println!("║    sombrero_energies.txt     - energy vs step               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
