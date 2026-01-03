//! Path Integral Monte Carlo for Harmonic Oscillator
//!
//! Run with: cargo run --release --example pimc_harmonic
//!
//! This demonstrates PIMC using Wick's rotation to compute the ground
//! state energy of a 1D quantum harmonic oscillator with parallel paths.

use rust_qmc::sampling::run_pimc_harmonic;

fn main() {
    // Simulation parameters
    let n_paths = 100;          // Number of parallel walkers
    let n_beads = 64;           // Trotter slices (M)
    let beta = 20.0;            // Inverse temperature (low T → ground state)
    let omega = 1.0;            // Oscillator frequency (ℏω = 1 in natural units)
    let n_thermalize = 2000;    // Thermalization sweeps
    let n_production = 5000;    // Production sweeps
    let use_staging = true;     // Use staging moves for better sampling

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Path Integral Monte Carlo - Harmonic Oscillator          ║");
    println!("║     Using Wick's Rotation with Parallel Paths & PBC          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    run_pimc_harmonic(
        n_paths,
        n_beads,
        beta,
        omega,
        n_thermalize,
        n_production,
        use_staging,
    );

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  For ground state of harmonic oscillator:                    ║");
    println!("║    E₀ = ½ℏω = 0.5 (in natural units where ℏ=m=ω=1)          ║");
    println!("║    <x²> = ℏ/(2mω) = 0.5                                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
