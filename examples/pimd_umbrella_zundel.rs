//! PIMD + Umbrella Sampling -- Zundel Cation Proton Transfer PMF
//!
//! Run with: cargo run --release --example pimd_umbrella_zundel
//!
//! Computes the potential of mean force (PMF) for proton transfer in the
//! Zundel cation H5O2+ using path integral molecular dynamics combined
//! with umbrella sampling and WHAM.
//!
//! The PMF W(δ) is computed along the proton transfer coordinate:
//!   δ = d(O1-H*) - d(O2-H*)
//!
//! Negative δ: proton closer to O1 (donor)
//! Positive δ: proton closer to O2 (acceptor)
//! δ = 0: transition state (proton shared equally)
//!
//! The simulation:
//!   1. Runs umbrella sampling with harmonic bias in ~16 windows
//!   2. Collects histograms of the centroid transfer coordinate
//!   3. Uses WHAM to reconstruct the unbiased PMF
//!   4. Compares classical (P=1) and quantum (P=32) PMFs
//!
//! Key physics:
//!   - Classical PMF has a higher barrier (thermal activation only)
//!   - Quantum PMF has a lower effective barrier (tunneling + ZPE)
//!   - The difference quantifies the quantum contribution to proton transfer

use rust_qmc::sampling::run_zundel_umbrella_sampling;

fn main() {
    // --- Physical parameters -----------------------------------------
    let temperature_k = 300.0;
    let beta = 315774.65 / temperature_k;  // Inverse temperature in a.u.

    // --- Simulation parameters ---------------------------------------
    let n_beads = 32;           // Ring polymer beads (Trotter number)
    let n_polymers = 10;        // Parallel replicas per window
    let dt = 0.3;               // Time step in a.u.
    let n_equilibrate = 10_000; // Equilibration steps per window
    let n_production = 30_000;  // Production steps per window

    // --- Umbrella sampling parameters --------------------------------
    // Window centers spanning the proton transfer pathway
    // δ ∈ [-1.5, 1.5] Bohr with spacing ~0.2 Bohr
    let window_centers: Vec<f64> = (-7..=7)
        .map(|i| i as f64 * 0.2)
        .collect();

    // Bias spring constant: κ = 0.08 Ha/Bohr² (~50 kcal/mol/Bohr²)
    // Strong enough to keep sampling in each window, but not too stiff
    let bias_spring_constant = 0.08;

    println!("================================================================");
    println!("|  PIMD + Umbrella Sampling for Proton Transfer PMF            |");
    println!("|                                                              |");
    println!("|       H   H              H   H                              |");
    println!("|        \\ /                \\ /                               |");
    println!("|    O - H+... O    <->    O ...H+- O                          |");
    println!("|        / \\                / \\                               |");
    println!("|       H   H              H   H                              |");
    println!("|                                                              |");
    println!("|   Zundel Cation H5O2+ -- Grotthuss Mechanism                |");
    println!("|                                                              |");
    println!("|   Method: Umbrella Sampling + WHAM                           |");
    println!("|   PMF along δ = d(O1-H*) - d(O2-H*)                        |");
    println!("================================================================");
    println!();
    println!("Parameters:");
    println!("  Temperature:     {:.1} K", temperature_k);
    println!("  Ring polymer:    P = {} beads", n_beads);
    println!("  Replicas/window: {}", n_polymers);
    println!("  Time step:       {:.4} a.u. ({:.4} fs)", dt, dt * 0.02419);
    println!("  Windows:         {} from δ = {:.2} to {:.2} Bohr",
             window_centers.len(),
             window_centers.first().unwrap(),
             window_centers.last().unwrap());
    println!("  Bias κ:          {:.4} Ha/Bohr² ({:.1} kcal/mol/Bohr²)",
             bias_spring_constant, bias_spring_constant * 627.509);
    println!("  Equil/window:    {} steps ({:.1} fs)",
             n_equilibrate, n_equilibrate as f64 * dt * 0.02419);
    println!("  Prod/window:     {} steps ({:.1} fs)",
             n_production, n_production as f64 * dt * 0.02419);
    println!("  Total PIMD steps: ~{:.0}k (classical + quantum)",
             2.0 * window_centers.len() as f64 * (n_equilibrate + n_production) as f64 / 1000.0);
    println!();

    run_zundel_umbrella_sampling(
        n_polymers,
        n_beads,
        beta,
        dt,
        n_equilibrate,
        n_production,
        &window_centers,
        bias_spring_constant,
    );

    println!();
    println!("================================================================");
    println!("|  Output files:                                                |");
    println!("|    pimd_umbrella_pmf.txt         -- WHAM free energy (PMF)    |");
    println!("|    pimd_umbrella_histograms.txt  -- per-window distributions  |");
    println!("|    pimd_umbrella_convergence.txt -- WHAM convergence info     |");
    println!("================================================================");
}
