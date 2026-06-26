//! PIMD + Well-Tempered Metadynamics for the Zundel Cation
//!
//! Run with: cargo run --release --example pimd_metadynamics_zundel
//!
//! Demonstrates well-tempered metadynamics combined with PIMD to compute
//! the free energy surface (FES) for proton transfer in H₅O₂⁺.
//!
//! Three-way comparison:
//!   1. Classical metadynamics (P=1) — no quantum effects
//!   2. Quantum PIMD metadynamics (P=32) — nuclear quantum effects
//!   3. Quantum+TI PIMD metadynamics (P=32) — fourth-order correction
//!
//! Well-tempered metadynamics deposits Gaussian hills along the CV
//! (proton transfer coordinate δ) with heights that decay as:
//!   w_k = w₀ × exp(-V_bias(s_k) / (kT × (γ - 1)))
//!
//! The FES is recovered: W(s) = -(γ/(γ-1)) × V_bias(s)
//!
//! References:
//!   - Laio & Parrinello, PNAS 99, 12562 (2002)
//!   - Barducci, Bussi, Parrinello, PRL 100, 020603 (2008)

use rust_qmc::sampling::run_zundel_metadynamics;

fn main() {
    // --- Physical parameters ---
    let temperature_k = 300.0;
    let beta = 315774.65 / temperature_k;

    // --- Simulation parameters ---
    let n_beads = 32;
    let n_polymers = 4;            // Fewer replicas than umbrella (no windows)
    let dt = 0.3;
    let n_equilibrate = 5_000;
    let n_production = 100_000;    // Longer single run (no window restarts)

    // --- Metadynamics parameters ---
    let initial_height = 0.0005;   // w₀ = 0.5 mHa ≈ 0.3 kcal/mol
    let sigma = 0.15;              // σ = 0.15 Bohr
    let bias_factor = 15.0;        // γ = 15 (well-tempered)
    let deposit_stride = 50;       // Deposit hill every 50 steps

    println!("================================================================");
    println!("|  PIMD + Well-Tempered Metadynamics                           |");
    println!("|                                                              |");
    println!("|       H   H              H   H                              |");
    println!("|        \\ /                \\ /                               |");
    println!("|    O - H+... O    <->    O ...H+- O                          |");
    println!("|        / \\                / \\                               |");
    println!("|       H   H              H   H                              |");
    println!("|                                                              |");
    println!("|   Zundel Cation H5O2+ — Proton Transfer FES                 |");
    println!("|                                                              |");
    println!("|   Unlike umbrella sampling (multiple biased windows + WHAM), |");
    println!("|   metadynamics uses a SINGLE adaptive simulation that        |");
    println!("|   progressively fills free energy wells with Gaussians.      |");
    println!("================================================================");
    println!();
    println!("Parameters:");
    println!("  Temperature:     {:.1} K", temperature_k);
    println!("  Ring polymer:    P = {} beads", n_beads);
    println!("  Replicas:        {}", n_polymers);
    println!("  Time step:       {:.4} a.u. ({:.4} fs)", dt, dt * 0.02419);
    println!("  Equilibration:   {} steps ({:.1} fs)",
             n_equilibrate, n_equilibrate as f64 * dt * 0.02419);
    println!("  Production:      {} steps ({:.1} fs)",
             n_production, n_production as f64 * dt * 0.02419);
    println!();
    println!("  Hill height w₀:  {:.4} mHa ({:.3} kcal/mol)",
             initial_height * 1000.0, initial_height * 627.509);
    println!("  Hill width σ:    {:.3} Bohr", sigma);
    println!("  Bias factor γ:   {:.1}", bias_factor);
    println!("  Deposit stride:  every {} steps", deposit_stride);
    println!("  Expected hills:  ~{}", n_production / deposit_stride);
    println!();

    run_zundel_metadynamics(
        n_polymers,
        n_beads,
        beta,
        dt,
        n_equilibrate,
        n_production,
        initial_height,
        sigma,
        bias_factor,
        deposit_stride,
    );

    println!("================================================================");
    println!("|  Output files:                                                |");
    println!("|    metadynamics_fes.txt          -- reconstructed FES         |");
    println!("|    metadynamics_hills_*.txt      -- deposited hills history   |");
    println!("|    metadynamics_cv_*.txt         -- CV time series            |");
    println!("================================================================");
}
