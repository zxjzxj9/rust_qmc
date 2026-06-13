//! PIMD + Umbrella Sampling + Takahashi-Imada Fourth-Order Correction
//!
//! Run with: cargo run --release --example pimd_ti_umbrella_zundel
//!
//! Demonstrates the combination of:
//!   1. Path Integral MD (quantum nuclear effects)
//!   2. Umbrella Sampling + WHAM (free energy profiling)
//!   3. Takahashi-Imada correction (O(1/P⁴) bead convergence)
//!
//! Compares three PMFs for the Zundel cation H5O2+ proton transfer:
//!   - Classical (P=1): no quantum effects
//!   - Quantum primitive (P=32): standard Trotter, O(1/P²)
//!   - Quantum+TI (P=32): fourth-order factorization, O(1/P⁴)
//!
//! The TI correction modifies the effective potential on each bead:
//!   V_TI(R) = V(R) + (dτ²/24) Σ_a |F_a(R)|²/m_a
//!
//! This achieves faster convergence with respect to the number of beads,
//! meaning the quantum PMF is more accurate at the same bead count, or
//! equivalently, fewer beads are needed for the same accuracy.
//!
//! References:
//!   - Takahashi & Imada, J. Phys. Soc. Jpn. 53, 3765 (1984)
//!   - Yamamoto, JCP 123, 104101 (2005) -- TI virial estimator
//!   - Torrie & Valleau, J. Comput. Phys. 23, 187 (1977) -- Umbrella sampling
//!   - Kumar et al., J. Comput. Chem. 13, 1011 (1992) -- WHAM

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
    // δ ∈ [-1.4, 1.4] Bohr with spacing ~0.2 Bohr
    let window_centers: Vec<f64> = (-7..=7)
        .map(|i| i as f64 * 0.2)
        .collect();

    // Bias spring constant: κ = 0.08 Ha/Bohr² (~50 kcal/mol/Bohr²)
    let bias_spring_constant = 0.08;

    println!("================================================================");
    println!("|  PIMD + Umbrella Sampling + TI Fourth-Order Correction       |");
    println!("|                                                              |");
    println!("|       H   H              H   H                              |");
    println!("|        \\ /                \\ /                               |");
    println!("|    O - H+... O    <->    O ...H+- O                          |");
    println!("|        / \\                / \\                               |");
    println!("|       H   H              H   H                              |");
    println!("|                                                              |");
    println!("|   Zundel Cation H5O2+ -- Grotthuss Mechanism                |");
    println!("|                                                              |");
    println!("|   Three-way comparison:                                      |");
    println!("|     1. Classical (P=1)                                       |");
    println!("|     2. Quantum primitive (P={:2}, O(1/P²))                   |", n_beads);
    println!("|     3. Quantum+TI (P={:2}, O(1/P⁴))                         |", n_beads);
    println!("|                                                              |");
    println!("|   The TI correction achieves better bead convergence,       |");
    println!("|   yielding more accurate quantum PMFs at the same P.        |");
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
    println!("  Total PIMD steps: ~{:.0}k (classical + quantum + TI)",
             3.0 * window_centers.len() as f64 * (n_equilibrate + n_production) as f64 / 1000.0);
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
    println!("|                                                              |");
    println!("|  The PMF file now contains three columns:                    |");
    println!("|    W_classical, W_quantum (primitive), W_quantum_TI          |");
    println!("================================================================");
}
