//! PIMD + Umbrella Sampling -- Zundel Cation Proton Transfer PMF
//! using the Toy DFTB / ETB Quantum Mechanics Engine!
//!
//! Run with: cargo run --release --example pimd_dftb_zundel

use rust_qmc::sampling::{run_pimd_umbrella_sampling, WHAMSolver, ToyDFTB};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let temperature_k = 300.0;
    let beta = 315774.65 / temperature_k;

    // We use fewer replicas and steps here because DFTB is slightly heavier 
    // than the empirical EVB, but it should still be very fast!
    let n_beads = 16;
    let n_polymers = 5;
    let dt = 0.2; // Slightly smaller dt for stability
    let n_equilibrate = 1000;
    let n_production = 5000;

    // Window centers spanning the proton transfer pathway
    let window_centers: Vec<f64> = (-5..=5)
        .map(|i| i as f64 * 0.2)
        .collect();

    let bias_spring_constant = 0.15; // Slightly stiffer for the toy model
    let n_hist_bins = 100;
    let hist_min = -2.0;
    let hist_max = 2.0;

    let donor = 0;
    let proton = 3;
    let acceptor = 4;

    let pot = ToyDFTB::new();

    println!("================================================================");
    println!("|  PIMD + Umbrella Sampling with Toy DFTB (ETB)                |");
    println!("|  Zundel Cation H5O2+ -- True First Principles!               |");
    println!("================================================================");
    
    // Classical
    let cl_windows = run_pimd_umbrella_sampling(
        pot.clone(), n_polymers, 1, beta, dt, n_equilibrate, n_production,
        &window_centers, bias_spring_constant, donor, proton, acceptor,
        n_hist_bins, hist_min, hist_max, "Classical DFTB"
    );

    // Quantum
    let q_windows = run_pimd_umbrella_sampling(
        pot.clone(), n_polymers, n_beads, beta, dt, n_equilibrate, n_production,
        &window_centers, bias_spring_constant, donor, proton, acceptor,
        n_hist_bins, hist_min, hist_max, "Quantum DFTB"
    );

    let wham = WHAMSolver::new(beta, hist_min, hist_max, n_hist_bins, 1000, 1e-6);

    let (cl_pmf, _cl_fe, iter_c) = wham.solve(&cl_windows);
    let (q_pmf, _q_fe, iter_q) = wham.solve(&q_windows);

    println!("WHAM iterations: Classical={}, Quantum={}", iter_c, iter_q);

    let file = File::create("pimd_dftb_pmf.txt").unwrap();
    let mut w = BufWriter::new(file);
    writeln!(w, "# delta W_cl(Ha) W_q(Ha)").unwrap();
    for i in 0..n_hist_bins {
        writeln!(w, "{:.6} {:.8} {:.8}",
                 wham.bin_centers[i], cl_pmf[i], q_pmf[i]).unwrap();
    }
    
    println!("Results written to pimd_dftb_pmf.txt!");
}
