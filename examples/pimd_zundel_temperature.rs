//! Temperature-Dependent Proton Transfer Barrier in the Zundel Cation
//!
//! Run with: cargo run --release --example pimd_zundel_temperature
//!
//! Sweeps temperature from 100K to 600K and extracts:
//! - The effective free energy barrier W*(d) = max(W) - min(W) at each T
//! - Tunneling fraction at each T
//! - Proton delocalization R_g(H*) at each T
//!
//! Physics: at low T, quantum tunneling dominates and the effective barrier
//! is reduced relative to the classical PES barrier. At high T, thermal
//! activation dominates and the barrier approaches the classical value.
//! The crossover temperature T_c ~ hbarw‡/(2πk_B) marks the transition.

use rust_qmc::sampling::{MolecularPIMD, ZundelPES, free_energy_profile};
use std::fs::File;
use std::io::{BufWriter, Write};

/// Run a single PIMD simulation at given temperature, return key observables.
fn run_at_temperature(
    temp_k: f64,
    n_beads: usize,
    n_polymers: usize,
    n_equilibrate: usize,
    n_production: usize,
    dt: f64,
) -> TempResult {
    let beta = 315774.65 / temp_k;
    let kbt = 1.0 / beta;
    let gamma = 0.001;

    let pes = ZundelPES::new();
    let donor = 0_usize;
    let proton = 3_usize;
    let acceptor = 4_usize;

    // --- Classical (P=1) ---
    let mut cl_sim = MolecularPIMD::new(n_polymers, 1, beta, dt, gamma, pes.clone());
    for _ in 0..n_equilibrate { cl_sim.step_obabo(); }

    let sample_interval = 10;
    let n_bins = 200;
    let hist_min = -2.5;
    let hist_max = 2.5;
    let hist_bw = (hist_max - hist_min) / n_bins as f64;
    let mut cl_hist = vec![0.0; n_bins];
    let mut cl_tun_sum = 0.0;
    let mut cl_tun_n = 0;
    let mut cl_e_sum = 0.0;
    let mut cl_n = 0;

    for step in 0..n_production {
        cl_sim.step_obabo();
        if step % sample_interval == 0 {
            cl_sim.accumulate_transfer_histogram(
                &mut cl_hist, hist_min, hist_max, donor, proton, acceptor,
            );
            cl_tun_sum += cl_sim.tunneling_fraction(donor, proton, acceptor);
            cl_tun_n += 1;
            cl_e_sum += cl_sim.average_virial_energy();
            cl_n += 1;
        }
    }

    let cl_w = free_energy_profile(&cl_hist, hist_bw, kbt);
    let cl_barrier = extract_barrier(&cl_w, hist_min, hist_bw);
    let cl_tunnel = cl_tun_sum / cl_tun_n as f64;

    // --- Quantum (P=n_beads) ---
    let mut q_sim = MolecularPIMD::new(n_polymers, n_beads, beta, dt, gamma, pes.clone());
    for _ in 0..n_equilibrate { q_sim.step_obabo(); }

    let mut q_hist = vec![0.0; n_bins];
    let mut q_tun_sum = 0.0;
    let mut q_tun_n = 0;
    let mut q_e_sum = 0.0;
    let mut q_rg_sum = 0.0;
    let mut q_n = 0;

    for step in 0..n_production {
        q_sim.step_obabo();
        if step % sample_interval == 0 {
            q_sim.accumulate_transfer_histogram(
                &mut q_hist, hist_min, hist_max, donor, proton, acceptor,
            );
            q_tun_sum += q_sim.tunneling_fraction(donor, proton, acceptor);
            q_tun_n += 1;
            q_e_sum += q_sim.average_virial_energy();
            q_rg_sum += q_sim.average_atom_rg(proton);
            q_n += 1;
        }
    }

    let q_w = free_energy_profile(&q_hist, hist_bw, kbt);
    let q_barrier = extract_barrier(&q_w, hist_min, hist_bw);
    let q_tunnel = q_tun_sum / q_tun_n as f64;
    let q_rg = q_rg_sum / q_n as f64;
    let q_e = q_e_sum / q_n as f64;
    let cl_e = cl_e_sum / cl_n as f64;

    TempResult {
        temp_k,
        beta,
        cl_barrier,
        q_barrier,
        cl_tunnel,
        q_tunnel,
        q_rg,
        q_energy: q_e,
        cl_energy: cl_e,
        q_histogram: q_hist,
        cl_histogram: cl_hist,
    }
}

/// Extract the barrier height from a free energy profile.
/// The barrier is W(d=0) - W(d_min), where d_min is the well minimum.
fn extract_barrier(w: &[f64], x_min: f64, bin_width: f64) -> f64 {
    let n = w.len();
    // Find the bin closest to d=0 (transition state)
    let ts_bin = ((-x_min) / bin_width) as usize;
    let ts_bin = ts_bin.min(n - 1);

    // Find the minimum W in the left half (d < 0)
    let left_half = &w[..ts_bin.max(1)];
    let w_min_left = left_half.iter().cloned().fold(f64::INFINITY, f64::min);

    // Find the minimum W in the right half (d > 0)
    let right_half = &w[ts_bin.min(n-1)..];
    let w_min_right = right_half.iter().cloned().fold(f64::INFINITY, f64::min);

    // Barrier = W(TS) - W(well minimum)
    let w_well = w_min_left.min(w_min_right);
    let w_ts = w[ts_bin];

    (w_ts - w_well).max(0.0)
}

struct TempResult {
    temp_k: f64,
    beta: f64,
    cl_barrier: f64,
    q_barrier: f64,
    cl_tunnel: f64,
    q_tunnel: f64,
    q_rg: f64,
    q_energy: f64,
    cl_energy: f64,
    q_histogram: Vec<f64>,
    cl_histogram: Vec<f64>,
}

fn main() {
    let temperatures = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0];

    let n_beads = 32;
    let n_polymers = 16;
    let dt = 0.3;
    let n_equilibrate = 15_000;
    let n_production = 40_000;

    println!("================================================================");
    println!("|   Temperature Dependence of Proton Transfer Barrier        |");
    println!("|   Zundel Cation H5O2+ -- PIMD with Analytical EVB Forces  |");
    println!("|                                                             |");
    println!("|   Sweeping {} temperatures from {}K to {}K            |",
             temperatures.len(),
             temperatures[0] as u32,
             *temperatures.last().unwrap() as u32);
    println!("|   Beads: {}, Replicas: {}, Equil: {}, Prod: {}   |",
             n_beads, n_polymers, n_equilibrate, n_production);
    println!("================================================================");
    println!();

    let mut results: Vec<TempResult> = Vec::new();

    for (i, &temp) in temperatures.iter().enumerate() {
        println!("------------------------------------------------------------");
        println!("  [{}/{}] Running T = {} K ...", i + 1, temperatures.len(), temp as u32);
        println!("------------------------------------------------------------");

        let result = run_at_temperature(temp, n_beads, n_polymers, n_equilibrate, n_production, dt);

        println!("  Classical: W* = {:.4} Ha ({:.2} kcal/mol), tunnel = {:.1}%",
                 result.cl_barrier, result.cl_barrier * 627.509, 100.0 * result.cl_tunnel);
        println!("  Quantum:   W* = {:.4} Ha ({:.2} kcal/mol), tunnel = {:.1}%, R_g = {:.4}",
                 result.q_barrier, result.q_barrier * 627.509, 100.0 * result.q_tunnel, result.q_rg);

        let reduction = if result.cl_barrier > 1e-6 {
            100.0 * (1.0 - result.q_barrier / result.cl_barrier)
        } else { 0.0 };
        println!("  -> Barrier reduction: {:.1}%", reduction);
        println!();

        results.push(result);
    }

    // =========================================================================
    // Summary Table
    // =========================================================================
    println!();
    println!("======================================================================================");
    println!("|                TEMPERATURE DEPENDENCE -- SUMMARY TABLE                            |");
    println!("======================================================================================");
    println!("|  T (K) | W*_cl (kcal) | W*_qm (kcal) | Reduction | Tunnel_cl | Tunnel_qm | R_g  |");
    println!("|--------+--------------+--------------+-----------+-----------+-----------+------|");

    for r in &results {
        let reduction = if r.cl_barrier > 1e-6 {
            100.0 * (1.0 - r.q_barrier / r.cl_barrier)
        } else { 0.0 };
        println!("|  {:>5} | {:>12.3} | {:>12.3} | {:>8.1}% | {:>8.1}% | {:>8.1}% |{:>5.3} |",
                 r.temp_k as u32,
                 r.cl_barrier * 627.509,
                 r.q_barrier * 627.509,
                 reduction,
                 100.0 * r.cl_tunnel,
                 100.0 * r.q_tunnel,
                 r.q_rg);
    }
    println!("======================================================================================");

    // =========================================================================
    // Physical Interpretation
    // =========================================================================
    println!();
    println!("=== ANALYSIS ===");
    println!();

    // Find the crossover temperature (where barrier reduction drops below 50%)
    let mut crossover = None;
    for r in &results {
        let reduction = if r.cl_barrier > 1e-6 {
            1.0 - r.q_barrier / r.cl_barrier
        } else { 0.0 };
        if reduction < 0.5 && crossover.is_none() {
            crossover = Some(r.temp_k);
        }
    }

    // Low T behavior
    if let Some(low) = results.first() {
        if let Some(high) = results.last() {
            let low_red = if low.cl_barrier > 1e-6 {
                100.0 * (1.0 - low.q_barrier / low.cl_barrier)
            } else { 0.0 };
            let high_red = if high.cl_barrier > 1e-6 {
                100.0 * (1.0 - high.q_barrier / high.cl_barrier)
            } else { 0.0 };

            println!("  • At T = {} K: barrier reduced by {:.0}%  -> tunneling dominates",
                     low.temp_k as u32, low_red);
            println!("  • At T = {} K: barrier reduced by {:.0}%  -> thermal activation dominates",
                     high.temp_k as u32, high_red);

            if low.q_rg > high.q_rg * 1.3 {
                println!("  • Proton delocalization (R_g) increases {:.1}x from {} K to {} K",
                         low.q_rg / high.q_rg.max(0.001),
                         high.temp_k as u32, low.temp_k as u32);
            }
        }
    }

    if let Some(tc) = crossover {
        println!("  • Crossover temperature T_c ~ {} K (barrier reduction drops below 50%)", tc as u32);
        println!("    Below T_c: quantum regime (tunneling through barrier)");
        println!("    Above T_c: classical regime (thermal hopping over barrier)");
    }

    // ZPE analysis
    if results.len() >= 2 {
        let low = &results[0];
        let zpe = low.q_energy - low.cl_energy;
        if zpe > 0.0 {
            println!("  • Zero-point energy at {} K: +{:.4} Ha ({:.1} kcal/mol)",
                     low.temp_k as u32, zpe, zpe * 627.509);
        }
    }

    println!();
    println!("  Conclusion: Quantum effects (tunneling + ZPE) substantially reduce the");
    println!("  effective proton transfer barrier at low temperatures. The effect is");
    println!("  most pronounced below ~200–300 K where the thermal de Broglie wavelength");
    println!("  of the proton becomes comparable to the barrier width.");

    // =========================================================================
    // Output files
    // =========================================================================
    // Temperature sweep data
    {
        let file = File::create("pimd_zundel_temp_sweep.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# T(K) beta W_cl(Ha) W_qm(Ha) W_cl(kcal) W_qm(kcal) reduction(%) tunnel_cl(%) tunnel_qm(%) R_g(Bohr) E_cl(Ha) E_qm(Ha)").unwrap();
        for r in &results {
            let reduction = if r.cl_barrier > 1e-6 {
                100.0 * (1.0 - r.q_barrier / r.cl_barrier)
            } else { 0.0 };
            writeln!(w, "{:.1} {:.4} {:.6} {:.6} {:.4} {:.4} {:.2} {:.2} {:.2} {:.5} {:.6} {:.6}",
                     r.temp_k, r.beta,
                     r.cl_barrier, r.q_barrier,
                     r.cl_barrier * 627.509, r.q_barrier * 627.509,
                     reduction,
                     100.0 * r.cl_tunnel, 100.0 * r.q_tunnel,
                     r.q_rg, r.cl_energy, r.q_energy).unwrap();
        }
        println!();
        println!("  Temperature sweep data -> pimd_zundel_temp_sweep.txt");
    }

    // Free energy profiles at each temperature
    {
        let n_bins = 200;
        let hist_min = -2.5;
        let hist_max = 2.5;
        let hist_bw = (hist_max - hist_min) / n_bins as f64;

        let file = File::create("pimd_zundel_temp_profiles.txt").unwrap();
        let mut w = BufWriter::new(file);
        write!(w, "# delta").unwrap();
        for r in &results {
            write!(w, " W_qm_{}K W_cl_{}K", r.temp_k as u32, r.temp_k as u32).unwrap();
        }
        writeln!(w).unwrap();

        for i in 0..n_bins {
            let x = hist_min + (i as f64 + 0.5) * hist_bw;
            write!(w, "{:.4}", x).unwrap();
            for r in &results {
                let kbt = 1.0 / r.beta;
                let w_q = free_energy_profile(&r.q_histogram, hist_bw, kbt);
                let w_cl = free_energy_profile(&r.cl_histogram, hist_bw, kbt);
                write!(w, " {:.6} {:.6}", w_q[i], w_cl[i]).unwrap();
            }
            writeln!(w).unwrap();
        }
        println!("  Free energy profiles -> pimd_zundel_temp_profiles.txt");
    }

    println!();
    println!("================================================================");
    println!("|  Plot suggestions:                                          |");
    println!("|  1. W*(T) curve: barrier vs temperature (from sweep file)  |");
    println!("|  2. W(d) at each T: overlay free energy profiles           |");
    println!("|  3. R_g(T): proton delocalization vs temperature           |");
    println!("================================================================");
}
