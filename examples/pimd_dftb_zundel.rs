//! Zundel Cation H5O2+ with Toy DFTB: Comparing PIMD Acceleration Methods
//!
//! Runs PIMD on the Zundel cation with a tight-binding electronic structure
//! potential, comparing four thermostatting/factorization strategies:
//!
//! 1. Standard PILE thermostat (baseline)
//! 2. Takahashi-Imada (TI) factorization
//! 3. PIQTB quantum thermostat
//! 4. TI + PIQTB (combined)
//!
//! The DFTB potential (13x13 Hamiltonian diagonalization per geometry) is
//! more expensive than the EVB model, making this a realistic test case
//! for acceleration methods.
//!
//! Usage: cargo run --release --example pimd_dftb_zundel

use rust_qmc::sampling::{
    ToyDFTB, MolecularPotential, MolecularPIMD, NormalModeTransform,
    MolecularPIQTB,
};

/// Results from a simulation run
struct SimResult {
    label: String,
    mean_energy: f64,
    std_energy: f64,
    mean_tc: f64,
    mean_rg_h: f64,
    tunnel_frac: f64,
}

fn run_simulation(
    label: &str,
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    gamma: f64,
    use_ti: bool,
    use_piqtb: bool,
    n_equil: usize,
    n_prod: usize,
    sample_every: usize,
) -> SimResult {
    let pes = ToyDFTB::new();
    let donor = 0_usize;
    let proton = 3_usize;
    let acceptor = 4_usize;

    // Create simulation
    let mut sim = if use_ti {
        MolecularPIMD::new_with_ti(n_polymers, n_beads, beta, dt, gamma, pes.clone())
    } else {
        MolecularPIMD::new(n_polymers, n_beads, beta, dt, gamma, pes.clone())
    };

    // Build PIQTB thermostat if requested
    let nm = NormalModeTransform::new(n_beads, beta);
    let piqtb = if use_piqtb {
        Some(MolecularPIQTB::new(
            n_beads, beta, dt, pes.masses(), gamma, &nm.frequencies,
        ))
    } else {
        None
    };

    // Equilibrate
    print!("  {} equilibrating...", label);
    for step in 0..n_equil {
        if let Some(ref qtb) = piqtb {
            sim.step_obabo_piqtb(qtb);
        } else {
            sim.step_obabo();
        }
        if step == n_equil / 2 {
            print!(" 50%...");
        }
    }
    println!(" done");

    // Production
    print!("  {} sampling...", label);
    let mut energies = Vec::new();
    let mut tcs = Vec::new();
    let mut rgs = Vec::new();
    let mut tunnel_sum = 0.0;
    let mut tunnel_n = 0;

    for step in 0..n_prod {
        if let Some(ref qtb) = piqtb {
            sim.step_obabo_piqtb(qtb);
        } else {
            sim.step_obabo();
        }

        if step % sample_every == 0 {
            energies.push(sim.average_virial_energy());
            tcs.push(sim.average_transfer_coordinate(donor, proton, acceptor));
            rgs.push(sim.average_atom_rg(proton));
            tunnel_sum += sim.tunneling_fraction(donor, proton, acceptor);
            tunnel_n += 1;
        }

        if step == n_prod / 2 {
            print!(" 50%...");
        }
    }
    println!(" done ({} samples)", energies.len());

    let n = energies.len() as f64;
    let mean_e = energies.iter().sum::<f64>() / n;
    let var_e = energies.iter().map(|&e| (e - mean_e).powi(2)).sum::<f64>() / n;
    let std_e = var_e.sqrt();
    let mean_tc = tcs.iter().sum::<f64>() / n;
    let mean_rg = rgs.iter().sum::<f64>() / n;
    let tunnel = tunnel_sum / tunnel_n as f64;

    SimResult {
        label: label.to_string(),
        mean_energy: mean_e,
        std_energy: std_e,
        mean_tc,
        mean_rg_h: mean_rg,
        tunnel_frac: tunnel,
    }
}

fn main() {
    println!("================================================================");
    println!("  Zundel H5O2+ PIMD with Toy DFTB");
    println!("  Comparing: PILE / TI / PIQTB / TI+PIQTB");
    println!("================================================================");
    println!();

    let pes = ToyDFTB::new();
    let geom = pes.reference_geometry();
    let e_ref = pes.energy(&geom);
    println!("  Reference energy: {:.6} Ha", e_ref);

    // Check barrier
    let mut geom_ts = geom.clone();
    geom_ts[9] = (geom[0] + geom[12]) / 2.0; // H* at midpoint O1-O2
    let e_ts = pes.energy(&geom_ts);
    println!("  TS energy:        {:.6} Ha", e_ts);
    println!("  Barrier:          {:.6} Ha ({:.1} kcal/mol)",
             e_ts - e_ref, (e_ts - e_ref) * 627.509);
    println!();

    // Simulation parameters
    let n_polymers = 4;
    let n_beads = 16;
    let beta = 1000.0;  // ~315 K
    let dt = 1.0;       // a.u.
    let gamma = 0.001;
    let n_equil = 2000;
    let n_prod = 5000;
    let sample_every = 10;

    let temp_k = 315774.65 / beta;
    println!("  Parameters:");
    println!("    Beads P = {}", n_beads);
    println!("    Replicas = {}", n_polymers);
    println!("    Temperature = {:.1} K (β = {:.1} a.u.)", temp_k, beta);
    println!("    Time step = {:.1} a.u.", dt);
    println!("    Equilibration = {} steps", n_equil);
    println!("    Production = {} steps ({} samples)",
             n_prod, n_prod / sample_every);
    println!();

    // Run all four methods
    println!("--- Running simulations ---");
    println!();

    let results: Vec<SimResult> = vec![
        run_simulation("PILE (standard)", n_polymers, n_beads, beta, dt, gamma,
                       false, false, n_equil, n_prod, sample_every),
        run_simulation("TI             ", n_polymers, n_beads, beta, dt, gamma,
                       true,  false, n_equil, n_prod, sample_every),
        run_simulation("PIQTB          ", n_polymers, n_beads, beta, dt, gamma,
                       false, true,  n_equil, n_prod, sample_every),
        run_simulation("TI + PIQTB     ", n_polymers, n_beads, beta, dt, gamma,
                       true,  true,  n_equil, n_prod, sample_every),
    ];

    // Print comparison table
    println!();
    println!("================================================================");
    println!("  Results Comparison (P = {}, T = {:.1} K)", n_beads, temp_k);
    println!("================================================================");
    println!();
    println!("  {:18} | {:>12} {:>8} | {:>8} | {:>8} | {:>7}",
             "Method", "⟨E⟩ (Ha)", "σ(E)", "⟨d⟩", "R_g(H*)", "Tunnel");
    println!("  {:->18}-+-{:->12}-{:->8}-+-{:->8}-+-{:->8}-+-{:->7}",
             "", "", "", "", "", "");

    for r in &results {
        println!("  {:18} | {:>12.6} {:>8.6} | {:>8.4} | {:>8.5} | {:>6.2}%",
                 r.label, r.mean_energy, r.std_energy,
                 r.mean_tc, r.mean_rg_h, 100.0 * r.tunnel_frac);
    }

    println!();

    // Bead convergence test: compare P=8 vs P=16 for each method
    println!("================================================================");
    println!("  Bead Convergence: P = 8 vs P = 16");
    println!("================================================================");
    println!();

    let n_beads_small = 8;
    let results_small: Vec<SimResult> = vec![
        run_simulation("PILE  P=8      ", n_polymers, n_beads_small, beta, dt, gamma,
                       false, false, n_equil, n_prod, sample_every),
        run_simulation("TI    P=8      ", n_polymers, n_beads_small, beta, dt, gamma,
                       true,  false, n_equil, n_prod, sample_every),
        run_simulation("PIQTB P=8      ", n_polymers, n_beads_small, beta, dt, gamma,
                       false, true,  n_equil, n_prod, sample_every),
        run_simulation("TI+PIQTB P=8   ", n_polymers, n_beads_small, beta, dt, gamma,
                       true,  true,  n_equil, n_prod, sample_every),
    ];

    println!();
    println!("  {:18} | {:>12} | {:>12} | {:>10}",
             "Method", "E(P=8)", "E(P=16)", "ΔE");
    println!("  {:->18}-+-{:->12}-+-{:->12}-+-{:->10}",
             "", "", "", "");

    for (r8, r16) in results_small.iter().zip(results.iter()) {
        let delta = (r8.mean_energy - r16.mean_energy).abs();
        println!("  {:18} | {:>12.6} | {:>12.6} | {:>10.6}",
                 r16.label, r8.mean_energy, r16.mean_energy, delta);
    }

    println!();
    println!("  Smaller ΔE = better bead convergence = fewer beads needed.");
    println!("  TI and PIQTB each reduce ΔE; TI+PIQTB should be the best.");
    println!();
    println!("================================================================");
    println!("  Done.");
    println!("================================================================");
}
