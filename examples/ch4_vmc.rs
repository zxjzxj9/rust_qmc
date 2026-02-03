//! CH4 Methane VMC Example
//!
//! Run VMC simulation on a methane molecule using Slater-Jastrow wavefunction.
//!
//! Usage:
//!   cargo run --example ch4_vmc --release -- [OPTIONS]
//!
//! Options:
//!   -n, --steps <N>       Number of VMC steps [default: 100000]
//!   -w, --walkers <N>     Number of walkers [default: 10]
//!   -c, --cusp <F>        Jastrow cusp parameter [default: 0.5]

use clap::Parser;
use rust_qmc::{Methane, MCMCParams, MCMCSimulation};

/// Methane (CH4) VMC Simulation
#[derive(Parser, Debug)]
#[command(version, about = "VMC simulation for Methane (CH4) molecule")]
struct Args {
    /// Jastrow cusp parameter
    #[arg(short = 'c', long, default_value_t = 0.5)]
    cusp: f64,

    /// Number of VMC steps
    #[arg(short = 'n', long, default_value_t = 100_000)]
    steps: usize,

    /// Number of walkers
    #[arg(short, long, default_value_t = 10)]
    walkers: usize,
}

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;

fn main() {
    let args = Args::parse();

    println!("CH4 Methane VMC Simulation");
    println!("==========================\n");
    
    // Create the methane wavefunction
    let wavefunction = Methane::new(args.cusp);
    let v_nn = wavefunction.nuclear_repulsion();
    
    println!("Molecular Geometry:");
    println!("  Carbon:    (0, 0, 0)");
    println!("  H atoms:   tetrahedral arrangement");
    println!("  C-H bond:  2.05 Bohr (~1.085 Å)");
    println!();
    
    println!("Wavefunction:");
    println!("  Type:      Slater-Jastrow (5↑ × 5↓ × J)");
    println!("  Electrons: 10 (6 from C, 4 from H)");
    println!("  MOs:       5 doubly-occupied");
    println!("  Jastrow:   cusp = {:.2}", args.cusp);
    println!();
    
    println!("Simulation Parameters:");
    println!("  Walkers:   {}", args.walkers);
    println!("  Steps:     {}", args.steps);
    println!();
    
    println!("Nuclear-nuclear repulsion: {:.4} Ha ({:.2} eV)", v_nn, v_nn * HA_TO_EV);
    println!();

    // MCMC parameters
    let params = MCMCParams {
        n_walkers: args.walkers,
        n_steps: args.steps,
        initial_step_size: 0.5,
        max_step_size: 2.0,
        min_step_size: 0.05,
        target_acceptance: 0.5,
        adaptation_interval: 100,
    };

    // Run VMC simulation
    println!("Running VMC simulation...");
    let mut simulation = MCMCSimulation::new(wavefunction, params);
    let results = simulation.run();

    // Print results
    println!("\nResults:");
    println!("--------");
    println!("Total energy:        {:.6} ± {:.6} Ha", results.energy, results.error);
    println!("Total energy:        {:.4} ± {:.4} eV", 
        results.energy * HA_TO_EV, results.error * HA_TO_EV);
    println!();
    println!("Autocorrelation:     {:.2} steps", results.autocorrelation_time);
    println!();
    
    // Reference values
    println!("Reference Values:");
    println!("  HF/STO-3G:  ~-39.7 Ha");
    println!("  Experiment: ~-40.5 Ha (at equilibrium)");
}
