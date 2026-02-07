//! CH4 (Methane) VMC Simulation - Improved Version with STO-6G and Jastrow3
//!
//! Uses Gaussian basis and electron-nucleus cusp for better accuracy.
//!
//! Usage:
//!   cargo run --example ch4_gto_vmc --release -- [OPTIONS]
//!
//! Options:
//!   --b-ee <F>     Electron-electron Jastrow decay [default: 1.5]
//!   --b-en <F>     Electron-nucleus Jastrow decay [default: 2.0]
//!   -n, --steps <N>   Number of VMC steps [default: 100000]
//!   -w, --walkers <N> Number of walkers [default: 10]

use clap::Parser;
use rust_qmc::{MCMCParams, MCMCSimulation, MethaneGTO};

/// Methane (CH4) VMC Simulation - Improved Version
#[derive(Parser, Debug)]
#[command(version, about = "VMC simulation for Methane (CH4) with STO-6G and Jastrow3")]
struct Args {
    /// Electron-electron Jastrow decay parameter
    #[arg(long, default_value_t = 1.5)]
    b_ee: f64,

    /// Electron-nucleus Jastrow decay parameter
    #[arg(long, default_value_t = 2.0)]
    b_en: f64,

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
    
    println!("CH4 Methane VMC - Improved Wavefunction");
    println!("========================================");
    println!();
    println!("Improvements over basic version:");
    println!("  • STO-6G Gaussian basis (vs simple STO)");
    println!("  • Spin-dependent e-e Jastrow (a=0.5 anti, 0.25 para)");
    println!("  • Electron-nucleus Jastrow with Kato cusp");
    println!();
    println!("Jastrow Parameters:");
    println!("  b_ee = {:.2} (electron-electron decay)", args.b_ee);
    println!("  b_en = {:.2} (electron-nucleus decay)", args.b_en);
    println!();
    println!("Simulation Parameters:");
    println!("  Walkers:   {}", args.walkers);
    println!("  Steps:     {}", args.steps);
    println!();
    
    let ch4 = MethaneGTO::new(args.b_ee, args.b_en);
    let v_nn = ch4.nuclear_repulsion();
    println!("Nuclear-nuclear repulsion: {:.4} Ha ({:.2} eV)", v_nn, v_nn * HA_TO_EV);
    println!();
    
    // MCMC parameters
    let params = MCMCParams {
        n_walkers: args.walkers,
        n_steps: args.steps,
        initial_step_size: 0.3,
        max_step_size: 2.0,
        min_step_size: 0.05,
        target_acceptance: 0.5,
        adaptation_interval: 100,
    };
    
    println!("Running VMC simulation...");
    let mut simulation = MCMCSimulation::new(ch4, params);
    let results = simulation.run();
    
    println!();
    println!("Results:");
    println!("--------");
    println!("Total energy:        {:.6} ± {:.6} Ha", results.energy, results.error);
    println!("Total energy:        {:.4} ± {:.4} eV", 
             results.energy * HA_TO_EV,
             results.error * HA_TO_EV);
    println!();
    println!("Autocorrelation:     {:.2} steps", results.autocorrelation_time);
    println!();
    println!("Reference Values:");
    println!("  HF/STO-3G:   ~-39.7 Ha");
    println!("  Experiment:  ~-40.5 Ha (equilibrium)");
    println!();
    
    // Energy quality assessment
    if results.energy < -35.0 && results.energy > -50.0 {
        println!("✓ Energy is in reasonable range for CH4");
        let diff = (results.energy + 40.0).abs();
        if diff < 2.0 {
            println!("✓ Within 2 Ha of experimental value - good accuracy!");
        } else if diff < 5.0 {
            println!("○ Within 5 Ha of experimental value - reasonable accuracy");
        } else {
            println!("△ More than 5 Ha from experimental - consider optimizing Jastrow parameters");
        }
    } else {
        println!("⚠ Energy outside expected range - check wavefunction");
    }
}
