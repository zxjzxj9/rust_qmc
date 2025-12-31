//! Lithium FCC Crystal VMC Example
//!
//! Run VMC simulation on a Lithium FCC crystal using Slater-Jastrow wavefunction.
//!
//! Usage:
//!   cargo run --example lithium_fcc_vmc --release -- [OPTIONS]
//!
//! Options:
//!   -a, --lattice-constant <BOHR>  Lattice constant in Bohr [default: 8.0]
//!   -s, --supercell <N>            Supercell size (N×N×N) [default: 1]
//!   -e, --electrons <N>            Electrons per atom (1 or 3) [default: 1]
//!   -n, --steps <N>                Number of VMC steps [default: 100000]
//!   -w, --walkers <N>              Number of walkers [default: 10]

use clap::Parser;
use rust_qmc::{LithiumFCC, MCMCParams, MCMCSimulation};

/// Lithium FCC Crystal VMC Simulation
#[derive(Parser, Debug)]
#[command(version, about = "VMC simulation for Lithium FCC crystal")]
struct Args {
    /// Lattice constant in Bohr
    #[arg(short = 'a', long, default_value_t = 8.0)]
    lattice_constant: f64,

    /// Supercell size (N×N×N)
    #[arg(short, long, default_value_t = 1)]
    supercell: usize,

    /// Electrons per Li atom (1 for pseudopotential, 3 for all-electron)
    #[arg(short, long, default_value_t = 1)]
    electrons: usize,

    /// Jastrow cusp parameter
    #[arg(short = 'c', long, default_value_t = 1.0)]
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

    // Create the wavefunction
    let wavefunction = if args.supercell > 1 {
        println!("Creating Li FCC crystal with {}×{}×{} supercell...",
            args.supercell, args.supercell, args.supercell);
        LithiumFCC::new_supercell(
            args.lattice_constant,
            args.supercell,
            args.electrons,
            args.cusp,
        )
    } else {
        println!("Creating Li FCC crystal (primitive cell)...");
        LithiumFCC::new(args.lattice_constant, args.electrons, args.cusp)
    };

    let num_atoms = wavefunction.ion_positions.len();
    let num_electrons = wavefunction.num_electrons;

    println!("\nSimulation Parameters:");
    println!("======================");
    println!("Lattice constant:    {:.2} Bohr ({:.2} Å)", 
        args.lattice_constant, args.lattice_constant * 0.529177);
    println!("Number of atoms:     {}", num_atoms);
    println!("Electrons per atom:  {}", args.electrons);
    println!("Total electrons:     {}", num_electrons);
    println!("Jastrow cusp param:  {:.2}", args.cusp);
    println!("VMC walkers:         {}", args.walkers);
    println!("VMC steps:           {}", args.steps);
    println!();

    // MCMC parameters
    let params = MCMCParams {
        n_walkers: args.walkers,
        n_steps: args.steps,
        initial_step_size: 1.0,
        max_step_size: 3.0,
        min_step_size: 0.1,
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
    
    if num_atoms > 0 {
        let energy_per_atom = results.energy / num_atoms as f64;
        let error_per_atom = results.error / num_atoms as f64;
        println!("Energy per atom:     {:.6} ± {:.6} Ha", energy_per_atom, error_per_atom);
        println!("Energy per atom:     {:.4} ± {:.4} eV", 
            energy_per_atom * HA_TO_EV, error_per_atom * HA_TO_EV);
    }
    
    println!("Autocorrelation:     {:.2} steps", results.autocorrelation_time);
}
