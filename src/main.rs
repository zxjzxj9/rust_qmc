mod conf;
mod dmc;
mod h2_mol;
mod jastrow;
mod lattice;
mod mcmc;
mod sto;
mod tests;
mod wfn;

use clap::Parser;
use nalgebra::Vector3;

use crate::jastrow::Jastrow2;
use crate::mcmc::{MCMCParams, MCMCSimulation};
use crate::sto::{init_li_sto, Lithium, STOSlaterDet};

/// Quantum Monte Carlo simulation program.
#[derive(Parser, Debug)]
#[command(version, about = "Quantum Monte Carlo simulation", long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config.yml")]
    config: String,
    
    /// Number of MCMC walkers
    #[arg(short = 'w', long, default_value_t = 10)]
    walkers: usize,
    
    /// Number of MCMC steps
    #[arg(short = 'n', long, default_value_t = 20_000_000)]
    steps: usize,
}

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;

fn main() {
    let args = Args::parse();
    
    // Build Lithium atom wavefunction
    // Three electrons: 1s↑, 1s↓, 2s↑ (all centered at origin)
    let origin = Vector3::zeros();
    let orbitals = vec![
        init_li_sto(origin, 1, 0, 0),  // 1s
        init_li_sto(origin, 1, 0, 0),  // 1s  
        init_li_sto(origin, 2, 0, 0),  // 2s
    ];
    let spins = vec![1, -1, 1];  // ↑↓↑
    
    let slater_det = STOSlaterDet::new(orbitals, spins);
    let jastrow = Jastrow2 {
        cusp_param: 1.0,
        num_electrons: 3,
    };
    let li_atom = Lithium::new(slater_det, jastrow);

    // MCMC parameters
    let params = MCMCParams {
        n_walkers: args.walkers,
        n_steps: args.steps,
        initial_step_size: 1.0,
        max_step_size: 2.0,
        min_step_size: 0.2,
        target_acceptance: 0.5,
        adaptation_interval: 100,
    };

    // Run simulation
    println!("Starting VMC simulation for Lithium atom");
    println!("=========================================");
    println!("Walkers: {}", params.n_walkers);
    println!("Steps: {}", params.n_steps);
    println!();
    
    let mut simulation = MCMCSimulation::new(li_atom, params);
    let results = simulation.run();

    // Print results
    println!();
    println!("Results");
    println!("-------");
    println!("Energy: {:.6} ± {:.6} Ha", results.energy, results.error);
    println!("Energy: {:.4} ± {:.4} eV", results.energy * HA_TO_EV, results.error * HA_TO_EV);
    println!("Autocorrelation time: {:.2} steps", results.autocorrelation_time);
}