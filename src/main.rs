mod h2_mol;
mod mcmc;
mod conf;
mod tests;
mod wfn;
mod sto;
mod dmc;

use nalgebra::Vector3;
use h2_mol::{H2MoleculeVB, Slater1s, Jastrow1};
use mcmc::{MCMCParams, MCMCSimulation, MCMCResults};
use clap::Parser;
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config.yml")]
    config: String,
}

const Ha_TO_eV: f64 = 27.21138602;

fn main() {
    // Set up the H2 molecule
    // let h2 = H2MoleculeVB {
    //     H1: Slater1s { alpha: 1.0, R: Vector3::new(0.0, 0.0, 0.7) },
    //     H2: Slater1s { alpha: 1.0, R: Vector3::new(0.0, 0.0, -0.7) },
    //     J: Jastrow1 { F: 1.0 },
    // };

    dmc::run_harmonic_dmc_sampling();
    return;

    // read the config file, with command line argument, use clap mod to input the file name
    let args = Args::parse();
    // let h2 = conf::read_h2molecule_vb(&args.config);
    let h2 = conf::read_h2molecule_mo(&args.config);


    // Set up MCMC parameters
    let params = MCMCParams {
        n_walkers: 10,
        n_steps: 20000000,
        initial_step_size: 1.0,
        max_step_size: 2.0,
        min_step_size: 0.2,
        target_acceptance: 0.5,
        adaptation_interval: 100,
    };

    // Create and run the MCMC simulation
    let mut simulation = MCMCSimulation::new(h2, params);
    let results = simulation.run();

    // Print results
    println!("MCMC Simulation Results for H2 Molecule");
    println!("----------------------------------------");
    println!("Number of walkers: {}", params.n_walkers);
    println!("Number of steps: {}", params.n_steps);
    println!("Final energy: {:.6} ± {:.6} Ha", results.energy, results.error);
    println!("Binding energy: {:.6} ± {:.6} eV", Ha_TO_eV * (results.energy + 1.0), Ha_TO_eV * results.error);
    println!("Autocorrelation time: {:.2} steps", results.autocorrelation_time);

    // Calculate and print average energy for the second half of the simulation
    // let mid_point = results.energies.len() / 2;
    // let avg_energy: f64 = results.energies[mid_point..].iter().sum::<f64>() / (results.energies.len() - mid_point) as f64;
    // println!("Average energy (second half): {:.6} Ha", avg_energy);

    // Optional: Plot energy convergence
    // plot_energy_convergence(&results);
}

fn plot_energy_convergence(results: &MCMCResults) {
    // This is a placeholder for plotting functionality.
    // You would typically use a plotting library like plotters or gnuplot here.
    println!("Plotting energy convergence...");
    println!("X-axis: MCMC steps");
    println!("Y-axis: Energy (Ha)");
    // println!("Data points: {:?}", results.energies);
}