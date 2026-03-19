//! Benzene (C₆H₆) VMC Simulation
//! Uses 6-31G split-valence Gaussian basis with Jastrow3 correlation.
//! Supports drift-diffusion sampling and SR optimization.
//!
//! Usage:
//!   cargo run --example benzene_vmc --release -- [OPTIONS]
//!
//! Options:
//!   --b-ee <F>       Electron-electron Jastrow decay [default: 1.5]
//!   --b-en <F>       Electron-nucleus Jastrow decay [default: 2.0]
//!   -n, --steps <N>  Number of VMC steps [default: 50000]
//!   -w, --walkers <N> Number of walkers [default: 10]
//!   --drift          Use drift-diffusion sampler with single-electron moves
//!   --optimize       Run SR optimization before production VMC
//!   --tau <F>        Time step for drift-diffusion [default: 0.002]

use clap::Parser;
use rust_qmc::{
    MCMCParams, MCMCSimulation, BenzeneGTO,
    DDVMCParams, DriftDiffusionVMC,
    SROptimizer, OptimizableWfn,
};

/// Benzene (C₆H₆) VMC Simulation
#[derive(Parser, Debug)]
#[command(version, about = "VMC simulation for Benzene (C₆H₆) with 6-31G and Jastrow3")]
struct Args {
    /// Electron-electron Jastrow decay parameter
    #[arg(long, default_value_t = 1.5)]
    b_ee: f64,

    /// Electron-nucleus Jastrow decay parameter
    #[arg(long, default_value_t = 2.0)]
    b_en: f64,

    /// Number of VMC steps
    #[arg(short = 'n', long, default_value_t = 50_000)]
    steps: usize,

    /// Number of walkers
    #[arg(short, long, default_value_t = 10)]
    walkers: usize,

    /// Use drift-diffusion sampler with single-electron moves
    #[arg(long)]
    drift: bool,

    /// Run SR Jastrow optimization before production VMC
    #[arg(long)]
    optimize: bool,

    /// Time step for drift-diffusion (τ)
    #[arg(long, default_value_t = 0.002)]
    tau: f64,
}

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;

fn main() {
    let args = Args::parse();
    
    println!("C₆H₆ Benzene VMC Simulation");
    println!("============================");
    println!();
    println!("Features:");
    println!("  • 6-31G split-valence basis (66 AOs) with analytical derivatives");
    println!("  • 21 doubly-occupied MOs (σ core + σ bond + π orbitals)");
    println!("  • 42 electrons (21 up, 21 down)");
    println!("  • Spin-dependent e-e Jastrow (a=0.5 anti, 0.25 para)");
    println!("  • Electron-nucleus Jastrow with Kato cusp");
    println!("  • 4 optimizable Jastrow parameters");
    println!();
    println!("Initial Jastrow Parameters:");
    println!("  b_ee = {:.2} (electron-electron decay)", args.b_ee);
    println!("  b_en = {:.2} (electron-nucleus decay)", args.b_en);
    println!();
    
    let mut benzene = BenzeneGTO::new(args.b_ee, args.b_en);
    let v_nn = benzene.nuclear_repulsion();
    println!("Nuclear-nuclear repulsion: {:.4} Ha ({:.2} eV)", v_nn, v_nn * HA_TO_EV);
    println!();
    
    // Optionally run SR optimization
    if args.optimize {
        println!("--- Jastrow Optimization via Stochastic Reconfiguration ---\n");
        
        let optimizer = SROptimizer::new()
            .with_n_samples(3000)
            .with_n_walkers(10)
            .with_max_iterations(20)
            .with_learning_rate(0.03)
            .with_sr_epsilon(0.001)
            .with_verbose(true);
        
        let result = optimizer.optimize(&mut benzene);
        
        println!("\n  Optimized params: b_ee={:.4}, b_en={:.4}, a_anti={:.4}, a_para={:.4}",
            result.final_params[0], result.final_params[1],
            result.final_params[2], result.final_params[3]);
        println!("  Optimization energy: {:.5} ± {:.4} Ha\n",
            result.final_energy,
            (result.final_variance / 3000.0).sqrt());
        println!("----------------------------------------------------------\n");
    }
    
    println!("Simulation Parameters:");
    println!("  Walkers:   {}", args.walkers);
    println!("  Steps:     {}", args.steps);
    if args.drift {
        println!("  Sampler:   Drift-diffusion (single-electron moves, analytical derivatives)");
        println!("  Time step: {:.4}", args.tau);
    } else {
        println!("  Sampler:   Random-walk Metropolis (all-electron moves)");
    }
    if args.optimize {
        let params = benzene.get_params();
        println!("  Jastrow:   b_ee={:.3}, b_en={:.3}, a_anti={:.3}, a_para={:.3}",
            params[0], params[1], params[2], params[3]);
    }
    println!();
    
    if args.drift {
        run_drift_diffusion(benzene, &args);
    } else {
        run_random_walk(benzene, &args);
    }
}

fn run_random_walk(benzene: BenzeneGTO, args: &Args) {
    let params = MCMCParams {
        n_walkers: args.walkers,
        n_steps: args.steps,
        initial_step_size: 0.2,
        max_step_size: 1.5,
        min_step_size: 0.02,
        target_acceptance: 0.5,
        adaptation_interval: 100,
    };
    
    println!("Running VMC (random-walk Metropolis)...");
    let mut simulation = MCMCSimulation::new(benzene, params);
    let results = simulation.run();
    
    print_results(results.energy, results.error, results.autocorrelation_time, None, None);
}

fn run_drift_diffusion(benzene: BenzeneGTO, args: &Args) {
    let params = DDVMCParams {
        n_walkers: args.walkers,
        n_steps: args.steps,
        n_burnin: 3000,
        time_step: args.tau,
        max_drift: 1.0,
        target_acceptance: 0.5,
        adaptation_interval: 100,
    };
    
    println!("Running VMC (drift-diffusion, single-electron moves)...");
    let mut simulation = DriftDiffusionVMC::new(benzene, params);
    let results = simulation.run();
    
    print_results(
        results.energy, results.error, results.autocorrelation_time,
        Some(results.acceptance_rate), Some(results.final_time_step),
    );
}

fn print_results(
    energy: f64, error: f64, autocorrelation_time: f64,
    acceptance_rate: Option<f64>, final_time_step: Option<f64>,
) {
    println!();
    println!("Results:");
    println!("--------");
    println!("Total energy:        {:.6} ± {:.6} Ha", energy, error);
    println!("Total energy:        {:.4} ± {:.4} eV", 
             energy * HA_TO_EV, error * HA_TO_EV);
    println!();
    if let Some(rate) = acceptance_rate {
        println!("Acceptance rate:     {:.1}% per electron", rate * 100.0);
    }
    if let Some(tau) = final_time_step {
        println!("Final time step:     {:.6}", tau);
    }
    println!("Autocorrelation:     {:.2} steps", autocorrelation_time);
    println!();
    println!("Reference Values:");
    println!("  HF/6-31G:    ~-230.6 Ha");
    println!("  Experiment:  ~-232.2 Ha (atomization energy corrected)");
    println!();
    
    // Energy quality assessment
    if energy < -180.0 && energy > -280.0 {
        println!("✓ Energy is in reasonable range for C₆H₆");
        let diff = (energy + 230.0).abs();
        if diff < 5.0 {
            println!("✓ Within 5 Ha of HF/6-31G value - excellent!");
        } else if diff < 15.0 {
            println!("✓ Within 15 Ha of HF/6-31G value - good accuracy!");
        } else if diff < 30.0 {
            println!("○ Within 30 Ha of HF - reasonable, try --optimize");
        } else {
            println!("△ More than 30 Ha from HF - consider --optimize");
        }
    } else {
        println!("⚠ Energy outside expected range - check wavefunction");
    }
}
