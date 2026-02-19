//! CH4 Stochastic Reconfiguration Optimization Example
//!
//! Demonstrates how to optimize Jastrow parameters using the Stochastic
//! Reconfiguration (SR) method, which uses natural gradient descent.
//!
//! Usage:
//!   cargo run --example ch4_sr_optimize --release

use rust_qmc::{MethaneGTO, SROptimizer, OptimizableWfn};

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;

fn main() {
    println!("CH4 Jastrow Optimization via Stochastic Reconfiguration");
    println!("=======================================================\n");
    
    // Initial Jastrow parameters (deliberately suboptimal)
    let b_ee_init = 1.0;
    let b_en_init = 2.0;
    
    println!("Initial Jastrow parameters:");
    println!("  b_ee = {:.4}", b_ee_init);
    println!("  b_en = {:.4}", b_en_init);
    println!();
    
    // Create wavefunction
    let mut wfn = MethaneGTO::new(b_ee_init, b_en_init);
    
    // Configure SR optimizer
    let optimizer = SROptimizer::new()
        .with_n_samples(5000)
        .with_n_walkers(20)
        .with_max_iterations(30)
        .with_learning_rate(0.05)
        .with_sr_epsilon(0.001)
        .with_verbose(true);
    
    println!("--- SR Optimization ---\n");
    let result = optimizer.optimize(&mut wfn);
    println!("\n-----------------------\n");
    
    // Summary
    println!("Optimization Summary:");
    println!("---------------------");
    println!("  Iterations:      {}", result.energy_history.len());
    println!();
    println!("  Initial params:  b_ee={:.4}, b_en={:.4}",
        result.param_history[0][0], result.param_history[0][1]);
    println!("  Final params:    b_ee={:.4}, b_en={:.4}",
        result.final_params[0], result.final_params[1]);
    println!();
    println!("  Final energy:    {:.5} Ha ({:.2} eV)", 
        result.final_energy, result.final_energy * HA_TO_EV);
    println!("  Final variance:  {:.4} Ha²", result.final_variance);
    println!("  Final error:     {:.4} Ha",
        (result.final_variance / 5000.0).sqrt());
    println!();
    
    // Energy improvement
    if !result.energy_history.is_empty() {
        let initial_energy = result.energy_history[0];
        let energy_change = initial_energy - result.final_energy;
        println!("  Energy change:   {:.4} Ha ({:.2} eV)",
            energy_change, energy_change * HA_TO_EV);
    }
    
    println!();
    println!("Reference Values:");
    println!("  HF/STO-3G:  ~-39.7 Ha");
    println!("  Experiment: ~-40.5 Ha");
}
