//! CH4 Methane Jastrow Optimization Example
//!
//! Demonstrates how to optimize Jastrow parameters to minimize variance
//! and improve VMC energy estimates.
//!
//! Usage:
//!   cargo run --example ch4_optimize --release

use rust_qmc::{MethaneGTO, JastrowOptimizer, JastrowParams};

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;

fn main() {
    println!("CH4 Methane Jastrow Optimization");
    println!("=================================\n");
    
    // Initial Jastrow parameters (unoptimized guess)
    let b_ee_init = 1.0;
    let b_en_init = 2.0;
    
    println!("Initial Jastrow parameters:");
    println!("  b_ee = {:.4}", b_ee_init);
    println!("  b_en = {:.4}", b_en_init);
    println!();
    
    // Create wavefunction with initial parameters
    let wfn = MethaneGTO::new(b_ee_init, b_en_init);
    
    // Configure optimizer
    let optimizer = JastrowOptimizer::new()
        .with_n_samples(3000)
        .with_learning_rate(0.15)
        .with_max_iterations(30)
        .with_verbose(true);
    
    println!("Optimizer settings:");
    println!("  Samples/iteration: 3000");
    println!("  Learning rate:     0.15");
    println!("  Max iterations:    30");
    println!();
    
    // Run optimization
    println!("--- Optimization Progress ---");
    let result = optimizer.optimize(&wfn);
    println!("-----------------------------\n");
    
    // Summary
    println!("Optimization Results:");
    println!("---------------------");
    println!("Optimized parameters:");
    println!("  b_ee = {:.4}", result.params.b_ee);
    println!("  b_en = {:.4}", result.params.b_en);
    println!();
    println!("Final energy:    {:.4} ± ~{:.4} Ha", 
        result.final_energy, 
        (result.final_variance / 3000.0).sqrt());
    println!("Final energy:    {:.2} eV", result.final_energy * HA_TO_EV);
    println!("Final variance:  {:.4} Ha²", result.final_variance);
    println!();
    
    // Compare with initial
    if !result.energy_history.is_empty() && !result.variance_history.is_empty() {
        let initial_energy = result.energy_history[0];
        let initial_variance = result.variance_history[0];
        let energy_improvement = initial_energy - result.final_energy;
        let variance_reduction = 100.0 * (1.0 - result.final_variance / initial_variance);
        
        println!("Improvement:");
        println!("  Energy change:     {:.4} Ha ({:.2} eV)", 
            energy_improvement, energy_improvement * HA_TO_EV);
        println!("  Variance reduced:  {:.1}%", variance_reduction);
    }
    
    println!();
    println!("Reference Values:");
    println!("  HF/STO-3G:  ~-39.7 Ha");
    println!("  Experiment: ~-40.5 Ha");
}
