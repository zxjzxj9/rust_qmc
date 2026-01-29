//! Homogeneous Electron Gas (HEG) VMC Example
//!
//! Run VMC simulation on the HEG (jellium) to compute correlation energies
//! for LDA parameterization.
//!
//! Usage:
//!   cargo run --example heg_vmc --release -- [OPTIONS]
//!
//! Options:
//!   -n, --electrons <N>     Number of electrons (use 2, 14, 38, 54, 66) [default: 14]
//!   -r, --rs <RS>           Wigner-Seitz radius in Bohr [default: 4.0]
//!   -j, --jastrow <F>       Jastrow parameter F [default: 1.25*rs]
//!   -s, --steps <N>         Number of VMC steps [default: 100000]
//!   -w, --walkers <N>       Number of walkers [default: 10]
//!   -t, --twists <N>        Monkhorst-Pack grid size for TABC (N×N×N) [default: 1]

use clap::Parser;
use rust_qmc::{HomogeneousElectronGas, MCMCParams, MCMCSimulation};

/// Homogeneous Electron Gas VMC Simulation
#[derive(Parser, Debug)]
#[command(version, about = "VMC simulation for Homogeneous Electron Gas (Jellium)")]
struct Args {
    /// Number of electrons (closed-shell: 2, 14, 38, 54, 66)
    #[arg(short = 'n', long, default_value_t = 14)]
    electrons: usize,

    /// Wigner-Seitz radius in Bohr (controls density)
    #[arg(short = 'r', long, default_value_t = 4.0)]
    rs: f64,

    /// Jastrow parameter F (defaults to rs/2)
    #[arg(short = 'j', long)]
    jastrow: Option<f64>,

    /// Number of VMC steps
    #[arg(short = 's', long, default_value_t = 100_000)]
    steps: usize,

    /// Number of walkers
    #[arg(short = 'w', long, default_value_t = 10)]
    walkers: usize,

    /// Monkhorst-Pack grid size for twist-averaging (N×N×N twists)
    #[arg(short = 't', long, default_value_t = 1)]
    twists: usize,
}

/// Hartree to eV conversion factor
const HA_TO_EV: f64 = 27.21138602;

/// Known Ceperley-Alder correlation energies for reference (Ha/electron)
fn ceperley_alder_correlation(rs: f64) -> f64 {
    // Perdew-Zunger parameterization of Ceperley-Alder data
    // Valid for rs >= 1 (paramagnetic phase)
    if rs >= 1.0 {
        let gamma = -0.1423;
        let beta1 = 1.0529;
        let beta2 = 0.3334;
        let rs_sqrt = rs.sqrt();
        gamma / (1.0 + beta1 * rs_sqrt + beta2 * rs)
    } else {
        // High density limit
        let a = 0.0311;
        let b = -0.048;
        let c = 0.0020;
        let d = -0.0116;
        a * rs.ln() + b + c * rs * rs.ln() + d * rs
    }
}

fn main() {
    let args = Args::parse();

    // Validate electron number
    let closed_shell = [2, 14, 38, 54, 66, 114, 162];
    if !closed_shell.contains(&args.electrons) {
        println!("Warning: {} electrons is not a closed-shell number.", args.electrons);
        println!("Recommended: {:?}", closed_shell);
    }

    // Jastrow parameter defaults to 5.5 (optimized for correlation energy accuracy at rs≈4)
    let jastrow_f = args.jastrow.unwrap_or(5.5);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Homogeneous Electron Gas (HEG) VMC Simulation            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    println!("║  System for LDA Correlation Energy Parameterization          ║");
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Generate Monkhorst-Pack twist grid
    let twist_grid = HomogeneousElectronGas::monkhorst_pack_grid(args.twists);
    let num_twists = twist_grid.len();

    // Create first HEG for display purposes
    let heg_display = HomogeneousElectronGas::new(args.electrons, args.rs, jastrow_f);

    println!("Simulation Parameters:");
    println!("══════════════════════");
    println!("  Electrons (N):       {}", args.electrons);
    println!("  Spin up / down:      {} / {}", heg_display.num_up, heg_display.num_down);
    println!("  Wigner-Seitz rs:     {:.2} Bohr", args.rs);
    println!("  Box length L:        {:.4} Bohr", heg_display.box_length);
    println!("  Density n:           {:.6} electrons/Bohr³", 
        args.electrons as f64 / heg_display.box_length.powi(3));
    println!("  Jastrow F:           {:.2}", jastrow_f);
    println!("  VMC walkers:         {}", args.walkers);
    println!("  VMC steps:           {}", args.steps);
    println!("  Twist grid:          {}×{}×{} = {} twists", 
        args.twists, args.twists, args.twists, num_twists);
    println!();

    // Reference values
    let e_hf = heg_display.hartree_fock_energy();
    let e_c_ref = ceperley_alder_correlation(args.rs);
    
    println!("Reference Values:");
    println!("═════════════════");
    println!("  Hartree-Fock E/N:    {:.6} Ha ({:.4} eV)", e_hf, e_hf * HA_TO_EV);
    println!("  CA Correlation εc:   {:.6} Ha ({:.4} eV)", e_c_ref, e_c_ref * HA_TO_EV);
    println!("  Expected total E/N:  {:.6} Ha", e_hf + e_c_ref);
    println!();

    // MCMC parameters
    let params = MCMCParams {
        n_walkers: args.walkers,
        n_steps: args.steps / num_twists.max(1),  // Split steps among twists
        initial_step_size: 0.5 * args.rs,
        max_step_size: 2.0 * args.rs,
        min_step_size: 0.1,
        target_acceptance: 0.5,
        adaptation_interval: 100,
    };

    // Run VMC simulation with twist averaging
    println!("Running VMC simulation with {} twist(s)...", num_twists);
    println!("════════════════════════════════════════════");
    
    let mut total_energy = 0.0;
    let mut total_variance = 0.0;
    
    for (i, twist) in twist_grid.iter().enumerate() {
        if num_twists > 1 && (i == 0 || (i + 1) % 10 == 0 || i == num_twists - 1) {
            println!("  Twist {}/{}: θ = ({:.3}, {:.3}, {:.3})", 
                i + 1, num_twists, twist.x, twist.y, twist.z);
        }
        
        // Create HEG with this twist
        let heg = HomogeneousElectronGas::new_with_twist(
            args.electrons, args.rs, jastrow_f, *twist
        );
        
        let mut simulation = MCMCSimulation::new(heg, params.clone());
        let results = simulation.run();
        
        total_energy += results.energy;
        total_variance += results.error * results.error;
    }
    
    // Average over twists
    let avg_energy = total_energy / num_twists as f64;
    let avg_error = (total_variance / num_twists as f64).sqrt();

    // Compute per-electron energy
    let energy_per_electron = avg_energy / args.electrons as f64;
    let error_per_electron = avg_error / args.electrons as f64;
    
    // Estimate correlation energy
    let correlation_estimate = energy_per_electron - e_hf;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                          RESULTS                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                              ║");
    if num_twists > 1 {
        println!("  [Twist-averaged over {} k-points]", num_twists);
        println!("║                                                              ║");
    }
    println!("  Total energy:        {:.6} ± {:.6} Ha", avg_energy, avg_error);
    println!("  Energy per electron: {:.6} ± {:.6} Ha", energy_per_electron, error_per_electron);
    println!("  Energy per electron: {:.4} ± {:.4} eV", 
        energy_per_electron * HA_TO_EV, error_per_electron * HA_TO_EV);
    println!("║                                                              ║");
    println!("  Hartree-Fock E/N:    {:.6} Ha", e_hf);
    println!("  VMC correlation εc:  {:.6} ± {:.6} Ha", correlation_estimate, error_per_electron);
    println!("  CA reference εc:     {:.6} Ha", e_c_ref);
    println!("  Difference:          {:.6} Ha ({:.1}%)", 
        correlation_estimate - e_c_ref,
        100.0 * (correlation_estimate - e_c_ref).abs() / e_c_ref.abs());
    println!("║                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Summary for LDA usage
    println!("For LDA parameterization:");
    println!("══════════════════════════");
    println!("  rs = {:.2}  →  εc = {:.6} Ha/electron", args.rs, correlation_estimate);
    println!();
    println!("Run at multiple rs values (1, 2, 5, 10, 20, 50, 100) to fit LDA functional.");
}

