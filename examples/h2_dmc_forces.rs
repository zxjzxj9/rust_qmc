//! H2 DMC Force Curve
//!
//! Scans the H2 bond distance using importance-sampled DMC to compute
//! energies and Hellmann-Feynman forces at each separation. The force
//! along the bond axis should match the numerical derivative of E(R).
//!
//! Usage:
//!   cargo run --example h2_dmc_forces --release -- [OPTIONS]
//!
//! Options:
//!   -n, --steps <N>     DMC production steps [default: 3000]
//!   -w, --walkers <N>   Target walker population [default: 200]
//!   --tau <τ>           Time step [default: 0.005]
//!   --burnin <N>        Burn-in steps [default: 500]
//!   --points <N>        Number of bond-distance points [default: 10]

use clap::Parser;
use nalgebra::Vector3;
use rust_qmc::{
    H2MoleculeMO, ISDMCParams, ImportanceSampledDMC,
    Slater1s, Jastrow1,
};

/// H2 DMC Force Curve
#[derive(Parser, Debug)]
#[command(version, about = "Compute H2 energy and force curve with IS-DMC")]
struct Args {
    /// DMC production steps per bond distance
    #[arg(short = 'n', long, default_value_t = 3000)]
    steps: usize,

    /// Target walker population
    #[arg(short, long, default_value_t = 200)]
    walkers: usize,

    /// Time step τ
    #[arg(long, default_value_t = 0.005)]
    tau: f64,

    /// Burn-in steps
    #[arg(long, default_value_t = 500)]
    burnin: usize,

    /// Number of bond-distance scan points
    #[arg(long, default_value_t = 10)]
    points: usize,
}

fn main() {
    let args = Args::parse();

    println!("================================================================");
    println!("|        H2 Potential Energy & Force Curve via IS-DMC        |");
    println!("================================================================");
    println!();
    println!("  Walkers:    {}", args.walkers);
    println!("  Steps:      {}", args.steps);
    println!("  Burn-in:    {}", args.burnin);
    println!("  τ:          {:.4}", args.tau);
    println!("  Scan pts:   {}", args.points);
    println!();

    // Scan bond distances from 0.8 to 5.0 Bohr
    let r_min = 0.8_f64;
    let r_max = 5.0_f64;
    let n_pts = args.points;
    let bond_distances: Vec<f64> = (0..n_pts)
        .map(|i| r_min + (r_max - r_min) * i as f64 / (n_pts - 1).max(1) as f64)
        .collect();

    // Collect results
    let mut results: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::new(); // (R, E, err, F_mixed, F_extrap, F_vmc)

    for (idx, &r) in bond_distances.iter().enumerate() {
        let half_r = r / 2.0;

        // Create H2 at bond distance R along z-axis
        let wfn = H2MoleculeMO {
            orbital1: Slater1s {
                center: Vector3::new(0.0, 0.0, -half_r),
                alpha: 1.0, // Optimized STO exponent for H2
            },
            orbital2: Slater1s {
                center: Vector3::new(0.0, 0.0, half_r),
                alpha: 1.0,
            },
            jastrow: Jastrow1 { cusp_param: 5.0 },
        };

        println!("-------------------------------------------------------------");
        println!("  Point {}/{}: R = {:.3} Bohr", idx + 1, n_pts, r);
        println!("-------------------------------------------------------------");

        let params = ISDMCParams {
            n_walkers: args.walkers,
            n_steps: args.steps,
            n_burnin: args.burnin,
            time_step: args.tau,
            print_interval: args.steps, // Print only at end
            ..Default::default()
        };

        let mut dmc = ImportanceSampledDMC::new(wfn, params);
        let result = dmc.run();

        // Force along bond axis (z-component of nucleus 1; by symmetry F1 = -F2)
        let f_mixed_z = result.forces_mixed[0].z;
        let f_extrap_z = result.forces_extrapolated[0].z;
        let f_vmc_z = result.forces_vmc[0].z;

        println!(
            "  E = {:10.6} +/- {:.5} Ha | pop = {:.0} | acc = {:.1}%",
            result.energy, result.error, result.avg_population, result.acceptance_rate * 100.0
        );
        println!(
            "  F_z(nuc1): mixed = {:+.5}, extrap = {:+.5}, VMC = {:+.5} Ha/Bohr",
            f_mixed_z, f_extrap_z, f_vmc_z
        );
        println!();

        results.push((r, result.energy, result.error, f_mixed_z, f_extrap_z, f_vmc_z));
    }

    // ------ Summary table ------
    println!();
    println!("================================================================================");
    println!("|                           Summary: E(R) and F(R)                            |");
    println!("================================================================================");
    println!("|  R(a0) |   E (Ha)   | +/-error   |  F_mixed   |  F_extrap  |  -dE/dR    | d(%)|");
    println!("=========╬============╬==========╬============╬============╬============╬=======");

    for (i, &(r, e, err, f_mix, f_ext, _f_vmc)) in results.iter().enumerate() {
        // Numerical dE/dR for comparison
        let de_dr = if i > 0 && i < results.len() - 1 {
            let (r_prev, e_prev, _, _, _, _) = results[i - 1];
            let (r_next, e_next, _, _, _, _) = results[i + 1];
            -(e_next - e_prev) / (r_next - r_prev) // -dE/dR = force
        } else if i == 0 && results.len() > 1 {
            let (r_next, e_next, _, _, _, _) = results[1];
            -(e_next - e) / (r_next - r)
        } else if i == results.len() - 1 && results.len() > 1 {
            let (r_prev, e_prev, _, _, _, _) = results[i - 1];
            -(e - e_prev) / (r - r_prev)
        } else {
            0.0
        };

        // Relative deviation of F_extrap from -dE/dR (if both nonzero)
        let dev_pct = if de_dr.abs() > 1e-4 && f_ext.abs() > 1e-4 {
            ((f_ext - de_dr) / de_dr * 100.0).abs()
        } else {
            0.0
        };

        println!(
            "| {:6.3} | {:10.6} | {:8.5} | {:+10.5} | {:+10.5} | {:+10.5} |{:5.1} |",
            r, e, err, f_mix, f_ext, de_dr, dev_pct
        );
    }

    println!("================================================================================");
    println!();
    println!("Notes:");
    println!("  • F_mixed  = DMC mixed estimator <Ψ0|F_HF|Ψ_T> (z-component on nucleus 1)");
    println!("  • F_extrap = 2xF_mixed - F_VMC  (extrapolated estimator, most accurate)");
    println!("  • -dE/dR   = numerical derivative from E(R) curve (for comparison)");
    println!("  • d(%)     = |F_extrap - (-dE/dR)| / |-dE/dR| x 100");
    println!("  • Exact H2 equilibrium: R ~ 1.401 Bohr, E ~ -1.17448 Ha");
    println!("  • At equilibrium, force should be ~ 0 (switching from attractive to repulsive)");
}
