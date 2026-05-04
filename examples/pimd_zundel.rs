//! Multi-Atom PIMD -- Zundel Cation H5O2+ Proton Transfer
//!
//! Run with: cargo run --release --example pimd_zundel
//!
//! Simulates quantum proton transfer in the Zundel cation:
//!   H2O - H+ ... OH2  <->  H2O ... H+ - OH2
//!
//! The shared proton (H+) shuttles between two water molecules.
//! This is the fundamental mechanism of proton conductivity in water
//! (the Grotthuss mechanism).
//!
//! Key physics:
//! - 7 atoms (2 O + 5 H), 21 degrees of freedom
//! - Very low barrier (~0.75 kcal/mol) -> strong quantum effects
//! - The proton is nearly equally shared between both oxygens
//!
//! Improvements over basic model:
//! - Analytical EVB forces (Hellmann-Feynman, ~20x faster)
//! - R_g decomposition into parallel/perpendicular components
//! - H/D kinetic isotope effect comparison
//! - Free energy profile W(d) = -kT ln P(d)
//! - R_OO vs d 2D correlation histogram

use rust_qmc::sampling::run_pimd_zundel;

fn main() {
    let temperature_k = 300.0;
    let beta = 315774.65 / temperature_k;

    let n_beads = 32;
    let n_polymers = 20;
    let dt = 0.3;
    let n_equilibrate = 20_000;
    let n_production = 50_000;

    println!("================================================================");
    println!("|   Zundel Cation H5O2+ -- Proton Transfer in Water          |");
    println!("|                                                             |");
    println!("|       H   H              H   H                            |");
    println!("|        \\ /                \\ /                             |");
    println!("|    O - H+... O    <->    O ...H+- O                        |");
    println!("|        / \\                / \\                             |");
    println!("|       H   H              H   H                            |");
    println!("|                                                             |");
    println!("|   The Grotthuss mechanism of proton conduction              |");
    println!("|                                                             |");
    println!("|   Features:                                                 |");
    println!("|   • Analytical EVB forces (Hellmann-Feynman)               |");
    println!("|   • H/D kinetic isotope effect                             |");
    println!("|   • Free energy profile W(d)                               |");
    println!("|   • R_OO vs d correlation                                  |");
    println!("================================================================");
    println!();

    run_pimd_zundel(
        n_polymers,
        n_beads,
        beta,
        dt,
        n_equilibrate,
        n_production,
    );

    println!();
    println!("================================================================");
    println!("|  Output files:                                              |");
    println!("|    pimd_zundel_distribution.txt -- P(d) + W(d) + H/D       |");
    println!("|    pimd_zundel_energy.txt       -- energy + R_gpar/perp + R_OO  |");
    println!("|    pimd_zundel_roo_correlation.txt -- 2D R_OO vs d         |");
    println!("|    pimd_zundel_beads.txt        -- 3D bead positions         |");
    println!("================================================================");
}
