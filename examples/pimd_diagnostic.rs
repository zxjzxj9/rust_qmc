// Quick PIMD diagnostic using PIMDSimulation (with corrected PILE thermostat)
use rust_qmc::sampling::{PIMDSimulation, HarmonicPotential};

fn main() {
    let pot = HarmonicPotential { mass: 1.0, omega: 1.0 };
    let n_beads: usize = 32;
    let beta: f64 = 20.0;
    let mass: f64 = 1.0;
    let dt: f64 = 0.05;
    let gamma: f64 = 1.0;

    let mut sim = PIMDSimulation::new(20, n_beads, beta, mass, dt, gamma, pot);

    // Equilibrate
    for _ in 0..10000 {
        sim.step_obabo();
    }

    // Sample
    let mut e_vir = Vec::new();
    let mut e_prim = Vec::new();
    let mut x2_vals = Vec::new();
    let mut rg_vals = Vec::new();
    
    for _ in 0..10000 {
        sim.step_obabo();
        e_vir.push(sim.average_virial_energy());
        e_prim.push(sim.average_primitive_energy());
        rg_vals.push(sim.average_radius_of_gyration());
        
        let mut x2: f64 = 0.0;
        for p in &sim.polymers {
            for &x in &p.positions {
                x2 += x * x;
            }
        }
        x2 /= (sim.polymers.len() * n_beads) as f64;
        x2_vals.push(x2);
    }

    let n = e_vir.len() as f64;
    let mean_e_vir = e_vir.iter().sum::<f64>() / n;
    let mean_e_prim = e_prim.iter().sum::<f64>() / n;
    let mean_x2 = x2_vals.iter().sum::<f64>() / n;
    let mean_rg = rg_vals.iter().sum::<f64>() / n;

    println!("PIMD diagnostic (ω=1, m=1, β=20, P=32)");
    println!("Expected: E₀ = 0.5, <x²> = 0.5");
    println!("E_virial    = {:.6}", mean_e_vir);
    println!("E_primitive = {:.6}", mean_e_prim);
    println!("<x²>        = {:.6}", mean_x2);
    println!("<R_g>       = {:.6}", mean_rg);
    println!();
    
    // Check MD kinetic energy per bead (should be ½kBT_P = P/(2β) for each bead)
    let avg_ke: f64 = sim.polymers.iter().map(|p| p.kinetic_energy_md()).sum::<f64>() / sim.polymers.len() as f64;
    let expected_total_ke = n_beads as f64 * 0.5 * (n_beads as f64 / beta); // N × ½kBT_P
    println!("Total MD KE: {:.6} (expected: {:.6})", avg_ke, expected_total_ke);
    println!("KE per bead: {:.6} (expected: {:.6})", avg_ke / n_beads as f64, 0.5 * n_beads as f64 / beta);
}
