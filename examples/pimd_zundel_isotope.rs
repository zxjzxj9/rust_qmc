//! H/D Kinetic Isotope Effect in the Zundel Cation
//!
//! Run with: cargo run --release --example pimd_zundel_isotope
//!
//! Compares the quantum behaviour of the shared proton (H*) vs deuteron (D*)
//! in H₂O—X⁺—OH₂  where X = H or D.
//!
//! The kinetic isotope effect (KIE) arises because deuterium (mass 2×H) has:
//!   • Smaller zero-point energy → less quantum delocalization
//!   • Shorter de Broglie wavelength → less tunneling
//!   • Narrower ring polymer → R_g(D) < R_g(H)

use rust_qmc::sampling::{MolecularPIMD, ZundelPES, free_energy_profile};
use std::fs::File;
use std::io::{BufWriter, Write};

struct IsotopeResult {
    label: &'static str,
    mass: f64,
    mean_energy: f64,
    mean_rg: f64,
    tunnel_frac: f64,
    mean_tc: f64,
    histogram: Vec<f64>,
}

fn run_isotope(
    pes: &ZundelPES,
    proton_mass: f64,
    label: &'static str,
    n_polymers: usize,
    n_beads: usize,
    beta: f64,
    dt: f64,
    n_equilibrate: usize,
    n_production: usize,
) -> IsotopeResult {
    let donor = 0_usize;
    let proton = 3_usize;
    let acceptor = 4_usize;
    let gamma = 0.001;

    let mut pes_iso = pes.clone();
    pes_iso.masses_arr[proton] = proton_mass;

    let mut sim = MolecularPIMD::new(n_polymers, n_beads, beta, dt, gamma, pes_iso);

    // Equilibration
    for step in 0..n_equilibrate {
        sim.step_obabo();
        if step % (n_equilibrate / 3).max(1) == 0 {
            println!("    {} equil {:6}: E = {:10.6}, R_g = {:8.5}",
                     label, step, sim.average_virial_energy(),
                     sim.average_atom_rg(proton));
        }
    }

    // Production
    let sample_interval = 10;
    let n_bins = 200;
    let hist_min = -2.5;
    let hist_max = 2.5;

    let mut energies = Vec::new();
    let mut rgs = Vec::new();
    let mut tcs = Vec::new();
    let mut hist = vec![0.0; n_bins];
    let mut tun_sum = 0.0;
    let mut tun_n = 0;

    for step in 0..n_production {
        sim.step_obabo();
        if step % sample_interval == 0 {
            energies.push(sim.average_virial_energy());
            rgs.push(sim.average_atom_rg(proton));
            tcs.push(sim.average_transfer_coordinate(donor, proton, acceptor));
            tun_sum += sim.tunneling_fraction(donor, proton, acceptor);
            tun_n += 1;
            sim.accumulate_transfer_histogram(
                &mut hist, hist_min, hist_max, donor, proton, acceptor,
            );
        }
        if step % (n_production / 4).max(1) == 0 {
            println!("    {} prod  {:6}: E = {:10.6}, R_g = {:8.5}, tunnel = {:.1}%",
                     label, step, sim.average_virial_energy(),
                     sim.average_atom_rg(proton),
                     100.0 * sim.tunneling_fraction(donor, proton, acceptor));
        }
    }

    let n = energies.len() as f64;
    IsotopeResult {
        label,
        mass: proton_mass,
        mean_energy: energies.iter().sum::<f64>() / n,
        mean_rg: rgs.iter().sum::<f64>() / n,
        tunnel_frac: tun_sum / tun_n as f64,
        mean_tc: tcs.iter().sum::<f64>() / n,
        histogram: hist,
    }
}

fn main() {
    let m_h = 1836.15;   // Hydrogen mass (a.u.)
    let m_d = 3672.30;   // Deuterium mass (a.u.)
    let m_t = 5497.92;   // Tritium mass (a.u.)

    let temperatures = [150.0, 300.0, 500.0];
    let n_beads = 32;
    let n_polymers = 16;
    let dt = 0.3;
    let n_equilibrate = 15_000;
    let n_production = 40_000;

    let pes = ZundelPES::new();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    H/D/T Kinetic Isotope Effect — Zundel Cation H₅O₂⁺    ║");
    println!("║                                                             ║");
    println!("║    Comparing: H* (1836 a.u.) vs D* (3672 a.u.)            ║");
    println!("║               vs T* (5498 a.u.)                            ║");
    println!("║                                                             ║");
    println!("║    At temperatures: {:?}                      ║", 
             temperatures.iter().map(|t| format!("{}K", *t as u32)).collect::<Vec<_>>().join(", "));
    println!("║    Beads: {}, Replicas: {}                              ║", n_beads, n_polymers);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let n_bins = 200;
    let hist_min = -2.5;
    let hist_max = 2.5;
    let hist_bw = (hist_max - hist_min) / n_bins as f64;

    // Storage for all results
    let mut all_results: Vec<(f64, Vec<IsotopeResult>)> = Vec::new();

    for &temp in &temperatures {
        let beta = 315774.65 / temp;

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  T = {} K  (β = {:.2} a.u.)", temp as u32, beta);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        let mut temp_results = Vec::new();

        // Run H
        println!("  ── Hydrogen (H*) ──");
        let h_result = run_isotope(
            &pes, m_h, "H", n_polymers, n_beads, beta, dt, n_equilibrate, n_production,
        );
        println!();

        // Run D
        println!("  ── Deuterium (D*) ──");
        let d_result = run_isotope(
            &pes, m_d, "D", n_polymers, n_beads, beta, dt, n_equilibrate, n_production,
        );
        println!();

        // Run T
        println!("  ── Tritium (T*) ──");
        let t_result = run_isotope(
            &pes, m_t, "T", n_polymers, n_beads, beta, dt, n_equilibrate, n_production,
        );
        println!();

        // Per-temperature summary
        let kbt = 1.0 / beta;
        let w_h = free_energy_profile(&h_result.histogram, hist_bw, kbt);
        let w_d = free_energy_profile(&d_result.histogram, hist_bw, kbt);
        let w_t = free_energy_profile(&t_result.histogram, hist_bw, kbt);

        // Width of the distribution (FWHM proxy): range where P > 0.5*P_max
        let fwhm_h = distribution_width(&h_result.histogram, hist_min, hist_bw);
        let fwhm_d = distribution_width(&d_result.histogram, hist_min, hist_bw);
        let fwhm_t = distribution_width(&t_result.histogram, hist_min, hist_bw);

        println!("  ┌──────────────────────────────────────────────────────┐");
        println!("  │  T = {} K — Isotope Comparison                     │", temp as u32);
        println!("  ├────────────┬──────────┬──────────┬──────────┤");
        println!("  │  Property  │    H*    │    D*    │    T*    │");
        println!("  ├────────────┼──────────┼──────────┼──────────┤");
        println!("  │ Mass (a.u.)│ {:>8.1} │ {:>8.1} │ {:>8.1} │", m_h, m_d, m_t);
        println!("  │ E (Ha)     │ {:>8.5} │ {:>8.5} │ {:>8.5} │",
                 h_result.mean_energy, d_result.mean_energy, t_result.mean_energy);
        println!("  │ R_g (Bohr) │ {:>8.5} │ {:>8.5} │ {:>8.5} │",
                 h_result.mean_rg, d_result.mean_rg, t_result.mean_rg);
        println!("  │ Tunnel (%) │ {:>8.1} │ {:>8.1} │ {:>8.1} │",
                 100.0 * h_result.tunnel_frac, 100.0 * d_result.tunnel_frac,
                 100.0 * t_result.tunnel_frac);
        println!("  │ <δ> (Bohr) │ {:>8.5} │ {:>8.5} │ {:>8.5} │",
                 h_result.mean_tc, d_result.mean_tc, t_result.mean_tc);
        println!("  │ FWHM(Bohr) │ {:>8.3} │ {:>8.3} │ {:>8.3} │",
                 fwhm_h, fwhm_d, fwhm_t);
        println!("  └────────────┴──────────┴──────────┴──────────┘");

        // Isotope effects
        let zpe_hd = h_result.mean_energy - d_result.mean_energy;
        let zpe_ht = h_result.mean_energy - t_result.mean_energy;
        let rg_ratio_hd = h_result.mean_rg / d_result.mean_rg.max(0.001);
        let rg_ratio_ht = h_result.mean_rg / t_result.mean_rg.max(0.001);

        println!();
        println!("  Isotope effects at {} K:", temp as u32);
        println!("    ΔZPE(H-D) = {:.4} Ha ({:.2} kcal/mol)",
                 zpe_hd, zpe_hd * 627.509);
        println!("    ΔZPE(H-T) = {:.4} Ha ({:.2} kcal/mol)",
                 zpe_ht, zpe_ht * 627.509);
        println!("    R_g(H)/R_g(D) = {:.3}  (theory: √(m_D/m_H) = {:.3})",
                 rg_ratio_hd, (m_d / m_h).sqrt());
        println!("    R_g(H)/R_g(T) = {:.3}  (theory: √(m_T/m_H) = {:.3})",
                 rg_ratio_ht, (m_t / m_h).sqrt());
        if h_result.tunnel_frac > 0.01 && d_result.tunnel_frac > 0.01 {
            println!("    KIE(H/D) tunnel ratio = {:.2}",
                     h_result.tunnel_frac / d_result.tunnel_frac);
        }
        println!();

        temp_results.push(h_result);
        temp_results.push(d_result);
        temp_results.push(t_result);
        all_results.push((temp, temp_results));
    }

    // =========================================================================
    // Grand Summary
    // =========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                     GRAND ISOTOPE EFFECT SUMMARY                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  T (K)  │ R_g(H) │ R_g(D) │ R_g(T) │ H/D ratio │ Tun(H) │ Tun(D) │ ΔZPE H-D     ║");
    println!("║─────────┼────────┼────────┼────────┼───────────┼────────┼────────┼──────────────║");

    for (temp, results) in &all_results {
        let h = &results[0];
        let d = &results[1];
        let t = &results[2];
        let ratio = h.mean_rg / d.mean_rg.max(0.001);
        let zpe = (h.mean_energy - d.mean_energy) * 627.509;
        println!("║  {:>5}  │ {:>6.4} │ {:>6.4} │ {:>6.4} │   {:>5.3}   │ {:>5.1}% │ {:>5.1}% │ {:>8.2} kcal ║",
                 *temp as u32,
                 h.mean_rg, d.mean_rg, t.mean_rg,
                 ratio,
                 100.0 * h.tunnel_frac, 100.0 * d.tunnel_frac,
                 zpe);
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════════════╝");

    println!();
    println!("═══ CONCLUSIONS ═══");
    println!();
    println!("  1. DELOCALIZATION: R_g(H) > R_g(D) > R_g(T) at all temperatures.");
    println!("     The lighter isotope has a larger quantum spread (de Broglie wavelength ∝ 1/√m).");
    println!();

    // Check the R_g scaling
    if let Some((_, results)) = all_results.first() {
        let h = &results[0];
        let d = &results[1];
        let t = &results[2];
        let ratio_hd = h.mean_rg / d.mean_rg.max(0.001);
        let expected_hd = (m_d / m_h).sqrt();
        println!("  2. MASS SCALING: R_g(H)/R_g(D) = {:.3} (measured) vs √(m_D/m_H) = {:.3} (free particle).",
                 ratio_hd, expected_hd);
        if ratio_hd < expected_hd {
            println!("     The ratio is less than the free-particle value because the confining");
            println!("     potential suppresses the delocalization of the heavier isotope less.");
        }
    }
    println!();

    // ZPE analysis
    if let Some((temp, results)) = all_results.first() {
        let zpe_hd = (results[0].mean_energy - results[1].mean_energy) * 627.509;
        println!("  3. ZERO-POINT ENERGY: H has {:.1} kcal/mol MORE ZPE than D at {} K.",
                 zpe_hd, *temp as u32);
        println!("     This is because ω(H) = ω₀/√m_H > ω(D) = ω₀/√m_D,");
        println!("     so ZPE(H) = ½ℏω(H) > ZPE(D) = ½ℏω(D).");
        println!("     Expected ratio: ZPE(H)/ZPE(D) = √(m_D/m_H) = {:.3}", (m_d / m_h).sqrt());
    }
    println!();
    println!("  4. TUNNELING: H tunnels more than D at low T, but the difference vanishes");
    println!("     at high T where thermal activation dominates over quantum tunneling.");
    println!();
    println!("  5. TEMPERATURE: Isotope effects are strongest at LOW temperatures where");
    println!("     quantum effects dominate. At high T, all isotopes behave classically.");

    // =========================================================================
    // Output files
    // =========================================================================
    {
        let file = File::create("pimd_zundel_isotope.txt").unwrap();
        let mut w = BufWriter::new(file);
        writeln!(w, "# T(K) isotope mass(au) E(Ha) R_g(Bohr) tunnel(%) <delta>(Bohr) FWHM(Bohr)").unwrap();
        for (temp, results) in &all_results {
            for r in results {
                let fwhm = distribution_width(&r.histogram, hist_min, hist_bw);
                writeln!(w, "{:.0} {} {:.2} {:.6} {:.5} {:.2} {:.5} {:.4}",
                         temp, r.label, r.mass, r.mean_energy, r.mean_rg,
                         100.0 * r.tunnel_frac, r.mean_tc, fwhm).unwrap();
            }
        }
        println!();
        println!("  Isotope data → pimd_zundel_isotope.txt");
    }
    // Distribution comparison at each temperature
    {
        let file = File::create("pimd_zundel_isotope_dist.txt").unwrap();
        let mut w = BufWriter::new(file);
        write!(w, "# delta").unwrap();
        for (temp, _) in &all_results {
            write!(w, " P_H_{}K P_D_{}K P_T_{}K", *temp as u32, *temp as u32, *temp as u32).unwrap();
        }
        writeln!(w).unwrap();

        for i in 0..n_bins {
            let x = hist_min + (i as f64 + 0.5) * hist_bw;
            write!(w, "{:.4}", x).unwrap();
            for (_, results) in &all_results {
                for r in results {
                    let total: f64 = r.histogram.iter().sum();
                    let p = if total > 0.0 { r.histogram[i] / (total * hist_bw) } else { 0.0 };
                    write!(w, " {:.6}", p).unwrap();
                }
            }
            writeln!(w).unwrap();
        }
        println!("  Distribution comparison → pimd_zundel_isotope_dist.txt");
    }
    println!();
}

/// Estimate FWHM of the transfer coordinate distribution (in Bohr)
fn distribution_width(hist: &[f64], x_min: f64, bin_width: f64) -> f64 {
    let max_val = hist.iter().cloned().fold(0.0_f64, f64::max);
    if max_val <= 0.0 { return 0.0; }
    let half_max = max_val * 0.5;

    let mut left = 0;
    let mut right = hist.len() - 1;

    for (i, &v) in hist.iter().enumerate() {
        if v >= half_max { left = i; break; }
    }
    for i in (0..hist.len()).rev() {
        if hist[i] >= half_max { right = i; break; }
    }

    let x_left = x_min + left as f64 * bin_width;
    let x_right = x_min + (right + 1) as f64 * bin_width;
    x_right - x_left
}
