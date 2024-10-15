use rand::{Rng, RngCore};
use rand_distr::{Normal, Distribution};

const MAX_CLONES: i32 = 3; //  number of clones to prevent explosion

// reference paper, https://www.thphys.uni-heidelberg.de/~wetzel/qmc2006/KOSZ96.pdf
// Define a trait for walker behavior
pub trait Walker {
    // Move the walker to a new position
    fn move_walker(&mut self);

    // Calculate local properties like energy
    fn calculate_local_energy(&mut self);

    // Update the walker's weight
    fn update_weight(&mut self, e_ref: f64);

    // Decide whether to branch (clone) or die
    fn branching_decision(&mut self) -> BranchingResult;

    // Check if the walker should be deleted
    fn should_be_deleted(&self) -> bool;

    // Mark the walker for deletion
    fn mark_for_deletion(&mut self);
}

// Define an enum for branching decisions
pub enum BranchingResult {
    Clone { n: usize }, // n is the number of clones
    Keep,               // The walker continues as is
    Kill,               // The walker should be removed
}

#[derive(Copy, Clone)]
struct HarmonicWalker {
    position: f64,
    dt: f64,  // Δτ
    sdt: f64, // √Δτ
    energy: f64,
    weight: f64,
    marked_for_deletion: bool,
}

// Initialize HarmonicWalker
impl HarmonicWalker {
    fn new(dt: f64, eref: f64) -> Self {
        let position = 0.0;
        let energy = 0.0;
        let weight = 1.0;
        let marked_for_deletion = false;
        let mut r = Self {
            position,
            dt,
            sdt: dt.sqrt(),
            energy,
            weight,
            marked_for_deletion,
        };
        r.calculate_local_energy();
        r.update_weight(eref);
        r
    }
}

impl Walker for HarmonicWalker {
    fn move_walker(&mut self) {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, self.sdt).unwrap();
        self.position += dist.sample(&mut rng);
    }

    fn calculate_local_energy(&mut self) {
        let x = self.position;
        self.energy = 0.5 * x * x;
    }

    fn update_weight(&mut self, e_ref: f64) {
        self.weight = ((-self.energy + e_ref) * self.dt).exp();
    }

    fn branching_decision(&mut self) -> BranchingResult {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen::<f64>();
        let cnt = ((self.weight + r).floor() as i32).max(0).min(MAX_CLONES);
        if cnt == 0 {
            self.marked_for_deletion = true;
        }
        match cnt {
            0 => BranchingResult::Kill,
            1 => BranchingResult::Keep,
            _ => BranchingResult::Clone { n: cnt as usize },
        }
    }

    fn should_be_deleted(&self) -> bool {
        self.marked_for_deletion
    }

    fn mark_for_deletion(&mut self) {
        self.marked_for_deletion = true;
    }
}

pub(crate) fn run_harmonic_dmc_sampling() {
    let n_walkers = 10000;
    let n_target = 10000;
    let n_steps = 10000;
    let dt = 0.01;
    let mut eref = 0.0;
    let mut walkers: Vec<HarmonicWalker> = vec![];
    for _ in 0..n_walkers {
        walkers.push(HarmonicWalker::new(dt, eref));
    }

    // Vectors to store mean energies and standard deviations
    let mut mean_energies = Vec::with_capacity(n_steps);
    let mut std_energies = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        // Move and update each walker
        for walker in walkers.iter_mut() {
            walker.move_walker();
            walker.calculate_local_energy();
            walker.update_weight(eref);
        }

        // Branching process
        let mut new_walkers: Vec<HarmonicWalker> = vec![];
        for walker in walkers.iter_mut() {
            match walker.branching_decision() {
                BranchingResult::Clone { n } => {
                    for _ in 0..n {
                        new_walkers.push(walker.clone());
                    }
                }
                BranchingResult::Keep => {
                    new_walkers.push(walker.clone());
                }
                BranchingResult::Kill => {
                    walker.mark_for_deletion();
                }
            }
        }

        walkers = new_walkers;
        eref = eref + (1.0 - walkers.len() as f64 / n_target as f64) / dt;

        // Calculate mean energy and standard deviation
        let energies: Vec<f64> = walkers.iter().map(|w| w.energy).collect();
        let n = energies.len() as f64;
        let mean_energy = energies.iter().sum::<f64>() / n;
        let variance_energy = energies
            .iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>()
            / n;
        let std_energy = variance_energy.sqrt();

        mean_energies.push(mean_energy);
        std_energies.push(std_energy);

        println!(
            "In step {:06}, Number of walkers: {:06}, energy: {:12.6}, std_dev: {:12.6}",
            step, walkers.len(), mean_energy, std_energy
        );
    }

    let n_walkers = walkers.len();
    println!("Final number of walkers: {}", n_walkers);

    // Write final positions to a file (wavefunction)
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = File::create("positions.txt").unwrap();
    let mut writer = BufWriter::new(file);
    for walker in walkers.iter() {
        writeln!(writer, "{}", walker.position).unwrap();
    }

    // Write energy trajectory to a file
    let file = File::create("energies.txt").unwrap();
    let mut writer = BufWriter::new(file);
    for (step, (mean_energy, std_energy)) in mean_energies
        .iter()
        .zip(std_energies.iter())
        .enumerate()
    {
        writeln!(
            writer,
            "{} {} {}",
            step, mean_energy, std_energy
        )
            .unwrap();
    }

    // Calculate overall average energy and standard deviation
    let n = mean_energies.len() as f64;
    let overall_mean = mean_energies.iter().sum::<f64>() / n;
    let overall_variance = mean_energies
        .iter()
        .map(|e| (e - overall_mean).powi(2))
        .sum::<f64>()
        / n;
    let overall_std_dev = overall_variance.sqrt();

    println!(
        "Overall mean energy: {}, overall std dev: {}",
        overall_mean, overall_std_dev
    );
}

fn main() {
    run_harmonic_dmc_sampling();
}
