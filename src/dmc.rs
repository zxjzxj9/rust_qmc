
use nalgebra::{min, ComplexField, DMatrix, Matrix, Vector3};
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;

// reference: https://www.thphys.uni-heidelberg.de/~wetzel/qmc2006/KOSZ96.pdf

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

    // // Clone the walker (for branching)
    // fn clone_walker(&self) -> Self
    // where
    //     Self: Sized;

    // Check if the walker should be deleted
    fn should_be_deleted(&self) -> bool;

    // Mark the walker for deletion
    fn mark_for_deletion(&mut self);
}

// Define an enum for branching decisions
pub enum BranchingResult {
    Clone{n: usize}, // n is the size to be cloned
    Keep,  // The walker continues as is
    Kill,  // The walker should be removed
}

#[derive(Copy, Clone)]
struct HarmonicWalker {
    position: f64,
    dt: f64, // \Delta \tau
    isdt: f64, // \frac{1}{\sqrt \Delta \tau}
    energy: f64,
    weight: f64,
    marked_for_deletion: bool,
}

/// initialize HarmonicWalker
impl HarmonicWalker {
    fn new(dt: f64, eref: f64) -> Self {
        let position = 0.0;
        let energy = 0.0;
        let weight = 1.0;
        let marked_for_deletion = false;
        let mut r = Self {
            position: position,
            dt: dt,
            isdt: 1.0 / dt.sqrt(),
            energy: energy,
            weight: weight,
            marked_for_deletion: false,
        };
        r.calculate_local_energy();
        r.update_weight(eref);
        r
    }
}

impl Walker for HarmonicWalker {
    fn move_walker(&mut self) {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, self.isdt).unwrap();
        self.position += dist.sample(&mut rng)
    }

    fn calculate_local_energy(&mut self) {
        let x = self.position;
        self.energy = 0.5 * x * x;
    }

    fn update_weight(&mut self, e_ref: f64) {
        self.weight = ((-self.energy + e_ref)*self.dt).exp();
    }

    fn branching_decision(&mut self) -> BranchingResult {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(0.0, 1.0);
        let r: f64 = dist.sample(&mut rng);
        // make a branching decision
        let cnt = min((self.weight + r) as i32, 3);
        if cnt == 0 {
            self.marked_for_deletion = true;
        }
        match cnt {
            0 => BranchingResult::Kill,
            1 => BranchingResult::Keep,
            _ => BranchingResult::Clone{n: cnt as usize},
        }
    }

    // fn clone_walker(&self) -> Self {
    //     Self {
    //         position: self.position,
    //         energy: self.energy,
    //         weight: self.weight,
    //         marked_for_deletion: false,
    //     }
    // }

    fn should_be_deleted(&self) -> bool {
        self.marked_for_deletion
    }

    fn mark_for_deletion(&mut self) {
        self.marked_for_deletion = true;
    }
}