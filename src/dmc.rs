
use nalgebra::{DMatrix, Matrix, Vector3};
use rand_distr::Normal;

trait DMCWalker {
    fn new(n: usize) -> Self;
    fn initialize(&mut self);
    fn evaluate(&mut self, r: &Vec<Vector3<f64>>);
    fn local_energy(&mut self, r: &Vec<Vector3<f64>>);
    fn drift(&mut self, r: &Vec<Vector3<f64>>) -> Vec<Vector3<f64>>;
    fn branching(&mut self, r: &Vec<Vector3<f64>>, w: f64, e: f64);
}