//! Configuration file parsing for QMC simulations.

use std::fs::File;
use std::io::{BufReader, Result};
use crate::h2_mol::{H2MoleculeVB, H2MoleculeMO};

/// Load H₂ molecule (VB wavefunction) from YAML file.
///
/// # Example YAML format
/// ```yaml
/// orbital1:
///   alpha: 1.0
///   center: [0.0, 0.0, 0.7]
/// orbital2:
///   alpha: 1.0
///   center: [0.0, 0.0, -0.7]
/// jastrow:
///   cusp_param: 1.0
/// ```
pub fn read_h2molecule_vb(filename: &str) -> Result<H2MoleculeVB> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let h2 = serde_yaml::from_reader(reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(h2)
}

/// Load H₂ molecule (MO wavefunction) from YAML file.
pub fn read_h2molecule_mo(filename: &str) -> Result<H2MoleculeMO> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let h2 = serde_yaml::from_reader(reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(h2)
}
