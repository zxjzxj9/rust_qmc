
// use rs to read the config file into the following H2 Molecular structure
/*
    let h2 = H2MoleculeVB {
        H1: Slater1s { alpha: 1.0, R: Vector3::new(0.0, 0.0, 0.7) },
        H2: Slater1s { alpha: 1.0, R: Vector3::new(0.0, 0.0, -0.7) },
        J: Jastrow1 { F: 1.0 },
    };
 */

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use crate::h2_mol::{H2MoleculeVB, H2MoleculeMO, Slater1s, Jastrow1};

/// write function to serialize H2MoleculeVB and deserialize H2MoleculeVB, with ymal format
pub fn read_h2molecule_vb(filename: &str) -> H2MoleculeVB {
    let file = std::fs::File::open(filename).unwrap();
    let reader = std::io::BufReader::new(file);
    let h2: H2MoleculeVB = serde_yaml::from_reader(reader).unwrap();
    h2
}

/// write function to serialize H2MoleculeMO and deserialize H2MoleculeVB, with ymal format
pub fn read_h2molecule_mo(filename: &str) -> H2MoleculeMO {
    let file = std::fs::File::open(filename).unwrap();
    let reader = std::io::BufReader::new(file);
    let h2: H2MoleculeMO = serde_yaml::from_reader(reader).unwrap();
    h2
}

// example of ymal file
// H1:
//   alpha: 1.0
//   R: [0.0, 0.0, 0.7]
// H2:
//   alpha: 1.0
//   R: [0.0, 0.0, -0.7]
// J:
//   F: 1.0
