# QMC program using Rust

---
This is a simple QMC program using Rust. The program is based on the VMC method. The program is written in Rust and uses the `ndarray` crate for array manipulation and the `rand` crate for random number generation.


## How to run the program
Simply run the following command in the terminal:
``` bash
cargo run --release
```

### H2 Result
Following is a table with $\alpha$, $F$ and $E$ for the H2 molecule. 
The values are obtained using the VMC method with $2\times10^6$ samples.

| Basis | $\alpha$ | $F$ | Binding Energy / eV |
|-------|----------|-----|---------------------|
| VB    | 0.6      | 0.0 | -2.932              |
| VB    | 0.7      | 5.0 | -4.136              |
| LCAO  | 1.2      | 0.0 | -3.488              |
| LCAO  | 1.3      | 3.0 | -4.287              |


## Todos

- [ ] Add rayon for parallelization
- [ ] Support LCAO-MO basis
- [ ] Other systems than H2 molecule
- [ ] Add more tests