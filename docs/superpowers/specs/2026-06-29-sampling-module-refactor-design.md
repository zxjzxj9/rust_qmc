# Sampling Module Refactor — Design

**Date:** 2026-06-29
**Goal:** Make `src/sampling/` more concise and modular by de-duplicating shared
abstractions and co-locating related code. API cleanup is permitted (examples and
`main.rs` will be updated to match).

## Problem

`src/sampling/` is ~13k lines across 20 files, several over 1000 lines
(`pimd_molecular.rs` 2532, `pimd.rs` 1468, `umbrella.rs` 1164, `metadynamics.rs`
1084, `pimc.rs` 1074, `piglet.rs` 1007). The duplication and tangling fall into
a few clear categories:

1. **Potential traits are buried in driver files.** `Potential` (1D scalar) lives
   in `pimc.rs`; `MolecularPotential` (flat coordinate array) lives in
   `pimd_molecular.rs`; `SplittablePotential` / `SplittableMolecularPotential` in
   `pimd_rpc.rs`. Consumers reach across modules (`super::pimc::Potential`,
   `super::pimd_molecular::MolecularPotential`).

2. **Concrete potentials are buried in driver files.** 1D potentials (Harmonic,
   Sombrero, DoubleWell, ProtonTransfer) live inside `pimc.rs`; molecular
   potentials (Bifluoride, Zundel) inside `pimd_molecular.rs`. They are physics
   definitions, not sampling drivers, and belong together.

3. **Shared machinery lives in one driver and is reached from others.**
   `NormalModeTransform` and `PILEThermostat` live in `pimd.rs` but are imported
   by `pimd_molecular.rs` and `pimd_rpc.rs` via `super::pimd::`. They should live
   in a neutral module that both depend on.

4. **Copy-pasted numerical helpers.** Spring/kinetic action and primitive-energy
   estimators are repeated across `pimc.rs` (twice: `QuantumPath` and
   `GeneralPath`), `pimc_fermion.rs`, `pimd.rs`, and `pimd_molecular.rs`. These
   operate on *three different data layouts* (scalar `f64`, `Vector3<f64>`, flat
   `&[f64]`), so they are not trivially mergeable.

## Scope decisions

- **Keep two potential traits** (`Potential` for 1D scalar PIMC, `MolecularPotential`
  for flat-coordinate PIMD). Unifying them would force scalar PIMC onto
  `Vec<f64>`/`&[f64]` in hot loops for no real gain and added risk. Rejected
  (this was "Approach B").
- **Relocate, don't force-merge, the numerical helpers.** Spring-action and
  estimator code that differs only by data layout stays with its driver. Only
  helpers that are *identical* (same types, same math) get extracted into a shared
  module. This avoids touching hot-loop numerics where layouts differ.
- **Electronic-structure MC is out of scope.** `vmc.rs`, `dmc.rs`, `is_dmc.rs`,
  `optimize.rs`, `sr_optimize.rs`, `geometry_opt.rs`, `force_variance.rs` are not
  part of the PIMC/PIMD duplication and are left as-is (they only move if a
  directory grouping requires a path change).

## Target layout

```
sampling/
  mod.rs              curated re-exports (grouped, documented)
  traits.rs           EnergyCalculator, ForceCalculator, Walker, VmcWalker (unchanged)

  potentials/
    mod.rs
    traits.rs         Potential, MolecularPotential,
                      SplittablePotential, SplittableMolecularPotential
    scalar.rs         Harmonic, Sombrero, DoubleWell, ProtonTransfer
    molecular.rs      Bifluoride (BifluoridePES), Zundel (ZundelPES)
    dftb.rs           ToyDFTB (already a molecular potential)

  pimd_core.rs        NormalModeTransform, PILEThermostat
                      (relocated from pimd.rs; the neutral home both 1D and
                       molecular drivers depend on)

  pimc.rs             QuantumPath, PIMCSimulation, GeneralPath, GeneralPIMC,
                      run_pimc_*  (potentials removed; import from potentials::)
  pimc_fermion.rs     TrialWavefunction, FermionPIMC, run_pimc_hydrogen

  pimd.rs             RingPolymer, PIMDSimulation, run_pimd_proton_transfer
                      (NormalModeTransform/PILEThermostat removed → pimd_core)
  pimd_molecular.rs   MolecularRingPolymer, MolecularPILE, MolecularPIMD,
                      run_pimd_bifluoride, run_pimd_zundel, free_energy_profile
                      (potentials + traits removed → potentials::)
  pimd_rpc.rs         RPContraction, RPCRingPolymer, RPCSimulation
                      (traits removed → potentials::)
  piglet.rs           PIQTBThermostat, MolecularPIQTB, PIGLETThermostat, helpers

  enhanced/
    mod.rs
    umbrella.rs       UmbrellaBias, BiasedMolecularPotential, WHAMSolver, runners
    metadynamics.rs   GaussianHill, MetadynamicsBias, MetadynamicsPotential, runners

  optimize.rs sr_optimize.rs geometry_opt.rs force_variance.rs  (unchanged)
  vmc.rs dmc.rs is_dmc.rs                                       (unchanged)
```

## Dependency direction (after refactor)

```
traits.rs        (no sampling deps)
potentials/      (no sampling deps; pure physics + trait defs)
pimd_core.rs     (no sampling deps; normal-mode + thermostat math)
   ↑
pimc, pimd, pimd_molecular, pimd_rpc, piglet, pimc_fermion
   ↑
enhanced/ (umbrella, metadynamics)  — wrap MolecularPotential as decorators
```

No more `super::pimc::` / `super::pimd::` reach-through between sibling drivers;
shared items come from `potentials::`, `pimd_core::`, or `traits::`.

## Public API changes

`mod.rs` and `lib.rs` re-exports are regrouped with the same type names, so most
external use keeps working. Where a type's canonical path changes (e.g.
`HarmonicPotential` now under `potentials::scalar`), `mod.rs` re-exports it so the
flat `rust_qmc::sampling::HarmonicPotential` path is preserved. Examples and
`main.rs` are updated only if a name they reference actually changes. The contract:
**`cargo build --examples` and `cargo test` both pass with no behavior change.**

## Migration order (low-risk, each step compiles + tests green)

1. Create `potentials/traits.rs`; move the four trait definitions there; update
   imports throughout. Build + test.
2. Create `potentials/scalar.rs` and `potentials/molecular.rs`; move concrete
   potentials out of `pimc.rs` / `pimd_molecular.rs`. Move `dftb.rs` →
   `potentials/dftb.rs`. Build + test.
3. Create `pimd_core.rs`; move `NormalModeTransform` + `PILEThermostat` out of
   `pimd.rs`; repoint `pimd_molecular`/`pimd_rpc` imports. Build + test.
4. Move `umbrella.rs` + `metadynamics.rs` into `enhanced/`; fix paths. Build + test.
5. Extract any *identical-layout* duplicated helpers into the appropriate module
   (only where types match exactly). Build + test.
6. Rewrite `mod.rs` re-exports into documented groups; reconcile `lib.rs`. Build
   `--examples` + test.

## Testing

- The existing `#[cfg(test)]` suites in `pimc.rs`, `pimd.rs`, `pimd_rpc.rs`,
  `pimd_molecular.rs`, `metadynamics.rs`, `pimc_fermion.rs`, `optimize.rs`,
  `sr_optimize.rs`, `dftb.rs`, and `lib.rs` move with their code and must keep
  passing unchanged — they are the behavior-preservation guarantee.
- After every migration step: `cargo test` and `cargo build --examples`.
- No new numerical logic is introduced, so no new physics tests are required;
  added value is structural.

## Out of scope (YAGNI)

- Unifying the two potential traits (Approach B).
- Merging numerical kernels across differing data layouts.
- Refactoring `systems/`, `wavefunction/`, or the embedded tests in `lib.rs`
  beyond what the import changes require.
- Touching the electronic-structure MC drivers' internals.
