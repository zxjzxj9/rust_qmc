# Sampling Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-duplicate and modularize `src/sampling/` by relocating the potential traits, concrete potentials, and shared ring-polymer machinery into focused modules, without changing any numerical behavior.

**Architecture:** Pure relocation/regrouping refactor. Shared abstractions (`Potential`/`MolecularPotential` traits, concrete potentials, `NormalModeTransform`/`PILEThermostat`) move into neutral modules (`potentials/`, `pimd_core.rs`) that drivers depend on, eliminating sibling `super::pimc::`/`super::pimd::` reach-through. Biasing methods group under `enhanced/`. No numerical kernels are merged.

**Tech Stack:** Rust 2021, nalgebra, rayon. Verification via `cargo test` and `cargo build --examples`.

---

## Verification model

This is a behavior-preserving refactor, not a feature build. The existing
`#[cfg(test)]` suites (in `pimc.rs`, `pimc_fermion.rs`, `pimd.rs`, `pimd_rpc.rs`,
`pimd_molecular.rs`, `metadynamics.rs`, `optimize.rs`, `sr_optimize.rs`, `dftb.rs`,
`lib.rs`) **are** the safety net. Every task ends by running them plus the examples
build; both must stay green. No new tests are written because no new behavior is
added — moving a test with its code and confirming it still passes is the proof.

**The two verification commands used throughout (referred to as "the full check"):**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected after every task: all tests `ok` (0 failed), examples build with no errors.

**Baseline (recorded 2026-06-29):** `test result: ok. 106 passed; 0 failed` and all
examples build clean (one pre-existing `unused_imports` warning in
`examples/ch4_sr_optimize.rs`, unrelated to this refactor). So **N = 106** — this
count must not change in any task. The full test suite takes ~4.6 min to run, so
each task's verification is not instant; budget for it. If the baseline is not
green when execution starts, stop — the refactor cannot be verified.

---

## File Structure (end state)

```
sampling/
  mod.rs              curated, grouped re-exports
  traits.rs           EnergyCalculator, ForceCalculator, Walker, VmcWalker (unchanged)
  potentials/
    mod.rs            re-exports the trait + potential types
    traits.rs         Potential, MolecularPotential, SplittablePotential, SplittableMolecularPotential
    scalar.rs         HarmonicPotential, SombreroPotential, DoubleWellPotential, ProtonTransferPotential
    molecular.rs      BifluoridePES, ZundelPES
    dftb.rs           ToyDFTB
  pimd_core.rs        NormalModeTransform, PILEThermostat
  pimc.rs             QuantumPath, PIMCSimulation, GeneralPath, GeneralPIMC, run_pimc_*
  pimc_fermion.rs     TrialWavefunction, Hydrogen1s, FermionPath, FermionPIMC, run_pimc_hydrogen
  pimd.rs             RingPolymer, PIMDSimulation, run_pimd_proton_transfer
  pimd_molecular.rs   MolecularRingPolymer, MolecularPILE, MolecularPIMD, run_pimd_bifluoride, run_pimd_zundel, free_energy_profile
  pimd_rpc.rs         RPContraction, RPCRingPolymer, RPCSimulation
  piglet.rs           PIQTBThermostat, MolecularPIQTB, PIGLETThermostat, helpers
  enhanced/
    mod.rs
    umbrella.rs       UmbrellaBias, BiasedMolecularPotential, UmbrellaWindow, WHAMSolver, runners
    metadynamics.rs   GaussianHill, MetadynamicsBias, MetadynamicsPotential, MetadynamicsResult, runners
  optimize.rs sr_optimize.rs geometry_opt.rs force_variance.rs   (unchanged)
  vmc.rs dmc.rs is_dmc.rs                                        (unchanged)
```

---

### Task 0: Confirm green baseline

**Files:** none (verification only)

- [ ] **Step 1: Run the full check on a clean tree**

```bash
git status --short          # working tree clean (ignore the stray *.txt data files)
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected: all tests `ok`, 0 failed; examples build, no errors. Record the total
test count shown (e.g. `test result: ok. N passed`) — N must not change in any
later task.

- [ ] **Step 2: Create a working branch**

```bash
git checkout -b refactor/sampling-modules
```

---

### Task 1: Extract potential traits into `potentials/traits.rs`

Move the four trait definitions out of the driver files. Nothing else moves yet.

**Files:**
- Create: `src/sampling/potentials/mod.rs`
- Create: `src/sampling/potentials/traits.rs`
- Modify: `src/sampling/pimc.rs` (remove `pub trait Potential`, ~lines 501-519)
- Modify: `src/sampling/pimd_molecular.rs` (remove `pub trait MolecularPotential`, ~lines 32-85)
- Modify: `src/sampling/pimd_rpc.rs` (remove `pub trait SplittablePotential` / `SplittableMolecularPotential`, ~lines 201-230; update its `use` lines 21-23)
- Modify: `src/sampling/mod.rs` (declare `pub mod potentials;`)

- [ ] **Step 1: Create `potentials/traits.rs`**

Move the exact, unchanged definitions of `Potential` (from `pimc.rs`),
`MolecularPotential` (from `pimd_molecular.rs`), and `SplittablePotential` +
`SplittableMolecularPotential` (from `pimd_rpc.rs`) into this file. Preserve every
doc comment and default-method body verbatim. `SplittablePotential: Potential` and
`SplittableMolecularPotential: MolecularPotential` now refer to the traits defined
above them in the same file.

File header:

```rust
//! Potential-energy abstractions shared by all PIMC/PIMD samplers.
//!
//! `Potential` is the 1D scalar form used by path-integral Monte Carlo;
//! `MolecularPotential` is the flat-Cartesian-coordinate form used by molecular
//! path-integral MD. The `Splittable*` variants add fast/slow force splitting
//! for ring-polymer contraction (`pimd_rpc`).

// (trait bodies moved verbatim from pimc.rs / pimd_molecular.rs / pimd_rpc.rs)
```

- [ ] **Step 2: Create `potentials/mod.rs`**

```rust
//! Potential-energy traits and concrete potentials for PIMC/PIMD.

pub mod traits;

pub use traits::{
    Potential, MolecularPotential,
    SplittablePotential, SplittableMolecularPotential,
};
```

- [ ] **Step 3: Wire the module in and repoint imports**

In `src/sampling/mod.rs` add `pub mod potentials;` (near the other `mod`
declarations). In `pimc.rs`, `pimd.rs`, `pimd_molecular.rs`, `pimd_rpc.rs`,
`piglet.rs`, `umbrella.rs`, `metadynamics.rs`, replace any local definition or
cross-module import of these traits with:

```rust
use super::potentials::{Potential, MolecularPotential,
                        SplittablePotential, SplittableMolecularPotential};
```

Import only the traits each file actually uses. Delete the now-removed trait
bodies from `pimc.rs`, `pimd_molecular.rs`, `pimd_rpc.rs`. In `pimd_rpc.rs`,
delete lines 21-23's old `use super::pimc::Potential;` /
`use super::pimd_molecular::MolecularPotential;` and fold them into the new
`super::potentials::` import.

- [ ] **Step 4: Run the full check**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected: same test count N as baseline, all `ok`; examples build clean.
If the compiler reports an unresolved trait, find the file still defining or
importing it from the old path and repoint it to `super::potentials::`.

- [ ] **Step 5: Commit**

```bash
git add src/sampling/potentials/ src/sampling/mod.rs src/sampling/pimc.rs src/sampling/pimd.rs src/sampling/pimd_molecular.rs src/sampling/pimd_rpc.rs src/sampling/piglet.rs src/sampling/umbrella.rs src/sampling/metadynamics.rs
git commit -m "refactor(sampling): move potential traits into potentials::traits"
```

---

### Task 2: Move concrete potentials into `potentials/scalar.rs` and `potentials/molecular.rs`

**Files:**
- Create: `src/sampling/potentials/scalar.rs`
- Create: `src/sampling/potentials/molecular.rs`
- Modify: `src/sampling/pimc.rs` (remove HarmonicPotential, SombreroPotential, DoubleWellPotential, ProtonTransferPotential structs + impls, ~lines 522-697)
- Modify: `src/sampling/pimd_molecular.rs` (remove BifluoridePES and ZundelPES structs + impls)
- Modify: `src/sampling/potentials/mod.rs` (add the two submodules + re-exports)

- [ ] **Step 1: Create `potentials/scalar.rs`**

Move the four 1D potential structs and their `impl ... {}` blocks **and their
`impl Potential for ... {}` blocks** verbatim from `pimc.rs`. Add at top:

```rust
//! Concrete 1D scalar potentials for path-integral Monte Carlo.

use super::traits::Potential;
```

If any of these also has a `SplittablePotential` impl currently in `pimd_rpc.rs`,
leave that impl where it is for now (it imports the type from here).

- [ ] **Step 2: Create `potentials/molecular.rs`**

Move `BifluoridePES` and `ZundelPES` (structs, inherent `impl`, and
`impl MolecularPotential for ...`) verbatim from `pimd_molecular.rs`. Add at top:

```rust
//! Concrete molecular (flat-Cartesian) potentials for path-integral MD.

use super::traits::MolecularPotential;
```

- [ ] **Step 3: Update `potentials/mod.rs`**

```rust
//! Potential-energy traits and concrete potentials for PIMC/PIMD.

pub mod traits;
pub mod scalar;
pub mod molecular;

pub use traits::{
    Potential, MolecularPotential,
    SplittablePotential, SplittableMolecularPotential,
};
pub use scalar::{
    HarmonicPotential, SombreroPotential, DoubleWellPotential, ProtonTransferPotential,
};
pub use molecular::{BifluoridePES, ZundelPES};
```

- [ ] **Step 4: Repoint consumers**

In `pimc.rs`, `pimd.rs`, `pimd_rpc.rs` add `use super::potentials::scalar::*;`
(or named imports) wherever the scalar potentials are referenced; delete the moved
struct bodies from `pimc.rs`. In `pimd_molecular.rs`, `umbrella.rs`,
`metadynamics.rs`, `pimd_rpc.rs` add `use super::potentials::molecular::{BifluoridePES, ZundelPES};`
where referenced (note: `umbrella.rs`/`metadynamics.rs` reference `ZundelPES` only
inside their `#[cfg(test)]` modules — update those `use super::super::...` paths to
`use super::super::potentials::molecular::ZundelPES;`). Delete the moved bodies
from `pimd_molecular.rs`.

- [ ] **Step 5: Run the full check**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected: test count N unchanged, all `ok`; examples build clean. Examples that
referenced e.g. `rust_qmc::sampling::HarmonicPotential` still work because `mod.rs`
re-exports it (reconciled in Task 6); if an example fails to resolve a potential
name now, note it — Task 6 fixes the flat re-exports. To keep this task green,
temporarily ensure `sampling/mod.rs` still re-exports the moved names (it does via
the existing `pub use pimc::{...}` / `pub use pimd_molecular::{...}` lines, which
you update to point at `potentials::` here).

- [ ] **Step 6: Commit**

```bash
git add src/sampling/potentials/ src/sampling/pimc.rs src/sampling/pimd.rs src/sampling/pimd_molecular.rs src/sampling/pimd_rpc.rs src/sampling/umbrella.rs src/sampling/metadynamics.rs
git commit -m "refactor(sampling): relocate concrete potentials into potentials::{scalar,molecular}"
```

---

### Task 3: Move DFTB into `potentials/dftb.rs`

**Files:**
- Create: `src/sampling/potentials/dftb.rs` (move from `src/sampling/dftb.rs`)
- Delete: `src/sampling/dftb.rs`
- Modify: `src/sampling/mod.rs` (remove `mod dftb;` and `pub use dftb::ToyDFTB;`)
- Modify: `src/sampling/potentials/mod.rs` (add `pub mod dftb; pub use dftb::ToyDFTB;`)

- [ ] **Step 1: Move the file**

```bash
git mv src/sampling/dftb.rs src/sampling/potentials/dftb.rs
```

- [ ] **Step 2: Fix imports inside `dftb.rs`**

`ToyDFTB` implements `MolecularPotential`. Change its trait import to
`use super::traits::MolecularPotential;` (it previously used
`use super::pimd_molecular::MolecularPotential;` or `super::MolecularPotential`).
Update its `#[cfg(test)]` module imports similarly if present.

- [ ] **Step 3: Update both `mod.rs` files**

In `src/sampling/mod.rs` delete the `mod dftb;` line and the `pub use dftb::ToyDFTB;`
line. In `src/sampling/potentials/mod.rs` add:

```rust
pub mod dftb;
pub use dftb::ToyDFTB;
```

Keep `ToyDFTB` reachable at `sampling::ToyDFTB` by adding to `sampling/mod.rs`:
`pub use potentials::ToyDFTB;` (reconciled with the rest in Task 6).

- [ ] **Step 4: Run the full check**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected: test count N unchanged, all `ok`; examples build clean.

- [ ] **Step 5: Commit**

```bash
git add src/sampling/
git commit -m "refactor(sampling): move ToyDFTB under potentials::dftb"
```

---

### Task 4: Extract `NormalModeTransform` + `PILEThermostat` into `pimd_core.rs`

**Files:**
- Create: `src/sampling/pimd_core.rs`
- Modify: `src/sampling/pimd.rs` (remove `NormalModeTransform` ~lines 39-327 and `PILEThermostat` ~lines 329-420)
- Modify: `src/sampling/pimd_molecular.rs` (`use super::pimd::NormalModeTransform` → `use super::pimd_core::NormalModeTransform`)
- Modify: `src/sampling/pimd_rpc.rs` (`use super::pimd::{NormalModeTransform, PILEThermostat}` → `use super::pimd_core::{...}`)
- Modify: `src/sampling/mod.rs` (declare `mod pimd_core;`, re-export the two types)

- [ ] **Step 1: Create `pimd_core.rs`**

Move `NormalModeTransform` (struct + full `impl`, including all `to_normal_modes*`
methods and the FFT/auto variants) and `PILEThermostat` (struct + `impl`) verbatim
from `pimd.rs`. Header:

```rust
//! Core ring-polymer machinery shared across PIMD samplers:
//! the normal-mode transform and the PILE thermostat.

use rand::Rng;
use rand_distr::{Distribution, Normal};
// (add exactly the imports the moved code used in pimd.rs)
```

Copy the imports these items actually used from `pimd.rs`'s top (e.g. rand,
rand_distr, any `std::f64::consts`). Do not guess — match what the moved code
references.

- [ ] **Step 2: Wire module + repoint imports**

In `src/sampling/mod.rs` add `mod pimd_core;` and (temporarily) keep the public
names available via the existing `pub use pimd::{NormalModeTransform, PILEThermostat, ...}`
line — change it to pull those two from `pimd_core` instead:
`pub use pimd_core::{NormalModeTransform, PILEThermostat};` and drop them from the
`pub use pimd::{...}` list. In `pimd.rs` add
`use super::pimd_core::{NormalModeTransform, PILEThermostat};` (it still uses them
internally) and delete the moved bodies. In `pimd_molecular.rs` and `pimd_rpc.rs`,
change `super::pimd::` to `super::pimd_core::` for these two types.

- [ ] **Step 3: Move the associated tests**

`pimd.rs` has `test_normal_mode_roundtrip` and `test_normal_mode_centroid`
(~lines 1247, 1259). Move these two tests into a `#[cfg(test)]` module in
`pimd_core.rs` so the normal-mode tests live with the code. Leave thermostat/PIMD
tests in `pimd.rs`.

- [ ] **Step 4: Run the full check**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected: test count N unchanged (tests moved, not removed), all `ok`; examples
build clean.

- [ ] **Step 5: Commit**

```bash
git add src/sampling/
git commit -m "refactor(sampling): extract NormalModeTransform and PILEThermostat into pimd_core"
```

---

### Task 5: Group biasing methods under `enhanced/`

**Files:**
- Create: `src/sampling/enhanced/mod.rs`
- Move: `src/sampling/umbrella.rs` → `src/sampling/enhanced/umbrella.rs`
- Move: `src/sampling/metadynamics.rs` → `src/sampling/enhanced/metadynamics.rs`
- Modify: `src/sampling/mod.rs` (remove `mod umbrella; mod metadynamics;`, add `mod enhanced;`)

- [ ] **Step 1: Move the files**

```bash
git mv src/sampling/umbrella.rs src/sampling/enhanced/umbrella.rs
git mv src/sampling/metadynamics.rs src/sampling/enhanced/metadynamics.rs
```

- [ ] **Step 2: Fix relative paths inside the moved files**

Both files now sit one level deeper, so `super::` references to sibling sampling
modules become `super::super::`. Update each import:
- `use super::potentials::...` → `use super::super::potentials::...`
- `use super::pimd_molecular::{MolecularPIMD, ...}` → `use super::super::pimd_molecular::{...}`
- `use super::pimd_core::...` (if any) → `use super::super::pimd_core::...`
- inside `#[cfg(test)]`: existing `use super::super::potentials::molecular::ZundelPES;`
  (set in Task 2) becomes `use super::super::super::potentials::molecular::ZundelPES;`
  — i.e. one more `super`. Verify by letting the compiler report the exact path.

- [ ] **Step 3: Create `enhanced/mod.rs`**

```rust
//! Enhanced-sampling / free-energy methods that bias a `MolecularPotential`:
//! umbrella sampling (+WHAM) and metadynamics.

pub mod umbrella;
pub mod metadynamics;

pub use umbrella::{
    UmbrellaBias, BiasedMolecularPotential, UmbrellaWindow, WHAMSolver,
    run_pimd_umbrella_sampling, run_zundel_umbrella_sampling,
};
pub use metadynamics::{
    GaussianHill, MetadynamicsBias, MetadynamicsPotential, MetadynamicsResult,
    run_pimd_metadynamics, run_zundel_metadynamics,
};
```

- [ ] **Step 4: Update `sampling/mod.rs`**

Replace the `mod umbrella;` / `mod metadynamics;` declarations with `mod enhanced;`
and change the two `pub use umbrella::{...}` / `pub use metadynamics::{...}` blocks
to `pub use enhanced::umbrella::{...}` / `pub use enhanced::metadynamics::{...}`
(same names; reconciled in Task 6).

- [ ] **Step 5: Run the full check**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected: test count N unchanged, all `ok`; examples build clean. The examples
`pimd_umbrella_zundel.rs`, `pimd_ti_umbrella_zundel.rs`, and
`pimd_metadynamics_zundel.rs` use the flat `rust_qmc::sampling::` names, which are
preserved — they must still compile.

- [ ] **Step 6: Commit**

```bash
git add src/sampling/
git commit -m "refactor(sampling): group umbrella and metadynamics under enhanced/"
```

---

### Task 6: Rewrite `mod.rs` re-exports and reconcile `lib.rs`

Now that everything is relocated, make the public surface clean and grouped, and
confirm the whole external API still resolves.

**Files:**
- Modify: `src/sampling/mod.rs` (rewrite into documented, grouped re-exports)
- Modify: `src/lib.rs` (update the `pub use sampling::{...}` line if any path changed)

- [ ] **Step 1: Rewrite `src/sampling/mod.rs`**

Organize declarations and re-exports into commented groups. Every public name that
was exported before must still be exported (same identifier) so examples keep
compiling. Structure:

```rust
//! Sampling module — Monte Carlo sampling methods for QMC.

mod traits;
pub mod potentials;
mod pimd_core;

mod vmc;
mod dmc;
mod is_dmc;
mod pimc;
mod pimc_fermion;
mod pimd;
mod pimd_molecular;
mod pimd_rpc;
mod piglet;
mod enhanced;
mod optimize;
mod sr_optimize;
pub mod geometry_opt;
pub mod force_variance;

// --- Core traits ---
pub use traits::{EnergyCalculator, ForceCalculator, Walker, BranchingResult, VmcWalker};

// --- Potentials ---
pub use potentials::{
    Potential, MolecularPotential, SplittablePotential, SplittableMolecularPotential,
    HarmonicPotential, SombreroPotential, DoubleWellPotential, ProtonTransferPotential,
    BifluoridePES, ZundelPES, ToyDFTB,
};

// --- Ring-polymer core ---
pub use pimd_core::{NormalModeTransform, PILEThermostat};

// --- Electronic-structure MC ---
pub use vmc::{MCMCParams, MCMCState, MCMCResults, MCMCSimulation,
              DDVMCParams, DDVMCResults, DriftDiffusionVMC};
pub use dmc::{run_dmc_sampling, HarmonicWalker, HydrogenAtomWalker, HydrogenMoleculeWalker};
pub use is_dmc::{ISDMCParams, ISDMCResults, ImportanceSampledDMC};
pub use optimize::{JastrowOptimizer, OptimizationResult, SamplingStats};
pub use sr_optimize::{SROptimizer, SRResult};
pub use geometry_opt::{GeometryOptimizer, GeometryOptResult};
pub use force_variance::ForceEstimator;

// --- Path-integral MC ---
pub use pimc::{QuantumPath, PIMCSimulation, run_pimc_harmonic,
               GeneralPath, GeneralPIMC, run_pimc_sombrero};
pub use pimc_fermion::{TrialWavefunction, Hydrogen1s, FermionPath, FermionPIMC, run_pimc_hydrogen};

// --- Path-integral MD ---
pub use pimd::{RingPolymer, PIMDSimulation, run_pimd_proton_transfer};
pub use pimd_molecular::{
    MolecularRingPolymer, MolecularPILE, MolecularPIMD,
    run_pimd_bifluoride, run_pimd_zundel, free_energy_profile,
};
pub use pimd_rpc::{RPContraction, RPCRingPolymer, RPCSimulation};
pub use piglet::{
    PIQTBThermostat, MolecularPIQTB, PIGLETThermostat,
    matrix_exponential, cholesky_decompose, load_piglet_matrices,
};

// --- Enhanced sampling / free energy ---
pub use enhanced::umbrella::{
    UmbrellaBias, BiasedMolecularPotential, UmbrellaWindow, WHAMSolver,
    run_pimd_umbrella_sampling, run_zundel_umbrella_sampling,
};
pub use enhanced::metadynamics::{
    GaussianHill, MetadynamicsBias, MetadynamicsPotential, MetadynamicsResult,
    run_pimd_metadynamics, run_zundel_metadynamics,
};
```

Cross-check this list against the *original* `mod.rs` re-exports (recorded in the
spec / git history) so no previously-public name is dropped. Note that the old list
re-exported some `pimc` potentials (`Potential, HarmonicPotential, ...`) and
`pimd_molecular` traits — those now come from `potentials::`; ensure each appears
exactly once.

- [ ] **Step 2: Reconcile `src/lib.rs`**

`lib.rs:16` re-exports a long flat list from `sampling`. Every name there must
still resolve. Update it to match the names re-exported in Step 1 (they are the
same identifiers, so likely no change is needed beyond confirming). If `lib.rs`
re-exported a name that moved trait-homes (e.g. nothing should have), fix it.

- [ ] **Step 3: Run the full check**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
```

Expected: test count N unchanged, all `ok`; **all 24 examples build clean**. This
is the final gate — if any example fails to resolve a name, a re-export was
dropped in Step 1; add it back.

- [ ] **Step 4: Confirm no warnings introduced**

```bash
cargo build 2>&1 | grep -i "warning" | grep -iv "unused.*txt" || echo "no new warnings"
```

Expected: no new `unused import` warnings from the moved code. Remove any dead
`use` left behind by the relocations.

- [ ] **Step 5: Commit**

```bash
git add src/sampling/mod.rs src/lib.rs
git commit -m "refactor(sampling): regroup public re-exports after modularization"
```

---

### Task 7: Final verification and file-size check

**Files:** none (verification only)

- [ ] **Step 1: Full test + examples + doc build**

```bash
cargo test 2>&1 | tail -20
cargo build --examples 2>&1 | tail -5
cargo build --release 2>&1 | tail -5
```

Expected: test count N unchanged, all `ok`; examples and release build clean.

- [ ] **Step 2: Confirm the structure shrank the giant files**

```bash
find src/sampling -name '*.rs' | xargs wc -l | sort -n | tail -20
```

Expected: `pimc.rs`, `pimd.rs`, `pimd_molecular.rs` are visibly smaller than the
baseline (potentials, traits, and core machinery now live in `potentials/` and
`pimd_core.rs`). No single concern is defined in two files.

- [ ] **Step 3: Update the README ToDos if relevant**

The README lists "Add rayon for parallelization" — leave content as-is unless the
refactor changed a documented entry point. No code change expected here.

- [ ] **Step 4: Final commit (if anything left)**

```bash
git status --short
# if clean, nothing to do
```

---

## Self-Review Notes

- **Spec coverage:** Tasks 1-3 cover spec problem #1 (traits) and #2 (concrete
  potentials buried in drivers); Task 4 covers #3 (shared machinery relocation);
  Task 5 covers the `enhanced/` grouping; Task 6 covers the public-API regrouping;
  Task 7 confirms file-size reduction. Spec's "relocate, don't force-merge" decision
  is honored — no task merges numerical kernels.
- **Out of scope** items from the spec (unify traits, merge kernels, touch
  systems/wavefunction/electronic-MC internals) appear in no task. Correct.
- **Type consistency:** trait names (`Potential`, `MolecularPotential`,
  `SplittablePotential`, `SplittableMolecularPotential`), potential type names
  (`HarmonicPotential`, `SombreroPotential`, `DoubleWellPotential`,
  `ProtonTransferPotential`, `BifluoridePES`, `ZundelPES`, `ToyDFTB`), and core
  types (`NormalModeTransform`, `PILEThermostat`) are used identically across all
  tasks and match the current `mod.rs` exports.
- **Verification:** every task ends with the same "full check"; line counts and
  exact `super::` depths are confirmed by the compiler rather than guessed, which
  is the safe move for path-sensitive relocations.
