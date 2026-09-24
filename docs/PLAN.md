# SuperfluidGPU — Plan

Shared working notes. Either of us can edit. Committed to the repo so history
is preserved alongside the code.

## Goal

Reach superfluid mode on GPU at 1024³ via PETSc + Kokkos. First milestone:
strip this repo down to only the **superfluid-mode, noise-free** pathway so
the surface area we port to Kokkos is minimal.

Seed state: `SuperQuench@c137ab88`. The two repos are allowed to diverge.

## Status (2026-09-24)

- `873e606` — pruning (phases 1–4 below) committed.
- `745fa60` — defaults set to the reference-run values in both `ModelA.h`
  and `scripts/grunner.py`: f² = χ = 5, dt = 0.5, `ABBBBBB`, Γ = 0.25,
  D = Γ/3. `grunner.py` is the launch workflow; defaults are deliberately
  kept in both places.
- Laugh test (NX=32, 8 ranks, debug queue, same JSON + seed): SuperQuench and
  the pruned code give identical `phi` output. Work area:
  `/pscratch/sd/d/dteaney/squench_comparisons/`.
- Regression baseline (NX=32, 128 ranks, each code launched through its own
  `grunner.prlmrun`, seed 122335456): `compare.py` PASS — every dataset
  bit-identical between SuperQuench@c137ab88 and SuperfluidGPU@f98d9d6.
  Reference output: `squench_comparisons/baseline/baseline_sq/`. The random
  initial conditions are seeded per rank, so comparisons need equal rank counts.

## Superfluid-mode pathway (what we kept)

Runtime selection: `superfluidmode = true` and
`evolverType = "SuperSplitStep"`; `grunner.checkinputs()` asserts both.

Nothing in the stepping path calls `NoiseGenerator`; the trajectory is
deterministic given the initial conditions.

**Kept:**

- `SuperPions.cxx` — main + `Run` (only `SuperSplitStep`)
- `Stepper.{h,cxx}` — `Stepper` base, `IdealPV2`,
  `ModelGExplicitDiffusionStep`, `SuperSplitStep`
- `ModelA.h` — all data structures, initialization, coefficients
- `initialize.{h,cxx}` — shared initialization helpers
- `O4AlgebraHelper.h` — used by `IdealPV2`
- `measurer*.{h,cxx}`, `ntuple.h`, `nvector.h` — measurement + HDF5 output
- `make_unique.h`, JSON in `nlohmann/`

## Phase 4 — coefficient cleanup (done, `873e606`)

Removed from `ModelACoefficients`: `mass0`, `dmassdt`, `lambda`. All three
were dead in the superfluid-only stepping path (`mass()` was only surfaced
for logging/HDF5, `lambda` was only used by the deleted heat-bath). Cascade:
- `ModelACoefficients::mass(t)` method deleted.
- `ModelAData::mass()` convenience method deleted.
- `SuperPions.cxx` periodic log line no longer prints `mass = %e`.
- `measurer_output.cxx` `timeout` ntuple shrunk from 2 columns to 1 (just
  `time`; the `mass` column is gone).
- `grunner.py`: `mass0`/`dmassdt`/`lambda` removed from `data`, the
  mass-sign checks in `checkinputs()` removed, and the obsolete
  `get_kzfilename` / `get_qkzfilename` filename helpers deleted.
  `getdefault_filename` no longer includes mass in the name.

## Phase 3 — ModelA.h cleanup (done, `873e606`)

Removed:
- `initialize_gaussian_const()`, `initialize_random_domains()` (dead — no
  callers).
- `thermalization_time` field, `quench_mode` bool field (stored/printed but
  never read).
- `quench_mode_mass0` field + the `"quench_mode"` initialization branch in
  `SuperPions.cxx` (obsolete).
- `restart` bool field + entire restart pathway: the check in `initialize()`,
  the `ModelARndm->read()` call in `ModelA` constructor, the
  `ModelARndm->write()` call in `finalize()`, the `FILE_MODE_APPEND` branch
  in `run_event`, and the `initialization == "restart"` branch in
  `initialize_event`.
- `ModelA::read()` method (only used by the deleted restart pathway).
- `restart` + `quench_mode_mass0` fields from `grunner.py`.

Kept per user directive:
- `data_node` (used by `initialize.cxx`).
- `writeFrequency` snapshot output (barely used but useful for plotting).
- `initialize_random_spins()`, `initialize_gaussian_charges()`.

Remaining initialization branches: `default`, `randomspins`, `gaussians`,
`spinwaves`.

## Phase 1 + 2 — done (`873e606`)

Removed:
- `EulerLangevinHB`, `ModelGChargeHB` (+ helpers `modelg_update_charge_pair`,
  `g_face_case`, `g_face_cases`), `PV2HBSplit`, `o4_stepper_monitor`
- `thermalize_event` (function + both call sites in `initialize_event`).
  `default` init is now a cold start with no thermalization. (Quench mode
  was removed entirely in phase 3.)
- `ModelGDiffusionStep` (the Crank-Nicolson class) — entire class,
  including `Form3PointLaplacian`. **PETSc KSP is no longer needed by the
  stepping path.**
- `use_implicit_step` flag everywhere (SuperSplitStep constructor, Run
  config parsing, grunner.py).
- `Run` branches for `PV2HBSplit23`, `PV2HBSplitGeneral`, and standalone
  `ModelGDiffusionStep`. `Run` now only recognizes `SuperSplitStep`.
- `IdealPV2::accept_reject` parameter and reject logic. `IdealPV2::step`
  still computes `oldEnergy` before and `newEnergy` after — exposed via
  `getEnergy()` for monitoring — but never rejects. The last
  `ModelARndm->uniform()` call in the stepping path is gone.
- `grunner.py`: `pv2hb_split_general` block, non-superfluid `checkinputs()`
  branch, `thermalization_time` field, `quench_mode` bool field.

Config changes:
- `ModelA.h` default `evolverType` → `"SuperSplitStep"`.
- `grunner.py` default `evolverType` → `"SuperSplitStep"`, default
  `superfluidmode` → `True`, added `SuperSplitStep` block. (Default
  `step_sequence` later set to `"ABBBBBB"` in `745fa60`.)

Kept per user directive:
- `NoiseGenerator.{h,cxx}` + `xoroshiro128plus.h`. Now truly unused in the
  stepping path — only allocated once in the `ModelA` constructor.
  Convenient to have around for future work.
- `ModelGExplicitDiffusionStep` (the only remaining diffusion class).

## Active call graph

```
main → Run
        └── SuperSplitStep::step
             ├── 'A' → IdealPV2::step             (deterministic; monitors energy)
             └── 'B' → ModelGExplicitDiffusionStep::step   (deterministic)
```

## Next

1. **Regression baseline** — `compare.py` in `squench_comparisons/` checks
   every HDF5 dataset between two runs. Baseline = NX=32 run of SuperQuench
   vs. SuperfluidGPU launched through its own `grunner.prlmrun`.
2. **Profile** the CPU code at NX=256/512 with `-log_view`: split between
   `IdealStep`, `ExplicitDiffusion`, and ghost exchange.
3. **Build PETSc+Kokkos+CUDA** spack env at
   `$CFS/m3722/opt/prlm/petsc-gpu-int64` (`+cuda +kokkos +kokkos-kernels
   +openmp +int64 +hdf5 cuda_arch=80`).
4. **Kokkos port** of the two hot loops: `IdealPV2::step_no_reject` and
   `ModelGExplicitDiffusionStep::step`.

Open: `NoiseGenerator` retention — revisit once we're confident nothing new
needs an RNG. Missing `SuperSplitStep` JSON block throws in `Run`
(`inputs["SuperSplitStep"]` is null); harmless while `grunner` always
writes it.
