# SuperfluidGPU — Plan

Shared working notes. Either of us can edit. Committed to the repo so history
is preserved alongside the code.

## Goal

Reach superfluid mode on GPU at 1024³ via PETSc + Kokkos. First milestone:
strip this repo down to only the **superfluid-mode, noise-free** pathway so
the surface area we port to Kokkos is minimal.

Seed state: identical to `SuperQuench@c137ab88` (single commit). The two repos
are allowed to diverge.

## Superfluid-mode pathway (what to keep)

Runtime selection: `superfluidmode = true` in JSON config
([ModelA.h:154](../src/ModelA.h#L154), read at line 172).
Constraint enforced in [scripts/grunner.py:91-95](../scripts/grunner.py#L91-L95):
`superfluidmode=true` **requires** `evolverType = "SuperSplitStep"`.

The active call graph in superfluid mode:

```
SuperPions.cxx::Run  ── etype == "SuperSplitStep" branch ──▶ SuperSplitStep
SuperSplitStep::step (Stepper.h:231)
  ├── 'A' → IdealPV2::step               (deterministic Verlet)
  └── 'B' → ModelGExplicitDiffusionStep  (deterministic; or implicit variant)
```

Neither leg calls `NoiseGenerator` for anything that feeds back into the
fields. `IdealPV2` calls `ModelARndm->uniform()` once per step for a
Metropolis accept/reject that only affects statistics
([Stepper.cxx:45](../src/Stepper.cxx#L45)) — the trajectory itself is
deterministic.

**Keep (used by the superfluid pathway):**

- `SuperPions.cxx` — main + `Run` (only the `SuperSplitStep` branch is live)
- `Stepper.{h,cxx}` — `Stepper` base, `IdealPV2`, `ModelGDiffusionStep`,
  `ModelGExplicitDiffusionStep`, `SuperSplitStep`
- `ModelA.h` — all data structures, initialization, coefficients
- `initialize.{h,cxx}` — shared initialization helpers
- `O4AlgebraHelper.h` — used by `IdealPV2`
- `measurer*.{h,cxx}`, `ntuple.h`, `nvector.h` — measurement + HDF5 output
- `make_unique.h`, JSON in `nlohmann/`

## What runs only in the noisy / non-superfluid pathway (removal candidates)

All of the following are reachable only through `PV2HBSplit` (i.e.
`evolverType ∈ {"PV2HBSplit23", "PV2HBSplitGeneral"}`), which the
superfluid-mode guard forbids:

| Symbol | Location | Why removable |
|---|---|---|
| `EulerLangevinHB` | [Stepper.h:124-136](../src/Stepper.h#L124-L136), [Stepper.cxx:285-432](../src/Stepper.cxx#L285-L432) | φ-field heat-bath with noise; only called by `PV2HBSplit` |
| `ModelGChargeHB` | [Stepper.h:140-153](../src/Stepper.h#L140-L153), [Stepper.cxx:477-567](../src/Stepper.cxx#L477-L567) | charge heat-bath with noise; only called by `PV2HBSplit` |
| `modelg_update_charge_pair` | [Stepper.cxx:453-475](../src/Stepper.cxx#L453-L475) | helper for `ModelGChargeHB` |
| `g_face_case` / `g_face_cases` | [Stepper.h:156-165](../src/Stepper.h#L156-L165), [Stepper.cxx:442-450](../src/Stepper.cxx#L442-L450) | data for `ModelGChargeHB` |
| `PV2HBSplit` | [Stepper.h:265-318](../src/Stepper.h#L265-L318), [Stepper.cxx:858-916](../src/Stepper.cxx#L858-L916) | the non-superfluid stepper itself |
| `Run` non-superfluid branches | [SuperPions.cxx:203-229](../src/SuperPions.cxx#L203-L229) | `PV2HBSplit23`, `PV2HBSplitGeneral`, standalone `ModelGDiffusionStep` branches |
| grunner.py non-superfluid config | [scripts/grunner.py:52-57, 97-101](../scripts/grunner.py#L52-L57) | `pv2hb_split_general` block + validation for non-superfluid |

## Phase 4 — coefficient cleanup (uncommitted)

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

## Phase 3 — ModelA.h cleanup (uncommitted)

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

## Phase 1 + 2 — done (uncommitted)

Removed:
- `EulerLangevinHB`, `ModelGChargeHB` (+ helpers `modelg_update_charge_pair`,
  `g_face_case`, `g_face_cases`), `PV2HBSplit`, `o4_stepper_monitor`
- `thermalize_event` (function + both call sites in `initialize_event`).
  `default` init is now a cold start with no thermalization; `quench_mode`
  sets the quench mass and initializes at it but no longer thermalizes.
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
  `superfluidmode` → `True`, added `SuperSplitStep` block with
  `step_sequence: "AB"`.

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

## Still to decide

- **Reference config + regression check** — build the current tree on
  Perlmutter, run a small case (e.g. NX=32), and save its trajectory as a
  regression baseline before we start the Kokkos port.
- **`NoiseGenerator` retention** — revisit once we're confident nothing new
  needs an RNG.

## Phase 3 (proposed)

- Kokkos port of the surviving hot loops: `IdealPV2::step_no_reject` and
  `ModelGExplicitDiffusionStep::step`.
