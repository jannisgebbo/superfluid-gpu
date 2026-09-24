# SuperfluidGPU — Plan

Shared working notes. Either of us can edit. Committed to the repo so history
is preserved alongside the code.

## Goal

Fast superfluid mode on GPU via PETSc + Kokkos. **Target: one Perlmutter
GPU node (4 × A100).** That fits about 960³ on 40 GB nodes and 1024³ on 80 GB
nodes (`-C gpu&hbm80g`). A very fast 512³ already counts as success. The grid
takes 80 B per site, and we keep two copies (`solution` + the ghosted local
vector), so 512³ = 21.5 GB fits on a single 40 GB A100.

Milestones: (1) 1 GPU / 1 rank, validated against the CPU code within a
tolerance (nvcc fuses multiply-adds, so bit-identity is not expected);
(2) 512³ on 1 GPU, profiled with the `Profiling.h` stages; (3) 4 GPUs on one
node, with GPU-aware MPI over NVLink (`MPICH_GPU_SUPPORT_ENABLED=1`) and a
GPU branch in `grunner.prlmrun`; (4) multi-node only if wanted.

Rule for the port: every loop that runs per substep must be a Kokkos
kernel (`DMDAVecGetKokkosOffsetViewDOF`). A host `DMDAVecGetArray` on a
Kokkos vector forces a full device↔host copy. First milestone:
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
- CPU profile (`squench_comparisons/profile.py`, 1 node = 128 ranks,
  `-log_view`): NX=256 takes 0.28 s/step, NX=512 takes 2.23 s/step. Both come to
  about 2.1 µs per site per step per core. Ghost exchange (`VecScatterBegin` +
  `End`) is 53–56% of the run. The exchange cost grows with volume (8× from 256
  to 512), not with face area (4×). So the full owned-block copy inside
  `DMGlobalToLocal` is probably what costs, not the MPI traffic. Fewer exchanges
  per step is the lever. At the time there were about 9.7 per step: 1 for the A
  step, 6 for the B steps, 2 for energy monitoring, and about 0.3 for
  measurements.
- Energy moved out of `IdealPV2` (it computed E before and after every A step
  with 2 extra ghost exchanges, and nothing read it) into
  `Measurer::computeEnergy`. That reuses the ghosted vector `computeSliceAverage`
  already fills, so it costs no extra exchange. It is written as the `energy`
  HDF5 dataset. H is the effective H·σ/f in superfluid mode, the same as the
  ideal step. Regression vs. SuperQuench is still bit-identical (`compare.py`
  treats `energy` as new-only).

- Instrumented profile (NX=256, 128 ranks, job 58836344; `Profiling.h`
  stages and events). The whole step is **memory-bandwidth bound**:
  - A reference `VecCopy` of the solution takes 11.6 ms (about 230 GB/s for
    the node). One `GhostExchange` takes about 18.5 ms, and `VecScatterBegin`
    alone is about 13.6 ms. So roughly 2/3 of each exchange is the local copy
    of the owned block; face packing (~3 ms) and MPI waiting (~5 ms) are the
    rest.
  - `DiffusionStencil` (14 ms/call), `IdealKick` (14 ms) and each
    `IdealRotate` (11 ms) also run at about 250–300 GB/s node-wide; the flop
    rates are low.
  - Time split: B substeps 70% (exchange 59%, stencil 46% of B, overlapping
    in the event nesting), A substeps 18% (exchange 3.7 s, rotate 4.4 s,
    kick 2.7 s), measurements 7%.
  - Levers, all about fewer full-grid memory sweeps: stop copying the owned
    block every substep (ping-pong ghosted buffers, update only the ghosts),
    fuse rotate+kick+rotate, and fuse B substeps.

## Superfluid-mode pathway (what we kept)

The code is always in superfluid mode; the `superfluidmode` flag and its
branches were removed. `evolverType = "SuperSplitStep"` is the only stepper;
`grunner.checkinputs()` asserts it.

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
- JSON in `nlohmann/` (`make_unique.h` removed; using `std::make_unique`)

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
             ├── 'A' → IdealPV2::step             (deterministic)
             └── 'B' → ModelGExplicitDiffusionStep::step   (deterministic)
```

## Next

1. ~~Build PETSc+Kokkos+CUDA~~ **done (2026-09-24)**, from source, not spack:
   PETSc 3.25.5 + Kokkos 5.1.0 + CUDA 13.2 at
   `$CFS/m3722/opt/prlm/petsc-gpu-int64`. See
   `platforms/perlmutter/Perlmutter.md`. `tests/kokkos_smoke.exe` (DMDA on
   Kokkos vectors, device kernel, periodic ghost exchange) passes on 1 and 4
   GPUs. Build with `cmake -B build-gpu -DSFGPU_KOKKOS=ON` after sourcing
   `setupprlm-gpu.sh`. Multi-rank runs need `--gpu-bind=none`.
   **Kokkos vectors alone** (`-dm_vec_type kokkos`, no code change) give
   bit-identical output with the host loops unchanged; they are slower
   because of host↔device syncs.
   **Bit-for-bit checking of the port:** the GPU build uses
   `-ffp-contract=off -fmad=false` (no fused multiply-add on host or device).
   Without it the Cray CC wrapper's `-march=znver3` alone changes the last
   bits. Test: NX=32, 4 ranks, one GPU node: (a) CPU build, (b) GPU build,
   (c) GPU build with `-dm_vec_type kokkos`. All three are bit-identical
   (work area `squench_comparisons/kokkosvec/`).
   **Diffusion stencil ported (done):** `diffusion_stencil_kokkos` in
   Stepper.cxx runs when the solution is a Kokkos vector. Output is
   bit-identical to the host loop. At NX=32 the stencil is 10× faster, and
   the B-stage uploads drop from 756 to 126 (one per step, after the host A
   step).
2. ~~Kokkos port of the A step~~ **done**: `ideal_step_kokkos` (first
   half-drift kernel, then kick fused with the second half-drift).
   `O4AlgebraHelper` functions and `rotate_by_charges` are shared by host
   and device. Against the CPU build (NX=32, 4 ranks, 126 steps) the output
   agrees to roundoff (≤ 4e-16 absolute, energy 2.5e-16 relative). It is
   not bit-identical only because CUDA's sin/cos differ from glibc. **The
   stepping path now has zero host↔device transfers.** At NX=32 the kernels
   are 10–30× faster and the total is exchange-latency bound.
3. Move the measurer's slice averages to Kokkos once host copies show up in
   the GPU profile. Measurement cost (6–8% on CPU) is otherwise not a
   concern; measure every 6–10 steps via `saveFrequency` if needed.
4. GPU branch in `grunner.prlmrun` (`-C gpu`, `-A m3722_g`, 4 ranks,
   `--gpu-bind=none`).

Open: `NoiseGenerator` retention — revisit once we're confident nothing new
needs an RNG. Missing `SuperSplitStep` JSON block throws in `Run`
(`inputs["SuperSplitStep"]` is null); harmless while `grunner` always
writes it.
