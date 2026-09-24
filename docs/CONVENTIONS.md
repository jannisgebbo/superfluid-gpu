# SuperfluidGPU — Conventions

Physics-side conventions that used to be tribal knowledge in SuperQuench.
Recorded here so we (and future us) don't have to re-derive them.

Reference runs: `/global/cfs/cdirs/m3722/DT/squench/`. These were kept
because they're good. **Ignore `/pscratch/sd/d/dteaney/squench/`** — that
tree is exploratory / tests.

## Parameter defaults (superfluid runs)

| Parameter | Value | Reason |
|---|---|---|
| `gamma` (Γ) | **0.25** | Sets the explicit-diffusion timestep bound; see stability. |
| `diffusion` (D) | **Γ / 3** | The ratio D = Γ/3 is always kept fixed. Never treated as an independent knob. |
| `deltat` | **0.5** | Sweet spot given the subcycled step sequence. |
| `f2_constant` (f²) | **5.0** | Superfluid stiffness. |
| `chi` (χ) | **5.0** | Chosen with f² so `v² = f²/χ = 1`, making the ideal-step CFL bound reduce to `dt < a`. |
| `evolverType` | `"SuperSplitStep"` | Only stepper in this build. |
| `SuperSplitStep.step_sequence` | `"ABBBBBB"` | 1 ideal + 6 diffusion substeps per outer `dt` (effective diffusion dt = dt/6). |
| `superfluidmode` | `true` | `grunner.checkinputs()` asserts this. |
| `eventmode` | `true` | Runs are ensembles; typically `nevents = 2`. |

Implicit relation the reference runs also use (hardcoded in `squench.py`,
not a code default):

```
σ² = 0.9 · f²    →    σ = √(0.9 · f²) ≈ 2.121
```

## Mass / H convention

External field `H` is derived from a dimensionless mass label `mlabel`
(mass in units of the smallest lattice momentum):

```
mk = 2π / NX · mlabel
H  = mk² · f² / σ                (with σ = √(0.9 · f²))
```

The output filename tag encodes `int(1000·mlabel)` — e.g. `m_2000` means
`mlabel = 2.000`.

## Naming convention (`hquench_*` runs)

```
hquench_g_{int(1000·γ):04d}_N{NX:04d}_m_{int(1000·mlabel):03d}
```

Example: `hquench_g_0250_N0512_m_2000` → γ=0.250, NX=512, mlabel=2.000.

Each launch produces several sbatch jobs, tagged with a random 8-hex
UUID appended to the run tag (`grunner.prlmrun(nodeid=True)`).

## Stability bounds

The bounds themselves live in comments in `src/Stepper.cxx`. Recap:

| Step | Bound | With defaults |
|---|---|---|
| Ideal (Verlet) | `v · dt < a` with `v² = f²/χ = 1` | `dt < 1` (weak) |
| Explicit diffusion, phi | `dt · Γ < 1/12 ≈ 0.083` | Binding constraint |
| Explicit diffusion, charge | `dt · D < 1/12` | Weaker than Γ bound since D = Γ/3 |

**With the `ABBBBBB` subcycling** the effective diffusion timestep is `dt/6`,
so the binding constraint becomes `(dt/6) · Γ < 1/12`, i.e.
`dt · Γ < 1/2`. At the defaults `dt = 0.5, Γ = 0.25`: `dt·Γ = 0.125`,
well inside.

## Reference-run layout

Per run in CFS:
- `<tag>_<uuid8>.json` — input
- `<tag>_<uuid8>.sh` — sbatch script
- `<tag>_<uuid8>_time.out` — wall-clock timing
- `<tag>_<uuid8>_NNNN.h5` — event output (`NNNN` = event index, since
  `eventmode=true`)

**No `_save.h5` snapshots exist.** Initial conditions are generated fresh
per run via `initialization = "randomspins"` with a random seed drawn at
launch time. To reproduce a specific reference run, feed its exact `.json`
to SuperfluidGPU — same seed + same RNG (both codebases use
`xoroshiro128plus`) → same trajectory (up to MPI reduction order).

## Old JSON compatibility

Reference JSONs contain fields we've since deleted (`mass0`, `dmassdt`,
`lambda`, `restart`, `quench_mode*`, `thermalization_time`,
`pv2hb_split_general`, `SuperSplitStep.use_implicit_step`). nlohmann
`.value()` silently ignores unknown keys, so the old JSONs parse and run
against SuperfluidGPU without modification.

## Generator script

`/pscratch/sd/d/dteaney/squench/squench.py` — the launcher used for the
reference runs. Imports SuperQuench's `grunner`, sets the parameters
above, sweeps a mass list, and calls `grunner.prlmrun()` per launch.
