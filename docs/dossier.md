# Project Dossier: Stochastic Proportional Fairness — Refactor & Formulation B Pilot

This document is a running technical record of everything done to this repository
across the modularization, repo cleanup, and Formulation B pilot work. It is meant
to let anyone (including future-you) pick the project back up without re-deriving
the reasoning from scratch. For a quick orientation, read the root `README.md`
first; come here for implementation-level detail, rationale, and the numerical
issues that were actually hit and fixed.

---

## 1. Where this started

The project began as a single monolithic notebook, `optModels_final.ipynb`, that
built and solved a stochastic proportional-fairness resource-allocation model
(a time-dependent interdependent network design problem, "tdINDP") for three
infrastructure networks (Gas, Power, Water) under power-utility fairness, using
Gurobi. It contained, inline, the network/parameter loading, the EV (expected
value), EEVS (expectation of the expected value solution), WS (wait-and-see),
and DE (deterministic equivalent) model formulations, plus a separate plotting
notebook (`plotter.ipynb`) that visualized results — largely by copy-pasting
numbers printed to stdout during a run.

## 2. Modularization

`optModels_final.ipynb` was refactored into standalone, importable Python
modules under `src/`, so the notebook could stay as a thin "assemble and run"
entry point instead of holding all the modeling logic inline. Each classical
model got its own file:

| File | Responsibility |
|---|---|
| `src/tdINDP_parameters_build.py` | Loads/derives all static network parameters (nodes, arcs, costs, capacities, supply/demand, interdependency structure). |
| `src/svi_data.py` | The `SVI` dictionary: `SVI[k][i] = [l_i, u_i]`, the per-demand-node uniform range for the random fairness exponent `alpha_i`. |
| `src/gurobi_utils.py` | Shared Gurobi parameter configuration (`configure_common_gurobi_params`) and results post-processing/CSV export helpers, reused by every model file so solver settings live in exactly one place. |
| `src/paths.py` | Centralizes the `results/` output directory so scripts/notebooks write to a predictable location regardless of current working directory. |
| `src/ev_eevs_model.py` | `EV_and_EEVS(...)`: builds/solves the EV model (population or sample-mean `alpha`), then the EEVS per-scenario re-solves. |
| `src/ws_model.py` | `WS(...)`: wait-and-see, one fresh model solved per scenario with its own realized `alpha`. |
| `src/de_model.py` | `DE(...)`: deterministic equivalent, all scenarios in one model, weighted by `rho`. |
| `src/run_experiments.py` | Orchestrates a full run across scarcity levels: loads parameters, samples `alpha`, calls EV/EEVS/WS/DE, computes EVPI/VSS, writes CSV/TXT/MD summaries. |

`notebooks/optModels_final.ipynb` now just imports from `src/` and calls
`run_experiments.run_all()` (or equivalent) rather than containing the model
code inline.

## 3. Repo cleanup / restructuring

The repo was reorganized into clearly separated top-level folders:

- `src/` — all Python modules (see table above).
- `notebooks/` — the two active notebooks (`optModels_final.ipynb`,
  `plotter.ipynb`).
- `results/` — default output location for every run (`DE_*`, `WS_*`,
  `EEVS_*`, `LOG_FILE_*`, `scarce_*` CSV/TXT/MD files, Formulation B pilot
  outputs). Contents are git-ignored (`.gitignore` excludes `/results/*`
  except its own `README.md`) so the repo doesn't accumulate result churn;
  regenerate by re-running the experiments.
- `old_files/` — retired/superseded notebooks and archived result sets that
  were not deleted outright (in case of future reference) but are no longer
  part of the active workflow: an old single-scarcity-level plotter and
  earlier archived result exports.
- `docs/` — this dossier (new).

## 4. Direction-exclusivity refactor: `i < j` → `i != j`

### What existed before
Every model file that needed to prevent flow from moving in both directions
of the same node pair simultaneously (`x_ij > 0` and `x_ji > 0` at once) used
a single shared binary `z` per *unordered* pair, only created when `i < j`
(originally a numeric/lexicographic ordering trick), with the exclusivity
constraint `z_ij + z_ji <= 1` guarded by that same `i < j` condition so it
was only added once per pair.

### Why it was changed
The user asked for this ordering trick to be removed everywhere it appears,
replacing the guard with `i != j` — i.e., stop relying on an arbitrary
ordering of node identifiers (which for this codebase happen to be strings
like `"Power_node_44"`, making the "ordering" incidental/string-lexicographic
rather than meaningful) to decide which direction owns the shared binary.
Explicit instruction: **do not use `i < j` or `j < i` — compare with
`i != j`.** This means the pairwise constraint may now be added twice (once
from each direction's iteration), which is accepted as harmless redundancy
(same constraint, added under two different constraint names) rather than a
bug to optimize away.

### What changed, file by file
- **`src/de_model.py`** — `i < j` → `i != j` in the pairwise exclusivity
  constraint guard (2 occurrences). Variable creation (`z`, `z_star`) was
  already using per-direction binaries from prior work; only the guard
  changed.
- **`src/ws_model.py`** — same `i < j` → `i != j` fix (2 occurrences), same
  reasoning.
- **`src/ev_eevs_model.py`** — this file's direction-exclusivity block had
  **not actually been updated** in an earlier pass despite being assumed
  done; it still had the fully old `i < j`-only pattern and, critically, had
  **no `z_star` for `x_star` (first-stage supply arcs) at all**. Rewrote both
  the EV block (~lines 78–120) and the EEVS per-omega block
  (~lines 226–252): added `z_star` for `x_star` and `z` for `x`, each gated
  by `x_star[(i,j,k,l)] <= u[(i,j,k)]*z_star[(i,j,k,l)]` /
  `x[(i,j,k,l,omega)] <= u[(i,j,k)]*z[(i,j,k,l)]`, plus the pairwise
  `z_ij + z_ji <= 1` constraint guarded by `i != j`.
- **`src/formulation_b_model.py`** — same treatment; this file had never had
  `z_star` for `x_star` at all (Formulation B is newer), so it was added for
  the first time here, with the `i != j` guard from the start.
- **`src/compare_formulations.py`** — `solve_formulation_a_ev`'s
  direction-exclusivity block was rewritten identically, to keep this
  function's model construction in lockstep with `ev_eevs_model.py`'s EV
  block. (This function exists as a deliberate duplicate of the EV solve —
  see Section 6 for why.)

## 5. Formulation B: log-of-expected-utility pilot

### 5.1 Motivation and math

Formulation A (the paper's primary, unchanged result) solves:

```
max  sum_i  E[alpha_i] * ln(phi_i)
```

This is exact only because `ln(phi_i)` is deterministic (`phi_i` is a decision
variable), so `E[.]` and `ln(.)` trivially commute. Formulation B instead
solves the *log of the expected* utility:

```
max  sum_i  ln( E[ phi_i ** alpha_i ] )
```

These are not equivalent once `alpha_i` is random: by Jensen's inequality
(`ln` is concave), `E[ln(u)] <= ln(E[u])`, with equality **only** when
`alpha_i` is deterministic. Formulation B is sensitive to the *width* of the
uncertainty in `alpha_i`, not just its mean.

For `alpha_i ~ Uniform(l_i, u_i)`, the closed form is:

```
E[phi_i ** alpha_i] = (phi_i**u_i - phi_i**l_i) / ((u_i - l_i) * ln(phi_i))     (phi_i > 0, phi_i != 1)
```

When `u_i - l_i` is (numerically) zero, this is a removable 0/0 singularity
whose true limit is `phi_i**l_i` — computed directly rather than through the
formula.

### 5.2 Gurobi construction (`src/formulation_b_model.py`)

Per demand node `i`, built with Gurobi *general constraints* (not raw Python
`**`/`/` inside a linear expression — Gurobi cannot evaluate those):

1. `p_hi = phi_i ** u_i`, `p_lo = phi_i ** l_i` via two `addGenConstrPow` calls.
2. `log_phi_i` via `addGenConstrLog(phi_i, log_phi_i)` (same pattern already
   used in Formulation A).
3. `expected_util_i`, linked via the bilinear equality
   `expected_util_i * (u_i - l_i) * log_phi_i == p_hi - p_lo` — deliberately a
   *multiplication* (`model.addConstr(...)`) rather than a division by
   `log_phi_i`, to sidestep the `phi_i = 1` singularity (`log_phi_i -> 0`)
   where a division would blow up. This is why `NonConvex=2` is required —
   this equality is a genuine bilinear/trilinear nonconvex constraint. (This
   Gurobi version has no `addGenConstrNL`; `NonConvex=2` is the documented
   fallback per the experiment spec.)
4. `log_expected_util_i` via `addGenConstrLog(expected_util_i, log_expected_util_i)`.
5. Objective: `max sum_i log_expected_util_i`.

Degenerate nodes (`u_i - l_i < WIDTH_EPS = 1e-6`) skip all of the above and
set `log_expected_util_i == l_i * log_phi_i` directly — mathematically
identical to Formulation A's term for that node. When *every* node is
degenerate, Formulation B reduces exactly to Formulation A; this is the basis
of the Stage 1 correctness check (Section 5.4).

Numerical hazard handling:
- `expected_util_i` is floored at `EXPECTED_UTIL_LB = 1e-6` (Section 4 of the
  spec).
- Every node whose optimal `phi_i` lands within `singularity_band` (default
  `1e-2`) of 1 is logged/reported (`result['singularity_nodes']`), per the
  spec's "log every instance where phi_i lands near 1" requirement, so we
  know empirically whether the hazard is theoretical or actually bites for
  this dataset.
- Gurobi params: `NonConvex=2`, `FuncPieceError=1e-4` (via the shared
  `configure_common_gurobi_params`), `NumericFocus=3`.

### 5.3 Warm start + relaxed MIP gap

Per user direction ("add a warm start as the solution of the EEV solution to
find an incumbent, and set MIPGap to 1% — don't stress about closing to
1e-4"):

- `solve_formulation_b` gained a `warm_start` parameter: a `varName -> value`
  dict (typically `solve_formulation_a_ev(...)['var_values']`, i.e.
  Formulation A's own EV solution on the same instance).
- Shared-name variables (`x_star`, `w`, `y`, `delta_plus`, `phi`,
  `delta_minus`, `x`, `z_star`, `z`) are set via a generic name-matching pass
  before `model.optimize()`.
- Formulation-B-only chain variables (`log_phi`, `p_hi`, `p_lo`,
  `expected_util`, `log_expected_util`) get `.Start` values **derived
  analytically** from the warm-started `phi` (i.e., computed in Python from
  the closed-form formula, not left to Gurobi to infer), so the MIP start is
  a complete, consistent point rather than a partial one.
- Default `mip_gap` changed from `None` (solve to Gurobi's default gap) to
  `0.01` (1%).
- `solve_formulation_a_ev` (in `compare_formulations.py`) was extended to
  also return `'var_values'` (the full `varName -> value` dict) specifically
  so it can be passed straight into `warm_start`.

### 5.4 Pilot driver (`src/run_formulation_b_pilot.py`)

Two stages, matching the spec's incremental pilot plan:

- **`run_stage1_degenerate_check`** — forces `l_i == u_i` everywhere (or for
  `keep_networks` only), confirms Formulation B's objective and `phi_i`
  match Formulation A's EV solution using that same degenerate value as
  `alpha_mu` (the Jensen equality case). `tol=2e-3` accounts for expected
  PWL-approximation noise from Gurobi's piecewise-linear `ln(phi)` summed
  across ~89 demand-node log terms.
- **`run_full_case_study`** — the real, full-scale run: real SVI-based
  uncertainty over the *original* problem (all three networks by default —
  the user explicitly said not to restrict to mini networks: "idc you
  running it I'll run it myself and wait for results myself, just give me
  functional code"). `time_limit=None` by default (no artificial truncation
  on the full nonconvex case study).

### 5.5 Standing default: `scarce = 0.75`

Per explicit correction from the user ("scarce cannot be 1, it has to be that
the total resource available is 0.75 of all available network... I just want
that value for all experiments going forward"), `scarce=0.75` is now the
default scarcity level threaded through: `run_experiments.run_all`'s default
`scarcity_levels=(0.75,)`, and every Formulation A/B pilot entry point's
`scarce` default.

## 6. EVPI/VSS analogs for Formulation B

### 6.1 Question: does Formulation B have its own EEVS/EVPI/VSS?

Not directly. Formulation A's stack (`run_experiments.py`) computes classic
two-stage stochastic programming metrics:

- `EVPI = OF_WS - OF_DE` (expected value of perfect information)
- `VSS = OF_EEVS - OF_DE` (value of the stochastic solution)

Formulation B has **no `OMEGA`/recourse loop at all** — `alpha` is integrated
out analytically into a single closed-form objective. So "DE for B" doesn't
exist as a distinct model, and a degenerate-SVI "EV for B" collapses back to
plain Formulation A rather than being a new object.

### 6.2 What *does* transplant cleanly: WS

WS (wait-and-see: solve each scenario with its own realized `alpha`, average
the results) is **shared** between A and B. Once you know `alpha` exactly for
a scenario, there is no aggregation-rule ambiguity left — Jensen's gap is a
property of *uncertainty*, and perfect per-scenario information collapses it
regardless of whether you'd have used A's or B's aggregation rule. So
`ws_model.WS(...)` is reused as-is for both formulations, with no new model
code.

### 6.3 The cheap metrics actually built

Implemented in `run_formulation_b_pilot.run_full_case_study` under
`compute_de_swap_metrics=True` (default): Formulation B's own objective
stands in for "DE" in the classic formulas, reusing existing WS/EEVS models
rather than building new ones ("swapping DE in A for B" per the user's
framing):

```
EVPI_B = OF_WS - OF_B
VSS_B  = OF_B  - OF_EEVS_alpha_mu_hat
```

where `OF_EEVS_alpha_mu_hat` is the sample-mean EEVS solve
(`ev_eevs_model.EV_and_EEVS` with `alpha_mu_type=alpha_mu_hat`, the
sample-mean `alpha`) already computed for Formulation A on the same instance.
This adds one real WS solve and one real EEVS solve (one Gurobi model per
scenario each) on top of the single Formulation B solve; set
`compute_de_swap_metrics=False` for a quick Formulation-B-only run.

Results are printed and saved to
`results/formulation_b_full_case_study_<network>_scarce_<scarce>_summary.json`.

## 7. Automating Formulation A's summary metrics (no more copy-pasting)

`notebooks/plotter.ipynb`'s cell 6 previously had a **hardcoded**
`summary_stats` dict — numbers manually copy-pasted from a run's stdout
output, confirmed by the user's own suspicion and verified via grep. This has
been automated:

- **`src/run_experiments.write_summary_json`** — writes a machine-readable
  `results/scarce_<scarce>_summary.json` alongside the existing `.txt`
  summary, containing `OF_EV_*`, `OF_EEVS_*`, `OF_WS`, `OF_DE`, `EVPI`,
  `VSS_alpha_mu`, `VSS_alpha_mu_hat`. Wired into `run_scarcity_level` so
  every scarcity-level run produces this file automatically, no manual
  transcription required.
- **`plotter.ipynb` cell 6** rewritten: loads `summary_stats` by globbing
  `results/scarce_*_summary.json`; falls back to the old hardcoded values
  (with an explicit warning printed) only if no JSON files exist yet (e.g., a
  fresh checkout before any run has been executed).
- **`plotter.ipynb` new cell 7**: loads
  `formulation_b_full_case_study_*_summary.json` files and bar-charts
  `EVPI_B` / `VSS_B` per (network, scarce); prints a message if none exist
  yet rather than erroring.

## 8. The EEVS infeasibility bug (found, diagnosed, and fixed)

### 8.1 Symptom

Running the real pipeline end-to-end (`run_experiments.run_all()`,
`scarce=0.75`, `num_scenarios=150`, population-mean `alpha_mu` from `SVI`
midpoints) crashed on the **very first EEVS scenario** with:

```
AttributeError: Unable to retrieve attribute 'x'
```

### 8.2 Root cause, as actually diagnosed

Reproduced directly (bypassing `OutputFlag=0` via a monkey-patch to force
verbose Gurobi output) and confirmed the underlying Gurobi model was reported
**"Model is infeasible"** — the `AttributeError` was just the symptom of code
that unconditionally read `var.x` for every variable
(`{var.varName: var.x for var in model.getVars()}`) without checking
`model.SolCount`/`model.status` first.

`computeIIS()` on the failing model (population-mean alpha, `scarce=0.75`,
scenario 1 of 150) isolated the conflict to a single tight arc pair in the
Power network: node `Power_node_44` receives its entire inflow from a single
supply arc `(Power_node_6, Power_node_44)` that is at **exactly** its
capacity in the EV solution, and passes almost all of it downstream through
arc `(Power_node_44, Power_node_68)`, which is *also* at exactly its
capacity. The EV model's own solved value for `x_star_(6,44)` sits about
`6e-9` *above* the arc's stated capacity `u` — completely normal LP solve
noise, absorbed by Gurobi's own (much looser) default `FeasibilityTol=1e-6`
within the EV solve itself.

The EEVS per-scenario block fixes first-stage `x_star` to the EV solution's
**exact** floating-point values via a hard equality constraint
(`model.addConstr(x_star[...] == x_star_EV[...])`). Re-applying that `6e-9`
residual as a hard equality on top of the *same* tight capacity constraint,
in a model that also needs simultaneous binary feasibility for the new
`z_star`/exclusivity variables, is enough to tip a razor-thin-feasible arc
into "infeasible" for that specific scenario. This was **not** a pre-existing
issue silently masked before — the earlier assumption (from an FYI-only
duplicate function, `compare_formulations.solve_formulation_a_ev`, which
never actually runs the EEVS loop) turned out to be wrong; the real
`ev_eevs_model.EV_and_EEVS` EEVS loop genuinely hits this on real data.

An intermediate fix attempt — additionally fixing `w`, `y`, `delta_plus`, and
`z_star` (all the true first-stage decisions, not just `x_star`, since none
of them have any per-scenario recourse in this model: `alpha` only ever
appears in the objective, never in a constraint) — was a methodologically
correct improvement (matches the standard EEVS definition: "fix first-stage
decisions, optimize recourse given the scenario") but **did not fully
resolve** the infeasibility on its own, because the exact-equality fixing of
the *continuous* first-stage values (`x_star`, `delta_plus`) was still
re-injecting the same `~6e-9` numerical residual.

### 8.3 The actual fix

In `src/ev_eevs_model.py`'s EEVS per-omega block:

1. **Fix all true first-stage decisions**, not just `x_star`: `x_star`,
   `delta_plus` (continuous), and `w`, `y`, `z_star` (binary).
2. **Continuous** first-stage values (`x_star`, `delta_plus`) are fixed
   within a small **tolerance band** (`fix_tol = 1e-6`, i.e.
   `ev_val - fix_tol <= var <= ev_val + fix_tol`) rather than via exact
   equality — absorbing ordinary LP solve noise from the EV solve without
   materially changing what "fixed to the EV solution" means.
3. **Binary** first-stage values (`w`, `y`, `z_star`) are 0/1-valued with no
   such noise, so they are `round()`-ed and fixed via exact equality.
4. A defensive check was added right after `model.optimize()`:
   ```python
   if model.SolCount == 0:
       raise RuntimeError(
           f"EEVS scenario {omega} (status {model.status}) produced no feasible "
           "solution when re-solved with the EV solution's first-stage decisions "
           "fixed. This should not happen since the EV solution itself is a feasible "
           "replay for any scenario (alpha only affects the objective, not "
           "constraints) -- check for numerical tolerance issues or a modeling bug."
       )
   ```
   so any future recurrence fails loudly and specifically instead of a raw,
   uninformative `AttributeError`.

### 8.4 Verification

Reproduced the failure standalone with the exact real-scale parameters
(`scarce=0.75`, `num_scenarios=150`, `seed=42`, population-mean `alpha_mu`
from `SVI` midpoints), confirmed via `computeIIS()` exactly which
constraints conflicted (see 8.2), applied the fix above, and re-ran the same
reproduction script to confirm scenario 1 no longer raises.

### 8.5 Not yet audited (left as an open follow-up)

- Whether `ws_model.py`/`de_model.py` have any analogous "fix first-stage
  values then re-solve/re-gate with fresh binaries" pattern that could hit
  the same class of numerical issue.
- The earlier Stage 1 phi-diff finding (~0.725 max `|phi_A - phi_B|` despite
  matching objectives to ~6e-4) was tentatively attributed to alternate
  optimal solutions introduced by the new `x_star` exclusivity constraints —
  this has not been re-examined in light of the EEVS fix above, since it's a
  different code path (Stage 1 uses `solve_formulation_a_ev`, not the full
  EEVS loop).

## 9. Gurobi solve diagnostics / audit trail

### 9.1 Gap identified

Formulation A's pipeline (`EV_and_EEVS`, `WS`, `DE`) only ever captured final
objective values and the resulting `phi`/`x` variable dicts — no Gurobi solve
diagnostics were recorded anywhere (not in the console beyond default Gurobi
log lines, which are themselves suppressed by `OutputFlag=0`, and not in any
saved file). That means, for any given run, there was no persisted record of:

- `Status` (optimal / time-limit / suboptimal / infeasible / numeric issues)
- `Runtime` (wall-clock solve time per model)
- `MIPGap` (relevant since these models all contain `w`, `y`, `z`, `z_star`
  binaries)
- `SolCount` (would have caught the EEVS "no incumbent" case in Section 8
  before it hit a raw `AttributeError`)
- Gurobi's own numerical-violation warnings (e.g. the ~4.7e-4 "max general
  constraint violation" warning seen while debugging Section 8, which
  previously vanished into a suppressed log)

Formulation B (`formulation_b_model.py`) already returns `status_name`,
`solve_time`, and `mip_gap` in its result dict, plus a list of
singularity-zone nodes — Formulation A had no equivalent instrumentation.

### 9.2 What was added

`gurobi_utils.py` gained a `collect_solve_diagnostics(model, label)` helper
that reads back, immediately after `model.optimize()`:

- `status` (numeric) and `status_name` (e.g. `OPTIMAL`, `TIME_LIMIT`,
  `SUBOPTIMAL`, `INFEASIBLE`)
- `runtime` (seconds)
- `sol_count`
- `obj_val`, `obj_bound`, `mip_gap` (`None` where not applicable, e.g. no
  bound for a pure LP)
- `num_vars`, `num_bin_vars`, `num_constrs`
- `work_units` (`model.Work`, if available)

`EV_and_EEVS`, `WS`, and `DE` were updated to call this after every
`model.optimize()` (EV solve, each EEVS-per-scenario solve, each WS-per-scenario
solve, and the single DE solve) and accumulate the results into a list of
per-model diagnostic records.

### 9.3 Reporting format: Markdown, not JSON

Per explicit instruction, these diagnostics are written as a **Markdown**
report, not JSON (the existing `scarce_*_summary.json` files are left
unchanged and still carry the objective-value summary for the plotter to
consume). New function `run_experiments.write_solve_diagnostics_md(...)`
writes `results/scarce_<scarce>_solve_diagnostics.md`: one table row per
Gurobi solve in that scarcity-level run (EV, both EEVS variants' per-scenario
solves, WS's per-scenario solves, DE), covering model label, scenario index
(where applicable), status, runtime, gap, objective, bound, and variable/
constraint counts, plus a short human-readable flag column (e.g. "⚠ hit time
limit", "⚠ no incumbent") so a reviewer can audit a run's numerical health at
a glance without re-running it.

This closes the reproducibility/audit gap: every full `run_all()` invocation
now leaves behind a persisted, human-readable record of exactly how each
Gurobi solve terminated, not just the final headline numbers.

### 9.4 Overwrite behavior on rerun, and why only `DE` has a raw Gurobi `LogFile`

Before rerunning experiments (e.g. after the supply-flow-balance fix in
Section 14), it's worth knowing exactly what gets cleanly replaced vs. what
doesn't:

- `results/scarce_<scarce>_..._.txt`, `..._summary.json`, and
  `..._solve_diagnostics.md` (`run_experiments.write_summary`,
  `write_summary_json`, `write_solve_diagnostics_md`) all use
  `open(filename, "w")` — a full overwrite every run, filenames unchanged.
- `results/{DE,EEVS_mu_hat,WS}_scarce_*.csv` (`save_results_csv`) — also a
  clean overwrite.
- `results/LOG_FILE_scarce_<scarce>_..._` — the one exception. This is a raw
  Gurobi console log, set only in `de_model.py` via `model.params.LogFile =
  ...`. Gurobi's `LogFile` **appends** to an existing file across separate
  process runs rather than truncating it, so without a fix, rerunning would
  silently concatenate old and new solver logs into the same file. Fixed by
  deleting any existing file at that path immediately before setting
  `LogFile` (`de_model.py`), so it's now also a clean overwrite per run.
- **Only `DE` sets a `LogFile` — this is intentional, not an oversight.** `DE`
  solves once for all scenarios in a single model, so one log file is
  meaningful and cheap. `EV` and `EEVS`/`WS` build a fresh model and call
  `model.optimize()` once per scenario (up to 150 times), so a per-scenario
  `LogFile` would mean either ~150 files of clutter or a shared file needing
  its own append-across-scenarios bookkeeping — not worth it, since the
  per-solve numbers that matter (status, runtime, gap, objective) are already
  captured by `collect_solve_diagnostics` into `solve_diagnostics.md` for
  every model, including EV/EEVS/WS's per-scenario solves. Decision: leave
  `LogFile` DE-only.

## 10. Key design facts worth remembering

- `x_star` (first-stage supply-arc flow) is never indexed by `omega`; `x`
  (second-stage demand/transshipment-arc flow) is indexed by `omega`
  wherever the model has per-scenario recourse.
- `alpha` (the random fairness exponent) only ever appears in the
  **objective**, never in any constraint, in this model family. This is why
  the same network/flow structure is feasible for every scenario regardless
  of which `alpha` realization is used — a fact that was central to
  diagnosing the EEVS bug in Section 8.
- `de_model.py` holds **all** scenarios in a single model (so its `z` for
  `x`-arcs needs `omega` baked into the dict key); `ws_model.py` and
  `ev_eevs_model.py`'s EEVS block build a **fresh** `Model()` per `omega`
  (so `z`'s dict key doesn't need `omega`, but constraint *names* do, to
  avoid duplicate-name warnings across the per-scenario rebuild loop).
- Gurobi version in use: 11.0.3. Shared solver settings
  (`gurobi_utils.configure_common_gurobi_params`): `OutputFlag=0` (override
  to `1` for debugging), `FuncPieceRatio=-1`, `FuncPieceError=1e-4`,
  `DualReductions=0`, `NumericFocus=3`, `IntegralityFocus=1`,
  `IntFeasTol=1e-9`. Note `FeasibilityTol` is left at Gurobi's default
  (`1e-6`) — not tightened — which is what makes the `~6e-9` LP noise from
  Section 8 normally invisible until it's re-injected as a hard equality.

## 11. Open items / suggested next steps

1. ~~Confirm the full `run_experiments.run_all()` completes end-to-end with
   the Section 8 fix in place.~~ **Done.** A full real-scale run
   (`scarce=0.75`, `num_scenarios=150`, seed=42, population-mean `alpha_mu`)
   completed with no exceptions in ~3288s: `OF_EV = 120.9486`,
   `OF_EEVS = 121.2022`. The fix holds under the real workload.
2. Audit `ws_model.py`/`de_model.py` for the same "fix-then-re-gate" pattern
   (see Section 8.5).
3. Re-examine the Stage 1 phi-diff finding once the EEVS path is confirmed
   stable, to decide whether the `2e-3` tolerance in
   `run_stage1_degenerate_check` needs adjusting or whether the alternate-optima
   explanation should be investigated further (e.g., by adding a
   tie-breaking secondary objective).
4. Once `run_all()` is confirmed stable, verify `plotter.ipynb`'s two
   auto-loading cells (Section 7) render correctly against real (not just
   smoke-tested) `results/*.json` output, and spot-check the new
   `scarce_*_solve_diagnostics.md` report (Section 9) against a real run.
5. **Formulation B does not converge at full 3-network scale.** A real run of
   `run_formulation_b_pilot.run_full_case_study` (`scarce=0.75`, real SVI
   uncertainty, all 3 networks, warm-started from Formulation A's EV
   solution, `NonConvex=2`, `MIPGap=0.01`, no time limit) explored 445,934+
   B&B nodes over 18,824s (~5.2h) with **zero incumbent solutions found** and
   an essentially flat best bound (255.667 -> 255.382). The MIP start was
   rejected twice ("did not produce a new incumbent solution"). Root cause
   diagnosed: the per-node warm-start Start values were internally
   *inconsistent* — `phi_i.Start` came from the raw (possibly exactly-0)
   Formulation-A value via a generic name-matching pass, while
   `log_phi_i.Start`/`p_hi.Start`/`p_lo.Start`/`expected_util.Start` were all
   derived from a *floored* copy of that same phi (`max(phi_i, 1e-9)`, to
   avoid `ln(0)`). That mismatch violates the `addGenConstrLog`/bilinear
   general constraints' Start pairing at every node where Formulation A's
   phi_i was 0, which is common under `scarce=0.75`. **Fixed** in
   `src/formulation_b_model.py`: `phi.Start` and `delta_minus.Start` are now
   set explicitly from the same floored `phi_warm` used for the rest of the
   chain, and the generic pass is updated to skip any variable name already
   given a consistent Start this way (`consistent_warm_names` set). Also
   added `MIPFocus=1`, `Heuristics=0.5`, `NoRelHeurTime=30` to prioritize
   finding *any* incumbent over closing the gap, given how weak the LP
   relaxation bound is on this model (89 quadratic + 316 SOS constraints
   after presolve). **Not yet re-verified against a real full-scale run** —
   this is a diagnosis-driven fix, not yet confirmed to resolve the
   non-convergence; the next full run should report whether an incumbent is
   now found early instead of never.

**Update (confirmed):** the fix worked, partially. On the user's actual re-run,
Gurobi accepted an incumbent immediately (`125.38488`, i.e. Formulation A's EV
solution scored under Formulation B's objective) instead of finding zero
solutions in 5+ hours. The Stage 1 degenerate-equivalence check still passes
cleanly. However, on the full 3-network case study the bound still moved only
`255.667 -> 255.011` after ~795s / 68,866 nodes (gap still ~103%), and the
incumbent never improved past the warm start. This confirms the remaining
problem is **structural** (a fundamentally loose LP relaxation from the
`log`/bilinear/SOS reformulation at full scale), not a warm-start or
parameter-tuning issue — throwing more time at Gurobi would not have closed
this gap in any practical timeframe. **Decision: stop trying to solve
Formulation B to proven optimality.** See Section 12 for what was done
instead.

## 12. Pivot: ex-post Jensen gap instead of solving Formulation B

### 12.1 The idea

Formulation B's objective is `sum_i log(E_omega[phi_i,omega ** alpha_i,omega])`
(log of expected utility) vs. Formulation A/DE's
`sum_i E_omega[alpha_i,omega * log(phi_i,omega)]` (expected log-utility). The
gap between these two — `log(E[phi**alpha]) - E[alpha*log(phi)]` — is exactly
a per-node instance of **Jensen's gap** for the concave function `log`. Rather
than re-solving a nonconvex MIQCP to get Formulation B's *own* optimal
allocation, we instead ask a much cheaper, still-meaningful question: **how big
is this Jensen gap when Formulation B's objective is evaluated ex-post on
Formulation A's existing DE solution?** This requires zero new Gurobi solves —
only reading the `phi_i,omega` values already saved in the DE results CSV and
evaluating an elementary expression per scenario.

This is mathematically legitimate (not an approximation) because `de_model.py`
genuinely gives each scenario its own `phi_i,omega` — even though `alpha` never
enters a constraint, the DE *objective* trades off allocation differently
per-scenario depending on that scenario's `alpha` draw, so `phi_i,omega` really
does vary across `omega` in the optimal DE solution. Averaging
`phi_i,omega ** alpha_i,omega` and `alpha_i,omega * log(phi_i,omega)`
separately across the 150 equally-weighted scenarios, then taking
`log(...)  - (...)`, is genuine per-scenario Monte Carlo evaluation, not a
proxy.

### 12.2 Implementation: `src/jensen_gap_from_de_csv.py`

- Reads an existing `results/DE_scarce_*.csv` directly — **no Gurobi, no
  re-solve, no dependency on `run_experiments.py`** (the user was explicit
  about not wanting to re-run the whole pipeline).
- Extracts per-node `phi_(...)` and `alpha_(...)` rows, distinguishing the
  150 per-scenario columns (`'0'`..`'149'`) from the appended summary-stat
  columns (`mean`, `std`, ...) via `.isdigit()`.
- For each node `i`: computes `E_omega[phi_i,omega ** alpha_i,omega]` and
  `E_omega[alpha_i,omega * log(phi_i,omega)]` (uniform `rho = 1/150`, floored
  at `1e-9` to avoid `log(0)`), then
  `node_jensen_gap = log(E[phi**alpha]) - E[alpha*log(phi)] >= 0`.
- `find_latest_de_csv()` auto-discovers the newest `DE_scarce_*.csv` in
  `results/`.
- Sums per-node gaps into a total: `OF_B_expost = OF_A_DE + total_gap`.
- Verification: `OF_A_DE_check` computed by this script from the raw CSV
  (`121.279165`) matches the pre-existing `scarce_0.75_..._summary.json`'s
  `OF_DE` value **exactly** — strong evidence the extraction/averaging logic
  is correct.

### 12.3 Result (scarce = 0.75, 150 scenarios)

**Post supply-flow-balance-fix rerun** (regenerated 2026-07-16, from the
corrected `DE_scarce_0.75_..._.csv` — see Section 14.4 for the full
stale-data catch and regeneration writeup; superseded the numbers below,
which were from the pre-fix run):

```
OF_A_DE (Formulation A, expected log-utility)   = 121.356
OF_B_expost (Formulation B objective, evaluated
             ex-post on the DE solution)         = 126.480
Jensen gap (log E[phi**alpha] - E[alpha log phi]) =   5.124
```

(Pre-fix numbers, kept for reference only — do not use: `OF_A_DE = 121.279`,
`OF_B_expost = 126.400`, `jensen gap = 5.12`. The fix changed the headline
totals negligibly at this precision, but see Section 14.4 for what *did*
change — which specific node has the peak dispersion in the Power network.)

The gap is strictly positive everywhere (as guaranteed by Jensen's inequality
for a concave function), and concentrates heavily at a handful of nodes with
the widest scenario-to-scenario dispersion in `phi` — see Section 13.2 for the
per-node breakdown, now visualized directly on the network maps.

Wired into `notebooks/optModels_final.ipynb` as a new final cell: imports
`jensen_gap_from_de_csv`, calls `find_latest_de_csv()` +
`jensen_gap_from_de_csv(csv_path, svi=SVI)`, prints the headline numbers, and
displays the top-10 per-node breakdown. Output also saved standalone to
`results/jensen_gap_expost_from_de.csv`.

## 13. Plotter overhaul: auto-detected scarcity + dispersion/Jensen-gap maps

### 13.1 What was wrong with `notebooks/plotter.ipynb`

- Cells hardcoded `scarce = [0.25, 0.5, 0.75, 1]` and loaded a
  `scarce_{level}_..._nat_log_linear_pieces_length_1.csv` file per level — a
  file that, under the *current* per-model CSV naming
  (`DE_scarce_*`/`EEVS_mu_hat_scarce_*`/`WS_scarce_*`), never actually existed;
  only `scarce=0.75` has been (re-)run since the refactor, so this errored
  immediately on any fresh checkout.
- Elaborate 4-panel EEVS/EVPI/VSS and Formulation-B EVPI_B/VSS_B plots — no
  longer the priority (VSS is very small at `scarce=0.75` and the user
  explicitly said not to prioritize these anymore).
- The existing Gas/Power/Water map cells relied on `SVI.png`, which was
  **missing from `notebooks/`** entirely (only present under
  `old_files/archived_results/latest_run_root/SVI.png`) — copied over to
  `notebooks/SVI.png` to unblock both the old and new map cells.
- The existing per-node dispersion visualization stacked 5 nested
  min/Q1/median/Q3/max circles per node, sized relative to demand — accurate,
  but requires visually comparing 5 circle radii per node instead of reading
  one number.

### 13.2 What was changed

1. **Auto-detected scarcity levels** (cell 2): now globs
   `results/DE_scarce_*_utility_type_power_num_scenarios_*.csv` and extracts
   the scarce value via regex instead of a hardcoded list, defaulting to
   `[0.75]` if nothing is found. Cell 3 loads directly from the `DE_scarce_*`
   CSV (the actual file that exists) instead of a nonexistent unprefixed
   `scarce_*.csv`.
2. **Fixed a latent single-scarcity-level bug**: several cells did
   `fig, axes = plt.subplots(num_scarcity_levels, 1, ...)` then indexed
   `axes[i]`/`axes[-1]` — this only works when `num_scarcity_levels > 1`;
   with exactly one level (our current reality), `plt.subplots` returns a
   bare `Axes`, not an array, and indexing it raises `TypeError`. Fixed by
   wrapping with `np.atleast_1d(axes)` right after creation in every affected
   cell.
3. **New cells** ("Dispersion & Jensen gap maps", right after the
   `proportional_fairness` boxplot, ahead of the now-deprioritized EEVS/EVPI/VSS
   section which is marked "legacy, lower priority" in a new markdown cell):
   for each network (Gas/Power/Water), plots the network map (reusing the
   `x_coord`/`y_coord`/`supply_nodes`/`transshipment_nodes`/`A_k`/`SVI.png`
   machinery already in cell 1) with **demand nodes colored by a single
   scalar metric** via a colorbar, instead of nested circles:
   - **Coefficient of variation** (`std/mean`) of each node's DE-solution
     `phi_i` across the 150 scenarios — a direct, standard dispersion measure.
   - **Ex-post Jensen gap per node**, read straight from
     `results/jensen_gap_expost_from_de.csv` (Section 12).
   Both are saved as PDFs to `results/` (`{network}_dispersion_cv_map.pdf`,
   `{network}_jensen_gap_map.pdf`) as well as displayed inline.
4. Verified by executing the full notebook end-to-end
   (`jupyter nbconvert --to notebook --execute --inplace`) against the real
   `scarce=0.75` data — all 27 cells run with no errors.
5. **Styling pass (after user feedback on the first version):** on-figure
   titles were removed entirely — each map's meaning is documented here
   instead (see "How to read each map" below) rather than repeated as text on
   every figure. Marker size was reduced (`base_size=550` vs. the original
   `1200`) and a white halo circle (`s=base_size*1.7`) is drawn behind every
   demand-node marker, with node-number labels given a white text outline
   (`matplotlib.patheffects.withStroke`) — this stops nearby nodes (e.g. Gas
   nodes 11/12/13, Water nodes 42/44/46) from visually merging into each
   other or into the SVI background. Supply/transshipment markers were
   switched to neutral gray (`dimgray`/`darkgray`) with white edges/labels so
   they read as "context," not competing with the metric's color scale.
   Colormaps were switched from `viridis`/`magma` to **single-hue sequential**
   `Blues` (dispersion) and `Purples` (Jensen gap) — both read light-to-dark
   with no red/orange/yellow hues, which is what makes them stand out clearly
   against the SVI choropleth's warm yellow/orange/dark-red palette (the
   original `viridis`/`magma` both have yellow or near-black endpoints that
   blended into the background in exactly the same tone range).

### How to read each map (no title is drawn on the figures themselves)

- **`{network}_dispersion_cv_map.pdf`** — demand nodes colored by the
  **coefficient of variation of `phi_i`** (its DE-solution allocation ratio)
  **across the 150 scenarios**, colorbar labeled "phi coefficient of variation
  (std/mean across DE scenarios)". Darker blue = that node's allocation swings
  more, relatively, from scenario to scenario.
- **`{network}_jensen_gap_map.pdf`** — demand nodes colored by the **ex-post
  Jensen gap** at that node (from `results/jensen_gap_expost_from_de.csv`),
  colorbar labeled "ex-post Jensen gap (nats)". Darker purple = evaluating
  Formulation B's objective on Formulation A's existing allocation "gains"
  more at that node purely from Jensen's inequality, i.e. that node is where
  the two formulations would disagree most.
- In both maps: diamonds = supply nodes, squares = transshipment nodes (both
  gray, no metric — they have no `phi`), circles = demand nodes (colored by
  the metric). Black lines are network arcs. The background choropleth is the
  Social Vulnerability Index (SVI), unrelated to either metric — it's kept
  only for geographic/equity context.

### What is "coefficient of variation" (CV)?

`CV = std(phi_i,omega across omega) / mean(phi_i,omega across omega)` for a
given demand node `i`. It's the standard deviation of that node's DE-solution
allocation ratio across the 150 Monte Carlo scenarios, **normalized by its own
mean** — this is what makes it comparable *across nodes with very different
demand/allocation magnitudes* (a node that's always allocated ~50 units with
+-5 swings has the same CV as one always allocated ~5 units with +-0.5 swings,
even though the raw standard deviation differs 10x). CV = 0 means that node's
allocation is identical in every scenario (fully insensitive to the random
`alpha` draw); higher CV means the optimal DE allocation at that node depends
heavily on which `alpha` scenario was realized.

### What does the ex-post Jensen gap mean, per node and system-wide?

`node_jensen_gap_i = log(E_omega[phi_i,omega ** alpha_i,omega]) - E_omega[alpha_i,omega * log(phi_i,omega)]`.

By Jensen's inequality this is **always >= 0** because `log` is concave: taking
the expectation *before* the log (Formulation B's `log(E[phi**alpha])`) can
only look as good as or better than taking the expectation *after* the log
(Formulation A's `E[alpha*log(phi)]`). Concretely:

- **Formulation A / DE** (`E[alpha*log(phi)]`) is a standard *risk-averse*
  expected-utility objective: every scenario's log-utility is computed first,
  so a scenario where a node gets a very small `phi` (bad realization) drags
  the objective down hard, in proportion to how bad it is, **and that penalty
  survives the averaging**. This is exactly what the paper commits to, and
  it's the more conservative (fairness-protective) of the two formulations.
- **Formulation B** (`log(E[phi**alpha])`) averages the raw allocation *first*
  and only takes the log of that average. A single bad scenario at a node gets
  smoothed out by averaging with the node's other (better) scenarios before
  the concave penalty is ever applied — so Formulation B's objective
  systematically **understates** how bad the worst scenarios are at a node,
  relative to Formulation A.
- **A large per-node Jensen gap therefore flags a node whose scenario
  distribution of `phi` is the most "tail-risky"** — not necessarily the node
  with the widest CV (see Gas_node_9 below, which has the single largest gap
  but is not the single highest-CV node), but the node where the *worst-case
  low-`phi` scenarios* are pulling Formulation A's objective down the most
  relative to what a naive average-then-judge view (Formulation B) would
  suggest. Practically: at those nodes, judging fairness by "the average
  allocation looks fine" would be misleading — the risk-averse view (which the
  paper reports) is catching real exposure to bad-scenario outcomes that a
  simple average would hide.
- **At the system level**, the total gap (5.12 nats, ~4% of `OF_A_DE`) is a
  direct, interpretable measure of how much *apparent* welfare a decision-maker
  would over-state by evaluating the existing (Formulation-A-optimal)
  allocation with Formulation B's more optimistic objective instead of
  Formulation A's. It is small in relative terms precisely because Formulation
  A's DE solution was already chosen to protect against these bad scenarios
  (it doesn't need to be "surprised" by them) — but it is not zero, and the
  per-node breakdown pinpoints exactly which nodes carry that residual
  scenario risk, which is actionable information (e.g. for prioritizing
  reinforcement/contingency planning) that a single system-wide number would
  hide.

### 13.3 Results / implications

> **Update 2026-07-16 (post supply-flow-balance fix):** the tables below are
> regenerated against the corrected `DE_scarce_0.75_..._.csv` (see Section
> 14.4). Only one thing materially changed: Power's peak-CV node moved from
> `Power_node_13` to `Power_node_25`. Every aggregate number (means, totals,
> Jensen gap ranking) is unchanged at reporting precision.

**Dispersion (coefficient of variation of `phi` across 150 DE scenarios):**

| Network | Mean CV | Max CV | Node with max CV |
|---|---|---|---|
| Gas | 0.14 | 0.44 | Gas_node_15 |
| Power | 0.09 | 0.51 | Power_node_25 |
| Water | 0.11 | 0.52 | Water_node_39 |

Water has both the highest average and highest peak dispersion, meaning its
optimal allocations swing the most across scenarios — i.e., Water's demand
nodes are the most sensitive to the SVI-driven uncertainty in `alpha`.

**Ex-post Jensen gap (total per network, `scarce=0.75`):**

| Network | Total gap (nats) | Mean per node | Max per node | Node |
|---|---|---|---|---|
| Gas | 0.95 | 0.073 | 0.577 | Gas_node_9 |
| Power | 2.39 | 0.057 | 0.448 | Power_node_19 |
| Water | 1.78 | 0.052 | 0.451 | Water_node_18 |

Total gap across all three networks is **5.124 nats** (`OF_A_DE = 121.356` ->
`OF_B_expost = 126.480`). Power contributes the largest *total* gap (more nodes
with moderate gaps), while the single largest *per-node* gap is at
`Gas_node_9` (0.58 nats) — this node's CV/dispersion in the map above is not
the single highest, meaning the Jensen gap concentration doesn't map 1:1 onto
raw dispersion; it also depends on where in the `alpha`/`phi` range the
scenario mass sits (Jensen's gap for `log` is largest when there is
probability mass near very small `phi` values, not just high variance per se).
The maps make this pattern directly visible instead of requiring a manual
top-10 table lookup.

**Practical takeaway:** the "Jensen gap" between optimizing expected
log-utility (Formulation A / DE, tractable, risk-averse) and optimizing
log-expected-utility (Formulation B, intractable at this scale, more
optimistic/tail-blind) is real but modest in aggregate (~4% of `OF_A_DE`), and
is concentrated at a small number of identifiable nodes rather than spread
uniformly. Those nodes (`Gas_node_9`, `Water_node_18`, `Power_node_19`, ...)
are exactly where Formulation A's risk-averse objective is doing real work
protecting against bad-scenario outcomes that a naive average-allocation view
would not catch — i.e., where reporting "the DE solution is fair on average"
would be most misleading without also reporting the scenario spread.

## 14. Supply-to-supply flow-balance fix, safe-rerun guarantees, and a sequential warm-start experiment

### 14.1 Supply flow-balance constraint fix (all four models)

The supply-node balance constraint in `EV_and_EEVS` (both the `EV` and `EEVS`
blocks), `WS`, and `DE` previously only balanced outflow against net supply:

```
sum_j x_star[i,j,k,l]                                        == b[i,k,l] - delta_plus[i,k,l]
```

This didn't allow a supply node to *receive* flow from another supply node
(supply-to-supply transfers), which the network topology allows for. Fixed by
adding the missing inflow term to all four occurrences (`ev_eevs_model.py`
lines ~78 and ~213, `ws_model.py` line ~76, `de_model.py` line ~80):

```
sum_j x_star[i,j,k,l] - sum_j x_star[j,i,k,l]  == b[i,k,l] - delta_plus[i,k,l]
                        (new inflow term, j in N_k_s[k])
```

Applied identically to all four solve blocks since this constraint has the
same structure everywhere. Verified all three files compile cleanly; the user
then manually reran the full `run_experiments.run_all()` pipeline with this
fix in place — see Section 14.4 for the post-fix Jensen gap/dispersion
results and a stale-data issue caught and fixed during that regeneration.

### 14.2 Safe-rerun guarantees confirmed before the manual rerun

Before rerunning, confirmed exactly what does and doesn't cleanly overwrite —
see Section 9.4 for the full writeup. Summary: `.txt`/`_summary.json`/
`_solve_diagnostics.md`/all result CSVs use `open(..., "w")` and always
overwrite cleanly; the one exception was the DE model's raw Gurobi `LogFile`
(appends by default across process runs) — fixed by deleting any existing
file at that path before each DE solve. Only `DE` sets a `LogFile` at all
(intentional — `EV`/`EEVS`/`WS` solve many small per-scenario models where a
log file per solve would be impractical clutter; their per-solve stats are
already captured in `solve_diagnostics.md`).

### 14.3 Sequential MIP warm-start: a small, verified runtime experiment

**Motivation.** In both `EEVS` and `WS`, every scenario's constraint set is
built only from `b`, `u`, `gamma`, etc. — none of which are indexed by
`omega`. Only `alpha` varies per scenario, and `alpha` appears **exclusively
in the objective**, never in a constraint. So every scenario solved in a
sequential loop shares an *identical feasible region*; the previous
scenario's optimal solution is therefore provably feasible for the next one,
making it a legitimate Gurobi MIP `Start` (warm start) candidate — pure
runtime experiment, does not change any result.

**Implementation** (`ws_model.py`, `ev_eevs_model.py`): added an opt-in
`warm_start_sequential=False` parameter (default off, so `run_experiments.py`
and all existing behavior are completely unaffected). When `True`, each
scenario after the first has every variable's `.Start` set from the previous
scenario's solved values before `model.optimize()`. Output filenames and
diagnostics labels get a `_warmstart` suffix whenever the flag is on, so a
benchmark run can never collide with or overwrite baseline result files.
`src/benchmark_warm_start.py` is a small standalone script/importable function
(`run_benchmark(scarce)`) that reruns only `EEVS` (sample-mean) and `WS` with
the flag on, using the same `seed`/parameter-loading as
`run_experiments.run_scarcity_level` for a fair comparison, and writes its own
`scarce_<scarce>_solve_diagnostics_warmstart.md`.

**Result (scarce = 0.75, 150 scenarios, same alpha draws as the baseline run
via matching `seed`):**

| Model | Baseline total (s) | Warm-start total (s) | Change | Baseline mean/solve | Warm-start mean/solve |
|---|---|---|---|---|---|
| EEVS (sample-mean) | 3113.1 | 2609.7 | **16.2% faster** | 20.75s | 17.40s |
| WS | 26.4 | 35.6 | **34.6% slower** | 0.176s | 0.237s |

**Correctness check:** compared `obj_val` scenario-by-scenario between
baseline and warm-start runs for both models — **zero mismatches** (all
within `1e-3`, well inside solver tolerance). Warm-starting only ever affects
*how fast* Gurobi reaches the optimum, never *what* the optimum is, as
expected since it's just a MIP start hint, not a change to the model.

**Interpretation — why it helps EEVS but hurts WS:** EEVS's per-scenario
solves are genuinely hard MIQCPs (baseline mean ~21s, max 75s) — a warm
incumbent measurably prunes the branch-and-bound tree, netting a real ~16%
time saving over 150 scenarios (~8 minutes saved on this run). WS's
per-scenario solves are already close to instant (baseline mean ~0.18s) —
here, the Python-side cost of setting hundreds of `.Start` values every
iteration plus Gurobi's own MIP-start processing overhead outweighs any
branch-and-bound savings on a problem that's already trivial to solve. **Net
takeaway: sequential warm-starting is worth keeping for `EEVS` (the
computationally expensive model) but not worth enabling for `WS` (already
fast) — a real, if modest, demonstrable contribution when applied selectively
rather than blanket-enabled.** Kept as an opt-in flag (default off) rather
than a mandatory pipeline change, given the mixed result.

### 14.4 Stale Jensen gap / dispersion data caught before it reached the paper

After the flow-balance fix's rerun completed, the user independently
spot-checked whether `results/jensen_gap_expost_from_de.csv` (Section 12) and
the dispersion/Jensen-gap maps (Section 13) had actually been regenerated
against the *new* DE solve, rather than silently left over from before the
fix. Two things were flagged:

1. **Stale by ~6 hours.** `jensen_gap_expost_from_de.csv` was timestamped
   19:22 the prior day; the corrected `DE_scarce_0.75_..._.csv` was
   timestamped 01:12 the next morning. Confirmed numerically too: summing
   `E_log_util_A` from the stale CSV reproduced the *old* `OF_DE = 121.279`
   exactly, not the corrected `summary.json`'s `OF_DE = 121.356`. Root cause:
   `jensen_gap_from_de_csv.py` was run once (Section 12) and never re-run
   after the model fix — it doesn't auto-run as part of `run_experiments.py`
   by design (per the earlier explicit instruction not to fold it into the
   main pipeline), so nothing re-triggered it automatically.
   - **Fix:** re-ran `python jensen_gap_from_de_csv.py` against the current
     DE CSV. New result: `OF_A_DE_check = 121.356` (now matches
     `summary.json` exactly), `OF_B_expost = 126.480`, `jensen_gap = 5.124`.
     Also re-executed `notebooks/plotter.ipynb` end-to-end (zero errors) to
     regenerate all 6 map PNGs against the corrected data. Section 12.3 and
     13.3's tables above are updated with these post-fix numbers.
   - **What actually changed vs. the pre-fix numbers:** almost nothing at
     reporting precision — total Jensen gap 5.124 vs. 5.12, same top-3 gap
     nodes in the same order (`Gas_node_9` > `Water_node_18` >
     `Power_node_19`). The one real change: Power's peak-CV node moved from
     `Power_node_13` to `Power_node_25`. The supply-flow-balance fix allows
     supply nodes to route flow to each other, which apparently redistributes
     enough to shift *which* Power node is most scenario-sensitive, without
     moving the aggregate totals much at this scarcity level.
2. **159 columns in the DE CSV — checked, not a bug.** The user noticed
   `phi_(...)` rows have 159 values instead of the expected 150 and flagged
   it for a sanity check rather than assuming a bug. Confirmed: the CSV has
   150 scenario columns (`'0'`..`'149'`) **plus 9 appended summary-stat
   columns** (`mean`, `std`, `min`, `max`, `median`, `mode`, `q1`, `q2`, `q3`)
   from `gurobi_utils`'s results-export helper — 150 + 9 = 159, exactly as
   expected. `jensen_gap_from_de_csv.py`'s `_omega_columns()` already filters
   to only the digit-named columns (`c.isdigit()`) before computing anything,
   so this was never a live bug in the Jensen gap script itself — confirmed
   by `Scenarios: 150` in its own printed output on the rerun above. Flagging
   it as "159 raw values" without that filter would look like a bug to a
   naive parse of the CSV, though, so worth remembering when writing any new
   ad hoc script against these result CSVs.

**Takeaway:** any script/notebook that consumes `DE_scarce_*.csv`,
`EEVS_mu_hat_scarce_*.csv`, or `WS_scarce_*.csv` downstream (Jensen gap,
plotter, any future analysis) needs to be explicitly re-run any time the
underlying model code changes and the pipeline is rerun — nothing
auto-detects staleness. Worth checking file timestamps (or, more robustly,
recomputing `OF_A_DE_check` and diffing against `summary.json`'s `OF_DE`, as
was done here) before trusting a downstream results file after a model
change.

---

## 15. The interdependency constraint was never built; solver reconfiguration; final run of record

This section supersedes earlier solver-configuration notes. It is written to be
self-contained for anyone (human or agent) picking the project up cold.

### 15.1 The headline bug: a silent 4-tuple / 5-tuple key mismatch

A coauthor flagged the inequality direction of the interdependency constraint
during revision. Investigating it turned up something worse: **the constraint
had never been added to any model.**

`gamma` is built in `tdINDP_parameters_build.py:354` with **5-tuple** keys
carrying a trailing time index:

```python
gamma[(dependee_node, depender_node, dependee_network, depender_network, t)] = 1
```

Every model file guarded constraint creation with a **4-tuple** lookup,
`if (i,j,k,k_tilde) in gamma.keys()`. That test is silently `False` for every
candidate pair -- verified: **0 matches out of 19,600 candidate pairs**, versus
27 matches for the 5-tuple form. No exception, no warning; the models built,
solved, and reported `OPTIMAL`. This is why the earlier `<=` -> `>=` sign fix
produced bit-identical results: it flipped the sign of a constraint that did
not exist.

**Root cause, precisely.** `load_base_parameters()` already collapses the time
dimension for `b` and `u` (lines 33-35) but never did so for `gamma`. The fix
matches that existing pattern rather than patching six call sites:

```python
# src/run_experiments.py, ~line 36
gamma = {(i, j, k, k_tilde): v for (i, j, k, k_tilde, t), v in gamma.items() if t == 0}
```

All `gamma` values are `1` and all live at `t == 0`, so the collapse is lossless.

**Semantics** (confirmed against the builder): `i` is the **dependee/provider**
in network `k`; `j` is the **depender** in network `k_tilde`. The constraint is

```python
quicksum(w[(i_dep,k,t)] * gamma[(i_dep,j,k,k_tilde,t)] for i_dep in dep_nodes) - w[(j,k_tilde,t)] >= 0
```

i.e. a depender may only be functional if its dependees are. `>=` is correct.
The aggregate form is used (21 constraints); a disaggregated per-pair variant
(27 constraints) was implemented and tested but rejected -- only 4 of 21
dependers have more than one dependee, so it strengthens almost nothing.

> **Correct the stale comment at `tdINDP_parameters_build.py:361`** -- it states
> the index order backwards, contradicting the construction two lines above,
> and is the likely origin of the original confusion. Still unfixed.

### 15.2 Why the published numbers survived

Before rerunning anything, all 21 constraints were evaluated against every
stored solution (21 constraints x 150 scenarios x DE/WS/EEVS): **zero
violations**, for both the aggregate and disaggregated forms. Since adding
constraints only shrinks the feasible region, and the previous optimum remains
feasible within it, that optimum remains optimal. The bug was real but
**non-binding at `scarce = 0.75`**.

Caveat for the paper: this is verified *at this scarcity level only*. Tighter
scarcity forces more `w = 0`, at which point the constraint can bind and
results may genuinely move. Note DE has all `w = 1`, but EEVS and WS genuinely
have 1-2 nodes with `w = 0` per scenario.

### 15.3 The performance bug: piecewise-linear expansion of `ln(phi)`

With the constraints finally active, EEVS/WS solves went from ~0.28 s median to
grinding for hours. The interdependency constraints were **not** the cause.

`addGenConstrLog` was being handled by static piecewise-linear approximation.
Minimal repro, a single log constraint: `FuncNonlinear=0` produced **225**
presolved columns; `FuncNonlinear=1` produced **3**, same objective. At 89 log
terms this is the observed `Presolve added 0 rows and 20788 columns` plus 89
SOS2 sets.

Setting `FuncNonlinear=1` (dynamic outer approximation, exact) reduced the
presolved model from **22,142 columns + 89 SOS to 389 columns, 0 SOS**.

### 15.4 The numerics were self-contradictory

The previous settings were `NumericFocus=3`, `IntegralityFocus=1`,
`IntFeasTol=1e-9`. That demanded integrality feasibility **six orders of
magnitude tighter than the accuracy of the model's own nonlinear constraints**
(the PWL expansion reported `max constraint violation ~5e-04 exceeds
tolerance` on essentially every solve), which explains the many `infeasible`
nodes explored at ~0 simplex iterations each.

With `FuncNonlinear=1` the log constraints are handled exactly, so the conflict
is gone and the expensive numerics are no longer justified. Relaxing them is
also what finally unstuck DE: under `FuncNonlinear=1` with the *tight*
numerics, DE's spatial branch-and-bound bound froze at its root value
(121.42097) for 2,300+ s. With the relaxed numerics **DE closes in 65.8 s.**

### 15.5 Final hyperparameters (the run of record)

Set once in `configure_common_gurobi_params` (`src/gurobi_utils.py`) and applied
uniformly to **all** models -- EV, EEVS, WS, DE. There is no longer a hybrid
per-model configuration.

| parameter | value | rationale |
|---|---|---|
| `FuncNonlinear` | `1` | exact log via outer approximation; avoids the ~20k-column PWL blowup |
| `MIPGap` | `1e-4` | Gurobi's own default, but now set **explicitly** so it is recorded |
| `NumericFocus` | `0` | conflict with nonlinear accuracy removed |
| `IntegralityFocus` | `0` | as above |
| `IntFeasTol` | `1e-7` | still 100x tighter than Gurobi's `1e-5` default |
| `FuncPieceError` | `1e-4` | only consulted if `FuncNonlinear=0` |
| `DualReductions` | `0` | keeps INFEASIBLE vs UNBOUNDED distinguishable |

**Environment:** Gurobi **11.0.3**, Python **3.8.0** (EOL Oct 2024), academic
license expiring 2027-06-08. Gurobi **13.0.2** is current and advertises
substantially faster MINLP plus a new nonlinear barrier method -- directly
relevant to the DE spatial-B&B bound. It requires **Python >= 3.10** (Gurobi 12
requires >= 3.9), so upgrading means rebuilding the environment and probably
regenerating the license file. Deliberately deferred, not rejected.

**Provenance.** `solver_config_snapshot()` reads these six parameters back off
each solved model, `collect_solve_diagnostics` carries them, and
`write_summary_json` emits a `solver_config` block **keyed by model label** in
`summary.json`. Results files are now self-describing: no run's configuration
has to be reconstructed from memory. DE additionally dumps its full solution to
`results/DE_solution_scarce_{scarce}_num_scenarios_{n}.json`, so any DE solve
can be replayed as a warm start.

### 15.6 Runtime, after

Whole pipeline, 453 solves: **150.7 s** (was hours).

| stage | n | total | median | max |
|---|---|---|---|---|
| EV (both) | 2 | 0.4 s | 0.18 s | 0.2 s |
| EEVS population-mean | 150 | 13.8 s | 0.09 s | 0.2 s |
| EEVS sample-mean | 150 | 32.7 s | 0.18 s | 0.4 s |
| WS | 150 | 38.0 s | 0.25 s | 0.4 s |
| DE | 1 | 65.8 s | -- | -- |

DE terminated `OPTIMAL`, objective 121.3373588, bound 121.3455042, **gap
0.0067%**. No solve raised a diagnostic flag.

### 15.7 Results comparison across the three configurations

- **A** = original/published: PWL, tight numerics, **interdependency constraints absent** (the bug)
- **B** = hybrid: DE=PWL, EEVS/WS=outer, tight numerics, constraints active
- **C** = final run of record: all-outer, `MIPGap=1e-4`, relaxed numerics, constraints active

| metric | A (published) | B (hybrid) | C (final) | C - A |
|---|---|---|---|---|
| `OF_EV_alpha_mu` | 121.02311 | 121.02454 | 121.02108 | -0.00202 |
| `OF_EV_alpha_mu_hat` | 120.98761 | 120.98880 | 120.98894 | +0.00133 |
| `OF_EEVS_alpha_mu` | 121.27090 | 121.28689 | 121.29304 | +0.02213 |
| `OF_EEVS_alpha_mu_hat` | 121.27423 | 121.32546 | 121.28893 | +0.01470 |
| `OF_WS` | 121.66233 | 121.66221 | 121.65964 | -0.00269 |
| `OF_DE` | 121.35597 | 121.35597 | 121.33736 | -0.01861 |
| `EVPI` | 0.30636 | 0.30624 | 0.32228 | +0.01592 |
| `VSS_alpha_mu` | 0.08506 | 0.06908 | 0.04432 | -0.04074 |
| `VSS_alpha_mu_hat` | 0.08174 | 0.03050 | 0.04843 | -0.03331 |

**Read this carefully before citing VSS.** The headline objectives are stable
to ~1e-4 relative. `EVPI` moves ~5%. But **`VSS` is not stable**: it is a small
difference of two large, nearly equal numbers, so it inherits the full solver
tolerance of both. `VSS_alpha_mu_hat` ranges 0.031-0.085 across configurations
-- a factor of 2.7 -- while nothing about the model changed. At `MIPGap=1e-4`
the tolerance on a ~121.3 objective is ~+/-0.012, which is a substantial
fraction of a VSS of ~0.044.

**Recommendation:** report `OF_DE` and `EVPI` as robust; report `VSS` with an
explicit tolerance caveat, or tighten `MIPGap` specifically for the DE/EEVS
pair if a sharp VSS claim is needed. Do not report VSS to more than two
significant figures.

### 15.8 The optimum is degenerate -- the most important finding for interpretation

Comparing B against C isolates solver configuration alone (both post-fix, same
data, same constraints, differing only in `FuncNonlinear` and numerics):

| quantity | B (PWL) | C (outer) | difference |
|---|---|---|---|
| `OF_DE` | 121.355966 | 121.337359 | **1.5e-04 relative** |
| total demand met | 2243.2463 | 2243.2463 | **0.0000** |
| `phi` entries differing > 1e-6 | -- | -- | **37.2%** |
| max abs difference in `phi` | -- | -- | **60.05 units** |
| nodes with identical allocation | -- | -- | **37 of 89** |
| `x_star` entries differing | -- | -- | 38.0% |
| `y` entries differing | -- | -- | 37.8% |
| `w` flips | -- | -- | 150 |

The objective and the aggregate totals are pinned; **the allocation is not.**
Over half the demand nodes (52 of 89) receive materially different allocations
between two runs of the same model, one node by as much as 60 units, at an
objective cost of 0.0186. All 150 `w` flips are the same node,
`Power_node_64`, restored in every scenario under B and in none under C.
Largest movers are concentrated in Gas (`Gas_node_15`, `Gas_node_8`,
`Gas_node_5`).

This is a **flat/degenerate optimum**, and it is a property of the model, not a
solver defect: proportional fairness at this scarcity admits a large family of
near-identical-value allocations.

**Consequences, which should be stated in the paper:**

1. Aggregate claims (`OF_DE`, `EVPI`, total met demand, per-network totals) are
   **identified and safe**.
2. **Node-level claims are not identified.** "Which node is most
   scenario-sensitive", per-node dispersion/CV rankings, and per-node Jensen
   gaps can change between solver configurations without any modeling change.
   This retroactively explains the earlier observation (Section 14.4) that
   Power's peak-CV node moved from `Power_node_13` to `Power_node_25`.
3. Any node-level figure needs either a tie-breaking secondary objective (e.g.
   lexicographic max-min, or a small regularizer) to make the allocation
   unique, or an explicit statement that node-level detail is illustrative
   rather than inferential.

### 15.9 Where the supply actually goes

Per network, supply is 1000 raw, scaled to **750** at `scarce = 0.75`; demand
is 1000. Mean demand met over 150 scenarios, DE, final run:

| network | supply @0.75 | demand | met | stranded |
|---|---|---|---|---|
| Gas | 750.000 | 1000.000 | 750.000 | 0.000 |
| Power | 750.000 | 1000.000 | 750.000 | 0.000 |
| Water | 750.000 | 1000.000 | 743.246 | **6.754** |
| **total** | 2250.000 | 3000.000 | **2243.246** | 6.754 |

Gas and Power route their full allocation. **Water strands 6.754 units (0.9%)**.

This was checked properly rather than assumed, because it looks like a bug.
It is not one:

- **The solver is achieving the true maximum.** An independent max-flow
  computation over the model's *actual* arc structure -- flow variables `x`
  exist only on arcs with both endpoints in `N_k_d_nod`, plus first-stage
  `x_star` arcs leaving `N_k_s` -- returns **743.246**, matching the DE
  solution to three decimals. (An unrestricted max-flow over all of `A_k`
  returns the full 750; the difference is the model's arc restriction, so
  comparing against the unrestricted number is misleading.)
- **It localizes to exactly one node.** `Water_node_11` holds 11.035 units of
  supply and can usefully deliver only 4.281, stranding **6.754** -- the entire
  system-wide shortfall. Its outgoing capacity is not the constraint (38.548
  across 3 arcs); the demand reachable from it is already fully served, and
  **Water has zero transshipment nodes** (`len(N_k_nod['Water']) == 0`), so
  there is no route to carry its surplus to unmet demand elsewhere.
- **Why the escape routes don't exist: the two-stage variable partition.**
  The stage split determines which arcs carry flow at all. The first stage
  dispatches supply, so `x_star` is defined on `i in N_k_s[k], j in N_k[k]`
  (supply -> anything). The second stage redistributes among demand nodes, so
  `x` is defined only on arcs with **both** endpoints in `N_k_d_nod[k]`
  (demand -> demand). **Demand -> supply arcs therefore belong to neither
  stage and carry no variable in either.** In the Water pocket, `40 -> 11`,
  `48 -> 11`, `37 -> 11`, `48 -> 13` and `40 -> 13` all exist in `A_k` but are
  structurally unusable for this reason. Combined with the absence of any
  `11 <-> 13` arc, `Water_node_11`'s only usable exits are into the 4.281-unit
  pocket, and its surplus cannot be handed to a neighbouring supply node.
  `delta_plus` (the unused-supply slack) absorbs the remainder, as designed.
- **This also shows the Section 14.1 supply-flow-balance fix earning its
  keep.** `Water_node_13` sits in the same pocket but owns `13 -> 14`, a
  supply -> supply arc, which that fix made usable. It therefore delivers its
  full 1.219 while `Water_node_11` strands. Before the fix, node 13 would have
  been trapped as well and the shortfall would have been larger.
- **The model states this itself.** The cleanest evidence is not the max-flow
  argument above but `delta_plus`, the unused-supply slack: across all 27
  supply-node entries in the entire three-network system, **exactly one is
  nonzero** -- `delta_plus_('Water_node_11', 'Water', 'Water') = 6.7537`. Any
  audit of supply utilisation should read that variable directly.
- **Invariant across every configuration tested.** Gas 750.0000, Power
  750.0000, Water 743.2463 (stranded 6.7537) hold identically for the original
  pre-fix run, the hybrid run, the `MIPGap = 1e-4` run and the certified
  `MIPGap = 1e-6` run. It is unaffected by the interdependency fix, by
  `FuncNonlinear`, by the numerics, and by the gap. It is a property of the
  Water topology under the two-stage variable partition, nothing else.
- **It predates everything done in this session.** Configurations A, B and C
  all strand exactly 6.754, and no Water node is ever `w = 0` in any of them.
  It is present in the originally published run. It is a property of the input
  topology and demand data, not of the interdependency fix or solver settings.

Worth a sentence in the paper: Water is the binding network, and one supply
node's output is structurally unusable at this scarcity level.

### 15.10 State at handoff / open items

Backups live **outside the repo** at
`~/.copilot/session-state/23e35144-.../files/`:
`results_backup_2026-08-14` (original pre-fix, 507 files),
`results_hybrid_pwl_de` (config B), `results_final_allouter_2026-08-14`
(config C, = current `results/`).

Open, in rough priority order:

1. ~~**Downstream artifacts are stale.**~~ **DONE.** `jensen_gap_from_de_csv.py`
   was re-run and `notebooks/plotter.ipynb` re-executed end-to-end against the
   final run. `OF_A_DE_check = 121.337359` now matches `summary.json`'s
   `OF_DE` exactly. New Jensen figures: `OF_B_expost = 126.455421`,
   **`jensen_gap = 5.118062`** (was 126.480 / 5.124 pre-fix). The top-3 gap
   nodes are unchanged in order: `Gas_node_9` > `Water_node_18` >
   `Power_node_19`. All 11 figures regenerated, including the three
   `*_jensen_gap_map.pdf` files that had been missing from the results set.
   Updated CV maxima (these moved, so any prior dispersion figure is stale):

   | network | max CV | node |
   |---|---|---|
   | Gas | 0.4800 | `Gas_node_8` (was `Gas_node_15`) |
   | Power | 0.5165 | `Power_node_25` |
   | Water | 0.5379 | `Water_node_39` |

   Note these node-level peaks are exactly the quantities Section 15.8 warns
   are not identified -- they move with solver configuration.
2. **Decide the VSS reporting stance** (Section 15.7).
3. **Decide whether node-level results are claimed at all** (Section 15.8).
   This is the substantive research decision outstanding.
4. Fix the stale comment at `tdINDP_parameters_build.py:361`.
5. EEVS's warm-start MIP start (`ev_eevs_model.py`, ~line 326) seeds only
   second-stage variables and is silently rejected; seed `w`, `y`, `x_star`,
   `delta_plus` if warm starts are wanted there.
6. `x_start` in `de_model.py` is vestigial -- written, never read. Either wire
   it or delete it.
7. Consider the Gurobi 13 upgrade in an isolated environment (Section 15.5),
   keeping 3.8 intact until it reproduces these numbers.

### 15.11 `MIPGap = 1e-6`: the run of record, and a correction

Rerun with `STOCHFAIR_MIPGAP=1e-6` (set in `notebooks/run_all_pipelines.ipynb`
cell 1; `gurobi_utils.DEFAULT_MIP_GAP` reads it at import time). Everything
else unchanged.

| metric | 1e-4 | **1e-6** |
|---|---|---|
| `OF_EV_alpha_mu` | 121.02108 | 121.02454 |
| `OF_EV_alpha_mu_hat` | 120.98894 | 120.98909 |
| `OF_EEVS_alpha_mu` | 121.29304 | 121.25442 |
| `OF_EEVS_alpha_mu_hat` | 121.28893 | 121.24103 |
| `OF_WS` | 121.65964 | 121.66306 |
| `OF_DE` | 121.33736 | **121.34945** |
| `EVPI` | 0.32228 | 0.31361 |
| `VSS_alpha_mu` | 0.04432 | **0.09504** |
| `VSS_alpha_mu_hat` | 0.04843 | **0.10842** |

**DE is now certified globally optimal:** `Best objective 121.3494513985, best
bound 121.3494513985, gap 0.0000%` in 83.3 s, 7 incumbents. No stall -- the
earlier bound freeze was an artifact of the tight numerics, not structural.

**Correction to Section 15.7.** That section argued from the 1e-4 run's
reported bound (121.3455042) that no allocation could score above it. **That
was wrong.** The 1e-6 run reaches 121.3494514 on identical data (`random.seed(42)`
before `sample_alpha`, so alpha draws are reproducible). The reason: when a
solve terminates on a nonzero `MIPGap`, Gurobi prunes nodes whose bound falls
within the gap cutoff, so the reported `ObjBound` is **not** a valid global
bound on termination -- the only guarantee is that the incumbent is within the
gap of the optimum. Do not use a gap-terminated `ObjBound` to rule out better
solutions. (The comparison against the PWL value stands: PWL's 121.35597 still
exceeds the *certified* optimum 121.34945 by 6.5e-03, so PWL over-reports.)

**VSS is now defensible.** At 1e-6 the solver tolerance is ~+/-1.2e-04 against
a VSS of ~0.095-0.108 -- roughly 800x, versus ~4x at 1e-4. The Section 15.7
caveat can be dropped **provided the 1e-6 numbers are the ones reported**.
Note VSS more than doubled, confirming the 1e-4 values were tolerance-dominated
rather than meaningful.

**Cost:** 453 solves in **1015.6 s (16.9 min)**, versus 150.7 s at 1e-4. Nearly
all of the increase is WS (849.0 s vs 38.0 s); DE grew only 65.8 -> 83.3 s.
Well worth it -- run at 1e-6.

**Warm start does not improve solutions, and now costs time.** Across all 301
warm-started solves the objective delta versus baseline is **exactly 0.00e+00**
(max, min and mean), and at this gap the warm start is *slower*: EEVS
sample-mean -6.6%, WS -21.9%. `benchmark_warm_start` never calls `DE` -- there
is no warm-started DE, and the diagnostics file contains no `DE` row. It is a
runtime-only experiment writing `_warmstart`-suffixed files.

**Downstream artifacts regenerated** against this run: `jensen_gap_expost_from_de.csv`
(`OF_A_DE_check = 121.349451` matches `OF_DE`; `OF_B_expost = 126.471260`,
**`jensen_gap = 5.121808`**) and all 11 figures including the three
`*_jensen_gap_map.pdf`.

**Supply utilisation is unchanged at 1e-6** -- see Section 15.9, which is the
full treatment. Re-verified against this run: Gas 750.0000, Power 750.0000,
Water 743.2463, with `delta_plus_('Water_node_11', 'Water', 'Water') = 6.7537`
the single nonzero unused-supply entry in the whole system, identical to the
1e-4 run to four decimals. The shortfall is not a solver artifact and does not
respond to the gap; it is the two-stage variable partition plus the Water
pocket topology.
