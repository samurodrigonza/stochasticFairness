# Stochastic Fairness — Power-Utility Proportional Fairness under Uncertainty

Stochastic proportional-fairness resource allocation for interdependent
infrastructure networks (Gas, Power, Water), modeled as a time-dependent
interdependent network design problem (tdINDP) and solved with Gurobi.

The core question: how should scarce resources be allocated across demand
nodes so as to be "fair" under a power-utility fairness criterion, when the
fairness weighting (`alpha`) itself is uncertain?

## Repository layout

```
src/            Python modules (network parameters, models, experiment drivers)
notebooks/      Active notebooks (thin entry points into src/)
results/        Generated run outputs (git-ignored; regenerate by re-running)
docs/           Deeper implementation notes and project history (see dossier.md)
old_files/      Retired notebooks / archived result sets kept for reference only
```

### `src/`

| File | Purpose |
|---|---|
| `tdINDP_parameters_build.py` | Loads/derives static network parameters (nodes, arcs, costs, capacities, supply/demand, interdependencies). |
| `svi_data.py` | `SVI[k][i] = [l_i, u_i]` — per-demand-node uniform range for the random fairness exponent `alpha_i`. |
| `gurobi_utils.py` | Shared Gurobi parameter configuration and solve-diagnostics/results-export helpers used by every model file. |
| `paths.py` | Central `results/` output path resolution. |
| `ev_eevs_model.py` | Formulation A: Expected Value (EV) model and Expectation-of-the-Expected-Value Solution (EEVS). |
| `ws_model.py` | Formulation A: Wait-and-See (WS) — one model solved per scenario with its realized `alpha`. |
| `de_model.py` | Formulation A: Deterministic Equivalent (DE) — all scenarios in one model. |
| `run_experiments.py` | Orchestrates EV/EEVS/WS/DE across scarcity levels; writes `.txt`/`.json`/`.md` summaries. |
| `formulation_b_model.py` | Formulation B (pilot): log-of-expected-utility objective, nonconvex, single model (no scenario loop — `alpha` is integrated out analytically). |
| `compare_formulations.py` | Scoring/diffing utilities to compare Formulation A vs. B on the same instance. |
| `run_formulation_b_pilot.py` | Pilot driver: degenerate correctness check (Stage 1) + full case study (Stage 2) for Formulation B. |
| `jensen_gap_from_de_csv.py` | Reads an existing `results/DE_scarce_*.csv` and computes the **ex-post Jensen gap** — Formulation B's objective evaluated on Formulation A's DE solution, per scenario then averaged — with no Gurobi solve required. |

### `notebooks/`

- `optModels_final.ipynb` — assembles and runs Formulation A (imports from `src/`); final cell runs the ex-post Jensen gap analysis against the latest DE results.
- `plotter.ipynb` — visualizes results; auto-detects available scarcity levels and auto-loads summary JSON files from `results/` (no hardcoded scarcity list, no manual copy-pasting of numbers). Includes per-network maps coloring each demand node by the coefficient of variation of its DE-solution `phi` across scenarios and by its ex-post Jensen gap.

## Formulation A vs. Formulation B

- **Formulation A** (paper's primary result): `max sum_i E[alpha_i] * ln(phi_i)`.
  Exact because `ln(phi_i)` is deterministic given `phi_i`.
- **Formulation B** (pilot experiment, not paper-committed):
  `max sum_i ln(E[phi_i ** alpha_i])`. Diverges from A once `alpha_i` is
  random (Jensen's inequality); sensitive to the *width* of uncertainty in
  `alpha_i`, not just its mean. Built as a nonconvex Gurobi model
  (`NonConvex=2`) using general power/log constraints, with a warm start
  from Formulation A's EV solution and a 1% MIP gap.

**Formulation B is not solved to optimality at full 3-network scale** — the
LP relaxation is too loose to close the gap in any practical time, even after
fixing a real warm-start bug (see `docs/dossier.md` Section 11). Instead, the
gap between the two formulations is reported as an **ex-post Jensen gap**:
Formulation B's objective evaluated directly on Formulation A's existing DE
solution, computed cheaply from the saved DE results CSV with no re-solve
(`src/jensen_gap_from_de_csv.py`, `docs/dossier.md` Section 12).

See `docs/dossier.md` for full implementation detail, the exact math, the
numerical hazards handled (the `phi_i = 1` singularity, the bilinear
division-avoidance trick), and a real infeasibility bug that was diagnosed
and fixed in the EEVS re-solve loop.

## Running the experiments

```python
# Formulation A: full pipeline (EV/EEVS/WS/DE) at the standing default
# scarcity level (0.75 = 75% of full network supply available)
import run_experiments
run_experiments.run_all()

# Formulation B pilot: degenerate correctness check, then (if it passes)
# the full three-network case study
import run_formulation_b_pilot as pilot
stage1 = pilot.run_stage1_degenerate_check()
if stage1['passed']:
    pilot.run_full_case_study()
```

Outputs land in `results/` — objective values (`scarce_*_summary.json`),
Gurobi solve diagnostics (`scarce_*_solve_diagnostics.md` — status, runtime,
MIP gap, and any warning flags for every model solved in the run), and
Formulation B pilot summaries/CSVs.

## Key metrics reported

- **EVPI** (Expected Value of Perfect Information) = `OF_WS - OF_DE`
- **VSS** (Value of the Stochastic Solution) = `OF_DE - OF_EEVS`
- **EVPI_B / VSS_B** (Formulation B analogs, reusing WS/EEVS with
  Formulation B's own objective standing in for DE — see `docs/dossier.md`
  Section 6 for why a direct "DE for B" doesn't exist)
- **Ex-post Jensen gap** = `log(E[phi**alpha]) - E[alpha*log(phi)]`, evaluated
  per node on Formulation A's DE solution and averaged over scenarios
  (`src/jensen_gap_from_de_csv.py`; `docs/dossier.md` Section 12) — the
  practical stand-in for "how much would Formulation B's optimum differ,"
  without solving Formulation B's intractable MIQCP.

## Further reading

- `docs/dossier.md` — full technical history: modularization details, the
  `i < j` → `i != j` direction-exclusivity refactor, Formulation B's math and
  Gurobi construction walkthrough, the EEVS infeasibility bug (root cause and
  fix), and the Gurobi solve-diagnostics audit trail.
- `results/README.md` — notes on the `results/` output directory conventions.
