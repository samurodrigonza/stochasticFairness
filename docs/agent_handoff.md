# Stochastic fairness: portable Copilot session handoff

Prepared 2026-10-31 for another agent resuming in a different checkout.
This file summarizes the available session history and verified repository
state; it is not a verbatim conversation transcript. Read it together with
`docs/dossier.md`, particularly sections 14, 15.9, and 15.11.

## Immediate state and scope

- Repository: `samurodrigonza/stochasticFairness`.
- Development branch: `local_dev_sam`; target branch: `main`.
- Open PR: https://github.com/samurodrigonza/stochasticFairness/pull/8
- PR title: "Modularize stochastic fairness pipeline, fix model constraints,
  and archive final results".
- `b7017cf` contains the final model fixes, solver configuration, and dossier.
- `6302f85` adds `results.zip`, the complete final-run snapshot.
- This handoff and the loose final outputs are being added in a subsequent
  commit. Use Git history for its hash rather than expecting a self-referential
  hash in this document.
- Before the archive commit, this branch was 12 commits ahead of fetched
  `origin/main`, with no commits behind. The PR consolidates a substantial
  development history, not one isolated fix.
- The user explicitly requested committing all project files and results
  and a portable session dossier. Do not include `.git/` or Python caches.
- No new optimization run was requested or performed while preparing the PR.
  Do not imply that the full pipeline was freshly rerun in October.

## What the project does

Stochastic proportional-fairness allocation over interdependent Gas, Power,
and Water infrastructure networks. Fairness exponents `alpha` are uncertain;
the main experiments compare EV, EEVS, WS, and DE under scarce supply.

The primary workflow is Formulation A. Formulation B changes the order of
logarithm and expectation; it is a nonconvex pilot, not an established
full-scale optimal comparison. The practical analysis became an ex-post
Jensen gap evaluated on the saved Formulation A DE allocation.

## Work completed during this session

1. Modularized the former large optimization notebook into model, loader,
   configuration, scoring, and experiment-driver modules under `src/`.
   Kept notebooks as entry points, moved active notebooks to `notebooks/`,
   and retained retired notebooks and historical results under `old_files/`.
2. Refactored direction-exclusivity variables and their dependent model
   constraints; investigated EEVS infeasibility following the refactor.
   The diagnosed EEVS issue was fixed and the main pipeline subsequently ran.
3. Implemented Formulation B, comparison/scoring utilities, a degenerate
   correctness check, and a pilot driver. Full-scale B did not converge to
   optimality in practical time. Pivoted to solver-free ex-post Jensen analysis.
4. Added automatic summaries, JSON exports, solve diagnostics, per-solve
   solver logs, and plots that detect available result sets.
5. Fixed supply-to-supply flow balance across models. The first-stage supply
   equations must account for incoming first-stage supply flows.
6. Investigated sequential warm starts, corrected stale downstream artifacts,
   and added a master pipeline notebook with plotting and benchmark stages.
7. Corrected interdependency inequality directions, then discovered the more
   important silent tuple-key mismatch: the constraints had not been built.
8. Diagnosed nonlinear-log performance problems, retuned Gurobi, and prepared
   notebook runs at `MIPGap=1e-4`, then `1e-6`, which the user executed.
9. Regenerated the Jensen-gap CSV and all 11 final figures, including missing
   Jensen maps and previously stale dispersion maps.
10. Documented supply utilization and solver-config sensitivity of allocations.
    Committed and pushed the final research-code state as `b7017cf`.
11. Created a separate solver-free capacity-reader folder in the parent of
    the original checkout, at the user's request. It is not part of this repo.
12. Reconciled stale session TODOs in October, committed the results archive,
    and updated existing PR #8 instead of creating a duplicate.

## Important model fixes and interpretation

### Interdependencies

`tdINDP_parameters_build.py` constructs
`gamma[(dependee, depender, dependee_network, depender_network, t)]`.
Models consume four-dimensional keys after time collapse.
`load_base_parameters()` originally collapsed `b` and `u`, but not `gamma`.
Four-tuple membership tests therefore silently failed for every candidate.

The fix in `src/run_experiments.py` collapses `gamma` at `t == 0`.
There are 27 dependency entries and 21 aggregate depender constraints.
The provider is the first node, not the second; a stale source comment
still reverses these semantics. The active aggregate inequality uses `>=`.
The aggregate form is an intentional retained modeling choice; do not
silently replace it with a stronger disaggregated interpretation.

Stored-solution checks found no dependency violations at scarcity 0.75.
That does not establish nonbinding behavior at every other scarcity.

### Solver configuration

Static piecewise-linear handling of log terms expanded the model to roughly
22,142 presolved columns and 89 SOS sets. `FuncNonlinear=1` reduced it to
roughly 389 columns with no SOS sets. Conservative numerical settings also
caused severe runtime problems.

The final run uses common settings:

| Parameter | Value |
| --- | --- |
| `FuncNonlinear` | 1 |
| `MIPGap` | 1e-6 |
| `NumericFocus` | 0 |
| `IntegralityFocus` | 0 |
| `IntFeasTol` | 1e-7 |
| `FuncPieceError` | 1e-4; relevant to static PWL mode |
| `DualReductions` | 0 |

The module default remains `1e-4` unless `STOCHFAIR_MIPGAP` is set before
import. The master notebook sets it to `1e-6`. `solver_config_snapshot`
records actual model settings in the summary JSON.

These are numerical optimization results within solver tolerances, not
symbolic certificates. Earlier session explanations about the meaning of
gap-terminated `ObjBound` and cross-configuration bounds were corrected
after the tighter run. Do not reuse those explanations as a general Gurobi
theorem without checking the solver's documented semantics and rescoring
comparable solutions with identical objectives and constraints.

### Allocation sensitivity

Earlier configurations gave materially different node allocations while
objective values and total delivered supply remained close. The dossier
calls this degeneracy. The comparisons demonstrate sensitivity among
near-optimal solutions, not a rigorous proof of multiple exact optima
under the final tighter settings. Node-level CV rankings and Jensen maps
should be illustrative unless robustness is investigated.

### Water supply utilization

Raw supply is 1000 per network; scarcity 0.75 scales each to 750.
Gas and Power use all 750. Water delivers 743.2463 and strands 6.7537.
The only nonzero unused-supply entry is at `Water_node_11`.

That node has 11.035 supply but its usable outgoing pocket has only 4.281
demand. There is no arc joining supply nodes 11 and 13. Demand-to-supply
arcs exist in the input but have no flow variable in either stage:
first-stage `x_star` leaves supply nodes; second-stage `x` stays among
demand/transshipment nodes. This partition blocks escape routes from the
pocket. Node 13 escapes via a supply-to-supply arc to node 14, made usable
by the supply-balance fix. The shortfall is unchanged across tested runs.
See dossier section 15.9; do not dismiss the shortfall as solver error.

## Final run and outputs

Scarcity 0.75, 150 scenarios, seed 42, power utility, equal probabilities.
The summary file is:

`results/scarce_0.75_utility_type_power_num_scenarios_150_equal_probabilities_True_nat_log_linear_pieces_length_1_summary.json`

| Metric | Value |
| --- | ---: |
| EV, population mean | 121.02454096708414 |
| EV, sample mean | 120.9890906251476 |
| EEVS, population mean | 121.25441580028877 |
| EEVS, sample mean | 121.24102965302687 |
| WS | 121.66305812081414 |
| DE | 121.34945128384796 |
| EVPI | 0.3136068369661871 |
| VSS, population mean | 0.09503548355918667 |
| VSS, sample mean | 0.10842163082108414 |

The saved DE log reports objective and bound equal to displayed precision,
gap 0.0000%, about 83.3 seconds. The full baseline pipeline recorded 453
solves, zero review flags, and about 1015.6 seconds total.

Jensen analysis reconciles its Formulation A score with DE:
`OF_A_DE_check = 121.349451`, `OF_B_expost = 126.471260`,
system Jensen gap approximately 5.121808.
This is evaluation on A's allocation, not B's optimized objective.

All 776 current result files exactly match the file contents in `results.zip`,
verified while preparing this handoff. The archive also has directory entries.
There are 753 solver-log files and 11 figures. Both the ZIP and loose final
results are included by explicit user request. Treat historical result sets
under `old_files/` as historical, not the run of record.

The DE JSON exports direction variables `z`/`z_star` absent from the CSV;
the CSV includes input/derived rows such as `alpha`, demand, and supply.
Scenario columns are digit-named (`0` through `149`); appended summary
statistics are not additional scenarios. Filter columns accordingly.

Warm starts produced exactly identical objectives across 301 benchmarked
solves and were slower at the final tolerance. There is no warm-started DE
in `benchmark_warm_start.py`. The README's earlier speedup statement is stale.

## Reading data and capacities

`src/tdINDP_parameters_build.py` loads network CSVs from
`src/csv_files_folder/`. Gas, Power, and Water are active; Telecommunication
loading is commented out. Parameters are loaded at module import as well as
through the public loading function.

`find_capacities()` uses an internal `build = False` flag:

- False reads `src/pareto_of_capacities.csv`, drops its leading index column,
  and selects zero-based row 18.
- True invokes the original Gurobi capacity-design sweep and writes that CSV.

The shipped frontier has 19 rows and 360 distinct capacity columns plus
`sum_u` and `sum_log_u`. Row 18 is currently the last row. Tuple-string
headers parse into `(tail, head, network, t)` keys with `t = 0`.
All 360 selected capacities differ from nominal network arc CSV capacities.

A separate sibling folder, `tdindp-capacities`, was created with a capacity
reader, partial solver-free network-parameter rewrite, data, README, and
`example.py`. It was meant for manual publication elsewhere. It is not a
full replacement for every original parameter or the capacity optimizer.
Selected capacities and the loaded core network dictionaries were compared
against the original and matched in that checkout.

Capacity-story caveats from that extraction:

- Stored frontier objective columns do not fully reconcile: row 0's total
  differs by roughly 11.97; rows 1-18 have an approximately 13.8155 log offset.
  Recompute aggregate values from capacity columns when reporting.
- The original code traverses arcs containing duplicates, while capacity
  dictionaries have unique keys. Do not assume all arc-list counts equal
  distinct-key counts or that bidirectionality exactly doubles every count.
- The earlier sibling README called row 18 a "knee" and inferred why the
  sweep stopped. Those interpretations were not established by the extraction.
  Read the generating code and validate objective semantics before citing them.

## Running in another environment

The last recorded environment used Python 3.8 and gurobipy 11.0.3.
No version upgrade was performed. A valid Gurobi installation/license is
needed for optimization, but not for reading saved results.

Use `notebooks/run_all_pipelines.ipynb` for the full workflow:
experiments, Jensen analysis, plotting via the plotter notebook, and the
warm-start benchmark. Inspect its cells before executing; running them
updates existing result files and removes/recreates per-solve logs.
Preserve the final snapshot before any new run.

For the core experiment driver from the repository root in PowerShell:

```powershell
$env:STOCHFAIR_MIPGAP = '1e-6'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -c "import run_experiments; run_experiments.run_all(beep_when_done=False)"
```

That command runs the core experiments, not the later plotting/benchmark
stages. Set the environment variable before importing any model helpers;
restart a notebook kernel if an old value was already imported.

Portability is not established: the driver imports Windows-only `winsound`.
Disabling the completion beep alone does not remove that import. On another
OS, inspect and adapt that dependency before attempting a run. No lockfile
or dependency manifest exists in this repository; inspect imports rather
than assuming the separate capacity reader's pandas-only requirements apply.

Generated results remain ignored by `.gitignore` for new paths; final output
files were explicitly force-added as a snapshot. Existing tracked outputs
can therefore become modified on reruns. Stage them deliberately.

## Outstanding work, in priority order

1. Align the separately maintained manuscript with the final 1e-6 numbers
   and regenerated figures. There is no manuscript source in this checkout;
   incorporation by the user's other agent has not been verified.
2. Decide whether node-level plots are illustrative or need allocation
   robustness/tie-breaking analysis. Do not claim uniqueness from current work.
3. Retain the Water supply explanation and distinguish ex-post Jensen
   evaluation from full Formulation B optimization.
4. Refresh the main README's warm-start statement and explain the master
   notebook's final gap setting. Dossier section 15.11 supersedes older runs.
5. Fix the reversed `gamma` comment. Optional cleanup: vestigial `x_start`,
   EEVS partial MIP starts, and earlier unaudited fix/re-gate paths.
6. Formulation B optimization and an isolated Python/Gurobi upgrade remain
   optional research extensions, not prerequisites for the final A results.

The session task database was updated to mark the final rerun and Jensen
regeneration done. Its remaining pending item is documentation cleanup;
the blocked Formulation B execution item means deferred research, not a
broken main pipeline. That database is not portable; this list replaces it.

## Preservation and validation

Earlier results were overwritten during debugging. Backups were placed
outside the original repo in a local Copilot session folder and several
runs were restored or cleared at the user's direction. Those external
backups are not included here and must not be assumed available elsewhere.
The committed final results are the portable source of truth.

PR-preparation checks: all 14 active Python modules parsed; all four active
notebooks had valid JSON and nbformat 4; ZIP CRC checks passed; final required
Jensen artifacts were present; every current result file matched its archived
counterpart. These checks are not a new model solve or a full mathematical audit.

Do not launch expensive solves, clear outputs, amend commits, or merge the PR
without explicit authorization. Keep changes surgical and report uncertainty
instead of repeating earlier speculative explanations as settled findings.
